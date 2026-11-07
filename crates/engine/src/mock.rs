//! A deterministic CPU "model" for exercising the engine without a GPU. Its logits depend only on the
//! context, so it satisfies the row-invariance contract by construction. `Fault` makes it violate the
//! contract on purpose, to prove the harness catches a window-dependent kernel.
//!
//! `with_deep_state` adds what a hybrid model has and the plain mock lacks: a recurrent state, folded over
//! every token ever fed, that steers the logits. Rolling back after a rejected draft must restore it exactly;
//! `Fault::RollbackLeak` keeps the rejected rows' state instead.
//!
//! `with_prefix_cache` gives it what the prefix cache needs and the plain mock lacks, so the engine's cache logic can be tested without a
//! GPU: a paged KV (256-token pages with reference counts; the logits are computed from the tokens READ from the pages, so a shared page that
//! someone clobbered, or a page freed while still referenced, changes the replies), snapshot slots, the draft's context ring
//! (`with_native_draft`: drafts are computed from what the ring holds, so a window that was not restored changes the speculation counters),
//! and a recurrent state that depends on the prefill path (`with_path_dependent_state`, as the Qwen backend's does). `PrefixFault` breaks one
//! invariant at a time.
//!
//! `with_glm_shape` is the GLM-5.3-Flash backend's shape of the same contract (`docs/PREFIX_CACHE.md`, GLM): ONE arithmetic (no path rule: a prompt
//! of any length may be cached), and what a GLM sequence keeps outside its recurrent state and its pages: a per-slot ring of raw rows, whose last
//! `len % 4` rows complete the first pool group after a prefix is adopted at a length that is not a multiple of 4; a pooled-key plane that lives IN
//! the pages (so the copy of a partial page must carry it) and steers the logits; and a draft carry (the last committed position's tap) that steers
//! the drafts. Restoring any of them wrongly changes a reply or a speculation counter.

use crate::{Backend, BackendError, Captured, Drafter, PrefixCaps, PrefixHandle, RowOutput, RowRequest};
use gb_sampling::{top_candidates, Sampling};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    None,
    /// Rows after the first in a window get a shifted logit: the classic "K split depends on row count" bug.
    WindowDependent,
    /// Prefill in chunks differs from one-shot prefill in the last bit.
    ChunkDependent,
    /// `commit(keep)` leaves the recurrent state at the end of the whole window instead of rolling it back to
    /// `keep` rows. Only visible with `with_deep_state`.
    RollbackLeak,
}

/// Failures the mock produces on purpose, to exercise the engine's and the server's error paths. Unlike [`Fault`] (which breaks the
/// exactness contract silently), these are loud: they are reported the way the CUDA backend reports its own (`Backend::take_error`
/// plus NaN rows), or they panic. Steps and prefill slices are counted from 1.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Faults {
    /// Panic inside this `step`.
    pub panic_at_step: Option<u64>,
    /// Record a fatal error at this `step` (and keep reporting it, as a sticky device error does).
    pub fatal_at_step: Option<u64>,
    /// Record a recoverable error at `retryable_steps` consecutive steps from this one.
    pub retryable_from_step: Option<u64>,
    pub retryable_steps: u64,
    /// Panic inside this `prefill_slice`.
    pub panic_at_prefill: Option<u64>,
    /// Return NaN rows from every `step` for a sequence whose prompt starts with this token: one poisoned sequence.
    pub poison_first_token: Option<u32>,
}

/// Options of the mock for the server's `--mock` flag (the CPU backend is for protocol tests, so these are test controls).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MockOptions {
    /// Sleep this long in every `step` (the mock is otherwise instantaneous, so a test could not observe a running request).
    pub step_ms: u64,
    pub boot_ms: u64,
    /// Sleep this long in every prefill slice.
    pub prefill_ms: u64,
    /// Prefill in slices of this many tokens (0: the whole prompt at once).
    pub slice: usize,
    /// Vocabulary size (0: the tokenizer's).
    pub vocab: usize,
    pub faults: Faults,
    /// Panic while the server builds the engine (the server checks this; the backend itself is never built).
    pub panic_boot: bool,
    /// A prefix cache of this many entries (0: none): paged KV, snapshot slots, a path-dependent state and a native draft, as
    /// `MockBackend::with_prefix_cache` gives them.
    pub prefix_cache: usize,
    /// Report CUDA-graph cache counters (`Backend::graph_counters`) as a backend with graphs does: every step a replay, every prefill slice
    /// an eager window. Off: the mock reports none, as before.
    pub graphs: bool,
    /// State a prefill floor (`Backend::prefill_floor`) of this many microseconds a token (0: none, as before).
    pub prefill_floor_us: u64,
}

impl MockOptions {
    /// `key=value,key=value`: `step_ms`, `prefill_ms`, `slice`, `vocab`, `panic_step`, `fatal_step`, `retry_step` (with
    /// `retry_count`, default 1), `panic_prefill`, `poison` (a token id), `panic_boot` (1: panic while building the engine),
    /// `prefix_cache` (entries of a prefix cache), `graphs` (1: report graph-cache counters), `prefill_floor_us` (a prefill floor a token).
    pub fn parse(spec: &str) -> Result<Self, String> {
        let mut o = Self::default();
        for kv in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').ok_or_else(|| format!("--mock: `{kv}` is not key=value"))?;
            let n: u64 = v.parse().map_err(|_| format!("--mock: `{v}` is not a number (in `{kv}`)"))?;
            match k {
                "boot_ms" => o.boot_ms = n,
                "step_ms" => o.step_ms = n,
                "prefill_ms" => o.prefill_ms = n,
                "slice" => o.slice = n as usize,
                "vocab" => o.vocab = n as usize,
                "panic_step" => o.faults.panic_at_step = Some(n),
                "fatal_step" => o.faults.fatal_at_step = Some(n),
                "retry_step" => {
                    o.faults.retryable_from_step = Some(n);
                    o.faults.retryable_steps = o.faults.retryable_steps.max(1);
                }
                "retry_count" => o.faults.retryable_steps = n,
                "panic_prefill" => o.faults.panic_at_prefill = Some(n),
                "poison" => o.faults.poison_first_token = Some(n as u32),
                "panic_boot" => o.panic_boot = n != 0,
                "prefix_cache" => o.prefix_cache = n as usize,
                "graphs" => o.graphs = n != 0,
                "prefill_floor_us" => o.prefill_floor_us = n,
                _ => return Err(format!("--mock: unknown option `{k}`")),
            }
        }
        Ok(o)
    }

    /// A backend with these options; `default_vocab` is used when `vocab` is 0.
    pub fn build(&self, default_vocab: usize, slots: usize) -> MockBackend {
        let vocab = if self.vocab == 0 { default_vocab } else { self.vocab };
        let mut b = MockBackend::new(vocab, slots, Fault::None)
            .with_latency(Duration::from_millis(self.step_ms), Duration::from_millis(self.prefill_ms))
            .with_faults(self.faults.clone());
        if self.slice > 0 {
            b = b.with_prefill_slice(self.slice);
        }
        if self.graphs {
            b.graphs = true;
        }
        if self.prefill_floor_us > 0 {
            b = b.with_prefill_floor(Duration::from_micros(self.prefill_floor_us));
        }
        if self.prefix_cache > 0 {
            let units = self.prefix_cache;
            let budget = gb_sched::prefix::CacheBudget {
                snapshot_slots: units,
                page_blocks: units * 8,
                snapshot_bytes: 1 << 20,
                page_bytes: 1 << 16,
            };
            b = b.with_path_dependent_state().with_native_draft().with_prefix_cache(budget, PrefixFault::None);
        }
        b
    }
}

fn h(a: u64, b: u64) -> u64 {
    let mut x = a.wrapping_mul(0x9E3779B97F4A7C15) ^ b.wrapping_add(0xD1B54A32D192ED03);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^ (x >> 29)
}

/// Logits for the position after `ctx`: two strong tokens chosen from the last two tokens plus
/// hashed background, so decoding is structured enough for drafts to be accepted sometimes.
pub fn mock_logits(ctx: &[u32], vocab: usize) -> Vec<f32> {
    let a = *ctx.last().unwrap_or(&0) as u64;
    let b = if ctx.len() >= 2 { ctx[ctx.len() - 2] as u64 } else { 0 };
    let v = vocab as u64;
    // copy behaviour, as real LLMs have: continue the latest earlier occurrence of the last bigram
    let n = ctx.len();
    let copied = (n >= 3)
        .then(|| (0..n - 2).rev().find(|&i| ctx[i] as u64 == b && ctx[i + 1] as u64 == a).map(|i| ctx[i + 2] as usize))
        .flatten();
    let peak = copied.unwrap_or(((a * 31 + b * 7 + 3) % v) as usize);
    let second = ((a * 17 + 5) % v) as usize;
    let key = h(a, b);
    let mut out: Vec<f32> = (0..vocab as u64).map(|i| ((h(key, i) >> 40) as f32 / 16777216.0 * 4.0) - 2.0).collect();
    out[peak] = 6.0;
    out[second] = out[second].max(4.5);
    out
}

/// The recurrent state after feeding one more token.
fn fold_state(state: u64, token: u32) -> u64 {
    fold_state_on(state, token, 0)
}

/// [`fold_state`] on an arithmetic path: `salt` 0 is the plain fold, another salt makes the state the path's own.
fn fold_state_on(state: u64, token: u32, salt: u64) -> u64 {
    h(state ^ 0x5851F42D4C957F2D, (token as u64 + 1).wrapping_add(salt))
}

/// Tokens per page of the mock's KV (the Qwen backend's page size).
pub const MOCK_PAGE: usize = 256;
/// The draft ring's capacity and the positions its attention can look back over (the real draft's `DraftCtxLayout`).
pub const MOCK_RING: usize = 2560;
pub const MOCK_WINDOW: usize = 2048;
/// Positions behind the end of a prompt the draft's context is kept for (`draft::CTX_KEEP`).
pub const MOCK_CTX_KEEP: usize = MOCK_WINDOW + 16;
/// Longest prompt on the GEMV path (`PrefillPath::GEMV_MAX_PROMPT`).
pub const MOCK_GEMV_MAX: usize = 128;

/// One invariant of the prefix cache the mock can break on purpose: the engine-level test for it must fail under it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PrefixFault {
    #[default]
    None,
    /// A restore shares the partial last page with the entry instead of copying it: the first write of the suffix clobbers the donor's page.
    NoCopyOnWrite,
    /// A restore leaves the draft's context ring as the slot's previous owner had it.
    DraftNotRestored,
    /// A restore leaves the recurrent state as it was (zero) instead of the snapshot's.
    StateNotRestored,
    /// Every prompt is cacheable on one path, the GEMV-path ones included (the path rule ignored).
    PathIgnored,
    /// Releasing a handle does not drop its page references: pages leak.
    RefcountLeak,
    /// Releasing a handle frees its pages whatever else still references them.
    FreeWhileReferenced,
    /// (GLM shape) A restore leaves the raw-row ring as the slot's previous owner had it: the first pool completed after a prefix of a length that is
    /// not a multiple of 4 is built from stale rows.
    RingTailNotRestored,
    /// (GLM shape) A restore leaves the draft's carry (the last position's tap) as the previous owner had it.
    CarryNotRestored,
    /// (GLM shape) The copy of the partial last page leaves the pooled-key plane behind (the tokens are copied, the pools are not).
    PoolsNotCopied,
}

/// Pools per page in the GLM shape (one per 4 tokens).
const MOCK_POOLS_PER_PAGE: usize = MOCK_PAGE / 4;
/// Raw rows the GLM shape's per-slot ring keeps.
const MOCK_RAW_RING: usize = 256;

struct MockKv {
    pages: Vec<Vec<u32>>,
    /// The pooled-key plane, per page (the GLM shape; never cleared, like device memory a page's previous holder wrote).
    pools: Vec<Vec<u64>>,
    /// The NextN drafter's cache entry of each position of a page (the GLM shape): a function of the position's token and the tap of the position before it.
    mtp: Vec<Vec<u64>>,
    refs: Vec<u32>,
    free: Vec<u32>,
    tables: Vec<Vec<u32>>,
}

impl MockKv {
    fn new(pages: usize, slots: usize) -> Self {
        Self {
            pages: vec![vec![0; MOCK_PAGE]; pages],
            pools: vec![vec![0; MOCK_POOLS_PER_PAGE]; pages],
            mtp: vec![vec![0; MOCK_PAGE]; pages],
            refs: vec![0; pages],
            free: (0..pages as u32).rev().collect(),
            tables: vec![Vec::new(); slots],
        }
    }
    fn alloc(&mut self) -> u32 {
        let p = self.free.pop().expect("the mock's page pool is exhausted (a test sized it too small)");
        self.refs[p as usize] = 1;
        p
    }
    fn retain(&mut self, p: u32) {
        assert!(self.refs[p as usize] > 0, "retain of a free page {p}");
        self.refs[p as usize] += 1;
    }
    fn release(&mut self, p: u32) {
        let r = &mut self.refs[p as usize];
        assert!(*r > 0, "double release of page {p}");
        *r -= 1;
        if *r == 0 {
            self.free.push(p);
        }
    }
    fn ensure(&mut self, slot: usize, tokens: usize) {
        while self.tables[slot].len() * MOCK_PAGE < tokens {
            let p = self.alloc();
            self.tables[slot].push(p);
        }
    }
    fn write(&mut self, slot: usize, pos: usize, token: u32) {
        self.ensure(slot, pos + 1);
        let p = self.tables[slot][pos / MOCK_PAGE] as usize;
        self.pages[p][pos % MOCK_PAGE] = token.wrapping_add(1);
    }
    fn read(&self, slot: usize, n: usize) -> Vec<u32> {
        (0..n)
            .map(|pos| match self.tables[slot].get(pos / MOCK_PAGE) {
                Some(&p) => self.pages[p as usize][pos % MOCK_PAGE].wrapping_sub(1),
                None => 0,
            })
            .collect()
    }
    fn release_slot(&mut self, slot: usize) {
        for p in std::mem::take(&mut self.tables[slot]) {
            self.release(p);
        }
    }
    fn in_use(&self) -> usize {
        self.refs.iter().filter(|&&r| r > 0).count()
    }
}

struct MockSnap {
    state: u64,
    /// (GLM shape) the raw rows of the incomplete pool group, positions `len - len % 4 .. len`.
    raw_tail: Vec<u64>,
    /// (GLM shape) the draft's carry after `len` tokens.
    carry: u64,
    /// The draft ring's entries for positions `[len - 2048, len)`, in position order: `(position written, token)`.
    ring: Vec<(i64, u32)>,
    lo: usize,
}

struct MockCache {
    caps: PrefixCaps,
    snaps: Vec<Option<MockSnap>>,
    fault: PrefixFault,
}

pub struct MockBackend {
    pub vocab: usize,
    pub fault: Fault,
    deep: bool,
    ctx: Vec<Vec<u32>>,
    pending: Vec<Vec<u32>>,
    prefill_calls: Vec<usize>,
    /// Recurrent state after the committed context, per slot (deep mode).
    state: Vec<u64>,
    /// Recurrent state after each row of the slot's last step (deep mode).
    pending_state: Vec<Vec<u64>>,
    /// Parent row of each row of the slot's last step (a chain: `i - 1`).
    pending_parents: Vec<Vec<i32>>,
    /// Verify trees (`Backend::step_tree`): every row's logits and state come from its own root path.
    trees: bool,
    slice: Option<usize>,
    deadline_rows: Option<usize>,
    /// `set_prefill_row_limit` answers false (a backend that cannot limit its prefill rows): the engine's control case.
    no_row_limit: bool,
    faults: Faults,
    step_delay: Duration,
    prefill_delay: Duration,
    /// `step` and `prefill_slice` calls so far (what `Faults` counts).
    step_calls: u64,
    slice_calls: u64,
    /// The error `take_error` hands out next, and whether a fatal one has been seen (it sticks).
    error: Option<BackendError>,
    latched: Option<BackendError>,
    kv: Option<MockKv>,
    cache: Option<MockCache>,
    /// The draft's context ring per slot (`position written, token`; position -1: never written), when drafts are native.
    ring: Option<Vec<Vec<(i64, u32)>>>,
    /// The recurrent state depends on the arithmetic path that fed a token (prefill above 128 tokens, prefill up to 128, decode).
    path_state: bool,
    /// The GLM shape (see the module documentation): the raw-row ring per slot and the draft carry per slot.
    glm: Option<GlmMock>,
    /// Report graph-cache counters (`MockOptions::graphs`).
    graphs: bool,
    /// The prefill floor a token (`with_prefill_floor`).
    floor_per_token: Option<Duration>,
    /// The next score request wants every row (`Backend::score_all_rows`), and the rows of the one that finished
    /// (`Backend::take_score_rows`): row `i` is the logits after the scored context's tokens `0..=i`.
    score_rows_on: bool,
    score_rows: Vec<Vec<f32>>,
    /// The prefix cache keeps idle pool pages, takes cuts in a batched call and at a prompt's end, as the GLM group does
    /// (`with_pool_cache`).
    pool_cache: bool,
    /// Jobs of `prefill_slices` calls of two jobs or more that had cuts to take or resumed a restored slot (`batched_cache_jobs`).
    batched_cache_jobs: usize,
}

/// What the GLM shape keeps per slot outside the pages.
struct GlmMock {
    raw: Vec<Vec<u64>>,
    carry: Vec<u64>,
}

impl MockBackend {
    pub fn new(vocab: usize, slots: usize, fault: Fault) -> Self {
        Self {
            vocab,
            fault,
            deep: false,
            ctx: vec![Vec::new(); slots],
            pending: vec![Vec::new(); slots],
            prefill_calls: vec![0; slots],
            state: vec![0; slots],
            pending_state: vec![Vec::new(); slots],
            pending_parents: vec![Vec::new(); slots],
            trees: false,
            slice: None,
            deadline_rows: None,
            no_row_limit: false,
            faults: Faults::default(),
            step_delay: Duration::ZERO,
            prefill_delay: Duration::ZERO,
            step_calls: 0,
            slice_calls: 0,
            error: None,
            latched: None,
            kv: None,
            cache: None,
            ring: None,
            path_state: false,
            glm: None,
            graphs: false,
            floor_per_token: None,
            score_rows_on: false,
            score_rows: Vec::new(),
            pool_cache: false,
            batched_cache_jobs: 0,
        }
    }
    /// State a lower bound on prefill time (`Backend::prefill_floor`) of `per_token` a prompt token, as a family with measured rates does.
    pub fn with_prefill_floor(mut self, per_token: Duration) -> Self {
        self.floor_per_token = Some(per_token);
        self
    }
    /// A paged KV the logits are read from (see the module documentation), with room for `pages` pages in all.
    pub fn with_paged_kv(mut self, pages: usize) -> Self {
        let slots = self.ctx.len();
        self.kv = Some(MockKv::new(pages, slots));
        self
    }
    /// A prefix cache of `budget`: paged KV (sized for every slot to hold 64 pages, plus the cache's pages), snapshot slots, and the fault to
    /// inject (`PrefixFault::None` for the honest backend).
    pub fn with_prefix_cache(mut self, budget: gb_sched::prefix::CacheBudget, fault: PrefixFault) -> Self {
        let slots = self.ctx.len();
        self.kv = Some(MockKv::new(slots * 64 + budget.page_blocks, slots));
        self.cache = Some(MockCache {
            caps: PrefixCaps { namespace: 0x6D6F_636B, budget },
            snaps: (0..budget.snapshot_slots).map(|_| None).collect(),
            fault,
        });
        self.deep = true;
        self
    }
    /// A prefix cache of `snapshots` snapshot slots whose entries keep idle pages of a KV pool of `pages` pages (the index may hold all of
    /// them), as a GLM group's `--prefix-cache on` does: `Backend::kv_pool` reports the pool, `prefill_slices` takes each job's cuts and
    /// resumes restored slots, and a cut may be the prompt's end.
    pub fn with_pool_cache(mut self, snapshots: usize, pages: usize, fault: PrefixFault) -> Self {
        let slots = self.ctx.len();
        let budget = gb_sched::prefix::CacheBudget { snapshot_slots: snapshots, page_blocks: pages, snapshot_bytes: 141 << 20, page_bytes: 3 << 20 };
        self.kv = Some(MockKv::new(pages, slots));
        self.cache = Some(MockCache {
            caps: PrefixCaps { namespace: 0x6D6F_636B, budget },
            snaps: (0..snapshots).map(|_| None).collect(),
            fault,
        });
        self.deep = true;
        self.pool_cache = true;
        self
    }
    /// Drafts come from the draft context ring (kept like the real draft's: prompt rows from `total - 2064`, accepted rows as they commit).
    pub fn with_native_draft(mut self) -> Self {
        self.ring = Some(vec![vec![(-1, 0); MOCK_RING]; self.ctx.len()]);
        self
    }
    /// The recurrent state a token leaves depends on the path that fed it, as the Qwen backend's does (the decode GEMV for a prompt of up to
    /// 128 tokens, the pinned GEMM above, the window kernels for generated tokens). Implies a deep state.
    pub fn with_path_dependent_state(mut self) -> Self {
        self.path_state = true;
        self.deep = true;
        self
    }
    /// The GLM shape (see the module documentation): one arithmetic for every prompt length, a raw-row ring, a pooled-key plane in the pages and a
    /// draft carry. Chains only (no verify trees). Implies a deep state and no path dependence.
    pub fn with_glm_shape(mut self) -> Self {
        let slots = self.ctx.len();
        self.glm = Some(GlmMock { raw: vec![vec![0xDEAD_BEEF; MOCK_RAW_RING]; slots], carry: vec![0; slots] });
        self.path_state = false;
        self.deep = true;
        self
    }
    /// Jobs that reached `prefill_slices` beside another job with cuts to take or a restored slot to resume.
    pub fn batched_cache_jobs(&self) -> usize {
        self.batched_cache_jobs
    }
    /// Pages with at least one holder (a slot or an entry).
    pub fn pages_in_use(&self) -> usize {
        self.kv.as_ref().map_or(0, |k| k.in_use())
    }
    /// Snapshot slots holding a snapshot.
    pub fn snapshots_in_use(&self) -> usize {
        self.cache.as_ref().map_or(0, |c| c.snaps.iter().filter(|s| s.is_some()).count())
    }
    /// Pages the slot's table holds, for a test that checks sharing.
    pub fn slot_pages(&self, slot: usize) -> Vec<u32> {
        self.kv.as_ref().map(|k| k.tables[slot].clone()).unwrap_or_default()
    }
    /// How many holders a page has.
    pub fn page_refs(&self, page: u32) -> u32 {
        self.kv.as_ref().map_or(0, |k| k.refs[page as usize])
    }
    fn salt_prefill(&self, total: usize) -> u64 {
        if !self.path_state {
            0
        } else if total > MOCK_GEMV_MAX {
            0x1111_0000_0000
        } else {
            0x2222_0000_0000
        }
    }
    fn salt_decode(&self) -> u64 {
        if self.path_state {
            0x3333_0000_0000
        } else {
            0
        }
    }
    /// (GLM shape) The tap of a position: a function of the whole prefix, as a hidden state is.
    fn tap(state: u64) -> u64 {
        h(state, 0xCA77_0000_0001)
    }
    /// (GLM shape) Append position `pos`'s raw row (after the state `st`) to the slot's ring and complete the pool this position completes: its four
    /// members come from the ring, the pool lands in the page the slot's table holds for it.
    /// The drafter's cache entry of the position is built from its token and `prev_tap` (the tap of the position before it: the carry, for the first row
    /// of a chunk that continues a prompt).
    fn glm_append(&mut self, slot: usize, pos: usize, st: u64, token: u32, prev_tap: u64) {
        let Some(g) = self.glm.as_mut() else { return };
        g.raw[slot][pos % MOCK_RAW_RING] = h(st ^ 0x7A77_0000_0000, pos as u64);
        {
            let kv = self.kv.as_mut().expect("the GLM shape needs the paged KV");
            let page = kv.tables[slot][pos / MOCK_PAGE] as usize;
            kv.mtp[page][pos % MOCK_PAGE] = h(prev_tap, token as u64 + 1);
        }
        if (pos + 1) % 4 == 0 {
            let p = (pos + 1) / 4 - 1;
            let v = (0..4).rev().fold(0x9001u64, |v, back| h(v, g.raw[slot][(pos - back) % MOCK_RAW_RING]));
            let kv = self.kv.as_mut().expect("the GLM shape needs the paged KV");
            let page = kv.tables[slot][p / MOCK_POOLS_PER_PAGE] as usize;
            kv.pools[page][p % MOCK_POOLS_PER_PAGE] = v;
        }
    }
    /// (GLM shape) A digest of the drafter's cache entries of positions `0..n`, read through the slot's pages: 0 for another shape.
    fn glm_mtp_digest(&self, slot: usize, n: usize) -> u64 {
        if self.glm.is_none() {
            return 0;
        }
        let kv = self.kv.as_ref().expect("the GLM shape needs the paged KV");
        (0..n).fold(0x77DEAD, |d, pos| h(d, kv.mtp[kv.tables[slot][pos / MOCK_PAGE] as usize][pos % MOCK_PAGE]))
    }
    /// (GLM shape) A digest of the pools a row at position `pos` sees (the complete ones), read through the slot's pages: 0 for another shape.
    fn glm_digest(&self, slot: usize, pos: usize) -> u64 {
        if self.glm.is_none() {
            return 0;
        }
        let kv = self.kv.as_ref().expect("the GLM shape needs the paged KV");
        (0..(pos + 1) / 4).fold(0x1D1E57u64, |d, p| {
            let page = kv.tables[slot][p / MOCK_POOLS_PER_PAGE] as usize;
            h(d, kv.pools[page][p % MOCK_POOLS_PER_PAGE])
        })
    }
    /// Feed `tokens` (positions `start..`) of a prompt of `total` tokens into `slot`: KV, recurrent state, the draft's ring, and a snapshot at
    /// every cut the feed passes. Returns the candidates after the last token.
    #[allow(clippy::too_many_arguments)]
    fn feed(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        total: usize,
        cuts: &[usize],
        captured: &mut Vec<Captured>,
        candidates: usize,
    ) -> RowOutput {
        assert_eq!(self.ctx[slot].len(), start, "prefill must continue the slot's context");
        let salt = self.salt_prefill(total);
        // the draft's context starts at the end of the prompt minus its keep, or earlier when a snapshot needs the window before its cut
        let ctx_from = cuts
            .iter()
            .filter(|&&c| c > start)
            .map(|&c| c.saturating_sub(MOCK_WINDOW))
            .chain(std::iter::once(total.saturating_sub(MOCK_CTX_KEEP)))
            .min()
            .unwrap();
        // (GLM shape) the first row of a chunk that continues a prompt reads the carry: the tap of the position before it
        let mut prev_tap = match &self.glm {
            Some(g) if start > 0 => g.carry[slot],
            _ => 0,
        };
        for (i, &t) in tokens.iter().enumerate() {
            let pos = start + i;
            self.ctx[slot].push(t);
            if let Some(kv) = self.kv.as_mut() {
                kv.write(slot, pos, t);
            }
            self.state[slot] = fold_state_on(self.state[slot], t, salt);
            self.glm_append(slot, pos, self.state[slot], t, prev_tap);
            prev_tap = Self::tap(self.state[slot]);
            if let Some(ring) = self.ring.as_mut() {
                if pos >= ctx_from {
                    ring[slot][pos % MOCK_RING] = (pos as i64, t);
                }
            }
            if self.cache.is_some() && cuts.contains(&(pos + 1)) {
                if let Some(c) = self.capture(slot, pos + 1) {
                    captured.push(c);
                }
            }
        }
        self.prefill_calls[slot] += 1;
        let n = start + tokens.len();
        let ctx = match &self.kv {
            Some(kv) => kv.read(slot, n),
            None => self.ctx[slot].clone(),
        };
        if let Some(g) = self.glm.as_mut() {
            g.carry[slot] = Self::tap(self.state[slot]);
        }
        let st = self.state[slot] ^ self.glm_digest(slot, n - 1);
        self.row(&ctx, st, candidates)
    }
    /// A snapshot of `slot`'s state after `len` tokens: the recurrent state, the draft ring's window, references to the pages.
    fn capture(&mut self, slot: usize, len: usize) -> Option<Captured> {
        let cache = self.cache.as_mut()?;
        let snap = cache.snaps.iter().position(|s| s.is_none())?;
        let kv = self.kv.as_mut()?;
        let pages: Vec<u32> = kv.tables[slot][..len.div_ceil(MOCK_PAGE)].to_vec();
        for &p in &pages {
            kv.retain(p);
        }
        let lo = len.saturating_sub(MOCK_WINDOW);
        let ring = match &self.ring {
            Some(r) => (lo..len).map(|pos| r[slot][pos % MOCK_RING]).collect(),
            None => Vec::new(),
        };
        let tail = len % 4;
        let raw_tail = match &self.glm {
            Some(g) => (0..tail).map(|i| g.raw[slot][(len - tail + i) % MOCK_RAW_RING]).collect(),
            None => Vec::new(),
        };
        cache.snaps[snap] = Some(MockSnap { state: self.state[slot], raw_tail, carry: Self::tap(self.state[slot]), ring, lo });
        Some(Captured { len, handle: PrefixHandle { snap: snap as u32, pages } })
    }
    fn clear_slot(&mut self, slot: usize) {
        self.ctx[slot].clear();
        self.pending[slot].clear();
        self.pending_state[slot].clear();
        self.pending_parents[slot].clear();
        self.state[slot] = 0;
        self.prefill_calls[slot] = 0;
        if let Some(kv) = self.kv.as_mut() {
            kv.release_slot(slot);
        }
    }
    /// Verify trees (`Backend::step_tree`): every row's logits come from its own root path, as serial decoding of that path gives them.
    pub fn with_trees(mut self) -> Self {
        self.trees = true;
        self
    }
    /// One window of every request: `(slot, tokens, parents, first position)`, parents as row indices (a chain is `i - 1`). Row `j`'s context is
    /// the committed context plus the tokens on its root path, its recurrent state the state folded along that path. A chain's rows are written
    /// to the paged KV at their positions (a rejected row's stays, beyond the committed length); a tree's never are (siblings share positions):
    /// `commit_path` writes the accepted path's.
    fn run_step(&mut self, reqs: &[(usize, Vec<u32>, Vec<i32>, u64)], candidates: usize) -> Vec<Vec<RowOutput>> {
        self.step_calls += 1;
        let n = self.step_calls;
        if self.faults.panic_at_step == Some(n) {
            panic!("injected panic at step {n}");
        }
        if !self.step_delay.is_zero() {
            std::thread::sleep(self.step_delay);
        }
        if self.faults.fatal_at_step.is_some_and(|f| n >= f) && self.latched.is_none() {
            self.latched = Some(BackendError::fatal(format!("injected fatal error at step {n}")));
        }
        if let Some(from) = self.faults.retryable_from_step {
            if n >= from && n < from + self.faults.retryable_steps {
                self.error = Some(BackendError::retryable(format!("injected recoverable error at step {n}")));
            }
        }
        // like the CUDA backend, a step that failed hands back NaN rows along with its error
        let failed = self.latched.is_some() || self.error.is_some();
        let poison = self.faults.poison_first_token;
        reqs.iter()
            .map(|(slot, tokens, parents, first)| {
                let slot = *slot;
                let poisoned = failed || poison.is_some_and(|t| self.ctx[slot].first() == Some(&t));
                assert_eq!(self.ctx[slot].len() as u64, *first, "step must continue the slot's context");
                self.pending[slot] = tokens.clone();
                self.pending_parents[slot] = parents.clone();
                let salt = self.salt_decode();
                let mut states: Vec<u64> = Vec::with_capacity(tokens.len());
                for (j, &t) in tokens.iter().enumerate() {
                    let from = if parents[j] < 0 { self.state[slot] } else { states[parents[j] as usize] };
                    states.push(fold_state_on(from, t, salt));
                }
                self.pending_state[slot] = states.clone();
                let chain = parents.iter().enumerate().all(|(i, &p)| p == i as i32 - 1);
                assert!(chain || self.glm.is_none(), "the GLM shape verifies chains only");
                let base = match self.kv.as_mut() {
                    Some(kv) => {
                        if chain {
                            for (j, &t) in tokens.iter().enumerate() {
                                kv.write(slot, *first as usize + j, t);
                            }
                        }
                        Some(kv.read(slot, *first as usize))
                    }
                    None => None,
                };
                if chain && self.glm.is_some() {
                    // the window appends every row's raw row and completes the pools they complete (a rejected row's are overwritten by the next window)
                    for j in 0..tokens.len() {
                        let prev = Self::tap(if j == 0 { self.state[slot] } else { states[j - 1] });
                        self.glm_append(slot, *first as usize + j, states[j], tokens[j], prev);
                    }
                }
                (0..tokens.len())
                    .map(|j| {
                        let mut path = vec![j];
                        while parents[*path.last().unwrap()] >= 0 {
                            path.push(parents[*path.last().unwrap()] as usize);
                        }
                        let mut c = base.clone().unwrap_or_else(|| self.ctx[slot].clone());
                        c.extend(path.iter().rev().map(|&r| tokens[r]));
                        let mut r = self.row(&c, states[j] ^ self.glm_digest(slot, *first as usize + j), candidates);
                        if self.fault == Fault::WindowDependent && j >= 1 {
                            // demote the top candidate below the second one
                            if r.values.len() > 1 {
                                r.values[0] = r.values[1] - 0.25;
                                let (ids, vals) = reorder(&r.ids, &r.values);
                                r = RowOutput { ids, values: vals };
                            }
                        }
                        if poisoned {
                            r.values.iter_mut().for_each(|v| *v = f32::NAN);
                        }
                        r
                    })
                    .collect()
            })
            .collect()
    }
    /// Loud failures on demand (see [`Faults`]).
    pub fn with_faults(mut self, faults: Faults) -> Self {
        self.faults = faults;
        self
    }
    /// Sleep in every `step` and every prefill slice, so a test can observe requests that are queued, prefilling or running.
    pub fn with_latency(mut self, step: Duration, prefill_slice: Duration) -> Self {
        self.step_delay = step;
        self.prefill_delay = prefill_slice;
        self
    }
    /// Prefill in slices of `n` tokens (the default prefills everything at once): lets tests interleave decoding with a prefill.
    pub fn with_prefill_slice(mut self, n: usize) -> Self {
        self.slice = Some(n.max(1));
        self
    }
    /// Refuse the engine's prefill row limit (`set_prefill_row_limit` returns false), as a backend that cannot slice row-invariantly.
    pub fn without_row_limit(mut self) -> Self {
        self.no_row_limit = true;
        self
    }
    /// Make the logits depend on a persisted recurrent state, not only on the token list.
    pub fn with_deep_state(mut self) -> Self {
        self.deep = true;
        self
    }
    fn row(&self, ctx: &[u32], state: u64, candidates: usize) -> RowOutput {
        let mut l = mock_logits(ctx, self.vocab);
        if self.deep && h(state, 99) % 4 == 0 {
            // a quarter of positions have their argmax decided by the recurrent state alone
            let k = (h(state, 7) % self.vocab as u64) as usize;
            l[k] = 7.0;
        }
        let (ids, values) = top_candidates(&l, candidates);
        RowOutput { ids, values }
    }
}

impl Backend for MockBackend {
    fn set_prefill_row_limit(&mut self, rows: Option<usize>) -> bool {
        if self.no_row_limit {
            return false;
        }
        self.deadline_rows = rows;
        true
    }
    fn cold_step_budget(&self) -> Option<Duration> {
        Some(self.step_delay + self.prefill_delay + Duration::from_millis(50))
    }
    fn prefill_floor(&self, from: usize, to: usize) -> Option<Duration> {
        self.floor_per_token.map(|t| t.saturating_mul(to.saturating_sub(from) as u32))
    }
    fn graph_counters(&self) -> Option<crate::GraphCounters> {
        self.graphs.then(|| crate::GraphCounters {
            replays: self.step_calls,
            eager_windows: self.slice_calls,
            ..Default::default()
        })
    }
    fn score_supported(&self) -> bool {
        self.glm.is_none()
    }
    fn score_all_rows(&mut self, on: bool) {
        self.score_rows_on = on;
        self.score_rows.clear();
    }
    fn take_score_rows(&mut self) -> Vec<Vec<f32>> {
        std::mem::take(&mut self.score_rows)
    }
    fn score_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        done: usize,
        cuts: &[usize],
    ) -> Result<crate::score::ScoreStep, BackendError> {
        if !self.score_supported() {
            return Err(crate::score::unsupported_backend());
        }
        let c = self.prefill_slice_cuts(slot, prompt, &[], done, 1, cuts);
        if let Some(e) = self.take_error() {
            for h in c.captured {
                self.prefix_release(h.handle);
            }
            return Err(e);
        }
        let row = if c.step.done == prompt.len() {
            // Read the CURRENT slot's fed context, including any restored prefix, not the caller's
            // unexecuted prompt or a previous candidate buffer.
            let ctx = self.kv.as_ref().map_or_else(|| self.ctx[slot].clone(), |kv| kv.read(slot, self.ctx[slot].len()));
            let mut logits = mock_logits(&ctx, self.vocab);
            if self.deep && h(self.state[slot], 99) % 4 == 0 {
                logits[(h(self.state[slot], 7) % self.vocab as u64) as usize] = 7.0;
            }
            if self.score_rows_on {
                // every row of the scored context: row i follows tokens 0..=i (it predicts token i + 1); the last is the returned row
                self.score_rows = (1..ctx.len()).map(|n| mock_logits(&ctx[..n], self.vocab)).collect();
                self.score_rows.push(logits.clone());
            }
            Some(logits)
        } else {
            None
        };
        Ok(crate::score::ScoreStep { done: c.step.done, row, captured: c.captured })
    }
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn take_error(&mut self) -> Option<BackendError> {
        // a fatal error sticks, as a device fault does; anything else is reported once
        self.latched.clone().or_else(|| self.error.take())
    }
    fn prefill_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[crate::ImageSpan],
        done: usize,
        candidates: usize,
    ) -> crate::PrefillStep {
        self.prefill_slice_cuts(slot, prompt, images, done, candidates, &[]).step
    }
    fn prefill_slice_cuts(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[crate::ImageSpan],
        done: usize,
        candidates: usize,
        cuts: &[usize],
    ) -> crate::CutStep {
        self.slice_calls += 1;
        if self.faults.panic_at_prefill == Some(self.slice_calls) {
            panic!("injected panic in prefill slice {}", self.slice_calls);
        }
        if !self.prefill_delay.is_zero() {
            std::thread::sleep(self.prefill_delay);
        }
        let n = prompt.len();
        let end = (done + self.slice.unwrap_or(n).min(self.deadline_rows.unwrap_or(usize::MAX))).min(n);
        if self.kv.is_some() || self.ring.is_some() || self.path_state {
            // the prompt's total picks the path; a slice that starts a prompt begins its slot (a restored one has `done > 0`)
            if done == 0 {
                self.clear_slot(slot);
            }
            let mut captured = Vec::new();
            let r = self.feed(slot, done, &prompt[done..end], n, cuts, &mut captured, candidates);
            return crate::CutStep { step: crate::PrefillStep { done: end, row: (end == n).then_some(r) }, captured };
        }
        if done == 0 && images.is_empty() && self.slice.is_none() {
            let row = Some(self.prefill(slot, 0, prompt, candidates));
            return crate::CutStep { step: crate::PrefillStep { done: n, row }, captured: Vec::new() };
        }
        let r = self.prefill(slot, done as u64, &prompt[done..end], candidates);
        crate::CutStep { step: crate::PrefillStep { done: end, row: (end == n).then_some(r) }, captured: Vec::new() }
    }
    fn prefill_slices(&mut self, jobs: &[crate::PrefillJob<'_>]) -> Vec<crate::CutStep> {
        if jobs.len() >= 2 {
            self.batched_cache_jobs += jobs.iter().filter(|j| !j.cuts.is_empty() || j.done > 0).count();
        }
        jobs.iter().map(|j| self.prefill_slice_cuts(j.slot, j.prompt, j.images, j.done, j.candidates, j.cuts)).collect()
    }
    fn prefill_slices_take_cuts(&self) -> bool {
        self.pool_cache
    }
    fn prefix_end_cut(&self) -> bool {
        self.pool_cache
    }
    fn kv_pool(&self) -> Option<crate::KvPool> {
        let kv = self.kv.as_ref().filter(|_| self.pool_cache)?;
        Some(crate::KvPool { free: kv.free.len(), slot_pages: kv.tables.iter().map(|t| t.len()).collect() })
    }
    fn prefix_caps(&self) -> Option<PrefixCaps> {
        self.cache.as_ref().map(|c| c.caps)
    }
    fn prefix_path(&self, total: usize) -> Option<u32> {
        let c = self.cache.as_ref()?;
        // the GLM shape has one arithmetic for every length: no path rule, every prompt may be cached
        (total > MOCK_GEMV_MAX || c.fault == PrefixFault::PathIgnored || self.glm.is_some()).then_some(1)
    }
    fn prefix_free_snapshots(&self) -> usize {
        self.cache.as_ref().map_or(0, |c| c.snaps.iter().filter(|s| s.is_none()).count())
    }
    fn prefix_restore(&mut self, slot: usize, handle: &PrefixHandle, len: usize) -> Result<(), String> {
        self.clear_slot(slot);
        let (Some(cache), Some(kv)) = (self.cache.as_ref(), self.kv.as_mut()) else { return Err("no prefix cache".into()) };
        let snap = cache.snaps.get(handle.snap as usize).and_then(|s| s.as_ref()).ok_or("the snapshot slot is empty")?;
        let fault = cache.fault;
        let full = len / MOCK_PAGE;
        for (i, &p) in handle.pages.iter().enumerate() {
            if i < full || fault == PrefixFault::NoCopyOnWrite {
                kv.retain(p);
                kv.tables[slot].push(p);
            } else {
                // the partial last page: the slot gets its own copy, so its first write cannot touch the entry's (or the donor's) page
                let np = kv.alloc();
                let content = kv.pages[p as usize].clone();
                kv.pages[np as usize] = content;
                if fault != PrefixFault::PoolsNotCopied {
                    let pools = kv.pools[p as usize].clone();
                    kv.pools[np as usize] = pools;
                }
                let entries = kv.mtp[p as usize].clone();
                kv.mtp[np as usize] = entries;
                kv.tables[slot].push(np);
            }
        }
        if let Some(g) = self.glm.as_mut() {
            // the raw rows of the incomplete pool group go back into the slot's ring, and the draft's carry with them
            if fault != PrefixFault::RingTailNotRestored {
                let tail = len % 4;
                for (i, &v) in snap.raw_tail.iter().enumerate() {
                    g.raw[slot][(len - tail + i) % MOCK_RAW_RING] = v;
                }
            }
            if fault != PrefixFault::CarryNotRestored {
                g.carry[slot] = snap.carry;
            }
        }
        if fault != PrefixFault::StateNotRestored {
            self.state[slot] = snap.state;
        }
        if let (Some(ring), true) = (self.ring.as_mut(), fault != PrefixFault::DraftNotRestored) {
            for (i, &e) in snap.ring.iter().enumerate() {
                ring[slot][(snap.lo + i) % MOCK_RING] = e;
            }
        }
        self.ctx[slot] = kv.read(slot, len);
        Ok(())
    }
    fn prefix_release(&mut self, handle: PrefixHandle) {
        let (Some(cache), Some(kv)) = (self.cache.as_mut(), self.kv.as_mut()) else { return };
        assert!(cache.snaps[handle.snap as usize].take().is_some(), "snapshot slot {} released twice", handle.snap);
        match cache.fault {
            PrefixFault::RefcountLeak => {}
            PrefixFault::FreeWhileReferenced => {
                for &p in &handle.pages {
                    if kv.refs[p as usize] > 0 {
                        kv.refs[p as usize] = 0;
                        kv.free.push(p);
                    }
                }
            }
            _ => {
                for &p in &handle.pages {
                    kv.release(p);
                }
            }
        }
    }
    fn propose(&mut self, slot: usize, tokens: &[u32], _s: &Sampling, max: usize) -> Option<Vec<u32>> {
        if self.glm.is_some() {
            // the GLM drafter reads its cache from the pages (a hit shares them): the entries of the committed positions `0..p` (the token at `p` has
            // not been fed), each built from its token and the tap before it (a restored carry builds the first entry after the cut); the pools of those
            // positions take part too, as the draft's own indexer reads them
            let p = tokens.len() - 1;
            let acc = h(h(0x5EED, self.glm_mtp_digest(slot, p)), if p > 0 { self.glm_digest(slot, p - 1) } else { 0 });
            let mut ctx = tokens.to_vec();
            let mut out = Vec::new();
            for i in 0..max {
                let l = mock_logits(&ctx, self.vocab);
                let (ids, _) = top_candidates(&l, 1);
                let mut t = ids[0];
                if h(acc, (p + i) as u64) % 5 == 0 {
                    t = (t + 1) % self.vocab as u32;
                }
                out.push(t);
                ctx.push(t);
            }
            return Some(out);
        }
        let ring = self.ring.as_ref()?;
        // the draft reads its context from the ring: the positions a block at the anchor's position can attend to
        let p = tokens.len() - 1;
        let lo = p.saturating_sub(MOCK_WINDOW);
        let mut acc = 0x5EED_u64;
        let mut ctx: Vec<u32> = (lo..p)
            .map(|pos| {
                let (at, t) = ring[slot][pos % MOCK_RING];
                let t = if at == pos as i64 { t } else { 0 };
                acc = h(acc, (t as u64) << 1 | (at == pos as i64) as u64);
                t
            })
            .collect();
        ctx.push(tokens[p]);
        let mut out = Vec::new();
        for i in 0..max {
            let l = mock_logits(&ctx, self.vocab);
            let (ids, _) = top_candidates(&l, 1);
            let mut t = ids[0];
            if h(acc, (p + i) as u64) % 5 == 0 {
                t = (t + 1) % self.vocab as u32;
            }
            out.push(t);
            ctx.push(t);
        }
        Some(out)
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        if self.kv.is_some() || self.ring.is_some() || self.path_state {
            let total = start as usize + tokens.len();
            return self.feed(slot, start as usize, tokens, total, &[], &mut Vec::new(), candidates);
        }
        assert_eq!(self.ctx[slot].len() as u64, start, "prefill must continue the slot's context");
        self.ctx[slot].extend_from_slice(tokens);
        self.prefill_calls[slot] += 1;
        for &t in tokens {
            self.state[slot] = fold_state(self.state[slot], t);
        }
        let mut r = self.row(&self.ctx[slot], self.state[slot], candidates);
        if self.fault == Fault::ChunkDependent && self.prefill_calls[slot] > 1 {
            r.values[0] = f32::from_bits(r.values[0].to_bits() ^ 1);
            // flip a tie-break-relevant bit and swap the top two so the sample can change
            if r.values.len() > 1 {
                r.values.swap(0, 1);
                r.ids.swap(0, 1);
            }
        }
        r
    }
    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        // a chain: every row's parent is the row before it
        let chains: Vec<(usize, Vec<u32>, Vec<i32>, u64)> = reqs
            .iter()
            .map(|q| (q.slot, q.tokens.clone(), (0..q.tokens.len() as i32).map(|i| i - 1).collect(), q.first_position))
            .collect();
        self.run_step(&chains, candidates)
    }
    fn supports_trees(&self) -> bool {
        self.trees
    }
    fn step_tree(&mut self, reqs: &[crate::TreeRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        assert!(
            self.trees || reqs.iter().all(crate::TreeRequest::is_chain),
            "this backend does not verify trees (supports_trees is false)"
        );
        let trees: Vec<(usize, Vec<u32>, Vec<i32>, u64)> =
            reqs.iter().map(|q| (q.slot, q.tokens.clone(), q.parents.clone(), q.first_position)).collect();
        self.run_step(&trees, candidates)
    }
    fn commit_path(&mut self, slot: usize, path: &[usize]) {
        let p = std::mem::take(&mut self.pending[slot]);
        let st = std::mem::take(&mut self.pending_state[slot]);
        let parents = std::mem::take(&mut self.pending_parents[slot]);
        assert!(
            self.trees || path.iter().enumerate().all(|(i, &r)| i == r),
            "this backend does not commit a tree path: {path:?}"
        );
        assert!(!path.is_empty() && path[0] == 0, "a committed path starts at the root: {path:?}");
        for w in path.windows(2) {
            assert_eq!(parents[w[1]], w[0] as i32, "row {} is not a child of row {} in {parents:?}", w[1], w[0]);
        }
        assert!(path.iter().all(|&r| r < p.len()), "commit_path({path:?}) of a {}-row step", p.len());
        let first = self.ctx[slot].len();
        self.ctx[slot].extend(path.iter().map(|&r| p[r]));
        // the accepted rows enter the paged KV at their places on the path (a tree window wrote none of its rows)
        if let Some(kv) = self.kv.as_mut() {
            for (j, &r) in path.iter().enumerate() {
                kv.write(slot, first + j, p[r]);
            }
        }
        // the draft's context grows by exactly the rows the target committed
        if let Some(ring) = self.ring.as_mut() {
            for (j, &r) in path.iter().enumerate() {
                ring[slot][(first + j) % MOCK_RING] = ((first + j) as i64, p[r]);
            }
        }
        self.state[slot] = if self.fault == Fault::RollbackLeak { *st.last().unwrap() } else { st[*path.last().unwrap()] };
        if let Some(g) = self.glm.as_mut() {
            g.carry[slot] = Self::tap(self.state[slot]);
        }
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        let p = std::mem::take(&mut self.pending[slot]);
        let st = std::mem::take(&mut self.pending_state[slot]);
        assert!(keep >= 1 && keep <= p.len(), "commit({keep}) of a {}-row step", p.len());
        let first = self.ctx[slot].len();
        self.ctx[slot].extend_from_slice(&p[..keep]);
        // the draft's context grows by exactly the rows the target committed
        if let Some(ring) = self.ring.as_mut() {
            for (j, &t) in p[..keep].iter().enumerate() {
                ring[slot][(first + j) % MOCK_RING] = ((first + j) as i64, t);
            }
        }
        self.state[slot] = if self.fault == Fault::RollbackLeak { *st.last().unwrap() } else { st[keep - 1] };
        if let Some(g) = self.glm.as_mut() {
            g.carry[slot] = Self::tap(self.state[slot]);
        }
    }
    fn release(&mut self, slot: usize) {
        // the draft's ring is not cleared: the next owner of the slot finds the previous one's keys, as on the GPU
        self.clear_slot(slot);
    }
}

fn reorder(ids: &[u32], vals: &[f32]) -> (Vec<u32>, Vec<f32>) {
    let mut idx: Vec<usize> = (0..ids.len()).collect();
    idx.sort_by(|&a, &b| vals[b].partial_cmp(&vals[a]).unwrap().then(ids[a].cmp(&ids[b])));
    (idx.iter().map(|&i| ids[i]).collect(), idx.iter().map(|&i| vals[i]).collect())
}

/// Test drafter: follows the mock's argmax chain and corrupts some guesses deterministically, so rounds
/// see full, partial and zero acceptance. Stands in for DFlash2 until the GPU drafter exists.
pub struct ChainDrafter {
    pub vocab: usize,
    pub corrupt_every: u64,
}

/// A drafter with a distribution of its own, close to the target's: the mock's logits plus a deterministic perturbation of size `eps` per
/// token (`eps` 0: the drafter's logits ARE the target's). It chains like a NextN head (each draft is fed back as the next token) and proposes
/// by `pick`: its argmax, or the target's keyed draw over its own candidates at the draft's own position (`gb_spec::DraftPick`), which is what the
/// GLM NextN head does with `REXL3_GLM_DRAFT_NOISE`. `wrong_at` makes every draft of that depth (0-based) a wrong id, `shift_position` draws the
/// noise of another position (a deliberate bug for the mutation checks). Counts the proposals it made.
pub struct HeadDrafter {
    pub vocab: usize,
    pub eps: f32,
    pub pick: gb_spec::DraftPick,
    pub wrong_at: Option<usize>,
    /// Every draft is a wrong id (a head that is numerically clean and always wrong).
    pub always_wrong: bool,
    pub shift_position: i64,
}

impl HeadDrafter {
    pub fn new(vocab: usize, eps: f32, pick: gb_spec::DraftPick) -> Self {
        Self { vocab, eps, pick, wrong_at: None, always_wrong: false, shift_position: 0 }
    }

    /// The drafter's logits after `ctx`.
    pub fn logits(&self, ctx: &[u32]) -> Vec<f32> {
        let mut l = mock_logits(ctx, self.vocab);
        if self.eps != 0.0 {
            let key = h(ctx.len() as u64 ^ 0xA5A5, *ctx.last().unwrap_or(&0) as u64);
            for (i, v) in l.iter_mut().enumerate() {
                *v += self.eps * (((h(key, i as u64) >> 40) as f32 / 16777216.0) * 2.0 - 1.0);
            }
        }
        l
    }
}

impl Drafter for HeadDrafter {
    fn propose(&mut self, tokens: &[u32], s: &Sampling, max: usize) -> Vec<u32> {
        let mut ctx = tokens.to_vec();
        let mut out = Vec::new();
        for k in 0..max {
            // the target's own candidate count: the drafter reads exactly what the engine's sampler would read of a row
            let (ids, vals) = top_candidates(&self.logits(&ctx), s.candidate_count(self.vocab));
            let pos = (gb_spec::draft_position(tokens.len(), k) as i64 + self.shift_position) as u64;
            let mut t = gb_spec::try_draft_pick(&vals, &ids, pos, s, self.pick).expect("the mock's logits are finite");
            if self.always_wrong || self.wrong_at == Some(k) {
                t = (t + 1) % self.vocab as u32;
            }
            out.push(t);
            ctx.push(t);
        }
        out
    }
}

impl Drafter for ChainDrafter {
    fn propose(&mut self, tokens: &[u32], _s: &Sampling, max: usize) -> Vec<u32> {
        let mut ctx = tokens.to_vec();
        let mut out = Vec::new();
        for i in 0..max {
            let l = mock_logits(&ctx, self.vocab);
            let (ids, _) = top_candidates(&l, 1);
            let mut t = ids[0];
            if h(ctx.len() as u64, i as u64) % self.corrupt_every == 0 {
                t = (t + 1) % self.vocab as u32;
            }
            out.push(t);
            ctx.push(t);
        }
        out
    }
}
