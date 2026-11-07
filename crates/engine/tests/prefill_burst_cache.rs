//! The prefix cache under the burst prefill policy (`PrefillPolicy::Burst`: several prefill slices a step while prompts queue up, the slices of
//! the prompts that use no cache handed to the backend in ONE `Backend::prefill_slices` call). The join this file proves:
//!
//! * every prompt a burst step picks consults the cache right before its own slice (`plan_prefix`), so a prompt behind one that leaves a
//!   snapshot in the same step finds it, and `cache: false` and a family without a cache are honoured;
//! * a prompt that restored from a hit or has cuts to take gets a slice of its own (`prefill_slice_cuts`/`prefill_slice`: a restore resumes ONE
//!   slot and a snapshot is taken inside ONE prompt's chunk); only prompts that use the cache in no way are batched;
//! * every prompt, whichever call served it, ends in the same tail: snapshots committed or given back, `cached_tokens` recorded, a failure
//!   isolated to its own prompt, a cancel or a fatal error leaving nothing allocated.
//!
//! The backend is the cache-enabled mock (paged KV the logits are read from, a path-dependent recurrent state, a draft ring) behind a wrapper
//! that logs every call that matters and reports, as `Backend::prefill_slices` promises, nothing it was not asked.
//!
//! The same scenarios run with DRAFT TREES (`Setup::trees`: the wrapper drafts a candidate lattice, the mock verifies trees and commits a path,
//! the engine plans trees): a tree round and the burst prefill share steps, and a hit's restore feeds the tree rounds that follow it
//! (docs/TREE_DRAFTING.md "Interaction with the burst prefill, the prefix cache and `--window-rows`").

use gb_engine::mock::{mock_logits, ChainDrafter, Fault, MockBackend, PrefixFault};
use gb_engine::{
    Backend, BackendError, CacheBudget, CutPolicy, CutStep, Drafter, Engine, EngineConfig, FinishReason, ImageSpan, Output,
    PrefillJob, PrefillPolicy, PrefillStep, PrefixCaps, PrefixConfig, PrefixHandle, Proposal, ProposeRequest, Request, RowOutput,
    RowRequest, StepError, TreeConfig, TreeRequest, PREFILL_PASS_GUARD,
};
use gb_sampling::Sampling;
use gb_sched::{Admission, CostModel, ModelMemory};
use gb_spec::tree::Lattice;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

const VOCAB: usize = 512;
const SLOTS: usize = 24;
/// The tail margin of the cut policy under test: a prompt of `n` tokens above 128 is cut at `n - 8` (when that is at least 64).
const TAIL: usize = 8;

/// One call the engine made to the backend that tells where a prompt's slice went (prompts are named by their length: every prompt of a scenario
/// has a length of its own).
#[derive(Clone, Debug, PartialEq)]
enum Ev {
    /// `prefill_slices`: the prompt length of each job, in job order.
    Batch(Vec<usize>),
    /// `prefill_slice_cuts`: prompt length, `done`, number of cuts given.
    Cuts(usize, usize, usize),
    /// `prefill_slice`: prompt length, `done`.
    Single(usize, usize),
    /// `prefix_restore`: the length restored.
    Restore(usize),
}

type Log = Rc<RefCell<Vec<Ev>>>;

struct Routed {
    inner: MockBackend,
    /// The drafts carry a candidate lattice (the backend drafts trees) and the mock verifies them.
    trees: bool,
    log: Log,
    /// Slots restored from a hit and not released since: a restored prompt in a batch call is a contract violation.
    restored: Vec<bool>,
    /// Prompts of these lengths cannot be served: their row is NaN and the call reports a recoverable error (as the Qwen backend answers).
    poison: Vec<usize>,
    /// A prompt of this length (first) fails from the slice that starts at this many tokens (second) on: its earlier slices were served, so any
    /// snapshot they took is held on the prompt when it fails.
    late: Option<(usize, usize)>,
    /// A prompt of this length ends its slice in a fatal error.
    fatal: Option<usize>,
    error: Option<BackendError>,
    violations: Rc<RefCell<Vec<String>>>,
}

impl Routed {
    fn hurt(&mut self, len: usize, done: usize, candidates: usize, step: &mut PrefillStep) {
        if self.late.is_some_and(|(l, from)| l == len && done >= from) {
            step.done = len;
            step.row = Some(RowOutput { ids: (0..candidates as u32).collect(), values: vec![f32::NAN; candidates] });
            self.error.get_or_insert(BackendError::retryable(format!("injected late failure of the prompt of {len} tokens")));
            return;
        }
        if self.fatal == Some(len) {
            self.error = Some(BackendError::fatal(format!("injected fatal error in the prompt of {len} tokens")));
        } else if self.poison.contains(&len) {
            step.done = len;
            step.row = Some(RowOutput { ids: (0..candidates as u32).collect(), values: vec![f32::NAN; candidates] });
            self.error.get_or_insert(BackendError::retryable(format!("injected failure of the prompt of {len} tokens")));
        }
    }
}

impl Backend for Routed {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn take_error(&mut self) -> Option<BackendError> {
        self.error.take().or_else(|| self.inner.take_error())
    }
    fn supports_trees(&self) -> bool {
        self.inner.supports_trees()
    }
    fn step_tree(&mut self, reqs: &[TreeRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        self.inner.step_tree(reqs, candidates)
    }
    fn commit_path(&mut self, slot: usize, path: &[usize]) {
        self.inner.commit_path(slot, path)
    }
    fn propose_batch_scored(&mut self, reqs: &[ProposeRequest]) -> Vec<Option<Proposal>> {
        if !self.trees {
            return self
                .propose_batch(reqs)
                .into_iter()
                .map(|o| o.map(|tokens| Proposal { tokens, ..Default::default() }))
                .collect();
        }
        reqs.iter()
            .map(|r| {
                let (chain, l) = lattice(r.tokens, r.max);
                Some(Proposal { tokens: chain, lattice: Some(l), ..Default::default() })
            })
            .collect()
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.inner.prefill(slot, start, tokens, candidates)
    }
    fn prefill_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
    ) -> PrefillStep {
        self.log.borrow_mut().push(Ev::Single(prompt.len(), done));
        let mut step = self.inner.prefill_slice(slot, prompt, images, done, candidates);
        self.hurt(prompt.len(), done, candidates, &mut step);
        step
    }
    fn prefill_slices(&mut self, jobs: &[PrefillJob<'_>]) -> Vec<CutStep> {
        self.log.borrow_mut().push(Ev::Batch(jobs.iter().map(|j| j.prompt.len()).collect()));
        for j in jobs {
            if !j.cuts.is_empty() {
                self.violations.borrow_mut().push(format!(
                    "the prompt of {} tokens was put in a batch with cuts {:?}",
                    j.prompt.len(),
                    j.cuts
                ));
            }
            if self.restored[j.slot] {
                self.violations
                    .borrow_mut()
                    .push(format!("the prompt of {} tokens, restored from a hit, was put in a batch", j.prompt.len()));
            }
        }
        jobs.iter()
            .map(|j| {
                // what the real backend does with a job it cannot batch: its own `prefill_slice`
                let mut step = self.inner.prefill_slice(j.slot, j.prompt, j.images, j.done, j.candidates);
                self.hurt(j.prompt.len(), j.done, j.candidates, &mut step);
                CutStep { step, captured: Vec::new() }
            })
            .collect()
    }
    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        self.inner.step(reqs, candidates)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.inner.commit(slot, keep)
    }
    fn release(&mut self, slot: usize) {
        self.restored[slot] = false;
        self.inner.release(slot)
    }
    fn propose(&mut self, slot: usize, tokens: &[u32], sampling: &Sampling, max: usize) -> Option<Vec<u32>> {
        self.inner.propose(slot, tokens, sampling, max)
    }
    fn prefix_caps(&self) -> Option<PrefixCaps> {
        self.inner.prefix_caps()
    }
    fn prefix_path(&self, total: usize) -> Option<u32> {
        self.inner.prefix_path(total)
    }
    fn prefix_free_snapshots(&self) -> usize {
        self.inner.prefix_free_snapshots()
    }
    fn prefix_restore(&mut self, slot: usize, handle: &PrefixHandle, len: usize) -> Result<(), String> {
        self.log.borrow_mut().push(Ev::Restore(len));
        let r = self.inner.prefix_restore(slot, handle, len);
        self.restored[slot] = r.is_ok();
        r
    }
    fn prefix_release(&mut self, handle: PrefixHandle) {
        self.inner.prefix_release(handle)
    }
    fn prefill_slice_cuts(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
        cuts: &[usize],
    ) -> CutStep {
        self.log.borrow_mut().push(Ev::Cuts(prompt.len(), done, cuts.len()));
        let mut cs = self.inner.prefill_slice_cuts(slot, prompt, images, done, candidates, cuts);
        self.hurt(prompt.len(), done, candidates, &mut cs.step);
        cs
    }
}

const K: usize = 8;

fn hash(a: u64, b: u64) -> u64 {
    let mut x = a.wrapping_mul(0x9E3779B97F4A7C15) ^ b.wrapping_add(0xD1B54A32D192ED03);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^ (x >> 29)
}

/// A drafter's lattice over the mock's own next-token function (the one `tests/tree_rounds.rs` drafts with): `rows` levels of the `K` best tokens
/// under the chain's context or under any of the previous level's top candidates as the parent, the unary score carrying a hashed error so that
/// the best-first walk sometimes picks a wrong token with the right one a sibling. Returns the selector's own walk as the chain too.
fn lattice(tokens: &[u32], rows: usize) -> (Vec<u32>, Lattice) {
    let mut ids = vec![0u32; rows * K];
    let mut unary = vec![0f32; rows * K];
    let mut edge = vec![0f32; rows * K * K];
    let mut ctx = tokens.to_vec();
    let mut chain = Vec::new();
    let mut parent = 0usize;
    for d in 0..rows {
        let parents: Vec<Vec<f32>> = if d == 0 {
            vec![mock_logits(&ctx, VOCAB)]
        } else {
            let mut base = ctx.clone();
            base.pop();
            (0..K)
                .map(|p| {
                    let mut c = base.clone();
                    c.push(ids[(d - 1) * K + p]);
                    mock_logits(&c, VOCAB)
                })
                .collect()
        };
        let own = mock_logits(&ctx, VOCAB);
        let mut score: Vec<(f32, u32)> =
            (0..VOCAB).map(|t| (parents.iter().map(|l| l[t]).fold(f32::NEG_INFINITY, f32::max), t as u32)).collect();
        score.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut cands: Vec<u32> = score.iter().take(K).map(|s| s.1).collect();
        let noisy = |t: u32| {
            own[t as usize] + ((hash(d as u64 * 7919 + ctx.len() as u64, t as u64) >> 40) as f32 / 16777216.0 - 0.5) * 3.0
        };
        cands.sort_by(|&a, &b| noisy(b).total_cmp(&noisy(a)).then(a.cmp(&b)));
        for (c, &t) in cands.iter().enumerate() {
            ids[d * K + c] = t;
            unary[d * K + c] = noisy(t);
            for p in 0..K {
                edge[(d * K + p) * K + c] = if d == 0 { 0.0 } else { parents[p][t as usize] };
            }
        }
        let sc = |c: usize| unary[d * K + c] + edge[(d * K + parent) * K + c];
        let best = (1..K).fold(0, |b, c| if sc(c) > sc(b) { c } else { b });
        chain.push(ids[d * K + best]);
        ctx.push(ids[d * K + best]);
        parent = best;
    }
    (chain, Lattice { rows, k: K, ids, unary, edge, restricted_head: false })
}

fn admission() -> Admission {
    let mm = ModelMemory {
        weights_bytes: 1 << 30,
        kv_bytes_per_token: 65536,
        state_bytes_per_slot: 150 << 20,
        activation_reserve: 1 << 30,
    };
    Admission::new(64 << 30, 6 << 30, mm, SLOTS).unwrap()
}

fn engine_config() -> EngineConfig {
    EngineConfig {
        max_rows: 64,
        max_drafts: 7,
        cost: CostModel::from_table(vec![(1, 70.0), (16, 74.0), (32, 110.0), (64, 190.0)]),
        drafting: true,
    }
}

fn drafter() -> Box<dyn Fn() -> Box<dyn Drafter>> {
    Box::new(|| Box::new(ChainDrafter { vocab: VOCAB, corrupt_every: 5 }) as Box<dyn Drafter>)
}

fn cut_policy() -> PrefixConfig {
    PrefixConfig {
        policy: CutPolicy { min_cut: 64, tail: TAIL, ghost_min_gain: 256, min_gap: 0 },
        ghost_blocks: 1 << 12,
        ..Default::default()
    }
}

fn budget(units: usize) -> CacheBudget {
    CacheBudget { snapshot_slots: units, page_blocks: units * 8, snapshot_bytes: 1 << 20, page_bytes: 1 << 16 }
}

struct Setup {
    units: usize,
    slice: Option<usize>,
    policy: PrefillPolicy,
    batching: bool,
    /// The engine runs the cache (a family whose `prefix_cache()` is false never calls `with_prefix_cache`, whatever the backend could do).
    cache: bool,
    poison: Vec<usize>,
    late: Option<(usize, usize)>,
    fatal: Option<usize>,
    /// The engine verifies draft trees (`Engine::with_trees`) on a backend that drafts lattices and verifies them.
    trees: bool,
    /// With trees: the shortest match of the copy branch (`TreeConfig::copy_min_match`), 0 for none.
    copy: usize,
}

impl Default for Setup {
    fn default() -> Self {
        Setup {
            units: 12,
            slice: None,
            policy: PrefillPolicy::Burst { div: 1, cap: 32 },
            batching: true,
            cache: true,
            poison: Vec::new(),
            late: None,
            fatal: None,
            trees: false,
            copy: 0,
        }
    }
}

struct Rig {
    e: Engine<Routed>,
    log: Log,
    violations: Rc<RefCell<Vec<String>>>,
}

fn rig(s: Setup) -> Rig {
    let (log, violations) = (Log::default(), Rc::new(RefCell::new(Vec::new())));
    let mut inner = MockBackend::new(VOCAB, SLOTS, Fault::None)
        .with_path_dependent_state()
        .with_native_draft()
        .with_prefix_cache(budget(s.units), PrefixFault::None);
    if let Some(n) = s.slice {
        inner = inner.with_prefill_slice(n);
    }
    if s.trees {
        inner = inner.with_trees();
    }
    let backend = Routed {
        inner,
        trees: s.trees,
        log: log.clone(),
        restored: vec![false; SLOTS],
        poison: s.poison,
        late: s.late,
        fatal: s.fatal,
        error: None,
        violations: violations.clone(),
    };
    let mut e = Engine::new(backend, admission(), engine_config(), drafter())
        .with_prefill_policy(s.policy)
        .with_prefill_batching(s.batching);
    if s.cache {
        e = e.with_prefix_cache(cut_policy());
    }
    if s.trees {
        e = e.with_trees(TreeConfig { copy_min_match: s.copy, ..TreeConfig::default() });
        assert!(e.verifies_trees());
    }
    Rig { e, log, violations }
}

/// The reply of `r` alone, uncached, on the first release's serial path.
fn reference(r: &Request, slice: Option<usize>) -> Output {
    let mut b =
        MockBackend::new(VOCAB, SLOTS, Fault::None).with_path_dependent_state().with_native_draft().with_paged_kv(SLOTS * 64);
    if let Some(n) = slice {
        b = b.with_prefill_slice(n);
    }
    let mut e = Engine::new(b, admission(), engine_config(), drafter());
    let id = e.submit(Request { cache: false, ..r.clone() }).unwrap();
    e.run().unwrap();
    e.take_outputs().into_iter().find(|o| o.id == id).unwrap()
}

/// Tokens and finish reason (what a client sees); the speculation counters depend on who else is in a round.
fn same_reply(a: &Output, b: &Output) -> Result<(), String> {
    if a.tokens != b.tokens {
        let at = a.tokens.iter().zip(&b.tokens).position(|(x, y)| x != y).unwrap_or(a.tokens.len().min(b.tokens.len()));
        return Err(format!("tokens differ at {at} ({} vs {} tokens)", a.tokens.len(), b.tokens.len()));
    }
    if (a.finish, a.stopped) != (b.finish, b.stopped) {
        return Err(format!("finish {:?} vs {:?}", a.finish, b.finish));
    }
    Ok(())
}

fn toks(seed: u64, n: usize) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1 + (x % (VOCAB as u64 - 1)) as u32
        })
        .collect()
}

fn request(prompt: Vec<u32>, seed: u64) -> Request {
    Request {
        prompt,
        sampling: Sampling { seed, temperature: if seed % 2 == 0 { 0.0 } else { 0.9 }, top_k: 20, top_p: 0.95, min_p: 0.0 },
        max_new: 24 + (seed % 5) as usize * 7,
        stop: vec![],
        draft: true,
        images: vec![],
        cache: true,
    }
}

fn no_cache(r: Request) -> Request {
    Request { cache: false, ..r }
}

/// A prompt that starts with `base` and goes on for `extra` tokens of its own.
fn extended(base: &[u32], seed: u64, extra: usize) -> Vec<u32> {
    let mut p = base.to_vec();
    p.extend(toks(seed, extra));
    p
}

/// Pages and snapshot slots are all accounted for: with nothing running, what the backend holds is exactly what the index holds.
fn no_leaks(r: &Rig) -> Result<(), String> {
    let st = r.e.prefix_stats();
    if r.e.backend.inner.pages_in_use() != st.pages_held as usize {
        return Err(format!("{} pages in use, the index holds {}", r.e.backend.inner.pages_in_use(), st.pages_held));
    }
    if r.e.backend.inner.snapshots_in_use() != st.entries as usize {
        return Err(format!(
            "{} snapshot slots in use, the index holds {} entries",
            r.e.backend.inner.snapshots_in_use(),
            st.entries
        ));
    }
    Ok(())
}

/// Rounds the engine verified as trees so far.
fn tree_rounds(e: &Engine<Routed>) -> u64 {
    e.spec_stats().trees.iter().map(|t| t.rounds).sum()
}

/// Rounds whose tree carried a copy branch, and the copied tokens accepted, so far.
fn copy_rounds(e: &Engine<Routed>) -> (u64, u64) {
    let st = e.spec_stats();
    (st.trees.iter().map(|t| t.copy_rounds).sum(), st.trees.iter().map(|t| t.copy_accepted).sum())
}

fn all_slots_back(r: &Rig) {
    assert_eq!(r.e.admission().available(), r.e.admission().capacity(), "every slot and byte is back");
}

fn outputs(e: &mut Engine<Routed>) -> BTreeMap<u64, Output> {
    e.take_outputs().into_iter().map(|o| (o.id, o)).collect()
}

// ------------------------------------------------------------------------------------------------------------------------------------
// the mixed burst: one step holds every kind of prompt
// ------------------------------------------------------------------------------------------------------------------------------------

/// What a prompt of the mixed burst is, for the cache.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// Longer than 128 tokens, opted in, starts with an entry already stored: restored, and has its own end cut to take.
    HitAndCut,
    /// The warm prompt itself again: restored from its end cut, and nothing left to cut (the entry for that cut is stored already).
    HitOnly,
    /// Longer than 128 tokens, opted in, cold: an end cut to take.
    Cut,
    /// Up to 128 tokens: the GEMV path, never cached.
    Short,
    /// Opted out (`cache: false`): nothing looked up, nothing stored, whatever its length.
    OptedOut,
}

struct Case {
    r: Request,
    kind: Kind,
}

/// The prompt every `HitAndCut` request starts with, warmed into the cache before the burst (600 tokens: its end cut is 592).
fn warm_prompt() -> Vec<u32> {
    toks(1, 600)
}

/// Twelve requests: 3 that hit the warm entry (two sharers of 621 and 634 tokens, and the warm prompt itself again, 600), 3 cold long prompts with a cut plan (130, 200 and 260), 3 short ones (40,
/// 55 and 70), and 3 that opted out (a short one of 90, a cold long one of 300 and a sharer of the warm prompt of 640). Their rows add up to 3,640,
/// inside the 4,096 a burst step may prefill, so with `Burst { div: 1, cap: 32 }` ONE step picks them all.
fn mixed() -> Vec<Case> {
    let w = warm_prompt();
    let mut v = Vec::new();
    let mut add = |prompt: Vec<u32>, kind: Kind| {
        let seed = v.len() as u64 + 10;
        let r = request(prompt, seed);
        v.push(Case { r: if kind == Kind::OptedOut { no_cache(r) } else { r }, kind });
    };
    add(extended(&w, 101, 21), Kind::HitAndCut);
    add(toks(11, 200), Kind::Cut);
    add(toks(12, 40), Kind::Short);
    add(extended(&w, 102, 34), Kind::HitAndCut);
    add(toks(13, 260), Kind::Cut);
    add(toks(14, 55), Kind::Short);
    add(toks(15, 90), Kind::OptedOut);
    add(w.clone(), Kind::HitOnly);
    add(toks(16, 130), Kind::Cut);
    add(toks(17, 70), Kind::Short);
    add(toks(18, 300), Kind::OptedOut);
    add(extended(&w, 104, 40), Kind::OptedOut);
    v
}

fn expected_cached(c: &Case) -> usize {
    match c.kind {
        Kind::HitAndCut | Kind::HitOnly => 592,
        _ => 0,
    }
}

fn warmed(setup: Setup) -> Rig {
    let mut r = rig(setup);
    let a = request(warm_prompt(), 3);
    let id = r.e.submit(a.clone()).unwrap();
    r.e.run().unwrap();
    let o = outputs(&mut r.e).remove(&id).unwrap();
    assert_eq!(o.cached_tokens, 0, "a cold cache");
    assert_eq!(r.e.prefix_stats().entries, 1, "the warm prompt left its end cut");
    r.log.borrow_mut().clear();
    r
}

#[test]
fn a_mixed_burst_sends_each_prompt_where_the_cache_needs_it_and_every_reply_is_exact() {
    mixed_burst(false);
}

/// The same burst with draft trees: the prefill step is the same (nothing is decoding yet), and every reply that follows, after hits that
/// restored state and cuts that took snapshots, comes out of tree rounds equal to the uncached serial reply.
#[test]
fn a_mixed_burst_is_exact_with_draft_trees_and_the_trees_ran() {
    mixed_burst(true);
}

fn mixed_burst(trees: bool) {
    let cases = mixed();
    let mut r = warmed(Setup { trees, ..Default::default() });
    let ids: Vec<u64> = cases.iter().map(|c| r.e.submit_deferred(c.r.clone()).unwrap()).collect();
    // ONE step: the decode round has nothing to run, so everything below is what `advance_prefills` did with the 12 picked prompts
    r.e.step().unwrap();
    assert_eq!(r.e.prefilling(), 0, "all twelve were picked together and each finished its prompt in its one slice");
    let log = r.log.borrow().clone();
    // the six prompts that use the cache got a slice of their own, in the order they were picked (fewest rows first), each after its own
    // lookup (a restore comes right before its slice); the six that use none went together in one call at the end
    let want = vec![
        Ev::Cuts(130, 0, 1),
        Ev::Cuts(200, 0, 1),
        Ev::Cuts(260, 0, 1),
        Ev::Restore(592),
        Ev::Single(600, 592),
        Ev::Restore(592),
        Ev::Cuts(621, 592, 1),
        Ev::Restore(592),
        Ev::Cuts(634, 592, 1),
        Ev::Batch(vec![40, 55, 70, 90, 300, 640]),
    ];
    assert_eq!(log, want, "where each prompt's slice went");
    assert_eq!(r.violations.borrow().len(), 0, "{:?}", r.violations.borrow());
    r.e.run().unwrap();
    let outs = outputs(&mut r.e);
    for (c, id) in cases.iter().zip(&ids) {
        let o = &outs[id];
        assert_eq!(o.cached_tokens, expected_cached(c), "{:?} of {} tokens", c.kind, c.r.prompt.len());
        same_reply(o, &reference(&c.r, None)).unwrap_or_else(|m| panic!("{:?} of {} tokens: {m}", c.kind, c.r.prompt.len()));
    }
    if trees {
        let branched: u64 = r.e.spec_stats().trees.iter().map(|t| t.rank_accepted[1..].iter().sum::<u64>()).sum();
        assert!(tree_rounds(&r.e) > 20, "trees must have run after the burst ({} rounds)", tree_rounds(&r.e));
        assert!(branched > 0, "a sibling branch was accepted and committed through the path commit");
    } else {
        assert_eq!(tree_rounds(&r.e), 0);
    }
    // the books: lookups by the 7 opted-in prompts above 128 tokens (the warm one included), 3 hits, an entry for each prompt that took cuts
    let s = r.e.prefix_stats();
    assert_eq!((s.lookups, s.hits, s.tokens_saved), (1 + 6, 3, 3 * 592), "{s:?}");
    assert_eq!(s.entries, 1 + 5, "the warm entry and one end cut for each of the five prompts that took one: {s:?}");
    assert_eq!(s.captures_discarded, 0);
    no_leaks(&r).unwrap();
    all_slots_back(&r);
    // a snapshot a batched step took survived the burst: the cold prompts' end cuts serve their next turns, the others' do not exist
    for (len, want) in [(200usize, 192usize), (260, 252), (130, 122), (40, 0), (90, 0), (300, 0)] {
        let c = cases.iter().find(|c| c.r.prompt.len() == len).unwrap();
        let again = Request { sampling: Sampling { seed: 5, ..c.r.sampling }, ..c.r.clone() };
        let id = r.e.submit(again.clone()).unwrap();
        r.e.run().unwrap();
        let o = outputs(&mut r.e).remove(&id).unwrap();
        same_reply(&o, &reference(&again, None)).unwrap();
        assert_eq!(o.cached_tokens, want, "the prompt of {len} tokens, asked again");
    }
    no_leaks(&r).unwrap();
}

#[test]
fn the_same_mixed_burst_is_exact_under_every_policy_batching_and_slicing() {
    let cases = mixed();
    for slice in [None, Some(64), Some(100)] {
        let refs: Vec<Output> = cases.iter().map(|c| reference(&c.r, slice)).collect();
        for (policy, batching, trees) in [
            (PrefillPolicy::Serial, true, false),
            (PrefillPolicy::Burst { div: 1, cap: 32 }, true, false),
            (PrefillPolicy::Burst { div: 1, cap: 32 }, false, false),
            (PrefillPolicy::Burst { div: 2, cap: 3 }, true, false),
            (PrefillPolicy::BATCHED_SHORT, true, false),
            // with draft trees: the serial policy, the burst policy as served, and one that batches little
            (PrefillPolicy::Serial, true, true),
            (PrefillPolicy::BATCHED_SHORT, true, true),
            (PrefillPolicy::Burst { div: 2, cap: 3 }, false, true),
        ] {
            let mut r = warmed(Setup { slice, policy, batching, trees, ..Default::default() });
            let ids: Vec<u64> = cases.iter().map(|c| r.e.submit_deferred(c.r.clone()).unwrap()).collect();
            r.e.run().unwrap();
            let outs = outputs(&mut r.e);
            for ((c, id), want) in cases.iter().zip(&ids).zip(&refs) {
                let o = &outs[id];
                same_reply(o, want).unwrap_or_else(|m| {
                    panic!(
                        "{:?} of {} tokens, {policy:?} batching {batching} trees {trees} slice {slice:?}: {m}",
                        c.kind,
                        c.r.prompt.len()
                    )
                });
                if c.kind == Kind::OptedOut || c.kind == Kind::Short {
                    assert_eq!(o.cached_tokens, 0);
                } else if matches!(c.kind, Kind::HitAndCut | Kind::HitOnly) {
                    assert!(o.cached_tokens > 0, "a sharer hits the warm entry under {policy:?} slice {slice:?}");
                }
            }
            no_leaks(&r).unwrap_or_else(|m| panic!("{policy:?} batching {batching} trees {trees} slice {slice:?}: {m}"));
            all_slots_back(&r);
            assert_eq!(r.violations.borrow().len(), 0);
            assert_eq!(tree_rounds(&r.e) > 0, trees, "{policy:?} batching {batching} slice {slice:?}: tree rounds");
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------
// planning right before each slice: sharers in one burst find what an earlier prompt of the same step left
// ------------------------------------------------------------------------------------------------------------------------------------

fn sharers_burst(setup: Setup) -> Vec<usize> {
    // eight requests that share a 600-token system prompt and differ after it (lengths 620, 621, ...: the order they are picked is the order
    // they arrive), cold cache
    let sys = toks(21, 600);
    let reqs: Vec<Request> = (0..8u64).map(|i| request(extended(&sys, 30 + i, 20 + i as usize), 40 + i)).collect();
    let mut r = rig(setup);
    let ids: Vec<u64> = reqs.iter().map(|q| r.e.submit_deferred(q.clone()).unwrap()).collect();
    r.e.run().unwrap();
    let outs = outputs(&mut r.e);
    for (q, id) in reqs.iter().zip(&ids) {
        same_reply(&outs[id], &reference(q, None)).unwrap();
    }
    no_leaks(&r).unwrap();
    ids.iter().map(|id| outs[id].cached_tokens).collect()
}

#[test]
fn a_prompt_behind_another_in_the_same_burst_step_finds_the_snapshot_it_left() {
    // the first request misses (cold), the second misses and discovers the shared blocks (it snapshots at 512), from the third on every one hits
    // 512: in the same step as the ones that left them. One slice a step (serial) gives the same hits: the burst policy changes when a first token
    // arrives, not what the cache does for a request
    let burst = sharers_burst(Setup::default());
    assert_eq!(burst, vec![0, 0, 512, 512, 512, 512, 512, 512]);
    let serial = sharers_burst(Setup { policy: PrefillPolicy::Serial, ..Default::default() });
    assert_eq!(serial, burst);
    let unbatched = sharers_burst(Setup { batching: false, ..Default::default() });
    assert_eq!(unbatched, burst);
}

// ------------------------------------------------------------------------------------------------------------------------------------
// tree rounds and burst prefill in the same steps
// ------------------------------------------------------------------------------------------------------------------------------------

/// Three sequences decode (tree rounds, then a burst of prompts arrives whose slices run in slices of 64 tokens, so for many steps one step is a
/// tree round for the runners followed by a burst step: the batched forward and the cache's restores and snapshots run between two tree rounds
/// of sequences whose committed state they must not touch, and a sharer that restored from a hit joins the running set while trees are planned.
#[test]
fn tree_rounds_and_burst_prefill_share_steps_and_every_reply_stays_exact() {
    let mut r = warmed(Setup { slice: Some(64), trees: true, ..Default::default() });
    let runners: Vec<Request> =
        (0..3u64).map(|i| Request { max_new: 160, ..request(toks(70 + i, 150 + 20 * i as usize), 300 + i) }).collect();
    let runner_ids: Vec<u64> = runners.iter().map(|q| r.e.submit_deferred(q.clone()).unwrap()).collect();
    let mut steps = 0;
    while r.e.active() < 3 {
        r.e.step().unwrap();
        steps += 1;
        assert!(steps < 200, "the runners never started");
    }
    let cases = mixed();
    let burst_ids: Vec<u64> = cases.iter().map(|c| r.e.submit_deferred(c.r.clone()).unwrap()).collect();
    let before = tree_rounds(&r.e);
    let (mut shared, mut shared_with_trees, mut widest_prefill) = (0usize, 0usize, 0usize);
    while r.e.busy() {
        let both = r.e.active() > 0 && r.e.prefilling() > 0;
        widest_prefill = widest_prefill.max(r.e.prefilling());
        let rounds = tree_rounds(&r.e);
        r.e.step().unwrap();
        if both {
            shared += 1;
            shared_with_trees += usize::from(tree_rounds(&r.e) > rounds);
        }
        steps += 1;
        assert!(steps < 5000);
    }
    assert!(shared >= 8, "the runners and the burst shared only {shared} steps");
    assert!(shared_with_trees >= 8, "only {shared_with_trees} of the {shared} shared steps ran a tree round");
    assert!(widest_prefill >= 10, "the burst was never waiting together ({widest_prefill})");
    assert!(tree_rounds(&r.e) - before > 30, "{} tree rounds after the burst arrived", tree_rounds(&r.e) - before);
    let outs = outputs(&mut r.e);
    for (q, id) in runners.iter().zip(&runner_ids) {
        same_reply(&outs[id], &reference(q, Some(64))).unwrap_or_else(|m| panic!("a runner of {} tokens: {m}", q.prompt.len()));
    }
    for (c, id) in cases.iter().zip(&burst_ids) {
        let o = &outs[id];
        same_reply(o, &reference(&c.r, Some(64))).unwrap_or_else(|m| panic!("{:?} of {} tokens: {m}", c.kind, c.r.prompt.len()));
        if matches!(c.kind, Kind::HitAndCut | Kind::HitOnly) {
            assert!(o.cached_tokens > 0, "a sharer of the warm prompt hits it ({} tokens)", c.r.prompt.len());
        }
    }
    assert!(r.e.prefix_stats().hits >= 3, "{:?}", r.e.prefix_stats());
    assert_eq!(r.violations.borrow().len(), 0);
    no_leaks(&r).unwrap();
    all_slots_back(&r);
}

// ------------------------------------------------------------------------------------------------------------------------------------
// opt-outs
// ------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn a_request_that_opts_out_neither_looks_up_nor_leaves_anything_in_a_burst() {
    let w = warm_prompt();
    let mut r = warmed(Setup::default());
    let before = r.e.prefix_stats();
    // five sharers of the warm entry that all opted out, and two cold long prompts that did, in one burst
    let outed: Vec<Request> = (0..5u64).map(|i| no_cache(request(extended(&w, 200 + i, 25 + i as usize), 60 + i))).collect();
    let cold: Vec<Request> = vec![request(toks(231, 233), 70), request(toks(232, 280), 71)];
    let ids: Vec<u64> = outed.iter().chain(&cold).map(|q| r.e.submit_deferred(q.clone()).unwrap()).collect();
    r.e.step().unwrap();
    // the opted-out ones are batched (no cache plays a part), the cold ones take their own slice (an end cut to take)
    let log = r.log.borrow().clone();
    assert_eq!(
        log,
        vec![Ev::Cuts(233, 0, 1), Ev::Cuts(280, 0, 1), Ev::Batch(vec![625, 626, 627, 628, 629])],
        "an opted-out prompt that would hit is not restored and is batched"
    );
    r.e.run().unwrap();
    let outs = outputs(&mut r.e);
    for (q, id) in outed.iter().chain(&cold).zip(&ids) {
        assert_eq!(outs[id].cached_tokens, 0, "{} tokens", q.prompt.len());
        same_reply(&outs[id], &reference(q, None)).unwrap();
    }
    let after = r.e.prefix_stats();
    assert_eq!(after.lookups - before.lookups, 2, "only the two cold prompts looked up");
    assert_eq!((after.hits, after.entries), (before.hits, before.entries + 2));
    no_leaks(&r).unwrap();
}

#[test]
fn an_engine_without_a_cache_bursts_as_before_and_never_plans() {
    // a family whose `prefix_cache()` is false builds its engine without one, whatever its backend could do: no lookup, no cut, no restore
    let w = warm_prompt();
    let mut r = rig(Setup { cache: false, ..Default::default() });
    let reqs: Vec<Request> = (0..6u64).map(|i| request(extended(&w, 300 + i, 20 + i as usize), 80 + i)).collect();
    let ids: Vec<u64> = reqs.iter().map(|q| r.e.submit_deferred(q.clone()).unwrap()).collect();
    r.e.step().unwrap();
    let log = r.log.borrow().clone();
    assert_eq!(log, vec![Ev::Batch(vec![620, 621, 622, 623, 624, 625])], "no prompt uses a cache: one batch call");
    r.e.run().unwrap();
    let outs = outputs(&mut r.e);
    for (q, id) in reqs.iter().zip(&ids) {
        assert_eq!(outs[id].cached_tokens, 0);
        same_reply(&outs[id], &reference(q, None)).unwrap();
    }
    let s = r.e.prefix_stats();
    assert_eq!((s.enabled, s.lookups, s.entries), (false, 0, 0));
    assert_eq!(r.e.backend.inner.snapshots_in_use(), 0);
}

// ------------------------------------------------------------------------------------------------------------------------------------
// failure isolation, cancel, a fatal error
// ------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn a_prompt_that_fails_in_a_burst_fails_alone_and_gives_its_snapshots_and_slot_back() {
    // in one burst: a cold prompt with a cut plan (260; its end cut is already taken inside its failing slice), a sharer that restored from a hit
    // (634) and a short one in the batch (55) cannot be served
    let cases = mixed();
    let mut r = warmed(Setup { poison: vec![260, 634, 55], ..Default::default() });
    let ids: Vec<u64> = cases.iter().map(|c| r.e.submit_deferred(c.r.clone()).unwrap()).collect();
    r.e.run().unwrap();
    let mut failed = r.e.take_failed();
    failed.sort();
    let named: Vec<u64> =
        cases.iter().zip(&ids).filter(|(c, _)| [260, 634, 55].contains(&c.r.prompt.len())).map(|(_, &id)| id).collect();
    assert_eq!(failed.iter().map(|f| f.0).collect::<Vec<_>>(), {
        let mut n = named.clone();
        n.sort();
        n
    });
    assert!(failed.iter().all(|f| f.1.contains("injected failure")), "{failed:?}");
    let outs = outputs(&mut r.e);
    for (c, id) in cases.iter().zip(&ids) {
        if named.contains(id) {
            assert!(!outs.contains_key(id), "a failed prompt has no reply");
            continue;
        }
        same_reply(&outs[id], &reference(&c.r, None))
            .unwrap_or_else(|m| panic!("{:?} of {} tokens: {m}", c.kind, c.r.prompt.len()));
        assert_eq!(outs[id].cached_tokens, expected_cached(c));
    }
    let s = r.e.prefix_stats();
    assert!(s.captures_discarded >= 2, "the failed prompts' snapshots were given back unused: {s:?}");
    assert_eq!(s.entries, 1 + 3, "the warm entry and the end cuts of the three other prompts that took one (130, 200 and 621)");
    no_leaks(&r).unwrap();
    all_slots_back(&r);
}

#[test]
fn a_prompt_that_fails_after_an_earlier_slice_took_a_snapshot_gives_it_back_and_the_rest_finish() {
    // slices of 64: the shortest sharer of a 600-token system prompt (the one the burst picks first: it discovers the shared blocks) takes its snapshot at 512 in its eighth slice and then fails in its tenth
    // (from 576 on). The snapshot is held on the prompt when the failure comes: it must go back, with the slot, and the others finish exact. Without
    // and with draft trees (the failing prompt never reached a decode round; the others decode in tree rounds)
    for trees in [false, true] {
        let sys = toks(41, 600);
        let mut r = rig(Setup { slice: Some(64), late: Some((639, 576)), trees, ..Default::default() });
        let first = request(extended(&sys, 50, 40), 90);
        let id = r.e.submit(first).unwrap();
        r.e.run().unwrap();
        outputs(&mut r.e).remove(&id).unwrap();
        let reqs: Vec<Request> = (0..3u64).map(|i| request(extended(&sys, 60 + i, 41 - i as usize), 100 + i)).collect();
        assert_eq!(reqs.iter().map(|q| q.prompt.len()).collect::<Vec<_>>(), vec![641, 640, 639]);
        let ids: Vec<u64> = reqs.iter().map(|q| r.e.submit_deferred(q.clone()).unwrap()).collect();
        r.e.run().unwrap();
        let failed = r.e.take_failed();
        assert_eq!(failed.iter().map(|f| f.0).collect::<Vec<_>>(), vec![ids[2]], "trees {trees}: {failed:?}");
        assert!(failed[0].1.contains("injected late failure"), "{failed:?}");
        let outs = outputs(&mut r.e);
        assert!(!outs.contains_key(&ids[2]), "a failed prompt has no reply");
        for (q, id) in reqs.iter().zip(&ids).take(2) {
            same_reply(&outs[id], &reference(q, Some(64))).unwrap();
        }
        let s = r.e.prefix_stats();
        assert!(s.captures_discarded >= 1, "trees {trees}: the failed prompt's snapshot was given back unused: {s:?}");
        no_leaks(&r).unwrap_or_else(|m| panic!("trees {trees}: {m}"));
        all_slots_back(&r);
    }
}

#[test]
fn prompts_cancelled_in_a_burst_give_back_their_snapshots_and_the_rest_finish() {
    // slices of 64: a sharer's snapshot at 512 is taken eight slices in, while the prompt still has its tail to prefill. Cancel them there.
    let sys = toks(41, 600);
    let mut r = rig(Setup { slice: Some(64), ..Default::default() });
    let first = request(extended(&sys, 50, 40), 90);
    let id = r.e.submit(first).unwrap();
    r.e.run().unwrap();
    outputs(&mut r.e).remove(&id).unwrap();
    let reqs: Vec<Request> = (0..4u64).map(|i| request(extended(&sys, 60 + i, 41 + i as usize), 100 + i)).collect();
    let ids: Vec<u64> = reqs.iter().map(|q| r.e.submit_deferred(q.clone()).unwrap()).collect();
    let mut steps = 0;
    while r.e.backend.inner.snapshots_in_use() <= r.e.prefix_stats().entries as usize {
        r.e.step().unwrap();
        steps += 1;
        assert!(steps < 100, "no prompt took its snapshot at 512");
    }
    assert!(r.e.prefilling() > 0, "the prompts are still being prefilled with a snapshot held");
    // cancel the first two; one is also cancelled before it ever started, a third time is refused
    assert!(r.e.cancel(ids[0]) && r.e.cancel(ids[1]));
    assert!(!r.e.cancel(ids[0]));
    r.e.run().unwrap();
    let outs = outputs(&mut r.e);
    assert_eq!(outs.values().filter(|o| o.finish == Some(FinishReason::Cancelled)).count(), 2);
    for (q, id) in reqs.iter().zip(&ids).skip(2) {
        same_reply(&outs[id], &reference(q, Some(64))).unwrap();
    }
    assert!(r.e.prefix_stats().captures_discarded >= 1, "{:?}", r.e.prefix_stats());
    no_leaks(&r).unwrap();
    all_slots_back(&r);
}

#[test]
fn a_fatal_error_in_a_burst_keeps_every_unserved_prompt_queued_and_abort_gives_everything_back() {
    let cases = mixed();
    let mut r = warmed(Setup { fatal: Some(260), ..Default::default() });
    for c in &cases {
        r.e.submit_deferred(c.r.clone()).unwrap();
    }
    let err = r.e.step().unwrap_err();
    assert!(matches!(err, StepError::Fatal(_)), "{err}");
    // 130 and 200 were served before it (their first tokens are out); the other ten are back in the queue, none failed or lost
    assert_eq!((r.e.active(), r.e.prefilling()), (2, 10));
    assert!(r.e.take_failed().is_empty());
    assert_eq!(r.e.abort_all(), 12);
    // the books: the warm entry and the two end cuts that were committed; nothing the aborted prompts held
    let s = r.e.prefix_stats();
    assert_eq!(s.entries, 1 + 2);
    no_leaks(&r).unwrap();
    all_slots_back(&r);
}

// ------------------------------------------------------------------------------------------------------------------------------------
// the copy branch and the prefix cache
// ------------------------------------------------------------------------------------------------------------------------------------

/// A hit restores the backend's state for the cached prefix, but the engine's copy index is built from the stream's own token list, which holds the
/// WHOLE prompt (`Seq::tokens` is the request's prompt, not the part the cache did not cover): the first tree round after a hit copies from text the
/// cache served. The warm prompt (600 tokens of a 24-token block over and over; the mock continues the block by its own bigram copy) is asked again:
/// 592 tokens come from the cache, 8 are prefilled, and the first decode round already carries a copy branch whose match lies in the cached part (the
/// 8 prefilled tokens, the first sample and the draft's 7 picks are 16 tokens, too few to hold an earlier occurrence of their last 4 at the block's
/// distance of 24). Every reply equals its uncached serial reply, a hit followed by copy rounds
/// leaves nothing owed or leaked.
#[test]
fn the_copy_branch_reads_the_whole_prompt_after_a_cache_hit_and_every_reply_is_exact() {
    let block: Vec<u32> = (0..24u32).map(|j| (j * 11 + 5) % VOCAB as u32).collect();
    let prompt: Vec<u32> = block.iter().cycle().take(600).copied().collect();
    let mut r = rig(Setup { trees: true, copy: 4, ..Default::default() });
    let warm = Request { max_new: 30, ..request(prompt.clone(), 3) };
    let id = r.e.submit(warm.clone()).unwrap();
    r.e.run().unwrap();
    let o = outputs(&mut r.e).remove(&id).unwrap();
    assert_eq!(o.cached_tokens, 0, "a cold cache");
    same_reply(&o, &reference(&warm, None)).unwrap();
    assert_eq!(r.e.prefix_stats().entries, 1, "the warm prompt left its end cut");
    let (trees_before, (copies_before, _)) = (tree_rounds(&r.e), copy_rounds(&r.e));
    // the same prompt again, greedy and sampled: restored from the end cut at 592
    for seed in [4u64, 5] {
        let again = Request { max_new: 40, ..request(prompt.clone(), seed) };
        let id = r.e.submit_deferred(again.clone()).unwrap();
        let (rounds, (copies, _)) = (tree_rounds(&r.e), copy_rounds(&r.e));
        let mut steps = 0;
        while tree_rounds(&r.e) == rounds {
            r.e.step().unwrap();
            steps += 1;
            assert!(steps < 50, "no tree round ran");
        }
        assert!(
            copy_rounds(&r.e).0 > copies,
            "seed {seed}: the first tree round after a hit carried no copy branch: the index does not see the cached part of the prompt"
        );
        r.e.run().unwrap();
        let o = outputs(&mut r.e).remove(&id).unwrap();
        assert_eq!(o.cached_tokens, 592, "seed {seed}: restored from the end cut");
        same_reply(&o, &reference(&again, None)).unwrap_or_else(|m| panic!("seed {seed}: {m}"));
    }
    // (whether a copied token lands depends on the mock draft's 7 picks being right first; acceptance of copied tokens is the tree_rounds.rs test's)
    let (trees_after, (copies_after, _)) = (tree_rounds(&r.e), copy_rounds(&r.e));
    assert!(trees_after > trees_before && copies_after > copies_before, "{trees_after} tree rounds, {copies_after} with copies");
    no_leaks(&r).unwrap();
    all_slots_back(&r);
}

// ------------------------------------------------------------------------------------------------------------------------------------
// fairness
// ------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn a_long_prompt_that_uses_the_cache_is_slowed_among_short_arrivals_but_never_starved() {
    // a 200-token prompt with a cut plan, in slices of 4 (50 slices), while a new 5-token prompt arrives every step: as without a cache, the
    // pass-over guard serves it at least once in every PREFILL_PASS_GUARD + 1 slices, and its snapshot is committed when it finishes
    let mut r = rig(Setup { slice: Some(4), policy: PrefillPolicy::BATCHED_SHORT, ..Default::default() });
    let long = request(toks(51, 200), 110);
    let long_id = r.e.submit_deferred(long.clone()).unwrap();
    let mut first = None;
    let mut next = 0u64;
    for step in 1..=700 {
        if step <= 600 {
            r.e.submit_deferred(Request { max_new: 2, ..request(toks(1000 + next, 5), 120 + next) }).unwrap();
            next += 1;
        }
        r.e.step().unwrap();
        for ev in r.e.take_events() {
            if ev.id == long_id && !ev.tokens.is_empty() {
                first.get_or_insert(step);
            }
        }
        if first.is_some() {
            break;
        }
    }
    let at = first.expect("the long prompt never got its first token");
    assert!(at < 600, "the long prompt waited until the arrivals stopped (step {at}): it was starved");
    assert!(at <= 50 * (PREFILL_PASS_GUARD as usize + 1), "step {at}");
    r.e.run().unwrap();
    let outs = outputs(&mut r.e);
    same_reply(&outs[&long_id], &reference(&long, Some(4))).unwrap();
    assert_eq!(r.e.prefix_stats().entries, 1, "the long prompt left its end cut");
    no_leaks(&r).unwrap();
    all_slots_back(&r);
}
