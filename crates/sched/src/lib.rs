//! Scheduling for one GB10: memory admission that fails closed, a paged block pool with refcounts,
//! the prefix index (`prefix`: hybrid models keep recurrent state that cannot be sliced per token, so a cache entry is a state
//! snapshot plus references to KV pages), and the global verify-row budget shared by all streams.
//!
//! Nothing here touches the GPU. Byte counts come from the model config and from measurement
//! at startup; the step-time table comes from timing real forwards on the box.
#![forbid(unsafe_code)]

pub mod memory;
pub mod prefix;
pub use memory::{KvLayerClass, KvLayout, KvPlane, MemoryError, MemoryModel, MemoryProfile, StepShape, VisionMemory};

// ------------------------------------------------------------------ memory

/// Per-model memory costs. Fill from config.json and measurement, never from guesses; build it with
/// [`ModelMemory::from_profile`] so the per-token and per-slot costs come from a validated family memory model.
/// [`Admission::new`] refuses a zero KV cost whatever built it.
#[derive(Clone, Debug)]
pub struct ModelMemory {
    /// Target + draft weights as loaded.
    pub weights_bytes: u64,
    /// KV bytes per token over every layer that caches context (attention planes, latents, indexer planes). Never zero.
    pub kv_bytes_per_token: u64,
    /// Recurrent (linear-attention) state per sequence slot, including speculative history rows.
    pub state_bytes_per_slot: u64,
    /// Workspace for the widest step: verify rows, a prefill chunk, a vision encode.
    pub activation_reserve: u64,
}

impl ModelMemory {
    /// The costs a validated family profile states. `weights_bytes` and `activation_reserve` are what the caller still has to
    /// hand to admission itself (a planner that already took them out of the capacity passes zero for both).
    pub fn from_profile(profile: &MemoryProfile, weights_bytes: u64, activation_reserve: u64) -> Self {
        Self {
            weights_bytes,
            kv_bytes_per_token: profile.kv_bytes_per_token(),
            state_bytes_per_slot: profile.state_bytes_per_slot(),
            activation_reserve,
        }
    }

    /// A model that costs no KV per token would be admitted at any context length: refuse it.
    pub fn validate(&self) -> Result<(), MemoryError> {
        if self.kv_bytes_per_token == 0 {
            return Err(MemoryError::Zero("KV bytes per token (admission would hand out unlimited context)".to_string()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmitError {
    /// Not enough memory above the floor; carries (needed, available) bytes.
    NoMemory {
        needed: u64,
        available: u64,
    },
    NoSlot,
    /// The request's byte cost does not fit in 64 bits: refused, never wrapped around.
    TooLarge,
    /// prompt + max new tokens exceed the most one request may hold: the configured context limit, or the KV pool the memory plan
    /// allocated ([`Admission::with_kv_tokens`]), whichever is smaller. `max` is that smaller figure.
    ContextTooLong {
        tokens: u64,
        max: u64,
    },
    /// The request's cost exceeds everything this server could ever give one sequence (the whole admission capacity), so waiting
    /// cannot help. Only [`Admission::check`] reports it: [`Admission::admit`] says `NoMemory` for any shortfall.
    NeverFits {
        needed: u64,
        capacity: u64,
    },
}

/// Fail-closed admission. Every sequence reserves its whole budget (state slot + KV for
/// prompt + max_tokens) before it runs; there is no preemption, so an admitted sequence can
/// always finish. A unified-memory OOM on GB10 can hang the host, so refusing early is the point.
#[derive(Debug)]
pub struct Admission {
    capacity: u64,
    used: u64,
    /// Bytes held outside any sequence (prefix-cache blocks, state snapshots): charged against capacity.
    pinned: u64,
    max_context: Option<u64>,
    /// The tokens the KV pool holds for all sequences together (the plan's `kv_tokens`), when the caller declared it.
    kv_tokens: Option<u64>,
    /// Tokens reserved by the running sequences (the sum of their `prompt + max_new`).
    kv_in_use: u64,
    model: ModelMemory,
    /// Per slot: the bytes and the tokens the sequence in it reserved.
    slots: Vec<Option<(u64, u64)>>,
}

impl Admission {
    /// `allocatable_bytes`: what the device can actually allocate at startup (measure: min of
    /// cudaMemGetInfo free and MemAvailable). `floor_bytes`: kept free for the OS and driver.
    ///
    /// Fails closed: a model whose KV cost per token is zero is refused (see [`ModelMemory::validate`]), so an `Admission`
    /// always charges every token of every sequence.
    pub fn new(allocatable_bytes: u64, floor_bytes: u64, model: ModelMemory, max_slots: usize) -> Result<Self, MemoryError> {
        model.validate()?;
        let fixed = floor_bytes.saturating_add(model.weights_bytes).saturating_add(model.activation_reserve);
        Ok(Self {
            capacity: allocatable_bytes.saturating_sub(fixed),
            used: 0,
            pinned: 0,
            max_context: None,
            kv_tokens: None,
            kv_in_use: 0,
            model,
            slots: vec![None; max_slots],
        })
    }
    /// Refuse any sequence whose prompt + max new tokens exceed `tokens`.
    pub fn with_max_context(mut self, tokens: u64) -> Self {
        self.max_context = Some(tokens);
        self
    }
    /// Declare the KV pool: it holds `tokens` for all sequences together (a memory plan's `kv_tokens`). The byte capacity prices the slot
    /// state of every sequence, but the state of ALL slots is allocated up front, so with fewer sequences running the bytes of the idle
    /// slots would read as room for KV that the pool does not have (a 197,048-token prompt into a 194,048-token pool: admitted, prefilled for
    /// five minutes, `OutOfPages`). With the pool declared, a request above it is refused at once as [`AdmitError::ContextTooLong`] (`max` the
    /// pool), and a request that fits alone but not beside the ones running gets [`AdmitError::NoMemory`], which the serving loop queues:
    /// the sum of the reserved tokens never exceeds the pool, so a running sequence always finds its pages. The pool is in tokens, not
    /// bytes, because the planner's guarantee is in tokens (`sum(ceil(T / page)) <= tokens / page + slots`).
    pub fn with_kv_tokens(mut self, tokens: u64) -> Self {
        self.kv_tokens = Some(tokens);
        self
    }
    /// The most tokens (prompt + generation) one request may hold, whatever else is running: the smaller of the context limit and the KV
    /// pool, `None` when neither was declared. What a client needs to know to decide whether a prompt fits.
    pub fn request_limit(&self) -> Option<u64> {
        match (self.max_context, self.kv_tokens) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }
    /// The KV pool in tokens, when declared ([`Admission::with_kv_tokens`]).
    pub fn kv_tokens(&self) -> Option<u64> {
        self.kv_tokens
    }
    /// Tokens the running sequences have reserved.
    pub fn kv_tokens_in_use(&self) -> u64 {
        self.kv_in_use
    }
    /// Each reserving slot and the tokens its sequence reserved (prompt + max new tokens).
    pub fn reserved_tokens(&self) -> impl Iterator<Item = (usize, u64)> + '_ {
        self.slots.iter().enumerate().filter_map(|(slot, s)| s.map(|(_, tokens)| (slot, tokens)))
    }
    pub fn capacity(&self) -> u64 {
        self.capacity
    }
    pub fn available(&self) -> u64 {
        self.capacity.saturating_sub(self.used).saturating_sub(self.pinned)
    }
    /// Byte cost of a sequence of `tokens`, or None when it overflows 64 bits.
    pub fn checked_cost(&self, tokens: u64) -> Option<u64> {
        tokens.checked_mul(self.model.kv_bytes_per_token)?.checked_add(self.model.state_bytes_per_slot)
    }
    /// Like `checked_cost`, saturating at `u64::MAX` (which no capacity can satisfy).
    pub fn cost(&self, tokens: u64) -> u64 {
        self.checked_cost(tokens).unwrap_or(u64::MAX)
    }
    /// Why a sequence of `tokens` can never be admitted, whatever is running: past the context limit, a cost that overflows, or a
    /// cost above the whole capacity. `Ok` says nothing about the present (see [`Admission::can_admit`]). Pure.
    pub fn check(&self, tokens: u64) -> Result<(), AdmitError> {
        if let Some(max) = self.request_limit() {
            if tokens > max {
                return Err(AdmitError::ContextTooLong { tokens, max });
            }
        }
        let need = self.checked_cost(tokens).ok_or(AdmitError::TooLarge)?;
        if need > self.capacity {
            return Err(AdmitError::NeverFits { needed: need, capacity: self.capacity });
        }
        Ok(())
    }
    /// What [`Admission::admit`] would answer for `tokens` right now, without reserving anything.
    pub fn can_admit(&self, tokens: u64) -> Result<u64, AdmitError> {
        if let Some(max) = self.request_limit() {
            if tokens > max {
                return Err(AdmitError::ContextTooLong { tokens, max });
            }
        }
        let need = self.checked_cost(tokens).ok_or(AdmitError::TooLarge)?;
        if need > self.available() {
            return Err(AdmitError::NoMemory { needed: need, available: self.available() });
        }
        if let Some(pool) = self.kv_tokens {
            // the request is within the pool alone (above); what the running sequences hold may leave less than it needs. The figures are
            // the KV bytes of the tokens asked for and of the tokens left in the pool
            let left = pool.saturating_sub(self.kv_in_use);
            if tokens > left {
                let per = self.model.kv_bytes_per_token;
                return Err(AdmitError::NoMemory { needed: tokens.saturating_mul(per), available: left.saturating_mul(per) });
            }
        }
        self.slots.iter().position(|s| s.is_none()).ok_or(AdmitError::NoSlot)?;
        Ok(need)
    }
    /// Reserve a slot for a sequence of up to `tokens` (prompt + max new tokens).
    pub fn admit(&mut self, tokens: u64) -> Result<usize, AdmitError> {
        let need = self.can_admit(tokens)?;
        let slot = self.slots.iter().position(|s| s.is_none()).ok_or(AdmitError::NoSlot)?;
        self.slots[slot] = Some((need, tokens));
        self.used += need; // need <= available <= capacity, so this cannot overflow
        self.kv_in_use = self.kv_in_use.saturating_add(tokens); // tokens <= the pool's tokens when it is declared
        Ok(slot)
    }
    pub fn release(&mut self, slot: usize) {
        if let Some((bytes, tokens)) = self.slots.get_mut(slot).and_then(|s| s.take()) {
            self.used = self.used.saturating_sub(bytes);
            self.kv_in_use = self.kv_in_use.saturating_sub(tokens);
        }
    }
    /// Charge `bytes` held outside any sequence (a cached prefix, a state snapshot) against capacity.
    /// Fails closed like `admit`; pair every success with an `unpin` when the memory is freed.
    pub fn pin(&mut self, bytes: u64) -> Result<(), AdmitError> {
        if bytes > self.available() {
            return Err(AdmitError::NoMemory { needed: bytes, available: self.available() });
        }
        self.pinned += bytes;
        Ok(())
    }
    pub fn unpin(&mut self, bytes: u64) {
        self.pinned = self.pinned.saturating_sub(bytes);
    }
    pub fn pinned(&self) -> u64 {
        self.pinned
    }
    /// How many sequences of `tokens` each would fit right now.
    pub fn fits(&self, tokens: u64) -> u64 {
        self.available() / self.cost(tokens).max(1)
    }
}

// ------------------------------------------------------------------ paged blocks

pub type BlockId = u32;

/// Fixed pool of KV blocks with reference counts (shared prefixes hold one ref per user).
#[derive(Debug)]
pub struct BlockPool {
    free: Vec<BlockId>,
    refs: Vec<u32>,
    pub block_tokens: usize,
}

impl BlockPool {
    pub fn new(blocks: usize, block_tokens: usize) -> Self {
        Self { free: (0..blocks as BlockId).rev().collect(), refs: vec![0; blocks], block_tokens }
    }
    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }
    pub fn blocks_for(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.block_tokens)
    }
    /// All or nothing.
    pub fn alloc(&mut self, n: usize) -> Option<Vec<BlockId>> {
        if n > self.free.len() {
            return None;
        }
        let v: Vec<BlockId> = (0..n).map(|_| self.free.pop().unwrap()).collect();
        for &b in &v {
            self.refs[b as usize] = 1;
        }
        Some(v)
    }
    pub fn retain(&mut self, blocks: &[BlockId]) {
        for &b in blocks {
            assert!(self.refs[b as usize] > 0, "retain of a free block {b}");
            self.refs[b as usize] += 1;
        }
    }
    pub fn release(&mut self, blocks: &[BlockId]) {
        for &b in blocks {
            let r = &mut self.refs[b as usize];
            assert!(*r > 0, "double free of block {b}");
            *r -= 1;
            if *r == 0 {
                self.free.push(b);
            }
        }
    }
    pub fn refcount(&self, b: BlockId) -> u32 {
        self.refs[b as usize]
    }
}

// ------------------------------------------------------------------ prefix cache

/// A cache key token: text ids and image content share one stream so an image changes the key.
/// Image patches enter as `IMAGE_BASE | (hash of the image bytes and patch index)`.
pub type CacheToken = u64;
pub const IMAGE_BASE: u64 = 1 << 63;

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// A hash that names an image to the cache: the pixel bytes' hash plus everything that changes what the
/// model sees (decoded size, resize and normalisation parameters). Two images that differ in any of these
/// must never share cached state.
pub fn image_content_hash(pixel_hash: u64, width: u32, height: u32, preprocessing: u64) -> u64 {
    let h = mix64(pixel_hash ^ 0xA076_1D64_78BD_642F);
    let h = mix64(h ^ ((width as u64) << 32 | height as u64));
    mix64(h ^ preprocessing.wrapping_mul(0xE703_7ED1_A0B4_28DB))
}

/// One cache-key token for patch `patch` of an image. The hash is mixed first, so images whose hashes differ in
/// any single bit get unrelated tokens (the old `>> 1` made bit-0 twins collide). 63 bits remain, so two
/// different (image, patch) pairs collide with probability about 2^-63; lookups still compare full token lists.
pub fn image_token(image_hash: u64, patch: u32) -> CacheToken {
    IMAGE_BASE | (mix64(image_hash ^ (patch as u64).wrapping_mul(0x9E3779B97F4A7C15)) & !IMAGE_BASE)
}

/// The 32-bit token of patch `patch` of an image in the prefix index's key stream (`prefix::PrefixIndex` compares `u32` tokens): the high bit is
/// set, so it can never equal a text token id (those are far below 2^31), and the other 31 bits come from [`image_token`]. Two different images
/// get the same run of these tokens only if their content hashes agree; the index also compares the tokens themselves, so a collision of a few
/// patches cannot make two images share state (a cut is never taken inside an image).
pub fn image_key_token(image_hash: u64, patch: u32) -> u32 {
    0x8000_0000 | (image_token(image_hash, patch) as u32 & 0x7FFF_FFFF)
}

fn chain(h: u64, t: CacheToken) -> u64 {
    let mut x = h ^ t.wrapping_mul(0xD1B54A32D192ED03);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^ (x >> 32)
}

const HASH_SEED: u64 = 0x6A09E667F3BCC908;

/// Prefix hashes: out[i] identifies tokens[..i+1].
pub fn prefix_hashes(tokens: &[CacheToken]) -> Vec<u64> {
    prefix_hashes_in(0, tokens)
}

/// Prefix hashes inside a namespace. Fold into `namespace` everything that makes cached state
/// non-interchangeable: the model and draft identity, adapter, KV dtype, rope scaling.
pub fn prefix_hashes_in(namespace: u64, tokens: &[CacheToken]) -> Vec<u64> {
    let mut h = if namespace == 0 { HASH_SEED } else { chain(HASH_SEED, mix64(namespace)) };
    tokens
        .iter()
        .map(|&t| {
            h = chain(h, t);
            h
        })
        .collect()
}

// ------------------------------------------------------------------ verify-row budget

/// Draft positions a [`Demand`] can describe.
pub const MAX_DRAFT_POSITIONS: usize = 16;

/// What one stream could use this step: how many drafts are on offer and what each is worth.
///
/// The value of the `j`-th draft is the probability that drafts `1..=j` are all accepted (a draft only counts when every
/// earlier one did), so it never rises with `j` and a greedy allocation by value is optimal. How that probability is built is
/// the caller's business: [`Demand::geometric`] assumes every position is as likely as the first, [`Demand::from_conditional`]
/// takes a measured conditional acceptance per position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Demand {
    /// Drafts on offer this step (drafter output length, already capped per stream).
    pub drafts: usize,
    /// `cum[j - 1]` = P(drafts 1..=j all accepted).
    cum: [f64; MAX_DRAFT_POSITIONS],
    /// Tokens of context this stream's KV holds (0 when unknown): what its attention reads each round.
    pub kv_tokens: u64,
}

impl Demand {
    /// Each draft accepted with probability `accept`, independently of position.
    pub fn geometric(accept: f64, drafts: usize) -> Self {
        Self::from_conditional(&[accept; MAX_DRAFT_POSITIONS], drafts)
    }

    /// `conditional[j - 1]` = P(draft `j` accepted | drafts `1..j` accepted). NaN and negative values count as 0, above 1 as 1;
    /// positions past the slice repeat its last value.
    pub fn from_conditional(conditional: &[f64], drafts: usize) -> Self {
        let mut cum = [0.0; MAX_DRAFT_POSITIONS];
        let mut p = 1.0;
        for (j, c) in cum.iter_mut().enumerate() {
            p *= sane(conditional.get(j).or(conditional.last()).copied().unwrap_or(0.0));
            *c = p;
        }
        Self { drafts: drafts.min(MAX_DRAFT_POSITIONS), cum, kv_tokens: 0 }
    }

    pub fn with_kv_tokens(mut self, kv_tokens: u64) -> Self {
        self.kv_tokens = kv_tokens;
        self
    }

    /// Value of the `j`-th draft (counted from 1): P(drafts 1..=j all accepted).
    pub fn marginal(&self, j: usize) -> f64 {
        self.cum[j.clamp(1, MAX_DRAFT_POSITIONS) - 1]
    }

    /// Expected committed tokens with `k` drafts verified: the sample past the accepted drafts, plus each draft's value.
    pub fn expected_tokens(&self, k: usize) -> f64 {
        1.0 + self.cum[..k.min(self.drafts)].iter().sum::<f64>()
    }
}

/// Expected committed tokens from one stream with `d` drafts under a geometric acceptance: 1 + a + a^2 + ... + a^d.
pub fn expected_tokens(accept: f64, d: usize) -> f64 {
    Demand::geometric(accept, d).expected_tokens(d)
}

/// Share `budget_rows` among streams. Every stream gets its base row (the token it must feed);
/// extra rows go one at a time to the stream whose next draft has the highest value (see [`Demand`]). Marginals fall along
/// each stream, so greedy is optimal here.
/// Returns drafts per stream; if the budget is below the stream count, the tail gets none and
/// the caller must not step those streams (see `rows_used`).
pub fn allocate(demands: &[Demand], budget_rows: usize) -> Vec<usize> {
    let mut order = allocation_order(demands, budget_rows.saturating_sub(demands.len()));
    let mut d = vec![0usize; demands.len()];
    for i in order.drain(..) {
        d[i] += 1;
    }
    d
}

/// An acceptance probability usable for ranking: NaN and negatives count as 0 (no drafts), above 1 as 1.
fn sane(a: f64) -> f64 {
    if a.is_nan() {
        0.0
    } else {
        a.clamp(0.0, 1.0)
    }
}

/// Measured step time as a function of total rows in the step (piecewise linear between points).
#[derive(Clone, Debug)]
pub struct CostModel {
    points: Vec<(usize, f64)>,
}

impl CostModel {
    /// `points`: (rows, milliseconds), measured on the target box. Must include rows = 1, hold each row
    /// count once, and carry finite positive times. Measurement noise that makes a wider step look faster
    /// than a narrower one is flattened (the cost never decreases with rows), so extrapolation cannot go
    /// negative and the budget cannot be lured to the widest step.
    pub fn try_from_table(mut points: Vec<(usize, f64)>) -> Result<Self, String> {
        points.sort_by_key(|p| p.0);
        if points.is_empty() || points[0].0 != 1 {
            return Err("the table must start at 1 row".into());
        }
        if let Some(w) = points.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(format!("row count {} appears twice", w[0].0));
        }
        if let Some(p) = points.iter().find(|p| !p.1.is_finite() || p.1 <= 0.0) {
            return Err(format!("step time {} ms at {} rows is not finite and positive", p.1, p.0));
        }
        let mut floor = 0.0f64;
        for p in points.iter_mut() {
            floor = floor.max(p.1);
            p.1 = floor;
        }
        Ok(Self { points })
    }
    /// `try_from_table` for tables written in code: panics on a malformed one.
    pub fn from_table(points: Vec<(usize, f64)>) -> Self {
        Self::try_from_table(points).unwrap_or_else(|e| panic!("bad cost table: {e}"))
    }
    pub fn ms(&self, rows: usize) -> f64 {
        let p = &self.points;
        if rows <= p[0].0 {
            return p[0].1;
        }
        for w in p.windows(2) {
            let ((r0, t0), (r1, t1)) = (w[0], w[1]);
            if rows <= r1 {
                return t0 + (t1 - t0) * (rows - r0) as f64 / (r1 - r0) as f64;
            }
        }
        // past the table: extend the last segment's slope
        let n = p.len();
        if n == 1 {
            return p[0].1;
        }
        let ((r0, t0), (r1, t1)) = (p[n - 2], p[n - 1]);
        t1 + (t1 - t0) / (r1 - r0) as f64 * (rows - r1) as f64
    }
    pub fn max_rows(&self) -> usize {
        self.points.last().unwrap().0
    }
}

/// The order in which `allocate` hands out extra rows as the budget grows: entry r is the stream that
/// receives the (r+1)-th draft row. Greedy allocations are nested, so every budget's allocation is a prefix.
fn allocation_order(demands: &[Demand], max_extra: usize) -> Vec<usize> {
    let mut d = vec![0usize; demands.len()];
    let mut next: Vec<f64> = demands.iter().map(|x| if x.drafts > 0 { x.marginal(1) } else { -1.0 }).collect();
    let mut order = Vec::new();
    while order.len() < max_extra {
        let Some((i, &m)) = next.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0))) else { break };
        if m <= 0.0 {
            break;
        }
        d[i] += 1;
        order.push(i);
        next[i] = if d[i] < demands[i].drafts { demands[i].marginal(d[i] + 1) } else { -1.0 };
    }
    order
}

/// Pick the total row budget that maximizes expected committed tokens per millisecond for this step.
/// One pass over the nested allocation order, not a fresh allocation per candidate budget.
///
/// Only the verify step's cost is counted: this is the choice to make once the drafts exist (their cost is spent).
/// [`plan_round`] is the same search with the draft's cost and the context length in the denominator.
pub fn choose_budget(cost: &CostModel, demands: &[Demand], max_rows: usize) -> (usize, Vec<usize>) {
    let p = plan_with(demands, max_rows, |s| cost.ms(s.rows));
    (p.rows, p.alloc)
}

/// Verify-step time (with its commits) as a function of the rows of the step AND the streams they belong to. A row costs about
/// 0.45 ms and a stream about 0.8 ms (its recurrent state, its commit), so the same 24 rows are dearer split over 6 streams than
/// over 3, and the step's kernels change plan with the row count (a cliff just past 32 rows on the GB10), which a table in rows
/// alone can only smear. Built from a grid measured by `crates/model/tests/bench.rs` (`REXL3_BENCH_GRID`): one rows-to-ms curve per
/// stream count, interpolated between stream counts and extended past the last one by its slope.
#[derive(Clone, Debug)]
pub struct StepCost {
    /// Ascending stream counts, each with its rows-to-ms curve.
    lines: Vec<(usize, CostModel)>,
}

impl StepCost {
    /// A cost that depends on rows only (a stream costs nothing extra): the old single table.
    pub fn rows_only(rows: CostModel) -> Self {
        Self { lines: vec![(1, rows)] }
    }

    /// `points`: `(streams, rows, ms)`. Every stream count needs at least one point and the usual validity of
    /// [`CostModel::try_from_table`] (finite and positive times; a noisy dip in a curve is flattened). Rows below a stream
    /// count's first measured row cost what its first row does.
    pub fn try_from_grid(points: Vec<(usize, usize, f64)>) -> Result<Self, String> {
        let mut by: std::collections::BTreeMap<usize, Vec<(usize, f64)>> = Default::default();
        for (s, r, ms) in points {
            if s == 0 || r == 0 {
                return Err(format!("grid point ({s} streams, {r} rows): both must be at least 1"));
            }
            by.entry(s).or_default().push((r, ms));
        }
        if by.is_empty() {
            return Err("the grid is empty".into());
        }
        let mut lines = Vec::new();
        for (s, mut pts) in by {
            pts.sort_by_key(|p| p.0);
            if pts[0].0 != 1 {
                pts.insert(0, (1, pts[0].1));
            }
            lines.push((s, CostModel::try_from_table(pts).map_err(|e| format!("{s} streams: {e}"))?));
        }
        Ok(Self { lines })
    }

    pub fn from_grid(points: Vec<(usize, usize, f64)>) -> Self {
        Self::try_from_grid(points).unwrap_or_else(|e| panic!("bad step cost grid: {e}"))
    }

    /// The shape the grid was measured to: its most streams, and the most rows of that last line. Past either, `ms` extends the last
    /// segment's slope (rows) or the last stream's per-stream slope (streams): an estimate, not a measurement.
    pub fn measured(&self) -> (usize, usize) {
        let (k, last) = &self.lines[self.lines.len() - 1];
        (*k, last.max_rows())
    }

    /// Milliseconds of a verify step of `rows` rows over `streams` streams.
    pub fn ms(&self, rows: usize, streams: usize) -> f64 {
        let s = streams.max(1);
        let l = &self.lines;
        let hi = l.partition_point(|(k, _)| *k < s); // first line with k >= s
        if hi == l.len() {
            // more streams than any measured: the last line plus its per-stream slope at this many rows
            let (k, last) = &l[l.len() - 1];
            let base = last.ms(rows);
            return match l.len().checked_sub(2).map(|i| &l[i]) {
                Some((kp, prev)) => base + ((base - prev.ms(rows)) / (*k - *kp) as f64).max(0.0) * (s - *k) as f64,
                None => base,
            };
        }
        let (k1, up) = &l[hi];
        if *k1 == s || hi == 0 {
            return up.ms(rows);
        }
        let (k0, low) = &l[hi - 1];
        let (a, b) = (low.ms(rows), up.ms(rows));
        a + (b - a) * (s - *k0) as f64 / (*k1 - *k0) as f64
    }
}

/// The shape of one decode round, as the cost of it depends on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundShape {
    /// Streams in the round (each has its base row in the verify step).
    pub streams: usize,
    /// Rows of the verify step: one per stream plus the drafts it carries.
    pub rows: usize,
    /// Streams that run the draft (those given at least one draft).
    pub drafting: usize,
    /// Rows through the draft's vocabulary head: the drafts asked of the drafter (one per draft position).
    pub head_rows: usize,
    /// Context tokens held by all of the round's streams (what attention reads).
    pub kv_tokens: u64,
}

/// What a round's draft costs, from timings of the draft on the real model: its layers by the number of streams in the round
/// (every drafting stream runs a whole block through them whatever number of drafts it is later asked for), and its vocabulary
/// head by the rows it is given (one per draft position asked for).
#[derive(Clone, Debug)]
pub struct DraftCost {
    layers: CostModel,
    head: CostModel,
    fixed_ms: f64,
    /// Most sequences one draft round carries (the model's `MAX_DRAFT_SEQS`): more run as further rounds, each reading the draft's
    /// weights again and paying the round's sync. `usize::MAX` (the default) prices every number of streams as one round.
    group: usize,
    /// Rows of the draft's block (8 the trained value, 16 the depth block's `REXL3_DRAFT_BLOCK`): the draft's layers run a whole
    /// block per sequence, so the same number of drafting streams costs more ms at 16 rows, and a 16-row block halves the rows a
    /// decode round may hold (`128 / block`), which halves `group`. 0 means "the trained 8" for tables written before the knob.
    block: usize,
}

impl DraftCost {
    /// `layers`: (drafting streams, ms of the draft without its head); `head`: (rows, ms); `fixed_ms`: what any draft round costs
    /// on top (a sync and the uploads). Fails on a malformed table (see [`CostModel::try_from_table`]).
    pub fn try_new(layers: Vec<(usize, f64)>, head: Vec<(usize, f64)>, fixed_ms: f64) -> Result<Self, String> {
        if !fixed_ms.is_finite() || fixed_ms < 0.0 {
            return Err(format!("fixed draft cost {fixed_ms} ms is not finite and non-negative"));
        }
        Ok(Self {
            layers: CostModel::try_from_table(layers).map_err(|e| format!("draft layers: {e}"))?,
            head: CostModel::try_from_table(head).map_err(|e| format!("draft head: {e}"))?,
            fixed_ms,
            group: usize::MAX,
            block: 0,
        })
    }
    /// `try_new` for tables written in code.
    pub fn new(layers: Vec<(usize, f64)>, head: Vec<(usize, f64)>, fixed_ms: f64) -> Self {
        Self::try_new(layers, head, fixed_ms).unwrap_or_else(|e| panic!("bad draft cost: {e}"))
    }
    /// A draft that carries at most `n` sequences a round: with more drafting streams the cost is that of several rounds (see
    /// [`DraftCost::ms`]). Up to `n` streams nothing changes, bit for bit.
    pub fn with_group(mut self, n: usize) -> Self {
        self.group = n.max(1);
        self
    }
    /// A draft whose block runs `rows` rows a sequence (8 the trained value, 16 the depth block): the layers table is measured
    /// at the trained 8, so a 16-row block's layers cost is read at twice the streams (`layers16(n) = layers8(2n)`: the block
    /// runs the same total rows) and the head at twice the rows (`head16(r) = head8(2r)`: every sequence asks for 15 picks, not
    /// 7), and the sequences a round may hold halves (`128 / rows`). Up to 8 rows nothing changes, bit for bit.
    pub fn with_block(mut self, rows: usize) -> Self {
        self.block = rows.max(1);
        self
    }
    /// One draft round of `n` sequences and `rows` rows through the head, at this draft's block.
    fn round_ms(&self, n: usize, rows: usize) -> f64 {
        let (n, rows) = self.block_scale(n, rows);
        self.fixed_ms + self.layers.ms(n) + self.head.ms((rows.max(1.0)) as usize)
    }
    /// The (streams, head rows) the measured tables want for this draft's block: identity at the trained 8, the doubled numbers
    /// at 16 (a 16-row block runs twice the rows a sequence, so `n` sequences at 16 cost the layers what `2n` do at 8).
    fn block_scale(&self, n: usize, rows: usize) -> (usize, f64) {
        match self.block {
            0 | 8 => (n, rows as f64),
            b => (n * b / 8, rows as f64 * b as f64 / 8.0),
        }
    }
    /// The sequences one draft round carries at this draft's block (`group` was set for the trained 8: a 16-row block halves it).
    fn group_of(&self) -> usize {
        match self.block {
            0 | 8 => self.group,
            b => (self.group / b * 8).max(1),
        }
    }
    /// Milliseconds of a draft for `drafting` streams and `head_rows` rows through the head; nothing when nothing drafts. Past one
    /// group of streams it is the sum of the rounds the draft runs: full groups first, the rest last, the head rows shared out in
    /// proportion to the streams of each (the planner knows the total, not which stream is in which round).
    pub fn ms(&self, drafting: usize, head_rows: usize) -> f64 {
        if drafting == 0 {
            return 0.0;
        }
        let group = self.group_of();
        if drafting <= group {
            return self.round_ms(drafting, head_rows);
        }
        let (mut left, mut rows_left, mut total) = (drafting, head_rows, 0.0);
        while left > 0 {
            let n = left.min(group);
            let rows = if n == left { rows_left } else { ((head_rows * n + drafting / 2) / drafting).min(rows_left) };
            total += self.round_ms(n, rows);
            left -= n;
            rows_left -= rows;
        }
        total
    }
}

/// Everything a decode round costs that the budget can move: the verify step, the draft that feeds it, and the attention over
/// the streams' contexts.
#[derive(Clone, Debug)]
pub struct RoundCost {
    /// Verify step with its commits, by total rows, measured at a short context.
    pub verify: StepCost,
    /// The same for a window that is a TREE (`plan_trees` prices rows with it): measured with random trees up to the 64 rows of a window, since
    /// `verify` stops at 8 rows a stream and a tree window runs past that. `None`: `verify` prices tree windows too.
    pub tree_verify: Option<StepCost>,
    /// The draft; `None` leaves it out (a drafter that costs nothing worth counting, such as the copy drafter).
    pub draft: Option<DraftCost>,
    /// Extra verify milliseconds per context token held by the round's streams, on top of `verify`.
    pub kv_ms_per_token: f64,
    /// Extra milliseconds per context token for every further block of [`ATTN_ROW_BLOCK`] rows a stream's window holds: the attention serves
    /// that many rows a CTA, so a window of more rows reads the stream's committed KV once more per block (the flat `kv_ms_per_token` is
    /// the first pass). Only tree windows go past one block in serving; `plan_trees` charges it. 0: a window's rows do not touch it.
    pub kv_pass_ms_per_token: f64,
    /// Milliseconds every round costs whatever it plans (for example absorbing the committed rows into the draft's context, which
    /// happens whether or not the next round drafts).
    pub fixed_ms: f64,
}

/// The most rows a window of TREES is planned with when no tree table says otherwise (`RoundCost::tree_window_rows`): the 64 rows the tree
/// kernels were first built and measured for. With a tree table the cap is the table's reach within its streams and the engine's window past
/// them.
pub const TREE_MAX_ROWS: usize = 64;

/// Rows the attention kernels serve a CTA (`RBLK` in `crates/cuda/csrc/attention.cu`): a window of `r` rows of one stream reads its committed
/// KV `ceil(r / ATTN_ROW_BLOCK)` times.
pub const ATTN_ROW_BLOCK: usize = 8;

/// How a round's draft enters its cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DraftCharge {
    /// By the shape of the plan: what a plan made BEFORE the draft runs must count, so a short plan saves draft work.
    Planned,
    /// A fixed amount already spent (the draft has run). The objective is still tokens per millisecond of the whole round, so a
    /// decision made after the draft keeps the draft's milliseconds in the denominator as a constant instead of dropping them:
    /// dropping them would raise the rate every row is held to and trim drafts the whole-round rate says to keep.
    Spent(f64),
    /// Not counted at all.
    Free,
}

impl RoundCost {
    /// Milliseconds of a round of this shape, with the draft charged as given.
    pub fn ms_charged(&self, s: &RoundShape, charge: DraftCharge) -> f64 {
        self.ms_with(self.verify.ms(s.rows, s.streams), s, charge)
    }

    /// [`RoundCost::ms_charged`] for a round whose window is a tree: the verify step is priced by [`RoundCost::tree_step_ms`].
    pub fn ms_tree_charged(&self, s: &RoundShape, charge: DraftCharge) -> f64 {
        self.ms_with(self.tree_step_ms(s.rows, s.streams), s, charge)
    }

    /// The most rows a window of trees over `streams` streams is planned with, before the engine's own window (`--window-rows`) caps it. Within
    /// the tree table's streams, the table's reach (64 rows: past it the table extrapolates, and the 65th row of a verify step opens a second
    /// tile-decode group of the decode kernel, which only c32's chain grid prices); past its streams the chain table, measured to 128 rows,
    /// prices the window (plus the tree's premium: `tree_step_ms`), so the engine's whole window may be used, which is what lets a
    /// `--window-rows 128` server fill 128 rows with trees at many streams as it does with chains. Without a tree table: [`TREE_MAX_ROWS`].
    pub fn tree_window_rows(&self, streams: usize) -> usize {
        match &self.tree_verify {
            Some(t) => {
                let (top, top_rows) = t.measured();
                if streams <= top {
                    top_rows
                } else {
                    usize::MAX
                }
            }
            None => TREE_MAX_ROWS,
        }
    }

    /// Verify-step milliseconds of a window that is a TREE of `rows` rows over `streams` streams. Two tables exist and each is right where
    /// it was measured: `tree_verify` (random trees, 1 to 8 streams, up to 64 rows) and `verify` (chains, up to the wide grid's 32 streams
    /// and 128 rows, but only 8 rows a stream, which is all a chain has). Within the tree table's streams the tree table prices the window.
    /// Past its last stream count the chain table's measured stream dependence is kept and the tree's premium over a chain (the tree table
    /// minus the chain table at the tree table's last stream count, rows held to what it measured, never negative) is added: a window of
    /// trees is a chain window plus the tail kernel's per-row phase and the slot stack, and that is a nearly constant extra (+0.7 to +2.6 ms
    /// at 8 streams, +0.5 to +6 at one). Without a tree table the chain table prices the tree.
    pub fn tree_step_ms(&self, rows: usize, streams: usize) -> f64 {
        let Some(tree) = &self.tree_verify else { return self.verify.ms(rows, streams) };
        let (top, top_rows) = tree.measured();
        if streams <= top {
            return tree.ms(rows, streams);
        }
        let r = rows.min(top_rows);
        self.verify.ms(rows, streams) + (tree.ms(r, top) - self.verify.ms(r, top)).max(0.0)
    }

    fn ms_with(&self, step_ms: f64, s: &RoundShape, charge: DraftCharge) -> f64 {
        let draft = match (&self.draft, charge) {
            (Some(d), DraftCharge::Planned) => d.ms(s.drafting, s.head_rows),
            (_, DraftCharge::Spent(ms)) => ms,
            _ => 0.0,
        };
        self.fixed_ms + step_ms + self.kv_ms_per_token * s.kv_tokens as f64 + draft
    }

    /// [`RoundCost::ms_charged`] with the draft counted by the plan's shape (`draft_sunk` false) or not at all (true).
    pub fn ms(&self, s: &RoundShape, draft_sunk: bool) -> f64 {
        self.ms_charged(s, if draft_sunk { DraftCharge::Free } else { DraftCharge::Planned })
    }

    /// What the draft of this shape costs by the model (0 when the cost has no draft model).
    pub fn draft_ms(&self, drafting: usize, head_rows: usize) -> f64 {
        self.draft.as_ref().map_or(0.0, |d| d.ms(drafting, head_rows))
    }
}

/// How many drafts each stream gets, and what the round is expected to yield.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// Drafts per stream (never more than its `Demand::drafts`).
    pub alloc: Vec<usize>,
    /// Verify rows: one per stream plus the drafts.
    pub rows: usize,
    /// Streams that get at least one draft.
    pub drafting: usize,
    /// Expected committed tokens over all streams.
    pub tokens: f64,
    /// Expected milliseconds of the round.
    pub ms: f64,
}

/// The cheapest-per-token round: the allocation, among the nested greedy allocations, that maximizes expected committed tokens
/// per millisecond of `cost`. With `draft_sunk` false the draft's cost is counted, so the plan can be made BEFORE the draft runs
/// and a short plan saves draft work (a plan that gives no stream a draft needs no draft round at all).
///
/// Streams' drafts are handed out in order of value (see [`allocation_order`]); the search takes each prefix of that order as a
/// candidate, so it finds the best allocation of that family. A draft stream's fixed cost (its block through the draft's layers)
/// makes the true optimum slightly non-nested in principle; the draft layers cost under 1 ms a stream against 70+ ms of verify, so
/// the nested family is the optimum to within a fraction of a percent (`the_plan_is_near_the_exhaustive_optimum` checks it).
pub fn plan_round(cost: &RoundCost, demands: &[Demand], max_rows: usize, draft_sunk: bool) -> Plan {
    plan_round_charged(cost, demands, max_rows, if draft_sunk { DraftCharge::Free } else { DraftCharge::Planned })
}

/// [`plan_round`] with the draft charged as given (see [`DraftCharge`]).
pub fn plan_round_charged(cost: &RoundCost, demands: &[Demand], max_rows: usize, charge: DraftCharge) -> Plan {
    let kv: u64 = demands.iter().map(|d| d.kv_tokens).sum();
    plan_with(demands, max_rows, |s| cost.ms_charged(&RoundShape { kv_tokens: kv, ..*s }, charge))
}

/// What a stream's draft TREE offers the verify window: the probability that each node, in the order the tree search popped them, lies on
/// the path the target accepts (a calibrated number: `gb_spec::tree::calibrated`). A node adds one row and, when it is on the accepted path,
/// one committed token, so its value is that probability; the values fall along the order (a node is never likelier than the nodes popped
/// before it), and the first `k` nodes are the `k`-node tree, so a greedy allocation by value is optimal.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeDemand {
    /// `probs[i]`: P(node `i` of the pop order is on the accepted path).
    pub probs: Vec<f64>,
    /// Draft rows (levels) the stream's draft ran through its vocabulary head.
    pub head_rows: usize,
    /// Tokens of context this stream's KV holds (what its attention reads each round).
    pub kv_tokens: u64,
}

impl TreeDemand {
    /// Expected committed tokens with the first `k` nodes verified: the sample past the accepted path, plus each node's probability.
    pub fn expected_tokens(&self, k: usize) -> f64 {
        1.0 + self.probs[..k.min(self.probs.len())].iter().map(|&p| sane(p)).sum::<f64>()
    }
}

/// The least and most rows a depth-adaptive draft block may run: the trained 8 and the depth block's 16
/// (`REXL3_DRAFT_BLOCK`, `docs/TREE_DRAFTING.md` section 13).
pub const MIN_DRAFT_BLOCK: usize = 8;
/// [`MIN_DRAFT_BLOCK`]'s upper end: the kernels are built for exactly 8 and 16.
pub const MAX_DRAFT_BLOCK: usize = 16;
/// The most streams that still run one 16-row draft group (a decode round holds 128 rows: 8 sequences x 16 rows).
pub const DEEP_BLOCK_STREAMS: usize = 4;

/// Rows the draft's block should run next round: the depth the round just KEPT, plus a margin. A stream whose round accepted
/// `kept` of its drafts (the deepest level verified and kept on the accepted path) is telling the draft its block was deep
/// enough to hold `kept` levels and no deeper: next round's block is `kept + 2`, or `kept + 5` when the block's LAST level was
/// itself kept (the block may be too short to show how deep the stream wants to go). The floor is the trained block the
/// checkpoint was trained with (the calibration and the head's candidates are the trained 8's), the cap the kernels' 16; past
/// [`DEEP_BLOCK_STREAMS`] streams the block drops back to the trained 8 whatever the streams want (a 16-row block halves the
/// sequences a draft round carries, and a second draft group costs another full pass over the draft's weights).
///
/// A block change only changes which levels the draft PROPOSES (a draft never changes what is committed: `docs/QWEN38_SPEC.md`),
/// and the floors and caps keep every block inside what the kernels and the calibration cover.
pub fn adaptive_block(kept: usize, last_level_kept: bool, streams: usize) -> usize {
    let want = kept + if last_level_kept { 5 } else { 2 };
    let cap = if streams > DEEP_BLOCK_STREAMS { MIN_DRAFT_BLOCK } else { MAX_DRAFT_BLOCK };
    want.clamp(MIN_DRAFT_BLOCK, cap)
}

/// How many nodes of each stream's tree ride in the verify window: the nested greedy allocation (every stream's nodes ranked together by
/// value) whose prefix maximizes expected committed tokens per millisecond of `cost`. Every stream keeps its root row; a stream's nodes
/// are a prefix of its pop order. The draft has run (the trees come from its lattice), so its cost is a constant in the denominator
/// (`charge`, normally [`DraftCharge::Spent`]). `max_rows` caps the window's rows (roots included), and [`RoundCost::tree_window_rows`] caps it
/// further: a tree window is not planned wider than the price tables reach for it.
pub fn plan_trees(cost: &RoundCost, demands: &[TreeDemand], max_rows: usize, charge: DraftCharge) -> Plan {
    let n = demands.len();
    let max_rows = max_rows.min(cost.tree_window_rows(n));
    let kv: u64 = demands.iter().map(|d| d.kv_tokens).sum();
    let head_rows: usize = demands.iter().map(|d| d.head_rows).sum();
    // the milliseconds of a round of `rows` rows when `extra_kv` context tokens are read one more time (the attention blocks of 8 rows)
    let ms = |rows: usize, extra_kv: u64| -> f64 {
        cost.ms_tree_charged(&RoundShape { streams: n, rows, drafting: n, head_rows, kv_tokens: kv }, charge)
            + cost.kv_pass_ms_per_token * extra_kv as f64
    };
    // each stream hands out its nodes in pop order (probabilities never rise along it, so a prefix is the best k nodes); a row that opens a
    // new block of 8 rows of its stream costs a pass over that stream's context on top of the step, so the next node to buy is the one with the
    // most probability per millisecond it adds, not the likeliest
    let mut alloc = vec![0usize; n];
    let mut tokens = n as f64;
    let mut extra_kv = 0u64;
    let (mut best_rate, mut best_ms) = (tokens / ms(n, 0), ms(n, 0));
    let mut best = (n, tokens, alloc.clone());
    let mut rows = n;
    while rows < max_rows.max(n) {
        let step = (ms(rows + 1, extra_kv) - ms(rows, extra_kv)).max(0.05);
        let mut pick: Option<(f64, usize)> = None; // (probability per added ms, stream)
        for (i, d) in demands.iter().enumerate() {
            let Some(&p) = d.probs.get(alloc[i]) else { continue };
            if sane(p) <= 0.0 {
                continue;
            }
            let opens_block = (1 + alloc[i]) % ATTN_ROW_BLOCK == 0; // this node becomes row `1 + alloc[i]`, row 8 (0-based) starts block 2
            let added = step + if opens_block { cost.kv_pass_ms_per_token * d.kv_tokens as f64 } else { 0.0 };
            let value = sane(p) / added;
            if pick.is_none_or(|(v, _)| value > v) {
                pick = Some((value, i));
            }
        }
        let Some((_, i)) = pick else { break };
        if (1 + alloc[i]) % ATTN_ROW_BLOCK == 0 {
            extra_kv += demands[i].kv_tokens;
        }
        tokens += sane(demands[i].probs[alloc[i]]);
        alloc[i] += 1;
        rows += 1;
        let t = ms(rows, extra_kv);
        if tokens / t > best_rate {
            best_rate = tokens / t;
            best_ms = t;
            best = (rows, tokens, alloc.clone());
        }
    }
    Plan { alloc: best.2, rows: best.0, drafting: n, tokens: best.1, ms: best_ms }
}

fn plan_with(demands: &[Demand], max_rows: usize, ms: impl Fn(&RoundShape) -> f64) -> Plan {
    let n = demands.len();
    let order = allocation_order(demands, max_rows.max(n) - n);
    let mut d = vec![0usize; n];
    let mut drafting = 0usize;
    // expected tokens per stream, recomputed from its draft count each time so the total is the same sum whatever the order
    let mut each: Vec<f64> = demands.iter().map(|x| x.expected_tokens(0)).collect();
    let eval = |tokens: f64, rows: usize, drafting: usize| -> (f64, f64) {
        let t = ms(&RoundShape { streams: n, rows, drafting, head_rows: rows - n, kv_tokens: 0 });
        (tokens / t, t)
    };
    let tokens: f64 = each.iter().sum();
    let (mut best_rate, mut best_ms) = eval(tokens, n, 0);
    let mut best = (n, 0usize, tokens, d.clone());
    for (r, &i) in order.iter().enumerate() {
        if d[i] == 0 {
            drafting += 1;
        }
        d[i] += 1;
        each[i] = demands[i].expected_tokens(d[i]);
        let tokens: f64 = each.iter().sum();
        let rows = n + r + 1;
        let (rate, t) = eval(tokens, rows, drafting);
        if rate > best_rate {
            best_rate = rate;
            best_ms = t;
            best = (rows, drafting, tokens, d.clone());
        }
    }
    Plan { rows: best.0, drafting: best.1, tokens: best.2, ms: best_ms, alloc: best.3 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mm() -> ModelMemory {
        ModelMemory { weights_bytes: 20, kv_bytes_per_token: 1, state_bytes_per_slot: 10, activation_reserve: 5 }
    }

    #[test]
    fn admission_fails_closed_and_releases() {
        let mut a = Admission::new(100, 15, mm(), 4).unwrap(); // capacity 100 - 15 - 20 - 5 = 60
        assert_eq!(a.capacity(), 60);
        let s0 = a.admit(20).unwrap(); // 30
        let _s1 = a.admit(20).unwrap(); // 60
        assert!(matches!(a.admit(1), Err(AdmitError::NoMemory { .. })));
        a.release(s0);
        assert!(a.admit(20).is_ok());
    }

    #[test]
    fn qwen38_27b_kv_matches_published_budget() {
        // r0b0tlab/qwen38-exl3-dflash2: 16 full-attention layers x 4 KV heads x 256 dim x 2, fp16:
        // 16.0 GiB at 262,144 tokens.
        let per = KvLayout::new().with(KvLayerClass::gqa(16, 4, 256, 2)).bytes_per_token("qwen", 256).unwrap();
        assert_eq!(per, 65536);
        assert_eq!(per * 262144, 16 * (1u64 << 30));
    }

    #[test]
    fn admission_refuses_a_model_that_costs_no_kv() {
        // zero KV per token would charge nothing for context: every request would be admitted at any length
        for state in [0, 1 << 20] {
            let zero =
                ModelMemory { weights_bytes: 0, kv_bytes_per_token: 0, state_bytes_per_slot: state, activation_reserve: 0 };
            let e = Admission::new(64 << 30, 0, zero, 4).unwrap_err();
            assert!(matches!(&e, MemoryError::Zero(m) if m.contains("KV bytes per token")), "{e}");
        }
        assert!(Admission::new(64 << 30, 0, mm(), 4).is_ok());
    }

    #[test]
    fn admission_from_a_profile_charges_every_token() {
        struct OneLayer;
        impl MemoryModel for OneLayer {
            fn family(&self) -> &str {
                "one"
            }
            fn kv_layout(&self) -> KvLayout {
                KvLayout::new().with(KvLayerClass::gqa(1, 1, 128, 2)) // 512 bytes a token
            }
            fn recurrent_layers(&self) -> u64 {
                0
            }
            fn state_bytes_per_seq(&self) -> u64 {
                0
            }
            fn window_record_bytes_per_row(&self) -> u64 {
                0
            }
            fn scratch_bytes(&self, rows: u64) -> u64 {
                rows * 10
            }
            fn activation_reserve(&self) -> u64 {
                1
            }
            fn drafter_state_bytes_per_slot(&self) -> u64 {
                0
            }
            fn vision(&self) -> Option<VisionMemory> {
                None
            }
        }
        let p = MemoryProfile::new(&OneLayer, StepShape { page_tokens: 256, window_rows: 8, scratch_rows: 8 }).unwrap();
        let mm = ModelMemory::from_profile(&p, 0, 0);
        assert_eq!((mm.kv_bytes_per_token, mm.state_bytes_per_slot), (512, 0));
        let mut a = Admission::new(512 * 1000, 0, mm, 4).unwrap();
        assert_eq!(a.cost(100), 51_200);
        assert_eq!(a.fits(100), 10);
        assert!(a.admit(1000).is_ok());
        assert!(matches!(a.admit(1), Err(AdmitError::NoMemory { .. })), "1000 tokens used all of 1000 tokens of capacity");
    }

    #[test]
    fn blocks_refcount() {
        let mut p = BlockPool::new(4, 16);
        let a = p.alloc(3).unwrap();
        assert!(p.alloc(2).is_none());
        p.retain(&a[..1]);
        p.release(&a);
        assert_eq!(p.free_blocks(), 3);
        p.release(&a[..1]);
        assert_eq!(p.free_blocks(), 4);
    }

    #[test]
    fn image_changes_the_key() {
        let a = vec![1, 2, image_token(111, 0), 3];
        let b = vec![1, 2, image_token(222, 0), 3];
        assert_ne!(prefix_hashes(&a)[3], prefix_hashes(&b)[3]);
        assert_eq!(prefix_hashes(&a)[1], prefix_hashes(&b)[1]);
    }

    #[test]
    fn allocation_prefers_confident_streams() {
        let d = [Demand::geometric(0.9, 7), Demand::geometric(0.3, 7)];
        let a = allocate(&d, 2 + 4);
        assert_eq!(a.iter().sum::<usize>(), 4);
        assert!(a[0] > a[1]);
        // budget below the stream count: no drafts at all
        assert_eq!(allocate(&d, 1), vec![0, 0]);
    }

    #[test]
    fn budget_stops_at_the_knee() {
        // flat to 16 rows, then +4 ms per extra row: widening past 16 should not pay at 0.5 acceptance
        let cost = CostModel::from_table(vec![(1, 70.0), (16, 72.0), (32, 136.0)]);
        let d = vec![Demand::geometric(0.5, 7); 4];
        let (rows, _) = choose_budget(&cost, &d, 32);
        assert!(rows <= 16, "chose {rows}");
    }

    fn verify_table() -> CostModel {
        CostModel::from_table(vec![(1, 66.0), (8, 69.1), (16, 72.3), (32, 80.7), (48, 97.7), (64, 127.5)])
    }

    /// A grid shaped like the measured one: 0.45 ms a row, 0.8 ms a stream, and 10 ms more once a step passes 32 rows.
    fn grid() -> StepCost {
        let mut pts = Vec::new();
        for s in 1..=8usize {
            for r in s..=8 * s {
                let cliff = if r > 32 { 10.0 } else { 0.0 };
                pts.push((s, r, 65.7 + 0.45 * (r - 1) as f64 + 0.8 * (s - 1) as f64 + cliff));
            }
        }
        StepCost::from_grid(pts)
    }

    #[test]
    fn the_step_cost_reads_the_grid_and_interpolates_between_stream_counts() {
        let g = grid();
        // exact at a measured point
        assert!((g.ms(24, 6) - (65.7 + 0.45 * 23.0 + 0.8 * 5.0)).abs() < 1e-9);
        // the cliff is where the kernel plan changes, not smeared between 32 and 48 rows
        assert!(g.ms(33, 6) - g.ms(32, 6) > 9.0, "{} -> {}", g.ms(32, 6), g.ms(33, 6));
        assert!(g.ms(34, 6) - g.ms(33, 6) < 1.0);
        // the same rows over more streams cost more
        assert!(g.ms(24, 6) > g.ms(24, 3));
        // between two measured stream counts (a grid with lines only at 2, 4 and 8 streams): in proportion to the distance
        let mut sparse = Vec::new();
        for s in [2usize, 4, 8] {
            for r in s..=8 * s {
                sparse.push((s, r, 65.0 + 0.45 * r as f64 + 2.0 * s as f64));
            }
        }
        let sg = StepCost::from_grid(sparse);
        let (a, b) = (sg.ms(20, 4), sg.ms(20, 8));
        assert!((sg.ms(20, 6) - (a + b) / 2.0).abs() < 1e-9, "halfway between 4 and 8 streams");
        assert!((sg.ms(20, 5) - (a + (b - a) / 4.0)).abs() < 1e-9, "a quarter of the way");
        assert!(sg.ms(20, 6) > sg.ms(20, 4) && sg.ms(20, 6) < sg.ms(20, 8));
        // beyond the last measured count: its line plus the per-stream slope
        let slope = g.ms(40, 8) - g.ms(40, 7);
        assert!((g.ms(40, 10) - (g.ms(40, 8) + 2.0 * slope)).abs() < 1e-9);
        assert!(g.ms(40, 12) > g.ms(40, 10));
        // a rows-only cost ignores the stream count
        let r = StepCost::rows_only(verify_table());
        assert_eq!(r.ms(16, 1), r.ms(16, 9));
        assert_eq!(r.ms(16, 1), verify_table().ms(16));
    }

    #[test]
    fn a_malformed_grid_is_refused() {
        assert!(StepCost::try_from_grid(vec![]).is_err());
        assert!(StepCost::try_from_grid(vec![(0, 1, 1.0)]).is_err());
        assert!(StepCost::try_from_grid(vec![(1, 0, 1.0)]).is_err());
        assert!(StepCost::try_from_grid(vec![(1, 1, f64::NAN)]).is_err());
        assert!(StepCost::try_from_grid(vec![(2, 4, 10.0), (2, 4, 11.0)]).is_err(), "a row count twice on one line");
        // a line that starts above one row is flat below its first measured row
        let g = StepCost::from_grid(vec![(3, 3, 70.0), (3, 8, 75.0)]);
        assert_eq!(g.ms(1, 3), 70.0);
    }

    #[test]
    fn the_budget_stops_at_the_cliff_when_it_is_in_the_cost_and_walks_over_it_when_it_is_smeared() {
        // 6 streams that accept well: 32 rows is 26 drafts; the 33rd row costs 10 ms
        let d = vec![Demand::geometric(0.8, 7); 6];
        let smeared = RoundCost {
            verify: StepCost::rows_only(verify_table()),
            tree_verify: None,
            draft: None,
            kv_ms_per_token: 0.0,
            kv_pass_ms_per_token: 0.0,
            fixed_ms: 0.0,
        };
        let exact = RoundCost {
            verify: grid(),
            tree_verify: None,
            draft: None,
            kv_ms_per_token: 0.0,
            kv_pass_ms_per_token: 0.0,
            fixed_ms: 0.0,
        };
        let a = plan_round(&smeared, &d, 64, true);
        let b = plan_round(&exact, &d, 64, true);
        assert!(b.rows <= 32, "the cliff makes row 33 cost 10 ms: {b:?}");
        assert!(a.rows > 32, "the smeared table prices row 33 at about a millisecond: {a:?}");
    }

    fn round_cost() -> RoundCost {
        RoundCost {
            verify: StepCost::rows_only(verify_table()),
            tree_verify: None,
            draft: Some(DraftCost::new(
                vec![(1, 7.5), (2, 8.0), (4, 9.5), (6, 11.0)],
                vec![(1, 3.0), (7, 4.0), (16, 4.5), (32, 8.0), (48, 12.0), (64, 16.0)],
                0.5,
            )),
            kv_ms_per_token: 0.0,
            kv_pass_ms_per_token: 0.0,
            fixed_ms: 0.0,
        }
    }

    fn tree_demand(probs: &[f64]) -> TreeDemand {
        TreeDemand { probs: probs.to_vec(), head_rows: 7, kv_tokens: 0 }
    }

    /// One stream: nodes are added while the next one raises tokens per millisecond, and the search keeps a prefix of the pop order.
    #[test]
    fn trees_take_nodes_while_they_pay_and_stop_at_the_knee() {
        let cost = round_cost();
        let steep = tree_demand(&[0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]);
        let p = plan_trees(&cost, std::slice::from_ref(&steep), 64, DraftCharge::Spent(9.5));
        assert!(p.alloc[0] >= 3 && p.alloc[0] < 12, "worth some nodes, not all: {p:?}");
        assert_eq!(p.rows, 1 + p.alloc[0]);
        assert!((p.tokens - steep.expected_tokens(p.alloc[0])).abs() < 1e-12);
        // a worthless tree gets no node, whatever the budget
        let flat = tree_demand(&[0.001; 20]);
        assert_eq!(plan_trees(&cost, &[flat], 64, DraftCharge::Spent(9.5)).alloc, vec![0]);
        // the row cap holds (roots included)
        let rich = tree_demand(&[0.99; 40]);
        let capped = plan_trees(&cost, std::slice::from_ref(&rich), 6, DraftCharge::Spent(9.5));
        assert!(capped.rows <= 6, "{capped:?}");
        assert!(plan_trees(&cost, &[rich], 64, DraftCharge::Spent(9.5)).alloc[0] > capped.alloc[0]);
    }

    /// Several streams share the rows by value: the stream whose nodes are likelier gets more of them, and each keeps a prefix.
    #[test]
    fn rows_go_to_the_likelier_nodes_across_streams() {
        let cost = round_cost();
        let a = tree_demand(&[0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]);
        let b = tree_demand(&[0.3, 0.1, 0.05, 0.02]);
        let p = plan_trees(&cost, &[a, b], 64, DraftCharge::Spent(12.0));
        assert!(p.alloc[0] > p.alloc[1], "{p:?}");
        assert_eq!(p.rows, 2 + p.alloc[0] + p.alloc[1]);
        // identical streams split evenly (ties go to the lower stream, so the split differs by at most one)
        let c = tree_demand(&[0.7, 0.5, 0.3, 0.2, 0.1]);
        let q = plan_trees(&cost, &[c.clone(), c.clone(), c], 64, DraftCharge::Spent(12.0));
        let (lo, hi) = (q.alloc.iter().min().unwrap(), q.alloc.iter().max().unwrap());
        assert!(hi - lo <= 1, "{q:?}");
    }

    /// A tree window has its own cost table when one was measured: the plan prices the rows with it, not with the chain's.
    #[test]
    fn trees_are_priced_by_the_tree_table_when_there_is_one() {
        let chain_cost = round_cost();
        let rich = tree_demand(&[0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.28, 0.26, 0.24, 0.22, 0.2, 0.18, 0.16, 0.14, 0.12]);
        let cheap = plan_trees(&chain_cost, std::slice::from_ref(&rich), 64, DraftCharge::Spent(9.5));
        // the same cost, but tree windows cost three times as much for every row past the second
        let dear_table = StepCost::rows_only(CostModel::from_table(vec![
            (1, 66.0),
            (2, 66.5),
            (8, 66.5 + 3.0 * 2.6),
            (16, 66.5 + 3.0 * 5.8),
            (64, 66.5 + 3.0 * 61.0),
        ]));
        let with_tree_table = RoundCost { tree_verify: Some(dear_table), ..chain_cost.clone() };
        let dear = plan_trees(&with_tree_table, std::slice::from_ref(&rich), 64, DraftCharge::Spent(9.5));
        assert!(dear.alloc[0] < cheap.alloc[0], "rows that cost more are bought less: {dear:?} against {cheap:?}");
        // chain windows keep the chain table
        assert_eq!(
            with_tree_table
                .ms_charged(&RoundShape { streams: 1, rows: 16, drafting: 1, head_rows: 7, kv_tokens: 0 }, DraftCharge::Free),
            chain_cost
                .ms_charged(&RoundShape { streams: 1, rows: 16, drafting: 1, head_rows: 7, kv_tokens: 0 }, DraftCharge::Free)
        );
    }

    /// A chain grid shaped like c32's wide one: 1 to 16 streams, up to 8 rows a stream, 0.45 ms a row and 0.8 ms a stream.
    fn wide_chain_grid() -> StepCost {
        let mut pts = Vec::new();
        for st in 1..=16usize {
            for r in st..=8 * st {
                pts.push((st, r, chain_ms(r, st)));
            }
        }
        StepCost::from_grid(pts)
    }

    fn chain_ms(rows: usize, streams: usize) -> f64 {
        65.7 + 0.45 * (rows - 1) as f64 + 0.8 * (streams - 1) as f64
    }

    /// A tree grid shaped like the tree work's: 1 to 4 streams, up to 64 rows a window, the chain's price plus a premium that grows with the rows.
    fn narrow_tree_grid(premium: impl Fn(usize) -> f64) -> StepCost {
        let mut pts = Vec::new();
        for st in 1..=4usize {
            for r in st..=64 {
                pts.push((st, r, chain_ms(r, st) + premium(r)));
            }
        }
        StepCost::from_grid(pts)
    }

    /// Two cost tables, each right where it was measured: within the tree table's streams a tree window is priced by it, past them by the
    /// chain table's stream dependence plus the tree's premium over a chain at the tree table's last stream count (rows held to what it
    /// measured, never negative), and with no tree table by the chain table.
    #[test]
    fn a_window_of_trees_is_priced_by_the_tree_table_within_its_streams_and_by_the_chain_table_plus_the_premium_past_them() {
        let verify = wide_chain_grid();
        let tree = narrow_tree_grid(|r| 1.0 + 0.02 * r as f64);
        assert_eq!(tree.measured(), (4, 64));
        let rc = RoundCost {
            verify: verify.clone(),
            tree_verify: Some(tree.clone()),
            draft: None,
            kv_ms_per_token: 0.0,
            kv_pass_ms_per_token: 0.0,
            fixed_ms: 0.0,
        };
        // within the tree table's streams: its own prices, to the bit, for any rows (past 64 it extends its own last segment)
        for streams in 1..=4usize {
            for rows in [streams, 8, 24, 64, 90] {
                assert_eq!(rc.tree_step_ms(rows, streams), tree.ms(rows, streams), "{rows} rows over {streams} streams");
            }
        }
        // past them: the chain table at this many streams plus the premium of 4 streams at the same rows (1.0 + 0.02 a row)
        for (rows, streams) in [(6usize, 6usize), (32, 6), (48, 10), (64, 16)] {
            let want = chain_ms(rows, streams) + 1.0 + 0.02 * rows as f64;
            assert!((rc.tree_step_ms(rows, streams) - want).abs() < 1e-9, "{rows} rows over {streams} streams");
        }
        // rows past what the tree table measured keep the premium of its last row (no extrapolated slope: the tree table stops at 64)
        let want = chain_ms(100, 12) + 1.0 + 0.02 * 64.0;
        assert!((rc.tree_step_ms(100, 12) - want).abs() < 1e-9, "{}", rc.tree_step_ms(100, 12));
        // a tree table that is CHEAPER than the chain at its last stream count adds nothing (the premium is never negative)
        let cheaper = RoundCost { tree_verify: Some(narrow_tree_grid(|_| -5.0)), ..rc.clone() };
        assert_eq!(cheaper.tree_step_ms(32, 6), verify.ms(32, 6));
        assert_eq!(
            cheaper.tree_step_ms(32, 3),
            cheaper.tree_verify.as_ref().unwrap().ms(32, 3),
            "inside its streams it is the table"
        );
        // no tree table: the chain table prices trees at every shape
        let none = RoundCost { tree_verify: None, ..rc.clone() };
        for (rows, streams) in [(8usize, 1usize), (32, 4), (48, 6), (64, 16)] {
            assert_eq!(none.tree_step_ms(rows, streams), verify.ms(rows, streams));
        }
        // and the round's price reads it: a plan of trees at 6 streams is charged the premium, a chain plan at the same shape is not
        let shape = RoundShape { streams: 6, rows: 32, drafting: 6, head_rows: 42, kv_tokens: 0 };
        let diff = rc.ms_tree_charged(&shape, DraftCharge::Free) - rc.ms_charged(&shape, DraftCharge::Free);
        assert!((diff - (1.0 + 0.02 * 32.0)).abs() < 1e-9, "{diff}");
    }

    /// A window of trees is planned inside what the price tables reach for trees (`RoundCost::tree_window_rows`), whatever the engine's window
    /// allows chains: with rows nearly free every node is worth buying, so only the caps bind.
    #[test]
    fn a_tree_window_is_never_planned_wider_than_the_rows_trees_were_measured_to() {
        let flat = RoundCost {
            verify: StepCost::rows_only(CostModel::from_table(vec![(1, 70.0), (256, 71.0)])),
            tree_verify: None,
            draft: None,
            kv_ms_per_token: 0.0,
            kv_pass_ms_per_token: 0.0,
            fixed_ms: 0.0,
        };
        let rich = tree_demand(&[0.99; 63]);
        for streams in [2usize, 3, 16] {
            let d = vec![rich.clone(); streams];
            let wide = plan_trees(&flat, &d, 128, DraftCharge::Free);
            assert_eq!(wide.rows, TREE_MAX_ROWS, "{streams} streams: {wide:?}");
            assert_eq!(
                wide,
                plan_trees(&flat, &d, TREE_MAX_ROWS, DraftCharge::Free),
                "a 128-row window plans the same trees as a 64-row one"
            );
        }
        // a window below the cap binds first (roots included)
        let three = vec![rich.clone(); 3];
        assert_eq!(plan_trees(&flat, &three, 20, DraftCharge::Free).rows, 20);
        // with a tree table the cap is its reach (64 rows) within its streams (4 here) and the engine's window past them, where the chain table
        // prices rows to 128: 4 streams stay at 64 rows, 6 streams (the chain grid's lines to 16) take a 128-row window, 70 rows of a 70-row one
        let table = RoundCost {
            verify: StepCost::rows_only(CostModel::from_table(vec![(1, 70.0), (256, 71.0)])),
            tree_verify: Some(narrow_tree_grid(|_| 0.5)),
            ..flat.clone()
        };
        assert_eq!(table.tree_window_rows(4), 64);
        assert_eq!(table.tree_window_rows(5), usize::MAX);
        assert_eq!(flat.tree_window_rows(5), TREE_MAX_ROWS, "no tree table: the default cap at every stream count");
        let plan = |streams: usize, window: usize| {
            plan_trees(&table, &vec![tree_demand(&[0.99; 63]); streams], window, DraftCharge::Free).rows
        };
        assert_eq!(plan(4, 128), 64, "within the tree table's streams: its reach");
        assert_eq!(plan(6, 128), 128, "past them: the engine's window");
        assert_eq!(plan(6, 70), 70);
        assert_eq!(plan(6, 64), 64);
        // more streams than the cap keep their roots and nothing else
        let many = vec![rich; TREE_MAX_ROWS + 6];
        let p = plan_trees(&flat, &many, 128, DraftCharge::Free);
        assert_eq!((p.rows, p.alloc.iter().sum::<usize>()), (TREE_MAX_ROWS + 6, 0));
    }

    /// A window of more than 8 rows reads its stream's committed KV once more per block of 8: at a long context the plan stops at the block
    /// boundary (it does not buy a ninth row that costs a whole pass over the context), at a short one it goes on.
    #[test]
    fn rows_past_a_block_of_eight_pay_for_a_pass_over_the_context() {
        let probs: Vec<f64> = (0..30).map(|k| 0.4 * 0.85f64.powi(k)).collect();
        let mut cost = round_cost();
        cost.kv_pass_ms_per_token = 0.0005;
        let at = |cost: &RoundCost, kv: u64| {
            let d = TreeDemand { probs: probs.clone(), head_rows: 7, kv_tokens: kv };
            plan_trees(cost, std::slice::from_ref(&d), 64, DraftCharge::Spent(9.5))
        };
        let short = at(&cost, 300);
        let long = at(&cost, 32_768);
        assert!(short.alloc[0] > 8, "at a short context a likely node is worth its row: {short:?}");
        assert!(long.alloc[0] <= 7, "at 32k tokens a ninth row costs 16 ms: the plan stops inside the first block: {long:?}");
        // two streams of different context lengths: the long one stays inside one block, the short one may go past it
        let a = TreeDemand { probs: probs.clone(), head_rows: 7, kv_tokens: 40_000 };
        let b = TreeDemand { probs: probs.clone(), head_rows: 7, kv_tokens: 200 };
        let p = plan_trees(&cost, &[a, b], 64, DraftCharge::Spent(12.0));
        assert!(p.alloc[0] <= 7, "{p:?}");
        assert!(p.alloc[1] > 8, "{p:?}");
        // with the pass free the long context changes nothing
        cost.kv_pass_ms_per_token = 0.0;
        assert!(at(&cost, 32_768).alloc[0] > 8);
    }

    /// The cost curve is in the plan: past a cliff in the verify step the plan stops, as it does for chains.
    #[test]
    fn trees_stop_before_a_cliff_in_the_step() {
        let g = grid(); // a cliff at 33 rows for 6 streams in the test grid
        let cost = RoundCost {
            verify: g,
            tree_verify: None,
            draft: None,
            kv_ms_per_token: 0.0,
            kv_pass_ms_per_token: 0.0,
            fixed_ms: 0.0,
        };
        let rich = tree_demand(&[0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02]);
        let p = plan_trees(&cost, &vec![rich; 6], 64, DraftCharge::Free);
        assert!(p.rows <= 32, "{p:?}");
        assert!(p.rows >= 24, "the nodes below the cliff are worth taking: {p:?}");
    }

    #[test]
    fn a_measured_curve_spends_rows_where_the_chain_holds_and_stops_where_it_breaks() {
        // six strong positions, then a cliff: the geometric average (0.79) smears the cliff over every position, so it undervalues
        // the strong ones and gives up on them too early
        let curve = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.05];
        let geo = curve.iter().sum::<f64>() / curve.len() as f64;
        let cost = round_cost();
        let measured = vec![Demand::from_conditional(&curve, 7); 6];
        let assumed = vec![Demand::geometric(geo, 7); 6];
        let a = plan_round(&cost, &measured, 64, false);
        let b = plan_round(&cost, &assumed, 64, false);
        assert!(a.alloc.iter().all(|&k| k == 6), "the seventh draft is worth 0.03 of a token: {a:?}");
        assert!(b.alloc.iter().all(|&k| k < 6), "the geometric average gives up on the strong positions: {b:?}");
        assert!(a.tokens > b.tokens, "and the measured plan expects more tokens: {} against {}", a.tokens, b.tokens);
        // marginal values follow the curve
        let d = Demand::from_conditional(&curve, 7);
        assert!((d.marginal(1) - 0.9).abs() < 1e-12 && (d.marginal(2) - 0.81).abs() < 1e-12);
        assert!((d.marginal(7) - 0.9f64.powi(6) * 0.05).abs() < 1e-12);
        assert!((d.expected_tokens(2) - (1.0 + 0.9 + 0.81)).abs() < 1e-12);
    }

    #[test]
    fn deciding_before_the_draft_counts_its_cost_and_can_skip_it() {
        let cost = round_cost();
        // one stream that almost never accepts: a draft round costs ~12 ms for 0.02 of a token
        let d = [Demand::geometric(0.02, 7)];
        let before = plan_round(&cost, &d, 64, false);
        assert_eq!((before.rows, before.drafting, before.alloc.clone()), (1, 0, vec![0]), "no draft round at all");
        // the same stream after the draft has run: its cost is spent, and 7 extra rows cost only ~3 ms
        let after = plan_round(&cost, &d, 64, true);
        assert!(after.rows >= 1);
        // a stream that accepts well still drafts all 7 before the draft
        let good = plan_round(&cost, &[Demand::geometric(0.9, 7)], 64, false);
        assert_eq!(good.alloc, vec![7]);
        assert_eq!(good.drafting, 1);
    }

    #[test]
    fn a_decision_after_the_draft_keeps_its_milliseconds_in_the_rate() {
        // dropping the draft's milliseconds raises the tokens-per-millisecond every extra row is held to, and trims drafts that the
        // whole-round rate keeps; charging them as a constant gives the same decision the pre-draft plan made
        let cost = round_cost();
        let d = vec![Demand::geometric(0.7, 7); 6];
        let planned = plan_round(&cost, &d, 64, false);
        let spent = plan_round_charged(&cost, &d, 64, DraftCharge::Spent(cost.draft_ms(planned.drafting, planned.rows - 6)));
        let free = plan_round_charged(&cost, &d, 64, DraftCharge::Free);
        assert_eq!(spent.alloc, planned.alloc, "the same whole-round rate, the same plan");
        assert!(free.rows <= spent.rows, "free {} rows, spent {} rows", free.rows, spent.rows);
        assert!(spent.ms > free.ms, "the spent draft is in the round's milliseconds");
    }

    #[test]
    fn a_dead_stream_does_not_pay_for_a_block_in_the_draft() {
        let cost = round_cost();
        let mut d = vec![Demand::geometric(0.8, 7); 5];
        d.push(Demand::geometric(0.001, 7));
        let p = plan_round(&cost, &d, 64, false);
        assert_eq!(p.alloc[5], 0, "{p:?}");
        assert_eq!(p.drafting, 5);
        assert!(p.alloc[..5].iter().all(|&k| k > 0));
    }

    #[test]
    fn long_contexts_make_the_round_dearer_and_so_drafts_cheaper_by_comparison() {
        // attention over a long context is paid whatever the draft count, so each extra row is a smaller part of the round
        let mut cost = round_cost();
        let d = vec![Demand::geometric(0.62, 7); 6];
        let short = plan_round(&cost, &d, 64, false);
        cost.kv_ms_per_token = 0.0005; // 6 streams x 20k tokens = 60 ms of attention
        let long: Vec<Demand> = d.iter().map(|x| x.with_kv_tokens(20_000)).collect();
        let p = plan_round(&cost, &long, 64, false);
        assert!(p.rows > short.rows, "short {} rows, long {} rows", short.rows, p.rows);
        assert!(
            p.ms > short.ms + 55.0 && p.ms < short.ms + 70.0,
            "the 120k tokens of context cost ~60 ms: {} vs {}",
            p.ms,
            short.ms
        );
    }

    #[test]
    fn verify_only_costs_match_the_old_budget_when_the_draft_is_sunk() {
        let cost = round_cost();
        let d = vec![Demand::geometric(0.7, 7), Demand::geometric(0.4, 7), Demand::geometric(0.9, 5)];
        let p = plan_round(&cost, &d, 64, true);
        let (rows, alloc) = choose_budget(&verify_table(), &d, 64);
        assert_eq!((p.rows, p.alloc), (rows, alloc));
    }

    /// Every allocation of up to `max_k` drafts to each stream, scored by tokens per millisecond.
    fn exhaustive(cost: &RoundCost, d: &[Demand], max_k: usize, draft_sunk: bool) -> f64 {
        let n = d.len();
        let kv: u64 = d.iter().map(|x| x.kv_tokens).sum();
        let mut best = 0.0f64;
        let mut k = vec![0usize; n];
        loop {
            let tokens: f64 = d.iter().zip(&k).map(|(x, &c)| x.expected_tokens(c)).sum();
            let rows = n + k.iter().sum::<usize>();
            let shape = RoundShape {
                streams: n,
                rows,
                drafting: k.iter().filter(|&&c| c > 0).count(),
                head_rows: rows - n,
                kv_tokens: kv,
            };
            best = best.max(tokens / cost.ms(&shape, draft_sunk));
            // next vector, each digit 0..=min(max_k, drafts)
            let mut i = 0;
            loop {
                if i == n {
                    return best;
                }
                if k[i] < max_k.min(d[i].drafts) {
                    k[i] += 1;
                    break;
                }
                k[i] = 0;
                i += 1;
            }
        }
    }

    #[test]
    fn the_plan_is_near_the_exhaustive_optimum() {
        let mut r = 7u64;
        let mut rnd = |m: u64| {
            r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (r >> 33) % m
        };
        let mut exact = 0;
        let cases = 300;
        for _ in 0..cases {
            let n = 1 + rnd(4) as usize;
            let demands: Vec<Demand> = (0..n)
                .map(|_| {
                    let c: Vec<f64> = (0..5).map(|_| rnd(1000) as f64 / 1000.0).collect();
                    Demand::from_conditional(&c, 1 + rnd(5) as usize).with_kv_tokens(rnd(40_000))
                })
                .collect();
            let mut cost = round_cost();
            cost.kv_ms_per_token = 0.0003;
            for sunk in [false, true] {
                let p = plan_round(&cost, &demands, 64, sunk);
                let best = exhaustive(&cost, &demands, 5, sunk);
                let got = p.tokens / p.ms;
                assert!(got <= best * (1.0 + 1e-12), "the plan beat the exhaustive search?! {got} > {best}");
                assert!(got >= best * 0.99, "{demands:?} sunk {sunk}: plan {got} against optimum {best}");
                exact += (got >= best * (1.0 - 1e-12)) as usize;
            }
        }
        assert!(exact * 100 >= cases * 2 * 95, "only {exact} of {} plans were the exact optimum", cases * 2);
    }

    #[test]
    fn a_round_with_nothing_on_offer_plans_nothing() {
        let cost = round_cost();
        let p = plan_round(&cost, &[Demand::geometric(0.9, 0), Demand::geometric(0.9, 0)], 64, false);
        assert_eq!((p.rows, p.drafting, p.alloc), (2, 0, vec![0, 0]));
        assert_eq!(plan_round(&cost, &[], 64, false).rows, 0);
    }

    #[test]
    fn cost_tables_are_validated() {
        assert!(DraftCost::try_new(vec![], vec![(1, 1.0)], 0.0).unwrap_err().contains("draft layers"));
        assert!(DraftCost::try_new(vec![(1, 1.0)], vec![(2, 1.0)], 0.0).unwrap_err().contains("draft head"));
        assert!(DraftCost::try_new(vec![(1, 1.0)], vec![(1, 1.0)], -1.0).is_err());
        let d = DraftCost::new(vec![(1, 7.0)], vec![(1, 3.0)], 0.5);
        assert_eq!(d.ms(0, 0), 0.0);
        assert!((d.ms(1, 7) - 10.5).abs() < 1e-12);
    }

    fn draft_for_groups() -> (DraftCost, DraftCost) {
        // layers 5 ms + 0.5 a stream, a head that costs 4 ms whatever the rows (the measured shape at 1..16 streams)
        let layers: Vec<(usize, f64)> = (1..=16).map(|n| (n, 5.0 + 0.5 * n as f64)).collect();
        let head = vec![(1, 4.0), (128, 4.4)];
        (DraftCost::new(layers.clone(), head.clone(), 0.25), DraftCost::new(layers, head, 0.25).with_group(16))
    }

    #[test]
    fn a_draft_group_changes_nothing_up_to_its_width_bit_for_bit() {
        let (one_round, grouped) = draft_for_groups();
        for n in 0..=16 {
            for rows in [0, 1, 7, 16, 48, 64, 128] {
                assert_eq!(grouped.ms(n, rows).to_bits(), one_round.ms(n, rows).to_bits(), "{n} streams, {rows} head rows");
            }
        }
    }

    #[test]
    fn past_the_width_the_draft_costs_the_sum_of_its_rounds() {
        let (one_round, grouped) = draft_for_groups();
        // 17 streams: a round of 16 and a round of 1, the head rows shared by streams (rows 136 -> 128 + 8)
        let a = one_round.round_ms(16, 128);
        let b = one_round.round_ms(1, 8);
        assert_eq!(grouped.ms(17, 136).to_bits(), (a + b).to_bits());
        // 32 streams: two full rounds, the rows split evenly
        assert_eq!(grouped.ms(32, 128).to_bits(), (one_round.round_ms(16, 64) + one_round.round_ms(16, 64)).to_bits());
        // 33: three rounds. The cost never falls as streams join, and the 17th stream pays a whole extra round, not a stream's slope
        let mut prev = 0.0;
        for n in 1..=48 {
            let c = grouped.ms(n, n * 4);
            assert!(c >= prev, "{n} streams: {c} after {prev}");
            prev = c;
        }
        let jump = grouped.ms(17, 68) - grouped.ms(16, 64);
        assert!(jump >= 5.5 + 4.0, "the 17th stream starts a second draft round: +{jump} ms");
        assert!(one_round.ms(17, 68) - one_round.ms(16, 64) < 1.0, "the one-round model prices it at a stream's slope");
        assert!(grouped.ms(33, 132) - grouped.ms(32, 128) >= 9.5, "and the 33rd a third");
        // every head row is accounted for: no round is asked for more than the plan holds in all
        for rows in [0usize, 1, 2, 5, 31, 129] {
            assert!(grouped.ms(20, rows).is_finite() && grouped.ms(20, rows) > 0.0);
        }
    }

    #[test]
    fn the_planner_does_not_open_a_second_draft_round_for_a_stream_that_will_not_pay_for_it() {
        // 17 streams that each have one draft likely to land (0.9); the verify step is flat enough that every row is nearly free
        let verify = CostModel::from_table(vec![(1, 70.0), (64, 74.0)]);
        let demands = vec![Demand::geometric(0.9, 1); 17];
        let rc = |draft: DraftCost| RoundCost {
            verify: StepCost::rows_only(verify.clone()),
            tree_verify: None,
            draft: Some(draft),
            kv_ms_per_token: 0.0,
            kv_pass_ms_per_token: 0.0,
            fixed_ms: 0.0,
        };
        let (one_round, grouped) = draft_for_groups();
        let a = plan_round(&rc(one_round), &demands, 64, false);
        let b = plan_round(&rc(grouped), &demands, 64, false);
        assert_eq!(a.drafting, 17, "priced as one round the 17th stream is nearly free: {a:?}");
        assert_eq!(b.drafting, 16, "priced as a second round (9.75 ms for 0.9 token) it is not worth it: {b:?}");
        // up to 16 streams the two models plan the same round
        let few = vec![Demand::geometric(0.9, 1); 16];
        let (one_round, grouped) = draft_for_groups();
        assert_eq!(plan_round(&rc(one_round), &few, 64, false), plan_round(&rc(grouped), &few, 64, false));
    }

    #[test]
    fn huge_requests_are_refused_not_wrapped() {
        // kv_bytes_per_token = 65536: a max_new near u64::MAX / 65536 would wrap a plain multiplication to a small cost
        let mm =
            ModelMemory { weights_bytes: 0, kv_bytes_per_token: 65536, state_bytes_per_slot: 1 << 20, activation_reserve: 0 };
        let mut a = Admission::new(64 << 30, 0, mm, 4).unwrap();
        let wraps_to_small = (u64::MAX / 65536) + 2; // 65536 * this wraps to 65536 * 1 + ... in unchecked u64 math
        assert_eq!(a.admit(wraps_to_small), Err(AdmitError::TooLarge));
        assert_eq!(a.admit(u64::MAX), Err(AdmitError::TooLarge));
        // saturating cost: never smaller than a real one
        assert_eq!(a.cost(u64::MAX), u64::MAX);
        assert_eq!(a.available(), a.capacity(), "a refused request must reserve nothing");
        // the state bytes alone overflowing is also a refusal, not a wrap
        let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1, state_bytes_per_slot: u64::MAX, activation_reserve: 0 };
        let mut a = Admission::new(u64::MAX, 0, mm, 1).unwrap();
        assert_eq!(a.admit(1), Err(AdmitError::TooLarge));
    }

    #[test]
    fn construction_cannot_overflow() {
        let mm =
            ModelMemory { weights_bytes: u64::MAX, kv_bytes_per_token: 1, state_bytes_per_slot: 0, activation_reserve: u64::MAX };
        let a = Admission::new(100, u64::MAX, mm, 1).unwrap();
        assert_eq!(a.capacity(), 0);
        assert_eq!(a.available(), 0);
    }

    #[test]
    fn check_separates_never_from_not_now_and_can_admit_matches_admit() {
        let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1, state_bytes_per_slot: 0, activation_reserve: 0 };
        let mut a = Admission::new(100, 0, mm, 2).unwrap().with_max_context(80);
        // a request that fits the capacity can wait for memory; one above the whole capacity can never run, even on an idle server
        assert_eq!(a.check(80), Ok(()));
        assert_eq!(a.check(81), Err(AdmitError::ContextTooLong { tokens: 81, max: 80 }));
        let mut b = Admission::new(100, 0, ModelMemory { kv_bytes_per_token: 2, ..mm_one() }, 2).unwrap();
        assert_eq!(b.check(50), Ok(()));
        assert_eq!(b.check(51), Err(AdmitError::NeverFits { needed: 102, capacity: 100 }));
        assert_eq!(b.check(u64::MAX), Err(AdmitError::TooLarge));
        // `admit` keeps saying NoMemory for any shortfall; only `check` knows it is hopeless
        assert!(matches!(b.admit(51), Err(AdmitError::NoMemory { .. })));
        // can_admit is admit without the reservation
        assert_eq!(a.can_admit(60), Ok(60));
        assert_eq!(a.available(), 100, "can_admit reserved nothing");
        a.admit(60).unwrap();
        assert!(matches!(a.can_admit(60), Err(AdmitError::NoMemory { needed: 60, available: 40 })));
        assert_eq!(a.check(60), Ok(()), "not now is not never");
        a.admit(30).unwrap();
        assert_eq!(a.can_admit(1), Err(AdmitError::NoSlot));
    }

    fn mm_one() -> ModelMemory {
        ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1, state_bytes_per_slot: 0, activation_reserve: 0 }
    }

    #[test]
    fn context_limit_and_pinning() {
        let mut a = Admission::new(
            100,
            0,
            ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1, state_bytes_per_slot: 0, activation_reserve: 0 },
            4,
        )
        .unwrap()
        .with_max_context(50);
        assert_eq!(a.admit(51), Err(AdmitError::ContextTooLong { tokens: 51, max: 50 }));
        assert!(a.admit(50).is_ok());
        a.pin(30).unwrap();
        assert_eq!(a.available(), 20);
        assert!(matches!(a.admit(30), Err(AdmitError::NoMemory { needed: 30, available: 20 })));
        assert!(a.pin(21).is_err());
        a.unpin(30);
        assert!(a.admit(30).is_ok());
        a.unpin(u64::MAX); // over-unpinning cannot underflow
        assert_eq!(a.pinned(), 0);
        a.release(99); // an unknown slot is ignored, not a panic
    }

    // ---- the KV pool in tokens (serve hardening, finding 1): the byte capacity prices every sequence's slot state, but the state of ALL
    // slots is allocated up front, so with fewer sequences running the bytes of the idle slots look like room for KV that the pool
    // does not have. The numbers below are the shape of a real plan: a slot costs 1,000 tokens of KV, the pool holds 10,000.

    const POOL: u64 = 10_000;
    const SLOTS: usize = 8;

    fn pool_mm() -> ModelMemory {
        ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1000, state_bytes_per_slot: 1_000_000, activation_reserve: 0 }
    }

    /// Admission the way a plan builds it: capacity = the pool's KV bytes plus the state of every slot.
    fn plan_shaped() -> Admission {
        Admission::new(POOL * 1000 + SLOTS as u64 * 1_000_000, 0, pool_mm(), SLOTS).unwrap()
    }

    #[test]
    fn the_byte_capacity_alone_admits_more_tokens_than_the_pool_holds() {
        // this is the bug, pinned as a fact about the old rule: one sequence of 17,000 tokens costs 18,000,000 bytes, and the capacity is
        // 10,000,000 + 8,000,000 of slot state, so it is admitted into a pool of 10,000 tokens (OutOfPages at its prefill, minutes later)
        let mut a = plan_shaped().with_max_context(262_144);
        assert_eq!(a.check(17_000), Ok(()));
        assert!(a.admit(17_000).is_ok());
    }

    #[test]
    fn one_request_past_the_pool_is_refused_at_once_as_a_context_error() {
        let a = plan_shaped().with_max_context(262_144).with_kv_tokens(POOL);
        assert_eq!(a.check(POOL), Ok(()), "the pool's own size fits");
        assert_eq!(a.check(POOL + 1), Err(AdmitError::ContextTooLong { tokens: POOL + 1, max: POOL }));
        assert_eq!(a.check(17_000), Err(AdmitError::ContextTooLong { tokens: 17_000, max: POOL }));
        assert_eq!(a.can_admit(POOL + 1), Err(AdmitError::ContextTooLong { tokens: POOL + 1, max: POOL }));
        // the window still limits when it is the smaller one, and the error names the smaller
        let a = plan_shaped().with_max_context(4_000).with_kv_tokens(POOL);
        assert_eq!(a.check(4_001), Err(AdmitError::ContextTooLong { tokens: 4_001, max: 4_000 }));
        assert_eq!(a.request_limit(), Some(4_000));
        assert_eq!(plan_shaped().with_max_context(262_144).with_kv_tokens(POOL).request_limit(), Some(POOL));
        assert_eq!(plan_shaped().with_kv_tokens(POOL).request_limit(), Some(POOL));
        assert_eq!(plan_shaped().with_max_context(7).request_limit(), Some(7));
        assert_eq!(plan_shaped().request_limit(), None, "no limit declared: nothing to report");
    }

    #[test]
    fn two_requests_that_fit_alone_but_not_together_wait_and_neither_is_failed() {
        let mut a = plan_shaped().with_max_context(262_144).with_kv_tokens(POOL);
        // each is a legitimate request alone (6,000 of 10,000), and the bytes of the six idle slots would have let both in
        assert_eq!(a.check(6_000), Ok(()));
        let first = a.admit(6_000).unwrap();
        // the second is `NoMemory`: the engine's loop queues on exactly this (and on `NoSlot`) and fails anything else
        assert!(matches!(a.can_admit(6_000), Err(AdmitError::NoMemory { .. })), "{:?}", a.can_admit(6_000));
        assert!(matches!(a.admit(6_000), Err(AdmitError::NoMemory { .. })));
        assert_eq!(a.check(6_000), Ok(()), "not now is not never");
        // what fits beside it is still admitted: 4,000 more is exactly the pool
        assert_eq!(a.can_admit(4_001).map(|_| ()), Err(AdmitError::NoMemory { needed: 4_001 * 1000, available: 4_000 * 1000 }));
        let second = a.admit(4_000).unwrap();
        assert!(matches!(a.can_admit(1), Err(AdmitError::NoMemory { .. })));
        // the pool comes back when the running one ends, and then the queued one gets it
        a.release(first);
        assert!(a.admit(6_000).is_ok());
        a.release(second);
        assert_eq!(a.kv_tokens_in_use(), 6_000);
    }

    #[test]
    fn a_pool_sized_request_and_the_slot_count_are_independent_limits() {
        // every slot in use at 1,250 tokens is the pool exactly (and the bytes too): a ninth sequence waits whichever way it is counted
        let mut a = plan_shaped().with_kv_tokens(POOL);
        for _ in 0..SLOTS {
            a.admit(1_250).unwrap();
        }
        assert!(matches!(a.can_admit(1), Err(AdmitError::NoMemory { .. })));
        // with room left in the pool, it is the slot count that says no
        let mut a = plan_shaped().with_kv_tokens(POOL);
        for _ in 0..SLOTS {
            a.admit(500).unwrap();
        }
        assert_eq!(a.can_admit(1), Err(AdmitError::NoSlot));
        let mut a = plan_shaped().with_kv_tokens(POOL);
        for _ in 0..SLOTS - 1 {
            a.admit(1_250).unwrap();
        }
        assert!(matches!(a.can_admit(1_251), Err(AdmitError::NoMemory { .. })));
        assert!(a.can_admit(1_250).is_ok());
    }

    #[test]
    fn nothing_that_fits_the_pool_is_refused_by_the_token_budget() {
        // the pool rule only ever refuses more than the old rule did: with every sequence at most the pool, the sum under the pool is
        // admitted by both. Sweep a grid of (first, second) token counts and compare with the byte rule alone.
        for first in (500..=POOL).step_by(500) {
            for second in (500..=POOL).step_by(500) {
                let mut old = plan_shaped();
                let mut new = plan_shaped().with_kv_tokens(POOL);
                let (o1, n1) = (old.admit(first), new.admit(first));
                assert_eq!(o1.is_ok(), n1.is_ok(), "{first}");
                let (o2, n2) = (old.admit(second).is_ok(), new.admit(second).is_ok());
                assert!(!n2 || o2, "{first} + {second}: the pool rule admitted what the byte rule refused");
                assert_eq!(n2, first + second <= POOL, "{first} + {second}");
            }
        }
    }

    #[test]
    fn image_tokens_are_distinct_and_flagged() {
        let (a, b) = (1000u64, 1001u64); // differ only in bit 0: the old >> 1 mapped these to one token
        assert_ne!(image_token(a, 0), image_token(b, 0));
        assert_ne!(image_token(a, 0), image_token(a, 1));
        assert!(image_token(a, 0) & IMAGE_BASE != 0, "image tokens stay in the high half, apart from text ids");
        let h = image_content_hash(5, 448, 448, 1);
        assert_ne!(h, image_content_hash(5, 448, 224, 1), "size is part of the identity");
        assert_ne!(h, image_content_hash(5, 448, 448, 2), "preprocessing is part of the identity");
        assert_ne!(h, image_content_hash(6, 448, 448, 1));
    }

    #[test]
    fn cost_table_is_validated() {
        assert!(CostModel::try_from_table(vec![]).is_err());
        assert!(CostModel::try_from_table(vec![(2, 1.0)]).is_err(), "must start at 1 row");
        assert!(CostModel::try_from_table(vec![(1, 1.0), (4, 2.0), (4, 3.0)]).is_err(), "duplicate rows");
        assert!(CostModel::try_from_table(vec![(1, f64::NAN)]).is_err());
        assert!(CostModel::try_from_table(vec![(1, 0.0)]).is_err());
        // a noisy dip is flattened, so extrapolating past the table cannot go negative
        let c = CostModel::try_from_table(vec![(1, 70.0), (8, 90.0), (16, 60.0)]).unwrap();
        assert_eq!(c.ms(16), 90.0);
        assert!(c.ms(1000) >= 90.0);
    }

    #[test]
    fn nan_acceptance_gets_no_drafts_instead_of_a_panic() {
        let d = [Demand::geometric(f64::NAN, 7), Demand::geometric(0.5, 7)];
        let a = allocate(&d, 6);
        assert_eq!(a[0], 0);
        assert_eq!(a[1], 4);
        assert_eq!(expected_tokens(f64::NAN, 5), 1.0);
        let cost = CostModel::from_table(vec![(1, 70.0), (16, 74.0)]);
        let _ = choose_budget(&cost, &d, 16); // must not panic
    }

    /// The old search: a fresh `allocate` for every candidate budget.
    fn naive_budget(cost: &CostModel, demands: &[Demand], max_rows: usize) -> (usize, Vec<usize>) {
        let n = demands.len();
        let mut best = (n, vec![0; n], f64::NEG_INFINITY);
        for b in n..=max_rows.max(n) {
            let d = allocate(demands, b);
            let used = n + d.iter().sum::<usize>();
            if used < b {
                break;
            }
            let tokens: f64 = demands.iter().zip(&d).map(|(x, &k)| x.expected_tokens(k)).sum();
            let rate = tokens / cost.ms(used);
            if rate > best.2 {
                best = (used, d, rate);
            }
        }
        (best.0, best.1)
    }

    #[test]
    fn incremental_budget_equals_the_exhaustive_search() {
        let mut r = 99u64;
        let mut rnd = |m: u64| {
            r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (r >> 33) % m
        };
        let tables = [
            CostModel::from_table(vec![(1, 70.0), (16, 74.0), (32, 110.0), (64, 190.0)]),
            CostModel::from_table(vec![(1, 70.0), (8, 71.0), (64, 400.0)]),
            CostModel::from_table(vec![(1, 40.0), (4, 80.0), (12, 130.0)]),
        ];
        for _ in 0..400 {
            let n = 1 + rnd(8) as usize;
            let demands: Vec<Demand> = (0..n).map(|_| Demand::geometric(rnd(1000) as f64 / 1000.0, rnd(8) as usize)).collect();
            let max_rows = n + rnd(70) as usize;
            let cost = &tables[rnd(3) as usize];
            assert_eq!(choose_budget(cost, &demands, max_rows), naive_budget(cost, &demands, max_rows), "{demands:?} {max_rows}");
        }
    }
}
