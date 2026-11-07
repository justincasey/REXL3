//! The engine-facing contract of the composed model, and the options that size it. Wave 1 task E1 builds the backend, engine and
//! memory model on [`NemotronRuntime`]; task CMP implements it for `NemotronModel`. The trait mirrors `FlashNextRuntime` without the
//! prefix-cache and MTP calls, which arrive with M2 and M5 (additive: new methods get default bodies).

use crate::error::ModelError;
use crate::forward::Candidates;
use crate::rows::RowBatch;
use gb_engine::PrefixHandle;

/// Sizing for one loaded model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelOptions {
    /// Concurrent sequences (slots of the Mamba pools).
    pub slots: usize,
    /// KV blocks of 256 tokens in the pool.
    pub kv_pages: usize,
    /// The longest context a sequence may reach.
    pub max_context: usize,
    /// Rows of one prefill slice (at most [`PREFILL_ROWS_MAX`]; above [`WINDOW_ROWS`] the MoE layers take the large-row prefill path,
    /// whose scratch the model then allocates).
    pub prefill_rows: usize,
    /// Load only the first `n` trunk layers (the layer-limited oracle and tests); `None` is the whole model.
    pub layers: Option<usize>,
    /// Load the MTP head and give it its KV stripe and tap stores. The plan prices the head exactly when this is true (`engine::head_priced`),
    /// so the pages it plans match what is allocated. Ignored for a layer-limited model, a pack without a head and `REXL3_MTP=off`.
    pub mtp: bool,
}

/// The narrow adapter the host scheduling contract runs against, so it works without a GPU or a pack.
pub trait NemotronRuntime {
    fn vocab_size(&self) -> usize;
    fn has_mtp(&self) -> bool;
    /// The form of the loaded MTP head's projections, which keys the arbiter's draft-call tables (`spec::CostTables::shipped_for`):
    /// bf16 unless the runtime holds a 4-bit head.
    fn head_form(&self) -> super::spec::HeadForm {
        super::spec::HeadForm::Bf16
    }
    /// Zeroes the slot's recurrent state and KV length. An error leaves the slot unusable.
    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError>;
    /// Releases the slot. Idempotent.
    fn end_slot(&mut self, slot: usize);
    /// One prefill slice of at most `ModelOptions::prefill_rows` rows. `want` is the row (within the batch) whose candidates the
    /// caller needs, or `None` for none; the state is committed at `keep = rows`.
    fn prefill_chunk(&mut self, b: &RowBatch, want: Option<usize>) -> Result<Option<Candidates>, ModelError>;
    /// [`NemotronRuntime::prefill_chunk`] for a slice with image rows: `splice` writes rows of an image's features over embedding rows of
    /// the slice (batch rows). Additive: the default serves an empty splice as `prefill_chunk` and refuses any other by name, so a runtime
    /// that cannot splice (a fake) stays a text runtime.
    fn prefill_chunk_mm(
        &mut self,
        b: &RowBatch,
        want: Option<usize>,
        splice: &[super::vision::SpliceOp],
    ) -> Result<Option<Candidates>, ModelError> {
        if splice.is_empty() {
            return self.prefill_chunk(b, want);
        }
        Err(images_not_built())
    }
    /// A verify or decode window: candidates for every row, state untouched until `commit`.
    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError>;
    /// [`NemotronRuntime::window_candidates`] for a decode round in which every single-row sequence will be committed with `keep = 1`
    /// whatever it samples (the engine's contract for a sequence that drafted nothing): the runtime may advance such a sequence's state in
    /// the window itself, so that its `commit` has no state left to move and the state is read and written once instead of three times.
    /// Candidates, the state after the commit and the slot's length are those of the plain window then `commit`; a caller that may NOT
    /// commit a one-row sequence (it rolls it back, or runs the same position again) must use `window_candidates`. Additive: the default
    /// is the plain window, which is correct for every runtime.
    fn window_candidates_fused(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        self.window_candidates(b, count)
    }
    /// Whether [`NemotronRuntime::prefill_chunk_rows`] is built: the backend then puts the next slices of several waiting prompts through
    /// one forward (`Backend::prefill_slices`). Additive: a runtime that keeps the default is asked for one slice at a time.
    fn prefill_rows_supported(&self) -> bool {
        false
    }
    /// [`NemotronRuntime::prefill_chunk`] that wants the candidates of several rows of the slice (`wants`, batch rows; the result's row `i`
    /// is the candidates of row `wants[i]`, with the bits `prefill_chunk(.., Some(wants[i]))` gives it). `wants` empty: no logits.
    fn prefill_chunk_rows(&mut self, _b: &RowBatch, _wants: &[usize]) -> Result<Option<Candidates>, ModelError> {
        Err(ModelError::NotBuilt("nemotron prefill of several prompts in one forward".into()))
    }
    /// Keeps the first `keep` rows (1..=window) of the slot's last window: the Mamba state advances by exactly those rows.
    fn commit(&mut self, slot: usize, keep: usize);
    /// Waits for the device and returns any error a stream recorded.
    fn flush(&mut self) -> Result<(), ModelError>;

    // ---- the MTP head (milestone M2). Defaults: a runtime without a head drafts nothing. Additive.

    /// Runs the head for each ask: absorbs the slot's pending committed rows in one head window, then chains `depth` drafts. Entry `i`
    /// of the result answers ask `i`; `None` means the slot drafts nothing this round (drafting off for the slot, or no pending rows).
    /// `keyed[i]` is the request's sampler when keyed-noise drafting is on for it (a sampled request), else `None` (the head's argmax).
    /// Touches only the head's own cache and scratch, never the trunk's state.
    fn mtp_draft(
        &mut self,
        _asks: &[super::mtp::DraftAsk<'_>],
        _keyed: &[Option<gb_sampling::Sampling>],
    ) -> Result<Vec<Option<super::mtp::Drafts>>, ModelError> {
        Err(ModelError::NotBuilt("nemotron mtp head".into()))
    }

    /// Switches the slot's drafting off for the rest of its sequence (never an error: the slot just drafts nothing).
    fn mtp_disable_slot(&mut self, _slot: usize) {}

    /// Committed rows of `slot` the head has not absorbed yet, or `None` when the slot does not draft.
    fn mtp_pending_rows(&self, _slot: usize) -> Option<usize> {
        None
    }

    /// The CUDA-graph cache's counters (windows and the head's draft calls), for `/metrics`; `None` (the default) for a runtime without graphs.
    fn graph_counters(&self) -> Option<gb_engine::GraphCounters> {
        None
    }

    // ---- the prefix cache (milestone M5; design and the GPU worker's brief: `docs/NEMOTRON_LONG_CONTEXT.md`). Additive: a runtime that keeps
    // the defaults cannot snapshot, and the plan never carves a cache for it.

    /// Whether this runtime TYPE implements the snapshot methods below. The memory plan is made before a model exists, so the question is
    /// asked of the type, not of an instance. A runtime that answers `true` must implement every method of this group.
    fn snapshots_supported() -> bool
    where
        Self: Sized,
    {
        false
    }

    /// Allocates the snapshot arena: `slots` snapshot slots of [`super::prefix::snapshot_bytes`] bytes each, once, at boot (the plan counted
    /// them). Called once, after the model exists and before any prompt.
    fn enable_snapshots(&mut self, _slots: usize) -> Result<(), ModelError> {
        Err(snapshots_not_built())
    }

    /// Device bytes of the arena `enable_snapshots` allocated (0 before it): the boot compares it with the plan.
    fn snapshot_arena_bytes(&self) -> u64 {
        0
    }

    /// Snapshot slots no snapshot holds right now.
    fn free_snapshots(&self) -> usize {
        0
    }

    /// Snapshots `slot`'s state after exactly `len` tokens: the slot must be live with `len` tokens committed and nothing owed (a prefill
    /// slice has just ended). Copies the Mamba state and the head's carry into a free arena slot and takes a reference to pages
    /// `0..ceil(len / 256)`. `Ok(None)`: no free snapshot slot (the cut is skipped, not an error). On `Err` nothing stays allocated.
    fn snapshot_slot(&mut self, _slot: usize, _len: usize) -> Result<Option<PrefixHandle>, ModelError> {
        Err(snapshots_not_built())
    }

    /// Puts a slot that has just been begun (empty) in the state `handle` holds for `len` tokens: pages `0..last` by reference, the page
    /// `last = (len - 1) / 256` copied in every stripe the head's included ([`super::prefix::restore_pages`]), the Mamba state, the carry in
    /// the head's store row 0, the head's book as `{ ok, first: len, n: 1 }`, the position `len`. A handle that is not live, a `len` that is
    /// not the snapshot's or a slot that is not empty is refused by name and changes nothing.
    fn restore_snapshot(&mut self, _slot: usize, _handle: &PrefixHandle, _len: usize) -> Result<(), ModelError> {
        Err(snapshots_not_built())
    }

    /// Frees the arena slot and drops the page references. Exactly once per handle: a second release is an error by name.
    fn release_snapshot(&mut self, _handle: PrefixHandle) -> Result<(), ModelError> {
        Err(snapshots_not_built())
    }
}

fn snapshots_not_built() -> ModelError {
    ModelError::NotBuilt(
        "nemotron prefix snapshots (this runtime has not built snapshot_slot, restore_snapshot and release_snapshot)".into(),
    )
}

// ------------------------------------------------------------------------------------------------------------- the backend
//
// Everything below is the scheduler-facing side: `NemotronBackend` drives any [`NemotronRuntime`] as a `gb_engine::Backend`. It owns
// no device state. It slices prompts on an absolute grid, keeps the slot and window bookkeeping the runtime's contract states in prose
// (a slot is begun before it is used and ended once; a commit keeps rows of a window that exists), and turns a runtime error into the
// latched error the engine asks for.

use super::model::NemotronModel;
use super::prefix;
use super::spec::{CostTables, SpecConfig, SpecRuntime};
use super::vision::{plan_splice, NemotronImageFeatures, SpliceOp};
use gb_engine::{
    Backend, BackendError, CacheBudget, Captured, CutStep, ImageSpan, PrefillJob, PrefillStep, PrefixCaps, Proposal,
    ProposeRequest, RowOutput, RowRequest,
};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

/// Rows of the widest verify window, and so the most rows a `commit` may keep. One window is one pass of the Mamba window kernels
/// (`nemotron_ssm::window`, verified for 1..=64 rows) and of the shared MoE scratch.
pub const WINDOW_ROWS: usize = 64;
/// The opt-out prefill slice cap (`model::MAX_PREFILL_ROWS`, 256; `REXL3_NEMO_PREFILL_ROWS_MAX=256`). The cap a process runs with is
/// [`prefill_rows_max`] (512 by default): above the 64 rows of a window the MoE layers run the large-row prefill path (package P1), which
/// reads each expert's weights once for the slice. The engine clamps a larger `--prefill-chunk` to it with a log line
/// (`engine::prefill_rows`) and the backend clamps to it again.
pub const PREFILL_ROWS_MAX: usize = 256;

/// The most rows one prefill slice may carry in this process: 512 by default (`model::WIDE_PREFILL_ROWS`, since 2026-10-15), or
/// [`PREFILL_ROWS_MAX`] (256) under the opt-out `REXL3_NEMO_PREFILL_ROWS_MAX=256` (`model::prefill_rows_cap`, which has the rules).
/// Under the prefill contract P2 (`contract`, not built yet: refused at load) the cap is P2's.
pub fn prefill_rows_max() -> usize {
    super::contract::current().max_prefill_rows(super::model::prefill_rows_cap())
}
/// The most candidates per row the runtime returns (the `topk_ids` and `topk_vals` buffers of `model::Bufs` are sized for it).
pub const MAX_CANDIDATES: usize = 64;

/// A prefill slice that resumes at `done` ends at the next multiple of `chunk`, not at `done + chunk`: the cuts of a prompt then sit on
/// one absolute grid whatever the pieces it arrived in, which is what lets the runtime make "chunked equals whole" a row-invariance
/// property instead of a bookkeeping one.
pub fn next_chunk_end(done: usize, total: usize, chunk: usize) -> usize {
    done.saturating_add(chunk - done % chunk).min(total)
}

fn nan_row(n: usize) -> RowOutput {
    RowOutput { ids: (0..n as u32).collect(), values: vec![f32::NAN; n] }
}

/// Row `r` of `c` as the engine's `RowOutput`, its first `n` candidates (they are sorted, so the first `n` of a longer list are the top
/// `n`). A runtime that returned fewer candidates than the sampler needs, or a malformed block, is a contract violation reported by
/// name, never an out-of-range slice on the engine thread.
fn row_of(c: &Candidates, r: usize, n: usize) -> Result<RowOutput, ModelError> {
    let lo = r.checked_mul(c.count);
    let hi = r.checked_add(1).and_then(|r1| r1.checked_mul(c.count));
    let (Some(lo), Some(hi)) = (lo, hi) else {
        return Err(ModelError::Request(format!(
            "the runtime's candidates have {} rows of {}: row {r} overflows",
            c.rows, c.count
        )));
    };
    if r >= c.rows || c.ids.len() < hi || c.vals.len() < hi {
        return Err(ModelError::Request(format!(
            "the runtime's candidates are malformed: {} rows of {} (ids {}, values {}), row {r} asked for",
            c.rows,
            c.count,
            c.ids.len(),
            c.vals.len()
        )));
    }
    if c.count < n {
        return Err(ModelError::Request(format!("the runtime returned {} candidates per row, the sampler needs {n}", c.count)));
    }
    Ok(RowOutput { ids: c.ids[lo..lo + n].iter().map(|&x| x as u32).collect(), values: c.vals[lo..lo + n].to_vec() })
}

/// Whether none of the `rows` positions from `first` is in one of `spans` (the image spans of a prompt): the predicate behind
/// [`NemotronBackend::spec_allowed_for_window`], free of the backend so the drafting runtime can ask it with the spans it is handed.
pub fn window_is_text(spans: &[Range<usize>], first: usize, rows: usize) -> bool {
    let end = first.saturating_add(rows);
    !spans.iter().any(|s| s.start < end && first < s.end)
}

fn images_not_built() -> ModelError {
    ModelError::NotBuilt(
        "this Nemotron runtime cannot splice image features (nemotron_radio_vision is the model's, not this runtime's)".into(),
    )
}

/// The dense prefill rate the floor assumes, tokens a second: 1.25 x the fastest measured through the server (520 tok/s at 256-row slices,
/// `docs/NEMOTRON_ATTN_LONG.md`; 525 to 544 at 28,818 tokens in the window-7/8 runs).
pub const FLOOR_DENSE_TOK_S: f64 = 650.0;
/// [`FLOOR_DENSE_TOK_S`] under the prefill contract P2 (the default since rc2d): 1.25 x the fastest P2 prefill measured through the
/// server, so that the floor stays a lower bound under P2 too (P1's 650 would refuse P2 prompts that fit their deadline).
/// Re-fitted after the integrated measurement on rc2d (`docs/NEMOTRON_PREFILL_SPEED.md` 6.16): 1,794.2 tok/s, lane S6's fastest run,
/// one node, the rc2d build (95b0cec4) with the qualified profile's serve flags, 10,833 prompt tokens to the first token in 6.038 s with
/// the prefix-cache store opted out per request. The same profile at TP=2 measured 1,722.8 tok/s (10,833 tokens) and 1,685.6 (28,818),
/// medians of 3, so the fastest run is a single-node one. Was 1.25 x 1,750 in rc2d before the re-fit, 1.25 x 1,607 in rc2b/rc2c and
/// 1.25 x 1,434 in rc2. P1 (`REXL3_NEMO_PREFILL_CONTRACT=p1`) keeps [`FLOOR_DENSE_TOK_S`].
pub const FLOOR_DENSE_TOK_S_P2: f64 = 1.25 * 1794.2;
/// The flash attention cost the floor assumes, seconds per prompt row per attention layer per 1,048,576 positions of depth: half of the measured
/// 0.2787 ms (71.347 ms for 256 rows at 1M depth, `docs/NEMOTRON_ATTN_LONG.md`; linear in depth, the rate is flat), so that a flash kernel up
/// to twice as fast keeps the bound a bound.
pub const FLOOR_ATTN_S_PER_ROW_LAYER_1M: f64 = 0.5 * 0.000_278_7;

/// A LOWER bound on the seconds a prefill of positions `from..to` takes on one GB10 (a tensor-parallel group is no faster: `docs/NEMOTRON_TP.md`,
/// the allreduce cancels the halved GEMMs), with `layers` attention layers (8, 9 with the head): the dense part at [`FLOOR_DENSE_TOK_S`] plus the
/// flash attention, linear in depth, at [`FLOOR_ATTN_S_PER_ROW_LAYER_1M`]. Optimistic by construction (the tree below 8,192 positions, the tower,
/// the snapshot copies and every co-tenant are left out): the admission refuses a request on it only when even this cannot fit the deadline
/// (`docs/LONG_PREFILL_POLICY.md`). At 1M from cold it is 2,270 s against a modelled 3,332 s.
pub fn prefill_floor_secs(from: usize, to: usize, layers: usize) -> f64 {
    prefill_floor_secs_at(from, to, layers, FLOOR_DENSE_TOK_S)
}

/// [`prefill_floor_secs`] at the dense rate `dense_tok_s` ([`FLOOR_DENSE_TOK_S`] under P1, [`FLOOR_DENSE_TOK_S_P2`] under P2).
pub fn prefill_floor_secs_at(from: usize, to: usize, layers: usize, dense_tok_s: f64) -> f64 {
    if to <= from {
        return 0.0;
    }
    let (a, b) = (from as f64, to as f64);
    let dense = (b - a) / dense_tok_s;
    // sum over the rows of (depth / 1M) x per-row cost: the integral of a linear function
    let attn = layers as f64 * FLOOR_ATTN_S_PER_ROW_LAYER_1M * (b * b - a * a) / (2.0 * 1_048_576.0);
    dense + attn
}

/// The engine's backend over a Nemotron runtime.
///
/// Bookkeeping, all of it per slot:
/// * `live`: slots `begin_slot` succeeded for and nothing has ended. A live slot is ended exactly once, by `release` or by the error
///   path that gave it up, whichever comes first; a slot nobody began is never ended (the one exception is the half-begun slot of a
///   failed `begin_slot`, ended once where it failed). The runtime's own idempotence is never relied on.
/// * `windows`: the rows of the slot's last window that no `commit` has consumed. A commit keeps `1..=windows[slot]` of them and
///   consumes the window, so a commit can neither advance a state by rows that were never verified nor advance it twice.
///
/// **Speculation** (`spec.rs`): a backend built `with_speculation` answers `Backend::propose_batch_scored` itself (the MTP head through
/// the runtime's `mtp_draft`, one call a round for every stream that wants head drafts, and the copy drafter), keeps the per-slot record
/// of what each round offered (`on_step`, `on_commit`) and the telemetry the measurements read (`spec.stats`). Without it the backend
/// answers "no drafts" and the engine falls back to its own per-sequence drafter. A head error never fails a request: its slots stop
/// drafting and the round goes on (`SpecRuntime::propose`). The backend does no head work in `commit`: the runtime moves the tapped rows
/// in its own `commit`.
///
/// **The prefix cache** (M5, `docs/NEMOTRON_LONG_CONTEXT.md`): a backend built `with_prefix_cache` answers `prefix_caps`, `prefix_path` (the id of the
/// arithmetic the prompt's rows run on by the position rule, `prefix::path_of`), `prefix_restore`, `prefix_release` and `prefill_slice_cuts` over the runtime's snapshot methods. A cut ends
/// its slice (the grid after it is unchanged, `prefix::slice_end`) and the snapshot is taken after the slice; with no cuts `prefill_slice_cuts` is
/// the slice `prefill_slice` runs (one routine). Without it the defaults mean "never cached".
pub struct NemotronBackend<M: NemotronRuntime = NemotronModel> {
    pub m: M,
    chunk: usize,
    slots: usize,
    deadline_rows: Option<usize>,
    /// The most rows of one window (all sequences): [`WINDOW_ROWS`], or the wide model's (`with_window_rows`).
    window_rows: usize,
    /// Rows the runtime's prefill takes at once (`ModelOptions::prefill_rows`, at most [`PREFILL_ROWS_MAX`]): the slices of several prompts
    /// share a forward up to this many (`prefill_slices`).
    pack_rows: usize,
    /// Run a step through `window_candidates_fused`: a sequence of one row has its state advanced inside the window, and the engine's
    /// `commit(slot, 1)` that follows moves nothing but the slot's length and the head's taps (`with_fused_decode`).
    fuse_decode: bool,
    live: BTreeSet<usize>,
    windows: BTreeMap<usize, usize>,
    error: Option<BackendError>,
    fatal: Option<BackendError>,
    /// The drafters and their telemetry; `None`: the backend offers no drafts of its own.
    pub spec: Option<SpecRuntime>,
    /// Per slot, the positions of the prompt that hold image rows, from the start of its prefill until the slot is given up (set on every
    /// slice, not only the first). A window over them is refused by name, and the head asks it before it absorbs rows from its tap store.
    slot_images: BTreeMap<usize, Vec<Range<usize>>>,
    /// The id the placeholders of an image span must have (`VisionSpec::image_token_id`); `None` checks only that a span's tokens agree.
    image_token: Option<u32>,
    /// The prefix cache's namespace and budget, when the backend has one (`with_prefix_cache`).
    prefix: Option<(u64, CacheBudget)>,
}

impl NemotronBackend<NemotronModel> {
    /// The backend of a loaded model, sized as the model was (`m.core.opts`). `prefill_chunk` is the serving option; it is clamped to
    /// what the model was built for.
    pub fn new(m: NemotronModel, prefill_chunk: usize) -> Result<Self, ModelError> {
        let opts = m.core.opts.clone();
        Self::with_runtime(m, prefill_chunk, &opts)
    }
}

impl<M: NemotronRuntime> NemotronBackend<M> {
    /// A backend over `m`, which was built with `opts`. A prefill slice has at most `min(prefill_chunk, opts.prefill_rows, 64)` rows. A
    /// zero in the options that size it is refused by name, not asserted on the engine thread.
    pub fn with_runtime(m: M, prefill_chunk: usize, opts: &ModelOptions) -> Result<Self, ModelError> {
        if prefill_chunk == 0 {
            return Err(ModelError::Config("the prefill chunk must be at least 1 row".into()));
        }
        if opts.prefill_rows == 0 || opts.slots == 0 {
            return Err(ModelError::Config(format!(
                "the model options need at least one slot and one prefill row, got {} slots and {} prefill rows",
                opts.slots, opts.prefill_rows
            )));
        }
        Ok(Self {
            m,
            chunk: prefill_chunk.min(opts.prefill_rows).min(prefill_rows_max()),
            slots: opts.slots,
            deadline_rows: None,
            window_rows: WINDOW_ROWS,
            pack_rows: opts.prefill_rows.min(prefill_rows_max()),
            fuse_decode: false,
            live: BTreeSet::new(),
            windows: BTreeMap::new(),
            error: None,
            fatal: None,
            spec: None,
            slot_images: BTreeMap::new(),
            image_token: None,
            prefix: None,
        })
    }

    /// The runtime's decode window takes up to `rows` rows (a wide model, `NemotronModel::new_wide`): a step and a commit are checked
    /// against it instead of [`WINDOW_ROWS`]. Refused by name outside `WINDOW_ROWS..=MAX_WINDOW_ROWS` (`model::check_window_rows`).
    pub fn with_window_rows(mut self, rows: usize) -> Result<Self, ModelError> {
        super::model::check_window_rows(rows)?;
        self.window_rows = rows;
        if let Some(s) = self.spec.as_mut() {
            s.cfg.window_rows = rows;
        }
        Ok(self)
    }

    /// Rows of the widest window this backend runs.
    pub fn window_rows(&self) -> usize {
        self.window_rows
    }

    /// Steps go through `NemotronRuntime::window_candidates_fused`: the engine commits `keep = 1` for every sequence of a one-row window
    /// (it drafted nothing, so there is nothing to roll back), and the runtime advances such a sequence's state in the window itself.
    /// Off by default: a caller that steps a slot and does not commit it (the rollback tests) must not have its state moved.
    pub fn with_fused_decode(mut self, on: bool) -> Self {
        self.fuse_decode = on;
        self
    }

    pub fn fused_decode(&self) -> bool {
        self.fuse_decode
    }

    /// The prompts of this backend carry images whose placeholders are token `id`: a span over other tokens is refused by name.
    pub fn with_image_token(mut self, id: u32) -> Self {
        self.image_token = Some(id);
        self
    }

    /// This backend keeps a prefix cache of `budget` (snapshot slots and pages the plan carved out; the runtime's arena must hold that many
    /// slots: `NemotronRuntime::enable_snapshots`) in `namespace` (`prefix::namespace_of`). Refused by name for a runtime that cannot
    /// snapshot and for a budget that holds no entry; nothing is cached without it.
    pub fn with_prefix_cache(mut self, namespace: u64, budget: CacheBudget) -> Result<Self, ModelError> {
        if !M::snapshots_supported() {
            return Err(ModelError::NotBuilt(
                "the prefix cache: this runtime cannot snapshot a slot (NemotronRuntime::snapshots_supported is false)".into(),
            ));
        }
        if !budget.enabled() {
            return Err(ModelError::Config("the prefix cache needs a budget of at least one snapshot slot".into()));
        }
        self.prefix = Some((namespace, budget));
        Ok(self)
    }

    /// This backend drafts by `cfg` with the arbiter priced by `costs` (`CostTables::shipped()`, or the measured tables): its
    /// `propose_batch_scored` returns every asking stream's drafts, an empty list for none. A malformed table is refused by name.
    pub fn with_speculation(mut self, cfg: SpecConfig, costs: &CostTables) -> Result<Self, ModelError> {
        let mut spec = SpecRuntime::new(SpecConfig { window_rows: self.window_rows, ..cfg }, self.slots);
        spec.set_costs(costs).map_err(ModelError::Config)?;
        self.spec = Some(spec);
        Ok(self)
    }

    /// Whether speculation may run over the `rows` positions from `first_position` of `slot`: false when any of them holds an image
    /// row. The head asks it about the rows it is about to absorb from the tap store: an image's rows enter the head's cache from its
    /// features, in the prompt's slices (`mtp_prefill_hook`), and a store of pending rows that holds one can only have the placeholder's
    /// embedding to offer, so the slot drafts nothing then, never a placeholder.
    pub fn spec_allowed_for_window(&self, slot: usize, first_position: usize, rows: usize) -> bool {
        window_is_text(self.slot_images.get(&slot).map_or(&[], |v| v.as_slice()), first_position, rows)
    }

    /// Rows of the widest prefill slice this backend runs (before any deadline limit).
    pub fn prefill_rows(&self) -> usize {
        self.chunk
    }

    /// Slots begun and not yet ended.
    pub fn live_slots(&self) -> Vec<usize> {
        self.live.iter().copied().collect()
    }

    /// Rows of `slot`'s last window that no commit has consumed.
    pub fn pending_rows(&self, slot: usize) -> Option<usize> {
        self.windows.get(&slot).copied()
    }

    fn record(&mut self, context: &str, e: &ModelError) {
        let message = format!("{context}: {e}");
        if e.is_fatal() {
            self.fatal.get_or_insert_with(|| BackendError::fatal(message));
        } else {
            self.error.get_or_insert_with(|| BackendError::retryable(message));
        }
    }

    fn dead(&self) -> bool {
        self.fatal.is_some()
    }

    /// Gives `slot` up: forgets its bookkeeping and, if it was live, ends it in the runtime. Once: the slot is no longer live afterwards, so
    /// the engine's `release` that follows an error is a no-op, and a slot nobody began is never ended. After a fatal error the runtime is
    /// left alone, as it cannot be trusted to answer.
    fn drop_slot(&mut self, slot: usize) {
        self.windows.remove(&slot);
        self.slot_images.remove(&slot);
        if let Some(s) = self.spec.as_mut() {
            s.end(slot);
        }
        if self.live.remove(&slot) && !self.dead() {
            self.m.end_slot(slot);
        }
    }

    fn effective_chunk(&self) -> usize {
        self.chunk.min(self.deadline_rows.unwrap_or(usize::MAX)).max(1)
    }

    fn check_slot(&self, slot: usize) -> Result<(), ModelError> {
        if slot >= self.slots {
            return Err(ModelError::Request(format!("slot {slot} is out of range: the model has {} slots", self.slots)));
        }
        Ok(())
    }

    fn check_count(candidates: usize) -> Result<(), ModelError> {
        if candidates == 0 || candidates > MAX_CANDIDATES {
            return Err(ModelError::Request(format!("candidate count must be 1..={MAX_CANDIDATES}, got {candidates}")));
        }
        Ok(())
    }

    /// Positions travel as `i32` in a `RowBatch`.
    fn check_position(end: usize) -> Result<(), ModelError> {
        if end > i32::MAX as usize {
            return Err(ModelError::Request(format!("position {end} does not fit the row index (positions are 32-bit)")));
        }
        Ok(())
    }

    /// Starts `slot` over: whatever it held is dropped (a prompt re-prefilled at position 0 replaces the sequence in the slot, as the
    /// harness and a reused backend do), the runtime zeroes the state, and only a success registers the slot. A slot that is live is
    /// ENDED in the runtime first, once: a runtime refuses to begin a slot that is already begun (the engine releases before it reuses
    /// a slot, but a harness or a reused backend does not). A failure leaves nothing registered; the runtime may hold the half-begun
    /// slot, so it is ended here, once, and no later `release` ends it again.
    fn begin(&mut self, slot: usize) -> Result<(), ModelError> {
        self.check_slot(slot)?;
        self.windows.remove(&slot);
        self.slot_images.remove(&slot);
        if self.live.remove(&slot) && !self.dead() {
            self.m.end_slot(slot);
        }
        match self.m.begin_slot(slot) {
            Ok(()) => {
                self.live.insert(slot);
                if let Some(s) = self.spec.as_mut() {
                    s.begin(slot);
                }
                Ok(())
            }
            Err(e) => {
                if !e.is_fatal() {
                    self.m.end_slot(slot);
                }
                Err(e)
            }
        }
    }

    /// A prefill that continues `slot` must find it live: a slot whose first slice failed, or that was released, has no state to continue.
    fn resume(&self, slot: usize, at: usize) -> Result<(), ModelError> {
        self.check_slot(slot)?;
        if !self.live.contains(&slot) {
            return Err(ModelError::Request(format!("slot {slot} has no prefill in progress (resumed at position {at})")));
        }
        Ok(())
    }

    /// One prefill slice of `slot`: `tokens` at `first..`, with the candidates of its last row when `candidates` is set (only the final
    /// slice of a prompt or piece sets it). The runtime commits the slice's state itself.
    fn run_chunk(
        &mut self,
        slot: usize,
        first: usize,
        tokens: &[u32],
        candidates: Option<usize>,
        splice: &[SpliceOp],
    ) -> Result<Option<RowOutput>, ModelError> {
        // the MTP head's pass over this slice is the runtime's (`prefill_chunk`'s hook), with this slice's splice for its image rows
        let b = RowBatch::from_requests(&[RowRequest { slot, first_position: first as u64, tokens: tokens.to_vec() }]);
        let want = candidates.map(|_| b.rows - 1);
        let c = self.m.prefill_chunk_mm(&b, want, splice)?;
        let Some(n) = candidates else { return Ok(None) };
        let c = c.ok_or_else(|| ModelError::Request("the runtime returned no candidates for the final prefill slice".into()))?;
        if c.rows != 1 {
            return Err(ModelError::Request(format!(
                "the runtime returned {} candidate rows for the one row a prefill slice asks for",
                c.rows
            )));
        }
        row_of(&c, 0, n).map(Some)
    }

    /// A piece of a prompt of `total` tokens, cut on the absolute grid. The final slice of the piece asks for candidates.
    fn prefill_inner(
        &mut self,
        slot: usize,
        start: u64,
        tokens: &[u32],
        total: usize,
        candidates: usize,
    ) -> Result<RowOutput, ModelError> {
        let first = usize::try_from(start).map_err(|_| ModelError::Request("prefill position overflows usize".into()))?;
        let piece_end = match first.checked_add(tokens.len()) {
            Some(n) if !tokens.is_empty() && n <= total => n,
            _ => return Err(ModelError::Request("empty prefill piece or piece past prompt total".into())),
        };
        Self::check_count(candidates)?;
        Self::check_position(total)?;
        if first == 0 {
            self.begin(slot)?;
        } else {
            self.resume(slot, first)?;
        }
        self.m.flush()?;
        let mut at = first;
        let mut last = None;
        while at < piece_end {
            let end = next_chunk_end(at, piece_end, self.effective_chunk());
            let want = (end == piece_end).then_some(candidates);
            last = self.run_chunk(slot, at, &tokens[at - first..end - first], want, &[])?;
            at = end;
        }
        last.ok_or_else(|| ModelError::Request("prefill produced no candidates".into()))
    }

    fn prefill_slice_inner(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
    ) -> Result<PrefillStep, ModelError> {
        self.slice_with_cuts(slot, prompt, images, done, candidates, &[]).map(|c| c.step)
    }

    /// One prefill slice of `slot` that also takes the snapshot of each of `cuts` that lies at the slice's end: the slice ends at the first cut
    /// inside it (`prefix::slice_end`; the grid after the cut is the grid it would have had) and the snapshot is taken after the slice, when
    /// the Mamba state and the head's carry are those of exactly `end` tokens. With no cuts (or no cache) it is the slice `prefill_slice` runs.
    fn slice_with_cuts(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
        cuts: &[usize],
    ) -> Result<CutStep, ModelError> {
        if done >= prompt.len() {
            return Err(ModelError::Request("empty or completed prefill slice".into()));
        }
        // the images are checked on every slice (it is cheap), before any state of the slot is touched: a bad one fails the sequence
        let feats = self.image_features(prompt, images)?;
        Self::check_count(candidates)?;
        Self::check_position(prompt.len())?;
        if done == 0 {
            self.begin(slot)?;
        } else {
            self.resume(slot, done)?;
        }
        // set on every slice, after `begin` (which forgets the slot's spans): the window predicate and the head's absorb read them
        if images.is_empty() {
            self.slot_images.remove(&slot);
        } else {
            self.slot_images.insert(slot, images.iter().map(|im| im.start..im.start + im.tokens).collect());
        }
        self.m.flush()?;
        // a backend without a cache takes no snapshot, whatever the caller plans
        let cuts = if self.prefix.is_some() { cuts } else { &[] };
        let end = prefix::slice_end(done, prompt.len(), self.effective_chunk(), cuts);
        let splice = Self::splice_for(images, &feats, done, end);
        let row = self.run_chunk(slot, done, &prompt[done..end], (end == prompt.len()).then_some(candidates), &splice)?;
        let mut captured = Vec::new();
        // a cut at the prompt's end is no use (a strict prefix leaves a row to compute); a full arena skips the cut
        if end < prompt.len() && cuts.contains(&end) {
            if let Some(handle) = self.m.snapshot_slot(slot, end)? {
                captured.push(Captured { len: end, handle });
            }
        }
        Ok(CutStep { step: PrefillStep { done: end, row }, captured })
    }

    /// The images of a prompt as this backend's features, checked against the prompt before anything is touched: features its encoder
    /// produced, one row per placeholder, the placeholder tokens at the span, the spans in order and inside the prompt.
    fn image_features(&self, prompt: &[u32], images: &[ImageSpan]) -> Result<Vec<Arc<NemotronImageFeatures>>, ModelError> {
        let mut out = Vec::with_capacity(images.len());
        let mut at = 0usize;
        for im in images {
            let f = Arc::clone(&im.features).downcast::<NemotronImageFeatures>().map_err(|_| {
                ModelError::Request(
                    "image features were not produced by this backend (its encoder is nemotron_radio_vision)".into(),
                )
            })?;
            if f.tokens != im.tokens || f.merged != im.merged {
                return Err(ModelError::Request(format!(
                    "image has {} encoder rows ({}x{}) for {} placeholders ({}x{})",
                    f.tokens, f.merged.0, f.merged.1, im.tokens, im.merged.0, im.merged.1
                )));
            }
            if im.tokens == 0 || im.start < at || im.start.checked_add(im.tokens).is_none_or(|e| e > prompt.len()) {
                return Err(ModelError::Request(format!(
                    "image span at {} of {} tokens does not fit the prompt of {} tokens, or overlaps the one before it",
                    im.start,
                    im.tokens,
                    prompt.len()
                )));
            }
            let run = &prompt[im.start..im.start + im.tokens];
            let want = self.image_token.unwrap_or(run[0]);
            if run.iter().any(|&x| x != want) {
                return Err(ModelError::Request(format!(
                    "the {} tokens at {} are not all image placeholders{}",
                    im.tokens,
                    im.start,
                    self.image_token.map(|t| format!(" (token {t})")).unwrap_or_default()
                )));
            }
            at = im.start + im.tokens;
            out.push(f);
        }
        Ok(out)
    }

    /// The rows of each image that fall inside the slice `at..end`, as the splice the model writes after the embedding (a slice may start
    /// or end inside an image: its share is the overlap).
    fn splice_for(images: &[ImageSpan], feats: &[Arc<NemotronImageFeatures>], at: usize, end: usize) -> Vec<SpliceOp> {
        let spans: Vec<(usize, usize)> = images.iter().map(|im| (im.start, im.tokens)).collect();
        plan_splice(&spans, at, end)
            .into_iter()
            .map(|(i, dst_row, src_row, n)| SpliceOp { feats: feats[i].clone(), dst_row, src_row, n })
            .collect()
    }

    /// What one text slice of `j` needs before it runs (the checks and the begin or resume of [`NemotronBackend::slice_with_cuts`], without
    /// cuts or images): the end of the slice on the absolute grid.
    fn batch_slice_end(&mut self, j: &PrefillJob<'_>) -> Result<usize, ModelError> {
        if j.done >= j.prompt.len() {
            return Err(ModelError::Request("empty or completed prefill slice".into()));
        }
        Self::check_count(j.candidates)?;
        Self::check_position(j.prompt.len())?;
        if j.done == 0 {
            self.begin(j.slot)?;
        } else {
            self.resume(j.slot, j.done)?;
        }
        // a text prompt: no image spans (set after `begin`, which forgets them)
        self.slot_images.remove(&j.slot);
        Ok(prefix::slice_end(j.done, j.prompt.len(), self.effective_chunk(), &[]))
    }

    /// [`Backend::prefill_slices`] for text prompts on a runtime that gathers rows: see there.
    fn slices_batched(&mut self, jobs: &[PrefillJob<'_>]) -> Vec<PrefillStep> {
        let failed = |j: &PrefillJob<'_>| PrefillStep { done: j.prompt.len(), row: Some(nan_row(j.candidates)) };
        let mut out: Vec<Option<PrefillStep>> = jobs.iter().map(|_| None).collect();
        // (job, first position, end) of every slice that passed its checks, in job order
        let mut ready: Vec<(usize, usize, usize)> = Vec::with_capacity(jobs.len());
        for (i, j) in jobs.iter().enumerate() {
            match self.batch_slice_end(j) {
                Ok(end) => ready.push((i, j.done, end)),
                Err(e) => {
                    self.record(&format!("prefill slice of slot {}", j.slot), &e);
                    self.drop_slot(j.slot);
                    out[i] = Some(failed(j));
                }
            }
        }
        if !ready.is_empty() {
            if let Err(e) = self.m.flush() {
                self.record("prefill slices", &e);
                for &(i, _, _) in &ready {
                    self.drop_slot(jobs[i].slot);
                    out[i] = Some(failed(&jobs[i]));
                }
                ready.clear();
            }
        }
        let mut at = 0;
        while at < ready.len() {
            // the next forward: slices side by side up to the rows the model's prefill takes
            let (mut rows, mut group) = (0usize, Vec::new());
            while at < ready.len() {
                let (_, done, end) = ready[at];
                if !group.is_empty() && rows + (end - done) > self.pack_rows {
                    break;
                }
                rows += end - done;
                group.push(ready[at]);
                at += 1;
            }
            let reqs: Vec<RowRequest> = group
                .iter()
                .map(|&(i, done, end)| RowRequest {
                    slot: jobs[i].slot,
                    first_position: done as u64,
                    tokens: jobs[i].prompt[done..end].to_vec(),
                })
                .collect();
            let b = RowBatch::from_requests(&reqs);
            // the candidates of the last row of every prompt that ends in this forward
            let (mut wants, mut want_of) = (Vec::new(), Vec::with_capacity(group.len()));
            for (g, &(i, _, end)) in group.iter().enumerate() {
                if end == jobs[i].prompt.len() {
                    wants.push(b.seq_rows(g).end - 1);
                    want_of.push(Some(wants.len() - 1));
                } else {
                    want_of.push(None);
                }
            }
            match self.m.prefill_chunk_rows(&b, &wants) {
                Ok(c) => {
                    for (g, &(i, _, end)) in group.iter().enumerate() {
                        let j = &jobs[i];
                        let row = match want_of[g] {
                            None => None,
                            Some(k) => {
                                let got = c
                                    .as_ref()
                                    .ok_or_else(|| {
                                        ModelError::Request("the runtime returned no candidates for a final prefill slice".into())
                                    })
                                    .and_then(|c| row_of(c, k, j.candidates));
                                match got {
                                    Ok(r) => Some(r),
                                    Err(e) => {
                                        self.record(&format!("prefill slice of slot {}", j.slot), &e);
                                        self.drop_slot(j.slot);
                                        out[i] = Some(failed(j));
                                        continue;
                                    }
                                }
                            }
                        };
                        out[i] = Some(PrefillStep { done: end, row });
                    }
                }
                Err(e) => {
                    // the forward failed: its state is unusable for every prompt in it, and the error cannot be pinned on one
                    self.record(&format!("prefill of {} prompts in one forward", group.len()), &e);
                    for &(i, _, _) in &group {
                        self.drop_slot(jobs[i].slot);
                        out[i] = Some(failed(&jobs[i]));
                    }
                }
            }
        }
        jobs.iter().zip(out).map(|(j, o)| o.unwrap_or_else(|| failed(j))).collect()
    }

    /// The checks that must hold before the runtime sees a window: bounded rows and candidates, live and distinct slots.
    fn check_window(&self, reqs: &[RowRequest], candidates: usize) -> Result<(), ModelError> {
        Self::check_count(candidates)?;
        let rows: usize = reqs.iter().map(|r| r.tokens.len()).sum();
        if rows == 0 || rows > self.window_rows {
            return Err(ModelError::Request(format!("a window takes 1..={} rows, got {rows}", self.window_rows)));
        }
        let mut seen = BTreeSet::new();
        for r in reqs {
            self.check_slot(r.slot)?;
            if r.tokens.is_empty() {
                return Err(ModelError::Request(format!("slot {} has a window of no rows", r.slot)));
            }
            if !self.live.contains(&r.slot) {
                return Err(ModelError::Request(format!("slot {} has no prefilled sequence to run a window on", r.slot)));
            }
            if !seen.insert(r.slot) {
                return Err(ModelError::Request(format!("slot {} appears twice in one window", r.slot)));
            }
            let end = usize::try_from(r.first_position).ok().and_then(|p| p.checked_add(r.tokens.len()));
            let end = end.ok_or_else(|| ModelError::Request("window position overflows usize".into()))?;
            Self::check_position(end)?;
            // image rows enter a sequence through a prefill slice, from the encoder's features; a window carries tokens only, so one over
            // an image's span would run the placeholder's embedding where the image should be
            let first = end - r.tokens.len();
            if !self.spec_allowed_for_window(r.slot, first, r.tokens.len()) {
                return Err(ModelError::Request(format!(
                    "window rows {first}..{end} of slot {} fall inside an image span: image rows are only supplied by a prefill slice, \
                     never by a verify or decode window",
                    r.slot
                )));
            }
        }
        Ok(())
    }
}

impl<M: NemotronRuntime> Backend for NemotronBackend<M> {
    /// Slices are at most 64 rows, so a limit of 128 rows (the only one the engine asks for) is met without change; a smaller limit
    /// puts the slices on a finer grid, which the runtime's row invariance makes exact.
    fn set_prefill_row_limit(&mut self, rows: Option<usize>) -> bool {
        self.deadline_rows = rows.map(|r| r.max(1));
        true
    }

    fn quiesce(&mut self) -> Result<(), String> {
        self.m.flush().map_err(|e| e.to_string())
    }

    fn graph_counters(&self) -> Option<gb_engine::GraphCounters> {
        self.m.graph_counters()
    }

    /// [`prefill_floor_secs`] with the head's attention layer counted when the runtime has the head.
    fn prefill_floor(&self, from: usize, to: usize) -> Option<std::time::Duration> {
        let layers = if self.m.has_mtp() { 9 } else { 8 };
        let dense = match super::contract::current() {
            super::contract::PrefillContract::P1 => FLOOR_DENSE_TOK_S,
            super::contract::PrefillContract::P2 => FLOOR_DENSE_TOK_S_P2,
        };
        Some(std::time::Duration::from_secs_f64(prefill_floor_secs_at(from, to, layers, dense)))
    }

    // ---- the prefix cache (`docs/NEMOTRON_LONG_CONTEXT.md`): the defaults of `Backend` mean "never cached", and so does a backend that was
    // not built `with_prefix_cache`.

    fn prefix_caps(&self) -> Option<PrefixCaps> {
        self.prefix.map(|(namespace, budget)| PrefixCaps { namespace, budget })
    }

    /// The arithmetic a prompt of `total` tokens runs on, by the position rule (`prefix::path_of`): the tree alone up to 8,192 tokens,
    /// the tree below position 8,192 and the flash kernel above it for a longer prompt. A cut at or below 8,192 is valid under both.
    fn prefix_path(&self, total: usize) -> Option<u32> {
        self.prefix.map(|_| prefix::path_of(total))
    }

    fn prefix_free_snapshots(&self) -> usize {
        if self.prefix.is_some() {
            self.m.free_snapshots()
        } else {
            0
        }
    }

    /// The sequence in `slot` starts over from `handle`: the slot is begun (its previous state and pages are dropped, the backend's own
    /// bookkeeping starts fresh, as a cold prompt's does) and the runtime fills it. The next `prefill_slice(_cuts)` is called with `done == len`.
    /// On an error the slot is given up (once) and the text returned: the engine drops the entry and prefills cold from 0.
    fn prefix_restore(&mut self, slot: usize, handle: &PrefixHandle, len: usize) -> Result<(), String> {
        if self.prefix.is_none() {
            return Err("this backend has no prefix cache".into());
        }
        if self.dead() {
            return Err("the backend has failed".into());
        }
        let r = self.begin(slot).and_then(|()| self.m.restore_snapshot(slot, handle, len));
        match r {
            Ok(()) => Ok(()),
            Err(e) => {
                // a recoverable refusal spoils this restore only; a lost device is the engine's to stop on
                if e.is_fatal() {
                    self.record(&format!("restore of a snapshot into slot {slot}"), &e);
                }
                self.drop_slot(slot);
                Err(e.to_string())
            }
        }
    }

    /// Lets go of a snapshot (an entry was evicted or a capture was discarded): its arena slot and its page references. Once per handle.
    fn prefix_release(&mut self, handle: PrefixHandle) {
        if self.dead() {
            return;
        }
        if self.prefix.is_none() {
            self.record("prefix release", &ModelError::Config("this backend has no prefix cache".into()));
            return;
        }
        if let Err(e) = self.m.release_snapshot(handle) {
            self.record("prefix release", &e);
        }
    }

    /// [`Backend::prefill_slice`] that also takes a snapshot at each of `cuts` that falls at the end of the slice it runs. The slice is
    /// shortened to end at the first cut inside it; the bits of the prompt are the same with or without cuts.
    fn prefill_slice_cuts(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
        cuts: &[usize],
    ) -> CutStep {
        if self.dead() {
            return CutStep { step: PrefillStep { done: prompt.len(), row: Some(nan_row(candidates)) }, captured: Vec::new() };
        }
        match self.slice_with_cuts(slot, prompt, images, done, candidates, cuts) {
            Ok(step) => step,
            Err(e) => {
                self.record(&format!("prefill slice of slot {slot}"), &e);
                self.drop_slot(slot);
                CutStep { step: PrefillStep { done: prompt.len(), row: Some(nan_row(candidates)) }, captured: Vec::new() }
            }
        }
    }

    fn vocab_size(&self) -> usize {
        self.m.vocab_size()
    }

    fn take_error(&mut self) -> Option<BackendError> {
        self.fatal.clone().or_else(|| self.error.take())
    }

    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.prefill_piece(slot, start, tokens, (start as usize).saturating_add(tokens.len()), candidates)
    }

    fn prefill_piece(&mut self, slot: usize, start: u64, tokens: &[u32], total: usize, candidates: usize) -> RowOutput {
        if self.dead() {
            return nan_row(candidates);
        }
        match self.prefill_inner(slot, start, tokens, total, candidates) {
            Ok(row) => row,
            Err(e) => {
                self.record(&format!("prefill of slot {slot}"), &e);
                self.drop_slot(slot);
                nan_row(candidates)
            }
        }
    }

    fn prefill_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
    ) -> PrefillStep {
        if self.dead() {
            return PrefillStep { done: prompt.len(), row: Some(nan_row(candidates)) };
        }
        match self.prefill_slice_inner(slot, prompt, images, done, candidates) {
            Ok(step) => step,
            Err(e) => {
                self.record(&format!("prefill slice of slot {slot}"), &e);
                self.drop_slot(slot);
                PrefillStep { done: prompt.len(), row: Some(nan_row(candidates)) }
            }
        }
    }

    /// The next slice of each of several waiting prompts: one forward per (up to `prefill_rows`) rows of them, the slices of several
    /// prompts side by side, and the candidates of each prompt that ends in a forward from one gathered lm_head. Every kernel of a prefill
    /// is row invariant, so each prompt comes out with the bits its own `prefill_slice` gives it; a prompt that fails is failed alone
    /// when the failure is its own (a refused request) and with its forward's other prompts when the forward failed. Prompts with images
    /// and runtimes that cannot gather rows are served one slice at a time, as the default does.
    fn prefill_slices(&mut self, jobs: &[PrefillJob<'_>]) -> Vec<CutStep> {
        let steps = if self.dead() {
            jobs.iter().map(|j| PrefillStep { done: j.prompt.len(), row: Some(nan_row(j.candidates)) }).collect()
        } else if jobs.len() < 2 || !self.m.prefill_rows_supported() || jobs.iter().any(|j| !j.images.is_empty()) {
            jobs.iter().map(|j| self.prefill_slice(j.slot, j.prompt, j.images, j.done, j.candidates)).collect()
        } else {
            self.slices_batched(jobs)
        };
        steps.into_iter().map(|step| CutStep { step, captured: Vec::new() }).collect::<Vec<_>>()
    }

    fn prefill_mm(&mut self, slot: usize, start: u64, tokens: &[u32], images: &[ImageSpan], candidates: usize) -> RowOutput {
        if images.is_empty() {
            return self.prefill(slot, start, tokens, candidates);
        }
        if self.dead() {
            return nan_row(candidates);
        }
        // a prompt with images is prefilled from its start, whole, by the same slices the engine's `prefill_slice` runs
        if start != 0 {
            let e = ModelError::Request("a prompt with images must be prefilled from its start".into());
            self.record("multimodal prefill", &e);
            self.drop_slot(slot);
            return nan_row(candidates);
        }
        let mut done = 0;
        loop {
            let step = self.prefill_slice(slot, tokens, images, done, candidates);
            done = step.done;
            if done >= tokens.len() {
                return step.row.unwrap_or_else(|| nan_row(candidates));
            }
        }
    }

    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        let nans = || reqs.iter().map(|r| r.tokens.iter().map(|_| nan_row(candidates)).collect()).collect();
        if self.dead() {
            return nans();
        }
        if reqs.is_empty() {
            return Vec::new();
        }
        let run = (|| -> Result<Vec<Vec<RowOutput>>, ModelError> {
            self.check_window(reqs, candidates)?;
            self.m.flush()?;
            if let Some(s) = self.spec.as_mut() {
                for r in reqs {
                    s.on_step(r.slot, r.tokens.len());
                }
            }
            let b = RowBatch::from_requests(reqs);
            let started = std::time::Instant::now();
            let c = if self.fuse_decode {
                self.m.window_candidates_fused(&b, candidates)?
            } else {
                self.m.window_candidates(&b, candidates)?
            };
            if c.rows != b.rows {
                return Err(ModelError::Request(format!(
                    "the runtime returned {} candidate rows for a window of {}",
                    c.rows, b.rows
                )));
            }
            let rows: Result<Vec<Vec<RowOutput>>, ModelError> =
                (0..reqs.len()).map(|i| b.seq_rows(i).map(|r| row_of(&c, r, candidates)).collect()).collect();
            if rows.is_ok() {
                if let Some(s) = self.spec.as_mut() {
                    s.on_window(reqs.len(), b.rows, started.elapsed().as_secs_f64() * 1e3);
                }
            }
            rows
        })();
        match run {
            Ok(rows) => {
                for r in reqs {
                    self.windows.insert(r.slot, r.tokens.len());
                }
                rows
            }
            Err(e) => {
                // the rows are unusable and the state untouched: no window of this step is pending, and the engine decides the slots
                for r in reqs {
                    self.windows.remove(&r.slot);
                }
                self.record("window step", &e);
                nans()
            }
        }
    }

    fn commit(&mut self, slot: usize, keep: usize) {
        if self.dead() {
            return;
        }
        let check = || -> Result<(), ModelError> {
            if keep == 0 || keep > self.window_rows {
                return Err(ModelError::Request(format!("a commit keeps 1..={} rows, asked for {keep}", self.window_rows)));
            }
            let Some(rows) = self.windows.get(&slot).copied() else {
                return Err(ModelError::Request(format!("slot {slot} has no window to commit")));
            };
            if keep > rows {
                return Err(ModelError::Request(format!("a commit of {keep} rows, but slot {slot}'s last window had {rows}")));
            }
            Ok(())
        };
        match check() {
            // a refused commit changes nothing: the window stays as it was and the engine fails the sequence it served
            Err(e) => self.record(&format!("commit of slot {slot}"), &e),
            Ok(()) => {
                self.windows.remove(&slot);
                self.m.commit(slot, keep);
                // the round's record is consumed by its commit; the runtime moved the head's tapped rows in `commit` itself
                if let Some(s) = self.spec.as_mut() {
                    s.on_commit(slot, keep);
                }
            }
        }
    }

    fn release(&mut self, slot: usize) {
        self.drop_slot(slot);
    }

    /// Drafts for the streams that may draft this round, with the head's confidence in each pick (`Proposal::conf`). One entry per
    /// request: `Some` with no tokens means "no drafts this round" (the engine's own drafter is not consulted), `None` that the backend
    /// has no say (drafting is off, the backend has failed, or the request was refused) and the engine's drafter fills in.
    /// A head error costs the round its head drafts only; drafts never change what is emitted.
    fn propose_batch_scored(&mut self, reqs: &[ProposeRequest]) -> Vec<Option<Proposal>> {
        if self.dead() || self.spec.is_none() || reqs.is_empty() {
            return vec![None; reqs.len()];
        }
        let NemotronBackend { m, spec, slot_images, .. } = self;
        let allowed = |slot: usize, first: usize, rows: usize| {
            window_is_text(slot_images.get(&slot).map_or(&[], |v| v.as_slice()), first, rows)
        };
        match spec.as_mut().expect("checked above").propose(m, reqs, &allowed) {
            Ok(v) => v,
            Err(e) => {
                self.record(&format!("drafting for {} sequences", reqs.len()), &e);
                vec![None; reqs.len()]
            }
        }
    }

    /// [`Backend::propose_batch_scored`] without the confidences.
    fn propose_batch(&mut self, reqs: &[ProposeRequest]) -> Vec<Option<Vec<u32>>> {
        self.propose_batch_scored(reqs).into_iter().map(|p| p.map(|p| p.tokens)).collect()
    }
}

#[cfg(test)]
mod floor_tests {
    use super::prefill_floor_secs;

    /// The floor is additive over a split of the prompt (a cached prefix only removes its own share), and it stays below the measured model
    /// of the review (`T(N) = N / 520 + 9 x 0.2787 ms x N^2 / 2M`) at every depth, by at least a fifth.
    #[test]
    fn the_prefill_floor_is_additive_and_below_the_measured_model() {
        let full = prefill_floor_secs(0, 1 << 20, 9);
        assert!((full - 2270.7).abs() < 0.5, "{full}");
        let a = 700_000;
        assert!((prefill_floor_secs(0, a, 9) + prefill_floor_secs(a, 1 << 20, 9) - full).abs() < 1e-6);
        assert_eq!(prefill_floor_secs(5, 5, 9), 0.0);
        for n in [32_768usize, 131_072, 202_690, 524_288, 1 << 20] {
            let x = n as f64;
            let model = x / 520.0 + 9.0 * 0.000_278_7 * x * x / (2.0 * 1_048_576.0);
            let floor = prefill_floor_secs(0, n, 9);
            assert!(floor < 0.8 * model, "{n}: floor {floor:.0} s against the model's {model:.0} s");
        }
        // the measured runs it must stay below (seconds, through the server): 28,818 in 52.98; 152,014 in 298.6; 202,690 in 407.8
        for (n, measured) in [(28_818usize, 52.98), (152_014, 298.6), (202_690, 407.8)] {
            assert!(prefill_floor_secs(0, n, 9) < measured, "{n}");
        }
        // the default --request-timeout (1,800 s) is refused from about 875k tokens cold
        assert!(prefill_floor_secs(0, 860_000, 9) < 1800.0 && prefill_floor_secs(0, 890_000, 9) > 1800.0);
    }

    /// Under P2 the floor stays below P2's measured runs (seconds, through the server: lane S 8,692 tokens in 6.06, 40,879 in 29.2; lane S5
    /// one node 8,692 in 5.40, 40,879 in 26.06; lane S TP=2 10,833 in 6.98, 28,818 in 18.88; lane S6 on rc2d with the qualified profile,
    /// one node 10,833 in 6.038, 28,818 in 16.548, TP=2 10,833 in 6.285, 28,818 in 17.033) and below P1's, and its dense rate is 1.25 x
    /// the fastest of those runs (the re-fit rule: a faster run, once listed here, moves the constant with it).
    #[test]
    fn the_p2_floor_is_below_p2s_measured_runs_and_below_p1s_floor() {
        use super::{prefill_floor_secs_at, FLOOR_DENSE_TOK_S_P2};
        // seconds measured through the server under P2 (single node and TP=2): S5 b0dfa25f 1,750 / 1,700 tok/s, the TP=2 head split
        // (ce020678) 6.56 / 17.90 s, and S6's rc2d runs on the qualified profile (the fastest reps), the fastest of all last
        let runs = [
            (8_692usize, 6.06),
            (40_879, 29.2),
            (8_692, 5.40),
            (40_879, 26.06),
            (10_833, 6.98),
            (28_818, 18.88),
            (10_833, 6.56),
            (28_818, 17.90),
            (8_692, 8_692.0 / 1_750.0),
            (40_879, 40_879.0 / 1_700.0),
            (10_833, 6.285),
            (28_818, 17.033),
            (28_818, 16.548),
            (10_833, 6.038),
        ];
        for (n, measured) in runs {
            assert!(prefill_floor_secs_at(0, n, 9, FLOOR_DENSE_TOK_S_P2) < measured, "{n}");
        }
        // the constant is tied to the evidence: 1.25 x the fastest listed rate (10,833 tokens in 6.038 s = 1,794.1 tok/s; the constant
        // uses S6's 1,794.2 from the unrounded 6.0379 s): a faster run added here without re-fitting the constant fails
        let fastest = runs.iter().map(|&(n, s)| n as f64 / s).fold(0.0, f64::max);
        assert!((FLOOR_DENSE_TOK_S_P2 - 1.25 * fastest).abs() < 0.5, "{FLOOR_DENSE_TOK_S_P2} against 1.25 x {fastest:.1}");
        for n in [8_692usize, 131_072, 1 << 20] {
            assert!(prefill_floor_secs_at(0, n, 9, FLOOR_DENSE_TOK_S_P2) < prefill_floor_secs(0, n, 9), "{n}");
        }
    }
}
