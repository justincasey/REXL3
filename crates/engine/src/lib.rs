//! The step loop, written against a `Backend` trait so the same loop and the same exactness harness
//! run on the CPU mock here and on the CUDA backend on the GB10.
//!
//! Positions: a sequence's `tokens` end with one token not yet fed to the backend. A step feeds
//! `[last, d1..dw]` at positions `len-1 ..`; row j predicts position `len + j`, sampled with the keyed
//! sampler at that position. Drafts are accepted while `d(j+1) == sample(j)`. With `a` accepted, the
//! engine commits `a + 1` tokens and tells the backend to keep `a + 1` rows of KV and recurrent state.
#![forbid(unsafe_code)]

pub mod boot;
pub mod harness;
pub mod mock;
pub mod prefix;
pub mod score;

pub use prefix::{CacheBudget, Captured, CutPolicy, Eviction, PrefixCaps, PrefixConfig, PrefixHandle, PrefixStats};
use prefix::{PrefillCache, PrefixState};

use gb_sampling::{try_choose, Sampling, SamplingError};
use gb_sched::{
    choose_budget, plan_round, plan_round_charged, plan_trees, Admission, AdmitError, CostModel, Demand, DraftCharge, RoundCost,
    StepCost, TreeDemand,
};
use gb_spec::stats::{conditional_prior, AcceptCurve, Calibration, MarginTable, Mode, SpecStats, MAX_POSITIONS};
use gb_spec::tree::{accept_tree, best_first, limit_depth_with, with_copy, CopyBranch, DraftTree, Lattice, Policy};
use gb_spec::{accept, AcceptStats, CopyDrafter};
use std::fmt;
use std::time::Instant;

/// Why a request was refused. Nothing is reserved when `submit` returns one of these.
#[derive(Clone, Debug, PartialEq)]
pub enum SubmitError {
    /// Memory, slots, context length or size limits (see `AdmitError`).
    Admit(AdmitError),
    /// The request itself is malformed: empty prompt, `max_new` of 0, a token outside the vocabulary,
    /// or a sampling setting the engine does not support.
    Invalid(String),
    /// The first row could not be sampled (for example a NaN logit from the backend).
    Sampling(SamplingError),
    /// The backend reported an error while prefilling the prompt (`Engine::submit`, which prefills inside the call).
    Backend(String),
}

impl From<AdmitError> for SubmitError {
    fn from(e: AdmitError) -> Self {
        SubmitError::Admit(e)
    }
}

impl From<SamplingError> for SubmitError {
    fn from(e: SamplingError) -> Self {
        SubmitError::Sampling(e)
    }
}

impl fmt::Display for SubmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SubmitError::Admit(e) => write!(f, "refused: {e:?}"),
            SubmitError::Invalid(m) => write!(f, "invalid request: {m}"),
            SubmitError::Sampling(e) => write!(f, "sampling failed: {e}"),
            SubmitError::Backend(m) => write!(f, "backend failed: {m}"),
        }
    }
}

impl std::error::Error for SubmitError {}

/// Why a step failed. A step that returns one of these did not run to the end: the engine's own bookkeeping is untouched, but the
/// backend may hold half-run rows, so the caller gives the running sequences back (`Engine::fail_active`, `abort_all`) or, for a
/// fatal error, stops serving.
///
/// A row that cannot be sampled (a NaN logit) is not a step error: it fails only its own sequence (`Engine::take_failed`).
#[derive(Clone, Debug, PartialEq)]
pub enum StepError {
    /// The backend broke the `Backend::step` contract: wrong number of outputs.
    BackendShape(String),
    /// The backend reported a recoverable error for this step (see `Backend::take_error`): the step's rows are unusable but the
    /// device is sound, so the next step may succeed.
    Backend(String),
    /// The backend reported an error it cannot recover from (a sticky device fault, or state it could not keep consistent): every
    /// later step would be wrong, so the process must stop serving.
    Fatal(String),
}

impl StepError {
    /// The error cannot be recovered from: stop serving.
    pub fn is_fatal(&self) -> bool {
        matches!(self, StepError::Fatal(_))
    }
}

impl fmt::Display for StepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepError::BackendShape(m) => write!(f, "backend contract violated: {m}"),
            StepError::Backend(m) => write!(f, "backend step failed: {m}"),
            StepError::Fatal(m) => write!(f, "fatal backend error: {m}"),
        }
    }
}

impl std::error::Error for StepError {}

/// An error a backend recorded while running (see `Backend::take_error`). `fatal` means the device or the backend's own state is no
/// longer sound (a sticky CUDA error, a state commit that did not complete); anything else only spoils the call it came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendError {
    pub fatal: bool,
    pub message: String,
}

impl BackendError {
    pub fn fatal(message: impl Into<String>) -> Self {
        Self { fatal: true, message: message.into() }
    }
    pub fn retryable(message: impl Into<String>) -> Self {
        Self { fatal: false, message: message.into() }
    }
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for BackendError {}

/// Why one sequence failed while the others went on.
#[derive(Clone, Debug, PartialEq)]
enum Failure {
    /// A row of its own could not be sampled.
    Sampling(SamplingError),
    /// The backend reported a recoverable error in a call that served it.
    Backend(String),
}

impl Failure {
    fn message(&self) -> String {
        match self {
            Failure::Sampling(e) => e.to_string(),
            Failure::Backend(m) => m.clone(),
        }
    }
    fn into_submit_error(self) -> SubmitError {
        match self {
            Failure::Sampling(e) => SubmitError::Sampling(e),
            Failure::Backend(m) => SubmitError::Backend(m),
        }
    }
}

/// Top candidates of one row, ordered by (value desc, id asc).
#[derive(Clone, Debug, PartialEq)]
pub struct RowOutput {
    pub ids: Vec<u32>,
    pub values: Vec<f32>,
}

impl RowOutput {
    /// Bit-level equality (f32 compared by bits, so -0.0 != 0.0 and NaN patterns count).
    pub fn bits_eq(&self, o: &RowOutput) -> bool {
        self.ids == o.ids && self.values.iter().map(|v| v.to_bits()).eq(o.values.iter().map(|v| v.to_bits()))
    }
}

/// One sequence's rows in a step.
#[derive(Clone, Debug)]
pub struct RowRequest {
    pub slot: usize,
    /// `[last committed token, draft 1, ..]`
    pub tokens: Vec<u32>,
    /// Absolute position of `tokens[0]`.
    pub first_position: u64,
}

/// One sequence's TREE window in a step (`Backend::step_tree`): row 0 is the pending token (the root) and every other row a drafted token
/// whose parent is an earlier row. A row's context is the committed prefix plus its ancestors, so it sits at position `first_position +
/// depth` and its outputs must be bit-identical to serial decoding of that root path (docs/TREE_DRAFTING.md).
#[derive(Clone, Debug)]
pub struct TreeRequest {
    pub slot: usize,
    /// `[pending token, drafted tokens ..]`, parents before children.
    pub tokens: Vec<u32>,
    /// `parents[i]` is the row index of row i's parent; `parents[0] == -1` and `0 <= parents[i] < i` otherwise. A chain is `i - 1`.
    pub parents: Vec<i32>,
    /// Absolute position of the root.
    pub first_position: u64,
}

impl TreeRequest {
    /// Whether every row's parent is the row before it (an ordinary window).
    pub fn is_chain(&self) -> bool {
        self.parents.iter().enumerate().all(|(i, &p)| p == i as i32 - 1)
    }
    /// The depth of every row (the root is 0).
    pub fn depths(&self) -> Vec<usize> {
        let mut d = vec![0usize; self.parents.len()];
        for i in 1..d.len() {
            d[i] = d[self.parents[i].max(0) as usize] + 1;
        }
        d
    }
}

/// What the engine needs from a model runtime.
///
/// The contract: every row's output is bit-identical to what the same backend returns for that row in a
/// one-row step. Batch composition, window width and chunking must not change a single bit.
/// `harness` checks this.
pub trait Backend {
    /// Safe, row-invariant prefill slicing only. Return false for unsupported backends; never emulate by changing prompt length.
    fn set_prefill_row_limit(&mut self, _rows: Option<usize>) -> bool {
        false
    }
    /// Before the prefill slices of a step: whether a sequence is decoding (each slice the step runs delays its next token). A backend
    /// with an adaptive slice (GLM's tensor-parallel rank 0) runs its big slices only when none is; the slices are row-invariant, so
    /// no bit depends on it. The default ignores it.
    fn set_decode_waiting(&mut self, _waiting: bool) {}
    /// A prompt slice the backend runs in parts (several `prefill_slice` calls that each return the prompt's `done` unchanged and no row
    /// until the last, with the decode rounds of the steps between them) and has begun: its slot, and the most rows another prompt's slice
    /// may have while it waits (0: none). The engine then serves that prompt before any other whose rows left would not fit those rows
    /// (and before every other once it was passed over [`PREFILL_PASS_GUARD`] steps). Only when slices run moves, never a bit. The
    /// default: none.
    fn prefill_in_flight(&self) -> Option<(usize, usize)> {
        None
    }
    /// A proven cold-call bound, when available (CPU mock); native families require observed work, not a guessed rate.
    fn cold_step_budget(&self) -> Option<std::time::Duration> {
        None
    }
    /// Synchronize device work before owner-thread teardown. CPU backends need nothing.
    fn quiesce(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn score_supported(&self) -> bool {
        false
    }
    /// One score prefill slice, including prefix captures. The default refuses BEFORE touching state.
    fn score_slice(
        &mut self,
        _slot: usize,
        _prompt: &[u32],
        _done: usize,
        _cuts: &[usize],
    ) -> Result<score::ScoreStep, BackendError> {
        Err(score::unsupported_backend())
    }
    /// The full-vocabulary logits of every row of the prompt a `ScoreRequest { all_rows: true }` just finished, `[len][vocab]`:
    /// row `i` is the logits after prompt tokens `0..=i` (the prediction of token `i + 1`), the last one the request's own row.
    /// Read once at completion, after [`Backend::score_slice`] returned the prompt's last row; the default has none.
    fn take_score_rows(&mut self) -> Vec<Vec<f32>> {
        Vec::new()
    }
    /// Tell the backend whether the next score request wants [`Backend::take_score_rows`]. A backend that cannot produce
    /// them keeps the default and answers the request with the last row only.
    fn score_all_rows(&mut self, _on: bool) {}
    fn vocab_size(&self) -> usize;
    /// Whether [`Backend::step_tree`] verifies real trees (a backend that does not keeps the default `false` and only ever gets chains).
    fn supports_trees(&self) -> bool {
        false
    }
    /// [`Backend::step`] for tree windows: one output per row of every request, row i the candidates after the root path of row i.
    /// Only called when [`Backend::supports_trees`]. The default runs chains through `step` and refuses a real tree.
    fn step_tree(&mut self, reqs: &[TreeRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        assert!(reqs.iter().all(TreeRequest::is_chain), "this backend does not verify trees (supports_trees is false)");
        let chains: Vec<RowRequest> = reqs
            .iter()
            .map(|r| RowRequest { slot: r.slot, tokens: r.tokens.clone(), first_position: r.first_position })
            .collect();
        self.step(&chains, candidates)
    }
    /// [`Backend::commit`] for a tree window: the accepted rows of `slot`'s last window are `path` (row indices within the window, root
    /// first, each the parent of the next). The default accepts only a prefix `0..n`, which is `commit(slot, n)`.
    fn commit_path(&mut self, slot: usize, path: &[usize]) {
        assert!(path.iter().enumerate().all(|(i, &r)| i == r), "this backend does not commit a tree path: {path:?}");
        self.commit(slot, path.len());
    }
    /// The first error recorded since the last call (cleared by the call). A backend whose methods cannot fail (they answer with
    /// NaN rows, an empty draft or nothing) says why here: the engine asks after every `prefill_slice`, `step`, `commit` and
    /// `propose_batch`, fails the sequences that call served when the error is recoverable, and stops when it is `fatal`.
    /// The default never has an error to report.
    fn take_error(&mut self) -> Option<BackendError> {
        None
    }
    /// Append `tokens` to `slot`'s context starting at absolute position `start` and return the
    /// candidates predicting the position after the last one.
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput;
    /// [`Backend::prefill`] for tokens `start..start + tokens.len()` that are one piece of a prompt of `total` tokens in all. A
    /// backend whose arithmetic depends on the length of the prompt (the Qwen backend picks its prefill path by it, so that
    /// every row of a prompt takes the same one) needs the total. The default ignores it. The bits of the last row must not
    /// depend on how the prompt is cut into pieces: `harness::chunked_prefill_equals_whole` checks that.
    fn prefill_piece(&mut self, slot: usize, start: u64, tokens: &[u32], total: usize, candidates: usize) -> RowOutput {
        let _ = total;
        self.prefill(slot, start, tokens, candidates)
    }
    /// One forward over several sequences; returns, per request, one output per input token.
    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>>;
    /// [`Backend::step`] with the greedy flag of every request (`greedy[i]` is true exactly when request
    /// `i` samples at temperature 0). A backend that can verify greedy windows on the device (fewer
    /// readback bytes per round) overrides this; the default is exactly `step`. A backend must return
    /// the same rows a `step` call would, whatever the flags say — they are only a hint to the backend.
    fn step_verified(&mut self, reqs: &[RowRequest], candidates: usize, greedy: &[bool]) -> Vec<Vec<RowOutput>> {
        assert_eq!(reqs.len(), greedy.len(), "step_verified: flags and requests differ in length");
        self.step(reqs, candidates)
    }
    /// Keep the first `keep` rows of `slot`'s last step (drop the rejected drafts' KV and state).
    fn commit(&mut self, slot: usize, keep: usize);
    fn release(&mut self, slot: usize);
    /// Drafts made by the backend itself, for drafters that need device state (DFlash2 reads the target's hidden
    /// states, which stay on the GPU). `tokens` is the sequence so far, ending with the one not yet fed. Return None
    /// when the backend has no native drafter; the engine then asks the sequence's `Drafter`.
    fn propose(&mut self, _slot: usize, _tokens: &[u32], _sampling: &Sampling, _max: usize) -> Option<Vec<u32>> {
        None
    }
    /// `prefill` for a prompt that contains images: the rows of each span take the span's encoder output instead of a token
    /// embedding, and positions follow the model's multimodal rule. A backend without vision keeps this default, which treats
    /// the prompt as text.
    fn prefill_mm(&mut self, slot: usize, start: u64, tokens: &[u32], _images: &[ImageSpan], candidates: usize) -> RowOutput {
        self.prefill(slot, start, tokens, candidates)
    }
    /// Advance the prefill of `prompt` in `slot` by one slice: `done` tokens are already in the slot (0 starts it). The backend
    /// chooses the slice (a backend that prefills in chunks returns after one chunk, so the engine can run a decode round for the
    /// other streams in between) and reports the new `done`; `row` is the candidates after the prompt once `done == prompt.len()`.
    /// The default prefills everything in one go. A backend must give the same bits whatever the slicing it is driven with.
    fn prefill_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
    ) -> PrefillStep {
        debug_assert_eq!(done, 0, "the default prefill_slice cannot resume");
        let row = if images.is_empty() {
            self.prefill(slot, 0, prompt, candidates)
        } else {
            self.prefill_mm(slot, 0, prompt, images, candidates)
        };
        PrefillStep { done: prompt.len(), row: Some(row) }
    }
    /// One [`Backend::prefill_slice`] for each of several prompts (distinct slots), in one call, one step per job in job order. The
    /// default runs them one after another. A backend that can put the rows of several short prompts through one forward (the Qwen
    /// backend: prompts of at most 128 tokens share one pass of the row-invariant decode kernel) overrides this; each prompt must come out
    /// with the bits its own `prefill_slice` gives it, whatever else is in the call, and a prompt that fails must not fail the others
    /// (a recoverable error is reported through `take_error`, and the failed prompt's row is NaN). The engine only puts a prompt in this call
    /// when the prefix cache plays no part in its slice (it was not restored from a hit and has no snapshot to take: those need
    /// [`Backend::prefill_slice_cuts`], one prompt at a time), so a backend needs no cut handling here.
    fn prefill_slices(&mut self, jobs: &[PrefillJob<'_>]) -> Vec<PrefillStep> {
        jobs.iter().map(|j| self.prefill_slice(j.slot, j.prompt, j.images, j.done, j.candidates)).collect()
    }
    /// `propose` for several sequences at once, one entry per request. A backend whose drafter shares work across
    /// sequences (one read of the draft's weights for all of them) overrides this.
    fn propose_batch(&mut self, reqs: &[ProposeRequest]) -> Vec<Option<Vec<u32>>> {
        reqs.iter().map(|r| self.propose(r.slot, r.tokens, r.sampling, r.max)).collect()
    }
    /// `propose_batch` with the drafter's own confidence in each token, where it has one. The engine calls this; a backend whose
    /// drafter scores its picks overrides it, and every other backend keeps the default (no scores).
    fn propose_batch_scored(&mut self, reqs: &[ProposeRequest]) -> Vec<Option<Proposal>> {
        self.propose_batch(reqs).into_iter().map(|o| o.map(|tokens| Proposal { tokens, ..Default::default() })).collect()
    }

    /// The backend's CUDA-graph cache counters, when it keeps one; `None` (the default) for a backend without graphs.
    fn graph_counters(&self) -> Option<GraphCounters> {
        None
    }
    /// A LOWER bound on the time to prefill positions `from..to` of a prompt (`from` tokens restored from the prefix cache), when the backend
    /// can state one (its family measured its rates; the bound must be optimistic, so a request it refuses could never have finished in
    /// time). `None` (the default): no bound, nothing is refused on time.
    fn prefill_floor(&self, _from: usize, _to: usize) -> Option<std::time::Duration> {
        None
    }

    // ---- prefix cache (docs/PREFIX_CACHE.md). A backend that keeps the defaults is never cached.

    /// What the backend reserved for a prefix cache (snapshot slots, KV pages) and the namespace its snapshots live in; `None`: this
    /// backend cannot snapshot and restore a prompt's state (a family that has not proven it), so the engine never looks anything up.
    fn prefix_caps(&self) -> Option<PrefixCaps> {
        None
    }
    /// The prefill path id of a prompt of `total` tokens when its state may be cached, `None` when it may not. State depends on the
    /// arithmetic that produced it, and the arithmetic is picked by the prompt's TOTAL length (the Qwen backend: the decode GEMV up to
    /// 128 tokens, the pinned GEMM above), so a snapshot is only ever interchangeable between prompts of one path, and the GEMV path is
    /// not cached at all. A backend with ONE arithmetic for every length (GLM: every prefill chunk equals the decode kernels' bits) answers
    /// one id for every total.
    fn prefix_path(&self, _total: usize) -> Option<u32> {
        None
    }
    /// Snapshot slots no capture holds right now.
    fn prefix_free_snapshots(&self) -> usize {
        0
    }
    /// Put `slot` in the state a prefill of the first `len` tokens of the entry's prompt would have left it in: the sequence starts over from
    /// `handle` (the slot's previous state and pages are dropped, the entry's pages are shared by reference with the partial last page
    /// copied, the recurrent state, conv history and the draft's context are copied in, nothing is owed). The next `prefill_slice` is called
    /// with `done == len`. An `Err` leaves the slot unusable until a prefill from 0 begins it again.
    fn prefix_restore(&mut self, _slot: usize, _handle: &PrefixHandle, _len: usize) -> Result<(), String> {
        Err("this backend has no prefix cache".into())
    }
    /// Let go of a snapshot slot and its page references (an entry was evicted, a capture was discarded). Called exactly once per handle.
    fn prefix_release(&mut self, _handle: PrefixHandle) {}
    /// [`Backend::prefill_slice`] that also takes a snapshot of the slot's state at each of `cuts` (ascending token counts, the engine's
    /// plan for the whole prompt) that lies in `done < cut <= the slice's end`, returned in `CutStep::captured`. A cut is taken
    /// exactly: the slice need not end there, and the prompt's bits are the same with or without cuts. A snapshot that cannot be taken
    /// (no free slot) is skipped, not an error. On an error nothing is returned and nothing stays allocated. The default takes none.
    fn prefill_slice_cuts(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
        cuts: &[usize],
    ) -> CutStep {
        let _ = cuts;
        CutStep { step: self.prefill_slice(slot, prompt, images, done, candidates), captured: Vec::new() }
    }
}

/// Drafts for one sequence, with the drafter's confidence in each when it has one.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Proposal {
    pub tokens: Vec<u32>,
    /// The drafter's score for each pick (higher is more confident; DFlash2: the winning selector score, on the target head's logit
    /// scale); empty when the drafter does not score.
    pub conf: Vec<f32>,
    /// How far each pick is ahead of the runner-up in what the pick maximised (DFlash2: the noised score in temperature units when
    /// sampled, the raw gap when greedy): a large margin means the target, whose own scores differ a little, will likely choose the
    /// same token. Empty when the drafter has none.
    pub margin: Vec<f32>,
    /// The drafter's own calibrated probability that each pick is accepted GIVEN that every pick before it was (a trained confidence head
    /// whose output has been checked against what the target accepted: MiniCPM-V-4.7's DSpark head, within 0.06 of the observed rate per
    /// decile). The `Margin` budget policy spends verify rows by it directly, scaled by the stream's own calibration factor; it does not go
    /// through the margin table, which is a DFlash2 measurement. Empty when the drafter has none. Only how many picks are verified depends
    /// on it, never which token is committed.
    pub accept: Vec<f32>,
    /// The selector's candidate lattice behind these picks, when the drafter keeps one (DFlash2 with lattice capture on): what the tree
    /// search of a tree round (`Engine::with_trees`) builds its tree from. `None`: the picks are a chain.
    pub lattice: Option<Lattice>,
}

/// The prefix index's key stream for a prompt: the token ids, with every image's placeholder run replaced by tokens derived from the image's
/// content key (so the same text around another image is another prefix, and the same image under the same text is the same one).
fn cache_tokens(prompt: &[u32], images: &[ImageSpan]) -> Vec<u32> {
    let mut t = prompt.to_vec();
    for im in images {
        for k in 0..im.tokens {
            t[im.start + k] = gb_sched::image_key_token(im.key, k as u32);
        }
    }
    t
}

/// Where a cut at `c` may be taken: a cut that falls strictly inside an image span moves to the span's first placeholder when that is still a useful cut
/// (at least `min_cut`, above `hit`) and is dropped (`None`) otherwise; a cut outside every image stays.
fn outside_images(c: usize, images: &[ImageSpan], min_cut: usize, hit: usize) -> Option<usize> {
    match images.iter().find(|im| im.start < c && c < im.start + im.tokens) {
        Some(im) => (im.start >= min_cut && im.start > hit).then_some(im.start),
        None => Some(c),
    }
}

/// One prompt's turn in [`Backend::prefill_slices`]: the arguments of a [`Backend::prefill_slice`] call.
pub struct PrefillJob<'a> {
    pub slot: usize,
    pub prompt: &'a [u32],
    pub images: &'a [ImageSpan],
    pub done: usize,
    pub candidates: usize,
}

/// Progress of one `Backend::prefill_slice`.
pub struct PrefillStep {
    /// Prompt tokens now in the slot.
    pub done: usize,
    /// The candidates after the last prompt token; `Some` exactly when `done` reached the prompt's length.
    pub row: Option<RowOutput>,
}

/// One `Backend::prefill_slice_cuts`: the slice's progress, and the snapshots it took (the engine owns them from here on).
pub struct CutStep {
    pub step: PrefillStep,
    pub captured: Vec<Captured>,
}

/// One sequence's ask in `Backend::propose_batch`.
pub struct ProposeRequest<'a> {
    pub slot: usize,
    pub tokens: &'a [u32],
    pub sampling: &'a Sampling,
    pub max: usize,
}

/// A source of drafts for one sequence.
pub trait Drafter {
    fn propose(&mut self, tokens: &[u32], sampling: &Sampling, max: usize) -> Vec<u32>;
}

pub struct NoDrafts;
impl Drafter for NoDrafts {
    fn propose(&mut self, _: &[u32], _: &Sampling, _: usize) -> Vec<u32> {
        Vec::new()
    }
}

impl Drafter for CopyDrafter {
    fn propose(&mut self, tokens: &[u32], _: &Sampling, max: usize) -> Vec<u32> {
        CopyDrafter::propose(self, tokens, max)
    }
}

/// One image's run of placeholder tokens in a prompt and its encoder output. The engine only checks the geometry and hands the
/// span to the backend, which downcasts `features` to whatever its own encoder produced.
#[derive(Clone)]
pub struct ImageSpan {
    /// Index of the first placeholder token in the prompt.
    pub start: usize,
    /// Placeholder tokens (= rows of encoder output).
    pub tokens: usize,
    /// The merged patch grid (rows, columns); `tokens == rows * columns`.
    pub merged: (usize, usize),
    pub features: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    /// The image's identity for the prefix cache: a hash of everything that decides its encoder output (the preprocessed pixels, their size, the
    /// preprocessing's id: `sched::image_content_hash`). Two spans with the same key must carry the same features. **0 means unknown: a request
    /// with such a span is never cached.**
    pub key: u64,
}

impl std::fmt::Debug for ImageSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ImageSpan {{ start: {}, tokens: {}, merged: {:?} }}", self.start, self.tokens, self.merged)
    }
}

#[derive(Clone, Debug)]
pub struct Request {
    pub prompt: Vec<u32>,
    pub sampling: Sampling,
    pub max_new: usize,
    pub stop: Vec<u32>,
    /// Speculate for this request (when the engine can). Off gives the serial reference path.
    pub draft: bool,
    /// Images in the prompt, in order; empty for text. Their placeholder runs are already expanded in `prompt`.
    pub images: Vec<ImageSpan>,
    /// May use the prefix cache (look a prefix up, leave snapshots of this prompt). `false` is the uncached reference path: nothing is
    /// looked up and nothing is stored. Ignored when the engine has no cache.
    pub cache: bool,
}

/// A cap on a reply's open reasoning block (`docs/NEMOTRON_REASONING_API.md`, "Reasoning budget"). The reply starts inside a thought that
/// `close` ends. When the reply's generated token at index `after` (0-based: after `after` thinking tokens) would be committed while no
/// `close` has been committed yet, `close` is committed there instead of the sample. Nothing else changes.
///
/// Exactness (drafted == serial): the rule is a function of the committed tokens only. Every round clips the drafts so that no verify row
/// samples index `after` while the thought is still open (so a drafted round never needs a sample there that depends on unverified drafts),
/// and, independently of the clip, the commit loops apply the rule to the tokens they commit and cut the accepted run at a forced token
/// (rows past it were conditioned on a draft that is no longer the committed token): the serial path and every drafted path commit the
/// same token at every index. A sequence without a budget is untouched (no clip, no check).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReasoningBudget {
    /// Thinking tokens allowed before `close` is forced (the forced token is the reply's token at this index).
    pub after: usize,
    /// The token id that ends the thought (`</think>`).
    pub close: u32,
}

/// How often the reasoning budget acted ([`Engine::budget_stats`]): tokens forced, and of those how many cut a drafted run short (only
/// possible when the draft clip is off or a drafter returned more than it was asked for).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BudgetStats {
    pub forced: u64,
    pub cut_runs: u64,
}

/// Apply `budget` (already known to be open at the start of the run) to the run of tokens a round is about to commit, generated index
/// `g` onwards: the committed tokens are `run(0..=last)`. Returns the forced index within the run, if the rule forces a token there (the
/// caller commits `close` at that index and nothing after it).
fn budget_force_in_run(budget: ReasoningBudget, g: usize, last: usize, run: impl Fn(usize) -> u32) -> Option<usize> {
    for idx in 0..=last {
        let t = run(idx);
        if g + idx == budget.after {
            return (t != budget.close).then_some(idx);
        }
        if t == budget.close {
            return None; // the model closed the thought itself: the budget is spent
        }
    }
    None
}

/// Why a sequence ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// A stop token (the model's end-of-turn, or one the request named).
    Stop,
    /// `max_new` tokens generated.
    Length,
    /// Removed by `Engine::cancel`.
    Cancelled,
}

/// Tokens newly committed to one sequence since the last `take_events`, in order, and how it ended if it did.
/// A server turns these into streamed chunks. Concatenating a sequence's events gives exactly its `Output`.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub id: u64,
    pub tokens: Vec<u32>,
    pub finish: Option<FinishReason>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Output {
    pub id: u64,
    pub tokens: Vec<u32>,
    pub rounds: u64,
    pub drafted: u64,
    pub accepted: u64,
    /// Finished on a stop token (rather than reaching `max_new`).
    pub stopped: bool,
    /// The stop token was an accepted draft or the sample past one, not the round's first sample.
    pub stopped_in_draft: bool,
    pub finish: Option<FinishReason>,
    /// Prompt tokens a prefix-cache hit supplied (they were not prefilled for this request).
    pub cached_tokens: usize,
}

impl Output {
    /// Mean tokens committed per verify round (the acceptance length).
    pub fn tokens_per_round(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            self.tokens.len().saturating_sub(1) as f64 / self.rounds as f64
        }
    }
}

struct Seq {
    slot: usize,
    tokens: Vec<u32>,
    prompt_len: usize,
    req: Request,
    stats: AcceptStats,
    /// Conditional acceptance by draft position (what the `Curve` and `Plan` policies read).
    curve: AcceptCurve,
    /// How this stream has done against the margin model's predictions (the `Margin` policy scales its predictions by it).
    cal: Calibration,
    drafter: Box<dyn Drafter>,
    draft: bool,
    mode: Mode,
    out: Output,
    /// Generated tokens already reported through `take_events`.
    emitted: usize,
    /// How this stream has done against the tree model's node probabilities (a tree round scales them by it).
    tree_cal: Calibration,
    /// The stream's copy index for the copy branch of its draft trees (built on the first tree round that asks for one).
    copy: Option<CopyDrafter>,
    /// Drafts the stream's last tree round KEPT (the accepted path's length past the anchor): what the depth-adaptive block
    /// (`gb_sched::adaptive_block`) reads to choose the next round's draft depth.
    kept_depth: usize,
    /// Whether that round's DEEPEST verified level was itself kept (the block may be too short for how deep the stream wants).
    last_level_kept: bool,
    /// The reasoning budget while the thought is still open (`None` without a budget, and once `close` was committed).
    budget: Option<ReasoningBudget>,
}

impl Seq {
    fn generated(&self) -> usize {
        self.tokens.len() - self.prompt_len
    }
    /// The most drafts a round may verify so that no row samples the budget's index while the thought is open (`None`: no limit).
    fn budget_draft_cap(&self) -> Option<usize> {
        self.budget.map(|b| b.after.saturating_sub(self.generated() + 1))
    }
    /// Append a committed token; true when the sequence is finished.
    fn push(&mut self, t: u32) -> bool {
        // `REXL3_DEBUG_TOKENS=1`: the ids a stream commits, so a reply that decodes to nothing can be traced back to
        // either the sampler or the detokeniser. One line per token, on stderr like the op timing.
        if std::env::var_os("REXL3_DEBUG_TOKENS").is_some() {
            eprintln!("rexl3 event=token id={t} index={}", self.generated());
        }
        if self.budget.is_some_and(|b| b.close == t) {
            self.budget = None;
        }
        self.tokens.push(t);
        if self.req.stop.contains(&t) {
            self.out.stopped = true;
            self.out.finish = Some(FinishReason::Stop);
            return true;
        }
        if self.generated() >= self.req.max_new {
            self.out.finish = Some(FinishReason::Length);
            return true;
        }
        false
    }
}

pub struct EngineConfig {
    /// Most rows in one forward (all streams, base rows + drafts).
    pub max_rows: usize,
    /// Most drafts one stream may offer per round (DFlash2 blocks of 8 give 7).
    pub max_drafts: usize,
    pub cost: CostModel,
    pub drafting: bool,
}

/// How the engine decides how many of a stream's drafts ride in the verify window. Every policy only changes how many drafts
/// are verified, never which token is committed at a position, so replies are byte-identical across them (a different proposal
/// changes acceptance, not text).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BudgetPolicy {
    /// The first release's: every stream drafts its whole block, each stream's acceptance is one geometric rate, and the
    /// draft is trimmed afterwards by the verify step's cost alone.
    #[default]
    Legacy,
    /// Each stream's measured conditional acceptance by draft position replaces the geometric rate (and the context length
    /// enters the cost); still decided after the draft has run.
    Curve,
    /// `Curve`, decided BEFORE the draft with the draft's own cost in the model, so a stream that is not worth drafting is not
    /// drafted and the draft is asked only for the positions that will be verified.
    Plan,
    /// `Plan`, then trimmed again after the draft has run by the draft's own confidence in each pick: how far a pick leads the
    /// runner-up (its margin) predicts whether the target will choose the same token (AUC 0.87 on the GB10), so a stream whose
    /// first pick is a coin flip is not given verify rows for the picks behind it, and rows go to the chains likely to land.
    /// Only matters when rows are scarce (several streams); with one stream the verify step is flat and nothing is trimmed.
    Margin,
    /// For a drafter whose own confidences are calibrated (`Proposal::accept`) and whose cost does not depend on how many picks are
    /// asked of it (a block that runs whole): the pre-draft plan decides only WHICH streams draft, the streams it picks are asked for
    /// every pick the drafter can give, and the confidences trim what rides in the window after the draft has run, as `Margin` does.
    /// A depth cut before the draft can only lose: it saves no draft work, and a stream's measured curve (a class prior and the
    /// positions the budget let it verify) can hold a pick back that the drafter is sure of, which then never gets tested.
    Scored,
}

impl BudgetPolicy {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "legacy" => Some(Self::Legacy),
            "curve" => Some(Self::Curve),
            "plan" => Some(Self::Plan),
            "margin" => Some(Self::Margin),
            "scored" => Some(Self::Scored),
            _ => None,
        }
    }

    /// The pre-draft plan decides how many picks the drafter is asked for.
    fn plans_before_the_draft(self) -> bool {
        matches!(self, Self::Plan | Self::Margin | Self::Scored)
    }
}

/// How many prefill slices a [`Engine::step`] runs after its decode round, and which waiting prompt gets each. Prefill is per sequence and
/// every kernel it uses is row-invariant, so no policy changes a bit of any reply: only when a prompt's first token arrives, and how long
/// the streams already decoding wait for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillPolicy {
    /// One slice a step, the waiting prompts in turn (the first release's behaviour): the 32nd of 32 simultaneous short prompts waits for
    /// 31 decode rounds and 31 slices.
    Serial,
    /// While prompts queue up, `ceil(waiting / div)` slices a step, at most `cap` (and one while at most one prompt waits, so a lone
    /// request sees nothing different), and never more than [`PREFILL_STEP_ROWS`] prompt rows in a step (so a queue of long prompts still
    /// gets one chunk a step, the stall the serial policy always allowed). Each slice goes to the prompt with the fewest rows left, oldest
    /// among equals; a prompt passed over [`PREFILL_PASS_GUARD`] steps goes first, so a long prompt among short ones is slowed, never starved.
    Burst { div: usize, cap: usize },
}

impl Default for PrefillPolicy {
    /// An engine that does not ask for more serves prompts as the first release did: one slice a step. The policy is a choice of the
    /// family that builds the engine (it needs a backend whose prefill is cheap to share, [`Backend::prefill_slices`]); see
    /// [`PrefillPolicy::BATCHED_SHORT`].
    fn default() -> Self {
        Self::Serial
    }
}

impl PrefillPolicy {
    /// What the Qwen family serves with. Measured on a GB10 at 32 simultaneous short prompts (`docs/EVAL_RESULTS.md`, "c32 short-token
    /// sweep") together with its batched short-prompt prefill: the aggregate at 16 new tokens is 79% above the serial policy's, the
    /// first-token p99 5.4 s -> 2.3 s, at the price of a 0.6 s p99 wait for the streams already decoding while such a burst is prefilled.
    /// Another family opts in by passing this (or its own numbers) to [`Engine::with_prefill_policy`]; nothing else changes its prefill.
    pub const BATCHED_SHORT: Self = Self::Burst { div: 2, cap: 16 };
}

/// The most prompt rows a `Burst` step prefills (a slice counts the rows left in its prompt, at most this many): the rows of one full
/// chunk, so that many short prompts share a step but a queue of long ones gets one chunk, as before.
pub const PREFILL_STEP_ROWS: usize = 4096;

/// Steps after which a waiting prompt that keeps being passed over for shorter ones is served first (`PrefillPolicy::Burst`).
pub const PREFILL_PASS_GUARD: u32 = 8;

/// Pseudo-trials of the logistic prior behind each bin of the margin-to-acceptance table.
const MARGIN_PRIOR_WEIGHT: f64 = 20.0;
/// Expected hits of agreement behind a stream's calibration factor (a stream with less history than this stays near the pooled table).
const CALIBRATION_PRIOR: f64 = 3.0;

/// Pseudo-trials of class prior behind each position of a stream's measured curve.
const PRIOR_WEIGHT: f64 = 4.0;
/// Tests of a position before the class's own counters replace the cold-start guess as the prior.
const PRIOR_MIN_TRIALS: u64 = 30;

/// How the engine verifies DRAFT TREES (`Engine::with_trees`): the search policy and the most nodes one stream's tree may hold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TreeConfig {
    pub policy: Policy,
    /// Most nodes (rows past the anchor) the search ranks for one stream.
    pub max_nodes: usize,
    /// Shortest match (tokens) of a copy branch: a chain copied from the stream's own prompt and output, hung off the end of the draft's
    /// own picks, whose rows are allocated like every other node's. 0: no copy branch.
    pub copy_min_match: usize,
    /// Most tokens one copy branch carries (at most `gb_spec::tree::MAX_COPY`).
    pub max_copy: usize,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self { policy: Policy::default(), max_nodes: 63, copy_min_match: 0, max_copy: gb_spec::tree::MAX_COPY }
    }
}

/// Cumulative successful backend prefill progress and restored tokens; no inference decisions changed.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrefillStats {
    pub processed: u64,
    pub cached: u64,
}

/// What a backend's CUDA-graph cache has done so far (counters only grow). Hits are replays of a captured graph; misses are windows that ran
/// eagerly (a first sighting, a shape marked eager, a failed capture) plus the captures themselves. Reported by a backend that keeps such a
/// cache (`Backend::graph_counters`); the server exports it on `/metrics` and emits nothing for a backend that keeps none.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GraphCounters {
    /// Verify/decode windows that replayed a captured graph.
    pub replays: u64,
    /// Windows captured into a graph (each is then replayed in the same call).
    pub captures: u64,
    /// Windows that ran eagerly.
    pub eager_windows: u64,
    /// Captures that failed for a non-sticky reason (the shape then runs eagerly for the rest of the process).
    pub capture_failures: u64,
    /// Graphs destroyed to keep the cache under its limit (least recently used first).
    pub evictions: u64,
    /// Milliseconds spent capturing, in all.
    pub capture_ms: f64,
    /// The draft head's own graph cache: calls that replayed, captured, and ran eagerly.
    pub draft_replays: u64,
    pub draft_captures: u64,
    pub draft_eager: u64,
}

pub struct Engine<B: Backend> {
    prefill_stats: PrefillStats,
    pub backend: B,
    admission: Admission,
    cfg: EngineConfig,
    /// Verify trees instead of chains (needs a backend with `supports_trees`).
    tree: Option<TreeConfig>,
    policy: BudgetPolicy,
    /// Whether the budget's class priors and margin table are read from the traffic the engine has seen (`with_learning`) or are
    /// the frozen measurements in `gb_spec::measured`.
    learn: bool,
    /// The margin table of each sampling mode when not learning (built once: it never changes).
    frozen_margin: [MarginTable; 2],
    round_cost: RoundCost,
    make_drafter: Box<dyn Fn() -> Box<dyn Drafter>>,
    active: Vec<Seq>,
    /// Admitted sequences whose prompt is still being prefilled, in turn: each `step` runs one slice of the front one and sends it
    /// to the back, so a short prompt behind a long one waits a few slices, not the long prompt's whole prefill.
    prefilling: std::collections::VecDeque<Prefilling>,
    scores: Vec<(u64, score::Scored)>,
    prefill_policy: PrefillPolicy,
    /// Whether a burst's slices go to the backend as ONE `prefill_slices` call (default) or one `prefill_slice` call each.
    prefill_batching: bool,
    /// How long the serving loop waits, when a request wakes the idle engine, for the ones sent with it ([`ServeEngine::gather_window`]).
    gather: std::time::Duration,
    slice_deadline: Option<std::time::Instant>,
    max_step_time: std::time::Duration,
    /// Sequences that failed while the others went on (prefill, a row that could not be sampled, a backend error that served
    /// only them), with the reason (`take_failed`). Each is already released.
    failed: Vec<(u64, Failure)>,
    done: Vec<Output>,
    events: Vec<Event>,
    next_id: u64,
    spec: SpecStats,
    /// The prefix cache, when the backend supports one and it was asked for (`with_prefix_cache`).
    prefix: Option<PrefixState>,
    /// Clip drafts at a reasoning budget's index (on unless a test turns it off to prove the commit-time rule alone).
    budget_clip: bool,
    budget_stats: BudgetStats,
}

struct Prefilling {
    full_row: bool,
    /// The request's reasoning budget ([`Engine::submit_deferred_with_budget`]).
    budget: Option<ReasoningBudget>,
    id: u64,
    slot: usize,
    req: Request,
    done: usize,
    /// The prefix cache's part of this prefill, once planned (the first slice looks the cache up).
    cache: Option<PrefillCache>,
    /// Whether the cache has been consulted for this prompt (it is, once, when the first slice is about to run).
    planned: bool,
    /// Prompt tokens a cache hit supplied.
    hit: usize,
    /// Steps that served other prompts while this one waited (`PrefillPolicy::Burst`'s guard against starving a long prompt).
    passed: u32,
}

impl Prefilling {
    /// The prompt uses the prefix cache: it was restored from a hit, or it has snapshots to take or hold. Its slices run one prompt at a time
    /// (a restore resumes ONE slot; a snapshot is taken inside ONE prompt's chunk, `Backend::prefill_slice_cuts`); only a prompt that does not
    /// is handed to `Backend::prefill_slices` with others.
    fn uses_cache(&self) -> bool {
        self.full_row || self.hit > 0 || self.cache.is_some()
    }
}

/// How one slice of one prompt came out (the input of `Engine::settle_slice`, the one tail every slice ends in).
enum SliceResult {
    /// The backend ran the slice: its progress, and the snapshots it took at the cuts it passed.
    Done { step: PrefillStep, captured: Vec<Captured> },
    /// The backend reported a recoverable error that belongs to this prompt.
    Failed { why: String, captured: Vec<Captured> },
}

impl<B: Backend> Engine<B> {
    pub fn new(backend: B, admission: Admission, cfg: EngineConfig, make_drafter: Box<dyn Fn() -> Box<dyn Drafter>>) -> Self {
        let round_cost = RoundCost {
            verify: StepCost::rows_only(cfg.cost.clone()),
            tree_verify: None,
            draft: None,
            kv_ms_per_token: 0.0,
            kv_pass_ms_per_token: 0.0,
            fixed_ms: 0.0,
        };
        Self {
            prefill_stats: PrefillStats::default(),
            backend,
            admission,
            cfg,
            tree: None,
            policy: BudgetPolicy::Legacy,
            learn: false,
            frozen_margin: Mode::ALL.map(|_| MarginTable::measured(MARGIN_PRIOR_WEIGHT)),
            round_cost,
            make_drafter,
            active: Vec::new(),
            prefilling: std::collections::VecDeque::new(),
            scores: Vec::new(),
            prefill_policy: PrefillPolicy::default(),
            prefill_batching: true,
            gather: std::time::Duration::ZERO,
            slice_deadline: None,
            max_step_time: std::time::Duration::ZERO,
            failed: Vec::new(),
            done: Vec::new(),
            events: Vec::new(),
            next_id: 0,
            spec: SpecStats::default(),
            prefix: None,
            budget_clip: true,
            budget_stats: BudgetStats::default(),
        }
    }

    /// TEST ONLY: turn off the draft clip at a reasoning budget's index, so the commit-time rule is what keeps a drafted reply equal to
    /// the serial one (the tests prove each mechanism alone). Never call this in a server.
    #[doc(hidden)]
    pub fn without_budget_draft_clip(mut self) -> Self {
        self.budget_clip = false;
        self
    }

    /// How often reasoning budgets acted since the engine started.
    pub fn budget_stats(&self) -> BudgetStats {
        self.budget_stats
    }

    /// Choose how drafts are budgeted, with the cost of a round (verify step, draft, attention over the contexts) the `Curve` and
    /// `Plan` policies read. `cost.verify` replaces `EngineConfig::cost` for them.
    pub fn with_budget(mut self, policy: BudgetPolicy, cost: RoundCost) -> Self {
        self.policy = policy;
        self.round_cost = cost;
        self
    }

    /// Verify draft TREES: each round the drafter's lattice becomes a best-first tree per stream, the rows are shared among streams by
    /// expected tokens per millisecond (`gb_sched::plan_trees`), the window is a tree (`Backend::step_tree`) and the accepted path is
    /// committed (`Backend::commit_path`). Only takes effect on a backend that `supports_trees`; the others keep chains. Replies are byte
    /// for byte those of the serial path whatever the tree: a node's logits are those of serial decoding of its root path.
    pub fn with_trees(mut self, cfg: TreeConfig) -> Self {
        self.tree = Some(cfg);
        self
    }

    /// Whether this engine's rounds verify trees.
    pub fn verifies_trees(&self) -> bool {
        self.tree.is_some() && self.backend.supports_trees()
    }

    /// Let the budget learn its class priors and margin table from the traffic this engine sees, instead of using the measured
    /// tables frozen in `gb_spec::measured` (the default). Learning adapts to a workload the GB10 session did not cover, and it
    /// makes the number of drafts a request is offered depend on what the server ran before it (never what text it gets): with it
    /// off, the speculation counters of a request are a function of the request.
    pub fn with_learning(mut self, on: bool) -> Self {
        self.learn = on;
        self
    }

    /// Use the backend's prefix cache (`docs/PREFIX_CACHE.md`): look a prompt's prefix up before prefilling it, restore the state of the
    /// longest stored one, and leave snapshots of this prompt behind. A backend that cannot snapshot (`Backend::prefix_caps` is `None`) or
    /// whose budget cannot hold two entries leaves the engine uncached, whatever this is asked.
    pub fn with_prefix_cache(mut self, cfg: PrefixConfig) -> Self {
        if let Some(caps) = self.backend.prefix_caps() {
            if caps.budget.enabled() {
                self.prefix = Some(PrefixState::new(caps, cfg));
            }
        }
        self
    }

    /// The prefix cache's counters and gauges (all zero, `enabled == false`, when the engine has none).
    pub fn prefix_stats(&self) -> PrefixStats {
        self.prefix.as_ref().map(|p| p.snapshot()).unwrap_or_default()
    }

    /// Whether the budget learns from the traffic (`with_learning`); off for a new engine.
    pub fn learns(&self) -> bool {
        self.learn
    }

    /// Choose how prefill slices are shared out among waiting prompts (see [`PrefillPolicy`]); the default is `Serial`, the first release's one slice a step, which a family leaves unless it passes
    /// [`PrefillPolicy::BATCHED_SHORT`] or its own numbers here.
    pub fn with_prefill_policy(mut self, policy: PrefillPolicy) -> Self {
        self.prefill_policy = policy;
        self
    }

    pub fn prefill_policy(&self) -> PrefillPolicy {
        self.prefill_policy
    }

    /// Whether the slices a burst runs in a step are handed to the backend together (`Backend::prefill_slices`, on by default) or one by
    /// one. Off is the behaviour before batching; replies are the same either way.
    pub fn with_prefill_batching(mut self, on: bool) -> Self {
        self.prefill_batching = on;
        self
    }

    /// When a request arrives at the idle engine, the serving loop waits `d` for the ones sent with it before the first step, so prompts
    /// sent together are filled together (a backend that puts several prompts through one forward: `Backend::prefill_slices`); a request
    /// alone starts `d` later. Zero (the default): the first step starts at once. Moves when slices run, never a bit.
    pub fn with_gather_window(mut self, d: std::time::Duration) -> Self {
        self.gather = d;
        self
    }

    pub fn budget_policy(&self) -> BudgetPolicy {
        self.policy
    }

    /// The class priors of the conditional acceptance by position, per sampling mode, from what the engine has counted so far.
    fn priors(&self) -> [[f64; MAX_POSITIONS]; 2] {
        if self.learn {
            Mode::ALL.map(|m| conditional_prior(&self.spec.mode_total(m), PRIOR_MIN_TRIALS, 0.5, 0.9))
        } else {
            gb_spec::measured::CLASS_PRIOR
        }
    }

    /// What speculation has bought so far: per-position acceptance and the milliseconds of each phase of a round.
    pub fn spec_stats(&self) -> &SpecStats {
        &self.spec
    }

    /// Everything about `req` that no running sequence can change: malformed fields, and a size that can never be admitted (past
    /// the context limit, an overflowing cost, or more than the whole admission capacity). `Ok` does not say a slot is free now
    /// (see [`Engine::would_admit`]). Pure: reserves nothing.
    pub fn validate(&self, req: &Request) -> Result<(), SubmitError> {
        if req.prompt.is_empty() {
            return Err(SubmitError::Invalid("empty prompt".into()));
        }
        if req.max_new == 0 {
            return Err(SubmitError::Invalid("max_new must be at least 1".into()));
        }
        req.sampling.validate().map_err(|e| SubmitError::Invalid(e.to_string()))?;
        if req.sampling.top_k == 0 {
            return Err(SubmitError::Invalid("top_k 0 needs the full-vocabulary path, not built yet".into()));
        }
        let vocab = self.backend.vocab_size();
        if let Some(t) = req.prompt.iter().find(|&&t| t as usize >= vocab) {
            return Err(SubmitError::Invalid(format!("token id {t} is outside the vocabulary of {vocab}")));
        }
        let mut at = 0usize;
        for im in &req.images {
            if im.tokens == 0
                || im.tokens != im.merged.0 * im.merged.1
                || im.start < at
                || im.start + im.tokens > req.prompt.len()
            {
                return Err(SubmitError::Invalid(format!(
                    "image span at {} of {} tokens ({}x{}) does not fit the prompt of {} tokens",
                    im.start,
                    im.tokens,
                    im.merged.0,
                    im.merged.1,
                    req.prompt.len()
                )));
            }
            at = im.start + im.tokens;
        }
        Self::tokens_of(req).and_then(|tokens| self.admission.check(tokens).map_err(SubmitError::from)).map(|_| ())
    }

    /// A lower bound on the time `req`'s prefill takes (`Backend::prefill_floor`), from the longest prefix the cache holds for it now (peeked:
    /// nothing is pinned or counted; one chain hash over the prompt). `None` when the backend states no bound, and then the cache is not read.
    pub fn prefill_floor(&self, req: &Request) -> Option<std::time::Duration> {
        let total = req.prompt.len();
        let cold = self.backend.prefill_floor(0, total)?;
        let from = match (&self.prefix, self.backend.prefix_path(total)) {
            (Some(st), Some(path)) if req.cache && !req.images.iter().any(|im| im.key == 0) => {
                st.index.peek(path, &cache_tokens(&req.prompt, &req.images)).unwrap_or(0)
            }
            _ => 0,
        };
        if from == 0 {
            Some(cold)
        } else {
            self.backend.prefill_floor(from, total)
        }
    }

    /// Prompt plus generation: what admission charges for. An overflow is a refusal, never a wrap.
    fn tokens_of(req: &Request) -> Result<u64, SubmitError> {
        (req.prompt.len() as u64).checked_add(req.max_new as u64).ok_or(AdmitError::TooLarge.into())
    }

    /// What admitting `req` would answer right now (`NoSlot`, `NoMemory`, ...), without reserving anything. A request that passed
    /// [`Engine::validate`] and gets `Ok` here is admitted by `submit_deferred`.
    pub fn would_admit(&self, req: &Request) -> Result<(), AdmitError> {
        if self.active.len() + self.prefilling.len() >= self.cfg.max_rows {
            return Err(AdmitError::NoSlot);
        }
        let tokens = (req.prompt.len() as u64).checked_add(req.max_new as u64).ok_or(AdmitError::TooLarge)?;
        self.admission.can_admit(tokens).map(|_| ())
    }

    /// Admit a request and queue its prefill without running it: the prompt is prefilled one slice a `step` (so the sequences
    /// already decoding keep their cadence), and the first token is sampled and reported when it completes. Fails closed when the
    /// request is malformed or memory, context or a slot is short; on any error nothing stays reserved.
    pub fn submit_deferred(&mut self, req: Request) -> Result<u64, SubmitError> {
        self.submit_deferred_with_budget(req, None)
    }

    /// [`Engine::submit_deferred`] with a reasoning budget ([`ReasoningBudget`]): the reply starts inside a thought, and `close` is
    /// committed at index `budget.after` unless the reply closed the thought before.
    pub fn submit_deferred_with_budget(&mut self, req: Request, budget: Option<ReasoningBudget>) -> Result<u64, SubmitError> {
        self.validate(&req)?;
        self.would_admit(&req)?;
        let tokens = Self::tokens_of(&req)?;
        let slot = self.admission.admit(tokens)?;
        let id = self.next_id;
        self.next_id += 1;
        self.prefilling.push_back(Prefilling {
            id,
            slot,
            req,
            done: 0,
            cache: None,
            planned: false,
            hit: 0,
            passed: 0,
            full_row: false,
            budget,
        });
        Ok(id)
    }

    /// [`Engine::submit`] with a reasoning budget (see [`Engine::submit_deferred_with_budget`]).
    pub fn submit_with_budget(&mut self, req: Request, budget: Option<ReasoningBudget>) -> Result<u64, SubmitError> {
        let id = self.submit_deferred_with_budget(req, budget)?;
        while self.prefilling.iter().any(|p| p.id == id) {
            self.advance_prefill().map_err(|e| SubmitError::Backend(e.to_string()))?;
        }
        match self.failed.iter().position(|(f, _)| *f == id) {
            Some(at) => Err(self.failed.remove(at).1.into_submit_error()),
            None => Ok(id),
        }
    }

    /// Admit, prefill (all of it, now) and sample the first token. `submit_deferred` plus driving the queue until this request has
    /// its first token; on any error nothing stays reserved.
    pub fn submit(&mut self, req: Request) -> Result<u64, SubmitError> {
        let id = self.submit_deferred(req)?;
        while self.prefilling.iter().any(|p| p.id == id) {
            self.advance_prefill().map_err(|e| SubmitError::Backend(e.to_string()))?;
        }
        match self.failed.iter().position(|(f, _)| *f == id) {
            Some(at) => Err(self.failed.remove(at).1.into_submit_error()),
            None => Ok(id),
        }
    }

    /// Ask the backend what went wrong in the call that just returned. A fatal error is returned as one; a recoverable one is
    /// handed back for the caller to apply to whatever the call served.
    fn backend_fault(&mut self) -> Result<Option<String>, StepError> {
        match self.backend.take_error() {
            None => Ok(None),
            Some(e) if e.fatal => Err(StepError::Fatal(e.message)),
            Some(e) => Ok(Some(e.message)),
        }
    }

    /// Give a sequence's slot and memory back and record why it failed (the others go on).
    fn fail_seq(&mut self, slot: usize, id: u64, why: Failure) {
        self.backend.release(slot);
        self.admission.release(slot);
        self.failed.push((id, why));
    }

    /// Consult the prefix cache for a prompt about to start its prefill: restore the longest stored prefix and plan the snapshots this
    /// prefill takes. Runs once per prompt, when its first slice is about to run (not at submission, so a request queued behind the one
    /// that will leave the snapshot still finds it). A request that opted out, carries images, or is on the GEMV path (a prompt of 128
    /// tokens or fewer) touches nothing: its state is not interchangeable with any other path's.
    fn plan_prefix(&mut self, p: &mut Prefilling) -> Result<(), StepError> {
        p.planned = true;
        let total = p.req.prompt.len();
        if self.prefix.is_none() || !p.req.cache || p.req.images.iter().any(|im| im.key == 0) {
            return Ok(());
        }
        let Some(path) = self.backend.prefix_path(total) else { return Ok(()) };
        // the key stream: the prompt's tokens, with each image's placeholders replaced by tokens that name the image's content
        let prompt = std::sync::Arc::new(cache_tokens(&p.req.prompt, &p.req.images));
        let hit = {
            let st = self.prefix.as_mut().unwrap();
            st.stats.lookups += 1;
            st.stats.lookup_tokens += total as u64;
            st.index.lookup(path, &prompt)
        };
        let mut hit_len = 0usize;
        if let Some(hit) = hit {
            let restored = self.backend.prefix_restore(p.slot, &hit.handle, hit.len);
            let fault = self.backend_fault()?; // a fatal error stops the engine; a recoverable one spoils only this restore
            let st = self.prefix.as_mut().unwrap();
            match (restored, fault) {
                (Ok(()), None) => {
                    st.index.unpin(hit.key);
                    st.stats.hits += 1;
                    st.stats.tokens_saved += hit.len as u64;
                    self.prefill_stats.cached = self.prefill_stats.cached.saturating_add(hit.len as u64);
                    p.done = hit.len;
                    p.hit = hit.len;
                    hit_len = hit.len;
                }
                (r, f) => {
                    // the slot is in no usable state and the entry is suspect: drop it and prefill cold (done stays 0, which
                    // begins the slot again)
                    eprintln!(
                        "rexl3 level=warn event=prefix_restore_failed len={} error=\"{}\"",
                        hit.len,
                        f.or(r.err()).unwrap_or_default()
                    );
                    st.stats.restore_failures += 1;
                    if let Some(h) = st.index.remove(hit.key) {
                        self.backend.prefix_release(h);
                    }
                }
            }
        }
        let st = self.prefix.as_mut().unwrap();
        let ghost = st.ghost.observe(path, &prompt);
        // a cut never falls inside an image: it moves to the image's first placeholder (a prefix that ends before it), or goes; that happens BEFORE the
        // gap rule, so a moved or dropped cut does not suppress the one after it
        let (min_cut, images) = (st.cfg.policy.min_cut, &p.req.images);
        // cut_align: a planned cut is moved down to the nearest multiple, still before the gap rule, so
        // a snapshot always ends on a derived-structure boundary (a whole KV page) and a restore never
        // copies a partial page that could carry another sequence's continuation. Off (0/1) for families
        // whose restore is exact at any cut (Flash-Next scrubs the copied partial page).
        let align = st.cfg.cut_align.max(1);
        let fit = |c: usize| outside_images(c, images, min_cut, hit_len).map(|c| gb_sched::prefix::align_down(c, align));
        let mut cuts = gb_sched::prefix::plan_cuts_with(
            &st.cfg.policy,
            total,
            hit_len,
            ghost,
            &|l| st.index.contains(path, &prompt, l),
            &fit,
        );
        cuts.retain(|&c| !st.index.contains(path, &prompt, c));
        if !cuts.is_empty() {
            // a snapshot needs a free slot: evict the least recently used entries until there is one per cut, and take fewer cuts (the
            // latest ones: the end of the prompt serves the next turn) when the whole cache is smaller than the plan
            let free = self.backend.prefix_free_snapshots();
            for v in st.index.evict_for_snapshots(cuts.len(), free) {
                self.backend.prefix_release(v);
            }
            let free = self.backend.prefix_free_snapshots();
            if free < cuts.len() {
                cuts.drain(..cuts.len() - free);
            }
        }
        // progressive cuts (`PrefixConfig::checkpoint_every`): every multiple of the period between the resume point and the end cut, outside
        // images and not stored yet. Only into snapshot slots that are free now (a checkpoint is insurance, it never evicts an entry), and only
        // with two to spare beyond the planned cuts (the next checkpoint is taken before the one it replaces is released)
        let every = st.cfg.checkpoint_every;
        let mut rolling = Vec::new();
        if every > 0 && total.saturating_sub(hit_len) > every {
            let end = total.saturating_sub(st.cfg.policy.tail);
            let mut c = (hit_len / every + 1) * every;
            while c < end {
                // no index probe per checkpoint (each would hash the prompt up to it: quadratic in the prompt); a checkpoint that is stored
                // already comes back from `insert` as a duplicate and is released
                let ac = gb_sched::prefix::align_down(c, align);
                if ac >= min_cut && !cuts.contains(&ac) && outside_images(ac, images, min_cut, hit_len) == Some(ac) {
                    rolling.push(ac);
                }
                c += every;
            }
            if self.backend.prefix_free_snapshots() < cuts.len() + 2 {
                rolling.clear();
            }
        }
        if !rolling.is_empty() {
            cuts.extend_from_slice(&rolling);
            cuts.sort_unstable();
            cuts.dedup();
        }
        if !cuts.is_empty() {
            p.cache = Some(PrefillCache { path, prompt, cuts, pending: Vec::new(), rolling });
        }
        Ok(())
    }

    /// Give back snapshots that will not enter the index: the prompt did not complete (the sequence failed or was cancelled), or a slice failed.
    fn prefix_discard(&mut self, cache: Option<PrefillCache>, extra: Vec<Captured>) {
        let mut n = 0u64;
        for cap in cache.into_iter().flat_map(|c| c.pending).chain(extra) {
            self.backend.prefix_release(cap.handle);
            n += 1;
        }
        if let Some(st) = self.prefix.as_mut() {
            st.stats.captures_discarded += n;
        }
    }

    /// The prompt is complete and its first token was sampled: the snapshots it took become entries.
    fn prefix_commit(&mut self, cache: PrefillCache) {
        let mut release = Vec::new();
        match self.prefix.as_mut() {
            Some(st) => {
                for cap in cache.pending {
                    release.extend(st.index.insert(cache.path, cache.prompt.clone(), cap).into_release());
                }
            }
            None => release.extend(cache.pending.into_iter().map(|c| c.handle)),
        }
        for h in release {
            self.backend.prefix_release(h);
        }
    }

    /// Consult the prefix cache for a prompt whose first slice is about to run (once: `plan_prefix` marks it). Every route a prompt's slice can
    /// take starts here, the one-at-a-time slice and the burst's batched call alike, so no prompt reaches the backend unplanned.
    fn plan_first_slice(&mut self, p: &mut Prefilling) -> Result<(), StepError> {
        if p.planned {
            Ok(())
        } else {
            self.plan_prefix(p)
        }
    }

    /// Run one prefill slice of the next prompt in turn; when its prompt is complete, sample its first token and move it to the
    /// running set (or finish it). Returns Ok(false) when nothing was queued.
    fn advance_prefill(&mut self) -> Result<bool, StepError> {
        let Some(mut p) = self.prefilling.pop_front() else { return Ok(false) };
        if let Err(e) = self.plan_first_slice(&mut p) {
            self.prefilling.push_front(p);
            return Err(e);
        }
        match self.slice_alone(&p) {
            Ok(r) => self.settle_slice(p, r),
            Err(e) => {
                self.prefilling.push_front(p);
                return Err(e);
            }
        }
        Ok(true)
    }

    /// One slice of `p` by itself: `Backend::prefill_slice`, or `prefill_slice_cuts` while the prompt has cuts left to take (a snapshot is
    /// taken inside the chunk of ONE prompt: the only call that can). A recoverable backend error fails this prompt only (`SliceResult::Failed`);
    /// a fatal one is the `Err`, with `p` untouched for the caller to put back.
    fn slice_alone(&mut self, p: &Prefilling) -> Result<SliceResult, StepError> {
        if p.full_row {
            let cuts = p.cache.as_ref().map_or(&[][..], |c| c.cuts.as_slice());
            return match self.backend.score_slice(p.slot, &p.req.prompt, p.done, cuts) {
                Ok(s) => {
                    let valid = s.done > p.done
                        && s.done <= p.req.prompt.len()
                        && s.row.is_some() == (s.done == p.req.prompt.len())
                        && s.row.as_ref().is_none_or(|r| r.len() == self.backend.vocab_size() && r.iter().all(|v| v.is_finite()));
                    if !valid {
                        // a bare "invalid score slice" hides whether the shape or the arithmetic was wrong: name both
                        let (len, bad) = match s.row.as_ref() {
                            None => (0, None),
                            Some(r) => (r.len(), r.iter().position(|v| !v.is_finite())),
                        };
                        eprintln!(
                            "rexl3 level=warn event=score_slice_invalid done={} from={} prompt={} row_len={len} vocab={} first_non_finite={bad:?}",
                            s.done,
                            p.done,
                            p.req.prompt.len(),
                            self.backend.vocab_size()
                        );
                    }
                    if !valid {
                        Ok(SliceResult::Failed { why: "invalid full-vocabulary score slice".into(), captured: s.captured })
                    } else {
                        Ok(SliceResult::Done {
                            step: PrefillStep { done: s.done, row: s.row.map(|values| RowOutput { ids: Vec::new(), values }) },
                            captured: s.captured,
                        })
                    }
                }
                Err(e) if e.fatal => Err(StepError::Fatal(e.message)),
                Err(e) => Ok(SliceResult::Failed { why: e.message, captured: Vec::new() }),
            };
        }
        let cand = p.req.sampling.candidate_count(self.backend.vocab_size());
        let cuts: &[usize] = p.cache.as_ref().map_or(&[], |c| &c.cuts);
        let (step, captured) = if cuts.is_empty() {
            (self.backend.prefill_slice(p.slot, &p.req.prompt, &p.req.images, p.done, cand), Vec::new())
        } else {
            let c = self.backend.prefill_slice_cuts(p.slot, &p.req.prompt, &p.req.images, p.done, cand, cuts);
            (c.step, c.captured)
        };
        match self.backend_fault() {
            Ok(None) => Ok(SliceResult::Done { step, captured }),
            Ok(Some(why)) => Ok(SliceResult::Failed { why, captured }),
            Err(e) => {
                // a fatal error: the snapshots this slice handed over are the engine's, and nothing else will give them back
                self.prefix_discard(None, captured);
                Err(e)
            }
        }
    }

    /// The tail every slice of every prompt ends in, whichever call ran it (`slice_alone`, or one prompt's step of the burst's batched
    /// `prefill_slices`): a failed slice gives the prompt's snapshots (those of earlier slices and any this slice took) and its slot back; a
    /// slice that took snapshots keeps them on the prompt; an unfinished prompt goes to the back of the queue; a finished one is completed
    /// (`complete_prefill`: first token, snapshots committed, `cached_tokens` recorded).
    fn settle_slice(&mut self, mut p: Prefilling, r: SliceResult) {
        let (step, mut captured) = match r {
            SliceResult::Failed { why, captured } => {
                // a recoverable backend error spoils this prompt only
                self.prefix_discard(p.cache.take(), captured);
                self.fail_seq(p.slot, p.id, Failure::Backend(why));
                return;
            }
            SliceResult::Done { step, captured } => (step, captured),
        };
        self.prefill_stats.processed = self.prefill_stats.processed.saturating_add(step.done.saturating_sub(p.done) as u64);
        p.done = step.done;
        if !captured.is_empty() {
            match p.cache.as_mut() {
                Some(c) => {
                    c.cuts.retain(|&x| x > step.done);
                    let taken = captured.iter().filter(|x| c.rolling.contains(&x.len)).count() as u64;
                    c.pending.append(&mut captured);
                    // of the checkpoints only the latest is kept: it holds every page the earlier ones do
                    let latest = c.pending.iter().filter(|x| c.rolling.contains(&x.len)).map(|x| x.len).max();
                    let (keep, old): (Vec<Captured>, Vec<Captured>) = std::mem::take(&mut c.pending)
                        .into_iter()
                        .partition(|x| !c.rolling.contains(&x.len) || Some(x.len) == latest);
                    c.pending = keep;
                    let superseded = old.len() as u64;
                    for x in old {
                        self.backend.prefix_release(x.handle);
                    }
                    if let Some(st) = self.prefix.as_mut() {
                        st.stats.checkpoints += taken;
                        st.stats.checkpoints_superseded += superseded;
                    }
                }
                None => self.prefix_discard(None, std::mem::take(&mut captured)),
            }
        }
        match step.row {
            None => self.prefilling.push_back(p), // more to do: to the back, so every prefilling prompt gets a slice in turn
            Some(row) => self.complete_prefill(p, row),
        }
    }

    /// A prompt whose prefill is done, with the candidates after its last token: sample its first token and move it to the running
    /// set (or finish it). A row that cannot be sampled fails this sequence alone and gives its slot back.
    fn complete_prefill(&mut self, mut p: Prefilling, row: RowOutput) {
        if p.full_row {
            if let Some(c) = p.cache.take() {
                self.prefix_commit(c);
            }
            self.backend.release(p.slot);
            self.admission.release(p.slot);
            let logits_rows = self.backend.take_score_rows();
            self.scores.push((p.id, score::Scored { logits_row: row.values, logits_rows, cached_tokens: p.hit }));
            return;
        }
        let first = match try_choose(&row.values, &row.ids, p.req.prompt.len() as u64, &p.req.sampling) {
            // a budget of 0 thinking tokens forces the close as the first token (index 0 is always committed alone)
            Ok(t) => match p.budget {
                Some(b) if b.after == 0 && t != b.close => {
                    self.budget_stats.forced += 1;
                    b.close
                }
                _ => t,
            },
            Err(e) => {
                // the backend already holds this slot's prompt: give everything back (its snapshots included: a prompt whose last row
                // could not be sampled is not trusted to have left sound state)
                self.prefix_discard(p.cache.take(), Vec::new());
                self.fail_seq(p.slot, p.id, Failure::Sampling(e));
                return;
            }
        };
        if let Some(c) = p.cache.take() {
            self.prefix_commit(c);
        }
        let req_draft = p.req.draft;
        let mode = Mode::of(&p.req.sampling);
        let drafter = if self.cfg.drafting { (self.make_drafter)() } else { Box::new(NoDrafts) as Box<dyn Drafter> };
        let mut seq = Seq {
            slot: p.slot,
            tokens: p.req.prompt.clone(),
            prompt_len: p.req.prompt.len(),
            req: p.req,
            stats: AcceptStats::new(0.5, 4.0, 0.9),
            curve: AcceptCurve::new(0.9),
            cal: Calibration::new(0.9),
            drafter,
            draft: req_draft,
            mode,
            out: Output { id: p.id, cached_tokens: p.hit, ..Default::default() },
            emitted: 0,
            tree_cal: Calibration::new(0.9),
            copy: None,
            kept_depth: 0,
            last_level_kept: false,
            budget: p.budget,
        };
        if seq.push(first) {
            self.finish(seq);
        } else {
            Self::report_progress(&mut self.events, &mut seq);
            self.active.push(seq);
        }
    }

    /// Sequences that failed since the last call (a prompt that could not be prefilled, a row that could not be sampled, a backend
    /// error that served only them), with the reason. Each is already released; the other sequences were not touched.
    pub fn take_failed(&mut self) -> Vec<(u64, String)> {
        std::mem::take(&mut self.failed).into_iter().map(|(id, f)| (id, f.message())).collect()
    }

    /// Fail every running (decoding) sequence with `why` and give back its slot and memory; queued prefills are left alone. For
    /// the caller to use after a recoverable `StepError`, which leaves the backend's half-run rows behind. Returns how many.
    pub fn fail_active(&mut self, why: &str) -> usize {
        let n = self.active.len();
        for s in std::mem::take(&mut self.active) {
            self.fail_seq(s.slot, s.out.id, Failure::Backend(why.to_string()));
        }
        n
    }

    /// Prompts still being prefilled.
    pub fn prefilling(&self) -> usize {
        self.prefilling.len()
    }

    /// Work remains: running sequences or queued prefills.
    pub fn busy(&self) -> bool {
        !self.active.is_empty() || !self.prefilling.is_empty()
    }

    fn finish(&mut self, mut seq: Seq) {
        self.backend.release(seq.slot);
        self.admission.release(seq.slot);
        seq.out.tokens = seq.tokens[seq.prompt_len..].to_vec();
        self.events.push(Event {
            id: seq.out.id,
            tokens: seq.tokens[seq.prompt_len + seq.emitted..].to_vec(),
            finish: seq.out.finish,
        });
        self.done.push(seq.out);
    }

    /// Remember the tokens a still-running sequence gained, for `take_events`.
    fn report_progress(events: &mut Vec<Event>, s: &mut Seq) {
        let have = s.generated();
        if have > s.emitted {
            events.push(Event { id: s.out.id, tokens: s.tokens[s.prompt_len + s.emitted..].to_vec(), finish: None });
            s.emitted = have;
        }
    }

    /// Stop one sequence now (a client went away, or a text-level stop string matched). Its slot and memory are
    /// released; what it had generated is kept as a finished `Output` with `FinishReason::Cancelled`.
    /// Returns false when no running sequence has that id.
    pub fn cancel(&mut self, id: u64) -> bool {
        if let Some(pos) = self.prefilling.iter().position(|p| p.id == id) {
            let mut p = self.prefilling.remove(pos).unwrap();
            // with checkpoints on, the snapshots a cancelled prefill completed are offered to the index (each is the state after exactly its
            // cut, taken by a slice that succeeded: what a completed prompt would have committed at that cut); otherwise they are given back
            let keep = self.prefix.as_ref().is_some_and(|st| st.cfg.checkpoint_every > 0);
            match p.cache.take() {
                Some(c) if keep && !c.pending.is_empty() => {
                    if let Some(st) = self.prefix.as_mut() {
                        st.stats.kept_on_cancel += c.pending.len() as u64;
                    }
                    self.prefix_commit(c);
                }
                c => self.prefix_discard(c, Vec::new()),
            }
            self.backend.release(p.slot);
            self.admission.release(p.slot);
            if p.full_row {
                return true;
            }
            self.events.push(Event { id, tokens: Vec::new(), finish: Some(FinishReason::Cancelled) });
            self.done.push(Output { id, finish: Some(FinishReason::Cancelled), ..Default::default() });
            return true;
        }
        let Some(pos) = self.active.iter().position(|s| s.out.id == id) else { return false };
        let mut s = self.active.remove(pos);
        s.out.finish = Some(FinishReason::Cancelled);
        self.finish(s);
        true
    }

    /// Newly committed tokens per sequence since the last call, in commit order, with the finish reason on a
    /// sequence's last event. A sequence's events concatenate to exactly its `Output::tokens`.
    pub fn take_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }

    /// One verify round for every active sequence, then one prefill slice. Returns Ok(false) when nothing is left to run.
    ///
    /// The backend's rows are checked before any bookkeeping changes, so an Err leaves the engine's sequences exactly as they
    /// were (see `StepError`). A row that cannot be sampled fails only its own sequence (`take_failed`); the others go on.
    pub fn step(&mut self) -> Result<bool, StepError> {
        let urgent = self.slice_deadline.is_some_and(|d| {
            d.saturating_duration_since(std::time::Instant::now())
                <= std::time::Duration::from_secs(10)
                    .max(self.max_step_time.saturating_mul(2) + std::time::Duration::from_millis(20))
        });
        if urgent && !self.work_fits() {
            return Ok(false);
        }
        // A single score slice can still monopolize a decode turn if it contains
        // thousands of rows. Reuse the backend-supported, row-invariant slicing
        // limit when score prefill competes with generation. Unsupported backends
        // retain the one-slice fairness guard; urgent teardown stays fail-closed.
        let mixed_score = self.prefilling.iter().any(|p| p.full_row)
            && (!self.active.is_empty() || self.prefilling.iter().any(|p| !p.full_row));
        let limited = (urgent || mixed_score) && self.backend.set_prefill_row_limit(Some(128));
        if urgent && !limited {
            return Ok(false);
        }
        let t = std::time::Instant::now();
        let result = self.step_inner(urgent);
        self.max_step_time = self.max_step_time.max(t.elapsed());
        if limited {
            self.backend.set_prefill_row_limit(None);
        }
        result
    }

    fn work_fits(&self) -> bool {
        self.slice_deadline.is_none_or(|d| {
            let estimate = if self.max_step_time.is_zero() { self.backend.cold_step_budget() } else { Some(self.max_step_time) };
            estimate.is_some_and(|t| {
                d.saturating_duration_since(std::time::Instant::now())
                    > t.saturating_mul(2) + std::time::Duration::from_millis(20)
            })
        })
    }

    fn step_inner(&mut self, urgent: bool) -> Result<bool, StepError> {
        // one decode round for the running sequences, then one prefill slice for the next queued prompt in turn: a long prompt
        // delays the others by one slice a round instead of stalling them for its whole prefill, and a short prompt behind it
        // gets its slice at the next turn
        let decoded = !self.active.is_empty();
        let more = if decoded { self.decode_round()? } else { false };
        // `Serial` runs one slice; `Burst` several while a burst of prompts waits, each to the prompt `pick_next_prefill` puts in front,
        // and the backend gets them in one call when it can put short prompts through one forward
        if urgent && !self.work_fits() {
            return Ok(more);
        }
        let slices = if urgent { 1 } else { self.prefill_slices() };
        // a round that ended its last sequences counts too: their last tokens leave when the step ends, after its prefill
        self.backend.set_decode_waiting(decoded || !self.active.is_empty());
        if matches!(self.prefill_policy, PrefillPolicy::Burst { .. }) {
            // every prompt waiting at the start of the step is passed over by it unless it is served (a pick resets its count): the guard
            // counts steps, not the slices of one step, which are served together
            for p in self.prefilling.iter_mut() {
                p.passed += 1;
            }
        }
        if slices >= 2 && self.prefill_batching {
            self.advance_prefills(slices)?;
        } else {
            let mut rows = 0usize;
            for i in 0..slices {
                let Some(at) = self.choose_prefill(rows) else { break };
                let r = self.slice_rows(at);
                if i > 0 && rows + r > PREFILL_STEP_ROWS {
                    break;
                }
                rows += r;
                self.pick_prefill(at);
                if !self.advance_prefill()? {
                    break;
                }
            }
        }
        Ok(more || self.busy())
    }

    /// Rows a slice of the waiting prompt at `at` counts against [`PREFILL_STEP_ROWS`]: what is left of it, at most one budget.
    fn slice_rows(&self, at: usize) -> usize {
        let p = &self.prefilling[at];
        (p.req.prompt.len() - p.done).min(PREFILL_STEP_ROWS)
    }

    /// One slice for each of the next `k` waiting prompts (the order `pick_next_prefill` gives), in turn. Each prompt first consults the prefix
    /// cache exactly as `advance_prefill` does for its own, right before its slice (`plan_first_slice`: a hit is restored, the cuts are planned),
    /// so a prompt picked behind one that leaves a snapshot in this same step still finds it. One that uses the cache (a hit, or cuts to take:
    /// `Prefilling::uses_cache`) gets its slice at once, alone (`slice_alone`: the call that restores into one slot and takes snapshots inside
    /// one prompt's chunk). The others (prompts of 128 tokens or fewer are always these: they are on the GEMV path and never cached) wait, and
    /// go to the backend together in ONE `prefill_slices` call, which is what lets it share a forward among the short ones. Every prompt then
    /// ends in the same tail, `settle_slice`, whichever call served it: a recoverable backend error fails the prompts whose rows the backend
    /// could not produce (NaN) and no other, a finished prompt is sampled, leaves its snapshots and joins the running set, an unfinished one
    /// goes to the back of the queue. A fatal error puts back every prompt not yet settled, in order, and stops.
    fn advance_prefills(&mut self, k: usize) -> Result<(), StepError> {
        let mut picked: Vec<Option<Prefilling>> = Vec::new();
        let mut rows = 0usize;
        for i in 0..k {
            let Some(at) = self.choose_prefill(rows) else { break };
            let r = self.slice_rows(at);
            if i > 0 && rows + r > PREFILL_STEP_ROWS {
                break;
            }
            rows += r;
            self.pick_prefill(at);
            match self.prefilling.pop_front() {
                Some(p) => picked.push(Some(p)),
                None => break,
            }
        }
        let mut together: Vec<usize> = Vec::new();
        for i in 0..picked.len() {
            if let Err(e) = self.plan_first_slice(picked[i].as_mut().expect("waiting")) {
                self.requeue(picked);
                return Err(e);
            }
            if !picked[i].as_ref().expect("waiting").uses_cache() {
                together.push(i);
                continue;
            }
            let p = picked[i].take().expect("waiting");
            match self.slice_alone(&p) {
                Ok(r) => self.settle_slice(p, r),
                Err(e) => {
                    picked[i] = Some(p);
                    self.requeue(picked);
                    return Err(e);
                }
            }
        }
        if together.is_empty() {
            return Ok(());
        }
        match self.slices_together(&picked, &together) {
            Ok(results) => {
                for (&i, r) in together.iter().zip(results) {
                    let p = picked[i].take().expect("waiting");
                    self.settle_slice(p, r);
                }
                Ok(())
            }
            Err(e) => {
                self.requeue(picked);
                Err(e)
            }
        }
    }

    /// Prompts a fatal error interrupted go back to the front of the queue, in the order they were picked: nothing is lost, and nothing is
    /// failed by an error that is not theirs.
    fn requeue(&mut self, picked: Vec<Option<Prefilling>>) {
        for p in picked.into_iter().rev().flatten() {
            self.prefilling.push_front(p);
        }
    }

    /// The prompts of `picked` at `ix` (none of which uses the cache), one slice each in ONE `Backend::prefill_slices` call, with each
    /// prompt's outcome in the same order. A recoverable error pins on the prompts whose row came back NaN; one that left no NaN row cannot be
    /// pinned on a prompt and fails the whole call. A fatal error, or an answer of the wrong length, is the `Err`.
    fn slices_together(&mut self, picked: &[Option<Prefilling>], ix: &[usize]) -> Result<Vec<SliceResult>, StepError> {
        let vocab = self.backend.vocab_size();
        let ps: Vec<&Prefilling> = ix.iter().map(|&i| picked[i].as_ref().expect("waiting")).collect();
        let steps = {
            let jobs: Vec<PrefillJob<'_>> = ps
                .iter()
                .map(|p| PrefillJob {
                    slot: p.slot,
                    prompt: &p.req.prompt,
                    images: &p.req.images,
                    done: p.done,
                    candidates: p.req.sampling.candidate_count(vocab),
                })
                .collect();
            self.backend.prefill_slices(&jobs)
        };
        let fault = self.backend_fault()?;
        if steps.len() != ps.len() {
            return Err(StepError::BackendShape(format!("{} prefill steps for {} prompts", steps.len(), ps.len())));
        }
        let broken = |step: &PrefillStep| step.row.as_ref().is_some_and(|r| r.values.iter().any(|v| v.is_nan()));
        let blame_all = fault.is_some() && !steps.iter().any(broken);
        Ok(steps
            .into_iter()
            .map(|step| match &fault {
                Some(why) if blame_all || broken(&step) => SliceResult::Failed { why: why.clone(), captured: Vec::new() },
                _ => SliceResult::Done { step, captured: Vec::new() },
            })
            .collect())
    }

    /// Prefill slices this step runs, from the number of prompts waiting now (see [`PrefillPolicy`]).
    fn prefill_slices(&self) -> usize {
        // A full-vocabulary score head cannot share a batched forward. Multiple
        // score heads in one burst postpone the next decode turn even after a
        // short generation prompt wins the queue ordering. Yield after one slice
        // whenever scoring competes with queued or active generation; preserve
        // the original budget for pure generation and score-only workloads.
        if self.prefilling.iter().any(|p| p.full_row) && (!self.active.is_empty() || self.prefilling.iter().any(|p| !p.full_row))
        {
            return 1;
        }
        match self.prefill_policy {
            PrefillPolicy::Serial => 1,
            PrefillPolicy::Burst { div, cap } => {
                let waiting = self.prefilling.len();
                if waiting <= 1 {
                    1
                } else {
                    waiting.div_ceil(div.max(1)).min(cap.max(1)).min(waiting)
                }
            }
        }
    }

    /// Index of the prompt that gets the next slice: the front one under `Serial`; under `Burst` one passed over `PREFILL_PASS_GUARD` steps
    /// (the oldest of them), else the one with the fewest rows left (the oldest among equals). While the backend runs a prompt's slice in
    /// parts ([`Backend::prefill_in_flight`]), that prompt goes first, except that under `Burst` another whose rows left fit the backend's
    /// room (with the `taken` rows this step already holds) goes before it, the fewest rows left first, until the pass-over guard serves
    /// it. `None` with nothing waiting. No side effects.
    fn choose_prefill(&self, taken: usize) -> Option<usize> {
        if self.prefilling.is_empty() {
            return None;
        }
        let burst = matches!(self.prefill_policy, PrefillPolicy::Burst { .. });
        let left = |p: &Prefilling| p.req.prompt.len() - p.done;
        if let Some((slot, room)) = self.backend.prefill_in_flight() {
            let pin = self.prefilling.iter().position(|p| p.slot == slot);
            if let Some(at) = pin.filter(|&at| !burst || self.prefilling[at].passed >= PREFILL_PASS_GUARD) {
                return Some(at);
            }
            let fits = self
                .prefilling
                .iter()
                .enumerate()
                .filter(|(_, p)| p.slot != slot && !p.full_row && taken + left(p) <= room)
                .min_by_key(|(i, p)| (left(p), *i))
                .map(|(i, _)| i);
            return fits.or(pin);
        }
        if !burst {
            return Some(0);
        }
        self.prefilling
            .iter()
            .position(|p| p.passed >= PREFILL_PASS_GUARD)
            .or_else(|| self.prefilling.iter().enumerate().min_by_key(|(i, p)| (left(p), *i)).map(|(i, _)| i))
    }

    /// Put the prompt [`Engine::choose_prefill`] chose (at `at`) at the front of the queue, where the slice takes it from. Under `Burst`
    /// its own passed-over count restarts when others wait (they were counted at the start of the step).
    fn pick_prefill(&mut self, at: usize) {
        if matches!(self.prefill_policy, PrefillPolicy::Burst { .. }) && self.prefilling.len() >= 2 {
            self.prefilling[at].passed = 0;
        }
        if at > 0 {
            let p = self.prefilling.remove(at).expect("an index inside the queue");
            self.prefilling.push_front(p);
        }
    }

    fn decode_round(&mut self) -> Result<bool, StepError> {
        if self.active.is_empty() {
            return Ok(false);
        }
        if self.verifies_trees() {
            return self.decode_round_trees();
        }
        let maxes: Vec<usize> = self
            .active
            .iter()
            .map(|s| {
                let left = s.req.max_new - s.generated();
                if !self.cfg.drafting || !s.draft {
                    0
                } else {
                    let m = self.cfg.max_drafts.min(left.saturating_sub(1));
                    match s.budget_draft_cap().filter(|_| self.budget_clip) {
                        Some(cap) => m.min(cap),
                        None => m,
                    }
                }
            })
            .collect();
        // How many drafts to ask of the drafter for each stream. `Plan` decides this now, from measured acceptance and the cost of
        // the draft itself, so a short plan saves draft work; the others ask for everything and trim after the draft has run.
        let priors = if self.policy == BudgetPolicy::Legacy { None } else { Some(self.priors()) };
        let curve_demands = |s: &Seq, drafts: usize| -> Demand {
            let prior = &priors.as_ref().expect("a measured policy has priors")[s.mode as usize];
            Demand::from_conditional(&s.curve.curve(prior, PRIOR_WEIGHT), drafts).with_kv_tokens(s.tokens.len() as u64)
        };
        let ask_max: Vec<usize> = if self.policy.plans_before_the_draft() {
            let demands: Vec<Demand> = self.active.iter().zip(&maxes).map(|(s, &m)| curve_demands(s, m)).collect();
            let alloc = plan_round(&self.round_cost, &demands, self.cfg.max_rows, false).alloc;
            if self.policy == BudgetPolicy::Scored {
                // the plan picked the streams worth a draft round; the drafter is asked for all it has and its scores do the trimming
                alloc.iter().zip(&maxes).map(|(&a, &m)| if a > 0 { m } else { 0 }).collect()
            } else {
                alloc
            }
        } else {
            maxes.clone()
        };
        // one backend call for every sequence that wants drafts, so a drafter that batches reads its weights once
        let asks: Vec<usize> = (0..self.active.len()).filter(|&i| ask_max[i] > 0).collect();
        let t_draft = Instant::now();
        let native = {
            let reqs: Vec<ProposeRequest> = asks
                .iter()
                .map(|&i| {
                    let s = &self.active[i];
                    ProposeRequest { slot: s.slot, tokens: &s.tokens, sampling: &s.req.sampling, max: ask_max[i] }
                })
                .collect();
            self.backend.propose_batch_scored(&reqs)
        };
        // a recoverable drafting error only costs this round its native drafts (the sequences' own drafters fill in; drafts
        // never change what is emitted); a fatal one stops the engine
        self.backend_fault()?;
        let mut proposals: Vec<Proposal> = vec![Proposal::default(); self.active.len()];
        for (&i, d) in asks.iter().zip(native) {
            let s = &mut self.active[i];
            proposals[i] = match d {
                Some(d) => d,
                None => Proposal { tokens: s.drafter.propose(&s.tokens, &s.req.sampling, ask_max[i]), ..Default::default() },
            };
        }
        let draft_ns = t_draft.elapsed().as_nanos() as u64;
        // what the draft that ran is modelled to have cost: a decision made after it holds the round to its whole-round rate
        let spent_draft_ms = self.round_cost.draft_ms(asks.len(), proposals.iter().map(|p| p.tokens.len()).sum());
        // the margin model's probability for each pick (the `Margin` policy), kept to score the model against what lands
        let mut margin_preds: Vec<Vec<f64>> = vec![Vec::new(); self.active.len()];
        let alloc: Vec<usize> = match self.policy {
            BudgetPolicy::Legacy => {
                let demands: Vec<Demand> = self
                    .active
                    .iter()
                    .zip(&proposals)
                    .map(|(s, p)| Demand::geometric(s.stats.accept_rate(), p.tokens.len()))
                    .collect();
                choose_budget(&self.cfg.cost, &demands, self.cfg.max_rows).1
            }
            BudgetPolicy::Curve => {
                let demands: Vec<Demand> =
                    self.active.iter().zip(&proposals).map(|(s, p)| curve_demands(s, p.tokens.len())).collect();
                plan_round_charged(&self.round_cost, &demands, self.cfg.max_rows, DraftCharge::Spent(spent_draft_ms)).alloc
            }
            // already planned: the drafter was asked for exactly the drafts that ride (fewer if it had fewer to give)
            BudgetPolicy::Plan => proposals.iter().map(|p| p.tokens.len()).collect(),
            BudgetPolicy::Margin | BudgetPolicy::Scored => {
                // each pick's own chance of landing, from how far it leads the runner-up; picks without a margin (a drafter that
                // does not report one) keep the stream's measured curve
                let tables = if self.learn {
                    Mode::ALL.map(|m| MarginTable::from_stats(&self.spec.margin[m as usize], MARGIN_PRIOR_WEIGHT))
                } else {
                    self.frozen_margin.clone()
                };
                let mut demands: Vec<Demand> = Vec::with_capacity(self.active.len());
                for (i, (s, p)) in self.active.iter().zip(&proposals).enumerate() {
                    let n = p.tokens.len();
                    if n > 0 && p.accept.len() >= n {
                        // the drafter's own calibrated chance for each pick; kept honest per stream by the same calibration factor
                        margin_preds[i] = p.accept[..n].iter().map(|&a| f64::from(a).clamp(0.0, 1.0)).collect();
                        let f = s.cal.factor(CALIBRATION_PRIOR);
                        let cond: Vec<f64> = margin_preds[i].iter().map(|&q| (q * f).min(1.0)).collect();
                        demands.push(Demand::from_conditional(&cond, n).with_kv_tokens(s.tokens.len() as u64));
                    } else if n > 0 && p.margin.len() >= n {
                        // the pooled table's probability, kept honest per stream: scaled by how this stream has done against it
                        let t = &tables[s.mode as usize];
                        margin_preds[i] = p.margin[..n].iter().map(|&m| t.p(m)).collect();
                        let f = s.cal.factor(CALIBRATION_PRIOR);
                        let cond: Vec<f64> = margin_preds[i].iter().map(|&q| (q * f).min(1.0)).collect();
                        demands.push(Demand::from_conditional(&cond, n).with_kv_tokens(s.tokens.len() as u64));
                    } else {
                        demands.push(curve_demands(s, n));
                    }
                }
                plan_round_charged(&self.round_cost, &demands, self.cfg.max_rows, DraftCharge::Spent(spent_draft_ms)).alloc
            }
        };
        let reqs: Vec<RowRequest> = self
            .active
            .iter()
            .zip(&proposals)
            .zip(&alloc)
            .map(|((s, p), &d)| {
                let mut tokens = vec![*s.tokens.last().unwrap()];
                tokens.extend_from_slice(&p.tokens[..d]);
                RowRequest { slot: s.slot, tokens, first_position: (s.tokens.len() - 1) as u64 }
            })
            .collect();
        let vocab = self.backend.vocab_size();
        let cand = self.active.iter().map(|s| s.req.sampling.candidate_count(vocab)).max().unwrap();
        // the backend may verify all-greedy windows on the device (fewer readback bytes); the flag of
        // each request is exactly "samples at temperature 0" and a backend must ignore it if it cannot
        let greedy: Vec<bool> = self.active.iter().map(|s| s.req.sampling.temperature == 0.0).collect();
        let t_verify = Instant::now();
        let outs = self.backend.step_verified(&reqs, cand, &greedy);
        let verify_ns = t_verify.elapsed().as_nanos() as u64;
        // the backend's own account of the step comes first: a step that reported an error has no usable rows
        if let Some(m) = self.backend_fault()? {
            return Err(StepError::Backend(m));
        }
        if outs.len() != reqs.len() {
            return Err(StepError::BackendShape(format!("{} outputs for {} requests", outs.len(), reqs.len())));
        }
        for (i, (rows, req)) in outs.iter().zip(&reqs).enumerate() {
            if rows.len() != req.tokens.len() {
                return Err(StepError::BackendShape(format!("request {i}: {} rows for {} tokens", rows.len(), req.tokens.len())));
            }
        }
        // each sequence's rows are sampled on their own: one that cannot be (a NaN logit) fails alone below
        let all_samples: Vec<Result<Vec<u32>, SamplingError>> = self
            .active
            .iter()
            .zip(&outs)
            .map(|(s, rows)| {
                let base = s.tokens.len() as u64;
                rows.iter().enumerate().map(|(j, r)| try_choose(&r.values, &r.ids, base + j as u64, &s.req.sampling)).collect()
            })
            .collect();
        let mut still = Vec::with_capacity(self.active.len());
        let mut fatal: Option<StepError> = None;
        let streams = reqs.len();
        let rows: usize = reqs.iter().map(|r| r.tokens.len()).sum();
        let mut commit_ns = 0u64;
        for ((((mut s, req), samples), prop), preds) in
            std::mem::take(&mut self.active).into_iter().zip(&reqs).zip(all_samples).zip(&proposals).zip(&margin_preds)
        {
            if fatal.is_some() {
                still.push(s); // the engine is stopping: leave the rest as they were
                continue;
            }
            let mut samples = match samples {
                Ok(v) => v,
                Err(e) => {
                    self.fail_seq(s.slot, s.out.id, Failure::Sampling(e));
                    continue;
                }
            };
            let drafts = &req.tokens[1..];
            let mut a = accept(drafts, &samples);
            // the reasoning budget, on the tokens this round commits: a forced close ends the run there (rows past it were conditioned
            // on the draft it replaces)
            if let Some(b) = s.budget {
                if let Some(j) = budget_force_in_run(b, s.generated(), a, |i| samples[i]) {
                    samples[j] = b.close;
                    self.budget_stats.forced += 1;
                    if j < a {
                        self.budget_stats.cut_runs += 1;
                    }
                    a = j;
                }
            }
            s.stats.update(drafts.len(), a);
            if drafts.is_empty() {
                // a stream budgeted no drafts learns nothing from this round: every estimate that could keep it at zero drifts back
                // to its prior (the legacy rate, the measured curve and the margin model's calibration)
                s.stats.relax();
                s.curve.relax();
                s.cal.relax();
            }
            s.cal.update(preds, drafts.len(), a);
            s.curve.update(drafts.len(), a);
            self.spec.record(
                s.mode,
                streams,
                prop.tokens.len(),
                &prop.conf[..prop.conf.len().min(drafts.len())],
                &prop.margin[..prop.margin.len().min(drafts.len())],
                drafts.len(),
                a,
            );
            s.out.rounds += 1;
            s.out.drafted += drafts.len() as u64;
            s.out.accepted += a as u64;
            let mut finished = false;
            for (idx, &t) in samples[..=a].iter().enumerate() {
                if s.push(t) {
                    finished = true;
                    s.out.stopped_in_draft = s.out.stopped && idx > 0;
                    break;
                }
            }
            if finished {
                // no commit: the slot's rows are dropped with the slot
                self.finish(s);
            } else {
                let t = Instant::now();
                self.backend.commit(s.slot, a + 1);
                commit_ns += t.elapsed().as_nanos() as u64;
                match self.backend_fault() {
                    Ok(None) => {
                        Self::report_progress(&mut self.events, &mut s);
                        still.push(s);
                    }
                    // its state may be half-committed: this sequence ends here, the others are unaffected
                    Ok(Some(m)) => self.fail_seq(s.slot, s.out.id, Failure::Backend(m)),
                    Err(e) => {
                        fatal = Some(e);
                        still.push(s);
                    }
                }
            }
        }
        self.spec.phases.record(streams, rows, asks.len(), draft_ns, verify_ns, commit_ns);
        self.active = still;
        match fatal {
            Some(e) => Err(e),
            None => Ok(!self.active.is_empty()),
        }
    }

    /// One verify round with TREES (`Engine::with_trees`, on a backend that verifies them). Same contract as [`Engine::decode_round`]: the
    /// committed tokens are the target's keyed samples, a stream that cannot be sampled fails alone, nothing here changes a byte of any
    /// reply. What differs: the drafter's lattice (or its chain) becomes a best-first tree per stream, the rows are shared by expected
    /// tokens per millisecond over the nodes' calibrated probabilities, row `r` of a window is sampled at its own depth's position, and
    /// the round commits the accepted PATH.
    fn decode_round_trees(&mut self) -> Result<bool, StepError> {
        let tcfg = self.tree.expect("a tree round needs a tree configuration");
        let n = self.active.len();
        // the depth-adaptive block (`gb_sched::adaptive_block`, the idea of TensorFold's `_deepest` re-expressed: no code of
        // theirs): each stream's last tree round kept `kept` levels, so next round's draft runs `kept + 2` rows of levels (or
        // `kept + 5` when the block's last level was itself kept), floored at the trained 8 the checkpoint and the calibration
        // cover, capped at the kernels' 16, and back at the trained 8 past 4 streams (a 16-row block halves the sequences a
        // draft round carries). The block only changes how deep the draft PROPOSES: nothing a draft does can move a committed
        // token, and a smaller ask is a truncation the existing exactness covers.
        let blocks: Vec<usize> =
            self.active.iter().map(|s| gb_sched::adaptive_block(s.kept_depth, s.last_level_kept, n)).collect();
        let maxes: Vec<usize> = self
            .active
            .iter()
            .zip(&blocks)
            .map(|(s, &b)| {
                let left = s.req.max_new - s.generated();
                if !self.cfg.drafting || !s.draft {
                    0
                } else {
                    let m = (b - 1).min(left.saturating_sub(1));
                    match s.budget_draft_cap().filter(|_| self.budget_clip) {
                        Some(cap) => m.min(cap),
                        None => m,
                    }
                }
            })
            .collect();
        // the draft runs its whole block for every stream that may draft: the tree search needs every level
        let asks: Vec<usize> = (0..n).filter(|&i| maxes[i] > 0).collect();
        let t_draft = Instant::now();
        let native = {
            let reqs: Vec<ProposeRequest> = asks
                .iter()
                .map(|&i| {
                    let s = &self.active[i];
                    // the depth-adaptive ask of this round (the block the stream's own record says it wants), not a flat 7
                    ProposeRequest { slot: s.slot, tokens: &s.tokens, sampling: &s.req.sampling, max: maxes[i] }
                })
                .collect();
            self.backend.propose_batch_scored(&reqs)
        };
        self.backend_fault()?;
        let mut proposals: Vec<Proposal> = vec![Proposal::default(); n];
        for (&i, d) in asks.iter().zip(native) {
            let s = &mut self.active[i];
            proposals[i] = match d {
                Some(d) => d,
                None => {
                    Proposal { tokens: s.drafter.propose(&s.tokens, &s.req.sampling, self.cfg.max_drafts), ..Default::default() }
                }
            };
        }
        let draft_ns = t_draft.elapsed().as_nanos() as u64;
        let t_plan = Instant::now();
        let spent_draft_ms = self.round_cost.draft_ms(asks.len(), proposals.iter().map(|p| p.tokens.len()).sum());
        // the trees and the probability of each node: the search's raw path number through the frozen calibration (a chain from a
        // drafter without a lattice takes the conditional-acceptance curve of the stream), scaled by how this stream has done
        let priors = self.priors();
        struct Cand {
            tree: DraftTree,
            /// P(node on the accepted path) by the model, in pop order.
            model: Vec<f64>,
            /// The same scaled by the stream's own record: what the allocation spends rows by.
            probs: Vec<f64>,
            /// Which nodes are tokens copied from the stream's own text (the copy branch).
            copied: Vec<bool>,
            from_lattice: bool,
        }
        // the copy continuation of each stream (needs its copy index, so before the immutable pass below)
        let copies: Vec<(usize, Vec<u32>)> = if tcfg.copy_min_match > 0 {
            self.active
                .iter_mut()
                .enumerate()
                .map(|(i, s)| match &proposals[i].lattice {
                    Some(_) if maxes[i] > 0 && !proposals[i].tokens.is_empty() => {
                        let cd = s.copy.get_or_insert_with(|| CopyDrafter::new(tcfg.copy_min_match, tcfg.max_copy));
                        let (m, cont) = cd.propose_after(&s.tokens, &proposals[i].tokens, tcfg.max_copy);
                        // a continuation shorter than 3 tokens is not worth a branch
                        if cont.len() >= 3 {
                            (m, cont)
                        } else {
                            (0, Vec::new())
                        }
                    }
                    _ => (0, Vec::new()),
                })
                .collect()
        } else {
            vec![(0, Vec::new()); n]
        };
        let mut cands: Vec<Cand> = Vec::with_capacity(n);
        for (i, s) in self.active.iter().enumerate() {
            let p = &proposals[i];
            let f = s.tree_cal.factor(CALIBRATION_PRIOR);
            let c = if maxes[i] == 0 {
                Cand { tree: DraftTree::chain(&[]), model: vec![], probs: vec![], copied: vec![], from_lattice: false }
            } else if let Some(l) = &p.lattice {
                let keyed = (s.req.sampling.temperature > 0.0).then_some((&s.req.sampling, s.tokens.len() as u64));
                let plain = best_first(l, keyed, &tcfg.policy, tcfg.max_nodes);
                let (m, cont) = &copies[i];
                let copying = !cont.is_empty();
                let branch = copying.then_some(CopyBranch { picks: &p.tokens, copy: cont, match_len: *m });
                let (tree, model, copied) = with_copy(&plain, l, keyed, &tcfg.policy, s.mode, branch);
                // a stream with `left` tokens to go gains nothing from nodes deeper than left - 1 (a draft is at most as deep
                // as its block unless it carries a copy branch), and a tree window is never deeper than `--window-rows`
                let left = s.req.max_new - s.generated();
                let cap = (if copying { left.saturating_sub(1) } else { maxes[i] }).min(self.cfg.max_rows.saturating_sub(1));
                let (tree, model, copied) = limit_depth_with(tree, model, copied, cap);
                let probs = model.iter().map(|&q| (q * f).min(1.0)).collect();
                Cand { tree, model, probs, copied, from_lattice: true }
            } else {
                let k = p.tokens.len().min(maxes[i]);
                let demand = Demand::from_conditional(&s.curve.curve(&priors[s.mode as usize], PRIOR_WEIGHT), k);
                let model: Vec<f64> = (1..=k).map(|j| demand.marginal(j)).collect();
                Cand {
                    tree: DraftTree::chain(&p.tokens[..k]),
                    probs: model.clone(),
                    model,
                    copied: vec![false; k],
                    from_lattice: false,
                }
            };
            cands.push(c);
        }
        let demands: Vec<TreeDemand> = self
            .active
            .iter()
            .zip(&cands)
            .zip(&proposals)
            .map(|((s, c), p)| TreeDemand { probs: c.probs.clone(), head_rows: p.tokens.len(), kv_tokens: s.tokens.len() as u64 })
            .collect();
        let plan = plan_trees(&self.round_cost, &demands, self.cfg.max_rows, DraftCharge::Spent(spent_draft_ms));
        let kept: Vec<DraftTree> = cands.iter().zip(&plan.alloc).map(|(c, &k)| c.tree.prefix(k)).collect();
        let plan_ns = t_plan.elapsed().as_nanos() as u64;
        let reqs: Vec<TreeRequest> = self
            .active
            .iter()
            .zip(&kept)
            .map(|(s, t)| {
                let (tokens, parents) = t.rows(*s.tokens.last().unwrap());
                TreeRequest { slot: s.slot, tokens, parents, first_position: (s.tokens.len() - 1) as u64 }
            })
            .collect();
        let vocab = self.backend.vocab_size();
        let cand_count = self.active.iter().map(|s| s.req.sampling.candidate_count(vocab)).max().unwrap();
        let t_verify = Instant::now();
        let outs = self.backend.step_tree(&reqs, cand_count);
        let verify_ns = t_verify.elapsed().as_nanos() as u64;
        if let Some(m) = self.backend_fault()? {
            return Err(StepError::Backend(m));
        }
        if outs.len() != reqs.len() {
            return Err(StepError::BackendShape(format!("{} outputs for {} requests", outs.len(), reqs.len())));
        }
        for (i, (rows, req)) in outs.iter().zip(&reqs).enumerate() {
            if rows.len() != req.tokens.len() {
                return Err(StepError::BackendShape(format!("request {i}: {} rows for {} tokens", rows.len(), req.tokens.len())));
            }
        }
        // every row is sampled at its OWN position (the sequence's length plus its depth): siblings share the draw of their depth
        let all_samples: Vec<Result<Vec<u32>, SamplingError>> = self
            .active
            .iter()
            .zip(&outs)
            .zip(&reqs)
            .map(|((s, rows), req)| {
                let base = s.tokens.len() as u64;
                let depth = req.depths();
                rows.iter()
                    .enumerate()
                    .map(|(j, r)| try_choose(&r.values, &r.ids, base + depth[j] as u64, &s.req.sampling))
                    .collect()
            })
            .collect();
        let mut still = Vec::with_capacity(n);
        let mut fatal: Option<StepError> = None;
        let streams = reqs.len();
        let rows: usize = reqs.iter().map(|r| r.tokens.len()).sum();
        let mut commit_ns = 0u64;
        for (i, (((mut s, req), samples), cand)) in
            std::mem::take(&mut self.active).into_iter().zip(&reqs).zip(all_samples).zip(&cands).enumerate()
        {
            if fatal.is_some() {
                still.push(s); // the engine is stopping: leave the rest as they were
                continue;
            }
            let samples = match samples {
                Ok(v) => v,
                Err(e) => {
                    self.fail_seq(s.slot, s.out.id, Failure::Sampling(e));
                    continue;
                }
            };
            let mut samples = samples;
            let mut path = accept_tree(&req.tokens, &req.parents, &samples);
            // the reasoning budget on the committed path (as in `decode_round`): a forced close ends the path there
            if let Some(b) = s.budget {
                if let Some(j) = budget_force_in_run(b, s.generated(), path.len() - 1, |i| samples[path[i]]) {
                    samples[path[j]] = b.close;
                    self.budget_stats.forced += 1;
                    if j + 1 < path.len() {
                        self.budget_stats.cut_runs += 1;
                    }
                    path.truncate(j + 1);
                }
            }
            let a = path.len() - 1;
            let verified = req.tokens.len() - 1;
            // the depth-adaptive block's inputs: how deep this round's window went and whether its last level was kept
            let window_depth = req.depths().into_iter().max().unwrap_or(0);
            s.kept_depth = a;
            s.last_level_kept = a >= window_depth.max(1) && verified > 0;
            if verified == 0 {
                s.tree_cal.relax();
                s.curve.relax();
            } else {
                // the model expected the verified nodes' probabilities (without this stream's own factor, so the factor tracks the model)
                s.tree_cal.update_tree(cand.model[..verified].iter().sum(), a);
                if !cand.from_lattice {
                    s.curve.update(verified, a);
                }
            }
            self.spec.trees[s.mode as usize].record(&req.parents, &path, cand.tree.len(), cand.model[..verified].iter().sum());
            self.spec.trees[s.mode as usize].record_copy(&cand.copied[..verified], &cand.model[..verified], &path);
            // the per-position counters (what `tools/spec_hist.py` and the B6 tables read) see a tree round as the chain it contains: the window's
            // depth is what was verified, the accepted path's length what was accepted (positions past MAX_POSITIONS only reach the tree counters)
            self.spec.record(s.mode, streams, proposals[i].tokens.len().max(window_depth), &[], &[], window_depth, a);
            s.out.rounds += 1;
            s.out.drafted += verified as u64;
            s.out.accepted += a as u64;
            let mut finished = false;
            for (idx, &r) in path.iter().enumerate() {
                if s.push(samples[r]) {
                    finished = true;
                    s.out.stopped_in_draft = s.out.stopped && idx > 0;
                    break;
                }
            }
            if finished {
                // no commit: the slot's rows are dropped with the slot
                self.finish(s);
            } else {
                let t = Instant::now();
                self.backend.commit_path(s.slot, &path);
                commit_ns += t.elapsed().as_nanos() as u64;
                match self.backend_fault() {
                    Ok(None) => {
                        Self::report_progress(&mut self.events, &mut s);
                        still.push(s);
                    }
                    // its state may be half-committed: this sequence ends here, the others are unaffected
                    Ok(Some(m)) => self.fail_seq(s.slot, s.out.id, Failure::Backend(m)),
                    Err(e) => {
                        fatal = Some(e);
                        still.push(s);
                    }
                }
            }
        }
        self.spec.phases.record(streams, rows, asks.len(), draft_ns, verify_ns, commit_ns);
        self.spec.phases.record_plan(streams, plan_ns);
        self.active = still;
        match fatal {
            Some(e) => Err(e),
            None => Ok(!self.active.is_empty()),
        }
    }

    pub fn run(&mut self) -> Result<(), StepError> {
        while self.step()? {}
        Ok(())
    }

    /// Drop every running sequence and give back its slot and memory (after a `StepError`, or at shutdown).
    /// Returns how many were dropped; their partial output is discarded.
    pub fn abort_all(&mut self) -> usize {
        let n = self.active.len() + self.prefilling.len();
        for s in std::mem::take(&mut self.active) {
            self.backend.release(s.slot);
            self.admission.release(s.slot);
        }
        for mut p in std::mem::take(&mut self.prefilling) {
            self.prefix_discard(p.cache.take(), Vec::new());
            self.backend.release(p.slot);
            self.admission.release(p.slot);
        }
        n
    }

    pub fn admission(&self) -> &Admission {
        &self.admission
    }

    pub fn take_outputs(&mut self) -> Vec<Output> {
        let mut v = std::mem::take(&mut self.done);
        v.sort_by_key(|o| o.id);
        v
    }

    pub fn active(&self) -> usize {
        self.active.len()
    }
}

/// The object-safe view of an [`Engine`] that a serving loop needs.
///
/// A model family's `build` returns `Box<dyn ServeEngine>`: this is the one place the backend type is erased, once, at startup.
/// `Engine<B>` and every `Backend` call inside it stay monomorphised; the loop above it makes one virtual call per engine step
/// (a step is tens of milliseconds of GPU work), never one per row or kernel.
pub trait ServeEngine {
    /// Deadline boundary; defaults preserve third-party family source compatibility. Unknown work must be refused near expiry.
    fn set_slice_deadline(&mut self, _deadline: Option<std::time::Instant>) {}
    /// True only when the engine refuses projected over-budget work at safe boundaries.
    fn supports_slice_deadline(&self) -> bool {
        false
    }
    fn quiesce(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn score_supported(&self) -> bool {
        false
    }
    fn score_validate(&self, _req: &score::ScoreRequest) -> Result<(), SubmitError> {
        Err(score::unsupported())
    }
    fn score_would_admit(&self, _req: &score::ScoreRequest) -> Result<(), AdmitError> {
        Err(AdmitError::NoSlot)
    }
    fn score_submit(&mut self, _req: score::ScoreRequest) -> Result<u64, SubmitError> {
        Err(score::unsupported())
    }
    fn take_scores(&mut self) -> Vec<(u64, score::Scored)> {
        Vec::new()
    }
    /// Reserved KV tokens and declared capacity; absent if not declared.
    fn kv_stats(&self) -> Option<(u64, u64)> {
        None
    }
    /// Admission bytes used (including pinned cache), and capacity; not GPU allocator memory.
    fn admission_stats(&self) -> Option<(u64, u64)> {
        None
    }
    /// Actual successful prefill slice progress; absent for an engine without this instrumentation.
    fn prefill_stats(&self) -> Option<PrefillStats> {
        None
    }
    /// What no running sequence can change about a request: malformed fields and a size that can never be admitted. Pure.
    fn validate(&self, req: &Request) -> Result<(), SubmitError>;
    /// What admitting `req` would answer right now, without reserving anything. Pure.
    fn would_admit(&self, req: &Request) -> Result<(), AdmitError>;
    fn submit_deferred(&mut self, req: Request) -> Result<u64, SubmitError>;
    /// `submit_deferred` with a reasoning budget ([`ReasoningBudget`]). The default enforces nothing, so it refuses a budget (fails closed)
    /// and passes a request without one to `submit_deferred` unchanged.
    fn submit_deferred_with_budget(&mut self, req: Request, budget: Option<ReasoningBudget>) -> Result<u64, SubmitError> {
        match budget {
            None => self.submit_deferred(req),
            Some(_) => Err(SubmitError::Invalid("this engine does not enforce a reasoning budget".into())),
        }
    }
    fn busy(&self) -> bool;
    /// How long the serving loop waits, when a request arrives at this idle engine, for the requests sent with it before the first step
    /// (`Engine::with_gather_window`). The default: none.
    fn gather_window(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
    fn step(&mut self) -> Result<bool, StepError>;
    fn cancel(&mut self, id: u64) -> bool;
    fn abort_all(&mut self) -> usize;
    /// Fail every decoding sequence (queued prefills stay) after a recoverable `StepError`; returns how many.
    fn fail_active(&mut self, why: &str) -> usize;
    fn active(&self) -> usize;
    /// Prompts still being prefilled.
    fn prefilling(&self) -> usize;
    fn take_events(&mut self) -> Vec<Event>;
    fn take_failed(&mut self) -> Vec<(u64, String)>;
    fn take_outputs(&mut self) -> Vec<Output>;
    /// A copy of the speculation counters (per-position acceptance, phase timers). Cheap enough to take once a step.
    fn spec_stats(&self) -> SpecStats;
    /// The prefix cache's counters and gauges (`enabled == false` and zeros when the engine has no cache). Cheap enough to take once a step.
    fn prefix_stats(&self) -> PrefixStats {
        PrefixStats::default()
    }
    /// The most tokens (prompt plus generation) one request may hold on this engine: the smaller of the context window and the KV pool the
    /// plan allocated (`Admission::request_limit`). `None` when the engine declares no limit. What a client needs to decide whether a prompt
    /// fits; the server reports it as `max_model_len`.
    fn request_limit(&self) -> Option<u64> {
        None
    }
    /// The backend's CUDA-graph cache counters (`Backend::graph_counters`); `None` for an engine without graphs.
    fn graph_counters(&self) -> Option<GraphCounters> {
        None
    }
    /// A lower bound on the time `req`'s prefill takes from the longest prefix the cache holds now (`Backend::prefill_floor`); `None` when the
    /// backend states none. Pure: the cache is only peeked.
    fn prefill_floor(&self, _req: &Request) -> Option<std::time::Duration> {
        None
    }
}

impl<B: Backend> ServeEngine for Engine<B> {
    fn set_slice_deadline(&mut self, deadline: Option<std::time::Instant>) {
        self.slice_deadline = deadline;
    }
    fn supports_slice_deadline(&self) -> bool {
        true
    }
    fn quiesce(&mut self) -> Result<(), String> {
        self.backend.quiesce()
    }
    fn score_supported(&self) -> bool {
        Engine::score_supported(self)
    }
    fn score_validate(&self, req: &score::ScoreRequest) -> Result<(), SubmitError> {
        Engine::score_validate(self, req)
    }
    fn score_would_admit(&self, req: &score::ScoreRequest) -> Result<(), AdmitError> {
        Engine::score_would_admit(self, req)
    }
    fn score_submit(&mut self, req: score::ScoreRequest) -> Result<u64, SubmitError> {
        Engine::score_submit(self, req)
    }
    fn take_scores(&mut self) -> Vec<(u64, score::Scored)> {
        Engine::take_scores(self)
    }
    fn kv_stats(&self) -> Option<(u64, u64)> {
        self.admission.kv_tokens().map(|total| (self.admission.kv_tokens_in_use(), total))
    }
    fn admission_stats(&self) -> Option<(u64, u64)> {
        let total = self.admission.capacity();
        Some((total.saturating_sub(self.admission.available()), total))
    }
    fn prefill_stats(&self) -> Option<PrefillStats> {
        Some(self.prefill_stats)
    }
    fn validate(&self, req: &Request) -> Result<(), SubmitError> {
        Engine::validate(self, req)
    }
    fn would_admit(&self, req: &Request) -> Result<(), AdmitError> {
        Engine::would_admit(self, req)
    }
    fn fail_active(&mut self, why: &str) -> usize {
        Engine::fail_active(self, why)
    }
    fn prefilling(&self) -> usize {
        Engine::prefilling(self)
    }
    fn submit_deferred(&mut self, req: Request) -> Result<u64, SubmitError> {
        Engine::submit_deferred(self, req)
    }
    fn submit_deferred_with_budget(&mut self, req: Request, budget: Option<ReasoningBudget>) -> Result<u64, SubmitError> {
        Engine::submit_deferred_with_budget(self, req, budget)
    }
    fn busy(&self) -> bool {
        Engine::busy(self)
    }
    fn gather_window(&self) -> std::time::Duration {
        self.gather
    }
    fn step(&mut self) -> Result<bool, StepError> {
        Engine::step(self)
    }
    fn cancel(&mut self, id: u64) -> bool {
        Engine::cancel(self, id)
    }
    fn abort_all(&mut self) -> usize {
        Engine::abort_all(self)
    }
    fn active(&self) -> usize {
        Engine::active(self)
    }
    fn take_events(&mut self) -> Vec<Event> {
        Engine::take_events(self)
    }
    fn take_failed(&mut self) -> Vec<(u64, String)> {
        Engine::take_failed(self)
    }
    fn take_outputs(&mut self) -> Vec<Output> {
        Engine::take_outputs(self)
    }
    fn spec_stats(&self) -> SpecStats {
        Engine::spec_stats(self).clone()
    }
    fn prefix_stats(&self) -> PrefixStats {
        Engine::prefix_stats(self)
    }
    fn request_limit(&self) -> Option<u64> {
        self.admission.request_limit()
    }
    fn graph_counters(&self) -> Option<GraphCounters> {
        self.backend.graph_counters()
    }
    fn prefill_floor(&self, req: &Request) -> Option<std::time::Duration> {
        Engine::prefill_floor(self, req)
    }
}
