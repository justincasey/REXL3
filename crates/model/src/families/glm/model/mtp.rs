//! The NextN (MTP) drafter's layer on the GPU: layer 45 of the pack, a full MLA + DSA indexer + MoE block on a PLAIN fp32 residual (no mHC),
//! fed by the trunk's tapped hidden state and the embedding of the next token, chained K times to draft K tokens. The drafter only
//! proposes: the engine's verify window decides every token (drafted == serial by construction), so nothing here can change an output
//! byte, only how many tokens a round commits.
//!
//! **The recipe** (what the ExLlamaV3 fork's `Glm5NextMTPModel` does; `docs/GLM53.md`, "The MTP head"): the MTP row at position `m` takes the
//! token `t_m` and the trunk's tap `h_{m-1}`, the POST-final-norm hidden state (the mean of the four mHC streams through `model.norm`, fp16:
//! `Op::HeadMean`'s output) of the row that PREDICTED `t_m`. `out = f16(shared_head.norm(block(eh_proj([enorm(embed(t_m)) | hnorm(h_{m-1})]))))`,
//! `logits = lm_head(out)`, draft = argmax. A draft chains: the next row takes the draft `d` as its token and the previous `out` as its hidden
//! (the fork feeds the drafter's own post-norm output back through `hnorm`). Its entry in the MTP cache is overwritten later by the true
//! (tap, token) pair if the draft is accepted, and simply overwritten if not (a rejected window row is overwritten before anyone reads it,
//! as in the trunk).
//!
//! **State.** The layer has its own latent cache and DSA planes (a 12th MLA layer in the shared pages: same block table, same positions) and a
//! raw-row ring per slot. Per slot, the trunk's committed windows leave their taps in a small store (`TapBook` is the bookkeeping, pure and
//! tested on the CPU): the rows whose MTP entry is not built yet. A draft call absorbs them (one MTP window over all pending rows of all asking
//! sequences, only the last row of each needs the head) and chains `depth - 1` more single rows, all asynchronous on the stream with one sync
//! at the end. The draft state therefore advances only with committed rows: a window's rejected rows never enter the store.
//!
//! The index semantics of `index_share_for_mtp_iteration` (reuse the trunk's last top-k instead of the layer's own indexer) are NOT built:
//! this layer scores with its own indexer, as the fork does; below `index_topk` (2,048) tokens the two are identical.

use super::layers;
use super::{zeroed, GlmModel, RMS_EPS};
use crate::error::ModelError;
use crate::families::glm::alloc::{MAX_CANDIDATES, MAX_ROWS, MTP_MAX_DEPTH, MTP_STEPS, MTP_TAP_CAP, RING_LEN, TOKENIZER_VOCAB};
use crate::families::glm::config::GlmConfig;
use crate::families::glm::device::MtpMoe;

/// The drafter's refusal of a NextN layer that holds only a share of its routed experts (a tensor-parallel rank's map).
pub const MTP_SHARE: &str = "the drafter does not run on a share of the NextN layer's experts";
use crate::families::glm::vision_serve::{GlmImageFeatures, SpliceOp};
use crate::rows::RowBatch;
use gb_cuda::attention::RowIndex;
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::mla::LatentCache;
use gb_cuda::{dsa, glm_glue, glm_mtp, glue, moe, DevBuf, Stream};
use gb_sampling::Sampling;

// ---------------------------------------------------------------------------------------------------------------- bookkeeping

/// Why a slot's draft state cannot serve a call. Any of these switches the slot's MTP off for the rest of its sequence (drafts are
/// proposals: losing them costs speed, never a byte); the slot's next `begin_slot` switches it back on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BookError {
    /// The slot's MTP state was switched off (a gap, an overflow, a mismatch, or rows pending that are an image's).
    Off,
    /// A prefill chunk or a window did not continue where the slot's tapped rows end.
    Gap,
    /// More rows are waiting for an absorb than the slot's tap store holds.
    Overflow,
    /// The tokens the engine reports and the rows the store holds disagree.
    Mismatch,
    /// Nothing is pending: there is no new hidden row to draft from.
    Nothing,
}

/// The bookkeeping of one slot's pending tapped rows. The store holds `n` rows: the taps of the committed positions `first - 1 ..
/// first - 1 + n`, i.e. the hidden inputs of the MTP entries `first .. first + n` that are not built yet (entry `m` pairs tap `h_{m-1}`
/// with token `t_m`); every entry below `first` is built. A fresh slot is `first = 0, n = 0`; after a prompt of `T` tokens it is
/// `first = T, n = 1` (the carry `h_{T-1}`); after an absorb of a sequence of `len` tokens, `first = len, n = 0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TapBook {
    pub ok: bool,
    pub first: usize,
    pub n: usize,
}

impl Default for TapBook {
    fn default() -> Self {
        Self::new()
    }
}

impl TapBook {
    pub const fn new() -> Self {
        Self { ok: true, first: 0, n: 0 }
    }

    /// A new sequence in the slot.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// No drafts from this slot until it is reset (an image in the window, or any [`BookError`]).
    pub fn disable(&mut self) {
        self.ok = false;
    }

    /// A prompt chunk of `rows` rows from position `start` was run through the trunk: the MTP entries `start .. start + rows` are built in
    /// the same call, from the carry (the tap of position `start - 1`, held in the store's row 0; zeros for `start == 0`) and this chunk's
    /// taps, and the chunk's last tap becomes the new carry. Returns whether the chunk's first row reads the carry.
    pub fn prefill_chunk(&mut self, start: usize, rows: usize) -> Result<bool, BookError> {
        if !self.ok {
            return Err(BookError::Off);
        }
        let carry = start > 0;
        let continues = if carry { self.n == 1 && self.first == start } else { self.n == 0 && self.first == 0 };
        if !continues || rows == 0 {
            self.ok = false;
            return Err(BookError::Gap);
        }
        self.first = start + rows;
        self.n = 1;
        Ok(carry)
    }

    /// A trunk window whose first row sat at position `row0_pos` was committed with `keep` rows: their taps join the store (only the kept
    /// rows: the draft state advances only with accepted rows). Returns the store row the first of them goes to.
    pub fn window_commit(&mut self, row0_pos: usize, keep: usize, cap: usize) -> Result<usize, BookError> {
        if !self.ok {
            return Err(BookError::Off);
        }
        if keep == 0 {
            return Ok(self.n);
        }
        if self.n == 0 {
            self.first = row0_pos + 1;
        } else if row0_pos + 1 != self.first + self.n {
            self.ok = false;
            return Err(BookError::Gap);
        }
        if self.n + keep > cap {
            self.ok = false;
            return Err(BookError::Overflow);
        }
        let dst = self.n;
        self.n += keep;
        Ok(dst)
    }

    /// May a draft call for a sequence of `tokens` tokens (the last one not yet fed to the trunk) absorb the store? Its entries `first ..
    /// tokens - 1` need exactly the `n` rows held.
    pub fn absorb_check(&mut self, tokens: usize) -> Result<(), BookError> {
        if !self.ok {
            return Err(BookError::Off);
        }
        if self.n == 0 {
            return Err(BookError::Nothing);
        }
        if tokens != self.first + self.n {
            self.ok = false;
            return Err(BookError::Mismatch);
        }
        Ok(())
    }

    /// [`TapBook::absorb_check`] without switching the slot off on a mismatch: could a draft call absorb now?
    pub fn can_absorb(&self, tokens: usize) -> bool {
        self.ok && self.n > 0 && tokens == self.first + self.n
    }

    /// The store was absorbed for a sequence of `tokens` tokens: entries below `tokens` are built.
    pub fn absorbed(&mut self, tokens: usize) {
        self.first = tokens;
        self.n = 0;
    }
}

// ----------------------------------------------------------------------------------------------------------------------- state

/// A deliberate bug in the drafter, so the tests can show they notice it. Production sets none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpMutant {
    /// The tap is the mean of the four streams WITHOUT the final norm (the `hc_head` instead of the `model.norm` output).
    TapBeforeNorm,
    /// A committed window's tapped rows all enter the store, kept or not (the draft state advances on rejected rows).
    AdvanceAllRows,
    /// The hidden input of an MTP row is the tap of its own position instead of the position before it (the shift is dropped).
    NoShift,
    /// A draft call absorbs only the LAST pending row of each sequence: the entries a previous call wrote speculatively (from the drafter's own
    /// hidden state, for drafts that were then accepted) are never rewritten from the target's true hidden rows (no rollback of the draft state).
    AbsorbLastOnly,
    /// Every draft is the id after the one the head chose: a drafter that is numerically clean and always wrong (its acceptance is about zero), for
    /// the arbiter's fall-back test.
    WrongDrafts,
    /// A prompt chunk's image rows enter the drafter's cache with the placeholder token's embedding, not the image's features (the splice is not
    /// handed to the drafter): the text-only drafter the oracle's `indexed_embeddings` rules out.
    PlaceholderImages,
    /// The restricted head's argmax is not mapped from its column to the token id (the draft is the column number): the drafts are wrong ids.
    RestrictedNoMap,
    /// Keyed drafting (`DraftAsk` with a sampler): the restricted head's candidates reach the keyed pick as COLUMN numbers (the noise is keyed by column),
    /// and the picked column is mapped to its id afterwards: a draft that is a real token, of the wrong noise.
    KeyedNoiseByColumn,
    /// Keyed drafting: every draft draws the noise of the position one past the one the target will sample it at.
    KeyedPositionOff,
    /// Keyed drafting: the sampler's `top_k` is dropped from the records (a pick over every candidate the call fetched).
    KeyedNoTopK,
    /// Keyed drafting: a greedy request is keyed too (temperature 1 in its record).
    KeyedGreedyNoise,
}

/// The strips of 128 token ids of the default restricted draft vocabulary, in addition to the first `prefix_strips` (the tokenizer's merge order puts
/// the frequent tokens first): the numeric block (ids 98,304 to 103,423: the numbers 0 to 999, which the first strips do not hold and the answers of
/// math, code and tables are full of) and the last strip (the special tokens, `</think>`, the chat roles). Ascending, disjoint, touching ranges merged.
/// `vocab` is the head's width (154,880: 1,210 strips).
pub fn vocab_ranges(prefix_strips: usize, vocab: usize) -> Vec<(usize, usize)> {
    let last = vocab / 128;
    let mut parts = [(0, prefix_strips.min(last)), (768, 810.min(last)), (last - 1, last)];
    parts.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (lo, hi) in parts {
        match out.last_mut() {
            Some(prev) if lo <= prev.1 => prev.1 = prev.1.max(hi),
            _ if lo < hi => out.push((lo, hi)),
            _ => {}
        }
    }
    out
}

/// Which vocabulary the drafter's head argmaxes over.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VocabMode {
    /// All of it (the trunk's `lm_head`, as the oracle does).
    Full,
    /// The restricted set every call (a draft only proposes: a token outside the set is never drafted, and nothing is lost in exactness).
    Restricted,
    /// The restricted set in a call where every asking sequence has at least this share of its last tokens inside it, the full head otherwise
    /// (text the set does not cover, such as another script, keeps the full head).
    Adaptive(f32),
}

/// The drafter's own copy of some columns of the trunk's `lm_head` (whole 128-column strips): the head over them reads that share of the 0.48 GB.
/// Every kept column is bit-identical to the full head's.
pub struct SubHead {
    w: gb_cuda::exl3_linear::Exl3Weights,
    /// i32 `[cols]`: the token id of each column.
    map: DevBuf,
    /// fp16 `[MAX_ROWS][cols]`.
    logits: DevBuf,
    ws: DevBuf,
    cols: usize,
    /// Columns that are tokens (the last 24 columns of the full head are padding ids, the last strip's; they are the set's last columns too).
    valid: usize,
    /// Per strip of 128 ids: in the set.
    strip_in: Vec<bool>,
}

impl SubHead {
    /// The share of `ids` that are tokens of the set.
    pub fn coverage(&self, ids: &[u32]) -> f32 {
        if ids.is_empty() {
            return 1.0;
        }
        ids.iter().filter(|&&t| self.strip_in.get(t as usize / 128).copied().unwrap_or(false)).count() as f32 / ids.len() as f32
    }
    pub fn strips(&self) -> usize {
        self.cols / 128
    }
}

/// The drafter's device state and scratch (everything `alloc.rs` prices under `mtp_*`, plus the layer's cache and planes).
pub struct MtpState {
    /// The restricted head, when one was built (`GlmModel::mtp_restrict_vocab`), and when it is used.
    pub(super) sub: Option<SubHead>,
    pub(super) vocab: VocabMode,
    /// Whether the call being run uses the restricted head (decided per call).
    pub(super) vocab_active: bool,
    /// fp16 `[chunk rows][hidden]`: the trunk's last forward's final-normed hidden rows (`Op::HeadMean`'s output), copied before the head
    /// reuses `hidden`.
    pub(super) tap: DevBuf,
    /// fp16 `[slots][MTP_TAP_CAP][hidden]`: per slot, the pending tapped rows.
    store: DevBuf,
    /// fp32 `[MAX_ROWS][hidden]`: the layer's plain residual.
    x: DevBuf,
    /// fp16 `[MAX_ROWS][2 * hidden]`: the fused input `[enorm(embed) | hnorm(hidden)]`.
    cat: DevBuf,
    /// fp16 `[MAX_ROWS][hidden]`: the hidden rows an absorb reads (gathered from the stores).
    h_in: DevBuf,
    /// fp16 `[MAX_ROWS][hidden]`: the block's output after `shared_head.norm`.
    out: DevBuf,
    /// fp16 `[MAX_ROWS][hidden]`: the last row of each sequence of an absorb (the head's input, and the first chain step's hidden).
    lm_in: DevBuf,
    /// i32 `[MAX_ROWS]`: the token ids of an absorb.
    ids: DevBuf,
    /// One (slot, position) index per step of a call.
    rowidx: Vec<RowIndex>,
    /// Per chain depth: the argmax ids (i32) and their logit values (f32) of every asking sequence.
    draft_ids: Vec<DevBuf>,
    draft_vals: Vec<DevBuf>,
    /// Per chain depth, for the keyed pick: the head's top candidates of every asking sequence (ids i32 and values f32, `[MAX_ROWS][MAX_CANDIDATES]`
    /// with a row stride of the call's candidate count; a restricted head's ids are already mapped to token ids).
    cand_ids: Vec<DevBuf>,
    cand_vals: Vec<DevBuf>,
    /// `glm_mtp::KeyedRow` `[MAX_ROWS]`: the sampling record of each sequence of the call being run.
    keyed: DevBuf,
    latent: LatentCache,
    planes: dsa::Planes,
    book: Vec<TapBook>,
    pub mutant: Option<MtpMutant>,
    /// The image rows of the prompt chunk being absorbed (chunk-relative, as the trunk's splice has them): their embedding half is the vision feature
    /// row, not a table row (`glm_mtp::mtp_embed_f32`). Empty for every call but a prompt chunk with an image in it.
    img: Vec<SpliceOp>,
    /// Debug: milliseconds by phase over the calls since profiling was switched on (CUDA events around each phase; it adds a host sync per call).
    profile: Option<std::collections::BTreeMap<&'static str, f64>>,
}

/// What a debug forward of MTP rows captured (the oracle comparison): the last `tail` rows of each stage.
#[derive(Default, Debug)]
pub struct MtpCapture {
    /// fp32 `[tail][hidden]`: `eh_proj`'s output (the oracle's `mtp_in`).
    pub x_in: Vec<f32>,
    /// fp32 `[tail][hidden]`: the block's output on the plain residual (`mtp_block`).
    pub block: Vec<f32>,
    /// fp16 bits `[tail][hidden]`: `shared_head.norm`'s output (`mtp_norm`).
    pub norm: Vec<u16>,
    /// fp16 bits `[tail][vocab]`: the logits through the trunk's `lm_head` (`mtp_logits`).
    pub logits: Vec<u16>,
}

impl MtpState {
    /// `tap_rows`: the rows of the trunk's widest forward (a prefill chunk), which the tap buffer holds.
    pub fn new(cfg: &GlmConfig, kv_blocks: usize, slots: usize, tap_rows: usize) -> Result<Self, ModelError> {
        let (r, h) = (MAX_ROWS, cfg.hidden);
        let cr = tap_rows.max(r);
        let mut rowidx = Vec::new();
        for step in 0..MTP_STEPS {
            // step 0 is the prompt chunk's: as many rows as the chunk, in views of one block (the attention's slabs cut it)
            rowidx.push(if step == 0 {
                RowIndex { slot: DevBuf::zeroed_view(cr * 4)?, pos: DevBuf::zeroed_view(cr * 4)?, rows: cr }
            } else {
                RowIndex::from_host(&vec![0; r], &vec![0; r])?
            });
        }
        let mut draft_ids = Vec::new();
        let mut draft_vals = Vec::new();
        let mut cand_ids = Vec::new();
        let mut cand_vals = Vec::new();
        for _ in 0..MTP_MAX_DEPTH {
            draft_ids.push(zeroed(r * 4)?);
            draft_vals.push(zeroed(r * 4)?);
            cand_ids.push(zeroed(r * MAX_CANDIDATES * 4)?);
            cand_vals.push(zeroed(r * MAX_CANDIDATES * 4)?);
        }
        Ok(Self {
            sub: None,
            vocab: VocabMode::Full,
            vocab_active: false,
            tap: zeroed(cr * h * 2)?,
            store: zeroed(slots * MTP_TAP_CAP * h * 2)?,
            x: zeroed(cr * h * 4)?,
            cat: zeroed(cr * 2 * h * 2)?,
            h_in: zeroed(cr * h * 2)?,
            out: zeroed(cr * h * 2)?,
            lm_in: zeroed(r * h * 2)?,
            ids: zeroed(cr * 4)?,
            rowidx,
            draft_ids,
            draft_vals,
            cand_ids,
            cand_vals,
            keyed: zeroed(r * std::mem::size_of::<glm_mtp::KeyedRow>())?,
            latent: LatentCache::new(kv_blocks, Dtype::F16, cfg.latent_kv)?,
            planes: dsa::Planes::new(kv_blocks, slots, RING_LEN)?,
            book: vec![TapBook::new(); slots],
            mutant: None,
            img: Vec::new(),
            profile: None,
        })
    }

    /// The bytes this state holds, by the names of `alloc::AllocPlan`: `(name, bytes)`, where the cache, ring and pool entries are the
    /// layer's share of the shared names (`mla_latent`, `dsa_rings`, `dsa_pools`).
    pub fn allocations(&self) -> Vec<(&'static str, u64)> {
        let b = |d: &DevBuf| d.bytes() as u64;
        let sum = |v: &[DevBuf]| v.iter().map(|d| d.bytes() as u64).sum::<u64>();
        let rows: u64 = b(&self.ids) + self.rowidx.iter().map(|r| b(&r.slot) + b(&r.pos)).sum::<u64>();
        vec![
            ("mla_latent", b(&self.latent.lat)),
            ("dsa_rings", b(&self.planes.ring)),
            ("dsa_pools", b(&self.planes.pool)),
            ("mtp_taps", b(&self.store)),
            ("mtp_tap", b(&self.tap)),
            ("mtp_x", b(&self.x)),
            ("mtp_cat", b(&self.cat)),
            ("mtp_h_in", b(&self.h_in)),
            ("mtp_out", b(&self.out)),
            ("mtp_lm_in", b(&self.lm_in)),
            ("mtp_rows", rows),
            (
                "mtp_drafts",
                sum(&self.draft_ids) + sum(&self.draft_vals) + sum(&self.cand_ids) + sum(&self.cand_vals) + b(&self.keyed),
            ),
        ]
    }

    /// Slot `slot`'s bookkeeping (a test's and the backend's view).
    pub fn book(&self, slot: usize) -> &TapBook {
        &self.book[slot]
    }

    // the prefix cache's view (`snapshot.rs`): the layer's own cache, planes and tap store, and a slot's bookkeeping
    pub(super) fn latent(&self) -> &LatentCache {
        &self.latent
    }
    pub(super) fn latent_mut(&mut self) -> &mut LatentCache {
        &mut self.latent
    }
    pub(super) fn planes(&self) -> &dsa::Planes {
        &self.planes
    }
    pub(super) fn planes_mut(&mut self) -> &mut dsa::Planes {
        &mut self.planes
    }
    pub(super) fn store(&self) -> &DevBuf {
        &self.store
    }
    pub(super) fn store_mut(&mut self) -> &mut DevBuf {
        &mut self.store
    }
    pub(super) fn book_mut(&mut self, slot: usize) -> &mut TapBook {
        &mut self.book[slot]
    }
}

/// Copy `rows` rows of `h` 16-bit elements from `src` row `src_row` to `dst` row `dst_row` (asynchronous, raw bits).
fn copy_h(
    src: &DevBuf,
    src_row: usize,
    dst: &DevBuf,
    dst_row: usize,
    rows: usize,
    h: usize,
    st: &Stream,
) -> Result<(), ModelError> {
    if rows == 0 {
        return Ok(());
    }
    if (src_row + rows) * h * 2 > src.bytes() || (dst_row + rows) * h * 2 > dst.bytes() {
        return Err(ModelError::Request("MTP row copy out of range".into()));
    }
    // SAFETY: both extents were checked above and the copy is asynchronous on `st`
    unsafe { glue::copy_rows(src.ptr_at(src_row * h * 2), h, dst.ptr_at(dst_row * h * 2), h, rows, h, st)? };
    Ok(())
}

/// Where a step's token ids and hidden rows come from.
#[derive(Clone, Copy)]
enum IdsFrom {
    /// The host-uploaded ids (an absorb or a debug forward).
    Host,
    /// The argmax ids of chain depth `k` (device resident: the chain never touches the host).
    Draft(usize),
}

#[derive(Clone, Copy)]
enum HidFrom {
    HIn,
    LmIn,
    Out,
}

/// One sequence's ask of a draft call.
pub struct DraftAsk<'a> {
    pub slot: usize,
    /// The sequence so far, ending with the one token not yet fed to the trunk.
    pub tokens: &'a [u32],
    /// Drafts wanted (0..=`MTP_MAX_DEPTH`; 0 only absorbs the pending rows, to make room in the slot's store, and drafts nothing).
    pub depth: usize,
}

/// The head's candidates at one chain depth (token ids, logits), by value descending and id ascending (`GlmModel::mtp_debug_chain`).
pub type HeadList = (Vec<u32>, Vec<f32>);

/// What a debug chain feeds to its next step: the token chosen from `(depth, ids, values)` of the depth just run.
pub type ChainPick<'a> = &'a mut dyn FnMut(usize, &[u32], &[f32]) -> u32;

/// One sequence's drafts and the drafter's own confidence in each (the logit value of the pick: the argmax, or the keyed draw's token).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Drafts {
    pub ids: Vec<u32>,
    pub vals: Vec<f32>,
}

/// The keyed pick's record for one sequence of `len` tokens (its depth-0 draft is compared at position `len`), and the number of candidates the call must
/// fetch for it (0: the row is not keyed and takes the head's argmax). A request is keyed when it is sampled (temperature above 0) and its `top_k` is
/// one the call can fetch (`1..=MAX_CANDIDATES`: the server caps `top_k` at 56); a greedy request has no noise.
fn keyed_record(s: Option<&Sampling>, len: usize, mutant: Option<MtpMutant>) -> (glm_mtp::KeyedRow, usize) {
    let off = glm_mtp::KeyedRow { first_position: len as u64, ..Default::default() };
    let Some(s) = s else { return (off, 0) };
    let greedy_noise = mutant == Some(MtpMutant::KeyedGreedyNoise);
    let temperature = if s.temperature == 0.0 && greedy_noise { 1.0 } else { s.temperature };
    if temperature <= 0.0 || !(1..=MAX_CANDIDATES).contains(&s.top_k) {
        return (off, 0);
    }
    let rec = glm_mtp::KeyedRow {
        seed: s.seed,
        first_position: len as u64 + u64::from(mutant == Some(MtpMutant::KeyedPositionOff)),
        temperature,
        top_p: s.top_p,
        min_p: s.min_p,
        top_k: if mutant == Some(MtpMutant::KeyedNoTopK) { 0 } else { s.top_k as i32 },
        keyed: 1,
    };
    (rec, s.top_k)
}

impl GlmModel {
    // ---------------------------------------------------------------------------------------------------- slots, taps and commits

    /// Whether this model drafts with the pack's NextN layer.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// A new sequence in `slot` (called by `begin_slot`).
    pub(super) fn mtp_begin(&mut self, slot: usize) {
        if let Some(m) = &mut self.mtp {
            m.book[slot].reset();
        }
    }

    /// Switch the slot's drafting off for the rest of its sequence (rows pending an absorb that are an image's, which only a prompt chunk's splice can
    /// supply: `GlmBackend::spec_allowed_for_window`). Never an error: the slot just drafts nothing.
    pub fn mtp_disable_slot(&mut self, slot: usize) {
        if let Some(m) = &mut self.mtp {
            m.book[slot].disable();
        }
    }

    /// The slot's bookkeeping, when the model drafts.
    pub fn mtp_book(&self, slot: usize) -> Option<&TapBook> {
        self.mtp.as_ref().map(|m| &m.book[slot])
    }

    /// Debug: start (or, with `false`, stop and discard) timing the drafter's phases.
    pub fn set_mtp_profile(&mut self, on: bool) {
        if let Some(m) = &mut self.mtp {
            m.profile = on.then(Default::default);
        }
    }

    /// Debug: the milliseconds by phase since profiling was switched on (and clear them).
    pub fn take_mtp_profile(&mut self) -> Vec<(&'static str, f64)> {
        match self.mtp.as_mut().and_then(|m| m.profile.as_mut()) {
            Some(p) => std::mem::take(p).into_iter().collect(),
            None => Vec::new(),
        }
    }

    /// Debug: set a deliberate drafter bug (tests only).
    pub fn set_mtp_mutant(&mut self, m: Option<MtpMutant>) {
        if let Some(s) = &mut self.mtp {
            s.mutant = m;
        }
    }

    /// The last window's committed rows join their slots' stores (called by the commit flush, before the next forward overwrites the tap
    /// buffer). `keep[i]` is the number of rows of sequence `i` of `win` that were accepted.
    pub(super) fn mtp_commit_taps(&mut self, win: &RowBatch, keep: &[i32]) -> Result<(), ModelError> {
        let Some(m) = self.mtp.as_mut() else { return Ok(()) };
        let h = self.cfg.hidden;
        for (i, &slot) in win.seq_slot.iter().enumerate() {
            let slot = slot as usize;
            let rows = win.seq_rows(i);
            let kept = if m.mutant == Some(MtpMutant::AdvanceAllRows) { rows.len() } else { keep[i] as usize };
            if kept == 0 {
                continue;
            }
            match m.book[slot].window_commit(win.positions[rows.start] as usize, kept, MTP_TAP_CAP) {
                Ok(dst) => copy_h(&m.tap, rows.start, &m.store, slot * MTP_TAP_CAP + dst, kept, h, &self.stream)?,
                // a sequence that never asks for drafts (a serial request) just fills its store: not worth a line
                Err(BookError::Off | BookError::Overflow) => {}
                Err(e) => {
                    eprintln!("rexl3 level=warn event=mtp_slot_off slot={slot} why={e:?} (no MTP drafts for this sequence)")
                }
            }
        }
        Ok(())
    }

    /// After a prompt forward ran through the trunk (its taps are in the tap buffer): build the MTP entries of `slot`'s positions
    /// `start .. start + tokens.len()`, whose taps are the tap buffer's rows from `row0`, from the carry and the taps, and keep the last of
    /// them as the new carry.
    pub(super) fn mtp_prefill_rows(&mut self, slot: usize, start: usize, tokens: &[u32], row0: usize) -> Result<(), ModelError> {
        let n = tokens.len();
        let Some(m) = self.mtp.as_mut() else { return Ok(()) };
        let h = self.cfg.hidden;
        let carry = match m.book[slot].prefill_chunk(start, n) {
            Ok(c) => c,
            Err(BookError::Off) => return Ok(()),
            Err(e) => {
                eprintln!("rexl3 level=warn event=mtp_slot_off slot={slot} why={e:?} (no MTP drafts for this sequence)");
                return Ok(());
            }
        };
        // the chunk is one block call over all its rows: above MAX_ROWS rows the layer runs the large-row kernels and cuts its attention into slabs
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        m.ids.upload(&ids)?;
        let pos: Vec<i32> = (start..start + n).map(|p| p as i32).collect();
        m.rowidx[0].upload(&vec![slot as i32; n], &pos)?;
        let st = &self.stream;
        // the carry is the last tap of the previous chunk, in the store's row 0: read it before the chunk's own last tap replaces it
        if carry {
            copy_h(&m.store, slot * MTP_TAP_CAP, &m.h_in, 0, 1, h, st)?;
        } else {
            m.h_in.memset_range_async(0, h * 2, 0, st)?;
        }
        if m.mutant == Some(MtpMutant::NoShift) {
            // the dropped shift: every row reads the tap of its own position
            copy_h(&m.tap, row0, &m.h_in, 0, n, h, st)?;
        } else {
            copy_h(&m.tap, row0, &m.h_in, 1, n - 1, h, st)?;
        }
        // the new carry: this chunk's last tap
        copy_h(&m.tap, row0 + n - 1, &m.store, slot * MTP_TAP_CAP, 1, h, st)?;
        self.ensure_positions(&[(slot, start + n - 1)])?;
        // an image in this chunk (the trunk's splice, still set: this runs inside the chunk's call): its rows enter the drafter's cache from the
        // features as well. The oracle's drafter does the same (`indexed_embeddings` of its prefill); a placeholder's own embedding would be a
        // different row than the one the drafter's cache holds there in the fork
        let splice = if self.mtp.as_ref().is_some_and(|m| m.mutant == Some(MtpMutant::PlaceholderImages)) {
            Vec::new()
        } else {
            self.splice.clone()
        };
        if let Some(m) = self.mtp.as_mut() {
            m.img = splice;
        }
        let r = self.mtp_block(n, 0, IdsFrom::Host, HidFrom::HIn, start + n - 1, None, true);
        if let Some(m) = self.mtp.as_mut() {
            m.img.clear();
        }
        r?;
        // no head: only the cache entries are wanted. The call stays asynchronous; the next forward on this stream orders after it.
        Ok(())
    }

    /// Make sure every `(slot, last position)` has pages and upload the block table (the trunk's `ensure_pages` for positions it has not
    /// run yet: a draft writes MTP entries ahead of the window that will verify them).
    fn ensure_positions(&mut self, last: &[(usize, usize)]) -> Result<(), ModelError> {
        for &(slot, p) in last {
            if p >= self.max_context {
                return Err(ModelError::Request(format!("position {p} is past the model's context of {}", self.max_context)));
            }
            self.kv_tab.ensure(slot, p + 1).map_err(|e| ModelError::Request(format!("{e:?}")))?;
        }
        self.upload_table()?;
        Ok(())
    }

    // ------------------------------------------------------------------------------------------------------------- the layer

    /// One forward of the NextN layer over `rows` rows (`step` picks the row index set uploaded for it; `max_pos` is the largest position among
    /// them). Leaves the block's output in `out`; the caller runs the head. With `cap`, the last `tail` rows of each stage are copied to the
    /// host (a debug path: it synchronises). `prompt`: the rows are a prompt chunk's (the drafter's prefill), so under the P2 contract the
    /// layer's dense linears and shared expert take the P2 GEMM; an absorb of committed window rows and every draft step are P1 under both.
    #[allow(clippy::too_many_arguments)]
    fn mtp_block(
        &mut self,
        rows: usize,
        step: usize,
        ids_from: IdsFrom,
        hid_from: HidFrom,
        max_pos: usize,
        mut cap: Option<(&mut MtpCapture, usize)>,
        prompt: bool,
    ) -> Result<(), ModelError> {
        // a draft call through its graphs (`mtp_draft_group` sets `run_shape` as the marker)
        let shaped = self.run_shape.is_some();
        let GlmModel {
            cfg,
            dev,
            stream,
            s,
            sig,
            mtp,
            table,
            max_context,
            snaps,
            snap_cuts,
            planes: trunk_planes,
            own_s,
            tp,
            prefill,
            ..
        } = self;
        let st = &*stream;
        let n_trunk_rings = trunk_planes.len();
        let m = mtp.as_mut().ok_or_else(|| ModelError::Config("the model has no NextN layer".into()))?;
        let w = dev.mtp.as_ref().ok_or_else(|| ModelError::Config("the weights have no NextN layer".into()))?;
        let MtpState { ids, h_in, lm_in, out, x, cat, rowidx, draft_ids, latent, planes, mutant, profile, img, .. } = m;
        let _ = mutant;
        let h = cfg.hidden;
        let timer = if profile.is_some() { Some(moe::EventTimer::new(5)?) } else { None };
        let mark = |i: usize| -> Result<(), ModelError> {
            if let Some(t) = &timer {
                t.record(i, st)?;
            }
            Ok(())
        };
        mark(0)?;
        let tail_of = |c: &Option<(&mut MtpCapture, usize)>| c.as_ref().map(|(_, t)| (*t).min(rows));

        // the input: [enorm(embed(token)) | hnorm(hidden)] -> eh_proj into the fp32 residual
        {
            let idb: &DevBuf = match ids_from {
                IdsFrom::Host => ids,
                IdsFrom::Draft(k) => &draft_ids[k],
            };
            let hid: &DevBuf = match hid_from {
                HidFrom::HIn => h_in,
                HidFrom::LmIn => lm_in,
                HidFrom::Out => out,
            };
            glm_mtp::mtp_input(&dev.embed, idb, hid, &w.enorm, &w.hnorm, cat, rows, h, cfg.vocab, RMS_EPS, st)?;
            // the rows of an image in the prompt: the embedding half comes from the vision features (the fork's `indexed_embeddings`), the hidden
            // half (the trunk's tap, which carries the image) is as it was
            for op in img.iter() {
                glm_mtp::mtp_embed_f32(&op.feats.buf, op.src_row, &w.enorm, cat, op.dst_row, op.n, h, RMS_EPS, st)?;
            }
        }
        // the prompt chunk of a prefill that takes snapshots inside itself runs the large-row kernels whatever its rows
        let big = rows > MAX_ROWS || (step == 0 && !snap_cuts.is_empty() && s.pf.is_some());
        // under the P2 contract a prompt chunk's rows take the P2 GEMM at every width (the drafter's prefill); draft steps never
        let p2 = prefill.p2_prefill(prompt);
        let path = layers::LinPath::of(p2, big);
        layers::lin(&w.eh_proj, cat, x, rows, Dtype::F16, Dtype::F32, &mut s.ws, st, path)?;
        mark(1)?;
        if let Some(t) = tail_of(&cap) {
            st.sync()?;
            let v: Vec<f32> = x.to_vec()?;
            cap.as_mut().unwrap().0.x_in = v[(rows - t) * h..rows * h].to_vec();
        }

        // attention: norm, then MLA with the layer's own latent cache and indexer planes
        glm_mtp::rmsnorm_f32_f16(x, &w.input_norm, &mut s.hidden, rows, h, RMS_EPS, st)?;
        let a = &*w.mixer;
        // a draft call through its graphs takes the windows' context-independent constants for this block's regime (`graphs::RunShape`: the
        // dense attention's chunk grid at 2,048, the indexer's grid at the scratch's capacity), as a window graph does: the extra chunks and
        // blocks do nothing, so the bits are the eager call's
        let shape = shaped.then(|| super::graphs::RunShape::for_window(dsa::is_sparse(max_pos), s.dsa.s_stride, *max_context));
        let ctx = layers::MlaCtx {
            st,
            cfg,
            table,
            max_context: *max_context,
            n_pools_max: shape.map_or((max_pos + 1) / dsa::P, |sh| sh.n_pools_max),
            any_sparse: shape.map_or(dsa::is_sparse(max_pos), |sh| sh.any_sparse),
            mla_ctx: shape.map_or(max_pos + 1, |sh| sh.mla_ctx),
            big,
            lin: path,
        };
        // the ring tail of each snapshot taken inside the prompt chunk this call builds the drafter's entries for (the NextN layer's own indexer)
        let cuts_here: &[super::snapshot::InChunkCut] = if step == 0 && big { snap_cuts.as_slice() } else { &[] };
        let mut ring_hook = |a: usize, n: usize, pl: &dsa::Planes| -> Result<(), ModelError> {
            for c in cuts_here.iter().filter(|c| a < c.row && c.row <= a + n) {
                if let Some(sp) = snaps.as_mut() {
                    sp.save_ring(c.snap, n_trunk_rings, pl, c.slot, c.at, st)?;
                }
            }
            Ok(())
        };
        let hook: Option<layers::AfterAppend<'_>> = if cuts_here.is_empty() { None } else { Some(&mut ring_hook) };
        layers::mla_dsa_mixer(&ctx, a, latent, planes, s, Some(&rowidx[step]), rows, max_pos, None, None, None, hook, &mut |_| Ok(()))?;
        mark(2)?;

        // x += attention output, norm for the feed-forward block (one pass), the shared expert, the router and the routed experts
        glm_mtp::add_rmsnorm_f32(x, &s.mixer_out, &w.post_norm, &mut s.hidden, rows, h, RMS_EPS, st)?;
        // under P2 the shared expert of a prompt chunk on the P2 GEMM, first (as in every arm below); the arms then skip theirs
        if p2 {
            let big_sc = s.pf.as_mut().map(|p| &mut p.shared);
            layers::gated(&w.shared, &s.hidden, rows, &mut s.moe_shared_out, s.shared.as_mut(), big_sc, st, path)?;
        }
        // under P2 the routed experts of a prompt chunk take the P2 expert GEMMs at every width (Lane P step 3): the router (P1's kernels)
        // writes its picks into the prefill scratch whatever the rows, then `glm_p2_moe::run` (the two-group combine, the shared expert last)
        if p2 {
            // one bank: the whole bank on one node, a rank's (whole and masked, or its share as one subbank) on a tensor-parallel rank;
            // a mixed-width layer's subbanks run one pairs stage each (`glm_p2_moe::run_subbanks`, a rank's: `model/tp/p2.rs`)
            let router = match &w.moe {
                MtpMoe::Single(wm) => &wm.router,
                MtpMoe::Subbanks(wm) => &wm.router,
            };
            let pf =
                s.pf.as_mut()
                    .ok_or_else(|| ModelError::Config("P2 needs the prefill scratch (a chunk above 128 rows)".into()))?;
            router.logits(&s.hidden, Dtype::F16, rows, &mut pf.moe.logits, st)?;
            router.select(sig, &pf.moe.logits, rows, &mut pf.moe.pick, &mut pf.moe.wts, st)?;
            let shared = Some(&s.moe_shared_out);
            match (&w.moe, tp.as_mut()) {
                (MtpMoe::Single(wm), None) => {
                    gb_cuda::glm_p2_moe::run(&wm.experts, &s.hidden, Dtype::F16, rows, shared, &mut s.mixer_out, &mut pf.moe, st)?
                }
                (MtpMoe::Subbanks(wm), None) => {
                    let os = own_s.as_mut().ok_or_else(|| ModelError::Config("no subbank scratch".into()))?;
                    gb_cuda::glm_p2_moe::run_subbanks(
                        &wm.experts,
                        &s.hidden,
                        Dtype::F16,
                        rows,
                        shared,
                        &mut s.mixer_out,
                        &mut pf.moe,
                        os,
                        &mut s.moe,
                        st,
                    )?
                }
                // the rank path (`model/tp/p2.rs`): its pairs and part, the swap, the parts' sum (the shared expert ran above)
                (moe, Some(t)) => {
                    let ex = match moe {
                        MtpMoe::Single(wm) => super::tp::P2Routed::Bank(&wm.experts),
                        MtpMoe::Subbanks(wm) => super::tp::P2Routed::Subbanks(&wm.experts),
                    };
                    t.p2_begin(ex, &s.hidden, rows, &mut pf.moe, &mut s.moe, None, st, &mut |_| Ok(()))?;
                    t.p2_finish(&ex.dims(), &pf.moe, shared, &mut s.mixer_out, None, st)?;
                }
            }
        } else {
            match &w.moe {
                // a rank of a tensor-parallel group (`docs/GLM_TP.md` 9): this rank's pairs of the layer's experts (a share bound as subbanks, or
                // the whole bank masked to the share), ONE exchange of the pair buffer, the single-node combine with the shared expert last.
                // Exactly the trunk's rank path (`model/tp.rs`); one collective per forward of this layer, with a count that is the call's rows
                MtpMoe::Single(wm) if tp.is_some() => {
                    let t = tp.as_mut().unwrap();
                    if big {
                        let pf = s.pf.as_mut().ok_or_else(|| ModelError::Config("no prefill scratch".into()))?;
                        if !p2 {
                            w.shared.forward_big(&s.hidden, Dtype::F16, rows, &mut s.moe_shared_out, &mut pf.shared, st)?;
                        }
                        wm.router.logits(&s.hidden, Dtype::F16, rows, &mut pf.moe.logits, st)?;
                        wm.router.select(sig, &pf.moe.logits, rows, &mut pf.moe.pick, &mut pf.moe.wts, st)?;
                        let sh = Some(&s.moe_shared_out);
                        t.experts_big(wm, &s.hidden, rows, &mut pf.moe, sh, &mut s.mixer_out, st)?;
                    } else {
                        let sc = s.shared.as_mut().ok_or_else(|| ModelError::Config("no shared-expert scratch".into()))?;
                        if !p2 {
                            w.shared.forward(&s.hidden, Dtype::F16, rows, &mut s.moe_shared_out, sc, st)?;
                        }
                        wm.route(&s.hidden, Dtype::F16, rows, sig, &mut s.moe, st)?;
                        let sh = Some(&s.moe_shared_out);
                        // the NextN layer runs eagerly on the model's stream (its shared expert on it too): nothing to join
                        t.experts(wm, &s.hidden, rows, &mut s.moe, sh, &mut s.mixer_out, st, &mut || Ok(()))?;
                    }
                }
                MtpMoe::Subbanks(wm) if tp.is_some() => {
                    let t = tp.as_mut().unwrap();
                    if big {
                        let pf = s.pf.as_mut().ok_or_else(|| ModelError::Config("no prefill scratch".into()))?;
                        if !p2 {
                            w.shared.forward_big(&s.hidden, Dtype::F16, rows, &mut s.moe_shared_out, &mut pf.shared, st)?;
                        }
                        wm.router.logits(&s.hidden, Dtype::F16, rows, &mut pf.moe.logits, st)?;
                        wm.router.select(sig, &pf.moe.logits, rows, &mut pf.moe.pick, &mut pf.moe.wts, st)?;
                        let sh = Some(&s.moe_shared_out);
                        t.subbanks_big(wm, &s.hidden, rows, &mut pf.moe, &mut s.moe, sh, &mut s.mixer_out, st)?;
                    } else {
                        let sc = s.shared.as_mut().ok_or_else(|| ModelError::Config("no shared-expert scratch".into()))?;
                        if !p2 {
                            w.shared.forward(&s.hidden, Dtype::F16, rows, &mut s.moe_shared_out, sc, st)?;
                        }
                        wm.route(&s.hidden, Dtype::F16, rows, sig, &mut s.moe, st)?;
                        let sh = Some(&s.moe_shared_out);
                        t.subbanks(wm, &s.hidden, rows, &mut s.moe, sh, &mut s.mixer_out, st, &mut || Ok(()))?;
                    }
                }
                MtpMoe::Single(wm) if big => {
                    let pf = s.pf.as_mut().ok_or_else(|| ModelError::Config("no prefill scratch".into()))?;
                    if !p2 {
                        w.shared.forward_big(&s.hidden, Dtype::F16, rows, &mut s.moe_shared_out, &mut pf.shared, st)?;
                    }
                    wm.router.logits(&s.hidden, Dtype::F16, rows, &mut pf.moe.logits, st)?;
                    wm.router.select(sig, &pf.moe.logits, rows, &mut pf.moe.pick, &mut pf.moe.wts, st)?;
                    wm.experts.run_prefill(
                        &s.hidden,
                        Dtype::F16,
                        rows,
                        Some(&s.moe_shared_out),
                        &mut s.mixer_out,
                        &mut pf.moe,
                        st,
                    )?;
                }
                MtpMoe::Single(wm) => {
                    let sc = s.shared.as_mut().ok_or_else(|| ModelError::Config("no shared-expert scratch".into()))?;
                    if !p2 {
                        w.shared.forward(&s.hidden, Dtype::F16, rows, &mut s.moe_shared_out, sc, st)?;
                    }
                    wm.route(&s.hidden, Dtype::F16, rows, sig, &mut s.moe, st)?;
                    wm.experts_picked(&s.hidden, Dtype::F16, rows, Some(&s.moe_shared_out), &mut s.mixer_out, &mut s.moe, st)?;
                }
                // a mixed-width NextN layer (every expert, one node): the subbanks' pairs, then the combine with the shared expert last
                MtpMoe::Subbanks(wm) => {
                    if !wm.experts.ownership().is_all() {
                        return Err(ModelError::Config(format!(
                        "{MTP_SHARE}: the NextN layer holds {} of its {} routed experts (a tensor-parallel rank's share); its pairs would \
                         need the exchange",
                        wm.experts.ownership().local(),
                        wm.dims.experts
                    )));
                    }
                    let os = own_s.as_mut().ok_or_else(|| ModelError::Config("no subbank scratch".into()))?;
                    let d = wm.dims;
                    if big {
                        let pf = s.pf.as_mut().ok_or_else(|| ModelError::Config("no prefill scratch".into()))?;
                        if !p2 {
                            w.shared.forward_big(&s.hidden, Dtype::F16, rows, &mut s.moe_shared_out, &mut pf.shared, st)?;
                        }
                        wm.router.logits(&s.hidden, Dtype::F16, rows, &mut pf.moe.logits, st)?;
                        wm.router.select(sig, &pf.moe.logits, rows, &mut pf.moe.pick, &mut pf.moe.wts, st)?;
                        wm.experts.run_prefill_pairs_windowed(&s.hidden, Dtype::F16, rows, &mut pf.moe, &mut s.moe, os, st)?;
                        let sh = Some(&s.moe_shared_out);
                        moe::combine_pairs(&pf.moe.y_pairs, &pf.moe.pick, &pf.moe.wts, sh, &mut s.mixer_out, rows, &d, st)?;
                    } else {
                        let sc = s.shared.as_mut().ok_or_else(|| ModelError::Config("no shared-expert scratch".into()))?;
                        if !p2 {
                            w.shared.forward(&s.hidden, Dtype::F16, rows, &mut s.moe_shared_out, sc, st)?;
                        }
                        wm.route(&s.hidden, Dtype::F16, rows, sig, &mut s.moe, st)?;
                        wm.experts.run_pairs(&s.hidden, Dtype::F16, rows, &mut s.moe, os, st)?;
                        let sh = Some(&s.moe_shared_out);
                        moe::combine_pairs(&s.moe.y_pairs, &s.moe.pick, &s.moe.wts, sh, &mut s.mixer_out, rows, &d, st)?;
                    }
                }
            }
        }
        glm_mtp::add_f32(x, &s.mixer_out, rows * h, st)?;
        mark(3)?;
        if let Some(t) = tail_of(&cap) {
            st.sync()?;
            let v: Vec<f32> = x.to_vec()?;
            cap.as_mut().unwrap().0.block = v[(rows - t) * h..rows * h].to_vec();
        }
        glm_mtp::rmsnorm_f32_f16(x, &w.shared_head_norm, out, rows, h, RMS_EPS, st)?;
        mark(4)?;
        if let (Some(t), Some(p)) = (&timer, profile.as_mut()) {
            for (name, a, b) in [
                ("mtp input fusion + eh_proj", 0, 1),
                ("mtp mla + dsa indexer", 1, 2),
                ("mtp moe (shared, router, experts) + residual", 2, 3),
                ("mtp final norm", 3, 4),
            ] {
                *p.entry(name).or_insert(0.0) += t.ms(a, b)? as f64;
            }
        }
        if let Some(t) = tail_of(&cap) {
            st.sync()?;
            let v: Vec<u16> = out.to_vec()?;
            cap.as_mut().unwrap().0.norm = v[(rows - t) * h..rows * h].to_vec();
        }
        Ok(())
    }

    /// The head over `n` rows of `src` (`lm_in` or `out`) into chain depth `k`'s `draft_ids` / `draft_vals`: the top-1 (the argmax, `keyed` None: the
    /// launches of every call before keyed drafting), or the TARGET's keyed draw over the head's top `count` candidates (`keyed` Some(count): the
    /// top-`count` into the depth's candidate lists, a restricted head's columns mapped to token ids, then `glm_mtp::keyed_pick` with the per-row records of
    /// `MtpState::keyed`, drawing the noise of position `first_position + k`: the position the verify window will sample this draft at).
    fn mtp_head(
        &mut self,
        n: usize,
        from_lm_in: bool,
        k: usize,
        restricted: bool,
        keyed: Option<usize>,
    ) -> Result<(), ModelError> {
        let GlmModel { cfg, dev, stream, s, mtp, .. } = self;
        let st = &*stream;
        let m = mtp.as_mut().ok_or_else(|| ModelError::Config("the model has no NextN layer".into()))?;
        let m_mutant = m.mutant;
        let MtpState { sub, lm_in, out, draft_ids, draft_vals, cand_ids, cand_vals, keyed: records, profile, .. } = m;
        let src: &DevBuf = if from_lm_in { lm_in } else { out };
        let timer = if profile.is_some() { Some(moe::EventTimer::new(2)?) } else { None };
        if let Some(t) = &timer {
            t.record(0, st)?;
        }
        match (sub.as_mut().filter(|_| restricted), keyed) {
            (Some(sh), None) => {
                // the head over the set's columns, the top-1 among them, and the column's token id
                sh.w.forward(src, &mut sh.logits, n, Dtype::F16, Dtype::F16, &mut sh.ws, st)?;
                glm_glue::topk_f16(
                    &sh.logits,
                    n,
                    sh.cols,
                    sh.valid,
                    1,
                    &mut draft_ids[k],
                    &mut draft_vals[k],
                    &mut s.topk_ws,
                    st,
                )?;
                if m_mutant != Some(MtpMutant::RestrictedNoMap) {
                    gb_cuda::glm_perf::map_ids(&mut draft_ids[k], &sh.map, n, sh.cols, st)?;
                }
            }
            (Some(sh), Some(count)) => {
                sh.w.forward(src, &mut sh.logits, n, Dtype::F16, Dtype::F16, &mut sh.ws, st)?;
                glm_glue::topk_f16(
                    &sh.logits,
                    n,
                    sh.cols,
                    sh.valid,
                    count,
                    &mut cand_ids[k],
                    &mut cand_vals[k],
                    &mut s.topk_ws,
                    st,
                )?;
                // the noise is keyed by TOKEN ID: the columns become ids BEFORE the pick (the kept columns are bit-identical to the full head's and the
                // columns are in id order, so the list is the full head's list restricted to the set)
                let by_column = m_mutant == Some(MtpMutant::KeyedNoiseByColumn);
                if m_mutant != Some(MtpMutant::RestrictedNoMap) && !by_column {
                    gb_cuda::glm_perf::map_ids(&mut cand_ids[k], &sh.map, n * count, sh.cols, st)?;
                }
                glm_mtp::keyed_pick(
                    &cand_ids[k],
                    &cand_vals[k],
                    count,
                    n,
                    records,
                    k,
                    &mut draft_ids[k],
                    &mut draft_vals[k],
                    st,
                )?;
                if by_column {
                    gb_cuda::glm_perf::map_ids(&mut draft_ids[k], &sh.map, n, sh.cols, st)?;
                }
            }
            (None, None) => {
                dev.lm_head.forward(src, &mut s.logits, n, Dtype::F16, Dtype::F16, &mut s.ws, st)?;
                glm_glue::topk_f16(
                    &s.logits,
                    n,
                    cfg.vocab,
                    TOKENIZER_VOCAB,
                    1,
                    &mut draft_ids[k],
                    &mut draft_vals[k],
                    &mut s.topk_ws,
                    st,
                )?;
            }
            (None, Some(count)) => {
                dev.lm_head.forward(src, &mut s.logits, n, Dtype::F16, Dtype::F16, &mut s.ws, st)?;
                glm_glue::topk_f16(
                    &s.logits,
                    n,
                    cfg.vocab,
                    TOKENIZER_VOCAB,
                    count,
                    &mut cand_ids[k],
                    &mut cand_vals[k],
                    &mut s.topk_ws,
                    st,
                )?;
                glm_mtp::keyed_pick(
                    &cand_ids[k],
                    &cand_vals[k],
                    count,
                    n,
                    records,
                    k,
                    &mut draft_ids[k],
                    &mut draft_vals[k],
                    st,
                )?;
            }
        }
        if let (Some(t), Some(p)) = (&timer, profile.as_mut()) {
            t.record(1, st)?;
            *p.entry("mtp head (lm_head + top-1)").or_insert(0.0) += t.ms(0, 1)? as f64;
        }
        Ok(())
    }

    /// Restrict the drafter's head to the trunk `lm_head`'s columns of these 128-column strips (`ranges`: half-open, ascending, disjoint, in strips of
    /// 128 token ids; at least 64 strips). The kept columns are bit-identical to the full head's, so a draft is the full head's whenever the full head's
    /// choice lies inside the set. The copy goes through the host (a boot-time cost); returns the device bytes added. `set_mtp_vocab` turns it on.
    pub fn mtp_restrict_vocab(&mut self, ranges: &[(usize, usize)]) -> Result<usize, ModelError> {
        let w = self.dev.lm_head.select_strips(ranges)?;
        let cols = w.n;
        let strips_full = self.cfg.vocab / 128;
        let mut strip_in = vec![false; strips_full];
        let mut map = Vec::with_capacity(cols);
        for &(lo, hi) in ranges {
            strip_in[lo..hi].fill(true);
            map.extend((lo * 128..hi * 128).map(|id| id as i32));
        }
        // the full head's last columns are padding ids (never offered); they are the set's last columns only when the last strip is in it
        let pad = self.cfg.vocab - TOKENIZER_VOCAB;
        let valid = if ranges.last().is_some_and(|&(_, hi)| hi == strips_full) { cols - pad } else { cols };
        let ws = DevBuf::zeroed(gb_cuda::exl3_linear::linear_workspace_bytes(MAX_ROWS, self.cfg.hidden, cols, w.bits))?;
        let logits = DevBuf::zeroed(MAX_ROWS * cols * 2)?;
        let map = DevBuf::from_slice(&map)?;
        let bytes = w.weight_bytes() + ws.bytes() + logits.bytes() + map.bytes();
        let m = self.mtp.as_mut().ok_or_else(|| ModelError::Config("the model has no NextN layer".into()))?;
        m.sub = Some(SubHead { w, map, logits, ws, cols, valid, strip_in });
        Ok(bytes)
    }

    /// Which vocabulary the drafter's head uses (`Full` until [`GlmModel::mtp_restrict_vocab`] has built a set).
    pub fn set_mtp_vocab(&mut self, mode: VocabMode) {
        if let Some(m) = self.mtp.as_mut() {
            m.vocab = if m.sub.is_some() { mode } else { VocabMode::Full };
        }
    }

    /// The restricted head, when built.
    pub fn mtp_sub_head(&self) -> Option<&SubHead> {
        self.mtp.as_ref().and_then(|m| m.sub.as_ref())
    }

    // ----------------------------------------------------------------------------------------------------------------- drafting

    /// Whether this call's draft parts go through their graphs (`graphs::ENV_DRAFT`; unset: on for a rank of a tensor-parallel group,
    /// off on one node): only with window graphs on, and nothing that adds host work or launches to a call (debug captures, forced
    /// routing, profiling, the drafter's mutants), and on a rank only once it has run a collective eagerly (as the windows).
    pub(super) fn draft_graphs_eligible(&self) -> bool {
        super::graphs::draft_graphs_setting().unwrap_or(self.tp.is_some())
            && self.graphs.on
            && self.capture.is_none()
            && self.forced.is_none()
            && self.profile.is_none()
            && self.mutant.is_none()
            && self.splice.is_empty()
            && !self.oracle_rounding
            && self.mtp.as_ref().is_some_and(|m| m.mutant.is_none() && m.profile.is_none() && m.img.is_empty())
            && self.tp.as_ref().is_none_or(|t| t.graphs_ready())
    }

    /// Draft for several sequences at once: absorb every pending tapped row (one MTP window over all asking sequences), then chain up to
    /// the deepest ask. `out[i]` is `None` for an ask whose slot cannot draft (switched off, nothing pending, no room); the others get
    /// `depth` ids. One synchronisation, at the end.
    pub fn mtp_draft(&mut self, asks: &[DraftAsk]) -> Result<Vec<Option<Drafts>>, ModelError> {
        self.mtp_draft_keyed(asks, &[])
    }

    /// [`GlmModel::mtp_draft`] where the drafts of the asks that carry a sampler are the TARGET's keyed draw over the head's candidates (`keyed[i]` is
    /// ask `i`'s request sampling; an empty slice, a `None`, a greedy request or a `top_k` the call cannot fetch (0, or past `MAX_CANDIDATES`) keeps
    /// the head's argmax for that ask, exactly as `mtp_draft` has always done). A draft is a proposal either way: the verify window decides every token.
    ///
    /// Draft `k` (0-based) of a sequence of `len` tokens draws the noise of `(seed, len + k, token id)` over the head's top `top_k` candidates after
    /// temperature, `top_p` and `min_p`: what the engine's sampler draws at the position the window verifies draft `k` at, over the target's distribution.
    /// A call with no keyed ask runs the very launches `mtp_draft` ran before keyed drafting existed.
    pub fn mtp_draft_keyed(&mut self, asks: &[DraftAsk], keyed: &[Option<Sampling>]) -> Result<Vec<Option<Drafts>>, ModelError> {
        let mut result: Vec<Option<Drafts>> = vec![None; asks.len()];
        if self.mtp.is_none() || asks.is_empty() {
            return Ok(result);
        }
        // the previous window's commits: the KDA flips and the tapped rows joining their stores (idempotent when nothing is pending)
        self.flush()?;
        let h = self.cfg.hidden;
        // which asks can run, and how many rows each absorbs
        let mut live: Vec<usize> = Vec::new();
        for (i, a) in asks.iter().enumerate() {
            let m = self.mtp.as_mut().unwrap();
            // the chain writes the positions `len ..= len + depth - 2`: all of them inside the model's context
            if a.depth > MTP_MAX_DEPTH || a.tokens.is_empty() || a.tokens.len() + a.depth.max(1) - 1 > self.max_context {
                continue;
            }
            match m.book[a.slot].absorb_check(a.tokens.len()) {
                Ok(()) => live.push(i),
                Err(BookError::Off | BookError::Nothing) => {}
                Err(e) => {
                    eprintln!("rexl3 level=warn event=mtp_slot_off slot={} why={e:?} (no MTP drafts for this sequence)", a.slot)
                }
            }
        }
        // the head of this call: the restricted vocabulary, or the full one (see `VocabMode`)
        let restricted = match self.mtp.as_ref().map(|m| (m.vocab, m.sub.as_ref())) {
            Some((VocabMode::Restricted, Some(_))) => true,
            Some((VocabMode::Adaptive(min), Some(sh))) => live.iter().all(|&i| {
                let t = asks[i].tokens;
                sh.coverage(&t[t.len().saturating_sub(32)..]) >= min
            }),
            _ => false,
        };
        if let Some(m) = self.mtp.as_mut() {
            m.vocab_active = restricted;
        }
        // an absorb call holds at most MAX_ROWS rows: split the sequences into calls
        let mut at = 0;
        while at < live.len() {
            let mut rows = 0usize;
            let mut end = at;
            while end < live.len() {
                let n = self.mtp.as_ref().unwrap().book[asks[live[end]].slot].n;
                if end > at && rows + n > MAX_ROWS {
                    break;
                }
                rows += n;
                end += 1;
            }
            self.mtp_draft_group(asks, keyed, &live[at..end], &mut result, h)?;
            at = end;
        }
        Ok(result)
    }

    fn mtp_draft_group(
        &mut self,
        asks: &[DraftAsk],
        keyed: &[Option<Sampling>],
        group: &[usize],
        result: &mut [Option<Drafts>],
        h: usize,
    ) -> Result<(), ModelError> {
        let s_n = group.len();
        // the chain's deepest row of the call is the deepest ask's, but no row may sit past the model's context: near its end every stream in
        // the call drafts a little less
        let room = group.iter().map(|&i| self.max_context + 1 - asks[i].tokens.len()).min().unwrap_or(0);
        let depth = group.iter().map(|&i| asks[i].depth).max().unwrap_or(0).min(room);
        // host arrays: the absorb's rows, then each chain step's
        let (mut ids, mut slots, mut pos) = (Vec::new(), Vec::new(), Vec::new());
        let mut ends = Vec::with_capacity(s_n); // the row index of each sequence's last absorbed row
        let mut last_pos = Vec::with_capacity(s_n);
        // the first pending row each sequence absorbs (all of them; the mutant drops all but the last)
        let from = |m: &MtpState, slot: usize| if m.mutant == Some(MtpMutant::AbsorbLastOnly) { m.book[slot].n - 1 } else { 0 };
        {
            let m = self.mtp.as_ref().unwrap();
            for &i in group {
                let a = &asks[i];
                let b = &m.book[a.slot];
                for j in from(m, a.slot)..b.n {
                    ids.push(a.tokens[b.first + j] as i32);
                    slots.push(a.slot as i32);
                    pos.push((b.first + j) as i32);
                }
                ends.push(ids.len() - 1);
                // the absorb's last row sits at position n - 1; chain row `k` (1-based) of a sequence of n tokens at position n + k - 1
                last_pos.push((a.slot, a.tokens.len() - 1 + depth.saturating_sub(1)));
            }
        }
        let rows = ids.len();
        if rows == 0 || rows > MAX_ROWS {
            return Ok(());
        }
        // pages for every position the call writes (`mtp_draft` clipped each ask's depth to the model's context)
        self.ensure_positions(&last_pos)?;
        // the keyed asks' sampling records, and how many candidates the call fetches (None: no row is keyed, today's launches)
        let (records, pick) = {
            let mutant = self.mtp.as_ref().and_then(|m| m.mutant);
            let (mut recs, mut count) = (Vec::with_capacity(s_n), 0usize);
            for &i in group {
                let (r, k) = keyed_record(keyed.get(i).and_then(|s| s.as_ref()), asks[i].tokens.len(), mutant);
                recs.push(r);
                count = count.max(k);
            }
            (recs, (count > 0).then_some(count))
        };
        // the call's parts through their graphs (`draft_graphs_eligible`): the absorb keyed by its rows and regime, the head and chain by the
        // asking sequences, the depth, the head's vocabulary, the pick and each chained step's regime
        let graphed = self.draft_graphs_eligible();
        let max_pos = *pos.iter().max().unwrap() as usize;
        let chain_pos = |k: usize| group.iter().map(|&i| asks[i].tokens.len() + k - 1).max().unwrap();
        let restricted_call = self.mtp.as_ref().is_some_and(|m| m.vocab_active);
        let absorb_key = super::graphs::DraftKey {
            chain: false,
            rows,
            depth: 0,
            restricted: false,
            pick: 0,
            sparse: u32::from(dsa::is_sparse(max_pos)),
        };
        let chain_key = super::graphs::DraftKey {
            chain: true,
            rows: s_n,
            depth,
            restricted: restricted_call,
            pick: pick.unwrap_or(0),
            sparse: (1..depth).fold(0u32, |b, k| b | (u32::from(dsa::is_sparse(chain_pos(k))) << (k - 1))),
        };
        // the draft graph mutant: a replayed chain is not given this call's chained row index sets
        let stale_chain = graphed
            && depth > 1
            && self.graphs.mutant == Some(super::graphs::GraphMutant::DraftStaleRows)
            && self.draft_would_replay(&chain_key);
        {
            let m = self.mtp.as_mut().unwrap();
            if pick.is_some() {
                m.keyed.upload(&records)?;
            }
            m.ids.upload(&ids)?;
            m.rowidx[0].upload(&slots, &pos)?;
            let n_s: Vec<i32> = group.iter().map(|&i| asks[i].slot as i32).collect();
            for k in 1..depth {
                let p: Vec<i32> = group.iter().map(|&i| (asks[i].tokens.len() + k - 1) as i32).collect();
                if !stale_chain {
                    m.rowidx[k].upload(&n_s, &p)?;
                }
            }
            // the absorb's hidden rows, from each sequence's store, in row order
            let st = &self.stream;
            let mut row = 0usize;
            for &i in group {
                let a = &asks[i];
                let (n, f) = (m.book[a.slot].n, from(m, a.slot));
                copy_h(&m.store, a.slot * MTP_TAP_CAP + f, &m.h_in, row, n - f, h, st)?;
                row += n - f;
            }
        }
        if graphed {
            // the marker `mtp_block` reads: the windows' context-independent launch constants (graphs::RunShape) for every block of the call
            self.run_shape = Some(super::graphs::RunShape::for_window(false, self.s.dsa.s_stride, self.max_context));
        }
        let r = self.mtp_draft_parts(graphed, absorb_key, chain_key, rows, s_n, depth, &ends, &chain_pos, max_pos, pick, h);
        self.run_shape = None;
        r?;
        self.stream.sync()?;
        // the drafts, from the device
        let m = self.mtp.as_mut().unwrap();
        let mut ids_k: Vec<Vec<i32>> = Vec::with_capacity(depth);
        let mut vals_k: Vec<Vec<f32>> = Vec::with_capacity(depth);
        for k in 0..depth {
            let mut a = vec![0i32; s_n];
            let mut v = vec![0f32; s_n];
            m.draft_ids[k].download(&mut a)?;
            m.draft_vals[k].download(&mut v)?;
            ids_k.push(a);
            vals_k.push(v);
        }
        let wrong = m.mutant == Some(MtpMutant::WrongDrafts);
        let id_of = |k: usize, j: usize| {
            let id = ids_k[k][j] as u32;
            if wrong {
                (id + 1) % TOKENIZER_VOCAB as u32
            } else {
                id
            }
        };
        for (j, &i) in group.iter().enumerate() {
            let want = asks[i].depth.min(depth);
            let d = Drafts { ids: (0..want).map(|k| id_of(k, j)).collect(), vals: (0..want).map(|k| vals_k[k][j]).collect() };
            result[i] = Some(d);
            m.book[asks[i].slot].absorbed(asks[i].tokens.len());
        }
        Ok(())
    }

    /// The launches of a draft call after its uploads: the absorb (one forward of the layer over every absorbed row), the gather of each
    /// sequence's last row into `lm_in` (host-indexed copies: eager, between the two graphed parts), then the head and the chain (`depth - 1`
    /// further layer rows and heads). With `graphed`, the absorb and the head-and-chain run through `run_draft` under their keys; the launches
    /// are the same either way.
    #[allow(clippy::too_many_arguments)]
    fn mtp_draft_parts(
        &mut self,
        graphed: bool,
        absorb_key: super::graphs::DraftKey,
        chain_key: super::graphs::DraftKey,
        rows: usize,
        s_n: usize,
        depth: usize,
        ends: &[usize],
        chain_pos: &dyn Fn(usize) -> usize,
        max_pos: usize,
        pick: Option<usize>,
        h: usize,
    ) -> Result<(), ModelError> {
        let mut absorb = |m: &mut GlmModel| m.mtp_block(rows, 0, IdsFrom::Host, HidFrom::HIn, max_pos, None, false);
        if graphed {
            self.run_draft(absorb_key, &mut absorb)?;
        } else {
            absorb(self)?;
        }
        // the head needs only each sequence's last row: gather them to the front of `lm_in` (an absorb-only call has no head)
        if depth == 0 {
            return Ok(());
        }
        {
            let m = self.mtp.as_mut().unwrap();
            let st = &self.stream;
            for (j, &e) in ends.iter().enumerate() {
                copy_h(&m.out, e, &m.lm_in, j, 1, h, st)?;
            }
        }
        let restricted = self.mtp.as_ref().is_some_and(|m| m.vocab_active);
        let mut chain = |m: &mut GlmModel| -> Result<(), ModelError> {
            m.mtp_head(s_n, true, 0, restricted, pick)?;
            for k in 1..depth {
                m.mtp_block(
                    s_n,
                    k,
                    IdsFrom::Draft(k - 1),
                    if k == 1 { HidFrom::LmIn } else { HidFrom::Out },
                    chain_pos(k),
                    None,
                    false,
                )?;
                m.mtp_head(s_n, false, k, restricted, pick)?;
            }
            Ok(())
        };
        if graphed {
            self.run_draft(chain_key, &mut chain)
        } else {
            chain(self)
        }
    }

    // ----------------------------------------------------------------------------------------------------------------- debug

    /// Debug (the offline study and the tests of keyed drafting): the head's top-`count` candidates (ids as token ids, values, by value descending and id
    /// ascending) at each of `depth` chain depths for ONE sequence, with the chain fed whatever `next(depth, ids, vals)` returns (the host's pick, the
    /// true continuation, ...). The absorb and the chain write the drafter's cache exactly as `mtp_draft` does, but the slot's pending rows are NOT marked
    /// absorbed unless `absorb` is set (a sequence that goes on committing must absorb, or its store of pending rows overflows after a few rounds): without
    /// it this can be called again, or followed by a real `mtp_draft`, which then rewrites the very same entries. `None` when the slot cannot draft.
    /// Synchronises at every depth.
    #[allow(clippy::too_many_arguments)]
    pub fn mtp_debug_chain(
        &mut self,
        slot: usize,
        tokens: &[u32],
        depth: usize,
        count: usize,
        restricted: bool,
        absorb: bool,
        next: ChainPick,
    ) -> Result<Option<Vec<HeadList>>, ModelError> {
        if self.mtp.is_none()
            || depth == 0
            || depth > MTP_MAX_DEPTH
            || count == 0
            || count > MAX_CANDIDATES
            || tokens.is_empty()
            || tokens.len() + depth - 1 > self.max_context
        {
            return Ok(None);
        }
        self.flush()?;
        let h = self.cfg.hidden;
        let (first, n) = {
            let m = self.mtp.as_mut().unwrap();
            match m.book[slot].absorb_check(tokens.len()) {
                Ok(()) => (m.book[slot].first, m.book[slot].n),
                Err(BookError::Off | BookError::Nothing) => return Ok(None),
                Err(e) => return Err(ModelError::Request(format!("the slot's draft state: {e:?}"))),
            }
        };
        self.ensure_positions(&[(slot, tokens.len() - 1 + depth - 1)])?;
        {
            let m = self.mtp.as_mut().unwrap();
            let ids: Vec<i32> = (0..n).map(|j| tokens[first + j] as i32).collect();
            m.ids.upload(&ids)?;
            let pos: Vec<i32> = (first..first + n).map(|p| p as i32).collect();
            m.rowidx[0].upload(&vec![slot as i32; n], &pos)?;
            for k in 1..depth {
                m.rowidx[k].upload(&[slot as i32], &[(tokens.len() + k - 1) as i32])?;
            }
            copy_h(&m.store, slot * MTP_TAP_CAP, &m.h_in, 0, n, h, &self.stream)?;
        }
        self.mtp_block(n, 0, IdsFrom::Host, HidFrom::HIn, first + n - 1, None, false)?;
        {
            let m = self.mtp.as_mut().unwrap();
            copy_h(&m.out, n - 1, &m.lm_in, 0, 1, h, &self.stream)?;
        }
        let restricted = restricted && self.mtp.as_ref().is_some_and(|m| m.sub.is_some());
        let mut out = Vec::with_capacity(depth);
        for k in 0..depth {
            if k > 0 {
                self.mtp_block(
                    1,
                    k,
                    IdsFrom::Host,
                    if k == 1 { HidFrom::LmIn } else { HidFrom::Out },
                    tokens.len() + k - 1,
                    None,
                    false,
                )?;
            }
            self.mtp_head(1, k == 0, k, restricted, Some(count))?;
            self.stream.sync()?;
            let m = self.mtp.as_mut().unwrap();
            let (mut ids, mut vals) = (vec![0i32; count], vec![0f32; count]);
            m.cand_ids[k].download(&mut ids)?;
            m.cand_vals[k].download(&mut vals)?;
            let ids: Vec<u32> = ids.into_iter().map(|i| i as u32).collect();
            if k + 1 < depth {
                let t = next(k, &ids, &vals);
                m.ids.upload(&[t as i32])?;
            } else {
                next(k, &ids, &vals);
            }
            out.push((ids, vals));
        }
        if absorb {
            self.mtp.as_mut().unwrap().book[slot].absorbed(tokens.len());
        }
        Ok(Some(out))
    }

    /// The tap buffer's first `rows` rows (fp16 bits `[rows][hidden]`): the last forward's final-normed hidden rows, as the drafter will read them (a debug
    /// path: it synchronises).
    pub fn mtp_tap_rows(&mut self, rows: usize) -> Result<Vec<u16>, ModelError> {
        self.stream.sync()?;
        let m = self.mtp.as_ref().ok_or_else(|| ModelError::Config("the model has no NextN layer".into()))?;
        let all: Vec<u16> = m.tap.to_vec()?;
        Ok(all[..rows * self.cfg.hidden].to_vec())
    }

    /// The NextN layer's latent rows `0..n` for `slot` (fp16 bits, `[n][512]`, dequantised from an fp8 cache), through the slot's page table
    /// (a debug path).
    pub fn mtp_latent_rows(&mut self, slot: usize, n: usize) -> Result<Vec<u16>, ModelError> {
        self.stream.sync()?;
        let m = self.mtp.as_ref().ok_or_else(|| ModelError::Config("the model has no NextN layer".into()))?;
        Ok(m.latent.read_rows(self.kv_tab.blocks_of(slot), n)?)
    }

    /// The NextN layer's DSA pool rows of pools `0..n_pools` for `slot` (fp16 bits, `[n_pools][128]`).
    pub fn mtp_pool_rows(&mut self, slot: usize, n_pools: usize) -> Result<Vec<u16>, ModelError> {
        self.stream.sync()?;
        let m = self.mtp.as_ref().ok_or_else(|| ModelError::Config("the model has no NextN layer".into()))?;
        let all: Vec<u16> = m.planes.pool.to_vec()?;
        let blocks = self.kv_tab.blocks_of(slot);
        let per = dsa::POOLS_PER_PAGE;
        let mut out = Vec::with_capacity(n_pools * dsa::D);
        for p in 0..n_pools {
            let at = (blocks[p / per] as usize * per + p % per) * dsa::D;
            out.extend_from_slice(&all[at..at + dsa::D]);
        }
        Ok(out)
    }

    /// Debug (the oracle comparison): run MTP rows at positions `start .. start + tokens.len()` of `slot` with the HOST-supplied tapped hidden
    /// rows (`hidden`: fp16 bits `[rows][hidden]`, row `i` the tap that pairs with `tokens[i]`), in calls of at most 128 rows, and capture the last
    /// `tail` rows of every stage, with the head's logits over them. Synchronises. Uses the production layer code and the slot's cache.
    pub fn mtp_debug_forward(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        hidden: &[u16],
        tail: usize,
    ) -> Result<MtpCapture, ModelError> {
        self.mtp_debug_forward_mm(slot, start, tokens, hidden, tail, &[])
    }

    /// [`GlmModel::mtp_debug_forward`] over a prompt with images: `images` are `(features, first position, rows)` in positions of the sequence
    /// (`start` being the position of `tokens[0]`), the rows from row 0 of each image's features; the MTP rows at those positions take the
    /// features as their embedding, as the prefill's chunks do.
    pub fn mtp_debug_forward_mm(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        hidden: &[u16],
        tail: usize,
        images: &[(std::sync::Arc<GlmImageFeatures>, usize, usize)],
    ) -> Result<MtpCapture, ModelError> {
        let h = self.cfg.hidden;
        let n = tokens.len();
        if self.mtp.is_none() || n == 0 || hidden.len() != n * h {
            return Err(ModelError::Request(
                "mtp_debug_forward: a model with the NextN layer, tokens and one hidden row per token".into(),
            ));
        }
        self.flush()?;
        self.stream.sync()?;
        let mut cap = MtpCapture::default();
        let mut at = 0;
        while at < n {
            // the captured tail comes from the last call: keep at least `tail` rows for it (the layer is row-invariant, so the cut changes no bit)
            let left = n - at;
            let end = if left > MAX_ROWS && left - MAX_ROWS < tail { n - tail } else { (at + MAX_ROWS).min(n) };
            let rows = end - at;
            self.ensure_positions(&[(slot, start + end - 1)])?;
            {
                let m = self.mtp.as_mut().unwrap();
                let ids: Vec<i32> = tokens[at..end].iter().map(|&t| t as i32).collect();
                m.ids.upload(&ids)?;
                let pos: Vec<i32> = (start + at..start + end).map(|p| p as i32).collect();
                m.rowidx[0].upload(&vec![slot as i32; rows], &pos)?;
                m.h_in.upload(&hidden[at * h..end * h])?;
                // the image rows of this call, chunk-relative
                m.img = images
                    .iter()
                    .filter_map(|(f, first, len)| {
                        let (lo, hi) = ((*first).max(start + at), (first + len).min(start + end));
                        (lo < hi).then(|| SpliceOp {
                            feats: f.clone(),
                            dst_row: lo - (start + at),
                            src_row: lo - first,
                            n: hi - lo,
                        })
                    })
                    .collect();
            }
            let last = end == n;
            let t = if last { tail.min(rows) } else { 0 };
            let r = self.mtp_block(rows, 0, IdsFrom::Host, HidFrom::HIn, start + end - 1, (t > 0).then_some((&mut cap, t)), true);
            self.mtp.as_mut().unwrap().img.clear();
            r?;
            if last && t > 0 {
                let m = self.mtp.as_mut().unwrap();
                copy_h(&m.out, rows - t, &m.lm_in, 0, t, h, &self.stream)?;
                self.mtp_head(t, true, 0, false, None)?;
                self.stream.sync()?;
                let v = self.cfg.vocab;
                let all: Vec<u16> = self.s.logits.to_vec()?;
                cap.logits = all[..t * v].to_vec();
            }
            at = end;
        }
        self.stream.sync()?;
        Ok(cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_keyed_record_says_where_the_draft_is_verified_and_what_the_sampler_keeps() {
        let s = Sampling { seed: 9, temperature: 0.7, top_k: 20, top_p: 0.95, min_p: 0.05 };
        // the first draft of a sequence of 300 tokens is compared with the sample of position 300 (the window's row 0 samples at `len`)
        let (r, fetch) = keyed_record(Some(&s), 300, None);
        assert_eq!((r.keyed, r.seed, r.first_position, r.top_k, fetch), (1, 9, 300, 20, 20));
        assert_eq!((r.temperature, r.top_p, r.min_p), (0.7, 0.95, 0.05));
        // not keyed: no sampler, a greedy request (no noise), a top_k the call cannot fetch (0 is "every candidate", past MAX_CANDIDATES is out of reach)
        for other in [
            None,
            Some(Sampling { temperature: 0.0, ..s }),
            Some(Sampling { top_k: 0, ..s }),
            Some(Sampling { top_k: MAX_CANDIDATES + 1, ..s }),
        ] {
            let (r, fetch) = keyed_record(other.as_ref(), 300, None);
            assert_eq!((r.keyed, fetch, r.first_position), (0, 0, 300), "{other:?}");
        }
        // the largest top_k the server allows (56) and the largest the call can fetch (64) are keyed
        assert_eq!(keyed_record(Some(&Sampling { top_k: 56, ..s }), 10, None).1, 56);
        assert_eq!(keyed_record(Some(&Sampling { top_k: MAX_CANDIDATES, ..s }), 10, None).1, MAX_CANDIDATES);
        // the deliberate bugs of the GPU mutation check each move exactly one thing (production sets none)
        assert_eq!(keyed_record(Some(&s), 300, Some(MtpMutant::KeyedPositionOff)).0.first_position, 301);
        assert_eq!(keyed_record(Some(&s), 300, Some(MtpMutant::KeyedNoTopK)).0.top_k, 0);
        let greedy = Sampling { temperature: 0.0, ..s };
        assert_eq!(keyed_record(Some(&greedy), 300, Some(MtpMutant::KeyedGreedyNoise)).0.keyed, 1);
    }

    #[test]
    fn the_default_draft_vocabulary_is_a_prefix_the_numbers_and_the_special_tokens() {
        assert_eq!(vocab_ranges(512, 154_880), vec![(0, 512), (768, 810), (1209, 1210)]);
        // a prefix that reaches the numeric block swallows it, one that reaches the end swallows the specials
        assert_eq!(vocab_ranges(800, 154_880), vec![(0, 810), (1209, 1210)]);
        assert_eq!(vocab_ranges(1210, 154_880), vec![(0, 1210)]);
        // ascending, disjoint and at least the 64 strips the kernel's plan needs
        for p in [64, 192, 384, 512, 767, 768, 900, 1209] {
            let r = vocab_ranges(p, 154_880);
            assert!(r.windows(2).all(|w| w[0].1 < w[1].0), "{p}: {r:?}");
            assert!(r.iter().map(|&(lo, hi)| hi - lo).sum::<usize>() >= 64, "{p}");
            assert_eq!(r.last().unwrap().1, 1210, "the last strip (the specials) is always in: {p}");
        }
    }

    const CAP: usize = MTP_TAP_CAP;

    #[test]
    fn a_prompt_leaves_the_last_tap_as_the_carry() {
        let mut b = TapBook::new();
        // 300 tokens in chunks of 128: the first chunk starts from zeros, the next two from the carry
        assert_eq!(b.prefill_chunk(0, 128), Ok(false));
        assert_eq!((b.first, b.n), (128, 1));
        assert_eq!(b.prefill_chunk(128, 128), Ok(true));
        assert_eq!(b.prefill_chunk(256, 44), Ok(true));
        assert_eq!((b.first, b.n), (300, 1));
        // the first sampled token is the 301st: the entry for position 300 is the one pending
        assert!(b.can_absorb(301) && !b.can_absorb(300) && !b.can_absorb(302), "exactly the sequence length the store matches");
        assert_eq!(b.absorb_check(301), Ok(()));
        b.absorbed(301);
        assert_eq!((b.first, b.n), (301, 0));
    }

    #[test]
    fn only_the_kept_rows_of_a_window_join_the_store_and_the_next_absorb_needs_exactly_them() {
        let mut b = TapBook::new();
        b.prefill_chunk(0, 20).unwrap();
        b.absorb_check(21).unwrap();
        b.absorbed(21);
        // a window of 4 rows (the pending token at position 20 and three drafts) with 2 kept: positions 20 and 21 were committed, and with
        // them two new tokens (an accepted draft and a fresh sample) make the sequence 23 tokens long
        assert_eq!(b.window_commit(20, 2, CAP), Ok(0));
        assert_eq!((b.first, b.n), (21, 2));
        assert_eq!(b.absorb_check(23), Ok(()), "entries 21 and 22 need the taps of positions 20 and 21");
        // a store with the rejected rows in it no longer matches the tokens: the slot switches itself off instead of drafting from the
        // wrong hidden rows
        let mut c = TapBook::new();
        c.prefill_chunk(0, 20).unwrap();
        c.absorbed(21);
        c.window_commit(20, 4, CAP).unwrap();
        assert_eq!(c.absorb_check(23), Err(BookError::Mismatch));
        assert!(!c.ok);
        assert_eq!(c.absorb_check(23), Err(BookError::Off));
    }

    #[test]
    fn rows_accumulate_until_a_draft_call_absorbs_them_and_a_gap_or_overflow_switches_the_slot_off() {
        let mut b = TapBook::new();
        b.prefill_chunk(0, 10).unwrap();
        // no draft call between three windows: the rows pile up in order behind the carry
        assert_eq!(b.window_commit(10, 1, CAP), Ok(1));
        assert_eq!(b.window_commit(11, 3, CAP), Ok(2));
        assert_eq!(b.window_commit(14, 2, CAP), Ok(5));
        assert_eq!((b.first, b.n), (10, 7));
        assert_eq!(b.absorb_check(17), Ok(()));
        // a window that does not continue where the rows end is a gap
        let mut g = b.clone();
        assert_eq!(g.window_commit(17, 1, CAP), Err(BookError::Gap), "the next row is at position 16");
        assert!(!g.ok);
        // more rows than the store holds
        let mut o = TapBook::new();
        o.prefill_chunk(0, 10).unwrap();
        assert_eq!(o.window_commit(10, CAP, CAP), Err(BookError::Overflow));
        assert!(!o.ok);
        // a switched-off slot stays off, for windows and prefill chunks, until it is reset
        assert_eq!(o.window_commit(10, 1, CAP), Err(BookError::Off));
        assert_eq!(o.prefill_chunk(10, 5), Err(BookError::Off));
        o.reset();
        assert_eq!(o.prefill_chunk(0, 5), Ok(false));
    }

    #[test]
    fn a_prefill_that_does_not_continue_is_refused_and_nothing_pending_means_no_draft() {
        let mut b = TapBook::new();
        assert_eq!(b.prefill_chunk(5, 5), Err(BookError::Gap), "a chunk that is not the first, on a fresh slot");
        let mut b = TapBook::new();
        b.prefill_chunk(0, 8).unwrap();
        assert_eq!(b.prefill_chunk(9, 4), Err(BookError::Gap), "a chunk that skips a row");
        let mut b = TapBook::new();
        b.prefill_chunk(0, 8).unwrap();
        b.absorbed(9);
        assert_eq!(b.absorb_check(9), Err(BookError::Nothing));
        // an image in the prompt: the slot is switched off before or during its prefill and drafts nothing
        let mut b = TapBook::new();
        b.disable();
        assert_eq!(b.prefill_chunk(0, 8), Err(BookError::Off));
        assert_eq!(b.absorb_check(9), Err(BookError::Off));
    }
}
