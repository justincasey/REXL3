//! The DFlash2 drafter of GLM-5.3-Flash (`--draft DIR` with the incoai/GLM-5.3-Flash-DFlash2 checkpoint): the trunk's taps, the
//! draft's context ring, and a draft round.
//!
//! **Taps.** After each tapped layer `l` of every forward (a prompt chunk, a decode or verify window; one node or a tensor-parallel
//! rank; eager or replayed from a CUDA graph), the mean of the row's four mHC streams (fp32, `((s0 + s1) + (s2 + s3)) / 4`, rounded
//! to bf16) goes to column block `i` of the tap buffer `[rows][taps][hidden]`. That is the completed residual of layer `l` contracted
//! the way the drafter was trained on (SGLang's `hc_contract`, TensorFold's `df_tap`). The kernel only reads the streams, so the trunk's
//! bits are what they are without a drafter.
//!
//! **Context.** A prompt chunk's rows are absorbed right after its forward (every row: the ring keeps the last 2,560 positions of a
//! slot and attention reads the last 2,048); a window's rows are absorbed when its commit is applied (`flush`), the accepted prefix of
//! each sequence only. A rejected row's position is written again by the window that next covers it. When the backend has named the
//! prompt's end ([`GlmModel::dflash_prompt_end`]), the rows of a chunk more than a ring behind it are neither tapped nor absorbed: the
//! prompt's later rows would overwrite their ring entries before any draft reads the ring (a row's entry is its own row's alone, so the
//! ring after the prompt holds the same bits).
//!
//! **Drafting** runs the draft's block for each asking sequence at the position of its pending token and returns the selector's chain
//! with each pick's score and margin (`draft::Picked`). It reads the trunk's embedding and `lm_head` and writes only the draft's own ring
//! and scratch: the trunk's state and every committed token are untouched, so drafted == serial whatever it proposes. On a
//! tensor-parallel group it runs on rank 0 alone (the draft is dense, the head replicated): nothing is forwarded to a follower.
//!
//! **Prefix cache.** A restored prefix does not bring its draft context back: the ring keeps whatever the slot held at those
//! positions, so drafts after a restore are weaker until 2,048 new positions have been absorbed. Exactness is unaffected.

use super::GlmModel;
use crate::draft::{Absorb, Draft, DraftReq, Picked};
use crate::error::ModelError;
use crate::rows::RowBatch;
use gb_cuda::{DevBuf, Stream};

/// The drafter's device state on a model.
pub struct DflashState {
    pub draft: Draft,
    /// bf16 `[rows][taps][hidden]`: the last forward's tap rows (rows: the widest forward, a prompt chunk or a window).
    pub(super) taps: DevBuf,
    /// Per trunk layer: its column block in a tap row, or `None` when it is not tapped.
    pub(super) tap_of: Vec<Option<usize>>,
    rows: usize,
    /// Per slot: the end of the prompt being prefilled, when the backend named it (`GlmModel::dflash_prompt_end`; `None` after
    /// `begin_slot`).
    prompt_end: Vec<Option<usize>>,
    /// The prompt chunk in flight absorbs none of its rows (all of them more than a ring before its prompt's end): `tap` writes nothing.
    skip_taps: bool,
    /// Positions absorbed, prompt positions skipped (more than a ring before their prompt's end) and draft rounds run (the boot and the
    /// logs read them).
    pub absorbed: u64,
    pub rounds: u64,
}

/// One sequence's ask of a draft round.
#[derive(Clone, Copy, Debug)]
pub struct DflashAsk<'a> {
    pub slot: usize,
    /// The sequence so far, ending with the one token not yet fed to the trunk (the anchor).
    pub tokens: &'a [u32],
    pub sampling: &'a gb_sampling::Sampling,
    /// Drafts wanted, 1..=7.
    pub want: usize,
}

impl DflashState {
    /// The tap buffer for forwards of up to `rows` rows over a trunk of `layers` layers tapping `tap_layers` (ascending).
    pub fn new(draft: Draft, tap_layers: &[usize], layers: usize, rows: usize) -> Result<Self, ModelError> {
        let d = draft.dims;
        if tap_layers.len() != d.taps || tap_layers.iter().any(|&l| l >= layers) {
            return Err(ModelError::Config(format!(
                "the draft taps layers {tap_layers:?} ({} taps), the trunk has {layers} layers",
                d.taps
            )));
        }
        let mut tap_of = vec![None; layers];
        for (i, &l) in tap_layers.iter().enumerate() {
            tap_of[l] = Some(i);
        }
        Ok(Self {
            taps: DevBuf::zeroed_view(rows * d.fc_in() * 2)?,
            tap_of,
            rows,
            draft,
            prompt_end: Vec::new(),
            skip_taps: false,
            absorbed: 0,
            skipped: 0,
            rounds: 0,
        })
    }

    /// Device bytes beyond the draft's own: the tap buffer.
    pub fn tap_bytes(&self) -> usize {
        self.taps.bytes()
    }

    /// The tap buffer's bytes for these shapes (what a memory plan holds back for it).
    pub fn planned_tap_bytes(dims: &crate::draft::DraftDims, rows: usize) -> usize {
        rows * dims.fc_in() * 2
    }

    /// After a forward of `rows` rows: write layer `layer`'s stream mean into its tap column, when the layer is tapped.
    pub(super) fn tap(&mut self, streams: &DevBuf, layer: usize, rows: usize, st: &Stream) -> Result<(), ModelError> {
        let Some(i) = self.tap_of.get(layer).copied().flatten() else { return Ok(()) };
        if rows > self.rows {
            return Err(ModelError::Request(format!("a forward of {rows} rows; the draft's taps hold {}", self.rows)));
        }
        let d = self.draft.dims;
        gb_cuda::dsv4_mtp_gpu::tap_mean_bf16_dev(streams, &mut self.taps, rows, d.hidden, i, d.taps, st)?;
        Ok(())
    }
}

impl GlmModel {
    /// Draft with DFlash2 from now on: `tap_layers` are the trunk layers the draft reads (its `target_layer_ids`). The tap buffer holds
    /// the model's widest forward.
    pub fn attach_dflash(&mut self, draft: Draft, tap_layers: &[usize]) -> Result<usize, ModelError> {
        let rows = self.s.rows.max(self.window_rows);
        let st = DflashState::new(draft, tap_layers, self.layers, rows)?;
        let bytes = st.tap_bytes();
        self.dflash = Some(st);
        Ok(bytes)
    }

    /// Whether this model drafts with DFlash2.
    pub fn has_dflash(&self) -> bool {
        self.dflash.is_some()
    }

    /// The drafter (the restricted head, the diagnostics).
    pub fn dflash_mut(&mut self) -> Option<&mut DflashState> {
        self.dflash.as_mut()
    }

    /// The draft head over these token-id ranges of the trunk's `lm_head` (`(first id, end id)`, whole strips); with `adaptive`, only in
    /// rounds where every asking sequence has at least that share of its recent tokens inside them. Returns the bytes it added.
    pub fn dflash_restrict_vocab(&mut self, ranges: &[(usize, usize)], adaptive: Option<f32>) -> Result<usize, ModelError> {
        let GlmModel { dflash, dev, .. } = self;
        let d = dflash.as_mut().ok_or_else(|| ModelError::Config("this model has no DFlash2 drafter".into()))?;
        let added = d.draft.restrict_vocab(&dev.lm_head, ranges)?;
        d.draft.set_adaptive(adaptive);
        Ok(added)
    }

    /// After a prompt chunk of `slot` ran through the trunk (its taps are in the tap buffer): its rows join the slot's draft context at
    /// positions `start..start + n`, in writes of at most the ring's window (a write must span fewer positions than the ring holds; a
    /// row's ring entry is row-invariant, so a 4,096-row chunk leaves the ring a 2,048-row chunk pair leaves), all but the rows more than a
    /// ring before the prompt's end when the backend named it.
    pub(super) fn dflash_prefill_chunk(&mut self, slot: usize, start: usize, n: usize) -> Result<(), ModelError> {
        let Some(d) = self.dflash.as_mut() else { return Ok(()) };
        d.skip_taps = false;
        let first = d.keep_from(slot).saturating_sub(start).min(n);
        let mut at = first;
        while at < n {
            let k = (n - at).min(d.draft_rows()).min(gb_cuda::dflash2::DraftCtxLayout::WINDOW);
            d.draft.absorb_batch(&d.taps, &[Absorb { row0: at, n: k, slot, pos0: start + at }], &self.stream)?;
            at += k;
        }
        d.absorbed += n as u64;
        Ok(())
    }

    /// The last window's accepted rows join their slots' draft contexts (called by the commit flush, before the next forward overwrites the
    /// tap buffer). `keep[i]` is the number of rows of sequence `i` of `win` that were accepted.
    pub(super) fn dflash_commit_taps(&mut self, win: &RowBatch, keep: &[i32]) -> Result<(), ModelError> {
        let Some(d) = self.dflash.as_mut() else { return Ok(()) };
        let items: Vec<Absorb> = win
            .seq_slot
            .iter()
            .enumerate()
            .filter_map(|(i, &slot)| {
                let rows = win.seq_rows(i);
                let kept = (keep[i].max(0) as usize).min(rows.len());
                (kept > 0).then(|| Absorb { row0: rows.start, n: kept, slot: slot as usize, pos0: win.positions[rows.start] as usize })
            })
            .collect();
    pub skipped: u64,
        let n: usize = items.iter().map(|a| a.n).sum();
        d.draft.absorb_batch(&d.taps, &items, &self.stream)?;
        d.absorbed += (n - first) as u64;
        d.skipped += first as u64;
        Ok(())
    }

    /// One draft round: per ask, up to `want` drafts after its pending token, with the draft's score and margin of each. The pending commits
    /// must have been applied (`flush`): the context ring then holds every committed position. Synchronises once per batch of
    /// `Draft::max_seqs` asks.
    pub fn dflash_draft(&mut self, asks: &[DflashAsk<'_>]) -> Result<Vec<Picked>, ModelError> {
        let GlmModel { dflash, dev, cfg, stream, .. } = self;
        let d = dflash.as_mut().ok_or_else(|| ModelError::Config("this model has no DFlash2 drafter".into()))?;
        let mut out = Vec::with_capacity(asks.len());
        for group in asks.chunks(d.draft.max_seqs.max(1)) {
            let reqs: Vec<DraftReq> = group
                .iter()
                .map(|a| {
                    let p = a.tokens.len().saturating_sub(1);
                    let lo = a.tokens.len().saturating_sub(crate::draft::COVERAGE_WINDOW);
                    DraftReq {
                        anchor: *a.tokens.last().unwrap_or(&0),
                        slot: a.slot,
                        p,
                        sampling: a.sampling,
                        want: a.want.clamp(1, crate::draft::DRAFTS),
                        coverage: d.draft.coverage(&a.tokens[lo..]),
                    }
                })
                .collect();
            out.extend(d.draft.propose_batch_scored(&dev.embed, &dev.lm_head, cfg.vocab, &reqs, stream)?);
            d.rounds += 1;
        }
        Ok(out)
    }
}

impl DflashState {
    /// The most rows one absorb takes (the draft's context scratch).
    fn draft_rows(&self) -> usize {
        self.draft.max_ctx_rows()
    }
}
        if self.skip_taps {
            return Ok(());
        }

    /// The first position of `slot`'s prompt whose row reaches the ring after the prompt: `end - cap` when the backend named the prompt's
    /// end, 0 otherwise (every row is absorbed).
    fn keep_from(&self, slot: usize) -> usize {
        self.prompt_end
            .get(slot)
            .copied()
            .flatten()
            .map_or(0, |end| end.saturating_sub(gb_cuda::dflash2::DraftCtxLayout::DEFAULT_CAP))
    }
    /// The backend's word that `slot`'s prompt being prefilled ends at position `end` (`None`: unknown, every row is absorbed). Set before
    /// each prompt slice; `begin_slot` forgets it. The rows more than a ring before `end` are neither tapped nor absorbed (see the module).
    pub fn dflash_prompt_end(&mut self, slot: usize, end: Option<usize>) {
        let Some(d) = self.dflash.as_mut() else { return };
        if d.prompt_end.len() <= slot {
            d.prompt_end.resize(slot + 1, None);
        }
        d.prompt_end[slot] = end;
    }

    /// Before the forward of a prompt chunk of `slot` at `start..start + n`: whether any of its rows will be absorbed (when none is, the
    /// forward writes no taps).
    pub(super) fn dflash_begin_chunk(&mut self, slot: usize, start: usize, n: usize) {
        if let Some(d) = self.dflash.as_mut() {
            d.skip_taps = start + n <= d.keep_from(slot);
        }
    }

    /// After a prompt chunk's forward: the next forward taps every row again.
    pub(super) fn dflash_end_chunk(&mut self) {
        if let Some(d) = self.dflash.as_mut() {
            d.skip_taps = false;
        }
    }

