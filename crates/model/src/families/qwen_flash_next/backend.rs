//! The engine's text-only Flash-Next backend. The model owns all device state; this layer
//! chooses prompt-wide arithmetic, slices on an absolute grid and translates errors.
use super::model::FlashNextModel;
use super::prefix::{self, SliceRequest, SliceResult, SnapshotArena};
use crate::error::ModelError;
use crate::forward::{Candidates, PrefillPath};
use crate::rows::RowBatch;
use gb_engine::prefix::{PrefixCaps, PrefixHandle};
use gb_engine::{Backend, BackendError, CutStep, ImageSpan, PrefillStep, Proposal, ProposeRequest, RowOutput, RowRequest};
use std::collections::BTreeMap;

/// Narrow adapter so the host scheduling contract can run without a GPU or a model pack.
pub trait FlashNextRuntime {
    fn vocab_size(&self) -> usize;
    fn has_mtp(&self) -> bool;
    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError>;
    fn end_slot(&mut self, slot: usize);
    fn prefill_chunk(&mut self, b: &RowBatch, path: PrefillPath, want: Option<usize>) -> Result<Option<Candidates>, ModelError>;
    fn mtp_after_prefill(
        &mut self,
        slot: usize,
        first: usize,
        chunk: &[u32],
        next: Option<u32>,
        path: PrefillPath,
    ) -> Result<(), ModelError>;
    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError>;
    /// Like [`FlashNextRuntime::window_candidates`], but with the device-side greedy gather: the same
    /// graph runs, and the returned candidates (when `Some`) carry one candidate per row — the greedy
    /// argmax. `None` means the runtime does not support the gather: the caller falls back to the full
    /// `window_candidates` readback. A runtime that supports the gather always returns `Some` — there
    /// is no switch.
    fn window_candidates_dv(&mut self, _b: &RowBatch, _count: usize) -> Result<Option<Candidates>, ModelError> {
        Ok(None)
    }
    /// The next prefill chunk is `prompt[first..end]`: its PLE rows may be gathered now, off the critical path
    /// (changes when table bytes are read, never which). A runtime without a PLE stage ignores it.
    fn ple_prefetch_next(&mut self, _slot: usize, _prompt: &[u32], _first: usize, _end: usize) {}
    fn commit(&mut self, slot: usize, keep: usize);
    fn flush(&mut self) -> Result<(), ModelError>;
    /// Up to `k` drafts with the head's probability of each (see `FlashNextModel::mtp_propose_scored`).
    fn mtp_propose(&mut self, slot: usize, tokens: &[u32], k: usize) -> Result<(Vec<u32>, Vec<f32>), ModelError>;
    fn cached_slice(&mut self, _arena: &mut SnapshotArena, _req: SliceRequest<'_>) -> Result<SliceResult, ModelError> {
        Err(ModelError::Config("prefix snapshots need the device model".into()))
    }
    fn restore_snapshot(
        &mut self,
        _arena: &SnapshotArena,
        _slot: usize,
        _handle: &PrefixHandle,
        _len: usize,
    ) -> Result<(), ModelError> {
        Err(ModelError::Config("prefix snapshots need the device model".into()))
    }
    fn release_snapshot(&mut self, _arena: &mut SnapshotArena, _handle: PrefixHandle) -> Result<(), ModelError> {
        Err(ModelError::Config("prefix snapshots need the device model".into()))
    }
}

impl FlashNextRuntime for FlashNextModel {
    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
    fn has_mtp(&self) -> bool {
        self.has_mtp()
    }
    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        self.begin_slot(slot)
    }
    fn end_slot(&mut self, slot: usize) {
        self.end_slot(slot)
    }
    fn prefill_chunk(&mut self, b: &RowBatch, path: PrefillPath, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        self.prefill_chunk(b, path, want)
    }
    fn mtp_after_prefill(
        &mut self,
        slot: usize,
        first: usize,
        chunk: &[u32],
        next: Option<u32>,
        path: PrefillPath,
    ) -> Result<(), ModelError> {
        self.mtp_after_prefill(slot, first, chunk, next, path)
    }
    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        self.window_candidates(b, count)
    }
    fn window_candidates_dv(&mut self, b: &RowBatch, count: usize) -> Result<Option<Candidates>, ModelError> {
        self.window_candidates_dv(b, count)
    }
    fn ple_prefetch_next(&mut self, slot: usize, prompt: &[u32], first: usize, end: usize) {
        self.ple_prefetch_next(slot, prompt, first, end)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.commit(slot, keep)
    }
    fn flush(&mut self) -> Result<(), ModelError> {
        self.flush()
    }
    fn mtp_propose(&mut self, slot: usize, tokens: &[u32], k: usize) -> Result<(Vec<u32>, Vec<f32>), ModelError> {
        self.mtp_propose_scored(slot, tokens, k)
    }
    fn cached_slice(&mut self, arena: &mut SnapshotArena, req: SliceRequest<'_>) -> Result<SliceResult, ModelError> {
        prefix::slice_with_cuts(self, arena, req)
    }
    fn restore_snapshot(
        &mut self,
        arena: &SnapshotArena,
        slot: usize,
        handle: &PrefixHandle,
        len: usize,
    ) -> Result<(), ModelError> {
        prefix::restore(self, arena, slot, handle, len)
    }
    fn release_snapshot(&mut self, arena: &mut SnapshotArena, handle: PrefixHandle) -> Result<(), ModelError> {
        prefix::release(self, arena, handle)
    }
}

pub struct FlashNextBackend<M: FlashNextRuntime = FlashNextModel> {
    pub m: M,
    pub chunk: usize,
    pub mtp_k: usize,
    /// The engine's row limit for a prefill slice while a deadline is near (`Backend::set_prefill_row_limit`): the slices go on the
    /// absolute grid of `min(chunk, limit)` rows, which the chunk-invariant forward makes the same bits (`effective_chunk`).
    deadline_rows: Option<usize>,
    adaptive: bool,
    last_acc: BTreeMap<usize, usize>,
    error: Option<BackendError>,
    fatal: Option<BackendError>,
    prefix: Option<(SnapshotArena, String)>,
}

fn nan_row(n: usize) -> RowOutput {
    RowOutput { ids: (0..n as u32).collect(), values: vec![f32::NAN; n] }
}

fn row_of(c: &Candidates, r: usize, n: usize) -> RowOutput {
    RowOutput {
        ids: c.ids[r * c.count..r * c.count + n].iter().map(|&x| x as u32).collect(),
        values: c.vals[r * c.count..r * c.count + n].to_vec(),
    }
}

/// A resumed slice ends at the next absolute chunk boundary, not `done + chunk`.
fn next_chunk_end(done: usize, total: usize, chunk: usize) -> usize {
    done.saturating_add(chunk - done % chunk).min(total)
}

impl FlashNextBackend<FlashNextModel> {
    pub fn new(m: FlashNextModel, chunk: usize, mtp_k: usize) -> Self {
        let adaptive = std::env::var("REXL3_MTP_ADAPT").map_or(true, |v| v != "0");
        Self::with_runtime(m, chunk, mtp_k, adaptive)
    }

    pub fn with_prefix_cache(mut self, budget: gb_sched::prefix::CacheBudget, fingerprint: String) -> Result<Self, ModelError> {
        self.prefix = Some((SnapshotArena::new(&self.m, budget)?, fingerprint));
        Ok(self)
    }
}

impl<M: FlashNextRuntime> FlashNextBackend<M> {
    pub fn with_runtime(m: M, chunk: usize, mtp_k: usize, adaptive: bool) -> Self {
        assert!(chunk > 0, "prefill chunk is nonzero");
        Self { m, chunk, mtp_k, deadline_rows: None, adaptive, last_acc: BTreeMap::new(), error: None, fatal: None, prefix: None }
    }

    /// The rows of the grid the engine-driven slices are cut on: the chunk, or the engine's row limit when it is smaller. Only the
    /// slices of `prefill_slice` / `prefill_slice_cuts` take it (the whole-prompt `prefill_piece` keeps the chunk).
    fn effective_chunk(&self) -> usize {
        self.chunk.min(self.deadline_rows.unwrap_or(usize::MAX)).max(1)
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

    fn run_chunk(
        &mut self,
        slot: usize,
        first: usize,
        tokens: &[u32],
        next: Option<u32>,
        total: usize,
        want: Option<usize>,
    ) -> Result<Option<Candidates>, ModelError> {
        let b = RowBatch::from_requests(&[RowRequest { slot, first_position: first as u64, tokens: tokens.to_vec() }]);
        let path = PrefillPath::for_prompt(total);
        let c = self.m.prefill_chunk(&b, path, want)?;
        if self.mtp_k > 0 {
            self.m.mtp_after_prefill(slot, first, tokens, next, path)?;
        }
        Ok(c)
    }

    fn cached_slice_inner(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
        cuts: &[usize],
    ) -> Result<CutStep, ModelError> {
        if !images.is_empty() {
            return Err(ModelError::Request("Flash-Next vision is not built".into()));
        }
        let chunk = self.effective_chunk();
        let (arena, _) = self.prefix.as_mut().ok_or_else(|| ModelError::Config("prefix arena unavailable".into()))?;
        let result = self.m.cached_slice(arena, SliceRequest { slot, prompt, done, candidates, cuts, chunk })?;
        Ok(CutStep {
            step: PrefillStep { done: result.end, row: result.row.map(|c| row_of(&c, 0, candidates)) },
            captured: result.captured,
        })
    }

    fn prefill_slice_inner(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
    ) -> Result<PrefillStep, ModelError> {
        if !images.is_empty() {
            return Err(ModelError::Request("Flash-Next vision is not built".into()));
        }
        if done >= prompt.len() {
            return Err(ModelError::Request("empty or completed prefill slice".into()));
        }
        if done == 0 {
            self.m.begin_slot(slot)?;
        }
        self.m.flush()?;
        let end = next_chunk_end(done, prompt.len(), self.effective_chunk());
        if end < prompt.len() {
            self.m.ple_prefetch_next(slot, prompt, end, next_chunk_end(end, prompt.len(), self.effective_chunk()));
        }
        let c = self.run_chunk(
            slot,
            done,
            &prompt[done..end],
            prompt.get(end).copied(),
            prompt.len(),
            (end == prompt.len()).then_some(candidates),
        )?;
        Ok(PrefillStep { done: end, row: c.map(|c| row_of(&c, 0, candidates)) })
    }
}

impl<M: FlashNextRuntime> Backend for FlashNextBackend<M> {
    /// Row-invariant slicing: the GDN prefill is chunk-invariant (`gb_cuda::gdn::prefill`), the pinned prefill GEMM row invariant and the
    /// prefill path keyed on the prompt's total, so the engine's slices go on the absolute grid of the limit with the same bits (the
    /// off-grid cuts of the prefix cache rely on the same property).
    fn set_prefill_row_limit(&mut self, rows: Option<usize>) -> bool {
        self.deadline_rows = rows.map(|r| r.max(1));
        true
    }
    fn prefix_caps(&self) -> Option<PrefixCaps> {
        self.prefix
            .as_ref()
            .map(|(arena, fingerprint)| prefix::caps(arena, fingerprint, &gb_cuda::exl3_linear::prefill_pin_name()))
    }
    fn prefix_path(&self, total: usize) -> Option<u32> {
        self.prefix.as_ref().and_then(|_| prefix::path(total))
    }
    fn prefix_free_snapshots(&self) -> usize {
        self.prefix.as_ref().map_or(0, |(arena, _)| arena.free_count())
    }
    fn prefix_restore(&mut self, slot: usize, handle: &PrefixHandle, len: usize) -> Result<(), String> {
        let result = match self.prefix.as_ref() {
            Some((arena, _)) => self.m.restore_snapshot(arena, slot, handle, len),
            None => Err(ModelError::Config("prefix arena unavailable".into())),
        };
        if result.is_err() {
            self.m.end_slot(slot);
        }
        result.map_err(|e| e.to_string())
    }
    fn prefix_release(&mut self, handle: PrefixHandle) {
        if let Some((arena, _)) = self.prefix.as_mut() {
            if let Err(e) = self.m.release_snapshot(arena, handle) {
                self.record("prefix release", &e);
            }
        } else {
            self.record("prefix release", &ModelError::Config("prefix arena unavailable".into()));
        }
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
        if self.dead() {
            return CutStep { step: PrefillStep { done: prompt.len(), row: Some(nan_row(candidates)) }, captured: Vec::new() };
        }
        match self.cached_slice_inner(slot, prompt, images, done, candidates, cuts) {
            Ok(cut) => cut,
            Err(e) => {
                self.record(&format!("prefill slice with cuts of slot {slot}"), &e);
                if !self.dead() {
                    self.m.end_slot(slot);
                }
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
        let run = (|| -> Result<RowOutput, ModelError> {
            let first = usize::try_from(start).map_err(|_| ModelError::Request("prefill position overflows usize".into()))?;
            if tokens.is_empty() || first.checked_add(tokens.len()).is_none_or(|n| n > total) {
                return Err(ModelError::Request("empty prefill piece or piece past prompt total".into()));
            }
            if first == 0 {
                self.m.begin_slot(slot)?;
            }
            self.m.flush()?;
            let mut at = first;
            let mut last = None;
            while at < first + tokens.len() {
                let end = next_chunk_end(at, first + tokens.len(), self.chunk);
                let c = self.run_chunk(
                    slot,
                    at,
                    &tokens[at - first..end - first],
                    tokens.get(end - first).copied(),
                    total,
                    (end == first + tokens.len()).then_some(candidates),
                )?;
                last = c;
                at = end;
            }
            let c = last.ok_or_else(|| ModelError::Request("prefill produced no candidates".into()))?;
            Ok(row_of(&c, 0, candidates))
        })();
        match run {
            Ok(row) => row,
            Err(e) => {
                self.record(&format!("prefill of slot {slot}"), &e);
                if !self.dead() {
                    self.m.end_slot(slot);
                }
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
        let result = if self.prefix.is_some() {
            self.cached_slice_inner(slot, prompt, images, done, candidates, &[]).map(|cut| cut.step)
        } else {
            self.prefill_slice_inner(slot, prompt, images, done, candidates)
        };
        match result {
            Ok(step) => step,
            Err(e) => {
                self.record(&format!("prefill slice of slot {slot}"), &e);
                if !self.dead() {
                    self.m.end_slot(slot);
                }
                PrefillStep { done: prompt.len(), row: Some(nan_row(candidates)) }
            }
        }
    }

    fn prefill_mm(&mut self, slot: usize, _: u64, _: &[u32], _: &[ImageSpan], candidates: usize) -> RowOutput {
        self.record("multimodal prefill", &ModelError::Request("Flash-Next vision is not built".into()));
        if !self.dead() {
            self.m.end_slot(slot);
        }
        nan_row(candidates)
    }

    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        let nans = || reqs.iter().map(|r| r.tokens.iter().map(|_| nan_row(candidates)).collect()).collect();
        if self.dead() {
            return nans();
        }
        let run = (|| -> Result<_, ModelError> {
            // no flush: the window folds the commit the last one made into its own kernels
            let b = RowBatch::from_requests(reqs);
            let c = self.m.window_candidates(&b, candidates)?;
            Ok((0..reqs.len()).map(|i| b.seq_rows(i).map(|r| row_of(&c, r, candidates)).collect()).collect())
        })();
        match run {
            Ok(rows) => rows,
            Err(e) => {
                self.record("window step", &e);
                nans()
            }
        }
    }

    /// [`Backend::step`] with the device-side greedy gather: when every request is greedy, the window
    /// runs as usual but the candidate table is packed to its argmax column on the device
    /// (`FlashNextRuntime::window_candidates_dv`), so the round reads back `rows * 4` bytes instead of
    /// `rows * count * 8`. The gathered rows are the same candidates the full download would give a
    /// greedy sampler (column 0 of the sorted table), so the engine's accept/counters are untouched.
    /// When the runtime does not gather and for any batch with a sampled request, this is exactly
    /// [`Backend::step`].
    fn step_verified(&mut self, reqs: &[RowRequest], candidates: usize, greedy: &[bool]) -> Vec<Vec<RowOutput>> {
        let nans = || reqs.iter().map(|r| r.tokens.iter().map(|_| nan_row(candidates)).collect()).collect();
        if self.dead() {
            return nans();
        }
        let all_greedy = greedy.len() == reqs.len() && greedy.iter().all(|&g| g);
        if !all_greedy {
            return self.step(reqs, candidates);
        }
        let run = (|| -> Result<_, ModelError> {
            // no flush: the window folds the commit the last one made into its own kernels
            let b = RowBatch::from_requests(reqs);
            if let Some(c) = self.m.window_candidates_dv(&b, candidates)? {
                debug_assert_eq!(c.count, 1, "the gather returns one candidate per row");
                return Ok((0..reqs.len())
                    .map(|i| b.seq_rows(i).map(|r| RowOutput { ids: vec![c.ids[r] as u32], values: vec![c.vals[r]] }).collect())
                    .collect());
            }
            // fallback: the full readback, exactly like `step`
            let c = self.m.window_candidates(&b, candidates)?;
            Ok((0..reqs.len()).map(|i| b.seq_rows(i).map(|r| row_of(&c, r, candidates)).collect()).collect())
        })();
        match run {
            Ok(rows) => rows,
            Err(e) => {
                self.record("window step", &e);
                nans()
            }
        }
    }

    fn commit(&mut self, slot: usize, keep: usize) {
        if self.dead() {
            return;
        }
        self.last_acc.insert(slot, keep.saturating_sub(1));
        self.m.commit(slot, keep);
    }
    fn release(&mut self, slot: usize) {
        self.last_acc.remove(&slot);
        if !self.dead() {
            self.m.end_slot(slot);
        }
    }

    fn propose_batch(&mut self, reqs: &[ProposeRequest<'_>]) -> Vec<Option<Vec<u32>>> {
        self.propose_batch_scored(reqs).into_iter().map(|p| p.map(|p| p.tokens)).collect()
    }

    fn propose_batch_scored(&mut self, reqs: &[ProposeRequest<'_>]) -> Vec<Option<Proposal>> {
        if self.dead() || self.mtp_k == 0 || reqs.len() != 1 {
            return vec![None; reqs.len()];
        }
        let r = &reqs[0];
        let k = if self.adaptive {
            self.last_acc.get(&r.slot).copied().unwrap_or(2).saturating_add(2).min(self.mtp_k)
        } else {
            self.mtp_k
        };
        let k = k.min(r.max);
        if k == 0 {
            return vec![None];
        }
        let result = self.m.mtp_propose(r.slot, r.tokens, k);
        match result {
            Ok((mut tokens, mut accept)) if !tokens.is_empty() => {
                tokens.truncate(k);
                accept.truncate(tokens.len());
                // the head's own probability for each pick is what the budget's `Margin`/`Scored` policies spend rows by (the engine
                // keeps it honest per stream with its calibration factor); the score histogram (`rexl3_spec_conf_*`, bins 2.0 wide
                // from -8) reads it on the logit scale, which spreads probabilities near 1 over several bins
                let conf = accept.iter().map(|&p| logit(p)).collect();
                vec![Some(Proposal { tokens, conf, accept, ..Default::default() })]
            }
            Ok(_) => vec![None],
            Err(e) => {
                self.record(&format!("draft of slot {}", r.slot), &e);
                vec![None]
            }
        }
    }
}

/// `ln(p / (1 - p))`, with `p` held inside `1e-6..=1 - 1e-6` (and NaN taken as 0.5).
fn logit(p: f32) -> f32 {
    let p = if p.is_nan() { 0.5 } else { p.clamp(1e-6, 1.0 - 1e-6) };
    (p / (1.0 - p)).ln()
}

#[cfg(test)]
#[path = "backend_tests.rs"]
mod tests;
