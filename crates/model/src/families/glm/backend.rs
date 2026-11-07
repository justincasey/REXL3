//! `gb_engine::Backend` over the GLM-5.3-Flash model (text and images, no drafting by the model itself).
//!
//! The trait's methods cannot fail, so a GPU or request error is logged, answered with NaN candidate rows (the engine's sampler refuses NaN)
//! and recorded for `Backend::take_error`, which the engine reads after every call: a recoverable error fails the sequences the call
//! served, a fatal one (a sticky CUDA error, lost recurrent state: `ModelError::is_fatal`) stops the engine and the process. Once a fatal
//! error has been seen it is latched and no further call touches the device.
//!
//! A prompt is fed in chunks (`GlmBackend::chunk` rows, 128 up to the model's chunk capacity; above 128 the large-row kernels, which give every
//! row the decode kernels' bits), one chunk per `prefill_slice` so the engine can run a decode round for the other sequences in between. Every
//! chunk is committed at once, so the bits of a prompt row do not depend on how the prompt is cut.
//!
//! **Images.** A prompt with `ImageSpan`s (the server expanded each `<|image|>` into one placeholder per merged 2x2 block of patches and encoded the
//! image into f32 features, `vision::GlmImageFeatures`) is prefilled by the same chunks: the rows of a span that fall inside a chunk are written over
//! the embeddings of the chunk's placeholder tokens right after the embedding stage (`GlmModel::prefill_chunk_mm`), in all four mHC streams, and
//! nothing else about an image row differs from a text row. The text attention is NoPE (MLA without rope, KDA's recurrence, the DSA indexer on cache
//! positions), so there is no position rule to follow, unlike Qwen's M-RoPE: an image row's position is its index. A span may straddle chunks.
//! Text-only calls never reach the splice: they launch what they always did and their bits are those of the text path.
//!
//! **Speculation over images.** The NextN head's cache is built over a prompt chunk by chunk as it is prefilled (`model/mtp.rs`), and the rows of an
//! image enter it from the image's features, as the oracle's drafter does (`indexed_embeddings` of its prefill), not from the placeholder token's
//! embedding. After the prompt the drafter works on generated tokens only. [`GlmBackend::spec_allowed_for_window`] says whether a window of rows of a slot
//! touches an image span; the drafting runtime asks it about the rows it is about to absorb from the tap store, which can only be a prompt's image
//! rows if their chunk's splice was never seen: then the slot drafts nothing rather than absorb a placeholder. An adopted prefix (the prefix cache) brings the
//! drafter's cache for its image rows with it (the pages and the carry are copied), and the image spans are set again by the first slice after a restore.
//!
//! **Prefix cache.** `with_prefix_cache` gives the model a pool of snapshot slots (`model/snapshot.rs`); `prefill_slice_cuts` takes a snapshot at each cut
//! the slice passes: inside the chunk, in the same forward, when the model has the large-row scratch (any chunk of two rows or more), or by ending the slice
//! at the cut for a model built for chunks of 128 rows or fewer (every row's bits are chunk-grid independent, so a shorter slice changes nothing);
//! `prefix_restore` puts a slot in the state of an entry. One path id for every prompt length: GLM has one arithmetic (`prefix.rs`).

use super::alloc::TOKENIZER_VOCAB;
use super::digest_log::DigestLog;
use super::model::GlmModel;
use super::prefix;
use super::spec::{SpecConfig, SpecRuntime};
use super::vision_serve::{GlmImageFeatures, SpliceOp};
use crate::error::ModelError;
use crate::forward::Candidates;
use crate::rows::RowBatch;
use gb_engine::{
    Backend, BackendError, CacheBudget, Captured, CutStep, ImageSpan, PrefillStep, PrefixCaps, PrefixHandle, ProposeRequest,
    RowOutput, RowRequest,
};
use std::ops::Range;
use std::sync::Arc;

pub struct GlmBackend {
    pub m: GlmModel,
    /// Per slot, the positions of the prompt that hold image rows (empty for a text prompt), from the start of its prefill until the slot is released.
    slot_images: Vec<Vec<Range<usize>>>,
    /// The first recoverable error since `take_error` last ran.
    error: Option<BackendError>,
    /// The first fatal error: it never clears.
    fatal: Option<BackendError>,
    /// The drafters (the NextN layer, the copy drafter) and their telemetry; `None`: the backend offers no drafts of its own.
    pub spec: Option<SpecRuntime>,
    /// Rows of one prefill slice (one chunk of the model; at most the model's `chunk_rows`).
    chunk: usize,
    deadline_rows: Option<usize>,
    /// The prefix cache this backend reserved (its namespace and budget), when one is enabled (`docs/PREFIX_CACHE.md`, "GLM-5.3-Flash").
    prefix: Option<PrefixCaps>,
    /// `REXL3_GLM_DIGEST_LOG`: the candidates handed to the engine, per row (host only; off unless set).
    digests: DigestLog,
}

/// Whether none of the `rows` positions from `first` is in one of `spans` (the image spans of a prompt): the predicate behind
/// [`GlmBackend::spec_allowed_for_window`], free of the backend so the drafting runtime can ask it with the spans it is handed.
pub fn window_is_text(spans: &[Range<usize>], first: usize, rows: usize) -> bool {
    let end = first.saturating_add(rows);
    !spans.iter().any(|s| s.start < end && first < s.end)
}

fn nan_row(n: usize) -> RowOutput {
    RowOutput { ids: (0..n as u32).collect(), values: vec![f32::NAN; n] }
}

fn row_of(c: &Candidates, r: usize, n: usize) -> RowOutput {
    let ids = c.ids[r * c.count..r * c.count + n].iter().map(|&x| x as u32).collect();
    let values = c.vals[r * c.count..r * c.count + n].to_vec();
    RowOutput { ids, values }
}

impl GlmBackend {
    /// A backend that offers no drafts (the engine's own per-sequence drafter, if any, is used).
    pub fn new(m: GlmModel) -> Self {
        let slot_images = vec![Vec::new(); m.slots];
        let chunk = m.chunk_rows();
        let digests = DigestLog::from_env(m.slots);
        Self { m, slot_images, error: None, fatal: None, spec: None, chunk, deadline_rows: None, prefix: None, digests }
    }

    /// A backend that drafts by `cfg`: its `propose_batch` returns every asking stream's drafts (an empty list for none).
    pub fn with_spec(m: GlmModel, cfg: SpecConfig) -> Self {
        let slots = m.slots;
        let slot_images = vec![Vec::new(); slots];
        let chunk = m.chunk_rows();
        let digests = DigestLog::from_env(slots);
        Self {
            m,
            slot_images,
            error: None,
            fatal: None,
            spec: Some(SpecRuntime::new(cfg, slots)),
            chunk,
            deadline_rows: None,
            prefix: None,
            digests,
        }
    }

    /// Enable the prefix cache: allocates `budget.snapshot_slots` snapshot slots in the model (unless it has them already: the engine builder does that
    /// before it checks the allocations against the plan). `namespace` binds every key to what the snapshots depend on (`prefix::namespace_of`). The KV
    /// pool must already hold `budget.page_blocks` pages more than admission hands out (the memory plan does that).
    pub fn with_prefix_cache(mut self, namespace: u64, budget: CacheBudget) -> Result<Self, ModelError> {
        if !self.m.has_prefix_cache() {
            self.m.enable_prefix_cache(budget.snapshot_slots)?;
        }
        self.prefix = Some(PrefixCaps { namespace, budget });
        Ok(self)
    }

    /// Prefill in slices of `rows` rows (clamped to `1..=` the model's chunk capacity): one chunk of the model per `prefill_slice`. Every row's bits
    /// are the same whatever the slice (the large-row kernels give each row the decode kernels' bits), so this only trades the throughput of a long
    /// prompt against how long a decoding stream waits for its next round.
    pub fn with_chunk(mut self, rows: usize) -> Self {
        self.chunk = rows.clamp(1, self.m.chunk_rows());
        self
    }

    /// Rows of one prefill slice.
    pub fn chunk(&self) -> usize {
        self.chunk
    }

    /// [`GlmBackend::with_chunk`] on a backend that is already in use (a test moves one engine between chunk sizes).
    pub fn set_chunk(&mut self, rows: usize) {
        self.chunk = rows.clamp(1, self.m.chunk_rows());
    }

    /// Whether speculation may run over the `rows` positions from `first_position` of `slot`: false when any of them holds an image row. The
    /// drafters of this family (the MTP head, once it is built) call it before they absorb or draft over positions of a prompt that has images.
    pub fn spec_allowed_for_window(&self, slot: usize, first_position: usize, rows: usize) -> bool {
        window_is_text(self.slot_images.get(slot).map_or(&[], |v| v.as_slice()), first_position, rows)
    }

    /// The images of a prompt of `n` tokens, as this backend's features, checked against the prompt before anything is touched: features it produced,
    /// one row per placeholder, the placeholder tokens at the span, the spans in order and inside the prompt.
    fn image_features(&self, prompt: &[u32], images: &[ImageSpan]) -> Result<Vec<Arc<GlmImageFeatures>>, ModelError> {
        let token = self.m.cfg.vision.as_ref().map(|v| v.image_token_id);
        let mut out = Vec::with_capacity(images.len());
        let mut at = 0usize;
        for im in images {
            let f = Arc::clone(&im.features)
                .downcast::<GlmImageFeatures>()
                .map_err(|_| ModelError::Request("image features were not produced by this backend".into()))?;
            if f.tokens != im.tokens || f.merged != im.merged {
                return Err(ModelError::Request(format!(
                    "image has {} encoder rows ({}x{}) for {} placeholders ({}x{})",
                    f.tokens, f.merged.0, f.merged.1, im.tokens, im.merged.0, im.merged.1
                )));
            }
            if im.start < at || im.start + im.tokens > prompt.len() {
                return Err(ModelError::Request(format!(
                    "image span at {} of {} tokens does not fit the prompt of {} tokens",
                    im.start,
                    im.tokens,
                    prompt.len()
                )));
            }
            if token.is_some_and(|t| prompt[im.start..im.start + im.tokens].iter().any(|&x| x != t)) {
                return Err(ModelError::Request(format!(
                    "the {} tokens at {} are not all image placeholders",
                    im.tokens, im.start
                )));
            }
            at = im.start + im.tokens;
            out.push(f);
        }
        Ok(out)
    }

    /// The rows of each image that fall inside the chunk `at..end`, as the splice the model writes after the embedding.
    fn splice_for(images: &[ImageSpan], feats: &[Arc<GlmImageFeatures>], at: usize, end: usize) -> Vec<SpliceOp> {
        images
            .iter()
            .zip(feats)
            .filter_map(|(im, f)| {
                let (lo, hi) = (im.start.max(at), (im.start + im.tokens).min(end));
                (lo < hi).then(|| SpliceOp { feats: f.clone(), dst_row: lo - at, src_row: lo - im.start, n: hi - lo })
            })
            .collect()
    }

    /// One prefill slice of `prompt` (text, or with images) from `done`: one chunk, with a snapshot of the slot's state taken at each of `cuts` it passes
    /// (inside the chunk, or, without the large-row scratch, by ending the chunk at the first of them). With no cuts this is exactly the slice
    /// `Backend::prefill_slice` has always run.
    fn prefill_slice_at(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
        cuts: &[usize],
    ) -> CutStep {
        let n = prompt.len();
        if self.dead() {
            return CutStep { step: PrefillStep { done: n, row: Some(nan_row(candidates)) }, captured: Vec::new() };
        }
        // under P2 a slice ends on its segment's 16-row KDA grid unless it ends the prompt (`prefix::slice_rows_for`)
        let contract = super::contract::current();
        let chunk = prefix::slice_rows_for(contract, self.chunk.min(self.deadline_rows.unwrap_or(usize::MAX)));
        // the snapshot slots taken for this slice's cuts and not yet complete, and the snapshots that are (with their page references): given back if the
        // slice fails, handed to the engine if it does not
        let mut taken: Vec<u32> = Vec::new();
        let mut captured: Vec<Captured> = Vec::new();
        let mut run = |taken: &mut Vec<u32>, captured: &mut Vec<Captured>| -> Result<PrefillStep, ModelError> {
            // the images are checked on every slice (it is cheap) so that a bad one fails the sequence before any state of the slot is touched
            let feats = self.image_features(prompt, images)?;
            if done == 0 {
                self.m.begin_slot(slot)?;
                if let Some(s) = self.spec.as_mut() {
                    s.begin(slot);
                }
                self.digests.begin(slot);
            }
            // set on every slice, not only the first: a slot that restored a cached prefix starts at `done > 0`
            self.slot_images[slot] = images.iter().map(|im| im.start..im.start + im.tokens).collect();
            // the drafter absorbs only the rows that reach its ring after the prompt
            self.m.dflash_prompt_end(slot, Some(n));
            // one chunk of the grid a whole-prompt prefill would use. A chunk of two rows or more takes the cuts inside it in the same forward; a model
            // without the large-row scratch ends the chunk at the first cut (`prefix::plan_slice`). Only as many cuts are taken as there are free snapshot slots.
            let free = self.m.prefix_free_snapshots();
            // under P2 the planned cuts move down to the segment's grid (`prefix::cuts_for`); the snapshots report their own lengths
            let anchor = if done == 0 { 0 } else { self.m.p2_anchor(slot).unwrap_or(done) };
            let inside = |x: usize| images.iter().any(|im| x > im.start && x < im.start + im.tokens);
            let cuts = prefix::cuts_for(contract, anchor, done, cuts, &inside);
            let cuts = &cuts[..cuts.len().min(free)];
            let plan = prefix::plan_slice(done, n, chunk, cuts, self.m.can_snapshot_inside_chunks());
            let (at, end) = (done, plan.end);
            let want = (end == n).then_some(candidates);
            let splice = Self::splice_for(images, &feats, at, end);
            let c = if plan.interior.is_empty() {
                if splice.is_empty() {
                    self.m.prefill_chunk(slot, at, &prompt[at..end], want)?
                } else {
                    self.m.prefill_chunk_mm(slot, at, &prompt[at..end], want, splice)?
                }
            } else {
                let mut cut_snaps = Vec::new();
                for &cut in &plan.interior {
                    let snap = self.m.alloc_snapshot().ok_or_else(|| ModelError::Request("no free snapshot slot".into()))?;
                    taken.push(snap);
                    cut_snaps.push((cut, snap));
                }
                let c = self.m.prefill_chunk_cuts(slot, at, &prompt[at..end], want, splice, &cut_snaps)?;
                // the snapshots are complete: the entries take their page references now and own their slots from here, and each is
                // sealed (its content digest recorded: a restore of it is verified)
                taken.clear();
                for &(cut, snap) in &cut_snaps {
                    captured
                        .push(Captured { len: cut, handle: PrefixHandle { snap, pages: self.m.retain_prefix_pages(slot, cut) } });
                }
                for cap in captured.iter() {
                    self.m.seal_snapshot(cap.handle.snap, &cap.handle.pages)?;
                }
                c
            };
            if end < n && cuts.contains(&end) {
                if let Some(snap) = self.m.alloc_snapshot() {
                    taken.push(snap);
                    self.m.prefix_snapshot(slot, snap, end)?;
                    taken.clear();
                    let pages = self.m.retain_prefix_pages(slot, end);
                    captured.push(Captured { len: end, handle: PrefixHandle { snap, pages } });
                    if let Some(cap) = captured.last() {
                        self.m.seal_snapshot(cap.handle.snap, &cap.handle.pages)?;
                    }
                }
            }
            let row = c.map(|c| row_of(&c, 0, candidates));
            if let Some(r) = &row {
                self.digests.row(slot, n - 1, prompt[n - 1], r);
            }
            Ok(PrefillStep { done: end, row })
        };
        match run(&mut taken, &mut captured) {
            Ok(step) => CutStep { step, captured },
            Err(e) => {
                for snap in taken {
                    self.m.release_prefix(snap, &[]);
                }
                for c in captured {
                    self.m.release_prefix(c.handle.snap, &c.handle.pages);
                }
                self.record(&format!("prefill of slot {slot}"), &e, false);
                self.drop_slot(slot);
                CutStep { step: PrefillStep { done: n, row: Some(nan_row(candidates)) }, captured: Vec::new() }
            }
        }
    }

    /// Give `slot`'s state back (and forget its image spans and its drafting state).
    fn drop_slot(&mut self, slot: usize) {
        self.m.end_slot(slot);
        if let Some(s) = self.slot_images.get_mut(slot) {
            s.clear();
        }
        if let Some(s) = self.spec.as_mut() {
            s.end(slot);
        }
    }

    /// Log `what` failed with `e` and remember it for `take_error`. `fatal_anyway`: the failure leaves state the engine relies on out of
    /// step with the committed tokens, so it is fatal whatever the error's own class.
    fn record(&mut self, what: &str, e: &ModelError, fatal_anyway: bool) {
        let fatal = fatal_anyway || e.is_fatal();
        let message = format!("{what}: {e}");
        eprintln!("rexl3 level={} event=backend_error {message}", if fatal { "fatal" } else { "error" });
        if fatal {
            self.fatal.get_or_insert(BackendError::fatal(message));
        } else if self.error.is_none() {
            self.error = Some(BackendError::retryable(message));
        }
    }

    fn dead(&self) -> bool {
        self.fatal.is_some()
    }
}

impl Backend for GlmBackend {
    fn set_prefill_row_limit(&mut self, rows: Option<usize>) -> bool {
        self.deadline_rows = rows;
        true
    }
    fn quiesce(&mut self) -> Result<(), String> {
        gb_cuda::rt::device_sync().map_err(|e| e.to_string())
    }
    /// The tokenizer's vocabulary: the model's logits are 154,880 wide but the last 24 ids are padding that is never offered.
    fn vocab_size(&self) -> usize {
        TOKENIZER_VOCAB
    }

    fn take_error(&mut self) -> Option<BackendError> {
        self.fatal.clone().or_else(|| self.error.take())
    }

    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.prefill_piece(slot, start, tokens, start as usize + tokens.len(), candidates)
    }

    fn prefill_piece(&mut self, slot: usize, start: u64, tokens: &[u32], total: usize, candidates: usize) -> RowOutput {
        if self.dead() {
            return nan_row(candidates);
        }
        let chunk = prefix::slice_rows_for(super::contract::current(), self.chunk);
        let mut run = || -> Result<RowOutput, ModelError> {
            if start == 0 {
                self.m.begin_slot(slot)?;
                if let Some(s) = self.spec.as_mut() {
                    s.begin(slot);
                }
                self.digests.begin(slot);
            }
            self.m.dflash_prompt_end(slot, Some(total.max(start as usize + tokens.len())));
            let (mut pos, n) = (start as usize, tokens.len());
            let mut at = 0;
            let mut last = None;
            while at < n {
                let end = (at + chunk).min(n);
                let want = (end == n).then_some(candidates);
                last = self.m.prefill_chunk(slot, pos, &tokens[at..end], want)?;
                pos += end - at;
                at = end;
            }
            let c = last.ok_or_else(|| ModelError::Request("empty prefill".into()))?;
            let r = row_of(&c, 0, candidates);
            self.digests.row(slot, start as usize + n - 1, tokens[n - 1], &r);
            Ok(r)
        };
        match run() {
            Ok(r) => r,
            Err(e) => {
                self.record(&format!("prefill of slot {slot}"), &e, false);
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
        self.prefill_slice_at(slot, prompt, images, done, candidates, &[]).step
    }

    fn prefix_caps(&self) -> Option<PrefixCaps> {
        self.prefix
    }

    /// GLM has one arithmetic: a prompt of any length leaves the bits its tokens decoded alone leave, whatever the chunk grid, so one path id serves
    /// every total (no 128-token rule).
    fn prefix_path(&self, _total: usize) -> Option<u32> {
        self.prefix.as_ref().map(|_| prefix::PATH_ID)
    }

    fn prefix_free_snapshots(&self) -> usize {
        self.m.prefix_free_snapshots()
    }

    fn prefix_restore(&mut self, slot: usize, handle: &PrefixHandle, len: usize) -> Result<(), String> {
        if self.dead() {
            return Err("the backend has failed".into());
        }
        match self.m.restore_prefix(slot, handle.snap, &handle.pages, len) {
            Ok(()) => {
                // the sequence starts over from the entry: its drafting state is new (the model reset the drafter's books and set the carry), and
                // its image spans are set by its next slice
                if let Some(s) = self.slot_images.get_mut(slot) {
                    s.clear();
                }
                if let Some(s) = self.spec.as_mut() {
                    s.begin(slot);
                }
                self.digests.begin(slot);
                Ok(())
            }
            Err(e) => {
                self.record(&format!("prefix restore into slot {slot}"), &e, false);
                self.drop_slot(slot);
                Err(e.to_string())
            }
        }
    }

    fn prefix_release(&mut self, handle: PrefixHandle) {
        self.m.release_prefix(handle.snap, &handle.pages);
    }

    /// [`Backend::prefill_slice`] that takes a snapshot at each of `cuts` the slice passes (`prefill_slice_at`): inside the chunk, in the same forward, or
    /// (a model without the large-row scratch) with the slice ended at the cut. The bits of every row do not depend on the chunk grid.
    fn prefill_slice_cuts(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
        cuts: &[usize],
    ) -> CutStep {
        let cuts = if self.prefix.is_some() { cuts } else { &[] };
        self.prefill_slice_at(slot, prompt, images, done, candidates, cuts)
    }

    /// A prompt with images from its start, in one call (the engine uses `prefill_slice`).
    fn prefill_mm(&mut self, slot: usize, start: u64, tokens: &[u32], images: &[ImageSpan], candidates: usize) -> RowOutput {
        if images.is_empty() {
            return self.prefill(slot, start, tokens, candidates);
        }
        if start != 0 {
            let e = ModelError::Request("a prompt with images must be prefilled from its start".into());
            self.record(&format!("prefill of slot {slot}"), &e, false);
            return nan_row(candidates);
        }
        let mut done = 0;
        loop {
            let step = self.prefill_slice(slot, tokens, images, done, candidates);
            done = step.done;
            if let Some(row) = step.row {
                return row;
            }
        }
    }

    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        let nans = || reqs.iter().map(|q| q.tokens.iter().map(|_| nan_row(candidates)).collect()).collect();
        if self.dead() {
            return nans();
        }
        let b = RowBatch::from_requests(reqs);
        if let Some(s) = self.spec.as_mut() {
            for q in reqs {
                s.on_step(q.slot, q.tokens.len());
            }
        }
        let started = std::time::Instant::now();
        match self.m.window_candidates(&b, candidates) {
            Ok(c) => {
                if let Some(s) = self.spec.as_mut() {
                    s.on_window(b.rows, started.elapsed().as_secs_f64() * 1e3);
                }
                let out: Vec<Vec<RowOutput>> =
                    (0..reqs.len()).map(|i| b.seq_rows(i).map(|r| row_of(&c, r, candidates)).collect()).collect();
                if self.digests.on() {
                    for (q, rows) in reqs.iter().zip(&out) {
                        for (j, r) in rows.iter().enumerate() {
                            self.digests.row(q.slot, q.first_position as usize + j, q.tokens[j], r);
                        }
                    }
                }
                out
            }
            Err(e) => {
                self.record("window step", &e, false);
                nans()
            }
        }
    }

    fn commit(&mut self, slot: usize, keep: usize) {
        // only records the keep count: the KDA commits of every layer are applied together before the next forward
        self.m.commit(slot, keep);
        if let Some(s) = self.spec.as_mut() {
            s.on_commit(slot, keep);
        }
    }

    fn release(&mut self, slot: usize) {
        self.drop_slot(slot);
    }

    /// Drafts for the streams that may draft this round. A drafting error (a recoverable one) costs the round its native drafts only: the
    /// engine's own per-sequence drafter fills in, and drafts never change what is emitted.
    fn propose_batch(&mut self, reqs: &[ProposeRequest]) -> Vec<Option<Vec<u32>>> {
        if self.dead() || self.spec.is_none() {
            return vec![None; reqs.len()];
        }
        let GlmBackend { m, spec, slot_images, .. } = self;
        match spec.as_mut().unwrap().propose(m, reqs, slot_images) {
            Ok(v) => v,
            Err(e) => {
                self.record(&format!("drafting for {} sequences", reqs.len()), &e, false);
                vec![None; reqs.len()]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::window_is_text;

    #[test]
    fn a_window_that_touches_an_image_span_is_not_text() {
        let spans = vec![10..20, 40..45];
        // no spans: every window is text
        assert!(window_is_text(&[], 0, 100));
        // the rows 0..10 end where the first span starts; 0..11 reach its first row
        assert!(window_is_text(&spans, 0, 10));
        assert!(!window_is_text(&spans, 0, 11));
        // a window inside a span, one that starts on its last row, one that starts just after it
        assert!(!window_is_text(&spans, 12, 2));
        assert!(!window_is_text(&spans, 19, 5));
        assert!(window_is_text(&spans, 20, 20));
        // the window that bridges the gap and the second span
        assert!(!window_is_text(&spans, 20, 21));
        assert!(window_is_text(&spans, 45, 1000));
        // a window of no rows inside a span touches nothing it would absorb, but the rule says where it is: still inside
        assert!(!window_is_text(&spans, 15, 0));
        // a huge row count does not overflow
        assert!(!window_is_text(&spans, 5, usize::MAX));
    }
}
