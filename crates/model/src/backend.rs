//! `gb_engine::Backend` over the GPU model.
//!
//! The trait's methods cannot fail, so a GPU or request error is logged, answered with NaN candidate rows (the engine's sampler
//! refuses NaN) and recorded for `Backend::take_error`, which the engine reads after every call: a recoverable error fails the
//! sequences the call served, a fatal one (a sticky CUDA error, lost recurrent state: `ModelError::is_fatal`) stops the engine and
//! the process. Once a fatal error has been seen it is latched and no further call touches the device.

use crate::forward::{BatchPrompt, Candidates, HeadMode, Model, PrefillPath, SnapCut, SpliceOp};
use crate::mrope;
use crate::rows::RowBatch;
use crate::vision::ImageFeatures;
use gb_cuda::dflash2::DraftCtxLayout;
use gb_engine::{
    Backend, BackendError, CacheBudget, Captured, CutStep, ImageSpan, PrefillJob, PrefillStep, PrefixCaps, PrefixHandle,
    RowOutput, RowRequest, TreeRequest,
};

pub struct QwenBackend {
    pub m: Model,
    /// Rows per prefill chunk.
    pub chunk: usize,
    deadline_rows: Option<usize>,
    /// The first recoverable error since `take_error` last ran.
    error: Option<BackendError>,
    /// The first fatal error: it never clears.
    fatal: Option<BackendError>,
    /// The prefix cache this backend reserved (its namespace and budget), when one is enabled.
    prefix: Option<PrefixCaps>,
}

fn nan_row(n: usize) -> RowOutput {
    RowOutput { ids: (0..n as u32).collect(), values: vec![f32::NAN; n] }
}

fn row_of(c: &Candidates, r: usize, n: usize) -> RowOutput {
    let ids = c.ids[r * c.count..r * c.count + n].iter().map(|&x| x as u32).collect();
    let values = c.vals[r * c.count..r * c.count + n].to_vec();
    RowOutput { ids, values }
}

impl QwenBackend {
    pub fn new(m: Model, chunk: usize) -> Self {
        Self { m, chunk, deadline_rows: None, error: None, fatal: None, prefix: None }
    }

    /// Enable the prefix cache: allocates `budget.snapshot_slots` snapshot slots (the draft's window included when a draft is attached to the
    /// model: attach it first). `namespace` binds every key to what the snapshots depend on (see `serve::prefix_namespace`). The KV pool must
    /// already hold `budget.page_blocks` pages more than admission hands out (the memory plan does that).
    pub fn with_prefix_cache(mut self, namespace: u64, budget: CacheBudget) -> Result<Self, crate::error::ModelError> {
        self.m.enable_prefix_cache(budget.snapshot_slots)?;
        self.prefix = Some(PrefixCaps { namespace, budget });
        Ok(self)
    }

    /// Log `what` failed with `e` and remember it for `take_error`. `fatal_anyway`: the failure leaves state the engine relies on
    /// out of step with the committed tokens, so it is fatal whatever the error's own class.
    fn record(&mut self, what: &str, e: &crate::error::ModelError, fatal_anyway: bool) {
        let fatal = fatal_anyway || e.is_fatal();
        let message = format!("{what}: {e}");
        eprintln!("rexl3 level={} event=backend_error {message}", if fatal { "fatal" } else { "error" });
        if fatal {
            self.fatal.get_or_insert(BackendError::fatal(message));
        } else if self.error.is_none() {
            self.error = Some(BackendError::retryable(message));
        }
    }

    /// A fatal error was seen: the device or the recurrent state cannot be trusted any more.
    fn dead(&self) -> bool {
        self.fatal.is_some()
    }
}

impl QwenBackend {
    /// One prefill slice of `prompt` (text, or with images) from `done`, taking a snapshot at each of `cuts` (ascending, absolute) the slice
    /// passes. With no cuts this is exactly the slice `Backend::prefill_slice` has always run.
    fn prefill_slice_at(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
        cuts: &[usize],
    ) -> CutStep {
        self.prefill_slice_at_head(slot, prompt, images, done, candidates, cuts, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_slice_at_head(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
        cuts: &[usize],
        full_row: bool,
    ) -> CutStep {
        let n = prompt.len();
        if self.dead() {
            return CutStep { step: PrefillStep { done: n, row: Some(nan_row(candidates)) }, captured: Vec::new() };
        }
        // snapshot slots taken for this slice's cuts: given back if the slice fails, handed to the engine (with page references) if it does not
        let mut taken: Vec<(usize, u32)> = Vec::new();
        let result = {
            let mut run = |taken: &mut Vec<(usize, u32)>| -> Result<CutStep, crate::error::ModelError> {
                use crate::error::ModelError;
                let mut feats: Vec<&ImageFeatures> = Vec::with_capacity(images.len());
                for im in images {
                    let f = crate::serve::feature_ref::<ImageFeatures>(&*im.features)
                        .ok_or_else(|| ModelError::Request("image features were not produced by this backend".into()))?;
                    if f.tokens != im.tokens {
                        return Err(ModelError::Request(format!(
                            "image has {} encoder rows for {} placeholders",
                            f.tokens, im.tokens
                        )));
                    }
                    feats.push(f);
                }
                if done == 0 {
                    self.m.begin_slot(slot)?;
                }
                // one chunk of the same grid a whole-prompt prefill would use. The bits do not depend on how often we are called:
                // the prefill arithmetic is picked by the prompt's length `n`, not by this chunk's rows, and is row invariant
                let (at, end) = (done, (done + self.chunk.min(self.deadline_rows.unwrap_or(usize::MAX))).min(n));
                for &c in cuts.iter().filter(|&&c| c > at && c <= end) {
                    match self.m.snaps.as_mut().and_then(|s| s.alloc()) {
                        Some(snap) => taken.push((c, snap)),
                        None => break, // no free slot: that cut is skipped, the prompt does not mind
                    }
                }
                // the draft's context starts at the end of the prompt minus its keep, or earlier when a cut still to come needs the window
                // before it
                let lowest = cuts.iter().filter(|&&c| c > at).map(|&c| c.saturating_sub(DraftCtxLayout::WINDOW)).min();
                let ctx_from = n.saturating_sub(crate::draft::CTX_KEEP).min(lowest.unwrap_or(usize::MAX));
                let snap: Vec<SnapCut> = taken.iter().map(|&(at, snap)| SnapCut { at, snap }).collect();
                let want = (end == n).then_some(candidates);
                let c = if images.is_empty() {
                    let head = if end == n && full_row {
                        HeadMode::FullLastRow
                    } else {
                        want.map_or(HeadMode::None, HeadMode::Candidates)
                    };
                    self.m.prefill_chunk_snap_head(slot, at, &prompt[at..end], n, head, ctx_from, None, &[], &snap)?
                } else {
                    let runs: Vec<mrope::Run> = images
                        .iter()
                        .map(|im| mrope::Run { start: im.start, tokens: im.tokens, rows: im.merged.0, cols: im.merged.1 })
                        .collect();
                    let (pos3, rope_end) = mrope::positions(n, &runs);
                    let has_image = images.iter().any(|im| im.start < end && im.start + im.tokens > at);
                    // text-only chunk: rope = cache index + delta (the delta of the rows before the chunk); image chunk: explicit streams
                    self.m.set_rope_delta(slot, pos3[at][0] - at as i32);
                    let rope3 = has_image.then(|| pos3[at..end].iter().flatten().copied().collect::<Vec<i32>>());
                    let splice: Vec<SpliceOp> = images
                        .iter()
                        .zip(&feats)
                        .filter_map(|(im, f)| {
                            let (lo, hi) = (im.start.max(at), (im.start + im.tokens).min(end));
                            (lo < hi).then_some(SpliceOp { feats: &f.buf, dst_row: lo - at, src_row: lo - im.start, n: hi - lo })
                        })
                        .collect();
                    let c = self.m.prefill_chunk_snap(slot, at, &prompt[at..end], n, want, ctx_from, rope3, &splice, &snap)?;
                    if end == n {
                        // generated tokens continue at rope position `rope_end`, i.e. `rope_end - n` behind their cache index
                        self.m.set_rope_delta(slot, rope_end - n as i32);
                    }
                    c
                };
                // Gather only after THIS request's explicit head succeeded. Prefill writes row zero,
                // regardless of the last chunk's size or a prefix cache hit.
                let row = if full_row && end == n {
                    Some(RowOutput { ids: Vec::new(), values: self.m.score_head_row_zero()? })
                } else {
                    c.map(|c| row_of(&c, 0, candidates))
                };
                // the snapshots are complete: the entries take their page references now and own their slots from here
                let captured = taken
                    .drain(..)
                    .map(|(len, snap)| Captured {
                        len,
                        handle: PrefixHandle { snap, pages: self.m.retain_prefix_pages(slot, len) },
                    })
                    .collect();
                Ok(CutStep { step: PrefillStep { done: end, row }, captured })
            };
            run(&mut taken)
        };
        match result {
            Ok(r) => r,
            Err(e) => {
                if let Some(sp) = self.m.snaps.as_mut() {
                    for (_, snap) in taken {
                        sp.free_slot(snap);
                    }
                }
                self.record(&format!("prefill of slot {slot}"), &e, false);
                self.m.end_slot(slot);
                CutStep { step: PrefillStep { done: n, row: Some(nan_row(candidates)) }, captured: Vec::new() }
            }
        }
    }
}

impl Backend for QwenBackend {
    fn set_prefill_row_limit(&mut self, rows: Option<usize>) -> bool {
        self.deadline_rows = rows;
        true
    }
    fn quiesce(&mut self) -> Result<(), String> {
        gb_cuda::rt::device_sync().map_err(|e| e.to_string())
    }
    fn score_supported(&self) -> bool {
        true
    }
    fn score_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        done: usize,
        cuts: &[usize],
    ) -> Result<gb_engine::score::ScoreStep, BackendError> {
        if let Some(e) = self.fatal.clone() {
            return Err(e);
        }
        let r = self.prefill_slice_at_head(slot, prompt, &[], done, 1, cuts, true);
        if let Some(e) = self.take_error() {
            for c in r.captured {
                self.prefix_release(c.handle);
            }
            return Err(e);
        }
        Ok(gb_engine::score::ScoreStep { done: r.step.done, row: r.step.row.map(|r| r.values), captured: r.captured })
    }
    fn vocab_size(&self) -> usize {
        self.m.cfg.vocab
    }

    fn take_error(&mut self) -> Option<BackendError> {
        self.fatal.clone().or_else(|| self.error.take())
    }

    /// The tokens end the prompt (its length is `start + tokens.len()`), which picks the prefill arithmetic; a piece of a longer
    /// prompt goes through [`Backend::prefill_piece`].
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.prefill_piece(slot, start, tokens, start as usize + tokens.len(), candidates)
    }

    fn prefill_piece(&mut self, slot: usize, start: u64, tokens: &[u32], total: usize, candidates: usize) -> RowOutput {
        if self.dead() {
            return nan_row(candidates);
        }
        let mut run = || -> Result<RowOutput, crate::error::ModelError> {
            if start == 0 {
                self.m.begin_slot(slot)?;
            }
            let mut pos = start as usize;
            let ctx_from = total.saturating_sub(crate::draft::CTX_KEEP);
            let mut last = None;
            let n = tokens.len();
            let mut at = 0;
            while at < n {
                let end = (at + self.chunk).min(n);
                let want = (end == n).then_some(candidates);
                last = self.m.prefill_chunk(slot, pos, &tokens[at..end], total, want, ctx_from)?;
                pos += end - at;
                at = end;
            }
            let c = last.ok_or_else(|| crate::error::ModelError::Request("empty prefill".into()))?;
            Ok(row_of(&c, 0, candidates))
        };
        match run() {
            Ok(r) => r,
            Err(e) => {
                self.record(&format!("prefill of slot {slot}"), &e, false);
                self.m.end_slot(slot);
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

    /// Prompts above 128 tokens prefill on the pinned GEMM (path 1); a prompt of 128 or fewer is on the decode GEMV, whose state no other
    /// path's prompt may resume from, so it is not cached at all.
    fn prefix_path(&self, total: usize) -> Option<u32> {
        self.prefix.as_ref()?;
        (PrefillPath::for_prompt(total) == PrefillPath::Tiled).then_some(1)
    }

    fn prefix_free_snapshots(&self) -> usize {
        self.m.snaps.as_ref().map_or(0, |s| s.free_count())
    }

    fn prefix_restore(&mut self, slot: usize, handle: &PrefixHandle, len: usize) -> Result<(), String> {
        if self.dead() {
            return Err("the backend has failed".into());
        }
        match self.m.restore_prefix(slot, handle.snap, &handle.pages, len) {
            Ok(()) => Ok(()),
            Err(e) => {
                self.record(&format!("prefix restore into slot {slot}"), &e, false);
                self.m.end_slot(slot);
                Err(e.to_string())
            }
        }
    }

    fn prefix_release(&mut self, handle: PrefixHandle) {
        self.m.release_prefix(handle.snap, &handle.pages);
    }

    /// [`Backend::prefill_slice`] that takes a snapshot at each of `cuts` the slice passes. The chunk is the same as without cuts; see
    /// `Model::prefill_chunk_snap` for what is copied and why no extra W_q reconstruct is paid.
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

    /// The prompts that are whole, short and text (done 0, at most 128 tokens, one chunk) go through the model together, as many to a
    /// forward as fit in 128 rows: they share one pass of every EXL3 layer, which is what a lone short prompt spends nearly all of its
    /// time on. Every prompt comes out with the bits its own `prefill_slice` gives it (`Model::prefill_batch`), and one that cannot be
    /// served with the others is prefilled on its own, so a failure belongs to the prompt that has it. Everything else (a long prompt, a
    /// resumed one, one with images) takes `prefill_slice`, in job order.
    fn prefill_slices(&mut self, jobs: &[PrefillJob<'_>]) -> Vec<CutStep> {
        let short = |j: &PrefillJob<'_>| {
            j.done == 0 && j.images.is_empty() && j.prompt.len() <= PrefillPath::GEMV_MAX_PROMPT && j.prompt.len() <= self.chunk
        };
        // groups of consecutive short jobs, each of at most 128 rows in all and at most a window of prompts
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut rows = 0usize;
        for (i, j) in jobs.iter().enumerate().filter(|(_, j)| short(j)) {
            let fits = groups
                .last()
                .is_some_and(|g| rows + j.prompt.len() <= crate::scratch::MAX_DECODE_ROWS && g.len() < self.m.window_rows());
            if fits {
                rows += j.prompt.len();
                groups.last_mut().unwrap().push(i);
            } else {
                rows = j.prompt.len();
                groups.push(vec![i]);
            }
        }
        let mut out: Vec<Option<PrefillStep>> = jobs.iter().map(|_| None).collect();
        for g in groups.iter().filter(|g| g.len() >= 2) {
            if self.dead() {
                for &i in g {
                    out[i] = Some(PrefillStep { done: jobs[i].prompt.len(), row: Some(nan_row(jobs[i].candidates)) });
                }
                continue;
            }
            let prompts: Vec<BatchPrompt<'_>> = g
                .iter()
                .map(|&i| BatchPrompt {
                    slot: jobs[i].slot,
                    tokens: jobs[i].prompt,
                    ctx_from: jobs[i].prompt.len().saturating_sub(crate::draft::CTX_KEEP),
                })
                .collect();
            let want = g.iter().map(|&i| jobs[i].candidates).max().unwrap_or(1);
            match self.m.prefill_batch(&prompts, want) {
                Ok(c) => {
                    for (k, &i) in g.iter().enumerate() {
                        out[i] = Some(PrefillStep { done: jobs[i].prompt.len(), row: Some(row_of(&c, k, jobs[i].candidates)) });
                    }
                }
                Err(e) if e.is_fatal() => {
                    self.record(&format!("batched prefill of {} prompts", g.len()), &e, false);
                    for &i in g {
                        out[i] = Some(PrefillStep { done: jobs[i].prompt.len(), row: Some(nan_row(jobs[i].candidates)) });
                    }
                }
                Err(e) => {
                    // not fatal: say so and serve each prompt alone below, so that the one that has the problem is the one that fails
                    eprintln!("rexl3 level=warn event=prefill_batch_fallback prompts={} error=\"{e}\"", g.len());
                    for &i in g {
                        self.m.end_slot(jobs[i].slot);
                    }
                }
            }
        }
        jobs.iter()
            .enumerate()
            .map(|(i, j)| {
                let step = match out[i].take() {
                    Some(step) => step,
                    None => self.prefill_slice(j.slot, j.prompt, j.images, j.done, j.candidates),
                };
                CutStep { step, captured: Vec::new() }
            })
            .collect()
    }

    fn prefill_mm(&mut self, slot: usize, start: u64, tokens: &[u32], images: &[ImageSpan], candidates: usize) -> RowOutput {
        if self.dead() {
            return nan_row(candidates);
        }
        let mut run = || -> Result<RowOutput, crate::error::ModelError> {
            use crate::error::ModelError;
            if start != 0 {
                return Err(ModelError::Request("a prompt with images must be prefilled from its start".into()));
            }
            let mut feats: Vec<&ImageFeatures> = Vec::with_capacity(images.len());
            for im in images {
                let f = crate::serve::feature_ref::<ImageFeatures>(&*im.features)
                    .ok_or_else(|| ModelError::Request("image features were not produced by this backend".into()))?;
                if f.tokens != im.tokens {
                    return Err(ModelError::Request(format!(
                        "image has {} encoder rows for {} placeholders",
                        f.tokens, im.tokens
                    )));
                }
                feats.push(f);
            }
            self.m.begin_slot(slot)?;
            let n = tokens.len();
            let runs: Vec<mrope::Run> = images
                .iter()
                .map(|im| mrope::Run { start: im.start, tokens: im.tokens, rows: im.merged.0, cols: im.merged.1 })
                .collect();
            let (pos3, rope_end) = mrope::positions(n, &runs);
            let ctx_from = n.saturating_sub(crate::draft::CTX_KEEP);
            let (mut last, mut at) = (None, 0usize);
            while at < n {
                let end = (at + self.chunk).min(n);
                let want = (end == n).then_some(candidates);
                let has_image = images.iter().any(|im| im.start < end && im.start + im.tokens > at);
                // text-only chunk: rope = cache index + delta (the delta of the rows before the chunk); image chunk: explicit streams
                self.m.set_rope_delta(slot, pos3[at][0] - at as i32);
                let rope3 = has_image.then(|| pos3[at..end].iter().flatten().copied().collect::<Vec<i32>>());
                let splice: Vec<SpliceOp> = images
                    .iter()
                    .zip(&feats)
                    .filter_map(|(im, f)| {
                        let (lo, hi) = (im.start.max(at), (im.start + im.tokens).min(end));
                        (lo < hi).then_some(SpliceOp { feats: &f.buf, dst_row: lo - at, src_row: lo - im.start, n: hi - lo })
                    })
                    .collect();
                last = self.m.prefill_chunk_mm(slot, at, &tokens[at..end], n, want, ctx_from, rope3, &splice)?;
                at = end;
            }
            // generated tokens continue at rope position `rope_end`, i.e. `rope_end - n` behind their cache index
            self.m.set_rope_delta(slot, rope_end - n as i32);
            let c = last.ok_or_else(|| ModelError::Request("empty prefill".into()))?;
            Ok(row_of(&c, 0, candidates))
        };
        match run() {
            Ok(r) => r,
            Err(e) => {
                self.record(&format!("multimodal prefill of slot {slot}"), &e, false);
                self.m.end_slot(slot);
                nan_row(candidates)
            }
        }
    }

    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        let nans = || reqs.iter().map(|q| q.tokens.iter().map(|_| nan_row(candidates)).collect()).collect();
        if self.dead() {
            return nans();
        }
        let b = RowBatch::from_requests(reqs);
        match self.m.window_candidates(&b, candidates) {
            Ok(c) => (0..reqs.len()).map(|i| b.seq_rows(i).map(|r| row_of(&c, r, candidates)).collect()).collect(),
            Err(e) => {
                self.record("window step", &e, false);
                nans()
            }
        }
    }

    fn supports_trees(&self) -> bool {
        self.m.trees_enabled().0
    }

    fn step_tree(&mut self, reqs: &[TreeRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        let nans = || reqs.iter().map(|q| q.tokens.iter().map(|_| nan_row(candidates)).collect()).collect();
        if self.dead() {
            return nans();
        }
        let b = RowBatch::from_trees(reqs);
        match self.m.window_candidates(&b, candidates) {
            Ok(c) => (0..reqs.len()).map(|i| b.seq_rows(i).map(|r| row_of(&c, r, candidates)).collect()).collect(),
            Err(e) => {
                self.record("tree window step", &e, false);
                nans()
            }
        }
    }

    fn commit_path(&mut self, slot: usize, path: &[usize]) {
        // a path that is not a root path of the window's tree would leave the state out of step with the tokens: fatal
        if let Err(e) = self.m.commit_path(slot, path) {
            self.record(&format!("commit of a path in slot {slot}"), &e, true);
        }
    }

    fn commit(&mut self, slot: usize, keep: usize) {
        // only records the keep count: the conv history is committed before the next forward (`Model::flush`) and the replay into
        // S is folded into the sequence's next window kernel (rows of a sequence that sits that window out are replayed before it,
        // a slot about to be prefilled settles first). A failed conv commit or a failed replay for sequences that sit the window
        // out is `StateLost` and fatal
        self.m.commit(slot, keep);
    }

    fn release(&mut self, slot: usize) {
        self.m.end_slot(slot);
    }

    fn propose_batch(&mut self, reqs: &[gb_engine::ProposeRequest]) -> Vec<Option<Vec<u32>>> {
        self.propose_batch_scored(reqs).into_iter().map(|o| o.map(|p| p.tokens)).collect()
    }

    fn propose_batch_scored(&mut self, reqs: &[gb_engine::ProposeRequest]) -> Vec<Option<gb_engine::Proposal>> {
        if self.dead() {
            return vec![None; reqs.len()];
        }
        // the draft computes only the picks asked for (the head, top-k and walk run for those rows): the ask is the block the
        // engine wants (the depth-adaptive block of a tree round, 15 at a 16-row block) clamped to the block the draft was
        // loaded with, so an 8-row draft never sees a 15-deep ask
        let block = self.m.draft.as_ref().map_or(crate::draft::DRAFTS + 1, |d| d.block);
        let asks: Vec<(usize, &[u32], &gb_sampling::Sampling, usize)> =
            reqs.iter().map(|r| (r.slot, r.tokens, r.sampling, r.max.min(block - 1))).collect();
        match self.m.propose_drafts_batch_want(&asks) {
            Ok(out) => out
                .into_iter()
                .zip(reqs)
                .map(|(o, r)| {
                    o.map(|mut p| {
                        p.tokens.truncate(r.max);
                        p.score.truncate(r.max);
                        p.margin.truncate(r.max);
                        gb_engine::Proposal {
                            tokens: p.tokens,
                            conf: p.score,
                            margin: p.margin,
                            lattice: p.lattice,
                            ..Default::default()
                        }
                    })
                })
                .collect(),
            Err(e) => {
                // a recoverable drafting error only costs this round its native drafts; a fatal one (or lost state) stops the engine
                self.record(&format!("drafting for {} sequences", reqs.len()), &e, false);
                vec![None; reqs.len()]
            }
        }
    }

    fn propose(&mut self, slot: usize, tokens: &[u32], sampling: &gb_sampling::Sampling, max: usize) -> Option<Vec<u32>> {
        if self.dead() {
            return None;
        }
        match self.m.propose_drafts(slot, tokens, sampling) {
            Ok(Some(mut d)) => {
                d.truncate(max);
                Some(d)
            }
            Ok(None) => None,
            Err(e) => {
                self.record(&format!("drafting for slot {slot}"), &e, false);
                None
            }
        }
    }
}
