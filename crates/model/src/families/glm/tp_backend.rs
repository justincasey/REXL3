//! `gb_engine::Backend` of rank 0 of a GLM tensor-parallel group: the text-only subset of `backend::GlmBackend` over a
//! [`TpLeader`], which forwards every state-changing call to the followers before it runs it here (`crate::lockstep`).
//!
//! It makes the model calls `GlmBackend` makes for the same engine calls, in the same order and with the same arguments, for the
//! configuration this milestone covers (`docs/GLM_TP.md`): images when rank 0 has the vision tower ([`GlmTpBackend::with_vision`]:
//! a chunk with image rows is the family call `PrefillImage`, `tp_vision.rs`), the backend's drafters ([`GlmTpBackend::with_spec`]:
//! rank 0 decides, the NextN draft is the family call `MtpDraft`, `tp_mtp.rs`), both together (`docs/GLM_TP.md` 11), and the prefix
//! cache ([`GlmTpBackend::with_prefix_cache`]). A prompt is fed in the same chunks (`prefix::plan_slice`) and every text chunk is one core
//! `prefill_chunk` call with the batch `RowBatch::from_requests` makes, which is exactly the batch `GlmModel::prefill_chunk` builds on one
//! node. Anything outside the subset is refused per request, before any call is forwarded. `GlmBackend` itself is not touched: one node
//! runs what it always ran.
//!
//! **The prefix cache** (`docs/GLM_TP.md` 4.2): rank 0's engine decides every hit, each rank snapshots and restores its own state through
//! the calls of `tp.rs` (`EnableSnapshots`, `Snapshot`, `Restore`, `Release`), and a snapshot is taken only BETWEEN chunks: a slice ends at
//! each cut (`prefix::plan_slice` with `split_ok` false; a rank of a KDA head split runs each chunk's KDA whole, so no cut is taken inside
//! one), which moves no bit. Rank 0 translates its own handles to the sequence number of the `Snapshot` call that made them, which is what
//! a follower keys its handle by. The snapshot, restore and release are cache calls (`lockstep::TpLeader::forward_cache_call`): rank 0 waits
//! for every rank's answer to a snapshot and a restore, keeps every rank's content digest of each snapshot in its `lockstep::SnapshotBook`,
//! and uses a restore only when every rank's digest of what it restored equals its own at snapshot time. A snapshot any rank did not take
//! is released everywhere and never handed to the engine; a restore that is not verified is a request error, which the engine answers with
//! a cold prefill that begins the slot again on every rank.
//!
//! Generic over the runtime so that the CPU tests drive this very backend over a recording mock (`tests` below).

use super::alloc::TOKENIZER_VOCAB;
use super::digest_log::DigestLog;
use super::prefix;
use super::spec::{SpecConfig, SpecRuntime};
use super::tp::{self, GlmCall, PrefixRuntime, Taken};
use super::tp_mtp::GlmMtpRuntime;
use super::tp_part::{self, PartRuntime, PrefillPart, TO_END};
use super::tp_vision::{self, ChunkCall, ImageRuntime, SlotImages, TpVision};
use crate::error::ModelError;
use crate::forward::Candidates;
use crate::lockstep::{self, TpLeader};
use crate::rows::RowBatch;
use gb_engine::{
    Backend, BackendError, CacheBudget, Captured, CutStep, ImageSpan, PrefillJob, PrefillStep, PrefixCaps, PrefixHandle,
    ProposeRequest, RowOutput, RowRequest,
};
use std::collections::BTreeMap;

fn nan_row(n: usize) -> RowOutput {
    RowOutput { ids: (0..n as u32).collect(), values: vec![f32::NAN; n] }
}

fn row_of(c: &Candidates, r: usize, n: usize) -> RowOutput {
    let ids = c.ids[r * c.count..r * c.count + n].iter().map(|&x| x as u32).collect();
    let values = c.vals[r * c.count..r * c.count + n].to_vec();
    RowOutput { ids, values }
}

/// Rank 0's backend of a GLM group.
pub struct GlmTpBackend<R: GlmMtpRuntime + ImageRuntime + PrefixRuntime + PartRuntime> {
    pub leader: TpLeader<R>,
    /// Rows of one prefill slice (the model's chunk: the gate's `prefill_rows`).
    chunk: usize,
    /// The adaptive slice (lane E Phase 1, `with_adaptive`): rows of a slice while a sequence is decoding (`chunk` otherwise); `chunk`
    /// when the slice has one size. Every slice is a `prefill_chunk` of consecutive rows, so no bit depends on the grid.
    small: usize,
    /// The engine's last word (`Backend::set_decode_waiting`): a sequence is decoding.
    decode_waiting: bool,
    /// The prefill contract whose slice grid this backend cuts prompts on: the process's ([`super::contract::current`]; P2 since the owner's
    /// decision of 2026-10-19) unless a test pins one ([`GlmTpBackend::with_contract`]).
    contract: super::contract::PrefillContract,
    deadline_rows: Option<usize>,
    error: Option<BackendError>,
    fatal: Option<BackendError>,
    /// `REXL3_GLM_DIGEST_LOG`, as `GlmBackend`'s (the two-node comparison diffs the two logs).
    digests: DigestLog,
    /// Images: rank 0 holds the vision tower (`None`: an image is refused by name, before any call is forwarded for it).
    vision: Option<TpVision>,
    /// Per slot, the prompt positions that hold image rows (rank 0's host state, as `GlmBackend::slot_images`; never forwarded): the
    /// drafting runtime reads them (`spec_allowed_for_window`), as one node's does.
    images: SlotImages,
    /// The drafters and the arbiter (`--spec mtp|auto|copy`), rank 0's alone: what they decide reaches the followers as the arguments of
    /// the draft call (`tp_mtp`). The hooks run where `GlmBackend` runs them.
    pub spec: Option<SpecRuntime>,
    /// The prefix cache this group reserved (its namespace and budget), when one is enabled ([`GlmTpBackend::with_prefix_cache`]).
    prefix: Option<PrefixCaps>,
    /// Rank 0's translation of its snapshot handles (Nemotron's leader `snaps`): its own snapshot slot to the sequence number of the
    /// `Snapshot` call that made it (what every follower keys its own handle by) and the pages it retained. A handle is forwarded only if
    /// it is in here exactly as it was made.
    snaps: BTreeMap<u32, (u64, Vec<u32>)>,
    /// Every rank's content digest of each snapshot every rank holds (by the call that took it).
    book: lockstep::SnapshotBook,
    /// Snapshots taken at a slice end whose drafter book was not sound (`SnapMeta::mtp_ok` false): a warm turn from one of them would not
    /// draft. 0 is the expected count (the book is built chunk by chunk up to the slice's end).
    snaps_without_drafts: u64,
    /// The slices of several waiting prompts go through ONE forward (`tp_part`, P1 and P2): every rank runs them.
    batches: bool,
    /// The rows a forward may have while a prompt forward is paused (every rank's stash: `tp_part::PART_ROOM`; 0: no parts).
    part_room: usize,
    /// While a sequence decodes, a slice longer than the room runs in parts of about this many milliseconds (`tp_part::PART_MS_ENV`; 0:
    /// whole slices, the adaptive small slice instead).
    part_ms: f64,
    /// While a sequence decodes, the most rows the fresh short slices of one forward have (`tp_part::BATCH_ROWS_ENV`).
    batch_rows: usize,
    /// A call ran the fresh short slices while another job waited (the slice in parts, a long slice, an image): the next call runs that
    /// one ([`GlmTpBackend::plan_call`]).
    other_owed: bool,
    /// The layers of the forward (a part's bound is a layer).
    layers: usize,
    /// Measured over modelled time of the parts run so far, smoothed ([`GlmTpBackend::part_layers`]).
    part_scale: f64,
    /// The prompt slice running in parts, while one is.
    inflight: Option<Inflight>,
}

/// A prompt slice running in parts: its slot, prompt length and rows `done..end`, the candidates its prompt asks for, the next layer.
#[derive(Clone, Copy, Debug)]
struct Inflight {
    slot: usize,
    n: usize,
    done: usize,
    end: usize,
    candidates: usize,
    next: usize,
}

/// Modelled milliseconds of one layer of a P2 prompt forward of `rows` rows on a two-rank group (the measured 4,096-row chunk of 2.8 s
/// over 45 layers, and the weight reads' floor of a small one); the measured parts scale it ([`GlmTpBackend::part_layers`]).
fn layer_ms(rows: usize) -> f64 {
    6.0 + 0.014 * rows as f64
}

impl<R: GlmMtpRuntime + ImageRuntime + PrefixRuntime + PartRuntime> GlmTpBackend<R> {
    /// Rank 0's backend over `leader`, with prefill slices of `chunk` rows, for a model of `slots` slots.
    pub fn new(leader: TpLeader<R>, chunk: usize, slots: usize) -> Self {
        let (batches, part_room, layers) =
            (leader.inner().prefill_batches(), leader.inner().prefill_part_rows(), leader.inner().prefill_layers());
        Self {
            leader,
            chunk: chunk.max(1),
            small: chunk.max(1),
            decode_waiting: false,
            contract: super::contract::current(),
            deadline_rows: None,
            error: None,
            fatal: None,
            digests: DigestLog::from_env(slots),
            vision: None,
            images: SlotImages::new(slots),
            spec: None,
            prefix: None,
            snaps: BTreeMap::new(),
            book: lockstep::SnapshotBook::default(),
            snaps_without_drafts: 0,
            batches,
            part_room,
            part_ms: 0.0,
            batch_rows: chunk.max(1),
            other_owed: false,
            layers,
            part_scale: 1.0,
            inflight: None,
        }
    }

    /// While a sequence decodes, run a slice longer than the room (`tp_part::PART_ROOM`) in parts of about `ms` milliseconds (`tp_part`;
    /// 0: whole slices). Only when every rank holds the stash (`PartRuntime::prefill_part_rows`); a slice then has the chunk's rows
    /// whether or not a sequence decodes.
    pub fn with_parts(mut self, ms: f64) -> Self {
        self.part_ms = if self.part_room > 0 && self.layers > 1 { ms.max(0.0) } else { 0.0 };
        self
    }

    /// While a sequence decodes, put at most `rows` rows of fresh short slices through one forward (`tp_part::BATCH_ROWS_ENV`; with
    /// nothing decoding, the chunk's).
    pub fn with_batch_rows(mut self, rows: usize) -> Self {
        self.batch_rows = rows.min(self.chunk);
        self
    }

    fn parts_on(&self) -> bool {
        self.part_ms > 0.0
    }

    /// Whether the slices of several prompts go through one forward (the runtime's `PartRuntime::prefill_batches`): the engine then
    /// serves every waiting prompt a step (`tp_boot`).
    pub fn batches(&self) -> bool {
        self.batches
    }

    /// Enable the prefix cache on every rank (as `GlmBackend::with_prefix_cache`): forwards `EnableSnapshots` of `budget.snapshot_slots`,
    /// so each rank builds its own pool of rank 0's size, and waits until every follower has built it (a boot-time call: a rank whose pool
    /// holds another slot size, or that cannot build it, fails the boot naming `enable_snapshots`). `namespace` binds every key to what
    /// the snapshots depend on (`prefix::namespace_of`). Every rank's KV pool must already hold `budget.page_blocks` pages more than
    /// admission hands out (the rank planner does that); the pools are built here, never before, so both ranks build them from this call.
    pub fn with_prefix_cache(mut self, namespace: u64, budget: CacheBudget) -> Result<Self, ModelError> {
        let n = budget.snapshot_slots;
        let slots = u32::try_from(n).map_err(|_| ModelError::Config(format!("{n} snapshot slots")))?;
        self.leader.forward_call(GlmCall::EnableSnapshots { slots }, |r, _| r.enable_snapshots(n), |bytes| *bytes)?;
        let seq = self.leader.last_seq();
        self.leader.settle(seq)?;
        self.prefix = Some(PrefixCaps { namespace, budget });
        Ok(self)
    }

    /// Snapshots taken whose restore would not draft (see the field): what a two-node gate reads beside the warm turn's drafted counters.
    pub fn snapshots_without_drafts(&self) -> u64 {
        self.snaps_without_drafts
    }

    /// [`GlmTpBackend::new`] that drafts by `cfg` (as `GlmBackend::with_spec`).
    pub fn with_spec(leader: TpLeader<R>, chunk: usize, slots: usize, cfg: SpecConfig) -> Self {
        let mut b = Self::new(leader, chunk, slots);
        b.spec = Some(SpecRuntime::new(cfg, slots));
        b
    }

    /// Slice at `small` rows while a sequence is decoding and at the chunk otherwise (`small` is clamped to `1..=chunk`; the chunk itself
    /// is one size).
    pub fn with_adaptive(mut self, small: usize) -> Self {
        self.small = small.clamp(1, self.chunk);
        self
    }

    /// Cut prompts on contract `c`'s slice grid instead of the process's (the tests that pin P1's slicing or P2's, whatever the environment).
    pub fn with_contract(mut self, c: super::contract::PrefillContract) -> Self {
        self.contract = c;
        self
    }

    /// Rows of the next slice: the small size while a sequence is decoding (unless long slices run in parts), the chunk otherwise; at most
    /// the room of a paused part while one is; under the engine's deadline limit.
    pub fn slice_rows(&self) -> usize {
        let base = if self.decode_waiting && !self.parts_on() { self.small } else { self.chunk };
        let base = if self.inflight.is_some() { base.min(self.part_room) } else { base };
        // under P2 a slice ends on its segment's 16-row KDA grid unless it ends the prompt (`prefix::slice_rows_for`)
        prefix::slice_rows_for(self.contract, base.min(self.deadline_rows.unwrap_or(usize::MAX)).max(1))
    }

    /// Serve images: rank 0's tower encodes them (the server's encoder), and the rows of each chunk travel in a `PrefillImage`.
    pub fn with_vision(mut self, v: TpVision) -> Self {
        self.vision = Some(v);
        self
    }

    /// Whether speculation may run over the `rows` positions from `first_position` of `slot` (`GlmBackend::spec_allowed_for_window`):
    /// false when any of them holds an image row. Rank 0's alone.
    pub fn spec_allowed_for_window(&self, slot: usize, first_position: usize, rows: usize) -> bool {
        self.images.allows(slot, first_position, rows)
    }

    /// Give `slot`'s state back on every rank and forget its image spans, its drafting state and its slice in flight (as
    /// `GlmBackend::drop_slot`; every rank's model forgets a paused forward of the slot as it ends it).
    fn drop_slot(&mut self, slot: usize) {
        if self.inflight.is_some_and(|f| f.slot == slot) {
            self.inflight = None;
        }
        self.leader.end_slot(slot);
        self.images.clear(slot);
        if let Some(s) = self.spec.as_mut() {
            s.end(slot);
        }
    }

    pub fn chunk(&self) -> usize {
        self.chunk
    }

    fn dead(&self) -> bool {
        self.fatal.is_some()
    }

    /// As `GlmBackend::record`; a lost group is `StateLost`, so fatal.
    fn record(&mut self, what: &str, e: &ModelError) {
        let message = format!("{what}: {e}");
        let fatal = e.is_fatal();
        eprintln!("rexl3 level={} event=backend_error {message}", if fatal { "fatal" } else { "error" });
        if fatal {
            self.fatal.get_or_insert(BackendError::fatal(message));
        } else if self.error.is_none() {
            self.error = Some(BackendError::retryable(message));
        }
    }

    fn chunk_batch(slot: usize, at: usize, tokens: &[u32]) -> RowBatch {
        RowBatch::from_requests(&[RowRequest { slot, tokens: tokens.to_vec(), first_position: at as u64 }])
    }

    /// Where the slice from `done` ends: the slice's rows, cut short at the first of `cuts` it would pass (`prefix::plan_slice` with
    /// `split_ok` false), so a snapshot at a cut is of the state between two chunks. A plan with a cut inside the chunk is refused by name:
    /// a rank of a KDA head split runs each chunk's KDA whole and would copy the state after the chunk, not at the cut.
    fn slice_end(&self, done: usize, n: usize, cuts: &[usize]) -> Result<usize, ModelError> {
        let plan = prefix::plan_slice(done, n, self.slice_rows(), cuts, false);
        if !plan.interior.is_empty() {
            return Err(ModelError::Request(format!(
                "a tensor-parallel slice {done}..{} with snapshot cuts {:?} inside it: a GLM rank snapshots only between chunks",
                plan.end, plan.interior
            )));
        }
        Ok(plan.end)
    }

    /// One prefill slice from `done` without images (as `GlmBackend::prefill_slice_at` for a model that takes no cut inside a chunk),
    /// ended at the first of `cuts` it would pass.
    fn slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        done: usize,
        candidates: usize,
        cuts: &[usize],
    ) -> Result<PrefillStep, ModelError> {
        let n = prompt.len();
        if done >= n {
            return Err(ModelError::Request(format!("a slice from {done} of a prompt of {n} tokens")));
        }
        let end = self.slice_end(done, n, cuts)?;
        self.open_slice(slot, done, n)?;
        let at = done;
        let want = (end == n).then_some(candidates);
        let t = std::time::Instant::now();
        let c = self.leader.prefill_chunk(&Self::chunk_batch(slot, at, &prompt[at..end]), want)?;
        super::spec::cost_log(format_args!("P {} 1 {} {:.4}", end - at, self.layers, t.elapsed().as_secs_f64() * 1e3));
        let row = c.map(|c| row_of(&c, 0, candidates));
        if let Some(r) = &row {
            self.digests.row(slot, n - 1, prompt[n - 1], r);
        }
        Ok(PrefillStep { done: end, row })
    }

    /// Before a slice of `slot`'s prompt of `n` tokens from `done`: a prompt's first slice begins the slot on every rank (and its drafting
    /// state and digests here); rank 0's drafter absorbs only the rows that reach its ring after the prompt.
    fn open_slice(&mut self, slot: usize, done: usize, n: usize) -> Result<(), ModelError> {
        if done == 0 {
            self.leader.begin_slot(slot)?;
            if let Some(s) = self.spec.as_mut() {
                s.begin(slot);
            }
            self.digests.begin(slot);
        }
        self.leader.inner_mut().dflash_prompt_end(slot, Some(n));
        Ok(())
    }

    /// Layers of the next part of a forward of `rows` rows, `left` layers still to run: about [`GlmTpBackend::part_ms`] of them by the
    /// modelled layer time scaled by the parts measured so far, at least one, and all that are left when the rest would be a sliver.
    fn part_layers(&self, rows: usize, left: usize) -> usize {
        let k = ((self.part_ms / (layer_ms(rows) * self.part_scale)).floor() as usize).clamp(1, left);
        if left - k < k.div_ceil(2) {
            left
        } else {
            k
        }
    }

    /// A part of `layers` layers of a forward of `rows` rows took `ms`: the model's scale follows (smoothed, bounded).
    fn part_took(&mut self, rows: usize, layers: usize, ms: f64) {
        super::spec::cost_log(format_args!("P {rows} 1 {layers} {ms:.4}"));
        let ratio = ms / (layer_ms(rows) * layers as f64);
        if ratio.is_finite() && ratio > 0.0 {
            self.part_scale = (0.7 * self.part_scale + 0.3 * ratio).clamp(0.25, 4.0);
        }
    }

    /// Whether the slice `done..end` of a text prompt runs in parts: parts are on, a sequence decodes, none runs in parts now, and the
    /// slice has more rows than may run between two parts of another (a shorter one runs whole, alone or with others: `plan_call`).
    fn wants_parts(&self, done: usize, end: usize) -> bool {
        self.parts_on() && self.decode_waiting && self.inflight.is_none() && end - done > self.part_room
    }

    /// The first part of the slice `done..end` of `slot`'s prompt (`tp_part`): the forward begins, its first layers run, it pauses.
    fn begin_part(&mut self, slot: usize, prompt: &[u32], done: usize, end: usize, candidates: usize) -> Result<PrefillStep, ModelError> {
        let n = prompt.len();
        self.open_slice(slot, done, n)?;
        let k = self.part_layers(end - done, self.layers);
        let until = if k >= self.layers { TO_END } else { k as u32 };
        let want = (end == n).then_some(candidates as u32);
        let t = std::time::Instant::now();
        let c = tp_part::forward_part(
            &mut self.leader,
            PrefillPart { begin: Some(Self::chunk_batch(slot, done, &prompt[done..end])), want, until },
        )?;
        self.part_took(end - done, k, t.elapsed().as_secs_f64() * 1e3);
        if until == TO_END {
            return Ok(self.slice_done(slot, prompt, end, candidates, c));
        }
        self.inflight = Some(Inflight { slot, n, done, end, candidates, next: k });
        Ok(PrefillStep { done, row: None })
    }

    /// The next part of the slice in flight: about a part's layers while a sequence decodes, the rest otherwise. Its last part ends the
    /// slice (`done` moves to its end, and the candidates come when it ends the prompt).
    fn next_part(&mut self, prompt: &[u32]) -> Result<PrefillStep, ModelError> {
        let f = self.inflight.expect("a slice in flight");
        let left = self.layers - f.next;
        let k = if self.decode_waiting { self.part_layers(f.end - f.done, left) } else { left };
        let until = if f.next + k >= self.layers { TO_END } else { (f.next + k) as u32 };
        let want = (f.end == f.n).then_some(f.candidates as u32);
        let t = std::time::Instant::now();
        let c = tp_part::forward_part(&mut self.leader, PrefillPart { begin: None, want, until })?;
        self.part_took(f.end - f.done, k, t.elapsed().as_secs_f64() * 1e3);
        if until == TO_END {
            self.inflight = None;
            return Ok(self.slice_done(f.slot, prompt, f.end, f.candidates, c));
        }
        self.inflight = Some(Inflight { next: f.next + k, ..f });
        Ok(PrefillStep { done: f.done, row: None })
    }

    /// A slice of `slot`'s prompt that ended at `end`: its row when it ended the prompt.
    fn slice_done(&mut self, slot: usize, prompt: &[u32], end: usize, candidates: usize, c: Option<Candidates>) -> PrefillStep {
        let n = prompt.len();
        let row = c.filter(|_| end == n).map(|c| row_of(&c, 0, candidates));
        if let Some(r) = &row {
            self.digests.row(slot, n - 1, prompt[n - 1], r);
        }
        PrefillStep { done: end, row }
    }

    /// The slices of several text prompts in ONE forward (`tp_part`; each `(job, end)`: the job's slice is `done..end`): each prompt's
    /// rows the bits of its own slice alone, its row when its slice ends its prompt.
    fn batch_slices(&mut self, jobs: &[(&PrefillJob<'_>, usize)]) -> Result<Vec<PrefillStep>, ModelError> {
        let mut reqs = Vec::with_capacity(jobs.len());
        for &(j, end) in jobs {
            self.open_slice(j.slot, j.done, j.prompt.len())?;
            reqs.push(RowRequest { slot: j.slot, tokens: j.prompt[j.done..end].to_vec(), first_position: j.done as u64 });
        }
        let want = jobs.iter().filter(|(j, end)| *end == j.prompt.len()).map(|(j, _)| j.candidates as u32).max();
        let b = RowBatch::from_requests(&reqs);
        let t = std::time::Instant::now();
        let rows = b.rows;
        let c = tp_part::forward_part(&mut self.leader, PrefillPart { begin: Some(b), want, until: TO_END })?;
        super::spec::cost_log(format_args!("P {rows} {} {} {:.4}", jobs.len(), self.layers, t.elapsed().as_secs_f64() * 1e3));
        Ok(jobs
            .iter()
            .enumerate()
            .map(|(i, &(j, end))| {
                let n = j.prompt.len();
                let row = c.as_ref().filter(|_| end == n).map(|c| row_of(c, i, j.candidates));
                if let Some(r) = &row {
                    self.digests.row(j.slot, n - 1, j.prompt[n - 1], r);
                }
                PrefillStep { done: end, row }
            })
            .collect())
    }

    /// The jobs of one `prefill_slices` call (in the engine's order): the fresh short text slices that go through one forward together,
    /// as many as fit its rows (each `(job, end)`; a slice that fits alone but not with these waits for the next call), and the first job
    /// that goes alone: the slice in parts, a slice that runs in parts (`wants_parts`) or would not fit even alone, one with images. The
    /// forward's rows: a paused part's room while one is paused (and then only prompts that end in it, as `slice_at` runs meanwhile),
    /// the batch rows while a sequence decodes, the chunk otherwise.
    fn plan_call(&self, jobs: &[PrefillJob<'_>]) -> (Vec<(usize, usize)>, Option<usize>) {
        let paused = self.inflight.is_some();
        let room = if paused {
            self.part_room
        } else if self.decode_waiting {
            self.batch_rows
        } else {
            self.chunk
        };
        let (mut group, mut other, mut rows) = (Vec::new(), None, 0usize);
        for (i, j) in jobs.iter().enumerate() {
            let n = j.prompt.len();
            let fresh = j.images.is_empty() && j.done < n && !self.inflight.is_some_and(|f| f.slot == j.slot);
            let end = if fresh { self.slice_end(j.done, n, &[]).ok() } else { None };
            match end.filter(|&end| end - j.done <= room && !self.wants_parts(j.done, end) && (!paused || end == n)) {
                Some(end) if rows + (end - j.done) <= room => {
                    rows += end - j.done;
                    group.push((i, end));
                }
                Some(_) => {}
                None => {
                    other.get_or_insert(i);
                }
            }
        }
        (group, other)
    }

    /// One prefill slice of a prompt with images (as `GlmBackend::prefill_slice_at` for a model that takes no cut inside a chunk), ended
    /// at the first of `cuts` it would pass: the images are checked before anything is forwarded; a chunk with image rows is a
    /// `PrefillImage` (ended earlier if it would not fit one frame), any other chunk the core prefill.
    fn image_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[gb_engine::ImageSpan],
        done: usize,
        candidates: usize,
        cuts: &[usize],
    ) -> Result<PrefillStep, ModelError> {
        let n = prompt.len();
        let v = self.vision.as_ref().ok_or_else(|| {
            ModelError::Request("this GLM tensor-parallel group serves text only (rank 0 has no vision tower)".into())
        })?;
        if done >= n {
            return Err(ModelError::Request(format!("a slice from {done} of a prompt of {n} tokens")));
        }
        v.check_images(prompt, images)?;
        let planned = self.slice_end(done, n, cuts)?;
        // the rows are read off the device before anything of the slot is touched: a failure here forwards nothing
        let (call, end) = v.chunk_call(slot, prompt, images, done, planned, candidates)?;
        self.open_slice(slot, done, n)?;
        // set on every slice, as the single node does
        self.images.set(slot, images);
        let c = match call {
            ChunkCall::Text(b, want) => self.leader.prefill_chunk(&b, want)?,
            ChunkCall::Image(p) => tp_vision::forward_image(&mut self.leader, p)?,
        };
        let row = c.map(|c| row_of(&c, 0, candidates));
        if let Some(r) = &row {
            self.digests.row(slot, n - 1, prompt[n - 1], r);
        }
        Ok(PrefillStep { done: end, row })
    }

    /// One prefill slice of `prompt` from `done`, with a snapshot of the slot's state when the slice ends at one of `cuts` (as
    /// `GlmBackend::prefill_slice_at` for a model that takes no cut inside a chunk). With no cuts this is exactly the slice
    /// `prefill_slice` has always run, except that while a sequence decodes a long text slice runs in parts (`tp_part`: each call one
    /// part, `done` unchanged until the last), and while one runs in parts another prompt's slice runs only when it fits the paused
    /// forward's room (otherwise nothing runs: `done` unchanged).
    fn slice_at(
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
        let text = images.is_empty() && cuts.is_empty() && done < n;
        let part = match self.inflight {
            Some(f) if f.slot == slot => Some(None),
            Some(_) if !(text && n - done <= self.part_room) => {
                return CutStep { step: PrefillStep { done, row: None }, captured: Vec::new() };
            }
            Some(_) => None,
            None if text => self.slice_end(done, n, &[]).ok().filter(|&end| self.wants_parts(done, end)).map(Some),
            None => None,
        };
        if let Some(begin) = part {
            let r = match begin {
                None => self.next_part(prompt),
                Some(end) => self.begin_part(slot, prompt, done, end, candidates),
            };
            return match r {
                Ok(step) => CutStep { step, captured: Vec::new() },
                Err(e) => {
                    self.record(&format!("prefill of slot {slot}"), &e);
                    self.drop_slot(slot);
                    CutStep { step: PrefillStep { done: n, row: Some(nan_row(candidates)) }, captured: Vec::new() }
                }
            };
        }
        // the snapshots this slice took, handed to the engine if it does not fail and given back (on every rank) if it does
        let mut captured: Vec<Captured> = Vec::new();
        match self.cut_slice(slot, prompt, images, done, candidates, cuts, &mut captured) {
            Ok(step) => CutStep { step, captured },
            Err(e) => {
                for c in captured {
                    self.prefix_release(c.handle);
                }
                self.record(&format!("prefill of slot {slot}"), &e);
                self.drop_slot(slot);
                CutStep { step: PrefillStep { done: n, row: Some(nan_row(candidates)) }, captured: Vec::new() }
            }
        }
    }

    /// The body of [`GlmTpBackend::slice_at`]. The engine's cuts move to the contract's grid (`prefix::cuts_for`: under P2 the slot's
    /// segment's 16-row KDA grid, never inside an image), as many as there are free snapshot slots; the slice ENDS at the first one it
    /// would pass, and the snapshot is a `Snapshot` call after its chunk.
    #[allow(clippy::too_many_arguments)]
    fn cut_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
        cuts: &[usize],
        captured: &mut Vec<Captured>,
    ) -> Result<PrefillStep, ModelError> {
        let n = prompt.len();
        let cuts = if cuts.is_empty() {
            Vec::new()
        } else {
            let free = self.leader.inner().free_snapshots();
            let anchor = if done == 0 { 0 } else { self.leader.inner().p2_anchor(slot).unwrap_or(done) };
            let inside = |x: usize| images.iter().any(|im| x > im.start && x < im.start + im.tokens);
            let mut cuts = prefix::cuts_for(self.contract, anchor, done, cuts, &inside);
            cuts.truncate(free);
            cuts
        };
        let step = if images.is_empty() {
            self.slice(slot, prompt, done, candidates, &cuts)?
        } else if self.vision.is_none() {
            return Err(ModelError::Request(
                "a GLM tensor-parallel group serves text only (start rank 0 with --no-vision)".into(),
            ));
        } else {
            self.image_slice(slot, prompt, images, done, candidates, &cuts)?
        };
        let end = step.done;
        if end < n && cuts.contains(&end) {
            if let Some(t) = self.snapshot(slot, end)? {
                captured.push(Captured { len: end, handle: PrefixHandle { snap: t.snap, pages: t.pages } });
            }
        }
        Ok(step)
    }

    /// Rank 0's half of a [`GlmCall::Snapshot`] of `slot` after `len` tokens: forwarded first, then taken here by the steps every follower
    /// takes (`tp::take_snapshot`), and this rank's handle remembered by the call's sequence number. A cache call: rank 0 waits for every
    /// rank's answer and records every rank's content digest. `None`: the snapshot is not held by every rank (a full pool, a rank's error,
    /// no answer in time); whatever any rank took of it is released on every rank, and the engine never sees it.
    fn snapshot(&mut self, slot: usize, len: usize) -> Result<Option<Taken>, ModelError> {
        let call = GlmCall::Snapshot {
            slot: slot as u32,
            len: u32::try_from(len).map_err(|_| ModelError::Request(format!("a snapshot of {len} tokens")))?,
        };
        let out = self.leader.forward_cache_call(call, |r, _| tp::take_snapshot(r, slot, len), true)?;
        let seq = out.seq;
        let (rank0, taken) = match out.local {
            Ok(Some(t)) => (lockstep::taken_digest(t.digest), Some(t)),
            Ok(None) => (lockstep::SNAPSHOT_SKIPPED, None),
            Err(e) => {
                eprintln!("rexl3 level=warn event=tp_glm_snapshot_failed rank=0 call={seq} slot={slot} len={len} error=\"{e}\"");
                (lockstep::SNAPSHOT_SKIPPED, None)
            }
        };
        // the handle every rank knows, kept until the release whatever the outcome
        if let Some(t) = &taken {
            self.snaps.insert(t.snap, (seq, t.pages.clone()));
        }
        if let Err(why) = self.book.settle_snapshot(seq, rank0, &out.followers) {
            let all_skipped =
                taken.is_none() && out.followers.iter().all(|a| *a == lockstep::RankAnswer::Ok(lockstep::SNAPSHOT_SKIPPED));
            if !all_skipped {
                eprintln!("rexl3 level=warn event=tp_glm_snapshot_refused call={seq} slot={slot} len={len} reason=\"{why}\"");
            }
            match taken {
                Some(t) => self.prefix_release(PrefixHandle { snap: t.snap, pages: t.pages }),
                // rank 0 holds none: free whatever a follower took of it
                None if !all_skipped => {
                    let _ = self.leader.forward_cache_call(GlmCall::Release { of: seq }, |_, _| Ok(()), false);
                }
                None => {}
            }
            return Ok(None);
        }
        let Some(t) = taken else { return Ok(None) };
        // the book is built chunk by chunk up to the slice's end, so a snapshot there carries a sound drafter: one that does not would cost
        // every warm turn from it its drafts (exactness is unaffected), so it is counted and said once
        if !t.mtp_ok {
            self.snaps_without_drafts += 1;
            if self.snaps_without_drafts == 1 {
                eprintln!(
                    "rexl3 level=warn event=tp_glm_snapshot_without_drafts slot={slot} len={len}: a restore of it will not draft (the \
                     drafter's book was not sound at the slice's end)"
                );
            }
        }
        Ok(Some(t))
    }

    /// Rank 0's handle as the followers know it: the sequence number of the call that made it, when this rank made it and still holds it
    /// exactly as it was made.
    fn forwarded(&self, handle: &PrefixHandle) -> Result<u64, ModelError> {
        match self.snaps.get(&handle.snap) {
            Some((of, pages)) if *pages == handle.pages => Ok(*of),
            _ => Err(ModelError::Request(format!(
                "snapshot {} with {} pages is not a handle rank 0 made and still holds (never forwarded to the followers)",
                handle.snap,
                handle.pages.len()
            ))),
        }
    }

    /// Rank 0's half of a [`GlmCall::Restore`]: every follower is named the call that made the snapshot, never this rank's slot or page
    /// ids. A handle this rank did not make, or no longer holds, or that not every rank holds, is refused by name before anything is
    /// touched or forwarded (a restore here alone would start the slot over on this rank only). A cache call: rank 0 waits for every
    /// rank's answer before the suffix can be dispatched, and the restore is used only when every rank's digest of the state it restored
    /// equals its own digest of the snapshot. Otherwise it is a request error: the engine drops the entry and prefills cold, which begins
    /// the slot again on every rank.
    fn restore(&mut self, slot: usize, handle: &PrefixHandle, len: usize) -> Result<(), ModelError> {
        let of = self.forwarded(handle)?;
        if !self.book.holds(of) {
            return Err(ModelError::Request(format!("the snapshot of call {of} is not held by every rank (never restored)")));
        }
        let len32 = u32::try_from(len).map_err(|_| ModelError::Request(format!("a restore of {len} tokens")))?;
        let (snap, pages) = (handle.snap, &handle.pages);
        let out = self.leader.forward_cache_call(
            GlmCall::Restore { slot: slot as u32, len: len32, of },
            |r, _| {
                r.restore_prefix(slot, snap, pages, len)?;
                r.prefix_state_digest(slot, len).map(lockstep::taken_digest)
            },
            true,
        )?;
        let rank0 = out.local.map_err(|e| e.to_string());
        self.book.verify_restore(of, rank0, &out.followers).map_err(|why| {
            eprintln!(
                "rexl3 level=warn event=tp_glm_restore_refused call={} of={of} slot={slot} len={len} reason=\"{why}\"",
                out.seq
            );
            ModelError::Request(format!(
                "the prefix restore of call {} (snapshot call {of}) was not verified on every rank: {why}",
                out.seq
            ))
        })
    }
}

/// Rank 0's window graph counters when its backend ends (the follower prints its own as it exits: `tp_glm_graphs`), so a two-node leg
/// records whether the graphs ran on both ranks.
impl<R: GlmMtpRuntime + ImageRuntime + PrefixRuntime + PartRuntime> Drop for GlmTpBackend<R> {
    fn drop(&mut self) {
        if let Some(g) = self.leader.inner().graph_counters() {
            eprintln!(
                "rexl3 event=tp_glm_graphs rank=0 graph_replays={} graph_captures={} graph_eager={} graph_capture_failures={} {}",
                g.replays,
                g.captures,
                g.eager_windows,
                g.capture_failures,
                super::tp::rank_draft_graphs_line()
            );
            eprintln!("rexl3 event=tp_roce_paths rank=0 {}", super::tp_boot::roce_paths_line());
        }
    }
}

impl<R: GlmMtpRuntime + ImageRuntime + PrefixRuntime + PartRuntime> Backend for GlmTpBackend<R> {
    fn set_prefill_row_limit(&mut self, rows: Option<usize>) -> bool {
        self.deadline_rows = rows;
        true
    }

    fn set_decode_waiting(&mut self, waiting: bool) {
        self.decode_waiting = waiting;
    }

    /// The slice running in parts (`tp_part`), and the rows another prompt's slice may have meanwhile (the ranks' stash).
    fn prefill_in_flight(&self) -> Option<(usize, usize)> {
        self.inflight.map(|f| (f.slot, self.part_room))
    }

    /// The fresh short text slices of several waiting prompts go through ONE forward (`tp_part`, P1 and P2: each prompt's rows the bits of
    /// its own slice alone; [`GlmTpBackend::plan_call`] picks them). A call runs ONE forward: that group, or the first job that goes alone
    /// (the slice in parts, a long slice, one with images), the two taking turns while both wait; every job it does not run is not served
    /// (`done` unchanged and no row: the engine asks again at its next step, after a decode round). So the decoding streams wait for one
    /// forward a step, and the prompts of a group get their first tokens together.
    fn prefill_slices(&mut self, jobs: &[PrefillJob<'_>]) -> Vec<PrefillStep> {
        if !self.batches || self.dead() || jobs.len() < 2 {
            return jobs.iter().map(|j| self.prefill_slice(j.slot, j.prompt, j.images, j.done, j.candidates)).collect();
        }
        let (group, other) = self.plan_call(jobs);
        let mut out: Vec<PrefillStep> = jobs.iter().map(|j| PrefillStep { done: j.done, row: None }).collect();
        let alone = match (other, &group[..]) {
            (Some(i), _) if group.is_empty() || self.other_owed => Some(i),
            (_, &[(i, _)]) => Some(i),
            _ => None,
        };
        self.other_owed = other.is_some() && alone != other;
        if let Some(i) = alone {
            let j = &jobs[i];
            out[i] = self.prefill_slice(j.slot, j.prompt, j.images, j.done, j.candidates);
            return out;
        }
        let batch: Vec<(&PrefillJob<'_>, usize)> = group.iter().map(|&(i, end)| (&jobs[i], end)).collect();
        match self.batch_slices(&batch) {
            Ok(steps) => {
                for (&(i, _), s) in group.iter().zip(steps) {
                    out[i] = s;
                }
            }
            Err(e) => {
                self.record(&format!("prefill of {} prompts in one forward", group.len()), &e);
                for &(i, _) in &group {
                    self.drop_slot(jobs[i].slot);
                    out[i] = PrefillStep { done: jobs[i].prompt.len(), row: Some(nan_row(jobs[i].candidates)) };
                }
            }
        }
        out
    }

    /// Rank 0's own graph cache (`/metrics`); a follower prints its own (`tp_follower_stats`, `tp_glm_graphs`).
    fn graph_counters(&self) -> Option<gb_engine::GraphCounters> {
        self.leader.inner().graph_counters()
    }

    /// Rank 0's device idle and every follower caught up with the calls before this one (a barrier).
    fn quiesce(&mut self) -> Result<(), String> {
        if gb_cuda::device_available() {
            gb_cuda::rt::device_sync().map_err(|e| e.to_string())?;
        }
        self.leader.barrier().map_err(|e| e.to_string())
    }

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
        let chunk = self.chunk;
        let mut run = || -> Result<RowOutput, ModelError> {
            if start == 0 {
                self.leader.begin_slot(slot)?;
                if let Some(s) = self.spec.as_mut() {
                    s.begin(slot);
                }
                self.digests.begin(slot);
            }
            self.leader.inner_mut().dflash_prompt_end(slot, Some(total.max(start as usize + tokens.len())));
            let (mut pos, n) = (start as usize, tokens.len());
            let mut at = 0;
            let mut last = None;
            while at < n {
                let end = (at + chunk).min(n);
                let want = (end == n).then_some(candidates);
                last = self.leader.prefill_chunk(&Self::chunk_batch(slot, pos, &tokens[at..end]), want)?;
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
        self.slice_at(slot, prompt, images, done, candidates, &[]).step
    }

    fn prefix_caps(&self) -> Option<PrefixCaps> {
        self.prefix
    }

    /// One path id for every total, as `GlmBackend`'s: GLM has one arithmetic, on every rank.
    fn prefix_path(&self, _total: usize) -> Option<u32> {
        self.prefix.as_ref().map(|_| prefix::PATH_ID)
    }

    /// Rank 0's free snapshot slots: every follower's pool moves by the same calls, so it has as many.
    fn prefix_free_snapshots(&self) -> usize {
        self.leader.inner().free_snapshots()
    }

    /// As `GlmBackend::prefix_restore`, with the restore forwarded ([`GlmCall::Restore`]): the sequence starts over from the entry on
    /// every rank, its drafting state is new (each model reset the drafter's book and set the carry), its image spans are set by its next
    /// slice.
    fn prefix_restore(&mut self, slot: usize, handle: &PrefixHandle, len: usize) -> Result<(), String> {
        if self.dead() {
            return Err("the backend has failed".into());
        }
        match self.restore(slot, handle, len) {
            Ok(()) => {
                self.images.clear(slot);
                if let Some(s) = self.spec.as_mut() {
                    s.begin(slot);
                }
                self.digests.begin(slot);
                Ok(())
            }
            Err(e) => {
                self.record(&format!("prefix restore into slot {slot}"), &e);
                self.drop_slot(slot);
                Err(e.to_string())
            }
        }
    }

    /// As `GlmBackend::prefix_release`, forwarded ([`GlmCall::Release`]): every rank gives its own snapshot slot and page references back,
    /// at rank 0's word (a follower never evicts on its own). A handle rank 0 did not make is never released anywhere (a release of a
    /// slot twice is a panic of the pool), only reported.
    fn prefix_release(&mut self, handle: PrefixHandle) {
        let of = match self.forwarded(&handle) {
            Ok(of) => of,
            Err(e) => {
                eprintln!("rexl3 level=error event=backend_error prefix release: {e}; nothing released");
                return;
            }
        };
        self.snaps.remove(&handle.snap);
        self.book.forget(of);
        // a cache call, not waited for: a rank that holds no such snapshot answers Ok, and one that never answers is a stall
        let r = self.leader.forward_cache_call(
            GlmCall::Release { of },
            |r, _| {
                r.release_prefix(handle.snap, &handle.pages);
                Ok(())
            },
            false,
        );
        if let Err(e) = r {
            self.record(&format!("prefix release of snapshot {}", handle.snap), &e);
        }
    }

    /// [`Backend::prefill_slice`] that takes a snapshot at each of `cuts` the slice reaches, by ending the slice there (`slice_at`): the
    /// bits of every row do not depend on the slice grid.
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
        self.slice_at(slot, prompt, images, done, candidates, cuts)
    }

    /// A prompt with images from its start, in one call (as `GlmBackend::prefill_mm`; the engine uses `prefill_slice`).
    fn prefill_mm(&mut self, slot: usize, start: u64, tokens: &[u32], images: &[ImageSpan], candidates: usize) -> RowOutput {
        if images.is_empty() {
            return self.prefill(slot, start, tokens, candidates);
        }
        if self.vision.is_none() || start != 0 {
            let e = ModelError::Request(if self.vision.is_none() {
                "a GLM tensor-parallel group serves text only (start rank 0 with --no-vision)".into()
            } else {
                "a prompt with images must be prefilled from its start".into()
            });
            self.record(&format!("prefill of slot {slot}"), &e);
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
        match self.leader.window_candidates(&b, candidates) {
            Ok(c) => {
                if let Some(s) = self.spec.as_mut() {
                    s.on_window_of(reqs.len(), b.rows, started.elapsed().as_secs_f64() * 1e3);
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
                self.record("window step", &e);
                nans()
            }
        }
    }

    fn commit(&mut self, slot: usize, keep: usize) {
        self.leader.commit(slot, keep);
        if let Some(s) = self.spec.as_mut() {
            s.on_commit(slot, keep);
        }
    }

    fn release(&mut self, slot: usize) {
        self.drop_slot(slot);
    }

    /// As `GlmBackend::propose_batch` over the leader, with the slots' image spans (a draft window over an image row stays host-only on
    /// rank 0: `spec_allowed_for_window`). The draft call is forwarded (`tp_mtp`); a lost group is fatal, an error both ranks met alike
    /// costs the round its native drafts only.
    fn propose_batch(&mut self, reqs: &[ProposeRequest]) -> Vec<Option<Vec<u32>>> {
        if self.dead() || self.spec.is_none() {
            return vec![None; reqs.len()];
        }
        let GlmTpBackend { leader, spec, images, .. } = self;
        match spec.as_mut().unwrap().propose(leader, reqs, images.spans()) {
            Ok(v) => v,
            Err(e) => {
                self.record(&format!("drafting for {} sequences", reqs.len()), &e);
                vec![None; reqs.len()]
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    //! The family's real backend over the leader on a RECORDING MOCK runtime (checklist 3.1, item 5): the follower replays rank 0's call
    //! stream exactly, the digests agree, and the stream is the single-node backend's model calls.
    use super::*;
    use crate::families::glm::tp::{GlmCall, GlmFacts};
    use crate::lockstep::{
        run_follower, BootGate, CallReply, FollowerConfig, FollowerExit, Link, LockstepRuntime, Msg, TpTimeouts,
    };
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    pub(crate) type Log = Arc<Mutex<Vec<String>>>;

    /// The mock's snapshot pool: its free slots, and each held slot's (length, drafter flag).
    pub(crate) type MockPool = (Vec<u32>, BTreeMap<u32, (usize, bool)>);

    /// What a rank's prefix cache does wrong (shared with the test, so it can change between requests).
    #[derive(Default)]
    pub(crate) struct RecFaults {
        /// The restore: 0 right, 1 an error, 2 the slot starts over and nothing is restored (Ok), 3 the slot is left as it was (Ok).
        pub(crate) restore: std::sync::atomic::AtomicU8,
        /// The pool has no free slot for a snapshot (this rank skips it).
        pub(crate) skip: std::sync::atomic::AtomicBool,
        /// The restore sleeps this long first.
        pub(crate) restore_sleep_ms: std::sync::atomic::AtomicU64,
    }

    /// A runtime that records every call and answers candidates that are a pure function of the batch (as a model's are). It keeps a
    /// real `TapBook` per slot, moved by the calls as the model moves it (a prefill chunk, a window's commit, an absorb), so that the
    /// speculation runtime sees slots that can draft, and drafts that are a pure function of the ask. For the prefix cache it keeps a
    /// snapshot pool, each slot's KV pages and P2 segment as the model keeps them (`model/snapshot.rs`), and a follower's handles.
    pub(crate) struct Rec {
        pub(crate) log: Log,
        /// Fail the next window (an asymmetric error when only one rank has it).
        pub(crate) fail_window: bool,
        pub(crate) books: Vec<crate::families::glm::model::TapBook>,
        /// The last window, for its commits.
        pub(crate) last: Option<RowBatch>,
        /// Draft with this rank's own bias (a rank that drafts differently).
        pub(crate) draft_bias: u32,
        /// The snapshot pool (`enable_snapshots`).
        pub(crate) pool: Option<MockPool>,
        /// This rank's numbering of its snapshot slots (`snap_base..`): a follower's slot ids differ from rank 0's, as nothing makes them agree.
        pub(crate) snap_base: u32,
        /// The bytes of one snapshot slot this rank's pool reports.
        pub(crate) snap_bytes: u64,
        /// Per slot, the KV pages of its positions (one per `BLOCK_TOKENS`, numbered by a counter the call stream moves alike on every
        /// rank, as the model's page allocator), and the first position of its open P2 segment.
        pub(crate) kv: Vec<Vec<u32>>,
        pub(crate) next_page: u32,
        pub(crate) anchor: Vec<Option<usize>>,
        pub(crate) handles: crate::families::glm::tp::SnapHandles,
        /// Per slot, a hash chain over the (position, token) pairs it holds: the slot's "state", which a snapshot copies and a restore
        /// writes back, and a content digest reads.
        pub(crate) chain: Vec<u64>,
        pub(crate) snap_chain: BTreeMap<u32, u64>,
        pub(crate) faults: Arc<RecFaults>,
        /// Prompt forwards of several prompts and in parts (`tp_part`): `(room rows, layers)`, `None`: the runtime runs neither.
        pub(crate) parts: Option<(usize, usize)>,
        /// The paused prompt forward and its `want`.
        pub(crate) paused: Option<(RowBatch, Option<u32>)>,
    }

    impl Rec {
        pub(crate) fn new(log: Log) -> Rec {
            Rec {
                log,
                fail_window: false,
                books: vec![Default::default(); 4],
                last: None,
                draft_bias: 0,
                pool: None,
                snap_base: 0,
                snap_bytes: 4096,
                kv: vec![Vec::new(); 4],
                next_page: 0,
                anchor: vec![None; 4],
                handles: Default::default(),
                chain: vec![0; 4],
                snap_chain: BTreeMap::new(),
                faults: Arc::default(),
                parts: None,
                paused: None,
            }
        }

        /// The slot's chain over `tokens` from `start` (a slot starts over at 0).
        fn eat(&mut self, slot: usize, start: usize, tokens: &[u32]) {
            let mut h = if start == 0 { 0 } else { self.chain[slot] };
            for (i, &t) in tokens.iter().enumerate() {
                h = gb_cuda::state_digest::mix64(h ^ ((t as u64) << 24) ^ ((start + i) as u64));
            }
            self.chain[slot] = h;
        }

        /// The mock's content digest: the state chain, the length, the drafter flag and the full pages.
        fn digest_of(len: usize, mtp_ok: bool, chain: u64, pages: &[u32]) -> u64 {
            let full = (len / crate::kvtables::BLOCK_TOKENS).min(pages.len());
            let mut b = Vec::new();
            b.extend_from_slice(&(len as u64).to_le_bytes());
            b.push(u8::from(mtp_ok));
            b.extend_from_slice(&chain.to_le_bytes());
            for p in &pages[..full] {
                b.extend_from_slice(&p.to_le_bytes());
            }
            gb_cuda::state_digest::digest_host(&[&b])
        }

        /// The slot's pages cover its first `n` positions.
        fn cover(&mut self, slot: usize, n: usize) {
            while self.kv[slot].len() < n.div_ceil(crate::kvtables::BLOCK_TOKENS) {
                self.kv[slot].push(self.next_page);
                self.next_page += 1;
            }
        }

        /// The slot starts over: its pages given back, its segment closed, its drafter's book reset.
        fn fresh(&mut self, slot: usize) {
            self.kv[slot].clear();
            self.anchor[slot] = None;
            self.books[slot].reset();
            self.chain[slot] = 0;
        }
    }

    fn cands(b: &RowBatch, rows: usize, count: usize) -> Candidates {
        cands_of_rows(b, &(b.rows - rows..b.rows).collect::<Vec<_>>(), count)
    }

    /// The mock's candidates of rows `src` of `b`: a pure function of each row's token and position.
    fn cands_of_rows(b: &RowBatch, src: &[usize], count: usize) -> Candidates {
        let mut ids = Vec::new();
        let mut vals = Vec::new();
        for &r in src {
            for k in 0..count {
                ids.push((b.ids[r] * 31 + b.positions[r] * 7 + k as i32) % 1000);
                vals.push(-(k as f32) - b.positions[r] as f32 * 0.5);
            }
        }
        Candidates { rows: src.len(), count, ids, vals }
    }

    /// [`cands`] of the last row of each sequence of `b`, in sequence order (the model's head of a forward of several prompts).
    fn seq_cands(b: &RowBatch, count: usize) -> Candidates {
        cands_of_rows(b, &(0..b.n_seq()).map(|i| b.seq_rows(i).end - 1).collect::<Vec<_>>(), count)
    }

    /// As `GlmModel::prefill_rows` / `prefill_resume`: a forward of one or several prompts, paused before `until` when that is short of
    /// the layers; its prompts' rows join their slots (books, pages, segments, chains) when it ends.
    impl crate::families::glm::tp_part::PartRuntime for Rec {
        fn prefill_batches(&self) -> bool {
            self.parts.is_some()
        }
        fn prefill_part_rows(&self) -> usize {
            self.parts.map_or(0, |p| p.0)
        }
        fn prefill_layers(&self) -> usize {
            self.parts.map_or(0, |p| p.1)
        }
        fn prefill_part(&mut self, call: &PrefillPart) -> Result<Option<Candidates>, ModelError> {
            let (room, layers) = self.parts.ok_or_else(|| ModelError::NotBuilt("parts".into()))?;
            call.check(4)?;
            let ends = call.until == TO_END || call.until as usize >= layers;
            let (b, want) = match &call.begin {
                Some(b) => {
                    let seqs: Vec<String> =
                        (0..b.n_seq()).map(|i| format!("{}@{}+{}", b.seq_slot[i], b.positions[b.seq_rows(i).start], b.seq_rows(i).len())).collect();
                    self.log.lock().unwrap().push(format!("part [{}] {:?} until={}", seqs.join(" "), call.want, call.until));
                    if self.paused.is_some() && (!ends || b.rows > room) {
                        return Err(ModelError::Request("a forward too big for the paused one's room".into()));
                    }
                    (b.clone(), call.want)
                }
                None => {
                    self.log.lock().unwrap().push(format!("resume until={}", call.until));
                    self.paused.take().ok_or_else(|| ModelError::Request("nothing paused".into()))?
                }
            };
            if !ends {
                self.paused = Some((b, want));
                return Ok(None);
            }
            for i in 0..b.n_seq() {
                let r = b.seq_rows(i);
                let (slot, start) = (b.seq_slot[i] as usize, b.positions[r.start] as usize);
                let tokens: Vec<u32> = b.ids[r].iter().map(|&t| t as u32).collect();
                let _ = self.books[slot].prefill_chunk(start, tokens.len());
                self.cover(slot, start + tokens.len());
                self.anchor[slot].get_or_insert(start);
                self.eat(slot, start, &tokens);
            }
            Ok(want.map(|w| seq_cands(&b, w as usize)))
        }
    }

    impl LockstepRuntime for Rec {
        type Call = GlmCall;
        fn vocab_size(&self) -> usize {
            TOKENIZER_VOCAB
        }
        fn has_mtp(&self) -> bool {
            true
        }
        fn snapshots_supported() -> bool {
            true
        }
        fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
            self.log.lock().unwrap().push(format!("begin {slot}"));
            self.fresh(slot);
            Ok(())
        }
        fn end_slot(&mut self, slot: usize) {
            self.log.lock().unwrap().push(format!("end {slot}"));
            self.kv[slot].clear();
            self.anchor[slot] = None;
        }
        fn prefill_chunk(&mut self, b: &RowBatch, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
            let (slot, start, tokens) = crate::families::glm::tp::chunk_of(b)?;
            self.log.lock().unwrap().push(format!("prefill {slot} {start} {tokens:?} {want:?}"));
            let _ = self.books[slot].prefill_chunk(start, tokens.len());
            self.cover(slot, start + tokens.len());
            self.anchor[slot].get_or_insert(start);
            self.eat(slot, start, &tokens);
            Ok(want.map(|w| cands(b, 1, w)))
        }
        fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
            self.log.lock().unwrap().push(format!("window {:?} {:?} {:?} {count}", b.ids, b.positions, b.seq_slot));
            if std::mem::take(&mut self.fail_window) {
                return Err(ModelError::Request("injected".into()));
            }
            self.last = Some(b.clone());
            Ok(cands(b, b.rows, count))
        }
        fn commit(&mut self, slot: usize, keep: usize) {
            self.log.lock().unwrap().push(format!("commit {slot} {keep}"));
            if let Some(b) = &self.last {
                if let Some(i) = b.seq_slot.iter().position(|&s| s as usize == slot) {
                    let row0 = b.positions[b.seq_rows(i).start] as usize;
                    let _ = self.books[slot].window_commit(row0, keep, crate::families::glm::alloc::MTP_TAP_CAP);
                }
            }
        }
        fn flush(&mut self) -> Result<(), ModelError> {
            self.log.lock().unwrap().push("flush".into());
            Ok(())
        }
        fn apply_call(&mut self, seq: u64, call: GlmCall) -> Result<CallReply, ModelError> {
            if call.is_prefix() {
                return crate::families::glm::tp::apply_prefix(self, seq, call);
            }
            crate::families::glm::tp::apply_family(self, call)
        }
    }

    /// The model's prefix-cache calls on the mock's books, logged without the rank-local snapshot ids (the two ranks' logs compare equal
    /// only if they did the same things to the same pages and lengths).
    impl crate::families::glm::tp::PrefixRuntime for Rec {
        fn enable_snapshots(&mut self, slots: usize) -> Result<u64, ModelError> {
            if self.pool.is_some() {
                return Err(ModelError::Config("the prefix cache is already enabled".into()));
            }
            self.log.lock().unwrap().push(format!("enable_snapshots {slots}"));
            self.pool = Some(((0..slots as u32).rev().map(|i| self.snap_base + i).collect(), BTreeMap::new()));
            Ok(self.snap_bytes)
        }
        fn free_snapshots(&self) -> usize {
            self.pool.as_ref().map_or(0, |p| p.0.len())
        }
        fn p2_anchor(&self, slot: usize) -> Option<usize> {
            self.anchor[slot]
        }
        fn alloc_snapshot(&mut self) -> Option<u32> {
            if self.faults.skip.load(std::sync::atomic::Ordering::SeqCst) {
                return None;
            }
            self.pool.as_mut().and_then(|p| p.0.pop())
        }
        /// As `GlmModel::prefix_snapshot`: the drafter's book is sound when it was built up to exactly `len` with the carry held.
        fn prefix_snapshot(&mut self, slot: usize, snap: u32, len: usize) -> Result<bool, ModelError> {
            let b = self.books[slot].clone();
            let mtp_ok = b.ok && b.first == len && b.n == 1;
            self.pool
                .as_mut()
                .ok_or_else(|| ModelError::Request("the prefix cache is not enabled".into()))?
                .1
                .insert(snap, (len, mtp_ok));
            self.snap_chain.insert(snap, self.chain[slot]);
            self.log.lock().unwrap().push(format!("snapshot {slot} {len} mtp_ok={mtp_ok}"));
            Ok(mtp_ok)
        }
        fn retain_prefix_pages(&mut self, slot: usize, len: usize) -> Vec<u32> {
            let pages = self.kv[slot][..len.div_ceil(crate::kvtables::BLOCK_TOKENS)].to_vec();
            self.log.lock().unwrap().push(format!("retain {slot} {len} {pages:?}"));
            pages
        }
        /// As `GlmModel::restore_prefix`: refused unless `snap` holds a snapshot of `len` tokens over that many pages; the slot starts
        /// over with the full pages shared, a page of its own for a partial last one, the drafter's book from the carry and a P2 segment
        /// opened at `len`.
        fn restore_prefix(&mut self, slot: usize, snap: u32, pages: &[u32], len: usize) -> Result<(), ModelError> {
            use std::sync::atomic::Ordering::SeqCst;
            // re-read every tick: a test releases a stuck rank by setting it to 0
            let t0 = std::time::Instant::now();
            while t0.elapsed() < Duration::from_millis(self.faults.restore_sleep_ms.load(SeqCst)) {
                std::thread::sleep(Duration::from_millis(5));
            }
            match self.faults.restore.load(SeqCst) {
                1 => return Err(ModelError::Request("injected restore failure".into())),
                2 => {
                    self.fresh(slot);
                    return Ok(());
                }
                3 => return Ok(()),
                _ => {}
            }
            let held = self.pool.as_ref().and_then(|p| p.1.get(&snap).copied());
            let Some((_, mtp_ok)) = held.filter(|&(l, _)| l == len && pages.len() == len.div_ceil(crate::kvtables::BLOCK_TOKENS))
            else {
                return Err(ModelError::Request(format!("snapshot {snap} is not a snapshot of {len} tokens")));
            };
            self.fresh(slot);
            self.chain[slot] = self.snap_chain.get(&snap).copied().unwrap_or(0);
            let full = len / crate::kvtables::BLOCK_TOKENS;
            self.kv[slot] = pages[..full].to_vec();
            self.cover(slot, len);
            self.books[slot] = crate::families::glm::model::TapBook { ok: mtp_ok, first: len, n: 1 };
            self.anchor[slot] = Some(len);
            self.log.lock().unwrap().push(format!("restore {slot} {len} {pages:?}"));
            Ok(())
        }
        fn release_prefix(&mut self, snap: u32, pages: &[u32]) {
            let (free, meta) = self.pool.as_mut().expect("a release without a pool");
            assert!(!free.contains(&snap), "snapshot slot {snap} freed twice");
            meta.remove(&snap);
            free.push(snap);
            self.snap_chain.remove(&snap);
            self.log.lock().unwrap().push(format!("release {pages:?}"));
        }
        fn seal_snapshot(&mut self, snap: u32, pages: &[u32]) -> Result<u64, ModelError> {
            let (len, mtp_ok) = self
                .pool
                .as_ref()
                .and_then(|p| p.1.get(&snap).copied())
                .ok_or_else(|| ModelError::Request(format!("snapshot slot {snap} holds no snapshot to seal")))?;
            Ok(Rec::digest_of(len, mtp_ok, self.snap_chain.get(&snap).copied().unwrap_or(0), pages))
        }
        fn prefix_state_digest(&mut self, slot: usize, len: usize) -> Result<u64, ModelError> {
            let b = &self.books[slot];
            let mtp_ok = b.ok && b.first == len && b.n == 1;
            Ok(Rec::digest_of(len, mtp_ok, self.chain[slot], &self.kv[slot]))
        }
        fn snapshot_handles(&mut self) -> &mut crate::families::glm::tp::SnapHandles {
            &mut self.handles
        }
    }

    fn fnv(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
    }

    impl ImageRuntime for Rec {
        fn prefill_image(&mut self, call: &tp_vision::PrefillImage) -> Result<Option<Candidates>, ModelError> {
            let (slot, start, tokens) = call.check()?;
            let rows: Vec<(u64, u64, String)> =
                call.splice.iter().map(|s| (s.dst_row, s.n, format!("{:016x}", fnv(&s.data)))).collect();
            self.log.lock().unwrap().push(format!("image {slot} {start} {tokens:?} {:?} {rows:?}", call.want));
            // the model's image chunk is a prefill chunk to the drafter's book (`prefill_chunk_mm` -> `mtp_prefill_chunk`)
            let _ = self.books[slot].prefill_chunk(start, tokens.len());
            self.cover(slot, start + tokens.len());
            self.anchor[slot].get_or_insert(start);
            self.eat(slot, start, &tokens);
            Ok(call.want.map(|w| cands(&call.batch, 1, w as usize)))
        }
    }

    impl crate::families::glm::tp_mtp::GlmMtpRuntime for Rec {
        fn mtp_book(&self, slot: usize) -> Option<crate::families::glm::model::TapBook> {
            self.books.get(slot).cloned()
        }
        fn mtp_draft_keyed(
            &mut self,
            asks: &[crate::families::glm::model::DraftAsk<'_>],
            keyed: &[Option<gb_sampling::Sampling>],
        ) -> Result<Vec<Option<crate::families::glm::model::Drafts>>, ModelError> {
            let line: Vec<String> = asks
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let seed = keyed.get(i).and_then(|k| k.as_ref()).map(|k| k.seed);
                    format!("({} {} {:?} {:?})", a.slot, a.depth, a.tokens, seed)
                })
                .collect();
            self.log.lock().unwrap().push(format!("draft {}", line.join(" ")));
            let mut out = Vec::new();
            for a in asks {
                let b = &mut self.books[a.slot];
                if b.absorb_check(a.tokens.len()).is_err() {
                    out.push(None);
                    continue;
                }
                b.absorbed(a.tokens.len());
                let last = *a.tokens.last().unwrap();
                let ids: Vec<u32> = (0..a.depth as u32).map(|k| (last * 7 + k + self.draft_bias) % 1000).collect();
                let vals = ids.iter().map(|&i| i as f32 * 0.25).collect();
                out.push(Some(crate::families::glm::model::Drafts { ids, vals }));
            }
            Ok(out)
        }
        fn mtp_disable_slot(&mut self, slot: usize) {
            self.log.lock().unwrap().push(format!("disable {slot}"));
            self.books[slot].disable();
        }
    }

    pub(crate) fn t() -> TpTimeouts {
        TpTimeouts {
            connect: Duration::from_secs(5),
            ready: Duration::from_secs(5),
            peer: Duration::from_secs(5),
            write: Duration::from_secs(5),
            barrier: Duration::from_secs(5),
            // the process-level backstop of a lost group would end the whole test binary 20 s after these tests lose a group on purpose
            hang_exit: None,
            ..TpTimeouts::default()
        }
    }

    /// A leader backend over a follower thread, both recording.
    pub(crate) fn group(
        chunk: usize,
        fail_follower_window: bool,
    ) -> (GlmTpBackend<Rec>, Log, Log, std::thread::JoinHandle<FollowerExit>) {
        let (a, b) = Link::<GlmCall>::pair(Duration::from_secs(30)).unwrap();
        let (l0, l1): (Log, Log) = (Arc::default(), Arc::default());
        let f = Rec { fail_window: fail_follower_window, ..Rec::new(l1.clone()) };
        let h = std::thread::spawn(move || run_follower(f, b, FollowerConfig::new(t())).unwrap());
        let leader = TpLeader::lead(Rec::new(l0.clone()), a, t());
        // these tests pin P1's slicing (chunks of 3 or 8 rows: P2 cuts a prompt on its 16-row KDA grid, below); the process default is P2
        (GlmTpBackend::new(leader, chunk, 4).with_contract(super::super::contract::PrefillContract::P1), l0, l1, h)
    }

    /// The single-node backend's model calls for the same engine calls, written out by hand from `GlmBackend` (chunks of `chunk` rows,
    /// `begin_slot` at the first slice, the batch of the window as given, commits recorded as given).
    fn drive(be: &mut dyn Backend) -> Vec<Vec<RowOutput>> {
        let mut out = Vec::new();
        // a prompt of 7 tokens in slices (chunk 3: 3, 3, 1), a second one whole, then two windows and their commits, then a release
        let p: Vec<u32> = (10..17).collect();
        let mut done = 0;
        loop {
            let s = be.prefill_slice(0, &p, &[], done, 4);
            done = s.done;
            if let Some(r) = s.row {
                out.push(vec![r]);
                break;
            }
        }
        out.push(vec![be.prefill(1, 0, &[40, 41], 4)]);
        let w = vec![
            RowRequest { slot: 0, tokens: vec![99, 98, 97], first_position: 7 },
            RowRequest { slot: 1, tokens: vec![55], first_position: 2 },
        ];
        out.extend(be.step(&w, 4));
        be.commit(0, 2);
        be.commit(1, 1);
        out.extend(be.step(&[RowRequest { slot: 0, tokens: vec![5], first_position: 9 }], 4));
        be.commit(0, 1);
        be.release(1);
        be.release(0);
        out
    }

    #[test]
    fn the_follower_replays_rank_zeros_calls_exactly_and_they_are_the_single_node_backends_calls() {
        let (mut be, l0, l1, h) = group(3, false);
        let rows = drive(&mut be);
        assert!(be.take_error().is_none());
        be.quiesce().unwrap();
        drop(be);
        assert!(matches!(h.join().unwrap(), FollowerExit::Shutdown(_)));
        let (a, b) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
        assert_eq!(a, b, "the follower's call stream is rank 0's");
        let want = vec![
            "begin 0",
            "prefill 0 0 [10, 11, 12] None",
            "prefill 0 3 [13, 14, 15] None",
            "prefill 0 6 [16] Some(4)",
            "begin 1",
            "prefill 1 0 [40, 41] Some(4)",
            "window [99, 98, 97, 55] [7, 8, 9, 2] [0, 1] 4",
            "commit 0 2",
            "commit 1 1",
            "window [5] [9] [0] 4",
            "commit 0 1",
            "end 1",
            "end 0",
        ];
        assert_eq!(a, want);
        // the answers are rank 0's: one row per prefill, one per window row
        assert_eq!(rows.len(), 2 + 2 + 1);
        assert_eq!(rows[2].len(), 3);
        assert!(rows.iter().flatten().all(|r| r.values.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn the_digest_log_has_one_line_per_answered_row_keyed_by_sequence_position_and_token() {
        let dir = std::env::temp_dir().join(format!("glm-digest-{}", std::process::id()));
        let _ = std::fs::remove_file(&dir);
        let (mut be, _l0, _l1, h) = group(3, false);
        be.digests = DigestLog::to_path(&dir, 4).unwrap();
        let rows = drive(&mut be);
        drop(be);
        h.join().unwrap();
        let text = std::fs::read_to_string(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);
        let lines: Vec<&str> = text.lines().collect();
        // two prefills (one row each), a window of 3 + 1 rows, a window of 1 row
        assert_eq!(lines.len(), rows.iter().map(|r| r.len()).sum::<usize>(), "{text}");
        let keys: Vec<String> = lines.iter().map(|l| l.split(' ').take(3).collect::<Vec<_>>().join(" ")).collect();
        assert_eq!(keys, ["1 6 16", "2 1 41", "1 7 99", "1 8 98", "1 9 97", "2 2 55", "1 9 5"]);
    }

    #[test]
    fn a_chunk_size_change_moves_only_the_chunk_grid() {
        let (mut be, l0, _l1, h) = group(128, false);
        let mut done = 0;
        let p: Vec<u32> = (0..300).collect();
        loop {
            let s = be.prefill_slice(2, &p, &[], done, 1);
            done = s.done;
            if s.row.is_some() {
                break;
            }
        }
        drop(be);
        h.join().unwrap();
        let a = l0.lock().unwrap().clone();
        let chunks: Vec<&String> = a.iter().filter(|s| s.starts_with("prefill")).collect();
        assert_eq!(chunks.len(), 3, "128 + 128 + 44: {a:?}");
        assert!(chunks[2].starts_with("prefill 2 256 ") && chunks[2].ends_with("Some(1)"));
    }

    /// Under P2 (the default contract) the leader cuts a prompt into slices of a whole number of 16-row KDA chunks (the last one ends the
    /// prompt), and the follower replays exactly those slices.
    #[test]
    fn under_p2_the_leader_slices_land_on_the_16_row_grid_and_the_follower_replays_them() {
        use super::super::contract::PrefillContract::P2;
        let (be, l0, l1, h) = group(40, false);
        let mut be = be.with_contract(P2);
        assert_eq!(be.slice_rows(), 32, "a 40-row chunk is 2 KDA chunks of 16");
        let p: Vec<u32> = (0..100).collect();
        let (mut done, mut row) = (0, None);
        while row.is_none() {
            let s = be.prefill_slice(1, &p, &[], done, 2);
            (done, row) = (s.done, s.row);
        }
        drop(be);
        assert!(matches!(h.join().unwrap(), FollowerExit::Shutdown(_)));
        let (a, b) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
        assert_eq!(a, b, "the follower replays rank 0's slices");
        let starts: Vec<String> =
            a.iter().filter(|x| x.starts_with("prefill")).map(|x| x.split(' ').take(3).collect::<Vec<_>>().join(" ")).collect();
        assert_eq!(starts, ["prefill 1 0", "prefill 1 32", "prefill 1 64", "prefill 1 96"], "32 + 32 + 32 + 4 rows");
    }

    /// A P2 leader backend over a follower thread, both recording, whose runtimes run forwards of several prompts and in parts (`room`
    /// rows meanwhile, `layers` layers), with parts of `part_ms` while a sequence decodes.
    fn parts_group(
        chunk: usize,
        room: usize,
        layers: usize,
        part_ms: f64,
    ) -> (GlmTpBackend<Rec>, Log, Log, std::thread::JoinHandle<FollowerExit>) {
        let (a, b) = Link::<GlmCall>::pair(Duration::from_secs(30)).unwrap();
        let (l0, l1): (Log, Log) = (Arc::default(), Arc::default());
        let f = Rec { parts: Some((room, layers)), ..Rec::new(l1.clone()) };
        let h = std::thread::spawn(move || run_follower(f, b, FollowerConfig::new(t())).unwrap());
        let leader = TpLeader::lead(Rec { parts: Some((room, layers)), ..Rec::new(l0.clone()) }, a, t());
        let be = GlmTpBackend::new(leader, chunk, 4).with_contract(super::super::contract::PrefillContract::P2).with_parts(part_ms);
        (be, l0, l1, h)
    }

    /// Each prompt's row alone (slot = its index): the P2 backend over a runtime that runs neither batches nor parts, slice by slice.
    fn solo_rows(chunk: usize, prompts: &[Vec<u32>], candidates: usize) -> Vec<RowOutput> {
        let (be, _l0, _l1, h) = group(chunk, false);
        let mut be = be.with_contract(super::super::contract::PrefillContract::P2);
        let rows = prompts
            .iter()
            .enumerate()
            .map(|(slot, p)| {
                let mut done = 0;
                loop {
                    let s = be.prefill_slice(slot, p, &[], done, candidates);
                    done = s.done;
                    if let Some(r) = s.row {
                        return r;
                    }
                }
            })
            .collect();
        drop(be);
        h.join().unwrap();
        rows
    }

    /// `tp_part`: the slices of several waiting prompts go through ONE `PrefillPart` call (every rank runs it), as many as fit the chunk;
    /// the one that does not fit waits (not served: `done` unchanged) and is served alone by the next call; each prompt's row is the row
    /// it gets alone.
    #[test]
    fn the_slices_of_several_waiting_prompts_share_one_forward_and_each_row_is_its_solo_row() {
        let prompts: Vec<Vec<u32>> = vec![(1..20).collect(), (30..33).collect(), (50..90).collect(), (100..130).collect()];
        let (mut be, l0, l1, h) = parts_group(64, 32, 6, 300.0);
        assert!(be.batches());
        let jobs: Vec<PrefillJob<'_>> =
            prompts.iter().enumerate().map(|(slot, p)| PrefillJob { slot, prompt: p, images: &[], done: 0, candidates: 3 }).collect();
        let mut steps = be.prefill_slices(&jobs);
        assert!(steps[3].done == 0 && steps[3].row.is_none(), "the prompt that does not fit with the others waits");
        steps[3] = be.prefill_slices(&jobs[3..]).remove(0);
        assert!(be.take_error().is_none());
        drop(be);
        assert!(matches!(h.join().unwrap(), FollowerExit::Shutdown(_)));
        let (a, b) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
        assert_eq!(a, b, "the follower runs rank 0's forwards");
        assert_eq!(
            a,
            [
                "begin 0",
                "begin 1",
                "begin 2",
                "part [0@0+19 1@0+3 2@0+40] Some(3) until=4294967295",
                "begin 3",
                "prefill 3 0 [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129] Some(3)",
            ],
            "19 + 3 + 40 rows fit the 64-row chunk; the 30 more do not"
        );
        let solo = solo_rows(64, &prompts, 3);
        for (i, (s, r)) in steps.iter().zip(&solo).enumerate() {
            assert_eq!(s.done, prompts[i].len());
            assert_eq!(s.row.as_ref(), Some(r), "prompt {i}");
        }
    }

    /// `tp_part`: while a sequence decodes a call runs ONE forward: the fresh short prompts together (at most the batch rows), or the
    /// first job that goes alone (here a long slice, run in parts), the two taking turns while both wait; a short prompt that arrives
    /// while the long one is paused runs whole between two parts (its turn), and the parts go on. Every job a call does not run is not
    /// served. The follower runs rank 0's calls, and every row is the one its prompt gets alone.
    #[test]
    fn while_a_sequence_decodes_a_call_runs_one_forward_and_short_prompts_and_a_long_slice_take_turns() {
        let prompts: Vec<Vec<u32>> = vec![(1..11).collect(), (20..32).collect(), (100..300).collect(), (400..408).collect()];
        let (be, l0, l1, h) = parts_group(256, 32, 6, 1e-3);
        let mut be = be.with_batch_rows(64);
        be.set_decode_waiting(true);
        let job = |slot: usize| PrefillJob { slot, prompt: &prompts[slot], images: &[], done: 0, candidates: 2 };
        let mut rows: Vec<Option<RowOutput>> = vec![None; prompts.len()];
        // the two short prompts together; the long one waits
        let s = be.prefill_slices(&[job(0), job(1), job(2)]);
        assert!(s[0].row.is_some() && s[1].row.is_some() && s[2].done == 0 && s[2].row.is_none());
        (rows[0], rows[1]) = (s[0].row.clone(), s[1].row.clone());
        // its turn: the long slice's first part; the short prompt that arrived waits
        let s = be.prefill_slices(&[job(3), job(2)]);
        assert!(s[0].done == 0 && s[0].row.is_none() && s[1].done == 0 && s[1].row.is_none());
        assert_eq!(be.prefill_in_flight(), Some((2, 32)));
        // the short prompt's turn, whole while the long one is paused
        let s = be.prefill_slices(&[job(3), job(2)]);
        assert!(s[0].done == 8 && s[1].done == 0 && s[1].row.is_none());
        rows[3] = s[0].row.clone();
        let s = be.prefill_slices(&[job(2)]);
        assert!(s[0].done == 0 && s[0].row.is_none());
        be.set_decode_waiting(false);
        let s = be.prefill_slices(&[job(2)]);
        assert_eq!(s[0].done, 200);
        rows[2] = s[0].row.clone();
        assert!(be.take_error().is_none());
        drop(be);
        assert!(matches!(h.join().unwrap(), FollowerExit::Shutdown(_)));
        let (a, b) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
        assert_eq!(a, b, "the follower runs rank 0's calls");
        assert_eq!(
            a,
            [
                "begin 0",
                "begin 1",
                "part [0@0+10 1@0+12] Some(2) until=4294967295",
                "begin 2",
                "part [2@0+200] Some(2) until=1",
                "begin 3",
                "prefill 3 0 [400, 401, 402, 403, 404, 405, 406, 407] Some(2)",
                "resume until=2",
                "resume until=4294967295",
            ]
        );
        let solo = solo_rows(256, &prompts, 2);
        for (i, (r, s)) in rows.iter().zip(&solo).enumerate() {
            assert_eq!(r.as_ref(), Some(s), "prompt {i}");
        }
    }

    /// `tp_part`: while a sequence decodes, a long slice runs in parts (each call one, `done` unchanged and no row until the last; the
    /// backend names it in flight); a prompt that fits the room runs whole between two parts, one that does not is not run at all; when
    /// nothing decodes any more the rest runs in one part. The follower runs rank 0's calls, and the row is the one the prompt gets alone.
    #[test]
    fn a_long_slice_runs_in_parts_while_a_sequence_decodes_and_a_short_prompt_runs_between_two() {
        let long: Vec<u32> = (0..200).collect();
        let short: Vec<u32> = (500..510).collect();
        let other: Vec<u32> = (700..800).collect();
        let (mut be, l0, l1, h) = parts_group(256, 32, 6, 1e-3);
        be.set_decode_waiting(true);
        let s = be.prefill_slice(0, &long, &[], 0, 2);
        assert!(s.done == 0 && s.row.is_none());
        assert_eq!(be.prefill_in_flight(), Some((0, 32)));
        let s = be.prefill_slice(1, &short, &[], 0, 2);
        assert!(s.done == short.len() && s.row.is_some(), "a prompt that fits the room runs whole meanwhile");
        let s = be.prefill_slice(2, &other, &[], 0, 2);
        assert!(s.done == 0 && s.row.is_none(), "one that does not fit is not run");
        let s = be.prefill_slice(0, &long, &[], 0, 2);
        assert!(s.done == 0 && s.row.is_none());
        be.set_decode_waiting(false);
        let s = be.prefill_slice(0, &long, &[], 0, 2);
        assert_eq!(s.done, long.len());
        assert_eq!(be.prefill_in_flight(), None);
        let row = s.row.expect("the last part ends the prompt");
        assert!(be.take_error().is_none());
        drop(be);
        assert!(matches!(h.join().unwrap(), FollowerExit::Shutdown(_)));
        let (a, b) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
        assert_eq!(a, b, "the follower runs rank 0's parts");
        assert_eq!(
            a,
            [
                "begin 0",
                "part [0@0+200] Some(2) until=1",
                "begin 1",
                "prefill 1 0 [500, 501, 502, 503, 504, 505, 506, 507, 508, 509] Some(2)",
                "resume until=2",
                "resume until=4294967295",
            ]
        );
        assert_eq!(row, solo_rows(256, &[long], 2)[0]);
    }

    /// Lane E: the adaptive slice. With no sequence decoding a prompt goes in the big chunk; while one is, in the small one; the follower
    /// replays exactly rank 0's slices (its call stream is rank 0's), and the answer (a pure function of the last row, as the model's is
    /// chunk-invariant) is the one-size backend's. The engine's deadline limit still caps a slice.
    #[test]
    fn the_adaptive_slice_is_big_when_nothing_decodes_and_small_while_a_sequence_does() {
        let run = |adaptive: bool, waiting: &[bool]| {
            let (be, l0, l1, h) = group(8, false);
            let mut be = if adaptive { be.with_adaptive(3) } else { be };
            let p: Vec<u32> = (0..20).collect();
            let (mut done, mut i, mut row) = (0, 0, None);
            while row.is_none() {
                be.set_decode_waiting(waiting[i.min(waiting.len() - 1)]);
                let s = be.prefill_slice(1, &p, &[], done, 2);
                (done, row, i) = (s.done, s.row, i + 1);
            }
            drop(be);
            assert!(matches!(h.join().unwrap(), FollowerExit::Shutdown(_)));
            let (a, b) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
            assert_eq!(a, b, "the follower replays rank 0's slices");
            let starts: Vec<String> = a
                .iter()
                .filter(|x| x.starts_with("prefill"))
                .map(|x| x.split(' ').take(3).collect::<Vec<_>>().join(" "))
                .collect();
            (starts, row.unwrap())
        };
        // nothing decodes: 8 + 8 + 4 (the chunk), with or without the adaptive slice
        let (big, r_big) = run(true, &[false]);
        assert_eq!(big, ["prefill 1 0", "prefill 1 8", "prefill 1 16"]);
        let (one, r_one) = run(false, &[true]);
        assert_eq!(one, big, "one size: the chunk whatever the engine says");
        // a sequence decodes for the first two slices (3 + 3), then none does (8 + 6)
        let (mixed, r_mixed) = run(true, &[true, true, false]);
        assert_eq!(mixed, ["prefill 1 0", "prefill 1 3", "prefill 1 6", "prefill 1 14"]);
        assert!(r_big == r_one && r_one == r_mixed, "the answer does not depend on the grid");
        // the deadline limit caps either size
        let (be, _l0, _l1, h) = group(8, false);
        let mut be = be.with_adaptive(3);
        be.set_prefill_row_limit(Some(2));
        assert_eq!(be.slice_rows(), 2);
        be.set_decode_waiting(true);
        assert_eq!(be.slice_rows(), 2);
        be.set_prefill_row_limit(None);
        assert_eq!(be.slice_rows(), 3);
        drop(be);
        h.join().unwrap();
    }

    // ---------------------------------------------------------------------------------------------------------- the prefix cache

    /// A leader backend with the prefix cache over a follower thread, both recording: `slots` snapshot slots on each rank (the follower
    /// numbers its own from 1000, so none of its ids is rank 0's), slices of `chunk` rows on `contract`'s grid, drafting by `spec`.
    fn cache_group(
        chunk: usize,
        contract: super::super::contract::PrefillContract,
        slots: usize,
        spec: Option<gb_spec::SpecMode>,
    ) -> (GlmTpBackend<Rec>, Log, Log, std::thread::JoinHandle<FollowerExit>) {
        let (a, b) = Link::<GlmCall>::pair(Duration::from_secs(30)).unwrap();
        let (l0, l1): (Log, Log) = (Arc::default(), Arc::default());
        let f = Rec { snap_base: 1000, ..Rec::new(l1.clone()) };
        let h = std::thread::spawn(move || run_follower(f, b, FollowerConfig::new(t())).unwrap());
        let leader = TpLeader::lead(Rec::new(l0.clone()), a, t());
        let be = match spec {
            Some(m) => GlmTpBackend::with_spec(leader, chunk, 4, crate::families::glm::spec::SpecConfig::new(m)),
            None => GlmTpBackend::new(leader, chunk, 4),
        };
        let budget = CacheBudget { snapshot_slots: slots, page_blocks: 32, snapshot_bytes: 4096, page_bytes: 1 << 20 };
        (be.with_contract(contract).with_prefix_cache(7, budget).unwrap(), l0, l1, h)
    }

    /// `prompt` into `slot` from `done` by `prefill_slice_cuts` with the engine's `cuts`: the answer and every snapshot the slices took.
    fn cut_slices(
        be: &mut GlmTpBackend<Rec>,
        slot: usize,
        prompt: &[u32],
        mut done: usize,
        cuts: &[usize],
    ) -> (RowOutput, Vec<Captured>) {
        let mut captured = Vec::new();
        loop {
            let s = be.prefill_slice_cuts(slot, prompt, &[], done, 4, cuts);
            captured.extend(s.captured);
            done = s.step.done;
            if let Some(r) = s.step.row {
                return (r, captured);
            }
        }
    }

    /// The prefix cache's call stream (`docs/GLM_TP.md` 4.2): a cold prompt takes a snapshot at each of the engine's cuts (a `Snapshot` call
    /// after the slice that ends there), a warm turn of it restores the later entry into another slot and prefills the rest, a request that
    /// shares only the first entry's tokens restores that one, and both entries are released at rank 0's word. The follower replays exactly
    /// rank 0's stream on its OWN snapshot slots (numbered from 1000, rank 0's from 0): rank 0 names a handle by the sequence number of the
    /// call that made it, so the follower restores and releases the snapshot it took in that call. The warm answer is the cold one, both
    /// entries carry a sound drafter, and nothing stays held on either rank.
    #[test]
    fn a_cached_prompt_replays_on_the_follower_with_seq_keyed_handles() {
        use super::super::contract::PrefillContract::P2;
        let (mut be, l0, l1, h) = cache_group(128, P2, 4, None);
        assert_eq!((be.prefix_caps().map(|c| c.namespace), be.prefix_path(600)), (Some(7), Some(prefix::PATH_ID)));
        let p: Vec<u32> = (0..600).map(|i| (i * 7 % 1000) as u32).collect();
        let (cold, caps) = cut_slices(&mut be, 0, &p, 0, &[272, 528]);
        assert_eq!(caps.iter().map(|c| c.len).collect::<Vec<_>>(), [272, 528]);
        assert_eq!(caps[0].handle, PrefixHandle { snap: 0, pages: vec![0, 1] }, "rank 0's own slot ids");
        assert_eq!(caps[1].handle, PrefixHandle { snap: 1, pages: vec![0, 1, 2] });
        assert_eq!(be.prefix_free_snapshots(), 2);
        be.release(0);
        be.prefix_restore(1, &caps[1].handle, 528).unwrap();
        let (warm, more) = cut_slices(&mut be, 1, &p, 528, &[]);
        assert!(more.is_empty());
        assert_eq!(warm, cold, "the warm turn answers as the cold prompt");
        let mut q = p[..272].to_vec();
        q.extend(900..980);
        be.prefix_restore(2, &caps[0].handle, 272).unwrap();
        cut_slices(&mut be, 2, &q, 272, &[]);
        be.step(&[RowRequest { slot: 1, tokens: vec![5], first_position: 600 }], 4);
        be.commit(1, 1);
        for c in caps {
            be.prefix_release(c.handle);
        }
        be.release(1);
        be.release(2);
        assert!(be.take_error().is_none());
        assert_eq!((be.snapshots_without_drafts(), be.prefix_free_snapshots()), (0, 4));
        be.quiesce().unwrap();
        assert_eq!(be.leader.failure(), None);
        drop(be);
        assert!(matches!(h.join().unwrap(), FollowerExit::Shutdown(_)));
        let (a, b) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
        assert_eq!(a, b, "the follower's call stream is rank 0's");
        let cache: Vec<&str> = a.iter().filter(|x| !x.starts_with("prefill")).map(String::as_str).collect();
        assert_eq!(
            cache,
            [
                "enable_snapshots 4",
                "begin 0",
                "snapshot 0 272 mtp_ok=true",
                "retain 0 272 [0, 1]",
                "snapshot 0 528 mtp_ok=true",
                "retain 0 528 [0, 1, 2]",
                "end 0",
                "restore 1 528 [0, 1, 2]",
                "restore 2 272 [0, 1]",
                "window [5] [600] [1] 4",
                "commit 1 1",
                "release [0, 1]",
                "release [0, 1, 2]",
                "end 1",
                "end 2",
            ]
        );
    }

    /// The pages a snapshot holds are retained in the same call on every rank, for the same length: a cut one token past a page boundary
    /// (P1's grid takes any cut) holds the partial page too, and the follower retains exactly rank 0's pages (the call log of each rank;
    /// each rank's full page ids are in its own content digest, verified at its restore).
    #[test]
    fn the_follower_retains_the_same_pages() {
        use super::super::contract::PrefillContract::P1;
        let (mut be, l0, l1, h) = cache_group(128, P1, 4, None);
        let p: Vec<u32> = (0..600).collect();
        let (_, caps) = cut_slices(&mut be, 0, &p, 0, &[257, 513]);
        assert_eq!(
            caps.iter().map(|c| (c.len, c.handle.pages.clone())).collect::<Vec<_>>(),
            [(257, vec![0, 1]), (513, vec![0, 1, 2])]
        );
        be.quiesce().unwrap();
        assert_eq!(be.leader.failure(), None);
        for c in caps {
            be.prefix_release(c.handle);
        }
        be.release(0);
        be.quiesce().unwrap();
        drop(be);
        assert!(matches!(h.join().unwrap(), FollowerExit::Shutdown(_)));
        let retained = |l: &Log| l.lock().unwrap().iter().filter(|x| x.starts_with("retain")).cloned().collect::<Vec<_>>();
        assert_eq!(retained(&l0), ["retain 0 257 [0, 1]", "retain 0 513 [0, 1, 2]"]);
        assert_eq!(retained(&l1), retained(&l0), "the follower's pages are rank 0's");
    }

    /// Under P2 (the serving contract) a slice ENDS at each of the engine's cuts, moved down to the slot's 16-row grid, and the snapshot is a
    /// call after that chunk: no prefill chunk on either rank has a cut strictly inside it, the snapshot of every cut follows the chunk that
    /// ends there, and the captures are at the moved cuts (a rank of a KDA head split runs each chunk's KDA whole, so a cut inside a chunk
    /// would copy the state after the chunk).
    #[test]
    fn a_tp_slice_never_cuts_inside_a_chunk() {
        use super::super::contract::PrefillContract::P2;
        let (mut be, l0, l1, h) = cache_group(128, P2, 4, None);
        let p: Vec<u32> = (0..700).collect();
        let (_, caps) = cut_slices(&mut be, 0, &p, 0, &[200, 300, 650]);
        let at: Vec<usize> = caps.iter().map(|c| c.len).collect();
        assert_eq!(at, [192, 288, 640], "the cuts on the 16-row grid, each captured");
        be.quiesce().unwrap();
        drop(be);
        assert!(matches!(h.join().unwrap(), FollowerExit::Shutdown(_)));
        let (a, b) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
        assert_eq!(a, b, "the follower's call stream is rank 0's");
        // (line, first row, end) of every prompt chunk: `prefill 0 <start> [<tokens>] <want>`
        let chunks: Vec<(usize, usize, usize)> = a
            .iter()
            .enumerate()
            .filter_map(|(i, x)| {
                let rest = x.strip_prefix("prefill 0 ")?;
                let start: usize = rest.split(' ').next()?.parse().ok()?;
                let tokens = &rest[rest.find('[')? + 1..rest.find(']')?];
                Some((i, start, start + tokens.split(", ").count()))
            })
            .collect();
        assert_eq!(
            chunks.iter().map(|&(_, s, e)| (s, e)).collect::<Vec<_>>(),
            [(0, 128), (128, 192), (192, 288), (288, 416), (416, 544), (544, 640), (640, 700)]
        );
        for &c in &at {
            assert!(chunks.iter().all(|&(_, s, e)| !(s < c && c < e)), "a chunk holds the cut at {c} inside it: {chunks:?}");
            let &(i, _, _) = chunks.iter().find(|&&(_, _, e)| e == c).expect("a chunk ends at every cut");
            assert_eq!(a[i + 1], format!("snapshot 0 {c} mtp_ok=true"), "the snapshot right after the chunk that ends at {c}");
        }
    }

    /// The arena is agreed again in `EnableSnapshots`' ack (`docs/GLM_TP.md` 4.2, pitfall 1): a follower whose pool holds another slot size
    /// than rank 0's fails the boot, naming the call, before anything is served.
    #[test]
    fn a_follower_whose_snapshot_slot_is_another_size_fails_the_boot_naming_enable_snapshots() {
        let (a, b) = Link::<GlmCall>::pair(Duration::from_secs(30)).unwrap();
        let f = Rec { snap_bytes: 2048, ..Rec::new(Log::default()) };
        let h = std::thread::spawn(move || run_follower(f, b, FollowerConfig::new(t())));
        let leader = TpLeader::lead(Rec::new(Log::default()), a, t());
        let budget = CacheBudget { snapshot_slots: 4, page_blocks: 32, snapshot_bytes: 4096, page_bytes: 1 << 20 };
        let e = GlmTpBackend::new(leader, 128, 4).with_prefix_cache(7, budget).err().expect("the boot fails").to_string();
        assert!(e.contains("diverged") && e.contains("enable_snapshots") && e.contains("rank 1"), "{e}");
        let _ = h.join();
    }

    /// Rank 0 forwards only handles it made and still holds exactly as made (Nemotron's leader `snaps`): a foreign slot id, or its own
    /// slot with other pages, is refused by name with no `Restore` or `Release` sent (the slot of a refused restore is released on every
    /// rank, as one node drops it); a handle released once is refused the second time, so no rank frees a snapshot slot twice. The group
    /// serves on.
    #[test]
    fn a_handle_rank_0_did_not_make_or_no_longer_holds_is_refused_and_never_forwarded() {
        use super::super::contract::PrefillContract::P2;
        let (mut be, l0, l1, h) = cache_group(128, P2, 4, None);
        let p: Vec<u32> = (0..600).collect();
        let (_, caps) = cut_slices(&mut be, 0, &p, 0, &[272]);
        let real = caps[0].handle.clone();
        let foreign = PrefixHandle { snap: 99, pages: real.pages.clone() };
        let stale = PrefixHandle { snap: real.snap, pages: vec![7] };
        for bad in [&foreign, &stale] {
            let sent = be.leader.stats().commands;
            let e = be.prefix_restore(1, bad, 272).unwrap_err();
            assert!(e.contains("is not a handle rank 0 made and still holds"), "{e}");
            assert!(!be.take_error().unwrap().fatal);
            assert_eq!(be.leader.stats().commands, sent + 1, "only the slot's release (end_slot) is forwarded");
        }
        let sent = be.leader.stats().commands;
        be.prefix_release(stale);
        assert_eq!(be.leader.stats().commands, sent, "a stale handle's release sends nothing");
        be.prefix_release(real.clone());
        assert_eq!(be.leader.stats().commands, sent + 1);
        be.prefix_release(real);
        assert_eq!(be.leader.stats().commands, sent + 1, "a second release of one handle sends nothing");
        assert_eq!(be.prefix_free_snapshots(), 4);
        be.release(0);
        be.quiesce().unwrap();
        assert_eq!(be.leader.failure(), None);
        drop(be);
        assert!(matches!(h.join().unwrap(), FollowerExit::Shutdown(_)));
        let (a, b) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
        assert_eq!(a, b, "the follower's call stream is rank 0's");
        let cache: Vec<&str> = a
            .iter()
            .filter(|x| x.starts_with("restore") || x.starts_with("release") || x.starts_with("end"))
            .map(String::as_str)
            .collect();
        assert_eq!(cache, ["end 1", "end 1", "release [0, 1]", "end 0"]);
    }

    /// The drafter survives a restore (`docs/GLM_TP.md` 4.2, pitfall 4): a slice-end snapshot carries a sound book (`mtp_ok`), so the warm
    /// turn of a restored slot drafts at once, and its draft call reaches the follower like any other.
    #[test]
    fn a_restored_slot_drafts_on_its_warm_turn() {
        use super::super::contract::PrefillContract::P2;
        let (mut be, l0, l1, h) = cache_group(128, P2, 4, Some(gb_spec::SpecMode::Mtp));
        let p: Vec<u32> = (0..600).map(|i| (i * 3 % 1000) as u32).collect();
        let (_, caps) = cut_slices(&mut be, 0, &p, 0, &[528]);
        be.release(0);
        be.prefix_restore(1, &caps[0].handle, 528).unwrap();
        cut_slices(&mut be, 1, &p, 528, &[]);
        be.step(&[RowRequest { slot: 1, tokens: vec![500], first_position: 600 }], 4);
        be.commit(1, 1);
        let mut seq = p.clone();
        seq.extend([500, 501]);
        let greedy = gb_sampling::Sampling { temperature: 0.0, ..Default::default() };
        let d = be.propose_batch(&[ProposeRequest { slot: 1, tokens: &seq, sampling: &greedy, max: 7 }]);
        assert!(d[0].as_ref().is_some_and(|x| !x.is_empty()), "the restored slot drafts: {d:?}");
        assert_eq!(be.snapshots_without_drafts(), 0);
        assert!(be.take_error().is_none());
        be.quiesce().unwrap();
        drop(be);
        assert!(matches!(h.join().unwrap(), FollowerExit::Shutdown(_)));
        let (a, b) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
        assert_eq!(a, b, "the follower's call stream is rank 0's");
        assert!(a.iter().any(|x| x.starts_with("draft (1 ")), "{a:?}");
    }

    /// The ENGINE over rank 0's backend with the cache on (`tp_boot::engine_over` and the engine's own cache, as the boot will serve it): a
    /// first request leaves an entry at its end cut, a second with the same prompt hits it (`cached_tokens`, one `Restore` forwarded) and
    /// answers as the first, and both ranks saw the same stream.
    #[test]
    fn the_engine_serves_a_cached_prompt_through_the_group() {
        use super::super::contract::PrefillContract::P2;
        use crate::families::glm::tp_boot::engine_over;
        use gb_engine::{PrefixConfig, Request};
        use gb_sched::{Admission, ModelMemory};
        let (be, l0, l1, h) = cache_group(128, P2, 4, None);
        let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1, state_bytes_per_slot: 1, activation_reserve: 0 };
        let mut e = engine_over(be, Admission::new(1 << 30, 0, mm, 4).unwrap(), false).with_prefix_cache(PrefixConfig::default());
        let prompt: Vec<u32> = (0..1100).map(|i| (i * 5 % 1000) as u32).collect();
        let mut outputs = Vec::new();
        for _ in 0..2 {
            let req = Request {
                prompt: prompt.clone(),
                sampling: gb_sampling::Sampling::default(),
                max_new: 3,
                stop: vec![],
                draft: false,
                images: vec![],
                cache: true,
            };
            let id = e.submit(req).unwrap();
            let mut out = None;
            for _ in 0..200 {
                e.step().unwrap();
                if let Some(o) = e.take_outputs().into_iter().find(|o| o.id == id) {
                    out = Some(o);
                    break;
                }
            }
            outputs.push(out.expect("the request finished"));
        }
        let stats = e.prefix_stats();
        assert!(stats.enabled && stats.inserts >= 1 && stats.hits == 1, "{stats:?}");
        assert_eq!(outputs[0].cached_tokens, 0);
        assert_eq!(outputs[1].cached_tokens, 1072, "the end cut (1,100 - the tail of 16) moved down to the 16-row grid");
        assert_eq!(outputs[1].tokens, outputs[0].tokens, "the cached request answers as the cold one");
        drop(e);
        h.join().unwrap();
        let (a, b) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
        assert_eq!(a, b, "the follower's call stream is rank 0's");
        let restored = format!(" {} [", outputs[1].cached_tokens);
        assert_eq!(a.iter().filter(|x| x.starts_with("restore ") && x.contains(&restored)).count(), 1, "{a:?}");
    }

    // ---------------------------------------------------------------------------------------- the restore verified per rank, cold otherwise

    /// The engine over a two-rank cache group whose FOLLOWER has the fault switches `faults` (shared with the test), with these cache
    /// deadlines: the serving path of `glm53-flash-tp2-1m` on the recording mock (each rank's state is its chain, `Rec::eat`).
    fn faulty_engine(
        faults: Arc<RecFaults>,
        deadlines: lockstep::CacheDeadlines,
    ) -> (gb_engine::Engine<GlmTpBackend<Rec>>, std::thread::JoinHandle<Result<FollowerExit, ModelError>>) {
        use super::super::contract::PrefillContract::P2;
        use crate::families::glm::tp_boot::engine_over;
        use gb_engine::PrefixConfig;
        use gb_sched::{Admission, ModelMemory};
        let (a, b) = Link::<GlmCall>::pair(Duration::from_secs(30)).unwrap();
        let f = Rec { snap_base: 1000, faults, ..Rec::new(Log::default()) };
        let h = std::thread::spawn(move || run_follower(f, b, FollowerConfig::new(t())));
        let leader = TpLeader::lead(Rec::new(Log::default()), a, t());
        leader.set_cache_deadlines(deadlines);
        let budget = CacheBudget { snapshot_slots: 4, page_blocks: 32, snapshot_bytes: 4096, page_bytes: 1 << 20 };
        let be = GlmTpBackend::new(leader, 128, 4).with_contract(P2).with_prefix_cache(7, budget).unwrap();
        let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1, state_bytes_per_slot: 1, activation_reserve: 0 };
        (engine_over(be, Admission::new(1 << 30, 0, mm, 4).unwrap(), false).with_prefix_cache(PrefixConfig::default()), h)
    }

    /// One request to its end on `e`: its output, or the engine's error.
    fn serve(e: &mut gb_engine::Engine<GlmTpBackend<Rec>>, prompt: &[u32], cache: bool) -> Result<gb_engine::Output, String> {
        let req = gb_engine::Request {
            prompt: prompt.to_vec(),
            sampling: gb_sampling::Sampling::default(),
            max_new: 3,
            stop: vec![],
            draft: false,
            images: vec![],
            cache,
        };
        let id = e.submit(req).map_err(|x| format!("{x:?}"))?;
        for _ in 0..400 {
            e.step().map_err(|x| format!("{x:?}"))?;
            if let Some(o) = e.take_outputs().into_iter().find(|o| o.id == id) {
                return Ok(o);
            }
        }
        Err("the request did not finish".into())
    }

    fn glm_turns() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let t1: Vec<u32> = (0..1100).map(|i| (i * 5 % 1000) as u32).collect();
        let t2 = [t1.clone(), (0..300).map(|i| (i * 7 % 997) as u32).collect()].concat();
        let t3 = [t2.clone(), (0..300).map(|i| (i * 11 % 991) as u32).collect()].concat();
        (t1, t2, t3)
    }

    /// The scenario every restore fault runs on the GLM group: turn 1 leaves an entry on both ranks; an unrelated cold prompt leaves
    /// other state; the follower is given `fault`; turn 2 (turn 1 plus more) must be refused (counted), answer as its cold run, and leave
    /// the group up; a clean turn 3 hits. Returns turn 2's (hits, restore failures).
    fn glm_restore_fault(fault: impl Fn(&RecFaults)) -> (u64, u64) {
        let faults = Arc::new(RecFaults::default());
        let d = lockstep::CacheDeadlines { ack: Duration::from_millis(300), stall: Duration::from_secs(20) };
        let (mut e, h) = faulty_engine(faults.clone(), d);
        let (t1, t2, t3) = glm_turns();
        let cold2 = serve(&mut e, &t2, false).unwrap();
        serve(&mut e, &t1, true).unwrap();
        assert!(e.prefix_stats().entries >= 1, "turn 1 left an entry");
        serve(&mut e, &(0..700).map(|i| (i * 13 % 983) as u32).collect::<Vec<_>>(), false).unwrap();
        fault(&faults);
        let before = e.prefix_stats();
        let warm2 = serve(&mut e, &t2, true).unwrap();
        let after = e.prefix_stats();
        assert_eq!(warm2.tokens, cold2.tokens, "turn 2 answers as its cold run");
        assert_eq!(e.backend.leader.failure(), None, "the group stays up");
        faults.restore.store(0, std::sync::atomic::Ordering::SeqCst);
        faults.restore_sleep_ms.store(0, std::sync::atomic::Ordering::SeqCst);
        faults.skip.store(false, std::sync::atomic::Ordering::SeqCst);
        // the cache serves again (after every answer is in: on this mock nothing holds rank 0 back while a slow follower sleeps, where a
        // real group's next collective would)
        e.backend.leader.barrier().unwrap();
        let hits = e.prefix_stats().hits;
        serve(&mut e, &t3, true).unwrap();
        serve(&mut e, &t3, true).unwrap();
        assert!(e.prefix_stats().hits > hits, "the cache serves again");
        drop(e);
        assert!(matches!(h.join().unwrap(), Ok(FollowerExit::Shutdown(_))));
        (after.hits - before.hits, after.restore_failures - before.restore_failures)
    }

    #[test]
    fn a_glm_restore_verified_on_every_rank_is_a_hit() {
        assert_eq!(glm_restore_fault(|_| {}), (1, 0));
    }

    /// The follower's restore fails: both ranks go cold (before: a lost group).
    #[test]
    fn a_glm_follower_restore_error_falls_back_cold() {
        assert_eq!(glm_restore_fault(|f| f.restore.store(1, std::sync::atomic::Ordering::SeqCst)), (0, 1));
    }

    /// The follower restores nothing and answers Ok (the 43baf947 class): its digest of what it restored is not its snapshot's, per request.
    #[test]
    fn a_glm_follower_that_restores_nothing_but_answers_ok_is_caught_per_request() {
        assert_eq!(glm_restore_fault(|f| f.restore.store(2, std::sync::atomic::Ordering::SeqCst)), (0, 1));
    }

    /// The follower leaves the previous prompt's state and answers Ok: caught by its digest.
    #[test]
    fn a_glm_follower_stale_restore_that_answers_ok_is_caught_per_request() {
        assert_eq!(glm_restore_fault(|f| f.restore.store(3, std::sync::atomic::Ordering::SeqCst)), (0, 1));
    }

    /// A follower slower than the wait: the restore fails over to a cold prefill and the late answer is consumed.
    #[test]
    fn a_slow_glm_follower_restore_falls_back_cold_and_keeps_the_group() {
        assert_eq!(glm_restore_fault(|f| f.restore_sleep_ms.store(1_500, std::sync::atomic::Ordering::SeqCst)), (0, 1));
    }

    /// A snapshot the follower could not take (no free slot on that rank only) is released on every rank and never enters the index.
    #[test]
    fn a_glm_snapshot_skipped_on_one_rank_is_never_entered_or_used() {
        use std::sync::atomic::Ordering::SeqCst;
        let faults = Arc::new(RecFaults::default());
        let (mut e, h) = faulty_engine(faults.clone(), lockstep::CacheDeadlines::default());
        let (t1, t2, _) = glm_turns();
        let cold2 = serve(&mut e, &t2, false).unwrap();
        faults.skip.store(true, SeqCst);
        serve(&mut e, &t1, true).unwrap();
        assert_eq!(e.prefix_stats().entries, 0, "the skipped snapshot never entered the index");
        assert_eq!(e.backend.prefix_free_snapshots(), 4, "rank 0 gave its slot back");
        assert!(e.backend.book.is_empty() && e.backend.snaps.is_empty());
        let hits = e.prefix_stats().hits;
        assert_eq!(serve(&mut e, &t2, true).unwrap().tokens, cold2.tokens);
        assert_eq!(e.prefix_stats().hits, hits, "nothing to hit");
        faults.skip.store(false, SeqCst);
        serve(&mut e, &t1, true).unwrap();
        assert!(e.prefix_stats().entries >= 1);
        assert_eq!(serve(&mut e, &t2, true).unwrap().tokens, cold2.tokens);
        assert_eq!(e.prefix_stats().hits, hits + 1);
        assert_eq!(e.backend.leader.failure(), None);
        drop(e);
        assert!(matches!(h.join().unwrap(), Ok(FollowerExit::Shutdown(_))));
    }

    /// A follower stuck in a restore (its reader answers pings): the restore fails over at once, and at the stall deadline the group is
    /// lost by name, not left waiting.
    #[test]
    fn a_stuck_glm_follower_restore_is_an_explicit_group_loss() {
        use std::sync::atomic::Ordering::SeqCst;
        let faults = Arc::new(RecFaults::default());
        let d = lockstep::CacheDeadlines { ack: Duration::from_millis(200), stall: Duration::from_millis(1_000) };
        let (mut e, h) = faulty_engine(faults.clone(), d);
        let (t1, t2, _) = glm_turns();
        serve(&mut e, &t1, true).unwrap();
        faults.restore_sleep_ms.store(600_000, SeqCst);
        let t0 = std::time::Instant::now();
        let r = serve(&mut e, &t2, true);
        let mut why = e.backend.leader.failure();
        while why.is_none() && t0.elapsed() < Duration::from_secs(20) {
            std::thread::sleep(Duration::from_millis(20));
            why = e.backend.leader.failure();
        }
        let why = why.expect("the group is lost");
        assert!(
            why.contains("rank 1") && why.contains("has not answered cache call") && why.contains("restore_prefix"),
            "{why} / {r:?}"
        );
        faults.restore_sleep_ms.store(0, SeqCst);
        drop(e);
        let _ = h.join();
    }

    #[test]
    fn an_image_is_refused_before_any_call_is_forwarded_for_it() {
        let (mut be, l0, l1, h) = group(3, false);
        let feats: Arc<dyn std::any::Any + Send + Sync> = Arc::new(0u8);
        let im = ImageSpan { start: 0, tokens: 1, merged: (1, 1), features: feats, key: 0 };
        let s = be.prefill_slice(0, &[1, 2], &[im], 0, 2);
        assert_eq!(s.done, 2);
        assert!(s.row.unwrap().values.iter().all(|v| v.is_nan()));
        let e = be.take_error().expect("a request error");
        assert!(!e.fatal, "{e:?}");
        drop(be);
        h.join().unwrap();
        // only the slot's release was forwarded: no begin, no prefill
        assert_eq!(l0.lock().unwrap().clone(), vec!["end 0".to_string()]);
        assert_eq!(l1.lock().unwrap().clone(), vec!["end 0".to_string()]);
    }

    /// Host features of `tokens` rows whose bytes are a function of `seed` and the row (no two rows alike).
    pub(crate) fn host(tokens: usize, seed: u32) -> Arc<dyn std::any::Any + Send + Sync> {
        use crate::families::glm::tp_vision::HostImageFeatures;
        let rows = (0..tokens * 4096).map(|i| (i as u32).wrapping_mul(2_654_435_761) as f32 * 1e-9 + seed as f32).collect();
        Arc::new(HostImageFeatures { rows, tokens, merged: (1, tokens) })
    }

    /// The fnv of rows `lo..hi` of host features, as the recording runtime logs a splice.
    pub(crate) fn rows_fnv(f: &Arc<dyn std::any::Any + Send + Sync>, lo: usize, hi: usize) -> String {
        let h = f.downcast_ref::<crate::families::glm::tp_vision::HostImageFeatures>().unwrap();
        let b: Vec<u8> = h.rows[lo * 4096..hi * 4096].iter().flat_map(|x| x.to_le_bytes()).collect();
        format!("{:016x}", fnv(&b))
    }

    pub(crate) fn vision() -> TpVision {
        TpVision {
            source: Box::new(crate::families::glm::tp_vision::HostFeatures),
            image_token: Some(7),
            frame_budget: crate::families::glm::tp_vision::MAX_IMAGE_BODY,
        }
    }

    pub(crate) fn slices(be: &mut dyn Backend, slot: usize, p: &[u32], images: &[ImageSpan], n: usize) -> RowOutput {
        let mut done = 0;
        loop {
            let s = be.prefill_slice(slot, p, images, done, n);
            done = s.done;
            if let Some(r) = s.row {
                return r;
            }
        }
    }

    /// The image path's call stream (checklist 3.1, item 5): two images (one straddling a chunk boundary, so rank 0 reads it from a row that
    /// is not its first), text after them and a chunk with no image row. The follower (which holds no tower: it is handed rows) replays rank
    /// 0's stream exactly, and the stream is the single-node backend's: `prefill_chunk_mm` exactly where `GlmBackend` makes it, with the
    /// image rows of THAT chunk, and the core `prefill_chunk` for a chunk with no image row.
    #[test]
    fn an_image_prompt_replays_on_a_follower_with_no_tower_and_is_the_single_node_backends_calls() {
        let (be, l0, l1, h) = group(3, false);
        let mut be = be.with_vision(vision());
        let (fa, fb) = (host(4, 1), host(2, 2));
        let ims = vec![
            ImageSpan { start: 1, tokens: 4, merged: (1, 4), features: fa.clone(), key: 0 },
            ImageSpan { start: 6, tokens: 2, merged: (1, 2), features: fb.clone(), key: 0 },
        ];
        let p = vec![1, 7, 7, 7, 7, 2, 7, 7, 3, 4, 5, 6];
        let r = slices(&mut be, 0, &p, &ims, 4);
        assert!(r.values.iter().all(|v| v.is_finite()));
        assert!(be.take_error().is_none());
        // rank 0's spans: the drafter would not absorb rows 0..2 (row 1 is an image row), it may from row 8
        assert!(
            !be.spec_allowed_for_window(0, 0, 2) && be.spec_allowed_for_window(0, 8, 4) && !be.spec_allowed_for_window(0, 7, 1)
        );
        let w = be.step(&[RowRequest { slot: 0, tokens: vec![9], first_position: 12 }], 4);
        assert!(w[0][0].values.iter().all(|v| v.is_finite()));
        be.commit(0, 1);
        be.release(0);
        assert!(be.spec_allowed_for_window(0, 0, 100), "a released slot has no image span");
        be.quiesce().unwrap();
        drop(be);
        assert!(matches!(h.join().unwrap(), FollowerExit::Shutdown(_)));
        let (a, b) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
        assert_eq!(a, b, "the follower's call stream is rank 0's");
        let want = vec![
            "begin 0".to_string(),
            format!("image 0 0 [1, 7, 7] None [(1, 2, \"{}\")]", rows_fnv(&fa, 0, 2)),
            format!("image 0 3 [7, 7, 2] None [(0, 2, \"{}\")]", rows_fnv(&fa, 2, 4)),
            format!("image 0 6 [7, 7, 3] None [(0, 2, \"{}\")]", rows_fnv(&fb, 0, 2)),
            "prefill 0 9 [4, 5, 6] Some(4)".to_string(),
            "window [9] [12] [0] 4".to_string(),
            "commit 0 1".to_string(),
            "end 0".to_string(),
        ];
        assert_eq!(a, want);
    }

    /// The ENGINE over rank 0's backend (`tp_boot::engine_over`, what `build_leader` serves with), a request with an image submitted as the
    /// server submits it: the image slices reach the follower as `PrefillImage` calls (no engine default bypasses the backend's image path),
    /// the request generates, and both ranks saw the same stream.
    #[test]
    fn the_engine_serves_an_image_request_through_the_group() {
        use crate::families::glm::tp_boot::engine_over;
        use gb_engine::Request;
        use gb_sampling::Sampling;
        use gb_sched::{Admission, ModelMemory};
        let (be, l0, l1, h) = group(4, false);
        let be = be.with_vision(vision());
        let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1, state_bytes_per_slot: 1, activation_reserve: 0 };
        let adm = Admission::new(1 << 30, 0, mm, 4).unwrap();
        let mut e = engine_over(be, adm, false);
        let f = host(6, 3);
        let prompt = vec![1, 2, 7, 7, 7, 7, 7, 7, 3, 4];
        let images = vec![ImageSpan { start: 2, tokens: 6, merged: (1, 6), features: f.clone(), key: 0 }];
        let id = e
            .submit(Request {
                prompt,
                sampling: Sampling::default(),
                max_new: 3,
                stop: vec![],
                draft: false,
                images,
                cache: false,
            })
            .unwrap();
        let mut done = false;
        for _ in 0..200 {
            e.step().unwrap();
            if e.take_events().iter().any(|ev| ev.id == id && ev.finish.is_some()) {
                done = true;
                break;
            }
        }
        assert!(done, "the image request did not finish");
        drop(e);
        h.join().unwrap();
        let (a, b) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
        assert_eq!(a, b, "the follower's call stream is rank 0's");
        let images: Vec<&String> = a.iter().filter(|s| s.starts_with("image")).collect();
        // the engine picks the slices (at most the backend's 4 rows): the first image call starts at the image's first row
        assert!(!images.is_empty() && images[0].starts_with("image 0 0 [1, 2, 7, 7]"), "{a:?}");
        assert!(a.iter().any(|s| (s.starts_with("image") || s.starts_with("prefill")) && s.contains("Some(")), "{a:?}");
        assert!(a.iter().any(|s| s.starts_with("window")), "the request generated: {a:?}");
    }

    /// Two images in ONE chunk, the chunk ending the prompt: one call, two splices in order, candidates asked.
    #[test]
    fn two_images_in_one_chunk_travel_in_one_call() {
        let (be, l0, _l1, h) = group(16, false);
        let mut be = be.with_vision(vision());
        let (fa, fb) = (host(3, 5), host(2, 6));
        let ims = vec![
            ImageSpan { start: 1, tokens: 3, merged: (1, 3), features: fa.clone(), key: 0 },
            ImageSpan { start: 5, tokens: 2, merged: (1, 2), features: fb.clone(), key: 0 },
        ];
        let p = vec![1, 7, 7, 7, 2, 7, 7, 3];
        slices(&mut be, 1, &p, &ims, 2);
        drop(be);
        h.join().unwrap();
        let a = l0.lock().unwrap().clone();
        assert_eq!(
            a[1],
            format!(
                "image 1 0 [1, 7, 7, 7, 2, 7, 7, 3] Some(2) [(1, 3, \"{}\"), (5, 2, \"{}\")]",
                rows_fnv(&fa, 0, 3),
                rows_fnv(&fb, 0, 2)
            )
        );
    }

    /// A chunk whose image rows would make a frame over `MAX_FRAME` (5,000 rows of 16 KiB: 78 MiB) is ended earlier, never truncated: the
    /// follower receives every row of the image, byte for byte, in calls that each fit a frame; nothing is lost and the group lives on.
    #[test]
    fn a_chunk_whose_rows_exceed_one_frame_is_split_and_every_row_arrives() {
        let (be, l0, l1, h) = group(5000, false);
        let mut be = be.with_vision(vision());
        let rows = 5000;
        let f = host(rows, 9);
        let mut p = vec![7u32; rows + 2];
        (p[0], p[rows + 1]) = (1, 2);
        let ims = vec![ImageSpan { start: 1, tokens: rows, merged: (1, rows), features: f.clone(), key: 0 }];
        let r = slices(&mut be, 0, &p, &ims, 2);
        assert!(r.values.iter().all(|v| v.is_finite()));
        assert!(be.take_error().is_none());
        // the group lives on: a text prompt after it
        let r = be.prefill(1, 0, &[3, 4], 2);
        assert!(r.values.iter().all(|v| v.is_finite()));
        be.quiesce().unwrap();
        drop(be);
        assert!(matches!(h.join().unwrap(), FollowerExit::Shutdown(_)));
        let b = l1.lock().unwrap().clone();
        assert_eq!(b, l0.lock().unwrap().clone());
        let images: Vec<&String> = b.iter().filter(|s| s.starts_with("image")).collect();
        // 5,002 rows: the first chunk of 5,000 rows (4,999 image rows, 78 MiB) is halved to 2,500 (39 MiB); the rest (2,502 rows, 2,501 of
        // them image rows) fits one frame and ends the prompt
        assert_eq!(images.len(), 2, "{:?}", b.iter().map(|s| &s[..s.len().min(40)]).collect::<Vec<_>>());
        assert!(
            images[0].starts_with("image 0 0 ")
                && images[0].contains(&format!("None [(1, 2499, \"{}\")]", rows_fnv(&f, 0, 2499)))
        );
        assert!(
            images[1].starts_with("image 0 2500 ")
                && images[1].contains(&format!("Some(2) [(0, 2501, \"{}\")]", rows_fnv(&f, 2499, 5000)))
        );
    }

    /// A call body over one frame is refused by name at the wire (`write_frame`), never cut: the backend measures every call so this
    /// never reaches a live link (which would lose the group).
    #[test]
    fn a_frame_over_max_frame_is_refused_by_name_at_the_wire() {
        use crate::families::glm::tp_vision::{PrefillImage, SpliceWire, ROW_BYTES};
        use crate::lockstep::{write_frame, MAX_FRAME};
        let rows = MAX_FRAME / ROW_BYTES + 1;
        let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: vec![7; rows], first_position: 0 }]);
        let c = PrefillImage {
            batch: b,
            want: None,
            splice: vec![SpliceWire { dst_row: 0, n: rows as u64, data: vec![0; rows * ROW_BYTES] }],
        };
        let mut sink: Vec<u8> = Vec::new();
        let e = write_frame(&mut sink, 1, &Msg::Call(GlmCall::PrefillImage(c))).unwrap_err().to_string();
        assert!(e.contains("exceeds the") && e.contains("byte limit"), "{e}");
        assert!(sink.is_empty(), "nothing of the frame was written");
    }

    /// Images to a group whose rank 0 has no tower: refused by name for the request, nothing forwarded but the release; images to a group
    /// that has one but with features this backend did not make: refused by name before anything is forwarded either.
    #[test]
    fn a_bad_image_is_refused_before_any_call_is_forwarded_for_it() {
        let (be, l0, _l1, h) = group(3, false);
        let mut be = be.with_vision(vision());
        let other: Arc<dyn std::any::Any + Send + Sync> = Arc::new(0u8);
        let im = ImageSpan { start: 1, tokens: 1, merged: (1, 1), features: other, key: 0 };
        let s = be.prefill_slice(0, &[1, 7], &[im], 0, 2);
        assert!(s.row.unwrap().values.iter().all(|v| v.is_nan()));
        let e = be.take_error().expect("a request error");
        assert!(!e.fatal && e.message.contains("not produced by this backend"), "{e:?}");
        drop(be);
        h.join().unwrap();
        assert_eq!(l0.lock().unwrap().clone(), vec!["end 0".to_string()]);
    }

    #[test]
    fn an_error_on_one_rank_only_loses_the_group_and_every_later_call_fails_fatally() {
        let (mut be, _l0, _l1, h) = group(3, true);
        let _ = be.prefill(0, 0, &[1, 2], 2);
        let r = be.step(&[RowRequest { slot: 0, tokens: vec![3], first_position: 2 }], 2);
        // rank 0 computed its window; the follower failed it: the group is lost at the ack, and the next fallible call says so
        let _ = r;
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while be.leader.failure().is_none() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        let why = be.leader.failure().expect("the group is lost");
        assert!(why.contains("window_candidates") || why.contains("rank 1"), "{why}");
        let r = be.step(&[RowRequest { slot: 0, tokens: vec![4], first_position: 3 }], 2);
        assert!(r[0][0].values.iter().all(|v| v.is_nan()));
        let e = be.take_error().expect("an error");
        assert!(e.fatal, "a lost group is fatal: {e:?}");
        drop(be);
        let _ = h.join();
    }

    #[test]
    fn a_lost_follower_fails_the_next_call_fatally_naming_the_rank() {
        let (a, b) = Link::<GlmCall>::pair(Duration::from_secs(5)).unwrap();
        let log: Log = Arc::default();
        let leader = TpLeader::lead(Rec::new(log.clone()), a, t());
        let mut be = GlmTpBackend::new(leader, 4, 4);
        drop(b); // the follower's process is gone
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while be.leader.failure().is_none() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        let r = be.prefill(0, 0, &[1], 2);
        assert!(r.values.iter().all(|v| v.is_nan()));
        let e = be.take_error().expect("an error");
        assert!(e.fatal && e.message.contains("rank 1"), "{e:?}");
    }

    #[test]
    fn the_boot_gate_refuses_another_pack_another_ownership_and_another_fact_by_name() {
        use crate::families::glm::model::ModelOptions;
        use crate::families::glm::tp::{gate, ownership_digest, range_table};
        let mo = ModelOptions { slots: 4, kv_blocks: 64, max_context: 8192, layers: 45 };
        let facts = GlmFacts {
            graphs: false,
            overlap: 0,
            graph_at_first: false,
            draft_graphs: false,
            prefill_small: 0,
            mtp: false,
            mtp_depth: 0,
            window_rows: 64,
            mtp_vocab: "none".into(),
            ownership: ownership_digest(&range_table(288, 2)),
            collective: "ring".into(),
            vision: false,
            split: 0,
            prefill: "contract=p1".into(),
            cache_request: "off".into(),
            cache_slots: 0,
            cache_slot_bytes: 0,
            cache_pages: 0,
            kv: "fp16".into(),
            taps: "none".into(),
        };
        let g0 = gate("aaaa".into(), &mo, 512, facts.clone()).at(0, 2);
        assert!(BootGate::check(&g0, &gate("aaaa".into(), &mo, 512, facts.clone()).at(1, 2)).is_ok());
        let e = BootGate::check(&g0, &gate("bbbb".into(), &mo, 512, facts.clone()).at(1, 2)).unwrap_err();
        assert!(e.contains("pack (config.json and safetensors index digest): rank 0 has aaaa, rank 1 has bbbb"), "{e}");
        let mut other = facts.clone();
        other.ownership = ownership_digest(&range_table(288, 3));
        let e = BootGate::check(&g0, &gate("aaaa".into(), &mo, 512, other).at(1, 2)).unwrap_err();
        assert!(e.contains("the expert ownership table (digest)"), "{e}");
        let mut other = facts.clone();
        other.graphs = true;
        let e = BootGate::check(&g0, &gate("aaaa".into(), &mo, 512, other).at(1, 2)).unwrap_err();
        assert!(e.contains("CUDA graphs (REXL3_GLM_GRAPHS): rank 0 has off, rank 1 has on"), "{e}");
        // lane E: graphs on (the exchange captured) with their branch parts and capture policy, each refused by name
        let on = facts.clone().with_graphs(true, 3, false);
        let g_on = gate("aaaa".into(), &mo, 512, on.clone()).at(0, 2);
        assert!(BootGate::check(&g_on, &gate("aaaa".into(), &mo, 512, on.clone()).at(1, 2)).is_ok());
        let e =
            BootGate::check(&g_on, &gate("aaaa".into(), &mo, 512, on.clone().with_graphs(true, 1, false)).at(1, 2)).unwrap_err();
        assert!(e.contains("the graph windows' layer branches (REXL3_GLM_OVERLAP): rank 0 has 3, rank 1 has 1"), "{e}");
        let e =
            BootGate::check(&g_on, &gate("aaaa".into(), &mo, 512, on.clone().with_graphs(true, 3, true)).at(1, 2)).unwrap_err();
        assert!(e.contains("the graph capture policy (REXL3_GLM_GRAPH_AT)"), "{e}");
        let e = BootGate::check(&g_on, &gate("aaaa".into(), &mo, 512, facts.clone()).at(1, 2)).unwrap_err();
        assert!(e.contains("CUDA graphs (REXL3_GLM_GRAPHS): rank 0 has on, rank 1 has off"), "{e}");
        // lane T: the NextN draft calls' graphs, a fact only with graphs on and the layer loaded; two ranks that differ are refused by name
        let mtp_on = facts.clone().with_mtp(3, "full".into()).with_graphs(true, 3, false);
        let d_on = mtp_on.clone().with_draft_graphs(true);
        assert!(d_on.draft_graphs && !facts.clone().with_graphs(true, 3, false).with_draft_graphs(true).draft_graphs);
        assert!(
            !mtp_on.clone().with_graphs(false, 3, false).with_draft_graphs(true).draft_graphs,
            "no draft graphs without graphs"
        );
        let g_d = gate("aaaa".into(), &mo, 512, d_on.clone()).at(0, 2);
        assert!(BootGate::check(&g_d, &gate("aaaa".into(), &mo, 512, d_on.clone()).at(1, 2)).is_ok());
        let e = BootGate::check(&g_d, &gate("aaaa".into(), &mo, 512, mtp_on.clone()).at(1, 2)).unwrap_err();
        assert!(e.contains("the NextN draft calls' graphs (REXL3_GLM_DRAFT_GRAPHS): rank 0 has on, rank 1 has off"), "{e}");
        // graphs off: the branch settings are not facts (nothing captured), so they cannot refuse a group
        assert_eq!(facts.clone().with_graphs(false, 3, true), facts.clone());
        let mut other = facts.clone();
        other.collective = "oneshot".into();
        let e = BootGate::check(&g0, &gate("aaaa".into(), &mo, 512, other).at(1, 2)).unwrap_err();
        assert!(e.contains("the collective policy"), "{e}");
        let e = BootGate::check(&g0, &gate("aaaa".into(), &mo, 1024, facts.clone()).at(1, 2)).unwrap_err();
        assert!(e.contains("prefill rows"), "{e}");
        // the prefill contract (Lane P): a default (P1) rank and a P2 rank are refused by name
        let mut other = facts.clone();
        other.prefill =
            crate::families::glm::contract::prefill_fingerprint_for(crate::families::glm::contract::PrefillContract::P2);
        let e = BootGate::check(&g0, &gate("aaaa".into(), &mo, 512, other.clone()).at(1, 2)).unwrap_err();
        assert!(
            e.contains("the prefill contract (REXL3_GLM_PREFILL_CONTRACT): rank 0 has contract=p1, rank 1 has contract=p2"),
            "{e}"
        );
        let p2_leader = gate("aaaa".into(), &mo, 512, other.clone()).at(0, 2);
        assert!(BootGate::check(&p2_leader, &gate("aaaa".into(), &mo, 512, other).at(1, 2)).is_ok(), "two P2 ranks agree");
        // a follower may hold more slots and pages, never fewer
        let more = ModelOptions { slots: 6, kv_blocks: 80, ..mo };
        assert!(BootGate::check(&g0, &gate("aaaa".into(), &more, 512, facts.clone()).at(1, 2)).is_ok());
        let less = ModelOptions { kv_blocks: 63, ..mo };
        let e = BootGate::check(&g0, &gate("aaaa".into(), &less, 512, facts.clone()).at(1, 2)).unwrap_err();
        assert!(e.contains("KV pages: rank 0 planned 64, rank 1 can hold only 63"), "{e}");
        // vision is a LEADER-ONLY fact: a leader with the tower over a follower without is the configuration; a follower with one is
        // refused by name, whatever the leader holds
        let tower = facts.clone().with_vision(true);
        let g_tower = gate("aaaa".into(), &mo, 512, tower.clone()).at(0, 2);
        assert!(BootGate::check(&g_tower, &gate("aaaa".into(), &mo, 512, facts.clone()).at(1, 2)).is_ok());
        for leader in [&g0, &g_tower] {
            let e = BootGate::check(leader, &gate("aaaa".into(), &mo, 512, tower.clone()).at(1, 2)).unwrap_err();
            assert!(
                e.contains("the vision tower (--no-vision)") && e.contains("rank 1 has one") && e.contains("only rank 0"),
                "{e}"
            );
        }
    }

    #[test]
    fn the_hello_carries_the_glm_facts_with_these_exact_bytes() {
        use crate::families::glm::model::ModelOptions;
        use crate::families::glm::tp::gate;
        let mo = ModelOptions { slots: 2, kv_blocks: 16, max_context: 4096, layers: 45 };
        let facts = GlmFacts {
            graphs: false,
            overlap: 0,
            graph_at_first: false,
            draft_graphs: false,
            prefill_small: 0,
            mtp: false,
            mtp_depth: 0,
            window_rows: 64,
            mtp_vocab: "none".into(),
            ownership: 0x0123_4567_89ab_cdef,
            collective: "ring".into(),
            vision: false,
            split: 0,
            prefill: "contract=p1".into(),
            cache_request: "off".into(),
            cache_slots: 0,
            cache_slot_bytes: 0,
            cache_pages: 0,
            kv: "fp16".into(),
            taps: "none".into(),
        };
        let mut g = gate("p".into(), &mo, 512, facts).at(1, 2);
        // the build and the NCCL fingerprint depend on the box: pin them for the golden
        g.build = "b".into();
        g.nccl = "n".into();
        let m: Msg<GlmCall> = Msg::Hello(g.clone());
        let bytes = m.encode(0);
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        let (_, back) = Msg::<GlmCall>::decode(&bytes[4..]).unwrap();
        assert_eq!(back, m, "round trip");
        assert_eq!(
            hex, GOLDEN_HELLO,
            "the GLM Hello's bytes moved: if on purpose, bump lockstep::PROTOCOL_VERSION's history note and update the golden"
        );
    }

    /// Lane E (2026-10-18): a Hello with graphs on carries the branch parts and the capture policy after the vision fact (two bytes:
    /// the Hello with graphs off keeps [`GOLDEN_HELLO`]'s bytes), and round-trips.
    #[test]
    fn the_hello_with_graphs_on_carries_the_branch_parts_and_the_capture_policy_last() {
        use crate::families::glm::model::ModelOptions;
        use crate::families::glm::tp::gate;
        let mo = ModelOptions { slots: 2, kv_blocks: 16, max_context: 4096, layers: 45 };
        let facts = GlmFacts {
            graphs: false,
            overlap: 0,
            graph_at_first: false,
            draft_graphs: false,
            prefill_small: 0,
            mtp: false,
            mtp_depth: 0,
            window_rows: 64,
            mtp_vocab: "none".into(),
            ownership: 0x0123_4567_89ab_cdef,
            collective: "ring".into(),
            vision: false,
            split: 0,
            prefill: "contract=p1".into(),
            cache_request: "off".into(),
            cache_slots: 0,
            cache_slot_bytes: 0,
            cache_pages: 0,
            kv: "fp16".into(),
            taps: "none".into(),
        };
        let enc = |f: GlmFacts| {
            let mut g = gate("p".into(), &mo, 512, f).at(1, 2);
            g.build = "b".into();
            g.nccl = "n".into();
            let m: Msg<GlmCall> = Msg::Hello(g);
            let bytes = m.encode(0);
            let (_, back) = Msg::<GlmCall>::decode(&bytes[4..]).unwrap();
            assert_eq!(back, m, "round trip");
            bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
        };
        assert_eq!(enc(facts.clone()), GOLDEN_HELLO);
        let on = enc(facts.clone().with_graphs(true, 3, true));
        // length + 2, graphs 01, and `03 01` after the vision byte (`00` after `ring`; Lane P's prefill fact follows them)
        let off = GOLDEN_HELLO;
        assert_eq!(u8::from_str_radix(&on[..2], 16).unwrap(), u8::from_str_radix(&off[..2], 16).unwrap() + 2);
        let at = off.find("72696e6700").unwrap() + 10;
        assert_eq!(&on[at..at + 4], "0301", "{on}");
        assert_eq!(&on[at + 4..], &off[at..]);
        // lane T: the draft graphs are bit 2 of the first fact byte (graphs on), and the Hello round-trips; without graphs nothing changes
        let draft = |f: GlmFacts| GlmFacts { draft_graphs: true, ..f };
        assert_eq!(enc(facts.clone().with_draft_graphs(true)), GOLDEN_HELLO);
        let on_d = enc(draft(facts.clone().with_graphs(true, 3, true)));
        assert_eq!(on_d.len(), on.len());
        assert_ne!(on_d, on, "the draft graphs bit is on the wire");
        // lane E, MAIN's layout (2026-10-19): a head split is bit 3 of the first fact byte and one byte after every other fact; with the
        // draft graphs (bit 2) and graphs on (bit 0) the first fact byte is 0x0d and the split byte follows the graph settings
        use crate::families::glm::model::tp::{SPLIT_KDA, SPLIT_LM, SPLIT_MLA};
        let bits = SPLIT_KDA | SPLIT_LM | SPLIT_MLA;
        let changed = |a: &str, b: &str| -> Vec<(usize, String, String)> {
            (0..a.len() / 2)
                .filter(|&i| a[2 * i..2 * i + 2] != b[2 * i..2 * i + 2])
                .map(|i| (i, a[2 * i..2 * i + 2].into(), b[2 * i..2 * i + 2].into()))
                .collect()
        };
        let sp = enc(facts.clone().with_split(bits));
        // (the lengths relative to GOLDEN_HELLO's: Lane P's prefill fact, the last field, follows the split byte)
        let glen = u8::from_str_radix(&GOLDEN_HELLO[..2], 16).unwrap();
        assert_eq!(&sp[..2], format!("{:02x}", glen + 1), "one byte longer");
        assert_eq!(&sp[at..at + 2], "07", "the split byte right after the vision fact (graphs off)");
        let without = format!("{}{}", &sp[..at], &sp[at + 2..]);
        let d = changed(&without, GOLDEN_HELLO);
        assert_eq!(d.len(), 2, "the length and the first fact byte: {d:?}");
        assert_eq!((d[1].1.as_str(), d[1].2.as_str()), ("08", "00"), "bit 3: {d:?}");
        let all = enc(draft(facts.clone().with_graphs(true, 3, true)).with_split(bits));
        assert_eq!(&all[..2], format!("{:02x}", glen + 3));
        assert_eq!(&all[at..at + 6], "030107", "graph settings, then the split byte: {all}");
        let d = changed(&format!("{}{}", &all[..at + 4], &all[at + 6..]), &on_d);
        assert_eq!(d.len(), 2, "{d:?}");
        assert_eq!((d[1].1.as_str(), d[1].2.as_str()), ("0d", "05"), "bits 0, 2 and 3: {d:?}");
    }

    /// The prefix cache's arena (`docs/GLM_TP.md` 4.2): with the cache off the Hello keeps [`GOLDEN_HELLO`]'s bytes, captured before the
    /// cache's facts existed; with it on, bit 4 of the first fact byte is set and the request and the three counts follow the prefill
    /// contract (`mib=3584` as a u64 length and its 8 bytes, then the snapshot slots, a slot's bytes and the pages as u64s: 40 bytes
    /// more), nothing else moves, and the Hello round-trips. A flags byte with a bit above bit 4 is refused by name.
    #[test]
    fn the_hello_with_the_prefix_cache_off_keeps_its_bytes_and_with_it_on_carries_bit_4_and_the_arena_last() {
        use crate::families::glm::model::ModelOptions;
        use crate::families::glm::tp::gate;
        use crate::lockstep::{Dec, Enc, FamilyCall};
        let mo = ModelOptions { slots: 2, kv_blocks: 16, max_context: 4096, layers: 45 };
        let facts = GlmFacts {
            graphs: false,
            overlap: 0,
            graph_at_first: false,
            draft_graphs: false,
            prefill_small: 0,
            mtp: false,
            mtp_depth: 0,
            window_rows: 64,
            mtp_vocab: "none".into(),
            ownership: 0x0123_4567_89ab_cdef,
            collective: "ring".into(),
            vision: false,
            split: 0,
            prefill: "contract=p1".into(),
            cache_request: "off".into(),
            cache_slots: 0,
            cache_slot_bytes: 0,
            cache_pages: 0,
            kv: "fp16".into(),
            taps: "none".into(),
        };
        let enc = |f: GlmFacts| {
            let mut g = gate("p".into(), &mo, 512, f).at(1, 2);
            g.build = "b".into();
            g.nccl = "n".into();
            let m: Msg<GlmCall> = Msg::Hello(g);
            let bytes = m.encode(0);
            let (_, back) = Msg::<GlmCall>::decode(&bytes[4..]).unwrap();
            assert_eq!(back, m, "round trip");
            bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
        };
        assert_eq!(enc(facts.clone()), GOLDEN_HELLO, "the Hello with the cache off moved");
        let budget =
            gb_engine::CacheBudget { snapshot_slots: 4, page_blocks: 14, snapshot_bytes: 73_828_352, page_bytes: 1 << 20 };
        let on = enc(facts.clone().with_cache("mib=3584".into(), &budget));
        let glen = u8::from_str_radix(&GOLDEN_HELLO[..2], 16).unwrap();
        assert_eq!(&on[..2], format!("{:02x}", glen + 40), "40 bytes longer");
        let contract = "636f6e74726163743d7031"; // `contract=p1`, the field before the arena
        let at = GOLDEN_HELLO.find(contract).unwrap() + contract.len();
        let le = |v: u64| v.to_le_bytes().iter().map(|b| format!("{b:02x}")).collect::<String>();
        let arena = format!("{}{}{}{}{}", le(8), "6d69623d33353834", le(4), le(73_828_352), le(14));
        assert_eq!(&on[at..at + arena.len()], arena, "{on}");
        assert_eq!(&on[at + arena.len()..], &GOLDEN_HELLO[at..], "the build, the rank and the NCCL setting follow as before");
        let d: Vec<(usize, &str, &str)> = (0..at / 2)
            .filter(|&i| on[2 * i..2 * i + 2] != GOLDEN_HELLO[2 * i..2 * i + 2])
            .map(|i| (i, &on[2 * i..2 * i + 2], &GOLDEN_HELLO[2 * i..2 * i + 2]))
            .collect();
        assert_eq!(d.len(), 2, "the length and the first fact byte: {d:?}");
        assert_eq!((d[1].1, d[1].2), ("10", "00"), "bit 4: {d:?}");
        let mut e = Enc::new();
        GlmCall::encode_facts(&facts, &mut e);
        e.0[0] = 0x80;
        let err = GlmCall::decode_facts(&mut Dec::new(&e.0)).unwrap_err().to_string();
        assert!(err.contains("flags 0x80") && err.contains("bits 0 to 6 only"), "{err}");
        // an fp8 latent cache (`--kv-dtype fp8`): bit 5 and the format's name after every other fact (`fp8` as a u64 length and 3 bytes:
        // 11 bytes more), nothing else moves, the Hello round-trips, and a group whose ranks disagree is refused by name
        let fp8 = GlmFacts { kv: "fp8".into(), ..facts.clone() };
        let on8 = enc(fp8.clone());
        assert_eq!(&on8[..2], format!("{:02x}", glen + 11), "11 bytes longer");
        assert_eq!(&on8[at..at + 22], format!("{}{}", le(3), "667038"), "{on8}");
        assert_eq!(&on8[at + 22..], &GOLDEN_HELLO[at..]);
        let d: Vec<(usize, &str, &str)> = (0..at / 2)
            .filter(|&i| on8[2 * i..2 * i + 2] != GOLDEN_HELLO[2 * i..2 * i + 2])
            .map(|i| (i, &on8[2 * i..2 * i + 2], &GOLDEN_HELLO[2 * i..2 * i + 2]))
            .collect();
        assert_eq!(d.len(), 2, "the length and the first fact byte: {d:?}");
        assert_eq!((d[1].1, d[1].2), ("20", "00"), "bit 5: {d:?}");
        let leader8 = gate("p".into(), &mo, 512, fp8.clone()).at(0, 2);
        let e = BootGate::check(&leader8, &gate("p".into(), &mo, 512, facts.clone()).at(1, 2)).unwrap_err();
        assert!(e.contains("the latent cache format (--kv-dtype / REXL3_GLM_KV; pass the same to every rank)"), "{e}");
        assert!(e.contains("rank 0 has fp8, rank 1 has fp16"), "{e}");
        assert!(BootGate::check(&leader8, &gate("p".into(), &mo, 512, fp8).at(1, 2)).is_ok(), "two fp8 ranks agree");
        // a drafting group (`--draft`): bit 6 and the tapped layers after every other fact (`5,14` as a u64 length and 4 bytes: 12 bytes
        // more), nothing else moves, and ranks that tap other layers (or one without the drafter) are refused by name
        let taps = facts.clone().with_taps(&[5, 14]);
        let ont = enc(taps.clone());
        assert_eq!(&ont[..2], format!("{:02x}", glen + 12), "12 bytes longer");
        assert_eq!(&ont[at..at + 24], format!("{}{}", le(4), "352c3134"), "{ont}");
        assert_eq!(&ont[at + 24..], &GOLDEN_HELLO[at..]);
        let lt = gate("p".into(), &mo, 512, taps.clone()).at(0, 2);
        let e = BootGate::check(&lt, &gate("p".into(), &mo, 512, facts.clone()).at(1, 2)).unwrap_err();
        assert!(e.contains("the DFlash2 drafter's taps (--draft; pass the same to every rank)") && e.contains("5,14"), "{e}");
        assert!(BootGate::check(&lt, &gate("p".into(), &mo, 512, taps).at(1, 2)).is_ok(), "two ranks on the same taps agree");
    }

    /// Captured 2026-10-14 (lane L2b): length 0x73, protocol 4, kind 1 (Hello), seq 0, `RXTP`, pack `p`, layers 45, slots 2, KV pages 16,
    /// context 4096, prefill rows 512, then the GLM facts (graphs 0, mtp 0, window 64, `none`, the digest little-endian, `ring`), build
    /// `b`, rank 1, ranks 2, NCCL `n`. Lane L13 (2026-10-14) appended the vision fact (one byte, `00`, after `ring`): length 0x74. A GLM
    /// Hello of the build before it does not decode against this one (the body is one byte short): the ranks run different builds.
    /// Lane P (2026-10-18) appended the prefill contract fact after the vision byte (the string `contract=p1`: its u64 length 11 and 11
    /// bytes): length 0x87; every other byte is L13's. The protocol version is unchanged (as for L13's fact: only GLM's facts moved).
    const GOLDEN_HELLO: &str = "8700000005010000000000000000525854500100000000000000702d00000002000000100000000000000000100000000000000002000000004000000004000000000000006e6f6e65efcdab8967452301040000000000000072696e67000b00000000000000636f6e74726163743d7031010000000000000062010000000200000001000000000000006e";

    /// The wire bytes of a `PrefillImage` (kind 40): a chunk of 3 rows of slot 1 from position 5, `want` 2, one splice of one row at chunk
    /// row 1 whose 16 KiB are a ramp (the golden pins the head and the length, then the FNV of the whole frame).
    #[test]
    fn the_prefill_image_call_has_these_exact_bytes() {
        use crate::families::glm::tp_vision::{PrefillImage, SpliceWire, ROW_BYTES};
        let b = RowBatch::from_requests(&[RowRequest { slot: 1, tokens: vec![10, 7, 11], first_position: 5 }]);
        let data: Vec<u8> = (0..ROW_BYTES).map(|i| (i % 251) as u8).collect();
        let m: Msg<GlmCall> = Msg::Call(GlmCall::PrefillImage(PrefillImage {
            batch: b,
            want: Some(2),
            splice: vec![SpliceWire { dst_row: 1, n: 1, data }],
        }));
        let bytes = m.encode(7);
        let (seq, back) = Msg::<GlmCall>::decode(&bytes[4..]).unwrap();
        assert_eq!((seq, &back), (7, &m), "round trip");
        let head: String = bytes[..bytes.len() - ROW_BYTES].iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            (head.as_str(), bytes.len(), fnv(&bytes)),
            (GOLDEN_IMAGE_HEAD, GOLDEN_IMAGE_LEN, GOLDEN_IMAGE_FNV),
            "the PrefillImage frame's bytes moved: if on purpose, update the golden and docs/GLM_TP.md 10"
        );
    }

    /// Captured 2026-10-14 (lane L13): payload length 0x4091, protocol 4, kind 0x28 (40), seq 7; the batch (3 rows: ids 10 7 11,
    /// positions 5 6 7, row slots 1 1 1, seq_start 0 3, seq_slot 1, no rope3, no parents); `want` Some(2); one splice: dst_row 1, n 1,
    /// 0x4000 bytes (then the ramp, covered by the length and the FNV of the whole frame).
    const GOLDEN_IMAGE_HEAD: &str = "9140000005280700000000000000030000000000000003000000000000000a000000070000000b0000000300000000000000050000000600000007000000030000000000000001000000010000000100000002000000000000000000000003000000010000000000000001000000000001020000000100000000000000010000000000000001000000000000000040000000000000";
    const GOLDEN_IMAGE_LEN: usize = 16533;
    const GOLDEN_IMAGE_FNV: u64 = 8304697039780532213;
}
