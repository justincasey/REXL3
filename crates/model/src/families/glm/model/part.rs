//! Prompt forwards of several prompts, and prompt forwards run in parts.
//!
//! **Several prompts in one forward** (P1 and P2). Every op of a prompt forward is row-independent (P1: every row its serial decode
//! step's bits, chunk-grid independent; P2: `contract.rs`, batched == solo, chunk independence) except the ones that read a sequence's
//! own earlier rows: the KDA recurrence and its conv history (run per sequence on its own slot and rows, `model.rs`'s `kda_segs`: P1's
//! row-by-row path, P2's chunked one), the MLA latents, the DSA rings and the attention (indexed per row by its `(slot, position)`), the
//! drafters' contexts (absorbed per sequence from its own tap rows) and the head (each sequence's last row). So a forward of the
//! consecutive prompt rows of several sequences gives each row the bits of its own prompt's forward alone, and reads the weights once
//! for all of them. Under P2 each sequence joins (or opens) its own P2 segment, as it would alone.
//!
//! **A forward in parts.** [`GlmModel::prefill_rows`] with a layer bound runs the layers before it and PAUSES at that layer's start;
//! [`GlmModel::prefill_resume`] runs the next layers. Between two layers a prompt forward's whole state is its rows of the mHC streams,
//! the partials of the next site (`apply_partials` took them) and the drafter's tap rows of the layers already run: everything else is
//! written by the layer that reads it, or is the slot's own state (KDA pools, latents, rings), which no other forward touches. While a
//! forward is paused, windows (decode rounds) and whole prompt forwards of other slots may run: they write the scratch's first rows only,
//! so the paused forward's first [`GlmModel::prefill_part_rows`] rows of those three buffers are copied aside at the pause and back at the
//! resume (after the commits of the windows that ran meanwhile, which read their own tap rows). A part is the same launches on the same
//! inputs as the whole forward's, so the pause moves no bit; nothing that runs meanwhile reads or writes the paused slot.

use super::{Begun, GlmModel, Head, Mode};
use crate::error::ModelError;
use crate::forward::Candidates;
use crate::rows::RowBatch;
use gb_cuda::{mhc, DevBuf};

use super::super::ops::{Op, Site, SiteKind};

/// A prompt forward paused at the start of a layer.
pub(in crate::families::glm) struct Paused {
    b: RowBatch,
    plan: Vec<Op>,
    max_pos: usize,
    segs: Vec<(usize, usize)>,
    head: Head,
    want: Option<usize>,
    /// The op the next part starts at (a layer's first, its attention site's `Finalize`).
    next: usize,
    skip_taps: bool,
}

/// The copies of a paused forward's first rows (see the module).
pub(in crate::families::glm) struct Stash {
    rows: usize,
    streams: DevBuf,
    part: DevBuf,
    taps: Option<DevBuf>,
}

/// The index of the first op of layer `layer` in `plan` (its attention site's `Finalize`), or the plan's end when it has no such layer.
fn layer_op(plan: &[Op], layer: usize) -> usize {
    plan.iter().position(|op| *op == Op::Finalize(Site { layer, kind: SiteKind::Attn })).unwrap_or(plan.len())
}

impl GlmModel {
    /// Let prompt forwards run in parts (`prefill_rows` with a layer bound): the stash of `rows` rows (at least a window's) of the streams,
    /// the mHC partials and the drafter's taps. Returns its bytes. Every rank of a group enables the same rows (the leader's parts are
    /// calls every rank runs).
    pub fn enable_prefill_parts(&mut self, rows: usize) -> Result<usize, ModelError> {
        let rows = rows.max(self.window_rows).min(self.s.rows);
        let part_row = self.s.mhc.part.bytes() / self.s.mhc.rows_cap.max(1);
        let tap_row = self.dflash_tap_row_bytes();
        let streams = DevBuf::zeroed_view(rows * mhc::ROW * 4)?;
        let part = DevBuf::zeroed_view(rows * part_row)?;
        let taps = if tap_row > 0 { Some(DevBuf::zeroed_view(rows * tap_row)?) } else { None };
        let bytes = streams.bytes() + part.bytes() + taps.as_ref().map_or(0, |t| t.bytes());
        self.stash = Some(Stash { rows, streams, part, taps });
        Ok(bytes)
    }

    /// The rows another forward may have while a prompt forward is paused (0: parts are off).
    pub fn prefill_part_rows(&self) -> usize {
        self.stash.as_ref().map_or(0, |s| s.rows)
    }

    /// The batch of the prompt forward paused at a layer boundary, if one is.
    pub fn prefill_paused(&self) -> Option<&RowBatch> {
        self.paused.as_ref().map(|p| &p.b)
    }

    /// Forget a paused prompt forward of `slot` (its slot ended or begins again: the state it left in the slot is the slot's to reset).
    pub(super) fn drop_paused_of(&mut self, slot: usize) {
        if self.paused.as_ref().is_some_and(|p| p.b.seq_slot.contains(&(slot as i32))) {
            self.paused = None;
        }
    }

    /// A forward of several prompts (see the module): KDA through the row-by-row (P1) or chunked (P2) prompt path, no snapshot inside it,
    /// no image, no forced routing, no capture, every sequence in its own slot.
    pub(super) fn check_prompts(&self, b: &RowBatch) -> Result<(), ModelError> {
        let why = if !self.snap_cuts.is_empty() {
            Some("takes no snapshot inside itself")
        } else if !self.splice.is_empty() {
            Some("carries no image")
        } else if self.forced.is_some() || self.capture.is_some() {
            Some("runs without forced routing or a capture")
        } else {
            let mut slots: Vec<i32> = b.seq_slot.clone();
            slots.sort_unstable();
            slots.dedup();
            (slots.len() != b.n_seq()).then_some("holds each sequence once")
        };
        match why {
            Some(w) => Err(ModelError::Request(format!("a prefill forward of {} prompts {w}", b.n_seq()))),
            None => Ok(()),
        }
    }

    /// While a forward is paused, what runs must fit the stash and stay off the paused slots.
    pub(super) fn check_room(&self, b: &RowBatch) -> Result<(), ModelError> {
        let Some(p) = self.paused.as_ref() else { return Ok(()) };
        let room = self.prefill_part_rows();
        if b.rows > room {
            return Err(ModelError::Request(format!(
                "a forward of {} rows while a prompt forward is paused (at most {room} rows run meanwhile)",
                b.rows
            )));
        }
        if let Some(sl) = b.seq_slot.iter().find(|s| p.b.seq_slot.contains(s)) {
            return Err(ModelError::Request(format!("a forward on slot {sl} while its prompt forward is paused")));
        }
        Ok(())
    }

    /// Prefill `b`: the consecutive prompt rows of one sequence or (under P2) several, each continuing its own prompt. With `until` at or
    /// past the model's layers the whole forward runs, its rows join the drafters' contexts, and with `want` the candidates of each
    /// sequence's LAST row come back, in sequence order (each sequence's slice must then end its prompt for them to be used). With a
    /// smaller `until` (one sequence, parts enabled, none paused) only the layers before `until` run and the forward PAUSES there:
    /// `Ok(None)`, finished by [`GlmModel::prefill_resume`].
    pub fn prefill_rows(&mut self, b: &RowBatch, want: Option<usize>, until: usize) -> Result<Option<Candidates>, ModelError> {
        let whole = until >= self.layers;
        if !whole && (b.n_seq() != 1 || self.paused.is_some()) {
            return Err(ModelError::Request("a prompt forward in parts is of one sequence, with no other paused".into()));
        }
        if !whole && self.stash.is_none() {
            return Err(ModelError::Request("prompt forwards in parts are not enabled (enable_prefill_parts)".into()));
        }
        let head = if want.is_some() { Head::LastRow } else { Head::None };
        self.dflash_begin_rows(b);
        let f = match self.forward_begin(b, Mode::Prefill, head) {
            Ok(f) => f,
            Err(e) => {
                // the next forward taps again
                self.dflash_end_chunk();
                return Err(e);
            }
        };
        let end = if whole { f.plan.len() } else { layer_op(&f.plan, until) };
        if let Err(e) = self.run(&f.plan[..end], b, Mode::Prefill, head, f.max_pos) {
            self.dflash_end_chunk();
            return Err(e);
        }
        let Begun { plan, max_pos, segs } = f;
        if end < plan.len() {
            let skip_taps = self.dflash_skips_taps();
            self.pause(Paused { b: b.clone(), plan, max_pos, segs, head, want, next: end, skip_taps })?;
            return Ok(None);
        }
        self.forward_end(b, Mode::Prefill, &segs);
        self.prefill_finish(b, want)
    }

    /// Run the paused prompt forward's layers up to `until` (at or past the model's layers: to its end, with its drafters' absorbs and
    /// its candidates, as [`GlmModel::prefill_rows`] gives them); before `until`'s layer it pauses again (`Ok(None)`). An error drops the
    /// paused forward: its slot's state is then the slot's to begin again.
    pub fn prefill_resume(&mut self, until: usize) -> Result<Option<Candidates>, ModelError> {
        let mut p = self.paused.take().ok_or_else(|| ModelError::Request("no prompt forward is paused".into()))?;
        // the windows that ran meanwhile are committed first (their commits read their own index arrays and tap rows), then the paused
        // forward's rows come back and its row arrays and pages are bound again
        self.flush()?;
        self.restore_stash(p.b.rows)?;
        self.upload_rows(&p.b)?;
        self.ensure_pages(&p.b, true)?;
        self.dflash_set_skip_taps(p.skip_taps);
        let end = if until >= self.layers { p.plan.len() } else { layer_op(&p.plan, until).max(p.next) };
        if let Err(e) = self.run(&p.plan[p.next..end], &p.b, Mode::Prefill, p.head, p.max_pos) {
            self.dflash_end_chunk();
            return Err(e);
        }
        if end < p.plan.len() {
            p.next = end;
            self.pause(p)?;
            return Ok(None);
        }
        self.forward_end(&p.b, Mode::Prefill, &p.segs);
        self.prefill_finish(&p.b, p.want)
    }

    /// The end of a whole prompt forward: each sequence's rows join the NextN drafter's book and DFlash2's context (from its own tap rows),
    /// then the candidates of the head's rows (or a sync).
    fn prefill_finish(&mut self, b: &RowBatch, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        for i in 0..b.n_seq() {
            let r = b.seq_rows(i);
            let (slot, start) = (b.seq_slot[i] as usize, b.positions[r.start] as usize);
            let tokens: Vec<u32> = b.ids[r.clone()].iter().map(|&t| t as u32).collect();
            self.mtp_prefill_rows(slot, start, &tokens, r.start)?;
            self.dflash_prefill_rows(slot, start, r.len(), r.start)?;
        }
        match want {
            None => {
                self.stream.sync()?;
                Ok(None)
            }
            Some(count) => self.candidates_of(b.n_seq(), count).map(Some),
        }
    }

    /// Keep `p` paused: its first rows of the streams, the partials and the taps copied aside, the taps on again for what runs meanwhile.
    /// Synchronises (a part ends when its work has, as a whole prompt forward does).
    fn pause(&mut self, p: Paused) -> Result<(), ModelError> {
        let n = p.b.rows.min(self.prefill_part_rows());
        let part_row = self.s.mhc.part.bytes() / self.s.mhc.rows_cap.max(1);
        let tap_row = self.dflash_tap_row_bytes();
        let GlmModel { stash, s, dflash, stream, .. } = self;
        let sh = stash.as_mut().ok_or_else(|| ModelError::Request("prompt forwards in parts are not enabled".into()))?;
        sh.streams.copy_from_at(0, &s.streams, 0, n * mhc::ROW * 4, stream)?;
        sh.part.copy_from_at(0, &s.mhc.part, 0, n * part_row, stream)?;
        if let (Some(t), Some(d)) = (sh.taps.as_mut(), dflash.as_ref()) {
            t.copy_from_at(0, &d.taps, 0, n * tap_row, stream)?;
        }
        self.dflash_end_chunk();
        self.stream.sync()?;
        self.paused = Some(p);
        Ok(())
    }

    /// The stashed rows of a forward of `rows` rows back into the scratch.
    fn restore_stash(&mut self, rows: usize) -> Result<(), ModelError> {
        let n = rows.min(self.prefill_part_rows());
        let part_row = self.s.mhc.part.bytes() / self.s.mhc.rows_cap.max(1);
        let tap_row = self.dflash_tap_row_bytes();
        let GlmModel { stash, s, dflash, stream, .. } = self;
        let sh = stash.as_ref().ok_or_else(|| ModelError::Request("prompt forwards in parts are not enabled".into()))?;
        s.streams.copy_from_at(0, &sh.streams, 0, n * mhc::ROW * 4, stream)?;
        s.mhc.part.copy_from_at(0, &sh.part, 0, n * part_row, stream)?;
        if let (Some(t), Some(d)) = (sh.taps.as_ref(), dflash.as_mut()) {
            d.taps.copy_from_at(0, t, 0, n * tap_row, stream)?;
        }
        Ok(())
    }
}
