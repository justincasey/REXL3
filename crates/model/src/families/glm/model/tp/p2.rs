//! The routed experts of a P2 prompt chunk on a tensor-parallel rank (Lane P step (c); the design is `docs/GLM_PREFILL_P2.md` section 7).
//!
//! Under P2 a row's routed sum is defined over the two expert groups of the TP=2 ownership (`gb_cuda::glm_p2_moe`, the combine's
//! order: `part_g` per group, then `(part_0 + part_1) + shared`). Rank r owns exactly group r, so the rank computes its own pairs only,
//! sums them into `part_r` (`CombineMode::Own(r)`), swaps that ONE fp32 row per prompt row with the peer (`TpComm::swap_bytes`: a byte
//! copy, no arithmetic, 8x fewer bytes than the pair all-reduce of P1), and both ranks add `part_0 + part_1 + shared` in rank order
//! (`CombineMode::Parts`): the operations of one node's `CombineMode::Both` on the same values, so P2 at TP=2 equals P2 on one node bit
//! for bit. The swap runs on its own stream, ordered after the rank's part by an event, while the main stream computes the layer's shared
//! expert (deferred to that point by the caller: it is replicated and only the final sum reads it); the sum waits on the swap's event.
//! A group of one rank (`SingleRank`) holds every expert: its pairs, then the one-node combine.
//!
//! A mixed-width layer's share on a rank is several subbanks ([`P2Routed::Subbanks`]): the rank's pairs are the P2 pairs stage once per
//! subbank (`glm_p2_moe::pairs_subbanks_rank`; a split gate/up subbank through the decode pairs stage in windows of the decode scratch),
//! into the one pair buffer, then the same `Own(r)` part, swap and `Parts` sum: each pair's bits are its row's and its expert's alone, so
//! the rank's part is the one node's group-r part bit for bit.
//!
//! **The expert width split** (`gb_cuda::moe_width`, `REXL3_GLM_TP_EXPERTS=width`, the default). A rank holds every expert at its half
//! of the width, so its pairs are EVERY pair at that half (the P2 GEMMs on its gate/up columns and its down K rows, the down epilogue on
//! that half's sum) and its part is every slot's (`CombineMode::All`); the swap and the `Parts` sum are unchanged. The parts are then the
//! two halves of the width, not the two expert groups: every P2 invariant holds (a pair's bits are its row's and its expert's alone,
//! nothing depends on the chunk), but P2 at TP=2 no longer equals P2 on one node bit for bit (`moe_width::P2_WIDTH_ID` names the
//! arithmetic).
//!
//! **Row split** ([`RowSplit`], a trunk MoE layer of a chunk of at least two 128-row slabs). Every P2 op is row-independent, so the
//! replicated per-row work after the part need not be replicated: rank r sends the peer only the peer's rows of its part, computes the
//! shared expert and the parts' sum on its OWN rows (`glm_p2_moe::combine_parts_rows`), and the ranks swap those output rows (byte
//! copies). The same bytes cross as in the full swap (half a part each way, then half the output each way: 16 KiB a row in all), every
//! output row is the operations of the full sum on the same values, so the bits are the one node's. The MLA layers of the chunk split
//! their per-row tail the same way (`layers::mla_dsa_mixer`, [`GlmTp::swap_rows`]).

use super::GlmTp;
use crate::error::ModelError;
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::glm_p2_moe::{self, CombineMode};
use gb_cuda::moe::{MoeDims, MoeExperts, MoeScratch};
use gb_cuda::moe_prefill::PrefillScratch;
use gb_cuda::moe_subbanks::MoeSubbanks;
use gb_cuda::rt::Event;
use gb_cuda::{CudaError, DevBuf, Stream};

/// A prompt chunk's rows cut between the two ranks of a group: this rank's `[lo, hi)` and the peer's `[plo, phi)` (rank 0 takes the first
/// half of the 128-row slabs, rank 1 the rest; both ranks derive the same cut from the rows).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::families::glm) struct RowSplit {
    pub lo: usize,
    pub hi: usize,
    pub plo: usize,
    pub phi: usize,
}

impl RowSplit {
    /// The cut of `rows` rows for `rank` of two: `None` below two slabs of `slab` rows (one rank would get nothing).
    pub(in crate::families::glm) fn of(rows: usize, slab: usize, rank: usize) -> Option<Self> {
        if rows < 2 * slab || rank > 1 {
            return None;
        }
        let m = (rows.div_ceil(slab).div_ceil(2) * slab).min(rows);
        let ((lo, hi), (plo, phi)) = if rank == 0 { ((0, m), (m, rows)) } else { ((m, rows), (0, m)) };
        Some(Self { lo, hi, plo, phi })
    }

    pub(in crate::families::glm) fn rows(&self) -> usize {
        self.hi - self.lo
    }
}

/// The routed experts of a P2 MoE layer on a rank: one bank (the whole bank, masked to the share, or the share bound as one subbank), or
/// the share's several subbanks (a mixed-width layer).
#[derive(Clone, Copy)]
pub(in crate::families::glm) enum P2Routed<'a> {
    Bank(&'a MoeExperts),
    Subbanks(&'a MoeSubbanks),
}

impl P2Routed<'_> {
    pub(in crate::families::glm) fn dims(&self) -> MoeDims {
        match self {
            P2Routed::Bank(b) => b.dims,
            P2Routed::Subbanks(sb) => sb.dims,
        }
    }
}

/// The P2 exchange's buffers and ordering: the two groups' parts (`parts[g]`: group g's rows, the rank's own written by its combine, the
/// peer's by the swap), the swap's stream and its two events.
pub(in crate::families::glm) struct P2Exchange {
    parts: [DevBuf; 2],
    stream: Stream,
    ready: Event,
    done: Event,
    max_rows: usize,
    hidden: usize,
    /// The rows of a begun exchange (`p2_begin` without its `p2_finish`).
    pending: Option<usize>,
    /// Part swaps queued so far (one a MoE layer of a P2 prompt chunk at TP=2; the launch audit).
    swaps: u64,
    /// Lane T: the begun exchange's parts when it went zero-copy over REXL3's RoCE transport (`REXL3_TP_TRANSPORT=roce`,
    /// `REXL3_ROCE_ZERO_COPY` on): the transport site's two halves, `[part_0, part_1]`, the rank's own written by its combine in
    /// place and the peer's by the wire, read in place by the sum.
    zero_copy: Option<crate::families::nemotron_h_omni::tp_roce::ZeroCopySwap>,
    /// Of `swaps`, the zero-copy ones (the launch audit).
    zero_copy_swaps: u64,
    /// Under the sequence-parallel residual: the hidden rows' swap begun on the exchange stream ([`GlmTp::sp_gather_begin`]), its two
    /// events, and whether it is in flight (not yet joined by the main stream).
    gather_ready: Event,
    gather_done: Event,
    gather_pending: bool,
}

impl P2Exchange {
    pub(in crate::families::glm) fn new(max_rows: usize, hidden: usize) -> Result<Self, ModelError> {
        let b = || DevBuf::zeroed(max_rows * hidden * 4);
        Ok(Self {
            parts: [b()?, b()?],
            // at the greatest priority: a swap begun beside a compute grid (the sequence-parallel gathers before the router, a KDA
            // layer's qkv or an MLA layer's query chain; the part swap beside the shared expert) gets its blocks as soon as SMs free up
            // instead of after the grid's last wave (MEASURED, pair A nsys, 4,096-row chunk: the hidden gather before a MoE layer's
            // router started when the router ended, 1.4 ms of the main stream's wait a layer; at this priority it starts with the router,
            // 0.85 ms). Only what overlaps in time changes.
            stream: Stream::new_nonblocking_high_priority()?,
            ready: Event::new()?,
            done: Event::new()?,
            max_rows,
            hidden,
            pending: None,
            swaps: 0,
            zero_copy: None,
            zero_copy_swaps: 0,
            gather_ready: Event::new()?,
            gather_done: Event::new()?,
            gather_pending: false,
        })
    }

    /// Bytes of device memory the exchange holds.
    pub fn bytes(&self) -> usize {
        self.parts.iter().map(|p| p.bytes()).sum()
    }
}

impl GlmTp {
    /// Bytes of device memory the P2 exchange holds (0 for a P1 rank).
    pub fn p2_exchange_bytes(&self) -> usize {
        self.p2.as_ref().map_or(0, |p| p.bytes())
    }

    /// The P2 part swaps this rank has queued (0 for a P1 rank): the launch audit counts one a MoE layer of every prompt chunk, none in a
    /// window.
    pub fn p2_swaps(&self) -> u64 {
        self.p2.as_ref().map_or(0, |p| p.swaps)
    }

    /// Of [`GlmTp::p2_swaps`], those that went zero-copy over the RoCE transport (lane T).
    pub fn p2_zero_copy_swaps(&self) -> u64 {
        self.p2.as_ref().map_or(0, |p| p.zero_copy_swaps)
    }

    /// The row split of a P2 prompt chunk of `rows` rows on this rank (`None`: not a P2 rank of two, or under two 128-row slabs).
    pub(in crate::families::glm::model) fn p2_row_split(&self, rows: usize) -> Option<RowSplit> {
        let t = self.comm.topology();
        if t.ranks != 2 || self.p2.is_none() {
            return None;
        }
        RowSplit::of(rows, crate::families::glm::alloc::MAX_ROWS, t.rank)
    }

    /// The rows of a row-split buffer `buf` (`row_bytes` a row): this rank's rows `[lo, hi)` to the peer, the peer's `[plo, phi)` into
    /// `buf` (byte copies on `st`, the prefill gather's sites).
    pub(in crate::families::glm::model) fn swap_rows(
        &mut self,
        buf: &DevBuf,
        row_bytes: usize,
        sp: RowSplit,
        st: &Stream,
    ) -> Result<(), ModelError> {
        let send = buf.subview(sp.lo * row_bytes, sp.rows() * row_bytes);
        let mut recv = buf.subview(sp.plo * row_bytes, (sp.phi - sp.plo) * row_bytes);
        let (sb, rb) = (send.bytes(), recv.bytes());
        self.comm.swap_bytes_for(
            crate::families::nemotron_h_omni::tp_roce::SiteUser::PrefillGather,
            &send,
            0,
            sb,
            &mut recv,
            0,
            rb,
            st,
        )?;
        // the larger direction (the ranks of an uneven cut count alike)
        self.exchanged_bytes += sb.max(rb) as u64;
        self.exchanges += 1;
        Ok(())
    }

    /// Under the sequence-parallel residual: [`GlmTp::swap_rows`] of `buf` begun on the exchange stream after everything queued on `st`,
    /// so that `st` computes what needs only this rank's rows meanwhile; [`GlmTp::sp_gather_join`] makes `st` wait for it. At most one
    /// is in flight, and nothing else crosses the link until it is joined (the caller joins before its next exchange): every exchange
    /// keeps one order on both ranks.
    pub(in crate::families::glm::model) fn sp_gather_begin(
        &mut self,
        buf: &DevBuf,
        row_bytes: usize,
        sp: RowSplit,
        st: &Stream,
    ) -> Result<(), ModelError> {
        let p2 = self.p2.as_mut().ok_or_else(|| ModelError::Config("a sequence-parallel gather off a P2 rank".into()))?;
        if p2.gather_pending {
            return Err(ModelError::Config("a sequence-parallel gather begun while another is in flight".into()));
        }
        gb_cuda::graph::fork(st, &p2.stream, &p2.gather_ready)?;
        let send = buf.subview(sp.lo * row_bytes, sp.rows() * row_bytes);
        let mut recv = buf.subview(sp.plo * row_bytes, (sp.phi - sp.plo) * row_bytes);
        let (sb, rb) = (send.bytes(), recv.bytes());
        self.comm.swap_bytes_for(
            crate::families::nemotron_h_omni::tp_roce::SiteUser::PrefillGather,
            &send,
            0,
            sb,
            &mut recv,
            0,
            rb,
            &p2.stream,
        )?;
        p2.gather_done.record(&p2.stream)?;
        p2.gather_pending = true;
        self.exchanged_bytes += sb.max(rb) as u64;
        self.exchanges += 1;
        Ok(())
    }

    /// The main stream `st` waits for the gather [`GlmTp::sp_gather_begin`] began (nothing when none is in flight).
    pub(in crate::families::glm::model) fn sp_gather_join(&mut self, st: &Stream) -> Result<(), ModelError> {
        if let Some(p2) = self.p2.as_mut().filter(|p| p.gather_pending) {
            gb_cuda::graph::wait(st, &p2.gather_done)?;
            p2.gather_pending = false;
        }
        Ok(())
    }

    /// The first half of a P2 MoE layer on this rank: the rank's pairs (`glm_p2_moe::pairs_rank` on the global picks in `s`, or
    /// `pairs_subbanks_rank` for a share of several subbanks, whose split gate/up subbank runs through the decode scratch `sd`), its part
    /// (`Own(rank)`), and the swap of the parts queued on the exchange stream (with `split`: only the peer's rows of this rank's part go,
    /// only this rank's rows of the peer's come). The caller may queue replicated work on `st` (the shared expert; with `split` on this
    /// rank's rows only) before [`GlmTp::p2_finish`].
    #[allow(clippy::too_many_arguments)]
    pub(in crate::families::glm::model) fn p2_begin(
        &mut self,
        ex: P2Routed<'_>,
        x: &DevBuf,
        rows: usize,
        s: &mut PrefillScratch,
        sd: &mut MoeScratch,
        split: Option<RowSplit>,
        st: &Stream,
        mark: &mut dyn FnMut(&'static str) -> Result<(), CudaError>,
    ) -> Result<(), ModelError> {
        let t = self.comm.topology();
        let p2 = self
            .p2
            .as_mut()
            .ok_or_else(|| ModelError::Config("a P2 MoE layer on a rank built without the P2 exchange".into()))?;
        let d = ex.dims();
        if p2.pending.is_some() || rows == 0 || rows > p2.max_rows || d.hidden != p2.hidden {
            return Err(ModelError::Config(format!(
                "a P2 exchange of {rows} rows (pending {:?}, at most {})",
                p2.pending, p2.max_rows
            )));
        }
        let order = glm_p2_moe::DEFAULT_ORDER;
        match ex {
            P2Routed::Bank(b) => glm_p2_moe::pairs_rank(b, x, Dtype::F16, rows, s, &mut self.os, order, st, mark)?,
            P2Routed::Subbanks(sb) => {
                glm_p2_moe::pairs_subbanks_rank(sb, x, Dtype::F16, rows, s, &mut self.os, sd, order, st, mark)?
            }
        }
        // lane T: over REXL3's RoCE transport the part goes straight into the transport site's own half (the same combine, another
        // output buffer) and the swap only publishes it: no staging copy, no copy-out (the same bytes on the wire, so the same bits)
        let zc = if t.ranks == 2 && split.is_none() {
            self.comm.roce_tp().and_then(|rt| rt.zero_copy_swap(rows * p2.hidden * 4))
        } else {
            None
        };
        // the rank's part: its own expert group's slots (whole experts), or every slot of its half of the width (the width split)
        let own_part = if self.width.is_some() { CombineMode::All } else { CombineMode::Own(t.rank) };
        if let Some(mut z) = zc {
            let r = t.rank;
            glm_p2_moe::combine_dims(own_part, &s.y_pairs, &s.pick, &s.wts, None, None, &mut z.halves[r], rows, &d, st)?;
            mark("combine")?;
            gb_cuda::graph::fork(st, &p2.stream, &p2.ready)?;
            self.comm.roce_tp().expect("checked").publish_swap(&z, &p2.stream)?;
            p2.done.record(&p2.stream)?;
            self.exchanged_bytes += z.bytes as u64;
            self.exchanges += 1;
            p2.swaps += 1;
            p2.zero_copy_swaps += 1;
            p2.zero_copy = Some(z);
        } else if t.ranks == 2 {
            let r = t.rank;
            glm_p2_moe::combine_dims(own_part, &s.y_pairs, &s.pick, &s.wts, None, None, &mut p2.parts[r], rows, &d, st)?;
            mark("combine")?;
            // the swap on its own stream, after the part
            gb_cuda::graph::fork(st, &p2.stream, &p2.ready)?;
            let rb = p2.hidden * 4;
            let (soff, sbytes, roff, rbytes) = match split {
                // the peer's rows of this part to the peer, this rank's rows of the peer's part from it
                Some(sp) => (sp.plo * rb, (sp.phi - sp.plo) * rb, sp.lo * rb, sp.rows() * rb),
                None => (0, rows * rb, 0, rows * rb),
            };
            let [p0, p1] = &mut p2.parts;
            let (send, recv) = if r == 0 { (&*p0, p1) } else { (&*p1, p0) };
            // the part swap's own sites under REXL3's RoCE transport (lane T: never shared with another user of its size class)
            self.comm.swap_bytes_for(
                crate::families::nemotron_h_omni::tp_roce::SiteUser::PartSwap,
                send,
                soff,
                sbytes,
                recv,
                roff,
                rbytes,
                &p2.stream,
            )?;
            let bytes = sbytes.max(rbytes);
            p2.done.record(&p2.stream)?;
            self.exchanged_bytes += bytes as u64;
            self.exchanges += 1;
            p2.swaps += 1;
        }
        p2.pending = Some(rows);
        Ok(())
    }

    /// The second half: wait for the swap, then `out = (part_0 + part_1) + shared` (a group of one rank: the one-node combine of its pairs).
    /// With `split` (the one `p2_begin` took): the sum on this rank's rows only (`shared` holds those rows), then the output rows swapped
    /// with the peer's on the exchange stream, which `st` waits for.
    pub(in crate::families::glm::model) fn p2_finish(
        &mut self,
        d: &MoeDims,
        s: &PrefillScratch,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        split: Option<RowSplit>,
        st: &Stream,
    ) -> Result<(), ModelError> {
        let t = self.comm.topology();
        let p2 = self
            .p2
            .as_mut()
            .ok_or_else(|| ModelError::Config("a P2 MoE layer on a rank built without the P2 exchange".into()))?;
        let rows = p2.pending.take().ok_or_else(|| ModelError::Config("a P2 exchange finished before it began".into()))?;
        match split {
            Some(sp) if t.ranks == 2 => {
                gb_cuda::graph::wait(st, &p2.done)?;
                let parts = (&p2.parts[0], &p2.parts[1]);
                glm_p2_moe::combine_parts_rows(shared, parts, out, sp.lo, sp.rows(), d, st)?;
                // the output rows on the exchange stream (its sites are the part swap's: one user, in order), then the main stream waits
                gb_cuda::graph::fork(st, &p2.stream, &p2.ready)?;
                let rb = d.hidden * 4;
                let send = out.subview(sp.lo * rb, sp.rows() * rb);
                let mut recv = out.subview(sp.plo * rb, (sp.phi - sp.plo) * rb);
                let (sb, rcb) = (send.bytes(), recv.bytes());
                self.comm.swap_bytes_for(
                    crate::families::nemotron_h_omni::tp_roce::SiteUser::PartSwap,
                    &send,
                    0,
                    sb,
                    &mut recv,
                    0,
                    rcb,
                    &p2.stream,
                )?;
                p2.done.record(&p2.stream)?;
                gb_cuda::graph::wait(st, &p2.done)?;
                self.exchanged_bytes += sb.max(rcb) as u64;
                self.exchanges += 1;
            }
            _ if t.ranks == 2 => {
                gb_cuda::graph::wait(st, &p2.done)?;
                let z = p2.zero_copy.take();
                let parts = match &z {
                    Some(z) => (&z.halves[0], &z.halves[1]),
                    None => (&p2.parts[0], &p2.parts[1]),
                };
                glm_p2_moe::combine_dims(CombineMode::Parts, &s.y_pairs, &s.pick, &s.wts, shared, Some(parts), out, rows, d, st)?;
            }
            _ => {
                glm_p2_moe::combine_dims(CombineMode::Both, &s.y_pairs, &s.pick, &s.wts, shared, None, out, rows, d, st)?;
            }
        }
        Ok(())
    }

    /// [`GlmTp::p2_finish`] of a row-split chunk under the sequence-parallel residual (`GlmModel::run`'s `sp`): the parts' sum on this
    /// rank's rows `[sp.lo, sp.hi)` of `out` alone, and no output swap (the rank's residual holds only its own rows). The rows it writes
    /// are `p2_finish`'s with `Some(sp)`: the same sum of the same values.
    pub(in crate::families::glm::model) fn p2_finish_own_rows(
        &mut self,
        d: &MoeDims,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        sp: RowSplit,
        st: &Stream,
    ) -> Result<(), ModelError> {
        let p2 = self
            .p2
            .as_mut()
            .ok_or_else(|| ModelError::Config("a P2 MoE layer on a rank built without the P2 exchange".into()))?;
        p2.pending.take().ok_or_else(|| ModelError::Config("a P2 exchange finished before it began".into()))?;
        gb_cuda::graph::wait(st, &p2.done)?;
        glm_p2_moe::combine_parts_rows(shared, (&p2.parts[0], &p2.parts[1]), out, sp.lo, sp.rows(), d, st)?;
        Ok(())
    }
}

impl super::GlmModel {
    /// What a P2 rank needs before it joins (refused by name otherwise): at most two ranks; at two, the range ownership (rank r owns
    /// combine group r: `ExpertOwnership::range(E, r, 2)`), or under the expert width split every expert at the rank's half of the width
    /// (its part is every slot's), and a communicator with the byte swap; at one, every expert; every MoE layer's routed experts (and
    /// the NextN layer's) one bank the P2 expert GEMMs take, or subbanks a P2 prefill takes (`glm_p2_moe::supported_subbanks`: a
    /// mixed-width share, its split gate/up subbank through the decode pairs stage).
    pub(super) fn check_p2_tp(
        &self,
        comm: &dyn crate::families::nemotron_h_omni::tp::TpComm,
        own: &gb_cuda::moe::ExpertOwnership,
    ) -> Result<(), ModelError> {
        use crate::families::glm::device::{FfnDevice, MtpMoe};
        let var = crate::families::glm::contract::CONTRACT_VAR;
        let t = comm.topology();
        let bad = |what: String| ModelError::Config(format!("{var}=p2 on a tensor-parallel rank: {what}"));
        match t.ranks {
            1 if own.is_all() => {}
            1 => return Err(bad("a group of one rank must own every expert".into())),
            2 => {
                if let Some(w) = self.cfg.ffn.width {
                    if !own.is_all() || (w.rank, w.ranks) != (t.rank, 2) {
                        return Err(bad(format!(
                            "the width share of rank {} of {} on rank {} of 2 owning {} experts (the width split holds every expert)",
                            w.rank,
                            w.ranks,
                            t.rank,
                            own.local()
                        )));
                    }
                } else {
                    let want = gb_cuda::moe::ExpertOwnership::range(own.experts, t.rank, 2)
                        .map_err(|e| bad(format!("no range ownership for rank {}: {e:?}", t.rank)))?;
                    if *own != want {
                        return Err(bad(format!(
                            "rank {} must own the experts of combine group {} (ExpertOwnership::range: {} experts), not this ownership of {}",
                            t.rank,
                            t.rank,
                            want.local(),
                            own.local()
                        )));
                    }
                }
                if !comm.supports_swap() {
                    return Err(bad("the communicator has no byte swap (TpComm::swap_bytes) for the parts' exchange".into()));
                }
            }
            r => return Err(bad(format!("{r} ranks (the P2 combine has two groups: one or two ranks)"))),
        }
        let one = |name: String, b: &MoeExperts| -> Result<(), ModelError> {
            if glm_p2_moe::supported_bank(b) {
                Ok(())
            } else {
                Err(bad(format!(
                    "{name} ({} / {} bits, {:?}) are not a format the P2 expert GEMMs take",
                    b.bits_gu, b.bits_d, b.codebook
                )))
            }
        };
        let sub = |name: String, sb: &MoeSubbanks| -> Result<(), ModelError> {
            glm_p2_moe::supported_subbanks(sb).map_err(|why| bad(format!("{name} (subbanks): {why}")))
        };
        for (l, d) in self.dev.layers.iter().enumerate() {
            match &d.ffn {
                FfnDevice::Moe { layer: m, .. } => one(format!("layer {l}'s routed experts"), &m.experts)?,
                FfnDevice::MoeSubbanks { layer: m, .. } => sub(format!("layer {l}'s routed experts"), &m.experts)?,
                FfnDevice::Dense(_) => {}
            }
        }
        if let Some(w) = self.dev.mtp.as_ref() {
            match &w.moe {
                MtpMoe::Single(m) => one("the NextN layer's routed experts".into(), &m.experts)?,
                MtpMoe::Subbanks(m) => sub("the NextN layer's routed experts".into(), &m.experts)?,
            }
        }
        Ok(())
    }
}
