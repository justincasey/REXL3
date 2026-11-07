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

use super::GlmTp;
use crate::error::ModelError;
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::glm_p2_moe::{self, CombineMode};
use gb_cuda::moe::{MoeDims, MoeExperts, MoeScratch};
use gb_cuda::moe_prefill::PrefillScratch;
use gb_cuda::moe_subbanks::MoeSubbanks;
use gb_cuda::rt::Event;
use gb_cuda::{CudaError, DevBuf, Stream};

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
}

impl P2Exchange {
    pub(in crate::families::glm) fn new(max_rows: usize, hidden: usize) -> Result<Self, ModelError> {
        let b = || DevBuf::zeroed(max_rows * hidden * 4);
        Ok(Self {
            parts: [b()?, b()?],
            stream: Stream::new_nonblocking()?,
            ready: Event::new()?,
            done: Event::new()?,
            max_rows,
            hidden,
            pending: None,
            swaps: 0,
            zero_copy: None,
            zero_copy_swaps: 0,
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

    /// The first half of a P2 MoE layer on this rank: the rank's pairs (`glm_p2_moe::pairs_rank` on the global picks in `s`, or
    /// `pairs_subbanks_rank` for a share of several subbanks, whose split gate/up subbank runs through the decode scratch `sd`), its part
    /// (`Own(rank)`), and the swap of the parts queued on the exchange stream. The caller may queue replicated work on `st` (the shared
    /// expert) before [`GlmTp::p2_finish`].
    #[allow(clippy::too_many_arguments)]
    pub(in crate::families::glm::model) fn p2_begin(
        &mut self,
        ex: P2Routed<'_>,
        x: &DevBuf,
        rows: usize,
        s: &mut PrefillScratch,
        sd: &mut MoeScratch,
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
        let zc = if t.ranks == 2 { self.comm.roce_tp().and_then(|rt| rt.zero_copy_swap(rows * p2.hidden * 4)) } else { None };
        if let Some(mut z) = zc {
            let r = t.rank;
            glm_p2_moe::combine_dims(CombineMode::Own(r), &s.y_pairs, &s.pick, &s.wts, None, None, &mut z.halves[r], rows, &d, st)?;
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
            glm_p2_moe::combine_dims(CombineMode::Own(r), &s.y_pairs, &s.pick, &s.wts, None, None, &mut p2.parts[r], rows, &d, st)?;
            mark("combine")?;
            // the swap on its own stream, after the part
            gb_cuda::graph::fork(st, &p2.stream, &p2.ready)?;
            let bytes = rows * p2.hidden * 4;
            let [p0, p1] = &mut p2.parts;
            let (send, recv) = if r == 0 { (&*p0, p1) } else { (&*p1, p0) };
            // the part swap's own sites under REXL3's RoCE transport (lane T: never shared with another user of its size class)
            self.comm.swap_bytes_for(
                crate::families::nemotron_h_omni::tp_roce::SiteUser::PartSwap,
                send,
                0,
                bytes,
                recv,
                0,
                bytes,
                &p2.stream,
            )?;
            p2.done.record(&p2.stream)?;
            self.exchanged_bytes += bytes as u64;
            self.exchanges += 1;
            p2.swaps += 1;
        }
        p2.pending = Some(rows);
        Ok(())
    }

    /// The second half: wait for the swap, then `out = (part_0 + part_1) + shared` (a group of one rank: the one-node combine of its pairs).
    pub(in crate::families::glm::model) fn p2_finish(
        &mut self,
        d: &MoeDims,
        s: &PrefillScratch,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), ModelError> {
        let t = self.comm.topology();
        let p2 = self
            .p2
            .as_mut()
            .ok_or_else(|| ModelError::Config("a P2 MoE layer on a rank built without the P2 exchange".into()))?;
        let rows = p2.pending.take().ok_or_else(|| ModelError::Config("a P2 exchange finished before it began".into()))?;
        if t.ranks == 2 {
            gb_cuda::graph::wait(st, &p2.done)?;
            let z = p2.zero_copy.take();
            let parts = match &z {
                Some(z) => (&z.halves[0], &z.halves[1]),
                None => (&p2.parts[0], &p2.parts[1]),
            };
            glm_p2_moe::combine_dims(CombineMode::Parts, &s.y_pairs, &s.pick, &s.wts, shared, Some(parts), out, rows, d, st)?;
        } else {
            glm_p2_moe::combine_dims(CombineMode::Both, &s.y_pairs, &s.pick, &s.wts, shared, None, out, rows, d, st)?;
        }
        Ok(())
    }
}

impl super::GlmModel {
    /// What a P2 rank needs before it joins (refused by name otherwise): at most two ranks; at two, the range ownership (rank r owns
    /// combine group r: `ExpertOwnership::range(E, r, 2)`) and a communicator with the byte swap; at one, every expert; every MoE layer's
    /// routed experts (and the NextN layer's) one bank the P2 expert GEMMs take, or subbanks a P2 prefill takes
    /// (`glm_p2_moe::supported_subbanks`: a mixed-width share, its split gate/up subbank through the decode pairs stage).
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
