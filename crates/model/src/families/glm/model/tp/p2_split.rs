//! P2 on lane E's KDA head split (Lane P step P5b; `docs/GLM_PREFILL_P2.md` section 10): a P2 prompt chunk's KDA layer on a rank that
//! runs half the heads.
//!
//! * **qkv**: the rank's stored strip cut (`PlannedExl3::w`, its 32 heads' q, k and v strips laid out as a layer of the kept columns)
//!   through the P2 GEMM (`gb_cuda::glm_p2::linear`). The P2 K order is a function of K alone and its N tiles sit on strip boundaries, so
//!   every column is the P2 single node's bit for bit (`crates/cuda/tests/glm_p2_dense.rs`, `p2_on_a_strip_cut_...`).
//! * **the recurrence**: the chunked KDA (`kda::prefill_chunked`) on the rank's 32 heads and its 32-head pool. Heads are independent in
//!   every KDA kernel (lane E's `kda_split.rs`), the chunk kernel's arithmetic per head is the single node's (`kda=glm-kda-chunk16-v1`).
//! * **o_proj**: option (i) of MAIN's P5b brief: the fp16 core outputs gathered (this rank's 4,096 columns, the peer's swapped in through
//!   `TpComm::swap_bytes_for(SiteUser::PrefillGather, ..)`: byte copies, no arithmetic), then the full P2 o_proj, replicated. So the
//!   layer's output is the P2 single node's by construction (no new P2 fingerprint id). Under the sequence-parallel residual
//!   ([`super::RowSplit`], `GlmModel::run`'s `sp`) only this rank's rows are gathered (the peer's rows of this rank's columns go to the
//!   peer) and the o_proj runs on them alone: every P2 GEMM row is its own row's arithmetic, so those rows are the same bits.

use super::GlmTp;
use super::KdaIo;
use crate::error::ModelError;
use crate::families::nemotron_h_omni::tp_roce::SiteUser;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::kda;
use gb_cuda::{DevBuf, Stream};

impl GlmTp {
    /// `full` `[rows, 2 w]` fp16 from this rank's half `own` `[rows, w]` (its columns `[r w, (r + 1) w)`) and the peer's, swapped in through
    /// the prefill gather's own RoCE sites (`SiteUser::PrefillGather`; NCCL on every other communicator): byte copies.
    fn gather_cols_f16_prefill(
        &mut self,
        own: &DevBuf,
        full: &mut DevBuf,
        rows: usize,
        w: usize,
        st: &Stream,
    ) -> Result<(), ModelError> {
        let r = self.split.as_ref().ok_or_else(|| ModelError::Config("no head split".into()))?.rank;
        let bytes = rows * w * 2;
        let mut peer = self.split.as_ref().unwrap().peer.subview(0, bytes);
        self.comm.swap_bytes_for(SiteUser::PrefillGather, own, 0, bytes, &mut peer, 0, bytes, st)?;
        self.exchanged_bytes += bytes as u64;
        self.exchanges += 1;
        full.copy_2d_from(r * w * 2, 2 * w * 2, own, 0, w * 2, w * 2, rows, st)?;
        full.copy_2d_from((1 - r) * w * 2, 2 * w * 2, &peer, 0, w * 2, w * 2, rows, st)?;
        Ok(())
    }

    /// One KDA layer of a P2 prompt chunk on this rank's heads (see the module): the qkv strips through the P2 GEMM, the chunked
    /// recurrence on the rank's pool (pieces of `piece` rows, `desc` from `kda::prefill_descriptor_pieces`), the gathered core output
    /// (`io.out` `[rows, 64 * 128]`) and the full P2 o_proj into `io.mixer_out`; with `sp`, the gather and the o_proj of this rank's rows
    /// `[sp.lo, sp.hi)` alone (the other rows of `io.mixer_out` are left as they were). With `join` (and `sp`) the peer's hidden rows are
    /// still crossing (`GlmTp::sp_gather_begin`): the qkv strips of this rank's rows run first, then the join, then the peer's rows (every
    /// P2 GEMM row is its own row's arithmetic, so the two launches leave the one launch's bits).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::families::glm::model) fn kda_split_layer_p2(
        &mut self,
        ord: usize,
        o: &Exl3Weights,
        dims: kda::Dims,
        params: &kda::Params,
        pool: &mut kda::Pool,
        io: KdaIo<'_>,
        slot: usize,
        rows: usize,
        rec: &mut kda::Record,
        desc: &DevBuf,
        piece: usize,
        st: &Stream,
        sp: Option<super::RowSplit>,
        join: bool,
    ) -> Result<(), ModelError> {
        let KdaIo { hidden, qkv, out, mixer_out, ws, kws } = io;
        let c = dims.v_dim();
        let mut own = self.split.as_ref().ok_or_else(|| ModelError::Config("no head split".into()))?.own.subview(0, rows * c * 2);
        let no_rank = || ModelError::Config(format!("no KDA head split for KDA layer {ord}"));
        match sp.filter(|_| join) {
            Some(sp) => {
                let (k, n) = {
                    let w = &self.kda_rank(ord).ok_or_else(no_rank)?.qkv.w;
                    (w.k, w.n)
                };
                let mut strips = |t: &Self, lo: usize, m: usize| -> Result<(), ModelError> {
                    let w = &t.kda_rank(ord).ok_or_else(no_rank)?.qkv.w;
                    let x = hidden.subview(lo * k * 2, m * k * 2);
                    let mut y = qkv.subview(lo * n * 4, m * n * 4);
                    gb_cuda::glm_p2::linear(w, &x, &mut y, m, Dtype::F16, Dtype::F32, ws, st)?;
                    Ok(())
                };
                strips(self, sp.lo, sp.rows())?;
                self.sp_gather_join(st)?;
                strips(self, sp.plo, sp.phi - sp.plo)?;
            }
            None => {
                self.sp_gather_join(st)?;
                let kr = self.kda_rank(ord).ok_or_else(no_rank)?;
                gb_cuda::glm_p2::linear(&kr.qkv.w, hidden, qkv, rows, Dtype::F16, Dtype::F32, ws, st)?;
            }
        }
        {
            let kr = self.kda_rank(ord).ok_or_else(no_rank)?;
            kda::prefill_chunked(
                st,
                dims,
                &kr.w.view(),
                params,
                hidden,
                Dtype::F16,
                qkv,
                Dtype::F32,
                pool,
                slot,
                rows,
                rec,
                kws,
                &mut own,
                desc,
                piece,
            )?;
        }
        match sp {
            None => {
                self.gather_cols_f16_prefill(&own, out, rows, c, st)?;
                gb_cuda::glm_p2::linear(o, out, mixer_out, rows, Dtype::F16, Dtype::F32, ws, st)?;
            }
            Some(sp) => {
                let r = self.split.as_ref().unwrap().rank;
                let (n, pn) = (sp.rows(), sp.phi - sp.plo);
                // the peer's rows of this rank's columns to the peer, this rank's rows of the peer's columns from it
                let send = own.subview(sp.plo * c * 2, pn * c * 2);
                let mut peer = self.split.as_ref().unwrap().peer.subview(0, n * c * 2);
                self.comm.swap_bytes_for(SiteUser::PrefillGather, &send, 0, pn * c * 2, &mut peer, 0, n * c * 2, st)?;
                self.exchanged_bytes += (n.max(pn) * c * 2) as u64;
                self.exchanges += 1;
                let w2 = 2 * c * 2;
                out.copy_2d_from(sp.lo * w2 + r * c * 2, w2, &own, sp.lo * c * 2, c * 2, c * 2, n, st)?;
                out.copy_2d_from(sp.lo * w2 + (1 - r) * c * 2, w2, &peer, 0, c * 2, c * 2, n, st)?;
                let x = out.subview(sp.lo * w2, n * w2);
                let mut y = mixer_out.subview(sp.lo * o.n * 4, n * o.n * 4);
                gb_cuda::glm_p2::linear(o, &x, &mut y, n, Dtype::F16, Dtype::F32, ws, st)?;
            }
        }
        Ok(())
    }
}
