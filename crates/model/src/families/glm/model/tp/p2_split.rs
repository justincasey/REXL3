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

    /// One KDA layer of a P2 prompt forward on this rank's heads (see the module): the qkv strips through the P2 GEMM over every row, the
    /// chunked recurrence of each sequence of `segs` on its own slot of the rank's pool (pieces of `piece` rows, each segment's `desc` from
    /// `kda::prefill_descriptor_pieces`), the gathered core output (`io.out` `[rows, 64 * 128]`) and the full P2 o_proj into
    /// `io.mixer_out`; with `sp`, the gather and the o_proj of this rank's rows `[sp.lo, sp.hi)` alone (the other rows of `io.mixer_out` are
    /// left as they were). With `join` (and `sp`) the peer's hidden rows are still crossing (`GlmTp::sp_gather_begin`): the qkv strips of
    /// this rank's rows run first, then the join, then the peer's rows (every P2 GEMM row is its own row's arithmetic, so the two launches
    /// leave the one launch's bits; so does a forward of several prompts, whose recurrences never read each other's rows). Without `o`
    /// (the last layer of a forward whose tail runs on the head rows, `ops::plan_forward_head_rows`) it stops at the core output, this
    /// rank's columns of every row ([`GlmTp::kda_head_rows_p2`] gathers the head rows').
    #[allow(clippy::too_many_arguments)]
    pub(in crate::families::glm::model) fn kda_split_layer_p2(
        &mut self,
        ord: usize,
        o: Option<&Exl3Weights>,
        dims: kda::Dims,
        params: &kda::Params,
        pool: &mut kda::Pool,
        io: KdaIo<'_>,
        segs: &[super::KdaSeg<'_>],
        rows: usize,
        rec: &mut kda::Record,
        piece: usize,
        st: &Stream,
        sp: Option<super::RowSplit>,
        join: bool,
    ) -> Result<(), ModelError> {
        let KdaIo { hidden, qkv, out, mixer_out, ws, kws } = io;
        let c = dims.v_dim();
        let own = self.split.as_ref().ok_or_else(|| ModelError::Config("no head split".into()))?.own.subview(0, rows * c * 2);
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
            let (h, cd) = (dims.hidden, dims.conv_dim());
            for g in segs {
                let x = hidden.subview(g.row0 * h * 2, g.rows * h * 2);
                let q = qkv.subview(g.row0 * cd * 4, g.rows * cd * 4);
                let mut y = own.subview(g.row0 * c * 2, g.rows * c * 2);
                kda::prefill_chunked(
                    st,
                    dims,
                    &kr.w.view(),
                    params,
                    &x,
                    Dtype::F16,
                    &q,
                    Dtype::F32,
                    pool,
                    g.slot,
                    g.rows,
                    rec,
                    kws,
                    &mut y,
                    g.desc,
                    piece,
                )?;
            }
        }
        let Some(o) = o else { return Ok(()) };
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

    /// [`Op::HeadRows`](crate::families::glm::ops::Op::HeadRows) of a P2 forward on this rank's KDA heads, after the last layer's core
    /// ([`GlmTp::kda_split_layer_p2`] without `o`): each `(i, last)` of `moves` (sequence order, `last >= i`) moves row `last` of this rank's
    /// core columns to row `i`, then rows `[0, moves.len())` are gathered with the peer's columns into `out` `[rows, 2 c]` (`c`: this rank's
    /// columns), the full row the P2 o_proj reads. Byte copies.
    pub(in crate::families::glm::model) fn kda_head_rows_p2(
        &mut self,
        out: &mut DevBuf,
        moves: &[(usize, usize)],
        c: usize,
        st: &Stream,
    ) -> Result<(), ModelError> {
        let sp = self.split.as_mut().ok_or_else(|| ModelError::Config("no head split".into()))?;
        for &(i, last) in moves.iter().filter(|(i, last)| i != last) {
            sp.own.copy_within(i * c * 2, last * c * 2, c * 2, st)?;
        }
        let n = moves.len();
        let own = sp.own.subview(0, n * c * 2);
        self.gather_cols_f16_prefill(&own, out, n, c, st)
    }

    /// Under the sequence-parallel residual, the head rows of a residual buffer `buf` (`row_bytes` a row) on both ranks: each rank has
    /// moved the head rows it owns (`mine[i]`) to row `i`; the rows `[0, mine.len())` are swapped through the head split's gather buffer
    /// (in pieces it holds) and each rank keeps the peer's copy of the rows it does not own. Both ranks make the same exchanges; byte
    /// copies.
    pub(in crate::families::glm::model) fn sp_head_rows(
        &mut self,
        buf: &mut DevBuf,
        row_bytes: usize,
        mine: &[bool],
        st: &Stream,
    ) -> Result<(), ModelError> {
        let stage_bytes = self.split.as_ref().ok_or_else(|| ModelError::Config("no head split".into()))?.peer.bytes();
        let per = stage_bytes / row_bytes;
        if per == 0 {
            return Err(ModelError::Config(format!("the head split's gather buffer ({stage_bytes} B) holds no {row_bytes}-byte row")));
        }
        let n = mine.len();
        let mut a = 0;
        while a < n {
            let k = (n - a).min(per);
            let bytes = k * row_bytes;
            let mut stage = self.split.as_ref().unwrap().peer.subview(0, bytes);
            // (the residual buffers are not views of a block: the rows go by offset)
            self.comm.swap_bytes_for(SiteUser::PrefillGather, buf, a * row_bytes, bytes, &mut stage, 0, bytes, st)?;
            self.exchanged_bytes += bytes as u64;
            self.exchanges += 1;
            for i in (a..a + k).filter(|&i| !mine[i]) {
                buf.copy_range_from(i * row_bytes, &stage, (i - a) * row_bytes, row_bytes, st)?;
            }
            a += k;
        }
        Ok(())
    }
}
