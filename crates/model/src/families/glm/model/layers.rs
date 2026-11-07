//! Layer bodies that run on calls of any number of rows up to the scratch's, shared by the trunk (`GlmModel::run`) and the NextN block.
//!
//! A call of at most [`MAX_ROWS`] rows (a decode or verify window, a small prompt chunk) runs the decode kernels exactly as before. A larger
//! prefill chunk (`big`) uses the large-row kernels, which give each row the decode kernels' bits (`gb_cuda::exl3_big`, `gb_cuda::moe_prefill`), and
//! cuts the attention into slabs of at most [`MAX_ROWS`] rows in position order (the DSA ring holds 256 raw rows: `RING_LEN` = a slab + a pool).
//!
//! Under the P2 prefill contract (`families/glm/contract.rs`) every dense EXL3 linear of a prompt chunk takes the tensor-core GEMM
//! (`gb_cuda::glm_p2`) whatever the chunk's rows ([`LinPath::P2`]): a chunk's width never chooses the arithmetic of a prompt row.

use super::{Scratch, RMS_EPS};
use crate::error::ModelError;
use crate::families::glm::alloc::MAX_ROWS;
use crate::families::glm::config::GlmConfig;
use crate::families::glm::device::MlaDevice;
use gb_cuda::attention::{BlockTable, RowIndex};
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::mla::{self, LatentCache, SparseSel};
use gb_cuda::{dsa, glm_glue, glm_p2_mla, mla_prefill, moe, moe_prefill, DevBuf, Stream};

/// The kernel a dense EXL3 linear of a forward runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LinPath {
    /// The decode kernel (`Exl3Weights::forward`, 1..=[`MAX_ROWS`] rows): every window, and a P1 prompt chunk of at most [`MAX_ROWS`] rows.
    Decode,
    /// The large-row kernel with the decode kernel's bits (`Exl3Weights::forward_big`): a P1 prompt chunk of more rows (or one that takes
    /// snapshots inside itself).
    Big,
    /// The P2 tensor-core GEMM (`gb_cuda::glm_p2`): every prompt chunk under P2, at every width.
    P2,
}

impl LinPath {
    /// The path of a forward: P2 when the forward is a prefill under the P2 contract (`p2_prefill`), else the large-row kernel when `big`.
    pub(crate) const fn of(p2_prefill: bool, big: bool) -> Self {
        if p2_prefill {
            LinPath::P2
        } else if big {
            LinPath::Big
        } else {
            LinPath::Decode
        }
    }
}

/// `y = layer(x)` for `rows` rows on `path`: the decode kernel up to [`MAX_ROWS`], the large-row kernel (the same bits) above, or under P2 the
/// tensor-core GEMM at any rows.
#[allow(clippy::too_many_arguments)]
pub(super) fn lin(
    w: &Exl3Weights,
    x: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    xd: Dtype,
    yd: Dtype,
    ws: &mut DevBuf,
    st: &Stream,
    path: LinPath,
) -> Result<(), ModelError> {
    match path {
        LinPath::P2 => gb_cuda::glm_p2::linear(w, x, y, rows, xd, yd, ws, st)?,
        LinPath::Big => w.forward_big(x, y, rows, xd, yd, ws, st)?,
        LinPath::Decode => w.forward(x, y, rows, xd, yd, ws, st)?,
    }
    Ok(())
}

/// A gated MLP (the shared expert, a dense MLP: `moe::SharedExpert`) of `rows` rows of `x` (fp16) into `out` (fp32) on `path`: the decode
/// kernels in `small` (1..=[`MAX_ROWS`] rows), the large-row kernels in `big` (any rows, in its pieces), or under P2 the tensor-core GEMM in
/// whichever of the two the caller has (`big` when the chunk has more than [`MAX_ROWS`] rows).
#[allow(clippy::too_many_arguments)]
pub(super) fn gated(
    m: &moe::SharedExpert,
    x: &DevBuf,
    rows: usize,
    out: &mut DevBuf,
    small: Option<&mut moe::SharedScratch>,
    big: Option<&mut moe_prefill::SharedScratchBig>,
    st: &Stream,
    path: LinPath,
) -> Result<(), ModelError> {
    let none = || ModelError::Config("no gated-MLP scratch for this chunk".into());
    match path {
        LinPath::P2 => {
            let bufs = match (big, small) {
                (Some(b), _) if rows > MAX_ROWS => b.p2_bufs(),
                (_, Some(s)) if rows <= s.max_rows => s.p2_bufs(),
                (Some(b), _) => b.p2_bufs(),
                _ => return Err(none()),
            };
            gb_cuda::glm_p2::gated(m, x, Dtype::F16, rows, out, bufs, st)?;
        }
        LinPath::Big => m.forward_big(x, Dtype::F16, rows, out, big.ok_or_else(none)?, st)?,
        LinPath::Decode => m.forward(x, Dtype::F16, rows, out, small.ok_or_else(none)?, st)?,
    }
    Ok(())
}

/// Called after each slab's `dsa::append` of a large chunk with the slab's first row, its rows and the layer's planes (see [`mla_dsa_mixer`]).
pub(super) type AfterAppend<'a> = &'a mut dyn FnMut(usize, usize, &dsa::Planes) -> Result<(), ModelError>;

/// What the MLA mixer needs besides the scratch and the layer's own state.
pub(super) struct MlaCtx<'a> {
    pub st: &'a Stream,
    pub cfg: &'a GlmConfig,
    pub table: &'a BlockTable,
    pub max_context: usize,
    /// What the rows' positions decide for the small path: the indexer's pool bound, whether any row selects (sparse regime), and the attention's
    /// context bound. The trunk takes them from the graph path's `RunShape` when one is bound (they must not depend on the context length there),
    /// from the batch's largest position otherwise; the NextN block and every large chunk from the largest position.
    pub n_pools_max: usize,
    pub any_sparse: bool,
    pub mla_ctx: usize,
    /// The large-row path: the projections over all the rows, the attention in slabs (a chunk of more than `MAX_ROWS` rows, or any chunk that takes
    /// snapshots inside itself).
    pub big: bool,
    /// The kernel of the layer's dense EXL3 linears (q_a, q_b, kv_a, o): [`LinPath::of`] the forward's contract and `big`. The indexer's
    /// linears are not on it: they stay P1 under every contract.
    pub lin: LinPath,
}

/// Rows `a .. a + n` of a row index.
fn slab_index(ri: &RowIndex, a: usize, n: usize) -> RowIndex {
    RowIndex { slot: ri.slot.subview(a * 4, n * 4), pos: ri.pos.subview(a * 4, n * 4), rows: n }
}

/// One MLA layer with the DSA indexer on `rows` rows of `s.hidden` (the final-normed input) at consecutive positions ending at `max_pos`:
/// writes the sublayer output to `s.mixer_out` (fp32). `ri`: the rows' (slot, position) index, `None` for `s.rowidx`. `mark(name)` is called after
/// each step (profiling). Above [`MAX_ROWS`] rows (`big`) the projections run over all the rows, the attention in slabs. `after_append`, for a large
/// chunk only: called after each slab's `dsa::append` with the slab's first row, its rows and the layer's planes (the prefix cache copies the ring tail
/// of a snapshot taken inside the chunk there: the ring holds the last 256 raw rows, a slab is at most 128).
///
/// `own` (a large P2 chunk on a row-split rank of two, `tp::RowSplit`: slab-aligned rows `[lo, hi)`): the latent, the indexer's keys and every
/// slab's `dsa::append` run over all the rows (the layer's state), the per-row rest (q_a, q_b, the indexer's query and weights, the selection,
/// the attention and o_proj) over this rank's rows only; `s.mixer_out` then holds this rank's rows, which the caller swaps with the peer's. Every
/// op is row-independent, so each row's bits are the full call's.
#[allow(clippy::too_many_arguments)]
pub(super) fn mla_dsa_mixer(
    c: &MlaCtx<'_>,
    w: &MlaDevice,
    latent: &mut LatentCache,
    planes: &mut dsa::Planes,
    s: &mut Scratch,
    ri: Option<&RowIndex>,
    rows: usize,
    max_pos: usize,
    own: Option<(usize, usize)>,
    mut after_append: Option<AfterAppend<'_>>,
    mark: &mut dyn FnMut(&'static str) -> Result<(), ModelError>,
) -> Result<(), ModelError> {
    let MlaCtx { st, cfg, table, max_context, n_pools_max, any_sparse, mla_ctx, big, lin: path } = *c;
    let ri: &RowIndex = match ri {
        Some(r) => r,
        None => &s.rowidx,
    };
    let ix = &w.indexer;
    let h = cfg.hidden;
    let (lo, hi) = own.unwrap_or((0, rows));
    assert!(
        own.is_none() || (big && lo < hi && hi <= rows && lo % MAX_ROWS == 0 && (hi == rows || hi % MAX_ROWS == 0)),
        "a row split of an MLA layer is slab-aligned rows of a large chunk"
    );
    let n_own = hi - lo;
    // rows [lo, hi) of a scratch buffer of `width` elements of `es` bytes a row
    let part = |b: &DevBuf, width: usize, es: usize| b.subview(lo * width * es, n_own * width * es);
    let (qa_w, qd) = (cfg.mla.q_lora_rank, cfg.mla.heads * cfg.mla.qk_head_dim());
    if own.is_some() {
        let x = part(&s.hidden, h, 2);
        let mut qa = part(&s.q_a, qa_w, 2);
        lin(&w.q_a, &x, &mut qa, n_own, Dtype::F16, Dtype::F16, &mut s.ws, st, path)?;
        let mut qr = part(&s.q_resid, qa_w, 2);
        glm_glue::rmsnorm_f16(&qa, &w.q_a_norm, &mut qr, n_own, qa_w, RMS_EPS, st)?;
        let mut q = part(&s.q, qd, 2);
        lin(&w.q_b, &qr, &mut q, n_own, Dtype::F16, Dtype::F16, &mut s.ws, st, path)?;
    } else {
        lin(&w.q_a, &s.hidden, &mut s.q_a, rows, Dtype::F16, Dtype::F16, &mut s.ws, st, path)?;
        glm_glue::rmsnorm_f16(&s.q_a, &w.q_a_norm, &mut s.q_resid, rows, cfg.mla.q_lora_rank, RMS_EPS, st)?;
        lin(&w.q_b, &s.q_resid, &mut s.q, rows, Dtype::F16, Dtype::F16, &mut s.ws, st, path)?;
    }
    lin(&w.kv_a, &s.hidden, &mut s.kv_a, rows, Dtype::F16, Dtype::F16, &mut s.ws, st, path)?;
    glm_glue::rmsnorm_f16(&s.kv_a, &w.kv_a_norm, &mut s.ckv, rows, cfg.mla.kv_lora_rank, RMS_EPS, st)?;
    mark("mla: q_a, q_b, kv_a and their norms")?;
    mla::latent_write(&s.ckv, ri, table, latent, rows, st)?;
    moe::linear_f16(&s.hidden, Dtype::F16, &ix.wk, &mut s.k_raw, rows, h, cfg.indexer.head_dim, st)?;
    moe::linear_f16(&s.hidden, Dtype::F16, &ix.compress_gate, &mut s.gate, rows, h, cfg.indexer.head_dim, st)?;
    mark("mla: latent write, indexer wk and gate")?;
    if !big {
        dsa::append(&ix.dsa, planes, &mut s.dsa, &s.k_raw, &s.gate, ri, rows, table, st)?;
        mark("mla: dsa append")?;
        if any_sparse {
            moe::linear_f16(&s.hidden, Dtype::F16, &ix.weights_proj, &mut s.wproj, rows, h, cfg.indexer.heads, st)?;
            ix.wq_b.forward(&s.q_resid, &mut s.iq, rows, &mut s.ws, st, false)?;
            assert!(n_pools_max <= s.dsa.s_stride, "the indexer's score stride bounds the context");
            dsa::select(planes, &mut s.dsa, &s.iq, &s.wproj, ri, rows, n_pools_max, table, st)?;
            mark("mla: dsa select (weights_proj, wq_b, scoring, top-k)")?;
        }
        let sel = SparseSel { indices: &s.dsa.indices, meta: &s.dsa.meta, stride: dsa::K_STRIDE };
        // under P2 the absorb and expand of a prompt chunk's rows take the P2 tensor-core kernels (step (d)); the attention is P1's
        let rows_auto = if path == LinPath::P2 { glm_p2_mla::mla_rows_auto_p2 } else { mla::mla_rows_auto };
        rows_auto(
            &s.q,
            cfg.mla.heads * cfg.mla.qk_head_dim(),
            &w.kv_b,
            latent,
            table,
            ri,
            rows,
            &sel,
            &mut s.mla_out,
            mla_ctx.min(max_context),
            &mut s.mla,
            st,
        )?;
        mark("mla: attention")?;
        lin(&w.o, &s.mla_out, &mut s.mixer_out, rows, Dtype::F16, Dtype::F32, &mut s.ws, st, path)?;
        mark("mla: o_proj")?;
        return Ok(());
    }

    // a large chunk: the indexer's two remaining projections over the rows that select (only when some slab selects), then the slabs in
    // order (every slab's append, this rank's slabs' selection and attention)
    if any_sparse {
        if own.is_some() {
            let (x, qr) = (part(&s.hidden, h, 2), part(&s.q_resid, qa_w, 2));
            let mut wp = part(&s.wproj, cfg.indexer.heads, 2);
            let mut iq = part(&s.iq, cfg.indexer.heads * cfg.indexer.head_dim, 2);
            moe::linear_f16(&x, Dtype::F16, &ix.weights_proj, &mut wp, n_own, h, cfg.indexer.heads, st)?;
            ix.wq_b.forward(&qr, &mut iq, n_own, &mut s.ws, st, true)?;
        } else {
            moe::linear_f16(&s.hidden, Dtype::F16, &ix.weights_proj, &mut s.wproj, rows, h, cfg.indexer.heads, st)?;
            ix.wq_b.forward(&s.q_resid, &mut s.iq, rows, &mut s.ws, st, true)?;
        }
        mark("mla: indexer weights_proj and wq_b")?;
    }
    let (qd, ihd, ihs) = (cfg.mla.heads * cfg.mla.qk_head_dim(), cfg.indexer.head_dim, cfg.indexer.heads);
    let ovd = cfg.mla.heads * cfg.mla.v_head_dim;
    let mut a = 0usize;
    while a < rows {
        let n = (rows - a).min(MAX_ROWS);
        let slab_max = max_pos - (rows - (a + n));
        let ri_s = slab_index(ri, a, n);
        let (kraw, gate) = (s.k_raw.subview(a * ihd * 2, n * ihd * 2), s.gate.subview(a * ihd * 2, n * ihd * 2));
        dsa::append(&ix.dsa, planes, &mut s.dsa, &kraw, &gate, &ri_s, n, table, st)?;
        if let Some(h) = after_append.as_mut() {
            h(a, n, &*planes)?;
        }
        mark("mla: dsa append (slabs)")?;
        if a < lo || a >= hi {
            a += n;
            continue;
        }
        if dsa::is_sparse(slab_max) {
            let n_pools_max = (slab_max + 1) / dsa::P;
            let iq = s.iq.subview(a * ihs * ihd * 2, n * ihs * ihd * 2);
            let wproj = s.wproj.subview(a * ihs * 2, n * ihs * 2);
            assert!(n_pools_max <= s.dsa.s_stride, "the indexer's score stride bounds the context");
            dsa::select(planes, &mut s.dsa, &iq, &wproj, &ri_s, n, n_pools_max, table, st)?;
            mark("mla: dsa select (slabs)")?;
        }
        let sel = SparseSel { indices: &s.dsa.indices, meta: &s.dsa.meta, stride: dsa::K_STRIDE };
        let qv = s.q.subview(a * qd * 2, n * qd * 2);
        let mut ov = s.mla_out.subview(a * ovd * 2, n * ovd * 2);
        // the fused kernel: dense and sparse rows in one launch, the chunk partials merged in registers (same bits as `mla_rows_auto`, no workspace);
        // under P2 between the P2 absorb and expand (step (d))
        let rows_pf = if path == LinPath::P2 { glm_p2_mla::mla_rows_pf_p2 } else { mla_prefill::mla_rows_pf };
        rows_pf(&qv, qd, &w.kv_b, latent, table, &ri_s, n, &sel, &mut ov, &mut s.mla, st)?;
        mark("mla: attention (slabs)")?;
        a += n;
    }
    if own.is_some() {
        let x = part(&s.mla_out, ovd, 2);
        let mut y = part(&s.mixer_out, h, 4);
        lin(&w.o, &x, &mut y, n_own, Dtype::F16, Dtype::F32, &mut s.ws, st, path)?;
    } else {
        lin(&w.o, &s.mla_out, &mut s.mixer_out, rows, Dtype::F16, Dtype::F32, &mut s.ws, st, path)?;
    }
    mark("mla: o_proj")?;
    Ok(())
}

/// [`mla_dsa_mixer`]'s small path (a window, or a chunk of at most `MAX_ROWS` rows) on a rank of an MLA head split (lane E Phase 2,
/// `model/tp.rs`): the latent, the indexer and the selection as on one node (replicated), the query and the attention on the rank's heads
/// (its q_b strips, `mla_rows_auto_heads`), the o_proj by K-split partials swapped with the peer. Writes `s.mixer_out` exactly as the single
/// node does. The trunk's rows only (`s.rowidx`).
#[allow(clippy::too_many_arguments)]
pub(super) fn mla_dsa_mixer_heads(
    c: &MlaCtx<'_>,
    w: &MlaDevice,
    latent: &mut LatentCache,
    planes: &mut dsa::Planes,
    s: &mut Scratch,
    rows: usize,
    t: &mut super::tp::GlmTp,
) -> Result<(), ModelError> {
    let MlaCtx { st, cfg, table, max_context, n_pools_max, any_sparse, mla_ctx, big, lin: path } = *c;
    assert!(!big, "the MLA head split runs windows and small chunks only");
    assert!(path == LinPath::Decode, "the MLA head split runs the decode kernels (a P2 prompt chunk never takes it)");
    let (h0, nh) = t.mla_heads().ok_or_else(|| ModelError::Config("no MLA head split".into()))?;
    let ix = &w.indexer;
    let h = cfg.hidden;
    let qk = cfg.mla.qk_head_dim();
    lin(&w.q_a, &s.hidden, &mut s.q_a, rows, Dtype::F16, Dtype::F16, &mut s.ws, st, path)?;
    glm_glue::rmsnorm_f16(&s.q_a, &w.q_a_norm, &mut s.q_resid, rows, cfg.mla.q_lora_rank, RMS_EPS, st)?;
    w.q_b.forward_strip(
        h0 * qk / 128,
        (h0 + nh) * qk / 128,
        &s.q_resid,
        &mut s.q,
        rows,
        Dtype::F16,
        Dtype::F16,
        &mut s.ws,
        st,
    )?;
    lin(&w.kv_a, &s.hidden, &mut s.kv_a, rows, Dtype::F16, Dtype::F16, &mut s.ws, st, path)?;
    glm_glue::rmsnorm_f16(&s.kv_a, &w.kv_a_norm, &mut s.ckv, rows, cfg.mla.kv_lora_rank, RMS_EPS, st)?;
    mla::latent_write(&s.ckv, &s.rowidx, table, latent, rows, st)?;
    moe::linear_f16(&s.hidden, Dtype::F16, &ix.wk, &mut s.k_raw, rows, h, cfg.indexer.head_dim, st)?;
    moe::linear_f16(&s.hidden, Dtype::F16, &ix.compress_gate, &mut s.gate, rows, h, cfg.indexer.head_dim, st)?;
    dsa::append(&ix.dsa, planes, &mut s.dsa, &s.k_raw, &s.gate, &s.rowidx, rows, table, st)?;
    if any_sparse {
        moe::linear_f16(&s.hidden, Dtype::F16, &ix.weights_proj, &mut s.wproj, rows, h, cfg.indexer.heads, st)?;
        ix.wq_b.forward(&s.q_resid, &mut s.iq, rows, &mut s.ws, st, false)?;
        assert!(n_pools_max <= s.dsa.s_stride, "the indexer's score stride bounds the context");
        dsa::select(planes, &mut s.dsa, &s.iq, &s.wproj, &s.rowidx, rows, n_pools_max, table, st)?;
    }
    let sel = SparseSel { indices: &s.dsa.indices, meta: &s.dsa.meta, stride: dsa::K_STRIDE };
    mla::mla_rows_auto_heads(
        &s.q,
        nh * qk,
        &w.kv_b,
        h0,
        nh,
        latent,
        table,
        &s.rowidx,
        rows,
        &sel,
        &mut s.mla_out,
        mla_ctx.min(max_context),
        &mut s.mla,
        st,
    )?;
    t.split_o(&w.o, &s.mla_out, rows, &mut s.mixer_out, &mut s.ws, st)
}
