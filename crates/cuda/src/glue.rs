//! Embedding, norms, rope/qk-norm, gates, swiglu, small linears and top-k (kernels in `csrc/glue.cu`).
//!
//! Every function takes raw device pointers (`DevBuf::ptr()` / `ptr_at()`), launches on the given stream and returns
//! without synchronising. Activations are bf16 (`u16` on the host); norm weights are f32 with any `+1` already folded in.
//! Row invariance: a row's output bits depend on that row alone (fixed reduction order set by the row length).
//!
//! Shape/alignment limits the launchers enforce by returning `CudaError(1)` (cudaErrorInvalidValue), never by faulting:
//! `dim % 8 == 0` and 16-byte aligned pointers for the norms, gated norm and small linear (`k % 8 == 0`); see the notes
//! on each function. Zero rows (or zero elements) is a successful no-op.
//!
//! # Safety
//! All functions are `unsafe`: the pointers must be valid device addresses for the sizes implied by the arguments, and the
//! buffers must stay alive until the stream has run the launch.
#![allow(clippy::missing_safety_doc)] // the module docs above carry the one safety contract

use core::ffi::c_void;

use crate::rt::{CudaError, Stream};

/// Candidates that stage 1 of top-k keeps per block: a power of two >= `count` (16, 32 or 64).
pub fn topk_pool(count: usize) -> usize {
    match count {
        0..=16 => 16,
        17..=32 => 32,
        _ => 64,
    }
}

/// Vocabulary slice handled by one stage-1 block (mirrors `TK_CHUNK` in `glue.cu`).
pub const TOPK_CHUNK: usize = 16384;

/// Workspace bytes `topk` needs (0 for an unsupported `count`; `topk` then fails). Equals `gb_topk_workspace_bytes`.
pub fn topk_workspace_bytes(rows: usize, vocab: usize, count: usize) -> usize {
    if rows == 0 || vocab == 0 || count == 0 || count > 64 {
        return 0;
    }
    rows * vocab.div_ceil(TOPK_CHUNK) * topk_pool(count) * 8
}

/// Static shape of an attention prep: heads, kv heads, head dim, rotary dims, whether q carries a gate half.
///
/// Qwen3.8 target: `heads 24, kv_heads 4, head_dim 256, rotary_dim 64, has_gate true, theta 1e7, eps 1e-6` (norm weights
/// are `gamma - 1` in the pack: fold the `+1` first). DFlash2 draft: `32, 8, 128, rotary_dim 128, has_gate false`
/// (plain-gamma weights, rotate-half over all 128 dims).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttnPrepCfg {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub has_gate: bool,
    pub eps: f32,
    pub theta: f32,
}

impl AttnPrepCfg {
    pub const QWEN_TARGET: AttnPrepCfg =
        AttnPrepCfg { heads: 24, kv_heads: 4, head_dim: 256, rotary_dim: 64, has_gate: true, eps: 1e-6, theta: 1e7 };
    pub const DFLASH2_DRAFT: AttnPrepCfg =
        AttnPrepCfg { heads: 32, kv_heads: 8, head_dim: 128, rotary_dim: 128, has_gate: false, eps: 1e-6, theta: 1e7 };
    /// MiniCPM-V-4.7's text backbone (the Qwen3.5/3.6-35B-A3B config): 16 query heads over 2 KV heads of 256, partial rotary 0.25 (64
    /// of 256 dims, interleaved M-RoPE sections 11/11/10 through [`attn_prep_mrope`]), theta 1e7, q|gate interleaved per head. The
    /// kernel's `(J, JH) = (8, 1)` instantiation is the Qwen3.8-27B target's, so there is no new code; its DSpark draft (8 heads, no
    /// gate, rotary dims read from the draft config) takes the same kernel.
    pub const MINICPM_TARGET: AttnPrepCfg =
        AttnPrepCfg { heads: 16, kv_heads: 2, head_dim: 256, rotary_dim: 64, has_gate: true, eps: 1e-6, theta: 1e7 };
}

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_embed(
            table: *const c_void,
            ids: *const c_void,
            out: *mut c_void,
            rows: i32,
            dim: i32,
            vocab: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_rmsnorm(
            x: *const c_void,
            w: *const c_void,
            y: *mut c_void,
            rows: i32,
            dim: i32,
            eps: f32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_add_rmsnorm(
            x: *const c_void,
            res: *const c_void,
            w: *const c_void,
            h_out: *mut c_void,
            y: *mut c_void,
            rows: i32,
            dim: i32,
            eps: f32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_swiglu(g: *const c_void, u: *const c_void, out: *mut c_void, n: i64, s: *mut c_void) -> i32;
        pub fn gb_add(a: *const c_void, b: *const c_void, out: *mut c_void, n: i64, s: *mut c_void) -> i32;
        pub fn gb_gate_mul(o: *const c_void, gate: *const c_void, out: *mut c_void, n: i64, s: *mut c_void) -> i32;
        pub fn gb_attn_prep(
            q_in: *const c_void,
            k_in: *const c_void,
            positions: *const c_void,
            q_w: *const c_void,
            k_w: *const c_void,
            q_out: *mut c_void,
            gate_out: *mut c_void,
            k_out: *mut c_void,
            rows: i32,
            heads: i32,
            kv_heads: i32,
            head_dim: i32,
            rotary_dim: i32,
            has_gate: i32,
            eps: f32,
            theta: f32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_attn_prep_mrope(
            q_in: *const c_void,
            k_in: *const c_void,
            pos3: *const c_void,
            q_w: *const c_void,
            k_w: *const c_void,
            q_out: *mut c_void,
            gate_out: *mut c_void,
            k_out: *mut c_void,
            rows: i32,
            heads: i32,
            kv_heads: i32,
            head_dim: i32,
            rotary_dim: i32,
            has_gate: i32,
            eps: f32,
            theta: f32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_gated_rmsnorm(
            o: *const c_void,
            z: *const c_void,
            w: *const c_void,
            out: *mut c_void,
            rows: i32,
            heads: i32,
            dim: i32,
            eps: f32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_small_linear_f16w(
            x: *const c_void,
            w: *const c_void,
            y: *mut c_void,
            rows: i32,
            k: i32,
            n: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_small_linear_f16w_pair(
            x: *const c_void,
            w1: *const c_void,
            y1: *mut c_void,
            n1: i32,
            w2: *const c_void,
            y2: *mut c_void,
            n2: i32,
            rows: i32,
            k: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_small_linear_f16w_f32out(
            x: *const c_void,
            w: *const c_void,
            y: *mut c_void,
            rows: i32,
            k: i32,
            n: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_small_linear_f16w_rows(
            x: *const c_void,
            w: *const c_void,
            y: *mut c_void,
            rows: i32,
            k: i32,
            n: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_small_linear_f16w_f32out_rows(
            x: *const c_void,
            w: *const c_void,
            y: *mut c_void,
            rows: i32,
            k: i32,
            n: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_small_linear_f16w_rows_ref(
            x: *const c_void,
            w: *const c_void,
            y: *mut c_void,
            rows: i32,
            k: i32,
            n: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_topk_workspace_bytes(rows: i32, vocab: i32, count: i32) -> usize;
        pub fn gb_topk_valid_workspace_bytes(rows: i32, valid: i32, count: i32) -> usize;
        pub fn gb_topk_valid(
            logits: *const c_void,
            rows: i32,
            stride: i32,
            valid: i32,
            count: i32,
            ids: *mut c_void,
            vals: *mut c_void,
            ws: *mut c_void,
            ws_bytes: usize,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_topk(
            logits: *const c_void,
            rows: i32,
            vocab: i32,
            count: i32,
            ids: *mut c_void,
            vals: *mut c_void,
            ws: *mut c_void,
            ws_bytes: usize,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_copy_rows(
            src: *const c_void,
            src_stride: i64,
            dst: *mut c_void,
            dst_stride: i64,
            rows: i32,
            width: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_gather_rows(
            src: *const c_void,
            row_idx: *const c_void,
            dst: *mut c_void,
            n: i32,
            width: i32,
            s: *mut c_void,
        ) -> i32;
    }
}

/// Checked `usize -> i32` for FFI shape arguments (a too-large value is an invalid-value error, not a wrap).
fn i(v: usize) -> Result<i32, CudaError> {
    i32::try_from(v).map_err(|_| CudaError(1))
}

fn l(v: usize) -> Result<i64, CudaError> {
    i64::try_from(v).map_err(|_| CudaError(1))
}

/// `out[r, :] = table[ids[r], :]`; an id outside `0..vocab` (negative included) writes zeros. table bf16 `[vocab, dim]`,
/// ids i32 `[rows]`, out bf16 `[rows, dim]`.
pub unsafe fn embed(
    table: *const c_void,
    ids: *const c_void,
    out: *mut c_void,
    rows: usize,
    dim: usize,
    vocab: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_embed(table, ids, out, i(rows)?, i(dim)?, i(vocab)?, stream.raw()))
}

/// `y = bf16(x * rsqrt(mean(x^2) + eps) * w)`; x, y bf16 `[rows, dim]`, w f32 `[dim]`. `dim % 8 == 0`, 16-byte aligned
/// pointers. In place (`y == x`) is allowed.
pub unsafe fn rmsnorm(
    x: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    rows: usize,
    dim: usize,
    eps: f32,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_rmsnorm(x, w, y, i(rows)?, i(dim)?, eps, stream.raw()))
}

/// `h = bf16(x + res)` stored to `h_out` (the new residual stream), `y = rmsnorm(h)`. `h_out` may alias `x` or `res`; `y`
/// may alias `x` or `res`, but not `h_out`. Same shape rules as [`rmsnorm`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn add_rmsnorm(
    x: *const c_void,
    res: *const c_void,
    w: *const c_void,
    h_out: *mut c_void,
    y: *mut c_void,
    rows: usize,
    dim: usize,
    eps: f32,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_add_rmsnorm(x, res, w, h_out, y, i(rows)?, i(dim)?, eps, stream.raw()))
}

/// `out = bf16(silu(g) * u)` over `n` elements (fp32 math, `silu(x) = x / (1 + expf(-x))`). `out` may alias `g` or `u`.
pub unsafe fn swiglu(g: *const c_void, u: *const c_void, out: *mut c_void, n: usize, stream: &Stream) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_swiglu(g, u, out, l(n)?, stream.raw()))
}

/// `out = bf16(a + b)` over `n` elements. `out` may alias `a` or `b`.
pub unsafe fn add(a: *const c_void, b: *const c_void, out: *mut c_void, n: usize, stream: &Stream) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_add(a, b, out, l(n)?, stream.raw()))
}

/// `out = bf16(o * sigmoid(gate))` over `n` elements (fp32 math). `out` may alias `o`.
pub unsafe fn gate_mul(
    o: *const c_void,
    gate: *const c_void,
    out: *mut c_void,
    n: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_gate_mul(o, gate, out, l(n)?, stream.raw()))
}

/// Attention prep: per-head RMSNorm of q and k (fp32 weights, bf16 rounding), then rotate-half rope on the first
/// `rotary_dim` dims (pairs `(d, d + rotary_dim/2)`, `inv_freq[i] = theta^(-2i/rotary_dim)` in f64 then f32, angle in f32).
///
/// * `q_in`: `[rows, heads, 2*head_dim]` (`q | gate` per head) when `cfg.has_gate`, else `[rows, heads, head_dim]`;
/// * `k_in`: `[rows, kv_heads, head_dim]`; `positions`: i32 `[rows]`, absolute;
/// * `q_w`, `k_w`: f32 `[head_dim]` with the `+1` already folded in where the model needs it;
/// * `q_out`: `[rows, heads, head_dim]`; `gate_out`: `[rows, heads, head_dim]`, a copy of the gate halves, may be null when
///   `!has_gate`; `k_out`: `[rows, kv_heads, head_dim]`.
///
/// `head_dim` in {64, 128, 256, 512}, `rotary_dim` in {0, 64, 128, 256} and `<= head_dim`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_prep(
    cfg: &AttnPrepCfg,
    q_in: *const c_void,
    k_in: *const c_void,
    positions: *const c_void,
    q_w: *const c_void,
    k_w: *const c_void,
    q_out: *mut c_void,
    gate_out: *mut c_void,
    k_out: *mut c_void,
    rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_attn_prep(
        q_in,
        k_in,
        positions,
        q_w,
        k_w,
        q_out,
        gate_out,
        k_out,
        i(rows)?,
        i(cfg.heads)?,
        i(cfg.kv_heads)?,
        i(cfg.head_dim)?,
        i(cfg.rotary_dim)?,
        cfg.has_gate as i32,
        cfg.eps,
        cfg.theta,
        stream.raw(),
    ))
}

/// [`attn_prep`] with interleaved M-RoPE: `pos3` i32 `[rows, 3]` holds each row's (T, H, W) position and `cfg.rotary_dim` must be non-zero.
/// Frequency pair `i` uses stream `i % 3` (H only below 33, W only below 30; sections 11, 11, 10). With all three streams equal
/// the result is bit-identical to [`attn_prep`] at that position.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_prep_mrope(
    cfg: &AttnPrepCfg,
    q_in: *const c_void,
    k_in: *const c_void,
    pos3: *const c_void,
    q_w: *const c_void,
    k_w: *const c_void,
    q_out: *mut c_void,
    gate_out: *mut c_void,
    k_out: *mut c_void,
    rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_attn_prep_mrope(
        q_in,
        k_in,
        pos3,
        q_w,
        k_w,
        q_out,
        gate_out,
        k_out,
        i(rows)?,
        i(cfg.heads)?,
        i(cfg.kv_heads)?,
        i(cfg.head_dim)?,
        i(cfg.rotary_dim)?,
        cfg.has_gate as i32,
        cfg.eps,
        cfg.theta,
        stream.raw(),
    ))
}

/// GDN gated norm (spec 5): `out = bf16(o * rsqrt(mean_j o^2 + eps) * w * silu(z))` per `(row, head)` over `dim`.
/// o, z, out bf16 `[rows, heads, dim]`, w f32 `[dim]` plain gamma (no +1). `dim` in {128, 256}; 16-byte aligned; `out == o`
/// is allowed.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gated_rmsnorm(
    o: *const c_void,
    z: *const c_void,
    w: *const c_void,
    out: *mut c_void,
    rows: usize,
    heads: usize,
    dim: usize,
    eps: f32,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_gated_rmsnorm(o, z, w, out, i(rows)?, i(heads)?, i(dim)?, eps, stream.raw()))
}

/// `y[r, j] = bf16(sum_k x[r, k] * w[j, k])`, fp32 accumulation in an order set by `k` alone. x bf16 `[rows, k]`, w f16
/// `[n, k]` row-major (PyTorch `[out, in]`), y bf16 `[rows, n]`. `k % 8 == 0`, x and w 16-byte aligned.
pub unsafe fn small_linear_f16w(
    x: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    rows: usize,
    k: usize,
    n: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_small_linear_f16w(x, w, y, i(rows)?, i(k)?, i(n)?, stream.raw()))
}

/// [`small_linear_f16w`] of two weights `w1` `[n1, k]` and `w2` `[n2, k]` on the same `x` in ONE launch (`y1` `[rows, n1]`, `y2`
/// `[rows, n2]`): the very bits of two separate calls, without the second waiting for the first.
pub unsafe fn small_linear_f16w_pair(
    x: *const c_void,
    w1: *const c_void,
    y1: *mut c_void,
    n1: usize,
    w2: *const c_void,
    y2: *mut c_void,
    n2: usize,
    rows: usize,
    k: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_small_linear_f16w_pair(x, w1, y1, i(n1)?, w2, y2, i(n2)?, i(rows)?, i(k)?, stream.raw()))
}

/// Same as [`small_linear_f16w`] but `y` is f32 `[rows, n]` (the unrounded accumulator).
pub unsafe fn small_linear_f16w_f32out(
    x: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    rows: usize,
    k: usize,
    n: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_small_linear_f16w_f32out(x, w, y, i(rows)?, i(k)?, i(n)?, stream.raw()))
}

/// [`small_linear_f16w`] for many rows (a prefill chunk): the SAME bits (the per-output fp32 chain, the butterfly and the warp order
/// are the kernel's own; tests/glue.rs compares the two over row counts, widths and `k`), 8 rows by 8 columns a block, so a weight
/// row is read once an 8-row group and an input row once an 8-column group.
pub unsafe fn small_linear_f16w_rows(
    x: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    rows: usize,
    k: usize,
    n: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_small_linear_f16w_rows(x, w, y, i(rows)?, i(k)?, i(n)?, stream.raw()))
}

/// [`small_linear_f16w_rows`] through the weight-stationary kernel it replaced (selects in the reduce-scatter): the timing and test reference.
pub unsafe fn small_linear_f16w_rows_ref(
    x: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    rows: usize,
    k: usize,
    n: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_small_linear_f16w_rows_ref(x, w, y, i(rows)?, i(k)?, i(n)?, stream.raw()))
}

/// [`small_linear_f16w_rows`] with f32 output (the unrounded accumulator, as [`small_linear_f16w_f32out`]).
pub unsafe fn small_linear_f16w_f32out_rows(
    x: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    rows: usize,
    k: usize,
    n: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_small_linear_f16w_f32out_rows(x, w, y, i(rows)?, i(k)?, i(n)?, stream.raw()))
}

/// Exact per-row top-`count` of bf16 logits `[rows, vocab]` by (value DESC, id ASC) (every column is a candidate: for a padded
/// vocabulary use [`topk_valid`]). `ids` i32 `[rows, count]`, `vals` f32
/// `[rows, count]` (bf16 widened; NaN counts as and is reported as -inf; -0 ties with +0). `1 <= count <= min(64, vocab)`.
/// `workspace` must hold [`topk_workspace_bytes`] bytes, 16-byte aligned.
#[allow(clippy::too_many_arguments)]
pub unsafe fn topk(
    logits: *const c_void,
    rows: usize,
    vocab: usize,
    count: usize,
    ids: *mut c_void,
    vals: *mut c_void,
    workspace: *mut c_void,
    ws_bytes: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_topk(logits, i(rows)?, i(vocab)?, i(count)?, ids, vals, workspace, ws_bytes, stream.raw()))
}

/// The C-side workspace formula (for a cross-check against [`topk_workspace_bytes`]).
pub fn topk_workspace_bytes_c(rows: usize, vocab: usize, count: usize) -> usize {
    unsafe { ffi::gb_topk_workspace_bytes(rows as i32, vocab as i32, count as i32) }
}

/// Workspace bytes [`topk_valid`] needs: it depends on the valid columns only (the chunks of a row that are read), so it is
/// [`topk_workspace_bytes`] of `valid`.
pub fn topk_valid_workspace_bytes(rows: usize, valid: usize, count: usize) -> usize {
    topk_workspace_bytes(rows, valid, count)
}

/// The C-side formula of [`topk_valid_workspace_bytes`].
pub fn topk_valid_workspace_bytes_c(rows: usize, valid: usize, count: usize) -> usize {
    unsafe { ffi::gb_topk_valid_workspace_bytes(rows as i32, valid as i32, count as i32) }
}

/// [`topk`] over the first `valid` columns of rows that are `stride` columns apart (`valid <= stride`): logits bf16
/// `[rows, stride]`. Columns `[valid, stride)` are never read, so an id `>= valid` is never returned whatever they hold: the
/// padded lm_head of a vocabulary that is not a multiple of 128 (MiniCPM-V-4.7: stride 248192, valid 248144; the GLM
/// `glm_glue::topk_f16(valid)` precedent) leaves its pad rows' garbage in those columns. `1 <= count <= min(64, valid)`;
/// `workspace` holds [`topk_valid_workspace_bytes`] bytes, 16-byte aligned. With `valid == stride` this is exactly [`topk`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn topk_valid(
    logits: *const c_void,
    rows: usize,
    stride: usize,
    valid: usize,
    count: usize,
    ids: *mut c_void,
    vals: *mut c_void,
    workspace: *mut c_void,
    ws_bytes: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_topk_valid(
        logits,
        i(rows)?,
        i(stride)?,
        i(valid)?,
        i(count)?,
        ids,
        vals,
        workspace,
        ws_bytes,
        stream.raw(),
    ))
}

/// `dst[r, :width] = src[r, :width]` for `r < rows`, with element strides `src_stride` / `dst_stride` between rows (bf16).
/// Use it to export tap rows into a `[rows, 5, 5120]` buffer: `dst = base + tap * 5120`, `dst_stride = 5 * 5120`.
pub unsafe fn copy_rows(
    src: *const c_void,
    src_stride: usize,
    dst: *mut c_void,
    dst_stride: usize,
    rows: usize,
    width: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_copy_rows(src, l(src_stride)?, dst, l(dst_stride)?, i(rows)?, i(width)?, stream.raw()))
}

/// `dst[i, :] = src[row_idx[i], :]` for `i < n` (bf16 rows of `width` elements, contiguous); a negative index writes zeros.
pub unsafe fn gather_rows(
    src: *const c_void,
    row_idx: *const c_void,
    dst: *mut c_void,
    n: usize,
    width: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_gather_rows(src, row_idx, dst, i(n)?, i(width)?, stream.raw()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_arithmetic() {
        assert_eq!(topk_pool(1), 16);
        assert_eq!(topk_pool(16), 16);
        assert_eq!(topk_pool(28), 32);
        assert_eq!(topk_pool(64), 64);
        assert_eq!(topk_workspace_bytes(8, 248320, 16), 8 * 248320usize.div_ceil(TOPK_CHUNK) * 16 * 8);
        assert_eq!(topk_workspace_bytes(1, 100, 65), 0);
    }
}
