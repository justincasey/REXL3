//! GLM-5.3-Flash prefill under contract P2 (Lane P; `families/glm/contract.rs` has the contract): the dense EXL3 linears of a prompt chunk
//! on tensor cores, at every chunk width.
//!
//! **P2 here, in short.** (a) A row's output bits depend on that row and the weights only, never on the chunk's width, its boundary, the
//! padding or the row order (mma.sync keeps rows apart; rows past the call are zero-filled in shared memory). (b) Every output's K sum is
//! in an order that is a function of K alone ([`ARITH_ID`] names it): the tile shape and M never change a bit. (c) Activations enter
//! the MMA as fp16 (`(x * suh) H_128 / sqrt 128` computed in fp32, rounded once), the trellis decoded by the decode path's own decoder
//! (`exl3_linear_decode.cuh`, bit for bit), fp32 accumulation. (d) Deterministic: no atomics, no data-dependent schedule. The bits
//! differ from the decode path's (accepted by the owner for prefill rows); decode and verify windows never come here.
//!
//! Which layers: every dense EXL3 linear of a GLM prompt chunk (KDA qkv / o, MLA q_a / q_b / kv_a / o, the NextN layer's eh_proj) through
//! [`linear`], and the two gated MLPs (the shared expert, the dense MLP of layers 0-2: `moe::SharedExpert`) through [`gated`]. The
//! router, the DSA indexer, the routed experts and the lm_head are not P2 (the model never sends them here).
//!
//! **Kernel** (`csrc/glm_p2_dense.cu`): GLM's own copy of the tensor-core GEMM REXL3 built for Nemotron's dense prefill linears
//! (`csrc/nemotron_dense_mma.cu`), so that a change there cannot move a GLM P2 bit. It takes every GLM dense shape (K and N multiples of
//! 128, Mul1 at 2 to 8 bits, fp16 scales). Two K orders ([`Variant`]): `Chain`, ONE chain of m16n8k16 MMAs over the k tiles in order from
//! +0.0 (bit for bit Nemotron's `gb_ndm_gemm`), and `Fold128`, per 128-k block a chain from +0.0 added to the running sum in block order
//! (a single long chain picks up a systematic shrink that grows with K on GB10). [`DEFAULT_VARIANT`] is what a GLM P2 prefill runs, and
//! [`ARITH_ID`] names it. The gated MLP is gate -> up (fp32) -> the routed experts' clamped SwiGLU (`gb_moe_swiglu_limit`,
//! elementwise) -> the down projection on the fp16 activation.
//!
//! **No process switch.** The model decides per forward (its contract and the forward's kind); this module has no global opt-in. It
//! counts its GEMM launches ([`launches`]) so a test can audit that every prompt row of a P2 prefill went through it.

use crate::exl3_linear::{Dtype, Exl3Weights};
use crate::moe::{SharedExpert, SharedScratch};
use crate::moe_prefill::SharedScratchBig;
use crate::rt::{CudaError, DevBuf, Stream};
use core::ffi::c_void;
use gb_exl3::Codebook;
use std::sync::atomic::{AtomicU64, Ordering};

/// The K order of the P2 GEMM (both P2: a function of K alone, row-independent, deterministic).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// One chain of MMAs over every k tile, from +0.0.
    Chain,
    /// Per 128-k block a chain from +0.0, the blocks' partials added in block order.
    Fold128,
}

impl Variant {
    /// The id the arithmetic tag carries.
    pub const fn id(self) -> &'static str {
        match self {
            Variant::Chain => "glm-chain-v1",
            Variant::Fold128 => "glm-fold128-v1",
        }
    }
}

/// The K order a GLM P2 prefill runs: `Fold128` (Lane P, MEASURED on node 4, `tests/glm_p2_dense.rs`): against an fp64 reference the
/// single chain shrinks the outputs of the long-K layers by 1.0e-5 to 1.3e-5 of their rms (K 12,288 and 16,384: the dense MLP's down, MLA
/// o_proj), the decode path and `Fold128` by under 1e-6 at the same rms error; `Fold128` costs 5 to 8 % of the GEMM's time (64 to 74
/// against 70 to 84 TFLOP/s at 2,048 rows on the big shapes), about 1 % of a prompt chunk (COMPUTED).
pub const DEFAULT_VARIANT: Variant = Variant::Fold128;

/// The version of the P2 dense arithmetic this module runs (the K order, the rounding points, the epilogue order): GLM's prefix namespace
/// and its TP Hello carry it, so any change that moves a P2 bit must change this string.
pub const ARITH_ID: &str = DEFAULT_VARIANT.id();

const INVALID: CudaError = CudaError(1);

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_glm_p2_supported(k: i32, n: i32, bits: i32, codebook: i32) -> i32;
        pub fn gb_glm_p2_rotate(x: *const c_void, x_dtype: i32, ldx: i64, suh: *const c_void, suh_f32: i32, a: *mut c_void, m: i32, k: i32,
                                stream: *mut c_void) -> i32;
        pub fn gb_glm_p2_gemm(a: *const c_void, trellis: *const c_void, svh: *const c_void, bias: *const c_void, scale_f32: i32,
                              y: *mut c_void, y_dtype: i32, ldy: i64, m: i32, k: i32, n: i32, bits: i32, codebook: i32, fold: i32,
                              tile_force: i32, stream: *mut c_void) -> i32;
        pub fn gb_glm_p2_pick_tile(m: i32, k: i32, n: i32, fold: i32) -> i32;
        pub fn gb_moe_swiglu_limit(g: *const c_void, u: *const c_void, a: *mut c_void, n: i64, limit: f32, stream: *mut c_void) -> i32;
    }
}

static LAUNCHES: AtomicU64 = AtomicU64::new(0);

/// P2 GEMM launches of this process so far (every [`linear`] call and each of a [`gated`] call's three projections count one). A test reads
/// it before and after a prefill to check that every dense linear of every prompt row went through P2.
pub fn launches() -> u64 {
    LAUNCHES.load(Ordering::Relaxed)
}

thread_local! {
    static LAUNCHES_HERE: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
}

/// [`launches`] made by THIS thread: the audit of one rank where several ranks run in one process (the loopback TP test, a rank a thread).
pub fn launches_here() -> u64 {
    LAUNCHES_HERE.with(|c| c.get())
}

fn codebook_code(cb: Codebook) -> i32 {
    match cb {
        Codebook::Inst3 => 0,
        Codebook::Mcg => 1,
        Codebook::Mul1 => 2,
    }
}

/// Whether the P2 GEMM takes this layer (K and N multiples of 128, Mul1 at 2 to 8 bits: every GLM pack's dense linears).
pub fn supported(w: &Exl3Weights) -> bool {
    // SAFETY: a pure function of its integer arguments
    unsafe { ffi::gb_glm_p2_supported(w.k as i32, w.n as i32, w.bits as i32, codebook_code(w.codebook)) == 1 }
}

/// The tile of the fixed-tile kernels the GEMM picks for `m` rows of a `k` x `n` layer under `v` (`BM * 10 + SN`), for the tables (a chain
/// layer of 3 to 8 bits runs the 128 x 128 chain GEMM instead).
pub fn pick_tile(m: usize, k: usize, n: usize, v: Variant) -> i32 {
    // SAFETY: a pure function of its integer arguments
    unsafe { ffi::gb_glm_p2_pick_tile(m as i32, k as i32, n as i32, (v == Variant::Fold128) as i32) }
}

impl SharedExpert {
    /// The gate linear (hidden -> inter).
    pub fn gate_weights(&self) -> &Exl3Weights {
        &self.gate
    }
    /// The up linear (hidden -> inter).
    pub fn up_weights(&self) -> &Exl3Weights {
        &self.up
    }
    /// The down linear (inter -> hidden).
    pub fn down_weights(&self) -> &Exl3Weights {
        &self.down
    }
}

/// Whether a gated MLP's three linears are all [`supported`].
pub fn gated_supported(m: &SharedExpert) -> bool {
    supported(&m.gate) && supported(&m.up) && supported(&m.down)
}

/// Bytes of the rotated fp16 input of `rows` rows of a layer `k` wide: what [`linear`] needs of its workspace.
pub fn workspace_bytes(rows: usize, k: usize) -> usize {
    rows * k * 2
}

/// `y [rows, n] = w(x [rows, k])` under P2, any `rows >= 1`. `x` of `x_dtype`, `y` of `y_dtype` (row-major, packed). `ws` needs
/// [`workspace_bytes`]`(rows, k)` bytes (its contents on entry do not matter; it holds the rotated input when the call returns).
/// Asynchronous on `st`.
#[allow(clippy::too_many_arguments)]
pub fn linear(
    w: &Exl3Weights,
    x: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    x_dtype: Dtype,
    y_dtype: Dtype,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    linear_tiled(w, x, y, rows, x_dtype, y_dtype, ws, 0, st)
}

/// [`linear`] with a forced tile shape (`0`: the kernel's pick, else `BM * 10 + SN`: 641, 642, 1281, 1282, or for the chain order
/// `10000 +` a configuration of the 128 x 128 chain GEMM: 10001 to 10004). The tile never changes a bit (the tests check that).
#[allow(clippy::too_many_arguments)]
pub fn linear_tiled(
    w: &Exl3Weights,
    x: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    x_dtype: Dtype,
    y_dtype: Dtype,
    ws: &mut DevBuf,
    tile: i32,
    st: &Stream,
) -> Result<(), CudaError> {
    linear_variant(w, x, y, rows, x_dtype, y_dtype, ws, DEFAULT_VARIANT, tile, st)
}

/// [`linear_tiled`] in a given K order (the tests and the bench compare the two; the model runs [`DEFAULT_VARIANT`]).
#[allow(clippy::too_many_arguments)]
pub fn linear_variant(
    w: &Exl3Weights,
    x: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    x_dtype: Dtype,
    y_dtype: Dtype,
    ws: &mut DevBuf,
    v: Variant,
    tile: i32,
    st: &Stream,
) -> Result<(), CudaError> {
    if rows == 0
        || rows > i32::MAX as usize / w.k.max(w.n).max(1)
        || !supported(w)
        || x.bytes() < rows * w.k * x_dtype.size()
        || y.bytes() < rows * w.n * y_dtype.size()
        || ws.bytes() < workspace_bytes(rows, w.k)
    {
        return Err(INVALID);
    }
    // SAFETY: the sizes were checked above; every pointer is a device allocation of at least that size; asynchronous on `st`
    unsafe {
        CudaError::check(ffi::gb_glm_p2_rotate(
            x.ptr() as *const c_void,
            x_dtype as i32,
            w.k as i64,
            w.suh.ptr() as *const c_void,
            0,
            ws.ptr(),
            rows as i32,
            w.k as i32,
            st.raw(),
        ))?;
        CudaError::check(ffi::gb_glm_p2_gemm(
            ws.ptr() as *const c_void,
            w.trellis.ptr() as *const c_void,
            w.svh.ptr() as *const c_void,
            w.bias.as_ref().map_or(core::ptr::null(), |b| b.ptr() as *const c_void),
            0,
            y.ptr(),
            y_dtype as i32,
            w.n as i64,
            rows as i32,
            w.k as i32,
            w.n as i32,
            w.bits as i32,
            codebook_code(w.codebook),
            (v == Variant::Fold128) as i32,
            tile,
            st.raw(),
        ))?;
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    LAUNCHES_HERE.with(|c| c.set(c.get() + 1));
    Ok(())
}

/// The buffers a gated MLP's P2 forward works in: the fp32 gate and up outputs, the fp16 activation and the EXL3 workspace of one piece
/// of at most `max_rows` rows (a decode-size [`SharedScratch`] or a prefill-size [`SharedScratchBig`]; their contents never matter).
pub struct GatedBufs<'a> {
    pub max_rows: usize,
    pub inter: usize,
    pub g: &'a mut DevBuf,
    pub u: &'a mut DevBuf,
    pub a: &'a mut DevBuf,
    pub ws: &'a mut DevBuf,
}

/// Whether buffers of these sizes hold a [`gated`] piece of `max_rows` rows of `m` (the gate and up outputs, the activation, the rotated
/// input of the widest of its three linears).
fn bufs_ok(m: &SharedExpert, max_rows: usize, inter: usize, g: &DevBuf, u: &DevBuf, a: &DevBuf, ws: &DevBuf) -> bool {
    let k = m.hidden.max(m.inter);
    max_rows >= 1
        && inter == m.inter
        && g.bytes() >= max_rows * m.inter * 4
        && u.bytes() >= max_rows * m.inter * 4
        && a.bytes() >= max_rows * m.inter * 2
        && ws.bytes() >= workspace_bytes(max_rows, k)
}

impl SharedScratch {
    /// This scratch as the buffers of [`gated`].
    pub fn p2_bufs(&mut self) -> GatedBufs<'_> {
        GatedBufs { max_rows: self.max_rows, inter: self.inter, g: &mut self.g, u: &mut self.u, a: &mut self.a, ws: &mut self.ws }
    }

    /// Whether this scratch holds a [`gated`] piece of its rows of `m`.
    pub fn p2_ok(&self, m: &SharedExpert) -> bool {
        bufs_ok(m, self.max_rows, self.inter, &self.g, &self.u, &self.a, &self.ws)
    }
}

impl SharedScratchBig {
    /// This scratch as the buffers of [`gated`].
    pub fn p2_bufs(&mut self) -> GatedBufs<'_> {
        GatedBufs { max_rows: self.max_rows, inter: self.inter, g: &mut self.g, u: &mut self.u, a: &mut self.a, ws: &mut self.ws }
    }

    /// Whether this scratch holds a [`gated`] piece of its rows of `m`.
    pub fn p2_ok(&self, m: &SharedExpert) -> bool {
        bufs_ok(m, self.max_rows, self.inter, &self.g, &self.u, &self.a, &self.ws)
    }
}

/// `out [rows, hidden]` fp32 = down(swiglu_limit(gate(x), up(x))) under P2, `x [rows, hidden]` of `x_dtype`, any `rows >= 1`: in pieces of
/// the scratch's rows (a row's bits do not depend on the piece). `x` and `out` must be sub-viewable (`DevBuf::zeroed_view`) when `rows`
/// exceeds the scratch's `max_rows`. Asynchronous on `st`.
pub fn gated(
    m: &SharedExpert,
    x: &DevBuf,
    x_dtype: Dtype,
    rows: usize,
    out: &mut DevBuf,
    b: GatedBufs<'_>,
    st: &Stream,
) -> Result<(), CudaError> {
    if rows == 0
        || b.max_rows == 0
        || b.inter != m.inter
        || x.bytes() < rows * m.hidden * x_dtype.size()
        || out.bytes() < rows * m.hidden * 4
        || !gated_supported(m)
    {
        return Err(INVALID);
    }
    let GatedBufs { max_rows, g, u, a, ws, .. } = b;
    let mut at = 0;
    while at < rows {
        let n = max_rows.min(rows - at);
        let xv;
        let mut ov;
        let (xr, or): (&DevBuf, &mut DevBuf) = if rows <= max_rows {
            (x, &mut *out)
        } else {
            xv = x.subview(at * m.hidden * x_dtype.size(), n * m.hidden * x_dtype.size());
            ov = out.subview(at * m.hidden * 4, n * m.hidden * 4);
            (&xv, &mut ov)
        };
        linear(&m.gate, xr, g, n, x_dtype, Dtype::F32, ws, st)?;
        linear(&m.up, xr, u, n, x_dtype, Dtype::F32, ws, st)?;
        // SAFETY: g, u hold n * inter f32 and a n * inter fp16 (the scratch is sized for max_rows >= n); asynchronous on `st`
        unsafe {
            CudaError::check(ffi::gb_moe_swiglu_limit(
                g.ptr() as *const c_void,
                u.ptr() as *const c_void,
                a.ptr(),
                (n * m.inter) as i64,
                m.limit,
                st.raw(),
            ))?;
        }
        linear(&m.down, a, or, n, Dtype::F16, Dtype::F32, ws, st)?;
        at += n;
    }
    Ok(())
}
