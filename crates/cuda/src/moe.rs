//! The MoE layer of GLM-5.3-Flash on the GPU: sigmoid router with selection bias, grouped EXL3 routed experts, slot combine.
//! Kernels: `csrc/moe_router.cu`, `csrc/moe_experts.cu` (design and the row-invariance argument are in their headers; the expert
//! kernel is ported from TensorFold's grouped EXL3 GEMV, MIT, pinned 9cd52ab, see THIRD_PARTY_NOTICES.md).
//!
//! The contract every function here keeps: for a pair (row, slot) every number depends on that row's input, the expert it picked
//! and the layer's weights alone, never on how many rows share the call (1..=128) or on what the other rows picked. Nothing
//! here synchronizes with the host, allocates in a launch path, or uses atomics.
//!
//! # The interface the model code calls (nothing here synchronizes with the host)
//!
//! ```text
//! // once per process
//! let tab = SigmoidTable::new()?;                                    // 65,536 fp32 (256 KiB), the router's sigmoid
//! // once per MoE layer (3..=44 of the GLM pack; layer 45 is the MTP layer, same shapes at 4 bits)
//! let layer  = MoeLayer::from_checkpoint(&ck, l, MoeDims::GLM53)?;   // fp16 gate, centred fp16 bias, 288 experts at the layer's width
//! let shared = SharedExpert::from_checkpoint(&ck, l, MoeDims::GLM53.limit)?;
//! // once per stream (contents on entry never matter, so every layer shares them): ~100 MiB at 128 rows
//! let mut s  = MoeScratch::new(&MoeDims::GLM53, MAX_ROWS)?;
//! let mut ss = SharedScratch::new(&shared, MAX_ROWS)?;
//! // per layer and step: x [rows, 4096] fp16 (bf16, fp32 also accepted), 1 <= rows <= 128
//! shared.forward(&x, Dtype::F16, rows, &mut sh_out_f32, &mut ss, &st)?;               // may run on a second stream
//! layer.forward(&x, Dtype::F16, rows, &tab, Some(&sh_out_f32), &mut out_f32, &mut s, &st)?;
//!   // = layer.route(..) (router logits, select) then layer.experts_picked(..) (group, rotate, gate/up, activation, down, combine);
//!   // s.pick / s.wts hold the picks and fp16 weights until the next call (telemetry: experts per window, the expert union)
//! ```
//!
//! `out` is fp32 [rows, 4096]: the routed experts summed in slot (rank) order with `fmaf(weight, y, acc)`, then the shared expert's
//! fp32 output added last. Rows above 128 (prefill) are a different path (batched reconstruct + GEMM, not built here).
//! `MoeScratch::set_picks` lets a caller route elsewhere (tests, a different router); a pick outside `0..experts` contributes nothing.

use crate::exl3_linear::{Dtype, Exl3Weights};
pub use crate::moe_owned::{combine_pairs, ExpertOwnership, OwnScratch};
pub use crate::moe_subbanks::{subbank_partition, MoeSubbankLayer, MoeSubbanks, SubbankKey, SubbankRegion};
use crate::rt::{CudaError, DevBuf, Stream};
use gb_exl3::checkpoint::{Checkpoint, LoadedGroup};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_exl3::Codebook;

#[allow(non_snake_case)] // cudart event names; in a stub build they become Rust functions
mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_moe_router_logits(
            x: *const c_void,
            x_dtype: i32,
            gate: *const c_void,
            logits: *mut c_void,
            rows: i32,
            k: i32,
            e: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_router_logits_ex(
            x: *const c_void,
            x_dtype: i32,
            gate: *const c_void,
            logits: *mut c_void,
            rows: i32,
            k: i32,
            e: i32,
            rb: i32,
            warps: i32,
            un: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_router_logits_ex2(
            x: *const c_void,
            x_dtype: i32,
            gate: *const c_void,
            logits: *mut c_void,
            rows: i32,
            k: i32,
            e: i32,
            rb: i32,
            eb: i32,
            warps: i32,
            un: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_router_select(
            logits: *const c_void,
            sig: *const c_void,
            bias: *const c_void,
            pick: *mut c_void,
            wts: *mut c_void,
            scale: f32,
            rows: i32,
            e: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_router_route(
            x: *const c_void,
            x_dtype: i32,
            gate: *const c_void,
            logits: *mut c_void,
            sig: *const c_void,
            bias: *const c_void,
            pick: *mut c_void,
            wts: *mut c_void,
            scale: f32,
            rows: i32,
            k: i32,
            e: i32,
            topk: i32,
            done: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_default_plan(k: i32, n: i32, bits: i32, mats: i32, out: *mut i32) -> i32;
        pub fn gb_moe_swiglu_limit(g: *const c_void, u: *const c_void, a: *mut c_void, n: i64, limit: f32, stream: *mut c_void) -> i32;
        pub fn gb_moe_swiglu_clamped(g: *const c_void, u: *const c_void, a: *mut c_void, n: i64, limit: f32, stream: *mut c_void) -> i32;
        pub fn gb_moe_read_probe(p: *const c_void, bytes: usize, out: *mut c_void, blocks: i32, stream: *mut c_void) -> i32;
        pub fn cudaEventCreate(e: *mut *mut c_void) -> i32;
        pub fn cudaEventDestroy(e: *mut c_void) -> i32;
        pub fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> i32;
        pub fn cudaEventSynchronize(e: *mut c_void) -> i32;
        pub fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> i32;
        pub fn gb_moe_group(
            pick: *const c_void,
            uids: *mut c_void,
            ucount: *mut c_void,
            mcount: *mut c_void,
            members: *mut c_void,
            rows: i32,
            topk: i32,
            e: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_rot_in(
            x: *const c_void,
            x_dtype: i32,
            pick: *const c_void,
            suh0: *const c_void,
            suh1: *const c_void,
            out0: *mut c_void,
            out1: *mut c_void,
            rows: i32,
            topk: i32,
            k: i32,
            e: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_gemv(
            x0: *const c_void,
            x1: *const c_void,
            tp0: *const c_void,
            tp1: *const c_void,
            uids: *const c_void,
            ucount: *const c_void,
            mcount: *const c_void,
            members: *const c_void,
            z: *mut c_void,
            mats: i32,
            k: i32,
            n: i32,
            p: i32,
            bits: i32,
            codebook: i32,
            sk: i32,
            wk: i32,
            umax: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_gateup_epilogue(
            z: *const c_void,
            pick: *const c_void,
            svh_g: *const c_void,
            svh_u: *const c_void,
            suh_d: *const c_void,
            xd: *mut c_void,
            rows: i32,
            topk: i32,
            n: i32,
            sk: i32,
            e: i32,
            limit: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_down_combine(
            z: *const c_void,
            pick: *const c_void,
            svh_d: *const c_void,
            wts: *const c_void,
            shared: *const c_void,
            y_pairs: *mut c_void,
            out: *mut c_void,
            rows: i32,
            topk: i32,
            d: i32,
            sk: i32,
            e: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

const INVALID: CudaError = CudaError(1);

/// Most rows one call takes (the decode contract of the EXL3 kernels).
pub const MAX_ROWS: usize = 128;

fn check(rc: i32) -> Result<(), CudaError> {
    CudaError::check(rc)
}

fn codebook_code(cb: Codebook) -> i32 {
    match cb {
        Codebook::Inst3 => 0,
        Codebook::Mcg => 1,
        Codebook::Mul1 => 2,
    }
}

// ----------------------------------------------------------------------------------------------------------------
// host mirrors (the bit-exact reference for the router kernels)

/// sigmoid of every fp16 value, rounded once to fp32 from float64 (index = the fp16 bits). NaN stays NaN.
pub fn sigmoid_table_host() -> Vec<f32> {
    (0u32..65536)
        .map(|h| {
            let x = f16_to_f64(h as u16);
            if x.is_nan() {
                f32::NAN
            } else {
                (1.0 / (1.0 + (-x).exp())) as f32
            }
        })
        .collect()
}

/// torch's CUDA `mean` of an fp32 vector, reproduced on the host: 64 threads, each taking 4-element chunks t, t + 64, ... into four lane
/// accumulators (combined lane 0 + 1 + 2 + 3), the 64 thread sums added by a halving tree (offsets 32 .. 1), then multiplied by `1 / n` in
/// fp32. Found by searching reduction orders against the device (`tools/explore_bias_mean.py`): it equals torch's mean on the 288-entry
/// biases of all 43 MoE layers of the GLM pack, where the float64 mean rounded to fp32 differs on 14 of them. Other lengths use the same
/// order (unverified).
pub fn torch_mean_f32(x: &[f32]) -> f32 {
    let mut lanes = [[0f32; 4]; 64];
    for (i, &v) in x.iter().enumerate() {
        let (chunk, e) = (i / 4, i % 4);
        lanes[chunk % 64][e] += v;
    }
    let mut acc: Vec<f32> = lanes.iter().map(|l| ((l[0] + l[1]) + l[2]) + l[3]).collect();
    let mut h = 32;
    while h >= 1 {
        for i in 0..h {
            acc[i] += acc[i + h];
        }
        h /= 2;
    }
    acc[0] * (1.0f32 / x.len() as f32)
}

/// The oracle's selection bias: `(esb - esb.mean()).half()`, as exact fp32 values, bit for bit what the fork computes on the device
/// (`torch_mean_f32` for the mean; checked against the device on every MoE layer of the pack by `centred_bias_matches_the_oracle_on_every_layer`).
pub fn centred_bias_host(esb: &[f32]) -> Vec<f32> {
    let mean = torch_mean_f32(esb);
    esb.iter().map(|&v| f16_to_f64(f64_to_f16((v - mean) as f64)) as f32).collect()
}

/// fp16 logits of every row: the router GEMV's arithmetic, lane by lane (two fmaf per half2 pair over pairs j, j + 32, ...,
/// then the shfl_down tree). `x` holds the values the kernel reads (fp16, bf16 or fp32 already decoded to f32).
pub fn router_logits_host(x: &[f32], gate: &[u16], rows: usize, k: usize, e: usize) -> Vec<u16> {
    assert_eq!(x.len(), rows * k);
    assert_eq!(gate.len(), e * k);
    let g32: Vec<f32> = gate.iter().map(|&h| f16_to_f64(h) as f32).collect();
    let mut out = vec![0u16; rows * e];
    for r in 0..rows {
        let xr = &x[r * k..(r + 1) * k];
        for ex in 0..e {
            let w = &g32[ex * k..(ex + 1) * k];
            let mut lane = [0f32; 32];
            for (l, acc) in lane.iter_mut().enumerate() {
                let mut j = l;
                while j < k / 2 {
                    *acc = xr[2 * j].mul_add(w[2 * j], *acc);
                    *acc = xr[2 * j + 1].mul_add(w[2 * j + 1], *acc);
                    j += 32;
                }
            }
            for off in [16usize, 8, 4, 2, 1] {
                let prev = lane;
                for l in 0..32 {
                    // shfl_down: a lane whose source is out of range reads its own value
                    let src = if l + off < 32 { prev[l + off] } else { prev[l] };
                    lane[l] = prev[l] + src;
                }
            }
            out[r * e + ex] = f64_to_f16(lane[0] as f64);
        }
    }
    out
}

/// The sort key code of `router_select_kernel` (larger float, larger code; NaN smallest).
fn key_code(k: f32) -> u32 {
    if k.is_nan() {
        return 0;
    }
    let u = k.to_bits();
    if u & 0x8000_0000 != 0 {
        !u
    } else {
        u | 0x8000_0000
    }
}

/// Top-`topk` per row by `sig[logit bits] + bias[e]` with ties to the lower expert id, and the weights
/// `half(act * scale / (tree_sum + 1e-20))`, exactly as `router_select_kernel` computes them. Returns (picks, weight bits).
pub fn router_select_host(
    logits: &[u16],
    bias: &[f32],
    sig: &[f32],
    rows: usize,
    e: usize,
    topk: usize,
    scale: f32,
) -> (Vec<i32>, Vec<u16>) {
    assert!(topk <= e && topk <= 16 && logits.len() == rows * e && bias.len() == e && sig.len() == 65536);
    let mut picks = vec![0i32; rows * topk];
    let mut wts = vec![0u16; rows * topk];
    for r in 0..rows {
        let lg = &logits[r * e..(r + 1) * e];
        let key = |i: usize| sig[lg[i] as usize] + bias[i];
        let mut order: Vec<usize> = (0..e).collect();
        // larger code first, then the lower id
        order.sort_by(|&a, &b| key_code(key(b)).cmp(&key_code(key(a))).then(a.cmp(&b)));
        let acts: Vec<f32> = order[..topk].iter().map(|&i| sig[lg[i] as usize]).collect();
        // lanes >= topk hold 0; the shfl_down tree with the own-value fallback
        let mut lane = [0f32; 32];
        lane[..topk].copy_from_slice(&acts);
        for off in [16usize, 8, 4, 2, 1] {
            let prev = lane;
            for l in 0..32 {
                let src = if l + off < 32 { prev[l + off] } else { prev[l] };
                lane[l] = prev[l] + src;
            }
        }
        let s = lane[0] + 1e-20f32;
        for k in 0..topk {
            picks[r * topk + k] = order[k] as i32;
            wts[r * topk + k] = f64_to_f16(((acts[k] * scale) / s) as f64);
        }
    }
    (picks, wts)
}

// ----------------------------------------------------------------------------------------------------------------
// device objects

/// The 65,536-entry fp32 sigmoid table (indexed by fp16 bits) the select kernel reads. One per process.
pub struct SigmoidTable {
    buf: DevBuf,
}

impl SigmoidTable {
    pub fn new() -> Result<Self, CudaError> {
        Ok(Self { buf: DevBuf::from_slice(&sigmoid_table_host())? })
    }
}

/// The shapes of one MoE layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeDims {
    /// Model width (gate/up K, down N): 4096.
    pub hidden: usize,
    /// Expert width (gate/up N, down K): 2048.
    pub inter: usize,
    /// Routed experts: 288.
    pub experts: usize,
    /// Experts a token picks: 8.
    pub topk: usize,
    /// Routed scaling factor: 2.5.
    pub scale: f32,
    /// SwiGLU limit: 10.0 (`<= 0` disables the clamps).
    pub limit: f32,
}

impl MoeDims {
    pub const GLM53: MoeDims = MoeDims { hidden: 4096, inter: 2048, experts: 288, topk: 8, scale: 2.5, limit: 10.0 };
}

/// The router of one layer: fp16 gate [E, K] and the fp32 selection bias (already centred and fp16-rounded when matching the
/// oracle: [`centred_bias_host`]).
pub struct MoeRouter {
    gate: DevBuf,
    bias: DevBuf,
    pub hidden: usize,
    pub experts: usize,
    pub topk: usize,
    pub scale: f32,
    /// u32, 0 between launches: the one-launch decode route's count of finished blocks ([`MoeRouter::route`]).
    done: DevBuf,
}

impl MoeRouter {
    pub fn new(gate_f16: &[u16], bias_f32: &[f32], hidden: usize, topk: usize, scale: f32) -> Result<Self, CudaError> {
        let experts = bias_f32.len();
        if experts == 0
            || gate_f16.len() != experts * hidden
            || hidden % 2 != 0
            || topk == 0
            || topk > experts
            || topk > 16
            || experts > 512
        {
            return Err(INVALID);
        }
        Ok(Self {
            gate: DevBuf::from_slice(gate_f16)?,
            bias: DevBuf::from_slice(bias_f32)?,
            hidden,
            experts,
            topk,
            scale,
            done: DevBuf::zeroed(4)?,
        })
    }

    /// The router from device buffers the loader fills: `gate` fp16 `[experts, hidden]` as the pack stores it and `bias` fp32
    /// `[experts]` already centred and rounded to fp16 ([`centred_bias_host`]).
    pub fn from_device(gate: DevBuf, bias: DevBuf, hidden: usize, topk: usize, scale: f32) -> Result<Self, CudaError> {
        let experts = bias.bytes() / 4;
        if experts == 0
            || bias.bytes() != experts * 4
            || gate.bytes() != experts * hidden * 2
            || hidden % 2 != 0
            || topk == 0
            || topk > experts
            || topk > 16
            || experts > 512
        {
            return Err(INVALID);
        }
        Ok(Self { gate, bias, hidden, experts, topk, scale, done: DevBuf::zeroed(4)? })
    }

    /// logits [rows, E] fp16 into `logits`. The same arithmetic at every row count.
    pub fn logits(&self, x: &DevBuf, x_dtype: Dtype, rows: usize, logits: &mut DevBuf, st: &Stream) -> Result<(), CudaError> {
        if rows == 0 || x.bytes() < rows * self.hidden * x_dtype.size() || logits.bytes() < rows * self.experts * 2 {
            return Err(INVALID);
        }
        unsafe {
            check(router_logits_launch(
                x.ptr(),
                x_dtype as i32,
                self.gate.ptr(),
                logits.ptr(),
                rows,
                self.hidden,
                self.experts,
                st,
            ))
        }
    }

    /// [`MoeRouter::logits`] with the blocking spelled out (rows a warp 1/2/4, warps a block 2/4/8, loads in flight 1/4/8): the same bits
    /// for every shape (tested), only the speed differs.
    #[allow(clippy::too_many_arguments)]
    pub fn logits_blocked(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        logits: &mut DevBuf,
        rb: usize,
        warps: usize,
        un: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if rows == 0 || x.bytes() < rows * self.hidden * x_dtype.size() || logits.bytes() < rows * self.experts * 2 {
            return Err(INVALID);
        }
        unsafe {
            check(ffi::gb_moe_router_logits_ex(
                x.ptr(),
                x_dtype as i32,
                self.gate.ptr(),
                logits.ptr(),
                rows as i32,
                self.hidden as i32,
                self.experts as i32,
                rb as i32,
                warps as i32,
                un as i32,
                st.raw(),
            ))
        }
    }

    /// [`MoeRouter::logits_blocked`] with `eb` experts a warp as well (the prefill tiling, `router_logits_eb_kernel`: rows a warp 2/4/8,
    /// experts a warp 4/8, warps a block 4/8, loads in flight 1/2; eb 1 is `logits_blocked`'s blocking): the same bits for every shape
    /// (tested), only the speed differs.
    #[allow(clippy::too_many_arguments)]
    pub fn logits_blocked_eb(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        logits: &mut DevBuf,
        rb: usize,
        eb: usize,
        warps: usize,
        un: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if rows == 0 || x.bytes() < rows * self.hidden * x_dtype.size() || logits.bytes() < rows * self.experts * 2 {
            return Err(INVALID);
        }
        unsafe {
            check(ffi::gb_moe_router_logits_ex2(
                x.ptr(),
                x_dtype as i32,
                self.gate.ptr(),
                logits.ptr(),
                rows as i32,
                self.hidden as i32,
                self.experts as i32,
                rb as i32,
                eb as i32,
                warps as i32,
                un as i32,
                st.raw(),
            ))
        }
    }

    /// pick [rows, topk] i32 and wts [rows, topk] fp16 from fp16 logits.
    pub fn select(
        &self,
        tab: &SigmoidTable,
        logits: &DevBuf,
        rows: usize,
        pick: &mut DevBuf,
        wts: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if rows == 0
            || logits.bytes() < rows * self.experts * 2
            || pick.bytes() < rows * self.topk * 4
            || wts.bytes() < rows * self.topk * 2
        {
            return Err(INVALID);
        }
        unsafe {
            check(ffi::gb_moe_router_select(
                logits.ptr(),
                tab.buf.ptr(),
                self.bias.ptr(),
                pick.ptr(),
                wts.ptr(),
                self.scale,
                rows as i32,
                self.experts as i32,
                self.topk as i32,
                st.raw(),
            ))
        }
    }

    /// [`MoeRouter::logits`] then [`MoeRouter::select`]: the same logits, picks and weights. A decode or verify window (under 32 rows) runs
    /// them as one launch, the last block to finish selecting every row (unless a test hook sets the router's blocking).
    #[allow(clippy::too_many_arguments)]
    pub fn route(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        tab: &SigmoidTable,
        logits: &mut DevBuf,
        pick: &mut DevBuf,
        wts: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if rows >= 32 || ROUTER_SHAPE.load(std::sync::atomic::Ordering::Relaxed) != 0 {
            self.logits(x, x_dtype, rows, logits, st)?;
            return self.select(tab, logits, rows, pick, wts, st);
        }
        if rows == 0
            || x.bytes() < rows * self.hidden * x_dtype.size()
            || logits.bytes() < rows * self.experts * 2
            || pick.bytes() < rows * self.topk * 4
            || wts.bytes() < rows * self.topk * 2
        {
            return Err(INVALID);
        }
        unsafe {
            check(ffi::gb_moe_router_route(
                x.ptr(),
                x_dtype as i32,
                self.gate.ptr(),
                logits.ptr(),
                tab.buf.ptr(),
                self.bias.ptr(),
                pick.ptr(),
                wts.ptr(),
                self.scale,
                rows as i32,
                self.hidden as i32,
                self.experts as i32,
                self.topk as i32,
                self.done.ptr(),
                st.raw(),
            ))
        }
    }
}

/// `y [rows, n]` fp16 = `x [rows, k]` (any activation type) times the fp16 weight `w [n, k]` transposed: the router's GEMV chain
/// (two `fmaf` per half2 pair over pairs lane, lane + 32, ..., a `shfl_down` tree, one rounding to fp16), so it is row-invariant
/// like the router. For the small plain-fp16 projections of the DSA indexer (`wk`, `index_kpool_compress_gate`, `weights_proj`).
/// `k` even. Asynchronous on `st`.
#[allow(clippy::too_many_arguments)]
pub fn linear_f16(
    x: &DevBuf,
    x_dtype: Dtype,
    w: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    k: usize,
    n: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if rows == 0
        || n == 0
        || k % 2 != 0
        || x.bytes() < rows * k * x_dtype.size()
        || w.bytes() < n * k * 2
        || y.bytes() < rows * n * 2
    {
        return Err(INVALID);
    }
    unsafe { check(router_logits_launch(x.ptr(), x_dtype as i32, w.ptr(), y.ptr(), rows, k, n, st)) }
}

/// Test hook: the blocking `(rows a warp, warps a block, loads in flight)` every router-GEMV launch of this process takes, instead of the
/// default of the launcher (0 = the default). Every shape gives the same bits; only the speed differs.
static ROUTER_SHAPE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

#[doc(hidden)]
pub fn set_router_shape(shape: Option<(usize, usize, usize)>) {
    let v = shape.map_or(0, |(rb, warps, un)| ((rb as u32) << 16) | ((warps as u32) << 8) | un as u32);
    ROUTER_SHAPE.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// [`set_router_shape`] with experts a warp too: `(rows a warp, experts a warp, warps a block, loads in flight)`; eb 1 is the
/// `set_router_shape` blocking. Every shape gives the same bits; only the speed differs.
#[doc(hidden)]
pub fn set_router_shape_eb(shape: Option<(usize, usize, usize, usize)>) {
    let v = shape.map_or(0, |(rb, eb, warps, un)| ((eb as u32) << 24) | ((rb as u32) << 16) | ((warps as u32) << 8) | un as u32);
    ROUTER_SHAPE.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// One launch of the router's GEMV chain (the default blocking, or the hook's).
#[allow(clippy::too_many_arguments)]
unsafe fn router_logits_launch(
    x: *mut core::ffi::c_void,
    x_dtype: i32,
    w: *mut core::ffi::c_void,
    y: *mut core::ffi::c_void,
    rows: usize,
    k: usize,
    n: usize,
    st: &Stream,
) -> i32 {
    match ROUTER_SHAPE.load(std::sync::atomic::Ordering::Relaxed) {
        0 => ffi::gb_moe_router_logits(x, x_dtype, w, y, rows as i32, k as i32, n as i32, st.raw()),
        v if v >> 24 != 0 => ffi::gb_moe_router_logits_ex2(
            x,
            x_dtype,
            w,
            y,
            rows as i32,
            k as i32,
            n as i32,
            ((v >> 16) & 0xff) as i32,
            (v >> 24) as i32,
            ((v >> 8) & 0xff) as i32,
            (v & 0xff) as i32,
            st.raw(),
        ),
        v => ffi::gb_moe_router_logits_ex(
            x,
            x_dtype,
            w,
            y,
            rows as i32,
            k as i32,
            n as i32,
            (v >> 16) as i32,
            ((v >> 8) & 0xff) as i32,
            (v & 0xff) as i32,
            st.raw(),
        ),
    }
}

/// K splits and warps a block for a K x N layer run as `mats` matrices (2: gate and up, 1: down): the shape's alone.
pub fn default_plan(k: usize, n: usize, bits: usize, mats: usize) -> Result<(usize, usize), CudaError> {
    let mut out = [0i32; 2];
    unsafe { check(ffi::gb_moe_default_plan(k as i32, n as i32, bits as i32, mats as i32, out.as_mut_ptr()))? };
    Ok((out[0] as usize, out[1] as usize))
}

/// One band of [`MoeExperts`]: the experts that share a (bits_gu, bits_d, codebook) triple. Every buffer is LOCAL to the band:
/// membership lists, the local index of a global expert (or -1), the band's pointer tables and its stacked fp16 scales. When the
/// layer is uniform there is one band whose buffers are sub-views of the `MoeExperts` fields below (no second allocation).
pub(crate) struct MoeBand {
    pub bits_gu: usize,
    pub bits_d: usize,
    pub codebook: Codebook,
    /// (K splits, warps) of gate/up and of down, for this band's widths.
    pub plan_gu: (usize, usize),
    pub plan_d: (usize, usize),
    /// Members of the band (global expert ids, ascending); `local_of` indexes them.
    pub members: Vec<usize>,
    /// `[experts]`: the local index of each global expert in this band, `-1` outside it (the remap kernel's table).
    pub(crate) local_of: DevBuf,
    pub(crate) gate_ptrs: DevBuf,
    pub(crate) up_ptrs: DevBuf,
    pub(crate) down_ptrs: DevBuf,
    pub(crate) suh_g: DevBuf,
    pub(crate) suh_u: DevBuf,
    pub(crate) svh_g: DevBuf,
    pub(crate) svh_u: DevBuf,
    pub(crate) suh_d: DevBuf,
    pub(crate) svh_d: DevBuf,
}

/// The widths and codebook one band of a layer's experts shares (gate/up fused, as the kernels run them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BandKey {
    bits_gu: usize,
    bits_d: usize,
    codebook: Codebook,
}

/// Split `e` experts into bands of equal `BandKey`, in order of first occurrence, each band's members in ascending global id.
/// `key_of(g)` returns `None` for an expert the grouped path cannot represent (its band would be ill-formed): the caller then
/// refuses the layer (a clean fallback to the ungrouped path, never a wrong result).
fn band_partition(e: usize, key_of: impl Fn(usize) -> Option<BandKey>) -> Result<Vec<Vec<usize>>, CudaError> {
    let mut keys: Vec<BandKey> = Vec::new();
    let mut bands: Vec<Vec<usize>> = Vec::new();
    for g in 0..e {
        let k = key_of(g).ok_or(INVALID)?;
        let i = match keys.iter().position(|&x| x == k) {
            Some(i) => i,
            None => {
                keys.push(k);
                bands.push(Vec::new());
                keys.len() - 1
            }
        };
        bands[i].push(g);
    }
    Ok(bands)
}

/// One layer's routed experts on the device: per-expert trellises (each in its own allocation, addressed through pointer tables),
/// stacked fp16 scales, uniform width and codebook per projection.
pub struct MoeExperts {
    pub dims: MoeDims,
    pub bits_gu: usize,
    pub bits_d: usize,
    pub codebook: Codebook,
    /// (K splits, warps) of gate/up and of down; `default_plan` unless a test overrides them.
    pub plan_gu: (usize, usize),
    pub plan_d: (usize, usize),
    pub(crate) gate_ptrs: DevBuf,
    pub(crate) up_ptrs: DevBuf,
    pub(crate) down_ptrs: DevBuf,
    pub suh_g: DevBuf,
    pub suh_u: DevBuf,
    pub svh_g: DevBuf,
    pub svh_u: DevBuf,
    pub suh_d: DevBuf,
    pub svh_d: DevBuf,
    pub(crate) keep: Vec<DevBuf>,
    /// The device region the experts live in, when they were built by [`MoeExperts::from_region`] (then `keep` is empty and the
    /// scale buffers above are views into it).
    pub(crate) region: Option<DevBuf>,
    /// Bytes of one expert's three trellises plus its scales (what a pair reads).
    pub expert_bytes: usize,
    /// `None`: every expert of `dims.experts`, indexed by global id (the single-node object, today's). `Some`: a rank's bank of the
    /// owned experts only (pointer tables and scales hold `E_local` entries, ascending global id), run through
    /// [`MoeExperts::run_pairs`] / [`MoeExperts::run_prefill_pairs`] (`crate::moe_owned`).
    pub(crate) own: Option<crate::moe_owned::OwnedMap>,
    /// Width of up when it differs from gate's (`crate::moe_subbanks`: a split-width subbank, whose gate and up GEMVs run as two
    /// one-matrix launches). Equal to `bits_gu` for every bank built here.
    pub(crate) bits_u: usize,
    /// Run gate and up as two one-matrix launches even at equal widths (a test hook of `crate::moe_subbanks`).
    pub(crate) split_gu: bool,
    /// The width bands of this layer (always at least one; one when uniform). `run_prefill` runs each band with the band's own
    /// tables, stacks and bits; the `run` family (decode) is single-band only and refuses a mixed layer.
    pub(crate) bands: Vec<MoeBand>,
}

fn f16s(v: &[f64]) -> Vec<u16> {
    v.iter().map(|&x| f64_to_f16(x)).collect()
}

/// A sub-view of the whole buffer (same pointer and owner, no copy): the cheap way to hand the same device memory to two owners.
fn view_of(b: &DevBuf) -> DevBuf {
    b.subview(0, b.bytes())
}

/// Where one projection of a bank of experts sits inside a region, in bytes from the region's start: expert `e`'s int16 trellis is
/// at `trellis + e * trellis_stride`, and the stacked fp16 scales are `[experts, K]` (`suh`) and `[experts, N]` (`svh`) from
/// `suh` and `svh` (the layout `docs/GLM_LOAD.md` describes and the model crate's `ProjLayout` carries).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjOffsets {
    pub trellis: usize,
    pub trellis_stride: usize,
    pub suh: usize,
    pub svh: usize,
}

/// The layout of one MoE layer's experts inside one region (see [`MoeExperts::from_region`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertRegion {
    pub gate: ProjOffsets,
    pub up: ProjOffsets,
    pub down: ProjOffsets,
    /// Whole bits per weight of gate and up, and of down.
    pub bits_gu: usize,
    pub bits_d: usize,
    pub codebook: Codebook,
}

impl MoeExperts {
    /// From the loaded groups of every expert (index = expert id). Groups experts by equal (bits_gu, bits_d, codebook): a uniform
    /// layer is one band (the previous behaviour, buffers unchanged); a mixed layer keeps the same per-expert check and builds one
    /// band per width triple. An expert the grouped path cannot represent (split gate/up widths, a shape mismatch) refuses the layer,
    /// so a caller can fall back to the ungrouped path instead of failing the boot.
    pub fn from_groups(gate: &[LoadedGroup], up: &[LoadedGroup], down: &[LoadedGroup], dims: MoeDims) -> Result<Self, CudaError> {
        let e = dims.experts;
        if gate.len() != e || up.len() != e || down.len() != e || e == 0 {
            return Err(INVALID);
        }
        let (d, i) = (dims.hidden, dims.inter);
        let bits_of = |g: &LoadedGroup| -> Result<usize, CudaError> {
            if g.group.bits.twice() % 2 != 0 {
                return Err(INVALID);
            }
            Ok(g.group.bits.twice() as usize / 2)
        };
        let key_of = |g: usize| -> Option<BandKey> {
            let (gg, ug, dg) = (&gate[g], &up[g], &down[g]);
            if gg.group.k != d
                || gg.group.n != i
                || ug.group.k != d
                || ug.group.n != i
                || dg.group.k != i
                || dg.group.n != d
                || gg.group.codebook != ug.group.codebook
                || gg.group.codebook != dg.group.codebook
            {
                return None;
            }
            let bg = bits_of(gg).ok()?;
            if bits_of(ug).ok()? != bg {
                return None;
            }
            let bd = bits_of(dg).ok()?;
            Some(BandKey { bits_gu: bg, bits_d: bd, codebook: gg.group.codebook })
        };
        let members = band_partition(e, key_of)?;
        let mut keep = Vec::with_capacity(3 * e);
        let stack_of = |gs: &[usize], proj: usize, suh: bool| -> Result<DevBuf, CudaError> {
            let mut v = Vec::new();
            for &g in gs {
                let src = match proj {
                    0 => &gate[g],
                    1 => &up[g],
                    _ => &down[g],
                };
                v.extend(f16s(if suh { &src.suh } else { &src.svh }));
            }
            DevBuf::from_slice(&v)
        };
        if members.len() == 1 {
            // uniform: exactly the previous single-band behaviour
            let gs = &members[0];
            let (bg, bd) = (bits_of(&gate[gs[0]])?, bits_of(&down[gs[0]])?);
            let cb = gate[gs[0]].group.codebook;
            for &g in gs {
                if gate[g].group.k != d || gate[g].group.n != i || bits_of(&gate[g])? != bg || gate[g].group.codebook != cb {
                    return Err(INVALID);
                }
                if up[g].group.k != d || up[g].group.n != i || bits_of(&up[g])? != bg || up[g].group.codebook != cb {
                    return Err(INVALID);
                }
                if down[g].group.k != i || down[g].group.n != d || bits_of(&down[g])? != bd || down[g].group.codebook != cb {
                    return Err(INVALID);
                }
            }
            let mut gate_ptrs = Vec::with_capacity(e);
            let mut up_ptrs = Vec::with_capacity(e);
            let mut down_ptrs = Vec::with_capacity(e);
            for &g in gs {
                for (ptrs, grp) in [(&mut gate_ptrs, &gate[g]), (&mut up_ptrs, &up[g]), (&mut down_ptrs, &down[g])] {
                    let b = DevBuf::from_slice(&grp.trellis)?;
                    ptrs.push(b.ptr() as u64);
                    keep.push(b);
                }
            }
            let (gate_ptrs, up_ptrs, down_ptrs) =
                (DevBuf::from_slice(&gate_ptrs)?, DevBuf::from_slice(&up_ptrs)?, DevBuf::from_slice(&down_ptrs)?);
            let (suh_g, suh_u, svh_g, svh_u, suh_d, svh_d) = (
                stack_of(gs, 0, true)?,
                stack_of(gs, 1, true)?,
                stack_of(gs, 0, false)?,
                stack_of(gs, 1, false)?,
                stack_of(gs, 2, true)?,
                stack_of(gs, 2, false)?,
            );
            let expert_bytes =
                [&gate[gs[0]], &up[gs[0]], &down[gs[0]]].iter().map(|g| g.trellis.len() * 2 + (g.group.k + g.group.n) * 2).sum();
            let local_of = DevBuf::from_slice(&(0..e).map(|g| g as i32).collect::<Vec<i32>>())?;
            let band = MoeBand {
                bits_gu: bg,
                bits_d: bd,
                codebook: cb,
                plan_gu: default_plan(d, i, bg, 2)?,
                plan_d: default_plan(i, d, bd, 1)?,
                members: gs.clone(),
                local_of,
                gate_ptrs: view_of(&gate_ptrs),
                up_ptrs: view_of(&up_ptrs),
                down_ptrs: view_of(&down_ptrs),
                suh_g: view_of(&suh_g),
                suh_u: view_of(&suh_u),
                svh_g: view_of(&svh_g),
                svh_u: view_of(&svh_u),
                suh_d: view_of(&suh_d),
                svh_d: view_of(&svh_d),
            };
            return Ok(Self {
                dims,
                bits_gu: bg,
                bits_d: bd,
                codebook: cb,
                plan_gu: band.plan_gu,
                plan_d: band.plan_d,
                gate_ptrs,
                up_ptrs,
                down_ptrs,
                suh_g,
                suh_u,
                svh_g,
                svh_u,
                suh_d,
                svh_d,
                keep,
                region: None,
                expert_bytes,
                own: None,
                bits_u: bg,
                split_gu: false,
                bands: vec![band],
            });
        }
        // mixed widths: one band per (bits_gu, bits_d, codebook) triple, each band's buffers local to its members
        let mut bands = Vec::with_capacity(members.len());
        let mut expert_bytes = 0usize;
        for gs in &members {
            let eb = gs.len();
            let mut gate_ptrs = Vec::with_capacity(eb);
            let mut up_ptrs = Vec::with_capacity(eb);
            let mut down_ptrs = Vec::with_capacity(eb);
            for &g in gs {
                for (ptrs, grp) in [(&mut gate_ptrs, &gate[g]), (&mut up_ptrs, &up[g]), (&mut down_ptrs, &down[g])] {
                    let b = DevBuf::from_slice(&grp.trellis)?;
                    ptrs.push(b.ptr() as u64);
                    keep.push(b);
                }
                let e = &gate[g];
                expert_bytes += e.trellis.len() * 2 + (e.group.k + e.group.n) * 2;
                let e = &up[g];
                expert_bytes += e.trellis.len() * 2 + (e.group.k + e.group.n) * 2;
                let e = &down[g];
                expert_bytes += e.trellis.len() * 2 + (e.group.k + e.group.n) * 2;
            }
            let bg = bits_of(&gate[gs[0]])?;
            let bd = bits_of(&down[gs[0]])?;
            let cb = gate[gs[0]].group.codebook;
            let gate_ptrs = DevBuf::from_slice(&gate_ptrs)?;
            let up_ptrs = DevBuf::from_slice(&up_ptrs)?;
            let down_ptrs = DevBuf::from_slice(&down_ptrs)?;
            let suh_g = stack_of(gs, 0, true)?;
            let suh_u = stack_of(gs, 1, true)?;
            let svh_g = stack_of(gs, 0, false)?;
            let svh_u = stack_of(gs, 1, false)?;
            let suh_d = stack_of(gs, 2, true)?;
            let svh_d = stack_of(gs, 2, false)?;
            let mut local_of = vec![-1i32; e];
            for (l, &g) in gs.iter().enumerate() {
                local_of[g] = l as i32;
            }
            bands.push(MoeBand {
                bits_gu: bg,
                bits_d: bd,
                codebook: cb,
                plan_gu: default_plan(d, i, bg, 2)?,
                plan_d: default_plan(i, d, bd, 1)?,
                members: gs.clone(),
                local_of: DevBuf::from_slice(&local_of)?,
                gate_ptrs,
                up_ptrs,
                down_ptrs,
                suh_g,
                suh_u,
                svh_g,
                svh_u,
                suh_d,
                svh_d,
            });
        }
        // legacy fields mirror band 0; the decode paths refuse a mixed layer (`run` checks `bands.len()`)
        let b0 = &bands[0];
        Ok(Self {
            dims,
            bits_gu: b0.bits_gu,
            bits_d: b0.bits_d,
            codebook: b0.codebook,
            plan_gu: b0.plan_gu,
            plan_d: b0.plan_d,
            gate_ptrs: view_of(&b0.gate_ptrs),
            up_ptrs: view_of(&b0.up_ptrs),
            down_ptrs: view_of(&b0.down_ptrs),
            suh_g: view_of(&b0.suh_g),
            suh_u: view_of(&b0.suh_u),
            svh_g: view_of(&b0.svh_g),
            svh_u: view_of(&b0.svh_u),
            suh_d: view_of(&b0.suh_d),
            svh_d: view_of(&b0.svh_d),
            keep,
            region: None,
            expert_bytes,
            own: None,
            bits_u: b0.bits_gu,
            split_gu: false,
            bands,
        })
    }

    /// The experts of one layer from the per-expert `Exl3Weights` a loader keeps (the Qwen Flash-Next family: each
    /// expert's three projections are loaded as `Exl3Weights`, and the LoadedGroup/region the GLM path uses is gone).
    /// Construction only — no kernel or math change: the pointer tables point at the existing trellis buffers (which the
    /// caller keeps alive, e.g. in a weight Arc — `keep` stays empty), and the stacked fp16 scale buffers the grouped
    /// kernels read are uploaded once here (their bytes are the trellises' own, fp16-exact). Checks shapes, widths and
    /// codebooks are uniform, as [`MoeExperts::from_groups`] does.
    pub fn from_exl3(
        gate: &[&Exl3Weights],
        up: &[&Exl3Weights],
        down: &[&Exl3Weights],
        dims: MoeDims,
    ) -> Result<Self, CudaError> {
        let e = dims.experts;
        if gate.len() != e || up.len() != e || down.len() != e || e == 0 {
            return Err(INVALID);
        }
        let (d, i) = (dims.hidden, dims.inter);
        let ok_w = |w: &Exl3Weights, k: usize, n: usize| {
            w.k == k && w.n == n && w.bits >= 1 && w.bits <= 8 && k % 128 == 0 && n % 128 == 0
        };
        let key_of = |g: usize| -> Option<BandKey> {
            let (gg, ug, dg) = (gate[g], up[g], down[g]);
            if !ok_w(gg, d, i) || !ok_w(ug, d, i) || !ok_w(dg, i, d) || gg.codebook != ug.codebook || gg.codebook != dg.codebook {
                return None;
            }
            if gg.bits != ug.bits {
                return None;
            }
            Some(BandKey { bits_gu: gg.bits, bits_d: dg.bits, codebook: gg.codebook })
        };
        let members = band_partition(e, key_of)?;
        // the pointer tables: the device addresses of the existing trellis buffers (no byte is copied)
        let stack = |gs: &[usize], proj: usize, suh: bool| -> Result<DevBuf, CudaError> {
            let mut v: Vec<u16> = Vec::new();
            for &g in gs {
                let w = match proj {
                    0 => gate[g],
                    1 => up[g],
                    _ => down[g],
                };
                let n = if suh { w.k } else { w.n };
                let mut tmp = vec![0u16; n];
                (if suh { &w.suh } else { &w.svh }).download(&mut tmp)?;
                v.extend_from_slice(&tmp);
            }
            DevBuf::from_slice(&v)
        };
        if members.len() == 1 {
            // uniform: exactly the previous single-band behaviour
            let gs = &members[0];
            let (bg, bd) = (gate[gs[0]].bits, down[gs[0]].bits);
            let cb = gate[gs[0]].codebook;
            for &g in gs {
                if !ok_w(gate[g], d, i) || gate[g].bits != bg || gate[g].codebook != cb {
                    return Err(INVALID);
                }
                if !ok_w(up[g], d, i) || up[g].bits != bg || up[g].codebook != cb {
                    return Err(INVALID);
                }
                if !ok_w(down[g], i, d) || down[g].bits != bd || down[g].codebook != cb {
                    return Err(INVALID);
                }
            }
            let table = |gs: &[usize], proj: usize| -> Result<DevBuf, CudaError> {
                let ptrs: Vec<u64> = gs
                    .iter()
                    .map(|&g| match proj {
                        0 => gate[g].trellis.ptr() as u64,
                        1 => up[g].trellis.ptr() as u64,
                        _ => down[g].trellis.ptr() as u64,
                    })
                    .collect();
                DevBuf::from_slice(&ptrs)
            };
            let (gate_ptrs, up_ptrs, down_ptrs) = (table(gs, 0)?, table(gs, 1)?, table(gs, 2)?);
            let (suh_g, suh_u, svh_g, svh_u, suh_d, svh_d) = (
                stack(gs, 0, true)?,
                stack(gs, 1, true)?,
                stack(gs, 0, false)?,
                stack(gs, 1, false)?,
                stack(gs, 2, true)?,
                stack(gs, 2, false)?,
            );
            let expert_bytes = [gate[gs[0]], up[gs[0]], down[gs[0]]]
                .iter()
                .map(|w| w.trellis.bytes() + w.suh.bytes() + w.svh.bytes())
                .sum::<usize>();
            let local_of = DevBuf::from_slice(&(0..e).map(|g| g as i32).collect::<Vec<i32>>())?;
            let band = MoeBand {
                bits_gu: bg,
                bits_d: bd,
                codebook: cb,
                plan_gu: default_plan(d, i, bg, 2)?,
                plan_d: default_plan(i, d, bd, 1)?,
                members: gs.clone(),
                local_of,
                gate_ptrs: view_of(&gate_ptrs),
                up_ptrs: view_of(&up_ptrs),
                down_ptrs: view_of(&down_ptrs),
                suh_g: view_of(&suh_g),
                suh_u: view_of(&suh_u),
                svh_g: view_of(&svh_g),
                svh_u: view_of(&svh_u),
                suh_d: view_of(&suh_d),
                svh_d: view_of(&svh_d),
            };
            return Ok(Self {
                dims,
                bits_gu: bg,
                bits_d: bd,
                codebook: cb,
                plan_gu: band.plan_gu,
                plan_d: band.plan_d,
                gate_ptrs,
                up_ptrs,
                down_ptrs,
                suh_g,
                suh_u,
                svh_g,
                svh_u,
                suh_d,
                svh_d,
                keep: Vec::new(),
                region: None,
                expert_bytes,
                own: None,
                bits_u: bg,
                split_gu: false,
                bands: vec![band],
            });
        }
        // mixed widths: one band per (bits_gu, bits_d, codebook) triple, the pointers and scales local to the band
        let mut bands = Vec::with_capacity(members.len());
        let mut expert_bytes = 0usize;
        for gs in &members {
            let table = |proj: usize| -> Result<DevBuf, CudaError> {
                let ptrs: Vec<u64> = gs
                    .iter()
                    .map(|&g| match proj {
                        0 => gate[g].trellis.ptr() as u64,
                        1 => up[g].trellis.ptr() as u64,
                        _ => down[g].trellis.ptr() as u64,
                    })
                    .collect();
                DevBuf::from_slice(&ptrs)
            };
            let (gate_ptrs, up_ptrs, down_ptrs) = (table(0)?, table(1)?, table(2)?);
            let (suh_g, suh_u, svh_g, svh_u, suh_d, svh_d) = (
                stack(gs, 0, true)?,
                stack(gs, 1, true)?,
                stack(gs, 0, false)?,
                stack(gs, 1, false)?,
                stack(gs, 2, true)?,
                stack(gs, 2, false)?,
            );
            let bg = gate[gs[0]].bits;
            let bd = down[gs[0]].bits;
            let cb = gate[gs[0]].codebook;
            for &g in gs {
                expert_bytes += gate[g].trellis.bytes() + gate[g].suh.bytes() + gate[g].svh.bytes();
                expert_bytes += up[g].trellis.bytes() + up[g].suh.bytes() + up[g].svh.bytes();
                expert_bytes += down[g].trellis.bytes() + down[g].suh.bytes() + down[g].svh.bytes();
            }
            let mut local_of = vec![-1i32; e];
            for (l, &g) in gs.iter().enumerate() {
                local_of[g] = l as i32;
            }
            bands.push(MoeBand {
                bits_gu: bg,
                bits_d: bd,
                codebook: cb,
                plan_gu: default_plan(d, i, bg, 2)?,
                plan_d: default_plan(i, d, bd, 1)?,
                members: gs.clone(),
                local_of: DevBuf::from_slice(&local_of)?,
                gate_ptrs,
                up_ptrs,
                down_ptrs,
                suh_g,
                suh_u,
                svh_g,
                svh_u,
                suh_d,
                svh_d,
            });
        }
        // legacy fields mirror band 0; the decode paths refuse a mixed layer (`run` checks `bands.len()`)
        let b0 = &bands[0];
        Ok(Self {
            dims,
            bits_gu: b0.bits_gu,
            bits_d: b0.bits_d,
            codebook: b0.codebook,
            plan_gu: b0.plan_gu,
            plan_d: b0.plan_d,
            gate_ptrs: view_of(&b0.gate_ptrs),
            up_ptrs: view_of(&b0.up_ptrs),
            down_ptrs: view_of(&b0.down_ptrs),
            suh_g: view_of(&b0.suh_g),
            suh_u: view_of(&b0.suh_u),
            svh_g: view_of(&b0.svh_g),
            svh_u: view_of(&b0.svh_u),
            suh_d: view_of(&b0.suh_d),
            svh_d: view_of(&b0.svh_d),
            keep: Vec::new(),
            region: None,
            expert_bytes,
            own: None,
            bits_u: b0.bits_gu,
            split_gu: false,
            bands,
        })
    }

    /// The experts of one layer as they sit in ONE device region (`region` is a view of the whole region, already filled or about
    /// to be filled by the loader; `lay` says where each projection's trellises and stacked scales are). No byte is copied: the
    /// pointer tables the grouped kernel indexes are built from the region's base address and the strides, and the scale buffers
    /// are views. Equal, buffer for buffer, to [`MoeExperts::from_groups`] on the same weights (the model crate's GPU test checks
    /// that on a real layer).
    pub fn from_region(region: DevBuf, lay: &ExpertRegion, dims: MoeDims) -> Result<Self, CudaError> {
        let e = dims.experts;
        let (d, i) = (dims.hidden, dims.inter);
        if e == 0 || lay.bits_gu == 0 || lay.bits_d == 0 || d % 128 != 0 || i % 128 != 0 {
            return Err(INVALID);
        }
        // trellis [K/16, N/16, 16 * bits] int16: (K / 16) * (N / 16) * 16 * bits * 2 bytes
        let trellis_bytes = |k: usize, n: usize, bits: usize| (k / 16) * (n / 16) * 16 * bits * 2;
        let (tg, td) = (trellis_bytes(d, i, lay.bits_gu), trellis_bytes(i, d, lay.bits_d));
        // every block must lie inside the region
        let inside = |off: usize, bytes: usize| off.checked_add(bytes).is_some_and(|end| end <= region.bytes());
        let ok = [(&lay.gate, d, i, tg), (&lay.up, d, i, tg), (&lay.down, i, d, td)].iter().all(|(p, k, n, t)| {
            p.trellis_stride == *t
                && p.trellis % 16 == 0
                && inside(p.trellis, e * *t)
                && inside(p.suh, e * *k * 2)
                && inside(p.svh, e * *n * 2)
        });
        if !ok {
            return Err(INVALID);
        }
        let base = region.ptr() as u64;
        let table = |p: &ProjOffsets| -> Result<DevBuf, CudaError> {
            let ptrs: Vec<u64> = (0..e).map(|x| base + (p.trellis + x * p.trellis_stride) as u64).collect();
            DevBuf::from_slice(&ptrs)
        };
        let (gate_ptrs, up_ptrs, down_ptrs) = (table(&lay.gate)?, table(&lay.up)?, table(&lay.down)?);
        let expert_bytes = [(tg, d, i), (tg, d, i), (td, i, d)].iter().map(|&(t, k, n)| t + (k + n) * 2).sum();
        let (suh_g, suh_u, svh_g, svh_u, suh_d, svh_d) = (
            region.subview(lay.gate.suh, e * d * 2),
            region.subview(lay.up.suh, e * d * 2),
            region.subview(lay.gate.svh, e * i * 2),
            region.subview(lay.up.svh, e * i * 2),
            region.subview(lay.down.suh, e * i * 2),
            region.subview(lay.down.svh, e * d * 2),
        );
        let local_of = DevBuf::from_slice(&(0..e).map(|g| g as i32).collect::<Vec<i32>>())?;
        let band = MoeBand {
            bits_gu: lay.bits_gu,
            bits_d: lay.bits_d,
            codebook: lay.codebook,
            plan_gu: default_plan(d, i, lay.bits_gu, 2)?,
            plan_d: default_plan(i, d, lay.bits_d, 1)?,
            members: (0..e).collect(),
            local_of,
            gate_ptrs: view_of(&gate_ptrs),
            up_ptrs: view_of(&up_ptrs),
            down_ptrs: view_of(&down_ptrs),
            suh_g: view_of(&suh_g),
            suh_u: view_of(&suh_u),
            svh_g: view_of(&svh_g),
            svh_u: view_of(&svh_u),
            suh_d: view_of(&suh_d),
            svh_d: view_of(&svh_d),
        };
        Ok(Self {
            dims,
            bits_gu: lay.bits_gu,
            bits_d: lay.bits_d,
            codebook: lay.codebook,
            plan_gu: band.plan_gu,
            plan_d: band.plan_d,
            gate_ptrs,
            up_ptrs,
            down_ptrs,
            suh_g,
            suh_u,
            svh_g,
            svh_u,
            suh_d,
            svh_d,
            keep: Vec::new(),
            region: Some(region),
            expert_bytes,
            own: None,
            bits_u: lay.bits_gu,
            split_gu: false,
            bands: vec![band],
        })
    }

    /// Bytes of the device memory this layer's experts hold (including the per-band allocations of a mixed layer; band 0's
    /// buffers are views of the fields below and are not counted twice).
    pub fn device_bytes(&self) -> usize {
        let legacy = if let Some(r) = &self.region {
            r.bytes()
        } else {
            self.keep.iter().map(|b| b.bytes()).sum::<usize>()
                + [&self.suh_g, &self.suh_u, &self.svh_g, &self.svh_u, &self.suh_d, &self.svh_d]
                    .iter()
                    .map(|b| b.bytes())
                    .sum::<usize>()
        };
        let extra: usize = self
            .bands
            .iter()
            .skip(1)
            .map(|b| {
                [&b.gate_ptrs, &b.up_ptrs, &b.down_ptrs, &b.local_of, &b.suh_g, &b.suh_u, &b.svh_g, &b.svh_u, &b.suh_d, &b.svh_d]
                    .iter()
                    .map(|x| x.bytes())
                    .sum::<usize>()
            })
            .sum();
        legacy + extra
    }

    /// The width bands of this layer, for tests and diagnostics: `(bits_gu, bits_d, codebook, members)` per band, members
    /// ascending global expert ids. A uniform layer has one band with every expert.
    pub fn bands_desc(&self) -> Vec<(usize, usize, Codebook, Vec<usize>)> {
        self.bands.iter().map(|b| (b.bits_gu, b.bits_d, b.codebook, b.members.clone())).collect()
    }

    /// Run the routed experts on the picks in `s` (`s.pick`, `s.wts`): rotate, grouped gate/up GEMV, activation, grouped down GEMV,
    /// combine. `shared` (fp32 [rows, D]) is added last. Writes out [rows, D] fp32; fills `s.y_pairs` when `want_pairs`.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        s: &mut MoeScratch,
        want_pairs: bool,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.run_marked(x, x_dtype, rows, shared, out, s, want_pairs, (self.plan_gu, self.plan_d), st, &mut |_| Ok(()))
    }

    /// [`MoeExperts::run`] calling `mark(stage)` after each launch (profiling: record an event there).
    #[allow(clippy::too_many_arguments)]
    pub fn run_marked(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        s: &mut MoeScratch,
        want_pairs: bool,
        plans: ((usize, usize), (usize, usize)),
        st: &Stream,
        mark: &mut dyn FnMut(&'static str) -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        let d = self.dims;
        let p = rows * d.topk;
        if rows == 0 || rows > s.max_rows || x.bytes() < rows * d.hidden * x_dtype.size() || out.bytes() < rows * d.hidden * 4 {
            return Err(INVALID);
        }
        let ((sk_gu, wk_gu), (sk_d, wk_d)) = plans;
        if !self.holds_every_expert()
            || !self.fused_gate_up()
            || self.bands.len() > 1
            || !s.fits_plans(&d, plans)
            || shared.is_some_and(|b| b.bytes() < rows * d.hidden * 4)
        {
            return Err(INVALID);
        }
        let umax = p.min(d.experts);
        unsafe {
            check(ffi::gb_moe_group(
                s.pick.ptr(),
                s.uids.ptr(),
                s.ucount.ptr(),
                s.mcount.ptr(),
                s.members.ptr(),
                rows as i32,
                d.topk as i32,
                d.experts as i32,
                st.raw(),
            ))?;
            mark("group")?;
            check(ffi::gb_moe_rot_in(
                x.ptr(),
                x_dtype as i32,
                s.pick.ptr(),
                self.suh_g.ptr(),
                self.suh_u.ptr(),
                s.xg.ptr(),
                s.xu.ptr(),
                rows as i32,
                d.topk as i32,
                d.hidden as i32,
                d.experts as i32,
                st.raw(),
            ))?;
            mark("rot_in")?;
            check(ffi::gb_moe_gemv(
                s.xg.ptr(),
                s.xu.ptr(),
                self.gate_ptrs.ptr(),
                self.up_ptrs.ptr(),
                s.uids.ptr(),
                s.ucount.ptr(),
                s.mcount.ptr(),
                s.members.ptr(),
                s.z.ptr(),
                2,
                d.hidden as i32,
                d.inter as i32,
                p as i32,
                self.bits_gu as i32,
                codebook_code(self.codebook),
                sk_gu as i32,
                wk_gu as i32,
                umax as i32,
                st.raw(),
            ))?;
            mark("gemv_gate_up")?;
            check(ffi::gb_moe_gateup_epilogue(
                s.z.ptr(),
                s.pick.ptr(),
                self.svh_g.ptr(),
                self.svh_u.ptr(),
                self.suh_d.ptr(),
                s.xd.ptr(),
                rows as i32,
                d.topk as i32,
                d.inter as i32,
                sk_gu as i32,
                d.experts as i32,
                d.limit,
                st.raw(),
            ))?;
            mark("gate_up_epilogue")?;
            check(ffi::gb_moe_gemv(
                s.xd.ptr(),
                s.xd.ptr(),
                self.down_ptrs.ptr(),
                self.down_ptrs.ptr(),
                s.uids.ptr(),
                s.ucount.ptr(),
                s.mcount.ptr(),
                s.members.ptr(),
                s.z.ptr(),
                1,
                d.inter as i32,
                d.hidden as i32,
                p as i32,
                self.bits_d as i32,
                codebook_code(self.codebook),
                sk_d as i32,
                wk_d as i32,
                umax as i32,
                st.raw(),
            ))?;
            mark("gemv_down")?;
            check(ffi::gb_moe_down_combine(
                s.z.ptr(),
                s.pick.ptr(),
                self.svh_d.ptr(),
                s.wts.ptr(),
                shared.map_or(core::ptr::null(), |b| b.ptr() as *const _),
                if want_pairs { s.y_pairs.ptr() } else { core::ptr::null_mut() },
                out.ptr(),
                rows as i32,
                d.topk as i32,
                d.hidden as i32,
                sk_d as i32,
                d.experts as i32,
                st.raw(),
            ))?;
            mark("down_combine")
        }
    }
}

/// The gate/up epilogue on its own (unit tests and probes): `z` [2][sk][P][inter] fp32 partials, `pick` [P], stacked fp16 scales
/// `svh_g`/`svh_u` [E, inter] and `suh_d` [E, inter]; writes `xd` [P, inter] fp16.
#[allow(clippy::too_many_arguments)]
pub fn gateup_epilogue_raw(
    z: &DevBuf,
    pick: &DevBuf,
    svh_g: &DevBuf,
    svh_u: &DevBuf,
    suh_d: &DevBuf,
    xd: &mut DevBuf,
    rows: usize,
    topk: usize,
    inter: usize,
    sk: usize,
    experts: usize,
    limit: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    unsafe {
        check(ffi::gb_moe_gateup_epilogue(
            z.ptr(),
            pick.ptr(),
            svh_g.ptr(),
            svh_u.ptr(),
            suh_d.ptr(),
            xd.ptr(),
            rows as i32,
            topk as i32,
            inter as i32,
            sk as i32,
            experts as i32,
            limit,
            st.raw(),
        ))
    }
}

/// The down epilogue and slot combine on their own: `z` [sk][P][hidden] fp32 partials, `svh_d` [E, hidden] fp16, `wts` [P] fp16,
/// `shared` [rows, hidden] fp32 (added last) when given; writes `out` [rows, hidden] fp32 and `y_pairs` [P, hidden] when given.
#[allow(clippy::too_many_arguments)]
pub fn down_combine_raw(
    z: &DevBuf,
    pick: &DevBuf,
    svh_d: &DevBuf,
    wts: &DevBuf,
    shared: Option<&DevBuf>,
    y_pairs: Option<&mut DevBuf>,
    out: &mut DevBuf,
    rows: usize,
    topk: usize,
    hidden: usize,
    sk: usize,
    experts: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    unsafe {
        check(ffi::gb_moe_down_combine(
            z.ptr(),
            pick.ptr(),
            svh_d.ptr(),
            wts.ptr(),
            shared.map_or(core::ptr::null(), |b| b.ptr() as *const _),
            y_pairs.map_or(core::ptr::null_mut(), |b| b.ptr()),
            out.ptr(),
            rows as i32,
            topk as i32,
            hidden as i32,
            sk as i32,
            experts as i32,
            st.raw(),
        ))
    }
}

/// Stream-read `bytes` of `buf` once (a multiple of 16) and discard them: the bandwidth ceiling the expert GEMV is compared with.
/// `out` is a scratch of at least 4 bytes.
pub fn read_probe(buf: &DevBuf, bytes: usize, out: &mut DevBuf, st: &Stream) -> Result<(), CudaError> {
    if bytes > buf.bytes() || out.bytes() < 4 {
        return Err(INVALID);
    }
    unsafe { check(ffi::gb_moe_read_probe(buf.ptr(), bytes, out.ptr(), 48 * 16, st.raw())) }
}

/// CUDA events for timing (benchmarks and the per-stage profile). Not used on any serving path.
pub struct EventTimer {
    ev: Vec<*mut core::ffi::c_void>,
}

impl EventTimer {
    pub fn new(n: usize) -> Result<Self, CudaError> {
        let mut ev = Vec::with_capacity(n);
        for _ in 0..n {
            let mut e = core::ptr::null_mut();
            unsafe { check(ffi::cudaEventCreate(&mut e))? };
            ev.push(e);
        }
        Ok(Self { ev })
    }
    pub fn record(&self, i: usize, st: &Stream) -> Result<(), CudaError> {
        unsafe { check(ffi::cudaEventRecord(self.ev[i], st.raw())) }
    }
    /// Milliseconds from event `a` to event `b` (waits for `b`).
    pub fn ms(&self, a: usize, b: usize) -> Result<f32, CudaError> {
        let mut ms = 0f32;
        unsafe {
            check(ffi::cudaEventSynchronize(self.ev[b]))?;
            check(ffi::cudaEventElapsedTime(&mut ms, self.ev[a], self.ev[b]))?;
        }
        Ok(ms)
    }
}

impl Drop for EventTimer {
    fn drop(&mut self) {
        for &e in &self.ev {
            unsafe {
                ffi::cudaEventDestroy(e);
            }
        }
    }
}

/// The shared expert of a MoE layer: a dense gated MLP of three EXL3 linears (hidden -> inter twice, inter -> hidden) with the same
/// clamped SwiGLU as the routed experts. Runs on the existing row-invariant decode linears (1..=128 rows) and an elementwise kernel, so
/// it is row-invariant too. Its fp32 output is what [`MoeLayer::forward`] adds last.
pub struct SharedExpert {
    pub(crate) gate: Exl3Weights,
    pub(crate) up: Exl3Weights,
    pub(crate) down: Exl3Weights,
    pub hidden: usize,
    pub inter: usize,
    pub limit: f32,
}

impl SharedExpert {
    pub fn new(gate: Exl3Weights, up: Exl3Weights, down: Exl3Weights, limit: f32) -> Result<Self, CudaError> {
        let (hidden, inter) = (gate.k, gate.n);
        if up.k != hidden || up.n != inter || down.k != inter || down.n != hidden {
            return Err(INVALID);
        }
        Ok(Self { gate, up, down, hidden, inter, limit })
    }

    /// `model.language_model.layers.{layer}.mlp.shared_experts.{gate,up,down}_proj` of an EXL3 pack.
    pub fn from_checkpoint(ck: &Checkpoint, layer: usize, limit: f32) -> Result<Self, String> {
        let load = |p: &str| -> Result<Exl3Weights, String> {
            let g = ck
                .load_group(&format!("model.language_model.layers.{layer}.mlp.shared_experts.{p}"))
                .map_err(|e| format!("{p}: {e}"))?;
            Exl3Weights::from_loaded(&g).map_err(|e| format!("{p}: {e}"))
        };
        Self::new(load("gate_proj")?, load("up_proj")?, load("down_proj")?, limit).map_err(|e| e.to_string())
    }

    /// Bytes the three linears read per call (trellis and scales).
    pub fn weight_bytes(&self) -> usize {
        self.gate.weight_bytes() + self.up.weight_bytes() + self.down.weight_bytes()
    }

    /// out [rows, hidden] fp32 = down(swiglu_limit(gate(x), up(x))), x [rows, hidden] in `x_dtype`.
    pub fn forward(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        out: &mut DevBuf,
        s: &mut SharedScratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if rows == 0 || rows > s.max_rows || s.inter != self.inter || out.bytes() < rows * self.hidden * 4 {
            return Err(INVALID);
        }
        self.gate.forward(x, &mut s.g, rows, x_dtype, Dtype::F32, &mut s.ws, st)?;
        self.up.forward(x, &mut s.u, rows, x_dtype, Dtype::F32, &mut s.ws, st)?;
        unsafe {
            check(ffi::gb_moe_swiglu_limit(s.g.ptr(), s.u.ptr(), s.a.ptr(), (rows * self.inter) as i64, self.limit, st.raw()))?;
        }
        self.down.forward(&s.a, out, rows, Dtype::F16, Dtype::F32, &mut s.ws, st)
    }

    /// The down projection's plan `(sk, wk)` (lane E, GLM Phase 2 E2.6: the shared-expert split cuts at its K splits).
    pub fn down_plan(&self) -> Result<(usize, usize), CudaError> {
        self.down.plan()
    }

    /// The intermediate columns `[c0, c1)` that the down projection's K splits `[s0, s1)` read.
    pub fn down_split_k_range(&self, s0: usize, s1: usize) -> Result<(usize, usize), CudaError> {
        self.down.split_k_range(s0, s1)
    }

    /// Workspace of [`forward_split_partials`](Self::forward_split_partials) over the down splits `[s0, s1)` at `rows` rows.
    pub fn split_workspace_bytes(&self, s0: usize, s1: usize, rows: usize) -> Result<usize, CudaError> {
        let (c0, c1) = self.down_split_k_range(s0, s1)?;
        let strips = (c1 - c0) / 128;
        Ok(self
            .gate
            .strip_workspace_bytes(strips, rows)
            .max(self.up.strip_workspace_bytes(strips, rows))
            .max(self.down.split_partials_workspace_bytes(s0, s1, rows))
            .max(16))
    }

    /// Lane E, GLM Phase 2 E2.6: a tensor-parallel rank's half of the shared expert, up to the down projection's partials.
    ///
    /// - Columns: the intermediate columns `[c0, c1)` that the down projection's K splits `[s0, s1)` read. gate and up give those columns
    ///   as 128-column strips with their own (the parent's) plans, the clamped SwiGLU runs on them, and the down projection writes the
    ///   fp32 partials of the splits `[s0, s1)` into `z` from byte `z_off` (`[s1 - s0][rows][hidden]`, split `s0` first).
    /// - Exactness: every value is the one [`forward`](Self::forward) computes for those columns and splits (`exl3_split`: strips and
    ///   K-split partials with the parent's plans). The partials of every split gathered into `[sk][rows][hidden]` and finished by
    ///   [`finish_split`](Self::finish_split) are `forward`'s output bit for bit (`tests/shared_split.rs`).
    /// - Scratch: `s` holds the half (`[rows][c1 - c0]`); `ws` is at least [`split_workspace_bytes`](Self::split_workspace_bytes).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_split_partials(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        splits: (usize, usize),
        z: &mut DevBuf,
        z_off: usize,
        s: &mut SharedScratch,
        ws: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let (s0, s1) = splits;
        let (c0, c1) = self.down_split_k_range(s0, s1)?;
        if rows == 0 || rows > s.max_rows || s.inter != self.inter || c0 % 128 != 0 || c1 % 128 != 0 || c1 > self.inter {
            return Err(INVALID);
        }
        let n = c1 - c0;
        self.gate.forward_strip(c0 / 128, c1 / 128, x, &mut s.g, rows, x_dtype, Dtype::F32, ws, st)?;
        self.up.forward_strip(c0 / 128, c1 / 128, x, &mut s.u, rows, x_dtype, Dtype::F32, ws, st)?;
        unsafe {
            check(ffi::gb_moe_swiglu_limit(s.g.ptr(), s.u.ptr(), s.a.ptr(), (rows * n) as i64, self.limit, st.raw()))?;
        }
        self.down.forward_split_partials(s0, s1, &s.a, Dtype::F16, z, z_off, rows, ws, st)
    }

    /// The shared expert's output `out` `[rows, hidden]` fp32 from the down projection's partials of every split, `z` `[sk][rows][hidden]`
    /// (split order: the down projection's `rot_out`), as [`forward`](Self::forward) writes it.
    pub fn finish_split(&self, z: &DevBuf, rows: usize, out: &mut DevBuf, st: &Stream) -> Result<(), CudaError> {
        self.down.finish_partials(z, rows, out, Dtype::F32, st)
    }
}

/// `a [n]` fp16 = `silu(min(g, L)) * clamp(u, -L, L)` of `g`/`u` fp32 `[n]` (the Ling-3.1-flash-1019 clamp; `limit <= 0`
/// disables the clamp). The dense/shared-expert SwiGLU activation between its two EXL3 linears.
pub fn swiglu_clamped(g: &DevBuf, u: &DevBuf, a: &mut DevBuf, n: i64, limit: f32, st: &Stream) -> Result<(), CudaError> {
    if n < 0 || g.bytes() < n as usize * 4 || u.bytes() < n as usize * 4 || a.bytes() < n as usize * 2 {
        return Err(INVALID);
    }
    unsafe { check(ffi::gb_moe_swiglu_clamped(g.ptr(), u.ptr(), a.ptr(), n, limit, st.raw())) }
}

/// Scratch of [`SharedExpert::forward`]: contents on entry never matter.
pub struct SharedScratch {
    pub max_rows: usize,
    pub(crate) inter: usize,
    pub(crate) g: DevBuf,
    pub(crate) u: DevBuf,
    pub(crate) a: DevBuf,
    pub(crate) ws: DevBuf,
}

impl SharedScratch {
    /// Bytes of device memory [`SharedScratch::new`] holds for a gated MLP of `hidden` -> `inter` -> `hidden` and `max_rows` rows, computed
    /// without allocating (the EXL3 workspace does not depend on the width in bits).
    pub fn bytes_for(hidden: usize, inter: usize, max_rows: usize) -> usize {
        use crate::exl3_linear::linear_workspace_bytes;
        let ws = linear_workspace_bytes(max_rows, hidden, inter, 2).max(linear_workspace_bytes(max_rows, inter, hidden, 2));
        max_rows * inter * 4 * 2 + max_rows * inter * 2 + ws.max(16)
    }

    /// Bytes of device memory this scratch holds.
    pub fn bytes(&self) -> usize {
        self.g.bytes() + self.u.bytes() + self.a.bytes() + self.ws.bytes()
    }

    pub fn new(sh: &SharedExpert, max_rows: usize) -> Result<Self, CudaError> {
        if max_rows == 0 || max_rows > MAX_ROWS {
            return Err(INVALID);
        }
        let ws = [&sh.gate, &sh.up, &sh.down].iter().map(|w| w.workspace_bytes(max_rows)).max().unwrap_or(0);
        Ok(Self {
            max_rows,
            inter: sh.inter,
            g: DevBuf::zeroed(max_rows * sh.inter * 4)?,
            u: DevBuf::zeroed(max_rows * sh.inter * 4)?,
            a: DevBuf::zeroed(max_rows * sh.inter * 2)?,
            ws: DevBuf::zeroed(ws.max(16))?,
        })
    }
}

/// Router plus routed experts of one layer.
pub struct MoeLayer {
    pub router: MoeRouter,
    pub experts: MoeExperts,
    pub dims: MoeDims,
}

impl MoeLayer {
    pub fn new(router: MoeRouter, experts: MoeExperts) -> Result<Self, CudaError> {
        let d = experts.dims;
        if router.hidden != d.hidden || router.experts != d.experts || router.topk != d.topk || router.scale != d.scale {
            return Err(INVALID);
        }
        Ok(Self { dims: d, router, experts })
    }

    /// Load layer `layer` (`model.language_model.layers.N.mlp...`) of an EXL3 GLM pack: the fp16 gate, the fp32 correction bias
    /// (centred and rounded to fp16 as the oracle does) and every expert's three projections.
    pub fn from_checkpoint(ck: &Checkpoint, layer: usize, dims: MoeDims) -> Result<Self, String> {
        let base = format!("model.language_model.layers.{layer}.mlp");
        let read = |name: &str| -> Result<(Vec<u8>, String), String> {
            let key = format!("{base}.{name}");
            let e = ck.plain.get(&key).ok_or_else(|| format!("{key} is not in the checkpoint"))?;
            Ok((ck.read_tensor(e).map_err(|e| format!("{key}: {e}"))?, e.dtype.clone()))
        };
        let (gb, gt) = read("gate.weight")?;
        if gt != "F16" {
            return Err(format!("router gate is {gt}, expected F16"));
        }
        let gate: Vec<u16> = gb.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let (bb, bt) = read("gate.e_score_correction_bias")?;
        let esb: Vec<f32> = match bt.as_str() {
            "F32" => bb.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "BF16" => bb.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            "F16" => bb.chunks_exact(2).map(|c| f16_to_f64(u16::from_le_bytes([c[0], c[1]])) as f32).collect(),
            other => return Err(format!("e_score_correction_bias is {other}")),
        };
        if esb.len() != dims.experts || gate.len() != dims.experts * dims.hidden {
            return Err(format!("router shapes {} / {} do not match {dims:?}", gate.len(), esb.len()));
        }
        let router =
            MoeRouter::new(&gate, &centred_bias_host(&esb), dims.hidden, dims.topk, dims.scale).map_err(|e| e.to_string())?;
        let base_ref = base.as_str();
        let load = |proj: &str| -> Result<Vec<LoadedGroup>, String> {
            let n = dims.experts;
            let threads = 8usize.min(n);
            let mut out: Vec<Option<Result<LoadedGroup, String>>> = (0..n).map(|_| None).collect();
            std::thread::scope(|sc| {
                let mut hs = Vec::new();
                for (t, chunk) in out.chunks_mut(n.div_ceil(threads)).enumerate() {
                    let per = n.div_ceil(threads);
                    hs.push(sc.spawn(move || {
                        for (j, slot) in chunk.iter_mut().enumerate() {
                            let ex = t * per + j;
                            let prefix = format!("{base_ref}.experts.{ex}.{proj}");
                            *slot = Some(ck.load_group(&prefix).map_err(|e| format!("{prefix}: {e}")));
                        }
                    }));
                }
                for h in hs {
                    h.join().unwrap();
                }
            });
            out.into_iter().map(|o| o.unwrap()).collect()
        };
        let experts = MoeExperts::from_groups(&load("gate_proj")?, &load("up_proj")?, &load("down_proj")?, dims)
            .map_err(|e| e.to_string())?;
        Self::new(router, experts).map_err(|e| e.to_string())
    }

    /// Router: fills `s.logits`, `s.pick`, `s.wts`.
    pub fn route(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        tab: &SigmoidTable,
        s: &mut MoeScratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if rows == 0 || rows > s.max_rows || !s.fits(&self.dims, &self.experts) {
            return Err(INVALID);
        }
        self.router.route(x, x_dtype, rows, tab, &mut s.logits, &mut s.pick, &mut s.wts, st)
    }

    /// The routed experts on the picks already in `s` (from `route`, or written by the caller with `MoeScratch::set_picks`).
    #[allow(clippy::too_many_arguments)]
    pub fn experts_picked(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        s: &mut MoeScratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.experts.run(x, x_dtype, rows, shared, out, s, false, st)
    }

    /// [`MoeLayer::experts_picked`] with `before_combine` called between the down GEMV and the slot combine, the one place that reads the shared
    /// expert's output: a caller that runs the shared expert on a side stream waits for it there (stream wait), so the shared expert overlaps the
    /// router and the whole routed GEMV chain. The launches and their order on `st` are those of `experts_picked`.
    #[allow(clippy::too_many_arguments)]
    pub fn experts_picked_joined(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        s: &mut MoeScratch,
        st: &Stream,
        before_combine: &mut dyn FnMut() -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        let plans = (self.experts.plan_gu, self.experts.plan_d);
        self.experts.run_marked(x, x_dtype, rows, shared, out, s, false, plans, st, &mut |stage| {
            if stage == "gemv_down" {
                before_combine()
            } else {
                Ok(())
            }
        })
    }

    /// route + experts_picked.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        tab: &SigmoidTable,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        s: &mut MoeScratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.route(x, x_dtype, rows, tab, s, st)?;
        self.experts_picked(x, x_dtype, rows, shared, out, s, st)
    }

    /// [`MoeLayer::forward`] with a CUDA event between launches: returns (stage, milliseconds) for the router GEMV, the router
    /// select and each expert stage, and synchronizes the stream. For benchmarks.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_profiled(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        tab: &SigmoidTable,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        s: &mut MoeScratch,
        st: &Stream,
    ) -> Result<Vec<(&'static str, f32)>, CudaError> {
        let t = EventTimer::new(9)?;
        t.record(0, st)?;
        self.router.logits(x, x_dtype, rows, &mut s.logits, st)?;
        t.record(1, st)?;
        self.router.select(tab, &s.logits, rows, &mut s.pick, &mut s.wts, st)?;
        t.record(2, st)?;
        let mut n = 2;
        let mut names = vec!["router_logits", "router_select"];
        self.experts.run_marked(
            x,
            x_dtype,
            rows,
            shared,
            out,
            s,
            false,
            (self.experts.plan_gu, self.experts.plan_d),
            st,
            &mut |name| {
                n += 1;
                names.push(name);
                t.record(n, st)
            },
        )?;
        let mut v = Vec::new();
        for (i, name) in names.iter().enumerate() {
            v.push((*name, t.ms(i, i + 1)?));
        }
        Ok(v)
    }
}

/// Scratch for up to `max_rows` rows of one layer shape; contents on entry never matter, so one scratch serves every layer on a
/// stream. About 100 MiB at 128 rows.
pub struct MoeScratch {
    pub max_rows: usize,
    pub(crate) dims: MoeDims,
    pub logits: DevBuf,
    pub pick: DevBuf,
    pub wts: DevBuf,
    pub(crate) uids: DevBuf,
    pub(crate) ucount: DevBuf,
    pub(crate) mcount: DevBuf,
    pub(crate) members: DevBuf,
    pub(crate) xg: DevBuf,
    pub(crate) xu: DevBuf,
    pub(crate) xd: DevBuf,
    pub(crate) z: DevBuf,
    pub y_pairs: DevBuf,
}

impl MoeScratch {
    pub fn new(dims: &MoeDims, max_rows: usize) -> Result<Self, CudaError> {
        Self::with_splits(dims, max_rows, 1, 1)
    }

    /// Bytes of device memory [`MoeScratch::new`] holds for `max_rows` rows of `dims`, computed without allocating (the memory model of
    /// a family that runs the layer uses it; a GPU test checks it against [`MoeScratch::bytes`] of a real one).
    pub fn bytes_for(dims: &MoeDims, max_rows: usize) -> Result<usize, CudaError> {
        if max_rows == 0 || max_rows > MAX_ROWS {
            return Err(INVALID);
        }
        let p = max_rows * dims.topk;
        let umax = p.min(dims.experts);
        let (mut sk_gu, mut sk_d) = (1usize, 1usize);
        for bits in 1..=8 {
            sk_gu = sk_gu.max(default_plan(dims.hidden, dims.inter, bits, 2)?.0);
            sk_d = sk_d.max(default_plan(dims.inter, dims.hidden, bits, 1)?.0);
        }
        let z_floats = (2 * sk_gu * p * dims.inter).max(sk_d * p * dims.hidden);
        Ok(max_rows * dims.experts * 2   // logits
            + p * 4                      // pick
            + p * 2                      // wts
            + umax * 4                   // uids
            + 4                          // ucount
            + umax * 4                   // mcount
            + umax * p * 4               // members
            + 2 * p * dims.hidden * 2    // xg, xu
            + p * dims.inter * 2         // xd
            + z_floats * 4               // z
            + p * dims.hidden * 4) // y_pairs
    }

    /// Bytes of device memory this scratch holds.
    pub fn bytes(&self) -> usize {
        [
            &self.logits,
            &self.pick,
            &self.wts,
            &self.uids,
            &self.ucount,
            &self.mcount,
            &self.members,
            &self.xg,
            &self.xu,
            &self.xd,
            &self.z,
            &self.y_pairs,
        ]
        .iter()
        .map(|b| b.bytes())
        .sum()
    }

    /// [`MoeScratch::new`] with room for plans of up to `sk_gu` (gate/up) and `sk_d` (down) K splits (tuning sweeps).
    pub fn with_splits(dims: &MoeDims, max_rows: usize, sk_gu: usize, sk_d: usize) -> Result<Self, CudaError> {
        if max_rows == 0 || max_rows > MAX_ROWS {
            return Err(INVALID);
        }
        let p = max_rows * dims.topk;
        let umax = p.min(dims.experts);
        // the largest plan over every width, so any layer of this shape fits
        let (mut sk_gu, mut sk_d) = (sk_gu.max(1), sk_d.max(1));
        for bits in 1..=8 {
            sk_gu = sk_gu.max(default_plan(dims.hidden, dims.inter, bits, 2)?.0);
            sk_d = sk_d.max(default_plan(dims.inter, dims.hidden, bits, 1)?.0);
        }
        let z_floats = (2 * sk_gu * p * dims.inter).max(sk_d * p * dims.hidden);
        Ok(Self {
            max_rows,
            dims: *dims,
            logits: DevBuf::zeroed(max_rows * dims.experts * 2)?,
            pick: DevBuf::zeroed(p * 4)?,
            wts: DevBuf::zeroed(p * 2)?,
            uids: DevBuf::zeroed(umax * 4)?,
            ucount: DevBuf::zeroed(4)?,
            mcount: DevBuf::zeroed(umax * 4)?,
            members: DevBuf::zeroed(umax * p * 4)?,
            xg: DevBuf::zeroed(p * dims.hidden * 2)?,
            xu: DevBuf::zeroed(p * dims.hidden * 2)?,
            xd: DevBuf::zeroed(p * dims.inter * 2)?,
            z: DevBuf::zeroed(z_floats * 4)?,
            // a view of a block of its own (its bytes and contents as an allocation's): a GLM rank of the shared-expert split (lane E,
            // E2.6) cuts the shared expert's down partials out of the tail of a window's pair buffer, so one exchange carries both
            y_pairs: DevBuf::zeroed_view(p * dims.hidden * 4)?,
        })
    }

    /// Fill every buffer with `byte` (a test that the pipeline never reads what it did not write).
    pub fn poison(&mut self, byte: u8, st: &Stream) -> Result<(), CudaError> {
        for b in [
            &mut self.logits,
            &mut self.pick,
            &mut self.wts,
            &mut self.uids,
            &mut self.ucount,
            &mut self.mcount,
            &mut self.members,
            &mut self.xg,
            &mut self.xu,
            &mut self.xd,
            &mut self.z,
            &mut self.y_pairs,
        ] {
            b.memset_async(byte, st)?;
        }
        Ok(())
    }

    fn fits(&self, d: &MoeDims, ex: &MoeExperts) -> bool {
        self.fits_plans(d, (ex.plan_gu, ex.plan_d))
    }

    pub(crate) fn fits_plans(&self, d: &MoeDims, plans: ((usize, usize), (usize, usize))) -> bool {
        let (sk_gu, sk_d) = (plans.0 .0, plans.1 .0);
        self.dims.hidden == d.hidden
            && self.dims.inter == d.inter
            && self.dims.experts == d.experts
            && self.dims.topk == d.topk
            && self.z.bytes() >= (2 * sk_gu * self.max_rows * d.topk * d.inter).max(sk_d * self.max_rows * d.topk * d.hidden) * 4
    }

    /// Overwrite the picks and weights (tests, or a caller routing elsewhere). `pick` [rows * topk] expert ids (an id outside
    /// [0, E) is skipped), `wts` the fp16 weight bits.
    pub fn set_picks(&mut self, pick: &[i32], wts: &[u16]) -> Result<(), CudaError> {
        if pick.len() != wts.len() || pick.len() > self.max_rows * self.dims.topk {
            return Err(INVALID);
        }
        self.pick.upload(pick)?;
        self.wts.upload(wts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_table_is_exact_at_known_points() {
        let t = sigmoid_table_host();
        assert_eq!(t[0], 0.5); // +0
        assert_eq!(t[0x8000], 0.5); // -0
        assert_eq!(t[0x7C00], 1.0); // +inf
        assert_eq!(t[0xFC00], 0.0); // -inf
        assert!(t[0x7E00].is_nan());
        let one = f64_to_f16(1.0) as usize;
        assert_eq!(t[one], (1.0 / (1.0 + (-1.0f64).exp())) as f32);
    }

    #[test]
    fn host_select_breaks_ties_to_the_lower_id_and_ranks_by_key() {
        let sig = sigmoid_table_host();
        let e = 40;
        // every expert has the same logit and bias: a total tie, so the picks are 0..K in order
        let one = f64_to_f16(0.5);
        let logits = vec![one; e];
        let bias = vec![0.0f32; e];
        let (p, w) = router_select_host(&logits, &bias, &sig, 1, e, 8, 2.5);
        assert_eq!(p, (0..8).collect::<Vec<i32>>());
        // equal activations: 2.5 / 8 each
        assert!(w.iter().all(|&h| h == f64_to_f16(2.5 / 8.0)), "{w:?}");
        // expert 1 and expert 32 tie for the best key (bias lifts both): the lower id ranks first
        let mut bias = vec![0.0f32; e];
        bias[1] = 1.0;
        bias[32] = 1.0;
        let (p, _) = router_select_host(&logits, &bias, &sig, 1, e, 8, 2.5);
        assert_eq!(&p[..3], &[1, 32, 0]);
    }

    #[test]
    fn centred_bias_is_zero_mean_and_fp16_exact() {
        let esb: Vec<f32> = (0..288).map(|i| 34.0 + ((i * 7919) % 101) as f32 * 0.01).collect();
        let b = centred_bias_host(&esb);
        for &v in &b {
            assert_eq!(f16_to_f64(f64_to_f16(v as f64)) as f32, v);
        }
        let mean: f64 = b.iter().map(|&v| v as f64).sum::<f64>() / b.len() as f64;
        assert!(mean.abs() < 1e-3, "{mean}");
    }

    #[test]
    fn band_partition_groups_equal_widths_in_first_occurrence_order() {
        let cb = Codebook::Mul1;
        let key = |gu: usize, d: usize| BandKey { bits_gu: gu, bits_d: d, codebook: cb };
        let e = 12;
        let widths = [2usize, 2, 4, 4, 2, 8, 4, 2, 8, 4, 2, 8];
        // 0..2: 2, 2..4: 4, 4: 2, 5: 8, 6: 4, 7: 2, 8: 8, 9: 4, 10: 2, 11: 8
        let bands = band_partition(e, |g| Some(key(widths[g], 3))).unwrap();
        assert_eq!(bands.len(), 3);
        assert_eq!(bands[0], vec![0, 1, 4, 7, 10], "order of first occurrence, members ascending");
        assert_eq!(bands[1], vec![2, 3, 6, 9]);
        assert_eq!(bands[2], vec![5, 8, 11]);
        // a missing key (an unrepresentable expert) refuses the whole layer
        assert!(band_partition(e, |g| if g == 6 { None } else { Some(key(widths[g], 3)) }).is_err());
        // codebook differences separate bands even at equal bits
        let bands = band_partition(e, |g| {
            let k = key(widths[g], 3);
            Some(if g % 2 == 0 { k } else { BandKey { codebook: Codebook::Inst3, ..k } })
        })
        .unwrap();
        assert_eq!(bands.len(), 6);
        // all members accounted for exactly once
        let mut all: Vec<usize> = bands.iter().flatten().copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..e).collect::<Vec<usize>>());
    }

    #[test]
    fn band_partition_single_uniform_band_is_all_experts() {
        let cb = Codebook::Mul1;
        let bands = band_partition(32, |_| Some(BandKey { bits_gu: 4, bits_d: 3, codebook: cb })).unwrap();
        assert_eq!(bands.len(), 1);
        assert_eq!(bands[0], (0..32).collect::<Vec<usize>>());
    }
}
