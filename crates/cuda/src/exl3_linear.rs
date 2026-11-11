//! The EXL3 linear layer on the GPU: `y = ((x * suh) H_K . W_q . H_N) * svh (+ bias)`, W_q from the trellis through a
//! codebook (`csrc/exl3_linear.cu`, ported from TensorFold's `cuda/exl3`, MIT; see THIRD_PARTY_NOTICES.md).
//!
//! Two paths:
//! * [`Exl3Weights::forward`], the decode path, 1..=128 rows. ROW INVARIANT: a row's output bits are the same
//!   whether it is computed alone or inside any window of up to 128 rows at any position.
//! * [`Exl3Weights::forward_prefill_pinned`], any number of rows: W_q is reconstructed to fp16 and multiplied by a
//!   cublasLt GEMM on one pinned algorithm (no split-K, no workspace). ROW INVARIANT for every m >= 1: a row's output bits
//!   do not depend on m, on the row's index in the call or on the other rows. Its bits differ from the decode path's.
//!   The target model's prefill uses it for prompts above [`MAX_DECODE_ROWS`] tokens.
//! * [`Exl3Weights::forward_prefill`], the same steps on `cublasGemmEx`, whose kernel pick depends on m (below 17 rows
//!   it switches to split-K and CUDA-core kernels). Not row invariant. Kept for the draft and the vision encoder.
//!
//! The decode workspace needs no initialization and holds only scratch, so one workspace sized for the largest layer
//! serves all layers on a stream. There are no atomics and no counters anywhere. Pointers of `x` and `y` must be 16-byte aligned (any `DevBuf` is).

use crate::rt::{CudaError, DevBuf, Stream};
use gb_exl3::checkpoint::LoadedGroup;
use gb_exl3::f16::f64_to_f16;
use gb_exl3::Codebook;

/// Element type of an activation buffer. The discriminants are the C ABI's dtype codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Dtype {
    F16 = 0,
    BF16 = 1,
    F32 = 2,
}

impl Dtype {
    pub fn size(self) -> usize {
        match self {
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::F32 => 4,
        }
    }
}

/// Most rows the decode path takes in one call.
pub const MAX_DECODE_ROWS: usize = 128;

/// The geometry of one EXL3 group, without the group: `k`, `n`, the integer bit width and the codebook. What a caller that fills a group's
/// bytes itself (the expert page cache) keeps per projection instead of the pack's `Group`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exl3Dims {
    pub k: usize,
    pub n: usize,
    pub bits: usize,
    pub codebook: Codebook,
}

#[cfg(not(gb_no_cuda))]
fn codebook_code(cb: Codebook) -> i32 {
    match cb {
        Codebook::Inst3 => 0,
        Codebook::Mcg => 1,
        Codebook::Mul1 => 2,
    }
}

#[cfg(not(gb_no_cuda))]
mod ffi {
    use core::ffi::c_void;
    extern "C" {
        pub fn gb_exl3_linear_workspace_bytes(m: i32, k: i32, n: i32, bits: i32) -> usize;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_exl3_linear(
            x: *const c_void,
            x_dtype: i32,
            trellis: *const c_void,
            suh: *const c_void,
            svh: *const c_void,
            bias: *const c_void,
            y: *mut c_void,
            y_dtype: i32,
            m: i32,
            k: i32,
            n: i32,
            bits: i32,
            codebook: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        /// Half-integer width variant of `gb_exl3_linear`: `k2` is twice the bits (3 = 1.5,
        /// 5 = 2.5, 7 = 3.5), in place of the integer `bits` argument.
        #[allow(clippy::too_many_arguments)]
        pub fn gb_exl3_linear_k2(
            x: *const c_void,
            x_dtype: i32,
            trellis: *const c_void,
            suh: *const c_void,
            svh: *const c_void,
            bias: *const c_void,
            y: *mut c_void,
            y_dtype: i32,
            m: i32,
            k: i32,
            n: i32,
            k2: i32,
            codebook: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_exl3_reconstruct(
            trellis: *const c_void,
            w: *mut c_void,
            k: i32,
            n: i32,
            bits: i32,
            codebook: i32,
            stream: *mut c_void,
        ) -> i32;
        /// Half-integer width variant of `gb_exl3_reconstruct`: `k2` is twice the bits (3 = 1.5,
        /// 5 = 2.5, 7 = 3.5), which no `i32` bit count can name.
        pub fn gb_exl3_reconstruct_k2(
            trellis: *const c_void,
            w: *mut c_void,
            k: i32,
            n: i32,
            k2: i32,
            codebook: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_exl3_rotate_in(
            x: *const c_void,
            x_dtype: i32,
            suh: *const c_void,
            xh: *mut c_void,
            m: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_exl3_prefill_init() -> i32;
        pub fn gb_exl3_prefill_pin_name() -> *const core::ffi::c_char;
        pub fn gb_exl3_prefill_pin_for(k: i32, n: i32, tile: *mut u32, stages: *mut u32, custom: *mut u32) -> i32;
        pub fn gb_exl3_plan(k: i32, n: i32, sk: *mut i32, wk: *mut i32) -> i32;
        pub fn gb_pdl_set_mask(mask: i32) -> i32;
        pub fn gb_exl3_set_fuse(mask: i32) -> i32;
        pub fn gb_exl3_fused_in(m: i32, k: i32, n: i32, bits: i32) -> i32;
        pub fn gb_exl3_prefill_pin_check(m: i32, k: i32, n: i32) -> i32;
        pub fn gb_exl3_prefill_scratch_bytes(m: i32, k: i32, n: i32) -> usize;
        pub fn gb_exl3_prefill_gemm(
            xh: *const c_void,
            w: *const c_void,
            c: *mut c_void,
            m: i32,
            k: i32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_exl3_rotate_out(
            c: *const c_void,
            svh: *const c_void,
            bias: *const c_void,
            y: *mut c_void,
            y_dtype: i32,
            m: i32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_exl3_prefill_pinned(
            x: *const c_void,
            x_dtype: i32,
            trellis: *const c_void,
            suh: *const c_void,
            svh: *const c_void,
            bias: *const c_void,
            y: *mut c_void,
            y_dtype: i32,
            m: i32,
            k: i32,
            n: i32,
            bits: i32,
            codebook: i32,
            scratch: *mut c_void,
            scratch_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_exl3_prefill_pinned_with_w(
            x: *const c_void,
            x_dtype: i32,
            w: *const c_void,
            suh: *const c_void,
            svh: *const c_void,
            bias: *const c_void,
            y: *mut c_void,
            y_dtype: i32,
            m: i32,
            k: i32,
            n: i32,
            scratch: *mut c_void,
            scratch_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_exl3_prefill(
            x: *const c_void,
            x_dtype: i32,
            trellis: *const c_void,
            suh: *const c_void,
            svh: *const c_void,
            bias: *const c_void,
            y: *mut c_void,
            y_dtype: i32,
            m: i32,
            k: i32,
            n: i32,
            bits: i32,
            codebook: i32,
            scratch: *mut c_void,
            scratch_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// Parameter errors (a shape the kernels do not take, a buffer too small) are reported as cudaErrorInvalidValue.
const INVALID: CudaError = CudaError(1);

/// Bytes of decode workspace for `m` rows of a K x N layer (0 for a shape the decode path does not take).
pub fn linear_workspace_bytes(m: usize, k: usize, n: usize, bits: usize) -> usize {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        ffi::gb_exl3_linear_workspace_bytes(m as i32, k as i32, n as i32, bits as i32)
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (m, k, n, bits);
        0
    }
}

/// Create this host thread's cuBLAS handle now. The first prefill call on a thread does it otherwise, which allocates
/// device memory and so must not happen during CUDA graph capture; call this at start-up on any thread that prefills.
pub fn prefill_init() -> Result<(), CudaError> {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        CudaError::check(ffi::gb_exl3_prefill_init())
    }
    #[cfg(gb_no_cuda)]
    Err(CudaError(-1))
}

/// What the pinned prefill GEMM is, for the start-up log and the build identity.
pub fn prefill_pin_name() -> String {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        std::ffi::CStr::from_ptr(ffi::gb_exl3_prefill_pin_name()).to_string_lossy().into_owned()
    }
    #[cfg(gb_no_cuda)]
    "no CUDA".to_string()
}

/// Test hook: which kernel families the GLM decode path launches with the programmatic-dependent-launch attribute (`csrc/pdl.cuh`): bit 1
/// the EXL3 chain (`rot_in`, the linear kernel, `rot_out`), bit 2 mHC, bit 4 the routed MoE chain; 7 all, 0 none (the default; the
/// environment variable `REXL3_GLM_PDL` sets it at start). The bits do not depend on it; only the speed.
#[doc(hidden)]
pub fn set_pdl_mask(mask: i32) -> Result<(), CudaError> {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        CudaError::check(ffi::gb_pdl_set_mask(mask))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = mask;
        Err(CudaError(-1))
    }
}

/// Test hook: which launches the decode path folds into the linear kernel: bit 1 the input rotation of 1 to 8 rows (the kernel rotates
/// its own input instead of a `rot_in` launch ahead of it). The environment variable `REXL3_EXL3_FUSE` sets it at start. The bits do not
/// depend on it; only the launch count and the speed.
#[doc(hidden)]
pub fn set_fuse_mask(mask: i32) -> Result<(), CudaError> {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        CudaError::check(ffi::gb_exl3_set_fuse(mask))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = mask;
        Err(CudaError(-1))
    }
}

/// Whether [`Exl3Weights::forward`] of `m` rows of a `k` x `n` layer of `bits` bits rotates its own input under the current fuse mask (see
/// [`set_fuse_mask`]): the mask is on, `m` is 1 to 8 and the rotated rows fit beside the weight ring in shared memory.
#[doc(hidden)]
pub fn fused_in(m: usize, k: usize, n: usize, bits: usize) -> bool {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        ffi::gb_exl3_fused_in(m as i32, k as i32, n as i32, bits as i32) != 0
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (m, k, n, bits);
        false
    }
}

/// [`set_pdl_mask`] with every family (`true`) or none.
#[doc(hidden)]
pub fn set_pdl(on: bool) -> Result<(), CudaError> {
    set_pdl_mask(if on { 7 } else { 0 })
}

/// The decode path's plan of a `k` x `n` layer: `(K splits, warps a block)`. It is the shape's alone, so a layer keeps one reduction
/// order at every row count; narrow layers split K to fill the SMs (see `plan_for` in `csrc/exl3_linear.cu`).
pub fn decode_plan(k: usize, n: usize) -> Result<(usize, usize), CudaError> {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        let (mut sk, mut wk) = (0i32, 0i32);
        CudaError::check(ffi::gb_exl3_plan(k as i32, n as i32, &mut sk, &mut wk))?;
        Ok((sk as usize, wk as usize))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (k, n);
        Err(CudaError(-1))
    }
}

/// The pinned prefill GEMM's `(tile, stages, custom option)` for a `k` x `n` layer (cublasLt enum values): the default tile for every
/// shape but the Qwen3.8-27B down_proj's.
pub fn prefill_pin_for(k: usize, n: usize) -> Result<(u32, u32, u32), CudaError> {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        let (mut t, mut s, mut c) = (0u32, 0u32, 0u32);
        CudaError::check(ffi::gb_exl3_prefill_pin_for(k as i32, n as i32, &mut t, &mut s, &mut c))?;
        Ok((t, s, c))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (k, n);
        Err(CudaError(-1))
    }
}

/// Whether the pinned prefill GEMM runs an `m` x `k` x `n` problem on this device and cuBLAS. Call it at start-up for every
/// prefill shape: a cuBLAS that lacks the algorithm must stop the boot, not quietly change a prompt's bits.
pub fn prefill_pin_check(m: usize, k: usize, n: usize) -> Result<(), CudaError> {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        CudaError::check(ffi::gb_exl3_prefill_pin_check(m as i32, k as i32, n as i32))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (m, k, n);
        Err(CudaError(-1))
    }
}

/// Bytes of scratch for the prefill path with `m` rows of a K x N layer.
pub fn prefill_scratch_bytes(m: usize, k: usize, n: usize) -> usize {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        ffi::gb_exl3_prefill_scratch_bytes(m as i32, k as i32, n as i32)
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (m, k, n);
        0
    }
}

/// Reconstruct W_q for a half-integer width on the GPU: `k2` is twice the bits (3 = 1.5,
/// 5 = 2.5, 7 = 3.5, which no `i32` bit count can name). Same contract as
/// [`Exl3Weights::reconstruct`]: `w` holds the layer's `k * n` fp16 words row-major.
pub unsafe fn reconstruct_k2_raw(
    trellis: *const core::ffi::c_void,
    w: *mut core::ffi::c_void,
    k: i32,
    n: i32,
    k2: i32,
    codebook: i32,
    stream: *mut core::ffi::c_void,
) -> Result<(), CudaError> {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        CudaError::check(ffi::gb_exl3_reconstruct_k2(trellis, w, k, n, k2, codebook, stream))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (trellis, w, k, n, k2, codebook, stream);
        Err(CudaError(-1))
    }
}

/// The decode path ([`Exl3Weights::forward`]) for a half-integer width: `k2` is twice the bits
/// (3 = 1.5, 5 = 2.5, 7 = 3.5), in place of `Exl3Weights::bits`. The other arguments are
/// `forward`'s: `workspace` needs `linear_workspace_bytes`' sizing for the shape (see
/// `csrc/exl3_linear.cu` for the half-integer workspace contract).
#[allow(clippy::too_many_arguments)]
pub unsafe fn linear_k2_raw(
    x: *const core::ffi::c_void,
    x_dtype: i32,
    trellis: *const core::ffi::c_void,
    suh: *const core::ffi::c_void,
    svh: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    y_dtype: i32,
    m: i32,
    k: i32,
    n: i32,
    k2: i32,
    codebook: i32,
    workspace: *mut core::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut core::ffi::c_void,
) -> Result<(), CudaError> {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        CudaError::check(ffi::gb_exl3_linear_k2(
            x,
            x_dtype,
            trellis,
            suh,
            svh,
            bias,
            y,
            y_dtype,
            m,
            k,
            n,
            k2,
            codebook,
            workspace,
            workspace_bytes,
            stream,
        ))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (x, x_dtype, trellis, suh, svh, bias, y, y_dtype, m, k, n, k2, codebook, workspace, workspace_bytes, stream);
        Err(CudaError(-1))
    }
}

/// One EXL3 layer's tensors on the device. The trellis stays in its stored layout `[K/16, N/16, 16 * bits]` int16.
#[derive(Debug)]
pub struct Exl3Weights {
    pub(crate) trellis: DevBuf,
    /// fp16 [K]
    pub(crate) suh: DevBuf,
    /// fp16 [N]
    pub(crate) svh: DevBuf,
    /// fp16 [N]
    pub(crate) bias: Option<DevBuf>,
    pub k: usize,
    pub n: usize,
    pub bits: usize,
    pub codebook: Codebook,
}

fn f16_bits(v: &[f64]) -> Vec<u16> {
    v.iter().map(|&x| f64_to_f16(x)).collect()
}

/// One rectangular byte copy: `height` runs of `width` bytes, `src_pitch` bytes apart from `src_off` in the source and `dst_pitch` bytes
/// apart from `dst_off` in the destination (`cudaMemcpy2D`'s arguments). `height == 1` is a plain copy of `width` bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Copy2d {
    pub(crate) src_off: usize,
    pub(crate) src_pitch: usize,
    pub(crate) dst_off: usize,
    pub(crate) dst_pitch: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

/// The copies that gather the trellis of some 128-column strips of a K x N layer (`ranges`: ascending, disjoint, half-open, in
/// strips) into a trellis of the kept strips, one rectangle per range. Layout (see `exl3_linear.cu`): a tile is `8 * bits` 32-bit
/// words, a strip is 8 consecutive tiles, and a row of tiles (`kt`) holds `N / 16` tiles in column order, so one strip of one `kt`
/// row is a contiguous run of `64 * bits` words and the `K / 16` rows of a range are `N / 128` strips apart.
pub(crate) fn strip_trellis_plan(k: usize, n: usize, bits: usize, ranges: &[(usize, usize)]) -> Vec<Copy2d> {
    let strip = 64 * bits * 4; // bytes
    let src_pitch = (n / 128) * strip;
    let dst_pitch = ranges.iter().map(|&(lo, hi)| hi - lo).sum::<usize>() * strip;
    let mut dst_off = 0;
    ranges
        .iter()
        .map(|&(lo, hi)| {
            let c = Copy2d { src_off: lo * strip, src_pitch, dst_off, dst_pitch, width: (hi - lo) * strip, height: k / 16 };
            dst_off += c.width;
            c
        })
        .collect()
}

/// The copies that gather the entries of a per-column fp16 vector (`svh`, the bias) for the same strips: 128 columns of 2 bytes a strip.
pub(crate) fn strip_vector_plan(ranges: &[(usize, usize)]) -> Vec<Copy2d> {
    let mut dst_off = 0;
    ranges
        .iter()
        .map(|&(lo, hi)| {
            let width = (hi - lo) * 128 * 2;
            let c = Copy2d { src_off: lo * 128 * 2, src_pitch: width, dst_off, dst_pitch: width, width, height: 1 };
            dst_off += width;
            c
        })
        .collect()
}

/// Where [`Exl3Weights::select_strips_timed`] spent its wall time, in seconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StripTimes {
    /// `cudaMalloc` of the four new buffers.
    pub alloc: f64,
    /// Queueing the device-to-device copies and waiting for them.
    pub copy: f64,
    /// The whole call.
    pub total: f64,
}

impl Exl3Weights {
    /// Whole bits per weight of a group the kernels can run: integer widths and K, N multiples of 128, which is everything
    /// [`Self::from_loaded`] checks that does not depend on the data (the scale and trellis lengths follow from the shapes
    /// `gb_exl3::checkpoint::parse_group` has already validated).
    pub fn bits_of(g: &gb_exl3::checkpoint::Group) -> Result<usize, CudaError> {
        let bits2 = g.bits.twice() as usize;
        if bits2 % 2 != 0 || g.k % 128 != 0 || g.n % 128 != 0 {
            return Err(INVALID);
        }
        Ok(bits2 / 2)
    }

    /// Assemble a layer from device buffers the caller has filled (or will fill before the first launch) with exactly what
    /// [`Self::from_loaded`] would upload: the trellis as stored, `suh` and `svh` as fp16, and the optional fp16 bias.
    pub fn from_parts(
        g: &gb_exl3::checkpoint::Group,
        trellis: DevBuf,
        suh: DevBuf,
        svh: DevBuf,
        bias: Option<DevBuf>,
    ) -> Result<Self, CudaError> {
        let bits = Self::bits_of(g)?;
        let (k, n) = (g.k, g.n);
        if trellis.bytes() != (k / 16) * (n / 16) * 16 * bits * 2
            || suh.bytes() != k * 2
            || svh.bytes() != n * 2
            || bias.as_ref().is_some_and(|b| b.bytes() != n * 2)
            || bias.is_some() != g.bias
        {
            return Err(INVALID);
        }
        Ok(Self { trellis, suh, svh, bias, k, n, bits, codebook: g.codebook })
    }

    /// [`Self::from_parts`] from a group's *numbers* rather than the group: what an expert page cache needs, because a slot is filled from
    /// the pack's bytes and no `Group` travels with it. The shape checks are the same.
    pub fn from_dims(d: Exl3Dims, trellis: DevBuf, suh: DevBuf, svh: DevBuf, bias: Option<DevBuf>) -> Result<Self, CudaError> {
        let (k, n, bits) = (d.k, d.n, d.bits);
        if bits == 0 || trellis.bytes() != (k / 16) * (n / 16) * 16 * bits * 2 || suh.bytes() != k * 2 || svh.bytes() != n * 2 {
            return Err(INVALID);
        }
        if let Some(b) = &bias {
            if b.bytes() != n * 2 {
                return Err(INVALID);
            }
        }
        Ok(Self { trellis, suh, svh, bias, k, n, bits, codebook: d.codebook })
    }

    /// The buffers in a fixed order (trellis, suh, svh, bias), named, for comparing two loads of the same pack.
    pub fn buffers(&self) -> Vec<(&'static str, &DevBuf)> {
        let mut v = vec![("trellis", &self.trellis), ("suh", &self.suh), ("svh", &self.svh)];
        if let Some(b) = &self.bias {
            v.push(("bias", b));
        }
        v
    }

    /// Upload a loaded checkpoint group. Scales and bias become fp16 (they are fp16-exact in the checkpoint except
    /// a bf16 bias, which is rounded). Fractional widths (1.5, 2.5, 3.5 bits) are not supported.
    pub fn from_loaded(g: &LoadedGroup) -> Result<Self, CudaError> {
        let bits2 = g.group.bits.twice() as usize;
        let (k, n) = (g.group.k, g.group.n);
        if bits2 % 2 != 0 || k % 128 != 0 || n % 128 != 0 || g.suh.len() != k || g.svh.len() != n {
            return Err(INVALID);
        }
        let bits = bits2 / 2;
        if g.trellis.len() != (k / 16) * (n / 16) * 16 * bits {
            return Err(INVALID);
        }
        if g.bias.as_ref().is_some_and(|b| b.len() != n) {
            return Err(INVALID);
        }
        let (suh, svh, bias) = {
            let _t = gb_exl3::loadstats::timer(gb_exl3::loadstats::Stage::Convert, (g.suh.len() + g.svh.len()) as u64 * 8);
            (f16_bits(&g.suh), f16_bits(&g.svh), g.bias.as_deref().map(f16_bits))
        };
        Ok(Self {
            trellis: DevBuf::from_slice(&g.trellis)?,
            suh: DevBuf::from_slice(&suh)?,
            svh: DevBuf::from_slice(&svh)?,
            bias: match &bias {
                Some(b) => Some(DevBuf::from_slice(b)?),
                None => None,
            },
            k,
            n,
            bits,
            codebook: g.group.codebook,
        })
    }

    /// A layer made of only some of this one's output columns: `ranges` are half-open ranges of 128-column strips (the
    /// unit the output Hadamard works in), ascending and disjoint, so the new layer's columns are the old layer's columns of
    /// those strips in the same order. Its decode kernel runs with the same K split and warp count as the full layer
    /// (both come from `(K, N)` and any `N >= 128 * 64` gives the same plan) and every strip is computed from its own tiles
    /// alone, so each kept column is bit-identical to the full layer's, at any row count.
    ///
    /// This is the draft's vocabulary head: a candidate set of tokens is a handful of strips of the target's `lm_head`.
    /// The strips are gathered on the device (one `cudaMemcpy2D` per range for the trellis, one copy per range for `svh` and the bias,
    /// one copy of `suh`), so the new layer owns its memory and no host memory is touched. The bytes are exactly the ones a copy through
    /// the host makes (the unit tests keep that path as the reference). Going through the host was measured to take 0.3 s, or 8 s when
    /// the driver let the GPU fault in the 0.95 GB host destination itself (`docs/EVAL_RESULTS.md`, "Boot").
    pub fn select_strips(&self, ranges: &[(usize, usize)]) -> Result<Exl3Weights, CudaError> {
        Ok(self.select_strips_timed(ranges)?.0)
    }

    /// [`Self::select_strips`] with the wall time of its two steps (three clock reads).
    pub fn select_strips_timed(&self, ranges: &[(usize, usize)]) -> Result<(Exl3Weights, StripTimes), CudaError> {
        let t0 = std::time::Instant::now();
        let strips = self.n / 128;
        let mut at = 0usize;
        for &(lo, hi) in ranges {
            if lo < at || hi <= lo || hi > strips {
                return Err(INVALID);
            }
            at = hi;
        }
        let kept: usize = ranges.iter().map(|&(lo, hi)| hi - lo).sum();
        if kept == 0 || kept < 64 {
            return Err(INVALID); // fewer than 64 strips would change the kernel's K split (see `plan_for` in exl3_linear.cu)
        }
        let mut times = StripTimes::default();
        let st = Stream::new()?;
        let mut trellis = DevBuf::alloc((self.k / 16) * kept * 64 * self.bits * 4)?;
        let mut suh = DevBuf::alloc(self.suh.bytes())?;
        let mut svh = DevBuf::alloc(kept * 128 * 2)?;
        let mut bias = match &self.bias {
            Some(_) => Some(DevBuf::alloc(kept * 128 * 2)?),
            None => None,
        };
        let t1 = std::time::Instant::now();
        times.alloc = t1.duration_since(t0).as_secs_f64();
        for c in strip_trellis_plan(self.k, self.n, self.bits, ranges) {
            trellis.copy_2d_from(c.dst_off, c.dst_pitch, &self.trellis, c.src_off, c.src_pitch, c.width, c.height, &st)?;
        }
        suh.copy_from(&self.suh, self.suh.bytes(), &st)?;
        for c in strip_vector_plan(ranges) {
            svh.copy_range_from(c.dst_off, &self.svh, c.src_off, c.width, &st)?;
            if let (Some(dst), Some(src)) = (bias.as_mut(), self.bias.as_ref()) {
                dst.copy_range_from(c.dst_off, src, c.src_off, c.width, &st)?;
            }
        }
        st.sync()?;
        times.copy = t1.elapsed().as_secs_f64();
        let w = Exl3Weights { trellis, suh, svh, bias, k: self.k, n: kept * 128, bits: self.bits, codebook: self.codebook };
        times.total = t0.elapsed().as_secs_f64();
        Ok((w, times))
    }

    /// Bytes the kernels read at M <= 16: trellis plus scales (and bias).
    pub fn weight_bytes(&self) -> usize {
        self.trellis.bytes() + self.suh.bytes() + self.svh.bytes() + self.bias.as_ref().map_or(0, |b| b.bytes())
    }

    /// Decode workspace for `m` rows (see the module docs for its zero-start contract).
    pub fn workspace_bytes(&self, m: usize) -> usize {
        linear_workspace_bytes(m, self.k, self.n, self.bits)
    }

    /// Prefill scratch for `m` rows.
    pub fn prefill_scratch_bytes(&self, m: usize) -> usize {
        prefill_scratch_bytes(m, self.k, self.n)
    }

    #[cfg(not(gb_no_cuda))]
    fn bias_ptr(&self) -> *const core::ffi::c_void {
        self.bias.as_ref().map_or(core::ptr::null(), |b| b.ptr() as *const _)
    }

    fn check_io(&self, x: &DevBuf, y: &DevBuf, m: usize, x_dtype: Dtype, y_dtype: Dtype) -> Result<(), CudaError> {
        if m == 0 || x.bytes() < m * self.k * x_dtype.size() || y.bytes() < m * self.n * y_dtype.size() {
            return Err(INVALID);
        }
        Ok(())
    }

    /// Decode path: y [m, n] = layer(x [m, k]) for 1 <= m <= 128, row invariant. Asynchronous on `st`.
    /// `ws` needs `workspace_bytes(m)` bytes; its contents on entry do not matter.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &DevBuf,
        y: &mut DevBuf,
        m: usize,
        x_dtype: Dtype,
        y_dtype: Dtype,
        ws: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.check_io(x, y, m, x_dtype, y_dtype)?;
        if m > MAX_DECODE_ROWS || ws.bytes() < self.workspace_bytes(m) {
            return Err(INVALID);
        }
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_exl3_linear(
                x.ptr(),
                x_dtype as i32,
                self.trellis.ptr(),
                self.suh.ptr(),
                self.svh.ptr(),
                self.bias_ptr(),
                y.ptr(),
                y_dtype as i32,
                m as i32,
                self.k as i32,
                self.n as i32,
                self.bits as i32,
                codebook_code(self.codebook),
                ws.ptr(),
                ws.bytes(),
                st.raw(),
            ))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (ws, st);
            Err(CudaError(-1))
        }
    }

    /// Decode path looped in chunks of [`MAX_DECODE_ROWS`] (128) rows using pointer offsets (`ptr_at`), so any `m >= 1`
    /// runs row-invariantly without requiring `x` or `y` to be block-backed views.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_chunked(
        &self,
        x: &DevBuf,
        y: &mut DevBuf,
        m: usize,
        x_dtype: Dtype,
        y_dtype: Dtype,
        ws: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.check_io(x, y, m, x_dtype, y_dtype)?;
        let max_c = m.min(MAX_DECODE_ROWS);
        if ws.bytes() < self.workspace_bytes(max_c) {
            return Err(INVALID);
        }
        #[cfg(not(gb_no_cuda))]
        unsafe {
            let mut r0 = 0usize;
            while r0 < m {
                let chunk = (m - r0).min(MAX_DECODE_ROWS);
                let x_ptr = x.ptr_at(r0 * self.k * x_dtype.size()) as *const core::ffi::c_void;
                let y_ptr = y.ptr_at(r0 * self.n * y_dtype.size());
                CudaError::check(ffi::gb_exl3_linear(
                    x_ptr,
                    x_dtype as i32,
                    self.trellis.ptr(),
                    self.suh.ptr(),
                    self.svh.ptr(),
                    self.bias_ptr(),
                    y_ptr,
                    y_dtype as i32,
                    chunk as i32,
                    self.k as i32,
                    self.n as i32,
                    self.bits as i32,
                    codebook_code(self.codebook),
                    ws.ptr(),
                    ws.bytes(),
                    st.raw(),
                ))?;
                r0 += chunk;
            }
            Ok(())
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (ws, st);
            Err(CudaError(-1))
        }
    }
    /// Decode path with half-integer widths: the same as [`Exl3Weights::forward`] but `k2 = 2 * bits` in 2..16
    /// (`gb_exl3_linear_k2`; odd k2 3/5/7 = 1.5/2.5/3.5 bits with the mul1 codebook). Row-invariant, asynchronous on `st`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_k2(
        &self,
        x: &DevBuf,
        y: &mut DevBuf,
        m: usize,
        x_dtype: Dtype,
        y_dtype: Dtype,
        k2: usize,
        ws: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        Self::k2_forward(
            x,
            y,
            m,
            x_dtype,
            y_dtype,
            self.k,
            self.n,
            k2,
            self.codebook,
            &self.trellis,
            &self.suh,
            &self.svh,
            self.bias.as_ref(),
            ws,
            st,
        )
    }

    /// The raw `gb_exl3_linear_k2` launch over caller-owned buffers: `x` `[m, k]`, `y` `[m, n]`, `k2 = 2 * bits`
    /// (2..16), `codebook`, `trellis`/`suh`/`svh` (optional `bias`) as stored, `ws` at least [`linear_workspace_bytes`].
    #[allow(clippy::too_many_arguments)]
    pub fn k2_forward(
        x: &DevBuf,
        y: &mut DevBuf,
        m: usize,
        x_dtype: Dtype,
        y_dtype: Dtype,
        k: usize,
        n: usize,
        k2: usize,
        codebook: Codebook,
        trellis: &DevBuf,
        suh: &DevBuf,
        svh: &DevBuf,
        bias: Option<&DevBuf>,
        ws: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if m < 1 || m > MAX_DECODE_ROWS || k2 < 2 || k2 > 16 || k == 0 || n == 0 {
            return Err(INVALID);
        }
        if x.bytes() < m * k * x_dtype.size() || y.bytes() < m * n * y_dtype.size() {
            return Err(INVALID);
        }
        if ws.bytes() < linear_workspace_bytes(m, k, n, 1) {
            return Err(INVALID);
        }
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_exl3_linear_k2(
                x.ptr(),
                x_dtype as i32,
                trellis.ptr(),
                suh.ptr(),
                svh.ptr(),
                bias.map_or(core::ptr::null(), |b| b.ptr()),
                y.ptr(),
                y_dtype as i32,
                m as i32,
                k as i32,
                n as i32,
                k2 as i32,
                codebook_code(codebook),
                ws.ptr(),
                ws.bytes(),
                st.raw(),
            ))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (ws, st);
            Err(CudaError(-1))
        }
    }

    /// W_q [k, n] fp16 row-major into `w` (k * n * 2 bytes).
    pub fn reconstruct(&self, w: &mut DevBuf, st: &Stream) -> Result<(), CudaError> {
        if w.bytes() < self.k * self.n * 2 {
            return Err(INVALID);
        }
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_exl3_reconstruct(
                self.trellis.ptr(),
                w.ptr(),
                self.k as i32,
                self.n as i32,
                self.bits as i32,
                codebook_code(self.codebook),
                st.raw(),
            ))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = st;
            Err(CudaError(-1))
        }
    }

    /// Prefill path for any m >= 1: reconstruct W_q, rotate x, `cublasGemmEx` (fp16 inputs, fp32 accumulate), rotate the
    /// result. `scratch` needs `prefill_scratch_bytes(m)`. Not row invariant (the kernel cuBLAS picks depends on m); the
    /// target model uses [`Exl3Weights::forward_prefill_pinned`]. x and y share `dtype`.
    pub fn forward_prefill(
        &self,
        x: &DevBuf,
        y: &mut DevBuf,
        m: usize,
        dtype: Dtype,
        scratch: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.forward_prefill_io(x, y, m, dtype, dtype, scratch, st)
    }

    /// [`Exl3Weights::forward_prefill`] with separate input and output dtypes (e.g. fp16 in, fp32 out).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill_io(
        &self,
        x: &DevBuf,
        y: &mut DevBuf,
        m: usize,
        x_dtype: Dtype,
        y_dtype: Dtype,
        scratch: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.check_io(x, y, m, x_dtype, y_dtype)?;
        if scratch.bytes() < self.prefill_scratch_bytes(m) {
            return Err(INVALID);
        }
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_exl3_prefill(
                x.ptr(),
                x_dtype as i32,
                self.trellis.ptr(),
                self.suh.ptr(),
                self.svh.ptr(),
                self.bias_ptr(),
                y.ptr(),
                y_dtype as i32,
                m as i32,
                self.k as i32,
                self.n as i32,
                self.bits as i32,
                codebook_code(self.codebook),
                scratch.ptr(),
                scratch.bytes(),
                st.raw(),
            ))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (scratch, st);
            Err(CudaError(-1))
        }
    }

    /// Prefill path for any m >= 1 on the pinned cublasLt algorithm: reconstruct W_q, rotate x, GEMM (fp16 inputs, fp32
    /// accumulate), rotate the result. Row invariant (see the module docs): a row's bits are the same in a call of 1 row
    /// or of 4,096, at any index. `scratch` needs `prefill_scratch_bytes(m)`. x and y share `dtype`.
    pub fn forward_prefill_pinned(
        &self,
        x: &DevBuf,
        y: &mut DevBuf,
        m: usize,
        dtype: Dtype,
        scratch: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.check_io(x, y, m, dtype, dtype)?;
        if scratch.bytes() < self.prefill_scratch_bytes(m) {
            return Err(INVALID);
        }
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_exl3_prefill_pinned(
                x.ptr(),
                dtype as i32,
                self.trellis.ptr(),
                self.suh.ptr(),
                self.svh.ptr(),
                self.bias_ptr(),
                y.ptr(),
                dtype as i32,
                m as i32,
                self.k as i32,
                self.n as i32,
                self.bits as i32,
                codebook_code(self.codebook),
                scratch.ptr(),
                scratch.bytes(),
                st.raw(),
            ))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (scratch, st);
            Err(CudaError(-1))
        }
    }

    /// [`Exl3Weights::forward_prefill_pinned`] over caller-owned decomposed parts (the bailing `Lin` image): `k2 = 2 * bits`
    /// covers every EXL3 width in the pack (`k2` 2..16: 1.0 through 8.0 bits, including the half-integer 1.5/2.5/3.5),
    /// no bias. The same pinned cublasLt GEMM, the same rotations, row-invariant, any `x_dtype`/`y_dtype` pair the
    /// rotate kernels support (BF16/F16/F32 in and out). The trellis is decoded with `fmoe::unpack` into the scratch's
    /// `W_q` section (a pure function of the trellis, so the GEMM sees exactly the bits `gb_exl3_reconstruct` would have
    /// produced — and unlike it, every `k2` including the half-integer ones).
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_pinned_parts(
        x: &DevBuf,
        y: &mut DevBuf,
        m: usize,
        x_dtype: Dtype,
        y_dtype: Dtype,
        k: usize,
        n: usize,
        k2: usize,
        _codebook: Codebook,
        trellis: &DevBuf,
        suh: &DevBuf,
        svh: &DevBuf,
        scratch: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if m < 1 || k2 < 2 || k2 > 16 || k == 0 || n == 0 {
            return Err(INVALID);
        }
        if x.bytes() < m * k * x_dtype.size() || y.bytes() < m * n * y_dtype.size() {
            return Err(INVALID);
        }
        if scratch.bytes() < prefill_scratch_bytes(m, k, n) {
            return Err(INVALID);
        }
        #[cfg(not(gb_no_cuda))]
        unsafe {
            // the pinned prefill scratch lays out [xh: m*k fp16][W_q: k*n fp16][c: m*n fp32], each 256-aligned; the
            // W_q section is exactly `fmoe::unpack`'s destination
            let w_off = (m * k * 2 + 255) & !255;
            let urc = crate::fmoe::unpack(trellis, &mut scratch.subview(w_off, k * n * 2), k, n, k2, st);
            urc?;
            let wrc = CudaError::check(ffi::gb_exl3_prefill_pinned_with_w(
                x.ptr(),
                x_dtype as i32,
                scratch.ptr_at(w_off),
                suh.ptr(),
                svh.ptr(),
                core::ptr::null(),
                y.ptr(),
                y_dtype as i32,
                m as i32,
                k as i32,
                n as i32,
                scratch.ptr(),
                scratch.bytes(),
                st.raw(),
            ));
            wrc
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (x, y, trellis, suh, svh, scratch, st);
            Err(CudaError(-1))
        }
    }

    /// [`Exl3Weights::forward_prefill_pinned`] with `W_q` from the caller: `w` (at least `k * n * 2` bytes) holds the
    /// layer's `W_q` as [`Exl3Weights::reconstruct`] writes it (a pure function of the trellis, so bits the per-chunk
    /// reconstruct would have produced: the persistent path is bit-identical to today's pinned prefill by construction).
    /// The same pinned cublasLt GEMM, the same rotations, the same `scratch` (its `W_q` section is unused). Row
    /// invariant, like the pinned path. Do NOT use [`Exl3Weights::prefill_with_w`] on a resident `W_q` instead: that
    /// method runs the heuristic `cublasGemmEx`, which is not row invariant and would move bits.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill_pinned_with_w(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        w: &DevBuf,
        y: &mut DevBuf,
        y_dtype: Dtype,
        m: usize,
        scratch: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.check_io(x, y, m, x_dtype, y_dtype)?;
        if w.bytes() < self.k * self.n * 2 || scratch.bytes() < self.prefill_scratch_bytes(m) {
            return Err(INVALID);
        }
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_exl3_prefill_pinned_with_w(
                x.ptr(),
                x_dtype as i32,
                w.ptr(),
                self.suh.ptr(),
                self.svh.ptr(),
                self.bias_ptr(),
                y.ptr(),
                y_dtype as i32,
                m as i32,
                self.k as i32,
                self.n as i32,
                scratch.ptr(),
                scratch.bytes(),
                st.raw(),
            ))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (w, scratch, st);
            Err(CudaError(-1))
        }
    }

    /// The prefill pieces for a caller that keeps W_q reconstructed across chunks: `xh` fp16 [m, k] and `c` fp32
    /// [m, n] are the caller's buffers, `w` a buffer filled by [`Exl3Weights::reconstruct`].
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_with_w(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        w: &DevBuf,
        xh: &mut DevBuf,
        c: &mut DevBuf,
        y: &mut DevBuf,
        y_dtype: Dtype,
        m: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.check_io(x, y, m, x_dtype, y_dtype)?;
        if xh.bytes() < m * self.k * 2 || c.bytes() < m * self.n * 4 || w.bytes() < self.k * self.n * 2 {
            return Err(INVALID);
        }
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_exl3_rotate_in(
                x.ptr(),
                x_dtype as i32,
                self.suh.ptr(),
                xh.ptr(),
                m as i32,
                self.k as i32,
                st.raw(),
            ))?;
            CudaError::check(ffi::gb_exl3_prefill_gemm(
                xh.ptr(),
                w.ptr(),
                c.ptr(),
                m as i32,
                self.k as i32,
                self.n as i32,
                st.raw(),
            ))?;
            CudaError::check(ffi::gb_exl3_rotate_out(
                c.ptr(),
                self.svh.ptr(),
                self.bias_ptr(),
                y.ptr(),
                y_dtype as i32,
                m as i32,
                self.n as i32,
                st.raw(),
            ))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (w, xh, c, st);
            Err(CudaError(-1))
        }
    }
}

/// The row-invariance sweep: the evidence that [`Exl3Weights::forward_prefill_pinned`] keeps its promise on this device and
/// cuBLAS, for one layer shape. Used by the unit tests here (synthetic weights) and by the model's tests (every prefill shape
/// of the real pack); a diagnostic, never on the serving path.
pub mod sweep {
    use super::*;
    use crate::glue;

    /// Rows of the reference call: the largest prefill chunk (`--prefill-chunk` is capped at 4096).
    pub const REF_ROWS: usize = 4096;

    /// Which prefill GEMM the sweep runs.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Path {
        /// [`Exl3Weights::forward_prefill_pinned`]
        Pinned,
        /// [`Exl3Weights::forward_prefill`] (`cublasGemmEx`): the negative control, which is NOT row invariant at small `m`.
        Legacy,
    }

    fn run(w: &Exl3Weights, path: Path, x: &DevBuf, y: &mut DevBuf, m: usize, scratch: &mut DevBuf, st: &Stream) {
        match path {
            Path::Pinned => w.forward_prefill_pinned(x, y, m, Dtype::BF16, scratch, st),
            Path::Legacy => w.forward_prefill(x, y, m, Dtype::BF16, scratch, st),
        }
        .expect("prefill call");
    }

    /// Deterministic bf16 activations `[rows, k]` (a spread of magnitudes and signs, no NaN or infinity).
    fn activations(rows: usize, k: usize, seed: u64) -> Vec<u16> {
        let mut z = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        (0..rows * k)
            .map(|_| {
                z ^= z << 13;
                z ^= z >> 7;
                z ^= z << 17;
                // sum of 4 uniforms: roughly normal
                let u: f32 = (0..4).map(|i| ((z >> (16 * i)) & 0xffff) as f32 / 65536.0).sum::<f32>() - 2.0;
                let b = (u * 1.7).to_bits();
                (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
            })
            .collect()
    }

    /// The row counts of the default sweep: every m from 1 to 300 (the GEMV limit of 128 and the 256 boundary included), the
    /// neighbours of the tile edges up to 4096, odd sizes, and 4,000 to 4,096 in full. `full` is every m up to 4,096 (several
    /// minutes for the biggest shapes).
    pub fn grid(full: bool) -> Vec<usize> {
        let mut v: Vec<usize> = if full { (1..=REF_ROWS).collect() } else { (1..=300).collect() };
        if !full {
            for b in [384usize, 512, 640, 768, 896, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096] {
                v.extend([b - 1, b, b + 1]);
            }
            v.extend([333, 517, 777, 1000, 1337, 1777, 2222, 2999, 3333, 3777, 4001, 4033]);
            v.extend(4000..=REF_ROWS);
        }
        v.retain(|&m| (1..=REF_ROWS).contains(&m));
        v.sort_unstable();
        v.dedup();
        v
    }

    /// For each `m` of `ms`: rows `o..o + m` of one `REF_ROWS`-row input, run as a call of `m` rows at two or three offsets `o`
    /// (the start, a ragged one, and for every third `m` the end of the input), must give exactly the bits that the
    /// `REF_ROWS`-row call gave those rows. Returns the number of mismatching (m, offset) cases (and prints the first few);
    /// zero is the guarantee.
    pub fn row_invariance(w: &Exl3Weights, path: Path, ms: &[usize], st: &Stream) -> Result<usize, String> {
        let (k, n) = (w.k, w.n);
        let e = |e: CudaError| e.to_string();
        let x_host = activations(REF_ROWS, k, 0x5eed ^ ((k as u64) << 20) ^ n as u64);
        let x_full = DevBuf::from_slice(&x_host).map_err(e)?;
        let mut scratch = DevBuf::zeroed(w.prefill_scratch_bytes(REF_ROWS)).map_err(e)?;
        let mut y = DevBuf::zeroed(REF_ROWS * n * 2).map_err(e)?;
        run(w, path, &x_full, &mut y, REF_ROWS, &mut scratch, st);
        st.sync().map_err(e)?;
        let mut y_ref = vec![0u16; REF_ROWS * n];
        y.download(&mut y_ref).map_err(e)?;
        let mut got = vec![0u16; REF_ROWS * n];
        let x_part = DevBuf::zeroed(REF_ROWS * k * 2).map_err(e)?;
        let mut bad = 0usize;
        let mut first: Vec<String> = Vec::new();
        for &m in ms {
            let room = REF_ROWS - m;
            let ragged = if room == 0 { 0 } else { (m * 7 + 13) % (room + 1) };
            let mut offsets = vec![0, ragged];
            if room > 0 && m % 3 == 0 {
                offsets.push(room); // flush with the end of the input
            }
            offsets.dedup();
            for o in offsets {
                unsafe {
                    glue::copy_rows((x_full.ptr() as *const u16).add(o * k) as *const _, k, x_part.ptr(), k, m, k, st)
                        .map_err(e)?;
                }
                run(w, path, &x_part, &mut y, m, &mut scratch, st);
                st.sync().map_err(e)?;
                y.download(&mut got[..m * n]).map_err(e)?;
                let want = &y_ref[o * n..(o + m) * n];
                let rows_bad = (0..m).filter(|&r| got[r * n..(r + 1) * n] != want[r * n..(r + 1) * n]).count();
                if rows_bad > 0 {
                    bad += 1;
                    if first.len() < 6 {
                        first.push(format!("m={m} offset={o}: {rows_bad} of {m} rows differ"));
                    }
                }
            }
        }
        if bad > 0 {
            eprintln!("{path:?} k={k} n={n}: {bad} (m, offset) cases differ from the {REF_ROWS}-row call; first: {first:?}");
        }
        Ok(bad)
    }

    /// The `m` values of `ms` for which the pinned path and the legacy `cublasGemmEx` path give different bits (rows from 0).
    /// For an m above 128, absence from the answer means pinning the GEMM did not move the bits of a chunk of that many rows;
    /// the sizes in the answer are those at which the legacy path was off the large-m bits (the model's bits-change set).
    pub fn pinned_vs_legacy(w: &Exl3Weights, ms: &[usize], st: &Stream) -> Result<Vec<usize>, String> {
        let (k, n) = (w.k, w.n);
        let e = |e: CudaError| e.to_string();
        let big = ms.iter().copied().max().unwrap_or(1);
        let x = DevBuf::from_slice(&activations(big, k, 0xfeed ^ n as u64)).map_err(e)?;
        let mut scratch = DevBuf::zeroed(w.prefill_scratch_bytes(big)).map_err(e)?;
        let mut y = DevBuf::zeroed(big * n * 2).map_err(e)?;
        let mut differ = Vec::new();
        let (mut a, mut b) = (vec![0u16; big * n], vec![0u16; big * n]);
        for &m in ms {
            run(w, Path::Pinned, &x, &mut y, m, &mut scratch, st);
            st.sync().map_err(e)?;
            y.download(&mut a[..m * n]).map_err(e)?;
            run(w, Path::Legacy, &x, &mut y, m, &mut scratch, st);
            st.sync().map_err(e)?;
            y.download(&mut b[..m * n]).map_err(e)?;
            if a[..m * n] != b[..m * n] {
                differ.push(m);
            }
        }
        Ok(differ)
    }
}

#[cfg(all(test, not(gb_no_cuda)))]
mod tests {
    use super::*;
    use crate::rt::device_available;
    use gb_exl3::checkpoint::{scan, Checkpoint, Group};
    use gb_exl3::f16::f16_to_f64;
    use gb_exl3::{codebook, forward, unpack, Bits};
    use std::path::PathBuf;

    // ---------------------------------------------------------------- helpers

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            // splitmix64
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn uniform(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let (a, b) = (self.uniform().max(1e-300), self.uniform());
            (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    fn round_f16(x: f64) -> f64 {
        f16_to_f64(f64_to_f16(x))
    }

    /// f32 -> bf16 bits, round to nearest even.
    fn bf16_bits(x: f32) -> u16 {
        let b = x.to_bits();
        ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
    }

    fn bf16_to_f64(h: u16) -> f64 {
        f32::from_bits((h as u32) << 16) as f64
    }

    /// Activation values as the device sees them (rounded to `dtype`), plus their raw bytes.
    fn encode(vals: &[f64], dtype: Dtype) -> (Vec<f64>, Vec<u8>) {
        let mut rounded = Vec::with_capacity(vals.len());
        let mut bytes = Vec::new();
        for &v in vals {
            match dtype {
                Dtype::F16 => {
                    let h = f64_to_f16(v);
                    rounded.push(f16_to_f64(h));
                    bytes.extend_from_slice(&h.to_le_bytes());
                }
                Dtype::BF16 => {
                    let h = bf16_bits(v as f32);
                    rounded.push(bf16_to_f64(h));
                    bytes.extend_from_slice(&h.to_le_bytes());
                }
                Dtype::F32 => {
                    rounded.push(v as f32 as f64);
                    bytes.extend_from_slice(&(v as f32).to_le_bytes());
                }
            }
        }
        (rounded, bytes)
    }

    fn decode(bytes: &[u8], dtype: Dtype) -> Vec<f64> {
        match dtype {
            Dtype::F16 => bytes.chunks_exact(2).map(|c| f16_to_f64(u16::from_le_bytes([c[0], c[1]]))).collect(),
            Dtype::BF16 => bytes.chunks_exact(2).map(|c| bf16_to_f64(u16::from_le_bytes([c[0], c[1]]))).collect(),
            Dtype::F32 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect(),
        }
    }

    fn scale_vec(r: &mut Rng, len: usize) -> Vec<f64> {
        (0..len).map(|_| round_f16((if r.next() & 1 == 1 { -1.0 } else { 1.0 }) * (0.5 + r.uniform()))).collect()
    }

    fn make_group(k: usize, n: usize, bits: usize, cb: Codebook, bias: bool, seed: u64) -> LoadedGroup {
        let mut r = Rng(seed);
        let words = (k / 16) * (n / 16) * 16 * bits;
        let trellis: Vec<i16> = (0..words).map(|_| r.next() as i16).collect();
        let b = bias.then(|| (0..n).map(|_| round_f16(0.3 * r.normal())).collect());
        let group = Group {
            prefix: "synthetic".into(),
            bits: Bits::from_twice(2 * bits as u8).unwrap(),
            codebook: cb,
            k,
            n,
            in_scales: "suh",
            out_scales: "svh",
            bias,
            files: vec![],
            bytes: 0,
        };
        LoadedGroup { group, trellis, kt: k / 16, nt: n / 16, suh: scale_vec(&mut r, k), svh: scale_vec(&mut r, n), bias: b }
    }

    fn run_decode(w: &Exl3Weights, xbytes: &[u8], m: usize, xd: Dtype, yd: Dtype, ws: &mut DevBuf, st: &Stream) -> Vec<u8> {
        let x = DevBuf::from_slice(xbytes).unwrap();
        let mut y = DevBuf::zeroed(m * w.n * yd.size()).unwrap();
        w.forward(&x, &mut y, m, xd, yd, ws, st).unwrap();
        st.sync().unwrap();
        y.to_vec::<u8>().unwrap()
    }

    /// Error of a result against the float64 reference, per row.
    ///
    /// Bound for element j of a row with reference rms R: |err_j| <= eps_out * |ref_j| + C * R, where
    /// * eps_out * |ref_j| is the output rounding (half an ulp: 2^-11 relative for fp16, 2^-8 for bf16, ~0 for fp32);
    /// * C * R covers the rest, which is relative to the row's scale, not to element j: the kernel rounds the
    ///   rotated activations to fp16 (each of K values off by up to 2^-11 relative, independent, which adds
    ///   about 2^-11 / sqrt(3) * R of noise to a pre-rotation sum, and the output rotation spreads it evenly
    ///   over the 128 columns of a block), plus fp32 accumulation over K (2^-24 * sqrt(K), negligible next to it).
    ///   The worst of ~10^5 elements sits near 4.5 sigma = 1.3e-3 R, so C = 2^-9 (1.95e-3) leaves some margin and
    ///   still catches any real bug (a wrong tile or scale is an O(1) R error).
    ///
    /// Returns (worst err / R, worst err / bound).
    fn compare(got: &[f64], want: &[f64], rows: usize, n: usize, yd: Dtype, what: &str) -> (f64, f64) {
        assert_eq!(got.len(), rows * n);
        let eps_out = match yd {
            Dtype::F16 => 2f64.powi(-11),
            Dtype::BF16 => 2f64.powi(-8),
            Dtype::F32 => 2f64.powi(-23),
        };
        let c = 2f64.powi(-9);
        let (mut worst_rel, mut worst_ratio) = (0f64, 0f64);
        for r in 0..rows {
            let (g, w) = (&got[r * n..(r + 1) * n], &want[r * n..(r + 1) * n]);
            let rms = (w.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
            for j in 0..n {
                let err = (g[j] - w[j]).abs();
                let bound = eps_out * w[j].abs() + c * rms;
                worst_rel = worst_rel.max(err / rms);
                worst_ratio = worst_ratio.max(err / bound);
                assert!(
                    err <= bound,
                    "{what}: row {r} col {j}: got {} want {} (err {err:.3e}, bound {bound:.3e}, rms {rms:.3e})",
                    g[j],
                    w[j]
                );
            }
        }
        (worst_rel, worst_ratio)
    }

    /// A workspace full of junk: the decode path must not need it to start in any state.
    fn poisoned(bytes: usize, st: &Stream) -> DevBuf {
        let mut b = DevBuf::alloc(bytes).unwrap();
        b.memset_async(0xAB, st).unwrap();
        st.sync().unwrap();
        b
    }

    fn gpu() -> Option<Stream> {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return None;
        }
        Some(Stream::new().unwrap())
    }

    fn xs(rows: usize, k: usize, seed: u64) -> Vec<f64> {
        let mut r = Rng(seed);
        (0..rows * k).map(|_| r.normal()).collect()
    }

    // ---------------------------------------------------------------- exact decode

    #[test]
    fn reconstruct_is_bit_exact_for_every_width_and_codebook() {
        let Some(st) = gpu() else { return };
        for bits in 1..=8usize {
            for cb in Codebook::ALL {
                let g = make_group(256, 384, bits, cb, false, 11 + bits as u64);
                let w = Exl3Weights::from_loaded(&g).unwrap();
                let mut d = DevBuf::zeroed(256 * 384 * 2).unwrap();
                w.reconstruct(&mut d, &st).unwrap();
                st.sync().unwrap();
                let got: Vec<u16> = d.to_vec().unwrap();
                let want = unpack(&g.trellis, g.kt, g.nt, g.group.bits, &codebook(cb));
                assert!(got == want, "reconstruct differs from gb_exl3::unpack at {bits} bits {}", cb.name());
            }
        }
    }

    // ---------------------------------------------------------------- (a) against the float64 reference

    const M_LIST: [usize; 14] = [1, 2, 3, 8, 16, 17, 24, 32, 33, 48, 49, 63, 64, 128];

    #[allow(clippy::too_many_arguments)]
    fn check_against_reference(
        k: usize,
        n: usize,
        bits: usize,
        cb: Codebook,
        bias: bool,
        ms: &[usize],
        ref_rows: usize,
        worst: &mut [(f64, f64); 2],
    ) {
        let st = gpu().unwrap();
        let g = make_group(k, n, bits, cb, bias, 1000 + (k + n + bits) as u64);
        let w = Exl3Weights::from_loaded(&g).unwrap();
        let table = codebook(cb);
        let mut ws = poisoned(ms.iter().map(|&m| w.workspace_bytes(m)).max().unwrap(), &st);
        let x = xs(ref_rows, k, 7 + bits as u64);
        for xd in [Dtype::F16, Dtype::BF16] {
            let (xr, xb) = encode(&x, xd);
            let want = forward(&xr, ref_rows, &g.trellis, g.kt, g.nt, g.group.bits, &table, &g.suh, &g.svh, g.bias.as_deref());
            for &m in ms {
                let out = run_decode(&w, &xb[..m * k * xd.size()], m, xd, xd, &mut ws, &st);
                let got = decode(&out, xd);
                let what = format!("{bits} bits {} k={k} n={n} m={m} {xd:?} bias={bias}", cb.name());
                let (rel, ratio) = compare(&got, &want[..m * n], m, n, xd, &what);
                let slot = &mut worst[(xd == Dtype::BF16) as usize];
                slot.0 = slot.0.max(rel);
                slot.1 = slot.1.max(ratio);
            }
        }
    }

    fn report_worst(what: &str, w: &[(f64, f64); 2]) {
        eprintln!(
            "worst error vs f64 ({what}): fp16 {:.3e} of row rms ({:.3} of the bound), bf16 {:.3e} ({:.3})",
            w[0].0, w[0].1, w[1].0, w[1].1
        );
    }

    #[test]
    fn matches_float64_reference_all_widths() {
        if gpu().is_none() {
            return;
        }
        let mut worst = [(0.0, 0.0); 2];
        for bits in 1..=8usize {
            check_against_reference(512, 384, bits, Codebook::Mul1, bits % 2 == 0, &M_LIST, 128, &mut worst);
        }
        // every codebook at the other widths too, on a smaller M set
        for cb in [Codebook::Mcg, Codebook::Inst3] {
            check_against_reference(512, 384, 4, cb, true, &M_LIST, 128, &mut worst);
            for bits in [1usize, 2, 3, 5, 6, 7, 8] {
                check_against_reference(512, 384, bits, cb, false, &[1, 16, 17], 17, &mut worst);
            }
        }
        report_worst("all widths", &worst);
    }

    #[test]
    fn matches_float64_reference_every_plan_branch() {
        if gpu().is_none() {
            return;
        }
        let mut worst = [(0.0, 0.0); 2];
        // K=4096 N=256 splits K (SK=8, WK=4); K=1024 N=8192 runs 8 warps (SK=1, WK=8); both with bias
        check_against_reference(4096, 256, 4, Codebook::Mul1, true, &M_LIST, 128, &mut worst);
        check_against_reference(4096, 256, 6, Codebook::Mul1, false, &[1, 16, 33], 33, &mut worst);
        check_against_reference(1024, 8192, 4, Codebook::Mul1, true, &[1, 3, 17], 17, &mut worst);
        check_against_reference(1024, 8192, 6, Codebook::Mul1, false, &[1, 16], 16, &mut worst);
        report_worst("plan branches", &worst);
    }

    // ---------------------------------------------------------------- MiniCPM-V-4.7 (the Qwen3.5/3.6-35B-A3B text backbone + DSpark draft)

    /// (the layers, K, N) of every EXL3 linear the text backbone and its DSpark draft run: hidden 2048, 16 query / 2 KV heads of 256,
    /// GDN 16 key x 32 value heads of 128, 256 experts of 512 plus a shared one, 248,192 padded logits; draft intermediate 6144 and a
    /// 10,240-wide fusion fc. Every other linear (router, in_proj_a/b, the shared expert gate) is not EXL3.
    const MINICPM_SHAPES: [(&str, usize, usize); 10] = [
        ("q_proj, GDN in_proj_qkv", 2048, 8192),
        ("GDN in_proj_z", 2048, 4096),
        ("o_proj, GDN out_proj", 4096, 2048),
        ("k_proj, v_proj, expert gate/up, shared gate/up, draft k/v", 2048, 512),
        ("expert down, shared down", 512, 2048),
        ("lm_head", 2048, 248_192),
        ("draft fc", 10_240, 2048),
        ("draft q/o", 2048, 2048),
        ("draft gate/up", 2048, 6144),
        ("draft down", 6144, 2048),
    ];

    /// The decode plan (K splits, warps a block) and the pinned prefill tile of every MiniCPM shape, written down: a change to
    /// `plan_for` or `pin_for` that touches one of these layers (its reduction order, so its bits) must show up here. The plan is a
    /// function of (K, N) alone; the table was derived by hand from `plan_for` (a narrow layer splits K to fill the SMs, at least 8
    /// k tiles per split) before it was asserted.
    #[test]
    fn minicpm_shapes_have_the_documented_decode_plans_and_the_default_prefill_pin() {
        if gpu().is_none() {
            return;
        }
        let want: [((usize, usize), (usize, usize)); 10] = [
            ((2048, 8192), (1, 8)),
            ((2048, 4096), (4, 4)),
            ((4096, 2048), (8, 4)),
            ((2048, 512), (4, 4)),
            ((512, 2048), (1, 4)),
            ((2048, 248_192), (1, 8)),
            ((10_240, 2048), (16, 4)),
            ((2048, 2048), (4, 4)),
            ((2048, 6144), (4, 4)),
            ((6144, 2048), (8, 4)),
        ];
        assert_eq!(want.len(), MINICPM_SHAPES.len());
        let default_pin = prefill_pin_for(5120, 17408).unwrap();
        assert_ne!(prefill_pin_for(17408, 5120).unwrap(), default_pin, "the 27B's down_proj has its own tile");
        for ((k, n), plan) in want {
            assert!(MINICPM_SHAPES.iter().any(|s| (s.1, s.2) == (k, n)), "{k}x{n} is not a MiniCPM shape");
            assert_eq!(decode_plan(k, n).unwrap(), plan, "decode plan of {k}x{n}");
            // the invariants every plan keeps: the splits divide the k tiles and each split has work
            let (sk, wk) = plan;
            assert!((k / 16) % (sk * wk) == 0 || sk * wk == 1, "{k}x{n}");
            assert_eq!(prefill_pin_for(k, n).unwrap(), default_pin, "{k}x{n} runs the default pinned tile");
        }
        assert!(decode_plan(100, 128).is_err() && decode_plan(128, 100).is_err());
    }

    /// Row invariance of the decode path at every MiniCPM shape and every width 3..8 (the runtime widths the model's packs use):
    /// a row's output bits are the same alone and inside a window of 2..128 rows, in any order. The lm_head (a 0.5 GB trellis a width)
    /// runs the shipping width 6 and 4 on a smaller pool.
    #[test]
    fn minicpm_shapes_decode_rows_are_invariant_at_every_width() {
        if gpu().is_none() {
            return;
        }
        for &(what, k, n) in &MINICPM_SHAPES {
            let widths: &[usize] = if n > 100_000 { &[6, 4] } else { &[3, 4, 5, 6, 7, 8] };
            let pool = if n > 100_000 { 33 } else { 128 };
            for &bits in widths {
                eprintln!("decode row invariance: {what} ({k} x {n}) {bits} bits");
                check_row_invariance(k, n, bits, Codebook::Mul1, false, pool, &COMBOS[1..2]);
            }
        }
    }

    /// The decode path against the float64 reference at every MiniCPM shape (not the lm_head: a CPU reference over 0.5 G weights) at
    /// three widths and the window sizes a decode step, a 4-stream verify and a draft block use.
    #[test]
    fn minicpm_shapes_decode_matches_the_float64_reference() {
        if gpu().is_none() {
            return;
        }
        let mut worst = [(0.0, 0.0); 2];
        for &(_, k, n) in MINICPM_SHAPES.iter().filter(|s| s.2 < 100_000) {
            for bits in [3usize, 6, 8] {
                check_against_reference(k, n, bits, Codebook::Mul1, false, &[1, 8, 16], 16, &mut worst);
            }
        }
        report_worst("MiniCPM shapes", &worst);
    }

    /// The lm_head at its real width against the float64 reference on a few 128-column blocks: the (2048, 248192) layer's blocks near
    /// the start, in the middle and at the end (the padded tail the top-k masks), through `select_strips` (which keeps >= 64 strips).
    #[test]
    fn minicpm_lm_head_strips_match_the_float64_reference() {
        let Some(st) = gpu() else { return };
        let (k, n, bits) = (2048usize, 248_192usize, 6usize);
        let g = make_group(k, n, bits, Codebook::Mul1, false, 4242);
        let w = Exl3Weights::from_loaded(&g).unwrap();
        let table = codebook(Codebook::Mul1);
        let x = xs(4, k, 77);
        let (xr, xb) = encode(&x, Dtype::BF16);
        let nb = n / 128;
        for (first, count) in [(0usize, 64usize), (nb / 2, 64), (nb - 64, 64)] {
            let part = w.select_strips(&[(first, first + count)]).unwrap();
            let cols = count * 128;
            let blocks: Vec<usize> = (first..first + count).collect();
            let sub = slice_blocks(&g, &blocks);
            let want = forward(&xr, 4, &sub.trellis, sub.kt, sub.nt, sub.group.bits, &table, &sub.suh, &sub.svh, None);
            let mut ws = poisoned(part.workspace_bytes(4), &st);
            let out = run_decode(&part, &xb[..4 * k * 2], 4, Dtype::BF16, Dtype::BF16, &mut ws, &st);
            let got = decode(&out, Dtype::BF16);
            compare(&got, &want, 4, cols, Dtype::BF16, &format!("lm_head blocks {first}..{}", first + count));
        }
    }

    /// 128 rows is the most one decode call takes. Past it the call is refused in Rust before anything launches (the C launcher refuses
    /// too, with `cudaErrorInvalidValue`, and its workspace-size query answers 0 for the shape), never run on part of the rows and left
    /// to return zeros: the output buffer is untouched. A caller that wants more rows splits them into calls of up to 128 itself; no
    /// path of the engine does (a window is capped at 128, a draft round at 16 sequences x 8 rows).
    #[test]
    fn a_decode_call_past_128_rows_is_refused_and_writes_nothing() {
        let Some(st) = gpu() else { return };
        let g = make_group(512, 384, 4, Codebook::Mul1, false, 17);
        let w = Exl3Weights::from_loaded(&g).unwrap();
        assert!(w.workspace_bytes(128) > 0);
        for m in [129usize, 130, 255, 256, 4096] {
            assert_eq!(w.workspace_bytes(m), 0, "the size query says 0 for {m} rows (an unsupported shape)");
            let mut ws = poisoned(w.workspace_bytes(128), &st);
            let x = DevBuf::zeroed(m * 512 * 2).unwrap();
            let mut y = poisoned(m * 384 * 2, &st);
            let r = w.forward(&x, &mut y, m, Dtype::BF16, Dtype::BF16, &mut ws, &st);
            assert!(r.is_err(), "{m} rows must be refused, not run");
            st.sync().unwrap();
            let bytes: Vec<u8> = y.to_vec().unwrap();
            assert!(bytes.iter().all(|&b| b == 0xAB), "a refused call at {m} rows wrote into the output");
        }
        // 128 rows are fine, and the same buffers serve them
        let mut ws = poisoned(w.workspace_bytes(128), &st);
        let x = DevBuf::zeroed(128 * 512 * 2).unwrap();
        let mut y = poisoned(128 * 384 * 2, &st);
        w.forward(&x, &mut y, 128, Dtype::BF16, Dtype::BF16, &mut ws, &st).unwrap();
        st.sync().unwrap();
        let bytes: Vec<u8> = y.to_vec().unwrap();
        assert!(bytes.iter().any(|&b| b != 0xAB), "128 rows must have written their output");
    }

    #[test]
    fn f32_activations_match_reference() {
        let Some(st) = gpu() else { return };
        let g = make_group(512, 256, 4, Codebook::Mul1, true, 5);
        let w = Exl3Weights::from_loaded(&g).unwrap();
        let mut ws = poisoned(w.workspace_bytes(5), &st);
        let x = xs(5, 512, 3);
        let (xr, xb) = encode(&x, Dtype::F32);
        let want =
            forward(&xr, 5, &g.trellis, g.kt, g.nt, g.group.bits, &codebook(Codebook::Mul1), &g.suh, &g.svh, g.bias.as_deref());
        let out = run_decode(&w, &xb, 5, Dtype::F32, Dtype::F32, &mut ws, &st);
        compare(&decode(&out, Dtype::F32), &want, 5, 256, Dtype::F32, "f32");
    }

    // ---------------------------------------------------------------- strips of a layer (the draft's vocabulary head)

    // The reference: what `select_strips` did before it gathered on the device. The whole trellis goes to host memory, the strips are
    // gathered there, and the result is uploaded; the device gather must write exactly these bytes.

    /// The trellis words of some 128-column strips of a K x N layer (`ranges`: ascending, disjoint, half-open, in strips).
    /// A strip of one row of tiles (`kt`) is a contiguous run of `64 * bits` words (see `strip_trellis_plan`).
    fn gather_strip_words(words: &[u32], k: usize, n: usize, bits: usize, ranges: &[(usize, usize)]) -> Vec<u32> {
        let strip = 64 * bits;
        let row = (n / 128) * strip;
        assert_eq!(words.len(), (k / 16) * row, "trellis size does not match K x N x bits");
        let kept: usize = ranges.iter().map(|&(lo, hi)| hi - lo).sum();
        let mut out = Vec::with_capacity((k / 16) * kept * strip);
        for kt in 0..k / 16 {
            for &(lo, hi) in ranges {
                out.extend_from_slice(&words[kt * row + lo * strip..kt * row + hi * strip]);
            }
        }
        out
    }

    /// The host-staged `select_strips`, as it was before the device gather.
    fn select_strips_host(w: &Exl3Weights, ranges: &[(usize, usize)]) -> Exl3Weights {
        let kept: usize = ranges.iter().map(|&(lo, hi)| hi - lo).sum();
        let words: Vec<u32> = w.trellis.to_vec::<u32>().unwrap();
        let trellis = gather_strip_words(&words, w.k, w.n, w.bits, ranges);
        let pick =
            |full: &[u16]| -> Vec<u16> { ranges.iter().flat_map(|&(lo, hi)| full[lo * 128..hi * 128].iter().copied()).collect() };
        let svh = pick(&w.svh.to_vec::<u16>().unwrap());
        let bias = w.bias.as_ref().map(|b| DevBuf::from_slice(&pick(&b.to_vec::<u16>().unwrap())).unwrap());
        Exl3Weights {
            trellis: DevBuf::from_slice(&trellis).unwrap(),
            suh: DevBuf::from_slice(&w.suh.to_vec::<u16>().unwrap()).unwrap(),
            svh: DevBuf::from_slice(&svh).unwrap(),
            bias,
            k: w.k,
            n: kept * 128,
            bits: w.bits,
            codebook: w.codebook,
        }
    }

    /// Run a copy plan on host bytes: `height` rows of `width` bytes from `src` into `dst`, as `cudaMemcpy2D` does.
    fn apply_plan(plan: &[Copy2d], src: &[u8], dst: &mut [u8]) {
        for c in plan {
            for row in 0..c.height {
                let (s, d) = (c.src_off + row * c.src_pitch, c.dst_off + row * c.dst_pitch);
                dst[d..d + c.width].copy_from_slice(&src[s..s + c.width]);
            }
        }
    }

    fn le_bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// Words that are all different (an odd multiplier is a bijection on u32), so a copy from a wrong offset, pitch or order differs
    /// at its first wrong byte.
    fn pattern_words(count: usize, salt: u32) -> Vec<u32> {
        (0..count).map(|i| (i as u32 ^ salt).wrapping_mul(0x9E37_79B1).rotate_left(7)).collect()
    }

    fn pattern_u16(count: usize, salt: u32) -> Vec<u16> {
        (0..count).map(|i| (i as u32 ^ salt).wrapping_mul(0x9E37_79B1).rotate_left(13) as u16).collect()
    }

    /// A layer of the given shape made of pattern bytes (not a valid quantisation: only the copy of its bytes is under test).
    fn pattern_layer(k: usize, n: usize, bits: usize, bias: bool) -> Exl3Weights {
        let words = pattern_words((k / 16) * (n / 16) * 8 * bits, 0x51);
        Exl3Weights {
            trellis: DevBuf::from_slice(&words).unwrap(),
            suh: DevBuf::from_slice(&pattern_u16(k, 0x52)).unwrap(),
            svh: DevBuf::from_slice(&pattern_u16(n, 0x53)).unwrap(),
            bias: bias.then(|| DevBuf::from_slice(&pattern_u16(n, 0x54)).unwrap()),
            k,
            n,
            bits,
            codebook: Codebook::Mul1,
        }
    }

    /// Every field and every byte of two layers equal; the first differing byte is named.
    fn assert_same_layer(a: &Exl3Weights, b: &Exl3Weights, what: &str) {
        assert_eq!((a.k, a.n, a.bits, a.codebook), (b.k, b.n, b.bits, b.codebook), "{what}: shape");
        let (x, y) = (a.buffers(), b.buffers());
        assert_eq!(x.len(), y.len(), "{what}: which buffers exist (bias on one side only)");
        for ((nx, bx), (ny, by)) in x.iter().zip(&y) {
            assert_eq!(nx, ny, "{what}: buffer order");
            assert_eq!(bx.bytes(), by.bytes(), "{what}: {nx} size");
            let (u, v) = (bx.to_vec::<u8>().unwrap(), by.to_vec::<u8>().unwrap());
            if u != v {
                let at = u.iter().zip(&v).position(|(p, q)| p != q).unwrap();
                panic!("{what}: {nx} differs, first at byte {at} of {} ({:#04x} against {:#04x})", u.len(), u[at], v[at]);
            }
        }
    }

    /// The gate sets `REXL3_REQUIRE_GPU=1`: a test that cannot run for want of a device or a pack must then fail, not pass vacuously.
    fn require_gpu() -> bool {
        std::env::var("REXL3_REQUIRE_GPU").is_ok_and(|v| v == "1")
    }

    /// `not run` (never the word the gate reads as a vacuous pass) when there is no device; a failure under `REXL3_REQUIRE_GPU=1`.
    fn gpu_or_not_run(test: &str) -> bool {
        if device_available() {
            return true;
        }
        assert!(!require_gpu(), "REXL3_REQUIRE_GPU=1 but {test} found no CUDA device");
        eprintln!("{test}: not run (no CUDA device)");
        false
    }

    /// TensorFold's vocabulary on the real head, as strips: ids [0, 98304) and the specials at the top.
    const TF_STRIPS: [(usize, usize); 2] = [(0, 768), (1937, 1940)];

    /// A layer shape and the strip ranges to keep: (K, N, bits, ranges).
    type StripCase = (usize, usize, usize, &'static [(usize, usize)]);

    #[test]
    fn strip_plan_equals_the_host_gather() {
        // shapes and range sets: the draft's real column count with a short K, a last strip, touching ranges, four ranges, one range, and bits 1..6
        let cases: [StripCase; 7] = [
            (48, 128 * 6, 3, &[(0, 2), (4, 6)]),
            (64, 248_320, 6, &TF_STRIPS),
            (64, 128 * 100, 6, &[(10, 40), (50, 60), (70, 80), (90, 100)]),
            (32, 128 * 100, 6, &[(0, 64)]),
            (32, 128 * 100, 4, &[(36, 100)]),
            (16, 128 * 70, 2, &[(0, 30), (30, 66), (66, 70)]),
            (96, 128 * 130, 1, &[(1, 33), (60, 91)]),
        ];
        for (k, n, bits, ranges) in cases {
            let words = pattern_words((k / 16) * (n / 16) * 8 * bits, 0x51);
            let want = le_bytes(&gather_strip_words(&words, k, n, bits, ranges));
            let kept: usize = ranges.iter().map(|&(lo, hi)| hi - lo).sum();
            let mut got = vec![0xA5u8; want.len()]; // poisoned: a byte no copy reaches stays 0xA5 and differs
            apply_plan(&strip_trellis_plan(k, n, bits, ranges), &le_bytes(&words), &mut got);
            assert_eq!(got.len(), (k / 16) * kept * 64 * bits * 4, "k {k} n {n} bits {bits}");
            if got != want {
                let at = got.iter().zip(&want).position(|(p, q)| p != q).unwrap();
                panic!("k {k} n {n} bits {bits} {ranges:?}: first wrong byte {at} of {}", want.len());
            }
            // the per-column vectors (svh, bias): the same columns, 2 bytes each
            let full = pattern_u16(n, 0x53);
            let want: Vec<u16> = ranges.iter().flat_map(|&(lo, hi)| full[lo * 128..hi * 128].iter().copied()).collect();
            let mut got = vec![0xA5u8; want.len() * 2];
            let bytes: Vec<u8> = full.iter().flat_map(|v| v.to_le_bytes()).collect();
            apply_plan(&strip_vector_plan(ranges), &bytes, &mut got);
            assert_eq!(got, want.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(), "vector, k {k} n {n} {ranges:?}");
        }
        // the real lm_head (K 5120, N 248,320, 6 bits): two rectangles of 320 rows, the first 768 strips wide
        let plan = strip_trellis_plan(5120, 248_320, 6, &TF_STRIPS);
        let strip = 64 * 6 * 4;
        assert_eq!(plan.len(), 2);
        assert_eq!(
            plan[0],
            Copy2d { src_off: 0, src_pitch: 1940 * strip, dst_off: 0, dst_pitch: 771 * strip, width: 768 * strip, height: 320 }
        );
        assert_eq!(
            plan[1],
            Copy2d {
                src_off: 1937 * strip,
                src_pitch: 1940 * strip,
                dst_off: 768 * strip,
                dst_pitch: 771 * strip,
                width: 3 * strip,
                height: 320
            }
        );
    }

    #[test]
    fn device_strip_gather_equals_the_host_path_byte_for_byte() {
        if !gpu_or_not_run("device_strip_gather_equals_the_host_path_byte_for_byte") {
            return;
        }
        // with and without a bias; one, two and four ranges, touching ranges, the first and the last strip; 1 to 6 bits
        let cases: [StripCase; 6] = [
            (512, 128 * 100, 6, &[(0, 50), (80, 100)]),
            (256, 128 * 100, 6, &[(0, 64)]),
            (256, 128 * 100, 4, &[(36, 100)]),
            (1024, 128 * 130, 6, &[(1, 20), (30, 50), (60, 90), (100, 130)]),
            (128, 128 * 70, 2, &[(0, 30), (30, 66), (66, 70)]),
            (1024, 128 * 97, 1, &[(0, 33), (40, 97)]),
        ];
        for bias in [false, true] {
            for (k, n, bits, ranges) in cases {
                let full = pattern_layer(k, n, bits, bias);
                let device = full.select_strips(ranges).unwrap();
                let host = select_strips_host(&full, ranges);
                assert_same_layer(&device, &host, &format!("k {k} n {n} bits {bits} bias {bias} {ranges:?}"));
            }
        }
    }

    #[test]
    fn device_strip_gather_of_the_real_head_shape_equals_the_host_path_byte_for_byte() {
        if !gpu_or_not_run("device_strip_gather_of_the_real_head_shape_equals_the_host_path_byte_for_byte") {
            return;
        }
        // the target's lm_head: K 5120, N 248,320, 6 bits (0.95 GB of trellis, no bias) and the vocabulary the server builds
        let mut full = pattern_layer(5120, 248_320, 6, false);
        let device = full.select_strips(&TF_STRIPS).unwrap();
        assert_eq!((device.k, device.n, device.bits), (5120, 771 * 128, 6));
        assert_same_layer(&device, &select_strips_host(&full, &TF_STRIPS), "real shape");
        // and with a bias, in case a head ever carries one
        full.bias = Some(DevBuf::from_slice(&pattern_u16(248_320, 0x54)).unwrap());
        let device = full.select_strips(&TF_STRIPS).unwrap();
        assert_same_layer(&device, &select_strips_host(&full, &TF_STRIPS), "real shape with bias");
    }

    /// The real pack's `lm_head`: `REXL3_QWEN_DIR=<Qwen3.8-27B-EXL3-4.00bpw> cargo test --release -p gb-cuda --lib real_lm_head -- --nocapture`.
    #[test]
    fn real_lm_head_strips_equal_the_host_path_byte_for_byte() {
        if !gpu_or_not_run("real_lm_head_strips_equal_the_host_path_byte_for_byte") {
            return;
        }
        let Some(dir) = qwen_dir() else {
            assert!(!require_gpu(), "REXL3_REQUIRE_GPU=1 but REXL3_QWEN_DIR is not set or missing");
            eprintln!("real_lm_head_strips_equal_the_host_path_byte_for_byte: not run (REXL3_QWEN_DIR not set or missing)");
            return;
        };
        let ck = scan(&dir, true).unwrap();
        let prefix = ck.groups.keys().find(|p| p.ends_with("lm_head")).expect("the pack has an lm_head").clone();
        let full = Exl3Weights::from_loaded(&ck.load_group(&prefix).unwrap()).unwrap();
        let device = full.select_strips(&TF_STRIPS).unwrap();
        assert_eq!((device.k, device.bits), (full.k, full.bits));
        assert_same_layer(&device, &select_strips_host(&full, &TF_STRIPS), &format!("{prefix} of the real pack"));
        eprintln!(
            "{prefix}: k={} n={} bits={}: {} strips gathered on the device equal the host path",
            full.k, full.n, full.bits, 771
        );
    }

    #[test]
    fn gathered_strips_are_the_right_words_in_order() {
        // every word carries (kt, strip, word-in-strip), so a wrong offset, stride or order shows up as a wrong value
        let (k, n, bits) = (48usize, 128 * 6, 3usize);
        let strip = 64 * bits;
        let mut words = Vec::new();
        for kt in 0..k / 16 {
            for s in 0..n / 128 {
                for w in 0..strip {
                    words.push(((kt * 100 + s) * 1000 + w) as u32);
                }
            }
        }
        let got = gather_strip_words(&words, k, n, bits, &[(0, 2), (4, 6)]);
        assert_eq!(got.len(), (k / 16) * 4 * strip);
        let mut at = 0;
        for kt in 0..k / 16 {
            for s in [0usize, 1, 4, 5] {
                for w in 0..strip {
                    assert_eq!(got[at], ((kt * 100 + s) * 1000 + w) as u32, "kt {kt} strip {s} word {w}");
                    at += 1;
                }
            }
        }
    }

    #[test]
    fn kept_columns_equal_the_full_layers_columns_bit_for_bit() {
        let Some(st) = gpu() else { return };
        for bias in [false, true] {
            // a 6-bit MUL1 layer like the target's lm_head, with enough strips that the K split matches the full layer's
            let (k, n, bits) = (512usize, 128 * 100, 6usize);
            let g = make_group(k, n, bits, Codebook::Mul1, bias, 41);
            let full = Exl3Weights::from_loaded(&g).unwrap();
            let ranges = [(0usize, 50usize), (80, 100)];
            let sub = full.select_strips(&ranges).unwrap();
            assert_eq!((sub.k, sub.n, sub.bits), (k, 128 * 70, bits));
            let cols: Vec<usize> = ranges.iter().flat_map(|&(lo, hi)| lo * 128..hi * 128).collect();
            let mut ws = poisoned(full.workspace_bytes(64), &st);
            for m in [1usize, 5, 16, 17, 32, 33, 48, 64] {
                let (_, xb) = encode(&xs(m, k, 3 + m as u64), Dtype::BF16);
                let a = run_decode(&full, &xb, m, Dtype::BF16, Dtype::BF16, &mut ws, &st);
                let b = run_decode(&sub, &xb, m, Dtype::BF16, Dtype::BF16, &mut ws, &st);
                for r in 0..m {
                    for (c, &gc) in cols.iter().enumerate() {
                        let (i, j) = ((r * n + gc) * 2, (r * sub.n + c) * 2);
                        assert_eq!(a[i..i + 2], b[j..j + 2], "row {r} of {m}, kept column {c} (full column {gc}), bias {bias}");
                    }
                }
            }
        }
    }

    #[test]
    fn strip_selection_refuses_what_it_cannot_keep_exact() {
        let Some(_) = gpu() else { return };
        let g = make_group(256, 128 * 100, 6, Codebook::Mul1, false, 2);
        let w = Exl3Weights::from_loaded(&g).unwrap();
        assert!(w.select_strips(&[]).is_err());
        assert!(w.select_strips(&[(0, 63)]).is_err(), "fewer than 64 strips would change the K split");
        assert!(w.select_strips(&[(10, 50), (40, 90)]).is_err(), "overlapping ranges");
        assert!(w.select_strips(&[(50, 90), (0, 40)]).is_err(), "descending ranges");
        assert!(w.select_strips(&[(0, 101)]).is_err(), "past the last strip");
        assert!(w.select_strips(&[(0, 64)]).is_ok());
    }

    // ---------------------------------------------------------------- (b) row invariance, bit for bit

    fn check_row_invariance(k: usize, n: usize, bits: usize, cb: Codebook, bias: bool, pool: usize, combos: &[(Dtype, Dtype)]) {
        let st = gpu().unwrap();
        let g = make_group(k, n, bits, cb, bias, 77 + (k * 3 + n) as u64);
        let w = Exl3Weights::from_loaded(&g).unwrap();
        let mut ws = poisoned(w.workspace_bytes(128), &st);
        let mut rng = Rng(99);
        let x = xs(pool, k, 13 + n as u64);
        for &(xd, yd) in combos {
            let (_, xb) = encode(&x, xd);
            let (xrow, yrow) = (k * xd.size(), n * yd.size());
            // every pool row alone
            let single: Vec<Vec<u8>> =
                (0..pool).map(|r| run_decode(&w, &xb[r * xrow..(r + 1) * xrow], 1, xd, yd, &mut ws, &st)).collect();
            for &m in &M_LIST {
                if m > pool {
                    continue;
                }
                // a prefix, the reverse of it, and random picks in random order (with the occasional repeated row)
                let mut windows: Vec<Vec<usize>> = vec![(0..m).collect(), (0..m).rev().collect()];
                for _ in 0..3 {
                    windows.push((0..m).map(|_| rng.below(pool)).collect());
                }
                windows.push((0..m).map(|i| (pool - 1 - i % pool + 7 * i) % pool).collect());
                for rows in windows {
                    let mut xw = Vec::with_capacity(m * xrow);
                    for &r in &rows {
                        xw.extend_from_slice(&xb[r * xrow..(r + 1) * xrow]);
                    }
                    let out = run_decode(&w, &xw, m, xd, yd, &mut ws, &st);
                    for (i, &r) in rows.iter().enumerate() {
                        assert!(
                            out[i * yrow..(i + 1) * yrow] == single[r][..],
                            "row {r} at position {i} of {m} differs from its M=1 result (k={k} n={n} {bits} bits {} {xd:?}->{yd:?} bias={bias})",
                            cb.name()
                        );
                    }
                }
            }
        }
    }

    const COMBOS: [(Dtype, Dtype); 5] = [
        (Dtype::F16, Dtype::F16),
        (Dtype::BF16, Dtype::BF16),
        (Dtype::F32, Dtype::F32),
        (Dtype::BF16, Dtype::F32),
        (Dtype::F16, Dtype::BF16),
    ];

    #[test]
    fn rows_are_invariant_small_shape() {
        if gpu().is_none() {
            return;
        }
        check_row_invariance(512, 384, 4, Codebook::Mul1, true, 128, &COMBOS);
        check_row_invariance(512, 384, 6, Codebook::Mul1, false, 128, &COMBOS[..2]);
        check_row_invariance(512, 384, 4, Codebook::Mcg, true, 128, &COMBOS[..1]);
        check_row_invariance(512, 384, 4, Codebook::Inst3, false, 128, &COMBOS[..1]);
        check_row_invariance(512, 384, 2, Codebook::Mul1, true, 128, &COMBOS[..1]);
        check_row_invariance(512, 384, 8, Codebook::Mul1, true, 128, &COMBOS[..1]);
    }

    #[test]
    fn rows_are_invariant_split_k() {
        if gpu().is_none() {
            return;
        }
        // K=4096 N=256: 8 K splits, partials summed in split order by the finish kernel
        check_row_invariance(4096, 256, 4, Codebook::Mul1, true, 128, &COMBOS);
        check_row_invariance(4096, 256, 6, Codebook::Mul1, false, 128, &COMBOS[..2]);
        check_row_invariance(2048, 128, 4, Codebook::Mul1, true, 128, &COMBOS[..2]);
    }

    #[test]
    fn rows_are_invariant_eight_warps() {
        if gpu().is_none() {
            return;
        }
        // N=8192: 64 column blocks, so 8 warps a block
        check_row_invariance(1024, 8192, 4, Codebook::Mul1, true, 128, &COMBOS[..3]);
        check_row_invariance(1024, 8192, 6, Codebook::Mul1, false, 128, &COMBOS[..1]);
    }

    // ---------------------------------------------------------------- prefill

    #[test]
    fn prefill_matches_reference() {
        let Some(st) = gpu() else { return };
        let mut worst = (0.0f64, 0.0f64);
        for (bits, cb, bias) in [
            (4usize, Codebook::Mul1, true),
            (6, Codebook::Mul1, false),
            (3, Codebook::Mul1, true),
            (4, Codebook::Mcg, false),
            (4, Codebook::Inst3, true),
            (2, Codebook::Mul1, false),
        ] {
            let (k, n) = (512, 384);
            let g = make_group(k, n, bits, cb, bias, 3 + bits as u64);
            let w = Exl3Weights::from_loaded(&g).unwrap();
            let ref_rows = 300;
            let x = xs(ref_rows, k, 21);
            for xd in [Dtype::F16, Dtype::BF16] {
                let (xr, xb) = encode(&x, xd);
                let want = forward(
                    &xr,
                    ref_rows,
                    &g.trellis,
                    g.kt,
                    g.nt,
                    g.group.bits,
                    &codebook(cb),
                    &g.suh,
                    &g.svh,
                    g.bias.as_deref(),
                );
                for m in [1usize, 5, 64, 300] {
                    let xdv = DevBuf::from_slice(&xb[..m * k * xd.size()]).unwrap();
                    let mut y = DevBuf::zeroed(m * n * xd.size()).unwrap();
                    let mut scratch = DevBuf::zeroed(w.prefill_scratch_bytes(m)).unwrap();
                    w.forward_prefill(&xdv, &mut y, m, xd, &mut scratch, &st).unwrap();
                    st.sync().unwrap();
                    let got = decode(&y.to_vec::<u8>().unwrap(), xd);
                    let (rel, ratio) =
                        compare(&got, &want[..m * n], m, n, xd, &format!("prefill {bits} bits {} m={m} {xd:?}", cb.name()));
                    worst = (worst.0.max(rel), worst.1.max(ratio));
                }
            }
        }
        eprintln!("worst prefill error: {:.3e} of row rms, {:.3} of the bound", worst.0, worst.1);
    }

    // ---------------------------------------------------------------- pinned prefill: row invariance

    /// (K, N) of every linear the Qwen3.8-27B prefill runs: MLP gate/up and down, the linear-attention qkv, z and out
    /// projections, the full-attention q (with its gate), k/v and o projections. The model-level test derives the same list
    /// from the loaded pack; this one needs no pack.
    const PREFILL_SHAPES: [(usize, usize); 7] =
        [(5120, 17408), (17408, 5120), (5120, 10240), (5120, 6144), (6144, 5120), (5120, 12288), (5120, 1024)];

    fn synthetic_for_sweep(k: usize, n: usize) -> Exl3Weights {
        let mut rng = Rng(11 + (k + n) as u64);
        let suh: Vec<u16> = (0..k).map(|_| f64_to_f16(0.5 + rng.uniform())).collect();
        let svh: Vec<u16> = (0..n).map(|_| f64_to_f16(0.5 + rng.uniform())).collect();
        synthetic_layer(&random_trellis(k * n * 4 / 16, 7), &suh, &svh, k, n, 4)
    }

    fn sweep_full() -> bool {
        std::env::var("REXL3_SWEEP_FULL").is_ok_and(|v| v != "0")
    }

    #[test]
    fn pinned_prefill_names_its_algorithm_and_runs_every_prefill_shape() {
        if gpu().is_none() {
            return;
        }
        assert!(prefill_pin_name().contains("algo 67") && prefill_pin_name().contains("split-K 1"), "{}", prefill_pin_name());
        prefill_init().unwrap();
        for (k, n) in PREFILL_SHAPES {
            for m in [1usize, 17, 128, 129, 4096] {
                prefill_pin_check(m, k, n).unwrap_or_else(|e| panic!("pinned GEMM refused {m} x {k} x {n}: {e}"));
            }
        }
    }

    /// The guarantee: for every prefill shape, row r of a call of m rows (m = 1..=4096 on the default grid, every m with
    /// REXL3_SWEEP_FULL=1), at any index in the call, has the bits of the same row in a call of 4,096 rows.
    #[test]
    fn pinned_prefill_rows_are_invariant_over_m() {
        let Some(st) = gpu() else { return };
        prefill_init().unwrap();
        let ms = sweep::grid(sweep_full());
        for (k, n) in PREFILL_SHAPES {
            let w = synthetic_for_sweep(k, n);
            let bad = sweep::row_invariance(&w, sweep::Path::Pinned, &ms, &st).unwrap();
            assert_eq!(bad, 0, "pinned prefill K={k} N={n}: {bad} (m, offset) cases differ from the 4096-row call");
        }
        eprintln!("pinned prefill: {} row counts x every prefill shape, row invariant", ms.len());
    }

    /// The negative control, so the sweep above is known to be able to fail: `cublasGemmEx` is not row invariant at small m
    /// (its pick switches to split-K and CUDA-core kernels below 17 rows on this cuBLAS), and the sweep must say so.
    #[test]
    fn the_sweep_catches_the_heuristic_gemm() {
        let Some(st) = gpu() else { return };
        prefill_init().unwrap();
        let w = synthetic_for_sweep(5120, 17408);
        let ms: Vec<usize> = (1..=16).collect();
        let bad = sweep::row_invariance(&w, sweep::Path::Legacy, &ms, &st).unwrap();
        assert!(bad > 0, "cublasGemmEx became row invariant at 1..=16 rows: the control no longer discriminates, revisit it");
    }

    /// The pinned GEMM gives the bits of the large-M `cublasGemmEx` result, which is what every prefill chunk of more than 128 rows
    /// got before Phase 4, at these sizes. NOT at every size: measured with `bench_pinned_vs_legacy_map` on cuBLAS 13.1.1 (a raw `cublasGemmEx` probe gave the same set on 13.1.0.3),
    /// `cublasGemmEx` leaves the large-M bits for k_proj/v_proj (K=5120, N=1024) at m = 193..=248 and 385..=391, where its heuristic
    /// picks another algorithm; every other shape agrees at every m in 129..=4096. The sizes below avoid those ranges (the model-level
    /// consequence is the Phase 4 bits-change set, see docs/EVAL_RESULTS.md).
    #[test]
    fn pinned_prefill_equals_the_legacy_gemm_outside_its_unstable_sizes() {
        let Some(st) = gpu() else { return };
        prefill_init().unwrap();
        let ms = [129usize, 130, 191, 192, 249, 250, 255, 256, 257, 300, 384, 392, 512, 777, 1000, 1024, 2048, 3000, 4095, 4096];
        for (k, n) in PREFILL_SHAPES {
            let w = synthetic_for_sweep(k, n);
            let differ = sweep::pinned_vs_legacy(&w, &ms, &st).unwrap();
            assert!(differ.is_empty(), "K={k} N={n}: the pinned GEMM differs from cublasGemmEx at m = {differ:?}");
        }
    }

    /// The pinned prefill GEMM at every MiniCPM shape: it runs (the algorithm exists at the shape, no workspace) and its rows are
    /// invariant over the call's row count, as for the 27B's: row r of a call of m rows (m = 1..=300 plus the tile edges up to 4096,
    /// every m with REXL3_SWEEP_FULL=1) has the bits of the same row in a call of 4,096 rows. The lm_head (a 4,096-row call is 4 GB of
    /// fp32 partials and 2 GB of output) is swept on the row counts a logits call can have (1..=129 and a few tile edges to 1024).
    #[test]
    fn minicpm_shapes_pinned_prefill_rows_are_invariant_over_m() {
        let Some(st) = gpu() else { return };
        prefill_init().unwrap();
        let full = sweep_full();
        for &(what, k, n) in &MINICPM_SHAPES {
            for m in [1usize, 17, 128, 129, 4096] {
                prefill_pin_check(m, k, n).unwrap_or_else(|e| panic!("pinned GEMM refused {m} x {k} x {n} ({what}): {e}"));
            }
            let ms: Vec<usize> =
                if n > 100_000 { (1..=129).chain([192, 255, 256, 257, 384, 512, 1024]).collect() } else { sweep::grid(full) };
            let w = synthetic_for_sweep(k, n);
            let bad = sweep::row_invariance(&w, sweep::Path::Pinned, &ms, &st).unwrap();
            assert_eq!(bad, 0, "pinned prefill {what} K={k} N={n}: {bad} (m, offset) cases differ from the 4096-row call");
            eprintln!("pinned prefill {what} ({k} x {n}): {} row counts, row invariant", ms.len());
        }
    }

    /// Informational and exhaustive (not part of the gate, which skips `bench`): every m in 129..=4096 at which the pinned GEMM
    /// and the legacy `cublasGemmEx` give different bits, per prefill shape, as ranges.
    #[test]
    #[ignore]
    fn bench_pinned_vs_legacy_map() {
        let Some(st) = gpu() else { return };
        prefill_init().unwrap();
        let ms: Vec<usize> = (129..=4096).collect();
        for (k, n) in PREFILL_SHAPES {
            let w = synthetic_for_sweep(k, n);
            let differ = sweep::pinned_vs_legacy(&w, &ms, &st).unwrap();
            let mut ranges: Vec<String> = Vec::new();
            let mut i = 0;
            while i < differ.len() {
                let mut j = i;
                while j + 1 < differ.len() && differ[j + 1] == differ[j] + 1 {
                    j += 1;
                }
                ranges.push(if i == j { format!("{}", differ[i]) } else { format!("{}-{}", differ[i], differ[j]) });
                i = j + 1;
            }
            eprintln!("K={k} N={n}: pinned != cublasGemmEx at {} of {} row counts: {}", differ.len(), ms.len(), ranges.join(" "));
        }
    }

    // ---------------------------------------------------------------- (c) the real pack

    fn qwen_dir() -> Option<PathBuf> {
        let p = PathBuf::from(std::env::var("REXL3_QWEN_DIR").ok()?);
        p.exists().then_some(p)
    }

    /// The reference restricted to a few 128-column blocks of a group (the output rotation is per block, so the
    /// other columns do not matter): sub-trellis, scales and bias of just those blocks.
    fn slice_blocks(g: &LoadedGroup, blocks: &[usize]) -> LoadedGroup {
        let tw = 8 * (g.group.bits.twice() as usize);
        let mut trellis = Vec::new();
        let mut svh = Vec::new();
        let mut bias = g.bias.as_ref().map(|_| Vec::new());
        for kk in 0..g.kt {
            for &b in blocks {
                let start = (kk * g.nt + b * 8) * tw;
                trellis.extend_from_slice(&g.trellis[start..start + 8 * tw]);
            }
        }
        for &b in blocks {
            svh.extend_from_slice(&g.svh[b * 128..(b + 1) * 128]);
            if let (Some(out), Some(src)) = (bias.as_mut(), g.bias.as_ref()) {
                out.extend_from_slice(&src[b * 128..(b + 1) * 128]);
            }
        }
        let mut grp = g.group.clone();
        grp.n = blocks.len() * 128;
        LoadedGroup { group: grp, trellis, kt: g.kt, nt: blocks.len() * 8, suh: g.suh.clone(), svh, bias }
    }

    /// Run one real group on the GPU (the whole layer) and check three column blocks of it against the reference.
    fn check_pack_group(ck: &Checkpoint, prefix: &str, st: &Stream, worst: &mut (f64, f64)) {
        let g = ck.load_group(prefix).unwrap();
        let (k, n) = (g.group.k, g.group.n);
        let w = Exl3Weights::from_loaded(&g).unwrap();
        let nb = n / 128;
        let mut blocks = vec![0usize, nb / 2, nb - 1];
        blocks.dedup();
        let sub = slice_blocks(&g, &blocks);
        let table = codebook(g.group.codebook);
        let rows = 16;
        let x = xs(rows, k, 5);
        let mut ws = poisoned(w.workspace_bytes(rows), st);
        for xd in [Dtype::F16, Dtype::BF16] {
            let (xr, xb) = encode(&x, xd);
            let want =
                forward(&xr, rows, &sub.trellis, sub.kt, sub.nt, sub.group.bits, &table, &sub.suh, &sub.svh, sub.bias.as_deref());
            for m in [1usize, 4, 16] {
                let out = run_decode(&w, &xb[..m * k * xd.size()], m, xd, xd, &mut ws, st);
                let full = decode(&out, xd);
                let mut got = Vec::new();
                for r in 0..m {
                    for &b in &blocks {
                        got.extend_from_slice(&full[r * n + b * 128..r * n + (b + 1) * 128]);
                    }
                }
                let what = format!("{prefix} ({} bits {}) m={m} {xd:?}", g.group.bits.as_f64(), g.group.codebook.name());
                let (rel, ratio) = compare(&got, &want[..m * blocks.len() * 128], m, blocks.len() * 128, xd, &what);
                *worst = (worst.0.max(rel), worst.1.max(ratio));
            }
        }
        eprintln!("{prefix}: k={k} n={n} {} bits {}: ok", g.group.bits.as_f64(), g.group.codebook.name());
    }

    #[test]
    fn real_pack_groups_match_reference() {
        let Some(st) = gpu() else { return };
        let Some(dir) = qwen_dir() else {
            eprintln!("REXL3_QWEN_DIR not set or missing: skipping the real-pack test");
            return;
        };
        let ck = scan(&dir, true).unwrap();
        let mut picked: Vec<String> = Vec::new();
        for suffix in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"] {
            let found = ck.groups.keys().filter(|p| p.ends_with(suffix)).find(|p| p.contains("layers.3.") || suffix == "lm_head");
            let found = found.or_else(|| ck.groups.keys().find(|p| p.ends_with(suffix)));
            if let Some(p) = found {
                picked.push(p.clone());
            }
        }
        for extra in ["in_proj_qkv", "in_proj_z", "out_proj", "linear_fc1"] {
            if let Some(p) = ck.groups.keys().find(|p| p.ends_with(extra)) {
                picked.push(p.clone());
            }
        }
        assert!(picked.len() >= 6, "found only {picked:?}");
        let mut worst = (0.0f64, 0.0f64);
        for prefix in &picked {
            check_pack_group(&ck, prefix, &st, &mut worst);
        }
        eprintln!("worst real-pack error: {:.3e} of row rms, {:.3} of the bound", worst.0, worst.1);
    }

    /// Every group of the DFlash2 draft pack (REXL3_DRAFT_DIR, default the r0b0tlab pack when it exists).
    #[test]
    fn draft_pack_groups_match_reference() {
        let Some(st) = gpu() else { return };
        let dir = std::env::var("REXL3_DRAFT_DIR").map(PathBuf::from).ok().filter(|p| p.exists());
        let Some(dir) = dir else {
            eprintln!("REXL3_DRAFT_DIR not set or missing: skipping the draft-pack test");
            return;
        };
        let ck = scan(&dir, true).unwrap();
        let mut worst = (0.0f64, 0.0f64);
        let names: Vec<String> = ck.groups.keys().cloned().collect();
        assert!(!names.is_empty());
        for prefix in &names {
            check_pack_group(&ck, prefix, &st, &mut worst);
        }
        eprintln!("worst draft-pack error over {} groups: {:.3e} of row rms, {:.3} of the bound", names.len(), worst.0, worst.1);
    }

    // ---------------------------------------------------------------- (d) benchmarks (ignored)

    mod ev {
        use core::ffi::c_void;
        extern "C" {
            pub fn cudaEventCreate(e: *mut *mut c_void) -> i32;
            pub fn cudaEventDestroy(e: *mut c_void) -> i32;
            pub fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> i32;
            pub fn cudaEventSynchronize(e: *mut c_void) -> i32;
            pub fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> i32;
        }
    }

    /// Mean milliseconds of `f` over `iters` back-to-back runs on `st` (cudaEvents), the best of three trials (the box
    /// is shared, so single trials are noisy).
    fn time_ms(st: &Stream, iters: usize, mut f: impl FnMut(usize)) -> f64 {
        (0..3).map(|_| time_once(st, iters, &mut f)).fold(f64::INFINITY, f64::min)
    }

    fn time_once(st: &Stream, iters: usize, f: &mut impl FnMut(usize)) -> f64 {
        unsafe {
            let (mut a, mut b) = (core::ptr::null_mut(), core::ptr::null_mut());
            ev::cudaEventCreate(&mut a);
            ev::cudaEventCreate(&mut b);
            for i in 0..iters.min(5) {
                f(i);
            }
            st.sync().unwrap();
            ev::cudaEventRecord(a, st.raw());
            for i in 0..iters {
                f(i);
            }
            ev::cudaEventRecord(b, st.raw());
            ev::cudaEventSynchronize(b);
            let mut ms = 0f32;
            ev::cudaEventElapsedTime(&mut ms, a, b);
            ev::cudaEventDestroy(a);
            ev::cudaEventDestroy(b);
            ms as f64 / iters as f64
        }
    }

    fn random_trellis(len: usize, seed: u64) -> Vec<i16> {
        let mut r = Rng(seed);
        (0..len).map(|_| r.next() as i16).collect()
    }

    fn synthetic_layer(host: &[i16], suh: &[u16], svh: &[u16], k: usize, n: usize, bits: usize) -> Exl3Weights {
        Exl3Weights {
            trellis: DevBuf::from_slice(host).unwrap(),
            suh: DevBuf::from_slice(suh).unwrap(),
            svh: DevBuf::from_slice(svh).unwrap(),
            bias: None,
            k,
            n,
            bits,
            codebook: Codebook::Mul1,
        }
    }

    /// Decode-path time per call (ms) for each M, with `rotate_bytes` of distinct weight copies in rotation (set it
    /// above the L2 for a cold run, 0 for one hot copy).
    fn bench_shape(st: &Stream, k: usize, n: usize, bits: usize, ms: &[usize], rotate_bytes: usize) -> Vec<(usize, f64)> {
        let tr_words = k * n * bits / 16;
        let copies = (rotate_bytes / (tr_words * 2).max(1)).clamp(0, 96) + 1;
        let host = random_trellis(tr_words, 1);
        let mut rng = Rng(5);
        let suh: Vec<u16> = (0..k).map(|_| f64_to_f16(0.5 + rng.uniform())).collect();
        let svh: Vec<u16> = (0..n).map(|_| f64_to_f16(0.5 + rng.uniform())).collect();
        let layers: Vec<Exl3Weights> = (0..copies).map(|_| synthetic_layer(&host, &suh, &svh, k, n, bits)).collect();
        let mut ws = poisoned(layers[0].workspace_bytes(128), st);
        let xh_host: Vec<u16> = (0..128 * k).map(|_| f64_to_f16(rng.normal())).collect();
        let x = DevBuf::from_slice(&xh_host).unwrap();
        let mut y = DevBuf::zeroed(128 * n * 2).unwrap();
        ms.iter()
            .map(|&m| {
                let iters = (copies * 4).clamp(40, 400);
                let t = time_ms(st, iters, |i| {
                    layers[i % copies].forward(&x, &mut y, m, Dtype::F16, Dtype::F16, &mut ws, st).unwrap()
                });
                (m, t)
            })
            .collect()
    }

    fn report(label: &str, k: usize, n: usize, bits: usize, res: &[(usize, f64)]) {
        let bytes = (k * n * bits / 8 + 2 * k + 2 * n) as f64;
        let mut line = format!("{label:<52} k={k:<6} n={n:<6} {bits}b {:>6.1} MB |", bytes / 1e6);
        for (m, t) in res {
            line += &format!(" M={m}: {:>7.1}us {:>6.1}GB/s |", t * 1e3, bytes / (t * 1e-3) / 1e9);
        }
        if res.len() > 1 {
            line += &format!(" t(M=16)/t(M=1)={:.3}", res.iter().find(|r| r.0 == 16).map_or(f64::NAN, |r| r.1) / res[0].1);
        }
        eprintln!("{line}");
    }

    /// Output hashes of the decode path for the pack's layer shapes (random trellis words) and a spread of row counts:
    /// a kernel change that must keep every bit diffs this file against one written by the old build.
    ///
    ///     REXL3_GOLDEN_OUT=golden.txt REXL3_QWEN_DIR=... cargo test --release -p gb-cuda --lib golden_decode_hashes -- --ignored
    #[test]
    #[ignore]
    fn golden_decode_hashes() {
        use std::io::Write;
        let Some(st) = gpu() else { return };
        let path = std::env::var("REXL3_GOLDEN_OUT").expect("REXL3_GOLDEN_OUT");
        let mut f = std::fs::File::create(path).unwrap();
        // (k, n, bits, codebook, activation dtypes)
        type Case = (usize, usize, usize, Codebook, Vec<(Dtype, Dtype)>);
        let mut cases: Vec<Case> = Vec::new();
        for bits in 1..=8usize {
            for cb in Codebook::ALL {
                cases.push((512, 384, bits, cb, vec![(Dtype::F16, Dtype::F16), (Dtype::BF16, Dtype::BF16)]));
            }
        }
        cases.push((4096, 256, 4, Codebook::Mul1, vec![(Dtype::BF16, Dtype::BF16)]));
        cases.push((1024, 8192, 4, Codebook::Mul1, vec![(Dtype::BF16, Dtype::BF16)]));
        if let Some(dir) = qwen_dir() {
            let ck = scan(&dir, true).unwrap();
            let mut seen = std::collections::BTreeSet::new();
            for g in ck.groups.values().filter(|g| g.bits.twice() % 2 == 0) {
                let key = (g.k, g.n, g.bits.twice() as usize / 2, g.codebook.name().to_string());
                if seen.insert(key) {
                    cases.push((g.k, g.n, g.bits.twice() as usize / 2, g.codebook, vec![(Dtype::BF16, Dtype::BF16)]));
                }
            }
        }
        // MiniCPM-V-4.7's shapes, appended after the pack's so every line above is a prefix of the file as it was: widths 3..8 of every
        // layer of the text backbone and the draft (the lm_head at its shipping width 6 only), synthetic trellis words
        for &(_, k, n) in &MINICPM_SHAPES {
            let widths: &[usize] = if n > 100_000 { &[6] } else { &[3, 4, 5, 6, 7, 8] };
            for &bits in widths {
                cases.push((k, n, bits, Codebook::Mul1, vec![(Dtype::BF16, Dtype::BF16)]));
            }
        }
        let fnv = |b: &[u8]| b.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3));
        for (k, n, bits, cb, combos) in cases {
            let g = make_group(k, n, bits, cb, n % 256 == 0, 31 + (k * 7 + n + bits) as u64);
            let w = Exl3Weights::from_loaded(&g).unwrap();
            drop(g);
            let mut ws = poisoned(w.workspace_bytes(128), &st);
            let x = xs(128, k, 5 + n as u64);
            for (xd, yd) in combos {
                let (_, xb) = encode(&x, xd);
                for m in [1usize, 2, 8, 15, 16, 17, 24, 31, 32, 33, 40, 48, 49, 63, 64, 65, 80, 96, 127, 128] {
                    let y = run_decode(&w, &xb[..m * k * xd.size()], m, xd, yd, &mut ws, &st);
                    writeln!(f, "k={k} n={n} bits={bits} {} {xd:?}->{yd:?} M={m} {:016x}", cb.name(), fnv(&y)).unwrap();
                }
            }
        }
    }

    /// Decode throughput of every MiniCPM-V-4.7 shape at 4 and 6 bits and 1, 4, 8, 16, 64 rows, cold (the weights rotate through at
    /// least 192 MB of distinct copies, as a layer's weights do in a real step); the lock held alone. Run it as
    /// `flock /tmp/rexl3-gpu.lock cargo test --release -p gb-cuda --lib bench_minicpm_decode_gbps -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_minicpm_decode_gbps() {
        let Some(st) = gpu() else { return };
        eprintln!(
            "cold = weights rotated through >= 192 MB of distinct copies (best of 3 trials); the GB10's roofline is 273 GB/s"
        );
        for &(what, k, n) in &MINICPM_SHAPES {
            let (sk, wk) = decode_plan(k, n).unwrap();
            for bits in [4usize, 6] {
                let rotate = if k * n * bits / 8 > (192 << 20) { 0 } else { 192 << 20 };
                let res = bench_shape(&st, k, n, bits, &[1, 4, 8, 16, 64], rotate);
                report(&format!("{what} plan sk={sk} wk={wk}"), k, n, bits, &res);
            }
        }
    }

    #[test]
    #[ignore]
    fn bench_decode_gbps() {
        let Some(st) = gpu() else { return };
        let ms: Vec<usize> = match std::env::var("REXL3_BENCH_MS") {
            Ok(v) => v.split(',').map(|x| x.trim().parse().unwrap()).collect(),
            Err(_) => vec![1, 2, 8, 16, 32],
        };
        // (category, first prefix with the shape, how many groups share it, k, n, bits)
        let mut shapes: Vec<(&str, String, usize, usize, usize, usize)> = Vec::new();
        if let Some(dir) = qwen_dir() {
            let ck = scan(&dir, true).unwrap();
            let mut seen = std::collections::BTreeMap::new();
            for (p, g) in &ck.groups {
                let bits = g.bits.twice() as usize / 2;
                let cat = if p.contains("lm_head") {
                    "head"
                } else if p.contains("mtp") {
                    "mtp"
                } else if p.contains("visual") || p.contains("vision") || bits == 6 {
                    "vision"
                } else {
                    "decoder"
                };
                let e = seen.entry((cat, g.k, g.n, bits)).or_insert_with(|| (p.clone(), 0usize));
                e.1 += 1;
            }
            for ((cat, k, n, bits), (p, count)) in seen {
                shapes.push((cat, p, count, k, n, bits));
            }
            shapes.sort_by_key(|s| (s.0 != "decoder", s.0 != "head", s.0.to_string(), std::cmp::Reverse(s.3 * s.4)));
        } else {
            for (k, n) in [(5120, 17408), (17408, 5120), (5120, 5120), (5120, 12288), (6144, 5120), (5120, 1024)] {
                shapes.push(("synthetic", "synthetic".into(), 1, k, n, 4));
            }
        }
        eprintln!("cold = weights rotated through >= 192 MB of distinct copies (best of 3 trials); hot = one copy, mostly L2");
        let filter = std::env::var("REXL3_BENCH_FILTER").unwrap_or_default();
        for (cat, name, count, k, n, bits) in &shapes {
            if !name.contains(&filter) {
                continue;
            }
            let short = name.rsplit_once("layers.").map_or(name.as_str(), |x| x.1);
            let label = format!("[{cat}] x{count} {short}");
            let rotate = if k * n * bits / 8 > (192 << 20) { 0 } else { 192 << 20 };
            let cold = bench_shape(&st, *k, *n, *bits, &ms, rotate);
            report(&format!("cold {label}"), *k, *n, *bits, &cold);
            if k * n * bits / 8 < (64 << 20) {
                let hot = bench_shape(&st, *k, *n, *bits, &ms[..1], 0);
                report(&format!("hot  {label}"), *k, *n, *bits, &hot);
            }
        }
    }

    /// A layer whose three buffers are views in `arena` (device memory, or mapped pinned host memory for the zero-copy test).
    fn arena_layer(
        arena: &crate::arena::Arena,
        host: &[i16],
        suh: &[u16],
        svh: &[u16],
        k: usize,
        n: usize,
        bits: usize,
    ) -> Exl3Weights {
        let put = |b: &mut DevBuf, bytes: &[u8]| {
            // the arena's blocks may be host memory: copy with a plain H2D copy, which works for both
            b.upload(bytes).unwrap();
        };
        let (mut trellis, mut su, mut sv) =
            (arena.alloc(host.len() * 2).unwrap(), arena.alloc(suh.len() * 2).unwrap(), arena.alloc(svh.len() * 2).unwrap());
        put(&mut trellis, unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 2) });
        put(&mut su, unsafe { std::slice::from_raw_parts(suh.as_ptr() as *const u8, suh.len() * 2) });
        put(&mut sv, unsafe { std::slice::from_raw_parts(svh.as_ptr() as *const u8, svh.len() * 2) });
        Exl3Weights { trellis, suh: su, svh: sv, bias: None, k, n, bits, codebook: Codebook::Mul1 }
    }

    /// Time of the decode path per call (ms) with the layer in `arena`, as in [`bench_shape`]: `copies` distinct layers in
    /// rotation so the weights are streamed from memory, not served from the L2.
    fn bench_shape_in(
        st: &Stream,
        arena: &crate::arena::Arena,
        k: usize,
        n: usize,
        bits: usize,
        ms: &[usize],
        copies: usize,
    ) -> Vec<(usize, f64)> {
        let tr_words = k * n * bits / 16;
        let host = random_trellis(tr_words, 1);
        let mut rng = Rng(5);
        let suh: Vec<u16> = (0..k).map(|_| f64_to_f16(0.5 + rng.uniform())).collect();
        let svh: Vec<u16> = (0..n).map(|_| f64_to_f16(0.5 + rng.uniform())).collect();
        let layers: Vec<Exl3Weights> = (0..copies).map(|_| arena_layer(arena, &host, &suh, &svh, k, n, bits)).collect();
        let mut ws = poisoned(layers[0].workspace_bytes(128), st);
        let xh_host: Vec<u16> = (0..128 * k).map(|_| f64_to_f16(rng.normal())).collect();
        let x = DevBuf::from_slice(&xh_host).unwrap();
        let mut y = DevBuf::zeroed(128 * n * 2).unwrap();
        ms.iter()
            .map(|&m| {
                let iters = (copies * 4).clamp(40, 400);
                let t = time_ms(st, iters, |i| {
                    layers[i % copies].forward(&x, &mut y, m, Dtype::F16, Dtype::F16, &mut ws, st).unwrap()
                });
                (m, t)
            })
            .collect()
    }

    /// Stage B of the boot plan (and a probe of a third option): does decode lose bandwidth when the weights are read from
    /// mapped pinned host memory (zero copy: no device allocation, no upload), or from ordinary pageable host memory
    /// through the system's address translation, instead of device memory? Same layers, same kernel, rotated through at
    /// least 192 MB of distinct copies so every call streams from memory; run in rounds (device, mapped, pageable) so a
    /// drifting clock or a busy neighbour shows up as spread between rounds.
    ///
    ///     flock /tmp/rexl3-gpu.lock cargo test --release -p gb-cuda --lib bench_decode_gbps_mapped -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_decode_gbps_mapped() {
        use crate::arena::BlockKind;
        let Some(st) = gpu() else { return };
        let ms = [1usize, 8, 16, 32];
        let rounds: usize = std::env::var("REXL3_BENCH_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        let kinds = [BlockKind::Device, BlockKind::MappedHost, BlockKind::PageableHost];
        // the decoder's layer shapes at 4 bits and the lm_head at 6
        for &(k, n, bits) in
            &[(5120usize, 17408usize, 4usize), (17408, 5120, 4), (5120, 5120, 4), (5120, 12288, 4), (5120, 248320, 6)]
        {
            let bytes = k * n * bits / 8;
            let copies = if bytes > (192 << 20) { 1 } else { (192 << 20) / bytes + 1 };
            let mut res: Vec<Vec<Vec<(usize, f64)>>> = vec![Vec::new(); kinds.len()];
            for _ in 0..rounds {
                for (i, &kind) in kinds.iter().enumerate() {
                    let a = crate::arena::Arena::with_kind(1 << 30, kind);
                    res[i].push(bench_shape_in(&st, &a, k, n, bits, &ms, copies));
                    drop(a);
                }
            }
            let wbytes = (k * n * bits / 8 + 2 * k + 2 * n) as f64;
            for (mi, &m) in ms.iter().enumerate() {
                let best = |v: &[Vec<(usize, f64)>]| v.iter().map(|r| r[mi].1).fold(f64::INFINITY, f64::min);
                let spread = |v: &[Vec<(usize, f64)>]| {
                    let (lo, hi) = (best(v), v.iter().map(|r| r[mi].1).fold(0.0, f64::max));
                    (hi - lo) / lo * 100.0
                };
                let d = best(&res[0]);
                let mut line = format!(
                    "k={k:<6} n={n:<6} {bits}b M={m:<3} device {:>7.1} us {:>6.1} GB/s",
                    d * 1e3,
                    wbytes / (d * 1e-3) / 1e9
                );
                for (i, name) in [(1usize, "mapped"), (2, "pageable")] {
                    let t = best(&res[i]);
                    line += &format!(" | {name} {:>7.1} us {:>6.1} GB/s time x{:.3}", t * 1e3, wbytes / (t * 1e-3) / 1e9, t / d);
                }
                line += &format!(" | round spread {:.1}% / {:.1}% / {:.1}%", spread(&res[0]), spread(&res[1]), spread(&res[2]));
                eprintln!("{line}");
            }
        }
    }

    #[test]
    #[ignore]
    fn bench_prefill_tflops() {
        let Some(st) = gpu() else { return };
        let (k, n, bits) = (5120usize, 17408usize, 4usize);
        let host = random_trellis(k * n * bits / 16, 2);
        let mut rng = Rng(5);
        let suh: Vec<u16> = (0..k).map(|_| f64_to_f16(0.5 + rng.uniform())).collect();
        let svh: Vec<u16> = (0..n).map(|_| f64_to_f16(0.5 + rng.uniform())).collect();
        let w = synthetic_layer(&host, &suh, &svh, k, n, bits);
        let mut wq = DevBuf::zeroed(k * n * 2).unwrap();
        let recon = time_ms(&st, 10, |_| w.reconstruct(&mut wq, &st).unwrap());
        for m in [512usize, 1024, 2048, 4096] {
            let xh_host: Vec<u16> = (0..m * k).map(|_| f64_to_f16(rng.normal())).collect();
            let x = DevBuf::from_slice(&xh_host).unwrap();
            let mut y = DevBuf::zeroed(m * n * 2).unwrap();
            let mut scratch = DevBuf::zeroed(w.prefill_scratch_bytes(m)).unwrap();
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            let tf = |ms: f64| flops / (ms * 1e-3) / 1e12;
            let full = time_ms(&st, 10, |_| w.forward_prefill(&x, &mut y, m, Dtype::F16, &mut scratch, &st).unwrap());
            let mut xh = DevBuf::zeroed(m * k * 2).unwrap();
            let mut c = DevBuf::zeroed(m * n * 4).unwrap();
            let rest =
                time_ms(&st, 10, |_| w.prefill_with_w(&x, Dtype::F16, &wq, &mut xh, &mut c, &mut y, Dtype::F16, m, &st).unwrap());
            eprintln!(
                "prefill {m}x{k}x{n} 4-bit: reconstruct {recon:.2} ms, rotate+GEMM+rotate {rest:.2} ms ({:.1} TFLOPS), end to end {full:.2} ms ({:.1} TFLOPS)",
                tf(rest),
                tf(full)
            );
        }
    }

    /// The P4-TAIL ceiling probe (`docs/TAIL_FLOOR_CEILING.md`): what one `W_q` reconstruct costs at every EXL3 matrix
    /// shape of the target pack (Qwen3.8-27B EXL3 4.00 bpw: 64 layers, 400 matrices a prefill chunk runs), each call on a
    /// distinct trellis copy rotated through at least ~1 GiB so every call streams from memory the way the real path does
    /// with its 12 GB working set; then the pinned prefill at tail and full row counts, so the reconstruct's share of a
    /// chunk is `total - reconstruct`. A measurement without assertions.
    ///
    ///     flock /tmp/rexl3-gpu.lock cargo test --release -p gb-cuda --lib bench_tail_floor -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_tail_floor_reconstruct() {
        let Some(st) = gpu() else { return };
        // The pack's shapes with their per-chunk multiplicities (quantization_config.json): 48 linear-attention layers
        // (in_proj_qkv, in_proj_z, out_proj, gate, up, down) and 16 full-attention layers (q, k, v, o, gate, up, down).
        let shapes = [
            (5120usize, 17408usize, 128usize), // mlp gate + up, both layer kinds
            (17408usize, 5120usize, 64usize),  // mlp down
            (5120usize, 12288usize, 16usize),  // self_attn q_proj
            (5120usize, 10240usize, 48usize),  // linear_attn in_proj_qkv
            (5120usize, 6144usize, 48usize),   // linear_attn in_proj_z
            (6144usize, 5120usize, 64usize),   // out_proj, both layer kinds
            (5120usize, 1024usize, 32usize),   // self_attn k_proj + v_proj
        ];
        const BITS: usize = 4;
        let (mut chunk_ms, mut chunk_gb) = (0f64, 0f64);
        let mut big_recon = 0f64;
        eprintln!("| k x n | mats/chunk | ms/call | GB/s | ms/chunk |");
        eprintln!("| --- | ---: | ---: | ---: | ---: |");
        for &(k, n, count) in &shapes {
            let tr_words = k * n * BITS / 16;
            let copies = ((1 << 30) / (tr_words * 2)).clamp(1, 64);
            let host = random_trellis(tr_words, 7);
            let mut rng = Rng(11);
            let suh: Vec<u16> = (0..k).map(|_| f64_to_f16(0.5 + rng.uniform())).collect();
            let svh: Vec<u16> = (0..n).map(|_| f64_to_f16(0.5 + rng.uniform())).collect();
            let layers: Vec<Exl3Weights> = (0..copies).map(|_| synthetic_layer(&host, &suh, &svh, k, n, BITS)).collect();
            let mut wq = DevBuf::zeroed(k * n * 2).unwrap();
            let iters = 3 * copies;
            let per_call = time_ms(&st, iters, |i| layers[i % copies].reconstruct(&mut wq, &st).unwrap());
            let gb = (tr_words * 2 + k * n * 2) as f64 / 1e9;
            if k == 5120 && n == 17408 {
                big_recon = per_call;
            }
            chunk_ms += count as f64 * per_call;
            chunk_gb += count as f64 * gb;
            let chunk_shape_ms = count as f64 * per_call;
            eprintln!("| {k}x{n} | {count} | {per_call:.4} | {:.0} | {chunk_shape_ms:.2} |", gb / (per_call * 1e-3));
        }
        eprintln!(
            "reconstruct per chunk (400 matrices): {chunk_ms:.1} ms, {chunk_gb:.1} GB moved ({:.0} GB/s effective); \
             a 32,768-token prompt at chunk 4,096 pays it 8 times: {:.0} ms",
            chunk_gb / (chunk_ms * 1e-3),
            8.0 * chunk_ms
        );

        // The pinned prefill at the big MLP shape with and without the reconstruct in the same call: what a persisted W_q
        // would save per layer at tail and full row counts (`rest` = rotate + pinned GEMM + rotate).
        let (k, n) = (5120usize, 17408usize);
        let host = random_trellis(k * n * BITS / 16, 13);
        let mut rng = Rng(17);
        let suh: Vec<u16> = (0..k).map(|_| f64_to_f16(0.5 + rng.uniform())).collect();
        let svh: Vec<u16> = (0..n).map(|_| f64_to_f16(0.5 + rng.uniform())).collect();
        let w = synthetic_layer(&host, &suh, &svh, k, n, BITS);
        for m in [16usize, 64, 128, 512, 4096] {
            let xb: Vec<u16> = (0..m * k).map(|_| bf16_bits(rng.normal() as f32)).collect();
            let x = DevBuf::from_slice(&xb).unwrap();
            let mut y = DevBuf::zeroed(m * n * 2).unwrap();
            let mut scratch = DevBuf::zeroed(w.prefill_scratch_bytes(m)).unwrap();
            let total = time_ms(&st, 10, |_| w.forward_prefill_pinned(&x, &mut y, m, Dtype::BF16, &mut scratch, &st).unwrap());
            eprintln!(
                "pinned prefill {m}x{k}x{n}: {total:.3} ms total, reconstruct {big_recon:.3} ms, rotate+GEMM+rotate {:.3} ms",
                total - big_recon
            );
        }

        // The decode kernel on the same shape and rows (the decode-into-GEMM candidate): it reads the trellis directly, no
        // W_q at all, so its floor is what a tail chunk could cost if its bits were allowed to move to this path.
        let mut ws = DevBuf::zeroed(w.workspace_bytes(128)).unwrap();
        for m in [1usize, 16, 128] {
            let xb: Vec<u16> = (0..m * k).map(|_| bf16_bits(rng.normal() as f32)).collect();
            let x = DevBuf::from_slice(&xb).unwrap();
            let mut y = DevBuf::zeroed(m * n * 2).unwrap();
            let d = time_ms(&st, 10, |_| w.forward(&x, &mut y, m, Dtype::BF16, Dtype::BF16, &mut ws, &st).unwrap());
            eprintln!("decode kernel {m}x{k}x{n}: {d:.3} ms (trellis-direct, no W_q)");
        }
    }
}
