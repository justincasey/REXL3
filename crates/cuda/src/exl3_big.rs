//! The EXL3 linear layer for LARGE row counts with the DECODE path's bits (`csrc/exl3_big.cu`; the design and the bit-identity argument are in
//! its header). [`Exl3Weights::forward_big`] takes any number of rows (a prefill chunk) and gives each row exactly the output
//! [`Exl3Weights::forward`] gives it alone or in any window of up to 128 rows: no second arithmetic, so a prompt chunk of any size leaves the
//! bits it would leave decoded one token at a time, and the chunk grid cannot change a bit.
//!
//! The workspace is the rotated input in fragment order only (`big_workspace_bytes`: `ceil(m / 16) * 16 * K * 2` bytes); there are no
//! split-K partials in global memory.
//!
//! # Launch API and contract (for any family with EXL3 linears)
//!
//! ```text
//! let bytes = w.big_workspace_bytes(rows);                       // price it for the memory plan; one workspace serves every layer on a stream
//! w.forward_big(&x, &mut y, rows, Dtype::F16, Dtype::F32, &mut ws, &st)?;   // rows: 1..=65,535; same arguments as `forward` (any layer width and bit width)
//! ```
//!
//! Nothing is model-specific: K and N come from the layer (multiples of 128), the bit width (1..=8), the codebook and the optional bias from
//! its weights, the dtypes from the arguments. Row invariance: a row's output is the bits [`Exl3Weights::forward`] gives that row alone or in any
//! window of up to 128 rows, whatever the number of rows, the row's position or the other rows, because the K split (`big_plan` = the decode
//! kernel's `plan_for`), every warp's k range and the warp-order then split-order sums are the decode kernel's own, the splits are summed in shared
//! memory by the block that owns the (64-row group, 128-column strip) in the same order, and nothing is atomic. Tests: `crates/cuda/tests/exl3_big.rs`
//! (every width x codebook x dtype, the ten GLM shapes, call independence, plan equality), mutations in `tools/mutation_check_glm_prefill.py`, golden
//! `testdata/gpu/golden_exl3_big.txt`.

use crate::exl3_linear::{Dtype, Exl3Weights};
use crate::rt::{CudaError, DevBuf, Stream};

#[cfg(not(gb_no_cuda))]
mod ffi {
    use core::ffi::c_void;
    extern "C" {
        pub fn gb_exl3_big_workspace_bytes(m: i32, k: i32, n: i32, bits: i32) -> usize;
        pub fn gb_exl3_big_plan(k: i32, n: i32, out: *mut i32) -> i32;
        pub fn gb_exl3_big_set_second_form(v: i32) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_exl3_linear_big(
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
    }
}

const INVALID: CudaError = CudaError(1);

#[cfg(not(gb_no_cuda))]
fn codebook_code(cb: gb_exl3::Codebook) -> i32 {
    match cb {
        gb_exl3::Codebook::Inst3 => 0,
        gb_exl3::Codebook::Mcg => 1,
        gb_exl3::Codebook::Mul1 => 2,
    }
}

/// Bytes of workspace for `m` rows of a K x N layer (0 for a shape the kernels do not take).
pub fn big_workspace_bytes(m: usize, k: usize, n: usize, bits: usize) -> usize {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        ffi::gb_exl3_big_workspace_bytes(m as i32, k as i32, n as i32, bits as i32)
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (m, k, n, bits);
        0
    }
}

/// The (K splits, warps a block) the decode path uses for a K x N layer (and so does [`Exl3Weights::forward_big`]).
pub fn big_plan(k: usize, n: usize) -> Result<(usize, usize), CudaError> {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        let mut out = [0i32; 2];
        CudaError::check(ffi::gb_exl3_big_plan(k as i32, n as i32, out.as_mut_ptr()))?;
        Ok((out[0] as usize, out[1] as usize))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (k, n);
        Err(CudaError(-1))
    }
}

/// Forces the large-row linear's form for the Flash-Next shared expert's two shapes (2,560 x 640 and 640 x 2,560, the MUL1 codebook at 4 to 8 bits, 1,024
/// rows or more) for the calls that follow: `Some(true)` the second form (the default: a warp owns 16 columns and runs every chain), `Some(false)` the first
/// (each warp a K quarter of all 128 columns), `None` the `REXL3_BIG_SECOND_FORM` environment again. Both give the same bits; only the time differs.
pub fn set_second_form(v: Option<bool>) {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        ffi::gb_exl3_big_set_second_form(v.map_or(-1, i32::from));
    }
    #[cfg(gb_no_cuda)]
    let _ = v;
}

impl Exl3Weights {
    /// Workspace for `m` rows of [`Exl3Weights::forward_big`].
    pub fn big_workspace_bytes(&self, m: usize) -> usize {
        big_workspace_bytes(m, self.k, self.n, self.bits)
    }

    /// y [m, n] = layer(x [m, k]) for any m >= 1, each row bit for bit what the decode path ([`Exl3Weights::forward`]) gives it. Asynchronous
    /// on `st`; `ws` needs `big_workspace_bytes(m)` bytes, its contents on entry do not matter.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_big(
        &self,
        x: &DevBuf,
        y: &mut DevBuf,
        m: usize,
        x_dtype: Dtype,
        y_dtype: Dtype,
        ws: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if m == 0
            || m > i32::MAX as usize / 128
            || x.bytes() < m * self.k * x_dtype.size()
            || y.bytes() < m * self.n * y_dtype.size()
            || ws.bytes() < self.big_workspace_bytes(m)
        {
            return Err(INVALID);
        }
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::gb_exl3_linear_big(
                x.ptr(),
                x_dtype as i32,
                self.trellis.ptr(),
                self.suh.ptr(),
                self.svh.ptr(),
                self.bias.as_ref().map_or(core::ptr::null(), |b| b.ptr() as *const _),
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
}
