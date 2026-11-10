//! HCX: fused multi-HyperConnection pre-mix (see `csrc/hcx.cu`). Same contract as `mhc::hc_pre` / `mhc::hc_head`.
//!
//! Decode fast path: `gb_hcx_pre` / `gb_hcx_pre_z` unconditionally run the bit-exact fast `k1b`/`k2c` kernels
//! (bit-identical to the reference kernels; see the GPU tests in `tests/hcx_perf.rs`). `gb_hcx_pre_ref` /
//! `gb_hcx_pre_z_ref` call the reference `k1_kernel`/`k2_kernel` directly, for those tests' A/B comparison.
#![allow(clippy::too_many_arguments)]

use crate::fn_mhc::Mixer;
use crate::rt::{CudaError, DevBuf, Stream};

/// Refuses a buffer smaller than what a launch touches, before anything is queued (`CudaError(1)`, as an invalid value).
fn need(b: &DevBuf, bytes: usize, what: &str) -> Result<(), CudaError> {
    if b.bytes() < bytes {
        eprintln!("{what}: buffer of {} bytes, the launch touches {bytes}", b.bytes());
        return Err(CudaError(1));
    }
    Ok(())
}

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_hcx_scratch_bytes(hc: i32, rank: i32, ninj: i32) -> usize;
        pub fn gb_hcx_pre_ref(
            streams: *const c_void, hc_norm_w: *const c_void, down_w: *const c_void, up_w: *const c_void,
            inject_w: *const c_void, y_out: *mut c_void, inj_out: *mut f32, scratch: *mut c_void,
            hidden_size: u32, hc_mult: u32, rank: u32, eps: f32, num_tokens: u32, stream: *mut c_void,
        ) -> i32;
        pub fn gb_hcx_pre_z(
            streams: *const c_void, hc_norm_w: *const c_void, down_w: *const c_void, up_w: *const c_void,
            inject_w: *const c_void, y_out: *mut c_void, inj_out: *mut f32, scratch: *mut c_void,
            z_out: *mut f32, hidden_size: u32, hc_mult: u32, rank: u32, eps: f32, num_tokens: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_hcx_pre_z_ref(
            streams: *const c_void, hc_norm_w: *const c_void, down_w: *const c_void, up_w: *const c_void,
            inject_w: *const c_void, y_out: *mut c_void, inj_out: *mut f32, scratch: *mut c_void,
            z_out: *mut f32, hidden_size: u32, hc_mult: u32, rank: u32, eps: f32, num_tokens: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_hcx_pre(
            streams: *const c_void, hc_norm_w: *const c_void, down_w: *const c_void, up_w: *const c_void,
            inject_w: *const c_void, y_out: *mut c_void, inj_out: *mut f32, scratch: *mut c_void,
            hidden_size: u32, hc_mult: u32, rank: u32, eps: f32, num_tokens: u32, stream: *mut c_void,
        ) -> i32;
        pub fn gb_hcg_scratch_bytes(rows: i32, h: i32, hc: i32, rank: i32) -> usize;
        pub fn gb_hcg_pre_post(
            streams: *mut c_void, post_block: *const c_void, post_inj: *const f32, norm_w: *const c_void, down_w: *const c_void,
            up_w: *const c_void, inject_w: *const c_void, y_out: *mut c_void, inj_out: *mut f32, scratch: *mut c_void,
            h: i32, hc: i32, rank: i32, eps: f32, t: i32, stream: *mut c_void,
        ) -> i32;
        pub fn gb_hcg_pre_ref(
            streams: *const c_void, norm_w: *const c_void, down_w: *const c_void, up_w: *const c_void,
            inject_w: *const c_void, y_out: *mut c_void, inj_out: *mut f32, scratch: *mut c_void,
            h: i32, hc: i32, rank: i32, eps: f32, t: i32, stream: *mut c_void,
        ) -> i32;
        pub fn gb_hcg_pre(
            streams: *const c_void, norm_w: *const c_void, down_w: *const c_void, up_w: *const c_void,
            inject_w: *const c_void, y_out: *mut c_void, inj_out: *mut f32, scratch: *mut c_void,
            h: i32, hc: i32, rank: i32, eps: f32, t: i32, stream: *mut c_void,
        ) -> i32;
    }
}

/// Scratch bytes for a call with the inject rows (enough for the head call too).
pub fn scratch_bytes(hc: usize, rank: usize) -> usize {
    unsafe { ffi::gb_hcx_scratch_bytes(hc as i32, rank as i32, hc as i32) }
}

/// `y` `[rows, h]` bf16 from `streams` `[rows, hc, h]` f32, plus `inj` `[rows, hc]` f32 when the mixer has inject
/// weights (`inj_out` must be given exactly then).
pub fn pre(
    streams: &DevBuf,
    mix: &Mixer<'_>,
    y_out: &mut DevBuf,
    inj_out: Option<&mut DevBuf>,
    scratch: &mut DevBuf,
    h: usize,
    hc: usize,
    rank: usize,
    eps: f32,
    rows: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if mix.inject.is_some() != inj_out.is_some() {
        eprintln!("rexl3 level=error launch contract: hcx inj_out must be given exactly when the mixer has inject weights");
        return Err(CudaError(1));
    }
    mix.check(h, hc, rank, "hcx")?;
    need(streams, rows * hc * h * 4, "hcx streams")?;
    need(y_out, rows * h * 2, "hcx y")?;
    let yp = y_out.ptr();
    let ip = match inj_out {
        Some(b) => {
            need(b, rows * hc * 4, "hcx inj")?;
            b.ptr() as *mut f32
        }
        None => core::ptr::null_mut(),
    };
    need(scratch, scratch_bytes(hc, rank), "hcx scratch")?;
    let sp = scratch.ptr();
    let inj_w = mix.inject.map_or(core::ptr::null(), |b| b.ptr());
    CudaError::check(unsafe {
        ffi::gb_hcx_pre(
            streams.ptr(),
            mix.norm.ptr(),
            mix.down.ptr(),
            mix.up.ptr(),
            inj_w,
            yp,
            ip,
            sp,
            h as u32,
            hc as u32,
            rank as u32,
            eps,
            rows as u32,
            st.raw(),
        )
    })
}

/// Same as [`pre`] but always on the reference kernels (`k1_kernel`/`k2_kernel` directly): the bit-exact GPU tests
/// compare the decode fast path against this.
pub fn pre_ref(
    streams: &DevBuf,
    mix: &Mixer<'_>,
    y_out: &mut DevBuf,
    inj_out: Option<&mut DevBuf>,
    scratch: &mut DevBuf,
    h: usize,
    hc: usize,
    rank: usize,
    eps: f32,
    rows: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if mix.inject.is_some() != inj_out.is_some() {
        eprintln!("rexl3 level=error launch contract: hcx inj_out must be given exactly when the mixer has inject weights");
        return Err(CudaError(1));
    }
    mix.check(h, hc, rank, "hcx")?;
    need(streams, rows * hc * h * 4, "hcx streams")?;
    need(y_out, rows * h * 2, "hcx y")?;
    let yp = y_out.ptr();
    let ip = match inj_out {
        Some(b) => {
            need(b, rows * hc * 4, "hcx inj")?;
            b.ptr() as *mut f32
        }
        None => core::ptr::null_mut(),
    };
    need(scratch, scratch_bytes(hc, rank), "hcx scratch")?;
    let sp = scratch.ptr();
    let inj_w = mix.inject.map_or(core::ptr::null(), |b| b.ptr());
    CudaError::check(unsafe {
        ffi::gb_hcx_pre_ref(
            streams.ptr(),
            mix.norm.ptr(),
            mix.down.ptr(),
            mix.up.ptr(),
            inj_w,
            yp,
            ip,
            sp,
            h as u32,
            hc as u32,
            rank as u32,
            eps,
            rows as u32,
            st.raw(),
        )
    })
}

/// z-dump entry points (debug aids, not a production path): hcx::pre with the k2 pre-sigmoid
/// sums written to `z_out` ([hc * rows, H] f32) for bit-exact A/B of the k2 kernels.
pub fn hcx_ffi_z(
    streams: *const core::ffi::c_void,
    norm: *const core::ffi::c_void,
    down: *const core::ffi::c_void,
    up: *const core::ffi::c_void,
    inject: *const core::ffi::c_void,
    y_out: *mut core::ffi::c_void,
    inj_out: *mut f32,
    scratch: *mut core::ffi::c_void,
    z_out: *mut f32,
    h: u32,
    hc: u32,
    rank: u32,
    eps: f32,
    rows: u32,
    st: *mut core::ffi::c_void,
) -> i32 {
    unsafe { ffi::gb_hcx_pre_z(streams, norm, down, up, inject, y_out, inj_out, scratch, z_out, h, hc, rank, eps, rows, st) }
}

/// Development-only z-dump ref entry point.
pub fn hcx_ffi_z_ref(
    streams: *const core::ffi::c_void,
    norm: *const core::ffi::c_void,
    down: *const core::ffi::c_void,
    up: *const core::ffi::c_void,
    inject: *const core::ffi::c_void,
    y_out: *mut core::ffi::c_void,
    inj_out: *mut f32,
    scratch: *mut core::ffi::c_void,
    z_out: *mut f32,
    h: u32,
    hc: u32,
    rank: u32,
    eps: f32,
    rows: u32,
    st: *mut core::ffi::c_void,
) -> i32 {
    unsafe { ffi::gb_hcx_pre_z_ref(streams, norm, down, up, inject, y_out, inj_out, scratch, z_out, h, hc, rank, eps, rows, st) }
}

/// Scratch bytes of [`pre_gemm`] for up to `rows` rows.
pub fn gemm_scratch_bytes(rows: usize, h: usize, hc: usize, rank: usize) -> usize {
    unsafe { ffi::gb_hcg_scratch_bytes(rows as i32, h as i32, hc as i32, rank as i32) }
}

/// [`pre`] for large row counts (prefill chunks): the two low-rank products run as tensor-core GEMMs (fp16 operands,
/// fp32 accumulation) instead of per-row passes. `scratch` holds [`gemm_scratch_bytes`] for `rows`. The weight
/// operands are read exclusively at their declared rank stride (down `[rank, hc*h]`, up `[hc*h, rank]`); the inject
/// weights are their own buffer read by the separate inject GEMMs. `Mixer::check` refuses any weight buffer smaller
/// than what the launches touch.
pub fn pre_gemm(
    streams: &DevBuf,
    mix: &Mixer<'_>,
    y_out: &mut DevBuf,
    inj_out: Option<&mut DevBuf>,
    scratch: &mut DevBuf,
    h: usize,
    hc: usize,
    rank: usize,
    eps: f32,
    rows: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    pre_gemm_with(false, streams, mix, y_out, inj_out, scratch, h, hc, rank, eps, rows, st)
}

/// Whether [`pre_gemm_post`] takes this shape (four streams, hidden size up to 3,072).
pub fn post_fusable(h: usize, hc: usize) -> bool {
    hc == 4 && h % 4 == 0 && h <= 3 * 1024
}

/// [`pre_gemm`] after the hc_post of the previous block, in one pass over the streams: `streams` first becomes
/// `streams + bf16(block_out) * inj_prev` per stream (what `fn_mhc::hc_post(block_out, None, inj_prev, streams, ..)` writes, bit for
/// bit), then the pre-mix reads the new values. Needs [`post_fusable`].
pub fn pre_gemm_post(
    streams: &mut DevBuf,
    block_out: &DevBuf,
    inj_prev: &DevBuf,
    mix: &Mixer<'_>,
    y_out: &mut DevBuf,
    inj_out: Option<&mut DevBuf>,
    scratch: &mut DevBuf,
    h: usize,
    hc: usize,
    rank: usize,
    eps: f32,
    rows: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if mix.inject.is_some() != inj_out.is_some() || !post_fusable(h, hc) {
        eprintln!("rexl3 level=error launch contract: hcg pre_gemm_post needs inject rows matching inj_out and four streams of at most 3,072");
        return Err(CudaError(1));
    }
    mix.check(h, hc, rank, "hcg")?;
    need(streams, rows * hc * h * 4, "hcg streams")?;
    need(block_out, rows * h * 2, "hcg post block_out")?;
    need(inj_prev, rows * hc * 4, "hcg post inj")?;
    need(y_out, rows * h * 2, "hcg y")?;
    let yp = y_out.ptr();
    let ip = match inj_out {
        Some(b) => {
            need(b, rows * hc * 4, "hcg inj")?;
            b.ptr() as *mut f32
        }
        None => core::ptr::null_mut(),
    };
    need(scratch, gemm_scratch_bytes(rows, h, hc, rank), "hcg scratch")?;
    let sp = scratch.ptr();
    let inj_w = mix.inject.map_or(core::ptr::null(), |b| b.ptr());
    CudaError::check(unsafe {
        ffi::gb_hcg_pre_post(
            streams.ptr(),
            block_out.ptr(),
            inj_prev.ptr() as *const f32,
            mix.norm.ptr(),
            mix.down.ptr(),
            mix.up.ptr(),
            inj_w,
            yp,
            ip,
            sp,
            h as i32,
            hc as i32,
            rank as i32,
            eps,
            rows as i32,
            st.raw(),
        )
    })
}

/// [`pre_gemm`] on the 64-row slab loop for every shape (the arithmetic [`pre_gemm`] must reproduce bit for bit; the tests compare).
pub fn pre_gemm_ref(
    streams: &DevBuf,
    mix: &Mixer<'_>,
    y_out: &mut DevBuf,
    inj_out: Option<&mut DevBuf>,
    scratch: &mut DevBuf,
    h: usize,
    hc: usize,
    rank: usize,
    eps: f32,
    rows: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    pre_gemm_with(true, streams, mix, y_out, inj_out, scratch, h, hc, rank, eps, rows, st)
}

#[allow(clippy::too_many_arguments)]
fn pre_gemm_with(
    reference: bool,
    streams: &DevBuf,
    mix: &Mixer<'_>,
    y_out: &mut DevBuf,
    inj_out: Option<&mut DevBuf>,
    scratch: &mut DevBuf,
    h: usize,
    hc: usize,
    rank: usize,
    eps: f32,
    rows: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if mix.inject.is_some() != inj_out.is_some() {
        eprintln!("rexl3 level=error launch contract: hcg inj_out must be given exactly when the mixer has inject weights");
        return Err(CudaError(1));
    }
    mix.check(h, hc, rank, "hcg")?;
    need(streams, rows * hc * h * 4, "hcg streams")?;
    need(y_out, rows * h * 2, "hcg y")?;
    let yp = y_out.ptr();
    let ip = match inj_out {
        Some(b) => {
            need(b, rows * hc * 4, "hcg inj")?;
            b.ptr() as *mut f32
        }
        None => core::ptr::null_mut(),
    };
    need(scratch, gemm_scratch_bytes(rows, h, hc, rank), "hcg scratch")?;
    let sp = scratch.ptr();
    let inj_w = mix.inject.map_or(core::ptr::null(), |b| b.ptr());
    let f = if reference { ffi::gb_hcg_pre_ref } else { ffi::gb_hcg_pre };
    CudaError::check(unsafe {
        f(
            streams.ptr(),
            mix.norm.ptr(),
            mix.down.ptr(),
            mix.up.ptr(),
            inj_w,
            yp,
            ip,
            sp,
            h as i32,
            hc as i32,
            rank as i32,
            eps,
            rows as i32,
            st.raw(),
        )
    })
}
