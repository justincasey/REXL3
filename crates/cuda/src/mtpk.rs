//! MTP head input kernels and the device-resident draft chain (see `csrc/mtp.cu`).
#![allow(clippy::too_many_arguments)]

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
        pub fn gb_mtp_input(stack: *const c_void, norm_hidden: *const c_void, emb: *const c_void, norm_emb: *const c_void,
                            xs: *mut c_void, xe: *mut c_void, rows: i32, hc: i32, h: i32, eps: f32, stream: *mut c_void) -> i32;
        pub fn gb_mtp_combine(h: *const c_void, e: *const c_void, dst: *mut c_void, rows: i32, hc: i32, hd: i32, stream: *mut c_void) -> i32;
        pub fn gb_copy_i32(dst: *mut c_void, src: *const c_void, n: i32, stream: *mut c_void) -> i32;
        pub fn gb_mtp_stage(ids: *mut c_void, positions: *mut c_void, row_pos: *mut c_void, row_slot: *mut c_void,
                            seq_start: *mut c_void, seq_slot: *const c_void, drafts: *const c_void, chain_pos: *const c_void,
                            n: i32, stride: i32, j: i32, stream: *mut c_void) -> i32;
        pub fn gb_mtp_pick(logits: *const c_void, n: i32, out_id: *mut c_void, out_p: *mut c_void, j: i32, stream: *mut c_void) -> i32;
    }
}

/// Normed stack rows `xs` `[rows * hc][h]` f16 (`stack` f32 rows, `norm_hidden` f16 `[hc * h]` applied as `1 + w`) and
/// normed embedding rows `xe` `[rows][h]` f16 (`emb` bf16, `norm_emb` f32 `[h]`, already `1 + w`).
pub fn input(
    stack: &DevBuf,
    norm_hidden: &DevBuf,
    emb: &DevBuf,
    norm_emb: &DevBuf,
    xs: &mut DevBuf,
    xe: &mut DevBuf,
    rows: usize,
    hc: usize,
    h: usize,
    eps: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    need(stack, rows * hc * h * 4, "mtp input stack")?;
    need(norm_hidden, hc * h * 2, "mtp input norm_hidden")?;
    need(emb, rows * h * 2, "mtp input emb")?;
    need(norm_emb, h * 4, "mtp input norm_emb")?;
    need(xs, rows * hc * h * 2, "mtp input xs")?;
    let sp = xs.ptr();
    need(xe, rows * h * 2, "mtp input xe")?;
    let ep = xe.ptr();
    CudaError::check(unsafe {
        ffi::gb_mtp_input(
            stack.ptr(),
            norm_hidden.ptr(),
            emb.ptr(),
            norm_emb.ptr(),
            sp,
            ep,
            rows as i32,
            hc as i32,
            h as i32,
            eps,
            st.raw(),
        )
    })
}

/// `dst` `[rows][hc][h]` f32 = `hs` `[rows * hc][h]` f16 + `e` `[rows][h]` f16 broadcast over the streams.
pub fn combine(
    hs: &DevBuf,
    e: &DevBuf,
    dst: &mut DevBuf,
    rows: usize,
    hc: usize,
    h: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    need(hs, rows * hc * h * 2, "mtp combine h")?;
    need(e, rows * h * 2, "mtp combine e")?;
    need(dst, rows * hc * h * 4, "mtp combine dst")?;
    let dp = dst.ptr();
    CudaError::check(unsafe { ffi::gb_mtp_combine(hs.ptr(), e.ptr(), dp, rows as i32, hc as i32, h as i32, st.raw()) })
}

/// `dst[0 .. n) = src[0 .. n)` (i32) on the stream.
pub fn copy_i32(dst: &mut DevBuf, src: &DevBuf, n: usize, st: &Stream) -> Result<(), CudaError> {
    need(src, n * 4, "copy_i32 src")?;
    need(dst, n * 4, "copy_i32 dst")?;
    let dp = dst.ptr();
    CudaError::check(unsafe { ffi::gb_copy_i32(dp, src.ptr(), n as i32, st.raw()) })
}

/// The window buffers a draft step writes on the device: `n` sequences of one row each.
pub struct StepInputs<'a> {
    pub ids: &'a mut DevBuf,
    pub positions: &'a mut DevBuf,
    pub row_pos: &'a mut DevBuf,
    pub row_slot: &'a mut DevBuf,
    pub seq_start: &'a mut DevBuf,
}

/// Draft step `j`'s inputs from device state, for `n` sequences of one row (sequence `s`'s arrays start at `s * stride`): token =
/// `drafts[s * stride + j - 1]`, position = `chain_pos[s * stride + j]`, slot = `seq_slot[s]`; `seq_start` = `0..=n`.
pub fn stage(
    io: StepInputs<'_>,
    seq_slot: &DevBuf,
    drafts: &DevBuf,
    chain_pos: &DevBuf,
    n: usize,
    stride: usize,
    j: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if j == 0 || n == 0 || n > 32 || stride <= j {
        return Err(CudaError(1));
    }
    need(drafts, ((n - 1) * stride + j) * 4, "mtp stage drafts")?;
    need(chain_pos, ((n - 1) * stride + j + 1) * 4, "mtp stage chain_pos")?;
    need(seq_slot, n * 4, "mtp stage seq_slot")?;
    need(io.ids, n * 4, "mtp stage ids")?;
    let ip = io.ids.ptr();
    need(io.positions, n * 4, "mtp stage positions")?;
    let pp = io.positions.ptr();
    need(io.row_pos, n * 4, "mtp stage row_pos")?;
    let rp = io.row_pos.ptr();
    need(io.row_slot, n * 4, "mtp stage row_slot")?;
    let lp = io.row_slot.ptr();
    need(io.seq_start, (n + 1) * 4, "mtp stage seq_start")?;
    let sp = io.seq_start.ptr();
    CudaError::check(unsafe {
        ffi::gb_mtp_stage(ip, pp, rp, lp, sp, seq_slot.ptr(), drafts.ptr(), chain_pos.ptr(), n as i32, stride as i32, j as i32, st.raw())
    })
}

/// `out_id[j]` = argmax of the `n` bf16 logits starting `at` elements into `logits` (lowest id on ties), `out_p[j]` its softmax
/// probability: row `r` of a `[rows][width]` logits buffer is `at = r * width`.
pub fn pick(
    logits: &DevBuf,
    at: usize,
    n: usize,
    out_id: &mut DevBuf,
    out_p: &mut DevBuf,
    j: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    need(logits, (at + n) * 2, "mtp pick logits")?;
    need(out_id, (j + 1) * 4, "mtp pick out_id")?;
    let ip = out_id.ptr();
    need(out_p, (j + 1) * 4, "mtp pick out_p")?;
    let pp = out_p.ptr();
    CudaError::check(unsafe { ffi::gb_mtp_pick(logits.ptr_at(at * 2), n as i32, ip, pp, j as i32, st.raw()) })
}
