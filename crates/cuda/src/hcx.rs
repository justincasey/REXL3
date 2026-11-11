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
        pub fn gb_hcx_pre_post(
            streams: *mut c_void, block_out: *const c_void, inj_prev: *const f32, hc_norm_w: *const c_void,
            down_w: *const c_void, up_w: *const c_void, inject_w: *const c_void, y_out: *mut c_void,
            inj_out: *mut f32, scratch: *mut c_void, hidden_size: u32, hc_mult: u32, rank: u32, eps: f32,
            num_tokens: u32, stream: *mut c_void,
        ) -> i32;
        pub fn gb_hcx_pre_pk(
            streams: *const c_void, hc_norm_w: *const c_void, down_w: *const c_void, up_w: *const c_void,
            inject_w: *const c_void, y_out: *mut c_void, inj_out: *mut f32, scratch: *mut c_void,
            hidden_size: u32, hc_mult: u32, rank: u32, eps: f32, num_tokens: u32, stream: *mut c_void,
            pk_down: *const c_void, pk_inj: *const c_void, pk_up: *const c_void,
        ) -> i32;
        pub fn gb_hcx_pre_post_pk(
            streams: *mut c_void, block_out: *const c_void, inj_prev: *const f32, hc_norm_w: *const c_void,
            down_w: *const c_void, up_w: *const c_void, inject_w: *const c_void, y_out: *mut c_void,
            inj_out: *mut f32, scratch: *mut c_void, hidden_size: u32, hc_mult: u32, rank: u32, eps: f32,
            num_tokens: u32, stream: *mut c_void, pk_down: *const c_void, pk_inj: *const c_void, pk_up: *const c_void,
        ) -> i32;
        pub fn gb_hcx_pk_job_bytes() -> usize;
        pub fn gb_hcx_pk2_job_bytes() -> usize;
        pub fn gb_hcx_pk2_pack(w: *const c_void, h: i32, eh0: i32, out: *mut c_void, stream: *mut c_void) -> i32;
        pub fn gb_hcx_pk2_check(w: *const c_void, packed: *const c_void, h: i32, bad: *mut u64, stream: *mut c_void) -> i32;
        pub fn gb_hcx_pk_scan(w: *const c_void, n: usize, st: *mut i32, stream: *mut c_void) -> i32;
        pub fn gb_hcx_pk1_pack(w: *const c_void, rows: i32, h: i32, eh0: i32, out: *mut c_void, stream: *mut c_void) -> i32;
        pub fn gb_hcx_pk1_check(
            w: *const c_void, packed: *const c_void, rows: i32, h: i32, bad: *mut u64, stream: *mut c_void,
        ) -> i32;
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

/// Whether [`pre_post`] takes this shape (four streams, a low rank of at most 512, hidden size a multiple of 256: the decode kernels' shape).
pub fn pre_post_foldable(h: usize, hc: usize, rank: usize) -> bool {
    hc == 4 && rank <= 512 && rank % 8 == 0 && h % 256 == 0
}

/// [`pre`] after the hc_post of the previous block, in the same two launches: `streams` first becomes `streams + bf16(block_out) *
/// inj_prev` per stream (what `fn_mhc::hc_post(block_out, None, inj_prev, streams, ..)` writes, bit for bit), then the pre-mix reads the
/// new values. The mixer must have inject weights (`inj_out` given) and the shape must pass [`pre_post_foldable`].
pub fn pre_post(
    streams: &mut DevBuf,
    block_out: &DevBuf,
    inj_prev: &DevBuf,
    mix: &Mixer<'_>,
    y_out: &mut DevBuf,
    inj_out: &mut DevBuf,
    scratch: &mut DevBuf,
    h: usize,
    hc: usize,
    rank: usize,
    eps: f32,
    rows: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if mix.inject.is_none() || !pre_post_foldable(h, hc, rank) {
        eprintln!("rexl3 level=error launch contract: hcx pre_post needs inject weights, four streams, rank <= 512 and a hidden size multiple of 256");
        return Err(CudaError(1));
    }
    mix.check(h, hc, rank, "hcx")?;
    need(streams, rows * hc * h * 4, "hcx streams")?;
    need(block_out, rows * h * 2, "hcx post block_out")?;
    need(inj_prev, rows * hc * 4, "hcx post inj")?;
    need(y_out, rows * h * 2, "hcx y")?;
    let yp = y_out.ptr();
    need(inj_out, rows * hc * 4, "hcx inj")?;
    let ip = inj_out.ptr() as *mut f32;
    need(scratch, scratch_bytes(hc, rank), "hcx scratch")?;
    let sp = scratch.ptr();
    let inj_w = mix.inject.map_or(core::ptr::null(), |b| b.ptr());
    CudaError::check(unsafe {
        ffi::gb_hcx_pre_post(
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
            h as u32,
            hc as u32,
            rank as u32,
            eps,
            rows as u32,
            st.raw(),
        )
    })
}

/// A mixer's down, inject and up rows in the 13-bit packed layout of `csrc/hcx_pk.cuh`: the fp16 weights are bf16 numbers, so a raw byte and
/// a five-bit code hold each one exactly and the decode kernels stream 18.75% fewer bytes. The decoded floats are the fp16 weights
/// widened, so [`pre_packed`] and [`pre_post_packed`] give the bits [`pre`] and [`pre_post`] give. Made by [`pack_mixer`], which checks every
/// weight with the decode the kernel runs before it hands the buffers out. `inj` is absent for a head collapse (no inject rows).
pub struct Packed {
    down: DevBuf,
    inj: Option<DevBuf>,
    up: DevBuf,
}

impl Packed {
    /// Device bytes of the packed copy.
    pub fn bytes(&self) -> usize {
        self.down.bytes() + self.inj.as_ref().map_or(0, DevBuf::bytes) + self.up.bytes()
    }
}

/// The shapes the packed kernels cover (the Flash-Next family: hidden size 2,560, four streams, low rank 320).
pub fn packable(h: usize, hc: usize, rank: usize) -> bool {
    h == 2560 && hc == 4 && rank == 320
}

/// Device bytes [`pack_mixer`] allocates for a mixer of this shape (`inject`: with the `hc` inject rows), so a memory plan can account for
/// them before the weights are loaded. Zero when the shape has no packed form.
pub fn packed_plan_bytes(h: usize, hc: usize, rank: usize, inject: bool) -> usize {
    if !packable(h, hc, rank) {
        return 0;
    }
    let (job1, job2) = unsafe { (ffi::gb_hcx_pk_job_bytes(), ffi::gb_hcx_pk2_job_bytes()) };
    const LUT: usize = 4096;
    hc * (rank / 4) * job1 + LUT + if inject { hc * job1 + LUT } else { 0 } + (h / 4) * job2 + LUT
}

/// Which rows a tensor holds: the down / inject rows `[rows][4 * 2,560]` (k1p) or the up rows `[4 * 2,560][320]` (k2tp).
#[derive(Clone, Copy, PartialEq)]
enum Rows {
    Down(usize),
    Up,
}

/// One tensor of fp16 into the packed layout: the buffer (the jobs, then the tensor's 4,096-byte table); `None` when a value cannot be held
/// (more than 7 mantissa bits, a non-finite value, an exponent-pair span past 15) or the packed copy does not decode to the weight.
fn pack_one(w: &DevBuf, kind: Rows, st: &Stream) -> Result<Option<DevBuf>, CudaError> {
    let h = 2560usize;
    let n = match kind {
        Rows::Down(rows) => rows * 4 * h,
        Rows::Up => 4 * h * 320,
    };
    need(w, n * 2, "hcx pack weight")?;
    let stats = DevBuf::from_slice(&[i32::MAX, i32::MIN, 0i32])?;
    crate::rt::device_sync()?;
    CudaError::check(unsafe { ffi::gb_hcx_pk_scan(w.ptr(), n, stats.ptr() as *mut i32, st.raw()) })?;
    st.sync()?;
    let mut s = [0i32; 3];
    stats.download(&mut s)?;
    if s[2] != 0 || s[0] > s[1] || s[1] - s[0] + 1 > 15 {
        return Ok(None);
    }
    let eh0 = s[0];
    let jobs_bytes = unsafe {
        match kind {
            Rows::Down(rows) => 4 * (rows / 4) * ffi::gb_hcx_pk_job_bytes(),
            Rows::Up => (h / 4) * ffi::gb_hcx_pk2_job_bytes(),
        }
    };
    let out = DevBuf::zeroed(jobs_bytes + 4096)?;
    let bad = DevBuf::zeroed(8)?;
    crate::rt::device_sync()?;
    unsafe {
        match kind {
            Rows::Down(rows) => {
                CudaError::check(ffi::gb_hcx_pk1_pack(w.ptr(), rows as i32, h as i32, eh0, out.ptr(), st.raw()))?;
                CudaError::check(ffi::gb_hcx_pk1_check(w.ptr(), out.ptr(), rows as i32, h as i32, bad.ptr() as *mut u64, st.raw()))?;
            }
            Rows::Up => {
                CudaError::check(ffi::gb_hcx_pk2_pack(w.ptr(), h as i32, eh0, out.ptr(), st.raw()))?;
                CudaError::check(ffi::gb_hcx_pk2_check(w.ptr(), out.ptr(), h as i32, bad.ptr() as *mut u64, st.raw()))?;
            }
        }
    }
    st.sync()?;
    let mut miss = [0u64; 1];
    bad.download(&mut miss)?;
    Ok((miss[0] == 0).then_some(out))
}

/// The packed copy of a pre-mix's rows for the Flash-Next shape (see [`packable`]) when every tensor passes [`pack_one`]; `None`
/// otherwise (the fp16 weights then serve every call). Run at load, after the weights have landed on the device.
pub fn pack_mixer(mix: &Mixer<'_>, h: usize, hc: usize, rank: usize) -> Result<Option<Packed>, CudaError> {
    if !packable(h, hc, rank) {
        return Ok(None);
    }
    mix.check(h, hc, rank, "hcx pack")?;
    let st = Stream::new()?;
    let Some(down) = pack_one(mix.down, Rows::Down(rank), &st)? else { return Ok(None) };
    let inj = match mix.inject {
        Some(inject) => match pack_one(inject, Rows::Down(hc), &st)? {
            Some(buf) => Some(buf),
            None => return Ok(None),
        },
        None => None,
    };
    let Some(up) = pack_one(mix.up, Rows::Up, &st)? else { return Ok(None) };
    Ok(Some(Packed { down, inj, up }))
}

/// [`pre`] on the packed copy of `mix`'s rows (`mix.inject` and `inj_out` as in [`pre`]): the same bits.
pub fn pre_packed(
    streams: &DevBuf,
    mix: &Mixer<'_>,
    packed: &Packed,
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
    if mix.inject.is_some() != inj_out.is_some() || mix.inject.is_some() != packed.inj.is_some() || !packable(h, hc, rank) {
        eprintln!("rexl3 level=error launch contract: hcx packed pre-mix needs the Flash-Next shape and inject rows, inj_out and the packed inject rows together");
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
        ffi::gb_hcx_pre_pk(
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
            packed.down.ptr(),
            packed.inj.as_ref().map_or(core::ptr::null(), |b| b.ptr()),
            packed.up.ptr(),
        )
    })
}

/// [`pre_post`] on the packed copy of `mix`'s rows: the same bits.
pub fn pre_post_packed(
    streams: &mut DevBuf,
    block_out: &DevBuf,
    inj_prev: &DevBuf,
    mix: &Mixer<'_>,
    packed: &Packed,
    y_out: &mut DevBuf,
    inj_out: &mut DevBuf,
    scratch: &mut DevBuf,
    h: usize,
    hc: usize,
    rank: usize,
    eps: f32,
    rows: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if mix.inject.is_none() || packed.inj.is_none() || !pre_post_foldable(h, hc, rank) || !packable(h, hc, rank) {
        eprintln!("rexl3 level=error launch contract: hcx packed pre_post needs the Flash-Next shape and the packed inject rows");
        return Err(CudaError(1));
    }
    mix.check(h, hc, rank, "hcx")?;
    need(streams, rows * hc * h * 4, "hcx streams")?;
    need(block_out, rows * h * 2, "hcx post block_out")?;
    need(inj_prev, rows * hc * 4, "hcx post inj")?;
    need(y_out, rows * h * 2, "hcx y")?;
    let yp = y_out.ptr();
    need(inj_out, rows * hc * 4, "hcx inj")?;
    let ip = inj_out.ptr() as *mut f32;
    need(scratch, scratch_bytes(hc, rank), "hcx scratch")?;
    let sp = scratch.ptr();
    CudaError::check(unsafe {
        ffi::gb_hcx_pre_post_pk(
            streams.ptr(),
            block_out.ptr(),
            inj_prev.ptr() as *const f32,
            mix.norm.ptr(),
            mix.down.ptr(),
            mix.up.ptr(),
            mix.inject.map_or(core::ptr::null(), |b| b.ptr()),
            yp,
            ip,
            sp,
            h as u32,
            hc as u32,
            rank as u32,
            eps,
            rows as u32,
            st.raw(),
            packed.down.ptr(),
            packed.inj.as_ref().map_or(core::ptr::null(), |b| b.ptr()),
            packed.up.ptr(),
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
