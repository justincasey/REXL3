//! The routed experts of a MoE layer for LARGE row counts (a prefill chunk, 1..=[`MAX_ROWS`] rows): the same arithmetic as the decode path of
//! [`crate::moe`], bit for bit, with every expert's weights streamed once and decoded once per group of up to 64 rows. Kernels:
//! `csrc/moe_prefill.cu` (the design and the bit-identity argument are in its header). Nothing here belongs to one model: the expert count,
//! widths, top-k, weight width and codebook, the shared expert (or none), the routed scale and the SwiGLU clamp are all parameters.
//!
//! # Launch API (what a family calls, per MoE layer, on one stream)
//!
//! ```text
//! // once per model: the layer shapes, from the family's config (hidden K, expert width N, E experts, top-k, routed scale, SwiGLU clamp)
//! let d = MoeDims { hidden, inter, experts, topk, scale, limit };
//! // once per stream, for the largest chunk: contents on entry never matter, so ONE scratch serves every layer; PrefillScratch::bytes_for(&d, max_rows)
//! // prices it by buffer name (items_for) so the memory plan can equal the allocation
//! let mut s = PrefillScratch::new(&d, max_rows)?;
//! // per layer: the router (the decode router's kernels, any row count) fills s.pick / s.wts (the routed scale and the top-k normalisation are the
//! // router's: the combine multiplies the fp16 weights as given)
//! router.logits(&x, Dtype::F16, rows, &mut s.logits, &st)?;
//! router.select(&sig, &s.logits, rows, &mut s.pick, &mut s.wts, &st)?;
//! // the experts: out (fp32 [rows, hidden]) = sum over the row's picks in slot order of w * expert(x), then `shared` added last
//! experts.run_prefill(&x, Dtype::F16, rows, Some(&shared_out_f32), &mut out_f32, &mut s, &st)?;     // shared: None when the family has none
//! // the Qwen grouped path's variant: the router's f32 weights stay f32 end-to-end (no f32 -> fp16 rounding):
//! experts.run_prefill_f32_wts(&x, Dtype::BF16, rows, None, &mut out_f32, &mut s, &st)?;
//! ```
//!
//! Preconditions (violations return an error, never undefined behaviour): `1 <= rows <= s.max_rows <= MAX_ROWS` (4,096); `experts <= 1024`,
//! `topk <= 16`; hidden and expert width multiples of 128 (the EXL3 format's); `x` in fp16, bf16 or fp32 `[rows, hidden]`; `shared` fp32
//! `[rows, hidden]` (a family computes its shared expert itself, e.g. with [`SharedExpert::forward_big`]); an expert id outside `[0, E)`
//! in a pick (a dropped or unrouted slot) contributes nothing, as in the decode path; `MoeDims::limit <= 0` disables the SwiGLU clamps.
//! The activation is the decode path's clamped SwiGLU (`min(silu(gate), limit) * clamp(up, -limit, limit)`); another activation would be a new
//! epilogue beside `gateup_kernel` (the grouped GEMM loop is shared in `csrc/exl3_grouped.cuh`).
//!
//! # The contract: row invariance, and equality with the decode path
//!
//! `out` row r and every pair's `y` (`s.y_pairs`) equal those of [`MoeExperts::run`] on a window holding the same row, whatever the number of
//! rows of the call, the position of the row in it, the other rows' picks or how the pairs group into tiles and groups, and whatever way a prompt
//! is cut into calls. By construction: a (row, expert) pair is one row of a 16-row mma tile (mma keeps rows apart), the K split and the warp count
//! come from the layer's shape alone (`gb_moe_default_plan`, never from the row count), every sum (over a warp's k tiles, over the warps, over the
//! K splits, over the row's slots) has a fixed order, nothing is atomic and the sorted order is a function of the picks alone. A prompt chunk of
//! any size through this path therefore leaves the bits it would leave decoded one token at a time, so the chunk grid, the oracle comparison and a
//! prefix cache's key need nothing new. What it does NOT promise: the `y` slot of an unrouted pair (stale scratch, as in the decode path), or the
//! bits of a different `limit`, bit width or codebook (those are other layers).
//!
//! Tests that pin it (`crates/cuda/tests/moe_prefill.rs`, synthetic layers, no pack): equality with the decode path over 5 weight widths,
//! 13 row counts and 5 routing styles; other models' shapes (non-power-of-two expert counts, top-k 6 / 8 / 16, no clamp, other scales); GLM widths;
//! chunk grids; a permutation of the rows; a dirty or oversized scratch; a tight clamp; a K-split detector. `tools/mutation_check_glm_prefill.py`
//! breaks each invariant (grouping, combine order, K split, partial tiles, pass split) and names the test that fails; goldens in
//! `testdata/gpu/golden_moe_prefill.txt`.

use crate::exl3_linear::{Dtype, Exl3Weights};
use crate::moe::{MoeDims, MoeExperts};
use crate::rt::{CudaError, DevBuf, Stream};
use core::ffi::c_void;

#[allow(non_snake_case)]
mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_moe_pf_plan(
            pick: *const c_void,
            spair: *mut c_void,
            items: *mut c_void,
            nitems: *mut c_void,
            rows: i32,
            topk: i32,
            e: i32,
            cap: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_pf_plan_parallel(
            pick: *const c_void,
            spair: *mut c_void,
            items: *mut c_void,
            nitems: *mut c_void,
            hist: *mut c_void,
            toff: *mut c_void,
            cnt: *mut c_void,
            cls: *mut c_void,
            rows: i32,
            topk: i32,
            e: i32,
            cap: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_pf_rot_in(
            x: *const c_void,
            x_dtype: i32,
            pick: *const c_void,
            spair: *const c_void,
            nitems: *const c_void,
            suh0: *const c_void,
            suh1: *const c_void,
            xs0: *mut c_void,
            xs1: *mut c_void,
            rows: i32,
            topk: i32,
            k: i32,
            e: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_pf_gateup(
            xs0: *const c_void,
            xs1: *const c_void,
            tp0: *const c_void,
            tp1: *const c_void,
            items: *const c_void,
            nitems: *const c_void,
            svh_g: *const c_void,
            svh_u: *const c_void,
            suh_d: *const c_void,
            xd: *mut c_void,
            k: i32,
            n: i32,
            bits: i32,
            codebook: i32,
            sk: i32,
            wk: i32,
            cap: i32,
            limit: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_pf_down(
            xd: *const c_void,
            tp: *const c_void,
            items: *const c_void,
            nitems: *const c_void,
            svh_d: *const c_void,
            spair: *const c_void,
            y: *mut c_void,
            k: i32,
            n: i32,
            bits: i32,
            codebook: i32,
            sk: i32,
            wk: i32,
            cap: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_pf_combine(
            y: *const c_void,
            pick: *const c_void,
            wts: *const c_void,
            shared: *const c_void,
            out: *mut c_void,
            rows: i32,
            topk: i32,
            d: i32,
            e: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_pf_combine_f32(
            y: *const c_void,
            pick: *const c_void,
            wts: *const c_void,
            shared: *const c_void,
            out: *mut c_void,
            rows: i32,
            topk: i32,
            d: i32,
            e: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_pf_wts_f32_f16(
            src: *const c_void,
            dst: *mut c_void,
            n: i64,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_pf_finish_shared(
            routed: *const c_void,
            shared: *const c_void,
            gate: *const c_void,
            out: *mut c_void,
            rows: i32,
            d: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

const INVALID: CudaError = CudaError(1);

/// Most rows one prefill call takes.
pub const MAX_ROWS: usize = 4096;

/// Picks per block of the parallel plan (`csrc/moe_prefill.cu` `kPlanChunk`). The parallel plan's histogram and
/// scatter grids are `ceil(P / PLAN_CHUNK)` blocks; it is only a partitioning constant, never part of an arithmetic.
pub const PLAN_CHUNK: usize = 256;

fn check(rc: i32) -> Result<(), CudaError> {
    CudaError::check(rc)
}

fn codebook_code(cb: gb_exl3::Codebook) -> i32 {
    match cb {
        gb_exl3::Codebook::Inst3 => 0,
        gb_exl3::Codebook::Mcg => 1,
        gb_exl3::Codebook::Mul1 => 2,
    }
}

/// Item slots per class for `pairs` pairs and `experts` experts (the kernels' own bound, `gb_moe_pf_item_cap`).
pub fn item_cap(pairs: usize, experts: usize) -> usize {
    let e = experts.min(pairs);
    (pairs + 15 * e) / 16 / 4 + e + 1
}

/// Rows of the sorted order at most: every expert's pairs padded to whole 16-row tiles.
pub fn padded_rows(pairs: usize, experts: usize) -> usize {
    (pairs + 15 * experts.min(pairs)).div_ceil(16) * 16
}

/// Scratch of [`MoeExperts::run_prefill`] and of the router that feeds it, for up to `max_rows` rows of one layer shape. Contents on entry
/// never matter, so one scratch serves every layer on a stream.
pub struct PrefillScratch {
    pub max_rows: usize,
    pub(crate) dims: MoeDims,
    pub(crate) cap: usize,
    pub logits: DevBuf,
    pub pick: DevBuf,
    pub wts: DevBuf,
    /// The router's f32 weights (the Qwen grouped path keeps them end-to-end; the fp16 combine reads `wts`).
    pub wts_f32: DevBuf,
    /// The band-local picks a mixed-width layer remaps into (one band at a time; `-1` outside the band).
    pub(crate) pick_local: DevBuf,
    pub(crate) spair: DevBuf,
    pub(crate) items: DevBuf,
    pub(crate) nitems: DevBuf,
    /// The parallel plan's workspaces (`gb_moe_pf_plan_parallel`): local histograms then block offsets `[B * E]`,
    /// per-expert tile offsets `[E]`, per-expert counts `[E]`, and the three class offsets `[3 * E]`.
    pub(crate) plan_hist: DevBuf,
    pub(crate) plan_toff: DevBuf,
    pub(crate) plan_cnt: DevBuf,
    pub(crate) plan_cls: DevBuf,
    /// Use the multi-block deterministic plan (`gb_moe_pf_plan_parallel`) instead of the single-block `plan_kernel`.
    /// Always on for the Qwen Flash-Next grouped prefill (the only builder; the `plan_kernel` legacy path remains for
    /// other families). Both produce bit-identical spair / items / nitems; this only changes how the plan is built.
    plan_parallel: bool,
    pub(crate) xs_g: DevBuf,
    pub(crate) xs_u: DevBuf,
    pub(crate) xd: DevBuf,
    /// Every pair's fp32 output (`[rows * topk, hidden]`): what the combine sums, and what tests compare with the decode path's `y_pairs`.
    pub y_pairs: DevBuf,
}

impl PrefillScratch {
    /// The sizes in bytes, by buffer name, computed without allocating: the memory model prices a prefill chunk with it and a GPU test
    /// checks it against [`PrefillScratch::bytes`].
    pub fn items_for(dims: &MoeDims, max_rows: usize) -> Result<Vec<(&'static str, usize)>, CudaError> {
        if max_rows == 0 || max_rows > MAX_ROWS || dims.experts > 1024 || dims.topk > 16 {
            return Err(INVALID);
        }
        let p = max_rows * dims.topk;
        let cap = item_cap(p, dims.experts);
        let s = padded_rows(p, dims.experts);
        let b = p.div_ceil(PLAN_CHUNK);
        let e = dims.experts;
        Ok(vec![
            ("logits", max_rows * dims.experts * 2),
            ("pick", p * 4),
            ("wts", p * 2),
            ("wts_f32", p * 4),
            ("pick_local", p * 4),
            ("spair", s * 4),
            ("items", 3 * cap * 16),
            ("nitems", 16),
            ("plan_hist", b * e * 4),
            ("plan_toff", e * 4),
            ("plan_cnt", e * 4),
            ("plan_cls", 3 * e * 4),
            ("xs_g", s * dims.hidden * 2),
            ("xs_u", s * dims.hidden * 2),
            ("xd", s * dims.inter * 2),
            ("y_pairs", p * dims.hidden * 4),
        ])
    }

    pub fn bytes_for(dims: &MoeDims, max_rows: usize) -> Result<usize, CudaError> {
        Ok(Self::items_for(dims, max_rows)?.iter().map(|x| x.1).sum())
    }

    pub fn new(dims: &MoeDims, max_rows: usize) -> Result<Self, CudaError> {
        let it = Self::items_for(dims, max_rows)?;
        let get = |n: &str| it.iter().find(|x| x.0 == n).map(|x| x.1).unwrap_or(0);
        let p = max_rows * dims.topk;
        Ok(Self {
            max_rows,
            dims: *dims,
            cap: item_cap(p, dims.experts),
            // views: a rank of the sequence-parallel residual routes its own rows into them (`DevBuf::subview`)
            logits: DevBuf::zeroed_view(get("logits"))?,
            pick: DevBuf::zeroed_view(get("pick"))?,
            wts: DevBuf::zeroed_view(get("wts"))?,
            wts_f32: DevBuf::zeroed(get("wts_f32"))?,
            pick_local: DevBuf::zeroed(get("pick_local"))?,
            spair: DevBuf::zeroed(get("spair"))?,
            items: DevBuf::zeroed(get("items"))?,
            nitems: DevBuf::zeroed(get("nitems"))?,
            plan_hist: DevBuf::zeroed(get("plan_hist"))?,
            plan_toff: DevBuf::zeroed(get("plan_toff"))?,
            plan_cnt: DevBuf::zeroed(get("plan_cnt"))?,
            plan_cls: DevBuf::zeroed(get("plan_cls"))?,
            plan_parallel: false,
            xs_g: DevBuf::zeroed(get("xs_g"))?,
            xs_u: DevBuf::zeroed(get("xs_u"))?,
            xd: DevBuf::zeroed(get("xd"))?,
            y_pairs: DevBuf::zeroed(get("y_pairs"))?,
        })
    }

    /// Select the parallel plan builder (`gb_moe_pf_plan_parallel`) instead of the single-block `plan_kernel`.
    /// Both produce bit-identical spair / items / nitems; this only changes how the plan is computed. Off by default.
    pub fn use_parallel_plan(&mut self, on: bool) {
        self.plan_parallel = on;
    }

    /// Run the currently selected planner on `pick` and return the plan arrays (spair, items as `i32` int4 bytes,
    /// nitems). A debug/test view: it uploads, runs the plan kernel(s) and synchronises. The bit-identity of the two
    /// builders is asserted by the GPU tests with this.
    pub fn plan(&mut self, pick: &[i32], rows: usize, e: i32, st: &Stream) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>), CudaError> {
        if pick.len() > self.max_rows * self.dims.topk {
            return Err(INVALID);
        }
        self.pick.upload(pick)?;
        self.run_plan(self.pick.ptr(), rows, e, st)?;
        Ok((self.spair.to_vec()?, self.items.to_vec()?, self.nitems.to_vec()?))
    }

    /// Run the plan on `pick` (the scratch's own `pick`/`pick_local`), either the single-block `plan_kernel` (the
    /// default and the fallback) or the multi-block deterministic `gb_moe_pf_plan_parallel` when
    /// `use_parallel_plan(true)` was called. Both write the identical spair / items / nitems, so `mark("plan")` and
    /// every downstream kernel are unchanged.
    pub(crate) fn run_plan(&self, pick: *const c_void, rows: usize, e: i32, st: &Stream) -> Result<(), CudaError> {
        unsafe {
            if self.plan_parallel {
                check(ffi::gb_moe_pf_plan_parallel(
                    pick,
                    self.spair.ptr(),
                    self.items.ptr(),
                    self.nitems.ptr(),
                    self.plan_hist.ptr(),
                    self.plan_toff.ptr(),
                    self.plan_cnt.ptr(),
                    self.plan_cls.ptr(),
                    rows as i32,
                    self.dims.topk as i32,
                    e,
                    self.cap as i32,
                    st.raw(),
                ))
            } else {
                check(ffi::gb_moe_pf_plan(
                    pick,
                    self.spair.ptr(),
                    self.items.ptr(),
                    self.nitems.ptr(),
                    rows as i32,
                    self.dims.topk as i32,
                    e,
                    self.cap as i32,
                    st.raw(),
                ))
            }
        }
    }

    /// The work items of the last call by class (1 tile, 2 tiles, 3-4 tiles of one expert): a debug view (it reads the device buffer, so synchronise first).
    pub fn item_counts(&self) -> Result<[i32; 3], CudaError> {
        let v: Vec<i32> = self.nitems.to_vec()?;
        Ok([v[0], v[1], v[2]])
    }

    /// Test-only bit snapshots of the rotating intermediates a GPU A/B compares between the two plan builders: the
    /// gate/up input fragments (`xs_g` / `xs_u`) and the down GEMM's input fragment buffer (`xd`). Hidden from docs;
    /// no production caller reads these (the kernels write and read them in-place).
    #[doc(hidden)]
    pub fn dbg_xs_g(&self) -> Result<Vec<u16>, CudaError> {
        self.xs_g.to_vec()
    }
    #[doc(hidden)]
    pub fn dbg_xs_u(&self) -> Result<Vec<u16>, CudaError> {
        self.xs_u.to_vec()
    }
    #[doc(hidden)]
    pub fn dbg_xd(&self) -> Result<Vec<u16>, CudaError> {
        self.xd.to_vec()
    }
    #[doc(hidden)]
    pub fn dbg_set_xs_g(&mut self, v: &[u16]) -> Result<(), CudaError> {
        self.xs_g.upload(v)
    }
    #[doc(hidden)]
    pub fn dbg_set_xs_u(&mut self, v: &[u16]) -> Result<(), CudaError> {
        self.xs_u.upload(v)
    }
    #[doc(hidden)]
    pub fn dbg_set_xd(&mut self, v: &[u16]) -> Result<(), CudaError> {
        self.xd.upload(v)
    }

    /// Bytes of device memory this scratch holds.
    pub fn bytes(&self) -> usize {
        [
            &self.logits,
            &self.pick,
            &self.wts,
            &self.wts_f32,
            &self.pick_local,
            &self.spair,
            &self.items,
            &self.nitems,
            &self.plan_hist,
            &self.plan_toff,
            &self.plan_cnt,
            &self.plan_cls,
            &self.xs_g,
            &self.xs_u,
            &self.xd,
            &self.y_pairs,
        ]
        .iter()
        .map(|b| b.bytes())
        .sum()
    }

    /// Overwrite the picks and weights (tests, or a caller routing elsewhere): `pick` expert ids (an id outside `[0, E)` contributes
    /// nothing), `wts` the fp16 weight bits.
    pub fn set_picks(&mut self, pick: &[i32], wts: &[u16]) -> Result<(), CudaError> {
        if pick.len() != wts.len() || pick.len() > self.max_rows * self.dims.topk {
            return Err(INVALID);
        }
        self.pick.upload(pick)?;
        self.wts.upload(wts)
    }

    /// Overwrite the picks and the router's f32 weights (the Qwen grouped path's variant): `wts_f32` f32, the same slot
    /// order as `pick`.
    pub fn set_picks_f32(&mut self, pick: &[i32], wts_f32: &[f32]) -> Result<(), CudaError> {
        if pick.len() != wts_f32.len() || pick.len() > self.max_rows * self.dims.topk {
            return Err(INVALID);
        }
        self.pick.upload(pick)?;
        self.wts_f32.upload(wts_f32)
    }

    /// Fill every buffer with `byte` (a test that the pipeline never reads what it did not write).
    pub fn poison(&mut self, byte: u8, st: &Stream) -> Result<(), CudaError> {
        for b in [
            &mut self.logits,
            &mut self.pick,
            &mut self.wts,
            &mut self.wts_f32,
            &mut self.pick_local,
            &mut self.spair,
            &mut self.items,
            &mut self.nitems,
            &mut self.plan_hist,
            &mut self.plan_toff,
            &mut self.plan_cnt,
            &mut self.plan_cls,
            &mut self.xs_g,
            &mut self.xs_u,
            &mut self.xd,
            &mut self.y_pairs,
        ] {
            b.memset_async(byte, st)?;
        }
        Ok(())
    }

    pub(crate) fn fits(&self, d: &MoeDims) -> bool {
        self.dims.hidden == d.hidden && self.dims.inter == d.inter && self.dims.experts == d.experts && self.dims.topk == d.topk
    }
}

impl MoeExperts {
    /// The routed experts of `rows` rows (1..=`s.max_rows`) on the picks in `s` (`s.pick`, `s.wts`): sort the pairs by expert, rotate,
    /// grouped gate/up GEMM with its epilogue, grouped down GEMM with its epilogue, combine. `shared` (fp32 `[rows, D]`) is added last. Writes
    /// `out` `[rows, D]` fp32 (and `s.y_pairs`). The same bits as [`MoeExperts::run`] on any window of these rows.
    #[allow(clippy::too_many_arguments)]
    pub fn run_prefill(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        s: &mut PrefillScratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.run_prefill_impl(x, x_dtype, rows, shared, out, s, st, false, &mut |_| Ok(()))
    }

    /// [`MoeExperts::run_prefill`] calling `mark(stage)` after each launch (profiling: record an event there).
    #[allow(clippy::too_many_arguments)]
    pub fn run_prefill_marked(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        s: &mut PrefillScratch,
        st: &Stream,
        mark: &mut dyn FnMut(&'static str) -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        self.run_prefill_impl(x, x_dtype, rows, shared, out, s, st, false, mark)
    }

    /// The f32-weight variant of [`MoeExperts::run_prefill`] (the Qwen grouped path): the router's f32 weights in `s.wts_f32`
    /// are kept end-to-end — the combine reads them f32, no f32 -> fp16 rounding — everything else (picks, expert order,
    /// routed/shared accumulation order, one output cast) is the fp16 variant's. Other families keep [`MoeExperts::run_prefill`].
    #[allow(clippy::too_many_arguments)]
    pub fn run_prefill_f32_wts(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        s: &mut PrefillScratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.run_prefill_impl(x, x_dtype, rows, shared, out, s, st, true, &mut |_| Ok(()))
    }

    /// [`MoeExperts::run_prefill_f32_wts`] calling `mark(stage)` after each launch.
    #[allow(clippy::too_many_arguments)]
    pub fn run_prefill_f32_wts_marked(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        s: &mut PrefillScratch,
        st: &Stream,
        mark: &mut dyn FnMut(&'static str) -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        self.run_prefill_impl(x, x_dtype, rows, shared, out, s, st, true, mark)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_prefill_impl(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        s: &mut PrefillScratch,
        st: &Stream,
        wts_f32: bool,
        mark: &mut dyn FnMut(&'static str) -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        let d = self.dims;
        if rows == 0
            || !self.holds_every_expert()
            || !self.fused_gate_up()
            || rows > s.max_rows
            || !s.fits(&d)
            || x.bytes() < rows * d.hidden * x_dtype.size()
            || out.bytes() < rows * d.hidden * 4
            || shared.is_some_and(|b| b.bytes() < rows * d.hidden * 4)
            || if wts_f32 { s.wts_f32.bytes() < rows * d.topk * 4 } else { s.wts.bytes() < rows * d.topk * 2 }
        {
            return Err(INVALID);
        }
        if self.bands.len() > 1 {
            // Mixed widths: each band runs the unchanged single-band kernels on its own local picks, tables and stacks. The
            // remap kernel (`moe_owned`) turns the global picks into band-local ids (`-1` outside the band) and fills nothing
            // (null y). One band at a time, end to end: the per-band sort/rotate buffers (spair/items/pick_local/xs_g/xs_u/xd)
            // are reused safely because a band's down runs before the next band's gateup overwrites them; the final combine
            // (global picks, slot order, shared last) reads the global y_pairs slots, disjoint per band.
            let p = rows * d.topk;
            unsafe {
                for b in &self.bands {
                    let (sk_gu, wk_gu) = b.plan_gu;
                    let (sk_d, wk_d) = b.plan_d;
                    let cb = codebook_code(b.codebook);
                    let n = b.members.len() as i32;
                    check(crate::moe_owned::ffi::gb_moe_own_remap(
                        s.pick.ptr(),
                        b.local_of.ptr(),
                        s.pick_local.ptr(),
                        core::ptr::null_mut(),
                        p as i32,
                        d.hidden as i32,
                        d.experts as i32,
                        st.raw(),
                    ))?;
                    let pl = s.pick_local.ptr();
                    s.run_plan(pl, rows, n, st)?;
                    mark("plan")?;
                    check(ffi::gb_moe_pf_rot_in(
                        x.ptr(),
                        x_dtype as i32,
                        pl,
                        s.spair.ptr(),
                        s.nitems.ptr(),
                        b.suh_g.ptr(),
                        b.suh_u.ptr(),
                        s.xs_g.ptr(),
                        s.xs_u.ptr(),
                        rows as i32,
                        d.topk as i32,
                        d.hidden as i32,
                        n,
                        st.raw(),
                    ))?;
                    mark("rot_in")?;
                    check(ffi::gb_moe_pf_gateup(
                        s.xs_g.ptr(),
                        s.xs_u.ptr(),
                        b.gate_ptrs.ptr(),
                        b.up_ptrs.ptr(),
                        s.items.ptr(),
                        s.nitems.ptr(),
                        b.svh_g.ptr(),
                        b.svh_u.ptr(),
                        b.suh_d.ptr(),
                        s.xd.ptr(),
                        d.hidden as i32,
                        d.inter as i32,
                        b.bits_gu as i32,
                        cb,
                        sk_gu as i32,
                        wk_gu as i32,
                        s.cap as i32,
                        d.limit,
                        st.raw(),
                    ))?;
                    mark("gate_up")?;
                    check(crate::moe_owned::ffi::gb_moe_own_remap(
                        s.pick.ptr(),
                        b.local_of.ptr(),
                        s.pick_local.ptr(),
                        core::ptr::null_mut(),
                        p as i32,
                        d.hidden as i32,
                        d.experts as i32,
                        st.raw(),
                    ))?;
                    s.run_plan(pl, rows, n, st)?;
                    check(ffi::gb_moe_pf_down(
                        s.xd.ptr(),
                        b.down_ptrs.ptr(),
                        s.items.ptr(),
                        s.nitems.ptr(),
                        b.svh_d.ptr(),
                        s.spair.ptr(),
                        s.y_pairs.ptr(),
                        d.inter as i32,
                        d.hidden as i32,
                        b.bits_d as i32,
                        cb,
                        sk_d as i32,
                        wk_d as i32,
                        s.cap as i32,
                        st.raw(),
                    ))?;
                    mark("down")?;
                }
                let wts_ptr = if wts_f32 { s.wts_f32.ptr() as *const _ } else { s.wts.ptr() as *const _ };
                let rc = if wts_f32 {
                    check(ffi::gb_moe_pf_combine_f32(
                        s.y_pairs.ptr(),
                        s.pick.ptr(),
                        wts_ptr,
                        shared.map_or(core::ptr::null(), |b| b.ptr() as *const _),
                        out.ptr(),
                        rows as i32,
                        d.topk as i32,
                        d.hidden as i32,
                        d.experts as i32,
                        st.raw(),
                    ))
                } else {
                    check(ffi::gb_moe_pf_combine(
                        s.y_pairs.ptr(),
                        s.pick.ptr(),
                        wts_ptr,
                        shared.map_or(core::ptr::null(), |b| b.ptr() as *const _),
                        out.ptr(),
                        rows as i32,
                        d.topk as i32,
                        d.hidden as i32,
                        d.experts as i32,
                        st.raw(),
                    ))
                };
                rc?;
                mark("combine")
            }
        } else {
            let (sk_gu, wk_gu) = self.plan_gu;
            let (sk_d, wk_d) = self.plan_d;
            let cb = codebook_code(self.codebook);
            unsafe {
                s.run_plan(s.pick.ptr(), rows, d.experts as i32, st)?;
                mark("plan")?;
                check(ffi::gb_moe_pf_rot_in(
                    x.ptr(),
                    x_dtype as i32,
                    s.pick.ptr(),
                    s.spair.ptr(),
                    s.nitems.ptr(),
                    self.suh_g.ptr(),
                    self.suh_u.ptr(),
                    s.xs_g.ptr(),
                    s.xs_u.ptr(),
                    rows as i32,
                    d.topk as i32,
                    d.hidden as i32,
                    d.experts as i32,
                    st.raw(),
                ))?;
                mark("rot_in")?;
                check(ffi::gb_moe_pf_gateup(
                    s.xs_g.ptr(),
                    s.xs_u.ptr(),
                    self.gate_ptrs.ptr(),
                    self.up_ptrs.ptr(),
                    s.items.ptr(),
                    s.nitems.ptr(),
                    self.svh_g.ptr(),
                    self.svh_u.ptr(),
                    self.suh_d.ptr(),
                    s.xd.ptr(),
                    d.hidden as i32,
                    d.inter as i32,
                    self.bits_gu as i32,
                    cb,
                    sk_gu as i32,
                    wk_gu as i32,
                    s.cap as i32,
                    d.limit,
                    st.raw(),
                ))?;
                mark("gate_up")?;
                check(ffi::gb_moe_pf_down(
                    s.xd.ptr(),
                    self.down_ptrs.ptr(),
                    s.items.ptr(),
                    s.nitems.ptr(),
                    self.svh_d.ptr(),
                    s.spair.ptr(),
                    s.y_pairs.ptr(),
                    d.inter as i32,
                    d.hidden as i32,
                    self.bits_d as i32,
                    cb,
                    sk_d as i32,
                    wk_d as i32,
                    s.cap as i32,
                    st.raw(),
                ))?;
                mark("down")?;
                let wts_ptr = if wts_f32 { s.wts_f32.ptr() as *const _ } else { s.wts.ptr() as *const _ };
                let rc = if wts_f32 {
                    check(ffi::gb_moe_pf_combine_f32(
                        s.y_pairs.ptr(),
                        s.pick.ptr(),
                        wts_ptr,
                        shared.map_or(core::ptr::null(), |b| b.ptr() as *const _),
                        out.ptr(),
                        rows as i32,
                        d.topk as i32,
                        d.hidden as i32,
                        d.experts as i32,
                        st.raw(),
                    ))
                } else {
                    check(ffi::gb_moe_pf_combine(
                        s.y_pairs.ptr(),
                        s.pick.ptr(),
                        wts_ptr,
                        shared.map_or(core::ptr::null(), |b| b.ptr() as *const _),
                        out.ptr(),
                        rows as i32,
                        d.topk as i32,
                        d.hidden as i32,
                        d.experts as i32,
                        st.raw(),
                    ))
                };
                rc?;
                mark("combine")
            }
        }
    }
}

/// The router's fp32 weights (`moe_topk_softmax` writes f32) into the scratch's fp16 weights: `dst` [n] fp16 =
/// fp16(`src` [n] f32), round to nearest even (the format [`PrefillScratch::set_picks`] and the fp16 combine read; the
/// Qwen grouped path does not call this: it keeps the f32 weights end-to-end via [`MoeExperts::run_prefill_f32_wts`]).
pub fn wts_f32_f16(src: &DevBuf, dst: &mut DevBuf, n: usize, st: &Stream) -> Result<(), CudaError> {
    if n == 0 || src.bytes() < n * 4 || dst.bytes() < n * 2 {
        return Err(INVALID);
    }
    unsafe { check(ffi::gb_moe_pf_wts_f32_f16(src.ptr(), dst.ptr(), n as i64, st.raw())) }
}

/// The shared expert's finish of the Qwen path: `out` [rows, D] bf16 = bf16(`routed` + sigmoid(`gate`) * `shared`),
/// `gate` bf16 [rows]. One fp32 accumulate, one bf16 round (the CoopMK slot combine's arithmetic: the routed slots
/// summed in slot order, the shared slot weighted by its sigmoid added last, stored bf16 once).
pub fn finish_shared(
    routed: &DevBuf,
    shared: &DevBuf,
    gate: &DevBuf,
    out: &mut DevBuf,
    rows: usize,
    d: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if rows == 0
        || d == 0
        || d % 128 != 0
        || routed.bytes() < rows * d * 4
        || shared.bytes() < rows * d * 4
        || gate.bytes() < rows * 2
        || out.bytes() < rows * d * 2
    {
        return Err(INVALID);
    }
    unsafe {
        check(ffi::gb_moe_pf_finish_shared(routed.ptr(), shared.ptr(), gate.ptr(), out.ptr(), rows as i32, d as i32, st.raw()))
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// the dense gated MLPs (the shared expert, the dense layers) for large row counts

use crate::moe::SharedExpert;

mod ffi_swiglu {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_moe_swiglu_limit(g: *const c_void, u: *const c_void, a: *mut c_void, n: i64, limit: f32, stream: *mut c_void) -> i32;
    }
}

/// Scratch of [`SharedExpert::forward_big`]: contents on entry never matter. A call of more than `max_rows` rows runs in pieces of that many
/// (the weights are streamed once a piece; a row's bits do not depend on the piece).
pub struct SharedScratchBig {
    pub max_rows: usize,
    pub(crate) inter: usize,
    pub(crate) g: DevBuf,
    pub(crate) u: DevBuf,
    pub(crate) a: DevBuf,
    pub(crate) ws: DevBuf,
}

impl SharedScratchBig {
    /// Bytes of device memory [`SharedScratchBig::new`] holds for a gated MLP of `hidden` -> `inter` -> `hidden` and `max_rows` rows, computed without
    /// allocating.
    pub fn bytes_for(hidden: usize, inter: usize, max_rows: usize) -> usize {
        use crate::exl3_big::big_workspace_bytes;
        let ws = big_workspace_bytes(max_rows, hidden, inter, 2).max(big_workspace_bytes(max_rows, inter, hidden, 2));
        max_rows * inter * 4 * 2 + max_rows * inter * 2 + ws.max(16)
    }

    pub fn bytes(&self) -> usize {
        self.g.bytes() + self.u.bytes() + self.a.bytes() + self.ws.bytes()
    }

    pub fn new(sh: &SharedExpert, max_rows: usize) -> Result<Self, CudaError> {
        Self::new_dims(sh.hidden, sh.inter, max_rows)
    }

    /// [`SharedScratchBig::new`] for a gated MLP of `hidden` -> `inter` -> `hidden` whose weights are not owned by a
    /// [`SharedExpert`] (a family that keeps its three linears inside a weight Arc).
    pub fn new_dims(hidden: usize, inter: usize, max_rows: usize) -> Result<Self, CudaError> {
        if max_rows == 0 || max_rows > MAX_ROWS {
            return Err(INVALID);
        }
        let ws = Self::bytes_for(hidden, inter, max_rows) - max_rows * inter * 4 * 2 - max_rows * inter * 2;
        Ok(Self {
            max_rows,
            inter,
            g: DevBuf::zeroed(max_rows * inter * 4)?,
            u: DevBuf::zeroed(max_rows * inter * 4)?,
            a: DevBuf::zeroed(max_rows * inter * 2)?,
            ws: DevBuf::zeroed(ws)?,
        })
    }
}

impl SharedExpert {
    /// [`SharedExpert::forward`] for any number of rows, each row bit for bit what the decode path gives it: `out` [rows, hidden] fp32 =
    /// down(swiglu_limit(gate(x), up(x))), `x` [rows, hidden] in `x_dtype`. Both buffers must be sub-viewable (`DevBuf::zeroed_view`) when
    /// `rows` exceeds the scratch's `max_rows`.
    pub fn forward_big(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        out: &mut DevBuf,
        s: &mut SharedScratchBig,
        st: &Stream,
    ) -> Result<(), CudaError> {
        Self::forward_big_refs(
            &self.gate,
            &self.up,
            &self.down,
            self.hidden,
            self.inter,
            self.limit,
            x,
            x_dtype,
            rows,
            out,
            s,
            st,
        )
    }

    /// [`SharedExpert::forward_big`] on three linears the caller keeps elsewhere (the Qwen Flash-Next family's shared
    /// expert lives inside a weight Arc): the same bits for the same weights, `limit <= 0` disables the SwiGLU clamps.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_big_refs(
        gate: &Exl3Weights,
        up: &Exl3Weights,
        down: &Exl3Weights,
        hidden: usize,
        inter: usize,
        limit: f32,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        out: &mut DevBuf,
        s: &mut SharedScratchBig,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if rows == 0 || s.inter != inter || x.bytes() < rows * hidden * x_dtype.size() || out.bytes() < rows * hidden * 4 {
            return Err(INVALID);
        }
        let mut at = 0;
        while at < rows {
            let n = s.max_rows.min(rows - at);
            let xv;
            let mut ov;
            let (xr, or): (&DevBuf, &mut DevBuf) = if rows <= s.max_rows {
                (x, &mut *out)
            } else {
                xv = x.subview(at * hidden * x_dtype.size(), n * hidden * x_dtype.size());
                ov = out.subview(at * hidden * 4, n * hidden * 4);
                (&xv, &mut ov)
            };
            gate.forward_big(xr, &mut s.g, n, x_dtype, Dtype::F32, &mut s.ws, st)?;
            up.forward_big(xr, &mut s.u, n, x_dtype, Dtype::F32, &mut s.ws, st)?;
            unsafe {
                check(ffi_swiglu::gb_moe_swiglu_limit(s.g.ptr(), s.u.ptr(), s.a.ptr(), (n * inter) as i64, limit, st.raw()))?;
            }
            down.forward_big(&s.a, or, n, Dtype::F16, Dtype::F32, &mut s.ws, st)?;
            at += n;
        }
        Ok(())
    }
}
