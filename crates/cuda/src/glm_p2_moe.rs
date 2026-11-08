//! GLM-5.3-Flash prefill under contract P2 (Lane P step 3; `families/glm/contract.rs` has the contract, `docs/GLM_PREFILL_P2.md` section 6
//! the design): the ROUTED EXPERTS of a prompt chunk on tensor cores, at every chunk width. Kernels: `csrc/glm_p2_moe.cu`.
//!
//! **What P2 means here.** A (row, expert) pair's output depends on its row and its expert's weights only: not on the chunk, its width,
//! which pairs share a work item, their order or the block that computes them (mma.sync keeps rows apart; padding rows are zero-filled
//! and never stored). Every K sum is in an order that is a function of K alone ([`Order`]; [`ARITH_ID`] names the default). The rounding
//! points are the decode path's (`moe_prefill.rs`): the fp16 rotated inputs, the fp32 sums, the clamped SwiGLU in fp32, the fp16 down
//! input, the fp32 y. Deterministic (no atomics). The bits differ from the decode path's (accepted for prompt rows); decode and verify
//! windows never come here.
//!
//! **The combine** ([`combine`]) groups a row's slots by expert id at `gsplit` ([`group_split`]: the TP=2 ownership's cut, 144 of 288):
//! `part_g` = an fmaf chain over group g's slots in slot order from +0.0, then `out = (part_0 + part_1) + shared`. One node runs both
//! groups in one kernel ([`CombineMode::Both`]); a TP=2 rank runs its own group ([`CombineMode::Own`]), the ranks exchange one fp32 row
//! each, and [`CombineMode::Parts`] adds them in the same order: P2 at TP=2 equals P2 on one node bit for bit.
//!
//! **Schedule** (per layer, one stream): the plan (P1's sorted rows, `gb_moe_pf_plan`: pairs sorted by expert; P1's work items of 1 to 4
//! tiles of 16 rows of one expert, listed in expert order so that one launch per GEMM takes them all and an expert's items share its
//! weights through L2), the rotation into row-major fp16 A rows, ONE grouped gate/up GEMM with the SwiGLU epilogue (warps 0-3 gate, 4-7
//! up, the same 128 columns), the grouped down GEMM writing each pair's y, the combine. The scratch is P1's [`PrefillScratch`] (its
//! sorted-row buffers have the bytes the row-major rows need), so P2 needs no memory of its own.
//!
//! **Mixed-width layers** ([`run_subbanks`]). A layer whose experts do not share one width binds as homogeneous subbanks
//! (`moe_subbanks.rs`). Because a pair's output depends on its row and its expert's weights only, the pairs stage runs once per subbank
//! over the pairs whose expert that subbank holds (the picks remapped to the subbank's local ids, -1 for every other expert, exactly the
//! tensor-parallel rank's remap), each subbank writing its own pairs' y into the one pair buffer; then ONE [`combine`] over all pairs
//! groups them by GLOBAL expert id as before. No bit of a single-bank layer moves, and a subbank split of a uniform layer equals the
//! single bank bit for bit (a test).
//!
//! **Split gate/up subbanks** (gate and up of different widths: one subbank of RED-SNOW 4.91bpw's layer 13). No P2 GEMM takes two widths
//! in one block, so such a subbank's pairs run the decode path's pairs stage in windows of the decode scratch's rows
//! (`MoeExperts::subbank_prefill_pairs_windowed`, P1's route for it on the batched path): each such pair's y is the decode path's, a
//! function of its row and its expert's weights alone (batched == solo, any window, any rank), so row and chunk independence, chunked ==
//! whole, determinism and TP == one node hold for it as for every other pair. Its bits are P1's, named in the tag by [`SPLIT_GU_ID`].
//! The other subbanks of the layer and the combine are unchanged.
//!
//! **A tensor-parallel rank's subbanks** ([`pairs_subbanks_rank`]): the same loop over the rank's subbanks (its share of the experts, by
//! width), then the rank's own combine group (`CombineMode::Own`), whose pairs are exactly the ones its subbanks wrote.

use crate::exl3_linear::Dtype;
use crate::moe::{MoeDims, MoeExperts, MoeScratch};
use crate::moe_owned::OwnScratch;
use crate::moe_prefill::PrefillScratch;
use crate::moe_subbanks::MoeSubbanks;
use crate::rt::{CudaError, DevBuf, Stream};
use core::ffi::c_void;
use gb_exl3::Codebook;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

const INVALID: CudaError = CudaError(1);

/// The K order of the expert GEMMs (both P2: a function of K alone).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// One chain of MMAs over every k tile, from +0.0.
    Chain,
    /// Per 128-k block a chain from +0.0, the blocks' partials added in block order (the dense P2 GEMM's default).
    Fold128,
}

impl Order {
    pub const fn id(self) -> &'static str {
        match self {
            Order::Chain => "glm-moe-chain-2grp-v1",
            Order::Fold128 => "glm-moe-fold128-2grp-v1",
        }
    }
}

/// The order a GLM P2 prefill runs (see the measurement in `docs/GLM_PREFILL_P2.md` section 6).
pub const DEFAULT_ORDER: Order = Order::Fold128;

/// The version of the P2 expert arithmetic (the K order, the rounding points, the combine's grouping): GLM's prefix namespace and its TP
/// Hello carry it, so any change that moves a P2 bit must change this string.
pub const ARITH_ID: &str = DEFAULT_ORDER.id();

/// The arithmetic of a split gate/up subbank's pairs under P2 (no P2 GEMM takes gate != up widths): the decode path's pairs stage, P1's
/// bits per pair. P2's prefix namespace and its TP Hello carry it beside [`ARITH_ID`].
pub const SPLIT_GU_ID: &str = "p1";

/// How [`combine`] sums a row's slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineMode {
    /// `(part_0 + part_1) + shared` from the pair outputs (one node).
    Both,
    /// `part_g` of one group alone (a TP=2 rank's own group: rank r owns group r).
    Own(usize),
    /// One part over EVERY slot of the row (an fmaf chain in slot order from +0.0, no shared): a rank of the width split
    /// ([`crate::moe_width`]), whose pairs are its half of every expert's width.
    All,
    /// `(part0 + part1) + shared` from two given part buffers (the TP=2 exchange's rows).
    Parts,
}

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_glm_p2m_supported(hidden: i32, inter: i32, bits_gu: i32, bits_d: i32, codebook: i32) -> i32;
        pub fn gb_glm_p2m_rotate(x: *const c_void, x_dtype: i32, pick: *const c_void, pos: *const c_void, suh0: *const c_void,
                                 suh1: *const c_void, tab: i32, a0: *mut c_void, a1: *mut c_void, rows: i32, topk: i32, k: i32,
                                 stream: *mut c_void) -> i32;
        pub fn gb_glm_p2m_gateup(a0: *const c_void, a1: *const c_void, tg: *const c_void, tu: *const c_void, items: *const c_void,
                                 nitems: *const c_void, svh_g: *const c_void, svh_u: *const c_void, suh_d: *const c_void, ad: *mut c_void,
                                 k: i32, n: i32, bits: i32, fold: i32, shape: i32, cap: i32, limit: f32, stream: *mut c_void) -> i32;
        pub fn gb_glm_p2m_down(ad: *const c_void, td: *const c_void, items: *const c_void, nitems: *const c_void, svh_d: *const c_void,
                               spair: *const c_void, y: *mut c_void, k: i32, n: i32, bits: i32, fold: i32, shape: i32, cap: i32,
                               stream: *mut c_void) -> i32;
        pub fn gb_glm_p2m_gateup_mixed(a0: *const c_void, a1: *const c_void, tg: *const c_void, tu: *const c_void, bits: *const c_void,
                                       items: *const c_void, nitems: *const c_void, sc: *const c_void, e: i32, ad: *mut c_void, k: i32,
                                       n: i32, wmax: i32, fold: i32, cap: i32, limit: f32, stream: *mut c_void) -> i32;
        pub fn gb_glm_p2m_down_mixed(ad: *const c_void, td: *const c_void, bits: *const c_void, items: *const c_void,
                                     nitems: *const c_void, sc: *const c_void, e: i32, spair: *const c_void, y: *mut c_void, k: i32,
                                     n: i32, wmax: i32, fold: i32, cap: i32, stream: *mut c_void) -> i32;
        pub fn gb_glm_p2m_combine(mode: i32, y: *const c_void, pick: *const c_void, wts: *const c_void, shared: *const c_void,
                                  part0: *const c_void, part1: *const c_void, out: *mut c_void, rows: i32, topk: i32, d: i32, e: i32,
                                  gsplit: i32, only: i32, stream: *mut c_void) -> i32;
        pub fn gb_glm_p2m_plan_ws_bytes(rows: i32, topk: i32, e: i32) -> i64;
        pub fn gb_glm_p2m_pos_offset(rows: i32, topk: i32, e: i32) -> i64;
        pub fn gb_moe_own_remap(pick: *const c_void, local_of: *const c_void, pick_local: *mut c_void, y: *mut c_void, p: i32, d: i32,
                                e: i32, stream: *mut c_void) -> i32;
        pub fn gb_glm_p2m_plan(pick: *const c_void, spair: *mut c_void, items: *mut c_void, nitems: *mut c_void, ws: *mut c_void,
                               ws_bytes: i64, rows: i32, topk: i32, e: i32, cap: i32, stream: *mut c_void) -> i32;
    }
}

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

static RUNS: AtomicU64 = AtomicU64::new(0);
static SHAPE_GU: AtomicI32 = AtomicI32::new(0);
static SHAPE_D: AtomicI32 = AtomicI32::new(0);

/// The GEMMs' block shapes (`csrc/glm_p2_moe.cu` `launch_shape`; 0 = the default): which warp computes which output, never a bit (a test
/// checks every shape against the default). For the benches; a model never sets it.
pub fn set_shapes(gate_up: i32, down: i32) {
    SHAPE_GU.store(gate_up, Ordering::Relaxed);
    SHAPE_D.store(down, Ordering::Relaxed);
}

/// P2 expert blocks run by this process so far (one a [`run`] / [`pairs`] call): a test reads it before and after a prefill chunk to check
/// that every MoE layer of every prompt row took P2 (the every-row-through-P2 audit).
pub fn runs() -> u64 {
    RUNS.load(Ordering::Relaxed)
}

thread_local! {
    static RUNS_HERE: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
}

/// [`runs`] made by THIS thread: the audit of one rank where several ranks run in one process (the loopback TP test, a rank a thread).
pub fn runs_here() -> u64 {
    RUNS_HERE.with(|c| c.get())
}

/// Whether the P2 expert GEMMs take this bank: every expert of the layer (a single node's bank; a rank's share is step (c)), one width for
/// gate and up, Mul1 at 2 to 6 bits, widths multiples of 128.
pub fn supported(ex: &MoeExperts) -> bool {
    let d = ex.dims;
    ex.holds_every_expert()
        && ex.bits_u == ex.bits_gu
        && !ex.split_gu
        // SAFETY: a pure function of its integer arguments
        && unsafe {
            ffi::gb_glm_p2m_supported(d.hidden as i32, d.inter as i32, ex.bits_gu as i32, ex.bits_d as i32, codebook_code(ex.codebook))
                == 1
        }
}

/// The first expert id of combine group 1: the TP=2 ownership's cut (`ExpertOwnership::range(E, 1, 2)` starts at `E * 1 / 2`).
pub fn group_split(experts: usize) -> usize {
    experts / 2
}

/// The routed experts of `rows` rows under P2 on the picks in `s` (`s.pick`, `s.wts`): plan, rotation, gate/up, down, then the two-group
/// combine with `shared` (fp32 `[rows, hidden]`) added last into `out` (fp32 `[rows, hidden]`). Fills `s.y_pairs` (every pair's y).
#[allow(clippy::too_many_arguments)]
pub fn run(
    ex: &MoeExperts,
    x: &DevBuf,
    x_dtype: Dtype,
    rows: usize,
    shared: Option<&DevBuf>,
    out: &mut DevBuf,
    s: &mut PrefillScratch,
    st: &Stream,
) -> Result<(), CudaError> {
    run_with(ex, x, x_dtype, rows, shared, out, s, DEFAULT_ORDER, st, &mut |_| Ok(()))
}

/// [`run`] with the K order given (the tests compare both) and `mark(stage)` called after each stage (profiling).
#[allow(clippy::too_many_arguments)]
pub fn run_with(
    ex: &MoeExperts,
    x: &DevBuf,
    x_dtype: Dtype,
    rows: usize,
    shared: Option<&DevBuf>,
    out: &mut DevBuf,
    s: &mut PrefillScratch,
    order: Order,
    st: &Stream,
    mark: &mut dyn FnMut(&'static str) -> Result<(), CudaError>,
) -> Result<(), CudaError> {
    let d = ex.dims;
    if out.bytes() < rows * d.hidden * 4 || shared.is_some_and(|b| b.bytes() < rows * d.hidden * 4) {
        return Err(INVALID);
    }
    pairs_with(ex, x, x_dtype, rows, s, order, st, mark)?;
    combine(CombineMode::Both, &s.y_pairs, &s.pick, &s.wts, shared, None, out, rows, ex, st)?;
    mark("combine")
}

/// The pairs stage alone (plan, rotation, gate/up, down): every pair's fp32 y in `s.y_pairs` (`[rows * topk, hidden]`, at the pair index
/// `row * topk + slot`; an unrouted pair's row is not written).
#[allow(clippy::too_many_arguments)]
pub fn pairs_with(
    ex: &MoeExperts,
    x: &DevBuf,
    x_dtype: Dtype,
    rows: usize,
    s: &mut PrefillScratch,
    order: Order,
    st: &Stream,
    mark: &mut dyn FnMut(&'static str) -> Result<(), CudaError>,
) -> Result<(), CudaError> {
    if !supported(ex) {
        return Err(INVALID);
    }
    let pick = s.pick.ptr() as *const c_void;
    pairs_on(ex, x, x_dtype, rows, s, pick, ex.dims.experts, order, true, st, mark)
}

/// Whether the P2 expert GEMMs take this bank whatever experts it holds (a TP rank's share included): one width for gate and up, Mul1 at
/// 2 to 6 bits, widths multiples of 128.
pub fn supported_bank(ex: &MoeExperts) -> bool {
    let d = ex.dims;
    ex.bits_u == ex.bits_gu
        && !ex.split_gu
        // SAFETY: a pure function of its integer arguments
        && unsafe {
            ffi::gb_glm_p2m_supported(d.hidden as i32, d.inter as i32, ex.bits_gu as i32, ex.bits_d as i32, codebook_code(ex.codebook))
                == 1
        }
}

/// The pairs stage of a tensor-parallel RANK (Lane P step (c)): the global picks in `s.pick` remapped to this bank's experts (a rank bank's
/// own `local_of`, or the table of `os` for a whole bank masked to the rank's share: `OwnScratch::masked`; -1 for every other expert),
/// then plan, rotation, gate/up and down over this rank's pairs only. Every pair this rank computes gets its y in `s.y_pairs` at the
/// pair index; the others are not written (the rank's combine, `CombineMode::Own`, never reads them). Each pair's bits are a function of
/// its row and its expert's weights, exactly as on one node.
#[allow(clippy::too_many_arguments)]
pub fn pairs_rank(
    ex: &MoeExperts,
    x: &DevBuf,
    x_dtype: Dtype,
    rows: usize,
    s: &mut PrefillScratch,
    os: &mut OwnScratch,
    order: Order,
    st: &Stream,
    mark: &mut dyn FnMut(&'static str) -> Result<(), CudaError>,
) -> Result<(), CudaError> {
    pairs_bank(ex, x, x_dtype, rows, s, os, order, true, st, mark)
}

/// [`pairs_rank`] with the audit count optional (see `pairs_on`).
#[allow(clippy::too_many_arguments)]
fn pairs_bank(
    ex: &MoeExperts,
    x: &DevBuf,
    x_dtype: Dtype,
    rows: usize,
    s: &mut PrefillScratch,
    os: &mut OwnScratch,
    order: Order,
    count: bool,
    st: &Stream,
    mark: &mut dyn FnMut(&'static str) -> Result<(), CudaError>,
) -> Result<(), CudaError> {
    let d = ex.dims;
    if !supported_bank(ex) || rows == 0 || rows > os.max_rows || os.dims.experts != d.experts || os.dims.topk != d.topk {
        return Err(INVALID);
    }
    let local_of = ex.own.as_ref().map_or(&os.identity, |o| &o.local_of);
    // SAFETY: pick and pick_local hold rows * topk ids (the scratches are sized for at least `rows` rows), local_of [E] ids
    unsafe {
        check(ffi::gb_moe_own_remap(
            s.pick.ptr(),
            local_of.ptr(),
            os.pick_local.ptr(),
            core::ptr::null_mut(),
            (rows * d.topk) as i32,
            d.hidden as i32,
            d.experts as i32,
            st.raw(),
        ))?;
    }
    let pick = os.pick_local.ptr() as *const c_void;
    pairs_on(ex, x, x_dtype, rows, s, pick, ex.experts_local(), order, count, st, mark)
}

/// Whether a P2 prefill takes every subbank of a layer: a bank of one gate/up width on the P2 expert GEMMs ([`supported_bank`]), a split
/// gate/up bank through the decode path's pairs stage in windows (any width the decode kernels take: the loader built it). The refusal
/// names the first subbank that fails. Whether the subbanks hold every expert (one node) is [`run_subbanks`]'s check.
pub fn supported_subbanks(sb: &MoeSubbanks) -> Result<(), String> {
    for (i, b) in sb.banks().iter().enumerate() {
        if b.fused_gate_up() && !supported_bank(b) {
            let k = b.subbank_key();
            return Err(format!(
                "subbank {i} ({}/{}/{} bits for gate/up/down, {:?}) is not a format the P2 expert GEMMs take",
                k.bits_gate, k.bits_up, k.bits_down, k.codebook
            ));
        }
    }
    Ok(())
}

/// The routed experts of a mixed-width layer (homogeneous subbanks holding every expert) under P2: the pairs stage once per subbank
/// ([`pairs_subbanks_with`]), then the one two-group [`combine`] over the GLOBAL picks with `shared` added last into `out`. Fills
/// `s.y_pairs`. `os` is the subbank path's pick table (`OwnScratch`, at least `rows` rows); `sd` the decode scratch a split gate/up
/// subbank's windows run through (of the layer's dims; untouched when the layer has none).
#[allow(clippy::too_many_arguments)]
pub fn run_subbanks(
    sb: &MoeSubbanks,
    x: &DevBuf,
    x_dtype: Dtype,
    rows: usize,
    shared: Option<&DevBuf>,
    out: &mut DevBuf,
    s: &mut PrefillScratch,
    os: &mut OwnScratch,
    sd: &mut MoeScratch,
    st: &Stream,
) -> Result<(), CudaError> {
    let d = sb.dims;
    if out.bytes() < rows * d.hidden * 4 || shared.is_some_and(|b| b.bytes() < rows * d.hidden * 4) {
        return Err(INVALID);
    }
    pairs_subbanks_with(sb, x, x_dtype, rows, s, os, sd, DEFAULT_ORDER, st, &mut |_| Ok(()))?;
    combine_dims(CombineMode::Both, &s.y_pairs, &s.pick, &s.wts, shared, None, out, rows, &d, st)
}

/// The pairs stage of a mixed-width layer: for each subbank in order, the global picks in `s.pick` remapped to its local ids (-1 for the
/// experts of the other subbanks), then plan, rotation, gate/up and down over its pairs only (a split gate/up subbank: the decode pairs
/// stage in windows of `sd`'s rows). A (row, expert) pair's bits depend on its row and its expert's weights alone, so every valid pair
/// gets in `s.y_pairs`, at its pair index, the y the expert gives in a bank of its own width, whatever the other subbanks hold; a pair
/// whose pick is not a valid expert id is not written (the combine skips it). One P2 expert block in [`runs`] a layer, not one a subbank.
/// `mark` sees each subbank's stages in turn.
#[allow(clippy::too_many_arguments)]
pub fn pairs_subbanks_with(
    sb: &MoeSubbanks,
    x: &DevBuf,
    x_dtype: Dtype,
    rows: usize,
    s: &mut PrefillScratch,
    os: &mut OwnScratch,
    sd: &mut MoeScratch,
    order: Order,
    st: &Stream,
    mark: &mut dyn FnMut(&'static str) -> Result<(), CudaError>,
) -> Result<(), CudaError> {
    // every expert on this node: a pair whose expert no subbank held would keep a stale y for the combine to read (a width share holds
    // every expert at half its width: a rank's, never a node's)
    if !sb.ownership().is_all() || sb.width_share().is_some() {
        return Err(INVALID);
    }
    pairs_subbanks_on(sb, x, x_dtype, rows, s, os, sd, order, st, mark)
}

/// The pairs stage of a tensor-parallel RANK whose share of a layer is several subbanks (a mixed-width layer at TP=2): the loop of
/// [`pairs_subbanks_with`] over the rank's subbanks. Every pair whose expert the rank owns gets its y, the others are not written: the
/// rank's combine (`CombineMode::Own(rank)`, the ownership being the combine group's: the caller's check) reads exactly the former.
/// A width share ([`crate::moe_width`]) holds every expert at its columns: every valid pair gets the y of the rank's half of the
/// expert width (its gate/up columns, its down K rows, the down epilogue on that half's sum), and the rank's part is
/// [`CombineMode::All`].
#[allow(clippy::too_many_arguments)]
pub fn pairs_subbanks_rank(
    sb: &MoeSubbanks,
    x: &DevBuf,
    x_dtype: Dtype,
    rows: usize,
    s: &mut PrefillScratch,
    os: &mut OwnScratch,
    sd: &mut MoeScratch,
    order: Order,
    st: &Stream,
    mark: &mut dyn FnMut(&'static str) -> Result<(), CudaError>,
) -> Result<(), CudaError> {
    pairs_subbanks_on(sb, x, x_dtype, rows, s, os, sd, order, st, mark)
}

#[allow(clippy::too_many_arguments)]
fn pairs_subbanks_on(
    sb: &MoeSubbanks,
    x: &DevBuf,
    x_dtype: Dtype,
    rows: usize,
    s: &mut PrefillScratch,
    os: &mut OwnScratch,
    sd: &mut MoeScratch,
    order: Order,
    st: &Stream,
    mark: &mut dyn FnMut(&'static str) -> Result<(), CudaError>,
) -> Result<(), CudaError> {
    if supported_subbanks(sb).is_err() || (sb.has_split_gate_up() && sd.dims != sb.dims) {
        return Err(INVALID);
    }
    RUNS.fetch_add(1, Ordering::Relaxed);
    RUNS_HERE.with(|c| c.set(c.get() + 1));
    let shapes_default = SHAPE_GU.load(Ordering::Relaxed) <= 1 && SHAPE_D.load(Ordering::Relaxed) <= 1;
    match sb.multi().filter(|m| shapes_default && m.wmax_gu > 0) {
        Some(m) => pairs_multi(sb, m, x, x_dtype, rows, s, os, order, st, mark)?,
        None => {
            for b in sb.banks().iter().filter(|b| b.fused_gate_up()) {
                pairs_bank(b, x, x_dtype, rows, s, os, order, false, st, mark)?;
            }
        }
    }
    for b in sb.banks().iter().filter(|b| !b.fused_gate_up()) {
        b.subbank_prefill_pairs_windowed(x, x_dtype, rows, s, sd, os, false, crate::moe_subbanks::DownOut::Y, st)?;
        mark("split_gu_windows")?;
    }
    Ok(())
}

/// The pairs stage of every subbank of one gate/up width (the P2 GEMMs') in ONE launch a stage, over the layer's tables by global id
/// ([`crate::moe_subbanks::MultiTables`]): the picks of the other experts (a split gate/up bank's, the ones the rank does not hold) remapped
/// to -1, one plan over the global ids, one rotation, one gate/up and one down GEMM whose blocks each take their expert's width. Each
/// pair's arithmetic is its bank's launch's (the same block at the same width on the same scales), so its y is the per-bank chain's.
#[allow(clippy::too_many_arguments)]
fn pairs_multi(
    sb: &MoeSubbanks,
    m: &crate::moe_subbanks::MultiTables,
    x: &DevBuf,
    x_dtype: Dtype,
    rows: usize,
    s: &mut PrefillScratch,
    os: &mut OwnScratch,
    order: Order,
    st: &Stream,
    mark: &mut dyn FnMut(&'static str) -> Result<(), CudaError>,
) -> Result<(), CudaError> {
    let d = sb.banks()[0].dims; // the banks' (a width share's: its half of the expert width)
    if rows == 0
        || rows > s.max_rows
        || rows > os.max_rows
        || !s.fits(&d)
        || x.bytes() < rows * d.hidden * x_dtype.size()
        || os.dims.experts != d.experts
        || os.dims.topk != d.topk
        || m.wmax_gu == 0
    {
        return Err(INVALID);
    }
    let fold = (order == Order::Fold128) as i32;
    let e = d.experts as i32;
    // SAFETY: every buffer was checked against the shapes above (the scratches are priced for at least `rows` rows of these dims); the
    // tables hold [E] entries (u64 addresses of the banks' own buffers, which the subbanks own as long as `m`)
    unsafe {
        let pick = if m.fused_all {
            s.pick.ptr() as *const c_void
        } else {
            check(ffi::gb_moe_own_remap(
                s.pick.ptr(),
                m.fused_ids.ptr(),
                os.pick_local.ptr(),
                core::ptr::null_mut(),
                (rows * d.topk) as i32,
                d.hidden as i32,
                e,
                st.raw(),
            ))?;
            os.pick_local.ptr() as *const c_void
        };
        let pos = plan_on(s, pick, rows, d.topk, d.experts, st)?;
        mark("plan")?;
        check(ffi::gb_glm_p2m_rotate(
            x.ptr(),
            x_dtype as i32,
            pick,
            pos,
            m.suh[0].ptr(),
            m.suh[1].ptr(),
            1,
            s.xs_g.ptr(),
            s.xs_u.ptr(),
            rows as i32,
            d.topk as i32,
            d.hidden as i32,
            st.raw(),
        ))?;
        mark("rot_in")?;
        check(ffi::gb_glm_p2m_gateup_mixed(
            s.xs_g.ptr(),
            s.xs_u.ptr(),
            m.tp[0].ptr(),
            m.tp[1].ptr(),
            m.bits[0].ptr(),
            s.items.ptr(),
            s.nitems.ptr(),
            m.epi.ptr(),
            e,
            s.xd.ptr(),
            d.hidden as i32,
            d.inter as i32,
            m.wmax_gu,
            fold,
            s.cap as i32,
            d.limit,
            st.raw(),
        ))?;
        mark("gate_up")?;
        check(ffi::gb_glm_p2m_down_mixed(
            s.xd.ptr(),
            m.tp[2].ptr(),
            m.bits[2].ptr(),
            s.items.ptr(),
            s.nitems.ptr(),
            m.svh_d.ptr(),
            e,
            s.spair.ptr(),
            s.y_pairs.ptr(),
            d.inter as i32,
            d.hidden as i32,
            m.wmax_d,
            fold,
            s.cap as i32,
            st.raw(),
        ))?;
        mark("down")
    }
}

/// The pairs stage on `pick` (`rows * topk` ids into this bank's tables, -1 for none) over `experts` table entries. `count`: whether this
/// call counts as a P2 expert block in [`runs`] (a layer's later subbanks do not: one block a layer).
#[allow(clippy::too_many_arguments)]
fn pairs_on(
    ex: &MoeExperts,
    x: &DevBuf,
    x_dtype: Dtype,
    rows: usize,
    s: &mut PrefillScratch,
    pick: *const c_void,
    experts: usize,
    order: Order,
    count: bool,
    st: &Stream,
    mark: &mut dyn FnMut(&'static str) -> Result<(), CudaError>,
) -> Result<(), CudaError> {
    let d = ex.dims;
    if rows == 0
        || rows > s.max_rows
        || !s.fits(&d)
        || x.bytes() < rows * d.hidden * x_dtype.size()
        || experts == 0
        || experts > d.experts
    {
        return Err(INVALID);
    }
    let fold = (order == Order::Fold128) as i32;
    if count {
        RUNS.fetch_add(1, Ordering::Relaxed);
        RUNS_HERE.with(|c| c.set(c.get() + 1));
    }
    // SAFETY: every buffer was checked against the shapes above (the scratch is priced for `s.max_rows` rows of these dims; its sorted-row
    // buffers hold [padded rows][hidden] / [padded rows][inter] fp16, the bytes of P1's fragment layout)
    unsafe {
        let pos = plan_on(s, pick, rows, d.topk, experts, st)?;
        mark("plan")?;
        check(ffi::gb_glm_p2m_rotate(
            x.ptr(),
            x_dtype as i32,
            pick,
            pos,
            ex.suh_g.ptr(),
            ex.suh_u.ptr(),
            0,
            s.xs_g.ptr(),
            s.xs_u.ptr(),
            rows as i32,
            d.topk as i32,
            d.hidden as i32,
            st.raw(),
        ))?;
        mark("rot_in")?;
        check(ffi::gb_glm_p2m_gateup(
            s.xs_g.ptr(),
            s.xs_u.ptr(),
            ex.gate_ptrs.ptr(),
            ex.up_ptrs.ptr(),
            s.items.ptr(),
            s.nitems.ptr(),
            ex.svh_g.ptr(),
            ex.svh_u.ptr(),
            ex.suh_d.ptr(),
            s.xd.ptr(),
            d.hidden as i32,
            d.inter as i32,
            ex.bits_gu as i32,
            fold,
            SHAPE_GU.load(Ordering::Relaxed),
            s.cap as i32,
            d.limit,
            st.raw(),
        ))?;
        mark("gate_up")?;
        check(ffi::gb_glm_p2m_down(
            s.xd.ptr(),
            ex.down_ptrs.ptr(),
            s.items.ptr(),
            s.nitems.ptr(),
            ex.svh_d.ptr(),
            s.spair.ptr(),
            s.y_pairs.ptr(),
            d.inter as i32,
            d.hidden as i32,
            ex.bits_d as i32,
            fold,
            SHAPE_D.load(Ordering::Relaxed),
            s.cap as i32,
            st.raw(),
        ))?;
        mark("down")
    }
}

/// The plan (`s.spair`: P1's sorted pairs, the same bytes as `gb_moe_pf_plan` gives; `s.items`: P1's items in expert order, `s.nitems`:
/// {items, 0, 0, sorted rows}) on many blocks. Its workspace is the down projection's A rows (`s.xd`), which nothing reads before the
/// gate/up GEMM writes them.
pub(crate) fn plan(s: &mut PrefillScratch, rows: usize, d: &crate::moe::MoeDims, st: &Stream) -> Result<(), CudaError> {
    let pick = s.pick.ptr() as *const c_void;
    plan_on(s, pick, rows, d.topk, d.experts, st).map(|_| ())
}

/// The plan of `pick`; returns where in its workspace (`s.xd`) it left each pair's sorted row (-1: not planned), which the rotation reads.
fn plan_on(
    s: &mut PrefillScratch,
    pick: *const c_void,
    rows: usize,
    topk: usize,
    experts: usize,
    st: &Stream,
) -> Result<*const c_void, CudaError> {
    // SAFETY: pure functions of their integer arguments
    let (ws, pos) = unsafe {
        (
            ffi::gb_glm_p2m_plan_ws_bytes(rows as i32, topk as i32, experts as i32),
            ffi::gb_glm_p2m_pos_offset(rows as i32, topk as i32, experts as i32),
        )
    };
    if ws < 0 || pos < 0 || ws as usize > s.xd.bytes() {
        return Err(INVALID);
    }
    // SAFETY: the scratch is sized for `s.max_rows >= rows` rows of these dims, the workspace checked above
    unsafe {
        check(ffi::gb_glm_p2m_plan(
            pick,
            s.spair.ptr(),
            s.items.ptr(),
            s.nitems.ptr(),
            s.xd.ptr(),
            s.xd.bytes() as i64,
            rows as i32,
            topk as i32,
            experts as i32,
            s.cap as i32,
            st.raw(),
        ))?;
    }
    Ok((s.xd.ptr() as usize + pos as usize) as *const c_void)
}

/// The plan alone into `s` (a test compares it with P1's).
pub fn plan_only(s: &mut PrefillScratch, rows: usize, st: &Stream) -> Result<(), CudaError> {
    let d = s.dims;
    if rows == 0 || rows > s.max_rows {
        return Err(INVALID);
    }
    plan(s, rows, &d, st)
}

/// What [`plan_view`] returns: the sorted pair of each row, the items (in expert order), and the counts, as host vectors.
pub type PlanView = (Vec<i32>, Vec<i32>, Vec<i32>);

/// The plan's outputs (sorted pair of each row, the items in expert order, the counts `{items, 0, 0, sorted rows}`) as host vectors, for
/// a test (synchronises).
pub fn plan_view(s: &PrefillScratch, st: &Stream) -> Result<PlanView, CudaError> {
    st.sync()?;
    Ok((s.spair.to_vec()?, s.items.to_vec()?, s.nitems.to_vec()?))
}

/// The P2 combine ([`CombineMode`]): `y` the pair outputs `[rows * topk, hidden]` (unused for `Parts`), `pick` / `wts` the GLOBAL picks,
/// `parts` the two part buffers for `Parts` (`[rows, hidden]` each), `shared` added last (`Both`, `Parts`).
#[allow(clippy::too_many_arguments)]
pub fn combine(
    mode: CombineMode,
    y: &DevBuf,
    pick: &DevBuf,
    wts: &DevBuf,
    shared: Option<&DevBuf>,
    parts: Option<(&DevBuf, &DevBuf)>,
    out: &mut DevBuf,
    rows: usize,
    ex: &MoeExperts,
    st: &Stream,
) -> Result<(), CudaError> {
    combine_dims(mode, y, pick, wts, shared, parts, out, rows, &ex.dims, st)
}

/// [`combine`] on the layer's dims alone (a layer of subbanks has no one bank to take them from).
#[allow(clippy::too_many_arguments)]
pub fn combine_dims(
    mode: CombineMode,
    y: &DevBuf,
    pick: &DevBuf,
    wts: &DevBuf,
    shared: Option<&DevBuf>,
    parts: Option<(&DevBuf, &DevBuf)>,
    out: &mut DevBuf,
    rows: usize,
    d: &MoeDims,
    st: &Stream,
) -> Result<(), CudaError> {
    combine_dims_rows(mode, y, pick, wts, shared, parts, out, 0, rows, d, st)
}

/// [`combine_dims`] on rows `row0 .. row0 + rows` of every row-indexed buffer (the pairs of those rows in `y`, `pick`, `wts`; the rows
/// of `shared`, the parts and `out`, each from row 0): exactly the whole call's operations on those rows (every row is its own sum), so
/// a caller may run a chunk's rows in several calls, e.g. the rows a TP rank sends its peer first.
#[allow(clippy::too_many_arguments)]
pub fn combine_dims_rows(
    mode: CombineMode,
    y: &DevBuf,
    pick: &DevBuf,
    wts: &DevBuf,
    shared: Option<&DevBuf>,
    parts: Option<(&DevBuf, &DevBuf)>,
    out: &mut DevBuf,
    row0: usize,
    rows: usize,
    d: &MoeDims,
    st: &Stream,
) -> Result<(), CudaError> {
    let (rb, end) = (d.hidden * 4, (row0 + rows) * d.hidden * 4);
    // `All` is the one-group part: every expert id below the cut (the cut at E), group 0 alone
    let (m, only, gsplit) = match mode {
        CombineMode::Both => (0, 0, group_split(d.experts)),
        CombineMode::Own(g) if g < 2 => (1, g as i32, group_split(d.experts)),
        CombineMode::Own(_) => return Err(INVALID),
        CombineMode::All => (1, 0, d.experts),
        CombineMode::Parts => (2, 0, group_split(d.experts)),
    };
    let pairs_end = (row0 + rows) * d.topk;
    if rows == 0
        || out.bytes() < end
        || (m != 2 && (y.bytes() < pairs_end * d.hidden * 4 || pick.bytes() < pairs_end * 4 || wts.bytes() < pairs_end * 2))
        || (m == 2 && !parts.is_some_and(|(a, b)| a.bytes() >= end && b.bytes() >= end))
        || shared.is_some_and(|b| b.bytes() < end)
    {
        return Err(INVALID);
    }
    let (off, poff) = (row0 * rb, row0 * d.topk);
    let null = core::ptr::null();
    let (p0, p1) = parts.map_or((null, null), |(a, b)| (a.ptr_at(off) as *const _, b.ptr_at(off) as *const _));
    // SAFETY: every buffer holds the rows (or their pairs) up to row0 + rows (checked above); mode 2 reads no pair buffer
    unsafe {
        check(ffi::gb_glm_p2m_combine(
            m,
            if m == 2 { null } else { y.ptr_at(poff * d.hidden * 4) as *const _ },
            if m == 2 { null } else { pick.ptr_at(poff * 4) as *const _ },
            if m == 2 { null } else { wts.ptr_at(poff * 2) as *const _ },
            shared.map_or(null, |b| b.ptr_at(off) as *const _),
            p0,
            p1,
            out.ptr_at(off),
            rows as i32,
            d.topk as i32,
            d.hidden as i32,
            d.experts as i32,
            gsplit as i32,
            only,
            st.raw(),
        ))
    }
}

/// [`CombineMode::Parts`] on rows `row0 .. row0 + rows` of `parts`, `shared` and `out` (each `[.., hidden]` fp32 from row 0): `out = (part_0
/// + part_1) + shared` per element, exactly the full call's operations on those rows (a row-split TP rank sums only its own rows).
#[allow(clippy::too_many_arguments)]
pub fn combine_parts_rows(
    shared: Option<&DevBuf>,
    parts: (&DevBuf, &DevBuf),
    out: &mut DevBuf,
    row0: usize,
    rows: usize,
    d: &MoeDims,
    st: &Stream,
) -> Result<(), CudaError> {
    let (rb, end) = (d.hidden * 4, (row0 + rows) * d.hidden * 4);
    if rows == 0 || out.bytes() < end || parts.0.bytes() < end || parts.1.bytes() < end || shared.is_some_and(|b| b.bytes() < end) {
        return Err(INVALID);
    }
    let off = row0 * rb;
    let null = core::ptr::null();
    // SAFETY: every buffer holds rows 0 .. row0 + rows (checked above); the Parts combine reads only the parts and the shared rows
    unsafe {
        check(ffi::gb_glm_p2m_combine(
            2,
            null,
            null,
            null,
            shared.map_or(null, |b| b.ptr_at(off) as *const _),
            parts.0.ptr_at(off) as *const _,
            parts.1.ptr_at(off) as *const _,
            out.ptr_at(off),
            rows as i32,
            d.topk as i32,
            d.hidden as i32,
            d.experts as i32,
            group_split(d.experts) as i32,
            0,
            st.raw(),
        ))
    }
}
