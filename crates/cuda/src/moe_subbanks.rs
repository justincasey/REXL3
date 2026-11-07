//! Mixed-width routed experts: one MoE layer whose experts do not share one width (a pack quantized per expert, e.g. 4, 5 and 6 bits
//! inside one layer and projection) runs as HOMOGENEOUS SUBBANKS, each an ownership bank of [`crate::moe_owned`] (L1) over the experts
//! of one [`SubbankKey`]. Nothing here touches the single-bank path: a uniform layer keeps [`MoeExperts::from_region`] and
//! [`MoeExperts::run`] exactly as before.
//!
//! # The scheme
//!
//! * [`subbank_partition`] splits a rank's [`ExpertOwnership`] by key (gate, up and down widths, codebook): every owned expert lands
//!   in exactly one subbank, ids ascending, subbanks in key order. Each subbank is a [`MoeExperts`] with `own` = its ids, so its
//!   `local_of` maps a global id to the subbank's local index or `-1`; `dims.experts` stays global.
//! * The pairs stage runs once per subbank on the replicated global picks, into ONE pair buffer `y_pairs`
//!   `[rows * topk, hidden]` fp32. The FIRST subbank's remap writes the `-0.0` filler into every pair row it does not compute; the
//!   others remap without the filler, and their down stage overwrites exactly their own pairs (the decode `down_combine` writes `y`
//!   only for a valid local pick, the batched `pf_down` only for sorted pairs). So after the last subbank every pair row holds the
//!   owning subbank's `y`, or `-0.0` when no subbank of this rank holds the pick: bit for bit what summing S filler-padded buffers
//!   would give (`x + (-0.0) == x`), without the S buffers. A pair's arithmetic never depends on which other experts share its bank
//!   (the plans are the shape's alone; grouping only decides which block computes a pair), so each `y` is the single bank's.
//! * Then, as for L1's rank banks: the rank exchange over `y_pairs` (more than one rank) and ONE [`combine_pairs`] with the GLOBAL
//!   picks, the shared expert added last. The shared expert never enters a pairs stage.
//! * A subbank whose gate and up widths differ runs its gate/up GEMV as two one-matrix launches with the two-matrix plan's K split
//!   (the kernel indexes `mat = blockIdx.y / SK`, so the blocks and their arithmetic are those of the fused launch); the epilogue is
//!   unchanged. The batched path has no split-width kernel: such a subbank's batched pairs stage is refused ([`SPLIT_PREFILL`]).
//!
//! ```text
//! let parts = subbank_partition(&keys, &own)?;                                       // [(key, ownership of the subbank)]
//! let banks = parts.iter().map(|(k, o)| MoeExperts::from_region_subbank(region.clone(), &lay_of(k), dims, o.clone())).collect()?;
//! let sb = MoeSubbanks::new(dims, own, banks)?;
//! sb.run_pairs(&x, Dtype::F16, rows, &mut s, &mut os, &st)?;                          // or run_prefill_pairs
//! comm.allreduce_sum_f32(&mut s.y_pairs, ..)?;                                         // ranks > 1 only
//! combine_pairs(&s.y_pairs, &s.pick, &s.wts, shared, &mut out, rows, &dims, &st)?;    // or sb.run(..) when it holds every expert
//! ```

use crate::exl3_linear::Dtype;
use crate::moe::{default_plan, ExpertRegion, MoeDims, MoeExperts, MoeRouter, MoeScratch, ProjOffsets, SigmoidTable};
use crate::moe_owned::{combine_pairs, ffi, ExpertOwnership, OwnScratch, OwnedMap, OwnershipError};
use crate::moe_prefill::PrefillScratch;
use crate::rt::{CudaError, DevBuf, Stream};
use gb_exl3::checkpoint::LoadedGroup;
use gb_exl3::f16::f64_to_f16;
use gb_exl3::Codebook;

const INVALID: CudaError = CudaError(1);
/// A split-width subbank (gate and up of different widths) on the batched path: no kernel takes it.
pub const SPLIT_PREFILL: CudaError = CudaError(-1101);

#[allow(non_snake_case)]
pub(crate) mod ffi_multi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_moe_rot_in_multi(
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
        pub fn gb_moe_gemv_multi(
            x0: *const c_void,
            x1: *const c_void,
            tp0: *const c_void,
            tp1: *const c_void,
            bits0: *const c_void,
            bits1: *const c_void,
            uids: *const c_void,
            ucount: *const c_void,
            mcount: *const c_void,
            members: *const c_void,
            z: *mut c_void,
            mats: i32,
            k: i32,
            n: i32,
            p: i32,
            widths: i32,
            codebook: i32,
            sk: i32,
            wk: i32,
            umax: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_gateup_epilogue_multi(
            z: *const c_void,
            pick: *const c_void,
            tab: *const c_void,
            xd: *mut c_void,
            rows: i32,
            topk: i32,
            n: i32,
            sk: i32,
            e: i32,
            limit: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_moe_down_pairs_multi(
            z: *const c_void,
            pick: *const c_void,
            svh: *const c_void,
            y_pairs: *mut c_void,
            rows: i32,
            topk: i32,
            d: i32,
            sk: i32,
            e: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

#[allow(non_snake_case)]
pub(crate) mod ffi_width {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_moe_down_combine_multi(
            z: *const c_void,
            pick: *const c_void,
            svh: *const c_void,
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
        pub fn gb_moe_pf_down_raw(
            xd: *const c_void,
            tp: *const c_void,
            items: *const c_void,
            nitems: *const c_void,
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
    }
}

/// What a subbank's down stage writes into a pair row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DownOut {
    /// The pair's `y` (the down epilogue: the K splits summed in split order, H, `* svh_down`).
    Y,
    /// The bank's ONE K split's partial alone (a down plan of one split; no H, no scale): a rank of the width split
    /// ([`crate::moe_width`]), whose exchange sums the ranks' partials before the epilogue.
    Partial,
}

/// A rank's experts of every subbank as one set of per-expert tables indexed by GLOBAL id, so the decode pairs stage runs each of its
/// seven stages as ONE launch over every subbank (the picks stay global ids; each block finds its expert's width, trellis and scales in
/// the tables). A block's arithmetic is the one-bank launch's: the plans are the shape's alone and every block runs the same
/// `gemv_block` at its expert's width (`csrc/moe_experts.cu`), so each pair's `y` is the per-subbank chain's bit for bit.
pub(crate) struct MultiTables {
    /// i32 `[E]`: `g` where this rank holds expert `g`, `-1` elsewhere (the remap's table).
    owned_ids: DevBuf,
    /// u64 `[E]` each: the trellis addresses of gate, up and down.
    pub(crate) tp: [DevBuf; 3],
    /// i32 `[E]` each: the widths of gate, up and down.
    pub(crate) bits: [DevBuf; 3],
    /// u64 `[E]` each: the suh rows of gate and up.
    pub(crate) suh: [DevBuf; 2],
    /// u64 `[3][E]`: the svh_g, svh_u and suh_d rows (the gate/up epilogue's).
    pub(crate) epi: DevBuf,
    /// u64 `[E]`: the svh_down rows.
    pub(crate) svh_d: DevBuf,
    /// The widths present (bit `b - 1` for `b` bits) among gate and up, and among down.
    pub(crate) widths_gu: i32,
    pub(crate) widths_d: i32,
    pub(crate) codebook: i32,
    pub(crate) plan_gu: (usize, usize),
    pub(crate) plan_d: (usize, usize),
}

impl MultiTables {
    /// The tables of `banks` (disjoint, covering the rank's experts), or `None` when they cannot share launches: different codebooks or
    /// plans (a test's override).
    pub(crate) fn of(dims: MoeDims, banks: &[MoeExperts]) -> Result<Option<Self>, CudaError> {
        let first = &banks[0];
        if banks.iter().any(|b| b.codebook != first.codebook || b.plan_gu != first.plan_gu || b.plan_d != first.plan_d) {
            return Ok(None);
        }
        let (e, h, i) = (dims.experts, dims.hidden as u64, dims.inter as u64);
        let mut owned_ids = vec![-1i32; e];
        let mut tp = [vec![0u64; e], vec![0u64; e], vec![0u64; e]];
        let mut bits = [vec![0i32; e], vec![0i32; e], vec![0i32; e]];
        let mut suh = [vec![0u64; e], vec![0u64; e]];
        let mut epi = vec![0u64; 3 * e];
        let mut svh_d = vec![0u64; e];
        let (mut widths_gu, mut widths_d) = (0i32, 0i32);
        for b in banks {
            let o = b.own.as_ref().ok_or(INVALID)?;
            let el = o.own.local();
            let mut ptrs = [vec![0u64; el], vec![0u64; el], vec![0u64; el]];
            for (dst, src) in ptrs.iter_mut().zip([&b.gate_ptrs, &b.up_ptrs, &b.down_ptrs]) {
                src.download(dst)?;
            }
            let widths = [b.bits_gu, b.bits_u, b.bits_d];
            widths_gu |= (1 << (widths[0] - 1)) | (1 << (widths[1] - 1));
            widths_d |= 1 << (widths[2] - 1);
            for (l, &g) in o.own.owned().iter().enumerate() {
                let (g, l64) = (g as usize, l as u64);
                owned_ids[g] = g as i32;
                for m in 0..3 {
                    tp[m][g] = ptrs[m][l];
                    bits[m][g] = widths[m] as i32;
                }
                suh[0][g] = b.suh_g.ptr() as u64 + l64 * h * 2;
                suh[1][g] = b.suh_u.ptr() as u64 + l64 * h * 2;
                epi[g] = b.svh_g.ptr() as u64 + l64 * i * 2;
                epi[e + g] = b.svh_u.ptr() as u64 + l64 * i * 2;
                epi[2 * e + g] = b.suh_d.ptr() as u64 + l64 * i * 2;
                svh_d[g] = b.svh_d.ptr() as u64 + l64 * h * 2;
            }
        }
        let up = |v: &[u64]| DevBuf::from_slice(v);
        let upi = |v: &[i32]| DevBuf::from_slice(v);
        Ok(Some(Self {
            owned_ids: upi(&owned_ids)?,
            tp: [up(&tp[0])?, up(&tp[1])?, up(&tp[2])?],
            bits: [upi(&bits[0])?, upi(&bits[1])?, upi(&bits[2])?],
            suh: [up(&suh[0])?, up(&suh[1])?],
            epi: up(&epi)?,
            svh_d: up(&svh_d)?,
            widths_gu,
            widths_d,
            codebook: codebook_code(first.codebook),
            plan_gu: first.plan_gu,
            plan_d: first.plan_d,
        }))
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

/// What makes experts share a subbank: whole bits of gate, up and down, and the codebook.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubbankKey {
    pub bits_gate: u8,
    pub bits_up: u8,
    pub bits_down: u8,
    pub codebook: Codebook,
}

impl SubbankKey {
    /// The order subbanks are laid out and run in.
    fn ord(&self) -> (i32, u8, u8, u8) {
        (codebook_code(self.codebook), self.bits_gate, self.bits_up, self.bits_down)
    }

    /// Whether gate and up have one width (the fused gate/up launch).
    pub fn fused_gate_up(&self) -> bool {
        self.bits_gate == self.bits_up
    }

    /// The key of one expert's three loaded groups (`None`: a fractional width).
    pub fn of_groups(gate: &LoadedGroup, up: &LoadedGroup, down: &LoadedGroup) -> Option<Self> {
        let b = |g: &LoadedGroup| (g.group.bits.twice() % 2 == 0).then_some(g.group.bits.twice() / 2);
        if gate.group.codebook != up.group.codebook || gate.group.codebook != down.group.codebook {
            return None;
        }
        Some(Self { bits_gate: b(gate)?, bits_up: b(up)?, bits_down: b(down)?, codebook: gate.group.codebook })
    }
}

impl PartialOrd for SubbankKey {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for SubbankKey {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.ord().cmp(&o.ord())
    }
}

/// The subbanks of one rank's experts of a layer: `keys[e]` is global expert `e`'s key (`keys.len()` = the global count). Returns, in
/// key order, each distinct key among the OWNED experts with the ownership of its experts (ascending global ids). Every owned expert is
/// in exactly one subbank; a uniform layer gives one subbank whose ownership is `own` itself.
pub fn subbank_partition(
    keys: &[SubbankKey],
    own: &ExpertOwnership,
) -> Result<Vec<(SubbankKey, ExpertOwnership)>, OwnershipError> {
    if keys.len() != own.experts {
        return Err(OwnershipError::BadShape);
    }
    let mut by: std::collections::BTreeMap<SubbankKey, Vec<u32>> = std::collections::BTreeMap::new();
    for &g in own.owned() {
        by.entry(keys[g as usize]).or_default().push(g);
    }
    by.into_iter().map(|(k, ids)| Ok((k, ExpertOwnership::from_ids(own.experts, ids)?))).collect()
}

/// Where one subbank's experts sit in a region (as [`ExpertRegion`], with the widths of the key): the loader's layout of a subbank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubbankRegion {
    pub gate: ProjOffsets,
    pub up: ProjOffsets,
    pub down: ProjOffsets,
    pub key: SubbankKey,
}

/// int16 trellis bytes of a `k x n` matrix at `bits`.
fn trellis_bytes(k: usize, n: usize, bits: usize) -> usize {
    (k / 16) * (n / 16) * 16 * bits * 2
}

impl SubbankRegion {
    /// The layout the GLM loader uses for `count` experts of `key` starting at byte `base` of a region (each block aligned to
    /// `align`): gate trellises, up trellises, down trellises, then `suh`/`svh` of gate, of up, of down. Returns the layout and the
    /// end offset. For one subbank of every expert at `base = 0` this is the single bank's layout.
    pub fn stacked(key: SubbankKey, count: usize, hidden: usize, inter: usize, base: usize, align: usize) -> (Self, usize) {
        let mut cursor = base;
        let mut block = |bytes: usize| {
            let off = cursor.div_ceil(align) * align;
            cursor = off + bytes;
            off
        };
        let (bg, bu, bd) = (key.bits_gate as usize, key.bits_up as usize, key.bits_down as usize);
        let (tg, tu, td) = (trellis_bytes(hidden, inter, bg), trellis_bytes(hidden, inter, bu), trellis_bytes(inter, hidden, bd));
        let (og, ou, od) = (block(count * tg), block(count * tu), block(count * td));
        let (suh_g, svh_g) = (block(count * hidden * 2), block(count * inter * 2));
        let (suh_u, svh_u) = (block(count * hidden * 2), block(count * inter * 2));
        let (suh_d, svh_d) = (block(count * inter * 2), block(count * hidden * 2));
        let p = |trellis, trellis_stride, suh, svh| ProjOffsets { trellis, trellis_stride, suh, svh };
        let lay = Self { gate: p(og, tg, suh_g, svh_g), up: p(ou, tu, suh_u, svh_u), down: p(od, td, suh_d, svh_d), key };
        (lay, cursor)
    }

    /// Bytes of one expert's three trellises and six scale vectors.
    pub fn expert_bytes(key: SubbankKey, hidden: usize, inter: usize) -> usize {
        trellis_bytes(hidden, inter, key.bits_gate as usize)
            + trellis_bytes(hidden, inter, key.bits_up as usize)
            + trellis_bytes(inter, hidden, key.bits_down as usize)
            + 3 * (hidden + inter) * 2
    }
}

impl MoeExperts {
    /// One subbank of a rank from a region the loader filled: L1's [`MoeExperts::from_region_owned`] when gate and up share a width;
    /// a split-width bank otherwise (its gate/up GEMV runs as two one-matrix launches; only [`MoeSubbanks`] runs it).
    pub fn from_region_subbank(
        region: DevBuf,
        lay: &SubbankRegion,
        dims: MoeDims,
        own: ExpertOwnership,
    ) -> Result<Self, CudaError> {
        let k = lay.key;
        if k.fused_gate_up() {
            let er = ExpertRegion {
                gate: lay.gate,
                up: lay.up,
                down: lay.down,
                bits_gu: k.bits_gate as usize,
                bits_d: k.bits_down as usize,
                codebook: k.codebook,
            };
            return Self::from_region_owned(region, &er, dims, own);
        }
        // split widths: build as a fused bank at gate's width with up's offsets checked here, then set up's width
        let (d, i, e) = (dims.hidden, dims.inter, own.local());
        if own.experts != dims.experts || e == 0 {
            return Err(INVALID);
        }
        let tu = trellis_bytes(d, i, k.bits_up as usize);
        let inside = |off: usize, bytes: usize| off.checked_add(bytes).is_some_and(|end| end <= region.bytes());
        if lay.up.trellis_stride != tu || lay.up.trellis % 16 != 0 || !inside(lay.up.trellis, e * tu) {
            return Err(INVALID);
        }
        // the generic constructor checks gate and down, and up's scales; it checks up's trellis against gate's width, so hand it
        // gate's trellis block for up and patch the up pointer table after
        let er = ExpertRegion {
            gate: lay.gate,
            up: ProjOffsets { trellis: lay.gate.trellis, trellis_stride: lay.gate.trellis_stride, ..lay.up },
            down: lay.down,
            bits_gu: k.bits_gate as usize,
            bits_d: k.bits_down as usize,
            codebook: k.codebook,
        };
        let base = region.ptr() as u64;
        let mut m = Self::from_region_owned(region, &er, dims, own)?;
        let ptrs: Vec<u64> = (0..e).map(|x| base + (lay.up.trellis + x * tu) as u64).collect();
        m.up_ptrs = DevBuf::from_slice(&ptrs)?;
        m.bits_u = k.bits_up as usize;
        m.expert_bytes = m.expert_bytes - trellis_bytes(d, i, k.bits_gate as usize) + tu;
        Ok(m)
    }

    /// The key of this bank.
    pub fn subbank_key(&self) -> SubbankKey {
        SubbankKey {
            bits_gate: self.bits_gu as u8,
            bits_up: self.bits_u as u8,
            bits_down: self.bits_d as u8,
            codebook: self.codebook,
        }
    }

    /// Whether gate and up run as one fused launch (every single-bank path requires it).
    pub(crate) fn fused_gate_up(&self) -> bool {
        !self.split_gu && self.bits_u == self.bits_gu
    }

    /// Test hook: run gate and up as two one-matrix launches in [`MoeSubbanks`] even at one width (the split-width launch, checked
    /// against the fused one on a uniform bank). The single-bank entry points refuse such a bank.
    #[doc(hidden)]
    pub fn force_split_gate_up(&mut self, on: bool) {
        self.split_gu = on;
    }

    #[allow(clippy::too_many_arguments)]
    /// The decode pairs stage of one subbank: L1's [`MoeExperts::run_pairs`] with the filler optional and the split gate/up launch.
    pub(crate) fn subbank_pairs(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        s: &mut MoeScratch,
        os: &mut OwnScratch,
        fill: bool,
        out: DownOut,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if x.bytes() < rows * self.dims.hidden * x_dtype.size() {
            return Err(INVALID);
        }
        let (xp, pick, wts, y) = (x.ptr(), s.pick.ptr(), s.wts.ptr(), s.y_pairs.ptr());
        self.subbank_pairs_at(xp, x_dtype, rows, pick, wts, y, s, os, fill, out, st)
    }

    #[allow(clippy::too_many_arguments)]
    /// [`Self::subbank_pairs`] on explicit row arrays: `rows` rows of `x` (`x_dtype`), their global picks (`i32 [rows * topk]`) and
    /// weights (`fp16 [rows * topk]`), and the pair rows `y` (`fp32 [rows * topk, hidden]`) it writes; every other buffer is `s`'s and
    /// `os`'s. The decode entry passes `s`'s own arrays; the batched fallback of a split-width subbank passes a window of a prefill
    /// scratch's (the caller has checked those bounds).
    fn subbank_pairs_at(
        &self,
        x: *mut core::ffi::c_void,
        x_dtype: Dtype,
        rows: usize,
        pick: *mut core::ffi::c_void,
        wts: *mut core::ffi::c_void,
        y: *mut core::ffi::c_void,
        s: &mut MoeScratch,
        os: &mut OwnScratch,
        fill: bool,
        out: DownOut,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let d = self.dims;
        let p = rows * d.topk;
        let plans = (self.plan_gu, self.plan_d);
        let own = self.own.as_ref().ok_or(INVALID)?;
        if rows == 0
            || rows > s.max_rows
            || rows > os.max_rows
            || os.dims.experts != d.experts
            || os.dims.topk != d.topk
            || !s.fits_plans(&d, plans)
            || (out == DownOut::Partial && plans.1 .0 != 1)
            || s.xg.bytes() < rows * d.hidden * 4
        {
            return Err(INVALID);
        }
        let ((sk_gu, wk_gu), (sk_d, wk_d)) = plans;
        let el = own.own.local();
        let umax = p.min(el);
        let cb = codebook_code(self.codebook);
        unsafe {
            check(ffi::gb_moe_own_remap(
                pick,
                own.local_of.ptr(),
                os.pick_local.ptr(),
                if fill { y } else { core::ptr::null_mut() },
                p as i32,
                d.hidden as i32,
                d.experts as i32,
                st.raw(),
            ))?;
            let pl = os.pick_local.ptr();
            check(ffi::gb_moe_group(
                pl,
                s.uids.ptr(),
                s.ucount.ptr(),
                s.mcount.ptr(),
                s.members.ptr(),
                rows as i32,
                d.topk as i32,
                el as i32,
                st.raw(),
            ))?;
            check(ffi::gb_moe_rot_in(
                x,
                x_dtype as i32,
                pl,
                self.suh_g.ptr(),
                self.suh_u.ptr(),
                s.xg.ptr(),
                s.xu.ptr(),
                rows as i32,
                d.topk as i32,
                d.hidden as i32,
                el as i32,
                st.raw(),
            ))?;
            if self.fused_gate_up() {
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
                    cb,
                    sk_gu as i32,
                    wk_gu as i32,
                    umax as i32,
                    st.raw(),
                ))?;
            } else {
                // Z [2][sk][P][inter]: gate's half from one one-matrix launch, up's half (at sk * P * inter floats) from another, each
                // with the two-matrix plan's K split, so every block is the fused launch's block
                for (xs, tp, bits, zoff) in [
                    (s.xg.ptr(), self.gate_ptrs.ptr(), self.bits_gu, 0usize),
                    (s.xu.ptr(), self.up_ptrs.ptr(), self.bits_u, sk_gu * p * d.inter * 4),
                ] {
                    check(ffi::gb_moe_gemv(
                        xs,
                        xs,
                        tp,
                        tp,
                        s.uids.ptr(),
                        s.ucount.ptr(),
                        s.mcount.ptr(),
                        s.members.ptr(),
                        s.z.ptr_at(zoff),
                        1,
                        d.hidden as i32,
                        d.inter as i32,
                        p as i32,
                        bits as i32,
                        cb,
                        sk_gu as i32,
                        wk_gu as i32,
                        umax as i32,
                        st.raw(),
                    ))?;
                }
            }
            check(ffi::gb_moe_gateup_epilogue(
                s.z.ptr(),
                pl,
                self.svh_g.ptr(),
                self.svh_u.ptr(),
                self.suh_d.ptr(),
                s.xd.ptr(),
                rows as i32,
                d.topk as i32,
                d.inter as i32,
                sk_gu as i32,
                el as i32,
                d.limit,
                st.raw(),
            ))?;
            // `Partial` (one K split: z [1][P][hidden] is the pair rows' layout): the GEMV writes this subbank's pairs' partials
            // straight into `y`, and nothing follows
            let z = if out == DownOut::Partial { y } else { s.z.ptr() };
            check(ffi::gb_moe_gemv(
                s.xd.ptr(),
                s.xd.ptr(),
                self.down_ptrs.ptr(),
                self.down_ptrs.ptr(),
                s.uids.ptr(),
                s.ucount.ptr(),
                s.mcount.ptr(),
                s.members.ptr(),
                z,
                1,
                d.inter as i32,
                d.hidden as i32,
                p as i32,
                self.bits_d as i32,
                cb,
                sk_d as i32,
                wk_d as i32,
                umax as i32,
                st.raw(),
            ))?;
            if out == DownOut::Partial {
                return Ok(());
            }
            // the single-node down epilogue: y of this subbank's pairs only; its combined rows (no shared) go to s.xg, unused
            check(ffi::gb_moe_down_combine(
                s.z.ptr(),
                pl,
                self.svh_d.ptr(),
                wts,
                core::ptr::null(),
                y,
                s.xg.ptr(),
                rows as i32,
                d.topk as i32,
                d.hidden as i32,
                sk_d as i32,
                el as i32,
                st.raw(),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// The batched pairs stage of a SPLIT-WIDTH subbank (gate and up of different widths; no batched kernel takes it): the decode pairs
    /// stage over consecutive windows of at most `s.max_rows` rows, reading the window's rows of `x` and of the prefill scratch's picks and
    /// weights, and writing the window's pair rows of `ps.y_pairs` in place (the decode down epilogue writes only this subbank's pairs;
    /// with `fill` the window's other pair rows get the `-0.0` filler; `out`: `y`, or a width share's K-split partial). A pair's `y` is
    /// the decode path's, which is the batched path's bit for bit (`tests/moe_prefill.rs`, and `tests/moe_subbanks.rs` per pair on the
    /// RED-SNOW layers). Clobbers `s` (not its picks).
    pub(crate) fn subbank_prefill_pairs_windowed(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        ps: &mut PrefillScratch,
        s: &mut MoeScratch,
        os: &mut OwnScratch,
        fill: bool,
        out: DownOut,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let d = self.dims;
        if rows == 0
            || rows > ps.max_rows
            || s.max_rows == 0
            || !ps.fits(&d)
            || x.bytes() < rows * d.hidden * x_dtype.size()
            || ps.pick.bytes() < rows * d.topk * 4
            || ps.wts.bytes() < rows * d.topk * 2
            || ps.y_pairs.bytes() < rows * d.topk * d.hidden * 4
        {
            return Err(INVALID);
        }
        let mut a = 0usize;
        while a < rows {
            let n = (rows - a).min(s.max_rows);
            let (q, h) = (a * d.topk, d.hidden);
            self.subbank_pairs_at(
                x.ptr_at(a * h * x_dtype.size()),
                x_dtype,
                n,
                ps.pick.ptr_at(q * 4),
                ps.wts.ptr_at(q * 2),
                ps.y_pairs.ptr_at(q * h * 4),
                s,
                os,
                fill,
                out,
                st,
            )?;
            a += n;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// The batched pairs stage of one subbank: L1's [`MoeExperts::run_prefill_pairs`] with the filler optional (`out`: `y`, or a width
    /// share's K-split partial).
    fn subbank_prefill_pairs(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        s: &mut PrefillScratch,
        os: &mut OwnScratch,
        fill: bool,
        out: DownOut,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let d = self.dims;
        let p = rows * d.topk;
        let own = self.own.as_ref().ok_or(INVALID)?;
        if !self.fused_gate_up() {
            return Err(SPLIT_PREFILL);
        }
        if rows == 0
            || rows > s.max_rows
            || rows > os.max_rows
            || !s.fits(&d)
            || os.dims.experts != d.experts
            || os.dims.topk != d.topk
            || x.bytes() < rows * d.hidden * x_dtype.size()
            || (out == DownOut::Partial && self.plan_d.0 != 1)
        {
            return Err(INVALID);
        }
        let (sk_gu, wk_gu) = self.plan_gu;
        let (sk_d, wk_d) = self.plan_d;
        let cb = codebook_code(self.codebook);
        let el = own.own.local();
        unsafe {
            check(ffi::gb_moe_own_remap(
                s.pick.ptr(),
                own.local_of.ptr(),
                os.pick_local.ptr(),
                if fill { s.y_pairs.ptr() } else { core::ptr::null_mut() },
                p as i32,
                d.hidden as i32,
                d.experts as i32,
                st.raw(),
            ))?;
            let pl = os.pick_local.ptr();
            check(ffi::gb_moe_pf_plan(
                pl,
                s.spair.ptr(),
                s.items.ptr(),
                s.nitems.ptr(),
                rows as i32,
                d.topk as i32,
                el as i32,
                s.cap as i32,
                st.raw(),
            ))?;
            check(ffi::gb_moe_pf_rot_in(
                x.ptr(),
                x_dtype as i32,
                pl,
                s.spair.ptr(),
                s.nitems.ptr(),
                self.suh_g.ptr(),
                self.suh_u.ptr(),
                s.xs_g.ptr(),
                s.xs_u.ptr(),
                rows as i32,
                d.topk as i32,
                d.hidden as i32,
                el as i32,
                st.raw(),
            ))?;
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
            if out == DownOut::Partial {
                return check(ffi_width::gb_moe_pf_down_raw(
                    s.xd.ptr(),
                    self.down_ptrs.ptr(),
                    s.items.ptr(),
                    s.nitems.ptr(),
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
                ));
            }
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
            ))
        }
    }
}

/// One rank's routed experts of one MoE layer as homogeneous subbanks (see the module documentation), or a rank's WIDTH SHARE of every
/// expert ([`crate::moe_width`]: `width` set, every bank at the share's half of the expert width).
pub struct MoeSubbanks {
    /// `dims.experts` is the GLOBAL count; `dims.inter` the whole expert width (a width share's banks hold `dims.inter / ranks`).
    pub dims: MoeDims,
    pub(crate) own: ExpertOwnership,
    pub(crate) banks: Vec<MoeExperts>,
    /// The banks' experts as one set of tables (`None`: the banks differ in codebook or plan, and each runs its own chain).
    pub(crate) multi: Option<MultiTables>,
    /// `Some`: this rank holds its columns of every expert ([`crate::moe_width::WidthShare`]), and its pairs stage leaves K-split
    /// partials that the exchange sums and [`MoeSubbanks::finish`] turns into the layer's output.
    pub(crate) width: Option<crate::moe_width::WidthShare>,
    /// A width share's u64 `[E]` svh_down row addresses by global id (the down epilogue after the exchange).
    pub(crate) svh_tab: Option<DevBuf>,
}

impl MoeSubbanks {
    /// From the subbanks of `own` (each built with its ownership: [`MoeExperts::from_region_subbank`], [`MoeExperts::from_groups_owned`],
    /// ...), in the order they run. Checks they are disjoint, cover `own` exactly and share `dims`.
    pub fn new(dims: MoeDims, own: ExpertOwnership, banks: Vec<MoeExperts>) -> Result<Self, CudaError> {
        if banks.is_empty() || own.experts != dims.experts {
            return Err(INVALID);
        }
        let mut seen = vec![false; dims.experts];
        for b in &banks {
            let o: &OwnedMap = b.own.as_ref().ok_or(INVALID)?;
            if b.dims != dims || o.own.experts != dims.experts {
                return Err(INVALID);
            }
            for &g in o.own.owned() {
                if seen[g as usize] || !own.owns(g as usize) {
                    return Err(INVALID);
                }
                seen[g as usize] = true;
            }
        }
        if seen.iter().filter(|&&v| v).count() != own.local() {
            return Err(INVALID);
        }
        // every subbank's plans are the shape's alone (one scratch fits all of them)
        for b in &banks {
            if b.plan_gu != default_plan(dims.hidden, dims.inter, b.bits_gu, 2)?
                || b.plan_d != default_plan(dims.inter, dims.hidden, b.bits_d, 1)?
            {
                return Err(INVALID);
            }
        }
        let multi = MultiTables::of(dims, &banks)?;
        Ok(Self { dims, own, banks, multi, width: None, svh_tab: None })
    }

    /// From every expert's loaded groups (global id order; only the owned ones are read): partitions `own` by key and uploads each
    /// subbank into a region of its own laid out by [`SubbankRegion::stacked`], so the loader's constructor is the one exercised.
    pub fn from_groups(
        gate: &[LoadedGroup],
        up: &[LoadedGroup],
        down: &[LoadedGroup],
        dims: MoeDims,
        own: ExpertOwnership,
    ) -> Result<Self, CudaError> {
        let e = dims.experts;
        if gate.len() != e || up.len() != e || down.len() != e || own.experts != e {
            return Err(INVALID);
        }
        // keys of the owned experts (others: a placeholder, never looked at)
        let mut keys = vec![SubbankKey { bits_gate: 0, bits_up: 0, bits_down: 0, codebook: Codebook::Mul1 }; e];
        for &g in own.owned() {
            let g = g as usize;
            keys[g] = SubbankKey::of_groups(&gate[g], &up[g], &down[g]).ok_or(INVALID)?;
        }
        let parts = subbank_partition(&keys, &own).map_err(|_| INVALID)?;
        let mut banks = Vec::with_capacity(parts.len());
        for (key, o) in parts {
            let (region, lay) = upload_subbank(gate, up, down, key, &o, dims.hidden, dims.inter)?;
            banks.push(MoeExperts::from_region_subbank(region, &lay, dims, o)?);
        }
        Self::new(dims, own, banks)
    }

    /// The width share of these subbanks (`None`: whole experts).
    pub fn width_share(&self) -> Option<crate::moe_width::WidthShare> {
        self.width
    }

    pub fn ownership(&self) -> &ExpertOwnership {
        &self.own
    }

    pub fn banks(&self) -> &[MoeExperts] {
        &self.banks
    }

    pub fn banks_mut(&mut self) -> &mut [MoeExperts] {
        &mut self.banks
    }

    /// Bytes of expert weights the subbanks hold (trellises and scales of the owned experts, no padding).
    pub fn expert_bytes(&self) -> usize {
        self.banks.iter().map(|b| b.expert_bytes * b.experts_local()).sum()
    }

    /// The tables of one launch a stage, when every bank still runs the codebook and plans they were built for (a test may override a
    /// bank's plan after construction).
    pub(crate) fn multi(&self) -> Option<&MultiTables> {
        self.multi.as_ref().filter(|m| {
            self.banks.iter().all(|b| codebook_code(b.codebook) == m.codebook && b.plan_gu == m.plan_gu && b.plan_d == m.plan_d)
        })
    }

    /// Launches one decode call makes (the pairs stage: seven over every subbank at once, or each subbank's chain; then one combine; a
    /// width share: five over every subbank at once, or each subbank's chain without its down epilogue, then the finish).
    pub fn decode_launches(&self) -> usize {
        match (self.multi(), self.width.is_some()) {
            (Some(_), false) => 7 + 1,
            (Some(_), true) => 5 + 1,
            (None, w) => self.banks.iter().map(|b| (if b.fused_gate_up() { 7 } else { 8 }) - usize::from(w)).sum::<usize>() + 1,
        }
    }

    /// The decode pairs stage of this rank (1..=`s.max_rows` rows). `s.y_pairs` then holds, per pair, the `y` of the subbank that holds
    /// its pick, or `-0.0` when this rank holds none. Clobbers `s.xg`. Every stage runs as one launch over all the subbanks
    /// ([`MultiTables`]); banks that cannot share launches run in order, the first one writing the `-0.0` filler. A width share leaves
    /// instead its K-split partial of every valid pair ([`crate::moe_width`]; [`MoeSubbanks::finish`] after the exchange).
    pub fn run_pairs(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        s: &mut MoeScratch,
        os: &mut OwnScratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if self.width.is_some() {
            return self.width_pairs(x, x_dtype, rows, s, os, st);
        }
        if let Some(m) = self.multi() {
            return self.multi_pairs(m, x, x_dtype, rows, s, os, st);
        }
        for (i, b) in self.banks.iter().enumerate() {
            b.subbank_pairs(x, x_dtype, rows, s, os, i == 0, DownOut::Y, st)?;
        }
        Ok(())
    }

    /// [`MoeSubbanks::run_pairs`] as seven launches over every subbank: the remap to global ids of this rank (and the `-0.0` filler), the
    /// grouping over global ids, then rot_in, the gate/up GEMV, its epilogue, the down GEMV and the down epilogue, each finding a pair's
    /// expert in `m`. Each pair's arithmetic is its subbank chain's.
    #[allow(clippy::too_many_arguments)]
    fn multi_pairs(
        &self,
        m: &MultiTables,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        s: &mut MoeScratch,
        os: &mut OwnScratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let d = self.dims;
        let p = rows * d.topk;
        if rows == 0
            || rows > s.max_rows
            || rows > os.max_rows
            || os.dims.experts != d.experts
            || os.dims.topk != d.topk
            || !s.fits_plans(&d, (m.plan_gu, m.plan_d))
            || s.xg.bytes() < rows * d.hidden * 4
            || x.bytes() < rows * d.hidden * x_dtype.size()
        {
            return Err(INVALID);
        }
        let ((sk_gu, wk_gu), (sk_d, wk_d)) = (m.plan_gu, m.plan_d);
        let e = d.experts as i32;
        let umax = p.min(self.own.local()) as i32;
        let (r, k, pi) = (rows as i32, d.topk as i32, p as i32);
        let pl = os.pick_local.ptr();
        unsafe {
            check(ffi::gb_moe_own_remap(s.pick.ptr(), m.owned_ids.ptr(), pl, s.y_pairs.ptr(), pi, d.hidden as i32, e, st.raw()))?;
            check(ffi::gb_moe_group(pl, s.uids.ptr(), s.ucount.ptr(), s.mcount.ptr(), s.members.ptr(), r, k, e, st.raw()))?;
            check(ffi_multi::gb_moe_rot_in_multi(
                x.ptr(),
                x_dtype as i32,
                pl,
                m.suh[0].ptr(),
                m.suh[1].ptr(),
                s.xg.ptr(),
                s.xu.ptr(),
                r,
                k,
                d.hidden as i32,
                e,
                st.raw(),
            ))?;
            check(ffi_multi::gb_moe_gemv_multi(
                s.xg.ptr(),
                s.xu.ptr(),
                m.tp[0].ptr(),
                m.tp[1].ptr(),
                m.bits[0].ptr(),
                m.bits[1].ptr(),
                s.uids.ptr(),
                s.ucount.ptr(),
                s.mcount.ptr(),
                s.members.ptr(),
                s.z.ptr(),
                2,
                d.hidden as i32,
                d.inter as i32,
                pi,
                m.widths_gu,
                m.codebook,
                sk_gu as i32,
                wk_gu as i32,
                umax,
                st.raw(),
            ))?;
            check(ffi_multi::gb_moe_gateup_epilogue_multi(
                s.z.ptr(),
                pl,
                m.epi.ptr(),
                s.xd.ptr(),
                r,
                k,
                d.inter as i32,
                sk_gu as i32,
                e,
                d.limit,
                st.raw(),
            ))?;
            check(ffi_multi::gb_moe_gemv_multi(
                s.xd.ptr(),
                s.xd.ptr(),
                m.tp[2].ptr(),
                m.tp[2].ptr(),
                m.bits[2].ptr(),
                m.bits[2].ptr(),
                s.uids.ptr(),
                s.ucount.ptr(),
                s.mcount.ptr(),
                s.members.ptr(),
                s.z.ptr(),
                1,
                d.inter as i32,
                d.hidden as i32,
                pi,
                m.widths_d,
                m.codebook,
                sk_d as i32,
                wk_d as i32,
                umax,
                st.raw(),
            ))?;
            check(ffi_multi::gb_moe_down_pairs_multi(
                s.z.ptr(),
                pl,
                m.svh_d.ptr(),
                s.y_pairs.ptr(),
                r,
                k,
                d.hidden as i32,
                sk_d as i32,
                e,
                st.raw(),
            ))
        }
    }

    /// The batched pairs stage of this rank (1..=`s.max_rows` rows), as [`MoeSubbanks::run_pairs`].
    pub fn run_prefill_pairs(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        s: &mut PrefillScratch,
        os: &mut OwnScratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if self.banks.iter().any(|b| !b.fused_gate_up()) {
            return Err(SPLIT_PREFILL);
        }
        let out = self.down_out();
        for (i, b) in self.banks.iter().enumerate() {
            b.subbank_prefill_pairs(x, x_dtype, rows, s, os, i == 0 && out == DownOut::Y, out, st)?;
        }
        Ok(())
    }

    /// The batched pairs stage of this rank when some subbank has split gate/up widths: every fused subbank on the batched kernels
    /// ([`MoeSubbanks::run_prefill_pairs`]'s stage), every split one through the decode kernels in windows of `s.max_rows` rows (`s`: a
    /// decode scratch of the same dims, clobbered except its picks). With no split subbank this is exactly
    /// [`MoeSubbanks::run_prefill_pairs`] (`s` untouched). `ps.y_pairs` then holds what `run_prefill_pairs` would hold if a batched
    /// kernel took the split widths: the decode path's `y`, which equals the batched path's bit for bit.
    #[allow(clippy::too_many_arguments)]
    pub fn run_prefill_pairs_windowed(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        ps: &mut PrefillScratch,
        s: &mut MoeScratch,
        os: &mut OwnScratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if s.dims != self.dims {
            return Err(INVALID);
        }
        let out = self.down_out();
        for (i, b) in self.banks.iter().enumerate() {
            let fill = i == 0 && out == DownOut::Y;
            if b.fused_gate_up() {
                b.subbank_prefill_pairs(x, x_dtype, rows, ps, os, fill, out, st)?;
            } else {
                b.subbank_prefill_pairs_windowed(x, x_dtype, rows, ps, s, os, fill, out, st)?;
            }
        }
        Ok(())
    }

    /// What the pairs stages leave in a pair row: `y`, or a width share's K-split partial.
    fn down_out(&self) -> DownOut {
        if self.width.is_some() {
            DownOut::Partial
        } else {
            DownOut::Y
        }
    }

    /// Whether some subbank has split gate/up widths (its batched stage runs through [`MoeSubbanks::run_prefill_pairs_windowed`]).
    pub fn has_split_gate_up(&self) -> bool {
        self.banks.iter().any(|b| !b.fused_gate_up())
    }

    /// Every expert on one rank: the decode pairs stage, then the combine (global picks, `shared` added last) into `out`. A partial
    /// ownership is refused (its pairs need the exchange first).
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        s: &mut MoeScratch,
        os: &mut OwnScratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if !self.own.is_all() || self.width.is_some() {
            return Err(INVALID);
        }
        self.run_pairs(x, x_dtype, rows, s, os, st)?;
        combine_pairs(&s.y_pairs, &s.pick, &s.wts, shared, out, rows, &self.dims, st)
    }

    /// [`MoeSubbanks::run`] on the batched path.
    #[allow(clippy::too_many_arguments)]
    pub fn run_prefill(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        s: &mut PrefillScratch,
        os: &mut OwnScratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if !self.own.is_all() || self.width.is_some() {
            return Err(INVALID);
        }
        self.run_prefill_pairs(x, x_dtype, rows, s, os, st)?;
        combine_pairs(&s.y_pairs, &s.pick, &s.wts, shared, out, rows, &self.dims, st)
    }
}

/// One subbank's experts (`o`'s ids, of `gate` / `up` / `down` indexed by global id) uploaded into a region of its own, laid out by
/// [`SubbankRegion::stacked`] at `hidden` x `inter` (the loader's layout: what [`MoeExperts::from_region_subbank`] reads).
pub(crate) fn upload_subbank(
    gate: &[LoadedGroup],
    up: &[LoadedGroup],
    down: &[LoadedGroup],
    key: SubbankKey,
    o: &ExpertOwnership,
    hidden: usize,
    inter: usize,
) -> Result<(DevBuf, SubbankRegion), CudaError> {
    let (lay, end) = SubbankRegion::stacked(key, o.local(), hidden, inter, 0, 256);
    let mut host = vec![0u8; end.div_ceil(256) * 256];
    let put16 = |host: &mut Vec<u8>, off: usize, v: &[u16]| {
        for (j, h) in v.iter().enumerate() {
            host[off + 2 * j..off + 2 * j + 2].copy_from_slice(&h.to_le_bytes());
        }
    };
    for (l, &g) in o.owned().iter().enumerate() {
        let g = g as usize;
        for (grp, p) in [(&gate[g], &lay.gate), (&up[g], &lay.up), (&down[g], &lay.down)] {
            if grp.trellis.len() * 2 != p.trellis_stride || grp.group.k * grp.group.n == 0 {
                return Err(INVALID);
            }
            let t: Vec<u16> = grp.trellis.iter().map(|&v| v as u16).collect();
            put16(&mut host, p.trellis + l * p.trellis_stride, &t);
            let suh: Vec<u16> = grp.suh.iter().map(|&v| f64_to_f16(v)).collect();
            let svh: Vec<u16> = grp.svh.iter().map(|&v| f64_to_f16(v)).collect();
            put16(&mut host, p.suh + l * grp.group.k * 2, &suh);
            put16(&mut host, p.svh + l * grp.group.n * 2, &svh);
        }
    }
    let mut region = DevBuf::zeroed_view(host.len())?;
    region.upload(&host)?;
    Ok((region, lay))
}

/// Router plus subbanked routed experts of one layer (the loader's object for a mixed-width layer or a rank's partial bank; a uniform
/// layer on one node stays a [`crate::moe::MoeLayer`]).
pub struct MoeSubbankLayer {
    pub router: MoeRouter,
    pub experts: MoeSubbanks,
    pub dims: MoeDims,
}

impl MoeSubbankLayer {
    pub fn new(router: MoeRouter, experts: MoeSubbanks) -> Result<Self, CudaError> {
        let d = experts.dims;
        if router.hidden != d.hidden || router.experts != d.experts || router.topk != d.topk || router.scale != d.scale {
            return Err(INVALID);
        }
        Ok(Self { dims: d, router, experts })
    }

    /// Router (replicated on every rank): fills `s.logits`, `s.pick`, `s.wts`, as [`crate::moe::MoeLayer::route`].
    pub fn route(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        tab: &SigmoidTable,
        s: &mut MoeScratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if rows == 0 || rows > s.max_rows || s.dims != self.dims {
            return Err(INVALID);
        }
        self.router.route(x, x_dtype, rows, tab, &mut s.logits, &mut s.pick, &mut s.wts, st)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(g: u8, u: u8, d: u8) -> SubbankKey {
        SubbankKey { bits_gate: g, bits_up: u, bits_down: d, codebook: Codebook::Mul1 }
    }

    /// A mixed layer like RED-SNOW's layer 7: 5/5/4 mostly, 4/4/4, 6/6/4, 5/5/5, one 5/5/6, one 6/6/5, scattered.
    fn mixed_keys(e: usize) -> Vec<SubbankKey> {
        (0..e)
            .map(|i| match (i * 37 + i / 7) % 23 {
                0..=2 => k(4, 4, 4),
                3 => k(6, 6, 4),
                4 if i % 2 == 0 => k(5, 5, 5),
                5 if i == 101 => k(5, 5, 6),
                6 if i == 202 => k(6, 5, 4),
                _ => k(5, 5, 4),
            })
            .collect()
    }

    fn assert_partition(keys: &[SubbankKey], own: &ExpertOwnership, parts: &[(SubbankKey, ExpertOwnership)]) {
        let mut seen = vec![0u32; keys.len()];
        for w in parts.windows(2) {
            assert!(w[0].0 < w[1].0, "subbanks in key order, distinct");
        }
        for (key, o) in parts {
            assert!(o.local() > 0);
            assert!(o.owned().windows(2).all(|w| w[0] < w[1]));
            for &g in o.owned() {
                assert_eq!(keys[g as usize], *key, "expert {g} is in the subbank of its key");
                assert!(own.owns(g as usize), "expert {g} is owned");
                seen[g as usize] += 1;
            }
        }
        for (g, &n) in seen.iter().enumerate() {
            assert_eq!(n, own.owns(g) as u32, "expert {g}: in exactly one subbank iff owned");
        }
    }

    #[test]
    fn every_owned_expert_is_in_exactly_one_subbank() {
        let keys = mixed_keys(288);
        for own in [
            ExpertOwnership::all(288),
            ExpertOwnership::range(288, 0, 2).unwrap(),
            ExpertOwnership::range(288, 1, 2).unwrap(),
            ExpertOwnership::from_table(&(0..288u32).map(|e| (e * 7 + e / 5) % 3).collect::<Vec<_>>(), 2, 3).unwrap(),
            ExpertOwnership::from_ids(288, vec![5]).unwrap(),
            ExpertOwnership::from_ids(288, vec![]).unwrap(),
        ] {
            let parts = subbank_partition(&keys, &own).unwrap();
            assert_partition(&keys, &own, &parts);
        }
    }

    #[test]
    fn ownership_composes_with_the_partition() {
        // for every ranks count and table, the subbanks of all ranks together partition 0..E, and the subbanks of one key over the
        // ranks are the key's experts split by the table
        let keys = mixed_keys(288);
        for ranks in 1..=4usize {
            for table in [
                (0..288u32).map(|e| (e as usize * ranks / 288) as u32).collect::<Vec<u32>>(),
                (0..288u32).map(|e| (e * 7 + e / 5) % ranks as u32).collect(),
            ] {
                let mut seen = vec![0u32; 288];
                let mut per_key: std::collections::BTreeMap<SubbankKey, Vec<u32>> = Default::default();
                for r in 0..ranks {
                    let own = ExpertOwnership::from_table(&table, r, ranks).unwrap();
                    let parts = subbank_partition(&keys, &own).unwrap();
                    assert_partition(&keys, &own, &parts);
                    for (key, o) in &parts {
                        for &g in o.owned() {
                            seen[g as usize] += 1;
                            assert_eq!(table[g as usize] as usize, r);
                        }
                        per_key.entry(*key).or_default().extend_from_slice(o.owned());
                    }
                }
                assert!(seen.iter().all(|&c| c == 1), "R={ranks}");
                let whole = subbank_partition(&keys, &ExpertOwnership::all(288)).unwrap();
                assert_eq!(whole.len(), per_key.len());
                for (key, o) in &whole {
                    let mut v = per_key[key].clone();
                    v.sort();
                    assert_eq!(v.as_slice(), o.owned(), "key {key:?}: the ranks' subbanks are the whole subbank split");
                }
            }
        }
    }

    #[test]
    fn a_uniform_layer_is_one_subbank_equal_to_the_ownership() {
        let keys = vec![k(2, 2, 2); 288];
        let parts = subbank_partition(&keys, &ExpertOwnership::all(288)).unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].1, ExpertOwnership::all(288));
        let r1 = ExpertOwnership::range(288, 1, 2).unwrap();
        let parts = subbank_partition(&keys, &r1).unwrap();
        assert_eq!(parts, vec![(k(2, 2, 2), r1)]);
    }

    #[test]
    fn the_stacked_layout_of_one_subbank_is_the_single_banks() {
        // the single bank's layout (bind.rs / moe::from_region): 3 trellis blocks, then 6 scale stacks, each aligned
        let (d, i, e) = (4096usize, 2048usize, 288usize);
        let (lay, end) = SubbankRegion::stacked(k(3, 3, 2), e, d, i, 0, 256);
        let (tg, td) = (trellis_bytes(d, i, 3), trellis_bytes(i, d, 2));
        assert_eq!(lay.gate.trellis, 0);
        assert_eq!(lay.up.trellis, e * tg);
        assert_eq!(lay.down.trellis, 2 * e * tg);
        assert_eq!(lay.gate.suh, 2 * e * tg + e * td);
        assert_eq!(lay.down.svh + e * d * 2, end);
        assert_eq!(SubbankRegion::expert_bytes(k(3, 3, 2), d, i), 2 * tg + td + 3 * (d + i) * 2);
    }

    #[test]
    fn a_bad_shape_is_refused() {
        assert!(subbank_partition(&[k(4, 4, 4); 7], &ExpertOwnership::all(8)).is_err());
    }
}
