//! Whole-expert ownership for the routed experts of [`crate::moe`] (decode windows) and [`crate::moe_prefill`] (batched chunks): a rank
//! holds only its experts and still produces, after a sum of the ranks' pair buffers, the single-node engine's bits. New kernel:
//! `csrc/moe_owned.cu` (the pick remap and the `-0.0` filler); every expert kernel it drives is the unchanged single-node one.
//!
//! # The scheme
//!
//! * A rank's bank ([`MoeExperts::from_groups_owned`], [`MoeExperts::from_region_owned`]) holds `E_local` experts in ascending global id;
//!   `dims.experts` stays the GLOBAL count (the router, the scratches and [`crate::moe::MoeLayer::new`] compare it).
//! * The router runs replicated, so every rank has the same global picks in `s.pick` / `s.wts`. A remap turns them into local ids
//!   (`-1` for an expert this rank does not hold) and the expert kernels run with `E = E_local`: both paths already make a pick
//!   outside `[0, E)` contribute nothing at every stage.
//! * The exchange point is the pair buffer `y_pairs` `[rows * topk, hidden]` fp32 (pair `p = row * topk + slot`), each pair's expert
//!   output AFTER the down epilogue (`* svh_down` of its expert, applied by the owner) and BEFORE the weighted combine. A rank writes
//!   `-0.0` into every pair row it does not compute (absent here, or an invalid pick), so the ranks' buffers have disjoint support
//!   and their f32 sum (no flush to zero) is the single-node `y` bit for bit on every valid pair: `x + (-0.0) == x` for every `x`.
//!   No rank needs a scale of an expert it does not own.
//! * Then [`combine_pairs`], the batched path's unchanged combine (slot order, `fmaf(w, y, acc)` from `+0`, the shared expert added
//!   last), runs on every rank over the summed buffer; on the decode path it is the same chain `down_combine` fuses. Its inputs are
//!   the single-node inputs, so its output is the single-node output.
//!
//! ```text
//! let own = ExpertOwnership::range(288, rank, 2);                       // 0..144 | 144..288
//! let ex  = MoeExperts::from_region_owned(region, &lay_local, MoeDims::GLM53, own)?;
//! let mut os = OwnScratch::new(&MoeDims::GLM53, max_rows)?;
//! // per layer and step
//! router.logits(..); router.select(.., &mut s.pick, &mut s.wts, ..);    // replicated
//! ex.run_pairs(&x, Dtype::F16, rows, &mut s, &mut os, &st)?;             // or run_prefill_pairs with a PrefillScratch
//! comm.allreduce_sum_f32(&mut s.y_pairs, 0, rows * topk * hidden, &st)?; // the family's TpComm (loopback or NCCL)
//! combine_pairs(&s.y_pairs, &s.pick, &s.wts, Some(&shared_out), &mut out, rows, &ex.dims, &st)?;
//! ```
//!
//! The shared expert must NOT be in the pairs stage (it would be summed R times): it goes to [`combine_pairs`] only.

use crate::exl3_linear::Dtype;
use crate::moe::{default_plan, MoeDims, MoeExperts, MoeScratch};
use crate::moe_prefill::PrefillScratch;
use crate::rt::{CudaError, DevBuf, Stream};
use gb_exl3::checkpoint::LoadedGroup;
use gb_exl3::Codebook;

#[allow(non_snake_case)]
pub(crate) mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_moe_own_remap(
            pick: *const c_void,
            local_of: *const c_void,
            pick_local: *mut c_void,
            y: *mut c_void,
            p: i32,
            d: i32,
            e: i32,
            stream: *mut c_void,
        ) -> i32;
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
    }
}

const INVALID: CudaError = CudaError(1);

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

/// Why an ownership could not be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OwnershipError {
    /// No experts, no ranks, or a rank outside `0..ranks`.
    BadShape,
    /// An owned id is `>= experts`, repeated, or not ascending.
    BadIds,
    /// A table entry names a rank `>= ranks`, or the table length is not the expert count.
    BadTable,
}

impl std::fmt::Display for OwnershipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Which global experts one rank holds: ascending global ids; the local index of an expert is its position in `owned`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertOwnership {
    /// The global expert count (288 for GLM-5.3-Flash).
    pub experts: usize,
    owned: Vec<u32>,
}

impl ExpertOwnership {
    /// Every expert (the single-node bank).
    pub fn all(experts: usize) -> Self {
        Self { experts, owned: (0..experts as u32).collect() }
    }

    /// Rank `rank` of `ranks`: the contiguous range `[floor(E r / R), floor(E (r + 1) / R))` (docs/CLUSTER_FAMILY_PORTING.md 3.5; GLM at
    /// R = 2: 0..144 and 144..288). Every expert lands on exactly one rank and the counts differ by at most one.
    pub fn range(experts: usize, rank: usize, ranks: usize) -> Result<Self, OwnershipError> {
        if experts == 0 || ranks == 0 || rank >= ranks {
            return Err(OwnershipError::BadShape);
        }
        let (lo, hi) = (experts * rank / ranks, experts * (rank + 1) / ranks);
        Ok(Self { experts, owned: (lo as u32..hi as u32).collect() })
    }

    /// Rank `rank`'s experts of an ownership table (`table[e]` = the rank that owns global expert `e`, e.g. a balanced table of
    /// docs/NEMOTRON_TP.md section 16).
    pub fn from_table(table: &[u32], rank: usize, ranks: usize) -> Result<Self, OwnershipError> {
        if table.is_empty() || ranks == 0 || rank >= ranks {
            return Err(OwnershipError::BadShape);
        }
        if table.iter().any(|&r| r as usize >= ranks) {
            return Err(OwnershipError::BadTable);
        }
        let owned = table.iter().enumerate().filter(|(_, &r)| r as usize == rank).map(|(e, _)| e as u32).collect();
        Ok(Self { experts: table.len(), owned })
    }

    /// An explicit list of owned global ids (ascending, distinct, `< experts`).
    pub fn from_ids(experts: usize, owned: Vec<u32>) -> Result<Self, OwnershipError> {
        if experts == 0 {
            return Err(OwnershipError::BadShape);
        }
        if owned.windows(2).any(|w| w[0] >= w[1]) || owned.last().is_some_and(|&e| e as usize >= experts) {
            return Err(OwnershipError::BadIds);
        }
        Ok(Self { experts, owned })
    }

    /// The owned global ids, ascending (local index order).
    pub fn owned(&self) -> &[u32] {
        &self.owned
    }

    /// `E_local`.
    pub fn local(&self) -> usize {
        self.owned.len()
    }

    pub fn is_all(&self) -> bool {
        self.owned.len() == self.experts
    }

    pub fn owns(&self, global: usize) -> bool {
        self.owned.binary_search(&(global as u32)).is_ok()
    }

    /// `[experts]`: the local index of each global expert, `-1` where this rank does not hold it (the remap kernel's table).
    pub fn local_of(&self) -> Vec<i32> {
        let mut v = vec![-1i32; self.experts];
        for (l, &g) in self.owned.iter().enumerate() {
            v[g as usize] = l as i32;
        }
        v
    }
}

/// The device side of a rank bank's ownership.
pub(crate) struct OwnedMap {
    pub(crate) own: ExpertOwnership,
    pub(crate) local_of: DevBuf,
}

/// The extra scratch of the rank entries (the existing `MoeScratch` / `PrefillScratch` are not changed, so their byte counts in the
/// memory plan stay what they are): the local picks, and an identity table for a bank built without an ownership.
pub struct OwnScratch {
    pub max_rows: usize,
    pub(crate) dims: MoeDims,
    pub(crate) pick_local: DevBuf,
    pub(crate) identity: DevBuf,
}

impl OwnScratch {
    /// Bytes [`OwnScratch::new`] holds, without allocating.
    pub fn bytes_for(dims: &MoeDims, max_rows: usize) -> usize {
        max_rows * dims.topk * 4 + dims.experts * 4
    }

    /// For up to `max_rows` rows (1..=`moe_prefill::MAX_ROWS`; one scratch serves the decode and the batched entry).
    pub fn new(dims: &MoeDims, max_rows: usize) -> Result<Self, CudaError> {
        if max_rows == 0 || max_rows > crate::moe_prefill::MAX_ROWS || dims.experts == 0 || dims.topk == 0 {
            return Err(INVALID);
        }
        let id: Vec<i32> = (0..dims.experts as i32).collect();
        Ok(Self {
            max_rows,
            dims: *dims,
            pick_local: DevBuf::zeroed(max_rows * dims.topk * 4)?,
            identity: DevBuf::from_slice(&id)?,
        })
    }

    /// For a bank that holds EVERY expert (`from_groups` / `from_region`: a rank of a group that loaded the whole pack) but must compute
    /// only the experts `own` names: the table the pairs entries use for such a bank maps an owned global id to itself and every other id
    /// to `-1` (absent: `-0.0` in its pair rows, nothing computed), so `run_pairs` / `run_prefill_pairs` of that bank are the rank's share
    /// with the bank's `E` unchanged. A rank bank (`from_*_owned`) ignores this table and uses its own `local_of`, so one runtime serves
    /// both. Same bytes as [`OwnScratch::new`].
    pub fn masked(dims: &MoeDims, max_rows: usize, own: &ExpertOwnership) -> Result<Self, CudaError> {
        if own.experts != dims.experts {
            return Err(INVALID);
        }
        let mut s = Self::new(dims, max_rows)?;
        let table: Vec<i32> = (0..dims.experts).map(|g| if own.owns(g) { g as i32 } else { -1 }).collect();
        s.identity = DevBuf::from_slice(&table)?;
        Ok(s)
    }

    pub fn bytes(&self) -> usize {
        self.pick_local.bytes() + self.identity.bytes()
    }

    /// The local picks of the last call (`-1` = absent here). Synchronise first.
    pub fn pick_local(&self, rows: usize) -> Result<Vec<i32>, CudaError> {
        let v: Vec<i32> = self.pick_local.to_vec()?;
        Ok(v[..rows * self.dims.topk].to_vec())
    }
}

impl MoeExperts {
    /// A rank bank from the loaded groups of the OWNED experts only, in `own.owned()` order. `dims.experts` is the global count.
    pub fn from_groups_owned(
        gate: &[LoadedGroup],
        up: &[LoadedGroup],
        down: &[LoadedGroup],
        dims: MoeDims,
        own: ExpertOwnership,
    ) -> Result<Self, CudaError> {
        if own.experts != dims.experts || own.local() == 0 || gate.len() != own.local() {
            return Err(INVALID);
        }
        let mut m = Self::from_groups(gate, up, down, MoeDims { experts: own.local(), ..dims })?;
        m.adopt(dims, own)?;
        Ok(m)
    }

    /// A rank bank from ONE device region holding the owned experts only, stacked in `own.owned()` order with the layout of
    /// [`MoeExperts::from_region`] sized for `E_local` (what the loader provides). `dims.experts` is the global count.
    pub fn from_region_owned(
        region: DevBuf,
        lay: &crate::moe::ExpertRegion,
        dims: MoeDims,
        own: ExpertOwnership,
    ) -> Result<Self, CudaError> {
        if own.experts != dims.experts || own.local() == 0 {
            return Err(INVALID);
        }
        let mut m = Self::from_region(region, lay, MoeDims { experts: own.local(), ..dims })?;
        m.adopt(dims, own)?;
        Ok(m)
    }

    fn adopt(&mut self, dims: MoeDims, own: ExpertOwnership) -> Result<(), CudaError> {
        // the plans are the shape's alone, so the local bank's equal the whole bank's (checked, not assumed)
        if self.plan_gu != default_plan(dims.hidden, dims.inter, self.bits_gu, 2)?
            || self.plan_d != default_plan(dims.inter, dims.hidden, self.bits_d, 1)?
        {
            return Err(INVALID);
        }
        self.dims = dims;
        self.own = Some(OwnedMap { local_of: DevBuf::from_slice(&own.local_of())?, own });
        Ok(())
    }

    /// The ownership of a rank bank (`None`: built by `from_groups` / `from_region`, every expert).
    pub fn ownership(&self) -> Option<&ExpertOwnership> {
        self.own.as_ref().map(|o| &o.own)
    }

    /// Experts this object holds (`dims.experts` unless it is a partial rank bank).
    pub fn experts_local(&self) -> usize {
        self.own.as_ref().map_or(self.dims.experts, |o| o.own.local())
    }

    /// Whether every global expert is here (then [`MoeExperts::run`] and [`MoeExperts::run_prefill`] are allowed; a partial bank
    /// refuses them, since they would silently drop the absent experts).
    pub fn holds_every_expert(&self) -> bool {
        self.own.as_ref().is_none_or(|o| o.own.is_all())
    }

    fn local_of<'a>(&'a self, os: &'a OwnScratch) -> &'a DevBuf {
        self.own.as_ref().map_or(&os.identity, |o| &o.local_of)
    }

    /// The decode-window pairs stage of a rank (1..=`s.max_rows` rows): the routed experts this rank holds, on the global picks in
    /// `s.pick`, to `s.y_pairs` `[rows * topk, hidden]` fp32, `-0.0` in every pair row this rank does not compute. No combine, no
    /// shared expert. Clobbers `s.xg` (the throwaway target of the fused down kernel's combined rows).
    pub fn run_pairs(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        s: &mut MoeScratch,
        os: &mut OwnScratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let d = self.dims;
        let p = rows * d.topk;
        let plans = (self.plan_gu, self.plan_d);
        if rows == 0
            || rows > s.max_rows
            || rows > os.max_rows
            || os.dims.experts != d.experts
            || os.dims.topk != d.topk
            || x.bytes() < rows * d.hidden * x_dtype.size()
            || !s.fits_plans(&d, plans)
            || !self.fused_gate_up()
            || self.bands.len() > 1
            || s.xg.bytes() < rows * d.hidden * 4
        {
            return Err(INVALID);
        }
        let ((sk_gu, wk_gu), (sk_d, wk_d)) = plans;
        let el = self.experts_local();
        let umax = p.min(el);
        unsafe {
            check(ffi::gb_moe_own_remap(
                s.pick.ptr(),
                self.local_of(os).ptr(),
                os.pick_local.ptr(),
                s.y_pairs.ptr(),
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
                x.ptr(),
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
            // the single-node down epilogue, writing y of the pairs this rank holds; its combined rows (no shared) go to s.xg, unused
            check(ffi::gb_moe_down_combine(
                s.z.ptr(),
                pl,
                self.svh_d.ptr(),
                s.wts.ptr(),
                core::ptr::null(),
                s.y_pairs.ptr(),
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

    /// The batched (prefill chunk) pairs stage of a rank (1..=`s.max_rows` rows): as [`MoeExperts::run_pairs`], to `s.y_pairs`, through
    /// the unchanged sorted-pair kernels of [`MoeExperts::run_prefill`].
    pub fn run_prefill_pairs(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        s: &mut PrefillScratch,
        os: &mut OwnScratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let d = self.dims;
        let p = rows * d.topk;
        if rows == 0
            || rows > s.max_rows
            || rows > os.max_rows
            || !s.fits(&d)
            || !self.fused_gate_up()
            || self.bands.len() > 1
            || os.dims.experts != d.experts
            || os.dims.topk != d.topk
            || x.bytes() < rows * d.hidden * x_dtype.size()
        {
            return Err(INVALID);
        }
        let (sk_gu, wk_gu) = self.plan_gu;
        let (sk_d, wk_d) = self.plan_d;
        let cb = codebook_code(self.codebook);
        let el = self.experts_local();
        unsafe {
            check(ffi::gb_moe_own_remap(
                s.pick.ptr(),
                self.local_of(os).ptr(),
                os.pick_local.ptr(),
                s.y_pairs.ptr(),
                p as i32,
                d.hidden as i32,
                d.experts as i32,
                st.raw(),
            ))?;
            let pl = os.pick_local.ptr();
            // the scratch's planner (the single block, or the bit-identical multi-block plan the family selected)
            s.run_plan(pl, rows, el as i32, st)?;
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

/// The unchanged slot combine over a (summed) pair buffer, on every rank: `out` `[rows, hidden]` fp32 = over the row's slots in slot
/// order `acc = fmaf(w, y, acc)` from `+0` (a pick outside `[0, dims.experts)` skipped), then `shared` `[rows, hidden]` fp32 added last.
/// `y` `[rows * topk, hidden]` fp32, `pick` the GLOBAL picks `[rows * topk]` i32, `wts` fp16 `[rows * topk]`. Any row count up to
/// `moe_prefill::MAX_ROWS`.
#[allow(clippy::too_many_arguments)]
pub fn combine_pairs(
    y: &DevBuf,
    pick: &DevBuf,
    wts: &DevBuf,
    shared: Option<&DevBuf>,
    out: &mut DevBuf,
    rows: usize,
    dims: &MoeDims,
    st: &Stream,
) -> Result<(), CudaError> {
    let p = rows * dims.topk;
    if rows == 0
        || y.bytes() < p * dims.hidden * 4
        || pick.bytes() < p * 4
        || wts.bytes() < p * 2
        || out.bytes() < rows * dims.hidden * 4
        || shared.is_some_and(|b| b.bytes() < rows * dims.hidden * 4)
    {
        return Err(INVALID);
    }
    unsafe {
        check(ffi::gb_moe_pf_combine(
            y.ptr(),
            pick.ptr(),
            wts.ptr(),
            shared.map_or(core::ptr::null(), |b| b.ptr() as *const _),
            out.ptr(),
            rows as i32,
            dims.topk as i32,
            dims.hidden as i32,
            dims.experts as i32,
            st.raw(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_puts_every_expert_on_exactly_one_rank_with_skew_at_most_one() {
        for e in [1usize, 7, 64, 256, 288, 289, 512] {
            for r in 1..=8usize {
                if r > e {
                    continue;
                }
                let parts: Vec<ExpertOwnership> = (0..r).map(|k| ExpertOwnership::range(e, k, r).unwrap()).collect();
                let mut seen = vec![0u32; e];
                for p in &parts {
                    for &g in p.owned() {
                        seen[g as usize] += 1;
                    }
                    assert!(p.owned().windows(2).all(|w| w[0] < w[1]));
                }
                assert!(seen.iter().all(|&c| c == 1), "E={e} R={r}");
                let (mn, mx) = (parts.iter().map(|p| p.local()).min().unwrap(), parts.iter().map(|p| p.local()).max().unwrap());
                assert!(mx - mn <= 1, "E={e} R={r}: {mn}..{mx}");
            }
        }
    }

    #[test]
    fn glm_two_ranks_split_at_144() {
        let a = ExpertOwnership::range(288, 0, 2).unwrap();
        let b = ExpertOwnership::range(288, 1, 2).unwrap();
        assert_eq!(a.owned(), (0..144).collect::<Vec<u32>>().as_slice());
        assert_eq!(b.owned(), (144..288).collect::<Vec<u32>>().as_slice());
        assert_eq!(b.local_of()[144], 0);
        assert_eq!(b.local_of()[143], -1);
        assert_eq!(a.local_of()[143], 143);
        assert_eq!(a.local_of()[144], -1);
    }

    #[test]
    fn one_rank_is_the_identity() {
        let o = ExpertOwnership::range(288, 0, 1).unwrap();
        assert_eq!(o, ExpertOwnership::all(288));
        assert!(o.is_all());
        assert_eq!(o.local_of(), (0..288).collect::<Vec<i32>>());
    }

    #[test]
    fn local_of_round_trips_for_an_arbitrary_table() {
        // a scattered table over 3 ranks
        let table: Vec<u32> = (0..288u32).map(|e| (e * 7 + e / 5) % 3).collect();
        let parts: Vec<ExpertOwnership> = (0..3).map(|r| ExpertOwnership::from_table(&table, r, 3).unwrap()).collect();
        for (r, p) in parts.iter().enumerate() {
            let lo = p.local_of();
            for (g, &l) in lo.iter().enumerate() {
                if table[g] as usize == r {
                    assert!(l >= 0 && p.owned()[l as usize] as usize == g);
                    assert!(p.owns(g));
                } else {
                    assert_eq!(l, -1);
                    assert!(!p.owns(g));
                }
            }
        }
        assert_eq!(parts.iter().map(|p| p.local()).sum::<usize>(), 288);
    }

    #[test]
    fn bad_inputs_are_refused() {
        assert_eq!(ExpertOwnership::range(0, 0, 1), Err(OwnershipError::BadShape));
        assert_eq!(ExpertOwnership::range(8, 2, 2), Err(OwnershipError::BadShape));
        assert_eq!(ExpertOwnership::from_table(&[0, 1, 2], 0, 2), Err(OwnershipError::BadTable));
        assert_eq!(ExpertOwnership::from_ids(8, vec![3, 3]), Err(OwnershipError::BadIds));
        assert_eq!(ExpertOwnership::from_ids(8, vec![4, 2]), Err(OwnershipError::BadIds));
        assert_eq!(ExpertOwnership::from_ids(8, vec![8]), Err(OwnershipError::BadIds));
        assert!(ExpertOwnership::from_ids(8, vec![0, 5, 7]).is_ok());
    }
}
