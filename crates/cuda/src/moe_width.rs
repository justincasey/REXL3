//! The WIDTH SPLIT of a layer's routed experts over the ranks of a tensor-parallel group: rank `r` of `R` holds, for EVERY expert, the
//! intermediate columns `[r I / R, (r + 1) I / R)` (gate and up: those output columns, as whole 128-column trellis strips, and their
//! `svh`; down: those input rows of the trellis and their `suh`; the model-width scales `suh` of gate/up and `svh` of down whole), in
//! place of whole experts ([`crate::moe_owned`]). Every rank then runs every routed pair of a window at its share of the width, so the
//! ranks read the same bytes a token (whole experts leave the slower rank about 5 of the 8 a token reads; it waits at every exchange).
//!
//! # Exactness: the single node's bits (decode windows and the batched path)
//!
//! * Gate/up. A column's arithmetic is its own: the K sum over the model width runs whole on every rank, with the WHOLE width's plan
//!   (`default_plan(hidden, inter, ..)`: its K split and warps, at the rank's columns), and the epilogue's Hadamard rotation works on
//!   128-column blocks, of which a rank's columns are whole ones. So the rank's down input columns `xd` are the single node's.
//! * Down. The single node's plan cuts K = `inter` into exactly R splits (`default_plan(inter, hidden, ..)`: 2 for GLM-5.3-Flash at every
//!   width), and split `r` is the rank's rows. The rank runs ONE split with the same k tiles a warp in the same order (its plan is
//!   `(1, wk)` at K = `inter / R`), so its fp32 partial is the single node's `Z[split r]` bit for bit ([`DownOut::Partial`]: no epilogue).
//!   The exchange (the group's all-reduce over the pair rows: two operands, `add.rn` in rank order or NCCL's two-rank sum, no flush to
//!   zero) leaves `Z[0] + Z[1]` on both ranks: the single node's `s = Z[0]; s += Z[1]` (fp32 addition commutes).
//! * Then [`MoeSubbanks::finish`] runs the single node's down epilogue and slot combine on the sum (`down_combine` with each expert's
//!   `svh_down` found by global id, one split), the shared expert added last: the single node's output, on every rank.
//!
//! The exchange carries the pair rows `[rows * topk, hidden]` f32 as the whole-expert split does. [`width_plans`] refuses a shape whose
//! down plan does not have exactly R splits (a rank's partial would then be a sum of splits, and the exchange's sum another order).
//!
//! # P2 prefill (`glm_p2_moe`)
//!
//! A rank runs the P2 expert GEMMs on its half of every expert: gate/up at its columns (the column's arithmetic is the one node's), down
//! on its K rows with the P2 order of a K of `inter / R` and the down epilogue on that half's sum: `y_r` per pair. Its part is the slot
//! chain over EVERY slot (`CombineMode::All`), the ranks swap parts as before, and both add `(part_0 + part_1) + shared`. Every P2
//! invariant holds (a pair's bits are a function of its row and its expert's weights; nothing depends on the chunk), but the bits are not
//! the one node's P2 (that sums the two expert groups of the whole-expert split): [`P2_WIDTH_ID`] names them.
//!
//! ```text
//! let share = WidthShare::new(rank, 2)?;
//! let sb = MoeSubbanks::new_width(dims, share, banks)?;     // banks: MoeExperts::from_region_width per subbank (every expert, its key)
//! sb.run_pairs(&x, Dtype::F16, rows, &mut s, &mut os, &st)?; // or run_prefill_pairs_windowed: the K-split partial of every pair
//! comm.allreduce_sum_f32(&mut s.y_pairs, 0, rows * topk * hidden, &st)?;
//! sb.finish(&s.y_pairs, &s.pick, &s.wts, shared, &mut out, rows, &st)?;
//! ```

use crate::exl3_linear::Dtype;
use crate::moe::{default_plan, MoeDims, MoeExperts, MoeScratch};
use crate::moe_owned::{combine_pairs, ffi, ExpertOwnership, OwnScratch};
use crate::moe_subbanks::{
    ffi_multi, ffi_width, subbank_partition, upload_subbank, DownOut, MoeSubbanks, MultiTables, SubbankKey, SubbankRegion,
};
use crate::rt::{CudaError, DevBuf, Stream};
use gb_exl3::checkpoint::LoadedGroup;
use gb_exl3::Codebook;

const INVALID: CudaError = CudaError(1);

/// The tag of the P2 arithmetic of a width share (its parts are the ranks' halves of the width, not the whole-expert split's groups):
/// a P2 rank of the width split names it beside `glm_p2_moe::ARITH_ID`.
pub const P2_WIDTH_ID: &str = "width2";

/// Rank `rank` of `ranks` holds the intermediate columns `[rank * I / ranks, (rank + 1) * I / ranks)` of every routed expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WidthShare {
    pub rank: usize,
    pub ranks: usize,
}

impl WidthShare {
    pub fn new(rank: usize, ranks: usize) -> Result<Self, CudaError> {
        if ranks < 2 || rank >= ranks {
            return Err(INVALID);
        }
        Ok(Self { rank, ranks })
    }

    /// The rank's columns `[c0, c1)` of an expert width `inter`: equal shares of whole 128-column blocks (`None` when `inter` does not
    /// cut so).
    pub fn cols(&self, inter: usize) -> Option<(usize, usize)> {
        (inter % (128 * self.ranks) == 0).then(|| (inter * self.rank / self.ranks, inter * (self.rank + 1) / self.ranks))
    }

    /// The dims of the rank's banks: the layer's with the share's width.
    pub fn local_dims(&self, d: &MoeDims) -> Option<MoeDims> {
        let (c0, c1) = self.cols(d.inter)?;
        Some(MoeDims { inter: c1 - c0, ..*d })
    }
}

/// The plans a width share's bank runs, for a layer of `d` (the whole width): gate/up the whole width's (K split and warps; a column's
/// sum is the same at any column count), down ONE split of the whole width's (`(1, wk)` at K = `inter / ranks`). Refused unless the
/// whole width's down plan has exactly `ranks` splits (then split r is rank r's rows, and the exchange's two-operand sum is the single
/// node's split sum).
pub fn width_plans(
    d: &MoeDims,
    share: WidthShare,
    bits_gu: usize,
    bits_d: usize,
) -> Result<((usize, usize), (usize, usize)), CudaError> {
    share.cols(d.inter).ok_or(INVALID)?;
    let gu = default_plan(d.hidden, d.inter, bits_gu, 2)?;
    let (sk, wk) = default_plan(d.inter, d.hidden, bits_d, 1)?;
    if sk != share.ranks {
        return Err(INVALID);
    }
    Ok((gu, (1, wk)))
}

/// Output columns `[c0, c1)` of a loaded group (gate/up of a width share): the trellis tiles of those columns in every K-tile row, `svh`
/// (and a bias) cut with them, `suh` whole. Whole 128-column blocks only.
pub fn group_cols(g: &LoadedGroup, c0: usize, c1: usize) -> Option<LoadedGroup> {
    let (k, n) = (g.group.k, g.group.n);
    if c0 % 128 != 0 || c1 % 128 != 0 || c0 >= c1 || c1 > n || g.svh.len() != n {
        return None;
    }
    let (nt, tw) = (n / 16, g.trellis.len() / ((k / 16) * (n / 16)));
    let mut trellis = Vec::with_capacity((k / 16) * (c1 - c0) / 16 * tw);
    for kk in 0..k / 16 {
        trellis.extend_from_slice(&g.trellis[(kk * nt + c0 / 16) * tw..(kk * nt + c1 / 16) * tw]);
    }
    let mut out = g.clone();
    out.group.n = c1 - c0;
    out.nt = (c1 - c0) / 16;
    out.trellis = trellis;
    out.svh = g.svh[c0..c1].to_vec();
    out.bias = g.bias.as_ref().map(|b| b[c0..c1].to_vec());
    Some(out)
}

/// Input rows `[k0, k1)` of a loaded group (down of a width share): those K-tile rows of the trellis and `suh`, `svh` whole. Whole
/// 128-row blocks only.
pub fn group_rows(g: &LoadedGroup, k0: usize, k1: usize) -> Option<LoadedGroup> {
    let (k, n) = (g.group.k, g.group.n);
    if k0 % 128 != 0 || k1 % 128 != 0 || k0 >= k1 || k1 > k || g.suh.len() != k {
        return None;
    }
    let row = g.trellis.len() / (k / 16);
    let mut out = g.clone();
    out.group.k = k1 - k0;
    out.kt = (k1 - k0) / 16;
    out.trellis = g.trellis[k0 / 16 * row..k1 / 16 * row].to_vec();
    out.suh = g.suh[k0..k1].to_vec();
    debug_assert_eq!(out.trellis.len(), (k1 - k0) / 16 * (n / 16) * (row / (n / 16)));
    Some(out)
}

impl MoeExperts {
    /// One subbank of a width share from a region the loader filled with its experts' shares (`lay`: [`SubbankRegion::stacked`] at the
    /// share's width; `own`: the subbank's ids), for a layer of `full` dims: the bank runs at the share's dims with [`width_plans`].
    pub fn from_region_width(
        region: DevBuf,
        lay: &SubbankRegion,
        full: MoeDims,
        share: WidthShare,
        own: ExpertOwnership,
    ) -> Result<Self, CudaError> {
        let local = share.local_dims(&full).ok_or(INVALID)?;
        let mut m = Self::from_region_subbank(region, lay, local, own)?;
        let (gu, d) = width_plans(&full, share, m.bits_gu.max(m.bits_u), m.bits_d)?;
        m.plan_gu = gu;
        m.plan_d = d;
        for b in &mut m.bands {
            b.plan_gu = gu;
            b.plan_d = d;
        }
        Ok(m)
    }
}

impl MoeSubbanks {
    /// A rank's width share of a layer of `full` dims: `banks` (each [`MoeExperts::from_region_width`]) hold every expert exactly once,
    /// at the share's width. `dims` stays the layer's (the router, the scratches and the combine see the whole layer).
    pub fn new_width(full: MoeDims, share: WidthShare, banks: Vec<MoeExperts>) -> Result<Self, CudaError> {
        let local = share.local_dims(&full).ok_or(INVALID)?;
        if banks.is_empty() {
            return Err(INVALID);
        }
        let (e, h) = (full.experts, full.hidden);
        let mut seen = vec![false; e];
        let mut svh = vec![0u64; e];
        for b in &banks {
            let o = b.own.as_ref().ok_or(INVALID)?;
            if b.dims != local
                || o.own.experts != e
                || (b.plan_gu, b.plan_d) != width_plans(&full, share, b.bits_gu.max(b.bits_u), b.bits_d)?
            {
                return Err(INVALID);
            }
            for (l, &g) in o.own.owned().iter().enumerate() {
                if std::mem::replace(&mut seen[g as usize], true) {
                    return Err(INVALID);
                }
                svh[g as usize] = b.svh_d.ptr() as u64 + (l * h * 2) as u64;
            }
        }
        if seen.iter().any(|&v| !v) {
            return Err(INVALID);
        }
        let multi = MultiTables::of(local, &banks)?;
        Ok(Self {
            dims: full,
            own: ExpertOwnership::all(e),
            banks,
            multi,
            width: Some(share),
            svh_tab: Some(DevBuf::from_slice(&svh)?),
        })
    }

    /// A width share from every expert's whole loaded groups (global id order): each expert cut to the share ([`group_cols`] for gate
    /// and up, [`group_rows`] for down), partitioned by key and uploaded per subbank as the loader lays it out.
    pub fn from_groups_width(
        gate: &[LoadedGroup],
        up: &[LoadedGroup],
        down: &[LoadedGroup],
        full: MoeDims,
        share: WidthShare,
    ) -> Result<Self, CudaError> {
        let e = full.experts;
        let (c0, c1) = share.cols(full.inter).ok_or(INVALID)?;
        if gate.len() != e || up.len() != e || down.len() != e {
            return Err(INVALID);
        }
        let cut = |v: &[LoadedGroup], f: fn(&LoadedGroup, usize, usize) -> Option<LoadedGroup>| -> Result<Vec<LoadedGroup>, CudaError> {
            v.iter().map(|g| f(g, c0, c1).ok_or(INVALID)).collect()
        };
        let (gate, up, down) = (cut(gate, group_cols)?, cut(up, group_cols)?, cut(down, group_rows)?);
        let mut keys = vec![SubbankKey { bits_gate: 0, bits_up: 0, bits_down: 0, codebook: Codebook::Mul1 }; e];
        for (g, k) in keys.iter_mut().enumerate() {
            *k = SubbankKey::of_groups(&gate[g], &up[g], &down[g]).ok_or(INVALID)?;
        }
        let parts = subbank_partition(&keys, &ExpertOwnership::all(e)).map_err(|_| INVALID)?;
        let mut banks = Vec::with_capacity(parts.len());
        for (key, o) in parts {
            let (region, lay) = upload_subbank(&gate, &up, &down, key, &o, full.hidden, c1 - c0)?;
            banks.push(MoeExperts::from_region_width(region, &lay, full, share, o)?);
        }
        Self::new_width(full, share, banks)
    }

    /// The decode pairs stage of a width share: every valid pair's K-split partial (this rank's split) in `s.y_pairs`, unwritten rows
    /// for an invalid pick (the finish skips them). One launch a stage over every subbank ([`MultiTables`]), or each subbank's chain.
    pub(crate) fn width_pairs(
        &self,
        x: &DevBuf,
        x_dtype: Dtype,
        rows: usize,
        s: &mut MoeScratch,
        os: &mut OwnScratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let Some(m) = self.multi() else {
            for b in &self.banks {
                b.subbank_pairs(x, x_dtype, rows, s, os, false, DownOut::Partial, st)?;
            }
            return Ok(());
        };
        let d = self.dims;
        let local = self.banks[0].dims;
        let p = rows * d.topk;
        let ((sk_gu, wk_gu), (sk_d, wk_d)) = (m.plan_gu, m.plan_d);
        if rows == 0
            || rows > s.max_rows
            || sk_d != 1
            || !s.fits_plans(&local, (m.plan_gu, m.plan_d))
            || s.y_pairs.bytes() < p * d.hidden * 4
            || x.bytes() < rows * d.hidden * x_dtype.size()
        {
            return Err(INVALID);
        }
        let (e, l) = (d.experts as i32, local.inter as i32);
        let umax = p.min(d.experts) as i32;
        let (r, k, pi) = (rows as i32, d.topk as i32, p as i32);
        let pick = s.pick.ptr();
        // SAFETY: every buffer was checked against these shapes above (the scratch is the layer's, wider than the share's banks)
        unsafe {
            check(ffi::gb_moe_group(pick, s.uids.ptr(), s.ucount.ptr(), s.mcount.ptr(), s.members.ptr(), r, k, e, st.raw()))?;
            check(ffi_multi::gb_moe_rot_in_multi(
                x.ptr(),
                x_dtype as i32,
                pick,
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
                l,
                pi,
                m.widths_gu,
                m.codebook,
                sk_gu as i32,
                wk_gu as i32,
                umax,
                st.raw(),
            ))?;
            check(ffi_multi::gb_moe_gateup_epilogue_multi(s.z.ptr(), pick, m.epi.ptr(), s.xd.ptr(), r, k, l, sk_gu as i32, e, d.limit, st.raw()))?;
            // one K split: its partials [1][P][hidden] are the pair rows' layout, written straight into them
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
                s.y_pairs.ptr(),
                1,
                l,
                d.hidden as i32,
                pi,
                m.widths_d,
                m.codebook,
                1,
                wk_d as i32,
                umax,
                st.raw(),
            ))
        }
    }

    /// The layer's output after the exchange, into `out` `[rows, hidden]` f32: whole experts, the slot combine over the summed pair
    /// buffer `y` ([`combine_pairs`]); a width share, the down epilogue on the summed K-split partials in `y` and the same combine (the
    /// single node's `down_combine`, one split, each expert's `svh_down` by global id). `pick` / `wts` the global picks and weights,
    /// `shared` added last.
    #[allow(clippy::too_many_arguments)]
    pub fn finish(
        &self,
        y: &DevBuf,
        pick: &DevBuf,
        wts: &DevBuf,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        rows: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        match self.svh_tab.as_ref().filter(|_| self.width.is_some()) {
            None => combine_pairs(y, pick, wts, shared, out, rows, &self.dims, st),
            Some(_) => self.finish_width(y, None, pick, wts, shared, out, rows, st),
        }
    }

    /// [`MoeSubbanks::finish`] of a width share on rows whose pairs' K-split partials are this rank's in `y` and the peer's in `peer`
    /// (the same pair rows, `[rows * topk, hidden]` f32): each pair's sum `y + peer` is the exchange's exact two-operand add (fp32
    /// addition commutes, so either rank's order), made on the rows this rank finishes instead of on every row by an all-reduce. The
    /// caller passes views of those rows' pair rows, picks, weights, shared rows and output rows.
    #[allow(clippy::too_many_arguments)]
    pub fn finish_sum(
        &self,
        y: &DevBuf,
        peer: &DevBuf,
        pick: &DevBuf,
        wts: &DevBuf,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        rows: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if self.width.is_none() || peer.bytes() < rows * self.dims.topk * self.dims.hidden * 4 {
            return Err(INVALID);
        }
        self.finish_width(y, Some(peer), pick, wts, shared, out, rows, st)
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_width(
        &self,
        y: &DevBuf,
        peer: Option<&DevBuf>,
        pick: &DevBuf,
        wts: &DevBuf,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        rows: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let d = self.dims;
        let tab = self.svh_tab.as_ref().ok_or(INVALID)?;
        let p = rows * d.topk;
        if rows == 0
            || y.bytes() < p * d.hidden * 4
            || pick.bytes() < p * 4
            || wts.bytes() < p * 2
            || out.bytes() < rows * d.hidden * 4
            || shared.is_some_and(|b| b.bytes() < rows * d.hidden * 4)
        {
            return Err(INVALID);
        }
        // SAFETY: the sizes were checked above (`peer` by the caller); `tab` holds an address for every global id
        unsafe {
            check(ffi_width::gb_moe_down_combine_multi(
                y.ptr(),
                peer.map_or(core::ptr::null(), |b| b.ptr() as *const _),
                pick.ptr(),
                tab.ptr(),
                wts.ptr(),
                shared.map_or(core::ptr::null(), |b| b.ptr() as *const _),
                core::ptr::null_mut(),
                out.ptr(),
                rows as i32,
                d.topk as i32,
                d.hidden as i32,
                1,
                d.experts as i32,
                st.raw(),
            ))
        }
    }
}

fn check(rc: i32) -> Result<(), CudaError> {
    CudaError::check(rc)
}
