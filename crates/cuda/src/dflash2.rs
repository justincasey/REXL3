//! DFlash2 draft kernels: grouped dynamic conv, context K/V ring, draft attention and the candidate-selector walk
//! (docs/QWEN38_SPEC.md section 8), plus the draft context-cache layout helper and `cpu`, a pure-Rust reference of the
//! whole draft block and the walk.
//!
//! # What the model runtime calls between these kernels
//!
//! The kernels here are only the draft-specific pieces. One draft round for `n_seq` sequences is (shared kernels in
//! *italics*, `[EXL3]` = the EXL3 linear):
//!
//! ```text
//! once per round, for the rows the target committed last round (m+1 tap rows per sequence):
//!   [EXL3] fc (25600 -> 5120) on concat(taps)        *rmsnorm*(hidden_norm, plain gamma)          -> ctx_c
//!   per draft layer l:  [EXL3] k_proj, v_proj on ctx_c  ->  dflash2::ctx_kv(kproj, vproj, k_norm_l, positions, slots, ring_l)
//! block:  x = *embedding*([anchor, MASK x 7]) (target table), per sequence 8 rows
//!   per draft layer l:
//!     h   = *rmsnorm*(x, input_layernorm_l)                       (plain gamma)
//!     dyn = *small_linear_f16w*(h, attention_conv.kernel_projection)  [8, 1280]   (bf16 out, layout [prepare/finish][tap][group])
//!     y   = dflash2::dynconv(h, dyn, base_attn, which = 0)
//!     q,k,v = [EXL3] q_proj / k_proj / v_proj (y)
//!     *attn_prep*: q <- rope128(rmsnorm_plain(q, q_norm)), k <- rope128(rmsnorm_plain(k, k_norm)), positions P..P+7; v unchanged
//!     att = dflash2::attn(q, k, v, ring_l, slots, P)              [8, 4096]
//!     o   = [EXL3] o_proj (att)
//!     x'  = dflash2::dynconv(o, dyn, base_attn, which = 1, resid = Some(x))       (optional fusion: x' = bf16(x + conv), one rounding, residual kept in bf16 (EX keeps fp32); or resid = None + a separate *add*)
//!     h   = *rmsnorm*(x', post_attention_layernorm_l)
//!     dyn = *small_linear_f16w*(h, mlp_conv.kernel_projection);  y = dflash2::dynconv(h, dyn, base_mlp, 0)
//!     m   = [EXL3] down( *swiglu*([EXL3] gate(y), [EXL3] up(y)) )
//!     x'' = dflash2::dynconv(m, dyn, base_mlp, 1, resid = Some(x'))
//!   state = *rmsnorm*(x, norm)                                    [n_seq*8, 5120]
//!   sel   = *copy_rows*(state, rows t = 1..7 of every sequence)   [n_seq*7, 5120]   (rows 1..7 are not contiguous once n_seq > 1)
//!   logits = [EXL3] lm_head(sel)                                  (target head, rows 1..7 only)
//!   (ids, vals) = *topk*(logits, count = 16)                      value descending, ties lowest id; ids i32, vals f32
//!   gate = *small_linear_f16w*(sel, candidate_selector.hidden_projection)   [n_seq*7, 256]  (bf16 or f32)
//!   out = dflash2::selector_walk(anchor, ids, vals, gate, pred_codebook, succ_codebook)  -> d_1..d_7 per sequence
//! ```
//!
//! The in-block K/V of the current block never touch the ring (they are separate buffers), so the ring holds only
//! committed context and there is nothing to "overwrite" the next round (EX writes the transient block K/V into its
//! cache and overwrites it later; the result is the same).
//!
//! # Draft context cache layout
//!
//! Per draft layer, two buffers (K and V), each `[slots, cap, kv_heads, head_dim]` bf16. Position `pos` of slot `s`
//! lives at `(s * cap + pos % cap)`. `cap = 2560` (>= window 2048 + slack). The ring must contain every committed
//! position in `[max(0, P - 2048), P)` when the block at position `P` is attended; older entries are dead and are
//! overwritten in place. No clearing is needed when a slot changes owner: the attention never reads a position
//! `>= P` or `< P - 2048`, and the new owner wrote every position in that range. Rows of one slot written in one
//! launch must span fewer than `cap` consecutive positions (on a long prefill write only the last `cap` rows, or
//! better only `ctx_window_start(P)..P`).
//!
//! Per-slot capacity: `5 layers x 2 (K,V) x 2560 x 8 x 128 x 2 B = 52,428,800 B` (50 MiB).

pub use cpu::Dims;

/// Layout and arithmetic of the per-slot draft context ring (see the module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftCtxLayout {
    pub layers: usize,
    pub slots: usize,
    pub cap: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub window: usize,
}

impl DraftCtxLayout {
    pub const DEFAULT_CAP: usize = 2560;
    pub const WINDOW: usize = 2048;

    /// Qwen3.8 draft: 5 layers, 8 kv heads x 128, cap 2560, window 2048.
    pub fn qwen38(slots: usize) -> Self {
        Self { layers: 5, slots, cap: Self::DEFAULT_CAP, kv_heads: 8, head_dim: 128, window: Self::WINDOW }
    }
    /// bf16 elements of one layer's K (or V) buffer for one slot.
    pub fn elems_per_slot_layer(&self) -> usize {
        self.cap * self.kv_heads * self.head_dim
    }
    /// Bytes of one slot across all layers, K and V.
    pub fn bytes_per_slot(&self) -> usize {
        self.layers * 2 * self.elems_per_slot_layer() * 2
    }
    /// Bytes of one layer's K (or V) buffer for all slots.
    pub fn bytes_per_layer_buffer(&self) -> usize {
        self.slots * self.elems_per_slot_layer() * 2
    }
    pub fn total_bytes(&self) -> usize {
        self.slots * self.bytes_per_slot()
    }
    /// Ring index of an absolute position.
    pub fn ring_index(&self, pos: usize) -> usize {
        pos % self.cap
    }
    /// First position whose K/V the block at position `p` can see (row 0 sees `p - window`).
    pub fn ctx_window_start(&self, p: usize) -> usize {
        p.saturating_sub(self.window)
    }
    /// Element offset (bf16) of (slot, pos, kv head 0) inside a layer buffer.
    pub fn elem_offset(&self, slot: usize, pos: usize) -> usize {
        (slot * self.cap + self.ring_index(pos)) * self.kv_heads * self.head_dim
    }
    /// True when the ring can hold the window plus `extra` positions written ahead without clobbering a visible key.
    pub fn fits(&self, extra: usize) -> bool {
        self.cap >= self.window + extra
    }
}

pub use gpu::*;

mod gpu {
    use super::DraftCtxLayout;
    use crate::{CudaError, DevBuf, Stream};

    pub mod ffi {
        use core::ffi::c_void;
        crate::launchers! {
            pub fn gb_dflash_dynconv(
                u: *const c_void,
                dyn_: *const c_void,
                base: *const c_void,
                which: i32,
                n_seq: i32,
                rows: i32,
                hidden: i32,
                group: i32,
                resid: *const c_void,
                out: *mut c_void,
                stream: *mut c_void,
            ) -> i32;
            pub fn gb_dflash_ctx_kv(
                kproj: *const c_void,
                vproj: *const c_void,
                gamma: *const c_void,
                positions: *const c_void,
                slots: *const c_void,
                n: i32,
                kv_heads: i32,
                cap: i32,
                eps: f32,
                rope_theta: f64,
                kc: *mut c_void,
                vc: *mut c_void,
                stream: *mut c_void,
            ) -> i32;
            pub fn gb_dflash_attn_workspace_bytes(n_seq: i32, window: i32, n_q_heads: i32, rows: i32) -> usize;
            pub fn gb_dflash_attn(
                q: *const c_void,
                block_k: *const c_void,
                block_v: *const c_void,
                kc: *const c_void,
                vc: *const c_void,
                slots: *const c_void,
                p: *const c_void,
                n_seq: i32,
                n_q_heads: i32,
                n_kv_heads: i32,
                cap: i32,
                window: i32,
                rows: i32,
                ws: *mut c_void,
                out: *mut c_void,
                stream: *mut c_void,
            ) -> i32;
            pub fn gb_dflash_selector_walk(
                anchor: *const c_void,
                topk_ids: *const c_void,
                topk_vals: *const c_void,
                gate: *const c_void,
                gate_is_f32: i32,
                pred_codebook: *const c_void,
                succ_codebook: *const c_void,
                n_seq: i32,
                rows: i32,
                k: i32,
                rank: i32,
                out_tokens: *mut c_void,
                conf: *mut c_void,
                stream: *mut c_void,
            ) -> i32;
            pub fn gb_dflash_selector_walk_keyed(
                anchor: *const c_void,
                topk_ids: *const c_void,
                topk_vals: *const c_void,
                gate: *const c_void,
                gate_is_f32: i32,
                pred_codebook: *const c_void,
                succ_codebook: *const c_void,
                n_seq: i32,
                rows: i32,
                k: i32,
                rank: i32,
                sampling: *const c_void,
                out_tokens: *mut c_void,
                conf: *mut c_void,
                stream: *mut c_void,
            ) -> i32;
            pub fn gb_dflash_selector_walk_ragged(
                anchor: *const c_void,
                topk_ids: *const c_void,
                topk_vals: *const c_void,
                gate: *const c_void,
                gate_is_f32: i32,
                pred_codebook: *const c_void,
                succ_codebook: *const c_void,
                n_seq: i32,
                k: i32,
                rank: i32,
                sampling: *const c_void,
                row_off: *const c_void,
                out_tokens: *mut c_void,
                conf: *mut c_void,
                margin: *mut c_void,
                stream: *mut c_void,
            ) -> i32;
            pub fn gb_dflash_selector_edges(
                anchor: *const c_void,
                topk_ids: *const c_void,
                gate: *const c_void,
                gate_is_f32: i32,
                pred_codebook: *const c_void,
                succ_codebook: *const c_void,
                n_seq: i32,
                total_rows: i32,
                k: i32,
                rank: i32,
                row_off: *const c_void,
                edge: *mut c_void,
                stream: *mut c_void,
            ) -> i32;
            pub fn gb_dflash_remap_ids(
                ids: *mut c_void,
                count: i32,
                n_ranges: i32,
                cum: *const i32,
                base: *const i32,
                stream: *mut c_void,
            ) -> i32;
            pub fn gb_dflash_merge_topk(
                ia: *const c_void,
                va: *const c_void,
                ib: *const c_void,
                vb: *const c_void,
                io: *mut c_void,
                vo: *mut c_void,
                rows: i32,
                k: i32,
                stream: *mut c_void,
            ) -> i32;
        }
    }

    /// Device-side draft context cache: per layer a K and a V ring buffer for all slots (layout in the module docs).
    pub struct DraftCache {
        pub layout: DraftCtxLayout,
        k: Vec<DevBuf>,
        v: Vec<DevBuf>,
    }

    impl DraftCache {
        pub fn new(layout: DraftCtxLayout) -> Result<Self, CudaError> {
            let mut k = Vec::with_capacity(layout.layers);
            let mut v = Vec::with_capacity(layout.layers);
            for _ in 0..layout.layers {
                k.push(DevBuf::zeroed(layout.bytes_per_layer_buffer())?);
                v.push(DevBuf::zeroed(layout.bytes_per_layer_buffer())?);
            }
            Ok(Self { layout, k, v })
        }
        pub fn k(&self, layer: usize) -> &DevBuf {
            &self.k[layer]
        }
        /// Both buffers of `layer` for writing: K, V.
        pub fn kv_mut(&mut self, layer: usize) -> (&mut DevBuf, &mut DevBuf) {
            (&mut self.k[layer], &mut self.v[layer])
        }
        pub fn v(&self, layer: usize) -> &DevBuf {
            &self.v[layer]
        }
        /// Write `n` committed context rows of draft layer `layer` (see [`ctx_kv`]).
        #[allow(clippy::too_many_arguments)]
        pub fn write_ctx(
            &mut self,
            layer: usize,
            kproj: &DevBuf,
            vproj: &DevBuf,
            k_norm: &DevBuf,
            positions: &DevBuf,
            slots: &DevBuf,
            n: usize,
            eps: f32,
            rope_theta: f64,
            stream: &Stream,
        ) -> Result<(), CudaError> {
            let l = self.layout;
            let (kc, vc) = (&mut self.k[layer], &mut self.v[layer]);
            ctx_kv(kproj, vproj, k_norm, positions, slots, n, l.kv_heads, l.cap, eps, rope_theta, kc, vc, stream)
        }
    }

    fn need(buf: &DevBuf, bytes: usize, what: &str) {
        assert!(buf.bytes() >= bytes, "{what}: buffer has {} bytes, needs {bytes}", buf.bytes());
    }

    /// Grouped two-tap dynamic conv. `u`, `out` bf16 `[n_seq*rows, hidden]`; `dyn_` bf16 `[n_seq*rows, 4*hidden/group]`
    /// (`[prepare/finish][tap][group]`); `base` f16 `[2][2][hidden]`; `which` 0 = prepare, 1 = finish. `resid` (optional,
    /// may be the same buffer as `out`) makes it `out = bf16(resid + conv)`. `out` must not alias `u`. Each block of `rows`
    /// consecutive rows is independent (`u[-1] = 0` at the block start); `rows <= 16` for the draft.
    #[allow(clippy::too_many_arguments)]
    pub fn dynconv(
        u: &DevBuf,
        dyn_: &DevBuf,
        base: &DevBuf,
        which: usize,
        n_seq: usize,
        rows: usize,
        hidden: usize,
        group: usize,
        resid: Option<&DevBuf>,
        out: &mut DevBuf,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        let r = n_seq * rows;
        need(u, r * hidden * 2, "dynconv u");
        need(out, r * hidden * 2, "dynconv out");
        need(dyn_, r * 4 * (hidden / group) * 2, "dynconv dyn");
        need(base, 4 * hidden * 2, "dynconv base");
        if let Some(x) = resid {
            need(x, r * hidden * 2, "dynconv resid");
        }
        let rp = resid.map_or(core::ptr::null(), |b| b.ptr() as *const _);
        let rc = unsafe {
            ffi::gb_dflash_dynconv(
                u.ptr(),
                dyn_.ptr(),
                base.ptr(),
                which as i32,
                n_seq as i32,
                rows as i32,
                hidden as i32,
                group as i32,
                rp,
                out.ptr(),
                stream.raw(),
            )
        };
        CudaError::check(rc)
    }

    /// Context K/V write: for each of `n` rows, `k = rope(rmsnorm_plain(kproj_head) * k_norm, pos)` and `v = vproj` go to
    /// ring entry `slots[i] * cap + positions[i] % cap` of `kc`/`vc` (`[n_slots, cap, kv_heads, 128]`).
    /// `kproj`/`vproj` bf16 `[n, kv_heads*128]`, `k_norm` f32 `[128]`, `positions`/`slots` i32 `[n]`.
    #[allow(clippy::too_many_arguments)]
    pub fn ctx_kv(
        kproj: &DevBuf,
        vproj: &DevBuf,
        k_norm: &DevBuf,
        positions: &DevBuf,
        slots: &DevBuf,
        n: usize,
        kv_heads: usize,
        cap: usize,
        eps: f32,
        rope_theta: f64,
        kc: &mut DevBuf,
        vc: &mut DevBuf,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        need(kproj, n * kv_heads * 128 * 2, "ctx_kv kproj");
        need(vproj, n * kv_heads * 128 * 2, "ctx_kv vproj");
        need(k_norm, 128 * 4, "ctx_kv k_norm");
        need(positions, n * 4, "ctx_kv positions");
        need(slots, n * 4, "ctx_kv slots");
        let rc = unsafe {
            ffi::gb_dflash_ctx_kv(
                kproj.ptr(),
                vproj.ptr(),
                k_norm.ptr(),
                positions.ptr(),
                slots.ptr(),
                n as i32,
                kv_heads as i32,
                cap as i32,
                eps,
                rope_theta,
                kc.ptr(),
                vc.ptr(),
                stream.raw(),
            )
        };
        CudaError::check(rc)
    }

    /// Workspace bytes `attn` needs for `n_seq` sequences at a block of `rows` rows (8 or 16).
    pub fn attn_workspace_bytes(n_seq: usize, window: usize, n_q_heads: usize, rows: usize) -> usize {
        unsafe { ffi::gb_dflash_attn_workspace_bytes(n_seq as i32, window as i32, n_q_heads as i32, rows as i32) }
    }

    /// Draft attention for `n_seq` blocks of `rows` rows (the draft block: 8 the trained value, 16 the depth study's
    /// `REXL3_DRAFT_BLOCK`). `q` bf16 `[n_seq*rows, n_q_heads, 128]` and `block_k`, `block_v` bf16
    /// `[n_seq*rows, n_kv_heads, 128]` are already normed and roped; `kc`/`vc` are this layer's ring buffers; `slots` and
    /// `p` are i32 `[n_seq]` (cache slot, and the position of block row 0 = committed context length). Row `t` sees the
    /// ring positions `kpos >= p + t - window`, `kpos < p`, and block rows `0..=t`. `out` bf16 `[n_seq*rows, n_q_heads*128]`.
    /// `ws` is caller scratch of `attn_workspace_bytes` (contents irrelevant on entry; reusable across layers).
    /// In-block attention is causal, so rows 0..7 of a 16-row block are the 8-row block's bits exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn attn(
        q: &DevBuf,
        block_k: &DevBuf,
        block_v: &DevBuf,
        kc: &DevBuf,
        vc: &DevBuf,
        slots: &DevBuf,
        p: &DevBuf,
        n_seq: usize,
        n_q_heads: usize,
        n_kv_heads: usize,
        cap: usize,
        window: usize,
        rows: usize,
        ws: &mut DevBuf,
        out: &mut DevBuf,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        need(q, n_seq * rows * n_q_heads * 128 * 2, "attn q");
        need(block_k, n_seq * rows * n_kv_heads * 128 * 2, "attn block_k");
        need(block_v, n_seq * rows * n_kv_heads * 128 * 2, "attn block_v");
        need(slots, n_seq * 4, "attn slots");
        need(p, n_seq * 4, "attn p");
        need(out, n_seq * rows * n_q_heads * 128 * 2, "attn out");
        need(ws, attn_workspace_bytes(n_seq, window, n_q_heads, rows), "attn workspace");
        let rc = unsafe {
            ffi::gb_dflash_attn(
                q.ptr(),
                block_k.ptr(),
                block_v.ptr(),
                kc.ptr(),
                vc.ptr(),
                slots.ptr(),
                p.ptr(),
                n_seq as i32,
                n_q_heads as i32,
                n_kv_heads as i32,
                cap as i32,
                window as i32,
                rows as i32,
                ws.ptr(),
                out.ptr(),
                stream.raw(),
            )
        };
        CudaError::check(rc)
    }

    /// Gate input of the selector walk.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum GateDtype {
        Bf16,
        F32,
    }

    /// Per-sequence sampling record of [`selector_walk_keyed`]; layout mirrors `WalkSampling` in `dflash2.cu`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct WalkSampling {
        pub seed: u64,
        /// Absolute position of draft row 0, the token the target samples right after the anchor.
        pub first_position: u64,
        pub temperature: f64,
        pub top_p: f64,
        pub min_p: f64,
        /// 0 walks by argmax of the selector score; 1 draws with the target's keyed sampler.
        pub keyed: u32,
        pub pad: u32,
    }

    /// [`selector_walk`] with one [`WalkSampling`] per sequence in `sampling` (device array `[n_seq]`; `None` is all greedy).
    #[allow(clippy::too_many_arguments)]
    pub fn selector_walk_keyed(
        anchor: &DevBuf,
        topk_ids: &DevBuf,
        topk_vals: &DevBuf,
        gate: &DevBuf,
        gate_dtype: GateDtype,
        pred_codebook: &DevBuf,
        succ_codebook: &DevBuf,
        n_seq: usize,
        rows: usize,
        k: usize,
        rank: usize,
        out_tokens: &mut DevBuf,
        sampling: Option<&DevBuf>,
        conf: Option<&mut DevBuf>,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        if let Some(sm) = sampling {
            need(sm, n_seq * core::mem::size_of::<WalkSampling>(), "walk sampling");
        }
        let gb = if gate_dtype == GateDtype::F32 { 4 } else { 2 };
        need(anchor, n_seq * 4, "walk anchor");
        need(topk_ids, n_seq * rows * k * 4, "walk ids");
        need(topk_vals, n_seq * rows * k * 4, "walk vals");
        need(gate, n_seq * rows * rank * gb, "walk gate");
        need(out_tokens, n_seq * rows * 4, "walk out");
        let cp = match conf {
            Some(c) => {
                need(c, n_seq * rows * 4, "walk conf");
                c.ptr()
            }
            None => core::ptr::null_mut(),
        };
        let rc = unsafe {
            ffi::gb_dflash_selector_walk_keyed(
                anchor.ptr(),
                topk_ids.ptr(),
                topk_vals.ptr(),
                gate.ptr(),
                (gate_dtype == GateDtype::F32) as i32,
                pred_codebook.ptr(),
                succ_codebook.ptr(),
                n_seq as i32,
                rows as i32,
                k as i32,
                rank as i32,
                sampling.map_or(core::ptr::null(), |b| b.ptr() as *const _),
                out_tokens.ptr(),
                cp,
                stream.raw(),
            )
        };
        CudaError::check(rc)
    }

    /// [`selector_walk_keyed`] for sequences that have different numbers of rows: `topk_ids`, `topk_vals`, `gate`, `out_tokens`
    /// and `conf` and `margin` (the winner's lead over the runner-up, see `keyed_choose` in dflash2.cu) hold the sequences' rows packed one after another, `row_off` (device i32 `[n_seq + 1]`, starting at 0) says where
    /// each sequence's rows start, and `total_rows` is `row_off[n_seq]`. Row `i` of a sequence is drawn at the position
    /// `first_position + i` as in the uniform walk, so a sequence's first `r` picks do not depend on how many rows follow them.
    #[allow(clippy::too_many_arguments)]
    pub fn selector_walk_ragged(
        anchor: &DevBuf,
        topk_ids: &DevBuf,
        topk_vals: &DevBuf,
        gate: &DevBuf,
        gate_dtype: GateDtype,
        pred_codebook: &DevBuf,
        succ_codebook: &DevBuf,
        n_seq: usize,
        total_rows: usize,
        k: usize,
        rank: usize,
        out_tokens: &mut DevBuf,
        sampling: Option<&DevBuf>,
        row_off: &DevBuf,
        conf: Option<&mut DevBuf>,
        margin: Option<&mut DevBuf>,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        if let Some(sm) = sampling {
            need(sm, n_seq * core::mem::size_of::<WalkSampling>(), "walk sampling");
        }
        let gb = if gate_dtype == GateDtype::F32 { 4 } else { 2 };
        need(anchor, n_seq * 4, "walk anchor");
        need(row_off, (n_seq + 1) * 4, "walk row offsets");
        need(topk_ids, total_rows * k * 4, "walk ids");
        need(topk_vals, total_rows * k * 4, "walk vals");
        need(gate, total_rows * rank * gb, "walk gate");
        need(out_tokens, total_rows * 4, "walk out");
        let cp = match conf {
            Some(c) => {
                need(c, total_rows * 4, "walk conf");
                c.ptr()
            }
            None => core::ptr::null_mut(),
        };
        let mp = match margin {
            Some(m) => {
                need(m, total_rows * 4, "walk margin");
                m.ptr()
            }
            None => core::ptr::null_mut(),
        };
        let rc = unsafe {
            ffi::gb_dflash_selector_walk_ragged(
                anchor.ptr(),
                topk_ids.ptr(),
                topk_vals.ptr(),
                gate.ptr(),
                (gate_dtype == GateDtype::F32) as i32,
                pred_codebook.ptr(),
                succ_codebook.ptr(),
                n_seq as i32,
                k as i32,
                rank as i32,
                sampling.map_or(core::ptr::null(), |b| b.ptr() as *const _),
                row_off.ptr(),
                out_tokens.ptr(),
                cp,
                mp,
                stream.raw(),
            )
        };
        CudaError::check(rc)
    }

    /// The lattice behind the selector's chain walk: for every row and every parent a node of the next level could have, the dot
    /// term of each of the row's `k` candidates (the walk's score of candidate c after pick `pred` is `unary[c] + dot`). `anchor`,
    /// `topk_ids`, `gate`, `row_off` and the codebooks are the ragged walk's; `edge` is f32 `[total_rows, k, k]`, `edge[row][p][c]`:
    /// the parent `p` is the sequence's anchor (`p = 0`, the other entries zero) for its first row and candidate `p` of the row
    /// before it otherwise. Same arithmetic as the walk, so a chain through it scores exactly as the walk does.
    #[allow(clippy::too_many_arguments)]
    pub fn selector_edges(
        anchor: &DevBuf,
        topk_ids: &DevBuf,
        gate: &DevBuf,
        gate_dtype: GateDtype,
        pred_codebook: &DevBuf,
        succ_codebook: &DevBuf,
        n_seq: usize,
        total_rows: usize,
        k: usize,
        rank: usize,
        row_off: &DevBuf,
        edge: &mut DevBuf,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        let gb = if gate_dtype == GateDtype::F32 { 4 } else { 2 };
        need(anchor, n_seq * 4, "edge anchor");
        need(row_off, (n_seq + 1) * 4, "edge row offsets");
        need(topk_ids, total_rows * k * 4, "edge ids");
        need(gate, total_rows * rank * gb, "edge gate");
        need(edge, total_rows * k * k * 4, "edge out");
        let rc = unsafe {
            ffi::gb_dflash_selector_edges(
                anchor.ptr(),
                topk_ids.ptr(),
                gate.ptr(),
                (gate_dtype == GateDtype::F32) as i32,
                pred_codebook.ptr(),
                succ_codebook.ptr(),
                n_seq as i32,
                total_rows as i32,
                k as i32,
                rank as i32,
                row_off.ptr(),
                edge.ptr(),
                stream.raw(),
            )
        };
        CudaError::check(rc)
    }

    /// Map the first `count` ids of `ids` (i32) from a restricted candidate set back to token ids, in place. The set is
    /// `ranges` laid end to end, as `(first token id, length)` pairs in ascending order (at most 4): compact id `c` in the
    /// `r`-th range is token `first_r + (c - start_r)`.
    pub fn remap_ids(ids: &mut DevBuf, count: usize, ranges: &[(usize, usize)], stream: &Stream) -> Result<(), CudaError> {
        assert!((1..=4).contains(&ranges.len()), "1 to 4 id ranges");
        need(ids, count * 4, "remap ids");
        let mut cum = [0i32; 5];
        let mut base = [0i32; 4];
        for (r, &(first, len)) in ranges.iter().enumerate() {
            base[r] = first as i32;
            cum[r + 1] = cum[r] + len as i32;
        }
        let rc = unsafe {
            ffi::gb_dflash_remap_ids(ids.ptr(), count as i32, ranges.len() as i32, cum.as_ptr(), base.as_ptr(), stream.raw())
        };
        CudaError::check(rc)
    }

    /// The top-k of a draft head split over two ranks: `ids`/`vals` (i32 / f32 `[rows, k]`) hold this rank's top `k` of its columns
    /// (token ids), `peer_ids`/`peer_vals` the other rank's; each row becomes the `k` largest of the `2k` by (value DESC, id ASC), the
    /// full head's top-k order, in place. Both ranks get the same lists (the halves' ids are disjoint). `k <= 16`.
    pub fn merge_topk(
        ids: &mut DevBuf,
        vals: &mut DevBuf,
        peer_ids: &DevBuf,
        peer_vals: &DevBuf,
        rows: usize,
        k: usize,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        assert!((1..=16).contains(&k), "1 to 16 candidates a row");
        for (b, what) in [(&*ids, "merge ids"), (&*vals, "merge vals"), (peer_ids, "merge peer ids"), (peer_vals, "merge peer vals")] {
            need(b, rows * k * 4, what);
        }
        let rc = unsafe {
            ffi::gb_dflash_merge_topk(
                ids.ptr(),
                vals.ptr(),
                peer_ids.ptr(),
                peer_vals.ptr(),
                ids.ptr(),
                vals.ptr(),
                rows as i32,
                k as i32,
                stream.raw(),
            )
        };
        CudaError::check(rc)
    }

    /// The selector chain walk (spec 8.4), one block per sequence. `anchor` i32 `[n_seq]`, `topk_ids` i32
    /// `[n_seq, rows, k]`, `topk_vals` f32 `[n_seq, rows, k]` (descending, ties lowest id), `gate` bf16 or f32
    /// `[n_seq, rows, rank]`, codebooks f16 `[V, rank]`, `out_tokens` i32 `[n_seq, rows]` (d_1..d_rows; the anchor is not
    /// repeated), `conf` optional f32 `[n_seq, rows]` winning scores.
    #[allow(clippy::too_many_arguments)]
    pub fn selector_walk(
        anchor: &DevBuf,
        topk_ids: &DevBuf,
        topk_vals: &DevBuf,
        gate: &DevBuf,
        gate_dtype: GateDtype,
        pred_codebook: &DevBuf,
        succ_codebook: &DevBuf,
        n_seq: usize,
        rows: usize,
        k: usize,
        rank: usize,
        out_tokens: &mut DevBuf,
        conf: Option<&mut DevBuf>,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        selector_walk_keyed(
            anchor,
            topk_ids,
            topk_vals,
            gate,
            gate_dtype,
            pred_codebook,
            succ_codebook,
            n_seq,
            rows,
            k,
            rank,
            out_tokens,
            None,
            conf,
            stream,
        )
    }
}

/// Pure-Rust reference of the draft block (spec 8.2-8.4) in f64. Weights are plain float arrays (the EXL3 layers are
/// supplied through the [`Linear`] trait, so a test can dequantise them however it likes). No CUDA, no extra crates.
#[allow(clippy::needless_range_loop)]
pub mod cpu {
    /// Model dimensions of the draft.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Dims {
        pub hidden: usize,
        pub q_heads: usize,
        pub kv_heads: usize,
        pub head_dim: usize,
        /// Channels per dynamic-conv group.
        pub conv_group: usize,
        /// Draft rows per block (the conv taps reach back at most this far, in-block only).
        pub rows: usize,
        pub window: usize,
        pub eps: f64,
        pub rope_theta: f64,
    }

    impl Dims {
        pub const QWEN38: Dims = Dims {
            hidden: 5120,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            conv_group: 16,
            rows: 8,
            window: 2048,
            eps: 1e-6,
            rope_theta: 1e7,
        };
        pub fn groups(&self) -> usize {
            self.hidden / self.conv_group
        }
        /// Width of the kernel_projection output: `[prepare/finish][tap 2][group]`.
        pub fn dyn_width(&self) -> usize {
            4 * self.groups()
        }
    }

    /// `y = x W^T` for a `[rows, in]` row-major `x`.
    pub trait Linear {
        fn forward(&self, x: &[f64], rows: usize) -> Vec<f64>;
    }

    /// A plain dense layer, PyTorch layout `w[out][in]`.
    pub struct DenseLinear {
        pub w: Vec<f32>,
        pub in_features: usize,
        pub out_features: usize,
    }

    impl Linear for DenseLinear {
        fn forward(&self, x: &[f64], rows: usize) -> Vec<f64> {
            assert_eq!(x.len(), rows * self.in_features);
            let mut y = vec![0.0; rows * self.out_features];
            for r in 0..rows {
                let xr = &x[r * self.in_features..(r + 1) * self.in_features];
                for o in 0..self.out_features {
                    let wr = &self.w[o * self.in_features..(o + 1) * self.in_features];
                    y[r * self.out_features + o] = xr.iter().zip(wr).map(|(a, b)| a * *b as f64).sum();
                }
            }
            y
        }
    }

    /// One dynamic conv: `base` f32-converted `[2][2][hidden]` and the kernel projection (`hidden -> 4 * groups`).
    pub struct ConvWeights {
        pub base: Vec<f32>,
        pub proj: Box<dyn Linear>,
    }

    pub struct LayerWeights {
        pub input_ln: Vec<f32>,
        pub attn_conv: ConvWeights,
        pub q: Box<dyn Linear>,
        pub k: Box<dyn Linear>,
        pub v: Box<dyn Linear>,
        pub o: Box<dyn Linear>,
        pub q_norm: Vec<f32>,
        pub k_norm: Vec<f32>,
        pub post_ln: Vec<f32>,
        pub mlp_conv: ConvWeights,
        pub gate: Box<dyn Linear>,
        pub up: Box<dyn Linear>,
        pub down: Box<dyn Linear>,
    }

    /// `x * rsqrt(mean(x^2) + eps) * w` (plain gamma) over one vector.
    pub fn rmsnorm_plain(x: &[f64], w: &[f32], eps: f64) -> Vec<f64> {
        assert_eq!(x.len(), w.len());
        let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let inv = 1.0 / (ms + eps).sqrt();
        x.iter().zip(w).map(|(v, g)| v * inv * *g as f64).collect()
    }

    /// `rmsnorm_plain` on every `w.len()`-wide chunk of `x` (rows, or heads within rows).
    pub fn rmsnorm_chunks(x: &[f64], w: &[f32], eps: f64) -> Vec<f64> {
        x.chunks(w.len()).flat_map(|c| rmsnorm_plain(c, w, eps)).collect()
    }

    /// Rotate-half rope over a whole head vector: pairs `(d, d + hd/2)`, `inv_freq[i] = theta^(-2i/hd)`.
    pub fn rope_head(x: &mut [f64], pos: usize, theta: f64) {
        let hd = x.len();
        let half = hd / 2;
        for i in 0..half {
            let ang = pos as f64 * theta.powf(-(i as f64) / half as f64);
            let (s, c) = ang.sin_cos();
            let (a, b) = (x[i], x[i + half]);
            x[i] = a * c - b * s;
            x[i + half] = b * c + a * s;
        }
    }

    /// Grouped dynamic conv over one block: `u` `[rows, hidden]`, `dyn_` `[rows, 4 * groups]`, `base` `[2][2][hidden]`.
    pub fn dynconv(d: &Dims, u: &[f64], rows: usize, dyn_: &[f64], base: &[f32], which: usize) -> Vec<f64> {
        let (h, g, gs) = (d.hidden, d.groups(), d.conv_group);
        assert_eq!(u.len(), rows * h);
        assert_eq!(dyn_.len(), rows * 4 * g);
        let mut out = vec![0.0; rows * h];
        for t in 0..rows {
            for c in 0..h {
                let mut acc = 0.0;
                for tap in 0..2 {
                    if tap > t {
                        break;
                    }
                    let w = base[(which * 2 + tap) * h + c] as f64 + dyn_[t * 4 * g + (which * 2 + tap) * g + c / gs];
                    acc += w * u[(t - tap) * h + c];
                }
                out[t * h + c] = acc;
            }
        }
        out
    }

    /// Context K/V of one draft layer: `kproj`, `vproj` `[n, kv_heads*hd]` (the K and V linear outputs of `ctx_c`).
    /// Returns `(k, v)` where `k = rope(rmsnorm_plain(kproj_head, gamma), positions[i])`, `v = vproj`.
    pub fn ctx_kv(d: &Dims, kproj: &[f64], vproj: &[f64], gamma: &[f32], positions: &[usize]) -> (Vec<f64>, Vec<f64>) {
        let w = d.kv_heads * d.head_dim;
        let n = positions.len();
        assert_eq!(kproj.len(), n * w);
        let mut k = Vec::with_capacity(n * w);
        for (i, &pos) in positions.iter().enumerate() {
            for hd in kproj[i * w..(i + 1) * w].chunks(d.head_dim) {
                let mut v = rmsnorm_plain(hd, gamma, d.eps);
                rope_head(&mut v, pos, d.rope_theta);
                k.extend(v);
            }
        }
        (k, vproj.to_vec())
    }

    /// Draft attention of one block. `q` `[rows, q_heads*hd]`, `bk`/`bv` `[rows, kv_heads*hd]`; `ctx_k`/`ctx_v` hold
    /// positions `0..p` as `[p, kv_heads*hd]` (absolute positions, no ring). Row t sees ctx `kpos >= p + t - window`
    /// and block rows `0..=t`. Returns `[rows, q_heads*hd]`.
    pub fn attention(d: &Dims, q: &[f64], bk: &[f64], bv: &[f64], ctx_k: &[f64], ctx_v: &[f64], p: usize) -> Vec<f64> {
        let (hd, kvw, qw) = (d.head_dim, d.kv_heads * d.head_dim, d.q_heads * d.head_dim);
        let rows = q.len() / qw;
        assert_eq!(ctx_k.len(), p * kvw);
        let group = d.q_heads / d.kv_heads;
        let scale = 1.0 / (hd as f64).sqrt();
        let mut out = vec![0.0; rows * qw];
        for t in 0..rows {
            for h in 0..d.q_heads {
                let kvh = h / group;
                let qv = &q[t * qw + h * hd..t * qw + (h + 1) * hd];
                let lo = (p + t).saturating_sub(d.window);
                let mut scores = Vec::new();
                let mut vals: Vec<&[f64]> = Vec::new();
                for kp in lo..p {
                    let kv = &ctx_k[kp * kvw + kvh * hd..kp * kvw + (kvh + 1) * hd];
                    scores.push(qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f64>() * scale);
                    vals.push(&ctx_v[kp * kvw + kvh * hd..kp * kvw + (kvh + 1) * hd]);
                }
                for j in 0..=t {
                    let kv = &bk[j * kvw + kvh * hd..j * kvw + (kvh + 1) * hd];
                    scores.push(qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f64>() * scale);
                    vals.push(&bv[j * kvw + kvh * hd..j * kvw + (kvh + 1) * hd]);
                }
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let ps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let l: f64 = ps.iter().sum();
                let o = &mut out[t * qw + h * hd..t * qw + (h + 1) * hd];
                for (pj, v) in ps.iter().zip(&vals) {
                    for i in 0..hd {
                        o[i] += pj / l * v[i];
                    }
                }
            }
        }
        out
    }

    /// The two kernels a GPU port can swap out: everything else in [`block_forward`] stays on the CPU.
    pub trait DraftOps {
        fn dynconv(&mut self, d: &Dims, u: &[f64], rows: usize, dyn_: &[f64], base: &[f32], which: usize) -> Vec<f64>;
        /// `layer`'s attention for the current block; `q`, `bk` already normed and roped; the context is the ops' own.
        fn attention(&mut self, d: &Dims, layer: usize, q: &[f64], bk: &[f64], bv: &[f64]) -> Vec<f64>;
    }

    /// All-CPU ops over a context held as absolute-position arrays per layer.
    pub struct CpuOps {
        pub p: usize,
        pub ctx_k: Vec<Vec<f64>>,
        pub ctx_v: Vec<Vec<f64>>,
    }

    impl DraftOps for CpuOps {
        fn dynconv(&mut self, d: &Dims, u: &[f64], rows: usize, dyn_: &[f64], base: &[f32], which: usize) -> Vec<f64> {
            dynconv(d, u, rows, dyn_, base, which)
        }
        fn attention(&mut self, d: &Dims, layer: usize, q: &[f64], bk: &[f64], bv: &[f64]) -> Vec<f64> {
            attention(d, q, bk, bv, &self.ctx_k[layer], &self.ctx_v[layer], self.p)
        }
    }

    fn silu(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    /// `ctx_c = rmsnorm_plain(hidden_norm)(fc(taps))` for `rows` rows of concatenated taps.
    pub fn ctx_hidden(d: &Dims, taps_concat: &[f64], rows: usize, fc: &dyn Linear, hidden_norm: &[f32]) -> Vec<f64> {
        rmsnorm_chunks(&fc.forward(taps_concat, rows), hidden_norm, d.eps)
    }

    /// One draft block (spec 8.3): `x0` `[rows, hidden]` embeddings of `[anchor, MASK...]`, block position `p`.
    /// Returns `state = rmsnorm_plain(norm)(x)` after all layers.
    pub fn block_forward(
        d: &Dims,
        layers: &[LayerWeights],
        norm: &[f32],
        x0: &[f64],
        p: usize,
        ops: &mut dyn DraftOps,
    ) -> Vec<f64> {
        let (h, hd) = (d.hidden, d.head_dim);
        let rows = x0.len() / h;
        let mut x = x0.to_vec();
        for (li, l) in layers.iter().enumerate() {
            // attention sublayer
            let hn = rmsnorm_chunks(&x, &l.input_ln, d.eps);
            let dynv = l.attn_conv.proj.forward(&hn, rows);
            let y = ops.dynconv(d, &hn, rows, &dynv, &l.attn_conv.base, 0);
            let mut q = rmsnorm_chunks(&l.q.forward(&y, rows), &l.q_norm, d.eps);
            let mut k = rmsnorm_chunks(&l.k.forward(&y, rows), &l.k_norm, d.eps);
            let v = l.v.forward(&y, rows);
            for t in 0..rows {
                for hq in 0..d.q_heads {
                    let o = (t * d.q_heads + hq) * hd;
                    rope_head(&mut q[o..o + hd], p + t, d.rope_theta);
                }
                for hk in 0..d.kv_heads {
                    let o = (t * d.kv_heads + hk) * hd;
                    rope_head(&mut k[o..o + hd], p + t, d.rope_theta);
                }
            }
            let att = ops.attention(d, li, &q, &k, &v);
            let o = l.o.forward(&att, rows);
            let c = ops.dynconv(d, &o, rows, &dynv, &l.attn_conv.base, 1);
            for (xi, ci) in x.iter_mut().zip(&c) {
                *xi += ci;
            }
            // MLP sublayer
            let hn = rmsnorm_chunks(&x, &l.post_ln, d.eps);
            let dynv = l.mlp_conv.proj.forward(&hn, rows);
            let y = ops.dynconv(d, &hn, rows, &dynv, &l.mlp_conv.base, 0);
            let g = l.gate.forward(&y, rows);
            let u = l.up.forward(&y, rows);
            let a: Vec<f64> = g.iter().zip(&u).map(|(g, u)| silu(*g) * u).collect();
            let m = l.down.forward(&a, rows);
            let c = ops.dynconv(d, &m, rows, &dynv, &l.mlp_conv.base, 1);
            for (xi, ci) in x.iter_mut().zip(&c) {
                *xi += ci;
            }
        }
        rmsnorm_chunks(&x, norm, d.eps)
    }

    /// Top-`k` of one logits row: value descending, ties to the lowest id. Returns `(id, value)`.
    pub fn topk_row(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut best: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        for (i, &v) in logits.iter().enumerate() {
            if best.len() == k && v <= best[k - 1].1 {
                continue; // equal values later in the row lose to earlier ids
            }
            let pos = best.iter().position(|&(_, bv)| v > bv).unwrap_or(best.len());
            best.insert(pos, (i as u32, v));
            best.truncate(k);
        }
        best
    }

    /// Selector chain walk (spec 8.4). `unary[t][c]`, `cand[t][c]` per row `t` (rows 1..=7 of the draft), `gate[t]` the
    /// `rank` projection of that row's state; `pred(id)` / `succ(id)` return a codebook row. Returns `d_1..d_rows`.
    pub fn selector_walk(
        anchor: u32,
        unary: &[Vec<f32>],
        cand: &[Vec<u32>],
        gate: &[Vec<f32>],
        pred: &dyn Fn(u32) -> Vec<f32>,
        succ: &dyn Fn(u32) -> Vec<f32>,
    ) -> Vec<u32> {
        let mut prev = anchor;
        let mut out = Vec::new();
        for t in 0..unary.len() {
            let a: Vec<f64> = pred(prev).iter().zip(&gate[t]).map(|(a, g)| *a as f64 * *g as f64).collect();
            let mut best = 0;
            let mut best_score = f64::NEG_INFINITY;
            for c in 0..cand[t].len() {
                let b = succ(cand[t][c]);
                let s = unary[t][c] as f64 + a.iter().zip(&b).map(|(a, b)| a * *b as f64).sum::<f64>();
                if s > best_score {
                    best_score = s;
                    best = c;
                }
            }
            prev = cand[t][best];
            out.push(prev);
        }
        out
    }
}

// The shared npy/metric helpers live in tests/common/mod.rs (not ours to reformat), so keep rustfmt out of it.
#[cfg(test)]
#[rustfmt::skip]
#[path = "../tests/common/mod.rs"]
mod test_common;

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::cpu::{self, Dims};
    use super::test_common::{cos_rel_rms, have_refs, ref_npy, Rng};
    use super::*;
    use crate::num::{bf16_to_f32, f32_to_bf16};
    use gb_exl3::checkpoint::{scan, Checkpoint};
    use gb_exl3::f16::f16_to_f64;
    use gb_exl3::f16::f64_to_f16;
    use std::path::PathBuf;

    const D: Dims = Dims::QWEN38;

    fn bf(v: f64) -> u16 {
        f32_to_bf16(v as f32)
    }
    fn bfs(v: &[f64]) -> Vec<u16> {
        v.iter().map(|&x| bf(x)).collect()
    }
    fn unbf(v: &[u16]) -> Vec<f64> {
        v.iter().map(|&x| bf16_to_f32(x) as f64).collect()
    }
    /// Round f64 values to bf16 precision (what the kernels see as their input).
    fn rbf(v: &[f64]) -> Vec<f64> {
        unbf(&bfs(v))
    }
    fn f32s(v: &[f64]) -> Vec<f32> {
        v.iter().map(|&x| x as f32).collect()
    }
    fn assert_close(what: &str, got: &[f64], want: &[f64], min_cos: f64, max_rel: f64) -> (f64, f64) {
        let (c, r) = cos_rel_rms(&f32s(got), &f32s(want));
        println!("{what}: cos {c:.8} relRMS {r:.3e}");
        assert!(c >= min_cos && r <= max_rel, "{what}: cos {c} relRMS {r}");
        (c, r)
    }

    fn pack_dir() -> PathBuf {
        PathBuf::from(std::env::var("REXL3_DRAFT_DIR").or_else(|_| std::env::var("REXL3_DRAFT_PACK")).unwrap_or_default())
    }

    fn plain_f32(ck: &Checkpoint, name: &str) -> Vec<f32> {
        let e = ck.plain.get(name).unwrap_or_else(|| panic!("no tensor {name}"));
        let b = ck.read_tensor(e).unwrap();
        match e.dtype.as_str() {
            "F16" => b.chunks_exact(2).map(|c| f16_to_f64(u16::from_le_bytes([c[0], c[1]])) as f32).collect(),
            "BF16" => b.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            d => panic!("{name}: dtype {d}"),
        }
    }

    fn plain_u16(ck: &Checkpoint, name: &str) -> Vec<u16> {
        let e = ck.plain.get(name).unwrap_or_else(|| panic!("no tensor {name}"));
        assert_eq!(e.dtype, "F16");
        ck.read_tensor(e).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    }

    // ------------------------------------------------------------------------------------------ layout (no device)

    #[test]
    fn layout_sizes_and_ring_arithmetic() {
        let l = DraftCtxLayout::qwen38(1);
        assert_eq!(l.bytes_per_slot(), 5 * 2 * 2560 * 8 * 128 * 2);
        assert_eq!(l.bytes_per_slot(), 52_428_800);
        assert_eq!(DraftCtxLayout::qwen38(3).total_bytes(), 3 * 52_428_800);
        assert_eq!(l.ring_index(2559), 2559);
        assert_eq!(l.ring_index(2560), 0);
        assert_eq!(l.ctx_window_start(100), 0);
        assert_eq!(l.ctx_window_start(5000), 2952);
        assert_eq!(l.elem_offset(2, 2561), (2 * 2560 + 1) * 1024);
        assert!(l.fits(512));
        assert!(!l.fits(513));
    }

    #[test]
    fn cpu_topk_orders_ties_by_lowest_id() {
        let t = cpu::topk_row(&[1.0, 3.0, 3.0, 0.5, 3.0, 2.0], 4);
        assert_eq!(t, vec![(1, 3.0), (2, 3.0), (4, 3.0), (5, 2.0)]);
    }

    #[test]
    fn cpu_dynconv_is_stateless_across_blocks_and_causal() {
        let d = Dims { hidden: 32, conv_group: 16, ..D };
        let mut rng = Rng(7);
        let g = d.groups();
        let u: Vec<f64> = rng.vec(8 * 32, -1.0, 1.0).iter().map(|&x| x as f64).collect();
        let dy: Vec<f64> = rng.vec(8 * 4 * g, -0.5, 0.5).iter().map(|&x| x as f64).collect();
        let base = rng.vec(4 * 32, -1.0, 1.0);
        let full = cpu::dynconv(&d, &u, 8, &dy, &base, 1);
        // changing a later row never changes an earlier output
        let mut u2 = u.clone();
        u2[7 * 32 + 3] += 5.0;
        let full2 = cpu::dynconv(&d, &u2, 8, &dy, &base, 1);
        assert_eq!(&full[..7 * 32], &full2[..7 * 32]);
        // row 0 only has tap 0
        let w0 = base[2 * 32 + 5] as f64 + dy[2 * g];
        assert!((full[5] - w0 * u[5]).abs() < 1e-12);
    }

    // ------------------------------------------------------------------------------------------------ GPU kernels

    mod gpu_tests {
        use super::*;
        use crate::{device_available, DevBuf, Stream};

        fn dev() -> Option<Stream> {
            if !device_available() {
                eprintln!("no CUDA device: skipping");
                return None;
            }
            Some(Stream::new().unwrap())
        }

        #[test]
        fn dynconv_bit_exact_vs_f32_mirror() {
            let Some(st) = dev() else { return };
            let mut rng = Rng(1);
            for &(n_seq, rows, which, with_resid) in
                &[(1usize, 8usize, 0usize, false), (3, 8, 1, false), (2, 16, 0, false), (3, 8, 1, true), (5, 3, 0, true)]
            {
                let (h, g) = (D.hidden, D.groups());
                let r = n_seq * rows;
                let u: Vec<u16> = rng.vec(r * h, -3.0, 3.0).iter().map(|&x| f32_to_bf16(x)).collect();
                let dy: Vec<u16> = rng.vec(r * 4 * g, -0.6, 0.6).iter().map(|&x| f32_to_bf16(x)).collect();
                let base: Vec<u16> = rng.vec(4 * h, -1.0, 1.0).iter().map(|&x| f64_to_f16(x as f64)).collect();
                let res: Vec<u16> = rng.vec(r * h, -5.0, 5.0).iter().map(|&x| f32_to_bf16(x)).collect();
                let (du, ddy, dbase, dres) = (
                    DevBuf::from_slice(&u).unwrap(),
                    DevBuf::from_slice(&dy).unwrap(),
                    DevBuf::from_slice(&base).unwrap(),
                    DevBuf::from_slice(&res).unwrap(),
                );
                let mut out = DevBuf::zeroed(r * h * 2).unwrap();
                dynconv(&du, &ddy, &dbase, which, n_seq, rows, h, 16, with_resid.then_some(&dres), &mut out, &st).unwrap();
                st.sync().unwrap();
                let got: Vec<u16> = out.to_vec().unwrap();
                let mut bad = 0;
                for row in 0..r {
                    let t = row % rows;
                    for c in 0..h {
                        let dyn_at = |tap: usize| bf16_to_f32(dy[row * 4 * g + (which * 2 + tap) * g + c / 16]);
                        let bs = |tap: usize| f16_to_f64(base[(which * 2 + tap) * h + c]) as f32;
                        let mut acc = (bs(0) + dyn_at(0)).mul_add(bf16_to_f32(u[row * h + c]), 0.0);
                        if t >= 1 {
                            acc = (bs(1) + dyn_at(1)).mul_add(bf16_to_f32(u[(row - 1) * h + c]), acc);
                        }
                        if with_resid {
                            acc += bf16_to_f32(res[row * h + c]);
                        }
                        if got[row * h + c] != f32_to_bf16(acc) {
                            bad += 1;
                        }
                    }
                }
                assert_eq!(bad, 0, "dynconv n_seq {n_seq} rows {rows} which {which} resid {with_resid}: {bad} mismatches");
                // and against the f64 reference: within bf16 rounding
                let cpu_out: Vec<f64> = (0..n_seq)
                    .flat_map(|s| {
                        let ur = unbf(&u[s * rows * h..(s + 1) * rows * h]);
                        let dr = unbf(&dy[s * rows * 4 * g..(s + 1) * rows * 4 * g]);
                        let b32: Vec<f32> = base.iter().map(|&b| f16_to_f64(b) as f32).collect();
                        let mut o = cpu::dynconv(&D, &ur, rows, &dr, &b32, which);
                        if with_resid {
                            for (x, rr) in o.iter_mut().zip(&res[s * rows * h..(s + 1) * rows * h]) {
                                *x += bf16_to_f32(*rr) as f64;
                            }
                        }
                        o
                    })
                    .collect();
                assert_close("dynconv vs f64", &unbf(&got), &cpu_out, 0.99999, 5e-3);
            }
        }

        #[test]
        fn dynconv_in_place_residual() {
            let Some(st) = dev() else { return };
            let mut rng = Rng(11);
            let (h, g) = (D.hidden, D.groups());
            let u: Vec<u16> = rng.vec(8 * h, -1.0, 1.0).iter().map(|&x| f32_to_bf16(x)).collect();
            let dy: Vec<u16> = rng.vec(8 * 4 * g, -0.5, 0.5).iter().map(|&x| f32_to_bf16(x)).collect();
            let base: Vec<u16> = rng.vec(4 * h, -1.0, 1.0).iter().map(|&x| f64_to_f16(x as f64)).collect();
            let res: Vec<u16> = rng.vec(8 * h, -2.0, 2.0).iter().map(|&x| f32_to_bf16(x)).collect();
            let (du, ddy, dbase) =
                (DevBuf::from_slice(&u).unwrap(), DevBuf::from_slice(&dy).unwrap(), DevBuf::from_slice(&base).unwrap());
            let mut sep = DevBuf::zeroed(8 * h * 2).unwrap();
            let dres = DevBuf::from_slice(&res).unwrap();
            dynconv(&du, &ddy, &dbase, 1, 1, 8, h, 16, Some(&dres), &mut sep, &st).unwrap();
            // in place: resid and out are the same allocation (the API takes &DevBuf + &mut DevBuf, so go through ffi)
            let io = DevBuf::from_slice(&res).unwrap();
            let rc = unsafe {
                ffi::gb_dflash_dynconv(du.ptr(), ddy.ptr(), dbase.ptr(), 1, 1, 8, h as i32, 16, io.ptr(), io.ptr(), st.raw())
            };
            assert_eq!(rc, 0);
            st.sync().unwrap();
            assert_eq!(sep.to_vec::<u16>().unwrap(), io.to_vec::<u16>().unwrap());
        }

        #[test]
        fn ctx_kv_matches_cpu_including_ring_wrap_and_long_positions() {
            let Some(st) = dev() else { return };
            let layout = DraftCtxLayout::qwen38(3);
            let mut rng = Rng(5);
            // slot 2: far positions (rope precision); slot 0: positions straddling the ring end; slot 1: tiny positions
            let mut rows: Vec<(usize, usize)> = Vec::new(); // (slot, pos)
            rows.extend((0..20).map(|i| (2, 100_000 + i)));
            rows.extend((0..24).map(|i| (0, 2550 + i)));
            rows.extend((0..6).map(|i| (1, i)));
            let n = rows.len();
            let w = 8 * 128;
            let kp = rbf(&rng.vec(n * w, -4.0, 4.0).iter().map(|&x| x as f64).collect::<Vec<_>>());
            let vp = rbf(&rng.vec(n * w, -2.0, 2.0).iter().map(|&x| x as f64).collect::<Vec<_>>());
            let gamma: Vec<f32> = rng.vec(128, 0.5, 1.5);
            let pos_i32: Vec<i32> = rows.iter().map(|r| r.1 as i32).collect();
            let slot_i32: Vec<i32> = rows.iter().map(|r| r.0 as i32).collect();
            let (dk, dv, dg, dp, ds) = (
                DevBuf::from_slice(&bfs(&kp)).unwrap(),
                DevBuf::from_slice(&bfs(&vp)).unwrap(),
                DevBuf::from_slice(&gamma).unwrap(),
                DevBuf::from_slice(&pos_i32).unwrap(),
                DevBuf::from_slice(&slot_i32).unwrap(),
            );
            let mut cache = DraftCache::new(layout).unwrap();
            cache.write_ctx(0, &dk, &dv, &dg, &dp, &ds, n, 1e-6, 1e7, &st).unwrap();
            st.sync().unwrap();
            let kc: Vec<u16> = cache.k(0).to_vec().unwrap();
            let vc: Vec<u16> = cache.v(0).to_vec().unwrap();
            let positions: Vec<usize> = rows.iter().map(|r| r.1).collect();
            let (ck, cv) = cpu::ctx_kv(&D, &kp, &vp, &gamma, &positions);
            let mut got = Vec::new();
            let mut gotv = Vec::new();
            for &(slot, pos) in &rows {
                let o = layout.elem_offset(slot, pos);
                got.extend(unbf(&kc[o..o + w]));
                gotv.extend(unbf(&vc[o..o + w]));
            }
            assert_close("ctx K (long positions, wrap)", &got, &ck, 0.99999, 5e-3);
            assert_eq!(gotv, rbf(&cv), "V must be a raw copy");
            // untouched ring entries stay zero
            let o = layout.elem_offset(1, 7);
            assert!(kc[o..o + w].iter().all(|&x| x == 0));
        }

        /// Fills a cache and runs the attention for several sequences with different P, comparing with the f64 reference.
        fn run_attn_case(st: &Stream, ps: &[usize], slots_of: &[usize], seed: u64, bench_iters: usize) -> f64 {
            run_attn_case_block(st, ps, slots_of, seed, bench_iters, 8)
        }

        /// The same at a block of `rows` (8 the trained value, 16 `REXL3_DRAFT_BLOCK`): the CPU reference is block agnostic
        /// (row t sees the context rows and block rows 0..=t whatever the block), so it checks both.
        fn run_attn_case_block(st: &Stream, ps: &[usize], slots_of: &[usize], seed: u64, bench_iters: usize, rows: usize) -> f64 {
            let n_seq = ps.len();
            let layout = DraftCtxLayout::qwen38(4);
            let mut rng = Rng(seed);
            let (kvw, qw) = (8 * 128, 32 * 128);
            // stale garbage everywhere first: anything the kernel wrongly reads shows up
            let mut kc_h: Vec<u16> =
                rng.vec(layout.bytes_per_layer_buffer() / 2, -50.0, 50.0).iter().map(|&x| f32_to_bf16(x)).collect();
            let mut vc_h: Vec<u16> =
                rng.vec(layout.bytes_per_layer_buffer() / 2, -50.0, 50.0).iter().map(|&x| f32_to_bf16(x)).collect();
            let mut ctx: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
            for (s, &p) in ps.iter().enumerate() {
                let k = rbf(&rng.vec(p * kvw, -2.0, 2.0).iter().map(|&x| x as f64).collect::<Vec<_>>());
                let v = rbf(&rng.vec(p * kvw, -2.0, 2.0).iter().map(|&x| x as f64).collect::<Vec<_>>());
                for pos in layout.ctx_window_start(p)..p {
                    let o = layout.elem_offset(slots_of[s], pos);
                    for i in 0..kvw {
                        kc_h[o + i] = bf(k[pos * kvw + i]);
                        vc_h[o + i] = bf(v[pos * kvw + i]);
                    }
                }
                ctx.push((k, v));
            }
            let q = rbf(&rng.vec(n_seq * rows * qw, -2.0, 2.0).iter().map(|&x| x as f64).collect::<Vec<_>>());
            let bk = rbf(&rng.vec(n_seq * rows * kvw, -2.0, 2.0).iter().map(|&x| x as f64).collect::<Vec<_>>());
            let bv = rbf(&rng.vec(n_seq * rows * kvw, -2.0, 2.0).iter().map(|&x| x as f64).collect::<Vec<_>>());
            let (dq, dbk, dbv, dkc, dvc) = (
                DevBuf::from_slice(&bfs(&q)).unwrap(),
                DevBuf::from_slice(&bfs(&bk)).unwrap(),
                DevBuf::from_slice(&bfs(&bv)).unwrap(),
                DevBuf::from_slice(&kc_h).unwrap(),
                DevBuf::from_slice(&vc_h).unwrap(),
            );
            let slots: Vec<i32> = slots_of.iter().map(|&s| s as i32).collect();
            let pv: Vec<i32> = ps.iter().map(|&p| p as i32).collect();
            let (dslots, dp) = (DevBuf::from_slice(&slots).unwrap(), DevBuf::from_slice(&pv).unwrap());
            let mut ws = DevBuf::alloc(attn_workspace_bytes(n_seq, 2048, 32, rows)).unwrap();
            ws.memset_async(0xFF, st).unwrap(); // NaN bytes: the kernel must not read what it did not write
            let mut out = DevBuf::zeroed(n_seq * rows * qw * 2).unwrap();
            let run = |ws: &mut DevBuf, out: &mut DevBuf| {
                attn(&dq, &dbk, &dbv, &dkc, &dvc, &dslots, &dp, n_seq, 32, 8, layout.cap, 2048, rows, ws, out, st).unwrap();
            };
            run(&mut ws, &mut out);
            st.sync().unwrap();
            let got: Vec<u16> = out.to_vec().unwrap();
            for s in 0..n_seq {
                let want = cpu::attention(
                    &D,
                    &q[s * rows * qw..(s + 1) * rows * qw],
                    &bk[s * rows * kvw..(s + 1) * rows * kvw],
                    &bv[s * rows * kvw..(s + 1) * rows * kvw],
                    &ctx[s].0,
                    &ctx[s].1,
                    ps[s],
                );
                assert_close(
                    &format!("attn block {rows} seq {s} P {}", ps[s]),
                    &unbf(&got[s * rows * qw..(s + 1) * rows * qw]),
                    &want,
                    0.99999,
                    5e-3,
                );
            }
            // deterministic: a second run (workspace re-poisoned) is bit-identical
            ws.memset_async(0xFF, st).unwrap();
            run(&mut ws, &mut out);
            st.sync().unwrap();
            assert_eq!(out.to_vec::<u16>().unwrap(), got, "attn is not run-to-run reproducible");
            if bench_iters == 0 {
                return 0.0;
            }
            let t0 = std::time::Instant::now();
            for _ in 0..bench_iters {
                run(&mut ws, &mut out);
            }
            st.sync().unwrap();
            t0.elapsed().as_secs_f64() / bench_iters as f64 * 1e6
        }

        #[test]
        fn attn_matches_cpu_window_ring_and_multi_seq() {
            let Some(st) = dev() else { return };
            // P = 0 (block only), tiny, window not yet full, exactly at the window, past it, past the ring capacity
            run_attn_case(&st, &[0, 7, 100, 2047], &[0, 1, 2, 3], 21, 0);
            run_attn_case(&st, &[2048, 2049, 2600, 5000], &[3, 2, 1, 0], 22, 0);
        }

        #[test]
        fn attn_single_sequence_edge_positions() {
            let Some(st) = dev() else { return };
            for (i, &p) in [1usize, 63, 64, 65, 2040, 2055, 2560, 2561, 4607].iter().enumerate() {
                run_attn_case(&st, &[p], &[i % 4], 100 + i as u64, 0);
            }
        }

        /// A 16-row block against the f64 reference at the same edge positions (the depth study's block: rows 8..15 are new,
        /// rows 0..7 are the trained ones).
        #[test]
        fn attn_block16_matches_the_reference() {
            let Some(st) = dev() else { return };
            for (i, &p) in [1usize, 63, 64, 65, 2040, 2055, 2560, 2561, 4607].iter().enumerate() {
                run_attn_case_block(&st, &[p], &[i % 4], 200 + i as u64, 0, 16);
            }
            run_attn_case_block(&st, &[0, 7, 100, 2047], &[0, 1, 2, 3], 221, 0, 16);
            run_attn_case_block(&st, &[2048, 2049, 2600, 5000], &[3, 2, 1, 0], 222, 0, 16);
        }

        /// In-block attention is causal (`vis = key <= i`), and a row's context keys do not change with the block, so rows
        /// 0..7 of a 16-row block must be the 8-row block's bits exactly: deepening the block must not move the trained
        /// levels (the depth study runs levels 8..15 OFF the same arithmetic).
        #[test]
        fn attn_block16_levels_1_to_7_equal_block8() {
            let Some(st) = dev() else { return };
            let (kvw, qw) = (8 * 128, 32 * 128);
            let layout = DraftCtxLayout::qwen38(4);
            let mut rng = Rng(31);
            for (case, &p) in [0usize, 63, 2047, 2048, 4607].iter().enumerate() {
                let mut kc_h: Vec<u16> =
                    rng.vec(layout.bytes_per_layer_buffer() / 2, -50.0, 50.0).iter().map(|&x| f32_to_bf16(x)).collect();
                let mut vc_h: Vec<u16> =
                    rng.vec(layout.bytes_per_layer_buffer() / 2, -50.0, 50.0).iter().map(|&x| f32_to_bf16(x)).collect();
                let (k, v) = (
                    rbf(&rng.vec(p * kvw, -2.0, 2.0).iter().map(|&x| x as f64).collect::<Vec<_>>()),
                    rbf(&rng.vec(p * kvw, -2.0, 2.0).iter().map(|&x| x as f64).collect::<Vec<_>>()),
                );
                for pos in layout.ctx_window_start(p)..p {
                    let o = layout.elem_offset(0, pos);
                    for i in 0..kvw {
                        kc_h[o + i] = bf(k[pos * kvw + i]);
                        vc_h[o + i] = bf(v[pos * kvw + i]);
                    }
                }
                let q = rbf(&rng.vec(16 * qw, -2.0, 2.0).iter().map(|&x| x as f64).collect::<Vec<_>>());
                let bk = rbf(&rng.vec(16 * kvw, -2.0, 2.0).iter().map(|&x| x as f64).collect::<Vec<_>>());
                let bv = rbf(&rng.vec(16 * kvw, -2.0, 2.0).iter().map(|&x| x as f64).collect::<Vec<_>>());
                let (dq, dbk, dbv) = (
                    DevBuf::from_slice(&bfs(&q)).unwrap(),
                    DevBuf::from_slice(&bfs(&bk)).unwrap(),
                    DevBuf::from_slice(&bfs(&bv)).unwrap(),
                );
                let (dkc, dvc) = (DevBuf::from_slice(&kc_h).unwrap(), DevBuf::from_slice(&vc_h).unwrap());
                let (dslots, dp) = (DevBuf::from_slice(&[0i32]).unwrap(), DevBuf::from_slice(&[p as i32]).unwrap());
                let mut out16 = DevBuf::zeroed(16 * qw * 2).unwrap();
                let mut ws16 = DevBuf::alloc(attn_workspace_bytes(1, 2048, 32, 16)).unwrap();
                attn(&dq, &dbk, &dbv, &dkc, &dvc, &dslots, &dp, 1, 32, 8, layout.cap, 2048, 16, &mut ws16, &mut out16, &st)
                    .unwrap();
                let mut out8 = DevBuf::zeroed(8 * qw * 2).unwrap();
                let mut ws8 = DevBuf::alloc(attn_workspace_bytes(1, 2048, 32, 8)).unwrap();
                attn(&dq, &dbk, &dbv, &dkc, &dvc, &dslots, &dp, 1, 32, 8, layout.cap, 2048, 8, &mut ws8, &mut out8, &st).unwrap();
                st.sync().unwrap();
                let (a, b) = (out16.to_vec::<u16>().unwrap(), out8.to_vec::<u16>().unwrap());
                assert_eq!(a[..8 * qw], b[..], "case {case} (P {p}): rows 0..7 of a 16-block differ from the 8-block");
            }
        }

        // ---------------------------------------------------------------------------------------------- selector

        struct WalkRef {
            anchor: u32,
            ids: Vec<u32>,
            vals: Vec<f32>,
            gate: Vec<f32>,
        }

        #[allow(clippy::too_many_arguments)]
        fn run_walk_gpu(
            st: &Stream,
            w: &[WalkRef],
            pred: &DevBuf,
            succ: &DevBuf,
            rows: usize,
            k: usize,
            rank: usize,
            f32_gate: bool,
        ) -> Vec<i32> {
            let n_seq = w.len();
            let anchors: Vec<i32> = w.iter().map(|x| x.anchor as i32).collect();
            let ids: Vec<i32> = w.iter().flat_map(|x| x.ids.iter().map(|&i| i as i32)).collect();
            let vals: Vec<f32> = w.iter().flat_map(|x| x.vals.clone()).collect();
            let gate_f: Vec<f32> = w.iter().flat_map(|x| x.gate.clone()).collect();
            let (da, di, dv) =
                (DevBuf::from_slice(&anchors).unwrap(), DevBuf::from_slice(&ids).unwrap(), DevBuf::from_slice(&vals).unwrap());
            let (dg, gd) = if f32_gate {
                (DevBuf::from_slice(&gate_f).unwrap(), GateDtype::F32)
            } else {
                (DevBuf::from_slice(&gate_f.iter().map(|&g| f32_to_bf16(g)).collect::<Vec<_>>()).unwrap(), GateDtype::Bf16)
            };
            let mut out = DevBuf::zeroed(n_seq * rows * 4).unwrap();
            selector_walk(&da, &di, &dv, &dg, gd, pred, succ, n_seq, rows, k, rank, &mut out, None, st).unwrap();
            st.sync().unwrap();
            out.to_vec().unwrap()
        }

        #[test]
        fn walk_matches_cpu_on_random_data() {
            let Some(st) = dev() else { return };
            let (v, rank, rows, k, n_seq) = (3000usize, 256usize, 7usize, 16usize, 5usize);
            let mut rng = Rng(77);
            let cb = |rng: &mut Rng| -> Vec<u16> { rng.vec(v * rank, -1.0, 1.0).iter().map(|&x| f64_to_f16(x as f64)).collect() };
            let (pc, sc) = (cb(&mut rng), cb(&mut rng));
            let (dpred, dsucc) = (DevBuf::from_slice(&pc).unwrap(), DevBuf::from_slice(&sc).unwrap());
            let mut seqs = Vec::new();
            for s in 0..n_seq {
                let mut ids = Vec::new();
                let mut vals = Vec::new();
                for _ in 0..rows {
                    // distinct ids per row, unary descending like a real top-k
                    let mut row: Vec<u32> = Vec::new();
                    while row.len() < k {
                        let c = (rng.next_u64() % v as u64) as u32;
                        if !row.contains(&c) {
                            row.push(c);
                        }
                    }
                    let mut u: Vec<f32> = rng.vec(k, 0.0, 6.0);
                    u.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    ids.extend(row);
                    vals.extend(u);
                }
                // the gate is a bf16 value on the bf16 path and exact on the f32 path; use bf16-representable values for both
                let gate: Vec<f32> = rng.vec(rows * rank, -1.0, 1.0).iter().map(|&g| bf16_to_f32(f32_to_bf16(g))).collect();
                seqs.push(WalkRef { anchor: (s * 37 + 5) as u32, ids, vals, gate });
            }
            for f32_gate in [true, false] {
                let got = run_walk_gpu(&st, &seqs, &dpred, &dsucc, rows, k, rank, f32_gate);
                for (s, w) in seqs.iter().enumerate() {
                    let unary: Vec<Vec<f32>> = (0..rows).map(|t| w.vals[t * k..(t + 1) * k].to_vec()).collect();
                    let cand: Vec<Vec<u32>> = (0..rows).map(|t| w.ids[t * k..(t + 1) * k].to_vec()).collect();
                    let gate: Vec<Vec<f32>> = (0..rows).map(|t| w.gate[t * rank..(t + 1) * rank].to_vec()).collect();
                    let row = |cbk: &[u16]| {
                        let cbk = cbk.to_vec();
                        move |id: u32| -> Vec<f32> {
                            cbk[id as usize * rank..(id as usize + 1) * rank].iter().map(|&h| f16_to_f64(h) as f32).collect()
                        }
                    };
                    let (pf, sf) = (row(&pc), row(&sc));
                    let want = cpu::selector_walk(w.anchor, &unary, &cand, &gate, &pf, &sf);
                    let g: Vec<u32> = got[s * rows..(s + 1) * rows].iter().map(|&x| x as u32).collect();
                    assert_eq!(g, want, "seq {s} f32_gate {f32_gate}");
                }
            }
        }

        /// With a zero gate the selector score is exactly the unary logit, so a keyed GPU walk must equal the target's own
        /// `try_choose` over the same candidates at positions `first_position + row` (ties by id, temperature, top_p, min_p).
        #[test]
        fn keyed_walk_matches_the_target_sampler() {
            let Some(st) = dev() else { return };
            let (v, rank, rows, k) = (3000usize, 256usize, 7usize, 16usize);
            let mut rng = Rng(91);
            let cb = |rng: &mut Rng| -> Vec<u16> { rng.vec(v * rank, -1.0, 1.0).iter().map(|&x| f64_to_f16(x as f64)).collect() };
            let (pc, sc) = (cb(&mut rng), cb(&mut rng));
            let (dpred, dsucc) = (DevBuf::from_slice(&pc).unwrap(), DevBuf::from_slice(&sc).unwrap());
            let settings: Vec<gb_sampling::Sampling> = vec![
                gb_sampling::Sampling { seed: 7, temperature: 1.0, top_k: k, top_p: 0.95, min_p: 0.0 },
                gb_sampling::Sampling { seed: u64::MAX >> 1, temperature: 0.7, top_k: k, top_p: 1.0, min_p: 0.0 },
                gb_sampling::Sampling { seed: 123456789, temperature: 1.3, top_k: k, top_p: 0.5, min_p: 0.05 },
                gb_sampling::Sampling { seed: 42, temperature: 1.0, top_k: k, top_p: 0.0, min_p: 0.2 },
                gb_sampling::Sampling { seed: 5, temperature: 0.0, top_k: k, top_p: 0.95, min_p: 0.0 },
                gb_sampling::Sampling { seed: 99, temperature: 2.0, top_k: k, top_p: 0.9, min_p: 0.0 },
            ];
            let n_seq = settings.len();
            let mut ids = Vec::new();
            let mut vals = Vec::new();
            for _ in 0..n_seq {
                for _ in 0..rows {
                    let mut row: Vec<u32> = Vec::new();
                    while row.len() < k {
                        let c = (rng.next_u64() % v as u64) as u32;
                        if !row.contains(&c) {
                            row.push(c);
                        }
                    }
                    // half-integer logits so exact ties occur and must resolve by id
                    let u: Vec<f32> = rng.vec(k, 0.0, 6.0).iter().map(|x| (x * 2.0).round() / 2.0).collect();
                    ids.extend(row);
                    vals.extend(u);
                }
            }
            let first: Vec<u64> = (0..n_seq).map(|s| 50 + 1000 * s as u64).collect();
            let rec: Vec<WalkSampling> = settings
                .iter()
                .zip(&first)
                .map(|(s, &f)| WalkSampling {
                    seed: s.seed,
                    first_position: f,
                    temperature: s.temperature,
                    top_p: s.top_p,
                    min_p: s.min_p,
                    keyed: 1,
                    pad: 0,
                })
                .collect();
            let anchors: Vec<i32> = (0..n_seq).map(|s| (s * 11 + 3) as i32).collect();
            let gate = vec![0u16; n_seq * rows * rank]; // bf16 zero
            let ids_i: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
            let (da, di, dv, dg, dr) = (
                DevBuf::from_slice(&anchors).unwrap(),
                DevBuf::from_slice(&ids_i).unwrap(),
                DevBuf::from_slice(&vals).unwrap(),
                DevBuf::from_slice(&gate).unwrap(),
                DevBuf::from_slice(&rec).unwrap(),
            );
            let mut out = DevBuf::zeroed(n_seq * rows * 4).unwrap();
            selector_walk_keyed(
                &da,
                &di,
                &dv,
                &dg,
                GateDtype::Bf16,
                &dpred,
                &dsucc,
                n_seq,
                rows,
                k,
                rank,
                &mut out,
                Some(&dr),
                None,
                &st,
            )
            .unwrap();
            st.sync().unwrap();
            let got: Vec<i32> = out.to_vec().unwrap();
            for s in 0..n_seq {
                for t in 0..rows {
                    let at = (s * rows + t) * k;
                    let want = gb_sampling::choose(&vals[at..at + k], &ids[at..at + k], first[s] + t as u64, &settings[s]);
                    assert_eq!(got[s * rows + t] as u32, want, "seq {s} row {t} {:?}", settings[s]);
                }
            }
            // keyed = 0 on the same records is the plain argmax walk
            let mut rec0 = rec.clone();
            rec0.iter_mut().for_each(|r| r.keyed = 0);
            let dr0 = DevBuf::from_slice(&rec0).unwrap();
            let mut out0 = DevBuf::zeroed(n_seq * rows * 4).unwrap();
            let mut out_none = DevBuf::zeroed(n_seq * rows * 4).unwrap();
            selector_walk_keyed(
                &da,
                &di,
                &dv,
                &dg,
                GateDtype::Bf16,
                &dpred,
                &dsucc,
                n_seq,
                rows,
                k,
                rank,
                &mut out0,
                Some(&dr0),
                None,
                &st,
            )
            .unwrap();
            selector_walk(&da, &di, &dv, &dg, GateDtype::Bf16, &dpred, &dsucc, n_seq, rows, k, rank, &mut out_none, None, &st)
                .unwrap();
            st.sync().unwrap();
            assert_eq!(out0.to_vec::<i32>().unwrap(), out_none.to_vec::<i32>().unwrap());
        }

        /// A ragged walk (each sequence its own number of rows, packed) picks for a sequence exactly the first `r` picks, and
        /// reports exactly the first `r` scores, of the same sequence walked with all its rows, keyed or greedy.
        #[test]
        fn ragged_walk_is_a_prefix_of_the_uniform_walk() {
            let Some(st) = dev() else { return };
            let (v, rank, rows, k) = (3000usize, 256usize, 7usize, 16usize);
            let mut rng = Rng(5150);
            let cb = |rng: &mut Rng| -> Vec<u16> { rng.vec(v * rank, -1.0, 1.0).iter().map(|&x| f64_to_f16(x as f64)).collect() };
            let (pc, sc) = (cb(&mut rng), cb(&mut rng));
            let (dpred, dsucc) = (DevBuf::from_slice(&pc).unwrap(), DevBuf::from_slice(&sc).unwrap());
            let want_rows = [7usize, 3, 0, 5, 1, 7, 2];
            let n_seq = want_rows.len();
            let mut ids = Vec::new();
            let mut vals = Vec::new();
            for _ in 0..n_seq * rows {
                let mut row: Vec<i32> = Vec::new();
                while row.len() < k {
                    let c = (rng.next_u64() % v as u64) as i32;
                    if !row.contains(&c) {
                        row.push(c);
                    }
                }
                ids.extend(row);
                vals.extend(rng.vec(k, 0.0, 6.0).iter().map(|x| (x * 2.0).round() / 2.0));
            }
            let gate: Vec<u16> = rng.vec(n_seq * rows * rank, -1.0, 1.0).iter().map(|&g| f32_to_bf16(g)).collect();
            let anchors: Vec<i32> = (0..n_seq).map(|s| (s * 13 + 1) as i32).collect();
            let rec: Vec<WalkSampling> = (0..n_seq)
                .map(|s| WalkSampling {
                    seed: 1000 + s as u64,
                    first_position: 500 + 17 * s as u64,
                    temperature: if s % 3 == 1 { 0.0 } else { 0.9 },
                    top_p: 0.95,
                    min_p: 0.0,
                    keyed: (s % 3 != 1) as u32,
                    pad: 0,
                })
                .collect();
            let (da, di, dv, dg, dr) = (
                DevBuf::from_slice(&anchors).unwrap(),
                DevBuf::from_slice(&ids).unwrap(),
                DevBuf::from_slice(&vals).unwrap(),
                DevBuf::from_slice(&gate).unwrap(),
                DevBuf::from_slice(&rec).unwrap(),
            );
            // the uniform walk over every row of every sequence
            let mut full = DevBuf::zeroed(n_seq * rows * 4).unwrap();
            let mut full_conf = DevBuf::zeroed(n_seq * rows * 4).unwrap();
            selector_walk_keyed(
                &da,
                &di,
                &dv,
                &dg,
                GateDtype::Bf16,
                &dpred,
                &dsucc,
                n_seq,
                rows,
                k,
                rank,
                &mut full,
                Some(&dr),
                Some(&mut full_conf),
                &st,
            )
            .unwrap();
            // the same inputs packed: only the first `want_rows[s]` rows of each sequence, one after another
            let mut off = vec![0i32];
            let (mut pids, mut pvals, mut pgate) = (Vec::new(), Vec::new(), Vec::new());
            for (s, &r) in want_rows.iter().enumerate() {
                off.push(off[s] + r as i32);
                pids.extend_from_slice(&ids[s * rows * k..(s * rows + r) * k]);
                pvals.extend_from_slice(&vals[s * rows * k..(s * rows + r) * k]);
                pgate.extend_from_slice(&gate[s * rows * rank..(s * rows + r) * rank]);
            }
            let total = off[n_seq] as usize;
            let (dpi, dpv, dpg, doff) = (
                DevBuf::from_slice(&pids).unwrap(),
                DevBuf::from_slice(&pvals).unwrap(),
                DevBuf::from_slice(&pgate).unwrap(),
                DevBuf::from_slice(&off).unwrap(),
            );
            let mut out = DevBuf::zeroed(total * 4).unwrap();
            let mut conf = DevBuf::zeroed(total * 4).unwrap();
            let mut margin = DevBuf::zeroed(total * 4).unwrap();
            selector_walk_ragged(
                &da,
                &dpi,
                &dpv,
                &dpg,
                GateDtype::Bf16,
                &dpred,
                &dsucc,
                n_seq,
                total,
                k,
                rank,
                &mut out,
                Some(&dr),
                &doff,
                Some(&mut conf),
                Some(&mut margin),
                &st,
            )
            .unwrap();
            // the same walk with every row of every sequence: the packed layout is then the uniform one
            let full_off: Vec<i32> = (0..=n_seq).map(|s| (s * rows) as i32).collect();
            let dfull_off = DevBuf::from_slice(&full_off).unwrap();
            let mut ragged_full = DevBuf::zeroed(n_seq * rows * 4).unwrap();
            let mut ragged_full_margin = DevBuf::zeroed(n_seq * rows * 4).unwrap();
            selector_walk_ragged(
                &da,
                &di,
                &dv,
                &dg,
                GateDtype::Bf16,
                &dpred,
                &dsucc,
                n_seq,
                n_seq * rows,
                k,
                rank,
                &mut ragged_full,
                Some(&dr),
                &dfull_off,
                None,
                Some(&mut ragged_full_margin),
                &st,
            )
            .unwrap();
            st.sync().unwrap();
            let (full, full_conf): (Vec<i32>, Vec<f32>) = (full.to_vec().unwrap(), full_conf.to_vec().unwrap());
            let (got, got_conf): (Vec<i32>, Vec<f32>) = (out.to_vec().unwrap(), conf.to_vec().unwrap());
            let (got_margin, full_margin): (Vec<f32>, Vec<f32>) =
                (margin.to_vec().unwrap(), ragged_full_margin.to_vec().unwrap());
            assert_eq!(ragged_full.to_vec::<i32>().unwrap(), full, "a ragged walk of equal rows is the uniform walk");
            let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            for (s, &r) in want_rows.iter().enumerate() {
                let o = off[s] as usize;
                assert_eq!(got[o..o + r], full[s * rows..s * rows + r], "seq {s} with {r} rows");
                assert_eq!(bits(&got_conf[o..o + r]), bits(&full_conf[s * rows..s * rows + r]), "seq {s} scores");
                assert_eq!(bits(&got_margin[o..o + r]), bits(&full_margin[s * rows..s * rows + r]), "seq {s} margins");
            }
            assert!(got_margin.iter().all(|m| m.is_finite() && *m >= 0.0 && *m <= 30.0), "margins are leads: {got_margin:?}");
        }

        /// The lattice edges are the walk's own dot terms: walking the chain through `unary + edge[parent][c]` (first max, as the
        /// greedy walk does) gives the walk's picks, and the winning score equals the walk's reported score bit for bit. Ragged
        /// sequences included (a first row has the anchor as its only parent).
        #[test]
        fn edges_reproduce_the_greedy_walk_bit_for_bit() {
            let Some(st) = dev() else { return };
            let (v, rank, k) = (3000usize, 256usize, 16usize);
            let mut rng = Rng(4242);
            let cb = |rng: &mut Rng| -> Vec<u16> { rng.vec(v * rank, -1.0, 1.0).iter().map(|&x| f64_to_f16(x as f64)).collect() };
            let (dpred, dsucc) = (DevBuf::from_slice(&cb(&mut rng)).unwrap(), DevBuf::from_slice(&cb(&mut rng)).unwrap());
            let rows_of = [7usize, 3, 5, 1, 7];
            let n_seq = rows_of.len();
            let mut off = vec![0i32];
            for &r in &rows_of {
                off.push(off.last().unwrap() + r as i32);
            }
            let total = off[n_seq] as usize;
            let (mut ids, mut vals) = (Vec::new(), Vec::new());
            for _ in 0..total {
                let mut row: Vec<i32> = Vec::new();
                while row.len() < k {
                    let c = (rng.next_u64() % v as u64) as i32;
                    if !row.contains(&c) {
                        row.push(c);
                    }
                }
                let mut u: Vec<f32> = rng.vec(k, 0.0, 6.0);
                u.sort_by(|a, b| b.partial_cmp(a).unwrap());
                ids.extend(row);
                vals.extend(u);
            }
            let gate: Vec<u16> = rng.vec(total * rank, -1.0, 1.0).iter().map(|&g| f32_to_bf16(g)).collect();
            let anchors: Vec<i32> = (0..n_seq).map(|s| (s * 29 + 3) as i32).collect();
            let (da, di, dv, dg, doff) = (
                DevBuf::from_slice(&anchors).unwrap(),
                DevBuf::from_slice(&ids).unwrap(),
                DevBuf::from_slice(&vals).unwrap(),
                DevBuf::from_slice(&gate).unwrap(),
                DevBuf::from_slice(&off).unwrap(),
            );
            let mut out = DevBuf::zeroed(total * 4).unwrap();
            let mut conf = DevBuf::zeroed(total * 4).unwrap();
            selector_walk_ragged(
                &da,
                &di,
                &dv,
                &dg,
                GateDtype::Bf16,
                &dpred,
                &dsucc,
                n_seq,
                total,
                k,
                rank,
                &mut out,
                None,
                &doff,
                Some(&mut conf),
                None,
                &st,
            )
            .unwrap();
            let mut edge = DevBuf::zeroed(total * k * k * 4).unwrap();
            selector_edges(&da, &di, &dg, GateDtype::Bf16, &dpred, &dsucc, n_seq, total, k, rank, &doff, &mut edge, &st).unwrap();
            st.sync().unwrap();
            let (picks, conf): (Vec<i32>, Vec<f32>) = (out.to_vec().unwrap(), conf.to_vec().unwrap());
            let edge: Vec<f32> = edge.to_vec().unwrap();
            for s in 0..n_seq {
                let mut parent = 0usize; // the anchor's slot in the first row's table
                for i in 0..rows_of[s] {
                    let g = off[s] as usize + i;
                    let score = |c: usize| vals[g * k + c] + edge[(g * k + parent) * k + c];
                    let mut best = 0usize;
                    for c in 1..k {
                        if score(c) > score(best) {
                            best = c;
                        }
                    }
                    assert_eq!(ids[g * k + best], picks[g], "seq {s} row {i}: the lattice chain left the walk's pick");
                    assert_eq!(score(best).to_bits(), conf[g].to_bits(), "seq {s} row {i}: score bits");
                    parent = best;
                }
            }
            // a first row has one parent: the other entries are zero (never read)
            for s in 0..n_seq {
                let g = off[s] as usize;
                assert!(
                    edge[(g * k + 1) * k..(g * k + k) * k].iter().all(|&e| e == 0.0),
                    "seq {s}: first-row parents past the anchor"
                );
            }
        }

        /// With a zero gate the selector score is the unary logit, so the margin is checkable on the host: the gap between the two
        /// best logits when greedy, and the gap between the two best noised logits (the target's own Gumbel draw, in temperature
        /// units) when keyed.
        #[test]
        fn the_walk_margin_is_the_lead_over_the_runner_up() {
            let Some(st) = dev() else { return };
            let (v, rank, rows, k) = (2000usize, 256usize, 4usize, 16usize);
            let mut rng = Rng(808);
            let cb = |rng: &mut Rng| -> Vec<u16> { rng.vec(v * rank, -1.0, 1.0).iter().map(|&x| f64_to_f16(x as f64)).collect() };
            let (dpred, dsucc) = (DevBuf::from_slice(&cb(&mut rng)).unwrap(), DevBuf::from_slice(&cb(&mut rng)).unwrap());
            let n_seq = 3usize;
            let temps = [0.0f64, 1.0, 0.7];
            let (mut ids, mut vals) = (Vec::new(), Vec::new());
            for _ in 0..n_seq * rows {
                let mut row: Vec<i32> = Vec::new();
                while row.len() < k {
                    let c = (rng.next_u64() % v as u64) as i32;
                    if !row.contains(&c) {
                        row.push(c);
                    }
                }
                let mut u: Vec<f32> = rng.vec(k, 0.0, 6.0);
                u.sort_by(|a, b| b.partial_cmp(a).unwrap());
                ids.extend(row);
                vals.extend(u);
            }
            let rec: Vec<WalkSampling> = (0..n_seq)
                .map(|s| WalkSampling {
                    seed: 31 + s as u64,
                    first_position: 100 + 10 * s as u64,
                    temperature: temps[s],
                    top_p: 1.0,
                    min_p: 0.0,
                    keyed: (temps[s] > 0.0) as u32,
                    pad: 0,
                })
                .collect();
            let off: Vec<i32> = (0..=n_seq).map(|s| (s * rows) as i32).collect();
            let zero_gate = vec![0u16; n_seq * rows * rank];
            let (da, di, dv, dg, dr, doff) = (
                DevBuf::from_slice(&[1i32, 2, 3]).unwrap(),
                DevBuf::from_slice(&ids).unwrap(),
                DevBuf::from_slice(&vals).unwrap(),
                DevBuf::from_slice(&zero_gate).unwrap(),
                DevBuf::from_slice(&rec).unwrap(),
                DevBuf::from_slice(&off).unwrap(),
            );
            let mut out = DevBuf::zeroed(n_seq * rows * 4).unwrap();
            let mut margin = DevBuf::zeroed(n_seq * rows * 4).unwrap();
            selector_walk_ragged(
                &da,
                &di,
                &dv,
                &dg,
                GateDtype::Bf16,
                &dpred,
                &dsucc,
                n_seq,
                n_seq * rows,
                k,
                rank,
                &mut out,
                Some(&dr),
                &doff,
                None,
                Some(&mut margin),
                &st,
            )
            .unwrap();
            st.sync().unwrap();
            let got: Vec<f32> = margin.to_vec().unwrap();
            for s in 0..n_seq {
                for t in 0..rows {
                    let at = (s * rows + t) * k;
                    let want = if temps[s] == 0.0 {
                        (vals[at] - vals[at + 1]) as f64
                    } else {
                        // candidates are sorted by logit; the kernel adds the keyed Gumbel of each id at this draft's position
                        let mut z: Vec<f64> = (0..k)
                            .map(|c| {
                                let u = gb_sampling::uniform(rec[s].seed, rec[s].first_position + t as u64, ids[at + c] as u64);
                                vals[at + c] as f64 / temps[s] + (-(-u.ln()).ln())
                            })
                            .collect();
                        z.sort_by(|a, b| b.partial_cmp(a).unwrap());
                        (z[0] - z[1]).min(30.0)
                    };
                    let g = got[s * rows + t] as f64;
                    assert!(
                        (g - want).abs() < 1e-3 * (1.0 + want.abs()),
                        "seq {s} row {t} (T={}): kernel {g}, host {want}",
                        temps[s]
                    );
                }
            }
        }

        /// Compact ids from a restricted candidate set come back as the token ids they stand for.
        #[test]
        fn remap_ids_follows_the_ranges() {
            let Some(st) = dev() else { return };
            // two ranges: tokens 0..300 and tokens 1000..1100 laid end to end (400 compact ids)
            let mut ids: Vec<i32> = vec![0, 1, 299, 300, 301, 399, 150];
            let mut d = DevBuf::from_slice(&ids).unwrap();
            remap_ids(&mut d, ids.len(), &[(0, 300), (1000, 100)], &st).unwrap();
            st.sync().unwrap();
            ids = d.to_vec().unwrap();
            assert_eq!(ids, vec![0, 1, 299, 1000, 1001, 1099, 150]);
            // three ranges laid end to end: compact 0..10 -> tokens 100..110, compact 10 -> token 500, compact 11.. -> tokens 900..
            let mut d = DevBuf::from_slice(&[5i32, 10, 11, 25]).unwrap();
            remap_ids(&mut d, 4, &[(100, 10), (500, 1), (900, 100)], &st).unwrap();
            st.sync().unwrap();
            assert_eq!(d.to_vec::<i32>().unwrap(), vec![105, 500, 900, 914]);
        }

        /// The selector walk on the real codebooks against EX's own selected path, all three dumped rounds.
        #[test]
        fn walk_reproduces_ex_path_with_real_codebooks() {
            let Some(st) = dev() else { return };
            if !have_refs() || !pack_dir().exists() {
                eprintln!("no reference dumps or draft pack: skipping");
                return;
            }
            let ck = scan(&pack_dir(), false).unwrap();
            let pc = plain_u16(&ck, "candidate_selector.predecessor_codebook.weight");
            let sc = plain_u16(&ck, "candidate_selector.successor_codebook.weight");
            let wsel = plain_f32(&ck, "candidate_selector.hidden_projection.weight"); // [256, 5120]
            let (dpred, dsucc) = (DevBuf::from_slice(&pc).unwrap(), DevBuf::from_slice(&sc).unwrap());
            let (rank, rows, k) = (256usize, 7usize, 16usize);
            for round in 0..3 {
                let hid = ref_npy(&format!("draft/round_{round}/sel_hidden.npy")).to_f32(); // [7, 5120]
                let logits = ref_npy(&format!("draft/round_{round}/sel_logits.npy")); // [1, 7, 248320] f16
                assert_eq!(logits.shape, vec![1, 7, 248320]);
                let lg = logits.to_f32();
                let want: Vec<i64> = ref_npy(&format!("draft/round_{round}/path.npy")).to_i64();
                let anchor = ref_npy(&format!("draft/round_{round}/anchor_ids.npy")).to_i64()[0] as u32;
                assert_eq!(want[0] as u32, anchor);
                let mut ids = Vec::new();
                let mut vals = Vec::new();
                for t in 0..rows {
                    for (i, v) in cpu::topk_row(&lg[t * 248320..(t + 1) * 248320], k) {
                        ids.push(i);
                        vals.push(v);
                    }
                }
                // gate = state @ Wsel^T, EX holds it in fp16
                let mut gate = vec![0f32; rows * rank];
                for t in 0..rows {
                    for r in 0..rank {
                        let s: f64 = (0..5120).map(|i| hid[t * 5120 + i] as f64 * wsel[r * 5120 + i] as f64).sum();
                        gate[t * rank + r] = f16_to_f64(f64_to_f16(s)) as f32;
                    }
                }
                let w = [WalkRef { anchor, ids: ids.clone(), vals: vals.clone(), gate: gate.clone() }];
                for f32_gate in [true, false] {
                    let got = run_walk_gpu(&st, &w, &dpred, &dsucc, rows, k, rank, f32_gate);
                    let want7: Vec<i32> = want[1..].iter().map(|&x| x as i32).collect();
                    println!("round {round} f32_gate {f32_gate}: {got:?} want {want7:?}");
                    assert_eq!(got, want7, "round {round} f32_gate {f32_gate}");
                }
                // the f64 CPU reference agrees too
                let unary: Vec<Vec<f32>> = (0..rows).map(|t| vals[t * k..(t + 1) * k].to_vec()).collect();
                let cand: Vec<Vec<u32>> = (0..rows).map(|t| ids[t * k..(t + 1) * k].to_vec()).collect();
                let gt: Vec<Vec<f32>> = (0..rows).map(|t| gate[t * rank..(t + 1) * rank].to_vec()).collect();
                let pf = |id: u32| -> Vec<f32> {
                    pc[id as usize * rank..(id as usize + 1) * rank].iter().map(|&h| f16_to_f64(h) as f32).collect()
                };
                let sf = |id: u32| -> Vec<f32> {
                    sc[id as usize * rank..(id as usize + 1) * rank].iter().map(|&h| f16_to_f64(h) as f32).collect()
                };
                let cpu_path = cpu::selector_walk(anchor, &unary, &cand, &gt, &pf, &sf);
                assert_eq!(cpu_path.iter().map(|&x| x as i64).collect::<Vec<_>>(), want[1..].to_vec());
            }
        }

        // ------------------------------------------------------------------------------------------------ benchmark

        #[test]
        #[ignore = "benchmark: cargo test --release -p gb-cuda dflash2 -- --ignored --nocapture"]
        fn bench_kernels() {
            let Some(st) = dev() else { return };
            let iters = 200;
            let time = |f: &mut dyn FnMut()| -> f64 {
                for _ in 0..10 {
                    f();
                }
                st.sync().unwrap();
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    f();
                }
                st.sync().unwrap();
                t0.elapsed().as_secs_f64() / iters as f64 * 1e6
            };
            let mut rng = Rng(3);
            // dynconv, 8 rows
            let (h, g) = (D.hidden, D.groups());
            for n_seq in [1usize, 8] {
                let r = n_seq * 8;
                let u =
                    DevBuf::from_slice(&rng.vec(r * h, -1.0, 1.0).iter().map(|&x| f32_to_bf16(x)).collect::<Vec<_>>()).unwrap();
                let dy = DevBuf::from_slice(&rng.vec(r * 4 * g, -1.0, 1.0).iter().map(|&x| f32_to_bf16(x)).collect::<Vec<_>>())
                    .unwrap();
                let base = DevBuf::zeroed(4 * h * 2).unwrap();
                let mut out = DevBuf::zeroed(r * h * 2).unwrap();
                let us = time(&mut || dynconv(&u, &dy, &base, 0, n_seq, 8, h, 16, None, &mut out, &st).unwrap());
                println!("dynconv n_seq {n_seq}: {us:.1} us/launch");
            }
            // ctx_kv
            let layout = DraftCtxLayout::qwen38(1);
            let mut cache = DraftCache::new(layout).unwrap();
            for n in [8usize, 64, 512] {
                let kp = DevBuf::from_slice(&rng.vec(n * 1024, -1.0, 1.0).iter().map(|&x| f32_to_bf16(x)).collect::<Vec<_>>())
                    .unwrap();
                let gm = DevBuf::from_slice(&[1.0f32; 128]).unwrap();
                let pos = DevBuf::from_slice(&(0..n as i32).collect::<Vec<_>>()).unwrap();
                let sl = DevBuf::zeroed(n * 4).unwrap();
                let us = time(&mut || cache.write_ctx(0, &kp, &kp, &gm, &pos, &sl, n, 1e-6, 1e7, &st).unwrap());
                println!("ctx_kv n {n}: {us:.1} us/launch");
            }
            drop(cache);
            // attention
            for (ps, sl) in
                [(vec![2048usize], vec![0usize]), (vec![512], vec![0]), (vec![2048, 2048, 2048, 2048], vec![0, 1, 2, 3])]
            {
                let us = run_attn_case(&st, &ps, &sl, 9, 100);
                println!("attn n_seq {} P {}: {us:.1} us/layer", ps.len(), ps[0]);
            }
            // walk with a full-size codebook
            let (v, rank, rows, k) = (248320usize, 256usize, 7usize, 16usize);
            let cbh: Vec<u16> = (0..v * rank).map(|i| f64_to_f16(((i % 97) as f64 - 48.0) / 48.0)).collect();
            let (dp, ds) = (DevBuf::from_slice(&cbh).unwrap(), DevBuf::from_slice(&cbh).unwrap());
            for n_seq in [1usize, 8] {
                let ids: Vec<i32> = (0..n_seq * rows * k).map(|i| ((i * 7919) % v) as i32).collect();
                let (di, dv) = (DevBuf::from_slice(&ids).unwrap(), DevBuf::zeroed(n_seq * rows * k * 4).unwrap());
                let da = DevBuf::zeroed(n_seq * 4).unwrap();
                let dg = DevBuf::zeroed(n_seq * rows * rank * 2).unwrap();
                let mut out = DevBuf::zeroed(n_seq * rows * 4).unwrap();
                let us = time(&mut || {
                    selector_walk(&da, &di, &dv, &dg, GateDtype::Bf16, &dp, &ds, n_seq, rows, k, rank, &mut out, None, &st)
                        .unwrap()
                });
                println!("walk n_seq {n_seq}: {us:.1} us/launch");
            }
        }
    }

    // ------------------------------------------------------------------------------- real pack, whole draft block

    mod real_pack {
        use super::*;
        use gb_exl3::{codebook, unpack, Codebook};
        use std::io::{Read, Seek, SeekFrom};
        use std::path::Path;

        /// A dequantised EXL3 layer for the CPU reference: `y = H_out( H_in(x * suh) @ W_q ) * svh` in f64
        /// (the same math as `gb_exl3::forward`, with a fast Hadamard and threads so the whole block runs in seconds).
        struct Exl3Cpu {
            w: Vec<f32>,
            k: usize,
            n: usize,
            suh: Vec<f64>,
            svh: Vec<f64>,
        }

        fn fwht128(x: &mut [f64]) {
            let s = 1.0 / 128f64.sqrt();
            for blk in x.chunks_mut(128) {
                let mut h = 1;
                while h < 128 {
                    for i in (0..128).step_by(h * 2) {
                        for j in i..i + h {
                            let (a, b) = (blk[j], blk[j + h]);
                            blk[j] = a + b;
                            blk[j + h] = a - b;
                        }
                    }
                    h *= 2;
                }
                for v in blk.iter_mut() {
                    *v *= s;
                }
            }
        }

        impl Exl3Cpu {
            fn load(ck: &Checkpoint, prefix: &str) -> Self {
                let lg = ck.load_group(prefix).unwrap_or_else(|e| panic!("{prefix}: {e}"));
                assert_eq!(lg.group.codebook, Codebook::Mul1);
                let table = codebook(lg.group.codebook);
                let lut: Vec<f32> = (0..65536u32).map(|i| f16_to_f64(i as u16) as f32).collect();
                let nw = lg.group.bits.tile_words();
                let (kt, nt) = (lg.kt, lg.nt);
                let per = kt.div_ceil(16);
                let parts: Vec<Vec<u16>> = std::thread::scope(|s| {
                    let hs: Vec<_> = (0..kt)
                        .step_by(per)
                        .map(|k0| {
                            let k1 = (k0 + per).min(kt);
                            let (tr, table, bits) = (&lg.trellis[k0 * nt * nw..k1 * nt * nw], &table, lg.group.bits);
                            s.spawn(move || unpack(tr, k1 - k0, nt, bits, table))
                        })
                        .collect();
                    hs.into_iter().map(|h| h.join().unwrap()).collect()
                });
                let w: Vec<f32> = parts.iter().flat_map(|p| p.iter().map(|&h| lut[h as usize])).collect();
                Self { w, k: lg.group.k, n: lg.group.n, suh: lg.suh, svh: lg.svh }
            }
        }

        impl cpu::Linear for Exl3Cpu {
            fn forward(&self, x: &[f64], rows: usize) -> Vec<f64> {
                let (k, n) = (self.k, self.n);
                assert_eq!(x.len(), rows * k);
                let mut xh: Vec<f64> = (0..rows * k).map(|i| x[i] * self.suh[i % k]).collect();
                fwht128(&mut xh);
                let cols_per = n.div_ceil(16).div_ceil(16) * 16;
                let parts: Vec<(usize, usize, Vec<f64>)> = std::thread::scope(|s| {
                    let hs: Vec<_> = (0..n)
                        .step_by(cols_per)
                        .map(|c0| {
                            let c1 = (c0 + cols_per).min(n);
                            let (xh, w) = (&xh, &self.w);
                            s.spawn(move || {
                                let cols = c1 - c0;
                                let mut local = vec![0f64; rows * cols];
                                for kk in 0..k {
                                    let wr = &w[kk * n + c0..kk * n + c1];
                                    for r in 0..rows {
                                        let xv = xh[r * k + kk];
                                        for (y, ww) in local[r * cols..(r + 1) * cols].iter_mut().zip(wr) {
                                            *y += xv * *ww as f64;
                                        }
                                    }
                                }
                                (c0, c1, local)
                            })
                        })
                        .collect();
                    hs.into_iter().map(|h| h.join().unwrap()).collect()
                });
                let mut y = vec![0f64; rows * n];
                for (c0, c1, local) in parts {
                    for r in 0..rows {
                        y[r * n + c0..r * n + c1].copy_from_slice(&local[r * (c1 - c0)..(r + 1) * (c1 - c0)]);
                    }
                }
                fwht128(&mut y);
                for (i, v) in y.iter_mut().enumerate() {
                    *v *= self.svh[i % n];
                }
                y
            }
        }

        struct Draft {
            layers: Vec<cpu::LayerWeights>,
            norm: Vec<f32>,
            hidden_norm: Vec<f32>,
            fc: Exl3Cpu,
        }

        fn load_draft(ck: &Checkpoint) -> Draft {
            let lin = |p: &str| -> Box<dyn cpu::Linear> { Box::new(Exl3Cpu::load(ck, p)) };
            let mut layers = Vec::new();
            for i in 0..5 {
                let pre = format!("layers.{i}");
                let conv = |name: &str| cpu::ConvWeights {
                    base: plain_f32(ck, &format!("{pre}.{name}.base_kernel")),
                    proj: Box::new(cpu::DenseLinear {
                        w: plain_f32(ck, &format!("{pre}.{name}.kernel_projection.weight")),
                        in_features: 5120,
                        out_features: 1280,
                    }),
                };
                layers.push(cpu::LayerWeights {
                    input_ln: plain_f32(ck, &format!("{pre}.input_layernorm.weight")),
                    attn_conv: conv("attention_conv"),
                    q: lin(&format!("{pre}.self_attn.q_proj")),
                    k: lin(&format!("{pre}.self_attn.k_proj")),
                    v: lin(&format!("{pre}.self_attn.v_proj")),
                    o: lin(&format!("{pre}.self_attn.o_proj")),
                    q_norm: plain_f32(ck, &format!("{pre}.self_attn.q_norm.weight")),
                    k_norm: plain_f32(ck, &format!("{pre}.self_attn.k_norm.weight")),
                    post_ln: plain_f32(ck, &format!("{pre}.post_attention_layernorm.weight")),
                    mlp_conv: conv("mlp_conv"),
                    gate: lin(&format!("{pre}.mlp.gate_proj")),
                    up: lin(&format!("{pre}.mlp.up_proj")),
                    down: lin(&format!("{pre}.mlp.down_proj")),
                });
            }
            Draft {
                layers,
                norm: plain_f32(ck, "norm.weight"),
                hidden_norm: plain_f32(ck, "hidden_norm.weight"),
                fc: Exl3Cpu::load(ck, "fc"),
            }
        }

        /// Rows of the target's bf16 embedding table, read by seeking in the shard.
        fn embed_rows(ids: &[usize]) -> Vec<f64> {
            let dir =
                Path::new(&std::env::var("REXL3_QWEN_DIR").or_else(|_| std::env::var("REXL3_TARGET_PACK")).unwrap_or_default())
                    .to_path_buf();
            let ck = scan(&dir, false).unwrap();
            let e = ck.plain.get("model.language_model.embed_tokens.weight").expect("embed_tokens");
            assert_eq!(e.dtype, "BF16");
            let mut f = std::fs::File::open(dir.join(&e.file)).unwrap();
            let mut len = [0u8; 8];
            f.read_exact(&mut len).unwrap();
            let data0 = 8 + u64::from_le_bytes(len) + e.offsets.0;
            let mut out = Vec::new();
            for &id in ids {
                f.seek(SeekFrom::Start(data0 + (id * 5120 * 2) as u64)).unwrap();
                let mut b = vec![0u8; 5120 * 2];
                f.read_exact(&mut b).unwrap();
                out.extend(b.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64));
            }
            out
        }

        fn npy_f64(rel: &str) -> Vec<f64> {
            ref_npy(rel).to_f32().iter().map(|&x| x as f64).collect()
        }

        /// Per-layer K and V projections of the first `n` context rows (taps_call_0..2 concatenated, positions 0..n).
        fn ctx_projections(dr: &Draft, n: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
            let mut taps = vec![0f64; n * 25600];
            let mut row0 = 0;
            for call in 0..3 {
                let rows = ref_npy(&format!("draft/taps_call_{call}/tap_0.npy")).shape[1];
                for tap in 0..5 {
                    let t = npy_f64(&format!("draft/taps_call_{call}/tap_{tap}.npy"));
                    for r in 0..rows {
                        if row0 + r < n {
                            let o = (row0 + r) * 25600 + tap * 5120;
                            taps[o..o + 5120].copy_from_slice(&t[r * 5120..(r + 1) * 5120]);
                        }
                    }
                }
                row0 += rows;
            }
            assert!(row0 >= n);
            let ctx_c = cpu::ctx_hidden(&D, &taps, n, &dr.fc, &dr.hidden_norm);
            dr.layers.iter().map(|l| (l.k.forward(&ctx_c, n), l.v.forward(&ctx_c, n))).collect()
        }

        fn round_inputs(round: usize) -> (usize, Vec<f64>) {
            let p = ref_npy(&format!("draft/round_{round}/cache_seqlens.npy")).to_f32()[0] as usize;
            let anchor = ref_npy(&format!("draft/round_{round}/anchor_ids.npy")).to_i64()[0] as usize;
            let mut ids = vec![248070usize; 8]; // MASK
            ids[0] = anchor;
            (p, embed_rows(&ids))
        }

        struct GpuOps {
            st: crate::Stream,
            cache: DraftCache,
            p: usize,
            ws: crate::DevBuf,
            /// f64 CPU op with bf16 boundaries, fed the very same inputs at every call: worst per-call relRMS is recorded
            yard: Bf16Ops,
            worst_dynconv: f64,
            worst_attn: f64,
        }

        impl cpu::DraftOps for GpuOps {
            fn dynconv(&mut self, d: &Dims, u: &[f64], rows: usize, dyn_: &[f64], base: &[f32], which: usize) -> Vec<f64> {
                use crate::DevBuf;
                let (du, dd) = (DevBuf::from_slice(&bfs(u)).unwrap(), DevBuf::from_slice(&bfs(dyn_)).unwrap());
                let db = DevBuf::from_slice(&base.iter().map(|&b| f64_to_f16(b as f64)).collect::<Vec<_>>()).unwrap();
                let mut out = DevBuf::zeroed(rows * d.hidden * 2).unwrap();
                dynconv(&du, &dd, &db, which, 1, rows, d.hidden, d.conv_group, None, &mut out, &self.st).unwrap();
                self.st.sync().unwrap();
                let got = unbf(&out.to_vec::<u16>().unwrap());
                let want = cpu::DraftOps::dynconv(&mut self.yard, d, u, rows, dyn_, base, which);
                self.worst_dynconv = self.worst_dynconv.max(cos_rel_rms(&f32s(&got), &f32s(&want)).1);
                got
            }
            fn attention(&mut self, d: &Dims, layer: usize, q: &[f64], bk: &[f64], bv: &[f64]) -> Vec<f64> {
                use crate::DevBuf;
                let (dq, dk, dv) = (
                    DevBuf::from_slice(&bfs(q)).unwrap(),
                    DevBuf::from_slice(&bfs(bk)).unwrap(),
                    DevBuf::from_slice(&bfs(bv)).unwrap(),
                );
                let (slots, p) = (DevBuf::from_slice(&[0i32]).unwrap(), DevBuf::from_slice(&[self.p as i32]).unwrap());
                let mut out = DevBuf::zeroed(8 * d.q_heads * d.head_dim * 2).unwrap();
                let l = self.cache.layout;
                let (kc, vc) = (self.cache.k(layer), self.cache.v(layer));
                attn(
                    &dq,
                    &dk,
                    &dv,
                    kc,
                    vc,
                    &slots,
                    &p,
                    1,
                    d.q_heads,
                    d.kv_heads,
                    l.cap,
                    l.window,
                    8,
                    &mut self.ws,
                    &mut out,
                    &self.st,
                )
                .unwrap();
                self.st.sync().unwrap();
                let got = unbf(&out.to_vec::<u16>().unwrap());
                let want = cpu::DraftOps::attention(&mut self.yard, d, layer, q, bk, bv);
                self.worst_attn = self.worst_attn.max(cos_rel_rms(&f32s(&got), &f32s(&want)).1);
                got
            }
        }

        /// CPU ops that round every kernel input and output to bf16 like the GPU kernels' boundaries (but compute in f64):
        /// the yardstick for "is the GPU kernel error only bf16 rounding".
        struct Bf16Ops(cpu::CpuOps);

        impl cpu::DraftOps for Bf16Ops {
            fn dynconv(&mut self, d: &Dims, u: &[f64], rows: usize, dyn_: &[f64], base: &[f32], which: usize) -> Vec<f64> {
                rbf(&self.0.dynconv(d, &rbf(u), rows, &rbf(dyn_), base, which))
            }
            fn attention(&mut self, d: &Dims, layer: usize, q: &[f64], bk: &[f64], bv: &[f64]) -> Vec<f64> {
                rbf(&self.0.attention(d, layer, &rbf(q), &rbf(bk), &rbf(bv)))
            }
        }

        #[test]
        fn fast_exl3_cpu_matches_gb_exl3_forward() {
            let dir = pack_dir();
            if !dir.exists() {
                eprintln!("no draft pack: skipping");
                return;
            }
            let ck = scan(&dir, false).unwrap();
            let lg = ck.load_group("layers.0.self_attn.k_proj").unwrap();
            let me = Exl3Cpu::load(&ck, "layers.0.self_attn.k_proj");
            let mut rng = Rng(4);
            let x: Vec<f64> = rng.vec(2 * 5120, -2.0, 2.0).iter().map(|&v| v as f64).collect();
            let want = gb_exl3::forward(
                &x,
                2,
                &lg.trellis,
                lg.kt,
                lg.nt,
                lg.group.bits,
                &codebook(lg.group.codebook),
                &lg.suh,
                &lg.svh,
                None,
            );
            let got = cpu::Linear::forward(&me, &x, 2);
            assert_close("fast EXL3 vs gb_exl3::forward", &got, &want, 0.9999999999, 1e-9);
        }

        /// The CPU reference of the whole draft block (spec 8.3, EXL3 weights dequantised in f64) against EX's captured
        /// `state` for the three dumped rounds, then the same block with dynconv + attention + ctx K/V on the GPU.
        #[test]
        #[cfg_attr(debug_assertions, ignore = "heavy f64 reference: run with --release")]
        fn draft_block_state_matches_ex() {
            let dir = pack_dir();
            if !have_refs() || !dir.exists() {
                eprintln!("no reference dumps or draft pack: skipping");
                return;
            }
            let t0 = std::time::Instant::now();
            let ck = scan(&dir, false).unwrap();
            let dr = load_draft(&ck);
            println!("weights loaded in {:.1}s", t0.elapsed().as_secs_f64());
            let proj = ctx_projections(&dr, 44);
            println!("context projections done at {:.1}s", t0.elapsed().as_secs_f64());
            let gpu = crate::device_available();
            for round in 0..3 {
                let (p, x0) = round_inputs(round);
                let want = npy_f64(&format!("draft/round_{round}/state.npy"));
                let positions: Vec<usize> = (0..p).collect();
                let mut ctx_k = Vec::new();
                let mut ctx_v = Vec::new();
                for (l, (kp, vp)) in proj.iter().enumerate() {
                    let (k, v) = cpu::ctx_kv(&D, &kp[..p * 1024], &vp[..p * 1024], &dr.layers[l].k_norm, &positions);
                    ctx_k.push(k);
                    ctx_v.push(v);
                }
                // the GPU ctx_kv kernel sees bf16 projections and stores bf16 K/V: mirror both in the yardstick
                let (mut rk, mut rv) = (Vec::new(), Vec::new());
                for (l, (kp, vp)) in proj.iter().enumerate() {
                    let (k, v) = cpu::ctx_kv(&D, &rbf(&kp[..p * 1024]), &rbf(&vp[..p * 1024]), &dr.layers[l].k_norm, &positions);
                    rk.push(rbf(&k));
                    rv.push(rbf(&v));
                }
                let mut ops = cpu::CpuOps { p, ctx_k, ctx_v };
                let state = cpu::block_forward(&D, &dr.layers, &dr.norm, &x0, p, &mut ops);
                let mut bops = Bf16Ops(cpu::CpuOps { p, ctx_k: rk.clone(), ctx_v: rv.clone() });
                let bstate = cpu::block_forward(&D, &dr.layers, &dr.norm, &x0, p, &mut bops);
                assert_close(&format!("round {round} CPU bf16-boundary block vs EX state"), &bstate, &want, 0.9999, 1e-2);
                assert_close(&format!("round {round} CPU f64 block vs EX state (P={p})"), &state, &want, 0.9999, 1e-2);
                for t in 0..8 {
                    let (cr, rr) = cos_rel_rms(&f32s(&state[t * 5120..(t + 1) * 5120]), &f32s(&want[t * 5120..(t + 1) * 5120]));
                    println!("   row {t}: cos {cr:.7} relRMS {rr:.2e}");
                    assert!(cr >= 0.9995, "row {t}");
                }

                if gpu {
                    use crate::{DevBuf, Stream};
                    let st = Stream::new().unwrap();
                    let mut cache = DraftCache::new(DraftCtxLayout::qwen38(1)).unwrap();
                    let (dpos, dslot) =
                        (DevBuf::from_slice(&(0..p as i32).collect::<Vec<_>>()).unwrap(), DevBuf::zeroed(p * 4).unwrap());
                    for (l, (kp, vp)) in proj.iter().enumerate() {
                        let gamma = DevBuf::from_slice(&dr.layers[l].k_norm).unwrap();
                        let (dk, dv) = (
                            DevBuf::from_slice(&bfs(&kp[..p * 1024])).unwrap(),
                            DevBuf::from_slice(&bfs(&vp[..p * 1024])).unwrap(),
                        );
                        cache.write_ctx(l, &dk, &dv, &gamma, &dpos, &dslot, p, 1e-6, 1e7, &st).unwrap();
                    }
                    st.sync().unwrap();
                    let ws = DevBuf::alloc(attn_workspace_bytes(1, 2048, 32, 8)).unwrap();
                    let yard = Bf16Ops(cpu::CpuOps { p, ctx_k: rk, ctx_v: rv });
                    let mut gops = GpuOps { st, cache, p, ws, yard, worst_dynconv: 0.0, worst_attn: 0.0 };
                    let gstate = cpu::block_forward(&D, &dr.layers, &dr.norm, &x0, p, &mut gops);
                    assert_close(
                        &format!("round {round} GPU dynconv+attn+ctx_kv block vs EX state"),
                        &gstate,
                        &want,
                        0.9999,
                        1e-2,
                    );
                    assert_close(&format!("round {round} GPU block vs CPU f64 block"), &gstate, &state, 0.9999, 1e-2);
                    // In-situ per-kernel accuracy on real activations: every dynconv / attention call of the block, GPU vs the f64
                    // reference fed the same bf16 inputs, is within a bf16 rounding of the result (relRMS 1.7e-3 is one rounding).
                    println!(
                        "round {round}: worst per-call relRMS vs f64 on identical inputs: dynconv {:.2e}, attention {:.2e}",
                        gops.worst_dynconv, gops.worst_attn
                    );
                    assert!(gops.worst_dynconv <= 3e-3 && gops.worst_attn <= 3e-3);
                    // End to end the GPU block sits as far from the bf16-boundary CPU block as either sits from f64 (~8e-3):
                    // the per-call differences above (rare 1-ulp ties) are amplified by the following layers, so this is a
                    // sanity bound at rounding-noise level, not a kernel-accuracy figure (the per-call numbers are).
                    assert_close(
                        &format!("round {round} GPU block vs CPU bf16-boundary block"),
                        &gstate,
                        &bstate,
                        0.9999,
                        1.5e-2,
                    );
                }
            }
            println!("total {:.1}s", t0.elapsed().as_secs_f64());
        }
    }
}
