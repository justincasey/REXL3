//! Gated delta net (linear attention) kernels: conv/gating prep, verify window, replay commit, prefill.
//!
//! Model dimensions: 16 key heads x 128 and 48 value heads x 128 (value head h reads key head h / 3), conv kernel 4 over 10240
//! channels laid out `[q 16x128 | k 16x128 | v 48x128]` (Qwen3.8-27B, the default shape), or 32 value heads (value head h reads key
//! head h / 2, 8192 channels: the Qwen3.5/3.6-35B-A3B text backbone of MiniCPM-V-4.7). The shape is a [`GdnShape`]; the module's
//! constants are the default shape's. Math: `docs/QWEN38_SPEC.md` section 5.
//!
//! Replay design (TensorFold): the committed recurrent state is READ-ONLY during a verify window. [`pre`] writes a
//! [`Record`] of the window's post-conv rows, [`window`] runs the recurrence from a register copy of the committed
//! state and writes only the outputs, and [`commit`] replays the accepted prefix of the record with the same `step()`
//! routine and writes the state back. A serial decode step is a 1-row window followed by `commit(keep = 1)`, so it
//! is bit-identical to a verify window followed by `commit(keep = m + 1)`. [`prefill`] is a separate fast path
//! (its summation order differs from the decode path; it is deterministic and chunk-invariant).
//!
//! Per-layer state pools, allocated by the caller and indexed by slot:
//! * S pool, f32 `[slots][value heads][128 value row j][128 key i]` ([`GdnShape::s_slot_elems`] floats per slot; `S_SLOT_ELEMS`
//!   for the default shape),
//! * conv pool, bf16 `[slots][3][conv channels]` (the last 3 pre-conv qkv rows, oldest first; [`GdnShape::conv_slot_elems`] per
//!   slot; `CONV_SLOT_ELEMS` for the default shape).
//!
//! Shapes: a [`Record`] carries the shape it was allocated for (the [`crate::attention::PagedKv`] pattern), and every function that
//! takes a record launches that shape's kernels, which are the one source of `csrc/gdn.cu` built twice (`csrc/gdn_hv32.cu`: the
//! 32-head launchers carry the suffix `_hv32`). The few functions with no record ([`commit_conv`], [`commit_conv_path`], [`prefill`],
//! [`prefill_part`], [`prefill_workspace_bytes`]) have a `_shaped` sibling that takes the shape; the unsuffixed name is the default
//! shape's, exactly as before. The 48-head kernels are unchanged by the parameter (tests/gdn.rs and its golden pin them).
//!
//! Sequences are described by device arrays ([`Batch`]): `seq_start` i32 `[n_seq + 1]` (row offsets into the
//! concatenated row arrays) and `slot` i32 `[n_seq]`. Rows of a sequence are consecutive and run in order.
//! Activations are bf16 stored as `u16`. Nothing here synchronizes the host or allocates device memory.
//!
//! Preconditions the host cannot cheaply check (violations are undefined behaviour on the device):
//! slots within one [`commit`]/[`prefill`] launch are distinct; every `keep[i]` is at most the length of sequence i
//! (it is clamped on the device, never read out of range); `seq_start` is non-decreasing and `seq_start[n_seq]` equals
//! `Batch::rows`; every `slot[i] < Batch::n_slots`; prefill sequences are at most 4096 rows per call (not enforced by the kernel,
//! only by the workspace the caller sizes).

use crate::rt::{CudaError, DevBuf};

pub const KEY_HEADS: usize = 16;
pub const VALUE_HEADS: usize = 48;
pub const HEAD_DIM: usize = 128;
pub const CONV_DIM: usize = 10240;
pub const CONV_TAPS: usize = 4;
/// Pre-conv qkv rows kept per slot.
pub const CONV_STATE_ROWS: usize = 3;
/// f32 elements of one slot of the S pool.
pub const S_SLOT_ELEMS: usize = VALUE_HEADS * HEAD_DIM * HEAD_DIM;
/// bf16 elements of one slot of the conv pool.
pub const CONV_SLOT_ELEMS: usize = CONV_STATE_ROWS * CONV_DIM;
/// Longest sequence a single [`prefill`] call is meant for.
pub const PREFILL_MAX_ROWS: usize = 4096;

/// The value-head counts the kernels are built for (16 key heads and a head dim of 128 in both).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum GdnShape {
    /// 16 key x 48 value heads, 10240 conv channels: Qwen3.8-27B (and every shape this crate had before the parameter).
    #[default]
    Hv48,
    /// 16 key x 32 value heads, 8192 conv channels: the Qwen3.5/3.6-35B-A3B backbone (MiniCPM-V-4.7).
    Hv32,
    /// 16 key x 64 value heads, 12288 conv channels: Qwen3.5-122B-A10B / Ornith-1.5-122B.
    Hv64,
}

impl GdnShape {
    /// Every shape the kernels are built for.
    pub const ALL: [GdnShape; 3] = [GdnShape::Hv48, GdnShape::Hv32, GdnShape::Hv64];

    /// The shape for a config's head counts and dims, if the kernels are built for it (key and value head dim equal).
    pub const fn from_counts(key_heads: usize, value_heads: usize, key_dim: usize, value_dim: usize) -> Option<Self> {
        if key_heads != KEY_HEADS || key_dim != HEAD_DIM || value_dim != HEAD_DIM {
            return None;
        }
        match value_heads {
            48 => Some(Self::Hv48),
            32 => Some(Self::Hv32),
            64 => Some(Self::Hv64),
            _ => None,
        }
    }
    pub const fn key_heads(self) -> usize {
        KEY_HEADS
    }
    pub const fn value_heads(self) -> usize {
        match self {
            Self::Hv48 => 48,
            Self::Hv32 => 32,
            Self::Hv64 => 64,
        }
    }
    pub const fn head_dim(self) -> usize {
        HEAD_DIM
    }
    /// Value heads that share one key head.
    pub const fn values_per_key(self) -> usize {
        self.value_heads() / KEY_HEADS
    }
    /// Width of the q (and of the k) block of the conv channels: key heads x head dim.
    pub const fn q_dim(self) -> usize {
        KEY_HEADS * HEAD_DIM
    }
    /// Width of the v block of the conv channels and of the core output: value heads x head dim.
    pub const fn v_dim(self) -> usize {
        self.value_heads() * HEAD_DIM
    }
    /// Conv channels of a row: `[q | k | v]`.
    pub const fn conv_dim(self) -> usize {
        2 * self.q_dim() + self.v_dim()
    }
    /// f32 elements of one slot of the S pool.
    pub const fn s_slot_elems(self) -> usize {
        self.value_heads() * HEAD_DIM * HEAD_DIM
    }
    /// bf16 elements of one slot of the conv pool.
    pub const fn conv_slot_elems(self) -> usize {
        CONV_STATE_ROWS * self.conv_dim()
    }
}

// the module constants are the default shape's
const _: () = {
    assert!(GdnShape::Hv48.value_heads() == VALUE_HEADS);
    assert!(GdnShape::Hv48.conv_dim() == CONV_DIM);
    assert!(GdnShape::Hv48.s_slot_elems() == S_SLOT_ELEMS);
    assert!(GdnShape::Hv48.conv_slot_elems() == CONV_SLOT_ELEMS);
    assert!(GdnShape::Hv32.conv_dim() == 8192);
};

/// Sequences of one launch, borrowed from device arrays the caller already owns (a per-round buffer, a graph
/// bucket): `seq_start` i32 `[n_seq + 1]`, `slot` i32 `[n_seq]`. Nothing is allocated or copied per launch.
#[derive(Clone, Copy)]
pub struct Batch<'a> {
    pub seq_start: &'a DevBuf,
    pub slot: &'a DevBuf,
    pub n_seq: usize,
    /// Total rows (`seq_start[n_seq]`); the row buffers must hold at least this many.
    pub rows: usize,
    /// Slots the pools hold; every `slot[i]` must be below it.
    pub n_slots: usize,
}

/// An owning [`Batch`] built from host lengths and slots (uploads once; for tests, benches and non-hot paths).
pub struct BatchBuf {
    seq_start: DevBuf,
    slot: DevBuf,
    n_seq: usize,
    rows: usize,
    n_slots: usize,
}

impl BatchBuf {
    /// `lens[i]` rows for sequence i (consecutive in the row arrays), pool slot `slots[i]`; the pools are assumed to
    /// hold `max(slots) + 1` slots (use [`BatchBuf::with_pool_slots`] when they hold more).
    pub fn new(lens: &[usize], slots: &[usize]) -> Result<Self, CudaError> {
        Self::with_pool_slots(lens, slots, slots.iter().copied().max().map_or(0, |m| m + 1))
    }
    pub fn with_pool_slots(lens: &[usize], slots: &[usize], n_slots: usize) -> Result<Self, CudaError> {
        assert_eq!(lens.len(), slots.len());
        assert!(slots.iter().all(|&s| s < n_slots), "slot out of range");
        let mut start = vec![0i32; lens.len() + 1];
        for (i, &l) in lens.iter().enumerate() {
            start[i + 1] = start[i] + l as i32;
        }
        let slot: Vec<i32> = slots.iter().map(|&s| s as i32).collect();
        Ok(Self {
            seq_start: DevBuf::from_slice(&start)?,
            slot: DevBuf::from_slice(&slot)?,
            n_seq: lens.len(),
            rows: start[lens.len()] as usize,
            n_slots,
        })
    }
    pub fn view(&self) -> Batch<'_> {
        Batch { seq_start: &self.seq_start, slot: &self.slot, n_seq: self.n_seq, rows: self.rows, n_slots: self.n_slots }
    }
}

/// The weights `pre` needs besides the conv weight's own layout.
pub struct PreWeights<'a> {
    /// bf16 `[conv channels, 4]`: `weight[ch, 0, k]`.
    pub conv_w: &'a DevBuf,
    /// f32 `[value heads]` (the loader converts the pack's bf16).
    pub a_log: &'a DevBuf,
    /// f32 `[value heads]`.
    pub dt_bias: &'a DevBuf,
}

/// Raw projection outputs of the rows: `qkv` bf16 `[rows, conv channels]`, `a` and `b` bf16 `[rows, value heads]`.
pub struct PreInputs<'a> {
    pub qkv: &'a DevBuf,
    pub a: &'a DevBuf,
    pub b: &'a DevBuf,
}

/// A window's post-conv rows (what replay needs): q, k bf16 `[rows,16,128]`, v bf16 `[rows,value heads,128]`, decay and
/// beta f32 `[rows,value heads]`.
pub struct Record {
    pub rows: usize,
    pub q: DevBuf,
    pub k: DevBuf,
    pub v: DevBuf,
    pub g_decay: DevBuf,
    pub beta: DevBuf,
    /// The shape the buffers are laid out for; every kernel call on this record uses it.
    pub shape: GdnShape,
}

impl Record {
    /// Allocates a record of the default (48 value heads) shape.
    pub fn alloc(rows: usize) -> Result<Self, CudaError> {
        Self::alloc_shaped(rows, GdnShape::Hv48)
    }
    /// Allocates a record of `shape`.
    pub fn alloc_shaped(rows: usize, shape: GdnShape) -> Result<Self, CudaError> {
        let (q, v) = (shape.q_dim(), shape.v_dim());
        Ok(Self {
            rows,
            q: DevBuf::alloc(rows * q * 2)?,
            k: DevBuf::alloc(rows * q * 2)?,
            v: DevBuf::alloc(rows * v * 2)?,
            g_decay: DevBuf::alloc(rows * shape.value_heads() * 4)?,
            beta: DevBuf::alloc(rows * shape.value_heads() * 4)?,
            shape,
        })
    }
}

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_gdn_pre(
            qkv: *const c_void,
            a: *const c_void,
            b: *const c_void,
            conv_w: *const c_void,
            a_log: *const f32,
            dt_bias: *const f32,
            conv_pool: *const c_void,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            rows: i32,
            q: *mut c_void,
            k: *mut c_void,
            v: *mut c_void,
            g: *mut f32,
            beta: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_window(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            s: *const f32,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            o: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_window_owed(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            o: *mut c_void,
            owed_k: *const c_void,
            owed_v: *const c_void,
            owed_g: *const f32,
            owed_beta: *const f32,
            owed_row0: *const i32,
            owed_keep: *const i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_commit_state(
            k: *const c_void,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_commit_conv(
            qkv: *const c_void,
            conv_pool: *mut c_void,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_commit_conv_all(
            layers: *const c_void,
            n_layers: i32,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_commit(
            k: *const c_void,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            qkv: *const c_void,
            s: *mut f32,
            conv_pool: *mut c_void,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_pre_tree(
            qkv: *const c_void,
            a: *const c_void,
            b: *const c_void,
            conv_w: *const c_void,
            a_log: *const f32,
            dt_bias: *const f32,
            conv_pool: *const c_void,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            rows: i32,
            q: *mut c_void,
            k: *mut c_void,
            v: *mut c_void,
            g: *mut f32,
            beta: *mut f32,
            parent: *const i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_window_tree(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            o: *mut c_void,
            owed_k: *const c_void,
            owed_v: *const c_void,
            owed_g: *const f32,
            owed_beta: *const f32,
            owed_row0: *const i32,
            owed_keep: *const i32,
            owed_rows: *const i32,
            owed_rows_stride: i32,
            plan: *const i32,
            plan_stride: i32,
            tier: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_commit_state_rows(
            k: *const c_void,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            rows: *const i32,
            rows_stride: i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_commit_conv_rows(
            qkv: *const c_void,
            conv_pool: *mut c_void,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            rows: *const i32,
            rows_stride: i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_prefill_workspace_bytes(rows: i32) -> usize;
        pub fn gb_gdn_prefill(
            qkv: *const c_void,
            a: *const c_void,
            b: *const c_void,
            conv_w: *const c_void,
            a_log: *const f32,
            dt_bias: *const f32,
            s: *mut f32,
            conv_pool: *mut c_void,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            rows: i32,
            workspace: *mut c_void,
            o: *mut c_void,
            stream: *mut c_void,
        ) -> i32;

        // the same launchers built for 32 value heads (csrc/gdn_hv32.cu)
        pub fn gb_gdn_pre_hv32(
            qkv: *const c_void,
            a: *const c_void,
            b: *const c_void,
            conv_w: *const c_void,
            a_log: *const f32,
            dt_bias: *const f32,
            conv_pool: *const c_void,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            rows: i32,
            q: *mut c_void,
            k: *mut c_void,
            v: *mut c_void,
            g: *mut f32,
            beta: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_window_hv32(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            s: *const f32,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            o: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_window_owed_hv32(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            o: *mut c_void,
            owed_k: *const c_void,
            owed_v: *const c_void,
            owed_g: *const f32,
            owed_beta: *const f32,
            owed_row0: *const i32,
            owed_keep: *const i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_commit_state_hv32(
            k: *const c_void,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_commit_conv_hv32(
            qkv: *const c_void,
            conv_pool: *mut c_void,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_commit_conv_all_hv32(
            layers: *const c_void,
            n_layers: i32,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_commit_hv32(
            k: *const c_void,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            qkv: *const c_void,
            s: *mut f32,
            conv_pool: *mut c_void,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_pre_tree_hv32(
            qkv: *const c_void,
            a: *const c_void,
            b: *const c_void,
            conv_w: *const c_void,
            a_log: *const f32,
            dt_bias: *const f32,
            conv_pool: *const c_void,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            rows: i32,
            q: *mut c_void,
            k: *mut c_void,
            v: *mut c_void,
            g: *mut f32,
            beta: *mut f32,
            parent: *const i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_window_tree_hv32(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            o: *mut c_void,
            owed_k: *const c_void,
            owed_v: *const c_void,
            owed_g: *const f32,
            owed_beta: *const f32,
            owed_row0: *const i32,
            owed_keep: *const i32,
            owed_rows: *const i32,
            owed_rows_stride: i32,
            plan: *const i32,
            plan_stride: i32,
            tier: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_commit_state_rows_hv32(
            k: *const c_void,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            rows: *const i32,
            rows_stride: i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_commit_conv_rows_hv32(
            qkv: *const c_void,
            conv_pool: *mut c_void,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            rows: *const i32,
            rows_stride: i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_prefill_workspace_bytes_hv32(rows: i32) -> usize;
        pub fn gb_gdn_prefill_hv32(
            qkv: *const c_void,
            a: *const c_void,
            b: *const c_void,
            conv_w: *const c_void,
            a_log: *const f32,
            dt_bias: *const f32,
            s: *mut f32,
            conv_pool: *mut c_void,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            rows: i32,
            workspace: *mut c_void,
            o: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        // the same launchers built for 64 value heads (csrc/gdn_hv64.cu)
        pub fn gb_gdn_pre_hv64(
            qkv: *const c_void,
            a: *const c_void,
            b: *const c_void,
            conv_w: *const c_void,
            a_log: *const f32,
            dt_bias: *const f32,
            conv_pool: *const c_void,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            rows: i32,
            q: *mut c_void,
            k: *mut c_void,
            v: *mut c_void,
            g: *mut f32,
            beta: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_window_hv64(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            s: *const f32,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            o: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_window_owed_hv64(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            o: *mut c_void,
            owed_k: *const c_void,
            owed_v: *const c_void,
            owed_g: *const f32,
            owed_beta: *const f32,
            owed_row0: *const i32,
            owed_keep: *const i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_commit_state_hv64(
            k: *const c_void,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_commit_conv_hv64(
            qkv: *const c_void,
            conv_pool: *mut c_void,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_commit_conv_all_hv64(
            layers: *const c_void,
            n_layers: i32,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_commit_hv64(
            k: *const c_void,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            qkv: *const c_void,
            s: *mut f32,
            conv_pool: *mut c_void,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_pre_tree_hv64(
            qkv: *const c_void,
            a: *const c_void,
            b: *const c_void,
            conv_w: *const c_void,
            a_log: *const f32,
            dt_bias: *const f32,
            conv_pool: *const c_void,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            rows: i32,
            q: *mut c_void,
            k: *mut c_void,
            v: *mut c_void,
            g: *mut f32,
            beta: *mut f32,
            parent: *const i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_window_tree_hv64(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            o: *mut c_void,
            owed_k: *const c_void,
            owed_v: *const c_void,
            owed_g: *const f32,
            owed_beta: *const f32,
            owed_row0: *const i32,
            owed_keep: *const i32,
            owed_rows: *const i32,
            owed_rows_stride: i32,
            plan: *const i32,
            plan_stride: i32,
            tier: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_commit_state_rows_hv64(
            k: *const c_void,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            rows: *const i32,
            rows_stride: i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_commit_conv_rows_hv64(
            qkv: *const c_void,
            conv_pool: *mut c_void,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            rows: *const i32,
            rows_stride: i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_gdn_prefill_workspace_bytes_hv64(rows: i32) -> usize;
        pub fn gb_gdn_prefill_hv64(
            qkv: *const c_void,
            a: *const c_void,
            b: *const c_void,
            conv_w: *const c_void,
            a_log: *const f32,
            dt_bias: *const f32,
            s: *mut f32,
            conv_pool: *mut c_void,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            rows: i32,
            workspace: *mut c_void,
            o: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
    }
}

use crate::rt::Stream;

/// The launcher of a shape: the default one, or its `_hv32` twin (the same signature).
macro_rules! launcher {
    ($shape:expr, $name:ident, $name32:ident, $name64:ident) => {
        match $shape {
            GdnShape::Hv48 => ffi::$name,
            GdnShape::Hv32 => ffi::$name32,
            GdnShape::Hv64 => ffi::$name64,
        }
    };
}

fn need(buf: &DevBuf, bytes: usize, what: &str) {
    assert!(buf.bytes() >= bytes, "gdn: {what} holds {} bytes, needs {bytes}", buf.bytes());
}

fn check_pools(shape: GdnShape, b: &Batch, s_pool: &DevBuf, conv_pool: &DevBuf) {
    need(s_pool, b.n_slots * shape.s_slot_elems() * 4, "S pool");
    need(conv_pool, b.n_slots * shape.conv_slot_elems() * 2, "conv pool");
}

fn check_inputs(shape: GdnShape, x: &PreInputs, rows: usize) {
    need(x.qkv, rows * shape.conv_dim() * 2, "qkv");
    need(x.a, rows * shape.value_heads() * 2, "a");
    need(x.b, rows * shape.value_heads() * 2, "b");
}

fn check_pre_weights(shape: GdnShape, w: &PreWeights) {
    need(w.conv_w, shape.conv_dim() * CONV_TAPS * 2, "conv_w");
    need(w.a_log, shape.value_heads() * 4, "a_log");
    need(w.dt_bias, shape.value_heads() * 4, "dt_bias");
}

/// Conv + silu + l2norm + gating for every row of the batch. Reads (never writes) `conv_pool`; fills `rec`
/// (`rec.rows >= batch.rows`).
pub fn pre(
    stream: &Stream,
    x: &PreInputs,
    w: &PreWeights,
    conv_pool: &DevBuf,
    batch: &Batch<'_>,
    rec: &mut Record,
) -> Result<(), CudaError> {
    let shape = rec.shape;
    assert!(rec.rows >= batch.rows, "gdn::pre: record has {} rows, batch {}", rec.rows, batch.rows);
    check_inputs(shape, x, batch.rows);
    check_pre_weights(shape, w);
    need(conv_pool, batch.n_slots * shape.conv_slot_elems() * 2, "conv pool");
    let rc = unsafe {
        launcher!(shape, gb_gdn_pre, gb_gdn_pre_hv32, gb_gdn_pre_hv64)(
            x.qkv.ptr(),
            x.a.ptr(),
            x.b.ptr(),
            w.conv_w.ptr(),
            w.a_log.ptr() as *const f32,
            w.dt_bias.ptr() as *const f32,
            conv_pool.ptr(),
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            batch.n_seq as i32,
            batch.rows as i32,
            rec.q.ptr(),
            rec.k.ptr(),
            rec.v.ptr(),
            rec.g_decay.ptr() as *mut f32,
            rec.beta.ptr() as *mut f32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Run the recurrence over every sequence's rows from the committed state (`s_pool` is only read) and write the
/// core output `o` bf16 `[rows, value heads, 128]`.
pub fn window(stream: &Stream, rec: &Record, s_pool: &DevBuf, batch: &Batch<'_>, o: &mut DevBuf) -> Result<(), CudaError> {
    let shape = rec.shape;
    assert!(rec.rows >= batch.rows);
    need(o, batch.rows * shape.v_dim() * 2, "o");
    need(s_pool, batch.n_slots * shape.s_slot_elems() * 4, "S pool");
    let rc = unsafe {
        launcher!(shape, gb_gdn_window, gb_gdn_window_hv32, gb_gdn_window_hv64)(
            rec.q.ptr(),
            rec.k.ptr(),
            rec.v.ptr(),
            rec.g_decay.ptr() as *const f32,
            rec.beta.ptr() as *const f32,
            s_pool.ptr() as *const f32,
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            batch.n_seq as i32,
            o.ptr(),
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Commit the first `keep[i]` rows of sequence i (i32 `[n_seq]` on the device, `0 <= keep[i] <= len_i`): replay them
/// into `s_pool` in place with the window's own `step()`, and shift the conv state (new = last 3 rows of
/// `concat(old 3, qkv rows 0..keep)`; `keep = 0` leaves a sequence untouched).
pub fn commit(
    stream: &Stream,
    rec: &Record,
    qkv: &DevBuf,
    s_pool: &mut DevBuf,
    conv_pool: &mut DevBuf,
    batch: &Batch<'_>,
    keep: &DevBuf,
) -> Result<(), CudaError> {
    let shape = rec.shape;
    assert!(rec.rows >= batch.rows);
    need(qkv, batch.rows * shape.conv_dim() * 2, "qkv");
    need(keep, batch.n_seq * 4, "keep");
    check_pools(shape, batch, s_pool, conv_pool);
    let rc = unsafe {
        launcher!(shape, gb_gdn_commit, gb_gdn_commit_hv32, gb_gdn_commit_hv64)(
            rec.k.ptr(),
            rec.v.ptr(),
            rec.g_decay.ptr() as *const f32,
            rec.beta.ptr() as *const f32,
            qkv.ptr(),
            s_pool.ptr() as *mut f32,
            conv_pool.ptr(),
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            keep.ptr() as *const i32,
            batch.n_seq as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// A commit the previous window still owes (the S half of [`commit`], deferred into the next window kernel): for the sequence
/// i of the NEW batch, the first `keep[i]` rows of its previous window, which start at row `row0[i]` of `rec`, have not been
/// replayed into the S pool. `keep[i] == 0` owes nothing. `0 <= keep[i] <=` the length of the sequence's previous window.
pub struct Owed<'a> {
    /// The previous window's record (the buffers [`pre`] filled for it, which must not have been overwritten since).
    pub rec: &'a Record,
    /// i32 `[n_seq]` of the new batch.
    pub row0: &'a DevBuf,
    /// i32 `[n_seq]` of the new batch.
    pub keep: &'a DevBuf,
}

/// [`window`] plus the commit the previous window still owes: for a sequence with `keep > 0` the kernel replays the owed rows
/// into `s_pool` first (the same `step()` [`commit`] runs, so the state is bit-identical to `commit` followed by `window`),
/// stores it, and runs this window from it. S is read once and written once per round instead of read by the window and
/// read and written again by the commit. Slots within one launch must be distinct (they are: one slot per sequence).
pub fn window_owed(
    stream: &Stream,
    rec: &Record,
    owed: &Owed<'_>,
    s_pool: &mut DevBuf,
    batch: &Batch<'_>,
    o: &mut DevBuf,
) -> Result<(), CudaError> {
    let shape = rec.shape;
    assert!(rec.rows >= batch.rows);
    assert_eq!(owed.rec.shape, shape, "gdn::window_owed: the owed record is of another shape");
    need(o, batch.rows * shape.v_dim() * 2, "o");
    need(s_pool, batch.n_slots * shape.s_slot_elems() * 4, "S pool");
    need(owed.row0, batch.n_seq * 4, "owed row0");
    need(owed.keep, batch.n_seq * 4, "owed keep");
    let rc = unsafe {
        launcher!(shape, gb_gdn_window_owed, gb_gdn_window_owed_hv32, gb_gdn_window_owed_hv64)(
            rec.q.ptr(),
            rec.k.ptr(),
            rec.v.ptr(),
            rec.g_decay.ptr() as *const f32,
            rec.beta.ptr() as *const f32,
            s_pool.ptr() as *mut f32,
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            batch.n_seq as i32,
            o.ptr(),
            owed.rec.k.ptr(),
            owed.rec.v.ptr(),
            owed.rec.g_decay.ptr() as *const f32,
            owed.rec.beta.ptr() as *const f32,
            owed.row0.ptr() as *const i32,
            owed.keep.ptr() as *const i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// The S half of [`commit`]: replay the first `keep[i]` rows of sequence i into `s_pool` in place. The conv history is not
/// touched (see [`commit_conv`]).
pub fn commit_state(
    stream: &Stream,
    rec: &Record,
    s_pool: &mut DevBuf,
    batch: &Batch<'_>,
    keep: &DevBuf,
) -> Result<(), CudaError> {
    let shape = rec.shape;
    assert!(rec.rows >= batch.rows);
    need(keep, batch.n_seq * 4, "keep");
    need(s_pool, batch.n_slots * shape.s_slot_elems() * 4, "S pool");
    let rc = unsafe {
        launcher!(shape, gb_gdn_commit_state, gb_gdn_commit_state_hv32, gb_gdn_commit_state_hv64)(
            rec.k.ptr(),
            rec.v.ptr(),
            rec.g_decay.ptr() as *const f32,
            rec.beta.ptr() as *const f32,
            s_pool.ptr() as *mut f32,
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            keep.ptr() as *const i32,
            batch.n_seq as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// The conv half of [`commit`]: new history = last 3 rows of `concat(old 3, qkv rows 0..keep)`. The default (48 value heads)
/// shape; [`commit_conv_shaped`] takes the shape.
pub fn commit_conv(
    stream: &Stream,
    qkv: &DevBuf,
    conv_pool: &mut DevBuf,
    batch: &Batch<'_>,
    keep: &DevBuf,
) -> Result<(), CudaError> {
    commit_conv_shaped(GdnShape::Hv48, stream, qkv, conv_pool, batch, keep)
}

/// [`commit_conv`] for `shape`.
pub fn commit_conv_shaped(
    shape: GdnShape,
    stream: &Stream,
    qkv: &DevBuf,
    conv_pool: &mut DevBuf,
    batch: &Batch<'_>,
    keep: &DevBuf,
) -> Result<(), CudaError> {
    need(qkv, batch.rows * shape.conv_dim() * 2, "qkv");
    need(keep, batch.n_seq * 4, "keep");
    need(conv_pool, batch.n_slots * shape.conv_slot_elems() * 2, "conv pool");
    let rc = unsafe {
        launcher!(shape, gb_gdn_commit_conv, gb_gdn_commit_conv_hv32, gb_gdn_commit_conv_hv64)(
            qkv.ptr(),
            conv_pool.ptr(),
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            keep.ptr() as *const i32,
            batch.n_seq as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// The table [`commit_conv_all`] reads: for each layer, the address of its window qkv rows and of its conv pool (two u64 each),
/// in a device buffer the caller keeps (the buffers' addresses never change).
pub fn conv_layer_table(qkv: &[&DevBuf], conv_pool: &[&DevBuf]) -> Result<DevBuf, CudaError> {
    assert_eq!(qkv.len(), conv_pool.len(), "one window buffer and one conv pool a layer");
    let words: Vec<u64> = qkv.iter().zip(conv_pool).flat_map(|(q, c)| [q.ptr() as u64, c.ptr() as u64]).collect();
    DevBuf::from_slice(&words)
}

/// [`commit_conv`] of every layer of `table` (see [`conv_layer_table`]) in one launch: each layer's pool ends with the bits a launch of
/// its own gives it. `shape` is the shape of every layer.
pub fn commit_conv_all(
    shape: GdnShape,
    stream: &Stream,
    table: &DevBuf,
    layers: usize,
    batch: &Batch<'_>,
    keep: &DevBuf,
) -> Result<(), CudaError> {
    need(table, layers * 16, "conv layer table");
    need(keep, batch.n_seq * 4, "keep");
    let rc = unsafe {
        launcher!(shape, gb_gdn_commit_conv_all, gb_gdn_commit_conv_all_hv32, gb_gdn_commit_conv_all_hv64)(
            table.ptr(),
            layers as i32,
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            keep.ptr() as *const i32,
            batch.n_seq as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

// ---------------------------------------------------------------------------------------------------------------------
// Tree windows (docs/TREE_DRAFTING.md, `crate::tree`): the same kernels over rows that form a tree per sequence. A row's outputs
// and the state it leaves are the bits serial decoding gives the token at the end of its root path; a chain is `parent[i] = i - 1`.
// ---------------------------------------------------------------------------------------------------------------------

/// [`pre`] over a tree window: `parent` i32 `[rows]` holds each row's parent as a GLOBAL row index (-1 for a sequence's root), so a
/// row's conv history is its own root path (its ancestors' raw qkv rows, then the committed history rows past the root) rather than
/// the rows before it in the window.
pub fn pre_tree(
    stream: &Stream,
    x: &PreInputs,
    w: &PreWeights,
    conv_pool: &DevBuf,
    batch: &Batch<'_>,
    rec: &mut Record,
    parent: &DevBuf,
) -> Result<(), CudaError> {
    let shape = rec.shape;
    assert!(rec.rows >= batch.rows, "gdn::pre_tree: record has {} rows, batch {}", rec.rows, batch.rows);
    check_inputs(shape, x, batch.rows);
    check_pre_weights(shape, w);
    need(conv_pool, batch.n_slots * shape.conv_slot_elems() * 2, "conv pool");
    need(parent, batch.rows * 4, "parent");
    let rc = unsafe {
        launcher!(shape, gb_gdn_pre_tree, gb_gdn_pre_tree_hv32, gb_gdn_pre_tree_hv64)(
            x.qkv.ptr(),
            x.a.ptr(),
            x.b.ptr(),
            w.conv_w.ptr(),
            w.a_log.ptr() as *const f32,
            w.dt_bias.ptr() as *const f32,
            conv_pool.ptr(),
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            batch.n_seq as i32,
            batch.rows as i32,
            rec.q.ptr(),
            rec.k.ptr(),
            rec.v.ptr(),
            rec.g_decay.ptr() as *mut f32,
            rec.beta.ptr() as *mut f32,
            parent.ptr() as *const i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// The rows an owed or committed PATH replays: `rows` i32 `[n_seq][stride]`, entry `j` of sequence `i` the GLOBAL record row of the
/// `j`-th accepted row (root first); `keep[i]` of them are valid.
#[derive(Clone, Copy)]
pub struct PathRows<'a> {
    pub rows: &'a DevBuf,
    pub stride: usize,
}

/// A commit the previous window still owes, whose rows are an accepted path of a tree window (see [`Owed`] for a prefix).
pub struct OwedPath<'a> {
    /// The previous window's record.
    pub rec: &'a Record,
    /// i32 `[n_seq]` of the new batch: how many path rows are owed (0: nothing).
    pub keep: &'a DevBuf,
    /// The path rows (global rows of `rec`) of each sequence of the new batch.
    pub path: PathRows<'a>,
}

/// [`window`] over a tree per sequence, with the previous window's accepted path (if any) replayed first. `plan` is i32
/// `[n_seq][plan_stride]` (`crate::tree::Plan::entries` per sequence, local node indices, `plan_stride >= 3 * longest window`);
/// `slots` the largest `Plan::slots` of the batch (the kernel's slot tier is the smallest that holds it). `o` row `r` is the core
/// output of window row `r`, whatever order the plan visits it in.
#[allow(clippy::too_many_arguments)]
pub fn window_tree(
    stream: &Stream,
    rec: &Record,
    owed: Option<&OwedPath<'_>>,
    s_pool: &mut DevBuf,
    batch: &Batch<'_>,
    o: &mut DevBuf,
    plan: &DevBuf,
    plan_stride: usize,
    slots: usize,
) -> Result<(), CudaError> {
    let shape = rec.shape;
    assert!(rec.rows >= batch.rows);
    assert!((3..=crate::tree::PLAN_STRIDE).contains(&plan_stride), "plan stride {plan_stride}");
    need(o, batch.rows * shape.v_dim() * 2, "o");
    need(s_pool, batch.n_slots * shape.s_slot_elems() * 4, "S pool");
    need(plan, batch.n_seq * plan_stride * 4, "plan");
    let tier = crate::tree::Plan::tier(slots);
    let null_p = core::ptr::null::<core::ffi::c_void>();
    let (ok, ov, og, ob, keep, rows, stride) = match owed {
        Some(w) => {
            assert_eq!(w.rec.shape, shape, "gdn::window_tree: the owed record is of another shape");
            need(w.keep, batch.n_seq * 4, "owed keep");
            need(w.path.rows, batch.n_seq * w.path.stride * 4, "owed path rows");
            (
                w.rec.k.ptr() as *const core::ffi::c_void,
                w.rec.v.ptr() as *const core::ffi::c_void,
                w.rec.g_decay.ptr() as *const f32,
                w.rec.beta.ptr() as *const f32,
                w.keep.ptr() as *const i32,
                w.path.rows.ptr() as *const i32,
                w.path.stride as i32,
            )
        }
        None => (null_p, null_p, core::ptr::null(), core::ptr::null(), core::ptr::null(), core::ptr::null(), 0),
    };
    let rc = unsafe {
        launcher!(shape, gb_gdn_window_tree, gb_gdn_window_tree_hv32, gb_gdn_window_tree_hv64)(
            rec.q.ptr(),
            rec.k.ptr(),
            rec.v.ptr(),
            rec.g_decay.ptr() as *const f32,
            rec.beta.ptr() as *const f32,
            s_pool.ptr() as *mut f32,
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            batch.n_seq as i32,
            o.ptr(),
            ok,
            ov,
            og,
            ob,
            core::ptr::null(),
            keep,
            rows,
            stride,
            plan.ptr() as *const i32,
            plan_stride as i32,
            tier as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// The S half of a commit over an accepted PATH: replay `keep[i]` rows of sequence i, row `j` of them `path.rows[i * stride + j]`.
pub fn commit_state_path(
    stream: &Stream,
    rec: &Record,
    s_pool: &mut DevBuf,
    batch: &Batch<'_>,
    keep: &DevBuf,
    path: PathRows<'_>,
) -> Result<(), CudaError> {
    let shape = rec.shape;
    assert!(rec.rows >= batch.rows);
    need(keep, batch.n_seq * 4, "keep");
    need(path.rows, batch.n_seq * path.stride * 4, "path rows");
    need(s_pool, batch.n_slots * shape.s_slot_elems() * 4, "S pool");
    let rc = unsafe {
        launcher!(shape, gb_gdn_commit_state_rows, gb_gdn_commit_state_rows_hv32, gb_gdn_commit_state_rows_hv64)(
            rec.k.ptr(),
            rec.v.ptr(),
            rec.g_decay.ptr() as *const f32,
            rec.beta.ptr() as *const f32,
            s_pool.ptr() as *mut f32,
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            keep.ptr() as *const i32,
            path.rows.ptr() as *const i32,
            path.stride as i32,
            batch.n_seq as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// The conv half of a commit over an accepted PATH: new history = last 3 rows of `concat(old 3, the path rows' qkv)`. The default
/// (48 value heads) shape; [`commit_conv_path_shaped`] takes the shape.
pub fn commit_conv_path(
    stream: &Stream,
    qkv: &DevBuf,
    conv_pool: &mut DevBuf,
    batch: &Batch<'_>,
    keep: &DevBuf,
    path: PathRows<'_>,
) -> Result<(), CudaError> {
    commit_conv_path_shaped(GdnShape::Hv48, stream, qkv, conv_pool, batch, keep, path)
}

/// [`commit_conv_path`] for `shape`.
pub fn commit_conv_path_shaped(
    shape: GdnShape,
    stream: &Stream,
    qkv: &DevBuf,
    conv_pool: &mut DevBuf,
    batch: &Batch<'_>,
    keep: &DevBuf,
    path: PathRows<'_>,
) -> Result<(), CudaError> {
    need(qkv, batch.rows * shape.conv_dim() * 2, "qkv");
    need(keep, batch.n_seq * 4, "keep");
    need(path.rows, batch.n_seq * path.stride * 4, "path rows");
    need(conv_pool, batch.n_slots * shape.conv_slot_elems() * 2, "conv pool");
    let rc = unsafe {
        launcher!(shape, gb_gdn_commit_conv_rows, gb_gdn_commit_conv_rows_hv32, gb_gdn_commit_conv_rows_hv64)(
            qkv.ptr(),
            conv_pool.ptr(),
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            keep.ptr() as *const i32,
            path.rows.ptr() as *const i32,
            path.stride as i32,
            batch.n_seq as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Bytes of caller workspace [`prefill`] needs for `rows` total rows (the default shape; [`prefill_workspace_bytes_shaped`]).
pub fn prefill_workspace_bytes(rows: usize) -> usize {
    prefill_workspace_bytes_shaped(GdnShape::Hv48, rows)
}

/// Bytes of caller workspace [`prefill_shaped`] needs for `rows` total rows of `shape`.
pub fn prefill_workspace_bytes_shaped(shape: GdnShape, rows: usize) -> usize {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        match shape {
            GdnShape::Hv48 => ffi::gb_gdn_prefill_workspace_bytes(rows as i32),
            GdnShape::Hv32 => ffi::gb_gdn_prefill_workspace_bytes_hv32(rows as i32),
            GdnShape::Hv64 => ffi::gb_gdn_prefill_workspace_bytes_hv64(rows as i32),
        }
    }
    #[cfg(gb_no_cuda)]
    {
        let a = |x: usize| (x + 255) & !255;
        a(rows * shape.q_dim() * 2) * 2 + a(rows * shape.v_dim() * 2) + a(rows * shape.value_heads() * 4) * 2
    }
}

/// [`prefill`] for rows `row0..row0 + rows` of one sequence's chunk: `x` and `o` hold the whole chunk and are addressed from `row0`, the
/// sequence is `[0, rows)` of those (`seq_start` holds `[0, rows]` as two i32 at byte offset `seq_start_off`), its slot is `slot` (i32 `[1]`).
/// Calling it for consecutive parts of a chunk, in order, gives exactly the bits of one [`prefill`] over the whole chunk: the recurrence is a
/// serial per-row scan whose state goes through the pools between calls, and the conv history a part leaves is the last three rows it
/// received (the next part's first rows read it from the pool). The pools hold the state after `row0 + rows` rows of the chunk when it returns,
/// which is what a prefix-cache snapshot copies out between two parts. `rows` is 1..=[`PREFILL_MAX_ROWS`] (a part shorter than the conv
/// history keeps the older rows it needs from the pool). The default (48 value heads) shape; [`prefill_part_shaped`] takes the shape.
#[allow(clippy::too_many_arguments)]
pub fn prefill_part(
    stream: &Stream,
    x: &PreInputs,
    row0: usize,
    rows: usize,
    w: &PreWeights,
    s_pool: &mut DevBuf,
    conv_pool: &mut DevBuf,
    seq_start: &DevBuf,
    seq_start_off: usize,
    slot: &DevBuf,
    n_slots: usize,
    ws: &mut DevBuf,
    o: &mut DevBuf,
) -> Result<(), CudaError> {
    prefill_part_shaped(
        GdnShape::Hv48,
        stream,
        x,
        row0,
        rows,
        w,
        s_pool,
        conv_pool,
        seq_start,
        seq_start_off,
        slot,
        n_slots,
        ws,
        o,
    )
}

/// [`prefill_part`] for `shape`.
#[allow(clippy::too_many_arguments)]
pub fn prefill_part_shaped(
    shape: GdnShape,
    stream: &Stream,
    x: &PreInputs,
    row0: usize,
    rows: usize,
    w: &PreWeights,
    s_pool: &mut DevBuf,
    conv_pool: &mut DevBuf,
    seq_start: &DevBuf,
    seq_start_off: usize,
    slot: &DevBuf,
    n_slots: usize,
    ws: &mut DevBuf,
    o: &mut DevBuf,
) -> Result<(), CudaError> {
    assert!((1..=PREFILL_MAX_ROWS).contains(&rows), "a prefill part takes 1..={PREFILL_MAX_ROWS} rows, got {rows}");
    let (conv, vh) = (shape.conv_dim(), shape.value_heads());
    check_inputs(shape, x, row0 + rows);
    check_pre_weights(shape, w);
    need(o, (row0 + rows) * shape.v_dim() * 2, "o");
    need(ws, prefill_workspace_bytes_shaped(shape, rows), "workspace");
    need(seq_start, seq_start_off + 8, "seq_start");
    need(slot, 4, "slot");
    need(s_pool, n_slots * shape.s_slot_elems() * 4, "S pool");
    need(conv_pool, n_slots * shape.conv_slot_elems() * 2, "conv pool");
    let rc = unsafe {
        launcher!(shape, gb_gdn_prefill, gb_gdn_prefill_hv32, gb_gdn_prefill_hv64)(
            x.qkv.ptr_at(row0 * conv * 2),
            x.a.ptr_at(row0 * vh * 2),
            x.b.ptr_at(row0 * vh * 2),
            w.conv_w.ptr(),
            w.a_log.ptr() as *const f32,
            w.dt_bias.ptr() as *const f32,
            s_pool.ptr() as *mut f32,
            conv_pool.ptr(),
            seq_start.ptr_at(seq_start_off) as *const i32,
            slot.ptr() as *const i32,
            1,
            rows as i32,
            ws.ptr(),
            o.ptr_at(row0 * shape.v_dim() * 2),
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Long prompt chunks: conv/silu/l2norm/gating fused into the recurrence launch sequence, S and the conv pool are
/// advanced in place, output `o` bf16 `[rows, value heads, 128]`. `ws` is at least `prefill_workspace_bytes(batch.rows)`.
/// Deterministic and chunk-invariant (any split of a prompt into consecutive calls gives identical bits); its bits
/// differ from the window/commit path. The default (48 value heads) shape; [`prefill_shaped`] takes the shape.
#[allow(clippy::too_many_arguments)]
pub fn prefill(
    stream: &Stream,
    x: &PreInputs,
    w: &PreWeights,
    s_pool: &mut DevBuf,
    conv_pool: &mut DevBuf,
    batch: &Batch<'_>,
    ws: &mut DevBuf,
    o: &mut DevBuf,
) -> Result<(), CudaError> {
    prefill_shaped(GdnShape::Hv48, stream, x, w, s_pool, conv_pool, batch, ws, o)
}

/// [`prefill`] for `shape`.
#[allow(clippy::too_many_arguments)]
pub fn prefill_shaped(
    shape: GdnShape,
    stream: &Stream,
    x: &PreInputs,
    w: &PreWeights,
    s_pool: &mut DevBuf,
    conv_pool: &mut DevBuf,
    batch: &Batch<'_>,
    ws: &mut DevBuf,
    o: &mut DevBuf,
) -> Result<(), CudaError> {
    check_inputs(shape, x, batch.rows);
    check_pre_weights(shape, w);
    need(o, batch.rows * shape.v_dim() * 2, "o");
    need(ws, prefill_workspace_bytes_shaped(shape, batch.rows), "workspace");
    check_pools(shape, batch, s_pool, conv_pool);
    let rc = unsafe {
        launcher!(shape, gb_gdn_prefill, gb_gdn_prefill_hv32, gb_gdn_prefill_hv64)(
            x.qkv.ptr(),
            x.a.ptr(),
            x.b.ptr(),
            w.conv_w.ptr(),
            w.a_log.ptr() as *const f32,
            w.dt_bias.ptr() as *const f32,
            s_pool.ptr() as *mut f32,
            conv_pool.ptr(),
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            batch.n_seq as i32,
            batch.rows as i32,
            ws.ptr(),
            o.ptr(),
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_are_what_the_models_declare() {
        assert_eq!(GdnShape::default(), GdnShape::Hv48);
        // Qwen3.8-27B: 16 key x 48 value heads of 128; the 35B-A3B backbone: 16 x 32
        assert_eq!(GdnShape::from_counts(16, 48, 128, 128), Some(GdnShape::Hv48));
        assert_eq!(GdnShape::from_counts(16, 32, 128, 128), Some(GdnShape::Hv32));
        for bad in [(16, 40, 128, 128), (8, 32, 128, 128), (16, 32, 64, 64), (16, 32, 128, 64), (16, 0, 128, 128)] {
            assert_eq!(GdnShape::from_counts(bad.0, bad.1, bad.2, bad.3), None, "{bad:?}");
        }
        // the conv channels are [q 16x128 | k 16x128 | v HVx128]: 10240 and 8192 (conv1d.weight [8192, 1, 4] in the 35B-A3B pack)
        assert_eq!((GdnShape::Hv48.conv_dim(), GdnShape::Hv32.conv_dim()), (10240, 8192));
        assert_eq!((GdnShape::Hv48.values_per_key(), GdnShape::Hv32.values_per_key()), (3, 2));
        // 30 layers x 32 x 128 x 128 x 4 B = 63 MB of S per sequence (the plan's memory model), conv 30 x 3 x 8192 x 2 B
        assert_eq!(30 * GdnShape::Hv32.s_slot_elems() * 4, 62_914_560);
        assert_eq!(30 * GdnShape::Hv32.conv_slot_elems() * 2, 1_474_560);
        assert_eq!(GdnShape::Hv32.v_dim(), 4096); // in_proj_z [4096, 2048], out_proj [2048, 4096]
    }

    #[cfg(not(gb_no_cuda))] // asks the compiled kernels
    #[test]
    fn workspace_formula_matches_the_kernel_side() {
        for shape in GdnShape::ALL {
            for rows in [0usize, 1, 7, 64, 4096] {
                let a = |x: usize| x.div_ceil(256) * 256;
                let want = a(rows * shape.q_dim() * 2) * 2 + a(rows * shape.v_dim() * 2) + a(rows * shape.value_heads() * 4) * 2;
                assert_eq!(prefill_workspace_bytes_shaped(shape, rows), want, "{shape:?} rows {rows}");
            }
        }
        assert_eq!(prefill_workspace_bytes(4096), prefill_workspace_bytes_shaped(GdnShape::Hv48, 4096));
    }
}
