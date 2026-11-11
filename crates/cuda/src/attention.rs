//! Paged KV cache and attention kernels for the 16 full-attention layers (spec section 4, 4.2, 9, 10).
//!
//! Per layer the cache is two bf16 buffers `[n_blocks][256 tokens][kv heads][256 dim]` ([`PagedKv`]); a device
//! block table `i32 [slots][max_blocks]` ([`BlockTable`]) maps `(slot, position / 256)` to a block id. Keys are stored
//! after norm + rope, values raw. Queries and outputs are bf16 `[rows][24][256]` (u16 on the Rust side); q head `h`
//! reads kv head `h / (24 / kv heads)`, the scale is 1/16, softmax is fp32. The q/k norm + rope and the output gate belong to other
//! kernels: `q` arrives normalised and roped, the output is the raw attention result.
//!
//! * [`kv_write`] scatters rows' K/V into their pages (before attention; the host guarantees the blocks exist).
//! * [`attn_decode`] serves decode and verify windows (1..=128 rows, any context): keys in absolute chunks of 512,
//!   per-chunk fp32 online-softmax partials merged in chunk order. A row's output bits depend only on its own query,
//!   its slot's keys and its position: not on the other rows, how many there are, their order, other sequences or
//!   where the blocks sit.
//! * [`attn_prefill`] serves a prefill chunk of one sequence (up to 4096 rows): flash-style, 64-key tiles by absolute
//!   position, chunk invariant. Its bits differ from decode's.
//!
//! The KV head count is a property of the cache ([`KvHeads`]: 4 for Qwen3.8-27B, the default, and 2 for Qwen3.8-Flash-Next), and
//! the layer's attention shape ([`AttnShape`]: query heads x KV heads) is a property of the cache too: 24/4 (the default), 24/2
//! (Flash-Next), 16/2 (the target layers of MiniCPM-V-4.7's backbone) and 8/2 (its DSpark draft). Every function here takes it from
//! the [`PagedKv`] it is given, so a caller never names it twice; the shapes are one kernel source built four times
//! (`csrc/attention.cu`, `csrc/attention_kv2.cu`, `csrc/attention_q16kv2.cu`, `csrc/attention_q8kv2.cu`) and the 24-head kernels are
//! unchanged by the parameters. Queries and outputs are bf16 `[rows][query heads][256]` (the text below says 24, the default).
//!
//! The kernel side lives in `csrc/attention.cu` (ported in part from TensorFold, see THIRD_PARTY_NOTICES.md).

use crate::rt::Stream;
use crate::rt::{CudaError, DevBuf, HostStage};

pub const BLOCK_TOKENS: usize = 256;
pub const N_Q_HEADS: usize = 24;
pub const N_KV_HEADS: usize = 4;
pub const HEAD_DIM: usize = 256;
/// Keys per decode chunk (absolute: chunk `c` covers positions `[512 c, 512 c + 512)`).
pub const CHUNK_KEYS: usize = 512;
pub const MAX_DECODE_ROWS: usize = 128;
pub const MAX_PREFILL_ROWS: usize = 8192;
const _: () = assert!(PagedKv::BLOCK_BYTES == 512 * 1024);

/// bf16 elements of one token's K (or V) row for all kv heads (of the default 4-head cache; [`KvHeads::token_elems`] in general).
pub const KV_TOKEN_ELEMS: usize = N_KV_HEADS * HEAD_DIM;

/// The grouped-query shapes the kernels are built for: 24 query heads of 256 over 4 KV heads (groups of 6) or 2 (groups of 12).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KvHeads {
    #[default]
    Four,
    Two,
}

impl KvHeads {
    /// The shape for a config's `num_key_value_heads`, if the kernels are built for it.
    pub const fn from_count(n: usize) -> Option<Self> {
        match n {
            4 => Some(Self::Four),
            2 => Some(Self::Two),
            _ => None,
        }
    }
    pub const fn count(self) -> usize {
        match self {
            Self::Four => 4,
            Self::Two => 2,
        }
    }
    /// bf16 elements of one token's K (or V) row.
    pub const fn token_elems(self) -> usize {
        self.count() * HEAD_DIM
    }
    /// Bytes of one cache block of K (or of V).
    pub const fn block_bytes(self) -> usize {
        BLOCK_TOKENS * self.token_elems() * 2
    }
}
/// bf16 elements of one query row (all q heads of the default 24-head shape; [`AttnShape::q_row_elems`] in general).
pub const Q_ROW_ELEMS: usize = N_Q_HEADS * HEAD_DIM;

/// The attention shapes the kernels are built for: query heads x KV heads of 256. The kernels are one source built four times
/// (`csrc/attention.cu` and the `_kv2`, `_q16kv2`, `_q8kv2` shims); a shape names which build a cache's calls launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum AttnShape {
    /// 24 query heads over 4 KV heads (groups of 6): Qwen3.8-27B (the default, and every shape this crate had before the parameter).
    #[default]
    Q24Kv4,
    /// 24 over 2 (groups of 12): Qwen3.8-Flash-Next.
    Q24Kv2,
    /// 16 over 2 (groups of 8): the full-attention layers of the Qwen3.5/3.6-35B-A3B text backbone (MiniCPM-V-4.7).
    Q16Kv2,
    /// 8 over 2 (groups of 4): the layers of MiniCPM-V-4.7's DSpark draft.
    Q8Kv2,
    /// 32 over 2 (groups of 16): Qwen3.5-122B-A10B / Ornith-1.5-122B.
    Q32Kv2,
}

impl AttnShape {
    /// Every shape the kernels are built for.
    pub const ALL: [AttnShape; 5] =
        [AttnShape::Q24Kv4, AttnShape::Q24Kv2, AttnShape::Q16Kv2, AttnShape::Q8Kv2, AttnShape::Q32Kv2];

    /// The shape for a config's head counts, if the kernels are built for it (head dim 256 in all).
    pub const fn from_counts(q_heads: usize, kv_heads: usize) -> Option<Self> {
        match (q_heads, kv_heads) {
            (24, 4) => Some(Self::Q24Kv4),
            (24, 2) => Some(Self::Q24Kv2),
            (16, 2) => Some(Self::Q16Kv2),
            (8, 2) => Some(Self::Q8Kv2),
            (32, 2) => Some(Self::Q32Kv2),
            _ => None,
        }
    }
    pub const fn q_heads(self) -> usize {
        match self {
            Self::Q24Kv4 | Self::Q24Kv2 => 24,
            Self::Q16Kv2 => 16,
            Self::Q8Kv2 => 8,
            Self::Q32Kv2 => 32,
        }
    }
    pub const fn kv_heads(self) -> KvHeads {
        match self {
            Self::Q24Kv4 => KvHeads::Four,
            Self::Q24Kv2 | Self::Q16Kv2 | Self::Q8Kv2 | Self::Q32Kv2 => KvHeads::Two,
        }
    }
    /// Query heads that share one KV head.
    pub const fn group(self) -> usize {
        self.q_heads() / self.kv_heads().count()
    }
    /// bf16 elements of one query (and output) row: all q heads.
    pub const fn q_row_elems(self) -> usize {
        self.q_heads() * HEAD_DIM
    }
    /// bf16 elements of one token's K (or V) row for all kv heads.
    pub const fn token_elems(self) -> usize {
        self.kv_heads().token_elems()
    }
    /// Bytes of one cache block of K (or of V).
    pub const fn block_bytes(self) -> usize {
        self.kv_heads().block_bytes()
    }
    /// Whether prefill CTA-shape `variant` (see [`attn_prefill_variant`]) exists at this shape: 0 is the default, 1 runs 1 head a CTA,
    /// 2 six and 3 three, 4 two. A variant whose heads a CTA do not divide the query group is no shape (6 and 3 heads at groups of 8
    /// and 4): the launcher refuses it with an error.
    pub const fn prefill_variant_supported(self, variant: i32) -> bool {
        let heads = match variant {
            0 | 4 => 2,
            1 => 1,
            2 => 6,
            3 => 3,
            _ => return false,
        };
        self.group() % heads == 0
    }
}

impl From<KvHeads> for AttnShape {
    /// The 24-query-head shape of a KV head count (what every cache was before the query heads became a parameter).
    fn from(k: KvHeads) -> Self {
        match k {
            KvHeads::Four => Self::Q24Kv4,
            KvHeads::Two => Self::Q24Kv2,
        }
    }
}

/// One layer's paged K and V caches.
#[derive(Debug)]
pub struct PagedKv {
    pub k: DevBuf,
    pub v: DevBuf,
    pub n_blocks: usize,
    pub block_tokens: usize,
    /// The KV head count the pages are laid out for; every kernel call on this cache uses it.
    pub kv_heads: KvHeads,
    /// The attention shape of the layer the cache serves (`shape.kv_heads() == kv_heads`); every kernel call on this cache launches it.
    pub shape: AttnShape,
}

impl PagedKv {
    /// Bytes of one block of K (or of V) of the default 4-head cache ([`PagedKv::block_bytes`] in general).
    pub const BLOCK_BYTES: usize = BLOCK_TOKENS * KV_TOKEN_ELEMS * 2;

    /// Allocates (uninitialised) 4-head caches of `n_blocks` pages each.
    pub fn new(n_blocks: usize) -> Result<Self, CudaError> {
        Self::with_kv_heads(n_blocks, KvHeads::Four)
    }
    /// Allocates (uninitialised) caches of `n_blocks` pages each for `kv_heads` (and 24 query heads).
    pub fn with_kv_heads(n_blocks: usize, kv_heads: KvHeads) -> Result<Self, CudaError> {
        Self::with_shape(n_blocks, kv_heads.into())
    }
    /// Allocates (uninitialised) caches of `n_blocks` pages each for the layer shape `shape`.
    pub fn with_shape(n_blocks: usize, shape: AttnShape) -> Result<Self, CudaError> {
        let bytes = n_blocks * shape.block_bytes();
        Ok(Self {
            k: DevBuf::alloc(bytes)?,
            v: DevBuf::alloc(bytes)?,
            n_blocks,
            block_tokens: BLOCK_TOKENS,
            kv_heads: shape.kv_heads(),
            shape,
        })
    }
    /// Bytes of one block of K (or of V).
    pub fn block_bytes(&self) -> usize {
        self.kv_heads.block_bytes()
    }
    /// Token capacity.
    pub fn capacity_tokens(&self) -> usize {
        self.n_blocks * self.block_tokens
    }
}

/// Device block table `i32 [slots][max_blocks]`; unused entries are 0.
#[derive(Debug)]
pub struct BlockTable {
    pub buf: DevBuf,
    pub slots: usize,
    pub max_blocks: usize,
}

impl BlockTable {
    /// Builds the table from per-slot block lists (slot `s` = `tables[s]`, position `p` lives in `tables[s][p / 256]`).
    pub fn from_host(tables: &[Vec<u32>], max_blocks: usize) -> Result<Self, CudaError> {
        let mut t = Self { buf: DevBuf::alloc((tables.len().max(1) * max_blocks * 4).max(4))?, slots: tables.len(), max_blocks };
        t.upload(tables)?;
        Ok(t)
    }
    /// Rewrites the whole table (at most `slots` lists of at most `max_blocks` blocks).
    pub fn upload(&mut self, tables: &[Vec<u32>]) -> Result<(), CudaError> {
        let flat = self.flat(tables);
        self.buf.upload(&flat)
    }
    /// [`BlockTable::upload`] in the order of `st`, through `stage` (see [`HostStage`]): a kernel launched on `st` afterwards
    /// reads the new table and one launched before has read the old one. A plain `upload` is a copy on the legacy default
    /// stream, which a non-blocking stream does not wait for.
    pub fn upload_on(&mut self, tables: &[Vec<u32>], stage: &mut HostStage, st: &Stream) -> Result<(), CudaError> {
        let flat = self.flat(tables);
        stage.upload(&mut self.buf, &flat, st)
    }
    /// The device layout of `tables`: `[slots][max_blocks]`, unused entries 0.
    fn flat(&self, tables: &[Vec<u32>]) -> Vec<i32> {
        assert!(tables.len() <= self.slots.max(1), "more block lists than slots");
        let mut flat = vec![0i32; self.slots.max(1) * self.max_blocks];
        for (s, blocks) in tables.iter().enumerate() {
            assert!(blocks.len() <= self.max_blocks, "slot {s}: {} blocks > max_blocks {}", blocks.len(), self.max_blocks);
            for (i, &b) in blocks.iter().enumerate() {
                flat[s * self.max_blocks + i] = b as i32;
            }
        }
        flat
    }
}

/// Per-row `(slot, absolute position)` arrays on the device.
#[derive(Debug)]
pub struct RowIndex {
    pub slot: DevBuf,
    pub pos: DevBuf,
    pub rows: usize,
}

impl RowIndex {
    pub fn from_host(slot: &[i32], pos: &[i32]) -> Result<Self, CudaError> {
        assert_eq!(slot.len(), pos.len());
        Ok(Self { slot: DevBuf::from_slice(slot)?, pos: DevBuf::from_slice(pos)?, rows: slot.len() })
    }
    pub fn upload(&mut self, slot: &[i32], pos: &[i32]) -> Result<(), CudaError> {
        assert!(slot.len() == pos.len() && slot.len() <= self.rows);
        self.slot.upload(slot)?;
        self.pos.upload(pos)
    }
}

/// Bytes of workspace [`attn_decode`] needs for up to `rows` rows and positions below `max_context` at the default 24 query
/// heads: fp32 partials `o [rows][24][chunks][256]`, `m` and `l [rows][24][chunks]`, `chunks = ceil(max_context / 512)`.
/// (About 1.6 GiB at 128 rows x 262144; 97 MiB at 8 rows.) [`decode_workspace_bytes_for`] is the same for a shape.
pub fn decode_workspace_bytes(rows: usize, max_context: usize) -> usize {
    decode_workspace_bytes_for(AttnShape::Q24Kv4, rows, max_context)
}

/// [`decode_workspace_bytes`] for `shape`: the partials are per query head, so it scales with the shape's head count (16 heads:
/// two thirds of the 24-head figure; 8 heads: a third).
pub fn decode_workspace_bytes_for(shape: AttnShape, rows: usize, max_context: usize) -> usize {
    let nch = max_context.div_ceil(CHUNK_KEYS).max(1);
    let pairs = rows * shape.q_heads() * nch;
    let a = |x: usize| x.div_ceil(256) * 256;
    a(pairs * HEAD_DIM * 4) + 2 * a(pairs * 4)
}

/// Bytes of workspace [`attn_prefill`] needs (none).
pub fn prefill_workspace_bytes(_rows: usize) -> usize {
    0
}

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_kv_write(
            k: *const c_void,
            v: *const c_void,
            row_slot: *const i32,
            row_pos: *const i32,
            block_table: *const i32,
            max_blocks: i32,
            k_cache: *mut c_void,
            v_cache: *mut c_void,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_kv_write_rows(
            k: *const c_void,
            v: *const c_void,
            src_row: *const i32,
            row_slot: *const i32,
            row_pos: *const i32,
            block_table: *const i32,
            max_blocks: i32,
            k_cache: *mut c_void,
            v_cache: *mut c_void,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_attn_decode_tree(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            k_nodes: *const c_void,
            v_nodes: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            row_base: *const i32,
            paths: *const i32,
            path_stride: i32,
            out: *mut c_void,
            rows: i32,
            max_context: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        #[allow(dead_code)] // checked against the Rust formula in the tests
        pub fn gb_attn_decode_workspace_bytes(rows: i32, max_context: i32) -> usize;
        pub fn gb_attn_decode(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            out: *mut c_void,
            rows: i32,
            max_context: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        #[allow(dead_code)]
        pub fn gb_attn_prefill_workspace_bytes(rows: i32) -> usize;
        pub fn gb_attn_prefill_variant(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            slot: i32,
            start: i32,
            rows: i32,
            out: *mut c_void,
            variant: i32,
            stream: *mut c_void,
        ) -> i32;
        // the same launchers built for 2 kv heads (csrc/attention_kv2.cu)
        pub fn gb_kv_write_kv2(
            k: *const c_void,
            v: *const c_void,
            row_slot: *const i32,
            row_pos: *const i32,
            block_table: *const i32,
            max_blocks: i32,
            k_cache: *mut c_void,
            v_cache: *mut c_void,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_kv_write_rows_kv2(
            k: *const c_void,
            v: *const c_void,
            src_row: *const i32,
            row_slot: *const i32,
            row_pos: *const i32,
            block_table: *const i32,
            max_blocks: i32,
            k_cache: *mut c_void,
            v_cache: *mut c_void,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_attn_decode_tree_kv2(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            k_nodes: *const c_void,
            v_nodes: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            row_base: *const i32,
            paths: *const i32,
            path_stride: i32,
            out: *mut c_void,
            rows: i32,
            max_context: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        #[allow(dead_code)]
        pub fn gb_attn_decode_workspace_bytes_kv2(rows: i32, max_context: i32) -> usize;
        pub fn gb_attn_decode_kv2(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            out: *mut c_void,
            rows: i32,
            max_context: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        #[allow(dead_code)]
        pub fn gb_attn_prefill_workspace_bytes_kv2(rows: i32) -> usize;
        pub fn gb_attn_prefill_variant_kv2(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            slot: i32,
            start: i32,
            rows: i32,
            out: *mut c_void,
            variant: i32,
            stream: *mut c_void,
        ) -> i32;
        // the same launchers built for 16 query heads over 2 kv heads (csrc/attention_q16kv2.cu)
        pub fn gb_kv_write_q16kv2(
            k: *const c_void,
            v: *const c_void,
            row_slot: *const i32,
            row_pos: *const i32,
            block_table: *const i32,
            max_blocks: i32,
            k_cache: *mut c_void,
            v_cache: *mut c_void,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_kv_write_rows_q16kv2(
            k: *const c_void,
            v: *const c_void,
            src_row: *const i32,
            row_slot: *const i32,
            row_pos: *const i32,
            block_table: *const i32,
            max_blocks: i32,
            k_cache: *mut c_void,
            v_cache: *mut c_void,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_attn_decode_tree_q16kv2(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            k_nodes: *const c_void,
            v_nodes: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            row_base: *const i32,
            paths: *const i32,
            path_stride: i32,
            out: *mut c_void,
            rows: i32,
            max_context: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        #[allow(dead_code)]
        pub fn gb_attn_decode_workspace_bytes_q16kv2(rows: i32, max_context: i32) -> usize;
        pub fn gb_attn_decode_q16kv2(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            out: *mut c_void,
            rows: i32,
            max_context: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        #[allow(dead_code)]
        pub fn gb_attn_prefill_workspace_bytes_q16kv2(rows: i32) -> usize;
        pub fn gb_attn_prefill_variant_q16kv2(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            slot: i32,
            start: i32,
            rows: i32,
            out: *mut c_void,
            variant: i32,
            stream: *mut c_void,
        ) -> i32;
        // the same launchers built for 8 query heads over 2 kv heads (csrc/attention_q8kv2.cu)
        pub fn gb_kv_write_q8kv2(
            k: *const c_void,
            v: *const c_void,
            row_slot: *const i32,
            row_pos: *const i32,
            block_table: *const i32,
            max_blocks: i32,
            k_cache: *mut c_void,
            v_cache: *mut c_void,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_kv_write_rows_q8kv2(
            k: *const c_void,
            v: *const c_void,
            src_row: *const i32,
            row_slot: *const i32,
            row_pos: *const i32,
            block_table: *const i32,
            max_blocks: i32,
            k_cache: *mut c_void,
            v_cache: *mut c_void,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_attn_decode_tree_q8kv2(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            k_nodes: *const c_void,
            v_nodes: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            row_base: *const i32,
            paths: *const i32,
            path_stride: i32,
            out: *mut c_void,
            rows: i32,
            max_context: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        #[allow(dead_code)]
        pub fn gb_attn_decode_workspace_bytes_q8kv2(rows: i32, max_context: i32) -> usize;
        pub fn gb_attn_decode_q8kv2(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            out: *mut c_void,
            rows: i32,
            max_context: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        #[allow(dead_code)]
        pub fn gb_attn_prefill_workspace_bytes_q8kv2(rows: i32) -> usize;
        pub fn gb_attn_prefill_variant_q8kv2(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            slot: i32,
            start: i32,
            rows: i32,
            out: *mut c_void,
            variant: i32,
            stream: *mut c_void,
        ) -> i32;
        // the same launchers built for 32 query heads over 2 kv heads (csrc/attention_q32kv2.cu)
        pub fn gb_kv_write_q32kv2(
            k: *const c_void,
            v: *const c_void,
            row_slot: *const i32,
            row_pos: *const i32,
            block_table: *const i32,
            max_blocks: i32,
            k_cache: *mut c_void,
            v_cache: *mut c_void,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_kv_write_rows_q32kv2(
            k: *const c_void,
            v: *const c_void,
            src_row: *const i32,
            row_slot: *const i32,
            row_pos: *const i32,
            block_table: *const i32,
            max_blocks: i32,
            k_cache: *mut c_void,
            v_cache: *mut c_void,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_attn_decode_tree_q32kv2(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            k_nodes: *const c_void,
            v_nodes: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            row_base: *const i32,
            paths: *const i32,
            path_stride: i32,
            out: *mut c_void,
            rows: i32,
            max_context: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        #[allow(dead_code)]
        pub fn gb_attn_decode_workspace_bytes_q32kv2(rows: i32, max_context: i32) -> usize;
        pub fn gb_attn_decode_q32kv2(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            out: *mut c_void,
            rows: i32,
            max_context: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        #[allow(dead_code)]
        pub fn gb_attn_prefill_workspace_bytes_q32kv2(rows: i32) -> usize;
        pub fn gb_attn_prefill_variant_q32kv2(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            slot: i32,
            start: i32,
            rows: i32,
            out: *mut c_void,
            variant: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// Picks a launcher by shape: the default (24 x 4) one, then its `_kv2`, `_q16kv2` and `_q8kv2` twins (the same signature).
macro_rules! by_shape {
    ($shape:expr, $a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {
        match $shape {
            AttnShape::Q24Kv4 => ffi::$a,
            AttnShape::Q24Kv2 => ffi::$b,
            AttnShape::Q16Kv2 => ffi::$c,
            AttnShape::Q8Kv2 => ffi::$d,
            AttnShape::Q32Kv2 => ffi::$e,
        }
    };
}

/// Writes row `r`'s K and V (`k`, `v`: bf16 `[rows][kv heads][256]`) into the cache at `(row_slot[r], row_pos[r])`.
/// A negative position skips the row. Rows must target distinct positions.
pub fn kv_write(
    k: &DevBuf,
    v: &DevBuf,
    rows: &RowIndex,
    table: &BlockTable,
    cache: &mut PagedKv,
    n_rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert!(n_rows <= rows.rows);
    let te = cache.kv_heads.token_elems();
    assert!(k.bytes() >= n_rows * te * 2 && v.bytes() >= n_rows * te * 2);
    let rc = unsafe {
        (by_shape!(cache.shape, gb_kv_write, gb_kv_write_kv2, gb_kv_write_q16kv2, gb_kv_write_q8kv2, gb_kv_write_q32kv2))(
            k.ptr(),
            v.ptr(),
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            table.buf.ptr() as *const i32,
            table.max_blocks as i32,
            cache.k.ptr(),
            cache.v.ptr(),
            n_rows as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Scatter window rows into the cache by an explicit source row: entry `i` copies row `src_row[i]` of `k` / `v` (bf16
/// `[rows][kv heads][256]`, the tree window's node buffers) to position `pos[i]` of `slot[i]` (all i32 `[n]` on the device); a negative position
/// skips the entry; destinations must be distinct. The commit of an accepted tree path: nothing but accepted rows ever enters the cache.
#[allow(clippy::too_many_arguments)]
pub fn kv_write_rows(
    k: &DevBuf,
    v: &DevBuf,
    src_row: &DevBuf,
    slot: &DevBuf,
    pos: &DevBuf,
    table: &BlockTable,
    cache: &mut PagedKv,
    n: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    for b in [src_row, slot, pos] {
        assert!(b.bytes() >= n * 4, "kv_write_rows: an index array holds fewer than {n} entries");
    }
    let rc = unsafe {
        (by_shape!(
            cache.shape,
            gb_kv_write_rows,
            gb_kv_write_rows_kv2,
            gb_kv_write_rows_q16kv2,
            gb_kv_write_rows_q8kv2,
            gb_kv_write_rows_q32kv2
        ))(
            k.ptr(),
            v.ptr(),
            src_row.ptr() as *const i32,
            slot.ptr() as *const i32,
            pos.ptr() as *const i32,
            table.buf.ptr() as *const i32,
            table.max_blocks as i32,
            cache.k.ptr(),
            cache.v.ptr(),
            n as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// What the tree attention reads per row besides `(slot, position)`: the sequence's committed length and the row's root-to-row path.
pub struct TreeRows<'a> {
    /// i32 `[rows]`: the committed length `L` of the row's sequence (its root's position); keys below it come from the cache.
    pub base: &'a DevBuf,
    /// i32 `[rows][stride]`: the window rows from the root to this row (global row indices into the node buffers), then -1. Key
    /// `L + j` of the row is the K/V of `paths[row][j]`.
    pub paths: &'a DevBuf,
    pub stride: usize,
}

/// Decode / verify attention over TREE windows, `n_rows` (1..=128) rows: `q`, `out` bf16 `[n_rows][24][256]`. Row `r` attends to the
/// committed keys `0..base[r]` of its slot (the cache, never written here) and then to its ancestors' window keys at the logical
/// positions `base[r]..=pos[r]`; `rows.pos` is the logical position (`base + depth`). The window's K and V live in `k_nodes` /
/// `v_nodes` (bf16 `[n_rows][kv heads][256]`, row-indexed), so rows that share a logical position cannot collide. A row's bits are those
/// [`attn_decode`] gives the same row at the end of a chain holding its path: the arithmetic is the same tile for tile.
#[allow(clippy::too_many_arguments)]
pub fn attn_decode_tree(
    q: &DevBuf,
    cache: &PagedKv,
    table: &BlockTable,
    rows: &RowIndex,
    tree: &TreeRows<'_>,
    k_nodes: &DevBuf,
    v_nodes: &DevBuf,
    n_rows: usize,
    out: &mut DevBuf,
    max_context: usize,
    workspace: &mut DevBuf,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert!((1..=MAX_DECODE_ROWS).contains(&n_rows) && n_rows <= rows.rows);
    let qe = cache.shape.q_row_elems();
    assert!(q.bytes() >= n_rows * qe * 2 && out.bytes() >= n_rows * qe * 2);
    let te = cache.kv_heads.token_elems();
    assert!(k_nodes.bytes() >= n_rows * te * 2 && v_nodes.bytes() >= n_rows * te * 2, "node K/V buffers");
    assert!(tree.base.bytes() >= n_rows * 4 && tree.paths.bytes() >= n_rows * tree.stride * 4, "tree index arrays");
    assert!(max_context >= 1 && max_context <= table.max_blocks * BLOCK_TOKENS, "max_context exceeds the block table");
    let need = decode_workspace_bytes_for(cache.shape, n_rows, max_context);
    assert!(workspace.bytes() >= need, "decode workspace: {} bytes < {need}", workspace.bytes());
    let rc = unsafe {
        (by_shape!(
            cache.shape,
            gb_attn_decode_tree,
            gb_attn_decode_tree_kv2,
            gb_attn_decode_tree_q16kv2,
            gb_attn_decode_tree_q8kv2,
            gb_attn_decode_tree_q32kv2
        ))(
            q.ptr(),
            cache.k.ptr(),
            cache.v.ptr(),
            k_nodes.ptr(),
            v_nodes.ptr(),
            table.buf.ptr() as *const i32,
            table.max_blocks as i32,
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            tree.base.ptr() as *const i32,
            tree.paths.ptr() as *const i32,
            tree.stride as i32,
            out.ptr(),
            n_rows as i32,
            max_context as i32,
            workspace.ptr(),
            workspace.bytes(),
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Decode / verify attention for `n_rows` (1..=128) rows: `q`, `out` bf16 `[n_rows][24][256]`. Row `r` attends to
/// keys `0..=row_pos[r]` of slot `row_slot[r]` (already written by [`kv_write`]); every position must be below
/// `max_context`, which sizes the grid and the workspace ([`decode_workspace_bytes`]).
#[allow(clippy::too_many_arguments)]
pub fn attn_decode(
    q: &DevBuf,
    cache: &PagedKv,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    out: &mut DevBuf,
    max_context: usize,
    workspace: &mut DevBuf,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert!((1..=MAX_DECODE_ROWS).contains(&n_rows) && n_rows <= rows.rows);
    let qe = cache.shape.q_row_elems();
    assert!(q.bytes() >= n_rows * qe * 2 && out.bytes() >= n_rows * qe * 2);
    assert!(max_context >= 1 && max_context <= table.max_blocks * BLOCK_TOKENS, "max_context exceeds the block table");
    let need = decode_workspace_bytes_for(cache.shape, n_rows, max_context);
    assert!(workspace.bytes() >= need, "decode workspace: {} bytes < {need}", workspace.bytes());
    let rc = unsafe {
        (by_shape!(
            cache.shape,
            gb_attn_decode,
            gb_attn_decode_kv2,
            gb_attn_decode_q16kv2,
            gb_attn_decode_q8kv2,
            gb_attn_decode_q32kv2
        ))(
            q.ptr(),
            cache.k.ptr(),
            cache.v.ptr(),
            table.buf.ptr() as *const i32,
            table.max_blocks as i32,
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            out.ptr(),
            n_rows as i32,
            max_context as i32,
            workspace.ptr(),
            workspace.bytes(),
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Prefill attention for a chunk of one sequence: `q`, `out` bf16 `[n_rows][24][256]` for positions
/// `start .. start + n_rows` (n_rows <= 4096) of table row `slot`; the chunk's K/V must already be in the cache.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill(
    q: &DevBuf,
    cache: &PagedKv,
    table: &BlockTable,
    slot: usize,
    start: usize,
    n_rows: usize,
    out: &mut DevBuf,
    stream: &Stream,
) -> Result<(), CudaError> {
    attn_prefill_variant(q, cache, table, slot, start, n_rows, out, 0, stream)
}

/// [`attn_prefill`] with an explicit CTA-shape variant (tuning; 0 is the default).
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill_variant(
    q: &DevBuf,
    cache: &PagedKv,
    table: &BlockTable,
    slot: usize,
    start: usize,
    n_rows: usize,
    out: &mut DevBuf,
    variant: i32,
    stream: &Stream,
) -> Result<(), CudaError> {
    attn_prefill_rows(q, cache, table, slot, start, 0, n_rows, out, variant, stream)
}

/// [`attn_prefill_variant`] for the rows `row0 .. row0 + n_rows` of `q` and `out`, which may hold several sequences' rows one after
/// another (a prefill of several short prompts in one forward): the chunk's query rows start `row0` rows into the buffers. With
/// `row0` 0 this is the call [`attn_prefill_variant`] always made, so a single sequence's bits are untouched.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill_rows(
    q: &DevBuf,
    cache: &PagedKv,
    table: &BlockTable,
    slot: usize,
    start: usize,
    row0: usize,
    n_rows: usize,
    out: &mut DevBuf,
    variant: i32,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert!(n_rows <= MAX_PREFILL_ROWS && slot < table.slots);
    let qe = cache.shape.q_row_elems();
    assert!(q.bytes() >= (row0 + n_rows) * qe * 2 && out.bytes() >= (row0 + n_rows) * qe * 2);
    assert!(start + n_rows <= table.max_blocks * BLOCK_TOKENS, "chunk runs past the block table");
    let off = row0 * qe * 2;
    let rc = unsafe {
        (by_shape!(
            cache.shape,
            gb_attn_prefill_variant,
            gb_attn_prefill_variant_kv2,
            gb_attn_prefill_variant_q16kv2,
            gb_attn_prefill_variant_q8kv2,
            gb_attn_prefill_variant_q32kv2
        ))(
            q.ptr_at(off) as *const core::ffi::c_void,
            cache.k.ptr(),
            cache.v.ptr(),
            table.buf.ptr() as *const i32,
            table.max_blocks as i32,
            slot as i32,
            start as i32,
            n_rows as i32,
            out.ptr_at(off),
            variant,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(gb_no_cuda))] // asks the compiled kernels
    #[test]
    fn workspace_formula_matches_the_kernel_side() {
        for (r, c) in [(1, 1), (1, 512), (1, 513), (8, 262144), (128, 5000)] {
            assert_eq!(decode_workspace_bytes(r, c), unsafe { ffi::gb_attn_decode_workspace_bytes(r as i32, c as i32) });
        }
        assert_eq!(prefill_workspace_bytes(4096), unsafe { ffi::gb_attn_prefill_workspace_bytes(4096) });
        // the 2-head build sizes its workspace the same way (it depends on the query heads only)
        for (r, c) in [(1, 1), (8, 262144), (128, 5000)] {
            assert_eq!(decode_workspace_bytes(r, c), unsafe { ffi::gb_attn_decode_workspace_bytes_kv2(r as i32, c as i32) });
        }
        assert_eq!(prefill_workspace_bytes(4096), unsafe { ffi::gb_attn_prefill_workspace_bytes_kv2(4096) });
        // the 16- and 8-head builds size it by their own head counts
        for (r, c) in [(1, 1), (1, 513), (8, 262144), (128, 5000)] {
            assert_eq!(decode_workspace_bytes_for(AttnShape::Q16Kv2, r, c), unsafe {
                ffi::gb_attn_decode_workspace_bytes_q16kv2(r as i32, c as i32)
            });
            assert_eq!(decode_workspace_bytes_for(AttnShape::Q8Kv2, r, c), unsafe {
                ffi::gb_attn_decode_workspace_bytes_q8kv2(r as i32, c as i32)
            });
        }
        assert_eq!(prefill_workspace_bytes(4096), unsafe { ffi::gb_attn_prefill_workspace_bytes_q16kv2(4096) });
        assert_eq!(prefill_workspace_bytes(4096), unsafe { ffi::gb_attn_prefill_workspace_bytes_q8kv2(4096) });
    }

    #[test]
    fn attention_shapes() {
        assert_eq!(AttnShape::default(), AttnShape::Q24Kv4);
        // the model configs: Qwen3.8-27B 24/4, Flash-Next 24/2, the 35B-A3B backbone 16/2, its DSpark draft 8/2
        for (q, kv, s) in
            [(24, 4, AttnShape::Q24Kv4), (24, 2, AttnShape::Q24Kv2), (16, 2, AttnShape::Q16Kv2), (8, 2, AttnShape::Q8Kv2)]
        {
            assert_eq!(AttnShape::from_counts(q, kv), Some(s));
            assert_eq!((s.q_heads(), s.kv_heads().count()), (q, kv));
            assert_eq!(s.group(), q / kv);
            assert_eq!(s.q_row_elems(), q * 256);
            assert_eq!(AttnShape::from(s.kv_heads()).kv_heads(), s.kv_heads());
        }
        for (q, kv) in [(32, 8), (16, 4), (8, 4), (12, 2), (0, 2)] {
            assert_eq!(AttnShape::from_counts(q, kv), None, "{q}/{kv}");
        }
        assert_eq!(AttnShape::from(KvHeads::Four), AttnShape::Q24Kv4);
        assert_eq!(AttnShape::from(KvHeads::Two), AttnShape::Q24Kv2);
        // prefill CTA shapes: 6 and 3 heads a CTA divide groups of 6 and 12 but not 8 or 4
        for v in [0, 1, 4] {
            assert!(AttnShape::ALL.iter().all(|s| s.prefill_variant_supported(v)), "variant {v}");
        }
        assert!(AttnShape::Q24Kv4.prefill_variant_supported(2) && AttnShape::Q24Kv4.prefill_variant_supported(3));
        assert!(AttnShape::Q24Kv2.prefill_variant_supported(2) && AttnShape::Q24Kv2.prefill_variant_supported(3));
        for s in [AttnShape::Q16Kv2, AttnShape::Q8Kv2] {
            assert!(!s.prefill_variant_supported(2) && !s.prefill_variant_supported(3), "{s:?}");
        }
        assert!(!AttnShape::Q24Kv4.prefill_variant_supported(5) && !AttnShape::Q24Kv4.prefill_variant_supported(-1));
        // workspace scales with the query heads
        assert_eq!(decode_workspace_bytes_for(AttnShape::Q24Kv4, 8, 262144), decode_workspace_bytes(8, 262144));
        assert_eq!(decode_workspace_bytes_for(AttnShape::Q16Kv2, 8, 262144) * 3, decode_workspace_bytes(8, 262144) * 2);
        // the plan's KV figure: 10 layers x 2 kv x 256 x (K + V) x 2 B = 20 KiB a token; the draft's 5 layers 10 KiB
        assert_eq!(10 * AttnShape::Q16Kv2.token_elems() * 2 * 2, 20 * 1024);
        assert_eq!(5 * AttnShape::Q8Kv2.token_elems() * 2 * 2, 10 * 1024);
    }

    #[test]
    fn kv_head_shapes() {
        assert_eq!(KvHeads::default(), KvHeads::Four);
        assert_eq!(
            (KvHeads::from_count(4), KvHeads::from_count(2), KvHeads::from_count(8)),
            (Some(KvHeads::Four), Some(KvHeads::Two), None)
        );
        assert_eq!(KvHeads::Four.block_bytes(), PagedKv::BLOCK_BYTES);
        assert_eq!(KvHeads::Two.block_bytes(), PagedKv::BLOCK_BYTES / 2);
        assert_eq!(KvHeads::Two.token_elems(), 512);
    }

    #[test]
    fn workspace_sizes() {
        assert_eq!(decode_workspace_bytes(8, 262144) >> 20, 96); // MiB
    }
}
