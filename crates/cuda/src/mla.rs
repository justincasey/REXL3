//! GLM-5.3-Flash multi-head latent attention (NoPE, 64 heads, 512-wide latent) for the 11 MLA layers (3, 7, ..., 43) and the
//! MTP layer 45: latent cache append, q absorb, attention over the latent for 1 row and K+1-row verify windows, expand. Two
//! attention kernels share everything else: the **dense** one (a row attends to all of its keys, up to [`DENSE_MAX_KEYS`] =
//! `index_topk` = 2,048 of them in the engine; it works at any context and is the reference at 262,144) and the **sparse** one
//! (a row attends to the keys its DSA selection names). The regime is decided **per row** by [`row_is_dense`], never per window or
//! batch (the ExLlamaV3 oracle decides batch-wide and window-inclusive, which makes a row's path depend on its neighbours).
//! The indexer itself (index keys, pool plane, scoring, top-k) is `dsa.rs`; this module consumes its `Selection`.
//!
//! # The per-token state, and the cache layout contract
//!
//! One MLA layer's cache is a [`LatentCache`]: a single buffer `T [n_blocks][256 tokens][512]` holding the **normalised**
//! 512-wide latent `c_kv` of every token (T = fp16 or bf16, one T per call chain). There is no rope key (`qk_rope_head_dim`
//! is 0) and the same vector is both K and V of all 64 heads. That is 1 KiB per token per layer, 11 KiB per token over the
//! 11 MLA layers (the MTP layer 45 keeps its own cache the same way). With `--kv-dtype fp8` ([`LatentKv::Fp8`]) a page holds the
//! 512 E4M3 values and one fp32 scale of each token instead (516 bytes a token and layer), quantised once by [`latent_write`] and
//! dequantised exactly into the same T tiles by every attention kernel ([`kv8`]).
//!
//! A device block table `i32 [slots][max_blocks]` ([`BlockTable`], shared with the GQA cache) maps `(slot, position / 256)`
//! to a block id, and `RowIndex` gives each row of a call its `(slot, absolute position)`. **A layer's other planes share
//! the block-id space**: the DSA indexer's planes are separate buffers addressed by the same `(block id, token in block)`:
//!
//! | plane | shape | element |
//! | --- | --- | --- |
//! | latent (this module) | `[n_blocks][256][512]` | T |
//! | indexer `[k ‖ gate]` | `[n_blocks][256][256]` | fp16 |
//! | indexer k-pool | `[n_blocks][64][128]` (one entry per 4 tokens, `index_kpool = 4`) | fp16 |
//!
//! [`plane_offset`] is the one addressing formula (`(block * entries_per_block + entry) * width`, in elements); the latent
//! write kernel and the tests use it, and a new plane only needs its own `width` and `entries_per_block`.
//! A page is 256 tokens, so a pool never straddles a page (`256 % 4 == 0`).
//!
//! # The verify-window contract (mirrors the GQA path)
//!
//! A window of rows of one sequence is written in place at its positions by [`latent_write`] (before attention; the host
//! guarantees the blocks exist), then [`attend`] masks per row by that row's own position, so row `r` sees its window
//! predecessors and never a later window row. Positions at and past the committed length hold stale data until the host
//! commits: nothing here stores a length, a rejected row is simply overwritten by the next window before anyone reads it
//! (`verify_window_equals_serial` proves the stale-data case against a NaN-filled cache). A padding row (position -1, what a
//! CUDA-graph bucket uses) is inert: `latent_write` skips it and `attend` writes zeros for it.
//!
//! # Row invariance (the contract this file exists to keep)
//!
//! Row `r`'s output bits depend only on its own query row, its slot's keys `0..=pos[r]` and its own position, for any row
//! count 1..=128, any position inside a window, any block placement, any `max_context` and any other rows in the call.
//! See the header of `csrc/mla.cu` for how each kernel guarantees it. The tiling (32-key tiles, 256-key chunks = one page,
//! four d-quarter warps) is a compile-time constant, never derived from rows, context or SM count.
//!
//! # Calling sequence for one MLA layer (what the model code does)
//!
//! 1. `q_b_proj` -> `q` `[rows][64*256]`, `kv_a_proj_with_mqa` + `kv_a_layernorm` -> `ckv` `[rows][512]` (the EXL3 linears and norms).
//! 2. [`latent_write`]`(ckv, rows, table, cache)`.
//! 3. [`mla_rows_auto`] (= [`absorb`], [`attend_auto`], [`expand`]) -> `out` `[rows][64*256]`, the `o_proj` input. `attend_auto` runs
//!    [`attend_below`] (dense rows) and [`attend_sparse`] (rows the selection marks sparse) into the same `o_lat`; each skips the
//!    other's rows, so a verify window straddling 2,048 tokens is served row by row, each exactly as if decoded alone. While every
//!    row of a call is below 2,048 tokens ([`MlaScratch`] sized for `max_context <= DENSE_MAX_KEYS`) the sparse launch is skipped
//!    and the selection is not read, so a model without the indexer calls [`mla_rows`] (dense only) the same way.

use crate::attention::{BlockTable, RowIndex};
use crate::exl3_linear::Dtype;
use crate::rt::{CudaError, DevBuf, Stream};

/// Latent width (`kv_lora_rank`).
pub const LATENT_DIM: usize = 512;
/// Query heads.
pub const N_HEADS: usize = 64;
/// `qk_nope_head_dim`, which equals `v_head_dim`.
pub const NOPE_DIM: usize = 256;
/// Tokens per cache block (the page size of every paged cache in the engine).
pub const BLOCK_TOKENS: usize = 256;
/// Keys per attention chunk (absolute: chunk `c` covers positions `[256 c, 256 c + 256)`).
pub const CHUNK_KEYS: usize = 256;
/// Rows per call.
pub const MAX_ROWS: usize = 128;
/// `index_topk`: a row that sees at most this many keys attends densely.
pub const DENSE_MAX_KEYS: usize = 2048;
/// `kv_b_proj` rows per head: `qk_nope_head_dim + v_head_dim`.
pub const KV_B_ROWS_PER_HEAD: usize = 512;

/// Dense or sparse for a row at absolute position `pos` (it sees `pos + 1` keys). This is per row, never per batch or per
/// window: the ExLlamaV3 oracle decides batch-wide and window-inclusive, which makes a row's path depend on its neighbours.
/// A padding row (negative position) is dense (it does nothing).
pub fn row_is_dense(pos: i32) -> bool {
    (pos as i64) < DENSE_MAX_KEYS as i64
}

/// Element offset of `(block, entry)` in a plane of `entries_per_block` entries of `width` elements each: the addressing
/// every plane of a layer shares (latent: 256 entries of 512; indexer k and gate: 256 of 256; k-pool: 64 of 128).
pub fn plane_offset(block: usize, entry: usize, entries_per_block: usize, width: usize) -> usize {
    (block * entries_per_block + entry) * width
}

fn check_dtype(d: Dtype) -> i32 {
    assert!(matches!(d, Dtype::F16 | Dtype::BF16), "MLA kernels take fp16 or bf16 activations, not {d:?}");
    d as i32
}

/// The latent cache's format (`--kv-dtype`, [`KV_ENV`]): the call chain's T (fp16 for GLM, the default: every golden was made with
/// it), or the fp8 page of `csrc/mla_kv8.cuh` ([`kv8`]): per token 512 E4M3 values and one fp32 scale `amax / 448`, 516 bytes against
/// 1,024. The kernels dequantise an fp8 page exactly into the T tiles they stage, so on any cache the fp8 path's output is the T
/// path's on the dequantised values ([`kv8::dequantise`]), bit for bit; every reader (decode, verify, prefill, a restored prefix)
/// reads the same quantised bytes, so drafted == serial, concurrent == solo and prefix-restored == uncached hold under fp8 too.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LatentKv {
    /// `T [n_blocks][256][512]`.
    #[default]
    Plain,
    /// E4M3 `[n_blocks][256][512]` then fp32 scales `[n_blocks][256]`, page by page.
    Fp8,
}

impl LatentKv {
    /// Bytes of one block (256 tokens) of latents of activation type `dtype`.
    pub fn block_bytes(self, dtype: Dtype) -> usize {
        match self {
            LatentKv::Plain => BLOCK_TOKENS * LATENT_DIM * dtype.size(),
            LatentKv::Fp8 => kv8::PAGE_BYTES,
        }
    }
    /// Bytes one token costs in one layer.
    pub fn token_bytes(self, dtype: Dtype) -> usize {
        self.block_bytes(dtype) / BLOCK_TOKENS
    }
    /// The flag's spelling (`fp16`, `fp8`).
    pub const fn name(self) -> &'static str {
        match self {
            LatentKv::Plain => "fp16",
            LatentKv::Fp8 => "fp8",
        }
    }
    /// Parses `fp16` (`f16`, `half`) or `fp8` (`e4m3`, `fp8_e4m3`, `fp8e4m3`). A typo is an error, never a silent fp16.
    pub fn parse(v: &str) -> Result<Self, String> {
        match v.trim().to_ascii_lowercase().as_str() {
            "fp16" | "f16" | "half" => Ok(LatentKv::Plain),
            "fp8" | "e4m3" | "fp8_e4m3" | "fp8e4m3" => Ok(LatentKv::Fp8),
            other => Err(format!("{other:?}: expected fp16 or fp8")),
        }
    }
    /// The kernels' `kv` argument.
    pub(crate) const fn id(self) -> i32 {
        match self {
            LatentKv::Plain => 0,
            LatentKv::Fp8 => 1,
        }
    }
}

/// The environment variable that selects GLM's latent cache format for the process (`fp16`, the default, or `fp8`); the server's
/// `--kv-dtype fp8` sets it. Every TP rank must run with the same value (the ranks compare it at the Hello).
pub const KV_ENV: &str = "REXL3_GLM_KV";

/// The format this process is configured for ([`KV_ENV`]; unset or empty: fp16). An unrecognised value is an error, so a typo never
/// silently serves the other format.
pub fn configured_kv() -> Result<LatentKv, String> {
    match std::env::var(KV_ENV) {
        Err(_) => Ok(LatentKv::Plain),
        Ok(v) if v.trim().is_empty() => Ok(LatentKv::Plain),
        Ok(v) => LatentKv::parse(&v).map_err(|e| format!("{KV_ENV}={v:?}: {e}")),
    }
}

/// The CPU mirror of `csrc/mla_kv8.cuh`: the fp8 page's addressing, the quantiser of the latent write and the dequantiser of the
/// attention kernels' staging, bit for bit (`crates/cuda/tests/mla_kv8.rs` holds the device to it).
pub mod kv8 {
    use super::{BLOCK_TOKENS, LATENT_DIM};
    use crate::exl3_linear::Dtype;

    /// Byte offset of the scales in a page: the values `[256][512]` come first.
    pub const SCALE_OFF: usize = BLOCK_TOKENS * LATENT_DIM;
    /// Bytes of one page: 131,072 of values and 1,024 of scales.
    pub const PAGE_BYTES: usize = SCALE_OFF + BLOCK_TOKENS * 4;
    /// The largest finite E4M3 magnitude.
    pub const E4M3_MAX: f32 = 448.0;

    /// The E4M3 (FN: bias 7, no infinities, `0x7F`/`0xFF` NaN) code of `x`, round to nearest even, saturating at +-448: what
    /// `__nv_cvt_float2_to_fp8x2(.., __NV_SATFINITE, __NV_E4M3)` gives.
    pub fn e4m3_rne(x: f32) -> u8 {
        let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
        if x.is_nan() {
            return 0x7F;
        }
        let a = x.abs();
        if a >= E4M3_MAX {
            return sign | 0x7E;
        }
        // below 2^-6 the codes are the multiples of 2^-9 (subnormals; 8 * 2^-9 is the smallest normal 0x08): a * 512 is exact
        if a < 1.0 / 64.0 {
            return sign | (a * 512.0).round_ties_even() as u8;
        }
        let bits = a.to_bits();
        let e = ((bits >> 23) & 0xFF) as i32 - 127;
        let mut man = ((bits >> 20) & 0x7) as i32;
        let rem = bits & 0x000F_FFFF;
        if rem > 0x8_0000 || (rem == 0x8_0000 && man & 1 == 1) {
            man += 1;
        }
        let mut exp = e + 7;
        if man == 8 {
            man = 0;
            exp += 1;
        }
        if exp > 15 || (exp == 15 && man > 6) {
            return sign | 0x7E;
        }
        sign | ((exp as u8) << 3) | man as u8
    }

    /// One latent row (`LATENT_DIM` values as f32, read from T) quantised: its E4M3 bytes and its scale `amax / 448`. A row whose
    /// scale is 0 (all zeros) is zero bytes.
    pub fn quantise_row(row: &[f32]) -> (Vec<u8>, f32) {
        assert_eq!(row.len(), LATENT_DIM);
        let amax = row.iter().fold(0.0f32, |m, &v| if v.abs() > m { v.abs() } else { m });
        let s = amax / E4M3_MAX;
        if !(s > 0.0) {
            return (vec![0u8; LATENT_DIM], s);
        }
        (row.iter().map(|&v| e4m3_rne(v / s)).collect(), s)
    }

    /// The T bits (fp16 or bf16) a kernel stages for E4M3 byte `b` at scale `s`: `round_T(f32(b) * s)`.
    pub fn dequantise(b: u8, s: f32, dtype: Dtype) -> u16 {
        let v = crate::fp8_linear::e4m3_to_f32(b) * s;
        match dtype {
            Dtype::F16 => gb_exl3::f16::f64_to_f16(v as f64),
            Dtype::BF16 => crate::num::f32_to_bf16(v),
            Dtype::F32 => panic!("the latent cache holds 16-bit values"),
        }
    }

    /// T bits of a 16-bit value as f32.
    pub fn to_f32(v: u16, dtype: Dtype) -> f32 {
        match dtype {
            Dtype::F16 => gb_exl3::f16::f16_to_f64(v) as f32,
            Dtype::BF16 => crate::num::bf16_to_f32(v),
            Dtype::F32 => panic!("the latent cache holds 16-bit values"),
        }
    }

    /// A row of T bits as the kernels read it back after a write: quantised, then dequantised.
    pub fn round_trip(row: &[u16], dtype: Dtype) -> Vec<u16> {
        let f: Vec<f32> = row.iter().map(|&v| to_f32(v, dtype)).collect();
        let (q, s) = quantise_row(&f);
        q.iter().map(|&b| dequantise(b, s, dtype)).collect()
    }

    /// The row of token `tok` of page `page` from a host copy of a whole fp8 cache, dequantised to T bits.
    pub fn read_row(cache: &[u8], page: usize, tok: usize, dtype: Dtype) -> Vec<u16> {
        let base = page * PAGE_BYTES;
        let s = f32::from_le_bytes(cache[base + SCALE_OFF + tok * 4..][..4].try_into().unwrap());
        cache[base + tok * LATENT_DIM..][..LATENT_DIM].iter().map(|&b| dequantise(b, s, dtype)).collect()
    }
}

/// One layer's paged latent cache: `T [n_blocks][256][512]`, or the fp8 page ([`LatentKv::Fp8`]).
#[derive(Debug)]
pub struct LatentCache {
    pub lat: DevBuf,
    pub n_blocks: usize,
    /// The activation type: what the kernels stage and attend in (the values themselves with [`LatentKv::Plain`]).
    pub dtype: Dtype,
    pub kv: LatentKv,
}

impl LatentCache {
    /// Bytes of one block of latents.
    pub fn block_bytes(dtype: Dtype, kv: LatentKv) -> usize {
        kv.block_bytes(dtype)
    }
    /// Allocates an (uninitialised) cache of `n_blocks` pages.
    pub fn new(n_blocks: usize, dtype: Dtype, kv: LatentKv) -> Result<Self, CudaError> {
        check_dtype(dtype);
        Ok(Self { lat: DevBuf::alloc(n_blocks * kv.block_bytes(dtype))?, n_blocks, dtype, kv })
    }
    pub fn capacity_tokens(&self) -> usize {
        self.n_blocks * BLOCK_TOKENS
    }
    /// Bytes of one block of this cache.
    pub fn page_bytes(&self) -> usize {
        self.kv.block_bytes(self.dtype)
    }
    /// The byte ranges `(offset, len)` of rows `0..rows` of page `page` (rows < 256): the values, then for fp8 the scales. A page copy
    /// or a digest of a partial page covers exactly these.
    pub fn row_ranges(&self, page: usize, rows: usize) -> Vec<(usize, usize)> {
        let base = page * self.page_bytes();
        match self.kv {
            LatentKv::Plain => vec![(base, rows * LATENT_DIM * self.dtype.size())],
            LatentKv::Fp8 => vec![(base, rows * LATENT_DIM), (base + kv8::SCALE_OFF, rows * 4)],
        }
    }
    /// Rows `0..n` of the sequence whose pages are `blocks`, as the kernels read them: T bits `[n][512]` (dequantised for fp8). A
    /// debug and test path: copies the whole cache to the host. Synchronous.
    pub fn read_rows(&self, blocks: &[u32], n: usize) -> Result<Vec<u16>, CudaError> {
        let mut out = Vec::with_capacity(n * LATENT_DIM);
        match self.kv {
            LatentKv::Plain => {
                let all: Vec<u16> = self.lat.to_vec()?;
                for p in 0..n {
                    let at = plane_offset(blocks[p / BLOCK_TOKENS] as usize, p % BLOCK_TOKENS, BLOCK_TOKENS, LATENT_DIM);
                    out.extend_from_slice(&all[at..at + LATENT_DIM]);
                }
            }
            LatentKv::Fp8 => {
                let all: Vec<u8> = self.lat.to_vec()?;
                for p in 0..n {
                    out.extend(kv8::read_row(&all, blocks[p / BLOCK_TOKENS] as usize, p % BLOCK_TOKENS, self.dtype));
                }
            }
        }
        Ok(out)
    }
}

/// `kv_b_proj` of one MLA layer split and packed for the kernels: `uk` `T [64][256 d][512 c]` (head `h`'s rows `h*512 + d`,
/// `d < 256`, contiguous in `c`) and `uvt` `T [64][512 c][256 n]` (rows `h*512 + 256 + n`, transposed to `[c][n]`), so both
/// products stream their weights with coalesced loads. Together they are the same 32 MiB as the checkpoint tensor.
#[derive(Debug)]
pub struct MlaWeights {
    pub uk: DevBuf,
    pub uvt: DevBuf,
    pub dtype: Dtype,
}

fn convert(v: u16, src: Dtype, dst: Dtype) -> u16 {
    if src == dst {
        return v;
    }
    let f = match src {
        Dtype::BF16 => crate::num::bf16_to_f32(v) as f64,
        Dtype::F16 => gb_exl3::f16::f16_to_f64(v),
        Dtype::F32 => panic!("kv_b_proj is 16-bit"),
    };
    match dst {
        Dtype::BF16 => crate::num::f32_to_bf16(f as f32),
        Dtype::F16 => gb_exl3::f16::f64_to_f16(f),
        Dtype::F32 => panic!("kv_b_proj is 16-bit"),
    }
}

/// Splits and packs the checkpoint's `kv_b_proj.weight` (`[64 * 512][512]`, 16-bit bits, row `h*512 + j`: `j < 256` is `W_UK`,
/// `j >= 256` is `W_UV`) into `(uk, uvt)` host vectors in the kernels' layouts, converting `src` -> `dst` through f32
/// (identity when equal).
pub fn pack_kv_b(rows: &[u16], src: Dtype, dst: Dtype) -> (Vec<u16>, Vec<u16>) {
    assert_eq!(rows.len(), N_HEADS * KV_B_ROWS_PER_HEAD * LATENT_DIM, "kv_b_proj must be [64*512][512]");
    let mut uk = vec![0u16; N_HEADS * NOPE_DIM * LATENT_DIM];
    let mut uvt = vec![0u16; N_HEADS * LATENT_DIM * NOPE_DIM];
    for h in 0..N_HEADS {
        for d in 0..NOPE_DIM {
            let k_row = &rows[(h * KV_B_ROWS_PER_HEAD + d) * LATENT_DIM..][..LATENT_DIM];
            for (c, &v) in k_row.iter().enumerate() {
                uk[(h * NOPE_DIM + d) * LATENT_DIM + c] = convert(v, src, dst);
            }
            let v_row = &rows[(h * KV_B_ROWS_PER_HEAD + NOPE_DIM + d) * LATENT_DIM..][..LATENT_DIM];
            for (c, &v) in v_row.iter().enumerate() {
                uvt[(h * LATENT_DIM + c) * NOPE_DIM + d] = convert(v, src, dst);
            }
        }
    }
    (uk, uvt)
}

impl MlaWeights {
    /// Uploads `kv_b_proj.weight` (see [`pack_kv_b`]).
    pub fn from_kv_b(rows: &[u16], src: Dtype, dst: Dtype) -> Result<Self, CudaError> {
        check_dtype(dst);
        let (uk, uvt) = pack_kv_b(rows, src, dst);
        Ok(Self { uk: DevBuf::from_slice(&uk)?, uvt: DevBuf::from_slice(&uvt)?, dtype: dst })
    }
    /// Device bytes of both halves.
    pub fn bytes(&self) -> usize {
        self.uk.bytes() + self.uvt.bytes()
    }
}

/// Bytes of workspace [`attend`] needs for up to `rows` rows and positions below `max_context`: fp32 chunk partials
/// `o [rows][64][chunks][512]`, `m` and `l [rows][64][chunks]`, `chunks = ceil(max_context / 256)`. (1 MiB per row at 2048
/// tokens: 9.4 MiB for a 9-row window.)
pub fn attend_workspace_bytes(rows: usize, max_context: usize) -> usize {
    let nch = max_context.div_ceil(CHUNK_KEYS).max(1);
    let pairs = rows * N_HEADS * nch;
    let a = |x: usize| x.div_ceil(256) * 256;
    a(pairs * LATENT_DIM * 4) + 2 * a(pairs * 4)
}

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        #[allow(clippy::too_many_arguments)]
        pub fn gb_mla_latent_write(
            lat: *const c_void,
            row_slot: *const i32,
            row_pos: *const i32,
            block_table: *const i32,
            max_blocks: i32,
            cache: *mut c_void,
            rows: i32,
            dtype: i32,
            kv: i32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_mla_absorb_nh(
            q: *const c_void,
            q_row_stride: i64,
            w_uk: *const c_void,
            q_lat: *mut c_void,
            rows: i32,
            dtype: i32,
            nh: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_mla_expand_nh(
            o_lat: *const c_void,
            w_uvt: *const c_void,
            out: *mut c_void,
            rows: i32,
            dtype: i32,
            nh: i32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_mla_attend_nh(
            q_lat: *const c_void,
            cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            dense_limit: i32,
            out: *mut c_void,
            rows: i32,
            max_context: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            dtype: i32,
            kv: i32,
            nh: i32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_mla_attend_sparse_nh(
            q_lat: *const c_void,
            cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            sel_idx: *const i32,
            sel_stride: i32,
            sel_meta: *const i32,
            out: *mut c_void,
            rows: i32,
            max_sel: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            dtype: i32,
            kv: i32,
            nh: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_mla_absorb(
            q: *const c_void,
            q_row_stride: i64,
            w_uk: *const c_void,
            q_lat: *mut c_void,
            rows: i32,
            dtype: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_mla_expand(
            o_lat: *const c_void,
            w_uvt: *const c_void,
            out: *mut c_void,
            rows: i32,
            dtype: i32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(dead_code)] // checked against the Rust formula in the tests
        pub fn gb_mla_attend_workspace_bytes(rows: i32, max_context: i32) -> usize;
        pub fn gb_mla_attend_workspace_bytes_variant(variant: i32, rows: i32, max_context: i32) -> usize;
        pub fn gb_mla_attend_variant(
            q_lat: *const c_void,
            cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            dense_limit: i32,
            out: *mut c_void,
            rows: i32,
            max_context: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            dtype: i32,
            kv: i32,
            variant: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_mla_attend_sparse(
            q_lat: *const c_void,
            cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            sel_idx: *const i32,
            sel_stride: i32,
            sel_meta: *const i32,
            out: *mut c_void,
            rows: i32,
            max_sel: i32,
            workspace: *mut c_void,
            workspace_bytes: usize,
            dtype: i32,
            kv: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// Writes row `r`'s normalised latent (`lat`: `T [rows][512]`) into the cache at `(row_slot[r], row_pos[r])`: the values, or for an
/// fp8 cache their E4M3 bytes and the row's scale ([`kv8::quantise_row`]). A negative position skips the row. Rows must target distinct
/// positions.
pub fn latent_write(
    lat: &DevBuf,
    rows: &RowIndex,
    table: &BlockTable,
    cache: &mut LatentCache,
    n_rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert!(n_rows <= rows.rows);
    assert!(lat.bytes() >= n_rows * LATENT_DIM * cache.dtype.size());
    let rc = unsafe {
        ffi::gb_mla_latent_write(
            lat.ptr(),
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            table.buf.ptr() as *const i32,
            table.max_blocks as i32,
            cache.lat.ptr(),
            n_rows as i32,
            check_dtype(cache.dtype),
            cache.kv.id(),
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Absorb: `q_lat[r][h][c] = sum_d q[r][h][d] * W_UK[h][d][c]` for `n_rows` rows (any count; the weights are read once per
/// pass of up to 16 rows). `q`: `T [rows][q_row_stride]` with head `h` at `h * 256` (use `64 * 256` for a plain `q_b_proj`
/// output); `q_lat`: `T [rows][64][512]`. Each output element is one fixed-order fp32 chain, rounded once to T.
pub fn absorb(
    q: &DevBuf,
    q_row_stride: usize,
    w: &MlaWeights,
    q_lat: &mut DevBuf,
    n_rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    let dt = check_dtype(w.dtype);
    assert!(q_row_stride >= N_HEADS * NOPE_DIM && q_row_stride % 8 == 0, "q rows must be 16-byte aligned and hold 64 x 256");
    assert!(n_rows == 0 || q.bytes() >= ((n_rows - 1) * q_row_stride + N_HEADS * NOPE_DIM) * w.dtype.size());
    assert!(q_lat.bytes() >= n_rows * N_HEADS * LATENT_DIM * w.dtype.size());
    let rc =
        unsafe { ffi::gb_mla_absorb(q.ptr(), q_row_stride as i64, w.uk.ptr(), q_lat.ptr(), n_rows as i32, dt, stream.raw()) };
    CudaError::check(rc)
}

/// Expand: `out[r][h*256 + n] = sum_c o_lat[r][h][c] * W_UV[h][c][n]`. `o_lat`: `T [rows][64][512]`, `out`: `T [rows][64*256]`.
pub fn expand(o_lat: &DevBuf, w: &MlaWeights, out: &mut DevBuf, n_rows: usize, stream: &Stream) -> Result<(), CudaError> {
    let dt = check_dtype(w.dtype);
    assert!(o_lat.bytes() >= n_rows * N_HEADS * LATENT_DIM * w.dtype.size());
    assert!(out.bytes() >= n_rows * N_HEADS * NOPE_DIM * w.dtype.size());
    let rc = unsafe { ffi::gb_mla_expand(o_lat.ptr(), w.uvt.ptr(), out.ptr(), n_rows as i32, dt, stream.raw()) };
    CudaError::check(rc)
}

/// Dense latent attention for `n_rows` (1..=128) rows: `q_lat`, `o_lat` `T [n_rows][64][512]`. Row `r` attends to keys
/// `0..=row_pos[r]` of slot `row_slot[r]` (already written by [`latent_write`]); a negative position is a padding row
/// (its `o_lat` is zeros). Every position must be below `max_context`, which sizes the grid and the workspace
/// ([`attend_workspace_bytes`]); a `max_context` larger than needed is a no-op for the extra chunks. Any context works (the
/// kernel is the dense reference at 262,144 tokens too); the engine routes rows past [`DENSE_MAX_KEYS`] to [`attend_sparse`].
#[allow(clippy::too_many_arguments)]
pub fn attend(
    q_lat: &DevBuf,
    cache: &LatentCache,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    o_lat: &mut DevBuf,
    max_context: usize,
    workspace: &mut DevBuf,
    stream: &Stream,
) -> Result<(), CudaError> {
    attend_variant(q_lat, cache, table, rows, n_rows, o_lat, max_context, workspace, 0, stream)
}

/// [`attend`] with an explicit tile/chunk shape (tuning; 0 is the default and the only contract: the others change bits).
#[allow(clippy::too_many_arguments)]
pub fn attend_variant(
    q_lat: &DevBuf,
    cache: &LatentCache,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    o_lat: &mut DevBuf,
    max_context: usize,
    workspace: &mut DevBuf,
    variant: i32,
    stream: &Stream,
) -> Result<(), CudaError> {
    attend_impl(q_lat, cache, table, rows, n_rows, o_lat, i32::MAX as usize, max_context, workspace, variant, stream)
}

/// [`attend`] for the rows that are dense by [`row_is_dense`] semantics at `dense_limit` (rows with `pos < dense_limit`); every
/// other row of the call is left untouched (no write at all, not even zeros), so the sparse kernel can fill them into the same
/// `o_lat`. The grid covers `min(max_context, dense_limit)` tokens only: a call at 262,144 tokens launches 8 chunks, not 1024.
#[allow(clippy::too_many_arguments)]
pub fn attend_below(
    q_lat: &DevBuf,
    cache: &LatentCache,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    o_lat: &mut DevBuf,
    dense_limit: usize,
    max_context: usize,
    workspace: &mut DevBuf,
    stream: &Stream,
) -> Result<(), CudaError> {
    attend_impl(q_lat, cache, table, rows, n_rows, o_lat, dense_limit, max_context.min(dense_limit), workspace, 0, stream)
}

#[allow(clippy::too_many_arguments)]
fn attend_impl(
    q_lat: &DevBuf,
    cache: &LatentCache,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    o_lat: &mut DevBuf,
    dense_limit: usize,
    max_context: usize,
    workspace: &mut DevBuf,
    variant: i32,
    stream: &Stream,
) -> Result<(), CudaError> {
    let dt = check_dtype(cache.dtype);
    assert!((1..=MAX_ROWS).contains(&n_rows) && n_rows <= rows.rows);
    let row_bytes = N_HEADS * LATENT_DIM * cache.dtype.size();
    assert!(q_lat.bytes() >= n_rows * row_bytes && o_lat.bytes() >= n_rows * row_bytes);
    assert!(max_context >= 1 && max_context <= table.max_blocks * BLOCK_TOKENS, "max_context exceeds the block table");
    let need = unsafe { ffi::gb_mla_attend_workspace_bytes_variant(variant, n_rows as i32, max_context as i32) };
    assert!(workspace.bytes() >= need, "MLA workspace: {} bytes < {need}", workspace.bytes());
    let rc = unsafe {
        ffi::gb_mla_attend_variant(
            q_lat.ptr(),
            cache.lat.ptr(),
            table.buf.ptr() as *const i32,
            table.max_blocks as i32,
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            dense_limit.min(i32::MAX as usize) as i32,
            o_lat.ptr(),
            n_rows as i32,
            max_context as i32,
            workspace.ptr(),
            workspace.bytes(),
            dt,
            cache.kv.id(),
            variant,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Longest selection list of a sparse row: 512 pools of 4 tokens plus the 0..3 raw tokens of the incomplete last pool
/// (`index_topk + index_kpool - 1`).
pub const MAX_SELECTION: usize = DENSE_MAX_KEYS + 3;

/// The DSA indexer's result for a call, as `dsa::Selection` produces it: for row `r`, `meta[2r] == 1` means the row is sparse
/// and `indices[r * stride ..][.. meta[2r + 1]]` are the attended token positions (absolute, `i32`; the indexer produces them
/// ascending, but the kernel takes any order: only the bit order of the sum follows the list). An entry holding `-1` inside the
/// count is a hole (the ExLlamaV3 k-pool path leaves them for pools without a score): it is masked, not scored as a zero key. A list
/// whose entries are all holes writes zeros. Entries past the count are never read. `meta[2r] == 0` means the row is dense and its
/// list is not read. `indices`: `i32 [rows][stride >= MAX_SELECTION]`, `meta`: `i32 [rows][2]`.
#[derive(Debug, Clone, Copy)]
pub struct SparseSel<'a> {
    pub indices: &'a DevBuf,
    pub meta: &'a DevBuf,
    /// Elements between rows of `indices`.
    pub stride: usize,
}

/// Bytes of workspace [`attend_sparse`] needs for `rows` rows: the chunk partials of a [`MAX_SELECTION`]-entry list.
pub fn attend_sparse_workspace_bytes(rows: usize) -> usize {
    attend_workspace_bytes(rows, MAX_SELECTION)
}

/// Sparse (DSA) latent attention: the rows with `sel.meta[2r] == 1` attend to exactly the keys their selection lists name, taken in
/// list order in chunks of 256 entries merged in chunk order (so a row's bits depend on its own list and query only, as in the
/// dense kernel). Every other row of the call is left untouched. `o_lat`: `T [n_rows][64][512]`, the same buffer [`attend_below`]
/// fills for the dense rows. Cost does not grow with the context: a sparse row reads 2048 + (pos+1) % 4 latents however long
/// its sequence is.
#[allow(clippy::too_many_arguments)]
pub fn attend_sparse(
    q_lat: &DevBuf,
    cache: &LatentCache,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    sel: &SparseSel,
    o_lat: &mut DevBuf,
    workspace: &mut DevBuf,
    stream: &Stream,
) -> Result<(), CudaError> {
    let dt = check_dtype(cache.dtype);
    assert!((1..=MAX_ROWS).contains(&n_rows) && n_rows <= rows.rows);
    let row_bytes = N_HEADS * LATENT_DIM * cache.dtype.size();
    assert!(q_lat.bytes() >= n_rows * row_bytes && o_lat.bytes() >= n_rows * row_bytes);
    assert!(sel.stride >= MAX_SELECTION && sel.indices.bytes() >= n_rows * sel.stride * 4, "selection indices too small");
    assert!(sel.meta.bytes() >= n_rows * 2 * 4, "selection meta too small");
    assert!(workspace.bytes() >= attend_sparse_workspace_bytes(n_rows), "MLA sparse workspace too small");
    let rc = unsafe {
        ffi::gb_mla_attend_sparse(
            q_lat.ptr(),
            cache.lat.ptr(),
            table.buf.ptr() as *const i32,
            table.max_blocks as i32,
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            sel.indices.ptr() as *const i32,
            sel.stride as i32,
            sel.meta.ptr() as *const i32,
            o_lat.ptr(),
            n_rows as i32,
            MAX_SELECTION as i32,
            workspace.ptr(),
            workspace.bytes(),
            dt,
            cache.kv.id(),
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Bytes of workspace [`attend_auto`] needs: the larger of its two launches (they run in stream order and share one buffer).
pub fn attend_auto_workspace_bytes(rows: usize, max_context: usize) -> usize {
    attend_workspace_bytes(rows, max_context.min(DENSE_MAX_KEYS)).max(attend_sparse_workspace_bytes(rows))
}

/// Attention for a window or batch of any mix of regimes, decided per row: rows with `pos < DENSE_MAX_KEYS` through the dense
/// kernel, the rows `sel` marks sparse through [`attend_sparse`]; each row exactly as if it were decoded alone. A verify window
/// straddling 2,048 tokens is the case this exists for. `max_context` bounds the positions of the call; when it is at most
/// [`DENSE_MAX_KEYS`] the sparse launch is skipped (`sel` is not read).
#[allow(clippy::too_many_arguments)]
pub fn attend_auto(
    q_lat: &DevBuf,
    cache: &LatentCache,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    sel: &SparseSel,
    o_lat: &mut DevBuf,
    max_context: usize,
    workspace: &mut DevBuf,
    stream: &Stream,
) -> Result<(), CudaError> {
    attend_below(q_lat, cache, table, rows, n_rows, o_lat, DENSE_MAX_KEYS, max_context, workspace, stream)?;
    if max_context > DENSE_MAX_KEYS {
        attend_sparse(q_lat, cache, table, rows, n_rows, sel, o_lat, workspace, stream)?;
    }
    Ok(())
}

/// Scratch for [`mla_rows`] and [`mla_rows_auto`]: absorbed queries, attended latents and the attention workspace, sized for up
/// to `max_rows` rows and positions below `max_context`.
#[derive(Debug)]
pub struct MlaScratch {
    pub q_lat: DevBuf,
    pub o_lat: DevBuf,
    pub ws: DevBuf,
    pub max_rows: usize,
    pub max_context: usize,
}

impl MlaScratch {
    /// Bytes of one scratch for the given limits (what the memory planner reserves per call chain).
    pub fn bytes(max_rows: usize, max_context: usize, dtype: Dtype) -> usize {
        2 * max_rows * N_HEADS * LATENT_DIM * dtype.size() + attend_auto_workspace_bytes(max_rows, max_context)
    }
    pub fn new(max_rows: usize, max_context: usize, dtype: Dtype) -> Result<Self, CudaError> {
        check_dtype(dtype);
        let row = N_HEADS * LATENT_DIM * dtype.size();
        Ok(Self {
            q_lat: DevBuf::alloc(max_rows * row)?,
            o_lat: DevBuf::alloc(max_rows * row)?,
            ws: DevBuf::alloc(attend_auto_workspace_bytes(max_rows, max_context))?,
            max_rows,
            max_context,
        })
    }
}

/// The whole attention math of one MLA layer for `n_rows` rows, after the latents are in the cache: absorb, dense attend, expand.
/// `q`: `T [rows][q_row_stride]` (nope parts, head `h` at `h * 256`); `out`: `T [rows][64*256]`, the `o_proj` input. Every row is
/// attended densely, so `max_context` must be small enough for the scratch's workspace ([`attend_workspace_bytes`]); engines use
/// [`mla_rows_auto`].
#[allow(clippy::too_many_arguments)]
pub fn mla_rows(
    q: &DevBuf,
    q_row_stride: usize,
    w: &MlaWeights,
    cache: &LatentCache,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    out: &mut DevBuf,
    max_context: usize,
    scratch: &mut MlaScratch,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert_eq!(w.dtype, cache.dtype, "weights and cache must share one dtype");
    assert!(n_rows <= scratch.max_rows && max_context <= scratch.max_context, "scratch too small");
    absorb(q, q_row_stride, w, &mut scratch.q_lat, n_rows, stream)?;
    attend(&scratch.q_lat, cache, table, rows, n_rows, &mut scratch.o_lat, max_context, &mut scratch.ws, stream)?;
    expand(&scratch.o_lat, w, out, n_rows, stream)
}

/// [`mla_rows`] with the per-row regime: dense rows through the dense kernel, the rows `sel` marks sparse through the sparse one
/// (see [`attend_auto`]). This is what the model code calls for every MLA layer.
#[allow(clippy::too_many_arguments)]
pub fn mla_rows_auto(
    q: &DevBuf,
    q_row_stride: usize,
    w: &MlaWeights,
    cache: &LatentCache,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    sel: &SparseSel,
    out: &mut DevBuf,
    max_context: usize,
    scratch: &mut MlaScratch,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert_eq!(w.dtype, cache.dtype, "weights and cache must share one dtype");
    assert!(n_rows <= scratch.max_rows && max_context <= scratch.max_context, "scratch too small");
    absorb(q, q_row_stride, w, &mut scratch.q_lat, n_rows, stream)?;
    attend_auto(&scratch.q_lat, cache, table, rows, n_rows, sel, &mut scratch.o_lat, max_context, &mut scratch.ws, stream)?;
    expand(&scratch.o_lat, w, out, n_rows, stream)
}

fn check_heads(h0: usize, nh: usize) {
    assert!(nh >= 16 && nh % 16 == 0 && h0 + nh <= N_HEADS, "heads {h0}..{} of {N_HEADS}", h0 + nh);
}

/// [`absorb`] on heads `[h0, h0 + nh)` (lane E, the TP head split): `q` `T [rows][q_row_stride]` holds those heads (head `h0 + i` at
/// `i * 256`), `q_lat` `T [rows][nh][512]`. Each head's output is the 64-head call's bit for bit.
#[allow(clippy::too_many_arguments)]
pub fn absorb_heads(
    q: &DevBuf,
    q_row_stride: usize,
    w: &MlaWeights,
    h0: usize,
    nh: usize,
    q_lat: &mut DevBuf,
    n_rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    check_heads(h0, nh);
    let (dt, es) = (check_dtype(w.dtype), w.dtype.size());
    assert!(q_row_stride >= nh * NOPE_DIM && q_row_stride % 8 == 0);
    assert!(n_rows == 0 || q.bytes() >= ((n_rows - 1) * q_row_stride + nh * NOPE_DIM) * es);
    assert!(q_lat.bytes() >= n_rows * nh * LATENT_DIM * es);
    let uk = w.uk.ptr_at(h0 * NOPE_DIM * LATENT_DIM * es);
    CudaError::check(unsafe {
        ffi::gb_mla_absorb_nh(q.ptr(), q_row_stride as i64, uk, q_lat.ptr(), n_rows as i32, dt, nh as i32, stream.raw())
    })
}

/// [`expand`] on heads `[h0, h0 + nh)`: `o_lat` `T [rows][nh][512]`, `out` `T [rows][nh * 256]`.
pub fn expand_heads(
    o_lat: &DevBuf,
    w: &MlaWeights,
    h0: usize,
    nh: usize,
    out: &mut DevBuf,
    n_rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    check_heads(h0, nh);
    let (dt, es) = (check_dtype(w.dtype), w.dtype.size());
    assert!(o_lat.bytes() >= n_rows * nh * LATENT_DIM * es && out.bytes() >= n_rows * nh * NOPE_DIM * es);
    let uvt = w.uvt.ptr_at(h0 * LATENT_DIM * NOPE_DIM * es);
    CudaError::check(unsafe { ffi::gb_mla_expand_nh(o_lat.ptr(), uvt, out.ptr(), n_rows as i32, dt, nh as i32, stream.raw()) })
}

/// [`attend_below`] (the default tiling) on `nh` heads: `q_lat`, `o_lat` `T [rows][nh][512]`.
#[allow(clippy::too_many_arguments)]
pub fn attend_below_heads(
    q_lat: &DevBuf,
    cache: &LatentCache,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    o_lat: &mut DevBuf,
    dense_limit: usize,
    max_context: usize,
    workspace: &mut DevBuf,
    nh: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    check_heads(0, nh);
    let dt = check_dtype(cache.dtype);
    assert!((1..=MAX_ROWS).contains(&n_rows) && n_rows <= rows.rows);
    let row_bytes = nh * LATENT_DIM * cache.dtype.size();
    assert!(q_lat.bytes() >= n_rows * row_bytes && o_lat.bytes() >= n_rows * row_bytes);
    let mc = max_context.min(dense_limit);
    assert!(mc >= 1 && mc <= table.max_blocks * BLOCK_TOKENS, "max_context exceeds the block table");
    let need = unsafe { ffi::gb_mla_attend_workspace_bytes_variant(0, n_rows as i32, mc as i32) };
    assert!(workspace.bytes() >= need, "MLA workspace: {} bytes < {need}", workspace.bytes());
    CudaError::check(unsafe {
        ffi::gb_mla_attend_nh(
            q_lat.ptr(),
            cache.lat.ptr(),
            table.buf.ptr() as *const i32,
            table.max_blocks as i32,
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            dense_limit.min(i32::MAX as usize) as i32,
            o_lat.ptr(),
            n_rows as i32,
            mc as i32,
            workspace.ptr(),
            workspace.bytes(),
            dt,
            cache.kv.id(),
            nh as i32,
            stream.raw(),
        )
    })
}

/// [`attend_sparse`] on `nh` heads.
#[allow(clippy::too_many_arguments)]
pub fn attend_sparse_heads(
    q_lat: &DevBuf,
    cache: &LatentCache,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    sel: &SparseSel,
    o_lat: &mut DevBuf,
    workspace: &mut DevBuf,
    nh: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    check_heads(0, nh);
    let dt = check_dtype(cache.dtype);
    assert!((1..=MAX_ROWS).contains(&n_rows) && n_rows <= rows.rows);
    let row_bytes = nh * LATENT_DIM * cache.dtype.size();
    assert!(q_lat.bytes() >= n_rows * row_bytes && o_lat.bytes() >= n_rows * row_bytes);
    assert!(sel.stride >= MAX_SELECTION && sel.indices.bytes() >= n_rows * sel.stride * 4, "selection indices too small");
    assert!(sel.meta.bytes() >= n_rows * 2 * 4, "selection meta too small");
    assert!(workspace.bytes() >= attend_sparse_workspace_bytes(n_rows), "MLA sparse workspace too small");
    CudaError::check(unsafe {
        ffi::gb_mla_attend_sparse_nh(
            q_lat.ptr(),
            cache.lat.ptr(),
            table.buf.ptr() as *const i32,
            table.max_blocks as i32,
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            sel.indices.ptr() as *const i32,
            sel.stride as i32,
            sel.meta.ptr() as *const i32,
            o_lat.ptr(),
            n_rows as i32,
            MAX_SELECTION as i32,
            workspace.ptr(),
            workspace.bytes(),
            dt,
            cache.kv.id(),
            nh as i32,
            stream.raw(),
        )
    })
}

/// [`mla_rows_auto`] on heads `[h0, h0 + nh)` of the layer (lane E, the TP head split: a rank's 32 heads): `q` `T [rows][q_row_stride]`
/// holds those heads (head `h0 + i` at `i * 256`), `out` `T [rows][nh * 256]` gets their attention outputs (the `o_proj` input columns of the
/// heads), the scratch's `q_lat` / `o_lat` hold `nh` heads. The same kernels on the heads' own weights and operands: every head's output is
/// the 64-head call's bit for bit (`crates/cuda/tests/mla_split.rs`). `nh`: a multiple of 16 up to 64.
#[allow(clippy::too_many_arguments)]
pub fn mla_rows_auto_heads(
    q: &DevBuf,
    q_row_stride: usize,
    w: &MlaWeights,
    h0: usize,
    nh: usize,
    cache: &LatentCache,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    sel: &SparseSel,
    out: &mut DevBuf,
    max_context: usize,
    scratch: &mut MlaScratch,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert_eq!(w.dtype, cache.dtype, "weights and cache must share one dtype");
    assert!(n_rows <= scratch.max_rows && max_context <= scratch.max_context, "scratch too small");
    absorb_heads(q, q_row_stride, w, h0, nh, &mut scratch.q_lat, n_rows, stream)?;
    attend_below_heads(
        &scratch.q_lat,
        cache,
        table,
        rows,
        n_rows,
        &mut scratch.o_lat,
        DENSE_MAX_KEYS,
        max_context,
        &mut scratch.ws,
        nh,
        stream,
    )?;
    if max_context > DENSE_MAX_KEYS {
        attend_sparse_heads(&scratch.q_lat, cache, table, rows, n_rows, sel, &mut scratch.o_lat, &mut scratch.ws, nh, stream)?;
    }
    expand_heads(&scratch.o_lat, w, h0, nh, out, n_rows, stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(gb_no_cuda))] // asks the compiled kernels
    #[test]
    fn workspace_formula_matches_the_kernel_side() {
        for (r, c) in [(1, 1), (1, 256), (1, 257), (8, 2048), (128, 5000)] {
            assert_eq!(attend_workspace_bytes(r, c), unsafe { ffi::gb_mla_attend_workspace_bytes(r as i32, c as i32) });
        }
    }

    #[test]
    fn workspace_sizes() {
        assert_eq!(attend_workspace_bytes(1, 2048) >> 20, 1); // MiB
        assert_eq!(attend_workspace_bytes(9, 2048) / (1 << 20), 9);
    }

    #[test]
    fn dense_predicate_is_per_row() {
        assert!(row_is_dense(-1) && row_is_dense(0) && row_is_dense(2047));
        assert!(!row_is_dense(2048) && !row_is_dense(100_000));
    }

    #[test]
    fn plane_addressing_is_shared() {
        // latent: (block 3, token 5) of a [n][256][512] plane; indexer k||gate: [n][256][256]; pool: [n][64][128] (token 5 -> entry 1)
        assert_eq!(plane_offset(3, 5, 256, 512), (3 * 256 + 5) * 512);
        assert_eq!(plane_offset(3, 5, 256, 256), (3 * 256 + 5) * 256);
        assert_eq!(plane_offset(3, 5 / 4, 64, 128), (3 * 64 + 1) * 128);
    }

    #[test]
    fn pack_kv_b_splits_k_and_v_per_head_and_transposes_v() {
        // element value encodes (row, col) so any mix-up of the split, the head stride or the transpose is visible
        let rows: Vec<u16> = (0..N_HEADS * KV_B_ROWS_PER_HEAD * LATENT_DIM)
            .map(|i| (((i / LATENT_DIM) * 7 + (i % LATENT_DIM) * 13) & 0x7fff) as u16)
            .collect();
        let (uk, uvt) = pack_kv_b(&rows, Dtype::BF16, Dtype::BF16);
        let at = |r: usize, c: usize| rows[r * LATENT_DIM + c];
        for (h, d, c) in [(0usize, 0usize, 0usize), (0, 255, 511), (1, 0, 3), (63, 255, 7), (17, 100, 400)] {
            assert_eq!(uk[(h * NOPE_DIM + d) * LATENT_DIM + c], at(h * 512 + d, c), "uk h{h} d{d} c{c}");
            assert_eq!(uvt[(h * LATENT_DIM + c) * NOPE_DIM + d], at(h * 512 + 256 + d, c), "uvt h{h} n{d} c{c}");
        }
    }
}
