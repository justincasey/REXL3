//! What the GLM backend allocates, as arithmetic: one list of named byte counts, computed from the config and the serving settings with the
//! kernel crates' own size functions, that the memory model (`memory.rs`) prices admission with and the backend's real allocations are
//! checked against (a GPU test sums every `DevBuf` the backend owns and compares item by item; a CPU test compares the plan with the memory
//! model). Nothing here touches the GPU.
//!
//! Four kinds of memory, in the order admission cares about them (with the NextN/MTP drafter on, each also holds that layer's share: a 12th
//! MLA layer in the pages and the per-slot ring, a store of pending tapped hidden rows per slot, and the drafter's scratch):
//! * **per slot** (a sequence's recurrent state, whatever its length): the KDA pools, double buffered (`2 x 4 MiB` of fp32 state per layer, plus
//!   the conv history and the parity word: `kda::Dims::slot_bytes`, 34 layers), and the DSA raw-row rings (`ring_len x 512 B` per MLA layer: the
//!   indexer's `[LayerNorm(k) | gate]` rows of the last `ring_len` tokens of the slot, which is all the pool update ever reads back: the per-token
//!   plane the oracle keeps is 512 B per token and layer and is NOT built);
//! * **per KV page** (256 tokens): the MLA latent (`256 x 512` fp16 per MLA layer; with `--kv-dtype fp8` `256 x 512` E4M3 and 256 fp32
//!   scales) and the DSA pool plane (`64 x 128` fp16 per MLA layer, one entry per 4 tokens): 11,968 B per token over the 11 MLA layers
//!   (6,380 B with fp8);
//! * **per window row**: the KDA window record of every layer (`kda::Dims::record_bytes_per_row`: the raw bf16 qkv row, the fp32 decay and beta;
//!   the conv shift reads the same raw rows, so there is no second copy), held until the window's commit;
//! * **scratch**: every activation and workspace buffer of one forward call, sized for `max_rows` rows (the kernels take at most 128 per call).

use super::config::GlmConfig;
use gb_cuda::exl3_big::big_workspace_bytes;
use gb_cuda::exl3_linear::{linear_workspace_bytes, Dtype};
use gb_cuda::kda::{self, Dims};
use gb_cuda::mhc::MhcWork;
use gb_cuda::mla::{LatentCache, MlaScratch};
use gb_cuda::moe::{MoeDims, MoeScratch, SharedScratch};
use gb_cuda::moe_prefill::{PrefillScratch, SharedScratchBig};
use gb_cuda::{dsa, glm_glue};

/// Most rows in one forward call (the EXL3 decode kernels, the router, the experts, MLA and DSA all take at most 128).
pub const MAX_ROWS: usize = 128;
/// Rows of the prefill chunk a server run uses when `--prefill-chunk` is left alone: 512 rows a chunk is about half a second at the measured
/// speed, about what one 128-row chunk took before the large-row kernels, so a decoding stream waits no longer for its next round.
pub const DEFAULT_PREFILL_CHUNK: usize = 512;
/// The widest prefill chunk the plan prices and the model allocates scratch for (`gb_cuda::moe_prefill::MAX_ROWS` is the kernels' own limit).
pub const MAX_PREFILL_CHUNK: usize = 2048;
/// The `--prefill-chunk` value every family's server default shares.
pub const SERVER_DEFAULT_PREFILL_CHUNK: usize = 4096;

/// The widest prefill chunk under the P2 prefill contract (Lane P, `contract.rs`): the large-row kernels' own limit
/// (`gb_cuda::moe_prefill::MAX_ROWS`, 8,192). P2's prompt rows are chunk-independent, so the width is a speed and memory choice only: a wider
/// chunk reads every routed expert once for more rows, and its scratch (about 0.6 MiB a row) comes out of what the KV pool may hold.
pub const P2_MAX_PREFILL_CHUNK: usize = 8192;

/// The prefill chunk a server run takes from `--prefill-chunk N` under the process's prefill contract: see [`prefill_chunk_for_contract`].
/// Idempotent.
pub fn prefill_chunk_for(flag: usize) -> usize {
    prefill_chunk_for_contract(flag, super::contract::current())
}

/// [`prefill_chunk_for`] under contract `c`. P1: the server's default (4096, shared by every family) means this family's default (512); any
/// other value is what was asked for, clamped to `MAX_ROWS..=MAX_PREFILL_CHUNK` (2,048). P2: what was asked for (the server's default is
/// 4,096 rows), clamped to `MAX_ROWS..=P2_MAX_PREFILL_CHUNK` (a value of `MAX_ROWS` or less is refused by `engine::check_options` before the
/// load: P2's routed experts run in the prefill scratch).
pub fn prefill_chunk_for_contract(flag: usize, c: super::contract::PrefillContract) -> usize {
    use super::contract::PrefillContract;
    match c {
        PrefillContract::P1 if flag == SERVER_DEFAULT_PREFILL_CHUNK => DEFAULT_PREFILL_CHUNK,
        PrefillContract::P1 => flag.clamp(MAX_ROWS, MAX_PREFILL_CHUNK),
        PrefillContract::P2 => flag.clamp(MAX_ROWS, P2_MAX_PREFILL_CHUNK),
    }
}

/// The rows the KDA prefill scratch (`kda_prefill_record`, `kda_workspace`) holds for a window of `rows` and prefill chunks of `chunk_rows`:
/// the window's under P1 (P1's prefill runs 128-row pieces); under the process's P2 contract one piece of the chunked recurrence
/// (`gb_cuda::kda::P2_PIECE_ROWS`, 512) when the chunk reaches it. A model switched to P2 after its build runs smaller pieces with the
/// same bits. The P2 KDA carry is 0 bytes (`docs/GLM_PREFILL_P2.md` 8.1): nothing else is priced for it.
pub fn kda_prefill_rows(rows: usize, chunk_rows: usize) -> usize {
    match super::contract::current() {
        super::contract::PrefillContract::P1 => rows,
        super::contract::PrefillContract::P2 => chunk_rows.min(kda::P2_PIECE_ROWS).max(rows),
    }
}

/// The rows of the pipelined KDA prefill's second buffer set (`kda::Workspace::add_pipe`, priced as `kda_pipe`): under P2, one piece
/// ([`kda_prefill_rows`]) when a chunk spans more than one, so that each piece's projections, prep and conv shift run on a side stream beside
/// the previous piece's recurrence (the same launches on the same operands: no bit moves); else 0. `REXL3_GLM_KDA_PIPE=0` turns it off
/// (the serial order, for timing comparisons).
pub fn kda_pipe_rows(rows: usize, chunk_rows: usize) -> usize {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| !std::env::var("REXL3_GLM_KDA_PIPE").is_ok_and(|v| v.trim() == "0"));
    let on = *ON;
    let piece = kda_prefill_rows(rows, chunk_rows);
    match super::contract::current() {
        super::contract::PrefillContract::P2 if on && chunk_rows > piece => piece,
        _ => 0,
    }
}

/// Bytes of the KDA layers' qkv rows (`kda_qkv`) for windows of `rows` and prefill chunks of `chunk_rows`: under P1 a chunk's fp32 rows of
/// every head (P1's prefill); under the process's P2 contract a window's fp32 rows of every head, or a chunk's bf16 rows of the heads whose
/// state the model holds (`KdaConfig::state_heads`: a KDA-split rank's half), whichever is larger: all a P2 prompt chunk writes there is
/// the P2 GEMM's bf16 qkv (a quarter of P1's bytes on a split rank). A model switched to P1, or to P2's row-by-row arm, after its build
/// grows the buffer then (`GlmModel::set_prefill_contract`, `set_p2_kda_chunked`).
pub fn kda_qkv_bytes(cfg: &GlmConfig, rows: usize, chunk_rows: usize) -> usize {
    let ch = cfg.kda.heads * cfg.kda.head_dim;
    match super::contract::current() {
        super::contract::PrefillContract::P1 => chunk_rows * 3 * ch * 4,
        super::contract::PrefillContract::P2 => {
            (rows * 3 * ch * 4).max(chunk_rows * 3 * cfg.kda.state_heads() * cfg.kda.head_dim * 2)
        }
    }
}

/// The widest chunk the plan prices under the process's contract.
pub fn max_prefill_chunk() -> usize {
    match super::contract::current() {
        super::contract::PrefillContract::P1 => MAX_PREFILL_CHUNK,
        super::contract::PrefillContract::P2 => P2_MAX_PREFILL_CHUNK,
    }
}

/// Rows of a dense layer's gated MLP one piece takes in a large prefill chunk (its intermediate is 12,288 wide: 8 fp32 bytes x 12,288 a row).
pub const PF_DENSE_ROWS: usize = 512;
/// Most rows of one decode or verify window across all sequences (the KDA window records are sized for it).
pub const WINDOW_ROWS: usize = 64;
/// Raw rows the DSA ring keeps per slot and layer (at least a window plus a pool: `dsa::RING_MIN`).
pub const RING_LEN: usize = 256;
/// Most candidates one row reports.
pub const MAX_CANDIDATES: usize = 64;
/// Tokens the tokenizer knows: the model's vocabulary is 154,880 wide, the last 24 ids are padding the sampler must never see.
pub const TOKENIZER_VOCAB: usize = 154_856;
/// Tokens per KV page.
pub const PAGE_TOKENS: usize = 256;
/// Tapped hidden rows (fp16 `[hidden]` each) the NextN drafter keeps per slot between the trunk's windows and its own absorb: the rows of the
/// committed windows whose MTP cache entry is not built yet.
pub const MTP_TAP_CAP: usize = 32;
/// The most drafts one round chains (the K of K-deep MTP drafting).
pub const MTP_MAX_DEPTH: usize = 4;
/// The most MTP forwards of one drafting call: the absorb window and `MTP_MAX_DEPTH - 1` chain steps.
pub const MTP_STEPS: usize = MTP_MAX_DEPTH;

/// The settings the arithmetic is evaluated at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocParams {
    pub slots: usize,
    /// Pages of the KV pool.
    pub kv_blocks: usize,
    pub max_context: usize,
    /// Rows of the widest forward call (at most [`MAX_ROWS`]).
    pub max_rows: usize,
    /// Rows of the widest decode or verify window (at most `max_rows`).
    pub window_rows: usize,
    /// The pack's NextN (MTP) layer drafts: its cache, ring, tap store and scratch are allocated.
    pub mtp: bool,
}

/// One named allocation (or group of equal ones).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Item {
    pub name: &'static str,
    pub bytes: u64,
}

/// The names of the items by kind (see the module docs). A slot's row of the block table (`max_blocks` i32) is per-slot state too.
pub const PER_SLOT: [&str; 4] = ["kda_pools", "dsa_rings", "block_table", "mtp_taps"];
pub const PER_PAGE: [&str; 2] = ["mla_latent", "dsa_pools"];
pub const WINDOW: [&str; 1] = ["kda_window_records"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllocPlan {
    pub items: Vec<Item>,
}

pub(super) fn dims_of(cfg: &GlmConfig) -> Result<Dims, String> {
    let d = Dims::glm();
    if cfg.kda.heads != d.heads
        || cfg.kda.head_dim != kda::HEAD_DIM
        || cfg.hidden != d.hidden
        || cfg.kda.conv_kernel != kda::CONV_TAPS
    {
        return Err(format!(
            "the KDA kernels are built for {} heads of {}, hidden {} and {} conv taps; the config has {} x {}, hidden {}, {} taps",
            d.heads,
            kda::HEAD_DIM,
            d.hidden,
            kda::CONV_TAPS,
            cfg.kda.heads,
            cfg.kda.head_dim,
            cfg.hidden,
            cfg.kda.conv_kernel
        ));
    }
    // a rank of the KDA head split holds the state of its heads only (`KdaConfig::rank`)
    match cfg.kda.rank {
        None => Ok(d),
        Some((h0, h1)) if h0 < h1 && h1 <= d.heads && (h1 - h0) % 8 == 0 => Ok(Dims { heads: h1 - h0, ..d }),
        Some((h0, h1)) => Err(format!("a rank's KDA heads {h0}..{h1} of {} are not a range of whole groups of 8", d.heads)),
    }
}

/// The MoE shapes of the config (the kernels take 288 experts, top 8, hidden 4096, width 2048).
pub fn moe_dims(cfg: &GlmConfig) -> MoeDims {
    MoeDims {
        hidden: cfg.hidden,
        inter: cfg.ffn.moe_intermediate,
        experts: cfg.ffn.routed_experts,
        topk: cfg.ffn.experts_per_token,
        scale: cfg.ffn.routed_scaling_factor as f32,
        limit: cfg.ffn.swiglu_limit as f32,
    }
}

/// The EXL3 linear shapes (K, N) of the text model: the largest decode workspace over them serves every layer on a stream.
pub fn exl3_shapes(cfg: &GlmConfig) -> Vec<(usize, usize)> {
    let h = cfg.hidden;
    let m = &cfg.mla;
    let kda_ch = cfg.kda.heads * cfg.kda.head_dim;
    let mut v = vec![
        (h, 3 * kda_ch),                                           // kda qkv_proj
        (kda_ch, h),                                               // kda o_proj
        (h, m.q_lora_rank),                                        // q_a_proj
        (m.q_lora_rank, m.heads * m.qk_head_dim()),                // q_b_proj
        (h, m.kv_lora_rank + m.qk_rope_head_dim),                  // kv_a_proj_with_mqa
        (m.heads * m.v_head_dim, h),                               // mla o_proj
        (m.q_lora_rank, cfg.indexer.heads * cfg.indexer.head_dim), // indexer wq_b
        (h, cfg.ffn.dense_intermediate),                           // dense gate, up
        (cfg.ffn.dense_intermediate, h),                           // dense down
        (h, cfg.ffn.moe_intermediate * cfg.ffn.shared_experts),    // shared gate, up
        (cfg.ffn.moe_intermediate * cfg.ffn.shared_experts, h),    // shared down
        (h, cfg.vocab),                                            // lm_head
    ];
    v.sort_unstable();
    v.dedup();
    v
}

/// The EXL3 workspace of one stream: the largest of the decode kernels' (a window of `MAX_ROWS` rows) and, for a prefill chunk of more than
/// `MAX_ROWS` rows, of the large-row kernels' (the rotated input of `chunk_rows` rows), over every layer shape.
pub fn exl3_workspace_bytes(cfg: &GlmConfig, chunk_rows: usize) -> usize {
    let shapes = exl3_shapes(cfg);
    let dec = shapes.iter().map(|&(k, n)| linear_workspace_bytes(MAX_ROWS, k, n, 2)).max().unwrap_or(0);
    let big = if chunk_rows > MAX_ROWS {
        shapes.iter().map(|&(k, n)| big_workspace_bytes(chunk_rows, k, n, 2)).max().unwrap_or(0)
    } else {
        0
    };
    dec.max(big)
}

impl AllocPlan {
    /// The whole plan at `p`, for prefill chunks of at most `MAX_ROWS` rows.
    pub fn new(cfg: &GlmConfig, p: &AllocParams) -> Result<Self, String> {
        Self::new_chunked(cfg, p, MAX_ROWS)
    }

    /// The whole plan at `p` for prefill chunks of up to `chunk_rows` rows (`p.max_rows..=gb_cuda::moe_prefill::MAX_ROWS`): the activations
    /// that scale with a chunk's rows are priced at `chunk_rows`, the window-scale scratch (the decode kernels' 128 rows) at `p.max_rows`.
    pub fn new_chunked(cfg: &GlmConfig, p: &AllocParams, chunk_rows: usize) -> Result<Self, String> {
        if chunk_rows < p.max_rows || chunk_rows > gb_cuda::moe_prefill::MAX_ROWS {
            return Err(format!(
                "a prefill chunk takes {}..={} rows, not {chunk_rows}",
                p.max_rows,
                gb_cuda::moe_prefill::MAX_ROWS
            ));
        }
        if p.slots == 0
            || p.kv_blocks == 0
            || p.max_rows == 0
            || p.max_rows > MAX_ROWS
            || p.window_rows == 0
            || p.window_rows > p.max_rows
        {
            return Err(format!("allocation settings out of range: {p:?}"));
        }
        let d = dims_of(cfg)?;
        let (n_kda, n_mla) = (cfg.kda_layers().len() as u64, cfg.mla_layers().len() as u64);
        let n_mla_state = n_mla + u64::from(p.mtp);
        let mut items: Vec<Item> = Self::slot_items(cfg, p.max_context, p.mtp)?
            .into_iter()
            .map(|i| Item { name: i.name, bytes: i.bytes * p.slots as u64 })
            .collect();
        items.extend([
            Item {
                name: "mla_latent",
                bytes: n_mla_state * p.kv_blocks as u64 * LatentCache::block_bytes(Dtype::F16, cfg.latent_kv) as u64,
            },
            Item { name: "dsa_pools", bytes: n_mla_state * dsa::Planes::pool_bytes(p.kv_blocks) as u64 },
            Item { name: "kda_window_records", bytes: n_kda * p.window_rows as u64 * d.record_bytes_per_row() as u64 },
        ]);
        items.extend(Self::scratch_items_chunked(cfg, p.max_rows, chunk_rows, p.max_context, p.mtp)?);
        Ok(Self { items })
    }

    /// The plan with the prefix cache's snapshot pool: `slots` snapshot slots of `prefix::snapshot_bytes` each, named `prefix_snapshots` (the pages the cache
    /// holds are part of `kv_blocks`, which the caller passes in the settings: the pool is `admitted pages + the cache's`). No slots: the plan as it was.
    pub fn with_prefix_snapshots(mut self, cfg: &GlmConfig, slots: usize, mtp: bool) -> Result<Self, String> {
        if slots > 0 {
            let per = super::prefix::snapshot_bytes(cfg, mtp)?;
            self.items.push(Item { name: "prefix_snapshots", bytes: slots as u64 * per });
        }
        Ok(self)
    }

    /// What ONE slot holds, by item ([`PER_SLOT`]): its KDA pools (double-buffered state, conv history, parity word, 34 layers), its DSA
    /// rings (11 layers, a 12th with `mtp`) and its row of the block table; with `mtp` also the store of pending tapped hidden rows.
    pub fn slot_items(cfg: &GlmConfig, max_context: usize, mtp: bool) -> Result<Vec<Item>, String> {
        let d = dims_of(cfg)?;
        let (n_kda, n_mla) = (cfg.kda_layers().len() as u64, cfg.mla_layers().len() as u64 + u64::from(mtp));
        let mut v = vec![
            Item { name: "kda_pools", bytes: n_kda * d.slot_bytes() as u64 },
            Item { name: "dsa_rings", bytes: n_mla * dsa::Planes::ring_bytes(1, RING_LEN) as u64 },
            Item { name: "block_table", bytes: max_context.div_ceil(PAGE_TOKENS) as u64 * 4 },
        ];
        if mtp {
            v.push(Item { name: "mtp_taps", bytes: (MTP_TAP_CAP * cfg.hidden * 2) as u64 });
        }
        Ok(v)
    }

    /// The scratch items alone (they do not depend on the slots' state or the page count): what [`GlmMemory`](super::memory::GlmMemory)
    /// prices a step's activations with.
    pub fn scratch_items(cfg: &GlmConfig, rows: usize, max_context: usize, mtp: bool) -> Result<Vec<Item>, String> {
        Self::scratch_items_chunked(cfg, rows, rows, max_context, mtp)
    }

    /// [`AllocPlan::scratch_items`] for a window scratch of `rows` rows (at most `MAX_ROWS`) and prefill chunks of up to `chunk_rows` rows.
    pub fn scratch_items_chunked(
        cfg: &GlmConfig,
        rows: usize,
        chunk_rows: usize,
        max_context: usize,
        mtp: bool,
    ) -> Result<Vec<Item>, String> {
        if rows == 0 || rows > MAX_ROWS {
            return Err(format!("a forward call takes 1..={MAX_ROWS} rows, not {rows}"));
        }
        if chunk_rows < rows || chunk_rows > gb_cuda::moe_prefill::MAX_ROWS {
            return Err(format!("a prefill chunk takes {rows}..={} rows, not {chunk_rows}", gb_cuda::moe_prefill::MAX_ROWS));
        }
        let d = dims_of(cfg)?;
        let r = rows as u64;
        let cr = chunk_rows as u64;
        let h = cfg.hidden as u64;
        let m = &cfg.mla;
        let ix = &cfg.indexer;
        let moe = moe_dims(cfg);
        let ws = {
            let shapes = exl3_shapes(cfg);
            let dec = shapes.iter().map(|&(k, n)| linear_workspace_bytes(rows, k, n, 2)).max().unwrap_or(0);
            let big = if chunk_rows > MAX_ROWS {
                shapes.iter().map(|&(k, n)| big_workspace_bytes(chunk_rows, k, n, 2)).max().unwrap_or(0)
            } else {
                0
            };
            dec.max(big) as u64
        };
        let mla_acts = (m.q_lora_rank * 2 + m.heads * m.qk_head_dim() + m.kv_lora_rank * 2 + m.heads * m.v_head_dim) as u64 * 2;
        let ix_acts = (2 * ix.head_dim + ix.heads + ix.heads * ix.head_dim) as u64 * 2;
        let ch = (cfg.kda.heads * cfg.kda.head_dim) as u64;
        let sc = |name, bytes: u64| Item { name, bytes };
        let mut v = vec![
            sc("kda_prefill_record", kda_prefill_rows(rows, chunk_rows) as u64 * d.record_bytes_per_row() as u64),
            sc("kda_workspace", kda_prefill_rows(rows, chunk_rows) as u64 * kda::Workspace::bytes_per_row(d) as u64),
            sc("kda_pipe", kda_pipe_rows(rows, chunk_rows) as u64 * kda::Workspace::pipe_bytes_per_row(d) as u64),
            // ids i32 [chunk], the row index (slot, position) i32 each, the sequences' starts [chunk + 1] and slots, the commit's keep counts [window]
            sc("row_arrays", cr * 4 + cr * 8 + (cr + 1) * 4 + cr * 4 + r * 4),
            sc("streams", cr * (gb_cuda::mhc::ROW * 4) as u64),
            sc("mhc_work", MhcWork::bytes(chunk_rows) as u64),
            sc("hidden", cr * h * 2),
            sc("mixer_out", cr * h * 4),
            sc("kda_qkv", kda_qkv_bytes(cfg, rows, chunk_rows) as u64),
            sc("kda_out", cr * ch * 2),
            sc("mla_acts", cr * mla_acts),
            sc("indexer_acts", cr * ix_acts),
            sc("mla_scratch", MlaScratch::bytes(rows, max_context, Dtype::F16) as u64),
            sc("dsa_scratch", dsa::Scratch::bytes_for(max_context) as u64),
            sc("moe_scratch", MoeScratch::bytes_for(&moe, rows).map_err(|e| e.to_string())? as u64),
            sc(
                "shared_scratch",
                SharedScratch::bytes_for(cfg.hidden, cfg.ffn.moe_intermediate * cfg.ffn.shared_experts, rows) as u64,
            ),
            sc("dense_scratch", SharedScratch::bytes_for(cfg.hidden, cfg.ffn.dense_intermediate, rows) as u64),
            sc("moe_shared_out", cr * h * 4),
            sc("exl3_workspace", ws),
            // the two side streams' (`model/graphs.rs`): `kv_a` and the indexer's `wq_b`
            sc(
                "exl3_workspace_side",
                (linear_workspace_bytes(rows, cfg.hidden, m.kv_lora_rank + m.qk_rope_head_dim, 2)
                    + linear_workspace_bytes(rows, m.q_lora_rank, ix.heads * ix.head_dim, 2)) as u64,
            ),
            sc("logits", r * cfg.vocab as u64 * 2),
            sc(
                "topk",
                r * MAX_CANDIDATES as u64 * 4 * 2 + glm_glue::topk_workspace_bytes(rows, TOKENIZER_VOCAB, MAX_CANDIDATES) as u64,
            ),
        ];
        if chunk_rows > MAX_ROWS {
            // the large-row path of a chunk: the routed experts' sorted pairs and sums, the gated MLPs' pieces
            v.extend([
                sc("moe_prefill_scratch", PrefillScratch::bytes_for(&moe, chunk_rows).map_err(|e| e.to_string())? as u64),
                sc(
                    "shared_prefill_scratch",
                    SharedScratchBig::bytes_for(cfg.hidden, cfg.ffn.moe_intermediate * cfg.ffn.shared_experts, chunk_rows) as u64,
                ),
                sc(
                    "dense_prefill_scratch",
                    SharedScratchBig::bytes_for(cfg.hidden, cfg.ffn.dense_intermediate, chunk_rows.min(PF_DENSE_ROWS)) as u64,
                ),
            ]);
        }
        if mtp {
            // the NextN drafter's buffers (`model/mtp.rs`): the trunk's tapped rows, the layer's fp32 residual, the fused input, the gathered
            // hidden rows, the block's output, the last rows of an absorb, the row-index arrays of each step, and each step's draft ids and values
            v.extend([
                sc("mtp_tap", cr * h * 2),
                sc("mtp_x", cr * h * 4),
                sc("mtp_cat", cr * 2 * h * 2),
                sc("mtp_h_in", cr * h * 2),
                sc("mtp_out", cr * h * 2),
                sc("mtp_lm_in", r * h * 2),
                // the ids of a chunk, the chunk's (slot, position) index (step 0), and the other steps' window-sized ones
                sc("mtp_rows", cr * 4 + cr * 8 + (MTP_STEPS as u64 - 1) * r * 8),
                // each depth's argmax ids and values, its keyed pick's candidate lists (ids and values of up to `MAX_CANDIDATES` a row), and the
                // per-row sampling records of the keyed pick (`glm_mtp::KeyedRow`, 48 bytes)
                sc("mtp_drafts", MTP_MAX_DEPTH as u64 * r * 8 + MTP_MAX_DEPTH as u64 * r * MAX_CANDIDATES as u64 * 8 + r * 48),
            ]);
        }
        Ok(v)
    }

    pub fn get(&self, name: &str) -> u64 {
        self.items.iter().find(|i| i.name == name).map_or(0, |i| i.bytes)
    }

    pub fn total(&self) -> u64 {
        self.items.iter().map(|i| i.bytes).sum()
    }

    pub fn sum_of(&self, names: &[&str]) -> u64 {
        names.iter().map(|n| self.get(n)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GLM_CONFIG: &str = include_str!("../../../../../testdata/glm53/config.json");

    fn real() -> GlmConfig {
        GlmConfig::from_json(&serde_json::from_str(GLM_CONFIG).unwrap()).unwrap()
    }

    /// The vocabulary mask is the tokenizer's size, taken from sources that are not this constant: the oracle's manifest and the server's
    /// tokenizer fixture both say 154,856 tokens, and the config's 154,880 is the model's (24 padding ids no tokenizer can produce).
    #[test]
    fn the_vocabulary_mask_is_the_tokenizers_not_the_models() {
        const MANIFEST: &str = include_str!("../../../../../testdata/glm53/ref_manifest.json");
        const TOKENIZER_CASES: &str = include_str!("../../../../server/tests/fixtures/glm53_tokenizer_cases.json");
        let m: serde_json::Value = serde_json::from_str(MANIFEST).unwrap();
        let t: serde_json::Value = serde_json::from_str(TOKENIZER_CASES).unwrap();
        assert_eq!(m["tokenizer_vocab"].as_u64(), Some(TOKENIZER_VOCAB as u64));
        assert_eq!(t["meta"]["vocab_size"].as_u64(), Some(TOKENIZER_VOCAB as u64));
        let c = real();
        assert_eq!(m["model_vocab"].as_u64(), Some(c.vocab as u64));
        assert_eq!(c.vocab - TOKENIZER_VOCAB, 24);
        // the model's own special ids all lie below the mask
        assert!(c.eos.iter().all(|&e| (e as usize) < TOKENIZER_VOCAB), "{:?}", c.eos);
    }

    #[test]
    fn the_state_and_page_numbers_by_hand() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let c = real();
        let p = AllocParams { slots: 8, kv_blocks: 1032, max_context: 262_144, max_rows: 128, window_rows: 64, mtp: false };
        let a = AllocPlan::new(&c, &p).unwrap();
        // one KDA layer of one slot: two fp32 state buffers of 64 x 128 x 128, the conv history of 3 raw rows x 24,576 bf16, the parity word
        let layer_slot = 2 * 4 * 64 * 128 * 128 + 3 * 24_576 * 2 + 4;
        assert_eq!(layer_slot, 8_536_068);
        assert_eq!(a.get("kda_pools"), 34 * 8 * layer_slot as u64);
        // a ring: 256 raw rows of [LayerNorm(k) | gate] = 256 x 128 fp16 each, 512 B a row: 131,072 B a slot and layer, 11 layers
        assert_eq!(a.get("dsa_rings"), 11 * 8 * 256 * 512);
        // a page of one MLA layer: 256 tokens x 512 fp16 of latent, and one 128-wide fp16 pooled key per 4 tokens
        assert_eq!(a.get("mla_latent"), 11 * 1032 * 256 * 512 * 2);
        assert_eq!(a.get("dsa_pools"), 11 * 1032 * 64 * 128 * 2);
        // 11,968 B a token over the 11 layers: (1,024 + 64) x 11
        assert_eq!((a.get("mla_latent") + a.get("dsa_pools")) / (1032 * 256), 11 * (1024 + 64));
        // the window record: raw qkv row bf16 49,152 + decay fp32 32,768 + beta fp32 256 = 82,176 a layer and row
        assert_eq!(a.get("kda_window_records"), 34 * 64 * 82_176);
        // names are unique
        let mut names: Vec<&str> = a.items.iter().map(|i| i.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), a.items.len());
    }

    #[test]
    fn scratch_grows_with_rows_and_the_big_items_are_the_ones_expected() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let c = real();
        let at = |rows| AllocPlan::scratch_items(&c, rows, 262_144, false).unwrap().iter().map(|i| i.bytes).sum::<u64>();
        assert!(at(1) < at(9) && at(9) < at(128));
        let a = AllocPlan::scratch_items(&c, 128, 262_144, false).unwrap();
        let by = |n: &str| a.iter().find(|i| i.name == n).unwrap().bytes;
        // the logits row: 154,880 fp16; the DSA scores: 128 rows x 65,536 pools x 2 B (the doc's "18 MB at 262,144 tokens" with the selection)
        assert_eq!(by("logits"), 128 * 154_880 * 2);
        assert_eq!(by("dsa_scratch"), 128 * 256 * 2 + 128 * 65_536 * 2 + 128 * 2056 * 4 + 128 * 8);
        // everything but the KV pool and the slots' state stays well under the 5 GiB engine blanket
        assert!(at(128) < 2 << 30, "scratch at 128 rows: {} MiB", at(128) >> 20);
    }

    #[test]
    fn the_nextn_drafter_adds_a_twelfth_attention_layer_a_tap_store_and_its_scratch() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let c = real();
        let off = AllocParams { slots: 8, kv_blocks: 1032, max_context: 262_144, max_rows: 128, window_rows: 64, mtp: false };
        let on = AllocParams { mtp: true, ..off };
        let (a, b) = (AllocPlan::new(&c, &off).unwrap(), AllocPlan::new(&c, &on).unwrap());
        // the pages: one more layer of latent (256 x 512 fp16) and pooled keys (64 x 128 fp16) per page, the rings one more per slot
        assert_eq!(b.get("mla_latent"), 12 * 1032 * 256 * 512 * 2);
        assert_eq!(b.get("dsa_pools"), 12 * 1032 * 64 * 128 * 2);
        assert_eq!(b.get("dsa_rings"), 12 * 8 * 256 * 512);
        // the tap store: MTP_TAP_CAP fp16 rows of 4,096 per slot
        assert_eq!(b.get("mtp_taps"), 8 * 32 * 4096 * 2);
        assert_eq!(a.get("mtp_taps"), 0, "nothing of the drafter is allocated without it");
        for name in ["mtp_tap", "mtp_x", "mtp_cat", "mtp_h_in", "mtp_out", "mtp_lm_in", "mtp_rows", "mtp_drafts"] {
            assert_eq!(a.get(name), 0, "{name}");
            assert!(b.get(name) > 0, "{name}");
        }
        // everything else is unchanged
        for name in ["kda_pools", "block_table", "kda_window_records", "logits", "moe_scratch", "mla_scratch"] {
            assert_eq!(a.get(name), b.get(name), "{name}");
        }
        // the drafter's scratch at 128 rows: 1 + 2 + 2 + 1 + 1 + 1 MiB of buffers plus the index arrays, a few MiB in all
        let extra: u64 = b.items.iter().filter(|i| i.name.starts_with("mtp_") && i.name != "mtp_taps").map(|i| i.bytes).sum();
        // ... and the draft buffers: each depth's argmax id and value (8 B a row), its keyed pick's candidate lists (64 ids and 64 values a row) and the 48-byte
        // sampling record of each row of a call
        let drafts = 4 * 128 * 8 + 4 * 128 * 64 * 8 + 128 * 48;
        assert_eq!(b.get("mtp_drafts"), drafts);
        assert_eq!(extra, 128 * 4096 * (2 + 4 + 4 + 2 + 2 + 2) + 128 * 4 * (1 + 2 * 4) + drafts);
        assert!(extra < 12 << 20, "{extra}");
    }

    #[test]
    fn the_chunk_scratch_is_priced_by_name_and_a_window_sized_chunk_adds_nothing() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        use gb_cuda::moe_prefill::PLAN_CHUNK;
        let c = real();
        // a chunk of the window's rows is the old plan, item for item
        for rows in [1usize, 9, 64, 128] {
            assert_eq!(
                AllocPlan::scratch_items(&c, rows, 262_144, true).unwrap(),
                AllocPlan::scratch_items_chunked(&c, rows, rows, 262_144, true).unwrap(),
                "{rows}"
            );
        }
        let base = AllocPlan::scratch_items(&c, 128, 262_144, false).unwrap();
        let at = |chunk| AllocPlan::scratch_items_chunked(&c, 128, chunk, 262_144, false).unwrap();
        let get = |v: &[Item], n: &str| v.iter().find(|i| i.name == n).map_or(0, |i| i.bytes);
        let big = at(1024);
        // the large-row path's own items appear above 128 rows and only then
        for n in ["moe_prefill_scratch", "shared_prefill_scratch", "dense_prefill_scratch"] {
            assert_eq!(get(&base, n), 0, "{n}");
            assert!(get(&big, n) > 0, "{n}");
        }
        // the items that scale with a chunk's rows scale by them; the window-scale ones do not move
        for n in ["streams", "hidden", "mixer_out", "kda_out", "mla_acts", "indexer_acts", "moe_shared_out", "mhc_work"] {
            assert_eq!(get(&big, n), 8 * get(&base, n), "{n}: 1,024 rows are 8 x 128");
        }
        // the KDA qkv rows: P1's fp32 rows of the chunk; under P2 the chunk's bf16 rows (every head on one node), 4 x the window's fp32 rows
        let qkv_scale = match crate::families::glm::contract::current() {
            crate::families::glm::contract::PrefillContract::P1 => 8,
            crate::families::glm::contract::PrefillContract::P2 => 4,
        };
        assert_eq!(get(&big, "kda_qkv"), qkv_scale * get(&base, "kda_qkv"), "kda_qkv");
        let split = c.with_kda_rank(0, 2).unwrap();
        let big_split = AllocPlan::scratch_items_chunked(&split, 128, 1024, 262_144, false).unwrap();
        // a KDA-split rank's P2 rows are its heads' (half); P1's are every head's
        let split_div = if qkv_scale == 4 { 2 } else { 1 };
        assert_eq!(get(&big_split, "kda_qkv"), get(&big, "kda_qkv") / split_div, "kda_qkv of a KDA-split rank");
        for n in ["logits", "topk", "moe_scratch", "shared_scratch", "dense_scratch", "mla_scratch", "dsa_scratch"] {
            assert_eq!(get(&big, n), get(&base, n), "{n} is the window's");
        }
        // the KDA prefill scratch (`kda_prefill_rows`): the window's under P1; under P2 (the default contract) one 512-row piece of the chunked
        // recurrence, 4 x the 128-row window's, for any chunk that reaches it
        let kda_scale = match crate::families::glm::contract::current() {
            crate::families::glm::contract::PrefillContract::P1 => 1,
            crate::families::glm::contract::PrefillContract::P2 => 4,
        };
        for n in ["kda_workspace", "kda_prefill_record"] {
            assert_eq!(get(&big, n), kda_scale * get(&base, n), "{n}: the window's under P1, a 512-row piece under P2");
        }
        // the pipelined KDA prefill's second buffer set: one piece, only for a P2 chunk of several pieces
        assert_eq!(get(&base, "kda_pipe"), 0);
        let pipe = kda_pipe_rows(128, 1024) as u64 * kda::Workspace::pipe_bytes_per_row(dims_of(&c).unwrap()) as u64;
        assert_eq!(get(&big, "kda_pipe"), pipe);
        // the routed experts' scratch at 2,048 rows by hand: sorted pairs padded to 16-row tiles (16,384 + 15 x 288), fp16 rotated inputs of both
        // matrices and the down input, fp32 y of every pair, the picks, the logits, the sorted-row to pair map and the work items, then the
        // pair planner's buffers (fp32 weights and local picks per pair, and per 256-pair block a histogram over the experts, plus the
        // experts' offsets, counts and the three class lists)
        let s = (16_384 + 15 * 288usize).div_ceil(16) * 16;
        let cap = (16_384 + 15 * 288) / 16 / 4 + 288 + 1;
        let by_hand = 2048 * 288 * 2
            + 16_384 * 4
            + 16_384 * 2
            + 16_384 * 4
            + 16_384 * 4
            + s * 4
            + 3 * cap * 16
            + 16
            + 16_384usize.div_ceil(PLAN_CHUNK) * 288 * 4
            + 288 * 4
            + 288 * 4
            + 3 * 288 * 4
            + 2 * s * 4096 * 2
            + s * 2048 * 2
            + 16_384 * 4096 * 4;
        assert_eq!(get(&at(2048), "moe_prefill_scratch"), by_hand as u64);
        // wider chunks cost more, and the whole chunk scratch at 2,048 rows stays under 3 GiB
        let total = |v: &[Item]| v.iter().map(|i| i.bytes).sum::<u64>();
        // (the figures docs/GLM53.md quotes: `cargo test -p gb-model --lib the_chunk_scratch -- --nocapture`)
        for chunk in [128usize, 256, 512, 1024, 2048, 4096, 6144, 8192] {
            let v = at(chunk);
            eprintln!(
                "scratch at a {chunk}-row chunk: total {:.1} MiB (routed experts {:.1}, shared expert {:.1}, dense {:.1}, exl3 workspace {:.1}, mhc {:.1}, activations {:.1})",
                total(&v) as f64 / 1048576.0,
                get(&v, "moe_prefill_scratch") as f64 / 1048576.0,
                get(&v, "shared_prefill_scratch") as f64 / 1048576.0,
                get(&v, "dense_prefill_scratch") as f64 / 1048576.0,
                get(&v, "exl3_workspace") as f64 / 1048576.0,
                get(&v, "mhc_work") as f64 / 1048576.0,
                ["streams", "hidden", "mixer_out", "kda_qkv", "kda_out", "mla_acts", "indexer_acts", "moe_shared_out"]
                    .iter()
                    .map(|n| get(&v, n))
                    .sum::<u64>() as f64
                    / 1048576.0
            );
        }
        assert!(total(&base) < total(&at(512)) && total(&at(512)) < total(&big) && total(&big) < total(&at(2048)));
        assert!(total(&at(2048)) < 3 << 30, "{} MiB", total(&at(2048)) >> 20);
        assert!(total(&at(4096)) < total(&at(6144)) && total(&at(6144)) < total(&at(8192)));
        // refusals
        assert!(AllocPlan::scratch_items_chunked(&c, 128, 127, 262_144, false).is_err());
        assert!(AllocPlan::scratch_items_chunked(&c, 128, 8193, 262_144, false).is_err());
        // the family's reading of --prefill-chunk, per contract (P2 is the process default since 2026-10-19, `=p1` selects P1): the process's
        // reading is its contract's
        use crate::families::glm::contract::PrefillContract::{P1, P2};
        assert_eq!(prefill_chunk_for_contract(4096, P1), DEFAULT_PREFILL_CHUNK);
        assert_eq!(prefill_chunk_for_contract(1, P1), 128);
        assert_eq!(prefill_chunk_for_contract(1024, P1), 1024);
        assert_eq!(prefill_chunk_for_contract(3000, P1), MAX_PREFILL_CHUNK);
        for f in [1usize, 777, 1024, 3000, 4096, 9000] {
            assert_eq!(prefill_chunk_for(f), prefill_chunk_for_contract(f, crate::families::glm::contract::current()), "{f}");
        }
        assert_eq!(prefill_chunk_for(prefill_chunk_for(777)), prefill_chunk_for(777));
        // Lane P: under P2 the server's default is a 4,096-row chunk and the clamp reaches 8,192; P1's figures above are unchanged
        assert_eq!(prefill_chunk_for_contract(4096, P1), DEFAULT_PREFILL_CHUNK);
        assert_eq!(prefill_chunk_for_contract(3000, P1), MAX_PREFILL_CHUNK);
        assert_eq!(prefill_chunk_for_contract(4096, P2), 4096);
        assert_eq!(prefill_chunk_for_contract(3000, P2), 3000);
        assert_eq!(prefill_chunk_for_contract(6144, P2), 6144);
        assert_eq!(prefill_chunk_for_contract(8192, P2), P2_MAX_PREFILL_CHUNK);
        assert_eq!(prefill_chunk_for_contract(9000, P2), P2_MAX_PREFILL_CHUNK);
        assert_eq!(prefill_chunk_for_contract(2048, P2), 2048);
        assert_eq!(prefill_chunk_for_contract(1, P2), MAX_ROWS);
        assert_eq!(P2_MAX_PREFILL_CHUNK, gb_cuda::moe_prefill::MAX_ROWS);
    }

    #[test]
    fn bad_settings_are_refused() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let c = real();
        let ok = AllocParams { slots: 1, kv_blocks: 1, max_context: 4096, max_rows: 128, window_rows: 64, mtp: false };
        assert!(AllocPlan::new(&c, &ok).is_ok());
        for bad in [
            AllocParams { slots: 0, ..ok },
            AllocParams { kv_blocks: 0, ..ok },
            AllocParams { max_rows: 129, ..ok },
            AllocParams { window_rows: 129, ..ok },
            AllocParams { max_rows: 0, ..ok },
        ] {
            assert!(AllocPlan::new(&c, &bad).is_err(), "{bad:?}");
        }
    }
}
