//! The Qwen3.8-Flash-Next memory model: what a token, a sequence slot, a window row and a step's rows cost, evaluated at the
//! serving settings. The arithmetic mirrors what the Flash-Next engine allocates; the backend's slice adds the GPU test that
//! checks this model against its real allocations, as GLM's does (`families/glm/alloc.rs`).

use super::config::FlashNextConfig;
use crate::memory::MemoryOptions;
use gb_sched::{KvLayerClass, KvLayout, KvPlane, MemoryError, MemoryModel, MemoryProfile, StepShape, VisionMemory};

/// Tokens of a KV page (the paged cache the attention, the QSA pool plane and the MTP layer share).
pub const PAGE_TOKENS: u64 = 256;
/// Rows of the widest verify window the engine allocates for.
pub const WINDOW_ROWS: usize = 64;
/// Positions of conv input the PLE stage keeps per slot (a window plus the conv's reach of 9 positions).
pub const PLE_RING: u64 = 128;
/// Raw indexer keys kept per slot and attention layer (a window plus the 3 earlier keys a block may need).
pub const QSA_RING: u64 = 128;
/// Rows a prefill chunk scores and selects at once (the score scratch is this many rows of the whole context).
pub const QSA_SLAB: u64 = 256;
/// Partial sums of a window row's sparse attention.
pub const QSA_WINDOW_PARTS: u64 = 32;
/// Candidates per row the engine keeps from the LM head.
pub const MAX_CANDIDATES: u64 = 64;
/// What the engine holds back for the workspaces it allocates after the plan: the EXL3 prefill and decode scratch, the dense
/// attention decode partials (774 MiB at a 64-row window and 262,144 tokens), the QSA score slab and attention partials, the GDN
/// prefill workspace and the MoE workspaces. The tests check the modelled part stays inside it.
pub const ACTIVATION_RESERVE: u64 = 5 << 30;

pub struct FlashNextMemory<'a> {
    pub cfg: &'a FlashNextConfig,
    pub opts: MemoryOptions,
}

impl FlashNextMemory<'_> {
    /// Attention layers whose K, V and QSA pool share the paged cache: the trunk's, and the MTP layer's when it drafts.
    fn attention_layers(&self) -> u64 {
        (self.cfg.full_layers() + if self.opts.mtp { self.cfg.mtp_layers } else { 0 }) as u64
    }

    /// Per-row bytes of the activation buffers one step's rows use (one set, reused by every layer). Named in the order the
    /// engine allocates them.
    pub fn scratch_bytes_per_row(&self) -> u64 {
        let c = self.cfg;
        let (h, hc, k1) = (c.hidden as u64, c.hc_streams as u64, c.experts_per_tok as u64 + 1);
        let (q_dim, kv_dim, inter) = ((c.heads * c.head_dim) as u64, (c.kv_heads * c.head_dim) as u64, c.moe_intermediate as u64);
        let (z, qkv) = ((c.lin_value_heads * c.lin_value_dim) as u64, (2 * q_dim).max(c.conv_channels() as u64));
        let streams = hc * h * 4 // residual streams, fp32
            + hc * h * 4 // their normed copy (the reference pre-mix)
            + 2 * hc * 4 // attention and MLP inject weights
            + c.hc_lowrank as u64 * 4; // low-rank mix
        let mixers = 3 * h * 2 // mixed input, block output, MoE output
            + q_dim.max(h) * 2 // mixer output (attention heads or GDN values)
            + qkv * 2 // q projection with its gate, or GDN qkv
            + z * 2 // GDN z
            + 2 * c.lin_value_heads as u64 * 2 // GDN a and b
            + 3 * kv_dim * 2 // k, v and the normed, roped k
            + 2 * q_dim * 2; // normed, roped q and the gate
        let moe = c.experts as u64 * 2 // router logits
            + 2 * k1 * 4 // picks and weights (routed + shared)
            + k1 * inter * 2 // expert activations
            + 3 * inter * 2 + h * 2 + 2; // shared expert gate, up, product, output and its gate scalar
        let ple = (c.ngram_size as u64 - 1) * c.heads_per_ngram as u64 * ple_words_per_row() * 2 // gathered table rows
            + c.ple_embed_dim as u64 * 2 // n-gram embedding
            + hc * h * 2 // key projection
            + h * 2 // value projection
            + hc * 4 // stream gates
            + hc * h * 2; // conv input
        let qsa = ((c.indexer_heads + c.indexer_kv_heads) * c.indexer_head_dim) as u64 * 2 // indexer projection
            + (c.indexer_heads * c.indexer_head_dim) as u64 * 2 // indexer queries
            + c.indexer_head_dim as u64 * 2 // raw keys
            + topk_blocks(c) * 4 + 4; // selections and their counts
        let index = 4 * 4; // ids, positions, row slot and row position (i32)
        let mtp = if self.opts.mtp {
            hc * h * 4 + hc * h * 2 + h * 2 + hc * h * 2 + h * 2 // the head's stream stack, its inputs and projections
        } else {
            0
        };
        streams + mixers + moe + ple + qsa + index + mtp
    }

    /// Buffers that do not scale with the step's rows: the window's logits (bf16) and top-k candidates, the per-slot sequence
    /// arrays, and the QSA score slab (the widest of a window and a prefill slab, over the whole context).
    pub fn fixed_bytes(&self) -> u64 {
        let w = WINDOW_ROWS as u64;
        let slots_arrays = 3 * 4 * 64; // sequence starts, slots and keep counts for up to 64 sequences
        let nb = (self.opts.max_context as u64).div_ceil(self.cfg.indexer_compress_ratio as u64);
        w * self.cfg.vocab as u64 * 2 + 2 * w * MAX_CANDIDATES * 4 + slots_arrays + QSA_SLAB.max(w) * nb * 4
    }
}

/// u16 words of one packed n-gram table row (an fp16 scale and 160 K-bit codes; the shipped table is K = 5).
fn ple_words_per_row() -> u64 {
    1 + 160 * 5 / 16
}

fn topk_blocks(c: &FlashNextConfig) -> u64 {
    (c.indexer_budget / c.indexer_compress_ratio) as u64
}

impl MemoryModel for FlashNextMemory<'_> {
    fn family(&self) -> &str {
        "qwen_flash_next"
    }
    /// bf16 K and V of 2 heads x 256 and the QSA pooled key (128 fp16 per 4 tokens) on every attention layer, the MTP layer's
    /// in the same pages when it drafts.
    fn kv_layout(&self) -> KvLayout {
        let c = self.cfg;
        let kv = (c.kv_heads * c.head_dim) as u64;
        KvLayout::new().with(KvLayerClass {
            name: "gqa_qsa",
            layers: self.attention_layers(),
            planes: vec![
                KvPlane::per_token("k", kv, 2),
                KvPlane::per_token("v", kv, 2),
                KvPlane::pooled("qsa_pool", c.indexer_head_dim as u64, c.indexer_compress_ratio as u64, 2),
            ],
        })
    }
    fn recurrent_layers(&self) -> u64 {
        self.cfg.linear_layers() as u64
    }
    /// Per GDN layer the fp32 state (48 x 128 x 128) and the bf16 conv history (3 x 10,240); then the slot's PLE conv ring
    /// (128 positions x 4 streams x 2,560, fp16) and a raw indexer key ring per attention layer (128 x 128, fp16).
    fn state_bytes_per_seq(&self) -> u64 {
        let c = self.cfg;
        let gdn =
            (c.lin_value_heads * c.lin_value_dim * c.lin_key_dim * 4 + (c.lin_conv_kernel - 1) * c.conv_channels() * 2) as u64;
        let ple = PLE_RING * (c.hc_streams * c.hidden) as u64 * 2;
        let qsa = self.attention_layers() * QSA_RING * c.indexer_head_dim as u64 * 2;
        c.linear_layers() as u64 * gdn + ple + qsa
    }
    /// Per GDN layer and row: two records (q, k at the key heads, v at the value heads, bf16; decay and beta fp32) and the raw
    /// qkv row the conv history is rebuilt from (bf16). The records exist twice: the window that follows a commit replays the
    /// previous window's set (its accepted rows, in the window kernel) while it fills the other.
    fn window_record_bytes_per_row(&self) -> u64 {
        let c = self.cfg;
        let record =
            2 * c.lin_key_heads * c.lin_key_dim * 2 + c.lin_value_heads * c.lin_value_dim * 2 + 2 * c.lin_value_heads * 4;
        (c.linear_layers() * (2 * record + c.conv_channels() * 2)) as u64
    }
    fn scratch_bytes(&self, rows: u64) -> u64 {
        rows.saturating_mul(self.scratch_bytes_per_row()).saturating_add(self.fixed_bytes())
    }
    fn activation_reserve(&self) -> u64 {
        ACTIVATION_RESERVE
    }
    /// The MTP head keeps nothing per slot outside the paged cache (its K, V and pool are planes of it).
    fn drafter_state_bytes_per_slot(&self) -> u64 {
        0
    }
    /// The vision tower is not built: nothing is reserved, and the server refuses images by naming it (`FamilyPlan::unbuilt_vision`).
    fn vision(&self) -> Option<VisionMemory> {
        None
    }
}

/// The validated profile of `cfg` at `opts`.
pub fn memory_profile(cfg: &FlashNextConfig, opts: &MemoryOptions) -> Result<MemoryProfile, MemoryError> {
    let shape = StepShape {
        page_tokens: PAGE_TOKENS,
        window_rows: opts.window_rows as u64,
        scratch_rows: (opts.prefill_rows as u64).max(WINDOW_ROWS as u64),
    };
    MemoryProfile::new(&FlashNextMemory { cfg, opts: *opts }, shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::qwen_flash_next::config::tests::FLASH_NEXT_CONFIG;

    fn real() -> FlashNextConfig {
        FlashNextConfig::from_json(&serde_json::from_str(FLASH_NEXT_CONFIG).unwrap()).unwrap()
    }

    fn opts() -> MemoryOptions {
        MemoryOptions { mtp: true, vision: None, ..MemoryOptions::serving_defaults() }
    }

    #[test]
    fn the_cache_is_kv_plus_the_qsa_pool_on_every_attention_layer() {
        let c = real();
        let p = memory_profile(&c, &opts()).unwrap();
        // per layer and token: K and V 2 x 2 x 256 x 2 B = 2,048, the pool 128 x 2 B / 4 tokens = 64; 12 trunk layers + the MTP layer
        assert_eq!(p.kv_bytes_per_token(), 13 * (2048 + 64));
        assert_eq!(p.kv_page_bytes(), 13 * (2048 + 64) * 256);
        // without drafting the MTP layer's planes are not charged
        let p = memory_profile(&c, &MemoryOptions { mtp: false, ..opts() }).unwrap();
        assert_eq!(p.kv_bytes_per_token(), 12 * 2112);
    }

    #[test]
    fn state_records_and_scratch_by_hand() {
        let c = real();
        let p = memory_profile(&c, &opts()).unwrap();
        // 36 GDN layers x (48 x 128 x 128 x 4 + 3 x 10,240 x 2) + PLE ring 128 x 10,240 x 2 + 13 QSA rings of 128 x 128 x 2
        assert_eq!(p.state_bytes_per_seq(), 36 * (3_145_728 + 61_440) + 2_621_440 + 13 * 32_768);
        assert_eq!(p.recurrent_layers(), 36);
        // per GDN layer and row: q, k 2 x 4,096 + v 12,288 + decay, beta 2 x 192 = 20,864 twice + raw qkv 20,480 = 62,208
        assert_eq!(p.window_record_bytes_per_row(), 36 * 62_208);
        let m = FlashNextMemory { cfg: &c, opts: opts() };
        // per row: streams 40,960 + 40,960 + 32 + 1,280; mixers 15,360 + 12,288 + 24,576 + 12,288 + 192 + 3,072 + 24,576;
        // MoE 1,024 + 88 + 14,080 + 3,840 + 5,120 + 2; PLE 1,632 + 5,120 + 20,480 + 5,120 + 16 + 20,480;
        // QSA 1,280 + 1,024 + 256 + 2,052; index 16; MTP 40,960 + 20,480 + 5,120 + 20,480 + 5,120
        let per_row = (40_960 + 40_960 + 32 + 1_280)
            + (15_360 + 12_288 + 24_576 + 12_288 + 192 + 3_072 + 24_576)
            + (1_024 + 88 + 14_080 + 3_840 + 5_120 + 2)
            + (1_632 + 5_120 + 20_480 + 5_120 + 16 + 20_480)
            + (1_280 + 1_024 + 256 + 2_052)
            + 16
            + (40_960 + 20_480 + 5_120 + 20_480 + 5_120);
        assert_eq!(m.scratch_bytes_per_row(), per_row);
        // a 4,096-row prefill chunk at 262,144 tokens: rows x per row + 64-row logits and candidates + sequence arrays + the score
        // slab (256 rows x 65,536 blocks x 4 B = 64 MiB)
        let o = MemoryOptions { prefill_rows: 4096, max_context: 262_144, ..opts() };
        let p = memory_profile(&c, &o).unwrap();
        assert_eq!(p.scratch_bytes(), 4096 * per_row + 64 * 248_320 * 2 + 2 * 64 * 64 * 4 + 768 + 256 * 65_536 * 4);
    }

    #[test]
    fn the_modelled_workspaces_fit_the_reserve_at_the_widest_settings() {
        let c = real();
        // the 8,192-row prefill chunk at the full 262,144-token context: the scratch, the window records and the two attention
        // workspaces the engine sizes for it (dense decode partials for 64 rows, and the QSA partials of the chunk in one part)
        let o = MemoryOptions { prefill_rows: 8192, window_rows: 64, max_context: 262_144, ..opts() };
        let p = memory_profile(&c, &o).unwrap();
        let dense_decode = 64u64 * 24 * (262_144 / 512) * (256 * 4 + 8);
        let qsa_parts = 8192u64 * 24 * (256 * 4 + 8);
        let modelled = p.scratch_bytes() + p.window_record_bytes() + dense_decode + qsa_parts;
        assert!(
            modelled < p.engine_reserve_bytes(),
            "{} MiB modelled against {} MiB",
            modelled >> 20,
            p.engine_reserve_bytes() >> 20
        );
        assert_eq!(p.engine_reserve_bytes(), ACTIVATION_RESERVE);
    }

    #[test]
    fn a_zero_kv_shape_cannot_reach_admission() {
        let mut v: serde_json::Value = serde_json::from_str(FLASH_NEXT_CONFIG).unwrap();
        v["text_config"]["head_dim"] = serde_json::json!(0);
        let c = FlashNextConfig::from_json(&v).unwrap();
        assert!(c.check_shapes().is_err(), "the shape check refuses it first");
        // and the memory model refuses it on its own as well
        let e = memory_profile(&c, &opts()).unwrap_err();
        assert!(matches!(e, MemoryError::Zero(_)), "{e}");
    }
}
