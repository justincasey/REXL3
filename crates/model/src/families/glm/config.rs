//! GLM-5.3-Flash (`Glm5NextForConditionalGeneration`, model type `glm5_next`): configuration read from a pack's `config.json`
//! (and `processor_config.json` beside it), with strict validation.
//!
//! The text model is 45 layers. Thirty-four mix tokens with Kimi delta attention (KDA, a linear recurrence); eleven (layers
//! 3, 7, ..., 43) with multi-head latent attention without RoPE, whose keys are selected by the DeepSeek sparse-attention indexer.
//! Layers 0-2 use a dense feed-forward block, layers 3-44 a mixture of 288 routed experts (top 8) plus one shared expert behind a
//! sigmoid router. Every trunk layer joins its residual through mHC hyper-connections (4 streams). One more layer, index 45, is the
//! NextN/MTP head: a full MLA + indexer + MoE layer on a plain residual that predicts the token after next.
//!
//! `head_dim` in this config is 0 and nothing reads it: MLA is described by `qk_nope_head_dim`, `qk_rope_head_dim`,
//! `v_head_dim` and `kv_lora_rank`. This type has no grouped-query fields at all, so a GLM pack cannot be priced as grouped-query
//! attention (which would cost zero KV bytes per token).

use crate::family::{Ffn, IndexerMode, LayerSpec, Mixer, Residual};
use gb_cuda::mla::LatentKv;
use gb_sched::{KvLayerClass, KvLayout, KvPlane};
use serde_json::Value;
use std::path::Path;

/// Multi-head latent attention (no RoPE).
#[derive(Clone, Debug, PartialEq)]
pub struct MlaConfig {
    pub heads: usize,
    pub q_lora_rank: usize,
    /// Width of the cached latent.
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    /// Always 0: GLM-5.3 attention carries no rotary part, so there is no shared rope key in the cache.
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
}

impl MlaConfig {
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

/// The DSA lightning indexer with k-pool compression.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexerConfig {
    pub heads: usize,
    pub head_dim: usize,
    /// Tokens selected per query.
    pub topk: usize,
    /// Keys are pooled this many at a time (0: no pooling).
    pub kpool: usize,
    /// Tokens of the incomplete tail pool are always selected, raw.
    pub kpool_always_select_tail: bool,
}

impl IndexerConfig {
    /// Elements of the per-token indexer plane on a layer with its own indexer: the roped key, and with pooling the per-token
    /// gate scores that pooling needs, side by side (the oracle's `idx_plane_dim`).
    pub fn plane_dim(&self) -> usize {
        if self.kpool > 0 {
            2 * self.head_dim
        } else {
            self.head_dim
        }
    }
}

/// Kimi delta attention.
#[derive(Clone, Debug, PartialEq)]
pub struct KdaConfig {
    /// Heads (key heads = value heads).
    pub heads: usize,
    /// Width of a key head and of a value head.
    pub head_dim: usize,
    /// Taps of the depthwise short convolution over q, k and v.
    pub conv_kernel: usize,
    pub gate_lower_bound: Option<f64>,
    /// The heads `[h0, h1)` a tensor-parallel rank holds under the KDA head split (lane E, GLM Phase 2: `REXL3_GLM_TP_SPLIT=kda`, set by
    /// the rank planner, `tp_boot`); `None` on a single node and on a rank without the split. The state the rank allocates follows it (the
    /// `kda_*_bytes_*` formulas, `alloc::dims_of`: pools, window records, the prefill record and workspace) and the mapper places only
    /// those heads' `qkv_proj` strips; the pack's tensors are still claimed at `heads`.
    pub rank: Option<(usize, usize)>,
}

impl KdaConfig {
    /// The heads whose state this model allocates: the rank's under the head split, else every head.
    pub fn state_heads(&self) -> usize {
        self.rank.map_or(self.heads, |(h0, h1)| h1 - h0)
    }
}

/// mHC hyper-connections.
#[derive(Clone, Debug, PartialEq)]
pub struct MhcConfig {
    pub streams: usize,
    pub sinkhorn_iters: usize,
    pub eps: f64,
}

/// Feed-forward blocks.
#[derive(Clone, Debug, PartialEq)]
pub struct FfnConfig {
    /// Width of the dense blocks (layers before `first_k_dense_replace`).
    pub dense_intermediate: usize,
    pub moe_intermediate: usize,
    pub routed_experts: usize,
    pub experts_per_token: usize,
    pub shared_experts: usize,
    pub routed_scaling_factor: f64,
    pub swiglu_limit: f64,
}

/// The vision tower, from `vision_config`, plus the special tokens that delimit media in the prompt.
#[derive(Clone, Debug, PartialEq)]
pub struct GlmVisionConfig {
    pub depth: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub in_channels: usize,
    /// Width of the embeddings handed to the language model (equals the text hidden size).
    pub out_hidden: usize,
    /// Width of the merger's gated MLP.
    pub projection_intermediate: usize,
    pub image_size: usize,
    pub rms_eps: f64,
    pub attention_bias: bool,
    pub swiglu_limit: f64,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub image_start_token_id: u32,
    pub image_end_token_id: u32,
    pub video_start_token_id: u32,
    pub video_end_token_id: u32,
    /// How the pack stores the tower's linears (`quantization_config.vision_bits`): the mapper claims the tensors of that format and
    /// checks every group's width against it, and the memory arithmetic (`vision::weight_region_bytes`, `vision::scratch_items`) follows it.
    pub tower: VisionTowerFormat,
}

/// The storage of the vision tower's linears in a pack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionTowerFormat {
    /// Every tensor plain bf16 (the 2.25 bpw pack, exllamav3 fork 1.5.0: no `vision_bits`): 347 tensors, the fused qkv one GEMM.
    Bf16,
    /// The block and merger linears EXL3 groups of `bits` (exllamav3 1.6.2's `vision_bits`): q, k and v split (the fused bf16 qkv the
    /// pack still holds is claimed and not loaded), fp16 group biases, the patch embedding, downsample and merger norm fp16, the RMSNorm
    /// gammas bf16.
    Exl3 { bits: u8 },
}

impl GlmVisionConfig {
    /// Values in one patch: channels x temporal x patch x patch.
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }
    /// Pixels on a side that one language-model token covers (patch x merge).
    pub fn pixels_per_token_side(&self) -> usize {
        self.patch_size * self.spatial_merge_size
    }
    /// Patches the encoder sees for `tokens` language-model tokens of a still image: the merge folds `merge x merge` patches
    /// into one token, and the temporal patch holds the (duplicated) single frame, so a still image has one temporal slice.
    pub fn patches_for_image_tokens(&self, tokens: usize) -> usize {
        tokens * self.spatial_merge_size * self.spatial_merge_size
    }
    /// Bytes of the tower's weights the load reads (the tensors' data, no alignment padding).
    ///
    /// bf16 tower (all plain bf16 tensors in the pack): patch embedding, `depth` blocks (fused qkv, per-head q/k norms, output projection,
    /// SwiGLU MLP with biases, two norms), the post norm, the 2x2 downsample convolution and the merger (projection, gate, up, down,
    /// post-projection norm). Checked against the pack's 347 tensors in the tests.
    ///
    /// EXL3 tower: the same tensors with every block and merger linear an EXL3 group (trellis `k x n x bits / 8` bytes, fp16 `suh` [k] and
    /// `svh` [n], the fp16 bias [n] of a block linear), q, k and v three groups; the fused bf16 qkv the pack still holds and the codebook
    /// markers are not loaded and not counted.
    pub fn weight_bytes(&self) -> u64 {
        let (h, i, o, p) =
            (self.hidden as u64, self.intermediate as u64, self.out_hidden as u64, self.projection_intermediate as u64);
        let head_dim = h / self.heads as u64;
        let m2 = (self.spatial_merge_size * self.spatial_merge_size) as u64;
        let attention_bias = if self.attention_bias { 1 } else { 0 };
        let patch = self.patch_dim() as u64 * h + h;
        let post_norm = h;
        let downsample = o * h * m2 + o;
        match self.tower {
            VisionTowerFormat::Bf16 => {
                let block = 3 * h * h + attention_bias * 3 * h // qkv
                    + 2 * head_dim // q_norm, k_norm
                    + h * h + attention_bias * h // output projection
                    + 3 * h * i + attention_bias * (2 * i + h) // gate, up, down (+ biases)
                    + 2 * h; // norm1, norm2
                let merger = o * o + 3 * o * p + 2 * o;
                2 * (self.depth as u64 * block + patch + post_norm + downsample + merger)
            }
            VisionTowerFormat::Exl3 { bits } => {
                // bytes of one group k -> n (trellis and the two scale vectors) and of its bias
                let g = |k: u64, n: u64| k * n * bits as u64 / 8 + 2 * (k + n);
                let b = |n: u64| attention_bias * 2 * n;
                let block = 2 * (2 * h + 2 * head_dim) // norm1, norm2, q_norm, k_norm (bf16)
                    + 4 * (g(h, h) + b(h)) // q, k, v, proj
                    + 2 * (g(h, i) + b(i)) // gate, up
                    + g(i, h) + b(h); // down
                let merger = g(o, o) + 2 * g(o, p) + g(p, o) + 2 * 2 * o;
                self.depth as u64 * block + 2 * (patch + post_norm + downsample) + merger
            }
        }
    }
    /// Tensors of the tower in the pack. bf16: 14 per block, 2 patch embedding, 1 post norm, 2 downsample, 6 merger. EXL3: per block the four
    /// norms, the fused bf16 qkv weight and bias, and 7 groups of 5 (trellis, suh, svh, codebook marker, bias); the merger's 4 groups of 4 and
    /// its norm's 2; patch embedding 2, post norm 1, downsample 2 (1,007 in the sc_4.10bpw pack).
    pub fn tensor_count(&self) -> usize {
        match self.tower {
            VisionTowerFormat::Bf16 => 14 * self.depth + 2 + 1 + 2 + 6,
            VisionTowerFormat::Exl3 { .. } => (4 + 2 + 7 * 5) * self.depth + 2 + 1 + 2 + 4 * 4 + 2,
        }
    }
    /// Bytes of an image's embeddings as the language model consumes them: one f32 row of `out_hidden` per token (the tower's merger writes fp32
    /// and the oracle splices it as it is, so an image row reaches the residual streams without a rounding the text rows do not have). They are
    /// held until the sequence that owns the image ends.
    pub fn feature_bytes(&self, tokens: usize) -> u64 {
        tokens as u64 * self.out_hidden as u64 * 4
    }
    /// Working buffers of one encode of an image of up to `tokens` language-model tokens: exactly what `vision::GlmVisionTower` allocates
    /// (`vision::scratch_items`, which the tower's allocations are built from, so the plan equals the allocations by construction and a GPU test
    /// sums the real ones). About 34 KB a patch, the cuBLAS workspace the tower hands the library, and a few bytes of constants.
    pub fn encoder_scratch_bytes(&self, tokens: usize) -> u64 {
        super::vision::scratch_bytes(self.patches_for_image_tokens(tokens), self.tower)
    }
}

/// Image preprocessing (`processor_config.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct ImageProcessorConfig {
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    /// Fewest and most language-model tokens one image may take after resizing.
    pub min_image_tokens: usize,
    pub max_image_tokens: usize,
    pub mean: [f64; 3],
    pub std: [f64; 3],
    pub rescale: bool,
}

/// Video preprocessing.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoProcessorConfig {
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_image_tokens: usize,
    pub max_image_tokens: usize,
    pub fps: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlmProcessorConfig {
    pub image: ImageProcessorConfig,
    pub video: Option<VideoProcessorConfig>,
}

/// The GLM-5.3-Flash text model, its MTP head and its vision tower.
#[derive(Clone, Debug, PartialEq)]
pub struct GlmConfig {
    pub hidden: usize,
    pub vocab: usize,
    pub rms_eps: f64,
    pub max_positions: usize,
    /// The 45 trunk layers.
    pub layers: Vec<LayerSpec>,
    /// NextN layers after the trunk (index `layers.len() + i`): empty, or one MLA + indexer + MoE layer on a plain residual.
    pub mtp: Vec<LayerSpec>,
    pub mla: MlaConfig,
    pub indexer: IndexerConfig,
    pub kda: KdaConfig,
    pub mhc: MhcConfig,
    pub ffn: FfnConfig,
    pub eos: Vec<u32>,
    pub pad_token_id: Option<u32>,
    pub vision: Option<GlmVisionConfig>,
    /// `None` when the pack ships no processor config (the vision pipeline must then refuse to start).
    pub processor: Option<GlmProcessorConfig>,
    /// The MLA latent cache's format: a serving setting, not the pack's (`--kv-dtype`, `gb_cuda::mla::KV_ENV`), applied by
    /// [`GlmConfig::from_dir`] from the process's configuration (fp16 from [`GlmConfig::from_json`]). The planner prices and the model
    /// allocates this format, so the two cannot disagree.
    pub latent_kv: LatentKv,
}

// ---------------------------------------------------------------------------------------------------------------- parsing

fn field<'a>(v: &'a Value, path: &str, k: &str) -> Result<&'a Value, String> {
    v.get(k).ok_or_else(|| format!("config.json: missing `{path}{k}`"))
}

fn usize_at(v: &Value, path: &str, k: &str) -> Result<usize, String> {
    field(v, path, k)?
        .as_u64()
        .map(|x| x as usize)
        .ok_or_else(|| format!("config.json: `{path}{k}` must be a non-negative integer"))
}

fn f64_at(v: &Value, path: &str, k: &str) -> Result<f64, String> {
    field(v, path, k)?.as_f64().ok_or_else(|| format!("config.json: `{path}{k}` must be a number"))
}

fn bool_at(v: &Value, path: &str, k: &str) -> Result<bool, String> {
    field(v, path, k)?.as_bool().ok_or_else(|| format!("config.json: `{path}{k}` must be true or false"))
}

fn str_at<'a>(v: &'a Value, path: &str, k: &str) -> Result<&'a str, String> {
    field(v, path, k)?.as_str().ok_or_else(|| format!("config.json: `{path}{k}` must be a string"))
}

/// The tower's storage from `quantization_config.vision_bits` (top level, where exllamav3 writes it): absent is the bf16 tower, a whole width of 1 to
/// 8 bits an EXL3 tower of that width. Anything else is refused (a fractional or out-of-range width is not a tower the binder can claim).
fn vision_tower_format(root: &Value) -> Result<VisionTowerFormat, String> {
    let Some(b) = root.get("quantization_config").and_then(|q| q.get("vision_bits")) else {
        return Ok(VisionTowerFormat::Bf16);
    };
    match b.as_u64() {
        Some(bits @ 1..=8) => Ok(VisionTowerFormat::Exl3 { bits: bits as u8 }),
        _ => Err(format!(
            "config.json: `quantization_config.vision_bits` is {b}; an EXL3 vision tower takes a whole width of 1 to 8 bits"
        )),
    }
}

fn positive(v: &Value, path: &str, k: &str) -> Result<usize, String> {
    match usize_at(v, path, k)? {
        0 => Err(format!("config.json: `{path}{k}` must be positive, found 0")),
        n => Ok(n),
    }
}

fn expect_str(v: &Value, path: &str, k: &str, want: &str) -> Result<(), String> {
    let got = str_at(v, path, k)?;
    if got != want {
        return Err(format!("config.json: `{path}{k}` is `{got}`, this engine implements `{want}`"));
    }
    Ok(())
}

fn expect_true(v: &Value, path: &str, k: &str) -> Result<(), String> {
    if !bool_at(v, path, k)? {
        return Err(format!("config.json: `{path}{k}` is false, this engine implements only true"));
    }
    Ok(())
}

fn strings<'a>(v: &'a Value, path: &str, k: &str) -> Result<Vec<&'a str>, String> {
    field(v, path, k)?
        .as_array()
        .ok_or_else(|| format!("config.json: `{path}{k}` must be a list"))?
        .iter()
        .map(|x| x.as_str().ok_or_else(|| format!("config.json: `{path}{k}` must be a list of strings")))
        .collect()
}

fn indices(v: &Value, path: &str, k: &str) -> Result<Vec<usize>, String> {
    field(v, path, k)?
        .as_array()
        .ok_or_else(|| format!("config.json: `{path}{k}` must be a list"))?
        .iter()
        .map(|x| {
            x.as_u64().map(|x| x as usize).ok_or_else(|| format!("config.json: `{path}{k}` must be a list of layer indices"))
        })
        .collect()
}

/// Finite and above zero (a NaN is neither).
fn is_positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

fn is_non_negative(x: f64) -> bool {
    x.is_finite() && x >= 0.0
}

fn token_id(v: &Value, path: &str, k: &str, vocab: usize) -> Result<u32, String> {
    let id = usize_at(v, path, k)?;
    if id >= vocab {
        return Err(format!("config.json: `{path}{k}` is {id}, outside the vocabulary of {vocab}"));
    }
    Ok(id as u32)
}

impl GlmConfig {
    pub fn from_json(root: &Value) -> Result<Self, String> {
        let t = root.get("text_config").ok_or("config.json: missing `text_config`")?;
        let tp = "text_config.";
        let hidden = positive(t, tp, "hidden_size")?;
        let vocab = positive(t, tp, "vocab_size")?;
        let n = positive(t, tp, "num_hidden_layers")?;

        // ---- the layer composition: mixer, feed-forward block and residual of every trunk layer
        let layer_types = strings(t, tp, "layer_types")?;
        let mlp_types = strings(t, tp, "mlp_layer_types")?;
        let indexer_types = strings(t, tp, "indexer_types")?;
        for (name, len) in
            [("layer_types", layer_types.len()), ("mlp_layer_types", mlp_types.len()), ("indexer_types", indexer_types.len())]
        {
            if len != n {
                return Err(format!("config.json: `{tp}{name}` has {len} entries but `{tp}num_hidden_layers` is {n}"));
            }
        }
        let first_dense = usize_at(t, tp, "first_k_dense_replace")?;
        expect_true(t, tp, "mhc")?;
        let mhc = MhcConfig {
            streams: positive(t, tp, "hc_mult")?,
            sinkhorn_iters: positive(t, tp, "hc_sinkhorn_iters")?,
            eps: f64_at(t, tp, "hc_eps")?,
        };
        if !is_positive(mhc.eps) {
            return Err("config.json: `text_config.hc_eps` must be positive".into());
        }
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let mixer = match layer_types[i] {
                "linear_attention" => Mixer::Kda,
                "deepseek_sparse_attention" => Mixer::MlaDsa {
                    indexer: match indexer_types[i] {
                        "full" => IndexerMode::Full,
                        "shared" => IndexerMode::Shared,
                        other => return Err(format!("config.json: unsupported indexer type `{other}` at layer {i}")),
                    },
                },
                other => return Err(format!("config.json: unsupported layer type `{other}` at layer {i}")),
            };
            let ffn = match mlp_types[i] {
                "dense" => Ffn::Dense,
                "sparse" => Ffn::Moe,
                other => return Err(format!("config.json: unsupported mlp layer type `{other}` at layer {i}")),
            };
            if (ffn == Ffn::Dense) != (i < first_dense) {
                return Err(format!(
                    "config.json: `{tp}mlp_layer_types` says layer {i} is {} but `{tp}first_k_dense_replace` is {first_dense}",
                    mlp_types[i]
                ));
            }
            layers.push(LayerSpec { mixer, ffn, residual: Residual::Mhc { streams: mhc.streams } });
        }
        // the first attention layer must score its own selection: a shared indexer reuses a preceding full one
        if let Some(first) = layers.iter().find_map(|l| match l.mixer {
            Mixer::MlaDsa { indexer } => Some(indexer),
            _ => None,
        }) {
            if first != IndexerMode::Full {
                return Err("config.json: the first sparse-attention layer cannot share an indexer selection".into());
            }
        }

        // ---- KDA
        let la = field(t, tp, "linear_attn_config")?;
        let lp = "text_config.linear_attn_config.";
        let kda = KdaConfig {
            heads: positive(la, lp, "num_heads")?,
            head_dim: positive(la, lp, "head_dim")?,
            conv_kernel: positive(la, lp, "short_conv_kernel_size")?,
            gate_lower_bound: la.get("gate_lower_bound").and_then(|x| x.as_f64()),
            rank: None,
        };
        if kda.conv_kernel < 2 {
            return Err("config.json: `text_config.linear_attn_config.short_conv_kernel_size` must be at least 2".into());
        }
        // the two layer lists must say what layer_types says, or one of the three is wrong and we cannot know which
        let kda_listed = indices(la, lp, "kda_layers")?;
        let mla_listed = indices(la, lp, "full_attn_layers")?;
        let kda_derived: Vec<usize> = (0..n).filter(|&i| layers[i].mixer == Mixer::Kda).collect();
        let mla_derived: Vec<usize> = (0..n).filter(|&i| layers[i].mixer != Mixer::Kda).collect();
        if kda_listed != kda_derived || mla_listed != mla_derived {
            return Err(format!(
                "config.json: `{lp}kda_layers` / `full_attn_layers` ({} / {} layers) disagree with `{tp}layer_types` ({} KDA / {} attention)",
                kda_listed.len(),
                mla_listed.len(),
                kda_derived.len(),
                mla_derived.len()
            ));
        }

        // ---- MLA, without RoPE
        expect_true(t, tp, "mla_use_nope")?;
        let mla = MlaConfig {
            heads: positive(t, tp, "num_attention_heads")?,
            q_lora_rank: positive(t, tp, "q_lora_rank")?,
            kv_lora_rank: positive(t, tp, "kv_lora_rank")?,
            qk_nope_head_dim: positive(t, tp, "qk_nope_head_dim")?,
            qk_rope_head_dim: usize_at(t, tp, "qk_rope_head_dim")?,
            v_head_dim: positive(t, tp, "v_head_dim")?,
        };
        if mla.qk_rope_head_dim != 0 {
            return Err(format!(
                "config.json: `{tp}qk_rope_head_dim` is {}; GLM-5.3 attention is NoPE and the cache has no rope plane",
                mla.qk_rope_head_dim
            ));
        }
        if usize_at(t, tp, "qk_head_dim")? != mla.qk_head_dim() {
            return Err("config.json: `text_config.qk_head_dim` is not qk_nope_head_dim + qk_rope_head_dim".into());
        }
        if usize_at(t, tp, "num_key_value_heads")? != mla.heads {
            return Err(
                "config.json: `text_config.num_key_value_heads` must equal `num_attention_heads` (MLA caches one latent)".into(),
            );
        }

        // ---- the indexer
        expect_true(t, tp, "index_kpool_compress")?;
        let indexer = IndexerConfig {
            heads: positive(t, tp, "index_n_heads")?,
            head_dim: positive(t, tp, "index_head_dim")?,
            topk: positive(t, tp, "index_topk")?,
            kpool: usize_at(t, tp, "index_kpool")?,
            kpool_always_select_tail: bool_at(t, tp, "index_kpool_always_select_tail")?,
        };

        // ---- feed-forward blocks and routing
        expect_str(t, tp, "hidden_act", "silu")?;
        expect_str(t, tp, "scoring_func", "sigmoid")?;
        expect_str(t, tp, "topk_method", "noaux_tc")?;
        expect_true(t, tp, "norm_topk_prob")?;
        for k in ["n_group", "topk_group"] {
            if usize_at(t, tp, k)? != 1 {
                return Err(format!("config.json: `{tp}{k}` must be 1 (group-limited routing is not implemented)"));
            }
        }
        let ffn = FfnConfig {
            dense_intermediate: positive(t, tp, "intermediate_size")?,
            moe_intermediate: positive(t, tp, "moe_intermediate_size")?,
            routed_experts: positive(t, tp, "n_routed_experts")?,
            experts_per_token: positive(t, tp, "num_experts_per_tok")?,
            shared_experts: usize_at(t, tp, "n_shared_experts")?,
            routed_scaling_factor: f64_at(t, tp, "routed_scaling_factor")?,
            swiglu_limit: f64_at(t, tp, "swiglu_limit")?,
        };
        if ffn.experts_per_token > ffn.routed_experts {
            return Err("config.json: `text_config.num_experts_per_tok` exceeds `n_routed_experts`".into());
        }
        if !is_positive(ffn.routed_scaling_factor) || !is_non_negative(ffn.swiglu_limit) {
            return Err("config.json: `routed_scaling_factor` must be positive and `swiglu_limit` non-negative".into());
        }
        let rms_eps = f64_at(t, tp, "rms_norm_eps")?;
        if !is_positive(rms_eps) {
            return Err("config.json: `text_config.rms_norm_eps` must be positive".into());
        }

        // ---- the NextN / MTP head: a full MLA + indexer + MoE layer on a plain residual
        let mtp = match usize_at(t, tp, "num_nextn_predict_layers")? {
            0 => vec![],
            1 => {
                vec![LayerSpec { mixer: Mixer::MlaDsa { indexer: IndexerMode::Full }, ffn: Ffn::Moe, residual: Residual::Plain }]
            }
            k => return Err(format!("config.json: `{tp}num_nextn_predict_layers` is {k}; one MTP layer is implemented")),
        };

        let eos = match t.get("eos_token_id") {
            Some(Value::Number(x)) => x.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
            Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect(),
            _ => vec![],
        };
        if let Some(bad) = eos.iter().find(|&&id| id as usize >= vocab) {
            return Err(format!("config.json: eos token {bad} is outside the vocabulary of {vocab}"));
        }
        let pad_token_id = t.get("pad_token_id").and_then(|x| x.as_u64()).map(|x| x as u32);

        let vision = match root.get("vision_config") {
            Some(v) => Some(Self::vision_from_json(root, v, hidden, vocab)?),
            None => None,
        };

        Ok(Self {
            hidden,
            vocab,
            rms_eps,
            max_positions: positive(t, tp, "max_position_embeddings")?,
            layers,
            mtp,
            mla,
            indexer,
            kda,
            mhc,
            ffn,
            eos,
            pad_token_id,
            vision,
            processor: None,
            latent_kv: LatentKv::Plain,
        })
    }

    fn vision_from_json(root: &Value, v: &Value, hidden: usize, vocab: usize) -> Result<GlmVisionConfig, String> {
        let vp = "vision_config.";
        expect_str(v, vp, "hidden_act", "silu")?;
        let c = GlmVisionConfig {
            depth: positive(v, vp, "depth")?,
            hidden: positive(v, vp, "hidden_size")?,
            intermediate: positive(v, vp, "intermediate_size")?,
            heads: positive(v, vp, "num_heads")?,
            patch_size: positive(v, vp, "patch_size")?,
            spatial_merge_size: positive(v, vp, "spatial_merge_size")?,
            temporal_patch_size: positive(v, vp, "temporal_patch_size")?,
            in_channels: positive(v, vp, "in_channels")?,
            out_hidden: positive(v, vp, "out_hidden_size")?,
            projection_intermediate: positive(v, vp, "projection_intermediate_size")?,
            image_size: positive(v, vp, "image_size")?,
            rms_eps: f64_at(v, vp, "rms_norm_eps")?,
            attention_bias: bool_at(v, vp, "attention_bias")?,
            swiglu_limit: f64_at(v, vp, "swiglu_limit")?,
            image_token_id: token_id(root, "", "image_token_id", vocab)?,
            video_token_id: token_id(root, "", "video_token_id", vocab)?,
            image_start_token_id: token_id(root, "", "image_start_token_id", vocab)?,
            image_end_token_id: token_id(root, "", "image_end_token_id", vocab)?,
            video_start_token_id: token_id(root, "", "video_start_token_id", vocab)?,
            video_end_token_id: token_id(root, "", "video_end_token_id", vocab)?,
            tower: vision_tower_format(root)?,
        };
        if c.hidden % c.heads != 0 {
            return Err("config.json: `vision_config.hidden_size` is not a multiple of `num_heads`".into());
        }
        if c.out_hidden != hidden {
            return Err(format!(
                "config.json: `vision_config.out_hidden_size` is {} but the language model's hidden size is {hidden}: image embeddings could not be spliced in",
                c.out_hidden
            ));
        }
        let mut ids = [
            c.image_token_id,
            c.video_token_id,
            c.image_start_token_id,
            c.image_end_token_id,
            c.video_start_token_id,
            c.video_end_token_id,
        ];
        ids.sort_unstable();
        if ids.windows(2).any(|w| w[0] == w[1]) {
            return Err("config.json: the image and video marker token ids are not all distinct".into());
        }
        Ok(c)
    }

    /// Parse `processor_config.json` (an `image_processor` section and an optional `video_processor` section).
    pub fn processor_from_json(root: &Value) -> Result<GlmProcessorConfig, String> {
        let ip = root.get("image_processor").ok_or("processor_config.json: missing `image_processor`")?;
        let triple = |v: &Value, k: &str| -> Result<[f64; 3], String> {
            let a = v.get(k).and_then(|x| x.as_array()).ok_or_else(|| format!("processor_config.json: missing `{k}`"))?;
            if a.len() != 3 {
                return Err(format!("processor_config.json: `{k}` must have 3 entries"));
            }
            let mut out = [0.0; 3];
            for (o, x) in out.iter_mut().zip(a) {
                *o = x.as_f64().ok_or_else(|| format!("processor_config.json: `{k}` must be numbers"))?;
            }
            Ok(out)
        };
        let uz = |v: &Value, k: &str| -> Result<usize, String> {
            match v.get(k).and_then(|x| x.as_u64()) {
                Some(0) | None => Err(format!("processor_config.json: `{k}` must be a positive integer")),
                Some(x) => Ok(x as usize),
            }
        };
        let image = ImageProcessorConfig {
            patch_size: uz(ip, "patch_size")?,
            merge_size: uz(ip, "merge_size")?,
            temporal_patch_size: uz(ip, "temporal_patch_size")?,
            min_image_tokens: uz(ip, "min_image_tokens")?,
            max_image_tokens: uz(ip, "max_image_tokens")?,
            mean: triple(ip, "image_mean")?,
            std: triple(ip, "image_std")?,
            rescale: ip.get("do_rescale").and_then(|x| x.as_bool()).unwrap_or(true),
        };
        if image.min_image_tokens > image.max_image_tokens {
            return Err("processor_config.json: `min_image_tokens` exceeds `max_image_tokens`".into());
        }
        // `patch_expand_factor` multiplies the patches a token covers; the token and memory arithmetic here assumes 1
        for (section, v) in [("image_processor", Some(ip)), ("video_processor", root.get("video_processor"))] {
            if let Some(f) = v.and_then(|v| v.get("patch_expand_factor")).and_then(|x| x.as_u64()) {
                if f != 1 {
                    return Err(format!(
                        "processor_config.json: `{section}.patch_expand_factor` is {f}; this engine implements 1"
                    ));
                }
            }
        }
        let video = match root.get("video_processor") {
            Some(vp) => Some(VideoProcessorConfig {
                patch_size: uz(vp, "patch_size")?,
                merge_size: uz(vp, "merge_size")?,
                temporal_patch_size: uz(vp, "temporal_patch_size")?,
                min_image_tokens: uz(vp, "min_image_tokens")?,
                max_image_tokens: uz(vp, "max_image_tokens")?,
                fps: vp.get("fps").and_then(|x| x.as_f64()).unwrap_or(0.0),
            }),
            None => None,
        };
        Ok(GlmProcessorConfig { image, video })
    }

    /// Read `config.json` and, when the pack ships one, its processor config (`processor_config.json`, or the split
    /// `preprocessor_config.json` + `video_preprocessor_config.json` layout), and check the two agree.
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let read = |name: &str| -> Result<Option<Value>, String> {
            let p = dir.join(name);
            if !p.exists() {
                return Ok(None);
            }
            let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            serde_json::from_str(&text).map(Some).map_err(|e| format!("{name}: {e}"))
        };
        let root = read("config.json")?.ok_or_else(|| format!("{}: not found", dir.join("config.json").display()))?;
        let mut c = Self::from_json(&root)?;
        let processor = match read("processor_config.json")? {
            Some(p) => Some(Self::processor_from_json(&p)?),
            None => match read("preprocessor_config.json")? {
                Some(img) => {
                    let mut combined = serde_json::json!({ "image_processor": img });
                    if let Some(vid) = read("video_preprocessor_config.json")? {
                        combined["video_processor"] = vid;
                    }
                    Some(Self::processor_from_json(&combined)?)
                }
                None => None,
            },
        };
        if let (Some(p), Some(v)) = (&processor, &c.vision) {
            for (what, got, want) in [
                ("patch_size", p.image.patch_size, v.patch_size),
                ("merge_size", p.image.merge_size, v.spatial_merge_size),
                ("temporal_patch_size", p.image.temporal_patch_size, v.temporal_patch_size),
            ] {
                if got != want {
                    return Err(format!("processor_config.json: image `{what}` is {got} but `vision_config` says {want}"));
                }
            }
        }
        c.processor = processor;
        c.latent_kv = gb_cuda::mla::configured_kv()?;
        Ok(c)
    }

    // ---------------------------------------------------------------------------------------------- derived composition

    /// Trunk layers with the given mixer kind, by index.
    fn indices_where(&self, f: impl Fn(&LayerSpec) -> bool) -> Vec<usize> {
        (0..self.layers.len()).filter(|&i| f(&self.layers[i])).collect()
    }
    /// Trunk layers that mix with KDA.
    pub fn kda_layers(&self) -> Vec<usize> {
        self.indices_where(|l| l.mixer == Mixer::Kda)
    }
    /// Trunk layers that mix with sparse-attention MLA.
    pub fn mla_layers(&self) -> Vec<usize> {
        self.indices_where(|l| matches!(l.mixer, Mixer::MlaDsa { .. }))
    }
    pub fn dense_layers(&self) -> Vec<usize> {
        self.indices_where(|l| l.ffn == Ffn::Dense)
    }
    pub fn moe_layers(&self) -> Vec<usize> {
        self.indices_where(|l| l.ffn == Ffn::Moe)
    }
    /// Index of the `i`-th MTP layer in the checkpoint's numbering (it follows the trunk).
    pub fn mtp_layer_index(&self, i: usize) -> usize {
        self.layers.len() + i
    }
    pub fn has_mtp(&self) -> bool {
        !self.mtp.is_empty()
    }

    // ------------------------------------------------------------------------------------------------ byte formulas

    /// Bytes of the committed KDA state of one sequence with one buffer: per KDA layer the delta-rule state
    /// `heads x head_dim x head_dim` in fp32, and the short-convolution history of the q, k and v channels
    /// `(conv_kernel - 1) x 3 x heads x head_dim` in bf16.
    pub fn kda_state_bytes_one_buffer(&self) -> u64 {
        self.kda_layers().len() as u64 * (self.kda_delta_state_bytes_per_layer() + self.kda_conv_history_bytes_per_layer())
    }
    /// The delta-rule state alone (fp32) in one KDA layer.
    pub fn kda_delta_state_bytes_per_layer(&self) -> u64 {
        (self.kda.state_heads() * self.kda.head_dim * self.kda.head_dim) as u64 * 4
    }
    /// The short-convolution history (bf16) in one KDA layer.
    pub fn kda_conv_history_bytes_per_layer(&self) -> u64 {
        ((self.kda.conv_kernel - 1) * 3 * self.kda.state_heads() * self.kda.head_dim) as u64 * 2
    }
    /// What a window keeps per row per KDA layer until its commit, so the accepted prefix can be replayed from the pre-window state with
    /// the serial update routine and the conv history shifted: the raw bf16 q|k|v row (`3 x channels` bf16), the per-channel decay (fp32)
    /// and beta (fp32 per head): `gb_cuda::kda::Dims::record_bytes_per_row`. The conv shift reads the same raw rows, so there is no second
    /// copy (the memory model of slice 1 counted one: 49,152 B a row and layer too many).
    pub fn kda_record_bytes_per_row_per_layer(&self) -> u64 {
        let channels = (self.kda.state_heads() * self.kda.head_dim) as u64;
        3 * channels * 2 + channels * 4 + self.kda.state_heads() as u64 * 4
    }

    /// This configuration for rank `rank` of a two-rank group under the KDA head split: heads `[rank * heads / 2, (rank + 1) * heads / 2)`
    /// (the split's halves, `model/tp.rs`). Refused unless the heads halve into whole 128-column strips.
    pub fn with_kda_rank(&self, rank: usize, ranks: usize) -> Result<GlmConfig, String> {
        let (heads, d) = (self.kda.heads, self.kda.head_dim);
        if ranks != 2 || rank >= ranks || heads % 2 != 0 || (heads / 2 * d) % 128 != 0 || self.kda.rank.is_some() {
            return Err(format!(
                "the KDA head split takes rank 0 or 1 of two ranks and heads that halve into 128-column strips (rank {rank} of {ranks}, {heads} heads of {d}{})",
                if self.kda.rank.is_some() { ", already a rank's" } else { "" }
            ));
        }
        let mut c = self.clone();
        c.kda.rank = Some((rank * heads / 2, (rank + 1) * heads / 2));
        Ok(c)
    }

    /// The KV layout of the MLA layers: per token a latent of `kv_lora_rank` fp16 values (with [`LatentKv::Fp8`]: as many E4M3 bytes and
    /// one fp32 scale, `gb_cuda::mla::kv8`) and, on a layer with its own indexer, the pooled
    /// key plane of `index_head_dim` values per `kpool` tokens. A shared-indexer layer caches the latent only. With pooling there is NO
    /// per-token indexer plane: the raw `[LayerNorm(k) | gate]` rows the pools are built from live in a ring of `ring_len` rows per slot
    /// (`alloc.rs`: per-slot state, not per-token cost), because a pool is complete four tokens after its first member and the raw rows
    /// are never read again. Without pooling (not built) the indexer would cache one `head_dim` key per token. `with_mtp` adds the MTP
    /// layers (always full-indexer), which keep a cache of their own at the same positions. No rope plane: `qk_rope_head_dim` is
    /// validated to be 0.
    pub fn kv_layout(&self, with_mtp: bool) -> KvLayout {
        let mut full =
            self.layers.iter().filter(|l| matches!(l.mixer, Mixer::MlaDsa { indexer: IndexerMode::Full })).count() as u64;
        let shared =
            self.layers.iter().filter(|l| matches!(l.mixer, Mixer::MlaDsa { indexer: IndexerMode::Shared })).count() as u64;
        if with_mtp {
            full += self.mtp.len() as u64;
        }
        let latent: Vec<KvPlane> = match self.latent_kv {
            LatentKv::Plain => vec![KvPlane::per_token("latent", self.mla.kv_lora_rank as u64, KV_ELEM_BYTES)],
            LatentKv::Fp8 => vec![
                KvPlane::per_token("latent", self.mla.kv_lora_rank as u64, 1),
                KvPlane::per_token("latent_scale", 1, 4),
            ],
        };
        let mut with_indexer = latent.clone();
        if self.indexer.kpool > 0 {
            with_indexer.push(KvPlane::pooled(
                "indexer_pool",
                self.indexer.head_dim as u64,
                self.indexer.kpool as u64,
                KV_ELEM_BYTES,
            ));
        } else {
            with_indexer.push(KvPlane::per_token("indexer_k", self.indexer.head_dim as u64, KV_ELEM_BYTES));
        }
        let mut layout = KvLayout::new();
        // classes with no layers are left out: an empty class is an error, a model without shared layers is not
        if full > 0 {
            layout = layout.with(KvLayerClass { name: "mla_dsa_full_indexer", layers: full, planes: with_indexer });
        }
        if shared > 0 {
            layout = layout.with(KvLayerClass { name: "mla_dsa_shared_indexer", layers: shared, planes: latent });
        }
        layout
    }
}

/// fp16, as the reference engine caches latents and indexer planes (`torch.half`).
const KV_ELEM_BYTES: u64 = 2;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const CONFIG: &str = include_str!("../../../../../testdata/glm53/config.json");
    const PROCESSOR: &str = include_str!("../../../../../testdata/glm53/processor_config.json");

    fn real_json() -> Value {
        serde_json::from_str(CONFIG).unwrap()
    }

    fn real() -> GlmConfig {
        GlmConfig::from_json(&real_json()).unwrap()
    }

    fn err_with(edit: impl FnOnce(&mut Value)) -> String {
        let mut v = real_json();
        edit(&mut v);
        GlmConfig::from_json(&v).unwrap_err()
    }

    #[test]
    fn the_real_config_is_the_45_layer_composition() {
        let c = real();
        assert_eq!((c.hidden, c.vocab, c.layers.len(), c.rms_eps), (4096, 154_880, 45, 1e-5));
        // 11 sparse-attention layers at 3, 7, ..., 43 and 34 KDA layers everywhere else
        let mla: Vec<usize> = (0..11).map(|k| 3 + 4 * k).collect();
        assert_eq!(mla.last(), Some(&43));
        assert_eq!(c.mla_layers(), mla);
        assert_eq!(c.kda_layers().len(), 34);
        assert!(c.kda_layers().iter().all(|i| i % 4 != 3));
        // layers 0-2 are dense, 3-44 are mixture of experts: 3 + 42
        assert_eq!(c.dense_layers(), vec![0, 1, 2]);
        assert_eq!(c.moe_layers(), (3..45).collect::<Vec<_>>());
        assert_eq!(c.moe_layers().len(), 42);
        // every trunk layer has mHC residuals of 4 streams; every attention layer scores its own indexer selection
        for (i, l) in c.layers.iter().enumerate() {
            assert_eq!(l.residual, Residual::Mhc { streams: 4 }, "layer {i}");
            match l.mixer {
                Mixer::Kda => assert!(i % 4 != 3, "layer {i}"),
                Mixer::MlaDsa { indexer } => assert_eq!((i % 4, indexer), (3, IndexerMode::Full), "layer {i}"),
                other => panic!("layer {i}: {other:?}"),
            }
            assert_eq!(l.ffn, if i < 3 { Ffn::Dense } else { Ffn::Moe }, "layer {i}");
        }
        // one NextN layer, numbered 45: a full MLA + indexer + MoE layer with a plain residual
        assert_eq!(c.mtp.len(), 1);
        assert_eq!(c.mtp_layer_index(0), 45);
        assert!(c.has_mtp());
        assert_eq!(
            c.mtp[0],
            LayerSpec { mixer: Mixer::MlaDsa { indexer: IndexerMode::Full }, ffn: Ffn::Moe, residual: Residual::Plain }
        );
    }

    #[test]
    fn the_real_config_parameters() {
        let c = real();
        assert_eq!(
            c.mla,
            MlaConfig {
                heads: 64,
                q_lora_rank: 1536,
                kv_lora_rank: 512,
                qk_nope_head_dim: 256,
                qk_rope_head_dim: 0,
                v_head_dim: 256
            }
        );
        assert_eq!(c.mla.qk_head_dim(), 256);
        assert_eq!(c.indexer, IndexerConfig { heads: 32, head_dim: 128, topk: 2048, kpool: 4, kpool_always_select_tail: true });
        assert_eq!(c.indexer.plane_dim(), 256, "k and the gate scores side by side");
        assert_eq!(c.kda, KdaConfig { heads: 64, head_dim: 128, conv_kernel: 4, gate_lower_bound: Some(-5.0), rank: None });
        assert_eq!(c.mhc, MhcConfig { streams: 4, sinkhorn_iters: 20, eps: 1e-6 });
        assert_eq!(
            c.ffn,
            FfnConfig {
                dense_intermediate: 12288,
                moe_intermediate: 2048,
                routed_experts: 288,
                experts_per_token: 8,
                shared_experts: 1,
                routed_scaling_factor: 2.5,
                swiglu_limit: 10.0
            }
        );
        assert_eq!(c.eos, vec![154820, 154827, 154829]);
        assert_eq!(c.pad_token_id, Some(154820));
        assert_eq!(c.max_positions, 1_048_576);
        assert!(c.processor.is_none(), "from_json reads config.json only");
    }

    #[test]
    fn the_vision_config_is_carried_in_full() {
        let v = real().vision.unwrap();
        assert_eq!((v.depth, v.hidden, v.intermediate, v.heads), (24, 1024, 4096, 16));
        assert_eq!((v.patch_size, v.spatial_merge_size, v.temporal_patch_size, v.in_channels), (14, 2, 2, 3));
        assert_eq!((v.out_hidden, v.projection_intermediate, v.image_size), (4096, 10240, 448));
        assert_eq!((v.rms_eps, v.attention_bias, v.swiglu_limit), (1e-5, true, 10.0));
        assert_eq!(
            (
                v.image_token_id,
                v.video_token_id,
                v.image_start_token_id,
                v.image_end_token_id,
                v.video_start_token_id,
                v.video_end_token_id
            ),
            (154854, 154855, 154830, 154831, 154832, 154833)
        );
        // 3 channels x 2 temporal x 14 x 14 = 1176 values a patch; 28 pixels on a side per language-model token
        assert_eq!(v.patch_dim(), 1176);
        assert_eq!(v.pixels_per_token_side(), 28);
    }

    #[test]
    fn image_tokens_and_patches() {
        let v = real().vision.unwrap();
        // a still image: merge 2 folds 2 x 2 patches into one token, and the two-frame temporal patch holds the one (duplicated)
        // frame, so a still image has one temporal slice and 4 patches per token (a video adds slices, not patches per token)
        assert_eq!(v.patches_for_image_tokens(1), 4);
        assert_eq!(v.patches_for_image_tokens(4096), 16_384);
        // an 896 x 896 canvas is 32 x 32 = 1024 tokens, 4096 patches
        let side = 896 / v.pixels_per_token_side();
        assert_eq!(side * side, 1024);
        assert_eq!(v.patches_for_image_tokens(side * side), 64 * 64);
        // 16 KiB of (f32) embeddings per token, whatever the image
        assert_eq!(v.feature_bytes(1), 16_384);
    }

    #[test]
    fn the_tower_weights_equal_the_packs_347_bf16_tensors() {
        let v = real().vision.unwrap();
        // summed from the safetensors headers of the real pack: 347 tensors, all BF16, 1,127,254,016 bytes
        assert_eq!(v.tensor_count(), 347);
        assert_eq!(v.weight_bytes(), 1_127_254_016);
        // one block by hand: qkv 3 x 1024 x 1024 + 3072, q/k norms 2 x 64, proj 1024 x 1024 + 1024, gate and up 2 x (4096 x 1024 + 4096),
        // down 1024 x 4096 + 1024, two norms 2 x 1024 = 16,792,704 values, 33,585,408 bytes
        let block = 3_148_800 + 128 + 1_049_600 + 2 * 4_198_400 + 4_195_328 + 2048;
        assert_eq!(block * 2, 33_585_408);
        // the rest: patch embedding, post norm, 2 x 2 downsample, merger
        let rest = (1_204_224 + 1024) + 1024 + (4096 * 1024 * 4 + 4096) + (4096 * 4096 + 3 * 4096 * 10240 + 2 * 4096);
        assert_eq!(2 * (24 * block + rest), 1_127_254_016);
    }

    #[test]
    fn the_processor_config_parses_and_agrees_with_the_tower() {
        let p = GlmConfig::processor_from_json(&serde_json::from_str(PROCESSOR).unwrap()).unwrap();
        assert_eq!((p.image.patch_size, p.image.merge_size, p.image.temporal_patch_size), (14, 2, 2));
        assert_eq!((p.image.min_image_tokens, p.image.max_image_tokens), (16, 8000));
        assert_eq!(p.image.mean, [0.48145466, 0.4578275, 0.40821073]);
        assert_eq!(p.image.std, [0.26862954, 0.26130258, 0.27577711]);
        assert!(p.image.rescale);
        let v = p.video.unwrap();
        assert_eq!((v.max_image_tokens, v.fps), (240_000, 2.0));
    }

    #[test]
    fn a_patch_expand_factor_other_than_one_is_refused() {
        // the real processor says 1 for both; anything else would change the patches a token covers
        let p: Value = serde_json::from_str(PROCESSOR).unwrap();
        assert_eq!(p["image_processor"]["patch_expand_factor"], 1);
        assert_eq!(p["video_processor"]["patch_expand_factor"], 1);
        for section in ["image_processor", "video_processor"] {
            let mut bad = p.clone();
            bad[section]["patch_expand_factor"] = json!(2);
            let e = GlmConfig::processor_from_json(&bad).unwrap_err();
            assert!(e.contains(section) && e.contains("patch_expand_factor"), "{e}");
        }
    }

    #[test]
    fn from_dir_reads_the_processor_beside_the_config() {
        let dir = std::env::temp_dir().join(format!("rexl3-glm-from-dir-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), CONFIG).unwrap();
        // no processor file: the config still loads, with none
        assert!(GlmConfig::from_dir(&dir).unwrap().processor.is_none());
        // the combined layout
        std::fs::write(dir.join("processor_config.json"), PROCESSOR).unwrap();
        assert_eq!(GlmConfig::from_dir(&dir).unwrap().processor.unwrap().image.max_image_tokens, 8000);
        // the split layout of the published pack (preprocessor_config.json + video_preprocessor_config.json)
        std::fs::remove_file(dir.join("processor_config.json")).unwrap();
        let p: Value = serde_json::from_str(PROCESSOR).unwrap();
        std::fs::write(dir.join("preprocessor_config.json"), p["image_processor"].to_string()).unwrap();
        std::fs::write(dir.join("video_preprocessor_config.json"), p["video_processor"].to_string()).unwrap();
        let split = GlmConfig::from_dir(&dir).unwrap().processor.unwrap();
        assert_eq!((split.image.max_image_tokens, split.video.unwrap().max_image_tokens), (8000, 240_000));
        // a processor that disagrees with the tower is refused
        let mut bad = p["image_processor"].clone();
        bad["patch_size"] = json!(16);
        std::fs::write(dir.join("preprocessor_config.json"), bad.to_string()).unwrap();
        assert!(GlmConfig::from_dir(&dir).unwrap_err().contains("patch_size"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_config_has_no_grouped_query_fields_to_misread() {
        // head_dim is 0 in the real config and GlmConfig neither reads nor stores it; a different value changes nothing
        let a = real();
        let mut v = real_json();
        v["text_config"]["head_dim"] = json!(64);
        assert_eq!(GlmConfig::from_json(&v).unwrap(), a);
    }

    // ---- strict validation: each of these must be an error, and the message must say what is wrong

    #[test]
    fn a_wrong_composition_is_refused() {
        assert!(err_with(|v| v["text_config"]["num_hidden_layers"] = json!(44)).contains("num_hidden_layers"));
        assert!(err_with(|v| v["text_config"]["layer_types"][0] = json!("full_attention")).contains("unsupported layer type"));
        assert!(err_with(|v| v["text_config"]["mlp_layer_types"][3] = json!("moe")).contains("unsupported mlp layer type"));
        assert!(err_with(|v| v["text_config"]["indexer_types"][3] = json!("sometimes")).contains("unsupported indexer type"));
        // the lists beside layer_types must say what layer_types says
        assert!(err_with(|v| v["text_config"]["linear_attn_config"]["kda_layers"][3] = json!(3)).contains("disagree"));
        assert!(err_with(|v| v["text_config"]["linear_attn_config"]["full_attn_layers"] = json!([3, 7])).contains("disagree"));
        // first_k_dense_replace and mlp_layer_types must agree
        assert!(err_with(|v| v["text_config"]["first_k_dense_replace"] = json!(4)).contains("first_k_dense_replace"));
        assert!(err_with(|v| v["text_config"]["mlp_layer_types"][1] = json!("sparse")).contains("layer 1"));
        // the first attention layer cannot reuse a selection nobody made
        assert!(err_with(|v| v["text_config"]["indexer_types"][3] = json!("shared")).contains("cannot share"));
    }

    #[test]
    fn what_the_engine_does_not_implement_is_refused() {
        assert!(err_with(|v| v["text_config"]["qk_rope_head_dim"] = json!(64)).contains("NoPE"));
        assert!(err_with(|v| v["text_config"]["mla_use_nope"] = json!(false)).contains("mla_use_nope"));
        assert!(err_with(|v| v["text_config"]["mhc"] = json!(false)).contains("`text_config.mhc`"));
        assert!(err_with(|v| v["text_config"]["index_kpool_compress"] = json!(false)).contains("index_kpool_compress"));
        assert!(err_with(|v| v["text_config"]["scoring_func"] = json!("softmax")).contains("scoring_func"));
        assert!(err_with(|v| v["text_config"]["topk_method"] = json!("greedy")).contains("topk_method"));
        assert!(err_with(|v| v["text_config"]["n_group"] = json!(8)).contains("n_group"));
        assert!(err_with(|v| v["text_config"]["hidden_act"] = json!("gelu")).contains("hidden_act"));
        assert!(err_with(|v| v["text_config"]["num_nextn_predict_layers"] = json!(3)).contains("one MTP layer"));
        assert!(err_with(|v| v["text_config"]["num_experts_per_tok"] = json!(289)).contains("exceeds"));
    }

    #[test]
    fn missing_or_malformed_fields_name_themselves() {
        assert!(err_with(|v| {
            v.as_object_mut().unwrap().remove("text_config");
        })
        .contains("text_config"));
        assert!(err_with(|v| {
            v["text_config"].as_object_mut().unwrap().remove("kv_lora_rank");
        })
        .contains("kv_lora_rank"));
        assert!(err_with(|v| v["text_config"]["kv_lora_rank"] = json!(0)).contains("kv_lora_rank"));
        assert!(err_with(|v| v["text_config"]["hidden_size"] = json!("4096")).contains("hidden_size"));
        assert!(err_with(|v| v["text_config"]["rms_norm_eps"] = json!(0.0)).contains("rms_norm_eps"));
        assert!(err_with(|v| v["text_config"]["qk_head_dim"] = json!(128)).contains("qk_head_dim"));
    }

    #[test]
    fn a_vision_tower_that_cannot_feed_the_language_model_is_refused() {
        let e = err_with(|v| v["vision_config"]["out_hidden_size"] = json!(2048));
        assert!(e.contains("out_hidden_size") && e.contains("spliced"), "{e}");
        assert!(err_with(|v| v["image_token_id"] = json!(154_880)).contains("outside the vocabulary"));
        assert!(err_with(|v| v["video_token_id"] = v["image_token_id"].clone()).contains("distinct"));
        assert!(err_with(|v| v["vision_config"]["hidden_size"] = json!(1000)).contains("multiple"));
        // a pack without a tower is a text-only pack, not an error
        let mut v = real_json();
        v.as_object_mut().unwrap().remove("vision_config");
        assert!(GlmConfig::from_json(&v).unwrap().vision.is_none());
    }

    #[test]
    fn kv_layout_follows_the_indexer_mode_of_each_layer() {
        // a layer that shares its indexer caches the latent only: 1,024 B a token against 1,088 (the pooled plane of the layer with its own indexer is 64 B a token; the raw rows are a per-slot ring)
        let mut v = real_json();
        v["text_config"]["indexer_types"][7] = json!("shared");
        let c = GlmConfig::from_json(&v).unwrap();
        assert_eq!(c.kv_layout(false).bytes_per_token("glm", 256), Ok(10 * 1088 + 1024));
        // with the MTP head (always a full indexer)
        assert_eq!(c.kv_layout(true).bytes_per_token("glm", 256), Ok(11 * 1088 + 1024));
        // no pooling: the indexer plane is the bare key (128), no pool plane: 1024 + 256 = 1280
        v["text_config"]["indexer_types"][7] = json!("full");
        v["text_config"]["index_kpool"] = json!(0);
        let c = GlmConfig::from_json(&v).unwrap();
        assert_eq!(c.kv_layout(false).bytes_per_token("glm", 256), Ok(11 * 1280));
    }

    #[test]
    fn a_pool_that_does_not_divide_the_page_cannot_be_priced() {
        let mut v = real_json();
        v["text_config"]["index_kpool"] = json!(3);
        let c = GlmConfig::from_json(&v).unwrap();
        assert!(matches!(c.kv_layout(false).bytes_per_page("glm", 256), Err(gb_sched::MemoryError::Inconsistent(_))));
    }
}
