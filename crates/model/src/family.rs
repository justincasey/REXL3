//! The model-family contract and the open registry the engine dispatches through.
//!
//! One engine serves a ladder of models, and the ladder grows as families are made. The engine core never names a family: a
//! family is a [`ModelFamily`], listed in one place (`crate::families::FAMILIES`), and everything the core needs from it comes
//! through that trait:
//!
//! * [`ModelFamily::architectures`] / [`ModelFamily::model_types`]: the names it accepts (`config.json` `architectures[0]`, with
//!   `model_type` consulted only when `architectures` is absent or empty);
//! * [`ModelFamily::probe`]: read and validate the pack's config into a [`FamilyPlan`]: the layer recipe, the memory model, the
//!   vision component or none, the chat format, the drafters offered, and the components the family requires;
//! * [`ModelFamily::build`]: start that family's engine. The backend type is erased here and nowhere else, once, at startup: the
//!   family returns a `Box<dyn ServeEngine>` and every `Backend` call inside its `Engine<B>` stays monomorphised.
//!
//! **Adding a family** is one module under `crate::families`, one line in `FAMILIES`, its fixtures and its tests; nothing in the
//! scheduler, admission, sampler, speculation logic or server changes. Components the family needs that the library lacks are
//! named in its `required_components` and stay out of [`BUILT_COMPONENTS`] until they are built, so a family can be probed,
//! planned and reported long before it can run.
//!
//! An architecture no registered family accepts is refused with a [`CapabilityReport`], never a vague failure and never a silent
//! fallback: the closest registered family, and which of the components the config calls for do not exist. `--allow-unvalidated-pack`
//! relaxes the pack allowlist and never this check.

use crate::error::ModelError;
use crate::memory::MemoryOptions;
use crate::serve::ServeOptions;
use gb_engine::ServeEngine;
use gb_sched::{MemoryError, MemoryProfile};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

// ------------------------------------------------------------------------------------------------------------- the recipe

/// How a layer mixes tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mixer {
    /// Nemotron-H Mamba2 SSM with a fixed-size state (the `M` blocks of Nemotron 3.5 Super).
    Mamba2,
    /// Attention without RoPE (the `*` blocks of the Nemotron-H family).
    GqaNope,
    /// No token mixer: the block is only its FFN (the `E` and `-` blocks in a Nemotron-H pattern).
    None,
    /// Grouped-query softmax attention with a paged K and V cache (Qwen3.5 full-attention layers).
    Gqa,
    /// Gated delta net, a linear-attention recurrence with a fixed-size state (Qwen3.5 linear layers).
    Gdn,
    /// Kimi delta attention, a delta rule with a per-channel decay (GLM-5.3 linear layers).
    Kda,
    /// Multi-head latent attention without RoPE, scored through the DeepSeek sparse-attention indexer (GLM-5.3 attention layers).
    MlaDsa { indexer: IndexerMode },
}

/// Whether a sparse-attention layer scores its own selection (and so caches indexer planes) or reuses the nearest preceding
/// full layer's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexerMode {
    Full,
    Shared,
}

/// The feed-forward block of a layer. The mixture-of-experts parameters are uniform across a model and live in its config.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ffn {
    Dense,
    /// No FFN: the block is only its token mixer (Nemotron-H `M` and `*` blocks).
    None,
    /// Nemotron-H latent MoE: routed experts run at a latent width behind a down-projection, with a ReLU-squared activation.
    LatentMoeRelu2,
    /// Routed experts plus shared experts behind a router.
    Moe,
}

/// How the layer's output joins the residual stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residual {
    Plain,
    /// Manifold-constrained hyper-connections: `streams` residual streams mixed by a Sinkhorn-normalised matrix.
    Mhc {
        streams: usize,
    },
}

/// One layer: a mixer, a feed-forward block and a residual scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerSpec {
    pub mixer: Mixer,
    pub ffn: Ffn,
    pub residual: Residual,
}

// ---------------------------------------------------------------------------------------------------------- the components

/// The components that exist: kernels, state managers, towers and drafters that are built and tested. A family's
/// `required_components` that are not listed here are what stands between its probe and its engine. Move a name here in the
/// commit that lands the component (with its row-invariance test, golden hash and oracle comparison).
pub const BUILT_COMPONENTS: &[&str] = &[
    "exl3_linear",
    "gqa_attention",
    "gdn",
    "dense_mlp",
    "plain_residual",
    "mrope",
    "vision_qwen3",
    "dflash2",
    "copy_drafter",
    // GLM-5.3-Flash: the serial forward, the NextN drafter (docs/GLM53.md) and the vision tower (`families::glm::vision`) all run
    "vision_glm5",
    "kda",
    "mla_dsa",
    "mla_gated",
    "mhc",
    "moe",
    // Flash-Next text forward, indexer, PLE, residual and MTP head; vision remains unbuilt.
    "gqa_attention_kv2",
    "qsa_indexer",
    "ngram_embedding",
    "hc_lowrank",
    "moe_softmax_topk",
    "mtp_flash_next",
    // Nemotron 3.5 Super VL text forward: Mamba2, GQA without RoPE, latent MoE
    "nemotron_mamba2",
    "nemotron_gqa_nope",
    "nemotron_latent_moe",
    // Nemotron's MTP head: loader, KV stripe, tap stores, draft call (families::nemotron_h_omni::mtp), the backend's speculation runtime
    "nemotron_mtp",
    // Nemotron's RADIO tower and pixel-shuffle projector as the image encoder, the feature splice in the prefill slices and the head's rows
    // for image positions (families::nemotron_h_omni::vision, docs/NEMOTRON_OMNI.md "Images")
    "nemotron_radio_vision",
    // MiniCPM-V-4.7: the weights loader and the boot with no forward (families::minicpmv4_7::{weights, device, state, boot}); its
    // kernels and the forward are separate components and are not listed until they are wired and gated at the model level
    "weights_minicpmv4_7",
    // MiniCPM-V-4.7 text forward (E3, gated by the oracle against the exllamav3 fork on the 6.0 bpw pack: top-1 0.9735 / cosine 0.9984
    // over 14,336 positions): the E1 kernels at this model's shapes (they run at model level now) and the serial forward, prefill,
    // windows, backend and engine around them (`families::minicpmv4_7::{model, backend, engine}`). The tower, the draft and the prefix
    // cache are not built.
    "gdn_hv32",
    "attention_q16kv2",
    "coopmk_h2048_i512",
    "exl3_pins_minicpmv",
    "topk_pad_mask",
    "forward_minicpmv4_7",
    // MiniCPM-V-4.7 DSpark draft in the serving engine (E4b, gated on the real pack: the target's bits with a draft attached, the taps
    // against the residual, drafted == serial greedy and sampled for one and several streams, the draft's context after partly
    // accepted windows against the serial path, plan == allocations; section 15 of docs/MINICPMV4_7_INTEGRATION_PLAN.md): the
    // five-layer non-causal block (8 query heads over 2 KV heads of 256) over runtime taps with its paged bf16 cache, the keyed
    // selector walk over the top-K candidates and the confidences that become `Proposal.accept`
    // (`families::minicpmv4_7::{dspark, spec}`).
    "attention_q8kv2_draft",
    "dspark_drafter",
    "dspark_selector_walk",
    "dspark_confidence_head",
    // MiniCPM-V-4.7 images and video (E5b; gated by the prompt ids and canvas positions equal to the Hugging Face processor's and
    // `get_rope_index` on procedural image, multi-image and video prompts, and the engine against the HF bf16 reference on them):
    // the 27-layer tower with its window merger as the image encoder, the canvas M-RoPE positions of the whole prompt, the
    // multi-slice preprocessing and the image layout of the chat format (`families::minicpmv4_7::{vision, splice}`,
    // `server::formats::minicpmv`).
    "vision_minicpmv",
    "mrope_canvas",
    "image_slicing_minicpmv",
    "chat_minicpmv",
    // DeepSeek-V4.1-Flash: FP8 MLA, CSA2 compressor, DSA indexer, noaux_tc MoE, mHC streams, Engram, DSpark NextN
    "mla_fp8",
    "dsa_compressor",
    "dsa_index",
    "moe_noaux_tc",
    "hc_streams",
    "engram",
    "mtp_dspark",
    // DeepSeek-V4.1-Flash vision tower (`families::deepseek_v41::vision`): the 32-block ViT with its 2-D RoPE and
    // bidirectional attention, the 3 x 3 aligner, the span splice into the prefill's embedding rows and the engram
    // DEAD mask over image spans; the preprocessing is `gb_vision::dsv4`. The kernels are `csrc/dsv4_vision.cu`
    // (host twins in `gb_cuda::dsv4_vision`), and the numerics gates are `crates/cuda/tests/dsv4_vision.rs`.
    "vision_deepseek",
];

/// Components of `required` that are not built, in the order given.
pub fn missing_components(required: &[&'static str]) -> Vec<&'static str> {
    required.iter().copied().filter(|c| !BUILT_COMPONENTS.contains(c)).collect()
}

/// Which components a config calls for, read from its keys alone (used for an architecture no family accepts, where there is no
/// probe to ask). Coarse on purpose: it names the capability a model needs, not the family that has it. The text and vision
/// sections are both searched.
pub fn infer_components(root: &Value) -> Vec<&'static str> {
    let sections: Vec<&Value> = [Some(root), root.get("text_config")].into_iter().flatten().collect();
    let has = |k: &str| sections.iter().any(|s| s.get(k).is_some_and(|v| !v.is_null()));
    let positive = |k: &str| sections.iter().any(|s| s.get(k).and_then(|v| v.as_u64()).unwrap_or(0) > 0);
    let layer_type = |t: &str| {
        sections
            .iter()
            .any(|s| s.get("layer_types").and_then(|v| v.as_array()).is_some_and(|a| a.iter().any(|x| x.as_str() == Some(t))))
    };
    let mut out = Vec::new();
    let mut want = |name: &'static str, cond: bool| {
        if cond && !out.contains(&name) {
            out.push(name);
        }
    };
    want(
        "sliding_attn",
        layer_type("sliding_attention")
            || sections.iter().any(|s| s.get("sliding_window").is_some_and(|v| v.as_u64().unwrap_or(0) > 0)),
    );
    want("mla", has("kv_lora_rank"));
    want("dsa_indexer", has("index_topk"));
    want("moe", has("n_routed_experts") || has("num_experts") || has("num_local_experts"));
    want("mtp", positive("num_nextn_predict_layers") || positive("mtp_num_hidden_layers"));
    want("mhc", has("hc_mult"));
    want("ple", has("hidden_size_per_layer_input"));
    want("kda", sections.iter().any(|s| s.get("linear_attn_config").is_some_and(|c| c.get("kda_layers").is_some())));
    want("gdn", has("linear_num_key_heads"));
    want("vision_tower", root.get("vision_config").is_some());
    out
}

// ------------------------------------------------------------------------------------------------------------------ plan

/// The memory model of a probed pack, evaluated at serving settings the caller chooses.
pub trait MemorySource: Send + Sync {
    /// The validated profile at these settings. Every shape that would price a token at zero bytes is an error here, before the
    /// planner or admission exist.
    fn profile(&self, opts: &MemoryOptions) -> Result<MemoryProfile, MemoryError>;
}

/// A family's vision component, as far as the engine core needs to know it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionSpec {
    /// The tower's name (`vision_qwen3`, `vision_glm5`).
    pub tower: &'static str,
    pub patch_size: usize,
    /// Patches per side folded into one language-model token.
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    /// Width of the embeddings handed to the language model.
    pub out_hidden: usize,
    pub image_token_id: u32,
    /// The most language-model tokens one image takes unless the operator lowers it (`--max-image-tokens`): the family's own cap (GLM:
    /// the processor's `max_image_tokens`; Qwen: the server's long-standing 4,096, the tower's scratch is sized for it).
    pub max_image_tokens: usize,
}

/// What a family learned from a pack's configuration: everything the engine core is allowed to know about the model.
pub struct FamilyPlan {
    /// The family's id.
    pub family: &'static str,
    pub dir: PathBuf,
    /// The trunk's layers, in order.
    pub layers: Vec<LayerSpec>,
    /// Layers after the trunk that predict further tokens (NextN/MTP), in order.
    pub mtp_layers: Vec<LayerSpec>,
    pub memory: Box<dyn MemorySource>,
    pub vision: Option<VisionSpec>,
    /// The chat format (template, thinking markers, tool-call parser) this family's prompts use.
    pub chat_format: &'static str,
    /// The drafters this family can run (`dflash2`, `mtp`, `copy`).
    pub drafters: Vec<&'static str>,
    pub required_components: Vec<&'static str>,
}

impl FamilyPlan {
    /// Required components that are not built.
    pub fn missing_components(&self) -> Vec<&'static str> {
        missing_components(&self.required_components)
    }

    /// The vision tower the family declares when it is not built: a family can serve text long before its tower exists, and a server
    /// asked for images then must refuse, naming the tower (see `serve::preflight`). `None` when there is no tower or it is built.
    pub fn unbuilt_vision(&self) -> Option<&'static str> {
        self.vision.as_ref().map(|v| v.tower).filter(|t| !BUILT_COMPONENTS.contains(t))
    }

    pub fn memory_profile(&self, opts: &MemoryOptions) -> Result<MemoryProfile, MemoryError> {
        self.memory.profile(opts)
    }

    /// The error a family returns from `build` while components it requires do not exist.
    pub fn not_built(&self, what: &str) -> ModelError {
        ModelError::NotBuilt(format!("{what}: needs {}: not built", self.missing_components().join(", ")))
    }
}

/// What `build` is given besides the plan.
pub struct BuildCtx<'a> {
    pub options: &'a ServeOptions,
}

/// A model family. The whole contract between the engine core and a model.
pub trait ModelFamily: Sync {
    /// Short stable name for logs and boot facts.
    fn id(&self) -> &'static str;
    /// `config.json` `architectures[0]` names this family accepts.
    fn architectures(&self) -> &'static [&'static str];
    /// `config.json` `model_type` names accepted when a config has no `architectures`.
    fn model_types(&self) -> &'static [&'static str] {
        &[]
    }
    /// Parse and validate the pack at `dir`.
    fn probe(&self, dir: &Path) -> Result<FamilyPlan, FamilyError>;
    /// Start this family's engine from a plan it produced. Runs on the engine thread (the CUDA context lives where it is used).
    fn build(&self, plan: FamilyPlan, ctx: &BuildCtx<'_>) -> Result<Box<dyn ServeEngine>, ModelError>;
    /// The family's image preprocessing for a server limited to `o.max_image_tokens` tokens an image: decode, canvas and token count, patch
    /// layout (`gb_vision::preprocessor::ImagePreprocessor`). `None` for a family with no vision component the server can feed.
    fn image_preprocessor(
        &self,
        _dir: &Path,
        _o: &ServeOptions,
    ) -> Result<Option<std::sync::Arc<dyn gb_vision::preprocessor::ImagePreprocessor>>, ModelError> {
        Ok(None)
    }
    /// Whether this family's vision tower also encodes video (a list of frames as tubelets: `serve::ImageEncoder::encode_video`). **`false`
    /// unless a family builds it**; the server turns video input on only for a family that says `true`, when vision is on.
    fn encodes_video(&self) -> bool {
        false
    }
    /// Whether this family's engine can snapshot and restore a prompt's state, so that a prefix cache may be enabled for it (`--prefix-cache`,
    /// `--prefix-cache-mib`). **`false` unless a family proves otherwise** (`docs/PREFIX_CACHE.md`): the serve starter turns the cache off in the
    /// options it hands `build` for a family that says `false`, whatever was asked, so a family cannot get one by accident. Qwen and GLM-5.3-Flash
    /// say `true`.
    fn prefix_cache(&self) -> bool {
        false
    }
    /// Whether the family sizes its verify window by `ServeOptions::window_rows` (`--window-rows`). A family that does not says so by
    /// leaving this `false`: the start refuses an explicit non-default value for it by name instead of ignoring the flag. Opt in only when
    /// `build` honours the option (its records, scratch and row budget) and the row-invariance tests cover the widest window.
    fn window_rows_adjustable(&self) -> bool {
        false
    }
    /// Whether the family proves the EXL3 format from its tensors instead of from `config.json`. A checkpoint whose config carries no
    /// `quantization_config` (the Nemotron 3.5 Super VL pack) is refused by the shared check unless the family says so; then its own probe
    /// must find the EXL3 tensors and refuse the checkpoint if they are missing. `false` for every family that reads the config.
    fn quantization_from_tensors(&self) -> bool {
        false
    }
    /// The widest `--prefill-chunk` the start lets through to this family (`serve::check_options_for`, before anything is loaded): the Qwen
    /// prefill kernels' 4,096 rows ([`crate::serve::DEFAULT_MAX_PREFILL_CHUNK`]) unless the family's own check and kernels take more.
    fn max_prefill_chunk(&self) -> usize {
        crate::serve::DEFAULT_MAX_PREFILL_CHUNK
    }
    /// The most ranks of a tensor-parallel group (`--tp-ranks N`, `docs/NEMOTRON_TP.md`, `docs/CLUSTER_FAMILY_PORTING.md`) this family's
    /// runtime supports: `1` (the default) means the family has none and the start refuses a group for it by name; a family that built
    /// one says how many ranks its shard plan, memory plan and tests cover. A cap is a statement about what the family can shard to,
    /// not about what has been measured: say so in its documentation.
    fn max_tp_ranks(&self) -> usize {
        1
    }
    /// Whether the family has a runtime for a tensor-parallel group of processes: [`ModelFamily::max_tp_ranks`] above one.
    fn tensor_parallel(&self) -> bool {
        self.max_tp_ranks() > 1
    }
    /// Runs this process as the follower of a tensor-parallel group: load this rank's shard and replay the leader's calls until it shuts
    /// down. Only called for a family whose [`ModelFamily::max_tp_ranks`] is above one.
    fn run_follower(&self, _plan: FamilyPlan, _ctx: &BuildCtx<'_>) -> Result<(), ModelError> {
        Err(ModelError::NotBuilt(format!("the {} family has no tensor-parallel follower", self.id())))
    }
}

// ----------------------------------------------------------------------------------------------------------------- errors

/// Why an architecture was refused, and what it would take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilityReport {
    /// `architectures[0]`, when the config has one.
    pub architecture: Option<String>,
    pub model_type: Option<String>,
    /// The registered family whose names are closest, when any shares a prefix of at least four characters.
    pub closest: Option<&'static str>,
    /// Components the config calls for (by its keys), whether or not they are built.
    pub inferred: Vec<&'static str>,
    /// Of those, the ones that do not exist.
    pub missing: Vec<&'static str>,
    /// Every architecture a registered family accepts.
    pub served: Vec<&'static str>,
}

impl std::fmt::Display for CapabilityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mt = |m: &Option<String>| m.as_ref().map(|m| format!("`{m}`")).unwrap_or_else(|| "absent".into());
        match (&self.architecture, &self.model_type) {
            (Some(a), _) => write!(f, "unsupported model architecture `{a}` (model_type {})", mt(&self.model_type))?,
            (None, Some(m)) => write!(f, "unsupported model type `{m}` (config.json has no `architectures`)")?,
            (None, None) => write!(f, "config.json names neither `architectures` nor `model_type`")?,
        }
        match self.closest {
            Some(c) => write!(f, "; closest served family: {c}")?,
            None => write!(f, "; no served family is close")?,
        }
        if self.inferred.is_empty() {
            write!(f, "; nothing in its config maps to a known component")?;
        } else if self.missing.is_empty() {
            write!(
                f,
                "; its config calls for {} (all built, but no family accepts this architecture)",
                self.inferred.join(", ")
            )?;
        } else {
            write!(f, "; its config calls for {} and needs {}: not built", self.inferred.join(", "), self.missing.join(", "))?;
        }
        write!(f, "; this release serves {} (--allow-unvalidated-pack does not relax this)", self.served.join(", "))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FamilyError {
    /// A config that cannot be read or fails validation.
    Config(String),
    /// No registered family accepts the architecture.
    Unsupported(Box<CapabilityReport>),
    /// The registry itself is malformed (two families claim one name): nothing is dispatched.
    Registry(String),
}

impl std::fmt::Display for FamilyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FamilyError::Config(m) | FamilyError::Registry(m) => write!(f, "{m}"),
            FamilyError::Unsupported(r) => write!(f, "{r}"),
        }
    }
}

impl std::error::Error for FamilyError {}

impl From<FamilyError> for ModelError {
    fn from(e: FamilyError) -> Self {
        ModelError::Config(e.to_string())
    }
}

// -------------------------------------------------------------------------------------------------------------- registry

/// This engine serves EXL3 packs only (scope: families within the EXL3 format; other quantisation formats are a separate axis).
/// Refuses a config whose `quantization_config.quant_method` (top level or `text_config`, as the pack scanner reads it) is missing
/// or not `exl3`, naming what was found. Run once, by [`Registry::probe`], before any family looks at the pack.
pub fn require_exl3(root: &Value) -> Result<(), FamilyError> {
    if gb_exl3::checkpoint::config_fields(root).is_some() {
        return Ok(());
    }
    let mut found: Vec<String> = Vec::new();
    for section in [Some(root), root.get("text_config")].into_iter().flatten() {
        for key in ["quantization_config", "quantization"] {
            if let Some(m) = section.get(key).and_then(|b| b.get("quant_method")).and_then(|m| m.as_str()) {
                found.push(m.to_string());
            }
        }
    }
    let what = match found.first() {
        Some(m) => format!("found quant_method={m}"),
        None => "found no quantization_config with a quant_method".to_string(),
    };
    Err(FamilyError::Config(format!("this engine serves EXL3 packs only; {what}")))
}

/// The registered families. Built once from a list; a list in which two families claim one name is refused whole, and a registry
/// built that way refuses every pack (fail closed) instead of picking one.
pub struct Registry {
    families: Vec<&'static dyn ModelFamily>,
    broken: Option<String>,
}

impl std::fmt::Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Registry")
            .field("families", &self.families.iter().map(|x| x.id()).collect::<Vec<_>>())
            .field("broken", &self.broken)
            .finish()
    }
}

fn normalise(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect()
}

fn common_prefix(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

impl Registry {
    /// Refuses a list in which a family has no id or accepts no architecture, or two families share an id, an architecture or a
    /// model type; the error names the colliding name and both families.
    pub fn new(families: &[&'static dyn ModelFamily]) -> Result<Self, String> {
        let mut ids: Vec<(&str, &str)> = Vec::new();
        let mut arch: Vec<(&str, &str)> = Vec::new();
        let mut types: Vec<(&str, &str)> = Vec::new();
        for fam in families {
            let id = fam.id();
            if id.is_empty() {
                return Err("a family has an empty id".into());
            }
            if fam.architectures().is_empty() {
                return Err(format!("family `{id}` accepts no architecture, so nothing could reach it"));
            }
            for (list, names, what) in [
                (&mut ids, &[id][..], "id"),
                (&mut arch, fam.architectures(), "architecture"),
                (&mut types, fam.model_types(), "model type"),
            ] {
                for &name in names {
                    if let Some((_, other)) = list.iter().find(|(n, _)| *n == name) {
                        return Err(format!("{what} `{name}` is claimed by both `{other}` and `{id}`"));
                    }
                    list.push((name, id));
                }
            }
        }
        Ok(Self { families: families.to_vec(), broken: None })
    }

    /// The registry of `crate::families::FAMILIES`. If that list is malformed the registry refuses every pack with the reason.
    pub fn global() -> &'static Registry {
        static GLOBAL: OnceLock<Registry> = OnceLock::new();
        GLOBAL.get_or_init(|| {
            Registry::new(crate::families::FAMILIES).unwrap_or_else(|e| Registry { families: Vec::new(), broken: Some(e) })
        })
    }

    pub fn families(&self) -> &[&'static dyn ModelFamily] {
        &self.families
    }

    /// Every architecture a registered family accepts.
    pub fn served(&self) -> Vec<&'static str> {
        self.families.iter().flat_map(|f| f.architectures().iter().copied()).collect()
    }

    /// Which family a config belongs to. `architectures[0]` decides; `model_type` is consulted only when there is no
    /// `architectures` (or it is empty). An architecture that is present and unknown is never rescued by a known `model_type`.
    pub fn detect(&self, root: &Value) -> Result<&'static dyn ModelFamily, FamilyError> {
        if let Some(why) = &self.broken {
            return Err(FamilyError::Registry(format!("the family registry is malformed and refuses every pack: {why}")));
        }
        let model_type = root.get("model_type").and_then(|x| x.as_str());
        let architecture = match root.get("architectures") {
            Some(Value::Array(a)) if !a.is_empty() => {
                Some(a[0].as_str().ok_or_else(|| FamilyError::Config("config.json: `architectures[0]` is not a string".into()))?)
            }
            Some(Value::Array(_)) | None => None,
            Some(_) => return Err(FamilyError::Config("config.json: `architectures` is not a list".into())),
        };
        let found = match architecture {
            Some(a) => self.families.iter().find(|f| f.architectures().contains(&a)),
            None => model_type.and_then(|m| self.families.iter().find(|f| f.model_types().contains(&m))),
        };
        match found {
            Some(f) => Ok(*f),
            None => Err(FamilyError::Unsupported(Box::new(self.report(root, architecture, model_type)))),
        }
    }

    /// The capability report for a config no family accepts.
    pub fn report(&self, root: &Value, architecture: Option<&str>, model_type: Option<&str>) -> CapabilityReport {
        let inferred = infer_components(root);
        let missing = missing_components(&inferred);
        // closest by shared name prefix: "Qwen3ForCausalLM" is near qwen3_5, "Gemma4ForConditionalGeneration" is near nothing (a shared
        // suffix such as ForConditionalGeneration says nothing about the architecture)
        let probe = normalise(architecture.or(model_type).unwrap_or(""));
        let mut best: Option<(usize, &'static str)> = None;
        for f in &self.families {
            let names = f.architectures().iter().chain(f.model_types()).copied().chain([f.id()]);
            let score = names.map(|n| common_prefix(&probe, &normalise(n))).max().unwrap_or(0);
            if score >= 4 && best.is_none_or(|(s, _)| score > s) {
                best = Some((score, f.id()));
            }
        }
        CapabilityReport {
            architecture: architecture.map(str::to_string),
            model_type: model_type.map(str::to_string),
            closest: best.map(|(_, id)| id),
            inferred,
            missing,
            served: self.served(),
        }
    }

    /// Detect the family of the pack at `dir` and have it probe the pack. The checks every family would otherwise repeat run first:
    /// the pack must be EXL3 ([`require_exl3`]), and a family must accept its architecture ([`Registry::detect`]).
    pub fn probe(&self, dir: &Path) -> Result<(&'static dyn ModelFamily, FamilyPlan), FamilyError> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| FamilyError::Config(format!("{}: {e}", path.display())))?;
        let root: Value = serde_json::from_str(&text).map_err(|e| FamilyError::Config(format!("config.json: {e}")))?;
        // The config must name EXL3, unless the family that claims this architecture proves it from the tensors.
        let exl3 = require_exl3(&root);
        let family = self.detect(&root);
        let family = match (exl3, family) {
            (Ok(()), family) => family?,
            (Err(_), Ok(f)) if f.quantization_from_tensors() => f,
            (Err(e), _) => return Err(e),
        };
        let plan = family.probe(dir)?;
        Ok((family, plan))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gb_engine::mock::{Fault, MockBackend};
    use gb_engine::{Engine, EngineConfig, NoDrafts};
    use gb_sched::{Admission, CostModel, KvLayerClass, KvLayout, MemoryModel, ModelMemory, StepShape, VisionMemory};
    use serde_json::json;

    const QWEN_CONFIG: &str = include_str!("../../../testdata/qwen38/config.json");
    const GLM_CONFIG: &str = include_str!("../../../testdata/glm53/config.json");

    fn dir_with(tag: &str, config: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rexl3-family-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), config).unwrap();
        dir
    }

    // ---- a third family, registered only here: the registry dispatches it without any engine-core file knowing it exists

    /// A toy family: `toy_layers` full-attention layers of `toy_kv_heads` x `toy_head_dim`, served by the mock backend. The names it
    /// accepts and the components it requires are fields, so one definition serves the well-behaved and the misbehaving cases.
    struct Toy {
        id: &'static str,
        archs: &'static [&'static str],
        types: &'static [&'static str],
        required: &'static [&'static str],
        /// What `ModelFamily::prefix_cache` answers.
        cache: bool,
        /// answers `window_rows_adjustable`
        wide: bool,
    }

    /// The `prefix_cache` option the last `build` of each toy family (by id) was handed: keyed by id because tests run in parallel and several build a toy.
    static TOY_SAW: std::sync::Mutex<Vec<(&'static str, crate::serve::PrefixCacheOption)>> = std::sync::Mutex::new(Vec::new());
    fn toy_saw(id: &str) -> Option<crate::serve::PrefixCacheOption> {
        TOY_SAW.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(i, _)| *i == id).map(|(_, o)| *o)
    }
    fn toy_forget(id: &str) {
        TOY_SAW.lock().unwrap_or_else(|e| e.into_inner()).retain(|(i, _)| *i != id);
    }

    struct ToyMemory {
        layers: u64,
        kv_heads: u64,
        head_dim: u64,
    }

    impl MemoryModel for ToyMemory {
        fn family(&self) -> &str {
            "toy"
        }
        fn kv_layout(&self) -> KvLayout {
            KvLayout::new().with(KvLayerClass::gqa(self.layers, self.kv_heads, self.head_dim, 2))
        }
        fn recurrent_layers(&self) -> u64 {
            0
        }
        fn state_bytes_per_seq(&self) -> u64 {
            0
        }
        fn window_record_bytes_per_row(&self) -> u64 {
            0
        }
        fn scratch_bytes(&self, rows: u64) -> u64 {
            rows * 1024
        }
        fn activation_reserve(&self) -> u64 {
            1 << 20
        }
        fn drafter_state_bytes_per_slot(&self) -> u64 {
            0
        }
        fn vision(&self) -> Option<VisionMemory> {
            None
        }
    }

    impl MemorySource for ToyMemory {
        fn profile(&self, opts: &MemoryOptions) -> Result<MemoryProfile, MemoryError> {
            let shape =
                StepShape { page_tokens: 256, window_rows: opts.window_rows as u64, scratch_rows: opts.prefill_rows as u64 };
            MemoryProfile::new(self, shape)
        }
    }

    impl ModelFamily for Toy {
        fn id(&self) -> &'static str {
            self.id
        }
        fn architectures(&self) -> &'static [&'static str] {
            self.archs
        }
        fn model_types(&self) -> &'static [&'static str] {
            self.types
        }
        fn prefix_cache(&self) -> bool {
            self.cache
        }
        fn window_rows_adjustable(&self) -> bool {
            self.wide
        }
        fn probe(&self, dir: &Path) -> Result<FamilyPlan, FamilyError> {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
            let get = |k: &str| v[k].as_u64().ok_or_else(|| FamilyError::Config(format!("toy config: missing `{k}`")));
            let (layers, kv_heads, head_dim) = (get("toy_layers")?, get("toy_kv_heads")?, get("toy_head_dim")?);
            Ok(FamilyPlan {
                family: self.id,
                dir: dir.to_path_buf(),
                layers: (0..layers)
                    .map(|_| LayerSpec { mixer: Mixer::Gqa, ffn: Ffn::Dense, residual: Residual::Plain })
                    .collect(),
                mtp_layers: vec![],
                memory: Box::new(ToyMemory { layers, kv_heads, head_dim }),
                vision: None,
                chat_format: "toy",
                drafters: vec!["copy"],
                required_components: self.required.to_vec(),
            })
        }
        fn build(&self, plan: FamilyPlan, ctx: &BuildCtx<'_>) -> Result<Box<dyn ServeEngine>, ModelError> {
            {
                let mut saw = TOY_SAW.lock().unwrap_or_else(|e| e.into_inner());
                saw.retain(|(i, _)| *i != self.id);
                saw.push((self.id, ctx.options.prefix_cache));
            }
            let opts =
                MemoryOptions { window_rows: 64, prefill_rows: 128, max_context: 4096, mtp: false, vision: None, trees: false };
            let profile = plan.memory_profile(&opts).map_err(|e| ModelError::Config(e.to_string()))?;
            let mm = ModelMemory::from_profile(&profile, 0, 0);
            let admission = Admission::new(1 << 30, 0, mm, ctx.options.slots).map_err(|e| ModelError::Config(e.to_string()))?;
            let cost = CostModel::from_table(vec![(1, 1.0), (64, 2.0)]);
            let engine = Engine::new(
                MockBackend::new(512, ctx.options.slots, Fault::None),
                admission,
                EngineConfig { max_rows: 64, max_drafts: 7, cost, drafting: false },
                Box::new(|| Box::new(NoDrafts) as Box<dyn gb_engine::Drafter>),
            );
            Ok(Box::new(engine))
        }
    }

    static TOY: Toy = Toy {
        id: "toy",
        archs: &["ToyForCausalLM"],
        types: &["toy"],
        required: &["gqa_attention", "dense_mlp"],
        cache: false,
        wide: false,
    };
    /// The toy that says it honours `--window-rows`.
    static WIDE_TOY: Toy = Toy {
        id: "wide-toy",
        archs: &["ToyForCausalLM"],
        types: &["toy"],
        required: &["gqa_attention", "dense_mlp"],
        cache: false,
        wide: true,
    };
    /// A second family that claims the toy's architecture name.
    static CLASH: Toy = Toy { id: "clash", archs: &["ToyForCausalLM"], types: &[], required: &[], cache: false, wide: false };
    /// A family that requires a component nobody has built, and whose `build` succeeds anyway: the registry must not trust it.
    static LEAKY: Toy = Toy {
        id: "leaky",
        archs: &["LeakyForCausalLM"],
        types: &[],
        required: &["gqa_attention", "nonexistent_kernel"],
        cache: false,
        wide: false,
    };
    /// A toy that does not say so (the default `prefix_cache()`), with an id of its own: the test of the clamp reads what its `build` was handed, and
    /// tests run in parallel, so no other test may build a family of that id.
    static NOCACHE: Toy = Toy {
        id: "nocache",
        archs: &["ToyForCausalLM"],
        types: &["toy"],
        required: &["gqa_attention", "dense_mlp"],
        cache: false,
        wide: false,
    };
    /// A toy that says it can be cached.
    static CACHEY: Toy = Toy {
        id: "cachey",
        archs: &["ToyForCausalLM"],
        types: &["toy"],
        required: &["gqa_attention", "dense_mlp"],
        cache: true,
        wide: false,
    };

    fn with_toy() -> Registry {
        let mut fams: Vec<&'static dyn ModelFamily> = crate::families::FAMILIES.to_vec();
        fams.push(&TOY);
        Registry::new(&fams).unwrap()
    }

    const TOY_CONFIG: &str = r#"{"architectures":["ToyForCausalLM"],"model_type":"toy","toy_layers":4,"toy_kv_heads":2,"toy_head_dim":64,
        "quantization_config":{"quant_method":"exl3","bits":4.0}}"#;

    #[test]
    fn a_family_registered_only_in_a_test_is_dispatched_probed_and_built() {
        let dir = dir_with("toy", TOY_CONFIG);
        let reg = with_toy();
        // the real families are still there, and the toy is found by architecture and by model_type
        assert_eq!(
            reg.families().iter().map(|f| f.id()).collect::<Vec<_>>(),
            vec![
                "qwen3_5",
                "glm5_next",
                "qwen_flash_next",
                "mimo_v2",
                "nemotron_h_omni",
                "minicpmv4_7",
                "deepseek_v41",
                "bailing_moe_v3",
                "toy"
            ]
        );
        let (family, plan) = reg.probe(&dir).unwrap();
        assert_eq!(family.id(), "toy");
        assert_eq!((plan.family, plan.chat_format, plan.layers.len()), ("toy", "toy", 4));
        assert!(plan.missing_components().is_empty());
        // its memory model went through the same validation: 4 layers x 2 x 64 x 2 (K, V) x 2 B = 2 KiB per token
        let profile = plan.memory_profile(&MemoryOptions::serving_defaults()).unwrap();
        assert_eq!(profile.kv_bytes_per_token(), 4 * 2 * 64 * 2 * 2);
        // and its engine starts behind the erased interface and serves a request
        let o = ServeOptions::new(dir.clone());
        let mut engine = family.build(plan, &BuildCtx { options: &o }).unwrap();
        let id = engine
            .submit_deferred(gb_engine::Request {
                prompt: vec![1, 2, 3],
                sampling: gb_sampling::Sampling { seed: 7, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 },
                max_new: 5,
                stop: vec![],
                draft: false,
                images: vec![],
                cache: true,
            })
            .unwrap();
        let mut tokens = Vec::new();
        while engine.busy() {
            engine.step().unwrap();
            for ev in engine.take_events().into_iter().filter(|e| e.id == id) {
                tokens.extend(ev.tokens);
            }
        }
        assert_eq!(tokens.len(), 5);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_family_that_builds_while_components_are_missing_is_refused_by_preflight_and_start() {
        let both: Vec<&'static dyn ModelFamily> = vec![&LEAKY, &TOY];
        let reg = Registry::new(&both).unwrap();
        let leaky_dir = dir_with(
            "leaky",
            r#"{"architectures":["LeakyForCausalLM"],"toy_layers":2,"toy_kv_heads":1,"toy_head_dim":8,"quantization_config":{"quant_method":"exl3"}}"#,
        );
        let (_, plan) = reg.probe(&leaky_dir).unwrap();
        assert_eq!(plan.missing_components(), vec!["nonexistent_kernel"]);
        // the family's `build` succeeds and hands back an engine; the registry-level paths refuse to use it
        // a toy pack is not on the validated list (`serve::check_pack_dirs` runs before the build): served as unvalidated
        let mut o = ServeOptions::new(leaky_dir.clone());
        o.allow_unvalidated_pack = true;
        let preflight = crate::serve::preflight_in(&reg, &o).unwrap_err().to_string();
        assert!(preflight.contains("leaky reported missing components yet built an engine"), "{preflight}");
        let start = crate::serve::start_in(&reg, &o).err().expect("must refuse").to_string();
        assert!(start.contains("leaky reported missing components yet built an engine"), "{start}");
        // control: a family with every component built passes both, and start hands back its engine
        let toy_dir = dir_with("toy-control", TOY_CONFIG);
        let mut o = ServeOptions::new(toy_dir.clone());
        o.allow_unvalidated_pack = true;
        assert!(crate::serve::preflight_in(&reg, &o).is_ok());
        assert!(crate::serve::start_in(&reg, &o).is_ok());
        std::fs::remove_dir_all(leaky_dir).unwrap();
        std::fs::remove_dir_all(toy_dir).unwrap();
    }

    #[test]
    fn a_family_that_does_not_size_its_window_refuses_an_explicit_window_rows_by_name() {
        // `--window-rows` is honoured by the Qwen model only: another family must not run with the flag silently ignored
        let both: Vec<&'static dyn ModelFamily> = vec![&TOY];
        let reg = Registry::new(&both).unwrap();
        let dir = dir_with("toy-window", TOY_CONFIG);
        let mut o = ServeOptions::new(dir.clone());
        o.allow_unvalidated_pack = true; // a toy pack is not on the validated list
        assert_eq!(o.window_rows, crate::forward::WINDOW_ROWS);
        assert!(crate::serve::start_in(&reg, &o).is_ok(), "the default window is nobody's business");
        for w in [128usize, 96, 32] {
            o.window_rows = w;
            for e in [
                crate::serve::start_in(&reg, &o).err().expect("refused").to_string(),
                crate::serve::preflight_in(&reg, &o).unwrap_err().to_string(),
            ] {
                assert!(
                    e.contains(&format!("--window-rows {w}")) && e.contains("the toy family has a fixed verify window"),
                    "{e}"
                );
            }
        }
        o.window_rows = crate::forward::WINDOW_ROWS;
        assert!(crate::serve::preflight_in(&reg, &o).is_ok());
        // a family that says it honours the option is let through at any legal width, and the check is the capability's, not a family's name
        let wide: Vec<&'static dyn ModelFamily> = vec![&WIDE_TOY];
        let reg_wide = Registry::new(&wide).unwrap();
        o.window_rows = 128;
        assert!(crate::serve::start_in(&reg_wide, &o).is_ok());
        assert!(crate::serve::preflight_in(&reg_wide, &o).is_ok());
        // the capability is opt-in: false unless a family says so, and the Qwen family does
        assert!(!TOY.window_rows_adjustable());
        assert!(crate::families::qwen::QwenFamily.window_rows_adjustable());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_toy_is_unknown_to_the_shipped_registry() {
        // without the test's registration the same pack is refused: dispatch comes from the registry and nowhere else
        let dir = dir_with("toy-unknown", TOY_CONFIG);
        let e = Registry::global().probe(&dir).err().expect("the shipped registry does not know the toy");
        assert!(matches!(&e, FamilyError::Unsupported(r) if r.architecture.as_deref() == Some("ToyForCausalLM")), "{e}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn two_families_may_not_claim_one_name() {
        let both: Vec<&'static dyn ModelFamily> = vec![&TOY, &CLASH];
        let e = Registry::new(&both).expect_err("must refuse");
        assert!(e.contains("architecture `ToyForCausalLM`") && e.contains("`toy`") && e.contains("`clash`"), "{e}");
        // the same family listed twice collides with itself, on its id first
        let twice: Vec<&'static dyn ModelFamily> = vec![&TOY, &TOY];
        assert!(Registry::new(&twice).unwrap_err().contains("id `toy`"));
        // a shipped registry is built from a list that must be valid: this fails the moment someone adds a clashing line
        assert!(Registry::new(crate::families::FAMILIES).is_ok());
    }

    #[test]
    fn a_registry_built_from_a_clashing_list_refuses_every_pack() {
        let broken = Registry { families: Vec::new(), broken: Some("architecture `X` is claimed by both `a` and `b`".into()) };
        let e = broken.detect(&serde_json::from_str(QWEN_CONFIG).unwrap()).err().unwrap();
        assert!(matches!(&e, FamilyError::Registry(m) if m.contains("claimed by both")), "{e}");
    }

    #[test]
    fn a_family_must_accept_something_and_be_named() {
        struct Mute;
        impl ModelFamily for Mute {
            fn id(&self) -> &'static str {
                "mute"
            }
            fn architectures(&self) -> &'static [&'static str] {
                &[]
            }
            fn probe(&self, _: &Path) -> Result<FamilyPlan, FamilyError> {
                unreachable!()
            }
            fn build(&self, _: FamilyPlan, _: &BuildCtx<'_>) -> Result<Box<dyn ServeEngine>, ModelError> {
                unreachable!()
            }
        }
        static MUTE: Mute = Mute;
        let l: Vec<&'static dyn ModelFamily> = vec![&MUTE];
        assert!(Registry::new(&l).unwrap_err().contains("accepts no architecture"));
    }

    // ---- the prefix cache is opt-in per family

    #[test]
    fn the_families_that_have_proven_a_prefix_cache_say_so() {
        let flags: Vec<(&str, bool)> = crate::families::FAMILIES.iter().map(|f| (f.id(), f.prefix_cache())).collect();
        assert_eq!(
            flags,
            vec![
                ("qwen3_5", true),
                ("glm5_next", true),
                ("qwen_flash_next", true),
                ("mimo_v2", false),
                ("nemotron_h_omni", true),
                ("minicpmv4_7", true),
                ("deepseek_v41", true),
                ("bailing_moe_v3", true)
            ],
            "a family says true only when its snapshot and restore are proven cached == uncached (a new family starts at false; Flash-Next's bitwise GPU gate remains separate)"
        );
        // the trait's default is no
        assert!(!TOY.prefix_cache() && CACHEY.prefix_cache());
    }

    #[test]
    fn a_family_that_does_not_say_so_is_started_with_the_cache_off_whatever_was_asked() {
        use crate::serve::{start_in, PrefixCacheOption, ServeOptions};
        let dir = dir_with("toycache", TOY_CONFIG);
        for asked in [PrefixCacheOption::Auto, PrefixCacheOption::Mib(2048), PrefixCacheOption::Off] {
            let mut o = ServeOptions::new(dir.clone());
            o.allow_unvalidated_pack = true; // a toy pack is not on the validated list
            o.prefix_cache = asked;
            // GLM's way: the default `prefix_cache()` is false
            let mut fams: Vec<&'static dyn ModelFamily> = vec![&NOCACHE];
            toy_forget("nocache");
            let reg = Registry::new(&fams).unwrap();
            assert!(start_in(&reg, &o).is_ok());
            assert_eq!(toy_saw("nocache"), Some(PrefixCacheOption::Off), "asked {asked:?}");
            // a family that says true gets the request as made
            fams = vec![&CACHEY];
            toy_forget("cachey");
            let reg = Registry::new(&fams).unwrap();
            assert!(start_in(&reg, &o).is_ok());
            assert_eq!(toy_saw("cachey"), Some(asked), "asked {asked:?}");
        }
    }

    // ---- detection and the capability report

    #[test]
    fn the_architecture_decides_the_family() {
        let reg = Registry::global();
        let id = |v: Value| reg.detect(&v).map(|f| f.id());
        assert_eq!(id(json!({"architectures": ["Qwen3_5ForConditionalGeneration"]})), Ok("qwen3_5"));
        assert_eq!(id(json!({"architectures": ["Glm5NextForConditionalGeneration"]})), Ok("glm5_next"));
        // the architecture wins over a model_type that points elsewhere
        assert_eq!(id(json!({"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "qwen3_5"})), Ok("glm5_next"));
    }

    #[test]
    fn model_type_is_only_a_fallback_when_there_are_no_architectures() {
        let reg = Registry::global();
        let id = |v: Value| reg.detect(&v).map(|f| f.id());
        assert_eq!(id(json!({"model_type": "glm5_next"})), Ok("glm5_next"));
        assert_eq!(id(json!({"model_type": "qwen3_5", "architectures": []})), Ok("qwen3_5"));
        // an architecture that is present and unknown is never rescued by a known model_type
        let e = reg.detect(&json!({"architectures": ["MysteryForCausalLM"], "model_type": "qwen3_5"})).err().unwrap();
        assert!(e.to_string().contains("`MysteryForCausalLM`"), "{e}");
    }

    #[test]
    fn an_unknown_architecture_gets_a_capability_report() {
        let reg = Registry::global();
        // a Gemma4-like config: sliding-window layers, per-layer embeddings and a vision tower
        let gemma = json!({
            "architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4",
            "text_config": {"layer_types": ["sliding_attention", "sliding_attention", "full_attention"], "hidden_size_per_layer_input": 256},
            "vision_config": {"depth": 27}
        });
        let FamilyError::Unsupported(r) = reg.detect(&gemma).err().unwrap() else { panic!("expected a report") };
        assert_eq!(r.architecture.as_deref(), Some("Gemma4ForConditionalGeneration"));
        assert_eq!(r.closest, None, "a shared suffix is not a resemblance");
        assert_eq!(r.inferred, vec!["sliding_attn", "ple", "vision_tower"]);
        assert_eq!(r.missing, vec!["sliding_attn", "ple", "vision_tower"]);
        let text = r.to_string();
        assert!(text.contains("`Gemma4ForConditionalGeneration`"), "{text}");
        assert!(text.contains("needs sliding_attn, ple, vision_tower: not built"), "{text}");
        assert!(text.contains("Qwen3_5ForConditionalGeneration") && text.contains("Glm5NextForConditionalGeneration"), "{text}");
        assert!(text.contains("--allow-unvalidated-pack does not relax"), "{text}");
    }

    #[test]
    fn the_closest_family_is_named_by_prefix() {
        let reg = Registry::global();
        let near = |a: &str| match reg.detect(&json!({"architectures": [a]})).err().unwrap() {
            FamilyError::Unsupported(r) => r.closest,
            other => panic!("{other}"),
        };
        assert_eq!(near("Qwen3ForCausalLM"), Some("qwen3_5"));
        assert_eq!(near("Glm5ForCausalLM"), Some("glm5_next"));
        assert_eq!(near("LlamaForCausalLM"), None);
        // an MoE + MTP config is told what it needs even when its name is close to a served family
        let FamilyError::Unsupported(r) = reg
            .detect(&json!({"architectures": ["Qwen3NextForCausalLM"], "num_experts": 512, "mtp_num_hidden_layers": 1, "linear_num_key_heads": 16}))
            .err()
            .unwrap()
        else {
            panic!()
        };
        // (the MoE layer is built since GLM's text model runs; an MTP drafter is not)
        assert_eq!((r.closest, r.inferred.clone(), r.missing.clone()), (Some("qwen3_5"), vec!["moe", "mtp", "gdn"], vec!["mtp"]));
    }

    #[test]
    fn a_config_without_a_name_is_a_malformed_config_not_a_guess() {
        let reg = Registry::global();
        assert!(reg.detect(&json!({})).err().unwrap().to_string().contains("neither"));
        assert!(reg.detect(&json!({"model_type": "mamba2"})).err().unwrap().to_string().contains("`mamba2`"));
        assert!(reg.detect(&json!({"architectures": [7]})).is_err());
        assert!(reg.detect(&json!({"architectures": "Glm5NextForConditionalGeneration"})).is_err());
    }

    #[test]
    fn components_are_inferred_from_keys() {
        let glm: Value = serde_json::from_str(GLM_CONFIG).unwrap();
        assert_eq!(infer_components(&glm), vec!["mla", "dsa_indexer", "moe", "mtp", "mhc", "kda", "vision_tower"]);
        let qwen: Value = serde_json::from_str(QWEN_CONFIG).unwrap();
        assert_eq!(infer_components(&qwen), vec!["mtp", "gdn", "vision_tower"]);
        assert!(infer_components(&json!({"hidden_size": 8})).is_empty());
    }

    // ---- the format check every family would otherwise repeat

    fn probe_text(tag: &str, config: &str) -> Result<String, String> {
        let dir = dir_with(tag, config);
        let r = Registry::global().probe(&dir).map(|(f, _)| f.id().to_string()).map_err(|e| e.to_string());
        std::fs::remove_dir_all(dir).unwrap();
        r
    }

    #[test]
    fn only_exl3_packs_are_served() {
        // an NVFP4-style pack of an otherwise served architecture: refused, naming what was found, before any family probes it
        let mut nvfp4: Value = serde_json::from_str(QWEN_CONFIG).unwrap();
        nvfp4["quantization_config"] = json!({"quant_method": "modelopt", "quant_algo": "NVFP4"});
        let e = probe_text("nvfp4", &nvfp4.to_string()).unwrap_err();
        assert_eq!(e, "this engine serves EXL3 packs only; found quant_method=modelopt");
        // no quantization_config at all
        let mut plain: Value = serde_json::from_str(GLM_CONFIG).unwrap();
        plain.as_object_mut().unwrap().remove("quantization_config");
        let e = probe_text("plain", &plain.to_string()).unwrap_err();
        assert!(e.contains("EXL3 packs only") && e.contains("no quantization_config"), "{e}");
        // an `exl3` method in any case, at the top level or in text_config, passes; another method in text_config does not
        let mut shouty: Value = serde_json::from_str(QWEN_CONFIG).unwrap();
        shouty["quantization_config"]["quant_method"] = json!("EXL3");
        assert_eq!(probe_text("shouty", &shouty.to_string()), Ok("qwen3_5".to_string()));
        let mut nested: Value = serde_json::from_str(QWEN_CONFIG).unwrap();
        let block = nested.as_object_mut().unwrap().remove("quantization_config").unwrap();
        nested["text_config"]["quantization_config"] = block;
        assert_eq!(probe_text("nested", &nested.to_string()), Ok("qwen3_5".to_string()));
        nested["text_config"]["quantization_config"]["quant_method"] = json!("awq");
        assert!(probe_text("nested-awq", &nested.to_string()).unwrap_err().contains("quant_method=awq"));
        // the real packs pass
        assert_eq!(probe_text("real-qwen-exl3", QWEN_CONFIG), Ok("qwen3_5".to_string()));
        assert_eq!(probe_text("real-glm-exl3", GLM_CONFIG), Ok("glm5_next".to_string()));
    }

    #[test]
    fn the_format_check_comes_before_the_architecture_check() {
        // an unknown architecture in a non-EXL3 pack is told about the format, which no family could fix
        let e = probe_text(
            "unknown-nvfp4",
            r#"{"architectures":["Gemma4ForConditionalGeneration"],"quantization_config":{"quant_method":"nvfp4"}}"#,
        )
        .unwrap_err();
        assert!(e.contains("quant_method=nvfp4"), "{e}");
    }

    // ---- the shipped families, through the registry

    #[test]
    fn the_shipped_registry_lists_qwen_first_and_glm_second() {
        let ids: Vec<&str> = Registry::global().families().iter().map(|f| f.id()).collect();
        assert_eq!(
            ids,
            vec![
                "qwen3_5",
                "glm5_next",
                "qwen_flash_next",
                "mimo_v2",
                "nemotron_h_omni",
                "minicpmv4_7",
                "deepseek_v41",
                "bailing_moe_v3"
            ]
        );
    }

    #[test]
    fn qwen_and_the_glm_text_model_and_both_towers_are_built() {
        let reg = Registry::global();
        let qdir = dir_with("real-qwen", QWEN_CONFIG);
        let (qf, qplan) = reg.probe(&qdir).unwrap();
        assert_eq!(qf.id(), "qwen3_5");
        assert!(
            qplan.missing_components().is_empty(),
            "every component Qwen requires is built: {:?}",
            qplan.missing_components()
        );
        assert_eq!(qplan.layers.len(), 64);
        let gdir = dir_with("real-glm", GLM_CONFIG);
        let (gf, gplan) = reg.probe(&gdir).unwrap();
        assert_eq!(gf.id(), "glm5_next");
        // the text model's components, the NextN drafter and the tower are all built
        assert!(gplan.missing_components().is_empty(), "{:?}", gplan.missing_components());
        assert_eq!(gplan.unbuilt_vision(), None, "GLM's tower is built");
        assert_eq!(qplan.unbuilt_vision(), None, "Qwen's tower is built");
        assert_eq!(gplan.drafters, vec!["mtp", "copy"], "the pack's NextN layer and the copy drafter are offered");
        assert_eq!((gplan.layers.len(), gplan.mtp_layers.len(), gplan.chat_format), (45, 1, "glm5_next"));
        std::fs::remove_dir_all(qdir).unwrap();
        std::fs::remove_dir_all(gdir).unwrap();
    }

    #[test]
    fn a_glm_pack_that_does_not_map_cannot_build() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let gdir = dir_with("glm-build", GLM_CONFIG);
        let (gf, gplan) = Registry::global().probe(&gdir).unwrap();
        let mut o = ServeOptions::new(gdir.clone());
        o.vision = None; // keep the plan cheap: the answer is the same
        let e = gf.build(gplan, &BuildCtx { options: &o }).err().unwrap();
        // this directory holds a config and no tensors, so the pack does not map onto the family's weights and is refused by name,
        // before a byte is read or a CUDA context exists
        assert!(matches!(e, ModelError::Config(_)), "{e}");
        assert!(e.to_string().contains("does not match the GLM-5.3 weight layout"), "{e}");
        std::fs::remove_dir_all(gdir).unwrap();
    }

    #[test]
    fn a_family_config_is_never_parsed_by_the_wrong_family() {
        // the GLM config has layer types the Qwen parser refuses, and the Qwen config has no MLA: neither may be read as the other
        let glm: Value = serde_json::from_str(GLM_CONFIG).unwrap();
        assert!(crate::config::QwenConfig::from_json(&glm).is_err());
        let qwen: Value = serde_json::from_str(QWEN_CONFIG).unwrap();
        assert!(crate::families::glm::GlmConfig::from_json(&qwen).is_err());
    }
}

/// The registry tests as nemotron/main has them, on the Nemotron 3.5 Super VL family (the Nemotron-only tree retargeted them from Qwen and
/// GLM): `tests` above holds the same rules on the Qwen and GLM configs, as main has them. Both run. Changed from nemotron/main's text: the
/// temporary folders are named apart from `tests`' (`rexl3-family-nemo-*`: both modules run in one process), the three assertions that
/// pinned the Nemotron-only registry now pin the Nemotron entry within the full one (`tests` pins the full lists), and the closest family
/// of a `Qwen3NextForCausalLM` config is `qwen3_5` now that the Qwen family is registered (it was none).
#[cfg(test)]
mod nemotron_tests {
    use super::*;
    use gb_engine::mock::{Fault, MockBackend};
    use gb_engine::{Engine, EngineConfig, NoDrafts};
    use gb_sched::{Admission, CostModel, KvLayerClass, KvLayout, MemoryModel, ModelMemory, StepShape, VisionMemory};
    use serde_json::json;

    const NEMO_CONFIG: &str = include_str!("../../../testdata/nemotron_omni/config.json");
    const NEMO_ARCH: &str = crate::families::nemotron_h_omni::config::ARCH;
    const NEMO_TYPE: &str = crate::families::nemotron_h_omni::config::MODEL_TYPE;

    fn dir_with(tag: &str, config: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rexl3-family-nemo-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), config).unwrap();
        dir
    }

    // ---- a third family, registered only here: the registry dispatches it without any engine-core file knowing it exists

    /// A toy family: `toy_layers` full-attention layers of `toy_kv_heads` x `toy_head_dim`, served by the mock backend. The names it
    /// accepts and the components it requires are fields, so one definition serves the well-behaved and the misbehaving cases.
    struct Toy {
        id: &'static str,
        archs: &'static [&'static str],
        types: &'static [&'static str],
        required: &'static [&'static str],
        /// What `ModelFamily::prefix_cache` answers.
        cache: bool,
        /// answers `window_rows_adjustable`
        wide: bool,
    }

    /// The `prefix_cache` option the last `build` of each toy family (by id) was handed: keyed by id because tests run in parallel and several build a toy.
    static TOY_SAW: std::sync::Mutex<Vec<(&'static str, crate::serve::PrefixCacheOption)>> = std::sync::Mutex::new(Vec::new());
    fn toy_saw(id: &str) -> Option<crate::serve::PrefixCacheOption> {
        TOY_SAW.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(i, _)| *i == id).map(|(_, o)| *o)
    }
    fn toy_forget(id: &str) {
        TOY_SAW.lock().unwrap_or_else(|e| e.into_inner()).retain(|(i, _)| *i != id);
    }

    struct ToyMemory {
        layers: u64,
        kv_heads: u64,
        head_dim: u64,
    }

    impl MemoryModel for ToyMemory {
        fn family(&self) -> &str {
            "toy"
        }
        fn kv_layout(&self) -> KvLayout {
            KvLayout::new().with(KvLayerClass::gqa(self.layers, self.kv_heads, self.head_dim, 2))
        }
        fn recurrent_layers(&self) -> u64 {
            0
        }
        fn state_bytes_per_seq(&self) -> u64 {
            0
        }
        fn window_record_bytes_per_row(&self) -> u64 {
            0
        }
        fn scratch_bytes(&self, rows: u64) -> u64 {
            rows * 1024
        }
        fn activation_reserve(&self) -> u64 {
            1 << 20
        }
        fn drafter_state_bytes_per_slot(&self) -> u64 {
            0
        }
        fn vision(&self) -> Option<VisionMemory> {
            None
        }
    }

    impl MemorySource for ToyMemory {
        fn profile(&self, opts: &MemoryOptions) -> Result<MemoryProfile, MemoryError> {
            let shape =
                StepShape { page_tokens: 256, window_rows: opts.window_rows as u64, scratch_rows: opts.prefill_rows as u64 };
            MemoryProfile::new(self, shape)
        }
    }

    impl ModelFamily for Toy {
        fn id(&self) -> &'static str {
            self.id
        }
        fn architectures(&self) -> &'static [&'static str] {
            self.archs
        }
        fn model_types(&self) -> &'static [&'static str] {
            self.types
        }
        fn prefix_cache(&self) -> bool {
            self.cache
        }
        fn window_rows_adjustable(&self) -> bool {
            self.wide
        }
        fn probe(&self, dir: &Path) -> Result<FamilyPlan, FamilyError> {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
            let get = |k: &str| v[k].as_u64().ok_or_else(|| FamilyError::Config(format!("toy config: missing `{k}`")));
            let (layers, kv_heads, head_dim) = (get("toy_layers")?, get("toy_kv_heads")?, get("toy_head_dim")?);
            Ok(FamilyPlan {
                family: self.id,
                dir: dir.to_path_buf(),
                layers: (0..layers)
                    .map(|_| LayerSpec { mixer: Mixer::Gqa, ffn: Ffn::Dense, residual: Residual::Plain })
                    .collect(),
                mtp_layers: vec![],
                memory: Box::new(ToyMemory { layers, kv_heads, head_dim }),
                vision: None,
                chat_format: "toy",
                drafters: vec!["copy"],
                required_components: self.required.to_vec(),
            })
        }
        fn build(&self, plan: FamilyPlan, ctx: &BuildCtx<'_>) -> Result<Box<dyn ServeEngine>, ModelError> {
            {
                let mut saw = TOY_SAW.lock().unwrap_or_else(|e| e.into_inner());
                saw.retain(|(i, _)| *i != self.id);
                saw.push((self.id, ctx.options.prefix_cache));
            }
            let opts =
                MemoryOptions { window_rows: 64, prefill_rows: 128, max_context: 4096, mtp: false, vision: None, trees: false };
            let profile = plan.memory_profile(&opts).map_err(|e| ModelError::Config(e.to_string()))?;
            let mm = ModelMemory::from_profile(&profile, 0, 0);
            let admission = Admission::new(1 << 30, 0, mm, ctx.options.slots).map_err(|e| ModelError::Config(e.to_string()))?;
            let cost = CostModel::from_table(vec![(1, 1.0), (64, 2.0)]);
            let engine = Engine::new(
                MockBackend::new(512, ctx.options.slots, Fault::None),
                admission,
                EngineConfig { max_rows: 64, max_drafts: 7, cost, drafting: false },
                Box::new(|| Box::new(NoDrafts) as Box<dyn gb_engine::Drafter>),
            );
            Ok(Box::new(engine))
        }
    }

    static TOY: Toy = Toy {
        id: "toy",
        archs: &["ToyForCausalLM"],
        types: &["toy"],
        required: &["gqa_attention", "dense_mlp"],
        cache: false,
        wide: false,
    };
    /// The toy that says it honours `--window-rows`.
    static WIDE_TOY: Toy = Toy {
        id: "wide-toy",
        archs: &["ToyForCausalLM"],
        types: &["toy"],
        required: &["gqa_attention", "dense_mlp"],
        cache: false,
        wide: true,
    };
    /// A second family that claims the toy's architecture name.
    static CLASH: Toy = Toy { id: "clash", archs: &["ToyForCausalLM"], types: &[], required: &[], cache: false, wide: false };
    /// A family that requires a component nobody has built, and whose `build` succeeds anyway: the registry must not trust it.
    static LEAKY: Toy = Toy {
        id: "leaky",
        archs: &["LeakyForCausalLM"],
        types: &[],
        required: &["gqa_attention", "nonexistent_kernel"],
        cache: false,
        wide: false,
    };
    /// A toy that does not say so (the default `prefix_cache()`), with an id of its own: the test of the clamp reads what its `build` was handed, and
    /// tests run in parallel, so no other test may build a family of that id.
    static NOCACHE: Toy = Toy {
        id: "nocache",
        archs: &["ToyForCausalLM"],
        types: &["toy"],
        required: &["gqa_attention", "dense_mlp"],
        cache: false,
        wide: false,
    };
    /// A toy that says it can be cached.
    static CACHEY: Toy = Toy {
        id: "cachey",
        archs: &["ToyForCausalLM"],
        types: &["toy"],
        required: &["gqa_attention", "dense_mlp"],
        cache: true,
        wide: false,
    };

    fn with_toy() -> Registry {
        let mut fams: Vec<&'static dyn ModelFamily> = crate::families::FAMILIES.to_vec();
        fams.push(&TOY);
        Registry::new(&fams).unwrap()
    }

    const TOY_CONFIG: &str = r#"{"architectures":["ToyForCausalLM"],"model_type":"toy","toy_layers":4,"toy_kv_heads":2,"toy_head_dim":64,
        "quantization_config":{"quant_method":"exl3","bits":4.0}}"#;

    #[test]
    fn a_family_registered_only_in_a_test_is_dispatched_probed_and_built() {
        let dir = dir_with("toy", TOY_CONFIG);
        let reg = with_toy();
        // the real families are still there, and the toy is found by architecture and by model_type
        let ids = reg.families().iter().map(|f| f.id()).collect::<Vec<_>>();
        assert!(ids.contains(&"nemotron_h_omni") && ids.last() == Some(&"toy"), "{ids:?}");
        let (family, plan) = reg.probe(&dir).unwrap();
        assert_eq!(family.id(), "toy");
        assert_eq!((plan.family, plan.chat_format, plan.layers.len()), ("toy", "toy", 4));
        assert!(plan.missing_components().is_empty());
        // its memory model went through the same validation: 4 layers x 2 x 64 x 2 (K, V) x 2 B = 2 KiB per token
        let profile = plan.memory_profile(&MemoryOptions::serving_defaults()).unwrap();
        assert_eq!(profile.kv_bytes_per_token(), 4 * 2 * 64 * 2 * 2);
        // and its engine starts behind the erased interface and serves a request
        let o = ServeOptions::new(dir.clone());
        let mut engine = family.build(plan, &BuildCtx { options: &o }).unwrap();
        let id = engine
            .submit_deferred(gb_engine::Request {
                prompt: vec![1, 2, 3],
                sampling: gb_sampling::Sampling { seed: 7, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 },
                max_new: 5,
                stop: vec![],
                draft: false,
                images: vec![],
                cache: true,
            })
            .unwrap();
        let mut tokens = Vec::new();
        while engine.busy() {
            engine.step().unwrap();
            for ev in engine.take_events().into_iter().filter(|e| e.id == id) {
                tokens.extend(ev.tokens);
            }
        }
        assert_eq!(tokens.len(), 5);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_family_that_builds_while_components_are_missing_is_refused_by_preflight_and_start() {
        let both: Vec<&'static dyn ModelFamily> = vec![&LEAKY, &TOY];
        let reg = Registry::new(&both).unwrap();
        let leaky_dir = dir_with(
            "leaky",
            r#"{"architectures":["LeakyForCausalLM"],"toy_layers":2,"toy_kv_heads":1,"toy_head_dim":8,"quantization_config":{"quant_method":"exl3"}}"#,
        );
        let (_, plan) = reg.probe(&leaky_dir).unwrap();
        assert_eq!(plan.missing_components(), vec!["nonexistent_kernel"]);
        // the family's `build` succeeds and hands back an engine; the registry-level paths refuse to use it
        // a toy pack is not on the validated list (`serve::check_pack_dirs` runs before the build): served as unvalidated
        let mut o = ServeOptions::new(leaky_dir.clone());
        o.allow_unvalidated_pack = true;
        let preflight = crate::serve::preflight_in(&reg, &o).unwrap_err().to_string();
        assert!(preflight.contains("leaky reported missing components yet built an engine"), "{preflight}");
        let start = crate::serve::start_in(&reg, &o).err().expect("must refuse").to_string();
        assert!(start.contains("leaky reported missing components yet built an engine"), "{start}");
        // control: a family with every component built passes both, and start hands back its engine
        let toy_dir = dir_with("toy-control", TOY_CONFIG);
        let mut o = ServeOptions::new(toy_dir.clone());
        o.allow_unvalidated_pack = true;
        assert!(crate::serve::preflight_in(&reg, &o).is_ok());
        assert!(crate::serve::start_in(&reg, &o).is_ok());
        std::fs::remove_dir_all(leaky_dir).unwrap();
        std::fs::remove_dir_all(toy_dir).unwrap();
    }

    #[test]
    fn a_family_that_does_not_size_its_window_refuses_an_explicit_window_rows_by_name() {
        // `--window-rows` is honoured only by a family that sizes its window: another family must not run with the flag silently ignored
        let both: Vec<&'static dyn ModelFamily> = vec![&TOY];
        let reg = Registry::new(&both).unwrap();
        let dir = dir_with("toy-window", TOY_CONFIG);
        let mut o = ServeOptions::new(dir.clone());
        o.allow_unvalidated_pack = true; // a toy pack is not on the validated list
        assert_eq!(o.window_rows, crate::forward::WINDOW_ROWS);
        assert!(crate::serve::start_in(&reg, &o).is_ok(), "the default window is nobody's business");
        for w in [128usize, 96, 32] {
            o.window_rows = w;
            for e in [
                crate::serve::start_in(&reg, &o).err().expect("refused").to_string(),
                crate::serve::preflight_in(&reg, &o).unwrap_err().to_string(),
            ] {
                assert!(
                    e.contains(&format!("--window-rows {w}")) && e.contains("the toy family has a fixed verify window"),
                    "{e}"
                );
            }
        }
        o.window_rows = crate::forward::WINDOW_ROWS;
        assert!(crate::serve::preflight_in(&reg, &o).is_ok());
        // a family that says it honours the option is let through at any legal width, and the check is the capability's, not a family's name
        let wide: Vec<&'static dyn ModelFamily> = vec![&WIDE_TOY];
        let reg_wide = Registry::new(&wide).unwrap();
        o.window_rows = 128;
        assert!(crate::serve::start_in(&reg_wide, &o).is_ok());
        assert!(crate::serve::preflight_in(&reg_wide, &o).is_ok());
        // the capability is opt-in: false unless a family says so, and the Nemotron family does
        assert!(!TOY.window_rows_adjustable());
        assert!(crate::families::nemotron_h_omni::NemotronOmniFamily.window_rows_adjustable());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_toy_is_unknown_to_the_shipped_registry() {
        // without the test's registration the same pack is refused: dispatch comes from the registry and nowhere else
        let dir = dir_with("toy-unknown", TOY_CONFIG);
        let e = Registry::global().probe(&dir).err().expect("the shipped registry does not know the toy");
        assert!(matches!(&e, FamilyError::Unsupported(r) if r.architecture.as_deref() == Some("ToyForCausalLM")), "{e}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn two_families_may_not_claim_one_name() {
        let both: Vec<&'static dyn ModelFamily> = vec![&TOY, &CLASH];
        let e = Registry::new(&both).expect_err("must refuse");
        assert!(e.contains("architecture `ToyForCausalLM`") && e.contains("`toy`") && e.contains("`clash`"), "{e}");
        // the same family listed twice collides with itself, on its id first
        let twice: Vec<&'static dyn ModelFamily> = vec![&TOY, &TOY];
        assert!(Registry::new(&twice).unwrap_err().contains("id `toy`"));
        // a shipped registry is built from a list that must be valid: this fails the moment someone adds a clashing line
        assert!(Registry::new(crate::families::FAMILIES).is_ok());
    }

    #[test]
    fn a_registry_built_from_a_clashing_list_refuses_every_pack() {
        let broken = Registry { families: Vec::new(), broken: Some("architecture `X` is claimed by both `a` and `b`".into()) };
        let e = broken.detect(&serde_json::from_str(NEMO_CONFIG).unwrap()).err().unwrap();
        assert!(matches!(&e, FamilyError::Registry(m) if m.contains("claimed by both")), "{e}");
    }

    #[test]
    fn a_family_must_accept_something_and_be_named() {
        struct Mute;
        impl ModelFamily for Mute {
            fn id(&self) -> &'static str {
                "mute"
            }
            fn architectures(&self) -> &'static [&'static str] {
                &[]
            }
            fn probe(&self, _: &Path) -> Result<FamilyPlan, FamilyError> {
                unreachable!()
            }
            fn build(&self, _: FamilyPlan, _: &BuildCtx<'_>) -> Result<Box<dyn ServeEngine>, ModelError> {
                unreachable!()
            }
        }
        static MUTE: Mute = Mute;
        let l: Vec<&'static dyn ModelFamily> = vec![&MUTE];
        assert!(Registry::new(&l).unwrap_err().contains("accepts no architecture"));
    }

    // ---- the prefix cache is opt-in per family

    #[test]
    fn the_families_that_have_proven_a_prefix_cache_say_so() {
        let flags: Vec<(&str, bool)> = crate::families::FAMILIES.iter().map(|f| (f.id(), f.prefix_cache())).collect();
        assert!(
            flags.contains(&("nemotron_h_omni", true)),
            "a family says true only when its snapshot and restore are proven cached == uncached (a new family starts at false): {flags:?}"
        );
        // the trait's default is no
        assert!(!TOY.prefix_cache() && CACHEY.prefix_cache());
    }

    #[test]
    fn a_family_that_does_not_say_so_is_started_with_the_cache_off_whatever_was_asked() {
        use crate::serve::{start_in, PrefixCacheOption, ServeOptions};
        let dir = dir_with("toycache", TOY_CONFIG);
        for asked in [PrefixCacheOption::Auto, PrefixCacheOption::Mib(2048), PrefixCacheOption::Off] {
            let mut o = ServeOptions::new(dir.clone());
            o.allow_unvalidated_pack = true; // a toy pack is not on the validated list
            o.prefix_cache = asked;
            // the trait's default: `prefix_cache()` is false
            let mut fams: Vec<&'static dyn ModelFamily> = vec![&NOCACHE];
            toy_forget("nocache");
            let reg = Registry::new(&fams).unwrap();
            assert!(start_in(&reg, &o).is_ok());
            assert_eq!(toy_saw("nocache"), Some(PrefixCacheOption::Off), "asked {asked:?}");
            // a family that says true gets the request as made
            fams = vec![&CACHEY];
            toy_forget("cachey");
            let reg = Registry::new(&fams).unwrap();
            assert!(start_in(&reg, &o).is_ok());
            assert_eq!(toy_saw("cachey"), Some(asked), "asked {asked:?}");
        }
    }

    // ---- detection and the capability report

    #[test]
    fn the_architecture_decides_the_family() {
        let reg = Registry::global();
        let id = |v: Value| reg.detect(&v).map(|f| f.id());
        assert_eq!(id(json!({"architectures": [NEMO_ARCH]})), Ok("nemotron_h_omni"));
        // the architecture wins over a model_type that points elsewhere
        let reg = with_toy();
        let id = |v: Value| reg.detect(&v).map(|f| f.id());
        assert_eq!(id(json!({"architectures": [NEMO_ARCH], "model_type": "toy"})), Ok("nemotron_h_omni"));
        assert_eq!(id(json!({"architectures": ["ToyForCausalLM"], "model_type": NEMO_TYPE})), Ok("toy"));
    }

    #[test]
    fn model_type_is_only_a_fallback_when_there_are_no_architectures() {
        let reg = Registry::global();
        let id = |v: Value| reg.detect(&v).map(|f| f.id());
        assert_eq!(id(json!({"model_type": NEMO_TYPE})), Ok("nemotron_h_omni"));
        assert_eq!(id(json!({"model_type": NEMO_TYPE, "architectures": []})), Ok("nemotron_h_omni"));
        // an architecture that is present and unknown is never rescued by a known model_type
        let e = reg.detect(&json!({"architectures": ["MysteryForCausalLM"], "model_type": NEMO_TYPE})).err().unwrap();
        assert!(e.to_string().contains("`MysteryForCausalLM`"), "{e}");
    }

    #[test]
    fn an_unknown_architecture_gets_a_capability_report() {
        let reg = Registry::global();
        // a Gemma4-like config: sliding-window layers, per-layer embeddings and a vision tower
        let gemma = json!({
            "architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4",
            "text_config": {"layer_types": ["sliding_attention", "sliding_attention", "full_attention"], "hidden_size_per_layer_input": 256},
            "vision_config": {"depth": 27}
        });
        let FamilyError::Unsupported(r) = reg.detect(&gemma).err().unwrap() else { panic!("expected a report") };
        assert_eq!(r.architecture.as_deref(), Some("Gemma4ForConditionalGeneration"));
        assert_eq!(r.closest, None, "a shared suffix is not a resemblance");
        assert_eq!(r.inferred, vec!["sliding_attn", "ple", "vision_tower"]);
        assert_eq!(r.missing, vec!["sliding_attn", "ple", "vision_tower"]);
        let text = r.to_string();
        assert!(text.contains("`Gemma4ForConditionalGeneration`"), "{text}");
        assert!(text.contains("needs sliding_attn, ple, vision_tower: not built"), "{text}");
        assert!(text.contains(NEMO_ARCH), "{text}");
        assert!(text.contains("--allow-unvalidated-pack does not relax"), "{text}");
    }

    #[test]
    fn the_closest_family_is_named_by_prefix() {
        let reg = Registry::global();
        let near = |a: &str| match reg.detect(&json!({"architectures": [a]})).err().unwrap() {
            FamilyError::Unsupported(r) => r.closest,
            other => panic!("{other}"),
        };
        assert_eq!(near("NemotronHForCausalLM"), Some("nemotron_h_omni"));
        assert_eq!(near("LlamaForCausalLM"), None);
        // an MoE + MTP config is told what it needs even when its name is close to a served family
        let FamilyError::Unsupported(r) = reg
            .detect(&json!({"architectures": ["Qwen3NextForCausalLM"], "num_experts": 512, "mtp_num_hidden_layers": 1, "linear_num_key_heads": 16}))
            .err()
            .unwrap()
        else {
            panic!()
        };
        // (the MoE and GDN kernels are built; an MTP drafter for this config is not; with the Qwen family registered its name is the
        // closest, where the Nemotron-only registry had none)
        assert_eq!((r.closest, r.inferred.clone(), r.missing.clone()), (Some("qwen3_5"), vec!["moe", "mtp", "gdn"], vec!["mtp"]));
    }

    #[test]
    fn a_config_without_a_name_is_a_malformed_config_not_a_guess() {
        let reg = Registry::global();
        assert!(reg.detect(&json!({})).err().unwrap().to_string().contains("neither"));
        assert!(reg.detect(&json!({"model_type": "mamba2"})).err().unwrap().to_string().contains("`mamba2`"));
        assert!(reg.detect(&json!({"architectures": [7]})).is_err());
        assert!(reg.detect(&json!({"architectures": NEMO_ARCH})).is_err());
    }

    #[test]
    fn components_are_inferred_from_keys() {
        let mla_config = json!({"text_config": {"kv_lora_rank": 512, "index_topk": 2048, "n_routed_experts": 64, "num_nextn_predict_layers": 1,
            "hc_mult": 4, "linear_attn_config": {"kda_layers": [1]}}, "vision_config": {}});
        assert_eq!(infer_components(&mla_config), vec!["mla", "dsa_indexer", "moe", "mtp", "mhc", "kda", "vision_tower"]);
        let gdn_config = json!({"text_config": {"mtp_num_hidden_layers": 1, "linear_num_key_heads": 16}, "vision_config": {}});
        assert_eq!(infer_components(&gdn_config), vec!["mtp", "gdn", "vision_tower"]);
        assert!(infer_components(&json!({"hidden_size": 8})).is_empty());
    }

    // ---- the format check every family would otherwise repeat

    fn probe_text(tag: &str, config: &str) -> Result<String, String> {
        let dir = dir_with(tag, config);
        // the toy family reads its quantization from the config, as every family that does not prove it from the tensors does
        let r = with_toy().probe(&dir).map(|(f, _)| f.id().to_string()).map_err(|e| e.to_string());
        std::fs::remove_dir_all(dir).unwrap();
        r
    }

    #[test]
    fn only_exl3_packs_are_served() {
        // an NVFP4-style pack of an otherwise served architecture: refused, naming what was found, before any family probes it
        let mut nvfp4: Value = serde_json::from_str(TOY_CONFIG).unwrap();
        nvfp4["quantization_config"] = json!({"quant_method": "modelopt", "quant_algo": "NVFP4"});
        let e = probe_text("nvfp4", &nvfp4.to_string()).unwrap_err();
        assert_eq!(e, "this engine serves EXL3 packs only; found quant_method=modelopt");
        // no quantization_config at all
        let mut plain: Value = serde_json::from_str(TOY_CONFIG).unwrap();
        plain.as_object_mut().unwrap().remove("quantization_config");
        let e = probe_text("plain", &plain.to_string()).unwrap_err();
        assert!(e.contains("EXL3 packs only") && e.contains("no quantization_config"), "{e}");
        // an `exl3` method in any case, at the top level or in text_config, passes; another method in text_config does not
        let mut shouty: Value = serde_json::from_str(TOY_CONFIG).unwrap();
        shouty["quantization_config"]["quant_method"] = json!("EXL3");
        assert_eq!(probe_text("shouty", &shouty.to_string()), Ok("toy".to_string()));
        let mut nested: Value = serde_json::from_str(TOY_CONFIG).unwrap();
        let block = nested.as_object_mut().unwrap().remove("quantization_config").unwrap();
        nested["text_config"] = json!({});
        nested["text_config"]["quantization_config"] = block;
        assert_eq!(probe_text("nested", &nested.to_string()), Ok("toy".to_string()));
        nested["text_config"]["quantization_config"]["quant_method"] = json!("awq");
        assert!(probe_text("nested-awq", &nested.to_string()).unwrap_err().contains("quant_method=awq"));
        // the toy pack as committed passes
        assert_eq!(probe_text("real-toy-exl3", TOY_CONFIG), Ok("toy".to_string()));
    }

    #[test]
    fn the_format_check_comes_before_the_architecture_check() {
        // an unknown architecture in a non-EXL3 pack is told about the format, which no family could fix
        let e = probe_text(
            "unknown-nvfp4",
            r#"{"architectures":["Gemma4ForConditionalGeneration"],"quantization_config":{"quant_method":"nvfp4"}}"#,
        )
        .unwrap_err();
        assert!(e.contains("quant_method=nvfp4"), "{e}");
    }

    // ---- the shipped families, through the registry

    #[test]
    fn the_shipped_registry_lists_nemotron() {
        let ids: Vec<&str> = Registry::global().families().iter().map(|f| f.id()).collect();
        assert!(ids.contains(&"nemotron_h_omni"), "{ids:?}");
    }
}
