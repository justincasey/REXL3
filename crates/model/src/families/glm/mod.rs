//! The GLM-5.3-Flash family (`Glm5NextForConditionalGeneration`): probe (configuration, composition, memory model, vision component) and
//! `build`, which serves the text model: the pack is mapped and loaded, the state and scratch are allocated, and a `GlmBackend` runs the
//! serial forward and, with `--spec mtp|auto`, the pack's NextN layer as a drafter (the vision tower is not built).
//!
//! The files, in the order a request meets them. Pack plumbing (slice 3): `inventory` (what a pack holds, as plain data), `weights` (the
//! typed description), `bind` (the fail-closed mapper, a pure function from inventory and config to the description) and `plan` (the load
//! plan: items with their conversions, regions). The GPU: `load` (the plan executed through the pipelined loader), `device` (the typed
//! device weights), `alloc` and `memory` (what the backend allocates, as arithmetic, and the memory model built on it), `ops` (the forward
//! as data), `model` (the state, the scratch and the executor), `backend` (`gb_engine::Backend`) and `engine` (building the engine).
//! [`GlmFamily::map_pack`] runs the plumbing on a pack directory.

pub mod alloc;
pub mod backend;
pub mod bind;
pub mod config;
pub mod contract;
pub mod device;
pub mod dflash;
pub mod digest_log;
pub mod dump;
pub mod engine;
pub mod inventory;
pub mod load;
pub mod memory;
pub mod model;
pub mod npy;
pub mod ops;
pub mod plan;
pub mod prefix;
pub mod profile_slices;
pub mod spec;
pub mod tp;
pub mod tp_backend;
pub mod tp_boot;
pub mod tp_mtp;
pub mod tp_vision;
pub mod vision;
pub mod vision_serve;
pub mod weights;

#[cfg(test)]
mod layout_tests;
#[cfg(test)]
mod vision_layout_tests;

pub use bind::{LayoutError, Problem};
pub use config::{GlmConfig, GlmProcessorConfig, GlmVisionConfig, VisionTowerFormat};
pub use inventory::{GroupInfo, PackInventory, TensorInfo};
pub use plan::{ItemPart, LoadItem, LoadPlan, PlanTotals, ReadExtent, Region, RegionKind};
pub use weights::{GlmWeights, Xform};

use crate::error::ModelError;
use crate::family::{BuildCtx, FamilyError, FamilyPlan, ModelFamily, VisionSpec};
use crate::memory::ENCODERS_IN_FLIGHT;
use crate::pack::PackError;
use crate::plan::{plan_memory, MemoryPlan, PlanError};
use crate::serve::{gib, memory_options, ServeOptions};
use gb_engine::{boot, ServeEngine};
use gb_exl3::checkpoint::Checkpoint;
use gb_sched::MemoryProfile;
use memory::GlmMemorySource;
use std::path::Path;

/// The components the text model needs (all in `BUILT_COMPONENTS`). The vision tower is not one of them: it is declared through the plan's
/// `VisionSpec` (`vision_glm5`, built: `vision::GlmVisionTower`) and loaded when the server asks for images; the MTP drafter is an option
/// the pack offers once [`MTP_BUILT`].
const REQUIRED: &[&str] = &["exl3_linear", "kda", "mla_dsa", "mhc", "moe"];

/// The pack's NextN layer is a drafter (`--spec mtp|auto`): the probe offers it, the plan charges its cache, ring, tap store and scratch when the
/// mode uses it, and the loader brings its 3.5 GiB in.
pub const MTP_BUILT: bool = true;

pub struct GlmFamily;

impl ModelFamily for GlmFamily {
    fn id(&self) -> &'static str {
        "glm5_next"
    }
    fn architectures(&self) -> &'static [&'static str] {
        &["Glm5NextForConditionalGeneration"]
    }
    fn model_types(&self) -> &'static [&'static str] {
        &["glm5_next", "glm5_next_text"]
    }

    fn probe(&self, dir: &Path) -> Result<FamilyPlan, FamilyError> {
        let cfg = GlmConfig::from_dir(dir).map_err(FamilyError::Config)?;
        let vision = cfg.vision.as_ref().map(|v| VisionSpec {
            tower: "vision_glm5",
            patch_size: v.patch_size,
            merge_size: v.spatial_merge_size,
            temporal_patch_size: v.temporal_patch_size,
            out_hidden: v.out_hidden,
            image_token_id: v.image_token_id,
            max_image_tokens: cfg.processor.as_ref().map_or(8000, |p| p.image.max_image_tokens),
        });
        let required: Vec<&'static str> = REQUIRED.to_vec();
        // the pack's NextN layer, an external DFlash2 draft (`--draft DIR`, in the NextN layer's place) and the copy drafter
        let drafters = if MTP_BUILT && cfg.has_mtp() { vec!["mtp", "dflash2", "copy"] } else { vec!["dflash2", "copy"] };
        Ok(FamilyPlan {
            family: self.id(),
            dir: dir.to_path_buf(),
            layers: cfg.layers.clone(),
            mtp_layers: cfg.mtp.clone(),
            memory: Box::new(GlmMemorySource(cfg)),
            vision,
            chat_format: "glm5_next",
            drafters,
            required_components: required,
        })
    }

    /// The engine: the pack is mapped (a pack whose tensors do not match the family's layout is refused by name, before a byte is read),
    /// its weights are loaded through the pipelined loader (the NextN layer when `--spec` drafts with it, the vision tower's when the server serves
    /// images), the state and scratch are allocated and the backend is wired into the engine.
    ///
    /// Images are served when `ctx.options.vision` is set (the server's `--no-vision` leaves it unset, and image parts are refused with a 400).
    fn build(&self, plan: FamilyPlan, ctx: &BuildCtx<'_>) -> Result<Box<dyn ServeEngine>, ModelError> {
        let o = ctx.options;
        eprintln!(
            "rexl3 event=family family={} layers={} mla_dsa={} kda={} dense={} moe={} mtp_layers={} vision={}",
            plan.family,
            plan.layers.len(),
            plan.layers.iter().filter(|l| matches!(l.mixer, crate::family::Mixer::MlaDsa { .. })).count(),
            plan.layers.iter().filter(|l| l.mixer == crate::family::Mixer::Kda).count(),
            plan.layers.iter().filter(|l| l.ffn == crate::family::Ffn::Dense).count(),
            plan.layers.iter().filter(|l| l.ffn == crate::family::Ffn::Moe).count(),
            plan.mtp_layers.len(),
            plan.vision.is_some()
        );
        boot::fact("family", plan.family);
        // a tensor-parallel group (`tp_boot`, `docs/GLM_TP.md`): what it cannot forward yet is refused by name, then rank 0's rendezvous
        // socket is bound before anything is read, so rank 1 can connect while this rank plans and loads. Without a group nothing changes.
        if let Some(tp) = o.tp.as_ref().filter(|t| t.is_group()) {
            tp_boot::check_tp_options(o)?;
            let listener = crate::lockstep::listen(&tp.peer)?;
            return tp_boot::build_leader(o, &listener, tp);
        }
        Ok(Box::new(engine::build_engine(o)?))
    }

    /// Two ranks, each with the whole pack and half of the routed experts (`tp.rs`; what the CPU tests, the one-GPU loopback and the
    /// two-node scripts of `docs/GLM_TP.md` cover). Not a measured speedup.
    fn max_tp_ranks(&self) -> usize {
        tp::MAX_TP_RANKS
    }

    fn run_follower(&self, _plan: FamilyPlan, ctx: &BuildCtx<'_>) -> Result<(), ModelError> {
        tp_boot::follow(ctx)
    }

    fn image_preprocessor(
        &self,
        dir: &Path,
        o: &ServeOptions,
    ) -> Result<Option<std::sync::Arc<dyn gb_vision::preprocessor::ImagePreprocessor>>, ModelError> {
        image_preprocessor(dir, o)
    }

    /// GLM snapshots the KDA state and conv history, the indexer's ring tail, the NextN drafter's carry and the KV pages of a prompt (`engine::build_engine`,
    /// `prefix.rs`; proven cached == uncached by `crates/model/tests/glm_prefix.rs` and `glm_prefix_engine.rs`, evidence in `docs/GLM53.md` "Prefix cache").
    fn prefix_cache(&self) -> bool {
        true
    }
}

/// GLM-5.3-Flash's image preprocessing for a server limited to `o.max_image_tokens` tokens an image: the pack's `processor_config.json` read by
/// `GlmPreprocessing::from_processor_config` in the `Fork` profile (the oracle the tower is validated against: alpha composited on white, PIL bicubic,
/// numpy normalisation, fp16), with the processor's token budget lowered to the server's limit (the processor shrinks an image to fit a budget, as HF's
/// `max_image_tokens` keyword does, so an image over the limit is scaled, not refused). `None` for a pack with no vision tower.
pub fn image_preprocessor(
    dir: &Path,
    o: &ServeOptions,
) -> Result<Option<std::sync::Arc<dyn gb_vision::preprocessor::ImagePreprocessor>>, ModelError> {
    let cfg = GlmConfig::from_dir(dir).map_err(ModelError::Config)?;
    if cfg.vision.is_none() {
        return Ok(None);
    }
    let path = dir.join("processor_config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| ModelError::Config(format!("{}: {e}", path.display())))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| ModelError::Config(format!("{}: {e}", path.display())))?;
    let pp = gb_vision::glm::GlmPreprocessing::from_processor_config(&json, gb_vision::glm::Profile::Fork)
        .map_err(|e| ModelError::Config(e.to_string()))?;
    let tokens = vision::image_token_cap(&cfg, o.max_image_tokens);
    let pp = pp.with_max_image_tokens(tokens).map_err(|e| ModelError::Config(e.to_string()))?;
    Ok(Some(std::sync::Arc::new(pp)))
}

// ------------------------------------------------------------------------------------------------------- the mapped pack

/// Why a pack directory could not be mapped onto the family's weights.
#[derive(Debug)]
pub enum GlmPackError {
    /// `config.json` or `processor_config.json` did not parse or validate.
    Config(String),
    /// The pack could not be scanned (unreadable shard, refused EXL3 group).
    Pack(PackError),
    /// The scan's metadata could not be turned into an inventory.
    Inventory(String),
    /// The pack's tensors do not match the layout the config implies: the problems are named.
    Layout(LayoutError),
    /// The mapper built a plan that fails its own consistency check: a bug, never a property of the pack.
    Plan(String),
}

impl std::fmt::Display for GlmPackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GlmPackError::Config(m) => write!(f, "{m}"),
            GlmPackError::Pack(e) => write!(f, "{e}"),
            GlmPackError::Inventory(m) => write!(f, "pack inventory: {m}"),
            GlmPackError::Layout(e) => write!(f, "{e}"),
            GlmPackError::Plan(m) => write!(f, "internal error: the load plan is inconsistent: {m}"),
        }
    }
}

impl std::error::Error for GlmPackError {}

impl From<GlmPackError> for ModelError {
    fn from(e: GlmPackError) -> Self {
        match e {
            GlmPackError::Pack(p) => ModelError::Pack(p),
            other => ModelError::Config(other.to_string()),
        }
    }
}

/// A GLM pack mapped, from its headers alone (no weight is read, no CUDA context exists): the configuration, the typed weights and
/// the plan that loads them. This is what the engine's later slices consume: the kernels take their tensors from `weights`, the
/// load pipeline executes `plan` (or `plan.select(..)` of it).
#[derive(Clone, Debug)]
pub struct GlmPack {
    pub config: GlmConfig,
    pub weights: GlmWeights,
    pub plan: LoadPlan,
}

impl GlmPack {
    /// Map an inventory (see [`PackInventory`]) with `config`.
    pub fn from_inventory(inv: &PackInventory, config: GlmConfig) -> Result<Self, GlmPackError> {
        let own = gb_cuda::moe::ExpertOwnership::all(config.ffn.routed_experts);
        Self::from_inventory_owned(inv, config, &own)
    }

    /// [`GlmPack::from_inventory`] for a rank holding only `own`'s routed experts (`GlmWeights::map_owned`): its plan reads and places
    /// only those.
    pub fn from_inventory_owned(
        inv: &PackInventory,
        config: GlmConfig,
        own: &gb_cuda::moe::ExpertOwnership,
    ) -> Result<Self, GlmPackError> {
        Self::from_inventory_with(inv, config, own, true)
    }

    /// [`GlmPack::from_inventory_owned`]; `vision` false (`--no-vision`, a follower) claims and skips the tower (`GlmWeights::map_with`): the
    /// plan has no `Vision` region and the text model maps whatever the tower's format.
    pub fn from_inventory_with(
        inv: &PackInventory,
        config: GlmConfig,
        own: &gb_cuda::moe::ExpertOwnership,
        vision: bool,
    ) -> Result<Self, GlmPackError> {
        let weights = GlmWeights::map_with(inv, &config, own, vision).map_err(GlmPackError::Layout)?;
        let plan = weights.load_plan().map_err(GlmPackError::Plan)?;
        Ok(Self { config, weights, plan })
    }

    /// Map a scanned pack.
    pub fn from_checkpoint(ck: &Checkpoint, config: GlmConfig) -> Result<Self, GlmPackError> {
        Self::from_checkpoint_with(ck, config, true)
    }

    /// [`GlmPack::from_checkpoint`] with the tower bound only when `vision` (see [`GlmPack::from_inventory_with`]).
    pub fn from_checkpoint_with(ck: &Checkpoint, config: GlmConfig, vision: bool) -> Result<Self, GlmPackError> {
        let inv = PackInventory::from_checkpoint(ck).map_err(GlmPackError::Inventory)?;
        let own = gb_cuda::moe::ExpertOwnership::all(config.ffn.routed_experts);
        Self::from_inventory_with(&inv, config, &own, vision)
    }

    /// The widths of the routed experts by layer count, as `bits:layers` (`2:30,3:12,4:1`), for logs.
    pub fn expert_bits_summary(&self) -> String {
        let mut by_bits: std::collections::BTreeMap<u8, usize> = Default::default();
        for b in self.weights.banks() {
            *by_bits.entry(b.bits_gate_up.twice()).or_default() += 1;
        }
        by_bits.iter().map(|(t, n)| format!("{}:{n}", *t as f64 / 2.0)).collect::<Vec<_>>().join(",")
    }
}

impl GlmFamily {
    /// Scan the pack in `dir` (headers only) and map it: the configuration, the typed weights and the load plan. Refuses a pack
    /// whose tensors do not match the layout its config implies, with every problem named.
    ///
    /// The 37,420 codebook markers are not read here (one tiny read each costs about 10 s on a cold page cache): the plan lists them
    /// (`LoadPlan::markers`) and whoever executes the plan checks them inside the long reads it makes anyway, so a pack is never
    /// decoded with the wrong codebook, and no boot step does random I/O.
    pub fn map_pack(dir: &Path) -> Result<GlmPack, GlmPackError> {
        let config = GlmConfig::from_dir(dir).map_err(GlmPackError::Config)?;
        let ck = gb_exl3::checkpoint::scan(dir, false).map_err(|e| GlmPackError::Pack(PackError::Io(e.to_string())))?;
        GlmPack::from_checkpoint(&ck, config)
    }

    /// [`GlmFamily::map_pack`] for a rank holding only `own`'s routed experts: boot it with `load::boot_mapped`.
    pub fn map_pack_owned(dir: &Path, own: &gb_cuda::moe::ExpertOwnership) -> Result<GlmPack, GlmPackError> {
        let config = GlmConfig::from_dir(dir).map_err(GlmPackError::Config)?;
        let ck = gb_exl3::checkpoint::scan(dir, false).map_err(|e| GlmPackError::Pack(PackError::Io(e.to_string())))?;
        let inv = PackInventory::from_checkpoint(&ck).map_err(GlmPackError::Inventory)?;
        GlmPack::from_inventory_owned(&inv, config, own)
    }
}

// ----------------------------------------------------------------------------------------------------------- the plan report

/// Bytes the pack's tensors occupy, from the safetensors headers: every EXL3 group and every plain tensor. The vision tower's
/// tensors (`model.visual.*`: plain bf16 in the bf16 tower; EXL3 groups with their biases, fp16 and bf16 tensors in the EXL3 tower) are
/// left out when the tower will not be loaded. With the tower this is an upper bound for the EXL3 tower: the fused bf16 qkv it does not
/// load is still counted.
pub fn pack_weight_bytes(ck: &Checkpoint, with_vision: bool) -> u64 {
    let keep = |name: &str| with_vision || !name.starts_with(bind::VISION_PREFIX);
    let groups: u64 = ck.groups.iter().filter(|(prefix, _)| keep(prefix)).map(|(_, g)| g.bytes).sum();
    let plain: u64 = ck.plain.iter().filter(|(name, _)| keep(name)).map(|(_, e)| e.nbytes()).sum();
    groups + plain
}

/// What the memory model says about a GLM pack, before anything is loaded.
pub struct GlmPlanReport {
    pub profile: MemoryProfile,
    /// Whether the pack's own MTP head is the drafter (its cache is in the per-token cost).
    pub mtp: bool,
    pub weights_bytes: u64,
    /// `MemAvailable` less the weights: the free memory once the pack is resident (assumes the loader keeps the pack's bytes:
    /// EXL3 groups and bf16 tensors are uploaded as stored).
    pub free_after_weights: u64,
    pub vision_reserve_bytes: u64,
    /// The plan, or why there is none (no room is an outcome, not a failure of this report).
    pub plan: Result<MemoryPlan, PlanError>,
}

/// The prospective memory plan of a GLM pack, without loading weights and without creating a CUDA context: free memory is
/// `mem_available - weights_bytes`, and every holdback comes out in the planner exactly as for a loaded engine.
pub fn glm_memory_plan(
    o: &ServeOptions,
    g: &GlmConfig,
    weights_bytes: u64,
    mem_available: u64,
) -> Result<GlmPlanReport, ModelError> {
    // the pack's own NextN layer is GLM's drafter when the mode uses it; `--no-drafts` / `--spec none|copy` turn it, and so its cache, off
    let mtp = tp_boot::mtp_wanted(o, g);
    let opts = memory::engine_options(&memory_options(o, mtp));
    let profile = memory::memory_profile(g, &opts).map_err(|e| ModelError::Config(e.to_string()))?;
    let vision_reserve_bytes = match opts.vision {
        Some(v) => profile
            .vision_reserve_bytes(ENCODERS_IN_FLIGHT, v.concurrent_images(o.slots))
            .map_err(|e| ModelError::Config(e.to_string()))?,
        None => 0,
    };
    let free_after_weights = mem_available.saturating_sub(weights_bytes);
    let kv_tokens_cap = o.kv_tokens.unwrap_or(o.max_model_len) as u64;
    let plan = plan_memory(&profile, free_after_weights, o.floor_bytes, o.slots, kv_tokens_cap, vision_reserve_bytes);
    Ok(GlmPlanReport { profile, mtp, weights_bytes, free_after_weights, vision_reserve_bytes, plan })
}

/// Log the weight map of a mapped pack.
fn log_weight_map(family: &str, mapped: &GlmPack) {
    let t = mapped.plan.totals();
    let (groups, plain) = mapped.weights.counts();
    let b = &t.uploaded;
    eprintln!(
        "rexl3 event=weight_map family={family} outcome=ok exl3_groups={groups} plain_tensors={plain} load_items={} regions={} \
         region_gib={:.2} uploaded_gib={:.2} routed_gib={:.2} always_read_gib={:.2} embedding_gib={:.2} vision_gib={:.2} \
         mtp_gib={:.2} markers_not_uploaded_bytes={} routed_layers_by_bits={}",
        t.items,
        t.regions,
        gib(t.region_bytes),
        gib(t.uploaded_total()),
        gib(b.routed),
        gib(b.always_read()),
        gib(b.embedding),
        gib(b.vision),
        gib(b.mtp),
        t.skipped_total(),
        mapped.expert_bits_summary()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::Registry;
    use crate::serve::SpecMode;
    use std::path::PathBuf;

    const GLM_CONFIG: &str = include_str!("../../../../../testdata/glm53/config.json");
    const GLM_PROCESSOR: &str = include_str!("../../../../../testdata/glm53/processor_config.json");
    const GIB: u64 = 1 << 30;

    /// A fresh directory holding the given `config.json` and one tiny safetensors shard (a single bf16 tensor), so `Pack::open`
    /// has something to scan without any weights.
    fn pack_dir(tag: &str, config: &str, processor: Option<&str>) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rexl3-glm-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), config).unwrap();
        if let Some(p) = processor {
            std::fs::write(dir.join("processor_config.json"), p).unwrap();
        }
        let header = br#"{"model.language_model.embed_tokens.weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
        let mut shard = (header.len() as u64).to_le_bytes().to_vec();
        shard.extend_from_slice(header);
        shard.extend_from_slice(&[0, 0, 0, 0]);
        std::fs::write(dir.join("model-00001-of-00001.safetensors"), shard).unwrap();
        dir
    }

    fn options(dir: PathBuf) -> ServeOptions {
        let mut o = ServeOptions::new(dir);
        o.vision = Some(Default::default());
        o
    }

    fn config(o: &ServeOptions) -> GlmConfig {
        GlmConfig::from_dir(&o.model_dir).unwrap()
    }

    #[test]
    fn the_probe_describes_the_pack_in_family_neutral_terms() {
        let dir = pack_dir("probe", GLM_CONFIG, Some(GLM_PROCESSOR));
        let plan = GlmFamily.probe(&dir).unwrap();
        assert_eq!((plan.family, plan.chat_format), ("glm5_next", "glm5_next"));
        assert_eq!((plan.layers.len(), plan.mtp_layers.len()), (45, 1));
        assert_eq!(plan.drafters, vec!["mtp", "dflash2", "copy"], "the pack's NextN layer, a DFlash2 draft and the copy drafter");
        assert_eq!(plan.required_components, vec!["exl3_linear", "kda", "mla_dsa", "mhc", "moe"]);
        assert!(plan.missing_components().is_empty());
        assert_eq!(plan.unbuilt_vision(), None, "the tower is built");
        let v = plan.vision.as_ref().unwrap();
        assert_eq!((v.tower, v.patch_size, v.merge_size, v.temporal_patch_size, v.out_hidden), ("vision_glm5", 14, 2, 2, 4096));
        assert_eq!(v.image_token_id, 154_854);
        assert_eq!(v.max_image_tokens, 8000, "the processor's own cap is the family's default per-image limit");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_glm_pack_that_does_not_map_is_refused_by_name() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        // the one-tensor pack of `pack_dir` has an embedding of the wrong shape and none of the other 1,213 tensors
        let dir = pack_dir("not-mapped", GLM_CONFIG, Some(GLM_PROCESSOR));
        let mut o = options(dir.clone());
        // the one-tensor pack is not on the validated list (`serve::check_pack_dirs` runs before the build): served as unvalidated, so the start
        // reaches the GLM build and its layout check
        o.allow_unvalidated_pack = true;
        // with images or without, the preflight passes (it reads no tensor) and the starter maps the pack and refuses the layout by name, before any
        // tower is loaded
        assert!(crate::serve::preflight(&o).is_ok());
        let e = crate::serve::start(&o).err().unwrap();
        assert!(matches!(e, ModelError::Config(_)), "a layout mismatch is its own error, not NotBuilt: {e}");
        o.vision = None;
        assert!(crate::serve::preflight(&o).is_ok());
        let e = crate::serve::start(&o).err().unwrap();
        assert!(matches!(e, ModelError::Config(_)), "a layout mismatch is its own error, not NotBuilt: {e}");
        let text = e.to_string();
        assert!(text.contains("does not match the GLM-5.3 weight layout"), "{text}");
        assert!(text.contains("`model.language_model.embed_tokens.weight`: expected shape [154880, 4096], found [2]"), "{text}");
        assert!(text.contains("is not in the pack") && text.contains("more"), "{text}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_plan_on_the_target_box() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let dir = pack_dir("plan", GLM_CONFIG, Some(GLM_PROCESSOR));
        let o = options(dir.clone());
        let g = config(&o);
        // 121 GiB visible, a 91 GiB pack: 30 GiB free once it is resident
        // the default mode is `auto`: the NextN layer drafts, so its cache, ring, tap store and scratch are charged
        let r = glm_memory_plan(&o, &g, 91 * GIB, 121 * GIB).unwrap();
        assert!(r.mtp, "--spec auto (the default) uses the NextN layer");
        assert_eq!(r.free_after_weights, 30 * GIB);
        // one encoder's scratch (589,824,068) and 8 slots x 4 images of 64 MiB features (4096 tokens, f32)
        assert_eq!(r.vision_reserve_bytes, 589_824_068 + 32 * 67_108_864);
        let plan = r.plan.unwrap();
        // 30 GiB - 5 GiB reserve - 2.55 GiB vision - 8 GiB floor leaves 14.4 GiB: the 262,144-token cap governs
        assert_eq!((plan.kv_blocks, plan.kv_tokens), (1024 + 8, 262_144));
        // a page: 12 MLA layers (the 12th is the NextN layer's) x (256 x 1,024 B of latent + 64 pooled keys of 256 B)
        assert_eq!(plan.block_bytes, 12 * 278_528);
        // a slot: 34 KDA pools of 8,536,068 B + 12 rings of 131,072 B + a 4 KiB block-table row + the tap store (262,144 B); a token: 13,056 B
        assert_eq!(plan.state_bytes_per_slot, 291_672_200 + 131_072 + 262_144);
        assert_eq!(plan.admit_bytes, 8 * (291_672_200 + 131_072 + 262_144) + 262_144 * 13_056);
        assert_eq!((plan.engine_reserve_bytes, plan.vision_reserve_bytes), (5 * GIB, r.vision_reserve_bytes));
        // `--spec copy` and `--spec none` never load the layer and charge nothing for it
        for spec in [SpecMode::Copy, SpecMode::None] {
            let mut c = options(dir.clone());
            c.spec = spec;
            let r = glm_memory_plan(&c, &g, 91 * GIB, 121 * GIB).unwrap();
            assert!(!r.mtp, "{spec}");
            let plan = r.plan.unwrap();
            assert_eq!(plan.block_bytes, 11 * 278_528, "{spec}");
            assert_eq!(plan.state_bytes_per_slot, 291_672_200, "{spec}");
            assert_eq!(plan.admit_bytes, 8 * 291_672_200 + 262_144 * 11_968, "{spec}");
            assert_eq!(plan.admit_bytes, 5_470_716_992, "{spec}");
        }
        // no vision: no reserve and no tower weights
        let mut text_only = options(dir.clone());
        text_only.drafting = false;
        text_only.vision = None;
        let r = glm_memory_plan(&text_only, &g, 91 * GIB, 121 * GIB).unwrap();
        assert!(!r.mtp);
        assert_eq!(r.vision_reserve_bytes, 0);
        let plan = r.plan.unwrap();
        assert_eq!((plan.block_bytes, plan.vision_reserve_bytes), (11 * 278_528, 0));
        assert_eq!(plan.admit_bytes, 8 * 291_672_200 + 262_144 * 11_968);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_plan_fails_closed_when_the_pack_leaves_no_room() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let dir = pack_dir("tight", GLM_CONFIG, None);
        let o = options(dir.clone());
        let g = config(&o);
        // a pack that leaves 12 GiB: the 5 GiB reserve and 8 GiB floor take all of it
        let r = glm_memory_plan(&o, &g, 100 * GIB, 112 * GIB).unwrap();
        assert!(matches!(r.plan, Err(PlanError::NoRoom { allocatable: 0, .. })), "{:?}", r.plan);
        // a pack larger than memory: nothing is free, and the subtraction cannot wrap
        let r = glm_memory_plan(&o, &g, 200 * GIB, 100 * GIB).unwrap();
        assert_eq!(r.free_after_weights, 0);
        assert!(r.plan.is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn mtp_is_not_charged_for_a_pack_without_an_mtp_layer() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let mut v: serde_json::Value = serde_json::from_str(GLM_CONFIG).unwrap();
        v["text_config"]["num_nextn_predict_layers"] = serde_json::json!(0);
        let dir = pack_dir("nomtp", &v.to_string(), None);
        let o = options(dir.clone());
        let r = glm_memory_plan(&o, &config(&o), 91 * GIB, 121 * GIB).unwrap();
        assert!(!r.mtp, "drafting is on but the pack has no NextN layer");
        assert_eq!(r.plan.unwrap().block_bytes, 11 * 278_528);
        assert_eq!(GlmFamily.probe(&dir).unwrap().drafters, vec!["dflash2", "copy"]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn pack_weight_bytes_counts_the_tower_only_when_it_will_load() {
        use gb_exl3::checkpoint::Entry;
        let mut ck = Checkpoint::default();
        let entry = |bytes: u64| Entry { file: "f".into(), dtype: "BF16".into(), shape: vec![1], offsets: (100, 100 + bytes) };
        ck.plain.insert("model.language_model.embed_tokens.weight".into(), entry(1_000));
        ck.plain.insert("model.visual.blocks.0.attn.qkv.weight".into(), entry(300));
        ck.plain.insert("model.visual.merger.proj.weight".into(), entry(20));
        assert_eq!(pack_weight_bytes(&ck, true), 1_320);
        assert_eq!(pack_weight_bytes(&ck, false), 1_000);
        // the EXL3 tower: its groups (and their biases, counted in the group's bytes) are left out with the tower, the text groups never
        let group = |prefix: &str, bytes: u64| gb_exl3::checkpoint::Group {
            prefix: prefix.into(),
            bits: gb_exl3::Bits::from_twice(12).unwrap(),
            codebook: gb_exl3::Codebook::Mul1,
            k: 1024,
            n: 1024,
            in_scales: "suh",
            out_scales: "svh",
            bias: prefix.starts_with("model.visual."),
            files: vec!["f".into()],
            bytes,
        };
        for (prefix, bytes) in
            [("model.visual.blocks.0.attn.q_proj", 4_000), ("model.visual.merger.proj", 500), ("lm_head", 70_000)]
        {
            ck.groups.insert(prefix.into(), group(prefix, bytes));
        }
        assert_eq!(pack_weight_bytes(&ck, true), 1_320 + 4_500 + 70_000);
        assert_eq!(pack_weight_bytes(&ck, false), 1_000 + 70_000);
    }

    #[test]
    fn head_dim_zero_cannot_reach_admission_end_to_end() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        // the real config (head_dim 0), through the registry, the plan and an admission that charges every token
        let dir = pack_dir("hazard", GLM_CONFIG, Some(GLM_PROCESSOR));
        let (_, plan) = Registry::global().probe(&dir).unwrap();
        let profile = plan.memory_profile(&crate::memory::MemoryOptions::serving_defaults()).unwrap();
        assert_eq!(profile.kv_bytes_per_token(), 11_968);
        let p = plan_memory(&profile, 40 * GIB, 8 * GIB, 4, 262_144, 0).unwrap();
        let mm = gb_sched::ModelMemory::from_profile(&profile, 0, 0);
        let mut a = gb_sched::Admission::new(p.admit_bytes, 0, mm, 4).unwrap();
        assert!(a.cost(1) >= 11_968 + profile.state_bytes_per_slot());
        assert!(a.admit(262_144).is_ok());
        // every token is charged: the planned pool holds exactly one full context, so a second one is refused, not admitted free
        assert!(matches!(a.admit(262_144), Err(gb_sched::AdmitError::NoMemory { .. })));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
