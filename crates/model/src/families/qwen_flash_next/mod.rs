//! Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`): text forward, MTP and prefix cache.
//!
//! The text components are built; the vision tower is not. `unbuilt_vision` refuses images
//! at preflight, while the text engine can start. GPU correctness is gated separately.

pub mod backend;
pub mod config;
pub mod device;
pub mod engine;
pub mod graph_cache;
pub mod inventory;
pub mod memory;
pub mod model;
pub mod ngram;
pub mod prefix;
pub mod weights;

use crate::error::ModelError;
use crate::family::{BuildCtx, FamilyError, FamilyPlan, MemorySource, ModelFamily, VisionSpec};
use crate::memory::MemoryOptions;
use config::FlashNextConfig;
use gb_engine::ServeEngine;
use gb_sched::{MemoryError, MemoryProfile};
use serde_json::Value;
use std::path::Path;

/// Everything a Flash-Next engine runs on. The built ones are shared with the other families.
pub const REQUIRED: &[&str] =
    &["exl3_linear", "gdn", "gqa_attention_kv2", "qsa_indexer", "ngram_embedding", "hc_lowrank", "moe_softmax_topk"];

pub struct FlashNextFamily;

impl ModelFamily for FlashNextFamily {
    fn id(&self) -> &'static str {
        "qwen_flash_next"
    }
    fn architectures(&self) -> &'static [&'static str] {
        &["Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM"]
    }
    fn model_types(&self) -> &'static [&'static str] {
        &["qwen4_exp", "qwen4_exp_text"]
    }

    fn probe(&self, dir: &Path) -> Result<FamilyPlan, FamilyError> {
        let root = read_config(dir)?;
        crate::family::require_exl3(&root)?;
        let cfg = FlashNextConfig::from_json(&root).map_err(FamilyError::Config)?;
        cfg.check_shapes().map_err(FamilyError::Config)?;
        let mut required: Vec<&'static str> = REQUIRED.to_vec();
        let mut drafters = vec!["copy"];
        if cfg.mtp_layers > 0 {
            required.push("mtp_flash_next");
            drafters.insert(0, "mtp");
        }
        let vision = vision_spec(&root);
        // Vision is optional: preflight refuses image input via unbuilt_vision,
        // while a text-only request may start this engine.
        Ok(FamilyPlan {
            family: self.id(),
            dir: dir.to_path_buf(),
            layers: cfg.layer_specs(),
            mtp_layers: cfg.mtp_specs(),
            memory: Box::new(FlashNextMemorySource(cfg)),
            vision,
            chat_format: "qwen",
            drafters,
            required_components: required,
        })
    }

    fn prefix_cache(&self) -> bool {
        true
    }

    fn build(&self, plan: FamilyPlan, ctx: &BuildCtx<'_>) -> Result<Box<dyn ServeEngine>, ModelError> {
        if !plan.missing_components().is_empty() {
            return Err(plan.not_built("Qwen3.8-Flash-Next engine"));
        }
        Ok(Box::new(engine::build_engine(ctx.options)?))
    }
}

fn read_config(dir: &Path) -> Result<Value, FamilyError> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| FamilyError::Config(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| FamilyError::Config(format!("config.json: {e}")))
}

/// The vision component from `vision_config`, when the pack has a complete one (the tower is not built: a server asked for
/// images refuses by naming it).
fn vision_spec(root: &Value) -> Option<VisionSpec> {
    let v = root.get("vision_config")?;
    let get = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
    let image = root.get("image_token_id").and_then(|x| x.as_u64())?;
    Some(VisionSpec {
        tower: "vision_flash_next",
        patch_size: get("patch_size")?,
        merge_size: get("spatial_merge_size")?,
        temporal_patch_size: get("temporal_patch_size")?,
        out_hidden: get("out_hidden_size")?,
        image_token_id: image as u32,
        max_image_tokens: 4096,
    })
}

/// The memory model of a Flash-Next pack, ready to be evaluated at any serving settings.
pub struct FlashNextMemorySource(pub FlashNextConfig);

impl MemorySource for FlashNextMemorySource {
    fn profile(&self, opts: &MemoryOptions) -> Result<MemoryProfile, MemoryError> {
        memory::memory_profile(&self.0, opts)
    }
}

#[cfg(test)]
mod tests {
    use super::config::tests::FLASH_NEXT_CONFIG;
    use super::*;
    use crate::family::{Ffn, Mixer, Registry, Residual};

    fn dir_with(name: &str, cfg: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("rexl3-fn-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), cfg).unwrap();
        dir
    }

    #[test]
    fn the_registry_sends_a_flash_next_pack_here_and_the_plan_names_what_is_missing() {
        let dir = dir_with("probe", FLASH_NEXT_CONFIG);
        let (fam, plan) = Registry::global().probe(&dir).unwrap();
        assert_eq!((fam.id(), plan.family), ("qwen_flash_next", "qwen_flash_next"));
        assert_eq!((plan.layers.len(), plan.mtp_layers.len()), (48, 1));
        assert_eq!(plan.layers.iter().filter(|l| l.mixer == Mixer::Gqa).count(), 12);
        assert!(plan
            .layers
            .iter()
            .chain(&plan.mtp_layers)
            .all(|l| l.ffn == Ffn::Moe && l.residual == Residual::Mhc { streams: 4 }));
        assert_eq!((plan.chat_format, plan.drafters.clone()), ("qwen", vec!["mtp", "copy"]));
        assert!(plan.missing_components().is_empty(), "text components are built");
        assert_eq!(plan.unbuilt_vision(), Some("vision_flash_next"));
        assert!(fam.prefix_cache());
        // the plan's memory model evaluates at serving settings
        let p = plan.memory_profile(&MemoryOptions { mtp: true, vision: None, ..MemoryOptions::serving_defaults() }).unwrap();
        assert_eq!(p.family(), "qwen_flash_next");
        // A build with an unsupported window refuses before loading weights or CUDA.
        let mut o = crate::serve::ServeOptions::new(dir.clone());
        o.window_rows = 128;
        let e = match FlashNextFamily.build(plan, &BuildCtx { options: &o }) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("built with unsupported window width"),
        };
        assert!(e.contains("--window-rows"), "{e}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_text_only_pack_without_mtp_drops_those_components() {
        let mut v: Value = serde_json::from_str(FLASH_NEXT_CONFIG).unwrap();
        v.as_object_mut().unwrap().remove("vision_config");
        v["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(0);
        let dir = dir_with("text", &v.to_string());
        let plan = FlashNextFamily.probe(&dir).unwrap();
        assert!(plan.vision.is_none() && plan.mtp_layers.is_empty());
        assert_eq!(plan.drafters, vec!["copy"]);
        assert!(
            !plan.required_components.contains(&"mtp_flash_next") && !plan.required_components.contains(&"vision_flash_next")
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_non_exl3_or_misshapen_pack_is_refused_at_the_probe() {
        let mut v: Value = serde_json::from_str(FLASH_NEXT_CONFIG).unwrap();
        v["quantization_config"]["quant_method"] = serde_json::json!("gguf");
        let dir = dir_with("gguf", &v.to_string());
        let e = FlashNextFamily.probe(&dir).err().unwrap();
        assert!(format!("{e:?}").contains("EXL3"), "{e:?}");
        let mut v: Value = serde_json::from_str(FLASH_NEXT_CONFIG).unwrap();
        v["text_config"]["num_key_value_heads"] = serde_json::json!(4);
        std::fs::write(dir.join("config.json"), v.to_string()).unwrap();
        let e = FlashNextFamily.probe(&dir).err().unwrap();
        assert!(format!("{e:?}").contains("num_key_value_heads 4"), "{e:?}");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
