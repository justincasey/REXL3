//! Startup checks of what a pack's configuration says against what the kernels were compiled for (release plan R11, R13).
//!
//! The attention, GDN and draft kernels are built for one set of shapes (24 query heads, 4 KV heads of 256, 48 value heads of 128, a
//! draft of 5 layers over 5,120 channels, and so on). A pack whose `config.json` says something else would pass the loader, which
//! only checks tensor shapes against these same constants, and then read or write device memory out of bounds or compute with the
//! wrong geometry. Everything here fails at startup, before any weight is read, with a message that names the field, what the pack
//! has and what the kernels were built for.

use crate::config::QwenConfig;
use crate::error::ModelError;
use gb_cuda::glue::AttnPrepCfg;
use serde_json::Value;
use std::path::Path;

/// The oldest CUDA version the driver may report (`cudaDriverGetVersion`): the kernels are built with nvcc 13.0 for sm_121.
pub const MIN_DRIVER_CUDA: (u32, u32) = (13, 0);

/// Collects mismatches so one startup error lists every one.
#[derive(Default)]
struct Mismatches(Vec<String>);

impl Mismatches {
    fn eq<T: PartialEq + std::fmt::Display>(&mut self, what: &str, has: T, compiled: T) {
        if has != compiled {
            self.0.push(format!("{what} is {has}, the kernels are built for {compiled}"));
        }
    }
    fn finish(self, whose: &str) -> Result<(), String> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(format!("{whose} does not match the compiled kernels: {}", self.0.join("; ")))
        }
    }
}

/// The target's attention and GDN geometry against the kernels (`gb_cuda::attention`, `gb_cuda::gdn`, `AttnPrepCfg::QWEN_TARGET`), and
/// the layer indices the draft reads its context from against the number of layers.
pub fn check_target(c: &QwenConfig) -> Result<(), String> {
    use gb_cuda::{attention as a, gdn};
    let q = AttnPrepCfg::QWEN_TARGET;
    let mut m = Mismatches::default();
    m.eq("num_attention_heads", c.heads, a::N_Q_HEADS);
    m.eq("num_key_value_heads", c.kv_heads, a::N_KV_HEADS);
    m.eq("head_dim", c.head_dim, a::HEAD_DIM);
    m.eq("rotary dim (head_dim x partial_rotary_factor)", c.rotary_dim, q.rotary_dim);
    m.eq("rope_theta", c.rope_theta as f32, q.theta);
    m.eq("rms_norm_eps", c.rms_eps as f32, q.eps);
    m.eq("linear_num_key_heads", c.lin_key_heads, gdn::KEY_HEADS);
    m.eq("linear_num_value_heads", c.lin_value_heads, gdn::VALUE_HEADS);
    m.eq("linear_key_head_dim", c.lin_key_dim, gdn::HEAD_DIM);
    m.eq("linear_value_head_dim", c.lin_value_dim, gdn::HEAD_DIM);
    m.eq("linear_conv_kernel_dim", c.conv_kernel, gdn::CONV_TAPS);
    m.eq("conv channels (2 x key heads x key dim + value heads x value dim)", c.conv_channels(), gdn::CONV_DIM);
    if let Some(&l) = crate::forward::TAP_LAYERS.iter().find(|&&l| l >= c.layers.len()) {
        m.0.push(format!("the draft reads layer {l}, the model has {} layers", c.layers.len()));
    }
    m.finish("the target model's config.json")
}

/// The DFlash2 draft's `config.json`, the fields the kernels depend on.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub rms_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: usize,
    pub num_target_layers: usize,
    pub block_size: usize,
    pub conv_group: usize,
    pub conv_kernel: usize,
    pub mask_token: u64,
    pub selector_rank: usize,
    pub selector_top_k: usize,
    pub target_layer_ids: Vec<usize>,
}

impl DraftConfig {
    pub fn from_json(root: &Value) -> Result<Self, String> {
        let uz = |v: &Value, k: &str| -> Result<usize, String> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| format!("the draft's config.json: missing or invalid `{k}`"))
        };
        let fz = |v: &Value, k: &str| -> Result<f64, String> {
            v.get(k).and_then(|x| x.as_f64()).ok_or_else(|| format!("the draft's config.json: missing or invalid `{k}`"))
        };
        let d = root
            .get("dflash_config")
            .ok_or("the draft's config.json: missing `dflash_config` (is this a DFlash2 draft pack?)")?;
        let ids = d
            .get("target_layer_ids")
            .and_then(|x| x.as_array())
            .ok_or("the draft's config.json: missing `dflash_config.target_layer_ids`")?
            .iter()
            .map(|x| x.as_u64().map(|x| x as usize).ok_or("the draft's config.json: `target_layer_ids` must be integers"))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            hidden: uz(root, "hidden_size")?,
            layers: uz(root, "num_hidden_layers")?,
            heads: uz(root, "num_attention_heads")?,
            kv_heads: uz(root, "num_key_value_heads")?,
            head_dim: uz(root, "head_dim")?,
            intermediate: uz(root, "intermediate_size")?,
            vocab: uz(root, "vocab_size")?,
            rms_eps: fz(root, "rms_norm_eps")?,
            rope_theta: root
                .get("rope_parameters")
                .and_then(|r| r.get("rope_theta"))
                .and_then(|x| x.as_f64())
                .or_else(|| root.get("rope_theta").and_then(|x| x.as_f64()))
                .ok_or("the draft's config.json: missing `rope_parameters.rope_theta`")?,
            sliding_window: uz(root, "sliding_window")?,
            num_target_layers: uz(root, "num_target_layers")?,
            block_size: uz(d, "block_size")?,
            conv_group: uz(d, "conv_group_size")?,
            conv_kernel: uz(d, "conv_kernel_size")?,
            mask_token: uz(d, "mask_token_id")? as u64,
            selector_rank: uz(d, "selector_rank")?,
            selector_top_k: uz(d, "selector_top_k")?,
            target_layer_ids: ids,
        })
    }

    /// The shapes compiled into the draft kernels (`crate::draft`'s constants: layers, heads, the conv and the selector, the window
    /// and the trained block), which every DFlash2 pack must have.
    fn kernel_mismatches(&self) -> Mismatches {
        use crate::draft as d;
        let mut m = Mismatches::default();
        m.eq("num_hidden_layers", self.layers, d::LAYERS);
        m.eq("num_attention_heads", self.heads, d::Q_HEADS);
        m.eq("num_key_value_heads", self.kv_heads, d::KV_HEADS);
        m.eq("head_dim", self.head_dim, d::HEAD_DIM);
        m.eq("sliding_window", self.sliding_window, gb_cuda::dflash2::DraftCtxLayout::WINDOW);
        // the pack's block is what the checkpoint was TRAINED with (8): the runtime knob `REXL3_DRAFT_BLOCK` (8..=16) runs the
        // block longer, it does not retune the pack, so a pack trained with another block is still a different draft
        if self.block_size != d::BLOCK {
            m.0.push(format!(
                "dflash_config.block_size is {}, the draft was trained with a block of {} (REXL3_DRAFT_BLOCK may extend the block at runtime; the pack itself is the trained one)",
                self.block_size, d::BLOCK
            ));
        }
        m.eq("dflash_config.conv_group_size", self.conv_group, d::CONV_GROUP);
        m.eq("dflash_config.conv_kernel_size", self.conv_kernel, 2);
        m.eq("dflash_config.selector_rank", self.selector_rank, d::SELECTOR_RANK);
        m.eq("dflash_config.selector_top_k", self.selector_top_k, d::SELECTOR_TOP_K);
        m
    }

    /// The shapes and constants the draft runs with, for a target head whose first `valid_vocab` columns are tokens.
    pub fn dims(&self, valid_vocab: usize) -> crate::draft::DraftDims {
        crate::draft::DraftDims {
            hidden: self.hidden,
            intermediate: self.intermediate,
            taps: self.target_layer_ids.len(),
            mask_token: self.mask_token as u32,
            eps: self.rms_eps as f32,
            rope_theta: self.rope_theta,
            valid_vocab,
        }
    }

    /// The draft against the kernels and against the Qwen3.8 target it drafts for (its own shapes are the Qwen pack's,
    /// `DraftDims::QWEN38`).
    pub fn check(&self, target: &QwenConfig) -> Result<(), String> {
        let q = crate::draft::DraftDims::QWEN38;
        let mut m = self.kernel_mismatches();
        m.eq("hidden_size", self.hidden, q.hidden);
        m.eq("intermediate_size", self.intermediate, q.intermediate);
        m.eq("rms_norm_eps", self.rms_eps as f32, q.eps);
        m.eq("rope_theta", self.rope_theta, q.rope_theta);
        m.eq("dflash_config.mask_token_id", self.mask_token, q.mask_token as u64);
        if self.target_layer_ids != crate::forward::TAP_LAYERS {
            m.0.push(format!(
                "dflash_config.target_layer_ids is {:?}, the engine taps layers {:?}",
                self.target_layer_ids,
                crate::forward::TAP_LAYERS
            ));
        }
        // the draft must belong to this target
        if self.hidden != target.hidden {
            m.0.push(format!("hidden_size {} differs from the target's {}", self.hidden, target.hidden));
        }
        if self.vocab != target.vocab {
            m.0.push(format!("vocab_size {} differs from the target's {}", self.vocab, target.vocab));
        }
        if self.num_target_layers != target.layers.len() {
            m.0.push(format!(
                "num_target_layers {} differs from the target's {} layers",
                self.num_target_layers,
                target.layers.len()
            ));
        }
        if q.mask_token as usize >= target.vocab {
            m.0.push(format!("the mask token {} is outside the target's vocabulary of {}", q.mask_token, target.vocab));
        }
        m.finish("the draft's config.json")
    }

    /// The draft against the kernels and against a target of `hidden` channels, `layers` layers and a head (and embedding) of `vocab`
    /// rows (a family whose draft shapes come from the pack: GLM-5.3-Flash's). The taps must be ascending layers of the target, the
    /// mask a row of the target's embedding (GLM's is the first padding id), and the widths ones the 4-bit linears and the convs take.
    pub fn check_for(&self, hidden: usize, layers: usize, vocab: usize) -> Result<(), String> {
        let mut m = self.kernel_mismatches();
        if self.hidden != hidden {
            m.0.push(format!("hidden_size {} differs from the target's {hidden}", self.hidden));
        }
        if self.vocab != vocab {
            m.0.push(format!("vocab_size {} differs from the target's {vocab}", self.vocab));
        }
        if self.num_target_layers != layers {
            m.0.push(format!("num_target_layers {} differs from the target's {layers} layers", self.num_target_layers));
        }
        let ids = &self.target_layer_ids;
        if ids.is_empty() || ids.windows(2).any(|w| w[0] >= w[1]) || ids.iter().any(|&l| l >= layers) {
            m.0.push(format!("dflash_config.target_layer_ids {ids:?} must be ascending layers of the target's {layers}"));
        }
        if self.mask_token as usize >= vocab {
            m.0.push(format!("the mask token {} is outside the target's embedding of {vocab} rows", self.mask_token));
        }
        for (what, v) in [("hidden_size", self.hidden), ("intermediate_size", self.intermediate)] {
            if v == 0 || v % gb_cuda::q4_linear::GROUP != 0 {
                m.0.push(format!("{what} {v} is not a multiple of {}", gb_cuda::q4_linear::GROUP));
            }
        }
        m.finish("the draft's config.json")
    }
}

/// Read and parse `dir/config.json` of a draft pack.
pub fn read_draft_config(dir: &Path) -> Result<DraftConfig, ModelError> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| ModelError::Config(format!("{}: {e}", path.display())))?;
    let root: Value = serde_json::from_str(&text).map_err(|e| ModelError::Config(format!("{}: {e}", path.display())))?;
    DraftConfig::from_json(&root).map_err(ModelError::Config)
}

/// Read `dir/config.json` of a draft pack and check it (see [`DraftConfig::check`]). A missing or unreadable config is an error: the
/// draft's shapes are compiled in, so the pack has to say they are the ones it was trained with.
pub fn check_draft_dir(dir: &Path, target: &QwenConfig) -> Result<(), ModelError> {
    read_draft_config(dir)?.check(target).map_err(ModelError::Config)
}

/// The driver must report CUDA 13.0 or newer. A missing driver is its own message.
pub fn check_driver(v: &gb_cuda::rt::Versions) -> Result<(), String> {
    let Some(d) = v.driver.as_deref() else {
        return Err("no CUDA driver is available (cudaDriverGetVersion failed): the GPU backend needs the NVIDIA driver".into());
    };
    let (maj, min) = d
        .split_once('.')
        .and_then(|(a, b)| Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?)))
        .ok_or_else(|| format!("cannot read the CUDA driver version `{d}`"))?;
    if (maj, min) < MIN_DRIVER_CUDA {
        return Err(format!(
            "the NVIDIA driver supports CUDA {maj}.{min}, but this build needs CUDA {}.{} or newer (it was built with nvcc {} for {}): update the driver",
            MIN_DRIVER_CUDA.0, MIN_DRIVER_CUDA.1, v.nvcc, v.arch
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn target() -> QwenConfig {
        QwenConfig::from_json(&serde_json::from_str(include_str!("../../../testdata/qwen38/config.json")).unwrap()).unwrap()
    }

    fn draft_json() -> Value {
        serde_json::from_str(include_str!("../../../testdata/qwen38/draft_config.json")).unwrap()
    }

    #[test]
    fn the_real_target_and_draft_configs_match_the_kernels() {
        check_target(&target()).unwrap();
        DraftConfig::from_json(&draft_json()).unwrap().check(&target()).unwrap();
    }

    #[test]
    fn the_glm_draft_config_gives_its_own_dims_and_checks_against_a_glm_shaped_target() {
        let v: Value = serde_json::from_str(include_str!("../../../testdata/glm53/dflash2_draft_config.json")).unwrap();
        let d = DraftConfig::from_json(&v).unwrap();
        d.check_for(4096, 45, 154880).unwrap();
        let dims = d.dims(154856);
        assert_eq!((dims.hidden, dims.intermediate, dims.taps, dims.mask_token), (4096, 12288, 5, 154856));
        assert_eq!((dims.eps, dims.rope_theta, dims.fc_in()), (1e-5, 1e4, 20480));
        assert_eq!(d.target_layer_ids, [5, 14, 24, 33, 42]);
        // a target it does not belong to, and a mask past the embedding, are refused by name
        let e = d.check_for(5120, 45, 154880).unwrap_err();
        assert!(e.contains("hidden_size 4096 differs"), "{e}");
        let e = d.check_for(4096, 40, 154880).unwrap_err();
        assert!(e.contains("num_target_layers 45") && e.contains("target_layer_ids"), "{e}");
        let mut small = d.clone();
        small.vocab = 154800;
        let e = small.check_for(4096, 45, 154800).unwrap_err();
        assert!(e.contains("mask token 154856"), "{e}");
        // the Qwen check refuses it: its shapes are not the Qwen pack's
        assert!(d.check(&target()).is_err());
    }

    #[test]
    fn a_target_with_other_attention_or_gdn_shapes_is_refused_by_field() {
        // each of these used to pass startup and index out of bounds on the device
        for (field, value, needle) in [
            ("num_attention_heads", json!(32), "num_attention_heads is 32, the kernels are built for 24"),
            ("num_key_value_heads", json!(8), "num_key_value_heads is 8, the kernels are built for 4"),
            ("head_dim", json!(128), "head_dim is 128, the kernels are built for 256"),
            ("linear_num_key_heads", json!(8), "linear_num_key_heads is 8, the kernels are built for 16"),
            ("linear_num_value_heads", json!(32), "linear_num_value_heads is 32, the kernels are built for 48"),
            ("linear_key_head_dim", json!(64), "linear_key_head_dim is 64, the kernels are built for 128"),
            ("linear_conv_kernel_dim", json!(5), "linear_conv_kernel_dim is 5, the kernels are built for 4"),
            ("rms_norm_eps", json!(1e-5), "rms_norm_eps"),
        ] {
            let mut root: Value = serde_json::from_str(include_str!("../../../testdata/qwen38/config.json")).unwrap();
            let t = if root.get("text_config").is_some() { &mut root["text_config"] } else { &mut root };
            t[field] = value;
            let cfg = QwenConfig::from_json(&root).unwrap_or_else(|e| panic!("{field}: the parser refuses it first: {e}"));
            let e = check_target(&cfg).unwrap_err();
            assert!(e.contains(needle), "{field}: {e}");
            assert!(e.contains("does not match the compiled kernels"), "{e}");
        }
        // several mismatches are reported together
        let mut root: Value = serde_json::from_str(include_str!("../../../testdata/qwen38/config.json")).unwrap();
        let t = if root.get("text_config").is_some() { &mut root["text_config"] } else { &mut root };
        t["head_dim"] = json!(128);
        t["num_key_value_heads"] = json!(8);
        let e = check_target(&QwenConfig::from_json(&root).unwrap()).unwrap_err();
        assert!(e.contains("head_dim") && e.contains("num_key_value_heads"), "{e}");
    }

    #[test]
    fn a_draft_with_other_shapes_or_for_another_target_is_refused_by_field() {
        for (path, value, needle) in [
            ("hidden_size", json!(4096), "hidden_size is 4096, the kernels are built for 5120"),
            ("num_hidden_layers", json!(4), "num_hidden_layers is 4, the kernels are built for 5"),
            ("num_attention_heads", json!(16), "num_attention_heads is 16"),
            ("head_dim", json!(64), "head_dim is 64"),
            ("intermediate_size", json!(8192), "intermediate_size is 8192"),
            ("sliding_window", json!(1024), "sliding_window is 1024"),
            ("vocab_size", json!(150000), "vocab_size 150000 differs from the target's 248320"),
            ("num_target_layers", json!(48), "num_target_layers 48 differs"),
        ] {
            let mut j = draft_json();
            j[path] = value;
            let e = DraftConfig::from_json(&j).unwrap().check(&target()).unwrap_err();
            assert!(e.contains(needle), "{path}: {e}");
        }
        for (path, value, needle) in [
            ("block_size", json!(16), "block_size is 16, the draft was trained with a block of 8"),
            ("mask_token_id", json!(1), "mask_token_id is 1"),
            ("selector_rank", json!(128), "selector_rank is 128"),
            ("conv_kernel_size", json!(3), "conv_kernel_size is 3"),
            ("target_layer_ids", json!([4, 19, 33, 47, 61]), "target_layer_ids is [4, 19, 33, 47, 61]"),
        ] {
            let mut j = draft_json();
            j["dflash_config"][path] = value;
            let e = DraftConfig::from_json(&j).unwrap().check(&target()).unwrap_err();
            assert!(e.contains(needle), "{path}: {e}");
        }
        // a config without the DFlash2 section is not a DFlash2 draft
        let mut j = draft_json();
        j.as_object_mut().unwrap().remove("dflash_config");
        assert!(DraftConfig::from_json(&j).unwrap_err().contains("dflash_config"));
    }

    #[test]
    fn a_draft_folder_with_no_readable_config_is_an_error() {
        let dir = std::env::temp_dir().join(format!("rexl3-draft-cfg-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(check_draft_dir(&dir, &target()).unwrap_err().to_string().contains("config.json"));
        std::fs::write(dir.join("config.json"), "{ not json").unwrap();
        assert!(check_draft_dir(&dir, &target()).is_err());
        std::fs::write(dir.join("config.json"), include_str!("../../../testdata/qwen38/draft_config.json")).unwrap();
        assert!(check_draft_dir(&dir, &target()).is_ok());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_driver_below_cuda_13_0_is_a_clear_error() {
        let mut v = gb_cuda::rt::versions();
        v.driver = Some("13.0".into());
        assert!(check_driver(&v).is_ok());
        v.driver = Some("13.2".into());
        assert!(check_driver(&v).is_ok());
        v.driver = Some("14.0".into());
        assert!(check_driver(&v).is_ok());
        v.driver = Some("12.9".into());
        let e = check_driver(&v).unwrap_err();
        assert!(e.contains("supports CUDA 12.9") && e.contains("13.0 or newer") && e.contains("update the driver"), "{e}");
        v.driver = None;
        assert!(check_driver(&v).unwrap_err().contains("no CUDA driver"));
        v.driver = Some("garbage".into());
        assert!(check_driver(&v).unwrap_err().contains("cannot read"));
    }
}
