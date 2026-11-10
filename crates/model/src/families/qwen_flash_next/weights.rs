//! CPU-only Flash-Next weight binding and upload recipe.
//!
//! [`WeightMap::bind`] claims every text tensor once from a scanned inventory, checks dtype and geometry,
//! and returns named slots (plain, EXL3 parts, or sidecar) with their conversion. [`WeightMap::load_plan`]
//! separates upload items from sidecar records: the EXL3 marker is validated by the scanner, not uploaded;
//! the n-gram table and hash metadata remain on the host for [`super::ngram::NgramTable`]. A forward
//! port addresses a slot by checkpoint name and reads `source` (file, absolute byte offset, byte count)
//! when bound with [`WeightMap::from_dir`]. `bind` is pure and uses synthetic offsets for inventory tests.
//! The executor must implement every [`Transform`], especially `Bf16AddOneToF32` for RMS norms.

use super::{
    config::FlashNextConfig,
    inventory::{self, Tensors},
};
use gb_exl3::checkpoint::{read_header_checked, scan};
use gb_exl3::{Bits, Codebook};
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    None,
    Bf16ToF32,
    Bf16AddOneToF32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Plain,
    Exl3Part,
    Marker,
    Sidecar,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Source {
    pub file: String,
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Destination {
    pub region: u32,
    pub offset: u64,
    pub bytes: u64,
}

/// One EXL3 linear, typed as the shared loader's bit width and codebook. Its four parts
/// are named `<prefix>.{trellis,suh,svh,mul1}` in `slots`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exl3Group {
    pub prefix: String,
    pub k: usize,
    pub n: usize,
    pub bits: Bits,
    pub codebook: Codebook,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub kind: Kind,
    pub transform: Transform,
    pub source: Option<Source>,
    pub destination: Option<Destination>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub name: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadPlan {
    /// Copies into family-owned regions; ordered by source file and offset when sources are known.
    pub items: Vec<Slot>,
    /// Host-only table shards/hash parameters and EXL3 markers already validated at scan time.
    pub sidecars: Vec<Slot>,
    /// Global, per-layer trunk, and per-layer routed expert allocations. Offsets are 256-byte aligned.
    pub regions: Vec<Region>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightMap {
    pub slots: BTreeMap<String, Slot>,
    pub groups: BTreeMap<String, Exl3Group>,
    pub regions: Vec<Region>,
}

fn specification(name: &str) -> (&'static str, Kind, Transform) {
    if name.contains(inventory::NGRAM_SHARD) {
        return ("I16", Kind::Sidecar, Transform::None);
    }
    if name.contains(".ple.ple_embedding.ngram_embedding.") {
        if name.ends_with(".head_bias") {
            return ("F16", Kind::Sidecar, Transform::None);
        }
        return ("I64", Kind::Sidecar, Transform::None);
    }
    if let Some(suffix) = name.rsplit('.').next() {
        match suffix {
            "trellis" => return ("I16", Kind::Exl3Part, Transform::None),
            "suh" | "svh" => return ("F16", Kind::Exl3Part, Transform::None),
            "mul1" => return ("I32", Kind::Marker, Transform::None),
            _ => {}
        }
    }
    if name == "model.language_model.embed_tokens.weight" {
        return ("BF16", Kind::Plain, Transform::None);
    }
    if name.contains(".linear_attn.") && (name.ends_with("A_log") || name.ends_with("dt_bias") || name.ends_with(".norm.weight"))
    {
        return ("BF16", Kind::Plain, Transform::Bf16ToF32);
    }
    if name.contains(".linear_attn.conv1d.") {
        return ("BF16", Kind::Plain, Transform::None);
    }
    if name.contains(".self_attn.") && name.ends_with("norm.weight") || name.contains(".ple.norm_") {
        return ("BF16", Kind::Plain, Transform::Bf16AddOneToF32);
    }
    if name == "mtp.pre_fc_norm_embedding.weight" {
        return ("BF16", Kind::Plain, Transform::Bf16AddOneToF32);
    }
    ("F16", Kind::Plain, Transform::None)
}

fn region_name(name: &str) -> String {
    let layer = name.strip_prefix("model.language_model.layers.").or_else(|| name.strip_prefix("mtp.layers."));
    if let Some(rest) = layer {
        if let Some((index, part)) = rest.split_once('.') {
            let prefix = if name.starts_with("mtp.") { "mtp" } else { "layer" };
            let kind = if part.starts_with("mlp.experts.") { "experts" } else { "trunk" };
            return format!("{prefix}_{index}_{kind}");
        }
    }
    if name.starts_with("mtp.") {
        "mtp_global".into()
    } else {
        "global".into()
    }
}

impl WeightMap {
    /// Bind the text forward's recipe, including MTP and both sidecars, exactly once. Vision is separate.
    pub fn bind(cfg: &FlashNextConfig, tensors: &Tensors) -> Result<Self, Vec<String>> {
        let (mut errors, claimed) = inventory::check_with_claims(cfg, tensors);
        // The supported pack's PLE sidecar is 128 contiguous shards. Without the final count a
        // truncated table can look like a valid, smaller table to a header-only inventory.
        if !cfg.ple_layer_ids.is_empty() {
            let (n, _) = inventory::ngram_table(cfg, tensors).unwrap_or((0, 0));
            if n != 128 {
                errors.push(format!(
                    "{}:{}{n}.trellis: expected 128 n-gram shards, found {n}",
                    inventory::NGRAM_FILE,
                    inventory::NGRAM_SHARD
                ));
            }
        }
        let mut slots = BTreeMap::new();
        let mut groups = BTreeMap::new();
        let mut regions: Vec<Region> = Vec::new();
        let mut region_ids: BTreeMap<String, u32> = BTreeMap::new();
        for name in &claimed {
            let Some((dtype, shape)) = tensors.get(name) else { continue };
            let (expected, kind, transform) = specification(name);
            if dtype != expected {
                errors.push(format!("{name}: dtype {dtype}, expected {expected}"));
            }
            if kind == Kind::Marker && !shape.is_empty() {
                errors.push(format!("{name}: marker must be scalar, got {shape:?}"));
            }
            if kind == Kind::Exl3Part && name.ends_with(".trellis") && shape.len() == 3 {
                let prefix = name.trim_end_matches(".trellis");
                if let Some(bits) = Bits::from_trellis_last(shape[2]) {
                    groups.insert(
                        prefix.to_string(),
                        Exl3Group {
                            prefix: prefix.to_string(),
                            k: shape[0] * 16,
                            n: shape[1] * 16,
                            bits,
                            codebook: Codebook::Mul1,
                        },
                    );
                } else {
                    errors.push(format!("{name}: unsupported EXL3 bit width"));
                }
            }
            let destination = if matches!(kind, Kind::Plain | Kind::Exl3Part) {
                let region = region_name(name);
                let id = *region_ids.entry(region.clone()).or_insert_with(|| {
                    let id = regions.len() as u32;
                    regions.push(Region { name: region, bytes: 0 });
                    id
                });
                let size = gb_exl3::checkpoint::dtype_size(expected).unwrap();
                let bytes = shape.iter().try_fold(size, |n, &d| n.checked_mul(d as u64));
                match bytes {
                    Some(bytes) => {
                        let r = &mut regions[id as usize];
                        let offset = r.bytes.div_ceil(256) * 256;
                        let bytes = if transform == Transform::None { bytes } else { bytes * 2 };
                        r.bytes = offset + bytes;
                        Some(Destination { region: id, offset, bytes })
                    }
                    None => {
                        errors.push(format!("{name}: byte size overflow"));
                        None
                    }
                }
            } else {
                None
            };
            slots.insert(
                name.clone(),
                Slot {
                    name: name.clone(),
                    dtype: dtype.clone(),
                    shape: shape.clone(),
                    kind,
                    transform,
                    source: None,
                    destination,
                },
            );
        }
        for region in &mut regions {
            region.bytes = region.bytes.div_ceil(256) * 256;
        }
        for name in tensors.keys().filter(|n| !n.starts_with("model.visual.") && !claimed.contains(*n)) {
            errors.push(format!("{name}: no Flash-Next text slot claims this tensor"));
        }
        if errors.is_empty() {
            Ok(Self { slots, groups, regions })
        } else {
            Err(errors)
        }
    }

    /// Scan headers and EXL3 codebook markers, not weight arrays; map both configured sidecars.
    pub fn from_dir(dir: &Path, cfg: &FlashNextConfig) -> Result<Self, Vec<String>> {
        let ck = scan(dir, true).map_err(|e| vec![e.to_string()])?;
        let bad: Vec<String> = ck
            .bad
            .iter()
            .filter(|(name, _)| !name.contains(inventory::NGRAM_SHARD))
            .map(|(name, why)| format!("{name}: refused EXL3 group: {why}"))
            .collect();
        if !bad.is_empty() {
            return Err(bad);
        }
        let issues: Vec<String> = ck
            .issues
            .iter()
            .filter(|issue| !issue.contains(" as the index says"))
            .map(|issue| format!("pack header: {issue}"))
            .collect();
        if !issues.is_empty() {
            return Err(issues);
        }
        let mut tensors = Tensors::new();
        let mut sources = BTreeMap::new();
        fn add(
            tensors: &mut Tensors,
            sources: &mut BTreeMap<String, Source>,
            name: &str,
            dtype: &str,
            shape: &[usize],
            source: Source,
        ) {
            tensors.insert(name.to_string(), (dtype.to_string(), shape.to_vec()));
            sources.insert(name.to_string(), source);
        }
        for (name, entry) in &ck.plain {
            let offset = ck.file_offset(entry).map_err(|e| vec![e.to_string()])?;
            add(
                &mut tensors,
                &mut sources,
                name,
                &entry.dtype,
                &entry.shape,
                Source { file: entry.file.clone(), offset, bytes: entry.nbytes() },
            );
        }
        for (prefix, group) in &ck.groups {
            let mut parts = vec!["trellis", group.in_scales, group.out_scales, group.codebook.name()];
            if group.bias {
                parts.push("bias");
            }
            for suffix in parts {
                let name = format!("{prefix}.{suffix}");
                if let Some(entry) = ck.part(&name) {
                    let offset = ck.file_offset(entry).map_err(|e| vec![e.to_string()])?;
                    add(
                        &mut tensors,
                        &mut sources,
                        &name,
                        &entry.dtype,
                        &entry.shape,
                        Source { file: entry.file.clone(), offset, bytes: entry.nbytes() },
                    );
                }
            }
        }
        // The scanner declines n-gram shard trellises as EXL3 linears (no scales).
        // Read their checked header independently; this does not map or read the 32 GB payload.
        // The MTP mixer patch file is turboderp's packaging; a pack that carries the same three tensors inside its
        // shards. A missing patch file is then fine, and `bind` still names any mixer tensor that is missing from both.
        for file in [inventory::NGRAM_FILE, inventory::MTP_MIXER_FILE].into_iter().filter(|file| {
            (*file != inventory::NGRAM_FILE || !cfg.ple_layer_ids.is_empty())
                && (*file != inventory::MTP_MIXER_FILE || (cfg.mtp_layers > 0 && dir.join(file).exists()))
        }) {
            let path = dir.join(file);
            let (header, data_len, entries) = read_header_checked(&path).map_err(|e| vec![e.to_string()])?;
            for (name, v) in entries {
                let dtype = v["dtype"].as_str().ok_or_else(|| vec![format!("{file}:{name}: dtype")])?;
                let shape: Vec<usize> = v["shape"]
                    .as_array()
                    .ok_or_else(|| vec![format!("{file}:{name}: shape")])?
                    .iter()
                    .map(|d| d.as_u64().map(|n| n as usize).ok_or_else(|| vec![format!("{file}:{name}: dimension")]))
                    .collect::<Result<_, _>>()?;
                let o = v["data_offsets"].as_array().ok_or_else(|| vec![format!("{file}:{name}: offsets")])?;
                let start = o.first().and_then(|x| x.as_u64()).ok_or_else(|| vec![format!("{file}:{name}: start")])?;
                let end = o.get(1).and_then(|x| x.as_u64()).ok_or_else(|| vec![format!("{file}:{name}: end")])?;
                let element =
                    gb_exl3::checkpoint::dtype_size(dtype).ok_or_else(|| vec![format!("{file}:{name}: dtype {dtype}")])?;
                let count = shape
                    .iter()
                    .try_fold(1u64, |n, d| n.checked_mul(*d as u64))
                    .ok_or_else(|| vec![format!("{file}:{name}: overflow")])?;
                if start > end || end > data_len || count.checked_mul(element) != Some(end - start) {
                    return Err(vec![format!("{file}:{name}: invalid payload extent")]);
                }
                // The scanner may also have this tensor in `plain`; identical metadata is required.
                if let Some((old_dtype, old_shape)) = tensors.get(&name) {
                    if old_dtype != dtype || old_shape != &shape {
                        return Err(vec![format!("{file}:{name}: conflicting duplicate")]);
                    }
                }
                add(
                    &mut tensors,
                    &mut sources,
                    &name,
                    dtype,
                    &shape,
                    Source { file: file.to_string(), offset: 8 + header + start, bytes: end - start },
                );
            }
        }
        let mut map = Self::bind(cfg, &tensors)?;
        for (name, slot) in &mut map.slots {
            slot.source = sources.remove(name);
        }
        Ok(map)
    }

    pub fn load_plan(&self) -> LoadPlan {
        let (mut items, mut sidecars) = (Vec::new(), Vec::new());
        for slot in self.slots.values() {
            if matches!(slot.kind, Kind::Sidecar | Kind::Marker) {
                sidecars.push(slot.clone());
            } else {
                items.push(slot.clone());
            }
        }
        items.sort_by(|a, b| {
            a.source
                .as_ref()
                .map(|p| (&p.file, p.offset))
                .cmp(&b.source.as_ref().map(|p| (&p.file, p.offset)))
                .then(a.name.cmp(&b.name))
        });
        LoadPlan { items, sidecars, regions: self.regions.clone() }
    }
}
