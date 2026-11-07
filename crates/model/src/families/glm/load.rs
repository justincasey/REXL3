//! Realising the load plan on the device: one `cudaMalloc` per plan region, then every item of the plan through the Phase 3 pipelined
//! loader (`gb_cuda::loader`): reader threads, `O_DIRECT` for cold spans, copies overlapped with the next read.
//!
//! The plan is the contract (`plan.rs`): an item says which bytes of which shard go where in which region, and through which
//! [`Xform`]. This module owes the plan its three promises and nothing else:
//!
//! * **Every byte lands where the plan says** (a view of the region at the item's offset is the loader's destination).
//! * **Every conversion the plan names is made, and only those** ([`convert`], pure and tested on the CPU): gammas widen to fp32, each
//!   mHC `fn` narrows to fp16 and fails the load if an element would round, the router bias is centred, `kv_b_proj` is packed.
//! * **Every codebook marker is checked** inside the reads the loader makes anyway (`Loader::check_file`): a pack decoded with the
//!   wrong codebook is silently wrong, so a marker that differs from the plan's ends the load, naming the group.
//!
//! The pool is one 121 GiB unified memory shared with the host, and `cudaMalloc` commits eagerly, so the regions are allocated one by one
//! and `MemAvailable` is checked before each: a load that would leave less than `floor_bytes` is refused before it can over-commit.

use super::plan::{ItemPart, LoadItem, LoadPlan, RegionKind};
use super::weights::Xform;
use crate::error::ModelError;
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::loader::{Loader, Options, Report, Xform as CopyXform};
use gb_cuda::rt::Block;
use gb_cuda::{mla, moe, DevBuf};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_exl3::loadstats::{self, Component, Stage};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

/// What a load does about memory and threads.
#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// Refuse to allocate a region that would leave less than this much `MemAvailable`.
    pub floor_bytes: u64,
    /// The pipelined loader's options (workers, staging slots, span size, `O_DIRECT` mode).
    pub loader: Options,
}

impl LoadOptions {
    /// `floor_bytes` and the loader's options from the environment (`REXL3_LOAD_WORKERS`, ...); the default is 16 reader threads: the pack
    /// is on one NVMe drive whose throughput grows with the number of threads reading it (1.1 GB/s at 1, 7.5 GB/s at 16: docs/GLM_LOAD.md).
    pub fn from_env(floor_bytes: u64) -> Self {
        let mut loader = Options::from_env();
        if std::env::var("REXL3_LOAD_WORKERS").is_err() {
            loader.workers = 16;
        }
        Self { floor_bytes, loader }
    }
}

/// The device regions of a plan, indexed by [`RegionId`] (`None` for a region the plan does not use).
pub struct Regions {
    views: Vec<Option<DevBuf>>,
}

impl Regions {
    pub fn get(&self, id: u32) -> Result<&DevBuf, ModelError> {
        self.views
            .get(id as usize)
            .and_then(|v| v.as_ref())
            .ok_or_else(|| ModelError::Config(format!("the load plan names region {id}, which was not allocated")))
    }
    /// A view of `bytes` bytes at `offset` into region `id`.
    pub fn view(&self, id: u32, offset: u64, bytes: u64) -> Result<DevBuf, ModelError> {
        let r = self.get(id)?;
        if offset + bytes > r.bytes() as u64 {
            return Err(ModelError::Config(format!(
                "a view of {bytes} bytes at {offset} runs past region {id} ({} bytes)",
                r.bytes()
            )));
        }
        Ok(r.subview(offset as usize, bytes as usize))
    }
    /// Bytes of device memory the regions hold.
    pub fn bytes(&self) -> u64 {
        self.views.iter().flatten().map(|v| v.bytes() as u64).sum()
    }
    pub fn count(&self) -> usize {
        self.views.iter().flatten().count()
    }
}

/// `MemAvailable` in bytes (`None` where /proc/meminfo has none).
fn mem_available() -> Option<u64> {
    crate::serve::mem_available()
}

/// Allocate every region of `plan`, one `cudaMalloc` each, checking `MemAvailable` before each. Contents are uninitialised until
/// [`run`] has filled them.
pub fn allocate(plan: &LoadPlan, floor_bytes: u64) -> Result<Regions, ModelError> {
    let max_id = plan.regions.iter().map(|r| r.id as usize).max().map_or(0, |m| m + 1);
    let mut views: Vec<Option<DevBuf>> = (0..max_id).map(|_| None).collect();
    let total: u64 = plan.regions.iter().map(|r| r.bytes).sum();
    for r in &plan.regions {
        if let Some(avail) = mem_available() {
            if avail < r.bytes.saturating_add(floor_bytes) {
                return Err(ModelError::Config(format!(
                    "refusing to allocate region {} ({}, {:.2} GiB): MemAvailable {:.2} GiB would fall below the {:.2} GiB floor (the plan needs {:.2} GiB in all); {} of {} regions are allocated",
                    r.id,
                    r.kind,
                    gib(r.bytes),
                    gib(avail),
                    gib(floor_bytes),
                    gib(total),
                    views.iter().flatten().count(),
                    plan.regions.len()
                )));
            }
        }
        let block = Arc::new(Block::alloc(r.bytes as usize)?);
        views[r.id as usize] = Some(DevBuf::view(&block, 0, r.bytes as usize));
    }
    Ok(Regions { views })
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

// ----------------------------------------------------------------------------------------------------------- conversions

/// The conversion `x` names, on the bytes of the item (little endian, as the shard stores them). Pure: the executor runs it on a loader
/// thread, the tests run it on synthetic bytes.
pub fn convert(x: Xform, src: &[u8]) -> Result<Vec<u8>, String> {
    match x {
        Xform::None => Ok(src.to_vec()),
        Xform::Bf16ToF32 => {
            if src.len() % 2 != 0 {
                return Err(format!("a bf16 tensor of {} bytes", src.len()));
            }
            Ok(src.chunks_exact(2).flat_map(|c| ((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).to_le_bytes()).collect())
        }
        Xform::F32ToF16Exact => {
            if src.len() % 4 != 0 {
                return Err(format!("an fp32 tensor of {} bytes", src.len()));
            }
            let mut out = Vec::with_capacity(src.len() / 2);
            for (i, c) in src.chunks_exact(4).enumerate() {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let h = f64_to_f16(v as f64);
                if !v.is_finite() || (f16_to_f64(h) as f32).to_bits() != v.to_bits() {
                    return Err(format!("element {i} ({v:e}) is not exactly representable in fp16; refusing to round it"));
                }
                out.extend_from_slice(&h.to_le_bytes());
            }
            Ok(out)
        }
        Xform::CentredBias => {
            if src.len() % 4 != 0 {
                return Err(format!("an fp32 bias of {} bytes", src.len()));
            }
            let esb: Vec<f32> = src.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            Ok(moe::centred_bias_host(&esb).iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        Xform::PackKvB => {
            let rows = mla::N_HEADS * mla::KV_B_ROWS_PER_HEAD * mla::LATENT_DIM;
            if src.len() != rows * 2 {
                return Err(format!("kv_b_proj has {} bytes, expected {}", src.len(), rows * 2));
            }
            let bits: Vec<u16> = src.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let (uk, uvt) = mla::pack_kv_b(&bits, Dtype::F16, Dtype::F16);
            Ok(uk.iter().chain(uvt.iter()).flat_map(|v| v.to_le_bytes()).collect())
        }
        Xform::FoldPatchF16 => fold_patch_f16(src),
        Xform::PermuteDownsampleF16 => permute_downsample_f16(src),
    }
}

fn f16_words(src: &[u8], what: &str) -> Result<Vec<u16>, String> {
    if src.len() % 2 != 0 {
        return Err(format!("{what}: an fp16 tensor of {} bytes", src.len()));
    }
    let v: Vec<u16> = src.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    // a stored fp16 value that is not finite is a broken tower, refused as the bf16 tower's conversion refuses an overflow
    if let Some(i) = v.iter().position(|&h| !f16_to_f64(h).is_finite()) {
        return Err(format!("{what}: element {i} is not finite"));
    }
    Ok(v)
}

/// [`Xform::FoldPatchF16`]: `[O, 3, 2, 14, 14]` fp16 -> `[O, 592]` fp16 at the front of a slot of the stored size (the rest zero), the two temporal
/// slots summed exactly (two fp16 values add exactly in f64) and rounded once to fp16 (round to nearest even).
fn fold_patch_f16(src: &[u8]) -> Result<Vec<u8>, String> {
    use super::vision::{PATCH_K, PATCH_KP};
    let w = f16_words(src, "patch embedding")?;
    let row = 2 * PATCH_K;
    if w.is_empty() || w.len() % row != 0 {
        return Err(format!("the patch embedding has {} values, not a multiple of {row}", w.len()));
    }
    let o = w.len() / row;
    let mut out = vec![0u16; w.len()];
    let px = PATCH_K / 3;
    for r in 0..o {
        for k in 0..PATCH_K {
            let (c, i) = (k / px, k % px);
            let base = r * row + c * 2 * px + i;
            let sum = f16_to_f64(w[base]) + f16_to_f64(w[base + px]);
            out[r * PATCH_KP + k] = f64_to_f16(sum);
        }
    }
    if let Some(i) = out.iter().position(|&h| !f16_to_f64(h).is_finite()) {
        return Err(format!("the folded patch embedding overflows fp16 at {i}"));
    }
    Ok(out.iter().flat_map(|v| v.to_le_bytes()).collect())
}

/// [`Xform::PermuteDownsampleF16`]: `[O, C, 2, 2]` fp16 -> `[O, 4C]` with column `(kh * 2 + kw) * C + c` (C the tower's hidden size).
fn permute_downsample_f16(src: &[u8]) -> Result<Vec<u8>, String> {
    let c = super::vision::HIDDEN;
    let w = f16_words(src, "downsample")?;
    if w.is_empty() || w.len() % (4 * c) != 0 {
        return Err(format!("the downsample weight has {} values, not a multiple of {}", w.len(), 4 * c));
    }
    let mut out = vec![0u16; w.len()];
    for (src_row, dst_row) in w.chunks_exact(4 * c).zip(out.chunks_exact_mut(4 * c)) {
        for ch in 0..c {
            for j in 0..4 {
                dst_row[j * c + ch] = src_row[ch * 4 + j];
            }
        }
    }
    Ok(out.iter().flat_map(|v| v.to_le_bytes()).collect())
}

fn read_range(path: &Path, off: u64, len: usize) -> io::Result<Vec<u8>> {
    let _t = loadstats::timer(Stage::Read, len as u64);
    let f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; len];
    f.read_exact_at(&mut buf, off)?;
    Ok(buf)
}

// -------------------------------------------------------------------------------------------------------------- the load

/// Execute `plan` into `regions` through the pipelined loader: every item at its offset, every conversion, every marker. Returns when
/// every copy has completed on the device. `dir` is the pack's directory (the plan names shards relative to it).
pub fn run(dir: &Path, plan: &LoadPlan, regions: &Regions, o: &Options) -> Result<Report, ModelError> {
    run_as(dir, plan, regions, o, Component::Target)
}

/// [`run`], charged to `component` in the boot statistics (the vision tower's plan: `Component::Vision`).
pub fn run_as(dir: &Path, plan: &LoadPlan, regions: &Regions, o: &Options, component: Component) -> Result<Report, ModelError> {
    let mut loader = Loader::new();
    let files: Vec<usize> = plan.files.iter().map(|f| loader.file(&dir.join(f))).collect();
    let _scope = loadstats::scoped_component(component);
    for it in &plan.items {
        let dst = regions.view(it.region, it.dst, it.dst_bytes)?;
        let fid = files[it.file as usize];
        match (it.xform, it.pitch) {
            (Xform::None, Some(p)) => {
                // a column cut of a trellis (a width share's gate/up): the rows of its source span, densely (`validate` checked the shape)
                loader.copy_file_strided(&dst, fid, it.offset, it.bytes as usize, p.width as usize, p.src_pitch as usize)?;
            }
            (Xform::None, None) => {
                // the scales are copied as stored: fp16 -> f64 -> fp16 is the identity on every value a pack holds
                loader.copy_file(&dst, fid, it.offset, it.bytes as usize, CopyXform::None)?;
            }
            (x, _) => {
                let path = dir.join(&plan.files[it.file as usize]);
                let (off, len, name) = (it.offset, it.bytes as usize, it.name.clone());
                loader.copy_compute(&dst, it.dst_bytes as usize, move || {
                    let src = read_range(&path, off, len)?;
                    let _t = loadstats::timer(Stage::Convert, len as u64);
                    convert(x, &src).map_err(|e| io::Error::other(format!("{name}: {e}")))
                })?;
            }
        }
    }
    for m in &plan.markers {
        loader.check_file(files[m.file as usize], m.offset, &m.want.to_le_bytes(), format!("codebook marker of `{}`", m.name))?;
    }
    Ok(loader.run(o)?)
}

/// A pack on the device: what mapping, allocating and loading it produced, and how long each took.
pub struct Booted {
    pub pack: super::GlmPack,
    /// The plan that was executed (the pack's, restricted to the layers loaded and without the vision tower and the MTP layer).
    pub plan: LoadPlan,
    pub device: super::device::GlmDevice,
    pub report: Report,
    /// Seconds: scan + map + plan, region allocation, the pipelined load (reads, copies, conversions, marker checks).
    pub scan_s: f64,
    pub alloc_s: f64,
    pub load_s: f64,
}

/// Map the pack in `dir`, allocate the regions of its first `layers` layers (all 45 with `None`) and load them: the text model's weights
/// on the device, bound to their kernels' types. The vision tower and the MTP layer are not loaded (neither runs yet).
pub fn boot(dir: &Path, layers: Option<usize>, o: &LoadOptions) -> Result<Booted, ModelError> {
    boot_with(dir, layers, false, o)
}

/// [`boot`], and with `with_mtp` the NextN layer too (a model of `layers` trunk layers plus the drafter: the tests load one trunk layer and
/// the NextN layer to check the drafter against the oracle without the 87 GiB).
pub fn boot_with(dir: &Path, layers: Option<usize>, with_mtp: bool, o: &LoadOptions) -> Result<Booted, ModelError> {
    gb_cuda::rt::init_context()?;
    let t = std::time::Instant::now();
    let pack = super::GlmFamily::map_pack(dir)?;
    boot_mapped_with(dir, pack, layers, with_mtp, o, t.elapsed().as_secs_f64())
}

/// [`boot`] for a pack the caller has already mapped (`scan_s`: the seconds that took).
pub fn boot_mapped(
    dir: &Path,
    pack: super::GlmPack,
    layers: Option<usize>,
    o: &LoadOptions,
    scan_s: f64,
) -> Result<Booted, ModelError> {
    boot_mapped_with(dir, pack, layers, false, o, scan_s)
}

/// [`boot_mapped`] that also loads the NextN (MTP) layer when `with_mtp` (its trunk and expert regions: 3.5 GiB, every group 4-bit) and
/// binds it as `device.mtp`.
pub fn boot_mapped_with(
    dir: &Path,
    pack: super::GlmPack,
    layers: Option<usize>,
    with_mtp: bool,
    o: &LoadOptions,
    scan_s: f64,
) -> Result<Booted, ModelError> {
    use std::time::Instant;
    let nl = layers.unwrap_or(pack.config.layers.len());
    if with_mtp && !pack.config.has_mtp() {
        return Err(ModelError::Config("the NextN (MTP) layer was asked for but the pack has none".into()));
    }
    let mtp_layer = with_mtp.then(|| pack.config.mtp_layer_index(0));
    let plan = pack.plan.select_regions(keep_prefix_mtp(nl, mtp_layer));
    let t = Instant::now();
    let regions = allocate(&plan, o.floor_bytes)?;
    let alloc_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let report = run(dir, &plan, &regions, &o.loader)?;
    let load_s = t.elapsed().as_secs_f64();
    let device = super::device::bind_with(&pack.weights, &pack.config, regions, nl, with_mtp)?;
    Ok(Booted { pack, plan, device, report, scan_s, alloc_s, load_s })
}

/// The kinds of region a prefix of the model needs: the global one, and the trunk and expert regions of layers below `layers`.
pub fn keep_prefix(layers: usize) -> impl Fn(&RegionKind) -> bool {
    keep_prefix_mtp(layers, None)
}

/// [`keep_prefix`] and, with `mtp_layer` (the NextN layer's checkpoint index), that layer's trunk and expert regions as well.
pub fn keep_prefix_mtp(layers: usize, mtp_layer: Option<usize>) -> impl Fn(&RegionKind) -> bool {
    move |k| match k {
        RegionKind::Global => true,
        RegionKind::Trunk(l) | RegionKind::Experts(l) => *l < layers || Some(*l) == mtp_layer,
        RegionKind::Vision => false,
    }
}

impl LoadPlan {
    /// The plan restricted to the regions `keep` accepts: their items, the markers of their groups, and the regions themselves (ids
    /// and offsets unchanged, so the typed tree's destinations stay valid). `select` filters by category, this by region: a prefix of
    /// the layers for a partial-depth model.
    pub fn select_regions(&self, keep: impl Fn(&RegionKind) -> bool) -> LoadPlan {
        let kept: std::collections::BTreeSet<u32> = self.regions.iter().filter(|r| keep(&r.kind)).map(|r| r.id).collect();
        let items: Vec<LoadItem> = self.items.iter().filter(|i| kept.contains(&i.region)).cloned().collect();
        // a marker is named `<prefix>.<codebook>` and its group's trellis item `<prefix>.trellis`
        let prefixes: std::collections::BTreeSet<&str> =
            items.iter().filter(|i| i.part == ItemPart::Trellis).filter_map(|i| i.name.strip_suffix(".trellis")).collect();
        let markers =
            self.markers.iter().filter(|m| m.name.rsplit_once('.').is_some_and(|(p, _)| prefixes.contains(p))).cloned().collect();
        let mut skipped = gb_exl3::budget::Bytes::default();
        for m in &self.markers {
            if m.name.rsplit_once('.').is_some_and(|(p, _)| prefixes.contains(p)) {
                skipped.add(m.category, super::plan::MARKER_BYTES);
            }
        }
        LoadPlan {
            files: self.files.clone(),
            regions: self.regions.iter().filter(|r| kept.contains(&r.id)).cloned().collect(),
            items,
            skipped,
            markers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32b(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn gammas_widen_exactly() {
        let bf: Vec<u8> = [0x3F80u16, 0xC000, 0x0000, 0x7F80, 0x8000].iter().flat_map(|b| b.to_le_bytes()).collect();
        let out = convert(Xform::Bf16ToF32, &bf).unwrap();
        assert_eq!(out, f32b(&[1.0, -2.0, 0.0, f32::INFINITY, -0.0]));
        assert!(convert(Xform::Bf16ToF32, &[1, 2, 3]).is_err());
    }

    #[test]
    fn an_mhc_fn_narrows_only_when_every_element_is_exact() {
        // 0.5, -3.0 and 2^-24 (a fp16 subnormal) are exact in fp16; 0.1f32 is not
        let ok = convert(Xform::F32ToF16Exact, &f32b(&[0.5, -3.0, 5.9604645e-8, 0.0])).unwrap();
        let want: Vec<u8> =
            [f64_to_f16(0.5), f64_to_f16(-3.0), f64_to_f16(5.9604645e-8), 0].iter().flat_map(|h| h.to_le_bytes()).collect();
        assert_eq!(ok, want);
        let e = convert(Xform::F32ToF16Exact, &f32b(&[0.5, 0.1])).unwrap_err();
        assert!(e.contains("element 1") && e.contains("not exactly representable"), "{e}");
        assert!(convert(Xform::F32ToF16Exact, &f32b(&[f32::NAN])).is_err());
        assert!(convert(Xform::F32ToF16Exact, &f32b(&[1.0e6])).is_err(), "out of range for fp16");
    }

    #[test]
    fn the_router_bias_is_centred_and_fp16_exact() {
        let esb: Vec<f32> = (0..288).map(|i| 34.0 + ((i * 7919) % 101) as f32 * 0.01).collect();
        let out = convert(Xform::CentredBias, &f32b(&esb)).unwrap();
        assert_eq!(out.len(), 288 * 4);
        let got: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(got, moe::centred_bias_host(&esb));
        assert!(got.iter().all(|&v| (f16_to_f64(f64_to_f16(v as f64)) as f32).to_bits() == v.to_bits()));
    }

    #[test]
    fn kv_b_is_packed_uk_then_uvt() {
        let n = mla::N_HEADS * mla::KV_B_ROWS_PER_HEAD * mla::LATENT_DIM;
        let rows: Vec<u16> = (0..n).map(|i| (((i / mla::LATENT_DIM) * 7 + (i % mla::LATENT_DIM) * 13) & 0x3fff) as u16).collect();
        let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = convert(Xform::PackKvB, &bytes).unwrap();
        assert_eq!(out.len(), bytes.len());
        let (uk, uvt) = mla::pack_kv_b(&rows, Dtype::F16, Dtype::F16);
        let half = uk.len() * 2;
        assert_eq!(&out[..half], uk.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>().as_slice());
        assert_eq!(&out[half..], uvt.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>().as_slice());
        assert!(convert(Xform::PackKvB, &bytes[..100]).is_err());
    }

    #[test]
    fn a_prefix_keeps_the_global_region_and_the_layers_below_it() {
        let keep = keep_prefix(3);
        assert!(keep(&RegionKind::Global) && keep(&RegionKind::Trunk(2)) && keep(&RegionKind::Experts(2)));
        assert!(!keep(&RegionKind::Trunk(3)) && !keep(&RegionKind::Experts(40)) && !keep(&RegionKind::Vision));
        // the NextN layer is kept only when asked for, and then with both of its regions and nothing else beyond the prefix
        let keep = keep_prefix_mtp(3, Some(45));
        assert!(keep(&RegionKind::Trunk(45)) && keep(&RegionKind::Experts(45)) && keep(&RegionKind::Trunk(2)));
        assert!(!keep(&RegionKind::Trunk(44)) && !keep(&RegionKind::Experts(3)) && !keep(&RegionKind::Vision));
        assert!(!keep_prefix(45)(&RegionKind::Trunk(45)), "the whole text model does not include the NextN layer");
    }
}
