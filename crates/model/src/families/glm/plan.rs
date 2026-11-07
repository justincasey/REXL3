//! The load plan: what a generic pipeline executes to put a GLM pack on the device.
//!
//! A [`LoadPlan`] is plain data: the shard files, the device [`Region`]s to allocate, and an ordered list of [`LoadItem`]s, each one
//! `(tensor, part) -> (shard, offset, bytes) -> (region, offset)`. Nothing in it names a family or a kernel, and building it touches
//! no file and no GPU, so the pipeline that executes it (pinned staging, reader threads, the allocation policy) is free to change
//! without the plan changing, and the same pipeline can run a Qwen plan.
//!
//! Order: items are sorted by (shard, source offset), the order that reads each shard front to back; [`LoadPlan::extents`] merges
//! neighbours into the long sequential reads the pack's layout allows (a group's parts and a whole expert's three projections are
//! contiguous on disk, with only the 4-byte codebook markers, which are not uploaded, between them).
//!
//! Regions: one for the embedding, final norm and `lm_head` (`Global`); per layer one for everything but the routed experts
//! (`Trunk`) and, for a MoE layer, one for the routed experts (`Experts`, about 1.7 GiB at 2 bits, 2.5 GiB at 3, 3.4 GiB at 4);
//! one for the vision tower. A pipeline allocates a region with one `cudaMalloc` and fills it at the items' offsets. Why per-layer
//! regions and not one arena or one allocation a group: see `docs/GLM_LOAD.md` (measured).

use super::weights::{GlmWeights, Leaf, RegionId, Xform, ALIGN};
use gb_exl3::budget::{Bytes, Category};
use std::collections::BTreeMap;

/// What a region holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    /// The embedding, the final norm and `lm_head`.
    Global,
    /// Every tensor of a layer but its routed experts (for the MTP layer, index 45, its `eh_proj`, norms and shared head too).
    Trunk(usize),
    /// A MoE layer's routed experts, stacked as the grouped expert kernel reads them.
    Experts(usize),
    /// The vision tower.
    Vision,
}

impl std::fmt::Display for RegionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegionKind::Global => write!(f, "global"),
            RegionKind::Trunk(l) => write!(f, "layer{l}"),
            RegionKind::Experts(l) => write!(f, "experts{l}"),
            RegionKind::Vision => write!(f, "vision"),
        }
    }
}

/// One device allocation of the plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub id: RegionId,
    pub kind: RegionKind,
    /// Bytes to allocate: the items placed in it plus alignment padding, a multiple of [`ALIGN`].
    pub bytes: u64,
}

/// Which tensor of a group (or a whole plain tensor) an item carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemPart {
    /// A plain tensor, verbatim.
    Tensor,
    /// An EXL3 group's trellis (int16).
    Trellis,
    /// An EXL3 group's input scales `suh` (fp16 `[K]`).
    Suh,
    /// An EXL3 group's output scales `svh` (fp16 `[N]`).
    Svh,
}

/// One copy: `bytes` at `offset` of shard `file`, through `xform`, to `dst` bytes into `region` (where it takes `dst_bytes`).
#[derive(Clone, PartialEq, Eq)]
pub struct LoadItem {
    /// The checkpoint tensor name (`<prefix>.trellis` for a group's parts).
    pub name: String,
    pub part: ItemPart,
    pub category: Category,
    pub file: u16,
    pub offset: u64,
    /// Bytes read from the shard (of a strided item: its whole source span, gaps included, so sorting, overlap and extents see the
    /// range the read covers).
    pub bytes: u64,
    /// What the executor does to them (`Xform::None` for every EXL3 part and nearly every plain tensor).
    pub xform: Xform,
    /// Bytes the item occupies in its region (`bytes`, unless `xform` changes the size or the item is strided).
    pub dst_bytes: u64,
    pub region: RegionId,
    pub dst: u64,
    /// `Some` for a strided item: the source span is rows of `width` bytes `src_pitch` apart (`bytes = (rows - 1) * src_pitch +
    /// width`), copied densely (`dst_bytes = rows * width`), unchanged (`xform` is `None`). One such item carries a one-range
    /// [`super::weights::StripCut`] of a trellis (a width share's gate/up columns: every k-tile row's kept tiles) where per-row items
    /// would be `k / 16` of them.
    pub pitch: Option<Pitch>,
}

/// The row shape of a strided [`LoadItem`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pitch {
    pub width: u64,
    pub src_pitch: u64,
}

impl LoadItem {
    /// Bytes the item copies (its rows, for a strided item; else the bytes it reads).
    pub fn copied_bytes(&self) -> u64 {
        if self.pitch.is_some() {
            self.dst_bytes
        } else {
            self.bytes
        }
    }
}

/// The derived form, with `pitch` shown only when it is `Some`: a plan without strided items prints (and so digests, `layout_tests`)
/// exactly as before strided items existed.
impl std::fmt::Debug for LoadItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("LoadItem");
        d.field("name", &self.name)
            .field("part", &self.part)
            .field("category", &self.category)
            .field("file", &self.file)
            .field("offset", &self.offset)
            .field("bytes", &self.bytes)
            .field("xform", &self.xform)
            .field("dst_bytes", &self.dst_bytes)
            .field("region", &self.region)
            .field("dst", &self.dst);
        if let Some(p) = &self.pitch {
            d.field("pitch", p);
        }
        d.finish()
    }
}

/// The 4-byte codebook marker of an EXL3 group (`mul1`: `0x83DCD12D`, `mcg`: `0xCBAC1FED`, little-endian). It is not uploaded; it says which
/// codebook the group's trellis decodes with, and decoding with the wrong one is silently wrong, so a pipeline must check it. The
/// markers sit between a group's scales and its trellis, inside the long reads [`LoadPlan::extents`] makes, so the check costs
/// nothing there; reading all 37,420 of them one by one at scan time costs about 10 s on a cold page cache (measured).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MarkerCheck {
    pub name: String,
    pub category: Category,
    pub file: u16,
    pub offset: u64,
    pub want: u32,
}

/// Bytes of a marker.
pub const MARKER_BYTES: u64 = 4;

/// A read a pipeline can issue as one: a contiguous (or nearly: see `max_gap`) run of items of one shard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadExtent {
    pub file: u16,
    /// First byte to read and how many (covers the gaps between the items).
    pub offset: u64,
    pub bytes: u64,
    /// The items inside, as a range of `LoadPlan::items`.
    pub items: std::ops::Range<usize>,
}

/// Bytes by category, as `exl3-inspect` sums them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlanTotals {
    pub items: usize,
    pub regions: usize,
    /// Bytes the items copy, by category.
    pub uploaded: Bytes,
    /// On-disk bytes the plan does not copy (the codebook markers), by category.
    pub skipped: Bytes,
    /// Device bytes the regions take, padding included.
    pub region_bytes: u64,
}

impl PlanTotals {
    pub fn uploaded_total(&self) -> u64 {
        let b = &self.uploaded;
        b.routed + b.decoder + b.head + b.other + b.mtp + b.embedding + b.vision
    }
    pub fn skipped_total(&self) -> u64 {
        let b = &self.skipped;
        b.routed + b.decoder + b.head + b.other + b.mtp + b.embedding + b.vision
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoadPlan {
    pub files: Vec<String>,
    pub regions: Vec<Region>,
    /// Sorted by (file, offset).
    pub items: Vec<LoadItem>,
    /// Markers and other on-disk bytes the plan leaves out, by category.
    pub skipped: Bytes,
    /// The codebook markers to verify, sorted by (file, offset): the executor of the plan owes this check.
    pub markers: Vec<MarkerCheck>,
}

impl GlmWeights {
    /// The plan that loads this tree: one item per tensor and per uploaded part of every group, in (shard, offset) order, after
    /// checking it is internally consistent ([`LoadPlan::validate`]).
    pub fn load_plan(&self) -> Result<LoadPlan, String> {
        let mut items = Vec::new();
        let mut markers = Vec::new();
        let mut skipped = Bytes::default();
        let mut bad: Vec<String> = Vec::new();
        self.visit(&mut |leaf| match leaf {
            Leaf::Plain(t) => items.push(LoadItem {
                name: t.name.clone(),
                part: ItemPart::Tensor,
                category: t.category,
                file: t.src.file,
                offset: t.src.offset,
                bytes: t.bytes,
                xform: t.xform,
                dst_bytes: t.dst_bytes,
                region: t.dst.region,
                dst: t.dst.offset,
                pitch: None,
            }),
            Leaf::Exl3(g) if g.cut.as_ref().is_some_and(|c| c.ranges.len() == 1) => {
                // one range of strips (a width share's gate/up columns): the trellis as ONE strided item (every k-tile row's kept tiles,
                // dense on the device as `select_strips_planned` lays them out), `suh` whole, `svh` the range's columns
                let (lo, hi) = g.cut.as_ref().unwrap().ranges[0];
                let bits = g.bits.twice() as u64 / 2;
                let strip = 256 * bits; // one 128-column strip of one k-tile row
                let (rows, width, src_pitch) = ((g.k / 16) as u64, (hi - lo) as u64 * strip, (g.n / 128) as u64 * strip);
                if gb_cuda::exl3_split::cut_strips(g.n, &[(lo, hi)]).is_none() || g.k % 16 != 0 || g.trellis.bytes != rows * width {
                    bad.push(format!("`{}`: strips [{lo}, {hi}) are not a cut of {} x {}", g.prefix, g.k, g.n));
                    return;
                }
                let item = |part, suffix: &str, p: &super::weights::Exl3Part, src: u64, bytes: u64, dst_bytes: u64, pitch| LoadItem {
                    name: format!("{}.{suffix}", g.prefix),
                    part,
                    category: g.category,
                    file: p.src.file,
                    offset: p.src.offset + src,
                    bytes,
                    xform: Xform::None,
                    dst_bytes,
                    region: p.dst.region,
                    dst: p.dst.offset,
                    pitch,
                };
                let (span, pitch) = ((rows - 1) * src_pitch + width, Some(Pitch { width, src_pitch }));
                items.push(item(ItemPart::Trellis, "trellis", &g.trellis, lo as u64 * strip, span, rows * width, pitch));
                items.push(item(ItemPart::Suh, "suh", &g.suh, 0, g.suh.bytes, g.suh.bytes, None));
                items.push(item(ItemPart::Svh, "svh", &g.svh, lo as u64 * 256, g.svh.bytes, g.svh.bytes, None));
                if let Some((src, bytes)) = g.marker {
                    skipped.add(g.category, bytes);
                    markers.push(marker_of(g, src));
                }
            }
            Leaf::Exl3(g) if g.cut.is_some() => {
                // a rank's strips of the group (`StripCut`): the trellis as one run per k-tile row and range, `svh` one per range, `suh` whole,
                // laid out as `select_strips_planned` lays out its device copy (the codebook marker is checked as for any group)
                let ranges = &g.cut.as_ref().unwrap().ranges;
                let bits = g.bits.twice() as usize / 2;
                let (Some(tr), Some(sv)) = (
                    gb_cuda::exl3_split::cut_trellis_runs(g.k, g.n, bits, ranges),
                    gb_cuda::exl3_split::cut_vector_runs(g.n, ranges),
                ) else {
                    bad.push(format!("`{}`: strips {ranges:?} are not a cut of {} x {}", g.prefix, g.k, g.n));
                    return;
                };
                let run = |part, suffix: &str, p: &super::weights::Exl3Part, (src, dst, bytes): (usize, usize, usize)| LoadItem {
                    name: format!("{}.{suffix}", g.prefix),
                    part,
                    category: g.category,
                    file: p.src.file,
                    offset: p.src.offset + src as u64,
                    bytes: bytes as u64,
                    xform: Xform::None,
                    dst_bytes: bytes as u64,
                    region: p.dst.region,
                    dst: p.dst.offset + dst as u64,
                    pitch: None,
                };
                items.extend(tr.into_iter().map(|r| run(ItemPart::Trellis, "trellis", &g.trellis, r)));
                items.push(run(ItemPart::Suh, "suh", &g.suh, (0, 0, g.suh.bytes as usize)));
                items.extend(sv.into_iter().map(|r| run(ItemPart::Svh, "svh", &g.svh, r)));
                if let Some((src, bytes)) = g.marker {
                    skipped.add(g.category, bytes);
                    markers.push(marker_of(g, src));
                }
            }
            Leaf::Exl3(g) => {
                // a row cut (a width share's down rows, `Exl3Linear::rows`): the trellis k-tile rows and `suh` entries `[k0, k1)` are
                // contiguous in the stored parts, `svh` whole; a whole group reads every part from its first byte
                let (tr_src, suh_src) = match g.rows {
                    Some((k0, _)) => ((k0 / 16) as u64 * g.n as u64 * (g.bits.twice() as u64 / 2) * 2, k0 as u64 * 2),
                    None => (0, 0),
                };
                for (part, suffix, p, at) in [
                    (ItemPart::Trellis, "trellis", &g.trellis, tr_src),
                    (ItemPart::Suh, "suh", &g.suh, suh_src),
                    (ItemPart::Svh, "svh", &g.svh, 0),
                ] {
                    items.push(LoadItem {
                        name: format!("{}.{suffix}", g.prefix),
                        part,
                        category: g.category,
                        file: p.src.file,
                        offset: p.src.offset + at,
                        bytes: p.bytes,
                        xform: Xform::None,
                        dst_bytes: p.bytes,
                        region: p.dst.region,
                        dst: p.dst.offset,
                        pitch: None,
                    });
                }
                if let Some((src, bytes)) = g.marker {
                    skipped.add(g.category, bytes);
                    markers.push(marker_of(g, src));
                }
            }
        });
        if !bad.is_empty() {
            return Err(bad.join("; "));
        }
        items.sort_by(|a, b| (a.file, a.offset).cmp(&(b.file, b.offset)));
        markers.sort_by(|a, b| (a.file, a.offset).cmp(&(b.file, b.offset)));
        let plan = LoadPlan { files: self.files.clone(), regions: self.regions.clone(), items, skipped, markers };
        plan.validate()?;
        Ok(plan)
    }
}

/// The codebook marker check of a group whose marker is at `src`.
fn marker_of(g: &super::weights::Exl3Linear, src: super::weights::Src) -> MarkerCheck {
    MarkerCheck {
        name: format!("{}.{}", g.prefix, g.codebook.name()),
        category: g.category,
        file: src.file,
        offset: src.offset,
        want: match g.codebook {
            gb_exl3::Codebook::Mcg => gb_exl3::MARKER_MCG,
            _ => gb_exl3::MARKER_MUL1,
        },
    }
}

impl LoadPlan {
    /// Items of one category.
    pub fn items_in(&self, c: Category) -> impl Iterator<Item = &LoadItem> {
        self.items.iter().filter(move |i| i.category == c)
    }

    pub fn region(&self, id: RegionId) -> Option<&Region> {
        self.regions.iter().find(|r| r.id == id)
    }

    pub fn totals(&self) -> PlanTotals {
        let mut uploaded = Bytes::default();
        for i in &self.items {
            uploaded.add(i.category, i.copied_bytes());
        }
        PlanTotals {
            items: self.items.len(),
            regions: self.regions.len(),
            uploaded,
            skipped: self.skipped,
            region_bytes: self.regions.iter().map(|r| r.bytes).sum(),
        }
    }

    /// The plan restricted to the items whose category passes `keep` (drop the vision tower, say, or the MTP head), with the
    /// regions they still use. Region ids and offsets are unchanged, so the typed tree's destinations stay valid; a region that
    /// still holds items keeps its full size (an MTP head and a trunk never share one; the global region is the only mixed one).
    pub fn select(&self, keep: impl Fn(Category) -> bool) -> LoadPlan {
        let items: Vec<LoadItem> = self.items.iter().filter(|i| keep(i.category)).cloned().collect();
        let used: std::collections::BTreeSet<RegionId> = items.iter().map(|i| i.region).collect();
        let mut skipped = Bytes::default();
        for c in [
            Category::RoutedExpert,
            Category::Decoder,
            Category::Head,
            Category::Other,
            Category::Mtp,
            Category::Embedding,
            Category::Vision,
        ] {
            if keep(c) {
                skipped.add(c, bytes_of(&self.skipped, c));
            }
        }
        LoadPlan {
            files: self.files.clone(),
            regions: self.regions.iter().filter(|r| used.contains(&r.id)).cloned().collect(),
            items,
            skipped,
            markers: self.markers.iter().filter(|m| keep(m.category)).cloned().collect(),
        }
    }

    /// The markers that lie wholly inside an extent: the ones a pipeline can check from the bytes it has just read.
    pub fn markers_in(&self, e: &ReadExtent) -> &[MarkerCheck] {
        let key = |m: &MarkerCheck| (m.file, m.offset);
        let lo = self.markers.partition_point(|m| key(m) < (e.file, e.offset));
        let hi = self.markers.partition_point(|m| key(m) < (e.file, e.offset + e.bytes));
        let inside = &self.markers[lo..hi];
        // a marker that starts inside but runs past the end belongs to no extent
        match inside.last() {
            Some(m) if m.offset + MARKER_BYTES > e.offset + e.bytes => &inside[..inside.len() - 1],
            _ => inside,
        }
    }

    /// The markers no extent of `extents` covers (a split between a group's scales and its trellis leaves its marker in neither
    /// read): a pipeline checks these with a read of their own.
    pub fn uncovered_markers(&self, extents: &[ReadExtent]) -> Vec<&MarkerCheck> {
        let mut covered = std::collections::BTreeSet::new();
        for e in extents {
            for m in self.markers_in(e) {
                covered.insert((m.file, m.offset));
            }
        }
        self.markers.iter().filter(|m| !covered.contains(&(m.file, m.offset))).collect()
    }

    /// Merge neighbouring items of one shard into reads: an item joins the open extent when the gap between them is at most
    /// `max_gap` bytes and the extent would not exceed `max_bytes` (an item larger than that is an extent of its own). A codebook
    /// marker that follows an extent's last item within `max_gap` is read with it (an extent may exceed `max_bytes` by that much).
    pub fn extents(&self, max_gap: u64, max_bytes: u64) -> Vec<ReadExtent> {
        let mut out: Vec<ReadExtent> = Vec::new();
        for (idx, it) in self.items.iter().enumerate() {
            if let Some(e) = out.last_mut() {
                let end = e.offset + e.bytes;
                if e.file == it.file
                    && it.offset >= end
                    && it.offset - end <= max_gap
                    && it.offset + it.bytes - e.offset <= max_bytes
                {
                    e.bytes = it.offset + it.bytes - e.offset;
                    e.items.end = idx + 1;
                    continue;
                }
            }
            out.push(ReadExtent { file: it.file, offset: it.offset, bytes: it.bytes, items: idx..idx + 1 });
        }
        // a marker that follows an extent's last item closely joins that extent (so a split between a group's scales and its trellis
        // does not strand the marker): the extent may then exceed `max_bytes`, by at most `max_gap`
        for e in &mut out {
            let next_item = self.items.get(e.items.end).filter(|i| i.file == e.file).map_or(u64::MAX, |i| i.offset);
            let mut at = self.markers.partition_point(|m| (m.file, m.offset) < (e.file, e.offset + e.bytes));
            while let Some(m) = self.markers.get(at) {
                if m.file != e.file || m.offset - (e.offset + e.bytes) > max_gap || m.offset + MARKER_BYTES > next_item {
                    break;
                }
                e.bytes = m.offset + MARKER_BYTES - e.offset;
                at += 1;
            }
        }
        out
    }

    /// The plan's internal consistency: items sorted by source and not overlapping, every destination inside its region and
    /// aligned, no two destinations of a region overlapping, every region id known. A plan that fails is a mapper bug.
    pub fn validate(&self) -> Result<(), String> {
        let known: BTreeMap<RegionId, &Region> = self.regions.iter().map(|r| (r.id, r)).collect();
        if known.len() != self.regions.len() {
            return Err("two regions share an id".into());
        }
        for r in &self.regions {
            if r.bytes % ALIGN != 0 {
                return Err(format!("region {} ({}) has {} bytes, not a multiple of {ALIGN}", r.id, r.kind, r.bytes));
            }
        }
        for w in self.items.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            if (a.file, a.offset) > (b.file, b.offset) {
                return Err(format!("items are not sorted: {} before {}", a.name, b.name));
            }
            if a.file == b.file && a.offset + a.bytes > b.offset {
                return Err(format!("{} and {} overlap in shard {}", a.name, b.name, a.file));
            }
        }
        for w in self.markers.windows(2) {
            if (w[0].file, w[0].offset) >= (w[1].file, w[1].offset) {
                return Err(format!("markers are not sorted: {} before {}", w[0].name, w[1].name));
            }
        }
        for m in &self.markers {
            // a marker is on disk and not an item: no item may cover it
            let at = self.items.partition_point(|i| (i.file, i.offset + i.bytes) <= (m.file, m.offset));
            if self.items.get(at).is_some_and(|i| i.file == m.file && i.offset < m.offset + MARKER_BYTES) {
                return Err(format!("marker {} overlaps an item", m.name));
            }
        }
        let mut by_region: BTreeMap<RegionId, Vec<(u64, u64, &str)>> = BTreeMap::new();
        for i in &self.items {
            if usize::from(i.file) >= self.files.len() {
                return Err(format!("{} names shard {} of {}", i.name, i.file, self.files.len()));
            }
            let Some(r) = known.get(&i.region) else {
                return Err(format!("{} is placed in region {}, which the plan does not have", i.name, i.region));
            };
            if i.dst % ALIGN != 0 {
                return Err(format!("{} is placed at {} in {}, not aligned to {ALIGN}", i.name, i.dst, r.kind));
            }
            if let Some(p) = i.pitch {
                // a strided item: whole rows of `width` bytes, `src_pitch` apart, copied densely and unchanged
                let rows_ok = p.width > 0 && p.src_pitch >= p.width && i.bytes >= p.width && (i.bytes - p.width) % p.src_pitch == 0;
                if !rows_ok || i.xform != Xform::None || i.dst_bytes != ((i.bytes - p.width) / p.src_pitch + 1) * p.width {
                    return Err(format!(
                        "{} reads a {}-byte span in rows of {} bytes {} apart through {:?} and says it takes {} on the device, which \
                         is not its rows copied unchanged",
                        i.name, i.bytes, p.width, p.src_pitch, i.xform, i.dst_bytes
                    ));
                }
            } else if i.xform.dst_bytes(i.bytes) != i.dst_bytes {
                return Err(format!(
                    "{} reads {} bytes through {:?} and says it takes {} on the device, which that conversion cannot make",
                    i.name, i.bytes, i.xform, i.dst_bytes
                ));
            }
            if i.dst + i.dst_bytes > r.bytes {
                return Err(format!(
                    "{} ({} bytes at {}) runs past region {} ({} bytes)",
                    i.name, i.dst_bytes, i.dst, r.kind, r.bytes
                ));
            }
            by_region.entry(i.region).or_default().push((i.dst, i.dst_bytes, &i.name));
        }
        for (id, mut v) in by_region {
            v.sort();
            for w in v.windows(2) {
                if w[0].0 + w[0].1 > w[1].0 {
                    return Err(format!("{} and {} overlap in region {}", w[0].2, w[1].2, known[&id].kind));
                }
            }
        }
        Ok(())
    }
}

/// The bytes of one category in a `Bytes`.
pub fn bytes_of(b: &Bytes, c: Category) -> u64 {
    match c {
        Category::RoutedExpert => b.routed,
        Category::Decoder => b.decoder,
        Category::Head => b.head,
        Category::Other => b.other,
        Category::Mtp => b.mtp,
        Category::Embedding => b.embedding,
        Category::Vision => b.vision,
    }
}
