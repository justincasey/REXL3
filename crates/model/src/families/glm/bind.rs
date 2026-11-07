//! Mapping a pack's inventory onto the typed description: [`GlmWeights::map`].
//!
//! One pure function walks the config's layer recipe and claims, for every slot, the tensor it expects: the name from the layer's
//! position, the dtype and shape from the config's fields (hidden size, head counts and widths, ranks, expert counts), never from
//! literals. Claiming marks the tensor used; after the walk, a tensor nobody claimed is an error. A claim that finds nothing, finds
//! the wrong dtype or shape, or finds a group the kernels cannot take, is an error too. All problems are collected and reported
//! together, each named; an `Ok` tree is therefore complete and exact: every tensor of the pack is in it once, and it holds nothing
//! the pack lacks.
//!
//! The same walk places every leaf: it opens the regions (see `plan::RegionKind`) and gives each tensor its offset, 256-byte
//! aligned, in the order of the walk; the routed experts of a layer are laid out as a bank (`ExpertBank`).

use super::config::{GlmConfig, VisionTowerFormat};
use super::inventory::{GroupInfo, PackInventory, TensorInfo};
use super::plan::{Region, RegionKind};
use super::weights::*;
use crate::family::{Ffn, IndexerMode, LayerSpec, Mixer, Residual};
use gb_exl3::budget::Category;
use std::collections::BTreeSet;

const P: &str = "model.language_model.";

/// One reason a pack does not map onto the family's weights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Problem {
    /// A tensor (or EXL3 group) the family needs is not in the pack.
    Missing {
        name: String,
        group: bool,
    },
    /// A tensor (or EXL3 group) is in the pack and no slot of the family takes it.
    Unexpected {
        name: String,
        group: bool,
    },
    Dtype {
        name: String,
        want: &'static str,
        got: String,
    },
    Shape {
        name: String,
        want: Vec<usize>,
        got: Vec<usize>,
    },
    /// An EXL3 group whose (K, N) is not the slot's.
    GroupShape {
        prefix: String,
        want: (usize, usize),
        got: (usize, usize),
    },
    /// An EXL3 group in a form the loader does not take (fractional width, packed-sign scales, a bias).
    Unsupported {
        prefix: String,
        why: String,
    },
    /// A bank of routed experts whose members differ in width, codebook or size.
    NonUniform {
        layer: usize,
        what: String,
    },
    /// A group the scan refused.
    Refused {
        prefix: String,
        why: String,
    },
    /// Something the scan noticed that is not the benign duplicate-resolved-by-the-index case.
    HeaderIssue(String),
}

impl std::fmt::Display for Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = |g: &bool| if *g { "EXL3 group" } else { "tensor" };
        match self {
            Problem::Missing { name, group } => write!(f, "{} `{name}` is not in the pack", kind(group)),
            Problem::Unexpected { name, group } => {
                write!(f, "{} `{name}` is in the pack and no slot of this family takes it", kind(group))
            }
            Problem::Dtype { name, want, got } => write!(f, "tensor `{name}`: expected {want}, found {got}"),
            Problem::Shape { name, want, got } => write!(f, "tensor `{name}`: expected shape {want:?}, found {got:?}"),
            Problem::GroupShape { prefix, want, got } => {
                write!(f, "EXL3 group `{prefix}`: expected K={} N={}, found K={} N={}", want.0, want.1, got.0, got.1)
            }
            Problem::Unsupported { prefix, why } => write!(f, "EXL3 group `{prefix}` is not loadable: {why}"),
            Problem::NonUniform { layer, what } => write!(f, "layer {layer}'s routed experts are not uniform: {what}"),
            Problem::Refused { prefix, why } => write!(f, "EXL3 group `{prefix}` was refused by the pack scan: {why}"),
            Problem::HeaderIssue(m) => write!(f, "pack header issue: {m}"),
        }
    }
}

/// Every problem of a failed mapping, in the order found (missing and mismatched tensors by layer, then the unexpected ones by name).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayoutError {
    pub problems: Vec<Problem>,
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const SHOWN: usize = 12;
        write!(f, "the pack does not match the GLM-5.3 weight layout ({} problem(s)):", self.problems.len())?;
        for p in self.problems.iter().take(SHOWN) {
            write!(f, "\n  - {p}")?;
        }
        if self.problems.len() > SHOWN {
            write!(f, "\n  - and {} more", self.problems.len() - SHOWN)?;
        }
        Ok(())
    }
}

impl std::error::Error for LayoutError {}

/// The scan's note that a tensor stored in two shards resolves to the copy the index names (the MTP layer's `eh_proj`, `enorm` and
/// `hnorm` are in this pack): resolved, so not a problem.
fn is_benign_issue(issue: &str) -> bool {
    issue.contains(" is also stored in ") && issue.ends_with(" as the index says")
}

struct Binder<'a> {
    inv: &'a PackInventory,
    cfg: &'a GlmConfig,
    problems: Vec<Problem>,
    used_plain: BTreeSet<&'a str>,
    used_groups: BTreeSet<&'a str>,
    regions: Vec<Region>,
    /// The routed experts this rank places and loads (every expert on a single node).
    own: gb_cuda::moe::ExpertOwnership,
    /// Claimed and not loaded.
    skipped: Vec<Skipped>,
}

fn src(t: &TensorInfo) -> Src {
    Src { file: t.file, offset: t.offset }
}

impl<'a> Binder<'a> {
    fn region(&mut self, kind: RegionKind) -> RegionId {
        let id = self.regions.len() as RegionId;
        self.regions.push(Region { id, kind, bytes: 0 });
        id
    }

    /// Reserve `bytes` in a region at the next aligned offset.
    fn place(&mut self, region: RegionId, bytes: u64) -> Dst {
        let r = &mut self.regions[region as usize];
        let offset = align_up(r.bytes);
        r.bytes = offset + bytes;
        Dst { region, offset }
    }

    fn plain(&mut self, name: &str, dtype: PlainDtype, shape: &[usize], region: RegionId, category: Category) -> PlainTensor {
        self.plain_x(name, dtype, shape, Xform::None, region, category)
    }

    /// A plain tensor the loader converts on the way (`xform`): the destination takes the converted size.
    fn plain_x(
        &mut self,
        name: &str,
        dtype: PlainDtype,
        shape: &[usize],
        xform: Xform,
        region: RegionId,
        category: Category,
    ) -> PlainTensor {
        let bytes = shape.iter().product::<usize>() as u64 * dtype.size();
        let mut out = PlainTensor {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
            bytes,
            xform,
            dst_bytes: xform.dst_bytes(bytes),
            src: Src { file: 0, offset: 0 },
            dst: Dst::NONE,
            category,
        };
        let Some((key, t)) = self.inv.plain.get_key_value(name) else {
            self.problems.push(Problem::Missing { name: name.to_string(), group: false });
            return out;
        };
        self.used_plain.insert(key.as_str());
        if t.dtype != dtype.name() {
            self.problems.push(Problem::Dtype { name: name.to_string(), want: dtype.name(), got: t.dtype.clone() });
        } else if t.shape != shape {
            self.problems.push(Problem::Shape { name: name.to_string(), want: shape.to_vec(), got: t.shape.clone() });
        } else {
            out.src = src(t);
            out.dst = self.place(region, out.dst_bytes);
        }
        out
    }

    /// Claim an EXL3 group of `k x n`. Parts are placed in `region` (trellis, `suh`, `svh`) unless it is `None` (a bank places
    /// its own).
    fn group(&mut self, prefix: &str, k: usize, n: usize, region: Option<RegionId>, category: Category) -> Exl3Linear {
        self.group_with(prefix, k, n, region, category, false)
    }

    /// [`Binder::group`], and with `allow_bias` a group that has a `.bias` is taken (the caller claims and places the bias: `group_bias`).
    /// The text model's kernels have no bias input, so only the vision tower allows one.
    fn group_with(
        &mut self,
        prefix: &str,
        k: usize,
        n: usize,
        region: Option<RegionId>,
        category: Category,
        allow_bias: bool,
    ) -> Exl3Linear {
        let none = Exl3Part { bytes: 0, src: Src { file: 0, offset: 0 }, dst: Dst::NONE };
        let mut out = Exl3Linear {
            prefix: prefix.to_string(),
            k,
            n,
            bits: gb_exl3::Bits::from_twice(2).expect("2 is a width"),
            codebook: gb_exl3::Codebook::Mul1,
            trellis: none,
            suh: none,
            svh: none,
            marker: None,
            category,
            cut: None,
            rows: None,
        };
        let Some((key, g)) = self.inv.groups.get_key_value(prefix) else {
            self.problems.push(Problem::Missing { name: prefix.to_string(), group: true });
            return out;
        };
        self.used_groups.insert(key.as_str());
        if (g.k, g.n) != (k, n) {
            self.problems.push(Problem::GroupShape { prefix: prefix.to_string(), want: (k, n), got: (g.k, g.n) });
            return out;
        }
        if let Some(why) = unsupported(g, allow_bias) {
            self.problems.push(Problem::Unsupported { prefix: prefix.to_string(), why });
            return out;
        }
        let part = |t: &TensorInfo| Exl3Part { bytes: t.bytes, src: src(t), dst: Dst::NONE };
        out.bits = g.bits;
        out.codebook = g.codebook;
        out.trellis = part(&g.trellis);
        out.suh = part(&g.in_scale);
        out.svh = part(&g.out_scale);
        out.marker = g.marker.as_ref().map(|m| (src(m), m.bytes));
        if let Some(r) = region {
            out.trellis.dst = self.place(r, out.trellis.bytes);
            out.suh.dst = self.place(r, out.suh.bytes);
            out.svh.dst = self.place(r, out.svh.bytes);
        }
        out
    }

    /// [`Binder::group`] that places only the 128-column strips `ranges` of the group in `region` (a [`StripCut`]: lane E, a TP rank's KDA
    /// `qkv_proj` heads): the group is claimed and checked whole, its trellis and `svh` placed at the kept strips' size, `suh` whole.
    fn group_cut(
        &mut self,
        prefix: &str,
        k: usize,
        n: usize,
        region: RegionId,
        category: Category,
        ranges: Vec<(usize, usize)>,
    ) -> Exl3Linear {
        let mut out = self.group(prefix, k, n, None, category);
        if out.trellis.bytes == 0 {
            return out; // the claim failed and said why
        }
        let bits = out.bits.twice() as usize / 2;
        let strip = (k / 16) as u64 * 256 * bits as u64; // one 128-column strip of every k-tile row
        let Some(kept) = gb_cuda::exl3_split::cut_strips(n, &ranges) else {
            self.problems.push(Problem::Unsupported {
                prefix: prefix.to_string(),
                why: format!("strips {ranges:?} are not a cut of {n} columns"),
            });
            return out;
        };
        if out.bits.twice() % 2 != 0 || out.trellis.bytes != (n / 128) as u64 * strip || out.svh.bytes != n as u64 * 2 {
            self.problems.push(Problem::Unsupported {
                prefix: prefix.to_string(),
                why: format!(
                    "a rank's strips need whole bits and the stored layout ({} trellis bytes for {n} columns)",
                    out.trellis.bytes
                ),
            });
            return out;
        }
        out.trellis.bytes = kept as u64 * strip;
        out.svh.bytes = kept as u64 * 128 * 2;
        out.cut = Some(StripCut { ranges });
        out.trellis.dst = self.place(region, out.trellis.bytes);
        out.suh.dst = self.place(region, out.suh.bytes);
        out.svh.dst = self.place(region, out.svh.bytes);
        out
    }

    /// Cut a claimed routed expert projection to the columns `[c0, c1)` of the expert width (whole 128-column blocks) for a width share
    /// ([`ExpertBank::width`]): `down` false is gate/up (`k` = hidden, `n` = the width: output columns, a one-range [`StripCut`], `svh`
    /// cut, `suh` whole), `down` true is down (`k` = the width, `n` = hidden: input rows, [`Exl3Linear::rows`], `suh` cut, `svh` whole).
    /// The placed byte sizes become the cut's (nothing is placed here: the bank lays its experts out); the stored layout is checked
    /// first, so a cut reads exactly the bytes the kernels expect.
    fn width_cut(&mut self, g: &mut Exl3Linear, down: bool, (c0, c1): (usize, usize)) {
        let bits = (g.bits.twice() / 2) as u64;
        let (k, n) = (g.k as u64, g.n as u64);
        let row = n * bits * 2; // one k-tile row: n / 16 tiles of 16 x bits int16
        let width = if down { k } else { n };
        if g.bits.twice() % 2 != 0 || g.trellis.bytes != k / 16 * row || g.suh.bytes != k * 2 || g.svh.bytes != n * 2 || c1 as u64 > width
        {
            self.problems.push(Problem::Unsupported {
                prefix: g.prefix.clone(),
                why: format!(
                    "a width share [{c0}, {c1}) needs whole bits and the stored layout ({} trellis bytes for {k} x {n})",
                    g.trellis.bytes
                ),
            });
            return;
        }
        let kept = (c1 - c0) as u64;
        if down {
            g.trellis.bytes = kept / 16 * row;
            g.suh.bytes = kept * 2;
            g.rows = Some((c0, c1));
        } else {
            g.trellis.bytes = k / 16 * kept * bits * 2;
            g.svh.bytes = kept * 2;
            g.cut = Some(StripCut { ranges: vec![(c0 / 128, c1 / 128)] });
        }
    }

    fn mhc_set(&mut self, l: &str, which: &str, streams: usize, region: RegionId, cat: Category) -> HcWeights {
        let (mix, h) = (streams * (streams + 2), streams * self.cfg.hidden);
        HcWeights {
            // fp32 in the shard, fp16 on the device: every `fn` of the pack is exactly representable (the executor checks each one)
            fn_: self.plain_x(&format!("{l}hc_{which}_fn"), PlainDtype::F32, &[mix, h], Xform::F32ToF16Exact, region, cat),
            base: self.plain(&format!("{l}hc_{which}_base"), PlainDtype::F32, &[mix], region, cat),
            scale: self.plain(&format!("{l}hc_{which}_scale"), PlainDtype::F32, &[3], region, cat),
        }
    }

    fn kda(&mut self, l: &str, region: RegionId, cat: Category) -> KdaWeights {
        let (h, k) = (self.cfg.hidden, &self.cfg.kda);
        let (heads, d, conv, rank) = (k.heads, k.head_dim, k.conv_kernel, k.rank);
        let channels = heads * d;
        let a = format!("{l}self_attn.");
        use PlainDtype::*;
        // a TP rank of the KDA head split (`KdaConfig::rank`) places only its heads' qkv strips, the rest of the layer whole
        let qkv_proj = match rank {
            None => self.group(&format!("{a}qkv_proj"), h, 3 * channels, Some(region), cat),
            Some((h0, h1)) => {
                self.group_cut(&format!("{a}qkv_proj"), h, 3 * channels, region, cat, kda_qkv_strips(heads, d, h0, h1))
            }
        };
        KdaWeights {
            qkv_proj,
            b_proj: self.plain(&format!("{a}b_proj.weight"), F16, &[heads, h], region, cat),
            // the low-rank gate projections have rank head_dim in this pack; the rank is not a config field
            f_a_proj: self.plain(&format!("{a}f_a_proj.weight"), F16, &[d, h], region, cat),
            f_b_proj: self.plain(&format!("{a}f_b_proj.weight"), F16, &[channels, d], region, cat),
            g_a_proj: self.plain(&format!("{a}g_a_proj.weight"), F16, &[d, h], region, cat),
            g_b_proj: self.plain(&format!("{a}g_b_proj.weight"), F16, &[channels, d], region, cat),
            a_log: self.plain(&format!("{a}A_log"), F32, &[heads], region, cat),
            dt_bias: self.plain(&format!("{a}dt_bias"), F32, &[channels], region, cat),
            conv1d_weight: self.plain(&format!("{a}conv1d.weight"), Bf16, &[3 * channels, 1, conv], region, cat),
            o_norm: self.plain(&format!("{a}o_norm.weight"), Bf16, &[d], region, cat),
            o_proj: self.group(&format!("{a}o_proj"), channels, h, Some(region), cat),
        }
    }

    fn mla(&mut self, l: &str, indexer: IndexerMode, region: RegionId, cat: Category) -> MlaWeights {
        let (h, m) = (self.cfg.hidden, &self.cfg.mla);
        let (heads, q_rank, kv_rank) = (m.heads, m.q_lora_rank, m.kv_lora_rank);
        let (qk, nope, rope, v) = (m.qk_head_dim(), m.qk_nope_head_dim, m.qk_rope_head_dim, m.v_head_dim);
        let a = format!("{l}self_attn.");
        use PlainDtype::*;
        let q_a_proj = self.group(&format!("{a}q_a_proj"), h, q_rank, Some(region), cat);
        let q_a_layernorm = self.plain_x(&format!("{a}q_a_layernorm.weight"), Bf16, &[q_rank], Xform::Bf16ToF32, region, cat);
        let q_b_proj = self.group(&format!("{a}q_b_proj"), q_rank, heads * qk, Some(region), cat);
        let kv_a_proj_with_mqa = self.group(&format!("{a}kv_a_proj_with_mqa"), h, kv_rank + rope, Some(region), cat);
        let kv_a_layernorm = self.plain_x(&format!("{a}kv_a_layernorm.weight"), Bf16, &[kv_rank], Xform::Bf16ToF32, region, cat);
        // plain fp16, not EXL3 and not bf16; the kernels read it split per head and packed (W_UK, transposed W_UV)
        let kv_b_proj =
            self.plain_x(&format!("{a}kv_b_proj.weight"), F16, &[heads * (nope + v), kv_rank], Xform::PackKvB, region, cat);
        let o_proj = self.group(&format!("{a}o_proj"), heads * v, h, Some(region), cat);
        let indexer = match indexer {
            IndexerMode::Full => Some(self.indexer(&a, region, cat)),
            IndexerMode::Shared => None,
        };
        MlaWeights { q_a_proj, q_a_layernorm, q_b_proj, kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, o_proj, indexer }
    }

    fn indexer(&mut self, a: &str, region: RegionId, cat: Category) -> IndexerWeights {
        let (h, ix, q_rank) = (self.cfg.hidden, &self.cfg.indexer, self.cfg.mla.q_lora_rank);
        let (heads, d, kpool) = (ix.heads, ix.head_dim, ix.kpool);
        let i = format!("{a}indexer.");
        use PlainDtype::*;
        // `wq_b` is an EXL3 group in most packs; a pack that stores it as plain fp16 has `wq_b.weight` `[heads x d, q_rank]` and no group
        // (claimed as such, never both: with neither, the group is reported missing)
        let wq_b =
            if !self.inv.groups.contains_key(&format!("{i}wq_b")) && self.inv.plain.contains_key(&format!("{i}wq_b.weight")) {
                IndexerWqB::PlainF16(self.plain(&format!("{i}wq_b.weight"), F16, &[heads * d, q_rank], region, cat))
            } else {
                IndexerWqB::Exl3(self.group(&format!("{i}wq_b"), q_rank, heads * d, Some(region), cat))
            };
        IndexerWeights {
            wq_b,
            wk: self.plain(&format!("{i}wk.weight"), F16, &[d, h], region, cat),
            weights_proj: self.plain(&format!("{i}weights_proj.weight"), F16, &[heads, h], region, cat),
            k_norm_weight: self.plain(&format!("{i}k_norm.weight"), F16, &[d], region, cat),
            k_norm_bias: self.plain(&format!("{i}k_norm.bias"), F16, &[d], region, cat),
            kpool: (kpool > 0).then(|| KpoolWeights {
                compress_ape: self.plain(&format!("{i}index_kpool_compress_ape"), F32, &[kpool, d], region, cat),
                compress_gate: self.plain(&format!("{i}index_kpool_compress_gate"), F16, &[d, h], region, cat),
            }),
        }
    }

    fn gated_mlp(&mut self, prefix: &str, inter: usize, region: RegionId, cat: Category) -> GatedMlp {
        let h = self.cfg.hidden;
        GatedMlp {
            gate_proj: self.group(&format!("{prefix}gate_proj"), h, inter, Some(region), cat),
            up_proj: self.group(&format!("{prefix}up_proj"), h, inter, Some(region), cat),
            down_proj: self.group(&format!("{prefix}down_proj"), inter, h, Some(region), cat),
        }
    }

    /// Claim the routed experts of one layer, group the OWNED ones into homogeneous subbanks (gate, up and down widths, codebook), and
    /// lay those out in a region of their own: per subbank (key order) today's block sequence, so a uniform layer on a single node is
    /// one subbank laid out exactly as the single bank. Every expert is claimed; a non-owned one keeps `Dst::NONE` and is not loaded.
    fn expert_bank(&mut self, l: &str, layer: usize, cat: Category) -> ExpertBank {
        use gb_cuda::moe::{subbank_partition, SubbankKey};
        let (h, mi, n) = (self.cfg.hidden, self.cfg.ffn.moe_intermediate, self.cfg.ffn.routed_experts);
        let region = self.region(RegionKind::Experts(layer));
        let before = self.problems.len();
        let mut experts = Vec::with_capacity(n);
        for e in 0..n {
            let p = format!("{l}mlp.experts.{e}.");
            experts.push(ExpertWeights {
                gate_proj: self.group(&format!("{p}gate_proj"), h, mi, None, cat),
                up_proj: self.group(&format!("{p}up_proj"), h, mi, None, cat),
                down_proj: self.group(&format!("{p}down_proj"), mi, h, None, cat),
            });
        }
        let (g0, d0) = (experts[0].gate_proj.clone(), experts[0].down_proj.clone());
        let unplaced = ExpertBank {
            layer,
            region,
            bits_gate_up: g0.bits,
            bits_down: d0.bits,
            codebook: g0.codebook,
            hidden: h,
            inter: mi,
            experts: vec![],
            layout: ExpertLayout { gate: ProjLayout::default(), up: ProjLayout::default(), down: ProjLayout::default() },
            subbanks: vec![],
            own: self.own.clone(),
            width: self.cfg.ffn.width,
        };
        if self.problems.len() != before {
            // some group is missing or malformed: those problems are the report, the subbanks of what is left are not meaningful
            return ExpertBank { experts, ..unplaced };
        }
        if self.own.experts != n {
            self.problems.push(Problem::NonUniform {
                layer,
                what: format!("the expert ownership is for {} experts, the layer has {n}", self.own.experts),
            });
            return ExpertBank { experts, ..unplaced };
        }
        // the key of every expert: its three projections share a codebook (whole widths: `unsupported` refused fractional ones)
        let mut keys = Vec::with_capacity(n);
        for (e, x) in experts.iter().enumerate() {
            if x.up_proj.codebook != x.gate_proj.codebook || x.down_proj.codebook != x.gate_proj.codebook {
                self.problems
                    .push(Problem::NonUniform { layer, what: format!("expert {e}'s projections use different codebooks") });
            }
            let b = |g: &Exl3Linear| g.bits.twice() / 2;
            keys.push(SubbankKey {
                bits_gate: b(&x.gate_proj),
                bits_up: b(&x.up_proj),
                bits_down: b(&x.down_proj),
                codebook: x.gate_proj.codebook,
            });
        }
        if self.problems.len() != before {
            return ExpertBank { experts, ..unplaced };
        }
        // a width share places every expert at its columns: the cut sizes are the placed sizes from here on (uniformity, layout)
        if let Some(share) = self.cfg.ffn.width {
            let Some(cols) = share.cols(mi).filter(|_| self.own.is_all()) else {
                self.problems.push(Problem::NonUniform {
                    layer,
                    what: format!(
                        "the expert width share {share:?} needs every expert owned ({} of {n}) and a width ({mi}) of equal 128-column shares",
                        self.own.local()
                    ),
                });
                return ExpertBank { experts, ..unplaced };
            };
            for x in experts.iter_mut() {
                self.width_cut(&mut x.gate_proj, false, cols);
                self.width_cut(&mut x.up_proj, false, cols);
                self.width_cut(&mut x.down_proj, true, cols);
            }
            if self.problems.len() != before {
                return ExpertBank { experts, ..unplaced };
            }
        }
        let parts = subbank_partition(&keys, &self.own).expect("the ownership is for this layer's expert count");
        // within a subbank every member has the first member's byte sizes (the strides are constants)
        for (_, o) in &parts {
            let f = &experts[o.owned()[0] as usize];
            for &g in o.owned() {
                let x = &experts[g as usize];
                for (proj, a, want) in [
                    ("gate_proj", &x.gate_proj, &f.gate_proj),
                    ("up_proj", &x.up_proj, &f.up_proj),
                    ("down_proj", &x.down_proj, &f.down_proj),
                ] {
                    if (a.trellis.bytes, a.suh.bytes, a.svh.bytes) != (want.trellis.bytes, want.suh.bytes, want.svh.bytes) {
                        self.problems.push(Problem::NonUniform {
                            layer,
                            what: format!(
                                "expert {g} {proj} has a different byte size from expert {}'s of the same width",
                                o.owned()[0]
                            ),
                        });
                    }
                }
            }
        }
        if self.problems.len() != before {
            return ExpertBank { experts, ..unplaced };
        }
        // per subbank: trellises of a projection back to back, then the stacked scales; each block aligned
        let mut cursor = 0u64;
        let mut subbanks = Vec::with_capacity(parts.len());
        for (key, o) in parts {
            let nb = o.local() as u64;
            let f = &experts[o.owned()[0] as usize];
            let (fg, fu, fd) = (f.gate_proj.clone(), f.up_proj.clone(), f.down_proj.clone());
            let mut block = |bytes: u64| {
                let off = align_up(cursor);
                cursor = off + bytes;
                off
            };
            let (tg, tu, td) = (block(nb * fg.trellis.bytes), block(nb * fu.trellis.bytes), block(nb * fd.trellis.bytes));
            let (suh_g, svh_g) = (block(nb * fg.suh.bytes), block(nb * fg.svh.bytes));
            let (suh_u, svh_u) = (block(nb * fu.suh.bytes), block(nb * fu.svh.bytes));
            let (suh_d, svh_d) = (block(nb * fd.suh.bytes), block(nb * fd.svh.bytes));
            let proj = |trellis, suh, svh, g: &Exl3Linear| ProjLayout {
                trellis,
                trellis_stride: g.trellis.bytes,
                suh,
                suh_stride: g.suh.bytes,
                svh,
                svh_stride: g.svh.bytes,
            };
            let layout = ExpertLayout {
                gate: proj(tg, suh_g, svh_g, &fg),
                up: proj(tu, suh_u, svh_u, &fu),
                down: proj(td, suh_d, svh_d, &fd),
            };
            for (local, &g) in o.owned().iter().enumerate() {
                let x = &mut experts[g as usize];
                for (g, lay) in [(&mut x.gate_proj, &layout.gate), (&mut x.up_proj, &layout.up), (&mut x.down_proj, &layout.down)]
                {
                    g.trellis.dst = Dst { region, offset: lay.trellis_at(local) };
                    g.suh.dst = Dst { region, offset: lay.suh_at(local) };
                    g.svh.dst = Dst { region, offset: lay.svh_at(local) };
                }
            }
            subbanks.push(ExpertSubbank { key, ids: o.owned().to_vec(), layout });
        }
        self.regions[region as usize].bytes = align_up(cursor);
        let first = subbanks.first().map(|s| (s.layout, s.key));
        let Some((layout, k)) = first else {
            // a rank that owns none of this layer's experts: an empty region
            return ExpertBank { experts, ..unplaced };
        };
        let bits = |b: u8| gb_exl3::Bits::from_twice(2 * b).expect("a whole width");
        ExpertBank {
            bits_gate_up: bits(k.bits_gate),
            bits_down: bits(k.bits_down),
            codebook: k.codebook,
            experts,
            layout,
            subbanks,
            ..unplaced
        }
    }

    /// One decoder layer: the trunk's or the MTP layer's body. `region` is the layer's trunk region (the caller may already have
    /// placed tensors in it), `trunk` the category of its non-expert tensors, `routed` the category of its experts.
    fn layer(
        &mut self,
        index: usize,
        spec: LayerSpec,
        region: RegionId,
        trunk: Category,
        routed: Category,
    ) -> Option<GlmLayerWeights> {
        let (h, ffn_cfg) = (self.cfg.hidden, &self.cfg.ffn);
        let (inter, shared_n, dense_inter) = (ffn_cfg.moe_intermediate, ffn_cfg.shared_experts, ffn_cfg.dense_intermediate);
        let l = format!("{P}layers.{index}.");
        let input_layernorm =
            self.plain_x(&format!("{l}input_layernorm.weight"), PlainDtype::Bf16, &[h], Xform::Bf16ToF32, region, trunk);
        let post_attention_layernorm =
            self.plain_x(&format!("{l}post_attention_layernorm.weight"), PlainDtype::Bf16, &[h], Xform::Bf16ToF32, region, trunk);
        let mhc = match spec.residual {
            Residual::Mhc { streams } => Some(MhcWeights {
                attn: self.mhc_set(&l, "attn", streams, region, trunk),
                ffn: self.mhc_set(&l, "ffn", streams, region, trunk),
            }),
            Residual::Plain => None,
        };
        let mixer = match spec.mixer {
            Mixer::Kda => MixerWeights::Kda(Box::new(self.kda(&l, region, trunk))),
            Mixer::MlaDsa { indexer } => MixerWeights::Mla(Box::new(self.mla(&l, indexer, region, trunk))),
            Mixer::Gqa | Mixer::Gdn | Mixer::Mamba2 | Mixer::GqaNope | Mixer::None => {
                self.problems.push(Problem::HeaderIssue(format!("layer {index}: the GLM family has no mixer {:?}", spec.mixer)));
                return None;
            }
        };
        let ffn = match spec.ffn {
            Ffn::Dense => FfnWeights::Dense(Box::new(self.gated_mlp(&format!("{l}mlp."), dense_inter, region, trunk))),
            Ffn::None | Ffn::LatentMoeRelu2 => {
                self.problems.push(Problem::HeaderIssue(format!("layer {index}: the GLM family has no FFN {:?}", spec.ffn)));
                return None;
            }
            Ffn::Moe => {
                let router =
                    self.plain(&format!("{l}mlp.gate.weight"), PlainDtype::F16, &[self.cfg.ffn.routed_experts, h], region, trunk);
                let bias = self.plain_x(
                    &format!("{l}mlp.gate.e_score_correction_bias"),
                    PlainDtype::F32,
                    &[self.cfg.ffn.routed_experts],
                    Xform::CentredBias,
                    region,
                    trunk,
                );
                let experts = self.expert_bank(&l, index, routed);
                let shared =
                    (shared_n > 0).then(|| self.gated_mlp(&format!("{l}mlp.shared_experts."), shared_n * inter, region, trunk));
                FfnWeights::Moe(Box::new(MoeWeights { router, e_score_correction_bias: bias, experts, shared }))
            }
        };
        Some(GlmLayerWeights { index, input_layernorm, post_attention_layernorm, mhc, mixer, ffn })
    }

    fn linear(&mut self, base: &str, w_shape: &[usize], bias: Option<usize>, region: RegionId) -> PlainLinear {
        self.linear_x(base, PlainDtype::Bf16, w_shape, Xform::None, bias, region)
    }

    /// A plain weight of `dtype` (through `xform`) and its optional bias of the same dtype.
    fn linear_x(
        &mut self,
        base: &str,
        dtype: PlainDtype,
        w_shape: &[usize],
        xform: Xform,
        bias: Option<usize>,
        region: RegionId,
    ) -> PlainLinear {
        PlainLinear {
            weight: self.plain_x(&format!("{base}.weight"), dtype, w_shape, xform, region, Category::Vision),
            bias: bias.map(|n| self.plain(&format!("{base}.bias"), dtype, &[n], region, Category::Vision)),
        }
    }

    /// An EXL3 vision group `k -> n` of width `bits` (the config's `vision_bits`: a group of another width is refused by name), its parts
    /// placed in `region`; with `bias`, the group's `.bias`, which must be fp16 `[n]` and is placed right after.
    fn vision_group(&mut self, prefix: &str, k: usize, n: usize, bits: u8, bias: bool, region: RegionId) -> VisionLinear {
        let group = self.vision_group_raw(prefix, k, n, bits, bias, region);
        let bias = if bias { self.group_bias(prefix, n, region) } else { None };
        VisionLinear::Exl3 { group, bias }
    }

    /// [`Binder::vision_group`] without claiming the bias (`allow_bias`: the group may have one, which the caller places).
    fn vision_group_raw(&mut self, prefix: &str, k: usize, n: usize, bits: u8, allow_bias: bool, region: RegionId) -> Exl3Linear {
        let group = self.group_with(prefix, k, n, Some(region), Category::Vision, allow_bias);
        if let Some(g) = self.inv.groups.get(prefix) {
            if (g.k, g.n) == (k, n) && g.bits.is_integer() && g.bits.twice() != 2 * bits {
                self.problems.push(Problem::Unsupported {
                    prefix: prefix.to_string(),
                    why: format!("{}-bit tiles; `quantization_config.vision_bits` is {bits}", g.bits.as_f64()),
                });
            }
        }
        group
    }

    /// The fp16 bias `[n]` of a claimed group, placed in `region` (hazard: the tower's kernels read `const half*`, so a bf16 or fp32 group bias
    /// is a named dtype problem here, never bytes reinterpreted on the device). `None` when the group is missing (already reported) or has no
    /// bias (reported here).
    fn group_bias(&mut self, prefix: &str, n: usize, region: RegionId) -> Option<PlainTensor> {
        let g = self.inv.groups.get(prefix)?;
        let name = format!("{prefix}.bias");
        let Some(t) = g.bias.clone() else {
            self.problems.push(Problem::Missing { name, group: false });
            return None;
        };
        let bytes = n as u64 * PlainDtype::F16.size();
        let mut out = PlainTensor {
            name: name.clone(),
            dtype: PlainDtype::F16,
            shape: vec![n],
            bytes,
            xform: Xform::None,
            dst_bytes: bytes,
            src: Src { file: 0, offset: 0 },
            dst: Dst::NONE,
            category: Category::Vision,
        };
        if t.dtype != PlainDtype::F16.name() {
            self.problems.push(Problem::Dtype { name, want: PlainDtype::F16.name(), got: t.dtype });
        } else if t.shape != [n] {
            self.problems.push(Problem::Shape { name, want: vec![n], got: t.shape });
        } else {
            out.src = src(&t);
            out.dst = self.place(region, bytes);
        }
        Some(out)
    }

    /// Claim a plain tensor the map does not load, when the pack has it: its dtype and shape are still checked (fail closed), it is placed
    /// nowhere and recorded in `skipped` with `why`.
    fn skip_plain(&mut self, name: &str, dtype: PlainDtype, shape: &[usize], why: &'static str) {
        let Some((key, t)) = self.inv.plain.get_key_value(name) else { return };
        self.used_plain.insert(key.as_str());
        if t.dtype != dtype.name() {
            self.problems.push(Problem::Dtype { name: name.to_string(), want: dtype.name(), got: t.dtype.clone() });
        } else if t.shape != shape {
            self.problems.push(Problem::Shape { name: name.to_string(), want: shape.to_vec(), got: t.shape.clone() });
        }
        self.skipped.push(Skipped { name: name.to_string(), group: false, why });
    }

    /// `--no-vision`: every `model.visual.*` plain tensor and EXL3 group is claimed and skipped, so the tower's format never decides whether
    /// the text model maps. Nothing else is claimed here: an unknown tensor outside the tower is still refused by name.
    fn skip_vision(&mut self) {
        const WHY: &str = "the vision tower is not loaded (--no-vision)";
        for name in self.inv.plain.keys().filter(|n| n.starts_with(VISION_PREFIX)) {
            self.used_plain.insert(name.as_str());
            self.skipped.push(Skipped { name: name.clone(), group: false, why: WHY });
        }
        for name in self.inv.groups.keys().filter(|n| n.starts_with(VISION_PREFIX)) {
            self.used_groups.insert(name.as_str());
            self.skipped.push(Skipped { name: name.clone(), group: true, why: WHY });
        }
    }

    fn vision(&mut self, v: &super::config::GlmVisionConfig) -> GlmVisionWeights {
        let region = self.region(RegionKind::Vision);
        let (h, i, o, p) = (v.hidden, v.intermediate, v.out_hidden, v.projection_intermediate);
        let hd = h / v.heads;
        let m = v.spatial_merge_size;
        let b = |n: usize| v.attention_bias.then_some(n);
        let vis = VISION_PREFIX;
        // the plain (non-gamma) tensors are bf16 in the bf16 tower and fp16 in the EXL3 tower; the fp16 patch and downsample weights are folded
        // and permuted on the host by the plan (the bf16 tower's, on the device after its conversion)
        let (dt, patch_x, ds_x) = match v.tower {
            VisionTowerFormat::Bf16 => (PlainDtype::Bf16, Xform::None, Xform::None),
            VisionTowerFormat::Exl3 { .. } => (PlainDtype::F16, Xform::FoldPatchF16, Xform::PermuteDownsampleF16),
        };
        let patch_embed = self.linear_x(
            &format!("{vis}patch_embed.proj"),
            dt,
            &[h, v.in_channels, v.temporal_patch_size, v.patch_size, v.patch_size],
            patch_x,
            Some(h),
            region,
        );
        let mut blocks = Vec::with_capacity(v.depth);
        for n in 0..v.depth {
            let bl = format!("{vis}blocks.{n}.");
            let norm = |s: &str, len: usize, this: &mut Self| {
                this.plain(&format!("{bl}{s}"), PlainDtype::Bf16, &[len], region, Category::Vision)
            };
            let (norm1, norm2) = (norm("norm1.weight", h, self), norm("norm2.weight", h, self));
            let (q_norm, k_norm) = (norm("attn.q_norm.weight", hd, self), norm("attn.k_norm.weight", hd, self));
            let block = match v.tower {
                VisionTowerFormat::Bf16 => VisionBlock {
                    norm1,
                    norm2,
                    qkv: VisionQkv::Fused(self.linear(&format!("{bl}attn.qkv"), &[3 * h, h], b(3 * h), region)),
                    proj: VisionLinear::Plain(self.linear(&format!("{bl}attn.proj"), &[h, h], b(h), region)),
                    q_norm,
                    k_norm,
                    gate_proj: VisionLinear::Plain(self.linear(&format!("{bl}mlp.gate_proj"), &[i, h], b(i), region)),
                    up_proj: VisionLinear::Plain(self.linear(&format!("{bl}mlp.up_proj"), &[i, h], b(i), region)),
                    down_proj: VisionLinear::Plain(self.linear(&format!("{bl}mlp.down_proj"), &[h, i], b(h), region)),
                },
                VisionTowerFormat::Exl3 { bits } => {
                    let ab = v.attention_bias;
                    // q, k and v first (trellis and scales), then their three biases back to back: the tower reads them as one [3h] vector
                    let groups: Vec<Exl3Linear> = ["q_proj", "k_proj", "v_proj"]
                        .iter()
                        .map(|s| self.vision_group_raw(&format!("{bl}attn.{s}"), h, h, bits, ab, region))
                        .collect();
                    let biases: Vec<Option<PlainTensor>> = ["q_proj", "k_proj", "v_proj"]
                        .iter()
                        .map(|s| if ab { self.group_bias(&format!("{bl}attn.{s}"), h, region) } else { None })
                        .collect();
                    let [q, k, vv]: [VisionLinear; 3] = groups
                        .into_iter()
                        .zip(biases)
                        .map(|(group, bias)| VisionLinear::Exl3 { group, bias })
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap_or_else(|_| unreachable!("three projections"));
                    // the pack also holds the original fused bf16 qkv; exllamav3 1.6.2 never reads it once the split groups exist
                    // (`Linear.load` tries `load_exl3(q_proj)` before the fused fp16 fallback), so it is claimed, checked and not loaded
                    const FUSED: &str =
                        "the EXL3 tower runs the split q/k/v groups; the fused bf16 qkv is not read (as in exllamav3 1.6.2)";
                    self.skip_plain(&format!("{bl}attn.qkv.weight"), PlainDtype::Bf16, &[3 * h, h], FUSED);
                    if ab {
                        self.skip_plain(&format!("{bl}attn.qkv.bias"), PlainDtype::Bf16, &[3 * h], FUSED);
                    }
                    VisionBlock {
                        norm1,
                        norm2,
                        qkv: VisionQkv::Split(Box::new([q, k, vv])),
                        proj: self.vision_group(&format!("{bl}attn.proj"), h, h, bits, ab, region),
                        q_norm,
                        k_norm,
                        gate_proj: self.vision_group(&format!("{bl}mlp.gate_proj"), h, i, bits, ab, region),
                        up_proj: self.vision_group(&format!("{bl}mlp.up_proj"), h, i, bits, ab, region),
                        down_proj: self.vision_group(&format!("{bl}mlp.down_proj"), i, h, bits, ab, region),
                    }
                }
            };
            blocks.push(block);
        }
        let post_layernorm = self.plain(&format!("{vis}post_layernorm.weight"), PlainDtype::Bf16, &[h], region, Category::Vision);
        let downsample = self.linear_x(&format!("{vis}downsample"), dt, &[o, h, m, m], ds_x, Some(o), region);
        let mg = format!("{vis}merger.");
        let lin = |s: &str, k: usize, n: usize, this: &mut Self| match v.tower {
            VisionTowerFormat::Bf16 => VisionLinear::Plain(PlainLinear {
                weight: this.plain(&format!("{mg}{s}.weight"), PlainDtype::Bf16, &[n, k], region, Category::Vision),
                bias: None,
            }),
            VisionTowerFormat::Exl3 { bits } => this.vision_group(&format!("{mg}{s}"), k, n, bits, false, region),
        };
        let proj = lin("proj", o, o, self);
        let gate_proj = lin("gate_proj", o, p, self);
        let up_proj = lin("up_proj", o, p, self);
        let down_proj = lin("down_proj", p, o, self);
        let merger = VisionMerger {
            proj,
            gate_proj,
            up_proj,
            down_proj,
            post_projection_norm_weight: self.plain(
                &format!("{mg}post_projection_norm.weight"),
                dt,
                &[o],
                region,
                Category::Vision,
            ),
            post_projection_norm_bias: self.plain(&format!("{mg}post_projection_norm.bias"), dt, &[o], region, Category::Vision),
        };
        GlmVisionWeights { format: v.tower, patch_embed, blocks, post_layernorm, downsample, merger }
    }
}

/// Every tensor of the vision tower is named under this prefix.
pub const VISION_PREFIX: &str = "model.visual.";

/// Why a group is in a form the loader (and the kernels behind it) cannot take, if it is. A bias is refused unless `allow_bias` (the vision
/// tower's groups, whose bias the caller claims as an fp16 tensor).
fn unsupported(g: &GroupInfo, allow_bias: bool) -> Option<String> {
    if !g.bits.is_integer() {
        return Some(format!("{}-bit tiles (fractional widths are not supported)", g.bits.as_f64()));
    }
    if g.in_scales != "suh" || g.out_scales != "svh" {
        return Some(format!("scales are `{}`/`{}`, only fp16 `suh`/`svh` are uploaded as stored", g.in_scales, g.out_scales));
    }
    if g.bias.is_some() && !allow_bias {
        return Some("it has a bias".into());
    }
    None
}

impl GlmWeights {
    /// Map a pack's inventory onto the family's weights. See the module documentation: the result is exact (every tensor of the pack
    /// is in the tree once or in `skipped` once, and nothing else is) or an error naming every problem.
    pub fn map(inv: &PackInventory, cfg: &GlmConfig) -> Result<GlmWeights, LayoutError> {
        Self::map_owned(inv, cfg, &gb_cuda::moe::ExpertOwnership::all(cfg.ffn.routed_experts))
    }

    /// [`GlmWeights::map`] for a rank that holds only `own`'s routed experts (of every MoE layer): every tensor is still claimed and
    /// checked, but only the owned experts are placed (grouped into homogeneous subbanks) and so loaded. `map` is
    /// `map_owned(.., ExpertOwnership::all(E))`. The vision tower is mapped (when the config has one).
    pub fn map_owned(
        inv: &PackInventory,
        cfg: &GlmConfig,
        own: &gb_cuda::moe::ExpertOwnership,
    ) -> Result<GlmWeights, LayoutError> {
        Self::map_with(inv, cfg, own, true)
    }

    /// [`GlmWeights::map_owned`], and with `vision` false (`--no-vision`, a follower rank) the tower is not bound: every `model.visual.*`
    /// tensor and group is claimed and recorded in `skipped`, and `vision` is `None`, so the text model maps whatever the tower's format. A
    /// tensor outside the tower that no slot claims is still refused by name.
    pub fn map_with(
        inv: &PackInventory,
        cfg: &GlmConfig,
        own: &gb_cuda::moe::ExpertOwnership,
        vision: bool,
    ) -> Result<GlmWeights, LayoutError> {
        let mut b = Binder {
            inv,
            cfg,
            problems: vec![],
            used_plain: BTreeSet::new(),
            used_groups: BTreeSet::new(),
            regions: vec![],
            own: own.clone(),
            skipped: vec![],
        };
        let skip_tower = cfg.vision.is_some() && !vision;
        for (prefix, why) in &inv.refused {
            if skip_tower && prefix.starts_with(VISION_PREFIX) {
                // a tower group the scan would not load does not matter to a map that does not load the tower
                b.skipped.push(Skipped {
                    name: prefix.clone(),
                    group: true,
                    why: "the vision tower is not loaded (--no-vision)",
                });
                continue;
            }
            b.problems.push(Problem::Refused { prefix: prefix.clone(), why: why.clone() });
        }
        for issue in inv.issues.iter().filter(|i| !is_benign_issue(i)) {
            b.problems.push(Problem::HeaderIssue(issue.clone()));
        }

        let global = b.region(RegionKind::Global);
        let embed_tokens =
            b.plain(&format!("{P}embed_tokens.weight"), PlainDtype::Bf16, &[cfg.vocab, cfg.hidden], global, Category::Embedding);
        let final_norm =
            b.plain_x(&format!("{P}norm.weight"), PlainDtype::Bf16, &[cfg.hidden], Xform::Bf16ToF32, global, Category::Other);
        let lm_head = b.group("lm_head", cfg.hidden, cfg.vocab, Some(global), Category::Head);

        let mut layers = Vec::with_capacity(cfg.layers.len());
        for (i, spec) in cfg.layers.iter().enumerate() {
            let region = b.region(RegionKind::Trunk(i));
            layers.extend(b.layer(i, *spec, region, Category::Decoder, Category::RoutedExpert));
        }
        let mut mtp = Vec::with_capacity(cfg.mtp.len());
        for (j, spec) in cfg.mtp.iter().enumerate() {
            let index = cfg.mtp_layer_index(j);
            let region = b.region(RegionKind::Trunk(index));
            let l = format!("{P}layers.{index}.");
            let h = cfg.hidden;
            let eh_proj = b.group(&format!("{l}eh_proj"), 2 * h, h, Some(region), Category::Mtp);
            let enorm = b.plain_x(&format!("{l}enorm.weight"), PlainDtype::Bf16, &[h], Xform::Bf16ToF32, region, Category::Mtp);
            let hnorm = b.plain_x(&format!("{l}hnorm.weight"), PlainDtype::Bf16, &[h], Xform::Bf16ToF32, region, Category::Mtp);
            let shared_head_norm = b.plain_x(
                &format!("{l}shared_head.norm.weight"),
                PlainDtype::Bf16,
                &[h],
                Xform::Bf16ToF32,
                region,
                Category::Mtp,
            );
            if let Some(layer) = b.layer(index, *spec, region, Category::Mtp, Category::Mtp) {
                mtp.push(GlmMtpWeights { eh_proj, enorm, hnorm, shared_head_norm, layer });
            }
        }
        let vision = match (&cfg.vision, vision) {
            (Some(v), true) => Some(b.vision(v)),
            (Some(_), false) => {
                b.skip_vision();
                None
            }
            // no vision section: the tower's tensors, if the pack has any, are unexpected below
            (None, _) => None,
        };

        // fail closed: whatever no slot claimed is an error
        for name in inv.plain.keys().filter(|n| !b.used_plain.contains(n.as_str())) {
            b.problems.push(Problem::Unexpected { name: name.clone(), group: false });
        }
        for name in inv.groups.keys().filter(|n| !b.used_groups.contains(n.as_str())) {
            b.problems.push(Problem::Unexpected { name: name.clone(), group: true });
        }
        if !b.problems.is_empty() {
            return Err(LayoutError { problems: b.problems });
        }
        let mut regions = b.regions;
        for r in &mut regions {
            r.bytes = align_up(r.bytes);
        }
        Ok(GlmWeights {
            files: inv.files.clone(),
            regions,
            embed_tokens,
            final_norm,
            lm_head,
            layers,
            mtp,
            vision,
            skipped: b.skipped,
        })
    }
}
