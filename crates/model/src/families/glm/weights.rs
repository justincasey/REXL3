//! The typed description of every tensor GLM-5.3-Flash's pack holds, and where each one goes.
//!
//! [`GlmWeights`] is a tree with one leaf per tensor or EXL3 group: the 45 trunk layers (mHC residual tensors, a KDA or an MLA mixer,
//! a dense or a MoE feed-forward block), the NextN/MTP layer 45, the embedding, the final norm, `lm_head`, and, as a component of its
//! own, the vision tower. Every leaf records what the pack says (dtype, shape, bits, the shard and offset the bytes are read from)
//! and where the load puts them (a [`Dst`]: a region of device memory and an offset in it). It is built by one pure function over a
//! [`PackInventory`]: [`GlmWeights::map`], in `bind.rs`.
//!
//! The mapper is fail-closed: every tensor the family expects must be in the pack with the dtype and shape its config implies, and
//! every tensor in the pack must be claimed by a slot. A missing tensor, an unexpected one, a wrong dtype or shape, a refused group
//! and a non-uniform expert bank are each a named [`Problem`]; there is no "close enough" and no silently ignored tensor.
//!
//! Bits are read from the trellis shapes (`Group::bits`), never from `quantization_config.json`, which has no layer 45 and no mHC
//! entries. Nothing here names a bit width: the 3-bit layers of this pack are a fact its test asserts, not a rule the loader knows.
//!
//! Bytes are uploaded verbatim (the pack's fp16 scales, bf16/fp16/fp32 tensors and int16 trellises are what the kernels read), so a
//! group's device footprint is its trellis, `suh` and `svh`; the codebook marker (a 4-byte int32 the scan has already verified) is
//! recorded but not uploaded.

use super::plan::Region;
use gb_exl3::budget::Category;
use gb_exl3::{Bits, Codebook};

/// A region of device memory the plan fills: an index into [`GlmWeights::regions`] (and `LoadPlan::regions`).
pub type RegionId = u32;

/// Every destination offset is a multiple of this (the kernels read 16-byte vectors; 256 keeps tiles and rows off shared lines).
pub const ALIGN: u64 = 256;

pub fn align_up(x: u64) -> u64 {
    x.div_ceil(ALIGN) * ALIGN
}

/// Dtypes of the plain tensors this family loads. EXL3 trellises are int16 and are described by [`Exl3Linear`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlainDtype {
    F16,
    Bf16,
    F32,
}

impl PlainDtype {
    /// The safetensors dtype name.
    pub fn name(self) -> &'static str {
        match self {
            PlainDtype::F16 => "F16",
            PlainDtype::Bf16 => "BF16",
            PlainDtype::F32 => "F32",
        }
    }
    pub fn size(self) -> u64 {
        match self {
            PlainDtype::F16 | PlainDtype::Bf16 => 2,
            PlainDtype::F32 => 4,
        }
    }
}

/// Where a tensor's bytes are read from: a shard (an index into `GlmWeights::files`) and an absolute offset in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Src {
    pub file: u16,
    pub offset: u64,
}

/// Where a tensor's bytes go: an offset into a region of device memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dst {
    pub region: RegionId,
    pub offset: u64,
}

impl Dst {
    /// A leaf the mapper has not placed (a placeholder of a failed mapping; never in a returned tree).
    pub const NONE: Dst = Dst { region: RegionId::MAX, offset: 0 };
}

/// What the loader does to a plain tensor's bytes between the shard and the device. Almost every tensor is copied as stored
/// ([`Xform::None`]); the few the kernels read in another form say so here, in the plan, so the destination's size follows and the
/// executor owes the conversion (nothing is converted behind the plan's back).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Xform {
    /// The file's bytes, unchanged.
    None,
    /// bf16 widened to fp32 (an RMSNorm gamma: the norm kernels multiply by an fp32 weight, and GLM's gamma has no `+ 1`).
    Bf16ToF32,
    /// fp32 narrowed to fp16, which must be exact for every element (an mHC `fn` matrix: all 90 of the pack's are, and fp16
    /// halves the dominant read of a site). An element that would round fails the load.
    F32ToF16Exact,
    /// The router's selection bias: fp32 `[experts]` minus its mean (torch's reduction order), rounded to fp16, as fp32
    /// (`gb_cuda::moe::centred_bias_host`), the arithmetic the oracle applies at load.
    CentredBias,
    /// `kv_b_proj` fp16 `[heads * (nope + v), kv_lora_rank]` split per head and packed into `W_UK` and transposed `W_UV` for the MLA
    /// kernels (`gb_cuda::mla::pack_kv_b`): the same number of bytes, `uk` first, then `uvt`.
    PackKvB,
    /// The EXL3 tower's fp16 patch-embedding conv3d weight `[O, 3, 2, 14, 14]` folded over its two temporal slots into the patch GEMM's
    /// `[O, 592]` fp16 (columns `(channel, row, col)`, 588 real and 4 zero): `w0 + w1` summed exactly and rounded once to fp16. The bf16
    /// tower does the same fold on the device after its bf16 -> fp16 conversion (`gb_cuda::glm_vision::fold_patch`); an fp16 source is
    /// never sent through that conversion. The slot keeps the stored size: the fold fills its front, the rest is zero.
    FoldPatchF16,
    /// The EXL3 tower's fp16 downsample conv2d weight `[O, C, 2, 2]` as a GEMM weight `[O, 4C]` over four consecutive block-major patches
    /// (row `(kh * 2 + kw) * C + c`): a pure permutation of the fp16 values, as `gb_cuda::glm_vision::permute_downsample` lays out the bf16
    /// tower's after converting it.
    PermuteDownsampleF16,
}

impl Xform {
    /// Bytes the device holds for `src_bytes` of file.
    pub fn dst_bytes(self, src_bytes: u64) -> u64 {
        match self {
            Xform::Bf16ToF32 => src_bytes * 2,
            Xform::F32ToF16Exact => src_bytes / 2,
            Xform::None | Xform::CentredBias | Xform::PackKvB | Xform::FoldPatchF16 | Xform::PermuteDownsampleF16 => src_bytes,
        }
    }
}

/// One plain tensor, uploaded as stored or through one named [`Xform`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlainTensor {
    pub name: String,
    pub dtype: PlainDtype,
    pub shape: Vec<usize>,
    /// Bytes in the shard.
    pub bytes: u64,
    /// What happens to them on the way to the device.
    pub xform: Xform,
    /// Bytes on the device (`xform.dst_bytes(bytes)`).
    pub dst_bytes: u64,
    pub src: Src,
    pub dst: Dst,
    pub category: Category,
}

/// One uploaded part of an EXL3 group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exl3Part {
    pub bytes: u64,
    pub src: Src,
    pub dst: Dst,
}

/// One EXL3 linear layer: int16 trellis `[K/16, N/16, 16 x bits]`, fp16 `suh` `[K]` and `svh` `[N]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Exl3Linear {
    /// The tensor name without `.trellis`.
    pub prefix: String,
    pub k: usize,
    pub n: usize,
    pub bits: Bits,
    pub codebook: Codebook,
    pub trellis: Exl3Part,
    pub suh: Exl3Part,
    pub svh: Exl3Part,
    /// The codebook marker: read by the scan, not uploaded.
    pub marker: Option<(Src, u64)>,
    pub category: Category,
    /// `Some` when only some of the group's 128-column strips are placed (lane E, a TP rank's KDA `qkv_proj` heads: [`StripCut`]); `None`
    /// (every group of a single node, and of a rank without the KDA head split) places the whole group.
    pub cut: Option<StripCut>,
    /// `Some((k0, k1))` when only the input rows `[k0, k1)` of the group are placed (an expert's `down_proj` under the expert WIDTH
    /// split, `FfnConfig::width`): whole 16-row k tiles, so the trellis rows are one contiguous run of the stored trellis. `k` stays the
    /// stored group's; `trellis.bytes` and `suh.bytes` are the PLACED bytes, `trellis.src` / `suh.src` the stored parts' first bytes
    /// (the plan reads from `src + k0 / 16 * row` and `src + 2 k0`, as [`StripCut`] does for columns); `svh` is whole. `None` for every
    /// other group.
    pub rows: Option<(usize, usize)>,
}

/// The 128-column strips of an EXL3 group that a rank places and loads (lane E, GLM Phase 2: "the rank stores only its own qkv strips").
/// `k` and `n` of the [`Exl3Linear`] stay the stored group's (the kernels reduce a cut with the parent's plan, `plan_for(k, n)`); its
/// `trellis.bytes` and `svh.bytes` are the PLACED bytes, and `trellis.src` / `svh.src` the stored parts' first bytes. The plan reads the
/// kept strips as runs (`gb_cuda::exl3_split::cut_trellis_runs` / `cut_vector_runs`), which lay the cut out exactly as
/// `Exl3Weights::select_strips_planned` copies it on the device, so a loaded cut and a copied one are the same bytes; `suh` is whole.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StripCut {
    /// Strips `[lo, hi)` of the stored `n` columns, ascending and disjoint.
    pub ranges: Vec<(usize, usize)>,
}

impl StripCut {
    /// The placed columns.
    pub fn kept_columns(&self) -> usize {
        self.ranges.iter().map(|&(lo, hi)| hi - lo).sum::<usize>() * 128
    }
}

/// The 128-column strips of a KDA layer's fused `qkv_proj` (`[q | k | v]`, each third `heads x head_dim` wide) that carry heads
/// `[h0, h1)`: one range in each third. The one definition of a rank's qkv columns, for the mapper's rank strips and for the head split
/// (`model/tp.rs`).
pub fn kda_qkv_strips(heads: usize, head_dim: usize, h0: usize, h1: usize) -> Vec<(usize, usize)> {
    let third = heads * head_dim / 128;
    (0..3).map(|j| (j * third + h0 * head_dim / 128, j * third + h1 * head_dim / 128)).collect()
}

impl Exl3Linear {
    /// Bytes the kernels read: trellis plus scales (a [`StripCut`]'s placed bytes).
    pub fn weight_bytes(&self) -> u64 {
        self.trellis.bytes + self.suh.bytes + self.svh.bytes
    }
    /// Bytes on disk: the weight bytes and the marker (of a whole group; a cut's are its placed bytes).
    pub fn disk_bytes(&self) -> u64 {
        self.weight_bytes() + self.marker.map_or(0, |m| m.1)
    }
    /// The columns placed on the device: `n`, or a [`StripCut`]'s kept columns.
    pub fn placed_n(&self) -> usize {
        self.cut.as_ref().map_or(self.n, |c| c.kept_columns())
    }
}

// --------------------------------------------------------------------------------------------------------------- the tree

/// mHC hyper-connection tensors of one sublayer: `fn` (fp32 `[streams x (streams + 2), streams x hidden]`), `base` (fp32
/// `[streams x (streams + 2)]`) and `scale` (fp32 `[3]`).
#[derive(Clone, Debug, PartialEq)]
pub struct HcWeights {
    pub fn_: PlainTensor,
    pub base: PlainTensor,
    pub scale: PlainTensor,
}

/// The mHC tensors of a trunk layer: one set around the mixer, one around the feed-forward block. The MTP layer has none (plain
/// residual).
#[derive(Clone, Debug, PartialEq)]
pub struct MhcWeights {
    pub attn: HcWeights,
    pub ffn: HcWeights,
}

/// Kimi delta attention. Everything lives under `self_attn.` (as for MLA layers). The low-rank `f_*` and `g_*` projections have
/// rank `head_dim` (128): not a config field, pinned to `kda.head_dim` because that is what the pack holds (a different rank is a
/// shape error).
#[derive(Clone, Debug, PartialEq)]
pub struct KdaWeights {
    /// `[hidden -> 3 x heads x head_dim]`: q, k and v fused.
    pub qkv_proj: Exl3Linear,
    /// fp16 `[heads, hidden]`.
    pub b_proj: PlainTensor,
    /// fp16 `[head_dim, hidden]`.
    pub f_a_proj: PlainTensor,
    /// fp16 `[heads x head_dim, head_dim]`.
    pub f_b_proj: PlainTensor,
    pub g_a_proj: PlainTensor,
    pub g_b_proj: PlainTensor,
    /// fp32 `[heads]`.
    pub a_log: PlainTensor,
    /// fp32 `[heads x head_dim]` (per channel, unlike `a_log`).
    pub dt_bias: PlainTensor,
    /// bf16 `[3 x heads x head_dim, 1, conv_kernel]`.
    pub conv1d_weight: PlainTensor,
    /// bf16 `[head_dim]`.
    pub o_norm: PlainTensor,
    /// `[heads x head_dim -> hidden]`.
    pub o_proj: Exl3Linear,
}

/// The DSA lightning indexer of an attention layer (the layer's own: a layer that shares a preceding one's selection has none).
#[derive(Clone, Debug, PartialEq)]
pub struct IndexerWeights {
    /// `[q_lora_rank -> heads x head_dim]`: EXL3 in most packs, plain fp16 `[heads x head_dim, q_lora_rank]` in some.
    pub wq_b: IndexerWqB,
    /// fp16 `[head_dim, hidden]`.
    pub wk: PlainTensor,
    /// fp16 `[heads, hidden]`.
    pub weights_proj: PlainTensor,
    /// fp16 `[head_dim]`: the key norm is a LayerNorm, so it has a bias.
    pub k_norm_weight: PlainTensor,
    pub k_norm_bias: PlainTensor,
    /// The k-pool compression, present when the config pools keys: fp32 `[kpool, head_dim]` and fp16 `[head_dim, hidden]`.
    pub kpool: Option<KpoolWeights>,
}

/// The indexer's query projection as the pack stores it.
#[derive(Clone, Debug, PartialEq)]
pub enum IndexerWqB {
    /// An EXL3 group `q_lora_rank x (heads x head_dim)`.
    Exl3(Exl3Linear),
    /// A plain fp16 matrix `[heads x head_dim, q_lora_rank]` (`wq_b.weight`, no EXL3 group): run by the fp16 GEMV (`moe::linear_f16`).
    PlainF16(PlainTensor),
}

impl IndexerWqB {
    /// The EXL3 group, when the pack stores one.
    pub fn exl3(&self) -> Option<&Exl3Linear> {
        match self {
            IndexerWqB::Exl3(g) => Some(g),
            IndexerWqB::PlainF16(_) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KpoolWeights {
    pub compress_ape: PlainTensor,
    pub compress_gate: PlainTensor,
}

/// Multi-head latent attention without RoPE.
#[derive(Clone, Debug, PartialEq)]
pub struct MlaWeights {
    /// `[hidden -> q_lora_rank]`.
    pub q_a_proj: Exl3Linear,
    /// bf16 `[q_lora_rank]`.
    pub q_a_layernorm: PlainTensor,
    /// `[q_lora_rank -> heads x qk_head_dim]`.
    pub q_b_proj: Exl3Linear,
    /// `[hidden -> kv_lora_rank + qk_rope_head_dim]`.
    pub kv_a_proj_with_mqa: Exl3Linear,
    /// bf16 `[kv_lora_rank]`.
    pub kv_a_layernorm: PlainTensor,
    /// fp16 `[heads x (qk_nope_head_dim + v_head_dim), kv_lora_rank]` (plain, not EXL3, and fp16, not bf16).
    pub kv_b_proj: PlainTensor,
    /// `[heads x v_head_dim -> hidden]`.
    pub o_proj: Exl3Linear,
    pub indexer: Option<IndexerWeights>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MixerWeights {
    Kda(Box<KdaWeights>),
    Mla(Box<MlaWeights>),
}

/// A gated MLP in EXL3: `down(act(gate(x)) * up(x))`.
#[derive(Clone, Debug, PartialEq)]
pub struct GatedMlp {
    pub gate_proj: Exl3Linear,
    pub up_proj: Exl3Linear,
    pub down_proj: Exl3Linear,
}

/// One routed expert (an entry of an [`ExpertBank`]).
#[derive(Clone, Debug, PartialEq)]
pub struct ExpertWeights {
    pub gate_proj: Exl3Linear,
    pub up_proj: Exl3Linear,
    pub down_proj: Exl3Linear,
}

/// Where one projection's tensors sit, stacked over the experts of a bank in its region: expert `e`'s trellis is at
/// `trellis + e x trellis_stride`, its `suh` at `suh + e x suh_stride`, its `svh` at `svh + e x svh_stride`. Strides are the
/// tensors' exact sizes, so a kernel can take (base, stride) or build pointer tables from the same numbers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProjLayout {
    pub trellis: u64,
    pub trellis_stride: u64,
    pub suh: u64,
    pub suh_stride: u64,
    pub svh: u64,
    pub svh_stride: u64,
}

impl ProjLayout {
    pub fn trellis_at(&self, expert: usize) -> u64 {
        self.trellis + expert as u64 * self.trellis_stride
    }
    pub fn suh_at(&self, expert: usize) -> u64 {
        self.suh + expert as u64 * self.suh_stride
    }
    pub fn svh_at(&self, expert: usize) -> u64 {
        self.svh + expert as u64 * self.svh_stride
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertLayout {
    pub gate: ProjLayout,
    pub up: ProjLayout,
    pub down: ProjLayout,
}

/// One homogeneous subbank of a layer's routed experts on this rank: the owned experts of one key (gate, up and down widths, codebook),
/// ascending global id, laid out in the bank's region as the grouped expert kernel consumes them (`layout`, constant strides; local
/// index = position in `ids`).
#[derive(Clone, Debug, PartialEq)]
pub struct ExpertSubbank {
    pub key: gb_cuda::moe::SubbankKey,
    pub ids: Vec<u32>,
    pub layout: ExpertLayout,
}

/// The routed experts of one MoE layer this rank holds, in one region of device memory: per [`ExpertSubbank`] (key order), all trellises
/// of a projection back to back, then `suh` and `svh` stacked (`[experts, K]` and `[experts, N]`), subbanks one after the other. A
/// uniform layer on a single node is ONE subbank of every expert, laid out exactly as the single bank always was (`layout`). `experts[e]`
/// is expert id `e` for EVERY global id (all are claimed; only the owned ones are placed and loaded): ids come from the tensor names,
/// never from the file order (the shards list them as 0, 1, 10, 100, 101, ...).
#[derive(Clone, Debug, PartialEq)]
pub struct ExpertBank {
    pub layer: usize,
    pub region: RegionId,
    /// Width of gate and up, and of down, of the first subbank (of every expert in a uniform layer).
    pub bits_gate_up: Bits,
    pub bits_down: Bits,
    pub codebook: Codebook,
    pub hidden: usize,
    /// The layer's expert width (the whole `moe_intermediate`, also under a [`ExpertBank::width`] share).
    pub inter: usize,
    pub experts: Vec<ExpertWeights>,
    /// The first subbank's layout (a uniform single-node bank: THE layout).
    pub layout: ExpertLayout,
    /// The owned experts by key, in key order (one for a uniform layer).
    pub subbanks: Vec<ExpertSubbank>,
    /// The experts this rank holds (`ExpertOwnership::all` on a single node and under a width share).
    pub own: gb_cuda::moe::ExpertOwnership,
    /// `Some` under the expert WIDTH split: every expert is placed with only the share's intermediate columns
    /// `share.cols(inter) = [c0, c1)` (gate/up: a one-range [`StripCut`] of strips `[c0/128, c1/128)`; down: [`Exl3Linear::rows`]
    /// `(c0, c1)`), so each subbank's layout is `gb_cuda::moe::SubbankRegion::stacked(key, n, hidden, c1 - c0, ..)` and the device binds
    /// it with `MoeExperts::from_region_width`. `None`: whole experts.
    pub width: Option<gb_cuda::moe::WidthShare>,
}

/// A MoE layer: fp16 router `[experts, hidden]`, fp32 selection bias `[experts]`, the routed experts and the shared expert.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeWeights {
    /// `mlp.gate.weight`.
    pub router: PlainTensor,
    /// `mlp.gate.e_score_correction_bias`.
    pub e_score_correction_bias: PlainTensor,
    pub experts: ExpertBank,
    /// `None` when the config has no shared expert.
    pub shared: Option<GatedMlp>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FfnWeights {
    Dense(Box<GatedMlp>),
    Moe(Box<MoeWeights>),
}

/// One decoder layer (a trunk layer, or the MTP layer's body).
#[derive(Clone, Debug, PartialEq)]
pub struct GlmLayerWeights {
    pub index: usize,
    /// bf16 `[hidden]`.
    pub input_layernorm: PlainTensor,
    pub post_attention_layernorm: PlainTensor,
    /// `None` for a plain-residual layer (the MTP layer).
    pub mhc: Option<MhcWeights>,
    pub mixer: MixerWeights,
    pub ffn: FfnWeights,
}

/// The NextN/MTP layer (index 45): `eh_proj` fuses the embedded token and the trunk's hidden, then a full MLA + indexer + MoE layer
/// on a plain residual, then `shared_head.norm` and the trunk's `lm_head`.
#[derive(Clone, Debug, PartialEq)]
pub struct GlmMtpWeights {
    /// `[2 x hidden -> hidden]`.
    pub eh_proj: Exl3Linear,
    pub enorm: PlainTensor,
    pub hnorm: PlainTensor,
    pub shared_head_norm: PlainTensor,
    pub layer: GlmLayerWeights,
}

/// A plain weight (bf16 in the bf16 tower, fp16 in the EXL3 tower's patch embedding and downsample) with its optional bias.
#[derive(Clone, Debug, PartialEq)]
pub struct PlainLinear {
    pub weight: PlainTensor,
    pub bias: Option<PlainTensor>,
}

/// A linear of the vision tower's blocks or merger as the pack stores it.
#[derive(Clone, Debug, PartialEq)]
pub enum VisionLinear {
    /// A bf16 weight `[n, k]` and its optional bias `[n]` (the bf16 tower: converted to fp16 on the device, run by the dense GEMM).
    Plain(PlainLinear),
    /// An EXL3 group `k -> n` and its optional bias `[n]`: the group's `.bias`, which the mapper takes only as fp16 (the tower's kernels read
    /// `const half*`; a bf16 or fp32 group bias is refused by name, never reinterpreted). Run by the EXL3 prefill GEMM without the bias; the
    /// tower adds the bias where the bf16 tower adds its own (fused into the next kernel, in fp16, as exllamav3's `y += bias`).
    Exl3 { group: Exl3Linear, bias: Option<PlainTensor> },
}

impl VisionLinear {
    pub fn bias(&self) -> Option<&PlainTensor> {
        match self {
            VisionLinear::Plain(l) => l.bias.as_ref(),
            VisionLinear::Exl3 { bias, .. } => bias.as_ref(),
        }
    }
    /// The tensor the mapper names this linear by (the weight, or the group's prefix).
    pub fn name(&self) -> &str {
        match self {
            VisionLinear::Plain(l) => &l.weight.name,
            VisionLinear::Exl3 { group, .. } => &group.prefix,
        }
    }
}

/// The attention input projection of a vision block.
#[derive(Clone, Debug, PartialEq)]
pub enum VisionQkv {
    /// One bf16 `[3 x hidden, hidden]` weight and its `[3 x hidden]` bias (the bf16 tower).
    Fused(PlainLinear),
    /// q, k and v: three EXL3 groups `hidden -> hidden` with their fp16 biases (the EXL3 tower). The three biases are placed back to back in
    /// the region (q, k, v), so the tower reads them as the fused `[3 x hidden]` bias; the tower checks the adjacency when it binds.
    Split(Box<[VisionLinear; 3]>),
}

/// One block of the vision tower: bf16 RMSNorm gammas, and linears plain bf16 or EXL3 ([`VisionTowerFormat`]).
#[derive(Clone, Debug, PartialEq)]
pub struct VisionBlock {
    pub norm1: PlainTensor,
    pub norm2: PlainTensor,
    pub qkv: VisionQkv,
    pub proj: VisionLinear,
    /// `[head_dim]`.
    pub q_norm: PlainTensor,
    pub k_norm: PlainTensor,
    pub gate_proj: VisionLinear,
    pub up_proj: VisionLinear,
    pub down_proj: VisionLinear,
}

/// The merger: proj, a gated MLP (gate, up, down; no biases) and the post-projection LayerNorm (bf16 or fp16 by format).
#[derive(Clone, Debug, PartialEq)]
pub struct VisionMerger {
    pub proj: VisionLinear,
    pub gate_proj: VisionLinear,
    pub up_proj: VisionLinear,
    pub down_proj: VisionLinear,
    pub post_projection_norm_weight: PlainTensor,
    pub post_projection_norm_bias: PlainTensor,
}

/// The vision tower: a component of its own, loaded or skipped as a whole. The bf16 tower is 347 plain bf16 tensors; the EXL3 tower is 7
/// biased groups a block and 4 merger groups, fp16 patch embedding, downsample and merger norm, and bf16 RMSNorm gammas.
#[derive(Clone, Debug, PartialEq)]
pub struct GlmVisionWeights {
    pub format: super::config::VisionTowerFormat,
    pub patch_embed: PlainLinear,
    pub blocks: Vec<VisionBlock>,
    pub post_layernorm: PlainTensor,
    pub downsample: PlainLinear,
    pub merger: VisionMerger,
}

/// A tensor or EXL3 group of the pack the mapper claimed and does not load, and why (the pack's tensors are the tree's leaves plus these,
/// each exactly once).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skipped {
    pub name: String,
    pub group: bool,
    pub why: &'static str,
}

/// Every tensor of the pack, typed.
#[derive(Clone, Debug, PartialEq)]
pub struct GlmWeights {
    /// The shard files, as in the inventory; a [`Src::file`] indexes this.
    pub files: Vec<String>,
    /// The device regions the leaves are placed in (sizes include alignment padding).
    pub regions: Vec<Region>,
    /// bf16 `[vocab, hidden]`.
    pub embed_tokens: PlainTensor,
    /// bf16 `[hidden]`.
    pub final_norm: PlainTensor,
    /// `[hidden -> vocab]`.
    pub lm_head: Exl3Linear,
    /// The trunk, `layers[i].index == i`.
    pub layers: Vec<GlmLayerWeights>,
    /// The NextN layers after the trunk (none or one).
    pub mtp: Vec<GlmMtpWeights>,
    /// `None` without a `vision_config`, or when the map skips the tower (`--no-vision`: every `model.visual.*` tensor is in `skipped`).
    pub vision: Option<GlmVisionWeights>,
    /// What the map claimed and does not load (the tower under `--no-vision`, the EXL3 tower's fused bf16 qkv), by name.
    pub skipped: Vec<Skipped>,
}

// -------------------------------------------------------------------------------------------------------------- traversal

/// A tensor or group of the tree, as the plan builder and the totals see it.
#[derive(Clone, Copy, Debug)]
pub enum Leaf<'a> {
    Plain(&'a PlainTensor),
    Exl3(&'a Exl3Linear),
}

type Visit<'a, 'f> = &'f mut dyn FnMut(Leaf<'a>);

impl HcWeights {
    fn visit<'a>(&'a self, f: Visit<'a, '_>) {
        f(Leaf::Plain(&self.fn_));
        f(Leaf::Plain(&self.base));
        f(Leaf::Plain(&self.scale));
    }
}

impl GatedMlp {
    fn visit<'a>(&'a self, f: Visit<'a, '_>) {
        f(Leaf::Exl3(&self.gate_proj));
        f(Leaf::Exl3(&self.up_proj));
        f(Leaf::Exl3(&self.down_proj));
    }
}

impl KdaWeights {
    fn visit<'a>(&'a self, f: Visit<'a, '_>) {
        f(Leaf::Exl3(&self.qkv_proj));
        for t in [
            &self.b_proj,
            &self.f_a_proj,
            &self.f_b_proj,
            &self.g_a_proj,
            &self.g_b_proj,
            &self.a_log,
            &self.dt_bias,
            &self.conv1d_weight,
            &self.o_norm,
        ] {
            f(Leaf::Plain(t));
        }
        f(Leaf::Exl3(&self.o_proj));
    }
}

impl MlaWeights {
    fn visit<'a>(&'a self, f: Visit<'a, '_>) {
        for g in [&self.q_a_proj, &self.q_b_proj, &self.kv_a_proj_with_mqa, &self.o_proj] {
            f(Leaf::Exl3(g));
        }
        for t in [&self.q_a_layernorm, &self.kv_a_layernorm, &self.kv_b_proj] {
            f(Leaf::Plain(t));
        }
        if let Some(ix) = &self.indexer {
            match &ix.wq_b {
                IndexerWqB::Exl3(g) => f(Leaf::Exl3(g)),
                IndexerWqB::PlainF16(t) => f(Leaf::Plain(t)),
            }
            for t in [&ix.wk, &ix.weights_proj, &ix.k_norm_weight, &ix.k_norm_bias] {
                f(Leaf::Plain(t));
            }
            if let Some(kp) = &ix.kpool {
                f(Leaf::Plain(&kp.compress_ape));
                f(Leaf::Plain(&kp.compress_gate));
            }
        }
    }
}

impl ExpertBank {
    /// Whether this is today's single bank: one subbank holding every expert, whole (a width share is never one).
    pub fn is_single(&self) -> bool {
        self.subbanks.len() == 1 && self.own.is_all() && self.width.is_none()
    }

    fn visit<'a>(&'a self, f: Visit<'a, '_>) {
        // only the owned experts are placed and loaded
        for e in self.own.owned().iter().map(|&g| &self.experts[g as usize]) {
            f(Leaf::Exl3(&e.gate_proj));
            f(Leaf::Exl3(&e.up_proj));
            f(Leaf::Exl3(&e.down_proj));
        }
    }
}

impl GlmLayerWeights {
    fn visit<'a>(&'a self, f: Visit<'a, '_>) {
        f(Leaf::Plain(&self.input_layernorm));
        f(Leaf::Plain(&self.post_attention_layernorm));
        if let Some(m) = &self.mhc {
            m.attn.visit(f);
            m.ffn.visit(f);
        }
        match &self.mixer {
            MixerWeights::Kda(k) => k.visit(f),
            MixerWeights::Mla(m) => m.visit(f),
        }
        match &self.ffn {
            FfnWeights::Dense(d) => d.visit(f),
            FfnWeights::Moe(m) => {
                f(Leaf::Plain(&m.router));
                f(Leaf::Plain(&m.e_score_correction_bias));
                m.experts.visit(f);
                if let Some(s) = &m.shared {
                    s.visit(f);
                }
            }
        }
    }
}

impl PlainLinear {
    fn visit<'a>(&'a self, f: Visit<'a, '_>) {
        f(Leaf::Plain(&self.weight));
        if let Some(b) = &self.bias {
            f(Leaf::Plain(b));
        }
    }
}

impl VisionLinear {
    fn visit<'a>(&'a self, f: Visit<'a, '_>) {
        match self {
            VisionLinear::Plain(l) => l.visit(f),
            VisionLinear::Exl3 { group, bias } => {
                f(Leaf::Exl3(group));
                if let Some(b) = bias {
                    f(Leaf::Plain(b));
                }
            }
        }
    }
}

impl GlmVisionWeights {
    /// Every tensor of the tower.
    pub fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        self.patch_embed.visit(f);
        for b in &self.blocks {
            for t in [&b.norm1, &b.norm2, &b.q_norm, &b.k_norm] {
                f(Leaf::Plain(t));
            }
            match &b.qkv {
                VisionQkv::Fused(l) => l.visit(f),
                VisionQkv::Split(qkv) => qkv.iter().for_each(|l| l.visit(f)),
            }
            for l in [&b.proj, &b.gate_proj, &b.up_proj, &b.down_proj] {
                l.visit(f);
            }
        }
        f(Leaf::Plain(&self.post_layernorm));
        self.downsample.visit(f);
        let m = &self.merger;
        for l in [&m.proj, &m.gate_proj, &m.up_proj, &m.down_proj] {
            l.visit(f);
        }
        f(Leaf::Plain(&m.post_projection_norm_weight));
        f(Leaf::Plain(&m.post_projection_norm_bias));
    }

    /// The plain tensors the device preparation converts bf16 -> fp16 in place (`vision::Weights::prepare`): every Linear, Conv and LayerNorm
    /// tensor of the bf16 tower except the patch and downsample weights (folded and permuted from bf16 by their own kernels). The RMSNorm gammas
    /// stay bf16. Empty for the EXL3 tower: its plain tensors are fp16 as stored (or bf16 gammas), and reinterpreting fp16 bytes as bf16 would
    /// decode garbage, so nothing of it is converted. Every tensor listed is a bf16 tensor of the plan (a CPU test holds the two to that).
    pub fn bf16_conversions(&self) -> Vec<&PlainTensor> {
        let mut v: Vec<&PlainTensor> = Vec::new();
        if self.format != super::config::VisionTowerFormat::Bf16 {
            return v;
        }
        fn lin(l: &PlainLinear) -> impl Iterator<Item = &PlainTensor> {
            std::iter::once(&l.weight).chain(l.bias.as_ref())
        }
        v.extend(self.patch_embed.bias.as_ref());
        v.extend(self.downsample.bias.as_ref());
        for b in &self.blocks {
            if let VisionQkv::Fused(l) = &b.qkv {
                v.extend(lin(l));
            }
            for l in [&b.proj, &b.gate_proj, &b.up_proj, &b.down_proj] {
                if let VisionLinear::Plain(l) = l {
                    v.extend(lin(l));
                }
            }
        }
        let m = &self.merger;
        for l in [&m.proj, &m.gate_proj, &m.up_proj, &m.down_proj] {
            if let VisionLinear::Plain(l) = l {
                v.extend(lin(l));
            }
        }
        v.push(&m.post_projection_norm_weight);
        v.push(&m.post_projection_norm_bias);
        v
    }
}

impl GlmWeights {
    /// Every leaf of the tree (text, MTP and vision), in tree order.
    pub fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        f(Leaf::Plain(&self.embed_tokens));
        f(Leaf::Plain(&self.final_norm));
        f(Leaf::Exl3(&self.lm_head));
        for l in &self.layers {
            l.visit(f);
        }
        for m in &self.mtp {
            f(Leaf::Exl3(&m.eh_proj));
            for t in [&m.enorm, &m.hnorm, &m.shared_head_norm] {
                f(Leaf::Plain(t));
            }
            m.layer.visit(f);
        }
        if let Some(v) = &self.vision {
            v.visit(f);
        }
    }

    /// (EXL3 groups, plain tensors) in the tree.
    pub fn counts(&self) -> (usize, usize) {
        let (mut groups, mut plain) = (0, 0);
        self.visit(&mut |l| match l {
            Leaf::Plain(_) => plain += 1,
            Leaf::Exl3(_) => groups += 1,
        });
        (groups, plain)
    }

    /// The MoE layers' banks (trunk, then MTP), in layer order.
    pub fn banks(&self) -> impl Iterator<Item = &ExpertBank> {
        let trunk = self.layers.iter();
        let mtp = self.mtp.iter().map(|m| &m.layer);
        trunk.chain(mtp).filter_map(|l| match &l.ffn {
            FfnWeights::Moe(m) => Some(&m.experts),
            FfnWeights::Dense(_) => None,
        })
    }

    /// The layer with checkpoint index `i` (the trunk, then the MTP layer).
    pub fn layer(&self, i: usize) -> Option<&GlmLayerWeights> {
        self.layers.get(i).or_else(|| self.mtp.iter().map(|m| &m.layer).find(|l| l.index == i))
    }
}
