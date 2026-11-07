//! The typed device weights of GLM-5.3-Flash: every tensor of the pack as the kernel that reads it wants it, bound to views of the
//! plan's regions (nothing is allocated here and nothing is copied: the loader fills the regions, and a view is an address).
//!
//! [`bind`] walks the typed description ([`GlmWeights`]) and turns each leaf into the kernel crate's type: [`Exl3Weights`] for every EXL3
//! linear, [`kda::Weights`]-shaped buffers for the KDA tensors, [`MlaWeights`] for the packed `kv_b_proj`, the DSA indexer's weights, an
//! [`MhcSite`] per mHC site, a [`MoeLayer`] whose experts sit in ONE region ([`MoeExperts::from_region`]), and the shared expert and the
//! three dense MLPs as [`SharedExpert`]s (the same clamped SwiGLU). A model of fewer layers than the pack's (`layers`) binds only those,
//! which is how the bisection against the oracle runs without the 92 GiB.

use super::config::GlmConfig;
use super::load::Regions;
use super::weights::{
    Exl3Linear, ExpertBank, FfnWeights, GatedMlp, GlmLayerWeights, GlmWeights, HcWeights, IndexerWqB, MixerWeights, PlainTensor,
};
use crate::error::ModelError;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::mhc::{MhcSite, Ty};
use gb_cuda::moe::{
    ExpertOwnership, ExpertRegion, MoeDims, MoeExperts, MoeLayer, MoeRouter, MoeSubbankLayer, MoeSubbanks, ProjOffsets,
    SharedExpert, SubbankRegion,
};
use gb_cuda::{dsa, kda, mla, DevBuf};
use gb_exl3::checkpoint::Group;

/// One KDA layer's tensors (`self_attn.*`): the fused `qkv_proj` and `o_proj` are EXL3, the rest plain, as the pack stores them.
pub struct KdaDevice {
    /// The fused q | k | v projection; on a TP rank that loaded only its heads' strips (`qkv_cut`), those strips (`placed_n` columns).
    pub qkv: Exl3Weights,
    /// The strips of the stored `qkv_proj` that `qkv` holds (lane E, the rank strips of the KDA head split; `None`: the whole group).
    pub qkv_cut: Option<super::weights::StripCut>,
    pub o: Exl3Weights,
    pub b: DevBuf,
    pub f_a: DevBuf,
    pub g_a: DevBuf,
    pub f_b: DevBuf,
    pub g_b: DevBuf,
    pub conv_w: DevBuf,
    pub a_log: DevBuf,
    pub dt_bias: DevBuf,
    pub o_norm: DevBuf,
}

impl KdaDevice {
    /// The borrowed view the KDA kernels take.
    pub fn view(&self) -> kda::Weights<'_> {
        kda::Weights {
            b: &self.b,
            f_a: &self.f_a,
            g_a: &self.g_a,
            f_b: &self.f_b,
            g_b: &self.g_b,
            conv_w: &self.conv_w,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            o_norm: &self.o_norm,
        }
    }
}

/// The DSA indexer of an MLA layer: `wq_b` is EXL3 (from the normed q latent), the three small projections are plain fp16 (run by the
/// router's GEMV chain), and the key LayerNorm and the pool's position embedding are the DSA kernels' own weights.
pub struct IndexerDevice {
    pub wq_b: WqBDevice,
    /// fp16 `[128, 4096]`: the index key.
    pub wk: DevBuf,
    /// fp16 `[32, 4096]`: the per-head scoring weights.
    pub weights_proj: DevBuf,
    /// fp16 `[128, 4096]`: the per-token pool gate scores.
    pub compress_gate: DevBuf,
    pub dsa: dsa::IndexerWeights,
}

/// The indexer's query projection on the device.
pub enum WqBDevice {
    Exl3(Exl3Weights),
    /// fp16 `[heads x head_dim, q_lora_rank]`, run by `moe::linear_f16` (the fp16 GEMV of `wk` and `weights_proj`) on every path.
    F16 {
        w: DevBuf,
        k: usize,
        n: usize,
    },
}

impl WqBDevice {
    /// `y = wq_b(x)` for `rows` fp16 rows of `x`, fp16 out; `big`: the EXL3 large-row path (the fp16 GEMV has one path).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &DevBuf,
        y: &mut DevBuf,
        rows: usize,
        ws: &mut DevBuf,
        st: &gb_cuda::Stream,
        big: bool,
    ) -> Result<(), ModelError> {
        match self {
            WqBDevice::Exl3(w) if big => w.forward_big(x, y, rows, Dtype::F16, Dtype::F16, ws, st)?,
            WqBDevice::Exl3(w) => w.forward(x, y, rows, Dtype::F16, Dtype::F16, ws, st)?,
            WqBDevice::F16 { w, k, n } => gb_cuda::moe::linear_f16(x, Dtype::F16, w, y, rows, *k, *n, st)?,
        }
        Ok(())
    }
}

/// One MLA layer's tensors.
pub struct MlaDevice {
    pub q_a: Exl3Weights,
    pub q_b: Exl3Weights,
    pub kv_a: Exl3Weights,
    pub o: Exl3Weights,
    /// fp32 `[1536]` and `[512]`: RMSNorm gammas, multiplied as stored.
    pub q_a_norm: DevBuf,
    pub kv_a_norm: DevBuf,
    /// `kv_b_proj` split and packed (`W_UK`, transposed `W_UV`).
    pub kv_b: mla::MlaWeights,
    pub indexer: IndexerDevice,
}

pub enum MixerDevice {
    Kda(Box<KdaDevice>),
    Mla(Box<MlaDevice>),
}

pub enum FfnDevice {
    /// Layers 0-2: a gated MLP of width 12,288 with the clamped SwiGLU.
    Dense(Box<SharedExpert>),
    /// Layers 3-44: router, the 288 routed experts in one region, and the shared expert.
    Moe { layer: Box<MoeLayer>, shared: Box<SharedExpert> },
    /// A MoE layer whose experts on this rank are more than one homogeneous subbank (a mixed-width pack) or a partial ownership: router,
    /// the subbanks (one region), and the shared expert. Run with `MoeSubbanks::{run_pairs, run_prefill_pairs}`, the exchange when
    /// there is more than one rank, and `moe::combine_pairs` (the shared expert goes there only).
    MoeSubbanks { layer: Box<MoeSubbankLayer>, shared: Box<SharedExpert> },
}

/// A MoE layer's routed experts as bound: today's single bank, or subbanks.
pub enum BoundExperts {
    Single(Box<MoeExperts>),
    Subbanks(MoeSubbanks),
}

pub struct LayerDevice {
    pub index: usize,
    /// fp32 `[4096]` RMSNorm gammas (`input_layernorm` before the mixer, `post_attention_layernorm` before the FFN).
    pub input_norm: DevBuf,
    pub post_norm: DevBuf,
    pub attn_site: MhcSite,
    pub ffn_site: MhcSite,
    pub mixer: MixerDevice,
    pub ffn: FfnDevice,
}

/// The NextN (MTP) layer on the device: `eh_proj` fuses the normed token embedding and the normed tapped hidden state, then a full MLA +
/// indexer + MoE block on a PLAIN residual (no mHC sites), then `shared_head.norm` in front of the trunk's `lm_head`. The norms are fp32
/// gammas multiplied as stored, like the trunk's.
pub struct MtpDevice {
    /// `[2 x hidden -> hidden]`, 4-bit.
    pub eh_proj: Exl3Weights,
    pub enorm: DevBuf,
    pub hnorm: DevBuf,
    pub shared_head_norm: DevBuf,
    /// `input_layernorm` (before the attention) and `post_attention_layernorm` (before the feed-forward block).
    pub input_norm: DevBuf,
    pub post_norm: DevBuf,
    pub mixer: Box<MlaDevice>,
    pub moe: MtpMoe,
    pub shared: Box<SharedExpert>,
}

/// The NextN layer's router and routed experts: one bank (a uniform layer), or subbanks (a mixed-width layer on one node, every expert; or a
/// tensor-parallel rank's share). The drafter runs subbanks holding every expert and refuses a share by name (`MTP_SHARE`): a share needs
/// the exchange, and `GlmModel::enable_tp` refuses a model with the NextN layer.
pub enum MtpMoe {
    Single(Box<MoeLayer>),
    Subbanks(Box<MoeSubbankLayer>),
}

/// Every weight of the text model on the device.
pub struct GlmDevice {
    /// bf16 `[vocab, hidden]`.
    pub embed: DevBuf,
    /// fp32 `[hidden]`: the final RMSNorm gamma.
    pub final_norm: DevBuf,
    pub lm_head: Exl3Weights,
    pub layers: Vec<LayerDevice>,
    /// The NextN layer, when it was loaded ([`bind_with`] `with_mtp`).
    pub mtp: Option<MtpDevice>,
    /// The device regions (the views above keep their blocks alive; this keeps the count and the bytes).
    pub regions: Regions,
}

struct Binder<'a> {
    regions: &'a Regions,
    moe: MoeDims,
}

fn cfg_err(m: String) -> ModelError {
    ModelError::Config(m)
}

/// The kernel crate's view of an EXL3 group the plan placed in `regions` (trellis, `suh`, `svh` as stored; no bias: the GLM kernels and the
/// vision tower add their biases themselves).
pub(super) fn exl3_weights(regions: &Regions, g: &Exl3Linear) -> Result<Exl3Weights, ModelError> {
    let group = Group {
        prefix: g.prefix.clone(),
        bits: g.bits,
        codebook: g.codebook,
        k: g.k,
        // a rank's strips (`StripCut`) are bound as the group of its placed columns
        n: g.placed_n(),
        in_scales: "suh",
        out_scales: "svh",
        bias: false,
        files: vec![],
        bytes: g.disk_bytes(),
    };
    let part = |p: &super::weights::Exl3Part| regions.view(p.dst.region, p.dst.offset, p.bytes);
    Exl3Weights::from_parts(&group, part(&g.trellis)?, part(&g.suh)?, part(&g.svh)?, None)
        .map_err(|e| cfg_err(format!("`{}`: {e}", g.prefix)))
}

impl Binder<'_> {
    fn plain(&self, t: &PlainTensor) -> Result<DevBuf, ModelError> {
        self.regions.view(t.dst.region, t.dst.offset, t.dst_bytes)
    }

    fn exl3(&self, g: &Exl3Linear) -> Result<Exl3Weights, ModelError> {
        exl3_weights(self.regions, g)
    }

    fn site(&self, h: &HcWeights) -> Result<MhcSite, ModelError> {
        // `fn` is fp16 on the device (the plan converted it, exactly), `base` and `scale` fp32 as stored
        Ok(MhcSite { fn_buf: self.plain(&h.fn_)?, fn_ty: Ty::F16, base: self.plain(&h.base)?, scale: self.plain(&h.scale)? })
    }

    fn mlp(&self, m: &GatedMlp, limit: f32) -> Result<SharedExpert, ModelError> {
        SharedExpert::new(self.exl3(&m.gate_proj)?, self.exl3(&m.up_proj)?, self.exl3(&m.down_proj)?, limit)
            .map_err(|e| cfg_err(format!("gated mlp `{}`: {e}", m.gate_proj.prefix)))
    }

    fn experts(&self, bank: &ExpertBank) -> Result<BoundExperts, ModelError> {
        let region = self.regions.get(bank.region)?.subview(0, self.regions.get(bank.region)?.bytes());
        let off = |p: &super::weights::ProjLayout| ProjOffsets {
            trellis: p.trellis as usize,
            trellis_stride: p.trellis_stride as usize,
            suh: p.suh as usize,
            svh: p.svh as usize,
        };
        let err = |e: gb_cuda::CudaError| cfg_err(format!("layer {}'s experts: {e}", bank.layer));
        if bank.is_single() {
            // today's single bank, unchanged
            let lay = ExpertRegion {
                gate: off(&bank.layout.gate),
                up: off(&bank.layout.up),
                down: off(&bank.layout.down),
                bits_gu: (bank.bits_gate_up.twice() / 2) as usize,
                bits_d: (bank.bits_down.twice() / 2) as usize,
                codebook: bank.codebook,
            };
            return MoeExperts::from_region(region, &lay, self.moe).map(|e| BoundExperts::Single(Box::new(e))).map_err(err);
        }
        if bank.subbanks.is_empty() {
            return Err(cfg_err(format!("layer {}: this rank holds none of its routed experts", bank.layer)));
        }
        let mut banks = Vec::with_capacity(bank.subbanks.len());
        for sb in &bank.subbanks {
            let lay =
                SubbankRegion { gate: off(&sb.layout.gate), up: off(&sb.layout.up), down: off(&sb.layout.down), key: sb.key };
            let own = ExpertOwnership::from_ids(self.moe.experts, sb.ids.clone())
                .map_err(|e| cfg_err(format!("layer {} subbank {:?}: {e}", bank.layer, sb.key)))?;
            let sub = region.subview(0, region.bytes());
            banks.push(
                match bank.width {
                    // a width share: the subbank holds its experts' columns at the share's width, run with the whole width's plans
                    Some(share) => MoeExperts::from_region_width(sub, &lay, self.moe, share, own),
                    None => MoeExperts::from_region_subbank(sub, &lay, self.moe, own),
                }
                .map_err(err)?,
            );
        }
        match bank.width {
            Some(share) => MoeSubbanks::new_width(self.moe, share, banks),
            None => MoeSubbanks::new(self.moe, bank.own.clone(), banks),
        }
        .map(BoundExperts::Subbanks)
        .map_err(err)
    }

    /// One MLA layer's tensors (`index`: the layer's checkpoint index, for the messages).
    fn mla(&self, m: &super::weights::MlaWeights, index: usize) -> Result<MlaDevice, ModelError> {
        let ix = m.indexer.as_ref().ok_or_else(|| cfg_err(format!("layer {index}: a shared indexer is not built")))?;
        let kp = ix.kpool.as_ref().ok_or_else(|| cfg_err(format!("layer {index}: the indexer has no k-pool tensors")))?;
        // `kv_b` is the packed copy: `uk` first (64 x 256 x 512 fp16), then `uvt` (64 x 512 x 256 fp16)
        let kvb = self.plain(&m.kv_b_proj)?;
        let half = kvb.bytes() / 2;
        Ok(MlaDevice {
            q_a: self.exl3(&m.q_a_proj)?,
            q_b: self.exl3(&m.q_b_proj)?,
            kv_a: self.exl3(&m.kv_a_proj_with_mqa)?,
            o: self.exl3(&m.o_proj)?,
            q_a_norm: self.plain(&m.q_a_layernorm)?,
            kv_a_norm: self.plain(&m.kv_a_layernorm)?,
            kv_b: mla::MlaWeights { uk: kvb.subview(0, half), uvt: kvb.subview(half, half), dtype: Dtype::F16 },
            indexer: IndexerDevice {
                wq_b: match &ix.wq_b {
                    IndexerWqB::Exl3(g) => WqBDevice::Exl3(self.exl3(g)?),
                    IndexerWqB::PlainF16(t) => WqBDevice::F16 { w: self.plain(t)?, k: t.shape[1], n: t.shape[0] },
                },
                wk: self.plain(&ix.wk)?,
                weights_proj: self.plain(&ix.weights_proj)?,
                compress_gate: self.plain(&kp.compress_gate)?,
                dsa: dsa::IndexerWeights {
                    ln_w: self.plain(&ix.k_norm_weight)?,
                    ln_b: self.plain(&ix.k_norm_bias)?,
                    ape: self.plain(&kp.compress_ape)?,
                    eps: dsa::LN_EPS,
                },
            },
        })
    }

    /// One MoE layer's router, routed experts and shared expert.
    fn moe(&self, m: &super::weights::MoeWeights, index: usize, limit: f32) -> Result<FfnDevice, ModelError> {
        let shared = m.shared.as_ref().ok_or_else(|| cfg_err(format!("layer {index} has no shared expert")))?;
        let router = MoeRouter::from_device(
            self.plain(&m.router)?,
            self.plain(&m.e_score_correction_bias)?,
            self.moe.hidden,
            self.moe.topk,
            self.moe.scale,
        )
        .map_err(|e| cfg_err(format!("layer {index}'s router: {e}")))?;
        let shared = Box::new(self.mlp(shared, limit)?);
        Ok(match self.experts(&m.experts)? {
            BoundExperts::Single(ex) => FfnDevice::Moe {
                layer: Box::new(MoeLayer::new(router, *ex).map_err(|e| cfg_err(format!("layer {index}: {e}")))?),
                shared,
            },
            BoundExperts::Subbanks(sb) => FfnDevice::MoeSubbanks {
                layer: Box::new(MoeSubbankLayer::new(router, sb).map_err(|e| cfg_err(format!("layer {index}: {e}")))?),
                shared,
            },
        })
    }

    fn layer(&self, l: &GlmLayerWeights, limit: f32) -> Result<LayerDevice, ModelError> {
        let mhc = l.mhc.as_ref().ok_or_else(|| cfg_err(format!("layer {} has no mHC weights", l.index)))?;
        let mixer = match &l.mixer {
            MixerWeights::Kda(k) => MixerDevice::Kda(Box::new(KdaDevice {
                qkv: self.exl3(&k.qkv_proj)?,
                qkv_cut: k.qkv_proj.cut.clone(),
                o: self.exl3(&k.o_proj)?,
                b: self.plain(&k.b_proj)?,
                f_a: self.plain(&k.f_a_proj)?,
                g_a: self.plain(&k.g_a_proj)?,
                f_b: self.plain(&k.f_b_proj)?,
                g_b: self.plain(&k.g_b_proj)?,
                conv_w: self.plain(&k.conv1d_weight)?,
                a_log: self.plain(&k.a_log)?,
                dt_bias: self.plain(&k.dt_bias)?,
                o_norm: self.plain(&k.o_norm)?,
            })),
            MixerWeights::Mla(m) => MixerDevice::Mla(Box::new(self.mla(m, l.index)?)),
        };
        let ffn = match &l.ffn {
            FfnWeights::Dense(m) => FfnDevice::Dense(Box::new(self.mlp(m, limit)?)),
            FfnWeights::Moe(m) => self.moe(m, l.index, limit)?,
        };
        Ok(LayerDevice {
            index: l.index,
            input_norm: self.plain(&l.input_layernorm)?,
            post_norm: self.plain(&l.post_attention_layernorm)?,
            attn_site: self.site(&mhc.attn)?,
            ffn_site: self.site(&mhc.ffn)?,
            mixer,
            ffn,
        })
    }

    /// The NextN layer: a plain-residual MLA + MoE block with its input fusion and head norm.
    fn mtp(&self, m: &super::weights::GlmMtpWeights, limit: f32) -> Result<MtpDevice, ModelError> {
        let l = &m.layer;
        if l.mhc.is_some() {
            return Err(cfg_err(format!(
                "layer {} (the NextN layer) must be a plain-residual block, but has mHC tensors",
                l.index
            )));
        }
        let MixerWeights::Mla(mla_w) = &l.mixer else {
            return Err(cfg_err(format!("layer {} (the NextN layer) has no MLA mixer", l.index)));
        };
        let FfnWeights::Moe(moe) = &l.ffn else {
            return Err(cfg_err(format!("layer {} (the NextN layer) has no MoE block", l.index)));
        };
        let (moe, shared) = match self.moe(moe, l.index, limit)? {
            FfnDevice::Moe { layer, shared } => (MtpMoe::Single(layer), shared),
            // every expert (a mixed-width layer on one node) or a rank's share (mapped with an ownership): the drafter runs the
            // former; it refuses the latter by name (a share needs the exchange, and the drafter is not run on a rank yet)
            FfnDevice::MoeSubbanks { layer, shared } => (MtpMoe::Subbanks(layer), shared),
            FfnDevice::Dense(_) => return Err(cfg_err(format!("layer {} (the NextN layer) has no MoE block", l.index))),
        };
        Ok(MtpDevice {
            eh_proj: self.exl3(&m.eh_proj)?,
            enorm: self.plain(&m.enorm)?,
            hnorm: self.plain(&m.hnorm)?,
            shared_head_norm: self.plain(&m.shared_head_norm)?,
            input_norm: self.plain(&l.input_layernorm)?,
            post_norm: self.plain(&l.post_attention_layernorm)?,
            mixer: Box::new(self.mla(mla_w, l.index)?),
            moe,
            shared,
        })
    }
}

/// Bind the first `layers` layers (and the embedding, final norm and head) of `w` to views of `regions`. The regions must hold at least
/// what those layers need (the plan the loader executed was `plan.select_regions(keep_prefix(layers))`).
pub fn bind(w: &GlmWeights, cfg: &GlmConfig, regions: Regions, layers: usize) -> Result<GlmDevice, ModelError> {
    bind_with(w, cfg, regions, layers, false)
}

/// [`bind`], and with `with_mtp` the NextN layer too (its regions must have been loaded: `keep_prefix_mtp`).
pub fn bind_with(
    w: &GlmWeights,
    cfg: &GlmConfig,
    regions: Regions,
    layers: usize,
    with_mtp: bool,
) -> Result<GlmDevice, ModelError> {
    if layers == 0 || layers > w.layers.len() {
        return Err(cfg_err(format!("a model of {layers} layers from a pack of {}", w.layers.len())));
    }
    if with_mtp && w.mtp.is_empty() {
        return Err(cfg_err("the NextN layer was asked for but the pack has none".into()));
    }
    let moe = MoeDims {
        hidden: cfg.hidden,
        inter: cfg.ffn.moe_intermediate,
        experts: cfg.ffn.routed_experts,
        topk: cfg.ffn.experts_per_token,
        scale: cfg.ffn.routed_scaling_factor as f32,
        limit: cfg.ffn.swiglu_limit as f32,
    };
    let (embed, final_norm, lm_head, out, mtp) = {
        let b = Binder { regions: &regions, moe };
        let embed = b.plain(&w.embed_tokens)?;
        let final_norm = b.plain(&w.final_norm)?;
        let lm_head = b.exl3(&w.lm_head)?;
        let mut out = Vec::with_capacity(layers);
        for l in &w.layers[..layers] {
            out.push(b.layer(l, moe.limit)?);
        }
        let mtp = if with_mtp { Some(b.mtp(&w.mtp[0], moe.limit)?) } else { None };
        (embed, final_norm, lm_head, out, mtp)
    };
    Ok(GlmDevice { embed, final_norm, lm_head, layers: out, mtp, regions })
}
