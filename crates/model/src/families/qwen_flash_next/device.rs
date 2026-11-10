//! The Flash-Next weights on the device, claimed slot by slot from the pack's [`WeightMap`].
//!
//! [`FlashNextWeights::load`] binds the text recipe with [`WeightMap::from_dir`] (a pack that disagrees with its own
//! config is refused by name before a byte is read), stages every bound slot on the device honouring its
//! [`Transform`], and then claims those buffers by checkpoint name into the typed structs the forward runs on. Nothing
//! is loaded by convention: a slot the recipe wants and the map does not hold is an error naming it, and a slot the
//! map holds and no struct claims is reported too, so a pack cannot quietly carry a tensor the forward ignores.
//!
//! Slots the shard index lists go through [`LoadCtx`] (the pipelined loader); the two sidecar files the scanner does
//! not index (the n-gram table and the MTP input mixer patch) are read from the slot's `source` extent, which is what
//! [`WeightMap::from_dir`] recorded for them.

use super::config::FlashNextConfig;
use super::ngram::NgramTable;
use super::weights::{Kind, Slot, Transform, WeightMap};
use crate::error::ModelError;
use crate::loadctx::LoadCtx;
use crate::pack::{Pack, PackError};
use gb_cuda::exl3_linear::Exl3Weights;
use gb_cuda::fn_mhc::Mixer;
use gb_cuda::fn_moe::{ktab_k2, ExpertTables};
use gb_cuda::DevBuf;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// Prefix of every trunk tensor of the text model.
pub const LM: &str = "model.language_model";

/// The head collapse mixer (no inject rows).
pub struct HeadHcWeights {
    pub norm: DevBuf,
    pub down: DevBuf,
    pub up: DevBuf,
}

/// One layer's hyper-connection pre-mix, with the inject weights the post-inject reads.
pub struct HcWeights {
    pub norm: DevBuf,
    pub down: DevBuf,
    pub up: DevBuf,
    pub inject: DevBuf,
}

impl HeadHcWeights {
    pub fn mixer(&self) -> Mixer<'_> {
        Mixer { norm: &self.norm, down: &self.down, up: &self.up, inject: None }
    }
    fn bytes(&self) -> usize {
        self.norm.bytes() + self.down.bytes() + self.up.bytes()
    }
}

impl HcWeights {
    pub fn mixer(&self) -> Mixer<'_> {
        Mixer { norm: &self.norm, down: &self.down, up: &self.up, inject: Some(&self.inject) }
    }
    fn bytes(&self) -> usize {
        self.norm.bytes() + self.down.bytes() + self.up.bytes() + self.inject.bytes()
    }
}

/// A gated delta net mixer.
pub struct LinearAttnWeights {
    pub in_proj_qkv: Exl3Weights,
    pub in_proj_z: Exl3Weights,
    pub in_proj_a: DevBuf,
    pub in_proj_b: DevBuf,
    pub conv_w: DevBuf,
    pub a_log: DevBuf,
    pub dt_bias: DevBuf,
    pub norm: DevBuf,
    pub out_proj: Exl3Weights,
}

/// A grouped-query attention mixer with its QSA indexer.
pub struct FullAttnWeights {
    pub q_proj: Exl3Weights,
    pub k_proj: Exl3Weights,
    pub v_proj: Exl3Weights,
    pub o_proj: Exl3Weights,
    pub q_norm: DevBuf,
    pub k_norm: DevBuf,
    /// Every attention layer of a supported pack has one; the model decides whether it runs (`REXL3_QSA=0` attends
    /// densely at any context).
    pub indexer: Option<IndexerWeights>,
}

/// The QSA indexer of one attention layer: the projection to 4 query heads and 1 raw key, and the two norms (f32, 1 + w).
pub struct IndexerWeights {
    pub qk_proj: Exl3Weights,
    pub q_norm: DevBuf,
    pub k_norm: DevBuf,
}

pub enum MixerWeights {
    Linear(Box<LinearAttnWeights>),
    Full(Box<FullAttnWeights>),
}

pub struct ExpertWeights {
    pub gate: Exl3Weights,
    pub up: Exl3Weights,
    pub down: Exl3Weights,
}

impl ExpertWeights {
    fn bytes(&self) -> usize {
        self.gate.weight_bytes() + self.up.weight_bytes() + self.down.weight_bytes()
    }
}

/// One layer's MoE: the router, the shared expert's sigmoid gate, the shared expert, the routed experts and the device
/// pointer tables the MoE kernels index by expert id (the shared expert is slot `experts`).
pub struct MoeWeights {
    pub router: DevBuf,
    pub shared_gate: DevBuf,
    pub shared_expert: Box<ExpertWeights>,
    pub experts: Vec<ExpertWeights>,
    pub g_trellis_ptrs: DevBuf,
    pub g_suh_ptrs: DevBuf,
    pub g_svh_ptrs: DevBuf,
    pub u_trellis_ptrs: DevBuf,
    pub u_suh_ptrs: DevBuf,
    pub u_svh_ptrs: DevBuf,
    pub d_trellis_ptrs: DevBuf,
    pub d_suh_ptrs: DevBuf,
    pub d_svh_ptrs: DevBuf,
    pub k_tab: DevBuf,
}

impl MoeWeights {
    pub fn tables(&self) -> ExpertTables<'_> {
        ExpertTables {
            g_t: &self.g_trellis_ptrs,
            g_su: &self.g_suh_ptrs,
            g_sv: &self.g_svh_ptrs,
            u_t: &self.u_trellis_ptrs,
            u_su: &self.u_suh_ptrs,
            u_sv: &self.u_svh_ptrs,
            d_t: &self.d_trellis_ptrs,
            d_su: &self.d_suh_ptrs,
            d_sv: &self.d_svh_ptrs,
            ktab: &self.k_tab,
        }
    }
    fn table_bytes(&self) -> usize {
        9 * self.g_trellis_ptrs.bytes() + self.k_tab.bytes()
    }
}

pub struct FlashNextLayer {
    pub attn_hc: HcWeights,
    pub mixer: MixerWeights,
    pub mlp_hc: HcWeights,
    pub moe: MoeWeights,
}

/// The multi-token-prediction draft head: the trunk's pre-collapse stream stack and the next token's embedding are
/// normed, projected and summed into a stream stack, run through one attention decoder layer, collapsed by the head's
/// own mixer and read out through the trunk's LM head.
pub struct MtpWeights {
    /// f16 `[hc * h]`, applied as `1 + w`.
    pub norm_hidden: DevBuf,
    /// f32 `[h]`, already `1 + w`.
    pub norm_emb: DevBuf,
    pub fc_hidden: Exl3Weights,
    pub fc_embedding: Exl3Weights,
    /// The head's own collapse mixer; `None` for a pack without one, where the trunk's is used.
    pub mixer: Option<HeadHcWeights>,
    pub layer: FlashNextLayer,
}

/// The PLE (hashed n-gram embedding) stage, which runs before one decoder layer.
pub struct PleWeights {
    pub key_proj: DevBuf,
    pub value_proj: DevBuf,
    pub norm_key: DevBuf,
    pub norm_query: DevBuf,
    pub norm_conv: DevBuf,
    pub conv_w: DevBuf,
    pub head_bias: DevBuf,
    pub table: Arc<NgramTable>,
    /// Index of the decoder layer the stage runs before.
    pub before_layer: usize,
}

pub struct FlashNextWeights {
    pub embed: DevBuf,
    pub layers: Vec<FlashNextLayer>,
    pub head_hc: HeadHcWeights,
    pub lm_head: Exl3Weights,
    pub mtp: Option<MtpWeights>,
    /// `None` when the config has no PLE layer, or with `REXL3_PLE=0`.
    pub ple: Option<PleWeights>,
    /// Device bytes of every weight buffer above.
    pub bytes: usize,
}

fn io_err(e: std::io::Error) -> ModelError {
    ModelError::Pack(PackError::Io(e.to_string()))
}

/// An on-by-default switch: only `0` turns it off.
fn env_on(key: &str) -> bool {
    std::env::var(key).map(|v| v != "0").unwrap_or(true)
}

/// The bytes of one sidecar slot, read from the extent [`WeightMap::from_dir`] recorded for it.
fn source_bytes(dir: &Path, slot: &Slot) -> Result<Vec<u8>, ModelError> {
    use std::io::{Read, Seek, SeekFrom};
    let src =
        slot.source.as_ref().ok_or_else(|| ModelError::Config(format!("{}: the bound slot has no source extent", slot.name)))?;
    let path = dir.join(&src.file);
    let mut f = std::fs::File::open(&path).map_err(io_err)?;
    f.seek(SeekFrom::Start(src.offset)).map_err(io_err)?;
    let n = usize::try_from(src.bytes).map_err(|_| ModelError::Config(format!("{}: extent too large", slot.name)))?;
    let mut v = vec![0u8; n];
    f.read_exact(&mut v).map_err(io_err)?;
    Ok(v)
}

/// The device addresses of one EXL3 linear's trellis, input scales and output scales, which the MoE kernels read out of
/// a per-expert pointer table. [`Exl3Weights::buffers`] names them in that fixed order.
fn ptrs_of(w: &Exl3Weights, what: &str) -> Result<[u64; 3], ModelError> {
    let b = w.buffers();
    match (b.first(), b.get(1), b.get(2)) {
        (Some(("trellis", t)), Some(("suh", su)), Some(("svh", sv))) => Ok([t.ptr() as u64, su.ptr() as u64, sv.ptr() as u64]),
        _ => Err(ModelError::Config(format!("{what}: an EXL3 linear without the three parts the MoE kernels read"))),
    }
}

/// [`Transform::Bf16ToF32`] / [`Transform::Bf16AddOneToF32`] on raw bf16 bits, exactly as `Pack::f32_from_bf16` does it.
fn widen(bytes: &[u8], add_one: bool) -> Vec<u8> {
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .flat_map(|b| (f32::from_bits((b as u32) << 16) + if add_one { 1.0 } else { 0.0 }).to_le_bytes())
        .collect()
}

/// [`Transform::Bf16ToF16`] on raw bf16 bits, exactly as `Pack::f16_from_bf16` does it.
fn narrow16(bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .flat_map(|b| gb_exl3::f16::f64_to_f16(f32::from_bits((b as u32) << 16) as f64).to_le_bytes())
        .collect()
}

/// Which bound slots this process will use. The MTP head and the n-gram stage can be switched off, and a slot that will
/// not be claimed must not be staged either: the loader's copies are declared now and run in [`LoadCtx::finish`], after
/// the claim, so dropping a staged buffer would leave a copy pointing at freed memory.
struct Wanted {
    mtp: bool,
    ple: bool,
}

impl Wanted {
    fn new(cfg: &FlashNextConfig) -> Self {
        let ple = !cfg.ple_layer_ids.is_empty() && env_on("REXL3_PLE");
        if !ple && !cfg.ple_layer_ids.is_empty() {
            eprintln!("rexl3 level=warn REXL3_PLE=0: the n-gram stage is off, so this is not the model's function");
        }
        Self { mtp: cfg.mtp_layers > 0 && env_on("REXL3_MTP"), ple }
    }
    fn wants(&self, name: &str) -> bool {
        if name.starts_with("mtp.") {
            return self.mtp;
        }
        if name.contains(".ple.") {
            return self.ple;
        }
        true
    }
}

/// A bound map's slots on the device, claimed by name.
struct Staged {
    plain: BTreeMap<String, DevBuf>,
    groups: BTreeMap<String, Exl3Weights>,
}

impl Staged {
    fn take(&mut self, name: &str) -> Result<DevBuf, ModelError> {
        self.plain.remove(name).ok_or_else(|| ModelError::Config(format!("{name}: the pack has no such weight")))
    }
    fn take_group(&mut self, prefix: &str) -> Result<Exl3Weights, ModelError> {
        self.groups.remove(prefix).ok_or_else(|| ModelError::Config(format!("{prefix}: the pack has no such EXL3 linear")))
    }
    fn has(&self, name: &str) -> bool {
        self.plain.contains_key(name)
    }
    fn has_group(&self, prefix: &str) -> bool {
        self.groups.contains_key(prefix)
    }
    /// Slots no struct claimed: a pack the forward would only partly run.
    fn leftovers(&self) -> Vec<String> {
        self.plain.keys().chain(self.groups.keys()).cloned().collect()
    }
}

/// The slots [`stage`] puts on the device, in source order: every `Plain` and `Exl3Part` slot of the load plan this
/// process wants. Sidecars are not among them: the n-gram shards and hash parameters stay on the host in
/// [`NgramTable`], and so does the n-gram `head_bias`, which the PLE claim uploads from the table's own copy.
fn upload_plan(map: &WeightMap, want: &Wanted) -> Vec<Slot> {
    map.load_plan().items.into_iter().filter(|s| want.wants(&s.name)).collect()
}

/// The plain weights of the PLE stage, after `<layer>.ple.`; its `head_bias` is a sidecar (see [`ple_head_bias`]).
const PLE_SLOTS: [&str; 6] =
    ["key_proj.weight", "value_proj.weight", "norm_key.weight", "norm_query.weight", "norm_conv.weight", "conv1d.weight"];

/// The n-gram `head_bias` (f16 `[heads, ROW]`) the PLE stage adds to every gathered row, as the table read it from the
/// sidecar file, checked against the stage width the config asks for.
fn ple_head_bias<'t>(table: &'t NgramTable, cfg: &FlashNextConfig) -> Result<&'t [u16], ModelError> {
    let width = table.heads * gb_cuda::ple::ROW;
    if width != cfg.ple_embed_dim {
        return Err(ModelError::Config(format!("n-gram table width {width} != ple_embed_dim {}", cfg.ple_embed_dim)));
    }
    if table.head_bias.len() != width {
        return Err(ModelError::Config(format!("n-gram head_bias holds {} values, the table {width}", table.head_bias.len())));
    }
    Ok(&table.head_bias)
}

/// Stage every `Plain` and `Exl3Part` slot of `map` on the device, in source order.
fn stage<'p>(ctx: &mut LoadCtx<'p>, pack: &'p Pack, dir: &Path, map: &WeightMap, want: &Wanted) -> Result<Staged, ModelError> {
    let mut out = Staged { plain: BTreeMap::new(), groups: BTreeMap::new() };
    for slot in upload_plan(map, want).iter() {
        match slot.kind {
            Kind::Plain => {
                let dtype = match slot.dtype.as_str() {
                    "BF16" => "BF16",
                    "F16" => "F16",
                    other => return Err(ModelError::Config(format!("{}: dtype {other} is not a weight", slot.name))),
                };
                let buf = if pack.ck.plain.contains_key(&slot.name) {
                    match slot.transform {
                        Transform::None => ctx.raw16(pack, &slot.name, dtype, &slot.shape)?,
                        Transform::Bf16ToF32 => ctx.f32_from_bf16(pack, &slot.name, &slot.shape, false)?,
                        Transform::Bf16AddOneToF32 => ctx.f32_from_bf16(pack, &slot.name, &slot.shape, true)?,
                        Transform::Bf16ToF16 => ctx.f16_from_bf16(pack, &slot.name, &slot.shape)?,
                    }
                } else {
                    // a sidecar the shard index does not list: its own extent, converted here
                    let raw = source_bytes(dir, slot)?;
                    let bytes = match slot.transform {
                        Transform::None => raw,
                        Transform::Bf16ToF32 => widen(&raw, false),
                        Transform::Bf16AddOneToF32 => widen(&raw, true),
                        Transform::Bf16ToF16 => narrow16(&raw),
                    };
                    let len = bytes.len();
                    ctx.computed(len, move || Ok(bytes.clone()))?
                };
                out.plain.insert(slot.name.clone(), buf);
            }
            Kind::Exl3Part => {
                let Some((prefix, _)) = slot.name.rsplit_once('.') else {
                    return Err(ModelError::Config(format!("{}: not an EXL3 part name", slot.name)));
                };
                if !out.groups.contains_key(prefix) {
                    if !map.groups.contains_key(prefix) {
                        return Err(ModelError::Config(format!("{prefix}: the bound map has no EXL3 group")));
                    }
                    let w = ctx.group(pack, prefix)?;
                    out.groups.insert(prefix.to_string(), w);
                }
            }
            Kind::Marker | Kind::Sidecar => {}
        }
    }
    Ok(out)
}

fn claim_hc(s: &mut Staged, prefix: &str) -> Result<HcWeights, ModelError> {
    Ok(HcWeights {
        norm: s.take(&format!("{prefix}.hc_norm.weight"))?,
        down: s.take(&format!("{prefix}.input_mix_weight_down.weight"))?,
        up: s.take(&format!("{prefix}.input_mix_weight_up.weight"))?,
        inject: s.take(&format!("{prefix}.block_inject_weight.weight"))?,
    })
}

fn claim_head_hc(s: &mut Staged, prefix: &str) -> Result<HeadHcWeights, ModelError> {
    Ok(HeadHcWeights {
        norm: s.take(&format!("{prefix}.hc_norm.weight"))?,
        down: s.take(&format!("{prefix}.input_mix_weight_down.weight"))?,
        up: s.take(&format!("{prefix}.input_mix_weight_up.weight"))?,
    })
}

fn claim_layer(
    s: &mut Staged,
    cfg: &FlashNextConfig,
    p: &str,
    full: bool,
    bytes: &mut usize,
) -> Result<FlashNextLayer, ModelError> {
    let mut b = 0usize;
    let attn_hc = claim_hc(s, &format!("{p}.attn_hyper_connection"))?;
    b += attn_hc.bytes();

    let mixer = if full {
        let ap = format!("{p}.self_attn");
        let q_proj = s.take_group(&format!("{ap}.q_proj"))?;
        let k_proj = s.take_group(&format!("{ap}.k_proj"))?;
        let v_proj = s.take_group(&format!("{ap}.v_proj"))?;
        let o_proj = s.take_group(&format!("{ap}.o_proj"))?;
        let q_norm = s.take(&format!("{ap}.q_norm.weight"))?;
        let k_norm = s.take(&format!("{ap}.k_norm.weight"))?;
        b += q_proj.weight_bytes() + k_proj.weight_bytes() + v_proj.weight_bytes() + o_proj.weight_bytes();
        b += q_norm.bytes() + k_norm.bytes();
        let ip = format!("{ap}.indexer");
        if !s.has_group(&format!("{ip}.index_qk_proj")) {
            return Err(ModelError::Config(format!("{ip}.index_qk_proj: the attention layer has no QSA indexer")));
        }
        let qk_proj = s.take_group(&format!("{ip}.index_qk_proj"))?;
        let iq = s.take(&format!("{ip}.q_layernorm.weight"))?;
        let ik = s.take(&format!("{ip}.k_layernorm.weight"))?;
        b += qk_proj.weight_bytes() + iq.bytes() + ik.bytes();
        let indexer = Some(IndexerWeights { qk_proj, q_norm: iq, k_norm: ik });
        MixerWeights::Full(Box::new(FullAttnWeights { q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, indexer }))
    } else {
        let ap = format!("{p}.linear_attn");
        let in_proj_qkv = s.take_group(&format!("{ap}.in_proj_qkv"))?;
        let in_proj_z = s.take_group(&format!("{ap}.in_proj_z"))?;
        let out_proj = s.take_group(&format!("{ap}.out_proj"))?;
        let in_proj_a = s.take(&format!("{ap}.in_proj_a.weight"))?;
        let in_proj_b = s.take(&format!("{ap}.in_proj_b.weight"))?;
        let conv_w = s.take(&format!("{ap}.conv1d.weight"))?;
        let a_log = s.take(&format!("{ap}.A_log"))?;
        let dt_bias = s.take(&format!("{ap}.dt_bias"))?;
        let norm = s.take(&format!("{ap}.norm.weight"))?;
        b += in_proj_qkv.weight_bytes() + in_proj_z.weight_bytes() + out_proj.weight_bytes();
        b += in_proj_a.bytes() + in_proj_b.bytes() + conv_w.bytes() + a_log.bytes() + dt_bias.bytes() + norm.bytes();
        MixerWeights::Linear(Box::new(LinearAttnWeights {
            in_proj_qkv,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            conv_w,
            a_log,
            dt_bias,
            norm,
            out_proj,
        }))
    };

    let mlp_hc = claim_hc(s, &format!("{p}.mlp_hyper_connection"))?;
    b += mlp_hc.bytes();

    let mp = format!("{p}.mlp");
    let router = s.take(&format!("{mp}.gate.weight"))?;
    let shared_gate = s.take(&format!("{mp}.shared_expert_gate.weight"))?;
    b += router.bytes() + shared_gate.bytes();
    let shared_expert = Box::new(ExpertWeights {
        gate: s.take_group(&format!("{mp}.shared_expert.gate_proj"))?,
        up: s.take_group(&format!("{mp}.shared_expert.up_proj"))?,
        down: s.take_group(&format!("{mp}.shared_expert.down_proj"))?,
    });
    b += shared_expert.bytes();
    let mut experts = Vec::with_capacity(cfg.experts);
    for e in 0..cfg.experts {
        let ep = format!("{mp}.experts.{e}");
        let w = ExpertWeights {
            gate: s.take_group(&format!("{ep}.gate_proj"))?,
            up: s.take_group(&format!("{ep}.up_proj"))?,
            down: s.take_group(&format!("{ep}.down_proj"))?,
        };
        b += w.bytes();
        experts.push(w);
    }

    let n = cfg.experts + 1;
    let moe = MoeWeights {
        router,
        shared_gate,
        shared_expert,
        experts,
        g_trellis_ptrs: DevBuf::zeroed(n * 8)?,
        g_suh_ptrs: DevBuf::zeroed(n * 8)?,
        g_svh_ptrs: DevBuf::zeroed(n * 8)?,
        u_trellis_ptrs: DevBuf::zeroed(n * 8)?,
        u_suh_ptrs: DevBuf::zeroed(n * 8)?,
        u_svh_ptrs: DevBuf::zeroed(n * 8)?,
        d_trellis_ptrs: DevBuf::zeroed(n * 8)?,
        d_suh_ptrs: DevBuf::zeroed(n * 8)?,
        d_svh_ptrs: DevBuf::zeroed(n * 8)?,
        k_tab: DevBuf::zeroed(n * 3 * 4)?,
    };
    b += moe.table_bytes();
    *bytes += b;
    Ok(FlashNextLayer { attn_hc, mixer, mlp_hc, moe })
}

impl FlashNextWeights {
    /// Bind, upload and claim the whole text model of the pack in `dir`. `progress(done, total)` counts trunk layers.
    /// When `warm` is set and the pack has a PLE stage, the n-gram table warm must complete before load returns:
    /// this is the readiness gate (the server cannot become ready before a successful warm).
    pub fn load(dir: &Path, cfg: &FlashNextConfig, warm: bool, progress: impl FnMut(usize, usize)) -> Result<Self, ModelError> {
        let map = WeightMap::from_dir(dir, cfg).map_err(|errs| {
            let head: Vec<&str> = errs.iter().take(8).map(String::as_str).collect();
            ModelError::Config(format!("Qwen3.8-Flash-Next pack: {} problems: {}", errs.len(), head.join("; ")))
        })?;
        let pack = Pack::open(dir)?;
        let want = Wanted::new(cfg);
        let mut ctx = LoadCtx::from_env();
        let mut staged = stage(&mut ctx, &pack, dir, &map, &want)?;
        let mut w = Self::claim(&mut staged, cfg, dir, &want, progress)?;
        // the loader's copies run here; every buffer claimed above holds its weight only afterwards
        ctx.finish()?;
        w.init_ptr_tables()?;
        if let Some(p) = &w.ple {
            super::ngram::warm_preflight(warm, Some(&p.table))
                .map_err(|e| ModelError::Config(format!("n-gram warm failed: {e}")))?;
        }
        Ok(w)
    }

    /// Claim every staged slot into the typed structs. Host work only: no copy has run yet.
    fn claim(
        s: &mut Staged,
        cfg: &FlashNextConfig,
        dir: &Path,
        want: &Wanted,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Self, ModelError> {
        let mut bytes = 0usize;
        let embed = s.take(&format!("{LM}.embed_tokens.weight"))?;
        bytes += embed.bytes();

        let mut layers = Vec::with_capacity(cfg.layers);
        for (i, t) in cfg.layer_types.iter().enumerate() {
            layers.push(claim_layer(s, cfg, &format!("{LM}.layers.{i}"), t == "full_attention", &mut bytes)?);
            progress(i + 1, cfg.layers);
        }

        let head_hc = claim_head_hc(s, &format!("{LM}.hyper_connection_mixer"))?;
        bytes += head_hc.bytes();
        let lm_head = s.take_group("lm_head")?;
        bytes += lm_head.weight_bytes();

        let mtp = if want.mtp {
            if cfg.mtp_layers != 1 {
                return Err(ModelError::Config(format!("{} MTP layers (the head runs one)", cfg.mtp_layers)));
            }
            let norm_hidden = s.take("mtp.pre_fc_norm_hidden.weight")?;
            let norm_emb = s.take("mtp.pre_fc_norm_embedding.weight")?;
            let fc_hidden = s.take_group("mtp.fc_hidden")?;
            let fc_embedding = s.take_group("mtp.fc_embedding")?;
            bytes += norm_hidden.bytes() + norm_emb.bytes() + fc_hidden.weight_bytes() + fc_embedding.weight_bytes();
            let mixer = if s.has("mtp.hyper_connection_mixer.hc_norm.weight") {
                let m = claim_head_hc(s, "mtp.hyper_connection_mixer")?;
                bytes += m.bytes();
                Some(m)
            } else {
                None
            };
            let layer = claim_layer(s, cfg, "mtp.layers.0", true, &mut bytes)?;
            Some(MtpWeights { norm_hidden, norm_emb, fc_hidden, fc_embedding, mixer, layer })
        } else {
            None
        };

        let ple = match cfg.ple_layer_ids.as_slice() {
            _ if !want.ple => None,
            [] => None,
            [id] if *id >= 1 => {
                let li = id - 1;
                let pp = format!("{LM}.layers.{li}.ple");
                let mut take = |i: usize| s.take(&format!("{pp}.{}", PLE_SLOTS[i]));
                let (key_proj, value_proj, norm_key) = (take(0)?, take(1)?, take(2)?);
                let (norm_query, norm_conv, conv_w) = (take(3)?, take(4)?, take(5)?);
                // the bias is a sidecar the loader never stages: upload the table's own copy of it
                let table = NgramTable::open(dir, cfg).map_err(ModelError::Config)?;
                let head_bias = DevBuf::from_slice(ple_head_bias(&table, cfg)?)?;
                bytes += key_proj.bytes()
                    + value_proj.bytes()
                    + norm_key.bytes()
                    + norm_query.bytes()
                    + norm_conv.bytes()
                    + conv_w.bytes()
                    + head_bias.bytes();
                Some(PleWeights {
                    key_proj,
                    value_proj,
                    norm_key,
                    norm_query,
                    norm_conv,
                    conv_w,
                    head_bias,
                    table: Arc::new(table),
                    before_layer: li,
                })
            }
            ids => return Err(ModelError::Config(format!("a PLE stage at layers {ids:?} is not supported (one is)"))),
        };

        // a weight the map bound and nothing above claimed would be one the forward silently ignores
        let left = s.leftovers();
        if !left.is_empty() {
            let head: Vec<&str> = left.iter().take(8).map(String::as_str).collect();
            return Err(ModelError::Config(format!(
                "{} bound weights no Flash-Next slot claims: {}",
                left.len(),
                head.join("; ")
            )));
        }
        Ok(Self { embed, layers, head_hc, lm_head, mtp, ple, bytes })
    }

    /// Fill the per-layer expert pointer and bit-width tables the MoE kernels index by expert id (the shared expert is
    /// the entry past the routed ones).
    pub fn init_ptr_tables(&mut self) -> Result<(), ModelError> {
        let mtp_layer = self.mtp.as_mut().map(|m| &mut m.layer);
        for layer in self.layers.iter_mut().chain(mtp_layer) {
            let m = &mut layer.moe;
            let n = m.experts.len() + 1;
            let mut t: Vec<Vec<u64>> = (0..9).map(|_| Vec::with_capacity(n)).collect();
            let mut kt: Vec<i32> = Vec::with_capacity(n * 3);
            for (slot, e) in m.experts.iter().chain(std::iter::once(&*m.shared_expert)).enumerate() {
                for (j, w) in [&e.gate, &e.up, &e.down].into_iter().enumerate() {
                    let [tr, su, sv] = ptrs_of(w, &format!("expert slot {slot} projection {j}"))?;
                    t[3 * j].push(tr);
                    t[3 * j + 1].push(su);
                    t[3 * j + 2].push(sv);
                    kt.push(ktab_k2(w.bits));
                }
            }
            for (buf, v) in [
                (&mut m.g_trellis_ptrs, 0),
                (&mut m.g_suh_ptrs, 1),
                (&mut m.g_svh_ptrs, 2),
                (&mut m.u_trellis_ptrs, 3),
                (&mut m.u_suh_ptrs, 4),
                (&mut m.u_svh_ptrs, 5),
                (&mut m.d_trellis_ptrs, 6),
                (&mut m.d_suh_ptrs, 7),
                (&mut m.d_svh_ptrs, 8),
            ] {
                buf.upload(&t[v])?;
            }
            m.k_tab.upload(&kt)?;
        }
        Ok(())
    }

    /// Whether the attention layers run their QSA indexer (the pooled planes are allocated only then).
    pub fn has_qsa(&self) -> bool {
        self.layers
            .iter()
            .chain(self.mtp.as_ref().map(|m| &m.layer))
            .any(|l| matches!(&l.mixer, MixerWeights::Full(m) if m.indexer.is_some()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bf16_norm_widens_to_f32_the_way_the_pack_reader_does() {
        // 1.0 and -2.0 as bf16, with and without the RMSNorm `+1`
        let raw: Vec<u8> = [0x3f80u16, 0xc000].iter().flat_map(|w| w.to_le_bytes()).collect();
        let plain = widen(&raw, false);
        let shifted = widen(&raw, true);
        let f32s = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
        assert_eq!(f32s(&plain), vec![1.0, -2.0]);
        assert_eq!(f32s(&shifted), vec![2.0, -1.0]);
    }

    use crate::families::qwen_flash_next::config::tests::FLASH_NEXT_CONFIG;
    use crate::families::qwen_flash_next::inventory;

    const TABLE: &str = include_str!("../../../../../testdata/qwen_flash_next/tensor_inventory.tsv");

    fn cfg() -> FlashNextConfig {
        FlashNextConfig::from_json(&serde_json::from_str(FLASH_NEXT_CONFIG).unwrap()).unwrap()
    }

    fn ple_prefix(cfg: &FlashNextConfig) -> String {
        format!("{LM}.layers.{}.ple", cfg.ple_layer_ids[0] - 1)
    }

    /// What the executor stages for the committed pack: never a sidecar (so never the n-gram `head_bias`, which the PLE
    /// claim takes from the table instead), every plain weight in a dtype the stager converts, every EXL3 part under a
    /// bound group, every PLE weight the claim takes; and with the MTP head and the n-gram stage off, none of theirs, so
    /// switching them off cannot leave a staged slot nothing claims.
    #[test]
    fn the_upload_plan_never_stages_a_sidecar_and_follows_the_switches() {
        let cfg = cfg();
        let map = WeightMap::bind(&cfg, &inventory::expand_table(TABLE).unwrap()).unwrap();
        let bias = format!("{}.ple_embedding.ngram_embedding.head_bias", ple_prefix(&cfg));
        assert_eq!(map.slots[&bias].kind, Kind::Sidecar);
        assert!(map.load_plan().sidecars.iter().any(|s| s.name == bias));

        let all = upload_plan(&map, &Wanted { mtp: true, ple: true });
        assert!(all.iter().all(|s| s.name != bias), "the n-gram head_bias must not be staged");
        for s in &all {
            match s.kind {
                Kind::Plain => assert!(matches!(s.dtype.as_str(), "BF16" | "F16"), "{}: {}", s.name, s.dtype),
                Kind::Exl3Part => {
                    let (prefix, _) = s.name.rsplit_once('.').unwrap();
                    assert!(map.groups.contains_key(prefix), "{}: no bound group", s.name);
                }
                Kind::Marker | Kind::Sidecar => panic!("{}: a {:?} slot in the upload plan", s.name, s.kind),
            }
        }
        let names: std::collections::BTreeSet<&str> = all.iter().map(|s| s.name.as_str()).collect();
        for n in PLE_SLOTS {
            let full = format!("{}.{n}", ple_prefix(&cfg));
            assert!(names.contains(full.as_str()), "{full}: the PLE claim takes it, the plan does not stage it");
        }
        assert!(names.iter().any(|n| n.starts_with("mtp.")));

        let off = upload_plan(&map, &Wanted { mtp: false, ple: false });
        assert!(off.iter().all(|s| !s.name.starts_with("mtp.") && !s.name.contains(".ple.")));
        let kept: std::collections::BTreeSet<&str> = off.iter().map(|s| s.name.as_str()).collect();
        assert!(kept.is_subset(&names));
        let dropped: Vec<&&str> = names.difference(&kept).collect();
        assert!(dropped.iter().all(|n| n.starts_with("mtp.") || n.contains(".ple.")), "{dropped:?}");
    }

    /// A tiny n-gram sidecar (one shard of 4 rows) with a known `head_bias`, in the layout `NgramTable::open` reads.
    fn write_ngram_sidecar(dir: &Path, cfg: &FlashNextConfig, bias: &[u16]) {
        let heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
        let (rows, words) = (4usize, inventory::ngram_row_words(1));
        let p = format!("{}.ple_embedding.ngram_embedding", ple_prefix(cfg));
        let i64s = |v: Vec<i64>| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let tensors: Vec<(String, &str, Vec<usize>, Vec<u8>)> = vec![
            (format!("{p}.shard_0.trellis"), "I16", vec![rows, words], vec![0u8; rows * words * 2]),
            (format!("{p}.head_offsets"), "I64", vec![heads], i64s(vec![0; heads])),
            (format!("{p}.head_vocab_sizes"), "I64", vec![heads], i64s(vec![rows as i64; heads])),
            (format!("{p}.layer_multipliers"), "I64", vec![3], i64s(vec![1, 3, 5])),
            (
                format!("{p}.head_bias"),
                "F16",
                vec![heads, gb_cuda::ple::ROW],
                bias.iter().flat_map(|b| b.to_le_bytes()).collect(),
            ),
        ];
        let mut head = serde_json::Map::new();
        head.insert(
            "__metadata__".into(),
            serde_json::json!({"format": "exl3_ngram_trellis", "codebook": "mul1", "K": "1", "rows": rows.to_string(),
                               "shard_rows": rows.to_string(), "row_dim": gb_cuda::ple::ROW.to_string()}),
        );
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in &tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            head.insert(name.clone(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}));
        }
        let text = serde_json::Value::Object(head).to_string().into_bytes();
        let mut file = (text.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&text);
        file.extend_from_slice(&data);
        std::fs::write(dir.join(inventory::NGRAM_FILE), file).unwrap();
    }

    /// The PLE stage's `head_bias` is the one the sidecar file holds, read through the table (no device, no staging),
    /// and a table narrower or wider than the stage the config asks for is refused.
    #[test]
    fn the_ple_head_bias_is_read_from_the_ngram_sidecar() {
        let cfg = cfg();
        let heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
        let bias: Vec<u16> = (0..heads * gb_cuda::ple::ROW).map(|i| (i as u16).wrapping_mul(7).wrapping_add(1)).collect();
        let dir = std::env::temp_dir().join(format!("rexl3-fn-headbias-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_ngram_sidecar(&dir, &cfg, &bias);

        let table = NgramTable::open(&dir, &cfg).unwrap();
        assert_eq!(ple_head_bias(&table, &cfg).unwrap(), &bias[..]);
        let mut wide = cfg.clone();
        wide.ple_embed_dim += gb_cuda::ple::ROW;
        let e = ple_head_bias(&table, &wide).unwrap_err().to_string();
        assert!(e.contains("ple_embed_dim"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
