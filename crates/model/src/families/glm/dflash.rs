//! Booting the DFlash2 drafter of GLM-5.3-Flash (`--draft DIR` with `--spec auto|mtp`): the draft's config checked against the trunk,
//! what it will allocate (the memory plan holds it back), and its load onto a model (`model/dflash.rs` drafts with it).
//!
//! The drafter replaces the pack's NextN layer: with a draft directory the NextN layer is not loaded and its cache is not allocated
//! (`tp_boot::mtp_wanted`, `engine::build_engine`). On a tensor-parallel group of two it is SPLIT over both ranks by default
//! ([`share_of`], [`SPLIT_ENV`]): each rank plans and loads its half (`draft::DraftShare`) and the rounds run on both. With the split off
//! it lives on rank 0 alone; the follower plans without it (more free memory, so it plans at least rank 0's KV pages and loads rank 0's
//! count).

use super::alloc::{self, TOKENIZER_VOCAB};
use super::config::GlmConfig;
use super::model::{DflashState, GlmModel};
use crate::draft::{Draft, DraftDims, DraftShare, BLOCK};
use crate::error::ModelError;
use crate::loadctx::LoadCtx;
use crate::pack::Pack;
use crate::serve::ServeOptions;
use crate::shapes::{read_draft_config, DraftConfig};
use std::path::PathBuf;

/// The drafter a boot will load: its directory, its config, this rank's share of it and the device bytes it will take.
#[derive(Clone, Debug)]
pub struct DflashPlan {
    pub dir: PathBuf,
    pub cfg: DraftConfig,
    pub dims: DraftDims,
    /// This rank's half of a split drafter ([`share_of`]); `None`: the whole drafter.
    pub share: Option<DraftShare>,
    /// Weights, context rings, scratch and the trunk's tap buffer.
    pub bytes: u64,
}

/// Whether these options draft with DFlash2: a draft directory, drafting on and a `--spec` that drafts with a model (`auto`, `mtp`).
pub fn wanted(o: &ServeOptions) -> bool {
    o.drafting && o.spec.uses_mtp() && o.draft_dir.is_some()
}

/// `REXL3_GLM_DRAFT_SPLIT`: `0` / `off` keeps the DFlash2 drafter of a two-rank group whole on rank 0 (the layout before the split, for an
/// A/B); unset or anything else splits it over both ranks ([`share_of`]). A fact of the Hello (`GlmFacts::draft_split`): both ranks must
/// say the same.
pub const SPLIT_ENV: &str = "REXL3_GLM_DRAFT_SPLIT";

/// Whether [`SPLIT_ENV`] asks for the split.
pub fn split_on() -> bool {
    !matches!(std::env::var(SPLIT_ENV).as_deref().map(str::trim), Ok("0" | "off" | "false" | "no"))
}

/// This rank's share of a split drafter: [`wanted`], a sharded group of two ranks (the swaps are the group's communicator's) and
/// [`split_on`]; `None` otherwise (one node, the split off, or no drafter).
pub fn share_of(o: &ServeOptions) -> Result<Option<DraftShare>, ModelError> {
    let Some(t) = o.tp.as_ref().filter(|t| t.is_group() && t.ranks == 2) else { return Ok(None) };
    if !wanted(o) || !split_on() || !crate::families::nemotron_h_omni::tp_boot::sharded() {
        return Ok(None);
    }
    DraftShare::of(t.rank, t.ranks).map(Some)
}

/// Whether this process loads the drafter: [`wanted`], on one node, on rank 0 of a group, or on any rank of a split ([`share_of`]).
pub fn loads_here(o: &ServeOptions) -> Result<bool, ModelError> {
    Ok(wanted(o) && (o.tp.as_ref().is_none_or(|t| t.rank == 0) || share_of(o)?.is_some()))
}

/// The Hello fact of the split (`GlmFacts::draft_split`): `off` without one, else `on:` and the FNV-1a 64 of the draft pack's
/// `config.json`, so two ranks split the same drafter or are refused by name before any weight is read.
pub fn split_fact(o: &ServeOptions) -> Result<String, ModelError> {
    if share_of(o)?.is_none() {
        return Ok("off".into());
    }
    let path = o.draft_dir.as_ref().expect("share_of() checked it").join("config.json");
    let bytes = std::fs::read(&path).map_err(|e| ModelError::Config(format!("--draft: {}: {e}", path.display())))?;
    let h = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
    Ok(format!("on:{h:016x}"))
}

/// The layers a group's DFlash2 drafter taps (its config's `target_layer_ids`; empty without [`wanted`]), read on every rank from its
/// own options: a fact of the TP Hello (`GlmFacts::taps`), since under the sequence-parallel residual both ranks relay the taps
/// (`model::dflash::TapRelay`).
pub fn tap_layers(o: &ServeOptions) -> Result<Vec<usize>, ModelError> {
    if !wanted(o) {
        return Ok(Vec::new());
    }
    let dir = o.draft_dir.clone().expect("wanted() checked it");
    Ok(read_draft_config(&dir)?.target_layer_ids)
}

/// Bytes a rank of a two-rank P2 group holds back for the tap relay's stage (`model::dflash::TapRelay`) at prompt chunks of `chunk`
/// rows: on both ranks when the group drafts with DFlash2, 0 otherwise.
pub fn relay_bytes(o: &ServeOptions, cfg: &GlmConfig, chunk: usize) -> u64 {
    let two = o.tp.as_ref().is_some_and(|t| t.ranks == 2);
    if wanted(o) && two && super::contract::current().p2_prefill(true) {
        super::model::TapRelay::planned_bytes(chunk, cfg.hidden) as u64
    } else {
        0
    }
}

/// The drafter's plan for a trunk of `cfg` with `slots` slots and prompt chunks of `chunk` rows, when this process loads one: its config
/// read and checked (by name, before any weight is read), this rank's share and its bytes.
pub fn plan(o: &ServeOptions, cfg: &GlmConfig, slots: usize, chunk: usize) -> Result<Option<DflashPlan>, ModelError> {
    if !loads_here(o)? {
        return Ok(None);
    }
    let dir = o.draft_dir.clone().expect("wanted() checked it");
    let dc = read_draft_config(&dir)?;
    dc.check_for(cfg.hidden, cfg.layers.len(), cfg.vocab)
        .map_err(|e| ModelError::Config(format!("--draft {}: {e}", dir.display())))?;
    let dims = dc.dims(TOKENIZER_VOCAB);
    let share = share_of(o)?;
    let rows = chunk.max(alloc::WINDOW_ROWS);
    let bytes = Draft::planned_bf16_bytes(&dims, cfg.vocab, slots, chunk, BLOCK, share) as u64
        + DflashState::planned_tap_bytes(&dims, rows) as u64
        + restricted_head_bytes(cfg, slots, share)?;
    Ok(Some(DflashPlan { dir, cfg: dc, dims, share, bytes }))
}

/// An upper bound of the restricted head's bytes (`REXL3_MTP_VOCAB`, the NextN head's setting, which the drafter shares): its strips of the
/// `lm_head` at up to 8 bits a weight (a split drafter's: its share of them, at most half rounded up), its logits and its workspace.
fn restricted_head_bytes(cfg: &GlmConfig, slots: usize, share: Option<DraftShare>) -> Result<u64, ModelError> {
    let Some((strips, _)) = super::tp_boot::vocab_setting()? else { return Ok(0) };
    let all: usize = super::model::vocab_ranges(strips, cfg.vocab).iter().map(|&(lo, hi)| hi - lo).sum();
    let cols = share.map_or(all, |s| all.div_ceil(s.ranks)) * 128;
    let rows = slots.min(crate::draft::MAX_DRAFT_SEQS) * crate::draft::DRAFTS;
    Ok((cols * cfg.hidden + rows * cols * 2 + (64 << 20)) as u64)
}

/// Load the drafter of `p` and attach it to `m` (its weights quantized on the loader threads, its ring and scratch allocated), with the
/// restricted head of `REXL3_MTP_VOCAB` when it is set. Logs one `event=dflash_load` line.
pub fn load(m: &mut GlmModel, p: &DflashPlan) -> Result<(), ModelError> {
    let t = std::time::Instant::now();
    let pack = Pack::open(&p.dir).map_err(|e| ModelError::Config(format!("--draft {}: {e}", p.dir.display())))?;
    // the arena is exactly the drafter's weights: a fixed 1 GiB block would strand ~0.22 GiB of rank 0 for the process's life
    let mut ctx = LoadCtx::exact_from_env(Draft::bf16_weight_bytes(&p.dims, m.cfg.vocab, p.share));
    let (draft, bytes) = Draft::declare_bf16(&mut ctx, &pack, p.dims, m.cfg.vocab, m.slots, m.chunk_rows(), BLOCK, p.share)
        .map_err(|e| ModelError::Config(format!("--draft {}: {e}", p.dir.display())))?;
    ctx.finish()?;
    let load_s = t.elapsed().as_secs_f64();
    let tap_bytes = m.attach_dflash(draft, &p.cfg.target_layer_ids)?;
    let mut head = String::from("full");
    if let Some((strips, share)) = super::tp_boot::vocab_setting()? {
        let ranges: Vec<(usize, usize)> =
            super::model::vocab_ranges(strips, m.cfg.vocab).iter().map(|&(lo, hi)| (lo * 128, hi * 128)).collect();
        let added = m.dflash_restrict_vocab(&ranges, (share > 0.0).then_some(share))?;
        head = format!("strips={} share={share} extra_mib={:.0}", strips, added as f64 / 1048576.0);
    }
    m.sync()?;
    eprintln!(
        "rexl3 event=dflash_load dir={} share={} gib={:.3} taps_mib={:.1} layers={:?} hidden={} intermediate={} head=\"{head}\" load_s={load_s:.2} total_s={:.2}",
        p.dir.display(),
        p.share.map_or("whole".to_string(), |s| format!("{}/{}", s.rank, s.ranks)),
        bytes as f64 / (1u64 << 30) as f64,
        tap_bytes as f64 / 1048576.0,
        p.cfg.target_layer_ids,
        p.dims.hidden,
        p.dims.intermediate,
        t.elapsed().as_secs_f64()
    );
    gb_engine::boot::fact("drafter", "dflash2");
    Ok(())
}
