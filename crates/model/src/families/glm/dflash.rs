//! Booting the DFlash2 drafter of GLM-5.3-Flash (`--draft DIR` with `--spec auto|mtp`): the draft's config checked against the trunk,
//! what it will allocate (the memory plan holds it back), and its load onto a model (`model/dflash.rs` drafts with it).
//!
//! The drafter replaces the pack's NextN layer: with a draft directory the NextN layer is not loaded and its cache is not allocated
//! (`tp_boot::mtp_wanted`, `engine::build_engine`). On a tensor-parallel group it lives on rank 0 alone; the follower plans without it
//! (more free memory, so it plans at least rank 0's KV pages and loads rank 0's count).

use super::alloc::{self, TOKENIZER_VOCAB};
use super::config::GlmConfig;
use super::model::{DflashState, GlmModel};
use crate::draft::{Draft, DraftDims, BLOCK};
use crate::error::ModelError;
use crate::loadctx::LoadCtx;
use crate::pack::Pack;
use crate::serve::ServeOptions;
use crate::shapes::{read_draft_config, DraftConfig};
use std::path::PathBuf;

/// The drafter a boot will load: its directory, its config and the device bytes it will take.
#[derive(Clone, Debug)]
pub struct DflashPlan {
    pub dir: PathBuf,
    pub cfg: DraftConfig,
    pub dims: DraftDims,
    /// Weights, context rings, scratch and the trunk's tap buffer.
    pub bytes: u64,
}

/// Whether these options draft with DFlash2: a draft directory, drafting on and a `--spec` that drafts with a model (`auto`, `mtp`).
pub fn wanted(o: &ServeOptions) -> bool {
    o.drafting && o.spec.uses_mtp() && o.draft_dir.is_some()
}

/// Whether this process loads the drafter: [`wanted`], on one node or on rank 0 of a group.
pub fn loads_here(o: &ServeOptions) -> bool {
    wanted(o) && o.tp.as_ref().is_none_or(|t| t.rank == 0)
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
/// read and checked (by name, before any weight is read) and its bytes.
pub fn plan(o: &ServeOptions, cfg: &GlmConfig, slots: usize, chunk: usize) -> Result<Option<DflashPlan>, ModelError> {
    if !loads_here(o) {
        return Ok(None);
    }
    let dir = o.draft_dir.clone().expect("wanted() checked it");
    let dc = read_draft_config(&dir)?;
    dc.check_for(cfg.hidden, cfg.layers.len(), cfg.vocab)
        .map_err(|e| ModelError::Config(format!("--draft {}: {e}", dir.display())))?;
    let dims = dc.dims(TOKENIZER_VOCAB);
    let rows = chunk.max(alloc::WINDOW_ROWS);
    let bytes = Draft::planned_bf16_bytes(&dims, cfg.vocab, slots, chunk, BLOCK) as u64
        + DflashState::planned_tap_bytes(&dims, rows) as u64
        + restricted_head_bytes(cfg, slots)?;
    Ok(Some(DflashPlan { dir, cfg: dc, dims, bytes }))
}

/// An upper bound of the restricted head's bytes (`REXL3_MTP_VOCAB`, the NextN head's setting, which the drafter shares): its strips of the
/// `lm_head` at up to 8 bits a weight, its logits and its workspace.
fn restricted_head_bytes(cfg: &GlmConfig, slots: usize) -> Result<u64, ModelError> {
    let Some((strips, _)) = super::tp_boot::vocab_setting()? else { return Ok(0) };
    let cols: usize = super::model::vocab_ranges(strips, cfg.vocab).iter().map(|&(lo, hi)| (hi - lo) * 128).sum();
    let rows = slots.min(crate::draft::MAX_DRAFT_SEQS) * crate::draft::DRAFTS;
    Ok((cols * cfg.hidden + rows * cols * 2 + (64 << 20)) as u64)
}

/// Load the drafter of `p` and attach it to `m` (its weights quantized on the loader threads, its ring and scratch allocated), with the
/// restricted head of `REXL3_MTP_VOCAB` when it is set. Logs one `event=dflash_load` line.
pub fn load(m: &mut GlmModel, p: &DflashPlan) -> Result<(), ModelError> {
    let t = std::time::Instant::now();
    let pack = Pack::open(&p.dir).map_err(|e| ModelError::Config(format!("--draft {}: {e}", p.dir.display())))?;
    let mut ctx = LoadCtx::from_env();
    let (draft, bytes) = Draft::declare_bf16(&mut ctx, &pack, p.dims, m.cfg.vocab, m.slots, m.chunk_rows(), BLOCK)
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
        "rexl3 event=dflash_load dir={} gib={:.3} taps_mib={:.1} layers={:?} hidden={} intermediate={} head=\"{head}\" load_s={load_s:.2} total_s={:.2}",
        p.dir.display(),
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
