//! Build a text-only Flash-Next engine. Weight regions and the warm n-gram table are
//! budgeted before loading, because the model contract allocates KV and weights together.
use super::backend::FlashNextBackend;
use super::config::FlashNextConfig;
use super::model::{FlashNextModel, ModelOptions};
use super::ngram::NgramTable;
use super::prefix;
use super::weights::WeightMap;
use crate::error::ModelError;
use crate::hostmem::{self, MemInfo};
use crate::memory::MemoryOptions;
use crate::pack::Pack;
use crate::plan::{choose_slots, plan_memory_cached_windowed, MIN_TOKENS_PER_SLOT};
use crate::serve::{gib, PrefixCacheOption, ServeOptions};
use gb_engine::{boot, BudgetPolicy, Drafter, Engine, EngineConfig, NoDrafts};
use gb_exl3::checkpoint::{read_header_checked, scan};
use gb_sched::{CostModel, DraftCost, ModelMemory, RoundCost, StepCost};
use sha2::{Digest, Sha256};
use std::path::Path;

const WINDOW_ROWS: usize = super::memory::WINDOW_ROWS;
/// The verify step by rows of ONE stream, in milliseconds as the engine thread waits for it (`step_verified` wall: the window's
/// graphs, the commit it folds in, the greedy gather and the readback), for `EngineConfig::cost` (the `legacy` policy's table).
/// MEASURED on the head (`rexl3_round_rows_*` counters of the server's `/metrics`, nodes A and B, the decode screen's six prompts at
/// short context, 2026-11-10; K = 7 and 12 runs for the rows past 7): 28.6 / 32.4 / 35.9 / 39.2 / 42.7 / 46.1 / 49.8 / 53.0 / 57.4 /
/// 63.5 / 69.7 ms for 1..11 rows. The rows from 16 are the earlier hand-coded table's (not measured: the one-stream rounds stop at
/// 11, and `STEP_GRID` prices the streams' rows).
const STEP_MS: &[(usize, f64)] = &[
    (1, 28.6),
    (2, 32.4),
    (3, 35.9),
    (4, 39.2),
    (5, 42.7),
    (6, 46.1),
    (7, 49.8),
    (8, 53.0),
    (9, 57.4),
    (10, 63.5),
    (11, 69.7),
    (16, 90.0),
    (32, 130.0),
    (64, 220.0),
];
/// The verify step by (streams, total rows) in milliseconds as the engine thread waits for it, for the policies that price a round
/// (`RoundCost::verify`). MEASURED the same way (`rexl3_round_rows_*` of `/metrics`: the decode screen for one stream, the 2- and
/// 4-stream concurrency sweep for the rest, one GB10, five runs each over 4 policies), cells with at least 4 rounds; the
/// few cells with fewer (a stream's first rows, 3 streams past 10 rows) are read along their neighbours. A window past 8 rows costs
/// a step more (8 to 9 rows: +4.4 ms at one stream, +7 to +9 at two or more, the second 8-row tile of the decode kernels), then
/// about 3.5 ms a row at several streams (up to 6 at one).
const STEP_GRID: &[(usize, usize, f64)] = &[
    (1, 1, 28.6),
    (1, 2, 32.4),
    (1, 3, 35.9),
    (1, 4, 39.2),
    (1, 5, 42.7),
    (1, 6, 46.1),
    (1, 7, 49.8),
    (1, 8, 53.0),
    (1, 9, 57.4),
    (1, 10, 63.5),
    (1, 11, 69.7),
    (2, 2, 36.0),
    (2, 3, 40.0),
    (2, 4, 43.6),
    (2, 5, 48.0),
    (2, 6, 49.5),
    (2, 7, 53.4),
    (2, 8, 57.2),
    (2, 9, 64.3),
    (2, 10, 67.8),
    (3, 3, 41.5),
    (3, 4, 46.0),
    (3, 5, 50.4),
    (3, 6, 54.9),
    (3, 7, 57.3),
    (3, 8, 61.1),
    (3, 9, 69.8),
    (3, 10, 73.8),
    (3, 12, 78.6),
    (4, 4, 45.0),
    (4, 5, 50.0),
    (4, 6, 55.0),
    (4, 7, 60.0),
    (4, 8, 62.3),
    (4, 9, 70.4),
    (4, 10, 74.8),
    (4, 11, 77.6),
    (4, 12, 81.3),
    (4, 13, 83.9),
    (4, 14, 87.6),
    (4, 15, 90.5),
    (4, 17, 96.5),
    (4, 18, 100.2),
    (4, 19, 106.8),
];
/// A draft chain of `k` MTP steps for one stream costs `MTP_CHAIN_BASE_MS + 1.2 * k` ms in the draft phase (the catch-up rows, the
/// replay and the host read, then one head step each). MEASURED: draft-phase wall per round at fixed k = 1..7 (`REXL3_MTP_ADAPT=0
/// REXL3_MTP_CONF=0`, one GB10, 1,600 tokens over four prompts): 1.49 / 2.75 / 3.90 / 5.08 / 6.34 / 7.39 / 8.66 ms. A chain of several
/// streams runs every stream's row in each step, about 0.2 ms more a step for each stream past the first (2 streams, 4 asked: 5 to
/// 6 ms; 4 streams, 5 asked: 9 to 10 ms): the planner prices a draft by its streams (`layers`) and the picks asked in all
/// (`MTP_HEAD_MS`, 1.2 ms a pick to 6, concave past it: several streams share each step).
const MTP_CHAIN_BASE_MS: f64 = 0.28;
const MTP_HEAD_MS: &[(usize, f64)] = &[(1, 1.2), (2, 2.4), (3, 3.6), (4, 4.8), (5, 6.0), (6, 7.2), (8, 8.6), (12, 10.5), (16, 12.5), (32, 19.0)];
/// The budget policy a Flash-Next server runs unless `REXL3_SPEC_BUDGET` says otherwise: `scored` (the plan decides before the chain
/// whether a round drafts, the head's own probabilities decide after it how many drafts ride). Decode screen, six prompts, replies
/// exact: 65.2 to 65.7 tok/s against 64.5 for `legacy` with the old table.
const DEFAULT_BUDGET: BudgetPolicy = BudgetPolicy::Scored;

/// What a round costs, as the engine's budget policies read it (`Engine::with_budget`): the one-stream verify table above, and the
/// MTP chain as a draft whose head rows are the steps. Measured at a short context and not corrected for a longer one (the context
/// terms are 0): at the lengths the decode screen runs, the attention of 12 of the 48 layers is not what a window waits for.
pub(super) fn round_cost(mtp: bool) -> RoundCost {
    let layers: Vec<(usize, f64)> = (1..=4).map(|n| (n, MTP_CHAIN_BASE_MS + 0.2 * (n - 1) as f64)).collect();
    RoundCost {
        verify: StepCost::from_grid(STEP_GRID.to_vec()),
        tree_verify: None,
        draft: mtp.then(|| DraftCost::new(layers, MTP_HEAD_MS.to_vec(), 0.0)),
        kv_ms_per_token: 0.0,
        kv_pass_ms_per_token: 0.0,
        fixed_ms: 0.0,
    }
}

/// Budget from reclaimable MemAvailable, not the smaller CUDA MemFree. The model
/// has not been loaded yet, so its weight regions must be deducted before the KV plan
/// is sized. The warmed n-gram table is NOT deducted: it lives in reclaimable page cache,
/// which `MemAvailable` already counts as recoverable, so subtracting it again would
/// double-count it as an anonymous/device reservation (and shrink the KV pool below what
/// a CF 131072 context needs). The planner takes its own reserve and floor out separately
/// (once).
///
/// The bytes the exact MTP draft-head copy adds to the weight plan: the first `MTP_HEAD_N`
/// columns of the packed LM head, derived from `cfg.vocab` (the same domain `mtpk::pick`
/// reads, validated again at model load). A pack whose checkpoint LM head is missing or does
/// not cover the candidate domain is refused here, before any memory plan is sized.
fn draft_head_plan_bytes(
    groups: &std::collections::BTreeMap<String, super::weights::Exl3Group>,
    vocab: usize,
) -> Result<u64, ModelError> {
    let h = groups.get("lm_head").ok_or_else(|| ModelError::Config("missing packed LM head".into()))?;
    let nv = super::model::draft_head_nv(vocab);
    super::model::check_draft_head_shapes(nv, h.n)?;
    let bits = h.bits.twice() as usize / 2;
    Ok(super::model::draft_head_prefix_bytes(h.k, nv, bits) as u64)
}

/// Device memory left for the KV plan after the loaded weights are deducted from the
/// planning figure (used by the 262K plan before the pool is sized).
fn free_after_load(cuda_free: u64, host: Option<&MemInfo>, weights: u64) -> u64 {
    hostmem::planning_free(cuda_free, host).saturating_sub(weights)
}

/// The per-sequence drafter factory. Any drafting run installs a copy drafter per stream, so a round the backend
/// serves no native drafts for — every multi-request decode batch (the Flash-Next MTP head proposes for one sequence
/// at a time) and any pack without an MTP head — still speculates per stream. Native MTP drafts override it whenever
/// the backend proposes them (single-request rounds). Drafts never change what is emitted, only how many rows ride in
/// the verify window, so the fallback keeps the exactness contract.
fn make_drafter(drafting: bool) -> Box<dyn Fn() -> Box<dyn Drafter>> {
    if drafting {
        Box::new(|| Box::new(gb_spec::CopyDrafter::new(8, 7)))
    } else {
        Box::new(|| Box::new(NoDrafts))
    }
}

/// `REXL3_NGRAM_WARM`: unset or `auto` decides from headroom, `0` turns warming off, any
/// other value forces it on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarmMode {
    Off,
    Auto,
    On,
}

pub fn warm_mode_from_env(value: Option<&str>) -> WarmMode {
    match value {
        None | Some("auto") => WarmMode::Auto,
        Some("0") => WarmMode::Off,
        Some(_) => WarmMode::On,
    }
}

/// Whether the n-gram warm should run. `Auto` runs it only when the reclaimable headroom left
/// after the weights still holds the whole table plus the floor; on a tight box the kernel would
/// reclaim the warm pages under pressure anyway, so skipping avoids the churn. `On` ignores
/// headroom, `Off` never warms. With no host figure (an old kernel) the device's numbers plan
/// the pool, and auto warms.
pub fn wants_warm(mode: WarmMode, host: Option<&MemInfo>, weights: u64, table: u64, floor: u64) -> bool {
    match mode {
        WarmMode::Off => false,
        WarmMode::On => true,
        WarmMode::Auto => host.map_or(true, |h| h.available.saturating_sub(weights) >= table.saturating_add(floor)),
    }
}

fn warm_word(want: bool, mode: WarmMode) -> &'static str {
    if want {
        "on"
    } else if mode == WarmMode::Auto {
        "auto-skip"
    } else {
        "off"
    }
}

fn check_flash_options(o: &ServeOptions) -> Result<(), ModelError> {
    crate::serve::check_options(o)?;
    if o.window_rows != WINDOW_ROWS {
        return Err(ModelError::Config(format!("Flash-Next --window-rows must be {WINDOW_ROWS}")));
    }
    if o.max_model_len == 0 {
        return Err(ModelError::Config("--max-model-len must be at least 1".into()));
    }
    if o.vision.is_some() {
        return Err(ModelError::NotBuilt("Flash-Next images need vision_flash_next; start with --no-vision".into()));
    }
    Ok(())
}

/// The upstream canonical pack fingerprint covers indexed EXL3 groups and plain tensors.
/// Sidecars are outside the shard index, so fold their checked tensor descriptions
/// (including payload offsets) and file lengths without reading their payloads.
/// As with upstream's fingerprint, a payload edit with unchanged metadata is not detected.
fn pack_header_fingerprint(dir: &Path) -> Result<String, ModelError> {
    let ck = scan(dir, true).map_err(|e| ModelError::Config(format!("pack fingerprint: {e}")))?;
    let issues: Vec<_> = ck.issues.iter().filter(|issue| !issue.contains(" as the index says")).collect();
    if !issues.is_empty() {
        return Err(ModelError::Config(format!(
            "pack fingerprint: {}",
            issues.into_iter().map(String::as_str).collect::<Vec<_>>().join("; ")
        )));
    }
    let mut h = Sha256::new();
    h.update(b"rexl3-flash-next-pack-headers-v1\0");
    h.update(crate::fingerprint::fingerprint(&ck).as_bytes());
    for file in [super::inventory::NGRAM_FILE, super::inventory::MTP_MIXER_FILE]
        .into_iter()
        .filter(|f| *f != super::inventory::MTP_MIXER_FILE || dir.join(f).exists())
    {
        let path = dir.join(file);
        let (_, _, entries) = read_header_checked(&path).map_err(|e| ModelError::Config(format!("pack fingerprint: {e}")))?;
        let size = path.metadata().map_err(|e| ModelError::Config(format!("{}: {e}", path.display())))?.len();
        h.update(file.as_bytes());
        h.update([0]);
        h.update(size.to_le_bytes());
        let mut header_digest = Sha256::new();
        header_digest.update((entries.len() as u64).to_le_bytes());
        for (name, entry) in entries {
            let dtype =
                entry.get("dtype").and_then(|v| v.as_str()).ok_or_else(|| ModelError::Config(format!("{file}:{name}: dtype")))?;
            let shape = entry
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| ModelError::Config(format!("{file}:{name}: shape")))?;
            let offsets = entry
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| ModelError::Config(format!("{file}:{name}: offsets")))?;
            header_digest.update(name.as_bytes());
            header_digest.update([0]);
            header_digest.update(dtype.as_bytes());
            header_digest.update([0]);
            for values in [shape, offsets] {
                header_digest.update((values.len() as u64).to_le_bytes());
                for value in values {
                    let n =
                        value.as_u64().ok_or_else(|| ModelError::Config(format!("{file}:{name}: noninteger shape or offset")))?;
                    header_digest.update(n.to_le_bytes());
                }
            }
        }
        h.update(header_digest.finalize());
    }
    Ok(format!("sha256:{:x}", h.finalize()))
}

fn validate_pack_before_load<T>(
    dir: &Path,
    allow_unvalidated: bool,
    load: impl FnOnce() -> Result<T, ModelError>,
) -> Result<T, ModelError> {
    let pack = Pack::open(dir)?;
    crate::serve::check_packs(&pack, None, allow_unvalidated)?;
    boot::mark("packs_checked", None);
    load()
}

pub fn build_engine(o: &ServeOptions) -> Result<Engine<FlashNextBackend>, ModelError> {
    check_flash_options(o)?;
    let cfg = FlashNextConfig::from_dir(&o.model_dir).map_err(ModelError::Config)?;
    cfg.check_shapes().map_err(ModelError::Config)?;
    // The declared prefill contract: the compensated-fp16 tensor-core hyper-connection pre-mix (hcg) is the
    // always-on default for tiled prefill on every qualifying shape (H % 4 == 0, hc >= 1, rank >= 1 — true of
    // this family); decode and GEMV prefill keep the fp32 `hcx::pre` reference. There is no switch (the env
    // reader was removed); the contract is unconditional. The marker is asserted by the gate recipe's expect_log.
    let prefill_contract = if cfg.hidden % 4 == 0 && cfg.hc_streams >= 1 && cfg.hc_lowrank >= 1 {
        "hcg-compensated-fp16"
    } else {
        "hcx-fp32-reference"
    };
    let max_context = o.max_model_len.min(cfg.max_position_embeddings);
    let with_mtp = cfg.mtp_layers > 0;
    let profile = super::memory::memory_profile(
        &cfg,
        &MemoryOptions {
            window_rows: WINDOW_ROWS,
            prefill_rows: o.prefill_chunk,
            max_context,
            mtp: with_mtp,
            vision: None,
            trees: false,
        },
    )
    .map_err(|e| ModelError::Config(e.to_string()))?;

    // WeightMap::from_dir opens and validates the entire text pack's headers,
    // including both sidecars, without reading its large tensor payloads.
    let weights = WeightMap::from_dir(&o.model_dir, &cfg)
        .map_err(|errors| ModelError::Config(format!("Flash-Next pack: {}", errors.join("; "))))?;
    boot::mark("packs_scanned", None);
    let mode = warm_mode_from_env(std::env::var("REXL3_NGRAM_WARM").ok().as_deref());
    let (table_bytes, cuda_free) = validate_pack_before_load(&o.model_dir, o.allow_unvalidated_pack, || {
        let table_bytes = if mode != WarmMode::Off && !cfg.ple_layer_ids.is_empty() {
            let table = NgramTable::open(&o.model_dir, &cfg).map_err(ModelError::Config)?;
            (table.rows as u64)
                .checked_mul(table.words as u64)
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| ModelError::Config("n-gram table size overflows".into()))?
        } else {
            0
        };
        gb_cuda::rt::init_context()?;
        Ok((table_bytes, gb_cuda::rt::mem_info()?.0 as u64))
    })?;
    boot::mark("cuda_ready", None);
    let mut weight_bytes = weights
        .regions
        .iter()
        .try_fold(0u64, |sum, r| sum.checked_add(r.bytes))
        .ok_or_else(|| ModelError::Config("weight region sizes overflow".into()))?;
    // The exact MTP draft-head prefix: the model owns a device copy of the first MTP_HEAD_N columns of the LM head
    // (the drafter's whole candidate domain), so the plan accounts for its bytes before `free_after_load` sizes the KV
    // plan. Same domain and bytes as the model's `FlashNextModel::new` (both derive from `cfg.vocab` and validate the
    // checkpoint head covers it), so the plan stays truthful for the 262K contract.
    if with_mtp {
        let extra = draft_head_plan_bytes(&weights.groups, cfg.vocab)?;
        weight_bytes =
            weight_bytes.checked_add(extra).ok_or_else(|| ModelError::Config("prefix-head accounting overflow".into()))?;
        eprintln!("rexl3 event=mtp_exact_prefix_plan extra_bytes={extra}");
    }
    let host = MemInfo::read();
    let want_warm = mode != WarmMode::Off
        && !cfg.ple_layer_ids.is_empty()
        && wants_warm(mode, host.as_ref(), weight_bytes, table_bytes, o.floor_bytes);
    let free = free_after_load(cuda_free, host.as_ref(), weight_bytes);
    eprintln!("{}", hostmem::memory_view_line(cuda_free, host.as_ref(), o.floor_bytes));
    eprintln!(
        "rexl3 event=flash_next_plan_weights weights_gib={:.2} warm_table_gib={:.2} ngram_warm={} free_after_gib={:.2} prefill_contract={}",
        gib(weight_bytes),
        gib(table_bytes),
        warm_word(want_warm, mode),
        gib(free),
        prefill_contract
    );
    boot::fact("plan_free_gib", format!("{:.1}", gib(free)));
    boot::fact("ngram_warm", warm_word(want_warm, mode));
    boot::fact("prefill_contract", prefill_contract);
    // an explicit `--kv-tokens` is the pool the operator asked for: every slot's full window and any padding on top (for example
    // 4 x 262,144 plus 128 pages). A cap at one window left four slots sharing a single window of pages.
    let kv_tokens_cap = o.kv_tokens.unwrap_or(max_context) as u64;
    let snapshot_bytes = prefix::snapshot_bytes(&cfg, with_mtp);
    // the cache is sized from the window, for `auto` and an explicit `--prefix-cache-mib` alike (`docs/PREFIX_CACHE_SIZING.md` section 5)
    let plan_at = |slots: usize| {
        plan_memory_cached_windowed(
            &profile,
            free,
            o.floor_bytes,
            slots,
            kv_tokens_cap,
            0,
            o.prefix_cache.request(),
            snapshot_bytes,
            0,
            max_context as u64,
        )
    };
    let choice = choose_slots(o.slots, o.slots_auto, kv_tokens_cap, plan_at).map_err(|e| {
        ModelError::Config(format!("Flash-Next memory plan: {e}; MemAvailable after weights and warm table {:.1} GiB", gib(free)))
    })?;
    if choice.plan.kv_tokens < (choice.slots as u64 * MIN_TOKENS_PER_SLOT).min(kv_tokens_cap) {
        return Err(ModelError::Config("Flash-Next KV pool too small for the requested slots".into()));
    }
    let slots = choice.slots;
    if choice.shrunk() {
        eprintln!("rexl3 event=slots_shrunk requested={} granted={slots}", choice.requested);
    }
    let plan = choice.plan;
    eprintln!(
        "rexl3 event=flash_next_kv_plan kv_tokens={} slots={} kv_pages={} free_after_gib={:.2}",
        plan.kv_tokens,
        slots,
        plan.kv_blocks,
        gib(free)
    );
    if plan.cache.enabled() {
        eprintln!(
            "rexl3 event=flash_next_prefix_plan entries={} pages={} gib={:.2} reach_tokens={} window={max_context}",
            plan.cache.snapshot_slots,
            plan.cache.page_blocks,
            gib(plan.cache.bytes()),
            plan.cache.reach_tokens()
        );
        boot::fact("prefix_cache_reach_tokens", plan.cache.reach_tokens().to_string());
        if plan.cache.reach_tokens() < max_context as u64 {
            eprintln!(
                "rexl3 level=warn event=prefix_cache_reach reach_tokens={} window={max_context} note=\"a prompt longer than the reach is never served from the cache; raise --prefix-cache-mib\"",
                plan.cache.reach_tokens()
            );
        }
    } else {
        let reason = if o.prefix_cache == PrefixCacheOption::Off { "off" } else { "insufficient spare memory for two entries" };
        eprintln!("rexl3 event=flash_next_prefix_plan entries=0 reason={reason}");
    }
    let fingerprint = if plan.cache.enabled() { Some(pack_header_fingerprint(&o.model_dir)?) } else { None };
    let pool_pages = plan
        .kv_blocks
        .checked_add(plan.cache.page_blocks)
        .ok_or_else(|| ModelError::Config("Flash-Next KV page count overflows".into()))?;
    boot::fact("slots", slots.to_string());
    boot::fact("kv_pages", plan.kv_blocks.to_string());
    boot::fact("kv_tokens", plan.kv_tokens.to_string());
    let model = FlashNextModel::load(
        &o.model_dir,
        ModelOptions { slots, kv_pages: pool_pages, max_context, prefill_rows: o.prefill_chunk, warm: want_warm },
    )
    .map_err(|e| {
        crate::serve::allocation_error(
            "Flash-Next weights and engine buffers",
            e,
            weight_bytes.saturating_add(plan.planned_bytes(slots)),
            o.floor_bytes,
            host.as_ref(),
        )
    })?;
    let mtp_k = if o.drafting && o.spec.uses_mtp() && model.has_mtp() {
        std::env::var("REXL3_MTP_K").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(7).min(WINDOW_ROWS - 1)
    } else {
        0
    };
    let mut backend = FlashNextBackend::new(model, o.prefill_chunk, mtp_k);
    if let Some(fingerprint) = fingerprint {
        backend = backend.with_prefix_cache(plan.cache, fingerprint).map_err(|e| {
            crate::serve::allocation_error(
                "Flash-Next prefix snapshot arena",
                e,
                weight_bytes.saturating_add(plan.planned_bytes(slots)),
                o.floor_bytes,
                host.as_ref(),
            )
        })?;
    }
    let mm = ModelMemory::from_profile(&profile, 0, 0);
    let admission = plan.admission(mm, slots, max_context).map_err(|e| ModelError::Config(e.to_string()))?;
    let drafting = o.drafting && o.spec.drafts();
    let make = make_drafter(drafting);
    boot::mark("kv_planned", Some(gib(plan.kv_tokens * profile.kv_bytes_per_token())));
    let engine = Engine::new(
        backend,
        admission,
        EngineConfig { max_rows: WINDOW_ROWS, max_drafts: mtp_k.max(7), cost: CostModel::from_table(STEP_MS.to_vec()), drafting },
        make,
    );
    let engine = engine.with_prefix_cache(flash_next_prefix_config(&|name| std::env::var(name).ok())?);
    // REXL3_SPEC_BUDGET picks the policy that trims each round's verify rows (`legacy`, `curve`, `plan`, `margin`, `scored`), and
    // REXL3_SPEC_LEARN=1 lets it learn its priors from the traffic; only how many drafts ride a window depends on them, never
    // which token is committed
    let policy = crate::costs::budget_policy_from_env(DEFAULT_BUDGET)?;
    let learn = crate::costs::spec_learn_from_env()?;
    boot::fact("spec_budget", format!("{policy:?}").to_lowercase());
    Ok(engine.with_budget(policy, round_cost(mtp_k > 0)).with_learning(learn))
}

/// The Flash-Next prefix config: the interactive cut policy (cuts that serve a second question about a long document and a short tool
/// loop; the default policy serves neither), then the environment's overrides as in [`crate::serve::prefix_config_from_env`]. The restore
/// path is exact at any cut: `restore` copies the partial page and scrubs the donor's continuation (K/V rows and QSA pooled-key groups at
/// or beyond the cut) before the resumed sequence rewrites those positions, so no `cut_align` rounding is needed and short shared prefixes
/// (under one KV page) hit again.
fn flash_next_prefix_config(var: &dyn Fn(&str) -> Option<String>) -> Result<gb_engine::PrefixConfig, ModelError> {
    let c = crate::serve::interactive_prefix_config_from(var)?;
    Ok(c)
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
