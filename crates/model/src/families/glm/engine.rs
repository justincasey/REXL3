//! Building a ready-to-serve GLM-5.3-Flash engine from a model folder: map the pack, plan memory, load the weights through the pipelined
//! loader (with the pack's NextN layer when `--spec` drafts with it), allocate the state and scratch, and wire the backend into the engine. Images
//! are served when the server asks for them (`o.vision`): the vision tower is loaded after the text weights (the plan's `Vision` region, 1.05 GiB,
//! and its scratch) and published as the server's image encoder; everything it holds is out of the room the KV pool is planned from, and the
//! features of the images in flight are held back from it. Drafting is the backend's (`spec.rs`): the NextN layer and the engine's copy drafter.

use super::alloc::{self, AllocParams, AllocPlan};
use super::backend::GlmBackend;
use super::config::GlmConfig;
use super::load::{boot_mapped_with, LoadOptions};
use super::memory;
use super::model::{GlmModel, ModelOptions, VocabMode};
use super::prefix;
use super::spec::SpecConfig;
use super::vision::{self, GlmVisionTower, TowerOptions};
use super::vision_serve::{FeatureBudget, GlmEncoder};
use super::{log_weight_map, GlmPack};
use crate::error::ModelError;
use crate::hostmem;
use crate::loadstats::{self, Component};
use crate::memory::VisionOptions;
use crate::pack::PackError;
use crate::plan::{
    choose_slots, largest_viable_slots, plan_memory_cached_windowed, CacheRequest, MemoryPlan, PlanError, SlotChoice,
    MIN_TOKENS_PER_SLOT,
};
use crate::serve::{allocation_error, gib, interactive_prefix_config_from_env, mem_available, memory_options, ServeOptions};
use gb_engine::{boot, Drafter, Engine, EngineConfig, NoDrafts};
use gb_sched::{Admission, CostModel, MemoryProfile, ModelMemory};
use gb_spec::{CopyDrafter, SpecMode};

/// The GLM engine, before its type is erased.
pub type GlmEngine = Engine<GlmBackend>;

/// Milliseconds per decode or verify step by rows (all sequences together, the commit of the previous step included), measured on the GB10
/// at a context of 2,300 tokens (`tests/glm_bench.rs`, `bench_steps`, 2026-10-02: 44.9, 54.4, 72.4, 103.8, 150.6, 223.4, 320.8). The step
/// is nearly flat in the context (44.8 ms at 6,144 tokens, 43.8 at 128) and the same for the rows of one sequence or of several (4 sequences of
/// one row: 74.7 ms; 4 rows of one sequence: 72.4): every routed expert a row picks is read once more, so the cost grows with rows, not slots.
pub(super) const STEP_MS: &[(usize, f64)] = &[(1, 45.0), (2, 54.0), (4, 72.0), (8, 104.0), (16, 151.0), (32, 223.0), (64, 321.0)];

/// Strips of 128 token ids (the first ones: the tokenizer's merge order puts the frequent tokens first) the NextN drafter's head keeps by default, beside
/// the numeric block and the special tokens (`model::vocab_ranges`): 512 strips = 65,536 ids, with those 555 of the head's 1,210 strips, 46% of its
/// 0.48 GB. A draft only proposes, so the restriction cannot change an output byte; it costs the drafts whose true token lies outside the set (about 3% of
/// the tokens of English prose and code at this size, measured by class in `docs/GLM53.md`) and saves a little over half of the head's 2.1 ms in each of a
/// round's draft steps.
pub const MTP_VOCAB_PREFIX_STRIPS: usize = 512;
/// A draft call uses the restricted head only when every asking stream has at least this share of its last 32 tokens inside the set; text the set does
/// not cover (another script) keeps the full head.
pub const MTP_VOCAB_MIN_SHARE: f32 = 0.9;

/// `REXL3_MTP_VOCAB`: unset is the default (`MTP_VOCAB_PREFIX_STRIPS`, share `MTP_VOCAB_MIN_SHARE`), `off` / `0` / `full` the full head, `N` that many
/// prefix strips with the default share, `N:S` with the share `S` (0 uses the restricted head in every call). `None`: the full head.
pub fn mtp_vocab_setting(v: Option<&str>) -> Result<Option<(usize, f32)>, String> {
    let Some(v) = v else { return Ok(Some((MTP_VOCAB_PREFIX_STRIPS, MTP_VOCAB_MIN_SHARE))) };
    let v = v.trim();
    if matches!(v, "off" | "0" | "full") {
        return Ok(None);
    }
    let (n, s) = match v.split_once(':') {
        Some((n, s)) => (n, Some(s)),
        None => (v, None),
    };
    let strips: usize = n.parse().map_err(|_| format!("REXL3_MTP_VOCAB={v}: not off, a number of strips, or N:share"))?;
    // the head's kernel plan needs 64 strips (the numeric block and the last strip are 43 of them)
    if !(21..=1209).contains(&strips) {
        return Err(format!("REXL3_MTP_VOCAB={v}: the prefix is 21..=1209 strips of 128 ids (it is {strips})"));
    }
    let share = match s {
        Some(s) => s
            .parse::<f32>()
            .ok()
            .filter(|x| (0.0..=1.0).contains(x))
            .ok_or_else(|| format!("REXL3_MTP_VOCAB={v}: the share is a number in 0..=1"))?,
        None => MTP_VOCAB_MIN_SHARE,
    };
    Ok(Some((strips, share)))
}

/// The planner's refusal, with what the operator can do about it. `free` is what the plan was sized from (MemAvailable after the weights, rounded down
/// to a GiB, `hostmem::planning_free`) and `cuda_free` what `cudaMemGetInfo` calls free at that moment (MemFree: the page cache is not in it, and
/// `cudaMalloc` takes the cache back on demand, so it is no reason to refuse and no longer one). `fit` is the largest slot count that would boot.
fn refusal_text(
    plan_error: &str,
    free: u64,
    cuda_free: u64,
    floor: u64,
    requested: usize,
    pinned: bool,
    fit: Option<&SlotChoice>,
) -> String {
    let what = format!(
        "{plan_error} (MemAvailable after the weights is {:.1} GiB, of which the device sees {:.1} GiB as free now and the rest is page cache that is given back on demand; \
         the floor is {:.1} GiB)",
        gib(free),
        gib(cuda_free),
        gib(floor)
    );
    match (fit, pinned) {
        (Some(f), true) => format!(
            "{what}; --slots {requested} does not fit, --slots {} does (a pool of {} KV tokens): pass it, or drop --slots to let the server choose",
            f.slots, f.plan.kv_tokens
        ),
        (Some(f), false) => format!("{what}; {} slots would fit", f.slots),
        (None, _) => format!(
            "{what}; not even one slot with {MIN_TOKENS_PER_SLOT} tokens of KV fits: free memory on this box, or start with --no-vision, --spec none or a lower --max-image-tokens"
        ),
    }
}

/// The engine's plan for `slots` slots. The features of the images in flight (`vision`: the tower is loaded) are the one vision cost that is not allocated
/// up front: they are held back from the pool, and they grow with the slots (every slot may hold a request of several images), so each candidate
/// slot count has its own reserve.
#[allow(clippy::too_many_arguments)]
fn plan_for_slots(
    profile: &MemoryProfile,
    vision: Option<VisionOptions>,
    free: u64,
    floor: u64,
    kv_tokens_cap: u64,
    slots: usize,
    cache: CacheRequest,
    snapshot_bytes: u64,
    window_tokens: u64,
) -> Result<MemoryPlan, PlanError> {
    let features = match vision {
        Some(v) => profile.vision_reserve_bytes(0, v.concurrent_images(slots))?,
        None => 0,
    };
    // the cache is sized from the window, for `auto` and an explicit `--prefix-cache-mib` alike: the pages of one full window before anything else
    // (`docs/PREFIX_CACHE_SIZING.md` section 5)
    plan_memory_cached_windowed(profile, free, floor, slots, kv_tokens_cap, features, cache, snapshot_bytes, 0, window_tokens)
}

/// Why the slot count asked for was not the one granted: the planner's refusal at that count, or the pool it would have left.
fn shrink_reason(at_requested: &Result<MemoryPlan, PlanError>, requested: usize) -> String {
    match at_requested {
        Err(e) => format!("{requested} slots: {e}"),
        Ok(p) => {
            format!("{requested} slots leave a pool of {} KV tokens, under the {MIN_TOKENS_PER_SLOT} a slot needs", p.kv_tokens)
        }
    }
}

/// Refuse settings the GLM kernels cannot honour, before anything is loaded.
pub fn check_options(o: &ServeOptions) -> Result<(), ModelError> {
    if o.slots == 0 {
        return Err(ModelError::Config("--slots must be at least 1".into()));
    }
    if o.max_model_len == 0 {
        return Err(ModelError::Config("--max-model-len must be at least 1".into()));
    }
    // `--prefill-chunk` is taken by `alloc::prefill_chunk_for`: under the default P2 the server's 4,096 default is the P2 slice (4,096 rows) and
    // 129..=4,096 in whole 16-row chunks is accepted; under `REXL3_GLM_PREFILL_CONTRACT=p1` 512 rows unless asked, 128..=2,048. The prefill contract (`REXL3_GLM_PREFILL_CONTRACT`, logged once as `event=glm_prefill_contract`): a value that is not p1 or p2 is
    // refused here by name, before any weight is read, never run as another contract
    let c = super::contract::check_selected().map_err(ModelError::Config)?;
    super::contract::check_built(c).map_err(ModelError::Config)?;
    check_prefill_chunk(o.prefill_chunk, c).map_err(ModelError::Config)?;
    Ok(())
}

/// Under P2 the routed experts of every prompt chunk run in the prefill scratch, which a model of `MAX_ROWS` rows or fewer does not have:
/// such a `--prefill-chunk` is refused here, by name, before a weight is read (`GlmModel::check_p2` would refuse the model after the load).
/// P1 takes any chunk. P2 is the default contract (owner decision 2026-10-19), so these refusals reach a user who never set
/// `REXL3_GLM_PREFILL_CONTRACT`: each message says how to get the other behaviour.
pub fn check_prefill_chunk(flag: usize, c: super::contract::PrefillContract) -> Result<(), String> {
    let rows = alloc::prefill_chunk_for_contract(flag, c);
    if c == super::contract::PrefillContract::P2 && rows <= alloc::MAX_ROWS {
        return Err(format!(
            "{}=p2 needs --prefill-chunk above {} rows (P2's routed experts run in the prefill scratch); --prefill-chunk {flag} gives {rows} \
             (P2 is the default prefill contract: pass a larger --prefill-chunk, or set {}=p1 to run P1)",
            super::contract::CONTRACT_VAR,
            alloc::MAX_ROWS,
            super::contract::CONTRACT_VAR
        ));
    }
    // P2's KDA chunks are 16 rows from a prefill segment's first row and every slice but a prompt's last ends on that grid: a chunk that is
    // not a whole number of them would be rounded down by the backend; refused by name instead (docs/GLM_PREFILL_P2.md 8.1)
    let k = gb_cuda::kda::P2_CHUNK_ROWS;
    if c == super::contract::PrefillContract::P2 && rows % k != 0 {
        return Err(format!(
            "{}=p2 needs --prefill-chunk to be a multiple of {k} rows (P2's KDA chunk; docs/GLM_PREFILL_P2.md 8.1); --prefill-chunk {flag} \
             gives {rows} (P2 is the default prefill contract: pass a multiple of {k}, or set {}=p1 to run P1)",
            super::contract::CONTRACT_VAR,
            super::contract::CONTRACT_VAR
        ));
    }
    Ok(())
}

pub fn build_engine(o: &ServeOptions) -> Result<GlmEngine, ModelError> {
    gb_engine::boot::checkpoint();
    check_options(o)?;
    loadstats::set_component(Component::Scan);
    let cfg = GlmConfig::from_dir(&o.model_dir).map_err(ModelError::Config)?;
    // the pack's NextN layer is loaded (3.5 GiB) and its cache allocated only when the mode drafts with it (`--spec auto|mtp`) and no DFlash2 draft
    // (`--draft DIR`) drafts in its place
    let with_mtp = super::tp_boot::mtp_wanted(o, &cfg);
    // the family's memory model, validated before a single weight is read
    let mem_opts = memory::engine_options(&memory_options(o, with_mtp));
    let profile = memory::memory_profile(&cfg, &mem_opts).map_err(|e| ModelError::Config(e.to_string()))?;
    let t = std::time::Instant::now();
    let ck = gb_exl3::checkpoint::scan(&o.model_dir, false).map_err(|e| ModelError::Pack(PackError::Io(e.to_string())))?;
    // `--no-vision` claims and skips the tower: the text model maps (and serves) whatever the tower's format
    let pack = GlmPack::from_checkpoint_with(&ck, cfg.clone(), o.vision.is_some())?;
    let scan_s = t.elapsed().as_secs_f64();
    boot::mark("packs_scanned", None);
    let target_fingerprint = check_pack(&ck, o.allow_unvalidated_pack)?;
    boot::mark("packs_checked", None);
    log_weight_map("glm5_next", &pack);
    // a pack that does not map, or one this release was not validated on, is refused above before the driver is asked anything
    crate::shapes::check_driver(&gb_cuda::rt::versions()).map_err(ModelError::Config)?;
    gb_cuda::rt::init_context()?;
    boot::mark("cuda_ready", None);
    crate::serve::boot_checkpoint()?;

    let layers = cfg.layers.len();
    let lo = LoadOptions::from_env(o.floor_bytes);
    loadstats::set_component(Component::Target);
    boot::mark("target_loading", None);
    let ph = loadstats::phase_start(Component::Target);
    let booted = boot_mapped_with(&o.model_dir, pack, Some(layers), with_mtp, &lo, scan_s)?;
    loadstats::phase_end(ph);
    boot::mark("target_loaded", Some(gib(booted.device.regions.bytes())));
    crate::serve::boot_checkpoint()?;
    eprintln!(
        "rexl3 event=glm_load regions={} gib={:.2} scan_s={:.2} alloc_s={:.2} load_s={:.2} workers={} spans={} direct_spans={} markers_checked={}",
        booted.device.regions.count(),
        gib(booted.device.regions.bytes()),
        booted.scan_s,
        booted.alloc_s,
        booted.load_s,
        booted.report.workers,
        booted.report.spans,
        booted.report.direct_spans,
        booted.plan.markers.len()
    );

    // The vision tower, when images are served: its weights (the plan's one `Vision` region, read through the pipelined loader and prepared in place)
    // and its scratch are allocated BEFORE the room that is left is measured, so they are out of what the KV pool is planned from. Fail closed: the
    // tower is refused (and the engine does not start) when it would leave less than the floor, before a byte of it is allocated.
    let mut tower: Option<(&crate::serve::VisionSlot, GlmVisionTower)> = None;
    if let Some(slot) = &o.vision {
        match (&cfg.vision, &booted.pack.weights.vision) {
            (Some(vc), Some(_)) => {
                loadstats::set_component(Component::Vision);
                let ph = loadstats::phase_start(Component::Vision);
                let tokens = vision::image_token_cap(&cfg, o.max_image_tokens);
                let need = vision::weight_region_bytes(vc) + vision::scratch_bytes(tokens * vision::PATCHES_PER_TOKEN, vc.tower);
                if let Some(avail) = mem_available() {
                    if avail < need.saturating_add(o.floor_bytes) {
                        return Err(ModelError::Config(format!(
                            "the vision tower needs {:.2} GiB (weights and scratch for images of {tokens} tokens) and MemAvailable is {:.2} GiB with a {:.2} GiB floor; \
                             start with --no-vision or a lower --max-image-tokens",
                            gib(need),
                            gib(avail),
                            gib(o.floor_bytes)
                        )));
                    }
                }
                let t = GlmVisionTower::load(
                    &booted.pack,
                    &o.model_dir,
                    &TowerOptions::new(tokens * vision::PATCHES_PER_TOKEN, o.floor_bytes),
                )?;
                // what was allocated is what the memory model priced, item by item
                if t.scratch_allocations() != vision::scratch_items(t.max_patches, t.format)
                    || t.weight_bytes() != vision::weight_region_bytes(vc)
                {
                    return Err(ModelError::Config("the vision memory model and the tower's allocations disagree".into()));
                }
                loadstats::phase_end(ph);
                eprintln!(
                    "rexl3 event=glm_vision_loaded tokens={tokens} weights_gib={:.3} scratch_gib={:.3} alloc_s={:.2} load_s={:.2} prepare_s={:.2} scratch_s={:.2}",
                    gib(t.weight_bytes()),
                    gib(t.scratch_bytes()),
                    t.boot.alloc_s,
                    t.boot.load_s,
                    t.boot.prepare_s,
                    t.boot.scratch_s
                );
                boot::mark("vision_loaded", Some(gib(t.weight_bytes() + t.scratch_bytes())));
                boot::fact("vision_scratch_gib", format!("{:.2}", gib(t.scratch_bytes())));
                tower = Some((slot, t));
            }
            _ => eprintln!(
                "rexl3 level=warn event=no_vision_tower reason=\"the pack has no vision tower\" (image requests will be refused)"
            ),
        }
    }

    // What is left to allocate. `cudaMemGetInfo` free is MemFree and leaves out the page cache (tens of GiB after the pack was read or anything was
    // built); MemAvailable counts it because the kernel gives it back, and `cudaMalloc` is a kernel allocation that takes it back on demand (measured:
    // 100 GiB allocated against 30 GiB of MemFree, `docs/EVAL_RESULTS.md` "Memory: capacity that does not depend on the boot history"). So the
    // plan is sized from MemAvailable alone, with the floor and the reserves out of it once (`plan_memory`); the clamp to the device's figure that
    // stood here refused a boot with 0.9 GiB of MemFree and 17.6 GiB available, for a plan that needs 2 GiB.
    loadstats::set_component(Component::Engine);
    let (cuda_free, _) = gb_cuda::rt::mem_info()?;
    let cuda_free = cuda_free as u64;
    let host = hostmem::MemInfo::read();
    let free = hostmem::planning_free(cuda_free, host.as_ref());
    eprintln!("{}", hostmem::memory_view_line(cuda_free, host.as_ref(), o.floor_bytes));
    boot::fact("plan_free_gib", format!("{:.1}", gib(free)));
    let kv_tokens_cap = o.kv_tokens.unwrap_or(o.max_model_len) as u64;
    // a DFlash2 drafter (`--draft DIR`): its weights, rings (one a slot), scratch and the trunk's taps are out of what the pool is planned from
    let chunk = alloc::prefill_chunk_for(o.prefill_chunk);
    let dflash_at = |slots: usize| super::dflash::plan(o, &cfg, slots, chunk);
    dflash_at(o.slots)?;
    let draft_bytes = |slots: usize| dflash_at(slots).ok().flatten().map_or(0, |d| d.bytes);
    // a snapshot slot of the prefix cache is the recurrent state of a prompt plus a few raw rows and the drafter's carry (the model's own figure,
    // `prefix::snapshot_bytes`); the cache is carved out of what the plan leaves (`Auto`) or before the KV pool is sized (an explicit size)
    let snapshot_bytes = prefix::snapshot_bytes(&cfg, with_mtp).map_err(ModelError::Config)?;
    let cache_request = o.prefix_cache.request();
    let has_tower = tower.is_some();
    let plan_at = |slots: usize| {
        plan_for_slots(
            &profile,
            mem_opts.vision.filter(|_| has_tower),
            free.saturating_sub(draft_bytes(slots)),
            o.floor_bytes,
            kv_tokens_cap,
            slots,
            cache_request,
            snapshot_bytes,
            o.max_model_len as u64,
        )
    };
    // the slot count: the one asked for, or (the default, `--slots` not given) the most the box holds; an explicit count is never changed
    let choice = choose_slots(o.slots, o.slots_auto, kv_tokens_cap, plan_at).map_err(|e| {
        let fit = largest_viable_slots(o.slots, kv_tokens_cap, plan_at);
        ModelError::Config(refusal_text(&e.to_string(), free, cuda_free, o.floor_bytes, o.slots, !o.slots_auto, fit.as_ref()))
    })?;
    let slots = choice.slots;
    let plan = choice.plan.clone();
    // what the plan held back for the features of the images in flight (zero without the tower)
    let features_reserve = plan.vision_reserve_bytes;
    if choice.shrunk() {
        eprintln!(
            "rexl3 level=warn event=slots_shrunk requested={} granted={slots} kv_tokens={} reason=\"{}\" (pass --slots N to pin the count)",
            o.slots,
            plan.kv_tokens,
            shrink_reason(&plan_at(o.slots), o.slots)
        );
        boot::fact("slots_requested", o.slots.to_string());
    }
    boot::fact("slots", slots.to_string());
    eprintln!(
        "rexl3 event=kv_plan free_gib={:.1} kv_pages={} kv_tokens={} slots={}",
        gib(free),
        plan.kv_blocks,
        plan.kv_tokens,
        slots
    );
    boot::fact("kv_pages", plan.kv_blocks.to_string());
    boot::fact("kv_tokens", plan.kv_tokens.to_string());
    if plan.cache.enabled() {
        eprintln!(
            "rexl3 event=prefix_cache_plan entries={} pages={} gib={:.2} spare_gib={:.1} reach_tokens={} window={}",
            plan.cache.snapshot_slots,
            plan.cache.page_blocks,
            gib(plan.cache.bytes()),
            gib(plan.spare_bytes),
            plan.cache.reach_tokens(),
            o.max_model_len
        );
        boot::fact("prefix_cache", format!("{} entries, {:.2} GiB", plan.cache.snapshot_slots, gib(plan.cache.bytes())));
        boot::fact("prefix_cache_reach_tokens", plan.cache.reach_tokens().to_string());
        if plan.cache.reach_tokens() < o.max_model_len as u64 {
            eprintln!(
                "rexl3 level=warn event=prefix_cache_reach reach_tokens={} window={} note=\"a prompt longer than the reach is never served from the cache; raise --prefix-cache-mib\"",
                plan.cache.reach_tokens(),
                o.max_model_len
            );
        }
    } else {
        // the request was `Off`, or `Auto` on a box with no spare memory: say which, so an operator is not left guessing
        let why = match o.prefix_cache.request() {
            CacheRequest::Off => "off",
            _ => "no spare memory for two entries after the KV pool (pass --prefix-cache-mib to carve one out of it)",
        };
        eprintln!("rexl3 event=prefix_cache_plan entries=0 reason=\"{why}\" spare_gib={:.1}", gib(plan.spare_bytes));
        boot::fact("prefix_cache", why);
    }
    crate::serve::boot_checkpoint()?;
    boot::mark("engine_allocating", None);
    let ph = loadstats::phase_start(Component::Engine);
    // an allocation the device refuses although the plan fits ends the boot with what was planned and what the host had, never a bare CUDA code
    let planned = plan.planned_bytes(slots);
    let refuse = |what: &str, e: ModelError| allocation_error(what, e, planned, o.floor_bytes, host.as_ref());
    // the prefill chunk: the scratch of a chunk of more than 128 rows (the large-row kernels') was priced into the plan above
    // the pool holds the pages the cache may keep on top of the pages admission hands out
    let pool_blocks = plan.kv_blocks + plan.cache.page_blocks;
    let mut model = GlmModel::new_chunked(
        cfg.clone(),
        booted.device,
        &ModelOptions { slots, kv_blocks: pool_blocks, max_context: o.max_model_len, layers },
        chunk,
    )
    .map_err(|e| refuse("the engine's buffers (KV pool, slot state, scratch)", e))?;
    if plan.cache.enabled() {
        model.enable_prefix_cache(plan.cache.snapshot_slots).map_err(|e| refuse("the prefix cache's snapshot pool", e))?;
    }
    if let Some(d) = dflash_at(slots)? {
        super::dflash::load(&mut model, &d).map_err(|e| refuse("the DFlash2 drafter", e))?;
    }
    loadstats::phase_end(ph);
    // the plan the admission was priced with is what was allocated: say so loudly if it is not (a bug in the memory model, never a property of the pack)
    let ap = AllocPlan::new_chunked(
        &cfg,
        &AllocParams {
            slots,
            kv_blocks: pool_blocks,
            max_context: o.max_model_len,
            max_rows: alloc::MAX_ROWS,
            window_rows: alloc::WINDOW_ROWS,
            mtp: model.has_mtp(),
        },
        chunk,
    )
    .and_then(|p| p.with_prefix_snapshots(&cfg, plan.cache.snapshot_slots, model.has_mtp()))
    .map_err(ModelError::Config)?;
    if let Err(e) = model.check_allocations(&ap) {
        return Err(ModelError::Config(format!("the memory model and the allocations disagree: {e}")));
    }
    eprintln!("rexl3 event=glm_allocated gib={:.2}", gib(ap.total()));
    // the tower becomes the server's image encoder only now that the engine is allocated: a request that carries an image cannot arrive before
    if let Some((slot, t)) = tower {
        let tokens = t.max_patches / vision::PATCHES_PER_TOKEN;
        let pp_id = super::image_preprocessor(&o.model_dir, o)?.map(|p| p.id()).unwrap_or_default();
        let enc = GlmEncoder::new(t, FeatureBudget::new(features_reserve), &pp_id)?;
        let _ = slot.set(std::sync::Arc::new(crate::serve::VisionService::new(Box::new(enc), tokens)));
    }
    // speculation: the backend owns the drafters (the NextN layer and the copy drafter) and offers drafts from `propose_batch`; with drafting off
    // (`--no-drafts`, `--spec none`) it offers none and the engine never asks
    let mut backend = if o.drafting && o.spec.drafts() {
        GlmBackend::with_spec(model, SpecConfig::from_env(o.spec)?)
    } else {
        GlmBackend::new(model)
    }
    .with_chunk(chunk);
    if plan.cache.enabled() {
        let ns = prefix::namespace_of(&target_fingerprint, backend.m.has_mtp());
        backend = backend.with_prefix_cache(ns, plan.cache)?;
    }
    // the NextN drafter's head over a part of the vocabulary (a draft only proposes: no output byte can change); its copy of those columns of the
    // `lm_head` (about 0.23 GiB at the default size) is made after the plan was checked and comes out of the headroom the planner leaves the device
    if backend.spec.is_some() && backend.m.has_mtp() {
        let setting = mtp_vocab_setting(std::env::var("REXL3_MTP_VOCAB").ok().as_deref()).map_err(ModelError::Config)?;
        if let Some((strips, share)) = setting {
            let ranges = super::model::vocab_ranges(strips, cfg.vocab);
            let t = std::time::Instant::now();
            let bytes = backend.m.mtp_restrict_vocab(&ranges).map_err(|e| refuse("the draft head's vocabulary strips", e))?;
            backend.m.set_mtp_vocab(if share > 0.0 { VocabMode::Adaptive(share) } else { VocabMode::Restricted });
            let kept: usize = ranges.iter().map(|&(lo, hi)| hi - lo).sum();
            eprintln!(
                "rexl3 event=mtp_vocab strips={kept} of={} extra_mib={:.0} min_share={share} build_s={:.2}",
                cfg.vocab / 128,
                bytes as f64 / 1048576.0,
                t.elapsed().as_secs_f64()
            );
        }
    }
    let mm = ModelMemory::from_profile(&profile, 0, 0);
    let admission = plan.admission(mm, slots, o.max_model_len).map_err(|e| ModelError::Config(e.to_string()))?;
    boot::mark("kv_planned", Some(gib(plan.kv_tokens * profile.kv_bytes_per_token())));
    Ok(engine_over(backend, admission, o.drafting).with_prefix_cache(interactive_prefix_config_from_env()?))
}

/// The engine over a backend: the step table of this model for the verify-row budget and, with `drafting`, the copy drafter per sequence as the
/// fallback for a backend that offers no drafts of its own (a backend with a speculation runtime answers every `propose_batch` itself).
pub fn engine_over(backend: GlmBackend, admission: Admission, drafting: bool) -> GlmEngine {
    // `--spec auto` hands the budget to the arbiter and `--spec mtp` drafts at the depth it was asked for: the engine's own row budget must not trim
    // either (it knows one acceptance rate per stream, whatever the drafter, and no draft cost). Flat: more rows never cost more, so it keeps every draft.
    let own_budget = backend.spec.as_ref().is_some_and(|s| matches!(s.cfg.mode, SpecMode::Auto | SpecMode::Mtp));
    let cost = if own_budget {
        CostModel::from_table(vec![(1, 1.0), (alloc::WINDOW_ROWS, 1.0)])
    } else {
        CostModel::from_table(STEP_MS.to_vec())
    };
    let make: Box<dyn Fn() -> Box<dyn Drafter>> = if drafting {
        Box::new(|| Box::new(CopyDrafter::new(8, 7)) as Box<dyn Drafter>)
    } else {
        Box::new(|| Box::new(NoDrafts) as Box<dyn Drafter>)
    };
    Engine::new(backend, admission, EngineConfig { max_rows: alloc::WINDOW_ROWS, max_drafts: 7, cost, drafting }, make)
}

/// The pack's fingerprint against the validated list; recorded as a boot fact. An unvalidated pack is refused unless the operator asks.
pub(super) fn check_pack(ck: &gb_exl3::checkpoint::Checkpoint, allow_unvalidated: bool) -> Result<String, ModelError> {
    let t = crate::fingerprint::fingerprint(ck);
    boot::fact("target_fingerprint", t.clone());
    match crate::fingerprint::validated_pack(&t, None) {
        Some(name) => {
            boot::fact("pack_validated", name);
            Ok(t)
        }
        None if allow_unvalidated => {
            boot::fact("pack_validated", "no (--allow-unvalidated-pack)");
            eprintln!("rexl3 level=warn event=unvalidated_pack target={t} draft=none");
            Ok(t)
        }
        None => Err(ModelError::Config(format!(
            "this pack is not one this release was validated on (target {t}); pass --allow-unvalidated-pack to serve it anyway"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::hostmem::{planning_free, MemInfo};
    use crate::memory::MemoryOptions;

    const GIB: u64 = 1 << 30;
    const GLM_CONFIG: &str = include_str!("../../../../../testdata/glm53/config.json");
    const FLOOR: u64 = 8 * GIB;
    const CAP: u64 = 262_144;

    /// The profile the shipped server plans a GLM pack with (the NextN layer drafts, images are served), and its vision options.
    fn shipped() -> (MemoryProfile, Option<VisionOptions>) {
        // the pack's processor config caps an image at 8,000 tokens and the GLM server's default `--max-image-tokens` is 8,000: each of the 4 images a
        // slot may hold is 8,000 x 4,096 x 4 B of features (the live boots of docs/EVAL_RESULTS.md "Memory" are these numbers)
        let mut cfg = GlmConfig::from_json(&serde_json::from_str(GLM_CONFIG).unwrap()).unwrap();
        let p: serde_json::Value =
            serde_json::from_str(include_str!("../../../../../testdata/glm53/processor_config.json")).unwrap();
        cfg.processor = Some(GlmConfig::processor_from_json(&p).unwrap());
        let vision = Some(VisionOptions { max_image_tokens: 8000, ..VisionOptions::SERVER_DEFAULT });
        let opts = memory::engine_options(&MemoryOptions { mtp: true, vision, ..MemoryOptions::serving_defaults() });
        (memory::memory_profile(&cfg, &opts).unwrap(), opts.vision)
    }

    /// The engine's plan for a given count at a given free memory, as `build_engine` makes it.
    fn plan_at(free: u64) -> impl Fn(usize) -> Result<MemoryPlan, PlanError> {
        let (profile, vision) = shipped();
        move |slots| plan_for_slots(&profile, vision, free, FLOOR, CAP, slots, CacheRequest::Off, 0, CAP)
    }

    #[test]
    fn the_draft_vocabulary_setting_parses_and_refuses_what_the_head_cannot_do() {
        assert_eq!(mtp_vocab_setting(None), Ok(Some((512, 0.9))));
        for off in ["off", "0", "full", " off "] {
            assert_eq!(mtp_vocab_setting(Some(off)), Ok(None), "{off}");
        }
        assert_eq!(mtp_vocab_setting(Some("384")), Ok(Some((384, 0.9))));
        assert_eq!(mtp_vocab_setting(Some("256:0")), Ok(Some((256, 0.0))));
        assert_eq!(mtp_vocab_setting(Some("512:0.75")), Ok(Some((512, 0.75))));
        for bad in ["", "x", "20", "1210", "512:", "512:2", "512:x", "-3"] {
            assert!(mtp_vocab_setting(Some(bad)).is_err(), "{bad:?} must be refused");
        }
    }

    #[test]
    fn the_flat_budget_keeps_every_draft_the_engine_is_offered() {
        // `--spec auto|mtp`: the engine's own row budget must not trim what the arbiter or the fixed depth asked for. With a flat step cost a draft row is
        // free, so every draft with a non-zero acceptance rides, one stream or eight, a high estimate or a low one
        use gb_sched::{choose_budget, CostModel, Demand};
        let flat = CostModel::from_table(vec![(1, 1.0), (alloc::WINDOW_ROWS, 1.0)]);
        for accept in [0.9, 0.5, 0.05, 0.001] {
            for streams in [1usize, 2, 8] {
                let demands: Vec<Demand> = (0..streams).map(|_| Demand::geometric(accept, 4)).collect();
                let (_, alloc) = choose_budget(&flat, &demands, alloc::WINDOW_ROWS);
                assert_eq!(alloc, vec![4; streams], "accept {accept}, {streams} streams");
            }
        }
    }

    #[test]
    fn a_refusal_names_the_figures_and_what_would_fit_and_does_not_blame_the_page_cache() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let base = "not enough memory: 0.0 GiB allocatable";
        let fit = largest_viable_slots(8, CAP, plan_at(16 * GIB)).unwrap();
        // an explicit count: refused by name, with the count that would boot
        let t = refusal_text(base, 16 * GIB, GIB, FLOOR, 8, true, Some(&fit));
        let want = [
            base,
            "MemAvailable after the weights is 16.0 GiB",
            "the device sees 1.0 GiB as free now",
            "given back on demand",
            "the floor is 8.0 GiB",
            "--slots 8 does not fit",
            &format!("--slots {} does", fit.slots),
            "drop --slots",
        ];
        for w in want {
            assert!(t.contains(w), "{w:?} missing from: {t}");
        }
        assert!(!t.contains("page_cache.py") && !t.contains("cannot use"), "the page cache is not the reason any more: {t}");
        // the default count: say what the box holds
        let t = refusal_text(base, 16 * GIB, GIB, FLOOR, 8, false, Some(&fit));
        assert!(t.contains(&format!("{} slots would fit", fit.slots)) && !t.contains("--slots 8 does not fit"), "{t}");
        // nothing viable: the levers that free memory
        let t = refusal_text(base, 9 * GIB, 9 * GIB, FLOOR, 8, true, None);
        assert!(t.contains("not even one slot") && t.contains("--no-vision") && t.contains("--spec none"), "{t}");
    }

    #[test]
    fn the_default_slot_count_shrinks_to_what_the_box_holds_and_an_explicit_one_never_does() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        // GLM after its weights on anonymized host-b leaves 16-17 GiB (MemAvailable, 2026-10-03): 8 GiB of floor, the 5 GiB reserve and the images' features
        // leave a pool that eight slots of KDA state cannot share
        let free = 16 * GIB;
        let at = plan_at(free);
        assert!(matches!(at(8), Err(PlanError::NoRoom { .. })), "eight slots are refused: {:?}", at(8));
        let auto = choose_slots(8, true, CAP, &at).unwrap();
        assert!(auto.shrunk() && auto.requested == 8, "{auto:?}");
        assert!(
            auto.plan.kv_tokens >= auto.slots as u64 * MIN_TOKENS_PER_SLOT,
            "every slot can hold {MIN_TOKENS_PER_SLOT} tokens: {auto:?}"
        );
        // the most that fit: one more slot either does not plan or leaves the pool too thin
        let more = at(auto.slots + 1);
        assert!(
            more.as_ref().map_or(true, |p| p.kv_tokens < (auto.slots as u64 + 1) * MIN_TOKENS_PER_SLOT),
            "{} slots would also have been viable: {more:?}",
            auto.slots + 1
        );
        // the shipped pair (NextN layer, images served) on 16 GiB after the weights, 8 GiB of floor: slots -> (pool pages, KV tokens); the table of
        // docs/EVAL_RESULTS.md "Memory". A slot costs 278 MiB of state and 4 images of features (8,000 x 4,096 x 4 B each, 125 MiB): 0.76 GiB, as much as
        // a pool of 260 pages (66,000 tokens)
        let table = [(1, 719, 183_808), (2, 475, 121_088), (3, 231, 58_368)];
        for (s, blocks, tokens) in table {
            let p = at(s).unwrap();
            assert_eq!((p.kv_blocks, p.kv_tokens), (blocks, tokens), "{s} slots");
        }
        assert_eq!(
            auto.slots, 3,
            "the most that fit: four slots have 1.05 GiB allocatable and need 1.09 GiB for their state and one page"
        );
        // what the live boots on anonymized host-b granted, as the plan of the same free figure (docs/EVAL_RESULTS.md "Memory": plan free 16, 17 and 18 GiB, and 19 as the next):
        // each GiB is one or two more slots, which is how closely the grant follows MemAvailable
        for (free_gib, slots, blocks, tokens) in
            [(16u64, 3usize, 231usize, 58_368u64), (17, 4, 308, 77_824), (18, 6, 140, 34_304), (19, 7, 217, 53_760)]
        {
            let c = choose_slots(8, true, CAP, plan_at(free_gib * GIB)).unwrap();
            assert_eq!(
                (c.slots, c.plan.kv_blocks, c.plan.kv_tokens),
                (slots, blocks, tokens),
                "{free_gib} GiB after the weights"
            );
        }
        // an explicit count is refused, never shrunk, in the planner's words
        assert!(matches!(choose_slots(8, false, CAP, &at), Err(PlanError::NoRoom { .. })));
        // and one that fits is kept as asked, whichever way it was asked
        for auto in [true, false] {
            let c = choose_slots(3, auto, CAP, &at).unwrap();
            assert_eq!((c.slots, c.shrunk()), (3, false));
            assert_eq!(c.plan, at(3).unwrap(), "the plan of the count asked for, untouched");
        }
        // a roomy box (the hand plan of docs/GLM53.md: 30 GiB after the weights) keeps eight slots and the whole context
        let roomy = choose_slots(8, true, CAP, plan_at(30 * GIB)).unwrap();
        assert_eq!((roomy.slots, roomy.plan.kv_tokens), (8, CAP), "{roomy:?}");
        // a box with nothing viable falls back to the request's own refusal
        let none = choose_slots(8, true, CAP, plan_at(10 * GIB));
        assert!(matches!(none, Err(PlanError::NoRoom { .. })), "{none:?}");
        assert!(largest_viable_slots(8, CAP, plan_at(10 * GIB)).is_none());
    }

    #[test]
    fn what_the_engine_allocates_at_every_granted_slot_count_fits_the_plan_that_granted_it() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        // plan == allocations at each count the shrink can grant (the GPU tests `allocations_equal_the_plan*` compare the real buffers with `AllocPlan` at the
        // count they run; this ties `AllocPlan` to the plan that admitted it at every count): the engine's buffers (`AllocPlan`: scratch, window records, slot
        // state, the pool) are inside what the plan held back for them (the engine reserve), the slots' state and the pool's pages, and each slot is exactly the
        // plan's state per slot
        let cfg = GlmConfig::from_json(&serde_json::from_str(GLM_CONFIG).unwrap()).unwrap();
        let chunk = alloc::prefill_chunk_for(4096);
        let alloc_total = |slots: usize, kv_blocks: usize| {
            let p = AllocParams {
                slots,
                kv_blocks,
                max_context: 262_144,
                max_rows: alloc::MAX_ROWS,
                window_rows: alloc::WINDOW_ROWS,
                mtp: true,
            };
            AllocPlan::new_chunked(&cfg, &p, chunk).unwrap().total()
        };
        let at = plan_at(17 * GIB);
        let mut counted = 0;
        for slots in 1..=8 {
            let Ok(plan) = at(slots) else { continue };
            let engine = alloc_total(slots, plan.kv_blocks);
            let budget = plan.planned_bytes(slots) - plan.vision_reserve_bytes - plan.cache.bytes();
            assert!(engine <= budget, "{slots} slots: the engine allocates {engine} B of the {budget} B the plan holds for it");
            // the pool and the slots are priced to the byte: one more slot or page is exactly its plan cost
            assert_eq!(alloc_total(slots + 1, plan.kv_blocks) - engine, plan.state_bytes_per_slot, "{slots} slots");
            assert_eq!(alloc_total(slots, plan.kv_blocks + 1) - engine, plan.block_bytes, "{slots} slots");
            counted += 1;
        }
        assert!(counted >= 5, "the box of the table grants several counts ({counted})");
    }

    #[test]
    fn the_granted_slots_do_not_depend_on_what_the_box_did_before_the_boot() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        // the same box after a cold boot and after 80 GiB of buffered reads: the device's free differs by 60 GiB, MemAvailable by nothing
        let cold = MemInfo { total: 121 * GIB, free: 16 * GIB + GIB / 2, available: 16 * GIB + GIB / 2, cached: GIB };
        let dirty = MemInfo { total: 121 * GIB, free: GIB, available: 16 * GIB + GIB / 3, cached: 17 * GIB };
        let (a, b) = (planning_free(cold.free, Some(&cold)), planning_free(dirty.free, Some(&dirty)));
        assert_eq!(a, b, "boot history does not move the figure the plan is sized from");
        let (pa, pb) = (choose_slots(8, true, CAP, plan_at(a)).unwrap(), choose_slots(8, true, CAP, plan_at(b)).unwrap());
        assert_eq!(pa, pb);
        // the old rule (the smaller of the device's free and MemAvailable) planned the dirty box 15 GiB lower: nothing viable
        assert!(largest_viable_slots(8, CAP, plan_at(dirty.free.min(dirty.available))).is_none());
    }
}
