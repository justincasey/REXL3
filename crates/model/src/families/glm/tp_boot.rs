//! The tensor-parallel boot of the GLM family (`docs/GLM_TP.md` section 5): rank 0's engine (a [`GlmTpBackend`] over a `TpLeader`) and
//! the follower process of rank 1. The protocol, the boot handshake and the failure model are `crate::lockstep`; the communicator is the
//! Nemotron module's `NcclTp` (model-neutral: `docs/CLUSTER_FAMILY_PORTING.md` 6.3), bootstrapped over the same connection.
//!
//! **Plan before load.** The boot gate compares slots and KV pages before any weight is read, and a single GLM node plans them from what is
//! free AFTER its load. A rank therefore plans from what is free now minus the bytes of the regions it will allocate ([`rank_weight_bytes`]:
//! the text trunk, its share of the routed experts, no NextN layer and no vision tower), and the group needs the geometry pinned:
//! `--slots N` and `--kv-tokens N` on every rank.
//!
//! **A rank loads only its share.** With a communicator, a rank maps the pack with its ownership (`ExpertOwnership::range`) and reads and
//! places only those routed experts (`GlmPack::from_inventory_owned`): every MoE layer is then subbanks of the owned experts (one per width
//! class), run as pairs, exchange, combine. Without one (`REXL3_TP_COMM=none`) every rank loads the whole single-node model. A follower loads with rank 0's options (it may hold
//! more, never less); an allocation that does not fit after the load fails the boot by name.
//!
//! **Refused by name before the rendezvous** ([`check_tp_options`]): images on a follower (only rank 0 holds the vision tower: it encodes
//! every image and sends the rows of each chunk, `tp_vision.rs`; rank 0 may serve images), a slot count or KV pool left to the planner.
//! The prefix cache is served (`docs/GLM_TP.md` 4.2): each rank plans and allocates its own snapshot pool, rank 0 forwards
//! `EnableSnapshots` and decides hits, and every rank snapshots and restores its own state between chunks. The backend's drafters (`--spec mtp|auto|copy`) run on rank 0 and the NextN layer on every
//! rank, its experts owned like the trunk's (`tp_mtp`, `docs/GLM_TP.md` 9, lane L12); images and drafting together: `docs/GLM_TP.md` 11.
//!
//! `REXL3_TP_COMM=none` boots the group without a communicator: every rank loads and runs the whole single-node model in lockstep (no
//! ownership, no exchange; the bring-up check of the call stream alone).

use super::alloc::{self, AllocParams, AllocPlan, WINDOW_ROWS};
use super::config::GlmConfig;
use super::engine::{check_options, check_pack, STEP_MS};
use super::load::{boot_mapped_with, LoadOptions};
use super::memory;
use super::model::{GlmModel, ModelOptions};
use super::tp::{self, GlmCall, GlmFacts, GlmRank};
use super::tp_backend::GlmTpBackend;
use super::tp_vision::{self, TpVision};
use super::GlmPack;
use crate::error::ModelError;
use crate::families::nemotron_h_omni::tp::{TpComm, TpTopology};
use crate::families::nemotron_h_omni::tp_boot::{default_nccl_channels, sharded};
use crate::families::nemotron_h_omni::tp_comm::NcclTp;
use crate::family::BuildCtx;
use crate::hostmem;
use crate::lockstep::{self, BootGate, FollowerConfig, FollowerExit, Rendezvous, TpTimeouts};
use crate::pack::PackError;
use crate::plan::plan_memory_cached_windowed;
use crate::serve::{gib, interactive_prefix_config_from_env, memory_options, ServeOptions, TpOptions};
use gb_cuda::moe::ExpertOwnership;
use gb_cuda::nccl::{NcclComm, NcclUniqueId};
use gb_engine::{boot, Drafter, Engine, EngineConfig, NoDrafts, PrefillPolicy, ServeEngine};
use gb_exl3::checkpoint::Checkpoint;
use gb_sched::{Admission, CostModel, ModelMemory};
use gb_spec::CopyDrafter;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The timeouts of a group from its options.
pub fn timeouts(tp: &TpOptions) -> TpTimeouts {
    TpTimeouts {
        connect: tp.connect_timeout,
        ready: tp.ready_timeout,
        peer: tp.peer_timeout,
        write: tp.peer_timeout,
        ..TpTimeouts::default()
    }
}

/// What a GLM group cannot run yet, refused by name before the rendezvous (rank 0 before it binds, a follower before it connects).
pub fn check_tp_options(o: &ServeOptions) -> Result<(), ModelError> {
    let Some(t) = o.tp.as_ref().filter(|t| t.is_group()) else { return Ok(()) };
    let mut why: Vec<String> = Vec::new();
    if t.ranks > tp::MAX_TP_RANKS {
        why.push(format!("--tp-ranks {}: a GLM group has at most {} ranks", t.ranks, tp::MAX_TP_RANKS));
    }
    // images: rank 0's (its tower encodes them and the rows travel in each chunk's call); a follower never holds the tower
    if o.vision.is_some() && t.rank >= 1 {
        why.push(format!(
            "images on rank {} (start it with --no-vision: only rank 0 runs the vision tower and sends the rows)",
            t.rank
        ));
    }
    if o.slots_auto {
        why.push(
            "a slot count left to the planner (pass --slots N, the same on every rank: the group plans before it loads)".into(),
        );
    }
    if o.kv_tokens.is_none() {
        why.push("a KV pool left to the planner (pass --kv-tokens N, the same on every rank)".into());
    }
    if why.is_empty() {
        Ok(())
    } else {
        Err(ModelError::Config(format!("a GLM tensor-parallel group does not support: {}", why.join("; "))))
    }
}

/// The group's communicator, kept for the abort hooks.
static GROUP_COMM: Mutex<Option<Arc<dyn TpComm>>> = Mutex::new(None);

/// Aborts this process's communicator, if one was made (a collective stuck on a lost peer returns). The leader's `on_group_lost` hook
/// and the follower's `abort_hook`.
pub fn abort_group_comm() {
    let c = GROUP_COMM.lock().unwrap_or_else(|p| p.into_inner()).clone();
    if let Some(c) = c {
        c.abort();
    }
}

/// `REXL3_GLM_TP_EXPERTS`: how the ranks of a sharded group hold the routed experts. `width` (unset: the default) gives every rank every
/// expert at its half of the intermediate width (`gb_cuda::moe_width`: gate/up columns, down K rows; equal bytes a rank and token, the
/// single node's decode bits); `range` keeps whole experts by `ExpertOwnership::range` (0..144 | 144..288), the layout before it, for an
/// A/B. Rank 0 sends its rule in the flags byte of the bootstrap, and a follower with another refuses to join.
pub const EXPERTS_ENV: &str = "REXL3_GLM_TP_EXPERTS";

/// The rule of [`EXPERTS_ENV`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertSplit {
    /// Whole experts, `ExpertOwnership::range`.
    Range,
    /// Every expert at the rank's share of its width.
    Width,
}

impl ExpertSplit {
    /// The rule of this process (refused by name when malformed).
    pub fn from_env() -> Result<Self, ModelError> {
        match std::env::var(EXPERTS_ENV).as_deref() {
            Err(_) | Ok("width") => Ok(Self::Width),
            Ok("range") => Ok(Self::Range),
            Ok(v) => Err(ModelError::Config(format!(
                "{EXPERTS_ENV}={v}: `width` (every expert's intermediate columns split between the ranks, the default) or `range` \
                 (whole experts by id range)"
            ))),
        }
    }

    /// The flags byte rank 0 appends to the NCCL unique id (`1` = range, `2` = width).
    fn flag(self) -> u8 {
        match self {
            Self::Range => 1,
            Self::Width => 2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Range => "range",
            Self::Width => "width",
        }
    }
}

/// The communicator (`NcclTp`), bootstrapped over the rendezvous: rank 0 offers the unique id and the flags byte (its [`ExpertSplit`]),
/// every follower checks it before it acknowledges, nobody joins a communicator another rank refused. An empty offer
/// (`REXL3_TP_COMM=none`): no communicator.
pub fn make_comm(topo: TpTopology, rv: &mut Rendezvous<'_, GlmCall>) -> Result<Option<Arc<dyn TpComm>>, ModelError> {
    default_nccl_channels();
    let n = NcclUniqueId::BYTES;
    let split = ExpertSplit::from_env()?;
    let flag = split.flag();
    let offer = if topo.rank == 0 {
        if !sharded() {
            Some(Vec::new())
        } else {
            let id = NcclComm::new_unique_id().map_err(|e| {
                ModelError::Config(format!("ncclGetUniqueId: {e} (is libnccl on LD_LIBRARY_PATH / REXL3_NCCL_LIB?)"))
            })?;
            let mut b = id.as_bytes().to_vec();
            b.push(flag);
            Some(b)
        }
    } else {
        None
    };
    let check = move |b: &[u8]| -> Result<(), String> {
        if b.is_empty() {
            return if sharded() {
                Err("rank 0 runs without a communicator (REXL3_TP_COMM=none) and this rank with one".into())
            } else {
                Ok(())
            };
        }
        if !sharded() {
            return Err("rank 0 runs with a communicator and this rank without (REXL3_TP_COMM=none)".into());
        }
        if b.len() != n + 1 {
            return Err(format!("the communicator's bootstrap is {} bytes, expected {} (the ranks run different builds)", b.len(), n + 1));
        }
        if b[n] != flag {
            return Err(format!(
                "rank 0 holds the routed experts by rule {} and this rank by {flag} ({}: {EXPERTS_ENV}; 1 = range, 2 = width)",
                b[n],
                split.name()
            ));
        }
        Ok(())
    };
    let bytes = rv.exchange_checked(offer, check)?;
    if bytes.is_empty() {
        eprintln!(
            "rexl3 event=tp_comm family=glm rank={} ranks={} status=none (replicated: every rank runs the whole model)",
            topo.rank, topo.ranks
        );
        return Ok(None);
    }
    let id = NcclUniqueId::from_bytes(&bytes[..n]).ok_or_else(|| ModelError::Config("the NCCL unique id is malformed".into()))?;
    gb_cuda::rt::init_context()?;
    let t0 = std::time::Instant::now();
    let comm = NcclTp::init(topo, &id).map_err(|e| {
        ModelError::Config(format!("rank {}: ncclCommInitRank failed: {e} (library {})", topo.rank, NcclComm::library_path()))
    })?;
    eprintln!(
        "rexl3 event=tp_comm family=glm rank={} ranks={} status=nccl experts={} init_s={:.2} env=\"{}\"",
        topo.rank,
        topo.ranks,
        split.name(),
        t0.elapsed().as_secs_f64(),
        gb_cuda::nccl::nccl_environment().join(" ")
    );
    // REXL3_TP_TRANSPORT: REXL3's own RoCE transport carries the exchange within its policy (the y_pairs sums up to
    // REXL3_ROCE_SUM_MAX_BYTES), NCCL the rest; both ranks' settings were compared at the Hello (the collective fact). `auto` (the
    // default) brings RoCE up if every rank can and stays on NCCL, by a named event, if any cannot; `roce` fails the boot instead; `nccl`
    // is the opt-out.
    use crate::families::nemotron_h_omni::tp_roce::{self, Connected, RoceTp, Transport};
    let setting = tp_roce::transport(DEFAULT_TRANSPORT).map_err(ModelError::Config)?;
    let comm: Arc<dyn TpComm> = match setting {
        Transport::Nccl => Arc::new(comm),
        Transport::Roce | Transport::Auto => {
            let t1 = std::time::Instant::now();
            let plan = roce_site_plan_here();
            let strict = setting == Transport::Roce;
            let up = RoceTp::connect_over_nccl_soft(comm, &plan, strict).map_err(|e| {
                ModelError::Config(format!("rank {}: the RoCE transport (REXL3_TP_TRANSPORT={}): {e}", topo.rank, setting.name()))
            })?;
            match up {
                Connected::Roce(r) => {
                    eprintln!(
                        "rexl3 event=tp_comm family=glm rank={} transport=roce setting={} init_s={:.2} sites={:?} {}",
                        topo.rank,
                        setting.name(),
                        t1.elapsed().as_secs_f64(),
                        plan.fact(),
                        r.fingerprint()
                    );
                    Arc::new(r)
                }
                Connected::Nccl { nccl, why } => {
                    eprintln!(
                        "rexl3 event=tp_comm family=glm rank={} transport=nccl setting=auto reason=\"{}\" (REXL3_TP_TRANSPORT=roce would refuse this boot, =nccl skips the attempt)",
                        topo.rank, why
                    );
                    Arc::new(nccl)
                }
            }
        }
    };
    *GROUP_COMM.lock().unwrap_or_else(|p| p.into_inner()) = Some(comm.clone());
    Ok(Some(comm))
}

/// The RoCE transport's sites (`REXL3_TP_TRANSPORT=roce`): one per size class of GLM's exchange, the in-place y_pairs sum of
/// `rows x topk x hidden` f32 (128 KiB a row on GLM-5.3-Flash), 1-8 rows of a decode window or an MTP verify; a bigger SUM (prefill
/// slices) is NCCL's by the transport's sum rule (`REXL3_ROCE_SUM_MAX_BYTES`, 1 MiB). Every RoCE boot registers these.
pub const ROCE_SITES: [usize; 4] = [128 << 10, 256 << 10, 512 << 10, 1 << 20];

/// Lane T: the step's sites past 1 MiB, for the windows whose in-place sum outgrows it (under the shared-expert split a row is 192 KiB:
/// a window of 6 rows and up, an 8-slot NextN verify of 32-40 rows). Registered up to the size that holds the largest window
/// (`alloc::WINDOW_ROWS` x [`STEP_ROW_BYTES`]), at most 8 MiB, the largest size MEASURED faster than NCCL's all-reduce (N3 + N2, in a
/// graph: 1.5 MiB 89.4 vs 117.0 us, 3 MiB 160.0 vs 199.0, 6 MiB 300.3 vs 330.9, 8 MiB 389.4 vs 428.1; T/TABLES2.md); the sum limit
/// rises with them (`SitePlan::with_sum_max`, in the collective fact as `summax=`). 4 x 14 MiB = 56 MiB of pinned memory a rank.
pub const ROCE_SITES_STEP_BULK: [usize; 3] = [2 << 20, 4 << 20, 8 << 20];

/// The largest per-row exchange of a window (lane T): the y_pairs sum with the shared expert's partials, (8 + 4) x 4096 x 4 B.
pub const STEP_ROW_BYTES: usize = 192 << 10;

/// The step's sites for windows of at most `window_rows` rows: [`ROCE_SITES`] and the bulk ones up to the first that holds
/// `window_rows` x [`STEP_ROW_BYTES`] (all of them past 8 MiB).
pub fn step_sites_for(window_rows: usize) -> Vec<usize> {
    let need = window_rows * STEP_ROW_BYTES;
    let mut v = ROCE_SITES.to_vec();
    for &c in &ROCE_SITES_STEP_BULK {
        if *v.last().unwrap() >= need {
            break;
        }
        v.push(c);
    }
    v
}

/// Lane T: the P2 part swap's own sites (`tp_roce::SiteUser::PartSwap`), one fp32 row a prompt row (16 KiB a row): 8 MiB (512 rows),
/// 32 MiB (2,048) and 64 MiB (4,096 = the P2c4096 chunk), eager prefill sites the P2 combine writes into in place
/// (`tp_roce::RoceTp::zero_copy_swap`); `REXL3_ROCE_MAX_BYTES=1048576` keeps the swap on NCCL. 4 x 104 MiB = 416 MiB of pinned host
/// memory a rank, registered only under the P2 contract (a Hello fact; the plan is in the collective fact, [`collective_fact`]).
pub const ROCE_SITES_PART_SWAP: [usize; 3] = [8 << 20, 32 << 20, 64 << 20];

/// Lane T, for P5b: the prefill KDA-out gather's own sites (`SiteUser::PrefillGather`: lane E's gathered heads, fp16 halves, 8 KiB a
/// row): 4 MiB (512 rows), 16 MiB (2,048) and 32 MiB (4,096). Under P2 they would share size classes with the part swap's (32 MiB at
/// 2,048 / 4,096 rows): separate sites keep each one's order and zero-copy parity. Registered only when P2 runs the KDA split
/// (`roce_site_plan`'s `kda_gather`, passed by `roce_site_plan_here` since Lane P's P5b): 4 x 52 MiB = 208 MiB a rank.
pub const ROCE_SITES_PREFILL_GATHER: [usize; 3] = [4 << 20, 16 << 20, 32 << 20];

/// The RoCE sites a rank registers, by user (`tp_roce::SitePlan`): the step's [`ROCE_SITES`] always; the part swap's under the P2
/// contract; the prefill gather's under P2 with the KDA split (`kda_gather`, P5b).
pub fn roce_site_plan(
    contract: super::contract::PrefillContract,
    kda_gather: bool,
) -> crate::families::nemotron_h_omni::tp_roce::SitePlan {
    use crate::families::nemotron_h_omni::tp_roce::{SitePlan, SiteUser};
    let step = step_sites_for(super::alloc::WINDOW_ROWS);
    let mut plan = SitePlan::step(&step);
    if let Some(&top) = step.last().filter(|&&c| c > 1 << 20) {
        plan = plan.with_sum_max(top);
    }
    if contract.p2_prefill(true) {
        plan = plan.with(SiteUser::PartSwap, &ROCE_SITES_PART_SWAP);
        if kda_gather {
            plan = plan.with(SiteUser::PrefillGather, &ROCE_SITES_PREFILL_GATHER);
        }
    }
    plan
}

/// This process's plan: the prefill gather's sites when this rank runs the KDA split (`REXL3_GLM_TP_SPLIT` with `kda`, a fact of the
/// Hello, so both ranks build the same regions) under P2 (Lane P's P5b: its prompt chunks gather the heads' core outputs).
pub fn roce_site_plan_here() -> crate::families::nemotron_h_omni::tp_roce::SitePlan {
    let kda_split = super::model::tp::split_from_env().is_ok_and(|b| b & super::model::tp::SPLIT_KDA != 0);
    roce_site_plan(super::contract::current(), kda_split)
}

/// The RoCE path counters of this rank's group (lane T: `tp_roce::RoceTp::paths_line`), or nothing on NCCL alone.
pub fn roce_paths_line() -> String {
    let c = GROUP_COMM.lock().unwrap_or_else(|p| p.into_inner()).clone();
    c.as_ref().and_then(|c| c.roce_tp().map(|t| t.paths_line())).unwrap_or_default()
}

/// The transport a GLM process runs when `REXL3_TP_TRANSPORT` is unset (lane T, w10): RoCE if every rank can, else NCCL.
pub const DEFAULT_TRANSPORT: crate::families::nemotron_h_omni::tp_roce::Transport =
    crate::families::nemotron_h_omni::tp_roce::Transport::Auto;

/// The collective fact of the Hello: the policy and transport (`tp_comm::collective_fingerprint`), and with `REXL3_TP_TRANSPORT=roce`
/// the RoCE site list this rank will register (lane T: a function of the prefill contract), so two ranks that would build different
/// regions are refused by name before NCCL. Under `auto` (the default) the same, behind `+transport=auto`: the SETTING and the plan
/// are what the two ranks must share; whether RoCE then comes up is decided after the communicator exists (the probe and the
/// per-step verdicts of `RoceTp::connect_over_nccl_soft`) and named by `event=tp_comm transport=`. NCCL alone: unchanged.
pub fn collective_fact() -> String {
    use crate::families::nemotron_h_omni::tp_roce;
    // an invalid setting is already `invalid(..)` in the shared fingerprint, which never equals a valid one
    let setting = tp_roce::transport(DEFAULT_TRANSPORT).unwrap_or(tp_roce::Transport::Nccl);
    collective_fact_for(&roce_site_plan_here(), setting)
}

fn collective_fact_for(
    plan: &crate::families::nemotron_h_omni::tp_roce::SitePlan,
    setting: crate::families::nemotron_h_omni::tp_roce::Transport,
) -> String {
    use crate::families::nemotron_h_omni::tp_roce::{self, Transport};
    // `roce` and `nccl` come from the shared fingerprint, which adds the RoCE rule under `roce` only
    let base = crate::families::nemotron_h_omni::tp_comm::collective_fingerprint();
    match setting {
        Transport::Nccl => base,
        Transport::Roce => format!("{base},glm_sites={}", plan.fact()),
        Transport::Auto => format!("{base}+transport=auto,{},glm_sites={}", tp_roce::policy_fingerprint(), plan.fact()),
    }
}

/// The transport setting a collective fact names (`auto`, `roce` or `nccl`): for the Hello's refusal text.
pub fn transport_of_fact(fact: &str) -> &'static str {
    if fact.contains("+transport=auto,") {
        "auto"
    } else if fact.contains(",glm_sites=") {
        "roce"
    } else {
        "nccl"
    }
}

/// What a rank planned before its load.
pub struct RankPlan {
    pub cfg: GlmConfig,
    pub ck: Checkpoint,
    /// The pack's fingerprint (`check_pack`), the prefix cache's namespace with the NextN flag (`prefix::namespace_of`); empty from
    /// [`plan_from`], which does not check the pack.
    pub fingerprint: String,
    /// The prefix cache this rank planned (`docs/GLM_TP.md` 4.2): its snapshot slots of the rank's own state and the pages they may keep,
    /// on top of the KV pool admission hands out (`mo.kv_blocks` holds both). A Hello fact: the arena rank 0 forwards in `EnableSnapshots`.
    pub cache: gb_engine::CacheBudget,
    /// The plan's admitted KV pool (pages and tokens, without the cache's pages) and what it left spare: the single node's `kv_plan` and
    /// `prefix_cache_plan` boot lines, which rank 0 prints so a recipe's markers read one format on every topology.
    pub kv_admitted: (usize, u64),
    pub spare_bytes: u64,
    /// The pack mapped for this rank: with `own`, only those routed experts are placed (and so read and allocated).
    pub pack: GlmPack,
    /// This rank's share of the routed experts (`None`: no communicator, the whole model).
    pub own: Option<ExpertOwnership>,
    pub mo: ModelOptions,
    pub chunk: usize,
    pub admission: Admission,
    pub weight_bytes: u64,
    /// Rank 0 serving images: the tower's weights and scratch, out of what the pool is planned from (0 on a follower).
    pub tower_bytes: u64,
    /// Rank 0 serving images: the features reserve held back from the pool (the encoder's budget; 0 on a follower).
    pub vision_reserve: u64,
    pub free_after_weights: u64,
    pub scan_s: f64,
    /// The NextN layer is loaded (`--spec mtp|auto` with drafting on, a pack that has it, no DFlash2 draft): its regions (the rank's share
    /// of its experts) are in `weight_bytes`, its cache in the plan's KV bytes per token.
    pub mtp: bool,
    /// The DFlash2 drafter this rank loads (rank 0 with `--draft DIR`): its bytes are out of what the pool is planned from.
    pub dflash: Option<super::dflash::DflashPlan>,
    /// The DFlash2 drafter's tapped layers (`dflash::tap_layers`: on every rank of a drafting group, a fact of the Hello) and, when this
    /// rank relays them under the sequence-parallel residual (`dflash::relay_bytes`), the bytes of the relay's stage.
    pub taps: Vec<usize>,
    pub relay_bytes: u64,
}

/// Whether a rank (or one node) loads the NextN layer: the same rule as `engine::build_engine`. A DFlash2 draft (`--draft DIR`) drafts in
/// its place, so with one the layer is neither loaded nor given a cache, on every rank.
pub fn mtp_wanted(o: &ServeOptions, cfg: &GlmConfig) -> bool {
    super::MTP_BUILT && o.drafting && o.spec.uses_mtp() && cfg.has_mtp() && !super::dflash::wanted(o)
}

/// The plan of a rank: the pack scanned and checked, the memory planned from what is free now minus the pack's text weights, at the
/// pinned slot count and KV cap (`check_tp_options`).
pub fn plan_rank(o: &ServeOptions) -> Result<RankPlan, ModelError> {
    check_options(o)?;
    let cfg = GlmConfig::from_dir(&o.model_dir).map_err(ModelError::Config)?;
    let t = std::time::Instant::now();
    let ck = gb_exl3::checkpoint::scan(&o.model_dir, false).map_err(|e| ModelError::Pack(PackError::Io(e.to_string())))?;
    let scan_s = t.elapsed().as_secs_f64();
    let fingerprint = check_pack(&ck, o.allow_unvalidated_pack)?;
    crate::shapes::check_driver(&gb_cuda::rt::versions()).map_err(ModelError::Config)?;
    gb_cuda::rt::init_context()?;
    let (cuda_free, _) = gb_cuda::rt::mem_info()?;
    let host = hostmem::MemInfo::read();
    let free_now = hostmem::planning_free(cuda_free as u64, host.as_ref());
    let own = rank_ownership(o, &cfg)?;
    Ok(RankPlan { fingerprint, ..plan_from(o, cfg, ck, free_now, scan_s, own)? })
}

/// This rank's share of the routed experts when the group has a communicator: `ExpertOwnership::range` of its place, or every expert
/// under the width split ([`ExpertSplit::Width`]: the rank holds them all at its share of the width, [`rank_expert_width`]); `None`
/// otherwise (`REXL3_TP_COMM=none`: every rank holds every expert).
pub fn rank_ownership(o: &ServeOptions, cfg: &GlmConfig) -> Result<Option<ExpertOwnership>, ModelError> {
    let Some(t) = o.tp.as_ref().filter(|t| t.is_group()) else { return Ok(None) };
    if !sharded() {
        return Ok(None);
    }
    let experts = alloc::moe_dims(cfg).experts;
    if ExpertSplit::from_env()? == ExpertSplit::Width {
        return Ok(Some(ExpertOwnership::all(experts)));
    }
    ExpertOwnership::range(experts, t.rank, t.ranks)
        .map(Some)
        .map_err(|e| ModelError::Config(format!("expert ownership of rank {} of {}: {e}", t.rank, t.ranks)))
}

/// The place `(rank, ranks)` of a rank of a sharded group under the expert width split ([`EXPERTS_ENV`] `width`, the default): it maps,
/// plans and loads its columns of every routed expert (`GlmConfig::with_expert_width`). `None` otherwise.
pub fn rank_expert_width(o: &ServeOptions) -> Result<Option<(usize, usize)>, ModelError> {
    let Some(t) = o.tp.as_ref().filter(|t| t.is_group()) else { return Ok(None) };
    Ok((sharded() && ExpertSplit::from_env()? == ExpertSplit::Width).then_some((t.rank, t.ranks)))
}

/// Bytes of the device regions a rank allocates for `pack` (mapped for its share): the global region and every trunk layer's trunk and
/// expert regions; the NextN layer (not loaded on a rank: `--spec none`) and the vision tower (`--no-vision`) are not counted.
pub fn rank_weight_bytes(pack: &GlmPack) -> u64 {
    rank_weight_bytes_with(pack, false)
}

/// [`rank_weight_bytes`], with the NextN layer's regions (its trunk, replicated, and the rank's share of its experts) when `mtp`.
pub fn rank_weight_bytes_with(pack: &GlmPack, mtp: bool) -> u64 {
    let layers = pack.config.layers.len();
    let mtp_layer = (mtp && pack.config.has_mtp()).then(|| pack.config.mtp_layer_index(0));
    pack.plan.select_regions(super::load::keep_prefix_mtp(layers, mtp_layer)).regions.iter().map(|r| r.bytes).sum()
}

/// The place `(rank, ranks)` of a rank that splits the KDA heads with its peer (lane E Phase 2: `REXL3_GLM_TP_SPLIT` with `kda`, a sharded
/// group): it maps, plans and allocates only its heads' qkv strips and state (the rank strips, `KdaConfig::rank`). `None` otherwise.
pub fn rank_kda_split(o: &ServeOptions) -> Result<Option<(usize, usize)>, ModelError> {
    let Some(t) = o.tp.as_ref().filter(|t| t.is_group()) else { return Ok(None) };
    let bits = super::model::tp::split_from_env().map_err(ModelError::Config)?;
    // a split the prefill contract cannot run is refused here, before the load (`enable_head_split` applies the same rule after it)
    let contract = std::env::var("REXL3_GLM_PREFILL_CONTRACT").ok();
    if let Some(why) = super::model::tp::split_refused_under_contract(bits, contract.as_deref()) {
        return Err(ModelError::Config(why));
    }
    Ok((sharded() && bits & super::model::tp::SPLIT_KDA != 0).then_some((t.rank, t.ranks)))
}

/// [`plan_rank`] at a given free figure (before the load): the planner's half, pure (the CPU check of the cluster file's flags). A rank of
/// the expert width split ([`rank_expert_width`]) plans with its share in the configuration (`own` is then every expert).
pub fn plan_from(
    o: &ServeOptions,
    cfg: GlmConfig,
    ck: Checkpoint,
    free_now: u64,
    scan_s: f64,
    own: Option<ExpertOwnership>,
) -> Result<RankPlan, ModelError> {
    let cfg = match rank_expert_width(o)? {
        Some((rank, ranks)) => cfg.with_expert_width(rank, ranks).map_err(ModelError::Config)?,
        None => cfg,
    };
    plan_from_split(o, cfg, ck, free_now, scan_s, own, rank_kda_split(o)?)
}

/// [`plan_from`] for a rank of the KDA head split at `kda_split` (`(rank, ranks)`, [`rank_kda_split`]): the configuration becomes the
/// rank's (`GlmConfig::with_kda_rank`), so the pack is mapped with its heads' qkv strips only (the weight bytes), and the state, records
/// and allocations are its heads' (the memory profile, `alloc::dims_of`). `None`: [`plan_from`] without the split, unchanged.
pub fn plan_from_split(
    o: &ServeOptions,
    cfg: GlmConfig,
    ck: Checkpoint,
    free_now: u64,
    scan_s: f64,
    own: Option<ExpertOwnership>,
    kda_split: Option<(usize, usize)>,
) -> Result<RankPlan, ModelError> {
    let cfg = match kda_split {
        Some((rank, ranks)) => cfg.with_kda_rank(rank, ranks).map_err(ModelError::Config)?,
        None => cfg,
    };
    if cfg.ffn.width.is_some() && !own.as_ref().is_some_and(|o| o.is_all()) {
        return Err(ModelError::Config("a rank of the expert width split maps every expert (at its share of the width)".into()));
    }
    let mtp = mtp_wanted(o, &cfg);
    let mem_opts = memory::engine_options(&memory_options(o, mtp));
    let profile = memory::memory_profile(&cfg, &mem_opts).map_err(|e| ModelError::Config(e.to_string()))?;
    let inv = super::PackInventory::from_checkpoint(&ck).map_err(ModelError::Config)?;
    // the tower is bound only where it is loaded (rank 0 serving images); a follower and a `--no-vision` rank claim and skip it
    let all = ExpertOwnership::all(cfg.ffn.routed_experts);
    let pack = GlmPack::from_inventory_with(&inv, cfg.clone(), own.as_ref().unwrap_or(&all), o.vision.is_some())?;
    let weight_bytes = rank_weight_bytes_with(&pack, mtp);
    // rank 0 serving images: the tower and its scratch are out of the room, the features of the images in flight held back (a follower: 0)
    let (tower_bytes, vision_reserve) = tp_vision::planned_vision(o, &pack, &profile, mem_opts.vision)?;
    let chunk = alloc::prefill_chunk_for(o.prefill_chunk);
    // rank 0 drafting with DFlash2: the drafter (weights, rings, scratch, the trunk's taps) is out of the room too (a follower: none)
    let dflash = super::dflash::plan(o, &cfg, o.slots, chunk)?;
    let draft_bytes = dflash.as_ref().map_or(0, |d| d.bytes);
    // a drafting group at TP=2 under P2: both ranks relay the taps of the follower's rows (`model::dflash::TapRelay`)
    let taps = super::dflash::tap_layers(o)?;
    let relay_bytes = super::dflash::relay_bytes(o, &cfg, chunk);
    let free = free_now.saturating_sub(weight_bytes + tower_bytes + draft_bytes + relay_bytes);
    let kv_cap = o.kv_tokens.unwrap_or(o.max_model_len) as u64;
    // the prefix cache as one node plans it (`engine::build_engine`): a snapshot slot is THIS rank's state (its KDA heads under the split,
    // `cfg` is the rank's), the cache carved out before the KV pool is sized (an explicit size) or out of what the plan leaves (`Auto`)
    let snapshot_bytes = super::prefix::snapshot_bytes(&cfg, mtp).map_err(ModelError::Config)?;
    let plan = plan_memory_cached_windowed(
        &profile,
        free,
        o.floor_bytes,
        o.slots,
        kv_cap,
        vision_reserve,
        o.prefix_cache.request(),
        snapshot_bytes,
        0,
        o.max_model_len as u64,
    )
    .map_err(|e| ModelError::Config(format!("this rank cannot plan {} slots: {e} (free {:.1} GiB after the pack's {:.1} GiB, the vision tower's {:.2} GiB and the drafter's {:.2} GiB, vision reserve {:.2} GiB)", o.slots, gib(free), gib(weight_bytes), gib(tower_bytes), gib(draft_bytes), gib(vision_reserve))))?;
    let mm = ModelMemory::from_profile(&profile, 0, 0);
    let admission = plan.admission(mm, o.slots, o.max_model_len).map_err(|e| ModelError::Config(e.to_string()))?;
    // the pool holds the pages the cache may keep on top of the pages admission hands out
    let kv_blocks = plan.kv_blocks + plan.cache.page_blocks;
    let mo = ModelOptions { slots: o.slots, kv_blocks, max_context: o.max_model_len, layers: cfg.layers.len() };
    Ok(RankPlan {
        cfg,
        ck,
        fingerprint: String::new(),
        cache: plan.cache,
        kv_admitted: (plan.kv_blocks, plan.kv_tokens),
        spare_bytes: plan.spare_bytes,
        pack,
        own,
        mo,
        chunk,
        admission,
        weight_bytes,
        tower_bytes,
        vision_reserve,
        free_after_weights: free,
        scan_s,
        mtp,
        dflash,
        taps,
        relay_bytes,
    })
}

/// Loads a rank: the text trunk and this rank's share of the routed experts (`pack`, mapped with `own`), the model with `mo` and
/// `chunk`, its allocations checked against the plan, and (with a communicator) the rank path. Without one: the whole single-node model
/// (`pack` mapped whole).
#[allow(clippy::too_many_arguments)]
pub fn load_rank(
    dir: &Path,
    cfg: &GlmConfig,
    pack: GlmPack,
    own: Option<&ExpertOwnership>,
    mo: &ModelOptions,
    chunk: usize,
    floor: u64,
    scan_s: f64,
    comm: Option<Arc<dyn TpComm>>,
) -> Result<GlmRank, ModelError> {
    load_rank_with(dir, cfg, pack, own, mo, chunk, floor, scan_s, comm, None)
}

/// [`load_rank`] with the NextN layer when `mtp` is `Some(setting)` (the draft head's vocabulary, `engine::mtp_vocab_setting`: the
/// restricted head is built from the same strips on every rank and chosen by the same mode, as one node builds it).
#[allow(clippy::too_many_arguments)]
pub fn load_rank_with(
    dir: &Path,
    cfg: &GlmConfig,
    pack: GlmPack,
    own: Option<&ExpertOwnership>,
    mo: &ModelOptions,
    chunk: usize,
    floor: u64,
    scan_s: f64,
    comm: Option<Arc<dyn TpComm>>,
    mtp: Option<Option<(usize, f32)>>,
) -> Result<GlmRank, ModelError> {
    let t0 = std::time::Instant::now();
    if comm.is_some() != own.is_some() {
        return Err(ModelError::Config(format!(
            "this rank was mapped {} but the group {} a communicator",
            if own.is_some() { "for a share of the routed experts" } else { "whole" },
            if comm.is_some() { "has" } else { "has no" }
        )));
    }
    let booted = boot_mapped_with(dir, pack, Some(mo.layers), mtp.is_some(), &LoadOptions::from_env(floor), scan_s)?;
    eprintln!("rexl3 event=tp_rank_regions family=glm gib={:.3}", gib(booted.device.regions.bytes()));
    // the single node's load line, so a recipe's boot markers read one log format on every topology (`engine::build_engine`)
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
    let mut m = GlmModel::new_chunked(cfg.clone(), booted.device, mo, chunk)?;
    let ap = AllocPlan::new_chunked(
        cfg,
        &AllocParams {
            slots: mo.slots,
            kv_blocks: mo.kv_blocks,
            max_context: mo.max_context,
            max_rows: alloc::MAX_ROWS,
            window_rows: WINDOW_ROWS,
            mtp: m.has_mtp(),
        },
        chunk,
    )
    .map_err(ModelError::Config)?;
    m.check_allocations(&ap).map_err(|e| ModelError::Config(format!("the memory model and the allocations disagree: {e}")))?;
    let sharded = comm.is_some();
    if let Some(comm) = comm {
        let topo = comm.topology();
        let experts = alloc::moe_dims(cfg).experts;
        let own = own.cloned().ok_or_else(|| ModelError::Config("a rank with a communicator has no expert share".into()))?;
        // whole experts: the range of the rank's place; the width split: every expert, at the share `cfg` names (`enable_tp` checks it)
        let want = match cfg.ffn.width {
            Some(_) => ExpertOwnership::all(experts),
            None => ExpertOwnership::range(experts, topo.rank, topo.ranks)
                .map_err(|e| ModelError::Config(format!("expert ownership of rank {} of {}: {e}", topo.rank, topo.ranks)))?,
        };
        if own != want {
            return Err(ModelError::Config(format!(
                "this rank was mapped for {} experts, but rank {} of {} owns {}",
                own.local(),
                topo.rank,
                topo.ranks,
                want.local()
            )));
        }
        match cfg.ffn.width.and_then(|w| w.cols(cfg.ffn.moe_intermediate)) {
            Some((c0, c1)) => eprintln!(
                "rexl3 event=tp_glm_ownership rank={} ranks={} split=width experts=0..={} of {experts} columns={c0}..{c1} of {}",
                topo.rank,
                topo.ranks,
                experts - 1,
                cfg.ffn.moe_intermediate
            ),
            None => eprintln!(
                "rexl3 event=tp_glm_ownership rank={} ranks={} split=range experts={}..={} of {experts}",
                topo.rank,
                topo.ranks,
                own.owned().first().copied().unwrap_or(0),
                own.owned().last().copied().unwrap_or(0)
            ),
        }
        m.enable_tp(comm, own)?;
        // lane E Phase 2: the replicated weights this rank splits with its peer (`REXL3_GLM_TP_SPLIT`, a fact of the Hello)
        m.enable_head_split(super::model::tp::split_from_env().map_err(ModelError::Config)?)?;
    }
    if let Some(Some((strips, share))) = mtp {
        let ranges = super::model::vocab_ranges(strips, cfg.vocab);
        let t = std::time::Instant::now();
        let bytes = m.mtp_restrict_vocab(&ranges)?;
        m.set_mtp_vocab(if share > 0.0 { super::model::VocabMode::Adaptive(share) } else { super::model::VocabMode::Restricted });
        let kept: usize = ranges.iter().map(|&(lo, hi)| hi - lo).sum();
        eprintln!(
            "rexl3 event=mtp_vocab strips={kept} of={} extra_mib={:.0} min_share={share} build_s={:.2}",
            cfg.vocab / 128,
            bytes as f64 / 1048576.0,
            t.elapsed().as_secs_f64()
        );
    }
    eprintln!("rexl3 event=tp_rank_loaded family=glm sharded={sharded} load_s={:.1}", t0.elapsed().as_secs_f64());
    Ok(GlmRank::new(m))
}

/// The gate of this rank from its plan.
fn gate_of(o: &ServeOptions, p: &RankPlan, tp: &TpOptions) -> Result<BootGate<GlmCall>, ModelError> {
    let pack = tp::pack_identity(&o.model_dir).map_err(ModelError::Config)?;
    let facts = facts_of(&p.cfg, tp.ranks, p.mtp)?
        .with_vision(tp_vision::holds_tower(o, &p.pack))
        .with_prefill_small(adaptive_small(p.chunk)?, p.chunk)
        .with_cache(crate::families::nemotron_h_omni::tp_runtime::cache_request_label(o.prefix_cache), &p.cache)
        .with_taps(&p.taps);
    Ok(tp::gate(pack, &p.mo, p.chunk, facts).at(tp.rank, tp.ranks))
}

/// The draft head's vocabulary setting of this process (`REXL3_MTP_VOCAB`), refused by name when malformed.
pub fn vocab_setting() -> Result<Option<(usize, f32)>, ModelError> {
    super::engine::mtp_vocab_setting(std::env::var("REXL3_MTP_VOCAB").ok().as_deref()).map_err(ModelError::Config)
}

/// `REXL3_GLM_ADAPTIVE_PREFILL`: the rows of rank 0's prefill slice while a sequence is decoding, when the kit's `--prefill-chunk` is
/// larger (the chunk is then the slice when none is: lane E Phase 1, the owner's adaptive chunk of 2026-10-18). Unset: the default chunk
/// (512, `alloc::DEFAULT_PREFILL_CHUNK`); `0` / `off`: one size (the chunk). A number is taken as `--prefill-chunk` takes one.
pub const ADAPTIVE_ENV: &str = "REXL3_GLM_ADAPTIVE_PREFILL";

/// The gather window of a P2 group's engine (`REXL3_PREFILL_GATHER_MS` overrides): a request that wakes the idle engine waits this many
/// milliseconds for the requests sent with it, whose prompts then share the first forward.
pub const PREFILL_GATHER_MS: u64 = 15;

/// A serving group's ranks capture a window shape at its first sighting (`REXL3_GLM_GRAPH_AT=1`: at its second, the model's default):
/// on a fresh server the windows of a new concurrency then never run eagerly (MEASURED on pair B, C = 8 from a fresh boot: 93.0 ->
/// 95.1 tok/s; the capture is ~8 ms a shape, under a second for a whole sweep).
pub const CAPTURE_FIRST: bool = true;

/// The adaptive slice's small rows for a rank whose chunk is `chunk` (`0`: one size), refused by name when malformed.
pub fn adaptive_small(chunk: usize) -> Result<usize, ModelError> {
    let small = match std::env::var(ADAPTIVE_ENV).as_deref() {
        Err(_) => alloc::DEFAULT_PREFILL_CHUNK,
        Ok("0" | "off" | "false" | "no") => 0,
        Ok(v) => v.trim().parse::<usize>().map(alloc::prefill_chunk_for).map_err(|_| {
            ModelError::Config(format!("{ADAPTIVE_ENV}={v}: the rows of a slice while a sequence decodes (a number), or 0 / off"))
        })?,
    };
    let small = if small == 0 || small >= chunk { 0 } else { small };
    // under P2 the adaptive slice is a whole number of 16-row KDA chunks too (refused by name, never rounded silently)
    let k = gb_cuda::kda::P2_CHUNK_ROWS;
    if small % k != 0 && super::contract::current() == super::contract::PrefillContract::P2 {
        return Err(ModelError::Config(format!(
            "{ADAPTIVE_ENV} gives a {small}-row slice under {}=p2: a P2 slice is a multiple of {k} rows (P2's KDA chunk)",
            super::contract::CONTRACT_VAR
        )));
    }
    Ok(small)
}

/// The GLM facts of a rank from its options (before any weight is read): the ownership at `ranks`, and with the NextN layer the draft
/// depth cap and the head's vocabulary. The vision fact (leader-only) is set by the caller (`GlmFacts::with_vision`), the adaptive
/// slice by [`gate_of`].
pub fn facts_of(cfg: &GlmConfig, ranks: usize, mtp: bool) -> Result<GlmFacts, ModelError> {
    let f = GlmFacts::local(alloc::moe_dims(cfg).experts, ranks);
    let f = if cfg.ffn.width.is_some() { f.with_width_split(alloc::moe_dims(cfg).experts, ranks) } else { f };
    let f =
        if mtp { f.with_mtp(alloc::MTP_MAX_DEPTH, super::tp_mtp::mtp_vocab_fact(true, vocab_setting()?, cfg.vocab)) } else { f };
    // the graph settings of a sharded rank (its communicator is NCCL's, capturable); a replicated group (`REXL3_TP_COMM=none`) runs no
    // collective and keeps its single-node windows (no rank path): no graph fact of the exchange to agree on
    let graphs = sharded() && super::model::tp::rank_graphs(true);
    // the NextN draft calls' graphs (lane T): on by default at TP when the windows run graphs and the layer is loaded
    let draft = super::model::graphs::draft_graphs_setting().unwrap_or(true);
    // lane E Phase 2: the replicated weights this rank splits with its peer (`REXL3_GLM_TP_SPLIT`, two ranks)
    // (a replicated group, `REXL3_TP_COMM=none`, runs no collective and splits nothing: its fact says so)
    let split = if ranks == 2 && sharded() { super::model::tp::split_from_env().map_err(ModelError::Config)? } else { 0 };
    Ok(f.with_graphs(graphs, super::model::graphs::overlap_parts(), super::model::graphs::capture_at_first_or(CAPTURE_FIRST))
        .with_draft_graphs(draft)
        .with_split(split))
}

/// Rank 0: plans, accepts and gates the followers, makes the communicator, loads its rank, waits for every Ready and builds the engine
/// around the leader. `listener` was bound by `GlmFamily::build` after [`check_tp_options`].
pub fn build_leader(
    o: &ServeOptions,
    listener: &std::net::TcpListener,
    tp_o: &TpOptions,
) -> Result<Box<dyn ServeEngine>, ModelError> {
    let t = timeouts(tp_o);
    default_nccl_channels();
    let p = plan_rank(o)?;
    let gate = gate_of(o, &p, tp_o)?;
    // rank 0's speculation settings (`REXL3_MTP_DEPTH`, `REXL3_GLM_DRAFT_NOISE`), refused by name BEFORE the group loads
    let spec_cfg = if o.drafting && o.spec.drafts() {
        // a two-rank group prices its rounds with the tables measured on two ranks (lane T: `spec::VERIFY_MS_TP`)
        Some(super::spec::SpecConfig { two_ranks: tp_o.ranks == 2, ..super::spec::SpecConfig::from_env(o.spec)? })
    } else {
        None
    };
    eprintln!(
        "rexl3 event=tp_leader_waiting family=glm peer={} ranks={} slots={} kv_pages={} max_context={} prefill_rows={} weights_gib={:.2} vision={} tower_gib={:.2} vision_reserve_gib={:.2} free_after_gib={:.2} nccl=\"{}\" collective=\"{}\"",
        tp_o.peer,
        tp_o.ranks,
        gate.slots,
        gate.kv_pages,
        gate.max_context,
        gate.prefill_rows,
        gib(p.weight_bytes),
        gate.family.vision,
        gib(p.tower_bytes),
        gib(p.vision_reserve),
        gib(p.free_after_weights),
        gate.nccl,
        gate.family.collective
    );
    let links = lockstep::accept_group(listener, &gate, &t)?;
    let topo = TpTopology { rank: tp_o.rank, ranks: tp_o.ranks };
    let t0 = std::time::Instant::now();
    let (dir, floor) = (o.model_dir.clone(), o.floor_bytes);
    let mtp = if p.mtp { Some(vocab_setting()?) } else { None };
    // rank 0's vision tower is loaded inside the boot (after its text weights and its NextN layer, before its Ready), so a tower that
    // does not load fails the group's boot by name; a follower never loads one
    let mut tower = None;
    let leader = lockstep::boot_leader_group(
        links,
        &gate,
        &t,
        |rv| make_comm(topo, rv),
        |comm| {
            let mut rank =
                load_rank_with(&dir, &p.cfg, p.pack.clone(), p.own.as_ref(), &p.mo, p.chunk, floor, p.scan_s, comm, mtp)?;
            // rank 0's DFlash2 drafter, before its Ready: a drafter that does not load fails the group's boot by name
            if let Some(d) = &p.dflash {
                super::dflash::load(&mut rank.m, d)?;
            }
            if p.relay_bytes > 0 {
                rank.m.enable_tap_relay(&p.taps)?;
            }
            // after the drafter: the stash of a paused prompt forward keeps its tap rows too
            super::tp_part::enable(&mut rank.m)?;
            rank.m.set_graph_capture_first(super::model::graphs::capture_at_first_or(CAPTURE_FIRST));
            tower = tp_vision::load_leader_tower(o, &p.pack)?;
            Ok(rank)
        },
    )?;
    leader.on_group_lost(Box::new(abort_group_comm));
    boot::mark("tp_group_ready", None);
    eprintln!("rexl3 event=tp_group_ready family=glm ranks={} boot_s={:.1}", tp_o.ranks, t0.elapsed().as_secs_f64());
    // speculation as one node runs it (`engine::build_engine`): the backend's drafters with drafting on and a drafting `--spec`
    let mut backend = match spec_cfg {
        Some(c) => GlmTpBackend::with_spec(leader, p.chunk, p.mo.slots, c),
        None => GlmTpBackend::new(leader, p.chunk, p.mo.slots),
    };
    // the adaptive slice (the Hello fact `prefill_small`, agreed with every follower above): small slices while a sequence decodes
    let small = gate.family.prefill_small as usize;
    if small != 0 {
        backend = backend.with_adaptive(small);
    }
    // long slices in parts while a sequence decodes (`tp_part`; every rank holds the stash): replaces the adaptive slice; and
    // the rows of fresh short prompts one forward takes meanwhile
    let part_ms = super::tp_part::part_ms_from_env()?;
    let batch_rows = super::tp_part::batch_rows_from_env()?;
    backend = backend.with_parts(part_ms).with_batch_rows(batch_rows);
    eprintln!(
        "rexl3 event=tp_glm_levers graphs={} overlap={} graph_at_first={} prefill_rows={} prefill_small={} part_ms={} batch_rows={} (REXL3_GLM_TP_GRAPHS, REXL3_GLM_OVERLAP, REXL3_GLM_GRAPH_AT, --prefill-chunk, REXL3_GLM_ADAPTIVE_PREFILL, {}, {})",
        gate.family.graphs,
        gate.family.overlap,
        gate.family.graph_at_first,
        p.chunk,
        small,
        part_ms,
        batch_rows,
        super::tp_part::PART_MS_ENV,
        super::tp_part::BATCH_ROWS_ENV
    );
    if tower.is_some() {
        backend = backend.with_vision(TpVision::device(p.cfg.vision.as_ref().map(|v| v.image_token_id)));
    }
    // the prefix cache (`docs/GLM_TP.md` 4.2): rank 0's engine decides hits; every rank snapshots and restores its own state between
    // chunks. `with_prefix_cache` forwards `EnableSnapshots` (each rank allocates its own pool, rank 0's arena checked on every follower)
    eprintln!(
        "rexl3 event=kv_plan free_gib={:.1} kv_pages={} kv_tokens={} slots={} kv={}",
        gib(p.free_after_weights),
        p.kv_admitted.0,
        p.kv_admitted.1,
        p.mo.slots,
        p.cfg.latent_kv.name()
    );
    if p.cache.enabled() {
        eprintln!(
            "rexl3 event=prefix_cache_plan entries={} pages={} gib={:.2} spare_gib={:.1} reach_tokens={} window={} ranks={}",
            p.cache.snapshot_slots,
            p.cache.page_blocks,
            gib(p.cache.bytes()),
            gib(p.spare_bytes),
            p.cache.reach_tokens(),
            o.max_model_len,
            tp_o.ranks
        );
        boot::fact("prefix_cache", format!("{} entries, {:.2} GiB a rank", p.cache.snapshot_slots, gib(p.cache.bytes())));
        boot::fact("prefix_cache_reach_tokens", p.cache.reach_tokens().to_string());
        let ns = super::prefix::namespace_of(&p.fingerprint, p.mtp);
        backend = backend.with_prefix_cache(ns, p.cache)?;
    } else {
        eprintln!("rexl3 event=prefix_cache_plan entries=0 reason=\"off\" ranks={}", tp_o.ranks);
        boot::fact("prefix_cache", "off");
    }
    let own_budget =
        backend.spec.as_ref().is_some_and(|s| matches!(s.cfg.mode, gb_spec::SpecMode::Auto | gb_spec::SpecMode::Mtp));
    let batches = backend.batches();
    let engine = engine_over_budget(backend, p.admission, o.drafting, own_budget);
    // the backend puts the slices of several waiting prompts through one forward: every waiting prompt gets a slice a step
    // (the fewest rows left first), so prompts that arrive together are filled together (`REXL3_PREFILL_BURST` overrides), and a request
    // that wakes the idle engine waits a few milliseconds for the ones sent with it (`REXL3_PREFILL_GATHER_MS`)
    let engine = if batches {
        let policy = crate::costs::prefill_policy_from_env(PrefillPolicy::Burst { div: 1, cap: p.mo.slots.max(1) })?;
        let batching = crate::costs::prefill_batching_from_env()?;
        let gather = crate::costs::prefill_gather_from_env(PREFILL_GATHER_MS)?;
        eprintln!(
            "rexl3 event=prefill_policy policy={} batching={batching} gather_ms={}",
            format!("{policy:?}").to_lowercase(),
            gather.as_millis()
        );
        engine.with_prefill_policy(policy).with_prefill_batching(batching).with_gather_window(gather)
    } else {
        engine
    };
    let engine = if p.cache.enabled() { engine.with_prefix_cache(interactive_prefix_config_from_env()?) } else { engine };
    // the encoder is published only now that the engine exists: a request that carries an image cannot arrive before
    if let Some(t) = tower {
        tp_vision::publish_encoder(o, t, p.vision_reserve)?;
    }
    Ok(Box::new(engine))
}

/// The engine over rank 0's backend: the single-node GLM engine's configuration (`engine::engine_over` without a backend drafter: the
/// measured step table, and the engine's copy drafter per sequence when drafting is on).
pub fn engine_over<B: gb_engine::Backend>(backend: B, admission: Admission, drafting: bool) -> Engine<B> {
    engine_over_budget(backend, admission, drafting, false)
}

/// [`engine_over`]; with `own_budget` (`--spec auto|mtp`: the arbiter or the fixed depth decides) the flat row budget of
/// `engine::engine_over`, so the engine keeps every draft it is offered exactly as one node's does.
pub fn engine_over_budget<B: gb_engine::Backend>(
    backend: B,
    admission: Admission,
    drafting: bool,
    own_budget: bool,
) -> Engine<B> {
    let make: Box<dyn Fn() -> Box<dyn Drafter>> = if drafting {
        Box::new(|| Box::new(CopyDrafter::new(8, 7)) as Box<dyn Drafter>)
    } else {
        Box::new(|| Box::new(NoDrafts) as Box<dyn Drafter>)
    };
    let cost = if own_budget {
        CostModel::from_table(vec![(1, 1.0), (WINDOW_ROWS, 1.0)])
    } else {
        CostModel::from_table(STEP_MS.to_vec())
    };
    Engine::new(backend, admission, EngineConfig { max_rows: WINDOW_ROWS, max_drafts: 7, cost, drafting }, make)
}

/// A follower (rank 1): checks, connects, plans as rank 0 does, gates, loads with rank 0's options, says Ready, replays.
pub fn follow(ctx: &BuildCtx<'_>) -> Result<(), ModelError> {
    let o = ctx.options;
    let tp_o =
        o.tp.as_ref()
            .filter(|t| t.is_group() && t.rank >= 1)
            .ok_or_else(|| ModelError::Config("a tensor-parallel follower needs --tp-ranks 2 --tp-rank 1".into()))?;
    check_tp_options(o)?;
    default_nccl_channels();
    let t = timeouts(tp_o);
    let mut link = lockstep::connect_leader_as::<GlmCall>(tp_o.rank, &tp_o.peer, t.connect, t.write)?;
    let planned = plan_rank(o).and_then(|p| gate_of(o, &p, tp_o).map(|g| (p, g)));
    let (p, gate) = match planned {
        Ok(x) => x,
        Err(e) => {
            link.refuse(&format!("rank {} cannot plan: {e}", tp_o.rank));
            return Err(e);
        }
    };
    eprintln!(
        "rexl3 event=tp_follower_plan family=glm rank={} peer={} slots={} kv_pages={} max_context={} prefill_rows={} weights_gib={:.2} free_after_gib={:.2}",
        tp_o.rank,
        tp_o.peer,
        gate.slots,
        gate.kv_pages,
        gate.max_context,
        gate.prefill_rows,
        gib(p.weight_bytes),
        gib(p.free_after_weights)
    );
    eprintln!(
        "rexl3 event=tp_glm_levers graphs={} overlap={} graph_at_first={} prefill_rows={} prefill_small={} rank={}",
        gate.family.graphs,
        gate.family.overlap,
        gate.family.graph_at_first,
        gate.prefill_rows,
        gate.family.prefill_small,
        tp_o.rank
    );
    boot::mark("tp_follower_planned", None);
    let topo = TpTopology { rank: tp_o.rank, ranks: tp_o.ranks };
    let t0 = std::time::Instant::now();
    let (dir, floor) = (o.model_dir.clone(), o.floor_bytes);
    let mtp = if p.mtp { Some(vocab_setting()?) } else { None };
    let (rt, link) = lockstep::boot_follower(
        link,
        &gate,
        &t,
        |rv| make_comm(topo, rv),
        |leader_gate, comm| {
            let (mo, chunk) = tp::options_of(leader_gate);
            let mut rank = load_rank_with(&dir, &p.cfg, p.pack.clone(), p.own.as_ref(), &mo, chunk, floor, p.scan_s, comm, mtp)?;
            // the follower's half of the drafter's tap relay (the taps are a fact of the Hello: rank 0's are these)
            if p.relay_bytes > 0 {
                rank.m.enable_tap_relay(&p.taps)?;
            }
            super::tp_part::enable(&mut rank.m)?;
            rank.m.set_graph_capture_first(super::model::graphs::capture_at_first_or(CAPTURE_FIRST));
            Ok(rank)
        },
    )?;
    eprintln!("rexl3 event=tp_follower_ready family=glm boot_s={:.1}", t0.elapsed().as_secs_f64());
    boot::mark("tp_follower_ready", None);
    let mut fc = FollowerConfig::new(t).with_rank(tp_o.rank);
    fc.stop = crate::serve::follower_stop_flag();
    fc.abort_hook = Some(Box::new(abort_group_comm));
    let rank = tp_o.rank;
    fc.on_progress = Some(Box::new(move |s| {
        eprintln!(
            "rexl3 event=tp_follower_stats family=glm applied={} errors={} prefills={} windows={} rank={rank} {}",
            s.applied,
            s.errors,
            s.prefills,
            s.windows,
            tp::rank_graphs_line()
        );
    }));
    let exit = lockstep::run_follower(rt, link, fc);
    eprintln!("rexl3 event=tp_glm_graphs rank={} {} {}", tp_o.rank, tp::rank_graphs_line(), tp::rank_draft_graphs_line());
    eprintln!("rexl3 event=tp_roce_paths rank={} {}", tp_o.rank, roce_paths_line());
    match exit? {
        FollowerExit::Shutdown(why) => {
            eprintln!("rexl3 event=tp_follower_exit reason=\"leader shutdown: {why}\"");
            Ok(())
        }
        FollowerExit::Stopped => {
            eprintln!("rexl3 event=tp_follower_exit reason=\"stopped\"");
            Ok(())
        }
        FollowerExit::LeaderLost(why) => Err(ModelError::StateLost(format!("rank 0 was lost: {why}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::PrefixCacheOption;

    #[test]
    fn each_user_has_its_own_sites_and_the_prefill_ones_only_under_p2() {
        use super::super::contract::PrefillContract;
        use crate::families::nemotron_h_omni::tp_roce::SiteUser;
        let p1 = roce_site_plan(PrefillContract::P1, false);
        let p2 = roce_site_plan(PrefillContract::P2, false);
        let p5b = roce_site_plan(PrefillContract::P2, true);
        // the step: the decode sites and the bulk ones sized from the window (64 rows x 192 KiB = 12 MiB: all three, 2 / 4 / 8 MiB)
        let step: Vec<usize> = ROCE_SITES.iter().chain(ROCE_SITES_STEP_BULK.iter()).copied().collect();
        assert_eq!(p1.caps(), step);
        assert_eq!(step_sites_for(1), ROCE_SITES.to_vec(), "a 1-row window needs no bulk site");
        assert_eq!(step_sites_for(8), [&ROCE_SITES[..], &[2 << 20]].concat(), "8 rows x 192 KiB = 1.5 MiB: the 2 MiB site");
        assert_eq!(step_sites_for(32), step, "32 rows = 6 MiB: up to 8 MiB");
        assert_eq!(roce_site_plan(PrefillContract::P1, true), p1, "no prefill gather sites under P1");
        assert_eq!(p2.caps().len(), 10);
        assert_eq!(p2.ids()[SiteUser::PartSwap as usize], vec![7, 8, 9]);
        // the gather and the part swap share a size class (32 MiB) but not a site
        let ids = p5b.ids();
        assert_eq!(ids[SiteUser::PrefillGather as usize], vec![10, 11, 12]);
        assert_eq!(p5b.caps()[8], 32 << 20);
        assert_eq!(p5b.caps()[12], 32 << 20);
        // the region: 4 x the caps a rank (the step's bulk sites 56 MiB, the part swap 416, the gather 208)
        let mib = |p: &crate::families::nemotron_h_omni::tp_roce::SitePlan| (p.caps().iter().sum::<usize>() * 4) >> 20;
        let decode = (ROCE_SITES.iter().sum::<usize>() * 4) >> 20;
        assert_eq!((mib(&p1) - decode, mib(&p2) - mib(&p1), mib(&p5b) - mib(&p2)), (56, 416, 208));
        // the sum limit rises with the step's largest site
        assert_eq!(p1.sum_max_with(|_| None), Ok(8 << 20));
        assert_eq!(
            p1.sum_max_with(|k| (k == crate::families::nemotron_h_omni::tp_roce::SUM_MAX_ENV).then(|| "1048576".into())),
            Ok(1 << 20)
        );
        use crate::families::nemotron_h_omni::tp_roce::{self, Transport};
        let base = crate::families::nemotron_h_omni::tp_comm::collective_fingerprint();
        // NCCL alone: the fact is the old one whatever the plan (every existing Hello unchanged)
        assert_eq!(collective_fact_for(&p1, Transport::Nccl), base);
        assert_eq!(collective_fact_for(&p2, Transport::Nccl), base);
        let f1 = collective_fact_for(&p1, Transport::Roce);
        let f2 = collective_fact_for(&p2, Transport::Roce);
        let st = "step:131072/262144/524288/1048576/2097152/4194304/8388608";
        assert!(f1.ends_with(&format!(",glm_sites={st};summax=8388608")), "{f1}");
        assert!(f2.ends_with(&format!(",glm_sites={st};part_swap:8388608/33554432/67108864;summax=8388608")), "{f2}");
        assert!(collective_fact_for(&p5b, Transport::Roce).ends_with(";prefill_gather:4194304/16777216/33554432;summax=8388608"));
        assert_ne!(f1, f2);
        assert_eq!(f1, format!("{base},glm_sites={}", p1.fact()), "`roce` keeps the bytes it had");
        // `auto`: the setting, the RoCE rule and the plan, so ranks that differ in any of them are refused by name
        let a1 = collective_fact_for(&p1, Transport::Auto);
        let a2 = collective_fact_for(&p2, Transport::Auto);
        assert_eq!(a1, format!("{base}+transport=auto,{},glm_sites={}", tp_roce::policy_fingerprint(), p1.fact()));
        assert!(a1.contains(",min=1,max=sites,summax=1048576,glm_sites=step:"), "{a1}");
        assert_ne!(a1, a2);
        assert_ne!(a1, f1);
        assert_ne!(a1, base);
        let names: Vec<&str> = [&base, &f1, &a1, &a2].iter().map(|f| transport_of_fact(f)).collect();
        assert_eq!(names, ["nccl", "roce", "auto", "auto"]);
    }

    /// The Hello refuses a pair of ranks on different transport settings by name, one pair at a time; equal settings pass.
    #[test]
    fn ranks_on_different_transport_settings_are_refused_by_name() {
        use super::super::contract::PrefillContract;
        use crate::families::nemotron_h_omni::tp_roce::Transport;
        use crate::lockstep::FamilyCall;
        let plan = roce_site_plan(PrefillContract::P1, false);
        let facts = |t: Transport| GlmFacts { collective: collective_fact_for(&plan, t), ..GlmFacts::local(288, 2) };
        let check = |a: Transport, b: Transport| <GlmCall as FamilyCall>::check_facts(&facts(a), &facts(b), 1);
        for t in [Transport::Auto, Transport::Roce, Transport::Nccl] {
            assert!(check(t, t).is_empty(), "{t:?} against itself");
        }
        for (a, b) in [
            (Transport::Roce, Transport::Nccl),
            (Transport::Nccl, Transport::Roce),
            (Transport::Auto, Transport::Roce),
            (Transport::Auto, Transport::Nccl),
        ] {
            let why = check(a, b);
            assert!(
                why.iter().any(|l| l.starts_with("the transport (REXL3_TP_TRANSPORT")
                    && l.contains(&format!("rank 0 has {}, rank 1 has {}", a.name(), b.name()))),
                "{a:?} against {b:?}: {why:?}"
            );
        }
    }

    fn opts() -> ServeOptions {
        let mut o = ServeOptions::new("/nonexistent".into());
        let d = std::time::Duration::from_secs(1);
        o.tp = Some(TpOptions {
            ranks: 2,
            rank: 0,
            peer: "127.0.0.1:1".into(),
            connect_timeout: d,
            ready_timeout: d,
            peer_timeout: d,
        });
        o
    }

    /// The follower's real boot path against rank 0's real accept: a rank 1 that cannot plan (here: no pack) says so on the connection,
    /// and rank 0's accept fails at once naming it, instead of waiting out its connect timeout.
    #[test]
    fn a_follower_that_cannot_plan_is_refused_by_name_on_rank_zero() {
        let l = crate::lockstep::listen("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let d = std::time::Duration::from_secs(10);
        let mut o = ServeOptions::new("/nonexistent-glm-pack".into());
        o.tp = Some(TpOptions { ranks: 2, rank: 1, peer: addr.clone(), connect_timeout: d, ready_timeout: d, peer_timeout: d });
        o.slots_auto = false;
        o.kv_tokens = Some(8192);
        o.prefix_cache = PrefixCacheOption::Off;
        o.spec = gb_spec::SpecMode::None;
        o.vision = None;
        let follower = std::thread::spawn(move || follow(&BuildCtx { options: &o }));
        let mo = ModelOptions { slots: 1, kv_blocks: 8, max_context: 8192, layers: 45 };
        let facts = GlmFacts::local(288, 2);
        let gate = tp::gate("p".into(), &mo, 512, facts).at(0, 2);
        let t = TpTimeouts { connect: d, ready: d, peer: d, write: d, barrier: d, hang_exit: None, ..TpTimeouts::default() };
        let e = lockstep::accept_group(&l, &gate, &t).map(|_| ()).unwrap_err().to_string();
        assert!(e.contains("rank 1") && e.contains("cannot plan"), "{e}");
        let fe = follower.join().unwrap().unwrap_err().to_string();
        assert!(fe.contains("config") || fe.contains("nonexistent"), "{fe}");
    }

    /// The cluster file's flags (`docs/examples/cluster.glm-tp2.json`) against the real pack's scan at the cluster file's 108 GiB of free
    /// memory, for rank 0's share (46.32 GiB of regions; the whole pack's 90.59 GiB before ranks loaded only their share): the plan grants 4
    /// slots and the 32,768-token pool. CPU only; needs `REXL3_GLM_DIR`.
    #[test]
    #[ignore]
    fn the_cluster_files_flags_plan_on_the_real_pack_at_108_gib() {
        let Ok(dir) = std::env::var("REXL3_GLM_DIR") else {
            eprintln!("REXL3_GLM_DIR is not set: not run");
            return;
        };
        let mut o = ServeOptions::new(dir.into());
        o.slots = 4;
        o.slots_auto = false;
        o.max_model_len = 8192;
        o.kv_tokens = Some(32768);
        o.prefill_chunk = 512;
        o.prefix_cache = PrefixCacheOption::Off;
        o.vision = None;
        let cfg = GlmConfig::from_dir(&o.model_dir).unwrap();
        let ck = gb_exl3::checkpoint::scan(&o.model_dir, false).unwrap();
        let mtp_bytes: u64 = ck.groups.iter().filter(|(k, _)| k.contains("layers.45.")).map(|(_, g)| g.bytes).sum::<u64>()
            + ck.plain.iter().filter(|(k, _)| k.contains("layers.45.")).map(|(_, e)| e.nbytes()).sum::<u64>();
        let own = ExpertOwnership::range(288, 0, 2).unwrap();
        let p = plan_from_split(&o, cfg, ck, 108 << 30, 0.0, Some(own), None).unwrap();
        eprintln!(
            "weights {:.2} GiB (of which the NextN layer {:.2} GiB), free after {:.2} GiB, slots {}, kv pages {} ({} tokens), chunk {}",
            gib(p.weight_bytes),
            gib(mtp_bytes),
            gib(p.free_after_weights),
            p.mo.slots,
            p.mo.kv_blocks,
            p.mo.kv_blocks * 256,
            p.chunk
        );
        assert_eq!((p.mo.slots, p.chunk), (4, 512));
        assert!(p.mo.kv_blocks * 256 >= 32768, "{}", p.mo.kv_blocks);
    }

    /// The two-node prefix cache's plan (`docs/GLM_TP.md` 4.2) at the first profile's geometry, rank 0 of the KDA head split: 1 slot,
    /// 262,144 tokens, the single node's 3,584 MiB pin, at 70 GiB free after nothing else. The cache is planned; a snapshot slot is the rank's
    /// own state (its 32 heads: less than the single node's slot); the pool holds the admitted window AND the pages the cache may keep.
    /// CPU only, on the real pack's headers: `REXL3_GLM_DIR` (2.25bpw).
    #[test]
    #[ignore]
    fn the_2p25bpw_ranks_plan_the_cache_at_262144_and_3584_mib() {
        let Some(p) = rank0_cache_plan(262144, 3584, 70) else { return };
        assert!(p.cache.reach_tokens() >= 262144, "the 262K pin reaches the window: {}", p.cache.reach_tokens());
    }

    /// The 1,048,576-token two-node choice (`glm53-flash-tp2-1m`): rank 0 of the KDA head split at 1 slot and the window's KV pool, from
    /// 110 GiB free (an idle GB10 reads about 116): its cache pin reaches a whole window, so a repeated full-window prompt is served from the
    /// cache. CPU only, on the real pack's headers: `REXL3_GLM_DIR` (2.25bpw).
    #[test]
    #[ignore]
    fn the_2p25bpw_ranks_plan_a_whole_window_cache_at_1m() {
        // the `glm53-flash-tp2-1m` choice's pin (13,056 MiB plans 4,051 pages, 1,037,056 tokens of reach, and fails here)
        let mib = 13440;
        let Some(p) = rank0_cache_plan(1048576, mib, 110) else { return };
        assert!(p.cache.reach_tokens() >= 1048576, "a {mib} MiB pin reaches {} tokens, not the window", p.cache.reach_tokens());
    }

    /// The plan of rank 0 of a KDA-split two-node group (1 slot, `window` tokens of KV, a `mib` MiB cache pin, no tower) at `free_gib` free
    /// before the pack's share; checks what every two-node cache plan must hold. `None` without `REXL3_GLM_DIR`.
    fn rank0_cache_plan(window: usize, mib: u64, free_gib: u64) -> Option<RankPlan> {
        let Ok(dir) = std::env::var("REXL3_GLM_DIR") else {
            eprintln!("REXL3_GLM_DIR is not set: not run");
            return None;
        };
        let mut o = ServeOptions::new(dir.into());
        o.slots = 1;
        o.slots_auto = false;
        o.max_model_len = window;
        o.kv_tokens = Some(window);
        o.prefill_chunk = 512;
        o.prefix_cache = PrefixCacheOption::Mib(mib);
        o.vision = None;
        let cfg = GlmConfig::from_dir(&o.model_dir).unwrap();
        let single = crate::families::glm::prefix::snapshot_bytes(&cfg, false).unwrap();
        let ck = gb_exl3::checkpoint::scan(&o.model_dir, false).unwrap();
        let own = ExpertOwnership::range(288, 0, 2).unwrap();
        let p = plan_from_split(&o, cfg, ck, free_gib << 30, 0.0, Some(own), Some((0, 2))).unwrap();
        let rank = crate::families::glm::prefix::snapshot_bytes(&p.cfg, p.mtp).unwrap();
        eprintln!(
            "rank 0 at {window}: cache {mib} MiB = {} entries x {:.1} MiB (single node {:.1} MiB), {} pages ({} tokens of reach); pool {} pages; weights {:.2} GiB, spare {:.1} GiB",
            p.cache.snapshot_slots,
            rank as f64 / 1048576.0,
            single as f64 / 1048576.0,
            p.cache.page_blocks,
            p.cache.reach_tokens(),
            p.mo.kv_blocks,
            gib(p.weight_bytes),
            gib(p.spare_bytes)
        );
        assert!(p.cache.enabled(), "the pinned cache is planned");
        assert_eq!(p.cache.snapshot_bytes, rank, "a slot is priced as the rank's own state");
        assert!(rank < single, "the rank's slot holds its heads only ({rank} vs {single})");
        assert!(
            (p.mo.kv_blocks - p.cache.page_blocks) * 256 >= window,
            "the pool holds the admitted window besides the cache's {} pages: {}",
            p.cache.page_blocks,
            p.mo.kv_blocks
        );
        Some(p)
    }

    /// Rank 0 serving images plans the tower (weights + scratch) out of its room and holds the features reserve back; the follower plans
    /// neither (it holds no tower). CPU only, on the real pack's headers: `REXL3_GLM_DIR` (2.25bpw) and/or `REXL3_REDSNOW_DIR`; the
    /// figures are what the two-node kits' flags are checked against (`REXL3_TP_VISION_FREE_GIB`, default 117, the node's free memory;
    /// `REXL3_TP_VISION_FLAGS` = "slots,max_model_len,kv_tokens,max_image_tokens", default "2,8192,16384,8000").
    #[test]
    #[ignore]
    fn the_leader_plans_the_tower_and_the_features_reserve_and_the_follower_neither() {
        let dirs: Vec<String> = ["REXL3_GLM_DIR", "REXL3_REDSNOW_DIR"].iter().filter_map(|k| std::env::var(k).ok()).collect();
        if dirs.is_empty() {
            eprintln!("REXL3_GLM_DIR / REXL3_REDSNOW_DIR are not set: not run");
            return;
        }
        let free_gib: u64 = std::env::var("REXL3_TP_VISION_FREE_GIB").ok().and_then(|v| v.parse().ok()).unwrap_or(117);
        let flags = std::env::var("REXL3_TP_VISION_FLAGS").unwrap_or_else(|_| "2,8192,16384,8000".into());
        let f: Vec<usize> = flags.split(',').map(|x| x.trim().parse().unwrap()).collect();
        for dir in dirs {
            let cfg = GlmConfig::from_dir(Path::new(&dir)).unwrap();
            for rank in [0usize, 1] {
                let mut o = opts();
                o.model_dir = dir.clone().into();
                o.tp.as_mut().unwrap().rank = rank;
                (o.slots, o.slots_auto, o.max_model_len, o.kv_tokens, o.max_image_tokens) = (f[0], false, f[1], Some(f[2]), f[3]);
                o.prefill_chunk = 512;
                o.prefix_cache = PrefixCacheOption::Off;
                // the server gives a follower no vision slot whatever its flags; rank 0 serves images
                o.vision = (rank == 0).then(Default::default);
                // the vision cluster files' serial legs (`--spec none`): no NextN layer (the merged default `--spec auto` would plan it)
                o.spec = gb_spec::SpecMode::None;
                let ck = gb_exl3::checkpoint::scan(&o.model_dir, false).unwrap();
                let own = ExpertOwnership::range(alloc::moe_dims(&cfg).experts, rank, 2).unwrap();
                let p = plan_from_split(&o, cfg.clone(), ck, free_gib << 30, 0.0, Some(own), None).unwrap();
                eprintln!(
                    "{dir} rank {rank}: weights {:.2} GiB, tower {:.3} GiB, vision reserve {:.3} GiB, free after {:.2} GiB, slots {}, kv pages {} ({} tokens)",
                    gib(p.weight_bytes),
                    gib(p.tower_bytes),
                    gib(p.vision_reserve),
                    gib(p.free_after_weights),
                    p.mo.slots,
                    p.mo.kv_blocks,
                    p.mo.kv_blocks * 256
                );
                if rank == 0 {
                    assert!(p.tower_bytes > 1 << 30 && p.vision_reserve > 0, "rank 0 plans the tower and the reserve");
                    assert!(tp_vision::holds_tower(&o, &p.pack));
                } else {
                    assert_eq!((p.tower_bytes, p.vision_reserve), (0, 0), "a follower plans no vision");
                    assert!(!tp_vision::holds_tower(&o, &p.pack));
                }
            }
        }
    }

    /// Lane L14 (`docs/GLM_TP.md` 11.4): the rank planner with the NextN layer AND images, CPU only, on the real packs' headers (the
    /// two-node kit's flags): rank 0 plans its NextN share, the tower and the features reserve; rank 1 its NextN share and no vision.
    /// Prints each rank's figures at the node's free figure and the least free figure at which the flags are granted. `REXL3_GLM_DIR`
    /// (2.25bpw: 4 slots, 12,288 context, 32,768 KV tokens, 8,000 image tokens) and/or `REXL3_REDSNOW_DIR` (RED-SNOW: 2 slots, 8,192,
    /// 16,384, 4,096); free figure `REXL3_TP_VISION_FREE_GIB` (default 117).
    #[test]
    #[ignore]
    fn the_leader_plans_the_nextn_share_and_the_tower_together() {
        let free_gib: u64 = std::env::var("REXL3_TP_VISION_FREE_GIB").ok().and_then(|v| v.parse().ok()).unwrap_or(117);
        let packs = [("REXL3_GLM_DIR", 4usize, 12288usize, 32768usize, 8000usize), ("REXL3_REDSNOW_DIR", 2, 8192, 16384, 4096)];
        let mut ran = 0;
        for (var, slots, ctx, kv, img) in packs {
            let Ok(dir) = std::env::var(var) else { continue };
            ran += 1;
            let cfg = GlmConfig::from_dir(Path::new(&dir)).unwrap();
            for rank in [0usize, 1] {
                let mut o = opts();
                o.model_dir = dir.clone().into();
                o.tp.as_mut().unwrap().rank = rank;
                (o.slots, o.slots_auto, o.max_model_len, o.kv_tokens, o.max_image_tokens) = (slots, false, ctx, Some(kv), img);
                o.prefill_chunk = 512;
                o.prefix_cache = PrefixCacheOption::Off;
                o.drafting = true;
                o.spec = gb_spec::SpecMode::Auto;
                o.vision = (rank == 0).then(Default::default);
                let scan = |o: &ServeOptions| gb_exl3::checkpoint::scan(&o.model_dir, false).unwrap();
                let own = || ExpertOwnership::range(alloc::moe_dims(&cfg).experts, rank, 2).unwrap();
                let p = plan_from_split(&o, cfg.clone(), scan(&o), free_gib << 30, 0.0, Some(own()), None).unwrap();
                assert!(p.mtp, "the NextN layer is planned");
                assert_eq!(p.tower_bytes > 0, rank == 0, "only rank 0 plans the tower");
                // the least free figure (in 1/8 GiB) at which the flags are granted
                let mut need = ((p.weight_bytes + p.tower_bytes) >> 27 << 27) + (8 << 30);
                loop {
                    if let Ok(q) = plan_from_split(&o, cfg.clone(), scan(&o), need, 0.0, Some(own()), None) {
                        if q.mo.kv_blocks * 256 >= kv {
                            break;
                        }
                    }
                    need += 1 << 27;
                }
                eprintln!(
                    "{var} rank {rank} (NextN on, vision {}): weights {:.2} GiB, tower {:.3} GiB, vision reserve {:.3} GiB, free after {:.2} GiB at \
                     {free_gib} GiB free; slots {}, KV pages {} ({} tokens); the flags need {:.3} GiB free",
                    rank == 0,
                    gib(p.weight_bytes),
                    gib(p.tower_bytes),
                    gib(p.vision_reserve),
                    gib(p.free_after_weights),
                    p.mo.slots,
                    p.mo.kv_blocks,
                    p.mo.kv_blocks * 256,
                    gib(need)
                );
            }
        }
        if ran == 0 {
            eprintln!("neither REXL3_GLM_DIR nor REXL3_REDSNOW_DIR is set: not run");
        }
    }

    /// The sc_4.10bpw pack (exllamav3 1.6.2, 156.19 GiB) at R = 2 on its real header, CPU only (`REXL3_GLM_SC410_DIR`): what each rank
    /// maps and charges for the RED-SNOW kits' three flag sets (text; text with the NextN layer; images on rank 0), and the least free memory
    /// (to 1/8 GiB) at which the planner grants them: the `memory.min_free_gib` of a share-only kit for this pack. A share-only rank holds the
    /// replicated trunk and half of the routed experts, so its figure is about 14 GiB under RED-SNOW's 108.4-111 GiB.
    #[test]
    #[ignore]
    fn the_sc410_ranks_plan_from_their_share() {
        let Ok(dir) = std::env::var("REXL3_GLM_SC410_DIR") else {
            eprintln!("REXL3_GLM_SC410_DIR is not set: not run");
            return;
        };
        let cfg = GlmConfig::from_dir(Path::new(&dir)).unwrap();
        assert!(matches!(cfg.vision.as_ref().map(|v| v.tower), Some(crate::families::glm::VisionTowerFormat::Exl3 { bits: 6 })));
        let scan = |o: &ServeOptions| gb_exl3::checkpoint::scan(&o.model_dir, false).unwrap();
        // (kit, --spec, rank 0 serves images): the flags of cluster.glm-redsnow-tp2{,-mtp,-vision}.json (2 slots, 8,192, 16,384 KV tokens,
        // 512-row chunk, prefix cache off; 4,096 image tokens); the last row is the vision kit with the NextN drafter on (the two-node
        // kit's image MTP legs swap `--spec none --no-drafts` for `--spec mtp|auto` in that file)
        // lane E Phase 1: each kit at the 2,048-row chunk too (the adaptive slice's big size: its scratch is the plan's)
        for (kit, spec, images, chunk) in [
            ("text", gb_spec::SpecMode::None, false, 512),
            ("text + NextN", gb_spec::SpecMode::Auto, false, 512),
            ("images on rank 0", gb_spec::SpecMode::None, true, 512),
            ("images on rank 0 + NextN", gb_spec::SpecMode::Auto, true, 512),
            ("text", gb_spec::SpecMode::None, false, 2048),
            ("text + NextN", gb_spec::SpecMode::Auto, false, 2048),
            ("images on rank 0", gb_spec::SpecMode::None, true, 2048),
            ("images on rank 0 + NextN", gb_spec::SpecMode::Auto, true, 2048),
        ] {
            for rank in [0usize, 1] {
                let mut o = opts();
                o.model_dir = dir.clone().into();
                o.tp.as_mut().unwrap().rank = rank;
                (o.slots, o.slots_auto, o.max_model_len, o.kv_tokens, o.max_image_tokens) = (2, false, 8192, Some(16384), 4096);
                o.prefill_chunk = chunk;
                o.prefix_cache = PrefixCacheOption::Off;
                o.drafting = spec != gb_spec::SpecMode::None;
                o.spec = spec;
                o.vision = (images && rank == 0).then(Default::default);
                let own = || ExpertOwnership::range(alloc::moe_dims(&cfg).experts, rank, 2).unwrap();
                let p = plan_from_split(&o, cfg.clone(), scan(&o), 117 << 30, 0.0, Some(own()), None).unwrap();
                assert_eq!(p.mtp, spec != gb_spec::SpecMode::None);
                assert_eq!(p.tower_bytes > 0, images && rank == 0, "only rank 0 serving images plans the tower");
                // a share-only rank maps no tower: its plan has no vision region whatever the pack holds
                assert_eq!(p.pack.weights.vision.is_some(), images && rank == 0);
                let mut need = ((p.weight_bytes + p.tower_bytes) >> 27 << 27) + (8 << 30);
                loop {
                    if let Ok(q) = plan_from_split(&o, cfg.clone(), scan(&o), need, 0.0, Some(own()), None) {
                        if q.mo.kv_blocks * 256 >= 16384 {
                            break;
                        }
                    }
                    need += 1 << 27;
                }
                assert_eq!(p.chunk, chunk);
                eprintln!(
                    "sc_4.10bpw {kit}, chunk {chunk}, rank {rank}: weights {:.2} GiB, tower {:.3} GiB, vision reserve {:.3} GiB; at 117 GiB free: free after {:.2} GiB, \
                     slots {}, KV pages {} ({} tokens); the flags need {:.3} GiB free",
                    gib(p.weight_bytes),
                    gib(p.tower_bytes),
                    gib(p.vision_reserve),
                    gib(p.free_after_weights),
                    p.mo.slots,
                    p.mo.kv_blocks,
                    p.mo.kv_blocks * 256,
                    gib(need)
                );
                assert!(gib(need) > 85.0 && gib(need) < 100.0, "{kit} rank {rank}: {:.2} GiB", gib(need));
            }
        }
    }

    /// The sc_4.10bpw LONG-CONTEXT kit (PASS_RULES_GLM_ADDENDUM_GEOMETRY D-G1: planned, then frozen as its own addendum before its first
    /// leg), by the method of `the_sc410_ranks_plan_from_their_share`: for each rank, text with and without the NextN drafter (no tower:
    /// NIAH and the whole-window request are text), what the planner grants at 117 GiB free and the least free memory (to 1/8 GiB) at which
    /// it grants the pinned slots and KV tokens. `REXL3_GLM_SC410_DIR` (the pack's header); `REXL3_GLM_LONGCTX=MAX_MODEL_LEN,KV_TOKENS`
    /// (default 131072,262144: two slots of the whole window). The least figure is found by bisection (the scan is re-read for every
    /// plan) and confirmed: granted at it, refused 1/8 GiB under it.
    /// Lane E, the rank strips of the KDA head split on the sc_4.10bpw header (CPU only, `REXL3_GLM_SC410_DIR`): with the split
    /// (`plan_from_split`, `REXL3_GLM_TP_SPLIT` with kda) a rank's weight bytes drop by exactly the other rank's qkv strips (34 layers), its
    /// per-slot KDA state halves, and the least free memory (to 1/8 GiB) at which the planner grants each kit's flags drops by both: the text
    /// kit, the long-context kit and the 8-slot flags of plan item 6b (`--slots 8 --kv-tokens 16384`). Without the split the plan is
    /// `plan_from`'s.
    #[test]
    #[ignore]
    fn the_sc410_kda_split_ranks_plan_their_strips_and_state() {
        let Ok(dir) = std::env::var("REXL3_GLM_SC410_DIR") else {
            eprintln!("REXL3_GLM_SC410_DIR is not set: not run");
            return;
        };
        let cfg = GlmConfig::from_dir(Path::new(&dir)).unwrap();
        let scan = |o: &ServeOptions| gb_exl3::checkpoint::scan(&o.model_dir, false).unwrap();
        // (kit, slots, --max-model-len, --kv-tokens, NextN drafting): item 6b's 8-slot kits with --kv-tokens max(16384, 8 x 2048)
        for (kit, slots, mml, kv, nextn) in [
            ("text", 2usize, 8192usize, 16384usize, false),
            ("long-context", 2, 131072, 262144, false),
            ("8 slots (item 6b)", 8, 8192, 16384, false),
            ("8 slots + NextN (item 6b)", 8, 8192, 16384, true),
        ] {
            for rank in [0usize, 1] {
                let mut o = opts();
                o.model_dir = dir.clone().into();
                o.tp.as_mut().unwrap().rank = rank;
                (o.slots, o.slots_auto, o.max_model_len, o.kv_tokens) = (slots, false, mml, Some(kv));
                o.prefill_chunk = 2048;
                o.prefix_cache = PrefixCacheOption::Off;
                (o.drafting, o.spec, o.vision) =
                    (nextn, if nextn { gb_spec::SpecMode::Auto } else { gb_spec::SpecMode::None }, None);
                let own = || ExpertOwnership::range(alloc::moe_dims(&cfg).experts, rank, 2).unwrap();
                let granted = |free: u64, split: Option<(usize, usize)>| -> Option<RankPlan> {
                    plan_from_split(&o, cfg.clone(), scan(&o), free, 0.0, Some(own()), split)
                        .ok()
                        .filter(|q| q.mo.kv_blocks * 256 >= kv && q.mo.slots == slots)
                };
                let need = |split: Option<(usize, usize)>| -> (u64, RankPlan) {
                    let p = granted(117 << 30, split).unwrap_or_else(|| panic!("{kit} rank {rank}: not granted at 117 GiB free"));
                    let (mut lo, mut hi) = (((p.weight_bytes >> 27) << 27) + (8 << 30), 117u64 << 30);
                    assert!(granted(lo, split).is_none());
                    while hi - lo > 1 << 27 {
                        let mid = lo + (((hi - lo) >> 28) << 27).max(1 << 27);
                        if granted(mid, split).is_some() {
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    }
                    (hi, p)
                };
                let (need_off, off) = need(None);
                let (need_on, on) = need(Some((rank, 2)));
                assert_eq!(on.cfg.kda.rank, Some((32 * rank, 32 * (rank + 1))));
                assert!(off.cfg.kda.rank.is_none());
                // the weights: exactly the other rank's qkv strips (half of every KDA layer's qkv trellis and svh)
                assert_eq!((off.mtp, on.mtp), (nextn, nextn));
                let saved: u64 = off
                    .pack
                    .weights
                    .layers
                    .iter()
                    .filter_map(|l| match &l.mixer {
                        crate::families::glm::weights::MixerWeights::Kda(k) => {
                            Some(k.qkv_proj.trellis.bytes / 2 + k.qkv_proj.svh.bytes / 2)
                        }
                        _ => None,
                    })
                    .sum();
                assert_eq!(off.weight_bytes - on.weight_bytes, saved, "{kit} rank {rank}");
                let pools = |c: &GlmConfig| {
                    AllocPlan::slot_items(c, mml, false).unwrap().iter().find(|i| i.name == "kda_pools").unwrap().bytes
                };
                let (p_off, p_on) = (pools(&off.cfg), pools(&on.cfg));
                assert_eq!(p_on, 34 * gb_cuda::kda::Dims { heads: 32, hidden: 4096 }.slot_bytes() as u64);
                assert!(need_on < need_off, "{kit} rank {rank}");
                eprintln!(
                    "sc_4.10bpw {kit} (--slots {slots} --max-model-len {mml} --kv-tokens {kv}, chunk 2048), rank {rank}: weights {:.3} -> {:.3} GiB \
                     (-{:.3}), KDA state a slot {:.1} -> {:.1} MiB; the flags need {:.3} -> {:.3} GiB free (-{:.3})",
                    gib(off.weight_bytes),
                    gib(on.weight_bytes),
                    gib(saved),
                    p_off as f64 / 1048576.0,
                    p_on as f64 / 1048576.0,
                    gib(need_off),
                    gib(need_on),
                    gib(need_off - need_on)
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn the_sc410_long_context_kit_plans_from_its_share() {
        let Ok(dir) = std::env::var("REXL3_GLM_SC410_DIR") else {
            eprintln!("REXL3_GLM_SC410_DIR is not set: not run");
            return;
        };
        let (mml, kv) = match std::env::var("REXL3_GLM_LONGCTX") {
            Ok(v) => {
                let (a, b) = v.split_once(',').expect("REXL3_GLM_LONGCTX=MAX_MODEL_LEN,KV_TOKENS");
                (a.trim().parse::<usize>().unwrap(), b.trim().parse::<usize>().unwrap())
            }
            Err(_) => (131072, 262144),
        };
        let cfg = GlmConfig::from_dir(Path::new(&dir)).unwrap();
        let scan = |o: &ServeOptions| gb_exl3::checkpoint::scan(&o.model_dir, false).unwrap();
        // lane E Phase 1: at the 2,048-row chunk too (`REXL3_GLM_LONGCTX_CHUNKS`, default 512,2048)
        let chunks: Vec<usize> = std::env::var("REXL3_GLM_LONGCTX_CHUNKS")
            .map(|v| v.split(',').map(|c| c.trim().parse().unwrap()).collect())
            .unwrap_or_else(|_| vec![512, 2048]);
        for (kit, spec, chunk) in chunks.iter().flat_map(|&c| {
            [("long-context text", gb_spec::SpecMode::None, c), ("long-context text + NextN", gb_spec::SpecMode::Auto, c)]
        }) {
            for rank in [0usize, 1] {
                let mut o = opts();
                o.model_dir = dir.clone().into();
                o.tp.as_mut().unwrap().rank = rank;
                (o.slots, o.slots_auto, o.max_model_len, o.kv_tokens) = (2, false, mml, Some(kv));
                o.prefill_chunk = chunk;
                o.prefix_cache = PrefixCacheOption::Off;
                o.drafting = spec != gb_spec::SpecMode::None;
                o.spec = spec;
                o.vision = None;
                let own = || ExpertOwnership::range(alloc::moe_dims(&cfg).experts, rank, 2).unwrap();
                let granted = |free: u64| -> Option<RankPlan> {
                    plan_from_split(&o, cfg.clone(), scan(&o), free, 0.0, Some(own()), None)
                        .ok()
                        .filter(|q| q.mo.kv_blocks * 256 >= kv && q.mo.slots == 2)
                };
                let p = granted(117 << 30).unwrap_or_else(|| panic!("{kit} rank {rank}: not granted at 117 GiB free"));
                assert_eq!(p.mtp, spec != gb_spec::SpecMode::None);
                assert_eq!(p.tower_bytes, 0);
                // bisection in 1/8 GiB steps between the weights + the 8 GiB floor (refused) and 117 GiB (granted)
                let (mut lo, mut hi) = (((p.weight_bytes >> 27) << 27) + (8 << 30), 117u64 << 30);
                assert!(granted(lo).is_none(), "{kit} rank {rank}: granted at the weights + the floor");
                while hi - lo > 1 << 27 {
                    let mid = lo + (((hi - lo) >> 28) << 27).max(1 << 27);
                    if granted(mid).is_some() {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                let need = hi;
                assert!(
                    granted(need).is_some() && granted(need - (1 << 27)).is_none(),
                    "{kit} rank {rank}: {:.3} GiB",
                    gib(need)
                );
                let q = granted(need).unwrap();
                eprintln!(
                    "sc_4.10bpw {kit} (--max-model-len {mml}, --kv-tokens {kv}, 2 slots, chunk {chunk}), rank {rank}: weights {:.2} GiB; at 117 GiB \
                     free: free after {:.2} GiB, slots {}, KV pages {} ({} tokens); the flags need {:.3} GiB free (granted there: KV pages {}, \
                     free after {:.2} GiB)",
                    gib(p.weight_bytes),
                    gib(p.free_after_weights),
                    p.mo.slots,
                    p.mo.kv_blocks,
                    p.mo.kv_blocks * 256,
                    gib(need),
                    q.mo.kv_blocks,
                    gib(q.free_after_weights)
                );
            }
        }
    }

    /// Lane E, Phase L0 (the full 1M-token context at TP=2; CPU only, `REXL3_GLM_SC410_DIR`): what each rank's share of the sc_4.10bpw pack needs
    /// for the long-context shapes `REXL3_GLM_LONGCTX_SHAPES` = `SLOTS:MAX_MODEL_LEN:KV_TOKENS;...` (default `1:1048576:1048576;2:1048576:1064960`:
    /// one slot of the whole window, and one more with 16,384 tokens for a short request beside it), text and NextN (no tower: a 1M request is
    /// text), with the KDA split's rank strips (`plan_from_split`, as the 8-slot test does) and without them (what the split is worth), at each
    /// chunk of `REXL3_GLM_LONGCTX_CHUNKS` (default 512,2048) under the process's prefill contract (`REXL3_GLM_PREFILL_CONTRACT`, read once a
    /// process: P2 is a second run, chunk 4,096). For each: the least free memory (to 1/8 GiB, confirmed) at which the planner grants the
    /// slots and KV tokens, searched up to 128 GiB (a shape the planner does not grant there is reported refused, not a panic); the breakdown
    /// (the rank's weights, the KV pool, the slots' state, the window-scaled scratch apart from the rest, the serving floor); and the verdict
    /// against the nodes' free memory before a load (`REXL3_GLM_FREE_BEFORE_LOAD_GIB`, default 114: measured 2026-10-17): a need within 10 GiB of
    /// it is refused (MAIN, Phase L0), and `min_free_gib` would be the need + 1.6 GiB, rounded up.
    #[test]
    #[ignore]
    fn the_sc410_long_context_1m_plans_from_its_share() {
        let Ok(dir) = std::env::var("REXL3_GLM_SC410_DIR") else {
            eprintln!("REXL3_GLM_SC410_DIR is not set: not run");
            return;
        };
        let shapes: Vec<(usize, usize, usize)> = std::env::var("REXL3_GLM_LONGCTX_SHAPES")
            .unwrap_or_else(|_| "1:1048576:1048576;2:1048576:1064960".into())
            .split(';')
            .filter(|s| !s.trim().is_empty())
            .map(|s| {
                let v: Vec<usize> = s.split(':').map(|x| x.trim().parse().unwrap()).collect();
                (v[0], v[1], v[2])
            })
            .collect();
        let chunks: Vec<usize> = std::env::var("REXL3_GLM_LONGCTX_CHUNKS")
            .map(|v| v.split(',').map(|c| c.trim().parse().unwrap()).collect())
            .unwrap_or_else(|_| vec![512, 2048]);
        let before: f64 = std::env::var("REXL3_GLM_FREE_BEFORE_LOAD_GIB").ok().and_then(|v| v.parse().ok()).unwrap_or(114.0);
        let contract = super::super::contract::current();
        let cfg = GlmConfig::from_dir(Path::new(&dir)).unwrap();
        let scan = |o: &ServeOptions| gb_exl3::checkpoint::scan(&o.model_dir, false).unwrap();
        let top: u64 = 128 << 30;
        for &(slots, mml, kv) in &shapes {
            for &chunk in &chunks {
                for nextn in [false, true] {
                    for split in [true, false] {
                        for rank in [0usize, 1] {
                            let mut o = opts();
                            o.model_dir = dir.clone().into();
                            o.tp.as_mut().unwrap().rank = rank;
                            (o.slots, o.slots_auto, o.max_model_len, o.kv_tokens) = (slots, false, mml, Some(kv));
                            o.prefill_chunk = chunk;
                            o.prefix_cache = PrefixCacheOption::Off;
                            (o.drafting, o.spec, o.vision) =
                                (nextn, if nextn { gb_spec::SpecMode::Auto } else { gb_spec::SpecMode::None }, None);
                            let sp = split.then_some((rank, 2usize));
                            let own = || ExpertOwnership::range(alloc::moe_dims(&cfg).experts, rank, 2).unwrap();
                            let granted = |free: u64| -> Option<RankPlan> {
                                plan_from_split(&o, cfg.clone(), scan(&o), free, 0.0, Some(own()), sp)
                                    .ok()
                                    .filter(|q| q.mo.kv_blocks * 256 >= kv && q.mo.slots == slots)
                            };
                            let tag = format!(
                                "{} ({}), --slots {slots} --max-model-len {mml} --kv-tokens {kv}, chunk {chunk} (contract {}), rank {rank}",
                                if nextn { "text + NextN" } else { "text" },
                                if split { "split strips" } else { "no split" },
                                contract.id()
                            );
                            let Some(p) = granted(top) else {
                                let why = plan_from_split(&o, cfg.clone(), scan(&o), top, 0.0, Some(own()), sp)
                                    .err()
                                    .map(|e| e.to_string())
                                    .unwrap_or_else(|| "granted fewer slots or KV tokens than asked".into());
                                eprintln!("L0 {tag}: REFUSED even at 128 GiB free: {why}");
                                continue;
                            };
                            let (mut lo, mut hi) = (((p.weight_bytes >> 27) << 27) + (8 << 30), top);
                            if granted(lo).is_some() {
                                hi = lo;
                            } else {
                                while hi - lo > 1 << 27 {
                                    let mid = lo + (((hi - lo) >> 28) << 27).max(1 << 27);
                                    if granted(mid).is_some() {
                                        hi = mid;
                                    } else {
                                        lo = mid;
                                    }
                                }
                                assert!(granted(hi).is_some() && granted(hi - (1 << 27)).is_none(), "{tag}");
                            }
                            let need = hi;
                            // the breakdown, from the rank's own memory profile and scratch items (the plan's inputs)
                            let rcfg = if split { cfg.with_kda_rank(rank, 2).unwrap() } else { cfg.clone() };
                            let mo = memory::engine_options(&memory_options(&o, p.mtp));
                            let prof = memory::memory_profile(&rcfg, &mo).unwrap();
                            let kv_pool = prof.kv_bytes_per_token() * kv as u64;
                            let state = (prof.state_bytes_per_slot() + prof.drafter_state_bytes_per_slot()) * slots as u64;
                            let items =
                                AllocPlan::scratch_items_chunked(&rcfg, WINDOW_ROWS, mo.prefill_rows, mml, p.mtp).unwrap();
                            let item = |n: &str| items.iter().find(|i| i.name == n).map_or(0, |i| i.bytes);
                            let slot_items = AllocPlan::slot_items(&rcfg, mml, p.mtp).unwrap();
                            let block_table =
                                slot_items.iter().find(|i| i.name == "block_table").map_or(0, |i| i.bytes) * slots as u64;
                            let window_scaled = item("dsa_scratch") + block_table;
                            let gibf = |b: u64| b as f64 / (1u64 << 30) as f64;
                            let need_g = gibf(need);
                            let alloc_g = need_g - gibf(o.floor_bytes);
                            let verdict = if need_g > before - 10.0 {
                                "REFUSED (within 10 GiB of the free memory before load)"
                            } else {
                                "fits"
                            };
                            eprintln!(
                                "L0 {tag}: needs {need_g:.3} GiB free -> min_free_gib {:.0}; {verdict}; MemAvailable while serving ~{:.1} GiB (C: {before:.0} - \
                                 {alloc_g:.2} allocated). Breakdown: weights {:.3} GiB; KV pool {:.3} GiB ({} B a token x {kv}); slots' state {:.3} GiB; \
                                 scratch {:.3} GiB, of which window-scaled {:.3} MiB (DSA scores {:.3} MiB at {mml}, block tables {:.3} MiB) and MLA \
                                 {:.3} MiB; engine reserve {:.3} GiB; floor {:.2} GiB; granted at the need: {} KV pages, free after the weights {:.2} GiB",
                                (need_g + 1.6).ceil(),
                                before - alloc_g,
                                gibf(p.weight_bytes),
                                gibf(kv_pool),
                                prof.kv_bytes_per_token(),
                                gibf(state),
                                gibf(prof.scratch_bytes()),
                                window_scaled as f64 / 1048576.0,
                                item("dsa_scratch") as f64 / 1048576.0,
                                block_table as f64 / 1048576.0,
                                item("mla_scratch") as f64 / 1048576.0,
                                gibf(prof.engine_reserve_bytes()),
                                gibf(o.floor_bytes),
                                granted(need).unwrap().mo.kv_blocks,
                                gibf(granted(need).unwrap().free_after_weights),
                            );
                        }
                    }
                }
            }
        }
    }

    /// R-G-rails (c) (PASS_RULES_GLM_ADDENDUM_AMEND1 D-A4): the fingerprint has no "TP=2 only" mark, so what keeps the sc_4.10bpw pack off a
    /// single node is the planner: a single-node plan (no group, the whole pack mapped: every routed expert) is refused by name, because the
    /// pack's 154 GiB of text regions do not fit one GB10 (121 GiB; 128 GiB is asked here, more than any GB10 has), while each rank of the
    /// two-rank group plans its share of the same flags from 117 GiB. On the pack's real header (`REXL3_GLM_SC410_DIR`), CPU only. The flags
    /// are the frozen text kit's (cluster.glm-sc410-tp2.json: 2 slots, 8,192 / 16,384, 512-row chunk, --no-vision, --spec none, prefix
    /// cache off), and the refusal is also asked of one slot with a 4,096-token window, so it does not hang on the kit's KV.
    #[test]
    #[ignore]
    fn a_single_node_plan_of_the_sc410_pack_is_refused_by_name() {
        let Ok(dir) = std::env::var("REXL3_GLM_SC410_DIR") else {
            eprintln!("REXL3_GLM_SC410_DIR is not set: not run");
            return;
        };
        let cfg = GlmConfig::from_dir(Path::new(&dir)).unwrap();
        let scan = |o: &ServeOptions| gb_exl3::checkpoint::scan(&o.model_dir, false).unwrap();
        let kit = |o: &mut ServeOptions, slots: usize, mml: usize, kv: usize| {
            o.model_dir = dir.clone().into();
            (o.slots, o.slots_auto, o.max_model_len, o.kv_tokens) = (slots, false, mml, Some(kv));
            o.prefill_chunk = 512;
            o.prefix_cache = PrefixCacheOption::Off;
            o.drafting = false;
            o.spec = gb_spec::SpecMode::None;
            o.vision = None;
        };
        // the whole pack's text regions, as a single node's load plan maps them (all 288 routed experts)
        let inv = crate::families::glm::PackInventory::from_checkpoint(&scan(&ServeOptions::new(dir.clone().into()))).unwrap();
        let all = ExpertOwnership::all(cfg.ffn.routed_experts);
        let whole = GlmPack::from_inventory_with(&inv, cfg.clone(), &all, false).unwrap();
        let whole_gib = gib(rank_weight_bytes(&whole));
        assert!(whole_gib > 128.0, "the whole pack's text regions are {whole_gib:.2} GiB");
        for (slots, mml, kv) in [(2usize, 8192usize, 16384usize), (1, 4096, 4096)] {
            let mut o = ServeOptions::new(dir.clone().into());
            o.tp = None;
            kit(&mut o, slots, mml, kv);
            let e = match plan_from_split(&o, cfg.clone(), scan(&o), 128 << 30, 0.0, None, None) {
                Ok(p) => {
                    panic!("a single node planned the whole sc_4.10bpw pack ({:.2} GiB, {slots} slots)", gib(p.weight_bytes))
                }
                Err(e) => e.to_string(),
            };
            let named = format!("after the pack's {whole_gib:.1} GiB");
            assert!(e.contains("cannot plan") && e.contains(&named), "the refusal names the pack's size ({named:?}): {e}");
            eprintln!("single node, {slots} slots, --max-model-len {mml}, --kv-tokens {kv}, 128 GiB free: refused: {e}");
        }
        // the contrast: each rank of the two-rank group plans its share of the kit's flags from 117 GiB
        for rank in [0usize, 1] {
            let mut o = opts();
            o.tp.as_mut().unwrap().rank = rank;
            kit(&mut o, 2, 8192, 16384);
            let own = ExpertOwnership::range(alloc::moe_dims(&cfg).experts, rank, 2).unwrap();
            let p = plan_from_split(&o, cfg.clone(), scan(&o), 117 << 30, 0.0, Some(own), None).unwrap();
            assert!(
                p.mo.slots == 2 && p.mo.kv_blocks * 256 >= 16384,
                "rank {rank}: {} slots, {} KV pages",
                p.mo.slots,
                p.mo.kv_blocks
            );
            eprintln!(
                "rank {rank} of 2, the kit's flags, 117 GiB free: granted: weights {:.2} GiB, free after {:.2} GiB, {} slots, {} KV pages",
                gib(p.weight_bytes),
                gib(p.free_after_weights),
                p.mo.slots,
                p.mo.kv_blocks
            );
        }
    }

    /// PW-1 (PASS_RULES_GLM_ADDENDUM_AMEND1 D-A6, informational): the frozen 8,192 text kit and its NextN twin with `--slots S` and
    /// `--kv-tokens max(16384, S x 2048)`, S in `REXL3_GLM_PW1_SLOTS` (default "32,16,8"), by the method of
    /// `the_sc410_long_context_kit_plans_from_its_share`: for each S, kit and rank, the least free memory (to 1/8 GiB, bisected and
    /// confirmed) at which the planner grants S slots and the KV tokens, or that 117 GiB does not grant them. `REXL3_GLM_SC410_DIR` (the
    /// pack's header), CPU only. The lane picks S from these lines (the largest whose figure is at most 99.0 GiB on both ranks).
    #[test]
    #[ignore]
    fn the_sc410_wide_slot_kits_plan_from_their_share() {
        let Ok(dir) = std::env::var("REXL3_GLM_SC410_DIR") else {
            eprintln!("REXL3_GLM_SC410_DIR is not set: not run");
            return;
        };
        let ss: Vec<usize> = std::env::var("REXL3_GLM_PW1_SLOTS")
            .unwrap_or_else(|_| "32,16,8".into())
            .split(',')
            .map(|x| x.trim().parse().expect("REXL3_GLM_PW1_SLOTS=S,S,..."))
            .collect();
        let cfg = GlmConfig::from_dir(Path::new(&dir)).unwrap();
        let scan = |o: &ServeOptions| gb_exl3::checkpoint::scan(&o.model_dir, false).unwrap();
        for s in ss {
            let kv = (s * 2048).max(16384);
            for (kit, spec) in [("text", gb_spec::SpecMode::None), ("text + NextN", gb_spec::SpecMode::Auto)] {
                for rank in [0usize, 1] {
                    let mut o = opts();
                    o.model_dir = dir.clone().into();
                    o.tp.as_mut().unwrap().rank = rank;
                    (o.slots, o.slots_auto, o.max_model_len, o.kv_tokens) = (s, false, 8192, Some(kv));
                    o.prefill_chunk = 512;
                    o.prefix_cache = PrefixCacheOption::Off;
                    o.drafting = spec != gb_spec::SpecMode::None;
                    o.spec = spec;
                    o.vision = None;
                    let own = || ExpertOwnership::range(alloc::moe_dims(&cfg).experts, rank, 2).unwrap();
                    let granted = |free: u64| -> Option<RankPlan> {
                        plan_from_split(&o, cfg.clone(), scan(&o), free, 0.0, Some(own()), None)
                            .ok()
                            .filter(|q| q.mo.kv_blocks * 256 >= kv && q.mo.slots == s)
                    };
                    let Some(p) = granted(117 << 30) else {
                        eprintln!(
                            "sc_4.10bpw PW-1 {kit} (--slots {s}, --max-model-len 8192, --kv-tokens {kv}, chunk 512), rank {rank}: NOT granted at \
                             117 GiB free"
                        );
                        continue;
                    };
                    assert_eq!(p.mtp, spec != gb_spec::SpecMode::None);
                    let (mut lo, mut hi) = (((p.weight_bytes >> 27) << 27) + (8 << 30), 117u64 << 30);
                    assert!(granted(lo).is_none(), "{kit} S={s} rank {rank}: granted at the weights + the floor");
                    while hi - lo > 1 << 27 {
                        let mid = lo + (((hi - lo) >> 28) << 27).max(1 << 27);
                        if granted(mid).is_some() {
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    }
                    let need = hi;
                    assert!(
                        granted(need).is_some() && granted(need - (1 << 27)).is_none(),
                        "{kit} S={s} rank {rank}: {:.3} GiB",
                        gib(need)
                    );
                    let q = granted(need).unwrap();
                    eprintln!(
                        "sc_4.10bpw PW-1 {kit} (--slots {s}, --max-model-len 8192, --kv-tokens {kv}, chunk 512), rank {rank}: weights {:.2} GiB; \
                         at 117 GiB free: free after {:.2} GiB, slots {}, KV pages {} ({} tokens); the flags need {:.3} GiB free (granted there: \
                         KV pages {}, free after {:.2} GiB)",
                        gib(p.weight_bytes),
                        gib(p.free_after_weights),
                        p.mo.slots,
                        p.mo.kv_blocks,
                        p.mo.kv_blocks * 256,
                        gib(need),
                        q.mo.kv_blocks,
                        gib(q.free_after_weights)
                    );
                }
            }
        }
    }

    /// The rank planner with the NextN layer (lane L12) on a real pack's header, CPU only: the weights a rank charges with and without
    /// the layer (its trunk replicated, its experts the rank's share), the KV bytes per token with and without the drafter's cache, and
    /// what the cluster file's flags are granted from a free figure. `REXL3_GLM_DIR` (the 2.25bpw pack, 4 slots, 32,768 tokens at 108
    /// GiB) and/or `REXL3_REDSNOW_DIR` (RED-SNOW, 2 slots, 16,384 tokens at 110 GiB); "not run" without either.
    #[test]
    #[ignore]
    fn the_rank_planner_charges_the_nextn_share_and_the_drafters_cache() {
        let packs = [("REXL3_GLM_DIR", 4usize, 32768usize, 108u64), ("REXL3_REDSNOW_DIR", 2, 16384, 112)];
        let mut ran = 0;
        for (var, slots, kv, free_gib) in packs {
            let Ok(dir) = std::env::var(var) else { continue };
            ran += 1;
            let mut o = ServeOptions::new(dir.into());
            o.slots = slots;
            o.slots_auto = false;
            o.max_model_len = 8192;
            o.kv_tokens = Some(kv);
            o.prefill_chunk = 512;
            o.prefix_cache = PrefixCacheOption::Off;
            o.vision = None;
            o.drafting = true;
            let cfg = GlmConfig::from_dir(&o.model_dir).unwrap();
            let scan = |o: &ServeOptions| gb_exl3::checkpoint::scan(&o.model_dir, false).unwrap();
            let kvb = |o: &ServeOptions, mtp: bool| {
                let mo = memory::engine_options(&memory_options(o, mtp));
                memory::memory_profile(&cfg, &mo).unwrap().kv_bytes_per_token()
            };
            for r in 0..2 {
                let own = ExpertOwnership::range(288, r, 2).unwrap();
                o.spec = gb_spec::SpecMode::None;
                let off = plan_from_split(&o, cfg.clone(), scan(&o), free_gib << 30, 0.0, Some(own.clone()), None).unwrap();
                o.spec = gb_spec::SpecMode::Auto;
                let on = plan_from_split(&o, cfg.clone(), scan(&o), free_gib << 30, 0.0, Some(own), None).unwrap();
                assert!(!off.mtp && on.mtp);
                assert!(on.weight_bytes > off.weight_bytes);
                eprintln!(
                    "{var} rank {r}: weights {:.2} GiB without the NextN layer, {:.2} GiB with it (+{:.2}); KV {} B/token without, {} B/token \
                     with; at {free_gib} GiB free: slots {}, KV pages {} ({} tokens), free after weights {:.2} GiB",
                    gib(off.weight_bytes),
                    gib(on.weight_bytes),
                    gib(on.weight_bytes - off.weight_bytes),
                    kvb(&o, false),
                    kvb(&o, true),
                    on.mo.slots,
                    on.mo.kv_blocks,
                    on.mo.kv_blocks * 256,
                    gib(on.free_after_weights)
                );
                assert_eq!(on.mo.slots, slots);
                // the least free figure (in 1/8 GiB) at which the flags are granted with the NextN layer
                let mut need = ((on.weight_bytes >> 27) << 27) + (10 << 30);
                loop {
                    o.spec = gb_spec::SpecMode::Auto;
                    let own = ExpertOwnership::range(288, r, 2).unwrap();
                    if let Ok(p) = plan_from_split(&o, cfg.clone(), scan(&o), need, 0.0, Some(own), None) {
                        if p.mo.kv_blocks * 256 >= kv {
                            break;
                        }
                    }
                    need += 1 << 27;
                }
                eprintln!(
                    "{var} rank {r}: the flags ({slots} slots, {kv} KV tokens) with the NextN layer need {:.3} GiB free",
                    gib(need)
                );
                if on.mo.kv_blocks * 256 < kv {
                    eprintln!("{var} rank {r}: NOT granted at {free_gib} GiB");
                }
            }
            assert_eq!(kvb(&o, true), 13_056, "the KV bytes per token with the NextN layer's cache");
        }
        if ran == 0 {
            eprintln!("neither REXL3_GLM_DIR nor REXL3_REDSNOW_DIR is set: not run");
        }
    }

    #[test]
    fn a_group_refuses_what_it_cannot_forward_by_name_and_accepts_the_plain_text_configuration() {
        let mut o = opts();
        o.slots_auto = true;
        o.kv_tokens = None;
        let e = check_tp_options(&o).unwrap_err().to_string();
        for w in ["--slots N", "--kv-tokens N"] {
            assert!(e.contains(w), "{w}: {e}");
        }
        // the prefix cache is forwarded (`docs/GLM_TP.md` 4.2): a group with it pinned is accepted, as one with it off
        assert!(!e.contains("prefix cache"), "{e}");
        let mut o = opts();
        o.slots_auto = false;
        o.kv_tokens = Some(65536);
        o.prefix_cache = PrefixCacheOption::Mib(3584);
        check_tp_options(&o).unwrap();
        let mut o = opts();
        o.slots_auto = false;
        o.kv_tokens = Some(65536);
        o.prefix_cache = PrefixCacheOption::Off;
        o.spec = gb_spec::SpecMode::None;
        o.vision = None;
        check_tp_options(&o).unwrap();
        // the engine's copy drafter (core calls only) stays allowed
        o.drafting = true;
        check_tp_options(&o).unwrap();
        // images: rank 0 may serve them (its tower); a follower with them is refused by name
        o.vision = Some(Default::default());
        check_tp_options(&o).unwrap();
        let mut f = o.clone();
        f.tp.as_mut().unwrap().rank = 1;
        let e = check_tp_options(&f).unwrap_err().to_string();
        assert!(e.contains("images on rank 1") && e.contains("--no-vision") && e.contains("only rank 0"), "{e}");
        f.vision = None;
        check_tp_options(&f).unwrap();
        // the backend's drafters and the NextN layer (lane L12), with and without images on rank 0 (`docs/GLM_TP.md` 11)
        for m in [gb_spec::SpecMode::Mtp, gb_spec::SpecMode::Auto, gb_spec::SpecMode::Copy] {
            o.spec = m;
            check_tp_options(&o).unwrap();
            f.spec = m;
            check_tp_options(&f).unwrap();
        }
        o.vision = None;
        for m in [gb_spec::SpecMode::Mtp, gb_spec::SpecMode::Auto, gb_spec::SpecMode::Copy] {
            o.spec = m;
            check_tp_options(&o).unwrap();
        }
        // one process: nothing to refuse
        let mut one = ServeOptions::new("/nonexistent".into());
        one.slots_auto = true;
        check_tp_options(&one).unwrap();
    }
}
