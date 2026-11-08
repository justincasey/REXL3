//! The GLM-5.3-Flash text forward on the GPU: the model's state (KDA pools, MLA latent caches, DSA planes and rings, page tables), its
//! scratch, and the executor that walks the schedule of `ops.rs`.
//!
//! **Two modes, one arithmetic.** A *window* (decode, or a verify window of drafts) runs the rows of one or several sequences through the
//! 45 layers and leaves each KDA layer's record behind; `commit(slot, keep)` later keeps the first `keep` rows of a sequence's window (the
//! KDA state flips to the buffer the window already wrote when every row is kept, and replays the accepted prefix from the pre-window state
//! when only some are; the MLA latents and DSA planes need nothing: a rejected row is overwritten by the next window before anyone reads
//! it). A *prefill chunk* is the same forward on up to 128 consecutive rows of one sequence followed, layer by layer, by a full-accept
//! commit, so the bits of a prompt row equal the bits of that token decoded alone: chunking changes no bit. Every kernel on the path is
//! row-invariant (`docs/GLM53.md`), so a row's logits do not depend on the other rows of the call, their number, or their position in it.
//!
//! The commit is applied lazily, like the Qwen engine's: `commit` records the keep count and the next forward (or slot change) applies all of
//! them in one pass, with the previous window's index arrays still in place.

use super::alloc::{self, AllocPlan, MAX_CANDIDATES, PF_DENSE_ROWS, RING_LEN, TOKENIZER_VOCAB};
use super::config::GlmConfig;
use super::contract::PrefillContract;
use super::device::{FfnDevice, GlmDevice, LayerDevice, MixerDevice};
use super::ops::{self, MixerKind, Mutant, Op, Site, SiteKind};
use super::vision_serve::SpliceOp;
use crate::error::ModelError;
use crate::forward::Candidates;
use crate::kvtables::{KvTables, BLOCK_TOKENS};
use crate::rows::RowBatch;
use core::ffi::c_void;
use gb_cuda::attention::{BlockTable, RowIndex};
use gb_cuda::exl3_linear::{linear_workspace_bytes, Dtype};
use gb_cuda::mhc::{self, MhcCfg, MhcWork, Tune, Ty};
use gb_cuda::mla::{self, LatentCache, MlaScratch, SparseSel};
use gb_cuda::moe::{self, MoeScratch, OwnScratch, SharedScratch, SigmoidTable};
use gb_cuda::moe_prefill::{PrefillScratch, SharedScratchBig};
use gb_cuda::{dsa, glm_glue, glue, kda, DevBuf, Stream};
use std::collections::BTreeMap;

pub mod dflash;
pub mod graphs;
mod layers;
mod part;
pub mod mtp;
pub mod snapshot;
pub mod tp;
pub use dflash::{DflashAsk, DflashReq, DflashState, TapRelay};
pub use graphs::{GraphKey, GraphMutant, GraphStats, Wait};
pub use mtp::{
    vocab_ranges, BookError, ChainPick, DraftAsk, Drafts, HeadList, MtpCapture, MtpMutant, SubHead, TapBook, VocabMode, MTP_SHARE,
};
pub use snapshot::{PrefixDigest, SnapMeta, SnapPool};

const RMS_EPS: f32 = 1e-5;

/// What a model is built with.
#[derive(Clone, Copy, Debug)]
pub struct ModelOptions {
    pub slots: usize,
    /// Pages of the KV pool (each holds 256 tokens of every attention layer).
    pub kv_blocks: usize,
    pub max_context: usize,
    /// Layers to run (the whole model has 45; a prefix is the bisection against the oracle).
    pub layers: usize,
}

/// How a forward ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Head {
    /// No logits (a prefill chunk that is not the last).
    None,
    /// The logits of every row (a decode or verify window).
    AllRows,
    /// The logits of the last row of each sequence, in sequence order (the end of a prefill: of one prompt, or of several in one forward).
    LastRow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Window,
    Prefill,
}

/// Debug capture of a forward (the oracle comparison): appended per call, concatenated over a prompt's chunks. Rows are in call order.
#[derive(Default)]
pub struct Capture {
    /// Keep only the last `tail` rows of the row-wise captures (a long prompt's dump holds the last 8 rows of every layer: 128 rows of 45 layers
    /// of four streams would otherwise be gigabytes of host memory). The routing is always kept for every row.
    pub tail: Option<usize>,
    /// fp32 `[rows][4096]`: the embedding (stream 0, all four are equal).
    pub embed: Vec<f32>,
    /// fp32 `[rows][4][4096]` per layer: the four streams right after the mixer site.
    pub mid: Vec<Vec<f32>>,
    /// fp32 `[rows][4][4096]` per layer: after the whole block.
    pub layer_out: Vec<Vec<f32>>,
    /// fp32 `[rows][4096]`: the mean of the streams (before the final norm).
    pub hc_head: Vec<f32>,
    /// fp16 bits `[rows][4096]`: the final norm's output.
    pub final_norm: Vec<u16>,
    /// per MoE layer (by layer index): picked expert ids `[rows][8]` and their fp16 gate weights, as the experts consumed them
    pub route_ids: BTreeMap<usize, Vec<i32>>,
    pub route_w: BTreeMap<usize, Vec<u16>>,
}

/// An oracle's routing to replay (forced mode): for MoE layer `layers[i]`, row at absolute position `p` of the prompt takes the eight
/// experts `ids[(i * t + p) * 8 ..]` with the fp16 gate weights `w[same]`.
pub struct ForcedRouting {
    pub layers: Vec<usize>,
    pub t: usize,
    pub ids: Vec<i32>,
    pub w: Vec<u16>,
}

struct Scratch {
    rows: usize,
    ids: DevBuf,
    rowidx: RowIndex,
    seq_start: DevBuf,
    seq_slot: DevBuf,
    keep: DevBuf,
    streams: DevBuf,
    mhc: MhcWork,
    hidden: DevBuf,
    mixer_out: DevBuf,
    kda_qkv: DevBuf,
    kda_out: DevBuf,
    kda_ws: kda::Workspace,
    kda_prefill_rec: kda::Record,
    q_a: DevBuf,
    q_resid: DevBuf,
    q: DevBuf,
    kv_a: DevBuf,
    ckv: DevBuf,
    mla_out: DevBuf,
    k_raw: DevBuf,
    gate: DevBuf,
    wproj: DevBuf,
    iq: DevBuf,
    mla: MlaScratch,
    dsa: dsa::Scratch,
    moe: MoeScratch,
    shared: Option<SharedScratch>,
    dense: Option<SharedScratch>,
    moe_shared_out: DevBuf,
    ws: DevBuf,
    /// The EXL3 decode workspaces of the two side streams (the linears that run on them: `kv_a` on the first, the indexer's `wq_b` on the second).
    ws_s1: DevBuf,
    ws_s2: DevBuf,
    logits: DevBuf,
    topk_ids: DevBuf,
    topk_vals: DevBuf,
    topk_ws: DevBuf,
    /// The large-row scratch of a chunk of more than `MAX_ROWS` rows (the routed experts' sorted pairs and sums, the gated MLPs' pieces).
    pf: Option<PfScratch>,
}

/// What a chunk of more than `MAX_ROWS` rows needs beyond the window scratch (`Scratch::rows` is the chunk's rows).
struct PfScratch {
    moe: PrefillScratch,
    shared: SharedScratchBig,
    dense: Option<SharedScratchBig>,
}

/// Milliseconds by op class over the profiled forwards (see [`GlmModel::profile`]).
#[derive(Default, Debug, Clone)]
pub struct Profile {
    pub ms: BTreeMap<&'static str, f64>,
    /// Milliseconds of the steps inside an op class (the mixers' linears and core, the experts' stages), by name: part of `ms`, not added to it.
    pub sub: BTreeMap<&'static str, f64>,
    pub forwards: usize,
}

/// The profile's name of a routed-experts stage.
/// One P2 MoE layer on a tensor-parallel rank (Lane P step (c), `model/tp/p2.rs`): the rank's pairs and part, the parts' swap queued, then
/// the layer's shared expert when it is owed (`owed`: deferred by `MoeShared` to overlap the swap), then the parts' sum with the shared
/// expert last into `mixer_out`. A chunk of two slabs or more is row-split (`tp::RowSplit`): the shared expert and the sum run on this
/// rank's rows, the output rows are swapped with the peer's, unless the residual is sequence-parallel (`sp_res`: the rank needs only its
/// own rows). `gated`: the owed shared expert's gate linear already ran on this rank's rows (`MoeRoute` under the sequence-parallel
/// residual, beside the hidden rows' gather): only the rest of it runs here (`layers::gated_part`, the same launches).
#[allow(clippy::too_many_arguments)]
fn p2_rank_layer(
    t: &mut tp::GlmTp,
    ex: tp::P2Routed<'_>,
    owed: Option<&gb_cuda::moe::SharedExpert>,
    with_shared: bool,
    s: &mut Scratch,
    rows: usize,
    st: &Stream,
    path: layers::LinPath,
    sp_res: bool,
    gated: bool,
    mark: &mut dyn FnMut(&'static str) -> Result<(), gb_cuda::CudaError>,
) -> Result<(), ModelError> {
    let split = t.p2_row_split(rows);
    let pf = s.pf.as_mut().ok_or_else(|| ModelError::Config("P2 needs the prefill scratch (a chunk above 128 rows)".into()))?;
    let PfScratch { moe: moe_sc, shared: sh_sc, .. } = pf;
    t.p2_begin(ex, &s.hidden, rows, moe_sc, &mut s.moe, split, st, mark)?;
    if let Some(sh) = owed {
        match split {
            Some(sp) => {
                let h = sh.hidden;
                let x = s.hidden.subview(sp.lo * h * 2, sp.rows() * h * 2);
                let mut y = s.moe_shared_out.subview(sp.lo * h * 4, sp.rows() * h * 4);
                if gated {
                    let rest = gb_cuda::glm_p2::GatedPart::Rest;
                    layers::gated_part(sh, &x, sp.rows(), &mut y, s.shared.as_mut(), Some(sh_sc), st, rest)?;
                } else {
                    layers::gated(sh, &x, sp.rows(), &mut y, s.shared.as_mut(), Some(sh_sc), st, path)?;
                }
            }
            None if gated => return Err(ModelError::Config("a shared expert gated on a row split the layer does not have".into())),
            None => layers::gated(sh, &s.hidden, rows, &mut s.moe_shared_out, s.shared.as_mut(), Some(sh_sc), st, path)?,
        }
    }
    let shared = with_shared.then_some(&s.moe_shared_out);
    match split {
        Some(sp) if sp_res => t.p2_finish_own_rows(&ex.dims(), moe_sc, shared, &mut s.mixer_out, sp, st)?,
        _ => t.p2_finish(&ex.dims(), moe_sc, shared, &mut s.mixer_out, split, st)?,
    }
    Ok(())
}

/// Whether a P2 prefill runs a MoE layer's shared expert on the side stream (`REXL3_GLM_P2_SHARED_SIDE=1` / `on`; unset is OFF: MEASURED on
/// node 4, the overlap gave the same prefill rate within 0.2 % at 2k / 8k / 32k tokens, the GPU being busy with the routed experts'
/// GEMMs, so the default keeps one stream). Read once a process.
fn p2_shared_side_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("REXL3_GLM_P2_SHARED_SIDE").as_deref(), Ok("1" | "on" | "true" | "yes")))
}

/// `REXL3_GLM_HEAD_ROWS` (`0`/`off` turns it off; on by default): a prompt forward (P1 or P2) whose last layer is KDA runs that layer's
/// tail on the head rows alone ([`ops::plan_forward_head_rows`]). Never a bit: a P1 or P2 row's bits do not depend on the rows beside it.
static HEAD_ROWS_ON: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| !matches!(std::env::var("REXL3_GLM_HEAD_ROWS").as_deref(), Ok("0" | "off" | "false" | "no")));

/// The rows of one piece of a P2 prompt chunk's chunked KDA recurrence over this scratch (`kda::p2_piece_rows`; 512 when the scratch holds
/// it). `REXL3_GLM_P2_KDA_PIECE` (a multiple of 16, at most the scratch's rows) overrides it for the benches: never a bit, pieces end on
/// chunk boundaries.
fn p2_kda_piece(rec: &kda::Record, ws: &kda::Workspace) -> usize {
    let cap = kda::p2_piece_rows(rec, ws);
    static OVR: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    match *OVR.get_or_init(|| std::env::var("REXL3_GLM_P2_KDA_PIECE").ok().and_then(|v| v.parse().ok())) {
        Some(p) if p > 0 && p % kda::P2_CHUNK_ROWS == 0 && p <= cap => p,
        _ => cap,
    }
}

/// The chunked KDA recurrence's descriptors of a P2 prompt forward: one per segment of [`kda_seg_rows`] (`kda::prefill_descriptor_pieces`
/// of its slot and rows), uploaded by the forward's first KDA layer and kept for the others.
fn kda_seg_descs(b: &RowBatch, piece: usize, cuts: &[Option<usize>]) -> Result<Vec<DevBuf>, ModelError> {
    kda_seg_rows(b, cuts)
        .into_iter()
        .map(|(i, _, rows, _)| Ok(kda::prefill_descriptor_pieces(b.seq_slot[i] as usize, rows, piece)?))
        .collect()
}

/// The recurrence's segments of a prompt forward, `(sequence, first row, rows, ends at the armed cut)`: each sequence of `b` whole, or,
/// with an armed cut inside it (`cuts[i]`: its row in the sequence, [`GlmModel::armed_row`]), in two segments that meet at the cut.
/// A P2 segment boundary on the 16-row grid changes no row's bits (the second segment's chunks are the ones the whole sequence runs).
fn kda_seg_rows(b: &RowBatch, cuts: &[Option<usize>]) -> Vec<(usize, usize, usize, bool)> {
    let mut out = Vec::with_capacity(b.n_seq() + 1);
    for i in 0..b.n_seq() {
        let r = b.seq_rows(i);
        match cuts.get(i).copied().flatten() {
            Some(c) => {
                out.push((i, r.start, c, true));
                out.push((i, r.start + c, r.len() - c, false));
            }
            None => out.push((i, r.start, r.len(), false)),
        }
    }
    out
}

/// Each segment of [`kda_seg_rows`] (`cuts` as there; none for P1's descriptors, one per sequence) with its descriptor.
fn kda_segs<'a>(b: &RowBatch, cuts: &[Option<usize>], descs: &'a [DevBuf]) -> Vec<tp::KdaSeg<'a>> {
    kda_seg_rows(b, cuts)
        .into_iter()
        .zip(descs)
        .map(|((i, row0, rows, cut), desc)| tp::KdaSeg { slot: b.seq_slot[i] as usize, row0, rows, desc, cut })
        .collect()
}

/// The row-by-row KDA recurrence's descriptors of a P1 prompt forward: one per sequence of `b` (`kda::prefill_descriptor` of its slot
/// and rows), uploaded by the forward's first KDA layer and kept for the others.
fn kda_p1_descs(b: &RowBatch) -> Result<Vec<DevBuf>, ModelError> {
    (0..b.n_seq()).map(|i| Ok(kda::prefill_descriptor(b.seq_slot[i] as usize, b.seq_rows(i).len())?)).collect()
}

fn moe_stage_name(stage: &'static str) -> &'static str {
    match stage {
        "group" => "experts: group",
        "rot_in" => "experts: rotate inputs",
        "gemv_gate_up" => "experts: gate/up GEMV",
        "gate_up_epilogue" => "experts: gate/up epilogue",
        "gemv_down" => "experts: down GEMV",
        "down_combine" => "experts: down epilogue + combine",
        "plan" => "experts: plan (sort pairs)",
        "gate_up" => "experts: gate/up GEMM + epilogue",
        "down" => "experts: down GEMM + epilogue",
        "combine" => "experts: combine",
        _ => "experts: other",
    }
}

/// The class an op's time is accounted to.
pub fn op_class(op: &Op) -> &'static str {
    match op {
        Op::Embed | Op::FirstPartials(_) => "embed+first partials",
        Op::Finalize(_) => "mhc finalize",
        Op::ApplyPartials { .. } | Op::Apply => "mhc apply+partials",
        Op::Mixer { kind: MixerKind::Kda, .. } => "kda layer (qkv, kda, o_proj)",
        Op::MixerCore { .. } => "kda layer (qkv, kda), head rows' tail after it",
        Op::HeadRows => "head rows gather",
        Op::MixerOut { .. } => "kda o_proj (head rows)",
        Op::Mixer { kind: MixerKind::MlaDsa, .. } => "mla+dsa layer",
        Op::DenseFfn { .. } => "dense mlp",
        Op::MoeShared { .. } => "moe shared expert",
        Op::MoeRoute { .. } => "moe router",
        Op::MoeExperts { .. } => "moe routed experts",
        Op::HeadMean => "head mean+norm",
        Op::LmHead => "lm_head",
    }
}

/// The last window's layout, until its commit is applied.
struct PendingWindow {
    win: RowBatch,
}

/// What [`GlmModel::forward_begin`] set up: the schedule, the batch's largest position and, under P2, each prompt's `(slot, segment
/// anchor)` in sequence order.
struct Begun {
    plan: Vec<Op>,
    max_pos: usize,
    segs: Vec<(usize, usize)>,
}

pub struct GlmModel {
    pub cfg: GlmConfig,
    pub dev: GlmDevice,
    pub stream: Stream,
    pub max_context: usize,
    pub slots: usize,
    layers: usize,
    /// One pool per KDA layer among the layers run, in layer order.
    pools: Vec<kda::Pool>,
    latent: Vec<LatentCache>,
    planes: Vec<dsa::Planes>,
    /// Window records of every KDA layer (`window_rows` rows each), kept until the window's commit.
    recs: Vec<kda::Record>,
    window_rows: usize,
    kv_tab: KvTables,
    table: BlockTable,
    /// The device block table's contents as last uploaded (`[slots][max_blocks]`), and each slot's length in it: a step uploads only
    /// the entries of the slots whose page list changed.
    table_host: Vec<i32>,
    table_len: Vec<usize>,
    /// The host bytes of the per-step uploads queued on `stream` (row arrays, block-table rows), kept until the stream has run them:
    /// the copies are queued in stream order without blocking the host behind the work already queued.
    stage: gb_cuda::rt::HostStage,
    s: Scratch,
    sig: SigmoidTable,
    kda_dims: kda::Dims,
    kda_params: kda::Params,
    last_window: Option<PendingWindow>,
    /// `commit(slot, keep)` calls waiting to be applied together.
    pending: BTreeMap<usize, usize>,
    /// The last flush's KDA commits run on the side's commit stream (`graphs::commit_side_on`): the next flush joins them first.
    commits_on_side: bool,
    /// Per KDA layer / MLA layer: ordinal of layer `l` among its kind.
    kda_ord: Vec<usize>,
    mla_ord: Vec<usize>,
    /// Debug: capture of the next forwards, and an oracle's routing to replay.
    pub capture: Option<Capture>,
    pub forced: Option<ForcedRouting>,
    /// Debug: use the oracle's rounding where the pack differs from the stored function (`dt_bias` as bf16 is a load-time change and is not
    /// done here; this switches the KDA core output to round toward zero, the fork's kernel).
    pub oracle_rounding: bool,
    /// Debug: time every op of the forwards that follow with CUDA events (host-synchronising at the end of each forward) and add the
    /// milliseconds to the class of the op ([`op_class`]).
    pub profile: Option<Profile>,
    /// Debug (the mutation tests): run with a deliberate bug, to show a test notices it. Never set in serving.
    pub mutant: Option<Mutant>,
    /// The image rows of the prefill chunk being run, written over the embedding rows right after the embedding (`prefill_chunk_mm`; empty for
    /// every text call, in which case the embedding stage launches exactly what it always did).
    splice: Vec<SpliceOp>,
    /// The NextN (MTP) drafter's layer, state and scratch: present exactly when the weights carry the layer (`dev.mtp`).
    mtp: Option<mtp::MtpState>,
    /// The DFlash2 drafter (`--draft DIR`, `attach_dflash`): its taps, context ring and scratch. Rank 0 alone on a tensor-parallel group.
    dflash: Option<dflash::DflashState>,
    /// On both ranks of a group whose rank 0 drafts with DFlash2 (`enable_tap_relay`): the stage the drafter's taps of the peer's rows
    /// cross in under the sequence-parallel residual of a P2 prompt chunk (`dflash::TapRelay`).
    tap_relay: Option<dflash::TapRelay>,
    /// The CUDA graphs of the decode windows (see `graphs.rs`).
    graphs: graphs::GraphCache,
    /// The shared expert's side stream and its two events (fork: the layer's input is ready, join: the expert's output is).
    side: Option<graphs::Side>,
    /// While a window runs through the graph cache: the launch arguments that do not depend on the context length (`graphs::RunShape`). `None`
    /// for everything else (a prefill chunk, a window with graphs off): the executor then takes them from the batch's own positions.
    run_shape: Option<graphs::RunShape>,
    /// Debug (the oracle dump): the logits of every row of a prefill chunk of more than `MAX_ROWS` rows with [`Head::AllRows`], which the window-sized
    /// `s.logits` cannot hold (`enable_dump_logits`; nothing is allocated unless a dump asks).
    dump_logits: Option<DevBuf>,
    /// The last head wrote `dump_logits`, not `s.logits`.
    dump_head: bool,
    /// The prefix cache's snapshot slots (`enable_prefix_cache`; `snapshot.rs`).
    snaps: Option<snapshot::SnapPool>,
    /// Snapshots to take INSIDE the prefill chunk being run (`prefill_chunk_cuts`): the state after the first `row` rows of the chunk goes to snapshot slot
    /// `snap`. Ascending, each strictly inside the chunk, only ever set for a chunk of more than `MAX_ROWS` rows. Empty for every other call.
    snap_cuts: Vec<snapshot::InChunkCut>,
    /// Per slot, the snapshot to take inside the slot's next P2 prompt forward that passes its length ([`GlmModel::arm_cut`], the
    /// tensor-parallel end cut: `snapshot.rs`): every KDA layer splits that sequence's chunked recurrence there and copies its conv
    /// history out between the two segments; the state and the rest are copied after the forward ([`GlmModel::finish_cut`]). `None` for
    /// every slot otherwise.
    armed: Vec<Option<snapshot::ArmedCut>>,
    /// Under P2, each slot's open prefill SEGMENT (`docs/GLM_PREFILL_P2.md` 8.1): `(anchor, next)`, the segment's first position and the
    /// position after its last prefilled row. The KDA chunks of a segment are 16-row runs from its anchor, so a prompt chunk that continues
    /// the segment (starts at `next`) must start a whole number of chunks after the anchor. `None` after `begin_slot`, `end_slot` and any
    /// window on the slot (the next prompt chunk starts a new segment); a prefix restore opens one at the restored length.
    p2_seg: Vec<Option<(usize, usize)>>,
    /// A rank of a tensor-parallel group (`model/tp.rs`, `enable_tp`): the collective and this rank's share of the routed experts. `None` (every
    /// single-node model) runs exactly what it always ran: nothing below reads it unless it is set.
    tp: Option<tp::GlmTp>,
    /// The subbank pairs stage's pick table (`OwnScratch`, `max(chunk rows, MAX_ROWS) x topk` i32 and an `[experts]` table: under 1 MiB),
    /// present exactly when some layer's routed experts bound as subbanks (`FfnDevice::MoeSubbanks`: a mixed-width pack, or a rank's
    /// share). Outside the allocation plan, like a rank's own table (`tp::GlmTp`), so a uniform pack's plan is unchanged.
    own_s: Option<OwnScratch>,
    /// The prefill arithmetic contract the model's prompt chunks run (`contract.rs`): the process's at construction
    /// (`REXL3_GLM_PREFILL_CONTRACT`, P1 by default). Under P2 every dense EXL3 linear of a prompt chunk takes the tensor-core GEMM
    /// (`layers::LinPath::P2`) whatever the chunk's rows; windows are P1 under both.
    prefill: PrefillContract,
    /// Under P2, a MoE layer's shared expert on the side stream beside the routed experts (`REXL3_GLM_P2_SHARED_SIDE=1`, default off;
    /// [`GlmModel::set_p2_shared_side`] for the benches): never a bit, only what overlaps in time.
    p2_side_on: bool,
    /// Under P2, the prompt rows' KDA recurrence chunked (`kda::prefill_chunked`, the contract's arithmetic: true) or P1's row-by-row path
    /// (false: the benches' A/B of step (e) on one load, [`GlmModel::set_p2_kda_chunked`]; NOT P2's tag, never a server setting).
    p2_kda_chunked: bool,
    /// Prompt chunks this model ran under the sequence-parallel residual (`run`'s `sp`; the tests' audit).
    sp_chunks: u64,
    /// A prompt forward whose last layer is KDA runs that layer's tail on the head rows alone (`REXL3_GLM_HEAD_ROWS`, on by default;
    /// [`GlmModel::set_head_rows`]; see [`GlmModel::head_rows_tail`]).
    head_rows_on: bool,
    /// The group's drafter taps the last trunk layer ([`GlmModel::keep_tapped_rows`]): every row's tail then runs.
    last_tapped: bool,
    /// Prompt forwards that ran the last layer's tail on the head rows alone (the tests' audit).
    head_rows_runs: u64,
    /// The prompt forwards paused at a layer boundary (`prefill_rows` with a layer bound, `part.rs`; a resume goes on with the last) and
    /// the stashes of the rows another forward may overwrite while one waits (`enable_prefill_parts`; empty: parts are off).
    paused: Vec<part::Paused>,
    stash: Vec<part::Stash>,
    /// `REXL3_GLM_PROFILE_SLICES` (`profile_slices.rs`): the forward calls a rank's runtime brackets with the profiler API. Host
    /// counters only; inert without the setting.
    pub(in crate::families::glm) slices: crate::families::glm::profile_slices::SliceProfiler,
}

fn zeroed(bytes: usize) -> Result<DevBuf, ModelError> {
    Ok(DevBuf::zeroed(bytes)?)
}

impl GlmModel {
    /// Allocate the state and scratch of a model over `dev` (its weights, already resident), for prefill chunks of at most `MAX_ROWS` rows.
    pub fn new(cfg: GlmConfig, dev: GlmDevice, o: &ModelOptions) -> Result<Self, ModelError> {
        Self::new_chunked(cfg, dev, o, alloc::MAX_ROWS)
    }

    /// [`GlmModel::new`] with scratch for prefill chunks of up to `chunk_rows` rows (`MAX_ROWS..=gb_cuda::moe_prefill::MAX_ROWS`). A chunk of
    /// more than `MAX_ROWS` rows runs the large-row kernels, which give every row the bits the decode kernels would (so no bit depends on
    /// `chunk_rows`); decode and verify windows are untouched.
    pub fn new_chunked(cfg: GlmConfig, dev: GlmDevice, o: &ModelOptions, chunk_rows: usize) -> Result<Self, ModelError> {
        let layers = dev.layers.len();
        if o.layers != layers {
            return Err(ModelError::Config(format!("a model of {} layers over weights of {layers}", o.layers)));
        }
        if o.slots == 0 || o.kv_blocks == 0 || o.max_context == 0 {
            return Err(ModelError::Config("a model needs slots, KV pages and a context".into()));
        }
        let max_blocks = o.max_context.div_ceil(BLOCK_TOKENS);
        let r = alloc::MAX_ROWS;
        if chunk_rows < r || chunk_rows > gb_cuda::moe_prefill::MAX_ROWS {
            return Err(ModelError::Config(format!(
                "a prefill chunk takes {r}..={} rows, not {chunk_rows}",
                gb_cuda::moe_prefill::MAX_ROWS
            )));
        }
        let cr = chunk_rows;
        // every head's state, or a KDA head split rank's heads only (`KdaConfig::rank`: its pools, records and workspace from the start)
        let dims = alloc::dims_of(&cfg).map_err(ModelError::Config)?;
        // Blackwell programmatic dependent launch (pdl.cuh): a launch may start while its predecessor still runs and reads the
        // predecessor's writes only after `pdl_wait`, so only timing changes. Default every family (7); `REXL3_GLM_PDL` (0..7) overrides.
        let pdl = std::env::var("REXL3_GLM_PDL").ok().and_then(|v| v.trim().parse::<i32>().ok()).unwrap_or(7) & 7;
        gb_cuda::exl3_linear::set_pdl_mask(pdl).map_err(|e| ModelError::Config(format!("pdl mask: {e}")))?;
        let kda_layers: Vec<usize> = (0..layers).filter(|&l| ops::mixer_kind(&cfg, l) == MixerKind::Kda).collect();
        let mla_layers: Vec<usize> = (0..layers).filter(|&l| ops::mixer_kind(&cfg, l) == MixerKind::MlaDsa).collect();
        let mut kda_ord = vec![usize::MAX; layers];
        for (i, &l) in kda_layers.iter().enumerate() {
            kda_ord[l] = i;
        }
        let mut mla_ord = vec![usize::MAX; layers];
        for (i, &l) in mla_layers.iter().enumerate() {
            mla_ord[l] = i;
        }
        let mut pools = Vec::new();
        let mut recs = Vec::new();
        for _ in &kda_layers {
            pools.push(kda::Pool::new(dims, o.slots)?);
            recs.push(kda::Record::alloc(dims, alloc::WINDOW_ROWS)?);
        }
        let mut latent = Vec::new();
        let mut planes = Vec::new();
        for _ in &mla_layers {
            latent.push(LatentCache::new(o.kv_blocks, Dtype::F16, cfg.latent_kv)?);
            planes.push(dsa::Planes::new(o.kv_blocks, o.slots, RING_LEN)?);
        }
        let moe_dims = alloc::moe_dims(&cfg);
        let shared_inter = cfg.ffn.moe_intermediate * cfg.ffn.shared_experts;
        // the shared expert's shapes size its scratch: the first MoE layer's, or the NextN layer's for a model whose trunk prefix has none
        let first_moe = dev
            .layers
            .iter()
            .find_map(|l| match &l.ffn {
                FfnDevice::Moe { shared, .. } | FfnDevice::MoeSubbanks { shared, .. } => Some(shared),
                _ => None,
            })
            .or_else(|| dev.mtp.as_ref().map(|m| &m.shared));
        let first_dense = dev.layers.iter().find_map(|l| match &l.ffn {
            FfnDevice::Dense(m) => Some(m),
            _ => None,
        });
        let exl3_ws = alloc::exl3_workspace_bytes(&cfg, cr);
        let h = cfg.hidden;
        let m = &cfg.mla;
        let ix = &cfg.indexer;
        let ch = cfg.kda.heads * cfg.kda.head_dim;
        let zv = DevBuf::zeroed_view;
        let s = Scratch {
            rows: cr,
            ids: zeroed(cr * 4)?,
            rowidx: RowIndex { slot: zv(cr * 4)?, pos: zv(cr * 4)?, rows: cr },
            seq_start: zeroed((cr + 1) * 4)?,
            seq_slot: zeroed(cr * 4)?,
            keep: zeroed(r * 4)?,
            streams: zv(cr * mhc::ROW * 4)?,
            mhc: MhcWork::new(cr)?,
            hidden: zv(cr * h * 2)?,
            mixer_out: zv(cr * h * 4)?,
            kda_qkv: zv(cr * 3 * ch * 4)?,
            kda_out: zv(cr * ch * 2)?,
            kda_ws: {
                let mut ws = kda::Workspace::alloc(dims, alloc::kda_prefill_rows(r, cr))?;
                if alloc::kda_pipe_rows(r, cr) > 0 {
                    ws.add_pipe()?;
                }
                ws
            },
            kda_prefill_rec: kda::Record::alloc(dims, alloc::kda_prefill_rows(r, cr))?,
            // views: a row-split rank cuts its rows out of them (`layers::mla_dsa_mixer`)
            q_a: zv(cr * m.q_lora_rank * 2)?,
            q_resid: zv(cr * m.q_lora_rank * 2)?,
            q: zv(cr * m.heads * m.qk_head_dim() * 2)?,
            kv_a: zeroed(cr * m.kv_lora_rank * 2)?,
            ckv: zeroed(cr * m.kv_lora_rank * 2)?,
            mla_out: zv(cr * m.heads * m.v_head_dim * 2)?,
            k_raw: zv(cr * ix.head_dim * 2)?,
            gate: zv(cr * ix.head_dim * 2)?,
            wproj: zv(cr * ix.heads * 2)?,
            iq: zv(cr * ix.heads * ix.head_dim * 2)?,
            mla: MlaScratch::new(r, o.max_context, Dtype::F16)?,
            dsa: dsa::Scratch::new(o.max_context)?,
            moe: MoeScratch::new(&moe_dims, r)?,
            shared: match first_moe {
                Some(sh) => Some(SharedScratch::new(sh, r)?),
                None => None,
            },
            dense: match first_dense {
                Some(d) => Some(SharedScratch::new(d, r)?),
                None => None,
            },
            moe_shared_out: zv(cr * h * 4)?,
            ws: zeroed(exl3_ws)?,
            ws_s1: zeroed(linear_workspace_bytes(r, h, m.kv_lora_rank + m.qk_rope_head_dim, 2))?,
            ws_s2: zeroed(linear_workspace_bytes(r, m.q_lora_rank, ix.heads * ix.head_dim, 2))?,
            // a view of a block of its own: the lm_head's vocabulary halves of a split rank are its two ranges (`model/tp.rs`)
            logits: zv(r * cfg.vocab * 2)?,
            topk_ids: zeroed(r * MAX_CANDIDATES * 4)?,
            topk_vals: zeroed(r * MAX_CANDIDATES * 4)?,
            topk_ws: zeroed(glm_glue::topk_workspace_bytes(r, TOKENIZER_VOCAB, MAX_CANDIDATES))?,
            pf: if cr > r {
                let sh =
                    first_moe.ok_or_else(|| ModelError::Config("a large prefill chunk needs a shared expert's shapes".into()))?;
                // the multi-block plan: the single-block one's bytes (`moe_prefill.rs` tests), without scanning every pair once per expert
                // in one block, which a mixed-width layer does once per subbank
                let mut moe = PrefillScratch::new(&moe_dims, cr)?;
                moe.use_parallel_plan(true);
                Some(PfScratch {
                    moe,
                    shared: SharedScratchBig::new(sh, cr)?,
                    dense: match first_dense {
                        Some(d) => Some(SharedScratchBig::new(d, cr.min(PF_DENSE_ROWS))?),
                        None => None,
                    },
                })
            } else {
                None
            },
        };
        // a model whose scratch was sized for another shared-expert width than its layers' would fail at run time with a size error: say so now
        if shared_inter != cfg.ffn.moe_intermediate * cfg.ffn.shared_experts {
            return Err(ModelError::Config("shared expert width".into()));
        }
        let mtp = match &dev.mtp {
            Some(_) => Some(mtp::MtpState::new(&cfg, o.kv_blocks, o.slots, cr)?),
            None => None,
        };
        let own_s = if dev.layers.iter().any(|l| matches!(l.ffn, FfnDevice::MoeSubbanks { .. }))
            || dev.mtp.as_ref().is_some_and(|m| matches!(m.moe, crate::families::glm::device::MtpMoe::Subbanks(_)))
        {
            Some(OwnScratch::new(&moe_dims, s.rows.max(alloc::MAX_ROWS))?)
        } else {
            None
        };
        let mut model = Self {
            stream: Stream::new()?,
            max_context: o.max_context,
            slots: o.slots,
            layers,
            pools,
            latent,
            planes,
            recs,
            window_rows: alloc::WINDOW_ROWS,
            kv_tab: KvTables::new(o.kv_blocks, o.slots, max_blocks),
            table: BlockTable::from_host(&vec![Vec::new(); o.slots], max_blocks)?,
            table_host: vec![0; o.slots.max(1) * max_blocks],
            table_len: vec![0; o.slots],
            stage: gb_cuda::rt::HostStage::new(),
            s,
            sig: SigmoidTable::new()?,
            kda_dims: dims,
            kda_params: kda::Params::glm(),
            last_window: None,
            pending: BTreeMap::new(),
            commits_on_side: false,
            kda_ord,
            mla_ord,
            capture: None,
            forced: None,
            oracle_rounding: false,
            profile: None,
            mutant: None,
            splice: Vec::new(),
            dump_logits: None,
            dump_head: false,
            snaps: None,
            snap_cuts: Vec::new(),
            armed: vec![None; o.slots],
            p2_seg: vec![None; o.slots],
            tp: None,
            own_s,
            mtp,
            dflash: None,
            tap_relay: None,
            graphs: graphs::GraphCache::new(),
            side: graphs::Side::new(),
            run_shape: None,
            prefill: PrefillContract::P1,
            p2_side_on: p2_shared_side_on(),
            p2_kda_chunked: true,
            sp_chunks: 0,
            head_rows_on: *HEAD_ROWS_ON,
            last_tapped: false,
            head_rows_runs: 0,
            paused: Vec::new(),
            stash: Vec::new(),
            slices: crate::families::glm::profile_slices::SliceProfiler::from_env()?,
            cfg,
            dev,
        };
        // the process's contract (P1 unless REXL3_GLM_PREFILL_CONTRACT says otherwise); P2 is checked against the weights and the scratch here,
        // before any chunk can run it
        model.set_prefill_contract(super::contract::current())?;
        Ok(model)
    }

    /// Every allocation the model owns beyond the weights, by the names of `alloc::AllocPlan` and in bytes, summed over the real buffers.
    pub fn allocations(&self) -> Vec<(&'static str, u64)> {
        let sum = |v: &[&DevBuf]| v.iter().map(|b| b.bytes() as u64).sum::<u64>();
        let s = &self.s;
        let rec = |r: &kda::Record| sum(&[&r.qkv, &r.g, &r.beta]);
        let moe_bytes = s.moe.bytes() as u64;
        let mut v = vec![
            ("kda_pools", self.pools.iter().map(|p| sum(&[&p.s, &p.parity, &p.conv])).sum()),
            ("dsa_rings", self.planes.iter().map(|p| p.ring.bytes() as u64).sum()),
            ("block_table", self.table.buf.bytes() as u64),
            ("mla_latent", self.latent.iter().map(|c| c.lat.bytes() as u64).sum()),
            ("dsa_pools", self.planes.iter().map(|p| p.pool.bytes() as u64).sum()),
            ("kda_window_records", self.recs.iter().map(rec).sum()),
            ("kda_prefill_record", rec(&s.kda_prefill_rec)),
            ("kda_workspace", kda::Workspace::bytes_per_row(self.kda_dims) as u64 * s.kda_ws.rows as u64),
            ("kda_pipe", if s.kda_ws.pipelined() { kda::Workspace::pipe_bytes_per_row(self.kda_dims) as u64 * s.kda_ws.rows as u64 } else { 0 }),
            ("row_arrays", sum(&[&s.ids, &s.rowidx.slot, &s.rowidx.pos, &s.seq_start, &s.seq_slot, &s.keep])),
            ("streams", s.streams.bytes() as u64),
            ("mhc_work", sum(&[&s.mhc.part, &s.mhc.post, &s.mhc.comb])),
            ("hidden", s.hidden.bytes() as u64),
            ("mixer_out", s.mixer_out.bytes() as u64),
            ("kda_qkv", s.kda_qkv.bytes() as u64),
            ("kda_out", s.kda_out.bytes() as u64),
            ("mla_acts", sum(&[&s.q_a, &s.q_resid, &s.q, &s.kv_a, &s.ckv, &s.mla_out])),
            ("indexer_acts", sum(&[&s.k_raw, &s.gate, &s.wproj, &s.iq])),
            ("mla_scratch", sum(&[&s.mla.q_lat, &s.mla.o_lat, &s.mla.ws])),
            ("dsa_scratch", s.dsa.bytes() as u64),
            ("moe_scratch", moe_bytes),
            ("moe_prefill_scratch", s.pf.as_ref().map_or(0, |p| p.moe.bytes() as u64)),
            ("shared_prefill_scratch", s.pf.as_ref().map_or(0, |p| p.shared.bytes() as u64)),
            ("dense_prefill_scratch", s.pf.as_ref().and_then(|p| p.dense.as_ref()).map_or(0, |d| d.bytes() as u64)),
            ("shared_scratch", s.shared.as_ref().map_or(0, |x| x.bytes() as u64)),
            ("dense_scratch", s.dense.as_ref().map_or(0, |x| x.bytes() as u64)),
            ("moe_shared_out", s.moe_shared_out.bytes() as u64),
            ("exl3_workspace", s.ws.bytes() as u64),
            ("exl3_workspace_side", sum(&[&s.ws_s1, &s.ws_s2])),
            ("logits", s.logits.bytes() as u64),
            ("topk", sum(&[&s.topk_ids, &s.topk_vals, &s.topk_ws])),
        ];
        if let Some(sp) = &self.snaps {
            v.push(("prefix_snapshots", sp.allocated_bytes()));
        }
        if let Some(m) = &self.mtp {
            // the NextN layer's cache, rings and pools are shares of the shared names; its taps and scratch are items of their own
            for (name, bytes) in m.allocations() {
                match v.iter_mut().find(|(n, _)| *n == name) {
                    Some(e) => e.1 += bytes,
                    None => v.push((name, bytes)),
                }
            }
        }
        v
    }

    /// The model's allocations equal the plan's (for a model of every layer; a prefix has fewer layers' state).
    pub fn check_allocations(&self, plan: &AllocPlan) -> Result<(), String> {
        let mut bad = Vec::new();
        for (name, got) in self.allocations() {
            let want = plan.get(name);
            if got != want {
                bad.push(format!("{name}: allocated {got} B, planned {want} B"));
            }
        }
        if bad.is_empty() {
            Ok(())
        } else {
            Err(bad.join("; "))
        }
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    /// Pages of the KV pool no slot and no snapshot holds (the books the prefix-cache tests balance).
    pub fn free_pages(&self) -> usize {
        self.kv_tab.free_blocks()
    }

    /// The pages `slot` holds, in position order, and how many holders (slots and snapshots) each has (a test's view of sharing).
    pub fn slot_pages(&self, slot: usize) -> Vec<(u32, u32)> {
        self.kv_tab.blocks_of(slot).iter().map(|&b| (b, self.kv_tab.refcount(b))).collect()
    }

    /// Debug (the oracle dump): allocate the buffer that holds the logits of every row of a chunk of more than `MAX_ROWS` rows (`chunk rows x vocab x 2`
    /// bytes, 634 MB at 2,048), so that [`Head::AllRows`] works on one. Not part of the serving memory plan.
    pub fn enable_dump_logits(&mut self) -> Result<(), ModelError> {
        if self.dump_logits.is_none() {
            self.dump_logits = Some(DevBuf::zeroed_view(self.s.rows * self.cfg.vocab * 2)?);
        }
        Ok(())
    }

    /// The most rows a prefill chunk can take (the scratch's).
    pub fn chunk_rows(&self) -> usize {
        self.s.rows
    }

    /// Under P2, run a MoE layer's shared expert on the side stream (true) or the main stream (false); the bits are the same either way.
    pub fn set_p2_shared_side(&mut self, on: bool) {
        self.p2_side_on = on;
    }

    /// The benches' switch of step (e): under P2, the chunked KDA recurrence (true, the contract) or P1's row-by-row path (false, the
    /// "before" arm of an A/B on one load: its bits are not P2's tag). Never a bit of a P1 chunk or a window.
    #[doc(hidden)]
    pub fn set_p2_kda_chunked(&mut self, on: bool) {
        self.p2_kda_chunked = on;
    }

    /// Run a prompt forward's last-layer tail on the head rows alone (true) or on every row (false); the bits are the same either way.
    /// Every rank of a group sets the same.
    pub fn set_head_rows(&mut self, on: bool) {
        self.head_rows_on = on;
    }

    /// The trunk layers the group's drafter taps (on every rank of a group: a fact of the Hello, though rank 0 alone drafts). A tap of the
    /// last layer reads every row's streams after its tail, so that tail then runs on every row.
    pub fn keep_tapped_rows(&mut self, taps: &[usize]) {
        self.last_tapped |= self.layers > 0 && taps.contains(&(self.layers.min(self.cfg.layers.len()) - 1));
    }

    /// Prompt forwards that ran the last layer's tail on the head rows alone (the tests' audit).
    pub fn head_rows_runs(&self) -> u64 {
        self.head_rows_runs
    }

    /// Whether this prompt forward runs its last layer's tail on the head rows alone ([`ops::plan_forward_head_rows`]): a prompt forward
    /// (P1, or P2 on its chunked KDA recurrence) whose head is each sequence's last row (or none) and whose rows after the last recurrence
    /// nothing else reads (no NextN drafter, which taps every row's final norm; no drafter's tap of the last layer; no capture or forced
    /// routing; no snapshot inside the chunk). Every input is a fact both ranks of a group share, so they skip the same exchanges. Under
    /// P2 the tail is a P2 forward of the head rows: every P2 op is row-independent at any rows, so they are the whole chunk's bits.
    fn head_rows_tail(&self, b: &RowBatch, mode: Mode, head: Head) -> bool {
        self.head_rows_on
            && mode == Mode::Prefill
            && head != Head::AllRows
            && (!self.prefill.p2_prefill(true) || self.p2_kda_chunked)
            && self.mtp.is_none()
            && !self.last_tapped
            && self.capture.is_none()
            && self.forced.is_none()
            && self.snap_cuts.is_empty()
            && b.n_seq() <= alloc::MAX_ROWS
    }

    /// The sequence-parallel residual's cut of a forward of `rows` rows in `mode` (`run`'s `sp`): a large P2 prompt chunk on a rank of two
    /// with the KDA head split ([`tp::GlmTp::p2_row_split`]), never with a capture, forced routing, an image splice, a snapshot inside the
    /// chunk, a drafter without the taps' relay, a NextN mutant, or `REXL3_GLM_P2_SP` off. `None`: the residual holds every row.
    fn sp_split(&self, rows: usize, mode: Mode) -> Option<tp::RowSplit> {
        let p2 = self.prefill.p2_prefill(mode == Mode::Prefill);
        let big = mode == Mode::Prefill && (rows > alloc::MAX_ROWS || (!self.snap_cuts.is_empty() && self.s.pf.is_some()));
        let on = p2
            && big
            && self.capture.is_none()
            && self.snap_cuts.is_empty()
            && self.splice.is_empty()
            && self.forced.is_none()
            && (self.dflash.is_none() || self.tap_relay.is_some())
            && self.mtp.as_ref().is_none_or(|m| m.mutant.is_none())
            && super::contract::p2_sp_on();
        if !on {
            return None;
        }
        self.tp.as_ref().filter(|t| t.split_bits() & tp::SPLIT_KDA != 0).and_then(|t| t.p2_row_split(rows))
    }

    /// The prefill contract the model's prompt chunks run (`contract.rs`).
    pub fn prefill_contract(&self) -> PrefillContract {
        self.prefill
    }

    /// Run the model's prompt chunks under `c` from now on. The model takes the process's contract at construction; tests and tools switch
    /// it to compare P1 and P2 on one load (a server never calls this: its contract is the process's, a fact of its TP Hello and its prefix
    /// namespace). Refused, by name, for a model with the prefix cache or a tensor-parallel rank (their namespace or Hello carries the
    /// process's contract), and for P2 when a dense linear or a scratch does not take the P2 GEMM ([`GlmModel::check_p2`]).
    pub fn set_prefill_contract(&mut self, c: PrefillContract) -> Result<(), ModelError> {
        if c == self.prefill {
            return Ok(());
        }
        if self.snaps.is_some() || self.tp.is_some() {
            return Err(ModelError::Config(format!(
                "the prefill contract of a model with the prefix cache or a tensor-parallel rank is the process's ({}={})",
                super::contract::CONTRACT_VAR,
                self.prefill.id()
            )));
        }
        if c == PrefillContract::P2 {
            super::contract::check_built(c).map_err(ModelError::Config)?;
            self.check_p2()?;
        }
        self.prefill = c;
        Ok(())
    }

    /// `Ok` when every dense EXL3 linear a P2 prompt chunk runs on the tensor-core GEMM is of a shape and format it takes, and every workspace
    /// it runs in holds a chunk's rotated input at the widest K: a P2 boot is refused by name here, never mid-prompt.
    pub fn check_p2(&self) -> Result<(), ModelError> {
        use gb_cuda::exl3_linear::Exl3Weights;
        use gb_cuda::glm_p2;
        let bad = |what: String| ModelError::Config(format!("{}=p2: {what}", super::contract::CONTRACT_VAR));
        let mut lins: Vec<(String, &Exl3Weights)> = Vec::new();
        // (name, the MLP, its decode-size scratch, its prefill-size scratch)
        let s = &self.s;
        let pf_dense = s.pf.as_ref().and_then(|p| p.dense.as_ref());
        let pf_shared = s.pf.as_ref().map(|p| &p.shared);
        type Mlp<'a> = (String, &'a gb_cuda::moe::SharedExpert, Option<&'a SharedScratch>, Option<&'a SharedScratchBig>);
        let mut mlps: Vec<Mlp<'_>> = Vec::new();
        for (l, d) in self.dev.layers.iter().enumerate() {
            match &d.mixer {
                MixerDevice::Kda(w) => {
                    lins.push((format!("layer {l} kda qkv_proj"), &w.qkv));
                    lins.push((format!("layer {l} kda o_proj"), &w.o));
                }
                MixerDevice::Mla(w) => {
                    for (n, x) in [("q_a", &w.q_a), ("q_b", &w.q_b), ("kv_a", &w.kv_a), ("o", &w.o)] {
                        lins.push((format!("layer {l} mla {n}"), x));
                    }
                }
            }
            match &d.ffn {
                FfnDevice::Dense(m) => mlps.push((format!("layer {l} dense mlp"), m, s.dense.as_ref(), pf_dense)),
                FfnDevice::Moe { shared, .. } | FfnDevice::MoeSubbanks { shared, .. } => {
                    mlps.push((format!("layer {l} shared expert"), shared, s.shared.as_ref(), pf_shared))
                }
            }
        }
        if let Some(w) = self.dev.mtp.as_ref() {
            lins.push(("the NextN eh_proj".into(), &w.eh_proj));
            let a = &*w.mixer;
            for (n, x) in [("q_a", &a.q_a), ("q_b", &a.q_b), ("kv_a", &a.kv_a), ("o", &a.o)] {
                lins.push((format!("the NextN mla {n}"), x));
            }
            mlps.push(("the NextN shared expert".into(), &w.shared, s.shared.as_ref(), pf_shared));
        }
        for (name, m, small, big) in &mlps {
            if !glm_p2::gated_supported(m) {
                return Err(bad(format!(
                    "{name} ({} -> {} -> {}) is not a shape or format the P2 GEMM takes",
                    m.hidden, m.inter, m.hidden
                )));
            }
            // a chunk of at most MAX_ROWS rows runs in the decode-size scratch, a larger one in the prefill-size scratch (in its pieces)
            let small_ok = small.is_some_and(|sc| sc.p2_ok(m));
            let big_ok = big.is_some_and(|sc| sc.p2_ok(m));
            if !small_ok || (s.rows > alloc::MAX_ROWS && !big_ok) {
                return Err(bad(format!(
                    "{name}: no scratch that holds a P2 piece (decode-size {small_ok}, prefill-size {big_ok})"
                )));
            }
        }
        // the routed experts (step 3): every MoE layer's bank (and the NextN layer's) on the P2 expert GEMMs, whose picks and sorted pairs
        // live in the prefill scratch at every chunk width
        if self.s.pf.is_none() {
            return Err(bad(format!(
                "the routed experts run in the prefill scratch, which a model built for chunks of {} rows does not have (more than {} rows: \
                 REXL3_GLM_CHUNK_ROWS / --prefill-chunk)",
                self.s.rows,
                alloc::MAX_ROWS
            )));
        }
        // a layer bound as subbanks (a mixed-width layer, or a rank's share) is checked bank by bank: every subbank of one gate/up width on
        // the P2 expert GEMMs, a split gate/up subbank through the decode pairs stage (`glm_p2_moe::supported_subbanks`)
        let mut banks: Vec<(String, &gb_cuda::moe::MoeExperts)> = Vec::new();
        let mut subbanked: Vec<(String, &gb_cuda::moe::MoeSubbanks)> = Vec::new();
        for (l, d) in self.dev.layers.iter().enumerate() {
            match &d.ffn {
                FfnDevice::Moe { layer: m, .. } => banks.push((format!("layer {l}'s routed experts"), &m.experts)),
                FfnDevice::MoeSubbanks { layer: m, .. } => subbanked.push((format!("layer {l}'s routed experts"), &m.experts)),
                FfnDevice::Dense(_) => {}
            }
        }
        if let Some(w) = self.dev.mtp.as_ref() {
            match &w.moe {
                super::device::MtpMoe::Single(m) => banks.push(("the NextN layer's routed experts".into(), &m.experts)),
                super::device::MtpMoe::Subbanks(m) => subbanked.push(("the NextN layer's routed experts".into(), &m.experts)),
            }
        }
        for (name, ex) in &banks {
            if !gb_cuda::glm_p2_moe::supported_bank(ex) {
                return Err(bad(format!(
                    "{name} ({} / {} bits, {:?}) are not a format the P2 expert GEMMs take",
                    ex.bits_gu, ex.bits_d, ex.codebook
                )));
            }
        }
        for (name, sb) in &subbanked {
            // on one node and on a rank alike (`glm_p2_moe::pairs_subbanks_with` / `pairs_subbanks_rank`)
            gb_cuda::glm_p2_moe::supported_subbanks(sb).map_err(|why| bad(format!("{name} (subbanks): {why}")))?;
        }
        let rows = self.s.rows;
        for (name, w) in &lins {
            if !glm_p2::supported(w) {
                return Err(bad(format!(
                    "{name} ({} x {}, {} bits, {:?}) is not a shape or format the P2 GEMM takes",
                    w.k, w.n, w.bits, w.codebook
                )));
            }
            if self.s.ws.bytes() < glm_p2::workspace_bytes(rows, w.k) {
                return Err(bad(format!(
                    "the EXL3 workspace holds {} bytes, {name} needs {} at {rows} rows",
                    self.s.ws.bytes(),
                    glm_p2::workspace_bytes(rows, w.k)
                )));
            }
        }
        Ok(())
    }

    pub fn sync(&self) -> Result<(), ModelError> {
        self.stream.sync()?;
        Ok(())
    }

    // ------------------------------------------------------------------------------------------------------ slots and commits

    /// Start a sequence in `slot`: zero its KDA state, drop any pages it held. (The DSA rings and pool planes need no reset: nothing reads
    /// a ring row or a pool entry of positions the sequence has not written.)
    pub fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        self.drop_paused_of(slot);
        self.flush()?;
        let st = &self.stream;
        for p in &mut self.pools {
            p.reset_slot(slot, st)?;
        }
        self.kv_tab.release(slot);
        self.mtp_begin(slot);
        self.p2_seg[slot] = None;
        self.disarm(slot);
        self.dflash_prompt_end(slot, None, None);
        Ok(())
    }

    /// Give back a slot's pages and forget its pending commit (and its paused prompt forward, if one is).
    pub fn end_slot(&mut self, slot: usize) {
        self.drop_paused_of(slot);
        self.pending.remove(&slot);
        self.kv_tab.release(slot);
        self.p2_seg[slot] = None;
        self.disarm(slot);
        self.dflash_prompt_end(slot, None, None);
    }

    /// Under P2, the first position of `slot`'s open prefill segment (`None`: the next prompt chunk starts one). The backend cuts a P2
    /// prompt's slices and snapshots at a whole number of 16-row KDA chunks from it (`docs/GLM_PREFILL_P2.md` 8.1).
    pub fn p2_anchor(&self, slot: usize) -> Option<usize> {
        self.p2_seg.get(slot).copied().flatten().map(|(a, _)| a)
    }

    /// Set `slot`'s open P2 prefill segment to start at `len` (a prefix restore: the restored state is the state after `len` prompt rows).
    pub(in crate::families::glm) fn p2_open_segment(&mut self, slot: usize, len: usize) {
        self.p2_seg[slot] = Some((len, len));
    }

    /// The P2 segment rule for a prompt chunk of `rows` rows at `at` in `slot` with snapshot cuts at the chunk rows `cuts`: `Ok(anchor)`, or
    /// the refusal by name of a chunk that continues a segment off its 16-row grid, or of a cut off it.
    fn p2_segment_of(&self, slot: usize, at: usize, cuts: &[usize]) -> Result<usize, ModelError> {
        let c = gb_cuda::kda::P2_CHUNK_ROWS;
        let anchor = match self.p2_seg[slot] {
            Some((a, next)) if next == at => {
                if (at - a) % c != 0 {
                    return Err(ModelError::Request(format!(
                        "{}=p2: a prompt chunk that continues a prefill segment must start a whole number of {c}-row KDA chunks after the \
                         segment's first row (slot {slot}: anchor {a}, start {at}); cut P2 prefill calls at multiples of {c} rows \
                         (docs/GLM_PREFILL_P2.md 8.1)",
                        super::contract::CONTRACT_VAR
                    )));
                }
                a
            }
            _ => at,
        };
        for &row in cuts {
            if (at + row - anchor) % c != 0 {
                return Err(ModelError::Request(format!(
                    "{}=p2: a snapshot inside a prompt chunk must cut a whole number of {c}-row KDA chunks after the segment's first row \
                     (slot {slot}: anchor {anchor}, cut {})",
                    super::contract::CONTRACT_VAR,
                    at + row
                )));
            }
        }
        Ok(anchor)
    }

    /// Record that `keep` rows of `slot`'s last window were accepted. Applied (for all slots at once) before the next forward.
    pub fn commit(&mut self, slot: usize, keep: usize) {
        self.pending.insert(slot, keep);
    }

    /// Apply the pending commits: per KDA layer, flip or replay the state of every sequence of the last window by its keep count (a
    /// sequence with no pending commit keeps 0: untouched). A failure leaves recurrent state that does not match the tokens committed, so
    /// it is `StateLost` (fatal). Every reader of the KDA state, the window record or the row arrays calls this first: on return the main
    /// stream is ordered after every commit (those [`GlmModel::flush_lazy`] left beside it too).
    pub fn flush(&mut self) -> Result<(), ModelError> {
        self.flush_as(false)
    }

    /// [`GlmModel::flush`] before a draft round (`DraftModel::flush`, on every rank of a group): the drafters' absorbs on the main stream,
    /// the KDA commits on the side's commit stream (`graphs::commit_side_on`), so they run beside the round, which reads none of their
    /// buffers; the next flush (a forward's first) joins them before anything else. The same kernels on the same inputs in the same order.
    pub fn flush_lazy(&mut self) -> Result<(), ModelError> {
        self.flush_as(true)
    }

    fn flush_as(&mut self, beside: bool) -> Result<(), ModelError> {
        self.join_commits().and_then(|()| self.flush_pending(beside)).map_err(|e| match e {
            ModelError::StateLost(m) => ModelError::StateLost(m),
            other => ModelError::StateLost(other.to_string()),
        })
    }

    /// The main stream waits for the KDA commits the last [`GlmModel::flush_lazy`] left on the side's commit stream (nothing when it left
    /// none).
    fn join_commits(&mut self) -> Result<(), ModelError> {
        if std::mem::take(&mut self.commits_on_side) {
            let sd = self.side.as_ref().expect("commits on the side need the side");
            gb_cuda::graph::wait(&self.stream, &sd.cm_done)?;
        }
        Ok(())
    }

    fn flush_pending(&mut self, beside: bool) -> Result<(), ModelError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let Some(w) = self.last_window.take() else {
            self.pending.clear();
            return Ok(());
        };
        let keep: Vec<i32> =
            w.win.seq_slot.iter().map(|&sl| self.pending.get(&(sl as usize)).copied().unwrap_or(0) as i32).collect();
        self.pending.clear();
        if keep.iter().all(|&k| k == 0) {
            return Ok(());
        }
        // the kept rows' final-normed hidden rows join their slots' stores (the NextN drafter's input) before the next forward overwrites them
        self.mtp_commit_taps(&w.win, &keep)?;
        // and the DFlash2 drafter's tap rows of the accepted rows join their slots' draft contexts
        self.dflash_commit_taps(&w.win, &keep)?;
        // every sequence kept whole: the window already wrote each one's final state, so its replay (which `skip_full` makes a no-op for a sequence
        // kept whole) is skipped altogether and the commit is the conv shift and the flip of every row (`keep == None`: two launches a layer fewer)
        let all_whole = keep.iter().enumerate().all(|(i, &k)| k >= 0 && k as usize >= w.win.seq_rows(i).len());
        if !all_whole {
            self.stage.upload(&mut self.s.keep, &keep, &self.stream)?;
        }
        if self.cfg.kda.rank.is_some() && self.tp.as_ref().is_none_or(|t| t.split_bits() & tp::SPLIT_KDA == 0) {
            return Err(kda_rank_without_split(&self.cfg));
        }
        let GlmModel { s, pools, recs, stream, kda_dims, kda_params, dev, kda_ord, tp, side, commits_on_side, .. } = self;
        let batch = kda::Batch { seq_start: &s.seq_start, slot: &s.seq_slot, n_seq: w.win.n_seq(), rows: w.win.rows };
        // a lazy flush's commits beside what the main stream runs next (after everything it was given: the absorbs above, the `keep`
        // upload), joined by the next flush (`join_commits`)
        let sd = side.as_ref().filter(|_| beside && graphs::commit_side_on());
        let st = match sd {
            Some(sd) => {
                gb_cuda::graph::fork(stream, &sd.cm, &sd.cm_go)?;
                &sd.cm
            }
            None => &*stream,
        };
        for (l, layer) in dev.layers.iter().enumerate() {
            let MixerDevice::Kda(k) = &layer.mixer else { continue };
            let ord = kda_ord[l];
            // a rank of a KDA head split commits its heads with their weights (`model/tp.rs`)
            let view = match tp.as_ref().and_then(|t| t.kda_rank(ord)) {
                Some(kr) => kr.w.view(),
                None => k.view(),
            };
            kda::commit(
                st,
                *kda_dims,
                &view,
                kda_params,
                &mut pools[ord],
                &batch,
                &recs[ord],
                &mut s.kda_ws,
                (!all_whole).then_some(&s.keep),
                true,
            )?;
        }
        if let Some(sd) = sd {
            sd.cm_done.record(&sd.cm)?;
            *commits_on_side = true;
        }
        Ok(())
    }

    // ------------------------------------------------------------------------------------------------------------- the forward

    fn upload_rows(&mut self, b: &RowBatch) -> Result<(), ModelError> {
        let (stage, st) = (&mut self.stage, &self.stream);
        stage.upload(&mut self.s.ids, &b.ids, st)?;
        stage.upload(&mut self.s.seq_start, &b.seq_start, st)?;
        stage.upload(&mut self.s.seq_slot, &b.seq_slot, st)?;
        stage.upload(&mut self.s.rowidx.slot, &b.row_slot, st)?;
        stage.upload(&mut self.s.rowidx.pos, &b.positions, st)?;
        Ok(())
    }

    fn ensure_pages(&mut self, b: &RowBatch, upload: bool) -> Result<(), ModelError> {
        for i in 0..b.n_seq() {
            let rows = b.seq_rows(i);
            let last = b.positions[rows.end - 1] as usize;
            if last >= self.max_context {
                return Err(ModelError::Request(format!("position {last} is past the model's context of {}", self.max_context)));
            }
            self.kv_tab.ensure(b.seq_slot[i] as usize, last + 1).map_err(|e| ModelError::Request(format!("{e:?}")))?;
        }
        if upload {
            self.upload_table()?;
        }
        Ok(())
    }

    /// Bring the device block table up to the host's page lists, in the stream's order: per slot, only the span of entries that differ
    /// from the last upload is copied (a decode step whose pages did not change copies nothing).
    pub(in crate::families::glm) fn upload_table(&mut self) -> Result<(), ModelError> {
        let mb = self.table.max_blocks;
        for s in 0..self.slots {
            let blocks = self.kv_tab.blocks_of(s);
            if blocks.len() > mb {
                return Err(ModelError::Request(format!("slot {s}: {} pages, the table holds {mb}", blocks.len())));
            }
            let span = blocks.len().max(self.table_len[s]);
            let row = &mut self.table_host[s * mb..s * mb + span];
            let want = |i: usize| blocks.get(i).map_or(0, |&b| b as i32);
            let Some(lo) = (0..span).find(|&i| row[i] != want(i)) else {
                self.table_len[s] = blocks.len();
                continue;
            };
            let hi = (lo..span).rfind(|&i| row[i] != want(i)).unwrap_or(lo) + 1;
            for (i, r) in row.iter_mut().enumerate().take(hi).skip(lo) {
                *r = want(i);
            }
            self.stage.upload_at(&mut self.table.buf, (s * mb + lo) * 4, &row[lo..hi], &self.stream)?;
            self.table_len[s] = blocks.len();
        }
        Ok(())
    }

    /// Run the layers over `b`'s rows. A [`Mode::Window`] takes 1..=`WINDOW_ROWS` rows of any number of sequences and leaves their KDA
    /// records for `commit`; a [`Mode::Prefill`] takes the consecutive prompt rows of ONE sequence (or, under P2, of several: `part.rs`),
    /// at most the scratch's rows, and commits them layer by layer. With a head, the logits are left in the scratch (see
    /// [`GlmModel::candidates_of`]).
    pub fn forward(&mut self, b: &RowBatch, mode: Mode, head: Head) -> Result<(), ModelError> {
        let f = self.forward_begin(b, mode, head)?;
        if self.graph_eligible(mode, head) {
            self.run_window(&f.plan, b, head, f.max_pos)?;
        } else {
            self.run_plan(&f.plan, b, mode, head, f.max_pos)?;
        }
        self.forward_end(b, mode, &f.segs);
        Ok(())
    }

    /// Everything a forward does before its layers run: the checks (refusals before any work), the P2 segments of its prompts, the
    /// previous window's commits, the row arrays, the pages and the schedule.
    fn forward_begin(&mut self, b: &RowBatch, mode: Mode, head: Head) -> Result<Begun, ModelError> {
        let rows = b.rows;
        match mode {
            Mode::Window if rows == 0 || rows > self.window_rows => {
                return Err(ModelError::Request(format!("a window takes 1..={} rows, got {rows}", self.window_rows)))
            }
            Mode::Prefill if b.n_seq() == 0 || rows == 0 || rows > self.s.rows => {
                return Err(ModelError::Request(format!("a prefill chunk takes 1..={} rows", self.s.rows)))
            }
            Mode::Prefill if b.n_seq() > 1 => self.check_prompts(b)?,
            _ => {}
        }
        // a prompt forward paused at a layer boundary keeps its rows of the streams in the stash: what runs meanwhile must fit it
        self.check_room(b)?;
        // an armed cut inside a sequence is taken by the P2 chunked recurrence alone, with fewer than 16 rows of the sequence after it
        // (one piece of the recurrence: `snapshot::save_armed`); `arm_cut` refuses every other model, and any other forward that would
        // pass it is refused here, by name, before any work
        let armed_tail_ok = (0..b.n_seq()).all(|i| {
            self.armed_row(b, i).is_none_or(|row| b.seq_rows(i).len() - row < gb_cuda::kda::P2_CHUNK_ROWS)
        });
        let armed_inside = (0..b.n_seq()).any(|i| self.armed_row(b, i).is_some());
        if armed_inside
            && !(mode == Mode::Prefill
                && self.prefill == PrefillContract::P2
                && self.p2_kda_chunked
                && self.snap_cuts.is_empty()
                && armed_tail_ok)
        {
            return Err(ModelError::Request(format!(
                "a snapshot armed inside a forward is taken by a P2 prompt forward on the chunked KDA recurrence alone, fewer than {} rows \
                 before its sequence's end",
                gb_cuda::kda::P2_CHUNK_ROWS
            )));
        }
        // under P2 each prompt joins (or opens) its slot's prefill segment, on the segment's 16-row grid (refused otherwise, before any work)
        let segs = if mode == Mode::Prefill && self.prefill == PrefillContract::P2 {
            let cuts: Vec<usize> = self.snap_cuts.iter().map(|c| c.row).collect();
            (0..b.n_seq())
                .map(|i| {
                    let slot = b.seq_slot[i] as usize;
                    let mut cuts = cuts.clone();
                    cuts.extend(self.armed_row(b, i));
                    Ok((slot, self.p2_segment_of(slot, b.positions[b.seq_rows(i).start] as usize, &cuts)?))
                })
                .collect::<Result<Vec<_>, ModelError>>()?
        } else {
            Vec::new()
        };
        // the previous window's commits first: they read its index arrays, which this forward overwrites
        self.flush()?;
        let max_pos = b.positions.iter().copied().max().unwrap_or(0) as usize;
        // (the graph mutants that leave an input stale at a replay: tests only)
        let replay = self.graph_eligible(mode, head) && self.graph_would_replay(b, head, max_pos);
        let stale = if replay { self.graphs_mutant() } else { None };
        if stale != Some(GraphMutant::StaleRows) {
            self.upload_rows(b)?;
        }
        self.ensure_pages(b, stale != Some(GraphMutant::StaleTable))?;
        // a replay launches the captured graph and never reads the schedule
        let plan = match self.mutant {
            _ if replay => Vec::new(),
            Some(m) => ops::plan_forward_mutated(&self.cfg, self.layers, head != Head::None, m),
            // the last layer's tail on the head rows alone (a prompt forward whose last layer is KDA, nothing else reading those rows)
            None if self.head_rows_tail(b, mode, head) => ops::plan_forward_head_rows(&self.cfg, self.layers, head != Head::None)
                .unwrap_or_else(|| ops::plan_forward(&self.cfg, self.layers, head != Head::None)),
            // the drafter taps the head's mean and norm for every row, with or without logits
            None if head == Head::None && self.mtp.is_some() => ops::plan_forward_tap(&self.cfg, self.layers),
            None => ops::plan_forward(&self.cfg, self.layers, head != Head::None),
        };
        Ok(Begun { plan, max_pos, segs })
    }

    /// After a forward's last layer: a window's layout kept for its commit (and the end of its slots' prefill segments), each prompt's
    /// P2 segment extended over its rows.
    fn forward_end(&mut self, b: &RowBatch, mode: Mode, segs: &[(usize, usize)]) {
        if mode == Mode::Window {
            self.last_window = Some(PendingWindow { win: b.clone() });
            // a window ends every open prefill segment of its slots: the next prompt chunk of one starts a new segment
            for &sl in &b.seq_slot {
                self.p2_seg[sl as usize] = None;
            }
        }
        for (i, &(slot, anchor)) in segs.iter().enumerate() {
            let r = b.seq_rows(i);
            self.p2_seg[slot] = Some((anchor, b.positions[r.start] as usize + r.len()));
        }
    }

    /// Run `plan` (a forward's schedule, or a part of a prompt forward's) over `b`. A schedule with the last layer's tail on the head rows
    /// ([`ops::plan_forward_head_rows`]) runs to its [`Op::HeadRows`] on every row, gathers the head rows ([`GlmModel::gather_head_rows`])
    /// and runs the rest on them alone.
    fn run_plan(&mut self, plan: &[Op], b: &RowBatch, mode: Mode, head: Head, max_pos: usize) -> Result<(), ModelError> {
        self.head_rows_runs += u64::from(plan.iter().any(|o| matches!(o, Op::MixerCore { .. })));
        let Some(k) = plan.iter().position(|o| *o == Op::HeadRows) else {
            return self.run(plan, b, mode, head, max_pos);
        };
        self.run(&plan[..k], b, mode, head, max_pos)?;
        let rows = self.gather_head_rows(b)?;
        self.run(&plan[k + 1..], &rows, mode, head, max_pos)
    }

    /// [`Op::HeadRows`], after the last KDA layer's recurrence ([`Op::MixerCore`]): each sequence's last row moved to row `i` (sequence
    /// order; row `i`'s source is at or past it, so no move reads a row an earlier one wrote) in everything the tail reads: the streams,
    /// the mixing of the layer's attention site (`post`, `comb`) and the core output (into `kda_out`, from a split rank's own half-width
    /// rows on P1's large path, `GlmTp::kda_split_layer`). Under P2 a split rank's core rows are gathered with the peer's columns
    /// (`GlmTp::kda_head_rows_p2`: the full row the P2 o_proj reads), and under the sequence-parallel residual of the forward
    /// ([`GlmModel::sp_split`]) the rank that owns a head row moves its streams and mixing, the other takes them from it
    /// (`GlmTp::sp_head_rows`). Returns the batch of those rows alone, each its own sequence.
    fn gather_head_rows(&mut self, b: &RowBatch) -> Result<RowBatch, ModelError> {
        let core = self.kda_dims.v_dim() * 2;
        let p2 = self.prefill.p2_prefill(true);
        let sp = self.sp_split(b.rows, Mode::Prefill);
        let GlmModel { s, tp, stream: st, .. } = self;
        let split = tp.as_ref().is_some_and(|t| t.split_bits() & tp::SPLIT_KDA != 0);
        // a split rank's core rows are its own half-width ones (P1's large path), gathered by the P2 o_proj's way (P2)
        let (split_big, split_p2) = (split && !p2 && b.rows > alloc::MAX_ROWS, split && p2);
        let (xr, pr, cr) = (mhc::ROW * 4, mhc::STREAMS * 4, mhc::STREAMS * mhc::STREAMS * 4);
        let n = b.n_seq();
        let mut reqs = Vec::with_capacity(n);
        let (mut moves, mut mine) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let last = b.seq_rows(i).end - 1;
            let owned = sp.is_none_or(|sp| (sp.lo..sp.hi).contains(&last));
            if last != i {
                if owned {
                    s.streams.copy_within(i * xr, last * xr, xr, st)?;
                    s.mhc.post.copy_within(i * pr, last * pr, pr, st)?;
                    s.mhc.comb.copy_within(i * cr, last * cr, cr, st)?;
                }
                if !split_big && !split_p2 {
                    s.kda_out.copy_within(i * core, last * core, core, st)?;
                }
            }
            if split_big {
                tp.as_ref().expect("a split rank").kda_core_row(&mut s.kda_out, i * core, last * core, core, st)?;
            }
            moves.push((i, last));
            mine.push(owned);
            reqs.push(gb_engine::RowRequest {
                slot: b.seq_slot[i] as usize,
                tokens: vec![b.ids[last] as u32],
                first_position: b.positions[last] as u64,
            });
        }
        if let Some(t) = tp.as_mut() {
            if sp.is_some() {
                t.sp_head_rows(&mut s.streams, xr, &mine, st)?;
                t.sp_head_rows(&mut s.mhc.post, pr, &mine, st)?;
                t.sp_head_rows(&mut s.mhc.comb, cr, &mine, st)?;
            }
            if split_p2 {
                t.kda_head_rows_p2(&mut s.kda_out, &moves, core / 2, st)?;
            }
        }
        Ok(RowBatch::from_requests(&reqs))
    }

    #[allow(clippy::too_many_lines, unused_assignments)]
    fn run(&mut self, plan: &[Op], b: &RowBatch, mode: Mode, head: Head, max_pos: usize) -> Result<(), ModelError> {
        let rows = b.rows;
        let sp = self.sp_split(rows, mode);
        // each sequence's armed cut inside it (`arm_cut`; only a P2 prompt forward on the chunked recurrence has one: `forward_begin`)
        let kda_cuts: Vec<Option<usize>> = (0..b.n_seq()).map(|i| self.armed_row(b, i)).collect();
        let GlmModel {
            cfg,
            dev,
            stream,
            pools,
            latent,
            planes,
            recs,
            table,
            s,
            sig,
            kda_dims,
            kda_params,
            kda_ord,
            mla_ord,
            capture,
            forced,
            max_context,
            oracle_rounding,
            profile,
            splice,
            mtp,
            dflash,
            tap_relay,
            run_shape,
            side,
            graphs,
            dump_logits,
            dump_head,
            snaps,
            snap_cuts,
            armed,
            tp,
            own_s,
            prefill,
            p2_side_on,
            p2_kda_chunked,
            sp_chunks,
            ..
        } = self;
        let gmut = graphs.mutant;
        let st = &*stream;
        // the shared expert runs on a side stream, concurrent with the router and the routed experts, in the windows that go through the graph path
        // (graphs or their eager fallback): the arithmetic and the order of every kernel's own inputs are unchanged, only what overlaps in time
        let side_streams = side.as_ref();
        let side = if run_shape.is_some() { side.as_ref().filter(|s| s.parts != 0) } else { None };
        // one event before the first op and after each, plus up to 12 more inside each op (the sub-steps `mark!` times)
        let slabs = rows.div_ceil(alloc::MAX_ROWS);
        let timer = if profile.is_some() {
            Some(moe::EventTimer::new(13 * plan.len() + 1 + 8 * slabs * dev.layers.len())?)
        } else {
            None
        };
        let mut marks: Vec<(&'static str, usize, usize)> = Vec::new();
        let mut ev_next = plan.len() + 1;
        let mut last_ev = 0usize;
        // time the step that just ran: (name, the event before it, the event now)
        macro_rules! mark {
            ($name:expr) => {
                if let Some(t) = &timer {
                    t.record(ev_next, st)?;
                    marks.push(($name, last_ev, ev_next));
                    last_ev = ev_next;
                    ev_next += 1;
                }
            };
        }
        let mut params = *kda_params;
        if *oracle_rounding {
            params.round = kda::Round::TowardZero;
        }
        let layer_of = |l: usize| -> &LayerDevice { &dev.layers[l] };
        let site_of = |sit: Site| {
            let l = layer_of(sit.layer);
            match sit.kind {
                SiteKind::Attn => &l.attn_site,
                SiteKind::Ffn => &l.ffn_site,
            }
        };
        let cap = capture.is_some();
        // a prefill chunk of more than the decode kernels' 128 rows: the large-row kernels (the same bits) and slabbed attention
        // (a chunk that takes snapshots inside itself runs the large-row kernels whatever its rows: they give every row the decode kernels' bits)
        let big = mode == Mode::Prefill && (rows > alloc::MAX_ROWS || (!snap_cuts.is_empty() && s.pf.is_some()));
        // under the P2 contract every dense EXL3 linear of a prompt chunk takes the tensor-core GEMM, whatever the chunk's rows
        let p2 = prefill.p2_prefill(mode == Mode::Prefill);
        let path = layers::LinPath::of(p2, big);
        // under P2 the routed experts of every prompt chunk take the P2 expert GEMMs (`glm_p2_moe`), whose picks and sorted pairs live in the
        // prefill scratch at every width: the router writes there too (its kernels are P1's either way)
        let moe_pf = big || p2;
        if p2 && side.is_some() {
            return Err(ModelError::Config("a P2 prefill chunk runs on one stream (the layer branches are a window's)".into()));
        }
        // The sequence-parallel residual (`contract::p2_sp_on`, on by default; a fact of the Hello through the P2 exchange id): a large P2
        // prompt chunk on a rank of two with the KDA head split keeps only this rank's rows (`tp::RowSplit`) of the mHC streams. The
        // embedding, every site's finalize and apply, the router and the head's mean run on them; the fp16 hidden rows are swapped with the
        // peer's where a layer reads every row (a mixer, the routed experts, the head), never before a dense MLP; the mixers', the dense
        // MLPs' and the experts' outputs are kept on this rank's rows alone (no output swap). Every one of these ops is row-independent,
        // so each row is the same bits on whichever rank computes it. The drafter's taps of the peer's rows cross the link
        // (`dflash::TapRelay`). Never with a capture, forced routing, an image splice or a snapshot inside the chunk ([`GlmModel::sp_split`]).
        // the rows of the mHC streams this rank keeps (every row without `sp`), and the mHC buffers at its first one
        let (r0, mrows) = sp.map_or((0, rows), |sp| (sp.lo, sp.rows()));
        // (counted once a forward: a forward run in parts starts with the embedding in its first part)
        *sp_chunks += u64::from(sp.is_some() && plan.first() == Some(&Op::Embed));
        let x = unsafe { (s.streams.ptr() as *mut f32).add(r0 * mhc::ROW) };
        let (post, comb, part) = unsafe {
            (
                (s.mhc.post.ptr() as *mut f32).add(r0 * mhc::STREAMS),
                (s.mhc.comb.ptr() as *mut f32).add(r0 * mhc::STREAMS * mhc::STREAMS),
                (s.mhc.part.ptr() as *mut f32).add(r0 * mhc::PART_FLOATS_PER_ROW),
            )
        };
        let hidden_at = unsafe { (s.hidden.ptr() as *mut u8).add(r0 * cfg.hidden * 2) as *mut c_void };
        let mixer_out_at = unsafe { (s.mixer_out.ptr() as *const u8).add(r0 * cfg.hidden * 4) as *const c_void };
        // under P2 a MoE layer's shared expert runs on the first side stream beside the router and the routed experts, joined right before the
        // P2 combine (its output's one reader): only what overlaps in time changes, never a bit (its own scratch, `hidden` read-only, its
        // output written by it alone). Opt-in (`REXL3_GLM_P2_SHARED_SIDE=1`, `set_p2_shared_side`): not faster on node 4 (MEASURED).
        let p2_side =
            if p2 && tp.is_none() && *p2_side_on { side_streams.filter(|s| s.parts & graphs::PART_SHARED != 0) } else { None };
        let mut shared_forked = false;
        // under P2 on a tensor-parallel rank the shared expert is computed between the rank's part and the parts' sum, beside the swap
        // (`model/tp/p2.rs`): `MoeShared` only records that it is owed
        let mut shared_deferred = false;
        // under `sp`: the owed shared expert's gate linear already ran beside the hidden rows' gather (`MoeRoute`)
        let mut shared_gated = false;
        // lane E, E2.6: a rank of the shared-expert split (`sh`) runs its half of every trunk MoE layer's shared expert in this window, the
        // down partials riding in the routed experts' exchange (`model/tp.rs` `ShSplit`); prefill chunks keep it replicated
        let sh_split = mode == Mode::Window && tp.as_ref().is_some_and(|t| t.shared_split_on(true, rows));
        let sh_topk = cfg.ffn.experts_per_token;
        // a rank of the dense-MLP split (`dense`) runs its half of every dense MLP in this window, the down partials swapped
        // (`model/tp.rs` `DnSplit`); prefill chunks keep them replicated
        let dn_split = mode == Mode::Window && tp.as_ref().is_some_and(|t| t.dense_split_on(true, rows));
        let mut last_sublayer_was_mixer = false;
        let mut cur_layer = 0usize;
        // MLA needs the batch's positions bound; the DSA grid is sized by the largest number of pools any row sees
        let (n_pools_max, any_sparse, mla_ctx) = match run_shape {
            Some(sh) => (sh.n_pools_max, sh.any_sparse, sh.mla_ctx),
            None => ((max_pos + 1) / dsa::P, dsa::is_sparse(max_pos), max_pos + 1),
        };
        // the KDA layers' prefill descriptors, one per sequence of the forward, uploaded by the first one and kept for the forward
        let mut kda_descs: Vec<DevBuf> = Vec::new();
        // a site whose finalize already ran with the apply-partials before it (`mhc::apply_partials_finalize`, an op pair of a large chunk)
        let mut finalized = None;
        // under `sp`: a gather of the peer's hidden rows is in flight on the exchange stream (`GlmTp::sp_gather_begin`); the first op that
        // reads the peer's rows (or exchanges anything) joins it first
        let mut gather_pending = false;
        macro_rules! join_gather {
            () => {
                if gather_pending {
                    tp.as_mut().expect("sp is a rank's").sp_gather_join(st)?;
                    gather_pending = false;
                }
            };
        }
        if let Some(t) = &timer {
            t.record(0, st)?;
        }
        for (i, op) in plan.iter().enumerate() {
            last_ev = i;
            // a gather in flight is joined before any op that does not read only this rank's rows
            if !matches!(op, Op::Mixer { .. } | Op::MixerCore { .. } | Op::MoeShared { .. } | Op::MoeRoute { .. }) {
                join_gather!();
            }
            match *op {
                Op::Embed => {
                    // (under `sp` this rank's rows alone)
                    let ids = unsafe { (s.ids.ptr() as *const i32).add(r0) as *const c_void };
                    unsafe { mhc::embed_expand(dev.embed.ptr(), ids, x, mrows, cfg.vocab, st)? };
                    // an image's rows replace the embeddings of its placeholder tokens, in all four streams (the text attention is NoPE: nothing else
                    // about an image position differs from a text one). With no image in the chunk this launches nothing.
                    for op in splice.iter() {
                        unsafe {
                            gb_cuda::glm_vision::splice(
                                x as *mut c_void,
                                &op.feats.buf,
                                op.dst_row,
                                op.src_row,
                                op.n,
                                cfg.hidden,
                                mhc::STREAMS,
                                st,
                            )?
                        };
                    }
                    if cap {
                        st.sync()?;
                        let v: Vec<f32> = s.streams.to_vec()?;
                        let c = capture.as_mut().unwrap();
                        let e: Vec<f32> = (0..rows).flat_map(|r| v[r * mhc::ROW..r * mhc::ROW + cfg.hidden].to_vec()).collect();
                        push_tail(&mut c.embed, &e, cfg.hidden, c.tail);
                    }
                }
                Op::FirstPartials(site) => unsafe {
                    mhc::partials(x, site_of(site), part, mrows, Tune::AUTO, st)?;
                },
                Op::Finalize(site) => {
                    if finalized.take() != Some(site) {
                        let l = layer_of(site.layer);
                        let norm = match site.kind {
                            SiteKind::Attn => &l.input_norm,
                            SiteKind::Ffn => &l.post_norm,
                        };
                        unsafe {
                            mhc::finalize(
                                x,
                                part,
                                site_of(site),
                                Some(norm.ptr() as *const f32),
                                post,
                                comb,
                                None,
                                hidden_at,
                                Ty::F16,
                                mrows,
                                &MhcCfg::GLM53,
                                RMS_EPS,
                                st,
                            )?;
                        }
                    }
                    // under `sp`, the peer's hidden rows swapped in where the next op reads every row (not a dense MLP: per row), on the
                    // exchange stream while the main stream runs what reads only this rank's rows (the router, a KDA layer's own rows)
                    if let Some(sp) = sp.filter(|_| !matches!(plan.get(i + 1), Some(Op::DenseFfn { .. }))) {
                        tp.as_mut().expect("sp is a rank's").sp_gather_begin(&s.hidden, cfg.hidden * 2, sp, st)?;
                        gather_pending = true;
                    }
                }
                o @ (Op::Mixer { .. } | Op::MixerCore { .. }) => {
                    // `core`: the last KDA layer of a forward whose tail runs on the head rows alone stops at its core output (no o_proj)
                    let (layer, kind, core) = match o {
                        Op::Mixer { layer, kind } => (layer, kind, false),
                        Op::MixerCore { layer } => (layer, MixerKind::Kda, true),
                        _ => unreachable!(),
                    };
                    // a KDA layer on a split rank joins the gather inside itself, after its own rows' qkv; a row-split MLA layer after its
                    // own rows' query chain (`layers::mla_dsa_mixer`'s `join`)
                    let kda_join = std::mem::take(&mut gather_pending);
                    let mla_join = kda_join && kind != MixerKind::Kda && big && p2 && sp.is_some();
                    if kda_join && kind != MixerKind::Kda && !mla_join {
                        tp.as_mut().expect("sp is a rank's").sp_gather_join(st)?;
                    }
                    last_sublayer_was_mixer = true;
                    cur_layer = layer;
                    let l = layer_of(layer);
                    match (&l.mixer, kind) {
                        // a rank of a head split (lane E Phase 2, `model/tp.rs`): this rank's 32 heads, the o_proj by K-split partials (windows)
                        // or over the gathered heads (a large chunk); no snapshot inside the chunk (`prefill_chunk_cuts` refuses a split rank: it
                        // snapshots between chunks)
                        // (a model built for a rank's heads alone, `KdaConfig::rank`, takes this arm and is refused by name without the split)
                        (MixerDevice::Kda(w), MixerKind::Kda)
                            if cfg.kda.rank.is_some() || tp.as_ref().is_some_and(|t| t.split_bits() & tp::SPLIT_KDA != 0) =>
                        {
                            let ord = kda_ord[layer];
                            let Some(t) = tp.as_mut().filter(|t| t.split_bits() & tp::SPLIT_KDA != 0) else {
                                return Err(kda_rank_without_split(cfg));
                            };
                            let io = tp::KdaIo {
                                hidden: &s.hidden,
                                qkv: &mut s.kda_qkv,
                                out: &mut s.kda_out,
                                mixer_out: &mut s.mixer_out,
                                ws: &mut s.ws,
                                kws: &mut s.kda_ws,
                            };
                            if p2 && *p2_kda_chunked {
                                // P5b: a P2 prompt forward on the rank's heads (any rows, one or several prompts): the qkv strips through
                                // the P2 GEMM, each prompt's chunked recurrence on its slot of the rank's pool, the gathered core output,
                                // the full P2 o_proj (`tp/p2_split.rs`)
                                let piece = p2_kda_piece(&s.kda_prefill_rec, io.kws);
                                if kda_descs.is_empty() {
                                    kda_descs = kda_seg_descs(b, piece, &kda_cuts)?;
                                }
                                let segs = kda_segs(b, &kda_cuts, &kda_descs);
                                t.kda_split_layer_p2(
                                    ord,
                                    (!core).then_some(&w.o),
                                    *kda_dims,
                                    &params,
                                    &mut pools[ord],
                                    io,
                                    &segs,
                                    rows,
                                    &mut s.kda_prefill_rec,
                                    piece,
                                    st,
                                    sp,
                                    kda_join,
                                    &mut |pool: &kda::Pool, slot: usize| {
                                        snapshot::save_armed(armed, snaps, ord, pool, slot, *kda_dims, st)
                                    },
                                )?;
                            } else if big {
                                // P1: each prompt's row-by-row recurrence on its own slot and rows (one or several prompts)
                                t.sp_gather_join(st)?;
                                if kda_descs.is_empty() {
                                    kda_descs = kda_p1_descs(b)?;
                                }
                                let segs = kda_segs(b, &[], &kda_descs);
                                let path = tp::KdaPath::Prefill { segs: &segs, rec: &mut s.kda_prefill_rec };
                                t.kda_split_layer(ord, (!core).then_some(&w.o), *kda_dims, &params, &mut pools[ord], io, path, rows, st)?;
                            } else {
                                t.sp_gather_join(st)?;
                                let batch = kda::Batch { seq_start: &s.seq_start, slot: &s.seq_slot, n_seq: b.n_seq(), rows };
                                let rec = if mode == Mode::Window { &mut recs[ord] } else { &mut s.kda_prefill_rec };
                                let path = tp::KdaPath::Window { batch: &batch, rec, commit: mode == Mode::Prefill, side };
                                t.kda_split_layer(ord, (!core).then_some(&w.o), *kda_dims, &params, &mut pools[ord], io, path, rows, st)?;
                            }
                            mark!("kda: a rank's heads (qkv strips, core, split o_proj)");
                        }
                        // under P2 every prompt chunk's KDA recurrence is the chunked one (16-row chunks from the call's first row, which the
                        // segment rule put on its segment's grid), whatever the chunk's rows; the snapshot cuts split it at chunk boundaries.
                        // A forward of several prompts runs each one's recurrence on its own rows and slot
                        (MixerDevice::Kda(w), MixerKind::Kda) if p2 && *p2_kda_chunked => {
                            let ord = kda_ord[layer];
                            layers::lin(&w.qkv, &s.hidden, &mut s.kda_qkv, rows, Dtype::F16, Dtype::BF16, &mut s.ws, st, path)?;
                            mark!("kda: qkv_proj");
                            let slot = b.seq_slot[0] as usize;
                            let piece = p2_kda_piece(&s.kda_prefill_rec, &s.kda_ws);
                            if snap_cuts.is_empty() {
                                if kda_descs.is_empty() {
                                    kda_descs = kda_seg_descs(b, piece, &kda_cuts)?;
                                }
                                let (h, ch) = (cfg.hidden, cfg.kda.heads * cfg.kda.head_dim);
                                for g in kda_segs(b, &kda_cuts, &kda_descs) {
                                    let hv = s.hidden.subview(g.row0 * h * 2, g.rows * h * 2);
                                    let qv = s.kda_qkv.subview(g.row0 * 3 * ch * 2, g.rows * 3 * ch * 2);
                                    let mut ov = s.kda_out.subview(g.row0 * ch * 2, g.rows * ch * 2);
                                    kda::prefill_chunked(
                                        st,
                                        *kda_dims,
                                        &w.view(),
                                        &params,
                                        &hv,
                                        Dtype::F16,
                                        &qv,
                                        Dtype::BF16,
                                        &mut pools[ord],
                                        g.slot,
                                        g.rows,
                                        &mut s.kda_prefill_rec,
                                        &mut s.kda_ws,
                                        &mut ov,
                                        g.desc,
                                        piece,
                                    )?;
                                    if g.cut {
                                        snapshot::save_armed(armed, snaps, ord, &pools[ord], g.slot, *kda_dims, st)?;
                                    }
                                }
                            } else {
                                let (h, ch) = (cfg.hidden, cfg.kda.heads * cfg.kda.head_dim);
                                let ends: Vec<usize> = snap_cuts.iter().map(|c| c.row).chain(std::iter::once(rows)).collect();
                                let mut from = 0usize;
                                for (i, &end) in ends.iter().enumerate() {
                                    let len = end - from;
                                    let desc = kda::prefill_descriptor_pieces(slot, len, piece)?;
                                    let hv = s.hidden.subview(from * h * 2, len * h * 2);
                                    let qv = s.kda_qkv.subview(from * 3 * ch * 2, len * 3 * ch * 2);
                                    let mut ov = s.kda_out.subview(from * ch * 2, len * ch * 2);
                                    kda::prefill_chunked(
                                        st,
                                        *kda_dims,
                                        &w.view(),
                                        &params,
                                        &hv,
                                        Dtype::F16,
                                        &qv,
                                        Dtype::BF16,
                                        &mut pools[ord],
                                        slot,
                                        len,
                                        &mut s.kda_prefill_rec,
                                        &mut s.kda_ws,
                                        &mut ov,
                                        &desc,
                                        piece,
                                    )?;
                                    if let Some(c) = snap_cuts.get(i) {
                                        let sp = snaps
                                            .as_mut()
                                            .ok_or_else(|| ModelError::Request("the prefix cache is not enabled".into()))?;
                                        sp.save_kda(c.snap, ord, &pools[ord], c.slot, *kda_dims, st)?;
                                    }
                                    from = end;
                                }
                            }
                            mark!("kda: P2 prefill (projections, prep, chunked state, norm, commit)");
                            if !core {
                                layers::lin(&w.o, &s.kda_out, &mut s.mixer_out, rows, Dtype::F16, Dtype::F32, &mut s.ws, st, path)?;
                                mark!("kda: o_proj");
                            }
                        }
                        (MixerDevice::Kda(w), MixerKind::Kda) if big => {
                            let ord = kda_ord[layer];
                            layers::lin(&w.qkv, &s.hidden, &mut s.kda_qkv, rows, Dtype::F16, Dtype::F32, &mut s.ws, st, path)?;
                            mark!("kda: qkv_proj");
                            // the exact row-by-row path in pieces of 128 rows, every row's bits the serial step's, the state committed piece by piece
                            if snap_cuts.is_empty() {
                                // each prompt's recurrence on its own slot and rows (one or several prompts: `part.rs`)
                                if kda_descs.is_empty() {
                                    kda_descs = kda_p1_descs(b)?;
                                }
                                let (h, ch) = (cfg.hidden, cfg.kda.heads * cfg.kda.head_dim);
                                for g in kda_segs(b, &[], &kda_descs) {
                                    let hv = s.hidden.subview(g.row0 * h * 2, g.rows * h * 2);
                                    let qv = s.kda_qkv.subview(g.row0 * 3 * ch * 4, g.rows * 3 * ch * 4);
                                    let mut ov = s.kda_out.subview(g.row0 * ch * 2, g.rows * ch * 2);
                                    kda::prefill_with(
                                        st,
                                        *kda_dims,
                                        &w.view(),
                                        &params,
                                        &hv,
                                        Dtype::F16,
                                        &qv,
                                        Dtype::F32,
                                        &mut pools[ord],
                                        g.slot,
                                        g.rows,
                                        &mut s.kda_prefill_rec,
                                        &mut s.kda_ws,
                                        &mut ov,
                                        g.desc,
                                    )?;
                                }
                            } else {
                                // snapshots inside the chunk: the same path in segments that end at each cut (the pieces start again at a cut, which
                                // changes no row's bits), the layer's state copied out between two segments
                                let (h, ch) = (cfg.hidden, cfg.kda.heads * cfg.kda.head_dim);
                                let ends: Vec<usize> = snap_cuts.iter().map(|c| c.row).chain(std::iter::once(rows)).collect();
                                let mut from = 0usize;
                                for (i, &end) in ends.iter().enumerate() {
                                    let len = end - from;
                                    let desc = kda::prefill_descriptor(b.seq_slot[0] as usize, len)?;
                                    let hv = s.hidden.subview(from * h * 2, len * h * 2);
                                    let qv = s.kda_qkv.subview(from * 3 * ch * 4, len * 3 * ch * 4);
                                    let mut ov = s.kda_out.subview(from * ch * 2, len * ch * 2);
                                    kda::prefill_with(
                                        st,
                                        *kda_dims,
                                        &w.view(),
                                        &params,
                                        &hv,
                                        Dtype::F16,
                                        &qv,
                                        Dtype::F32,
                                        &mut pools[ord],
                                        b.seq_slot[0] as usize,
                                        len,
                                        &mut s.kda_prefill_rec,
                                        &mut s.kda_ws,
                                        &mut ov,
                                        &desc,
                                    )?;
                                    if let Some(c) = snap_cuts.get(i) {
                                        let sp = snaps
                                            .as_mut()
                                            .ok_or_else(|| ModelError::Request("the prefix cache is not enabled".into()))?;
                                        sp.save_kda(c.snap, ord, &pools[ord], c.slot, *kda_dims, st)?;
                                    }
                                    from = end;
                                }
                            }
                            mark!("kda: prefill (projections, prep, state, norm, commit)");
                            if !core {
                                layers::lin(&w.o, &s.kda_out, &mut s.mixer_out, rows, Dtype::F16, Dtype::F32, &mut s.ws, st, path)?;
                                mark!("kda: o_proj");
                            }
                        }
                        (MixerDevice::Kda(w), MixerKind::Kda) => {
                            let ord = kda_ord[layer];
                            layers::lin(&w.qkv, &s.hidden, &mut s.kda_qkv, rows, Dtype::F16, Dtype::F32, &mut s.ws, st, path)?;
                            mark!("kda: qkv_proj");
                            let batch = kda::Batch { seq_start: &s.seq_start, slot: &s.seq_slot, n_seq: b.n_seq(), rows };
                            let rec = if mode == Mode::Window { &mut recs[ord] } else { &mut s.kda_prefill_rec };
                            kda::forward_window(
                                st,
                                *kda_dims,
                                &w.view(),
                                &params,
                                &s.hidden,
                                Dtype::F16,
                                &s.kda_qkv,
                                Dtype::F32,
                                &mut pools[ord],
                                &batch,
                                rec,
                                &mut s.kda_ws,
                                &mut s.kda_out,
                                true,
                            )?;
                            mark!("kda: window (projections, prep, state, norm)");
                            if !core {
                                layers::lin(&w.o, &s.kda_out, &mut s.mixer_out, rows, Dtype::F16, Dtype::F32, &mut s.ws, st, path)?;
                                mark!("kda: o_proj");
                            }
                            if mode == Mode::Prefill {
                                // every row of a prompt chunk is accepted: flip the buffer the window wrote and shift the conv history
                                kda::commit(
                                    st,
                                    *kda_dims,
                                    &w.view(),
                                    &params,
                                    &mut pools[ord],
                                    &batch,
                                    rec,
                                    &mut s.kda_ws,
                                    None,
                                    true,
                                )?;
                                mark!("kda: commit");
                            }
                        }
                        (MixerDevice::Mla(w), MixerKind::MlaDsa) if side.is_some_and(|sd| sd.parts & graphs::PART_MLA != 0) => {
                            // the three chains of an MLA layer run as branches: the query chain on the main stream; the latent chain and then the
                            // indexer's key side on `s1`; the indexer's query side and the selection on `s2`. The same launches, each with its inputs
                            // finished by an event wait: absorb needs `q`; attend needs the latent row (`b`) and, in the sparse regime, the selection
                            // (`done2`); the scoring needs the pool plane (`c`) and `wq_b` needs `q_resid` (`q`); every branch is joined before `expand`.
                            let sd = side.unwrap();
                            let ord = mla_ord[layer];
                            let ix = &w.indexer;
                            // a rank of an MLA head split (lane E Phase 2): the query, the attention and the expand on its heads, the o_proj by
                            // K-split partials (`model/tp.rs`); the latent and the indexer as on one node
                            let mh = tp.as_ref().and_then(|t| t.mla_heads());
                            let h = cfg.hidden;
                            let skip = |p: graphs::Wait| gmut == Some(graphs::GraphMutant::SkipWait(p));
                            let slow = gmut.is_some_and(|g| g.slows_branches());
                            gb_cuda::graph::fork(st, &sd.s1, &sd.fin)?;
                            gb_cuda::graph::wait(&sd.s2, &sd.fin)?;
                            // s1: the latent chain
                            w.kv_a.forward(&s.hidden, &mut s.kv_a, rows, Dtype::F16, Dtype::F16, &mut s.ws_s1, &sd.s1)?;
                            glm_glue::rmsnorm_f16(
                                &s.kv_a,
                                &w.kv_a_norm,
                                &mut s.ckv,
                                rows,
                                cfg.mla.kv_lora_rank,
                                RMS_EPS,
                                &sd.s1,
                            )?;
                            if slow {
                                // (the mutants: the write itself is late, so that an attention that does not wait for it reads the old rows)
                                gb_cuda::graph::spin(&sd.s1, 1_000_000)?;
                            }
                            mla::latent_write(&s.ckv, &s.rowidx, table, &mut latent[ord], rows, &sd.s1)?;
                            sd.b.record(&sd.s1)?;
                            // s1: the indexer's key side
                            moe::linear_f16(&s.hidden, Dtype::F16, &ix.wk, &mut s.k_raw, rows, h, cfg.indexer.head_dim, &sd.s1)?;
                            moe::linear_f16(
                                &s.hidden,
                                Dtype::F16,
                                &ix.compress_gate,
                                &mut s.gate,
                                rows,
                                h,
                                cfg.indexer.head_dim,
                                &sd.s1,
                            )?;
                            dsa::append(
                                &ix.dsa,
                                &mut planes[ord],
                                &mut s.dsa,
                                &s.k_raw,
                                &s.gate,
                                &s.rowidx,
                                rows,
                                table,
                                &sd.s1,
                            )?;
                            if slow {
                                gb_cuda::graph::spin(&sd.s1, 1_000_000)?;
                            }
                            sd.c.record(&sd.s1)?;
                            sd.done1.record(&sd.s1)?;
                            // main: the query chain
                            if skip(graphs::Wait::Q) {
                                gb_cuda::graph::spin(st, 400_000)?;
                            }
                            w.q_a.forward(&s.hidden, &mut s.q_a, rows, Dtype::F16, Dtype::F16, &mut s.ws, st)?;
                            glm_glue::rmsnorm_f16(&s.q_a, &w.q_a_norm, &mut s.q_resid, rows, cfg.mla.q_lora_rank, RMS_EPS, st)?;
                            sd.q.record(st)?;
                            // s2: the indexer's query side and the selection (sparse regime only)
                            if any_sparse {
                                moe::linear_f16(
                                    &s.hidden,
                                    Dtype::F16,
                                    &ix.weights_proj,
                                    &mut s.wproj,
                                    rows,
                                    h,
                                    cfg.indexer.heads,
                                    &sd.s2,
                                )?;
                                if !skip(graphs::Wait::Q) {
                                    gb_cuda::graph::wait(&sd.s2, &sd.q)?;
                                }
                                ix.wq_b.forward(&s.q_resid, &mut s.iq, rows, &mut s.ws_s2, &sd.s2, false)?;
                                if !skip(graphs::Wait::Pool) {
                                    gb_cuda::graph::wait(&sd.s2, &sd.c)?;
                                }
                                assert!(n_pools_max <= s.dsa.s_stride, "the indexer's score stride bounds the context");
                                dsa::select(
                                    &planes[ord],
                                    &mut s.dsa,
                                    &s.iq,
                                    &s.wproj,
                                    &s.rowidx,
                                    rows,
                                    n_pools_max,
                                    table,
                                    &sd.s2,
                                )?;
                                if slow {
                                    gb_cuda::graph::spin(&sd.s2, 1_000_000)?;
                                }
                            }
                            sd.done2.record(&sd.s2)?;
                            let qk = cfg.mla.qk_head_dim();
                            match mh {
                                Some((h0, nh)) => w.q_b.forward_strip(
                                    h0 * qk / 128,
                                    (h0 + nh) * qk / 128,
                                    &s.q_resid,
                                    &mut s.q,
                                    rows,
                                    Dtype::F16,
                                    Dtype::F16,
                                    &mut s.ws,
                                    st,
                                )?,
                                None => w.q_b.forward(&s.q_resid, &mut s.q, rows, Dtype::F16, Dtype::F16, &mut s.ws, st)?,
                            }
                            // attention: absorb, then the dense rows, then the sparse rows
                            let ctx = mla_ctx.min(*max_context);
                            assert!(rows <= s.mla.max_rows && ctx <= s.mla.max_context, "scratch too small");
                            match mh {
                                Some((h0, nh)) => mla::absorb_heads(&s.q, nh * qk, &w.kv_b, h0, nh, &mut s.mla.q_lat, rows, st)?,
                                None => mla::absorb(&s.q, cfg.mla.heads * qk, &w.kv_b, &mut s.mla.q_lat, rows, st)?,
                            }
                            if !skip(graphs::Wait::Latent) {
                                gb_cuda::graph::wait(st, &sd.b)?;
                            }
                            match mh {
                                Some((_, nh)) => mla::attend_below_heads(
                                    &s.mla.q_lat,
                                    &latent[ord],
                                    table,
                                    &s.rowidx,
                                    rows,
                                    &mut s.mla.o_lat,
                                    mla::DENSE_MAX_KEYS,
                                    ctx,
                                    &mut s.mla.ws,
                                    nh,
                                    st,
                                )?,
                                None => mla::attend_below(
                                    &s.mla.q_lat,
                                    &latent[ord],
                                    table,
                                    &s.rowidx,
                                    rows,
                                    &mut s.mla.o_lat,
                                    mla::DENSE_MAX_KEYS,
                                    ctx,
                                    &mut s.mla.ws,
                                    st,
                                )?,
                            }
                            if !skip(graphs::Wait::Select) {
                                gb_cuda::graph::wait(st, &sd.done2)?;
                            }
                            if ctx > mla::DENSE_MAX_KEYS {
                                let sel = SparseSel { indices: &s.dsa.indices, meta: &s.dsa.meta, stride: dsa::K_STRIDE };
                                match mh {
                                    Some((_, nh)) => mla::attend_sparse_heads(
                                        &s.mla.q_lat,
                                        &latent[ord],
                                        table,
                                        &s.rowidx,
                                        rows,
                                        &sel,
                                        &mut s.mla.o_lat,
                                        &mut s.mla.ws,
                                        nh,
                                        st,
                                    )?,
                                    None => mla::attend_sparse(
                                        &s.mla.q_lat,
                                        &latent[ord],
                                        table,
                                        &s.rowidx,
                                        rows,
                                        &sel,
                                        &mut s.mla.o_lat,
                                        &mut s.mla.ws,
                                        st,
                                    )?,
                                }
                            }
                            // every branch is joined before the layer's last consumers (the capture needs it, and the next layer reuses the buffers)
                            gb_cuda::graph::wait(st, &sd.done1)?;
                            gb_cuda::graph::wait(st, &sd.done2)?;
                            match mh {
                                Some((h0, nh)) => {
                                    mla::expand_heads(&s.mla.o_lat, &w.kv_b, h0, nh, &mut s.mla_out, rows, st)?;
                                    tp.as_mut().unwrap().split_o(&w.o, &s.mla_out, rows, &mut s.mixer_out, &mut s.ws, st)?;
                                }
                                None => {
                                    mla::expand(&s.mla.o_lat, &w.kv_b, &mut s.mla_out, rows, st)?;
                                    w.o.forward(&s.mla_out, &mut s.mixer_out, rows, Dtype::F16, Dtype::F32, &mut s.ws, st)?;
                                }
                            }
                        }
                        // a rank of an MLA head split, eagerly (no branches): its heads at decode; a large chunk stays on the replicated path
                        (MixerDevice::Mla(w), MixerKind::MlaDsa)
                            if !big && !p2 && tp.as_ref().is_some_and(|t| t.mla_heads().is_some()) =>
                        {
                            let ord = mla_ord[layer];
                            let ctx = layers::MlaCtx {
                                st,
                                cfg,
                                table,
                                max_context: *max_context,
                                n_pools_max,
                                any_sparse,
                                mla_ctx,
                                big,
                                lin: path,
                            };
                            layers::mla_dsa_mixer_heads(
                                &ctx,
                                w,
                                &mut latent[ord],
                                &mut planes[ord],
                                s,
                                rows,
                                tp.as_mut().unwrap(),
                            )?;
                            mark!("mla: a rank's heads (q_b strips, attention, split o_proj)");
                        }
                        (MixerDevice::Mla(w), MixerKind::MlaDsa) => {
                            let ord = mla_ord[layer];
                            let ctx = layers::MlaCtx {
                                st,
                                cfg,
                                table,
                                max_context: *max_context,
                                n_pools_max,
                                any_sparse,
                                mla_ctx,
                                big,
                                lin: path,
                            };
                            // the ring tails of the snapshots taken inside this chunk: copied right after the slab that holds the cut's last row
                            let cuts_here: &[snapshot::InChunkCut] = snap_cuts.as_slice();
                            let mut ring_hook = |a: usize, n: usize, pl: &dsa::Planes| -> Result<(), ModelError> {
                                for c in cuts_here.iter().filter(|c| a < c.row && c.row <= a + n) {
                                    if let Some(sp) = snaps.as_mut() {
                                        sp.save_ring(c.snap, ord, pl, c.slot, c.at, st)?;
                                    }
                                }
                                Ok(())
                            };
                            // a large P2 chunk on a rank of two: the per-row tail on this rank's rows, then the output rows swapped
                            // (`tp::RowSplit`)
                            let split = if big && p2 { tp.as_ref().and_then(|t| t.p2_row_split(rows)) } else { None };
                            // under `sp` the peer's hidden rows are joined inside, once this rank's rows' query chain is queued
                            if mla_join && split.is_none() {
                                tp.as_mut().expect("sp is a rank's").sp_gather_join(st)?;
                            }
                            let tp_join = tp.as_mut().filter(|_| mla_join && split.is_some());
                            let mut join_fn = tp_join.map(|t| move || t.sp_gather_join(st));
                            let join: Option<&mut dyn FnMut() -> Result<(), ModelError>> =
                                join_fn.as_mut().map(|f| f as &mut dyn FnMut() -> Result<(), ModelError>);
                            layers::mla_dsa_mixer(
                                &ctx,
                                w,
                                &mut latent[ord],
                                &mut planes[ord],
                                s,
                                None,
                                rows,
                                max_pos,
                                (b.n_seq() > 1).then_some(&b.positions[..]),
                                split.map(|sp| (sp.lo, sp.hi)),
                                join,
                                if big && !cuts_here.is_empty() { Some(&mut ring_hook) } else { None },
                                &mut |name| {
                                    if let Some(t) = &timer {
                                        t.record(ev_next, st)?;
                                        marks.push((name, last_ev, ev_next));
                                        last_ev = ev_next;
                                        ev_next += 1;
                                    }
                                    Ok(())
                                },
                            )?;
                            // (under `sp` the rank's residual reads its own rows alone)
                            if let Some(sp) = split.filter(|_| sp.is_none()) {
                                tp.as_mut().expect("a row split is a rank's").swap_rows(&s.mixer_out, cfg.hidden * 4, sp, st)?;
                                mark!("mla: the row split's output swap");
                            }
                        }
                        _ => {
                            return Err(ModelError::Config(format!(
                                "layer {layer}: its weights and the schedule disagree on the mixer"
                            )))
                        }
                    }
                }
                Op::MixerOut { layer } => {
                    // the o_proj of the last KDA layer's core output on the head rows (`gather_head_rows` put them first in `kda_out`): a
                    // P1 split rank's by K-split partials, as a window's; under P2 the full P2 o_proj of the gathered rows (a split rank's
                    // too: `GlmTp::kda_head_rows_p2`), every row's bits the whole chunk's
                    last_sublayer_was_mixer = true;
                    cur_layer = layer;
                    let MixerDevice::Kda(w) = &layer_of(layer).mixer else {
                        return Err(ModelError::Config(format!("layer {layer} has no KDA mixer")));
                    };
                    match tp.as_mut().filter(|t| !p2 && t.split_bits() & tp::SPLIT_KDA != 0) {
                        Some(t) => {
                            let x = s.kda_out.subview(0, rows * kda_dims.v_dim() * 2);
                            t.split_o(&w.o, &x, rows, &mut s.mixer_out, &mut s.ws, st)?;
                        }
                        None => layers::lin(&w.o, &s.kda_out, &mut s.mixer_out, rows, Dtype::F16, Dtype::F32, &mut s.ws, st, path)?,
                    }
                    mark!("kda: o_proj of the head rows");
                }
                Op::HeadRows => return Err(ModelError::Config("the head rows' gather runs between two runs (`run_plan`)".into())),
                Op::DenseFfn { layer } => {
                    last_sublayer_was_mixer = false;
                    cur_layer = layer;
                    let FfnDevice::Dense(m) = &layer_of(layer).ffn else {
                        return Err(ModelError::Config(format!("layer {layer} has no dense MLP")));
                    };
                    if p2 {
                        let big_sc = s.pf.as_mut().and_then(|p| p.dense.as_mut());
                        // a large chunk on a rank of two: this rank's rows, then the output rows swapped (`tp::RowSplit`; not under `sp`)
                        match tp.as_mut().filter(|_| big).and_then(|t| t.p2_row_split(rows).map(|rs| (t, rs))) {
                            Some((t, rs)) => {
                                let h = cfg.hidden;
                                let x = s.hidden.subview(rs.lo * h * 2, rs.rows() * h * 2);
                                let mut y = s.mixer_out.subview(rs.lo * h * 4, rs.rows() * h * 4);
                                layers::gated(m, &x, rs.rows(), &mut y, s.dense.as_mut(), big_sc, st, path)?;
                                if sp.is_none() {
                                    t.swap_rows(&s.mixer_out, h * 4, rs, st)?;
                                }
                            }
                            None => layers::gated(m, &s.hidden, rows, &mut s.mixer_out, s.dense.as_mut(), big_sc, st, path)?,
                        }
                    } else if big {
                        let sc =
                            s.pf.as_mut()
                                .and_then(|p| p.dense.as_mut())
                                .ok_or_else(|| ModelError::Config("no dense scratch".into()))?;
                        m.forward_big(&s.hidden, Dtype::F16, rows, &mut s.mixer_out, sc, st)?;
                    } else {
                        let sc = s.dense.as_mut().ok_or_else(|| ModelError::Config("no dense scratch".into()))?;
                        match tp.as_mut().filter(|_| dn_split) {
                            Some(t) => t.dense_split(m, &s.hidden, rows, &mut s.mixer_out, sc, st)?,
                            None => m.forward(&s.hidden, Dtype::F16, rows, &mut s.mixer_out, sc, st)?,
                        }
                    }
                }
                Op::MoeShared { layer } => {
                    last_sublayer_was_mixer = false;
                    cur_layer = layer;
                    let (FfnDevice::Moe { shared, .. } | FfnDevice::MoeSubbanks { shared, .. }) = &layer_of(layer).ffn else {
                        return Err(ModelError::Config(format!("layer {layer} has no shared expert")));
                    };
                    if p2 && tp.is_some() {
                        shared_deferred = true;
                    } else if p2 {
                        let big_sc = s.pf.as_mut().map(|p| &mut p.shared);
                        match p2_side {
                            Some(sd) => {
                                // fork after everything the main stream has queued (the layer's finalize wrote `hidden`)
                                gb_cuda::graph::fork(st, &sd.s1, &sd.fin)?;
                                layers::gated(
                                    shared,
                                    &s.hidden,
                                    rows,
                                    &mut s.moe_shared_out,
                                    s.shared.as_mut(),
                                    big_sc,
                                    &sd.s1,
                                    path,
                                )?;
                                sd.done1.record(&sd.s1)?;
                                shared_forked = true;
                            }
                            None => layers::gated(
                                shared,
                                &s.hidden,
                                rows,
                                &mut s.moe_shared_out,
                                s.shared.as_mut(),
                                big_sc,
                                st,
                                path,
                            )?,
                        }
                    } else if big {
                        let pf = s.pf.as_mut().ok_or_else(|| ModelError::Config("no shared-expert scratch".into()))?;
                        // a width-share rank that finishes only its own rows of this layer (`GlmTp::subbanks_big`) reads only those rows of
                        // the shared expert's output: it runs on them alone (each row's big-path arithmetic is its own)
                        let fin = match (&layer_of(layer).ffn, tp.as_ref()) {
                            (FfnDevice::MoeSubbanks { layer: m, .. }, Some(t)) => t.p1_finish_rows(m, rows, &pf.moe),
                            _ => None,
                        };
                        match fin {
                            Some(fr) => {
                                let h = cfg.hidden;
                                let xr = s.hidden.subview(fr.lo * h * 2, fr.rows() * h * 2);
                                let mut so = s.moe_shared_out.subview(fr.lo * h * 4, fr.rows() * h * 4);
                                shared.forward_big(&xr, Dtype::F16, fr.rows(), &mut so, &mut pf.shared, st)?;
                            }
                            None => shared.forward_big(&s.hidden, Dtype::F16, rows, &mut s.moe_shared_out, &mut pf.shared, st)?,
                        }
                    } else {
                        let sc = s.shared.as_mut().ok_or_else(|| ModelError::Config("no shared-expert scratch".into()))?;
                        match side.filter(|sd| sd.parts & graphs::PART_SHARED != 0) {
                            Some(sd) => {
                                // fork: the side stream starts after everything the main stream has queued (the layer's finalize wrote `hidden`)
                                gb_cuda::graph::fork(st, &sd.s1, &sd.fin)?;
                                let slow = matches!(gmut, Some(graphs::GraphMutant::SlowSide | graphs::GraphMutant::JoinLate));
                                let before = slow && graphs::slow_side_before();
                                if before {
                                    gb_cuda::graph::spin(&sd.s1, graphs::slow_side_cycles())?;
                                }
                                if sh_split {
                                    let t = tp.as_mut().unwrap();
                                    t.shared_partials(shared, &s.hidden, rows, &s.moe.y_pairs, sh_topk, &sd.s1)?;
                                } else {
                                    shared.forward(&s.hidden, Dtype::F16, rows, &mut s.moe_shared_out, sc, &sd.s1)?;
                                }
                                if slow && !before {
                                    gb_cuda::graph::spin(&sd.s1, graphs::slow_side_cycles())?;
                                }
                                sd.done1.record(&sd.s1)?;
                            }
                            None if sh_split => {
                                let t = tp.as_mut().unwrap();
                                t.shared_partials(shared, &s.hidden, rows, &s.moe.y_pairs, sh_topk, st)?
                            }
                            None => shared.forward(&s.hidden, Dtype::F16, rows, &mut s.moe_shared_out, sc, st)?,
                        }
                    }
                }
                Op::MoeRoute { layer } => {
                    // the router of a single bank or of subbanks: the same two launches (`MoeLayer::route` / `MoeSubbankLayer::route`)
                    let ffn = &layer_of(layer).ffn;
                    let router = match ffn {
                        FfnDevice::Moe { layer: m, .. } => &m.router,
                        FfnDevice::MoeSubbanks { layer: m, .. } => &m.router,
                        _ => return Err(ModelError::Config(format!("layer {layer} has no router"))),
                    };
                    if let Some(sp) = sp {
                        // under `sp`: the router on this rank's rows (beside the hidden rows' gather), then the gather joined and the peer's
                        // picks and weights swapped in (the routed experts read every row's)
                        let pf = s.pf.as_mut().ok_or_else(|| ModelError::Config("no prefill scratch".into()))?;
                        let (h, e, k) = (cfg.hidden, router.experts, router.topk);
                        let xr = s.hidden.subview(sp.lo * h * 2, sp.rows() * h * 2);
                        let mut lg = pf.moe.logits.subview(sp.lo * e * 2, sp.rows() * e * 2);
                        let mut pk = pf.moe.pick.subview(sp.lo * k * 4, sp.rows() * k * 4);
                        let mut wt = pf.moe.wts.subview(sp.lo * k * 2, sp.rows() * k * 2);
                        router.logits(&xr, Dtype::F16, sp.rows(), &mut lg, st)?;
                        router.select(sig, &lg, sp.rows(), &mut pk, &mut wt, st)?;
                        // the owed shared expert's gate linear on this rank's rows too, while the gather crosses (its rest runs beside the
                        // part swap, `p2_rank_layer`): the same launches, one piece of the prefill scratch's rows
                        let (FfnDevice::Moe { shared: sh, .. } | FfnDevice::MoeSubbanks { shared: sh, .. }) = ffn else { unreachable!() };
                        if shared_deferred && sp.rows() <= pf.shared.max_rows {
                            let mut so = s.moe_shared_out.subview(sp.lo * h * 4, sp.rows() * h * 4);
                            let gate = gb_cuda::glm_p2::GatedPart::Gate;
                            layers::gated_part(sh, &xr, sp.rows(), &mut so, s.shared.as_mut(), Some(&mut pf.shared), st, gate)?;
                            shared_gated = true;
                        }
                        let t = tp.as_mut().expect("sp is a rank's");
                        t.sp_gather_join(st)?;
                        gather_pending = false;
                        t.swap_rows(&pf.moe.pick, k * 4, sp, st)?;
                        t.swap_rows(&pf.moe.wts, k * 2, sp, st)?;
                    } else if moe_pf {
                        join_gather!();
                        let pf = s.pf.as_mut().ok_or_else(|| ModelError::Config("no prefill scratch".into()))?;
                        router.logits(&s.hidden, Dtype::F16, rows, &mut pf.moe.logits, st)?;
                        router.select(sig, &pf.moe.logits, rows, &mut pf.moe.pick, &mut pf.moe.wts, st)?;
                    } else {
                        match ffn {
                            FfnDevice::Moe { layer: m, .. } => m.route(&s.hidden, Dtype::F16, rows, sig, &mut s.moe, st)?,
                            FfnDevice::MoeSubbanks { layer: m, .. } => {
                                m.route(&s.hidden, Dtype::F16, rows, sig, &mut s.moe, st)?
                            }
                            _ => unreachable!(),
                        }
                    }
                    if let Some(f) = forced.as_ref() {
                        if let Some(i) = f.layers.iter().position(|&x| x == layer) {
                            let (mut ids, mut ws) = (Vec::with_capacity(rows * 8), Vec::with_capacity(rows * 8));
                            for &p in &b.positions {
                                let p = p as usize;
                                if p >= f.t {
                                    return Err(ModelError::Request(format!("forced routing covers {} positions, not {p}", f.t)));
                                }
                                let at = (i * f.t + p) * 8;
                                ids.extend_from_slice(&f.ids[at..at + 8]);
                                ws.extend_from_slice(&f.w[at..at + 8]);
                            }
                            match s.pf.as_mut() {
                                Some(pf) if moe_pf => pf.moe.set_picks(&ids, &ws)?,
                                _ => s.moe.set_picks(&ids, &ws)?,
                            }
                        }
                    }
                    if cap {
                        st.sync()?;
                        let c = capture.as_mut().unwrap();
                        let (ids, w): (Vec<i32>, Vec<u16>) = match s.pf.as_ref() {
                            Some(pf) if moe_pf => (pf.moe.pick.to_vec()?, pf.moe.wts.to_vec()?),
                            _ => (s.moe.pick.to_vec()?, s.moe.wts.to_vec()?),
                        };
                        c.route_ids.entry(layer).or_default().extend_from_slice(&ids[..rows * 8]);
                        c.route_w.entry(layer).or_default().extend_from_slice(&w[..rows * 8]);
                    }
                }
                // routed experts bound as subbanks (a mixed-width pack, or a rank's share): every subbank's pairs into one pair buffer, on a rank
                // the exchange of that buffer (`model/tp.rs`), then the slot combine with the global picks and the shared expert added last.
                // Under P2 on one node: the P2 pairs stage once per subbank, then the one two-group combine (`glm_p2_moe::run_subbanks`); on a
                // rank: the same pairs stage over the rank's subbanks, its part, the swap, the parts' sum (`model/tp/p2.rs`).
                Op::MoeExperts { layer, with_shared } if matches!(layer_of(layer).ffn, FfnDevice::MoeSubbanks { .. }) => 'p2arm: {
                    let FfnDevice::MoeSubbanks { layer: m, .. } = &layer_of(layer).ffn else { unreachable!() };
                    if p2 {
                        let FfnDevice::MoeSubbanks { shared, .. } = &layer_of(layer).ffn else { unreachable!() };
                        match tp.as_mut() {
                            // a tensor-parallel rank's share (one subbank or several): the P2 rank path (`model/tp/p2.rs`)
                            Some(t) => {
                                let owed = std::mem::take(&mut shared_deferred).then_some(&**shared);
                                let ex = tp::P2Routed::Subbanks(&m.experts);
                                let gated = std::mem::take(&mut shared_gated);
                                p2_rank_layer(t, ex, owed, with_shared, s, rows, st, path, sp.is_some(), gated, &mut |_| Ok(()))?;
                            }
                            // a mixed-width layer on one node: the pairs stage once per subbank, then the one combine
                            // (`glm_p2_moe::run_subbanks`); the layer's shared expert ran first (on the side stream, when it is on, joined
                            // right before the combine: its one reader)
                            None => {
                                let join = shared_forked.then_some(p2_side).flatten();
                                shared_forked = false;
                                let sh = with_shared.then_some(&s.moe_shared_out);
                                let os = own_s.as_mut().ok_or_else(|| ModelError::Config("no subbank scratch".into()))?;
                                let pf = s.pf.as_mut().ok_or_else(|| ModelError::Config("no prefill scratch".into()))?;
                                gb_cuda::glm_p2_moe::pairs_subbanks_with(
                                    &m.experts,
                                    &s.hidden,
                                    Dtype::F16,
                                    rows,
                                    &mut pf.moe,
                                    os,
                                    &mut s.moe,
                                    gb_cuda::glm_p2_moe::DEFAULT_ORDER,
                                    st,
                                    &mut |_| Ok(()),
                                )?;
                                if let Some(sd) = join {
                                    gb_cuda::graph::wait(st, &sd.done1)?;
                                }
                                gb_cuda::glm_p2_moe::combine_dims(
                                    gb_cuda::glm_p2_moe::CombineMode::Both,
                                    &pf.moe.y_pairs,
                                    &pf.moe.pick,
                                    &pf.moe.wts,
                                    sh,
                                    None,
                                    &mut s.mixer_out,
                                    rows,
                                    &m.dims,
                                    st,
                                )?;
                            }
                        }
                        break 'p2arm;
                    }
                    let sh = with_shared.then_some(&s.moe_shared_out);
                    if let Some(t) = tp.as_mut() {
                        if big {
                            let pf = s.pf.as_mut().ok_or_else(|| ModelError::Config("no prefill scratch".into()))?;
                            t.subbanks_big(m, &s.hidden, rows, &mut pf.moe, &mut s.moe, sh, &mut s.mixer_out, st)?;
                        } else {
                            // a rank in a window with branches (graphs on): the shared expert ran on a side stream; the main stream waits
                            // for it after the exchange, right before the combine (its one reader), as one node does before its combine
                            let sd = side.filter(|sd| with_shared && sd.parts & graphs::PART_SHARED != 0);
                            let late = gmut == Some(graphs::GraphMutant::JoinLate);
                            let mut join = || -> Result<(), ModelError> {
                                match sd {
                                    Some(sd) if !late => Ok(gb_cuda::graph::wait(st, &sd.done1)?),
                                    _ => Ok(()),
                                }
                            };
                            if sh_split {
                                // the shared-expert split: its partials are in the pair buffer's tail, joined BEFORE the one exchange
                                // (whether or not the combine adds the shared output: the exchange reads the partials either way)
                                let sd_sh = side.filter(|sd| sd.parts & graphs::PART_SHARED != 0);
                                let mut join_sh = || -> Result<(), ModelError> {
                                    match sd_sh {
                                        Some(sd) if !late => Ok(gb_cuda::graph::wait(st, &sd.done1)?),
                                        _ => Ok(()),
                                    }
                                };
                                let FfnDevice::MoeSubbanks { shared, .. } = &layer_of(layer).ffn else { unreachable!() };
                                let (so, mo) = (&mut s.moe_shared_out, &mut s.mixer_out);
                                t.subbanks_sh(m, &s.hidden, rows, &mut s.moe, shared, so, with_shared, mo, st, &mut join_sh)?;
                            } else {
                                t.subbanks(m, &s.hidden, rows, &mut s.moe, sh, &mut s.mixer_out, st, &mut join)?;
                            }
                            if let (Some(sd), true) = (sd, late) {
                                // the mutant: the wait comes after the combine
                                gb_cuda::graph::wait(st, &sd.done1)?;
                            }
                        }
                    } else {
                        let os = own_s.as_mut().ok_or_else(|| ModelError::Config("no subbank scratch".into()))?;
                        let d = m.dims;
                        if big {
                            let pf = s.pf.as_mut().ok_or_else(|| ModelError::Config("no prefill scratch".into()))?;
                            m.experts.run_prefill_pairs_windowed(&s.hidden, Dtype::F16, rows, &mut pf.moe, &mut s.moe, os, st)?;
                            moe::combine_pairs(&pf.moe.y_pairs, &pf.moe.pick, &pf.moe.wts, sh, &mut s.mixer_out, rows, &d, st)?;
                        } else {
                            m.experts.run_pairs(&s.hidden, Dtype::F16, rows, &mut s.moe, os, st)?;
                            // the combine is the one reader of the shared expert's output: a side-stream shared expert is joined right before it
                            match side.filter(|sd| sd.parts & graphs::PART_SHARED != 0) {
                                Some(sd) if with_shared && gmut == Some(graphs::GraphMutant::JoinLate) => {
                                    moe::combine_pairs(
                                        &s.moe.y_pairs,
                                        &s.moe.pick,
                                        &s.moe.wts,
                                        sh,
                                        &mut s.mixer_out,
                                        rows,
                                        &d,
                                        st,
                                    )?;
                                    gb_cuda::graph::wait(st, &sd.done1)?;
                                }
                                Some(sd) if with_shared => {
                                    gb_cuda::graph::wait(st, &sd.done1)?;
                                    moe::combine_pairs(
                                        &s.moe.y_pairs,
                                        &s.moe.pick,
                                        &s.moe.wts,
                                        sh,
                                        &mut s.mixer_out,
                                        rows,
                                        &d,
                                        st,
                                    )?;
                                }
                                _ => moe::combine_pairs(
                                    &s.moe.y_pairs,
                                    &s.moe.pick,
                                    &s.moe.wts,
                                    sh,
                                    &mut s.mixer_out,
                                    rows,
                                    &d,
                                    st,
                                )?,
                            }
                        }
                    }
                }
                // a rank of a tensor-parallel group (`model/tp.rs`): this rank's pairs, the exchange of the pair buffer, the slot combine over the sum
                Op::MoeExperts { layer, with_shared } if tp.is_some() => 'p2arm: {
                    let FfnDevice::Moe { layer: m, .. } = &layer_of(layer).ffn else {
                        return Err(ModelError::Config(format!("layer {layer} has no experts")));
                    };
                    let t = tp.as_mut().unwrap();
                    if p2 {
                        let FfnDevice::Moe { shared, .. } = &layer_of(layer).ffn else { unreachable!() };
                        let owed = std::mem::take(&mut shared_deferred).then_some(&**shared);
                        let ex = tp::P2Routed::Bank(&m.experts);
                        let gated = std::mem::take(&mut shared_gated);
                        p2_rank_layer(t, ex, owed, with_shared, s, rows, st, path, sp.is_some(), gated, &mut |_| Ok(()))?;
                        break 'p2arm;
                    }
                    let sh = with_shared.then_some(&s.moe_shared_out);
                    if big {
                        let pf = s.pf.as_mut().ok_or_else(|| ModelError::Config("no prefill scratch".into()))?;
                        t.experts_big(m, &s.hidden, rows, &mut pf.moe, sh, &mut s.mixer_out, st)?;
                    } else {
                        // as the subbank branch above: the side-stream shared expert is joined after the exchange, before the combine
                        let sd = side.filter(|sd| with_shared && sd.parts & graphs::PART_SHARED != 0);
                        let late = gmut == Some(graphs::GraphMutant::JoinLate);
                        let mut join = || -> Result<(), ModelError> {
                            match sd {
                                Some(sd) if !late => Ok(gb_cuda::graph::wait(st, &sd.done1)?),
                                _ => Ok(()),
                            }
                        };
                        if sh_split {
                            let sd_sh = side.filter(|sd| sd.parts & graphs::PART_SHARED != 0);
                            let mut join_sh = || -> Result<(), ModelError> {
                                match sd_sh {
                                    Some(sd) if !late => Ok(gb_cuda::graph::wait(st, &sd.done1)?),
                                    _ => Ok(()),
                                }
                            };
                            let FfnDevice::Moe { shared, .. } = &layer_of(layer).ffn else { unreachable!() };
                            let (so, mo) = (&mut s.moe_shared_out, &mut s.mixer_out);
                            t.experts_sh(m, &s.hidden, rows, &mut s.moe, shared, so, with_shared, mo, st, &mut join_sh)?;
                        } else {
                            t.experts(m, &s.hidden, rows, &mut s.moe, sh, &mut s.mixer_out, st, &mut join)?;
                        }
                        if let (Some(sd), true) = (sd, late) {
                            gb_cuda::graph::wait(st, &sd.done1)?;
                        }
                    }
                }
                Op::MoeExperts { layer, with_shared } => {
                    let FfnDevice::Moe { layer: m, .. } = &layer_of(layer).ffn else {
                        return Err(ModelError::Config(format!("layer {layer} has no experts")));
                    };
                    let sh = with_shared.then_some(&s.moe_shared_out);
                    if moe_pf {
                        let pf = s.pf.as_mut().ok_or_else(|| ModelError::Config("no prefill scratch".into()))?;
                        let (mut e_next, mut e_last) = (ev_next, last_ev);
                        let tm = timer.as_ref();
                        let join = shared_forked.then_some(p2_side).flatten();
                        shared_forked = false;
                        let mut mark = |name: &'static str| -> Result<(), gb_cuda::CudaError> {
                            if let Some(t) = tm {
                                t.record(e_next, st)?;
                                marks.push((moe_stage_name(name), e_last, e_next));
                                e_last = e_next;
                                e_next += 1;
                            }
                            // P2: the combine (next) reads the side stream's shared expert output
                            if name == "down" {
                                if let Some(sd) = join {
                                    gb_cuda::graph::wait(st, &sd.done1)?;
                                }
                            }
                            Ok(())
                        };
                        if p2 {
                            // P2: the routed experts on the P2 expert GEMMs (Lane P step 3), the two-group combine with the shared expert last
                            gb_cuda::glm_p2_moe::run_with(
                                &m.experts,
                                &s.hidden,
                                Dtype::F16,
                                rows,
                                sh,
                                &mut s.mixer_out,
                                &mut pf.moe,
                                gb_cuda::glm_p2_moe::DEFAULT_ORDER,
                                st,
                                &mut mark,
                            )?;
                        } else {
                            m.experts.run_prefill_marked(
                                &s.hidden,
                                Dtype::F16,
                                rows,
                                sh,
                                &mut s.mixer_out,
                                &mut pf.moe,
                                st,
                                &mut mark,
                            )?;
                        }
                        ev_next = e_next;
                        last_ev = e_last;
                    } else {
                        match side.filter(|sd| sd.parts & graphs::PART_SHARED != 0) {
                            Some(sd) if with_shared && gmut == Some(graphs::GraphMutant::JoinLate) => {
                                // the mutant: the wait comes after the combine
                                m.experts_picked(&s.hidden, Dtype::F16, rows, sh, &mut s.mixer_out, &mut s.moe, st)?;
                                gb_cuda::graph::wait(st, &sd.done1)?;
                            }
                            // the combine is the one reader of the shared expert's output: the main stream waits for it right before
                            Some(sd) if with_shared => m.experts_picked_joined(
                                &s.hidden,
                                Dtype::F16,
                                rows,
                                sh,
                                &mut s.mixer_out,
                                &mut s.moe,
                                st,
                                &mut || gb_cuda::graph::wait(st, &sd.done1),
                            )?,
                            _ => m.experts_picked(&s.hidden, Dtype::F16, rows, sh, &mut s.mixer_out, &mut s.moe, st)?,
                        }
                    }
                }
                Op::ApplyPartials { next } => {
                    // a large chunk's apply-partials and the next site's finalize right after it run in slabs of rows, each slab's
                    // finalize re-reading its streams from the L2 (row-independent kernels, per-row scratch: the same bits as the two ops)
                    if !cap && mrows > mhc::SLAB_ROWS && plan.get(i + 1) == Some(&Op::Finalize(next)) {
                        let l = layer_of(next.layer);
                        let norm = match next.kind {
                            SiteKind::Attn => &l.input_norm,
                            SiteKind::Ffn => &l.post_norm,
                        };
                        unsafe {
                            mhc::apply_partials_finalize(
                                x,
                                mixer_out_at,
                                Ty::F32,
                                post,
                                comb,
                                site_of(next),
                                part,
                                Some(norm.ptr() as *const f32),
                                hidden_at,
                                Ty::F16,
                                mrows,
                                mhc::SLAB_ROWS,
                                &MhcCfg::GLM53,
                                RMS_EPS,
                                st,
                            )?;
                        }
                        finalized = Some(next);
                    } else {
                        unsafe {
                            mhc::apply_partials(x, mixer_out_at, Ty::F32, post, comb, site_of(next), part, mrows, Tune::AUTO, st)?;
                        }
                    }
                    if cap {
                        capture_streams(capture.as_mut().unwrap(), &s.streams, st, rows, last_sublayer_was_mixer, cur_layer)?;
                    }
                    // the drafter's taps of a completed layer (its FFN's partials applied: the next site is the next layer's attention)
                    if next.kind == SiteKind::Attn {
                        dflash::tap_layer(dflash, tap_relay, tp, &s.streams, cur_layer, rows, sp, st)?;
                    }
                }
                Op::Apply => {
                    unsafe {
                        mhc::apply(x, mixer_out_at, Ty::F32, post, comb, mrows, st)?;
                    }
                    if cap {
                        capture_streams(capture.as_mut().unwrap(), &s.streams, st, rows, last_sublayer_was_mixer, cur_layer)?;
                    }
                    dflash::tap_layer(dflash, tap_relay, tp, &s.streams, cur_layer, rows, sp, st)?;
                }
                Op::HeadMean => {
                    if cap {
                        // the mean alone (the oracle's `hc_head`), then the fused mean + final norm
                        unsafe { mhc::mean(x, None, s.mixer_out.ptr(), Ty::F32, rows, RMS_EPS, st)? };
                        st.sync()?;
                        let v: Vec<f32> = s.mixer_out.to_vec()?;
                        let c = capture.as_mut().unwrap();
                        push_tail(&mut c.hc_head, &v[..rows * cfg.hidden], cfg.hidden, c.tail);
                    }
                    unsafe {
                        mhc::mean(x, Some(dev.final_norm.ptr() as *const f32), hidden_at, Ty::F16, mrows, RMS_EPS, st)?
                    };
                    // under `sp`, the peer's normed rows swapped in (the head and the NextN tap read them)
                    if let Some(sp) = sp {
                        tp.as_mut().expect("sp is a rank's").swap_rows(&s.hidden, cfg.hidden * 2, sp, st)?;
                    }
                    if let Some(m) = mtp.as_ref() {
                        // the NextN drafter's tap: this forward's final-normed rows, before the head moves rows around in `hidden`
                        let h = cfg.hidden;
                        if m.mutant == Some(MtpMutant::TapBeforeNorm) {
                            unsafe { mhc::mean(x, None, m.tap.ptr(), Ty::F16, rows, RMS_EPS, st)? };
                        } else {
                            unsafe { glue::copy_rows(s.hidden.ptr() as *const c_void, h, m.tap.ptr(), h, rows, h, st)? };
                        }
                    }
                    if cap {
                        st.sync()?;
                        let v: Vec<u16> = s.hidden.to_vec()?;
                        let c = capture.as_mut().unwrap();
                        push_tail(&mut c.final_norm, &v[..rows * cfg.hidden], cfg.hidden, c.tail);
                    }
                }
                Op::LmHead if big && head == Head::AllRows => {
                    // the logits of every row of a big chunk, in pieces of the head's 128 rows, into the debug buffer
                    let dl = dump_logits.as_mut().ok_or_else(|| {
                        ModelError::Request("all-row logits of a chunk above 128 rows need enable_dump_logits".into())
                    })?;
                    let (h, v) = (cfg.hidden, cfg.vocab);
                    let mut a = 0;
                    while a < rows {
                        let n = (rows - a).min(alloc::MAX_ROWS);
                        let hv = s.hidden.subview(a * h * 2, n * h * 2);
                        let mut lv = dl.subview(a * v * 2, n * v * 2);
                        dev.lm_head.forward(&hv, &mut lv, n, Dtype::F16, Dtype::F16, &mut s.ws, st)?;
                        a += n;
                    }
                    *dump_head = true;
                }
                Op::LmHead => {
                    *dump_head = false;
                    let n = match head {
                        Head::AllRows => rows,
                        Head::LastRow => {
                            // the head wants rows from the start of its input: move each sequence's last normed row to the front, in
                            // sequence order (row i's source is at or past row i, so no move reads a row an earlier move wrote)
                            for i in 0..b.n_seq() {
                                let last = b.seq_rows(i).end - 1;
                                if last != i {
                                    unsafe {
                                        glue::copy_rows(
                                            s.hidden.ptr_at(last * cfg.hidden * 2) as *const c_void,
                                            cfg.hidden,
                                            s.hidden.ptr_at(i * cfg.hidden * 2),
                                            cfg.hidden,
                                            1,
                                            cfg.hidden,
                                            st,
                                        )?;
                                    }
                                }
                            }
                            b.n_seq()
                        }
                        Head::None => return Err(ModelError::Config("the schedule has a head the call did not ask for".into())),
                    };
                    match tp.as_mut().filter(|t| t.split_bits() & tp::SPLIT_LM != 0) {
                        // a rank of a vocabulary split: its half of the head, the peer's half swapped in (`model/tp.rs`)
                        Some(t) => t.split_lm_head(&dev.lm_head, &s.hidden, &mut s.logits, n, &mut s.ws, st)?,
                        None => dev.lm_head.forward(&s.hidden, &mut s.logits, n, Dtype::F16, Dtype::F16, &mut s.ws, st)?,
                    }
                }
            }
            if let Some(t) = &timer {
                t.record(i + 1, st)?;
            }
        }
        if let (Some(t), Some(p)) = (&timer, profile.as_mut()) {
            for (i, op) in plan.iter().enumerate() {
                *p.ms.entry(op_class(op)).or_insert(0.0) += t.ms(i, i + 1)? as f64;
            }
            for &(name, a, b) in &marks {
                *p.sub.entry(name).or_insert(0.0) += t.ms(a, b)? as f64;
            }
            // (the head rows' tail is the rest of the forward its first run counted)
            p.forwards += usize::from(!matches!(plan.first(), Some(Op::MixerOut { .. })));
        }
        Ok(())
    }

    // ----------------------------------------------------------------------------------------------------- logits and candidates

    /// The top `count` candidates (value descending, ties to the lower id) of the first `rows` rows of logits in the scratch, over the
    /// tokenizer's vocabulary (the padding ids at and past `TOKENIZER_VOCAB` are never read). Synchronises.
    pub fn candidates_of(&mut self, rows: usize, count: usize) -> Result<Candidates, ModelError> {
        if count == 0 || count > MAX_CANDIDATES {
            return Err(ModelError::Request(format!("candidate count must be 1..={MAX_CANDIDATES}")));
        }
        let GlmModel { cfg, stream, s, mutant, stage, .. } = self;
        let valid = if *mutant == Some(Mutant::UnmaskedVocab) { cfg.vocab } else { TOKENIZER_VOCAB };
        glm_glue::topk_f16(&s.logits, rows, cfg.vocab, valid, count, &mut s.topk_ids, &mut s.topk_vals, &mut s.topk_ws, stream)?;
        stage.sync(stream)?;
        let mut ids = vec![0i32; rows * count];
        let mut vals = vec![0f32; rows * count];
        s.topk_ids.download(&mut ids)?;
        s.topk_vals.download(&mut vals)?;
        Ok(Candidates { rows, count, ids, vals })
    }

    /// Debug (a test of the vocabulary mask): overwrite the padding columns (ids at and past the tokenizer's vocabulary) of the first `rows`
    /// rows of logits with the largest finite fp16, so that a top-k that reads them returns them.
    pub fn poison_padding_logits(&mut self, rows: usize) -> Result<(), ModelError> {
        self.stream.sync()?;
        let mut all: Vec<u16> = self.s.logits.to_vec()?;
        for r in 0..rows {
            all[r * self.cfg.vocab + TOKENIZER_VOCAB..(r + 1) * self.cfg.vocab].fill(0x7BFF);
        }
        self.s.logits.upload(&all)?;
        Ok(())
    }

    /// The raw fp16 logits of rows `from..from + n` of the last head (a debug path: it synchronises).
    pub fn logits_host(&self, from: usize, n: usize) -> Result<Vec<u16>, ModelError> {
        self.stream.sync()?;
        let all: Vec<u16> = match (&self.dump_logits, self.dump_head) {
            (Some(d), true) => d.to_vec()?,
            _ => self.s.logits.to_vec()?,
        };
        let v = self.cfg.vocab;
        Ok(all[from * v..(from + n) * v].to_vec())
    }

    /// Candidates for every row of a window step (decode or verify): the window, the head, the top-k.
    pub fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        if count == 0 || count > MAX_CANDIDATES {
            return Err(ModelError::Request(format!("candidate count must be 1..={MAX_CANDIDATES}")));
        }
        self.forward(b, Mode::Window, Head::AllRows)?;
        self.candidates_of(b.rows, count)
    }

    /// Prefill one chunk of one sequence: `tokens` are positions `start..start + tokens.len()`. With `want` set, also returns the candidates
    /// of the chunk's LAST row (the chunk must then end the prompt). [`GlmModel::prefill_rows`] of that one sequence, whole.
    pub fn prefill_chunk(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        want: Option<usize>,
    ) -> Result<Option<Candidates>, ModelError> {
        let b = RowBatch::from_requests(&[gb_engine::RowRequest { slot, tokens: tokens.to_vec(), first_position: start as u64 }]);
        self.prefill_rows(&b, want, usize::MAX)
    }

    /// [`GlmModel::forward`] with an image splice (see [`GlmModel::prefill_chunk_mm`]): the capture-friendly entry the oracle dump uses.
    pub fn forward_mm(&mut self, b: &RowBatch, mode: Mode, head: Head, splice: Vec<SpliceOp>) -> Result<(), ModelError> {
        self.splice = splice;
        let r = self.forward(b, mode, head);
        self.splice.clear();
        r
    }

    /// [`GlmModel::prefill_chunk`] for a chunk that holds image rows: `splice` says which rows of which image's features replace which embedding
    /// rows of the chunk. Every other row, and everything after the embedding, is the text path unchanged. The splice is cleared when the call ends,
    /// whatever its outcome.
    pub fn prefill_chunk_mm(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        want: Option<usize>,
        splice: Vec<SpliceOp>,
    ) -> Result<Option<Candidates>, ModelError> {
        for op in &splice {
            if op.n == 0 || op.dst_row + op.n > tokens.len() || op.src_row + op.n > op.feats.tokens {
                return Err(ModelError::Request(format!(
                    "an image splice of {} rows (from row {} of {}) at chunk row {} does not fit a chunk of {} rows",
                    op.n,
                    op.src_row,
                    op.feats.tokens,
                    op.dst_row,
                    tokens.len()
                )));
            }
        }
        self.splice = splice;
        let r = self.prefill_chunk(slot, start, tokens, want);
        self.splice.clear();
        r
    }

    // ------------------------------------------------------------------------------------------------------------ state readout

    /// The slot's current KDA state of KDA layer `l` (f32 `[64][128][128]`) and conv history (bf16 `[3][24576]`), after applying every
    /// pending commit (a debug path).
    pub fn kda_state(&mut self, l: usize, slot: usize) -> Result<(Vec<f32>, Vec<u16>), ModelError> {
        self.flush()?;
        self.stream.sync()?;
        let ord = self.kda_ord[l];
        if ord == usize::MAX {
            return Err(ModelError::Request(format!("layer {l} is not a KDA layer of this model")));
        }
        Ok((self.pools[ord].read_state(slot)?, self.pools[ord].read_conv(slot)?))
    }

    /// Both KDA buffers, the parity and the conv history of layer `l` (a test's view of "the whole committed state").
    pub fn kda_state_full(&mut self, l: usize, slot: usize) -> Result<(Vec<f32>, i32, Vec<u16>), ModelError> {
        self.flush()?;
        self.stream.sync()?;
        let ord = self.kda_ord[l];
        let par = self.pools[ord].read_parity()?;
        Ok((self.pools[ord].read_both(slot)?, par[slot] & 1, self.pools[ord].read_conv(slot)?))
    }

    /// The MLA latent rows `0..n` of MLA layer `l` for `slot` (fp16 bits, `[n][512]`, dequantised from an fp8 cache: what the attention
    /// kernels read), through the slot's page table (a debug path).
    pub fn latent_rows(&mut self, l: usize, slot: usize, n: usize) -> Result<Vec<u16>, ModelError> {
        self.stream.sync()?;
        let ord = self.mla_ord[l];
        Ok(self.latent[ord].read_rows(self.kv_tab.blocks_of(slot), n)?)
    }

    /// The DSA pool rows of pools `0..n_pools` of MLA layer `l` for `slot` (fp16 bits, `[n_pools][128]`).
    pub fn pool_rows(&mut self, l: usize, slot: usize, n_pools: usize) -> Result<Vec<u16>, ModelError> {
        self.stream.sync()?;
        let ord = self.mla_ord[l];
        let all: Vec<u16> = self.planes[ord].pool.to_vec()?;
        let blocks = self.kv_tab.blocks_of(slot);
        let per = dsa::POOLS_PER_PAGE;
        let mut out = Vec::with_capacity(n_pools * dsa::D);
        for p in 0..n_pools {
            let at = (blocks[p / per] as usize * per + p % per) * dsa::D;
            out.extend_from_slice(&all[at..at + dsa::D]);
        }
        Ok(out)
    }
}

/// Append `new` (rows of `row` elements) to `dst`, keeping only the last `tail` rows when a tail is asked for.
fn push_tail<T: Copy>(dst: &mut Vec<T>, new: &[T], row: usize, tail: Option<usize>) {
    dst.extend_from_slice(new);
    if let Some(k) = tail {
        if dst.len() > k * row {
            let cut = dst.len() - k * row;
            dst.drain(..cut);
        }
    }
}

/// Download the four streams after a site into the capture (a debug path: it synchronises).
fn capture_streams(
    c: &mut Capture,
    streams: &DevBuf,
    st: &Stream,
    rows: usize,
    after_mixer: bool,
    layer: usize,
) -> Result<(), ModelError> {
    st.sync()?;
    let v: Vec<f32> = streams.to_vec()?;
    let dst = if after_mixer { &mut c.mid } else { &mut c.layer_out };
    while dst.len() <= layer {
        dst.push(Vec::new());
    }
    push_tail(&mut dst[layer], &v[..rows * mhc::ROW], mhc::ROW, c.tail);
    Ok(())
}

/// A model built for a TP rank's KDA heads alone (`KdaConfig::rank`: its qkv strips and its heads' state, the rank strips of
/// `REXL3_GLM_TP_SPLIT=kda`) runs only as that rank of the head split: anything else would mis-compute, so it is refused by name.
fn kda_rank_without_split(cfg: &GlmConfig) -> ModelError {
    ModelError::Config(format!(
        "this model holds only the KDA heads {:?} of a tensor-parallel rank (its qkv_proj strips and their state: REXL3_GLM_TP_SPLIT=kda at \
         load), and its head split is not enabled (enable_tp, then enable_head_split with kda)",
        cfg.kda.rank
    ))
}

#[cfg(test)]
mod tests {
    use super::push_tail;

    #[test]
    fn a_tail_keeps_only_the_last_rows_across_appends() {
        let mut v: Vec<u8> = Vec::new();
        push_tail(&mut v, &[1, 2, 3, 4, 5, 6], 2, None);
        assert_eq!(v, vec![1, 2, 3, 4, 5, 6], "no tail: everything");
        v.clear();
        push_tail(&mut v, &[1, 2, 3, 4, 5, 6], 2, Some(2));
        assert_eq!(v, vec![3, 4, 5, 6]);
        push_tail(&mut v, &[7, 8], 2, Some(2));
        assert_eq!(v, vec![5, 6, 7, 8], "rows from the next call push the oldest out");
        push_tail(&mut v, &[9, 10, 11, 12, 13, 14], 2, Some(2));
        assert_eq!(v, vec![11, 12, 13, 14]);
        let mut short: Vec<u8> = Vec::new();
        push_tail(&mut short, &[1, 2], 2, Some(8));
        assert_eq!(short, vec![1, 2], "fewer rows than the tail: all of them");
    }
}
