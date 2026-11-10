//! The Flash-Next model on the device: weights, per-slot state and the forward (prefill chunks, decode and verify
//! windows, the MTP head). This file is the API CONTRACT the backend (`backend.rs`) and the prefix cache (`prefix.rs`)
//! are written against; the forward port fills in the bodies and may add private fields and helpers, but keeps every
//! public item below (names, types, meaning).
//!
//! Per-slot state a sequence leaves behind (what a prefix-cache snapshot must hold, nothing else may differ):
//! - KV pages of the 12 trunk attention layers ([`FlashNextModel::kv`], 2 KV heads) and of the MTP layer
//!   ([`FlashNextModel::mtp_kv`]), and the QSA pooled-key planes ([`FlashNextModel::qsa_pool`]), all addressed through one
//!   page table ([`FlashNextModel::kv_tab`] on the host, [`FlashNextModel::table`] on the device);
//! - per GDN layer the delta-rule state and conv history ([`FlashNextModel::pools`]);
//! - the PLE conv ring and token history ([`FlashNextModel::ple_ring`], [`FlashNextModel::ple_tok`]);
//! - the QSA raw-key rings ([`FlashNextModel::qsa_ring`]);
//! - the MTP head's progress ([`FlashNextModel::mtp_done`]): it runs one position ahead of the trunk, so after a prefill
//!   chunk that ends at `L` it has written its K/V row at `L`.
//!
//! Bits rules (docs/PREFIX_CACHE.md, the chunk-invariance audit): a prefill chunk's arithmetic depends only on the
//! prompt's TOTAL length through [`PrefillPath`] (decode GEMV up to 128 tokens, never cached; the pinned GEMM for every
//! chunk of a longer prompt, a 1-row tail included), never on the chunk's own row count. Windows (`window_candidates`)
//! have their own arithmetic and are never cached.
//!
//! What that means for every arm of a prefill chunk, whatever its row count (the audit's "minimum changes" list):
//! every EXL3 linear takes [`PrefillPath`]'s arm; the hyper-connection pre-mix is the per-row `hcx::pre`; the router and
//! the shared-expert gate are the per-row `small_linear_f16w`; the routed experts are `moe_topk_softmax` plus
//! `coopmk_decode`; attention is the native `attn_prefill`; sparse attention is `qsa::attend` with one part; the n-gram
//! projections are `ple::proj_prefill`. The row-count-dependent arms (FrostMoE, the HCG GEMM pre-mix, the router GEMM,
//! `qsa::attend` with 32 parts, `ple::proj_window`) belong to windows alone and are not reachable from a prefill.
#![allow(dead_code)]

use super::config::FlashNextConfig;
use super::device::{FlashNextLayer, FlashNextWeights, MixerWeights, PleWeights};
use super::ngram::{Ahead, TokContext};
use crate::error::ModelError;
use crate::forward::{Candidates, Mode, PrefillPath};
use crate::kvtables::{KvTables, BLOCK_TOKENS};
use crate::rows::RowBatch;
use gb_cuda::attention::{self, BlockTable, KvHeads, PagedKv, RowIndex};
use gb_cuda::exl3_linear::{Dtype, Exl3Weights, MAX_DECODE_ROWS};
use gb_cuda::fn_mhc::{self, Mixer, PreScratch};
use gb_cuda::gdn::{self, Batch, PreInputs, PreWeights, Record};
use gb_cuda::glue::AttnPrepCfg;
use gb_cuda::graph::{CaptureError, Graph};
use gb_cuda::moe::{MoeDims, MoeExperts, SharedExpert};
use gb_cuda::moe_prefill::{finish_shared, PrefillScratch, SharedScratchBig};
use gb_cuda::rt::HostStage;
use gb_cuda::{fmoe, fn_glue, fn_moe, glue, hcx, mtpk, ple, qfn_verify, qsa, CudaError, DevBuf, Stream};
use gb_engine::RowRequest;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// Positions of the PLE conv ring and of the QSA raw-key rings, per slot (both indexed by absolute position mod this).
pub const PLE_RING: usize = 128;
pub const QSA_RING: usize = 128;

/// Rows of the widest decode / verify window.
pub const WINDOW_ROWS: usize = 64;
/// Candidates per row the LM head's top-k can return.
pub const MAX_CANDIDATES: usize = 64;
/// Columns of the shared LM head a draft is read from (ids past it are never drafted; verification reads all of them).
pub const MTP_HEAD_N: usize = 65536;
/// The candidate domain of the MTP drafter: the first [`MTP_HEAD_N`] ids of the shared LM head, the same prefix
/// `mtpk::pick` reads.
pub(crate) fn draft_head_nv(vocab: usize) -> usize {
    MTP_HEAD_N.min(vocab)
}

/// Fail-closed validation for the exact draft-head prefix. The strip copy keeps its kernel's K split and is
/// bit-identical to the full head only for aligned widths of at least 8192 columns (64 strips, the kernel's minimum),
/// and the checkpoint's LM head must cover the whole candidate domain (a narrower head could not serve the same `pick`
/// domain the full head serves).
pub(crate) fn check_draft_head_shapes(nv: usize, head_n: usize) -> Result<(), ModelError> {
    if nv % 128 != 0 || nv < 8192 {
        return Err(ModelError::Config(format!("exact MTP draft-head strips require an aligned width >= 8192, got {nv}")));
    }
    if head_n < nv {
        return Err(ModelError::Config(format!(
            "exact MTP draft-head: checkpoint LM head width {head_n} < candidate domain {nv}"
        )));
    }
    Ok(())
}

/// Device bytes of the exact draft-head copy: the packed trellis `[K/16, N/16, 16*bits]` int16 plus the fp16 `suh`
/// `[K]` and `svh` `[N]` scales (this family's LM head has no bias). Identical to the bytes a
/// [`Exl3Weights::select_strips`] copy of `nv` columns actually owns (`Exl3Weights::weight_bytes`).
pub(crate) fn draft_head_prefix_bytes(k: usize, nv: usize, bits: usize) -> usize {
    k * nv * bits * 2 / 16 + 2 * (k + nv)
}

/// Longest MTP draft chain.
pub const MAX_DRAFTS: usize = 16;
/// Rows a prefill chunk scores and selects at once.
pub const QSA_SLAB: usize = 256;
/// Contexts up to this many tokens attend densely (every block would be selected there, so it is the same function).
pub const QSA_DENSE_CTX: usize = 2048;
/// The epsilon every norm of this model runs with; a pack that says otherwise is refused at load.
const EPS: f32 = 1e-6;

const _: () = assert!(QSA_RING >= WINDOW_ROWS + 3, "a window must not overwrite a key the next block still reads");
const _: () = assert!(PLE_RING >= WINDOW_ROWS + 9, "a window must not overwrite a conv tap the next window still reads");
const _: () = assert!(QSA_DENSE_CTX < qsa::SPARSE_FROM);
const _: () = assert!(WINDOW_ROWS <= MAX_DECODE_ROWS);

/// How a step's host inputs reach the device. [`UploadMode::Ordered`] is the engine; the other two exist only so the race
/// test can run the code this replaced next to it. Never changed outside that test.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UploadMode {
    /// Queued on the model's stream from bytes the model keeps until the stream has run the copy ([`HostStage`]).
    Ordered,
    /// The code before the fix: `DevBuf::upload`, a pageable `cudaMemcpy` on the legacy default stream. It can return before
    /// its DMA lands, and the model's stream is non-blocking, so a kernel launched after it can read the previous step's bytes.
    Legacy,
    /// [`UploadMode::Legacy`] followed by `cudaDeviceSynchronize` after every write: the old code made safe by brute force, the
    /// reference the ordered writes must equal bit for bit.
    LegacySynced,
}

static UPLOAD_MODE: AtomicU8 = AtomicU8::new(0);

/// Test hook: the way the per-step uploads are made (see [`UploadMode`]).
#[doc(hidden)]
pub fn set_upload_mode(mode: UploadMode) {
    UPLOAD_MODE.store(mode as u8, Ordering::Relaxed);
}

fn upload_mode() -> UploadMode {
    match UPLOAD_MODE.load(Ordering::Relaxed) {
        1 => UploadMode::Legacy,
        2 => UploadMode::LegacySynced,
        _ => UploadMode::Ordered,
    }
}

/// Writes `v` at the start of `dst` in the order of `st` (every per-step input of the model goes through here): a kernel
/// launched on `st` after this call reads the new bytes, and a kernel of an earlier step still reading the old ones is not
/// overtaken. The bytes are copied into `stage`, which keeps them until the model's next stream sync, so `v` may be a
/// temporary. No device synchronisation is made.
fn put<T: Copy>(stage: &mut HostStage, dst: &mut DevBuf, v: &[T], st: &Stream) -> Result<(), CudaError> {
    match upload_mode() {
        UploadMode::Ordered => stage.upload(dst, v, st),
        UploadMode::Legacy => dst.upload(v),
        UploadMode::LegacySynced => {
            dst.upload(v)?;
            gb_cuda::rt::device_sync()
        }
    }
}

/// [`put`] for the device page table.
fn put_table(stage: &mut HostStage, table: &mut BlockTable, tables: &[Vec<u32>], st: &Stream) -> Result<(), CudaError> {
    match upload_mode() {
        UploadMode::Ordered => table.upload_on(tables, stage, st),
        UploadMode::Legacy => table.upload(tables),
        UploadMode::LegacySynced => {
            table.upload(tables)?;
            gb_cuda::rt::device_sync()
        }
    }
}

/// Per GDN (linear-attention) layer, in layer order: the delta-rule state `S` (f32 `[slots][48][128][128]`) and the conv
/// history (bf16 `[slots][3][10240]`).
pub struct FlashNextGdnPools {
    pub slots: usize,
    pub s: Vec<DevBuf>,
    pub conv: Vec<DevBuf>,
}

impl FlashNextGdnPools {
    /// Zeroed pools for `layers` gated delta net layers and `slots` sequence slots.
    pub fn new(layers: usize, slots: usize) -> Result<Self, ModelError> {
        let mut s = Vec::with_capacity(layers);
        let mut conv = Vec::with_capacity(layers);
        for _ in 0..layers {
            s.push(DevBuf::zeroed(slots * gdn::S_SLOT_ELEMS * 4)?);
            conv.push(DevBuf::zeroed(slots * gdn::CONV_SLOT_ELEMS * 2)?);
        }
        Ok(Self { slots, s, conv })
    }

    /// Zero one slot's state in every layer (a new sequence starts from a zero state).
    pub fn reset_slot(&mut self, slot: usize, st: &Stream) -> Result<(), ModelError> {
        if slot >= self.slots {
            return Err(ModelError::Request(format!("slot {slot} of {}", self.slots)));
        }
        for (s, conv) in self.s.iter_mut().zip(self.conv.iter_mut()) {
            s.memset_range_async(slot * gdn::S_SLOT_ELEMS * 4, gdn::S_SLOT_ELEMS * 4, 0, st)?;
            conv.memset_range_async(slot * gdn::CONV_SLOT_ELEMS * 2, gdn::CONV_SLOT_ELEMS * 2, 0, st)?;
        }
        Ok(())
    }

    pub fn bytes(&self) -> u64 {
        self.s.iter().chain(self.conv.iter()).map(|b| b.bytes() as u64).sum()
    }
}

/// Sizes the model is built for.
#[derive(Clone, Copy, Debug)]
pub struct ModelOptions {
    pub slots: usize,
    /// KV pages (256 tokens each) shared by every slot.
    pub kv_pages: usize,
    pub max_context: usize,
    /// Largest prefill chunk (rows).
    pub prefill_rows: usize,
    /// Whether the n-gram table warm must complete before load returns (readiness gate).
    pub warm: bool,
}

/// The attention grid a captured graph was sized for: a power-of-two bucket of 512-token chunks covering the live
/// context (the kernel skips chunks past each row's position, so one graph stays exact while the context grows inside
/// its bucket). A graph only replays for a window its grid covers.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AttnGrid(usize);

impl AttnGrid {
    pub fn cover(live: usize, max_context: usize) -> Self {
        Self((live.div_ceil(512).next_power_of_two() * 512).min(max_context).max(live))
    }
    pub fn ctx(self) -> usize {
        self.0
    }
    pub fn covers(self, live: usize) -> bool {
        live <= self.0
    }
}

/// The shape a captured graph was recorded for; replaying it on any other shape would be wrong.
trait GraphKey: Copy + Eq + std::hash::Hash + std::fmt::Debug {
    fn grid(&self) -> AttnGrid;
}

/// Model-local, load-time dispatch decisions. The graph maps belong to one model, but encoding these in the key also
/// documents the capture contract: changing any weight-side stage or runtime kernel family requires a new graph.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct GraphFeatures {
    qsa_trunk: bool,
    qsa_mtp: bool,
    ple: bool,
    mtp: bool,
    fast: bool,
}

/// One bit per row: exactly the host-side QSA run partition in `run_layer`, including interleaved sequences.
/// Row positions and slots themselves stay in the uploaded device arrays, not in the graph key.
fn sparse_rows(positions: &[i32], qsa: bool) -> u64 {
    assert!(positions.len() <= WINDOW_ROWS && WINDOW_ROWS <= u64::BITS as usize);
    if !qsa {
        return 0;
    }
    positions.iter().enumerate().fold(0, |bits, (r, &p)| bits | (u64::from(p as usize >= QSA_DENSE_CTX) << r))
}

/// A verify / decode window. Sequence boundaries matter even with the same number of rows and sequences: GDN and PLE
/// receive per-sequence grids, and QSA runs are partitioned by the absolute row order. `par` is the record set the window
/// fills (`recs[par]`; the other holds what the previous window owes) and `owed` whether its GDN window kernels fold that owed
/// replay in: both are baked into the launches.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct WindowKey {
    rows: usize,
    seqs: usize,
    seq_ends: u64,
    candidates: usize,
    sparse: u64,
    par: usize,
    owed: bool,
    features: GraphFeatures,
    grid: AttnGrid,
}

impl WindowKey {
    fn for_batch(b: &RowBatch, candidates: usize, grid: AttnGrid, features: GraphFeatures) -> Self {
        let seq_ends = b.seq_start.iter().skip(1).fold(0, |bits, &end| bits | (1u64 << (end as usize - 1)));
        Self {
            rows: b.rows,
            seqs: b.n_seq(),
            seq_ends,
            candidates,
            sparse: sparse_rows(&b.positions, features.qsa_trunk),
            par: 0,
            owed: false,
            features,
            grid,
        }
    }

    /// The same window in record set `par`, with or without an owed replay folded in.
    fn in_round(self, par: usize, owed: bool) -> Self {
        Self { par, owed, ..self }
    }
}

/// An MTP draft chain: catch-up rows read from trunk stack row `src_row`, then `drafts - 1` chained steps. Both the
/// catch-up mask AND the one-row mask of every step are needed: a chain can cross the QSA threshold between drafts.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct ChainKey {
    catchup_rows: usize,
    src_row: usize,
    drafts: usize,
    sparse_catchup: u64,
    sparse_steps: u16,
    features: GraphFeatures,
    grid: AttnGrid,
}

impl ChainKey {
    fn for_chain(b: &RowBatch, src_row: usize, drafts: usize, grid: AttnGrid, features: GraphFeatures) -> Self {
        let last = b.positions[b.rows - 1] as usize;
        let sparse_steps = if features.qsa_mtp {
            (1..drafts).fold(0, |bits, j| bits | (u16::from(last + j >= QSA_DENSE_CTX) << (j - 1)))
        } else {
            0
        };
        Self {
            catchup_rows: b.rows,
            src_row,
            drafts,
            sparse_catchup: sparse_rows(&b.positions, features.qsa_mtp),
            sparse_steps,
            features,
            grid,
        }
    }
}

impl GraphKey for WindowKey {
    fn grid(&self) -> AttnGrid {
        self.grid
    }
}
impl GraphKey for ChainKey {
    fn grid(&self) -> AttnGrid {
        self.grid
    }
}

/// Where the MTP head reads its input stack: rows of the trunk's last stack, or of its own previous output.
#[derive(Clone, Copy, Debug)]
enum StackSrc {
    Trunk(usize),
    Head(usize),
}

/// An announced next prefill chunk: where it is and the token history its rows hash.
struct PleNext {
    slot: usize,
    first: usize,
    rows: usize,
    hist: Vec<u32>,
}

/// The gather of the next prefill chunk's PLE rows, started early ([`FlashNextModel::ple_prefetch_next`]).
struct PleAhead {
    slot: usize,
    first: usize,
    rows: usize,
    ahead: Ahead,
}

/// The state replay a committed window still owes: the first `keep[j]` rows of its sequence j (rows from `win.seq_start[j]`
/// of record set `par`) are not yet in the recurrent state. The conv history is rebuilt at once (the next window's `pre`
/// reads it); the rows of S are replayed by the next window kernel that runs the sequence, or standalone when none does.
struct OwedWindow {
    par: usize,
    win: RowBatch,
    keep: Vec<i32>,
}

pub struct FlashNextModel {
    pub cfg: FlashNextConfig,
    pub opts: ModelOptions,
    pub stream: Stream,
    /// Paged KV of the 12 trunk attention layers, in layer order (`PagedKv::with_kv_heads(pages, KvHeads::Two)`).
    pub kv: Vec<PagedKv>,
    /// The MTP layer's paged KV (same page table); `None` without an MTP head.
    pub mtp_kv: Option<PagedKv>,
    /// QSA pooled-key planes, one per trunk attention layer then one for the MTP layer, paged by the same table.
    pub qsa_pool: Vec<DevBuf>,
    /// QSA raw-key rings `[slots][QSA_RING][128]` f16, same order as `qsa_pool`.
    pub qsa_ring: Vec<DevBuf>,
    pub pools: FlashNextGdnPools,
    /// PLE conv ring `[slots][PLE_RING][hc * hidden]` f16.
    pub ple_ring: DevBuf,
    /// PLE token history per slot (host).
    pub ple_tok: TokContext,
    pub kv_tab: KvTables,
    pub table: BlockTable,
    /// Per slot: the position the MTP head has run through (its K/V row there is written).
    pub mtp_done: BTreeMap<usize, usize>,

    // ---- private: the weights, the scratch and the bookkeeping the forward runs on ----
    /// Shared so a layer's weights can be borrowed while the model runs it (`run_layer` takes `&mut self`).
    w: Arc<FlashNextWeights>,
    /// The exact EXL3 copy of the draft-candidate prefix of the LM head (its first [`MTP_HEAD_N`] columns): the MTP
    /// drafter reads only those columns, so this is the same function on the same strips as the full head, bit-identical
    /// for the columns `mtpk::pick` reads, without evaluating the columns it never reads. `None` when the pack has no
    /// MTP head (the full head is then used and the `pick` domain is still the same prefix).
    draft_lm_head: Option<Exl3Weights>,
    /// Rows the scratch was sized for: `max(WINDOW_ROWS, opts.prefill_rows)`.
    rows: usize,
    rowidx: RowIndex,
    /// Per-row and per-sequence index arrays of the batch being run.
    ids: DevBuf,
    positions: DevBuf,
    seq_start: DevBuf,
    seq_slot: DevBuf,
    keep: DevBuf,
    /// The residual stream stack `[rows, hc, h]` f32 and the buffers one layer reuses.
    x_streams: DevBuf,
    x_mixed: DevBuf,
    x_normed: DevBuf,
    low_buf: DevBuf,
    inj_attn: DevBuf,
    inj_mlp: DevBuf,
    block_out: DevBuf,
    mixer_out: DevBuf,
    router_logits: DevBuf,
    topk_indices: DevBuf,
    topk_weights: DevBuf,
    act_workspace: DevBuf,
    moe_out: DevBuf,
    sh_gate_scalar: DevBuf,
    qkv: DevBuf,
    z: DevBuf,
    ga: DevBuf,
    gb: DevBuf,
    fk: DevBuf,
    fv: DevBuf,
    fq: DevBuf,
    fk2: DevBuf,
    fgate: DevBuf,
    /// EXL3 decode workspace, EXL3 pinned-prefill scratch, the GDN prefill workspace, the dense attention partials,
    /// the FrostMoE window workspace and the HCX pre-mix scratch.
    ws: DevBuf,
    prefill_ws: DevBuf,
    gdn_prefill_ws: DevBuf,
    attn_ws: DevBuf,
    fmoe_ws: DevBuf,
    hcx_scratch: DevBuf,
    /// Per GDN layer, the raw qkv rows the window's commit rebuilds the conv history from, and the window's records. The
    /// records are two sets that alternate between windows (`rec_par` is the one the last window wrote): the next window's
    /// kernel replays the rows the last one owes from the set it left, while it fills the other.
    win_qkv: Vec<DevBuf>,
    recs: [Vec<Record>; 2],
    rec_par: usize,
    /// Whether the window being run folds an owed replay in (a graph's key), and the replay the last window still owes.
    owed_in: bool,
    owed: Option<OwedWindow>,
    /// i32 `[n_seq]` of the window being run: the first row in the previous record, and the rows to replay, of each of its
    /// sequences (`gdn::Owed`).
    owed_row0: DevBuf,
    owed_keep: DevBuf,
    logits: DevBuf,
    topk_ws: DevBuf,
    topk_ids: DevBuf,
    topk_vals: DevBuf,
    /// Device-side greedy gather outputs: the packed argmax column of a verify window,
    /// `[WINDOW_ROWS]` each. See [`FlashNextModel::window_candidates_dv`].
    dv_ids: DevBuf,
    dv_vals: DevBuf,
    /// PLE stage scratch (1-byte buffers when the model has no PLE stage).
    ple_host: Vec<u16>,
    /// The PLE rows of the next prefill chunk, being gathered on a worker while the GPU runs this one.
    ple_ahead: Option<PleAhead>,
    /// An announced next chunk whose gather has not started yet.
    ple_next: Option<PleNext>,
    ple_packed: DevBuf,
    ple_emb: DevBuf,
    ple_key: DevBuf,
    ple_value: DevBuf,
    ple_sg: DevBuf,
    ple_nb: DevBuf,
    /// QSA batch scratch (1-byte buffers when no layer runs its indexer).
    qsa_qk: DevBuf,
    qsa_q: DevBuf,
    qsa_kraw: DevBuf,
    qsa_scores: DevBuf,
    qsa_sel: DevBuf,
    qsa_sel_n: DevBuf,
    qsa_ws: DevBuf,
    /// MTP head scratch and its device-resident draft chain.
    mtp_streams: DevBuf,
    mtp_xs: DevBuf,
    mtp_xe: DevBuf,
    mtp_h: DevBuf,
    mtp_e: DevBuf,
    chain_pos: DevBuf,
    drafts_dev: DevBuf,
    conf_dev: DevBuf,
    /// One row of zero K and V, and a one-row index, for the MTP cache's row at position 0 (see `zero_mtp_row0`).
    zero_kv: DevBuf,
    one_row: RowIndex,
    /// Drafts after the first one whose probability is below this are not verified.
    mtp_conf: f32,
    /// Trunk positions of the rows now in `x_streams` (the last window or prefill chunk of one sequence).
    stack_pos0: usize,
    stack_rows: usize,
    stack_slot: usize,
    /// Rows of the last prefill chunk or window whose collapsed trunk rows are still in `x_mixed` (0 once the MTP head
    /// has overwritten them).
    mixed_rows: usize,
    /// The last window, and per slot the rows of it a `commit` keeps until the next forward replays them.
    last_window: Option<RowBatch>,
    pending: BTreeMap<usize, usize>,
    /// Replayable windows and draft chains, and the context the grid of the one being run covers.
    graphs: HashMap<WindowKey, Graph>,
    mtp_graphs: HashMap<ChainKey, Graph>,
    attn_ctx: Option<usize>,
    use_graphs: bool,
    /// The fused hyper-connection pre-mix (`REXL3_FAST=0` takes the reference kernels instead).
    fast: bool,
    /// Whether the QSA indexer runs at all (`REXL3_QSA=0` leaves every attention layer dense).
    qsa: bool,
    /// A page table change made outside a forward that the device table has not seen yet.
    table_dirty: bool,
    /// The host bytes of the per-step uploads queued on `stream`, kept until the stream has run them (see [`put`]).
    stage: HostStage,
    /// Device bytes by category, for [`FlashNextModel::allocated_bytes`].
    bytes_weights: u64,
    bytes_kv: u64,
    bytes_state: u64,
    bytes_scratch: u64,
    /// Grouped MoE prefill scratch (`REXL3_FLASH_NEXT_GROUPED_MOE_PREFILL=1` and a supported shape), sized for
    /// `rows` rows of the exact Flash-Next dims. `None` keeps the CoopMK decode.
    moe_pf: Option<PrefillScratch>,
    /// The grouped prefill's per-layer routed experts (index == trunk layer index): pointer tables into the loaded
    /// weights and the stacked fp16 scales the grouped kernels read. Empty when the path is off; a `None` layer has an
    /// unsupported width layout and falls back to CoopMK for that layer.
    moe_experts: Vec<Option<MoeExperts>>,
    /// The shared expert's big-row scratch of the grouped prefill, and the two fp32 outputs the path works in (the
    /// routed sum, and the scaled shared-expert output), before the single fp32 -> bf16 cast.
    moe_shared_scratch: Option<SharedScratchBig>,
    moe_out_f32: DevBuf,
    shared_out_f32: DevBuf,
    /// The hcg hyper-connection pre-mix (the declared prefill contract): the same `hcx::pre` math as tensor-core
    /// GEMMs on fp16 hi+lo Dekker-split operands. Sized for `rows` rows of the exact Flash-Next dims; `None` keeps
    /// the fp32 `hcx::pre` everywhere (unsupported shape, or allocation refused).
    hcg: bool,
    hcg_scratch: Option<DevBuf>,
}

/// An on-by-default switch: only `0` turns it off.
fn env_on(key: &str) -> bool {
    std::env::var(key).map(|v| v != "0").unwrap_or(true)
}

/// The on-by-default switch that selects the grouped MoE prefill (`MoeExperts::run_prefill`) for Flash-Next.
/// Only an explicit `0` turns it off (a documented opt-out for debugging); unset means on.
fn grouped_moe_switch() -> bool {
    env_on("REXL3_FLASH_NEXT_GROUPED_MOE_PREFILL")
}

/// The declared prefill contract engages when the family shapes qualify for the tensor-core GEMM path
/// (H % 4 == 0, hc >= 1, rank >= 1 — true of Flash-Next). There is no switch; the compensated-fp16 hcg
/// pre-mix is unconditional for tiled prefill on qualifying shapes (the boot log names it as the active
/// contract; decode and GEMV prefill always keep the fp32 `hcx::pre` reference).
fn hcg_applies(h: usize, hc: usize, rank: usize) -> bool {
    h % 4 == 0 && hc >= 1 && rank >= 1
}

/// hcg engages only for tiled prefill: decode (`Mode::Window`) and GEMV prefill keep the fp32 `hcx::pre` path. This is
/// the only place the mode is allowed to decide, so the contract applies for the process but is exercised only in the
/// prefill chunk, not the hot decode token (fp32 all the way there).
fn hcg_for_mode(hcg: bool, mode: Mode) -> bool {
    hcg && matches!(mode, Mode::Prefill(PrefillPath::Tiled))
}

/// The exact Flash-Next MoE dims the grouped prefill path is built for: hidden 2560, inter 640, 512 routed
/// experts, top-k 10. `scale`/`limit` are unused by the grouped kernels because routing stays on the Qwen
/// `moe_topk_softmax` path (`limit <= 0` disables the SwiGLU clamps). `None` when the shape is outside what
/// the grouped path supports (the caller then keeps the CoopMK decode).
fn flash_next_moe_dims(hidden: usize, interm: usize, experts: usize, topk: usize) -> Option<MoeDims> {
    if hidden % 128 != 0 || interm % 128 != 0 || experts == 0 || experts > 1024 || topk == 0 || topk > 16 || topk > experts {
        return None;
    }
    Some(MoeDims { hidden, inter: interm, experts, topk, scale: 1.0, limit: 0.0 })
}

/// Enabled by default. `0` forces eager windows and MTP chains; capture refusal also falls back to eager.
fn graphs_enabled() -> bool {
    env_on("REXL3_GRAPHS")
}

/// Sizes the model can be built for, checked before a weight is read.
fn check_opts(opts: &ModelOptions) -> Result<(), ModelError> {
    let ModelOptions { slots, kv_pages, max_context, prefill_rows, .. } = *opts;
    if slots == 0 || kv_pages == 0 || max_context == 0 || prefill_rows == 0 {
        return Err(ModelError::Config(format!("{opts:?}: every size must be positive")));
    }
    let cap = gdn::PREFILL_MAX_ROWS.min(attention::MAX_PREFILL_ROWS);
    if prefill_rows > cap {
        return Err(ModelError::Config(format!("a prefill chunk of {prefill_rows} rows is past what the kernels take ({cap})")));
    }
    if max_context > kv_pages * BLOCK_TOKENS {
        return Err(ModelError::Config(format!(
            "a context of {max_context} tokens needs more than the {kv_pages} pages of {BLOCK_TOKENS} the pool holds"
        )));
    }
    Ok(())
}

/// One EXL3 linear of a trunk batch: the decode GEMV for a window and for the prefill of a short prompt, the pinned
/// fixed-algorithm GEMM for every chunk of a longer one. The choice is the mode's, never `m`'s (see [`PrefillPath`]).
#[allow(clippy::too_many_arguments)]
fn lin(
    wt: &Exl3Weights,
    x: &DevBuf,
    y: &mut DevBuf,
    m: usize,
    mode: Mode,
    ws: &mut DevBuf,
    pws: &mut DevBuf,
    st: &Stream,
) -> Result<(), ModelError> {
    match mode {
        Mode::Prefill(PrefillPath::Tiled) => wt.forward_prefill_pinned(x, y, m, Dtype::BF16, pws, st)?,
        Mode::Window | Mode::Prefill(PrefillPath::Gemv) => wt.forward(x, y, m, Dtype::BF16, Dtype::BF16, ws, st)?,
    }
    Ok(())
}

/// The MTP head's own two input projections, which run in fp16 over `rows * hc` stack rows. The Tiled prefill arm pins
/// them like any other prefill linear; everywhere else they take the decode GEMV, in slices of its row limit when the
/// stack is wider (the GEMV is row invariant, so the slicing is not a change of function).
#[allow(clippy::too_many_arguments)]
fn lin_f16(
    wt: &Exl3Weights,
    x: &DevBuf,
    y: &mut DevBuf,
    m: usize,
    mode: Mode,
    ws: &mut DevBuf,
    pws: &mut DevBuf,
    st: &Stream,
) -> Result<(), ModelError> {
    if matches!(mode, Mode::Prefill(PrefillPath::Tiled)) {
        wt.forward_prefill_pinned(x, y, m, Dtype::F16, pws, st)?;
        return Ok(());
    }
    // the decode GEMV is row invariant, so a stack wider than its row limit is the same function run in slices (and a
    // slice stays capturable in a graph, which the pinned GEMM is not)
    let mut r0 = 0;
    while r0 < m {
        let n = MAX_DECODE_ROWS.min(m - r0);
        let xs = x.subview(r0 * wt.k * 2, n * wt.k * 2);
        let mut ys = y.subview(r0 * wt.n * 2, n * wt.n * 2);
        wt.forward(&xs, &mut ys, n, Dtype::F16, Dtype::F16, ws, st)?;
        r0 += n;
    }
    Ok(())
}

/// The linears a trunk or MTP batch pushes through `Exl3Weights` (the routed experts go through the MoE kernels'
/// pointer tables instead, so they need none of this scratch).
fn forward_linears(w: &FlashNextWeights) -> Vec<&Exl3Weights> {
    let mut v = Vec::new();
    for layer in w.layers.iter().chain(w.mtp.as_ref().map(|m| &m.layer)) {
        match &layer.mixer {
            MixerWeights::Linear(m) => v.extend([&m.in_proj_qkv, &m.in_proj_z, &m.out_proj]),
            MixerWeights::Full(m) => {
                v.extend([&m.q_proj, &m.k_proj, &m.v_proj, &m.o_proj]);
                if let Some(ix) = &m.indexer {
                    v.push(&ix.qk_proj);
                }
            }
        }
    }
    v
}

type PinnedPrefillShape = (usize, usize, usize, Dtype);

/// Every row count a Tiled prefill can dispatch, retaining activation dtype because trunk and MTP use different buffers.
/// The pinned GEMM itself operates on reconstructed fp16 matrices; dtype is still recorded in diagnostics/tests so the
/// checked inventory mirrors the actual EXL3 calls. MTP's hidden projection sees `hc` stacked rows per token; the
/// embedding projection sees one row per token.
fn pinned_prefill_shapes(
    trunk: &[(usize, usize)],
    mtp_hidden: &[(usize, usize)],
    mtp_embedding: &[(usize, usize)],
    hc: usize,
    rows: usize,
) -> Vec<PinnedPrefillShape> {
    let mut shapes = Vec::new();
    for &(k, n) in trunk {
        shapes.extend((1..=rows).map(|m| (m, k, n, Dtype::BF16)));
    }
    for &(k, n) in mtp_hidden {
        shapes.extend((1..=rows).map(|m| (m * hc, k, n, Dtype::F16)));
    }
    for &(k, n) in mtp_embedding {
        shapes.extend((1..=rows).map(|m| (m, k, n, Dtype::F16)));
    }
    shapes.sort_by_key(|&(m, k, n, dtype)| (m, k, n, dtype as i32));
    shapes.dedup();
    shapes
}

/// Test seam for the loader's fail-closed pinned-GEMM boot gate.
fn run_pinned_prefill_boot<E>(
    shapes: &[PinnedPrefillShape],
    init: impl FnOnce() -> Result<(), E>,
    mut check: impl FnMut(usize, usize, usize, Dtype) -> Result<(), E>,
) -> Result<(), E> {
    init()?;
    for &(m, k, n, dtype) in shapes {
        check(m, k, n, dtype)?;
    }
    Ok(())
}

fn init_pinned_prefill(w: &FlashNextWeights, hc: usize, rows: usize) -> Result<(), ModelError> {
    let trunk = forward_linears(w).into_iter().map(|x| (x.k, x.n)).collect::<Vec<_>>();
    let mtp_hidden = w.mtp.as_ref().map_or_else(Vec::new, |m| vec![(m.fc_hidden.k, m.fc_hidden.n)]);
    let mtp_embedding = w.mtp.as_ref().map_or_else(Vec::new, |m| vec![(m.fc_embedding.k, m.fc_embedding.n)]);
    let shapes = pinned_prefill_shapes(&trunk, &mtp_hidden, &mtp_embedding, hc, rows);
    run_pinned_prefill_boot(
        &shapes,
        || gb_cuda::exl3_linear::prefill_init().map_err(|e| e.to_string()),
        |m, k, n, dtype| gb_cuda::exl3_linear::prefill_pin_check(m, k, n).map_err(|e| {
            format!(
                "the pinned prefill GEMM ({}) does not run a {m} x {k} x {n} {dtype:?} problem here ({e}); row-invariant prefill cannot be guaranteed on this cuBLAS",
                gb_cuda::exl3_linear::prefill_pin_name()
            )
        }),
    )
    .map_err(|e| ModelError::Config(e.to_string()))
}

/// The pinned-prefill scratch a chunk of `rows` rows needs: the widest of every linear it runs (the MTP head's hidden
/// projection takes `rows * hc` stack rows).
fn prefill_scratch_need(w: &FlashNextWeights, hc: usize, rows: usize) -> usize {
    let mut need = 1 << 20;
    for x in forward_linears(w) {
        need = need.max(x.prefill_scratch_bytes(rows));
    }
    if let Some(m) = &w.mtp {
        need = need.max(m.fc_hidden.prefill_scratch_bytes(rows * hc));
        need = need.max(m.fc_embedding.prefill_scratch_bytes(rows));
    }
    need
}

/// The decode workspace: the widest of every linear at the decode kernel's row limit, and of the LM head at a window.
fn decode_scratch_need(w: &FlashNextWeights, window_rows: usize) -> usize {
    let mut need = 1 << 20;
    for x in forward_linears(w) {
        need = need.max(x.workspace_bytes(MAX_DECODE_ROWS));
    }
    if let Some(m) = &w.mtp {
        need = need.max(m.fc_hidden.workspace_bytes(MAX_DECODE_ROWS));
        need = need.max(m.fc_embedding.workspace_bytes(MAX_DECODE_ROWS));
    }
    need.max(w.lm_head.workspace_bytes(window_rows)).max(w.lm_head.workspace_bytes(1))
}

// --- thin safe wrappers over the raw-pointer glue launchers (every argument is a whole device buffer) ---

fn g_embed(
    table: &DevBuf,
    ids: &DevBuf,
    out: &DevBuf,
    rows: usize,
    dim: usize,
    vocab: usize,
    st: &Stream,
) -> Result<(), ModelError> {
    // SAFETY: bf16 table `[vocab, dim]`, i32 ids `[rows]`, bf16 out `[rows, dim]`, all checked by the caller's sizing.
    unsafe { glue::embed(table.ptr().cast_const(), ids.ptr().cast_const(), out.ptr(), rows, dim, vocab, st) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn g_small_linear(x: &DevBuf, w: &DevBuf, y: &DevBuf, rows: usize, k: usize, n: usize, st: &Stream) -> Result<(), ModelError> {
    // SAFETY: bf16 x `[rows, k]`, f16 w `[n, k]`, bf16 y `[rows, n]`.
    unsafe { glue::small_linear_f16w(x.ptr().cast_const(), w.ptr().cast_const(), y.ptr(), rows, k, n, st) }?;
    Ok(())
}

/// `out = out * sigmoid(gate)` in place over `n` elements.
fn g_gate_mul(out: &DevBuf, gate: &DevBuf, n: usize, st: &Stream) -> Result<(), ModelError> {
    let p = out.ptr();
    // SAFETY: the kernel reads every element before it writes it, by the same thread, so in place is allowed.
    unsafe { glue::gate_mul(p.cast_const(), gate.ptr().cast_const(), p, n, st) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn g_topk(
    logits: &DevBuf,
    rows: usize,
    vocab: usize,
    count: usize,
    ids: &DevBuf,
    vals: &DevBuf,
    ws: &DevBuf,
    st: &Stream,
) -> Result<(), ModelError> {
    // SAFETY: bf16 logits `[rows, vocab]`, i32 ids and f32 vals `[rows, count]`, workspace of the asked size.
    unsafe { glue::topk(logits.ptr().cast_const(), rows, vocab, count, ids.ptr(), vals.ptr(), ws.ptr(), ws.bytes(), st) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn g_attn_prep(
    cfg: &AttnPrepCfg,
    q_in: &DevBuf,
    k_in: &DevBuf,
    positions: &DevBuf,
    q_w: &DevBuf,
    k_w: &DevBuf,
    q_out: &DevBuf,
    gate_out: &DevBuf,
    k_out: &DevBuf,
    rows: usize,
    st: &Stream,
) -> Result<(), ModelError> {
    // SAFETY: the shapes `AttnPrepCfg` documents, which this model's buffers are sized for.
    unsafe {
        glue::attn_prep(
            cfg,
            q_in.ptr().cast_const(),
            k_in.ptr().cast_const(),
            positions.ptr().cast_const(),
            q_w.ptr().cast_const(),
            k_w.ptr().cast_const(),
            q_out.ptr(),
            gate_out.ptr(),
            k_out.ptr(),
            rows,
            st,
        )
    }?;
    Ok(())
}

impl FlashNextModel {
    /// Load the pack in `dir` (weights through `WeightMap`, the n-gram table, its warm read) and allocate every buffer.
    pub fn load(dir: &Path, opts: ModelOptions) -> Result<Self, ModelError> {
        check_opts(&opts)?;
        let cfg = FlashNextConfig::from_dir(dir).map_err(ModelError::Config)?;
        cfg.check_shapes().map_err(ModelError::Config)?;
        if cfg.rms_norm_eps as f32 != EPS {
            return Err(ModelError::Config(format!(
                "rms_norm_eps {} (the Flash-Next kernels are built for {EPS})",
                cfg.rms_norm_eps
            )));
        }
        let w = FlashNextWeights::load(dir, &cfg, opts.warm, |_, _| {})?;
        Self::new(cfg, w, opts)
    }

    /// The model on already loaded weights (the path the device tests take).
    pub fn new(cfg: FlashNextConfig, w: FlashNextWeights, opts: ModelOptions) -> Result<Self, ModelError> {
        check_opts(&opts)?;
        let ModelOptions { slots, kv_pages, max_context, prefill_rows, .. } = opts;
        let (h, hc, rank) = (cfg.hidden, cfg.hc_streams, cfg.hc_lowrank);
        let (ch, pd) = (hc * h, cfg.ple_embed_dim);
        let (top_k, num_experts, interm) = (cfg.experts_per_tok, cfg.experts, cfg.moe_intermediate);
        let (q_dim, kv_dim) = (cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim);
        let rows = WINDOW_ROWS.max(prefill_rows);
        init_pinned_prefill(&w, hc, prefill_rows)?;
        let lin_layers = cfg.linear_layers();
        let full_layers = cfg.full_layers();
        let max_blocks = max_context.div_ceil(BLOCK_TOKENS);
        let has_mtp = w.mtp.is_some();
        // The exact draft-head prefix: the MTP drafter reads only the first MTP_HEAD_N columns of the shared LM head,
        // so the model owns an exact device copy of those 128-column strips and evaluates only them. The target head is
        // untouched and still used for verification; the strip copy is bit-identical for the columns `pick` reads.
        let draft_lm_head = if has_mtp {
            let nv = draft_head_nv(cfg.vocab);
            check_draft_head_shapes(nv, w.lm_head.n)?;
            let head = w.lm_head.select_strips(&[(0, nv / 128)]).map_err(ModelError::Cuda)?;
            eprintln!(
                "rexl3 event=mtp_exact_prefix enabled=true full_vocab={} draft_vocab={} added_bytes={} precision=unchanged_exl3",
                cfg.vocab,
                nv,
                head.weight_bytes()
            );
            Some(head)
        } else {
            None
        };
        // the planes and the indexer launches go together: with REXL3_QSA=0 the layers attend densely at any context
        let has_qsa = w.has_qsa() && env_on("REXL3_QSA");

        let mut bytes_kv = 0u64;
        let kv = (0..full_layers).map(|_| PagedKv::with_kv_heads(kv_pages, KvHeads::Two)).collect::<Result<Vec<_>, _>>()?;
        for c in &kv {
            bytes_kv += (c.k.bytes() + c.v.bytes()) as u64;
        }
        let mtp_kv = match has_mtp {
            true => {
                let c = PagedKv::with_kv_heads(kv_pages, KvHeads::Two)?;
                bytes_kv += (c.k.bytes() + c.v.bytes()) as u64;
                Some(c)
            }
            false => None,
        };
        // the pooled key planes live in the same pages as K and V
        let planes = if has_qsa { full_layers + has_mtp as usize } else { 0 };
        let qsa_pool =
            (0..planes).map(|_| DevBuf::zeroed(kv_pages * qsa::PAGE_BLOCKS * qsa::DK * 2)).collect::<Result<Vec<_>, _>>()?;
        bytes_kv += qsa_pool.iter().map(|b| b.bytes() as u64).sum::<u64>();

        let pools = FlashNextGdnPools::new(lin_layers, slots)?;
        let qsa_ring = (0..planes).map(|_| DevBuf::zeroed(slots * QSA_RING * qsa::DK * 2)).collect::<Result<Vec<_>, _>>()?;
        let ple_ring = DevBuf::zeroed(if w.ple.is_some() { slots * PLE_RING * ch * 2 } else { 1 })?;
        let bytes_state = pools.bytes() + qsa_ring.iter().map(|b| b.bytes() as u64).sum::<u64>() + ple_ring.bytes() as u64;

        let pws_bytes = prefill_scratch_need(&w, hc, rows);
        let ws_bytes = decode_scratch_need(&w, WINDOW_ROWS);
        let nb_cap = max_context.div_ceil(qsa::CR);
        let fast = std::env::var("REXL3_FAST").map(|v| v != "0").unwrap_or(true);
        let ple_on = w.ple.is_some();
        let ple_words = w.ple.as_ref().map_or(0, |p| p.table.heads * p.table.words);
        let one = |on: bool, n: usize| if on { n } else { 1 };
        let bytes_weights = w.bytes as u64 + draft_lm_head.as_ref().map_or(0, |h| h.weight_bytes() as u64);
        // the weights hold device buffers, which are Send but not Sync: this Arc never crosses a thread, it only lets a
        // layer be borrowed while `run_layer` takes the model mutably
        #[allow(clippy::arc_with_non_send_sync)]
        let w = Arc::new(w);

        let mut moe_pf = if grouped_moe_switch() {
            flash_next_moe_dims(cfg.hidden, cfg.moe_intermediate, cfg.experts, cfg.experts_per_tok)
                .and_then(|dims| PrefillScratch::new(&dims, rows).ok())
        } else {
            None
        };
        if let Some(p) = moe_pf.as_mut() {
            p.use_parallel_plan(true);
        }
        // The grouped prefill's per-layer routed experts and its shared-expert scratch, built when the switch is on and
        // the shape is supported. `MoeExperts::from_exl3` uploads only the stacked fp16 scales the grouped kernels read
        // (construction; the pointer tables reference the weights' own device buffers), so it runs once here, not per chunk.
        let (moe_experts, moe_shared_scratch, moe_out_f32, shared_out_f32) = if moe_pf.is_some() {
            let dims = flash_next_moe_dims(cfg.hidden, cfg.moe_intermediate, cfg.experts, cfg.experts_per_tok)
                .expect("the grouped scratch was allocated, so the shape is supported");
            let shared_inter = w.layers[0].moe.shared_expert.gate.n;
            if w.layers.iter().any(|l| l.moe.shared_expert.gate.n != shared_inter) {
                return Err(ModelError::Config("grouped MoE prefill: layers disagree on the shared-expert width".into()));
            }
            let mut exs = Vec::with_capacity(w.layers.len());
            for l in &w.layers {
                let g: Vec<&Exl3Weights> = l.moe.experts.iter().map(|e| &e.gate).collect();
                let u: Vec<&Exl3Weights> = l.moe.experts.iter().map(|e| &e.up).collect();
                let dn: Vec<&Exl3Weights> = l.moe.experts.iter().map(|e| &e.down).collect();
                // A layer whose width layout the grouped path cannot represent (e.g. split gate/up widths, a per-expert
                // shape mismatch) refuses the layer with `CudaError(1)` (INVALID): it keeps the CoopMK decode instead of
                // failing the boot. Any other error (allocation, fatal) still propagates.
                match MoeExperts::from_exl3(&g, &u, &dn, dims) {
                    Ok(ex) => exs.push(Some(ex)),
                    Err(e) if e.0 == 1 => {
                        eprintln!(
                            "grouped MoE prefill: layer {} has an unsupported mixed-width layout; using CoopMK for it",
                            exs.len()
                        );
                        exs.push(None);
                    }
                    Err(e) => return Err(ModelError::Cuda(e)),
                }
            }
            let ss = SharedScratchBig::new_dims(h, shared_inter, rows).map_err(ModelError::Cuda)?;
            (exs, Some(ss), DevBuf::zeroed(rows * h * 4)?, DevBuf::zeroed(rows * h * 4)?)
        } else {
            (Vec::new(), None, DevBuf::zeroed(1)?, DevBuf::zeroed(1)?)
        };

        // The hcg pre-mix scratch: a ~260 MB per-chunk buffer for the max row count on every qualifying shape
        // (H % 4 == 0, hc and rank positive — true of the family). `None` keeps the fp32 `hcx::pre` path
        // (unsupported shape, or allocation refused).
        let (hcg, hcg_scratch) = if hcg_applies(h, hc, rank) {
            (true, Some(DevBuf::zeroed(hcx::gemm_scratch_bytes(rows, h, hc, rank))?))
        } else {
            (false, None)
        };

        let mut m = Self {
            stream: Stream::new_nonblocking()?,
            kv,
            mtp_kv,
            qsa_pool,
            qsa_ring,
            pools,
            ple_ring,
            ple_tok: TokContext::new(slots),
            kv_tab: KvTables::new(kv_pages, slots, max_blocks),
            table: BlockTable::from_host(&vec![Vec::new(); slots], max_blocks)?,
            mtp_done: BTreeMap::new(),

            rows,
            rowidx: RowIndex { slot: DevBuf::zeroed_view(rows * 4)?, pos: DevBuf::zeroed_view(rows * 4)?, rows },
            ids: DevBuf::zeroed(rows * 4)?,
            positions: DevBuf::zeroed(rows * 4)?,
            seq_start: DevBuf::zeroed((slots + 1) * 4)?,
            seq_slot: DevBuf::zeroed(slots * 4)?,
            keep: DevBuf::zeroed((slots + 1) * 4)?,
            x_streams: DevBuf::zeroed_view(rows * ch * 4)?,
            x_mixed: DevBuf::zeroed_view(rows * h * 2)?,
            x_normed: DevBuf::zeroed(one(!fast, rows * ch * 4))?,
            low_buf: DevBuf::zeroed(one(!fast, rows * rank * 4))?,
            inj_attn: DevBuf::zeroed(rows * hc * 4)?,
            inj_mlp: DevBuf::zeroed(rows * hc * 4)?,
            block_out: DevBuf::zeroed(rows * h * 2)?,
            mixer_out: DevBuf::zeroed_view(rows * q_dim.max(gdn::VALUE_HEADS * gdn::HEAD_DIM) * 2)?,
            router_logits: DevBuf::zeroed(rows * num_experts * 2)?,
            topk_indices: DevBuf::zeroed(rows * (top_k + 1) * 4)?,
            topk_weights: DevBuf::zeroed(rows * (top_k + 1) * 4)?,
            act_workspace: DevBuf::zeroed(fn_moe::coopmk_workspace_bytes(rows, top_k + 1, interm))?,
            moe_out: DevBuf::zeroed(rows * h * 2)?,
            sh_gate_scalar: DevBuf::zeroed(rows * 2)?,
            qkv: DevBuf::zeroed(rows * (2 * q_dim).max(gdn::CONV_DIM) * 2)?,
            z: DevBuf::zeroed(rows * gdn::VALUE_HEADS * gdn::HEAD_DIM * 2)?,
            ga: DevBuf::zeroed(rows * gdn::VALUE_HEADS * 2)?,
            gb: DevBuf::zeroed(rows * gdn::VALUE_HEADS * 2)?,
            fk: DevBuf::zeroed(rows * kv_dim * 2)?,
            fv: DevBuf::zeroed(rows * kv_dim * 2)?,
            fq: DevBuf::zeroed_view(rows * q_dim * 2)?,
            fk2: DevBuf::zeroed(rows * kv_dim * 2)?,
            fgate: DevBuf::zeroed(rows * q_dim * 2)?,
            ws: DevBuf::zeroed(ws_bytes)?,
            prefill_ws: DevBuf::zeroed(pws_bytes)?,
            gdn_prefill_ws: DevBuf::zeroed(gdn::prefill_workspace_bytes(rows))?,
            attn_ws: DevBuf::zeroed(attention::decode_workspace_bytes(WINDOW_ROWS, max_context))?,
            fmoe_ws: fmoe::workspace(WINDOW_ROWS, top_k, h, interm, fmoe::DEFAULT_KC)?,
            hcx_scratch: DevBuf::zeroed(hcx::scratch_bytes(hc, rank))?,
            win_qkv: (0..lin_layers).map(|_| DevBuf::zeroed(WINDOW_ROWS * gdn::CONV_DIM * 2)).collect::<Result<Vec<_>, _>>()?,
            recs: [
                (0..lin_layers).map(|_| Record::alloc(WINDOW_ROWS)).collect::<Result<Vec<_>, _>>()?,
                (0..lin_layers).map(|_| Record::alloc(WINDOW_ROWS)).collect::<Result<Vec<_>, _>>()?,
            ],
            rec_par: 0,
            owed_in: false,
            owed: None,
            owed_row0: DevBuf::zeroed((slots + 1) * 4)?,
            owed_keep: DevBuf::zeroed((slots + 1) * 4)?,
            logits: DevBuf::zeroed(WINDOW_ROWS * cfg.vocab * 2)?,
            topk_ws: DevBuf::zeroed(glue::topk_workspace_bytes(WINDOW_ROWS, cfg.vocab, MAX_CANDIDATES))?,
            topk_ids: DevBuf::zeroed(WINDOW_ROWS * MAX_CANDIDATES * 4)?,
            topk_vals: DevBuf::zeroed(WINDOW_ROWS * MAX_CANDIDATES * 4)?,
            dv_ids: DevBuf::zeroed(WINDOW_ROWS * 4)?,
            dv_vals: DevBuf::zeroed(WINDOW_ROWS * 4)?,
            ple_host: Vec::new(),
            ple_ahead: None,
            ple_next: None,
            ple_packed: DevBuf::zeroed(one(ple_on, rows * ple_words * 2))?,
            ple_emb: DevBuf::zeroed(one(ple_on, rows * pd * 2))?,
            ple_key: DevBuf::zeroed(one(ple_on, rows * ch * 2))?,
            ple_value: DevBuf::zeroed(one(ple_on, rows * h * 2))?,
            ple_sg: DevBuf::zeroed(one(ple_on, rows * hc * 4))?,
            ple_nb: DevBuf::zeroed(one(ple_on, rows * ch * 2))?,
            qsa_qk: DevBuf::zeroed(one(has_qsa, rows * qsa::QK * 2))?,
            qsa_q: DevBuf::zeroed_view(one(has_qsa, rows * qsa::N_HEADS * qsa::DK * 2))?,
            qsa_kraw: DevBuf::zeroed(one(has_qsa, rows * qsa::DK * 2))?,
            qsa_scores: DevBuf::zeroed(one(has_qsa, QSA_SLAB.max(WINDOW_ROWS) * nb_cap * 4))?,
            qsa_sel: DevBuf::zeroed_view(one(has_qsa, rows * qsa::TOPK * 4))?,
            qsa_sel_n: DevBuf::zeroed_view(one(has_qsa, rows * 4))?,
            qsa_ws: DevBuf::zeroed(one(
                has_qsa,
                qsa::attend_workspace_bytes(rows, 1).max(qsa::attend_workspace_bytes(WINDOW_ROWS, qsa::WINDOW_PARTS)),
            ))?,
            mtp_streams: DevBuf::zeroed_view(one(has_mtp, rows * ch * 4))?,
            mtp_xs: DevBuf::zeroed_view(one(has_mtp, rows * ch * 2))?,
            mtp_xe: DevBuf::zeroed_view(one(has_mtp, rows * h * 2))?,
            mtp_h: DevBuf::zeroed_view(one(has_mtp, rows * ch * 2))?,
            mtp_e: DevBuf::zeroed_view(one(has_mtp, rows * h * 2))?,
            chain_pos: DevBuf::zeroed(MAX_DRAFTS * 4)?,
            drafts_dev: DevBuf::zeroed(MAX_DRAFTS * 4)?,
            conf_dev: DevBuf::zeroed(MAX_DRAFTS * 4)?,
            zero_kv: DevBuf::zeroed(one(has_mtp, KvHeads::Two.token_elems() * 2))?,
            one_row: RowIndex { slot: DevBuf::zeroed(4)?, pos: DevBuf::zeroed(4)?, rows: 1 },
            mtp_conf: std::env::var("REXL3_MTP_CONF").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0),
            stack_pos0: 0,
            stack_rows: 0,
            stack_slot: 0,
            mixed_rows: 0,
            last_window: None,
            pending: BTreeMap::new(),
            graphs: HashMap::new(),
            mtp_graphs: HashMap::new(),
            attn_ctx: None,
            use_graphs: graphs_enabled(),
            fast,
            qsa: has_qsa,
            moe_pf,
            moe_experts,
            moe_shared_scratch,
            moe_out_f32,
            shared_out_f32,
            hcg,
            hcg_scratch,
            draft_lm_head,
            table_dirty: false,
            stage: HostStage::new(),
            bytes_weights,
            bytes_kv,
            bytes_state,
            bytes_scratch: 0,
            w,
            cfg,
            opts,
        };
        m.bytes_scratch = m.scratch_bytes();
        // every buffer above was zeroed (`DevBuf::zeroed`: an asynchronous `cudaMemset`) or written (the empty page table, the
        // weights) on the legacy default stream, which the model's non-blocking stream does not wait for: settle them once
        // here, so no kernel of the first step can run before them or have its output zeroed after it
        gb_cuda::rt::device_sync()?;
        Ok(m)
    }

    pub fn vocab_size(&self) -> usize {
        self.cfg.vocab
    }

    pub fn has_mtp(&self) -> bool {
        self.w.mtp.is_some()
    }

    /// Start a sequence in `slot`: zero its recurrent state, drop its pages and MTP progress.
    pub fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        if slot >= self.opts.slots {
            return Err(ModelError::Request(format!("slot {slot} of {}", self.opts.slots)));
        }
        self.flush()?;
        self.mtp_done.remove(&slot);
        self.pending.remove(&slot);
        self.stack_rows = 0;
        self.mixed_rows = 0;
        self.pools.reset_slot(slot, &self.stream)?;
        let ch = self.cfg.hc_streams * self.cfg.hidden;
        if self.w.ple.is_some() {
            let (off, len) = (slot * PLE_RING * ch * 2, PLE_RING * ch * 2);
            self.ple_ring.memset_range_async(off, len, 0, &self.stream)?;
        }
        let (off, len) = (slot * QSA_RING * qsa::DK * 2, QSA_RING * qsa::DK * 2);
        for r in self.qsa_ring.iter_mut() {
            r.memset_range_async(off, len, 0, &self.stream)?;
        }
        self.kv_tab.release(slot);
        self.upload_table()?;
        self.sync()?;
        Ok(())
    }

    /// End the sequence in `slot` and give back its pages.
    pub fn end_slot(&mut self, slot: usize) {
        self.pending.remove(&slot);
        self.mtp_done.remove(&slot);
        // what the slot's last window still owes its state is dropped with the sequence
        if let Some(ow) = &mut self.owed {
            for (j, &sl) in ow.win.seq_slot.iter().enumerate() {
                if sl as usize == slot {
                    ow.keep[j] = 0;
                }
            }
            if ow.keep.iter().all(|&k| k == 0) {
                self.owed = None;
            }
        }
        if slot < self.opts.slots {
            self.kv_tab.release(slot);
            self.table_dirty = true;
        }
    }

    /// One prefill chunk of ONE sequence (`b`: rows of one slot at consecutive positions). `path` is
    /// `PrefillPath::for_prompt(total)` of the whole prompt, the same for every chunk of it. Candidates of the chunk's
    /// last row when `want` is set.
    pub fn prefill_chunk(
        &mut self,
        b: &RowBatch,
        path: PrefillPath,
        want: Option<usize>,
    ) -> Result<Option<Candidates>, ModelError> {
        if b.n_seq() != 1 {
            return Err(ModelError::Request(format!("a prefill chunk is one sequence, got {}", b.n_seq())));
        }
        if b.rows == 0 || b.rows > self.opts.prefill_rows {
            return Err(ModelError::Request(format!(
                "a prefill chunk takes 1..={} rows, got {}",
                self.opts.prefill_rows, b.rows
            )));
        }
        if path == PrefillPath::Gemv && b.rows > PrefillPath::GEMV_MAX_PROMPT {
            return Err(ModelError::Request(format!(
                "a prefill on the GEMV path takes at most {} rows, got {}",
                PrefillPath::GEMV_MAX_PROMPT,
                b.rows
            )));
        }
        if b.parents.is_some() || b.rope3.is_some() {
            return Err(ModelError::Request("a Flash-Next prefill chunk is a plain text chain".into()));
        }
        self.forward(b, Mode::Prefill(path))?;
        self.note_stack(b);
        self.mixed_rows = b.rows;
        let Some(count) = want else { return Ok(None) };
        self.candidates_of_last_row(b.rows, count)
    }

    /// A decode / verify window over one or more sequences: candidates for every row. The window's recurrent state is
    /// committed by [`FlashNextModel::commit`] + the next window (which folds the replay in) or [`FlashNextModel::flush`].
    pub fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        if count == 0 || count > MAX_CANDIDATES {
            return Err(ModelError::Request(format!("candidate count must be 1..={MAX_CANDIDATES}, got {count}")));
        }
        if b.rows == 0 || b.rows > WINDOW_ROWS {
            return Err(ModelError::Request(format!("a window takes 1..={WINDOW_ROWS} rows, got {}", b.rows)));
        }
        if b.parents.is_some() {
            return Err(ModelError::Request("a Flash-Next window of trees is not built".into()));
        }
        let rows = b.rows;
        let (par, owed) = self.begin_window(b)?;
        self.last_window = Some(b.clone());
        self.prepare_batch(b)?;
        self.prepare_ple(b)?;
        self.note_stack(b);
        let live = Self::live_of(b);
        let key = WindowKey::for_batch(b, count, AttnGrid::cover(live, self.opts.max_context), self.graph_features())
            .in_round(par, owed);
        self.run_graph(Self::window_graphs, key, live, |m| m.window_body(b, count))?;
        self.mixed_rows = rows;
        self.sync()?;
        let mut ids = vec![0i32; rows * count];
        let mut vals = vec![0.0f32; rows * count];
        self.topk_ids.download(&mut ids)?;
        self.topk_vals.download(&mut vals)?;
        Ok(Candidates { rows, count, ids, vals })
    }

    /// The device-side greedy verification window: the same graph as [`FlashNextModel::window_candidates`]
    /// runs, but instead of downloading the whole `[rows, count]` candidate table, the argmax column is
    /// packed on the device (`gb_cuda::qfn_verify::greedy_gather`) and only `[rows]` ids + values come
    /// back: `Candidates { count: 1 }` whose single candidate of each row is the greedy winner (ties by
    /// lowest id, exactly the `glue::topk` order `try_choose` resolves). A row with a NaN candidate comes
    /// back with a NaN value, exactly like the full download would show it.
    ///
    /// The gather is always on: there is no switch, and the genuine readback syncs of the window stay
    /// (this only swaps a full candidate-table download for the packed argmax column). Device errors are
    /// surfaced the same way (the backend turns them into NaN rows).
    pub fn window_candidates_dv(&mut self, b: &RowBatch, count: usize) -> Result<Option<Candidates>, ModelError> {
        if count == 0 || count > MAX_CANDIDATES {
            return Err(ModelError::Request(format!("candidate count must be 1..={MAX_CANDIDATES}, got {count}")));
        }
        if b.rows == 0 || b.rows > WINDOW_ROWS {
            return Err(ModelError::Request(format!("a window takes 1..={WINDOW_ROWS} rows, got {}", b.rows)));
        }
        if b.parents.is_some() {
            return Err(ModelError::Request("a Flash-Next window of trees is not built".into()));
        }
        let rows = b.rows;
        let (par, owed) = self.begin_window(b)?;
        self.last_window = Some(b.clone());
        self.prepare_batch(b)?;
        self.prepare_ple(b)?;
        self.note_stack(b);
        let live = Self::live_of(b);
        let key = WindowKey::for_batch(b, count, AttnGrid::cover(live, self.opts.max_context), self.graph_features())
            .in_round(par, owed);
        self.run_graph(Self::window_graphs, key, live, |m| m.window_body(b, count))?;
        self.mixed_rows = rows;
        self.sync()?;
        // the window's top-k table is on the device: pack its argmax column and read back only that
        unsafe {
            qfn_verify::greedy_gather(
                &self.topk_ids,
                &self.topk_vals,
                count,
                rows,
                &mut self.dv_ids,
                &mut self.dv_vals,
                &self.stream,
            )
        }
        .map_err(ModelError::Cuda)?;
        // the gather is asynchronous on the stream: sync before the readback, exactly like the full-table
        // path syncs before its downloads (`window_candidates`). Without it the synchronous memcpys below
        // race the launch and read stale dv buffers.
        self.sync()?;
        let mut ids = vec![0i32; rows];
        let mut vals = vec![0.0f32; rows];
        self.dv_ids.download_range(0, &mut ids)?;
        self.dv_vals.download_range(0, &mut vals)?;
        static DV_FIRST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DV_FIRST.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!("[qfn-dv] device-side greedy gather engaged: rows={rows}, candidates_per_row={count}");
        }
        Ok(Some(Candidates { rows, count: 1, ids, vals }))
    }

    /// The raw bf16 LM-head logits `[n, vocab]` of rows `first..first + n` of the last prefill chunk or window, whose
    /// collapsed trunk rows are still on the device, into `out[..n * vocab]`. At most [`WINDOW_ROWS`] rows a call, so an
    /// all-position parity test reads a long chunk in slices with bounded host memory. Call it before the MTP head runs:
    /// the head overwrites those rows, and asking for them afterwards is refused. Reads no per-slot state, writes none.
    pub fn last_rows_logits(&mut self, first: usize, n: usize, out: &mut [u16]) -> Result<(), ModelError> {
        let (h, vocab) = (self.cfg.hidden, self.cfg.vocab);
        if n == 0 || n > WINDOW_ROWS || first + n > self.mixed_rows {
            return Err(ModelError::Request(format!(
                "rows {first}..{} of the last chunk: it left {} rows, and a call reads 1..={WINDOW_ROWS}",
                first + n,
                self.mixed_rows
            )));
        }
        if out.len() < n * vocab {
            return Err(ModelError::Request(format!("{} logits for {n} rows of {vocab}", out.len())));
        }
        let x = self.x_mixed.subview(first * h * 2, n * h * 2);
        self.w.lm_head.forward(&x, &mut self.logits, n, Dtype::BF16, Dtype::BF16, &mut self.ws, &self.stream)?;
        self.sync()?;
        self.logits.download_range(0, &mut out[..n * vocab])?;
        Ok(())
    }

    /// Keep the first `keep` rows of `slot`'s last window.
    pub fn commit(&mut self, slot: usize, keep: usize) {
        self.pending.insert(slot, keep);
    }

    /// Apply pending commits now (before anything reads a slot's state): the conv history, and the rows of S replayed here
    /// instead of being folded into the next window kernel.
    pub fn flush(&mut self) -> Result<(), ModelError> {
        let mut launched = self.flush_conv()?;
        if let Some(ow) = self.owed.take() {
            self.replay_owed(ow.par, &ow.win, &ow.keep)?;
            launched = true;
        }
        if launched {
            self.sync()?;
        }
        Ok(())
    }

    /// The conv history of the pending commits, at once (the next window's `pre` reads it). The replay of their rows of S is
    /// recorded as owed (see [`OwedWindow`]), not run: the next window kernel that runs the sequence folds it in. Whether
    /// anything was launched. No synchronisation: the stream orders it before everything queued after.
    fn flush_conv(&mut self) -> Result<bool, ModelError> {
        if self.pending.is_empty() {
            return Ok(false);
        }
        let Some(win) = self.last_window.take() else {
            self.pending.clear();
            return Ok(false);
        };
        let keep: Vec<i32> = (0..win.n_seq())
            .map(|j| self.pending.get(&(win.seq_slot[j] as usize)).copied().unwrap_or(0).min(win.seq_rows(j).len()) as i32)
            .collect();
        self.pending.clear();
        if keep.iter().all(|&k| k == 0) {
            return Ok(false);
        }
        // every window folds in or replays what its predecessor owed, so nothing is owed here. If something were, the window that
        // just ran read a state that lacks those rows, and its record set is the one about to be written over
        if self.owed.is_some() {
            return Err(ModelError::StateLost("a window ran without taking the previous window's owed replay".into()));
        }
        put(&mut self.stage, &mut self.keep, &keep, &self.stream)?;
        // the window's own index arrays (a draft chain may have rewritten them since): the commit replays exactly its rows
        put(&mut self.stage, &mut self.seq_start, &win.seq_start, &self.stream)?;
        put(&mut self.stage, &mut self.seq_slot, &win.seq_slot, &self.stream)?;
        let batch = Batch {
            seq_start: &self.seq_start,
            slot: &self.seq_slot,
            n_seq: win.n_seq(),
            rows: win.rows,
            n_slots: self.pools.slots,
        };
        for li in 0..self.win_qkv.len() {
            gdn::commit_conv(&self.stream, &self.win_qkv[li], &mut self.pools.conv[li], &batch, &self.keep)?;
        }
        self.owed = Some(OwedWindow { par: self.rec_par, win, keep });
        Ok(true)
    }

    /// Replay `keep[j]` rows of sequence j of window `win` (its rows in record set `par`) into the recurrent state with the
    /// standalone commit kernel: the rows no window kernel is going to fold in (a sequence that does not run in the next window,
    /// or a caller that wants the state itself).
    fn replay_owed(&mut self, par: usize, win: &RowBatch, keep: &[i32]) -> Result<(), ModelError> {
        if keep.iter().all(|&k| k == 0) {
            return Ok(());
        }
        put(&mut self.stage, &mut self.keep, keep, &self.stream)?;
        put(&mut self.stage, &mut self.seq_start, &win.seq_start, &self.stream)?;
        put(&mut self.stage, &mut self.seq_slot, &win.seq_slot, &self.stream)?;
        let batch = Batch {
            seq_start: &self.seq_start,
            slot: &self.seq_slot,
            n_seq: win.n_seq(),
            rows: win.rows,
            n_slots: self.pools.slots,
        };
        for li in 0..self.recs[par].len() {
            gdn::commit_state(&self.stream, &self.recs[par][li], &mut self.pools.s[li], &batch, &self.keep)?;
        }
        Ok(())
    }

    /// The window over `b` is about to run: the previous window's owed rows of S, as its kernels will see them. For the
    /// sequences of `b` that owe rows, uploads their (first row in the previous record, rows to replay); the rows owed by
    /// sequences that do not run in `b` are replayed here, before their record set can be overwritten. Whether any is left
    /// for the window kernels to fold in (from the record set `rec_par` holds now).
    fn fold_owed(&mut self, b: &RowBatch) -> Result<bool, ModelError> {
        let Some(ow) = self.owed.take() else { return Ok(false) };
        if ow.par != self.rec_par {
            return Err(ModelError::StateLost("the owed replay's records are not the last window's".into()));
        }
        let n = b.n_seq();
        let (mut row0, mut keep) = (vec![0i32; n], vec![0i32; n]);
        let mut left = ow.keep.clone();
        for (i, &sl) in b.seq_slot.iter().enumerate() {
            if let Some(j) = ow.win.seq_slot.iter().position(|&x| x == sl) {
                row0[i] = ow.win.seq_start[j];
                keep[i] = ow.keep[j];
                left[j] = 0;
            }
        }
        // `owed` is already forgotten: a failure here leaves state that no longer matches the tokens committed, whatever the
        // CUDA code says
        self.replay_owed(ow.par, &ow.win, &left).map_err(|e| ModelError::StateLost(e.to_string()))?;
        if keep.iter().all(|&k| k == 0) {
            return Ok(false);
        }
        put(&mut self.stage, &mut self.owed_row0, &row0, &self.stream)?;
        put(&mut self.stage, &mut self.owed_keep, &keep, &self.stream)?;
        Ok(true)
    }

    /// Everything a window over `b` starts from: the pending commits made and the previous window's owed rows of S settled
    /// (see [`FlashNextModel::fold_owed`]). Returns the record set the window fills (the sets alternate while a replay is owed,
    /// so the one it folds in stays intact) and whether it folds a replay in.
    fn begin_window(&mut self, b: &RowBatch) -> Result<(usize, bool), ModelError> {
        self.flush_conv()?;
        let owed = self.fold_owed(b)?;
        if owed {
            self.rec_par ^= 1;
        }
        self.owed_in = owed;
        Ok((self.rec_par, owed))
    }

    /// After a prefill chunk of `slot` at `first..first + chunk.len()`: the MTP head over every row whose next token is
    /// known (`next`: the token after the chunk, if the prompt continues). Updates `mtp_done`.
    pub fn mtp_after_prefill(
        &mut self,
        slot: usize,
        first: usize,
        chunk: &[u32],
        next: Option<u32>,
        path: PrefillPath,
    ) -> Result<(), ModelError> {
        if !self.has_mtp() || chunk.is_empty() {
            return Ok(());
        }
        let mut ids: Vec<u32> = chunk[1..].to_vec();
        if let Some(t) = next {
            ids.push(t);
        }
        let rows = ids.len();
        // the head reads the trunk's stack, which must still hold exactly this chunk
        if rows == 0 || self.stack_slot != slot || self.stack_pos0 != first || self.stack_rows < rows {
            return Ok(());
        }
        if !self.mtp_done.contains_key(&slot) {
            self.zero_mtp_row0(slot)?;
        }
        let b = RowBatch::from_requests(&[RowRequest { slot, first_position: (first + 1) as u64, tokens: ids }]);
        self.prepare_batch(&b)?;
        let mode = Mode::Prefill(path);
        // The stateless prefill body is the only path for supported prefill calls: it writes the persistent MTP KV
        // and QSA indexer state, skipping the head's attention output / MoE / collapse / LM head, which the draft
        // chain re-derives from trunk rows (and falls back to the full `mtp_body` for unsupported mixer/mode).
        self.mtp_prefill_body(&b, StackSrc::Trunk(0), mode)?;
        self.sync()?;
        self.mtp_done.insert(slot, first + rows);
        Ok(())
    }

    /// Up to `k` MTP drafts for `slot`, whose history is `tokens` (the last one not yet run through the trunk). The head
    /// first catches up on every trunk row still in `x_streams` it has not seen (each paired with the token that followed
    /// it), then chains on its own output with the token and position of every step staged on the device, so the whole
    /// chain is one replayable graph and one host read. Drafts after the first one below `REXL3_MTP_CONF` are dropped.
    pub fn mtp_propose(&mut self, slot: usize, tokens: &[u32], k: usize) -> Result<Vec<u32>, ModelError> {
        let k = k.min(MAX_DRAFTS);
        if !self.has_mtp() || k == 0 || tokens.len() < 2 || self.stack_rows == 0 || self.stack_slot != slot {
            return Ok(Vec::new());
        }
        // the draft chain reads no recurrent state: the conv history of a pending commit is made now, its rows of S ride the
        // next window
        self.flush_conv()?;
        let end = tokens.len() - 1; // trunk positions [start, end) still need their MTP rows
        let start = self.mtp_done.get(&slot).copied().unwrap_or(0).max(self.stack_pos0);
        if end <= start || end > self.stack_pos0 + self.stack_rows || end - start > WINDOW_ROWS || end + k > self.opts.max_context
        {
            return Ok(Vec::new());
        }
        if !self.mtp_done.contains_key(&slot) {
            self.zero_mtp_row0(slot)?;
        }
        let b = RowBatch::from_requests(&[RowRequest {
            slot,
            first_position: (start + 1) as u64,
            tokens: tokens[start + 1..=end].to_vec(),
        }]);
        self.prepare_batch(&b)?;
        // pages for the whole chain, and its step positions
        let tail = RowBatch::from_requests(&[RowRequest { slot, first_position: (end + k - 1) as u64, tokens: vec![0] }]);
        self.ensure_pages(&tail)?;
        let pos: Vec<i32> = (0..k).map(|j| (end + j) as i32).collect();
        put(&mut self.stage, &mut self.chain_pos, &pos, &self.stream)?;
        let live = end + k;
        let src_row = start - self.stack_pos0;
        let key = ChainKey::for_chain(&b, src_row, k, AttnGrid::cover(live, self.opts.max_context), self.graph_features());
        self.run_graph(Self::chain_graphs, key, live, |m| m.mtp_chain(&b, src_row, k))?;
        // `mtp_body` does this during capture/eager, but replay does not execute its host bookkeeping.
        self.mixed_rows = 0;
        self.mtp_done.insert(slot, end);
        self.sync()?;
        let mut ids = vec![0i32; k];
        let mut conf = vec![0f32; k];
        self.drafts_dev.download_range(0, &mut ids)?;
        self.conf_dev.download_range(0, &mut conf)?;
        let mut drafts = Vec::with_capacity(k);
        for j in 0..k {
            if j > 0 && conf[j] < self.mtp_conf {
                break;
            }
            drafts.push(ids[j] as u32);
        }
        Ok(drafts)
    }

    /// Upload the host page table (`kv_tab`) to the device table after a change made outside a forward.
    pub fn upload_table(&mut self) -> Result<(), ModelError> {
        let tables: Vec<Vec<u32>> = (0..self.opts.slots).map(|s| self.kv_tab.blocks_of(s).to_vec()).collect();
        put_table(&mut self.stage, &mut self.table, &tables, &self.stream)?;
        self.table_dirty = false;
        Ok(())
    }

    /// Waits for the model's stream; the host bytes of the uploads it has run are reused from here on.
    fn sync(&mut self) -> Result<(), ModelError> {
        self.stage.sync(&self.stream)?;
        Ok(())
    }

    /// Device bytes allocated per category, for the plan == allocations test: (weights, kv pages, per-slot state,
    /// scratch and workspaces).
    pub fn allocated_bytes(&self) -> (u64, u64, u64, u64) {
        (self.bytes_weights, self.bytes_kv, self.bytes_state, self.bytes_scratch)
    }

    // ---------------------------------------------------------------- private

    /// Every buffer that is neither a weight, a KV page nor per-slot state.
    fn scratch_bytes(&self) -> u64 {
        let mut n = 0u64;
        for b in [
            &self.rowidx.slot,
            &self.rowidx.pos,
            &self.ids,
            &self.positions,
            &self.seq_start,
            &self.seq_slot,
            &self.keep,
            &self.x_streams,
            &self.x_mixed,
            &self.x_normed,
            &self.low_buf,
            &self.inj_attn,
            &self.inj_mlp,
            &self.block_out,
            &self.mixer_out,
            &self.router_logits,
            &self.topk_indices,
            &self.topk_weights,
            &self.act_workspace,
            &self.moe_out,
            &self.sh_gate_scalar,
            &self.qkv,
            &self.z,
            &self.ga,
            &self.gb,
            &self.fk,
            &self.fv,
            &self.fq,
            &self.fk2,
            &self.fgate,
            &self.ws,
            &self.prefill_ws,
            &self.gdn_prefill_ws,
            &self.attn_ws,
            &self.fmoe_ws,
            &self.hcx_scratch,
            &self.logits,
            &self.topk_ws,
            &self.topk_ids,
            &self.topk_vals,
            &self.dv_ids,
            &self.dv_vals,
            &self.ple_packed,
            &self.ple_emb,
            &self.ple_key,
            &self.ple_value,
            &self.ple_sg,
            &self.ple_nb,
            &self.qsa_qk,
            &self.qsa_q,
            &self.qsa_kraw,
            &self.qsa_scores,
            &self.qsa_sel,
            &self.qsa_sel_n,
            &self.qsa_ws,
            &self.mtp_streams,
            &self.mtp_xs,
            &self.mtp_xe,
            &self.mtp_h,
            &self.mtp_e,
            &self.chain_pos,
            &self.drafts_dev,
            &self.conf_dev,
            &self.owed_row0,
            &self.owed_keep,
            &self.zero_kv,
            &self.one_row.slot,
            &self.one_row.pos,
            &self.table.buf,
        ] {
            n += b.bytes() as u64;
        }
        for b in &self.win_qkv {
            n += b.bytes() as u64;
        }
        for r in self.recs.iter().flatten() {
            n += (r.q.bytes() + r.k.bytes() + r.v.bytes() + r.g_decay.bytes() + r.beta.bytes()) as u64;
        }
        if let Some(p) = &self.moe_pf {
            n += p.bytes() as u64;
        }
        n += self.moe_out_f32.bytes() as u64 + self.shared_out_f32.bytes() as u64;
        if let Some(ss) = &self.moe_shared_scratch {
            n += ss.bytes() as u64;
        }
        for ex in &self.moe_experts {
            n += ex.as_ref().map_or(0, |e| e.device_bytes()) as u64;
        }
        if let Some(s) = &self.hcg_scratch {
            n += s.bytes() as u64;
        }
        n
    }

    /// The live context a batch needs: one past its highest position.
    fn live_of(b: &RowBatch) -> usize {
        b.positions.iter().map(|&p| p as usize + 1).max().unwrap_or(1)
    }

    fn ensure_pages(&mut self, b: &RowBatch) -> Result<(), ModelError> {
        let mut changed = false;
        for i in 0..b.n_seq() {
            let r = b.seq_rows(i);
            if r.is_empty() {
                return Err(ModelError::Request("a sequence of the batch has no rows".into()));
            }
            let last = b.positions[r.end - 1] as usize;
            if last >= self.opts.max_context {
                return Err(ModelError::Request(format!("position {last} is past the context of {}", self.opts.max_context)));
            }
            let slot = b.seq_slot[i] as usize;
            if slot >= self.opts.slots {
                return Err(ModelError::Request(format!("slot {slot} of {}", self.opts.slots)));
            }
            if (last + 1).div_ceil(BLOCK_TOKENS) > self.kv_tab.blocks_of(slot).len() {
                self.kv_tab.ensure(slot, last + 1).map_err(|e| ModelError::Request(format!("{e:?}")))?;
                changed = true;
            }
        }
        if changed || self.table_dirty {
            self.upload_table()?;
        }
        Ok(())
    }

    fn prepare_batch(&mut self, b: &RowBatch) -> Result<(), ModelError> {
        if b.rows == 0 || b.rows > self.rows {
            return Err(ModelError::Request(format!("a step takes 1..={} rows, got {}", self.rows, b.rows)));
        }
        // in the stream's order (`put`): the first kernel that reads an array sees this step's bytes
        let (stage, st) = (&mut self.stage, &self.stream);
        put(stage, &mut self.ids, &b.ids, st)?;
        put(stage, &mut self.positions, &b.positions, st)?;
        put(stage, &mut self.seq_start, &b.seq_start, st)?;
        put(stage, &mut self.seq_slot, &b.seq_slot, st)?;
        put(stage, &mut self.rowidx.slot, &b.row_slot, st)?;
        put(stage, &mut self.rowidx.pos, &b.positions, st)?;
        self.ensure_pages(b)
    }

    /// Announce that the prefill chunk after the one about to run is `prompt[first..end]`. Its PLE rows are then gathered
    /// on a worker thread, started once the chunk about to run has its own rows (end of `prepare_ple`), so they are
    /// ready when the next chunk's `prepare_ple` runs and the GPU never waits for the table. The rows depend only on the
    /// prompt's tokens (and the EOS before position 0): this changes when table bytes are read, not which.
    /// `prepare_ple` uses the result only when the chunk's slot, first position and complete token history (context
    /// included) are exactly what was hashed here, and gathers inline otherwise.
    pub fn ple_prefetch_next(&mut self, slot: usize, prompt: &[u32], first: usize, end: usize) {
        self.ple_next = None;
        let Some(p) = &self.w.ple else { return };
        if first >= end || end > prompt.len() || !p.table.tuning().ahead {
            return;
        }
        let eos = p.table.eos;
        let at = |q: isize| if q < 0 { eos } else { prompt[q as usize] };
        let mut hist = Vec::with_capacity(end - first + 2);
        hist.push(at(first as isize - 2));
        hist.push(at(first as isize - 1));
        hist.extend(prompt[first..end].iter().copied());
        self.ple_next = Some(PleNext { slot, first, rows: end - first, hist });
    }

    /// The PLE stage's host work for a trunk batch: hash every row's n-grams, gather their packed table rows and upload
    /// them, and record the rows' tokens as context for later batches.
    fn prepare_ple(&mut self, b: &RowBatch) -> Result<(), ModelError> {
        let w = Arc::clone(&self.w);
        let Some(p) = &w.ple else { return Ok(()) };
        let t = &p.table;
        let per = t.heads * t.words;
        if self.ple_host.len() < b.rows * per {
            self.ple_host.resize(b.rows * per, 0);
        }
        let mut direct: Option<Vec<u16>> = None;
        for i in 0..b.n_seq() {
            let r = b.seq_rows(i);
            let slot = b.seq_slot[i] as usize;
            let first = b.positions[r.start] as usize;
            let hist = self.ple_tok.history(slot, first, &b.ids[r.clone()], t.eos);
            // a worker's gather is for exactly one single-sequence chunk; any other batch leaves it where it is
            let this = |a: &PleAhead| b.n_seq() == 1 && a.slot == slot && a.first == first && a.rows == r.len();
            let ready = if self.ple_ahead.as_ref().is_some_and(this) { self.ple_ahead.take() } else { None };
            direct = ready.and_then(|a| a.ahead.take(&hist, 2)).filter(|v| v.len() == b.rows * per);
            if direct.is_none() {
                t.gather(&hist, 2, &mut self.ple_host[r.start * per..r.end * per]).map_err(ModelError::Request)?;
            }
            self.ple_tok.store(slot, first, &b.ids[r.clone()]);
        }
        let rows = direct.as_deref().unwrap_or(&self.ple_host[..b.rows * per]);
        put(&mut self.stage, &mut self.ple_packed, rows, &self.stream)?;
        // this chunk is staged: the next one's rows are gathered while its layers run on the GPU
        if let Some(n) = self.ple_next.take() {
            self.ple_ahead = t.gather_ahead(n.hist, 2).ok().map(|ahead| PleAhead { slot: n.slot, first: n.first, rows: n.rows, ahead });
        }
        Ok(())
    }

    /// The PLE stage on the device: `x_streams += gate * value + silu(conv(rms(gate * value)))` for every row.
    fn run_ple(&mut self, p: &PleWeights, b: &RowBatch, mode: Mode) -> Result<(), ModelError> {
        let (rows, d, hc, pd) = (b.rows, self.cfg.hidden, self.cfg.hc_streams, self.cfg.ple_embed_dim);
        let (ch, slots, t) = (hc * d, self.pools.slots, &p.table);
        let st = &self.stream;
        ple::rows(&self.ple_packed, &p.head_bias, &mut self.ple_emb, rows, t.heads, t.k, st)?;
        // a window and a prefill chunk each keep their own bits whatever their size (a short last chunk included)
        let proj = match mode {
            Mode::Window => ple::proj_window,
            Mode::Prefill(_) => ple::proj_prefill,
        };
        proj(&self.ple_emb, &p.key_proj, &mut self.ple_key, rows, pd, ch, st)?;
        proj(&self.ple_emb, &p.value_proj, &mut self.ple_value, rows, pd, d, st)?;
        let norms = ple::GateNorms { key: &p.norm_key, query: &p.norm_query, conv: &p.norm_conv };
        let scale = 1.0 / (d as f32).sqrt();
        ple::gate(
            &self.x_streams,
            &self.ple_key,
            &self.ple_value,
            &norms,
            &mut self.ple_sg,
            &mut self.ple_nb,
            rows,
            d,
            hc,
            self.cfg.rms_norm_eps as f32,
            scale,
            st,
        )?;
        let br = ple::BatchRows {
            positions: &self.positions,
            seq_start: &self.seq_start,
            seq_slot: &self.seq_slot,
            n_seq: b.n_seq(),
            rows,
        };
        ple::conv(
            &mut self.x_streams,
            &self.ple_sg,
            &self.ple_value,
            &self.ple_nb,
            &self.ple_ring,
            &p.conv_w,
            &br,
            d,
            hc,
            slots,
            PLE_RING,
            st,
        )?;
        ple::ring_store(&self.ple_nb, &mut self.ple_ring, &br, ch, slots, PLE_RING, st)?;
        Ok(())
    }

    fn forward(&mut self, b: &RowBatch, mode: Mode) -> Result<(), ModelError> {
        self.flush()?;
        self.owed_in = false;
        if mode == Mode::Window {
            self.last_window = Some(b.clone());
        }
        self.prepare_batch(b)?;
        self.prepare_ple(b)?;
        self.compute_layers(b, mode)?;
        self.sync()?;
        Ok(())
    }

    /// The hyper-connection pre-mix of one stage. Both arms are per row (`REXL3_FAST=0` takes the reference kernels the
    /// fused one is checked against), so neither can make a chunk's bits depend on its row count; the arm is fixed for
    /// the whole process, never chosen per call. For tiled prefill with the hcg switch on (`hcx::pre_gemm`), the two
    /// low-rank products run as tensor-core GEMMs on fp16 operands; everything else keeps the fp32 `hcx::pre`.
    fn pre_mix(&mut self, mix: &Mixer<'_>, attn: bool, mode: Mode, rows: usize) -> Result<(), ModelError> {
        let (h, hc, rank) = (self.cfg.hidden, self.cfg.hc_streams, self.cfg.hc_lowrank);
        if self.fast {
            let inj = if attn { &mut self.inj_attn } else { &mut self.inj_mlp };
            if hcg_for_mode(self.hcg && self.hcg_scratch.is_some(), mode) {
                let s = self.hcg_scratch.as_mut().expect("hcg on: scratch allocated");
                hcx::pre_gemm(&self.x_streams, mix, &mut self.x_mixed, Some(inj), s, h, hc, rank, EPS, rows, &self.stream)?;
            } else {
                let s = &mut self.hcx_scratch;
                hcx::pre(&self.x_streams, mix, &mut self.x_mixed, Some(inj), s, h, hc, rank, EPS, rows, &self.stream)?;
            }
        } else {
            let scratch = PreScratch { normed: &mut self.x_normed, low: &mut self.low_buf };
            let inj = if attn { &mut self.inj_attn } else { &mut self.inj_mlp };
            fn_mhc::hc_pre(&self.x_streams, mix, scratch, &mut self.x_mixed, inj, h, hc, rank, EPS, rows, &self.stream)?;
        }
        Ok(())
    }

    /// Collapse the stream stack into `x_mixed` through `mix` (no inject rows).
    fn collapse(&mut self, mix: &Mixer<'_>, mode: Mode, rows: usize) -> Result<(), ModelError> {
        let (h, hc, rank) = (self.cfg.hidden, self.cfg.hc_streams, self.cfg.hc_lowrank);
        if self.fast {
            if hcg_for_mode(self.hcg && self.hcg_scratch.is_some(), mode) {
                let s = self.hcg_scratch.as_mut().expect("hcg on: scratch allocated");
                hcx::pre_gemm(&self.x_streams, mix, &mut self.x_mixed, None, s, h, hc, rank, EPS, rows, &self.stream)?;
            } else {
                let s = &mut self.hcx_scratch;
                hcx::pre(&self.x_streams, mix, &mut self.x_mixed, None, s, h, hc, rank, EPS, rows, &self.stream)?;
            }
        } else {
            let scratch = PreScratch { normed: &mut self.x_normed, low: &mut self.low_buf };
            fn_mhc::hc_head(&self.x_streams, mix, scratch, &mut self.x_mixed, h, hc, rank, EPS, rows, &self.stream)?;
        }
        Ok(())
    }

    /// One decoder layer on `self.x_streams`: `li` indexes the GDN pools / records of a linear layer, `fi` the KV cache of
    /// a full-attention layer (`None`: the MTP layer's own cache).
    fn run_layer(
        &mut self,
        layer: &FlashNextLayer,
        li: usize,
        fi: Option<usize>,
        b: &RowBatch,
        mode: Mode,
        moe_layer: Option<usize>,
    ) -> Result<(), ModelError> {
        let rows = b.rows;
        let h = self.cfg.hidden;
        let hc = self.cfg.hc_streams;
        let top_k = self.cfg.experts_per_tok;
        let num_experts = self.cfg.experts;
        let interm = self.cfg.moe_intermediate;
        let window = mode == Mode::Window;

        // 1. attention hyper-connection pre-mix
        self.pre_mix(&layer.attn_hc.mixer(), true, mode, rows)?;

        // 2. the mixer: a gated delta net or grouped-query attention
        match &layer.mixer {
            MixerWeights::Linear(m) => {
                let st = &self.stream;
                let qkv_buf = if window { &mut self.win_qkv[li] } else { &mut self.qkv };
                lin(&m.in_proj_qkv, &self.x_mixed, qkv_buf, rows, mode, &mut self.ws, &mut self.prefill_ws, st)?;
                lin(&m.in_proj_z, &self.x_mixed, &mut self.z, rows, mode, &mut self.ws, &mut self.prefill_ws, st)?;
                g_small_linear(&self.x_mixed, &m.in_proj_a, &self.ga, rows, h, gdn::VALUE_HEADS, st)?;
                g_small_linear(&self.x_mixed, &m.in_proj_b, &self.gb, rows, h, gdn::VALUE_HEADS, st)?;

                let batch =
                    Batch { seq_start: &self.seq_start, slot: &self.seq_slot, n_seq: b.n_seq(), rows, n_slots: self.pools.slots };
                let wts = PreWeights { conv_w: &m.conv_w, a_log: &m.a_log, dt_bias: &m.dt_bias };
                if window {
                    let inp = PreInputs { qkv: &self.win_qkv[li], a: &self.ga, b: &self.gb };
                    let par = self.rec_par;
                    gdn::pre(st, &inp, &wts, &self.pools.conv[li], &batch, &mut self.recs[par][li])?;
                    if self.owed_in {
                        let owed = gdn::Owed { rec: &self.recs[par ^ 1][li], row0: &self.owed_row0, keep: &self.owed_keep };
                        gdn::window_owed(st, &self.recs[par][li], &owed, &mut self.pools.s[li], &batch, &mut self.mixer_out)?;
                    } else {
                        gdn::window(st, &self.recs[par][li], &self.pools.s[li], &batch, &mut self.mixer_out)?;
                    }
                } else {
                    let inp = PreInputs { qkv: &self.qkv, a: &self.ga, b: &self.gb };
                    gdn::prefill(
                        st,
                        &inp,
                        &wts,
                        &mut self.pools.s[li],
                        &mut self.pools.conv[li],
                        &batch,
                        &mut self.gdn_prefill_ws,
                        &mut self.mixer_out,
                    )?;
                }
                fn_glue::gated_rmsnorm_sigmoid(
                    None,
                    &self.z,
                    &m.norm,
                    &mut self.mixer_out,
                    rows,
                    gdn::VALUE_HEADS,
                    gdn::HEAD_DIM,
                    EPS,
                    st,
                )?;
                lin(&m.out_proj, &self.mixer_out, &mut self.block_out, rows, mode, &mut self.ws, &mut self.prefill_ws, st)?;
            }
            MixerWeights::Full(m) => {
                let st = &self.stream;
                lin(&m.q_proj, &self.x_mixed, &mut self.qkv, rows, mode, &mut self.ws, &mut self.prefill_ws, st)?;
                lin(&m.k_proj, &self.x_mixed, &mut self.fk, rows, mode, &mut self.ws, &mut self.prefill_ws, st)?;
                lin(&m.v_proj, &self.x_mixed, &mut self.fv, rows, mode, &mut self.ws, &mut self.prefill_ws, st)?;
                let pcfg = AttnPrepCfg {
                    heads: self.cfg.heads,
                    kv_heads: self.cfg.kv_heads,
                    head_dim: self.cfg.head_dim,
                    rotary_dim: self.cfg.rotary_dim,
                    has_gate: true,
                    eps: EPS,
                    theta: self.cfg.rope_theta as f32,
                };
                g_attn_prep(
                    &pcfg,
                    &self.qkv,
                    &self.fk,
                    &self.positions,
                    &m.q_norm,
                    &m.k_norm,
                    &self.fq,
                    &self.fgate,
                    &self.fk2,
                    rows,
                    st,
                )?;

                let n_planes = if self.qsa { self.qsa_pool.len() } else { 0 };
                let slots = self.pools.slots;
                let kv = match fi {
                    Some(i) => &mut self.kv[i],
                    None => self.mtp_kv.as_mut().ok_or_else(|| ModelError::Request("this pack has no MTP head".into()))?,
                };
                let pages = kv.n_blocks;
                attention::kv_write(&self.fk2, &self.fv, &self.rowidx, &self.table, kv, rows, st)?;
                let live = self.attn_ctx.unwrap_or_else(|| Self::live_of(b));

                // QSA: the indexer planes are kept current on every forward, and selection takes over for a row once
                // its own position is past the dense range (below it every block would be selected, so it is the same
                // function). The threshold is the ROW's position, never the batch's size or last position, so no chunk
                // grid and no window shape can move a row's arithmetic.
                let plane = m.indexer.as_ref().filter(|_| n_planes > 0).map(|_| fi.unwrap_or(n_planes - 1));
                if let (Some(ix), Some(plane)) = (m.indexer.as_ref(), plane) {
                    lin(&ix.qk_proj, &self.x_mixed, &mut self.qsa_qk, rows, mode, &mut self.ws, &mut self.prefill_ws, st)?;
                    qsa::stage(
                        &self.qsa_qk,
                        &ix.q_norm,
                        &self.positions,
                        &mut self.qsa_q,
                        &mut self.qsa_kraw,
                        rows,
                        EPS,
                        self.cfg.rope_theta as f32,
                        st,
                    )?;
                    let seqs = qsa::Seqs {
                        positions: &self.positions,
                        seq_start: &self.seq_start,
                        seq_slot: &self.seq_slot,
                        n_seq: b.n_seq(),
                        rows,
                    };
                    qsa::pool(
                        &self.qsa_kraw,
                        &self.qsa_ring[plane],
                        &ix.k_norm,
                        &seqs,
                        &self.table,
                        &mut self.qsa_pool[plane],
                        pages,
                        slots,
                        QSA_RING,
                        EPS,
                        self.cfg.rope_theta as f32,
                        st,
                    )?;
                    qsa::ring_store(&self.qsa_kraw, &mut self.qsa_ring[plane], &seqs, slots, QSA_RING, st)?;
                }
                let qrow = attention::Q_ROW_ELEMS * 2;
                let qd = qsa::N_HEADS * qsa::DK * 2;
                let nb_cap = live.div_ceil(qsa::CR);
                let is_sparse = |r: usize| plane.is_some() && b.positions[r] as usize >= QSA_DENSE_CTX;
                let mut r0 = 0;
                while r0 < rows {
                    let sp = is_sparse(r0);
                    let mut r1 = r0 + 1;
                    while r1 < rows && is_sparse(r1) == sp {
                        r1 += 1;
                    }
                    let n = r1 - r0;
                    match (sp, plane) {
                        (true, Some(plane)) => {
                            // scores and selection of this run, in slabs; both are per row and causally bounded by the
                            // row's absolute position, so the slab size is not part of the arithmetic
                            let slab = if window { n } else { QSA_SLAB.min(n) };
                            let mut s0 = r0;
                            while s0 < r1 {
                                let sn = slab.min(r1 - s0);
                                let sl = self.rowidx.slot.subview(s0 * 4, sn * 4);
                                let ps = self.rowidx.pos.subview(s0 * 4, sn * 4);
                                let rr = qsa::Rows { slot: &sl, pos: &ps, rows: sn };
                                qsa::scores(
                                    &self.qsa_q.subview(s0 * qd, sn * qd),
                                    &self.qsa_pool[plane],
                                    pages,
                                    &self.table,
                                    &rr,
                                    &mut self.qsa_scores,
                                    nb_cap,
                                    st,
                                )?;
                                let mut sel = self.qsa_sel.subview(s0 * qsa::TOPK * 4, sn * qsa::TOPK * 4);
                                let mut sel_n = self.qsa_sel_n.subview(s0 * 4, sn * 4);
                                qsa::topk(&self.qsa_scores, &rr, nb_cap, &mut sel, &mut sel_n, st)?;
                                s0 += sn;
                            }
                            let sl = self.rowidx.slot.subview(r0 * 4, n * 4);
                            let ps = self.rowidx.pos.subview(r0 * 4, n * 4);
                            let rr = qsa::Rows { slot: &sl, pos: &ps, rows: n };
                            // a prefill chunk always attends in one part, a window in the fixed 32
                            let parts = if window { qsa::WINDOW_PARTS } else { 1 };
                            let q = self.fq.subview(r0 * qrow, n * qrow);
                            let sel = self.qsa_sel.subview(r0 * qsa::TOPK * 4, n * qsa::TOPK * 4);
                            let sel_n = self.qsa_sel_n.subview(r0 * 4, n * 4);
                            let mut out = self.mixer_out.subview(r0 * qrow, n * qrow);
                            qsa::attend(&q, kv, &self.table, &rr, &sel, &sel_n, parts, &mut self.qsa_ws, &mut out, st)?;
                        }
                        _ if window => {
                            let sl = self.rowidx.slot.subview(r0 * 4, n * 4);
                            let ps = self.rowidx.pos.subview(r0 * 4, n * 4);
                            let ri = RowIndex { slot: sl, pos: ps, rows: n };
                            let q = self.fq.subview(r0 * qrow, n * qrow);
                            let mut out = self.mixer_out.subview(r0 * qrow, n * qrow);
                            attention::attn_decode(&q, kv, &self.table, &ri, n, &mut out, live, &mut self.attn_ws, st)?;
                        }
                        _ => attention::attn_prefill_rows(
                            &self.fq,
                            kv,
                            &self.table,
                            b.seq_slot[0] as usize,
                            b.positions[r0] as usize,
                            r0,
                            n,
                            &mut self.mixer_out,
                            0,
                            st,
                        )?,
                    }
                    r0 = r1;
                }
                g_gate_mul(&self.mixer_out, &self.fgate, rows * self.cfg.heads * self.cfg.head_dim, st)?;
                lin(&m.o_proj, &self.mixer_out, &mut self.block_out, rows, mode, &mut self.ws, &mut self.prefill_ws, st)?;
            }
        }

        // 3. attention hyper-connection post-inject (the streams are updated in place)
        fn_mhc::hc_post(&self.block_out, None, &self.inj_attn, &mut self.x_streams, h, hc, rows, &self.stream)?;

        // 4. MLP hyper-connection pre-mix
        self.pre_mix(&layer.mlp_hc.mixer(), false, mode, rows)?;

        // 5. router and shared-expert gate: per row on every path, so a chunk's routing never depends on its row count
        let st = &self.stream;
        g_small_linear(&self.x_mixed, &layer.moe.router, &self.router_logits, rows, h, num_experts, st)?;
        g_small_linear(&self.x_mixed, &layer.moe.shared_gate, &self.sh_gate_scalar, rows, h, 1, st)?;

        // 6. the experts. A window groups them across its rows (FrostMoE); a prefill chunk always routes on the device
        // and runs the CoopMK decode, the shared expert riding along as slot `num_experts`.
        let tables = layer.moe.tables();
        if window {
            let shape =
                fmoe::Shape { t: rows, ne: num_experts, top_k, h, i: interm, kc: fmoe::DEFAULT_KC, max_tokens: WINDOW_ROWS };
            fmoe::forward(
                &self.x_mixed,
                &self.router_logits,
                &self.sh_gate_scalar,
                &tables,
                &mut self.moe_out,
                shape,
                None,
                fmoe::SwigluLimit::default(),
                &mut self.fmoe_ws,
                st,
            )?;
        } else if let (Some(pf), Some(ex)) = (self.moe_pf.as_mut(), moe_layer.and_then(|i| self.moe_experts[i].as_ref())) {
            // The grouped MoE prefill (`REXL3_FLASH_NEXT_GROUPED_MOE_PREFILL=1` and a supported shape): the SAME router
            // kernel as CoopMK (softmax over the gate logits, top-k, no shared slot), then the routed experts through the
            // grouped GEMMs, the shared expert separately (`SharedExpert::forward_big_refs` + its sigmoid gate, fp32), and
            // one fp32 -> bf16 finish: the CoopMK slot combine's arithmetic, in one cast.
            let st = &self.stream;
            // route: picks straight into the scratch (the router's u32 ids are the i32 picks' bits), weights f32, kept
            // f32 end-to-end (the f32 combine reads them as the router wrote them; no f32 -> fp16 rounding)
            fn_moe::moe_topk_softmax(&self.router_logits, None, &mut pf.pick, &mut pf.wts_f32, num_experts, top_k, rows, st)?;
            // the shared expert: the MLP (gate/up/down, unclamped SwiGLU as CoopMK runs it), then the sigmoid of the
            // shared-gate scalar in fp32 (CoopMK rides the shared expert as the last slot with exactly that weight)
            let se = &layer.moe.shared_expert;
            let ss = self.moe_shared_scratch.as_mut().expect("grouped prefill on: shared scratch allocated");
            SharedExpert::forward_big_refs(
                &se.gate,
                &se.up,
                &se.down,
                se.gate.k,
                se.gate.n,
                0.0,
                &self.x_mixed,
                Dtype::BF16,
                rows,
                &mut self.shared_out_f32,
                ss,
                st,
            )?;
            // the routed experts on the picks in `pf` (fp32 out, f32 weights; no shared slot here)
            ex.run_prefill_f32_wts(&self.x_mixed, Dtype::BF16, rows, None, &mut self.moe_out_f32, pf, st)?;
            // finish: bf16 out = bf16(routed + sigmoid(gate) * shared), the one cast
            finish_shared(&self.moe_out_f32, &self.shared_out_f32, &self.sh_gate_scalar, &mut self.moe_out, rows, h, st)?;
        } else {
            fn_moe::moe_topk_softmax(
                &self.router_logits,
                Some(&self.sh_gate_scalar),
                &mut self.topk_indices,
                &mut self.topk_weights,
                num_experts,
                top_k,
                rows,
                st,
            )?;
            fn_moe::coopmk_decode(
                &self.x_mixed,
                &self.topk_indices,
                &self.topk_weights,
                &tables,
                &mut self.act_workspace,
                &mut self.moe_out,
                h,
                interm,
                top_k + 1,
                rows,
                st,
            )?;
        }

        // 7. MLP hyper-connection post-inject
        fn_mhc::hc_post(&self.moe_out, None, &self.inj_mlp, &mut self.x_streams, h, hc, rows, &self.stream)?;
        Ok(())
    }

    fn compute_layers(&mut self, b: &RowBatch, mode: Mode) -> Result<(), ModelError> {
        let rows = b.rows;
        let (h, hc) = (self.cfg.hidden, self.cfg.hc_streams);
        let w = Arc::clone(&self.w);

        g_embed(&w.embed, &self.ids, &self.block_out, rows, h, self.cfg.vocab, &self.stream)?;
        fn_mhc::hc_expand(&self.block_out, &mut self.x_streams, h, hc, rows, &self.stream)?;

        let (mut li, mut fi) = (0usize, 0usize);
        for (i, layer) in w.layers.iter().enumerate() {
            if let Some(p) = w.ple.as_ref().filter(|p| p.before_layer == i) {
                self.run_ple(p, b, mode)?;
            }
            let moe_layer = self.moe_pf.as_ref().map(|_| i);
            match layer.mixer {
                MixerWeights::Linear(_) => {
                    self.run_layer(layer, li, None, b, mode, moe_layer)?;
                    li += 1;
                }
                MixerWeights::Full(_) => {
                    self.run_layer(layer, 0, Some(fi), b, mode, moe_layer)?;
                    fi += 1;
                }
            }
        }
        self.collapse(&w.head_hc.mixer(), mode, rows)
    }

    /// The device work of one window: every layer, the full LM head, top-`count` per row. No host sync, no upload.
    fn window_body(&mut self, b: &RowBatch, count: usize) -> Result<(), ModelError> {
        let rows = b.rows;
        self.compute_layers(b, Mode::Window)?;
        self.w.lm_head.forward(&self.x_mixed, &mut self.logits, rows, Dtype::BF16, Dtype::BF16, &mut self.ws, &self.stream)?;
        g_topk(&self.logits, rows, self.cfg.vocab, count, &self.topk_ids, &self.topk_vals, &self.topk_ws, &self.stream)
    }

    /// The LM head and top-`count` of the last row of the chunk now in `x_mixed`, always on exactly one row.
    fn candidates_of_last_row(&mut self, rows: usize, count: usize) -> Result<Option<Candidates>, ModelError> {
        if count == 0 || count > MAX_CANDIDATES {
            return Err(ModelError::Request(format!("candidate count must be 1..={MAX_CANDIDATES}, got {count}")));
        }
        let h = self.cfg.hidden;
        let last = self.x_mixed.subview((rows - 1) * h * 2, h * 2);
        self.w.lm_head.forward(&last, &mut self.logits, 1, Dtype::BF16, Dtype::BF16, &mut self.ws, &self.stream)?;
        g_topk(&self.logits, 1, self.cfg.vocab, count, &self.topk_ids, &self.topk_vals, &self.topk_ws, &self.stream)?;
        self.sync()?;
        let mut ids = vec![0i32; count];
        let mut vals = vec![0.0f32; count];
        self.topk_ids.download(&mut ids)?;
        self.topk_vals.download(&mut vals)?;
        Ok(Some(Candidates { rows: 1, count, ids, vals }))
    }

    /// The MTP cache's row at position 0 is never produced by the head: its first row pairs trunk position 0 with the
    /// token that follows it, which sits at position 1. Attention over positions `0..=p` reads that row, so write it as
    /// zeros before the head's first batch of a sequence instead of letting a recycled page decide a draft.
    fn zero_mtp_row0(&mut self, slot: usize) -> Result<(), ModelError> {
        if self.mtp_kv.is_none() {
            return Ok(());
        }
        put(&mut self.stage, &mut self.one_row.slot, &[slot as i32], &self.stream)?;
        put(&mut self.stage, &mut self.one_row.pos, &[0i32], &self.stream)?;
        let kv = self.mtp_kv.as_mut().expect("checked just above");
        attention::kv_write(&self.zero_kv, &self.zero_kv, &self.one_row, &self.table, kv, 1, &self.stream)?;
        Ok(())
    }

    fn note_stack(&mut self, b: &RowBatch) {
        if b.n_seq() == 1 {
            self.stack_pos0 = b.positions[0] as usize;
            self.stack_rows = b.rows;
            self.stack_slot = b.seq_slot[0] as usize;
        } else {
            self.stack_rows = 0;
        }
    }

    /// The MTP head on `b` (tokens = the next tokens, positions = their positions) with input stacks read from `src`
    /// (`[rows][hc][h]` f32). Leaves the head's output stack in `mtp_streams` and the last row's draft (argmax over the
    /// first [`MTP_HEAD_N`] ids of the shared LM head) and its probability in slot `j` of the draft arrays. Device work
    /// only.
    fn mtp_body(&mut self, b: &RowBatch, src: StackSrc, mode: Mode, j: usize) -> Result<(), ModelError> {
        let rows = b.rows;
        let (h, hc) = (self.cfg.hidden, self.cfg.hc_streams);
        let w = Arc::clone(&self.w);
        let Some(mtp) = w.mtp.as_ref() else {
            return Err(ModelError::Request("this pack has no MTP head".into()));
        };
        // the head's collapse below overwrites the trunk rows in x_mixed
        self.mixed_rows = 0;
        g_embed(&w.embed, &self.ids, &self.block_out, rows, h, self.cfg.vocab, &self.stream)?;
        {
            let row = hc * h * 4;
            let stack = match src {
                StackSrc::Trunk(r) => self.x_streams.subview(r * row, rows * row),
                StackSrc::Head(r) => self.mtp_streams.subview(r * row, rows * row),
            };
            mtpk::input(
                &stack,
                &mtp.norm_hidden,
                &self.block_out,
                &mtp.norm_emb,
                &mut self.mtp_xs,
                &mut self.mtp_xe,
                rows,
                hc,
                h,
                EPS,
                &self.stream,
            )?;
        }
        lin_f16(
            &mtp.fc_hidden,
            &self.mtp_xs,
            &mut self.mtp_h,
            rows * hc,
            mode,
            &mut self.ws,
            &mut self.prefill_ws,
            &self.stream,
        )?;
        lin_f16(&mtp.fc_embedding, &self.mtp_xe, &mut self.mtp_e, rows, mode, &mut self.ws, &mut self.prefill_ws, &self.stream)?;
        mtpk::combine(&self.mtp_h, &self.mtp_e, &mut self.mtp_streams, rows, hc, h, &self.stream)?;
        // the decoder layer and the collapse run on the head's stack, swapped into x_streams
        std::mem::swap(&mut self.x_streams, &mut self.mtp_streams);
        let mix = mtp.mixer.as_ref().unwrap_or(&w.head_hc).mixer();
        let r = self.run_layer(&mtp.layer, 0, None, b, mode, None).and_then(|_| self.collapse(&mix, mode, rows));
        std::mem::swap(&mut self.x_streams, &mut self.mtp_streams);
        r?;
        // The draft reads only the first MTP_HEAD_N columns; the model's exact draft-head copy evaluates exactly those
        // 128-column strips, so it is the same function as the full head for the columns `pick` reads (the full head is
        // still the one used for target verification). The strip copy is a device copy of the same packed weights, so the
        // retained logits are bit-identical.
        let nv = draft_head_nv(self.cfg.vocab);
        let last = self.x_mixed.subview((rows - 1) * h * 2, h * 2);
        let head = self.draft_lm_head.as_ref().unwrap_or(&self.w.lm_head);
        head.forward(&last, &mut self.logits, 1, Dtype::BF16, Dtype::BF16, &mut self.ws, &self.stream)?;
        mtpk::pick(&self.logits, nv, &mut self.drafts_dev, &mut self.conf_dev, j, &self.stream)?;
        Ok(())
    }

    /// The MTP head on a prefill chunk (`mtp_after_prefill`): populates only the persistent MTP KV cache and QSA
    /// indexer state, skipping the stateless attention output, MoE, stream collapse, and LM head that
    /// `mtp_propose` does not read (its chain re-derives the head's row states from the trunk rows still on the
    /// stack). Falls back to the full `mtp_body` when the head's mixer is not the full attention one the skip is
    /// written for.
    fn mtp_prefill_body(&mut self, b: &RowBatch, src: StackSrc, mode: Mode) -> Result<(), ModelError> {
        if !matches!(self.w.mtp.as_ref().map(|m| &m.layer.mixer), Some(MixerWeights::Full(_))) {
            return self.mtp_body(b, src, mode, 0);
        }
        let rows = b.rows;
        let (h, hc) = (self.cfg.hidden, self.cfg.hc_streams);
        let w = Arc::clone(&self.w);
        let Some(mtp) = w.mtp.as_ref() else {
            return Err(ModelError::Request("this pack has no MTP head".into()));
        };
        self.mixed_rows = 0;
        g_embed(&w.embed, &self.ids, &self.block_out, rows, h, self.cfg.vocab, &self.stream)?;
        {
            let row = hc * h * 4;
            let stack = match src {
                StackSrc::Trunk(r) => self.x_streams.subview(r * row, rows * row),
                StackSrc::Head(r) => self.mtp_streams.subview(r * row, rows * row),
            };
            mtpk::input(
                &stack,
                &mtp.norm_hidden,
                &self.block_out,
                &mtp.norm_emb,
                &mut self.mtp_xs,
                &mut self.mtp_xe,
                rows,
                hc,
                h,
                EPS,
                &self.stream,
            )?;
        }
        lin_f16(
            &mtp.fc_hidden,
            &self.mtp_xs,
            &mut self.mtp_h,
            rows * hc,
            mode,
            &mut self.ws,
            &mut self.prefill_ws,
            &self.stream,
        )?;
        lin_f16(&mtp.fc_embedding, &self.mtp_xe, &mut self.mtp_e, rows, mode, &mut self.ws, &mut self.prefill_ws, &self.stream)?;
        mtpk::combine(&self.mtp_h, &self.mtp_e, &mut self.mtp_streams, rows, hc, h, &self.stream)?;
        std::mem::swap(&mut self.x_streams, &mut self.mtp_streams);
        let r = (|| -> Result<(), ModelError> {
            self.pre_mix(&mtp.layer.attn_hc.mixer(), true, mode, rows)?;
            let MixerWeights::Full(m) = &mtp.layer.mixer else { unreachable!() };
            let st = &self.stream;
            lin(&m.k_proj, &self.x_mixed, &mut self.fk, rows, mode, &mut self.ws, &mut self.prefill_ws, st)?;
            lin(&m.v_proj, &self.x_mixed, &mut self.fv, rows, mode, &mut self.ws, &mut self.prefill_ws, st)?;
            let pcfg = AttnPrepCfg {
                heads: self.cfg.heads,
                kv_heads: self.cfg.kv_heads,
                head_dim: self.cfg.head_dim,
                rotary_dim: self.cfg.rotary_dim,
                has_gate: true,
                eps: EPS,
                theta: self.cfg.rope_theta as f32,
            };
            g_attn_prep(
                &pcfg,
                &self.qkv,
                &self.fk,
                &self.positions,
                &m.q_norm,
                &m.k_norm,
                &self.fq,
                &self.fgate,
                &self.fk2,
                rows,
                st,
            )?;
            let kv = self.mtp_kv.as_mut().ok_or_else(|| ModelError::Request("this pack has no MTP head".into()))?;
            let pages = kv.n_blocks;
            attention::kv_write(&self.fk2, &self.fv, &self.rowidx, &self.table, kv, rows, st)?;
            let n_planes = if self.qsa { self.qsa_pool.len() } else { 0 };
            let slots = self.pools.slots;
            let plane = m.indexer.as_ref().filter(|_| n_planes > 0).map(|_| n_planes - 1);
            if let (Some(ix), Some(plane)) = (m.indexer.as_ref(), plane) {
                lin(&ix.qk_proj, &self.x_mixed, &mut self.qsa_qk, rows, mode, &mut self.ws, &mut self.prefill_ws, st)?;
                qsa::stage(
                    &self.qsa_qk,
                    &ix.q_norm,
                    &self.positions,
                    &mut self.qsa_q,
                    &mut self.qsa_kraw,
                    rows,
                    EPS,
                    self.cfg.rope_theta as f32,
                    st,
                )?;
                let seqs = qsa::Seqs {
                    positions: &self.positions,
                    seq_start: &self.seq_start,
                    seq_slot: &self.seq_slot,
                    n_seq: b.n_seq(),
                    rows,
                };
                qsa::pool(
                    &self.qsa_kraw,
                    &self.qsa_ring[plane],
                    &ix.k_norm,
                    &seqs,
                    &self.table,
                    &mut self.qsa_pool[plane],
                    pages,
                    slots,
                    QSA_RING,
                    EPS,
                    self.cfg.rope_theta as f32,
                    st,
                )?;
                qsa::ring_store(&self.qsa_kraw, &mut self.qsa_ring[plane], &seqs, slots, QSA_RING, st)?;
            }
            Ok(())
        })();
        std::mem::swap(&mut self.x_streams, &mut self.mtp_streams);
        r
    }

    /// Device work of a draft chain: the catch-up rows (draft 0 from the last of them), then `k - 1` one-row steps whose
    /// token and position are staged on the device from the previous step.
    fn mtp_chain(&mut self, b: &RowBatch, src_row: usize, k: usize) -> Result<(), ModelError> {
        self.mtp_body(b, StackSrc::Trunk(src_row), Mode::Window, 0)?;
        let mut prev_rows = b.rows;
        let slot = b.seq_slot[0] as usize;
        let p0 = b.positions[b.rows - 1] as usize;
        for j in 1..k {
            let io = mtpk::StepInputs {
                ids: &mut self.ids,
                positions: &mut self.positions,
                row_pos: &mut self.rowidx.pos,
                seq_start: &mut self.seq_start,
            };
            mtpk::stage(io, &self.drafts_dev, &self.chain_pos, j, &self.stream)?;
            // the host-side view of the step: only its shape and position are read on the host
            let b1 = RowBatch::from_requests(&[RowRequest { slot, first_position: (p0 + j) as u64, tokens: vec![0] }]);
            self.mtp_body(&b1, StackSrc::Head(prev_rows - 1), Mode::Window, j)?;
            prev_rows = 1;
        }
        Ok(())
    }

    /// Snapshot the load-time kernel branches in a graph key. Layer order, weight shapes and indexer placement are
    /// immutable for this model, and its graph caches are never shared with another model instance.
    fn graph_features(&self) -> GraphFeatures {
        let indexed = |layer: &FlashNextLayer| match &layer.mixer {
            MixerWeights::Full(m) => m.indexer.is_some(),
            MixerWeights::Linear(_) => false,
        };
        GraphFeatures {
            qsa_trunk: self.qsa && self.w.layers.iter().any(indexed),
            qsa_mtp: self.qsa && self.w.mtp.as_ref().is_some_and(|m| indexed(&m.layer)),
            ple: self.w.ple.is_some(),
            mtp: self.has_mtp(),
            fast: self.fast,
        }
    }

    fn window_graphs(&mut self) -> &mut HashMap<WindowKey, Graph> {
        &mut self.graphs
    }

    fn chain_graphs(&mut self) -> &mut HashMap<ChainKey, Graph> {
        &mut self.mtp_graphs
    }

    /// Run `body` for a window of live context `live` under the graph recorded for `key`: captured on first use, then
    /// replayed. The key's attention grid must cover the window (a graph sized for less context would read a stale grid),
    /// and the grid is what attention launches with while `body` runs, graphs on or off.
    fn run_graph<K: GraphKey>(
        &mut self,
        graphs: fn(&mut Self) -> &mut HashMap<K, Graph>,
        key: K,
        live: usize,
        body: impl Fn(&mut Self) -> Result<(), ModelError>,
    ) -> Result<(), ModelError> {
        let grid = key.grid();
        if !grid.covers(live) {
            return Err(ModelError::Request(format!(
                "graph {key:?} covers {} tokens of context, the window needs {live}",
                grid.ctx()
            )));
        }
        self.attn_ctx = Some(grid.ctx());
        let r = if self.use_graphs {
            let mut map = std::mem::take(graphs(self));
            let r = self.capture_and_launch(&mut map, key, body);
            *graphs(self) = map;
            r
        } else {
            body(self)
        };
        self.attn_ctx = None;
        r
    }

    // not `entry()`: a capture the runtime refuses must leave the map untouched and run the work eagerly instead
    #[allow(clippy::map_entry)]
    fn capture_and_launch<K: GraphKey>(
        &mut self,
        map: &mut HashMap<K, Graph>,
        key: K,
        body: impl Fn(&mut Self) -> Result<(), ModelError>,
    ) -> Result<(), ModelError> {
        if !map.contains_key(&key) {
            let cap = match Graph::begin(&self.stream) {
                Ok(c) => c,
                Err(e) if e.is_fatal() => return Err(ModelError::Cuda(e)),
                Err(e) => {
                    eprintln!("rexl3 level=warn graph capture refused ({e}): running Flash-Next windows eagerly");
                    self.use_graphs = false;
                    return body(self);
                }
            };
            let ran = body(self);
            match cap.finish(ran) {
                Ok(g) => {
                    map.insert(key, g);
                }
                Err(CaptureError::Closure(e)) => return Err(e),
                Err(e) => {
                    if let Some(f) = e.fatal() {
                        return Err(ModelError::Cuda(gb_cuda::CudaError(f.0)));
                    }
                    eprintln!("rexl3 level=warn {e}: running Flash-Next windows eagerly");
                    self.use_graphs = false;
                    return body(self);
                }
            }
        }
        map[&key].launch(&self.stream)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The model's stream is non-blocking, so a write on the legacy default stream (`DevBuf::upload`, `BlockTable::upload`,
    /// `RowIndex::upload`: a pageable `cudaMemcpy` that can return before its DMA lands; `DevBuf::zeroed`: an asynchronous
    /// `cudaMemset`) is not ordered before the kernels that read it. Every per-step input must go through `put` / `put_table`
    /// (queued on the model's stream from bytes the stage keeps), buffers are made only while the model is built, the build
    /// settles their zeroing before the first step, and every stream sync also releases the stage.
    #[test]
    fn every_per_step_upload_is_queued_on_the_models_stream() {
        let code = include_str!("model.rs").split("#[cfg(test)]").next().unwrap();
        let code_of = |line: &'static str| line.split("//").next().unwrap();
        // the receiver of every plain upload: only the stage, and the legacy arms of the test hook inside `put` / `put_table`
        let call = [".up", "load("].concat();
        let mut receivers: Vec<(usize, String)> = Vec::new();
        for (i, line) in code.lines().enumerate() {
            let c = code_of(line);
            if let Some(at) = c.find(&call) {
                let recv = c[..at].rsplit(|ch: char| !(ch.is_alphanumeric() || ch == '_' || ch == '.')).next().unwrap();
                receivers.push((i + 1, recv.to_string()));
            }
        }
        let bad: Vec<_> = receivers.iter().filter(|(_, r)| !matches!(r.as_str(), "stage" | "dst" | "table")).collect();
        assert!(bad.is_empty(), "a model buffer is written on the legacy default stream (line, receiver): {bad:?}");
        assert_eq!(receivers.len(), 5, "put: stage + two legacy arms; put_table: two legacy arms; got {receivers:?}");
        // no buffer is made (zeroed or uploaded on the legacy stream) after the model is built
        let built = code.find("    pub fn vocab_size(").expect("the first method after `new`");
        for pat in ["zeroed(", "zeroed_view(", "from_slice(", "from_host("] {
            if let Some(at) = code[built..].find(pat) {
                let line = code[..built + at].lines().count();
                panic!("`{pat}` at line {line}: a buffer made after the build is written on the legacy default stream");
            }
        }
        // `new` settles the legacy-stream work of the build before it returns
        let new = &code[code.find("    pub fn new(cfg: FlashNextConfig").expect("FlashNextModel::new")..built];
        let tail = &new[new.rfind("m.bytes_scratch = m.scratch_bytes();").expect("the end of new")..];
        assert!(tail.contains("gb_cuda::rt::device_sync()?;"), "`new` returns before its legacy-stream zeroing has landed");
        // the stage's buffers are released only after the stream they were queued on has been waited for
        assert!(!code.contains(&["self.stream", ".sync()"].concat()), "a stream sync that does not release the stage");
        // and the engine's mode is the ordered one (only the GPU race test switches it)
        assert_eq!(upload_mode(), UploadMode::Ordered);
    }

    #[test]
    fn draft_head_nv_is_the_capped_pick_domain() {
        assert_eq!(draft_head_nv(248_320), MTP_HEAD_N);
        assert_eq!(draft_head_nv(MTP_HEAD_N), MTP_HEAD_N);
        assert_eq!(draft_head_nv(10_000), 10_000, "smaller vocabs are the whole domain");
        assert_eq!(draft_head_nv(0), 0);
    }

    #[test]
    fn draft_head_shape_checks_fail_closed() {
        // aligned width >= 8192, and the checkpoint head must cover the candidate domain
        assert!(check_draft_head_shapes(MTP_HEAD_N, 248_320).is_ok());
        assert!(check_draft_head_shapes(8192, 8192).is_ok());
        assert!(check_draft_head_shapes(8191, 248_320).is_err(), "< 8192 refused");
        assert!(check_draft_head_shapes(8192 + 64, 248_320).is_err(), "unaligned refused");
        assert!(check_draft_head_shapes(MTP_HEAD_N, 65535).is_err(), "head narrower than the domain refused");
    }

    #[test]
    fn draft_head_prefix_bytes_match_the_strip_copy_layout() {
        // The real Qwen3.8-Flash-Next LM head: k=2560, 5-bit EXL3, nv=65536 (MTP_HEAD_N).
        let (k, nv, bits) = (2560usize, MTP_HEAD_N, 5usize);
        let bytes = draft_head_prefix_bytes(k, nv, bits);
        // stored trellis [K/16, N/16, 16*bits] int16 + fp16 suh [K] + svh [N]
        let trellis = (k / 16) * (nv / 128) * 64 * bits * 4;
        assert_eq!(bytes, trellis + k * 2 + nv * 2);
        assert_eq!(bytes, 104_993_792, "the plan's +100.13 MiB for the 262K contract");
        let mib = bytes as f64 / (1024.0 * 1024.0);
        assert!((mib - 100.13).abs() < 0.01, "the +100.13 MiB the experiment measured (got {mib})");
    }

    #[test]
    fn pinned_prefill_boot_checks_every_trunk_and_mtp_row_shape() {
        let trunk = [(128, 256)];
        let mtp_hidden = [(512, 512)];
        let mtp_embedding = [(512, 128)];
        let shapes = pinned_prefill_shapes(&trunk, &mtp_hidden, &mtp_embedding, 2, 8);
        for (m, k, n, dtype) in [
            (1, 128, 256, Dtype::BF16),
            (8, 128, 256, Dtype::BF16),
            (7, 128, 256, Dtype::BF16),
            (2, 512, 512, Dtype::F16),
            (16, 512, 512, Dtype::F16),
            (14, 512, 512, Dtype::F16),
            (1, 512, 128, Dtype::F16),
            (8, 512, 128, Dtype::F16),
            (7, 512, 128, Dtype::F16),
        ] {
            assert!(shapes.contains(&(m, k, n, dtype)), "missing {m} x {k} x {n} {dtype:?}");
        }
        assert_eq!(shapes.len(), 8 + 8 + 8, "one check per reachable m and unique (k,n,dtype)");
    }

    #[test]
    fn pinned_prefill_boot_seam_initializes_then_checks_each_shape_fail_closed() {
        let shapes = [(1, 128, 256, Dtype::BF16), (8, 128, 256, Dtype::BF16), (7, 128, 256, Dtype::BF16)];
        use std::cell::RefCell;
        let calls = RefCell::new(Vec::new());
        let result = run_pinned_prefill_boot(
            &shapes,
            || {
                calls.borrow_mut().push((0, 0, 0, Dtype::BF16));
                Ok(())
            },
            |m, k, n, dtype| {
                calls.borrow_mut().push((m, k, n, dtype));
                if m == 7 {
                    Err("unsupported tail".to_owned())
                } else {
                    Ok(())
                }
            },
        );
        assert!(result.unwrap_err().contains("unsupported tail"));
        let calls = calls.into_inner();
        assert_eq!(calls[0], (0, 0, 0, Dtype::BF16), "initialization must precede validation");
        assert_eq!(&calls[1..], &shapes[..]);
    }

    #[test]
    fn attention_grid_buckets_cover_their_window_and_nothing_larger() {
        // power-of-two buckets of 512-token chunks, capped at the context and never below the live window
        assert_eq!(AttnGrid::cover(1, 8192).ctx(), 512);
        assert_eq!(AttnGrid::cover(512, 8192).ctx(), 512);
        assert_eq!(AttnGrid::cover(513, 8192).ctx(), 1024);
        assert_eq!(AttnGrid::cover(1537, 8192).ctx(), 2048);
        assert_eq!(AttnGrid::cover(5000, 6000).ctx(), 6000);
        assert_eq!(AttnGrid::cover(7000, 6000).ctx(), 7000);
        let g = AttnGrid::cover(700, 8192);
        assert!(g.covers(700) && g.covers(1024) && !g.covers(1025));
        // the grid is part of the key: the same window shape in another bucket is another graph
        let features = GraphFeatures { qsa_trunk: true, qsa_mtp: true, ple: true, mtp: true, fast: true };
        let k = |live| {
            let b = RowBatch::from_requests(&[RowRequest { slot: 0, first_position: 100, tokens: vec![1; 4] }]);
            WindowKey::for_batch(&b, 1, AttnGrid::cover(live, 8192), features)
        };
        assert_eq!(k(600), k(1000));
        assert_ne!(k(1000), k(1100));
    }

    #[test]
    fn window_keys_separate_qsa_splits_inside_one_grid_bucket() {
        let f = GraphFeatures { qsa_trunk: true, qsa_mtp: true, ple: true, mtp: true, fast: true };
        let batch = |first| RowBatch::from_requests(&[RowRequest { slot: 0, first_position: first, tokens: vec![1; 4] }]);
        let key = |first| {
            let b = batch(first);
            WindowKey::for_batch(&b, 1, AttnGrid::cover(first as usize + 4, 8192), f)
        };
        assert_ne!(key(2046), key(2048), "dense/dense/sparse/sparse and all sparse share a grid");
        assert_eq!(key(2046), key(2046));
        assert_eq!(key(2048), key(2200), "positions inside one stable sparse pattern reuse a graph");
    }

    #[test]
    fn graph_keys_cover_host_dispatch_decisions() {
        let f = GraphFeatures { qsa_trunk: true, qsa_mtp: true, ple: true, mtp: true, fast: true };
        let b = |first, sizes: &[usize]| {
            RowBatch::from_requests(
                &sizes
                    .iter()
                    .enumerate()
                    .map(|(slot, &n)| RowRequest { slot, first_position: first, tokens: vec![1; n] })
                    .collect::<Vec<_>>(),
            )
        };
        let g = AttnGrid::cover(2054, 8192);
        let base = WindowKey::for_batch(&b(2046, &[2, 2]), 2, g, f);
        let cases = [
            ("QSA row run mask", WindowKey::for_batch(&b(2048, &[2, 2]), 2, g, f)),
            ("row count", WindowKey::for_batch(&b(2046, &[2, 2, 1]), 2, g, f)),
            ("sequence count", WindowKey::for_batch(&b(2046, &[4]), 2, g, f)),
            ("rows per sequence / boundaries", WindowKey::for_batch(&b(2046, &[1, 3]), 2, g, f)),
            ("candidate count", WindowKey::for_batch(&b(2046, &[2, 2]), 3, g, f)),
            ("attention grid", WindowKey::for_batch(&b(2046, &[2, 2]), 2, AttnGrid::cover(4100, 8192), f)),
            ("QSA indexer", WindowKey::for_batch(&b(2046, &[2, 2]), 2, g, GraphFeatures { qsa_trunk: false, ..f })),
            ("PLE stage", WindowKey::for_batch(&b(2046, &[2, 2]), 2, g, GraphFeatures { ple: false, ..f })),
            ("MTP presence", WindowKey::for_batch(&b(2046, &[2, 2]), 2, g, GraphFeatures { mtp: false, ..f })),
            ("fast HC dispatch", WindowKey::for_batch(&b(2046, &[2, 2]), 2, g, GraphFeatures { fast: false, ..f })),
        ];
        for (branch, key) in cases {
            assert_ne!(base, key, "{branch}");
        }

        let catchup = b(2046, &[4]);
        let chain =
            |first, rows, src, drafts, grid, features| ChainKey::for_chain(&b(first, &[rows]), src, drafts, grid, features);
        let base = ChainKey::for_chain(&catchup, 0, 3, g, f);
        let cases = [
            ("catch-up QSA split", chain(2048, 4, 0, 3, g, f)),
            ("step QSA split", chain(2045, 4, 0, 3, g, f)),
            ("catch-up rows", chain(2046, 3, 0, 3, g, f)),
            ("source trunk row", chain(2046, 4, 1, 3, g, f)),
            ("draft count", chain(2046, 4, 0, 2, g, f)),
            ("attention grid", chain(2046, 4, 0, 3, AttnGrid::cover(4100, 8192), f)),
            ("MTP QSA indexer", chain(2046, 4, 0, 3, g, GraphFeatures { qsa_mtp: false, ..f })),
            ("MTP presence", chain(2046, 4, 0, 3, g, GraphFeatures { mtp: false, ..f })),
            ("PLE stage", chain(2046, 4, 0, 3, g, GraphFeatures { ple: false, ..f })),
            ("fast HC dispatch", chain(2046, 4, 0, 3, g, GraphFeatures { fast: false, ..f })),
        ];
        for (branch, key) in cases {
            assert_ne!(base, key, "{branch}");
        }
        assert_eq!(base, ChainKey::for_chain(&catchup, 0, 3, g, f));
        let before = chain(2043, 4, 0, 3, g, f);
        let across = chain(2044, 4, 0, 3, g, f);
        assert_eq!(before.sparse_catchup, across.sparse_catchup, "both catch-up runs are dense");
        assert_ne!(before.sparse_steps, across.sparse_steps, "the second draft crosses the QSA threshold");
        assert_ne!(before, across);
    }

    /// Real-pack gate: no model load or CUDA context on the CPU test path. Run only on an idle GPU after the pause.
    #[test]
    fn real_pack_graph_replay_equals_eager_across_qsa_threshold() {
        use std::path::PathBuf;
        if std::env::var("REXL3_FN_REAL").ok().as_deref() != Some("1") {
            eprintln!("not run: REXL3_FN_REAL=1 and REXL3_FLASH_NEXT_DIR are required for graph parity");
            return;
        }
        let dir = PathBuf::from(std::env::var("REXL3_FLASH_NEXT_DIR").expect("set the real-pack directory"));
        assert!(dir.join("config.json").exists(), "missing real-pack config.json");
        assert_ne!(std::env::var("REXL3_QSA").ok().as_deref(), Some("0"), "graph threshold parity requires QSA");
        assert!(gb_cuda::device_available(), "real-pack graph parity needs an idle CUDA device");
        let opts = ModelOptions { slots: 1, kv_pages: 64, max_context: 8192, prefill_rows: 1024, warm: false };
        let mut m = FlashNextModel::load(&dir, opts).unwrap();
        assert!(m.graph_features().qsa_trunk && m.graph_features().qsa_mtp, "pack needs both QSA indexers");
        assert!(m.has_mtp(), "pack needs the MTP head");

        // Each run begins from the same empty slot. 2046/2047 straddle the row-2048 QSA switch
        // inside a window and between catch-up and chained MTP steps; 2051 is sparse throughout.
        let run = |m: &mut FlashNextModel, n: usize| {
            m.begin_slot(0).unwrap();
            let vocab = m.vocab_size() as u64;
            let prompt: Vec<u32> = (0..n).map(|i| ((i as u64 * 2_654_435_761) % (vocab - 16) + 8) as u32).collect();
            let path = PrefillPath::for_prompt(n);
            let mut last = None;
            for first in (0..n).step_by(m.opts.prefill_rows) {
                let end = (first + m.opts.prefill_rows).min(n);
                let chunk = &prompt[first..end];
                let b = RowBatch::from_requests(&[RowRequest { slot: 0, first_position: first as u64, tokens: chunk.to_vec() }]);
                last = m.prefill_chunk(&b, path, (end == n).then_some(1)).unwrap();
                m.mtp_after_prefill(0, first, chunk, prompt.get(end).copied(), path).unwrap();
            }
            let next = last.unwrap().ids[0] as u32;
            let mut history = prompt;
            history.push(next);
            let drafts = m.mtp_propose(0, &history, 4).unwrap();
            assert!(!drafts.is_empty());
            let window = |m: &mut FlashNextModel, first, tokens: Vec<u32>| {
                let b = RowBatch::from_requests(&[RowRequest { slot: 0, first_position: first as u64, tokens }]);
                let out = m.window_candidates(&b, 4).unwrap();
                m.commit(0, b.rows);
                m.flush().unwrap();
                (out.ids, out.vals.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
            };
            let crossing = window(m, n, vec![next; 5]);
            let sparse = window(m, n + 5, vec![next; 4]);
            (drafts, crossing, sparse)
        };

        m.use_graphs = false;
        let eager: Vec<_> = [2046, 2047, 2051].map(|n| run(&mut m, n)).into();
        m.use_graphs = true;
        let captured: Vec<_> = [2046, 2047, 2051].map(|n| run(&mut m, n)).into();
        assert!(m.use_graphs && !m.graphs.is_empty() && !m.mtp_graphs.is_empty(), "capture refused; parity would be vacuous");
        assert_eq!(captured, eager, "capture differs bit for bit from eager");
        let sizes = (m.graphs.len(), m.mtp_graphs.len());
        let replayed: Vec<_> = [2046, 2047, 2051].map(|n| run(&mut m, n)).into();
        assert_eq!((m.graphs.len(), m.mtp_graphs.len()), sizes, "expected replay, not new captures");
        assert_eq!(replayed, eager, "replayed windows / MTP chains differ bit for bit from eager");
    }

    /// The one switch a prefill chunk is allowed to make, and the row counts that must not change it.
    #[test]
    fn the_prefill_path_follows_the_prompt_total_and_not_the_chunk() {
        assert_eq!(PrefillPath::GEMV_MAX_PROMPT, MAX_DECODE_ROWS);
        for total in [1, 2, 64, 127, 128] {
            assert_eq!(PrefillPath::for_prompt(total), PrefillPath::Gemv, "total {total}");
        }
        for total in [129, 256, 1000, 4096] {
            assert_eq!(PrefillPath::for_prompt(total), PrefillPath::Tiled, "total {total}");
        }
        // a 1-row tail of a long prompt is Tiled, like every other chunk of it
        assert_eq!(PrefillPath::for_prompt(4096), PrefillPath::for_prompt(129));
    }

    // ---- grouped MoE prefill switch (REXL3_FLASH_NEXT_GROUPED_MOE_PREFILL) ----

    #[test]
    fn the_grouped_moe_switch_is_on_by_default_and_only_zero_turns_it_off() {
        // unset means on: the shipped default. Only an explicit "0" opts out (debugging).
        std::env::remove_var("REXL3_FLASH_NEXT_GROUPED_MOE_PREFILL");
        assert!(grouped_moe_switch());
        std::env::set_var("REXL3_FLASH_NEXT_GROUPED_MOE_PREFILL", "0");
        assert!(!grouped_moe_switch());
        std::env::set_var("REXL3_FLASH_NEXT_GROUPED_MOE_PREFILL", "1");
        assert!(grouped_moe_switch());
        std::env::set_var("REXL3_FLASH_NEXT_GROUPED_MOE_PREFILL", "on");
        assert!(grouped_moe_switch());
        std::env::set_var("REXL3_FLASH_NEXT_GROUPED_MOE_PREFILL", "true");
        assert!(grouped_moe_switch());
        std::env::remove_var("REXL3_FLASH_NEXT_GROUPED_MOE_PREFILL");
    }

    // ---- hcg pre-mix: the declared prefill contract, no switch ----

    #[test]
    fn the_hcg_pre_mix_is_unconditional_on_the_family_shapes() {
        // The compensated-fp16 hcg pre-mix is the declared prefill contract: it engages for every qualifying
        // shape (H % 4 == 0, hc >= 1, rank >= 1 — the exact Flash-Next dims qualify). There is no switch.
        assert!(hcg_applies(2560, 4, 320)); // the family's exact dims
        assert!(hcg_applies(2560, 2, 64));
        assert!(!hcg_applies(2561, 4, 320)); // H % 4 != 0
        assert!(!hcg_applies(2560, 0, 320)); // no streams
        assert!(!hcg_applies(2560, 4, 0)); // no rank
    }

    #[test]
    fn the_hcg_pre_mix_only_engages_in_tiled_prefill() {
        // the contract applies for the process but is exercised only in the tiled prefill chunk; decode and
        // GEMV prefill always keep the fp32 `hcx::pre` reference
        assert!(!hcg_for_mode(true, Mode::Window));
        assert!(!hcg_for_mode(true, Mode::Prefill(PrefillPath::Gemv)));
        assert!(hcg_for_mode(true, Mode::Prefill(PrefillPath::Tiled)));
    }

    #[test]
    fn the_hcg_env_switch_was_removed() {
        // Regression guard: the HCG pre-mix switch env name (\x52EXL3_FLASH_NEXT_HCG_PREFILL) was deleted
        // (the pre-mix is unconditional). No reader
        // may come back in the model source and no registry entry may be re-added (check_switches.py --ci also
        // flags drift). The env name is spelled with a hex-escaped leading character so check_switches' code
        // scan (which flags any REXL3_* literal in crates/src) does not mistake the guard for a live reader.
        let hcg_switch = "\x52EXL3_FLASH_NEXT_HCG_PREFILL";
        let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/families/qwen_flash_next/model.rs")).unwrap();
        assert!(!src.contains(hcg_switch));
        let registry = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/switch_registry.json")).unwrap();
        assert!(!registry.contains(hcg_switch));
    }

    // ---- stateless MTP prefill skip: automatic, no switch (the env opt-ins for the skip and the
    // parallel plan builder were removed; both are the only paths now) ----

    #[test]
    fn removed_stateless_and_planner_switches_have_no_readers() {
        // Regression guard: the two switches were deleted. No reader may come back in the model source and
        // no registry entry may be re-added (check_switches.py --ci also flags drift). The automatic paths are
        // pinned: the stateless skip body is called unconditionally from `mtp_after_prefill` (the only MTP-head
        // body a prefill chunk runs; `mtp_chain` and decode windows always keep the full `mtp_body`), and the
        // parallel plan builder is the only builder (`use_parallel_plan(true)` at load).
        // The env names are spelled with a hex-escaped leading character so check_switches' code scan (which
        // flags any REXL3_* literal in crates/src) does not mistake the guard for a live reader.
        let stateless_switch = "\x52EXL3_FLASH_NEXT_STATELESS_MTP_PREFILL";
        let planner_switch = "\x52EXL3_FLASH_NEXT_MOE_PLAN_PARALLEL";
        let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/families/qwen_flash_next/model.rs")).unwrap();
        assert!(!src.contains(stateless_switch));
        assert!(!src.contains(planner_switch));
        assert!(src.contains("self.mtp_prefill_body(&b, StackSrc::Trunk(0), mode)?;"));
        assert!(src.contains("p.use_parallel_plan(true);"));
        let registry = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/switch_registry.json")).unwrap();
        assert!(!registry.contains(stateless_switch));
        assert!(!registry.contains(planner_switch));
    }

    #[test]
    fn flash_next_moe_dims_are_exact_for_the_family_and_none_otherwise() {
        // the exact Flash-Next shapes: hidden 2560, inter 640, 512 experts, top-k 10
        let d = flash_next_moe_dims(2560, 640, 512, 10).unwrap();
        assert_eq!(d.hidden, 2560);
        assert_eq!(d.inter, 640);
        assert_eq!(d.experts, 512);
        assert_eq!(d.topk, 10);
        // scale/limit are the CoopMK path's, verified against the pack config (no `routed_scaling_factor` key:
        // Qwen3 routes with softmax over the gate logits + top-k + renormalisation, the weights are final as the
        // `moe_topk_softmax` kernel writes them) and against `coopmk_decode` (it takes no `limit` argument: its
        // SwiGLU is unclamped, so `limit <= 0` disables the grouped kernels' clamps).
        assert_eq!(d.scale, 1.0);
        assert_eq!(d.limit, 0.0);
        // unsupported shapes fall back (None) rather than mis-route
        assert!(flash_next_moe_dims(2561, 640, 512, 10).is_none()); // hidden % 128 != 0
        assert!(flash_next_moe_dims(2560, 641, 512, 10).is_none()); // inter % 128 != 0
        assert!(flash_next_moe_dims(2560, 640, 0, 10).is_none()); // no experts
        assert!(flash_next_moe_dims(2560, 640, 512, 0).is_none()); // topk 0
        assert!(flash_next_moe_dims(2560, 640, 512, 32).is_none()); // topk > 16
        assert!(flash_next_moe_dims(2560, 640, 16, 17).is_none()); // topk > experts
    }

    #[test]
    fn grouped_prefill_scratch_sizes_match_the_layout_for_the_exact_dims() {
        let d = flash_next_moe_dims(2560, 640, 512, 10).unwrap();
        for rows in [1, 128, 1024, 4096] {
            let sum: usize = PrefillScratch::items_for(&d, rows).unwrap().iter().map(|(_, b)| b).sum();
            assert_eq!(PrefillScratch::bytes_for(&d, rows).unwrap(), sum, "bytes_for == items sum at {rows} rows");
            // every item is listed exactly once and the buffers are 4-byte multiples
            let names: Vec<&str> = PrefillScratch::items_for(&d, rows).unwrap().iter().map(|(n, _)| *n).collect();
            let unique: std::collections::BTreeSet<&str> = names.iter().copied().collect();
            assert_eq!(unique.len(), names.len(), "duplicate item name");
            assert!(PrefillScratch::items_for(&d, rows).unwrap().iter().all(|(_, b)| b % 4 == 0));
        }
    }

    #[test]
    fn the_grouped_route_representation_is_the_coopmk_route_without_the_shared_slot() {
        // The grouped path routes with the same kernel as CoopMK (`moe_topk_softmax`) but without the shared-gate slot
        // (the shared expert runs separately), so its output is `rows * top_k` entries and the scratch's arrays hold:
        // - picks: the router's u32 ids as the i32 bits (expert ids < 512 < 2^31), slot order preserved;
        // - weights: the router's f32 values, kept f32 end-to-end in `wts_f32` (the f32 combine reads them as written;
        //   no f32 -> fp16 rounding) — the fp16 `wts` buffer exists for the other families' path.
        use gb_exl3::f16::f64_to_f16;
        let e = 512usize;
        let ids: Vec<u32> = (0..16).map(|r| ((r * 37 + 11) % e) as u32).collect();
        let wts_f32: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.1) + 0.05).collect();
        let pairs: Vec<(i32, u16)> = ids.iter().zip(&wts_f32).map(|(&id, &w)| (id as i32, f64_to_f16(w as f64))).collect();
        for (k, &(pick, w16)) in pairs.iter().enumerate() {
            assert!(pick >= 0 && (pick as usize) < e, "slot {k}: pick {pick} inside [0, E)");
            assert_eq!(pick as u32, ids[k], "slot {k}: the pick's bits are the router's id bits");
            assert_eq!(w16, f64_to_f16(wts_f32[k] as f64), "slot {k}: the fp16 rounding the f32 variant avoids");
        }
        // slot order is preserved: the pairs are the router's in order
        assert_eq!(pairs[0], (11, f64_to_f16(0.05)));
        assert_eq!(pairs[15], ((15 * 37 + 11) % e as i32, f64_to_f16(1.55)));
        // the scratch prices the f32 weights twice the fp16 slot-count bytes, in the order the router writes them
        let d = flash_next_moe_dims(2560, 640, 512, 10).unwrap();
        for rows in [1usize, 257, 4096] {
            let items = PrefillScratch::items_for(&d, rows).unwrap();
            let get = |n: &str| items.iter().find(|x| x.0 == n).map(|x| x.1).unwrap_or(0);
            assert_eq!(get("wts"), rows * 10 * 2);
            assert_eq!(get("wts_f32"), rows * 10 * 4);
        }
    }

    #[test]
    fn the_finish_is_routed_plus_sigmoid_scaled_shared_in_one_bf16_round() {
        // The finish kernel: out = bf16(routed + sigmoid(gate) * shared) — one fp32 accumulate, one bf16 round: the
        // CoopMK slot combine's arithmetic (routed slots summed in slot order, the shared slot weighted by its sigmoid
        // added last, stored bf16 once). At gate 0 the sigmoid is exactly 0.5 (1 / (1 + expf(0)) == 1/2), so the host
        // mirror is exact. The scan also checks the single-round property is real: rounding the shared term separately
        // first would differ on some values.
        use gb_cuda::dsv4_attn::{bf16_bits, bf16_bits_to_f32};
        let sig = 0.5f32;
        let mut differs = None;
        'outer: for ai in 0..4000 {
            let a = (ai as f32) * 0.001 + 0.05;
            for bi in 0..4000 {
                let b = (bi as f32) * 0.001 + 0.05;
                let one_round = bf16_bits(a + b * sig);
                let separate = bf16_bits(bf16_bits_to_f32(bf16_bits(a)) + bf16_bits_to_f32(bf16_bits(b)) * sig);
                if one_round != separate {
                    differs = Some((a, b, one_round, separate));
                    break 'outer;
                }
            }
        }
        assert!(differs.is_some(), "found no value where a separate shared round would differ");
        let (a, b, one_round, separate) = differs.unwrap();
        assert_ne!(one_round, separate, "the single cast and a pre-rounded shared differ at ({a}, {b})");
        assert_eq!(bf16_bits(a + b * sig), one_round, "the kernel's expression is the one cast");
    }
}
