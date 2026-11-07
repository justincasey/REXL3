//! The target model's forward pass on the GPU, for both regimes the engine needs:
//!
//! * **window** (decode and DFlash2 verify): up to 64 rows from several sequences at once. Linear-attention state is
//!   read-only during the window; each layer keeps a record, and `commit` later replays the accepted rows with the same
//!   step routine serial decoding uses. That replay is folded into the sequence's NEXT window kernel (one read and one
//!   write of the 3 MiB per-layer state per round instead of three transfers); the conv history is committed at once.
//!   Attention K/V of the window rows are written into the paged cache at their positions and are simply never made
//!   visible if rejected.
//! * **prefill** (one sequence, a chunk of up to `prefill_rows` rows): the fast paths, which advance the recurrent state
//!   and the KV cache directly.
//!
//! Dataflow per layer (spec 3.1), with the adds fused into the norms:
//! `h = norm(x)`; `a = mixer(h)`; `r = x + a`, `h = norm(r)`; `m = mlp(h)`; `x = r + m`, `h = norm_next(x)`.

use crate::config::QwenConfig;
use crate::draft::{Absorb, Draft, DraftReq, Picked};
use crate::error::ModelError;
use crate::kvtables::{KvTables, BLOCK_TOKENS};
use crate::prefill_p2::PrefillGemm;
use crate::rows::RowBatch;
use crate::scratch::{Scratch, MAX_DECODE_ROWS};
use crate::snapshot::{SnapMeta, SnapPool};
use crate::state::GdnPools;
use crate::weights::{Mixer, Weights};
use crate::window_graph::{Decision, GraphKey, Slot, WindowGraphMutant, WindowGraphs, GRAPH_LIMIT};
use crate::wq::WqResidency;
use core::ffi::c_void;
use gb_cuda::attention::{self, BlockTable, PagedKv, RowIndex, TreeRows};
use gb_cuda::dflash2::DraftCtxLayout;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::gdn::{self, Batch, Owed, OwedPath, PathRows, PreInputs, PreWeights, Record};
use gb_cuda::graph::{is_capture_code, CaptureError, Graph};
use gb_cuda::tree::{Tree, PATH_STRIDE, PLAN_STRIDE};
use gb_cuda::{glue, DevBuf, Stream};
use std::collections::BTreeMap;
use std::time::Instant;

/// Layers whose outputs feed the DFlash2 draft (spec 3.1).
pub const TAP_LAYERS: [usize; 5] = [5, 19, 33, 47, 61];

/// Most snapshots one prefill chunk takes (the GDN recurrence is advanced in one more part than there are cuts inside the chunk).
pub const MAX_CHUNK_CUTS: usize = 4;

/// Ring positions beyond a prefix-cache cut that can be absorbed before the cut's draft window is copied out: the ring holds 2,560 positions,
/// the window 2,048, so the rows after the cut may fill 512 of them without overwriting the window.
const RING_SLACK: usize = DraftCtxLayout::DEFAULT_CAP - DraftCtxLayout::WINDOW;

/// Hashes of what a prompt leaves in a slot (see `Model::state_hashes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateHashes {
    pub s: u64,
    pub conv: u64,
    pub kv: u64,
}

/// Copy the draft ring's window before `c.at` into the snapshot slot and record where it lies.
fn save_window(dr: &Draft, snaps: Option<&mut SnapPool>, slot: usize, c: &SnapCut, st: &Stream) -> Result<(), ModelError> {
    let sn = snaps.ok_or_else(|| ModelError::Request("a snapshot was asked for without a snapshot pool".into()))?;
    let lo = c.at.saturating_sub(DraftCtxLayout::WINDOW);
    let n = c.at - lo;
    dr.save_window(slot, lo, n, &mut sn.dk, &mut sn.dv, c.snap as usize, st)?;
    sn.set_meta(c.snap, SnapMeta { len: c.at, lo, n });
    Ok(())
}

/// A snapshot a prefill chunk takes: the state after exactly `at` tokens of the prompt goes into snapshot slot `snap`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapCut {
    pub at: usize,
    pub snap: u32,
}

/// Rows of logits a window step can produce (the logits buffer is `[WINDOW_ROWS, vocab]` bf16).
pub const WINDOW_ROWS: usize = 64;
const MAX_CANDIDATES: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Window,
    Prefill(PrefillPath),
}

/// The arithmetic a prefill's EXL3 linears use. It is chosen by the TOTAL length of the prompt, never by the row count of a
/// chunk: every row of one prompt then takes the same path, so how the prompt is cut into chunks, or where a reused prefix
/// ended, cannot change a bit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrefillPath {
    /// The decode GEMV (row invariant, no W_q reconstruct): prompts of up to [`PrefillPath::GEMV_MAX_PROMPT`] tokens, so a short
    /// prompt keeps its time to first token.
    Gemv,
    /// W_q reconstruct plus the pinned fixed-algorithm GEMM (row invariant for any row count, tails of 1 row included):
    /// longer prompts, for every chunk.
    Tiled,
}

impl PrefillPath {
    /// Longest prompt that stays on the GEMV: the decode kernel's row limit.
    pub const GEMV_MAX_PROMPT: usize = MAX_DECODE_ROWS;

    /// The path for a prompt of `total` tokens (all of them, not the chunk being run).
    pub fn for_prompt(total: usize) -> Self {
        if total <= Self::GEMV_MAX_PROMPT {
            PrefillPath::Gemv
        } else {
            PrefillPath::Tiled
        }
    }
}

/// One whole short prompt of [`Model::prefill_batch`].
pub struct BatchPrompt<'a> {
    pub slot: usize,
    /// The prompt, all of it (it starts at position 0).
    pub tokens: &'a [u32],
    /// Rows at positions below this are not fed to the draft's context (it never looks that far back), as in `prefill_chunk`.
    pub ctx_from: usize,
}

/// Explicit prefill head selection. `None` truly skips lm_head; scoring MUST use FullLastRow.
#[derive(Clone, Copy, Debug)]
pub enum HeadMode {
    None,
    Candidates(usize),
    FullLastRow,
}

/// Top candidates of every row of a step, as the engine's sampler wants them.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidates {
    pub rows: usize,
    pub count: usize,
    /// `[rows, count]`, value descending, ties by lowest id.
    pub ids: Vec<i32>,
    pub vals: Vec<f32>,
}

/// One splice of an image's encoder output into a prefill chunk (see [`Model::forward_mm`]): rows `src_row..src_row + n` of
/// `feats` (bf16 `[tokens, hidden]`) replace chunk rows `dst_row..dst_row + n`.
pub struct SpliceOp<'a> {
    pub feats: &'a DevBuf,
    pub dst_row: usize,
    pub src_row: usize,
    pub n: usize,
}

pub struct Model {
    pub cfg: QwenConfig,
    pub w: Weights,
    pub s: Scratch,
    pub pools: GdnPools,
    pub kv: Vec<PagedKv>,
    pub kv_tab: KvTables,
    pub table: BlockTable,
    rowidx: RowIndex,
    pub stream: Stream,
    /// Window records of every linear layer, two sets used by alternate windows: the previous window's set is read (its
    /// accepted rows are replayed) by the window that follows it while that window fills the other set.
    recs: [Vec<Record>; 2],
    /// The set the latest window filled.
    rec_par: usize,
    /// Raw qkv rows of every linear layer's latest window, kept until its commit (the conv history is rebuilt from them).
    win_qkv: Vec<DevBuf>,
    /// The S replay of the latest committed window, not yet applied: the next window kernel folds it in.
    owed: Option<OwedWindow>,
    gdn_prefill_ws: DevBuf,
    attn_ws: DevBuf,
    logits: DevBuf,
    topk_ws: DevBuf,
    topk_ids: DevBuf,
    topk_vals: DevBuf,
    /// Most rows one window step takes (`Model::with_window_rows`; [`WINDOW_ROWS`] unless a caller asked for more).
    window_rows: usize,
    pub max_context: usize,
    /// The last window's rows, until its commits are flushed.
    last_window: Option<RowBatch>,
    /// `commit` / `commit_path` calls waiting to be applied together: per slot, the accepted rows of its last window (window-local row
    /// indices, root first; a prefix `0..keep` for a plain commit).
    pending: BTreeMap<usize, Vec<i32>>,
    /// Per slot: rope position minus cache index for every token after the last image of the prompt (0 for text-only
    /// sequences). M-RoPE compresses an image's positions, so later text sits `delta` behind its KV index.
    rope_delta: Vec<i32>,
    /// The DFlash2 draft, when one is attached.
    pub draft: Option<Draft>,
    /// The prefix cache's snapshot slots, when it is enabled (`Model::enable_prefix_cache`).
    pub snaps: Option<SnapPool>,
    /// The snapshots the chunk now being run takes inside its GDN layers: (chunk-local row the state is wanted after, snapshot slot).
    gdn_cuts: Vec<(usize, u32)>,
    /// Device `[0, rows]` pairs of the parts a chunk's GDN layers are run in when it has cuts.
    split_start: DevBuf,
    /// Debug: download each layer's output (bf16 bits) during the next forward.
    pub capture: bool,
    pub captured: Vec<Vec<u16>>,
    /// Buffers of tree windows, once [`Model::enable_trees`] has run.
    tree: Option<TreeBufs>,
    /// Run EVERY window through the tree kernels, chains included (`REXL3_TREE_KERNELS=all`: the exactness runs with trees on). Off: only a
    /// window whose batch carries parents does, and chain-only windows keep the plain kernels.
    tree_all: bool,
    /// The trees of the last window, when it was a tree window (what a committed path is checked against); `None` after a plain window.
    last_trees: Option<Vec<Tree>>,
    /// The CUDA graphs of the fixed-shape decode windows (see `window_graph`): the decision layer, keyed by every host value the
    /// captured region bakes in. `None`-free by construction; the capture stream lives beside it.
    graphs: WindowGraphs<Graph>,
    /// The stream captures run on (non-blocking: a capture on the model's blocking stream would order with the legacy default stream).
    graph_cap: Option<Stream>,
    /// The persistent `W_q` of the prefill's linears ([`crate::wq`]; `None` unless [`Model::enable_wq_residency`] ran:
    /// the knob is off by default). A resident matrix skips its per-chunk reconstruct; the bits are the same either
    /// way (the same pinned GEMM over the same pure function of the trellis).
    pub wq: Option<WqResidency>,
    /// The GEMM the EXL3 linears of a prompt of more than 128 tokens run ([`crate::prefill_p2`]): the process's `REXL3_QWEN_PREFILL_GEMM`
    /// (the fused tensor-core GEMM unless it says `pinned`), read when the model is built. Windows and the GEMV prefill never use it.
    prefill_gemm: PrefillGemm,
}

/// The accepted rows of a window whose replay into the recurrent state has been deferred (see `Model::flush`).
struct OwedWindow {
    /// Which record set holds the window's rows.
    par: usize,
    /// The window's layout (the records are addressed by its row offsets).
    win: RowBatch,
    /// Accepted rows per sequence of `win`, clamped to the sequence's length (0: nothing owed).
    keep: Vec<i32>,
    /// A tree window commits an accepted PATH: the global record rows of each sequence's accepted rows, root first (`None`: the first
    /// `keep` rows of the sequence, as a plain window does).
    paths: Option<Vec<Vec<i32>>>,
}

impl OwedWindow {
    /// Global record rows of the owed replay of sequence `j` of `win`.
    fn rows_of(&self, j: usize) -> Vec<i32> {
        let n = self.keep[j].max(0) as usize;
        match &self.paths {
            Some(p) => p[j][..n.min(p[j].len())].to_vec(),
            None => (0..n as i32).map(|i| self.win.seq_start[j] + i).collect(),
        }
    }
}

/// Bytes of the tree windows' buffers (counted by the memory plan when trees are enabled, see `families::qwen`): per window row the K and V of
/// every full-attention layer, and the index arrays of the window (parent, committed length, root-to-row path, a plan a sequence) and of the
/// commit (accepted rows, KV scatter, owed rows, standalone replay).
///
/// The buffers hold the model's window (`--window-rows`): [`tree_buffer_bytes`] is the served default window, [`tree_buffer_bytes_for`] any window.
pub fn tree_buffer_bytes(cfg: &QwenConfig) -> u64 {
    tree_buffer_bytes_for(cfg, WINDOW_ROWS)
}

/// [`tree_buffer_bytes`] for a window of `window_rows` rows (what `Model::enable_trees` allocates when the model was built with that window).
pub fn tree_buffer_bytes_for(cfg: &QwenConfig, window_rows: usize) -> u64 {
    window_rows as u64 * tree_bytes_per_row(cfg)
}

/// [`tree_buffer_bytes`] per window row (what the memory model adds to a window row's records when trees are enabled).
pub fn tree_bytes_per_row(cfg: &QwenConfig) -> u64 {
    let kv = (cfg.full_layers() * 2 * cfg.kv_heads * cfg.head_dim * 2) as u64;
    let idx =
        (4 + 4 + PATH_STRIDE * 4 + PLAN_STRIDE * 4 + 4 + PATH_STRIDE * 4 + 12 + 4 + PATH_STRIDE * 4 + PATH_STRIDE * 4) as u64;
    kv + idx
}

/// Device buffers of tree windows. A tree window never writes its rows into the paged cache (siblings share logical positions): their K/V
/// live in node-indexed buffers, and `flush` copies the accepted path's rows into the pages.
struct TreeBufs {
    /// Window rows the buffers hold: the model's window (a tree window can be as wide as `--window-rows`, so the buffers are sized by it, not by
    /// the served default [`WINDOW_ROWS`]).
    rows: usize,
    /// bf16 `[rows][4 * 256]` per full-attention layer: the window rows' keys (after norm and rope) and values.
    kn: Vec<DevBuf>,
    vn: Vec<DevBuf>,
    /// i32 `[rows]`: each row's parent (global row, -1 for a root) and the committed length of its sequence.
    parent: DevBuf,
    base: DevBuf,
    /// i32 `[rows][PATH_STRIDE]`: each row's root-to-row rows (global), then -1.
    paths: DevBuf,
    /// i32 `[rows][PLAN_STRIDE]`: the GDN plan of each sequence.
    plan: DevBuf,
    /// The commit of the last window: accepted rows per sequence and their global rows, and the KV scatter's source row, slot and position.
    cm_keep: DevBuf,
    cm_path: DevBuf,
    kv_src: DevBuf,
    kv_slot: DevBuf,
    kv_pos: DevBuf,
    /// The previous window's owed replay, for the window that follows it: rows owed per sequence of the new window and their rows.
    owed_keep: DevBuf,
    owed_path: DevBuf,
    /// The standalone replay of owed rows (a sequence that sits a window out, a slot about to be prefilled).
    mat_path: DevBuf,
}

impl TreeBufs {
    fn alloc(cfg: &QwenConfig, rows: usize) -> Result<Self, ModelError> {
        let kvrow = cfg.kv_heads * cfg.head_dim * 2;
        let z = DevBuf::zeroed;
        let mut kn = Vec::new();
        let mut vn = Vec::new();
        for _ in 0..cfg.full_layers() {
            kn.push(z(rows * kvrow)?);
            vn.push(z(rows * kvrow)?);
        }
        Ok(Self {
            rows,
            kn,
            vn,
            parent: z(rows * 4)?,
            base: z(rows * 4)?,
            paths: z(rows * PATH_STRIDE * 4)?,
            plan: z(rows * PLAN_STRIDE * 4)?,
            cm_keep: z(rows * 4)?,
            cm_path: z(rows * PATH_STRIDE * 4)?,
            kv_src: z(rows * 4)?,
            kv_slot: z(rows * 4)?,
            kv_pos: z(rows * 4)?,
            owed_keep: z(rows * 4)?,
            owed_path: z(rows * PATH_STRIDE * 4)?,
            mat_path: z(rows * PATH_STRIDE * 4)?,
        })
    }

    fn bytes(&self) -> usize {
        let mut n = 0;
        for b in self.kn.iter().chain(self.vn.iter()) {
            n += b.bytes();
        }
        for b in [
            &self.parent,
            &self.base,
            &self.paths,
            &self.plan,
            &self.cm_keep,
            &self.cm_path,
            &self.kv_src,
            &self.kv_slot,
            &self.kv_pos,
            &self.owed_keep,
            &self.owed_path,
            &self.mat_path,
        ] {
            n += b.bytes();
        }
        n
    }
}

/// The host side of one tree window: every sequence's tree, and the index arrays the kernels read.
#[derive(Debug)]
struct TreeLayout {
    trees: Vec<Tree>,
    /// Global parent row of every row (-1 for roots).
    parent: Vec<i32>,
    /// Committed length of each row's sequence.
    base: Vec<i32>,
    /// `rows * PATH_STRIDE`: root-to-row rows (global), then -1.
    paths: Vec<i32>,
    /// `n_seq * PLAN_STRIDE`: one plan per sequence.
    plan: Vec<i32>,
    /// The most saved states any plan needs.
    slots: usize,
}

/// The trees of a window (`b.parents`, or chains) and everything the tree kernels read about them. Fails on a malformed tree or a row whose
/// position is not its root's plus its depth (the kernels' logical positions).
fn tree_layout(b: &RowBatch) -> Result<TreeLayout, ModelError> {
    let rows = b.rows;
    let mut lay = TreeLayout {
        trees: Vec::with_capacity(b.n_seq()),
        parent: vec![-1; rows],
        base: vec![0; rows],
        paths: vec![-1; rows * PATH_STRIDE],
        plan: vec![0; b.n_seq() * PLAN_STRIDE],
        slots: 0,
    };
    for i in 0..b.n_seq() {
        let r = b.seq_rows(i);
        let local: Vec<i32> = match &b.parents {
            Some(p) => r.clone().map(|g| if p[g] < 0 { -1 } else { p[g] - r.start as i32 }).collect(),
            None => (0..r.len() as i32).map(|j| j - 1).collect(),
        };
        let t = Tree::new(&local).map_err(|e| ModelError::Request(format!("sequence {i} of the window: {e}")))?;
        let plan = t.plan().map_err(|e| ModelError::Request(format!("sequence {i} of the window: {e}")))?;
        let root_pos = b.positions[r.start];
        // `j` is the row's index in the sequence's window: it indexes `local` and the tree alike
        #[allow(clippy::needless_range_loop)]
        for j in 0..r.len() {
            let g = r.start + j;
            if b.positions[g] != root_pos + t.depth(j) as i32 {
                return Err(ModelError::Request(format!(
                    "row {g} of sequence {i} sits at position {} but its depth is {}: a tree row's position is the root's plus its depth",
                    b.positions[g],
                    t.depth(j)
                )));
            }
            lay.parent[g] = if local[j] < 0 { -1 } else { r.start as i32 + local[j] };
            lay.base[g] = root_pos;
            for (k, a) in t.path(j).into_iter().enumerate() {
                lay.paths[g * PATH_STRIDE + k] = (r.start + a) as i32;
            }
        }
        lay.plan[i * PLAN_STRIDE..i * PLAN_STRIDE + plan.entries.len()].copy_from_slice(&plan.entries);
        lay.slots = lay.slots.max(plan.slots);
        lay.trees.push(t);
    }
    Ok(lay)
}

/// A committed path must be a root path of the window it commits: rows of the sequence, the first the root, each the parent of the next
/// (a plain window's rows form a chain). An empty path (nothing committed) is fine.
fn check_path(path: &[usize], rows: usize, tree: Option<&Tree>) -> Result<(), ModelError> {
    if path.is_empty() {
        return Ok(());
    }
    let bad =
        |why: String| Err(ModelError::Request(format!("the committed rows {path:?} are not a root path of the window: {why}")));
    if let Some(&r) = path.iter().find(|&&r| r >= rows) {
        return bad(format!("row {r} is past the window's {rows} rows"));
    }
    if path[0] != 0 {
        return bad("the first row is not the root".into());
    }
    for i in 1..path.len() {
        let is_child = match tree {
            Some(t) => t.parents()[path[i]] == path[i - 1] as i32,
            None => path[i] == path[i - 1] + 1,
        };
        if !is_child {
            return bad(format!("row {} is not a child of row {}", path[i], path[i - 1]));
        }
    }
    Ok(())
}

/// Maximal runs of consecutive rows in `rows` (global row indices), as `(first row, length, offset into `rows`)`.
fn runs_of(rows: &[i32]) -> Vec<(usize, usize, usize)> {
    let mut v: Vec<(usize, usize, usize)> = Vec::new();
    for (k, &r) in rows.iter().enumerate() {
        match v.last_mut() {
            Some(last) if last.0 + last.1 == r as usize => last.1 += 1,
            _ => v.push((r as usize, 1, k)),
        }
    }
    v
}

fn cp(b: &DevBuf) -> *const c_void {
    b.ptr() as *const c_void
}

fn mp(b: &DevBuf) -> *mut c_void {
    b.ptr()
}

/// Copy the recurrent state of linear layer `li` between slot `slot` of the sequence pools and slot `snap` of a snapshot pool: `out` copies
/// the sequence's state into the snapshot, otherwise the snapshot's into the sequence's. Stream-ordered.
fn copy_gdn_slot(
    snap_pool: &mut GdnPools,
    snap: usize,
    pools: &mut GdnPools,
    slot: usize,
    li: usize,
    st: &Stream,
    out: bool,
) -> Result<(), ModelError> {
    let (sb, cb) = (gdn::S_SLOT_ELEMS * 4, gdn::CONV_SLOT_ELEMS * 2);
    if out {
        snap_pool.s[li].copy_range_from(snap * sb, &pools.s[li], slot * sb, sb, st)?;
        snap_pool.conv[li].copy_range_from(snap * cb, &pools.conv[li], slot * cb, cb, st)?;
    } else {
        pools.s[li].copy_range_from(slot * sb, &snap_pool.s[li], snap * sb, sb, st)?;
        pools.conv[li].copy_range_from(slot * cb, &snap_pool.conv[li], snap * cb, cb, st)?;
    }
    Ok(())
}

/// Whether an error a launch returned while capturing a window graph is the device's (fatal, as for any launch) and not the
/// capture's own refusal: the 900-series codes ("this call cannot be captured") spoil the capture only, whatever
/// `CudaError::is_fatal` says of a code it does not know.
fn closure_is_fatal(e: &ModelError) -> bool {
    match e {
        ModelError::Cuda(c) if is_capture_code(c.0) => false,
        other => other.is_fatal(),
    }
}

/// One EXL3 linear on `m` rows: the decode GEMV for windows and for the prefill of a short prompt, the pinned GEMM for the
/// prefill of a longer one. The choice is the mode's, never `m`'s (see [`PrefillPath`]).
///
/// The Tiled prefill arm, when `wq` holds a residency for this matrix, runs the pinned GEMM on the resident `W_q`
/// instead of reconstructing it into the scratch: the SAME pinned cublasLt algorithm over the same pure function of
/// the trellis, so a row's bits do not move (the knob-on exactness tests pin this on the real pack).
#[allow(clippy::too_many_arguments)]
fn lin(
    wt: &Exl3Weights,
    x: &DevBuf,
    y: &mut DevBuf,
    m: usize,
    mode: Mode,
    gemm: PrefillGemm,
    ws: &mut DevBuf,
    pws: &mut DevBuf,
    mut wq: Option<&mut WqResidency>,
    st: &Stream,
) -> Result<(), ModelError> {
    let wq = &mut wq.as_mut();
    match mode {
        // the fused tensor-core GEMM (`prefill_p2`; the pinned path's bits): the rotated fp16 input goes into the prefill scratch (it holds `rows * k` fp16 at every
        // K: `prefill_scratch_bytes` starts with that section), the trellis never becomes a `W_q`. A layer it cannot take (none in the
        // validated pack) falls through to the pinned arm.
        Mode::Prefill(PrefillPath::Tiled) if gemm == PrefillGemm::Fused && gb_cuda::glm_p2::supported(wt) => {
            gb_cuda::glm_p2::linear_variant(wt, x, y, m, Dtype::BF16, Dtype::BF16, pws, gb_cuda::glm_p2::Variant::Chain, 0, st)?;
        }
        Mode::Prefill(PrefillPath::Tiled) => {
            if let Some(r) = wq {
                // hit: skip the reconstruct, the pinned GEMM reads the kept W_q (bit-identical by construction);
                // miss: fill the slot now (an MRU victim may go) and run this chunk on the fresh W_q the reconstruct
                // just wrote — the same bits the scratch-held copy would have held
                if r.hit(wt) || r.fill(wt, st)? {
                    let w = r.w(wt).expect("a hit or a fill means the slot is filled");
                    wt.forward_prefill_pinned_with_w(x, Dtype::BF16, w, y, Dtype::BF16, m, pws, st)?;
                    return Ok(());
                }
                // the matrix cannot fit the budget: today's per-chunk path answers for it
            }
            wt.forward_prefill_pinned(x, y, m, Dtype::BF16, pws, st)?;
        }
        // the GEMV takes at most MAX_DECODE_ROWS rows; `forward_mm` refuses a Gemv prefill with more
        Mode::Window | Mode::Prefill(PrefillPath::Gemv) => wt.forward(x, y, m, Dtype::BF16, Dtype::BF16, ws, st)?,
    }
    Ok(())
}

/// A window cap the kernels can serve: 1..=[`MAX_DECODE_ROWS`] rows, which is what the EXL3 decode kernel, the attention decode and the
/// GDN window kernels take in one call. A wider window needs those kernels split into calls (a cap past 128 is refused here, not
/// launched and left to return nothing).
pub fn check_window_rows(rows: usize) -> Result<(), ModelError> {
    if rows == 0 || rows > MAX_DECODE_ROWS {
        return Err(ModelError::Config(format!("a window takes 1..={MAX_DECODE_ROWS} rows, asked for {rows}")));
    }
    Ok(())
}

impl Model {
    pub fn new(
        cfg: QwenConfig,
        w: Weights,
        slots: usize,
        kv_blocks: usize,
        max_context: usize,
        prefill_rows: usize,
    ) -> Result<Self, ModelError> {
        Self::with_window_rows(cfg, w, slots, kv_blocks, max_context, prefill_rows, WINDOW_ROWS)
    }

    /// [`Model::new`] with a window of up to `window_rows` rows (1..=[`MAX_DECODE_ROWS`], the rows the decode kernels take in one call;
    /// the served default is [`WINDOW_ROWS`]). Every buffer that grows with the window (the logits, the records, the attention
    /// workspace, which is 96 MiB per 8 rows at a 262,144-token context) is sized by it, so the cost of a wider window is its memory.
    pub fn with_window_rows(
        cfg: QwenConfig,
        w: Weights,
        slots: usize,
        kv_blocks: usize,
        max_context: usize,
        prefill_rows: usize,
        window_rows: usize,
    ) -> Result<Self, ModelError> {
        check_window_rows(window_rows)?;
        // a value the switch refuses stops the boot here, by name, before the scratch or a kernel exists
        let prefill_gemm = crate::prefill_p2::current().map_err(ModelError::Config)?;
        let max_blocks = max_context.div_ceil(BLOCK_TOKENS);
        let s = Scratch::new(&cfg, prefill_rows.max(MAX_DECODE_ROWS))?;
        let pools = GdnPools::new(&cfg, slots)?;
        let mut kv = Vec::new();
        for _ in 0..cfg.full_layers() {
            kv.push(PagedKv::new(kv_blocks)?);
        }
        let n_lin = cfg.linear_layers();
        let mut recs = [Vec::with_capacity(n_lin), Vec::with_capacity(n_lin)];
        let mut win_qkv = Vec::with_capacity(n_lin);
        for _ in 0..n_lin {
            recs[0].push(Record::alloc(window_rows)?);
            recs[1].push(Record::alloc(window_rows)?);
            win_qkv.push(DevBuf::zeroed(window_rows * cfg.conv_channels() * 2)?);
        }
        let rows = s.rows;
        gb_cuda::exl3_linear::prefill_init()?;
        // The prefill's row invariance is a property of ONE pinned cublasLt algorithm, so a cuBLAS that lacks it for any of the
        // model's shapes (at one row, at a full chunk) must stop the boot rather than quietly change a prompt's bits.
        let mut shapes: Vec<(usize, usize)> = w.prefill_linears().iter().map(|l| (l.k, l.n)).collect();
        shapes.sort_unstable();
        shapes.dedup();
        for &(k, n) in &shapes {
            for m in [1, prefill_rows.max(1)] {
                gb_cuda::exl3_linear::prefill_pin_check(m, k, n).map_err(|e| {
                    ModelError::Config(format!(
                        "the pinned prefill GEMM ({}) does not run a {m} x {k} x {n} problem here ({e}); row-invariant prefill cannot be guaranteed on this cuBLAS",
                        gb_cuda::exl3_linear::prefill_pin_name()
                    ))
                })?;
            }
        }
        let mut model = Self {
            kv_tab: KvTables::new(kv_blocks, slots, max_blocks),
            table: BlockTable::from_host(&vec![Vec::new(); slots], max_blocks)?,
            rowidx: RowIndex::from_host(&vec![0; rows], &vec![0; rows])?,
            stream: Stream::new()?,
            recs,
            rec_par: 0,
            win_qkv,
            owed: None,
            gdn_prefill_ws: DevBuf::zeroed(gdn::prefill_workspace_bytes(prefill_rows))?,
            attn_ws: DevBuf::zeroed(attention::decode_workspace_bytes(window_rows, max_context))?,
            logits: DevBuf::zeroed(window_rows * cfg.vocab * 2)?,
            topk_ws: DevBuf::zeroed(glue::topk_workspace_bytes(window_rows, cfg.vocab, MAX_CANDIDATES))?,
            topk_ids: DevBuf::zeroed(window_rows * MAX_CANDIDATES * 4)?,
            topk_vals: DevBuf::zeroed(window_rows * MAX_CANDIDATES * 4)?,
            window_rows,
            max_context,
            last_window: None,
            pending: BTreeMap::new(),
            rope_delta: vec![0; slots],
            draft: None,
            snaps: None,
            gdn_cuts: Vec::new(),
            split_start: DevBuf::zeroed(8 * (MAX_CHUNK_CUTS + 1))?,
            capture: false,
            captured: Vec::new(),
            tree: None,
            tree_all: false,
            last_trees: None,
            graphs: WindowGraphs::new(),
            graph_cap: Stream::new_nonblocking().ok(),
            wq: None,
            prefill_gemm,
            cfg,
            w,
            s,
            pools,
            kv,
        };
        // a whole run (a test binary, the gate, a server) can switch trees on without touching its code
        if std::env::var("REXL3_SPEC_TREE").as_deref() == Ok("1") {
            model.enable_trees(std::env::var("REXL3_TREE_KERNELS").as_deref() == Ok("all"))?;
        }
        Ok(model)
    }

    /// Allow tree windows (a batch whose `RowBatch::parents` is set): allocate their buffers. `all_windows` also sends every plain chain
    /// window through the tree kernels (the exactness runs: a chain through them must reproduce the plain kernels' bits). `REXL3_SPEC_TREE=1`
    /// and `REXL3_TREE_KERNELS=all` do the same at `Model::new`, so a whole test run or server can be switched without touching its code.
    pub fn enable_trees(&mut self, all_windows: bool) -> Result<(), ModelError> {
        if self.tree.is_none() {
            self.tree = Some(TreeBufs::alloc(&self.cfg, self.window_rows)?);
        }
        self.tree_all = all_windows;
        Ok(())
    }

    /// Whether tree windows are enabled, and whether every window takes the tree kernels.
    pub fn trees_enabled(&self) -> (bool, bool) {
        (self.tree.is_some(), self.tree_all)
    }

    /// Bytes held by the tree windows' buffers (0 when not enabled); the memory plan counts [`tree_buffer_bytes`].
    pub fn tree_bytes(&self) -> usize {
        self.tree.as_ref().map_or(0, TreeBufs::bytes)
    }

    /// The GEMM this model's prompt rows run (`REXL3_QWEN_PREFILL_GEMM`, read at construction).
    pub fn prefill_gemm(&self) -> PrefillGemm {
        self.prefill_gemm
    }

    /// Run the prompt rows of this model on `g` instead of the process's choice (the tests compare the two on one load). The prefix cache's
    /// namespace and the memory plan are the process's, so a server never calls this.
    pub fn set_prefill_gemm(&mut self, g: PrefillGemm) {
        self.prefill_gemm = g;
    }

    /// Turn the persistent `W_q` residency on ([`crate::wq`]): `budget_bytes` of reconstructed `W_q` kept beside the
    /// weights, filled lazily on each matrix's first use, MRU by matrix. Nothing is allocated here (the plan has
    /// already booked the budget; the slots fill as prompts run). Off by default; the bits are the pinned prefill's
    /// own either way.
    pub fn enable_wq_residency(&mut self, budget_bytes: u64) {
        self.wq = Some(WqResidency::new(&self.w.prefill_linears(), budget_bytes));
    }

    /// The residency's counters (hits, fills, evictions, resident bytes): what the boot log and the tests read.
    pub fn wq_stats(&self) -> crate::wq::WqStats {
        let Some(r) = self.wq.as_ref() else { return crate::wq::WqStats::default() };
        let p = r.policy();
        crate::wq::WqStats {
            hits: r.hits(),
            misses: r.misses(),
            evictions: p.evictions(),
            resident: p.resident_count(),
            resident_bytes: p.resident_bytes(),
            budget_bytes: p.budget() as u64,
        }
    }

    /// Most rows one window step takes.
    pub fn window_rows(&self) -> usize {
        self.window_rows
    }

    /// Start a sequence in `slot`: zero its recurrent state and drop any pages it held.
    pub fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        self.flush()?;
        // the slot's old sequence is gone: its owed replay must never reach the zeroed state
        self.forget_owed(slot);
        self.rope_delta[slot] = 0;
        self.pools.reset_slot(slot, &self.stream)?;
        self.kv_tab.release(slot);
        Ok(())
    }

    /// Give back a slot's pages and forget its pending commit.
    pub fn end_slot(&mut self, slot: usize) {
        self.pending.remove(&slot);
        self.forget_owed(slot);
        self.rope_delta[slot] = 0;
        self.kv_tab.release(slot);
    }

    /// Drop the owed replay of `slot` (its state is being discarded).
    fn forget_owed(&mut self, slot: usize) {
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
    }

    /// Replay `keep[j]` rows of sequence j of window `win` (its rows in record set `par`) into the recurrent state with the
    /// standalone commit kernel. Used for owed rows that no window kernel is going to fold in: a sequence that does not run in
    /// the next window, a slot about to be prefilled, or a caller that wants the state itself ([`Model::settle`]).
    ///
    /// `paths`: the accepted rows of a tree window (global record rows of each sequence, root first); `None` replays the first
    /// `keep[j]` rows of the sequence, as a plain window's commit does.
    fn replay_owed(&mut self, par: usize, win: &RowBatch, paths: Option<&Vec<Vec<i32>>>, keep: &[i32]) -> Result<(), ModelError> {
        if keep.iter().all(|&k| k == 0) {
            return Ok(());
        }
        self.s.mat_seq_start.upload(&win.seq_start)?;
        self.s.mat_slot.upload(&win.seq_slot)?;
        self.s.mat_keep.upload(keep)?;
        if let Some(p) = paths {
            let mut flat = vec![0i32; win.n_seq().max(1) * PATH_STRIDE];
            for (j, rows) in p.iter().enumerate() {
                let n = (keep[j].max(0) as usize).min(rows.len());
                flat[j * PATH_STRIDE..j * PATH_STRIDE + n].copy_from_slice(&rows[..n]);
            }
            let tr = self.tree.as_mut().ok_or_else(|| ModelError::Request("a tree commit needs the tree buffers".into()))?;
            tr.mat_path.upload(&flat)?;
        }
        let Model { s, pools, recs, stream, tree, .. } = self;
        let batch =
            Batch { seq_start: &s.mat_seq_start, slot: &s.mat_slot, n_seq: win.n_seq(), rows: win.rows, n_slots: pools.slots };
        for li in 0..recs[par].len() {
            match (paths, tree.as_ref()) {
                (Some(_), Some(tr)) => gdn::commit_state_path(
                    stream,
                    &recs[par][li],
                    &mut pools.s[li],
                    &batch,
                    &s.mat_keep,
                    PathRows { rows: &tr.mat_path, stride: PATH_STRIDE },
                )?,
                _ => gdn::commit_state(stream, &recs[par][li], &mut pools.s[li], &batch, &s.mat_keep)?,
            }
        }
        Ok(())
    }

    /// The previous window's owed rows, as this window's kernels will see them: for the sequences of `b` that have rows owed,
    /// uploads their (first row in the previous record, rows to replay) and returns the record set to read them from.
    /// Rows owed by sequences that do not run in `b` are replayed here, before their record set is overwritten.
    ///
    /// `tree_window`: this window runs the tree kernels, which fold the owed rows as a path (a prefix owed by a plain window is the path
    /// `0..keep`). A plain window folds only a prefix: a path owed by a tree window is replayed standalone first (it cannot be folded).
    fn fold_owed(&mut self, b: &RowBatch, tree_window: bool) -> Result<Option<usize>, ModelError> {
        let Some(ow) = self.owed.take() else { return Ok(None) };
        let n = b.n_seq();
        let (mut row0, mut keep) = (vec![0i32; n], vec![0i32; n]);
        let mut paths: Vec<Vec<i32>> = vec![Vec::new(); n];
        let mut left: Vec<i32> = ow.keep.clone();
        if tree_window || ow.paths.is_none() {
            for (i, &sl) in b.seq_slot.iter().enumerate() {
                if let Some(j) = ow.win.seq_slot.iter().position(|&x| x == sl) {
                    row0[i] = ow.win.seq_start[j];
                    keep[i] = ow.keep[j];
                    paths[i] = ow.rows_of(j);
                    left[j] = 0;
                }
            }
        }
        // the rows of sequences that sit this window out are replayed now (their record set is about to be overwritten) and `owed` is
        // already forgotten: a failure here leaves their state behind the tokens committed, whatever the CUDA code says, like a
        // failed `flush` does
        self.replay_owed(ow.par, &ow.win, ow.paths.as_ref(), &left).map_err(|e| ModelError::StateLost(e.to_string()))?;
        if keep.iter().all(|&k| k == 0) {
            return Ok(None);
        }
        if tree_window {
            let mut flat = vec![0i32; n.max(1) * PATH_STRIDE];
            for (i, rows) in paths.iter().enumerate() {
                flat[i * PATH_STRIDE..i * PATH_STRIDE + rows.len()].copy_from_slice(rows);
            }
            let tr = self.tree.as_mut().ok_or_else(|| ModelError::Request("a tree window needs the tree buffers".into()))?;
            tr.owed_keep.upload(&keep)?;
            tr.owed_path.upload(&flat)?;
        } else {
            self.s.owed_row0.upload(&row0)?;
            self.s.owed_keep.upload(&keep)?;
        }
        Ok(Some(ow.par))
    }

    /// Replay what is owed to `slot` only (a prefill is about to write its state); other slots stay deferred.
    fn settle_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        let Some(ow) = &mut self.owed else { return Ok(()) };
        let mut only = vec![0i32; ow.keep.len()];
        for (j, &sl) in ow.win.seq_slot.iter().enumerate() {
            if sl as usize == slot {
                only[j] = ow.keep[j];
                ow.keep[j] = 0;
            }
        }
        let (par, win, paths) = (ow.par, ow.win.clone(), ow.paths.clone());
        if ow.keep.iter().all(|&k| k == 0) {
            self.owed = None;
        }
        self.replay_owed(par, &win, paths.as_ref(), &only)
    }

    /// Apply everything still owed to the recurrent state: after this, the pools hold the state a `window` followed by an
    /// eager `commit` would have left. The serving path never needs it (the next window folds the replay in); tests that read
    /// or hash the state, and anything that wants to move a slot's state elsewhere, do.
    pub fn settle(&mut self) -> Result<(), ModelError> {
        self.flush()?;
        if let Some(ow) = self.owed.take() {
            self.replay_owed(ow.par, &ow.win, ow.paths.as_ref(), &ow.keep).map_err(|e| ModelError::StateLost(e.to_string()))?;
        }
        Ok(())
    }

    /// The slots whose last committed window still owes accepted rows to their recurrent state (ascending; empty after
    /// [`Model::settle`]). A diagnostic for tests that interleave commits with prefills: nothing on the serving path reads it.
    pub fn owed_slots(&self) -> Vec<usize> {
        let mut v: Vec<usize> = match &self.owed {
            Some(ow) => ow.win.seq_slot.iter().zip(&ow.keep).filter(|&(_, &k)| k > 0).map(|(&sl, _)| sl as usize).collect(),
            None => Vec::new(),
        };
        v.sort_unstable();
        v
    }

    fn upload_rows(&mut self, b: &RowBatch) -> Result<(), ModelError> {
        self.s.ids.upload(&b.ids)?;
        // rope positions: the cache index plus the slot's M-RoPE delta (zero for text-only sequences: the same bits as before)
        let rope: Vec<i32> = b.positions.iter().zip(&b.row_slot).map(|(&p, &sl)| p + self.rope_delta[sl as usize]).collect();
        self.s.positions.upload(&rope)?;
        if let Some(r3) = &b.rope3 {
            self.s.rope3.upload(r3)?;
        }
        self.s.seq_start.upload(&b.seq_start)?;
        self.s.seq_slot.upload(&b.seq_slot)?;
        self.rowidx.upload(&b.row_slot, &b.positions)?;
        Ok(())
    }

    fn ensure_pages(&mut self, b: &RowBatch) -> Result<(), ModelError> {
        for i in 0..b.n_seq() {
            let rows = b.seq_rows(i);
            // a plain window's last row is its highest; a tree's deepest row can be anywhere among the sequence's rows
            let last = rows.clone().map(|r| b.positions[r]).max().unwrap_or(0) as usize;
            if last >= self.max_context {
                return Err(ModelError::Request(format!("position {last} is past the model's context of {}", self.max_context)));
            }
            self.kv_tab.ensure(b.seq_slot[i] as usize, last + 1).map_err(|e| ModelError::Request(format!("{e:?}")))?;
        }
        let tables: Vec<Vec<u32>> = (0..self.pools.slots).map(|s| self.kv_tab.blocks_of(s).to_vec()).collect();
        self.table.upload(&tables)?;
        Ok(())
    }

    /// Run the layers. Leaves the final-normed hidden state of every row in `s.h`, the tap rows in `s.taps`, and (in a
    /// window) the per-layer records for `commit`.
    pub fn forward(&mut self, b: &RowBatch, mode: Mode) -> Result<(), ModelError> {
        self.forward_mm(b, mode, &[])
    }

    /// [`Model::forward`] for a prefill chunk that contains image rows: after the token embedding each splice op overwrites
    /// `n` rows from `dst_row` with rows of the image's encoder output (the `<|image_pad|>` placeholders).
    pub fn forward_mm(&mut self, b: &RowBatch, mode: Mode, splice: &[SpliceOp]) -> Result<(), ModelError> {
        let rows = b.rows;
        match mode {
            Mode::Window if rows == 0 || rows > self.window_rows => {
                return Err(ModelError::Request(format!("a window takes 1..={} rows, got {rows}", self.window_rows)))
            }
            // several sequences in one prefill forward only on the GEMV route (the rows of short whole prompts share one pass of the
            // row-invariant decode kernel); everything else is one sequence
            Mode::Prefill(path)
                if b.n_seq() == 0 || (b.n_seq() > 1 && path != PrefillPath::Gemv) || rows == 0 || rows > self.s.rows =>
            {
                return Err(ModelError::Request(format!("prefill takes one sequence of 1..={} rows", self.s.rows)))
            }
            Mode::Prefill(PrefillPath::Gemv) if rows > MAX_DECODE_ROWS => {
                return Err(ModelError::Request(format!(
                    "a prefill on the GEMV path takes at most {MAX_DECODE_ROWS} rows, got {rows}"
                )))
            }
            _ => {}
        }
        // pending commits use the previous window's index arrays: apply them before those are overwritten
        self.flush()?;
        self.upload_rows(b)?;
        self.ensure_pages(b)?;
        // a window of trees (or every window, for the exactness runs with trees on) takes the tree kernels
        let lay = if mode == Mode::Window && (self.tree_all || b.parents.is_some()) {
            let lay = tree_layout(b)?;
            let tr = self
                .tree
                .as_mut()
                .ok_or_else(|| ModelError::Request("tree windows are not enabled (Model::enable_trees)".into()))?;
            // the buffers hold the window the model was built with (`enable_trees` sizes them by it): a wider tree window would run past them
            if rows > tr.rows {
                return Err(ModelError::Request(format!(
                    "a tree window of {rows} rows exceeds the tree buffers' {} rows",
                    tr.rows
                )));
            }
            tr.parent.upload(&lay.parent)?;
            tr.base.upload(&lay.base)?;
            tr.paths.upload(&lay.paths)?;
            tr.plan.upload(&lay.plan)?;
            Some(lay)
        } else {
            None
        };
        // the replay the previous window still owes: this window's kernels fold it in (Window), or it is applied now for the slot
        // about to be prefilled (Prefill: the other slots stay owed)
        let (par, owed_prev) = match mode {
            Mode::Window => {
                let prev = self.fold_owed(b, lay.is_some())?;
                self.rec_par ^= 1;
                (self.rec_par, prev)
            }
            Mode::Prefill(_) => {
                for j in 0..b.n_seq() {
                    self.settle_slot(b.seq_slot[j] as usize)?;
                }
                (self.rec_par, None)
            }
        };
        self.captured.clear();
        if !self.gdn_cuts.is_empty() {
            if !matches!(mode, Mode::Prefill(_)) || self.gdn_cuts.len() > MAX_CHUNK_CUTS {
                return Err(ModelError::Request("snapshots are taken during a prefill chunk, at most four".into()));
            }
            // part k is rows [cuts[k-1], cuts[k]) (the last runs to the end of the chunk): its `seq_start` pair is [0, its length]
            let mut pairs = Vec::with_capacity(2 * (MAX_CHUNK_CUTS + 1));
            let mut at = 0i32;
            for end in self.gdn_cuts.iter().map(|c| c.0 as i32).chain(std::iter::once(rows as i32)) {
                pairs.extend([0, end - at]);
                at = end;
            }
            self.split_start.upload(&pairs)?;
        }
        self.launch_layers(b, mode, rows, par, owed_prev, lay.as_ref(), splice)?;
        if mode == Mode::Window {
            self.last_window = Some(b.clone());
            self.last_trees = lay.map(|l| l.trees);
        }
        Ok(())
    }

    /// The forward's launch region: everything from the embedding to the last layer's residual add — pure kernel launches on
    /// `self.stream`, in order, reading nothing the host changes between two windows of one shape (see `window_graph` for the audit).
    /// This is the part a CUDA graph can hold: the preamble (the flush of the previous window's commits, the uploads, `fold_owed`)
    /// stays eager. `par` / `owed_prev` are what the caller's parity flip and `fold_owed` produced; `lay` is a tree window's layout
    /// (already uploaded), `None` for a plain chain window.
    #[allow(clippy::too_many_arguments)]
    fn launch_layers(
        &mut self,
        b: &RowBatch,
        mode: Mode,
        rows: usize,
        par: usize,
        owed_prev: Option<usize>,
        lay: Option<&TreeLayout>,
        splice: &[SpliceOp],
    ) -> Result<(), ModelError> {
        let tree_slots = lay.map_or(0, |l| l.slots);
        let capture = self.capture;
        let Model {
            cfg,
            w,
            s,
            pools,
            kv,
            table,
            rowidx,
            stream,
            recs,
            win_qkv,
            gdn_prefill_ws,
            attn_ws,
            max_context,
            captured,
            tree,
            snaps,
            gdn_cuts,
            split_start,
            wq,
            prefill_gemm,
            ..
        } = self;
        let st = &*stream;
        let d = cfg.hidden;
        let eps = cfg.rms_eps as f32;
        let batch = Batch { seq_start: &s.seq_start, slot: &s.seq_slot, n_seq: b.n_seq(), rows, n_slots: pools.slots };
        unsafe {
            glue::embed(cp(&w.embed), cp(&s.ids), mp(&s.x), rows, d, cfg.vocab, st)?;
            for op in splice {
                if op.dst_row + op.n > rows || op.feats.bytes() < (op.src_row + op.n) * d * 2 {
                    return Err(ModelError::Request("image splice outside the chunk or the encoder output".into()));
                }
                glue::copy_rows(
                    (op.feats.ptr() as *const u16).add(op.src_row * d) as *const c_void,
                    d,
                    (s.x.ptr() as *mut u16).add(op.dst_row * d) as *mut c_void,
                    d,
                    op.n,
                    d,
                    st,
                )?;
            }
            glue::rmsnorm(cp(&s.x), cp(&w.layers[0].input_norm), mp(&s.h), rows, d, eps, st)?;
        }
        let (mut li, mut fi) = (0usize, 0usize);
        for (i, layer) in w.layers.iter().enumerate() {
            match &layer.mixer {
                Mixer::Linear(m) => {
                    // projections; in a window the raw qkv rows live in a per-layer buffer so commit can replay them
                    if mode == Mode::Window {
                        lin(
                            &m.in_proj_qkv,
                            &s.h,
                            &mut win_qkv[li],
                            rows,
                            mode,
                            *prefill_gemm,
                            &mut s.ws,
                            &mut s.prefill_ws,
                            wq.as_mut(),
                            st,
                        )?;
                    } else {
                        lin(
                            &m.in_proj_qkv,
                            &s.h,
                            &mut s.qkv,
                            rows,
                            mode,
                            *prefill_gemm,
                            &mut s.ws,
                            &mut s.prefill_ws,
                            wq.as_mut(),
                            st,
                        )?;
                    }
                    lin(&m.in_proj_z, &s.h, &mut s.z, rows, mode, *prefill_gemm, &mut s.ws, &mut s.prefill_ws, wq.as_mut(), st)?;
                    unsafe {
                        glue::small_linear_f16w(cp(&s.h), cp(&m.in_proj_a), mp(&s.ga), rows, d, cfg.lin_value_heads, st)?;
                        glue::small_linear_f16w(cp(&s.h), cp(&m.in_proj_b), mp(&s.gb), rows, d, cfg.lin_value_heads, st)?;
                    }
                    let wts = PreWeights { conv_w: &m.conv_w, a_log: &m.a_log, dt_bias: &m.dt_bias };
                    if mode == Mode::Window && lay.is_some() {
                        let tr = tree.as_ref().expect("tree buffers exist for a tree window");
                        let inp = PreInputs { qkv: &win_qkv[li], a: &s.ga, b: &s.gb };
                        gdn::pre_tree(st, &inp, &wts, &pools.conv[li], &batch, &mut recs[par][li], &tr.parent)?;
                        let owed = owed_prev.map(|pp| OwedPath {
                            rec: &recs[pp][li],
                            keep: &tr.owed_keep,
                            path: PathRows { rows: &tr.owed_path, stride: PATH_STRIDE },
                        });
                        gdn::window_tree(
                            st,
                            &recs[par][li],
                            owed.as_ref(),
                            &mut pools.s[li],
                            &batch,
                            &mut s.o,
                            &tr.plan,
                            PLAN_STRIDE,
                            tree_slots,
                        )?;
                    } else if mode == Mode::Window {
                        let inp = PreInputs { qkv: &win_qkv[li], a: &s.ga, b: &s.gb };
                        gdn::pre(st, &inp, &wts, &pools.conv[li], &batch, &mut recs[par][li])?;
                        match owed_prev {
                            Some(pp) => {
                                let owed = Owed { rec: &recs[pp][li], row0: &s.owed_row0, keep: &s.owed_keep };
                                gdn::window_owed(st, &recs[par][li], &owed, &mut pools.s[li], &batch, &mut s.o)?
                            }
                            None => gdn::window(st, &recs[par][li], &pools.s[li], &batch, &mut s.o)?,
                        }
                    } else {
                        let inp = PreInputs { qkv: &s.qkv, a: &s.ga, b: &s.gb };
                        if gdn_cuts.is_empty() {
                            gdn::prefill(
                                st,
                                &inp,
                                &wts,
                                &mut pools.s[li],
                                &mut pools.conv[li],
                                &batch,
                                gdn_prefill_ws,
                                &mut s.o,
                            )?;
                        } else {
                            // The chunk's recurrence in parts, split at the rows a snapshot is wanted after: exactly the bits of one call
                            // (`gdn::prefill_part`), with the state copied out between two parts. The linears of the layer ran once for
                            // every row, so no second W_q reconstruct is paid for a cut.
                            let sn = snaps
                                .as_mut()
                                .ok_or_else(|| ModelError::Request("a snapshot was asked for without a snapshot pool".into()))?;
                            let slot = b.seq_slot[0] as usize;
                            let mut at = 0usize;
                            for k in 0..=gdn_cuts.len() {
                                let end = gdn_cuts.get(k).map_or(rows, |c| c.0);
                                gdn::prefill_part(
                                    st,
                                    &inp,
                                    at,
                                    end - at,
                                    &wts,
                                    &mut pools.s[li],
                                    &mut pools.conv[li],
                                    split_start,
                                    8 * k,
                                    &s.seq_slot,
                                    pools.slots,
                                    gdn_prefill_ws,
                                    &mut s.o,
                                )?;
                                if let Some(&(_, snap)) = gdn_cuts.get(k) {
                                    copy_gdn_slot(&mut sn.gdn, snap as usize, pools, slot, li, st, true)?;
                                }
                                at = end;
                            }
                        }
                    }
                    unsafe {
                        glue::gated_rmsnorm(
                            cp(&s.o),
                            cp(&s.z),
                            cp(&m.norm),
                            mp(&s.o),
                            rows,
                            cfg.lin_value_heads,
                            cfg.lin_value_dim,
                            1e-6,
                            st,
                        )?;
                    }
                    lin(&m.out_proj, &s.o, &mut s.a, rows, mode, *prefill_gemm, &mut s.ws, &mut s.prefill_ws, wq.as_mut(), st)?;
                    li += 1;
                }
                Mixer::Full(m) => {
                    lin(&m.q_proj, &s.h, &mut s.qkv, rows, mode, *prefill_gemm, &mut s.ws, &mut s.prefill_ws, wq.as_mut(), st)?;
                    lin(&m.k_proj, &s.h, &mut s.fk, rows, mode, *prefill_gemm, &mut s.ws, &mut s.prefill_ws, wq.as_mut(), st)?;
                    // a tree window keeps its rows' K and V in the node buffers (siblings share logical positions, so nothing of the
                    // window may enter the paged cache before the commit has picked the accepted path)
                    let tw = mode == Mode::Window && lay.is_some();
                    if tw {
                        let tr = tree.as_mut().expect("tree buffers exist for a tree window");
                        lin(
                            &m.v_proj,
                            &s.h,
                            &mut tr.vn[fi],
                            rows,
                            mode,
                            *prefill_gemm,
                            &mut s.ws,
                            &mut s.prefill_ws,
                            wq.as_mut(),
                            st,
                        )?;
                    } else {
                        lin(
                            &m.v_proj,
                            &s.h,
                            &mut s.fv,
                            rows,
                            mode,
                            *prefill_gemm,
                            &mut s.ws,
                            &mut s.prefill_ws,
                            wq.as_mut(),
                            st,
                        )?;
                    }
                    unsafe {
                        let prep = if b.rope3.is_some() { glue::attn_prep_mrope } else { glue::attn_prep };
                        let pos = if b.rope3.is_some() { cp(&s.rope3) } else { cp(&s.positions) };
                        let k_out = if tw { mp(&tree.as_ref().expect("tree buffers").kn[fi]) } else { mp(&s.fk2) };
                        prep(
                            &glue::AttnPrepCfg::QWEN_TARGET,
                            cp(&s.qkv),
                            cp(&s.fk),
                            pos,
                            cp(&m.q_norm),
                            cp(&m.k_norm),
                            mp(&s.fq),
                            mp(&s.fgate),
                            k_out,
                            rows,
                            st,
                        )?;
                    }
                    if tw {
                        let tr = tree.as_ref().expect("tree buffers exist for a tree window");
                        let tree_rows = TreeRows { base: &tr.base, paths: &tr.paths, stride: PATH_STRIDE };
                        attention::attn_decode_tree(
                            &s.fq,
                            &kv[fi],
                            table,
                            rowidx,
                            &tree_rows,
                            &tr.kn[fi],
                            &tr.vn[fi],
                            rows,
                            &mut s.o,
                            *max_context,
                            attn_ws,
                            st,
                        )?;
                    } else {
                        attention::kv_write(&s.fk2, &s.fv, rowidx, table, &mut kv[fi], rows, st)?;
                        if mode == Mode::Window {
                            attention::attn_decode(&s.fq, &kv[fi], table, rowidx, rows, &mut s.o, *max_context, attn_ws, st)?;
                        } else {
                            // one call per sequence over its own rows (a single sequence is the call it always was: row0 0)
                            for j in 0..b.n_seq() {
                                let r = b.seq_rows(j);
                                let start = b.positions[r.start] as usize;
                                attention::attn_prefill_rows(
                                    &s.fq,
                                    &kv[fi],
                                    table,
                                    b.seq_slot[j] as usize,
                                    start,
                                    r.start,
                                    r.len(),
                                    &mut s.o,
                                    0,
                                    st,
                                )?;
                            }
                        }
                    }
                    unsafe {
                        glue::gate_mul(cp(&s.o), cp(&s.fgate), mp(&s.o), rows * cfg.heads * cfg.head_dim, st)?;
                    }
                    lin(&m.o_proj, &s.o, &mut s.a, rows, mode, *prefill_gemm, &mut s.ws, &mut s.prefill_ws, wq.as_mut(), st)?;
                    fi += 1;
                }
            }
            // r = x + a; h = norm_post(r)
            unsafe {
                glue::add_rmsnorm(cp(&s.x), cp(&s.a), cp(&layer.post_norm), mp(&s.r), mp(&s.h), rows, d, eps, st)?;
            }
            lin(&layer.mlp.gate, &s.h, &mut s.g, rows, mode, *prefill_gemm, &mut s.ws, &mut s.prefill_ws, wq.as_mut(), st)?;
            lin(&layer.mlp.up, &s.h, &mut s.u, rows, mode, *prefill_gemm, &mut s.ws, &mut s.prefill_ws, wq.as_mut(), st)?;
            unsafe {
                glue::swiglu(cp(&s.g), cp(&s.u), mp(&s.g), rows * cfg.intermediate, st)?;
            }
            lin(&layer.mlp.down, &s.g, &mut s.a, rows, mode, *prefill_gemm, &mut s.ws, &mut s.prefill_ws, wq.as_mut(), st)?;
            // x = r + m; h = norm_next(x)
            let next = if i + 1 < w.layers.len() { &w.layers[i + 1].input_norm } else { &w.final_norm };
            unsafe {
                glue::add_rmsnorm(cp(&s.r), cp(&s.a), cp(next), mp(&s.x), mp(&s.h), rows, d, eps, st)?;
            }
            if let Some(t) = TAP_LAYERS.iter().position(|&l| l == i) {
                unsafe {
                    glue::copy_rows(cp(&s.x), d, (s.taps.ptr() as *mut u16).add(t * d) as *mut c_void, 5 * d, rows, d, st)?;
                }
            }
            if capture {
                st.sync()?;
                let mut v = vec![0u16; rows * d];
                s.x.download(&mut v)?;
                captured.push(v);
            }
        }
        Ok(())
    }

    /// Candidates for every row of a window step (decode or verify): runs the window, the head, and the top-k. A fixed-shape
    /// chain window (the AR/serial decode path and a chain verify window: no `parents`, no tree kernels) goes through the CUDA
    /// graph cache (`window_graph`): a captured shape replays one graph launch instead of ~1,600, with every byte of input in
    /// the same device buffers the eager launches read. Tree windows and debug captures stay eager.
    pub fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        if count == 0 || count > MAX_CANDIDATES {
            return Err(ModelError::Request(format!("candidate count must be 1..={MAX_CANDIDATES}")));
        }
        let rows = b.rows;
        if rows == 0 || rows > self.window_rows {
            return Err(ModelError::Request(format!("a window takes 1..={} rows, got {rows}", self.window_rows)));
        }
        if self.window_eligible(b) {
            // the mutant that leaves the rows stale at a replay asks before the uploads it would skip
            let stale = self.graphs.would_replay(GraphKey::of(b, count, self.rec_par ^ 1, self.owed.is_some()))
                && self.graphs.mutant == Some(WindowGraphMutant::StaleRows);
            // the preamble, exactly what `forward` does before its launch region: the previous window's commits first (they read
            // its index arrays, which the uploads below overwrite), then this window's inputs, then the owed fold and the parity
            // flip
            self.flush()?;
            if !stale {
                self.upload_rows(b)?;
                self.ensure_pages(b)?;
            }
            let prev = self.fold_owed(b, false)?;
            self.rec_par ^= 1;
            self.run_window_graphed(b, count, self.rec_par, prev)?;
            self.last_window = Some(b.clone());
            self.last_trees = None;
            return self.download_candidates(rows, count);
        }
        self.forward(b, Mode::Window)?;
        let Model { w, s, stream, logits, .. } = self;
        w.lm_head.forward(&s.h, logits, rows, Dtype::BF16, Dtype::BF16, &mut s.ws, stream)?;
        self.candidates_of(rows, count)
    }

    /// Whether a window may go through the graph cache: the fixed-shape scope only. Trees (a batch with parents, or
    /// `REXL3_TREE_KERNELS=all`) and the debug capture add launches and host work the graph cannot hold, so they stay eager.
    fn window_eligible(&self, b: &RowBatch) -> bool {
        self.graphs.on
            && self.graph_cap.is_some()
            && b.parents.is_none()
            && b.rope3.is_none()
            && !self.tree_all
            && !self.capture
            && self.gdn_cuts.is_empty()
            && self.snaps.is_none()
    }

    /// A window's launches through the graph cache (the preamble has run): replay the shape's graph, capture it on the second
    /// sighting, or run the same launches eagerly — the layers, the head and the top-k in every arm, so the graph and the eager
    /// fallback do exactly the same work. A capture that fails for a non-sticky reason is logged, marks the shape eager for the
    /// rest of the process and the window runs eagerly; a sticky device error (or a fatal closure error) is returned as for any
    /// launch.
    fn run_window_graphed(&mut self, b: &RowBatch, count: usize, par: usize, owed_prev: Option<usize>) -> Result<(), ModelError> {
        let key = GraphKey::of(b, count, par, owed_prev.is_some());
        let run_eager = |m: &mut Model| -> Result<(), ModelError> {
            let rows = b.rows;
            m.launch_layers(b, Mode::Window, rows, par, owed_prev, None, &[])?;
            let Model { w, s, stream, logits, .. } = m;
            w.lm_head.forward(&s.h, logits, rows, Dtype::BF16, Dtype::BF16, &mut s.ws, stream)?;
            m.launch_topk(rows, count)
        };
        match self.graphs.decide(key) {
            Decision::Replay => {
                self.graphs.stats.replays += 1;
                let Model { graphs, stream, .. } = self;
                let g = match graphs.slots.get_mut(&key) {
                    Some(Slot::Ready(g, _)) => g,
                    _ => unreachable!("decide said Replay for a shape without a graph"),
                };
                g.launch(stream).map_err(ModelError::from)
            }
            Decision::Warmup | Decision::Eager => {
                self.graphs.stats.eager_windows += 1;
                run_eager(self)
            }
            Decision::Capture => {
                self.make_graph_room()?;
                let t = Instant::now();
                match self.capture_window_graph(b, count, par, owed_prev) {
                    Ok(g) => {
                        self.graphs.stats.captures += 1;
                        self.graphs.stats.capture_ms += t.elapsed().as_secs_f64() * 1e3;
                        self.graphs.mark_ready(key, g);
                        let Model { graphs, stream, .. } = self;
                        let g = match graphs.slots.get_mut(&key) {
                            Some(Slot::Ready(g, _)) => g,
                            _ => unreachable!("the graph was just made ready"),
                        };
                        g.launch(stream).map_err(ModelError::from)
                    }
                    Err(e) => {
                        if let Some(fatal) = e.fatal() {
                            return Err(ModelError::Cuda(fatal.clone()));
                        }
                        if let CaptureError::Closure(m) = &e {
                            if closure_is_fatal(m) {
                                return Err(match e {
                                    CaptureError::Closure(m) => m,
                                    CaptureError::Refused(c) => ModelError::Cuda(c),
                                });
                            }
                        }
                        let why = match &e {
                            CaptureError::Refused(c) => format!("the runtime refused the capture: {c}"),
                            CaptureError::Closure(m) => format!("a launch failed while capturing: {m}"),
                        };
                        eprintln!(
                            "rexl3 level=warn event=window_graph_capture_failed rows={} n_seq={} par={} {why} (this window shape runs eagerly)",
                            key.rows, key.n_seq, key.par
                        );
                        self.graphs.stats.capture_failures += 1;
                        self.graphs.stats.last_failure = Some(why);
                        self.graphs.mark_failed(key);
                        self.graphs.stats.eager_windows += 1;
                        run_eager(self)
                    }
                }
            }
        }
    }

    /// Capture a fixed-shape window's launches (layers, head and top-k) on the capture stream. The executor uses `self.stream`,
    /// so the two streams trade places for the capture; nothing runs. On `Err` nothing was captured and nothing is left behind
    /// (see `gb_cuda::graph`).
    fn capture_window_graph(
        &mut self,
        b: &RowBatch,
        count: usize,
        par: usize,
        owed_prev: Option<usize>,
    ) -> Result<Graph, CaptureError<ModelError>> {
        let Some(mut cap) = self.graph_cap.take() else {
            return Err(CaptureError::Refused(gb_cuda::CudaError(-1)));
        };
        std::mem::swap(&mut self.stream, &mut cap);
        let out = match Graph::begin(&self.stream) {
            Ok(c) => {
                let sync = self.graphs.mutant == Some(WindowGraphMutant::SyncInCapture);
                let ran = self.launch_window_graphed(b, count, par, owed_prev, sync);
                c.finish(ran)
            }
            Err(e) => Err(CaptureError::Refused(e)),
        };
        std::mem::swap(&mut self.stream, &mut cap);
        self.graph_cap = Some(cap);
        out
    }

    /// The captured region proper: the layers, the head and the top-k, nothing else. With `sync` set (the mutant) a host
    /// synchronisation is made in the middle, which the capture must refuse.
    fn launch_window_graphed(
        &mut self,
        b: &RowBatch,
        count: usize,
        par: usize,
        owed_prev: Option<usize>,
        sync: bool,
    ) -> Result<(), ModelError> {
        let rows = b.rows;
        if sync {
            self.stream.sync()?;
        }
        self.launch_layers(b, Mode::Window, rows, par, owed_prev, None, &[])?;
        let Model { w, s, stream, logits, .. } = self;
        w.lm_head.forward(&s.h, logits, rows, Dtype::BF16, Dtype::BF16, &mut s.ws, stream)?;
        if sync {
            stream.sync()?;
        }
        let Model { cfg, stream, logits, topk_ws, topk_ids, topk_vals, .. } = self;
        unsafe {
            glue::topk(cp(logits), rows, cfg.vocab, count, mp(topk_ids), mp(topk_vals), mp(topk_ws), topk_ws.bytes(), stream)?;
        }
        Ok(())
    }

    /// Keep the number of instantiated graphs under the limit: drop the least recently used (after the stream has drained: a
    /// graph that is still executing must not be destroyed).
    fn make_graph_room(&mut self) -> Result<(), ModelError> {
        if self.graphs.ready_len() < GRAPH_LIMIT {
            return Ok(());
        }
        self.stream.sync()?;
        self.graphs.demote_oldest();
        Ok(())
    }

    /// Switch the window graphs on or off for the windows that follow (eager and graphs run the same launches; the bit tests
    /// run both).
    pub fn set_graphs(&mut self, on: bool) {
        self.graphs.on = on && self.graph_cap.is_some();
    }

    pub fn graphs_on(&self) -> bool {
        self.graphs.on
    }

    pub fn graph_stats(&self) -> crate::window_graph::WindowGraphStats {
        self.graphs.stats.clone()
    }

    /// Forget every captured window graph (a test of the capture path; the stream is drained first).
    pub fn clear_graphs(&mut self) -> Result<(), ModelError> {
        self.stream.sync()?;
        self.graphs.clear();
        Ok(())
    }

    /// Debug (the mutation tests): run the graph path with a deliberate bug.
    pub fn set_window_graph_mutant(&mut self, m: Option<WindowGraphMutant>) {
        self.graphs.mutant = m;
    }

    /// The top-k launch alone (the graph holds it; `download_candidates` reads the results back).
    fn launch_topk(&mut self, rows: usize, count: usize) -> Result<(), ModelError> {
        let Model { cfg, stream, logits, topk_ws, topk_ids, topk_vals, .. } = self;
        unsafe {
            glue::topk(cp(logits), rows, cfg.vocab, count, mp(topk_ids), mp(topk_vals), mp(topk_ws), topk_ws.bytes(), stream)?;
        }
        Ok(())
    }

    fn download_candidates(&mut self, rows: usize, count: usize) -> Result<Candidates, ModelError> {
        let Model { stream, topk_ids, topk_vals, .. } = self;
        stream.sync()?;
        let mut ids = vec![0i32; rows * count];
        let mut vals = vec![0f32; rows * count];
        topk_ids.download(&mut ids)?;
        topk_vals.download(&mut vals)?;
        Ok(Candidates { rows, count, ids, vals })
    }

    fn candidates_of(&mut self, rows: usize, count: usize) -> Result<Candidates, ModelError> {
        let Model { cfg, stream, logits, topk_ws, topk_ids, topk_vals, .. } = self;
        unsafe {
            glue::topk(cp(logits), rows, cfg.vocab, count, mp(topk_ids), mp(topk_vals), mp(topk_ws), topk_ws.bytes(), stream)?;
        }
        stream.sync()?;
        let mut ids = vec![0i32; rows * count];
        let mut vals = vec![0f32; rows * count];
        topk_ids.download(&mut ids)?;
        topk_vals.download(&mut vals)?;
        Ok(Candidates { rows, count, ids, vals })
    }

    /// Prefill one chunk of one sequence: `tokens` are positions `start..start + tokens.len()` of a prompt of `total` tokens
    /// in all. `total` (not the chunk's row count) picks the arithmetic of the whole prompt, see [`PrefillPath`], so pass the
    /// same `total` for every chunk of a prompt. With `want` set, also returns the candidates of the chunk's LAST row. Rows
    /// at positions below `ctx_from` are not fed to the draft's context (it never looks that far back).
    pub fn prefill_chunk(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        total: usize,
        want: Option<usize>,
        ctx_from: usize,
    ) -> Result<Option<Candidates>, ModelError> {
        self.prefill_chunk_mm(slot, start, tokens, total, want, ctx_from, None, &[])
    }

    /// Set the slot's rope delta (rope position minus cache index of every later text token); see `rope_delta`.
    pub fn set_rope_delta(&mut self, slot: usize, delta: i32) {
        self.rope_delta[slot] = delta;
    }

    /// [`Model::prefill_chunk`] for a chunk with image rows: `rope3` holds the (T, H, W) rope position of every row of the chunk
    /// (`[rows * 3]`; pass `None` for a chunk of text whose rope positions are the slot's cache index plus its rope delta), and
    /// `splice` overwrites the image placeholder rows with encoder output.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_chunk_mm(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        total: usize,
        want: Option<usize>,
        ctx_from: usize,
        rope3: Option<Vec<i32>>,
        splice: &[SpliceOp],
    ) -> Result<Option<Candidates>, ModelError> {
        self.prefill_chunk_snap(slot, start, tokens, total, want, ctx_from, rope3, splice, &[])
    }

    /// [`Model::prefill_chunk_mm`] that also takes a snapshot of the slot's state after exactly `cut.at` tokens for each of `cuts` (ascending,
    /// each `start < at <= start + tokens.len()`): the recurrent state and conv history of every linear layer, and the draft's context window,
    /// are copied into snapshot slot `cut.snap` (the KV pages are not copied: the caller takes references to them, see
    /// [`Model::retain_prefix_pages`]). The prompt's bits are those of the same chunk without cuts: the linears run once for all rows,
    /// the GDN recurrence in parts. `ctx_from` is the first position the draft's context is built from; it must be at most `at - 2048` for
    /// every cut (the window must have been absorbed). Nothing is synchronized beyond what the chunk does anyway.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_chunk_snap(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        total: usize,
        want: Option<usize>,
        ctx_from: usize,
        rope3: Option<Vec<i32>>,
        splice: &[SpliceOp],
        cuts: &[SnapCut],
    ) -> Result<Option<Candidates>, ModelError> {
        self.prefill_chunk_snap_head(
            slot,
            start,
            tokens,
            total,
            want.map_or(HeadMode::None, HeadMode::Candidates),
            ctx_from,
            rope3,
            splice,
            cuts,
        )
    }

    /// Same arithmetic/captures as prefill_chunk_snap; explicit scoring head is never skipped.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_chunk_snap_head(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        total: usize,
        head: HeadMode,
        ctx_from: usize,
        rope3: Option<Vec<i32>>,
        splice: &[SpliceOp],
        cuts: &[SnapCut],
    ) -> Result<Option<Candidates>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::Request("empty prefill chunk".into()));
        }
        if start + tokens.len() > total {
            return Err(ModelError::Request(format!(
                "a chunk of {} tokens at position {start} runs past the prompt's length {total}",
                tokens.len()
            )));
        }
        let end = start + tokens.len();
        if !cuts.is_empty() {
            let ok = cuts.len() <= MAX_CHUNK_CUTS
                && cuts.windows(2).all(|w| w[0].at < w[1].at)
                && cuts.iter().all(|c| c.at > start && c.at <= end)
                && self.snaps.as_ref().is_some_and(|sp| cuts.iter().all(|c| (c.snap as usize) < sp.slots));
            if !ok {
                return Err(ModelError::Request(format!("bad snapshot cuts {cuts:?} for the chunk {start}..{end}")));
            }
        }
        let mut b =
            RowBatch::from_requests(&[gb_engine::RowRequest { slot, tokens: tokens.to_vec(), first_position: start as u64 }]);
        if let Some(r3) = &rope3 {
            if r3.len() != tokens.len() * 3 {
                return Err(ModelError::Request("rope3 must hold three positions per row".into()));
            }
        }
        b.rope3 = rope3;
        // the cuts inside the chunk are taken by its GDN layers; a cut at its end is copied out of the pools once it is done
        self.gdn_cuts = cuts.iter().filter(|c| c.at < end).map(|c| (c.at - start, c.snap)).collect();
        let r = self.forward_mm(&b, Mode::Prefill(PrefillPath::for_prompt(total)), splice);
        self.gdn_cuts.clear();
        r?;
        if let Some(c) = cuts.last().filter(|c| c.at == end) {
            let Model { pools, snaps, stream, cfg, .. } = self;
            let sn = snaps.as_mut().expect("checked above");
            for li in 0..cfg.linear_layers() {
                copy_gdn_slot(&mut sn.gdn, c.snap as usize, pools, slot, li, stream, true)?;
            }
        }
        if let Some(dr) = self.draft.as_mut() {
            // the draft sees at most its window of context: skip rows far behind the end of the prompt, and far behind every cut
            let n = tokens.len();
            let skip = ctx_from.saturating_sub(start).min(n);
            // A cut's window is copied out of the ring once the rows up to it are absorbed. The ring holds 512 positions more than the window, so
            // a cut within 512 rows of the end of this chunk is copied after the last absorb (one absorb call, as without cuts); a cut further
            // back splits the absorb there. A range the ring cannot take in one write (2,560 positions or more) is absorbed in groups of 2,048.
            let split: Vec<&SnapCut> = cuts.iter().filter(|c| end - c.at > RING_SLACK && c.at - start > skip).collect();
            let mut at = skip;
            for upto in split.iter().map(|c| c.at - start).chain(std::iter::once(n)) {
                let mut rows = upto - at;
                if rows >= DraftCtxLayout::DEFAULT_CAP {
                    while rows > 0 {
                        let g = rows.min(DraftCtxLayout::WINDOW);
                        dr.absorb(&self.s.taps, at, g, slot, start + at, &self.stream)?;
                        at += g;
                        rows -= g;
                    }
                } else {
                    dr.absorb(&self.s.taps, at, rows, slot, start + at, &self.stream)?;
                    at = upto;
                }
                if let Some(c) = split.iter().find(|c| c.at - start == upto) {
                    save_window(dr, self.snaps.as_mut(), slot, c, &self.stream)?;
                }
            }
            for c in cuts.iter().filter(|c| !split.iter().any(|x| x.at == c.at)) {
                save_window(dr, self.snaps.as_mut(), slot, c, &self.stream)?;
            }
        } else if let Some(sn) = self.snaps.as_mut() {
            for c in cuts {
                sn.set_meta(c.snap, SnapMeta { len: c.at, lo: c.at, n: 0 });
            }
        }
        if matches!(head, HeadMode::None) {
            self.stream.sync()?;
            return Ok(None);
        }
        let d = self.cfg.hidden;
        let rows = b.rows;
        {
            // the head wants one row: move the last normed row to the front of a free buffer
            let Model { w, s, stream, logits, .. } = self;
            unsafe {
                glue::copy_rows((s.h.ptr() as *const u16).add((rows - 1) * d) as *const c_void, d, mp(&s.r), d, 1, d, stream)?;
            }
            w.lm_head.forward(&s.r, logits, 1, Dtype::BF16, Dtype::BF16, &mut s.ws, stream)?;
        }
        match head {
            HeadMode::Candidates(count) => self.candidates_of(1, count).map(Some),
            HeadMode::FullLastRow => {
                self.stream.sync()?;
                Ok(None)
            }
            HeadMode::None => unreachable!(),
        }
    }

    /// Download the one-row prefill head (ALWAYS row zero), immediately after FullLastRow.
    /// Hidden input was copied from the current chunk's last normalized row into s.r.
    pub fn score_head_row_zero(&self) -> Result<Vec<f32>, ModelError> {
        let mut row = vec![0u16; self.cfg.vocab];
        self.logits.download_range(0, &mut row)?;
        Ok(row.into_iter().map(|v| f32::from_bits((v as u32) << 16)).collect())
    }

    /// Start a prefix-cache pool of `slots` snapshot slots (including the draft's window when a draft is attached: call this after
    /// `model.draft` is set). Allocates `slots x SnapPool::bytes_per_slot` once; errors when a pool exists already.
    pub fn enable_prefix_cache(&mut self, slots: usize) -> Result<(), ModelError> {
        if self.snaps.is_some() {
            return Err(ModelError::Config("the prefix cache is already enabled".into()));
        }
        self.snaps = Some(SnapPool::new(&self.cfg, slots, self.draft.is_some())?);
        Ok(())
    }

    /// Copy the slot's CURRENT recurrent state and conv history into snapshot slot `snap`, of a slot that may be mid-conversation: a decode
    /// round's accepted rows are owed to `S` until the next window kernel folds them in (B2), so they are applied first (`settle_slot`) and the
    /// copy is the state the committed tokens define. The serving path never needs this (prefill snapshots are taken from a slot that owes
    /// nothing); it is the primitive a snapshot of a running sequence would use, and the test that holds the settle in place.
    pub fn snapshot_state(&mut self, slot: usize, snap: u32) -> Result<(), ModelError> {
        self.flush()?;
        self.settle_slot(slot)?;
        let Model { pools, snaps, stream, cfg, .. } = self;
        let sn = snaps.as_mut().ok_or_else(|| ModelError::Request("no snapshot pool".into()))?;
        for li in 0..cfg.linear_layers() {
            copy_gdn_slot(&mut sn.gdn, snap as usize, pools, slot, li, stream, true)?;
        }
        Ok(())
    }

    /// Take references to the KV pages of the first `len` tokens of `slot` (a snapshot at `len` keeps them alive after the slot is gone) and return
    /// them. The last page is partial when `len` is not a multiple of the page size; the slot goes on writing into it above `len`, which the
    /// snapshot never reads, and a restore copies it (copy-on-write at attach).
    pub fn retain_prefix_pages(&mut self, slot: usize, len: usize) -> Vec<u32> {
        self.kv_tab.retain_prefix(slot, len.div_ceil(BLOCK_TOKENS))
    }

    /// Drop page references taken with [`Model::retain_prefix_pages`] (without touching a snapshot slot).
    pub fn release_pages(&mut self, pages: &[u32]) {
        self.kv_tab.release_pages(pages);
    }

    /// Give back a snapshot: its page references and its slot.
    pub fn release_prefix(&mut self, snap: u32, pages: &[u32]) {
        self.kv_tab.release_pages(pages);
        if let Some(sn) = self.snaps.as_mut() {
            sn.free_slot(snap);
        }
    }

    /// Put `slot` in the state a prefill of the snapshot's first `len` tokens would have left it in (see `gb_engine::Backend::prefix_restore`):
    /// the slot starts over (its old pages go back, its recurrent state is replaced, nothing is owed, rope delta 0), the snapshot's full pages are
    /// shared by reference, its partial last page is copied into a page the slot owns, and the recurrent state, conv history and draft window
    /// are copied in. Stream-ordered: a prefill queued after this sees all of it.
    pub fn restore_prefix(&mut self, slot: usize, snap: u32, pages: &[u32], len: usize) -> Result<(), ModelError> {
        let meta = self
            .snaps
            .as_ref()
            .and_then(|sn| sn.meta(snap))
            .filter(|m| m.len == len && pages.len() == len.div_ceil(BLOCK_TOKENS))
            .ok_or_else(|| ModelError::Request(format!("snapshot {snap} is not a snapshot of {len} tokens")))?;
        // begin_slot flushes the pending commits, forgets the slot's owed rows (its old sequence is gone: nothing may reach the state we
        // are about to write), zeroes its state and gives its pages back
        self.begin_slot(slot)?;
        let full = len / BLOCK_TOKENS;
        self.kv_tab.adopt_shared(slot, &pages[..full]);
        if full < pages.len() {
            let src = pages[full] as usize;
            let dst = self
                .kv_tab
                .alloc_block(slot)
                .map_err(|e| ModelError::Request(format!("no page for the copy of the last prefix page: {e:?}")))?
                as usize;
            for kv in &mut self.kv {
                kv.k.copy_within(dst * PagedKv::BLOCK_BYTES, src * PagedKv::BLOCK_BYTES, PagedKv::BLOCK_BYTES, &self.stream)?;
                kv.v.copy_within(dst * PagedKv::BLOCK_BYTES, src * PagedKv::BLOCK_BYTES, PagedKv::BLOCK_BYTES, &self.stream)?;
            }
        }
        let Model { pools, snaps, stream, cfg, draft, .. } = self;
        let sn = snaps.as_mut().expect("checked above");
        for li in 0..cfg.linear_layers() {
            copy_gdn_slot(&mut sn.gdn, snap as usize, pools, slot, li, stream, false)?;
        }
        if let Some(dr) = draft.as_mut() {
            dr.load_window(slot, meta.lo, meta.n, &sn.dk, &sn.dv, snap as usize, stream)?;
        }
        Ok(())
    }

    /// The `S` and conv history of every linear layer of `slot`, and the paged KV bytes of its first `len` tokens, hashed (FNV-1a over the
    /// raw bytes), for tests that compare two states bit for bit: settles first, so a window's owed rows are in `S`. The hash of a slot
    /// that was restored from a snapshot equals the hash of the slot that was snapshotted.
    pub fn state_hashes(&mut self, slot: usize, len: usize) -> Result<StateHashes, ModelError> {
        self.settle()?;
        self.stream.sync()?;
        let fnv = |h: &mut u64, bytes: &[u8]| {
            for &b in bytes {
                *h = (*h ^ b as u64).wrapping_mul(0x100_0000_01b3);
            }
        };
        let (mut hs, mut hc, mut hk) = (0xcbf2_9ce4_8422_2325u64, 0xcbf2_9ce4_8422_2325u64, 0xcbf2_9ce4_8422_2325u64);
        let (sb, cb) = (gdn::S_SLOT_ELEMS * 4, gdn::CONV_SLOT_ELEMS * 2);
        let mut buf = vec![0u8; sb.max(cb)];
        for li in 0..self.cfg.linear_layers() {
            self.pools.s[li].download_range(slot * sb, &mut buf[..sb])?;
            fnv(&mut hs, &buf[..sb]);
            self.pools.conv[li].download_range(slot * cb, &mut buf[..cb])?;
            fnv(&mut hc, &buf[..cb]);
        }
        let row = attention::KV_TOKEN_ELEMS * 2;
        let blocks = self.kv_tab.blocks_of(slot).to_vec();
        let mut page = vec![0u8; BLOCK_TOKENS * row];
        for kv in &self.kv {
            for (i, &blk) in blocks.iter().enumerate().take(len.div_ceil(BLOCK_TOKENS)) {
                let n = (len - i * BLOCK_TOKENS).min(BLOCK_TOKENS) * row;
                kv.k.download_range(blk as usize * PagedKv::BLOCK_BYTES, &mut page[..n])?;
                fnv(&mut hk, &page[..n]);
                kv.v.download_range(blk as usize * PagedKv::BLOCK_BYTES, &mut page[..n])?;
                fnv(&mut hk, &page[..n]);
            }
        }
        Ok(StateHashes { s: hs, conv: hc, kv: hk })
    }

    /// Prefill several whole short prompts in ONE forward: the rows of all of them share one pass of every EXL3 layer (the GEMV route, the
    /// row-invariant decode kernel: a row's bits do not depend on the other rows), while the recurrent chain, the attention and the KV
    /// writes run on each prompt's own state and slot. Each prompt starts at position 0 of its own (reset) slot; the arithmetic is the one
    /// `prefill_chunk` gives a prompt of that length alone, so every slot ends holding the bits a solo prefill leaves, and row `j` of the
    /// returned candidates (the top `want` of the prompt's LAST row, `want` at most 64) is what `prefill_chunk(.., Some(want), ..)` returns.
    ///
    /// A prompt may have at most [`PrefillPath::GEMV_MAX_PROMPT`] tokens, the prompts' rows together at most [`MAX_DECODE_ROWS`], and there
    /// are at most `window_rows` of them (the head runs once over their last rows); slots must differ. Anything else is a `Request` error
    /// before the model is touched.
    pub fn prefill_batch(&mut self, prompts: &[BatchPrompt<'_>], want: usize) -> Result<Candidates, ModelError> {
        let rows: usize = prompts.iter().map(|p| p.tokens.len()).sum();
        let bad = |why: String| Err(ModelError::Request(format!("a batched prefill: {why}")));
        if prompts.is_empty() || prompts.len() > self.window_rows {
            return bad(format!("takes 1..={} prompts, got {}", self.window_rows, prompts.len()));
        }
        if rows > MAX_DECODE_ROWS.min(self.s.rows) {
            return bad(format!("{rows} rows in all, at most {}", MAX_DECODE_ROWS.min(self.s.rows)));
        }
        if want == 0 || want > MAX_CANDIDATES {
            return bad(format!("candidate count must be 1..={MAX_CANDIDATES}"));
        }
        for (j, p) in prompts.iter().enumerate() {
            if p.tokens.is_empty() || p.tokens.len() > PrefillPath::GEMV_MAX_PROMPT {
                return bad(format!(
                    "prompt {j} has {} tokens, the route takes 1..={}",
                    p.tokens.len(),
                    PrefillPath::GEMV_MAX_PROMPT
                ));
            }
            if p.slot >= self.pools.slots || prompts[..j].iter().any(|q| q.slot == p.slot) {
                return bad(format!("prompt {j}: slot {} is out of range or used twice", p.slot));
            }
        }
        for p in prompts {
            self.begin_slot(p.slot)?;
        }
        let reqs: Vec<gb_engine::RowRequest> = prompts
            .iter()
            .map(|p| gb_engine::RowRequest { slot: p.slot, tokens: p.tokens.to_vec(), first_position: 0 })
            .collect();
        let b = RowBatch::from_requests(&reqs);
        self.forward_mm(&b, Mode::Prefill(PrefillPath::Gemv), &[])?;
        if let Some(dr) = self.draft.as_mut() {
            // the draft sees at most its window of context: skip rows far behind the end of each prompt (one pass for all of them)
            let items: Vec<Absorb> = prompts
                .iter()
                .enumerate()
                .map(|(j, p)| {
                    let skip = p.ctx_from.min(p.tokens.len());
                    Absorb { row0: b.seq_start[j] as usize + skip, n: p.tokens.len() - skip, slot: p.slot, pos0: skip }
                })
                .collect();
            dr.absorb_batch(&self.s.taps, &items, &self.stream)?;
        }
        let d = self.cfg.hidden;
        {
            // the head wants one row per prompt: move each prompt's last normed row to a row of a free buffer
            let Model { w, s, stream, logits, .. } = self;
            for j in 0..prompts.len() {
                let last = b.seq_start[j + 1] as usize - 1;
                unsafe {
                    glue::copy_rows(
                        (s.h.ptr() as *const u16).add(last * d) as *const c_void,
                        d,
                        (s.r.ptr() as *mut u16).add(j * d) as *mut c_void,
                        d,
                        1,
                        d,
                        stream,
                    )?;
                }
            }
            w.lm_head.forward(&s.r, logits, prompts.len(), Dtype::BF16, Dtype::BF16, &mut s.ws, stream)?;
        }
        self.candidates_of(prompts.len(), want)
    }

    /// Record that `keep` rows of `slot`'s last window were accepted. Applied (for all slots at once) before the next
    /// forward or slot reset.
    pub fn commit(&mut self, slot: usize, keep: usize) {
        // a keep past the window is clamped to its rows (the kernels clamp it too)
        let rows = self
            .last_window
            .as_ref()
            .and_then(|w| w.seq_slot.iter().position(|&s| s as usize == slot).map(|j| w.seq_rows(j).len()))
            .unwrap_or(keep);
        self.pending.insert(slot, (0..keep.min(rows) as i32).collect());
    }

    /// Record that the rows `path` of `slot`'s last window were accepted: window-local row indices, root first, each the parent of the
    /// next (for a plain window a prefix `0..n`, which is [`Model::commit`]). Applied, like a commit, before the next forward or slot reset.
    /// Refused when `path` is not a root path of the window's tree.
    pub fn commit_path(&mut self, slot: usize, path: &[usize]) -> Result<(), ModelError> {
        if let Some(w) = &self.last_window {
            if let Some(j) = w.seq_slot.iter().position(|&s| s as usize == slot) {
                check_path(path, w.seq_rows(j).len(), self.last_trees.as_ref().map(|t| &t[j]))?;
            }
        }
        self.pending.insert(slot, path.iter().map(|&r| r as i32).collect());
        Ok(())
    }

    /// Apply the pending commits. The conv history of each sequence is rebuilt at once (the next window's `pre` reads it); the
    /// replay of its accepted rows into the delta-rule state S is recorded as owed and folded into the sequence's next window
    /// kernel ([`gdn::window_owed`]), which reads S once and writes it once. The draft's context grows by the accepted rows now
    /// (the tap rows are overwritten by the next forward). The commits are forgotten as soon as the work starts, so a failure here
    /// (whatever its CUDA code) leaves recurrent state that no longer matches the tokens committed: it is reported as
    /// `StateLost`, which is fatal.
    pub fn flush(&mut self) -> Result<(), ModelError> {
        self.flush_pending().map_err(|e| ModelError::StateLost(e.to_string()))
    }

    fn flush_pending(&mut self) -> Result<(), ModelError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let Some(win) = self.last_window.take() else {
            self.pending.clear();
            return Ok(());
        };
        let trees = self.last_trees.take();
        let local: Vec<Vec<i32>> =
            win.seq_slot.iter().map(|&sl| self.pending.get(&(sl as usize)).cloned().unwrap_or_default()).collect();
        self.pending.clear();
        // what was committed must be a root path of what was verified (a tree window's rows are not a chain)
        for (j, p) in local.iter().enumerate() {
            let path: Vec<usize> = p.iter().map(|&r| r as usize).collect();
            check_path(&path, win.seq_rows(j).len(), trees.as_ref().map(|t| &t[j]))?;
        }
        let keep: Vec<i32> = local.iter().map(|p| p.len() as i32).collect();
        // the accepted rows as global rows of the window (a plain window's prefix is the same list)
        let global: Vec<Vec<i32>> =
            local.iter().enumerate().map(|(j, p)| p.iter().map(|&r| win.seq_start[j] + r).collect()).collect();
        let tree_window = trees.is_some();
        if keep.iter().any(|&k| k > 0) {
            // every window folds in or replays what its predecessor owed, so nothing is still owed here. If something were, the window
            // that just ran read a state that lacks those rows and its outputs are already wrong: say so (a `debug_assert` is gone in
            // a release build, and the next line would overwrite the rows)
            if self.owed.is_some() {
                return Err(ModelError::StateLost("a window ran without taking the previous window's owed replay".into()));
            }
            if tree_window {
                let n_seq = win.n_seq();
                let mut flat = vec![0i32; n_seq.max(1) * PATH_STRIDE];
                for (j, g) in global.iter().enumerate() {
                    flat[j * PATH_STRIDE..j * PATH_STRIDE + g.len()].copy_from_slice(g);
                }
                // the KV rows of the accepted path: window row -> its position (the root's plus its place on the path)
                let (mut src, mut slot, mut pos) = (Vec::new(), Vec::new(), Vec::new());
                for (j, g) in global.iter().enumerate() {
                    let root_pos = win.positions[win.seq_start[j] as usize];
                    for (k, &r) in g.iter().enumerate() {
                        src.push(r);
                        slot.push(win.seq_slot[j]);
                        pos.push(root_pos + k as i32);
                    }
                }
                let tr = self.tree.as_mut().ok_or_else(|| ModelError::Request("a tree commit needs the tree buffers".into()))?;
                tr.cm_keep.upload(&keep)?;
                tr.cm_path.upload(&flat)?;
                tr.kv_src.upload(&src)?;
                tr.kv_slot.upload(&slot)?;
                tr.kv_pos.upload(&pos)?;
                let Model { s, pools, win_qkv, stream, tree, kv, table, .. } = self;
                let tr = tree.as_ref().expect("checked above");
                let batch = Batch { seq_start: &s.seq_start, slot: &s.seq_slot, n_seq, rows: win.rows, n_slots: pools.slots };
                for (qkv, conv) in win_qkv.iter().zip(pools.conv.iter_mut()) {
                    gdn::commit_conv_path(
                        stream,
                        qkv,
                        conv,
                        &batch,
                        &tr.cm_keep,
                        PathRows { rows: &tr.cm_path, stride: PATH_STRIDE },
                    )?;
                }
                for (fi, cache) in kv.iter_mut().enumerate() {
                    attention::kv_write_rows(
                        &tr.kn[fi],
                        &tr.vn[fi],
                        &tr.kv_src,
                        &tr.kv_slot,
                        &tr.kv_pos,
                        table,
                        cache,
                        src.len(),
                        stream,
                    )?;
                }
            } else {
                self.s.keep.upload(&keep)?;
                let Model { s, pools, win_qkv, stream, .. } = self;
                let batch = Batch {
                    seq_start: &s.seq_start,
                    slot: &s.seq_slot,
                    n_seq: win.n_seq(),
                    rows: win.rows,
                    n_slots: pools.slots,
                };
                for (qkv, conv) in win_qkv.iter().zip(pools.conv.iter_mut()) {
                    gdn::commit_conv(stream, qkv, conv, &batch, &s.keep)?;
                }
            }
            self.owed = Some(OwedWindow {
                par: self.rec_par,
                win: win.clone(),
                keep: keep.clone(),
                paths: tree_window.then(|| global.clone()),
            });
        }
        // the draft's context grows by exactly the rows the target committed (anchor + accepted drafts), in path order: a path's tap rows
        // are not consecutive in a tree window, so each maximal run of consecutive rows is one item at its place on the path
        if let Some(dr) = self.draft.as_mut() {
            let mut items: Vec<Absorb> = Vec::new();
            for (j, g) in global.iter().enumerate().filter(|(_, g)| !g.is_empty()) {
                let root_pos = win.positions[win.seq_start[j] as usize] as usize;
                for (first, len, off) in runs_of(g) {
                    items.push(Absorb { row0: first, n: len, slot: win.seq_slot[j] as usize, pos0: root_pos + off });
                }
            }
            dr.absorb_batch(&self.s.taps, &items, &self.stream)?;
        }
        Ok(())
    }

    /// Seven draft tokens for the sequence in `slot` whose tokens so far are `tokens` (the last one not yet fed).
    pub fn propose_drafts(
        &mut self,
        slot: usize,
        tokens: &[u32],
        sampling: &gb_sampling::Sampling,
    ) -> Result<Option<Vec<u32>>, ModelError> {
        Ok(self.propose_drafts_batch(&[(slot, tokens, sampling)])?.pop().flatten())
    }

    /// `propose_drafts` for several sequences in as few draft rounds as the draft's batch width allows (one round for up
    /// to `Draft::max_seqs`). One entry per request, `None` when there is no draft or the sequence has no tokens.
    pub fn propose_drafts_batch(
        &mut self,
        reqs: &[(usize, &[u32], &gb_sampling::Sampling)],
    ) -> Result<Vec<Option<Vec<u32>>>, ModelError> {
        Ok(self.propose_drafts_batch_scored(reqs)?.into_iter().map(|o| o.map(|p| p.tokens)).collect())
    }

    /// [`Model::propose_drafts_batch`] with the draft's confidence in each token (see [`Picked`]).
    pub fn propose_drafts_batch_scored(
        &mut self,
        reqs: &[(usize, &[u32], &gb_sampling::Sampling)],
    ) -> Result<Vec<Option<Picked>>, ModelError> {
        let asks: Vec<(usize, &[u32], &gb_sampling::Sampling, usize)> =
            reqs.iter().map(|&(slot, tokens, sampling)| (slot, tokens, sampling, crate::draft::DRAFTS)).collect();
        self.propose_drafts_batch_want(&asks)
    }

    /// [`Model::propose_drafts_batch_scored`] where each request says how many drafts it wants (1..=7; 0 leaves the sequence out
    /// of the round and gives it `None`). The draft's layers still run a whole block per sequence in the round, but its vocabulary
    /// head, top-k and walk run only for the rows wanted, and each sequence's drafts are the first ones a full round would give it.
    pub fn propose_drafts_batch_want(
        &mut self,
        reqs: &[(usize, &[u32], &gb_sampling::Sampling, usize)],
    ) -> Result<Vec<Option<Picked>>, ModelError> {
        let mut out: Vec<Option<Picked>> = vec![None; reqs.len()];
        if self.draft.is_none() || reqs.is_empty() {
            return Ok(out);
        }
        let live: Vec<usize> = (0..reqs.len()).filter(|&i| !reqs[i].1.is_empty() && reqs[i].3 > 0).collect();
        if live.is_empty() {
            return Ok(out);
        }
        self.flush()?;
        let Model { draft, w, cfg, stream, .. } = self;
        let d = draft.as_mut().unwrap();
        for group in live.chunks(d.max_seqs) {
            let dr: Vec<DraftReq> = group
                .iter()
                .map(|&i| {
                    let (slot, tokens, sampling, want) = reqs[i];
                    let recent = &tokens[tokens.len().saturating_sub(crate::draft::COVERAGE_WINDOW)..];
                    let coverage = d.coverage(recent);
                    DraftReq { anchor: tokens[tokens.len() - 1], slot, p: tokens.len() - 1, sampling, want, coverage }
                })
                .collect();
            let t = d.propose_batch_scored(&w.embed, &w.lm_head, cfg.vocab, &dr, None, stream)?;
            for (&i, picked) in group.iter().zip(t) {
                out[i] = Some(picked);
            }
        }
        Ok(out)
    }

    /// Wait for everything queued so far.
    pub fn sync(&self) -> Result<(), ModelError> {
        self.stream.sync()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use gb_engine::TreeRequest;

    fn tree_batch() -> RowBatch {
        RowBatch::from_trees(&[
            // 0 -> {1, 2}, 1 -> 3, 3 -> 4 (rows 1 and 2 share depth 1)
            TreeRequest { slot: 2, tokens: vec![9, 8, 7, 6, 5], parents: vec![-1, 0, 0, 1, 3], first_position: 300 },
            // a chain of three
            TreeRequest { slot: 0, tokens: vec![1, 2, 3], parents: vec![-1, 0, 1], first_position: 40 },
        ])
    }

    /// The tree kernels read one parent, one committed length and one root-to-row path per row, and one plan per sequence.
    #[test]
    fn a_window_of_trees_lays_out_parents_paths_and_plans() {
        let lay = tree_layout(&tree_batch()).unwrap();
        assert_eq!(lay.parent, vec![-1, 0, 0, 1, 3, -1, 5, 6], "global parents, roots -1");
        assert_eq!(lay.base, vec![300, 300, 300, 300, 300, 40, 40, 40], "the committed length of each row's sequence");
        let path = |g: usize| {
            lay.paths[g * PATH_STRIDE..(g + 1) * PATH_STRIDE].iter().copied().take_while(|&x| x >= 0).collect::<Vec<_>>()
        };
        assert_eq!(path(0), vec![0]);
        assert_eq!(path(2), vec![0, 2]);
        assert_eq!(path(4), vec![0, 1, 3, 4], "ancestors in depth order, global rows");
        assert_eq!(path(7), vec![5, 6, 7], "the second sequence's rows start at 5");
        assert_eq!(lay.slots, 1, "one fork needs one saved state");
        let plan = |i: usize| lay.plan[i * PLAN_STRIDE..i * PLAN_STRIDE + 15].to_vec();
        // sequence 0: the root saves its state for its second child; equal needs keep the row order, so the chain 1 -> 3 -> 4 runs
        // first from registers and the leaf (row 2) reads the slot last
        assert_eq!(&plan(0)[..6], &[0, -1, 0, 1, -2, -1], "root saves, row 1 from registers");
        assert_eq!(&plan(0)[6..15], &[3, -2, -1, 4, -2, -1, 2, 0, -1], "the leaf reads the root's slot");
        // sequence 1 is a chain: no saves, each row from the registers
        assert_eq!(&lay.plan[PLAN_STRIDE..PLAN_STRIDE + 9], &[0, -1, -1, 1, -2, -1, 2, -2, -1]);
    }

    #[test]
    fn a_tree_row_must_sit_at_its_depth_and_a_tree_must_be_well_formed() {
        let mut b = tree_batch();
        b.positions[2] = 302; // row 2 is at depth 1: its position is 301, not the row index's
        let e = tree_layout(&b).unwrap_err().to_string();
        assert!(e.contains("depth"), "{e}");
        let mut b = tree_batch();
        b.parents.as_mut().unwrap()[1] = 3; // a parent that comes later
        assert!(tree_layout(&b).is_err());
        let mut b = tree_batch();
        b.parents.as_mut().unwrap()[3] = -1; // a second root
        assert!(tree_layout(&b).is_err());
        // a plain batch is chains, and is fine
        let chain = RowBatch::from_requests(&[gb_engine::RowRequest { slot: 1, tokens: vec![1, 2, 3, 4], first_position: 9 }]);
        let lay = tree_layout(&chain).unwrap();
        assert_eq!(lay.parent, vec![-1, 0, 1, 2]);
        assert_eq!(lay.slots, 0);
    }

    /// What a commit may name: a root path of the window's tree (a chain's rows for a plain window), nothing else.
    #[test]
    fn a_committed_path_must_be_a_root_path() {
        let t = Tree::new(&[-1, 0, 0, 1, 3]).unwrap();
        assert!(check_path(&[0, 1, 3, 4], 5, Some(&t)).is_ok());
        assert!(check_path(&[0, 2], 5, Some(&t)).is_ok());
        assert!(check_path(&[], 5, Some(&t)).is_ok());
        assert!(check_path(&[1, 3], 5, Some(&t)).is_err(), "not from the root");
        assert!(check_path(&[0, 2, 3], 5, Some(&t)).is_err(), "3 is not a child of 2");
        assert!(check_path(&[0, 1, 4], 5, Some(&t)).is_err(), "4 is not a child of 1");
        assert!(check_path(&[0, 5], 5, Some(&t)).is_err(), "past the window");
        // without a tree the window is a chain: only a prefix
        assert!(check_path(&[0, 1, 2], 4, None).is_ok());
        assert!(check_path(&[0, 2], 4, None).is_err());
    }

    #[test]
    fn accepted_rows_split_into_runs_of_consecutive_rows() {
        assert_eq!(runs_of(&[5, 6, 7]), vec![(5, 3, 0)]);
        assert_eq!(runs_of(&[0, 1, 3, 4, 9]), vec![(0, 2, 0), (3, 2, 2), (9, 1, 4)]);
        assert_eq!(runs_of(&[]), Vec::<(usize, usize, usize)>::new());
    }

    /// The prefill path is a function of the prompt's length alone: the GEMV up to the decode kernel's row limit, the pinned GEMM
    /// above it, whatever the chunk.
    #[test]
    fn prefill_path_is_picked_by_the_prompts_total_length() {
        assert_eq!(PrefillPath::GEMV_MAX_PROMPT, 128);
        assert_eq!(PrefillPath::GEMV_MAX_PROMPT, gb_cuda::exl3_linear::MAX_DECODE_ROWS);
        for total in [0usize, 1, 100, 127, 128] {
            assert_eq!(PrefillPath::for_prompt(total), PrefillPath::Gemv, "total {total}");
        }
        for total in [129usize, 130, 300, 4096, 4097, 4200, 40_000] {
            assert_eq!(PrefillPath::for_prompt(total), PrefillPath::Tiled, "total {total}");
        }
    }

    /// The window cap is a kernel limit: 128 rows is the most one decode call takes, so a model is never built for more, and the
    /// served 64 is inside it.
    #[test]
    fn a_window_cap_past_the_kernels_is_refused_when_the_model_is_built() {
        assert!(check_window_rows(WINDOW_ROWS).is_ok());
        for ok in [1usize, 64, 65, 96, 128] {
            assert!(check_window_rows(ok).is_ok(), "{ok}");
        }
        for bad in [0usize, 129, 130, 256, usize::MAX] {
            match check_window_rows(bad) {
                Err(ModelError::Config(m)) => assert!(m.contains("1..=128"), "{m}"),
                other => panic!("{bad}: {other:?}"),
            }
        }
        const _: () = assert!(WINDOW_ROWS <= MAX_DECODE_ROWS);
    }
}
