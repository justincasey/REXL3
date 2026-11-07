//! CUDA graphs of the decode windows: one captured graph per window shape, replayed with new inputs, eager launches for everything else.
//!
//! A window (one decode row, a verify window of 2..9 rows, several sequences' rows) is the same list of ~1,600 launches every time; only the
//! values the kernels READ from device memory change between two windows of one shape: the token ids, the rows' slots and positions, the page
//! table, the sequence layout (`RowBatch` uploads, written in place into the model's own buffers before every replay), the KDA state and the
//! caches. A graph bakes in every other argument. The audit of what `run()` hands a launcher that is NOT read from a device buffer:
//!
//! | host value | what it decides | in the key / made constant |
//! |---|---|---|
//! | `rows` | every launch's row count and grid | key |
//! | `n_seq` | the KDA kernels' sequence count and grid (their `seq_start` / `slot` arrays are device buffers) | key |
//! | `any_sparse` (some row's position is 2,048 or more) | whether the indexer's `weights_proj`, `wq_b`, scoring and selection launch, and whether the sparse attention launches | key |
//! | `n_pools_max` | the indexer scoring grid | the scratch's capacity in the sparse regime (the kernel's blocks past a row's own pools do nothing; the doc of `dsa` names this the graph caller's value), unused in the dense one |
//! | `mla_ctx` | the dense attention's chunk grid and the gate of the sparse launch | 2,048 in the dense regime (8 chunks, the extra ones are no-ops), the model's context in the sparse one |
//! | head (`AllRows`, none) | the schedule's tail | key |
//! | MTP present, mutants, capture, forced routing, profiling, splice, oracle rounding | extra launches and host work in the executor | a window with any of them set runs eagerly |
//!
//! Both the graph and its eager fallback take the constants of the table (`RunShape`), so a shape that fails to capture, or that has not been seen
//! twice yet, runs the same launches the graph would have replayed.
//!
//! **Policy.** The first window of a shape runs eagerly (it also warms every kernel's lazy initialisation), the second captures it and replays it,
//! every later one replays. A capture that fails for any non-sticky reason (the runtime refusing a call, a launcher's argument error) marks the shape
//! eager for the rest of the process and runs the window eagerly: a capture never fails a request and never latches the backend fatal; only a sticky
//! device error (`CudaError::is_fatal` on a code that is not a capture refusal) does, exactly as for a launch. The cache is bounded (`GRAPH_LIMIT`
//! instantiated graphs, least recently used out), and a graph is destroyed only after the stream has drained.

use super::{GlmModel, Head, Mode};
use crate::error::ModelError;
use crate::families::glm::ops::Op;
use crate::rows::RowBatch;
use gb_cuda::graph::{Capture, CaptureError, Graph};
use gb_cuda::{dsa, Stream};
use std::collections::HashMap;
use std::time::Instant;

/// Instantiated graphs kept at once. An instantiated window graph holds about 9 KB of driver memory per launch (two thirds of it host memory,
/// measured on GB10 with CUDA 13.0, whatever the kernels): 12 to 20 MiB for a window's 1,200 to 1,650 launches, held until the graph is evicted
/// and outside every plan. 96 graphs held up to 1.6 GiB; the bench's 5,067 windows touch 80 shapes, 55 of them more than once, and at 32 graphs
/// they cost 24 more captures of about 7 ms each (0.2 s over the whole run) for about 1 GiB less memory.
pub const GRAPH_LIMIT: usize = 32;
/// Instantiated graphs of NextN draft calls kept at once (their own cache beside the windows').
pub const DRAFT_GRAPH_LIMIT: usize = 32;

/// `REXL3_GLM_DRAFT_GRAPHS`: whether a NextN draft call (its absorb, then its head and chain) runs through graphs of its own (`0` / `off`: eager,
/// `1` / `on`: graphs). Unset: on for a rank of a tensor-parallel group, off on one node (whose draft calls keep their eager launches). Only
/// with window graphs on; a fact of the TP Hello (`GlmFacts::draft_graphs`).
pub const ENV_DRAFT: &str = "REXL3_GLM_DRAFT_GRAPHS";

/// The draft graphs setting: `Some(on)` when [`ENV_DRAFT`] says, `None` (the default for the kind of process) when unset or empty.
pub fn draft_graphs_setting() -> Option<bool> {
    match std::env::var(ENV_DRAFT).as_deref().map(str::trim) {
        Ok("0" | "off" | "false" | "no") => Some(false),
        Ok("1" | "on" | "true" | "yes") => Some(true),
        _ => None,
    }
}

/// What identifies a captured part of a NextN draft call: every host value its launches take (`model/mtp.rs`, `mtp_draft_group`). The
/// ABSORB part is one forward of the layer over `rows` rows (`sparse`: its regime, bit 0); the CHAIN part is the head over `rows` (the
/// asking sequences) then `depth - 1` chained steps of a layer row and a head each (`sparse`: bit `k - 1` the regime of chained step `k`),
/// with the head's vocabulary (`restricted`) and the keyed pick's candidates (`pick`, 0: the argmax).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DraftKey {
    pub chain: bool,
    pub rows: usize,
    pub depth: usize,
    pub restricted: bool,
    pub pick: usize,
    pub sparse: u32,
}

/// The environment variable that switches graphs: `0` / `off` is off, anything else (and unset) is on.
pub const ENV: &str = "REXL3_GLM_GRAPHS";

/// `REXL3_GLM_GRAPH_AT=0` captures a window shape on its FIRST sighting instead of its second (a test knob: the whole GPU test battery then runs through
/// captured graphs, the first window of every shape included, which is how the model-level tests of the window and of the backend exercise the replay).
pub const ENV_AT: &str = "REXL3_GLM_GRAPH_AT";

/// What identifies a captured window: every host-side value that reaches a launcher and is not read from a device buffer (see the table above).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GraphKey {
    pub rows: usize,
    pub n_seq: usize,
    pub sparse: bool,
    pub all_rows: bool,
}

/// The launch arguments a window takes that a context-length-independent graph cannot hold as the real value (see the module table).
#[derive(Clone, Copy, Debug)]
pub(super) struct RunShape {
    pub n_pools_max: usize,
    pub any_sparse: bool,
    pub mla_ctx: usize,
}

impl RunShape {
    pub(super) fn for_window(any_sparse: bool, s_stride: usize, max_context: usize) -> Self {
        if any_sparse {
            Self { n_pools_max: s_stride, any_sparse, mla_ctx: max_context }
        } else {
            Self { n_pools_max: 0, any_sparse, mla_ctx: max_context.min(gb_cuda::mla::DENSE_MAX_KEYS) }
        }
    }
}

/// Whether an error a launch returned while capturing is the device's (fatal, as for any launch) and not the capture's own refusal: the 900-series
/// codes ("this call cannot be captured") spoil the capture only, whatever `CudaError::is_fatal` says of a code it does not know.
fn closure_is_fatal(e: &ModelError) -> bool {
    match e {
        ModelError::Cuda(c) if gb_cuda::graph::is_capture_code(c.0) => false,
        other => other.is_fatal(),
    }
}

/// A deliberate bug of the graph path, so the tests can show they notice it (`REXL3_GRAPH_MUTANT` in `glm_perf.rs`). Never set in serving.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphMutant {
    /// A replay does not write this window's token ids, positions, slots and layout into the device buffers: the graph reads the previous window's.
    StaleRows,
    /// A replay does not write the page table: the graph reads the table of the last window that was uploaded.
    StaleTable,
    /// The window of R rows takes the graph of R - 1 rows (the key is off by one).
    WrongShape,
    /// A host synchronisation inside the capture (the capture must refuse, fall back to eager launches and never fail the window).
    SyncInCapture,
    /// The shared expert's side stream is made slow (about 0.2 ms spent after its last kernel) with the join intact: the control, the bits must not move.
    SlowSide,
    /// The join is too late (the main stream waits for the shared expert after the combine that reads its output) and the side stream is slow: the
    /// combine reads the previous layer's shared output.
    JoinLate,
    /// One wait of a layer's branches is left out and the branches are made slow (a spin kernel after each branch group), so that the consumer reads
    /// what the producer has not written yet.
    SkipWait(Wait),
    /// A replayed draft chain does not get the call's chained row index sets uploaded: its layer rows read the previous call's slots and
    /// positions (the drafts change; drafted == serial still holds, the speculation series must not).
    DraftStaleRows,
}

/// The waits of the KDA and MLA branches (`GraphMutant::SkipWait`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wait {
    /// The MLA attention waits for the latent row of the window to be in the cache.
    Latent,
    /// The indexer's scoring waits for the pool plane to be updated by the window's rows.
    Pool,
    /// The indexer's `wq_b` waits for the query chain's `q_resid`.
    Q,
    /// The sparse attention waits for the indexer's selection.
    Select,
}

/// `REXL3_GLM_GRAPH_MUTANT_SPIN=CYCLES` (a test knob): how long the `SlowSide` / `JoinLate` mutants make the shared expert's side stream
/// (default 500,000 cycles, about 0.2 ms: one node's race).
pub const ENV_MUTANT_SPIN: &str = "REXL3_GLM_GRAPH_MUTANT_SPIN";

/// `REXL3_GLM_GRAPH_MUTANT_SPIN_AT=before|after` (a test knob, default `after`): where the `SlowSide` / `JoinLate` spin goes on the side
/// stream. `after` (one node's mutant) spins after the shared expert has written its output, which delays only the join's event: on a
/// tensor-parallel rank the main stream's exchange already outlasts the shared expert before the combine, so `JoinLate` cannot lose the race
/// there whatever the spin's length. `before` spins before the shared expert's first kernel, so its output is written late: `JoinLate`'s
/// combine then reads the previous layer's shared output (the two-node digest must change), and `SlowSide` (the join in place) must not move
/// a bit.
pub const ENV_MUTANT_SPIN_AT: &str = "REXL3_GLM_GRAPH_MUTANT_SPIN_AT";

/// The side stream's slowdown of the `SlowSide` / `JoinLate` mutants, in clock cycles ([`ENV_MUTANT_SPIN`]).
pub fn slow_side_cycles() -> i64 {
    std::env::var(ENV_MUTANT_SPIN).ok().and_then(|v| v.trim().parse::<i64>().ok()).filter(|&c| c > 0).unwrap_or(500_000)
}

/// Whether the `SlowSide` / `JoinLate` spin comes before the shared expert's kernels ([`ENV_MUTANT_SPIN_AT`] = `before`).
pub fn slow_side_before() -> bool {
    std::env::var(ENV_MUTANT_SPIN_AT).is_ok_and(|v| v.trim() == "before")
}

/// `REXL3_GLM_GRAPH_MUTANT=NAME` (a test knob, NEVER set in serving): the process's graph path runs with that deliberate bug from its
/// first window, so a two-node leg can show that the TP digests notice it (lane E's graph mutants under TP; one node's tests set it
/// through [`GlmModel::set_graph_mutant`]). The names are the tests' (`stale_rows` .. `skip_select`).
pub const ENV_MUTANT: &str = "REXL3_GLM_GRAPH_MUTANT";

impl GraphMutant {
    /// The mutant of a test name, `None` for a name that is not one.
    pub fn from_name(n: &str) -> Option<GraphMutant> {
        Some(match n {
            "stale_rows" => GraphMutant::StaleRows,
            "stale_table" => GraphMutant::StaleTable,
            "wrong_shape" => GraphMutant::WrongShape,
            "sync_in_capture" => GraphMutant::SyncInCapture,
            "slow_side" => GraphMutant::SlowSide,
            "join_late" => GraphMutant::JoinLate,
            "skip_latent" => GraphMutant::SkipWait(Wait::Latent),
            "skip_pool" => GraphMutant::SkipWait(Wait::Pool),
            "skip_q" => GraphMutant::SkipWait(Wait::Q),
            "skip_select" => GraphMutant::SkipWait(Wait::Select),
            "draft_stale_rows" => GraphMutant::DraftStaleRows,
            _ => return None,
        })
    }

    /// Whether the branches of the layers are made slow on purpose.
    pub fn slows_branches(self) -> bool {
        !matches!(
            self,
            GraphMutant::StaleRows
                | GraphMutant::StaleTable
                | GraphMutant::WrongShape
                | GraphMutant::SyncInCapture
                | GraphMutant::DraftStaleRows
        )
    }
}

enum Slot {
    /// Run once eagerly; the next window of the shape captures.
    Seen,
    Ready(Graph, u64),
    /// A capture of this shape failed: eager from now on.
    Eager,
}

/// What the cache did, for tests and the server's log.
#[derive(Clone, Debug, Default)]
pub struct GraphStats {
    pub replays: u64,
    pub captures: u64,
    pub eager_windows: u64,
    pub capture_failures: u64,
    pub evictions: u64,
    pub capture_ms: f64,
    pub last_failure: Option<String>,
    /// The NextN draft calls' parts (`DraftKey`): replays, captures, eager runs and failed captures.
    pub draft_replays: u64,
    pub draft_captures: u64,
    pub draft_eager: u64,
    pub draft_capture_failures: u64,
}

/// The side streams of a layer's branches (non-blocking) and the events that order them with the main stream. A layer forks its branches after the
/// finalize that wrote the layer's input (`fin`), orders them among themselves with `q`, `b` and `c`, and joins them back into the main stream
/// (`done1`, `done2`) before the layer's last consumer: every branch is joined before the layer ends (a graph capture needs it), so the next layer
/// finds the buffers of this one finished.
pub(super) struct Side {
    pub s1: Stream,
    pub s2: Stream,
    /// recorded on the main stream: the layer's normed input (`hidden`) is ready
    pub fin: gb_cuda::rt::Event,
    /// recorded on the main stream: the MLA layer's `q_resid` is ready
    pub q: gb_cuda::rt::Event,
    /// recorded on `s1`: the MLA layer's latent row is in the cache
    pub b: gb_cuda::rt::Event,
    /// recorded on `s1`: the indexer's pool plane is updated
    pub c: gb_cuda::rt::Event,
    /// recorded on `s1` / `s2` at the end of the branch
    pub done1: gb_cuda::rt::Event,
    pub done2: gb_cuda::rt::Event,
    /// Which layer parts run as branches (`PART_*`).
    pub parts: u8,
}

/// The layer parts `REXL3_GLM_OVERLAP` runs as branches (`PART_*`; the Hello fact `overlap` of a rank with graphs).
pub fn overlap_parts() -> u8 {
    match std::env::var(ENV_OVERLAP).as_deref() {
        Ok("0" | "off" | "false" | "no") => 0,
        Ok(v) => v.parse::<u8>().map_or(PART_ALL, |m| m & PART_ALL),
        Err(_) => PART_ALL,
    }
}

/// `REXL3_GLM_GRAPH_AT=0`: shapes are captured at their first sighting (the Hello fact `graph_at_first`); otherwise at their second.
pub fn capture_at_first() -> bool {
    capture_at_first_or(false)
}

/// The capture policy with `default` when `REXL3_GLM_GRAPH_AT` is unset: `0` captures a shape at its first sighting, `1` at its second
/// (a serving group captures at the first: `tp_boot`).
pub fn capture_at_first_or(default: bool) -> bool {
    match std::env::var(ENV_AT).as_deref().map(str::trim) {
        Ok("0") => true,
        Ok("1") => false,
        _ => default,
    }
}

impl Side {
    pub(super) fn new() -> Option<Self> {
        let parts = overlap_parts();
        let ev = || gb_cuda::rt::Event::new().ok();
        Some(Self {
            s1: Stream::new_nonblocking().ok()?,
            s2: Stream::new_nonblocking().ok()?,
            fin: ev()?,
            q: ev()?,
            b: ev()?,
            c: ev()?,
            done1: ev()?,
            done2: ev()?,
            parts,
        })
    }
}

/// The environment variable that switches the layer branches: `0` / `off` is off, a number is the mask of parts (`PART_*`), anything else (and unset)
/// is all of them.
pub const ENV_OVERLAP: &str = "REXL3_GLM_OVERLAP";

/// The shared expert of a MoE layer beside the router and the routed experts.
pub const PART_SHARED: u8 = 1;
/// An MLA layer's latent and indexer chains beside its query chain.
pub const PART_MLA: u8 = 2;
pub const PART_ALL: u8 = PART_SHARED | PART_MLA;

pub(super) struct GraphCache {
    pub on: bool,
    /// Capture a shape at its first sighting (see `ENV_AT`).
    capture_first: bool,
    pub mutant: Option<GraphMutant>,
    slots: HashMap<GraphKey, Slot>,
    /// The NextN draft calls' graphs (`DraftKey`), bounded by `DRAFT_GRAPH_LIMIT`.
    draft_slots: HashMap<DraftKey, Slot>,
    tick: u64,
    /// The stream captures run on (non-blocking: a capture on the model's blocking stream would order with the legacy default stream).
    cap: Option<Stream>,
    pub stats: GraphStats,
}

impl GraphCache {
    pub(super) fn new() -> Self {
        let on = !matches!(std::env::var(ENV).as_deref(), Ok("0" | "off" | "false" | "no"));
        let mutant = std::env::var(ENV_MUTANT).ok().map(|n| {
            let m = GraphMutant::from_name(n.trim());
            eprintln!("rexl3 level=warn event=glm_graph_mutant name={n:?} parsed={m:?} (a TEST knob: this process computes wrong bits on purpose)");
            m
        });
        Self {
            on,
            capture_first: capture_at_first(),
            mutant: mutant.flatten(),
            slots: HashMap::new(),
            draft_slots: HashMap::new(),
            tick: 0,
            cap: Stream::new_nonblocking().ok(),
            stats: GraphStats::default(),
        }
    }
}

/// The key of a window: every host-side value that reaches a launcher and is not read from a device buffer (see the module table).
pub fn key_of(rows: usize, n_seq: usize, max_pos: usize, head: Head) -> GraphKey {
    GraphKey { rows, n_seq, sparse: dsa::is_sparse(max_pos), all_rows: head == Head::AllRows }
}

impl GlmModel {
    /// Switch graphs on or off for the windows that follow (eager and graphs run the same arithmetic; the digest tests run both).
    pub fn set_graphs(&mut self, on: bool) {
        self.graphs.on = on && self.graphs.cap.is_some();
    }

    /// Capture a window shape at its first sighting (`true`) or its second (the model's default, `REXL3_GLM_GRAPH_AT`). Only when a
    /// shape's graph is made moves: a replay and an eager window are the same bits.
    pub fn set_graph_capture_first(&mut self, on: bool) {
        self.graphs.capture_first = on;
    }

    /// Run the shared expert on a side stream, concurrent with the router and the routed experts, in the windows that go through the graph path. The
    /// captured graphs are forgotten (their launches change).
    pub fn set_overlap(&mut self, on: bool) -> Result<(), ModelError> {
        self.set_overlap_parts(if on { PART_ALL } else { 0 })
    }

    /// [`GlmModel::set_overlap`] for some parts only (`PART_SHARED | PART_MLA`).
    pub fn set_overlap_parts(&mut self, parts: u8) -> Result<(), ModelError> {
        self.clear_graphs()?;
        if let Some(s) = self.side.as_mut() {
            s.parts = parts & PART_ALL;
        }
        Ok(())
    }

    pub fn graphs_on(&self) -> bool {
        self.graphs.on
    }

    pub fn graph_stats(&self) -> GraphStats {
        self.graphs.stats.clone()
    }

    /// Forget every captured graph (a test of the capture path; the stream is drained first).
    pub fn clear_graphs(&mut self) -> Result<(), ModelError> {
        self.stream.sync()?;
        self.graphs.slots.clear();
        self.graphs.draft_slots.clear();
        Ok(())
    }

    /// May a forward of this shape go through the graph cache? Anything that adds host work or launches to the executor stays eager.
    pub(super) fn graph_eligible(&self, mode: Mode, head: Head) -> bool {
        self.graphs.on
            && mode == Mode::Window
            && head != Head::LastRow
            && self.capture.is_none()
            && self.forced.is_none()
            && self.profile.is_none()
            && self.mutant.is_none()
            && self.splice.is_empty()
            && !self.oracle_rounding
            && self.mtp.as_ref().is_none_or(|m| m.mutant.is_none())
            && self.tp.as_ref().is_none_or(|t| t.graphs_ready())
    }

    pub fn graph_key(&self, b: &RowBatch, head: Head, max_pos: usize) -> GraphKey {
        let mut k = key_of(b.rows, b.n_seq(), max_pos, head);
        if self.graphs.mutant == Some(GraphMutant::WrongShape) && k.rows > 1 {
            k.rows -= 1;
        }
        k
    }

    pub(super) fn graphs_mutant(&self) -> Option<GraphMutant> {
        self.graphs.mutant
    }

    /// Debug (the mutation tests): run the graph path with a deliberate bug.
    pub fn set_graph_mutant(&mut self, m: Option<GraphMutant>) {
        self.graphs.mutant = m;
    }

    /// Whether a window of this shape would replay a captured graph (the mutants that skip an upload ask it).
    pub(super) fn graph_would_replay(&self, b: &RowBatch, head: Head, max_pos: usize) -> bool {
        self.graphs.on && matches!(self.graphs.slots.get(&self.graph_key(b, head, max_pos)), Some(Slot::Ready(..)))
    }

    /// Run a window's launches through the graph cache: replay the graph of its shape, capture it on the second sighting, or run eagerly.
    pub(super) fn run_window(&mut self, plan: &[Op], b: &RowBatch, head: Head, max_pos: usize) -> Result<(), ModelError> {
        let key = self.graph_key(b, head, max_pos);
        let shape = RunShape::for_window(key.sparse, self.s.dsa.s_stride, self.max_context);
        self.run_shape = Some(shape);
        let r = self.run_window_with(plan, b, head, max_pos, key);
        self.run_shape = None;
        r
    }

    fn run_window_with(
        &mut self,
        plan: &[Op],
        b: &RowBatch,
        head: Head,
        max_pos: usize,
        key: GraphKey,
    ) -> Result<(), ModelError> {
        self.graphs.tick += 1;
        let tick = self.graphs.tick;
        match self.graphs.slots.get_mut(&key) {
            Some(Slot::Ready(g, used)) => {
                *used = tick;
                let r = g.launch(&self.stream);
                self.graphs.stats.replays += 1;
                return r.map_err(ModelError::from);
            }
            Some(Slot::Eager) => {
                self.graphs.stats.eager_windows += 1;
                return self.run(plan, b, Mode::Window, head, max_pos);
            }
            Some(Slot::Seen) => {}
            None if self.graphs.capture_first => {}
            None => {
                self.graphs.slots.insert(key, Slot::Seen);
                self.graphs.stats.eager_windows += 1;
                return self.run(plan, b, Mode::Window, head, max_pos);
            }
        }
        // second sighting: capture
        self.make_room()?;
        let t = Instant::now();
        match self.capture_window(plan, b, head, max_pos) {
            Ok(g) => {
                self.graphs.stats.captures += 1;
                let ms = t.elapsed().as_secs_f64() * 1e3;
                self.graphs.stats.capture_ms += ms;
                crate::families::glm::spec::cost_log(format_args!("G {} {} {ms:.4}", key.rows, key.n_seq));
                let r = g.launch(&self.stream);
                self.graphs.slots.insert(key, Slot::Ready(g, tick));
                r.map_err(ModelError::from)
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
                // a rank of a tensor-parallel group does not fall back alone: its peer may have captured the shape, and the ranks would
                // then issue this window's collectives by different paths (eager calls against a replay), which is not proved
                // (`model/tp.rs`). The window fails and the rank's state is lost by name; the group goes down (the fail-closed path).
                if self.tp.is_some() {
                    return Err(ModelError::StateLost(format!(
                        "a tensor-parallel rank's capture of the window rows={} n_seq={} sparse={} failed: {why} (a rank never falls back to \
                         eager launches alone; boot the group with REXL3_GLM_TP_GRAPHS=0 to run every window eagerly)",
                        key.rows, key.n_seq, key.sparse
                    )));
                }
                eprintln!(
                    "rexl3 level=warn event=graph_capture_failed rows={} n_seq={} sparse={} {why} (this window shape runs eagerly)",
                    key.rows, key.n_seq, key.sparse
                );
                self.graphs.stats.capture_failures += 1;
                self.graphs.stats.last_failure = Some(why);
                self.graphs.slots.insert(key, Slot::Eager);
                self.graphs.stats.eager_windows += 1;
                self.run(plan, b, Mode::Window, head, max_pos)
            }
        }
    }

    /// Capture `run()` on the capture stream. The executor uses `self.stream`, so the two streams trade places for the capture; nothing runs.
    fn capture_window(
        &mut self,
        plan: &[Op],
        b: &RowBatch,
        head: Head,
        max_pos: usize,
    ) -> Result<Graph, CaptureError<ModelError>> {
        let Some(mut cap_stream) = self.graphs.cap.take() else {
            return Err(CaptureError::Refused(gb_cuda::CudaError(-1)));
        };
        std::mem::swap(&mut self.stream, &mut cap_stream);
        let began: Result<Capture, gb_cuda::CudaError> = Graph::begin(&self.stream);
        let out = match began {
            Ok(c) => {
                // the mutant: a host synchronisation in the middle of the capture
                let ran = if self.graphs.mutant == Some(GraphMutant::SyncInCapture) {
                    self.stream.sync().map_err(ModelError::from).and_then(|_| self.run(plan, b, Mode::Window, head, max_pos))
                } else {
                    self.run(plan, b, Mode::Window, head, max_pos)
                };
                c.finish(ran)
            }
            Err(e) => Err(CaptureError::Refused(e)),
        };
        std::mem::swap(&mut self.stream, &mut cap_stream);
        self.graphs.cap = Some(cap_stream);
        out
    }

    /// Whether a draft part of this key would replay a captured graph (the draft mutant that skips an upload asks it).
    pub(super) fn draft_would_replay(&self, key: &DraftKey) -> bool {
        matches!(self.graphs.draft_slots.get(key), Some(Slot::Ready(..)))
    }

    /// Run a part of a NextN draft call (`f`: its launches on `self.stream`) through the draft graph cache, by the windows' policy: the first
    /// sighting of a key runs eagerly, the second captures and replays, every later one replays; a capture that fails runs eagerly from then
    /// on, except on a rank of a tensor-parallel group, which never falls back alone (the window's rule: the state is lost by name).
    pub(super) fn run_draft(
        &mut self,
        key: DraftKey,
        f: &mut dyn FnMut(&mut GlmModel) -> Result<(), ModelError>,
    ) -> Result<(), ModelError> {
        self.graphs.tick += 1;
        let tick = self.graphs.tick;
        match self.graphs.draft_slots.get_mut(&key) {
            Some(Slot::Ready(g, used)) => {
                *used = tick;
                let r = g.launch(&self.stream);
                self.graphs.stats.draft_replays += 1;
                return r.map_err(ModelError::from);
            }
            Some(Slot::Eager) => {
                self.graphs.stats.draft_eager += 1;
                return f(self);
            }
            Some(Slot::Seen) => {}
            None if self.graphs.capture_first => {}
            None => {
                self.graphs.draft_slots.insert(key, Slot::Seen);
                self.graphs.stats.draft_eager += 1;
                return f(self);
            }
        }
        self.make_draft_room()?;
        let t = Instant::now();
        let captured = {
            match self.graphs.cap.take() {
                None => Err(CaptureError::Refused(gb_cuda::CudaError(-1))),
                Some(mut cap_stream) => {
                    std::mem::swap(&mut self.stream, &mut cap_stream);
                    let out = match Graph::begin(&self.stream) {
                        Ok(c) => {
                            let ran = f(self);
                            c.finish(ran)
                        }
                        Err(e) => Err(CaptureError::Refused(e)),
                    };
                    std::mem::swap(&mut self.stream, &mut cap_stream);
                    self.graphs.cap = Some(cap_stream);
                    out
                }
            }
        };
        match captured {
            Ok(g) => {
                self.graphs.stats.draft_captures += 1;
                self.graphs.stats.capture_ms += t.elapsed().as_secs_f64() * 1e3;
                let r = g.launch(&self.stream);
                self.graphs.draft_slots.insert(key, Slot::Ready(g, tick));
                r.map_err(ModelError::from)
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
                if self.tp.is_some() {
                    return Err(ModelError::StateLost(format!(
                        "a tensor-parallel rank's capture of the draft part {key:?} failed: {why} (a rank never falls back to eager launches \
                         alone; boot the group with {ENV_DRAFT}=0 to run the draft calls eagerly)"
                    )));
                }
                eprintln!("rexl3 level=warn event=graph_capture_failed draft={key:?} {why} (this draft part runs eagerly)");
                self.graphs.stats.draft_capture_failures += 1;
                self.graphs.stats.last_failure = Some(why);
                self.graphs.draft_slots.insert(key, Slot::Eager);
                self.graphs.stats.draft_eager += 1;
                f(self)
            }
        }
    }

    /// [`GlmModel::make_room`] for the draft graphs.
    fn make_draft_room(&mut self) -> Result<(), ModelError> {
        let ready = self.graphs.draft_slots.values().filter(|s| matches!(s, Slot::Ready(..))).count();
        if ready < DRAFT_GRAPH_LIMIT {
            return Ok(());
        }
        self.stream.sync()?;
        let oldest = self
            .graphs
            .draft_slots
            .iter()
            .filter_map(|(k, s)| match s {
                Slot::Ready(_, used) => Some((*used, *k)),
                _ => None,
            })
            .min_by_key(|x| x.0);
        if let Some((_, k)) = oldest {
            self.graphs.draft_slots.insert(k, Slot::Seen);
            self.graphs.stats.evictions += 1;
        }
        Ok(())
    }

    /// Keep the number of instantiated graphs under the limit: drop the least recently used (after the stream has drained: a graph that is
    /// still executing must not be destroyed).
    fn make_room(&mut self) -> Result<(), ModelError> {
        let ready = self.graphs.slots.values().filter(|s| matches!(s, Slot::Ready(..))).count();
        if ready < GRAPH_LIMIT {
            return Ok(());
        }
        self.stream.sync()?;
        let oldest = self
            .graphs
            .slots
            .iter()
            .filter_map(|(k, s)| match s {
                Slot::Ready(_, used) => Some((*used, *k)),
                _ => None,
            })
            .min_by_key(|x| x.0);
        if let Some((_, k)) = oldest {
            self.graphs.slots.insert(k, Slot::Seen);
            self.graphs.stats.evictions += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_windows_key_is_every_host_value_the_launches_bake_in() {
        let k = key_of(3, 1, 2300, Head::AllRows);
        assert_eq!(k, GraphKey { rows: 3, n_seq: 1, sparse: true, all_rows: true });
        // rows, sequences, regime and head each make another graph
        assert_ne!(k, key_of(4, 1, 2300, Head::AllRows));
        assert_ne!(k, key_of(3, 2, 2300, Head::AllRows));
        assert_ne!(k, key_of(3, 1, 200, Head::AllRows));
        assert_ne!(k, key_of(3, 1, 2300, Head::None));
        // the regime flips between positions 2,047 (2,048 keys: dense) and 2,048
        assert!(!key_of(1, 1, 2047, Head::AllRows).sparse && key_of(1, 1, 2048, Head::AllRows).sparse);
        // the context beyond the regime is not in the key: a graph serves every depth of its regime
        assert_eq!(key_of(1, 1, 2048, Head::AllRows), key_of(1, 1, 200_000, Head::AllRows));
        assert_eq!(key_of(1, 1, 0, Head::AllRows), key_of(1, 1, 2047, Head::AllRows));
    }

    #[test]
    fn the_dense_graph_never_launches_the_sparse_attention_and_the_sparse_one_always_does() {
        // `attend_auto` launches the sparse kernel whenever the context it is given exceeds 2,048; in the dense regime the selection buffer is stale
        // (nothing wrote it), so the dense graph must be built with a context of at most 2,048 whatever the model's context is
        for max_context in [256usize, 2048, 8192, 262_144] {
            let dense = RunShape::for_window(false, 1 << 16, max_context);
            assert!(dense.mla_ctx <= gb_cuda::mla::DENSE_MAX_KEYS, "dense graph at context {max_context}: {}", dense.mla_ctx);
            assert!(!dense.any_sparse);
        }
        for max_context in [8192usize, 262_144] {
            let sparse = RunShape::for_window(true, 1 << 16, max_context);
            assert!(sparse.mla_ctx > gb_cuda::mla::DENSE_MAX_KEYS, "sparse graph at context {max_context}");
            assert!(sparse.any_sparse);
            // the scoring grid covers the scratch's whole capacity (a graph cannot know the context), never less
            assert_eq!(sparse.n_pools_max, 1 << 16);
        }
    }

    #[test]
    fn every_graph_mutant_has_its_test_name() {
        for (n, m) in [
            ("stale_rows", GraphMutant::StaleRows),
            ("stale_table", GraphMutant::StaleTable),
            ("wrong_shape", GraphMutant::WrongShape),
            ("sync_in_capture", GraphMutant::SyncInCapture),
            ("slow_side", GraphMutant::SlowSide),
            ("join_late", GraphMutant::JoinLate),
            ("skip_latent", GraphMutant::SkipWait(Wait::Latent)),
            ("skip_pool", GraphMutant::SkipWait(Wait::Pool)),
            ("skip_q", GraphMutant::SkipWait(Wait::Q)),
            ("skip_select", GraphMutant::SkipWait(Wait::Select)),
            ("draft_stale_rows", GraphMutant::DraftStaleRows),
        ] {
            assert_eq!(GraphMutant::from_name(n), Some(m), "{n}");
        }
        assert_eq!(GraphMutant::from_name("none"), None);
    }

    #[test]
    fn a_capture_refusal_is_not_fatal_but_a_device_fault_inside_the_capture_is() {
        use gb_cuda::CudaError;
        for code in [900, 901, 904, 906, 908, 910] {
            assert!(!closure_is_fatal(&ModelError::Cuda(CudaError(code))), "{code}");
        }
        for code in [700, 710, 719] {
            assert!(closure_is_fatal(&ModelError::Cuda(CudaError(code))), "{code}");
        }
        assert!(closure_is_fatal(&ModelError::StateLost("x".into())));
        assert!(!closure_is_fatal(&ModelError::Request("x".into())));
    }
}
