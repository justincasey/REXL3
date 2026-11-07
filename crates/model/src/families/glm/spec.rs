//! Speculation in the GLM backend: the drafters behind `Backend::propose_batch` (the pack's NextN layer, the copy drafter, nothing), the arbiter
//! that picks among them per round, the per-slot record of what each round offered, and the telemetry (per source and per draft depth) the
//! measurements read.
//!
//! The contract is the engine's: a drafter only PROPOSES. The verify window decides every committed token (`accept` = the keyed sample equals
//! the draft), so whatever this file proposes, drafted == serial, byte for byte. What it can change is speed, and the one thing it must not do
//! is disturb the trunk's state: a draft call touches only the drafter's own cache and scratch (`GlmModel::mtp_draft`).
//!
//! A round, as the backend sees it: `propose` (called by the engine just before the window, for the streams that may draft) returns each stream's
//! drafts and remembers what it offered for the slot; `on_step` learns how many of them the engine put in the window; `on_commit` learns how many
//! were accepted, updates the telemetry and, in `auto`, the arbiter's estimates. A slot's record is consumed by its commit; a sequence that ends
//! never commits and its slot is cleared by `end`.
//!
//! **Who decides.** `--spec mtp` drafts at a fixed depth and `--spec copy` offers what the copy drafter finds; both leave the engine's own row
//! budget in charge (`engine_over`: the measured step table trims an offer whose estimated acceptance does not pay for its rows). `--spec auto`
//! gives the decision to [`gb_spec::arbiter::Arbiter`]: per stream, none, copy or the head at a depth, by committed tokens per model-millisecond
//! with this model's measured costs ([`GlmCost`]), and the engine's budget is made flat so that it does not second-guess it.

use super::alloc::{MTP_MAX_DEPTH, MTP_TAP_CAP};
use super::backend::window_is_text;
use super::model::{DflashAsk, DraftAsk};
use super::tp_mtp::DraftModel;
use crate::error::ModelError;
use gb_engine::ProposeRequest;
use gb_sampling::Sampling;
use gb_sched::CostModel;
use gb_spec::arbiter::{Arbiter, ArbiterConfig, ArbiterCost, Ask, Choice};
use gb_spec::{CopyDrafter, DraftPick, SpecMode};
use std::time::Instant;

/// The depth `--spec mtp` drafts at (the measured best single-stream depth of the oracle's MTP: K = 2 as fast as K = 3 on the median, with fewer
/// expert reads per round); `REXL3_MTP_DEPTH` overrides it.
pub const MTP_DEFAULT_DEPTH: usize = 2;
/// Most drafts the copy drafter offers in a round (TensorFold's windows: enter on 8 matching tokens, offer up to 7).
pub const COPY_MAX: usize = 7;
const COPY_MIN_MATCH: usize = 8;
/// Depths the per-depth telemetry keeps.
const DEPTHS: usize = 8;
/// The most rows one stream's window can hold (the base row and the engine's seven drafts): the store of pending rows must have room for this
/// many more after every round, or the stream must absorb first.
const WINDOW_MAX: usize = 8;

/// Where a round's drafts came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// No drafts: one serial row.
    None,
    Copy,
    Mtp,
}

impl Source {
    fn idx(self) -> usize {
        match self {
            Source::None => 0,
            Source::Copy => 1,
            Source::Mtp => 2,
        }
    }
    fn of(c: Choice) -> Self {
        match c {
            Choice::None => Source::None,
            Choice::Copy(_) => Source::Copy,
            Choice::Mtp(_) => Source::Mtp,
        }
    }
}

/// What a backend speculates with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpecConfig {
    pub mode: SpecMode,
    /// Drafts per round of `--spec mtp`, 1..=`alloc::MTP_MAX_DEPTH`.
    pub mtp_depth: usize,
    /// The drafter's head runs over a restricted vocabulary (`GlmModel::mtp_restrict_vocab`): the arbiter prices a draft call with `MTP_COST_RESTRICTED`.
    pub restricted_head: bool,
    /// How the head proposes for a SAMPLED request: its argmax, or the target's own keyed draw over its candidates at each draft's position (keyed-noise
    /// drafting, `docs/GLM53.md`). Greedy requests are the argmax either way. `REXL3_GLM_DRAFT_NOISE=0|1`.
    pub noise: DraftPick,
    /// Rank 0 of a two-rank tensor-parallel group (the sc_4.10bpw pack on two GB10s): the arbiter prices rounds with the tables measured
    /// there ([`VERIFY_MS_TP`], [`MTP_COST_TP`] / [`MTP_COST_TP_RESTRICTED`]) instead of the single-node 2.25bpw ones. Set by the TP boot.
    pub two_ranks: bool,
}

/// Keyed-noise drafting unless `REXL3_GLM_DRAFT_NOISE=0` (the result of the live A/B is in `docs/GLM53.md`, "Keyed-noise drafting").
pub const DRAFT_NOISE_DEFAULT: DraftPick = DraftPick::Keyed;

impl SpecConfig {
    pub fn new(mode: SpecMode) -> Self {
        Self { mode, mtp_depth: MTP_DEFAULT_DEPTH, restricted_head: false, noise: DRAFT_NOISE_DEFAULT, two_ranks: false }
    }

    /// `new`, with the depth from `REXL3_MTP_DEPTH` when it is set (a malformed value is an error: a measurement must not silently run at the
    /// default).
    pub fn from_env(mode: SpecMode) -> Result<Self, ModelError> {
        let mut c = Self::new(mode);
        if let Ok(v) = std::env::var("REXL3_MTP_DEPTH") {
            c.mtp_depth = parse_depth(&v).map_err(ModelError::Config)?;
        }
        // the engine builds the restricted head from the same setting (`engine::build_engine`)
        c.restricted_head = super::engine::mtp_vocab_setting(std::env::var("REXL3_MTP_VOCAB").ok().as_deref())
            .map_err(ModelError::Config)?
            .is_some();
        c.noise = parse_noise(std::env::var("REXL3_GLM_DRAFT_NOISE").ok().as_deref()).map_err(ModelError::Config)?;
        eprintln!(
            "rexl3 event=draft_noise pick={} (REXL3_GLM_DRAFT_NOISE=0|1)",
            if c.noise.is_keyed() { "keyed" } else { "argmax" }
        );
        Ok(c)
    }
}

/// `REXL3_GLM_DRAFT_NOISE`: `0` (the head's argmax) or `1` (the target's keyed draw over the head's candidates); unset is [`DRAFT_NOISE_DEFAULT`]; anything
/// else is an error (a measurement must not silently run on the default).
fn parse_noise(v: Option<&str>) -> Result<DraftPick, String> {
    DraftPick::from_setting(v, "REXL3_GLM_DRAFT_NOISE", DRAFT_NOISE_DEFAULT)
}

/// The sampler each ask of a draft call carries: the request's own when keyed drafting is on, none when it is off (the head's argmax). The model leaves a
/// greedy request, and a `top_k` it cannot fetch, unkeyed by itself (`mtp::keyed_record`).
fn keyed_samplers(noise: DraftPick, samplings: &[&Sampling]) -> Vec<Option<Sampling>> {
    samplings.iter().map(|s| noise.is_keyed().then_some(**s)).collect()
}

/// `REXL3_MTP_DEPTH`: a number in `1..=alloc::MTP_MAX_DEPTH`.
fn parse_depth(v: &str) -> Result<usize, String> {
    let d: usize = v.trim().parse().map_err(|_| format!("REXL3_MTP_DEPTH={v} is not a number"))?;
    if d == 0 || d > MTP_MAX_DEPTH {
        return Err(format!("REXL3_MTP_DEPTH must be 1..={MTP_MAX_DEPTH}, got {d}"));
    }
    Ok(d)
}

// ------------------------------------------------------------------------------------------------------------------------ costs

/// The verify window of this model by total rows, in milliseconds: the step table with every decode-step lever on (CUDA graphs, the router's loads in
/// flight, the shared expert and the MLA chains on side branches; `glm_perf.rs`, `perf_verify_table`: one sequence at context ~2,300, `R` rows of real
/// text a window, each window committed in full so that the commit applied inside the next forward is in the time, the median of 11). It is concave:
/// every routed expert a row picks is read once more only when no row of the window picked it already, so the rows after the first are cheaper than the
/// first (9.2 ms for the second, 3-6 ms for the tenth), and the time hardly depends on how the rows are spread over sequences. (Before the graphs the
/// table was 44.9, 54.4, 65.1, 72.4, 81.0, 91.1, 96.9, 103.8, 110.4 ms for 1..9 rows and 320.8 at 64.)
pub const VERIFY_MS: &[(usize, f64)] = &[
    (1, 38.7),
    (2, 47.9),
    (3, 56.2),
    (4, 64.0),
    (5, 71.6),
    (6, 78.4),
    (7, 85.1),
    (8, 94.6),
    (9, 100.3),
    (12, 116.3),
    (16, 137.4),
    (24, 177.8),
    (32, 215.1),
    (48, 274.9),
    (64, 316.7),
];

/// The draft call's time, measured on the GB10 (`glm_spec.rs`, `mtp_draft_costs`: the wall time of `mtp_draft`, absorb + chain + one sync, median of
/// seven, 1 to 8 streams of about 300 tokens, 1 to 8 pending rows each). The shape the numbers have, and the model of it:
/// * **absorb**: a launch-and-sync base (1.3 ms) and 0.07 ms a stream, plus the layer over all the pending rows of all the streams, which is concave in
///   the rows like the trunk's window (the 8-expert reads of 4-bit experts overlap): 0.4 ms for 1 row, 3.2 for 8, 5.6 for 16, 9.0 for 32, 13.1 for 64;
/// * **head**: the `lm_head` over one row a stream and the top-1, 2.1 ms whatever the streams (it reads the 0.48 GB head once);
/// * **chain**: each further draft is one MTP row a stream and a head: 3.0 ms + 0.4 ms a stream (3.4 for one stream, 6.2 for eight).
///
/// The fit is within 2% of the measurements at the corners (1 stream 1 row 4 deep: 14.07 against 14.30 ms; 8 streams 8 rows each 4 deep: 35.7 against 36.3).
#[derive(Clone, Copy, Debug)]
pub struct MtpCost {
    pub base: f64,
    pub per_stream: f64,
    /// (pending rows in all, milliseconds of the layer over them), concave; zero rows cost nothing.
    pub rows: &'static [(usize, f64)],
    pub head: f64,
    pub chain: f64,
    pub chain_per_stream: f64,
}

impl MtpCost {
    fn rows_ms(&self, rows: usize) -> f64 {
        let t = self.rows;
        let (mut r0, mut t0) = (0usize, 0.0f64);
        for &(r1, t1) in t {
            if rows <= r1 {
                return t0 + (t1 - t0) * (rows - r0) as f64 / (r1 - r0) as f64;
            }
            (r0, t0) = (r1, t1);
        }
        // past the table: the last segment's slope
        let n = t.len();
        let ((ra, ta), (rb, tb)) = if n >= 2 { (t[n - 2], t[n - 1]) } else { ((0, 0.0), t[n - 1]) };
        tb + (tb - ta) / (rb - ra) as f64 * (rows - rb) as f64
    }

    /// Milliseconds of a call for `streams` streams absorbing `rows` pending rows in all and drafting `depth` deep (`depth == 0`: only absorbing).
    pub fn ms(&self, streams: usize, rows: usize, depth: usize) -> f64 {
        let s = streams as f64;
        let mut t = self.base + self.per_stream * s + self.rows_ms(rows);
        if depth >= 1 {
            t += self.head;
        }
        if depth >= 2 {
            t += (depth - 1) as f64 * (self.chain + self.chain_per_stream * s);
        }
        t
    }
}

/// What the arbiter prices a round with: the step table and the draft model, each scaled by how the last rounds really ran against them (clamped:
/// a single odd round must not move a decision far).
pub struct GlmCost {
    verify: CostModel,
    /// What a window costs a sequence more than the table's rows say (lane T, MEASURED on a two-rank group: attention and state per
    /// sequence; 0 on one node, whose table was measured without it). [`ArbiterCost::verify_ms_of`] adds it per extra sequence.
    pub per_stream_ms: f64,
    pub mtp: MtpCost,
    /// Measured over predicted, smoothed.
    pub verify_scale: f64,
    pub mtp_scale: f64,
}

const SCALE_DECAY: f64 = 0.9;
const SCALE_RANGE: (f64, f64) = (0.7, 1.6);

impl GlmCost {
    pub fn new(mtp: MtpCost) -> Self {
        Self::with_tables(VERIFY_MS, mtp)
    }
    /// A cost model over another step table (`VERIFY_MS_TP` for a two-rank group).
    pub fn with_tables(verify: &[(usize, f64)], mtp: MtpCost) -> Self {
        Self { verify: CostModel::from_table(verify.to_vec()), per_stream_ms: 0.0, mtp, verify_scale: 1.0, mtp_scale: 1.0 }
    }
    /// This model with a per-sequence window cost (see [`GlmCost::per_stream_ms`]).
    pub fn with_per_stream(mut self, ms: f64) -> Self {
        self.per_stream_ms = ms;
        self
    }
    /// The tables a backend prices with: the two-rank group's when `cfg.two_ranks`, one node's otherwise; the restricted head's draft model
    /// when `cfg.restricted_head`.
    pub fn for_config(cfg: &SpecConfig) -> Self {
        match (cfg.two_ranks, cfg.restricted_head) {
            (true, true) => Self::with_tables(VERIFY_MS_TP, MTP_COST_TP_RESTRICTED).with_per_stream(VERIFY_PER_STREAM_MS_TP),
            (true, false) => Self::with_tables(VERIFY_MS_TP, MTP_COST_TP).with_per_stream(VERIFY_PER_STREAM_MS_TP),
            (false, true) => Self::new(MTP_COST_RESTRICTED),
            (false, false) => Self::new(MTP_COST),
        }
    }
    /// A window of `rows` rows took `ms`.
    pub fn observe_window(&mut self, rows: usize, ms: f64) {
        self.observe_window_of(1, rows, ms);
    }
    /// A window of `streams` sequences over `rows` rows took `ms`.
    pub fn observe_window_of(&mut self, streams: usize, rows: usize, ms: f64) {
        let predicted = self.unscaled_window(streams, rows);
        self.verify_scale = Self::blend(self.verify_scale, ms / predicted);
    }
    fn unscaled_window(&self, streams: usize, rows: usize) -> f64 {
        self.verify.ms(rows) + self.per_stream_ms * streams.saturating_sub(1) as f64
    }
    /// A draft call took `ms`.
    pub fn observe_draft(&mut self, streams: usize, rows: usize, depth: usize, ms: f64) {
        let predicted = self.mtp.ms(streams, rows, depth);
        self.mtp_scale = Self::blend(self.mtp_scale, ms / predicted);
    }
    fn blend(scale: f64, observed_ratio: f64) -> f64 {
        if !observed_ratio.is_finite() || observed_ratio <= 0.0 {
            return scale;
        }
        (scale * SCALE_DECAY + observed_ratio * (1.0 - SCALE_DECAY)).clamp(SCALE_RANGE.0, SCALE_RANGE.1)
    }
}

impl ArbiterCost for GlmCost {
    fn verify_ms(&self, rows: usize) -> f64 {
        self.verify.ms(rows) * self.verify_scale
    }
    /// The table's rows plus [`GlmCost::per_stream_ms`] a sequence past the first (the default when it is 0: one node).
    fn verify_ms_of(&self, streams: usize, rows: usize) -> f64 {
        self.unscaled_window(streams, rows) * self.verify_scale
    }
    fn mtp_ms(&self, streams: usize, absorb_rows: usize, depth: usize) -> f64 {
        self.mtp.ms(streams, absorb_rows, depth) * self.mtp_scale
    }
}

/// The measured draft-call model with the trunk's full `lm_head` (see [`MtpCost`]); re-measured with `mtp_draft_costs` after the decode-step levers (the
/// layer's MoE and attention are the same kernels as before; the call is not graphed): within 7% of the 100 measurements, 1-3% at the corners.
pub const MTP_COST: MtpCost = MtpCost {
    base: 1.0,
    per_stream: 0.09,
    rows: &[(1, 0.50), (2, 0.95), (3, 1.45), (5, 2.10), (8, 3.31), (16, 5.86), (32, 9.67), (64, 13.58)],
    head: 2.06,
    chain: 3.16,
    chain_per_stream: 0.42,
};

/// The same with the drafter's head over the restricted vocabulary (`REXL3_MTP_VOCAB`, 555 strips of 1,210): the head 0.92 ms instead of 2.06, and each
/// chained draft 1.93 + 0.42 per stream instead of 3.16 (`REXL3_COST_VOCAB=512 mtp_draft_costs`; within 5% of the 100 measurements).
pub const MTP_COST_RESTRICTED: MtpCost = MtpCost {
    base: 1.0,
    per_stream: 0.07,
    rows: &[(1, 0.56), (2, 1.00), (3, 1.36), (5, 2.14), (8, 3.34), (16, 5.85), (32, 9.50), (64, 13.61)],
    head: 0.92,
    chain: 1.93,
    chain_per_stream: 0.42,
};

/// The step table of a two-rank group, rank 0's verify window by total rows, in milliseconds. MEASURED 2026-10-19 on N3 + N2:
/// - setup: sc_4.10bpw, lane E's decode split `REXL3_GLM_TP_SPLIT=kda,lm,mla,sh` (bin e-abe5d512), the Phase 1 levers,
///   `REXL3_TP_TRANSPORT=roce`, the draft calls graphed, the 8-slot kit;
/// - method: `tools/glm_tp_cost_tables.sh` at depths 0..4 and auto, 1 / 2 / 4 / 8 streams, then `tools/glm_tp_cost_fit.py`.
///
/// What the numbers say:
/// - The 1-row step is 39.7 ms (it was 55.4 before E2.5 / E2.6), then ~7-8 ms a row to 6 rows, falling to ~3 ms a row at 32-40 rows.
/// - A window of S sequences costs [`VERIFY_PER_STREAM_MS_TP`] a sequence more than one of the same rows on one sequence. The table
///   holds the points less that (S - 1) times, and [`GlmCost::verify_ms_of`] adds it back.
/// - Measured to 40 rows; 48 and 64 rows are the last measured slope (ESTIMATED: an 8-slot kit at depth 4 reaches 40).
/// - Windows above 1 MiB of exchange (6 rows and up under the `sh` split) ran on NCCL here (the transport's sum rule): re-measure
///   after the sum sites grow.
pub const VERIFY_MS_TP: &[(usize, f64)] = &[
    (1, 39.7),
    (2, 49.3),
    (3, 56.7),
    (4, 64.8),
    (5, 71.6),
    (6, 79.3),
    (8, 92.4),
    (10, 105.0),
    (12, 114.9),
    (16, 133.3),
    (20, 155.9),
    (24, 169.8),
    (32, 196.2),
    (40, 222.6),
    (48, 237.6),
    (64, 267.6),
];

/// What a window of a two-rank group costs per sequence past the first, beyond its rows (attention and recurrent state per sequence).
/// MEASURED as above: the median over 10 row counts measured at two stream counts.
pub const VERIFY_PER_STREAM_MS_TP: f64 = 4.38;

/// The draft-call model of a two-rank group with the trunk's full `lm_head` (`REXL3_MTP_VOCAB=0`), the call graphed. MEASURED as
/// [`VERIFY_MS_TP`]:
/// - data: a least-squares fit of 4,892 draft calls (1..8 streams, depths 1..4, 1..44 absorbed rows; 5 more than 25 % off a first fit
///   left out); median |error| 3.4 %, p90 9.5 %;
/// - base + head = 3.48 ms; 0.52 ms a stream; each chained draft 3.16 + 0.44 a stream;
/// - the absorbed rows cost nothing measurable up to 44 rows.
///
/// ESTIMATED parts:
/// - the base / head split (one node's proportion). Serving makes no absorb-only call to measure it apart, and the arbiter prices one
///   only for a sequence whose tap store is nearly full;
/// - the rows past 44 (the single-node increments x 0.5).
pub const MTP_COST_TP: MtpCost = MtpCost {
    base: 1.42,
    per_stream: 0.52,
    rows: &[(1, 0.00), (2, 0.00), (3, 0.00), (5, 0.00), (8, 0.00), (16, 0.00), (32, 0.00), (44, 0.00), (64, 1.22)],
    head: 2.06,
    chain: 3.16,
    chain_per_stream: 0.44,
};

/// The same with the restricted head (the serving default). MEASURED as [`VERIFY_MS_TP`]:
/// - data: 4,906 draft calls (59 more than 25 % off a first fit left out); median |error| 4.6 %, p90 12.7 %;
/// - base + head = 2.40 ms; 0.54 ms a stream; each chained draft 1.97 + 0.43 a stream;
/// - the absorbed rows cost nothing measurable up to 40 rows.
///
/// ESTIMATED parts: the base / head split, and the rows past 40 (as for the full head).
pub const MTP_COST_TP_RESTRICTED: MtpCost = MtpCost {
    base: 1.47,
    per_stream: 0.54,
    rows: &[(1, 0.00), (2, 0.00), (3, 0.00), (5, 0.00), (8, 0.00), (16, 0.00), (32, 0.00), (40, 0.00), (64, 1.54)],
    head: 0.94,
    chain: 1.97,
    chain_per_stream: 0.43,
};

/// `REXL3_GLM_COST_LOG=<file>`: every verify window (`W rows ms streams predicted_ms`), every draft call (`D streams rows depth ms`; a
/// DFlash2 call adds each stream's `slot:chain/kept/reach` with the chances the cut priced each depth with, `;`-separated), every
/// round's commit (`C slot verified accepted`) and every window graph captured (`G rows sequences ms`) is appended to the file, and on a
/// two-rank group every prompt forward or part of one rank 0 runs (`P rows prompts layers ms`, `tp_backend`). Each line ends with the
/// milliseconds since the first line (when it was written: the gaps between rounds). Unset: nothing is written.
pub const COST_LOG_ENV: &str = "REXL3_GLM_COST_LOG";

pub(super) fn cost_log(line: std::fmt::Arguments<'_>) {
    use std::io::Write;
    static LOG: std::sync::OnceLock<Option<(std::sync::Mutex<std::fs::File>, Instant)>> = std::sync::OnceLock::new();
    let f = LOG.get_or_init(|| {
        let p = std::env::var(COST_LOG_ENV).ok().filter(|p| !p.trim().is_empty())?;
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p.trim())
            .ok()
            .map(|f| (std::sync::Mutex::new(f), Instant::now()))
    });
    if let Some((m, t0)) = f {
        if let Ok(mut f) = m.lock() {
            let _ = writeln!(f, "{line} {:.3}", t0.elapsed().as_secs_f64() * 1e3);
        }
    }
}

/// Whether [`cost_log`] writes (so a caller formats nothing it would drop).
pub(super) fn cost_log_on() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var(COST_LOG_ENV).is_ok_and(|p| !p.trim().is_empty()));
    *ON
}

// ---------------------------------------------------------------------------------------------------------------------- telemetry

/// Counters per source (`Source::idx`) and per draft depth; the measurements read them.
#[derive(Clone, Debug, Default)]
pub struct SpecStats {
    /// Rounds a stream's window was fed with drafts from the source (`Source::None`: rounds with no drafts at all).
    pub rounds: [u64; 3],
    /// Drafts the engine verified, drafts accepted, tokens committed (accepted + 1 per round), by source.
    pub verified: [u64; 3],
    pub accepted: [u64; 3],
    pub committed: [u64; 3],
    /// Per source and depth `j` (index `j - 1`): rounds that verified a depth-`j` draft, and of those the rounds where it was accepted.
    pub reached: [[u64; DEPTHS]; 3],
    pub hits: [[u64; DEPTHS]; 3],
    /// Rounds by how many drafts rode in the window, per source (index = drafts, 0..=DEPTHS): what the arbiter or the fixed depth really asked for.
    pub by_depth: [[u64; DEPTHS + 1]; 3],
    /// Wall milliseconds and calls of the MTP draft call (absorb + chain), and the rows its absorbs held.
    pub mtp_ms: f64,
    pub mtp_calls: u64,
    pub mtp_rows: u64,
    /// Wall milliseconds, calls and rows of the verify windows.
    pub window_ms: f64,
    pub windows: u64,
    pub window_rows: u64,
}

impl SpecStats {
    /// Mean tokens committed per round of `s`.
    pub fn tokens_per_round(&self, s: Source) -> f64 {
        let i = s.idx();
        if self.rounds[i] == 0 {
            0.0
        } else {
            self.committed[i] as f64 / self.rounds[i] as f64
        }
    }
    /// Conditional acceptance of a depth-`j` draft (1-based) of `s`: accepted over reached.
    pub fn depth_acceptance(&self, s: Source, j: usize) -> f64 {
        let (r, h) = (self.reached[s.idx()][j - 1], self.hits[s.idx()][j - 1]);
        if r == 0 {
            0.0
        } else {
            h as f64 / r as f64
        }
    }

    fn record(&mut self, source: Source, verified: usize, accepted: usize) {
        let i = source.idx();
        self.rounds[i] += 1;
        self.verified[i] += verified as u64;
        self.accepted[i] += accepted as u64;
        self.committed[i] += accepted as u64 + 1;
        self.by_depth[i][verified.min(DEPTHS)] += 1;
        for j in 1..=verified.min(DEPTHS) {
            // depth j is reached when the j - 1 drafts before it were all accepted
            if accepted + 1 >= j {
                self.reached[i][j - 1] += 1;
                if accepted >= j {
                    self.hits[i][j - 1] += 1;
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------- DFlash2's cut

/// Upper edges of the margin bins of [`DflashCut`] (the selector's lead of a pick over the runner-up: a raw score gap for a greedy
/// request, the noised gap in temperature units for a sampled one; at most 30). The last bin is open.
const CUT_EDGES: [f32; 11] = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0];
const CUT_BINS: usize = CUT_EDGES.len() + 1;
/// The prior's weight in rounds, and the decay of a bin's counts each time it is observed (it follows the text, slowly).
const CUT_PRIOR_WEIGHT: f64 = 6.0;
const CUT_DECAY: f64 = 0.998;
/// The draft call's time before any was measured (ms), and the smoothing of the measurements.
const DFLASH_MS_PRIOR: f64 = 6.0;
const DFLASH_MS_DECAY: f64 = 0.9;

/// DFlash2's chain cut: the chance that a draft is accepted given the drafts before it were, by the selector's margin of the pick,
/// learned online from what the verify windows accepted (per margin bin: hits over trials, decayed, with a prior that rises with the
/// margin). A round's drafts are then cut where one more row no longer pays for itself ([`allocate_chains`]).
#[derive(Clone, Debug)]
pub struct DflashCut {
    hits: [f64; CUT_BINS],
    trials: [f64; CUT_BINS],
}

impl Default for DflashCut {
    fn default() -> Self {
        Self { hits: [0.0; CUT_BINS], trials: [0.0; CUT_BINS] }
    }
}

impl DflashCut {
    fn bin(margin: f32) -> usize {
        CUT_EDGES.iter().position(|&e| margin < e).unwrap_or(CUT_EDGES.len())
    }
    /// The prior chance of a bin: 0.3 at no margin, rising to 0.95 (the middle of the bin).
    fn prior(bin: usize) -> f64 {
        let lo = if bin == 0 { 0.0 } else { CUT_EDGES[bin - 1] as f64 };
        let hi = CUT_EDGES.get(bin).map_or(lo * 1.5, |&e| e as f64);
        0.3 + 0.65 * (1.0 - (-(lo + hi) / 2.0 / 2.5).exp())
    }
    /// The chance a pick of this margin is accepted, given every pick before it was.
    pub fn chance(&self, margin: f32) -> f64 {
        let b = Self::bin(margin);
        (self.hits[b] + Self::prior(b) * CUT_PRIOR_WEIGHT) / (self.trials[b] + CUT_PRIOR_WEIGHT)
    }
    /// A round verified the first `verified` picks (their margins) and accepted the first `accepted`: every pick up to the first rejected
    /// one was reached.
    pub fn observe(&mut self, margins: &[f32], verified: usize, accepted: usize) {
        for (j, &m) in margins.iter().take(verified).enumerate() {
            if j > accepted {
                break;
            }
            let b = Self::bin(m);
            self.hits[b] = self.hits[b] * CUT_DECAY + (j < accepted) as u8 as f64;
            self.trials[b] = self.trials[b] * CUT_DECAY + 1.0;
        }
    }
}

/// How many drafts of each stream's chain to verify: `reach[s][j]` is the chance draft `j` of stream `s` and every one before it are
/// accepted. Rows are added one at a time, the likeliest next draft of any stream first (ties to the lower stream), and the counts with the
/// most expected tokens per millisecond of `round_ms(streams, rows) + overhead_ms` are kept (TensorFold's allocation over node chances).
/// Every stream holds its base row; the window holds at most `max_rows`.
pub fn allocate_chains(reach: &[Vec<f64>], max_rows: usize, overhead_ms: f64, round_ms: &dyn Fn(usize, usize) -> f64) -> Vec<usize> {
    let s = reach.len();
    let mut counts = vec![0usize; s];
    let mut best_counts = counts.clone();
    let mut rows = s;
    let mut expected = s as f64;
    let rate = |rows: usize, expected: f64| expected / (round_ms(s, rows) + overhead_ms);
    let mut best = rate(rows, expected);
    while rows < max_rows {
        let pick = (0..s).filter(|&i| counts[i] < reach[i].len()).fold(None, |b: Option<usize>, i| match b {
            Some(j) if reach[j][counts[j]] >= reach[i][counts[i]] => Some(j),
            _ => Some(i),
        });
        let Some(i) = pick else { break };
        expected += reach[i][counts[i]];
        counts[i] += 1;
        rows += 1;
        let now = rate(rows, expected);
        if now > best {
            best = now;
            best_counts.clone_from(&counts);
        }
    }
    best_counts
}

// ----------------------------------------------------------------------------------------------------------------------- runtime

/// What a slot's last `propose` offered, until its commit.
struct Round {
    choice: Choice,
    drafts: Vec<u32>,
    /// How many of `drafts` were in the window (the engine may trim the offer); 0 until `on_step`.
    verified: usize,
    /// A DFlash2 round's margins of its drafts (what [`DflashCut`] learns from); empty for every other round.
    margins: Vec<f32>,
}

struct SlotSpec {
    copy: CopyDrafter,
    round: Option<Round>,
    /// Totals of this sequence, for the line `end` logs.
    rounds: u64,
    committed: u64,
    by_source: [u64; 3],
    /// The runtime's draft and window counters when the sequence began (`(draft ms, draft calls, window ms, windows, window rows)`).
    since: (f64, u64, f64, u64, u64),
}

impl SlotSpec {
    fn new() -> Self {
        Self {
            copy: CopyDrafter::new(COPY_MIN_MATCH, COPY_MAX),
            round: None,
            rounds: 0,
            committed: 0,
            by_source: [0; 3],
            since: (0.0, 0, 0.0, 0, 0),
        }
    }
}

/// The speculation state of one backend.
pub struct SpecRuntime {
    pub cfg: SpecConfig,
    slots: Vec<SlotSpec>,
    pub stats: SpecStats,
    pub arbiter: Arbiter,
    pub cost: GlmCost,
    /// DFlash2's chain cut and the draft call's smoothed time (ms): used when the model drafts with DFlash2.
    pub cut: DflashCut,
    pub dflash_ms: f64,
}

impl SpecRuntime {
    pub fn new(cfg: SpecConfig, slots: usize) -> Self {
        if cfg.mode.uses_mtp() {
            eprintln!(
                "rexl3 event=spec_cost_tables basis={} head={}",
                if cfg.two_ranks { "two_ranks (VERIFY_MS_TP, MTP_COST_TP*)" } else { "one_node (VERIFY_MS, MTP_COST*)" },
                if cfg.restricted_head { "restricted" } else { "full" }
            );
        }
        let arb = ArbiterConfig {
            allow_mtp: cfg.mode.uses_mtp(),
            allow_copy: cfg.mode.uses_copy(),
            max_mtp_depth: MTP_MAX_DEPTH,
            max_rows: super::alloc::WINDOW_ROWS,
            ..ArbiterConfig::default()
        };
        Self {
            cfg,
            slots: (0..slots).map(|_| SlotSpec::new()).collect(),
            stats: SpecStats::default(),
            arbiter: Arbiter::new(arb),
            cost: GlmCost::for_config(&cfg),
            cut: DflashCut::default(),
            dflash_ms: DFLASH_MS_PRIOR,
        }
    }

    /// A new sequence starts in `slot`.
    pub fn begin(&mut self, slot: usize) {
        let st = &self.stats;
        self.slots[slot] = SlotSpec { since: (st.mtp_ms, st.mtp_calls, st.window_ms, st.windows, st.window_rows), ..SlotSpec::new() };
        self.arbiter.begin(slot);
    }

    /// The sequence in `slot` ended (or failed): log its totals and forget it. The draft and window times are the runtime's over the
    /// sequence's life (every stream's: exact for one stream at a time).
    pub fn end(&mut self, slot: usize) {
        let s = std::mem::replace(&mut self.slots[slot], SlotSpec::new());
        if s.rounds > 0 {
            let st = &self.stats;
            let (dm, dc, wm, wc, wr) = s.since;
            let per = |ms: f64, n: u64| if n == 0 { 0.0 } else { ms / n as f64 };
            eprintln!(
                "rexl3 event=spec_seq slot={slot} rounds={} tokens={} tokens_per_round={:.3} rounds_none={} rounds_copy={} rounds_mtp={} draft_ms={:.2} draft_calls={} window_ms={:.2} window_rows={:.2} windows={}",
                s.rounds,
                s.committed,
                s.committed as f64 / s.rounds as f64,
                s.by_source[0],
                s.by_source[1],
                s.by_source[2],
                per(st.mtp_ms - dm, st.mtp_calls - dc),
                st.mtp_calls - dc,
                per(st.window_ms - wm, st.windows - wc),
                per((st.window_rows - wr) as f64, st.windows - wc),
                st.windows - wc
            );
        }
    }

    /// The drafts for the streams that asked this round (one entry per request: `Some(drafts)`, empty for none). The mode decides the source.
    pub fn propose<M: DraftModel + ?Sized>(
        &mut self,
        m: &mut M,
        reqs: &[ProposeRequest],
        images: &[Vec<std::ops::Range<usize>>],
    ) -> Result<Vec<Option<Vec<u32>>>, ModelError> {
        // a DFlash2 drafter takes the head's place (`--draft DIR`): one block pass drafts every asking stream, the cut decides the depths
        if m.has_dflash() && self.cfg.mode.uses_mtp() {
            return self.propose_dflash(m, reqs);
        }
        let mode = self.cfg.mode;
        let head = m.has_mtp() && mode.uses_mtp();
        // the previous round's commits (the KDA flips and the tapped rows joining their slots' stores) are applied lazily, at the next forward: apply them
        // BEFORE looking at what each slot's store holds, or every stream looks one round behind its tokens and the head never gets to draft
        if head {
            m.flush()?;
        }
        // the copy drafter's offer is free: look at it for every stream that may use it
        let copy: Vec<Vec<u32>> = reqs
            .iter()
            .map(|r| if mode.uses_copy() { self.slots[r.slot].copy.propose(r.tokens, r.max) } else { Vec::new() })
            .collect();
        // what the head could do for each stream: its drafter state must be intact, the pending rows must match the sequence, and none of them may be
        // an image's (`spec_allowed_for_window`: an absorb from the tap store has only the token's embedding to offer)
        let ready: Vec<Option<usize>> = reqs
            .iter()
            .map(|r| {
                let spans = images.get(r.slot).map_or(&[][..], |v| v.as_slice());
                m.mtp_book(r.slot)
                    .as_ref()
                    .filter(|b| head && r.max > 0 && b.can_absorb(r.tokens.len()) && window_is_text(spans, b.first, b.n))
                    .map(|b| b.n)
            })
            .collect();
        let must_absorb: Vec<bool> = ready.iter().map(|r| r.is_some_and(|n| n + WINDOW_MAX > MTP_TAP_CAP)).collect();
        let picks: Vec<Choice> = match mode {
            SpecMode::Auto => {
                let asks: Vec<Ask> = reqs
                    .iter()
                    .enumerate()
                    .map(|(i, r)| Ask {
                        stream: r.slot,
                        max: r.max,
                        copy_len: copy[i].len().min(r.max),
                        mtp_ok: ready[i].is_some(),
                        absorb_rows: ready[i].unwrap_or(0),
                        must_absorb: must_absorb[i],
                    })
                    .collect();
                self.arbiter.choose(&self.cost, &asks)
            }
            SpecMode::Mtp => reqs
                .iter()
                .enumerate()
                .map(|(i, r)| if ready[i].is_some() { Choice::Mtp(self.cfg.mtp_depth.min(r.max)) } else { Choice::None })
                .collect(),
            SpecMode::Copy => {
                (0..reqs.len()).map(|i| if copy[i].is_empty() { Choice::None } else { Choice::Copy(copy[i].len()) }).collect()
            }
            SpecMode::None => vec![Choice::None; reqs.len()],
        };
        // run the head for the streams that draft with it, and, to make room in a store that would overflow, absorb-only for those that do not
        let mut asks: Vec<(usize, DraftAsk)> = Vec::new();
        for (i, (r, c)) in reqs.iter().zip(&picks).enumerate() {
            match *c {
                Choice::Mtp(d) => asks.push((i, DraftAsk { slot: r.slot, tokens: r.tokens, depth: d })),
                _ if must_absorb[i] => asks.push((i, DraftAsk { slot: r.slot, tokens: r.tokens, depth: 0 })),
                _ => {}
            }
        }
        let mut out: Vec<Option<Vec<u32>>> = vec![Some(Vec::new()); reqs.len()];
        let mut choice = picks.clone();
        if !asks.is_empty() {
            let list: Vec<DraftAsk> =
                asks.iter().map(|(_, a)| DraftAsk { slot: a.slot, tokens: a.tokens, depth: a.depth }).collect();
            // the keyed asks: a sampled request's draft is the target's draw over the head's candidates (the argmax for the others, and for every ask
            // when `REXL3_GLM_DRAFT_NOISE=0`)
            let keyed = keyed_samplers(self.cfg.noise, &asks.iter().map(|(i, _)| reqs[*i].sampling).collect::<Vec<_>>());
            let (streams, rows, depth) = (
                list.len(),
                asks.iter().map(|(i, _)| ready[*i].unwrap_or(0)).sum::<usize>(),
                asks.iter().map(|(_, a)| a.depth).max().unwrap_or(0),
            );
            let t = Instant::now();
            let drafts = m.mtp_draft_keyed(&list, &keyed)?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            self.stats.mtp_ms += ms;
            self.stats.mtp_calls += 1;
            self.stats.mtp_rows += rows as u64;
            cost_log(format_args!("D {streams} {rows} {depth} {ms:.4}"));
            self.cost.observe_draft(streams, rows, depth, ms);
            for ((i, a), d) in asks.iter().zip(drafts) {
                if a.depth == 0 {
                    continue;
                }
                match d {
                    Some(d) if !d.ids.is_empty() => out[*i] = Some(d.ids),
                    // the slot could not draft after all: the copy offer, if the mode has one, is all there is
                    _ => {
                        choice[*i] = Choice::None;
                    }
                }
            }
        }
        for i in 0..reqs.len() {
            match picks[i] {
                Choice::Copy(n) => {
                    out[i] = Some(copy[i][..n.min(copy[i].len())].to_vec());
                }
                Choice::Mtp(_) if choice[i] == Choice::None && !copy[i].is_empty() && mode != SpecMode::Mtp => {
                    out[i] = Some(copy[i].clone());
                    choice[i] = Choice::Copy(copy[i].len());
                }
                _ => {}
            }
        }
        for (i, r) in reqs.iter().enumerate() {
            let drafts = out[i].clone().unwrap_or_default();
            // the choice as it ended up: a pick whose offer came out empty was a serial round
            let c = match choice[i] {
                Choice::Copy(_) if drafts.is_empty() => Choice::None,
                Choice::Copy(_) => Choice::Copy(drafts.len()),
                Choice::Mtp(_) if drafts.is_empty() => Choice::None,
                Choice::Mtp(_) => Choice::Mtp(drafts.len()),
                Choice::None => Choice::None,
            };
            self.slots[r.slot].round = Some(Round { choice: c, drafts, verified: 0, margins: Vec::new() });
        }
        Ok(out)
    }

    /// [`SpecRuntime::propose`] with the DFlash2 drafter: every stream that may draft gets a chain of up to `min(max, 7)` drafts with the
    /// selector's margin of each, and the chains are cut by [`allocate_chains`] over the chances [`DflashCut`] gives the margins, priced with
    /// the verify table and the draft call's measured time. `--spec mtp` keeps the fixed depth (`REXL3_MTP_DEPTH`) as a cap instead.
    fn propose_dflash<M: DraftModel + ?Sized>(
        &mut self,
        m: &mut M,
        reqs: &[ProposeRequest],
    ) -> Result<Vec<Option<Vec<u32>>>, ModelError> {
        // the last window's accepted rows join the draft contexts first (the commit is applied lazily)
        m.flush()?;
        let cap = if self.cfg.mode == SpecMode::Mtp { self.cfg.mtp_depth.max(1) } else { crate::draft::DRAFTS };
        let asking: Vec<usize> = (0..reqs.len()).filter(|&i| reqs[i].max > 0 && reqs[i].tokens.len() > 1).collect();
        let mut out: Vec<Option<Vec<u32>>> = vec![Some(Vec::new()); reqs.len()];
        let mut margins: Vec<Vec<f32>> = vec![Vec::new(); reqs.len()];
        if !asking.is_empty() {
            let asks: Vec<DflashAsk> = asking
                .iter()
                .map(|&i| DflashAsk {
                    slot: reqs[i].slot,
                    tokens: reqs[i].tokens,
                    sampling: reqs[i].sampling,
                    want: reqs[i].max.min(cap),
                })
                .collect();
            let t = Instant::now();
            let picked = m.dflash_draft(&asks)?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            self.stats.mtp_ms += ms;
            self.stats.mtp_calls += 1;
            let streams = asks.len();
            self.dflash_ms = self.dflash_ms * DFLASH_MS_DECAY + ms * (1.0 - DFLASH_MS_DECAY);
            // each stream's chance of reaching each depth, from its picks' margins
            let reach: Vec<Vec<f64>> = picked
                .iter()
                .map(|p| {
                    let mut r = 1.0;
                    p.margin.iter().map(|&mg| {
                        r *= self.cut.chance(mg);
                        r
                    })
                    .collect()
                })
                .collect();
            let counts = if self.cfg.mode == SpecMode::Mtp {
                reach.iter().map(|r| r.len()).collect()
            } else {
                let cost = &self.cost;
                allocate_chains(&reach, super::alloc::WINDOW_ROWS, self.dflash_ms, &|s, rows| cost.verify_ms_of(s, rows))
            };
            if cost_log_on() {
                // each stream: slot:chain/kept/the chances of reaching each depth (`reach`, what the cut priced)
                let per: Vec<String> = asking
                    .iter()
                    .zip(&picked)
                    .zip(&counts)
                    .zip(&reach)
                    .map(|(((&i, p), n), r)| {
                        let r: Vec<String> = r.iter().map(|x| format!("{x:.3}")).collect();
                        format!("{}:{}/{n}/{}", reqs[i].slot, p.tokens.len(), r.join(";"))
                    })
                    .collect();
                cost_log(format_args!("D {streams} 0 {cap} {ms:.4} {}", per.join(",")));
            }
            for ((&i, p), n) in asking.iter().zip(picked).zip(counts) {
                let n = n.min(p.tokens.len());
                out[i] = Some(p.tokens[..n].to_vec());
                margins[i] = p.margin[..n].to_vec();
            }
        }
        for (i, r) in reqs.iter().enumerate() {
            let drafts = out[i].clone().unwrap_or_default();
            let choice = if drafts.is_empty() { Choice::None } else { Choice::Mtp(drafts.len()) };
            let margins = std::mem::take(&mut margins[i]);
            self.slots[r.slot].round = Some(Round { choice, drafts, verified: 0, margins });
        }
        Ok(out)
    }

    /// A verify window of `rows` rows in all took `ms` of wall time (the forward and its top-k).
    pub fn on_window(&mut self, rows: usize, ms: f64) {
        self.on_window_of(1, rows, ms);
    }

    /// [`SpecRuntime::on_window`] of a window of `streams` sequences (the per-sequence cost of a two-rank group's table).
    pub fn on_window_of(&mut self, streams: usize, rows: usize, ms: f64) {
        cost_log(format_args!("W {rows} {ms:.4} {streams} {:.3}", self.cost.verify_ms_of(streams, rows)));
        self.stats.window_ms += ms;
        self.stats.windows += 1;
        self.stats.window_rows += rows as u64;
        self.cost.observe_window_of(streams, rows, ms);
    }

    /// The engine put `rows` rows (the base row and the drafts it kept) in `slot`'s window.
    pub fn on_step(&mut self, slot: usize, rows: usize) {
        if let Some(r) = self.slots[slot].round.as_mut() {
            r.verified = rows.saturating_sub(1).min(r.drafts.len());
        }
    }

    /// `keep` rows of `slot`'s window were accepted: `keep - 1` drafts, and one sample past them.
    pub fn on_commit(&mut self, slot: usize, keep: usize) {
        let accepted = keep.saturating_sub(1);
        let s = &mut self.slots[slot];
        let (choice, verified, margins) = match s.round.take() {
            Some(r) => (if r.verified == 0 { Choice::None } else { r.choice }, r.verified, r.margins),
            None => (Choice::None, 0, Vec::new()),
        };
        let source = Source::of(choice);
        s.rounds += 1;
        s.committed += accepted as u64 + 1;
        s.by_source[source.idx()] += 1;
        let accepted = accepted.min(verified);
        cost_log(format_args!("C {slot} {verified} {accepted}"));
        self.stats.record(source, verified, accepted);
        if !margins.is_empty() {
            // a DFlash2 round: its margins teach the cut (the arbiter does not choose its depths)
            self.cut.observe(&margins, verified, accepted);
        } else if self.cfg.mode == SpecMode::Auto {
            self.arbiter.observe(slot, choice, verified, accepted);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stats_count_a_round_by_depth_and_source() {
        let mut s = SpecStats::default();
        // an MTP round of three drafts, the first two accepted: depth 1 and 2 hit, depth 3 was reached and missed
        s.record(Source::Mtp, 3, 2);
        assert_eq!((s.rounds[2], s.verified[2], s.accepted[2], s.committed[2]), (1, 3, 2, 3));
        assert_eq!(s.reached[2][..3], [1, 1, 1]);
        assert_eq!(s.hits[2][..3], [1, 1, 0]);
        // a round whose first draft was rejected never reaches depth 2
        s.record(Source::Mtp, 3, 0);
        assert_eq!(s.reached[2][..3], [2, 1, 1]);
        assert_eq!(s.hits[2][..3], [1, 1, 0]);
        assert!((s.depth_acceptance(Source::Mtp, 1) - 0.5).abs() < 1e-12);
        assert!((s.depth_acceptance(Source::Mtp, 2) - 1.0).abs() < 1e-12);
        assert!((s.tokens_per_round(Source::Mtp) - 2.0).abs() < 1e-12);
        // all accepted: every depth reached and hit
        s.record(Source::Copy, 2, 2);
        assert_eq!(s.reached[1][..2], [1, 1]);
        assert_eq!(s.hits[1][..2], [1, 1]);
        assert_eq!(s.committed[1], 3);
    }

    #[test]
    fn a_round_with_no_drafts_in_the_window_counts_as_none() {
        let mut r = SpecRuntime::new(SpecConfig::new(SpecMode::Mtp), 2);
        // the engine asked, the drafter offered two, the engine's budget trimmed them to nothing: the round is a serial one
        r.slots[1].round = Some(Round { choice: Choice::Mtp(2), drafts: vec![5, 6], verified: 0, margins: Vec::new() });
        r.on_step(1, 1);
        r.on_commit(1, 1);
        assert_eq!((r.stats.rounds[0], r.stats.rounds[2]), (1, 0));
        // a trimmed offer: two drafts offered, one in the window and accepted
        r.slots[0].round = Some(Round { choice: Choice::Mtp(2), drafts: vec![5, 6], verified: 0, margins: Vec::new() });
        r.on_step(0, 2);
        r.on_commit(0, 2);
        assert_eq!((r.stats.rounds[2], r.stats.verified[2], r.stats.accepted[2], r.stats.committed[2]), (1, 1, 1, 2));
        // a commit with no recorded round (the stream was never asked) is a serial round too, and the record is consumed
        r.on_commit(0, 1);
        assert_eq!(r.stats.rounds[0], 2);
        assert!(r.slots[0].round.is_none());
    }

    #[test]
    fn auto_feeds_what_happened_back_to_the_arbiter_and_the_fixed_modes_do_not() {
        let mut auto = SpecRuntime::new(SpecConfig::new(SpecMode::Auto), 1);
        let before = auto.arbiter.mtp_alpha(0, 1);
        for _ in 0..30 {
            auto.slots[0].round = Some(Round { choice: Choice::Mtp(2), drafts: vec![1, 2], verified: 0, margins: Vec::new() });
            auto.on_step(0, 3);
            auto.on_commit(0, 1); // both drafts rejected
        }
        assert!(auto.arbiter.mtp_alpha(0, 1) < before - 0.3, "{} -> {}", before, auto.arbiter.mtp_alpha(0, 1));
        let mut fixed = SpecRuntime::new(SpecConfig::new(SpecMode::Mtp), 1);
        for _ in 0..30 {
            fixed.slots[0].round = Some(Round { choice: Choice::Mtp(2), drafts: vec![1, 2], verified: 0, margins: Vec::new() });
            fixed.on_step(0, 3);
            fixed.on_commit(0, 1);
        }
        assert!((fixed.arbiter.mtp_alpha(0, 1) - before).abs() < 1e-12, "a fixed depth never learns: it is what was asked for");
    }

    #[test]
    fn the_noise_setting_is_validated_and_defaults_to_the_documented_pick() {
        assert_eq!(SpecConfig::new(SpecMode::Mtp).noise, DRAFT_NOISE_DEFAULT);
        assert_eq!(parse_noise(None), Ok(DRAFT_NOISE_DEFAULT));
        assert_eq!(parse_noise(Some("0")), Ok(DraftPick::Argmax));
        assert_eq!(parse_noise(Some("1")), Ok(DraftPick::Keyed));
        let e = parse_noise(Some("2")).unwrap_err();
        assert!(e.contains("REXL3_GLM_DRAFT_NOISE=2"), "{e}");
        assert!(parse_noise(Some("")).is_err(), "an empty value is not a choice");
    }

    #[test]
    fn the_noise_setting_decides_which_asks_carry_their_requests_sampler() {
        let (a, b) = (
            Sampling { seed: 1, temperature: 0.7, ..Default::default() },
            Sampling { seed: 2, temperature: 0.0, ..Default::default() },
        );
        assert_eq!(keyed_samplers(DraftPick::Keyed, &[&a, &b]), vec![Some(a), Some(b)]);
        assert_eq!(keyed_samplers(DraftPick::Argmax, &[&a, &b]), vec![None, None]);
        assert!(keyed_samplers(DraftPick::Keyed, &[]).is_empty());
    }

    #[test]
    fn the_depth_setting_is_validated() {
        assert_eq!(SpecConfig::new(SpecMode::Mtp).mtp_depth, MTP_DEFAULT_DEPTH);
        assert_eq!(parse_depth("3"), Ok(3));
        assert_eq!(parse_depth(" 4 "), Ok(4));
        assert!(parse_depth("0").unwrap_err().contains("1..=4"));
        assert!(parse_depth("5").unwrap_err().contains("got 5"));
        assert!(parse_depth("two").unwrap_err().contains("not a number"));
    }

    #[test]
    fn the_cost_model_follows_the_table_and_scales_with_what_was_observed_within_bounds() {
        let mut c = GlmCost::new(MTP_COST);
        assert!(
            (c.verify_ms(1) - 38.7).abs() < 1e-9
                && (c.verify_ms(3) - 56.2).abs() < 1e-9
                && (c.verify_ms(64) - 316.7).abs() < 1e-9
        );
        // the window is concave: the second row costs more than the tenth
        assert!(c.verify_ms(2) - c.verify_ms(1) > c.verify_ms(10) - c.verify_ms(9));
        // a window slower than the table moves the scale up, by a smoothed step, to a limit
        for _ in 0..200 {
            c.observe_window(3, 56.2 * 10.0);
        }
        assert!((c.verify_scale - SCALE_RANGE.1).abs() < 1e-9, "{}", c.verify_scale);
        for _ in 0..200 {
            c.observe_window(3, 1.0);
        }
        assert!((c.verify_scale - SCALE_RANGE.0).abs() < 1e-9);
        // nonsense is ignored
        let s = c.verify_scale;
        c.observe_window(3, f64::NAN);
        c.observe_window(3, -4.0);
        assert_eq!(c.verify_scale, s);
        // a draft call costs more for more streams, more pending rows and more depth
        // the measured corners (`mtp_draft_costs` medians, anonymized host-d, after the decode-step levers): one stream one row at depths 0, 1, 4; eight streams of
        // eight rows at depths 0 and 4; four streams of two rows at depth 2; for the full head and for the restricted one
        let full = [
            (1usize, 1usize, 0usize, 1.586),
            (1, 1, 1, 3.670),
            (1, 1, 4, 14.056),
            (8, 8, 0, 15.236),
            (8, 8, 4, 37.059),
            (4, 2, 2, 11.363),
        ];
        let restricted = [
            (1usize, 1usize, 0usize, 1.631),
            (1, 1, 1, 2.557),
            (1, 1, 4, 9.435),
            (8, 8, 0, 15.196),
            (8, 8, 4, 31.840),
            (4, 2, 2, 8.965),
        ];
        for (m, corners, what) in [(MTP_COST, full, "full head"), (MTP_COST_RESTRICTED, restricted, "restricted head")] {
            assert!(m.ms(2, 4, 2) > m.ms(1, 4, 2) && m.ms(1, 8, 2) > m.ms(1, 4, 2) && m.ms(1, 4, 3) > m.ms(1, 4, 2), "{what}");
            assert!(m.ms(1, 4, 0) < m.ms(1, 4, 1), "{what}: absorb-only has no head");
            for (s, r, d, measured) in corners {
                let model = m.ms(s, s * r, d);
                assert!(
                    (model - measured).abs() / measured < 0.12,
                    "{what}: {s} streams x {r} rows, depth {d}: model {model:.2} ms against {measured} measured"
                );
            }
            // concave in the pending rows, like the window itself
            assert!(m.ms(1, 2, 0) - m.ms(1, 1, 0) > m.ms(1, 64, 0) - m.ms(1, 63, 0), "{what}");
        }
        // the restricted head saves its share of every head the call runs: one in the first draft, one in each chained draft
        assert!(MTP_COST_RESTRICTED.ms(1, 3, 2) < MTP_COST.ms(1, 3, 2) - 2.0);
        assert!(MTP_COST_RESTRICTED.ms(1, 3, 0) <= MTP_COST.ms(1, 3, 0) + 0.2, "an absorb alone runs no head");
    }

    #[test]
    fn a_two_rank_group_prices_with_its_own_tables_and_one_node_keeps_the_old_ones() {
        let cfg =
            |two_ranks: bool, restricted_head: bool| SpecConfig { two_ranks, restricted_head, ..SpecConfig::new(SpecMode::Auto) };
        assert!(!SpecConfig::new(SpecMode::Auto).two_ranks, "one node unless the TP boot says otherwise");
        for (two, restricted, verify, mtp) in [
            (false, false, VERIFY_MS, MTP_COST),
            (false, true, VERIFY_MS, MTP_COST_RESTRICTED),
            (true, false, VERIFY_MS_TP, MTP_COST_TP),
            (true, true, VERIFY_MS_TP, MTP_COST_TP_RESTRICTED),
        ] {
            let c = GlmCost::for_config(&cfg(two, restricted));
            for &(rows, ms) in verify {
                assert!((c.verify_ms(rows) - ms).abs() < 1e-9, "two_ranks={two}: {rows} rows");
            }
            for (s, r, d) in [(1, 1, 0), (1, 2, 1), (2, 5, 3), (2, 10, 4)] {
                assert!((c.mtp_ms(s, r, d) - mtp.ms(s, r, d)).abs() < 1e-9, "two_ranks={two} restricted={restricted}");
            }
        }
        // the restricted head stays the cheaper draft on two ranks too
        assert!(MTP_COST_TP_RESTRICTED.ms(1, 2, 2) < MTP_COST_TP.ms(1, 2, 2));
        // a window of several sequences costs VERIFY_PER_STREAM_MS_TP a sequence more on two ranks, nothing more on one node
        let tp = GlmCost::for_config(&cfg(true, true));
        assert!((tp.verify_ms_of(3, 8) - (tp.verify_ms(8) + 2.0 * VERIFY_PER_STREAM_MS_TP)).abs() < 1e-9);
        assert!((tp.verify_ms_of(1, 8) - tp.verify_ms(8)).abs() < 1e-9);
        let one = GlmCost::for_config(&cfg(false, true));
        assert!((one.verify_ms_of(5, 8) - one.verify_ms(8)).abs() < 1e-9);
        // observing a 3-sequence window that took exactly the model's figure leaves the scale at 1
        let mut t2 = GlmCost::for_config(&cfg(true, true));
        t2.observe_window_of(3, 8, VERIFY_MS_TP.iter().find(|p| p.0 == 8).unwrap().1 + 2.0 * VERIFY_PER_STREAM_MS_TP);
        assert!((t2.verify_scale - 1.0).abs() < 1e-9);
    }

    #[test]
    fn the_cut_learns_the_acceptance_of_a_margin_from_the_drafts_that_were_reached() {
        let mut c = DflashCut::default();
        // the prior rises with the margin
        assert!(c.chance(0.1) < c.chance(2.0) && c.chance(2.0) < c.chance(25.0));
        let (lo, hi) = (c.chance(0.1), c.chance(25.0));
        // a round of three drafts with margins 5, 5, 0.1: the first two accepted, the third rejected
        for _ in 0..200 {
            c.observe(&[5.0, 5.0, 0.1], 3, 2);
        }
        assert!(c.chance(5.0) > 0.97, "{}", c.chance(5.0));
        assert!(c.chance(0.1) < 0.05, "{}", c.chance(0.1));
        // a draft past the first rejected one was never reached: it teaches nothing
        let mut d = DflashCut::default();
        d.observe(&[0.1, 25.0, 25.0], 3, 0);
        assert!(d.chance(0.1) < lo);
        assert_eq!(d.chance(25.0), hi);
        // nor does a draft the window did not verify
        d.observe(&[25.0, 25.0], 0, 0);
        assert_eq!(d.chance(25.0), hi);
    }

    #[test]
    fn chains_are_cut_where_one_more_row_stops_paying_for_itself() {
        let reach = |p: &[f64]| {
            let mut r = 1.0;
            p.iter()
                .map(|&x| {
                    r *= x;
                    r
                })
                .collect::<Vec<f64>>()
        };
        let tp = GlmCost::for_config(&SpecConfig { two_ranks: true, ..SpecConfig::new(SpecMode::Auto) });
        let cost = |s: usize, rows: usize| tp.verify_ms_of(s, rows);
        // confident drafts all the way: the whole chain is verified
        assert_eq!(allocate_chains(&[reach(&[0.98; 7])], 64, 5.0, &cost), vec![7]);
        // a chain that falls off a cliff after two drafts stops there
        assert_eq!(allocate_chains(&[reach(&[0.95, 0.95, 0.05, 0.9, 0.9, 0.9, 0.9])], 64, 5.0, &cost), vec![2]);
        // nothing worth a row: a serial round
        assert_eq!(allocate_chains(&[reach(&[0.02; 7])], 64, 5.0, &cost), vec![0]);
        // two streams share the window: the likelier chain gets the deeper offer
        let two = allocate_chains(&[reach(&[0.95; 7]), reach(&[0.6; 7])], 64, 5.0, &cost);
        assert!(two[0] > two[1], "{two:?}");
        // the window's row cap holds
        let many: Vec<Vec<f64>> = (0..16).map(|_| reach(&[0.99; 7])).collect();
        assert!(allocate_chains(&many, 64, 5.0, &cost).iter().sum::<usize>() + 16 <= 64);
    }
}
