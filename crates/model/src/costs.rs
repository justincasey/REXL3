//! What a decode round costs on the GB10 for the Qwen3.8-27B EXL3 4.00bpw target with its DFlash2 draft, as the speculation budget
//! reads it, and the serve-time switches that choose the budget policy and the draft's vocabulary.
//!
//! Every number is a measurement from `crates/model/tests/bench.rs` (see `evals/b6_bench.sh`), not an estimate:
//!
//! * the verify step by (streams, rows): `costs_grid.rs`, generated from `REXL3_BENCH_GRID` by `tools/gen_cost_grid.py`: 1 to 32 streams and up
//!   to 128 rows (the wide grid, recorded in the maintainers' working repository; the 260 points of 1 to 8 streams and up to 64 rows were re-timed within 0.54% on average and keep their values);
//! * the verify step of a window of TREES: `costs_grid_tree.rs` (`TREE_STEP_GRID`, random trees of branching 3, 1 to 8 streams and up to 64 rows)
//!   within its streams, and past them (9 to 32 streams) the grid above plus the tree's premium over a chain (`RoundCost::tree_step_ms`); the
//!   tree planner plans at most 64 rows within the tree table's streams (`RoundCost::tree_window_rows`) and the engine's window past them;
//! * attention over the contexts: `KV_MS_PER_TOKEN`, and `KV_PASS_MS_PER_TOKEN` for each further block of 8 rows of a tree window;
//! * the draft: its layers by number of drafting streams and its vocabulary head by rows (`draft_time_by_streams`).
//!
//! A kernel change moves these (the verify step has a cliff just past 32 rows and the draft head another), so re-measure after one.

use crate::costs_grid::STEP_GRID;
use crate::costs_grid_tree::TREE_STEP_GRID;
use crate::error::ModelError;
use gb_engine::{BudgetPolicy, PrefillPolicy};
use gb_sched::{DraftCost, RoundCost, StepCost};

/// Extra verify-step milliseconds per token of context held by the round's streams: attention reads each stream's KV once per round
/// whatever the number of rows (1 row and 8 rows cost the same extra), at about 64 KiB a token and ~170 GB/s
/// (bench: 6.0 ms for 16,384 tokens, 12 ms for two such streams, 6.4 ms for 24,576 tokens over six).
pub const KV_MS_PER_TOKEN: f64 = 0.00038;

/// Extra milliseconds per token of a stream's context for every further block of 8 rows its window holds, on top of what those rows cost at a
/// short context (the attention serves 8 rows a CTA, so a window of 9 to 16 rows reads the committed KV twice). Bench
/// `evals/tree_s2_longctx_ab.sh`, 1 stream, tree windows of 8 / 16 / 24 / 32 rows: with the row blocks of one chunk launched far apart
/// (before) 73.6 / 83.9 / 95.7 / 106.6 ms at 16,384 tokens and 94.0 / 126.2 / 158.0 / 190.5 at 65,536, 0.0005 ms a token a block; with them
/// launched together (the tree kernel now) 73.6 / 78.9 / 85.0 / 90.3 and 96.9 / 103.1 / 111.0 / 118.7: 1.2 to 1.8 ms a block over the short-
/// context table at 16k and 2.7 to 3.8 at 64k, about 0.00007 ms a token.
pub const KV_PASS_MS_PER_TOKEN: f64 = 0.00007;

/// The draft's layers (everything but the vocabulary head), by streams in ONE draft round (at most `MAX_DRAFT_SEQS` = 16; `DraftCost::with_group`
/// prices more as further rounds): `(streams, ms)`. From the draft round asked for 4 picks a sequence (`draft_time_by_streams`) less the head's time
/// at those rows, the mean of two holds for 1 to 6 streams; 8, 12 and 16 are one hold on a quiet box (recorded in the maintainers' working repository: anonymized host-a, load 0.4;
/// the 1 to 6 re-timed within 3% of the old values). A stream is about 0.5 ms (more past 6: the layers pass 32 rows at 5 streams, 64 at 8).
const DRAFT_LAYERS: &[(usize, f64)] =
    &[(1, 4.9), (2, 5.45), (3, 6.0), (4, 6.4), (5, 7.25), (6, 7.8), (8, 8.7), (12, 12.9), (16, 15.4)];
/// The draft's vocabulary head over the full 248,320 columns, by rows: `(rows, ms)`, the mean of two holds. Flat to 32 rows, then a
/// step of 0.2 ms (the 4-pass decode kernel made it: it was 4.2 -> 6.3 ms with the 3-pass kernel). **Past 64 rows it steps by 2.8 ms** (4.5 -> 7.3 at the
/// 65th: the second tile-decode group of the EXL3 decode kernel, `group_tiles`); the rows from 56 up are one hold on a quiet box.
const HEAD_FULL: &[(usize, f64)] = &[
    (1, 4.18),
    (7, 4.17),
    (16, 4.13),
    (32, 4.16),
    (33, 4.39),
    (42, 4.37),
    (48, 4.38),
    (56, 4.46),
    (64, 4.50),
    (65, 7.29),
    (80, 7.39),
    (96, 7.60),
    (97, 7.82),
    (112, 7.98),
    (128, 8.12),
];
/// The same over TensorFold's candidate set (98,688 of the columns), measured: 40% of the columns, 40% of the time (the two holds'
/// mean; the step past 32 rows is 0.1 ms, it was 0.9; the 65th row costs +1.0 ms).
const HEAD_TF: &[(usize, f64)] = &[
    (1, 1.64),
    (7, 1.69),
    (16, 1.70),
    (26, 1.74),
    (32, 1.74),
    (33, 1.83),
    (42, 1.83),
    (48, 1.84),
    (64, 1.93),
    (65, 2.97),
    (96, 3.09),
    (128, 3.38),
];
/// Share of the full head's time that does not shrink with the columns read (launches, the rotation kernels), for a custom set.
const HEAD_FLOOR: f64 = 0.04;
/// Work every round does that the bench does not time (see `RoundCost::fixed_ms`): live rounds of one stream ran 0.55 ms (0.7%) over
/// the model, and the gap has no trend with the stream count (`tools/b6_calibrate.py`; 1, 2, 4 and 6 streams within 1.5%).
const ROUND_FIXED_MS: f64 = 0.5;

/// The cost of a round: verify step, draft, attention. `head` is the draft head's table (see [`head_table`]); `None` leaves the
/// draft out (no native draft: the copy drafter costs nothing worth counting). `block` is the draft's block (8 the trained value,
/// 16 the depth block's `REXL3_DRAFT_BLOCK`): `DraftCost::with_block` reads the measured tables at the block's scale and halves
/// the sequences a round carries.
pub fn round_cost(head: Option<Vec<(usize, f64)>>, block: usize) -> RoundCost {
    RoundCost {
        verify: StepCost::from_grid(STEP_GRID.to_vec()),
        tree_verify: Some(StepCost::from_grid(TREE_STEP_GRID.to_vec())),
        // the draft runs `MAX_DRAFT_SEQS` sequences a round at the trained block (`Model::propose_drafts_batch_want`): 17 or more
        // drafting streams are two rounds, and a 16-row block halves the width (`DraftCost::with_block`)
        draft: head
            .map(|h| DraftCost::new(DRAFT_LAYERS.to_vec(), h, 0.0).with_group(crate::draft::MAX_DRAFT_SEQS).with_block(block)),
        kv_ms_per_token: KV_MS_PER_TOKEN,
        kv_pass_ms_per_token: KV_PASS_MS_PER_TOKEN,
        fixed_ms: ROUND_FIXED_MS,
    }
}

/// The draft head's cost by rows for a draft vocabulary that reads `fraction` of the target head's columns: the measured table for
/// the full head and for TensorFold's set, and for any other set the full table scaled by the columns read (the head is bandwidth
/// bound) plus a small launch floor.
pub fn head_table(vocab: &DraftVocab, fraction: f64) -> Vec<(usize, f64)> {
    match vocab {
        DraftVocab::Full => HEAD_FULL.to_vec(),
        DraftVocab::Tf | DraftVocab::Auto => HEAD_TF.to_vec(),
        DraftVocab::Ranges(_) => {
            let f = fraction.clamp(0.0, 1.0);
            HEAD_FULL.iter().map(|&(rows, ms)| (rows, ms * (HEAD_FLOOR + (1.0 - HEAD_FLOOR) * f))).collect()
        }
    }
}

/// The budget policy a served Qwen draft runs with unless `REXL3_SPEC_BUDGET` says otherwise: trim each round's verify rows after
/// the draft from the draft's own margin. Measured in `docs/EVAL_RESULTS.md` (B6): verify ms -7% and aggregate tokens per second
/// +2.6% at 6 streams on the full head, flat at one stream, replies identical to every other policy's.
pub const DEFAULT_BUDGET: BudgetPolicy = BudgetPolicy::Margin;

/// Which token ids the draft may propose (`REXL3_DRAFT_VOCAB`; unset means [`DraftVocab::Auto`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DraftVocab {
    /// Every token: the draft's head reads all of the target's `lm_head`.
    Full,
    /// TensorFold's set for this family: ids `[0, 98304)` and the specials at the top. 40% of the head's columns. Every round.
    Tf,
    /// TensorFold's set, used only in rounds where every stream is writing text it can reach (at least 98% of its last 64 tokens
    /// inside the set) and the full head otherwise: English and code get the faster head, Chinese, Russian, Arabic and Korean
    /// (which `Tf` costs a quarter to a third of their acceptance) keep the full one.
    Auto,
    /// `lo-hi,lo-hi`: half-open ranges of token ids.
    Ranges(Vec<(usize, usize)>),
}

/// Share of a stream's recent tokens that must lie inside a restricted draft vocabulary for [`DraftVocab::Auto`] to use it.
pub const AUTO_MIN_IN_SET: f32 = 0.98;

impl DraftVocab {
    pub fn parse(s: &str) -> Result<Self, ModelError> {
        match s.trim() {
            "full" => Ok(Self::Full),
            "" | "auto" => Ok(Self::Auto),
            "tf" => Ok(Self::Tf),
            list => {
                let mut v = Vec::new();
                for part in list.split(',') {
                    let (lo, hi) = part.split_once('-').ok_or_else(|| bad_vocab(list))?;
                    v.push((lo.trim().parse().map_err(|_| bad_vocab(list))?, hi.trim().parse().map_err(|_| bad_vocab(list))?));
                }
                Ok(Self::Ranges(v))
            }
        }
    }

    /// `REXL3_DRAFT_VOCAB`, or the default (`auto`).
    pub fn from_env() -> Result<Self, ModelError> {
        Self::parse(&std::env::var("REXL3_DRAFT_VOCAB").unwrap_or_default())
    }

    /// The token-id ranges to restrict the draft head to; None for the full vocabulary.
    pub fn ranges(&self) -> Option<Vec<(usize, usize)>> {
        match self {
            Self::Full => None,
            Self::Tf | Self::Auto => Some(crate::draft::TF_DRAFT_VOCAB.to_vec()),
            Self::Ranges(v) => Some(v.clone()),
        }
    }

    pub fn name(&self) -> String {
        match self {
            Self::Full => "full".into(),
            Self::Tf => "tf".into(),
            Self::Auto => "auto".into(),
            Self::Ranges(v) => v.iter().map(|(a, b)| format!("{a}-{b}")).collect::<Vec<_>>().join(","),
        }
    }
}

fn bad_vocab(s: &str) -> ModelError {
    ModelError::Config(format!("REXL3_DRAFT_VOCAB={s:?}: expected `full`, `tf`, `auto` or ranges like `0-98304,248032-248320`"))
}

/// `REXL3_SPEC_BUDGET`: `legacy`, `curve`, `plan`, `margin` or `scored` (see [`BudgetPolicy`]); unset keeps `default`.
pub fn budget_policy_from_env(default: BudgetPolicy) -> Result<BudgetPolicy, ModelError> {
    match std::env::var("REXL3_SPEC_BUDGET") {
        Err(_) => Ok(default),
        Ok(s) if s.trim().is_empty() => Ok(default),
        Ok(s) => BudgetPolicy::parse(s.trim()).ok_or_else(|| {
            ModelError::Config(format!("REXL3_SPEC_BUDGET={s:?}: expected `legacy`, `curve`, `plan`, `margin` or `scored`"))
        }),
    }
}

/// `REXL3_PREFILL_BURST`: how prefill slices are shared out among waiting prompts (see [`PrefillPolicy`]). Unset or `default` keeps
/// the family's own default (Qwen passes `PrefillPolicy::BATCHED_SHORT`); `serial` is one slice a step with the prompts in turn (the first release's behaviour); `DIV,CAP` is the burst policy with
/// `ceil(waiting / DIV)` slices a step, at most `CAP`. Replies are byte-identical under every setting: only the time to a first token moves.
pub fn prefill_policy_from_env(default: PrefillPolicy) -> Result<PrefillPolicy, ModelError> {
    parse_prefill_policy(std::env::var("REXL3_PREFILL_BURST").ok().as_deref(), default)
}

/// The prefill policy the Qwen engine runs with: its own default (the measured burst policy, which its backend's batched short-prompt prefill
/// is built for) unless `REXL3_PREFILL_BURST` says otherwise. Another family's engine does not call this and keeps one slice a step.
pub fn qwen_prefill_policy() -> Result<PrefillPolicy, ModelError> {
    prefill_policy_from_env(PrefillPolicy::BATCHED_SHORT)
}

fn parse_prefill_policy(value: Option<&str>, default: PrefillPolicy) -> Result<PrefillPolicy, ModelError> {
    let bad = |s: &str| {
        ModelError::Config(format!(
            "REXL3_PREFILL_BURST={s:?}: expected `serial`, `default` or `DIV,CAP` (two numbers, at least 1)"
        ))
    };
    match value.map(str::trim) {
        None | Some("") | Some("default") => Ok(default),
        Some("serial") => Ok(PrefillPolicy::Serial),
        Some(s) => {
            let (d, c) = s.split_once(',').ok_or_else(|| bad(s))?;
            let (div, cap): (usize, usize) = (d.trim().parse().map_err(|_| bad(s))?, c.trim().parse().map_err(|_| bad(s))?);
            if div == 0 || cap == 0 {
                return Err(bad(s));
            }
            Ok(PrefillPolicy::Burst { div, cap })
        }
    }
}

/// `REXL3_PREFILL_BATCH`: `0`/`off` hands the slices of a burst to the backend one by one (the behaviour before batched prefill);
/// unset, `1` or `on` lets the backend put the short prompts of a burst through one forward. Replies are the same either way.
pub fn prefill_batching_from_env() -> Result<bool, ModelError> {
    parse_prefill_batching(std::env::var("REXL3_PREFILL_BATCH").ok().as_deref())
}

fn parse_prefill_batching(value: Option<&str>) -> Result<bool, ModelError> {
    match value.map(str::trim) {
        None | Some("") | Some("1") | Some("true") | Some("on") => Ok(true),
        Some("0") | Some("false") | Some("off") => Ok(false),
        Some(other) => Err(ModelError::Config(format!("REXL3_PREFILL_BATCH={other:?}: expected 1/true/on or 0/false/off"))),
    }
}

/// `REXL3_PREFILL_GATHER_MS`: how long a request that wakes an idle engine waits for the requests sent with it, so their prompts are
/// filled together (`Engine::with_gather_window`); unset: `default_ms` (the family's), `0`: none. Never a bit.
pub fn prefill_gather_from_env(default_ms: u64) -> Result<std::time::Duration, ModelError> {
    parse_prefill_gather(std::env::var("REXL3_PREFILL_GATHER_MS").ok().as_deref(), default_ms)
}

fn parse_prefill_gather(value: Option<&str>, default_ms: u64) -> Result<std::time::Duration, ModelError> {
    let ms = match value.map(str::trim) {
        None | Some("") => default_ms,
        Some(s) => s.parse::<u64>().map_err(|_| ModelError::Config(format!("REXL3_PREFILL_GATHER_MS={s:?}: expected milliseconds")))?,
    };
    Ok(std::time::Duration::from_millis(ms))
}

/// `REXL3_SPEC_LEARN`: `1`/`true`/`on` lets the budget learn its priors from the traffic; unset, `0`, `false` or `off` keeps the
/// measured tables.
pub fn spec_learn_from_env() -> Result<bool, ModelError> {
    parse_switch("REXL3_SPEC_LEARN", std::env::var("REXL3_SPEC_LEARN").ok().as_deref())
}

/// `REXL3_TREE_NODES` (most nodes one stream's draft tree is searched to, 1..=63), `REXL3_TREE_BRANCH` (children expanded under a node,
/// 1..=8) and `REXL3_TREE_COPY` (shortest match of the copy branch, 0 for none, else 2..=16) tune `--spec-tree` for measurements; unset keeps
/// the defaults (`TreeConfig::default`). The search's other constants (edge, noise and scale weights) are not knobs: the calibration table is
/// only valid for them.
pub fn tree_config_from_env() -> Result<gb_engine::TreeConfig, ModelError> {
    parse_tree_config(
        std::env::var("REXL3_TREE_NODES").ok().as_deref(),
        std::env::var("REXL3_TREE_BRANCH").ok().as_deref(),
        std::env::var("REXL3_TREE_COPY").ok().as_deref(),
    )
}

fn parse_tree_config(nodes: Option<&str>, branch: Option<&str>, copy: Option<&str>) -> Result<gb_engine::TreeConfig, ModelError> {
    let mut cfg = gb_engine::TreeConfig::default();
    let num = |name: &str, v: &str, lo: usize, hi: usize| -> Result<usize, ModelError> {
        v.trim()
            .parse::<usize>()
            .ok()
            .filter(|n| (lo..=hi).contains(n))
            .ok_or_else(|| ModelError::Config(format!("{name}={v:?}: expected a number from {lo} to {hi}")))
    };
    if let Some(v) = nodes.filter(|v| !v.trim().is_empty()) {
        cfg.max_nodes = num("REXL3_TREE_NODES", v, 1, 63)?;
    }
    if let Some(v) = branch.filter(|v| !v.trim().is_empty()) {
        cfg.policy.branch = num("REXL3_TREE_BRANCH", v, 1, 8)?;
    }
    if let Some(v) = copy.filter(|v| !v.trim().is_empty()) {
        cfg.copy_min_match = if v.trim() == "0" { 0 } else { num("REXL3_TREE_COPY", v, 2, 16)? };
    }
    Ok(cfg)
}

fn parse_switch(name: &str, value: Option<&str>) -> Result<bool, ModelError> {
    match value.map(str::trim) {
        None | Some("") | Some("0") | Some("false") | Some("off") => Ok(false),
        Some("1") | Some("true") | Some("on") => Ok(true),
        Some(other) => Err(ModelError::Config(format!("{name}={other:?}: expected 1/true/on or 0/false/off"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_knobs_default_to_the_frozen_study_and_refuse_nonsense() {
        let d = gb_engine::TreeConfig::default();
        assert_eq!(
            d.copy_min_match, 0,
            "the copy branch is off by default (docs/TREE_DRAFTING.md section 10): REXL3_TREE_COPY=4 turns it on"
        );
        assert_eq!(parse_tree_config(None, None, None).unwrap(), d);
        assert_eq!(parse_tree_config(Some(""), Some(" "), Some("")).unwrap(), d);
        let c = parse_tree_config(Some("24"), Some("3"), Some("4")).unwrap();
        assert_eq!((c.max_nodes, c.policy.branch, c.copy_min_match), (24, 3, 4));
        assert_eq!(parse_tree_config(None, None, Some("0")).unwrap().copy_min_match, 0);
        assert_eq!((c.policy.edge, c.policy.noise, c.policy.scale), (d.policy.edge, d.policy.noise, d.policy.scale));
        for bad in ["0", "64", "-1", "x", "1.5"] {
            assert!(parse_tree_config(Some(bad), None, None).is_err(), "{bad}");
        }
        for bad in ["0", "9", "x"] {
            assert!(parse_tree_config(None, Some(bad), None).is_err(), "{bad}");
        }
        for bad in ["1", "17", "x"] {
            assert!(parse_tree_config(None, None, Some(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn tree_windows_have_their_own_table_that_reaches_the_whole_window_and_tracks_the_chain_where_both_are_measured() {
        let c = round_cost(None, 8);
        let tree = c.tree_verify.as_ref().expect("the served round cost prices tree windows with the tree table");
        // the chain's grid stops at 8 rows a stream; the tree table runs to the 64 rows of a window for every stream count measured
        for streams in [1usize, 2, 3, 4, 6, 8] {
            let top = 64 / streams;
            assert!(tree.ms(streams * top, streams) > tree.ms(streams, streams), "{streams} streams: more rows cost more");
        }
        // a tree window over the chain's own range costs what the chain window does, within a few percent (bench: +0.5 ms at 8 rows)
        for (rows, streams) in [(8usize, 1usize), (16, 2), (32, 4), (48, 6)] {
            let (t, ch) = (tree.ms(rows, streams), c.verify.ms(rows, streams));
            assert!((t / ch - 1.0).abs() < 0.04, "{rows} rows over {streams} streams: tree {t:.1} ms against chain {ch:.1} ms");
        }
        // and the tree table, not the chain's extrapolation, prices a 60-row window at 6 streams (the chain grid ends at 48 rows)
        assert!(
            tree.ms(60, 6) > c.verify.ms(60, 6) + 3.0,
            "tree {:.1} ms against the extrapolated chain {:.1} ms",
            tree.ms(60, 6),
            c.verify.ms(60, 6)
        );
    }

    #[test]
    fn the_learning_switch_is_off_unless_asked_for() {
        assert!(!parse_switch("X", None).unwrap());
        assert!(!parse_switch("X", Some("")).unwrap());
        assert!(!parse_switch("X", Some("off")).unwrap());
        assert!(parse_switch("X", Some("1")).unwrap());
        assert!(parse_switch("X", Some("on")).unwrap());
        assert!(parse_switch("X", Some("maybe")).is_err());
    }

    #[test]
    fn the_prefill_policy_switch_parses_serial_default_and_two_numbers() {
        let d = PrefillPolicy::BATCHED_SHORT;
        assert_eq!(parse_prefill_policy(None, d).unwrap(), d);
        assert_eq!(parse_prefill_policy(Some(""), d).unwrap(), d);
        assert_eq!(parse_prefill_policy(Some("default"), d).unwrap(), d);
        assert_eq!(parse_prefill_policy(Some("serial"), d).unwrap(), PrefillPolicy::Serial);
        assert_eq!(parse_prefill_policy(Some("2, 16"), d).unwrap(), PrefillPolicy::Burst { div: 2, cap: 16 });
        for bad in ["0,8", "4,0", "4", "a,b", "4,8,2", "burst"] {
            assert!(parse_prefill_policy(Some(bad), d).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_qwen_engine_opts_in_to_the_burst_policy_and_the_engine_default_stays_serial() {
        // the opt-in is the family's: `Engine::new` alone is serial (crates/engine/tests/prefill_batch.rs), Qwen's serve path asks for this
        if std::env::var_os("REXL3_PREFILL_BURST").is_none() {
            assert_eq!(qwen_prefill_policy().unwrap(), PrefillPolicy::Burst { div: 2, cap: 16 });
        }
        assert_ne!(PrefillPolicy::default(), PrefillPolicy::BATCHED_SHORT);
    }

    #[test]
    fn prefill_batching_is_on_unless_switched_off() {
        for on in [None, Some(""), Some("1"), Some("on"), Some("true")] {
            assert!(parse_prefill_batching(on).unwrap(), "{on:?}");
        }
        for off in [Some("0"), Some("off"), Some("false")] {
            assert!(!parse_prefill_batching(off).unwrap(), "{off:?}");
        }
        assert!(parse_prefill_batching(Some("maybe")).is_err());
    }

    #[test]
    fn the_prefill_gather_is_the_familys_unless_set_in_milliseconds() {
        let ms = std::time::Duration::from_millis;
        assert_eq!(parse_prefill_gather(None, 15).unwrap(), ms(15));
        assert_eq!(parse_prefill_gather(Some(" "), 15).unwrap(), ms(15));
        assert_eq!(parse_prefill_gather(Some("0"), 15).unwrap(), ms(0));
        assert_eq!(parse_prefill_gather(Some("40"), 15).unwrap(), ms(40));
        assert!(parse_prefill_gather(Some("soon"), 15).is_err());
    }

    #[test]
    fn the_draft_vocabulary_switch_parses_names_and_ranges() {
        // unset is the adaptive set; `full` is the way back to every column
        assert_eq!(DraftVocab::parse("").unwrap(), DraftVocab::Auto);
        assert_eq!(DraftVocab::parse("full").unwrap(), DraftVocab::Full);
        assert!(DraftVocab::Full.ranges().is_none());
        assert_eq!(DraftVocab::parse("tf").unwrap().ranges().unwrap(), vec![(0, 98304), (248032, 248320)]);
        // auto is TensorFold's set used adaptively: the same ranges and the same measured head
        let auto = DraftVocab::parse("auto").unwrap();
        assert_eq!(auto.ranges(), DraftVocab::Tf.ranges());
        assert_eq!(auto.name(), "auto");
        assert_eq!(head_table(&auto, 0.397), head_table(&DraftVocab::Tf, 0.397));
        assert_eq!(DraftVocab::parse("0-100, 200-300").unwrap(), DraftVocab::Ranges(vec![(0, 100), (200, 300)]));
        assert_eq!(DraftVocab::parse("0-100,200-300").unwrap().name(), "0-100,200-300");
        for bad in ["half", "0-", "a-b", "100"] {
            assert!(DraftVocab::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_restricted_vocabulary_head_is_cheaper_by_about_the_columns_it_skips() {
        let full = head_table(&DraftVocab::Full, 1.0);
        let tf = head_table(&DraftVocab::Tf, 0.397);
        let at = |t: &[(usize, f64)], rows: usize| t.iter().find(|p| p.0 == rows).map(|p| p.1).unwrap();
        for rows in [7, 16, 32, 48] {
            let ratio = at(&tf, rows) / at(&full, rows);
            assert!(ratio > 0.38 && ratio < 0.45, "{rows} rows: {ratio}");
        }
        // a custom set scales the full table by its columns (plus a small floor)
        let half = head_table(&DraftVocab::Ranges(vec![(0, 124_160)]), 0.5);
        assert!((at(&half, 16) / at(&full, 16) - 0.52).abs() < 0.01);
        // the step (a few percent since the 4-pass decode kernel; 40-50% before it) is at the same row count in both
        assert!(at(&full, 33) / at(&full, 32) > 1.03 && at(&tf, 33) / at(&tf, 32) > 1.03);
    }

    #[test]
    fn the_head_tables_carry_the_step_at_the_65th_row_up_to_128_rows() {
        // the EXL3 decode kernel decodes each weight tile once per group of up to 4 sixteen-row passes: the 65th row of a head call starts a second
        // group (measured on anonymized host-a: the full head 4.50 -> 7.29 ms, TensorFold's set 1.93 -> 2.97 ms, 8.12 and 3.38 ms at 128 rows)
        let full = head_table(&DraftVocab::Full, 1.0);
        let tf = head_table(&DraftVocab::Tf, 0.397);
        let at = |t: &[(usize, f64)], rows: usize| {
            t.iter().find(|p| p.0 == rows).map(|p| p.1).unwrap_or_else(|| panic!("no point at {rows} rows"))
        };
        let (step_full, step_tf) = (at(&full, 65) - at(&full, 64), at(&tf, 65) - at(&tf, 64));
        assert!(step_full > 2.2 && step_full < 3.4, "full head: +{step_full} ms at the 65th row");
        assert!(step_tf > 0.8 && step_tf < 1.3, "candidate head: +{step_tf} ms at the 65th row");
        assert!(at(&full, 128) > 7.5 && at(&full, 128) < 8.8, "{} ms at 128 rows", at(&full, 128));
        assert!(at(&tf, 128) > 3.0 && at(&tf, 128) < 3.8, "{} ms at 128 rows", at(&tf, 128));
        // a column-restricted head still reads 40% of the columns in the second group (time scales with the columns, not the rows)
        for rows in [65, 128] {
            let ratio = at(&tf, rows) / at(&full, rows);
            assert!(ratio > 0.38 && ratio < 0.45, "{rows} rows: {ratio}");
        }
        // both are monotone to 128 rows
        for t in [&full, &tf] {
            assert!(t.windows(2).all(|w| w[1].0 > w[0].0 && w[1].1 >= w[0].1 - 0.05), "{t:?}");
            assert_eq!(t.last().unwrap().0, 128);
        }
    }

    #[test]
    fn the_measured_tables_make_a_valid_cost() {
        let c = round_cost(Some(head_table(&DraftVocab::Full, 1.0)), 8);
        // the verify step has a step past 32 rows (5.9 ms for 6 streams measured, 8.9 before the 4-pass decode kernel), the draft head
        // a smaller one (0.2 ms; 2.1 before)
        assert!(c.verify.ms(33, 6) - c.verify.ms(32, 6) > 4.0);
        let d = c.draft.as_ref().unwrap();
        assert!(d.ms(6, 33) - d.ms(6, 32) > 0.1);
        assert!(c.verify.ms(1, 1) > 60.0 && c.verify.ms(1, 1) < 70.0);
    }

    #[test]
    fn the_wide_grid_prices_the_second_tile_decode_group_past_64_rows_and_the_planner_uses_it_only_where_it_pays() {
        use gb_sched::{plan_round, Demand};
        let c = round_cost(Some(head_table(&DraftVocab::Full, 1.0)), 8);
        // measured to 32 streams and 128 rows, not extrapolated: the corner is in the table
        assert!(STEP_GRID.iter().any(|&(s, r, _)| (s, r) == (32, 128)) && STEP_GRID.iter().any(|&(s, r, _)| (s, r) == (16, 128)));
        // every EXL3 linear decodes each weight tile once per group of up to 4 sixteen-row passes (`group_tiles`): the 65th row starts a
        // second group, +40 ms at every stream count from 9 up
        for streams in [9usize, 16, 24, 32] {
            let step = c.verify.ms(65, streams) - c.verify.ms(64, streams);
            assert!(step > 30.0 && step < 50.0, "{streams} streams: +{step} ms at the 65th row");
            if streams >= 16 {
                // 16 streams are the fewest that can fill 128 rows (8 rows each); 9 to 15 are measured only to 8 x streams rows
                assert!(c.verify.ms(128, streams) - c.verify.ms(64, streams) > 60.0, "{streams} streams: 128 rows");
            }
        }
        let plan = |streams: usize, acc: f64, cap: usize| plan_round(&c, &vec![Demand::geometric(acc, 7); streams], cap, false);
        // a window of 128 rows changes no plan of 8 streams or fewer, and none at any stream count when drafts land half the time or
        // less: the gate's 6-stream fixtures cannot move under a wider window
        for acc in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9] {
            for streams in [1usize, 2, 4, 6, 8] {
                assert_eq!(plan(streams, acc, 64), plan(streams, acc, 128), "{streams} streams, acceptance {acc}");
            }
        }
        for streams in [4usize, 8, 16, 24, 32] {
            assert_eq!(plan(streams, 0.5, 64), plan(streams, 0.5, 128), "{streams} streams, acceptance 0.5");
        }
        // where drafts land (0.8 is what the sweeps measure on distinct prompts, 0.9 on code) and the streams are many, the second group pays: the
        // budget fills it, and the round earns 17% (0.8) to 29% (0.9) more tokens a millisecond at 32 streams than the best 64-row round
        for (acc, least) in [(0.8, 0.10), (0.9, 0.2)] {
            let (narrow, wide) = (plan(32, acc, 64), plan(32, acc, 128));
            assert!(narrow.rows <= 64 && wide.rows > 64, "acceptance {acc}: {} and {} rows", narrow.rows, wide.rows);
            let gain = (wide.tokens / wide.ms) / (narrow.tokens / narrow.ms) - 1.0;
            assert!(gain > least, "acceptance {acc}: the wide round earns {:.1}% more", 100.0 * gain);
        }
    }

    #[test]
    fn the_draft_cost_matches_the_measured_rounds_from_1_to_32_sequences() {
        // `draft_time_by_streams` on anonymized host-a (quiet box), the full-vocabulary draft asked for 4 picks a sequence: ms of the round by sequences. Up to 16
        // is one round; 17 and more are two (the model prices the second as its own round); the table values for 1 to 6 are older holds
        let measured = [
            (1usize, 8.98),
            (2, 9.38),
            (3, 9.99),
            (4, 10.50),
            (5, 11.25),
            (6, 12.05),
            (8, 12.88),
            (12, 17.27),
            (16, 19.88),
            (17, 29.34),
            (20, 30.96),
            (24, 33.20),
            (28, 37.03),
            (32, 39.64),
        ];
        let d = round_cost(Some(head_table(&DraftVocab::Full, 1.0)), 8).draft.unwrap();
        for (n, ms) in measured {
            let model = d.ms(n, 4 * n);
            assert!((model / ms - 1.0).abs() < 0.06, "{n} sequences: the model says {model:.2} ms, measured {ms}");
        }
        // the second round is what the 17th costs, not a sequence's slope: measured +9.5 ms
        let jump = d.ms(17, 68) - d.ms(16, 64);
        assert!(jump > 8.0 && jump < 11.0, "+{jump} ms at the 17th sequence");
    }

    #[test]
    fn the_served_draft_cost_prices_the_second_draft_round_from_the_17th_stream() {
        // the draft's round carries MAX_DRAFT_SEQS sequences, so stream 17 starts another one: a whole round's layers and head, not
        // the +0.5 ms a stream of the single-round table
        let d = round_cost(Some(head_table(&DraftVocab::Full, 1.0)), 8).draft.unwrap();
        let w = crate::draft::MAX_DRAFT_SEQS;
        let step = |n: usize| d.ms(n, n * 4) - d.ms(n - 1, (n - 1) * 4);
        assert!(step(w) < 1.5, "inside a round a stream costs its slope: {} ms", step(w));
        assert!(step(w + 1) > 8.0, "the stream that opens a second round pays for it: {} ms", step(w + 1));
        assert!(step(w + 2) < 1.5 && step(2 * w) < 1.5 && step(2 * w + 1) > 8.0, "and again at the third round");
        // below one round the table is untouched
        let single = DraftCost::new(DRAFT_LAYERS.to_vec(), head_table(&DraftVocab::Full, 1.0), 0.0);
        for n in 1..=w {
            assert_eq!(d.ms(n, n * 4).to_bits(), single.ms(n, n * 4).to_bits(), "{n} streams");
        }
    }
}
