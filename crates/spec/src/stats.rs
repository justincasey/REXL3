//! Speculation telemetry: what each draft position bought, and where a decode round's milliseconds went.
//!
//! Every later budget decision (how many drafts to verify, whether the draft is worth running at all) is read off these counters,
//! so they are kept per draft position, not as one acceptance rate. For draft position `j` (1 is the draft right after the token
//! being fed):
//!
//! * `offered`:  the drafter produced a draft at `j` (its work for `j` was paid);
//! * `verified`: the verify window carried it (a row of the target forward was spent on it);
//! * `reached`:  drafts `1..j` were all accepted, so `j` was actually tested against the target's sample;
//! * `accepted`: the draft at `j` equalled the target's sample (then drafts `1..=j` all did).
//!
//! `accepted / reached` is the conditional acceptance at `j` (the number a budget needs), `offered - reached` is draft work that
//! could never have counted, and `verified - reached` is verify rows that were computed and thrown away.
//!
//! Counters are split by sampling mode (greedy, sampled) and by whether the round ran one stream or several, because the budget
//! only truncates drafts when rows are scarce, so a one-stream and a six-stream round are different populations.

use gb_sampling::Sampling;

/// Draft positions tracked. DFlash2 drafts 7; the room is for deeper chains.
pub const MAX_POSITIONS: usize = 16;
/// Streams per round tracked by the phase timers (rounds with more land in the last slot).
pub const MAX_STREAMS: usize = 32;
/// Bins of the draft-confidence histogram.
pub const CONF_BINS: usize = 32;
/// The draft's winning selector score at or below this lands in bin 0.
pub const CONF_LO: f32 = -8.0;
pub const CONF_WIDTH: f32 = 2.0;

/// The part of a request that decides which population its rounds belong to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Greedy = 0,
    Sampled = 1,
}

impl Mode {
    pub const ALL: [Mode; 2] = [Mode::Greedy, Mode::Sampled];
    pub fn of(s: &Sampling) -> Mode {
        if s.temperature == 0.0 {
            Mode::Greedy
        } else {
            Mode::Sampled
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Mode::Greedy => "greedy",
            Mode::Sampled => "sampled",
        }
    }
}

/// Per-position counters of one population (see the module comment for the four columns).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PositionStats {
    /// Sequence-rounds recorded (including those that were offered nothing).
    pub rounds: u64,
    pub offered: [u64; MAX_POSITIONS],
    pub verified: [u64; MAX_POSITIONS],
    pub reached: [u64; MAX_POSITIONS],
    pub accepted: [u64; MAX_POSITIONS],
}

impl PositionStats {
    /// One sequence-round: the drafter `offered` drafts, `verified` of them rode in the window, and the first `accepted` of those matched.
    pub fn record(&mut self, offered: usize, verified: usize, accepted: usize) {
        debug_assert!(accepted <= verified && verified <= offered, "accepted {accepted} verified {verified} offered {offered}");
        self.rounds += 1;
        // draft j was tested exactly when drafts 1..j were accepted and j itself rode in the window
        let reached = verified.min(accepted + 1);
        for j in 0..offered.min(MAX_POSITIONS) {
            self.offered[j] += 1;
            if j < verified {
                self.verified[j] += 1;
            }
            if j < reached {
                self.reached[j] += 1;
            }
            if j < accepted {
                self.accepted[j] += 1;
            }
        }
    }

    pub fn merge(&mut self, o: &PositionStats) {
        self.rounds += o.rounds;
        for j in 0..MAX_POSITIONS {
            self.offered[j] += o.offered[j];
            self.verified[j] += o.verified[j];
            self.reached[j] += o.reached[j];
            self.accepted[j] += o.accepted[j];
        }
    }

    /// Conditional acceptance at draft position `j` (1-based): P(draft j accepted | drafts 1..j accepted). None before any trial.
    pub fn conditional(&self, j: usize) -> Option<f64> {
        let i = j.checked_sub(1)?;
        (i < MAX_POSITIONS && self.reached[i] > 0).then(|| self.accepted[i] as f64 / self.reached[i] as f64)
    }

    /// Tokens committed per round, counting the sample past the accepted drafts: 1 + sum of the accepted counts / rounds.
    pub fn tokens_per_round(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            1.0 + self.accepted.iter().sum::<u64>() as f64 / self.rounds as f64
        }
    }

    pub fn is_empty(&self) -> bool {
        self.rounds == 0
    }
}

/// One stream's conditional acceptance by draft position: decayed counts of how often position `j` was tested and how often it
/// passed. `AcceptStats` keeps one number and so assumes every position is as likely as the first; DFlash2's later positions are
/// not (a draft is conditioned on its own earlier picks, and the chain is only as good as its weakest link), so a budget that
/// ranks drafts by their marginal value needs the whole curve.
#[derive(Clone, Debug, PartialEq)]
pub struct AcceptCurve {
    reached: [f64; MAX_POSITIONS],
    accepted: [f64; MAX_POSITIONS],
    decay: f64,
}

impl AcceptCurve {
    /// `decay` is applied to every count on each update (0.9 forgets a round's evidence after about ten rounds).
    pub fn new(decay: f64) -> Self {
        Self { reached: [0.0; MAX_POSITIONS], accepted: [0.0; MAX_POSITIONS], decay }
    }

    /// A round carried `verified` drafts and the first `accepted` matched: positions `1..=min(verified, accepted + 1)` were tested.
    pub fn update(&mut self, verified: usize, accepted: usize) {
        if verified == 0 {
            return;
        }
        debug_assert!(accepted <= verified);
        let tested = verified.min(accepted + 1);
        for j in 0..MAX_POSITIONS {
            self.reached[j] *= self.decay;
            self.accepted[j] *= self.decay;
            if j < tested {
                self.reached[j] += 1.0;
            }
            if j < accepted {
                self.accepted[j] += 1.0;
            }
        }
    }

    /// A round in which this stream was given no drafts: nothing was tested, so its evidence fades one step toward the prior (the
    /// counts decay as in an update that tests nothing). Without this a stream the budget has stopped drafting would keep the
    /// estimate that stopped it for good, because only a round that verifies drafts updates the curve (`update` ignores
    /// `verified == 0`); the same reason `AcceptStats::relax` exists for the legacy budget.
    pub fn relax(&mut self) {
        for j in 0..MAX_POSITIONS {
            self.reached[j] *= self.decay;
            self.accepted[j] *= self.decay;
        }
    }

    /// Posterior mean of P(draft `j` accepted | drafts `1..j` accepted), `j` counted from 1: this stream's evidence plus `weight`
    /// pseudo-trials at `prior`. With no evidence it is the prior.
    pub fn conditional(&self, j: usize, prior: f64, weight: f64) -> f64 {
        let i = j.clamp(1, MAX_POSITIONS) - 1;
        ((self.accepted[i] + weight * prior) / (self.reached[i] + weight)).clamp(0.0, 1.0)
    }

    /// Decayed number of tests at position `j`.
    pub fn trials(&self, j: usize) -> f64 {
        self.reached[j.clamp(1, MAX_POSITIONS) - 1]
    }

    /// The conditional acceptance of every position, blending this stream's evidence with `prior` (`weight` pseudo-trials each).
    ///
    /// A position this stream has barely been tested at (a budget that never verified that deep gives it no evidence) would
    /// otherwise keep the class prior for ever, and a stream that is doing better than its class would never be offered the
    /// depth it could use. So where evidence is thin, the prior is scaled by how much better or worse this stream is than the
    /// prior at the position before (limited to a factor of 2 either way, and never above 1): a stream that accepts everything
    /// the class's first positions do not is assumed to keep doing so deeper, which is what earns it the rows to find out.
    pub fn curve(&self, prior: &[f64; MAX_POSITIONS], weight: f64) -> [f64; MAX_POSITIONS] {
        let mut out = [0.0; MAX_POSITIONS];
        for j in 0..MAX_POSITIONS {
            let mut p = prior[j];
            if j > 0 && self.reached[j] < 1.0 && prior[j - 1] > 1e-9 {
                p = (p * (out[j - 1] / prior[j - 1]).clamp(0.5, 2.0)).min(1.0);
            }
            out[j] = ((self.accepted[j] + weight * p) / (self.reached[j] + weight)).clamp(0.0, 1.0);
        }
        out
    }
}

/// The class-level prior of the conditional acceptance by position, from the engine's own counters: `PositionStats::conditional`
/// where a position has been tested at least `min_trials` times, and below that the previous position's value times `fade` (a
/// chain rarely gets better with depth), starting from `first`.
pub fn conditional_prior(stats: &PositionStats, min_trials: u64, first: f64, fade: f64) -> [f64; MAX_POSITIONS] {
    let mut out = [0.0; MAX_POSITIONS];
    let mut prev = first / fade.max(1e-9);
    for (i, o) in out.iter_mut().enumerate() {
        *o = if stats.reached[i] >= min_trials { stats.accepted[i] as f64 / stats.reached[i] as f64 } else { prev * fade };
        prev = *o;
    }
    out
}

/// Reached and accepted counts per (position, confidence bin): whether a confidence the draft reports says anything about
/// acceptance. The bins are `lo + k * width` (the first and last take everything below and above), so one type serves scores
/// and margins.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfStats {
    pub lo: f32,
    pub width: f32,
    pub reached: Vec<[u64; CONF_BINS]>,
    pub accepted: Vec<[u64; CONF_BINS]>,
}

impl Default for ConfStats {
    /// Bins for the draft's winning selector score.
    fn default() -> Self {
        Self::with_bins(CONF_LO, CONF_WIDTH)
    }
}

/// The bin a selector score falls in.
pub fn conf_bin(score: f32) -> usize {
    bin_of(score, CONF_LO, CONF_WIDTH)
}

fn bin_of(x: f32, lo: f32, width: f32) -> usize {
    if x.is_nan() {
        return 0;
    }
    (((x - lo) / width).floor().max(0.0) as usize).min(CONF_BINS - 1)
}

impl ConfStats {
    pub fn with_bins(lo: f32, width: f32) -> Self {
        Self { lo, width, reached: vec![[0; CONF_BINS]; MAX_POSITIONS], accepted: vec![[0; CONF_BINS]; MAX_POSITIONS] }
    }
    /// Bins for a margin (a lead over the runner-up, 0 to 30): half a unit wide up to 16.
    pub fn margins() -> Self {
        Self::with_bins(0.0, 0.5)
    }
    pub fn bin(&self, x: f32) -> usize {
        bin_of(x, self.lo, self.width)
    }
    /// One sequence-round with a confidence per verified draft: draft `j` was reached iff `j <= accepted` (0-based `j < reached`).
    pub fn record(&mut self, conf: &[f32], verified: usize, accepted: usize) {
        let reached = verified.min(accepted + 1).min(conf.len());
        for (j, &x) in conf.iter().enumerate().take(reached.min(MAX_POSITIONS)) {
            let b = self.bin(x);
            self.reached[j][b] += 1;
            if j < accepted {
                self.accepted[j][b] += 1;
            }
        }
    }
    pub fn merge(&mut self, o: &ConfStats) {
        for j in 0..MAX_POSITIONS {
            for b in 0..CONF_BINS {
                self.reached[j][b] += o.reached[j][b];
                self.accepted[j][b] += o.accepted[j][b];
            }
        }
    }
    pub fn is_empty(&self) -> bool {
        self.reached.iter().all(|r| r.iter().all(|&c| c == 0))
    }
}

/// How much better or worse than a model's predictions one stream has turned out to be: decayed counts of the drafts that landed
/// against the sum of the probabilities the model gave them. A calibrated model gives a ratio of 1; a stream whose target
/// disagrees with the draft more than the model thinks (the draft is good at predicting its own score, not this text) gives a
/// ratio below 1, and a prediction multiplied by it stops spending rows on that stream.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    hits: f64,
    expect: f64,
    decay: f64,
}

impl Calibration {
    pub fn new(decay: f64) -> Self {
        Self { hits: 0.0, expect: 0.0, decay }
    }

    /// A round verified `verified` drafts with the model's probability `pred[j]` for draft `j` (given the drafts before it landed)
    /// and the first `accepted` matched: drafts up to the first miss were tested.
    pub fn update(&mut self, pred: &[f64], verified: usize, accepted: usize) {
        let tested = verified.min(accepted + 1).min(pred.len());
        if tested == 0 {
            return;
        }
        self.hits = self.hits * self.decay + accepted.min(tested) as f64;
        self.expect = self.expect * self.decay + pred[..tested].iter().sum::<f64>();
    }

    /// A tree round: the model expected `expected` accepted drafts from the tree's verified nodes (the sum of their probabilities) and
    /// `accepted` were (the accepted path's length past the root). A tree has no chain of tested positions, so nothing is truncated.
    pub fn update_tree(&mut self, expected: f64, accepted: usize) {
        self.hits = self.hits * self.decay + accepted as f64;
        self.expect = self.expect * self.decay + expected;
    }

    /// A round in which this stream verified nothing: the history fades one step toward "agrees with the model" (a factor of 1),
    /// for the reason [`AcceptCurve::relax`] gives.
    pub fn relax(&mut self) {
        self.hits *= self.decay;
        self.expect *= self.decay;
    }

    /// Observed over expected, with `prior_trials` expected-hits of agreement behind it (so a stream with no history is 1).
    pub fn factor(&self, prior_trials: f64) -> f64 {
        (self.hits + prior_trials) / (self.expect + prior_trials)
    }
}

/// How likely a draft is to be accepted given its margin (its lead over the runner-up in what the pick maximised), per margin bin,
/// pooled over draft positions: the rate the engine has observed in that bin, with `weight` pseudo-trials at a logistic prior fitted
/// to the first measurement on the GB10 (accepted 37% at a margin of 0, 71% at 2, 93% at 5, 99% at 8; AUC 0.87), so a table with
/// no data yet is already sensible and one with plenty is the engine's own.
#[derive(Clone, Debug, PartialEq)]
pub struct MarginTable {
    lo: f32,
    width: f32,
    p: [f64; CONF_BINS],
}

/// The prior's acceptance probability at a margin.
pub fn margin_prior(margin: f64) -> f64 {
    1.0 / (1.0 + (-(0.62 * margin - 0.3)).exp())
}

impl MarginTable {
    /// The table of the counts frozen in [`crate::measured`] (the GB10 session they were measured in), with the same `weight`
    /// pseudo-trials of the prior behind each bin. It does not change with the traffic the engine sees.
    pub fn measured(weight: f64) -> Self {
        let mut c = ConfStats::margins();
        c.reached[0] = crate::measured::MARGIN_REACHED;
        c.accepted[0] = crate::measured::MARGIN_ACCEPTED;
        Self::from_stats(&c, weight)
    }

    pub fn from_stats(c: &ConfStats, weight: f64) -> Self {
        // per bin: the observed counts plus `weight` pseudo-trials at the prior, as (successes, trials, bins pooled)
        let mut blocks: Vec<(f64, f64, usize)> = Vec::with_capacity(CONF_BINS);
        for b in 0..CONF_BINS {
            let centre = (c.lo + (b as f32 + 0.5) * c.width) as f64;
            let (reached, accepted) =
                (0..MAX_POSITIONS).fold((0u64, 0u64), |a, j| (a.0 + c.reached[j][b], a.1 + c.accepted[j][b]));
            blocks.push((accepted as f64 + weight * margin_prior(centre), reached as f64 + weight, 1));
            // a lead can only make a draft likelier: where a bin's rate falls below the bin before it, pool the two
            // (pool-adjacent-violators, the isotonic fit), so noise between neighbouring bins cannot invert the table
            while blocks.len() > 1 {
                let n = blocks.len();
                let (prev, last) = (blocks[n - 2], blocks[n - 1]);
                if prev.0 / prev.1 <= last.0 / last.1 {
                    break;
                }
                blocks.truncate(n - 2);
                blocks.push((prev.0 + last.0, prev.1 + last.1, prev.2 + last.2));
            }
        }
        let mut p = [0.0; CONF_BINS];
        let mut at = 0;
        for (s, t, bins) in blocks {
            p[at..at + bins].fill(s / t);
            at += bins;
        }
        Self { lo: c.lo, width: c.width, p }
    }

    /// P(accepted | this margin, every earlier draft accepted).
    pub fn p(&self, margin: f32) -> f64 {
        self.p[bin_of(margin, self.lo, self.width)]
    }
}

/// Stream classes of the by-rows round table of [`PhaseStats`]: 1, 2, 3 and 4 or more streams.
pub const ROW_CLASSES: usize = 4;
/// Row counts of that table (index = rows of the verify step, the last slot also takes wider steps).
pub const ROW_BUCKETS: usize = 33;

/// Rounds of one (stream class, verify rows) cell, with the wall time their draft and verify phases took.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RowCell {
    pub rounds: u64,
    pub draft_ns: u64,
    pub verify_ns: u64,
}

/// Wall time of a decode round's phases, by how many streams the round ran. Host clocks around the backend calls: the draft and
/// the verify step each end in a device sync, so these are the milliseconds the engine thread waited for each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseStats {
    /// Index `n - 1` for a round of `n` streams (the last slot also takes wider rounds).
    pub rounds: [u64; MAX_STREAMS],
    /// Rows of the verify step (all streams, base rows and drafts).
    pub rows: [u64; MAX_STREAMS],
    /// Streams that ran the draft (those asked for drafts) and the draft positions requested of them.
    pub draft_streams: [u64; MAX_STREAMS],
    pub draft_ns: [u64; MAX_STREAMS],
    pub verify_ns: [u64; MAX_STREAMS],
    pub commit_ns: [u64; MAX_STREAMS],
    /// Tree rounds only: the host's time between the draft and the verify step (the best-first search, the copy continuation and the row
    /// allocation), which none of the phases above holds.
    pub plan_ns: [u64; MAX_STREAMS],
    /// The same rounds by (stream class, rows of the verify step): what a window of each width cost as served, the table the engine's
    /// cost model is read from (`[streams.min(4) - 1][rows.min(32)]`).
    pub by_rows: [[RowCell; ROW_BUCKETS]; ROW_CLASSES],
}

impl Default for PhaseStats {
    fn default() -> Self {
        Self {
            rounds: [0; MAX_STREAMS],
            rows: [0; MAX_STREAMS],
            draft_streams: [0; MAX_STREAMS],
            draft_ns: [0; MAX_STREAMS],
            verify_ns: [0; MAX_STREAMS],
            commit_ns: [0; MAX_STREAMS],
            plan_ns: [0; MAX_STREAMS],
            by_rows: [[RowCell::default(); ROW_BUCKETS]; ROW_CLASSES],
        }
    }
}

impl PhaseStats {
    /// A tree round's host planning time (see [`PhaseStats::plan_ns`]); call with the same `streams` as `record`.
    pub fn record_plan(&mut self, streams: usize, plan_ns: u64) {
        self.plan_ns[streams.clamp(1, MAX_STREAMS) - 1] += plan_ns;
    }
    #[allow(clippy::too_many_arguments)]
    pub fn record(&mut self, streams: usize, rows: usize, drafting: usize, draft_ns: u64, verify_ns: u64, commit_ns: u64) {
        let i = streams.clamp(1, MAX_STREAMS) - 1;
        self.rounds[i] += 1;
        self.rows[i] += rows as u64;
        self.draft_streams[i] += drafting as u64;
        self.draft_ns[i] += draft_ns;
        self.verify_ns[i] += verify_ns;
        self.commit_ns[i] += commit_ns;
        let cell = &mut self.by_rows[streams.clamp(1, ROW_CLASSES) - 1][rows.min(ROW_BUCKETS - 1)];
        cell.rounds += 1;
        cell.draft_ns += draft_ns;
        cell.verify_ns += verify_ns;
    }
    /// Mean milliseconds per round of `n` streams for each phase: (draft, verify, commit); None when no such round ran.
    pub fn mean_ms(&self, n: usize) -> Option<(f64, f64, f64)> {
        let i = n.clamp(1, MAX_STREAMS) - 1;
        let r = self.rounds[i];
        (r > 0).then(|| {
            let ms = |ns: u64| ns as f64 / r as f64 / 1e6;
            (ms(self.draft_ns[i]), ms(self.verify_ns[i]), ms(self.commit_ns[i]))
        })
    }
}

/// Sibling ranks tracked by the tree counters (the policy expands at most this many children under a node).
pub const MAX_BRANCH: usize = 8;

/// What the verify windows that carried TREES bought (one population per sampling mode), by depth and by sibling rank. For depth `d`
/// (1 is a child of the anchor): `verified` is nodes verified at `d`, `tested` the rounds in which the walk reached depth `d - 1` at a node
/// that had at least one child in the window, `accepted` the rounds in which the accepted path reached depth `d`; `accepted / tested` is the
/// conditional acceptance at `d` that a chain's counters give, with the siblings counted as one chance. For sibling rank `r` (0 is the child
/// the search ranked first): `offered` counts the nodes on the accepted path that had a child of rank `r` in the window, `accepted` those
/// whose accepted child had rank `r`: what a second, third and fourth sibling bought. `predicted` is the sum of the calibrated
/// probabilities of the verified nodes: against `accepted_total` it says whether the node probabilities the allocation spends rows by
/// are right.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TreeStats {
    /// Sequence-rounds that verified a tree window.
    pub rounds: u64,
    /// Nodes the search ranked (the tree before the allocation cut it).
    pub offered: u64,
    /// Nodes verified (rows past the root).
    pub verified: u64,
    /// Accepted drafts (accepted path length past the root), summed over rounds.
    pub accepted_total: u64,
    /// Sum of the calibrated probabilities of the verified nodes.
    pub predicted: f64,
    pub depth_verified: [u64; MAX_POSITIONS],
    pub depth_tested: [u64; MAX_POSITIONS],
    pub depth_accepted: [u64; MAX_POSITIONS],
    pub rank_offered: [u64; MAX_BRANCH],
    pub rank_accepted: [u64; MAX_BRANCH],
    /// Rounds whose window carried a copy branch (tokens copied from the stream's own text, hung off the draft's picks).
    pub copy_rounds: u64,
    /// Copied tokens verified, and the accepted ones (copied nodes on the accepted path).
    pub copy_verified: u64,
    pub copy_accepted: u64,
    /// Sum of the calibrated probabilities of the verified copied tokens: against `copy_accepted` it says whether the copy table (`measured_copy`)
    /// the allocation spends rows on copied tokens by is right on the traffic being served.
    pub copy_predicted: f64,
}

impl TreeStats {
    /// The copy branch of one sequence-round: `copied[i]` says whether the window's row `i + 1` is a copied token, `probs[i]` that node's
    /// probability of lying on the accepted path, `path` the accepted rows.
    pub fn record_copy(&mut self, copied: &[bool], probs: &[f64], path: &[usize]) {
        let n = copied.iter().filter(|&&c| c).count() as u64;
        if n == 0 {
            return;
        }
        self.copy_rounds += 1;
        self.copy_verified += n;
        self.copy_predicted += copied.iter().zip(probs).filter(|(&c, _)| c).map(|(_, &p)| p).sum::<f64>();
        self.copy_accepted += path.iter().filter(|&&r| r >= 1 && copied.get(r - 1).copied().unwrap_or(false)).count() as u64;
    }

    /// One sequence-round. `parents[r]` is the parent row of window row `r` (row 0 is the anchor, parent -1; parents come first), `path` the
    /// accepted rows root first, `offered` the nodes the search ranked and `predicted` the sum of the verified nodes' probabilities.
    pub fn record(&mut self, parents: &[i32], path: &[usize], offered: usize, predicted: f64) {
        self.rounds += 1;
        self.offered += offered as u64;
        self.verified += parents.len().saturating_sub(1) as u64;
        self.accepted_total += path.len().saturating_sub(1) as u64;
        self.predicted += predicted;
        let mut depth = vec![0usize; parents.len()];
        for r in 1..parents.len() {
            depth[r] = depth[parents[r].max(0) as usize] + 1;
            self.depth_verified[(depth[r] - 1).min(MAX_POSITIONS - 1)] += 1;
        }
        for (i, &node) in path.iter().enumerate() {
            let children: Vec<usize> = (0..parents.len()).filter(|&r| parents[r] == node as i32).collect();
            if children.is_empty() {
                continue;
            }
            let d = (depth[node]).min(MAX_POSITIONS - 1);
            self.depth_tested[d] += 1;
            for r in 0..children.len().min(MAX_BRANCH) {
                self.rank_offered[r] += 1;
            }
            if let Some(&next) = path.get(i + 1) {
                self.depth_accepted[d] += 1;
                if let Some(rank) = children.iter().position(|&c| c == next) {
                    self.rank_accepted[rank.min(MAX_BRANCH - 1)] += 1;
                }
            }
        }
    }

    pub fn merge(&mut self, o: &TreeStats) {
        self.rounds += o.rounds;
        self.offered += o.offered;
        self.verified += o.verified;
        self.accepted_total += o.accepted_total;
        self.predicted += o.predicted;
        for j in 0..MAX_POSITIONS {
            self.depth_verified[j] += o.depth_verified[j];
            self.depth_tested[j] += o.depth_tested[j];
            self.depth_accepted[j] += o.depth_accepted[j];
        }
        for r in 0..MAX_BRANCH {
            self.rank_offered[r] += o.rank_offered[r];
            self.rank_accepted[r] += o.rank_accepted[r];
        }
        self.copy_rounds += o.copy_rounds;
        self.copy_verified += o.copy_verified;
        self.copy_accepted += o.copy_accepted;
        self.copy_predicted += o.copy_predicted;
    }

    /// Tokens committed per round, counting the sample past the accepted path: 1 + accepted drafts per round.
    pub fn tokens_per_round(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            1.0 + self.accepted_total as f64 / self.rounds as f64
        }
    }

    /// Verified nodes per round: the rows the trees cost past the roots.
    pub fn nodes_per_round(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            self.verified as f64 / self.rounds as f64
        }
    }
}

/// Everything the engine counts about speculation.
#[derive(Clone, Debug, PartialEq)]
pub struct SpecStats {
    /// `[mode][multi]`: `multi` is 1 for a round that ran more than one stream.
    pub by: [[PositionStats; 2]; 2],
    /// Histograms of the draft's winning selector score against acceptance, by mode.
    pub conf: [ConfStats; 2],
    /// Histograms of the draft's margin (lead over the runner-up) against acceptance, by mode.
    pub margin: [ConfStats; 2],
    pub phases: PhaseStats,
    /// Rounds that verified trees (`Engine::with_trees`), by mode.
    pub trees: [TreeStats; 2],
}

impl Default for SpecStats {
    fn default() -> Self {
        Self {
            by: Default::default(),
            conf: Default::default(),
            margin: [ConfStats::margins(), ConfStats::margins()],
            phases: PhaseStats::default(),
            trees: Default::default(),
        }
    }
}

impl SpecStats {
    /// Record one sequence-round. `conf` and `margin` hold a number per verified draft (empty when the drafter has none).
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        mode: Mode,
        streams: usize,
        offered: usize,
        conf: &[f32],
        margin: &[f32],
        verified: usize,
        accepted: usize,
    ) {
        self.by[mode as usize][(streams > 1) as usize].record(offered, verified, accepted);
        if !conf.is_empty() {
            self.conf[mode as usize].record(conf, verified, accepted);
        }
        if !margin.is_empty() {
            self.margin[mode as usize].record(margin, verified, accepted);
        }
    }

    /// All rounds of one mode, one-stream and many-stream together.
    pub fn mode_total(&self, mode: Mode) -> PositionStats {
        let mut t = self.by[mode as usize][0].clone();
        t.merge(&self.by[mode as usize][1]);
        t
    }

    /// Every round.
    pub fn total(&self) -> PositionStats {
        let mut t = self.mode_total(Mode::Greedy);
        t.merge(&self.mode_total(Mode::Sampled));
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_round_counts_each_column_by_its_definition() {
        let mut s = PositionStats::default();
        // 7 offered, 5 rode in the window, the first 2 matched: draft 3 was tested and failed, drafts 4-5 were verified for nothing
        s.record(7, 5, 2);
        assert_eq!(s.offered[..8], [1, 1, 1, 1, 1, 1, 1, 0]);
        assert_eq!(s.verified[..8], [1, 1, 1, 1, 1, 0, 0, 0]);
        assert_eq!(s.reached[..8], [1, 1, 1, 0, 0, 0, 0, 0]);
        assert_eq!(s.accepted[..8], [1, 1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(s.conditional(1), Some(1.0));
        assert_eq!(s.conditional(3), Some(0.0));
        assert_eq!(s.conditional(4), None);
        assert_eq!(s.tokens_per_round(), 3.0);
        // everything accepted: every verified draft was tested, nothing is wasted
        let mut f = PositionStats::default();
        f.record(7, 7, 7);
        assert_eq!(f.reached, f.verified);
        assert_eq!(f.accepted, f.verified);
        // nothing verified (a budget of zero): offered work with no trial
        let mut z = PositionStats::default();
        z.record(7, 0, 0);
        assert_eq!((z.offered[0], z.reached[0], z.rounds), (1, 0, 1));
    }

    #[test]
    fn conditional_rate_is_accepted_over_reached() {
        let mut s = PositionStats::default();
        for a in [7, 7, 1, 0] {
            s.record(7, 7, a);
        }
        // position 1: all four rounds tested it, three accepted it
        assert_eq!(s.conditional(1), Some(0.75));
        // position 2: three rounds reached it (the a=0 round did not), two accepted
        assert_eq!(s.reached[1], 3);
        assert_eq!(s.conditional(2), Some(2.0 / 3.0));
        assert!((s.tokens_per_round() - (1.0 + (7 + 7 + 1) as f64 / 4.0)).abs() < 1e-12);
    }

    #[test]
    fn positions_past_the_table_are_ignored_not_a_panic() {
        let mut s = PositionStats::default();
        s.record(40, 40, 40);
        assert_eq!(s.accepted[MAX_POSITIONS - 1], 1);
        assert_eq!(s.rounds, 1);
    }

    #[test]
    fn merge_adds_every_column() {
        let mut a = PositionStats::default();
        let mut b = PositionStats::default();
        a.record(7, 7, 3);
        b.record(7, 4, 4);
        let mut m = a.clone();
        m.merge(&b);
        for j in 0..MAX_POSITIONS {
            assert_eq!(m.offered[j], a.offered[j] + b.offered[j]);
            assert_eq!(m.verified[j], a.verified[j] + b.verified[j]);
            assert_eq!(m.reached[j], a.reached[j] + b.reached[j]);
            assert_eq!(m.accepted[j], a.accepted[j] + b.accepted[j]);
        }
        assert_eq!(m.rounds, 2);
    }

    #[test]
    fn tree_stats_merge_adds_the_copy_counters_and_record_copy_counts_only_copied_rows() {
        let (parents, path) = ([-1, 0, 1, 2, 3], [0usize, 1, 2, 3]);
        let mut a = TreeStats::default();
        a.record(&parents, &path, 6, 2.0);
        // rows 1 and 2 are the picks, rows 3 and 4 copied tokens; the walk took rows 1 to 3
        a.record_copy(&[false, false, true, true], &[0.9, 0.8, 0.7, 0.6], &path);
        assert_eq!((a.copy_rounds, a.copy_verified, a.copy_accepted), (1, 2, 1));
        assert!((a.copy_predicted - 1.3).abs() < 1e-12, "only the copied rows' probabilities: {}", a.copy_predicted);
        let mut b = TreeStats::default();
        b.record(&parents, &[0, 1], 6, 1.0);
        b.record_copy(&[false, false, true, true], &[0.5, 0.5, 0.4, 0.3], &[0, 1]);
        assert_eq!((b.copy_rounds, b.copy_verified, b.copy_accepted), (1, 2, 0));
        let mut m = a.clone();
        m.merge(&b);
        assert_eq!((m.copy_rounds, m.copy_verified, m.copy_accepted), (2, 4, 1));
        assert!((m.copy_predicted - 2.0).abs() < 1e-12);
        assert_eq!((m.rounds, m.verified, m.accepted_total), (2, 8, 4));
    }

    #[test]
    fn spec_stats_split_by_mode_and_stream_count() {
        let mut s = SpecStats::default();
        s.record(Mode::of(&Sampling { temperature: 0.0, ..Default::default() }), 1, 7, &[], &[], 7, 7);
        s.record(Mode::of(&Sampling::default()), 6, 7, &[], &[], 3, 1);
        assert_eq!(s.by[Mode::Greedy as usize][0].rounds, 1);
        assert_eq!(s.by[Mode::Sampled as usize][1].rounds, 1);
        assert_eq!(s.by[Mode::Greedy as usize][1].rounds, 0);
        assert_eq!(s.total().rounds, 2);
        assert_eq!(s.mode_total(Mode::Sampled).accepted[0], 1);
    }

    #[test]
    fn conf_bins_clamp_and_record_only_tested_positions() {
        assert_eq!(conf_bin(f32::NAN), 0);
        assert_eq!(conf_bin(-100.0), 0);
        assert_eq!(conf_bin(CONF_LO), 0);
        assert_eq!(conf_bin(CONF_LO + CONF_WIDTH), 1);
        assert_eq!(conf_bin(1000.0), CONF_BINS - 1);
        let mut c = ConfStats::default();
        // 3 verified, 1 accepted: drafts 1 and 2 were tested (1 passed), draft 3 was not
        c.record(&[10.0, 20.0, 30.0], 3, 1);
        let b = |x: f32| conf_bin(x);
        assert_eq!(c.reached[0][b(10.0)], 1);
        assert_eq!(c.accepted[0][b(10.0)], 1);
        assert_eq!(c.reached[1][b(20.0)], 1);
        assert_eq!(c.accepted[1][b(20.0)], 0);
        assert_eq!(c.reached[2].iter().sum::<u64>(), 0);
    }

    #[test]
    fn the_curve_tracks_each_position_and_falls_back_to_its_prior() {
        let mut c = AcceptCurve::new(1.0);
        // no evidence: the prior, whatever the weight
        assert_eq!(c.conditional(3, 0.7, 4.0), 0.7);
        // 7 drafts verified, 4 accepted: positions 1-4 passed, position 5 failed, 6 and 7 were never tested
        c.update(7, 4);
        for j in 1..=4 {
            assert!(c.conditional(j, 0.5, 1.0) > 0.5, "position {j} passed once");
        }
        assert!(c.conditional(5, 0.5, 1.0) < 0.5, "position 5 failed once");
        assert_eq!(c.conditional(6, 0.5, 1.0), 0.5, "position 6 was never tested");
        assert_eq!(c.trials(5), 1.0);
        assert_eq!(c.trials(6), 0.0);
        // exactly: (1 accepted + 1 * 0.5) / (1 tested + 1)
        assert!((c.conditional(1, 0.5, 1.0) - 0.75).abs() < 1e-12);
        // a round with nothing verified says nothing
        let before = c.clone();
        c.update(0, 0);
        assert_eq!(c, before);
        // decay: old evidence fades toward the prior
        let mut d = AcceptCurve::new(0.5);
        d.update(1, 1);
        for _ in 0..30 {
            d.update(1, 0); // position 1 fails from now on
        }
        assert!(d.conditional(1, 0.9, 0.1) < 0.1);
    }

    #[test]
    fn a_stream_given_no_drafts_drifts_back_to_the_prior() {
        let prior = [0.7; MAX_POSITIONS];
        let mut c = AcceptCurve::new(0.9);
        for _ in 0..40 {
            c.update(7, 0); // position 1 fails every round
        }
        let low = c.curve(&prior, 4.0)[0];
        assert!(low < 0.25, "{low}");
        // `update(0, _)` alone changes nothing: this is what left a stream that was budgeted zero drafts at zero
        let before = c.clone();
        c.update(0, 0);
        assert_eq!(c, before);
        let mut prev = low;
        for _ in 0..60 {
            c.relax();
            let p = c.curve(&prior, 4.0)[0];
            assert!(p >= prev && p <= 0.7 + 1e-12, "monotone toward the prior from below: {prev} -> {p}");
            prev = p;
        }
        assert!(prev > 0.65, "after 60 idle rounds the curve is back near its prior: {prev}");
        // the same for the margin model's per-stream factor
        let mut k = Calibration::new(0.9);
        for _ in 0..40 {
            k.update(&[0.9], 1, 0);
        }
        let low = k.factor(3.0);
        assert!(low < 0.3, "{low}");
        for _ in 0..60 {
            k.relax();
        }
        assert!(k.factor(3.0) > 0.95, "{}", k.factor(3.0));
    }

    #[test]
    fn a_stream_better_than_its_class_is_assumed_better_at_depths_it_has_not_been_tested() {
        let prior = {
            let mut p = [0.0; MAX_POSITIONS];
            for (i, v) in p.iter_mut().enumerate() {
                *v = 0.6 - 0.02 * i as f64;
            }
            p
        };
        let mut c = AcceptCurve::new(1.0);
        // a budget that verifies 3 drafts: positions 1-3 are always accepted, nothing deeper is ever tested
        for _ in 0..20 {
            c.update(3, 3);
        }
        let curve = c.curve(&prior, 4.0);
        assert!(curve[0] > 0.9 && curve[2] > 0.9, "tested positions follow the evidence: {curve:?}");
        // untested: the class says 0.54 at position 4, but this stream is ~1.6x better than the class at position 3
        assert!(curve[3] > prior[3] * 1.3, "position 4 should be assumed better than the class's {}: {}", prior[3], curve[3]);
        assert!(curve[3] <= 1.0);
        // a stream worse than its class is assumed worse deeper too
        let mut w = AcceptCurve::new(1.0);
        for _ in 0..20 {
            w.update(3, 0);
        }
        let worse = w.curve(&prior, 4.0);
        assert!(worse[1] < prior[1] && worse[3] < prior[3], "{worse:?}");
        // no evidence at all: the prior itself
        assert_eq!(AcceptCurve::new(0.9).curve(&prior, 4.0), prior);
    }

    #[test]
    fn a_stream_that_beats_or_misses_the_model_gets_a_factor_above_or_below_one() {
        let mut c = Calibration::new(1.0);
        assert_eq!(c.factor(3.0), 1.0, "no history is a ratio of one");
        // the model gave 0.5 to each of three tested drafts and all three landed: a stream better than predicted
        c.update(&[0.5, 0.5, 0.5, 0.5], 4, 3);
        assert!(c.factor(3.0) > 1.0);
        // drafts past the first miss were not tested and do not count: 2 verified, the first missed, so only draft 1 was tested
        let mut d = Calibration::new(1.0);
        d.update(&[0.9, 0.9], 2, 0);
        assert!((d.factor(0.0) - 0.0).abs() < 1e-12, "tested once, expected 0.9, got none");
        for _ in 0..20 {
            d.update(&[0.43], 1, 0); // a dead stream: the model says 0.43 every round, nothing ever lands
        }
        assert!(d.factor(3.0) < 0.3, "{}", d.factor(3.0));
        // 3 verified, the second missed: drafts 1 and 2 were tested (one landed), draft 3 never was, so it is not a failure
        let mut f = Calibration::new(1.0);
        f.update(&[0.5, 0.5, 0.5], 3, 1);
        assert!((f.factor(0.0) - 1.0).abs() < 1e-12, "1 hit against 2 tested drafts at 0.5: {}", f.factor(0.0));
        // the decay forgets old rounds: after many good rounds a stream that was dead recovers
        let mut e = Calibration::new(0.8);
        for _ in 0..10 {
            e.update(&[0.5], 1, 0);
        }
        for _ in 0..40 {
            e.update(&[0.5], 1, 1);
        }
        assert!(e.factor(1.0) > 1.5);
    }

    #[test]
    fn the_margin_table_starts_at_the_prior_and_follows_the_data() {
        let empty = MarginTable::from_stats(&ConfStats::margins(), 20.0);
        // the prior: a coin flip near zero, nearly certain by 8, and never falling as the margin grows
        assert!(empty.p(0.1) > 0.35 && empty.p(0.1) < 0.5, "{}", empty.p(0.1));
        assert!(empty.p(2.2) > 0.65 && empty.p(2.2) < 0.78);
        assert!(empty.p(8.2) > 0.98);
        let mut last = 0.0;
        for m in 0..60 {
            let p = empty.p(m as f32 * 0.25);
            assert!(p >= last - 1e-12, "margin {m}");
            last = p;
        }
        assert_eq!(empty.p(40.0), empty.p(16.0), "the last bin takes everything above");
        // data moves a bin: 400 tests at a margin of 1.25, 40 accepted, against a prior near 0.6
        let mut c = ConfStats::margins();
        let b = c.bin(1.25);
        c.reached[2][b] = 300;
        c.accepted[2][b] = 30;
        c.reached[5][b] = 100;
        c.accepted[5][b] = 10;
        let t = MarginTable::from_stats(&c, 20.0);
        assert!(t.p(1.25) < 0.2, "the observed 10% rate outweighs the prior: {}", t.p(1.25));
        // and a bin above it is never below it (the table is monotone)
        assert!(t.p(3.0) >= t.p(1.25));
        // noise that inverts two bins (95% at a margin of 2.1, 40% at 3.1, 500 tests each) is pooled, not believed: a bigger lead is
        // never less likely to land
        let mut n = ConfStats::margins();
        let (lo, hi) = (n.bin(2.1), n.bin(3.1));
        n.reached[1][lo] = 500;
        n.accepted[1][lo] = 475;
        n.reached[1][hi] = 500;
        n.accepted[1][hi] = 200;
        let inv = MarginTable::from_stats(&n, 20.0);
        assert!(inv.p(2.1) <= inv.p(3.1) + 1e-12, "{} vs {}", inv.p(2.1), inv.p(3.1));
        assert!((inv.p(3.1) - 0.675).abs() < 0.05, "the two bins are pooled: {}", inv.p(3.1));
    }

    #[test]
    fn the_class_prior_uses_measured_positions_and_fades_beyond_them() {
        let mut s = PositionStats::default();
        for _ in 0..100 {
            s.record(7, 7, 2); // positions 1 and 2 always pass, 3 always fails
        }
        let p = conditional_prior(&s, 50, 0.5, 0.9);
        assert_eq!(p[0], 1.0);
        assert_eq!(p[1], 1.0);
        assert_eq!(p[2], 0.0);
        // position 4 was never tested: previous value times the fade
        assert_eq!(p[3], 0.0);
        let cold = conditional_prior(&PositionStats::default(), 50, 0.5, 0.9);
        assert!((cold[0] - 0.5).abs() < 1e-12);
        assert!((cold[1] - 0.45).abs() < 1e-12);
        assert!(cold.windows(2).all(|w| w[1] < w[0]));
    }

    #[test]
    fn the_measured_priors_are_the_gb10_session_they_were_generated_from() {
        use crate::measured::{CLASS_PRIOR, MARGIN_ACCEPTED, MARGIN_REACHED};
        // the margin table follows the frozen counts: monotone, low at a coin flip, near 1 at a big lead
        let t = MarginTable::measured(20.0);
        let rate = |b: usize| MARGIN_ACCEPTED[b] as f64 / MARGIN_REACHED[b] as f64;
        assert!(t.p(0.25) > 0.30 && t.p(0.25) < 0.42, "{}", t.p(0.25));
        assert!(t.p(2.25) > 0.65 && t.p(2.25) < 0.78, "{}", t.p(2.25));
        assert!(t.p(8.25) > 0.97, "{}", t.p(8.25));
        let mut last = 0.0;
        for b in 0..CONF_BINS {
            let p = t.p(b as f32 * 0.5 + 0.25);
            assert!(p >= last - 1e-12, "bin {b}: {p} after {last}");
            last = p;
        }
        // with hundreds of tests in every bin the prior behind it hardly moves the observed rate
        for b in [0, 4, 9, 16] {
            assert!((t.p(b as f32 * 0.5 + 0.25) - rate(b)).abs() < 0.05, "bin {b}");
        }
        // every bin was tested often enough to be a measurement, and nothing was accepted more often than it was tested
        for b in 0..CONF_BINS {
            assert!(MARGIN_REACHED[b] >= 200 && MARGIN_ACCEPTED[b] <= MARGIN_REACHED[b], "bin {b}");
        }
        // the class priors are probabilities and fade past the measured positions
        for row in CLASS_PRIOR {
            assert!(row.iter().all(|&p| p > 0.0 && p < 1.0));
            assert!(row[7] > row[8] && row[8] > row[9]);
        }
        // greedy text is easier to draft than sampled text at the first position
        assert!(CLASS_PRIOR[0][0] > CLASS_PRIOR[1][0]);
    }

    #[test]
    fn phase_means_are_per_round_of_that_width() {
        let mut p = PhaseStats::default();
        p.record(6, 48, 6, 16_000_000, 90_000_000, 7_000_000);
        p.record(6, 40, 6, 14_000_000, 80_000_000, 5_000_000);
        p.record(1, 8, 1, 11_000_000, 62_000_000, 1_000_000);
        let (d, v, c) = p.mean_ms(6).unwrap();
        assert!((d - 15.0).abs() < 1e-9 && (v - 85.0).abs() < 1e-9 && (c - 6.0).abs() < 1e-9);
        assert_eq!(p.mean_ms(3), None);
        assert_eq!(p.mean_ms(1).unwrap().0, 11.0);
        // wider than the table lands in the last slot
        p.record(100, 64, 16, 1, 1, 1);
        assert_eq!(p.rounds[MAX_STREAMS - 1], 1);
    }
}
