//! `/metrics` text for the engine's speculation counters (`gb_spec::stats`): per-position acceptance by sampling mode and stream
//! count, the draft-confidence histogram, and the milliseconds of each phase of a decode round by stream count.
//!
//! A client that wants the histogram of one workload reads `/metrics` before and after it and subtracts: the counters only grow.
//! Series that are zero are left out, so the output stays small until traffic has run.

use gb_spec::stats::{Mode, PositionStats, SpecStats, TreeStats, MAX_POSITIONS};
use std::fmt::Write;

/// One counter family: `# HELP`, `# TYPE counter`, and a sample per `(labels, value)`.
fn emit(out: &mut String, name: &str, help: &str, samples: Vec<(String, String)>) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} counter");
    for (labels, v) in samples {
        let _ = writeln!(out, "{name}{{{labels}}} {v}");
    }
}

/// The `streams` label of a population: a round of one stream, or of several.
pub fn streams_label(multi: usize) -> &'static str {
    if multi == 0 {
        "1"
    } else {
        "2+"
    }
}

type Column = fn(&PositionStats) -> &[u64; MAX_POSITIONS];
/// The `(labels, value)` samples of one counter family, and a per-depth or per-rank column of the tree counters.
type Samples = Vec<(String, String)>;
type TreeColumn = fn(&TreeStats) -> &[u64];

/// Prometheus text for every speculation counter.
pub fn render(s: &SpecStats) -> String {
    let mut out = String::new();
    let cols: [(&str, &str, Column); 4] = [
        ("offered", "Rounds in which the drafter produced a draft at this position.", |p| &p.offered),
        ("verified", "Rounds in which the verify window carried the draft at this position.", |p| &p.verified),
        ("reached", "Rounds in which every earlier draft was accepted, so this position was tested against the target.", |p| {
            &p.reached
        }),
        ("accepted", "Rounds in which the draft at this position equalled the target's sample.", |p| &p.accepted),
    ];
    for (name, help, col) in cols {
        let mut rows = Vec::new();
        for mode in Mode::ALL {
            for multi in 0..2 {
                for (j, &v) in col(&s.by[mode as usize][multi]).iter().enumerate() {
                    if v > 0 {
                        let labels =
                            format!("mode=\"{}\",streams=\"{}\",position=\"{}\"", mode.name(), streams_label(multi), j + 1);
                        rows.push((labels, v.to_string()));
                    }
                }
            }
        }
        emit(&mut out, &format!("rexl3_spec_position_{name}_total"), help, rows);
    }
    let mut rows = Vec::new();
    for mode in Mode::ALL {
        for multi in 0..2 {
            let r = s.by[mode as usize][multi].rounds;
            if r > 0 {
                rows.push((format!("mode=\"{}\",streams=\"{}\"", mode.name(), streams_label(multi)), r.to_string()));
            }
        }
    }
    emit(&mut out, "rexl3_spec_seq_rounds_total", "Sequence-rounds recorded in the position counters.", rows);
    for (name, help, reached) in [
        (
            "reached",
            "Draft positions tested, by the draft's own confidence (kind: score or margin; bin lower edge in `ge`).",
            true,
        ),
        (
            "accepted",
            "Draft positions accepted, by the draft's own confidence (kind: score or margin; bin lower edge in `ge`).",
            false,
        ),
    ] {
        let mut rows = Vec::new();
        for (kind, by_mode) in [("score", &s.conf), ("margin", &s.margin)] {
            for mode in Mode::ALL {
                let c = &by_mode[mode as usize];
                for (j, row) in (if reached { &c.reached } else { &c.accepted }).iter().enumerate() {
                    for (b, &v) in row.iter().enumerate().filter(|(_, &v)| v > 0) {
                        let ge = c.lo + b as f32 * c.width;
                        rows.push((
                            format!("kind=\"{kind}\",mode=\"{}\",position=\"{}\",ge=\"{ge}\"", mode.name(), j + 1),
                            v.to_string(),
                        ));
                    }
                }
            }
        }
        emit(&mut out, &format!("rexl3_spec_conf_{name}_total"), help, rows);
    }
    let ph = &s.phases;
    // (name, help, one value per stream count, as text)
    let by_streams: [(&str, &str, Vec<String>); 3] = [
        (
            "rexl3_round_streams_total",
            "Decode rounds, by how many streams ran in them (the last label also counts wider rounds).",
            ph.rounds.iter().map(|v| v.to_string()).collect(),
        ),
        (
            "rexl3_round_verify_rows_total",
            "Rows of the verify step summed over rounds, by streams in the round.",
            ph.rows.iter().map(|v| v.to_string()).collect(),
        ),
        (
            "rexl3_round_draft_streams_total",
            "Streams that asked the drafter for drafts, summed over rounds, by streams in the round.",
            ph.draft_streams.iter().map(|v| v.to_string()).collect(),
        ),
    ];
    for (name, help, vals) in by_streams {
        let rows = vals
            .into_iter()
            .enumerate()
            .filter(|&(i, _)| ph.rounds[i] > 0)
            .map(|(i, v)| (format!("streams=\"{}\"", i + 1), v))
            .collect();
        emit(&mut out, name, help, rows);
    }
    let mut rows = Vec::new();
    for (phase, col) in [("draft", &ph.draft_ns), ("verify", &ph.verify_ns), ("commit", &ph.commit_ns), ("plan", &ph.plan_ns)] {
        // the plan phase exists only in tree rounds: a server without --spec-tree exports exactly the series it did before
        for (i, &ns) in col.iter().enumerate().filter(|&(i, &ns)| ph.rounds[i] > 0 && (phase != "plan" || ns > 0)) {
            rows.push((format!("phase=\"{phase}\",streams=\"{}\"", i + 1), format!("{:.6}", ns as f64 / 1e9)));
        }
    }
    emit(
        &mut out,
        "rexl3_round_phase_seconds_total",
        "Engine-thread wall seconds in each phase of a decode round (draft, verify, commit; plan: a tree round's search and row allocation), by streams in the round.",
        rows,
    );
    // the same rounds by (stream class, rows of the verify step): the served cost of a window of each width
    let mut by_rows: [Vec<(String, String)>; 3] = Default::default();
    for (c, row) in ph.by_rows.iter().enumerate() {
        for (r, cell) in row.iter().enumerate().filter(|(_, cell)| cell.rounds > 0) {
            let labels = format!("streams=\"{}\",rows=\"{r}\"", if c + 1 == ph.by_rows.len() { "4+".to_string() } else { (c + 1).to_string() });
            by_rows[0].push((labels.clone(), cell.rounds.to_string()));
            by_rows[1].push((labels.clone(), format!("{:.6}", cell.verify_ns as f64 / 1e9)));
            by_rows[2].push((labels, format!("{:.6}", cell.draft_ns as f64 / 1e9)));
        }
    }
    let [rounds, verify, draft] = by_rows;
    emit(&mut out, "rexl3_round_rows_total", "Decode rounds by streams and rows of the verify step (the last rows label also counts wider steps).", rounds);
    emit(&mut out, "rexl3_round_rows_verify_seconds_total", "Verify-phase wall seconds of those rounds.", verify);
    emit(&mut out, "rexl3_round_rows_draft_seconds_total", "Draft-phase wall seconds of those rounds.", draft);
    render_trees(&mut out, s);
    out
}

/// The counters of the verify windows that carried draft TREES (`--spec-tree`): per mode the rounds, the nodes the search ranked and the
/// nodes verified (`drafted`), the accepted path lengths (`accepted`), the calibrated expectation to compare them with, and per depth and
/// per sibling rank what each bought. Nothing is exported before a tree round has run, so a server without `--spec-tree` is unchanged.
fn render_trees(out: &mut String, s: &SpecStats) {
    if s.trees.iter().all(|t| t.rounds == 0) {
        return;
    }
    let scalar = |name: &str, help: &str, f: &dyn Fn(&TreeStats) -> String| {
        let rows = Mode::ALL
            .iter()
            .filter(|m| s.trees[**m as usize].rounds > 0)
            .map(|m| (format!("mode=\"{}\"", m.name()), f(&s.trees[*m as usize])))
            .collect();
        (name.to_string(), help.to_string(), rows)
    };
    let families: [(String, String, Samples); 9] = [
        scalar("rexl3_spec_tree_rounds_total", "Sequence-rounds whose verify window carried a draft tree.", &|t| {
            t.rounds.to_string()
        }),
        scalar(
            "rexl3_spec_tree_nodes_offered_total",
            "Nodes the tree search ranked, before the row allocation cut the tree.",
            &|t| t.offered.to_string(),
        ),
        scalar(
            "rexl3_spec_tree_nodes_verified_total",
            "Nodes verified (drafted tokens in a tree round: rows past the anchor).",
            &|t| t.verified.to_string(),
        ),
        scalar(
            "rexl3_spec_tree_accepted_total",
            "Drafts accepted in tree rounds (length of the accepted path past the anchor).",
            &|t| t.accepted_total.to_string(),
        ),
        scalar(
            "rexl3_spec_tree_copy_rounds_total",
            "Sequence-rounds whose tree carried a copy branch (tokens copied from the stream's own text, hung off the draft's picks).",
            &|t| t.copy_rounds.to_string(),
        ),
        scalar("rexl3_spec_tree_copy_nodes_verified_total", "Copied tokens verified.", &|t| t.copy_verified.to_string()),
        scalar("rexl3_spec_tree_copy_accepted_total", "Copied tokens on the accepted path.", &|t| t.copy_accepted.to_string()),
        scalar(
            "rexl3_spec_tree_copy_predicted_accepted_total",
            "Sum of the probabilities of the verified copied tokens: what the allocation expected copy_accepted_total to be.",
            &|t| format!("{:.6}", t.copy_predicted),
        ),
        scalar(
            "rexl3_spec_tree_predicted_accepted_total",
            "Sum of the calibrated probabilities of the verified nodes: what the allocation expected accepted_total to be.",
            &|t| format!("{:.6}", t.predicted),
        ),
    ];
    for (name, help, rows) in families {
        emit(out, &name, &help, rows);
    }
    let by_depth: [(&str, &str, TreeColumn); 3] = [
        ("verified", "Nodes verified at this depth (1: a child of the anchor).", |t| &t.depth_verified),
        ("tested", "Rounds in which the walk reached the depth above at a node that had a child in the window.", |t| {
            &t.depth_tested
        }),
        ("accepted", "Rounds in which the accepted path reached this depth.", |t| &t.depth_accepted),
    ];
    for (name, help, col) in by_depth {
        let mut rows = Vec::new();
        for mode in Mode::ALL {
            for (d, &v) in col(&s.trees[mode as usize]).iter().enumerate().filter(|(_, &v)| v > 0) {
                rows.push((format!("mode=\"{}\",depth=\"{}\"", mode.name(), d + 1), v.to_string()));
            }
        }
        emit(out, &format!("rexl3_spec_tree_depth_{name}_total"), help, rows);
    }
    let by_rank: [(&str, &str, TreeColumn); 2] = [
        ("offered", "Accepted-path nodes that had a child of this sibling rank (0: ranked first) in the window.", |t| {
            &t.rank_offered
        }),
        ("accepted", "Accepted-path nodes whose accepted child had this sibling rank.", |t| &t.rank_accepted),
    ];
    for (name, help, col) in by_rank {
        let mut rows = Vec::new();
        for mode in Mode::ALL {
            for (r, &v) in col(&s.trees[mode as usize]).iter().enumerate().filter(|(_, &v)| v > 0) {
                rows.push((format!("mode=\"{}\",rank=\"{}\"", mode.name(), r), v.to_string()));
            }
        }
        emit(out, &format!("rexl3_spec_tree_rank_{name}_total"), help, rows);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_recorded_exports_only_headers() {
        let t = render(&SpecStats::default());
        assert!(t.contains("# TYPE rexl3_spec_position_accepted_total counter"));
        assert!(!t.lines().any(|l| !l.starts_with('#')), "no sample lines before traffic: {t}");
    }

    #[test]
    fn counters_come_out_labelled_by_mode_streams_and_position() {
        let mut s = SpecStats::default();
        s.record(Mode::Sampled, 1, 7, &[1.0, 3.0], &[0.5, 0.7], 2, 1);
        s.record(Mode::Greedy, 6, 7, &[], &[], 7, 7);
        s.phases.record(6, 48, 6, 16_000_000, 90_000_000, 7_000_000);
        let t = render(&s);
        // a one-stream sampled round that accepted one of two verified drafts
        assert!(t.contains("rexl3_spec_position_offered_total{mode=\"sampled\",streams=\"1\",position=\"7\"} 1"), "{t}");
        assert!(t.contains("rexl3_spec_position_verified_total{mode=\"sampled\",streams=\"1\",position=\"2\"} 1"), "{t}");
        assert!(!t.contains("rexl3_spec_position_verified_total{mode=\"sampled\",streams=\"1\",position=\"3\"}"), "{t}");
        assert!(t.contains("rexl3_spec_position_reached_total{mode=\"sampled\",streams=\"1\",position=\"2\"} 1"), "{t}");
        assert!(t.contains("rexl3_spec_position_accepted_total{mode=\"sampled\",streams=\"1\",position=\"1\"} 1"), "{t}");
        assert!(t.contains("rexl3_spec_position_accepted_total{mode=\"greedy\",streams=\"2+\",position=\"7\"} 1"), "{t}");
        assert!(t.contains("rexl3_spec_seq_rounds_total{mode=\"greedy\",streams=\"2+\"} 1"), "{t}");
        // draft 1 scored 1.0, which is in the bin starting at 0 (width 2, from -8)
        assert!(t.contains("rexl3_spec_conf_accepted_total{kind=\"score\",mode=\"sampled\",position=\"1\",ge=\"0\"} 1"), "{t}");
        assert!(t.contains("rexl3_round_phase_seconds_total{phase=\"draft\",streams=\"6\"} 0.016000"), "{t}");
        assert!(t.contains("rexl3_round_verify_rows_total{streams=\"6\"} 48"), "{t}");
        assert!(!t.contains("phase=\"plan\""), "no plan series without a tree round: {t}");
        s.phases.record_plan(6, 400_000);
        assert!(render(&s).contains("rexl3_round_phase_seconds_total{phase=\"plan\",streams=\"6\"} 0.000400"));
    }

    #[test]
    fn no_tree_series_until_a_tree_round_ran() {
        // a server without --spec-tree exports exactly what it did before trees existed
        let t = render(&SpecStats::default());
        assert!(!t.contains("rexl3_spec_tree"), "{t}");
        let mut s = SpecStats::default();
        s.record(Mode::Greedy, 1, 7, &[], &[], 7, 3);
        assert!(!render(&s).contains("rexl3_spec_tree"));
    }

    #[test]
    fn tree_counters_come_out_by_mode_depth_and_sibling_rank() {
        let mut s = SpecStats::default();
        // anchor with two children (rows 1, 2); row 2 has a child (row 3); the walk takes row 2, then row 3
        s.trees[Mode::Sampled as usize].record(&[-1, 0, 0, 2], &[0, 2, 3], 5, 1.25);
        let t = render(&s);
        assert!(t.contains("rexl3_spec_tree_rounds_total{mode=\"sampled\"} 1"), "{t}");
        assert!(t.contains("rexl3_spec_tree_nodes_offered_total{mode=\"sampled\"} 5"), "{t}");
        assert!(t.contains("rexl3_spec_tree_nodes_verified_total{mode=\"sampled\"} 3"), "{t}");
        assert!(t.contains("rexl3_spec_tree_accepted_total{mode=\"sampled\"} 2"), "{t}");
        assert!(t.contains("rexl3_spec_tree_predicted_accepted_total{mode=\"sampled\"} 1.250000"), "{t}");
        assert!(t.contains("rexl3_spec_tree_depth_verified_total{mode=\"sampled\",depth=\"1\"} 2"), "{t}");
        assert!(t.contains("rexl3_spec_tree_depth_verified_total{mode=\"sampled\",depth=\"2\"} 1"), "{t}");
        assert!(t.contains("rexl3_spec_tree_depth_tested_total{mode=\"sampled\",depth=\"1\"} 1"), "{t}");
        assert!(t.contains("rexl3_spec_tree_depth_accepted_total{mode=\"sampled\",depth=\"2\"} 1"), "{t}");
        // at the anchor the accepted child was the second ranked; at row 2 the first
        assert!(t.contains("rexl3_spec_tree_rank_offered_total{mode=\"sampled\",rank=\"1\"} 1"), "{t}");
        assert!(t.contains("rexl3_spec_tree_rank_accepted_total{mode=\"sampled\",rank=\"1\"} 1"), "{t}");
        assert!(t.contains("rexl3_spec_tree_rank_accepted_total{mode=\"sampled\",rank=\"0\"} 1"), "{t}");
        assert!(!t.contains("mode=\"greedy\",depth"), "{t}");
        // a tree round without a copy branch exports the copy counters as zeros, never a copy round
        assert!(t.contains("rexl3_spec_tree_copy_rounds_total{mode=\"sampled\"} 0"), "{t}");
    }

    #[test]
    fn copy_counters_count_the_copied_nodes_verified_accepted_and_expected() {
        let mut s = SpecStats::default();
        // rows 1..=4 are a window's nodes: the picks (rows 1, 2: not copied) and two copied tokens (rows 3, 4); the walk accepts rows 1, 2, 3
        let (parents, path) = ([-1, 0, 1, 2, 3], [0usize, 1, 2, 3]);
        let t = &mut s.trees[Mode::Greedy as usize];
        t.record(&parents, &path, 6, 2.0);
        t.record_copy(&[false, false, true, true], &[0.9, 0.8, 0.7, 0.6], &path);
        let out = render(&s);
        assert!(out.contains("rexl3_spec_tree_copy_rounds_total{mode=\"greedy\"} 1"), "{out}");
        assert!(out.contains("rexl3_spec_tree_copy_nodes_verified_total{mode=\"greedy\"} 2"), "{out}");
        assert!(out.contains("rexl3_spec_tree_copy_accepted_total{mode=\"greedy\"} 1"), "{out}");
        assert!(out.contains("rexl3_spec_tree_copy_predicted_accepted_total{mode=\"greedy\"} 1.300000"), "{out}");
        // a round whose window has no copied row leaves them alone
        s.trees[Mode::Greedy as usize].record_copy(&[false, false], &[0.9, 0.8], &[0, 1]);
        assert!(render(&s).contains("rexl3_spec_tree_copy_rounds_total{mode=\"greedy\"} 1"));
    }
}
