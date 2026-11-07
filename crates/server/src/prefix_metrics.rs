//! `/metrics` text for the prefix cache (`gb_engine::PrefixStats`): lookups, hits, tokens saved, inserts, evictions, and what the cache
//! holds. Counters only grow; a client that wants one workload's hit rate reads them before and after and subtracts.

use gb_engine::PrefixStats;
use std::fmt::Write;

fn put(out: &mut String, name: &str, kind: &str, help: &str, v: impl std::fmt::Display) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}");
}

/// Prometheus text for the cache. `rexl3_prefix_cache_enabled` is always there (0 when the engine runs without one), so a dashboard can tell
/// "off" from "no traffic yet"; the rest is only emitted when it is on.
pub fn render(s: &PrefixStats) -> String {
    let mut out = String::new();
    put(&mut out, "rexl3_prefix_cache_enabled", "gauge", "1 when the engine runs a prefix cache.", s.enabled as u8);
    if !s.enabled {
        return out;
    }
    let c = "counter";
    let g = "gauge";
    put(
        &mut out,
        "rexl3_prefix_cache_lookups_total",
        c,
        "Requests that looked the prefix cache up (above 128 prompt tokens, text, not opted out).",
        s.lookups,
    );
    put(
        &mut out,
        "rexl3_prefix_cache_lookup_tokens_total",
        c,
        "Prompt tokens of the requests that looked the cache up.",
        s.lookup_tokens,
    );
    put(&mut out, "rexl3_prefix_cache_hits_total", c, "Lookups that found a stored prefix and resumed from it.", s.hits);
    put(&mut out, "rexl3_prefix_cache_tokens_saved_total", c, "Prompt tokens a hit did not prefill.", s.tokens_saved);
    put(&mut out, "rexl3_prefix_cache_inserts_total", c, "Snapshots stored.", s.inserts);
    put(
        &mut out,
        "rexl3_prefix_cache_duplicates_total",
        c,
        "Snapshots dropped because the prefix was stored already.",
        s.duplicates,
    );
    put(&mut out, "rexl3_prefix_cache_no_room_total", c, "Snapshots dropped because they could not fit.", s.no_room);
    put(
        &mut out,
        "rexl3_prefix_cache_evictions_total",
        c,
        "Entries evicted (least recently used first, a prefix hit at least twice last).",
        s.evictions,
    );
    put(
        &mut out,
        "rexl3_prefix_cache_evicted_for_pages_total",
        c,
        "Evictions made to fit a new entry's pages in the page budget.",
        s.evicted_for_pages,
    );
    put(
        &mut out,
        "rexl3_prefix_cache_evicted_for_slots_total",
        c,
        "Evictions made to free a snapshot slot for a capture.",
        s.evicted_for_slots,
    );
    put(
        &mut out,
        "rexl3_prefix_cache_evicted_for_pool_total",
        c,
        "Evictions made to give idle KV pool pages back to the admitted sequences (a cache whose pages are the pool's).",
        s.evicted_for_pool,
    );
    put(
        &mut out,
        "rexl3_prefix_cache_restore_failures_total",
        c,
        "Restores that failed (the entry was dropped and the prompt prefilled cold).",
        s.restore_failures,
    );
    put(
        &mut out,
        "rexl3_prefix_cache_captures_discarded_total",
        c,
        "Snapshots released unused because their sequence failed or was cancelled while prefilling.",
        s.captures_discarded,
    );
    if s.long_prefill_policy {
        render_long_prefill(&mut out, s);
    }
    put(&mut out, "rexl3_prefix_cache_entries", g, "Entries stored.", s.entries);
    put(&mut out, "rexl3_prefix_cache_pages", g, "Distinct KV pages the entries hold.", s.pages_held);
    put(&mut out, "rexl3_prefix_cache_snapshot_slots", g, "Snapshot slots reserved for the cache.", s.snapshot_slots);
    put(&mut out, "rexl3_prefix_cache_page_blocks", g, "KV pages reserved for the cache.", s.page_blocks);
    put(&mut out, "rexl3_prefix_cache_bytes", g, "Bytes the entries hold (snapshot slots plus distinct pages).", s.bytes_held);
    put(&mut out, "rexl3_prefix_cache_budget_bytes", g, "Bytes reserved for the cache at boot.", s.bytes_budget);
    put(&mut out, "rexl3_prefix_cache_hit_ratio", g, "Hits per lookup since boot.", format!("{:.6}", s.hit_rate()));
    out
}

/// The long-prefill policy's counters (`docs/LONG_PREFILL_POLICY.md`), only when it is on.
fn render_long_prefill(out: &mut String, s: &PrefixStats) {
    let c = "counter";
    put(
        out,
        "rexl3_prefix_cache_refused_by_value_total",
        c,
        "Snapshots not stored because the room was held by entries of more value (size-aware eviction).",
        s.refused_by_value,
    );
    put(
        out,
        "rexl3_prefix_cache_checkpoints_total",
        c,
        "Checkpoint snapshots a long prefill took (progressive cuts).",
        s.checkpoints,
    );
    put(
        out,
        "rexl3_prefix_cache_checkpoints_superseded_total",
        c,
        "Checkpoint snapshots released because a later checkpoint of the same prefill replaced them.",
        s.checkpoints_superseded,
    );
    put(
        out,
        "rexl3_prefix_cache_kept_on_cancel_total",
        c,
        "Snapshots of a cancelled prefill offered to the cache instead of discarded (progressive cuts on).",
        s.kept_on_cancel,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_long_prefill_counters_appear_only_with_the_policy_on() {
        let off = render(&PrefixStats { enabled: true, ..Default::default() });
        assert!(!off.contains("checkpoints") && !off.contains("refused_by_value") && !off.contains("kept_on_cancel"));
        let on = render(&PrefixStats {
            enabled: true,
            long_prefill_policy: true,
            refused_by_value: 3,
            checkpoints: 5,
            checkpoints_superseded: 4,
            kept_on_cancel: 1,
            ..Default::default()
        });
        for line in [
            "rexl3_prefix_cache_refused_by_value_total 3",
            "rexl3_prefix_cache_checkpoints_total 5",
            "rexl3_prefix_cache_checkpoints_superseded_total 4",
            "rexl3_prefix_cache_kept_on_cancel_total 1",
        ] {
            assert!(on.contains(&format!("\n{line}\n")), "{line}");
        }
    }

    #[test]
    fn a_cache_that_is_off_says_so_and_nothing_else() {
        let t = render(&PrefixStats::default());
        assert!(t.contains("rexl3_prefix_cache_enabled 0"));
        assert!(!t.contains("hits_total"));
    }

    #[test]
    fn the_counters_and_the_hit_ratio_are_rendered() {
        let s = PrefixStats {
            enabled: true,
            lookups: 8,
            hits: 6,
            tokens_saved: 20_000,
            entries: 3,
            bytes_held: 1 << 30,
            ..Default::default()
        };
        let t = render(&s);
        for line in [
            "rexl3_prefix_cache_enabled 1",
            "rexl3_prefix_cache_lookups_total 8",
            "rexl3_prefix_cache_hits_total 6",
            "rexl3_prefix_cache_tokens_saved_total 20000",
            "rexl3_prefix_cache_entries 3",
            "rexl3_prefix_cache_bytes 1073741824",
            "rexl3_prefix_cache_hit_ratio 0.750000",
        ] {
            assert!(t.contains(line), "{line}\n{t}");
        }
        assert!(t.contains("# TYPE rexl3_prefix_cache_hits_total counter"));
    }
}
