//! What the Flash-Next window and MTP-chain graph caches have done: counters for `/metrics` and an engine event every
//! [`EVENT_EVERY`] captures, so a cache that grows (or thrashes) shows in the log before the node runs out of memory.

/// An engine event goes out after every this many captures (windows and chains together).
pub const EVENT_EVERY: u64 = 32;

/// Which cache a graph belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphKind {
    /// A verify / decode window (or one of its two parts when the PLE stage splits it).
    Window,
    /// An MTP draft chain.
    Chain,
}

/// Counters of one cache; they only grow.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KindStats {
    /// Launches of an existing graph.
    pub replays: u64,
    /// Graphs captured and instantiated (each is then replayed in the same call).
    pub captures: u64,
    /// Calls that ran their kernels eagerly (graphs off, or a capture the runtime refused).
    pub eager: u64,
}

/// The counters of both caches and what the captures cost.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GraphStats {
    pub window: KindStats,
    pub chain: KindStats,
    /// Milliseconds spent capturing and instantiating, in all.
    pub capture_ms: f64,
    /// The sum over every capture of the drop in the device's free memory (`cudaMemGetInfo`) across the capture and the
    /// instantiation. It includes whatever else allocated meanwhile, so it is an estimate that only means something over
    /// many captures.
    pub free_drop_bytes: i64,
}

/// What the process looked like when an event went out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostSample {
    /// Wall clock, seconds since the Unix epoch: lines the event up with the host's own memory samples.
    pub epoch_s: u64,
    /// Resident set of the process (`VmRSS`), KiB; 0 when unreadable.
    pub rss_kib: u64,
    /// The device's free memory (`cudaMemGetInfo`), bytes; 0 when unreadable.
    pub device_free_bytes: u64,
}

impl HostSample {
    /// Read the clock, the process's resident set and the device's free memory now.
    pub fn take() -> Self {
        let epoch_s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let rss_kib = std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| s.lines().find_map(|l| l.strip_prefix("VmRSS:")?.split_whitespace().next()?.parse().ok()))
            .unwrap_or(0);
        let device_free_bytes = gb_cuda::rt::mem_info().map_or(0, |(free, _)| free as u64);
        Self { epoch_s, rss_kib, device_free_bytes }
    }
}

impl GraphStats {
    pub fn of(&mut self, kind: GraphKind) -> &mut KindStats {
        match kind {
            GraphKind::Window => &mut self.window,
            GraphKind::Chain => &mut self.chain,
        }
    }

    pub fn captures(&self) -> u64 {
        self.window.captures + self.chain.captures
    }

    /// The engine's view of the counters (`Backend::graph_counters`, exported on `/metrics`): the windows are the cache, the
    /// chains are reported as the draft head's own cache.
    pub fn counters(&self) -> gb_engine::GraphCounters {
        gb_engine::GraphCounters {
            replays: self.window.replays,
            captures: self.window.captures,
            eager_windows: self.window.eager,
            capture_ms: self.capture_ms,
            draft_replays: self.chain.replays,
            draft_captures: self.chain.captures,
            draft_eager: self.chain.eager,
            ..Default::default()
        }
    }

    /// The `flash_next_graphs` event line: the cumulative counters, how many graphs are alive in each cache now (`live`: windows,
    /// chains), the mean free-memory drop per capture, and the process's own memory.
    pub fn event_line(&self, live: (usize, usize), host: HostSample) -> String {
        let mib = |bytes: f64| bytes / (1u64 << 20) as f64;
        let drop = mib(self.free_drop_bytes as f64);
        format!(
            "rexl3 event=flash_next_graphs epoch_s={} captures={} window_captures={} chain_captures={} window_live={} chain_live={} \
             window_replays={} chain_replays={} eager={} capture_ms={:.0} free_drop_mib={drop:.1} free_drop_per_capture_mib={:.3} \
             rss_mib={:.1} device_free_mib={:.1}",
            host.epoch_s,
            self.captures(),
            self.window.captures,
            self.chain.captures,
            live.0,
            live.1,
            self.window.replays,
            self.chain.replays,
            self.window.eager + self.chain.eager,
            self.capture_ms,
            drop / self.captures().max(1) as f64,
            host.rss_kib as f64 / 1024.0,
            mib(host.device_free_bytes as f64),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_event_names_every_counter_the_live_graphs_and_the_process_memory() {
        let mut s = GraphStats::default();
        s.of(GraphKind::Window).captures = 30;
        s.of(GraphKind::Chain).captures = 2;
        s.of(GraphKind::Window).replays = 500;
        s.of(GraphKind::Chain).replays = 40;
        s.of(GraphKind::Chain).eager = 3;
        s.capture_ms = 1234.4;
        s.free_drop_bytes = 64 << 20;
        assert_eq!(s.captures(), 32);
        let host = HostSample { epoch_s: 1_700_000_000, rss_kib: 3 * 1024 * 1024, device_free_bytes: 16 << 30 };
        let line = s.event_line((29, 2), host);
        for part in [
            "event=flash_next_graphs",
            "epoch_s=1700000000",
            "captures=32 ",
            "window_captures=30",
            "chain_captures=2",
            "window_live=29",
            "chain_live=2",
            "window_replays=500",
            "chain_replays=40",
            "eager=3",
            "capture_ms=1234",
            "free_drop_mib=64.0",
            "free_drop_per_capture_mib=2.000",
            "rss_mib=3072.0",
            "device_free_mib=16384.0",
        ] {
            assert!(line.contains(part), "{part:?} missing from {line}");
        }
    }

    #[test]
    fn an_empty_cache_reports_zero_without_dividing_by_zero() {
        let line = GraphStats::default().event_line((0, 0), HostSample::default());
        assert!(line.contains("captures=0 ") && line.contains("free_drop_per_capture_mib=0.000"), "{line}");
    }

    #[test]
    fn the_engine_sees_windows_as_the_cache_and_chains_as_the_draft_cache() {
        let mut s = GraphStats::default();
        *s.of(GraphKind::Window) = KindStats { replays: 7, captures: 2, eager: 1 };
        *s.of(GraphKind::Chain) = KindStats { replays: 5, captures: 3, eager: 4 };
        s.capture_ms = 9.5;
        let c = s.counters();
        assert_eq!((c.replays, c.captures, c.eager_windows), (7, 2, 1));
        assert_eq!((c.draft_replays, c.draft_captures, c.draft_eager), (5, 3, 4));
        assert_eq!((c.capture_ms, c.evictions, c.capture_failures), (9.5, 0, 0));
    }
}
