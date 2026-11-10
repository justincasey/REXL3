//! The Flash-Next window and MTP-chain graph caches: a bounded set of instantiated CUDA graphs with a policy for which shapes
//! earn one, and the counters (`/metrics`, an engine event every [`EVENT_EVERY`] captures) that show whether the cache holds.
//!
//! **Cost.** A captured window is about 1,300 kernel nodes. Measured on a GB10 node, every captured graph took about
//! 7.4 MiB of `MemAvailable` (3.9 to 4.3 MiB of it the process's resident set), 5.3 MiB without the fork/join branches
//! (`REXL3_FORK=0`). A shape is the rows of every sequence of the window, the QSA run mask, the context bucket and the draft length;
//! four concurrent chat streams with adaptive drafts met 4,252 distinct window shapes in 7,050 windows, and the caches, which had no
//! bound, took 14.3 GiB in 82 seconds.
//!
//! **Policy.** The first sighting of a shape runs its kernels eagerly and only remembers the key; the second one captures and
//! replays; every later one replays. A shape seen once costs no capture and no graph memory. The memory of first sightings is two
//! generations of [`GraphCache::limit`] keys each: a key is forgotten when it is older than that many other first sightings, so only
//! a shape that comes back soon, while a graph for it could still be in the cache, earns one. (Replaying a recorded four-stream
//! trace of 7,050 windows through both rules: 502 captures against 1,136 for a plain second-sighting rule, and 23.1% of the windows
//! replayed against 23.6%.) At most [`DEFAULT_WINDOW_GRAPHS`] window graphs and [`DEFAULT_CHAIN_GRAPHS`] chain graphs are alive at
//! once; a capture that finds its cache full retires the least recently launched graph first (that shape starts over at its next
//! sighting). The defaults leave room for the hot shapes of one stream (the six prompts of the decode screen, one after the other,
//! made 74 window and 22 chain graphs in their first 96 captures) and cost at most (128 + 64) x 7.4 MiB = 1.4 GiB.
//!
//! None of this changes a bit of output: eager launches, a capture's launches and a replay run the same kernels with the same
//! arguments in the same order; only when a graph exists, and for how long, changes.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Window graphs (a window, or each of its two parts when the PLE stage splits it) alive at once.
pub const DEFAULT_WINDOW_GRAPHS: usize = 128;
/// MTP draft-chain graphs alive at once.
pub const DEFAULT_CHAIN_GRAPHS: usize = 64;

/// `REXL3_FN_WINDOW_GRAPHS`: window graphs alive at once (a positive number; unset or unusable: [`DEFAULT_WINDOW_GRAPHS`]).
pub const ENV_WINDOW_GRAPHS: &str = "REXL3_FN_WINDOW_GRAPHS";
/// `REXL3_FN_CHAIN_GRAPHS`: MTP-chain graphs alive at once (a positive number; unset or unusable: [`DEFAULT_CHAIN_GRAPHS`]).
pub const ENV_CHAIN_GRAPHS: &str = "REXL3_FN_CHAIN_GRAPHS";
/// `REXL3_FN_GRAPH_AT=0` captures a shape at its first sighting instead of its second (a test knob: every shape then runs through
/// a captured graph from its first window).
pub const ENV_GRAPH_AT: &str = "REXL3_FN_GRAPH_AT";

/// A cache's cap from an environment value: a positive number, else `default`.
pub fn parse_limit(value: Option<&str>, default: usize) -> usize {
    value.and_then(|v| v.trim().parse::<usize>().ok()).filter(|&n| n > 0).unwrap_or(default)
}

/// The cap of a cache from the environment variable `key`.
pub fn limit_from_env(key: &str, default: usize) -> usize {
    parse_limit(std::env::var(key).ok().as_deref(), default)
}

/// Whether an environment value asks for captures at the first sighting (`0`).
pub fn parse_capture_first(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim() == "0")
}

/// Whether shapes are captured at their first sighting ([`ENV_GRAPH_AT`] is `0`).
pub fn capture_first_from_env() -> bool {
    parse_capture_first(std::env::var(ENV_GRAPH_AT).ok().as_deref())
}

/// What to do with a shape that has no graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admit {
    /// Capture it now and replay the capture.
    Capture,
    /// Run it eagerly: it has not been seen lately.
    Eager,
}

/// A bounded cache of instantiated graphs (`V`) by shape (`K`): least recently launched out, a graph only after a second
/// sighting that comes soon enough after the first. It makes no CUDA calls; the owner waits for its stream before it lets a
/// retired graph drop.
pub struct GraphCache<K, V> {
    ready: HashMap<K, (V, u64)>,
    /// Shapes seen once, no graph: the newer generation, and the one before it.
    young: HashSet<K>,
    old: HashSet<K>,
    limit: usize,
    /// Shapes a generation holds before the next one starts.
    generation: usize,
    capture_first: bool,
    tick: u64,
}

impl<K: Copy + Eq + Hash, V> GraphCache<K, V> {
    /// A cache of at most `limit` graphs (at least one). `capture_first`: every unseen shape is captured at once.
    pub fn new(limit: usize, capture_first: bool) -> Self {
        Self::with_generation(limit, limit, capture_first)
    }

    fn with_generation(limit: usize, generation: usize, capture_first: bool) -> Self {
        Self {
            ready: HashMap::new(),
            young: HashSet::new(),
            old: HashSet::new(),
            limit: limit.max(1),
            generation: generation.max(1),
            capture_first,
            tick: 0,
        }
    }

    /// Graphs alive now.
    pub fn len(&self) -> usize {
        self.ready.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ready.is_empty()
    }

    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Whether a new graph needs a retirement first.
    pub fn is_full(&self) -> bool {
        self.ready.len() >= self.limit
    }

    /// The graph of `key`, now the most recently launched.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.tick += 1;
        let tick = self.tick;
        self.ready.get_mut(key).map(|(graph, used)| {
            *used = tick;
            &*graph
        })
    }

    /// The verdict for a shape that has no graph: [`Admit::Capture`] when it was seen lately (or captures are at first sighting),
    /// [`Admit::Eager`] and a memory of the key when it is new.
    pub fn admit(&mut self, key: K) -> Admit {
        debug_assert!(!self.ready.contains_key(&key), "admit is for a shape without a graph");
        if self.capture_first || self.young.remove(&key) || self.old.remove(&key) {
            return Admit::Capture;
        }
        if self.young.len() >= self.generation {
            self.old = std::mem::take(&mut self.young);
        }
        self.young.insert(key);
        Admit::Eager
    }

    /// Retire the least recently launched graph; its shape starts over (two more sightings to earn a graph again). The caller
    /// MUST have waited for the stream the graphs launch on before it drops the returned graph: it may still be running.
    pub fn retire_oldest(&mut self) -> Option<V> {
        let oldest = self.ready.iter().min_by_key(|(_, (_, used))| *used).map(|(k, _)| *k)?;
        self.ready.remove(&oldest).map(|(graph, _)| graph)
    }

    /// Keep a freshly captured graph, as the most recently launched. The cache must have room ([`GraphCache::is_full`]).
    pub fn insert(&mut self, key: K, graph: V) {
        assert!(self.ready.len() < self.limit, "a graph was captured into a full cache");
        self.tick += 1;
        self.ready.insert(key, (graph, self.tick));
    }

    /// Shapes remembered as seen once (no graph).
    pub fn seen_len(&self) -> usize {
        self.young.len() + self.old.len()
    }
}

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
    /// Graphs retired to make room for another (the least recently launched first).
    pub evictions: u64,
    /// Calls that ran their kernels eagerly (graphs off, a shape's first sighting, or a capture the runtime refused).
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
    /// chains are reported as the draft head's own cache, and `evictions` counts both caches'.
    pub fn counters(&self) -> gb_engine::GraphCounters {
        gb_engine::GraphCounters {
            replays: self.window.replays,
            captures: self.window.captures,
            eager_windows: self.window.eager,
            evictions: self.window.evictions + self.chain.evictions,
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
             window_replays={} chain_replays={} window_evictions={} chain_evictions={} eager={} capture_ms={:.0} free_drop_mib={drop:.1} \
             free_drop_per_capture_mib={:.3} rss_mib={:.1} device_free_mib={:.1}",
            host.epoch_s,
            self.captures(),
            self.window.captures,
            self.chain.captures,
            live.0,
            live.1,
            self.window.replays,
            self.chain.replays,
            self.window.evictions,
            self.chain.evictions,
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
        s.of(GraphKind::Window).evictions = 4;
        s.of(GraphKind::Chain).evictions = 1;
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
            "window_evictions=4",
            "chain_evictions=1",
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
        *s.of(GraphKind::Window) = KindStats { replays: 7, captures: 2, evictions: 6, eager: 1 };
        *s.of(GraphKind::Chain) = KindStats { replays: 5, captures: 3, evictions: 2, eager: 4 };
        s.capture_ms = 9.5;
        let c = s.counters();
        assert_eq!((c.replays, c.captures, c.eager_windows), (7, 2, 1));
        assert_eq!((c.draft_replays, c.draft_captures, c.draft_eager), (5, 3, 4));
        assert_eq!((c.capture_ms, c.evictions, c.capture_failures), (9.5, 8, 0));
    }
}
