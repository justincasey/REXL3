//! The engine's side of the prefix cache (design: `docs/PREFIX_CACHE.md`): the index and its policy, what a backend must offer to be
//! cached, and the counters. The backend owns the memory (snapshot slots, KV pages) and does the copies; the engine owns the decisions
//! (what to look up, where to cut, what to keep, what to evict) and calls the backend to release whatever the index lets go of.

pub use gb_sched::prefix::{CacheBudget, Captured, CutPolicy, Eviction, PrefixHandle};
use gb_sched::prefix::{Ghost, PrefixIndex};

/// What a backend that can be cached reports (see `Backend::prefix_caps`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixCaps {
    /// Everything that makes snapshots non-interchangeable (model and draft identity, dtypes, the pinned algorithm, the cache format):
    /// the index's keys are bound to it.
    pub namespace: u64,
    /// The physical size of the cache: snapshot slots and pages the backend reserved for it.
    pub budget: CacheBudget,
}

/// How the engine uses the cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixConfig {
    pub policy: CutPolicy,
    /// Most block hashes the ghost set remembers (8 bytes each plus a queue entry).
    pub ghost_blocks: usize,
    /// What the index evicts first (`docs/LONG_PREFILL_POLICY.md`): [`Eviction::Lru`] (the default, as before) or [`Eviction::SizeAware`],
    /// whose `overhead_pages` the engine fills in from the backend's budget (a snapshot slot's bytes in pages), whatever is given here.
    pub eviction: Eviction,
    /// Progressive cuts (0, the default: off). A prompt with more than this many tokens left to prefill also cuts at every multiple of it (the
    /// "checkpoints"), keeping only the latest one, and a prefill that is cancelled (a client that left, a request deadline) commits the
    /// snapshots it completed instead of discarding them, so a retry or a second question resumes from the last checkpoint. A slice that fails
    /// still discards everything. Off: the first release's rule (only the end cut and the shared cut; nothing of an incomplete prefill is kept).
    pub checkpoint_every: usize,
    /// Every planned cut is moved down to the nearest multiple of this (tokens; 0 or 1: off). A cached
    /// prefix that ends on a page boundary is restored from whole pages only: no copy-on-write partial
    /// page exists, so no donor continuation can ride into the resumed slot inside a page or a pooled
    /// plane. Families whose snapshots carry per-page derived state whose restore does NOT scrub the
    /// partial page (see e.g. the Flash-Next `copy_page` scrub) can set this to the KV page size; the
    /// rest keep the default. Flash-Next's restore is exact at any cut, so it leaves this off. The cost
    /// of a hit is up to `align - 1` extra re-prefilled tokens.
    pub cut_align: usize,
}

impl Default for PrefixConfig {
    fn default() -> Self {
        Self { policy: CutPolicy::default(), ghost_blocks: 1 << 16, eviction: Eviction::Lru, checkpoint_every: 0, cut_align: 0 }
    }
}

/// The cache's counters and gauges (a copy, cheap enough to take every step).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrefixStats {
    /// The engine runs a cache (the backend supports one and it was asked for).
    pub enabled: bool,
    /// Requests that looked the cache up (a prompt of more than 128 tokens on the GEMM path, text only, not opted out).
    pub lookups: u64,
    /// Prompt tokens of those requests.
    pub lookup_tokens: u64,
    pub hits: u64,
    /// Prompt tokens a hit did not prefill.
    pub tokens_saved: u64,
    pub inserts: u64,
    /// Captures dropped because the same prefix was stored already.
    pub duplicates: u64,
    /// Captures dropped because they could not fit.
    pub no_room: u64,
    pub evictions: u64,
    /// Of the evictions: made to fit a new entry's pages in the cache's page budget, made to free a snapshot slot for a capture, and made to
    /// give a cache that keeps idle pool pages (`Backend::kv_pool`) back to the sequences that may claim them.
    pub evicted_for_pages: u64,
    pub evicted_for_slots: u64,
    pub evicted_for_pool: u64,
    /// A restore that failed (the entry was dropped and the prompt prefilled cold).
    pub restore_failures: u64,
    /// Snapshots a prefill took that were released unused (the sequence failed or was cancelled before its prompt was complete).
    pub captures_discarded: u64,
    /// The long-prefill policy is on (size-aware eviction or checkpoints, `docs/LONG_PREFILL_POLICY.md`): the four counters below are
    /// exported only then, so a server that does not ask for it serves the `/metrics` text it did before.
    pub long_prefill_policy: bool,
    /// Inserts refused because the room was held by entries of more value (size-aware eviction; also in `no_room`).
    pub refused_by_value: u64,
    /// Checkpoint snapshots taken (`PrefixConfig::checkpoint_every`), and those of them released because a later checkpoint replaced them.
    pub checkpoints: u64,
    pub checkpoints_superseded: u64,
    /// Snapshots of a prefill that was cancelled before its prompt was complete, offered to the index instead of discarded (checkpoints on).
    pub kept_on_cancel: u64,
    pub entries: u64,
    pub pages_held: u64,
    pub snapshot_slots: u64,
    pub page_blocks: u64,
    /// Bytes the entries hold: snapshot slots plus distinct pages.
    pub bytes_held: u64,
    /// Bytes the backend reserved for the cache.
    pub bytes_budget: u64,
}

impl PrefixStats {
    /// Fraction of lookups that hit.
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.lookups as f64
        }
    }
}

/// The per-sequence state of a prefill that uses the cache.
pub(crate) struct PrefillCache {
    pub path: u32,
    pub prompt: std::sync::Arc<Vec<u32>>,
    /// Cuts still to be taken, ascending (each above the prefill's progress).
    pub cuts: Vec<usize>,
    /// Snapshots taken so far, kept until the prompt is complete and its first token sampled.
    pub pending: Vec<Captured>,
    /// The cuts of `cuts` that are checkpoints (`PrefixConfig::checkpoint_every`): of their snapshots only the latest is kept.
    pub rolling: Vec<usize>,
}

pub(crate) struct PrefixState {
    pub index: PrefixIndex,
    pub ghost: Ghost,
    pub cfg: PrefixConfig,
    pub budget: CacheBudget,
    pub stats: PrefixStats,
}

impl PrefixState {
    pub fn new(caps: PrefixCaps, cfg: PrefixConfig) -> Self {
        let stats = PrefixStats {
            enabled: true,
            snapshot_slots: caps.budget.snapshot_slots as u64,
            page_blocks: caps.budget.page_blocks as u64,
            bytes_budget: caps.budget.bytes(),
            long_prefill_policy: cfg.eviction != Eviction::Lru || cfg.checkpoint_every > 0,
            ..Default::default()
        };
        let mut index = PrefixIndex::new(caps.namespace, caps.budget.limits());
        if let Eviction::SizeAware { .. } = cfg.eviction {
            let overhead_pages = caps.budget.snapshot_bytes.div_ceil(caps.budget.page_bytes.max(1)) as usize;
            index.set_eviction(Eviction::SizeAware { overhead_pages });
        }
        Self { index, ghost: Ghost::new(caps.namespace, cfg.ghost_blocks), cfg, budget: caps.budget, stats }
    }

    /// The counters with the gauges read from the index now.
    pub fn snapshot(&self) -> PrefixStats {
        let mut s = self.stats.clone();
        s.inserts = self.index.stats.inserts;
        s.duplicates = self.index.stats.duplicates;
        s.no_room = self.index.stats.no_room;
        s.evictions = self.index.stats.evictions;
        s.evicted_for_pages = self.index.stats.evicted_for_pages;
        s.evicted_for_slots = self.index.stats.evicted_for_slots;
        s.refused_by_value = self.index.stats.refused_by_value;
        s.entries = self.index.len() as u64;
        s.pages_held = self.index.pages_held() as u64;
        s.bytes_held = self.index.snapshots_held() as u64 * self.budget.snapshot_bytes
            + self.index.pages_held() as u64 * self.budget.page_bytes;
        s
    }
}
