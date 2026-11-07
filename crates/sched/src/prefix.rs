//! The prefix index: which prompts' states are kept, found by chain hash, evicted least recently used, charged against two physical
//! limits (snapshot slots and cache pages). Pure bookkeeping: it never touches a GPU and never frees anything itself, every method
//! that drops an entry hands its [`PrefixHandle`] back for the owner of the memory (the backend) to release exactly once.
//!
//! See `docs/PREFIX_CACHE.md` for the design. In short:
//!
//! * key = chain hash over the token stream, seeded by the namespace (model, draft, dtypes, pinned algorithm, cache format) and the
//!   prefill path id; the chain value at every multiple of the block size (256) is that block's hash, an entry whose length is not a
//!   multiple extends the chain by its partial block; a hit compares the tokens, never the hash alone;
//! * an entry is a snapshot slot plus references to the KV pages of its prefix; pages are counted once however many entries share them;
//! * a lookup returns the longest entry that is a STRICT prefix of the request (one token must be left to prefill: the first sampled
//!   token needs the last row's logits);
//! * entries are evicted least recently used first, pinned entries never.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

/// A KV page id (an index into the backend's block pool).
pub type PageId = u32;

/// Tokens per KV page and per hash block.
pub const BLOCK_TOKENS: usize = 256;

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// The chain hash of a token stream: `h` after `tokens[..n]` identifies exactly that prefix (within a namespace and a path).
#[derive(Clone, Copy, Debug)]
pub struct Chain {
    h: u64,
    mask: u64,
}

impl Chain {
    pub fn new(namespace: u64, path: u32) -> Self {
        Self::masked(namespace, path, u64::MAX)
    }
    /// A chain whose value is cut to `mask` bits (tests force collisions with it).
    pub fn masked(namespace: u64, path: u32, mask: u64) -> Self {
        let h = mix64(
            mix64(namespace ^ 0x6A09_E667_F3BC_C908) ^ (path as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xBB67_AE85_84CA_A73B,
        );
        Self { h, mask }
    }
    pub fn push(&mut self, t: u32) {
        let mut x = self.h ^ (t as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03);
        x ^= x >> 29;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        self.h = x ^ (x >> 32);
    }
    pub fn value(&self) -> u64 {
        self.h & self.mask
    }
}

/// The state a captured snapshot is kept in, as the backend describes it. Opaque to the index except for the pages (the budget).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixHandle {
    /// The backend's snapshot slot.
    pub snap: u32,
    /// The KV pages of the prefix, in order, each of which the handle holds one reference to (a partial last page included).
    pub pages: Vec<PageId>,
}

/// A snapshot a prefill took at `len` tokens, not yet in the index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Captured {
    pub len: usize,
    pub handle: PrefixHandle,
}

/// What the cache may hold at most: both are physical (preallocated) limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub snapshot_slots: usize,
    pub page_blocks: usize,
}

/// A hit: the longest stored prefix of the request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    pub len: usize,
    pub handle: PrefixHandle,
    /// The entry's key, for `unpin` / `remove`.
    pub key: EntryKey,
}

/// (path id, chain hash at the entry's length)
pub type EntryKey = (u32, u64);

/// An entry hit this many times is a prefix shared by several requests, and is kept ahead of the recency order while it is live.
const SHARED_HITS: u64 = 2;

/// How the index chooses what to evict (`docs/LONG_PREFILL_POLICY.md`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Eviction {
    /// Least recently used first, a prefix hit at least [`SHARED_HITS`] times last while it is live (`lru_victim`). Sizes play no part: an
    /// entry that holds every page of the cache is the victim of the next insert that needs one page. The default.
    #[default]
    Lru,
    /// GreedyDual-Size with frequency (Cao and Irani, 1997), deterministic in index operations. Each entry has a credit
    /// `H = L + f x tokens / (pages + overhead_pages)`: the prefill it saves per page it holds (`overhead_pages` is a snapshot slot in pages),
    /// times `f = min(hits + 1, FREQ_CAP)`, on top of the index's inflation `L`. A hit renews the credit. The victim is the unpinned entry of
    /// least credit, and `L` rises to each victim's credit. An insert that would have to evict an entry of MORE credit than its own is refused
    /// instead (`Insert::NoRoom`, counted in `refused_by_value`), and `L` rises to the newcomer's credit. So a large entry is never evicted by
    /// one small one, and never starves small ones: about `f x d_large / d_small` refused small inserts after its last hit, it is the victim.
    SizeAware {
        /// A snapshot slot's bytes in pages (`snapshot_bytes / page_bytes`, rounded up): what an entry costs besides its pages.
        overhead_pages: usize,
    },
}

/// The most a hit count multiplies an entry's credit by under [`Eviction::SizeAware`].
pub const FREQ_CAP: u64 = 4;

struct Entry {
    path: u32,
    len: usize,
    /// The donor's whole prompt; the entry's tokens are `prompt[..len]`.
    prompt: Arc<Vec<u32>>,
    handle: PrefixHandle,
    last_use: u64,
    pins: u32,
    hits: u64,
    /// The entry's credit under [`Eviction::SizeAware`] (0 and unused under `Lru`).
    credit: f64,
}

/// What `insert` did with a capture. Whatever comes back in `release` must be released by the caller.
#[derive(Debug, PartialEq, Eq)]
pub enum Insert {
    Stored {
        release: Vec<PrefixHandle>,
    },
    /// The same prefix was already stored: the capture is handed back to be released.
    Duplicate {
        release: Vec<PrefixHandle>,
    },
    /// Its own pages exceed the cache's page limit, or room could not be made (every candidate victim is pinned): handed back.
    NoRoom {
        release: Vec<PrefixHandle>,
    },
}

impl Insert {
    pub fn stored(&self) -> bool {
        matches!(self, Insert::Stored { .. })
    }
    pub fn into_release(self) -> Vec<PrefixHandle> {
        match self {
            Insert::Stored { release } | Insert::Duplicate { release } | Insert::NoRoom { release } => release,
        }
    }
}

/// Counters of the index (lookups and hits are the engine's to count per request, see `gb_engine::PrefixStats`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IndexStats {
    pub inserts: u64,
    pub duplicates: u64,
    pub no_room: u64,
    pub evictions: u64,
    /// Of the evictions: made to fit a new entry's pages in the page budget, and made to free a snapshot slot (a capture was about to take one).
    pub evicted_for_pages: u64,
    pub evicted_for_slots: u64,
    /// Inserts refused because the room they needed was held by entries of more credit ([`Eviction::SizeAware`]; also counted in `no_room`).
    pub refused_by_value: u64,
}

pub struct PrefixIndex {
    namespace: u64,
    mask: u64,
    limits: Limits,
    entries: HashMap<EntryKey, Entry>,
    /// Per path, the lengths that have at least one entry, with how many: what a lookup probes, longest first.
    lengths: BTreeMap<(u32, usize), usize>,
    /// Entries holding each page.
    page_refs: HashMap<PageId, u32>,
    clock: u64,
    pub stats: IndexStats,
    eviction: Eviction,
    /// GreedyDual's inflation `L`: the credit of the last entry evicted (or refused).
    inflation: f64,
}

impl PrefixIndex {
    pub fn new(namespace: u64, limits: Limits) -> Self {
        Self::with_mask(namespace, limits, u64::MAX)
    }
    /// An index whose chain hashes are cut to `mask` (tests force collisions with it; production uses all 64 bits).
    pub fn with_mask(namespace: u64, limits: Limits, mask: u64) -> Self {
        Self {
            namespace,
            mask,
            limits,
            entries: HashMap::new(),
            lengths: BTreeMap::new(),
            page_refs: HashMap::new(),
            clock: 0,
            stats: IndexStats::default(),
            eviction: Eviction::Lru,
            inflation: 0.0,
        }
    }

    /// Choose the eviction policy (before any entry is stored; [`Eviction::Lru`] is the default).
    pub fn set_eviction(&mut self, eviction: Eviction) {
        self.eviction = eviction;
    }
    pub fn eviction(&self) -> Eviction {
        self.eviction
    }

    /// An entry's credit under [`Eviction::SizeAware`] with `hits` hits so far, at the current inflation: the prefill it saves per page held.
    fn credit_of(&self, len: usize, hits: u64) -> f64 {
        let overhead = match self.eviction {
            Eviction::SizeAware { overhead_pages } => overhead_pages,
            Eviction::Lru => 0,
        };
        let cost = (len.div_ceil(BLOCK_TOKENS) + overhead).max(1) as f64;
        self.inflation + (hits + 1).min(FREQ_CAP) as f64 * len as f64 / cost
    }

    pub fn namespace(&self) -> u64 {
        self.namespace
    }
    pub fn limits(&self) -> Limits {
        self.limits
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Distinct pages the entries hold (each counted once).
    pub fn pages_held(&self) -> usize {
        self.page_refs.len()
    }
    /// Snapshot slots the entries hold (one each).
    pub fn snapshots_held(&self) -> usize {
        self.entries.len()
    }
    /// The cut points (token counts) stored for `path`, ascending, with duplicates removed.
    pub fn lengths_of(&self, path: u32) -> Vec<usize> {
        self.lengths.range((path, 0)..=(path, usize::MAX)).map(|(&(_, l), _)| l).collect()
    }
    /// How many entries hold `page`.
    pub fn page_holders(&self, page: PageId) -> u32 {
        self.page_refs.get(&page).copied().unwrap_or(0)
    }

    fn chain(&self, path: u32) -> Chain {
        Chain::masked(self.namespace, path, self.mask)
    }

    /// The chain value after each of the first `n` tokens of `tokens` that falls on one of `at` (ascending, each `<= tokens.len()`).
    fn hashes_at(&self, path: u32, tokens: &[u32], at: &[usize]) -> Vec<u64> {
        let mut c = self.chain(path);
        let mut out = Vec::with_capacity(at.len());
        let mut next = 0usize;
        for (i, &t) in tokens.iter().enumerate() {
            if next == at.len() {
                break;
            }
            c.push(t);
            while next < at.len() && at[next] == i + 1 {
                out.push(c.value());
                next += 1;
            }
        }
        out
    }

    /// The hash an entry of `len` tokens of `tokens` is stored under.
    pub fn key_of(&self, path: u32, tokens: &[u32], len: usize) -> EntryKey {
        let mut c = self.chain(path);
        for &t in &tokens[..len] {
            c.push(t);
        }
        (path, c.value())
    }

    /// Whether an entry for exactly `tokens[..len]` is stored (tokens compared).
    pub fn contains(&self, path: u32, tokens: &[u32], len: usize) -> bool {
        if len == 0 || len > tokens.len() {
            return false;
        }
        match self.entries.get(&self.key_of(path, tokens, len)) {
            Some(e) => e.len == len && e.prompt[..len] == tokens[..len],
            None => false,
        }
    }

    /// The longest stored entry that is a strict prefix of `tokens` (shorter than it: one token is left to prefill), pinned for the
    /// caller until `unpin`; refreshes its recency. Tokens are compared in full, a hash match alone is not a hit.
    pub fn lookup(&mut self, path: u32, tokens: &[u32]) -> Option<Hit> {
        self.clock += 1;
        let cands: Vec<usize> = self
            .lengths
            .range((path, 1)..(path, tokens.len().max(1)))
            .map(|(&(_, l), _)| l)
            .filter(|&l| l < tokens.len())
            .collect();
        if cands.is_empty() {
            return None;
        }
        let hs = self.hashes_at(path, tokens, &cands);
        for (&len, &h) in cands.iter().zip(&hs).rev() {
            if let Some(e) = self.entries.get_mut(&(path, h)) {
                if e.len == len && e.prompt[..len] == tokens[..len] {
                    e.last_use = self.clock;
                    e.pins += 1;
                    e.hits += 1;
                    let hit = Hit { len, handle: e.handle.clone(), key: (path, h) };
                    if matches!(self.eviction, Eviction::SizeAware { .. }) {
                        let hits = e.hits;
                        let credit = self.credit_of(len, hits);
                        if let Some(e) = self.entries.get_mut(&(path, h)) {
                            e.credit = credit;
                        }
                    }
                    return Some(hit);
                }
            }
        }
        None
    }

    /// The length of the entry [`PrefixIndex::lookup`] would return for `tokens`, without pinning it, counting it or touching its recency:
    /// what an admission estimate may read (a request that is refused must leave the cache as it was).
    pub fn peek(&self, path: u32, tokens: &[u32]) -> Option<usize> {
        let cands: Vec<usize> = self
            .lengths
            .range((path, 1)..(path, tokens.len().max(1)))
            .map(|(&(_, l), _)| l)
            .filter(|&l| l < tokens.len())
            .collect();
        if cands.is_empty() {
            return None;
        }
        let hs = self.hashes_at(path, tokens, &cands);
        cands.iter().zip(&hs).rev().find_map(|(&len, &h)| {
            self.entries.get(&(path, h)).filter(|e| e.len == len && e.prompt[..len] == tokens[..len]).map(|_| len)
        })
    }

    pub fn unpin(&mut self, key: EntryKey) {
        if let Some(e) = self.entries.get_mut(&key) {
            e.pins = e.pins.saturating_sub(1);
        }
    }

    fn drop_entry(&mut self, key: EntryKey) -> Option<PrefixHandle> {
        let e = self.entries.remove(&key)?;
        match self.lengths.get_mut(&(e.path, e.len)) {
            Some(n) if *n > 1 => *n -= 1,
            _ => {
                self.lengths.remove(&(e.path, e.len));
            }
        }
        for p in &e.handle.pages {
            match self.page_refs.get_mut(p) {
                Some(n) if *n > 1 => *n -= 1,
                _ => {
                    self.page_refs.remove(p);
                }
            }
        }
        self.stats.evictions += 1;
        Some(e.handle)
    }

    /// Remove the entry `key` (a restore from it failed, or a test): its handle is returned to be released.
    pub fn remove(&mut self, key: EntryKey) -> Option<PrefixHandle> {
        self.drop_entry(key)
    }

    /// Whom to evict next: the least recently used entry, except that an entry hit at least `SHARED_HITS` times (a prefix several requests share: a system
    /// prompt, not one conversation's own history) is protected while it is live, and goes last. Live means touched within 8 x `snapshot_slots` index
    /// operations: a shared prefix nobody asks for any more competes by age like the rest, so it does not hold the budget for ever. Pinned entries are never chosen.
    ///
    /// Why not "entries that never hit first": the entry a conversation resumes from next turn is, by construction, one that has not hit yet, while its
    /// superseded parents have; that rule evicts the frontier of every chat before the parents nobody needs again, and a request that misses leaves another
    /// entry that has not hit, so a miss begets a miss. A prefix hit twice is evidence of sharing, the one thing recency does not show
    /// (`conversations_in_flight_resume_from_their_own_last_turn_under_pressure`: the frontier hit 99 of 300 turns with 4 chats on 16 slots under that rule, 297 with this one).
    fn lru_victim(&self) -> Option<EntryKey> {
        if matches!(self.eviction, Eviction::SizeAware { .. }) {
            return self.credit_victims().into_iter().next();
        }
        let live = 8 * self.limits.snapshot_slots as u64;
        self.entries
            .iter()
            .filter(|(_, e)| e.pins == 0)
            .min_by_key(|(_, e)| (e.hits >= SHARED_HITS && self.clock.saturating_sub(e.last_use) <= live, e.last_use))
            .map(|(&k, _)| k)
    }

    /// The unpinned entries in the order [`Eviction::SizeAware`] evicts them: least credit first, the least recently used among equals.
    fn credit_victims(&self) -> Vec<EntryKey> {
        let mut v: Vec<(f64, u64, EntryKey)> =
            self.entries.iter().filter(|(_, e)| e.pins == 0).map(|(&k, e)| (e.credit, e.last_use, k)).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        v.into_iter().map(|(_, _, k)| k).collect()
    }

    /// Evict the next victim (see `lru_victim`): the least recently used, shared prefixes last; never a pinned one. Under
    /// [`Eviction::SizeAware`] the entry of least credit, and the inflation rises to its credit.
    pub fn evict_lru(&mut self) -> Option<PrefixHandle> {
        let k = self.lru_victim()?;
        if let Some(e) = self.entries.get(&k) {
            if matches!(self.eviction, Eviction::SizeAware { .. }) {
                self.inflation = self.inflation.max(e.credit);
            }
        }
        self.drop_entry(k)
    }

    /// [`Eviction::SizeAware`]: whether storing an entry of `credit` holding the distinct pages `distinct` may evict what it needs. The
    /// victims are taken in credit order, as `insert`'s loop would take them, until the pages fit and a snapshot slot is free; `false` when one
    /// of them holds more credit than the newcomer (or the pinned entries leave too little room, which `insert` finds itself).
    fn may_make_room(&self, distinct: &[PageId], credit: f64) -> bool {
        let mut refs = self.page_refs.clone();
        let mut entries = self.entries.len();
        let short = |refs: &HashMap<PageId, u32>, entries: usize| {
            let adds = distinct.iter().filter(|p| !refs.contains_key(p)).count();
            refs.len() + adds > self.limits.page_blocks || entries >= self.limits.snapshot_slots
        };
        for k in self.credit_victims() {
            if !short(&refs, entries) {
                return true;
            }
            let e = &self.entries[&k];
            if e.credit > credit {
                return false;
            }
            let mut d = e.handle.pages.clone();
            d.sort_unstable();
            d.dedup();
            for p in d {
                match refs.get_mut(&p) {
                    Some(n) if *n > 1 => *n -= 1,
                    _ => {
                        refs.remove(&p);
                    }
                }
            }
            entries -= 1;
        }
        true
    }

    /// Make `need` snapshot slots free given that the backend reports `free_now` free: evicts LRU entries until it is so, or until only
    /// pinned ones are left. Returns the handles to release. (`snapshots_held() + pending + free_now == limits.snapshot_slots` is the
    /// engine's accounting; this only says how many to drop.)
    pub fn evict_for_snapshots(&mut self, need: usize, free_now: usize) -> Vec<PrefixHandle> {
        let mut out = Vec::new();
        let mut free = free_now;
        while free < need {
            match self.evict_lru() {
                Some(h) => {
                    self.stats.evicted_for_slots += 1;
                    out.push(h);
                    free += 1;
                }
                None => break,
            }
        }
        out
    }

    /// Store a capture of the first `cap.len` tokens of `prompt`. Evicts LRU entries until the pages fit; refuses (handing the capture back)
    /// when the same prefix is already stored or its pages cannot fit.
    pub fn insert(&mut self, path: u32, prompt: Arc<Vec<u32>>, cap: Captured) -> Insert {
        let len = cap.len;
        if len == 0 || len > prompt.len() {
            self.stats.no_room += 1;
            return Insert::NoRoom { release: vec![cap.handle] };
        }
        let key = self.key_of(path, &prompt, len);
        if self.entries.contains_key(&key) {
            self.stats.duplicates += 1;
            return Insert::Duplicate { release: vec![cap.handle] };
        }
        let mut release = Vec::new();
        // pages of this capture that no entry holds yet are what it adds to the cache's page count
        let mut distinct = cap.handle.pages.clone();
        distinct.sort_unstable();
        distinct.dedup();
        let own = distinct.len();
        if own > self.limits.page_blocks {
            self.stats.no_room += 1;
            return Insert::NoRoom { release: vec![cap.handle] };
        }
        let credit = self.credit_of(len, 0);
        if matches!(self.eviction, Eviction::SizeAware { .. }) && !self.may_make_room(&distinct, credit) {
            // GreedyDual: the newcomer is the cheapest thing in the cache, so it is the one that goes; the inflation rises to its credit, so
            // every refusal brings the entries that refused it closer to eviction
            self.inflation = self.inflation.max(credit);
            self.stats.no_room += 1;
            self.stats.refused_by_value += 1;
            return Insert::NoRoom { release: vec![cap.handle] };
        }
        loop {
            let adds = distinct.iter().filter(|p| !self.page_refs.contains_key(p)).count();
            // both limits are physical: a snapshot slot per entry (the engine keeps pending captures inside the same count, so this only
            // trips when a caller inserts without having freed a slot) and the distinct pages
            let pages_short = self.page_refs.len() + adds > self.limits.page_blocks;
            if !pages_short && self.entries.len() < self.limits.snapshot_slots {
                break;
            }
            match self.evict_lru() {
                Some(h) => {
                    if pages_short {
                        self.stats.evicted_for_pages += 1;
                    } else {
                        self.stats.evicted_for_slots += 1;
                    }
                    release.push(h)
                }
                None => {
                    // every remaining entry is pinned: keep what was evicted so far (it is gone either way) and refuse
                    self.stats.no_room += 1;
                    release.push(cap.handle);
                    return Insert::NoRoom { release };
                }
            }
        }
        self.clock += 1;
        for &p in &distinct {
            *self.page_refs.entry(p).or_insert(0) += 1;
        }
        *self.lengths.entry((path, len)).or_insert(0) += 1;
        // the credit at the inflation AFTER the evictions this insert made (GreedyDual's `H = L + value / cost`)
        let credit = if matches!(self.eviction, Eviction::SizeAware { .. }) { self.credit_of(len, 0) } else { 0.0 };
        self.entries.insert(key, Entry { path, len, prompt, handle: cap.handle, last_use: self.clock, pins: 0, hits: 0, credit });
        self.stats.inserts += 1;
        Insert::Stored { release }
    }

    /// Everything the index believes about itself, checked: the page counts equal the sum over entries, the length table equals the entries,
    /// no entry exceeds a limit. For tests and debug assertions.
    pub fn check(&self) -> Result<(), String> {
        let mut pages: HashMap<PageId, u32> = HashMap::new();
        let mut lens: BTreeMap<(u32, usize), usize> = BTreeMap::new();
        for ((path, _), e) in &self.entries {
            let mut d = e.handle.pages.clone();
            d.sort_unstable();
            d.dedup();
            for p in d {
                *pages.entry(p).or_insert(0) += 1;
            }
            *lens.entry((*path, e.len)).or_insert(0) += 1;
            if e.handle.pages.len() != e.len.div_ceil(BLOCK_TOKENS) {
                return Err(format!(
                    "entry of {} tokens holds {} pages, expected {}",
                    e.len,
                    e.handle.pages.len(),
                    e.len.div_ceil(BLOCK_TOKENS)
                ));
            }
        }
        if pages != self.page_refs {
            return Err("page reference counts differ from the entries".into());
        }
        if lens != self.lengths {
            return Err("length table differs from the entries".into());
        }
        if self.page_refs.len() > self.limits.page_blocks {
            return Err(format!("{} pages held, limit {}", self.page_refs.len(), self.limits.page_blocks));
        }
        let mut snaps: Vec<u32> = self.entries.values().map(|e| e.handle.snap).collect();
        snaps.sort_unstable();
        let n = snaps.len();
        snaps.dedup();
        if snaps.len() != n {
            return Err("two entries hold one snapshot slot".into());
        }
        if n > self.limits.snapshot_slots {
            return Err(format!("{n} snapshots held, limit {}", self.limits.snapshot_slots));
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------------------------------------------- ghost blocks

/// Hashes of the 256-token blocks of recent prompts: how many leading blocks of a new prompt some earlier prompt also had says where a shared
/// prefix (a system prompt) ends, to block resolution, without storing any state.
pub struct Ghost {
    namespace: u64,
    seen: HashMap<u64, u64>,
    order: VecDeque<(u64, u64)>,
    cap: usize,
    clock: u64,
}

impl Ghost {
    pub fn new(namespace: u64, cap: usize) -> Self {
        Self { namespace, seen: HashMap::new(), order: VecDeque::new(), cap: cap.max(1), clock: 0 }
    }
    pub fn len(&self) -> usize {
        self.seen.len()
    }
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }

    /// Record the blocks of `tokens` (every full block) and return how many LEADING blocks were already known before this call.
    pub fn observe(&mut self, path: u32, tokens: &[u32]) -> usize {
        self.clock += 1;
        let mut c = Chain::new(self.namespace, path);
        let mut known = 0usize;
        let mut run = true;
        for (i, &t) in tokens.iter().enumerate() {
            c.push(t);
            if (i + 1) % BLOCK_TOKENS == 0 {
                let h = c.value();
                if run && self.seen.contains_key(&h) {
                    known += 1;
                } else {
                    run = false;
                }
                self.seen.insert(h, self.clock);
                self.order.push_back((h, self.clock));
            }
        }
        // bounded: forget the oldest sightings (an entry of `order` whose hash was seen again later is stale and only counts as a refresh)
        while self.seen.len() > self.cap {
            match self.order.pop_front() {
                Some((h, at)) => {
                    if self.seen.get(&h) == Some(&at) {
                        self.seen.remove(&h);
                    }
                }
                None => break,
            }
        }
        while self.order.len() > 4 * self.cap {
            self.order.pop_front();
        }
        known
    }
}

// ----------------------------------------------------------------------------------------------------------------- cut policy

/// Where snapshots are taken (see `docs/PREFIX_CACHE.md`, "Cut points").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CutPolicy {
    /// No cut below this many tokens: a snapshot costs 187 MiB and saves `min_cut` tokens of prefill at best.
    pub min_cut: usize,
    /// The end cut is `total - tail`: the generation prompt a chat's next turn renders differently stays outside the entry. 0 cuts at the
    /// prompt's end (a template whose next turn repeats this one's prompt token for token): the whole prompt is reused by the next turn and
    /// the cut costs no extra slice, but a request that differs from this one only after its last token is the only one that can hit it.
    pub tail: usize,
    /// The shared cut must save at least this many more tokens than the hit the request already got.
    pub ghost_min_gain: usize,
    /// A cut is skipped when it lies within this many tokens of the state the prefill already holds at that point (the hit it resumed from, or the cut
    /// planned before it: the earlier wins). A snapshot a few tokens past another saves a few tokens of prefill for 187 MiB.
    pub min_gap: usize,
}

impl Default for CutPolicy {
    fn default() -> Self {
        Self { min_cut: 512, tail: 16, ghost_min_gain: 1024, min_gap: 256 }
    }
}

impl CutPolicy {
    /// The policy for a family served to chat clients and agent loops (Qwen3.8): prompts of a few hundred tokens that grow by 50 to 100 tokens a
    /// turn, and long documents asked about with a question of 10 to 60 tokens that differs from request to request. With [`CutPolicy::default`]
    /// (`min_cut` 512, `tail` 16, `min_gap` 256) the end cut of a cold document prompt falls inside the question (the next question diverges
    /// before it: no hit until a second sighting stores the shared cut), a tool loop under 528 tokens has no cut at all, and one that grows
    /// by under 256 tokens a turn advances its entry once in four turns (`cached_tokens` 0, 0, 0, 0, 0, 523, 523, ..., 808). The end cut
    /// here stays 64 tokens clear of the end (the generation prompt is 9 tokens, a question up to about 50 tokens still lands after the cut),
    /// a prompt of 192 tokens is cut, and an entry advances once a turn has added 128. The price: a snapshot slot (of 26 on a 128 GB node)
    /// per 128 tokens of a growing conversation instead of per 256, and up to 64 more tokens of prefill on a resumed turn; the eviction
    /// order (least recently used, a prefix hit twice last) is unchanged. DeepSeek V4.1 sets its own values; every other family keeps the
    /// default.
    pub const fn interactive() -> Self {
        Self { min_cut: 128, tail: 64, ghost_min_gain: 256, min_gap: 128 }
    }
}

/// Move `tokens` down to the nearest multiple of `align` (1 or 0: unchanged). A cached prefix that
/// ends on a derived-structure boundary (a KV page, or a coarser grid) never restores a partial page,
/// so no copy-on-write page can carry a foreign continuation into the resumed slot.
pub fn align_down(tokens: usize, align: usize) -> usize {
    if align <= 1 {
        tokens
    } else {
        tokens - tokens % align
    }
}

/// The cuts to snapshot while prefilling a prompt of `total` tokens that resumed at `hit` (0: cold), ascending. `ghost_blocks` is the number of
/// leading blocks an earlier prompt shared (`Ghost::observe`); `have(len)` says whether an entry for exactly `tokens[..len]` is already stored.
/// A cut is only ever planned above `hit` and at most `total` (`total` itself only with a policy of no tail).
pub fn plan_cuts(policy: &CutPolicy, total: usize, hit: usize, ghost_blocks: usize, have: &dyn Fn(usize) -> bool) -> Vec<usize> {
    plan_cuts_with(policy, total, hit, ghost_blocks, have, &Some)
}

/// [`plan_cuts`] where every proposed cut first goes through `fit`: the place it may really be taken at, or `None` when it may not be taken at all (the
/// engine moves a cut that falls inside an image to the image's start and drops it when that is not worth a snapshot). `fit` runs BEFORE the gap rule,
/// so a cut that is moved or dropped neither counts as the state the prefill holds nor suppresses the cut after it: a second prompt about the same image
/// has its block-aligned shared prefix inside the image, which is dropped, and its end cut a few hundred tokens on must still be taken.
pub fn plan_cuts_with(
    policy: &CutPolicy,
    total: usize,
    hit: usize,
    ghost_blocks: usize,
    have: &dyn Fn(usize) -> bool,
    fit: &dyn Fn(usize) -> Option<usize>,
) -> Vec<usize> {
    let mut cuts = Vec::new();
    let end = total.saturating_sub(policy.tail);
    let shared = ghost_blocks * BLOCK_TOKENS;
    // `prev` is the last state the prefill holds before the next cut: the hit, then each cut taken
    let mut prev = hit;
    if shared < end && shared < total {
        if let Some(c) = fit(shared) {
            if c >= policy.min_cut && c >= hit + policy.ghost_min_gain && !have(c) && c >= prev + policy.min_gap {
                cuts.push(c);
                prev = c;
            }
        }
    }
    if let Some(c) = fit(end) {
        if c >= policy.min_cut && c > hit && c <= total && !have(c) && c >= prev + policy.min_gap && (cuts.is_empty() || c > prev)
        {
            cuts.push(c);
        }
    }
    cuts
}

// ----------------------------------------------------------------------------------------------------------------- budget

/// The cache's physical size, derived from the memory the plan gives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheBudget {
    pub snapshot_slots: usize,
    pub page_blocks: usize,
    pub snapshot_bytes: u64,
    pub page_bytes: u64,
}

/// Pages of KV an entry is assumed to hold on average when the budget is split between snapshot slots and pages.
pub const ASSUMED_PAGES_PER_ENTRY: u64 = 8;

/// Snapshot slots a window-sized cache always keeps ([`CacheBudget::fit_pages_first`]): one entry evicts itself on every insert, so two is the
/// least that is a cache. One named constant, so the floor is easy to change.
pub const MIN_CACHE_SNAPSHOT_SLOTS: u64 = 2;

/// The least reach, in tokens, that repays a snapshot: under it a window-sized cache is disabled with a named reason.
pub const MIN_CACHE_REACH_TOKENS: u64 = 4096;

/// Snapshot slots wanted per live sequence slot, and the fewest wanted whatever the slot count (`N = max(8, 4 x S)`): a live conversation keeps
/// an entry or two, and the cache needs room beyond them.
pub const WANTED_SLOTS_PER_LIVE_SLOT: u64 = 4;
pub const MIN_WANTED_SLOTS: u64 = 8;

/// The working set beside the one full-window entry: every live slot may keep an entry of this fraction of the window (`ceil(C / 8)` pages).
pub const WORKING_SET_WINDOW_FRACTION: u64 = 8;

/// Under the target, snapshot slots take at most this share of the budget (`1 / 4`) so that pages come first.
pub const LOW_BUDGET_SLOT_SHARE_DENOMINATOR: u64 = 4;

/// The cache's margin under the planner's whole-GiB quantisation: `MemAvailable` crossing a whole-GiB boundary between two boots moves the spare
/// memory by 1 GiB (seen on GLM and RED-SNOW, 16 to 15 GiB), and the cache must not be sized right at the edge of what fits. See
/// [`cache_margin_bytes`].
pub const CACHE_MARGIN_BYTES: u64 = 2 << 30;

/// What an `Auto` cache leaves unspent of `spare_bytes`: [`CACHE_MARGIN_BYTES`], at most a quarter of the spare memory so that a node with
/// little spare is not rounded down to nothing.
pub fn cache_margin_bytes(spare_bytes: u64) -> u64 {
    CACHE_MARGIN_BYTES.min(spare_bytes / 4)
}

/// What a window-sized cache is cut for: the longest prompt the server accepts and the sequence slots that may be live at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowRule {
    pub window_tokens: u64,
    pub live_slots: usize,
}

impl WindowRule {
    fn min_pages() -> u64 {
        MIN_CACHE_REACH_TOKENS.div_ceil(BLOCK_TOKENS as u64)
    }
    /// `C`: the pages one full-window entry needs (at least the pages of [`MIN_CACHE_REACH_TOKENS`]).
    pub fn full_entry_pages(&self) -> u64 {
        self.window_tokens.div_ceil(BLOCK_TOKENS as u64).max(Self::min_pages())
    }
    /// `N = max(8, 4 x S)`: the snapshot slots the target wants.
    pub fn wanted_slots(&self) -> u64 {
        MIN_WANTED_SLOTS.max(WANTED_SLOTS_PER_LIVE_SLOT.saturating_mul(self.live_slots as u64))
    }
    /// `C + S x ceil(C / 8)`: the full-window entry and a working set of fractional entries, one per live slot.
    pub fn target_pages(&self) -> u64 {
        let c = self.full_entry_pages();
        c.saturating_add((self.live_slots as u64).saturating_mul(c.div_ceil(WORKING_SET_WINDOW_FRACTION)))
    }
}

impl CacheBudget {
    pub const NONE: CacheBudget = CacheBudget { snapshot_slots: 0, page_blocks: 0, snapshot_bytes: 0, page_bytes: 0 };

    /// The budget that fits `bytes`: whole "entry units" of one snapshot slot plus [`ASSUMED_PAGES_PER_ENTRY`] pages. None (disabled) when
    /// not even two units fit: a one-entry cache evicts itself on every insert.
    pub fn fit(bytes: u64, snapshot_bytes: u64, page_bytes: u64) -> CacheBudget {
        let unit = snapshot_bytes.saturating_add(ASSUMED_PAGES_PER_ENTRY.saturating_mul(page_bytes));
        let units = if unit == 0 { 0 } else { bytes / unit };
        if units < 2 {
            return CacheBudget { snapshot_bytes, page_bytes, ..CacheBudget::NONE };
        }
        CacheBudget {
            snapshot_slots: units as usize,
            page_blocks: (units * ASSUMED_PAGES_PER_ENTRY) as usize,
            snapshot_bytes,
            page_bytes,
        }
    }

    /// The budget that fits `bytes` cut by the window rule (`docs/PREFIX_CACHE_SIZING.md` section 5), pages first; [`CacheBudget::fit`] stays the
    /// 1:8 cut for every other caller. With `C` the pages of one full-window entry, `N` the wanted slots and `T` the target (see [`WindowRule`]),
    /// and `F` = `C` pages plus [`MIN_CACHE_SNAPSHOT_SLOTS`] slots:
    ///
    /// - `bytes >= T`: exactly `T` (`N` slots, `T` pages). With `spend_remainder` (an explicit size: the operator asked for those bytes) the rest
    ///   is spent too, in the 1:8 units of [`CacheBudget::fit`], and what is left after the units goes to pages.
    /// - `F <= bytes < T`: pages first. The slots are `max(2, min(bytes / 4 / slot, (bytes - C pages) / slot, N))` and every page that is left
    ///   (never fewer than `C`).
    /// - `bytes < F`: the minimum slots and every page that fits; the reach (`page_blocks x BLOCK_TOKENS`) is then below the window, which the
    ///   boot says.
    /// - Disabled (`NONE`) only when the minimum slots do not fit, or the pages left reach fewer than [`MIN_CACHE_REACH_TOKENS`] tokens: a state
    ///   that does not repay its snapshot. [`CacheBudget::pages_first_minimum_bytes`] is the size that avoids it.
    pub fn fit_pages_first(
        bytes: u64,
        snapshot_bytes: u64,
        page_bytes: u64,
        rule: WindowRule,
        spend_remainder: bool,
    ) -> CacheBudget {
        let none = CacheBudget { snapshot_bytes, page_bytes, ..CacheBudget::NONE };
        let made = |slots: u64, pages: u64| CacheBudget {
            snapshot_slots: slots as usize,
            page_blocks: pages as usize,
            snapshot_bytes,
            page_bytes,
        };
        if page_bytes == 0 || snapshot_bytes == 0 {
            return none;
        }
        let Some(room) = bytes.checked_sub(snapshot_bytes.saturating_mul(MIN_CACHE_SNAPSHOT_SLOTS)) else {
            return none;
        };
        let pages_in_room = room / page_bytes;
        if pages_in_room < WindowRule::min_pages() {
            return none;
        }
        let (full, target_pages, wanted) = (rule.full_entry_pages(), rule.target_pages(), rule.wanted_slots());
        let target = target_pages.saturating_mul(page_bytes).saturating_add(wanted.saturating_mul(snapshot_bytes));
        if bytes >= target {
            if !spend_remainder {
                return made(wanted, target_pages);
            }
            let rest = bytes - target;
            let unit = snapshot_bytes.saturating_add(ASSUMED_PAGES_PER_ENTRY.saturating_mul(page_bytes));
            let units = rest / unit;
            let extra_pages = (rest - units * unit) / page_bytes;
            return made(wanted + units, target_pages + units * ASSUMED_PAGES_PER_ENTRY + extra_pages);
        }
        let floor = full.saturating_mul(page_bytes).saturating_add(snapshot_bytes.saturating_mul(MIN_CACHE_SNAPSHOT_SLOTS));
        if bytes >= floor {
            let by_share = bytes / LOW_BUDGET_SLOT_SHARE_DENOMINATOR / snapshot_bytes;
            let by_room = (bytes - full * page_bytes) / snapshot_bytes;
            let slots = by_share.min(by_room).min(wanted).max(MIN_CACHE_SNAPSHOT_SLOTS);
            return made(slots, (bytes - slots * snapshot_bytes) / page_bytes);
        }
        made(MIN_CACHE_SNAPSHOT_SLOTS, pages_in_room)
    }

    /// The least [`CacheBudget::fit_pages_first`] enables: the minimum slots and the pages of [`MIN_CACHE_REACH_TOKENS`] tokens.
    pub fn pages_first_minimum_bytes(snapshot_bytes: u64, page_bytes: u64) -> u64 {
        let pages = MIN_CACHE_REACH_TOKENS.div_ceil(BLOCK_TOKENS as u64);
        snapshot_bytes.saturating_mul(MIN_CACHE_SNAPSHOT_SLOTS).saturating_add(page_bytes.saturating_mul(pages))
    }

    /// The longest prefix the cache can ever reuse: a prefix of `L` tokens needs `ceil(L / BLOCK_TOKENS)` of its pages.
    pub fn reach_tokens(&self) -> u64 {
        self.page_blocks as u64 * BLOCK_TOKENS as u64
    }

    /// Bytes the backend allocates for it: the snapshot pool and the extra KV pages.
    pub fn bytes(&self) -> u64 {
        self.snapshot_slots as u64 * self.snapshot_bytes + self.page_blocks as u64 * self.page_bytes
    }
    pub fn enabled(&self) -> bool {
        self.snapshot_slots > 0
    }
    pub fn limits(&self) -> Limits {
        Limits { snapshot_slots: self.snapshot_slots, page_blocks: self.page_blocks }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cut_that_is_moved_or_dropped_does_not_suppress_the_cut_after_it() {
        let p = CutPolicy { min_cut: 64, tail: 8, ghost_min_gain: 256, min_gap: 256 };
        let none = |_: usize| false;
        // a shared prefix of 512 tokens inside an image that starts at 40 (below the minimum cut: dropped) and an end cut at 752, 240 tokens past the shared one
        let in_image = |c: usize| if c > 40 && c < 716 { None } else { Some(c) };
        assert_eq!(
            plan_cuts(&p, 760, 0, 2, &none),
            vec![512],
            "without the image rule the shared cut is taken and the end cut is within the gap"
        );
        assert_eq!(
            plan_cuts_with(&p, 760, 0, 2, &none, &in_image),
            vec![752],
            "dropped: it does not count as a state the prefill holds"
        );
        // moved to the image's start (above the minimum cut) it is a cut of its own, and the end cut follows it when it is far enough
        let moved = |c: usize| if c > 300 && c < 716 { Some(300) } else { Some(c) };
        assert_eq!(plan_cuts_with(&p, 760, 0, 2, &none, &moved), vec![300, 752]);
        // two cuts that land on the same place are one cut
        let all_to_300 = |c: usize| if c > 300 { Some(300) } else { Some(c) };
        assert_eq!(plan_cuts_with(&CutPolicy { min_gap: 0, ..p }, 760, 0, 2, &none, &all_to_300), vec![300]);
        // the identity `fit` is `plan_cuts`
        for (total, hit, ghost) in
            [(3400usize, 0usize, 0usize), (3400, 0, 13), (4200, 0, 13), (3400, 3328, 13), (3900, 3328, 13), (760, 0, 2)]
        {
            assert_eq!(plan_cuts_with(&p, total, hit, ghost, &none, &Some), plan_cuts(&p, total, hit, ghost, &none));
        }
    }

    const NS: u64 = 0xABCD;
    const LIM: Limits = Limits { snapshot_slots: 8, page_blocks: 64 };

    fn toks(n: usize, salt: u32) -> Vec<u32> {
        (0..n as u32).map(|i| i.wrapping_mul(2654435761).wrapping_add(salt) % 50_000).collect()
    }
    fn pages_for(len: usize, first: u32) -> Vec<PageId> {
        (0..len.div_ceil(BLOCK_TOKENS) as u32).map(|i| first + i).collect()
    }
    fn cap(len: usize, snap: u32, first_page: u32) -> Captured {
        Captured { len, handle: PrefixHandle { snap, pages: pages_for(len, first_page) } }
    }
    fn arc(t: &[u32]) -> Arc<Vec<u32>> {
        Arc::new(t.to_vec())
    }

    fn size_aware(limits: Limits) -> PrefixIndex {
        let mut ix = PrefixIndex::new(NS, limits);
        ix.set_eviction(Eviction::SizeAware { overhead_pages: 8 });
        ix
    }

    /// The review's case: one entry holds every page of the cache. Under LRU the next small insert evicts it; under the size-aware policy a
    /// small insert is refused instead, a bounded number of times (the credit ratio), after which the large entry goes; a hit on the large
    /// entry in between renews its credit (doubled by the hit count).
    #[test]
    fn a_large_entry_is_not_evicted_by_one_small_insert_and_does_not_starve_small_ones() {
        let lim = Limits { snapshot_slots: 8, page_blocks: 64 };
        let big = toks(64 * BLOCK_TOKENS, 1);
        // LRU, the default: the first small insert that needs a page evicts the 64-page entry
        let mut lru = PrefixIndex::new(NS, lim);
        assert!(lru.insert(1, arc(&big), cap(big.len(), 0, 0)).stored());
        let s = toks(512, 2);
        let r = lru.insert(1, arc(&s), cap(512, 1, 100));
        assert!(r.stored() && r.into_release().len() == 1 && lru.lookup(1, &big).is_none(), "LRU evicts the large entry");
        // size-aware: d_big = 16,384 / (64 + 8) = 227.6, d_small = 512 / (2 + 8) = 51.2, so four refusals raise the inflation to 204.8
        // and the fifth small insert (credit 256) outbids the large entry
        let run = |hit_after: Option<usize>| -> (usize, PrefixIndex) {
            let mut ix = size_aware(lim);
            assert!(ix.insert(1, arc(&big), cap(big.len(), 0, 0)).stored());
            for i in 0..100usize {
                if hit_after == Some(i) {
                    let h = ix.lookup(1, &[big.clone(), vec![9]].concat()).expect("the large entry is still there");
                    ix.unpin(h.key);
                }
                let s = toks(512, 10 + i as u32);
                match ix.insert(1, arc(&s), cap(512, 1, 1000 + 2 * i as u32)) {
                    Insert::Stored { release } => {
                        assert_eq!(release.len(), 1, "the large entry is the one victim");
                        ix.check().unwrap();
                        return (i, ix);
                    }
                    Insert::NoRoom { release } => assert_eq!(release.len(), 1, "only the capture comes back"),
                    Insert::Duplicate { .. } => unreachable!(),
                }
            }
            panic!("a small entry was starved");
        };
        let (refused, ix) = run(None);
        assert_eq!(refused, 4);
        assert_eq!((ix.stats.refused_by_value, ix.stats.no_room, ix.stats.evicted_for_pages), (4, 4, 1));
        // a hit after two refusals renews the large entry at the inflation then (102.4) with twice its density: 102.4 + 455.1 = 557.5,
        // which a small insert outbids once the inflation is 51.2 x 10 = 512: ten refusals in all instead of four
        let (refused, _) = run(Some(2));
        assert_eq!(refused, 10);
    }

    /// Small entries churn among themselves while a large one is live: when a snapshot slot is needed the small entry of least credit (the
    /// oldest among equals) goes, not the large one, and pages the large entry does not hold are used freely.
    #[test]
    fn small_entries_turn_over_beside_a_live_large_one() {
        let mut ix = size_aware(Limits { snapshot_slots: 4, page_blocks: 80 });
        let big = toks(64 * BLOCK_TOKENS, 1);
        assert!(ix.insert(1, arc(&big), cap(big.len(), 0, 0)).stored());
        let smalls: Vec<Vec<u32>> = (0..6).map(|i| toks(512, 20 + i)).collect();
        for (i, s) in smalls.iter().enumerate() {
            let r = ix.insert(1, arc(s), cap(512, 1 + i as u32, 200 + 2 * i as u32));
            assert!(r.stored(), "small {i}");
            ix.check().unwrap();
        }
        // four slots: the large entry and the three newest small ones
        assert_eq!(ix.len(), 4);
        assert!(ix.contains(1, &big, big.len()));
        for (i, s) in smalls.iter().enumerate() {
            assert_eq!(ix.contains(1, s, 512), i >= 3, "small {i}");
        }
        assert_eq!(ix.stats.evicted_for_slots, 3);
    }
    #[test]
    fn peek_finds_what_lookup_finds_and_changes_nothing() {
        let mut ix = PrefixIndex::new(NS, LIM);
        let t = toks(1000, 1);
        for (i, len) in [300usize, 600].into_iter().enumerate() {
            assert!(ix.insert(1, arc(&t), cap(len, i as u32, 10 * i as u32)).stored());
        }
        let clock = ix.clock;
        assert_eq!(ix.peek(1, &t), Some(600));
        assert_eq!(ix.peek(1, &t[..600]), Some(300), "strict prefix only");
        assert_eq!(ix.peek(1, &toks(1000, 2)), None);
        assert_eq!(ix.peek(2, &t), None, "another path");
        assert_eq!(ix.clock, clock);
        assert!(ix.entries.values().all(|e| e.pins == 0 && e.hits == 0));
        assert_eq!(ix.lookup(1, &t).map(|h| h.len), Some(600));
    }

    #[test]
    fn longest_strict_prefix_wins_and_the_whole_prompt_never_hits() {
        let mut ix = PrefixIndex::new(NS, LIM);
        let t = toks(1000, 1);
        for (i, len) in [300usize, 600, 1000].into_iter().enumerate() {
            assert!(ix.insert(1, arc(&t), cap(len, i as u32, 10 * i as u32)).stored());
        }
        let h = ix.lookup(1, &t).unwrap();
        assert_eq!(h.len, 600, "an entry as long as the request leaves nothing to prefill: not a hit");
        ix.unpin(h.key);
        let h = ix.lookup(1, &t[..601]).unwrap();
        assert_eq!(h.len, 600);
        ix.unpin(h.key);
        let h = ix.lookup(1, &t[..600]).unwrap();
        assert_eq!(h.len, 300, "a 600-token request cannot use the 600-token entry");
        ix.unpin(h.key);
        assert!(ix.lookup(1, &t[..300]).is_none());
        ix.check().unwrap();
    }

    #[test]
    fn a_different_token_anywhere_in_the_prefix_misses() {
        let mut ix = PrefixIndex::new(NS, LIM);
        let t = toks(700, 2);
        assert!(ix.insert(1, arc(&t), cap(600, 0, 0)).stored());
        for pos in [0usize, 1, 255, 256, 257, 511, 512, 599] {
            let mut u = t.clone();
            u[pos] ^= 1;
            assert!(ix.lookup(1, &u).is_none(), "differs at {pos}");
        }
        let mut u = t.clone();
        u[600] ^= 1; // past the entry: still a hit
        let h = ix.lookup(1, &u).unwrap();
        assert_eq!(h.len, 600);
    }

    #[test]
    fn path_and_namespace_separate_entries() {
        let mut ix = PrefixIndex::new(NS, LIM);
        let t = toks(700, 3);
        assert!(ix.insert(1, arc(&t), cap(600, 0, 0)).stored());
        assert!(ix.lookup(0, &t).is_none(), "another prefill path never sees the entry");
        assert!(ix.lookup(2, &t).is_none());
        assert!(!ix.contains(0, &t, 600) && ix.contains(1, &t, 600));
        // the same tokens under the other path are a different entry, not a duplicate
        assert!(ix.insert(0, arc(&t), cap(600, 1, 100)).stored());
        // a different namespace hashes differently: its keys never equal ours
        let other = PrefixIndex::new(NS + 1, LIM);
        assert_ne!(ix.key_of(1, &t, 600), other.key_of(1, &t, 600));
        assert_ne!(ix.key_of(1, &t, 600), ix.key_of(0, &t, 600));
        ix.check().unwrap();
    }

    #[test]
    fn a_hash_collision_is_caught_by_the_token_comparison() {
        // 4 hash bits: two different prompts of the same length collide at once
        let mut ix = PrefixIndex::with_mask(NS, LIM, 0xF);
        let a = toks(400, 4);
        let mut found = None;
        for salt in 5..200 {
            let b = toks(400, salt);
            if ix.key_of(1, &a, 300) == ix.key_of(1, &b, 300) && a[..300] != b[..300] {
                found = Some(b);
                break;
            }
        }
        let b = found.expect("a collision within 200 tries at 4 bits");
        assert!(ix.insert(1, arc(&a), cap(300, 0, 0)).stored());
        assert!(ix.lookup(1, &b).is_none(), "equal hash, different tokens: not a hit");
        assert!(!ix.contains(1, &b, 300));
        // inserting the colliding prefix is refused (the key is taken) and handed back, never overwriting the first
        match ix.insert(1, arc(&b), cap(300, 1, 50)) {
            Insert::Duplicate { release } => assert_eq!(release[0].snap, 1),
            o => panic!("{o:?}"),
        }
        let h = ix.lookup(1, &a).unwrap();
        assert_eq!(h.handle.snap, 0);
    }

    #[test]
    fn lru_evicts_the_least_recently_used_and_a_lookup_refreshes() {
        let lim = Limits { snapshot_slots: 8, page_blocks: 3 };
        let mut ix = PrefixIndex::new(NS, lim);
        let (a, b, c) = (toks(300, 10), toks(300, 11), toks(300, 12));
        assert!(ix.insert(1, arc(&a), cap(260, 0, 0)).stored()); // 2 pages: 0, 1
                                                                 // b needs 2 more pages: 3 held at most, so a goes
        match ix.insert(1, arc(&b), cap(260, 1, 10)) {
            Insert::Stored { release } => assert_eq!(release.iter().map(|h| h.snap).collect::<Vec<_>>(), vec![0]),
            o => panic!("{o:?}"),
        }
        assert!(ix.lookup(1, &a).is_none());
        let hb = ix.lookup(1, &b).unwrap();
        ix.unpin(hb.key);
        // c evicts b (the only one), then a fresh d evicts c unless c was used
        match ix.insert(1, arc(&c), cap(260, 2, 20)) {
            Insert::Stored { release } => assert_eq!(release.iter().map(|h| h.snap).collect::<Vec<_>>(), vec![1]),
            o => panic!("{o:?}"),
        }
        ix.check().unwrap();
    }

    #[test]
    fn lru_order_follows_use_with_several_entries() {
        let lim = Limits { snapshot_slots: 8, page_blocks: 3 };
        let mut ix = PrefixIndex::new(NS, lim);
        let ts: Vec<Vec<u32>> = (0..3).map(|i| toks(300, 20 + i)).collect();
        for (i, t) in ts.iter().enumerate() {
            assert!(ix.insert(1, arc(t), cap(100, i as u32, i as u32)).stored());
            // one page each
        }
        // use entry 0, so 1 is now the least recently used
        let h = ix.lookup(1, &ts[0]).unwrap();
        ix.unpin(h.key);
        let d = toks(300, 30);
        match ix.insert(1, arc(&d), cap(100, 3, 3)) {
            Insert::Stored { release } => assert_eq!(release.iter().map(|h| h.snap).collect::<Vec<_>>(), vec![1]),
            o => panic!("{o:?}"),
        }
        assert!(ix.lookup(1, &ts[1]).is_none());
        for i in [0usize, 2] {
            let h = ix.lookup(1, &ts[i]).unwrap_or_else(|| panic!("entry {i} was evicted"));
            ix.unpin(h.key);
        }
        ix.check().unwrap();
    }

    #[test]
    fn pages_shared_between_entries_are_counted_once_and_freed_with_the_last_holder() {
        let mut ix = PrefixIndex::new(NS, LIM);
        let t = toks(1000, 40);
        // two entries of one prompt: 600 tokens (3 pages: 0,1,2) and 1000 tokens (4 pages: 0,1,2,3): they share pages 0..3
        assert!(ix.insert(1, arc(&t), cap(600, 0, 0)).stored());
        assert!(ix.insert(1, arc(&t), cap(1000, 1, 0)).stored());
        assert_eq!(ix.pages_held(), 4);
        assert_eq!((ix.page_holders(0), ix.page_holders(3)), (2, 1));
        // evicting the older entry leaves the shared pages held by the other
        let h = ix.evict_lru().unwrap();
        assert_eq!(h.snap, 0);
        assert_eq!((ix.pages_held(), ix.page_holders(0)), (4, 1));
        let h = ix.evict_lru().unwrap();
        assert_eq!(h.snap, 1);
        assert_eq!(ix.pages_held(), 0);
        ix.check().unwrap();
    }

    #[test]
    fn pages_two_entries_share_are_charged_to_the_budget_once() {
        // a budget of exactly four pages holds an entry of three pages and one of four that starts with them: 4 distinct pages, not 7
        let mut ix = PrefixIndex::new(NS, Limits { snapshot_slots: 8, page_blocks: 4 });
        let t = toks(1000, 45);
        assert_eq!(ix.insert(1, arc(&t), cap(600, 0, 0)), Insert::Stored { release: vec![] });
        assert_eq!(
            ix.insert(1, arc(&t), cap(1000, 1, 0)),
            Insert::Stored { release: vec![] },
            "the shared pages were counted again"
        );
        assert_eq!((ix.len(), ix.pages_held()), (2, 4));
        // a third entry on other pages does not fit: the least recently used goes
        let u = toks(300, 46);
        match ix.insert(1, arc(&u), cap(100, 2, 50)) {
            Insert::Stored { release } => {
                assert_eq!(release.len(), 2, "both entries hold the shared pages: evicting one frees none of them")
            }
            o => panic!("{o:?}"),
        }
        ix.check().unwrap();
    }

    #[test]
    fn a_pinned_entry_is_never_evicted() {
        let lim = Limits { snapshot_slots: 2, page_blocks: 2 };
        let mut ix = PrefixIndex::new(NS, lim);
        let (a, b) = (toks(300, 50), toks(300, 51));
        assert!(ix.insert(1, arc(&a), cap(100, 0, 0)).stored());
        assert!(ix.insert(1, arc(&b), cap(100, 1, 1)).stored());
        let h = ix.lookup(1, &a).unwrap(); // pins a
                                           // everything is held and a is pinned: eviction takes b, never a
        assert_eq!(ix.evict_lru().unwrap().snap, 1);
        assert!(ix.evict_lru().is_none(), "only a pinned entry is left");
        // an insert that needs room beyond what the unpinned ones give is refused, nothing pinned is touched
        let c = toks(300, 52);
        match ix.insert(1, arc(&c), cap(300, 2, 5)) {
            Insert::NoRoom { release } => assert!(release.iter().any(|h| h.snap == 2)),
            o => panic!("{o:?}"),
        }
        assert!(ix.contains(1, &a, 100));
        ix.unpin(h.key);
        assert_eq!(ix.evict_lru().unwrap().snap, 0);
    }

    #[test]
    fn duplicates_and_oversized_captures_are_handed_back() {
        let lim = Limits { snapshot_slots: 4, page_blocks: 3 };
        let mut ix = PrefixIndex::new(NS, lim);
        let t = toks(2000, 60);
        assert!(ix.insert(1, arc(&t), cap(300, 0, 0)).stored());
        match ix.insert(1, arc(&t), cap(300, 1, 7)) {
            Insert::Duplicate { release } => assert_eq!(release, vec![PrefixHandle { snap: 1, pages: vec![7, 8] }]),
            o => panic!("{o:?}"),
        }
        // 1000 tokens need 4 pages: more than the cache may ever hold
        match ix.insert(1, arc(&t), cap(1000, 2, 20)) {
            Insert::NoRoom { release } => assert_eq!(release.len(), 1),
            o => panic!("{o:?}"),
        }
        assert_eq!(ix.len(), 1);
        assert_eq!(ix.stats.duplicates, 1);
        ix.check().unwrap();
    }

    #[test]
    fn evict_for_snapshots_frees_exactly_what_is_missing() {
        let mut ix = PrefixIndex::new(NS, LIM);
        for i in 0..4u32 {
            let t = toks(300, 70 + i);
            assert!(ix.insert(1, arc(&t), cap(100, i, i)).stored());
        }
        assert!(ix.evict_for_snapshots(2, 2).is_empty());
        assert_eq!(ix.evict_for_snapshots(3, 1).len(), 2);
        assert_eq!(ix.len(), 2);
        assert_eq!(ix.evict_for_snapshots(10, 0).len(), 2, "cannot free more than it holds");
        assert!(ix.is_empty());
    }

    #[test]
    fn random_operations_keep_the_books_balanced() {
        // a reference model: entries as (path, prompt id, len); compare presence and the page and snapshot accounting after every operation
        let mut r = 12345u64;
        let mut rnd = |m: u64| {
            r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (r >> 33) % m
        };
        let lim = Limits { snapshot_slots: 6, page_blocks: 12 };
        let mut ix = PrefixIndex::new(NS, lim);
        let prompts: Vec<Arc<Vec<u32>>> = (0..5).map(|i| arc(&toks(2000, 90 + i))).collect();
        let mut next_snap = 0u32;
        let mut live: HashMap<u32, PrefixHandle> = HashMap::new(); // handles the index holds or was handed back and not yet released
        for _ in 0..3000 {
            match rnd(4) {
                0 | 1 => {
                    let p = rnd(5) as usize;
                    let len = 100 + 100 * rnd(15) as usize;
                    let first = rnd(40) as u32;
                    let handle = PrefixHandle { snap: next_snap, pages: pages_for(len, first) };
                    next_snap += 1;
                    live.insert(handle.snap, handle.clone());
                    let out = ix.insert(1, prompts[p].clone(), Captured { len, handle });
                    for h in out.into_release() {
                        assert!(live.remove(&h.snap).is_some(), "released a handle twice or one never given");
                    }
                }
                2 => {
                    let p = rnd(5) as usize;
                    let n = 100 + rnd(1900) as usize;
                    if let Some(h) = ix.lookup(1, &prompts[p][..n]) {
                        assert!(h.len < n && live.contains_key(&h.handle.snap));
                        ix.unpin(h.key);
                    }
                }
                _ => {
                    if let Some(h) = ix.evict_lru() {
                        assert!(live.remove(&h.snap).is_some());
                    }
                }
            }
            ix.check().unwrap();
            assert_eq!(live.len(), ix.snapshots_held(), "every handle not handed back is held by exactly one entry");
        }
    }

    #[test]
    fn a_prefix_several_requests_share_outlives_a_flood_of_one_shot_entries() {
        let mut ix = PrefixIndex::new(NS, Limits { snapshot_slots: 4, page_blocks: 64 });
        let shared = toks(600, 300);
        assert!(ix.insert(1, arc(&shared), cap(300, 0, 0)).stored());
        for _ in 0..SHARED_HITS {
            let h = ix.lookup(1, &shared).unwrap(); // two requests resume from it: a shared prefix, and live
            ix.unpin(h.key);
        }
        // a stream of one-shot entries (never looked up again): each evicts another one-shot, never the shared prefix
        for i in 0..25u32 {
            let t = toks(300, 400 + i);
            assert!(ix.insert(1, arc(&t), cap(100, 10 + i, 10 + i)).stored());
            assert!(ix.contains(1, &shared, 300), "the shared prefix was evicted after {} one-shot entries", i + 1);
        }
        // but protection is for live entries: idle for 8 x 4 index operations it competes by age like the rest and goes
        for i in 25..80u32 {
            let t = toks(300, 400 + i);
            assert!(ix.insert(1, arc(&t), cap(100, 10 + i, 10 + i)).stored());
        }
        assert!(!ix.contains(1, &shared, 300), "a stale entry held the budget for ever");
        ix.check().unwrap();
    }

    #[test]
    fn one_hit_is_not_sharing_recency_decides_and_a_prefix_hit_twice_goes_last() {
        let mut ix = PrefixIndex::new(NS, Limits { snapshot_slots: 3, page_blocks: 64 });
        let (a, b, c) = (toks(300, 310), toks(300, 311), toks(300, 312));
        assert!(ix.insert(1, arc(&a), cap(100, 0, 0)).stored());
        assert!(ix.insert(1, arc(&b), cap(100, 1, 1)).stored());
        let h = ix.lookup(1, &b).unwrap(); // b: one hit (a conversation's next turn), older than c
        ix.unpin(h.key);
        assert!(ix.insert(1, arc(&c), cap(100, 2, 2)).stored()); // c: never hit
        for _ in 0..SHARED_HITS {
            let h = ix.lookup(1, &a).unwrap(); // a: the most recently used, hit twice
            ix.unpin(h.key);
        }
        // b was hit once, which proves nothing about the future: it is older than c and goes first (never-hit-first would take c); a goes last
        assert_eq!(ix.evict_lru().unwrap().snap, 1);
        assert_eq!(ix.evict_lru().unwrap().snap, 2);
        assert_eq!(ix.evict_lru().unwrap().snap, 0, "only then the shared one");
    }

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s >> 33
    }

    /// `chats` conversations in flight at a time (60 in all, six turns of 487 tokens each from a 1,152-token start, random order) through the index with
    /// `slots` snapshot slots and the real cut planner; returns (turns after the first, of them resumed from the previous turn's own end cut).
    fn conversations_through_the_index(slots: usize, chats: usize) -> (u64, u64) {
        let turns = 6usize;
        let policy = CutPolicy::default();
        let mut ix = PrefixIndex::new(NS, Limits { snapshot_slots: slots, page_blocks: 1_000_000 });
        let mut free: Vec<u32> = (0..slots as u32).collect();
        let stream = |id: usize| -> Vec<u32> {
            let mut s = 0x9E37_79B9_7F4A_7C15u64 ^ (id as u64 + 1).wrapping_mul(0xD6E8_FEB8_6659_FD93);
            (0..1152 + 487 * (turns - 1)).map(|_| (lcg(&mut s) % 50_000) as u32).collect()
        };
        let mut live: Vec<(usize, Vec<u32>)> = (0..chats).map(|c| (0, stream(c))).collect();
        let mut started = chats;
        let (mut seed, mut next_page, mut total, mut resumed) = (12345u64, 0u32, 0u64, 0u64);
        while !live.is_empty() {
            let i = lcg(&mut seed) as usize % live.len();
            let t = live[i].0;
            let len = 1152 + 487 * t;
            let prompt = Arc::new(live[i].1[..len].to_vec());
            let hit = ix.lookup(7, &prompt);
            let hit_len = hit.as_ref().map_or(0, |h| h.len);
            if let Some(h) = &hit {
                ix.unpin(h.key);
            }
            if t >= 1 {
                total += 1;
                resumed += (hit_len >= 1152 + 487 * (t - 1) - 16) as u64;
            }
            for c in plan_cuts(&policy, len, hit_len, 0, &|l| ix.contains(7, &prompt, l)) {
                if free.is_empty() {
                    free.extend(ix.evict_for_snapshots(1, 0).into_iter().map(|h| h.snap));
                }
                let snap = free.pop().unwrap();
                next_page += 1000;
                free.extend(ix.insert(7, prompt.clone(), cap(c, snap, next_page)).into_release().into_iter().map(|h| h.snap));
            }
            ix.check().unwrap();
            live[i].0 += 1;
            if live[i].0 == turns {
                if started < 60 {
                    live[i] = (0, stream(started));
                    started += 1;
                } else {
                    live.remove(i);
                }
            }
        }
        (total, resumed)
    }

    #[test]
    fn conversations_in_flight_resume_from_their_own_last_turn_under_pressure() {
        // 16 slots, and each conversation leaves one entry per turn: more entries than slots from the second turn on. The entry a chat resumes from
        // is the newest of its own, which has not hit yet; the ones it superseded have. Never-hit-first evicted the former and kept the latter: the
        // frontier hit 99 of 300 turns with 4 chats and 73 of 300 with 8; recency alone gets 297 and 246, and so does this order.
        let (total, resumed) = conversations_through_the_index(16, 4);
        eprintln!("4 chats on 16 slots: {resumed} of {total} turns resumed from their previous turn's entry");
        assert_eq!(total, 300);
        assert!(resumed >= 270, "4 chats on 16 slots: {resumed} of {total} turns resumed from their previous turn's entry");
        let (total, resumed) = conversations_through_the_index(16, 8);
        eprintln!("8 chats on 16 slots: {resumed} of {total}");
        assert!(resumed * 100 >= total * 75, "8 chats on 16 slots: {resumed} of {total}");
    }

    #[test]
    fn evictions_are_counted_by_cause() {
        let mut ix = PrefixIndex::new(NS, Limits { snapshot_slots: 8, page_blocks: 3 });
        let (a, b) = (toks(300, 320), toks(300, 321));
        assert!(ix.insert(1, arc(&a), cap(260, 0, 0)).stored()); // 2 pages
        assert!(ix.insert(1, arc(&b), cap(260, 1, 10)).stored()); // 2 more: the page budget evicts a
        assert_eq!((ix.stats.evictions, ix.stats.evicted_for_pages, ix.stats.evicted_for_slots), (1, 1, 0));
        assert_eq!(ix.evict_for_snapshots(1, 0).len(), 1); // a capture needs a slot: b goes
        assert_eq!((ix.stats.evictions, ix.stats.evicted_for_pages, ix.stats.evicted_for_slots), (2, 1, 1));
    }

    #[test]
    fn ghost_counts_leading_blocks_seen_before() {
        let mut g = Ghost::new(NS, 1000);
        let sys = toks(3 * 256 + 100, 100);
        assert_eq!(g.observe(1, &sys), 0);
        let mut b = sys.clone();
        b.extend(toks(500, 101));
        assert_eq!(g.observe(1, &b), 3, "three full blocks of the system prompt were seen");
        let mut c = toks(100, 102);
        c.extend_from_slice(&sys);
        assert_eq!(g.observe(1, &c), 0, "a shift by one token changes every block");
        assert_eq!(g.observe(2, &b), 0, "another path");
    }

    #[test]
    fn ghost_is_bounded() {
        let mut g = Ghost::new(NS, 10);
        for i in 0..100 {
            g.observe(1, &toks(1024, 200 + i));
            assert!(g.len() <= 10 + 4);
        }
    }

    #[test]
    fn cut_planning_rules() {
        let p = CutPolicy::default();
        let none = |_: usize| false;
        // a cold 3,400-token prompt, nothing shared yet: the end cut only
        assert_eq!(plan_cuts(&p, 3400, 0, 0, &none), vec![3384]);
        // a second prompt sharing 13 blocks: the shared cut, and the end cut 56 tokens behind it is skipped (min_gap 256: the earlier wins)
        assert_eq!(plan_cuts(&p, 3400, 0, 13, &none), vec![3328]);
        assert_eq!(plan_cuts(&CutPolicy { min_gap: 0, ..p }, 3400, 0, 13, &none), vec![3328, 3384]);
        // a longer prompt with the same shared blocks: both are far enough apart
        assert_eq!(plan_cuts(&p, 4200, 0, 13, &none), vec![3328, 4184]);
        // a request that hit 3,328 already covers the shared prefix: its own end cut is 56 tokens past the hit, not worth a snapshot
        assert_eq!(plan_cuts(&p, 3400, 3328, 13, &none), Vec::<usize>::new());
        assert_eq!(plan_cuts(&p, 3900, 3328, 13, &none), vec![3884]);
        // a conversation extending itself: the hit is 20 tokens short of the shared blocks, so no shared cut (gain < 1,024); the end cut is 380 past it
        assert_eq!(plan_cuts(&p, 3400, 3000, 13, &none), vec![3384]);
        // ... and a turn that adds fewer than 256 tokens does not advance the entry
        assert_eq!(plan_cuts(&p, 3150, 3000, 0, &none), Vec::<usize>::new());
        // below min_cut nothing is cut; a prompt of 129 tokens has no end cut (113 < 512)
        assert_eq!(plan_cuts(&p, 129, 0, 0, &none), Vec::<usize>::new());
        assert_eq!(plan_cuts(&p, 520, 0, 0, &none), Vec::<usize>::new());
        assert_eq!(plan_cuts(&p, 528, 0, 0, &none), vec![512]);
        // the same prompt again: the end cut is stored, nothing to take
        assert_eq!(plan_cuts(&p, 3400, 0, 0, &|l| l == 3384), Vec::<usize>::new());
        // a hit within the tail margin of the end: nothing above it to cut
        assert_eq!(plan_cuts(&p, 3400, 3384, 0, &none), Vec::<usize>::new());
        // cuts are strictly inside the prompt
        for total in [129usize, 600, 4096, 100_000] {
            for c in plan_cuts(&p, total, 0, 1000, &none) {
                assert!(c > 0 && c < total);
            }
        }
    }

    /// The cut planner, the ghost set and the index driven the way the engine drives them (`Engine::plan_prefix` then `prefix_commit`): the length
    /// each prompt resumes from. The captures are free of any state here, so only the policy is under test.
    fn replay(policy: &CutPolicy, prompts: &[Vec<u32>]) -> Vec<usize> {
        let mut ix = PrefixIndex::new(NS, Limits { snapshot_slots: 26, page_blocks: 208 });
        let mut ghost = Ghost::new(NS, 1 << 12);
        let (mut hits, mut snap, mut page) = (Vec::new(), 0u32, 0u32);
        for p in prompts {
            let hit = ix.lookup(1, p).map(|h| (h.len, h.key));
            if let Some((_, key)) = hit {
                ix.unpin(key);
            }
            let hit = hit.map_or(0, |h| h.0);
            hits.push(hit);
            let shared = ghost.observe(1, p);
            let cuts = plan_cuts(policy, p.len(), hit, shared, &|l| ix.contains(1, p, l));
            let prompt = arc(p);
            for c in cuts {
                // the engine hands back what `insert` refuses; a free slot is the engine's business, so make room as it does
                let _ = ix.insert(1, prompt.clone(), cap(c, snap, page));
                snap += 1;
                page += 64;
            }
        }
        hits
    }

    /// Two requests about one long document, each ending in its own question (the baseline gate's prefix-cache stage: 8,000 tokens, then 10 to 12
    /// question tokens and the 9-token generation prompt). The first sighting cannot know where the sharing stops, so its end cut must fall before
    /// the question: with the default tail (16) it lands inside the question and the second request resumes from nothing.
    #[test]
    fn a_second_question_about_a_document_resumes_from_the_first_sightings_cut() {
        let doc = toks(8003, 7);
        let ask = |q: u32| -> Vec<u32> {
            let mut p = doc.clone();
            p.extend(toks(2, 1)); // the words both questions start with
            p.extend((0..17).map(|i| 60_000 + q * 100 + i)); // the question's own words and the generation prompt
            p
        };
        let (first, second, third) = (ask(1), ask(2), ask(3));
        let old = replay(&CutPolicy::default(), &[first.clone(), second.clone(), third.clone()]);
        assert_eq!(
            old,
            vec![0, 0, 7936],
            "the default policy: the second question prefills the whole document; only the third request hits"
        );
        let new = replay(&CutPolicy::interactive(), &[first, second, third]);
        assert!(new[1] >= 4000 && new[1] <= 8003, "the second question resumes inside the document (got {new:?})");
        assert!(new[2] >= new[1], "and the third no later ({new:?})");
    }

    /// A tool loop (the baseline gate's long loop): the prompt is 311 tokens at turn 1 and grows 57 to 59 tokens a turn; each turn's prompt is the
    /// last one's tokens without its 9-token generation prompt, then the model's call, the tool's result and a new generation prompt.
    #[test]
    fn a_tool_loop_of_a_few_hundred_tokens_resumes_from_turn_two_and_never_goes_back() {
        let lens = [311usize, 368, 425, 482, 539, 596, 653, 710, 767, 824, 883, 942, 1001];
        let conversation = toks(1100, 3);
        let turns: Vec<Vec<u32>> = lens
            .iter()
            .enumerate()
            .map(|(k, &n)| {
                let mut p = conversation[..n - 9].to_vec();
                p.extend((0..9).map(|i| 70_000 + k as u32 * 10 + i)); // the generation prompt, rendered differently next turn
                p
            })
            .collect();
        let old = replay(&CutPolicy::default(), &turns);
        assert_eq!(
            old,
            vec![0, 0, 0, 0, 0, 523, 523, 523, 523, 523, 808, 808, 808],
            "the default policy: what the gate reported"
        );
        let new = replay(&CutPolicy::interactive(), &turns);
        assert!(new[1..].iter().all(|&c| c > 0), "cached from turn 2 on: {new:?}");
        assert!(new.windows(2).all(|w| w[0] <= w[1]), "never decreasing: {new:?}");
        assert!(*new.last().unwrap() + 300 >= 1001, "and advancing with the conversation, at most 300 tokens behind it: {new:?}");
    }

    #[test]
    fn budget_arithmetic_is_whole_units_and_never_over_the_bytes() {
        let (snap, page) = (195_887_104u64, 16u64 << 20);
        let unit = snap + 8 * page;
        for gib in [0u64, 1, 2, 4, 8, 16, 60] {
            let bytes = gib << 30;
            let b = CacheBudget::fit(bytes, snap, page);
            assert!(b.bytes() <= bytes, "{gib} GiB");
            assert_eq!(b.page_blocks, 8 * b.snapshot_slots);
            if b.enabled() {
                assert!(bytes - b.bytes() < unit, "uses every whole unit");
                assert!(b.snapshot_slots >= 2);
            }
        }
        assert!(!CacheBudget::fit(unit, snap, page).enabled(), "one unit is not a cache");
        assert!(CacheBudget::fit(2 * unit, snap, page).enabled());
        assert_eq!(CacheBudget::fit(8 << 30, snap, page).snapshot_slots, 8 * 1024 * 1024 * 1024 / unit as usize);
    }

    #[test]
    fn align_down_moves_a_cut_to_the_derived_structure_boundary() {
        assert_eq!(align_down(2067, BLOCK_TOKENS), 2048, "8x256+19 -> 8x256");
        assert_eq!(align_down(2048, BLOCK_TOKENS), 2048, "already aligned");
        assert_eq!(align_down(255, BLOCK_TOKENS), 0);
        assert_eq!(align_down(2067, 1), 2067, "1 (and 0) are no-ops");
        assert_eq!(align_down(2067, 0), 2067);
    }

    #[test]
    fn an_aligning_fit_keeps_every_planned_cut_on_the_page_grid() {
        // the shared cut is block-aligned by construction; the end cut is not
        let p = CutPolicy { min_cut: 64, tail: 8, ghost_min_gain: 0, min_gap: 0 };
        let none = |_: usize| false;
        let align = |c: usize| Some(align_down(c, BLOCK_TOKENS));
        // 2131 - 8 = 2123 -> 2048, and the shared cut at 2048 is kept as-is
        let cuts = plan_cuts_with(&p, 2131, 0, 8, &none, &align);
        assert_eq!(cuts, vec![2048]);
        assert!(cuts.iter().all(|&c| c % BLOCK_TOKENS == 0), "every cut page-aligned: {cuts:?}");
        // a prompt whose end cut aligns exactly onto the shared cut stores one entry
        let cuts = plan_cuts_with(&p, 2131, 0, 8, &none, &align);
        assert_eq!(cuts.len(), 1);
        // a long prompt keeps two distinct aligned cuts
        let cuts = plan_cuts_with(&p, 100_036, 0, 8, &none, &align);
        assert_eq!(cuts, vec![2048, 99_840]); // 100_036 - 8 = 100_028 -> 99_840
        assert!(cuts.iter().all(|&c| c % BLOCK_TOKENS == 0), "every cut page-aligned: {cuts:?}");
    }

    // ---- the cache sized from the window (`CacheBudget::fit_pages_first`, `docs/PREFIX_CACHE_SIZING.md` section 5)

    const MIB: u64 = 1 << 20;
    // the exact per-family bytes (`gb_model::plan::tests::the_page_and_snapshot_sizes_behind_the_pins_are_pinned` pins them): a KV page and a
    // snapshot slot with the draft / MTP carry
    const GLM: (u64, u64) = (3_342_336, 147_646_464);
    const QWEN: (u64, u64) = (16_777_216, 195_887_104);
    const WINDOW: u64 = 262_144;

    fn rule(window: u64, live: usize) -> WindowRule {
        WindowRule { window_tokens: window, live_slots: live }
    }

    fn pf(bytes: u64, (page, snap): (u64, u64), window: u64, live: usize, spend: bool) -> CacheBudget {
        CacheBudget::fit_pages_first(bytes, snap, page, rule(window, live), spend)
    }

    /// The budgets swept by the invariants: nothing to 64 GiB in 16 MiB steps, plus the exact edges of the three regions.
    fn budgets((page, snap): (u64, u64), window: u64, live: usize) -> Vec<u64> {
        let r = rule(window, live);
        let floor = r.full_entry_pages() * page + 2 * snap;
        let target = r.target_pages() * page + r.wanted_slots() * snap;
        let mut v: Vec<u64> = (0..4096).map(|i| i * 16 * MIB).collect();
        v.extend([
            floor - 1,
            floor,
            floor + 1,
            target - 1,
            target,
            target + 1,
            target + snap + 8 * page - 1,
            target + snap + 8 * page,
        ]);
        v.sort();
        v
    }

    const CASES: [((u64, u64), u64, usize); 4] = [(GLM, WINDOW, 1), (QWEN, WINDOW, 6), (GLM, 1_048_576, 1), (QWEN, 65_536, 8)];

    #[test]
    fn the_window_rule_is_the_design_rule() {
        // docs/PREFIX_CACHE_SIZING.md section 5: C = ceil(W / 256), N = max(8, 4 S), T = (C + S ceil(C / 8)) pages + N slots
        let r = rule(WINDOW, 6);
        assert_eq!((r.full_entry_pages(), r.wanted_slots(), r.target_pages()), (1024, 24, 1792));
        let r = rule(WINDOW, 1);
        assert_eq!((r.full_entry_pages(), r.wanted_slots(), r.target_pages()), (1024, 8, 1152));
        // a short window still gets the pages of the minimum reach (4,096 tokens = 16 pages)
        assert_eq!(rule(1000, 1).full_entry_pages(), 16);
        assert_eq!(rule(1_048_576, 1).full_entry_pages(), 4096);
    }

    #[test]
    fn the_margin_is_two_gib_and_at_most_a_quarter_of_the_spare_memory() {
        assert_eq!(CACHE_MARGIN_BYTES, 2 << 30);
        assert_eq!(cache_margin_bytes(16 << 30), 2 << 30);
        assert_eq!(cache_margin_bytes(8 << 30), 2 << 30);
        assert_eq!(cache_margin_bytes(4 << 30), 1 << 30, "a node with little spare keeps a quarter, not two GiB");
        assert_eq!(cache_margin_bytes(0), 0);
    }

    #[test]
    fn a_window_cache_holds_one_full_window_entry_whenever_the_budget_allows() {
        for (fam, window, live) in CASES {
            let r = rule(window, live);
            let floor = r.full_entry_pages() * fam.0 + 2 * fam.1;
            for spend in [false, true] {
                for bytes in budgets(fam, window, live).into_iter().filter(|&b| b >= floor) {
                    let b = pf(bytes, fam, window, live, spend);
                    assert!(
                        b.page_blocks as u64 >= r.full_entry_pages() && b.reach_tokens() >= window,
                        "{fam:?} window {window} live {live} budget {bytes}: {b:?}"
                    );
                    assert!(b.snapshot_slots >= 2);
                }
            }
        }
    }

    #[test]
    fn a_window_cache_below_the_target_puts_pages_before_slots() {
        for (fam, window, live) in CASES {
            let r = rule(window, live);
            let floor = r.full_entry_pages() * fam.0 + 2 * fam.1;
            let target = r.target_pages() * fam.0 + r.wanted_slots() * fam.1;
            for bytes in budgets(fam, window, live).into_iter().filter(|&b| b >= floor && b < target) {
                let b = pf(bytes, fam, window, live, false);
                let room = (bytes - r.full_entry_pages() * fam.0) / fam.1;
                assert!(b.snapshot_slots as u64 <= r.wanted_slots(), "{fam:?} budget {bytes}: never more slots than wanted");
                assert!(
                    b.snapshot_slots == 2 || b.snapshot_slots as u64 * fam.1 <= bytes / 4,
                    "{fam:?} budget {bytes}: slots take at most a quarter of the budget: {b:?}"
                );
                assert!(
                    b.snapshot_slots as u64 <= room.max(2),
                    "{fam:?} budget {bytes}: slots never eat the full-window entry's pages"
                );
                assert_eq!(b.page_blocks as u64, (bytes - b.snapshot_slots as u64 * fam.1) / fam.0, "every page that is left");
            }
        }
    }

    #[test]
    fn a_window_cache_at_the_target_is_exactly_the_target() {
        let r = rule(WINDOW, 6);
        let target = r.target_pages() * QWEN.0 + r.wanted_slots() * QWEN.1;
        assert_eq!(target, 34_766_061_568, "33,155.5 MiB");
        let b = pf(target, QWEN, WINDOW, 6, false);
        assert_eq!((b.snapshot_slots, b.page_blocks, b.bytes()), (24, 1792, target));
        // more budget does not grow an Auto cache (nothing is spent that was not asked for)
        assert_eq!(pf(target * 2, QWEN, WINDOW, 6, false), b);
        // one byte short: pages first, slots no more than wanted
        let below = pf(target - 1, QWEN, WINDOW, 6, false);
        assert!(below.snapshot_slots <= 24 && below.page_blocks as u64 >= 1024, "{below:?}");
        let g = rule(WINDOW, 1);
        let gt = g.target_pages() * GLM.0 + g.wanted_slots() * GLM.1;
        assert_eq!(gt, 5_031_542_784, "4,798.45 MiB");
        let gb = pf(gt, GLM, WINDOW, 1, false);
        assert_eq!((gb.snapshot_slots, gb.page_blocks), (8, 1152));
    }

    #[test]
    fn a_window_cache_below_the_floor_degrades_to_two_slots_and_every_page_that_fits() {
        for (fam, window, live) in CASES {
            let r = rule(window, live);
            let floor = r.full_entry_pages() * fam.0 + 2 * fam.1;
            for bytes in budgets(fam, window, live).into_iter().filter(|&b| b < floor) {
                let b = pf(bytes, fam, window, live, false);
                if !b.enabled() {
                    continue;
                }
                assert_eq!(b.snapshot_slots, 2, "{fam:?} budget {bytes}");
                assert_eq!(b.page_blocks as u64, (bytes - 2 * fam.1) / fam.0, "{fam:?} budget {bytes}: every page that fits");
                assert!(b.reach_tokens() < window, "{fam:?} budget {bytes}: below the floor the reach is below the window");
            }
        }
    }

    #[test]
    fn a_window_cache_is_disabled_only_under_the_minimum_reach() {
        for fam in [GLM, QWEN] {
            let min = CacheBudget::pages_first_minimum_bytes(fam.1, fam.0);
            assert_eq!(min, 2 * fam.1 + 16 * fam.0, "two slots and 4,096 tokens (16 pages) of reach");
            for spend in [false, true] {
                assert!(!pf(min - 1, fam, WINDOW, 1, spend).enabled(), "{fam:?}: one byte under the minimum");
                let b = pf(min, fam, WINDOW, 1, spend);
                assert!(b.enabled() && b.reach_tokens() >= 4096, "{fam:?}: {b:?}");
                assert_eq!(pf(0, fam, WINDOW, 1, spend).bytes(), 0);
            }
        }
    }

    #[test]
    fn a_window_cache_never_exceeds_its_budget_and_an_explicit_size_wastes_no_whole_page() {
        for (fam, window, live) in CASES {
            for bytes in budgets(fam, window, live) {
                for spend in [false, true] {
                    let b = pf(bytes, fam, window, live, spend);
                    if !b.enabled() {
                        continue;
                    }
                    assert!(b.bytes() <= bytes, "{fam:?} budget {bytes}: {b:?}");
                    if spend {
                        assert!(
                            bytes - b.bytes() < fam.0,
                            "{fam:?} pin {bytes}: {} bytes left, a page is {}",
                            bytes - b.bytes(),
                            fam.0
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_window_cache_reach_up_to_the_window_and_its_slots_only_grow_with_the_budget() {
        // reach is monotone up to the window; past it extra pages are a working set, and buying a slot may cost a few of them
        for (fam, window, live) in CASES {
            for spend in [false, true] {
                let (mut prev_reach, mut prev_slots) = (0u64, 0usize);
                for bytes in budgets(fam, window, live) {
                    let b = pf(bytes, fam, window, live, spend);
                    let reach = b.reach_tokens().min(window);
                    assert!(
                        reach >= prev_reach && b.snapshot_slots >= prev_slots,
                        "{fam:?} window {window} budget {bytes}: {b:?} after {prev_reach}/{prev_slots}"
                    );
                    (prev_reach, prev_slots) = (reach, b.snapshot_slots);
                }
            }
        }
    }

    #[test]
    fn the_unit_cut_of_every_other_caller_is_unchanged() {
        // GLM at its shipped pin and Qwen at 40 GiB: the 1:8 cut that reaches 43K tokens and 266K tokens (`docs/PREFIX_CACHE_SIZING.md`)
        let g = CacheBudget::fit(3584 * MIB, GLM.1, GLM.0);
        assert_eq!((g.snapshot_slots, g.page_blocks, g.reach_tokens()), (21, 168, 43_008));
        let q = CacheBudget::fit(40_960 * MIB, QWEN.1, QWEN.0);
        assert_eq!((q.snapshot_slots, q.page_blocks, q.reach_tokens()), (130, 1040, 266_240));
        // and the window rule is a different function of the same inputs, not a change of `fit`
        assert_ne!(g, pf(3584 * MIB, GLM, WINDOW, 1, true));
    }

    #[test]
    fn glm_at_its_shipped_pin_reaches_a_full_window_with_two_slots() {
        let b = pf(3584 * MIB, GLM, WINDOW, 1, true);
        assert_eq!((b.snapshot_slots, b.page_blocks, b.reach_tokens()), (2, 1036, 265_216));
        // the margin over one full-window entry (1,024 pages) is 12 pages, 3,072 tokens, 40,251,392 bytes of the pin
        assert_eq!(b.page_blocks - 1024, 12);
        assert_eq!(3584 * MIB - (1024 * GLM.0 + 2 * GLM.1), 40_251_392);
        assert!(b.bytes() <= 3584 * MIB);
    }

    #[test]
    fn qwen_at_its_shipped_pin_holds_the_target_and_spends_the_rest_in_units() {
        let b = pf(40_960 * MIB, QWEN, WINDOW, 6, true);
        assert_eq!((b.snapshot_slots, b.page_blocks, b.reach_tokens()), (48, 1999, 511_744));
        assert!(b.snapshot_slots >= 4 * 6, "room for four entries per live slot of the shipped six");
        assert!(b.bytes() <= 40_960 * MIB);
    }

    #[test]
    fn red_snow_at_its_pin_degrades_and_says_so() {
        // 1,024 MiB on the GLM-family pack: 6 slots and 48 pages (12K tokens) with the unit cut, 2 slots and 232 pages (59K) by the window
        // rule; the window (262,144 configured, 101,632 admitted on one node) is above the reach either way
        for window in [WINDOW, 101_632] {
            let b = pf(1024 * MIB, GLM, window, 1, true);
            assert_eq!((b.snapshot_slots, b.page_blocks, b.reach_tokens()), (2, 232, 59_392), "window {window}");
            assert!(b.reach_tokens() < window);
        }
        let old = CacheBudget::fit(1024 * MIB, GLM.1, GLM.0);
        assert_eq!((old.snapshot_slots, old.page_blocks, old.reach_tokens()), (6, 48, 12_288));
    }
}
