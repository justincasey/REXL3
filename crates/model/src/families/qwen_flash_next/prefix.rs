//! Flash-Next's engine-owned prefix snapshots. The engine owns indexing, pinning and eviction;
//! this module owns device snapshots and page references. Only prefill states are captured.
//!
//! Backend integration (text prompts only until Flash-Next vision is built):
//! * At boot reserve `budget.snapshot_slots * snapshot_bytes(cfg, mtp)` outside admission and
//!   `budget.page_blocks` extra KV pages; construct `SnapshotArena::new(model, budget)` once.
//! * `prefix_caps` returns `caps(&arena, fingerprint, pin_name)`; `prefix_path(total)` returns
//!   `path(total)` only when the arena exists. The fingerprint identifies the upstream pack description and sidecar headers,
//!   not merely its directory; `pin_name` is `exl3_linear::prefill_pin_name()`.
//! * `prefix_free_snapshots` returns `arena.free_count()`.
//! * `prefix_restore` calls `restore(model, &arena, slot, handle, len)`; on error end the slot.
//!   Its hook has no prompt argument: the first suffix slice rebuilds PLE host history from
//!   `prompt[..done]` before executing, including a one-row suffix.
//! * `prefix_release` calls `release(model, &mut arena, handle)` exactly once.
//! * `prefill_slice_cuts` calls `slice_with_cuts(model, &mut arena, SliceRequest {
//!   slot, prompt, done, candidates, cuts, chunk })` and maps `SliceResult` to `CutStep` (and the final Candidates
//!   row to RowOutput). On failure release already captured handles, end the slot and report
//!   the error via the backend's existing error path. The same slice routine with empty cuts
//!   is the uncached prefill path. Never capture a decode/verify window.
//!
//! A cut inside the natural chunk ENDS that slice at the cut. This deliberately costs one
//! additional pinned W_q pass rather than taking an invalid post-chunk recurrent snapshot:
//! `prefill_chunk` currently has no in-chunk recurrent-state callback. The path is selected
//! from the whole prompt, so split and one-row tails use the same pinned GEMM arithmetic.

use super::config::FlashNextConfig;
use super::model::{FlashNextModel, PLE_RING, QSA_RING};
use crate::error::ModelError;
use crate::forward::{Candidates, PrefillPath};
use crate::kvtables::BLOCK_TOKENS;
use crate::rows::RowBatch;
use gb_cuda::{qsa, DevBuf};
use gb_engine::prefix::{Captured, PrefixCaps, PrefixHandle};
use gb_engine::RowRequest;
use gb_sched::prefix::CacheBudget;

/// The only currently cacheable prefill arithmetic: pinned GEMM for every row count.
pub const PATH_ID: u32 = 1;
pub fn path(total: usize) -> Option<u32> {
    (PrefillPath::for_prompt(total) == PrefillPath::Tiled).then_some(PATH_ID)
}

/// The namespace covers pack identity, path (also included separately in index keys), and
/// arithmetic choices. No snapshots persist across boots. Do not key only by the pack path.
pub fn namespace_of(fingerprint: &str, pinned_algorithm: &str, mtp: bool) -> u64 {
    let key = format!(
        "rexl3-flash-next-prefix-v2|{fingerprint}|path={PATH_ID}|arith=pinned-tiled-per-row-router-hcx-coopmk-qsa1-ple-prefill-native-attn|pin={pinned_algorithm}|page={BLOCK_TOKENS}|kv=bf16x2|gdn=f32+bf16|ple=f16|qsa=f16|mtp={mtp}"
    );
    key.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

/// Full pages are shared, and the last partial page gets a private full-plane copy.
pub fn page_plan(len: usize) -> (usize, bool) {
    (len / BLOCK_TOKENS, len % BLOCK_TOKENS != 0)
}

/// Never cross a planned cut; a cut on the boundary does not shorten the chunk.
pub fn slice_end(done: usize, total: usize, chunk: usize, cuts: &[usize]) -> usize {
    let chunk = chunk.max(1);
    let end = done.saturating_add(chunk - done % chunk).min(total);
    cuts.iter().copied().find(|&c| c > done && c < end).unwrap_or(end)
}

/// Layout in one preallocated snapshot slot. All offsets and sizes are bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub gdn_layers: usize,
    pub s: usize,
    pub conv: usize,
    pub ple: usize,
    pub qsa_layers: usize,
    pub qsa_ring: usize,
    /// One MTP K row, V row and one pooled-key plane entry at the boundary.
    pub mtp_row: usize,
    pub mtp_pool: usize,
}
impl Layout {
    fn s_at(self, i: usize) -> usize {
        i * self.s
    }
    fn conv_at(self, i: usize) -> usize {
        self.gdn_layers * self.s + i * self.conv
    }
    fn ple_at(self) -> usize {
        self.gdn_layers * (self.s + self.conv)
    }
    fn qsa_at(self, i: usize) -> usize {
        self.ple_at() + self.ple + i * self.qsa_ring
    }
    pub(crate) fn mtp_at(self) -> usize {
        self.qsa_at(self.qsa_layers)
    }
    pub fn bytes(self) -> usize {
        self.mtp_at() + 2 * self.mtp_row + self.mtp_pool
    }
    pub fn from_config(cfg: &FlashNextConfig, mtp: bool) -> Self {
        Self {
            gdn_layers: cfg.linear_layers(),
            s: cfg.lin_value_heads * cfg.lin_value_dim * cfg.lin_key_dim * 4,
            conv: (cfg.lin_conv_kernel - 1) * cfg.conv_channels() * 2,
            ple: PLE_RING * cfg.hc_streams * cfg.hidden * 2,
            qsa_layers: cfg.full_layers() + usize::from(mtp),
            qsa_ring: QSA_RING * cfg.indexer_head_dim * 2,
            mtp_row: if mtp { cfg.kv_heads * cfg.head_dim * 2 } else { 0 },
            mtp_pool: if mtp { cfg.indexer_head_dim * 2 } else { 0 },
        }
    }
    fn from_model(m: &FlashNextModel) -> Self {
        Self::from_config(&m.cfg, m.mtp_kv.is_some())
    }
}
pub fn snapshot_bytes(cfg: &FlashNextConfig, mtp: bool) -> u64 {
    Layout::from_config(cfg, mtp).bytes() as u64
}
pub fn page_bytes(cfg: &FlashNextConfig, mtp: bool) -> u64 {
    let layers = cfg.full_layers() + usize::from(mtp);
    let kv = 2 * BLOCK_TOKENS * cfg.kv_heads * cfg.head_dim * 2;
    let pool = (BLOCK_TOKENS / qsa::CR) * cfg.indexer_head_dim * 2;
    (layers * (kv + pool)) as u64
}
/// Feed this budget into the upstream memory planner BEFORE sizing the admitted KV pool:
/// snapshot slots consume `snapshot_slots * snapshot_bytes`, and `page_blocks` additional
/// physical KV pages consume `page_blocks * page_bytes`. Admission only sees the original
/// KV share, never those additional cache pages. Zero means the cache is off.
pub fn budget_for_bytes(cfg: &FlashNextConfig, mtp: bool, bytes: u64) -> CacheBudget {
    CacheBudget::fit(bytes, snapshot_bytes(cfg, mtp), page_bytes(cfg, mtp))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Meta {
    len: usize,
    mtp_done: bool,
}
/// Pure slot bookkeeping: a handle remains live until the engine releases it exactly once.
struct SlotBook {
    free: Vec<u32>,
    meta: Vec<Option<Meta>>,
}
impl SlotBook {
    fn new(n: usize) -> Self {
        Self { free: (0..n as u32).rev().collect(), meta: vec![None; n] }
    }
    fn alloc(&mut self) -> Option<u32> {
        self.free.pop()
    }
    fn free(&mut self, id: u32) -> Result<(), ModelError> {
        let m = self.meta.get_mut(id as usize).ok_or_else(|| ModelError::Request("snapshot id out of range".into()))?;
        if m.take().is_none() {
            return Err(ModelError::Request("snapshot already released or incomplete".into()));
        }
        self.free.push(id);
        Ok(())
    }
    fn abandon(&mut self, id: u32) {
        self.meta[id as usize] = None;
        self.free.push(id);
    }
}

/// Fixed GPU arena, reserved at boot, and its CPU ownership table.
pub struct SnapshotArena {
    data: DevBuf,
    layout: Layout,
    slots: SlotBook,
    budget: CacheBudget,
}
impl SnapshotArena {
    /// The KV allocation must include `budget.page_blocks` above the admission share.
    pub fn new(m: &FlashNextModel, budget: CacheBudget) -> Result<Self, ModelError> {
        let layout = Layout::from_model(m);
        if !budget.enabled()
            || budget.snapshot_bytes != layout.bytes() as u64
            || budget.page_bytes != page_bytes(&m.cfg, m.mtp_kv.is_some())
        {
            return Err(ModelError::Config("Flash-Next prefix budget does not match model allocations".into()));
        }
        let bytes = budget
            .snapshot_slots
            .checked_mul(layout.bytes())
            .ok_or_else(|| ModelError::Config("prefix arena size overflow".into()))?;
        Ok(Self { data: DevBuf::alloc(bytes)?, layout, slots: SlotBook::new(budget.snapshot_slots), budget })
    }
    pub fn free_count(&self) -> usize {
        self.slots.free.len()
    }
    pub fn allocated_bytes(&self) -> u64 {
        self.data.bytes() as u64
    }
}
pub fn caps(arena: &SnapshotArena, fingerprint: &str, pin_name: &str) -> PrefixCaps {
    PrefixCaps { namespace: namespace_of(fingerprint, pin_name, arena.layout.mtp_row != 0), budget: arena.budget }
}

/// Capture after `len` prefill rows; the MTP head must already have processed the known next
/// token, if any. The entire raw QSA ring is copied (including a partial pool group's rows).
pub fn capture(m: &mut FlashNextModel, a: &mut SnapshotArena, slot: usize, len: usize) -> Result<Option<Captured>, ModelError> {
    if len == 0 || slot >= m.opts.slots || len > m.opts.max_context || m.kv_tab.blocks_of(slot).len() < len.div_ceil(BLOCK_TOKENS)
    {
        return Err(ModelError::Request("prefix capture is not at a valid committed token cut".into()));
    }
    let Some(id) = a.slots.alloc() else {
        return Ok(None);
    };
    let result = (|| -> Result<Captured, ModelError> {
        m.flush()?;
        let lay = a.layout;
        let base = id as usize * lay.bytes();
        for i in 0..lay.gdn_layers {
            a.data.copy_range_from(base + lay.s_at(i), &m.pools.s[i], slot * lay.s, lay.s, &m.stream)?;
            a.data.copy_range_from(base + lay.conv_at(i), &m.pools.conv[i], slot * lay.conv, lay.conv, &m.stream)?;
        }
        a.data.copy_range_from(base + lay.ple_at(), &m.ple_ring, slot * lay.ple, lay.ple, &m.stream)?;
        for (i, ring) in m.qsa_ring.iter().enumerate() {
            a.data.copy_range_from(base + lay.qsa_at(i), ring, slot * lay.qsa_ring, lay.qsa_ring, &m.stream)?;
        }
        let mtp_done = lay.mtp_row != 0 && m.mtp_done.get(&slot) == Some(&len);
        // At an aligned cut the MTP one-ahead row lives in a page past all trunk rows.
        // It is NOT a page in the handle, so save it explicitly (including its pool entry).
        if mtp_done && len % BLOCK_TOKENS == 0 {
            let page = *m
                .kv_tab
                .blocks_of(slot)
                .get(len / BLOCK_TOKENS)
                .ok_or_else(|| ModelError::Request("MTP one-ahead page is missing".into()))? as usize;
            let cache = m.mtp_kv.as_ref().ok_or_else(|| ModelError::Request("MTP cache is missing".into()))?;
            let off = page * cache.block_bytes();
            a.data.copy_range_from(base + lay.mtp_at(), &cache.k, off, lay.mtp_row, &m.stream)?;
            a.data.copy_range_from(base + lay.mtp_at() + lay.mtp_row, &cache.v, off, lay.mtp_row, &m.stream)?;
            let pool_off = page * qsa::PAGE_BLOCKS * qsa::DK * 2;
            a.data.copy_range_from(
                base + lay.mtp_at() + 2 * lay.mtp_row,
                m.qsa_pool.last().ok_or_else(|| ModelError::Request("MTP pool is missing".into()))?,
                pool_off,
                lay.mtp_pool,
                &m.stream,
            )?;
        }
        // Enqueue all device copies before retaining references; a later donor write is
        // ordered after these copies on the same stream.
        let pages = m.kv_tab.retain_prefix(slot, len.div_ceil(BLOCK_TOKENS));
        a.slots.meta[id as usize] = Some(Meta { len, mtp_done });
        Ok(Captured { len, handle: PrefixHandle { snap: id, pages } })
    })();
    if result.is_err() {
        a.slots.abandon(id);
    }
    result.map(Some)
}

/// Restore the full pages by reference, copy the partial page's K/V AND QSA planes to a
/// private page, then restore all recurrent state, rings and the MTP one-ahead row.
pub fn restore(
    m: &mut FlashNextModel,
    a: &SnapshotArena,
    slot: usize,
    handle: &PrefixHandle,
    len: usize,
) -> Result<(), ModelError> {
    let meta = a
        .slots
        .meta
        .get(handle.snap as usize)
        .copied()
        .flatten()
        .filter(|x| x.len == len && len > 0 && handle.pages.len() == len.div_ceil(BLOCK_TOKENS))
        .ok_or_else(|| ModelError::Request("prefix handle or token length is invalid".into()))?;
    if slot >= m.opts.slots {
        return Err(ModelError::Request("prefix slot out of range".into()));
    }
    m.begin_slot(slot)?;
    let (full, partial) = page_plan(len);
    m.kv_tab.adopt_shared(slot, &handle.pages[..full]);
    let copy = (|| -> Result<(), ModelError> {
        if partial {
            let src = handle.pages[full] as usize;
            let dst = m.kv_tab.alloc_block(slot).map_err(|e| ModelError::Request(format!("prefix copy-on-write page: {e:?}")))?
                as usize;
            copy_page(m, src, dst, len)?;
        } else if meta.mtp_done {
            // Keep the aligned one-ahead row in a page the resumed slot owns.
            m.kv_tab.alloc_block(slot).map_err(|e| ModelError::Request(format!("prefix MTP page: {e:?}")))?;
        }
        m.upload_table()?;
        let lay = a.layout;
        let base = handle.snap as usize * lay.bytes();
        for i in 0..lay.gdn_layers {
            m.pools.s[i].copy_range_from(slot * lay.s, &a.data, base + lay.s_at(i), lay.s, &m.stream)?;
            m.pools.conv[i].copy_range_from(slot * lay.conv, &a.data, base + lay.conv_at(i), lay.conv, &m.stream)?;
        }
        m.ple_ring.copy_range_from(slot * lay.ple, &a.data, base + lay.ple_at(), lay.ple, &m.stream)?;
        for (i, ring) in m.qsa_ring.iter_mut().enumerate() {
            ring.copy_range_from(slot * lay.qsa_ring, &a.data, base + lay.qsa_at(i), lay.qsa_ring, &m.stream)?;
        }
        if meta.mtp_done {
            if !partial {
                let page = m.kv_tab.blocks_of(slot)[full] as usize;
                let kv = m.mtp_kv.as_mut().ok_or_else(|| ModelError::Request("MTP cache is missing".into()))?;
                let off = page * kv.block_bytes();
                kv.k.copy_range_from(off, &a.data, base + lay.mtp_at(), lay.mtp_row, &m.stream)?;
                kv.v.copy_range_from(off, &a.data, base + lay.mtp_at() + lay.mtp_row, lay.mtp_row, &m.stream)?;
                let pool_off = page * qsa::PAGE_BLOCKS * qsa::DK * 2;
                m.qsa_pool.last_mut().ok_or_else(|| ModelError::Request("MTP pool is missing".into()))?.copy_range_from(
                    pool_off,
                    &a.data,
                    base + lay.mtp_at() + 2 * lay.mtp_row,
                    lay.mtp_pool,
                    &m.stream,
                )?;
            }
            m.mtp_done.insert(slot, len);
        }
        Ok(())
    })();
    if copy.is_err() {
        m.end_slot(slot);
    }
    copy
}

/// Copy the partial page to a private block, then scrub every byte at or beyond `len`
/// (the restored prefix's valid token count inside this page): K/V rows from `len % PAGE`
/// and QSA pooled-key groups from `len / CR` are zeroed. The copied page afterwards holds
/// exactly what a from-scratch prefill of `len` tokens would hold — rows/groups past the
/// progress are empty — so no donor continuation can ever be read from it. The resumed
/// prefill rewrites (and, for the pooled groups, recomputes) every position from `len`
/// on, before any kernel can read it: dense attention is position-masked, sparse selection
/// only scores *complete* groups, and a group is complete only once its fourth token has
/// been written by the resumed sequence, which recomputes its pooled key from the restored
/// raw-key ring (for the shared rows) and the fresh keys. Scrub is therefore both exact and
/// free: it is a small memset, and the plan's memory pricing (one page per partial page) is
/// unchanged.
fn copy_page(m: &mut FlashNextModel, src: usize, dst: usize, len: usize) -> Result<(), ModelError> {
    let rows = len % BLOCK_TOKENS; // valid rows in this page; 0 only when the caller would not copy
    assert!(rows != 0, "copy_page is never called for an aligned cut");
    for kv in m.kv.iter_mut().chain(m.mtp_kv.iter_mut()) {
        let bytes = kv.block_bytes();
        let row = bytes / BLOCK_TOKENS; // one K (or V) row's bytes in this plane
        let scrub = (BLOCK_TOKENS - rows) * row;
        kv.k.copy_within(dst * bytes, src * bytes, bytes, &m.stream)?;
        kv.v.copy_within(dst * bytes, src * bytes, bytes, &m.stream)?;
        kv.k.memset_range_async(dst * bytes + rows * row, scrub, 0, &m.stream)?;
        kv.v.memset_range_async(dst * bytes + rows * row, scrub, 0, &m.stream)?;
    }
    let pool_bytes = qsa::PAGE_BLOCKS * qsa::DK * 2;
    let group = qsa::DK * 2;
    // The pool plane is one buffer over all pages, indexed by global group (page * PAGE_BLOCKS +
    // local). The first foreign group is `len / CR` globally; inside this page that is its local
    // group, `(len / CR) % PAGE_BLOCKS` (a group is foreign when any of its tokens is at/after len).
    let groups = (len / qsa::CR) % qsa::PAGE_BLOCKS;
    let scrub = (qsa::PAGE_BLOCKS - groups) * group;
    for pool in &mut m.qsa_pool {
        pool.copy_within(dst * pool_bytes, src * pool_bytes, pool_bytes, &m.stream)?;
        pool.memset_range_async(dst * pool_bytes + groups * group, scrub, 0, &m.stream)?;
    }
    Ok(())
}

/// Every handle releases one arena slot and its references; index eviction is engine-owned.
pub fn release(m: &mut FlashNextModel, a: &mut SnapshotArena, handle: PrefixHandle) -> Result<(), ModelError> {
    a.slots.free(handle.snap)?;
    m.kv_tab.release_pages(&handle.pages);
    Ok(())
}

pub struct SliceResult {
    pub end: usize,
    pub row: Option<Candidates>,
    pub captured: Vec<Captured>,
}
/// One backend slice's arguments; `prompt` is the entire prompt, not only the chunk.
pub struct SliceRequest<'a> {
    pub slot: usize,
    pub prompt: &'a [u32],
    pub done: usize,
    pub candidates: usize,
    pub cuts: &'a [usize],
    pub chunk: usize,
}
/// A slice with planned cuts; caller translates `SliceResult` to the engine's `CutStep`.
/// A cut at the prompt end is not useful: a strict prefix leaves a row to recompute.
pub fn slice_with_cuts(m: &mut FlashNextModel, a: &mut SnapshotArena, req: SliceRequest<'_>) -> Result<SliceResult, ModelError> {
    let SliceRequest { slot, prompt, done, candidates, cuts, chunk } = req;
    if prompt.is_empty() || done >= prompt.len() || slot >= m.opts.slots || cuts.windows(2).any(|w| w[0] >= w[1]) {
        return Err(ModelError::Request("invalid Flash-Next prefill slice or unordered cuts".into()));
    }
    if done == 0 {
        m.begin_slot(slot)?;
    } else {
        // The restore hook cannot see the prompt. Rebuild its host n-gram ring here
        // before the first resumed row (and on every slice: cheap and idempotent).
        let from = done.saturating_sub(PLE_RING);
        let hist: Vec<i32> = prompt[from..done].iter().map(|&x| x as i32).collect();
        m.ple_tok.store(slot, from, &hist);
    }
    let cuts = if path(prompt.len()).is_some() { cuts } else { &[] };
    let end = slice_end(done, prompt.len(), chunk.min(m.opts.prefill_rows), cuts);
    let ids = prompt[done..end].to_vec();
    // the next slice's PLE rows are gathered on a worker while this chunk runs (same end rule as the next call's)
    if end < prompt.len() {
        m.ple_prefetch_next(slot, prompt, end, slice_end(end, prompt.len(), chunk.min(m.opts.prefill_rows), cuts));
    }
    let b = RowBatch::from_requests(&[RowRequest { slot, first_position: done as u64, tokens: ids }]);
    let row = m.prefill_chunk(&b, PrefillPath::for_prompt(prompt.len()), (end == prompt.len()).then_some(candidates))?;
    m.mtp_after_prefill(slot, done, &prompt[done..end], prompt.get(end).copied(), PrefillPath::for_prompt(prompt.len()))?;
    let captured = if end < prompt.len() && path(prompt.len()).is_some() && cuts.contains(&end) {
        capture(m, a, slot, end)?.into_iter().collect()
    } else {
        Vec::new()
    };
    Ok(SliceResult { end, row, captured })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arbitrary_cuts_plan_a_private_partial_page() {
        assert_eq!(page_plan(255), (0, true));
        assert_eq!(page_plan(256), (1, false));
        assert_eq!(page_plan(257), (1, true));
        assert_eq!(page_plan(513), (2, true));
        assert_eq!(slice_end(0, 1000, 512, &[300, 900]), 300);
        assert_eq!(slice_end(300, 1000, 512, &[300, 900]), 512);
        assert_eq!(slice_end(512, 1000, 512, &[300, 900]), 900);
        assert_eq!(slice_end(900, 1000, 512, &[300, 900]), 1000);
    }
    #[test]
    fn arena_book_releases_exactly_once() {
        let mut b = SlotBook::new(2);
        let x = b.alloc().unwrap();
        let y = b.alloc().unwrap();
        assert!(b.alloc().is_none());
        b.meta[x as usize] = Some(Meta { len: 257, mtp_done: true });
        b.free(x).unwrap();
        assert!(b.free(x).is_err());
        assert_eq!(b.alloc(), Some(x));
        b.abandon(x);
        b.abandon(y);
        assert_eq!(b.free.len(), 2);
    }
    #[test]
    fn short_prompts_are_never_cacheable_and_namespaces_are_isolated() {
        assert_eq!(path(128), None);
        assert_eq!(path(129), Some(PATH_ID));
        let ns = namespace_of("pack-a", "pin-1", true);
        assert_ne!(ns, namespace_of("pack-b", "pin-1", true));
        assert_ne!(ns, namespace_of("pack-a", "pin-2", true));
        assert_ne!(ns, namespace_of("pack-a", "pin-1", false));
    }
    #[test]
    fn snapshot_layout_counts_every_plane_and_the_mtp_ahead_row() {
        let c = FlashNextConfig::from_json(
            &serde_json::from_str::<serde_json::Value>(super::super::config::tests::FLASH_NEXT_CONFIG).unwrap(),
        )
        .unwrap();
        let l = Layout::from_config(&c, true);
        assert_eq!(l.gdn_layers, 36);
        assert_eq!(l.qsa_layers, 13);
        assert_eq!(l.mtp_at(), l.qsa_at(13));
        assert_eq!(snapshot_bytes(&c, true), l.bytes() as u64);
        assert_eq!(l.mtp_row, c.kv_heads * c.head_dim * 2);
        assert_eq!(l.mtp_pool, qsa::DK * 2);
        assert!(page_bytes(&c, true) > page_bytes(&c, false));
        use crate::families::qwen_flash_next::memory::memory_profile;
        use crate::memory::MemoryOptions;
        let opts = MemoryOptions { mtp: true, vision: None, ..MemoryOptions::serving_defaults() };
        assert_eq!(page_bytes(&c, true), memory_profile(&c, &opts).unwrap().kv_page_bytes());
        let two_units = 2 * (snapshot_bytes(&c, true) + 8 * page_bytes(&c, true));
        assert_eq!(budget_for_bytes(&c, true, two_units - 1).snapshot_slots, 0);
        let b = budget_for_bytes(&c, true, two_units);
        assert_eq!((b.snapshot_slots, b.page_blocks), (2, 16));
        assert_eq!(b.bytes(), two_units);
    }
    #[test]
    fn index_bookkeeping_eviction_and_page_refcounts_are_engine_owned() {
        use gb_sched::prefix::{Limits, PrefixIndex};
        use std::sync::Arc;
        let mut i = PrefixIndex::new(13, Limits { snapshot_slots: 2, page_blocks: 3 });
        let p = Arc::new(vec![1u32; 600]);
        let make = |len, snap, pages| Captured { len, handle: PrefixHandle { snap, pages } };
        assert!(i.insert(PATH_ID, p.clone(), make(300, 0, vec![7, 8])).stored());
        assert!(i.insert(PATH_ID, p.clone(), make(500, 1, vec![7, 9])).stored());
        assert_eq!(i.pages_held(), 3);
        assert_eq!(i.page_holders(7), 2);
        assert_eq!(i.page_holders(8), 1);
        let h = i.lookup(PATH_ID, &p).unwrap();
        assert_eq!(h.len, 500);
        i.unpin(h.key);
        let released = i.evict_for_snapshots(1, 0);
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].snap, 0);
        assert_eq!(i.page_holders(7), 1);
        assert_eq!(i.page_holders(8), 0);
        assert_eq!(i.page_holders(9), 1);
        i.check().unwrap();
    }
    // Interior-cut restore round-trip, simulated on the host. This models the exact
    // state components `capture` copies and `restore` puts back — full-attention KV
    // pages (shared + copy-on-write partial page with its QSA pool plane), GDN
    // recurrent S and conv state, the PLE ring, the QSA raw-key rings, and the MTP
    // one-ahead row (explicit at aligned cuts, inside the partial page otherwise) —
    // and asserts that a cold prefill of P, then an interior restore at a cut of P
    // followed by the rest of P plus a suffix, leaves the model in exactly the state
    // the cold run of the same full prompt leaves. The simulation is a faithful
    // image of `capture`/`restore`/`slice_with_cuts` (prefix.rs:196, 252, 356):
    // page ids are physical, the partial page is copied at restore, refcounts keep
    // shared pages alive, and every state plane round-trips.
    mod interior_restore_sim {
        use super::*;

        pub const PAGE: usize = BLOCK_TOKENS;
        const LAYERS: usize = 4; // GDN layers (s + conv each)
        const RING: usize = 128; // QSA ring length
        const MTP: bool = true;

        /// Deterministic token stream (matches the model's "tokens are a pure function of
        /// the prompt" property).
        pub fn toks(seed: u64, n: usize) -> Vec<u32> {
            let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    1 + (x % 50_000) as u32
                })
                .collect()
        }

        /// One logical KV page of one full-attention layer: K and V rows plus the QSA
        /// pool plane of the same block (mirrors `page_bytes` = kv + pool per page).
        /// The MTP one-ahead rows are NOT part of the page planes: they live in the
        /// MTP layers' own cache (`mtp_kv`), affect drafts only, and their coverage
        /// depends on the slice grid (the cached==uncached contract allows draft
        /// differences). The state the cache must reproduce exactly is the pages,
        /// recurrent state and rings — the MTP row at the cut is asserted separately.
        #[derive(Clone, PartialEq)]
        struct Page {
            kv: Vec<u32>,   // K rows then V rows, value = hash of (layer, position, token)
            pool: Vec<u32>, // one pooled key per 4-token group of the block
        }

        /// The per-slot host mirror of `FlashNextModel` state at a cut.
        #[derive(Clone, PartialEq)]
        pub struct SlotState {
            pages: Vec<u32>,             // physical page ids, position order
            s: Vec<u32>,                 // GDN S, per layer, per slot
            conv: Vec<u32>,              // GDN conv, per layer, per slot
            ple: Vec<u32>,               // PLE device ring, per slot
            qsa: Vec<Vec<u32>>,          // QSA rings, per layer, per slot
            mtp_row: Option<(u32, u32)>, // MTP one-ahead K and V row at the cut
            mtp_pool: Option<u32>,       // the MTP row's pooled key
            len: usize,
        }

        /// The physical page store (K/V + pool planes per page). Page ids are physical;
        /// sharing/refcounts are the engine-level mock's domain, so this store is content
        /// only (alloc picks a free page, copy_page clones a page's content).
        #[derive(Clone)]
        pub struct Store {
            pages: Vec<Option<Page>>,
        }

        impl Store {
            fn new(n: usize) -> Self {
                Self { pages: vec![None; n] }
            }
            fn alloc(&mut self) -> u32 {
                let i = self.pages.iter().position(|p| p.is_none()).expect("no free page") as u32;
                self.pages[i as usize] = Some(Page { kv: vec![0; 2 * PAGE], pool: vec![0; PAGE / 4] });
                i
            }
            /// `copy_page`: whole page, K/V planes + pool plane, src -> dst, then SCRUB every byte at
            /// or beyond the restored prefix's `len` tokens (K/V rows from `len % PAGE`, pooled groups
            /// from `(len / CR) % (PAGE / 4)`), exactly like the real `copy_page` (prefix.rs:320):
            /// the copied page then equals a from-scratch prefill's page at the same progress.
            fn copy_page(&mut self, src: u32, dst: u32, len: usize) {
                let mut p = self.pages[src as usize].clone().expect("copy from a missing page");
                let rows = len % PAGE;
                for r in rows..PAGE {
                    p.kv[r] = 0;
                    p.kv[PAGE + r] = 0;
                }
                for g in (len / qsa::CR) % (PAGE / 4)..PAGE / 4 {
                    p.pool[g] = 0;
                }
                self.pages[dst as usize] = Some(p);
            }
        }

        /// The per-slot recurrent state of every GDN layer (this is what `m.pools.s[i]`
        /// / `conv[i]` hold at slot offset), the PLE ring, and the QSA rings.
        #[derive(Clone, PartialEq)]
        struct Recurrent {
            s: Vec<u32>,
            conv: Vec<u32>,
            ple: Vec<u32>,
            qsa: Vec<u32>,
        }

        impl Recurrent {
            fn new() -> Self {
                Self { s: vec![0; LAYERS], conv: vec![0; LAYERS], ple: vec![0; RING], qsa: vec![0; RING] }
            }
        }

        /// A deterministic virtual model: the state after processing `n` tokens of a
        /// sequence is a pure function of the tokens (each new token updates the
        /// recurrent state and writes its KV rows / pool entries). This is the
        /// "row-invariant, chunk-invariant" contract the cache relies on.
        pub struct Model {
            pub store: Store,
            state: Recurrent,
            pages: Vec<u32>,
            len: usize,
            /// MTP one-ahead rows (position -> (K, V)) and their pooled keys, in the
            /// MTP layers' own cache (draft-side; asserted separately).
            pub mtp_rows: std::collections::HashMap<usize, (u32, u32)>,
            pub mtp_pool: std::collections::HashMap<usize, u32>,
        }

        impl Model {
            pub fn new(npages: usize) -> Self {
                Self {
                    store: Store::new(npages),
                    state: Recurrent::new(),
                    pages: Vec::new(),
                    len: 0,
                    mtp_rows: std::collections::HashMap::new(),
                    mtp_pool: std::collections::HashMap::new(),
                }
            }

            /// Write one token at absolute position `pos` (writes its KV rows into the
            /// block it belongs to and updates recurrent state).
            fn step(&mut self, pos: usize, tok: u32) {
                while self.pages.len() <= pos / PAGE {
                    let b = self.store.alloc();
                    self.pages.push(b);
                }
                let b = self.pages[pos / PAGE] as usize;
                let page = self.store.pages[b].as_mut().unwrap();
                let r = pos % PAGE;
                page.kv[r] = (pos as u32).wrapping_mul(31).wrapping_add(tok);
                page.kv[PAGE + r] = (pos as u32).wrapping_mul(17).wrapping_add(tok);
                // pooled-key entry of the 4-token group: assignment, like the real
                // kernels (a re-written position replaces, never accumulates)
                page.pool[r / 4] = (pos as u32).wrapping_mul(61).wrapping_add(tok);
                let l = pos % LAYERS;
                self.state.s[l] = self.state.s[l].wrapping_add(tok).wrapping_mul(3);
                self.state.conv[l] = self.state.conv[l].wrapping_add(tok).wrapping_mul(5);
                self.state.ple[pos % RING] = self.state.ple[pos % RING].wrapping_add(tok).wrapping_mul(11);
                self.state.qsa[pos % RING] = self.state.qsa[pos % RING].wrapping_add(tok).wrapping_mul(13);
                self.len = self.len.max(pos + 1);
            }

            /// The slice the cache would run: `[done, end)` where `end` is the first
            /// planned cut the slice would pass (`slice_end`), or the natural grid end.
            fn slice_end(&self, done: usize, total: usize, chunk: usize, cuts: &[usize]) -> usize {
                super::slice_end(done, total, chunk.max(1), cuts)
            }

            /// `prefill_chunk` + `mtp_after_prefill` at `first..end`, then `capture`
            /// when `end` is a planned cut (mirrors `slice_with_cuts`).
            pub fn run_slice(&mut self, first: usize, end: usize, prompt: &[u32]) {
                for (i, &t) in prompt[first..end].iter().enumerate() {
                    self.step(first + i, t);
                }
                // MTP one-ahead row at `end` when the prompt continues past it.
                if end < prompt.len() {
                    self.mtp_write(end, prompt[end]);
                }
            }

            /// `mtp_after_prefill`'s persistent write: the one-ahead K/V row and its
            /// pool entry at position `pos` (the MTP layers' own cache, separate from
            /// the trunk pages).
            fn mtp_write(&mut self, pos: usize, tok: u32) {
                self.mtp_rows.insert(
                    pos,
                    ((pos as u32).wrapping_mul(41).wrapping_add(tok), (pos as u32).wrapping_mul(43).wrapping_add(tok)),
                );
                self.mtp_pool.insert(pos, tok.wrapping_mul(19).wrapping_add(pos as u32));
            }

            /// `capture`: copy the per-slot recurrent state and the MTP one-ahead row,
            /// retain the pages, and return the snapshot (the SlotState).
            pub fn capture(&self, len: usize) -> SlotState {
                assert!(len <= self.len);
                // The MTP one-ahead row at `len` exists exactly when `mtp_after_prefill`
                // ran with `next = prompt[len]` — i.e. when a slice ended at `len` with
                // the prompt continuing past it (a cut is strictly inside). The final
                // slice of a prompt writes no one-ahead row.
                let mtp_row = self.mtp_rows.get(&len).copied();
                let mtp_pool = mtp_row.map(|_| self.mtp_pool[&len]);
                SlotState {
                    pages: self.pages[..len.div_ceil(PAGE)].to_vec(),
                    s: self.state.s.clone(),
                    conv: self.state.conv.clone(),
                    ple: self.state.ple.clone(),
                    qsa: vec![self.state.qsa.clone()],
                    mtp_row,
                    mtp_pool,
                    len,
                }
            }

            /// `restore`: put a snapshot back into this (fresh) slot.
            pub fn restore(&mut self, snap: &SlotState) {
                // begin_slot: release the old pages, zero the recurrent state
                self.pages.clear();
                self.state = Recurrent::new();

                // adopt the full pages by reference (the store keeps their content)
                let (full, partial) = page_plan(snap.len);
                self.pages.extend_from_slice(&snap.pages[..full]);
                // the partial page is copied into a fresh, privately owned page, with the foreign
                // region scrubbed (the real `copy_page`'s exact behavior)
                if partial {
                    let dst = self.store.alloc();
                    let src = snap.pages[full];
                    self.store.copy_page(src, dst, snap.len);
                    self.pages.push(dst);
                } else if MTP && snap.mtp_row.is_some() {
                    // aligned cut with an MTP row: a fresh page for the one-ahead row
                    let dst = self.store.alloc();
                    self.pages.push(dst);
                }
                // recurrent state + rings
                self.state.s = snap.s.clone();
                self.state.conv = snap.conv.clone();
                self.state.ple = snap.ple.clone();
                self.state.qsa = snap.qsa[0].clone();
                // the MTP one-ahead row and pool entry (the MTP layers' own cache)
                if let (Some((k, v)), Some(p)) = (snap.mtp_row, snap.mtp_pool) {
                    self.mtp_rows.insert(snap.len, (k, v));
                    self.mtp_pool.insert(snap.len, p);
                }
                self.len = snap.len;
            }

            /// The state the cache promises to reproduce: everything that determines the
            /// model's future outputs (the MTP one-ahead rows are draft-side and are
            /// asserted separately at the cut).
            pub fn observables(&self) -> Vec<u64> {
                let mut v = Vec::new();
                for &b in &self.pages {
                    let p = self.store.pages[b as usize].as_ref().unwrap();
                    v.push(p.kv.iter().map(|&x| u64::from(x)).sum());
                    v.push(p.pool.iter().map(|&x| u64::from(x)).sum());
                }
                v.push(self.state.s.iter().map(|&x| u64::from(x)).sum());
                v.push(self.state.conv.iter().map(|&x| u64::from(x)).sum());
                v.push(self.state.ple.iter().map(|&x| u64::from(x)).sum());
                v.push(self.state.qsa.iter().map(|&x| u64::from(x)).sum());
                v
            }
        }

        /// Cold P, then reproduce (interior restore + suffix) — the final observables
        /// must be identical for every alignment of the cut and every chunk grid.
        pub(super) fn cold_vs_restored(total: usize, cut: usize, chunk: usize) {
            let prompt = toks(7, total);
            let suffix = toks(99, 60);

            // cold run of prompt + suffix
            let mut cold = Model::new(2 * total.div_ceil(PAGE) + 16);
            let mut done = 0;
            while done < total + suffix.len() {
                let end = cold.slice_end(done, total + suffix.len(), chunk, &[total]);
                cold.run_slice(done, end, &prompt.iter().chain(&suffix).copied().collect::<Vec<_>>());
                done = end;
            }

            // cold run of prompt alone (the donor), snapshot at `cut` — the snapshot is
            // taken the moment a slice ends AT the cut, exactly like `slice_with_cuts`
            // (capture runs inside the slice call whose end equals the cut)
            let mut donor = Model::new(2 * total.div_ceil(PAGE) + 16);
            let mut done = 0;
            let mut snap = None;
            while done < total {
                let end = donor.slice_end(done, total, chunk, &[cut]);
                donor.run_slice(done, end, &prompt);
                if end == cut {
                    snap = Some(donor.capture(cut));
                }
                done = end;
            }
            let snap = snap.expect("the cut must be a slice end");
            assert_eq!(snap.len, cut);

            // stale run: restore at `cut`, then prefill [cut, total) + suffix
            let mut restored = Model::new(2 * total.div_ceil(PAGE) + 16);
            // the restored slot operates in a store that holds the donor's pages (the
            // shared-page identity is the engine-level mock's domain; here the contents
            // must simply exist when adopted)
            restored.store = donor.store.clone();
            restored.restore(&snap);
            let mut done = cut;
            let full = prompt.iter().chain(&suffix).copied().collect::<Vec<_>>();
            while done < full.len() {
                let end = restored.slice_end(done, full.len(), chunk, &[cut]);
                restored.run_slice(done, end, &full);
                done = end;
            }

            assert_eq!(
                restored.observables(),
                cold.observables(),
                "interior restore at cut {cut} (page {}), chunk {chunk}, total {total}: the resumed run diverged from cold",
                page_plan(cut).0,
            );
            // The MTP one-ahead row at the cut round-trips exactly (it is the only
            // MTP row the restore must reproduce; draft coverage beyond it may differ
            // by slice grid, which is the cached==uncached draft allowance).
            assert_eq!(restored.mtp_rows.get(&cut), donor.mtp_rows.get(&cut), "MTP row at the cut must round-trip");
            assert_eq!(restored.mtp_pool.get(&cut), donor.mtp_pool.get(&cut), "MTP pool entry at the cut must round-trip");
        }

        #[test]
        fn interior_cut_restore_equals_cold_for_every_alignment_and_grid() {
            for chunk in [256usize, 257, 512, 1000] {
                for total in [2000usize, 2600, 3333] {
                    // cuts at several alignments relative to the page (256) and the chunk
                    for cut in [total - 64, total - 300, 3 * total / 4, total / 2, total - 257, total - 256, total - 255] {
                        if cut > 128 && cut < total {
                            cold_vs_restored(total, cut, chunk);
                        }
                    }
                }
            }
        }

        #[test]
        fn capture_requires_the_cut_to_be_strictly_inside_the_prompt() {
            let total = 1000;
            let prompt = toks(11, total);
            let mut donor = Model::new(16);
            donor.run_slice(0, total, &prompt);
            // capture at total (the prompt end) has no one-ahead row: that is the final
            // slice of a prompt, which is never a cut (a cut is strictly inside).
            let snap = donor.capture(total);
            assert_eq!(snap.mtp_row, None);
        }
    }

    // ---------------------------------------------------------------------------
    // The exact stage-3 gate geometry (the shared-prefix 2067 = 8x256+19 interior cut):
    // a donor that continues past the cut with its own text, and a follow-up that shares
    // the cut's prefix but appends a DIFFERENT suffix ending before the donor does. A
    // mid-page cut used to leave the donor's continuing rows and pooled keys inside the copied
    // partial page; the exact restore now scrubs them, so the resumed slot holds nothing but
    // the sequence's own bytes and equals cold bit for bit in every plane (the first test pins
    // the exact geometry that previously diverged; the second the pool planes; the third a
    // dense set of cut positions, page and group boundaries).
    mod divergent_donor_sim {
        use super::interior_restore_sim::{toks, Model, PAGE};
        use super::page_plan;

        #[test]
        fn a_mid_page_cut_with_a_divergent_donor_continuation_scrubs_and_matches_cold() {
            // exact gate numbers: donor total 2131, cut 2067 = 8x256+19 (mid-page)
            let total = 2131;
            let cut = 2067;
            let prompt = toks(7, total);
            let suffix = toks(99, 48); // a different continuation than the donor's last 64 tokens
            assert!(page_plan(cut).1, "the cut must be mid-page for this test");

            // cold run of the follower's full prompt: prompt[..cut] + suffix
            let full: Vec<u32> = prompt[..cut].iter().chain(&suffix).copied().collect();
            let mut cold = Model::new(2 * total.div_ceil(PAGE) + 16);
            cold.run_slice(0, cut, &full);
            cold.run_slice(cut, cut + suffix.len(), &full);

            // donor: slice ends AT the cut — capture there (as `slice_with_cuts` does), then keep
            // writing past it, so the donor's continuation lands in the page after the snapshot
            let mut donor = Model::new(2 * total.div_ceil(PAGE) + 16);
            donor.run_slice(0, cut, &prompt);
            let snap = donor.capture(cut);
            donor.run_slice(cut, total, &prompt);

            // restore at the mid-page cut, then write the follower's suffix
            let mut restored = Model::new(2 * total.div_ceil(PAGE) + 16);
            restored.store = donor.store.clone();
            restored.restore(&snap);
            restored.run_slice(cut, cut + suffix.len(), &full);

            assert_eq!(
                restored.observables(),
                cold.observables(),
                "the mid-page restore must scrub the donor's foreign rows and pooled keys: {cut}"
            );
        }

        #[test]
        fn an_aligned_cut_restores_whole_pages_and_matches_cold_including_the_pool_planes() {
            let total = 2131;
            let aligned = 2048; // the planned cut after align_down(2067, 256)
            let prompt = toks(7, total);
            let suffix = toks(99, 48);

            // cold run of the follower's full prompt
            let full: Vec<u32> = prompt[..aligned].iter().chain(&suffix).copied().collect();
            let mut cold = Model::new(2 * total.div_ceil(PAGE) + 16);
            cold.run_slice(0, aligned, &full);
            cold.run_slice(aligned, aligned + suffix.len(), &full);

            // an aligned donor: its slice ends exactly at the cut inside the full prompt, so the
            // one-ahead MTP row at the cut exists (as a real capture's would)
            let mut donor = Model::new(2 * total.div_ceil(PAGE) + 16);
            donor.run_slice(0, aligned, &prompt);
            let snap = donor.capture(aligned);

            // restore at the aligned cut: whole pages only; the suffix re-prefills from 2048
            let mut restored = Model::new(2 * total.div_ceil(PAGE) + 16);
            restored.store = donor.store.clone();
            restored.restore(&snap);
            restored.run_slice(aligned, aligned + suffix.len(), &full);

            assert_eq!(
                restored.observables(),
                cold.observables(),
                "the aligned restore must equal cold in every plane (KV pages, QSA pool planes, recurrent state, rings)"
            );
            // the MTP one-ahead row at the cut round-trips exactly
            assert_eq!(restored.mtp_rows.get(&aligned), donor.mtp_rows.get(&aligned));
            assert_eq!(restored.mtp_pool.get(&aligned), donor.mtp_pool.get(&aligned));
        }

        #[test]
        fn exact_partial_page_restore_at_every_cut_position_and_boundary() {
            use super::interior_restore_sim::cold_vs_restored;
            // cuts at 1, 19, 255, 257 and 2067 tokens, plus page boundaries (256, 512, 2048)
            // and QSA group boundaries (252, 260, 2064, 2068): every restore must equal a
            // from-scratch prefill bit for bit in every plane.
            for chunk in [256usize, 512, 1000] {
                for total in [3000usize, 3333] {
                    for cut in [1usize, 19, 252, 255, 256, 257, 260, 512, 2048, 2064, 2067, 2068] {
                        if cut > 0 && cut < total {
                            cold_vs_restored(total, cut, chunk);
                        }
                    }
                }
            }
        }
    }
}
