//! The prefix cache's side of the GLM model: a pool of snapshot slots allocated once at boot, the snapshot of a slot's state after a prompt chunk, and
//! the restore that puts another slot in that state (design: `docs/PREFIX_CACHE.md`, "GLM-5.3-Flash"; the arithmetic is `families/glm/prefix.rs`).
//!
//! A snapshot is taken between two prefill chunks, when the slot's state is quiet (a chunk commits every row it ran), so that every copy is a plain
//! device-to-device copy of what the chunk left. It holds what the pages do not: the current KDA state buffer and the conv history of each KDA layer, the
//! raw rows of the incomplete pool group of each indexer layer (the 11 MLA layers and the NextN layer's own), and the NextN drafter's carry. A restore
//! starts the slot over (`begin_slot`), shares the snapshot's full pages by reference, copies the partial last page (latent cache AND pool plane of every
//! layer, the NextN layer's included: copy-on-write at attach), and writes the rest back. Nothing here is allocated while serving.
//!
//! A model with the DFlash2 drafter (rank 0 of a group, or either rank of a split drafter) also keeps the drafter's context window with each
//! snapshot: the ring rows of the last [`DraftCtxLayout::WINDOW`] positions before the cut (`Draft::save_window`; a split drafter's rows are its
//! kv heads'), put back by the restore (`Draft::load_window`), so drafts after a hit read what a cold prefill of the same tokens leaves. It is
//! this rank's alone: the slot bytes two ranks compare are the trunk's ([`GlmModel::prefix_snapshot_bytes`]), and the content digest does not
//! cover it (a draft only proposes: it moves no committed bit).
//!
//! Why the result is bit-identical to a cold prefill of the same tokens: every row a chunk writes (KV latent, pooled keys, the NextN entries) is a
//! function of its token and the rows before it, whatever the chunk grid (`docs/GLM53.md`, "Prefill"); the recurrent state and the conv history are the
//! committed state after those rows; the only other thing the next row reads is the ring tail and the drafter's carry, which are copied.

use super::{DevBuf, GlmModel, SpliceOp, TapBook};
use crate::draft::Draft;
use crate::error::ModelError;
use crate::families::glm::alloc::MTP_TAP_CAP;
use crate::families::glm::config::GlmConfig;
use crate::families::glm::prefix::{self, SnapLayout, MAX_TAIL_ROWS, RING_ROW_BYTES};
use crate::forward::Candidates;
use crate::kvtables::BLOCK_TOKENS;
use gb_cuda::dflash2::DraftCtxLayout;
use gb_cuda::{dsa, kda, Stream};

/// What a snapshot slot says about itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapMeta {
    /// Tokens of the prompt the snapshot is the state after.
    pub len: usize,
    /// The drafter's bookkeeping was sound when the snapshot was taken (`TapBook { ok, first: len, n: 1 }`); a restore of a snapshot that says `false`
    /// switches the slot's drafting off (a draft only proposes: nothing is lost in exactness).
    pub mtp_ok: bool,
    /// The snapshot's content digest ([`GlmModel::seal_snapshot`]), once sealed: a restore of it must leave the slot in a state of the same
    /// digest ([`GlmModel::prefix_state_digest`]), or it is refused and the prompt prefilled cold.
    pub digest: Option<u64>,
    /// The drafter's context window kept with the snapshot: positions `lo..lo + n` (`None`: no drafter, or the ring did not hold them).
    pub win: Option<(usize, usize)>,
}

/// A snapshot to take INSIDE the prefill chunk being run (`GlmModel::prefill_chunk_cuts`): the state of the slot after the first `row` rows of the chunk
/// (`at` tokens of the prompt) goes to snapshot slot `snap`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InChunkCut {
    pub row: usize,
    pub at: usize,
    pub snap: u32,
    pub slot: usize,
}

/// The snapshot slots of the model.
pub struct SnapPool {
    pub slots: usize,
    layout: SnapLayout,
    /// Per KDA layer: `[slots][state buffer]` fp32 and `[slots][conv]` bf16.
    kda_s: Vec<DevBuf>,
    kda_conv: Vec<DevBuf>,
    /// `[slots][ring layers][MAX_TAIL_ROWS][256]` fp16: the indexer layers' ring tails, the NextN layer's last.
    ring: DevBuf,
    /// `[slots][hidden]` fp16: the drafter's carry (only with the NextN layer).
    carry: Option<DevBuf>,
    /// Per DFlash2 draft layer, `[slots][WINDOW]` ring rows of K and of V (only on a model with the drafter).
    win: Option<(Vec<DevBuf>, Vec<DevBuf>)>,
    free: Vec<u32>,
    meta: Vec<Option<SnapMeta>>,
}

impl SnapPool {
    /// Bytes of one snapshot slot: the memory plan and the allocation both use this (`prefix::snapshot_bytes`).
    pub fn bytes_per_slot(cfg: &GlmConfig, mtp: bool) -> Result<u64, ModelError> {
        prefix::snapshot_bytes(cfg, mtp).map_err(ModelError::Config)
    }

    /// The pool of `slots` snapshot slots of the trunk's state, each with the DFlash2 drafter's context window when `dflash_row` is the
    /// drafter's ring row bytes (`Draft::ring_row_bytes`; a model that drafts with it).
    pub fn new(cfg: &GlmConfig, slots: usize, mtp: bool, dflash_row: Option<usize>) -> Result<Self, ModelError> {
        if slots == 0 {
            return Err(ModelError::Config("a prefix cache needs at least one snapshot slot".into()));
        }
        let layout = SnapLayout::of(cfg, mtp).map_err(ModelError::Config)?;
        // every slot is written by a snapshot before a restore reads it (`SnapMeta` says which hold one): nothing is zeroed at boot
        let (mut kda_s, mut kda_conv) = (Vec::new(), Vec::new());
        for _ in 0..layout.kda_layers {
            kda_s.push(DevBuf::alloc(slots * layout.kda_state as usize)?);
            kda_conv.push(DevBuf::alloc(slots * layout.kda_conv as usize)?);
        }
        let ring = DevBuf::alloc(slots * layout.ring_bytes() as usize)?;
        let carry = if mtp { Some(DevBuf::alloc(slots * layout.carry as usize)?) } else { None };
        let win = if let Some(row) = dflash_row {
            let per_layer = slots * DraftCtxLayout::WINDOW * row;
            let (mut k, mut v) = (Vec::new(), Vec::new());
            for _ in 0..crate::draft::LAYERS {
                k.push(DevBuf::alloc(per_layer)?);
                v.push(DevBuf::alloc(per_layer)?);
            }
            Some((k, v))
        } else {
            None
        };
        Ok(Self {
            slots,
            layout,
            kda_s,
            kda_conv,
            ring,
            carry,
            win,
            free: (0..slots as u32).rev().collect(),
            meta: vec![None; slots],
        })
    }

    /// Bytes allocated on the device for the trunk's state (what two ranks of a group compare: the drafter's window is rank 0's alone).
    pub fn allocated_bytes(&self) -> u64 {
        let sum = |v: &[DevBuf]| v.iter().map(|b| b.bytes() as u64).sum::<u64>();
        sum(&self.kda_s) + sum(&self.kda_conv) + self.ring.bytes() as u64 + self.carry.as_ref().map_or(0, |c| c.bytes() as u64)
    }

    /// Bytes allocated on the device for the drafter's windows (0 without the drafter).
    pub fn window_bytes(&self) -> u64 {
        self.win.as_ref().map_or(0, |(k, v)| k.iter().chain(v).map(|b| b.bytes() as u64).sum())
    }

    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// Take a free slot, or `None`.
    pub fn alloc(&mut self) -> Option<u32> {
        self.free.pop()
    }

    /// Give a slot back (a capture that was never finished, or an evicted entry).
    pub fn free_slot(&mut self, s: u32) {
        assert!((s as usize) < self.slots && !self.free.contains(&s), "snapshot slot {s} freed twice or out of range");
        self.meta[s as usize] = None;
        self.free.push(s);
    }

    pub fn meta(&self, s: u32) -> Option<SnapMeta> {
        self.meta.get(s as usize).copied().flatten()
    }

    /// The slots were laid out for the KDA pools the model runs: the copies stride by the live `dims`, the slots by the config the pool was made
    /// from, and the two part when a model loaded whole is split by heads later (`enable_head_split` gives the rank 32-head pools).
    pub(super) fn check_dims(&self, d: kda::Dims) -> Result<(), ModelError> {
        let live = ((d.s_buf_elems() * 4) as u64, (d.conv_slot_elems() * 2) as u64);
        if live != (self.layout.kda_state, self.layout.kda_conv) {
            return Err(ModelError::Config(format!(
                "the snapshot slots hold KDA state of {} + {} bytes but the model's pools hold {} + {} (a head split after the pool was \
                 made): build the rank from its head strips, or enable the prefix cache after the split",
                self.layout.kda_state, self.layout.kda_conv, live.0, live.1
            )));
        }
        Ok(())
    }

    /// Copy the current state buffer and the conv history of KDA layer `ord` for `slot` into snapshot `snap`. Synchronises: which of a slot's two state
    /// buffers is current is the parity word's, which lives on the device and says how many commits the prompt has taken, that is, depends on the
    /// chunk grid, so the copy is of the buffer it names and the host reads it.
    pub(super) fn save_kda(
        &mut self,
        snap: u32,
        ord: usize,
        pool: &kda::Pool,
        slot: usize,
        dims: kda::Dims,
        st: &Stream,
    ) -> Result<(), ModelError> {
        st.sync()?;
        let (e, ce) = (dims.s_buf_elems() * 4, dims.conv_slot_elems() * 2);
        let mut par = [0i32; 1];
        pool.parity.download_range(slot * 4, &mut par)?;
        let cur = (par[0] & 1) as usize;
        self.kda_s[ord].copy_range_from(snap as usize * e, &pool.s, (slot * 2 + cur) * e, e, st)?;
        self.kda_conv[ord].copy_range_from(snap as usize * ce, &pool.conv, slot * ce, ce, st)?;
        Ok(())
    }

    /// Copy the raw rows of the incomplete pool group of a prefix of `at` tokens (positions `at - at % 4 .. at`, none when `at` is a multiple of 4)
    /// from `slot`'s ring in `pl` into the tail of indexer layer `layer` (the NextN layer's is the one after the MLA layers) of snapshot `snap`.
    pub(super) fn save_ring(
        &mut self,
        snap: u32,
        layer: usize,
        pl: &dsa::Planes,
        slot: usize,
        at: usize,
        st: &Stream,
    ) -> Result<(), ModelError> {
        let (row0, n) = prefix::ring_tail_rows(at);
        if n == 0 {
            return Ok(());
        }
        let off = self.ring_off(snap, layer);
        self.ring.copy_range_from(off, &pl.ring, (slot * pl.ring_len + row0) * RING_ROW_BYTES, n * RING_ROW_BYTES, st)?;
        Ok(())
    }

    /// Byte offset of the ring tail of indexer layer `layer` in snapshot `snap`.
    fn ring_off(&self, snap: u32, layer: usize) -> usize {
        (snap as usize * self.layout.ring_layers + layer) * MAX_TAIL_ROWS * RING_ROW_BYTES
    }

    /// Copy `draft`'s context ring rows of positions `lo..lo + n` of `slot` (`win`) into snapshot `snap`'s window, and return what it holds:
    /// `None` (nothing copied) without a drafter, a pool without windows or no window to keep.
    pub(super) fn save_window(
        &mut self,
        snap: u32,
        draft: Option<&Draft>,
        slot: usize,
        win: Option<(usize, usize)>,
        st: &Stream,
    ) -> Result<Option<(usize, usize)>, ModelError> {
        let (Some((k, v)), Some(d), Some((lo, n))) = (self.win.as_mut(), draft, win) else { return Ok(None) };
        d.save_window(slot, lo, n, k, v, snap as usize, st)?;
        Ok(Some((lo, n)))
    }
}

/// Everything a prompt leaves in a slot that a cached run must reproduce, read back to the host (tests compare two of them bit for bit).
#[derive(Debug, PartialEq, Eq, Default)]
pub struct PrefixDigest {
    /// Per KDA layer: the current state's fp32 bits and the conv history.
    pub kda: Vec<(Vec<u32>, Vec<u16>)>,
    /// Per MLA layer: latent rows `0..len`, pool rows `0..len / 4`, and the raw rows of the incomplete group.
    pub latent: Vec<Vec<u16>>,
    pub pools: Vec<Vec<u16>>,
    pub ring: Vec<Vec<u16>>,
    /// The NextN layer's: latent rows, pool rows, ring tail; the carry (the store's row 0) and the drafter's bookkeeping `(ok, first, n)`.
    pub mtp_latent: Vec<u16>,
    pub mtp_pools: Vec<u16>,
    pub mtp_ring: Vec<u16>,
    pub mtp_carry: Vec<u16>,
    pub mtp_book: Option<(bool, usize, usize)>,
}

impl PrefixDigest {
    /// Names what differs from `other` (empty when nothing does).
    pub fn describe_diff(&self, other: &PrefixDigest) -> String {
        let mut d = Vec::new();
        let first = |a: &[Vec<u16>], b: &[Vec<u16>]| a.iter().zip(b).position(|(x, y)| x != y);
        if let Some(l) = self.kda.iter().zip(&other.kda).position(|(x, y)| x.0 != y.0) {
            d.push(format!("KDA state of KDA layer #{l}"));
        }
        if let Some(l) = self.kda.iter().zip(&other.kda).position(|(x, y)| x.1 != y.1) {
            d.push(format!("KDA conv of KDA layer #{l}"));
        }
        if let Some(l) = first(&self.latent, &other.latent) {
            d.push(format!("MLA latent rows of MLA layer #{l}"));
        }
        if let Some(l) = first(&self.pools, &other.pools) {
            d.push(format!("DSA pool rows of MLA layer #{l}"));
        }
        if let Some(l) = first(&self.ring, &other.ring) {
            d.push(format!("DSA ring tail of MLA layer #{l}"));
        }
        for (name, a, b) in [
            ("NextN latent rows", &self.mtp_latent, &other.mtp_latent),
            ("NextN pool rows", &self.mtp_pools, &other.mtp_pools),
            ("NextN ring tail", &self.mtp_ring, &other.mtp_ring),
            ("NextN carry", &self.mtp_carry, &other.mtp_carry),
        ] {
            if a != b {
                d.push(name.to_string());
            }
        }
        if self.mtp_book != other.mtp_book {
            d.push(format!("NextN bookkeeping {:?} vs {:?}", self.mtp_book, other.mtp_book));
        }
        d.join("; ")
    }
}

impl GlmModel {
    /// The drafter's context window of a snapshot at `len` when the ring holds it: positions `len - WINDOW .. len` (from 0 for a shorter prefix)
    /// were absorbed by this prompt (or put back with the prefix it resumed from) and no row absorbed since, up to position `latest`, has
    /// overwritten them. `None` without a drafter or when the ring does not hold them (a restore of the snapshot then leaves the ring as it is).
    fn dflash_window_of(&self, slot: usize, len: usize, latest: usize) -> Option<(usize, usize)> {
        self.dflash.as_ref()?;
        let lo = len.saturating_sub(DraftCtxLayout::WINDOW);
        (lo >= self.dflash_kept_from(slot) && latest - lo <= DraftCtxLayout::DEFAULT_CAP).then_some((lo, len - lo))
    }
    /// Allocate the snapshot pool: `slots` snapshot slots of [`SnapPool::bytes_per_slot`] bytes each, with the NextN layer's share when the model has the
    /// layer, and the DFlash2 drafter's context window ([`Draft::window_snapshot_bytes_of`] of its share) when it drafts with one (attach the
    /// drafter first). Errors when a pool exists already.
    pub fn enable_prefix_cache(&mut self, slots: usize) -> Result<(), ModelError> {
        if self.snaps.is_some() {
            return Err(ModelError::Config("the prefix cache is already enabled".into()));
        }
        let row = self.dflash.as_ref().map(|d| d.draft.ring_row_bytes());
        let pool = SnapPool::new(&self.cfg, slots, self.mtp.is_some(), row)?;
        pool.check_dims(self.kda_dims)?;
        self.snaps = Some(pool);
        Ok(())
    }

    /// Bytes of the snapshot pool's drafter windows (0 without a pool or a drafter): rank 0's alone in a group.
    pub fn prefix_window_bytes(&self) -> u64 {
        self.snaps.as_ref().map_or(0, |s| s.window_bytes())
    }

    /// Pages of the KV pool no slot and no snapshot holds.
    pub fn kv_free_pages(&self) -> usize {
        self.kv_tab.free_blocks()
    }

    /// Pages `slot`'s table holds (its own and those it shares with a snapshot it was restored from).
    pub fn kv_slot_pages(&self, slot: usize) -> usize {
        self.kv_tab.blocks_of(slot).len()
    }

    /// Whether the snapshot pool exists (`enable_prefix_cache`).
    pub fn has_prefix_cache(&self) -> bool {
        self.snaps.is_some()
    }

    /// Whether a prefill chunk of more than 128 rows can take snapshots inside itself: the snapshot pool exists and the model has the large-row scratch.
    pub fn can_snapshot_inside_chunks(&self) -> bool {
        self.snaps.is_some() && self.s.pf.is_some()
    }

    /// Snapshot slots no capture holds right now.
    pub fn prefix_free_snapshots(&self) -> usize {
        self.snaps.as_ref().map_or(0, |s| s.free_count())
    }

    /// Bytes of one slot of the snapshot pool's trunk state as it was built (0 without one; the drafter's window is not in it): a rank of a
    /// KDA head split holds its own heads' state (`SnapPool::check_dims`), so this is what two ranks of a group compare
    /// (`tp::GlmCall::EnableSnapshots`).
    pub fn prefix_snapshot_bytes(&self) -> u64 {
        self.snaps.as_ref().map_or(0, |s| s.allocated_bytes() / s.slots as u64)
    }

    /// What snapshot slot `snap` says about itself (`None`: it holds no snapshot).
    pub fn prefix_snapshot_meta(&self, snap: u32) -> Option<SnapMeta> {
        self.snaps.as_ref().and_then(|s| s.meta(snap))
    }

    /// Take a free snapshot slot.
    pub fn alloc_snapshot(&mut self) -> Option<u32> {
        self.snaps.as_mut().and_then(|s| s.alloc())
    }

    /// Copy the state of `slot`, which holds exactly the first `len` tokens of its prompt (a prefill chunk just ended at `len`), into snapshot slot
    /// `snap`. Synchronises (the parity words of the KDA pools are read on the host: which of a slot's two state buffers is current depends on how
    /// many commits the prompt took, that is, on the chunk grid, so the copy cannot be of raw pool bytes).
    pub fn prefix_snapshot(&mut self, slot: usize, snap: u32, len: usize) -> Result<(), ModelError> {
        if len == 0 {
            return Err(ModelError::Request("a snapshot of no tokens".into()));
        }
        // under P2 the state after `len` prompt rows is only resumable with the fresh prefill's bits when `len` is on its segment's 16-row grid
        if self.prefill == crate::families::glm::contract::PrefillContract::P2 {
            let c = gb_cuda::kda::P2_CHUNK_ROWS;
            match self.p2_seg[slot] {
                Some((a, next)) if next == len && (len - a) % c == 0 => {}
                seg => {
                    return Err(ModelError::Request(format!(
                        "{}=p2: a snapshot of {len} tokens of slot {slot} must follow a prompt chunk that ends a whole number of {c}-row KDA \
                         chunks after its segment's first row (segment {seg:?}; docs/GLM_PREFILL_P2.md 8.1)",
                        crate::families::glm::contract::CONTRACT_VAR
                    )))
                }
            }
        }
        // the previous window's commits are applied first and the stream is quiet: the state read below is the committed state
        self.flush()?;
        self.stream.sync()?;
        let mtp_ok = match self.mtp.as_ref() {
            Some(m) => {
                let b = m.book(slot);
                b.ok && b.first == len && b.n == 1
            }
            None => true,
        };
        let win = self.dflash_window_of(slot, len, len);
        let GlmModel { pools, planes, mtp, snaps, stream, kda_dims, dflash, .. } = self;
        let sp = snaps.as_mut().ok_or_else(|| ModelError::Request("the prefix cache is not enabled".into()))?;
        if snap as usize >= sp.slots {
            return Err(ModelError::Request(format!("snapshot slot {snap} is out of range")));
        }
        sp.check_dims(*kda_dims)?;
        for (ord, pool) in pools.iter().enumerate() {
            sp.save_kda(snap, ord, pool, slot, *kda_dims, stream)?;
        }
        let n_mla = planes.len();
        for (i, pl) in planes.iter().enumerate() {
            sp.save_ring(snap, i, pl, slot, len, stream)?;
        }
        if let Some(m) = mtp.as_ref() {
            sp.save_ring(snap, n_mla, m.planes(), slot, len, stream)?;
        }
        if let (Some(m), Some(c)) = (mtp.as_ref(), sp.carry.as_mut()) {
            let h2 = c.bytes() / sp.slots;
            c.copy_range_from(snap as usize * h2, m.store(), slot * MTP_TAP_CAP * h2, h2, stream)?;
        }
        let win = sp.save_window(snap, dflash.as_ref().map(|d| &d.draft), slot, win, stream)?;
        sp.meta[snap as usize] = Some(SnapMeta { len, mtp_ok, digest: None, win });
        stream.sync()?;
        Ok(())
    }

    /// [`GlmModel::prefill_chunk_mm`] that also takes a snapshot of `slot`'s state after exactly `at` tokens of the prompt for each of `cuts` (`(at, snapshot
    /// slot)`, ascending, each strictly inside the chunk `start .. start + tokens.len()`): no extra forward and no extra pass over the weights. The chunk
    /// (of at least two rows, run on the large-row kernels whatever its rows: they give every row the decode kernels' bits, and cut its attention into slabs) is run as before, with its KDA
    /// layers' row-by-row path in segments that end at each cut (a segment boundary changes no row's bits), the state copied out between two segments, the
    /// indexer's ring tail copied right after the slab that holds the cut's last row (the ring holds 256 raw rows, a slab 128), and the drafter's carry
    /// (the tap of the cut's last row) copied out of the chunk's taps afterwards. The bits of the prompt are those of the same chunk without cuts. A cut at
    /// the chunk's END is not an in-chunk cut: it is [`GlmModel::prefix_snapshot`] after the chunk.
    pub fn prefill_chunk_cuts(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        want: Option<usize>,
        splice: Vec<SpliceOp>,
        cuts: &[(usize, u32)],
    ) -> Result<Option<Candidates>, ModelError> {
        let rows = tokens.len();
        let bad = |why: String| Err(ModelError::Request(format!("snapshots inside a chunk: {why}")));
        if self.snaps.is_none() {
            return bad("the prefix cache is not enabled".into());
        }
        // a rank of a KDA head split runs each KDA layer's chunk whole on its heads (`model.rs`, the split arm reads no cut), so a copy taken "inside"
        // the chunk would be the state after the chunk: such a rank ends its slices at the cuts and snapshots between chunks (`prefix_snapshot`)
        if self.cfg.kda.rank.is_some() || self.tp.as_ref().is_some_and(|t| t.split_bits() & super::tp::SPLIT_KDA != 0) {
            return bad("a rank of a KDA head split takes snapshots only between chunks: end the slice at the cut".into());
        }
        if rows < 2 || self.s.pf.is_none() {
            return bad(format!("a chunk of {rows} rows has no row to cut inside, or the model was built for chunks of 128 rows or fewer (no large-row scratch: its chunks run the decode kernels, whose state is only taken between chunks)"));
        }
        let slots = self.snaps.as_ref().map_or(0, |s| s.slots);
        let ok = cuts.windows(2).all(|w| w[0].0 < w[1].0)
            && cuts.iter().all(|&(at, snap)| at > start && at < start + rows && (snap as usize) < slots)
            && cuts.iter().enumerate().all(|(i, c)| cuts[..i].iter().all(|d| d.1 != c.1));
        if !ok {
            return bad(format!("cuts {cuts:?} for the chunk {start}..{}", start + rows));
        }
        self.snap_cuts = cuts.iter().map(|&(at, snap)| InChunkCut { row: at - start, at, snap, slot }).collect();
        let r = self.prefill_chunk_mm(slot, start, tokens, want, splice);
        let taken = std::mem::take(&mut self.snap_cuts);
        let c = r?;
        // what only the whole chunk can say: the drafter's carry (the tap of each cut's last row, in the chunk's tap buffer until the next forward) and
        // whether its bookkeeping was sound
        let mtp_ok = match self.mtp.as_ref() {
            Some(m) => {
                let b = m.book(slot);
                b.ok && b.first == start + rows && b.n == 1
            }
            None => true,
        };
        // the drafter's window of each cut, when the chunk's later rows left it in the ring (the chunk's rows were absorbed after its forward)
        let wins: Vec<Option<(usize, usize)>> = taken.iter().map(|c| self.dflash_window_of(slot, c.at, start + rows)).collect();
        let GlmModel { mtp, snaps, stream, dflash, .. } = self;
        let sp = snaps.as_mut().expect("checked above");
        for (cut, win) in taken.iter().zip(wins) {
            if let (Some(m), Some(c)) = (mtp.as_ref(), sp.carry.as_mut()) {
                let h2 = c.bytes() / sp.slots;
                c.copy_range_from(cut.snap as usize * h2, &m.tap, (cut.row - 1) * h2, h2, stream)?;
            }
            let win = sp.save_window(cut.snap, dflash.as_ref().map(|d| &d.draft), slot, win, stream)?;
            sp.meta[cut.snap as usize] = Some(SnapMeta { len: cut.at, mtp_ok, digest: None, win });
        }
        stream.sync()?;
        Ok(c)
    }

    /// Take references to the KV pages of the first `len` tokens of `slot` (a snapshot at `len` keeps them alive after the slot is gone) and return them.
    /// The last page is partial when `len` is not a multiple of the page size; the slot goes on writing into it above `len`, which the snapshot never
    /// reads, and a restore copies it (copy-on-write at attach).
    pub fn retain_prefix_pages(&mut self, slot: usize, len: usize) -> Vec<u32> {
        self.kv_tab.retain_prefix(slot, len.div_ceil(BLOCK_TOKENS))
    }

    /// Give back a snapshot: its page references and its slot.
    pub fn release_prefix(&mut self, snap: u32, pages: &[u32]) {
        self.kv_tab.release_pages(pages);
        if let Some(sn) = self.snaps.as_mut() {
            sn.free_slot(snap);
        }
    }

    /// Copy page `src` over page `dst`: the latent cache (its values, and an fp8 cache's scales: the whole page) and the pool plane of every
    /// layer, the NextN layer's included.
    fn copy_page(&mut self, src: usize, dst: usize) -> Result<(), ModelError> {
        let GlmModel { latent, planes, mtp, stream, .. } = self;
        let pb = dsa::POOLS_PER_PAGE * dsa::D * 2;
        for l in latent.iter_mut() {
            let lb = l.page_bytes();
            l.lat.copy_within(dst * lb, src * lb, lb, stream)?;
        }
        for p in planes.iter_mut() {
            p.pool.copy_within(dst * pb, src * pb, pb, stream)?;
        }
        if let Some(m) = mtp.as_mut() {
            let l = m.latent_mut();
            let lb = l.page_bytes();
            l.lat.copy_within(dst * lb, src * lb, lb, stream)?;
            m.planes_mut().pool.copy_within(dst * pb, src * pb, pb, stream)?;
        }
        Ok(())
    }

    /// Put `slot` in the state a prefill of the snapshot's first `len` tokens would have left it in (see `gb_engine::Backend::prefix_restore`): the
    /// slot starts over, the snapshot's full pages are shared by reference, its partial last page is copied into a page the slot owns, and the KDA
    /// state, the conv history, the ring tails and the drafter's carry are written back. Stream-ordered after what the slot's previous owner left.
    pub fn restore_prefix(&mut self, slot: usize, snap: u32, pages: &[u32], len: usize) -> Result<(), ModelError> {
        let meta = self
            .snaps
            .as_ref()
            .and_then(|sn| sn.meta(snap))
            .filter(|m| m.len == len && pages.len() == len.div_ceil(BLOCK_TOKENS))
            .ok_or_else(|| ModelError::Request(format!("snapshot {snap} is not a snapshot of {len} tokens")))?;
        self.snaps.as_ref().expect("checked above").check_dims(self.kda_dims)?;
        // begin_slot flushes the pending commits, zeroes the slot's KDA state and conv history, gives its pages back and resets the drafter's books
        self.begin_slot(slot)?;
        let (full, partial) = prefix::split_pages(len);
        self.kv_tab.adopt_shared(slot, &pages[..full]);
        if partial {
            let src = pages[full] as usize;
            let dst = self
                .kv_tab
                .alloc_block(slot)
                .map_err(|e| ModelError::Request(format!("no page for the copy of the last prefix page: {e:?}")))?
                as usize;
            self.copy_page(src, dst)?;
        }
        // which buffer of the slot is current (begin_slot leaves the parity word alone, and the stream must be done with the commits it flushed)
        self.stream.sync()?;
        let GlmModel { pools, planes, mtp, snaps, stream, kda_dims, dflash, .. } = self;
        let sp = snaps.as_mut().expect("checked above");
        let (e, ce) = (kda_dims.s_buf_elems() * 4, kda_dims.conv_slot_elems() * 2);
        for (ord, pool) in pools.iter_mut().enumerate() {
            let mut par = [0i32; 1];
            pool.parity.download_range(slot * 4, &mut par)?;
            let cur = (par[0] & 1) as usize;
            pool.s.copy_range_from((slot * 2 + cur) * e, &sp.kda_s[ord], snap as usize * e, e, stream)?;
            pool.conv.copy_range_from(slot * ce, &sp.kda_conv[ord], snap as usize * ce, ce, stream)?;
        }
        let (row0, n) = prefix::ring_tail_rows(len);
        if n > 0 {
            let n_mla = planes.len();
            for (i, pl) in planes.iter_mut().enumerate() {
                let off = sp.ring_off(snap, i);
                let ring_len = pl.ring_len;
                pl.ring.copy_range_from((slot * ring_len + row0) * RING_ROW_BYTES, &sp.ring, off, n * RING_ROW_BYTES, stream)?;
            }
            if let Some(m) = mtp.as_mut() {
                let off = sp.ring_off(snap, n_mla);
                let ring_len = m.planes().ring_len;
                m.planes_mut().ring.copy_range_from(
                    (slot * ring_len + row0) * RING_ROW_BYTES,
                    &sp.ring,
                    off,
                    n * RING_ROW_BYTES,
                    stream,
                )?;
            }
        }
        if let (Some(m), Some(c)) = (mtp.as_mut(), sp.carry.as_ref()) {
            let h2 = c.bytes() / sp.slots;
            m.store_mut().copy_range_from(slot * MTP_TAP_CAP * h2, c, snap as usize * h2, h2, stream)?;
            // the next prompt chunk continues from the carry: entries below `len` are built (they are in the pages), the carry is the tap of `len - 1`
            *m.book_mut(slot) = TapBook { ok: meta.mtp_ok, first: len, n: 1 };
        }
        // the drafter's context window, when the snapshot kept one (`begin_slot` left the ring as the slot's previous owner left it)
        if let (Some((k, v)), Some(d), Some((lo, n))) = (sp.win.as_ref(), dflash.as_mut(), meta.win) {
            d.draft.load_window(slot, lo, n, k, v, snap as usize, stream)?;
        }
        // under P2 the restored state opens the slot's next prefill segment at `len`: the snapshot was taken on its own segment's 16-row grid, so
        // the resumed chunks fall where the fresh prefill's do (`docs/GLM_PREFILL_P2.md` 8.1)
        self.p2_open_segment(slot, len);
        // a sealed snapshot is verified: the slot must now hold exactly what was captured (a copy that did not happen, a stale or reused page,
        // another slot's state: refused, and the engine prefills the prompt cold, which begins the slot again)
        if let Some(want) = meta.digest {
            let got = self.prefix_state_digest(slot, len)?;
            if got != want {
                return Err(ModelError::Request(format!(
                    "the restore of snapshot {snap} ({len} tokens) left a state of digest {got:016x}, the snapshot's is {want:016x}: not used"
                )));
            }
        }
        Ok(())
    }

    /// Seals snapshot `snap` (its capture complete, its pages `pages` retained): computes its content digest from the snapshot slot and the
    /// pages, records it in the slot's [`SnapMeta`] (every restore of it is then verified) and returns it. Call it once the capture's chunk
    /// has ended (an in-chunk cut's rows below the cut are final then).
    ///
    /// The digest covers, in this order: every KDA layer's state buffer and conv history (the whole slot of the arena), each indexer layer's
    /// ring tail rows, the drafter's carry, the partial last page's rows below `len` (latent rows of every MLA layer and the NextN layer's,
    /// then their pool rows) on the device; on the host `len`, the drafter flag, and the id and generation ([`KvTables::generation`]) of each
    /// full page. A full page is shared by reference, never copied, so its CONTENT is not read: its id and generation say it is the page the
    /// snapshot retained and that it was not freed and handed out again since (what this does not see: a write into a shared full page
    /// through a slot that holds it, which the copy-on-write at attach exists to prevent).
    pub fn seal_snapshot(&mut self, snap: u32, pages: &[u32]) -> Result<u64, ModelError> {
        self.stream.sync()?;
        let (e, ce) = (self.kda_dims.s_buf_elems() * 4, self.kda_dims.conv_slot_elems() * 2);
        let sp = self.snaps.as_ref().ok_or_else(|| ModelError::Request("the prefix cache is not enabled".into()))?;
        let meta = sp.meta(snap).ok_or_else(|| ModelError::Request(format!("snapshot slot {snap} holds no snapshot to seal")))?;
        let s = snap as usize;
        let mut ranges: Vec<(*const std::ffi::c_void, usize)> = Vec::new();
        // exactly the layers `prefix_snapshot` saves and `restore_prefix` writes back: the model's KDA pools and indexer planes (the arena
        // may be laid out for more, from the config, than a model of a layer prefix runs)
        for ord in 0..self.pools.len() {
            ranges.push((sp.kda_s[ord].ptr_at(s * e) as *const _, e));
            ranges.push((sp.kda_conv[ord].ptr_at(s * ce) as *const _, ce));
        }
        let (_, n) = prefix::ring_tail_rows(meta.len);
        for layer in 0..self.planes.len() + usize::from(self.mtp.is_some()) {
            ranges.push((sp.ring.ptr_at(sp.ring_off(snap, layer)) as *const _, n * RING_ROW_BYTES));
        }
        if let Some(c) = sp.carry.as_ref() {
            let h2 = c.bytes() / sp.slots;
            ranges.push((c.ptr_at(s * h2) as *const _, h2));
        }
        let d = self.state_digest_of(ranges, pages, meta.len, meta.mtp_ok)?;
        if let Some(m) = self.snaps.as_mut().and_then(|sp| sp.meta.get_mut(s)).and_then(|m| m.as_mut()) {
            m.digest = Some(d);
        }
        Ok(d)
    }

    /// The content digest of `slot`'s LIVE state as a prefix of `len` tokens, in [`GlmModel::seal_snapshot`]'s order and lengths: the KDA
    /// layers' current state buffers and conv histories, the ring tail rows, the drafter's carry and flag, the slot's pages. Equal to the
    /// snapshot's digest after a restore of it exactly when the restore wrote back what was captured. Synchronises.
    pub fn prefix_state_digest(&mut self, slot: usize, len: usize) -> Result<u64, ModelError> {
        if slot >= self.slots || len == 0 {
            return Err(ModelError::Request(format!("a prefix state digest of slot {slot} at {len} tokens")));
        }
        self.stream.sync()?;
        let (e, ce) = (self.kda_dims.s_buf_elems() * 4, self.kda_dims.conv_slot_elems() * 2);
        let mut ranges: Vec<(*const std::ffi::c_void, usize)> = Vec::new();
        for pool in &self.pools {
            let mut par = [0i32; 1];
            pool.parity.download_range(slot * 4, &mut par)?;
            let cur = (par[0] & 1) as usize;
            ranges.push((pool.s.ptr_at((slot * 2 + cur) * e) as *const _, e));
            ranges.push((pool.conv.ptr_at(slot * ce) as *const _, ce));
        }
        let (row0, n) = prefix::ring_tail_rows(len);
        for pl in self.planes.iter().chain(self.mtp.as_ref().map(|m| m.planes())) {
            ranges.push((pl.ring.ptr_at((slot * pl.ring_len + row0) * RING_ROW_BYTES) as *const _, n * RING_ROW_BYTES));
        }
        let mtp_ok = match (self.mtp.as_ref(), self.snaps.as_ref().and_then(|sp| sp.carry.as_ref().map(|c| c.bytes() / sp.slots)))
        {
            (Some(m), Some(h2)) => {
                ranges.push((m.store().ptr_at(slot * MTP_TAP_CAP * h2) as *const _, h2));
                let b = m.book(slot);
                b.ok && b.first == len && b.n == 1
            }
            (Some(m), None) => {
                let b = m.book(slot);
                b.ok && b.first == len && b.n == 1
            }
            (None, _) => true,
        };
        let pages = self.kv_tab.blocks_of(slot).to_vec();
        self.state_digest_of(ranges, &pages, len, mtp_ok)
    }

    /// The pages' part of a prefix state digest, then the digest itself (see [`GlmModel::seal_snapshot`]).
    fn state_digest_of(
        &self,
        mut ranges: Vec<(*const std::ffi::c_void, usize)>,
        pages: &[u32],
        len: usize,
        mtp_ok: bool,
    ) -> Result<u64, ModelError> {
        let (full, partial) = prefix::split_pages(len);
        if pages.len() < full + usize::from(partial) {
            return Err(ModelError::Request(format!("{} pages for a prefix of {len} tokens", pages.len())));
        }
        let page = partial.then(|| pages[full] as usize);
        let rows = len % BLOCK_TOKENS;
        let pb = dsa::POOLS_PER_PAGE * dsa::D * 2;
        let pool_rows = (len / dsa::P).saturating_sub(full * dsa::POOLS_PER_PAGE);
        let none = (std::ptr::null(), 0usize);
        for l in self.latent.iter().chain(self.mtp.as_ref().map(|m| m.latent())) {
            // the rows' values, then (fp8) their scales; a whole prefix (no partial page) keeps one empty range per layer
            match page {
                Some(p) => {
                    ranges.extend(l.row_ranges(p, rows).into_iter().map(|(off, n)| (l.lat.ptr_at(off) as *const _, n)))
                }
                None => ranges.push(none),
            }
        }
        for pl in self.planes.iter().chain(self.mtp.as_ref().map(|m| m.planes())) {
            ranges.push(page.map_or(none, |p| (pl.pool.ptr_at(p * pb) as *const _, pool_rows * dsa::D * 2)));
        }
        let mut host = Vec::with_capacity(9 + full * 8);
        host.extend_from_slice(&(len as u64).to_le_bytes());
        host.push(u8::from(mtp_ok));
        for &p in &pages[..full] {
            host.extend_from_slice(&p.to_le_bytes());
            host.extend_from_slice(&self.kv_tab.generation(p).to_le_bytes());
        }
        let dev = gb_cuda::state_digest::sum_device(&ranges, &self.stream)?;
        let sum = dev.wrapping_add(gb_cuda::state_digest::sum_host(ranges.len(), &host));
        Ok(gb_cuda::state_digest::finish(sum, ranges.iter().map(|r| r.1).chain([host.len()])))
    }

    // ------------------------------------------------------------------------------------------------------------------------ state readout

    /// The raw rows `from..from + n` of the indexer ring of MLA layer `l` for `slot` (fp16 bits, `[n][256]`).
    pub fn ring_rows(&mut self, l: usize, slot: usize, from: usize, n: usize) -> Result<Vec<u16>, ModelError> {
        self.stream.sync()?;
        let ord = self.mla_ord[l];
        let pl = &self.planes[ord];
        read_ring(pl, slot, from, n)
    }

    /// Everything the first `len` tokens of `slot` left behind that a cached run must reproduce (see [`PrefixDigest`]); synchronises.
    pub fn prefix_digest(&mut self, slot: usize, len: usize) -> Result<PrefixDigest, ModelError> {
        self.flush()?;
        self.stream.sync()?;
        let mut d = PrefixDigest::default();
        let tail = prefix::ring_tail(len);
        for l in 0..self.layers {
            if self.kda_ord[l] != usize::MAX {
                let (s, c) = self.kda_state(l, slot)?;
                d.kda.push((s.iter().map(|x| x.to_bits()).collect(), c));
            } else if self.mla_ord[l] != usize::MAX {
                d.latent.push(self.latent_rows(l, slot, len)?);
                d.pools.push(self.pool_rows(l, slot, len / dsa::P)?);
                d.ring.push(self.ring_rows(l, slot, tail.start, tail.len())?);
            }
        }
        if let Some(m) = self.mtp.as_ref() {
            let blocks = self.kv_tab.blocks_of(slot).to_vec();
            d.mtp_latent = m.latent().read_rows(&blocks, len)?;
            let pool: Vec<u16> = m.planes().pool.to_vec()?;
            for p in 0..len / dsa::P {
                let at = (blocks[p / dsa::POOLS_PER_PAGE] as usize * dsa::POOLS_PER_PAGE + p % dsa::POOLS_PER_PAGE) * dsa::D;
                d.mtp_pools.extend_from_slice(&pool[at..at + dsa::D]);
            }
            d.mtp_ring = read_ring(m.planes(), slot, tail.start, tail.len())?;
            let h = self.cfg.hidden;
            let store: Vec<u16> = m.store().to_vec()?;
            d.mtp_carry = store[slot * MTP_TAP_CAP * h..slot * MTP_TAP_CAP * h + h].to_vec();
            let b = m.book(slot);
            d.mtp_book = Some((b.ok, b.first, b.n));
        }
        Ok(d)
    }
}

/// Raw rows `from..from + n` of `slot`'s ring in `pl` as fp16 bits.
fn read_ring(pl: &dsa::Planes, slot: usize, from: usize, n: usize) -> Result<Vec<u16>, ModelError> {
    let mut out = Vec::with_capacity(n * 2 * dsa::D);
    let all: Vec<u16> = pl.ring.to_vec()?;
    for p in from..from + n {
        let at = (slot * pl.ring_len + p % pl.ring_len) * 2 * dsa::D;
        out.extend_from_slice(&all[at..at + 2 * dsa::D]);
    }
    Ok(out)
}
