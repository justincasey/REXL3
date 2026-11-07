//! The GLM-5.3-Flash prefix cache's arithmetic: the pure functions the GPU side (`model/snapshot.rs`) and the backend (`backend.rs`) are built from, and
//! that the CPU tests pin. Design and evidence: `docs/PREFIX_CACHE.md` and `docs/GLM53.md`, "Prefix cache".
//!
//! **What a snapshot of `len` tokens is** (everything a prompt leaves behind that its pages do not carry):
//! * the committed KDA state of each of the 34 layers: ONE of the two state buffers (the one the slot's parity word names), fp32 `[64][128][128]`, and
//!   the conv history, bf16 `[3][24576]` (`snapshot_bytes`: 140.8 MiB);
//! * the raw rows of the DSA indexer's incomplete pool group, for the 11 MLA layers and the NextN layer's own indexer: positions
//!   `len - len % 4 .. len`, at most [`MAX_TAIL_ROWS`] rows of 512 B each (a pool is completed from its four members when the last one is appended,
//!   and the members before this call's first row come from the slot's ring, which a slot that adopts a prefix does not have);
//! * the NextN drafter's carry: the final-normed hidden row of position `len - 1` (8 KiB), which the first MTP entry after the cut reads, and a flag that
//!   says the drafter's bookkeeping was sound when the snapshot was taken;
//! * the KV pages by reference (latent cache and pool plane of 11 trunk layers and the NextN layer, 256 tokens a page): the full pages shared, the
//!   partial last page copied at restore.
//!
//! There is no path rule: a GLM prompt chunk of any size leaves, in every row, the bits the same token decoded alone leaves (`docs/GLM53.md`, "Prefill"),
//! so a snapshot is interchangeable between prompts of every length and every chunk grid, and ONE path id serves every total.

use super::alloc::{self, RING_LEN};
use super::config::GlmConfig;

/// The key's path id: GLM has one arithmetic, so every prompt, whatever its length, lives on this path.
pub const PATH_ID: u32 = 1;

/// Tokens per DSA pool group (`gb_cuda::dsa::P`).
pub const GROUP: usize = gb_cuda::dsa::P;

/// Most raw rows a snapshot keeps per indexer layer: the incomplete group of a length that is not a multiple of [`GROUP`].
pub const MAX_TAIL_ROWS: usize = GROUP - 1;

/// The ring holds a whole number of pool groups, so the tail of a prefix (which starts at a multiple of 4) never wraps it.
const _: () = assert!(RING_LEN % GROUP == 0);

/// Bytes of one raw row of the indexer's ring: `[LayerNorm(k) | gate]`, 2 x 128 fp16.
pub const RING_ROW_BYTES: usize = 2 * gb_cuda::dsa::D * 2;

/// The positions whose raw rows the incomplete pool group of a prefix of `len` tokens holds (empty when `len` is a multiple of 4: a cut at a multiple
/// of 4, a page-aligned one included, needs no ring rows).
pub fn ring_tail(len: usize) -> std::ops::Range<usize> {
    len - len % GROUP..len
}

/// The first row of the slot's ring that holds the tail (`position % ring_len`) and the number of rows to copy: the tail never wraps, because it starts
/// at a multiple of 4 and the ring length is one.
pub fn ring_tail_rows(len: usize) -> (usize, usize) {
    let t = ring_tail(len);
    (t.start % RING_LEN, t.len())
}

/// How a prefix of `len` tokens maps onto pages: the number of full pages (shared by reference) and whether there is a partial last page (copied at
/// restore, so the first row the sequence writes lands in a page it owns).
pub fn split_pages(len: usize) -> (usize, bool) {
    (len / alloc::PAGE_TOKENS, len % alloc::PAGE_TOKENS != 0)
}

/// Where the slice that starts at `done` ends: the model's chunk (`chunk` rows, at most the prompt's `n`), cut short at the first of `cuts` it would
/// pass, so that the state at a cut is the state after a whole slice and a snapshot is a copy of what the slice left. The bits of every row do not
/// depend on the chunk grid, so ending a slice early changes nothing but how many slices the prompt takes.
pub fn slice_end(done: usize, n: usize, chunk: usize, cuts: &[usize]) -> usize {
    let natural = (done + chunk).min(n);
    match cuts.iter().copied().find(|&c| c > done) {
        Some(c) if c < natural => c,
        _ => natural,
    }
}

/// The rows of a prompt slice under contract `c`. P2 (`docs/GLM_PREFILL_P2.md` 8.1): every slice of a prefill segment but its last must end
/// a whole number of 16-row KDA chunks after the segment's first row, so a slice that starts on that grid takes a multiple of 16 rows
/// (rounded down, at least 16); P1: `rows` as given (every grid is P1's).
pub fn slice_rows_for(c: super::contract::PrefillContract, rows: usize) -> usize {
    let k = gb_cuda::kda::P2_CHUNK_ROWS;
    match c {
        super::contract::PrefillContract::P1 => rows,
        super::contract::PrefillContract::P2 => (rows - rows % k).max(k),
    }
}

/// The snapshot cuts a slice that starts at `done` can take under contract `c`, from the engine's planned `cuts` (ascending). P2: each moves
/// down to the segment's grid (the largest `anchor + 16 k <= cut`), and is kept when it is past `done`, outside every image (`inside` says
/// whether a cut falls strictly inside one) and not a duplicate: a snapshot off the grid would resume with other bits than the fresh
/// prefill (the end cut `P - 16` therefore lands 16..31 tokens before the prompt's end; the engine keys an entry by its captured length).
/// P1: the cuts as planned.
pub fn cuts_for(
    c: super::contract::PrefillContract,
    anchor: usize,
    done: usize,
    cuts: &[usize],
    inside: &dyn Fn(usize) -> bool,
) -> Vec<usize> {
    let k = gb_cuda::kda::P2_CHUNK_ROWS;
    match c {
        super::contract::PrefillContract::P1 => cuts.to_vec(),
        super::contract::PrefillContract::P2 => {
            let mut out: Vec<usize> = Vec::new();
            for &cut in cuts {
                if cut < anchor {
                    continue;
                }
                let a = anchor + (cut - anchor) / k * k;
                if a > done && !inside(a) && out.last() != Some(&a) {
                    out.push(a);
                }
            }
            out
        }
    }
}

/// What one prefill slice does about the cuts of its prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlicePlan {
    /// Where the slice ends.
    pub end: usize,
    /// Cuts strictly inside the slice, to be snapshotted INSIDE the chunk (`GlmModel::prefill_chunk_cuts`, no extra forward). A cut at `end` is a snapshot
    /// after the chunk.
    pub interior: Vec<usize>,
}

/// The slice that starts at `done` for a prompt of `n` tokens in chunks of `chunk` rows. A chunk of at least two rows takes its interior cuts inside
/// itself (`split_ok`: the model has the large-row scratch, whose kernels run any chunk with the decode kernels' bits), so it runs whole and costs what it
/// always cost; without that scratch (a model built for chunks of 128 rows or fewer) the slice ends at the first cut it would pass (`slice_end`), which is
/// one more small forward for a cold prompt.
pub fn plan_slice(done: usize, n: usize, chunk: usize, cuts: &[usize], split_ok: bool) -> SlicePlan {
    let natural = (done + chunk).min(n);
    if split_ok && natural - done >= MIN_SPLIT_ROWS {
        return SlicePlan { end: natural, interior: cuts.iter().copied().filter(|&c| c > done && c < natural).collect() };
    }
    SlicePlan { end: slice_end(done, n, chunk, cuts), interior: Vec::new() }
}

/// Rows a chunk must have at least to take snapshots inside itself (a cut strictly inside leaves a row on each side).
pub const MIN_SPLIT_ROWS: usize = 2;

/// The byte layout of one snapshot slot of a model with `cfg`'s shapes (with the NextN layer's share when `mtp`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapLayout {
    pub kda_layers: usize,
    /// Bytes of one KDA layer's state buffer.
    pub kda_state: u64,
    /// Bytes of one KDA layer's conv history.
    pub kda_conv: u64,
    /// Indexer layers whose ring tail is kept: the MLA layers, and the NextN layer's.
    pub ring_layers: usize,
    /// The drafter's carry row, bytes (0 without the NextN layer).
    pub carry: u64,
}

impl SnapLayout {
    pub fn of(cfg: &GlmConfig, mtp: bool) -> Result<Self, String> {
        let d = alloc::dims_of(cfg)?;
        Ok(Self {
            kda_layers: cfg.kda_layers().len(),
            kda_state: (d.s_buf_elems() * 4) as u64,
            kda_conv: (d.conv_slot_elems() * 2) as u64,
            ring_layers: cfg.mla_layers().len() + usize::from(mtp),
            carry: if mtp { (cfg.hidden * 2) as u64 } else { 0 },
        })
    }
    /// Bytes of the ring tails of one snapshot (every indexer layer at the most rows a tail can have).
    pub fn ring_bytes(&self) -> u64 {
        (self.ring_layers * MAX_TAIL_ROWS * RING_ROW_BYTES) as u64
    }
    /// Bytes of the KDA part of one snapshot.
    pub fn kda_bytes(&self) -> u64 {
        self.kda_layers as u64 * (self.kda_state + self.kda_conv)
    }
    /// Bytes of one snapshot slot: the memory plan and the allocation both use this.
    pub fn bytes(&self) -> u64 {
        self.kda_bytes() + self.ring_bytes() + self.carry
    }
}

/// Bytes of one snapshot slot (see [`SnapLayout`]).
pub fn snapshot_bytes(cfg: &GlmConfig, mtp: bool) -> Result<u64, String> {
    Ok(SnapLayout::of(cfg, mtp)?.bytes())
}

/// The key namespace of this pack's snapshots: everything their bytes depend on besides the tokens. Another pack, another cache format or another
/// shape of state is another namespace, so entries are never shared across them (the index is empty at every boot anyway). The NextN layer is in
/// it: a snapshot with the drafter's state is not one without it. The prefill chunk is NOT: rows are chunk-grid invariant (under either contract).
/// The prefill contract IS (`contract::PrefillContract::arith_tag`): P1 adds nothing (its text is the one it always was: one arithmetic for
/// prompt and decoded rows), P2 is another arithmetic for prompt rows and so another namespace. The latent cache's format IS (`--kv-dtype`):
/// fp16 keeps the text it always had (`kv=f16`), fp8 pages are another namespace.
pub fn namespace_of(target_fingerprint: &str, mtp: bool) -> u64 {
    namespace_with(target_fingerprint, mtp, super::contract::current(), gb_cuda::mla::configured_kv().unwrap_or_default())
}

/// [`namespace_of`] under a given prefill contract and latent cache format (pure, for the tests).
pub fn namespace_with(
    target_fingerprint: &str,
    mtp: bool,
    contract: super::contract::PrefillContract,
    kv: gb_cuda::mla::LatentKv,
) -> u64 {
    let kv = match kv {
        gb_cuda::mla::LatentKv::Plain => "f16",
        gb_cuda::mla::LatentKv::Fp8 => "fp8-e4m3-token-f32(v1)+f16",
    };
    let text = format!(
        "rexl3-glm-prefix-cache-v1|{target_fingerprint}|mtp={}|page={}|ring={RING_LEN}|tail={MAX_TAIL_ROWS}|kv={kv}|state=f32+bf16|arith=one{}",
        if mtp { "nextn" } else { "none" },
        alloc::PAGE_TOKENS,
        contract.arith_tag(),
    );
    text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GLM_CONFIG: &str = include_str!("../../../../../testdata/glm53/config.json");

    fn cfg() -> GlmConfig {
        GlmConfig::from_json(&serde_json::from_str(GLM_CONFIG).unwrap()).unwrap()
    }

    #[test]
    fn the_ring_tail_is_the_incomplete_pool_group() {
        for len in [0usize, 1, 2, 3, 4, 5, 255, 256, 257, 258, 259, 260, 1023, 1024, 2047, 2048, 2049, 100_003] {
            let t = ring_tail(len);
            assert_eq!(t.end, len);
            assert_eq!(t.start % 4, 0, "the tail starts at a multiple of 4: {len}");
            assert_eq!(t.len(), len % 4);
            assert!(t.len() <= MAX_TAIL_ROWS);
            let (row, n) = ring_tail_rows(len);
            assert_eq!(n, len % 4);
            assert!(row + n <= RING_LEN, "a tail never wraps the ring: len {len}, row {row}, n {n}");
        }
        // a cut at a multiple of 4 (a page boundary included) needs nothing
        assert!(ring_tail(2048).is_empty() && ring_tail(256).is_empty() && ring_tail(4).is_empty());
        // the three residues that do
        assert_eq!(ring_tail(2049), 2048..2049);
        assert_eq!(ring_tail(2050), 2048..2050);
        assert_eq!(ring_tail(2051), 2048..2051);
        // the ring row of a tail across the ring's wrap
        assert_eq!(ring_tail_rows(257), (0, 1));
        assert_eq!(ring_tail_rows(255), (252, 3));
        assert_eq!(ring_tail_rows(2051), (0, 3));
    }

    #[test]
    fn a_prefix_maps_to_full_pages_and_a_partial_one() {
        assert_eq!(split_pages(1), (0, true));
        assert_eq!(split_pages(255), (0, true));
        assert_eq!(split_pages(256), (1, false));
        assert_eq!(split_pages(257), (1, true));
        assert_eq!(split_pages(2048), (8, false));
        assert_eq!(split_pages(2049), (8, true));
    }

    #[test]
    fn a_slice_ends_at_the_first_cut_it_would_pass() {
        // no cuts: the chunk, or the prompt's end
        assert_eq!(slice_end(0, 1000, 512, &[]), 512);
        assert_eq!(slice_end(512, 1000, 512, &[]), 1000);
        // a cut inside the chunk ends the slice there
        assert_eq!(slice_end(0, 1000, 512, &[300, 900]), 300);
        assert_eq!(slice_end(300, 1000, 512, &[300, 900]), 812);
        assert_eq!(slice_end(812, 1000, 512, &[300, 900]), 900);
        assert_eq!(slice_end(900, 1000, 512, &[300, 900]), 1000);
        // a cut at the chunk's end, or past it, changes nothing
        assert_eq!(slice_end(0, 1000, 512, &[512]), 512);
        assert_eq!(slice_end(0, 1000, 512, &[513, 900]), 512);
        // a cut at or below `done` is behind us
        assert_eq!(slice_end(512, 1000, 512, &[512]), 1000);
        // the slice always advances and never passes the prompt
        for done in 0..60usize {
            for chunk in 1..20usize {
                let end = slice_end(done, 60, chunk, &[7, 13, 14, 50]);
                assert!(end > done && end <= 60 && end - done <= chunk, "{done} {chunk} -> {end}");
            }
        }
    }

    #[test]
    fn a_large_chunk_takes_its_cuts_inside_and_a_small_one_ends_at_the_first() {
        // 1,000 tokens in chunks of 512: the first chunk has the cuts inside it (no extra slice), a cut at its end is the boundary snapshot
        let p = plan_slice(0, 1000, 512, &[300, 512, 900], true);
        assert_eq!(p, SlicePlan { end: 512, interior: vec![300] });
        let p = plan_slice(512, 1000, 512, &[900], true);
        assert_eq!(p, SlicePlan { end: 1000, interior: vec![900] });
        // a small chunk (the tail of a prompt) takes its cut inside too; only a model without the large-row scratch ends the slice at the cut
        assert_eq!(plan_slice(0, 1000, 128, &[100], true), SlicePlan { end: 128, interior: vec![100] });
        assert_eq!(
            plan_slice(900, 1000, 512, &[990], true),
            SlicePlan { end: 1000, interior: vec![990] },
            "a 100-row tail chunk"
        );
        assert_eq!(plan_slice(0, 1000, 512, &[300], false), SlicePlan { end: 300, interior: vec![] }, "no large-row scratch");
        assert_eq!(plan_slice(0, 1000, 128, &[64], false), SlicePlan { end: 64, interior: vec![] }, "no large-row scratch");
        // a chunk of one row has no inside
        assert_eq!(plan_slice(998, 1000, 1, &[999], true), SlicePlan { end: 999, interior: vec![] });
        assert_eq!(
            plan_slice(999, 1000, 1, &[999], true),
            SlicePlan { end: 1000, interior: vec![] },
            "a cut at or below `done` is behind us"
        );
        assert_eq!(plan_slice(997, 1000, 2, &[998], true), SlicePlan { end: 999, interior: vec![998] });
        // whatever the plan, the slice advances, never passes the prompt, and every cut it passes is inside it or its end
        for done in 0..40usize {
            for chunk in [1usize, 5, 128, 129, 200] {
                for split in [false, true] {
                    let p = plan_slice(done, 300, chunk, &[7, 50, 140, 141, 250], split);
                    assert!(p.end > done && p.end <= 300 && p.end - done <= chunk);
                    assert!(p.interior.iter().all(|&c| c > done && c < p.end), "{p:?}");
                    assert!(p.interior.windows(2).all(|w| w[0] < w[1]));
                }
            }
        }
    }

    #[test]
    fn a_snapshot_slot_is_the_kda_state_plus_a_few_rows() {
        let c = cfg();
        let off = SnapLayout::of(&c, false).unwrap();
        let on = SnapLayout::of(&c, true).unwrap();
        // 34 KDA layers x (64 x 128 x 128 fp32 + 3 x 24,576 bf16)
        assert_eq!((off.kda_layers, off.kda_state, off.kda_conv), (34, 4_194_304, 147_456));
        assert_eq!(off.kda_bytes(), 34 * 4_341_760);
        assert_eq!(off.kda_bytes(), 147_619_840, "140.8 MiB");
        // 11 indexer layers x 3 rows x 512 B, 12 with the NextN layer's; the carry is one fp16 hidden row
        assert_eq!((off.ring_layers, on.ring_layers), (11, 12));
        assert_eq!((off.ring_bytes(), on.ring_bytes()), (11 * 3 * 512, 12 * 3 * 512));
        assert_eq!((off.carry, on.carry), (0, 8192));
        assert_eq!(snapshot_bytes(&c, false).unwrap(), 147_619_840 + 16_896);
        assert_eq!(snapshot_bytes(&c, true).unwrap(), 147_619_840 + 18_432 + 8192);
        // far below the double-buffered state a slot holds (both buffers): the snapshot keeps the current one only
        assert!(on.bytes() < 34 * (2 * 4_194_304 + 147_456));
    }

    #[test]
    fn the_namespace_separates_what_makes_snapshots_non_interchangeable() {
        let a = namespace_of("pack-a", true);
        assert_eq!(a, namespace_of("pack-a", true), "a function of its inputs");
        assert_ne!(a, namespace_of("pack-b", true), "another pack");
        assert_ne!(a, namespace_of("pack-a", false), "a snapshot with the drafter's state is not one without it");
        use crate::families::glm::contract::PrefillContract;
        use gb_cuda::mla::LatentKv;
        assert_ne!(
            namespace_with("pack-a", true, PrefillContract::P2, LatentKv::Plain),
            namespace_with("pack-a", true, PrefillContract::P2, LatentKv::Fp8),
            "fp8 latent pages are not fp16 pages"
        );
    }

    #[test]
    fn p1_keeps_the_namespace_it_always_had_and_p2_is_another() {
        use crate::families::glm::contract::PrefillContract;
        // the namespace text before the prefill contract existed (`arith=one`, nothing after it)
        let before = |fp: &str, mtp: bool| {
            let text = format!(
                "rexl3-glm-prefix-cache-v1|{fp}|mtp={}|page={}|ring={RING_LEN}|tail={MAX_TAIL_ROWS}|kv=f16|state=f32+bf16|arith=one",
                if mtp { "nextn" } else { "none" },
                alloc::PAGE_TOKENS,
            );
            text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
        };
        use gb_cuda::mla::LatentKv::Plain;
        for (fp, mtp) in [("pack-a", true), ("pack-a", false), ("glm-prefix-engine-test", true)] {
            assert_eq!(namespace_with(fp, mtp, PrefillContract::P1, Plain), before(fp, mtp), "P1's namespace moved: {fp} {mtp}");
            assert_ne!(namespace_with(fp, mtp, PrefillContract::P2, Plain), before(fp, mtp), "P2 shares P1's namespace: {fp} {mtp}");
        }
        // a process's namespace is its contract's: the default (P2 since the owner's decision of 2026-10-19) is another namespace than the one
        // before the contract existed, and `REXL3_GLM_PREFILL_CONTRACT=p1` is exactly that one
        let cur = crate::families::glm::contract::current();
        let kv = gb_cuda::mla::configured_kv().unwrap_or_default();
        assert_eq!(namespace_of("pack-a", true), namespace_with("pack-a", true, cur, kv), "a process is on its contract's namespace");
        if cur == PrefillContract::P1 {
            assert_eq!(namespace_of("pack-a", true), before("pack-a", true), "a P1 process is on the namespace it always had");
        } else {
            assert_ne!(namespace_of("pack-a", true), before("pack-a", true), "a P2 process is not");
        }
    }

    /// Lane P step (e): under P2 a slice is a whole number of 16-row KDA chunks (at least one) and every planned cut moves down to its
    /// segment's grid, past the slice start, outside images, deduplicated; P1 is untouched.
    #[test]
    fn under_p2_slices_and_cuts_land_on_the_segment_grid() {
        use crate::families::glm::contract::PrefillContract::{P1, P2};
        assert_eq!(slice_rows_for(P1, 1000), 1000);
        assert_eq!(slice_rows_for(P2, 1000), 992);
        assert_eq!(slice_rows_for(P2, 4096), 4096);
        assert_eq!(slice_rows_for(P2, 512), 512);
        assert_eq!(slice_rows_for(P2, 7), 16);
        let none = |_: usize| false;
        assert_eq!(cuts_for(P1, 0, 0, &[1000, 2984], &none), vec![1000, 2984]);
        // the end cut P - 16 = 2984 of a 3,000-token prompt lands at 2976; a shared cut (256 M) stays
        assert_eq!(cuts_for(P2, 0, 0, &[1024, 2984], &none), vec![1024, 2976]);
        // relative to a restored anchor (a hit at 512 + 13 = 525 is not possible, but the anchor need not be a multiple of 16)
        assert_eq!(cuts_for(P2, 525, 525, &[1000, 2984], &none), vec![989, 2973]);
        // past the slice start only, and deduplicated
        assert_eq!(cuts_for(P2, 0, 992, &[1000, 1005, 1100], &none), vec![1088]);
        assert_eq!(cuts_for(P2, 0, 0, &[1000, 1005], &none), vec![992]);
        // a cut that moves inside an image goes
        let img = |x: usize| x > 980 && x < 1000;
        assert_eq!(cuts_for(P2, 0, 0, &[1000, 2000], &img), vec![2000]);
    }
}
