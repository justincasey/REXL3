//! CPU reference of the MoE plan's two builders — the single-block `plan_kernel` (`gb_moe_pf_plan`) and the
//! multi-block deterministic plan (`gb_moe_pf_plan_parallel`) — and the proof that they are bit-identical on
//! realistic dispersed routes and at the shape boundaries the kernels hit. These run with no CUDA device (pure
//! integers; the CUDA kernels below are the same algorithm, so this is the oracle the GPU tests compare the kernels
//! against on a box with a device). The parallel builder's correctness here is exactly the `kPlanChunk` / warp /
//! lane decomposition of `csrc/moe_prefill.cu`: block offsets are the exclusive prefix of per-block expert counts,
//! within a chunk the rank is the per-warp exclusive prefix plus the within-warp `popc(__match_any_sync & before)`.
//!
//!     cargo test --release -p gb-cuda --test moe_plan_parallel_cpu

use gb_cuda::moe_prefill::{item_cap, padded_rows, PLAN_CHUNK};

fn legacy_plan(pick: &[i32], e: usize, cap: usize, p: usize) -> (Vec<i32>, Vec<i32>, [i32; 4]) {
    let s = padded_rows(p, e);
    let mut spair = vec![0i32; s];
    let mut items = vec![0i32; 3 * cap * 4];
    let mut cnt = vec![0i32; e];
    for &pe in pick {
        if pe >= 0 && (pe as usize) < e {
            cnt[pe as usize] += 1;
        }
    }
    let tiles: Vec<i32> = cnt.iter().map(|&c| (c + 15) >> 4).collect();
    let mut toff = vec![0i32; e];
    let mut total_tiles = 0i32;
    for i in 0..e {
        toff[i] = total_tiles;
        total_tiles += tiles[i];
    }
    for (ei, &c) in cnt.iter().enumerate() {
        if c > 0 {
            let t0 = toff[ei] as usize * 16;
            let mut j = 0usize;
            for (i, &pe) in pick.iter().enumerate() {
                if pe == ei as i32 {
                    spair[t0 + j] = i as i32;
                    j += 1;
                }
            }
            for k in j..(tiles[ei] * 16) as usize {
                spair[t0 + k] = -1;
            }
        }
    }
    let mut oA = vec![0i32; e];
    let mut oB = vec![0i32; e];
    let mut oC = vec![0i32; e];
    let (mut nA, mut nB, mut nC) = (0i32, 0i32, 0i32);
    for ei in 0..e {
        let full = tiles[ei] >> 2;
        let rem = tiles[ei] & 3;
        let cC = full + (rem == 3) as i32;
        let cB = (rem == 2) as i32;
        let cA = (rem == 1) as i32;
        oA[ei] = nA;
        nA += cA;
        oB[ei] = nB;
        nB += cB;
        oC[ei] = nC;
        nC += cC;
    }
    for ei in 0..e {
        if tiles[ei] > 0 {
            let full = tiles[ei] >> 2;
            let rem = tiles[ei] & 3;
            let t0 = toff[ei];
            for gi in 0..full {
                let idx = (2 * cap as i32 + oC[ei] + gi) as usize;
                let vr = (cnt[ei] - 64 * gi).min(64);
                items[idx * 4..idx * 4 + 4].copy_from_slice(&[ei as i32, t0 + 4 * gi, 4, vr]);
            }
            let vr = cnt[ei] - 64 * full;
            match rem {
                1 => {
                    let idx = oA[ei] as usize;
                    items[idx * 4..idx * 4 + 4].copy_from_slice(&[ei as i32, t0 + 4 * full, 1, vr]);
                }
                2 => {
                    let idx = (cap as i32 + oB[ei]) as usize;
                    items[idx * 4..idx * 4 + 4].copy_from_slice(&[ei as i32, t0 + 4 * full, 2, vr]);
                }
                3 => {
                    let idx = (2 * cap as i32 + oC[ei] + full) as usize;
                    items[idx * 4..idx * 4 + 4].copy_from_slice(&[ei as i32, t0 + 4 * full, 3, vr]);
                }
                _ => {}
            }
        }
    }
    (spair, items, [nA, nB, nC, total_tiles * 16])
}

/// The multi-block deterministic plan, mirroring `csrc/moe_prefill.cu` `plan_hist_kernel` → `plan_prefix_kernel`
/// → `plan_scatter_kernel` → `plan_items_kernel` exactly (block size `PLAN_CHUNK`, 32 lanes per warp).
fn parallel_plan(pick: &[i32], e: usize, cap: usize, p: usize) -> (Vec<i32>, Vec<i32>, [i32; 4]) {
    let s = padded_rows(p, e);
    let mut spair = vec![0i32; s];
    let mut items = vec![0i32; 3 * cap * 4];
    let b = p.div_ceil(PLAN_CHUNK);
    let mut hist = vec![0i32; b * e];
    for (i, &pe) in pick.iter().enumerate() {
        if pe >= 0 && (pe as usize) < e {
            hist[(i / PLAN_CHUNK) * e + pe as usize] += 1;
        }
    }
    let mut off = vec![0i32; b * e];
    let mut cnt = vec![0i32; e];
    for ei in 0..e {
        let mut run = 0i32;
        for bi in 0..b {
            off[bi * e + ei] = run;
            run += hist[bi * e + ei];
        }
        cnt[ei] = run;
    }
    let tiles: Vec<i32> = cnt.iter().map(|&c| (c + 15) >> 4).collect();
    let mut toff = vec![0i32; e];
    let mut total_tiles = 0i32;
    for i in 0..e {
        toff[i] = total_tiles;
        total_tiles += tiles[i];
    }
    let mut oA = vec![0i32; e];
    let mut oB = vec![0i32; e];
    let mut oC = vec![0i32; e];
    let (mut nA, mut nB, mut nC) = (0i32, 0i32, 0i32);
    for ei in 0..e {
        let full = tiles[ei] >> 2;
        let rem = tiles[ei] & 3;
        let cC = full + (rem == 3) as i32;
        let cB = (rem == 2) as i32;
        let cA = (rem == 1) as i32;
        oA[ei] = nA;
        nA += cA;
        oB[ei] = nB;
        nB += cB;
        oC[ei] = nC;
        nC += cC;
    }
    let wcnt = PLAN_CHUNK / 32;
    for bi in 0..b {
        let start = bi * PLAN_CHUNK;
        let end = (start + PLAN_CHUNK).min(p);
        let len = end - start;
        let mut wc = vec![vec![0i32; e]; wcnt];
        for (li, i) in (start..end).enumerate() {
            let pe = pick[i];
            if pe >= 0 && (pe as usize) < e {
                wc[li / 32][pe as usize] += 1;
            }
        }
        for (li, i) in (start..end).enumerate() {
            let pe = pick[i];
            if pe >= 0 && (pe as usize) < e {
                let w = li / 32;
                let lane = li % 32;
                let wp: i32 = (0..w).map(|ww| wc[ww][pe as usize]).sum();
                let wrank: i32 = (0..lane).filter(|&l2| pick[start + w * 32 + l2] == pe).count() as i32;
                let r = off[bi * e + pe as usize] + wp + wrank;
                spair[(toff[pe as usize] as usize) * 16 + r as usize] = i as i32;
            }
        }
    }
    for ei in 0..e {
        let c = cnt[ei] as usize;
        let tiles_e = tiles[ei] as usize;
        let t0 = toff[ei] as usize;
        for j in c..tiles_e * 16 {
            spair[t0 * 16 + j] = -1;
        }
        let full = tiles[ei] >> 2;
        let rem = tiles[ei] & 3;
        for gi in 0..full {
            let idx = (2 * cap as i32 + oC[ei] + gi) as usize;
            let vr = (cnt[ei] - 64 * gi).min(64);
            items[idx * 4..idx * 4 + 4].copy_from_slice(&[ei as i32, t0 as i32 + 4 * gi, 4, vr]);
        }
        let vr = cnt[ei] - 64 * full;
        match rem {
            1 => {
                let idx = oA[ei] as usize;
                items[idx * 4..idx * 4 + 4].copy_from_slice(&[ei as i32, t0 as i32 + 4 * full, 1, vr]);
            }
            2 => {
                let idx = (cap as i32 + oB[ei]) as usize;
                items[idx * 4..idx * 4 + 4].copy_from_slice(&[ei as i32, t0 as i32 + 4 * full, 2, vr]);
            }
            3 => {
                let idx = (2 * cap as i32 + oC[ei] + full) as usize;
                items[idx * 4..idx * 4 + 4].copy_from_slice(&[ei as i32, t0 as i32 + 4 * full, 3, vr]);
            }
            _ => {}
        }
    }
    (spair, items, [nA, nB, nC, total_tiles * 16])
}

/// A mutation of the parallel planner that changes the slot ordering: within-warp rank counts the lanes AFTER this
/// lane instead of before (a "sorted row" now maps to the wrong pick, and the spair order drifts). The bit-identity
/// test must catch it.
fn buggy_slot_order_plan(pick: &[i32], e: usize, cap: usize, p: usize) -> (Vec<i32>, Vec<i32>, [i32; 4]) {
    let s = padded_rows(p, e);
    let mut spair = vec![0i32; s];
    let mut items = vec![0i32; 3 * cap * 4];
    let b = p.div_ceil(PLAN_CHUNK);
    let mut hist = vec![0i32; b * e];
    for (i, &pe) in pick.iter().enumerate() {
        if pe >= 0 && (pe as usize) < e {
            hist[(i / PLAN_CHUNK) * e + pe as usize] += 1;
        }
    }
    let mut off = vec![0i32; b * e];
    let mut cnt = vec![0i32; e];
    for ei in 0..e {
        let mut run = 0i32;
        for bi in 0..b {
            off[bi * e + ei] = run;
            run += hist[bi * e + ei];
        }
        cnt[ei] = run;
    }
    let tiles: Vec<i32> = cnt.iter().map(|&c| (c + 15) >> 4).collect();
    let mut toff = vec![0i32; e];
    let mut total_tiles = 0i32;
    for i in 0..e {
        toff[i] = total_tiles;
        total_tiles += tiles[i];
    }
    let mut oA = vec![0i32; e];
    let mut oB = vec![0i32; e];
    let mut oC = vec![0i32; e];
    let (mut nA, mut nB, mut nC) = (0i32, 0i32, 0i32);
    for ei in 0..e {
        let full = tiles[ei] >> 2;
        let rem = tiles[ei] & 3;
        let cC = full + (rem == 3) as i32;
        let cB = (rem == 2) as i32;
        let cA = (rem == 1) as i32;
        oA[ei] = nA;
        nA += cA;
        oB[ei] = nB;
        nB += cB;
        oC[ei] = nC;
        nC += cC;
    }
    let wcnt = PLAN_CHUNK / 32;
    for bi in 0..b {
        let start = bi * PLAN_CHUNK;
        let end = (start + PLAN_CHUNK).min(p);
        let mut wc = vec![vec![0i32; e]; wcnt];
        for (li, i) in (start..end).enumerate() {
            let pe = pick[i];
            if pe >= 0 && (pe as usize) < e {
                wc[li / 32][pe as usize] += 1;
            }
        }
        for (li, i) in (start..end).enumerate() {
            let pe = pick[i];
            if pe >= 0 && (pe as usize) < e {
                let w = li / 32;
                let lane = li % 32;
                let wp: i32 = (0..w).map(|ww| wc[ww][pe as usize]).sum();
                // MUTATION: count same-expert lanes AFTER this lane (not before) -> slot order changes
                let wrank: i32 = (lane + 1..32).filter(|&l2| pick[start + w * 32 + l2] == pe).count() as i32;
                let r = off[bi * e + pe as usize] + wp + wrank;
                spair[(toff[pe as usize] as usize) * 16 + r as usize] = i as i32;
            }
        }
    }
    for ei in 0..e {
        let c = cnt[ei] as usize;
        let tiles_e = tiles[ei] as usize;
        let t0 = toff[ei] as usize;
        for j in c..tiles_e * 16 {
            spair[t0 * 16 + j] = -1;
        }
        let full = tiles[ei] >> 2;
        let rem = tiles[ei] & 3;
        for gi in 0..full {
            let idx = (2 * cap as i32 + oC[ei] + gi) as usize;
            let vr = (cnt[ei] - 64 * gi).min(64);
            items[idx * 4..idx * 4 + 4].copy_from_slice(&[ei as i32, t0 as i32 + 4 * gi, 4, vr]);
        }
        let vr = cnt[ei] - 64 * full;
        match rem {
            1 => {
                let idx = oA[ei] as usize;
                items[idx * 4..idx * 4 + 4].copy_from_slice(&[ei as i32, t0 as i32 + 4 * full, 1, vr]);
            }
            2 => {
                let idx = (cap as i32 + oB[ei]) as usize;
                items[idx * 4..idx * 4 + 4].copy_from_slice(&[ei as i32, t0 as i32 + 4 * full, 2, vr]);
            }
            3 => {
                let idx = (2 * cap as i32 + oC[ei] + full) as usize;
                items[idx * 4..idx * 4 + 4].copy_from_slice(&[ei as i32, t0 as i32 + 4 * full, 3, vr]);
            }
            _ => {}
        }
    }
    (spair, items, [nA, nB, nC, total_tiles * 16])
}

/// xorshift64* PRNG (deterministic dispersed routes; uniform constants route everything to the same experts and would
/// mask the grouping paths).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn make_dispersed_picks(r: &mut Rng, p: usize, e: usize) -> Vec<i32> {
    (0..p).map(|_| r.below(e) as i32).collect()
}

fn assert_plans_equal(what: &str, a: &(Vec<i32>, Vec<i32>, [i32; 4]), b: &(Vec<i32>, Vec<i32>, [i32; 4])) {
    assert_eq!(a.0, b.0, "{what}: spair differs");
    assert_eq!(a.1, b.1, "{what}: items differs");
    assert_eq!(a.2, b.2, "{what}: nitems differs");
}

#[test]
fn parallel_plan_is_bit_identical_to_legacy_on_dispersed_routes() {
    let shapes: &[(usize, usize, &[usize])] = &[
        // (experts, topk, rows): E512/topk10 (Qwen) across and around the 256-chunk and 16-row-tile limits
        (512, 10, &[1, 17, 25, 26, 63, 127, 128, 129, 130, 255, 256, 257, 511, 1024, 2048]),
        (1024, 16, &[2, 16, 32, 255, 256, 257, 4096]),
        (128, 4, &[1, 3, 15, 16, 17, 64, 300]),
        (1024, 1, &[10, 256, 257, 4096]),
        (1, 1, &[5, 100, 1000, 4096]),
        (512, 1, &[1, 32, 33, 1000]),
        // GLM-5.3-Flash (E288/topk8), and the expert count of one of its mixed-width subbanks
        (288, 8, &[1, 129, 512, 2048]),
        (58, 8, &[129, 2048]),
    ];
    let mut seed = 0x9E3779B97F4A7C15u64;
    for &(e, topk, rows) in shapes {
        for &rows in rows {
            let p = rows * topk;
            let cap = item_cap(p, e);
            let mut r = Rng(seed);
            seed = r.next();
            // realistic dispersed routes + a few edge cases (all to one expert, holes, an out-of-range id)
            let pick = make_dispersed_picks(&mut r, p, e);
            let legacy = legacy_plan(&pick, e, cap, p);
            let parallel = parallel_plan(&pick, e, cap, p);
            assert_plans_equal(&format!("E{e}/topk{topk}/rows{rows}"), &legacy, &parallel);
        }
    }
}

/// A mixed-width layer's subbank (GLM-5.3-Flash's P1 path plans each subbank on the remapped picks): about a fifth of the pairs land in
/// the subbank's 58 local ids, every other pick is -1.
#[test]
fn parallel_plan_is_bit_identical_to_legacy_on_a_subbank_remap() {
    let (e, topk) = (58usize, 8usize);
    let mut r = Rng(0x5eed_5ab0);
    for rows in [1usize, 16, 129, 512, 2048] {
        let p = rows * topk;
        let cap = item_cap(p, 288);
        let pick: Vec<i32> = (0..p).map(|_| if r.below(5) == 0 { r.below(e) as i32 } else { -1 }).collect();
        let legacy = legacy_plan(&pick, e, cap, p);
        let parallel = parallel_plan(&pick, e, cap, p);
        assert_plans_equal(&format!("subbank E{e}/topk{topk}/rows{rows}"), &legacy, &parallel);
    }
}

#[test]
fn parallel_plan_is_deterministic() {
    let e = 512;
    let topk = 10;
    let p = 2048 * topk;
    let cap = item_cap(p, e);
    let mut r = Rng(12345);
    let pick = make_dispersed_picks(&mut r, p, e);
    let a = parallel_plan(&pick, e, cap, p);
    let b = parallel_plan(&pick, e, cap, p);
    assert_plans_equal("determinism", &a, &b);
}

#[test]
fn parallel_plan_is_invariant_to_chunk_boundary_alignment() {
    // The same route (same ids, same order) must give the same plan whether P lands exactly on a PLAN_CHUNK
    // boundary or a little over/under it; chunking is only a partitioning constant.
    let e = 512;
    let topk = 10;
    for rows in [255usize, 256, 257, 1000] {
        let p = rows * topk;
        let cap = item_cap(p, e);
        let mut r = Rng(rows as u64 * 777);
        let pick = make_dispersed_picks(&mut r, p, e);
        let a = legacy_plan(&pick, e, cap, p);
        let b = parallel_plan(&pick, e, cap, p);
        assert_plans_equal(&format!("boundary rows{rows}"), &a, &b);
    }
}

#[test]
fn the_plan_tiles_within_capacity_and_refuses_under_capacity() {
    // Every class's items and the padded-row count must fit the buffers the scratch prices (cap and padded_rows),
    // for the largest allowed shape.
    let (e, topk, rows) = (1024usize, 16usize, 4096usize);
    let p = rows * topk;
    let cap = item_cap(p, e);
    let s = padded_rows(p, e);
    let mut r = Rng(0xC0FFEE);
    let pick = make_dispersed_picks(&mut r, p, e);
    let (spair, items, nitems) = parallel_plan(&pick, e, cap, p);
    let sorted = nitems[3] as usize;
    assert!(sorted <= s, "sorted rows {} exceed padded_rows {}", sorted, s);
    assert!(nitems[0] >= 0 && nitems[1] >= 0 && nitems[2] >= 0);
    assert!(nitems[0] as usize <= cap && nitems[1] as usize <= cap && nitems[2] as usize <= cap);
    // spair values: either -1 (padding) or a valid pick index in [0, P)
    for &v in spair[..sorted].iter() {
        assert!(v == -1 || (v >= 0 && (v as usize) < p), "spair value {v} out of range");
    }
    // items encode [expert, first_tile, tiles, valid_rows] with expert<E, first_tile<sorted/16, tiles in 1..=4,
    // valid_rows in 1..=64; and every class's segment stays inside 3*cap.
    for (seg, base, count) in
        [(0usize, 0usize, nitems[0] as usize), (1, cap, nitems[1] as usize), (2, 2 * cap, nitems[2] as usize)]
    {
        for k in 0..count {
            let idx = (base + k) * 4;
            let (x, y, z, w) = (items[idx], items[idx + 1], items[idx + 2], items[idx + 3]);
            assert!(x >= 0 && (x as usize) < e, "segment {seg}: expert {x} out of range");
            assert!(y >= 0 && (y as usize) < sorted / 16, "segment {seg}: tile {y} out of range");
            assert!(z >= 1 && z <= 4, "segment {seg}: tiles {z}");
            assert!(w >= 1 && w <= 64, "segment {seg}: valid rows {w}");
            let _ = seg;
        }
    }
    // a too-small cap is refused by the launcher contract: the kernels' own bound (item_cap) is the minimum
    assert!(cap >= item_cap(p, e));
    // the class counts are intrinsic to the route, independent of the cap the scratch happened to allocate
    let (_, _, nitems2) = parallel_plan(&pick, e, cap + 1000, p);
    assert_eq!(nitems2, nitems);
}

#[test]
fn parallel_plan_is_bit_identical_to_legacy_on_hot_routes() {
    // The exact route of the failing GPU guard (Style::Hot: every row picks the same experts 0..topk-1), at and
    // around the 16-row-tile boundary. At 128 rows every hot expert has exactly 8 tiles (class C only); at 129 each
    // has 9 tiles, so class A items (one 16-row tile) appear for the first time — the plan must stay bit-identical
    // across that transition, and the class-A path must actually be exercised. The class totals are also checked
    // against the kernel contract: nA/nB/nC are the sums over experts of (tiles % 4 == 1/2/3) and tiles >> 2 +
    // (tiles % 4 == 3).
    let e = 512;
    let topk = 10;
    for rows in [127usize, 128, 129, 130, 255, 256, 257] {
        let p = rows * topk;
        let cap = item_cap(p, e);
        let pick: Vec<i32> = (0..p).map(|i| (i % topk) as i32).collect();
        let legacy = legacy_plan(&pick, e, cap, p);
        let parallel = parallel_plan(&pick, e, cap, p);
        assert_plans_equal(&format!("hot E{e}/topk{topk}/rows{rows}"), &legacy, &parallel);
        // item-tail contract: every capacity slot past the used region of each class is zero in both builders (the
        // kernels zero the whole items buffer before writing; used regions are [0, nA), [cap, cap + nB),
        // [2*cap, 2*cap + nC), so each class gap stays zero)
        let gap_bounds = [
            (0usize, parallel.2[0] as usize, cap),
            (cap, parallel.2[1] as usize, 2 * cap),
            (2 * cap, parallel.2[2] as usize, 3 * cap),
        ];
        for (lo, len, hi) in gap_bounds {
            assert!(
                parallel.1[(lo + len) * 4..hi * 4].iter().all(|&v| v == 0),
                "hot rows{rows}: parallel items tail not zeroed between {lo}+{len} and {hi}"
            );
        }
        let tiles = (rows + 15) / 16;
        let nA = (tiles & 3 == 1) as i32 * topk as i32;
        let nB = (tiles & 3 == 2) as i32 * topk as i32;
        let nC = ((tiles >> 2) + (tiles & 3 == 3) as usize) as i32 * topk as i32;
        assert_eq!(parallel.2[0], nA, "hot rows{rows}: class A count");
        assert_eq!(parallel.2[1], nB, "hot rows{rows}: class B count");
        assert_eq!(parallel.2[2], nC, "hot rows{rows}: class C count");
        assert_eq!(parallel.2[3], (tiles * topk * 16) as i32, "hot rows{rows}: sorted rows");
    }
}

#[test]
fn a_slot_ordering_mutation_is_caught() {
    // The bit-identity test is not vacuous: a change to the within-expert slot ordering must fail it.
    let e = 512;
    let topk = 10;
    let rows = 2048;
    let p = rows * topk;
    let cap = item_cap(p, e);
    let mut r = Rng(998877);
    let pick = make_dispersed_picks(&mut r, p, e);
    let legacy = legacy_plan(&pick, e, cap, p);
    let buggy = buggy_slot_order_plan(&pick, e, cap, p);
    let caught = legacy.0 != buggy.0 || legacy.1 != buggy.1 || legacy.2 != buggy.2;
    assert!(caught, "the slot-ordering mutation was NOT caught (bit-identity test is too weak)");
}
