//! The routed experts for large row counts (`csrc/moe_prefill.cu`, `src/moe_prefill.rs`): equal to the decode path (`MoeExperts::run`) bit for
//! bit at every row count, chunk grid, routing skew and weight width, on synthetic layers (no pack needed), plus the speed table.
//!
//!     cargo test --release -p gb-cuda --test moe_prefill -- --test-threads=1
//!     cargo test --release -p gb-cuda --test moe_prefill -- --ignored --nocapture bench_      # speed (REXL3_ROUTE_DIR: an oracle dump's routing)
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
mod common;

use common::exl3_synth::synth_group;
use common::*;
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::moe::*;
use gb_cuda::moe_prefill::{item_cap, PrefillScratch};
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::checkpoint::LoadedGroup;
use gb_exl3::f16::f64_to_f16;
use gb_exl3::Codebook;

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: not run");
            return;
        }
    };
}

fn up<T: Copy>(v: &[T]) -> DevBuf {
    DevBuf::from_slice(v).unwrap()
}

fn normal(r: &mut Rng) -> f64 {
    let (a, b) = ((r.next_u64() >> 11) as f64 / (1u64 << 53) as f64, (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64);
    (-2.0 * a.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
}

fn below(r: &mut Rng, n: usize) -> usize {
    (r.next_u64() % n as u64) as usize
}

fn h(x: f64) -> u16 {
    f64_to_f16(x)
}

fn rand_f16(r: &mut Rng, n: usize, sigma: f64) -> Vec<u16> {
    (0..n).map(|_| h(sigma * normal(r))).collect()
}

fn sync_dl<T: Copy + Default>(b: &DevBuf, st: &Stream) -> Vec<T> {
    st.sync().unwrap();
    b.to_vec().unwrap()
}

fn synth_layer(dims: MoeDims, bits_gu: usize, bits_d: usize, cb: Codebook, seed: u64) -> MoeExperts {
    let e = dims.experts;
    let mk = |k, n, bits, tag: u64| -> Vec<LoadedGroup> {
        (0..e).map(|x| synth_group(k, n, bits, cb, seed * 1000 + tag * 100 + x as u64)).collect()
    };
    let gate = mk(dims.hidden, dims.inter, bits_gu, 1);
    let up = mk(dims.hidden, dims.inter, bits_gu, 2);
    let down = mk(dims.inter, dims.hidden, bits_d, 3);
    MoeExperts::from_groups(&gate, &up, &down, dims).unwrap()
}

fn small_dims() -> MoeDims {
    MoeDims { hidden: 512, inter: 256, experts: 32, topk: 8, scale: 2.5, limit: 10.0 }
}

/// A shape whose gate/up plan splits K more than the small one: the GLM widths with fewer experts.
fn glm_dims(experts: usize) -> MoeDims {
    MoeDims { hidden: 4096, inter: 2048, experts, topk: 8, scale: 2.5, limit: 10.0 }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// routing styles

#[derive(Clone, Copy, Debug)]
enum Style {
    /// eight distinct uniformly random experts a row
    Uniform,
    /// a Zipf-like preference: a few hot experts take many rows, the tail is rare
    Skewed,
    /// every row picks the same eight experts (every one of them has `rows` members: the longest group lists)
    Hot,
    /// a quarter of the picks are outside [0, E) (an unrouted slot contributes nothing)
    Holes,
    /// experts 0..8 for a few rows, then nothing but one expert per row (many single-member experts)
    Sparse,
}

fn make_picks(r: &mut Rng, style: Style, rows: usize, e: usize, topk: usize) -> (Vec<i32>, Vec<u16>) {
    let mut pick = Vec::with_capacity(rows * topk);
    let mut wts = Vec::with_capacity(rows * topk);
    for row in 0..rows {
        let ids: Vec<usize> = match style {
            Style::Uniform | Style::Holes => {
                let mut all: Vec<usize> = (0..e).collect();
                for i in (1..all.len()).rev() {
                    all.swap(i, below(r, i + 1));
                }
                all[..topk].to_vec()
            }
            Style::Skewed => {
                let mut chosen: Vec<usize> = Vec::new();
                while chosen.len() < topk {
                    // rank ~ 1 / u: P(rank) falls like 1 / rank^2
                    let u = (r.next_u64() >> 40) as f64 / 16777216.0 + 1e-6;
                    let c = ((1.0 / u) as usize - 1).min(e - 1);
                    if !chosen.contains(&c) {
                        chosen.push(c);
                    }
                }
                chosen
            }
            Style::Hot => (0..topk).collect(),
            Style::Sparse => {
                if row < 3 {
                    (0..topk).collect()
                } else {
                    let mut v: Vec<usize> = (0..topk).map(|k| (row * 5 + k * 13 + 1) % e).collect();
                    v.sort_unstable();
                    v.dedup();
                    while v.len() < topk {
                        v.push((v[v.len() - 1] + 1) % e);
                        v.sort_unstable();
                        v.dedup();
                    }
                    v
                }
            }
        };
        for (k, &id) in ids.iter().enumerate() {
            let out_of_range = matches!(style, Style::Holes) && below(r, 4) == 0;
            pick.push(if out_of_range { -1 + (below(r, 2) as i32) * (e as i32 + 1) } else { id as i32 });
            wts.push(h(0.05 + 0.6 * (r.next_u64() >> 40) as f64 / 16777216.0));
            let _ = k;
        }
    }
    (pick, wts)
}

// ---------------------------------------------------------------------------------------------------------------------------------
// the two paths

struct Run {
    out: Vec<f32>,
    y: Vec<f32>,
}

/// The decode path on `rows` rows cut into windows of at most 128 (what every prompt row went through before this path existed).
fn decode_reference(
    ex: &MoeExperts,
    st: &Stream,
    xh: &[u16],
    pick: &[i32],
    wts: &[u16],
    shared: &[f32],
    rows: usize,
    window: usize,
) -> Run {
    let d = ex.dims;
    let mut s = MoeScratch::new(&d, 128).unwrap();
    let (mut out, mut y) = (Vec::new(), Vec::new());
    let mut at = 0;
    while at < rows {
        let n = window.min(rows - at);
        s.set_picks(&pick[at * d.topk..(at + n) * d.topk], &wts[at * d.topk..(at + n) * d.topk]).unwrap();
        let x = up(&xh[at * d.hidden..(at + n) * d.hidden]);
        let sh = up(&shared[at * d.hidden..(at + n) * d.hidden]);
        let mut o = DevBuf::zeroed(n * d.hidden * 4).unwrap();
        ex.run(&x, Dtype::F16, n, Some(&sh), &mut o, &mut s, true, st).unwrap();
        out.extend(sync_dl::<f32>(&o, st));
        let yy: Vec<f32> = sync_dl(&s.y_pairs, st);
        y.extend_from_slice(&yy[..n * d.topk * d.hidden]);
        at += n;
    }
    Run { out, y }
}

/// The large-row path on the same rows in calls of `chunk` rows (the last may be shorter).
fn prefill_path(
    ex: &MoeExperts,
    st: &Stream,
    s: &mut PrefillScratch,
    xh: &[u16],
    pick: &[i32],
    wts: &[u16],
    shared: &[f32],
    rows: usize,
    chunk: usize,
) -> Run {
    let d = ex.dims;
    let (mut out, mut y) = (Vec::new(), Vec::new());
    let mut at = 0;
    while at < rows {
        let n = chunk.min(rows - at);
        s.set_picks(&pick[at * d.topk..(at + n) * d.topk], &wts[at * d.topk..(at + n) * d.topk]).unwrap();
        let x = up(&xh[at * d.hidden..(at + n) * d.hidden]);
        let sh = up(&shared[at * d.hidden..(at + n) * d.hidden]);
        let mut o = DevBuf::zeroed(n * d.hidden * 4).unwrap();
        // a non-blocking `st` does not wait for the legacy-stream uploads and memsets above
        gb_cuda::rt::device_sync().unwrap();
        ex.run_prefill(&x, Dtype::F16, n, Some(&sh), &mut o, s, st).unwrap();
        out.extend(sync_dl::<f32>(&o, st));
        let yy: Vec<f32> = sync_dl(&s.y_pairs, st);
        y.extend_from_slice(&yy[..n * d.topk * d.hidden]);
        at += n;
    }
    Run { out, y }
}

fn first_diff(a: &[f32], b: &[f32]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x.to_bits() != y.to_bits())
}

/// Assert two fp32 slices are bit-identical (`to_bits`, so signed zero and NaN bit patterns count as differences).
fn assert_bit_eq_f32(what: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    let ab: Vec<u32> = a.iter().map(|v| v.to_bits()).collect();
    let bb: Vec<u32> = b.iter().map(|v| v.to_bits()).collect();
    assert_eq!(ab, bb, "{what}");
}

/// `got` equals `want` bit for bit: every row of `out`, and `y` of every pair that was computed (a pick outside [0, E) writes nothing: its `y` is
/// whatever the scratch held, in both paths).
fn assert_same(what: &str, got: &Run, want: &Run, d: &MoeDims, pick: &[i32]) {
    assert_eq!(got.out.len(), want.out.len(), "{what}: out length");
    for (p, &e) in pick.iter().enumerate() {
        if e < 0 || e as usize >= d.experts {
            continue;
        }
        let r = p * d.hidden..(p + 1) * d.hidden;
        if let Some(i) = first_diff(&got.y[r.clone()], &want.y[r.clone()]) {
            panic!(
                "{what}: pair {p} (row {}, slot {}) column {i} of y differs: {} vs {}",
                p / d.topk,
                p % d.topk,
                got.y[r.start + i],
                want.y[r.start + i]
            );
        }
    }
    if let Some(i) = first_diff(&got.out, &want.out) {
        panic!("{what}: row {} column {} of out differs: {} vs {}", i / d.hidden, i % d.hidden, got.out[i], want.out[i]);
    }
}

struct Inputs {
    xh: Vec<u16>,
    shared: Vec<f32>,
}

fn inputs(r: &mut Rng, rows: usize, hidden: usize, sigma: f64) -> Inputs {
    Inputs { xh: rand_f16(r, rows * hidden, sigma), shared: (0..rows * hidden).map(|_| normal(r) as f32).collect() }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// tests

const STYLES: [Style; 5] = [Style::Uniform, Style::Skewed, Style::Hot, Style::Holes, Style::Sparse];

/// The headline: the large-row path equals the decode path at row counts on both sides of every tile and group size, for every routing
/// style, at every weight width the kernels take.
#[test]
fn prefill_experts_equal_the_decode_path_bit_for_bit() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = small_dims();
    let mut checked = 0usize;
    for (bg, bd) in [(2usize, 2usize), (3, 3), (4, 2), (1, 6), (8, 5)] {
        let ex = synth_layer(d, bg, bd, Codebook::Mul1, 60 + bg as u64);
        let mut s = PrefillScratch::new(&d, 640).unwrap();
        s.poison(0xAB, &st).unwrap();
        let mut r = Rng(70 + bg as u64);
        for &rows in &[1usize, 2, 16, 17, 63, 64, 65, 127, 128, 129, 200, 333, 640] {
            let inp = inputs(&mut r, rows, d.hidden, 2.0);
            for style in STYLES {
                let (pick, wts) = make_picks(&mut r, style, rows, d.experts, d.topk);
                let want = decode_reference(&ex, &st, &inp.xh, &pick, &wts, &inp.shared, rows, 128);
                let got = prefill_path(&ex, &st, &mut s, &inp.xh, &pick, &wts, &inp.shared, rows, rows);
                assert_same(&format!("{bg}/{bd} bit, {rows} rows, {style:?}"), &got, &want, &d, &pick);
                checked += rows;
            }
        }
    }
    eprintln!("{checked} row results bit-identical to the decode path (5 widths, 13 row counts, 5 routing styles)");
}

/// A tight SwiGLU limit, so that the clamps of the activation (silu(gate) from above, up to +-limit) bite on a large share of the elements: the
/// clamp of the large-row epilogue is the decode epilogue's.
#[test]
fn a_tight_swiglu_limit_equals_the_decode_path() {
    gpu!();
    let st = Stream::new().unwrap();
    for limit in [0.05f32, 0.4] {
        let d = MoeDims { limit, ..small_dims() };
        let ex = synth_layer(d, 3, 2, Codebook::Mul1, 170);
        let mut s = PrefillScratch::new(&d, 300).unwrap();
        let mut r = Rng(171);
        for rows in [129usize, 300] {
            let inp = inputs(&mut r, rows, d.hidden, 3.0);
            let (pick, wts) = make_picks(&mut r, Style::Skewed, rows, d.experts, d.topk);
            let want = decode_reference(&ex, &st, &inp.xh, &pick, &wts, &inp.shared, rows, 128);
            let got = prefill_path(&ex, &st, &mut s, &inp.xh, &pick, &wts, &inp.shared, rows, rows);
            assert_same(&format!("limit {limit}, {rows} rows"), &got, &want, &d, &pick);
        }
    }
}

/// Real GLM widths (K = 4096 -> 2048 -> 4096, so the plan's K splits are the production ones), real expert bit widths.
#[test]
fn glm_shaped_prefill_equals_the_decode_path() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = glm_dims(40);
    for bits in [2usize, 3, 4] {
        let ex = synth_layer(d, bits, bits, Codebook::Mul1, 90 + bits as u64);
        let mut s = PrefillScratch::new(&d, 520).unwrap();
        let mut r = Rng(91 + bits as u64);
        for &rows in &[129usize, 257, 520] {
            let inp = inputs(&mut r, rows, d.hidden, 1.5);
            for style in [Style::Uniform, Style::Skewed, Style::Hot] {
                let (pick, wts) = make_picks(&mut r, style, rows, d.experts, d.topk);
                let want = decode_reference(&ex, &st, &inp.xh, &pick, &wts, &inp.shared, rows, 128);
                let got = prefill_path(&ex, &st, &mut s, &inp.xh, &pick, &wts, &inp.shared, rows, rows);
                assert_same(&format!("GLM widths {bits} bit, {rows} rows, {style:?}"), &got, &want, &d, &pick);
            }
        }
    }
}

/// Nothing in the kernels is GLM's: other expert counts (not a power of two, 64 .. 128), widths that are not powers of two (K 1,536 / inter
/// 640, 768, 384), top-k 6, 8 and the largest the kernels take (16), a model with no SwiGLU clamp (`limit` 0), another routed scale, and the
/// weight widths of other packs. Each equals the decode path bit for bit on uniform, skewed, one-hot and unrouted picks.
#[test]
fn other_models_shapes_equal_the_decode_path() {
    gpu!();
    let st = Stream::new().unwrap();
    let shapes = [
        MoeDims { hidden: 2048, inter: 768, experts: 128, topk: 8, scale: 1.0, limit: 0.0 },
        MoeDims { hidden: 1536, inter: 640, experts: 100, topk: 6, scale: 1.7, limit: 7.0 },
        MoeDims { hidden: 512, inter: 384, experts: 64, topk: 16, scale: 1.0, limit: 0.0 },
    ];
    let mut checked = 0usize;
    for (si, d) in shapes.into_iter().enumerate() {
        for (bg, bd) in [(2usize, 2usize), (3, 4)] {
            let ex = synth_layer(d, bg, bd, Codebook::Mul1, 300 + 10 * si as u64 + bg as u64);
            let mut s = PrefillScratch::new(&d, 420).unwrap();
            let mut r = Rng(310 + 10 * si as u64 + bg as u64);
            for &rows in &[129usize, 257, 420] {
                let inp = inputs(&mut r, rows, d.hidden, 1.5);
                for style in [Style::Uniform, Style::Skewed, Style::Hot, Style::Holes] {
                    let (pick, wts) = make_picks(&mut r, style, rows, d.experts, d.topk);
                    let want = decode_reference(&ex, &st, &inp.xh, &pick, &wts, &inp.shared, rows, 128);
                    let got = prefill_path(&ex, &st, &mut s, &inp.xh, &pick, &wts, &inp.shared, rows, rows);
                    assert_same(&format!("shape {si} ({bg}/{bd} bit), {rows} rows, {style:?}"), &got, &want, &d, &pick);
                    checked += rows;
                }
            }
        }
    }
    eprintln!("{checked} row results of 3 non-GLM shapes bit-identical to the decode path");
}

/// The same rows cut into calls in different ways (a chunk grid) give the same bits, and they are the decode path's.
#[test]
fn the_chunk_grid_changes_no_bit() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = small_dims();
    let ex = synth_layer(d, 3, 3, Codebook::Mul1, 120);
    let mut s = PrefillScratch::new(&d, 700).unwrap();
    let mut r = Rng(121);
    let rows = 700;
    let inp = inputs(&mut r, rows, d.hidden, 2.0);
    for style in [Style::Skewed, Style::Uniform] {
        let (pick, wts) = make_picks(&mut r, style, rows, d.experts, d.topk);
        let want = decode_reference(&ex, &st, &inp.xh, &pick, &wts, &inp.shared, rows, 128);
        for chunk in [700usize, 384, 256, 129, 128, 100, 33, 16, 1] {
            let got = prefill_path(&ex, &st, &mut s, &inp.xh, &pick, &wts, &inp.shared, rows, chunk);
            assert_same(&format!("{style:?}, chunk {chunk}"), &got, &want, &d, &pick);
        }
    }
}

/// The order of the rows is the order of the pairs inside every expert's group (the sort is by expert id, then pair index), so permuting the rows
/// permutes every group: no row's bits may change. (This is the invariant a grouping-order mutant cannot break: the order inside a group is
/// bit-neutral by construction, and this shows it, rather than a test the mutant would fail.)
#[test]
fn the_order_of_the_rows_changes_no_bit() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = small_dims();
    let ex = synth_layer(d, 2, 3, Codebook::Mul1, 150);
    let mut s = PrefillScratch::new(&d, 400).unwrap();
    let mut r = Rng(151);
    let rows = 400;
    let inp = inputs(&mut r, rows, d.hidden, 2.0);
    for style in [Style::Skewed, Style::Hot, Style::Uniform] {
        let (pick, wts) = make_picks(&mut r, style, rows, d.experts, d.topk);
        let base = prefill_path(&ex, &st, &mut s, &inp.xh, &pick, &wts, &inp.shared, rows, rows);
        // a random permutation of the rows, and the reverse
        let mut perms: Vec<Vec<usize>> = vec![(0..rows).rev().collect()];
        let mut p: Vec<usize> = (0..rows).collect();
        for i in (1..rows).rev() {
            p.swap(i, below(&mut r, i + 1));
        }
        perms.push(p);
        for perm in perms {
            let (mut xh, mut sh, mut pk, mut wt) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for &o in &perm {
                xh.extend_from_slice(&inp.xh[o * d.hidden..(o + 1) * d.hidden]);
                sh.extend_from_slice(&inp.shared[o * d.hidden..(o + 1) * d.hidden]);
                pk.extend_from_slice(&pick[o * d.topk..(o + 1) * d.topk]);
                wt.extend_from_slice(&wts[o * d.topk..(o + 1) * d.topk]);
            }
            let got = prefill_path(&ex, &st, &mut s, &xh, &pk, &wt, &sh, rows, rows);
            for (j, &o) in perm.iter().enumerate() {
                assert!(
                    got.out[j * d.hidden..(j + 1) * d.hidden] == base.out[o * d.hidden..(o + 1) * d.hidden],
                    "{style:?}: row {o} placed at {j} has other bits"
                );
                let (a, b) = (j * d.topk * d.hidden, (j + 1) * d.topk * d.hidden);
                let (c, e) = (o * d.topk * d.hidden, (o + 1) * d.topk * d.hidden);
                assert!(got.y[a..b] == base.y[c..e], "{style:?}: the pair outputs of row {o} placed at {j} have other bits");
            }
        }
    }
}

/// A scratch dirty from a wide call, and one sized for more rows than the call has: nothing of the earlier call leaks into the next.
#[test]
fn a_dirty_or_oversized_scratch_changes_no_bit() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = small_dims();
    let ex = synth_layer(d, 2, 4, Codebook::Mul1, 130);
    let mut r = Rng(131);
    let mut big = PrefillScratch::new(&d, 512).unwrap();
    let mut exact = PrefillScratch::new(&d, 77).unwrap();
    let wide = inputs(&mut r, 512, d.hidden, 2.0);
    let (wp, ww) = make_picks(&mut r, Style::Skewed, 512, d.experts, d.topk);
    let _ = prefill_path(&ex, &st, &mut big, &wide.xh, &wp, &ww, &wide.shared, 512, 512);
    let inp = inputs(&mut r, 77, d.hidden, 2.0);
    let (pick, wts) = make_picks(&mut r, Style::Uniform, 77, d.experts, d.topk);
    let want = decode_reference(&ex, &st, &inp.xh, &pick, &wts, &inp.shared, 77, 128);
    let a = prefill_path(&ex, &st, &mut big, &inp.xh, &pick, &wts, &inp.shared, 77, 77);
    let b = prefill_path(&ex, &st, &mut exact, &inp.xh, &pick, &wts, &inp.shared, 77, 77);
    assert_same("dirty wide scratch", &a, &want, &d, &pick);
    assert_same("exact scratch", &b, &want, &d, &pick);
}

/// The detector that makes the equality meaningful: a K split other than the shape's changes the bits (so a plan that depended on the
/// row count, or a kernel that took a different split, could not pass the tests above).
#[test]
fn a_wrong_k_split_is_caught() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = glm_dims(16);
    let mut ex = synth_layer(d, 3, 3, Codebook::Mul1, 140);
    let mut r = Rng(141);
    let rows = 200;
    let inp = inputs(&mut r, rows, d.hidden, 1.5);
    let (pick, wts) = make_picks(&mut r, Style::Uniform, rows, d.experts, d.topk);
    let want = decode_reference(&ex, &st, &inp.xh, &pick, &wts, &inp.shared, rows, 128);
    let mut s = PrefillScratch::new(&d, rows).unwrap();
    let right = prefill_path(&ex, &st, &mut s, &inp.xh, &pick, &wts, &inp.shared, rows, rows);
    assert_same("the plan's own split", &right, &want, &d, &pick);
    let (sk, wk) = ex.plan_gu;
    assert!(sk >= 2, "the GLM gate/up plan splits K");
    ex.plan_gu = (sk / 2, wk);
    let wrong = prefill_path(&ex, &st, &mut s, &inp.xh, &pick, &wts, &inp.shared, rows, rows);
    assert!(
        first_diff(&wrong.y, &want.y).is_some() || first_diff(&wrong.out, &want.out).is_some(),
        "a halved K split left every bit unchanged"
    );
}

/// The scratch's own size arithmetic equals what it allocates.
#[test]
fn the_scratch_size_arithmetic_equals_the_allocation() {
    gpu!();
    for (d, rows) in [(small_dims(), 100usize), (small_dims(), 640), (glm_dims(288), 256)] {
        let s = PrefillScratch::new(&d, rows).unwrap();
        assert_eq!(s.bytes(), PrefillScratch::bytes_for(&d, rows).unwrap(), "{d:?} {rows}");
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// speed

/// Real routing: an oracle dump's `route_ids.npy` ([layers, rows, 8] int16) at layer index `li`, tiled to `rows` rows; or a Zipf-like draw.
fn bench_picks(r: &mut Rng, rows: usize, e: usize, topk: usize, li: usize) -> (Vec<i32>, &'static str) {
    if let Ok(dir) = std::env::var("REXL3_ROUTE_DIR") {
        let p = std::path::Path::new(&dir).join("route_ids.npy");
        if p.exists() {
            let n = read_npy(&p);
            let (layers, t) = (n.shape[0], n.shape[1]);
            let li = li % layers;
            let v: Vec<i32> = n.data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as i32).collect();
            let mut out = Vec::with_capacity(rows * topk);
            for row in 0..rows {
                out.extend_from_slice(&v[(li * t + row % t) * topk..(li * t + row % t + 1) * topk]);
            }
            return (out, "real routing");
        }
    }
    let (p, _) = make_picks(r, Style::Skewed, rows, e, topk);
    (p, "zipf routing")
}

/// Per-stage milliseconds of the large-row pipeline against the decode path cut into 128-row windows (the way a prompt is prefilled today), on
/// a GLM-shaped layer of 288 synthetic experts (2-bit), L2 flushed before each call.
#[test]
#[ignore]
fn bench_prefill_experts() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = glm_dims(288);
    let bits: usize = std::env::var("REXL3_BENCH_BITS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let ex = synth_layer(d, bits, bits, Codebook::Mul1, 7);
    let flush = DevBuf::zeroed(256 << 20).unwrap();
    let mut flush_out = DevBuf::zeroed(16).unwrap();
    let iters = 3;
    let mut r = Rng(5);
    println!("GLM-shaped layer, {bits}-bit, 288 experts, expert bytes {:.2} GB", ex.expert_bytes as f64 * 288.0 / 1e9);
    // Plan-builder selector for the GB10 ON/OFF A/B: setting REXL3_BENCH_PLAN_PARALLEL measures the parallel
    // builder (gb_moe_pf_plan_parallel); unset (default) measures the legacy single-block plan_kernel.
    let plan_parallel = std::env::var("REXL3_BENCH_PLAN_PARALLEL").is_ok();
    for rows in [128usize, 256, 512, 1024, 2048] {
        let mut s = PrefillScratch::new(&d, rows).unwrap();
        s.use_parallel_plan(plan_parallel);
        let inp = inputs(&mut r, rows, d.hidden, 1.0);
        let x = up(&inp.xh);
        let shared = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
        let mut out = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
        let (pick, src) = bench_picks(&mut r, rows, d.experts, d.topk, 6);
        let wts: Vec<u16> = (0..rows * d.topk).map(|_| h(0.1)).collect();
        s.set_picks(&pick, &wts).unwrap();
        let t = EventTimer::new(8).unwrap();
        let mut stage = [0f64; 5];
        let names = ["plan", "rot_in", "gate_up", "down", "combine"];
        let mut total = 0f64;
        for it in 0..iters + 1 {
            gb_cuda::moe::read_probe(&flush, 256 << 20, &mut flush_out, &st).unwrap();
            t.record(0, &st).unwrap();
            let mut n = 0;
            ex.run_prefill_marked(&x, Dtype::F16, rows, Some(&shared), &mut out, &mut s, &st, &mut |_| {
                n += 1;
                t.record(n, &st)
            })
            .unwrap();
            if it > 0 {
                for i in 0..5 {
                    stage[i] += t.ms(i, i + 1).unwrap() as f64 / iters as f64;
                }
                total += t.ms(0, 5).unwrap() as f64 / iters as f64;
            }
        }
        // the decode path in 128-row windows
        let mut ds = MoeScratch::new(&d, 128).unwrap();
        let mut dtot = 0f64;
        for it in 0..iters + 1 {
            gb_cuda::moe::read_probe(&flush, 256 << 20, &mut flush_out, &st).unwrap();
            t.record(0, &st).unwrap();
            let mut at = 0;
            while at < rows {
                let n = 128.min(rows - at);
                ds.set_picks(&pick[at * d.topk..(at + n) * d.topk], &wts[at * d.topk..(at + n) * d.topk]).unwrap();
                let xw = up(&inp.xh[at * d.hidden..(at + n) * d.hidden]);
                let mut o = DevBuf::zeroed(n * d.hidden * 4).unwrap();
                ex.run(&xw, Dtype::F16, n, None, &mut o, &mut ds, false, &st).unwrap();
                at += n;
            }
            t.record(1, &st).unwrap();
            if it > 0 {
                dtot += t.ms(0, 1).unwrap() as f64 / iters as f64;
            }
        }
        let distinct = {
            let mut v = pick.clone();
            v.sort_unstable();
            v.dedup();
            v.len()
        };
        st.sync().unwrap();
        let ic = s.item_counts().unwrap();
        let flop = rows as f64 * 8.0 * 3.0 * 2.0 * 4096.0 * 2048.0;
        println!(
            "R={rows:5} ({src}, {distinct} experts, items 1/2/3-4 tiles: {}/{}/{}) new {total:8.2} ms = {:6.1} TFLOP/s | stages: {} | decode path in 128-row windows {dtot:8.2} ms | {:.1}x",
            ic[0],
            ic[1],
            ic[2],
            flop / 1e9 / total,
            names.iter().zip(stage).map(|(n, m)| format!("{n} {m:.2}")).collect::<Vec<_>>().join(", "),
            dtot / total
        );
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// the golden (scripts/gpu_gate.sh): hashes of the large-row path's outputs on synthetic layers. They equal the decode path's by the tests above, and
// are committed so that a change to either kernel family that moves a bit shows up in the gate even if both moved together.

fn hash_f32(v: &[f32]) -> u64 {
    gb_exl3::fnv1a(v.iter().flat_map(|x| x.to_bits().to_le_bytes()))
}

#[test]
#[ignore]
fn golden_moe_prefill_hashes() {
    gpu!();
    let path = std::env::var("REXL3_GOLDEN_OUT").expect("REXL3_GOLDEN_OUT");
    let st = Stream::new().unwrap();
    let mut lines = Vec::new();
    let d = small_dims();
    let cases: [(usize, usize, Codebook); 5] =
        [(2, 2, Codebook::Mul1), (3, 3, Codebook::Mul1), (4, 2, Codebook::Mul1), (3, 2, Codebook::Inst3), (4, 4, Codebook::Mcg)];
    for (i, &(bg, bd, cb)) in cases.iter().enumerate() {
        let ex = synth_layer(d, bg, bd, cb, 400 + i as u64);
        let mut s = PrefillScratch::new(&d, 640).unwrap();
        let mut r = Rng(410 + i as u64);
        for rows in [129usize, 300, 640] {
            let inp = inputs(&mut r, rows, d.hidden, 2.0);
            for style in STYLES {
                let (pick, wts) = make_picks(&mut r, style, rows, d.experts, d.topk);
                let got = prefill_path(&ex, &st, &mut s, &inp.xh, &pick, &wts, &inp.shared, rows, rows);
                // the y of a pair that was never computed is whatever the scratch held: hash the computed ones
                let mut y = got.y.clone();
                for (p, &e) in pick.iter().enumerate() {
                    if e < 0 || e as usize >= d.experts {
                        y[p * d.hidden..(p + 1) * d.hidden].fill(0.0);
                    }
                }
                lines.push(format!(
                    "prefill experts gu={bg} d={bd} {cb:?} rows={rows} {style:?} out {:016x} y {:016x}",
                    hash_f32(&got.out),
                    hash_f32(&y)
                ));
            }
        }
    }
    // the GLM widths (K = 4096 -> 2048 -> 4096: the production K splits), a few experts
    let g = glm_dims(40);
    let ex = synth_layer(g, 2, 2, Codebook::Mul1, 450);
    let mut s = PrefillScratch::new(&g, 520).unwrap();
    let mut r = Rng(451);
    for rows in [129usize, 520] {
        let inp = inputs(&mut r, rows, g.hidden, 1.5);
        let (pick, wts) = make_picks(&mut r, Style::Skewed, rows, g.experts, g.topk);
        let got = prefill_path(&ex, &st, &mut s, &inp.xh, &pick, &wts, &inp.shared, rows, rows);
        lines.push(format!("prefill experts glm widths rows={rows} out {:016x} y {:016x}", hash_f32(&got.out), hash_f32(&got.y)));
    }
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    eprintln!("wrote {} golden lines to {path}", lines.len());
}

/// The router at the row counts of a prefill chunk: the logits kernel's blockings (same bits, tested in moe.rs) and the select kernel, in microseconds,
/// L2 warm (a chunk's rows reuse the gate matrix) so that what is measured is the kernel, not the 2.4 MB read.
#[test]
#[ignore]
fn bench_router_big() {
    gpu!();
    let st = Stream::new().unwrap();
    let tab = SigmoidTable::new().unwrap();
    let mut r = Rng(31);
    let (e, k) = (288usize, 4096usize);
    let gate = rand_f16(&mut r, e * k, 0.03);
    let esb: Vec<f32> = (0..e).map(|_| (34.0 + 0.3 * normal(&mut r)) as f32).collect();
    let router = MoeRouter::new(&gate, &centred_bias_host(&esb), k, 8, 2.5).unwrap();
    let shapes: [(usize, usize); 8] = [(4, 1), (8, 1), (4, 4), (8, 4), (4, 8), (8, 8), (2, 8), (2, 4)];
    let t = EventTimer::new(2).unwrap();
    for rows in [128usize, 512, 2048, 4096] {
        let x = up(&rand_f16(&mut r, rows * k, 1.0));
        let mut reference: Option<Vec<u16>> = None;
        let mut logits = DevBuf::zeroed(rows * e * 2).unwrap();
        let mut line = Vec::new();
        let mut best = (f64::MAX, (0, 0, 0));
        for rb in [1usize, 2, 4] {
            for &(warps, un) in &shapes {
                router.logits_blocked(&x, Dtype::F16, rows, &mut logits, rb, warps, un, &st).unwrap();
                t.record(0, &st).unwrap();
                for _ in 0..5 {
                    router.logits_blocked(&x, Dtype::F16, rows, &mut logits, rb, warps, un, &st).unwrap();
                }
                t.record(1, &st).unwrap();
                let us = t.ms(0, 1).unwrap() as f64 * 1e3 / 5.0;
                let got: Vec<u16> = sync_dl(&logits, &st);
                match &reference {
                    None => reference = Some(got),
                    Some(want) => assert!(&got == want, "rows {rows}: shape rb={rb} warps={warps} un={un} changes the logits"),
                }
                if us < best.0 {
                    best = (us, (rb, warps, un));
                }
                line.push(format!("rb{rb}w{warps}u{un} {us:.0}"));
            }
        }
        // the prefill tiling (several experts a warp): same bits, timed the same way
        let mut best_eb = (f64::MAX, (0, 0, 0, 0));
        for (rb, eb, warps, un) in [
            (2usize, 4usize, 4usize, 1usize),
            (2, 8, 4, 1),
            (2, 8, 8, 1),
            (4, 4, 4, 1),
            (4, 4, 8, 1),
            (4, 8, 4, 1),
            (4, 8, 8, 1),
            (4, 8, 4, 2),
            (4, 8, 8, 2),
            (8, 4, 4, 1),
            (8, 4, 8, 1),
            (8, 8, 4, 1),
            (8, 8, 8, 1),
            (8, 8, 4, 2),
            (8, 4, 4, 2),
        ] {
            router.logits_blocked_eb(&x, Dtype::F16, rows, &mut logits, rb, eb, warps, un, &st).unwrap();
            t.record(0, &st).unwrap();
            for _ in 0..5 {
                router.logits_blocked_eb(&x, Dtype::F16, rows, &mut logits, rb, eb, warps, un, &st).unwrap();
            }
            t.record(1, &st).unwrap();
            let us = t.ms(0, 1).unwrap() as f64 * 1e3 / 5.0;
            let got: Vec<u16> = sync_dl(&logits, &st);
            assert!(
                Some(&got) == reference.as_ref(),
                "rows {rows}: shape rb={rb} eb={eb} warps={warps} un={un} changes the logits"
            );
            if us < best_eb.0 {
                best_eb = (us, (rb, eb, warps, un));
            }
            line.push(format!("rb{rb}e{eb}w{warps}u{un} {us:.0}"));
        }
        // the launcher's default (what the model runs)
        router.logits(&x, Dtype::F16, rows, &mut logits, &st).unwrap();
        t.record(0, &st).unwrap();
        for _ in 0..5 {
            router.logits(&x, Dtype::F16, rows, &mut logits, &st).unwrap();
        }
        t.record(1, &st).unwrap();
        let def = t.ms(0, 1).unwrap() as f64 * 1e3 / 5.0;
        assert!(Some(&sync_dl::<u16>(&logits, &st)) == reference.as_ref(), "rows {rows}: the default launch changes the logits");
        println!(
            "ROUTERBIG-EB R={rows:4}: one expert a warp best {:.0} us at {:?}; several best {:.0} us at {:?}; the default {def:.0} us",
            best.0, best.1, best_eb.0, best_eb.1
        );
        let mut pick = DevBuf::zeroed(rows * 8 * 4).unwrap();
        let mut wts = DevBuf::zeroed(rows * 8 * 2).unwrap();
        router.select(&tab, &logits, rows, &mut pick, &mut wts, &st).unwrap();
        t.record(0, &st).unwrap();
        for _ in 0..5 {
            router.select(&tab, &logits, rows, &mut pick, &mut wts, &st).unwrap();
        }
        t.record(1, &st).unwrap();
        let sel = t.ms(0, 1).unwrap() as f64 * 1e3 / 5.0;
        println!("ROUTERBIG R={rows:4}: logits best {:.0} us at {:?}, select {sel:.0} us | {}", best.0, best.1, line.join(" "));
    }
    // the DSA indexer's fp16 GEMVs of a prompt chunk (wk / compress_gate 4096 -> 128, weights_proj 4096 -> 32) run the same launcher: the
    // former default from 32 rows (four rows a warp, one expert, eight warps, one load) against today's
    for n in [128usize, 32] {
        let w = rand_f16(&mut r, n * k, 0.03);
        let gemv = MoeRouter::new(&w, &vec![0.0f32; n], k, 8, 1.0).unwrap();
        for rows in [512usize, 2048, 4096] {
            let x = up(&rand_f16(&mut r, rows * k, 1.0));
            let mut a = DevBuf::zeroed(rows * n * 2).unwrap();
            let mut b = DevBuf::zeroed(rows * n * 2).unwrap();
            let time = |f: &mut dyn FnMut()| {
                f();
                t.record(0, &st).unwrap();
                for _ in 0..5 {
                    f();
                }
                t.record(1, &st).unwrap();
                t.ms(0, 1).unwrap() as f64 * 1e3 / 5.0
            };
            let old = time(&mut || gemv.logits_blocked(&x, Dtype::F16, rows, &mut a, 4, 8, 1, &st).unwrap());
            let new = time(&mut || gemv.logits(&x, Dtype::F16, rows, &mut b, &st).unwrap());
            assert!(sync_dl::<u16>(&a, &st) == sync_dl::<u16>(&b, &st), "n {n} rows {rows}: the default changes the bits");
            println!("ROUTERBIG-GEMV N={n:3} R={rows:4}: former default {old:.0} us, today's {new:.0} us");
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Qwen Flash-Next grouped prefill (REXL3_FLASH_NEXT_GROUPED_MOE_PREFILL): the constructor from the per-expert
// Exl3Weights the Qwen loader keeps (MoeExperts::from_exl3), the route representation (u32 ids -> i32 picks, f32
// weights kept f32 end-to-end), and the shared-expert finish (one fp32 -> bf16 cast). Run on a GPU box, not in the stub build.

use gb_cuda::dsv4_attn::{bf16_bits, bf16_bits_to_f32};
use gb_cuda::exl3_linear::Exl3Weights;
use gb_cuda::moe_prefill::{finish_shared, wts_f32_f16};
use gb_exl3::f16::f16_to_f64;

fn qwen_dims() -> MoeDims {
    MoeDims { hidden: 2560, inter: 640, experts: 512, topk: 10, scale: 1.0, limit: 0.0 }
}

fn refs_all(gs: &[Exl3Weights]) -> Vec<&Exl3Weights> {
    gs.iter().collect()
}

/// [`MoeExperts::from_exl3`] is the same object as [`MoeExperts::from_groups`] on the same weights: pointer tables at
/// the same trellises, stacked scales with the same fp16 bits, same plans, and one run through both is identical.
#[test]
fn from_exl3_matches_from_groups_on_the_same_weights() {
    gpu!();
    let dims = qwen_dims();
    let mk = |k, n, bits, tag: u64| -> Vec<LoadedGroup> {
        (0..dims.experts).map(|x| synth_group(k, n, bits, Codebook::Mul1, 7000 + tag * 1000 + x as u64)).collect()
    };
    let gate = mk(dims.hidden, dims.inter, 2, 1);
    let upg = mk(dims.hidden, dims.inter, 2, 2);
    let down = mk(dims.inter, dims.hidden, 2, 3);
    let from_groups = MoeExperts::from_groups(&gate, &upg, &down, dims).unwrap();
    let w = |gs: &[LoadedGroup]| -> Vec<Exl3Weights> { gs.iter().map(|g| Exl3Weights::from_loaded(g).unwrap()).collect() };
    let (wg, wu, wd) = (w(&gate), w(&upg), w(&down));
    let from_exl3 = MoeExperts::from_exl3(&refs_all(&wg), &refs_all(&wu), &refs_all(&wd), dims).unwrap();

    assert_eq!(from_exl3.dims, dims);
    assert_eq!(from_exl3.bits_gu, from_groups.bits_gu);
    assert_eq!(from_exl3.bits_d, from_groups.bits_d);
    assert_eq!(from_exl3.codebook, from_groups.codebook);
    assert_eq!(from_exl3.plan_gu, from_groups.plan_gu);
    assert_eq!(from_exl3.plan_d, from_groups.plan_d);
    assert_eq!(from_exl3.expert_bytes, from_groups.expert_bytes);
    // Fused gate/up is structural here (the constructors group by equal gate/up widths), so the
    // public `holds_every_expert` pin suffices; `fused_gate_up` is an internal bank detail.
    assert!(from_exl3.holds_every_expert());

    let st = Stream::new_nonblocking().unwrap();

    // the stacked scales hold the same fp16 bits, buffer for buffer
    for (name, a, b) in [
        ("suh_g", &from_exl3.suh_g, &from_groups.suh_g),
        ("suh_u", &from_exl3.suh_u, &from_groups.suh_u),
        ("svh_g", &from_exl3.svh_g, &from_groups.svh_g),
        ("svh_u", &from_exl3.svh_u, &from_groups.svh_u),
        ("suh_d", &from_exl3.suh_d, &from_groups.suh_d),
        ("svh_d", &from_exl3.svh_d, &from_groups.svh_d),
    ] {
        assert_eq!(a.bytes(), b.bytes(), "{name} bytes");
        let va: Vec<u16> = sync_dl(a, &st);
        let vb: Vec<u16> = sync_dl(b, &st);
        assert_eq!(va, vb, "{name}: the stacked scales differ");
    }

    // one prefill through both, on the same picks/weights, is identical
    let rows = 16;
    let mut r = Rng(9);
    let (pick, wts) = make_picks(&mut r, Style::Uniform, rows, dims.experts, dims.topk);
    let xh = rand_f16(&mut r, rows * dims.hidden, 1.0);
    let x = up(&xh);
    let shared = up(&(0..rows * dims.hidden).map(|i| ((i % 97) as f32) * 0.01).collect::<Vec<f32>>());
    let run = |ex: &MoeExperts| {
        let mut s = PrefillScratch::new(&dims, rows).unwrap();
        s.set_picks(&pick, &wts).unwrap();
        let mut o = DevBuf::zeroed(rows * dims.hidden * 4).unwrap();
        // a non-blocking `st` does not wait for the legacy-stream uploads and memsets above
        gb_cuda::rt::device_sync().unwrap();
        ex.run_prefill(&x, Dtype::F16, rows, Some(&shared), &mut o, &mut s, &st).unwrap();
        sync_dl::<f32>(&o, &st)
    };
    assert_eq!(run(&from_groups), run(&from_exl3), "from_exl3 and from_groups run identically");
}

/// The route representation: `wts_f32_f16` is fp16 round-to-nearest-even of the router's f32 weights, and
/// `finish_shared` is `bf16(routed + sigmoid(gate) * shared)` — exact at gate 0 (± 0.5) and at ±inf.
#[test]
fn qwen_wts_convert_and_finish_shared_match_the_host_mirror() {
    gpu!();
    let st = Stream::new_nonblocking().unwrap();
    // wts: f32 -> fp16 exactly __float2half_rn (f64_to_f16 on the same value)
    let n = 1000usize;
    let src: Vec<f32> = (0..n).map(|i| ((i % 97) as f32) * 0.01 - 0.25).collect();
    let in_b = up(&src);
    let mut out_b = DevBuf::zeroed(n * 2).unwrap();
    // a non-blocking `st` does not wait for the legacy-stream upload and memset above
    gb_cuda::rt::device_sync().unwrap();
    wts_f32_f16(&in_b, &mut out_b, n, &st).unwrap();
    let got: Vec<u16> = sync_dl(&out_b, &st);
    for (i, (&v, &g)) in src.iter().zip(&got).enumerate() {
        assert_eq!(g, f64_to_f16(v as f64), "wts {i}");
    }
    // finish: exact at gate 0 (sigmoid == 0.5) — the host mirror is bit-exact
    let (rows, d) = (8usize, 512usize);
    let routed: Vec<f32> = (0..rows * d).map(|i| ((i % 101) as f32) * 0.01 - 0.5).collect();
    let shared: Vec<f32> = (0..rows * d).map(|i| ((i % 89) as f32) * 0.02 - 0.8).collect();
    let gate: Vec<u16> = vec![0u16; rows]; // bf16 0.0
    let routed_b = up(&routed);
    let shared_b = up(&shared);
    let gate_b = up(&gate);
    let mut out_b = DevBuf::zeroed(rows * d * 2).unwrap();
    gb_cuda::rt::device_sync().unwrap();
    finish_shared(&routed_b, &shared_b, &gate_b, &mut out_b, rows, d, &st).unwrap();
    let got: Vec<u16> = sync_dl(&out_b, &st);
    for i in 0..rows * d {
        let expect = bf16_bits(routed[i] + shared[i] * 0.5);
        assert_eq!(got[i], expect, "finish {i}: gate 0");
        // sanity: the separate-rounds variant (what a pre-rounded shared would give) differs somewhere
        let separate = bf16_bits(bf16_bits_to_f32(bf16_bits(routed[i])) + bf16_bits_to_f32(bf16_bits(shared[i])) * 0.5);
        if separate != expect {
            break;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// the f32-weight variant (`MoeExperts::run_prefill_f32_wts`): the router's f32 weights stay f32 end-to-end. The tests
// pin the combine's semantics against a host mirror (the kernel's own expression: fmaf(w_k, y_k, acc) in slot order,
// then + shared), the identity with the fp16 path on weights that are exactly representable in fp16, and that rounded
// weights do change the bits (the regression the variant fixes).

/// The large-row f32-weight path (mirrors [`prefill_path`] with `set_picks_f32` + `run_prefill_f32_wts`).
fn prefill_path_f32(
    ex: &MoeExperts,
    st: &Stream,
    s: &mut PrefillScratch,
    xh: &[u16],
    pick: &[i32],
    wts_f32: &[f32],
    shared: &[f32],
    rows: usize,
    chunk: usize,
) -> Run {
    let d = ex.dims;
    let (mut out, mut y) = (Vec::new(), Vec::new());
    let mut at = 0;
    while at < rows {
        let n = chunk.min(rows - at);
        s.set_picks_f32(&pick[at * d.topk..(at + n) * d.topk], &wts_f32[at * d.topk..(at + n) * d.topk]).unwrap();
        let x = up(&xh[at * d.hidden..(at + n) * d.hidden]);
        let sh = up(&shared[at * d.hidden..(at + n) * d.hidden]);
        let mut o = DevBuf::zeroed(n * d.hidden * 4).unwrap();
        // a non-blocking `st` does not wait for the legacy-stream uploads and memsets above
        gb_cuda::rt::device_sync().unwrap();
        ex.run_prefill_f32_wts(&x, Dtype::F16, n, Some(&sh), &mut o, s, st).unwrap();
        out.extend(sync_dl::<f32>(&o, st));
        let yy: Vec<f32> = sync_dl(&s.y_pairs, st);
        y.extend_from_slice(&yy[..n * d.topk * d.hidden]);
        at += n;
    }
    Run { out, y }
}

/// The host mirror of the kernel's f32 combine: for every column, acc = fmaf(w_k, y_k, acc) over the row's slots in
/// slot order (a pick outside [0, E) contributes nothing), then `+ shared` last — exactly `combine_f32_kernel`'s
/// expression, on the GPU's own `y` values (read back from `y_pairs`).
fn combine_f32_mirror(pick: &[i32], wts_f32: &[f32], shared: &[f32], y: &[f32], d: &MoeDims) -> Vec<f32> {
    let rows = pick.len() / d.topk;
    let mut out = vec![0.0f32; rows * d.hidden];
    for r in 0..rows {
        for n in 0..d.hidden {
            let mut acc = 0.0f32;
            for k in 0..d.topk {
                let e = pick[r * d.topk + k];
                if e >= 0 && (e as usize) < d.experts {
                    let w = wts_f32[r * d.topk + k];
                    let yv = y[(r * d.topk + k) * d.hidden + n];
                    acc = w.mul_add(yv, acc); // fmaf: one rounding
                }
            }
            out[r * d.hidden + n] = acc + shared[r * d.hidden + n];
        }
    }
    out
}

/// Random f32 router weights in (0, 1): softmax-like, essentially never exactly representable in fp16.
fn rand_wts_f32(r: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| ((0.25 * normal(r)) + 0.5).clamp(0.0, 1.0) as f32).collect()
}

/// The f32 variant's combine is the kernel's own expression: `out` equals the host mirror bit for bit on every row, and
/// the fp16-rounded variant differs somewhere on the same picks (the rounding the variant removes).
#[test]
fn qwen_f32_wts_prefill_matches_the_host_mirror() {
    gpu!();
    let st = Stream::new_nonblocking().unwrap();
    let d = qwen_dims();
    let ex = synth_layer(d, 2, 2, Codebook::Mul1, 810);
    let rows = 129;
    let mut s = PrefillScratch::new(&d, rows).unwrap();
    let mut r = Rng(811);
    let inp = inputs(&mut r, rows, d.hidden, 1.5);
    let (pick, _) = make_picks(&mut r, Style::Uniform, rows, d.experts, d.topk);
    let wts_f32 = rand_wts_f32(&mut r, rows * d.topk);
    let got = prefill_path_f32(&ex, &st, &mut s, &inp.xh, &pick, &wts_f32, &inp.shared, rows, rows);
    let want = combine_f32_mirror(&pick, &wts_f32, &inp.shared, &got.y, &d);
    if let Some(i) = first_diff(&got.out, &want) {
        panic!("f32 combine: row {} column {} differs: {} vs {}", i / d.hidden, i % d.hidden, got.out[i], want[i]);
    }
    // the fp16-rounded path on the same picks/weights differs somewhere (what this variant is for)
    let wts_f16: Vec<u16> = wts_f32.iter().map(|&w| f64_to_f16(w as f64)).collect();
    s.set_picks(&pick, &wts_f16).unwrap();
    let got16 = prefill_path(&ex, &st, &mut s, &inp.xh, &pick, &wts_f16, &inp.shared, rows, rows);
    assert!(first_diff(&got16.out, &got.out).is_some(), "fp16-rounded weights left every out bit unchanged at {rows} rows");
    eprintln!("f32-vs-fp16 out first diff at element {}", first_diff(&got16.out, &got.out).unwrap());
}

/// On weights that are exactly representable in fp16, the f32 and fp16 variants are the same bits (the only difference is
/// the weight rounding, not the pipeline).
#[test]
fn qwen_f32_and_fp16_wts_paths_agree_on_exact_fp16_weights() {
    gpu!();
    let st = Stream::new_nonblocking().unwrap();
    let d = qwen_dims();
    let ex = synth_layer(d, 3, 3, Codebook::Mul1, 820);
    let rows = 257;
    let mut s = PrefillScratch::new(&d, rows).unwrap();
    let mut r = Rng(821);
    let inp = inputs(&mut r, rows, d.hidden, 1.5);
    let (pick, wts_f16) = make_picks(&mut r, Style::Skewed, rows, d.experts, d.topk);
    // the f32 image of each fp16 word is exactly representable: fp16(w) == w
    let wts_f32: Vec<f32> = wts_f16.iter().map(|&h| f16_to_f64(h) as f32).collect();
    let got32 = prefill_path_f32(&ex, &st, &mut s, &inp.xh, &pick, &wts_f32, &inp.shared, rows, rows);
    let got16 = prefill_path(&ex, &st, &mut s, &inp.xh, &pick, &wts_f16, &inp.shared, rows, rows);
    assert_same("exact fp16 weights", &got32, &got16, &d, &pick);
}

/// The f32 variant changes the bits that the fp16 rounding changed: a single constant weight 0.1 f32 (0.1 rounds to a
/// different fp16 value) must produce a different `out` on every row.
#[test]
fn qwen_f32_vs_fp16_rounded_wts_differ() {
    gpu!();
    let st = Stream::new_nonblocking().unwrap();
    let d = qwen_dims();
    let ex = synth_layer(d, 2, 4, Codebook::Mul1, 830);
    let rows = 129;
    let mut s = PrefillScratch::new(&d, rows).unwrap();
    let mut r = Rng(831);
    let inp = inputs(&mut r, rows, d.hidden, 1.5);
    let (pick, _) = make_picks(&mut r, Style::Hot, rows, d.experts, d.topk);
    let w = 0.1f32; // fp16(0.1) != 0.1
    assert_ne!(f16_to_f64(f64_to_f16(w as f64)) as f32, w, "fp16(0.1) must differ from 0.1");
    let wts_f32 = vec![w; rows * d.topk];
    let wts_f16 = vec![f64_to_f16(w as f64); rows * d.topk];
    let got32 = prefill_path_f32(&ex, &st, &mut s, &inp.xh, &pick, &wts_f32, &inp.shared, rows, rows);
    let got16 = prefill_path(&ex, &st, &mut s, &inp.xh, &pick, &wts_f16, &inp.shared, rows, rows);
    let diff = first_diff(&got16.out, &got32.out).expect("fp16-rounded 0.1 must change the out bits");
    eprintln!("constant 0.1 f32 vs fp16: out first diff at element {diff}");
}

// ---------------------------------------------------------------------------------------------------------------------------------
// mixed per-expert widths (CYBER-FROST's SAGE layout: e.g. gate/up trellises [..,..,48] and [..,..,64] in one layer): the
// constructor groups experts by equal (bits_gu, bits_d, codebook) and the prefill runs one unchanged single-band pipeline per
// band on band-local picks. The tests pin the grouping, the global->local mapping and that a mixed run is bit-identical to
// the single-band runs of its own bands.

fn synth_mixed(dims: MoeDims, bits_a: usize, bits_b: usize, seed: u64) -> (Vec<LoadedGroup>, Vec<LoadedGroup>, Vec<LoadedGroup>) {
    let e = dims.experts;
    let mk = |k, n, bits_of: &dyn Fn(usize) -> usize, tag: u64| -> Vec<LoadedGroup> {
        (0..e).map(|x| synth_group(k, n, bits_of(x), Codebook::Mul1, seed * 1000 + tag * 100 + x as u64)).collect()
    };
    let (d, i) = (dims.hidden, dims.inter);
    let half = e / 2;
    let gate = mk(d, i, &|x| if x < half { bits_a } else { bits_b }, 1);
    let up = mk(d, i, &|x| if x < half { bits_a } else { bits_b }, 2);
    let down = mk(i, d, &|_| 3, 3);
    (gate, up, down)
}

/// The mixed constructor groups by (bits_gu, bits_d, codebook) in order of first occurrence, members ascending, and
/// exposes the bands; the decode path refuses a mixed layer (a clean error, not a wrong result).
#[test]
fn mixed_width_from_exl3_groups_by_band_with_global_mapping() {
    gpu!();
    let dims = MoeDims { hidden: 512, inter: 256, experts: 24, topk: 4, scale: 2.5, limit: 10.0 };
    let (gate, upg, down) = synth_mixed(dims, 4, 8, 900);
    let from_groups = MoeExperts::from_groups(&gate, &upg, &down, dims).unwrap();
    let w = |gs: &[LoadedGroup]| -> Vec<Exl3Weights> { gs.iter().map(|g| Exl3Weights::from_loaded(g).unwrap()).collect() };
    let (wg, wu, wd) = (w(&gate), w(&upg), w(&down));
    let from_exl3 = MoeExperts::from_exl3(&refs_all(&wg), &refs_all(&wu), &refs_all(&wd), dims).unwrap();

    for (name, ex) in [("from_groups", &from_groups), ("from_exl3", &from_exl3)] {
        let bands = ex.bands_desc();
        assert_eq!(bands.len(), 2, "{name}: two width bands");
        assert_eq!(bands[0], (4, 3, Codebook::Mul1, (0..12).collect()), "{name}: band 0 (48-ish width) members");
        assert_eq!(bands[1], (8, 3, Codebook::Mul1, (12..24).collect()), "{name}: band 1 (64-ish width) members");
        // The constructor groups by equal (bits_gu, bits_d, codebook), so every band is fused: one
        // gate/up width (`bands_desc`'s first field). `holds_every_expert` pins the full coverage.
        assert!(ex.holds_every_expert());
    }

    // the decode path refuses a mixed layer rather than compute with one band's width for all experts
    let st = Stream::new_nonblocking().unwrap();
    let mut s = MoeScratch::new(&dims, 16).unwrap();
    let (pick, wts) = make_picks(&mut Rng(901), Style::Uniform, 8, dims.experts, dims.topk);
    s.set_picks(&pick, &wts).unwrap();
    let x = up(&rand_f16(&mut Rng(902), 8 * dims.hidden, 1.0));
    let mut o = DevBuf::zeroed(8 * dims.hidden * 4).unwrap();
    assert!(from_exl3.run(&x, Dtype::F16, 8, None, &mut o, &mut s, false, &st).is_err(), "decode path must refuse a mixed layer");
}

/// A layer with split gate/up widths (gate bits != up bits on one expert) is not representable as a fused band: the
/// constructor refuses it (INVALID) so a caller falls back to CoopMK.
#[test]
fn mixed_width_split_gate_up_is_refused() {
    gpu!();
    let dims = MoeDims { hidden: 512, inter: 256, experts: 8, topk: 4, scale: 2.5, limit: 10.0 };
    let (mut gate, mut upg, down) = synth_mixed(dims, 4, 4, 910);
    // expert 3: gate 4 bits, up 8 bits (split) — not a fused band
    gate[3] = synth_group(dims.hidden, dims.inter, 4, Codebook::Mul1, 910_1001);
    upg[3] = synth_group(dims.hidden, dims.inter, 8, Codebook::Mul1, 910_1002);
    assert!(MoeExperts::from_groups(&gate, &upg, &down, dims).is_err());
    let w = |gs: &[LoadedGroup]| -> Vec<Exl3Weights> { gs.iter().map(|g| Exl3Weights::from_loaded(g).unwrap()).collect() };
    let (wg, wu, wd) = (w(&gate), w(&upg), w(&down));
    assert!(MoeExperts::from_exl3(&refs_all(&wg), &refs_all(&wu), &refs_all(&wd), dims).is_err());
}

/// A mixed layer's run equals its own bands' single-band runs: for every pair, the full run's `y` is bit-identical to the
/// restricted run (only that band's experts routed) of the same band, and the full run over the whole pick set is the same
/// per-pair values the owning band produced.
#[test]
fn mixed_run_prefill_matches_per_band_single_band_paths() {
    gpu!();
    let st = Stream::new_nonblocking().unwrap();
    let dims = MoeDims { hidden: 512, inter: 256, experts: 32, topk: 6, scale: 2.5, limit: 10.0 };
    let (gate, up, down) = synth_mixed(dims, 4, 8, 920);
    let ex = MoeExperts::from_groups(&gate, &up, &down, dims).unwrap();
    assert_eq!(ex.bands_desc().len(), 2);

    let half = dims.experts / 2;
    let w = |gs: &[LoadedGroup]| -> Vec<Exl3Weights> { gs.iter().map(|g| Exl3Weights::from_loaded(g).unwrap()).collect() };
    let (wg, wu, wd) = (w(&gate), w(&up), w(&down));
    let band_a = MoeExperts::from_exl3(
        &refs_all(&wg[..half]),
        &refs_all(&wu[..half]),
        &refs_all(&wd[..half]),
        MoeDims { experts: half, ..dims },
    )
    .unwrap();
    let band_b = MoeExperts::from_exl3(
        &refs_all(&wg[half..]),
        &refs_all(&wu[half..]),
        &refs_all(&wd[half..]),
        MoeDims { experts: dims.experts - half, ..dims },
    )
    .unwrap();

    let rows = 129;
    let mut r = Rng(921);
    let inp = inputs(&mut r, rows, dims.hidden, 1.5);
    let (pick, wts) = make_picks(&mut r, Style::Skewed, rows, dims.experts, dims.topk);

    // restricted picks: band A keeps global ids [0, half), band B keeps [half, E) mapped to local ids
    let pick_a: Vec<i32> = pick.iter().map(|&p| if p >= 0 && (p as usize) < half { p } else { -1 }).collect();
    let pick_b: Vec<i32> =
        pick.iter().map(|&p| if (p as usize) >= half && (p as usize) < dims.experts { p - half as i32 } else { -1 }).collect();

    let run_full =
        prefill_path(&ex, &st, &mut PrefillScratch::new(&dims, rows).unwrap(), &inp.xh, &pick, &wts, &inp.shared, rows, rows);
    let run_a = prefill_path(
        &band_a,
        &st,
        &mut PrefillScratch::new(&MoeDims { experts: half, ..dims }, rows).unwrap(),
        &inp.xh,
        &pick_a,
        &wts,
        &inp.shared,
        rows,
        rows,
    );
    let run_b = prefill_path(
        &band_b,
        &st,
        &mut PrefillScratch::new(&MoeDims { experts: dims.experts - half, ..dims }, rows).unwrap(),
        &inp.xh,
        &pick_b,
        &wts,
        &inp.shared,
        rows,
        rows,
    );

    let d = dims.hidden;
    for p in 0..rows * dims.topk {
        let g = pick[p];
        if g < 0 || (g as usize) >= dims.experts {
            continue; // an unrouted slot: no band owns it, neither run computes it
        }
        let (want, mine) = if (g as usize) < half { (&run_a.y, &run_full.y) } else { (&run_b.y, &run_full.y) };
        let (wa, wf) = (&want[p * d..(p + 1) * d], &mine[p * d..(p + 1) * d]);
        for (j, (a, b)) in wa.iter().zip(wf).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "pair {p} col {j}: mixed run's y differs from the owning band's run");
        }
    }
    eprintln!("mixed prefill: {} pairs bit-identical to their owning band's single-band run", rows * dims.topk);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// parallel plan (gb_moe_pf_plan_parallel) vs the single-block plan_kernel: bit-identical plan arrays and final output.

/// The parallel plan builder must produce exactly the spair / items / nitems of the single-block plan_kernel for the
/// same picks, on realistic dispersed routes and at every P/rows boundary (256-chunk, 16-row tile, class remainder).
#[test]
fn parallel_plan_matches_legacy_bit_for_bit_on_gpu() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = qwen_dims();
    let mut s = PrefillScratch::new(&d, 4096).unwrap();
    s.poison(0xAB, &st).unwrap();
    let mut r = Rng(99);
    for &rows in &[1usize, 17, 127, 128, 129, 130, 255, 256, 257, 1024, 2048, 4096] {
        let p = rows * d.topk;
        let cap = item_cap(p, d.experts);
        for style in [Style::Uniform, Style::Skewed, Style::Hot, Style::Sparse] {
            let pick = make_picks(&mut r, style, rows, d.experts, d.topk).0;
            s.use_parallel_plan(false);
            let legacy = s.plan(&pick, rows, d.experts as i32, &st).unwrap();
            s.use_parallel_plan(true);
            let parallel = s.plan(&pick, rows, d.experts as i32, &st).unwrap();
            assert_eq!(legacy.0, parallel.0, "spair differs at E512/topk10 {rows} rows {style:?}");
            assert_eq!(legacy.1, parallel.1, "items differs at E512/topk10 {rows} rows {style:?}");
            assert_eq!(legacy.2, parallel.2, "nitems differs at E512/topk10 {rows} rows {style:?}");
            assert!(cap >= item_cap(p, d.experts));
        }
    }
}

/// Repeating the same route under the parallel plan is deterministic, and it does not change the final model output
/// (`out` bit-identical to the legacy plan's). Both arms run the SAME route (`s.set_picks` feeds it to the scratch
/// before the legacy run; the parallel arm re-uploads the same picks with `plan`), so this compares like with like.
#[test]
fn parallel_plan_is_deterministic_and_out_matches_legacy_on_gpu() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = qwen_dims();
    let ex = synth_layer(d, 4, 2, Codebook::Mul1, 500);
    let mut s = PrefillScratch::new(&d, 2048).unwrap();
    s.poison(0xAB, &st).unwrap();
    let mut r = Rng(1234);
    for &rows in &[128usize, 129, 1024, 2048] {
        let inp = inputs(&mut r, rows, d.hidden, 2.0);
        let (pick, wts) = make_picks(&mut r, Style::Hot, rows, d.experts, d.topk);
        let sh = up(&inp.shared);
        let x = up(&inp.xh);
        let run = |s: &mut PrefillScratch| {
            let mut o = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
            ex.run_prefill(&x, Dtype::F16, rows, Some(&sh), &mut o, s, &st).unwrap();
            sync_dl::<f32>(&o, &st)
        };
        s.use_parallel_plan(false);
        s.set_picks(&pick, &wts).unwrap();
        let legacy_out = run(&mut s);
        s.use_parallel_plan(true);
        let p1 = s.plan(&pick, rows, d.experts as i32, &st).unwrap();
        let p2 = s.plan(&pick, rows, d.experts as i32, &st).unwrap();
        assert_eq!(p1.0, p2.0, "parallel spair not deterministic at {rows} rows");
        assert_eq!(p1.1, p2.1, "parallel items not deterministic at {rows} rows");
        let parallel_out = run(&mut s);
        assert_eq!(legacy_out, parallel_out, "final out differs between plans at {rows} rows");
    }
}

/// The parallel plan builder must produce exactly the spair / items / nitems of the single-block plan_kernel, and the
/// full pipeline fed the same route must leave every CONSUMED byte identical. This guard runs both plan builders through
/// the full prefill pipeline on the same route at the 16-row-tile and 256-pair-chunk boundaries (127/128/129/130 and
/// 255/256/257), poisoning the scratch before EACH arm so both start from identical 0xAB scratch. The contract compared:
///   * plan arrays — full buffers (spair / items / nitems): both planners write bit-identical plans (the item tail is
///     explicitly zeroed by both, so this is scratch-independent and deterministic);
///   * the intermediates xs_g / xs_u / xd — every element on a CONSUMED sorted row, i.e. a row with spair[r] >= 0
///     (the rotation writes exactly those rows and the grouped GEMMs consume them; sorted rows with spair[r] == -1 are
///     the padding of an expert's last tile — they are read by the mma instruction but contribute to no output row, so
///     their bytes are contractually undefined scratch, not plan output);
///   * y_pairs and final out — full bit-identical (the routed arithmetic's identity).
/// A canary re-runs the parallel arm with the undefined (padding) lanes pinned to a second poison 0xCE and asserts
/// y_pairs / out are still bit-identical: an out-of-contract read of those lanes would change the output and fail here.
/// The full-buffer intermediate equality of earlier revisions was not a valid contract: the scratch lanes beyond the
/// consumed rows hold whatever the previous pipeline left (stale state or poison), and the legacy single-block plan is
/// not required to clear them (proven: the two arms produced bit-identical y_pairs and out while their padding lanes
/// differed byte-for-byte — this canary makes that a permanent, explicit guard).
#[test]
fn parallel_plan_out_identity_at_boundary_rows_on_gpu() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = qwen_dims();
    let ex = synth_layer(d, 4, 2, Codebook::Mul1, 500);
    let mut s = PrefillScratch::new(&d, 2048).unwrap();
    s.poison(0xAB, &st).unwrap();
    let mut r = Rng(1234);
    for &rows in &[127usize, 128, 129, 130, 255, 256, 257] {
        for style in [Style::Hot, Style::Skewed] {
            let inp = inputs(&mut r, rows, d.hidden, 2.0);
            let (pick, wts) = make_picks(&mut r, style, rows, d.experts, d.topk);
            let sh = up(&inp.shared);
            let x = up(&inp.xh);
            let run = |s: &mut PrefillScratch| {
                let mut o = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
                ex.run_prefill(&x, Dtype::F16, rows, Some(&sh), &mut o, s, &st).unwrap();
                sync_dl::<f32>(&o, &st)
            };
            let e = d.experts as i32;
            // legacy arm: the route planned and run on freshly poisoned scratch
            s.poison(0xAB, &st).unwrap();
            s.use_parallel_plan(false);
            s.set_picks(&pick, &wts).unwrap();
            let legacy_plan = s.plan(&pick, rows, e, &st).unwrap();
            let legacy_out = run(&mut s);
            let legacy_y: Vec<f32> = sync_dl(&s.y_pairs, &st);
            let legacy_xs_g: Vec<u16> = s.dbg_xs_g().unwrap();
            let legacy_xs_u: Vec<u16> = s.dbg_xs_u().unwrap();
            let legacy_xd: Vec<u16> = s.dbg_xd().unwrap();
            // poison every scratch buffer again, identically: both arms start from the same 0xAB scratch
            s.poison(0xAB, &st).unwrap();
            // parallel arm: same route, same scratch, re-poisoned between the arms
            s.use_parallel_plan(true);
            s.set_picks(&pick, &wts).unwrap();
            let p1 = s.plan(&pick, rows, e, &st).unwrap();
            let p2 = s.plan(&pick, rows, e, &st).unwrap();
            assert_eq!(p1.0, p2.0, "parallel spair not deterministic at {rows} rows {style:?}");
            assert_eq!(p1.1, p2.1, "parallel items not deterministic at {rows} rows {style:?}");
            assert_eq!(legacy_plan.0, p1.0, "spair differs at {rows} rows {style:?}");
            assert_eq!(legacy_plan.1, p1.1, "items differs at {rows} rows {style:?}");
            assert_eq!(legacy_plan.2, p1.2, "nitems differs at {rows} rows {style:?}");
            let parallel_out = run(&mut s);
            let parallel_y: Vec<f32> = sync_dl(&s.y_pairs, &st);
            let parallel_xs_g: Vec<u16> = s.dbg_xs_g().unwrap();
            let parallel_xs_u: Vec<u16> = s.dbg_xs_u().unwrap();
            let parallel_xd: Vec<u16> = s.dbg_xd().unwrap();
            // consumed-lane masks: an element is contract-carrying iff its sorted row is real (spair[r] >= 0,
            // r < nitems[3]); rows beyond the plan's sorted rows are scratch, never consumed
            let sorted_rows = legacy_plan.2[3] as usize;
            let mask = |k: usize, buf_len: usize| -> Vec<bool> { frag_consumed_mask(&legacy_plan.0, sorted_rows, k, buf_len) };
            let (m_g, m_u, m_d) =
                (mask(d.hidden, legacy_xs_g.len()), mask(d.hidden, legacy_xs_u.len()), mask(d.inter, legacy_xd.len()));
            assert_consumed_eq(&m_g, &legacy_xs_g, &parallel_xs_g, d.hidden, &format!("xs_g differs at {rows} rows {style:?}"));
            assert_consumed_eq(&m_u, &legacy_xs_u, &parallel_xs_u, d.hidden, &format!("xs_u differs at {rows} rows {style:?}"));
            assert_consumed_eq(&m_d, &legacy_xd, &parallel_xd, d.inter, &format!("xd differs at {rows} rows {style:?}"));
            assert_bit_eq_f32(&format!("y_pairs differs at {rows} rows {style:?}"), &legacy_y, &parallel_y);
            assert_bit_eq_f32(&format!("final out differs at {rows} rows {style:?}"), &legacy_out, &parallel_out);
            // canary: pin every UNDEFINED lane (padding rows / beyond nitems[3]) of the intermediates to a second
            // poison and re-run the parallel arm; y_pairs and out must stay bit-identical, proving the undefined
            // lanes are never read for output (an out-of-contract read would change them).
            s.poison(0xAB, &st).unwrap();
            let mut pin = vec![0xABABu16; m_g.len()];
            for (i, &consumed) in m_g.iter().enumerate() {
                if !consumed {
                    pin[i] = 0xCECE;
                }
            }
            s.dbg_set_xs_g(&pin).unwrap();
            let mut pin = vec![0xABABu16; m_u.len()];
            for (i, &consumed) in m_u.iter().enumerate() {
                if !consumed {
                    pin[i] = 0xCECE;
                }
            }
            s.dbg_set_xs_u(&pin).unwrap();
            let mut pin = vec![0xABABu16; m_d.len()];
            for (i, &consumed) in m_d.iter().enumerate() {
                if !consumed {
                    pin[i] = 0xCECE;
                }
            }
            s.dbg_set_xd(&pin).unwrap();
            s.set_picks(&pick, &wts).unwrap();
            let canary_out = run(&mut s);
            let canary_y: Vec<f32> = sync_dl(&s.y_pairs, &st);
            assert_bit_eq_f32(
                &format!("canary y_pairs changed by undefined-lane pin at {rows} rows {style:?}"),
                &parallel_y,
                &canary_y,
            );
            assert_bit_eq_f32(
                &format!("canary final out changed by undefined-lane pin at {rows} rows {style:?}"),
                &parallel_out,
                &canary_out,
            );
        }
    }
}

/// Which u16 elements of an A-fragment intermediate (`xs_g`/`xs_u`/`xd`, column count `k`) belong to a real sorted row
/// (spair[r] >= 0, r < sorted_rows): exactly the rows the rotation kernel writes and the grouped GEMMs consume. The
/// fragment layout is `store_frag4`'s (csrc/moe_prefill.cu): sorted row s (tile mt = s>>4, lane g = s&7, row half
/// rh = (s>>3)&1) and column col (sub-tile kt = col>>4, in-tile column kk = col&15) live at u32 word
///   ((mt*kts + kt)*32 + g*4 + ((kk&7)>>1))*4 + 2*(kk>>3) + rh   (kts = k/16, kk even pairs share a word).
fn frag_consumed_mask(spair: &[i32], sorted_rows: usize, k: usize, buf_len: usize) -> Vec<bool> {
    let mut m = vec![false; buf_len];
    let kts = k >> 4;
    for (row, &p) in spair.iter().take(sorted_rows).enumerate() {
        if p < 0 {
            continue;
        }
        let mt = row >> 4;
        let g = row & 7;
        let rh = (row >> 3) & 1;
        for col in 0..k {
            let kk = col & 15;
            let word = ((mt * kts + (col >> 4)) * 32 + g * 4 + ((kk & 7) >> 1)) * 4 + 2 * (kk >> 3) + rh;
            if word * 2 + 1 < buf_len {
                m[word * 2] = true;
                m[word * 2 + 1] = true;
            }
        }
    }
    m
}

/// Inverse of the `store_frag4` layout: u16 index -> (sorted row, column).
fn frag_inv(o: usize, k: usize) -> (usize, usize) {
    let word = o / 2;
    let half = o % 2;
    let b = word & 3;
    let a = word >> 2;
    let rh = b & 1;
    let kk = (b >> 1) * 8 + (a & 3) * 2 + half;
    let g = (a & 31) >> 2;
    let kts = k >> 4;
    let mtkt = a >> 5;
    (mtkt / kts * 16 + g + 8 * rh, (mtkt % kts) * 16 + kk)
}

fn assert_consumed_eq(mask: &[bool], a: &[u16], b: &[u16], k: usize, msg: &str) {
    assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
    assert_eq!(a.len(), mask.len(), "{msg}: mask length mismatch");
    let mut shown = 0;
    for i in 0..a.len() {
        if mask[i] && a[i] != b[i] {
            let (row, col) = frag_inv(i, k);
            if shown < 8 {
                eprintln!("{msg}: consumed element {i} (row {row}, col {col}) legacy={} parallel={}", a[i], b[i]);
            }
            shown += 1;
        }
    }
    assert_eq!(shown, 0, "{msg}: {shown} consumed elements differ");
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Flash-Next dims (hidden 2560, expert width 640, 512 experts, top-10, bf16 input, f32 router weights, gate/up 3 bits, down 4 bits: the
// production layer), with a routing whose expert popularity is log-normal (a few hot experts, a long tail), drawn per row as the Gumbel top-k.

/// `rows * topk` picks in slot order (best score first) and their softmax f32 weights.
fn qwen_route(r: &mut Rng, rows: usize, e: usize, topk: usize, sigma: f64) -> (Vec<i32>, Vec<f32>) {
    let pop: Vec<f64> = (0..e).map(|_| sigma * normal(r)).collect();
    let (mut pick, mut wts) = (Vec::with_capacity(rows * topk), Vec::with_capacity(rows * topk));
    for _ in 0..rows {
        let mut sc: Vec<(f64, usize)> = (0..e)
            .map(|x| {
                let u = ((r.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
                (pop[x] - (-u.ln()).ln(), x)
            })
            .collect();
        sc.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let top = &sc[..topk];
        let z: f64 = top.iter().map(|t| (t.0 - top[0].0).exp()).sum();
        for t in top {
            pick.push(t.1 as i32);
            wts.push(((t.0 - top[0].0).exp() / z) as f32);
        }
    }
    (pick, wts)
}

fn qwen_x_bf16(r: &mut Rng, n: usize) -> Vec<u16> {
    (0..n).map(|_| gb_cuda::dsv4_attn::bf16_bits(normal(r) as f32)).collect()
}

/// The Flash-Next layer (hidden 2560, expert width 640, 512 experts, top-10: the production K splits and item grouping) equals the decode path bit for
/// bit, for the widths the pack has (gate/up 3 bits, down 4 bits) and for down 5 bits (a width whose shared memory does not fit the one-launch form for
/// every tile count: the class launches run, on the same plan), at row counts that give experts 1 .. 4 tiles, whole 64-row groups and every
/// remainder.
#[test]
fn qwen_shaped_prefill_equals_the_decode_path() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = qwen_dims();
    let mut checked = 0usize;
    for (bg, bd) in [(3usize, 4usize), (3, 5)] {
        let ex = synth_layer(d, bg, bd, Codebook::Mul1, 700 + bd as u64);
        let mut s = PrefillScratch::new(&d, 800).unwrap();
        s.use_parallel_plan(true);
        s.poison(0xAB, &st).unwrap();
        let mut r = Rng(710 + bd as u64);
        for &rows in &[1usize, 17, 129, 333, 800] {
            let inp = inputs(&mut r, rows, d.hidden, 2.0);
            for style in [Style::Uniform, Style::Skewed, Style::Hot, Style::Holes] {
                let (pick, wts) = make_picks(&mut r, style, rows, d.experts, d.topk);
                let want = decode_reference(&ex, &st, &inp.xh, &pick, &wts, &inp.shared, rows, 128);
                let got = prefill_path(&ex, &st, &mut s, &inp.xh, &pick, &wts, &inp.shared, rows, rows);
                assert_same(&format!("Flash-Next dims {bg}/{bd} bit, {rows} rows, {style:?}"), &got, &want, &d, &pick);
                checked += rows;
            }
        }
    }
    eprintln!("{checked} Flash-Next row results bit-identical to the decode path");
}

/// The shared expert's finish fused into the combine gives the bits of the combine into fp32 followed by `finish_shared`, on gates of every size
/// (including the ones whose sigmoid saturates to 0 and 1) and on picks with unrouted slots.
#[test]
fn qwen_fused_combine_finish_equals_combine_then_finish() {
    gpu!();
    let st = Stream::new_nonblocking().unwrap();
    let d = qwen_dims();
    let ex = synth_layer(d, 3, 4, Codebook::Mul1, 720);
    let mut s = PrefillScratch::new(&d, 600).unwrap();
    s.use_parallel_plan(true);
    let mut r = Rng(721);
    for (rows, style) in [(1usize, Style::Uniform), (129, Style::Skewed), (300, Style::Holes), (600, Style::Hot)] {
        let inp = inputs(&mut r, rows, d.hidden, 1.5);
        let (pick, _) = make_picks(&mut r, style, rows, d.experts, d.topk);
        let wts = rand_wts_f32(&mut r, rows * d.topk);
        let gate: Vec<u16> = (0..rows)
            .map(|i| {
                let g = match i % 7 {
                    0 => 80.0,
                    1 => -80.0,
                    2 => 0.0,
                    _ => 4.0 * normal(&mut r) as f32,
                };
                gb_cuda::dsv4_attn::bf16_bits(g)
            })
            .collect();
        s.set_picks_f32(&pick, &wts).unwrap();
        let (x, shared, gate_b) = (up(&inp.xh), up(&inp.shared), up(&gate));
        gb_cuda::rt::device_sync().unwrap();
        let mut routed = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
        ex.run_prefill_f32_wts(&x, Dtype::F16, rows, None, &mut routed, &mut s, &st).unwrap();
        let mut want = DevBuf::zeroed(rows * d.hidden * 2).unwrap();
        finish_shared(&routed, &shared, &gate_b, &mut want, rows, d.hidden, &st).unwrap();
        let mut got = DevBuf::zeroed(rows * d.hidden * 2).unwrap();
        ex.run_prefill_f32_wts_finish(&x, Dtype::F16, rows, &shared, &gate_b, &mut got, &mut s, &st).unwrap();
        let (w, g): (Vec<u16>, Vec<u16>) = (sync_dl(&want, &st), sync_dl(&got, &st));
        if let Some(i) = w.iter().zip(&g).position(|(a, b)| a != b) {
            panic!("{rows} rows {style:?}: fused combine + finish differs at row {} column {}: {:#06x} vs {:#06x}", i / d.hidden, i % d.hidden, g[i], w[i]);
        }
    }
}

/// The unified item list is the three classes' lists merged by first tile: every item once, strictly ascending first tiles (an expert's items
/// follow each other), its count in `nitems[4]`, the slots after it zero, from both plan builders.
#[test]
fn the_unified_item_list_is_the_merge_of_the_classes() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = qwen_dims();
    let mut s = PrefillScratch::new(&d, 4096).unwrap();
    s.poison(0xAB, &st).unwrap();
    let mut r = Rng(730);
    let cap = item_cap(4096 * d.topk, d.experts);       // the scratch's slots a class (it was built for 4,096 rows)
    for &rows in &[1usize, 16, 17, 129, 1000, 4096] {
        for style in [Style::Uniform, Style::Skewed, Style::Hot, Style::Sparse, Style::Holes] {
            let pick = make_picks(&mut r, style, rows, d.experts, d.topk).0;
            for parallel in [false, true] {
                s.use_parallel_plan(parallel);
                let (_, items, nitems) = s.plan(&pick, rows, d.experts as i32, &st).unwrap();
                let n = [nitems[0] as usize, nitems[1] as usize, nitems[2] as usize];
                let total = n[0] + n[1] + n[2];
                assert_eq!(nitems[4] as usize, total, "unified count, {rows} rows {style:?} parallel {parallel}");
                let item = |c: usize, i: usize| -> [i32; 4] {
                    let at = (c * cap + i) * 4;
                    [items[at], items[at + 1], items[at + 2], items[at + 3]]
                };
                let mut want: Vec<[i32; 4]> = (0..3).flat_map(|c| (0..n[c]).map(move |i| (c, i))).map(|(c, i)| item(c, i)).collect();
                want.sort_by_key(|it| it[1]);
                for i in 0..cap {
                    let got = item(3, i);
                    if i < total {
                        assert_eq!(got, want[i], "unified item {i}, {rows} rows {style:?} parallel {parallel}");
                        assert!(i == 0 || want[i - 1][1] < got[1], "first tiles ascend");
                    } else {
                        assert_eq!(got, [0; 4], "unified slot {i} past the count is zero");
                    }
                }
            }
        }
    }
}

/// Per-stage milliseconds of one Flash-Next layer of 4,096 rows (L2 flushed before each call) and the hashes of its outputs (`REXL3_BENCH_ROWS`,
/// `REXL3_BENCH_SIGMA`, `REXL3_BENCH_DOWN_BITS`, `REXL3_BENCH_ITERS` change the shape of the run).
#[test]
#[ignore]
fn bench_prefill_experts_qwen() {
    gpu!();
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let rows = env("REXL3_BENCH_ROWS", 4096);
    let down_bits = env("REXL3_BENCH_DOWN_BITS", 4);
    let iters = env("REXL3_BENCH_ITERS", 5);
    let sigma: f64 = std::env::var("REXL3_BENCH_SIGMA").ok().and_then(|v| v.parse().ok()).unwrap_or(0.8);
    let st = Stream::new().unwrap();
    let d = qwen_dims();
    let ex = synth_layer(d, 3, down_bits, Codebook::Mul1, 11);
    let flush = DevBuf::zeroed(256 << 20).unwrap();
    let mut flush_out = DevBuf::zeroed(16).unwrap();
    let mut r = Rng(5);
    let mut s = PrefillScratch::new(&d, rows).unwrap();
    s.use_parallel_plan(true);
    let x = up(&qwen_x_bf16(&mut r, rows * d.hidden));
    let (pick, wts) = qwen_route(&mut r, rows, d.experts, d.topk, sigma);
    s.set_picks_f32(&pick, &wts).unwrap();
    let t = EventTimer::new(8).unwrap();
    let names = ["plan", "rot_in", "gate_up", "down", "combine"];
    let mut stage = [0f64; 5];
    let mut total = 0f64;
    // the production form: the shared expert's finish fused into the combine, bf16 out
    let gate = up(&qwen_x_bf16(&mut r, rows));
    let shared: Vec<f32> = (0..rows * d.hidden).map(|_| normal(&mut r) as f32).collect();
    let shared = up(&shared);
    let mut out = DevBuf::zeroed(rows * d.hidden * 2).unwrap();
    for it in 0..iters + 1 {
        gb_cuda::moe::read_probe(&flush, 256 << 20, &mut flush_out, &st).unwrap();
        t.record(0, &st).unwrap();
        let mut n = 0;
        ex.run_prefill_f32_wts_finish_marked(&x, Dtype::BF16, rows, &shared, &gate, &mut out, &mut s, &st, &mut |_| {
            n += 1;
            t.record(n, &st)
        })
        .unwrap();
        if it > 0 {
            for i in 0..5 {
                stage[i] += t.ms(i, i + 1).unwrap() as f64 / iters as f64;
            }
            total += t.ms(0, 5).unwrap() as f64 / iters as f64;
        }
    }
    st.sync().unwrap();
    // the two-step it replaces (combine into fp32, then the finish): the bits must be the same, and the time of the tail
    let mut out_f32 = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
    let mut out_two = DevBuf::zeroed(rows * d.hidden * 2).unwrap();
    let mut two_ms = 0f64;
    for it in 0..iters + 1 {
        ex.run_prefill_f32_wts(&x, Dtype::BF16, rows, None, &mut out_f32, &mut s, &st).unwrap();
        gb_cuda::moe::read_probe(&flush, 256 << 20, &mut flush_out, &st).unwrap();
        t.record(6, &st).unwrap();
        gb_cuda::moe_prefill::finish_shared(&out_f32, &shared, &gate, &mut out_two, rows, d.hidden, &st).unwrap();
        t.record(7, &st).unwrap();
        if it > 0 {
            two_ms += t.ms(6, 7).unwrap() as f64 / iters as f64;
        }
    }
    let (a, b): (Vec<u16>, Vec<u16>) = (sync_dl(&out, &st), sync_dl(&out_two, &st));
    assert!(a == b, "the fused combine + finish differs from the two steps");
    println!("finish_shared alone {two_ms:.3} ms (the fused combine takes its place; fused output == two-step output)");
    let ic = s.item_counts().unwrap();
    let mut cnt = vec![0usize; d.experts];
    for &p in &pick {
        cnt[p as usize] += 1;
    }
    let (mx, used) = (cnt.iter().max().unwrap(), cnt.iter().filter(|&&c| c > 0).count());
    println!(
        "Flash-Next layer R={rows} sigma {sigma} down {down_bits} bit: {used} experts, max {mx} pairs, items 1/2/3-4 tiles {}/{}/{}: {total:.3} ms | {}",
        ic[0],
        ic[1],
        ic[2],
        names.iter().zip(stage).map(|(n, m)| format!("{n} {m:.3}")).collect::<Vec<_>>().join(", ")
    );
    // the DRAM floors of the layer's streams: the 420 MB the rotation writes (a memset of two input-fragment buffers) and the 420 MB the combine reads
    let mut big = DevBuf::zeroed(420 << 20).unwrap();
    let (mut w_ms, mut r_ms) = (0f64, 0f64);
    for it in 0..iters + 1 {
        gb_cuda::moe::read_probe(&flush, 256 << 20, &mut flush_out, &st).unwrap();
        t.record(0, &st).unwrap();
        big.memset_async(1, &st).unwrap();
        t.record(1, &st).unwrap();
        gb_cuda::moe::read_probe(&big, 420 << 20, &mut flush_out, &st).unwrap();
        t.record(2, &st).unwrap();
        if it > 0 {
            w_ms += t.ms(0, 1).unwrap() as f64 / iters as f64;
            r_ms += t.ms(1, 2).unwrap() as f64 / iters as f64;
        }
    }
    println!("420 MB: memset {w_ms:.3} ms ({:.0} GB/s), read {r_ms:.3} ms ({:.0} GB/s)", 0.4404 / w_ms * 1e3, 0.4404 / r_ms * 1e3);
    let o: Vec<u16> = sync_dl(&out, &st);
    let y: Vec<f32> = sync_dl(&s.y_pairs, &st);
    println!(
        "hash out {:016x} y {:016x}",
        gb_exl3::fnv1a(o.iter().flat_map(|x| x.to_le_bytes())),
        hash_f32(&y[..rows * d.topk * d.hidden])
    );
}

/// How much of a repeated stream the L2 keeps: memset then read of S MB, the bandwidth of each as S grows (a stream that fits stays out of DRAM).
#[test]
#[ignore]
fn bench_l2_capacity() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut flush_out = DevBuf::zeroed(16).unwrap();
    let t = EventTimer::new(4).unwrap();
    for mb in [2usize, 4, 8, 12, 16, 20, 24, 32, 48, 96, 400] {
        let mut buf = DevBuf::zeroed(mb << 20).unwrap();
        let reps = (2048 / mb).max(4);
        // write then read the same S MB, `reps` times over
        buf.memset_async(1, &st).unwrap();
        t.record(0, &st).unwrap();
        for _ in 0..reps {
            buf.memset_async(2, &st).unwrap();
        }
        t.record(1, &st).unwrap();
        for _ in 0..reps {
            gb_cuda::moe::read_probe(&buf, mb << 20, &mut flush_out, &st).unwrap();
        }
        t.record(2, &st).unwrap();
        let (w, r) = (t.ms(0, 1).unwrap() as f64 / reps as f64, t.ms(1, 2).unwrap() as f64 / reps as f64);
        println!("{mb:4} MB: memset {:6.0} GB/s, read {:6.0} GB/s", mb as f64 / 1024.0 * 1.0737 / w * 1e3, mb as f64 / 1024.0 * 1.0737 / r * 1e3);
    }
}
