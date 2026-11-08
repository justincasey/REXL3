//! Lane P step 3 (GLM-5.3-Flash, contract P2): the routed experts of a prompt chunk on tensor cores (`gb_cuda::glm_p2_moe`,
//! `csrc/glm_p2_moe.cu`), on SYNTHETIC Mul1 EXL3 experts of GLM's shapes (hidden 4,096, expert width 2,048, top-8; 2 and 3 bits as the
//! 2.25 bpw pack's layers, 4 as its NextN layer, 4 and 5 as sc_4.10's, 4, 5 and 6 as RED-SNOW 4.91bpw's).
//!
//! Gates (not ignored), for both K orders (`Order::Chain`, `Order::Fold128`):
//! * `p2_experts_are_as_close_to_fp64_as_the_decode_path`: every pair's y and every row's out against an f64 reference of the exact
//!   dequantized experts; P2's error beside the decode path's (P1's large-row kernel, the decode path's bits). P2 within 1.5x of P1's
//!   rms and under 1e-3.
//! * `p2_pairs_are_bit_identical_in_any_call`: 288 experts, five routing styles (uniform, skewed, one hot set, a quarter unrouted, many
//!   single-pair experts): run to run with a poisoned scratch, calls of 1, 17, 129 and 333 rows, a row permutation: every computed pair's
//!   y and every row's out bit for bit.
//! * `p2_six_bit_pairs_are_bit_identical_in_any_call`: the 6-bit instantiation (6/4 and 6/6) run to run, in calls of 17 and 129 rows and
//!   in the block shapes it fits, bit for bit.
//! * `the_two_group_combine_is_the_tp_split`: the combine of one node (`Both`) equals each rank's own group (`Own(0)`, `Own(1)`) followed by
//!   `Parts`, bit for bit, and equals the host's fmaf chains of the two groups (the contract's order, operation for operation).
//! * `p2_per_subbank_equals_p2_single_bank` and `p2_mixed_width_subbanks_equal_the_uniform_banks_they_are_made_of` (mixed-width layers,
//!   `glm_p2_moe::run_subbanks`): a layer split into artificial subbanks of one width runs P2 bit for bit as the single bank does, at 1,
//!   16, 129 and 300 rows and in calls of other widths; a layer whose subbanks have different widths gives every pair the y of a
//!   uniform bank of that pair's widths.
//! * `p2_split_gate_up_subbank_is_the_decode_pairs_and_two_ranks_equal_one_node`: a layer with a split gate/up subbank (gate 2 bits, up 3)
//!   beside fused ones: the split pairs' y is the decode pairs stage's, the fused pairs' the uniform P2 bank's, in calls of any width and
//!   run to run; two ranks of the range ownership (`pairs_subbanks_rank`, `Own(r)`, `Parts`) equal one node bit for bit.
//!
//! Timing (ignored, `bench` in the name): ms a layer and TFLOP/s of P1's large-row path against P2 at 512 to 4,096 rows, 288 experts,
//! uniform routing, a 256 MiB memset before each call, median of 5; the stages of P2 separately.
//!
//!     cargo test -p gb-cuda --release --test glm_p2_moe -- --test-threads=1 --nocapture
//!     cargo test -p gb-cuda --release --test glm_p2_moe -- --ignored --test-threads=1 --nocapture bench
#![cfg(not(gb_no_cuda))]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
mod common;
mod exl3_ref;

use common::exl3_synth::synth_group;
use common::Rng;
use exl3_ref::{err, linear_ref, weights};
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::glm_p2_moe::{self, CombineMode, Order};
use gb_cuda::moe::{MoeDims, MoeExperts, MoeScratch};
use gb_cuda::moe_owned::{ExpertOwnership, OwnScratch};
use gb_cuda::moe_prefill::PrefillScratch;
use gb_cuda::moe_subbanks::MoeSubbanks;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::checkpoint::LoadedGroup;
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_exl3::Codebook;

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

const ORDERS: [Order; 2] = [Order::Chain, Order::Fold128];

fn glm_dims(experts: usize) -> MoeDims {
    MoeDims { hidden: 4096, inter: 2048, experts, topk: 8, scale: 2.5, limit: 10.0 }
}

struct Layer {
    ex: MoeExperts,
    groups: Option<[Vec<LoadedGroup>; 3]>,
}

fn synth_layer(dims: MoeDims, bits_gu: usize, bits_d: usize, seed: u64, keep: bool) -> Layer {
    let e = dims.experts;
    let mk = |k, n, bits, tag: u64| -> Vec<LoadedGroup> {
        (0..e).map(|x| synth_group(k, n, bits, Codebook::Mul1, seed * 1000 + tag * 100 + x as u64)).collect()
    };
    let gate = mk(dims.hidden, dims.inter, bits_gu, 1);
    let up = mk(dims.hidden, dims.inter, bits_gu, 2);
    let down = mk(dims.inter, dims.hidden, bits_d, 3);
    let ex = MoeExperts::from_groups(&gate, &up, &down, dims).unwrap();
    Layer { ex, groups: keep.then_some([gate, up, down]) }
}

fn below(r: &mut Rng, n: usize) -> usize {
    (r.next_u64() % n as u64) as usize
}

fn normal(r: &mut Rng) -> f64 {
    let (a, b) = ((r.next_u64() >> 11) as f64 / (1u64 << 53) as f64, (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64);
    (-2.0 * a.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
}

#[derive(Clone, Copy, Debug)]
enum Style {
    Uniform,
    Skewed,
    Hot,
    Holes,
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
                    let u = (r.next_u64() >> 40) as f64 / 16777216.0 + 1e-6;
                    let c = ((1.0 / u) as usize - 1).min(e - 1);
                    if !chosen.contains(&c) {
                        chosen.push(c);
                    }
                }
                chosen
            }
            Style::Hot => (0..topk).map(|k| (k * 37) % e).collect(),
            Style::Sparse => {
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
        };
        for &id in &ids {
            let hole = matches!(style, Style::Holes) && below(r, 4) == 0;
            pick.push(if hole { -1 + (below(r, 2) as i32) * (e as i32 + 1) } else { id as i32 });
            wts.push(f64_to_f16(0.05 + 0.6 * (r.next_u64() >> 40) as f64 / 16777216.0));
        }
    }
    (pick, wts)
}

/// fp16 rows at the scale of a normed hidden row (a few percent of entries four times larger).
fn activations(r: &mut Rng, rows: usize, k: usize, scale: f64) -> Vec<u16> {
    (0..rows * k)
        .map(|_| {
            let v = r.uniform(-1.0, 1.0) as f64;
            let v = if r.next_u64() % 32 == 0 { 4.0 * v } else { v };
            f64_to_f16(v * scale)
        })
        .collect()
}

fn up<T: Copy>(v: &[T]) -> DevBuf {
    DevBuf::from_slice(v).unwrap()
}

struct Run {
    out: Vec<f32>,
    y: Vec<f32>,
}

/// `path`: None P1's large-row path (`run_prefill`: the decode path's bits), Some(order) P2. Calls of `chunk` rows (the last shorter).
fn run(
    ex: &MoeExperts,
    st: &Stream,
    s: &mut PrefillScratch,
    xh: &[u16],
    pick: &[i32],
    wts: &[u16],
    shared: &[f32],
    rows: usize,
    chunk: usize,
    path: Option<Order>,
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
        match path {
            None => ex.run_prefill(&x, Dtype::F16, n, Some(&sh), &mut o, s, st).unwrap(),
            Some(ord) => glm_p2_moe::run_with(ex, &x, Dtype::F16, n, Some(&sh), &mut o, s, ord, st, &mut |_| Ok(())).unwrap(),
        }
        st.sync().unwrap();
        out.extend(o.to_vec::<f32>().unwrap());
        // only this call's pairs (the scratch holds `s.max_rows` rows of them)
        let mut yy = vec![0f32; n * d.topk * d.hidden];
        s.y_pairs.download_range(0, &mut yy).unwrap();
        y.extend(yy);
        at += n;
    }
    Run { out, y }
}

fn valid(e: i32, d: &MoeDims) -> bool {
    e >= 0 && (e as usize) < d.experts
}

/// Every computed pair's y and every row's out equal, bit for bit.
fn assert_same(what: &str, got: &Run, want: &Run, d: &MoeDims, pick: &[i32]) {
    assert_eq!(got.out.len(), want.out.len(), "{what}: out length");
    for (p, &e) in pick.iter().enumerate() {
        if !valid(e, d) {
            continue;
        }
        let r = p * d.hidden..(p + 1) * d.hidden;
        if let Some(i) = got.y[r.clone()].iter().zip(&want.y[r.clone()]).position(|(a, b)| a.to_bits() != b.to_bits()) {
            panic!(
                "{what}: pair {p} (row {}, slot {}) column {i} of y differs: {} vs {}",
                p / d.topk,
                p % d.topk,
                got.y[r.start + i],
                want.y[r.start + i]
            );
        }
    }
    if let Some(i) = got.out.iter().zip(&want.out).position(|(a, b)| a.to_bits() != b.to_bits()) {
        panic!("{what}: row {} column {} of out differs: {} vs {}", i / d.hidden, i % d.hidden, got.out[i], want.out[i]);
    }
}

#[test]
fn p2_experts_are_as_close_to_fp64_as_the_decode_path() {
    gpu!();
    let st = Stream::new().unwrap();
    let rows = 48;
    for (li, &(bits_gu, bits_d)) in [(2usize, 2usize), (3, 3), (4, 4), (5, 5), (6, 6), (6, 4)].iter().enumerate() {
        let d = glm_dims(12);
        let lay = synth_layer(d, bits_gu, bits_d, 40 + li as u64, true);
        let [gate, upg, down] = lay.groups.as_ref().unwrap();
        let ex = &lay.ex;
        assert!(glm_p2_moe::supported(ex), "{bits_gu}/{bits_d} bits");
        let mut r = Rng(7 + li as u64);
        let xh = activations(&mut r, rows, d.hidden, 1.5);
        let (pick, wts) = make_picks(&mut r, Style::Uniform, rows, d.experts, d.topk);
        let shared: Vec<f32> = (0..rows * d.hidden).map(|_| 0.1 * normal(&mut r) as f32).collect();
        let mut s = PrefillScratch::new(&d, rows).unwrap();
        let p1 = run(ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, rows, None);
        let p2: Vec<Run> = ORDERS.iter().map(|&o| run(ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, rows, Some(o))).collect();
        // the f64 reference of every pair and row
        let wg: Vec<Vec<f32>> = gate.iter().map(|g| weights(g, &st)).collect();
        let wu: Vec<Vec<f32>> = upg.iter().map(|g| weights(g, &st)).collect();
        let wd: Vec<Vec<f32>> = down.iter().map(|g| weights(g, &st)).collect();
        let lim = d.limit as f64;
        let (mut yref, mut oref) = (Vec::new(), Vec::new());
        let (mut y1, mut y2): (Vec<f32>, Vec<Vec<f32>>) = (Vec::new(), vec![Vec::new(); 2]);
        for row in 0..rows {
            let x: Vec<f64> = (0..d.hidden).map(|i| f16_to_f64(xh[row * d.hidden + i])).collect();
            let mut o: Vec<f64> = (0..d.hidden).map(|i| shared[row * d.hidden + i] as f64).collect();
            for k in 0..d.topk {
                let p = row * d.topk + k;
                let e = pick[p] as usize;
                let g = linear_ref(&gate[e], &wg[e], &x);
                let u = linear_ref(&upg[e], &wu[e], &x);
                let a: Vec<f64> =
                    g.iter().zip(&u).map(|(&g, &u)| (g / (1.0 + (-g).exp())).min(lim) * u.clamp(-lim, lim)).collect();
                let y = linear_ref(&down[e], &wd[e], &a);
                let w = f16_to_f64(wts[p]);
                for j in 0..d.hidden {
                    o[j] += w * y[j];
                }
                yref.extend_from_slice(&y);
                y1.extend_from_slice(&p1.y[p * d.hidden..(p + 1) * d.hidden]);
                for (i, r2) in p2.iter().enumerate() {
                    y2[i].extend_from_slice(&r2.y[p * d.hidden..(p + 1) * d.hidden]);
                }
            }
            oref.extend(o);
        }
        let (m1, r1, s1, k1) = err(&yref, &y1);
        let (om1, or1, _, _) = err(&oref, &p1.out);
        let mut line = format!(
            "{bits_gu}/{bits_d}-bit experts, {rows} rows x 8: P1 y max {m1:.2e} rms {r1:.2e} mean {s1:+.1e} shrink {k1:+.1e} out rms {or1:.2e} (max {om1:.2e})"
        );
        for (i, &o) in ORDERS.iter().enumerate() {
            let (m2, r2, s2, k2) = err(&yref, &y2[i]);
            let (om2, or2, _, _) = err(&oref, &p2[i].out);
            line += &format!(
                " | {} y max {m2:.2e} rms {r2:.2e} mean {s2:+.1e} shrink {k2:+.1e} out rms {or2:.2e} (max {om2:.2e})",
                o.id()
            );
            assert!(r2 <= 1.5 * r1 + 1e-7 && r2 < 1e-3, "{bits_gu}/{bits_d} {o:?}: P2 y rms {r2:e} against P1's {r1:e}");
            assert!(or2 <= 1.5 * or1 + 1e-7, "{bits_gu}/{bits_d} {o:?}: P2 out rms {or2:e} against P1's {or1:e}");
            assert!(p2[i].y.iter().any(|v| *v != 0.0) && p2[i].out.iter().all(|v| v.is_finite()), "{o:?}: degenerate output");
        }
        eprintln!("{line}");
    }
}

#[test]
fn p2_pairs_are_bit_identical_in_any_call() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = glm_dims(288);
    let lay = synth_layer(d, 2, 2, 3, false);
    let ex = &lay.ex;
    let rows = 700;
    let mut s = PrefillScratch::new(&d, rows).unwrap();
    for (si, &style) in [Style::Uniform, Style::Skewed, Style::Hot, Style::Holes, Style::Sparse].iter().enumerate() {
        let mut r = Rng(100 + si as u64);
        let xh = activations(&mut r, rows, d.hidden, 1.5);
        let (pick, wts) = make_picks(&mut r, style, rows, d.experts, d.topk);
        let shared: Vec<f32> = (0..rows * d.hidden).map(|_| 0.1 * normal(&mut r) as f32).collect();
        for &o in &ORDERS {
            let whole = run(ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, rows, Some(o));
            assert!(whole.out.iter().all(|v| v.is_finite()) && whole.out.iter().any(|v| *v != 0.0), "{style:?}: degenerate");
            s.poison(0xa5, &st).unwrap();
            let again = run(ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, rows, Some(o));
            assert_same(&format!("{style:?} {o:?} run to run, poisoned scratch"), &again, &whole, &d, &pick);
            for chunk in [1usize, 17, 129, 333] {
                if chunk == 1 && !matches!(style, Style::Uniform) {
                    continue; // 700 one-row calls: once is enough
                }
                let got = run(ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, chunk, Some(o));
                assert_same(&format!("{style:?} {o:?} calls of {chunk}"), &got, &whole, &d, &pick);
            }
            // a permutation of the rows: every row and pair keeps its bits
            let mut perm: Vec<usize> = (0..rows).collect();
            for j in (1..rows).rev() {
                perm.swap(j, below(&mut r, j + 1));
            }
            let px: Vec<u16> = perm.iter().flat_map(|&q| xh[q * d.hidden..(q + 1) * d.hidden].to_vec()).collect();
            let pp: Vec<i32> = perm.iter().flat_map(|&q| pick[q * d.topk..(q + 1) * d.topk].to_vec()).collect();
            let pw: Vec<u16> = perm.iter().flat_map(|&q| wts[q * d.topk..(q + 1) * d.topk].to_vec()).collect();
            let ps: Vec<f32> = perm.iter().flat_map(|&q| shared[q * d.hidden..(q + 1) * d.hidden].to_vec()).collect();
            let got = run(ex, &st, &mut s, &px, &pp, &pw, &ps, rows, rows, Some(o));
            for (j, &q) in perm.iter().enumerate() {
                let (a, b) = (&got.out[j * d.hidden..(j + 1) * d.hidden], &whole.out[q * d.hidden..(q + 1) * d.hidden]);
                assert!(
                    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "{style:?} {o:?}: row {q} at position {j} of a permutation"
                );
            }
            // every block shape (which warp computes which output) gives the default's bits
            for (gu, dn) in [(1, 1), (2, 2), (3, 1)] {
                glm_p2_moe::set_shapes(gu, dn);
                let got = run(ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, rows, Some(o));
                glm_p2_moe::set_shapes(0, 0);
                assert_same(&format!("{style:?} {o:?} block shapes gate/up {gu} down {dn}"), &got, &whole, &d, &pick);
            }
            eprintln!(
                "{style:?} {}: bit-identical run to run (poisoned scratch), in calls of 1 / 17 / 129 / 333, permuted, every block shape",
                o.id()
            );
        }
        // P2 moved the bits (it ran): its y differs from P1's somewhere
        let p1 = run(ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, rows, None);
        let p2 = run(ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, rows, Some(glm_p2_moe::DEFAULT_ORDER));
        assert!(p1.out.iter().zip(&p2.out).any(|(a, b)| a.to_bits() != b.to_bits()), "{style:?}: P2 gave P1's bits");
    }
}

/// The 6-bit instantiation (RED-SNOW 4.91bpw's gate/up 6 and down 4, and its uniform 6/6/6 layer): every pair's y and every row's out bit
/// for bit run to run (poisoned scratch), in calls of 17 and 129 rows, and in every block shape the width fits (gate/up 3, 16 warps and 4
/// stages, does not: refused by its guard, the default's bits checked on the shapes that run).
#[test]
fn p2_six_bit_pairs_are_bit_identical_in_any_call() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = glm_dims(288);
    let rows = 400;
    let mut s = PrefillScratch::new(&d, rows).unwrap();
    for (li, &(bg, bd)) in [(6usize, 4usize), (6, 6)].iter().enumerate() {
        let lay = synth_layer(d, bg, bd, 90 + li as u64, false);
        assert!(glm_p2_moe::supported(&lay.ex), "{bg}/{bd} bits");
        for (si, &style) in [Style::Uniform, Style::Skewed, Style::Holes].iter().enumerate() {
            let mut r = Rng(4000 + 10 * li as u64 + si as u64);
            let xh = activations(&mut r, rows, d.hidden, 1.5);
            let (pick, wts) = make_picks(&mut r, style, rows, d.experts, d.topk);
            let shared: Vec<f32> = (0..rows * d.hidden).map(|_| 0.1 * normal(&mut r) as f32).collect();
            let o = Some(glm_p2_moe::DEFAULT_ORDER);
            let whole = run(&lay.ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, rows, o);
            assert!(whole.out.iter().all(|v| v.is_finite()) && whole.out.iter().any(|v| *v != 0.0), "{style:?}: degenerate");
            s.poison(0xa5, &st).unwrap();
            let again = run(&lay.ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, rows, o);
            assert_same(&format!("{bg}/{bd} {style:?} run to run"), &again, &whole, &d, &pick);
            for chunk in [17usize, 129] {
                let got = run(&lay.ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, chunk, o);
                assert_same(&format!("{bg}/{bd} {style:?} calls of {chunk}"), &got, &whole, &d, &pick);
            }
            for (gu, dn) in [(1, 1), (2, 2)] {
                glm_p2_moe::set_shapes(gu, dn);
                let got = run(&lay.ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, rows, o);
                glm_p2_moe::set_shapes(0, 0);
                assert_same(&format!("{bg}/{bd} {style:?} block shapes {gu}/{dn}"), &got, &whole, &d, &pick);
            }
        }
        eprintln!("{bg}/{bd}-bit P2 experts: bit-identical run to run, in calls of 17 / 129, in block shapes 1/1 and 2/2");
    }
}

/// The widest P2 prompt chunk (`moe_prefill::MAX_ROWS`, 8,192 rows) on GLM's expert shapes at RED-SNOW's widths (gate/up 6 bits, down 4):
/// every pair's y and every row's out of one 8,192-row call equal those of calls of 4,096 and 2,731 rows, bit for bit, under uniform
/// routing and with every row on a few hot experts (the longest item lists, the largest item count).
#[test]
fn p2_pairs_of_an_8192_row_call_are_those_of_smaller_calls() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = glm_dims(288);
    let rows = gb_cuda::moe_prefill::MAX_ROWS;
    assert_eq!(rows, 8192);
    let mut s = PrefillScratch::new(&d, rows).unwrap();
    let lay = synth_layer(d, 6, 4, 97, false);
    for (si, &style) in [Style::Uniform, Style::Hot].iter().enumerate() {
        let mut r = Rng(8192 + si as u64);
        let xh = activations(&mut r, rows, d.hidden, 1.5);
        let (pick, wts) = make_picks(&mut r, style, rows, d.experts, d.topk);
        let shared: Vec<f32> = (0..rows * d.hidden).map(|_| 0.1 * normal(&mut r) as f32).collect();
        let o = Some(glm_p2_moe::DEFAULT_ORDER);
        let whole = run(&lay.ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, rows, o);
        assert!(whole.out.iter().all(|v| v.is_finite()) && whole.out.iter().any(|v| *v != 0.0), "{style:?}: degenerate");
        for chunk in [4096usize, 2731] {
            let got = run(&lay.ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, chunk, o);
            assert_same(&format!("6/4 {style:?} 8,192 rows in calls of {chunk}"), &got, &whole, &d, &pick);
        }
        eprintln!("6/4-bit P2 experts, {style:?}: an 8,192-row call == calls of 4,096 / 2,731 rows, bit for bit");
    }
}

#[test]
fn the_plan_is_p1s_plan() {
    gpu!();
    let st = Stream::new().unwrap();
    // small experts: the plan does not read the weights
    let d = MoeDims { hidden: 128, inter: 128, experts: 288, topk: 8, scale: 2.5, limit: 10.0 };
    let lay = synth_layer(d, 2, 2, 11, false);
    let ex = &lay.ex;
    let mut s = PrefillScratch::new(&d, gb_cuda::moe_prefill::MAX_ROWS).unwrap();
    let mut checked = 0;
    for (si, &style) in [Style::Uniform, Style::Skewed, Style::Hot, Style::Holes, Style::Sparse].iter().enumerate() {
        for rows in [1usize, 17, 255, 256, 257, 700, 4096, 6144, 8192] {
            let mut r = Rng(300 + si as u64 * 100 + rows as u64);
            let (pick, wts) = make_picks(&mut r, style, rows, d.experts, d.topk);
            let xh = activations(&mut r, rows, d.hidden, 1.0);
            s.set_picks(&pick, &wts).unwrap();
            s.poison(0x5a, &st).unwrap();
            s.set_picks(&pick, &wts).unwrap();
            let mut o = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
            ex.run_prefill(&up(&xh), Dtype::F16, rows, None, &mut o, &mut s, &st).unwrap();
            let (sp1, it1, n1) = glm_p2_moe::plan_view(&s, &st).unwrap();
            s.poison(0xa5, &st).unwrap();
            s.set_picks(&pick, &wts).unwrap();
            glm_p2_moe::plan_only(&mut s, rows, &st).unwrap();
            let (sp2, it2, n2) = glm_p2_moe::plan_view(&s, &st).unwrap();
            assert_eq!(n1[3], n2[3], "{style:?} rows {rows}: the sorted rows");
            assert_eq!((n2[1], n2[2]), (0, 0), "{style:?} rows {rows}: one list of items");
            let sorted = n1[3] as usize;
            assert_eq!(sp1[..sorted], sp2[..sorted], "{style:?} rows {rows}: the sorted pairs");
            // P1's three classes regrouped per expert in tile order: P2's one list (the same items, in expert order)
            let cap = it1.len() / 12;
            let mut want: Vec<&[i32]> =
                (0..3).flat_map(|c| (0..n1[c] as usize).map(move |k| (c * cap + k) * 4)).map(|i| &it1[i..i + 4]).collect();
            want.sort_by_key(|it| (it[0], it[1]));
            let got: Vec<&[i32]> = (0..n2[0] as usize).map(|k| &it2[k * 4..k * 4 + 4]).collect();
            assert_eq!(want, got, "{style:?} rows {rows}: P1's items in expert order");
            checked += 1;
        }
    }
    eprintln!(
        "the P2 plan is P1's plan (sorted pairs, the items in expert order) on {checked} cases (5 routing styles x 9 row counts to 8,192, 288 experts)"
    );
}

#[test]
fn the_two_group_combine_is_the_tp_split() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = glm_dims(288);
    let lay = synth_layer(d, 2, 2, 5, false);
    let ex = &lay.ex;
    let rows = 300;
    let mut s = PrefillScratch::new(&d, rows).unwrap();
    let mut r = Rng(55);
    let xh = activations(&mut r, rows, d.hidden, 1.5);
    let (pick, wts) = make_picks(&mut r, Style::Holes, rows, d.experts, d.topk);
    let shared: Vec<f32> = (0..rows * d.hidden).map(|_| 0.1 * normal(&mut r) as f32).collect();
    let whole = run(ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, rows, Some(glm_p2_moe::DEFAULT_ORDER));
    // the TP=2 split of the same pairs: rank g's own group, the exchange (a copy), the parts' sum with the shared expert last
    let sh = up(&shared);
    let mut parts = [DevBuf::zeroed(rows * d.hidden * 4).unwrap(), DevBuf::zeroed(rows * d.hidden * 4).unwrap()];
    for (g, p) in parts.iter_mut().enumerate() {
        glm_p2_moe::combine(CombineMode::Own(g), &s.y_pairs, &s.pick, &s.wts, None, None, p, rows, ex, &st).unwrap();
    }
    let mut out = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
    glm_p2_moe::combine(
        CombineMode::Parts,
        &s.y_pairs,
        &s.pick,
        &s.wts,
        Some(&sh),
        Some((&parts[0], &parts[1])),
        &mut out,
        rows,
        ex,
        &st,
    )
    .unwrap();
    st.sync().unwrap();
    let split = out.to_vec::<f32>().unwrap();
    assert!(split.iter().zip(&whole.out).all(|(a, b)| a.to_bits() == b.to_bits()), "the TP split of the combine moved a bit");
    // a row-split rank's sum: Parts on its own rows only (`combine_parts_rows`), the two ranges together == the full call
    let mut rs = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
    for (r0, n) in [(0usize, 137usize), (137, rows - 137)] {
        glm_p2_moe::combine_parts_rows(Some(&sh), (&parts[0], &parts[1]), &mut rs, r0, n, &d, &st).unwrap();
    }
    st.sync().unwrap();
    let rsv = rs.to_vec::<f32>().unwrap();
    assert!(rsv.iter().zip(&whole.out).all(|(a, b)| a.to_bits() == b.to_bits()), "the row-range Parts sum moved a bit");
    // the host's fmaf chains of the two groups, then (p0 + p1) + shared: the contract's order
    let gs = glm_p2_moe::group_split(d.experts);
    assert_eq!(gs, 144);
    let mut host = vec![0f32; rows * d.hidden];
    for row in 0..rows {
        for j in 0..d.hidden {
            let (mut p0, mut p1) = (0f32, 0f32);
            for k in 0..d.topk {
                let p = row * d.topk + k;
                if !valid(pick[p], &d) {
                    continue;
                }
                let w = f16_to_f64(wts[p]) as f32;
                let y = whole.y[p * d.hidden + j];
                if (pick[p] as usize) < gs {
                    p0 = w.mul_add(y, p0);
                } else {
                    p1 = w.mul_add(y, p1);
                }
            }
            host[row * d.hidden + j] = (p0 + p1) + shared[row * d.hidden + j];
        }
    }
    let bad = host.iter().zip(&whole.out).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert_eq!(bad, 0, "{bad} values of the P2 combine differ from the host's two-group chains");
    eprintln!("the two-group combine: Own(0) + Own(1) -> Parts == Both == the host's chains, {rows} rows (a quarter of the slots unrouted)");
}

/// A layer of `d.experts` experts as subbanks, one per id set of `parts` (which together cover every expert once): each bank from the
/// groups of its own experts in `src` (gate, up, down of a whole layer: expert `e` of `src[i]` is global expert `e`).
fn subbanks_of(d: MoeDims, parts: &[(Vec<u32>, &[Vec<LoadedGroup>; 3])]) -> MoeSubbanks {
    let mut banks = Vec::new();
    for (ids, g) in parts {
        let pick = |v: &[LoadedGroup]| -> Vec<LoadedGroup> { ids.iter().map(|&e| v[e as usize].clone()).collect() };
        let o = ExpertOwnership::from_ids(d.experts, ids.clone()).unwrap();
        banks.push(MoeExperts::from_groups_owned(&pick(&g[0]), &pick(&g[1]), &pick(&g[2]), d, o).unwrap());
    }
    MoeSubbanks::new(d, ExpertOwnership::all(d.experts), banks).unwrap()
}

/// [`run`] for a layer of subbanks (always P2; `sd` the decode scratch a split gate/up subbank's windows run through).
fn run_sub(
    sb: &MoeSubbanks,
    st: &Stream,
    s: &mut PrefillScratch,
    os: &mut OwnScratch,
    sd: &mut MoeScratch,
    xh: &[u16],
    pick: &[i32],
    wts: &[u16],
    shared: &[f32],
    rows: usize,
    chunk: usize,
) -> Run {
    let d = sb.dims;
    let (mut out, mut y) = (Vec::new(), Vec::new());
    let mut at = 0;
    while at < rows {
        let n = chunk.min(rows - at);
        s.set_picks(&pick[at * d.topk..(at + n) * d.topk], &wts[at * d.topk..(at + n) * d.topk]).unwrap();
        let x = up(&xh[at * d.hidden..(at + n) * d.hidden]);
        let sh = up(&shared[at * d.hidden..(at + n) * d.hidden]);
        let mut o = DevBuf::zeroed(n * d.hidden * 4).unwrap();
        glm_p2_moe::run_subbanks(sb, &x, Dtype::F16, n, Some(&sh), &mut o, s, os, sd, st).unwrap();
        st.sync().unwrap();
        out.extend(o.to_vec::<f32>().unwrap());
        let mut yy = vec![0f32; n * d.topk * d.hidden];
        s.y_pairs.download_range(0, &mut yy).unwrap();
        y.extend(yy);
        at += n;
    }
    Run { out, y }
}

/// The partitions of the 288 experts the subbank tests use: scattered thirds, a skewed 1 + 287, eight interleaved, and the cut of the
/// combine's two groups (143 / 144) with a third bank of the experts around it.
fn partitions(e: usize) -> Vec<(&'static str, Vec<Vec<u32>>)> {
    let by = |n: usize, f: &dyn Fn(u32) -> usize| -> Vec<Vec<u32>> {
        let mut v = vec![Vec::new(); n];
        for g in 0..e as u32 {
            v[f(g)].push(g);
        }
        v.into_iter().filter(|x| !x.is_empty()).collect()
    };
    vec![
        ("scattered thirds", by(3, &|g| (g as usize * 7 + g as usize / 5) % 3)),
        ("1 + 287", by(2, &|g| (g == 200) as usize)),
        ("eight interleaved", by(8, &|g| g as usize % 8)),
        ("the 143/144 cut", by(3, &|g| if (140..148).contains(&g) { 2 } else { (g as usize >= 144) as usize })),
    ]
}

#[test]
fn p2_per_subbank_equals_p2_single_bank() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = glm_dims(288);
    let lay = synth_layer(d, 2, 2, 21, true);
    let groups = lay.groups.as_ref().unwrap();
    let max = 300;
    let mut s = PrefillScratch::new(&d, max).unwrap();
    let mut os = OwnScratch::new(&d, max).unwrap();
    let mut sd = MoeScratch::new(&d, 128).unwrap();
    for (name, parts) in partitions(d.experts) {
        let sb = subbanks_of(d, &parts.iter().map(|ids| (ids.clone(), groups)).collect::<Vec<_>>());
        assert_eq!(sb.banks().len(), parts.len());
        let mut cases = 0;
        for (si, &style) in [Style::Uniform, Style::Holes, Style::Skewed, Style::Sparse].iter().enumerate() {
            for rows in [1usize, 16, 129, 300] {
                let mut r = Rng(900 + 17 * si as u64 + rows as u64);
                let xh = activations(&mut r, rows, d.hidden, 1.5);
                let (pick, wts) = make_picks(&mut r, style, rows, d.experts, d.topk);
                let shared: Vec<f32> = (0..rows * d.hidden).map(|_| 0.1 * normal(&mut r) as f32).collect();
                let want = run(&lay.ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, rows, Some(glm_p2_moe::DEFAULT_ORDER));
                s.poison(0xa5, &st).unwrap();
                let r0 = glm_p2_moe::runs();
                let got = run_sub(&sb, &st, &mut s, &mut os, &mut sd, &xh, &pick, &wts, &shared, rows, rows);
                // the audit counts one P2 expert block a call, not one a subbank
                assert_eq!(glm_p2_moe::runs() - r0, 1, "{name}: P2 expert blocks counted for one layer call");
                assert_same(&format!("{name} {style:?} {rows} rows, one call"), &got, &want, &d, &pick);
                cases += 1;
                // the same rows in calls of other widths
                for chunk in [1usize, 16, 129] {
                    if chunk >= rows || (chunk == 1 && rows > 16) {
                        continue;
                    }
                    let got = run_sub(&sb, &st, &mut s, &mut os, &mut sd, &xh, &pick, &wts, &shared, rows, chunk);
                    assert_same(&format!("{name} {style:?} {rows} rows in calls of {chunk}"), &got, &want, &d, &pick);
                    cases += 1;
                }
            }
        }
        eprintln!(
            "{name} ({} subbanks): P2 per subbank == P2 single bank, bit for bit ({cases} cases: 4 routing styles x 1/16/129/300 rows x call widths)",
            parts.len()
        );
    }
}

/// Subbanks of DIFFERENT widths (gate/up 2, 3 and 4 bits; down 3, 2 and 4): every valid pair's y is the y of a uniform bank of its
/// expert's widths (the three uniform layers run whole, the pair taken from the one that holds its expert), and the row's out is the one
/// combine over those pairs. Includes unrouted picks (-1, 289).
#[test]
fn p2_mixed_width_subbanks_equal_the_uniform_banks_they_are_made_of() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = glm_dims(288);
    let widths = [(2usize, 3usize), (3, 2), (4, 4)];
    let lays: Vec<Layer> = widths.iter().enumerate().map(|(i, &(g, dn))| synth_layer(d, g, dn, 31 + i as u64, true)).collect();
    let of = |e: usize| (e * 7 + e / 5) % 3;
    let mut parts: Vec<Vec<u32>> = vec![Vec::new(); 3];
    for e in 0..d.experts {
        parts[of(e)].push(e as u32);
    }
    let sb = subbanks_of(d, &(0..3).map(|i| (parts[i].clone(), lays[i].groups.as_ref().unwrap())).collect::<Vec<_>>());
    assert_eq!(sb.banks().len(), 3);
    let keys: Vec<_> = sb.banks().iter().map(|b| b.subbank_key()).collect();
    assert!(keys.windows(2).all(|w| w[0] != w[1]), "three different widths: {keys:?}");
    let max = 300;
    let mut s = PrefillScratch::new(&d, max).unwrap();
    let mut os = OwnScratch::new(&d, max).unwrap();
    let mut sd = MoeScratch::new(&d, 128).unwrap();
    for (si, &style) in [Style::Uniform, Style::Holes, Style::Skewed, Style::Sparse].iter().enumerate() {
        for rows in [1usize, 16, 129, 300] {
            let mut r = Rng(1500 + 17 * si as u64 + rows as u64);
            let xh = activations(&mut r, rows, d.hidden, 1.5);
            let (pick, wts) = make_picks(&mut r, style, rows, d.experts, d.topk);
            let shared: Vec<f32> = (0..rows * d.hidden).map(|_| 0.1 * normal(&mut r) as f32).collect();
            let uniform: Vec<Run> = lays
                .iter()
                .map(|l| run(&l.ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, rows, Some(glm_p2_moe::DEFAULT_ORDER)))
                .collect();
            // the composite pair buffer: each valid pair from the uniform layer that holds its expert; the combine over it
            let mut y = vec![f32::NAN; rows * d.topk * d.hidden];
            for (p, &e) in pick.iter().enumerate().filter(|(_, &e)| valid(e, &d)) {
                let rg = p * d.hidden..(p + 1) * d.hidden;
                y[rg.clone()].copy_from_slice(&uniform[of(e as usize)].y[rg]);
            }
            let mut o = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
            glm_p2_moe::combine_dims(
                CombineMode::Both,
                &up(&y),
                &up(&pick),
                &up(&wts),
                Some(&up(&shared)),
                None,
                &mut o,
                rows,
                &d,
                &st,
            )
            .unwrap();
            st.sync().unwrap();
            let want = Run { out: o.to_vec::<f32>().unwrap(), y };
            s.poison(0x5a, &st).unwrap();
            let got = run_sub(&sb, &st, &mut s, &mut os, &mut sd, &xh, &pick, &wts, &shared, rows, rows);
            assert_same(&format!("mixed widths {style:?} {rows} rows"), &got, &want, &d, &pick);
            for chunk in [16usize, 129] {
                if chunk < rows {
                    let got = run_sub(&sb, &st, &mut s, &mut os, &mut sd, &xh, &pick, &wts, &shared, rows, chunk);
                    assert_same(&format!("mixed widths {style:?} {rows} rows in calls of {chunk}"), &got, &want, &d, &pick);
                }
            }
        }
    }
    eprintln!("mixed-width subbanks (gate/up 2/3/4 bits, down 3/2/4): every pair is the uniform bank's, the out is the one combine, bit for bit");
}

/// A layer with a SPLIT gate/up subbank (gate 2 bits, up 3: no P2 GEMM takes it) beside fused subbanks of 2/2/2 and 4/4/4 bits, every
/// expert in exactly one. Under P2 the split subbank's pairs run the decode pairs stage in windows of the decode scratch: each split pair's
/// y must be the decode path's (the whole layer's `run_pairs` in 64-row windows), each fused pair's the uniform P2 bank's, the out the one
/// combine over those pairs; in calls of 16 and 129 rows, with decode windows of 128 and 7 rows, run to run (poisoned scratches). Then two
/// ranks of the range ownership (each its subbanks of its share: `pairs_subbanks_rank`, its `Own(r)` part, the `Parts` sum): every
/// owned pair's y and every row's out equal the one node's, bit for bit.
#[test]
fn p2_split_gate_up_subbank_is_the_decode_pairs_and_two_ranks_equal_one_node() {
    gpu!();
    let st = Stream::new().unwrap();
    let d = glm_dims(288);
    let base = synth_layer(d, 2, 2, 61, true);
    let wide = synth_layer(d, 4, 4, 62, true);
    let (bg, wg) = (base.groups.as_ref().unwrap(), wide.groups.as_ref().unwrap());
    // which experts are split (2/3/2), which 4/4/4, the rest 2/2/2; experts on both sides of the 144 cut in each
    let split = |e: usize| e % 5 == 2 || (140..148).contains(&e);
    let four = |e: usize| !split(e) && e % 7 == 3;
    let up3: Vec<LoadedGroup> =
        (0..d.experts).map(|e| synth_group(d.hidden, d.inter, 3, Codebook::Mul1, 63_000 + e as u64)).collect();
    let sel = |i: usize| -> Vec<LoadedGroup> {
        (0..d.experts)
            .map(|e| match (four(e), split(e), i) {
                (true, _, _) => wg[i][e].clone(),
                (_, true, 1) => up3[e].clone(),
                _ => bg[i][e].clone(),
            })
            .collect()
    };
    let (gate, upg, down) = (sel(0), sel(1), sel(2));
    let of = |own: ExpertOwnership| MoeSubbanks::from_groups(&gate, &upg, &down, d, own).unwrap();
    let whole = of(ExpertOwnership::all(d.experts));
    let ranks = [of(ExpertOwnership::range(d.experts, 0, 2).unwrap()), of(ExpertOwnership::range(d.experts, 1, 2).unwrap())];
    let keys: Vec<_> = whole.banks().iter().map(|b| b.subbank_key()).collect();
    assert_eq!(keys.len(), 3, "{keys:?}");
    assert!(whole.has_split_gate_up() && ranks.iter().all(|r| r.has_split_gate_up()), "{keys:?}");
    assert!(glm_p2_moe::supported_subbanks(&whole).is_ok());
    let max = 300;
    let mut s = PrefillScratch::new(&d, max).unwrap();
    let mut os = OwnScratch::new(&d, max).unwrap();
    for (si, &style) in [Style::Uniform, Style::Holes, Style::Skewed].iter().enumerate() {
        for rows in [16usize, 129, 300] {
            let mut r = Rng(2500 + 17 * si as u64 + rows as u64);
            let xh = activations(&mut r, rows, d.hidden, 1.5);
            let (mut pick, wts) = make_picks(&mut r, style, rows, d.experts, d.topk);
            // a split expert of each side of the cut in every 5th row (every window of 7 rows sees one)
            for row in (0..rows).step_by(5) {
                let sl = &mut pick[row * d.topk..(row + 1) * d.topk];
                for (k, e) in [(1usize, 142i32), (6, 147)] {
                    if !sl.contains(&e) {
                        sl[k] = e;
                    }
                }
            }
            let shared: Vec<f32> = (0..rows * d.hidden).map(|_| 0.1 * normal(&mut r) as f32).collect();
            let ord = Some(glm_p2_moe::DEFAULT_ORDER);
            let p2_base = run(&base.ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, rows, ord);
            let p2_wide = run(&wide.ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, rows, ord);
            // the decode pairs stage of the whole layer in 64-row windows (the split pairs' reference)
            let mut y_dec = vec![0f32; rows * d.topk * d.hidden];
            {
                let mut sw = MoeScratch::new(&d, 64).unwrap();
                let mut a = 0;
                while a < rows {
                    let n = (rows - a).min(64);
                    sw.set_picks(&pick[a * d.topk..(a + n) * d.topk], &wts[a * d.topk..(a + n) * d.topk]).unwrap();
                    whole.run_pairs(&up(&xh[a * d.hidden..(a + n) * d.hidden]), Dtype::F16, n, &mut sw, &mut os, &st).unwrap();
                    st.sync().unwrap();
                    let yw: Vec<f32> = sw.y_pairs.to_vec().unwrap();
                    y_dec[a * d.topk * d.hidden..(a + n) * d.topk * d.hidden].copy_from_slice(&yw[..n * d.topk * d.hidden]);
                    a += n;
                }
            }
            let mut y = vec![f32::NAN; rows * d.topk * d.hidden];
            for (p, &e) in pick.iter().enumerate().filter(|(_, &e)| valid(e, &d)) {
                let rg = p * d.hidden..(p + 1) * d.hidden;
                let src = if split(e as usize) {
                    &y_dec
                } else if four(e as usize) {
                    &p2_wide.y
                } else {
                    &p2_base.y
                };
                y[rg.clone()].copy_from_slice(&src[rg]);
            }
            let (pk, w, sh) = (up(&pick), up(&wts), up(&shared));
            let mut o = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
            glm_p2_moe::combine_dims(CombineMode::Both, &up(&y), &pk, &w, Some(&sh), None, &mut o, rows, &d, &st).unwrap();
            st.sync().unwrap();
            let want = Run { out: o.to_vec::<f32>().unwrap(), y };
            for window in [128usize, 7] {
                let mut sd = MoeScratch::new(&d, window).unwrap();
                sd.poison(0x7f, &st).unwrap();
                s.poison(0x5a, &st).unwrap();
                let r0 = glm_p2_moe::runs();
                let got = run_sub(&whole, &st, &mut s, &mut os, &mut sd, &xh, &pick, &wts, &shared, rows, rows);
                assert_eq!(glm_p2_moe::runs() - r0, 1, "one P2 expert block counted for one layer call");
                let tag = format!("split {style:?} {rows} rows, decode windows of {window}");
                assert_same(&tag, &got, &want, &d, &pick);
                for chunk in [16usize, 129] {
                    if chunk < rows {
                        let got = run_sub(&whole, &st, &mut s, &mut os, &mut sd, &xh, &pick, &wts, &shared, rows, chunk);
                        assert_same(&format!("{tag} in calls of {chunk}"), &got, &want, &d, &pick);
                    }
                }
                // two ranks: each its share's subbanks, its own group's part; the parts' sum with the shared expert last
                let x = up(&xh);
                let mut parts = [DevBuf::zeroed(rows * d.hidden * 4).unwrap(), DevBuf::zeroed(rows * d.hidden * 4).unwrap()];
                for (k, sb) in ranks.iter().enumerate() {
                    s.poison(0x33 + k as u8, &st).unwrap();
                    sd.poison(0x7f, &st).unwrap();
                    s.set_picks(&pick, &wts).unwrap();
                    let order = glm_p2_moe::DEFAULT_ORDER;
                    glm_p2_moe::pairs_subbanks_rank(sb, &x, Dtype::F16, rows, &mut s, &mut os, &mut sd, order, &st, &mut |_| Ok(()))
                        .unwrap();
                    let mode = CombineMode::Own(k);
                    glm_p2_moe::combine_dims(mode, &s.y_pairs, &s.pick, &s.wts, None, None, &mut parts[k], rows, &d, &st).unwrap();
                    st.sync().unwrap();
                    let yr: Vec<f32> = s.y_pairs.to_vec().unwrap();
                    for (p, &e) in pick.iter().enumerate().filter(|(_, &e)| valid(e, &d) && sb.ownership().owns(e as usize)) {
                        let rg = p * d.hidden..(p + 1) * d.hidden;
                        assert!(
                            yr[rg.clone()].iter().zip(&want.y[rg]).all(|(a, b)| a.to_bits() == b.to_bits()),
                            "{tag}: rank {k}'s y of pair {p} (expert {e})"
                        );
                    }
                }
                let mut o = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
                let pp = Some((&parts[0], &parts[1]));
                glm_p2_moe::combine_dims(CombineMode::Parts, &s.y_pairs, &pk, &w, Some(&sh), pp, &mut o, rows, &d, &st).unwrap();
                st.sync().unwrap();
                let tp: Vec<f32> = o.to_vec().unwrap();
                assert!(tp.iter().zip(&want.out).all(|(a, b)| a.to_bits() == b.to_bits()), "{tag}: two ranks' out != one node's");
            }
        }
    }
    eprintln!(
        "split gate/up subbank (2/3/2) beside 2/2/2 and 4/4/4: split pairs == the decode pairs, fused pairs == the uniform P2 banks, out == \
         the one combine, in calls of 16 / 129 rows and decode windows of 128 / 7, run to run; two ranks == one node, bit for bit"
    );
}

#[test]
fn a_layer_of_subbanks_the_p2_gemms_do_not_take_is_refused_by_name() {
    gpu!();
    let d = MoeDims { hidden: 128, inter: 128, experts: 16, topk: 8, scale: 2.5, limit: 10.0 };
    let mk = |bits: usize, seed: u64| synth_layer(d, bits, bits, seed, true);
    // 7-bit experts: no P2 kernel; the supported subbanks of the same layer are
    let (a, b) = (mk(2, 1), mk(7, 2));
    fn parts(l: &Layer, ids: Vec<u32>) -> (Vec<u32>, &[Vec<LoadedGroup>; 3]) {
        (ids, l.groups.as_ref().unwrap())
    }
    let sb = subbanks_of(d, &[parts(&a, (0..8).collect()), parts(&b, (8..16).collect())]);
    let why = glm_p2_moe::supported_subbanks(&sb).unwrap_err();
    assert!(why.contains("subbank 1") && why.contains("7/7/7"), "{why}");
    let st = Stream::new().unwrap();
    let mut s = PrefillScratch::new(&d, 16).unwrap();
    let mut os = OwnScratch::new(&d, 16).unwrap();
    let mut sd = MoeScratch::new(&d, 16).unwrap();
    let x = DevBuf::zeroed(16 * d.hidden * 2).unwrap();
    let mut o = DevBuf::zeroed(16 * d.hidden * 4).unwrap();
    assert!(glm_p2_moe::run_subbanks(&sb, &x, Dtype::F16, 16, None, &mut o, &mut s, &mut os, &mut sd, &st).is_err());
    // a rank's share (not every expert) is the tensor-parallel path's, not this one
    let c = mk(2, 3);
    let own = ExpertOwnership::from_ids(16, (0..8).collect()).unwrap();
    let g = c.groups.as_ref().unwrap();
    let ids: Vec<usize> = (0..8).collect();
    let pk = |v: &Vec<LoadedGroup>| -> Vec<LoadedGroup> { ids.iter().map(|&e| v[e].clone()).collect() };
    let half = MoeSubbanks::new(
        d,
        own.clone(),
        vec![MoeExperts::from_groups_owned(&pk(&g[0]), &pk(&g[1]), &pk(&g[2]), d, own).unwrap()],
    )
    .unwrap();
    assert!(glm_p2_moe::supported_subbanks(&half).is_ok());
    assert!(glm_p2_moe::run_subbanks(&half, &x, Dtype::F16, 16, None, &mut o, &mut s, &mut os, &mut sd, &st).is_err());
    eprintln!("refused by name: {why}");
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// One timed call of the experts (`path`: None P1's large-row path, Some(order) P2), after a 256 MiB memset (cold L2); with `stages`, a
/// sync after each P2 stage and its host time recorded by name.
fn timed(
    ex: &MoeExperts,
    x: &DevBuf,
    sh: &DevBuf,
    out: &mut DevBuf,
    s: &mut PrefillScratch,
    rows: usize,
    path: Option<Order>,
    flush: &mut DevBuf,
    it: usize,
    st: &Stream,
    stages: Option<&mut std::collections::BTreeMap<&'static str, Vec<f64>>>,
) -> f64 {
    flush.memset_async(it as u8, st).unwrap();
    st.sync().unwrap();
    let t = std::time::Instant::now();
    match path {
        None => ex.run_prefill(x, Dtype::F16, rows, Some(sh), out, s, st).unwrap(),
        Some(ord) => match stages {
            None => glm_p2_moe::run_with(ex, x, Dtype::F16, rows, Some(sh), out, s, ord, st, &mut |_| Ok(())).unwrap(),
            Some(per) => {
                let mut last = std::time::Instant::now();
                glm_p2_moe::run_with(ex, x, Dtype::F16, rows, Some(sh), out, s, ord, st, &mut |name| {
                    st.sync()?;
                    let now = std::time::Instant::now();
                    per.entry(name).or_default().push(now.duration_since(last).as_secs_f64() * 1e3);
                    last = now;
                    Ok(())
                })
                .unwrap()
            }
        },
    }
    st.sync().unwrap();
    t.elapsed().as_secs_f64() * 1e3
}

/// ms a layer, P1's large-row path against P2 (both orders), 288 experts, uniform routing (REXL3_P2M_BITS, REXL3_P2M_ROWS and
/// REXL3_P2M_INTER override the widths, row counts and expert width), median of 5 after a warm-up; the P2 stages from a separate staged run.
#[test]
#[ignore]
fn bench_p2_experts_against_p1() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut flush = DevBuf::alloc(256 << 20).unwrap();
    let widths: Vec<usize> = match std::env::var("REXL3_P2M_BITS") {
        Ok(v) => v.split(',').map(|b| b.parse().unwrap()).collect(),
        Err(_) => vec![2, 3],
    };
    let row_list: Vec<usize> = match std::env::var("REXL3_P2M_ROWS") {
        Ok(v) => v.split(',').map(|b| b.parse().unwrap()).collect(),
        Err(_) => vec![512, 1024, 2048, 4096],
    };
    // REXL3_P2M_INTER: the expert width (1,024: a rank's half of GLM's 2,048 under the width split)
    let d = MoeDims { inter: std::env::var("REXL3_P2M_INTER").map_or(2048, |v| v.parse().unwrap()), ..glm_dims(288) };
    for &bits in &widths {
        let lay = synth_layer(d, bits, bits, 9 + bits as u64, false);
        let ex = &lay.ex;
        let mut s = PrefillScratch::new(&d, *row_list.iter().max().unwrap()).unwrap();
        for &rows in &row_list {
            let mut r = Rng(rows as u64);
            let xh = activations(&mut r, rows, d.hidden, 1.5);
            let (pick, wts) = make_picks(&mut r, Style::Uniform, rows, d.experts, d.topk);
            let x = up(&xh);
            let sh = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
            let mut out = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
            s.set_picks(&pick, &wts).unwrap();
            let flop = 2.0 * 3.0 * (d.hidden * d.inter) as f64 * (rows * d.topk) as f64;
            let med = |path: Option<Order>, out: &mut DevBuf, s: &mut PrefillScratch, flush: &mut DevBuf| -> f64 {
                median((0..6).map(|it| timed(ex, &x, &sh, out, s, rows, path, flush, it, &st, None)).skip(1).collect())
            };
            let t1 = med(None, &mut out, &mut s, &mut flush);
            let mut line = format!("{bits}-bit rows {rows:4}: P1 {t1:7.2} ms {:5.1} TF", flop / t1 / 1e9);
            let shapes: Vec<(i32, i32)> = match std::env::var("REXL3_P2M_SHAPES") {
                Ok(v) => v
                    .split(',')
                    .map(|p| {
                        let (a, b) = p.split_once('/').unwrap();
                        (a.parse().unwrap(), b.parse().unwrap())
                    })
                    .collect(),
                Err(_) => vec![(0, 0)],
            };
            for (&ord, &(gu, dn)) in ORDERS.iter().flat_map(|o| shapes.iter().map(move |sh| (o, sh))) {
                glm_p2_moe::set_shapes(gu, dn);
                let tp = med(Some(ord), &mut out, &mut s, &mut flush);
                let mut per = std::collections::BTreeMap::new();
                for it in 0..4 {
                    timed(ex, &x, &sh, &mut out, &mut s, rows, Some(ord), &mut flush, it, &st, Some(&mut per));
                }
                let st_txt: Vec<String> = per.into_iter().map(|(k, v)| format!("{k} {:.2}", median(v))).collect();
                line += &format!(
                    " | P2 {} shape {gu}/{dn} {tp:7.2} ms {:5.1} TF x{:.2} [{}]",
                    if ord == Order::Chain { "chain" } else { "fold" },
                    flop / tp / 1e9,
                    t1 / tp,
                    st_txt.join(", ")
                );
            }
            glm_p2_moe::set_shapes(0, 0);
            eprintln!("{line}");
        }
    }
}

/// ms a layer of a rank under the width split as RED-SNOW 4.91bpw binds it: 288 experts at an expert width of 1,024
/// (REXL3_P2M_INTER), five interleaved subbanks of gate/up / down bits 6/4, 6/6, 5/4, 5/5 and 4/4, uniform routing over REXL3_P2M_ROWS
/// rows (default 4,096): the P2 pairs stage (fold) and the combine, median of 5 after a warm-up, and each stage's time summed over the
/// subbanks from a separate staged run. Also checks that the staged and unstaged runs give the same bits.
#[test]
#[ignore]
fn bench_p2_subbanks() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut flush = DevBuf::alloc(256 << 20).unwrap();
    let d = MoeDims { inter: std::env::var("REXL3_P2M_INTER").map_or(1024, |v| v.parse().unwrap()), ..glm_dims(288) };
    let rows: usize = std::env::var("REXL3_P2M_ROWS").map_or(4096, |v| v.parse().unwrap());
    let widths = [(6usize, 4usize), (6, 6), (5, 4), (5, 5), (4, 4)];
    let mut banks = Vec::new();
    for (i, &(bg, bd)) in widths.iter().enumerate() {
        let ids: Vec<u32> = (0..d.experts as u32).filter(|e| *e as usize % widths.len() == i).collect();
        let mk = |k, n, bits, tag: u64| -> Vec<LoadedGroup> {
            ids.iter().map(|&x| synth_group(k, n, bits, Codebook::Mul1, 77_000 + tag * 1000 + x as u64)).collect()
        };
        let (g, u, dn) = (mk(d.hidden, d.inter, bg, 1), mk(d.hidden, d.inter, bg, 2), mk(d.inter, d.hidden, bd, 3));
        let o = ExpertOwnership::from_ids(d.experts, ids.clone()).unwrap();
        banks.push(MoeExperts::from_groups_owned(&g, &u, &dn, d, o).unwrap());
    }
    let sb = MoeSubbanks::new(d, ExpertOwnership::all(d.experts), banks).unwrap();
    let mut s = PrefillScratch::new(&d, rows).unwrap();
    let mut os = OwnScratch::new(&d, rows).unwrap();
    let mut sd = MoeScratch::new(&d, 128).unwrap();
    let mut r = Rng(rows as u64 + 5);
    let xh = activations(&mut r, rows, d.hidden, 1.5);
    let (pick, wts) = make_picks(&mut r, Style::Uniform, rows, d.experts, d.topk);
    let x = up(&xh);
    let sh = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
    let mut out = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
    s.set_picks(&pick, &wts).unwrap();
    let flop = 2.0 * 3.0 * (d.hidden * d.inter) as f64 * (rows * d.topk) as f64;
    let ord = glm_p2_moe::DEFAULT_ORDER;
    let mut call = |per: Option<&mut std::collections::BTreeMap<&'static str, f64>>,
                    s: &mut PrefillScratch,
                    out: &mut DevBuf,
                    it: usize|
     -> f64 {
        flush.memset_async(it as u8, &st).unwrap();
        st.sync().unwrap();
        let t = std::time::Instant::now();
        let mut last = t;
        let mut per = per;
        glm_p2_moe::pairs_subbanks_with(&sb, &x, Dtype::F16, rows, s, &mut os, &mut sd, ord, &st, &mut |name| {
            if let Some(p) = per.as_deref_mut() {
                st.sync()?;
                let now = std::time::Instant::now();
                *p.entry(name).or_default() += now.duration_since(last).as_secs_f64() * 1e3;
                last = now;
            }
            Ok(())
        })
        .unwrap();
        glm_p2_moe::combine_dims(CombineMode::Both, &s.y_pairs, &s.pick, &s.wts, Some(&sh), None, out, rows, &d, &st).unwrap();
        st.sync().unwrap();
        if let Some(p) = per {
            *p.entry("combine").or_default() += last.elapsed().as_secs_f64() * 1e3;
        }
        t.elapsed().as_secs_f64() * 1e3
    };
    let tp = median((0..6).map(|it| call(None, &mut s, &mut out, it)).skip(1).collect());
    let want: Vec<u32> = out.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect();
    let mut stages = Vec::new();
    for it in 0..4 {
        let mut per = std::collections::BTreeMap::new();
        call(Some(&mut per), &mut s, &mut out, it);
        stages.push(per);
    }
    let got: Vec<u32> = out.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect();
    assert!(want == got, "the staged run gave other bits");
    let names: Vec<&'static str> = stages[0].keys().copied().collect();
    let st_txt: Vec<String> = names.iter().map(|k| format!("{k} {:.2}", median(stages.iter().map(|p| p[k]).collect()))).collect();
    eprintln!(
        "5 subbanks, inter {}, rows {rows}: P2 {tp:7.2} ms {:5.1} TF [{}] out {:016x}",
        d.inter,
        flop / tp / 1e9,
        st_txt.join(", "),
        gb_exl3::fnv1a(want.iter().flat_map(|v| v.to_le_bytes()))
    );
}

/// The golden of the P2 experts (`testdata/gpu/golden_glm_p2_moe.txt`, scripts/gpu_gate.sh): hashes of y and out in both K orders on
/// synthetic layers at the GLM widths and the packs' bit widths, at row counts that are and are not multiples of a tile; committed so
/// that a change that moves a P2 expert bit shows up in the gate (and `glm_p2_moe::ARITH_ID` must change with it).
#[test]
#[ignore]
fn golden_glm_p2_moe_hashes() {
    gpu!();
    let path = std::env::var("REXL3_GOLDEN_OUT").expect("REXL3_GOLDEN_OUT");
    let st = Stream::new().unwrap();
    let hash = |v: &[f32]| gb_exl3::fnv1a(v.iter().flat_map(|x| x.to_bits().to_le_bytes()));
    let mut lines = vec![format!("arith {} default {:?}", glm_p2_moe::ARITH_ID, glm_p2_moe::DEFAULT_ORDER)];
    for (li, &(bg, bd)) in [(2usize, 2usize), (3, 3), (4, 4), (5, 4), (6, 4), (6, 6)].iter().enumerate() {
        let d = glm_dims(48);
        let lay = synth_layer(d, bg, bd, 70 + li as u64, false);
        let ex = &lay.ex;
        for rows in [1usize, 37, 300] {
            let mut r = Rng(800 + li as u64 * 10 + rows as u64);
            let xh = activations(&mut r, rows, d.hidden, 1.5);
            let (pick, wts) = make_picks(&mut r, Style::Holes, rows, d.experts, d.topk);
            let shared: Vec<f32> = (0..rows * d.hidden).map(|_| 0.1 * normal(&mut r) as f32).collect();
            let mut s = PrefillScratch::new(&d, rows).unwrap();
            for &o in &ORDERS {
                let got = run(ex, &st, &mut s, &xh, &pick, &wts, &shared, rows, rows, Some(o));
                let yv: Vec<f32> = pick
                    .iter()
                    .enumerate()
                    .filter(|(_, &e)| valid(e, &d))
                    .flat_map(|(p, _)| got.y[p * d.hidden..(p + 1) * d.hidden].to_vec())
                    .collect();
                lines.push(format!(
                    "p2m {} {bg}/{bd}-bit E=48 rows={rows} y {:016x} out {:016x}",
                    o.id(),
                    hash(&yv),
                    hash(&got.out)
                ));
            }
        }
    }
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    eprintln!("wrote {} golden lines to {path}", lines.len());
}
