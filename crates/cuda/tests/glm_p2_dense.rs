//! Lane P (GLM-5.3-Flash, contract P2): the dense EXL3 linears and the gated MLPs of a GLM prompt chunk on the P2 tensor-core GEMM
//! (`gb_cuda::glm_p2`), on SYNTHETIC Mul1 EXL3 weights of GLM's shapes (`common::exl3_synth`; the shapes and widths of the 2.25 bpw pack:
//! KDA qkv 4096 x 24576 and o 8192 x 4096 at 3 and 4 bits, MLA q_a 4096 x 1536, q_b 1536 x 16384, kv_a 4096 x 512 (4 and 5 bits), o
//! 16384 x 4096, the dense MLP 4096 x 12288 x 4096 at 3 bits, the shared expert 4096 x 2048 x 4096 at 3 and 4, the NextN eh_proj
//! 8192 x 4096; and sc_4.10's 6 and 8 bit dense layers on two shapes). Inputs are fp16, as every GLM call site passes them.
//!
//! Gates (not ignored), for both K orders (`glm_p2::Variant`: `Chain`, `Fold128`):
//! * `p2_is_as_close_to_fp64_as_the_decode_path_on_the_glm_shapes`: per shape, the fp64 reference of the exact dequantized weights on
//!   sampled rows; P2's max, rms and signed (mean, mean of err x sign(ref): a shrink toward zero when negative) error beside the decode
//!   path's (`Exl3Weights::forward` in 128-row calls). P2 must be within 1.5x of the decode path's rms error and under 5e-4.
//! * `p2_rows_are_bit_identical_in_any_call_on_the_glm_shapes`: repeat, slice widths (whole, 256, 129, 128, 100, 17, 1), every tile
//!   shape, a row permutation, NaN rows after the call's end: bit for bit.
//! * `glm_chain_has_nemotrons_bits`: GLM's own copy of the kernel in `Chain` order gives Nemotron's `gb_ndm_gemm` bits on every shape
//!   (the copy is faithful; `Fold128` is the one change).
//! * `the_p2_gated_mlp_is_row_invariant_and_composes_its_linears`: the gated MLP (shared expert, dense MLP) in pieces of any size
//!   equals one call, and equals gate -> up -> clamped SwiGLU -> down composed by hand from `linear`.
//!
//! Timing (ignored, `bench` in the name): ms and TFLOP/s per call at 512 and 2,048 rows, P2 against the large-row kernel (`forward_big`,
//! what a P1 prompt chunk runs), cold (a 256 MiB memset before each call), median of 7.
//!
//!     cargo test -p gb-cuda --release --test glm_p2_dense -- --test-threads=1 --nocapture
//!     cargo test -p gb-cuda --release --test glm_p2_dense -- --ignored --test-threads=1 --nocapture bench
#![cfg(not(gb_no_cuda))]
#![allow(clippy::needless_range_loop)]
mod common;

use common::exl3_synth::synth_group;
use common::Rng;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::glm_p2::{self, Variant};
use gb_cuda::moe::{SharedExpert, SharedScratch};
use gb_cuda::moe_prefill::SharedScratchBig;
use gb_cuda::{device_available, DevBuf, Stream};
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

/// (name, K, N, bits): every dense EXL3 linear shape and width of the 2.25 bpw pack, and two of sc_4.10's widths.
const SHAPES: &[(&str, usize, usize, usize)] = &[
    ("kda qkv", 4096, 24576, 4),
    ("kda qkv 3b", 4096, 24576, 3),
    ("kda o", 8192, 4096, 4),
    ("kda o 3b", 8192, 4096, 3),
    ("mla q_a", 4096, 1536, 4),
    ("mla q_b", 1536, 16384, 4),
    ("mla kv_a", 4096, 512, 4),
    ("mla kv_a 5b", 4096, 512, 5),
    ("mla o", 16384, 4096, 4),
    ("mla o 3b", 16384, 4096, 3),
    ("dense gate/up", 4096, 12288, 3),
    ("dense down", 12288, 4096, 3),
    ("shared gate/up", 4096, 2048, 4),
    ("shared down", 2048, 4096, 4),
    ("shared down 3b", 2048, 4096, 3),
    ("nextn eh_proj", 8192, 4096, 4),
    ("sc410 kda o 8b", 8192, 4096, 8),
    ("sc410 mla o 6b", 16384, 4096, 6),
];

fn layer(k: usize, n: usize, bits: usize, seed: u64) -> (Exl3Weights, gb_exl3::checkpoint::LoadedGroup) {
    let g = synth_group(k, n, bits, Codebook::Mul1, seed);
    (Exl3Weights::from_loaded(&g).unwrap(), g)
}

/// fp16 activations with heavier tails (a few percent of entries four times larger), as the fp16 bits and their values.
fn activations(seed: u64, rows: usize, k: usize, scale: f64) -> (Vec<u16>, Vec<f64>) {
    let mut r = Rng(seed);
    let bits: Vec<u16> = (0..rows * k)
        .map(|_| {
            let v = r.uniform(-1.0, 1.0) as f64;
            let v = if r.next_u64() % 32 == 0 { 4.0 * v } else { v };
            f64_to_f16(v * scale)
        })
        .collect();
    let vals = bits.iter().map(|&b| f16_to_f64(b)).collect();
    (bits, vals)
}

fn upload(x: &[u16]) -> DevBuf {
    let mut d = DevBuf::zeroed_view(x.len() * 2).unwrap();
    d.upload(x).unwrap();
    d
}

const VARIANTS: [Variant; 2] = [Variant::Chain, Variant::Fold128];

fn p2v(w: &Exl3Weights, x: &DevBuf, rows: usize, v: Variant, tile: i32, st: &Stream) -> Vec<f32> {
    let mut y = DevBuf::from_slice(&vec![f32::from_bits(0x7fc0_dead); rows * w.n]).unwrap();
    let mut ws = DevBuf::alloc(glm_p2::workspace_bytes(rows, w.k)).unwrap();
    glm_p2::linear_variant(w, x, &mut y, rows, Dtype::F16, Dtype::F32, &mut ws, v, tile, st).unwrap();
    st.sync().unwrap();
    y.to_vec::<f32>().unwrap()
}

/// The decode path in calls of at most 128 rows (what a window, or a P1 chunk of at most 128 rows, runs; `forward_big` has its bits).
fn decode_path(w: &Exl3Weights, x: &DevBuf, rows: usize, st: &Stream) -> Vec<f32> {
    let y = DevBuf::zeroed_view(rows * w.n * 4).unwrap();
    let mut ws = DevBuf::alloc(w.workspace_bytes(128)).unwrap();
    let mut s = 0;
    while s < rows {
        let m = (rows - s).min(128);
        let xs = x.subview(s * w.k * 2, m * w.k * 2);
        let mut ys = y.subview(s * w.n * 4, m * w.n * 4);
        w.forward(&xs, &mut ys, m, Dtype::F16, Dtype::F32, &mut ws, st).unwrap();
        s += m;
    }
    st.sync().unwrap();
    y.to_vec::<f32>().unwrap()
}

fn fwht(v: &mut [f64]) {
    let mut h = 1;
    while h < v.len() {
        for i in (0..v.len()).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

/// The exact dequantized weights [K][N] as fp16 bits (the values the kernels decode).
fn weights_f16(w: &Exl3Weights, st: &Stream) -> Vec<u16> {
    let mut wq = DevBuf::alloc(w.k * w.n * 2).unwrap();
    w.reconstruct(&mut wq, st).unwrap();
    st.sync().unwrap();
    wq.to_vec::<u16>().unwrap()
}

/// y rows of the exact layer in f64: ((x * suh) H / sqrt 128) . W_q . H / sqrt 128 * svh.
fn reference(g: &gb_exl3::checkpoint::LoadedGroup, wq: &[u16], x: &[f64], rows: &[usize]) -> Vec<f64> {
    let (k, n) = (g.group.k, g.group.n);
    let lut: Vec<f64> = (0..=u16::MAX).map(f16_to_f64).collect();
    let s = 1.0 / 128f64.sqrt();
    let mut out = Vec::with_capacity(rows.len() * n);
    for &r in rows {
        let mut xr: Vec<f64> = (0..k).map(|i| x[r * k + i] * g.suh[i]).collect();
        xr.chunks_mut(128).for_each(fwht);
        let mut c = vec![0f64; n];
        for i in 0..k {
            let a = xr[i] * s;
            let wrow = &wq[i * n..(i + 1) * n];
            for j in 0..n {
                c[j] += a * lut[wrow[j] as usize];
            }
        }
        c.chunks_mut(128).for_each(fwht);
        out.extend((0..n).map(|j| c[j] * s * g.svh[j]));
    }
    out
}

/// Errors over the sampled rows, relative to the reference: (max abs / max |ref|, rms / rms ref, mean / rms ref, mean of err x sign(ref) /
/// rms ref). The last two show a systematic bias (the third a shift, the fourth a shrink toward zero when negative).
fn err(refv: &[f64], got: &[f32], rows: &[usize], n: usize) -> (f64, f64, f64, f64) {
    let (mut ee, mut rr, mut mx, mut mr, mut sum, mut shrink) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    let cnt = (rows.len() * n) as f64;
    for (i, &r) in rows.iter().enumerate() {
        for j in 0..n {
            let (w, g) = (refv[i * n + j], got[r * n + j] as f64);
            let e = g - w;
            ee += e * e;
            rr += w * w;
            mx = mx.max(e.abs());
            mr = mr.max(w.abs());
            sum += e;
            shrink += e * w.signum();
        }
    }
    let rms = (rr / cnt).sqrt();
    (mx / mr, (ee / rr).sqrt(), sum / cnt / rms, shrink / cnt / rms)
}

#[test]
fn p2_is_as_close_to_fp64_as_the_decode_path_on_the_glm_shapes() {
    gpu!();
    let st = Stream::new().unwrap();
    let rows = 160;
    let sample: Vec<usize> = (0..rows).step_by(53).collect();
    for (i, &(name, k, n, bits)) in SHAPES.iter().enumerate() {
        let (w, g) = layer(k, n, bits, 1000 + i as u64);
        assert!(glm_p2::supported(&w), "{name}: the P2 GEMM must take every GLM dense shape");
        let wq = weights_f16(&w, &st);
        let (xb, xv) = activations(77 + i as u64, rows, k, 1.0);
        let xd = upload(&xb);
        let refv = reference(&g, &wq, &xv, &sample);
        let (dm, dr, ds, dk) = err(&refv, &decode_path(&w, &xd, rows, &st), &sample, n);
        let mut line =
            format!("{name:15} K {k:5} N {n:5} {bits}b: decode path max {dm:.2e} rms {dr:.2e} mean {ds:+.1e} shrink {dk:+.1e}");
        for v in VARIANTS {
            let (pm, pr, ps, pk) = err(&refv, &p2v(&w, &xd, rows, v, 0, &st), &sample, n);
            line += &format!(" | {} max {pm:.2e} rms {pr:.2e} mean {ps:+.1e} shrink {pk:+.1e}", v.id());
            assert!(pr <= 1.5 * dr + 1e-7 && pr <= 5e-4, "{name} {v:?}: P2 rms error {pr:e} against the decode path's {dr:e}");
            assert!(pm <= 1.5 * dm + 1e-6, "{name} {v:?}: P2 max error {pm:e} against the decode path's {dm:e}");
        }
        eprintln!("{line}");
    }
}

#[test]
fn p2_rows_are_bit_identical_in_any_call_on_the_glm_shapes() {
    gpu!();
    let st = Stream::new().unwrap();
    let rows = 300;
    for (i, &(name, k, n, bits)) in SHAPES.iter().enumerate() {
        let (w, _) = layer(k, n, bits, 2000 + i as u64);
        let (xb, _) = activations(91 + i as u64, rows, k, 1.0);
        let xd = upload(&xb);
        for v in VARIANTS {
            let p2 = |w: &Exl3Weights, x: &DevBuf, rows: usize, tile: i32, st: &Stream| p2v(w, x, rows, v, tile, st);
            let whole = p2(&w, &xd, rows, 0, &st);
            assert!(
                whole.iter().all(|v| v.is_finite()) && whole.iter().any(|v| *v != 0.0),
                "{name}: output not finite or all zero"
            );
            let again = p2(&w, &xd, rows, 0, &st);
            assert!(whole.iter().zip(&again).all(|(a, b)| a.to_bits() == b.to_bits()), "{name}: run to run");
            // fold128 has no 128 x 2 tile (its accumulators would not fit the registers)
            let tiles: &[i32] = match (n >= 4096, v) {
                (true, Variant::Chain) => &[641, 642, 1281, 1282],
                (true, Variant::Fold128) => &[641, 642, 1281],
                (false, Variant::Chain) => &[641, 1282],
                (false, Variant::Fold128) => &[641, 1281],
            };
            let widths: &[usize] = if k * n > 40_000_000 { &[256, 129, 1] } else { &[256, 129, 128, 100, 17, 1] };
            for &tile in tiles {
                for &wd in widths {
                    let mut got = Vec::with_capacity(rows * n);
                    let mut s = 0;
                    while s < rows {
                        let m = wd.min(rows - s);
                        let xs = upload(&xb[s * k..(s + m) * k]);
                        got.extend(p2(&w, &xs, m, tile, &st));
                        s += m;
                    }
                    let bad = got.iter().zip(&whole).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    assert_eq!(bad, 0, "{name}: tile {tile}, slices of {wd}: {bad} values differ from the whole call");
                }
            }
            // a permutation of the rows, and NaN rows after the call's last row (never read)
            let perm: Vec<usize> = {
                let mut r = Rng(5 + i as u64);
                let mut p: Vec<usize> = (0..rows).collect();
                for j in (1..rows).rev() {
                    p.swap(j, (r.next_u64() % (j as u64 + 1)) as usize);
                }
                p
            };
            let m = rows - 37;
            let mut xp: Vec<u16> = perm[..m].iter().flat_map(|&r| xb[r * k..(r + 1) * k].to_vec()).collect();
            xp.extend(std::iter::repeat_n(0x7e00u16, 37 * k));
            let yp = p2(&w, &upload(&xp), m, 0, &st);
            for (j, &r) in perm[..m].iter().enumerate() {
                assert!(
                    yp[j * n..(j + 1) * n].iter().zip(&whole[r * n..(r + 1) * n]).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{name}: row {r} at position {j} of a permuted, NaN-padded call"
                );
            }
            eprintln!(
                "{name:15} {}: bit-identical over runs, tiles {tiles:?}, slices {widths:?}, a permutation and NaN padding",
                v.id()
            );
        }
    }
}

/// Lane P step P5b: the P2 GEMM on a STORED STRIP CUT (`Exl3Weights::select_strips_planned`: a rank's KDA qkv strips, 3 ranges of 32
/// heads' 128-column strips in q, k and v, laid out as a layer of the kept columns) gives, bit for bit, those columns of the P2 GEMM on the
/// whole layer: the P2 K order (fold128) is a function of K alone and its N tiles sit on strip boundaries, so neither the layer's N nor the
/// tile it picks for it moves a column. Both ranks, the KDA qkv shape at the packs' widths, rows from 1 to 2,048, every forced tile.
#[test]
fn p2_on_a_strip_cut_gives_those_columns_of_the_whole_layer() {
    gpu!();
    let st = Stream::new().unwrap();
    let (k, n) = (4096usize, 24576usize);
    let mut cases = 0;
    for (bi, &bits) in [2usize, 3, 4, 8].iter().enumerate() {
        let (w, _) = layer(k, n, bits, 3100 + bi as u64);
        for rows in [1usize, 37, 300, 2048] {
            let (xb, _) = activations(311 + rows as u64, rows, k, 1.0);
            let xd = upload(&xb);
            let whole = p2v(&w, &xd, rows, Variant::Fold128, 0, &st);
            for r in 0..2usize {
                let (h0, h1) = (32 * r, 32 * r + 32);
                let ranges = [(h0, h1), (64 + h0, 64 + h1), (128 + h0, 128 + h1)];
                let cut = w.select_strips_planned(&ranges).unwrap();
                assert_eq!(cut.w.n, 3 * 32 * 128);
                for tile in [0, 641, 1281] {
                    let got = p2v(&cut.w, &xd, rows, Variant::Fold128, tile, &st);
                    let mut c = 0usize;
                    for &(a, b) in &ranges {
                        for col in a * 128..b * 128 {
                            for row in 0..rows {
                                let (x, y) = (got[row * cut.w.n + c + (col - a * 128)], whole[row * n + col]);
                                assert!(
                                    x.to_bits() == y.to_bits(),
                                    "{bits}-bit, {rows} rows, rank {r}, tile {tile}: column {col} row {row}: {x} vs {y}"
                                );
                            }
                        }
                        c += (b - a) * 128;
                    }
                    cases += 1;
                }
            }
        }
    }
    eprintln!("P2 on strip cuts: {cases} cases (4 widths x 4 row counts x 2 ranks x 3 tiles), every column the whole layer's bit for bit");
}

#[test]
fn glm_chain_has_nemotrons_bits() {
    gpu!();
    let st = Stream::new().unwrap();
    let rows = 200;
    for (i, &(name, k, n, bits)) in SHAPES.iter().enumerate() {
        let (w, _) = layer(k, n, bits, 6000 + i as u64);
        let (xb, _) = activations(29 + i as u64, rows, k, 1.0);
        let xd = upload(&xb);
        let ours = p2v(&w, &xd, rows, Variant::Chain, 0, &st);
        let mut y = DevBuf::from_slice(&vec![f32::from_bits(0x7fc0_beef); rows * n]).unwrap();
        let mut ws = DevBuf::alloc(glm_p2::workspace_bytes(rows, k)).unwrap();
        gb_cuda::nemotron_dense_mma::linear_tiled(&w, &xd, &mut y, rows, Dtype::F16, Dtype::F32, &mut ws, 0, &st).unwrap();
        st.sync().unwrap();
        let theirs = y.to_vec::<f32>().unwrap();
        let bad = ours.iter().zip(&theirs).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert_eq!(bad, 0, "{name}: {bad} values of GLM's chain differ from gb_ndm_gemm's");
        let fold = p2v(&w, &xd, rows, Variant::Fold128, 0, &st);
        let moved = fold.iter().zip(&theirs).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        eprintln!("{name:15}: chain == gb_ndm_gemm bit for bit; fold128 moves {moved} of {} values", rows * n);
        assert!(k <= 128 || moved > 0, "{name}: fold128 should change the K order of a K > 128 layer");
    }
}

/// A synthetic gated MLP of `hidden -> inter -> hidden` at `bits`, the routed experts' clamp `limit`.
fn mlp(hidden: usize, inter: usize, bits: usize, limit: f32, seed: u64) -> SharedExpert {
    let (g, _) = layer(hidden, inter, bits, seed);
    let (u, _) = layer(hidden, inter, bits, seed + 1);
    let (d, _) = layer(inter, hidden, bits, seed + 2);
    SharedExpert::new(g, u, d, limit).unwrap()
}

#[test]
fn the_p2_gated_mlp_is_row_invariant_and_composes_its_linears() {
    gpu!();
    let st = Stream::new().unwrap();
    for (name, inter, bits, limit, rows) in [
        ("shared expert", 2048usize, 4usize, 7.0f32, 600usize),
        ("dense mlp", 12288, 3, 7.0, 300),
        ("tight clamp", 2048, 3, 0.25, 300),
    ] {
        let h = 4096;
        let m = mlp(h, inter, bits, limit, 4000 + inter as u64 + bits as u64);
        assert!(glm_p2::gated_supported(&m), "{name}");
        // activations at the scale of a normed hidden row times its weight (the gate's outputs then reach the clamp)
        let (xb, _) = activations(17 + inter as u64, rows, h, 2.0);
        let x = upload(&xb);
        let run = |pieces: Option<usize>| -> Vec<f32> {
            let mut out = DevBuf::zeroed_view(rows * h * 4).unwrap();
            match pieces {
                None => {
                    let mut sc = SharedScratchBig::new(&m, rows).unwrap();
                    glm_p2::gated(&m, &x, Dtype::F16, rows, &mut out, sc.p2_bufs(), &st).unwrap();
                }
                Some(p) if p <= 128 => {
                    let mut sc = SharedScratch::new(&m, p).unwrap();
                    glm_p2::gated(&m, &x, Dtype::F16, rows, &mut out, sc.p2_bufs(), &st).unwrap();
                }
                Some(p) => {
                    let mut sc = SharedScratchBig::new(&m, p).unwrap();
                    glm_p2::gated(&m, &x, Dtype::F16, rows, &mut out, sc.p2_bufs(), &st).unwrap();
                }
            }
            st.sync().unwrap();
            out.to_vec::<f32>().unwrap()
        };
        let whole = run(None);
        assert!(whole.iter().all(|v| v.is_finite()) && whole.iter().any(|v| *v != 0.0), "{name}: not finite or all zero");
        for p in [256usize, 128, 100, 17] {
            let got = run(Some(p));
            let bad = got.iter().zip(&whole).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(bad, 0, "{name}: pieces of {p}: {bad} values differ from one call");
        }
        // composed by hand: gate, up (P2, f32), the clamped SwiGLU on the host in f32 (the elementwise kernel's operations), down (P2)
        let mut gbuf = DevBuf::alloc(rows * inter * 4).unwrap();
        let mut ubuf = DevBuf::alloc(rows * inter * 4).unwrap();
        let mut ws = DevBuf::alloc(glm_p2::workspace_bytes(rows, inter.max(h))).unwrap();
        let (gw, uw, dw) = (mlp_part(&m, 0), mlp_part(&m, 1), mlp_part(&m, 2));
        glm_p2::linear(gw, &x, &mut gbuf, rows, Dtype::F16, Dtype::F32, &mut ws, &st).unwrap();
        glm_p2::linear(uw, &x, &mut ubuf, rows, Dtype::F16, Dtype::F32, &mut ws, &st).unwrap();
        st.sync().unwrap();
        let (gv, uv) = (gbuf.to_vec::<f32>().unwrap(), ubuf.to_vec::<f32>().unwrap());
        let a: Vec<u16> = gv
            .iter()
            .zip(&uv)
            .map(|(&g, &u)| {
                // moe_experts.cu's swiglu_limit_kernel: the activated gate is clamped from above, the up to [-limit, limit]
                let x = (g / (1.0 + (-g).exp())).min(limit);
                let u = u.clamp(-limit, limit);
                f64_to_f16((x * u) as f64)
            })
            .collect();
        let mut out = DevBuf::alloc(rows * h * 4).unwrap();
        glm_p2::linear(dw, &upload(&a), &mut out, rows, Dtype::F16, Dtype::F32, &mut ws, &st).unwrap();
        st.sync().unwrap();
        let hand = out.to_vec::<f32>().unwrap();
        // the host's expf may differ from the device's in the last bit of an activation: compare closely, not bit for bit
        let (mut ee, mut rr) = (0f64, 0f64);
        for (x, y) in hand.iter().zip(&whole) {
            ee += ((x - y) as f64).powi(2);
            rr += (*y as f64).powi(2);
        }
        let rel = (ee / rr).sqrt();
        eprintln!("{name:13} inter {inter:5} {bits}b limit {limit}: pieces 256/128/100/17 bit-identical; hand-composed rel rms {rel:.2e}");
        assert!(rel < 2e-3, "{name}: the gated MLP is not gate -> up -> swiglu -> down (rel rms {rel:e})");
    }
}

/// The gate, up or down linear of a gated MLP (read back through the crate's public constructor order).
fn mlp_part(m: &SharedExpert, i: usize) -> &Exl3Weights {
    match i {
        0 => m.gate_weights(),
        1 => m.up_weights(),
        _ => m.down_weights(),
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore]
fn bench_p2_against_the_large_row_kernel_on_the_glm_shapes() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut flush = DevBuf::alloc(256 << 20).unwrap();
    for (i, &(name, k, n, bits)) in SHAPES.iter().enumerate() {
        let (w, _) = layer(k, n, bits, 5000 + i as u64);
        for rows in [512usize, 2048] {
            let (xb, _) = activations(3 + i as u64, rows, k, 1.0);
            let x = upload(&xb);
            let mut y = DevBuf::alloc(rows * n * 4).unwrap();
            let mut ws = DevBuf::alloc(glm_p2::workspace_bytes(rows, k).max(w.big_workspace_bytes(rows))).unwrap();
            // what: None the large-row kernel (P1), Some((variant, tile)) P2
            let mut time = |what: Option<(Variant, i32)>| -> f64 {
                let mut v = Vec::new();
                for it in 0..8 {
                    flush.memset_async(it as u8, &st).unwrap();
                    st.sync().unwrap();
                    let t = std::time::Instant::now();
                    match what {
                        Some((var, tile)) => {
                            glm_p2::linear_variant(&w, &x, &mut y, rows, Dtype::F16, Dtype::F32, &mut ws, var, tile, &st).unwrap()
                        }
                        None => w.forward_big(&x, &mut y, rows, Dtype::F16, Dtype::F32, &mut ws, &st).unwrap(),
                    }
                    st.sync().unwrap();
                    if it > 0 {
                        v.push(t.elapsed().as_secs_f64() * 1e3);
                    }
                }
                median(v)
            };
            let tf = 2.0 * rows as f64 * k as f64 * n as f64 / 1e9;
            let tb = time(None);
            let mut line = format!("BENCH {name:15} K {k:5} N {n:5} {bits}b rows {rows:5}: big {tb:7.3} ms {:5.1} TF", tf / tb);
            for var in VARIANTS {
                let pick = glm_p2::pick_tile(rows, k, n, var);
                for tile in [0, 641, 642, 1281, 1282] {
                    if var == Variant::Fold128 && tile == 1282 {
                        continue;
                    }
                    let tp = time(Some((var, tile)));
                    let tag = if tile == 0 { format!("pick{pick}") } else { format!("{tile}") };
                    line += &format!(
                        " | {} {tag} {tp:.3} ms {:5.1} TF x{:.2}",
                        if var == Variant::Chain { "chain" } else { "fold" },
                        tf / tp,
                        tb / tp
                    );
                }
            }
            eprintln!("{line}");
        }
    }
}

/// (name, M, K, N, bits, launches in one 4,096-row prompt chunk on a rank): the P2 dense GEMMs a TP=2 RED-SNOW 4.91 bpw prefill chunk
/// runs under the sequence-parallel row split (2,048 own rows a rank; kv_a over all 4,096), at each layer's most common width in the
/// pack (KDA qkv the rank's strip cut 4096 x 12288 at 6 bits in 29 of 34 layers, 7 and 8 in the rest; shared gate / up 8 bits in 29 of
/// 42, 7 in 13; shared down 6 in 27, 7 or 8 in 15).
const SERVING: &[(&str, usize, usize, usize, usize, usize)] = &[
    ("kda qkv strips", 2048, 4096, 12288, 6, 68),
    ("kda o", 2048, 8192, 4096, 6, 34),
    ("shared gate/up", 2048, 4096, 2048, 8, 82),
    ("shared down", 2048, 2048, 4096, 6, 41),
    ("mla q_a", 2048, 4096, 1536, 6, 11),
    ("mla q_b", 2048, 1536, 16384, 8, 11),
    ("mla kv_a", 4096, 4096, 512, 6, 11),
    ("mla o", 2048, 16384, 4096, 6, 11),
    ("dense gate/up", 2048, 4096, 12288, 8, 6),
    ("dense down", 2048, 12288, 4096, 6, 3),
];

/// The serving shapes ([`SERVING`]) in the serving K order (fold128), cold (a 256 MiB memset before each call), the median of
/// `REXL3_P2_BENCH_ROUNDS` (default 9) CUDA-event timed calls of the whole linear (the input rotation, a few % of the GEMM, then the
/// GEMM), the tiles interleaved round by round (GB10's clock drifts by a few % between runs); `REXL3_P2_BENCH_TILES` (comma separated,
/// default `0`: the pick) the tiles, each checked bit for bit against the pick, `REXL3_P2_BENCH_ONLY` a name filter. Last, the chunk's
/// sum over the launch counts at the pick.
#[test]
#[ignore]
fn bench_p2_on_the_serving_shapes() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut flush = DevBuf::alloc(256 << 20).unwrap();
    let tiles: Vec<i32> = std::env::var("REXL3_P2_BENCH_TILES")
        .unwrap_or_else(|_| "0".into())
        .split(',')
        .map(|t| t.trim().parse().expect("REXL3_P2_BENCH_TILES"))
        .collect();
    let only = std::env::var("REXL3_P2_BENCH_ONLY").ok();
    let events = gb_cuda::moe::EventTimer::new(2).unwrap();
    let rounds: usize = std::env::var("REXL3_P2_BENCH_ROUNDS").map_or(9, |r| r.parse().expect("REXL3_P2_BENCH_ROUNDS"));
    let mut chunk = 0f64;
    for (i, &(name, rows, k, n, bits, launches)) in SERVING.iter().enumerate() {
        if only.as_deref().is_some_and(|o| !name.contains(o)) {
            continue;
        }
        let (w, _) = layer(k, n, bits, 7000 + i as u64);
        let (xb, _) = activations(70 + i as u64, rows, k, 1.0);
        let x = upload(&xb);
        let mut y = DevBuf::alloc(rows * n * 4).unwrap();
        let mut ws = DevBuf::alloc(glm_p2::workspace_bytes(rows, k)).unwrap();
        let tf = 2.0 * rows as f64 * k as f64 * n as f64 / 1e9;
        let pick = glm_p2::pick_tile(rows, k, n, Variant::Fold128);
        let mut line = format!("SERVE {name:15} M {rows:5} K {k:5} N {n:5} {bits}b x{launches:3}");
        let want = p2v(&w, &x, rows, Variant::Fold128, 0, &st);
        let mut same = Vec::new();
        for &tile in &tiles {
            glm_p2::linear_variant(&w, &x, &mut y, rows, Dtype::F16, Dtype::F32, &mut ws, Variant::Fold128, tile, &st).unwrap();
            same.push(y.to_vec::<f32>().unwrap().iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        let mut t = vec![Vec::new(); tiles.len()];
        for r in 0..rounds {
            for (ti, &tile) in tiles.iter().enumerate() {
                flush.memset_async((r * tiles.len() + ti) as u8, &st).unwrap();
                events.record(0, &st).unwrap();
                glm_p2::linear_variant(&w, &x, &mut y, rows, Dtype::F16, Dtype::F32, &mut ws, Variant::Fold128, tile, &st).unwrap();
                events.record(1, &st).unwrap();
                t[ti].push(events.ms(0, 1).unwrap() as f64);
            }
        }
        for (ti, &tile) in tiles.iter().enumerate() {
            let ms = median(t[ti].clone());
            if tile == 0 {
                chunk += ms * launches as f64;
            }
            let tag = if tile == 0 { format!("pick{pick}") } else { format!("{tile}") };
            line += &format!(" | {tag} {ms:.3} ms {:5.1} TF{}", tf / ms, if same[ti] { "" } else { " BITS DIFFER" });
        }
        eprintln!("{line}");
    }
    eprintln!("SERVE chunk sum at the pick: {chunk:.1} ms");
}

/// The golden of GLM's P2 dense GEMM (`testdata/gpu/golden_glm_p2_dense.txt`, scripts/gpu_gate.sh): hashes of its outputs in both K orders
/// on synthetic layers of the GLM shapes, at widths that are and are not multiples of a tile, through the default tile and a forced one
/// (the same bits), and of the gated MLP; committed so that a change that moves a P2 bit shows up in the gate (and the arithmetic id
/// `glm_p2::ARITH_ID` must then change with it: the prefix namespace and the TP Hello carry it).
#[test]
#[ignore]
fn golden_glm_p2_hashes() {
    gpu!();
    let path = std::env::var("REXL3_GOLDEN_OUT").expect("REXL3_GOLDEN_OUT");
    let st = Stream::new().unwrap();
    let hash = |v: &[f32]| gb_exl3::fnv1a(v.iter().flat_map(|x| x.to_bits().to_le_bytes()));
    let mut lines = vec![format!("arith {} default {:?}", glm_p2::ARITH_ID, glm_p2::DEFAULT_VARIANT)];
    let shapes =
        [(4096usize, 24576usize, 4usize), (8192, 4096, 3), (4096, 1536, 4), (1536, 16384, 4), (4096, 512, 5), (16384, 4096, 4)];
    for (i, &(k, n, bits)) in shapes.iter().enumerate() {
        let (w, _) = layer(k, n, bits, 900 + i as u64);
        for rows in [1usize, 129, 300] {
            let (xb, _) = activations(910 + i as u64 + rows as u64, rows, k, 1.0);
            let x = upload(&xb);
            for v in VARIANTS {
                let y = p2v(&w, &x, rows, v, 0, &st);
                assert!(
                    y.iter().zip(&p2v(&w, &x, rows, v, 1281, &st)).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{k}x{n}: the tile moved a bit"
                );
                lines.push(format!("p2 {} {k}x{n} {bits}-bit rows={rows} {:016x}", v.id(), hash(&y)));
            }
        }
    }
    let m = mlp(4096, 2048, 3, 7.0, 950);
    let (xb, _) = activations(951, 300, 4096, 2.0);
    let x = upload(&xb);
    let mut out = DevBuf::zeroed_view(300 * 4096 * 4).unwrap();
    let mut sc = SharedScratchBig::new(&m, 300).unwrap();
    glm_p2::gated(&m, &x, Dtype::F16, 300, &mut out, sc.p2_bufs(), &st).unwrap();
    st.sync().unwrap();
    lines.push(format!("p2 gated 4096x2048x4096 3-bit limit=7 rows=300 {:016x}", hash(&out.to_vec::<f32>().unwrap())));
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    eprintln!("wrote {} golden lines to {path}", lines.len());
}
