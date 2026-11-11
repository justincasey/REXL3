//! Flash-Next's Tiled-prefill dense EXL3 linears (`glm_p2::linear_variant(.., Variant::Chain, ..)`) on synthetic Mul1 layers of the
//! Flash-Next pack's shapes (6-bit trellis; the GDN in_proj_qkv / in_proj_z / out_proj and the full-attention q / k / v / o and the
//! indexer's qk projection), bf16 in and out as the model calls it.
//!
//! Gates (not ignored): the chain GEMM the 3- to 8-bit layers run by default and every one of its configurations (`tile_force`
//! 10001 ..= 10004) give, bit for bit, what the old fixed-tile kernel gives (`tile_force` 642, 1282: the `gemm_kernel` the Tiled
//! prefill ran before), on the pack's shapes and on odd strip counts, at row counts that straddle the 64 / 128-row tiles (1, 7, 33,
//! 127, 128, 129, 1000, 4,096), on random trellis and scales and on adversarial activations (zero rows, rows that overflow fp16 after
//! the rotation, denormals, NaN and infinity), with and without a bias, for bf16, fp16 and fp32 outputs.
//!
//! * `bench_*` (ignored): ms and TFLOP/s per call at 4,096 rows (a steady chunk), weights from DRAM, the old kernel's tiles beside the
//!   new configurations.
//!
//!     cargo test -p gb-cuda --release --test fn_dense_chain -- --test-threads=1 --nocapture
//!     cargo test -p gb-cuda --release --test fn_dense_chain -- --ignored --test-threads=1 --nocapture bench
#![cfg(not(gb_no_cuda))]
mod common;

use common::exl3_synth::synth_group;
use common::Rng;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::glm_p2::{self, Variant};
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

/// (name, K, N, bits, calls per steady chunk): every dense linear a Flash-Next prefill chunk runs through the tiled GEMM.
const SHAPES: &[(&str, usize, usize, usize, usize)] = &[
    ("gdn qkv", 2560, 10240, 6, 45),
    ("gdn z", 2560, 6144, 6, 45),
    ("gdn out", 6144, 2560, 6, 45),
    ("attn q", 2560, 12288, 6, 15),
    ("attn k", 2560, 512, 6, 15),
    ("attn v", 2560, 512, 6, 15),
    ("attn o", 6144, 2560, 6, 15),
    ("indexer qk", 2560, 640, 6, 12),
    ("gdn qkv 8b", 2560, 10240, 8, 1),
    ("gdn out 8b", 6144, 2560, 8, 5),
];

fn layer(k: usize, n: usize, bits: usize, seed: u64) -> Exl3Weights {
    let g = synth_group(k, n, bits, Codebook::Mul1, seed);
    Exl3Weights::from_loaded(&g).unwrap()
}

/// A layer with an fp16-exact bias.
fn layer_bias(k: usize, n: usize, bits: usize, seed: u64) -> Exl3Weights {
    let mut g: LoadedGroup = synth_group(k, n, bits, Codebook::Mul1, seed);
    let mut r = Rng(seed ^ 0xb1a5);
    g.group.bias = true;
    g.bias = Some((0..n).map(|_| f16_to_f64(f64_to_f16(r.uniform(-2.0, 2.0) as f64))).collect());
    Exl3Weights::from_loaded(&g).unwrap()
}

fn bf16_bits(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7fc0;
    }
    let b = v.to_bits();
    if v.is_infinite() {
        return (b >> 16) as u16;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// Activations as bf16 or fp16 bits, row r of a kind that depends on r: 0 normal with heavy tails, 1 zeros, 2 large (the rotation
/// overflows fp16), 3 tiny, 4 denormal bit patterns, 5 small with one huge entry, 6 one NaN, 7 one infinity, 8 sparse spikes.
fn acts(seed: u64, rows: usize, k: usize, f16: bool) -> Vec<u16> {
    let mut r = Rng(seed);
    let enc = |v: f64| if f16 { f64_to_f16(v) } else { bf16_bits(v as f32) };
    let mut out = Vec::with_capacity(rows * k);
    for row in 0..rows {
        let kind = if row < 3 { 0 } else { (row * 7 + seed as usize) % 9 };
        let spot = (r.next_u64() as usize) % k;
        for c in 0..k {
            let u = r.uniform(-1.0, 1.0) as f64;
            let bits = match kind {
                0 => enc(if r.next_u64() % 32 == 0 { 4.0 * u } else { u }),
                1 => enc(0.0),
                2 => enc(if f16 { 6.0e3 * u } else { 3.0e4 * u }),
                3 => enc(u * if f16 { 1.0e-7 } else { 1.0e-30 }),
                4 => (r.next_u64() as u16 % if f16 { 0x3ff } else { 0x7f }) | if u < 0.0 { 0x8000 } else { 0 },
                5 => enc(if c == spot {
                    if f16 {
                        6.0e4
                    } else {
                        1.0e38
                    }
                } else {
                    1.0e-2 * u
                }),
                6 => {
                    if c == spot {
                        if f16 {
                            0x7e00
                        } else {
                            0x7fc0
                        }
                    } else {
                        enc(u)
                    }
                }
                7 => {
                    if c == spot {
                        if f16 {
                            0x7c00
                        } else {
                            0x7f80
                        }
                    } else {
                        enc(u)
                    }
                }
                _ => enc(if r.next_u64() % 64 == 0 { 64.0 * u.signum() } else { 0.0 }),
            };
            out.push(bits);
        }
    }
    out
}

fn upload(x: &[u16]) -> DevBuf {
    let mut d = DevBuf::zeroed_view(x.len() * 2).unwrap();
    d.upload(x).unwrap();
    d
}

/// The output bytes of `w` on `rows` rows of x at a tile (0: the pick) after filling y with a sentinel.
fn run(w: &Exl3Weights, x: &DevBuf, rows: usize, xd: Dtype, yd: Dtype, tile: i32, st: &Stream) -> Vec<u8> {
    let mut y = DevBuf::alloc(rows * w.n * yd.size()).unwrap();
    y.memset_async(0xa5, st).unwrap();
    let mut ws = DevBuf::alloc(glm_p2::workspace_bytes(rows, w.k)).unwrap();
    glm_p2::linear_variant(w, x, &mut y, rows, xd, yd, &mut ws, Variant::Chain, tile, st).unwrap();
    st.sync().unwrap();
    y.to_vec::<u8>().unwrap()
}

const ROWS: [usize; 8] = [1, 7, 33, 127, 128, 129, 1000, 4096];
/// The old kernel's tiles (the references) and the chain GEMM's pick and configurations.
const OLD: [i32; 2] = [642, 1282];
const NEW: [i32; 5] = [0, 10001, 10002, 10003, 10004];

/// `what` of rows `rows` of `w` equal between every old tile and every new configuration.
#[allow(clippy::too_many_arguments)]
fn check(name: &str, w: &Exl3Weights, rows: usize, seed: u64, xd: Dtype, yd: Dtype, st: &Stream, old: &[i32]) {
    let x = upload(&acts(seed, rows, w.k, xd == Dtype::F16));
    let want = run(w, &x, rows, xd, yd, old[0], st);
    for &o in &old[1..] {
        assert!(run(w, &x, rows, xd, yd, o, st) == want, "{name} rows {rows}: old tile {o} != old tile {}", old[0]);
    }
    for &t in &NEW {
        let got = run(w, &x, rows, xd, yd, t, st);
        if got != want {
            let bad = got.iter().zip(&want).position(|(a, b)| a != b).unwrap();
            panic!(
                "{name} ({} bit, K {}, N {}) rows {rows}, tile {t}: {} of {} bytes differ, first at byte {bad} (row {})",
                w.bits,
                w.k,
                w.n,
                got.iter().zip(&want).filter(|(a, b)| a != b).count(),
                want.len(),
                bad / (w.n * yd.size())
            );
        }
    }
}

#[test]
fn chain_gemm_equals_the_old_kernel_on_the_flash_next_shapes() {
    gpu!();
    let st = Stream::new().unwrap();
    for (i, &(name, k, n, bits, _)) in SHAPES.iter().chain(&[("gdn z 7b", 2560, 6144, 7, 0)]).enumerate() {
        let w = layer(k, n, bits, 700 + i as u64);
        for &rows in &ROWS {
            // the 4,096-row case against both old tiles, the rest against the 64 x 2
            let old: &[i32] = if rows == 4096 { &OLD } else { &OLD[..1] };
            check(name, &w, rows, 11 * i as u64 + rows as u64, Dtype::BF16, Dtype::BF16, &st, old);
        }
        eprintln!("exact: {name} ({k} x {n}, {bits} bit) at {ROWS:?} rows");
    }
}

#[test]
fn chain_gemm_equals_the_old_kernel_with_a_bias_and_any_dtype() {
    gpu!();
    let st = Stream::new().unwrap();
    for (i, &(k, n, bits)) in [(2560usize, 640usize, 6usize), (6144, 2560, 6), (2560, 6144, 8)].iter().enumerate() {
        let w = layer_bias(k, n, bits, 900 + i as u64);
        for &rows in &[1usize, 129, 1000] {
            for (xd, yd) in [(Dtype::BF16, Dtype::BF16), (Dtype::BF16, Dtype::F32), (Dtype::F16, Dtype::F16), (Dtype::F16, Dtype::F32)] {
                check("bias", &w, rows, 31 * i as u64 + rows as u64, xd, yd, &st, &OLD[..1]);
            }
        }
    }
}

#[test]
fn chain_gemm_equals_the_old_kernel_on_odd_strip_counts_and_every_bit_width() {
    gpu!();
    let st = Stream::new().unwrap();
    // N / 128 = 1, 3, 5, 7, 5 strips: the 256-column tiles of the old kernel and the 128-column tiles of the new one end mid-tile
    for (i, &(k, n, bits)) in [
        (128usize, 128usize, 6usize),
        (384, 640, 6),
        (640, 384, 7),
        (2560, 640, 8),
        (1536, 896, 6),
        (256, 640, 3),
        (512, 384, 4),
        (768, 256, 5),
    ]
    .iter()
    .enumerate()
    {
        let w = layer(k, n, bits, 1100 + i as u64);
        for &rows in &[1usize, 33, 129, 300] {
            check("odd", &w, rows, 17 * i as u64 + rows as u64, Dtype::BF16, Dtype::BF16, &st, &OLD[..1]);
        }
    }
}

/// The comparison has teeth: the fold128 K order (a different accumulation order on a long K) differs from the chain in some bit of
/// the same layer, and a row's bytes are NaN where the adversarial rows make them so.
#[test]
fn the_comparison_sees_a_changed_accumulation_order() {
    gpu!();
    let st = Stream::new().unwrap();
    let w = layer(6144, 2560, 6, 77);
    let rows = 129;
    let x = upload(&acts(5, rows, w.k, false));
    let want = run(&w, &x, rows, Dtype::BF16, Dtype::F32, 642, &st);
    let mut y = DevBuf::alloc(rows * w.n * 4).unwrap();
    let mut ws = DevBuf::alloc(glm_p2::workspace_bytes(rows, w.k)).unwrap();
    glm_p2::linear_variant(&w, &x, &mut y, rows, Dtype::BF16, Dtype::F32, &mut ws, Variant::Fold128, 0, &st).unwrap();
    st.sync().unwrap();
    let fold = y.to_vec::<u8>().unwrap();
    let differ = fold.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert!(differ > 1000, "fold128 and the chain agree on all but {differ} bytes: the comparison would not notice a reordered chain");
    let f: Vec<f32> = want.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let nan_rows = (0..rows).filter(|&r| f[r * w.n..(r + 1) * w.n].iter().any(|v| v.is_nan())).count();
    assert!(nan_rows >= 10 && nan_rows < rows, "{nan_rows} NaN rows of {rows}");
    eprintln!("fold128 differs in {differ} bytes; {nan_rows} of {rows} rows hold NaN");
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore]
fn bench_dense_chain_on_the_flash_next_shapes() {
    gpu!();
    let rows = 4096usize;
    let st = Stream::new().unwrap();
    let tiles: [i32; 6] = [642, 1282, 10001, 10002, 10003, 10004];
    let (mut total, mut total_flop) = ([0.0; 6], 0.0);
    for (i, &(name, k, n, bits, calls)) in SHAPES.iter().enumerate() {
        // four distinct layers back to back: the weights come from DRAM, as in a chunk, and no host gap separates the launches
        let ws_: Vec<Exl3Weights> = (0..4).map(|c| layer(k, n, bits, 5000 + 10 * i as u64 + c)).collect();
        let x = upload(&acts(3 + i as u64, rows, k, false));
        let mut y = DevBuf::alloc(rows * n * 2).unwrap();
        let mut ws = DevBuf::alloc(glm_p2::workspace_bytes(rows, k)).unwrap();
        let tf = 2.0 * rows as f64 * k as f64 * n as f64 / 1e9;
        let mut line = format!("BENCH {name:11} K {k:5} N {n:5} {bits}b rows {rows}:");
        for (ti, &tile) in tiles.iter().enumerate() {
            let mut v = Vec::new();
            for it in 0..6 {
                st.sync().unwrap();
                let t = std::time::Instant::now();
                for r in 0..8 {
                    let w = &ws_[r % 4];
                    glm_p2::linear_variant(w, &x, &mut y, rows, Dtype::BF16, Dtype::BF16, &mut ws, Variant::Chain, tile, &st).unwrap();
                }
                st.sync().unwrap();
                if it > 0 {
                    v.push(t.elapsed().as_secs_f64() * 1e3 / 8.0);
                }
            }
            let ms = median(v);
            total[ti] += ms * calls as f64;
            line += &format!(" | {tile} {ms:6.3} ms {:5.1} TF", tf / ms);
        }
        total_flop += tf * calls as f64;
        eprintln!("{line}  (x{calls} a chunk, pick {})", glm_p2::pick_tile(rows, k, n, Variant::Chain));
    }
    for (ti, &tile) in tiles.iter().enumerate() {
        eprintln!(
            "BENCH chunk total tile {tile}: {:.1} ms for {:.2} TFLOP = {:.1} TF/s",
            total[ti],
            total_flop / 1e3,
            total_flop / total[ti]
        );
    }
}

/// Where a CTA's time goes: the same N at K = 128 (two k stages: all fill and epilogue) and growing K.
#[test]
#[ignore]
fn bench_dense_chain_fill_and_epilogue() {
    gpu!();
    let rows = 4096usize;
    let st = Stream::new().unwrap();
    for &(k, n) in &[(128usize, 10240usize), (256, 10240), (640, 10240), (1280, 10240), (2560, 10240), (5120, 10240)] {
        let ws_: Vec<Exl3Weights> = (0..4).map(|c| layer(k, n, 6, 6000 + c)).collect();
        let x = upload(&acts(3, rows, k, false));
        let mut y = DevBuf::alloc(rows * n * 2).unwrap();
        let mut ws = DevBuf::alloc(glm_p2::workspace_bytes(rows, k)).unwrap();
        let mut v = Vec::new();
        for it in 0..6 {
            st.sync().unwrap();
            let t = std::time::Instant::now();
            for r in 0..8 {
                glm_p2::linear_variant(&ws_[r % 4], &x, &mut y, rows, Dtype::BF16, Dtype::BF16, &mut ws, Variant::Chain, 10001, &st)
                    .unwrap();
            }
            st.sync().unwrap();
            if it > 0 {
                v.push(t.elapsed().as_secs_f64() * 1e3 / 8.0);
            }
        }
        let ms = median(v);
        eprintln!("FILL K {k:5} N {n}: {ms:.3} ms = {:.1} TF", 2.0 * rows as f64 * k as f64 * n as f64 / 1e9 / ms);
    }
}

/// The chain GEMM at the narrower bit widths: the old tiles beside the new configuration 1 (which the pick takes from 3 bits up).
#[test]
#[ignore]
fn bench_dense_chain_narrow_bit_widths() {
    gpu!();
    let rows = 4096usize;
    let st = Stream::new().unwrap();
    let tiles: [i32; 3] = [642, 1282, 10001];
    for (i, &(k, n, bits)) in [
        (2560usize, 10240usize, 5usize),
        (2560, 10240, 4),
        (2560, 10240, 3),
        (6144, 2560, 4),
        (5120, 17408, 4),
        (17408, 5120, 4),
        (5120, 5120, 4),
        (2560, 640, 5),
    ]
    .iter()
    .enumerate()
    {
        let ws_: Vec<Exl3Weights> = (0..4).map(|c| layer(k, n, bits, 8000 + 10 * i as u64 + c)).collect();
        let x = upload(&acts(3 + i as u64, rows, k, false));
        let mut y = DevBuf::alloc(rows * n * 2).unwrap();
        let mut ws = DevBuf::alloc(glm_p2::workspace_bytes(rows, k)).unwrap();
        let tf = 2.0 * rows as f64 * k as f64 * n as f64 / 1e9;
        let mut line = format!("NARROW K {k:5} N {n:5} {bits}b:");
        for &tile in &tiles {
            let mut v = Vec::new();
            for it in 0..6 {
                st.sync().unwrap();
                let t = std::time::Instant::now();
                for r in 0..8 {
                    glm_p2::linear_variant(&ws_[r % 4], &x, &mut y, rows, Dtype::BF16, Dtype::BF16, &mut ws, Variant::Chain, tile, &st)
                        .unwrap();
                }
                st.sync().unwrap();
                if it > 0 {
                    v.push(t.elapsed().as_secs_f64() * 1e3 / 8.0);
                }
            }
            let ms = median(v);
            line += &format!(" | {tile} {ms:6.3} ms {:5.1} TF", tf / ms);
        }
        eprintln!("{line}");
    }
}
