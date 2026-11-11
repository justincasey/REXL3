//! The large-row EXL3 linear (`csrc/exl3_big.cu`, `Exl3Weights::forward_big`): every row equals the decode path's output bit for bit, at every
//! row count, on the shapes of the GLM text model, every width and codebook the kernels take, with and without bias, for every activation type.
//!
//!     cargo test --release -p gb-cuda --test exl3_big -- --test-threads=1
//!     cargo test --release -p gb-cuda --test exl3_big -- --ignored --nocapture bench_      # speed against the decode path in 128-row calls
#![cfg(not(gb_no_cuda))]
mod common;

use common::*;
use gb_cuda::exl3_big::big_plan;
use gb_cuda::exl3_linear::{linear_workspace_bytes, Dtype, Exl3Weights};
use gb_cuda::moe::EventTimer;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::checkpoint::{Group, LoadedGroup};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_exl3::{Bits, Codebook};

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: not run");
            return;
        }
    };
}

fn normal(r: &mut Rng) -> f64 {
    let (a, b) = ((r.next_u64() >> 11) as f64 / (1u64 << 53) as f64, (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64);
    (-2.0 * a.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
}

fn scale_vec(r: &mut Rng, n: usize, mag: f64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            f16_to_f64(f64_to_f16(
                (if r.next_u64() & 1 == 1 { -1.0 } else { 1.0 }) * mag * (0.5 + (r.next_u64() >> 40) as f64 / 16777216.0),
            ))
        })
        .collect()
}

fn synth(k: usize, n: usize, bits: usize, cb: Codebook, bias: bool, seed: u64) -> Exl3Weights {
    let mut r = Rng(seed);
    let words = (k / 16) * (n / 16) * 16 * bits;
    let trellis: Vec<i16> = (0..words).map(|_| r.next_u64() as i16).collect();
    let group = Group {
        prefix: "synthetic".into(),
        bits: Bits::from_twice(2 * bits as u8).unwrap(),
        codebook: cb,
        k,
        n,
        in_scales: "suh",
        out_scales: "svh",
        bias,
        files: vec![],
        bytes: 0,
    };
    let svh = scale_vec(&mut r, n, 1.0 / (k as f64).sqrt());
    let b = bias.then(|| scale_vec(&mut r, n, 0.5));
    let g = LoadedGroup { group, trellis, kt: k / 16, nt: n / 16, suh: scale_vec(&mut r, k, 1.0), svh, bias: b };
    Exl3Weights::from_loaded(&g).unwrap()
}

fn input(r: &mut Rng, rows: usize, k: usize, dt: Dtype) -> Vec<u8> {
    let vals: Vec<f64> = (0..rows * k).map(|_| normal(r)).collect();
    match dt {
        Dtype::F16 => vals.iter().flat_map(|&v| f64_to_f16(v).to_le_bytes()).collect(),
        Dtype::F32 => vals.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect(),
        Dtype::BF16 => vals
            .iter()
            .flat_map(|&v| {
                let bits = (v as f32).to_bits();
                (((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16).to_le_bytes()
            })
            .collect(),
    }
}

fn dl(b: &DevBuf, st: &Stream) -> Vec<u8> {
    st.sync().unwrap();
    b.to_vec().unwrap()
}

/// The decode path on rows cut into windows of at most 128.
fn decode_rows(w: &Exl3Weights, st: &Stream, x: &[u8], rows: usize, xd: Dtype, yd: Dtype) -> Vec<u8> {
    let mut ws = DevBuf::zeroed(w.workspace_bytes(128).max(16)).unwrap();
    let mut out = Vec::new();
    let mut at = 0;
    while at < rows {
        let n = 128.min(rows - at);
        let xb = DevBuf::from_slice(&x[at * w.k * xd.size()..(at + n) * w.k * xd.size()]).unwrap();
        let mut y = DevBuf::zeroed(n * w.n * yd.size()).unwrap();
        w.forward(&xb, &mut y, n, xd, yd, &mut ws, st).unwrap();
        out.extend(dl(&y, st));
        at += n;
    }
    out
}

fn big_rows(w: &Exl3Weights, st: &Stream, x: &[u8], rows: usize, xd: Dtype, yd: Dtype) -> Vec<u8> {
    let mut ws = DevBuf::zeroed(w.big_workspace_bytes(rows).max(16)).unwrap();
    // a poisoned workspace: contents on entry must not matter
    ws.memset_async(0xCD, st).unwrap();
    let xb = DevBuf::from_slice(x).unwrap();
    let mut y = DevBuf::zeroed(rows * w.n * yd.size()).unwrap();
    w.forward_big(&xb, &mut y, rows, xd, yd, &mut ws, st).unwrap();
    dl(&y, st)
}

fn same(what: &str, got: &[u8], want: &[u8], n: usize, ysz: usize) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        let row = i / (n * ysz);
        panic!("{what}: row {row} byte {} differs ({} vs {})", i - row * n * ysz, got[i], want[i]);
    }
}

#[test]
fn big_linear_equals_the_decode_path_on_the_glm_shapes() {
    gpu!();
    let st = Stream::new().unwrap();
    // (K, N, bits): every distinct shape of the GLM text model's EXL3 linears (the plans differ: split K, 4 or 8 warps)
    let shapes: [(usize, usize, usize); 10] = [
        (4096, 24576, 2),
        (8192, 4096, 3),
        (4096, 1536, 2),
        (1536, 16384, 4),
        (4096, 512, 2),
        (4096, 2048, 3),
        (2048, 4096, 2),
        (4096, 12288, 3),
        (12288, 4096, 4),
        (4096, 4096, 2),
    ];
    let rows_list = [1usize, 17, 64, 65, 128, 129, 200, 257];
    let mut total = 0usize;
    for (i, &(k, n, bits)) in shapes.iter().enumerate() {
        let (sk, wk) = big_plan(k, n).unwrap();
        let w = synth(k, n, bits, Codebook::Mul1, false, 10 + i as u64);
        let mut r = Rng(20 + i as u64);
        for &rows in &rows_list {
            let x = input(&mut r, rows, k, Dtype::F16);
            let want = decode_rows(&w, &st, &x, rows, Dtype::F16, Dtype::F32);
            let got = big_rows(&w, &st, &x, rows, Dtype::F16, Dtype::F32);
            same(&format!("{k}x{n} {bits}-bit (sk {sk}, wk {wk}), {rows} rows"), &got, &want, n, 4);
            total += rows;
        }
    }
    eprintln!("{total} rows of 10 GLM shapes bit-identical to the decode path");
}

/// Every width and codebook, with a bias, for the three input and output types.
#[test]
fn big_linear_equals_the_decode_path_for_every_width_codebook_and_dtype() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut n_cases = 0;
    for bits in 1..=8usize {
        for cb in [Codebook::Inst3, Codebook::Mcg, Codebook::Mul1] {
            let (k, n) = (512, 384);
            let w = synth(k, n, bits, cb, bits % 2 == 0, 100 + bits as u64);
            let mut r = Rng(200 + bits as u64);
            for (xd, yd) in [(Dtype::F16, Dtype::F16), (Dtype::BF16, Dtype::F32), (Dtype::F32, Dtype::BF16)] {
                for rows in [3usize, 64, 100, 130, 300] {
                    let x = input(&mut r, rows, k, xd);
                    let want = decode_rows(&w, &st, &x, rows, xd, yd);
                    let got = big_rows(&w, &st, &x, rows, xd, yd);
                    same(&format!("{bits}-bit {cb:?} {xd:?}->{yd:?} {rows} rows"), &got, &want, n, yd.size());
                    n_cases += 1;
                }
            }
        }
    }
    eprintln!("{n_cases} cases bit-identical");
}

/// Row r of a big call equals the same row in any other call: the chunk grid and the position in the call change nothing.
#[test]
fn big_linear_rows_do_not_depend_on_the_call() {
    gpu!();
    let st = Stream::new().unwrap();
    let (k, n) = (4096, 1536);
    let w = synth(k, n, 2, Codebook::Mul1, false, 300);
    let mut r = Rng(301);
    let rows = 1000;
    let x = input(&mut r, rows, k, Dtype::F16);
    let whole = big_rows(&w, &st, &x, rows, Dtype::F16, Dtype::F32);
    let row_bytes = n * 4;
    for cuts in [vec![1usize, 999], vec![63, 937], vec![128, 128, 744], vec![500, 500], vec![129, 129, 129, 613]] {
        let mut at = 0;
        let mut out = Vec::new();
        for c in &cuts {
            out.extend(big_rows(&w, &st, &x[at * k * 2..(at + c) * k * 2], *c, Dtype::F16, Dtype::F32));
            at += c;
        }
        same(&format!("cuts {cuts:?}"), &out, &whole, n, 4);
    }
    let _ = row_bytes;
}

/// The two block orders of the large-row kernel (the strips of a row group together for a layer of at most 24 strips, the row groups of a strip
/// together for a wider one) both give the decode path's bits at wide row counts: the Flash-Next shared expert's shapes (2,560 -> 640 and
/// 640 -> 2,560), a layer of 32 strips, row counts that leave a ragged last 64-row group and 8,192 rows (128 groups).
#[test]
fn big_linear_equals_the_decode_path_at_wide_row_counts_in_both_block_orders() {
    gpu!();
    let st = Stream::new().unwrap();
    // (K, N, bits): 5, 5, 20, 20 and 32 strips
    let shapes: [(usize, usize, usize); 5] = [(2560, 640, 4), (2560, 640, 6), (640, 2560, 4), (640, 2560, 8), (1024, 4096, 3)];
    let mut total = 0usize;
    for (i, &(k, n, bits)) in shapes.iter().enumerate() {
        let (sk, wk) = big_plan(k, n).unwrap();
        let w = synth(k, n, bits, Codebook::Mul1, i == 1, 400 + i as u64);
        let mut r = Rng(500 + i as u64);
        for rows in [4097usize, 5003, 8192] {
            let x = input(&mut r, rows, k, Dtype::BF16);
            let want = decode_rows(&w, &st, &x, rows, Dtype::BF16, Dtype::F32);
            let got = big_rows(&w, &st, &x, rows, Dtype::BF16, Dtype::F32);
            same(&format!("{k}x{n} {bits}-bit (sk {sk}, wk {wk}), {rows} rows"), &got, &want, n, 4);
            total += rows;
        }
    }
    eprintln!("{total} rows of 5 wide-call cases bit-identical to the decode path");
}

/// The decode plan the kernels use is the decode path's: a shape's workspace need (which includes the split-K partials iff SK > 1) and the
/// big path's plan agree on SK for the shapes of the model.
#[test]
fn the_big_plan_is_the_decode_plan() {
    gpu!();
    // the decode workspace is xf (rounded up to 256 B) plus SK * M * N fp32 when SK > 1: its size at two row counts gives SK
    for (k, n) in [
        (4096usize, 24576usize),
        (8192, 4096),
        (4096, 1536),
        (1536, 16384),
        (4096, 512),
        (4096, 2048),
        (2048, 4096),
        (4096, 12288),
    ] {
        let (sk, _wk) = big_plan(k, n).unwrap();
        let a = linear_workspace_bytes(16, k, n, 2);
        let b = linear_workspace_bytes(32, k, n, 2);
        let xf = |m: usize| (m.div_ceil(16) * 16 * k * 2).div_ceil(256) * 256;
        let z = |m: usize, ws: usize| ws - xf(m);
        let sk_decode = if z(16, a) == 0 { 1 } else { z(16, a) / (16 * n * 4) };
        assert_eq!(sk, sk_decode, "{k}x{n}");
        assert_eq!(z(32, b), if sk > 1 { sk * 32 * n * 4 } else { 0 }, "{k}x{n}");
    }
}

#[test]
#[ignore]
fn bench_big_linear() {
    gpu!();
    let st = Stream::new().unwrap();
    let shapes: [(&str, usize, usize, usize); 8] = [
        ("kda qkv", 4096, 24576, 2),
        ("kda o", 8192, 4096, 2),
        ("q_a", 4096, 1536, 2),
        ("q_b", 1536, 16384, 2),
        ("mla o", 16384, 4096, 2),
        ("shared gate/up", 4096, 2048, 2),
        ("shared down", 2048, 4096, 2),
        ("dense gate/up", 4096, 12288, 3),
    ];
    let t = EventTimer::new(2).unwrap();
    // `REXL3_BENCH_BITS=8` (or 6, ...) runs every shape at that width (the sc_4.10bpw pack's attention and shared experts are K8, its
    // dense MLP K6); `REXL3_BENCH_ROWS=512,2048` picks the row counts
    let bits_env: Option<usize> = std::env::var("REXL3_BENCH_BITS").ok().and_then(|v| v.parse().ok());
    let rows_env: Vec<usize> = std::env::var("REXL3_BENCH_ROWS")
        .map(|v| v.split(',').map(|r| r.trim().parse().expect("REXL3_BENCH_ROWS")).collect())
        .unwrap_or_else(|_| vec![128, 512, 1024, 2048]);
    for (name, k, n, bits) in shapes {
        let bits = bits_env.unwrap_or(bits);
        let w = synth(k, n, bits, Codebook::Mul1, false, 5);
        let mut r = Rng(6);
        for &rows in &rows_env {
            let x = DevBuf::from_slice(&input(&mut r, rows, k, Dtype::F16)).unwrap();
            let mut y = DevBuf::zeroed(rows * n * 4).unwrap();
            let mut ws = DevBuf::zeroed(w.big_workspace_bytes(rows).max(w.workspace_bytes(128))).unwrap();
            let time = |f: &mut dyn FnMut()| {
                f();
                st.sync().unwrap();
                t.record(0, &st).unwrap();
                for _ in 0..3 {
                    f();
                }
                t.record(1, &st).unwrap();
                t.ms(0, 1).unwrap() as f64 / 3.0
            };
            let big = time(&mut || w.forward_big(&x, &mut y, rows, Dtype::F16, Dtype::F32, &mut ws, &st).unwrap());
            // the decode path in 128-row windows (what a prompt costs today)
            let mut dws = DevBuf::zeroed(w.workspace_bytes(128).max(16)).unwrap();
            let xw = DevBuf::zeroed(128 * k * 2).unwrap();
            let mut yw = DevBuf::zeroed(128 * n * 4).unwrap();
            let one = time(&mut || w.forward(&xw, &mut yw, 128, Dtype::F16, Dtype::F32, &mut dws, &st).unwrap());
            let dec = one * (rows / 128) as f64;
            let flop = 2.0 * rows as f64 * k as f64 * n as f64;
            println!(
                "{name:14} {k}x{n} {bits}-bit R={rows:5} big {big:8.3} ms = {:5.1} TFLOP/s | decode path in 128-row calls {dec:8.3} ms ({:.1}x)",
                flop / 1e9 / big,
                dec / big
            );
        }
    }
}

/// The large-row linear at a MixedK Nemotron pack's dense shapes and widths (v0.1.2 lane K, P1-6: whether the tuned loads of 2 to 4 bits
/// pay at 5 to 8): ms per call at 256 to 2,048 rows, and the decode path in 128-row calls for comparison.
#[test]
#[ignore]
fn bench_big_linear_nemotron_widths() {
    gpu!();
    let st = Stream::new().unwrap();
    let shapes: [(&str, usize, usize, &[usize]); 5] = [
        ("attn q", 4096, 4096, &[4, 5]),
        ("attn k/v", 4096, 256, &[6, 7, 8]),
        ("mamba out", 8192, 4096, &[4, 5]),
        ("mamba in", 4096, 18560, &[4]),
        ("shared up", 4096, 5376, &[4, 5, 6]),
    ];
    let t = EventTimer::new(2).unwrap();
    for (name, k, n, widths) in shapes {
        for &bits in widths {
            let w = synth(k, n, bits, Codebook::Mul1, false, 7 + bits as u64);
            let mut r = Rng(8);
            for rows in [256usize, 1024, 2048] {
                let x = DevBuf::from_slice(&input(&mut r, rows, k, Dtype::F16)).unwrap();
                let mut y = DevBuf::zeroed(rows * n * 4).unwrap();
                let mut ws = DevBuf::zeroed(w.big_workspace_bytes(rows).max(w.workspace_bytes(128))).unwrap();
                let time = |f: &mut dyn FnMut()| {
                    f();
                    st.sync().unwrap();
                    t.record(0, &st).unwrap();
                    for _ in 0..3 {
                        f();
                    }
                    t.record(1, &st).unwrap();
                    t.ms(0, 1).unwrap() as f64 / 3.0
                };
                let big = time(&mut || w.forward_big(&x, &mut y, rows, Dtype::F16, Dtype::F32, &mut ws, &st).unwrap());
                let gbs = (k * n * bits / 8) as f64 / 1e6 / big;
                println!("{name:10} {k}x{n} {bits}-bit R={rows:5} big {big:8.3} ms ({gbs:6.1} GB/s of weights a call)");
            }
        }
    }
}

/// The golden of the large-row dense linear (scripts/gpu_gate.sh)/// The golden of the large-row dense linear (scripts/gpu_gate.sh): hashes of its outputs on synthetic layers of the GLM shapes and of every width and
/// codebook, committed so that a change that moves a bit in either kernel family shows up in the gate.
#[test]
#[ignore]
fn golden_exl3_big_hashes() {
    gpu!();
    let path = std::env::var("REXL3_GOLDEN_OUT").expect("REXL3_GOLDEN_OUT");
    let st = Stream::new().unwrap();
    let hash = |b: &[u8]| gb_exl3::fnv1a(b.iter().copied());
    let mut lines = Vec::new();
    let shapes =
        [(4096usize, 24576usize, 2usize), (8192, 4096, 3), (4096, 1536, 2), (1536, 16384, 4), (4096, 512, 2), (12288, 4096, 4)];
    for (i, &(k, n, bits)) in shapes.iter().enumerate() {
        let w = synth(k, n, bits, Codebook::Mul1, false, 500 + i as u64);
        let mut r = Rng(510 + i as u64);
        for rows in [129usize, 300, 1000] {
            let x = input(&mut r, rows, k, Dtype::F16);
            lines.push(format!(
                "big {k}x{n} {bits}-bit rows={rows} {:016x}",
                hash(&big_rows(&w, &st, &x, rows, Dtype::F16, Dtype::F32))
            ));
        }
    }
    for bits in 1..=8usize {
        for cb in [Codebook::Inst3, Codebook::Mcg, Codebook::Mul1] {
            let w = synth(512, 384, bits, cb, bits % 2 == 0, 600 + bits as u64);
            let mut r = Rng(610 + bits as u64);
            let x = input(&mut r, 200, 512, Dtype::BF16);
            let y = big_rows(&w, &st, &x, 200, Dtype::BF16, Dtype::BF16);
            lines.push(format!("big 512x384 {bits}-bit {cb:?} bias={} rows=200 {:016x}", bits % 2 == 0, hash(&y)));
        }
    }
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    eprintln!("wrote {} golden lines to {path}", lines.len());
}

/// The Flash-Next shared expert's projections at the steady chunk's 8,192 rows (gate/up 2,560 -> 640, down 640 -> 2,560), at the pack's
/// 4 to 8 bit widths: microseconds per call of the large-row linear, with the hash of its output.
#[test]
#[ignore]
fn bench_big_linear_shared_expert() {
    gpu!();
    let st = Stream::new().unwrap();
    let t = EventTimer::new(2).unwrap();
    let rows = 8192usize;
    for bits in [4usize, 6, 8] {
        for (name, k, n) in [("shared gate/up", 2560usize, 640usize), ("shared down", 640, 2560)] {
            let w = synth(k, n, bits, Codebook::Mul1, false, 5);
            let mut r = Rng(6);
            let x = DevBuf::from_slice(&input(&mut r, rows, k, Dtype::F16)).unwrap();
            let mut y = DevBuf::zeroed(rows * n * 4).unwrap();
            let mut ws = DevBuf::zeroed(w.big_workspace_bytes(rows)).unwrap();
            w.forward_big(&x, &mut y, rows, Dtype::F16, Dtype::F32, &mut ws, &st).unwrap();
            st.sync().unwrap();
            let mut best = f32::MAX;
            for _ in 0..6 {
                t.record(0, &st).unwrap();
                w.forward_big(&x, &mut y, rows, Dtype::F16, Dtype::F32, &mut ws, &st).unwrap();
                t.record(1, &st).unwrap();
                best = best.min(t.ms(0, 1).unwrap());
            }
            let h = y.to_vec::<u32>().unwrap().iter().fold(0xcbf29ce484222325u64, |h, &v| (h ^ v as u64).wrapping_mul(0x100000001b3));
            println!("BENCH shared {name:14} {k}x{n} {bits}-bit R={rows}: {:.1} us (rot_in + big_kernel) hash {h:016x}", best * 1e3);
        }
    }
}
