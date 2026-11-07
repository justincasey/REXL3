//! The DFlash2 drafter's affine 4-bit linear against its CPU reference (`q4_linear::dequantize`), at the drafter's shapes and
//! row counts (a draft block of 8 rows a sequence, up to 16 sequences; a prompt chunk's absorb), and its row invariance. Prints
//! the time of each drafter shape at 8 rows.
//!
//! ```text
//! cargo test --release -p gb-cuda --test q4_linear -- --nocapture
//! ```

use gb_cuda::num::{bf16_to_f32, f32_to_bf16};
use gb_cuda::q4_linear::{self, Q4Weights};
use gb_cuda::{device_available, DevBuf, Stream};

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let u = (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32;
        u * 2.0 - 1.0
    }
}

fn case(n: usize, k: usize, m: usize, seed: u64) -> (Vec<u16>, Vec<u8>, Vec<u16>) {
    let mut r = Rng(seed | 1);
    let w: Vec<u16> = (0..n * k).map(|_| f32_to_bf16(r.f() * 0.03)).collect();
    let x: Vec<u16> = (0..m * k).map(|_| f32_to_bf16(r.f())).collect();
    let q = q4_linear::quantize_bf16(&w, n, k);
    (w, q, x)
}

fn run(q: &[u8], n: usize, k: usize, x: &[u16], m: usize, st: &Stream) -> Vec<f32> {
    let w = Q4Weights::from_buf(DevBuf::from_slice(q).unwrap(), n, k).unwrap();
    let xd = DevBuf::from_slice(x).unwrap();
    let mut xs = DevBuf::zeroed(q4_linear::gsum_bytes(m, k)).unwrap();
    let mut y = DevBuf::zeroed(m * n * 2).unwrap();
    w.forward(&xd, &mut xs, &mut y, m, st).unwrap();
    st.sync().unwrap();
    let mut out = vec![0u16; m * n];
    y.download(&mut out).unwrap();
    out.into_iter().map(bf16_to_f32).collect()
}

#[test]
fn the_kernel_is_the_dequantized_matrix_product_at_every_drafter_shape() {
    if !device_available() {
        eprintln!("no CUDA device: skipped");
        return;
    }
    let st = Stream::new().unwrap();
    // (n, k): q, k/v, o, gate/up, down, fc, the conv projection; a 70-row matrix (padding inside the tile)
    for (i, &(n, k, m)) in [
        (4096, 4096, 8),
        (1024, 4096, 16),
        (4096, 4096, 128),
        (12288, 4096, 8),
        (4096, 12288, 24),
        (4096, 20480, 40),
        (4096, 20480, 300),
        (1024, 4096, 512),
        (70, 192, 33),
    ]
    .iter()
    .enumerate()
    {
        let (_, q, x) = case(n, k, m, 11 + i as u64);
        let y = run(&q, n, k, &x, m, &st);
        let w = q4_linear::dequantize(&q, n, k);
        let mut worst = 0f32;
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0f64;
                let mut mag = 0f64;
                for j in 0..k {
                    let p = w[c * k + j] as f64 * bf16_to_f32(x[r * k + j]) as f64;
                    acc += p;
                    mag += p.abs();
                }
                // bf16 output rounding (at most 2^-8 relative) plus fp32 accumulation over k products
                let tol = (acc.abs() * 2.0f64.powi(-8) + mag * 2e-5 + 1e-6) as f32;
                let e = (y[r * n + c] - acc as f32).abs();
                assert!(e <= tol, "[{n}x{k}] m={m} row {r} col {c}: kernel {} reference {acc} (tol {tol})", y[r * n + c]);
                worst = worst.max(e / tol);
            }
        }
        eprintln!("q4 [{n} x {k}] m={m} split_k={} worst error / tolerance {worst:.3}", q4_linear::split_k(n, k));
    }
}

#[test]
fn a_row_has_the_same_bits_in_any_batch() {
    if !device_available() {
        eprintln!("no CUDA device: skipped");
        return;
    }
    let st = Stream::new().unwrap();
    let (n, k) = (4096, 4096);
    let (_, q, x) = case(n, k, 128, 5);
    let all = run(&q, n, k, &x, 128, &st);
    for m in [1usize, 8, 16, 17, 40] {
        let part = run(&q, n, k, &x[..m * k], m, &st);
        assert!(part.iter().zip(&all).all(|(a, b)| a.to_bits() == b.to_bits()), "{m} rows differ from the same rows of 128");
    }
}

#[test]
fn drafter_shapes_timing() {
    if !device_available() {
        eprintln!("no CUDA device: skipped");
        return;
    }
    let st = Stream::new().unwrap();
    for &(n, k) in &[(4096, 4096), (1024, 4096), (12288, 4096), (4096, 12288), (4096, 20480)] {
        for m in [8usize, 16, 512] {
            let (_, q, x) = case(n, k, m, 3);
            let w = Q4Weights::from_buf(DevBuf::from_slice(&q).unwrap(), n, k).unwrap();
            let xd = DevBuf::from_slice(&x).unwrap();
            let mut xs = DevBuf::zeroed(q4_linear::gsum_bytes(m, k)).unwrap();
            let mut y = DevBuf::zeroed(m * n * 2).unwrap();
            for _ in 0..5 {
                w.forward(&xd, &mut xs, &mut y, m, &st).unwrap();
            }
            st.sync().unwrap();
            let it = 50;
            let t = std::time::Instant::now();
            for _ in 0..it {
                w.forward(&xd, &mut xs, &mut y, m, &st).unwrap();
            }
            st.sync().unwrap();
            let us = t.elapsed().as_secs_f64() * 1e6 / it as f64;
            let gbs = q4_linear::bytes(n, k) as f64 / us / 1e3;
            eprintln!("q4 timing [{n} x {k}] m={m}: {us:.1} us ({gbs:.0} GB/s of weights)");
        }
    }
}
