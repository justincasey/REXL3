//! The Flash-Next pre-mix's down product in the engine's own kernel (`hcx::pre_gemm` / `hcx::pre_gemm_post` on the Flash-Next shape: the normed rows
//! are formed on the fly from the post-inject streams and the product runs in the library's split-K order, see `csrc/hcg.cu`) against the 64-row
//! slab loop of cuBLAS (`hcx::pre_gemm_ref`, the arithmetic the fast path must reproduce): the mixed rows, the inject rows and the post-injected
//! streams, bit for bit, on the prefill chunk grid and at ragged row counts, with data that makes every accumulation order visible (streams spread
//! over six orders of magnitude, rows with outliers, zero rows, weights with exact zeros and a few large entries).
//!
//!     cargo test -p gb-cuda --release --test hcg_down -- --nocapture --test-threads=1
#![cfg(not(gb_no_cuda))]
mod common;

use common::Rng;
use gb_cuda::{fn_mhc, hcx, DevBuf, Stream};
use gb_exl3::f16::f64_to_f16;

const H: usize = 2560;
const HC: usize = 4;
const R: usize = 320;
const K: usize = HC * H;

fn gpu() -> Option<Stream> {
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    gb_cuda::rt::init_context().ok()?;
    Stream::new().ok()
}

fn bf(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn dl<T: Copy + Default>(b: &DevBuf, st: &Stream) -> Vec<T> {
    st.sync().unwrap();
    b.to_vec().unwrap()
}

/// fp16 weights: mostly small, some exact zeros, a few large entries.
fn weights(rng: &mut Rng, n: usize, small: f32) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let r = rng.uniform(0., 1.);
            let v = if r < 0.02 { 0.0 } else if r < 0.03 { rng.uniform(-0.6, 0.6) } else { rng.uniform(-small, small) };
            f64_to_f16(v as f64 * if i % 97 == 0 { 8.0 } else { 1.0 }) as u16
        })
        .collect()
}

/// Stream rows: a spread of magnitudes per row, outliers in some rows, a few all-zero rows.
fn streams(rng: &mut Rng, t: usize) -> Vec<f32> {
    let mut x = vec![0f32; t * K];
    for row in 0..t {
        let scale = 10f32.powf(rng.uniform(-3., 3.));
        let zero = row % 53 == 7;
        for i in 0..K {
            let mut v = rng.uniform(-3., 3.) * scale;
            if rng.uniform(0., 1.) < 0.001 {
                v *= 300.0;
            }
            x[row * K + i] = if zero { 0.0 } else { v };
        }
    }
    x
}

#[test]
fn the_down_kernel_equals_the_slab_loop_on_the_chunk_grid_and_ragged_rows() {
    let Some(st) = gpu() else { return };
    let mut rng = Rng(0xd0_17);
    let (nw, down, up, inject) = (
        weights(&mut rng, K, 0.2),
        weights(&mut rng, R * K, 0.06),
        weights(&mut rng, K * R, 0.06),
        weights(&mut rng, HC * K, 0.06),
    );
    let (nw, down, up, inject) = (
        DevBuf::from_slice(&nw).unwrap(),
        DevBuf::from_slice(&down).unwrap(),
        DevBuf::from_slice(&up).unwrap(),
        DevBuf::from_slice(&inject).unwrap(),
    );
    let mix = fn_mhc::Mixer { norm: &nw, down: &down, up: &up, inject: Some(&inject) };
    let mut rng = Rng(0xfeed);
    let mut checked = 0usize;
    for t in [1usize, 63, 64, 65, 127, 128, 129, 191, 1000, 2049, 4016, 4095, 4096] {
        let x = streams(&mut rng, t);
        let block: Vec<u16> = (0..t * H).map(|_| bf(rng.uniform(-4., 4.))).collect();
        let injp: Vec<f32> = (0..t * HC).map(|_| rng.uniform(-1.5, 1.5)).collect();
        let (block, injp) = (DevBuf::from_slice(&block).unwrap(), DevBuf::from_slice(&injp).unwrap());
        let scratch_bytes = hcx::gemm_scratch_bytes(t, H, HC, R);
        for post in [false, true] {
            let mut xr = DevBuf::from_slice(&x).unwrap();
            if post {
                fn_mhc::hc_post(&block, None, &injp, &mut xr, H, HC, t, &st).unwrap();
            }
            let (mut yr, mut ir, mut sr) =
                (DevBuf::zeroed(t * H * 2).unwrap(), DevBuf::zeroed(t * HC * 4).unwrap(), DevBuf::zeroed(scratch_bytes).unwrap());
            hcx::pre_gemm_ref(&xr, &mix, &mut yr, Some(&mut ir), &mut sr, H, HC, R, 1e-6, t, &st).unwrap();
            let mut xf = DevBuf::from_slice(&x).unwrap();
            let (mut yf, mut i_f, mut sf) =
                (DevBuf::zeroed(t * H * 2).unwrap(), DevBuf::zeroed(t * HC * 4).unwrap(), DevBuf::zeroed(scratch_bytes).unwrap());
            if post {
                hcx::pre_gemm_post(&mut xf, &block, &injp, &mix, &mut yf, Some(&mut i_f), &mut sf, H, HC, R, 1e-6, t, &st).unwrap();
            } else {
                hcx::pre_gemm(&xf, &mix, &mut yf, Some(&mut i_f), &mut sf, H, HC, R, 1e-6, t, &st).unwrap();
            }
            let what = format!("{t} rows, post {post}");
            let (a, b): (Vec<u16>, Vec<u16>) = (dl(&yf, &st), dl(&yr, &st));
            assert_eq!(a.iter().zip(&b).filter(|(p, q)| p != q).count(), 0, "mixed rows differ, {what}");
            let (a, b): (Vec<f32>, Vec<f32>) = (dl(&xf, &st), dl(&xr, &st));
            assert_eq!(a.iter().zip(&b).filter(|(p, q)| p.to_bits() != q.to_bits()).count(), 0, "streams differ, {what}");
            let (a, b): (Vec<f32>, Vec<f32>) = (dl(&i_f, &st), dl(&ir, &st));
            assert_eq!(a.iter().zip(&b).filter(|(p, q)| p.to_bits() != q.to_bits()).count(), 0, "inject rows differ, {what}");
            checked += t * H;
        }
    }
    eprintln!("down kernel == slab loop: 13 row counts x {{plain, fused post-inject}}, {checked} mixed outputs and their inject rows and streams, every bit equal");
}
