//! Timing of the Flash-Next pre-mix on the prefill chunk grid (4,016 rows: the steady chunk), whole call (norm + inject, down, up + mix).
//! Ignored: run by hand,
//!
//!     cargo test -p gb-cuda --release --test hcg_bench -- --ignored --nocapture --test-threads=1
#![cfg(not(gb_no_cuda))]
mod common;

use common::Rng;
use gb_cuda::{fn_mhc, hcx, DevBuf, Stream};
use gb_exl3::f16::f64_to_f16;
use std::time::Instant;

const H: usize = 2560;
const HC: usize = 4;
const R: usize = 320;
const K: usize = HC * H;

#[test]
#[ignore]
fn bench_pre_mix_on_the_chunk_grid() {
    if !gb_cuda::device_available() {
        return;
    }
    gb_cuda::rt::init_context().unwrap();
    let st = Stream::new().unwrap();
    let t: usize = std::env::var("HCG_BENCH_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(4016);
    let reps: usize = std::env::var("HCG_BENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let mut rng = Rng(0xa11ce);
    let mut half = |n: usize, lo: f32, hi: f32| (0..n).map(|_| f64_to_f16(rng.uniform(lo, hi) as f64)).collect::<Vec<u16>>();
    let (nw, down, up, inject) = (half(K, -0.2, 0.2), half(R * K, -0.06, 0.06), half(K * R, -0.06, 0.06), half(HC * K, -0.06, 0.06));
    let (nw, down, up, inject) = (
        DevBuf::from_slice(&nw).unwrap(),
        DevBuf::from_slice(&down).unwrap(),
        DevBuf::from_slice(&up).unwrap(),
        DevBuf::from_slice(&inject).unwrap(),
    );
    let mix = fn_mhc::Mixer { norm: &nw, down: &down, up: &up, inject: Some(&inject) };
    let mut rng = Rng(0xbeef);
    let x: Vec<f32> = (0..t * K).map(|i| rng.uniform(-3., 3.) * (1.0 + (i / K) as f32 * 0.01)).collect();
    let block: Vec<u16> = (0..t * H).map(|_| ((rng.uniform(-2., 2.)).to_bits() >> 16) as u16).collect();
    let injp: Vec<f32> = (0..t * HC).map(|_| rng.uniform(-0.001, 0.001)).collect();
    let (block, injp) = (DevBuf::from_slice(&block).unwrap(), DevBuf::from_slice(&injp).unwrap());
    let mut xs = DevBuf::from_slice(&x).unwrap();
    let (mut y, mut inj, mut sc) =
        (DevBuf::zeroed(t * H * 2).unwrap(), DevBuf::zeroed(t * HC * 4).unwrap(), DevBuf::zeroed(hcx::gemm_scratch_bytes(t, H, HC, R)).unwrap());
    // HCG_BENCH_FLUSH=<MB>: a memset of that many MB between two calls, so the next call starts with the L2 holding other data (as in the model,
    // where attention or the experts run between two pre-mixes); the wall time then includes the memset, the kernel times (nsys) do not
    let flush_mb: usize = std::env::var("HCG_BENCH_FLUSH").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut flush = DevBuf::zeroed(flush_mb.max(1) << 20).unwrap();
    let lazy = std::env::var("HCG_BENCH_LAZY").map(|v| v == "1").unwrap_or(false);
    let mut run = |xs: &mut DevBuf, y: &mut DevBuf, inj: &mut DevBuf, sc: &mut DevBuf| {
        if flush_mb > 0 {
            flush.memset_async(1, &st).unwrap();
        }
        if lazy {
            hcx::pre_gemm_post_lazy(xs, &block, &injp, &mix, y, inj, sc, H, HC, R, 1e-6, t, &st).unwrap();
        } else {
            hcx::pre_gemm_post(xs, &block, &injp, &mix, y, Some(inj), sc, H, HC, R, 1e-6, t, &st).unwrap();
        }
    };
    for _ in 0..5 {
        run(&mut xs, &mut y, &mut inj, &mut sc);
    }
    st.sync().unwrap();
    let mut best = f64::MAX;
    let mut all = Vec::new();
    for _ in 0..5 {
        let t0 = Instant::now();
        for _ in 0..reps {
            run(&mut xs, &mut y, &mut inj, &mut sc);
        }
        st.sync().unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        all.push(ms);
        best = best.min(ms);
    }
    eprintln!("hcg bench: {t} rows, pre-mix call {best:.3} ms best (rounds {all:.3?})");
}
