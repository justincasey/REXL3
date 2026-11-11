//! The Flash-Next pre-mix's fast path (`hcx::pre_gemm` / `hcx::pre_gemm_post`: one kernel forms the normed rows and the inject product's 20
//! splits, the library's down GEMMs, the fused up + mix) against the 64-row slab loop (`hcx::pre_gemm_ref`, the arithmetic the fast path must
//! reproduce): the mixed rows, the inject rows and the post-injected streams, bit for bit, at ragged row counts and with and without the fused
//! post-inject.
//!
//!     cargo test -p gb-cuda --release --test hcg_inject -- --nocapture --test-threads=1
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

#[test]
fn the_fast_pre_mix_equals_the_slab_loop_with_and_without_the_fused_post_inject() {
    let Some(st) = gpu() else { return };
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
    let head_mix = fn_mhc::Mixer { norm: &nw, down: &down, up: &up, inject: None };
    let mut rng = Rng(0xbeef);
    for t in [1usize, 7, 31, 32, 33, 63, 64, 65, 100, 1000, 2049, 15, 16, 17, 47, 48, 49, 4033, 4081] {
        // streams with a spread of magnitudes (rows differ by orders), block output and the previous block's inject weights
        let x: Vec<f32> = (0..t * K).map(|i| rng.uniform(-3., 3.) * (1.0 + (i / K) as f32 * 0.01)).collect();
        let block: Vec<u16> = (0..t * H).map(|_| bf(rng.uniform(-2., 2.))).collect();
        let injp: Vec<f32> = (0..t * HC).map(|_| rng.uniform(-1.5, 1.5)).collect();
        let (block, injp) = (DevBuf::from_slice(&block).unwrap(), DevBuf::from_slice(&injp).unwrap());
        let scratch_bytes = hcx::gemm_scratch_bytes(t, H, HC, R);
        for with_inject in [true, false] {
            let mx = if with_inject { &mix } else { &head_mix };
            for post in [false, true] {
                // the reference: hc_post (when asked) then the slab loop
                let mut xr = DevBuf::from_slice(&x).unwrap();
                if post {
                    fn_mhc::hc_post(&block, None, &injp, &mut xr, H, HC, t, &st).unwrap();
                }
                let (mut yr, mut ir, mut sr) =
                    (DevBuf::zeroed(t * H * 2).unwrap(), DevBuf::zeroed(t * HC * 4).unwrap(), DevBuf::zeroed(scratch_bytes).unwrap());
                hcx::pre_gemm_ref(&xr, mx, &mut yr, with_inject.then_some(&mut ir), &mut sr, H, HC, R, 1e-6, t, &st).unwrap();
                // the fast path
                let mut xf = DevBuf::from_slice(&x).unwrap();
                let (mut yf, mut i_f, mut sf) =
                    (DevBuf::zeroed(t * H * 2).unwrap(), DevBuf::zeroed(t * HC * 4).unwrap(), DevBuf::zeroed(scratch_bytes).unwrap());
                if post {
                    hcx::pre_gemm_post(&mut xf, &block, &injp, mx, &mut yf, with_inject.then_some(&mut i_f), &mut sf, H, HC, R, 1e-6, t, &st)
                        .unwrap();
                } else {
                    hcx::pre_gemm(&xf, mx, &mut yf, with_inject.then_some(&mut i_f), &mut sf, H, HC, R, 1e-6, t, &st).unwrap();
                }
                let what = format!("{t} rows, inject {with_inject}, post {post}");
                assert_eq!(dl::<u16>(&yf, &st), dl::<u16>(&yr, &st), "mixed rows, {what}");
                assert_eq!(dl::<f32>(&xf, &st), dl::<f32>(&xr, &st), "streams, {what}");
                if with_inject {
                    let (a, b): (Vec<f32>, Vec<f32>) = (dl(&i_f, &st), dl(&ir, &st));
                    assert_eq!(a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "inject rows, {what}");
                }
            }
        }
    }
    eprintln!("fast pre-mix == slab loop: 19 row counts x {{inject, head}} x {{plain, fused post-inject}}, every output bit equal");
}
