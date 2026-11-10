//! `hcx::pre_post` (the decode pre-mix with the previous block's hc_post folded into its two kernels) against the two launches it
//! replaces, `fn_mhc::hc_post` then `hcx::pre`: the streams it leaves, the mixed rows and the inject coefficients are the same bits,
//! under plain and programmatic launches, at the family's shape for 1..8 rows and for row counts that run several passes.
#![allow(clippy::needless_range_loop)]
mod common;
use common::Rng;
use gb_cuda::fn_mhc::{self, Mixer};
use gb_cuda::{hcx, DevBuf, Stream};
use gb_exl3::f16::f64_to_f16;

const H: usize = 2560;
const HC: usize = 4;
const R: usize = 320;
const EPS: f32 = 1e-6;
const PDL_HCX: i32 = 16; // gbp_pdl::kHcx

fn gpu() -> Option<Stream> {
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    gb_cuda::rt::init_context().ok()?;
    Stream::new().ok()
}

fn hf(x: f32) -> u16 {
    f64_to_f16(x as f64)
}

fn bf(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn up<T: Copy>(v: &[T]) -> DevBuf {
    DevBuf::from_slice(v).unwrap()
}

fn dl<T: Copy + Default>(b: &DevBuf, s: &Stream) -> Vec<T> {
    s.sync().unwrap();
    b.to_vec().unwrap()
}

#[test]
fn hcx_pre_post_equals_hc_post_then_pre() {
    let Some(st) = gpu() else { return };
    let mut rng = Rng(77);
    let nw: Vec<u16> = (0..HC * H).map(|_| hf(rng.uniform(-1.5, 1.5))).collect();
    let down: Vec<u16> = (0..(R + HC) * HC * H).map(|_| hf(rng.uniform(-1.5, 1.5))).collect();
    let upw: Vec<u16> = (0..HC * H * R).map(|_| hf(rng.uniform(-1.5, 1.5))).collect();
    let (dnw, ddown, dup) = (up(&nw), up(&down[..R * HC * H]), up(&upw));
    let dinj_w = up(&down[R * HC * H..]);
    let mix = Mixer { norm: &dnw, down: &ddown, up: &dup, inject: Some(&dinj_w) };
    assert!(hcx::pre_post_foldable(H, HC, R));
    for mask in [0, PDL_HCX] {
        gb_cuda::exl3_linear::set_pdl_mask(mask).unwrap();
        for t in [1usize, 2, 3, 4, 5, 6, 7, 8, 11, 17] {
            for seed in 0..2u64 {
                let mut r = Rng(seed * 4099 + t as u64);
                let x: Vec<f32> = r.vec(t * HC * H, -2.0, 2.0);
                let bo: Vec<u16> = (0..t * H).map(|_| bf(r.uniform(-3.0, 3.0))).collect();
                let ij: Vec<f32> = r.vec(t * HC, 0.0, 2.0);
                let (dbo, dij) = (up(&bo), up(&ij));
                let scr = hcx::scratch_bytes(HC, R);

                let mut x_ref = up(&x);
                let mut y_ref = DevBuf::zeroed(t * H * 2).unwrap();
                let mut j_ref = DevBuf::zeroed(t * HC * 4).unwrap();
                let mut s_ref = DevBuf::zeroed(scr).unwrap();
                fn_mhc::hc_post(&dbo, None, &dij, &mut x_ref, H, HC, t, &st).unwrap();
                hcx::pre(&x_ref, &mix, &mut y_ref, Some(&mut j_ref), &mut s_ref, H, HC, R, EPS, t, &st).unwrap();

                let mut x_new = up(&x);
                let mut y_new = DevBuf::zeroed(t * H * 2).unwrap();
                let mut j_new = DevBuf::zeroed(t * HC * 4).unwrap();
                let mut s_new = DevBuf::zeroed(scr).unwrap();
                hcx::pre_post(&mut x_new, &dbo, &dij, &mix, &mut y_new, &mut j_new, &mut s_new, H, HC, R, EPS, t, &st).unwrap();

                let (xa, xb): (Vec<u32>, Vec<u32>) = (dl::<f32>(&x_new, &st).iter().map(|v| v.to_bits()).collect(), dl::<f32>(&x_ref, &st).iter().map(|v| v.to_bits()).collect());
                let (ya, yb): (Vec<u16>, Vec<u16>) = (dl(&y_new, &st), dl(&y_ref, &st));
                let (ja, jb): (Vec<u32>, Vec<u32>) = (dl::<f32>(&j_new, &st).iter().map(|v| v.to_bits()).collect(), dl::<f32>(&j_ref, &st).iter().map(|v| v.to_bits()).collect());
                let diff = |a: &[u32], b: &[u32]| a.iter().zip(b).filter(|(p, q)| p != q).count();
                let yd = ya.iter().zip(&yb).filter(|(p, q)| p != q).count();
                assert_eq!(
                    (diff(&xa, &xb), yd, diff(&ja, &jb)),
                    (0, 0, 0),
                    "pre_post differs from hc_post + pre at rows {t} seed {seed} pdl mask {mask}"
                );
            }
        }
    }
    gb_cuda::exl3_linear::set_pdl_mask(0).unwrap();
}
