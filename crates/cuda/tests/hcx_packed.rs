//! The packed hyper-connection mixers (`hcx::pack_mixer`, 13 bits a weight): the packed kernels give the bits of the fp16 kernels for
//! `pre` and for `pre_post` at the family's shape, 1..8 rows and row counts that run several passes, with the programmatic launch
//! attribute off and on; a tensor the format cannot hold keeps the fp16 path.
#![allow(clippy::needless_range_loop)]
mod common;
use common::Rng;
use gb_cuda::fn_mhc::Mixer;
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

fn bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

fn bf(x: f32) -> u16 {
    (bf16_round(x).to_bits() >> 16) as u16
}

/// fp16 bits of bf16 values: random signs and mantissas, binary exponents spread over `span` orders from -1 (down to -span), about 1 in
/// 64 an exact zero (one in 128 of those a negative zero). A span of 40 reaches into fp16's subnormals and up to 2^14.
fn weights(rng: &mut Rng, n: usize, span: i32) -> Vec<u16> {
    (0..n)
        .map(|_| {
            let r = rng.next_u64();
            if r % 64 == 0 {
                return if (r >> 8) % 128 == 0 { 0x8000 } else { 0 };
            }
            let e = ((r >> 16) % span as u64) as i32;
            let top = if span >= 40 { 14 } else { -1 };
            let v = bf16_round(rng.uniform(-1.0, 1.0)) * (2.0f32).powi(top - e);
            f64_to_f16(bf16_round(v) as f64)
        })
        .collect()
}

fn up<T: Copy>(v: &[T]) -> DevBuf {
    DevBuf::from_slice(v).unwrap()
}

fn dl<T: Copy + Default>(b: &DevBuf, s: &Stream) -> Vec<T> {
    s.sync().unwrap();
    b.to_vec().unwrap()
}

struct Mix {
    norm: DevBuf,
    down: DevBuf,
    up: DevBuf,
    inject: DevBuf,
}

fn mix(seed: u64, span: i32) -> Mix {
    let mut rng = Rng(seed);
    let norm: Vec<u16> = (0..HC * H).map(|_| f64_to_f16(bf16_round(rng.uniform(-1.5, 1.5)) as f64)).collect();
    Mix {
        norm: up(&norm),
        down: up(&weights(&mut rng, R * HC * H, span)),
        up: up(&weights(&mut rng, HC * H * R, span)),
        inject: up(&weights(&mut rng, HC * HC * H, span)),
    }
}

#[test]
fn packed_mixers_equal_fp16_kernels_pre_and_pre_post() {
    let Some(st) = gpu() else { return };
    for (seed, span) in [(11u64, 30), (12, 22)] {
        let m = mix(seed, span);
        let mixer = Mixer { norm: &m.norm, down: &m.down, up: &m.up, inject: Some(&m.inject) };
        let packed = hcx::pack_mixer(&mixer, H, HC, R).unwrap().expect("a mixer inside the format packs");
        // 13 / 16 of the fp16 rows, plus a table a tensor
        let fp16 = (R * HC * H + HC * HC * H + HC * H * R) * 2;
        assert!(packed.bytes() < fp16 * 83 / 100, "packed {} fp16 {}", packed.bytes(), fp16);
        for mask in [0, PDL_HCX] {
            gb_cuda::exl3_linear::set_pdl_mask(mask).unwrap();
            for t in [1usize, 2, 3, 4, 5, 6, 7, 8, 11, 17] {
                let mut r = Rng(seed * 7919 + t as u64);
                let x: Vec<f32> = r.vec(t * HC * H, -2.0, 2.0);
                let bo: Vec<u16> = (0..t * H).map(|_| bf(r.uniform(-3.0, 3.0))).collect();
                let ij: Vec<f32> = r.vec(t * HC, 0.0, 2.0);
                let (dbo, dij) = (up(&bo), up(&ij));
                let scr = hcx::scratch_bytes(HC, R);
                let bits = |b: &DevBuf| dl::<f32>(b, &st).iter().map(|v| v.to_bits()).collect::<Vec<u32>>();

                // pre
                let dx = up(&x);
                let (mut y_a, mut y_b) = (DevBuf::zeroed(t * H * 2).unwrap(), DevBuf::zeroed(t * H * 2).unwrap());
                let (mut j_a, mut j_b) = (DevBuf::zeroed(t * HC * 4).unwrap(), DevBuf::zeroed(t * HC * 4).unwrap());
                let (mut s_a, mut s_b) = (DevBuf::zeroed(scr).unwrap(), DevBuf::zeroed(scr).unwrap());
                hcx::pre(&dx, &mixer, &mut y_a, Some(&mut j_a), &mut s_a, H, HC, R, EPS, t, &st).unwrap();
                hcx::pre_packed(&dx, &mixer, &packed, &mut y_b, Some(&mut j_b), &mut s_b, H, HC, R, EPS, t, &st).unwrap();
                assert_eq!(dl::<u16>(&y_a, &st), dl::<u16>(&y_b, &st), "pre: mixed rows differ at rows {t} pdl mask {mask}");
                assert_eq!(bits(&j_a), bits(&j_b), "pre: inject coefficients differ at rows {t} pdl mask {mask}");

                // pre_post
                let (mut x_a, mut x_b) = (up(&x), up(&x));
                let (mut y_a, mut y_b) = (DevBuf::zeroed(t * H * 2).unwrap(), DevBuf::zeroed(t * H * 2).unwrap());
                let (mut j_a, mut j_b) = (DevBuf::zeroed(t * HC * 4).unwrap(), DevBuf::zeroed(t * HC * 4).unwrap());
                hcx::pre_post(&mut x_a, &dbo, &dij, &mixer, &mut y_a, &mut j_a, &mut s_a, H, HC, R, EPS, t, &st).unwrap();
                hcx::pre_post_packed(&mut x_b, &dbo, &dij, &mixer, &packed, &mut y_b, &mut j_b, &mut s_b, H, HC, R, EPS, t, &st).unwrap();
                assert_eq!(bits(&x_a), bits(&x_b), "pre_post: streams differ at rows {t} pdl mask {mask}");
                assert_eq!(dl::<u16>(&y_a, &st), dl::<u16>(&y_b, &st), "pre_post: mixed rows differ at rows {t} pdl mask {mask}");
                assert_eq!(bits(&j_a), bits(&j_b), "pre_post: inject coefficients differ at rows {t} pdl mask {mask}");
            }
        }
    }
    gb_cuda::exl3_linear::set_pdl_mask(0).unwrap();
}

#[test]
fn packed_head_collapse_without_inject_rows_equals_fp16() {
    let Some(st) = gpu() else { return };
    let m = mix(21, 28);
    let mixer = Mixer { norm: &m.norm, down: &m.down, up: &m.up, inject: None };
    let packed = hcx::pack_mixer(&mixer, H, HC, R).unwrap().expect("pack");
    for t in [1usize, 3, 4, 6] {
        let mut r = Rng(5 + t as u64);
        let dx = up(&r.vec(t * HC * H, -2.0, 2.0));
        let scr = hcx::scratch_bytes(HC, R);
        let (mut y_a, mut y_b) = (DevBuf::zeroed(t * H * 2).unwrap(), DevBuf::zeroed(t * H * 2).unwrap());
        let (mut s_a, mut s_b) = (DevBuf::zeroed(scr).unwrap(), DevBuf::zeroed(scr).unwrap());
        hcx::pre(&dx, &mixer, &mut y_a, None, &mut s_a, H, HC, R, EPS, t, &st).unwrap();
        hcx::pre_packed(&dx, &mixer, &packed, &mut y_b, None, &mut s_b, H, HC, R, EPS, t, &st).unwrap();
        assert_eq!(dl::<u16>(&y_a, &st), dl::<u16>(&y_b, &st), "collapse differs at rows {t}");
    }
}

#[test]
fn a_tensor_outside_the_format_keeps_the_fp16_path() {
    let Some(st) = gpu() else { return };
    // exponents over 40 binary orders (2^14 down into the subnormals): the exponent pairs span more than 15 values
    let wide = mix(31, 40);
    let mixer = Mixer { norm: &wide.norm, down: &wide.down, up: &wide.up, inject: Some(&wide.inject) };
    assert!(hcx::pack_mixer(&mixer, H, HC, R).unwrap().is_none(), "a span past 15 exponent pairs must not pack");
    // a weight with more than seven mantissa bits (a plain fp16 value) cannot be held
    let mut d: Vec<u16> = weights(&mut Rng(41), R * HC * H, 20);
    d[12345] = f64_to_f16(1.0 + 1.0 / 512.0);
    let narrow = mix(32, 20);
    let down = up(&d);
    let mixer = Mixer { norm: &narrow.norm, down: &down, up: &narrow.up, inject: Some(&narrow.inject) };
    assert!(hcx::pack_mixer(&mixer, H, HC, R).unwrap().is_none(), "a value with eight mantissa bits must not pack");
    let _ = st;
    assert!(hcx::pack_mixer(&Mixer { norm: &narrow.norm, down: &narrow.down, up: &narrow.up, inject: Some(&narrow.inject) }, H, HC + 1, R)
        .unwrap()
        .is_none());
}
