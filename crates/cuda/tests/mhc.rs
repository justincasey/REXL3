//! mHC kernels (`csrc/mhc.cu`): bit-exact CPU mirrors of every non-transcendental stage, row invariance for 1..=128 rows and
//! beyond, fused == unfused, fn storage types, accuracy against a float64 reference, accuracy against the ExLlamaV3 fork's
//! `HyperConnection` on the real pack's layer tensors, golden hashes, and benchmarks. GPU tests skip when there is no device.
//!
//! Oracle tests need `REXL3_GLM_DIR` (the GLM pack: the layer tensors are read with `gb_exl3::Checkpoint`) and
//! `REXL3_MHC_REF_DIR` (dumps written by `tools/dump_mhc_reference.py` on a node with the fork); when either is unset they
//! print "not configured" and pass, so the Qwen gate stays green without the GLM pack.
//!
//!     REXL3_GOLDEN_OUT=golden_mhc.txt cargo test --release -p gb-cuda --test mhc golden_mhc_hashes -- --ignored
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
mod common;

use common::*;
use core::ffi::c_void;
use gb_cuda::mhc::{self, MhcCfg, MhcSite, MhcWork, Tune, Ty, HIDDEN, NFN, ROW};
use gb_cuda::num::*;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

const CFG: MhcCfg = MhcCfg::GLM53;
const NORM_EPS: f32 = 1e-5;

/// A plain `usize` is "split = n, one row per block" (0 = automatic split); a `Tune` is itself.
trait AsTune {
    fn tune(self) -> Tune;
}
impl AsTune for usize {
    fn tune(self) -> Tune {
        Tune { split: self, rows_per_block: 1 }
    }
}
impl AsTune for Tune {
    fn tune(self) -> Tune {
        self
    }
}

fn up<T: Copy>(v: &[T]) -> DevBuf {
    DevBuf::from_slice(v).unwrap()
}
fn dl<T: Copy + Default>(b: &DevBuf) -> Vec<T> {
    b.to_vec().unwrap()
}
/// A buffer filled with 0xFF bytes (NaN as f32, and as f16/bf16), so a value a kernel forgot to write shows up.
fn poisoned(bytes: usize, st: &Stream) -> DevBuf {
    let mut b = DevBuf::alloc(bytes).unwrap();
    b.memset_async(0xFF, st).unwrap();
    st.sync().unwrap();
    b
}
fn fp(b: &DevBuf) -> *const f32 {
    b.ptr() as *const f32
}
fn fm(b: &DevBuf) -> *mut f32 {
    b.ptr() as *mut f32
}
fn vp(b: &DevBuf) -> *const c_void {
    b.ptr() as *const c_void
}

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3))
}
fn bytes_of<T: Copy>(v: &[T]) -> Vec<u8> {
    let n = std::mem::size_of_val(v);
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n).to_vec() }
}

// ------------------------------------------------------------------------------------------------ synthetic data

fn normal(rng: &mut Rng) -> f32 {
    let (a, b) = (rng.uniform(1e-7, 1.0), rng.uniform(0.0, 1.0));
    (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
}

/// `fn` like the real tensors: rms about 0.02, exactly representable in both bf16 and fp16.
fn synth_fn(rng: &mut Rng) -> Vec<f32> {
    (0..NFN * ROW)
        .map(|_| {
            let v = bf16_to_f32(f32_to_bf16(normal(rng) * 0.02));
            if v.abs() < 6.2e-5 {
                0.0
            } else {
                v
            }
        })
        .collect()
}
fn synth_base(rng: &mut Rng) -> Vec<f32> {
    // pre and post logits around -8..0 as in the real layers, comb logits spread so the softmax is not saturated everywhere
    (0..NFN).map(|i| if i < 8 { rng.uniform(-8.0, 1.0) } else { rng.uniform(-3.0, 3.0) }).collect()
}
fn synth_scale(rng: &mut Rng) -> Vec<f32> {
    vec![rng.uniform(0.05, 0.6), rng.uniform(0.05, 0.6), rng.uniform(0.05, 0.6)]
}

/// Streams with per-channel scales (log-normal), distinct content per stream and a few massive channels; `amp` is the rms.
fn synth_streams(rng: &mut Rng, rows: usize, amp: f32) -> Vec<f32> {
    let chan: Vec<f32> = (0..HIDDEN).map(|_| (0.6 * normal(rng)).exp()).collect();
    let mut x = vec![0f32; rows * ROW];
    for r in 0..rows {
        let rs = (0.3 * normal(rng)).exp();
        let big = [rng.next_u64() as usize % HIDDEN, rng.next_u64() as usize % HIDDEN, rng.next_u64() as usize % HIDDEN];
        for h in 0..4 {
            let sh = rng.uniform(0.5, 1.5);
            for d in 0..HIDDEN {
                x[r * ROW + h * HIDDEN + d] = normal(rng) * chan[d] * amp * rs * sh;
            }
            for &b in &big {
                x[r * ROW + h * HIDDEN + b] = rng.uniform(100.0, 1500.0) * amp * if b % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
    }
    x
}

fn synth_rows(rng: &mut Rng, rows: usize, amp: f32) -> Vec<f32> {
    (0..rows * HIDDEN).map(|_| normal(rng) * amp).collect()
}

// ------------------------------------------------------------------------------------------------ CPU mirrors (bit-exact)

fn fma(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

/// Atom `a`'s 25 partials (slot 24 = sum of squares): lane chains over the 16 (stream, d) values, then the xor butterfly.
fn atom_partials(xr: &[f32], fnw: &[f32], a: usize) -> [f32; 25] {
    let mut v = vec![[0f32; 25]; 32];
    for lane in 0..32 {
        let d0 = a * 128 + lane * 4;
        let mut ss = 0f32;
        for h in 0..4 {
            for i in 0..4 {
                let t = xr[h * HIDDEN + d0 + i];
                ss = fma(t, t, ss);
            }
        }
        v[lane][24] = ss;
        for m in 0..NFN {
            let mut acc = 0f32;
            for h in 0..4 {
                for i in 0..4 {
                    acc = fma(xr[h * HIDDEN + d0 + i], fnw[m * ROW + h * HIDDEN + d0 + i], acc);
                }
            }
            v[lane][m] = acc;
        }
    }
    for mask in [16usize, 8, 4, 2, 1] {
        let b = v.clone();
        for l in 0..32 {
            for q in 0..25 {
                v[l][q] = b[l][q] + b[l ^ mask][q];
            }
        }
    }
    v[0]
}

/// The 32 atoms of a row, laid out like the kernel's workspace (stride 32 floats, slots 25..31 are zero).
fn row_partials(xr: &[f32], fnw: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; 32 * 32];
    for a in 0..32 {
        let p = atom_partials(xr, fnw, a);
        out[a * 32..a * 32 + 25].copy_from_slice(&p);
    }
    out
}

/// S[q]: the sequential chain over atoms.
fn reduce_atoms(part: &[f32]) -> [f32; 25] {
    let mut s = [0f32; 25];
    for q in 0..25 {
        let mut t = part[q];
        for a in 1..32 {
            t += part[a * 32 + q];
        }
        s[q] = t;
    }
    s
}

fn sigmoid32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

struct Tail32 {
    pre: [f32; 4],
    post: [f32; 4],
    comb: [f32; 16],
}

/// pre/post/comb from S in f32 with the kernel's operations (libm `exp` stands in for CUDA's `expf`: not bit-identical).
fn tail32(s: &[f32; 25], base: &[f32], scale: &[f32], cfg: &MhcCfg) -> Tail32 {
    let rmr = 1.0f32 / (s[24] / ROW as f32 + cfg.rms_eps).sqrt();
    let mut pre = [0f32; 4];
    let mut post = [0f32; 4];
    for h in 0..4 {
        pre[h] = sigmoid32(fma(s[h] * rmr, scale[0], base[h])) + cfg.hc_eps;
        post[h] = 2.0 * sigmoid32(fma(s[4 + h] * rmr, scale[1], base[4 + h]));
    }
    let mut v = [0f32; 16];
    for l in 0..16 {
        v[l] = fma(s[8 + l] * rmr, scale[2], base[8 + l]);
    }
    for i in 0..4 {
        let mx = (v[i * 4].max(v[i * 4 + 1])).max(v[i * 4 + 2].max(v[i * 4 + 3]));
        // the kernel's butterfly pairs (j, j^1) then (j, j^2): ((a0+a1)+(a2+a3))
        let e: Vec<f32> = (0..4).map(|j| (v[i * 4 + j] - mx).exp()).collect();
        let sm = (e[0] + e[1]) + (e[2] + e[3]);
        for j in 0..4 {
            v[i * 4 + j] = e[j] / sm + cfg.hc_eps;
        }
    }
    let colnorm = |v: &mut [f32; 16]| {
        for j in 0..4 {
            let cs = (v[j] + v[4 + j]) + (v[8 + j] + v[12 + j]);
            for i in 0..4 {
                v[i * 4 + j] /= cs + cfg.hc_eps;
            }
        }
    };
    let rownorm = |v: &mut [f32; 16]| {
        for i in 0..4 {
            let rs = (v[i * 4] + v[i * 4 + 1]) + (v[i * 4 + 2] + v[i * 4 + 3]);
            for j in 0..4 {
                v[i * 4 + j] /= rs + cfg.hc_eps;
            }
        }
    };
    colnorm(&mut v);
    for _ in 0..cfg.sinkhorn_iters - 1 {
        rownorm(&mut v);
        colnorm(&mut v);
    }
    Tail32 { pre, post, comb: v }
}

/// Collapsed row given pre (bit-exact: fma chain in stream order).
fn collapse32(xr: &[f32], pre: &[f32; 4]) -> Vec<f32> {
    (0..HIDDEN)
        .map(|d| {
            let mut o = 0f32;
            for h in 0..4 {
                o = fma(pre[h], xr[h * HIDDEN + d], o);
            }
            o
        })
        .collect()
}

fn butterfly32(mut a: [f32; 32]) -> f32 {
    for m in [16usize, 8, 4, 2, 1] {
        let b = a;
        for l in 0..32 {
            a[l] = b[l] + b[l ^ m];
        }
    }
    a[0]
}

/// RMSNorm of a collapsed row in the kernel's order: thread t (of 512) owns float4 columns t and t + 512, fma chain; butterfly per warp;
/// warps added in order. Returns the f32 normalised row (before the one output rounding).
fn norm32(c: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut thr = [0f32; 512];
    for t in 0..512 {
        let mut s = 0f32;
        for k in 0..2 {
            for i in 0..4 {
                let v = c[(t + 512 * k) * 4 + i];
                s = fma(v, v, s);
            }
        }
        thr[t] = s;
    }
    let mut tot = butterfly32(thr[0..32].try_into().unwrap());
    for wp in 1..16 {
        tot += butterfly32(thr[wp * 32..wp * 32 + 32].try_into().unwrap());
    }
    let inv = 1.0f32 / (tot / HIDDEN as f32 + eps).sqrt();
    c.iter().zip(w).map(|(&v, &w)| (v * inv) * w).collect()
}

/// apply for one row (bit-exact).
fn apply32(xr: &[f32], y: &[f32], post: &[f32], comb: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; ROW];
    for h in 0..4 {
        for d in 0..HIDDEN {
            let mut o = post[h] * y[d];
            for g in 0..4 {
                o = fma(comb[g * 4 + h], xr[g * HIDDEN + d], o);
            }
            out[h * HIDDEN + d] = o;
        }
    }
    out
}

fn mean32(xr: &[f32]) -> Vec<f32> {
    (0..HIDDEN).map(|d| (((xr[d] + xr[HIDDEN + d]) + xr[2 * HIDDEN + d]) + xr[3 * HIDDEN + d]) * 0.25).collect()
}

// ------------------------------------------------------------------------------------------------ float64 reference

struct Ref64 {
    pre: [f64; 4],
    post: [f64; 4],
    comb: [f64; 16],
    collapsed: Vec<f64>,
}

fn sigmoid64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The whole mixer in float64 from the (exact) f32 inputs; `iters` Sinkhorn iterations as the oracle counts them.
fn ref64(xr: &[f32], fnw: &[f32], base: &[f32], scale: &[f32], cfg: &MhcCfg) -> Ref64 {
    let mut s = [0f64; 25];
    for m in 0..NFN {
        let mut a = 0f64;
        for c in 0..ROW {
            a += xr[c] as f64 * fnw[m * ROW + c] as f64;
        }
        s[m] = a;
    }
    s[24] = xr.iter().map(|&v| v as f64 * v as f64).sum();
    let rmr = 1.0 / (s[24] / ROW as f64 + cfg.rms_eps as f64).sqrt();
    let eps = cfg.hc_eps as f64;
    let mut pre = [0f64; 4];
    let mut post = [0f64; 4];
    for h in 0..4 {
        pre[h] = sigmoid64(s[h] * rmr * scale[0] as f64 + base[h] as f64) + eps;
        post[h] = 2.0 * sigmoid64(s[4 + h] * rmr * scale[1] as f64 + base[4 + h] as f64);
    }
    let mut v = [0f64; 16];
    for l in 0..16 {
        v[l] = s[8 + l] * rmr * scale[2] as f64 + base[8 + l] as f64;
    }
    for i in 0..4 {
        let mx = v[i * 4..i * 4 + 4].iter().cloned().fold(f64::MIN, f64::max);
        let e: Vec<f64> = (0..4).map(|j| (v[i * 4 + j] - mx).exp()).collect();
        let sm: f64 = e.iter().sum();
        for j in 0..4 {
            v[i * 4 + j] = e[j] / sm + eps;
        }
    }
    let colnorm = |v: &mut [f64; 16]| {
        for j in 0..4 {
            let cs: f64 = (0..4).map(|i| v[i * 4 + j]).sum();
            for i in 0..4 {
                v[i * 4 + j] /= cs + eps;
            }
        }
    };
    let rownorm = |v: &mut [f64; 16]| {
        for i in 0..4 {
            let rs: f64 = (0..4).map(|j| v[i * 4 + j]).sum();
            for j in 0..4 {
                v[i * 4 + j] /= rs + eps;
            }
        }
    };
    colnorm(&mut v);
    for _ in 0..cfg.sinkhorn_iters - 1 {
        rownorm(&mut v);
        colnorm(&mut v);
    }
    let collapsed = (0..HIDDEN).map(|d| (0..4).map(|h| pre[h] * xr[h * HIDDEN + d] as f64).sum()).collect();
    Ref64 { pre, post, comb: v, collapsed }
}

fn rmsnorm64(c: &[f64], w: &[f32], eps: f64) -> Vec<f64> {
    let ss: f64 = c.iter().map(|v| v * v).sum();
    let inv = 1.0 / (ss / c.len() as f64 + eps).sqrt();
    c.iter().zip(w).map(|(&v, &w)| v * inv * w as f64).collect()
}

fn apply64(xr: &[f32], y: &[f32], post: &[f64], comb: &[f64]) -> Vec<f64> {
    let mut out = vec![0f64; ROW];
    for h in 0..4 {
        for d in 0..HIDDEN {
            out[h * HIDDEN + d] =
                post[h] * y[d] as f64 + (0..4).map(|g| comb[g * 4 + h] * xr[g * HIDDEN + d] as f64).sum::<f64>();
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------ GPU drivers

struct Site {
    fnw: Vec<f32>,
    base: Vec<f32>,
    scale: Vec<f32>,
    dev: MhcSite,
}

fn make_site(seed: u64) -> Site {
    let mut rng = Rng(seed);
    let fnw = synth_fn(&mut rng);
    let base = synth_base(&mut rng);
    let scale = synth_scale(&mut rng);
    let dev = MhcSite::from_f32(&fnw, &base, &scale).unwrap();
    assert_eq!(dev.fn_ty, Ty::F16, "synthetic fn is exact in fp16");
    Site { fnw, base, scale, dev }
}

fn ty_bytes(v: &[f32], ty: Ty) -> Vec<u8> {
    match ty {
        Ty::F32 => bytes_of(v),
        Ty::F16 => bytes_of(&v.iter().map(|&x| f64_to_f16(x as f64)).collect::<Vec<u16>>()),
        Ty::Bf16 => bytes_of(&bf16_vec(v)),
    }
}
fn from_ty_bytes(b: &[u8], ty: Ty) -> Vec<f32> {
    match ty {
        Ty::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        Ty::F16 => b.chunks_exact(2).map(|c| f16_to_f64(u16::from_le_bytes([c[0], c[1]])) as f32).collect(),
        Ty::Bf16 => b.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
    }
}

/// Everything a site produces for `rows` rows: partials, post, comb, pre, collapsed/normed output.
struct SiteOut {
    part: Vec<f32>,
    post: Vec<f32>,
    comb: Vec<f32>,
    pre: Vec<f32>,
    out: Vec<u8>,
}

fn run_site(
    site: &MhcSite,
    x: &[f32],
    rows: usize,
    norm_w: Option<&[f32]>,
    out_ty: Ty,
    split: impl AsTune,
    st: &Stream,
) -> SiteOut {
    let xd = up(x);
    let nw = norm_w.map(up);
    let part = poisoned(rows * mhc::PART_FLOATS_PER_ROW * 4, st);
    let post = poisoned(rows * 16, st);
    let comb = poisoned(rows * 64, st);
    let pre = poisoned(rows * 16, st);
    let out = poisoned(rows * HIDDEN * out_ty.size(), st);
    unsafe {
        mhc::partials(fp(&xd), site, fm(&part), rows, split.tune(), st).unwrap();
        mhc::finalize(
            fp(&xd),
            fp(&part),
            site,
            nw.as_ref().map(fp),
            fm(&post),
            fm(&comb),
            Some(fm(&pre)),
            out.ptr(),
            out_ty,
            rows,
            &CFG,
            NORM_EPS,
            st,
        )
        .unwrap();
    }
    st.sync().unwrap();
    SiteOut { part: dl(&part), post: dl(&post), comb: dl(&comb), pre: dl(&pre), out: dl::<u8>(&out) }
}

fn run_apply(x: &[f32], y: &[f32], y_ty: Ty, post: &[f32], comb: &[f32], rows: usize, st: &Stream) -> Vec<f32> {
    let xd = up(x);
    let yd = up(&ty_bytes(y, y_ty));
    let (pd, cd) = (up(post), up(comb));
    unsafe { mhc::apply(fm(&xd), vp(&yd), y_ty, fp(&pd), fp(&cd), rows, st).unwrap() };
    st.sync().unwrap();
    dl(&xd)
}

fn rows_of<T: Copy>(v: &[T], w: usize, r: usize) -> &[T] {
    &v[r * w..(r + 1) * w]
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn ulp32(a: f32, b: f32) -> u32 {
    let ord = |f: f32| -> i64 {
        let u = f.to_bits();
        if u & 0x8000_0000 != 0 {
            -((u & 0x7FFF_FFFF) as i64)
        } else {
            u as i64
        }
    };
    (ord(a) - ord(b)).unsigned_abs() as u32
}

// ================================================================================================ tests

/// The partials, the pre/post/comb tail, the collapse, the norm, the apply and the mean against CPU mirrors with the kernel's order.
/// Everything without a transcendental is compared bit for bit; the tail is compared to the f32 mirror in ulps.
#[test]
fn mirrors_match_bitwise() {
    gpu!();
    let st = Stream::new().unwrap();
    let site = make_site(11);
    let mut rng = Rng(5);
    let rows = 5;
    let x = synth_streams(&mut rng, rows, 3.0);
    let nw: Vec<f32> = (0..HIDDEN).map(|_| rng.uniform(0.5, 1.5)).collect();
    let o = run_site(&site.dev, &x, rows, Some(&nw), Ty::F32, 1, &st);
    let o_plain = run_site(&site.dev, &x, rows, None, Ty::F32, 1, &st);
    for r in 0..rows {
        let xr = rows_of(&x, ROW, r);
        // K1: bit-exact
        let want = row_partials(xr, &site.fnw);
        assert!(bits_eq(rows_of(&o.part, 1024, r), &want), "row {r}: partials differ from the mirror");
        // tail: f32 mirror within a few ulps (CUDA expf vs libm expf)
        let s = reduce_atoms(&want);
        let t = tail32(&s, &site.base, &site.scale, &CFG);
        let mut worst = 0u32;
        for h in 0..4 {
            worst = worst.max(ulp32(o.pre[r * 4 + h], t.pre[h])).max(ulp32(o.post[r * 4 + h], t.post[h]));
        }
        for k in 0..16 {
            worst = worst.max(ulp32(o.comb[r * 16 + k], t.comb[k]));
        }
        eprintln!("row {r}: tail max ulp vs f32 mirror = {worst}");
        assert!(worst <= 64, "row {r}: tail differs from the f32 mirror by {worst} ulp");
        // collapse given the GPU's own pre: bit-exact; norm: bit-exact given the collapsed row
        let pre: [f32; 4] = o.pre[r * 4..r * 4 + 4].try_into().unwrap();
        let c = collapse32(xr, &pre);
        assert!(bits_eq(rows_of(&o_plain.out_f32(), HIDDEN, r), &c), "row {r}: collapsed row differs from the mirror");
        let n = norm32(&c, &nw, NORM_EPS);
        assert!(bits_eq(rows_of(&o.out_f32(), HIDDEN, r), &n), "row {r}: normed row differs from the mirror");
    }
    // apply and mean: bit-exact
    let y = synth_rows(&mut rng, rows, 2.0);
    let post: Vec<f32> = (0..rows * 4).map(|_| rng.uniform(0.0, 2.0)).collect();
    let comb: Vec<f32> = (0..rows * 16).map(|_| rng.uniform(0.0, 1.0)).collect();
    let got = run_apply(&x, &y, Ty::F32, &post, &comb, rows, &st);
    for r in 0..rows {
        let want = apply32(rows_of(&x, ROW, r), rows_of(&y, HIDDEN, r), rows_of(&post, 4, r), rows_of(&comb, 16, r));
        assert!(bits_eq(rows_of(&got, ROW, r), &want), "row {r}: apply differs from the mirror");
    }
    let xd = up(&x);
    let md = poisoned(rows * HIDDEN * 4, &st);
    unsafe { mhc::mean(fp(&xd), None, md.ptr(), Ty::F32, rows, NORM_EPS, &st).unwrap() };
    st.sync().unwrap();
    let got: Vec<f32> = dl(&md);
    for r in 0..rows {
        assert!(bits_eq(rows_of(&got, HIDDEN, r), &mean32(rows_of(&x, ROW, r))), "row {r}: mean differs from the mirror");
    }
    // the head: mean then the model's final RMSNorm in one pass (it feeds lm_head and the MTP head), bit-exact with the same
    // thread-ownership order as the finalize's norm. Two amplitudes: at rms 3 (with massive channels) the epsilon is below the
    // rounding of the mean square and cannot show, at rms 0.01 it moves the result by several percent.
    let nwd = up(&nw);
    for amp in [3.0f32, 0.01] {
        let xs = synth_streams(&mut rng, rows, amp);
        let xsd = up(&xs);
        let md = poisoned(rows * HIDDEN * 4, &st);
        unsafe { mhc::mean(fp(&xsd), Some(fp(&nwd)), md.ptr(), Ty::F32, rows, NORM_EPS, &st).unwrap() };
        st.sync().unwrap();
        let got: Vec<f32> = dl(&md);
        for r in 0..rows {
            let want = norm32(&mean32(rows_of(&xs, ROW, r)), &nw, NORM_EPS);
            assert!(bits_eq(rows_of(&got, HIDDEN, r), &want), "amp {amp} row {r}: mean + norm differs from the mirror");
        }
    }
}

impl SiteOut {
    fn out_f32(&self) -> Vec<f32> {
        from_ty_bytes(&self.out, Ty::F32)
    }
}

/// Output types: f16 and bf16 outputs are the single rounding of the f32 result (collapsed and normed).
#[test]
fn output_types_round_once() {
    gpu!();
    let st = Stream::new().unwrap();
    let site = make_site(12);
    let mut rng = Rng(6);
    let rows = 3;
    let x = synth_streams(&mut rng, rows, 2.0);
    let nw: Vec<f32> = (0..HIDDEN).map(|_| rng.uniform(0.5, 1.5)).collect();
    for norm in [None, Some(&nw[..])] {
        let f = run_site(&site.dev, &x, rows, norm, Ty::F32, 1, &st).out_f32();
        let h = from_ty_bytes(&run_site(&site.dev, &x, rows, norm, Ty::F16, 1, &st).out, Ty::F16);
        let b = from_ty_bytes(&run_site(&site.dev, &x, rows, norm, Ty::Bf16, 1, &st).out, Ty::Bf16);
        let want_h: Vec<f32> = f.iter().map(|&v| f16_to_f64(f64_to_f16(v as f64)) as f32).collect();
        let want_b: Vec<f32> = f.iter().map(|&v| round_bf16(v)).collect();
        assert!(bits_eq(&h, &want_h), "f16 output (norm {}) is not the rounding of the f32 output", norm.is_some());
        assert!(bits_eq(&b, &want_b), "bf16 output (norm {}) is not the rounding of the f32 output", norm.is_some());
    }
}

/// THE contract: every kernel's row r is bit-identical for any row count (1..=128 and beyond) and any position in the batch.
#[test]
fn row_invariance_all_kernels() {
    gpu!();
    let st = Stream::new().unwrap();
    let site = make_site(21);
    let site2 = make_site(22);
    let mut rng = Rng(7);
    const N: usize = 128;
    let x = synth_streams(&mut rng, N, 4.0);
    let y = synth_rows(&mut rng, N, 3.0);
    let nw: Vec<f32> = (0..HIDDEN).map(|_| rng.uniform(0.5, 1.5)).collect();
    let post: Vec<f32> = (0..N * 4).map(|_| rng.uniform(0.0, 2.0)).collect();
    let comb: Vec<f32> = (0..N * 16).map(|_| rng.uniform(0.0, 1.0)).collect();
    // references: one row at a time
    let one: Vec<SiteOut> = (0..N).map(|r| run_site(&site.dev, rows_of(&x, ROW, r), 1, Some(&nw), Ty::F16, 1, &st)).collect();
    let one_plain: Vec<SiteOut> = (0..N).map(|r| run_site(&site.dev, rows_of(&x, ROW, r), 1, None, Ty::Bf16, 1, &st)).collect();
    let one_apply: Vec<Vec<f32>> = (0..N)
        .map(|r| {
            run_apply(rows_of(&x, ROW, r), rows_of(&y, HIDDEN, r), Ty::Bf16, rows_of(&post, 4, r), rows_of(&comb, 16, r), 1, &st)
        })
        .collect();
    let one_fused: Vec<(Vec<f32>, Vec<f32>)> = (0..N)
        .map(|r| {
            run_apply_partials(
                rows_of(&x, ROW, r),
                rows_of(&y, HIDDEN, r),
                Ty::F16,
                rows_of(&post, 4, r),
                rows_of(&comb, 16, r),
                &site2.dev,
                1,
                1,
                &st,
            )
        })
        .collect();
    let one_mean: Vec<Vec<f32>> = (0..N).map(|r| run_mean(rows_of(&x, ROW, r), Some(&nw), Ty::F16, 1, &st)).collect();
    let one_expand: Vec<Vec<f32>> = (0..N).map(|r| run_expand(rows_of(&y, HIDDEN, r), Ty::Bf16, 1, &st)).collect();
    let mut checked = 0usize;
    for rows in (1..=N).chain([129, 255, 256, 300, 513]) {
        // data for rows beyond N: repeat rows cyclically, at a shifted position
        let idx = |r: usize| (r * 7 + rows) % N;
        let xs: Vec<f32> = (0..rows).flat_map(|r| rows_of(&x, ROW, idx(r)).to_vec()).collect();
        let ys: Vec<f32> = (0..rows).flat_map(|r| rows_of(&y, HIDDEN, idx(r)).to_vec()).collect();
        let ps: Vec<f32> = (0..rows).flat_map(|r| rows_of(&post, 4, idx(r)).to_vec()).collect();
        let cs: Vec<f32> = (0..rows).flat_map(|r| rows_of(&comb, 16, idx(r)).to_vec()).collect();
        let split = Tune { split: [0usize, 1, 2, 4, 8][rows % 5], rows_per_block: [0usize, 1, 2, 4][rows % 4] };
        let o = run_site(&site.dev, &xs, rows, Some(&nw), Ty::F16, split, &st);
        let op = run_site(&site.dev, &xs, rows, None, Ty::Bf16, split, &st);
        let ap = run_apply(&xs, &ys, Ty::Bf16, &ps, &cs, rows, &st);
        let fu = run_apply_partials(&xs, &ys, Ty::F16, &ps, &cs, &site2.dev, rows, split, &st);
        let mn = run_mean(&xs, Some(&nw), Ty::F16, rows, &st);
        let ex = run_expand(&ys, Ty::Bf16, rows, &st);
        for r in 0..rows {
            let k = idx(r);
            assert!(bits_eq(rows_of(&o.part, 1024, r), &one[k].part), "partials: rows={rows} r={r} differs from the 1-row run");
            assert!(bits_eq(rows_of(&o.post, 4, r), &one[k].post), "post: rows={rows} r={r}");
            assert!(bits_eq(rows_of(&o.comb, 16, r), &one[k].comb), "comb: rows={rows} r={r}");
            assert!(bits_eq(rows_of(&o.pre, 4, r), &one[k].pre), "pre: rows={rows} r={r}");
            assert_eq!(rows_of(&o.out, HIDDEN * 2, r), &one[k].out[..], "normed f16 out: rows={rows} r={r}");
            assert_eq!(rows_of(&op.out, HIDDEN * 2, r), &one_plain[k].out[..], "collapsed bf16 out: rows={rows} r={r}");
            assert!(bits_eq(rows_of(&ap, ROW, r), &one_apply[k]), "apply: rows={rows} r={r}");
            assert!(bits_eq(rows_of(&fu.0, ROW, r), &one_fused[k].0), "apply_partials x: rows={rows} r={r}");
            assert!(bits_eq(rows_of(&fu.1, 1024, r), &one_fused[k].1), "apply_partials part: rows={rows} r={r}");
            assert!(bits_eq(rows_of(&mn, HIDDEN, r), &one_mean[k]), "mean: rows={rows} r={r}");
            assert!(bits_eq(rows_of(&ex, ROW, r), &one_expand[k]), "expand: rows={rows} r={r}");
            checked += 1;
        }
    }
    eprintln!("row invariance: {checked} (rows, row) pairs bit-identical to the 1-row runs, for 6 kernels x all row counts 1..=128, 129, 255, 256, 300, 513");
}

fn run_apply_partials(
    x: &[f32],
    y: &[f32],
    y_ty: Ty,
    post: &[f32],
    comb: &[f32],
    next: &MhcSite,
    rows: usize,
    split: impl AsTune,
    st: &Stream,
) -> (Vec<f32>, Vec<f32>) {
    let xd = up(x);
    let yd = up(&ty_bytes(y, y_ty));
    let (pd, cd) = (up(post), up(comb));
    let part = poisoned(rows * mhc::PART_FLOATS_PER_ROW * 4, st);
    unsafe { mhc::apply_partials(fm(&xd), vp(&yd), y_ty, fp(&pd), fp(&cd), next, fm(&part), rows, split.tune(), st).unwrap() };
    st.sync().unwrap();
    (dl(&xd), dl(&part))
}

fn run_mean(x: &[f32], nw: Option<&[f32]>, out_ty: Ty, rows: usize, st: &Stream) -> Vec<f32> {
    let xd = up(x);
    let w = nw.map(up);
    let md = poisoned(rows * HIDDEN * out_ty.size(), st);
    unsafe { mhc::mean(fp(&xd), w.as_ref().map(fp), md.ptr(), out_ty, rows, NORM_EPS, st).unwrap() };
    st.sync().unwrap();
    from_ty_bytes(&dl::<u8>(&md), out_ty)
}

fn run_expand(y: &[f32], in_ty: Ty, rows: usize, st: &Stream) -> Vec<f32> {
    let yd = up(&ty_bytes(y, in_ty));
    let xd = poisoned(rows * ROW * 4, st);
    unsafe { mhc::expand(vp(&yd), in_ty, fm(&xd), rows, st).unwrap() };
    st.sync().unwrap();
    dl(&xd)
}

/// apply_partials == apply then partials, bit for bit; expand writes four identical streams equal to the widened input.
#[test]
fn fused_equals_unfused_and_expand() {
    gpu!();
    let st = Stream::new().unwrap();
    let site = make_site(31);
    let mut rng = Rng(8);
    for rows in [1usize, 3, 17, 64] {
        let x = synth_streams(&mut rng, rows, 2.0);
        let y = synth_rows(&mut rng, rows, 1.0);
        let post: Vec<f32> = (0..rows * 4).map(|_| rng.uniform(0.0, 2.0)).collect();
        let comb: Vec<f32> = (0..rows * 16).map(|_| rng.uniform(0.0, 1.0)).collect();
        for y_ty in [Ty::F32, Ty::F16, Ty::Bf16] {
            let (fx, fpart) = run_apply_partials(&x, &y, y_ty, &post, &comb, &site.dev, rows, 2, &st);
            let ux = run_apply(&x, &y, y_ty, &post, &comb, rows, &st);
            let uo = run_site(&site.dev, &ux, rows, None, Ty::F32, 1, &st);
            assert!(bits_eq(&fx, &ux), "rows {rows} {y_ty:?}: fused x differs from apply");
            assert!(bits_eq(&fpart, &uo.part), "rows {rows} {y_ty:?}: fused partials differ from partials of the applied x");
            // against the mirror with y widened exactly
            let yw = from_ty_bytes(&ty_bytes(&y, y_ty), y_ty);
            let want = apply32(rows_of(&x, ROW, 0), rows_of(&yw, HIDDEN, 0), rows_of(&post, 4, 0), rows_of(&comb, 16, 0));
            assert!(bits_eq(rows_of(&ux, ROW, 0), &want), "rows {rows} {y_ty:?}: apply differs from the mirror");
        }
    }
    for in_ty in [Ty::F32, Ty::F16, Ty::Bf16] {
        let y = synth_rows(&mut rng, 7, 5.0);
        let yw = from_ty_bytes(&ty_bytes(&y, in_ty), in_ty);
        let x = run_expand(&y, in_ty, 7, &st);
        for r in 0..7 {
            for h in 0..4 {
                assert!(
                    bits_eq(&x[r * ROW + h * HIDDEN..r * ROW + (h + 1) * HIDDEN], rows_of(&yw, HIDDEN, r)),
                    "{in_ty:?} r={r} h={h}"
                );
            }
        }
    }
    // the bf16 embedding-table form, including an out-of-range id
    let vocab = 50usize;
    let table: Vec<u16> = bf16_vec(&synth_rows(&mut rng, vocab, 1.0));
    let ids = [3i32, 49, -1, 50, 0];
    let (td, idd) = (up(&table), up(&ids));
    let xd = poisoned(5 * ROW * 4, &st);
    unsafe { mhc::embed_expand(vp(&td), vp(&idd), fm(&xd), 5, vocab, &st).unwrap() };
    st.sync().unwrap();
    let x: Vec<f32> = dl(&xd);
    for (r, &id) in ids.iter().enumerate() {
        for h in 0..4 {
            for d in 0..HIDDEN {
                let want = if (0..vocab as i32).contains(&id) { bf16_to_f32(table[id as usize * HIDDEN + d]) } else { 0.0 };
                assert_eq!(x[r * ROW + h * HIDDEN + d].to_bits(), want.to_bits(), "embed_expand r={r} h={h} d={d}");
            }
        }
    }
}

/// The scheduling knobs of the partials kernel (m-split, rows per block, and their automatic choice) never change a bit: every
/// combination gives the same partials, with and without the fused apply, for row counts that are not multiples of the tile.
#[test]
fn tuning_knobs_do_not_change_bits() {
    gpu!();
    let st = Stream::new().unwrap();
    let site = make_site(35);
    let mut rng = Rng(14);
    for rows in [1usize, 2, 3, 5, 9, 17, 25, 40, 49, 100] {
        let x = synth_streams(&mut rng, rows, 2.0);
        let y = synth_rows(&mut rng, rows, 1.0);
        let post: Vec<f32> = (0..rows * 4).map(|_| rng.uniform(0.0, 2.0)).collect();
        let comb: Vec<f32> = (0..rows * 16).map(|_| rng.uniform(0.0, 1.0)).collect();
        let t1 = Tune { split: 1, rows_per_block: 1 };
        let base = run_site(&site.dev, &x, rows, None, Ty::F32, t1, &st);
        let (bx, bp) = run_apply_partials(&x, &y, Ty::Bf16, &post, &comb, &site.dev, rows, t1, &st);
        for split in [0usize, 1, 2, 4, 8] {
            for rb in [0usize, 1, 2, 4] {
                let t = Tune { split, rows_per_block: rb };
                let o = run_site(&site.dev, &x, rows, None, Ty::F32, t, &st);
                assert!(bits_eq(&o.part, &base.part), "rows {rows} {t:?}: partials differ from split 1 / rb 1");
                let (fx, fpart) = run_apply_partials(&x, &y, Ty::Bf16, &post, &comb, &site.dev, rows, t, &st);
                assert!(
                    bits_eq(&fx, &bx) && bits_eq(&fpart, &bp),
                    "rows {rows} {t:?}: apply_partials differs from split 1 / rb 1"
                );
            }
        }
    }
}

/// fn stored as f32, f16 or bf16 gives the same bits (the products are the same widened values); a lossy narrow type is refused.
#[test]
fn fn_storage_types_agree_and_lossy_is_refused() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(9);
    let fnw = synth_fn(&mut rng);
    let (base, scale) = (synth_base(&mut rng), synth_scale(&mut rng));
    let x = synth_streams(&mut rng, 4, 2.0);
    // every storage type under every kernel variant (the 8-warp split loads its weights early through a different code path)
    let sites: Vec<MhcSite> =
        [Ty::F32, Ty::F16, Ty::Bf16].iter().map(|&t| MhcSite::from_f32_as(&fnw, &base, &scale, t).unwrap()).collect();
    let want = run_site(&sites[0], &x, 4, None, Ty::F32, 1, &st);
    for (k, site) in sites.iter().enumerate() {
        for split in [0usize, 1, 2, 4, 8] {
            for rb in [1usize, 2] {
                let o = run_site(site, &x, 4, None, Ty::F32, Tune { split, rows_per_block: rb }, &st);
                assert!(bits_eq(&want.part, &o.part), "partials differ: storage type {k}, split {split}, rb {rb}");
                assert!(bits_eq(&want.comb, &o.comb) && bits_eq(&want.post, &o.post), "post/comb differ: storage type {k}");
            }
        }
    }
    // a value that needs more than 8 mantissa bits: bf16 refused; one that needs more than 11: fp16 refused
    let mut lossy = fnw.clone();
    lossy[5] = 0.1234567;
    assert!(MhcSite::from_f32_as(&lossy, &base, &scale, Ty::Bf16).is_err());
    assert!(MhcSite::from_f32_as(&lossy, &base, &scale, Ty::F16).is_err());
    assert_eq!(MhcSite::from_f32(&lossy, &base, &scale).unwrap().fn_ty, Ty::F32, "an inexact fn falls back to f32, never rounds");
    lossy[5] = 1.0 + 1.0 / 256.0; // 1 + 2^-8: exact in fp16 (10 mantissa bits), not in bf16 (7)
    assert!(MhcSite::from_f32_as(&lossy, &base, &scale, Ty::F16).is_ok());
    assert!(MhcSite::from_f32_as(&lossy, &base, &scale, Ty::Bf16).is_err());
    let mut nan = fnw.clone();
    nan[0] = f32::NAN;
    assert!(MhcSite::from_f32(&nan, &base, &scale).is_err());
    assert!(MhcSite::from_f32(&fnw[..100], &base, &scale).is_err());
}

/// Bad arguments are errors, never faults: misaligned pointers and out-of-range row counts; zero rows is a no-op.
#[test]
fn bad_arguments_are_errors() {
    gpu!();
    let st = Stream::new().unwrap();
    let site = make_site(41);
    let x = up(&vec![0f32; 2 * ROW + 4]);
    let part = up(&vec![0f32; 2 * 1024]);
    unsafe {
        let mis = (x.ptr() as *const f32).add(1);
        assert!(mhc::partials(mis, &site.dev, fm(&part), 1, Tune::AUTO, &st).is_err());
        assert!(mhc::partials(fp(&x), &site.dev, fm(&part), mhc::MAX_ROWS + 1, Tune::AUTO, &st).is_err());
        assert!(mhc::partials(fp(&x), &site.dev, fm(&part), 0, Tune::AUTO, &st).is_ok());
        let out = up(&vec![0u16; HIDDEN]);
        let post = up(&[0f32; 4]);
        let comb = up(&[0f32; 16]);
        let bad_iters = MhcCfg { sinkhorn_iters: 0, ..CFG };
        assert!(mhc::finalize(
            fp(&x),
            fp(&part),
            &site.dev,
            None,
            fm(&post),
            fm(&comb),
            None,
            out.ptr(),
            Ty::F16,
            1,
            &bad_iters,
            NORM_EPS,
            &st
        )
        .is_err());
        let mis_out = (out.ptr() as *mut u8).add(2) as *mut c_void;
        assert!(mhc::finalize(
            fp(&x),
            fp(&part),
            &site.dev,
            None,
            fm(&post),
            fm(&comb),
            None,
            mis_out,
            Ty::F16,
            1,
            &CFG,
            NORM_EPS,
            &st
        )
        .is_err());
    }
    st.sync().unwrap();
}

/// Sinkhorn output: columns sum to 1 (the last step) and rows to 1 within the 20-iteration convergence.
#[test]
fn comb_is_nearly_doubly_stochastic() {
    gpu!();
    let st = Stream::new().unwrap();
    let site = make_site(51);
    let mut rng = Rng(10);
    let x = synth_streams(&mut rng, 16, 3.0);
    let o = run_site(&site.dev, &x, 16, None, Ty::F32, 1, &st);
    for r in 0..16 {
        let c = rows_of(&o.comb, 16, r);
        for i in 0..4 {
            let rs: f32 = (0..4).map(|j| c[i * 4 + j]).sum();
            let cs: f32 = (0..4).map(|j| c[j * 4 + i]).sum();
            assert!((cs - 1.0).abs() < 1e-5 && (rs - 1.0).abs() < 1e-2, "row {r}: sums {rs} {cs}");
        }
        assert!(c.iter().all(|v| *v > 0.0 && *v < 1.0));
    }
}

// ================================================================================================ accuracy

fn rel_rms(got: &[f64], want: &[f64]) -> f64 {
    let (mut e, mut w) = (0f64, 0f64);
    for (g, t) in got.iter().zip(want) {
        e += (g - t) * (g - t);
        w += t * t;
    }
    (e / w.max(1e-300)).sqrt()
}

/// Errors of one candidate (this crate's kernels or one oracle configuration) against the float64 pipeline.
#[derive(Clone, Copy, Default, Debug)]
struct Err {
    post_abs: f64,
    post_rel: f64,
    comb_abs: f64,
    col_rms: f64,
    xa_rms: f64,
}

impl Err {
    fn max(self, o: Err) -> Err {
        Err {
            post_abs: self.post_abs.max(o.post_abs),
            post_rel: self.post_rel.max(o.post_rel),
            comb_abs: self.comb_abs.max(o.comb_abs),
            col_rms: self.col_rms.max(o.col_rms),
            xa_rms: self.xa_rms.max(o.xa_rms),
        }
    }
    fn line(&self) -> String {
        format!(
            "post abs {:.2e} rel {:.2e} | comb abs {:.2e} | collapsed relRMS {:.2e} | applied relRMS {:.2e}",
            self.post_abs, self.post_rel, self.comb_abs, self.col_rms, self.xa_rms
        )
    }
}

/// One candidate's outputs for `rows` rows (f32).
struct Cand {
    post: Vec<f32>,
    comb: Vec<f32>,
    col: Vec<f32>,
    xa: Vec<f32>,
}

struct Refs {
    r: Vec<Ref64>,
    xa: Vec<Vec<f64>>,
}

fn make_refs(x: &[f32], y: &[f32], fnw: &[f32], base: &[f32], scale: &[f32], rows: usize) -> Refs {
    let r: Vec<Ref64> = (0..rows).map(|i| ref64(rows_of(x, ROW, i), fnw, base, scale, &CFG)).collect();
    let xa = (0..rows).map(|i| apply64(rows_of(x, ROW, i), rows_of(y, HIDDEN, i), &r[i].post, &r[i].comb)).collect();
    Refs { r, xa }
}

fn err_vs_ref(c: &Cand, refs: &Refs, rows: usize) -> Err {
    let mut e = Err::default();
    for r in 0..rows {
        let rf = &refs.r[r];
        for h in 0..4 {
            let d = (c.post[r * 4 + h] as f64 - rf.post[h]).abs();
            e.post_abs = e.post_abs.max(d);
            e.post_rel = e.post_rel.max(d / rf.post[h].abs());
        }
        for k in 0..16 {
            e.comb_abs = e.comb_abs.max((c.comb[r * 16 + k] as f64 - rf.comb[k]).abs());
        }
        let col: Vec<f64> = rows_of(&c.col, HIDDEN, r).iter().map(|&v| v as f64).collect();
        e.col_rms = e.col_rms.max(rel_rms(&col, &rf.collapsed));
        let xa: Vec<f64> = rows_of(&c.xa, ROW, r).iter().map(|&v| v as f64).collect();
        e.xa_rms = e.xa_rms.max(rel_rms(&xa, &refs.xa[r]));
    }
    e
}

/// Distance between two candidates (same metrics, one standing in for the reference).
fn diff(a: &Cand, b: &Cand, rows: usize) -> Err {
    let mut e = Err::default();
    for r in 0..rows {
        for h in 0..4 {
            let d = (a.post[r * 4 + h] as f64 - b.post[r * 4 + h] as f64).abs();
            e.post_abs = e.post_abs.max(d);
            e.post_rel = e.post_rel.max(d / (b.post[r * 4 + h] as f64).abs());
        }
        for k in 0..16 {
            e.comb_abs = e.comb_abs.max((a.comb[r * 16 + k] as f64 - b.comb[r * 16 + k] as f64).abs());
        }
        let f = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
        e.col_rms = e.col_rms.max(rel_rms(&f(rows_of(&a.col, HIDDEN, r)), &f(rows_of(&b.col, HIDDEN, r))));
        e.xa_rms = e.xa_rms.max(rel_rms(&f(rows_of(&a.xa, ROW, r)), &f(rows_of(&b.xa, ROW, r))));
    }
    e
}

/// This crate's kernels on `rows` rows: partials + finalize (plain collapsed, f32) and apply with its own post/comb and an f32 y.
fn rexl3_cand(site: &MhcSite, x: &[f32], y: &[f32], rows: usize, st: &Stream) -> Cand {
    let o = run_site(site, x, rows, None, Ty::F32, 1, st);
    let xa = run_apply(x, y, Ty::F32, &o.post, &o.comb, rows, st);
    Cand { post: o.post.clone(), comb: o.comb.clone(), col: o.out_f32(), xa }
}

/// Always-on accuracy test: synthetic weights and streams at three amplitudes against the float64 pipeline.
#[test]
fn accuracy_vs_f64_synthetic() {
    gpu!();
    let st = Stream::new().unwrap();
    let site = make_site(61);
    let mut rng = Rng(12);
    let rows = 24;
    let mut worst = Err::default();
    for amp in [0.02f32, 2.0, 40.0] {
        let x = synth_streams(&mut rng, rows, amp);
        let y = synth_rows(&mut rng, rows, 1.5);
        let refs = make_refs(&x, &y, &site.fnw, &site.base, &site.scale, rows);
        let c = rexl3_cand(&site.dev, &x, &y, rows, &st);
        let e = err_vs_ref(&c, &refs, rows);
        // pre (the collapse weights; ~1e-6 + sigmoid) and the fused RMSNorm output against float64
        let nw: Vec<f32> = (0..HIDDEN).map(|_| rng.uniform(0.5, 1.5)).collect();
        let o = run_site(&site.dev, &x, rows, Some(&nw), Ty::F32, Tune::AUTO, &st);
        let normed = o.out_f32();
        let (mut pre_abs, mut norm_rms) = (0f64, 0f64);
        for r in 0..rows {
            for h in 0..4 {
                pre_abs = pre_abs.max((o.pre[r * 4 + h] as f64 - refs.r[r].pre[h]).abs());
            }
            let got: Vec<f64> = rows_of(&normed, HIDDEN, r).iter().map(|&v| v as f64).collect();
            norm_rms = norm_rms.max(rel_rms(&got, &rmsnorm64(&refs.r[r].collapsed, &nw, NORM_EPS as f64)));
        }
        eprintln!("synthetic amp {amp:>5}: {} | pre abs {pre_abs:.2e} | normed relRMS {norm_rms:.2e}", e.line());
        assert!(pre_abs < 5e-7 && norm_rms < 5e-7, "pre {pre_abs:e} / normed {norm_rms:e}");
        worst = worst.max(e);
    }
    assert!(worst.post_abs < 5e-7 && worst.comb_abs < 5e-7, "post/comb error {worst:?}");
    assert!(worst.col_rms < 5e-7 && worst.xa_rms < 3e-7, "collapsed/applied error {worst:?}");
}

fn glm_pack() -> Option<gb_exl3::checkpoint::Checkpoint> {
    let d = std::env::var("REXL3_GLM_DIR").ok()?;
    gb_exl3::checkpoint::scan(std::path::Path::new(&d), false).ok()
}

fn host_tensor(ck: &gb_exl3::checkpoint::Checkpoint, key: &str) -> Vec<f32> {
    let e = ck.plain.get(key).unwrap_or_else(|| panic!("{key} not in the pack"));
    let b = ck.read_tensor(e).unwrap();
    match e.dtype.as_str() {
        "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "BF16" => b.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        d => panic!("{key}: {d}"),
    }
}

/// Every `hc_*_fn` tensor of the real pack is exactly representable in fp16 and bf16, so the fp16 storage costs no bits.
#[test]
fn real_pack_fn_is_lossless_in_f16() {
    let Some(ck) = glm_pack() else { return eprintln!("REXL3_GLM_DIR not configured: real-pack mHC checks not run") };
    let mut n = 0;
    for l in 0..45 {
        for site in ["attn", "ffn"] {
            let f = host_tensor(&ck, &format!("model.language_model.layers.{l}.hc_{site}_fn"));
            let (h, b) = mhc::lossless_types(&f);
            assert!(h && b, "layer {l} {site}: fn is not exact in fp16/bf16 (fp16 {h}, bf16 {b})");
            n += 1;
        }
    }
    eprintln!("{n} hc fn tensors, all exactly representable in fp16 and bf16");
}

fn npy(dir: &std::path::Path, name: &str) -> Npy {
    read_npy(&dir.join(name))
}

/// Accuracy on the real pack's layer 3 and layer 40 tensors against the float64 pipeline and against the ExLlamaV3 fork's
/// HyperConnection in its three row-count configurations (dumps from `tools/dump_mhc_reference.py`).
#[test]
fn accuracy_vs_oracle_real_layers() {
    gpu!();
    let (Some(ck), Ok(refdir)) = (glm_pack(), std::env::var("REXL3_MHC_REF_DIR")) else {
        return eprintln!("REXL3_GLM_DIR / REXL3_MHC_REF_DIR not configured: oracle comparison not run");
    };
    let refdir = std::path::PathBuf::from(refdir);
    let st = Stream::new().unwrap();
    let rows = 64usize;
    let mut tot_rexl3 = Err::default();
    let mut tot_oracle = Err::default();
    let mut tot_vs = Err::default();
    let mut tot_drift = Err::default();
    let mut apply_bits = (0usize, 0usize);
    let mut mean_ulp = 0u32;
    for case in ["embed", "mid", "late"] {
        let x: Vec<f32> = npy(&refdir, &format!("x_{case}.npy")).to_f32();
        let y: Vec<f32> = npy(&refdir, &format!("y_{case}.npy")).to_f32();
        assert_eq!(x.len(), rows * ROW);
        // the head mean against torch's mean(dim=2)
        let ours = run_mean(&x, None, Ty::F32, rows, &st);
        let want = npy(&refdir, &format!("mean_{case}.npy")).to_f32();
        for (g, w) in ours.iter().zip(&want) {
            mean_ulp = mean_ulp.max(ulp32(*g, *w));
        }
        for layer in [3usize, 40] {
            for site in ["attn", "ffn"] {
                let prefix = format!("model.language_model.layers.{layer}.hc_{site}");
                let (fnw, base, scale) = (
                    host_tensor(&ck, &format!("{prefix}_fn")),
                    host_tensor(&ck, &format!("{prefix}_base")),
                    host_tensor(&ck, &format!("{prefix}_scale")),
                );
                let dev = MhcSite::load(&ck, &prefix).unwrap();
                assert_eq!(dev.fn_ty, Ty::F16);
                let refs = make_refs(&x, &y, &fnw, &base, &scale, rows);
                let ours = rexl3_cand(&dev, &x, &y, rows, &st);
                let e_ours = err_vs_ref(&ours, &refs, rows);
                let mut e_orc = Err::default();
                let mut cands = Vec::new();
                for cfg in ["r1", "r8", "r64"] {
                    let f = |n: &str| npy(&refdir, &format!("L{layer}_{site}_{case}_{cfg}_{n}.npy")).to_f32();
                    let c = Cand { post: f("post"), comb: f("comb"), col: f("col"), xa: f("xa") };
                    let e = err_vs_ref(&c, &refs, rows);
                    e_orc = e_orc.max(e);
                    eprintln!("L{layer} {site:<4} {case:<5} oracle {cfg:<3} vs f64: {}", e.line());
                    tot_vs = tot_vs.max(diff(&ours, &c, rows));
                    cands.push(c);
                }
                tot_drift = tot_drift.max(diff(&cands[0], &cands[2], rows));
                eprintln!("L{layer} {site:<4} {case:<5} REXL3      vs f64: {}", e_ours.line());
                tot_rexl3 = tot_rexl3.max(e_ours);
                tot_oracle = tot_oracle.max(e_orc);
                // apply is a bit-exact op: REXL3's apply on the ORACLE's post/comb and y against the oracle's own apply
                let c = &cands[1];
                let got = run_apply(&x, &y, Ty::F32, &c.post, &c.comb, rows, &st);
                apply_bits.0 += got.iter().zip(&c.xa).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                apply_bits.1 += got.len();
            }
        }
    }
    eprintln!("REXL3 worst vs f64 (all layers/sites/cases): {}", tot_rexl3.line());
    eprintln!("oracle worst vs f64 (r1, r8, r64)         : {}", tot_oracle.line());
    eprintln!("REXL3 vs oracle worst                     : {}", tot_vs.line());
    eprintln!("oracle r1 vs oracle r64 drift worst       : {}", tot_drift.line());
    eprintln!("apply with the oracle's post/comb: {} of {} elements differ from the oracle's apply", apply_bits.0, apply_bits.1);
    eprintln!("stream mean vs torch mean(dim=2): max {mean_ulp} ulp");
    // The bound (everything is at the fp32 rounding floor: post <= 2 has a ulp of 2.4e-7, comb < 1 has 6e-8..1.2e-7):
    //   post abs <= 1e-6 (rel <= 2e-6), comb abs <= 5e-7, collapsed (fp32) relRMS <= 1e-6, applied relRMS <= 3e-7,
    // and never more than 2x the worst oracle configuration (r1/r8/r64) against float64. Apply on identical inputs is bit-exact.
    assert!(
        tot_rexl3.post_abs <= 2.0 * tot_oracle.post_abs && tot_rexl3.comb_abs <= 2.0 * tot_oracle.comb_abs,
        "post/comb worse than 2x the oracle"
    );
    assert!(tot_rexl3.xa_rms <= 2.0 * tot_oracle.xa_rms, "applied streams worse than 2x the oracle");
    assert!(tot_rexl3.post_abs < 1e-6 && tot_rexl3.post_rel < 2e-6 && tot_rexl3.comb_abs < 5e-7, "post/comb absolute bound");
    assert!(tot_rexl3.col_rms < 1e-6 && tot_rexl3.xa_rms < 3e-7, "collapsed/applied bound");
    assert_eq!(apply_bits.0, 0, "apply on the oracle's own post/comb must equal the oracle's apply bit for bit");
    assert!(mean_ulp <= 1, "head mean");
}

// ================================================================================================ goldens

/// Approximately normal from four uniforms with exact IEEE operations only (no libm), so the golden inputs are the same bits on every
/// machine; the kernels' own transcendentals are part of what the golden pins.
fn gnormal(rng: &mut Rng) -> f32 {
    (rng.uniform(0.0, 1.0) + rng.uniform(0.0, 1.0) + rng.uniform(0.0, 1.0) + rng.uniform(0.0, 1.0) - 2.0) * 1.7320508
}

fn gold_site(seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng(seed);
    let fnw = (0..NFN * ROW)
        .map(|_| {
            let v = bf16_to_f32(f32_to_bf16(gnormal(&mut rng) * 0.02));
            if v.abs() < 6.2e-5 {
                0.0
            } else {
                v
            }
        })
        .collect();
    let base = (0..NFN).map(|i| if i < 8 { rng.uniform(-8.0, 1.0) } else { rng.uniform(-3.0, 3.0) }).collect();
    let scale = (0..3).map(|_| rng.uniform(0.05, 0.6)).collect();
    (fnw, base, scale)
}

fn gold_streams(rng: &mut Rng, rows: usize, amp: f32) -> Vec<f32> {
    let chan: Vec<f32> = (0..HIDDEN).map(|_| 0.4 + 1.6 * rng.uniform(0.0, 1.0)).collect();
    let mut x = vec![0f32; rows * ROW];
    for r in 0..rows {
        let big = [rng.next_u64() as usize % HIDDEN, rng.next_u64() as usize % HIDDEN, rng.next_u64() as usize % HIDDEN];
        for h in 0..4 {
            for d in 0..HIDDEN {
                x[r * ROW + h * HIDDEN + d] = gnormal(rng) * chan[d] * amp;
            }
            for &b in &big {
                x[r * ROW + h * HIDDEN + b] = rng.uniform(100.0, 1500.0) * amp * if b % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
    }
    x
}

/// Hashes of every kernel's output for a spread of row counts and two stream amplitudes on seeded synthetic inputs (the weights are the
/// fp16-exact kind, one site per fn storage type). A change that must keep every bit diffs this file against one written by the old build;
/// a different nvcc, a different expf or a changed reduction order is caught by name.
///
///     REXL3_GOLDEN_OUT=testdata/gpu/golden_mhc.txt cargo test --release -p gb-cuda --test mhc golden_mhc_hashes -- --ignored
#[test]
#[ignore]
fn golden_mhc_hashes() {
    use std::io::Write;
    gpu!();
    let path = std::env::var("REXL3_GOLDEN_OUT").expect("REXL3_GOLDEN_OUT");
    let mut f = std::fs::File::create(path).unwrap();
    let st = Stream::new().unwrap();
    let (fnw, base, scale) = gold_site(77);
    let (fnw2, base2, scale2) = gold_site(78);
    let sites: Vec<(Ty, MhcSite)> = vec![
        (Ty::F16, MhcSite::from_f32_as(&fnw, &base, &scale, Ty::F16).unwrap()),
        (Ty::Bf16, MhcSite::from_f32_as(&fnw2, &base2, &scale2, Ty::Bf16).unwrap()),
        (Ty::F32, MhcSite::from_f32_as(&fnw2, &base2, &scale2, Ty::F32).unwrap()),
    ];
    let next = MhcSite::from_f32_as(&fnw, &base, &scale, Ty::F16).unwrap();
    for rows in [1usize, 2, 3, 8, 15, 16, 17, 32, 33, 64, 127, 128, 300] {
        for amp in [0.05f32, 8.0] {
            let mut rng = Rng(1000 + rows as u64 * 7 + amp.to_bits() as u64);
            let x = gold_streams(&mut rng, rows, amp);
            let y: Vec<f32> = (0..rows * HIDDEN).map(|_| gnormal(&mut rng) * 2.0).collect();
            let nw: Vec<f32> = (0..HIDDEN).map(|_| rng.uniform(0.5, 1.5)).collect();
            let post_in: Vec<f32> = (0..rows * 4).map(|_| rng.uniform(0.0, 2.0)).collect();
            let comb_in: Vec<f32> = (0..rows * 16).map(|_| rng.uniform(0.0, 1.0)).collect();
            for (k, (ty, site)) in sites.iter().enumerate() {
                let o = run_site(site, &x, rows, Some(&nw), Ty::F16, Tune::AUTO, &st);
                let op = run_site(site, &x, rows, None, Ty::Bf16, Tune::AUTO, &st);
                writeln!(
                    f,
                    "rows={rows} amp={amp} fn={ty:?} partials {:016x} post {:016x} comb {:016x} pre {:016x} normed_f16 {:016x} collapsed_bf16 {:016x}",
                    fnv(&bytes_of(&o.part)),
                    fnv(&bytes_of(&o.post)),
                    fnv(&bytes_of(&o.comb)),
                    fnv(&bytes_of(&o.pre)),
                    fnv(&o.out),
                    fnv(&op.out)
                )
                .unwrap();
                if k == 0 {
                    let ap = run_apply(&x, &y, Ty::Bf16, &post_in, &comb_in, rows, &st);
                    let (fx, fpart) = run_apply_partials(&x, &y, Ty::F16, &post_in, &comb_in, &next, rows, Tune::AUTO, &st);
                    let mn = run_mean(&x, Some(&nw), Ty::F16, rows, &st);
                    let mp = run_mean(&x, None, Ty::F32, rows, &st);
                    let ex = run_expand(&y, Ty::Bf16, rows, &st);
                    writeln!(
                        f,
                        "rows={rows} amp={amp} apply_bf16 {:016x} apply_partials_x {:016x} apply_partials_part {:016x} mean_normed_f16 {:016x} mean_f32 {:016x} expand_bf16 {:016x}",
                        fnv(&bytes_of(&ap)),
                        fnv(&bytes_of(&fx)),
                        fnv(&bytes_of(&fpart)),
                        fnv(&bytes_of(&mn)),
                        fnv(&bytes_of(&mp)),
                        fnv(&bytes_of(&ex))
                    )
                    .unwrap();
                }
            }
        }
    }
}

// ================================================================================================ benchmarks (ignored)

mod cu {
    use core::ffi::c_void;
    extern "C" {
        pub fn cudaEventCreate(e: *mut *mut c_void) -> i32;
        pub fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> i32;
        pub fn cudaEventSynchronize(e: *mut c_void) -> i32;
        pub fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> i32;
        pub fn cudaStreamBeginCapture(s: *mut c_void, mode: i32) -> i32;
        pub fn cudaStreamEndCapture(s: *mut c_void, g: *mut *mut c_void) -> i32;
        pub fn cudaGraphInstantiate(e: *mut *mut c_void, g: *mut c_void, flags: u64) -> i32;
        pub fn cudaGraphLaunch(e: *mut c_void, s: *mut c_void) -> i32;
    }
}

/// Mean milliseconds of one call of `f`, which enqueues work on `st`: `iters` back-to-back calls between two events.
fn time_ms(st: &Stream, iters: usize, mut f: impl FnMut()) -> f64 {
    unsafe {
        let (mut a, mut b) = (core::ptr::null_mut(), core::ptr::null_mut());
        assert_eq!(cu::cudaEventCreate(&mut a), 0);
        assert_eq!(cu::cudaEventCreate(&mut b), 0);
        for _ in 0..3 {
            f();
        }
        st.sync().unwrap();
        cu::cudaEventRecord(a, st.raw());
        for _ in 0..iters {
            f();
        }
        cu::cudaEventRecord(b, st.raw());
        cu::cudaEventSynchronize(b);
        let mut ms = 0f32;
        cu::cudaEventElapsedTime(&mut ms, a, b);
        ms as f64 / iters as f64
    }
}

/// Capture `f` (a sequence of launches on `st`) into a CUDA graph and return the mean milliseconds of one replay.
fn graph_ms(st: &Stream, iters: usize, f: impl Fn()) -> f64 {
    unsafe {
        let (mut g, mut e) = (core::ptr::null_mut(), core::ptr::null_mut());
        assert_eq!(cu::cudaStreamBeginCapture(st.raw(), 2), 0);
        f();
        assert_eq!(cu::cudaStreamEndCapture(st.raw(), &mut g), 0);
        assert_eq!(cu::cudaGraphInstantiate(&mut e, g, 0), 0);
        time_ms(st, iters, || assert_eq!(cu::cudaGraphLaunch(e, st.raw()), 0))
    }
}

/// (split, rows per block) combinations a sweep tries; `REXL3_BENCH_TUNES=auto` limits it to the automatic choice.
fn tunes() -> Vec<(usize, usize)> {
    if std::env::var("REXL3_BENCH_TUNES").is_ok_and(|v| v == "auto") {
        return vec![(0, 0)];
    }
    let mut v = Vec::new();
    for rb in [1usize, 2, 4] {
        for split in [1usize, 2, 4, 8] {
            v.push((split, rb));
        }
    }
    v.push((0, 0));
    v
}

/// Mean milliseconds of one call of `f` with the calls captured 100 at a time into a CUDA graph (what an engine that graphs its step sees:
/// kernel durations without the stream-launch gaps of `time_ms`).
fn time_graph(st: &Stream, iters: usize, mut f: impl FnMut()) -> f64 {
    let per = 100usize;
    unsafe {
        let (mut g, mut e) = (core::ptr::null_mut(), core::ptr::null_mut());
        assert_eq!(cu::cudaStreamBeginCapture(st.raw(), 2), 0);
        for _ in 0..per {
            f();
        }
        assert_eq!(cu::cudaStreamEndCapture(st.raw(), &mut g), 0);
        assert_eq!(cu::cudaGraphInstantiate(&mut e, g, 0), 0);
        time_ms(st, (iters / per).max(3), || assert_eq!(cu::cudaGraphLaunch(e, st.raw()), 0)) / per as f64
    }
}

fn bench_rows() -> Vec<usize> {
    match std::env::var("REXL3_BENCH_ROWS") {
        Ok(v) => v.split(',').map(|x| x.trim().parse().unwrap()).collect(),
        Err(_) => vec![1, 2, 4, 8, 16, 32, 64, 128, 512, 4096],
    }
}

/// Per-kernel time and achieved GB/s (bytes counted explicitly, weights included). The 90 sites rotate so the weights stream
/// from DRAM as in a real step (90 x 0.75 MiB = 67 MiB, more than the L2); activation traffic of large row counts is DRAM too.
#[test]
#[ignore]
fn bench_mhc_kernels() {
    gpu!();
    let st = Stream::new().unwrap();
    let bcfg = MhcCfg { sinkhorn_iters: std::env::var("REXL3_BENCH_ITERS").map_or(20, |v| v.parse().unwrap()), ..CFG };
    let mut rng = Rng(3);
    let full: Vec<Site> = (0..90).map(|i| make_site(1000 + i as u64)).collect();
    // the same weights stored as f32 (what a checkpoint whose fn is not exactly fp16 would cost: twice the dominant decode read)
    let sites32: Vec<MhcSite> = full.iter().map(|s| MhcSite::from_f32_as(&s.fnw, &s.base, &s.scale, Ty::F32).unwrap()).collect();
    let sites: Vec<MhcSite> = full.into_iter().map(|s| s.dev).collect();
    let fn_bytes = (NFN * ROW * 2) as f64;
    let nw = up(&(0..HIDDEN).map(|_| rng.uniform(0.5, 1.5)).collect::<Vec<f32>>());
    eprintln!("{:<26} {:>5} {:>9} {:>9} {:>9}", "kernel", "rows", "us", "MB", "GB/s");
    for rows in bench_rows() {
        let xd = up(&synth_streams(&mut rng, rows.min(256), 3.0).iter().cycle().take(rows * ROW).cloned().collect::<Vec<f32>>());
        let yd = up(&vec![0u16; rows * HIDDEN]);
        let yf = up(&vec![0f32; rows * HIDDEN]);
        let ids = up(&vec![5i32; rows]);
        let table = up(&bf16_vec(&synth_rows(&mut rng, 16, 1.0)));
        let (post, comb) = (up(&vec![0.5f32; rows * 4]), up(&vec![0.25f32; rows * 16]));
        let w = MhcWork::new(rows).unwrap();
        let out16 = up(&vec![0u16; rows * HIDDEN]);
        let out32 = up(&vec![0f32; rows * HIDDEN]);
        let xb = (rows * ROW * 4) as f64;
        let pb = (rows * mhc::PART_FLOATS_PER_ROW * 4) as f64;
        let n = if rows >= 512 { 30 } else { 300 };
        let report = |name: &str, bytes: f64, ms: f64| {
            eprintln!("{name:<26} {rows:>5} {:>9.2} {:>9.2} {:>9.1}", ms * 1e3, bytes / 1e6, bytes / (ms * 1e-3) / 1e9);
        };
        let mut k = 0usize;
        let ms = time_graph(&st, n, || {
            k += 1;
            unsafe { mhc::partials(fp(&xd), &sites32[k % 90], fm(&w.part), rows, Tune::AUTO, &st).unwrap() }
        });
        report("partials fn=f32 (auto)", 2.0 * fn_bytes + xb + pb, ms);
        for (split, rb) in tunes() {
            let t = Tune { split, rows_per_block: rb };
            let ms = time_graph(&st, n, || {
                k += 1;
                unsafe { mhc::partials(fp(&xd), &sites[k % 90], fm(&w.part), rows, t, &st).unwrap() }
            });
            report(&format!("partials s={split} rb={rb}"), fn_bytes + xb + pb, ms);
        }
        let ms = time_graph(&st, n, || {
            k += 1;
            unsafe {
                mhc::finalize(
                    fp(&xd),
                    fp(&w.part),
                    &sites[k % 90],
                    Some(fp(&nw)),
                    fm(&w.post),
                    fm(&w.comb),
                    None,
                    out16.ptr(),
                    Ty::Bf16,
                    rows,
                    &bcfg,
                    NORM_EPS,
                    &st,
                )
                .unwrap()
            }
        });
        report("finalize+norm bf16", xb + pb + (rows * HIDDEN * 2) as f64, ms);
        let ms = time_graph(&st, n, || {
            k += 1;
            unsafe {
                mhc::finalize(
                    fp(&xd),
                    fp(&w.part),
                    &sites[k % 90],
                    None,
                    fm(&w.post),
                    fm(&w.comb),
                    None,
                    out32.ptr(),
                    Ty::F32,
                    rows,
                    &bcfg,
                    NORM_EPS,
                    &st,
                )
                .unwrap()
            }
        });
        report("finalize plain f32", xb + pb + (rows * HIDDEN * 4) as f64, ms);
        let ms =
            time_graph(&st, n, || unsafe { mhc::apply(fm(&xd), vp(&yd), Ty::Bf16, fp(&post), fp(&comb), rows, &st).unwrap() });
        report("apply (y bf16)", 2.0 * xb + (rows * HIDDEN * 2) as f64, ms);
        let ms =
            time_graph(&st, n, || unsafe { mhc::apply(fm(&xd), vp(&yf), Ty::F32, fp(&post), fp(&comb), rows, &st).unwrap() });
        report("apply (y f32)", 2.0 * xb + (rows * HIDDEN * 4) as f64, ms);
        for (split, rb) in tunes() {
            let t = Tune { split, rows_per_block: rb };
            let ms = time_graph(&st, n, || {
                k += 1;
                unsafe {
                    mhc::apply_partials(
                        fm(&xd),
                        vp(&yd),
                        Ty::Bf16,
                        fp(&post),
                        fp(&comb),
                        &sites[k % 90],
                        fm(&w.part),
                        rows,
                        t,
                        &st,
                    )
                    .unwrap()
                }
            });
            report(&format!("apply+partials s={split} rb={rb}"), fn_bytes + 2.0 * xb + pb + (rows * HIDDEN * 2) as f64, ms);
        }
        let ms = time_graph(&st, n, || unsafe {
            mhc::mean(fp(&xd), Some(fp(&nw)), out16.ptr(), Ty::Bf16, rows, NORM_EPS, &st).unwrap()
        });
        report("mean+norm bf16", xb + (rows * HIDDEN * 2) as f64, ms);
        let ms = time_graph(&st, n, || unsafe { mhc::embed_expand(vp(&table), vp(&ids), fm(&xd), rows, 16, &st).unwrap() });
        report("embed_expand", xb + (rows * HIDDEN * 2) as f64 / 4.0, ms);
    }
}

/// `apply_partials_finalize` in slabs: the bits of `apply_partials` over every row followed by `finalize` over every row (streams, post,
/// comb, the normed output), for slabs that divide the rows and slabs that leave a remainder; and its time against the two whole calls at
/// a prompt chunk's rows (`--ignored --nocapture`: REXL3_BENCH_SLABS=1 prints it).
#[test]
fn slabbed_apply_partials_finalize_equals_the_whole_calls() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(17);
    let (cur, next) = (make_site(31), make_site(32));
    let nw = up(&(0..HIDDEN).map(|_| rng.uniform(0.5, 1.5)).collect::<Vec<f32>>());
    let bench = std::env::var("REXL3_BENCH_SLABS").is_ok();
    for rows in if bench { vec![300usize, 2048, 4096] } else { vec![300usize] } {
        let x0 = synth_streams(&mut rng, rows.min(512), 3.0).iter().cycle().take(rows * ROW).cloned().collect::<Vec<f32>>();
        let y = up(&synth_rows(&mut rng, rows, 1.0));
        let w0 = MhcWork::new(rows).unwrap();
        let h0 = up(&vec![0u16; rows * HIDDEN]);
        let x_init = up(&x0);
        // a previous site's post and comb for every row
        unsafe {
            mhc::partials(fp(&x_init), &cur.dev, fm(&w0.part), rows, Tune::AUTO, &st).unwrap();
            mhc::finalize(fp(&x_init), fp(&w0.part), &cur.dev, None, fm(&w0.post), fm(&w0.comb), None, h0.ptr(), Ty::F16, rows, &CFG, NORM_EPS, &st)
                .unwrap();
        }
        let (post0, comb0): (Vec<f32>, Vec<f32>) = (w0.post.to_vec().unwrap(), w0.comb.to_vec().unwrap());
        let run = |slab: Option<usize>| {
            let (xd, mut w, out) = (up(&x0), MhcWork::new(rows).unwrap(), up(&vec![0u16; rows * HIDDEN]));
            w.post.upload(&post0).unwrap();
            w.comb.upload(&comb0).unwrap();
            let go = || unsafe {
                match slab {
                    None => {
                        mhc::apply_partials(fm(&xd), vp(&y), Ty::F32, fp(&w.post), fp(&w.comb), &next.dev, fm(&w.part), rows, Tune::AUTO, &st)
                            .unwrap();
                        mhc::finalize(fp(&xd), fp(&w.part), &next.dev, Some(fp(&nw)), fm(&w.post), fm(&w.comb), None, out.ptr(), Ty::F16, rows, &CFG, NORM_EPS, &st)
                            .unwrap();
                    }
                    Some(s) => mhc::apply_partials_finalize(
                        fm(&xd),
                        vp(&y),
                        Ty::F32,
                        fm(&w.post),
                        fm(&w.comb),
                        &next.dev,
                        fm(&w.part),
                        Some(fp(&nw)),
                        out.ptr(),
                        Ty::F16,
                        rows,
                        s,
                        &CFG,
                        NORM_EPS,
                        &st,
                    )
                    .unwrap(),
                }
            };
            go();
            st.sync().unwrap();
            let got = (xd.to_vec::<f32>().unwrap(), w.post.to_vec::<f32>().unwrap(), w.comb.to_vec::<f32>().unwrap(), out.to_vec::<u16>().unwrap());
            let mut ms = 0.0;
            if bench {
                // (the streams keep moving: every call applies another update; only the time is read)
                let t = std::time::Instant::now();
                for _ in 0..20 {
                    go();
                }
                st.sync().unwrap();
                ms = t.elapsed().as_secs_f64() * 1e3 / 20.0;
            }
            (got, ms)
        };
        let (want, ms_whole) = run(None);
        for slab in [64usize, 100, 128, 192, 256, 320, 384, 512] {
            let (got, ms) = run(Some(slab));
            assert!(bits_eq(&got.0, &want.0), "streams, {rows} rows, slab {slab}");
            assert!(bits_eq(&got.1, &want.1) && bits_eq(&got.2, &want.2), "post / comb, {rows} rows, slab {slab}");
            assert_eq!(got.3, want.3, "normed output, {rows} rows, slab {slab}");
            if bench {
                eprintln!("apply+partials+finalize {rows} rows: whole {ms_whole:.3} ms, slabs of {slab} {ms:.3} ms");
            }
        }
    }
}

/// One forward's mHC chain for 45 layers x 2 sites in a CUDA graph (activations between sites omitted), per variant:
///   unfused : partials, finalize (collapsed bf16), rmsnorm (glue), apply          = 4 launches per site, 8 per layer (the oracle's count)
///   norm    : partials, finalize + norm, apply                                      = 3 per site, 6 per layer
///   fused   : finalize + norm, apply + next partials                               = 2 per site, 4 per layer
#[test]
#[ignore]
fn bench_mhc_step_chain() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(4);
    let sites: Vec<MhcSite> = (0..90).map(|i| make_site(2000 + i as u64).dev).collect();
    let nw = up(&(0..HIDDEN).map(|_| rng.uniform(0.5, 1.5)).collect::<Vec<f32>>());
    let wbytes: f64 = sites.iter().map(|s| s.weight_bytes() as f64).sum();
    eprintln!("{:<8} {:>5} {:>9} {:>9} {:>9} {:>10}", "variant", "rows", "launches", "us/step", "MB/step", "GB/s");
    for rows in bench_rows().into_iter().filter(|&r| r <= 512) {
        let xd = up(&synth_streams(&mut rng, rows.min(128), 3.0).iter().cycle().take(rows * ROW).cloned().collect::<Vec<f32>>());
        let yd = up(&vec![0u16; rows * HIDDEN]);
        let (hid_b, hid_n) = (up(&vec![0u16; rows * HIDDEN]), up(&vec![0u16; rows * HIDDEN]));
        let w = MhcWork::new(rows).unwrap();
        let xb = (rows * ROW * 4) as f64;
        let pb = (rows * mhc::PART_FLOATS_PER_ROW * 4) as f64;
        let hb = (rows * HIDDEN * 2) as f64;
        let n = if rows >= 512 { 10 } else { 100 };
        for variant in ["unfused", "norm", "fused"] {
            let ms = graph_ms(&st, n, || unsafe {
                if variant == "fused" {
                    mhc::partials(fp(&xd), &sites[0], fm(&w.part), rows, Tune::AUTO, &st).unwrap();
                }
                for i in 0..90 {
                    let s = &sites[i];
                    match variant {
                        "unfused" => {
                            mhc::partials(fp(&xd), s, fm(&w.part), rows, Tune::AUTO, &st).unwrap();
                            mhc::finalize(
                                fp(&xd),
                                fp(&w.part),
                                s,
                                None,
                                fm(&w.post),
                                fm(&w.comb),
                                None,
                                hid_b.ptr(),
                                Ty::Bf16,
                                rows,
                                &CFG,
                                NORM_EPS,
                                &st,
                            )
                            .unwrap();
                            gb_cuda::glue::rmsnorm(vp(&hid_b), vp(&nw), hid_n.ptr(), rows, HIDDEN, NORM_EPS, &st).unwrap();
                            mhc::apply(fm(&xd), vp(&yd), Ty::Bf16, fp(&w.post), fp(&w.comb), rows, &st).unwrap();
                        }
                        "norm" => {
                            mhc::partials(fp(&xd), s, fm(&w.part), rows, Tune::AUTO, &st).unwrap();
                            mhc::finalize(
                                fp(&xd),
                                fp(&w.part),
                                s,
                                Some(fp(&nw)),
                                fm(&w.post),
                                fm(&w.comb),
                                None,
                                hid_n.ptr(),
                                Ty::Bf16,
                                rows,
                                &CFG,
                                NORM_EPS,
                                &st,
                            )
                            .unwrap();
                            mhc::apply(fm(&xd), vp(&yd), Ty::Bf16, fp(&w.post), fp(&w.comb), rows, &st).unwrap();
                        }
                        _ => {
                            mhc::finalize(
                                fp(&xd),
                                fp(&w.part),
                                s,
                                Some(fp(&nw)),
                                fm(&w.post),
                                fm(&w.comb),
                                None,
                                hid_n.ptr(),
                                Ty::Bf16,
                                rows,
                                &CFG,
                                NORM_EPS,
                                &st,
                            )
                            .unwrap();
                            let next = &sites[(i + 1) % 90];
                            mhc::apply_partials(
                                fm(&xd),
                                vp(&yd),
                                Ty::Bf16,
                                fp(&w.post),
                                fp(&w.comb),
                                next,
                                fm(&w.part),
                                rows,
                                Tune::AUTO,
                                &st,
                            )
                            .unwrap();
                        }
                    }
                }
            });
            // bytes per site besides the weights: unfused 4 x + 2 part + 4 hidden, norm 4 x + 2 part + 2 hidden, fused 3 x + 2 part + 2 hidden
            let (launches, per_site) = match variant {
                "unfused" => (360, 4.0 * xb + 2.0 * pb + 4.0 * hb),
                "norm" => (270, 4.0 * xb + 2.0 * pb + 2.0 * hb),
                _ => (181, 3.0 * xb + 2.0 * pb + 2.0 * hb),
            };
            let bytes = wbytes + 90.0 * per_site;
            eprintln!(
                "{variant:<8} {rows:>5} {launches:>9} {:>9.1} {:>9.2} {:>10.1}",
                ms * 1e3,
                bytes / 1e6,
                bytes / (ms * 1e-3) / 1e9
            );
        }
    }
}

/// Lane E: the mHC chain (partials, finalize, the apply fused with the next site's partials, finalize) launched with the programmatic-dependent-
/// launch attribute (mask 2) and captured into a CUDA graph on a non-blocking stream, then replayed: every result is bit-equal to the same chain
/// launched eagerly without the attribute (mask 0). The graph must capture (PDL kernels in a graph) and replay.
#[test]
fn pdl_graph_of_the_mhc_chain_equals_the_eager_chain_bit_for_bit() {
    use gb_cuda::graph::Graph;
    gpu!();
    let st = Stream::new_nonblocking().unwrap();
    let (a, b) = (make_site(940), make_site(941));
    for rows in [1usize, 2, 5] {
        let mut rng = Rng(950 + rows as u64);
        let x0 = synth_streams(&mut rng, rows, 1.0);
        let y0: Vec<f32> = synth_streams(&mut rng, rows, 1.0)[..rows * HIDDEN].to_vec();
        let nw: Vec<f32> = (0..HIDDEN).map(|i| 0.5 + (i % 7) as f32 * 0.125).collect();
        let run = |graph: bool| -> Vec<Vec<f32>> {
            let (x, y, nwd) = (up(&x0), up(&y0), up(&nw));
            let part_a = poisoned(rows * mhc::PART_FLOATS_PER_ROW * 4, &st);
            let part_b = poisoned(rows * mhc::PART_FLOATS_PER_ROW * 4, &st);
            let (post_a, comb_a) = (poisoned(rows * 16, &st), poisoned(rows * 64, &st));
            let (post_b, comb_b) = (poisoned(rows * 16, &st), poisoned(rows * 64, &st));
            let (out_a, out_b) = (poisoned(rows * HIDDEN * 4, &st), poisoned(rows * HIDDEN * 4, &st));
            // `st` is non-blocking: it does not wait for the legacy-stream `up` copies (st.sync() above does not either)
            gb_cuda::rt::device_sync().unwrap();
            let chain = || -> Result<(), gb_cuda::CudaError> {
                unsafe {
                    mhc::partials(fp(&x), &a.dev, fm(&part_a), rows, Tune::AUTO, &st)?;
                    mhc::finalize(
                        fp(&x),
                        fp(&part_a),
                        &a.dev,
                        Some(fp(&nwd)),
                        fm(&post_a),
                        fm(&comb_a),
                        None,
                        out_a.ptr(),
                        Ty::F32,
                        rows,
                        &CFG,
                        NORM_EPS,
                        &st,
                    )?;
                    mhc::apply_partials(
                        fm(&x),
                        vp(&y),
                        Ty::F32,
                        fp(&post_a),
                        fp(&comb_a),
                        &b.dev,
                        fm(&part_b),
                        rows,
                        Tune::AUTO,
                        &st,
                    )?;
                    mhc::finalize(
                        fp(&x),
                        fp(&part_b),
                        &b.dev,
                        None,
                        fm(&post_b),
                        fm(&comb_b),
                        None,
                        out_b.ptr(),
                        Ty::F32,
                        rows,
                        &CFG,
                        NORM_EPS,
                        &st,
                    )?;
                }
                Ok(())
            };
            if graph {
                let g = Graph::capture(&st, chain).unwrap_or_else(|_| panic!("the mHC chain did not capture"));
                g.launch(&st).unwrap();
            } else {
                chain().unwrap();
            }
            st.sync().unwrap();
            vec![dl(&x), dl(&part_a), dl(&post_a), dl(&comb_a), dl(&out_a), dl(&part_b), dl(&post_b), dl(&comb_b), dl(&out_b)]
        };
        gb_cuda::exl3_linear::set_pdl_mask(0).unwrap();
        let eager = run(false);
        gb_cuda::exl3_linear::set_pdl_mask(2).unwrap();
        let eager_pdl = run(false);
        let graphed = run(true);
        gb_cuda::exl3_linear::set_pdl_mask(0).unwrap();
        for (i, name) in ["x", "part_a", "post_a", "comb_a", "out_a", "part_b", "post_b", "comb_b", "out_b"].iter().enumerate() {
            assert!(bits_eq(&eager[i], &eager_pdl[i]), "{rows} rows: {name}: PDL launches differ from plain launches");
            assert!(bits_eq(&eager[i], &graphed[i]), "{rows} rows: {name}: the PDL graph differs from the eager chain");
        }
    }
}
