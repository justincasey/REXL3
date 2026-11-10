//! Synthetic low-rank hyper-connection reference, batch invariance and hashes.
#![allow(clippy::needless_range_loop)]
mod common;
use common::Rng;
use gb_cuda::{fn_mhc, hcx, DevBuf, Stream};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
const H: usize = 256;
const HC: usize = 2;
const R: usize = 32;
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
fn bff(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}
fn hf(x: f32) -> u16 {
    f64_to_f16(x as f64)
}
fn hff(x: u16) -> f32 {
    f16_to_f64(x) as f32
}
fn sig(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}
fn dl<T: Copy + Default>(b: &DevBuf, s: &Stream) -> Vec<T> {
    s.sync().unwrap();
    b.to_vec().unwrap()
}
fn cmp(a: &[f32], b: &[f32], tol: f64, name: &str) {
    assert_eq!(a.len(), b.len());
    let (e, n) = a.iter().zip(b).fold((0f64, 0f64), |(e, n), (&a, &b)| (e + ((a - b) as f64).powi(2), n + (b as f64).powi(2)));
    let r = (e / n.max(1e-30)).sqrt();
    assert!(r < tol, "{name}: rel_l2={r}, tol={tol}");
}
struct Rig {
    input: Vec<u16>,
    norm: Vec<u16>,
    down: Vec<u16>,
    up: Vec<u16>,
    inject: Vec<u16>,
}
fn rig_with(tmax: usize, h: usize, hc: usize, r: usize) -> Rig {
    let mut rng = Rng(0x35f029);
    Rig {
        input: (0..tmax * h).map(|_| bf(rng.uniform(-1., 1.))).collect(),
        norm: (0..hc * h).map(|_| hf(rng.uniform(-0.2, 0.2))).collect(),
        down: (0..r * hc * h).map(|_| hf(rng.uniform(-0.06, 0.06))).collect(),
        up: (0..hc * h * r).map(|_| hf(rng.uniform(-0.06, 0.06))).collect(),
        inject: (0..hc * hc * h).map(|_| hf(rng.uniform(-0.06, 0.06))).collect(),
    }
}
fn rig() -> Rig {
    rig_with(64, H, HC, R)
}
struct Out {
    expand: Vec<f32>,
    y: Vec<u16>,
    inj: Vec<f32>,
    head: Vec<u16>,
    hcx: Vec<u16>,
    hcx_inj: Vec<f32>,
    hcg: Vec<u16>,
    hcg_inj: Vec<f32>,
    post: Vec<f32>,
}
fn run_at(r: &Rig, t: usize, h: usize, hc: usize, rank: usize, st: &Stream) -> Out {
    let input = DevBuf::from_slice(&r.input).unwrap();
    let mut x = DevBuf::zeroed(t * hc * h * 4).unwrap();
    fn_mhc::hc_expand(&input, &mut x, h, hc, t, st).unwrap();
    let (norm, down, up, inject) = (
        DevBuf::from_slice(&r.norm).unwrap(),
        DevBuf::from_slice(&r.down).unwrap(),
        DevBuf::from_slice(&r.up).unwrap(),
        DevBuf::from_slice(&r.inject).unwrap(),
    );
    let mix = fn_mhc::Mixer { norm: &norm, down: &down, up: &up, inject: Some(&inject) };
    let head_mix = fn_mhc::Mixer { norm: &norm, down: &down, up: &up, inject: None };
    let (mut nx, mut low) = (DevBuf::zeroed(t * hc * h * 4).unwrap(), DevBuf::zeroed(t * rank * 4).unwrap());
    let (mut y, mut inj, mut head) =
        (DevBuf::zeroed(t * h * 2).unwrap(), DevBuf::zeroed(t * hc * 4).unwrap(), DevBuf::zeroed(t * h * 2).unwrap());
    fn_mhc::hc_pre(&x, &mix, fn_mhc::PreScratch { normed: &mut nx, low: &mut low }, &mut y, &mut inj, h, hc, rank, 1e-6, t, st)
        .unwrap();
    fn_mhc::hc_head(&x, &head_mix, fn_mhc::PreScratch { normed: &mut nx, low: &mut low }, &mut head, h, hc, rank, 1e-6, t, st)
        .unwrap();
    let (mut hy, mut hi, mut hs) = (
        DevBuf::zeroed(t * h * 2).unwrap(),
        DevBuf::zeroed(t * hc * 4).unwrap(),
        DevBuf::zeroed(hcx::scratch_bytes(hc, rank)).unwrap(),
    );
    hcx::pre(&x, &mix, &mut hy, Some(&mut hi), &mut hs, h, hc, rank, 1e-6, t, st).unwrap();
    let (mut gy, mut gi, mut gs) = (
        DevBuf::zeroed(t * h * 2).unwrap(),
        DevBuf::zeroed(t * hc * 4).unwrap(),
        DevBuf::zeroed(hcx::gemm_scratch_bytes(t, h, hc, rank)).unwrap(),
    );
    hcx::pre_gemm(&x, &mix, &mut gy, Some(&mut gi), &mut gs, h, hc, rank, 1e-6, t, st).unwrap();
    let mut post = DevBuf::zeroed(t * hc * h * 4).unwrap();
    fn_mhc::hc_post(&y, Some(&x), &inj, &mut post, h, hc, t, st).unwrap();
    Out {
        expand: dl(&x, st),
        y: dl(&y, st),
        inj: dl(&inj, st),
        head: dl(&head, st),
        hcx: dl(&hy, st),
        hcx_inj: dl(&hi, st),
        hcg: dl(&gy, st),
        hcg_inj: dl(&gi, st),
        post: dl(&post, st),
    }
}
fn run(r: &Rig, t: usize, st: &Stream) -> Out {
    run_at(r, t, H, HC, R, st)
}
fn cpu_at(r: &Rig, t: usize, h: usize, hc: usize, rank: usize) -> (Vec<u16>, Vec<f32>, Vec<f32>) {
    let mut ys = Vec::new();
    let mut ins = Vec::new();
    let mut posts = Vec::new();
    for row in 0..t {
        let mut nx = vec![0f32; hc * h];
        for s in 0..hc {
            let vals = &r.input[row * h..(row + 1) * h];
            let inv = (vals.iter().map(|v| bff(*v).powi(2)).sum::<f32>() / h as f32 + 1e-6).sqrt().recip();
            for d in 0..h {
                nx[s * h + d] = bff(vals[d]) * inv * (1. + hff(r.norm[s * h + d]));
            }
        }
        let mut lo = [0f32; 4096];
        for j in 0..rank {
            let dot = (0..hc * h).map(|i| nx[i] * hff(r.down[j * hc * h + i])).sum::<f32>() / hc as f32;
            lo[j] = dot * sig(dot);
        }
        let mut yr = Vec::new();
        for d in 0..h {
            let sum = (0..hc)
                .map(|s| {
                    let i = s * h + d;
                    let dot = (0..rank).map(|j| lo[j] * hff(r.up[i * rank + j])).sum::<f32>();
                    nx[i] * sig(dot)
                })
                .sum::<f32>();
            yr.push(bf(sum / hc as f32));
        }
        for s in 0..hc {
            let dot = (0..hc * h).map(|i| nx[i] * hff(r.inject[s * hc * h + i])).sum::<f32>() / hc as f32;
            ins.push(2. * sig(dot));
        }
        for s in 0..hc {
            for d in 0..h {
                posts.push(bff(r.input[row * h + d]) + bff(yr[d]) * ins[row * hc + s]);
            }
        }
        ys.extend(yr);
    }
    (ys, ins, posts)
}
fn cpu(r: &Rig, t: usize) -> (Vec<u16>, Vec<f32>, Vec<f32>) {
    cpu_at(r, t, H, HC, R)
}
fn fnv<T: Copy>(v: &[T]) -> u64 {
    let b = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) };
    b.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ u64::from(*x)).wrapping_mul(0x100000001b3))
}
#[test]
#[ignore]
fn low_rank_cpu_reference_and_row_invariance() {
    let Some(st) = gpu() else { return };
    let r = rig();
    let full = run(&r, 64, &st);
    let mut changed = rig();
    changed.input[H..].fill(bf(0.75));
    changed.input[36 * H..37 * H].copy_from_slice(&r.input[..H]);
    let mixed = run(&changed, 64, &st);
    for (name, a, b, stride) in [
        ("expand", &full.expand, &mixed.expand, HC * H),
        ("inject", &full.inj, &mixed.inj, HC),
        ("hcx inject", &full.hcx_inj, &mixed.hcx_inj, HC),
        ("hcg inject", &full.hcg_inj, &mixed.hcg_inj, HC),
        ("post", &full.post, &mixed.post, HC * H),
    ] {
        assert_eq!(&a[..stride], &b[..stride], "{name} other-row invariance");
        assert_eq!(&a[..stride], &b[36 * stride..37 * stride], "{name} position invariance");
    }
    for (name, a, b) in [
        ("pre", &full.y, &mixed.y),
        ("head", &full.head, &mixed.head),
        ("hcx", &full.hcx, &mixed.hcx),
        ("hcg", &full.hcg, &mixed.hcg),
    ] {
        assert_eq!(&a[..H], &b[..H], "{name} other-row invariance");
        assert_eq!(&a[..H], &b[36 * H..37 * H], "{name} position invariance");
    }

    for t in [1, 7, 37, 64] {
        let g = run(&r, t, &st);
        assert_eq!(g.expand, &full.expand[..t * HC * H], "expand row invariance");
        assert_eq!(g.y, &full.y[..t * H], "pre row invariance");
        assert_eq!(g.inj, &full.inj[..t * HC], "inject row invariance");
        assert_eq!(g.head, &full.head[..t * H], "head row invariance");
        assert_eq!(g.hcx, &full.hcx[..t * H], "hcx row invariance");
        assert_eq!(g.hcx_inj, &full.hcx_inj[..t * HC], "hcx inject row invariance");
        assert_eq!(g.hcg, &full.hcg[..t * H], "hcg row invariance");
        assert_eq!(g.hcg_inj, &full.hcg_inj[..t * HC], "hcg inject row invariance");
        assert_eq!(g.post, &full.post[..t * HC * H], "post row invariance");
        for row in 0..t {
            for s in 0..HC {
                for d in 0..H {
                    assert_eq!(g.expand[(row * HC + s) * H + d], bff(r.input[row * H + d]));
                }
            }
        }
        let (y, inj, post) = cpu(&r, t);
        let conv = |v: &[u16]| v.iter().map(|x| bff(*x)).collect::<Vec<_>>();
        cmp(&conv(&g.y), &conv(&y), 0.015, "mHC CPU y");
        cmp(&g.inj, &inj, 0.015, "mHC CPU inject");
        cmp(&conv(&g.head), &conv(&y), 0.015, "mHC head CPU y");
        cmp(&g.post, &post, 0.015, "mHC post CPU");
        cmp(&conv(&g.hcx), &conv(&y), 0.015, "HCX CPU y");
        cmp(&g.hcx_inj, &inj, 0.015, "HCX CPU inject");
        cmp(&conv(&g.hcg), &conv(&y), 0.025, "HCG CPU y");
        cmp(&g.hcg_inj, &inj, 0.025, "HCG CPU inject");
    }
}
#[test]
#[ignore]
fn golden_fn_mhc_hashes() {
    let Some(st) = gpu() else { return };
    let g = run(&rig(), 37, &st);
    let lines = [
        format!("expand {:016x}", fnv(&g.expand)),
        format!("pre {:016x}", fnv(&g.y)),
        format!("head {:016x}", fnv(&g.head)),
        format!("inject {:016x}", fnv(&g.inj)),
        format!("hcx {:016x}", fnv(&g.hcx)),
        format!("hcg {:016x}", fnv(&g.hcg)),
        format!("post {:016x}", fnv(&g.post)),
    ];
    std::fs::write(std::env::var("REXL3_GOLDEN_OUT").unwrap(), format!("{}\n", lines.join("\n"))).unwrap();
}

// The exact Flash-Next hyper-connection dims the model runs live (H=2560, hc=4, rank=320 routed + 4 inject columns).
// The hcg kernels read every weight operand solely at its declared rank stride; the inject weights are a separate
// [hc][hc*h] buffer (the old convention that spliced 4 trailing columns onto down/up and read them at stride 324 was
// retired: nothing relies on buffer adjacency). rank=320 is a multiple of 8, so hcx::pre (the fp32 reference) is
// callable here and the compensated hcg path can be compared against it directly.
const QH: usize = 2560;
const QHC: usize = 4;
const QR: usize = 320;

/// A rig whose input holds exactly the rows `[r0, r1)`, sharing the weight arrays with `base` (the chunk-independence
/// check slices one call's input into two calls and must feed each the same row data it saw in the single call).
fn rig_rows(base: &Rig, r0: usize, r1: usize, h: usize) -> Rig {
    Rig {
        input: base.input[r0 * h..r1 * h].to_vec(),
        norm: base.norm.clone(),
        down: base.down.clone(),
        up: base.up.clone(),
        inject: base.inject.clone(),
    }
}

/// hcg is the same HyperConnection pre-mix as hcx, with the two low-rank products as cuBLAS tensor-core GEMMs on fp16
/// operands. The operands are an fp16 hi+lo split of the fp32 values (Dekker pairing), so the incremental delta vs hcx
/// is the split residual (~2^-22) plus the GEMM summation order, not the ~2^-11 of a single fp16 rounding.
#[test]
#[ignore]
fn hcg_matches_hcx_within_compensated_fp16_budget_on_qfn_non_square_shapes() {
    let Some(st) = gpu() else { return };
    let r = rig_with(4096, QH, QHC, QR);
    // non-square shapes (K = hc*H != R, N = hc != R) and the 64-row pass boundaries of the fixed GEMM tiles
    for t in [1, 63, 64, 65, 127, 128, 129, 4096] {
        let g = run_at(&r, t, QH, QHC, QR, &st);
        let (y, inj, _) = cpu_at(&r, t, QH, QHC, QR);
        let conv = |v: &[u16]| v.iter().map(|x| bff(*x)).collect::<Vec<_>>();
        let report = |a: &[f32], b: &[f32], label: &str| {
            let n = a.len().min(b.len());
            let mut sum = 0f64;
            let mut sq = 0f64;
            let mut mx = 0f64;
            let mut dot = 0f64;
            let mut na = 0f64;
            let mut nb = 0f64;
            for i in 0..n {
                let x = a[i] as f64;
                let y = b[i] as f64;
                let d = (x - y).abs();
                let den = x.abs().max(1e-12);
                sum += d;
                sq += d * d;
                mx = mx.max(d / den);
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-300);
            eprintln!(
                "HCG {label} t={t}: n={n} mean_abs={:.3e} rmse={:.3e} max_rel={:.3e} cos={:.6}",
                sum / n as f64,
                (sq / n as f64).sqrt(),
                mx,
                cos
            );
        };
        report(&conv(&g.hcg), &conv(&y), "vs CPU y");
        report(&g.hcg_inj, &inj, "vs CPU inject");
        report(&conv(&g.hcg), &conv(&g.hcx), "vs HCX y");
        report(&g.hcg_inj, &g.hcx_inj, "vs HCX inject");
        // hcg vs the CPU oracle: the compensated delta plus reduction order stays within the accepted grouped-MoE budget
        cmp(&conv(&g.hcg), &conv(&y), 0.025, "HCG QFN CPU y");
        cmp(&g.hcg_inj, &inj, 0.025, "HCG QFN CPU inject");
        // the incremental delta vs hcx: the hi+lo split residual and the GEMM's summation order (fp32 accumulate). The
        // uncompensated fp16 path measured ~1e-3 relative here; the compensated path must stay two orders below that.
        cmp(&conv(&g.hcg), &conv(&g.hcx), 2e-4, "HCG vs HCX QFN y");
        cmp(&g.hcg_inj, &g.hcx_inj, 2e-4, "HCG vs HCX QFN inject");
    }
}

/// hcg is deterministic (a fixed 64-row tile and a per-shape cuBLAS pick give one bit pattern) and chunk-independent: a
/// row's output bits are the same whether it ran in a 1300-row call or in a separate 64-row call.
#[test]
#[ignore]
fn hcg_is_deterministic_and_chunk_independent() {
    let Some(st) = gpu() else { return };
    let r = rig_with(2048, H, HC, R);
    // determinism: two identical calls are bit-equal
    let d1 = run(&r, 1300, &st);
    let d2 = run(&r, 1300, &st);
    let eq_y = d1.hcg.iter().zip(&d2.hcg).filter(|(a, b)| a == b).count();
    let eq_inj = d1.hcg_inj.iter().zip(&d2.hcg_inj).filter(|(a, b)| a == b).count();
    eprintln!("hcg determinism y bit-equal {}/{} rows, inject {}/{}", eq_y, d1.hcg.len() / H, eq_inj, d1.hcg_inj.len() / HC);
    assert_eq!(d1.hcg, d2.hcg, "hcg determinism y");
    assert_eq!(d1.hcg_inj, d2.hcg_inj, "hcg determinism inject");
    // chunk independence: rows 0..700 and 700..1300, run as separate calls, concatenate to the single 1300-row call,
    // bit-for-bit, across the 64-row tile boundary at row 704
    let full = run(&r, 1300, &st);
    let a = run_at(&rig_rows(&r, 0, 700, H), 700, H, HC, R, &st);
    let b = run_at(&rig_rows(&r, 700, 1300, H), 600, H, HC, R, &st);
    assert_eq!(a.hcg, full.hcg[..700 * H], "hcg chunk 0..700");
    assert_eq!(b.hcg, full.hcg[700 * H..1300 * H], "hcg chunk 700..1300");
    assert_eq!(a.hcg_inj, full.hcg_inj[..700 * HC], "hcg inject chunk 0..700");
    assert_eq!(b.hcg_inj, full.hcg_inj[700 * HC..1300 * HC], "hcg inject chunk 700..1300");
}

/// The hcg GEMM leg only (pre_gemm), returning `y` and `inj`.
fn run_hcg_at(r: &Rig, t: usize, h: usize, hc: usize, rank: usize, st: &Stream) -> (Vec<u16>, Vec<f32>) {
    let input = DevBuf::from_slice(&r.input).unwrap();
    let mut x = DevBuf::zeroed(t * hc * h * 4).unwrap();
    fn_mhc::hc_expand(&input, &mut x, h, hc, t, st).unwrap();
    let (norm, down, up, inject) = (
        DevBuf::from_slice(&r.norm).unwrap(),
        DevBuf::from_slice(&r.down).unwrap(),
        DevBuf::from_slice(&r.up).unwrap(),
        DevBuf::from_slice(&r.inject).unwrap(),
    );
    let mix = fn_mhc::Mixer { norm: &norm, down: &down, up: &up, inject: Some(&inject) };
    let (mut gy, mut gi, mut gs) = (
        DevBuf::zeroed(t * h * 2).unwrap(),
        DevBuf::zeroed(t * hc * 4).unwrap(),
        DevBuf::zeroed(hcx::gemm_scratch_bytes(t, h, hc, rank)).unwrap(),
    );
    hcx::pre_gemm(&x, &mix, &mut gy, Some(&mut gi), &mut gs, h, hc, rank, 1e-6, t, st).unwrap();
    (dl(&gy, st), dl(&gi, st))
}

/// [`run_hcg_at`] on the 64-row slab loop (`pre_gemm_ref`), the arithmetic `pre_gemm` has to reproduce.
fn run_hcg_ref_at(r: &Rig, t: usize, h: usize, hc: usize, rank: usize, st: &Stream) -> (Vec<u16>, Vec<f32>) {
    let input = DevBuf::from_slice(&r.input).unwrap();
    let mut x = DevBuf::zeroed(t * hc * h * 4).unwrap();
    fn_mhc::hc_expand(&input, &mut x, h, hc, t, st).unwrap();
    let (norm, down, up, inject) = (
        DevBuf::from_slice(&r.norm).unwrap(),
        DevBuf::from_slice(&r.down).unwrap(),
        DevBuf::from_slice(&r.up).unwrap(),
        DevBuf::from_slice(&r.inject).unwrap(),
    );
    let mix = fn_mhc::Mixer { norm: &norm, down: &down, up: &up, inject: Some(&inject) };
    let (mut gy, mut gi, mut gs) = (
        DevBuf::zeroed(t * h * 2).unwrap(),
        DevBuf::zeroed(t * hc * 4).unwrap(),
        DevBuf::zeroed(hcx::gemm_scratch_bytes(t, h, hc, rank)).unwrap(),
    );
    hcx::pre_gemm_ref(&x, &mix, &mut gy, Some(&mut gi), &mut gs, h, hc, rank, 1e-6, t, st).unwrap();
    (dl(&gy, st), dl(&gi, st))
}

/// The Flash-Next shape (H 2560, hc 4, rank 320) runs the down, inject and up products as a few large calls with fixed cublasLt
/// algorithms (split-K 4 and 20 like the library's own M = 64 pick); the outputs, `y` and `inject`, are the slab loop's bit for bit at
/// every row count (the 64-row pass boundaries, a ragged tail, a whole 4,096-row chunk).
#[test]
#[ignore]
fn hcg_large_calls_equal_the_slab_loop_bit_for_bit_at_qfn_dims() {
    let Some(st) = gpu() else { return };
    let r = rig_with(4096, QH, QHC, QR);
    for t in [1usize, 7, 63, 64, 65, 129, 192, 1000, 2048, 4000, 4096] {
        let rr = rig_rows(&r, 0, t, QH);
        let big = run_hcg_at(&rr, t, QH, QHC, QR, &st);
        let slab = run_hcg_ref_at(&rr, t, QH, QHC, QR, &st);
        assert_eq!(big.0, slab.0, "hcg y, {t} rows");
        assert_eq!(big.1, slab.1, "hcg inject, {t} rows");
    }
}

/// `pre_gemm_post` (hc_post fused into the pre-mix's norm pass) is `hc_post` then `pre_gemm` bit for bit: the mixed rows, the inject rows
/// and the updated streams, at the Flash-Next shape and ragged row counts.
#[test]
#[ignore]
fn hcg_fused_post_equals_hc_post_then_pre_gemm_at_qfn_dims() {
    let Some(st) = gpu() else { return };
    let r = rig_with(4096, QH, QHC, QR);
    let mut rng = Rng(0x7a11);
    for t in [1usize, 63, 64, 100, 1000, 4096] {
        let (h, hc, rank) = (QH, QHC, QR);
        let input = DevBuf::from_slice(&r.input[..t * h]).unwrap();
        let (norm, down, up, inject) = (
            DevBuf::from_slice(&r.norm).unwrap(),
            DevBuf::from_slice(&r.down).unwrap(),
            DevBuf::from_slice(&r.up).unwrap(),
            DevBuf::from_slice(&r.inject).unwrap(),
        );
        let mix = fn_mhc::Mixer { norm: &norm, down: &down, up: &up, inject: Some(&inject) };
        let block: Vec<u16> = (0..t * h).map(|_| bf(rng.uniform(-2., 2.))).collect();
        let injp: Vec<f32> = (0..t * hc).map(|_| rng.uniform(-1.5, 1.5)).collect();
        let (block, injp) = (DevBuf::from_slice(&block).unwrap(), DevBuf::from_slice(&injp).unwrap());
        let run = |fused: bool| {
            let mut x = DevBuf::zeroed(t * hc * h * 4).unwrap();
            fn_mhc::hc_expand(&input, &mut x, h, hc, t, &st).unwrap();
            let (mut y, mut inj, mut s) = (
                DevBuf::zeroed(t * h * 2).unwrap(),
                DevBuf::zeroed(t * hc * 4).unwrap(),
                DevBuf::zeroed(hcx::gemm_scratch_bytes(t, h, hc, rank)).unwrap(),
            );
            if fused {
                hcx::pre_gemm_post(&mut x, &block, &injp, &mix, &mut y, Some(&mut inj), &mut s, h, hc, rank, 1e-6, t, &st).unwrap();
            } else {
                fn_mhc::hc_post(&block, None, &injp, &mut x, h, hc, t, &st).unwrap();
                hcx::pre_gemm(&x, &mix, &mut y, Some(&mut inj), &mut s, h, hc, rank, 1e-6, t, &st).unwrap();
            }
            (dl::<u16>(&y, &st), dl::<f32>(&inj, &st), dl::<f32>(&x, &st))
        };
        let (a, b) = (run(false), run(true));
        assert_eq!(a.0, b.0, "mixed rows, {t} rows");
        assert_eq!(a.1, b.1, "inject rows, {t} rows");
        assert_eq!(a.2, b.2, "streams, {t} rows");
    }
}

/// Prevents any read that leaks past a tensor from staying silent: every weight is placed in a fresh arena with NaN
/// poison immediately after it, and the run must reproduce the independently-allocated run bit for bit. Fresh-buffer
/// reproducibility is covered too: the arena is allocated fresh per run.
#[test]
#[ignore]
fn hcg_reads_only_its_declared_operands_with_poisoned_neighbors_at_qfn_dims() {
    let Some(st) = gpu() else { return };
    let r = rig_with(256, QH, QHC, QR);
    let (t, h, hc, rank) = (256, QH, QHC, QR);
    let base = run_hcg_at(&r, t, h, hc, rank, &st);
    // fp16 NaN poison (0x7E00): any leaked read feeds NaN into the GEMM and flips the bits.
    let nan: u16 = 0x7E00;
    let pad_rows = 64usize;
    let with_poison = |w: &[u16], wide: usize| {
        let wbytes = w.len() * 2;
        let padbytes = pad_rows * wide * 2;
        let arena = DevBuf::zeroed_view(wbytes + padbytes).unwrap();
        arena.subview(0, wbytes).upload(w).unwrap();
        let poison = vec![nan; padbytes / 2];
        arena.subview(wbytes, padbytes).upload(&poison).unwrap();
        arena.subview(0, wbytes)
    };
    let k = hc * h;
    let norm = with_poison(&r.norm, k);
    let down = with_poison(&r.down, k);
    let up = with_poison(&r.up, rank);
    let inject = with_poison(&r.inject, k);
    let input = DevBuf::from_slice(&r.input).unwrap();
    let mut x = DevBuf::zeroed(t * hc * h * 4).unwrap();
    fn_mhc::hc_expand(&input, &mut x, h, hc, t, &st).unwrap();
    let mix = fn_mhc::Mixer { norm: &norm, down: &down, up: &up, inject: Some(&inject) };
    let (mut gy, mut gi, mut gs) = (
        DevBuf::zeroed(t * h * 2).unwrap(),
        DevBuf::zeroed(t * hc * 4).unwrap(),
        DevBuf::zeroed(hcx::gemm_scratch_bytes(t, h, hc, rank)).unwrap(),
    );
    hcx::pre_gemm(&x, &mix, &mut gy, Some(&mut gi), &mut gs, h, hc, rank, 1e-6, t, &st).unwrap();
    let poisoned = (dl::<u16>(&gy, &st), dl::<f32>(&gi, &st));
    assert_eq!(base.0, poisoned.0, "hcg y changed with poisoned neighbors");
    assert_eq!(base.1, poisoned.1, "hcg inject changed with poisoned neighbors");
    // fresh-buffer reproducibility: a second fresh arena set reproduces the same bits
    let norm2 = with_poison(&r.norm, k);
    let down2 = with_poison(&r.down, k);
    let up2 = with_poison(&r.up, rank);
    let inject2 = with_poison(&r.inject, k);
    let mix2 = fn_mhc::Mixer { norm: &norm2, down: &down2, up: &up2, inject: Some(&inject2) };
    hcx::pre_gemm(&x, &mix2, &mut gy, Some(&mut gi), &mut gs, h, hc, rank, 1e-6, t, &st).unwrap();
    let fresh = (dl::<u16>(&gy, &st), dl::<f32>(&gi, &st));
    assert_eq!(base.0, fresh.0, "hcg y not reproducible on fresh buffers");
    assert_eq!(base.1, fresh.1, "hcg inject not reproducible on fresh buffers");
}
