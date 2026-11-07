//! KDA kernels (GLM-5.3-Flash): float64 CPU reference, bit-exact gates (window == serial, replay == serial, flip == replay,
//! row and batch invariance, parity over many rounds), the rounding decision, ExLlamaV3 dumps on real layer weights, golden
//! hashes and benchmarks. Every gate here was mutation-checked (see the commit messages): a test that cannot fail proves nothing.
#![cfg(not(gb_no_cuda))]
#![allow(clippy::needless_range_loop)] // the f64 reference reads better with the indices of the formulas

mod common;

use common::{cos_rel_rms, read_npy, Rng};
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::kda::*;
use gb_cuda::num::{bf16_to_f32, f32_to_bf16};
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use std::time::Instant;

const H: usize = GLM_HEADS;
const D: usize = HEAD_DIM;
const HID: usize = GLM_HIDDEN;
const C: usize = 3 * GLM_HEADS * HEAD_DIM;
const VD: usize = GLM_HEADS * HEAD_DIM;

macro_rules! need_gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

fn dims() -> Dims {
    Dims::glm()
}

// ------------------------------------------------------------------------------------------------ host weights and inputs

#[derive(Clone)]
struct HostW {
    b: Vec<u16>,
    f_a: Vec<u16>,
    g_a: Vec<u16>,
    f_b: Vec<u16>,
    g_b: Vec<u16>,
    conv_w: Vec<u16>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    o_norm: Vec<u16>,
}

fn f16s(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| f64_to_f16(x as f64)).collect()
}
fn bf16s(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| f32_to_bf16(x)).collect()
}

impl HostW {
    /// Seeded weights with gate / decay statistics in the range of the real layers (decay rates 1..15, dt_bias -4..0.5).
    fn random(rng: &mut Rng) -> Self {
        Self {
            b: f16s(&rng.vec(H * HID, -0.03, 0.03)),
            f_a: f16s(&rng.vec(RANK * HID, -0.03, 0.03)),
            g_a: f16s(&rng.vec(RANK * HID, -0.03, 0.03)),
            f_b: f16s(&rng.vec(VD * RANK, -0.1, 0.1)),
            g_b: f16s(&rng.vec(VD * RANK, -0.1, 0.1)),
            conv_w: bf16s(&rng.vec(C * 4, -0.6, 0.6)),
            a_log: rng.vec(H, 0.0, 2.5),
            dt_bias: rng.vec(VD, -4.0, 0.5),
            o_norm: bf16s(&rng.vec(D, 0.5, 1.5)),
        }
    }
}

struct DevW {
    b: DevBuf,
    f_a: DevBuf,
    g_a: DevBuf,
    f_b: DevBuf,
    g_b: DevBuf,
    conv_w: DevBuf,
    a_log: DevBuf,
    dt_bias: DevBuf,
    o_norm: DevBuf,
}

impl DevW {
    fn new(w: &HostW) -> Self {
        let u = |v: &Vec<u16>| DevBuf::from_slice(v).unwrap();
        Self {
            b: u(&w.b),
            f_a: u(&w.f_a),
            g_a: u(&w.g_a),
            f_b: u(&w.f_b),
            g_b: u(&w.g_b),
            conv_w: u(&w.conv_w),
            a_log: DevBuf::from_slice(&w.a_log).unwrap(),
            dt_bias: DevBuf::from_slice(&w.dt_bias).unwrap(),
            o_norm: u(&w.o_norm),
        }
    }
    fn view(&self) -> Weights<'_> {
        Weights {
            b: &self.b,
            f_a: &self.f_a,
            g_a: &self.g_a,
            f_b: &self.f_b,
            g_b: &self.g_b,
            conv_w: &self.conv_w,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            o_norm: &self.o_norm,
        }
    }
}

/// Layer input rows (f16) and the projection output rows (f32), as the EXL3 qkv_proj would hand them over.
#[derive(Clone, Default)]
struct Inp {
    x: Vec<u16>,
    qkv: Vec<f32>,
}

impl Inp {
    fn random(rng: &mut Rng, rows: usize) -> Self {
        Self { x: f16s(&rng.vec(rows * HID, -2.0, 2.0)), qkv: rng.vec(rows * C, -2.5, 2.5) }
    }
    fn rows(&self) -> usize {
        self.x.len() / HID
    }
    fn slice(&self, r0: usize, r1: usize) -> Inp {
        Inp { x: self.x[r0 * HID..r1 * HID].to_vec(), qkv: self.qkv[r0 * C..r1 * C].to_vec() }
    }
    fn concat(parts: &[Inp]) -> Inp {
        let mut o = Inp::default();
        for p in parts {
            o.x.extend(&p.x);
            o.qkv.extend(&p.qkv);
        }
        o
    }
}

/// Committed host snapshot of one slot.
#[derive(Clone, PartialEq)]
struct Slot {
    s: Vec<f32>,
    conv: Vec<u16>,
}

impl Slot {
    fn random(rng: &mut Rng) -> Self {
        Self { s: rng.vec(H * D * D, -0.3, 0.3), conv: bf16s(&rng.vec(3 * C, -2.5, 2.5)) }
    }
    fn zeros() -> Self {
        Self { s: vec![0.0; H * D * D], conv: vec![0; 3 * C] }
    }
}

struct Ctx {
    st: Stream,
    w: HostW,
    dw: DevW,
    p: Params,
}

impl Ctx {
    fn new(w: HostW) -> Self {
        Self { st: Stream::new().unwrap(), dw: DevW::new(&w), w, p: Params::glm() }
    }
    fn with_params(mut self, p: Params) -> Self {
        self.p = p;
        self
    }
}

fn pool_with(slots: &[(usize, &Slot)], n_slots: usize) -> Pool {
    let mut pool = Pool::new(dims(), n_slots).unwrap();
    for &(i, s) in slots {
        pool.write_slot(i, &s.s, &s.conv).unwrap();
    }
    pool
}

fn snap(pool: &Pool, slot: usize) -> Slot {
    Slot { s: pool.read_state(slot).unwrap(), conv: pool.read_conv(slot).unwrap() }
}

/// What a forward_window leaves behind.
struct Run {
    rec: Record,
    ws: Workspace,
    out: DevBuf,
    batch: BatchBuf,
    rows: usize,
}

fn fwd(ctx: &Ctx, pool: &mut Pool, inp: &Inp, lens: &[usize], slots: &[usize], write_final: bool) -> Run {
    let batch = BatchBuf::new(lens, slots).unwrap();
    let rows = batch.view().rows;
    assert_eq!(rows, inp.rows());
    let x = DevBuf::from_slice(&inp.x).unwrap();
    let qkv = DevBuf::from_slice(&inp.qkv).unwrap();
    let mut rec = Record::alloc(dims(), rows).unwrap();
    let mut ws = Workspace::alloc(dims(), rows).unwrap();
    let mut out = DevBuf::zeroed(rows * VD * 2).unwrap();
    forward_window(
        &ctx.st,
        dims(),
        &ctx.dw.view(),
        &ctx.p,
        &x,
        Dtype::F16,
        &qkv,
        Dtype::F32,
        pool,
        &batch.view(),
        &mut rec,
        &mut ws,
        &mut out,
        write_final,
    )
    .unwrap();
    ctx.st.sync().unwrap();
    Run { rec, ws, out, batch, rows }
}

impl Run {
    fn commit(&mut self, ctx: &Ctx, pool: &mut Pool, keep: Option<&[i32]>, wrote_final: bool) {
        let k = keep.map(|k| DevBuf::from_slice(k).unwrap());
        commit(
            &ctx.st,
            dims(),
            &ctx.dw.view(),
            &ctx.p,
            pool,
            &self.batch.view(),
            &self.rec,
            &mut self.ws,
            k.as_ref(),
            wrote_final,
        )
        .unwrap();
        ctx.st.sync().unwrap();
    }
    fn out(&self) -> Vec<u16> {
        self.out.to_vec::<u16>().unwrap()[..self.rows * VD].to_vec()
    }
    fn y(&self) -> Vec<u16> {
        self.ws.y().to_vec::<u16>().unwrap()[..self.rows * VD].to_vec()
    }
    fn z(&self) -> Vec<f32> {
        self.ws.z().to_vec::<f32>().unwrap()[..self.rows * VD].to_vec()
    }
    fn raw(&self) -> Vec<u16> {
        self.rec.qkv.to_vec::<u16>().unwrap()[..self.rows * C].to_vec()
    }
    fn g(&self) -> Vec<f32> {
        self.rec.g.to_vec::<f32>().unwrap()[..self.rows * VD].to_vec()
    }
    fn beta(&self) -> Vec<f32> {
        self.rec.beta.to_vec::<f32>().unwrap()[..self.rows * H].to_vec()
    }
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

fn hash_u16(v: &[u16]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in v {
        fnv(&mut h, &x.to_le_bytes());
    }
    h
}

fn hash_f32(v: &[f32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in v {
        fnv(&mut h, &x.to_bits().to_le_bytes());
    }
    h
}

fn hash_slot(s: &Slot) -> (u64, u64) {
    (hash_f32(&s.s), hash_u16(&s.conv))
}

fn assert_bits_eq<T: PartialEq + Copy + std::fmt::Debug>(what: &str, got: &[T], want: &[T]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!("{what}: first mismatch at {i}: {:?} vs {:?}", got[i], want[i]);
    }
}

fn assert_f32_bits_eq(what: &str, got: &[f32], want: &[f32]) {
    let g: Vec<u32> = got.iter().map(|x| x.to_bits()).collect();
    let w: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();
    assert_bits_eq(what, &g, &w);
}

fn assert_slot_eq(what: &str, got: &Slot, want: &Slot) {
    assert_f32_bits_eq(&format!("{what}: S"), &got.s, &want.s);
    assert_bits_eq(&format!("{what}: conv"), &got.conv, &want.conv);
}

// ------------------------------------------------------------------------------------------------ f64 CPU reference

fn bf(x: f64) -> f64 {
    bf16_to_f32(f32_to_bf16(x as f32)) as f64
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Reference of the projections and gates for one row: (beta, g [VD], z [VD]).
fn cpu_proj(w: &HostW, p: &Params, x: &[u16]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let xf: Vec<f64> = x.iter().map(|&h| f16_to_f64(h)).collect();
    let dot = |wt: &[u16], n: usize, row: usize, v: &[f64]| -> f64 { (0..n).map(|k| f16_to_f64(wt[row * n + k]) * v[k]).sum() };
    let beta: Vec<f64> = (0..H).map(|h| bf(sigmoid(dot(&w.b, HID, h, &xf)))).collect();
    let fa: Vec<f64> = (0..RANK).map(|j| dot(&w.f_a, HID, j, &xf)).collect();
    let ga: Vec<f64> = (0..RANK).map(|j| dot(&w.g_a, HID, j, &xf)).collect();
    let mut g = vec![0.0; VD];
    let mut z = vec![0.0; VD];
    for c in 0..VD {
        let fb = dot(&w.f_b, RANK, c, &fa);
        let decay = (w.a_log[c / D] as f64).exp();
        g[c] = (p.lower as f64 * sigmoid(decay * (fb + w.dt_bias[c] as f64))).exp();
        z[c] = dot(&w.g_b, RANK, c, &ga);
    }
    (beta, g, z)
}

/// conv + silu (rounded to bf16 once) of one row from its 4-row window (raw bf16 values as f64, oldest first).
fn cpu_conv(w: &HostW, win: &[Vec<f64>; 4]) -> Vec<f64> {
    (0..C)
        .map(|ch| {
            let mut acc = 0.0f64;
            for (j, row) in win.iter().enumerate() {
                acc += bf16_to_f32(w.conv_w[ch * 4 + j]) as f64 * row[ch];
            }
            bf(acc * sigmoid(acc))
        })
        .collect()
}

struct CpuPrep {
    q: Vec<f64>, // scaled, [H][D]
    k: Vec<f64>,
    v: Vec<f64>,
}

fn l2(c: &[f64]) -> Vec<f64> {
    let ss: f64 = c.iter().map(|x| x * x).sum();
    let inv = 1.0 / (ss + 1e-6).sqrt();
    c.iter().map(|x| x * inv).collect()
}

fn cpu_prep_row(c: &[f64]) -> CpuPrep {
    let mut q = vec![0.0; VD];
    let mut k = vec![0.0; VD];
    for h in 0..H {
        let qh = l2(&c[h * D..(h + 1) * D]);
        let kh = l2(&c[VD + h * D..VD + (h + 1) * D]);
        for i in 0..D {
            q[h * D + i] = qh[i] * 0.08838834764831845;
            k[h * D + i] = kh[i];
        }
    }
    CpuPrep { q, k, v: c[2 * VD..].to_vec() }
}

/// One delta-rule step in f64 on s [H][D j][D i]; returns the core output [H*D] (unrounded).
fn cpu_step(s: &mut [f64], pr: &CpuPrep, g: &[f64], beta: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; VD];
    for h in 0..H {
        for j in 0..D {
            let row = &mut s[(h * D + j) * D..(h * D + j + 1) * D];
            let mut mem = 0.0;
            for i in 0..D {
                row[i] *= g[h * D + i];
                mem += row[i] * pr.k[h * D + i];
            }
            let delta = (pr.v[h * D + j] - mem) * beta[h];
            let mut o = 0.0;
            for i in 0..D {
                row[i] += pr.k[h * D + i] * delta;
                o += row[i] * pr.q[h * D + i];
            }
            y[h * D + j] = o;
        }
    }
    y
}

fn cpu_norm(w: &HostW, p: &Params, y: &[f64], z: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; VD];
    for h in 0..H {
        let ys = &y[h * D..(h + 1) * D];
        let ms: f64 = ys.iter().map(|v| v * v).sum::<f64>() / D as f64;
        let r = 1.0 / (ms + p.eps as f64).sqrt();
        for j in 0..D {
            out[h * D + j] = ys[j] * bf16_to_f32(w.o_norm[j]) as f64 * r * sigmoid(z[h * D + j]);
        }
    }
    out
}

/// Whole-layer reference over `rows` rows from `slot`: (core y f64 [rows*VD], norm out f64, z, g, beta, final state f64).
struct CpuRun {
    y: Vec<f64>,
    out: Vec<f64>,
    g: Vec<f64>,
    beta: Vec<f64>,
    z: Vec<f64>,
    s: Vec<f64>,
}

fn cpu_forward(w: &HostW, p: &Params, inp: &Inp, slot: &Slot, round_y: bool) -> CpuRun {
    let rows = inp.rows();
    let mut s: Vec<f64> = slot.s.iter().map(|&v| v as f64).collect();
    let raw_of = |r: isize| -> Vec<f64> {
        if r >= 0 {
            inp.qkv[r as usize * C..(r as usize + 1) * C].iter().map(|&v| bf(v as f64)).collect()
        } else {
            slot.conv[(3 + r) as usize * C..(4 + r) as usize * C].iter().map(|&b| bf16_to_f32(b) as f64).collect()
        }
    };
    let mut run = CpuRun { y: vec![], out: vec![], g: vec![], beta: vec![], z: vec![], s: vec![] };
    for t in 0..rows {
        let (beta, g, z) = cpu_proj(w, p, &inp.x[t * HID..(t + 1) * HID]);
        let win = [raw_of(t as isize - 3), raw_of(t as isize - 2), raw_of(t as isize - 1), raw_of(t as isize)];
        let pr = cpu_prep_row(&cpu_conv(w, &win));
        let mut y = cpu_step(&mut s, &pr, &g, &beta);
        if round_y {
            for v in y.iter_mut() {
                *v = bf(*v);
            }
        }
        run.out.extend(cpu_norm(w, p, &y, &z));
        run.y.extend(y);
        run.g.extend(g);
        run.beta.extend(beta);
        run.z.extend(z);
    }
    run.s = s;
    run
}

fn f32s(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}
fn bf_to_f32(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&x| bf16_to_f32(x)).collect()
}
fn h_to_f32(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&x| f16_to_f64(x) as f32).collect()
}

/// Reference of the recurrence alone, fed with the GPU's OWN record and prep outputs (isolates the recurrence from the prep).
fn cpu_recurrence_from_gpu(run: &Run, slot: &Slot) -> (Vec<f64>, Vec<f64>) {
    let rows = run.rows;
    let q: Vec<f32> = run.ws.q().to_vec().unwrap();
    let k: Vec<f32> = run.ws.k().to_vec().unwrap();
    let v: Vec<u16> = run.ws.v().to_vec().unwrap();
    let g = run.g();
    let beta = run.beta();
    let mut s: Vec<f64> = slot.s.iter().map(|&v| v as f64).collect();
    let mut ys = Vec::new();
    for t in 0..rows {
        let pr = CpuPrep {
            q: q[t * VD..(t + 1) * VD].iter().map(|&x| x as f64).collect(),
            k: k[t * VD..(t + 1) * VD].iter().map(|&x| x as f64).collect(),
            v: v[t * VD..(t + 1) * VD].iter().map(|&x| bf16_to_f32(x) as f64).collect(),
        };
        let gt: Vec<f64> = g[t * VD..(t + 1) * VD].iter().map(|&x| x as f64).collect();
        let bt: Vec<f64> = beta[t * H..(t + 1) * H].iter().map(|&x| x as f64).collect();
        ys.extend(cpu_step(&mut s, &pr, &gt, &bt));
    }
    (ys, s)
}

fn rel(got: &[f32], want: &[f32]) -> (f64, f64) {
    cos_rel_rms(got, want)
}

// ------------------------------------------------------------------------------------------------ (a) f64 reference

#[test]
fn layer_matches_f64_reference() {
    need_gpu!();
    let mut rng = Rng(11);
    let ctx = Ctx::new(HostW::random(&mut rng));
    let inp = Inp::random(&mut rng, 5);
    let start = Slot::random(&mut rng);
    let mut pool = pool_with(&[(1, &start)], 3);
    let run = fwd(&ctx, &mut pool, &inp, &[5], &[1], true);
    let cpu = cpu_forward(&ctx.w, &ctx.p, &inp, &start, false);

    // projections and gates (fp32 on the GPU against f64)
    let chk = |what: &str, got: &[f32], want: &[f64], tol: f64| {
        let (cos, rr) = rel(got, &f32s(want));
        eprintln!("{what}: cos {cos:.9} relRMS {rr:.3e}");
        assert!(rr <= tol && cos >= 0.99999, "{what}: cos {cos} relRMS {rr}");
    };
    chk("g", &run.g(), &cpu.g, 2e-5);
    chk("beta", &run.beta(), &cpu.beta, 1e-5);
    chk("z", &run.z(), &cpu.z, 2e-6);
    // raw rows are exactly the bf16 rounding of the input
    let want_raw: Vec<u16> = inp.qkv.iter().map(|&v| f32_to_bf16(v)).collect();
    assert_bits_eq("raw record", &run.raw(), &want_raw);

    // the recurrence, fed with the GPU's own prep: bf16 rounding of y is the only difference
    let (ys, s_after) = cpu_recurrence_from_gpu(&run, &start);
    let (cos, rr) = rel(&bf_to_f32(&run.y()), &f32s(&ys));
    eprintln!("recurrence y: cos {cos:.9} relRMS {rr:.3e}");
    assert!(rr <= 2e-3 && cos >= 0.99999, "recurrence y: cos {cos} relRMS {rr}");
    // the window has written the state after row 5 into the other buffer; committing flips to it
    let mut run = run;
    run.commit(&ctx, &mut pool, None, true);
    let after = snap(&pool, 1);
    let (cos, rr) = rel(&after.s, &f32s(&s_after));
    eprintln!("recurrence S: cos {cos:.9} relRMS {rr:.3e}");
    assert!(rr <= 2e-6 && cos >= 0.9999999, "S: cos {cos} relRMS {rr}");

    // end to end (the whole prep included) against the f64 layer, y rounded to bf16 only at the spec points
    let (cos, rr) = rel(&bf_to_f32(&run.y()), &f32s(&cpu.y));
    eprintln!("end to end y: cos {cos:.9} relRMS {rr:.3e}");
    assert!(rr <= 3e-3 && cos >= 0.99999, "y: cos {cos} relRMS {rr}");
    let (cos, rr) = rel(&h_to_f32(&run.out()), &f32s(&cpu.out));
    eprintln!("end to end norm out: cos {cos:.9} relRMS {rr:.3e}");
    assert!(rr <= 4e-3 && cos >= 0.99999, "norm out: cos {cos} relRMS {rr}");
    let (cos, rr) = rel(&after.s, &f32s(&cpu.s));
    assert!(rr <= 2e-3 && cos >= 0.99999, "end to end S: cos {cos} relRMS {rr}");

    // the conv state after the commit: the last 3 raw rows
    let mut want_conv: Vec<u16> = start.conv.clone();
    want_conv.extend(&want_raw);
    assert_bits_eq("conv after commit", &after.conv, &want_conv[5 * C..8 * C]);
}

#[test]
fn prep_matches_f64_reference() {
    need_gpu!();
    let mut rng = Rng(12);
    let ctx = Ctx::new(HostW::random(&mut rng));
    // three sequences: the conv state of each must be used by its own first rows only
    let lens = [3usize, 1, 6];
    let slots = [2usize, 0, 1];
    let parts: Vec<Inp> = lens.iter().map(|&l| Inp::random(&mut rng, l)).collect();
    let starts: Vec<Slot> = (0..3).map(|_| Slot::random(&mut rng)).collect();
    let mut pool = pool_with(&[(0, &starts[1]), (1, &starts[2]), (2, &starts[0])], 3);
    let inp = Inp::concat(&parts);
    let run = fwd(&ctx, &mut pool, &inp, &lens, &slots, true);
    let q: Vec<f32> = run.ws.q().to_vec().unwrap();
    let k: Vec<f32> = run.ws.k().to_vec().unwrap();
    let v: Vec<u16> = run.ws.v().to_vec().unwrap();
    let mut r0 = 0;
    for (si, &len) in lens.iter().enumerate() {
        let st = &starts[si];
        for t in 0..len {
            let raw_of = |r: isize| -> Vec<f64> {
                if r >= 0 {
                    parts[si].qkv[r as usize * C..(r as usize + 1) * C].iter().map(|&x| bf(x as f64)).collect()
                } else {
                    st.conv[(3 + r) as usize * C..(4 + r) as usize * C].iter().map(|&b| bf16_to_f32(b) as f64).collect()
                }
            };
            let t = t as isize;
            let pr = cpu_prep_row(&cpu_conv(&ctx.w, &[raw_of(t - 3), raw_of(t - 2), raw_of(t - 1), raw_of(t)]));
            let row = r0 + t as usize;
            let cmp = |what: &str, got: &[f32], want: &[f64], tol: f64| {
                let (cos, rr) = rel(got, &f32s(want));
                assert!(rr <= tol && cos >= 0.99999, "seq {si} row {t} {what}: cos {cos} relRMS {rr}");
            };
            cmp("q", &q[row * VD..(row + 1) * VD], &pr.q, 1e-3);
            cmp("k", &k[row * VD..(row + 1) * VD], &pr.k, 1e-3);
            cmp("v", &bf_to_f32(&v[row * VD..(row + 1) * VD]), &pr.v, 1e-3);
        }
        r0 += len;
    }
}

// ------------------------------------------------------------------------------------------------ (b) bit-exact gates

/// The serial reference: every row as its own 1-row forward + all-rows commit. Per-row outputs and committed-state hashes.
struct Serial {
    out: Vec<u16>,
    y: Vec<u16>,
    raw: Vec<u16>,
    g: Vec<f32>,
    beta: Vec<f32>,
    z: Vec<f32>,
    /// (S hash, conv hash) after m rows, m = 0..=rows
    hashes: Vec<(u64, u64)>,
    last: Slot,
}

fn serial(ctx: &Ctx, inp: &Inp, slot: usize, start: &Slot) -> Serial {
    let mut pool = pool_with(&[(slot, start)], slot + 1);
    let mut sr = Serial {
        out: vec![],
        y: vec![],
        raw: vec![],
        g: vec![],
        beta: vec![],
        z: vec![],
        hashes: vec![hash_slot(start)],
        last: start.clone(),
    };
    for t in 0..inp.rows() {
        let mut run = fwd(ctx, &mut pool, &inp.slice(t, t + 1), &[1], &[slot], true);
        sr.out.extend(run.out());
        sr.y.extend(run.y());
        sr.raw.extend(run.raw());
        sr.g.extend(run.g());
        sr.beta.extend(run.beta());
        sr.z.extend(run.z());
        run.commit(ctx, &mut pool, None, true);
        sr.last = snap(&pool, slot);
        sr.hashes.push(hash_slot(&sr.last));
    }
    sr
}

#[test]
fn window_and_replay_equal_serial_bit_for_bit() {
    need_gpu!();
    let mut rng = Rng(21);
    let ctx = Ctx::new(HostW::random(&mut rng));
    let slot = 1;
    let start = Slot::random(&mut rng);
    // every window length that matters: 1..16, around the 32/64 boundaries, and the 128-row maximum
    let mut lengths: Vec<usize> = (1..=16).collect();
    lengths.extend([31, 32, 33, 63, 64, 65, 127, 128]);
    let max = 128;
    let inp = Inp::random(&mut rng, max);
    let sr = serial(&ctx, &inp, slot, &start);
    for &w in &lengths {
        let part = inp.slice(0, w);
        let mut pool = pool_with(&[(slot, &start)], 3);
        let mut run = fwd(&ctx, &mut pool, &part, &[w], &[slot], true);
        // the row-invariance claim: every per-row quantity of a w-row window equals the 1-row serial step
        assert_bits_eq(&format!("w={w} raw"), &run.raw(), &sr.raw[..w * C]);
        assert_f32_bits_eq(&format!("w={w} g"), &run.g(), &sr.g[..w * VD]);
        assert_f32_bits_eq(&format!("w={w} beta"), &run.beta(), &sr.beta[..w * H]);
        assert_f32_bits_eq(&format!("w={w} z"), &run.z(), &sr.z[..w * VD]);
        assert_bits_eq(&format!("w={w} y"), &run.y(), &sr.y[..w * VD]);
        assert_bits_eq(&format!("w={w} norm out"), &run.out(), &sr.out[..w * VD]);
        // the window did not touch the committed state (the current buffer, the conv state)
        assert_slot_eq(&format!("w={w} window left committed state"), &snap(&pool, slot), &start);
        // commit(m) == m serial steps, for every m on small windows, a spread on the large ones
        let ms: Vec<usize> = if w <= 9 { (0..=w).collect() } else { vec![0, 1, 2, w / 2, w - 1, w] };
        for m in ms {
            let mut p2 = pool_with(&[(slot, &start)], 3);
            // a fresh pool holding the same window result: re-run the window into it (the window is deterministic)
            let mut r2 = fwd(&ctx, &mut p2, &part, &[w], &[slot], true);
            r2.commit(&ctx, &mut p2, Some(&[m as i32]), true);
            let got = snap(&p2, slot);
            assert_eq!(hash_slot(&got), sr.hashes[m], "w={w} commit({m}) differs from {m} serial steps");
        }
        // None == full accept, flip only
        run.commit(&ctx, &mut pool, None, true);
        assert_eq!(hash_slot(&snap(&pool, slot)), sr.hashes[w], "w={w} full accept (flip) differs");
    }
}

#[test]
fn flip_equals_replay_and_final_state_modes_agree() {
    need_gpu!();
    let mut rng = Rng(22);
    let ctx = Ctx::new(HostW::random(&mut rng));
    let slot = 2;
    let start = Slot::random(&mut rng);
    let w = 9;
    let inp = Inp::random(&mut rng, w);
    // (1) window writes its final state, full accept = flip only
    let mut p1 = pool_with(&[(slot, &start)], 3);
    let mut r1 = fwd(&ctx, &mut p1, &inp, &[w], &[slot], true);
    r1.commit(&ctx, &mut p1, None, true);
    // (2) the same with an explicit device keep = len (skip_full path): still no replay
    let mut p2 = pool_with(&[(slot, &start)], 3);
    let mut r2 = fwd(&ctx, &mut p2, &inp, &[w], &[slot], true);
    r2.commit(&ctx, &mut p2, Some(&[w as i32]), true);
    // (3) the window did NOT write its final state: the full accept replays every row
    let mut p3 = pool_with(&[(slot, &start)], 3);
    let mut r3 = fwd(&ctx, &mut p3, &inp, &[w], &[slot], false);
    assert_f32_bits_eq(
        "no-final window left the other buffer alone",
        &p3.read_both(slot).unwrap()[D * D * H..],
        &vec![0.0; H * D * D],
    );
    r3.commit(&ctx, &mut p3, None, false);
    // (4) device keep = len but the window wrote nothing
    let mut p4 = pool_with(&[(slot, &start)], 3);
    let mut r4 = fwd(&ctx, &mut p4, &inp, &[w], &[slot], false);
    r4.commit(&ctx, &mut p4, Some(&[w as i32]), false);
    let a = snap(&p1, slot);
    assert_slot_eq("flip vs skip_full", &snap(&p2, slot), &a);
    assert_slot_eq("flip vs forced replay (None)", &snap(&p3, slot), &a);
    assert_slot_eq("flip vs forced replay (keep=len)", &snap(&p4, slot), &a);
    assert_eq!(p1.read_parity().unwrap()[slot], 1);
    assert_eq!(p3.read_parity().unwrap()[slot], 1);
    // keep = 0 changes nothing and does not flip
    let mut p5 = pool_with(&[(slot, &start)], 3);
    let mut r5 = fwd(&ctx, &mut p5, &inp, &[w], &[slot], true);
    r5.commit(&ctx, &mut p5, Some(&[0]), true);
    assert_slot_eq("keep 0", &snap(&p5, slot), &start);
    assert_eq!(p5.read_parity().unwrap()[slot], 0);
}

#[test]
fn parity_alternates_correctly_over_many_rounds() {
    need_gpu!();
    let mut rng = Rng(23);
    let ctx = Ctx::new(HostW::random(&mut rng));
    // three sequences in one batch each round, different slots, windows of 1..9 rows and keeps 0..=len
    let n_seq = 3;
    let slots = [3usize, 0, 2];
    let total = 40;
    let streams: Vec<Inp> = (0..n_seq).map(|_| Inp::random(&mut rng, total + 20)).collect();
    let starts: Vec<Slot> = (0..n_seq).map(|_| Slot::random(&mut rng)).collect();
    let serials: Vec<Serial> = (0..n_seq).map(|i| serial(&ctx, &streams[i], slots[i], &starts[i])).collect();
    let mut pool = pool_with(&(0..n_seq).map(|i| (slots[i], &starts[i])).collect::<Vec<_>>(), 4);
    let mut pos = vec![0usize; n_seq];
    let mut flips = vec![0usize; n_seq];
    let mut rounds = 0;
    while pos.iter().any(|&p| p < total) && rounds < 200 {
        rounds += 1;
        let lens: Vec<usize> = (0..n_seq).map(|_| 1 + (rng.next_u64() % 9) as usize).collect();
        let inp = Inp::concat(&(0..n_seq).map(|i| streams[i].slice(pos[i], pos[i] + lens[i])).collect::<Vec<_>>());
        let mut run = fwd(&ctx, &mut pool, &inp, &lens, &slots, true);
        let out = run.out();
        let mut r0 = 0;
        for i in 0..n_seq {
            // every window row, accepted or not, equals the serial row at its stream position
            assert_bits_eq(
                &format!("round {rounds} seq {i} out"),
                &out[r0 * VD..(r0 + lens[i]) * VD],
                &serials[i].out[pos[i] * VD..(pos[i] + lens[i]) * VD],
            );
            r0 += lens[i];
        }
        let keep: Vec<i32> = (0..n_seq)
            .map(|i| {
                let k = match rng.next_u64() % 4 {
                    0 => 0,
                    1 => lens[i],
                    _ => (rng.next_u64() % (lens[i] as u64 + 1)) as usize,
                };
                if pos[i] >= total {
                    0
                } else {
                    k as i32
                }
            })
            .collect();
        run.commit(&ctx, &mut pool, Some(&keep), true);
        for i in 0..n_seq {
            pos[i] += keep[i] as usize;
            flips[i] += (keep[i] > 0) as usize;
            let got = snap(&pool, slots[i]);
            assert_eq!(hash_slot(&got), serials[i].hashes[pos[i]], "round {rounds} seq {i} after {} accepted rows", pos[i]);
            assert_eq!(pool.read_parity().unwrap()[slots[i]] as usize, flips[i] % 2, "parity of seq {i}");
        }
    }
    assert!(pos.iter().all(|&p| p >= total), "did not finish in {rounds} rounds");
    eprintln!("{rounds} rounds, flips per sequence {flips:?}");
}

#[test]
fn batching_and_slots_do_not_change_bits() {
    need_gpu!();
    let mut rng = Rng(31);
    let ctx = Ctx::new(HostW::random(&mut rng));
    let lens = [5usize, 1, 8, 2];
    let parts: Vec<Inp> = lens.iter().map(|&l| Inp::random(&mut rng, l)).collect();
    let starts: Vec<Slot> = (0..4).map(|_| Slot::random(&mut rng)).collect();
    let keep_of = [3i32, 1, 8, 0];
    // solo reference: each sequence alone, alone in slot 0
    let mut solo = Vec::new();
    for i in 0..4 {
        let mut p = pool_with(&[(0, &starts[i])], 1);
        let mut run = fwd(&ctx, &mut p, &parts[i], &[lens[i]], &[0], true);
        let (o, y) = (run.out(), run.y());
        run.commit(&ctx, &mut p, Some(&[keep_of[i]]), true);
        solo.push((o, y, snap(&p, 0)));
    }
    // arrangements: different orders and slot assignments, all 4 in one launch
    let arrangements: [([usize; 4], [usize; 4]); 3] =
        [([0, 1, 2, 3], [0, 1, 2, 3]), ([3, 2, 1, 0], [5, 4, 1, 2]), ([1, 3, 0, 2], [7, 0, 6, 3])];
    for (order, slots) in arrangements {
        let blens: Vec<usize> = order.iter().map(|&i| lens[i]).collect();
        let inp = Inp::concat(&order.iter().map(|&i| parts[i].clone()).collect::<Vec<_>>());
        let mut pool = pool_with(&(0..4).map(|p| (slots[p], &starts[order[p]])).collect::<Vec<_>>(), 8);
        let mut run = fwd(&ctx, &mut pool, &inp, &blens, &slots, true);
        let (out, y) = (run.out(), run.y());
        let keep: Vec<i32> = order.iter().map(|&i| keep_of[i]).collect();
        run.commit(&ctx, &mut pool, Some(&keep), true);
        let mut r0 = 0;
        for (p, &i) in order.iter().enumerate() {
            let tag = format!("order {order:?} slots {slots:?} seq {i}");
            assert_bits_eq(&format!("{tag} out"), &out[r0 * VD..(r0 + lens[i]) * VD], &solo[i].0);
            assert_bits_eq(&format!("{tag} y"), &y[r0 * VD..(r0 + lens[i]) * VD], &solo[i].1);
            assert_slot_eq(&tag, &snap(&pool, slots[p]), &solo[i].2);
            r0 += lens[i];
        }
        // slots not in the batch are untouched
        for s in 0..8 {
            if !slots.contains(&s) {
                assert_slot_eq("unused slot", &snap(&pool, s), &Slot::zeros());
            }
        }
    }
}

mod graph_ffi {
    use core::ffi::c_void;
    extern "C" {
        pub fn cudaStreamBeginCapture(s: *mut c_void, mode: i32) -> i32;
        pub fn cudaStreamEndCapture(s: *mut c_void, g: *mut *mut c_void) -> i32;
        pub fn cudaGraphInstantiate(exec: *mut *mut c_void, g: *mut c_void, flags: u64) -> i32;
        pub fn cudaGraphLaunch(exec: *mut c_void, s: *mut c_void) -> i32;
        pub fn cudaGraphExecDestroy(exec: *mut c_void) -> i32;
        pub fn cudaGraphDestroy(g: *mut c_void) -> i32;
    }
}

/// One decode round (project, prep, window, norm, commit with a DEVICE keep array) captured ONCE into a CUDA graph and replayed for several
/// rounds with new inputs and different keeps, against the same rounds run eagerly: bit-identical committed state and outputs after every round.
/// This is what "no host synchronisation, fixed pointers, device-side keep and parity" buys: the verify-and-commit round is graph-capturable.
#[test]
fn a_decode_round_captured_in_a_cuda_graph_replays_with_new_keeps() {
    need_gpu!();
    let mut rng = Rng(35);
    let ctx = Ctx::new(HostW::random(&mut rng));
    let (n_seq, w) = (3usize, 5usize);
    let rows = n_seq * w;
    let slots = [2usize, 0, 1];
    let starts: Vec<Slot> = (0..n_seq).map(|_| Slot::random(&mut rng)).collect();
    let mk_pool = || pool_with(&(0..n_seq).map(|i| (slots[i], &starts[i])).collect::<Vec<_>>(), 3);
    let rounds: Vec<(Inp, Vec<i32>)> = [[5i32, 2, 0], [1, 5, 3], [0, 0, 5], [4, 4, 1], [5, 5, 5]]
        .iter()
        .map(|k| (Inp::random(&mut rng, rows), k.to_vec()))
        .collect();
    let batch = BatchBuf::new(&vec![w; n_seq], &slots).unwrap();
    let mk_bufs = || {
        (
            DevBuf::zeroed(rows * HID * 2).unwrap(),
            DevBuf::zeroed(rows * C * 4).unwrap(),
            DevBuf::zeroed(n_seq * 4).unwrap(),
            DevBuf::zeroed(rows * VD * 2).unwrap(),
            Record::alloc(dims(), rows).unwrap(),
            Workspace::alloc(dims(), rows).unwrap(),
        )
    };
    let w_dev = ctx.dw.view();
    let round = |pool: &mut Pool,
                 x: &DevBuf,
                 q: &DevBuf,
                 keep: &DevBuf,
                 out: &mut DevBuf,
                 rec: &mut Record,
                 ws: &mut Workspace| {
        forward_window(&ctx.st, dims(), &w_dev, &ctx.p, x, Dtype::F16, q, Dtype::F32, pool, &batch.view(), rec, ws, out, true)
            .unwrap();
        commit(&ctx.st, dims(), &w_dev, &ctx.p, pool, &batch.view(), rec, ws, Some(keep), true).unwrap();
    };
    // eager reference
    let (mut pool_e, (mut xe, mut qe, mut ke, mut oe, mut re, mut we)) = (mk_pool(), mk_bufs());
    let mut eager = Vec::new();
    for (inp, keep) in &rounds {
        xe.upload(&inp.x).unwrap();
        qe.upload(&inp.qkv).unwrap();
        ke.upload(keep).unwrap();
        round(&mut pool_e, &xe, &qe, &ke, &mut oe, &mut re, &mut we);
        ctx.st.sync().unwrap();
        eager.push((oe.to_vec::<u16>().unwrap(), (0..n_seq).map(|i| hash_slot(&snap(&pool_e, slots[i]))).collect::<Vec<_>>()));
    }
    // graph: capture once, replay per round
    let (mut pool_g, (mut xg, mut qg, mut kg, mut og, mut rg, mut wg)) = (mk_pool(), mk_bufs());
    let mut graph = core::ptr::null_mut();
    let mut exec = core::ptr::null_mut();
    unsafe {
        assert_eq!(graph_ffi::cudaStreamBeginCapture(ctx.st.raw(), 2), 0, "begin capture");
        round(&mut pool_g, &xg, &qg, &kg, &mut og, &mut rg, &mut wg);
        assert_eq!(graph_ffi::cudaStreamEndCapture(ctx.st.raw(), &mut graph), 0, "end capture");
        assert_eq!(graph_ffi::cudaGraphInstantiate(&mut exec, graph, 0), 0, "instantiate");
    }
    for (r, (inp, keep)) in rounds.iter().enumerate() {
        xg.upload(&inp.x).unwrap();
        qg.upload(&inp.qkv).unwrap();
        kg.upload(keep).unwrap();
        unsafe { assert_eq!(graph_ffi::cudaGraphLaunch(exec, ctx.st.raw()), 0, "launch") };
        ctx.st.sync().unwrap();
        assert_bits_eq(&format!("round {r} norm out"), &og.to_vec::<u16>().unwrap(), &eager[r].0);
        for i in 0..n_seq {
            assert_eq!(hash_slot(&snap(&pool_g, slots[i])), eager[r].1[i], "round {r} slot {} after keep {}", slots[i], keep[i]);
        }
    }
    unsafe {
        graph_ffi::cudaGraphExecDestroy(exec);
        graph_ffi::cudaGraphDestroy(graph);
    }
}

/// A zero-length sequence inside a batch (a request with nothing to verify this round) is skipped everywhere: its slot, parity and
/// conv state are untouched, and its neighbours are unaffected.
#[test]
fn empty_sequence_in_a_batch() {
    need_gpu!();
    let mut rng = Rng(34);
    let ctx = Ctx::new(HostW::random(&mut rng));
    let (a, b, c) = (Inp::random(&mut rng, 3), Inp::random(&mut rng, 2), Slot::random(&mut rng));
    let (sa, sb) = (Slot::random(&mut rng), Slot::random(&mut rng));
    let mut pool = pool_with(&[(0, &sa), (1, &c), (2, &sb)], 3);
    let inp = Inp::concat(&[a.clone(), b.clone()]);
    let mut run = fwd(&ctx, &mut pool, &inp, &[3, 0, 2], &[0, 1, 2], true);
    run.commit(&ctx, &mut pool, Some(&[3, 0, 1]), true);
    assert_slot_eq("empty sequence's slot", &snap(&pool, 1), &c);
    assert_eq!(pool.read_parity().unwrap(), vec![1, 0, 1]);
    for (i, (inp1, st, keep)) in [(a, &sa, 3i32), (b, &sb, 1)].into_iter().enumerate() {
        let slot = [0usize, 2][i];
        let mut p = pool_with(&[(0, st)], 1);
        let mut solo = fwd(&ctx, &mut p, &inp1, &[inp1.rows()], &[0], true);
        solo.commit(&ctx, &mut p, Some(&[keep]), true);
        assert_slot_eq(&format!("neighbour {i}"), &snap(&pool, slot), &snap(&p, 0));
    }
}

#[test]
fn one_hundred_twenty_eight_sequences_equal_solo_runs() {
    need_gpu!();
    let mut rng = Rng(32);
    let ctx = Ctx::new(HostW::random(&mut rng));
    let n = 128;
    let inp = Inp::random(&mut rng, n);
    // each of the 128 sequences has its own state; solo = the same 1-row step alone
    let starts: Vec<Slot> = (0..n).map(|_| Slot::random(&mut rng)).collect();
    let mut pool = pool_with(&starts.iter().enumerate().collect::<Vec<_>>(), n);
    let slots: Vec<usize> = (0..n).collect();
    let mut run = fwd(&ctx, &mut pool, &inp, &vec![1; n], &slots, true);
    let (out, y) = (run.out(), run.y());
    run.commit(&ctx, &mut pool, None, true);
    for &i in &[0usize, 1, 31, 64, 100, 127] {
        let mut p = pool_with(&[(0, &starts[i])], 1);
        let mut solo = fwd(&ctx, &mut p, &inp.slice(i, i + 1), &[1], &[0], true);
        assert_bits_eq(&format!("row {i} out"), &out[i * VD..(i + 1) * VD], &solo.out());
        assert_bits_eq(&format!("row {i} y"), &y[i * VD..(i + 1) * VD], &solo.y());
        solo.commit(&ctx, &mut p, None, true);
        assert_slot_eq(&format!("seq {i}"), &snap(&pool, i), &snap(&p, 0));
    }
}

#[test]
fn state_kernel_row_blocking_does_not_change_bits() {
    need_gpu!();
    let mut rng = Rng(33);
    let ctx = Ctx::new(HostW::random(&mut rng));
    let start = Slot::random(&mut rng);
    let inp = Inp::random(&mut rng, 11);
    let mut outs = Vec::new();
    for r_step in [4usize, 8] {
        let c =
            Ctx { st: Stream::new().unwrap(), w: ctx.w.clone(), dw: DevW::new(&ctx.w), p: Params { r_step, ..Params::glm() } };
        let mut pool = pool_with(&[(0, &start)], 1);
        let mut run = fwd(&c, &mut pool, &inp, &[11], &[0], true);
        let (y, o) = (run.y(), run.out());
        run.commit(&c, &mut pool, Some(&[6]), true);
        outs.push((y, o, snap(&pool, 0)));
    }
    assert_bits_eq("y", &outs[0].0, &outs[1].0);
    assert_bits_eq("out", &outs[0].1, &outs[1].1);
    assert_slot_eq("state", &outs[0].2, &outs[1].2);
}

/// The state kernel's cross-lane reduction (halves exchanged, R/2 + R/4 + .. + 1 shuffles) is bit-identical to the plain xor butterfly
/// for the row each lane ends up holding, on adversarial data (mixed magnitudes and signs, where the association order matters).
#[test]
fn halving_reduction_is_the_butterfly_tree() {
    need_gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(26);
    for r in [4usize, 8, 16, 32] {
        for _ in 0..200 {
            let mut v: Vec<f32> = rng.vec(32 * r, -1.0, 1.0);
            for x in v.iter_mut() {
                *x *= (2.0f32).powi((rng.next_u64() % 40) as i32 - 20); // magnitudes over 2^-20 .. 2^19: rounding depends on the order
            }
            let input = DevBuf::from_slice(&v).unwrap();
            let mut out = DevBuf::zeroed(32 * 2 * 4).unwrap();
            selftest_reduce(&st, &input, &mut out, r).unwrap();
            st.sync().unwrap();
            let o: Vec<f32> = out.to_vec().unwrap();
            for lane in 0..32 {
                assert_eq!(
                    o[lane * 2].to_bits(),
                    o[lane * 2 + 1].to_bits(),
                    "r={r} lane {lane}: butterfly {} vs halving {}",
                    o[lane * 2],
                    o[lane * 2 + 1]
                );
            }
        }
    }
}

/// project() alone: every output row of an r-row launch equals the same row computed alone, for the row counts around every kernel
/// configuration (proj_in one output per block up to 8 rows, four above, the tiled schedule from 64 rows; proj_out one thread an output
/// below 64 rows, tiled from 64) up to a 600-row launch; and the three input dtypes agree.
#[test]
fn projections_are_row_invariant_and_dtype_agnostic() {
    need_gpu!();
    let mut rng = Rng(24);
    let ctx = Ctx::new(HostW::random(&mut rng));
    let total = 600;
    let inp = Inp::random(&mut rng, total);
    let project_rows = |x: &DevBuf, dt: Dtype, rows: usize| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut rec = Record::alloc(dims(), rows).unwrap();
        let mut ws = Workspace::alloc(dims(), rows).unwrap();
        project(&ctx.st, dims(), &ctx.dw.view(), &ctx.p, x, dt, rows, &mut rec, &mut ws).unwrap();
        ctx.st.sync().unwrap();
        (
            rec.g.to_vec::<f32>().unwrap()[..rows * VD].to_vec(),
            rec.beta.to_vec::<f32>().unwrap()[..rows * H].to_vec(),
            ws.z().to_vec::<f32>().unwrap()[..rows * VD].to_vec(),
        )
    };
    let xs = DevBuf::from_slice(&inp.x).unwrap();
    let all = project_rows(&xs, Dtype::F16, total);
    for rows in [1usize, 2, 7, 8, 9, 15, 16, 17, 40, 63, 64, 65, 127, 200, 513] {
        let part = DevBuf::from_slice(&inp.x[..rows * HID]).unwrap();
        let got = project_rows(&part, Dtype::F16, rows);
        assert_f32_bits_eq(&format!("rows={rows} g"), &got.0, &all.0[..rows * VD]);
        assert_f32_bits_eq(&format!("rows={rows} beta"), &got.1, &all.1[..rows * H]);
        assert_f32_bits_eq(&format!("rows={rows} z"), &got.2, &all.2[..rows * VD]);
    }
    // each row alone (the 1-row configuration) equals its place in the 128-row launch
    for r in [0usize, 1, 7, 8, 63, 64, 127, 599] {
        let one = project_rows(&DevBuf::from_slice(&inp.x[r * HID..(r + 1) * HID]).unwrap(), Dtype::F16, 1);
        assert_f32_bits_eq(&format!("row {r} alone g"), &one.0, &all.0[r * VD..(r + 1) * VD]);
        assert_f32_bits_eq(&format!("row {r} alone z"), &one.2, &all.2[r * VD..(r + 1) * VD]);
    }
    // f32 and bf16 inputs holding the same values give the same bits as f16 does for its own values
    let xf32: Vec<f32> = inp.x.iter().map(|&h| f16_to_f64(h) as f32).collect();
    let f32_run = project_rows(&DevBuf::from_slice(&xf32).unwrap(), Dtype::F32, total);
    assert_f32_bits_eq("f32 input g", &f32_run.0, &all.0);
    assert_f32_bits_eq("f32 input z", &f32_run.2, &all.2);
    let xbf: Vec<u16> = xf32.iter().map(|&v| f32_to_bf16(v)).collect();
    let xbf_f32: Vec<f32> = xbf.iter().map(|&b| bf16_to_f32(b)).collect();
    let a = project_rows(&DevBuf::from_slice(&xbf).unwrap(), Dtype::BF16, total);
    let b = project_rows(&DevBuf::from_slice(&xbf_f32).unwrap(), Dtype::F32, total);
    assert_f32_bits_eq("bf16 input vs f32 input g", &a.0, &b.0);
    assert_f32_bits_eq("bf16 input vs f32 input z", &a.2, &b.2);
}

/// A bf16 qkv (what an EXL3 projection writing bf16 hands over) gives the same record and outputs as the f32 it was rounded from;
/// reset_slot starts a sequence from zero.
#[test]
fn bf16_qkv_and_reset_slot() {
    need_gpu!();
    let mut rng = Rng(25);
    let ctx = Ctx::new(HostW::random(&mut rng));
    let inp = Inp::random(&mut rng, 6);
    let start = Slot::random(&mut rng);
    let mut p1 = pool_with(&[(1, &start)], 2);
    let r1 = fwd(&ctx, &mut p1, &inp, &[6], &[1], true);
    // the same window with the qkv handed over as bf16 (rounded to nearest even by the producer)
    let qkv_bf: Vec<u16> = inp.qkv.iter().map(|&v| f32_to_bf16(v)).collect();
    let mut p2 = pool_with(&[(1, &start)], 2);
    let batch = BatchBuf::new(&[6], &[1]).unwrap();
    let x = DevBuf::from_slice(&inp.x).unwrap();
    let q = DevBuf::from_slice(&qkv_bf).unwrap();
    let mut rec = Record::alloc(dims(), 6).unwrap();
    let mut ws = Workspace::alloc(dims(), 6).unwrap();
    let mut out = DevBuf::zeroed(6 * VD * 2).unwrap();
    forward_window(
        &ctx.st,
        dims(),
        &ctx.dw.view(),
        &ctx.p,
        &x,
        Dtype::F16,
        &q,
        Dtype::BF16,
        &mut p2,
        &batch.view(),
        &mut rec,
        &mut ws,
        &mut out,
        true,
    )
    .unwrap();
    ctx.st.sync().unwrap();
    assert_bits_eq("bf16 qkv: record raw", &rec.qkv.to_vec::<u16>().unwrap()[..6 * C], &r1.raw());
    assert_bits_eq("bf16 qkv: y", &ws.y().to_vec::<u16>().unwrap()[..6 * VD], &r1.y());
    assert_bits_eq("bf16 qkv: norm out", &out.to_vec::<u16>().unwrap()[..6 * VD], &r1.out());

    // reset_slot: a used slot behaves like a fresh zero slot, and does not touch its neighbour
    let other = Slot::random(&mut rng);
    let mut pool = pool_with(&[(0, &other), (1, &start)], 2);
    let mut run = fwd(&ctx, &mut pool, &inp, &[6], &[1], true);
    run.commit(&ctx, &mut pool, None, true); // slot 1 is now on the other buffer
    pool.reset_slot(1, &ctx.st).unwrap();
    ctx.st.sync().unwrap();
    assert_slot_eq("reset slot reads zero", &snap(&pool, 1), &Slot::zeros());
    assert_slot_eq("neighbour untouched", &snap(&pool, 0), &other);
    let mut fresh = pool_with(&[], 2);
    let a = fwd(&ctx, &mut pool, &inp, &[6], &[1], true);
    let b = fwd(&ctx, &mut fresh, &inp, &[6], &[1], true);
    assert_bits_eq("reset slot == fresh slot: out", &a.out(), &b.out());
}

#[test]
fn conv_shift_for_every_keep() {
    need_gpu!();
    let mut rng = Rng(41);
    let ctx = Ctx::new(HostW::random(&mut rng));
    let rows = 9;
    let inp = Inp::random(&mut rng, rows);
    let start = Slot::random(&mut rng);
    let raw: Vec<u16> = inp.qkv.iter().map(|&v| f32_to_bf16(v)).collect();
    let mut all = start.conv.clone();
    all.extend(&raw);
    for keep in [0usize, 1, 2, 3, 5, 8, 9] {
        let mut pool = pool_with(&[(1, &start)], 2);
        let mut run = fwd(&ctx, &mut pool, &inp, &[rows], &[1], true);
        run.commit(&ctx, &mut pool, Some(&[keep as i32]), true);
        let want: Vec<u16> = if keep == 0 { start.conv.clone() } else { all[keep * C..(keep + 3) * C].to_vec() };
        assert_bits_eq(&format!("conv keep={keep}"), &pool.read_conv(1).unwrap(), &want);
        // the other slot stays zero
        assert!(pool.read_conv(0).unwrap().iter().all(|&b| b == 0));
    }
    // keep larger than the sequence is clamped to its length
    let mut pool = pool_with(&[(1, &start)], 2);
    let mut run = fwd(&ctx, &mut pool, &inp, &[rows], &[1], true);
    run.commit(&ctx, &mut pool, Some(&[100]), true);
    assert_bits_eq("conv keep=100", &pool.read_conv(1).unwrap(), &all[rows * C..(rows + 3) * C]);
    // a short window (fewer rows than the history) keeps part of the old state
    let mut pool = pool_with(&[(1, &start)], 2);
    let mut run = fwd(&ctx, &mut pool, &inp.slice(0, 2), &[2], &[1], true);
    run.commit(&ctx, &mut pool, None, true);
    assert_bits_eq("conv after a 2-row window", &pool.read_conv(1).unwrap(), &all[2 * C..5 * C]);
}

// ------------------------------------------------------------------------------------------------ prefill

fn run_prefill(ctx: &Ctx, inp: &Inp, slot: usize, pool: &mut Pool) -> Vec<u16> {
    let rows = inp.rows();
    let x = DevBuf::from_slice(&inp.x).unwrap();
    let qkv = DevBuf::from_slice(&inp.qkv).unwrap();
    let piece = rows.min(PREFILL_PIECE_ROWS);
    let mut rec = Record::alloc(dims(), piece).unwrap();
    let mut ws = Workspace::alloc(dims(), piece).unwrap();
    let mut out = DevBuf::zeroed(rows * VD * 2).unwrap();
    prefill(
        &ctx.st,
        dims(),
        &ctx.dw.view(),
        &ctx.p,
        &x,
        Dtype::F16,
        &qkv,
        Dtype::F32,
        pool,
        slot,
        rows,
        &mut rec,
        &mut ws,
        &mut out,
    )
    .unwrap();
    ctx.st.sync().unwrap();
    out.to_vec::<u16>().unwrap()
}

#[test]
fn prefill_equals_serial_decode_and_is_chunk_invariant() {
    need_gpu!();
    let mut rng = Rng(51);
    let ctx = Ctx::new(HostW::random(&mut rng));
    let total = 300;
    let inp = Inp::random(&mut rng, total);
    let start = Slot::zeros();
    let sr = serial(&ctx, &inp, 1, &start);
    // one call (3 pieces: 128 + 128 + 44)
    let mut p1 = pool_with(&[(1, &start)], 2);
    let o1 = run_prefill(&ctx, &inp, 1, &mut p1);
    assert_bits_eq("prefill out == serial decode out", &o1, &sr.out);
    assert_slot_eq("prefill state == serial decode state", &snap(&p1, 1), &sr.last);
    // the same prompt cut 77 / 1 / 222 across calls
    let mut p3 = pool_with(&[(1, &start)], 2);
    let mut o3 = Vec::new();
    for w in [0usize, 77, 78, 300].windows(2) {
        o3.extend(run_prefill(&ctx, &inp.slice(w[0], w[1]), 1, &mut p3));
    }
    assert_bits_eq("chunked prefill out", &o3, &o1);
    assert_slot_eq("chunked prefill state", &snap(&p3, 1), &snap(&p1, 1));
    // decoding on after a prefill continues the serial bits: row 301 from the prefilled state
    let next = Inp::random(&mut rng, 1);
    let mut cont = pool_with(&[(1, &start)], 2);
    let _ = run_prefill(&ctx, &inp, 1, &mut cont);
    let mut a = fwd(&ctx, &mut cont, &next, &[1], &[1], true);
    let mut ser = pool_with(&[(1, &sr.last)], 2);
    let mut b = fwd(&ctx, &mut ser, &next, &[1], &[1], true);
    assert_bits_eq("decode after prefill", &a.out(), &b.out());
    a.commit(&ctx, &mut cont, None, true);
    b.commit(&ctx, &mut ser, None, true);
    assert_slot_eq("state after decode after prefill", &snap(&cont, 1), &snap(&ser, 1));
}

/// One [`prefill_chunked`] call of `inp` in pieces of `piece` rows: qkv as given (f32) or rounded to bf16 first (the GEMM's bf16 output),
/// on a serial or a pipelined workspace.
fn run_chunked(ctx: &Ctx, inp: &Inp, slot: usize, pool: &mut Pool, piece: usize, bf16_rows: bool, pipe: bool) -> Vec<u16> {
    let rows = inp.rows();
    let x = DevBuf::from_slice(&inp.x).unwrap();
    let (qkv, qd) = if bf16_rows {
        (DevBuf::from_slice(&bf16s(&inp.qkv)).unwrap(), Dtype::BF16)
    } else {
        (DevBuf::from_slice(&inp.qkv).unwrap(), Dtype::F32)
    };
    let mut rec = Record::alloc(dims(), piece).unwrap();
    let mut ws = Workspace::alloc(dims(), piece).unwrap();
    if pipe {
        ws.add_pipe().unwrap();
    }
    let mut out = DevBuf::zeroed(rows * VD * 2).unwrap();
    let desc = prefill_descriptor_pieces(slot, rows, piece).unwrap();
    let w = ctx.dw.view();
    prefill_chunked(&ctx.st, dims(), &w, &ctx.p, &x, Dtype::F16, &qkv, qd, pool, slot, rows, &mut rec, &mut ws, &mut out, &desc, piece)
        .unwrap();
    ctx.st.sync().unwrap();
    out.to_vec::<u16>().unwrap()
}

/// The P2 prefill piece by piece through the public kernels (project, prep, the chunked recurrence, norm, commit), each on its own: what
/// [`prefill_chunked`] computed before its fusions, the reference its bits are held to.
fn run_chunked_reference(ctx: &Ctx, inp: &Inp, slot: usize, pool: &mut Pool, piece: usize) -> Vec<u16> {
    let rows = inp.rows();
    let mut rec = Record::alloc(dims(), piece).unwrap();
    let mut ws = Workspace::alloc(dims(), piece).unwrap();
    let w = ctx.dw.view();
    let mut out = Vec::new();
    for r0 in (0..rows).step_by(piece) {
        let part = inp.slice(r0, (r0 + piece).min(rows));
        let len = part.rows();
        let (x, qkv) = (DevBuf::from_slice(&part.x).unwrap(), DevBuf::from_slice(&part.qkv).unwrap());
        let batch = BatchBuf::new(&[len], &[slot]).unwrap();
        let mut o = DevBuf::zeroed(len * VD * 2).unwrap();
        project(&ctx.st, dims(), &w, &ctx.p, &x, Dtype::F16, len, &mut rec, &mut ws).unwrap();
        prep(&ctx.st, dims(), &w, &qkv, Dtype::F32, pool, &batch.view(), &mut rec, &mut ws).unwrap();
        window_chunked(&ctx.st, dims(), &ctx.p, pool, slot, len, &rec, &mut ws).unwrap();
        norm(&ctx.st, dims(), &w, &ctx.p, len, &ws, &mut o).unwrap();
        commit(&ctx.st, dims(), &w, &ctx.p, pool, &batch.view(), &rec, &mut ws, None, true).unwrap();
        ctx.st.sync().unwrap();
        out.extend(o.to_vec::<u16>().unwrap());
    }
    out
}

/// The P2 prefill (the decay alone from proj_out, the output gate with the norm in one pass; on f32 or bf16 qkv rows, bf16 rows needing no
/// record copy and shifting the conv history off the rows themselves; serial or pipelined: each piece's projections, prep and conv shift on
/// the side stream beside the previous piece's recurrence) gives the bits of the public kernels run one by one: output, state, conv
/// history and parity, in one call and across calls cut on a chunk boundary.
#[test]
fn chunked_prefill_on_bf16_rows_and_pipelined_keeps_the_bits() {
    need_gpu!();
    let mut rng = Rng(77);
    let ctx = Ctx::new(HostW::random(&mut rng));
    let total = 1000;
    let inp = Inp::random(&mut rng, total);
    let start = Slot::random(&mut rng);
    for piece in [256usize, 512] {
        let mut base = pool_with(&[(1, &start)], 2);
        let want = run_chunked_reference(&ctx, &inp, 1, &mut base, piece);
        let (want_slot, want_par) = (snap(&base, 1), base.read_parity().unwrap());
        for (bf, pipe) in [(false, false), (true, false), (true, true), (false, true)] {
            let what = format!("piece {piece}, bf16 rows {bf}, pipelined {pipe}");
            let mut pool = pool_with(&[(1, &start)], 2);
            let got = run_chunked(&ctx, &inp, 1, &mut pool, piece, bf, pipe);
            assert_bits_eq(&format!("out, {what}"), &got, &want);
            assert_slot_eq(&format!("state, {what}"), &snap(&pool, 1), &want_slot);
            assert_eq!(pool.read_parity().unwrap(), want_par, "parity, {what}");
        }
        // pipelined calls of 304 and 696 rows (the second starts on a chunk boundary): the one call's bits
        let mut pool = pool_with(&[(1, &start)], 2);
        let mut got = run_chunked(&ctx, &inp.slice(0, 304), 1, &mut pool, piece, true, true);
        got.extend(run_chunked(&ctx, &inp.slice(304, total), 1, &mut pool, piece, true, true));
        assert_bits_eq(&format!("out of two pipelined calls, piece {piece}"), &got, &want);
        assert_slot_eq(&format!("state after two pipelined calls, piece {piece}"), &snap(&pool, 1), &want_slot);
    }
}

// ------------------------------------------------------------------------------------------------ rounding decision

/// RN against RZ for the core output, measured against the f64 layer at the core output AND at the norm output, the tensor
/// the model actually consumes. Prints the numbers that the decision in the module docs rests on.
#[test]
fn rounding_decision() {
    need_gpu!();
    let mut rng = Rng(61);
    let w = HostW::random(&mut rng);
    let rows = 16;
    let inp = Inp::random(&mut rng, rows);
    let start = Slot::random(&mut rng);
    let cpu = cpu_forward(&w, &Params::glm(), &inp, &start, false);
    let mut res = Vec::new();
    for round in [Round::NearestEven, Round::TowardZero] {
        let ctx = Ctx::new(w.clone()).with_params(Params { round, ..Params::glm() });
        let mut pool = pool_with(&[(0, &start)], 1);
        let run = fwd(&ctx, &mut pool, &inp, &[rows], &[0], true);
        let (cy, ry) = rel(&bf_to_f32(&run.y()), &f32s(&cpu.y));
        let (co, ro) = rel(&h_to_f32(&run.out()), &f32s(&cpu.out));
        // mean signed relative error of the core output (the bias RZ adds)
        let y = bf_to_f32(&run.y());
        let (mut num, mut den) = (0f64, 0f64);
        for (a, b) in y.iter().zip(&cpu.y) {
            num += (*a as f64 - b) * b.signum();
            den += b.abs();
        }
        eprintln!(
            "{round:?}: core y cos {cy:.9} relRMS {ry:.3e} mean signed rel err {:.3e} | norm out cos {co:.9} relRMS {ro:.3e}",
            num / den
        );
        res.push((ry, ro, num / den));
    }
    // RN is the unbiased, more accurate choice at both tensors
    assert!(res[0].0 < res[1].0, "core relRMS: RN {} RZ {}", res[0].0, res[1].0);
    assert!(res[0].1 < res[1].1, "norm out relRMS: RN {} RZ {}", res[0].1, res[1].1);
    assert!(res[0].2.abs() < 0.2 * res[1].2.abs(), "RN must be (nearly) unbiased: {} vs RZ {}", res[0].2, res[1].2);
}

// ------------------------------------------------------------------------------------------------ (c) ExLlamaV3 dumps

/// The ExLlamaV3 dumps of tools/dump_kda_reference.py, or None when REXL3_KDA_DIR is not set (these are GLM-family tests: the Qwen gate has no GLM
/// pack and no dumps, and says "not run" without the word the gate treats as a vacuous pass). When it IS set it must be complete: a wrong path
/// is a failure, never a silent pass.
fn kda_dir() -> Option<std::path::PathBuf> {
    let d = std::path::PathBuf::from(std::env::var("REXL3_KDA_DIR").ok()?);
    assert!(
        d.join("manifest.txt").exists(),
        "REXL3_KDA_DIR={} has no manifest.txt (run tools/dump_kda_reference.py)",
        d.display()
    );
    Some(d)
}

/// `key=1,2,3` lines of manifest.txt.
fn manifest_list(key: &str) -> Vec<usize> {
    let d = kda_dir().unwrap();
    let text = std::fs::read_to_string(d.join("manifest.txt")).unwrap();
    let line =
        text.lines().find_map(|l| l.strip_prefix(&format!("{key}="))).unwrap_or_else(|| panic!("manifest.txt has no {key}"));
    line.split(',').filter(|s| !s.is_empty()).map(|s| s.trim().parse().unwrap()).collect()
}

/// One KDA layer's small tensors from the GLM pack (REXL3_GLM_DIR, required once REXL3_KDA_DIR is set).
fn layer_weights(layer: usize, bf16_dt_bias: bool) -> HostW {
    let dir = std::path::PathBuf::from(
        std::env::var("REXL3_GLM_DIR").expect("REXL3_GLM_DIR must name the GLM pack when REXL3_KDA_DIR is set"),
    );
    assert!(dir.join("model.safetensors.index.json").exists(), "REXL3_GLM_DIR={} is not a pack", dir.display());
    let ck = gb_exl3::checkpoint::scan(&dir, false).expect("scan the GLM pack");
    let get = |n: &str| -> Vec<u8> {
        let name = format!("model.language_model.layers.{layer}.self_attn.{n}");
        let e = ck.plain.get(&name).unwrap_or_else(|| panic!("{name} missing"));
        ck.read_tensor(e).unwrap()
    };
    let u16s = |b: Vec<u8>| -> Vec<u16> { b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() };
    let f32s_ = |b: Vec<u8>| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
    let mut dt_bias = f32s_(get("dt_bias"));
    if bf16_dt_bias {
        // the oracle's BC graph reads a bf16 copy
        for v in dt_bias.iter_mut() {
            *v = bf16_to_f32(f32_to_bf16(*v));
        }
    }
    HostW {
        b: u16s(get("b_proj.weight")),
        f_a: u16s(get("f_a_proj.weight")),
        g_a: u16s(get("g_a_proj.weight")),
        f_b: u16s(get("f_b_proj.weight")),
        g_b: u16s(get("g_b_proj.weight")),
        conv_w: u16s(get("conv1d.weight")),
        a_log: f32s_(get("A_log")),
        dt_bias,
        o_norm: u16s(get("o_norm.weight")),
    }
}

/// Reads the dumps written by tools/dump_kda_reference.py (see its header for the layout).
fn dump(layer: usize, name: &str) -> Vec<f32> {
    let d = kda_dir().unwrap();
    read_npy(&d.join(format!("L{layer:02}/{name}.npy"))).to_f32()
}

fn manifest_layers() -> Vec<usize> {
    manifest_list("layers")
}

fn manifest_steps() -> Vec<usize> {
    manifest_list("steps")
}

/// Transposes the oracle's `[key][value]` state to REXL3's `[value][key]`.
fn transpose_state(got: &[f32]) -> Vec<f32> {
    let mut gt = vec![0f32; got.len()];
    for h in 0..H {
        for j in 0..D {
            for i in 0..D {
                gt[(h * D + i) * D + j] = got[(h * D + j) * D + i];
            }
        }
    }
    gt
}

/// Result of one real-layer comparison: relRMS of core y, norm out and the final state, worst over blocks.
struct RealCmp {
    core: f64,
    norm: f64,
    state: f64,
}

/// The oracle's decode graph on real layer weights, T = 1, 8 and 64 rows carried over one state: the fork's kda_gate_op +
/// conv1d_update + recurrent kernel + gated norm against REXL3's pipeline fed the SAME x / qkv. dt_bias is rounded to bf16 on
/// the host (what the BC graph does) so the comparison isolates the kernels.
fn compare_real(layer: usize, round: Round, tight: bool) -> RealCmp {
    let w = layer_weights(layer, true);
    let ctx = Ctx::new(w).with_params(Params { round, ..Params::glm() });
    let mut pool = pool_with(&[(0, &Slot::zeros())], 1);
    let mut r0 = 0;
    let mut worst = RealCmp { core: 0.0, norm: 0.0, state: 0.0 };
    // (core y, norm out) bounds: the oracle rounds toward zero; with RN the difference is that rounding (~4e-3)
    let (core_tol, norm_tol) = if tight { (1.5e-3, 1.5e-3) } else { (8e-3, 8e-3) };
    for t in manifest_steps() {
        let x = dump(layer, &format!("x_{r0}_{t}"));
        let qkv = dump(layer, &format!("qkv_{r0}_{t}"));
        let inp = Inp { x: f16s(&x), qkv };
        let mut run = fwd(&ctx, &mut pool, &inp, &[t], &[0], true);
        let want_y = dump(layer, &format!("core_{r0}_{t}"));
        let want_out = dump(layer, &format!("norm_out_{r0}_{t}"));
        let want_z = dump(layer, &format!("z_{r0}_{t}"));
        let want_g = dump(layer, &format!("g_{r0}_{t}"));
        let want_beta = dump(layer, &format!("beta_{r0}_{t}"));
        let (cg, rg) = rel(&run.g().iter().map(|g| g.ln()).collect::<Vec<_>>(), &want_g);
        let (cb, rb) = rel(&run.beta(), &want_beta);
        let (cz, rz) = rel(&run.z(), &want_z);
        let (cy, ry) = rel(&bf_to_f32(&run.y()), &want_y);
        let (co, ro) = rel(&h_to_f32(&run.out()), &want_out);
        eprintln!(
            "L{layer:02} {round:?} rows {r0}..{} | log g cos {cg:.8} rel {rg:.2e} | beta rel {rb:.1e} | z rel {rz:.1e} | core y cos {cy:.7} rel {ry:.2e} | norm out cos {co:.7} rel {ro:.2e}",
            r0 + t
        );
        assert!(cg >= 0.999999 && rg <= 1e-5, "L{layer} log g: cos {cg} rel {rg}");
        assert!(cb >= 0.99999 && rb <= 1e-5, "L{layer} beta: cos {cb} rel {rb}");
        assert!(cz >= 0.999999 && rz <= 1e-5, "L{layer} z: cos {cz} rel {rz}");
        assert!(cy >= 0.9999 && ry <= core_tol, "L{layer} {round:?} core y: cos {cy} rel {ry}");
        assert!(co >= 0.9999 && ro <= norm_tol, "L{layer} {round:?} norm out: cos {co} rel {ro}");
        worst.core = worst.core.max(ry);
        worst.norm = worst.norm.max(ro);
        run.commit(&ctx, &mut pool, None, true);
        r0 += t;
    }
    // the state after all rows against the oracle's (transposed: the oracle keeps [key][value])
    let want_s = dump(layer, "state_final");
    let (cs, rs) = rel(&transpose_state(&pool.read_state(0).unwrap()), &want_s);
    eprintln!("L{layer:02} {round:?} final state: cos {cs:.8} rel {rs:.2e}");
    assert!(cs >= 0.9999 && rs <= 8e-3, "L{layer} state: cos {cs} rel {rs}");
    worst.state = rs;
    worst
}

#[test]
fn real_layers_match_exllamav3_decode_graph() {
    need_gpu!();
    if kda_dir().is_none() {
        return eprintln!("REXL3_KDA_DIR not set: the GLM real-layer comparison is not run here");
    }
    for layer in manifest_layers() {
        // RN (REXL3's choice): the difference to the oracle is dominated by its round-toward-zero output
        let rn = compare_real(layer, Round::NearestEven, false);
        // RZ: the same pipeline with the oracle's rounding agrees to the fp32 noise of two different summation orders
        let rz = compare_real(layer, Round::TowardZero, true);
        eprintln!(
            "L{layer:02} summary: RN core {:.2e} norm {:.2e} | RZ core {:.2e} norm {:.2e} | state {:.1e}",
            rn.core, rn.norm, rz.core, rz.norm, rz.state
        );
    }
}

/// How far the oracle's decode graph is from the layer function with the STORED f32 dt_bias: REXL3 on real weights with f32 dt_bias
/// against REXL3 with the bf16 copy the BC graph reads (everything else identical). Prints the effect on the decay, the core
/// output, the norm output and the state. A MEASUREMENT (no bound to meet, so no test that could fail): ignored, run with
/// `--ignored bench_dt_bias`; the numbers are in docs/KDA.md.
#[test]
#[ignore]
fn bench_dt_bias_rounding_effect() {
    need_gpu!();
    if kda_dir().is_none() {
        return eprintln!("REXL3_KDA_DIR not set: nothing to measure");
    }
    for layer in manifest_layers() {
        let (wf, wb) = (layer_weights(layer, false), layer_weights(layer, true));
        let (cf, cb) = (Ctx::new(wf), Ctx::new(wb));
        let (mut pf, mut pb) = (pool_with(&[(0, &Slot::zeros())], 1), pool_with(&[(0, &Slot::zeros())], 1));
        let mut r0 = 0;
        let mut rows = 0;
        let (mut g_rel, mut y_rel, mut o_rel) = (0f64, 0f64, 0f64);
        let mut max_dg = 0f32;
        for t in manifest_steps() {
            let inp = Inp { x: f16s(&dump(layer, &format!("x_{r0}_{t}"))), qkv: dump(layer, &format!("qkv_{r0}_{t}")) };
            let mut a = fwd(&cf, &mut pf, &inp, &[t], &[0], true);
            let mut b = fwd(&cb, &mut pb, &inp, &[t], &[0], true);
            let (ga, gb) = (a.g(), b.g());
            max_dg = max_dg.max(ga.iter().zip(&gb).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max));
            g_rel = g_rel.max(rel(&gb, &ga).1);
            y_rel = y_rel.max(rel(&bf_to_f32(&b.y()), &bf_to_f32(&a.y())).1);
            o_rel = o_rel.max(rel(&h_to_f32(&b.out()), &h_to_f32(&a.out())).1);
            a.commit(&cf, &mut pf, None, true);
            b.commit(&cb, &mut pb, None, true);
            r0 += t;
            rows += t;
        }
        let s_rel = rel(&pb.read_state(0).unwrap(), &pf.read_state(0).unwrap()).1;
        eprintln!(
            "L{layer:02} bf16 dt_bias vs f32 dt_bias over {rows} rows: decay g relRMS {g_rel:.2e} (max abs {max_dg:.3e}) | core y {y_rel:.2e} | norm out {o_rel:.2e} | final state {s_rel:.2e}"
        );
    }
}

/// The pack loader: every tensor of layer 0 uploaded exactly as stored, and a wrong layer (an MLA layer has no KDA tensors) or a wrong geometry refused
/// with an error naming the tensor.
#[test]
fn layer_weights_load_from_the_pack_and_refuse_a_wrong_layer_or_geometry() {
    need_gpu!();
    let Ok(dir) = std::env::var("REXL3_GLM_DIR") else {
        return eprintln!("REXL3_GLM_DIR not set: the pack loader test is not run here");
    };
    let ck = gb_exl3::checkpoint::scan(std::path::Path::new(&dir), false).unwrap();
    let lw = LayerWeights::from_checkpoint(&ck, 0, dims()).unwrap();
    let host = layer_weights(0, false);
    let w = lw.view();
    let u16s = |b: &DevBuf| b.to_vec::<u16>().unwrap();
    assert_bits_eq("b_proj", &u16s(w.b), &host.b);
    assert_bits_eq("f_a_proj", &u16s(w.f_a), &host.f_a);
    assert_bits_eq("g_a_proj", &u16s(w.g_a), &host.g_a);
    assert_bits_eq("f_b_proj", &u16s(w.f_b), &host.f_b);
    assert_bits_eq("g_b_proj", &u16s(w.g_b), &host.g_b);
    assert_bits_eq("conv1d", &u16s(w.conv_w), &host.conv_w);
    assert_bits_eq("o_norm", &u16s(w.o_norm), &host.o_norm);
    assert_f32_bits_eq("A_log", &w.a_log.to_vec::<f32>().unwrap(), &host.a_log);
    assert_f32_bits_eq("dt_bias", &w.dt_bias.to_vec::<f32>().unwrap(), &host.dt_bias);
    let e = LayerWeights::from_checkpoint(&ck, 3, dims()).err().expect("layer 3 is an MLA layer");
    assert!(e.contains("is missing"), "{e}");
    let e = LayerWeights::from_checkpoint(&ck, 0, Dims { heads: 32, hidden: GLM_HIDDEN })
        .err()
        .expect("32 heads is the wrong geometry");
    assert!(e.contains("expected"), "{e}");
}

// ------------------------------------------------------------------------------------------------ golden hashes

/// A fixed, seeded scenario per shape; the hashes pin every bit of project, prep, window, replay, norm, commit and prefill.
fn golden_lines() -> String {
    let mut out = String::new();
    let mut rng = Rng(7001);
    let ctx = Ctx::new(HostW::random(&mut rng));
    let start = Slot::random(&mut rng);
    let inp = Inp::random(&mut rng, 128);
    let line = |out: &mut String, name: &str, h: u64| out.push_str(&format!("{name} {h:016x}\n"));
    for &w in &[1usize, 2, 8, 9, 33, 128] {
        let part = inp.slice(0, w);
        let mut pool = pool_with(&[(0, &start)], 1);
        let mut run = fwd(&ctx, &mut pool, &part, &[w], &[0], true);
        line(&mut out, &format!("window R={w} raw"), hash_u16(&run.raw()));
        line(&mut out, &format!("window R={w} g"), hash_f32(&run.g()));
        line(&mut out, &format!("window R={w} beta"), hash_f32(&run.beta()));
        line(&mut out, &format!("window R={w} z"), hash_f32(&run.z()));
        line(&mut out, &format!("window R={w} y"), hash_u16(&run.y()));
        line(&mut out, &format!("window R={w} norm"), hash_u16(&run.out()));
        let keeps: Vec<usize> = if w == 1 { vec![1] } else { vec![w / 2, w] };
        for keep in keeps {
            let mut p2 = pool_with(&[(0, &start)], 1);
            let mut r2 = fwd(&ctx, &mut p2, &part, &[w], &[0], true);
            r2.commit(&ctx, &mut p2, Some(&[keep as i32]), true);
            let (hs, hc) = hash_slot(&snap(&p2, 0));
            line(&mut out, &format!("window R={w} commit {keep} S"), hs);
            line(&mut out, &format!("window R={w} commit {keep} conv"), hc);
        }
        run.commit(&ctx, &mut pool, None, true);
        let (hs, hc) = hash_slot(&snap(&pool, 0));
        line(&mut out, &format!("window R={w} flip S"), hs);
        line(&mut out, &format!("window R={w} flip conv"), hc);
    }
    // 3 sequences of different lengths in one launch
    let lens = [3usize, 7, 1];
    let parts = Inp::concat(&[inp.slice(0, 3), inp.slice(3, 10), inp.slice(10, 11)]);
    let mut pool = pool_with(&[(0, &start), (1, &start), (2, &start)], 3);
    let mut run = fwd(&ctx, &mut pool, &parts, &lens, &[2, 0, 1], true);
    line(&mut out, "batch 3,7,1 norm", hash_u16(&run.out()));
    run.commit(&ctx, &mut pool, Some(&[2, 7, 0]), true);
    for s in 0..3 {
        let (hs, hc) = hash_slot(&snap(&pool, s));
        line(&mut out, &format!("batch 3,7,1 commit 2,7,0 slot {s} S"), hs);
        line(&mut out, &format!("batch 3,7,1 commit 2,7,0 slot {s} conv"), hc);
    }
    // prefill 300 rows from a zero state
    let big = Inp::random(&mut rng, 300);
    let mut p = pool_with(&[(0, &Slot::zeros())], 1);
    let o = run_prefill(&ctx, &big, 0, &mut p);
    line(&mut out, "prefill 300 norm", hash_u16(&o));
    let (hs, hc) = hash_slot(&snap(&p, 0));
    line(&mut out, "prefill 300 S", hs);
    line(&mut out, "prefill 300 conv", hc);
    // the rounding choice is part of the bits: RZ output at R=8
    let ctx_rz = Ctx::new(ctx.w.clone()).with_params(Params { round: Round::TowardZero, ..Params::glm() });
    let mut pool = pool_with(&[(0, &start)], 1);
    let run = fwd(&ctx_rz, &mut pool, &inp.slice(0, 8), &[8], &[0], true);
    line(&mut out, "window R=8 RZ y", hash_u16(&run.y()));
    out
}

fn golden_path() -> std::path::PathBuf {
    common::repo_root().join("testdata/gpu/golden_kda.txt")
}

/// `REXL3_GOLDEN_OUT=golden_kda.txt cargo test --release -p gb-cuda --test kda golden_kda_hashes -- --ignored`
/// (scripts/gpu_gate.sh diffs the file against testdata/gpu/golden_kda.txt).
#[test]
#[ignore]
fn golden_kda_hashes() {
    need_gpu!();
    let Ok(path) = std::env::var("REXL3_GOLDEN_OUT") else {
        return eprintln!("REXL3_GOLDEN_OUT not set: skipping");
    };
    let out = golden_lines();
    eprint!("{out}");
    std::fs::write(path, out).unwrap();
}

/// The committed goldens still hold (the same comparison the gate script makes with diff, runnable by itself).
#[test]
fn golden_kda_matches_committed() {
    need_gpu!();
    let Ok(want) = std::fs::read_to_string(golden_path()) else {
        return eprintln!("no committed golden_kda.txt: skipping");
    };
    let got = golden_lines();
    if got != want {
        let diff: Vec<String> =
            got.lines().zip(want.lines()).filter(|(a, b)| a != b).map(|(a, b)| format!("  got  {a}\n  want {b}")).collect();
        panic!(
            "golden_kda.txt differs in {} lines (first 6):\n{}",
            diff.len(),
            diff.iter().take(6).cloned().collect::<Vec<_>>().join("\n")
        );
    }
}

// ------------------------------------------------------------------------------------------------ (e) benchmarks

fn timed(st: &Stream, iters: usize, mut f: impl FnMut(usize)) -> f64 {
    for i in 0..5 {
        f(i);
    }
    st.sync().unwrap();
    let t0 = Instant::now();
    for i in 0..iters {
        f(i);
    }
    st.sync().unwrap();
    t0.elapsed().as_secs_f64() / iters as f64
}

/// The roofline for the same traffic pattern (every byte read once and written once): one device-to-device copy of 256 MiB.
#[test]
#[ignore]
fn bench_copy_roofline() {
    need_gpu!();
    let st = Stream::new().unwrap();
    let total = 256usize << 20;
    let src = DevBuf::zeroed(total).unwrap();
    let mut dst = DevBuf::zeroed(total).unwrap();
    let t = timed(&st, 20, |_| dst.copy_from(&src, total, &st).unwrap());
    eprintln!(
        "d2d copy of 256 MiB: {:.2} ms = {:.0} GB/s read+write ({:.0} GB/s of one-way traffic)",
        t * 1e3,
        2.0 * total as f64 / t / 1e9,
        total as f64 / t / 1e9
    );
}

/// Window, replay and the whole decode step across many slots (the pool is far larger than L2, so every launch reads its 4 MiB
/// state from DRAM and writes 4 MiB). GB/s counts read + write of the state: the roofline is the 240 GB/s streaming ceiling.
#[test]
#[ignore]
fn bench_step_window_replay() {
    need_gpu!();
    let mut rng = Rng(71);
    let ctx = Ctx::new(HostW::random(&mut rng));
    let n_slots = 128; // 128 x 2 x 4 MiB = 1 GiB of state: far beyond L2 for every shape below
    let mut pool = Pool::new(dims(), n_slots).unwrap();
    let mib = (H * D * D * 4) as f64;
    let gb = |bytes: f64, t: f64| bytes / t / 1e9;
    for &r_step in &[4usize, 8] {
        let p = Params { r_step, ..Params::glm() };
        for &(n_seq, rows) in &[(1usize, 1usize), (1, 8), (4, 1), (4, 8), (6, 8), (8, 1), (16, 1), (32, 1), (128, 1), (1, 128)] {
            let inp = Inp::random(&mut rng, rows * n_seq);
            let x = DevBuf::from_slice(&inp.x).unwrap();
            let qkv = DevBuf::from_slice(&inp.qkv).unwrap();
            let n_batches = n_slots / n_seq;
            let batches: Vec<BatchBuf> = (0..n_batches)
                .map(|b| BatchBuf::new(&vec![rows; n_seq], &(0..n_seq).map(|j| b * n_seq + j).collect::<Vec<_>>()).unwrap())
                .collect();
            let total = rows * n_seq;
            let mut rec = Record::alloc(dims(), total).unwrap();
            let mut ws = Workspace::alloc(dims(), total).unwrap();
            let mut out = DevBuf::zeroed(total * VD * 2).unwrap();
            let keep = DevBuf::from_slice(&vec![(rows as i32 / 2).max(1); n_seq]).unwrap();
            let w = ctx.dw.view();
            project(&ctx.st, dims(), &w, &p, &x, Dtype::F16, total, &mut rec, &mut ws).unwrap();
            prep(&ctx.st, dims(), &w, &qkv, Dtype::F32, &pool, &batches[0].view(), &mut rec, &mut ws).unwrap();
            let t_proj =
                timed(&ctx.st, 200, |_| project(&ctx.st, dims(), &w, &p, &x, Dtype::F16, total, &mut rec, &mut ws).unwrap());
            let t_prep = timed(&ctx.st, 200, |i| {
                prep(&ctx.st, dims(), &w, &qkv, Dtype::F32, &pool, &batches[i % n_batches].view(), &mut rec, &mut ws).unwrap()
            });
            let t_win = timed(&ctx.st, 400, |i| {
                window(&ctx.st, dims(), &p, &mut pool, &batches[i % n_batches].view(), &rec, &mut ws, true).unwrap()
            });
            let t_norm = timed(&ctx.st, 200, |_| norm(&ctx.st, dims(), &w, &p, total, &ws, &mut out).unwrap());
            let t_commit_flip = timed(&ctx.st, 200, |i| {
                commit(&ctx.st, dims(), &w, &p, &mut pool, &batches[i % n_batches].view(), &rec, &mut ws, None, true).unwrap()
            });
            let t_commit_replay = if rows > 1 {
                timed(&ctx.st, 200, |i| {
                    commit(&ctx.st, dims(), &w, &p, &mut pool, &batches[i % n_batches].view(), &rec, &mut ws, Some(&keep), true)
                        .unwrap()
                })
            } else {
                f64::NAN
            };
            let t_step = timed(&ctx.st, 200, |i| {
                let b = &batches[i % n_batches].view();
                forward_window(
                    &ctx.st,
                    dims(),
                    &w,
                    &p,
                    &x,
                    Dtype::F16,
                    &qkv,
                    Dtype::F32,
                    &mut pool,
                    b,
                    &mut rec,
                    &mut ws,
                    &mut out,
                    true,
                )
                .unwrap();
                commit(&ctx.st, dims(), &w, &p, &mut pool, b, &rec, &mut ws, None, true).unwrap();
            });
            let state = mib * n_seq as f64;
            eprintln!(
                "r_step {r_step} seqs {n_seq} rows {rows}: window {:.1} us ({:.0} GB/s of read+write) | replay-commit {:.1} us ({:.0} GB/s) | flip-commit {:.1} us | proj {:.1} us | prep {:.1} us | norm {:.1} us | whole step+commit {:.1} us",
                t_win * 1e6,
                gb(2.0 * state, t_win),
                t_commit_replay * 1e6,
                gb(2.0 * state, t_commit_replay),
                t_commit_flip * 1e6,
                t_proj * 1e6,
                t_prep * 1e6,
                t_norm * 1e6,
                t_step * 1e6,
            );
        }
    }
}

/// The projections with COLD weights: 8 sets of layer weights (8 x 6.8 MB of f16 weights, more than the 24 MiB L2) used in rotation, as
/// consecutive layers do in a real step. Time per layer; the roofline for the weight stream alone is 6.8 MB / 240 GB/s = 28 us.
#[test]
#[ignore]
fn bench_projections_cold() {
    need_gpu!();
    let mut rng = Rng(73);
    let sets: Vec<Ctx> = (0..8).map(|_| Ctx::new(HostW::random(&mut rng))).collect();
    let st = &sets[0].st;
    for rows in [1usize, 2, 8, 9, 32, 128, 512] {
        let inp = Inp::random(&mut rng, rows);
        let x = DevBuf::from_slice(&inp.x).unwrap();
        let mut rec = Record::alloc(dims(), rows).unwrap();
        let mut ws = Workspace::alloc(dims(), rows).unwrap();
        for (name, stages, bytes) in [("both", 3u8, 6.8e6), ("b/f_a/g_a", 1, 2.6e6), ("f_b/g_b + gate", 2, 4.2e6)] {
            let t = timed(st, 400, |i| {
                let c = &sets[i % 8];
                project_stage(st, dims(), &c.dw.view(), &c.p, &x, Dtype::F16, rows, &mut rec, &mut ws, stages).unwrap()
            });
            eprintln!(
                "projections, cold weights, {rows} rows, {name}: {:.1} us per layer ({:.0} GB/s of weights)",
                t * 1e6,
                bytes / t / 1e9
            );
        }
    }
}

/// Prefill cost with the exact row-by-row path: per layer and over 34 layers.
#[test]
#[ignore]
fn bench_prefill_exact() {
    need_gpu!();
    let mut rng = Rng(72);
    let ctx = Ctx::new(HostW::random(&mut rng));
    let rows = 2048;
    let inp = Inp::random(&mut rng, rows);
    let x = DevBuf::from_slice(&inp.x).unwrap();
    let qkv = DevBuf::from_slice(&inp.qkv).unwrap();
    let mut pool = Pool::new(dims(), 1).unwrap();
    let mut rec = Record::alloc(dims(), PREFILL_PIECE_ROWS).unwrap();
    let mut ws = Workspace::alloc(dims(), PREFILL_PIECE_ROWS).unwrap();
    let mut out = DevBuf::zeroed(rows * VD * 2).unwrap();
    let t = timed(&ctx.st, 3, |_| {
        prefill(
            &ctx.st,
            dims(),
            &ctx.dw.view(),
            &ctx.p,
            &x,
            Dtype::F16,
            &qkv,
            Dtype::F32,
            &mut pool,
            0,
            rows,
            &mut rec,
            &mut ws,
            &mut out,
        )
        .unwrap()
    });
    eprintln!(
        "exact prefill {rows} rows, 1 layer: {:.2} ms ({:.1} us/row); 34 layers: {:.1} ms = {:.0} tokens/s (KDA layers only)",
        t * 1e3,
        t / rows as f64 * 1e6,
        t * 34e3,
        rows as f64 / (t * 34.0)
    );
}
