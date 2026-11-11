// The gated delta net cases, for the value-head shape `SHAPE` the including test target sets (tests/gdn.rs: 48 value heads,
// tests/gdn_hv32.rs: 32): f64 CPU reference, bit-exact replay/row-invariance gates, ExLlamaV3 dumps (48 heads only), benchmarks.
// Included, not a module, so both targets keep the plain test names; each declares `mod common` and `SHAPE`.

use common::{cos_rel_rms, ref_npy, Rng};
use gb_cuda::gdn::*;
use gb_cuda::num::{bf16_to_f32, f32_to_bf16};
use gb_cuda::{device_available, DevBuf, Stream};
use std::time::Instant;

const KH: usize = KEY_HEADS;
const VH: usize = SHAPE.value_heads();
const D: usize = HEAD_DIM;
const C: usize = SHAPE.conv_dim();
/// f32 elements of one S slot and bf16 elements of one conv slot, for `SHAPE`.
const S_SLOT: usize = SHAPE.s_slot_elems();
const CONV_SLOT: usize = SHAPE.conv_slot_elems();
/// Value heads that share a key head.
const GROUP: usize = VH / KH;

macro_rules! need_gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

// ------------------------------------------------------------------------------------------------ host data

#[derive(Clone)]
struct Weights {
    conv_w: Vec<u16>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
}

impl Weights {
    fn random(rng: &mut Rng) -> Self {
        Self {
            conv_w: rng.vec(C * 4, -0.6, 0.6).iter().map(|&x| f32_to_bf16(x)).collect(),
            a_log: rng.vec(VH, -1.5, 1.5),
            dt_bias: rng.vec(VH, -1.0, 1.0),
        }
    }
}

#[derive(Clone, Default)]
struct Inp {
    qkv: Vec<u16>,
    a: Vec<u16>,
    b: Vec<u16>,
}

impl Inp {
    fn random(rng: &mut Rng, rows: usize) -> Self {
        let bf = |v: Vec<f32>| -> Vec<u16> { v.iter().map(|&x| f32_to_bf16(x)).collect() };
        Self { qkv: bf(rng.vec(rows * C, -3.0, 3.0)), a: bf(rng.vec(rows * VH, -3.0, 3.0)), b: bf(rng.vec(rows * VH, -3.0, 3.0)) }
    }
    fn rows(&self) -> usize {
        self.a.len() / VH
    }
    fn slice(&self, r0: usize, r1: usize) -> Inp {
        Inp { qkv: self.qkv[r0 * C..r1 * C].to_vec(), a: self.a[r0 * VH..r1 * VH].to_vec(), b: self.b[r0 * VH..r1 * VH].to_vec() }
    }
    fn concat(parts: &[Inp]) -> Inp {
        let mut o = Inp::default();
        for p in parts {
            o.qkv.extend(&p.qkv);
            o.a.extend(&p.a);
            o.b.extend(&p.b);
        }
        o
    }
}

/// Host copy of both pools.
#[derive(Clone, PartialEq)]
struct HostPools {
    s: Vec<f32>,
    conv: Vec<u16>,
}

impl HostPools {
    fn random(rng: &mut Rng, slots: usize) -> Self {
        Self {
            s: rng.vec(slots * S_SLOT, -0.4, 0.4),
            conv: rng.vec(slots * CONV_SLOT, -3.0, 3.0).iter().map(|&x| f32_to_bf16(x)).collect(),
        }
    }
    fn zeros(slots: usize) -> Self {
        Self { s: vec![0.0; slots * S_SLOT], conv: vec![0; slots * CONV_SLOT] }
    }
    fn slot_s(&self, slot: usize) -> &[f32] {
        &self.s[slot * S_SLOT..(slot + 1) * S_SLOT]
    }
    fn slot_conv(&self, slot: usize) -> &[u16] {
        &self.conv[slot * CONV_SLOT..(slot + 1) * CONV_SLOT]
    }
}

struct DevPools {
    s: DevBuf,
    conv: DevBuf,
}

impl DevPools {
    fn new(h: &HostPools) -> Self {
        Self { s: DevBuf::from_slice(&h.s).unwrap(), conv: DevBuf::from_slice(&h.conv).unwrap() }
    }
    fn host(&self) -> HostPools {
        HostPools { s: self.s.to_vec().unwrap(), conv: self.conv.to_vec().unwrap() }
    }
}

struct Ctx {
    st: Stream,
    w: Weights,
    conv_w: DevBuf,
    a_log: DevBuf,
    dt_bias: DevBuf,
}

impl Ctx {
    fn new(w: Weights) -> Self {
        Self {
            st: Stream::new().unwrap(),
            conv_w: DevBuf::from_slice(&w.conv_w).unwrap(),
            a_log: DevBuf::from_slice(&w.a_log).unwrap(),
            dt_bias: DevBuf::from_slice(&w.dt_bias).unwrap(),
            w,
        }
    }
    fn weights(&self) -> PreWeights<'_> {
        PreWeights { conv_w: &self.conv_w, a_log: &self.a_log, dt_bias: &self.dt_bias }
    }
}

struct DevInp {
    qkv: DevBuf,
    a: DevBuf,
    b: DevBuf,
}

impl DevInp {
    fn new(i: &Inp) -> Self {
        Self {
            qkv: DevBuf::from_slice(&i.qkv).unwrap(),
            a: DevBuf::from_slice(&i.a).unwrap(),
            b: DevBuf::from_slice(&i.b).unwrap(),
        }
    }
    fn view(&self) -> PreInputs<'_> {
        PreInputs { qkv: &self.qkv, a: &self.a, b: &self.b }
    }
}

#[derive(PartialEq, Clone)]
struct HostRec {
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    g: Vec<f32>,
    beta: Vec<f32>,
}

/// pre + window of a whole batch; keeps the device record so `commit` can follow.
struct Run {
    x: DevInp,
    batch: BatchBuf,
    rec: Record,
    o: DevBuf,
}

fn run_pre_window(ctx: &Ctx, inp: &Inp, lens: &[usize], slots: &[usize], pools: &DevPools) -> Run {
    let batch = BatchBuf::new(lens, slots).unwrap();
    assert_eq!(batch.view().rows, inp.rows());
    let x = DevInp::new(inp);
    let rows = batch.view().rows;
    let mut rec = Record::alloc_shaped(rows, SHAPE).unwrap();
    pre(&ctx.st, &x.view(), &ctx.weights(), &pools.conv, &batch.view(), &mut rec).unwrap();
    let mut o = DevBuf::zeroed(rows * VH * D * 2).unwrap();
    window(&ctx.st, &rec, &pools.s, &batch.view(), &mut o).unwrap();
    ctx.st.sync().unwrap();
    Run { x, batch, rec, o }
}

impl Run {
    fn host_rec(&self) -> HostRec {
        let n = self.batch.view().rows;
        let take = |b: &DevBuf, len: usize| -> Vec<u16> { b.to_vec::<u16>().unwrap()[..len].to_vec() };
        HostRec {
            q: take(&self.rec.q, n * KH * D),
            k: take(&self.rec.k, n * KH * D),
            v: take(&self.rec.v, n * VH * D),
            g: self.rec.g_decay.to_vec::<f32>().unwrap()[..n * VH].to_vec(),
            beta: self.rec.beta.to_vec::<f32>().unwrap()[..n * VH].to_vec(),
        }
    }
    fn host_o(&self) -> Vec<u16> {
        self.o.to_vec::<u16>().unwrap()[..self.batch.view().rows * VH * D].to_vec()
    }
    fn commit(&self, ctx: &Ctx, keep: &[i32], pools: &mut DevPools) {
        let k = DevBuf::from_slice(keep).unwrap();
        commit(&ctx.st, &self.rec, &self.x.qkv, &mut pools.s, &mut pools.conv, &self.batch.view(), &k).unwrap();
        ctx.st.sync().unwrap();
    }
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

// ------------------------------------------------------------------------------------------------ CPU f64 reference

fn bf(x: f64) -> f64 {
    bf16_to_f32(f32_to_bf16(x as f32)) as f64
}

struct CpuRec {
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    g: Vec<f64>,
    beta: Vec<f64>,
}

/// Spec 5: conv over [state; rows] + silu (one bf16 rounding) + l2norm (+ 1/sqrt(128) on q) + gating.
fn cpu_pre(inp: &Inp, w: &Weights, conv_state: &[u16]) -> CpuRec {
    let rows = inp.rows();
    let x = |t: isize, ch: usize| -> f64 {
        if t >= 0 {
            bf16_to_f32(inp.qkv[t as usize * C + ch]) as f64
        } else {
            bf16_to_f32(conv_state[(3 + t) as usize * C + ch]) as f64
        }
    };
    let mut r = CpuRec {
        q: vec![0.0; rows * KH * D],
        k: vec![0.0; rows * KH * D],
        v: vec![0.0; rows * VH * D],
        g: vec![0.0; rows * VH],
        beta: vec![0.0; rows * VH],
    };
    for t in 0..rows {
        let mut c = vec![0f64; C];
        for (ch, cv) in c.iter_mut().enumerate() {
            let mut acc = 0f64;
            for kk in 0..4 {
                acc += bf16_to_f32(w.conv_w[ch * 4 + kk]) as f64 * x(t as isize - 3 + kk as isize, ch);
            }
            *cv = bf(acc / (1.0 + (-acc).exp()));
        }
        for h in 0..2 * KH {
            let seg = &c[h * D..(h + 1) * D];
            let inv = 1.0 / (seg.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt();
            let (dst, hh, scale) = if h < KH { (&mut r.q, h, 1.0 / (D as f64).sqrt()) } else { (&mut r.k, h - KH, 1.0) };
            for i in 0..D {
                dst[(t * KH + hh) * D + i] = bf(seg[i] * inv * scale);
            }
        }
        r.v[t * VH * D..(t + 1) * VH * D].copy_from_slice(&c[2 * KH * D..]);
        for h in 0..VH {
            let av = bf16_to_f32(inp.a[t * VH + h]) as f64 + w.dt_bias[h] as f64;
            let sp = if av > 20.0 { av } else { av.exp().ln_1p() };
            r.g[t * VH + h] = (-(w.a_log[h] as f64).exp() * sp).exp();
            r.beta[t * VH + h] = 1.0 / (1.0 + (-(bf16_to_f32(inp.b[t * VH + h]) as f64)).exp());
        }
    }
    r
}

/// One recurrence step for all 48 heads (spec 5, steps 1-5). `s` is `[48][128 j][128 i]`.
fn cpu_step(s: &mut [f64], q: &[f64], k: &[f64], v: &[f64], g: &[f64], beta: &[f64], o: Option<&mut [f64]>) {
    let mut o = o;
    for h in 0..VH {
        let hk = h / GROUP;
        let (kk, qq) = (&k[hk * D..(hk + 1) * D], &q[hk * D..(hk + 1) * D]);
        for j in 0..D {
            let row = &mut s[(h * D + j) * D..(h * D + j + 1) * D];
            let mut r = 0.0;
            for i in 0..D {
                row[i] *= g[h];
                r += row[i] * kk[i];
            }
            let delta = (v[h * D + j] - r) * beta[h];
            let mut out = 0.0;
            for i in 0..D {
                row[i] += kk[i] * delta;
                out += row[i] * qq[i];
            }
            if let Some(o) = o.as_deref_mut() {
                o[h * D + j] = out;
            }
        }
    }
}

fn to_f64_bf(v: &[u16]) -> Vec<f64> {
    v.iter().map(|&x| bf16_to_f32(x) as f64).collect()
}

/// Run `n` rows of the sequence starting at row `r0` of the record through the CPU recurrence from `s`; returns o.
fn cpu_run(s: &mut [f64], rec: &HostRec, r0: usize, n: usize, want_o: bool) -> Vec<f64> {
    let (q, k, v) = (to_f64_bf(&rec.q), to_f64_bf(&rec.k), to_f64_bf(&rec.v));
    let mut out = vec![0.0; if want_o { n * VH * D } else { 0 }];
    for t in 0..n {
        let r = r0 + t;
        let o = if want_o { Some(&mut out[t * VH * D..(t + 1) * VH * D]) } else { None };
        let gg: Vec<f64> = rec.g[r * VH..(r + 1) * VH].iter().map(|&x| x as f64).collect();
        let bb: Vec<f64> = rec.beta[r * VH..(r + 1) * VH].iter().map(|&x| x as f64).collect();
        cpu_step(
            s,
            &q[r * KH * D..(r + 1) * KH * D],
            &k[r * KH * D..(r + 1) * KH * D],
            &v[r * VH * D..(r + 1) * VH * D],
            &gg,
            &bb,
            o,
        );
    }
    out
}

fn slot_f64(h: &HostPools, slot: usize) -> Vec<f64> {
    h.slot_s(slot).iter().map(|&x| x as f64).collect()
}

fn bf_to_f32(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&x| bf16_to_f32(x)).collect()
}

fn rel(got: &[f32], want: &[f32]) -> (f64, f64) {
    cos_rel_rms(got, want)
}

fn f64s_to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

/// Conv state after committing `n` rows: last 3 rows of concat(old 3, rows 0..n).
fn cpu_conv_commit(old: &[u16], qkv: &[u16], n: usize) -> Vec<u16> {
    let mut all: Vec<u16> = old.to_vec();
    all.extend_from_slice(&qkv[..n * C]);
    all[n * C..(n + 3) * C].to_vec()
}

// ------------------------------------------------------------------------------------------------ (a) f64 reference

#[test]
fn pre_window_commit_match_f64_reference() {
    need_gpu!();
    let mut rng = Rng(11);
    let ctx = Ctx::new(Weights::random(&mut rng));
    let lens = [5usize, 1, 9, 3];
    let slots = [3usize, 0, 2, 4];
    let n_slots = 6;
    let parts: Vec<Inp> = lens.iter().map(|&l| Inp::random(&mut rng, l)).collect();
    let inp = Inp::concat(&parts);
    let host = HostPools::random(&mut rng, n_slots);
    let mut pools = DevPools::new(&host);
    let run = run_pre_window(&ctx, &inp, &lens, &slots, &pools);
    let rec = run.host_rec();
    let o = run.host_o();
    let mut r0 = 0;
    for (si, &len) in lens.iter().enumerate() {
        let slot = slots[si];
        let cpu = cpu_pre(&parts[si], &ctx.w, host.slot_conv(slot));
        let cmp = |what: &str, got: &[f32], want: &[f64], tol: f64| {
            let (cos, rr) = rel(got, &f64s_to_f32(want));
            assert!(rr <= tol && cos >= 0.99999, "seq {si} {what}: cos {cos} relRMS {rr}");
        };
        cmp("q", &bf_to_f32(&rec.q[r0 * KH * D..(r0 + len) * KH * D]), &cpu.q, 3e-3);
        cmp("k", &bf_to_f32(&rec.k[r0 * KH * D..(r0 + len) * KH * D]), &cpu.k, 3e-3);
        cmp("v", &bf_to_f32(&rec.v[r0 * VH * D..(r0 + len) * VH * D]), &cpu.v, 3e-3);
        cmp("g", &rec.g[r0 * VH..(r0 + len) * VH], &cpu.g, 1e-5);
        cmp("beta", &rec.beta[r0 * VH..(r0 + len) * VH], &cpu.beta, 1e-5);
        // window output: the CPU recurrence fed with the GPU's own record isolates the recurrence
        let mut s = slot_f64(&host, slot);
        let want = cpu_run(&mut s, &rec, r0, len, true);
        let (cos, rr) = rel(&bf_to_f32(&o[r0 * VH * D..(r0 + len) * VH * D]), &f64s_to_f32(&want));
        assert!(rr <= 3e-3 && cos >= 0.99999, "seq {si} window o: cos {cos} relRMS {rr}");
        r0 += len;
    }
    // the window never touched the committed state
    assert!(pools.host() == host, "window modified the pools");

    // commit different keeps per sequence (0 = untouched, 1, len, and a partial one)
    let keep = [3i32, 1, 0, 3];
    run.commit(&ctx, &keep, &mut pools);
    let after = pools.host();
    let mut r0 = 0;
    for (si, &len) in lens.iter().enumerate() {
        let slot = slots[si];
        let n = keep[si] as usize;
        let mut s = slot_f64(&host, slot);
        cpu_run(&mut s, &rec, r0, n, false);
        let (cos, rr) = rel(after.slot_s(slot), &f64s_to_f32(&s));
        assert!(rr <= 2e-6 && cos >= 0.9999999, "seq {si} committed S: cos {cos} relRMS {rr}");
        let want_conv = cpu_conv_commit(host.slot_conv(slot), &parts[si].qkv, n);
        assert_bits_eq("conv after commit", after.slot_conv(slot), &want_conv);
        r0 += len;
    }
    // slots not in the batch are untouched
    for slot in [1usize, 5] {
        assert_f32_bits_eq("unused S slot", after.slot_s(slot), host.slot_s(slot));
        assert_bits_eq("unused conv slot", after.slot_conv(slot), host.slot_conv(slot));
    }
}

#[test]
fn prefill_matches_f64_reference() {
    need_gpu!();
    let mut rng = Rng(12);
    let ctx = Ctx::new(Weights::random(&mut rng));
    let lens = [70usize, 33, 1];
    let slots = [2usize, 0, 1];
    let parts: Vec<Inp> = lens.iter().map(|&l| Inp::random(&mut rng, l)).collect();
    let inp = Inp::concat(&parts);
    let host = HostPools::random(&mut rng, 3);
    let mut pools = DevPools::new(&host);
    let batch = BatchBuf::new(&lens, &slots).unwrap();
    let x = DevInp::new(&inp);
    let mut ws = DevBuf::zeroed(prefill_workspace_bytes_shaped(SHAPE, batch.view().rows)).unwrap();
    let mut o = DevBuf::zeroed(batch.view().rows * VH * D * 2).unwrap();
    prefill_shaped(SHAPE, &ctx.st, &x.view(), &ctx.weights(), &mut pools.s, &mut pools.conv, &batch.view(), &mut ws, &mut o)
        .unwrap();
    ctx.st.sync().unwrap();
    let o: Vec<u16> = o.to_vec().unwrap();
    let after = pools.host();
    let mut r0 = 0;
    for (si, &len) in lens.iter().enumerate() {
        let slot = slots[si];
        let cpu = cpu_pre(&parts[si], &ctx.w, host.slot_conv(slot));
        // CPU record rounded to bf16 exactly as the kernel stores it (pre is checked separately above)
        let cr = HostRec {
            q: cpu.q.iter().map(|&x| f32_to_bf16(x as f32)).collect(),
            k: cpu.k.iter().map(|&x| f32_to_bf16(x as f32)).collect(),
            v: cpu.v.iter().map(|&x| f32_to_bf16(x as f32)).collect(),
            g: cpu.g.iter().map(|&x| x as f32).collect(),
            beta: cpu.beta.iter().map(|&x| x as f32).collect(),
        };
        let mut s = slot_f64(&host, slot);
        let want = cpu_run(&mut s, &cr, 0, len, true);
        let (cos, rr) = rel(&bf_to_f32(&o[r0 * VH * D..(r0 + len) * VH * D]), &f64s_to_f32(&want));
        assert!(rr <= 6e-3 && cos >= 0.9999, "seq {si} prefill o: cos {cos} relRMS {rr}");
        let (cos, rr) = rel(after.slot_s(slot), &f64s_to_f32(&s));
        assert!(rr <= 5e-3 && cos >= 0.9999, "seq {si} prefill S: cos {cos} relRMS {rr}");
        let want_conv = cpu_conv_commit(host.slot_conv(slot), &parts[si].qkv, len);
        assert_bits_eq("prefill conv", after.slot_conv(slot), &want_conv);
        r0 += len;
    }
}

// ------------------------------------------------------------------------------------------------ (b) bit-exact gates

/// Serial decode: for each row, pre(1 row) -> window(1 row) -> commit(1). Returns per-row record/o bits and the pool
/// snapshot after every step (index m = after m rows).
struct Serial {
    rec_q: Vec<u16>,
    rec_k: Vec<u16>,
    rec_v: Vec<u16>,
    rec_g: Vec<f32>,
    rec_b: Vec<f32>,
    o: Vec<u16>,
    states: Vec<HostPools>,
}

fn run_serial(ctx: &Ctx, inp: &Inp, slot: usize, start: &HostPools) -> Serial {
    let mut pools = DevPools::new(start);
    let mut sr = Serial {
        rec_q: vec![],
        rec_k: vec![],
        rec_v: vec![],
        rec_g: vec![],
        rec_b: vec![],
        o: vec![],
        states: vec![start.clone()],
    };
    for t in 0..inp.rows() {
        let run = run_pre_window(ctx, &inp.slice(t, t + 1), &[1], &[slot], &pools);
        let r = run.host_rec();
        sr.rec_q.extend(r.q);
        sr.rec_k.extend(r.k);
        sr.rec_v.extend(r.v);
        sr.rec_g.extend(r.g);
        sr.rec_b.extend(r.beta);
        sr.o.extend(run.host_o());
        run.commit(ctx, &[1], &mut pools);
        sr.states.push(pools.host());
    }
    sr
}

#[test]
fn window_and_commit_equal_serial_decode_bit_for_bit() {
    need_gpu!();
    let mut rng = Rng(21);
    let ctx = Ctx::new(Weights::random(&mut rng));
    let w = 8;
    let slot = 2;
    let inp = Inp::random(&mut rng, w);
    let start = HostPools::random(&mut rng, 4);
    let sr = run_serial(&ctx, &inp, slot, &start);

    // one 8-row window from the same initial pools
    let pools = DevPools::new(&start);
    let run = run_pre_window(&ctx, &inp, &[w], &[slot], &pools);
    let r = run.host_rec();
    assert_bits_eq("record q", &r.q, &sr.rec_q);
    assert_bits_eq("record k", &r.k, &sr.rec_k);
    assert_bits_eq("record v", &r.v, &sr.rec_v);
    assert_f32_bits_eq("record g", &r.g, &sr.rec_g);
    assert_f32_bits_eq("record beta", &r.beta, &sr.rec_b);
    assert_bits_eq("window o == serial o", &run.host_o(), &sr.o);
    assert!(pools.host() == start, "window must not touch the pools");

    // commit(m) after the window == m serial steps, for every m (state and conv, all slots)
    for m in 0..=w {
        let mut p = DevPools::new(&start);
        run.commit(&ctx, &[m as i32], &mut p);
        let h = p.host();
        assert_f32_bits_eq(&format!("S after commit({m})"), &h.s, &sr.states[m].s);
        assert_bits_eq(&format!("conv after commit({m})"), &h.conv, &sr.states[m].conv);
    }

    // commit(m), then a fresh window over the remaining rows == the serial outputs of those rows
    for m in [1usize, 3, 7] {
        let mut p = DevPools::new(&start);
        run.commit(&ctx, &[m as i32], &mut p);
        let rest = run_pre_window(&ctx, &inp.slice(m, w), &[w - m], &[slot], &p);
        assert_bits_eq(&format!("window after commit({m})"), &rest.host_o(), &sr.o[m * VH * D..]);
        let rr = rest.host_rec();
        assert_bits_eq("rest record q", &rr.q, &sr.rec_q[m * KH * D..]);
        assert_bits_eq("rest record v", &rr.v, &sr.rec_v[m * VH * D..]);
    }
}

#[test]
fn batching_does_not_change_bits() {
    need_gpu!();
    let mut rng = Rng(31);
    let ctx = Ctx::new(Weights::random(&mut rng));
    let lens = [5usize, 1, 8];
    let parts: Vec<Inp> = lens.iter().map(|&l| Inp::random(&mut rng, l)).collect();
    // identical per-slot state in every arrangement
    let n_slots = 6;
    let start = HostPools::random(&mut rng, n_slots);
    let keep_of = |i: usize| [2i32, 1, 8][i];

    // reference: each sequence alone in slot `slot_of[i]`
    let slot_of = [4usize, 1, 3];
    let mut solo = Vec::new();
    for i in 0..3 {
        let mut p = DevPools::new(&start);
        let run = run_pre_window(&ctx, &parts[i], &[lens[i]], &[slot_of[i]], &p);
        let (rec, o) = (run.host_rec(), run.host_o());
        run.commit(&ctx, &[keep_of(i)], &mut p);
        solo.push((rec, o, p.host()));
    }

    // arrangements: orders and a different slot for the same data in another batch
    let arrangements: [([usize; 3], [usize; 3]); 3] = [([0, 1, 2], [4, 1, 3]), ([2, 0, 1], [4, 1, 3]), ([1, 2, 0], [0, 5, 2])];
    for (order, slots) in arrangements {
        // `order[p]` = which sequence sits at batch position p; its slot is slots[p] (the *state* at that slot is
        // copied from the solo run's slot so the inputs match)
        let mut h = start.clone();
        for (p, &i) in order.iter().enumerate() {
            let (src, dst) = (slot_of[i], slots[p]);
            let (s, c) = (start.slot_s(src).to_vec(), start.slot_conv(src).to_vec());
            h.s[dst * S_SLOT..(dst + 1) * S_SLOT].copy_from_slice(&s);
            h.conv[dst * CONV_SLOT..(dst + 1) * CONV_SLOT].copy_from_slice(&c);
        }
        let blens: Vec<usize> = order.iter().map(|&i| lens[i]).collect();
        let inp = Inp::concat(&order.iter().map(|&i| parts[i].clone()).collect::<Vec<_>>());
        let mut p = DevPools::new(&h);
        let run = run_pre_window(&ctx, &inp, &blens, &slots, &p);
        let (rec, o) = (run.host_rec(), run.host_o());
        let keep: Vec<i32> = order.iter().map(|&i| keep_of(i)).collect();
        run.commit(&ctx, &keep, &mut p);
        let after = p.host();
        let mut r0 = 0;
        for (pos, &i) in order.iter().enumerate() {
            let len = lens[i];
            let (srec, so, sp) = &solo[i];
            let sr0 = 0usize; // the solo run holds only this sequence
            let tag = format!("order {order:?} slots {slots:?} seq {i}");
            assert_bits_eq(
                &format!("{tag} q"),
                &rec.q[r0 * KH * D..(r0 + len) * KH * D],
                &srec.q[sr0 * KH * D..(sr0 + len) * KH * D],
            );
            assert_bits_eq(
                &format!("{tag} k"),
                &rec.k[r0 * KH * D..(r0 + len) * KH * D],
                &srec.k[sr0 * KH * D..(sr0 + len) * KH * D],
            );
            assert_bits_eq(
                &format!("{tag} v"),
                &rec.v[r0 * VH * D..(r0 + len) * VH * D],
                &srec.v[sr0 * VH * D..(sr0 + len) * VH * D],
            );
            assert_f32_bits_eq(&format!("{tag} g"), &rec.g[r0 * VH..(r0 + len) * VH], &srec.g[sr0 * VH..(sr0 + len) * VH]);
            assert_f32_bits_eq(
                &format!("{tag} beta"),
                &rec.beta[r0 * VH..(r0 + len) * VH],
                &srec.beta[sr0 * VH..(sr0 + len) * VH],
            );
            assert_bits_eq(&format!("{tag} o"), &o[r0 * VH * D..(r0 + len) * VH * D], &so[sr0 * VH * D..(sr0 + len) * VH * D]);
            assert_f32_bits_eq(&format!("{tag} S"), after.slot_s(slots[pos]), sp.slot_s(slot_of[i]));
            assert_bits_eq(&format!("{tag} conv"), after.slot_conv(slots[pos]), sp.slot_conv(slot_of[i]));
            r0 += len;
        }
    }
}

// ------------------------------------------------------------------------------------------------ (c) conv commit

#[test]
fn conv_commit_for_every_keep() {
    need_gpu!();
    let mut rng = Rng(41);
    let ctx = Ctx::new(Weights::random(&mut rng));
    let rows = 9;
    let inp = Inp::random(&mut rng, rows);
    let start = HostPools::random(&mut rng, 2);
    let pools0 = DevPools::new(&start);
    let run = run_pre_window(&ctx, &inp, &[rows], &[1], &pools0);
    for keep in [0usize, 1, 2, 3, 5, 8, 9] {
        let mut p = DevPools::new(&start);
        run.commit(&ctx, &[keep as i32], &mut p);
        let h = p.host();
        assert_bits_eq(&format!("conv keep={keep}"), h.slot_conv(1), &cpu_conv_commit(start.slot_conv(1), &inp.qkv, keep));
        assert_bits_eq("other conv slot", h.slot_conv(0), start.slot_conv(0));
        if keep == 0 {
            assert_f32_bits_eq("S keep=0", &h.s, &start.s);
        }
    }
    // keep larger than the sequence is clamped to its length
    let mut p = DevPools::new(&start);
    run.commit(&ctx, &[100], &mut p);
    assert_bits_eq("conv keep=100", p.host().slot_conv(1), &cpu_conv_commit(start.slot_conv(1), &inp.qkv, rows));
}

// ------------------------------------------------------------------------------------------------ (d) deferred commit

/// What the previous round left behind for the deferred path: its record, raw qkv rows, batch and keeps.
struct Prev {
    rec: Record,
    batch: BatchBuf,
    slots: Vec<usize>,
    starts: Vec<usize>,
    keep: Vec<i32>,
}

fn dev_i32(v: &[i32]) -> DevBuf {
    DevBuf::from_slice(v).unwrap()
}

/// Replay `prev`'s owed rows of every sequence whose entry in `mask` is true (the others keep 0) into `pools` with the
/// standalone commit kernel.
fn settle(ctx: &Ctx, prev: &Prev, pools: &mut DevPools, mask: impl Fn(usize) -> bool) {
    let keep: Vec<i32> = prev.keep.iter().enumerate().map(|(i, &k)| if mask(prev.slots[i]) { k } else { 0 }).collect();
    commit_state(&ctx.st, &prev.rec, &mut pools.s, &prev.batch.view(), &dev_i32(&keep)).unwrap();
    ctx.st.sync().unwrap();
}

/// The deferred commit (S replay folded into the next window kernel, conv committed eagerly) must leave exactly the bits
/// of `window; commit` round after round: same outputs, and the same state once the owed rows are settled, for random
/// batches that gain, lose and reorder sequences, with keeps from 0 to the full window.
#[test]
fn deferred_commit_equals_commit_then_window_bit_for_bit() {
    need_gpu!();
    let mut rng = Rng(77);
    let ctx = Ctx::new(Weights::random(&mut rng));
    let n_slots = 6;
    let start = HostPools::random(&mut rng, n_slots);
    let mut a = DevPools::new(&start); // reference: window, then commit
    let mut b = DevPools::new(&start); // deferred
    let mut prev: Option<Prev> = None;
    let below = |rng: &mut Rng, n: usize| (rng.next_u64() % n as u64) as usize;
    let mut owed_rounds = 0;
    for round in 0..24 {
        // a random subset of the slots, in a random order, with random window lengths and keeps
        let mut slots: Vec<usize> = (0..n_slots).collect();
        for i in (1..slots.len()).rev() {
            slots.swap(i, below(&mut rng, i + 1));
        }
        slots.truncate(1 + below(&mut rng, 4));
        let lens: Vec<usize> = slots.iter().map(|_| 1 + below(&mut rng, 8)).collect();
        let keep: Vec<i32> = lens
            .iter()
            .map(|&l| match below(&mut rng, 4) {
                0 => l,                      // everything accepted
                1 => 0,                      // nothing (a sequence that did not commit)
                _ => below(&mut rng, l + 1), // part of it
            } as i32)
            .collect();
        let inp = Inp::random(&mut rng, lens.iter().sum());
        let tag = format!("round {round} slots {slots:?} lens {lens:?} keep {keep:?}");

        // reference
        let ra = run_pre_window(&ctx, &inp, &lens, &slots, &a);
        ra.commit(&ctx, &keep, &mut a);

        // deferred: owed rows of the sequences that run again fold into this window; the rest settle on their own
        let batch = BatchBuf::new(&lens, &slots).unwrap();
        let x = DevInp::new(&inp);
        let rows = batch.view().rows;
        let mut rec = Record::alloc_shaped(rows, SHAPE).unwrap();
        let mut o = DevBuf::zeroed(rows * VH * D * 2).unwrap();
        let (mut row0, mut owed_keep) = (vec![0i32; slots.len()], vec![0i32; slots.len()]);
        if let Some(p) = &prev {
            for (i, sl) in slots.iter().enumerate() {
                if let Some(j) = p.slots.iter().position(|s| s == sl) {
                    row0[i] = p.starts[j] as i32;
                    owed_keep[i] = p.keep[j];
                }
            }
            settle(&ctx, p, &mut b, |sl| !slots.contains(&sl));
        }
        owed_rounds += owed_keep.iter().filter(|&&k| k > 0).count();
        pre(&ctx.st, &x.view(), &ctx.weights(), &b.conv, &batch.view(), &mut rec).unwrap();
        match &prev {
            Some(p) => {
                let (r0, ok) = (dev_i32(&row0), dev_i32(&owed_keep));
                window_owed(&ctx.st, &rec, &Owed { rec: &p.rec, row0: &r0, keep: &ok }, &mut b.s, &batch.view(), &mut o).unwrap();
            }
            None => window(&ctx.st, &rec, &b.s, &batch.view(), &mut o).unwrap(),
        }
        commit_conv_shaped(SHAPE, &ctx.st, &x.qkv, &mut b.conv, &batch.view(), &dev_i32(&keep)).unwrap();
        ctx.st.sync().unwrap();
        let o_host = o.to_vec::<u16>().unwrap()[..rows * VH * D].to_vec();
        assert_bits_eq(&format!("{tag}: window output"), &o_host, &ra.host_o());

        let mut starts = vec![0usize; slots.len()];
        for i in 1..slots.len() {
            starts[i] = starts[i - 1] + lens[i - 1];
        }
        let now = Prev { rec, batch, slots: slots.clone(), starts, keep: keep.clone() };
        // state: the deferred pools, with this round's owed rows settled on a copy, equal the reference pools
        let mut c = DevPools {
            s: DevBuf::from_slice(&b.s.to_vec::<f32>().unwrap()).unwrap(),
            conv: DevBuf::from_slice(&b.conv.to_vec::<u16>().unwrap()).unwrap(),
        };
        settle(&ctx, &now, &mut c, |_| true);
        let (hc, ha) = (c.host(), a.host());
        assert_f32_bits_eq(&format!("{tag}: S"), &hc.s, &ha.s);
        assert_bits_eq(&format!("{tag}: conv"), &hc.conv, &ha.conv);
        assert_bits_eq(&format!("{tag}: conv (not settled, eager)"), &b.conv.to_vec::<u16>().unwrap(), &ha.conv);
        prev = Some(now);
    }
    assert!(owed_rounds >= 12, "the test must exercise the fold ({owed_rounds} owed sequences)");
}

// ------------------------------------------------------------------------------------------------ (e) tree windows

use gb_cuda::tree::{random_parents, Plan, Tree, PATH_STRIDE, PLAN_STRIDE};

/// The device arrays of a batch of trees (one per sequence): the batch, the parent of every row (global row, -1 for a root), the plans
/// (local nodes, `PLAN_STRIDE` ints a sequence) and the largest slot count among them.
struct TreeBatch {
    batch: BatchBuf,
    parent: DevBuf,
    plan: DevBuf,
    slots: usize,
    starts: Vec<usize>,
}

fn tree_batch(trees: &[Tree], slots: &[usize]) -> TreeBatch {
    let lens: Vec<usize> = trees.iter().map(|t| t.len()).collect();
    let batch = BatchBuf::new(&lens, slots).unwrap();
    let mut starts = vec![0usize; trees.len()];
    for i in 1..trees.len() {
        starts[i] = starts[i - 1] + lens[i - 1];
    }
    let mut parent = Vec::new();
    let mut plan = vec![0i32; trees.len() * PLAN_STRIDE];
    let mut max_slots = 0;
    for (s, t) in trees.iter().enumerate() {
        parent.extend(t.parents().iter().map(|&p| if p < 0 { -1 } else { (starts[s] as i32) + p }));
        let p = t.plan().unwrap();
        p.validate(t).unwrap();
        max_slots = max_slots.max(p.slots);
        plan[s * PLAN_STRIDE..s * PLAN_STRIDE + p.entries.len()].copy_from_slice(&p.entries);
    }
    TreeBatch { batch, parent: dev_i32(&parent), plan: dev_i32(&plan), slots: max_slots, starts }
}

fn host_rec_of(rec: &Record, n: usize) -> HostRec {
    let take = |b: &DevBuf, len: usize| -> Vec<u16> { b.to_vec::<u16>().unwrap()[..len].to_vec() };
    HostRec {
        q: take(&rec.q, n * KH * D),
        k: take(&rec.k, n * KH * D),
        v: take(&rec.v, n * VH * D),
        g: rec.g_decay.to_vec::<f32>().unwrap()[..n * VH].to_vec(),
        beta: rec.beta.to_vec::<f32>().unwrap()[..n * VH].to_vec(),
    }
}

/// pre_tree + window_tree (nothing owed) of a batch of trees from `pools`; returns the host record and the core outputs.
fn run_tree(ctx: &Ctx, inp: &Inp, tb: &TreeBatch, pools: &mut DevPools) -> (HostRec, Vec<u16>) {
    let x = DevInp::new(inp);
    let rows = tb.batch.view().rows;
    let mut rec = Record::alloc_shaped(rows, SHAPE).unwrap();
    pre_tree(&ctx.st, &x.view(), &ctx.weights(), &pools.conv, &tb.batch.view(), &mut rec, &tb.parent).unwrap();
    let mut o = DevBuf::zeroed(rows * VH * D * 2).unwrap();
    window_tree(&ctx.st, &rec, None, &mut pools.s, &tb.batch.view(), &mut o, &tb.plan, PLAN_STRIDE, tb.slots).unwrap();
    ctx.st.sync().unwrap();
    (host_rec_of(&rec, rows), o.to_vec::<u16>().unwrap()[..rows * VH * D].to_vec())
}

fn row_eq(what: &str, got: &HostRec, gr: usize, want: &HostRec, wr: usize) {
    assert_bits_eq(&format!("{what}: q"), &got.q[gr * KH * D..(gr + 1) * KH * D], &want.q[wr * KH * D..(wr + 1) * KH * D]);
    assert_bits_eq(&format!("{what}: k"), &got.k[gr * KH * D..(gr + 1) * KH * D], &want.k[wr * KH * D..(wr + 1) * KH * D]);
    assert_bits_eq(&format!("{what}: v"), &got.v[gr * VH * D..(gr + 1) * VH * D], &want.v[wr * VH * D..(wr + 1) * VH * D]);
    assert_f32_bits_eq(&format!("{what}: g"), &got.g[gr * VH..(gr + 1) * VH], &want.g[wr * VH..(wr + 1) * VH]);
    assert_f32_bits_eq(&format!("{what}: beta"), &got.beta[gr * VH..(gr + 1) * VH], &want.beta[wr * VH..(wr + 1) * VH]);
}

/// Every node of every tree equals the chain of its own root path, run through the plain window kernels from the same state:
/// the record row (conv history along the path, silu, l2norm, gates) and the core output row. Checked per LEAF path, which covers every
/// node (interior nodes many times). Chains (branch 1) are in the mix, so this is also chain == tree. Returns the number of nodes checked
/// and the tree run's record and outputs.
fn check_trees_against_chains(
    ctx: &Ctx,
    pools: &mut DevPools,
    trees: &[Tree],
    slots: &[usize],
    inp: &Inp,
    tag: &str,
) -> (usize, HostRec, Vec<u16>) {
    let tb = tree_batch(trees, slots);
    let (rec_t, o_t) = run_tree(ctx, inp, &tb, pools);
    let mut checked = 0;
    for (s, t) in trees.iter().enumerate() {
        let leaves: Vec<usize> = (0..t.len()).filter(|&i| !t.parents().contains(&(i as i32))).collect();
        for &leaf in &leaves {
            let path = t.path(leaf);
            let chain_inp =
                Inp::concat(&path.iter().map(|&i| inp.slice(tb.starts[s] + i, tb.starts[s] + i + 1)).collect::<Vec<_>>());
            let chain = run_pre_window(ctx, &chain_inp, &[path.len()], &[slots[s]], pools);
            let (rec_c, o_c) = (chain.host_rec(), chain.host_o());
            for (j, &node) in path.iter().enumerate() {
                let r = tb.starts[s] + node;
                row_eq(&format!("{tag} seq {s} leaf {leaf} node {node} (depth {j})"), &rec_t, r, &rec_c, j);
                assert_bits_eq(
                    &format!("{tag} seq {s} leaf {leaf} node {node}: window output"),
                    &o_t[r * VH * D..(r + 1) * VH * D],
                    &o_c[j * VH * D..(j + 1) * VH * D],
                );
                checked += 1;
            }
        }
    }
    (checked, rec_t, o_t)
}

#[test]
fn tree_rows_equal_their_root_path_chains_bit_for_bit() {
    need_gpu!();
    let mut rng = Rng(2024);
    let ctx = Ctx::new(Weights::random(&mut rng));
    let n_slots = 4;
    let mut pools = DevPools::new(&HostPools::random(&mut rng, n_slots));
    let mut nodes = 0;
    let mut tiers = std::collections::BTreeSet::new();
    for case in 0..24u64 {
        // one to three trees: sizes 1..=64, branching 1 (a chain) to 4, in a shuffled slot order
        let n_seq = 1 + (case % 3) as usize;
        let mut slots: Vec<usize> = (0..n_slots).collect();
        for i in (1..slots.len()).rev() {
            slots.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
        }
        slots.truncate(n_seq);
        let trees: Vec<Tree> = (0..n_seq)
            .map(|s| {
                let n = [1usize, 2, 3, 8, 9, 16, 31, 40, 64][((case + s as u64 * 3) % 9) as usize];
                let b = [1usize, 2, 3, 4][((case / 2 + s as u64) % 4) as usize];
                Tree::new(&random_parents(case * 17 + s as u64, n, b)).unwrap()
            })
            .collect();
        tiers.insert(Plan::tier(trees.iter().map(|t| t.plan().unwrap().slots).max().unwrap()));
        let rows: usize = trees.iter().map(|t| t.len()).sum();
        let inp = Inp::random(&mut rng, rows);
        nodes += check_trees_against_chains(&ctx, &mut pools, &trees, &slots, &inp, &format!("case {case}")).0;
    }
    assert!(nodes >= 400, "the test must cover many nodes ({nodes})");
    assert!(tiers.len() >= 3, "the test must exercise several slot tiers: {tiers:?}");
}

/// The multi-round session: trees of random shape per sequence, an accepted root path per tree, committed through the owed path fold
/// (conv eagerly, S in the next window kernel) against the same session committed eagerly by plain chains. After every round the node
/// outputs equal the per-path chains' and the pools (owed rows settled) equal the reference pools.
#[test]
fn tree_windows_commit_a_path_like_serial_chains_over_rounds() {
    need_gpu!();
    let mut rng = Rng(515);
    let ctx = Ctx::new(Weights::random(&mut rng));
    let n_slots = 4;
    let start = HostPools::random(&mut rng, n_slots);
    let mut a = DevPools::new(&start); // reference: each accepted path as a chain window followed by an eager commit
    let mut b = DevPools::new(&start); // trees: owed path folded into the next window
    struct TreePrev {
        rec: Record,
        batch: BatchBuf,
        slots: Vec<usize>,
        keep: Vec<i32>,
        paths: Vec<Vec<i32>>, // global rows of the accepted path in `rec`
    }
    let mut prev: Option<TreePrev> = None;
    let below = |rng: &mut Rng, n: usize| (rng.next_u64() % n as u64) as usize;
    let path_dev = |paths: &[Vec<i32>]| -> DevBuf {
        let mut flat = vec![0i32; paths.len().max(1) * PATH_STRIDE];
        for (i, p) in paths.iter().enumerate() {
            flat[i * PATH_STRIDE..i * PATH_STRIDE + p.len()].copy_from_slice(p);
        }
        dev_i32(&flat)
    };
    let (mut owed_rounds, mut deep_paths) = (0, 0);
    for round in 0..16u64 {
        let mut slots: Vec<usize> = (0..n_slots).collect();
        for i in (1..slots.len()).rev() {
            slots.swap(i, below(&mut rng, i + 1));
        }
        slots.truncate(1 + below(&mut rng, 3));
        let trees: Vec<Tree> = (0..slots.len())
            .map(|s| {
                let n = 1 + below(&mut rng, 40);
                Tree::new(&random_parents(round * 31 + s as u64, n, 1 + below(&mut rng, 4))).unwrap()
            })
            .collect();
        let rows: usize = trees.iter().map(|t| t.len()).sum();
        let inp = Inp::random(&mut rng, rows);
        let tag = format!("round {round} slots {slots:?} trees {:?}", trees.iter().map(|t| t.len()).collect::<Vec<_>>());

        // the accepted node of each tree (the longest accepted path ends there), its path as global rows of this window
        let tb = tree_batch(&trees, &slots);
        let accepted: Vec<usize> = trees.iter().map(|t| below(&mut rng, t.len())).collect();
        let keep: Vec<i32> = accepted.iter().zip(&trees).map(|(&n, t)| t.path(n).len() as i32).collect();
        let paths: Vec<Vec<i32>> = accepted
            .iter()
            .zip(&trees)
            .enumerate()
            .map(|(s, (&n, t))| t.path(n).iter().map(|&i| (tb.starts[s] + i) as i32).collect())
            .collect();
        deep_paths += keep.iter().filter(|&&k| k >= 4).count();

        // reference: every node against its path chain from the reference pools; then the accepted path as a chain window + eager commit
        let (_, rec_a, o_a) = check_trees_against_chains(&ctx, &mut a, &trees, &slots, &inp, &format!("{tag} [reference pools]"));
        for (s, t) in trees.iter().enumerate() {
            let path = t.path(accepted[s]);
            let chain_inp =
                Inp::concat(&path.iter().map(|&i| inp.slice(tb.starts[s] + i, tb.starts[s] + i + 1)).collect::<Vec<_>>());
            let chain = run_pre_window(&ctx, &chain_inp, &[path.len()], &[slots[s]], &a);
            chain.commit(&ctx, &[path.len() as i32], &mut a);
        }

        // trees: the owed paths of the sequences that run again fold into this window; the others settle on their own
        let x = DevInp::new(&inp);
        let rows_n = tb.batch.view().rows;
        let mut rec = Record::alloc_shaped(rows_n, SHAPE).unwrap();
        let mut o = DevBuf::zeroed(rows_n * VH * D * 2).unwrap();
        let mut owed_keep = vec![0i32; slots.len()];
        let mut owed_paths: Vec<Vec<i32>> = vec![Vec::new(); slots.len()];
        if let Some(p) = &prev {
            for (i, sl) in slots.iter().enumerate() {
                if let Some(j) = p.slots.iter().position(|s| s == sl) {
                    owed_keep[i] = p.keep[j];
                    owed_paths[i] = p.paths[j].clone();
                }
            }
            let sit_out: Vec<i32> =
                p.slots.iter().enumerate().map(|(j, sl)| if slots.contains(sl) { 0 } else { p.keep[j] }).collect();
            let rows_d = path_dev(&p.paths);
            commit_state_path(
                &ctx.st,
                &p.rec,
                &mut b.s,
                &p.batch.view(),
                &dev_i32(&sit_out),
                PathRows { rows: &rows_d, stride: PATH_STRIDE },
            )
            .unwrap();
        }
        owed_rounds += owed_keep.iter().filter(|&&k| k > 0).count();
        pre_tree(&ctx.st, &x.view(), &ctx.weights(), &b.conv, &tb.batch.view(), &mut rec, &tb.parent).unwrap();
        let (ok, orows) = (dev_i32(&owed_keep), path_dev(&owed_paths));
        let owed =
            prev.as_ref().map(|p| OwedPath { rec: &p.rec, keep: &ok, path: PathRows { rows: &orows, stride: PATH_STRIDE } });
        window_tree(&ctx.st, &rec, owed.as_ref(), &mut b.s, &tb.batch.view(), &mut o, &tb.plan, PLAN_STRIDE, tb.slots).unwrap();
        let pd = path_dev(&paths);
        commit_conv_path_shaped(
            SHAPE,
            &ctx.st,
            &x.qkv,
            &mut b.conv,
            &tb.batch.view(),
            &dev_i32(&keep),
            PathRows { rows: &pd, stride: PATH_STRIDE },
        )
        .unwrap();
        ctx.st.sync().unwrap();

        // the folded window's record and outputs are those of the tree run from the (eagerly committed) reference pools
        let rec_b = host_rec_of(&rec, rows_n);
        for r in 0..rows_n {
            row_eq(&format!("{tag}: record row {r} (owed fold vs eager)"), &rec_b, r, &rec_a, r);
        }
        assert_bits_eq(
            &format!("{tag}: window output (owed fold vs eager)"),
            &o.to_vec::<u16>().unwrap()[..rows_n * VH * D],
            &o_a,
        );

        let now = TreePrev { rec, batch: tb.batch, slots: slots.clone(), keep: keep.clone(), paths };
        // state: the folded pools, with this round's owed rows settled on a copy, equal the reference pools
        let mut c = DevPools {
            s: DevBuf::from_slice(&b.s.to_vec::<f32>().unwrap()).unwrap(),
            conv: DevBuf::from_slice(&b.conv.to_vec::<u16>().unwrap()).unwrap(),
        };
        let pd = path_dev(&now.paths);
        commit_state_path(
            &ctx.st,
            &now.rec,
            &mut c.s,
            &now.batch.view(),
            &dev_i32(&now.keep),
            PathRows { rows: &pd, stride: PATH_STRIDE },
        )
        .unwrap();
        ctx.st.sync().unwrap();
        let (hc, ha) = (c.host(), a.host());
        assert_f32_bits_eq(&format!("{tag}: S"), &hc.s, &ha.s);
        assert_bits_eq(&format!("{tag}: conv"), &hc.conv, &ha.conv);
        assert_bits_eq(&format!("{tag}: conv (eager)"), &b.conv.to_vec::<u16>().unwrap(), &ha.conv);
        prev = Some(now);
    }
    assert!(owed_rounds >= 8, "the test must exercise the fold ({owed_rounds} owed sequences)");
    assert!(deep_paths >= 6, "the test must commit paths of several rows ({deep_paths})");
}

// ------------------------------------------------------------------------------------------------ prefill gates

fn run_prefill(ctx: &Ctx, inp: &Inp, lens: &[usize], slots: &[usize], pools: &mut DevPools) -> Vec<u16> {
    let batch = BatchBuf::new(lens, slots).unwrap();
    let x = DevInp::new(inp);
    let mut ws = DevBuf::zeroed(prefill_workspace_bytes_shaped(SHAPE, batch.view().rows)).unwrap();
    let mut o = DevBuf::zeroed(batch.view().rows * VH * D * 2).unwrap();
    prefill_shaped(SHAPE, &ctx.st, &x.view(), &ctx.weights(), &mut pools.s, &mut pools.conv, &batch.view(), &mut ws, &mut o)
        .unwrap();
    ctx.st.sync().unwrap();
    o.to_vec().unwrap()
}

#[test]
fn prefill_is_chunk_invariant_and_batch_invariant() {
    need_gpu!();
    let mut rng = Rng(51);
    let ctx = Ctx::new(Weights::random(&mut rng));
    let total = 300;
    let inp = Inp::random(&mut rng, total);
    let start = HostPools::zeros(3);

    let mut p1 = DevPools::new(&start);
    let o1 = run_prefill(&ctx, &inp, &[total], &[1], &mut p1);
    let h1 = p1.host();

    // the same prompt as 3 consecutive calls (cuts inside a 32-step tile and shorter than the conv history)
    let mut p3 = DevPools::new(&start);
    let cuts = [0usize, 77, 78, 300];
    let mut o3 = Vec::new();
    for w in cuts.windows(2) {
        o3.extend(run_prefill(&ctx, &inp.slice(w[0], w[1]), &[w[1] - w[0]], &[1], &mut p3));
    }
    let h3 = p3.host();
    assert_bits_eq("chunked prefill o", &o3, &o1);
    assert_f32_bits_eq("chunked prefill S", &h3.s, &h1.s);
    assert_bits_eq("chunked prefill conv", &h3.conv, &h1.conv);

    // alone vs inside a batch with other sequences, other slot
    let other = Inp::random(&mut rng, 41);
    let both = Inp::concat(&[other.clone(), inp.clone()]);
    let mut pb = DevPools::new(&start);
    let ob = run_prefill(&ctx, &both, &[41, total], &[0, 2], &mut pb);
    let hb = pb.host();
    assert_bits_eq("batched prefill o", &ob[41 * VH * D..], &o1);
    assert_f32_bits_eq("batched prefill S", hb.slot_s(2), h1.slot_s(1));
    assert_bits_eq("batched prefill conv", hb.slot_conv(2), h1.slot_conv(1));
}

/// The prefill chain kernel that carries two value rows a thread is bit-identical to the one-row-a-thread kernel it replaced: the output,
/// the recurrent state of every slot and the conv history, from random (not zero) starting pools, for sequence lengths that end inside a
/// staging tile and on its edges (1, 2, 15, 16, 17, 31, 32, 33, 100), odd totals, several sequences of different lengths in one launch
/// with distinct slots, and a 4,096-row chunk.
#[test]
fn prefill_two_rows_a_thread_equals_one_row_kernel_bit_for_bit() {
    need_gpu!();
    let mut rng = Rng(0x7277);
    let ctx = Ctx::new(Weights::random(&mut rng));
    let cases: [(&[usize], &[usize]); 9] = [
        (&[1], &[2]),
        (&[2], &[0]),
        (&[15], &[1]),
        (&[16, 17], &[3, 0]),
        (&[31, 32, 33], &[2, 0, 1]),
        (&[100], &[3]),
        (&[41, 70], &[0, 2]),
        (&[1, 16, 17, 300], &[3, 1, 0, 2]),
        (&[4096], &[1]),
    ];
    for (lens, slots) in cases {
        let total: usize = lens.iter().sum();
        let inp = Inp::random(&mut rng, total);
        let start = HostPools::random(&mut rng, 4);
        let batch = BatchBuf::new(lens, slots).unwrap();
        let x = DevInp::new(&inp);
        let mut ws = DevBuf::zeroed(prefill_workspace_bytes_shaped(SHAPE, total)).unwrap();
        let run = |reference: bool, ws: &mut DevBuf| {
            let mut pools = DevPools::new(&start);
            let mut o = DevBuf::zeroed(total * VH * D * 2).unwrap();
            let f = if reference { prefill_ref_shaped } else { prefill_shaped };
            f(SHAPE, &ctx.st, &x.view(), &ctx.weights(), &mut pools.s, &mut pools.conv, &batch.view(), ws, &mut o).unwrap();
            ctx.st.sync().unwrap();
            (o.to_vec::<u16>().unwrap(), pools.host())
        };
        let (want_o, want) = run(true, &mut ws);
        let (got_o, got) = run(false, &mut ws);
        assert_bits_eq(&format!("lens {lens:?} o"), &got_o, &want_o);
        assert_f32_bits_eq(&format!("lens {lens:?} S"), &got.s, &want.s);
        assert_bits_eq(&format!("lens {lens:?} conv"), &got.conv, &want.conv);
        assert!(got_o.iter().any(|&x| x != 0), "lens {lens:?}: the outputs are all zero");
    }
}

/// The prefill with the gated RMS norm fused into the chain kernel (128 value rows a block, the norm of each tile's steps at the tile's end)
/// leaves exactly the bits of the prefill followed by the gated-norm kernel: the normed output, the recurrent state of every slot and the
/// conv history, from random starting pools, for sequence lengths that end inside a 16-step tile and on its edges (1, 2, 15, 16, 17, 31,
/// 32, 33, 100), odd totals, several sequences of different lengths in one launch with distinct slots, and a 4,096-row chunk. 48 value
/// heads only (the norm's width is the model's).
#[test]
fn prefill_gated_norm_fused_equals_chain_then_norm_bit_for_bit() {
    need_gpu!();
    if SHAPE != GdnShape::Hv48 {
        return;
    }
    let mut rng = Rng(0x6a7e);
    let ctx = Ctx::new(Weights::random(&mut rng));
    let nw = DevBuf::from_slice(&rng.vec(D, 0.4, 1.8)).unwrap();
    let eps = 1e-6f32;
    let cases: [(&[usize], &[usize]); 11] = [
        (&[1], &[2]),
        (&[2], &[0]),
        (&[15], &[1]),
        (&[16, 17], &[3, 0]),
        (&[31, 32, 33], &[2, 0, 1]),
        (&[100], &[3]),
        (&[41, 70], &[0, 2]),
        (&[1, 16, 17, 300], &[3, 1, 0, 2]),
        (&[4012], &[1]),
        (&[4096], &[1]),
        (&[8192], &[1]),
    ];
    for (lens, slots) in cases {
        let total: usize = lens.iter().sum();
        let inp = Inp::random(&mut rng, total);
        let zh: Vec<u16> = rng.vec(total * VH * D, -6.0, 6.0).iter().map(|&x| f32_to_bf16(x)).collect();
        let z = DevBuf::from_slice(&zh).unwrap();
        let start = HostPools::random(&mut rng, 4);
        let batch = BatchBuf::new(lens, slots).unwrap();
        let x = DevInp::new(&inp);
        let mut ws = DevBuf::zeroed(prefill_workspace_bytes_shaped(SHAPE, total)).unwrap();
        let mut run = |fused: bool| {
            let mut pools = DevPools::new(&start);
            let mut o = DevBuf::zeroed(total * VH * D * 2).unwrap();
            if fused {
                prefill_gated(&ctx.st, &x.view(), &ctx.weights(), &mut pools.s, &mut pools.conv, &batch.view(), &mut ws, &z, &nw, eps, &mut o)
                    .unwrap();
            } else {
                prefill_shaped(SHAPE, &ctx.st, &x.view(), &ctx.weights(), &mut pools.s, &mut pools.conv, &batch.view(), &mut ws, &mut o)
                    .unwrap();
                gb_cuda::fn_glue::gated_rmsnorm_sigmoid(None, &z, &nw, &mut o, total, VH, D, eps, &ctx.st).unwrap();
            }
            ctx.st.sync().unwrap();
            (o.to_vec::<u16>().unwrap(), pools.host())
        };
        let (want_o, want) = run(false);
        let (got_o, got) = run(true);
        assert_bits_eq(&format!("lens {lens:?} o"), &got_o, &want_o);
        assert_f32_bits_eq(&format!("lens {lens:?} S"), &got.s, &want.s);
        assert_bits_eq(&format!("lens {lens:?} conv"), &got.conv, &want.conv);
        assert!(got_o.iter().any(|&x| x != 0), "lens {lens:?}: the outputs are all zero");
    }
}

/// The chain kernel and the gated norm after it against the fused kernel, one layer of a 4,096-row chunk (and of 4,012 rows, the size of a
/// steady chunk of a 32K prompt).
///     cargo test --release -p gb-cuda --test gdn -- --ignored --nocapture bench_prefill_gated_norm
#[test]
#[ignore]
fn bench_prefill_gated_norm() {
    need_gpu!();
    if SHAPE != GdnShape::Hv48 {
        return;
    }
    let mut rng = Rng(63);
    let ctx = Ctx::new(Weights::random(&mut rng));
    let nw = DevBuf::from_slice(&rng.vec(D, 0.4, 1.8)).unwrap();
    for rows in [4012usize, 4096] {
        let inp = Inp::random(&mut rng, rows);
        let x = DevInp::new(&inp);
        let zh: Vec<u16> = rng.vec(rows * VH * D, -6.0, 6.0).iter().map(|&x| f32_to_bf16(x)).collect();
        let z = DevBuf::from_slice(&zh).unwrap();
        let mut pools = DevPools { s: DevBuf::zeroed(S_SLOT * 4).unwrap(), conv: DevBuf::zeroed(CONV_SLOT * 2).unwrap() };
        let batch = BatchBuf::new(&[rows], &[0]).unwrap();
        let mut ws = DevBuf::zeroed(prefill_workspace_bytes_shaped(SHAPE, rows)).unwrap();
        let mut o = DevBuf::zeroed(rows * VH * D * 2).unwrap();
        let t_old = timed(&ctx.st, 10, |_| {
            prefill_shaped(SHAPE, &ctx.st, &x.view(), &ctx.weights(), &mut pools.s, &mut pools.conv, &batch.view(), &mut ws, &mut o).unwrap();
            gb_cuda::fn_glue::gated_rmsnorm_sigmoid(None, &z, &nw, &mut o, rows, VH, D, 1e-6, &ctx.st).unwrap();
        });
        let t_new = timed(&ctx.st, 10, |_| {
            prefill_gated(&ctx.st, &x.view(), &ctx.weights(), &mut pools.s, &mut pools.conv, &batch.view(), &mut ws, &z, &nw, 1e-6, &mut o).unwrap()
        });
        eprintln!("BENCH {rows} rows, pre + chain + norm: previous {:.3} ms, fused {:.3} ms", t_old * 1e3, t_new * 1e3);
    }
}

#[test]
fn prefill_agrees_with_decode_path_approximately() {
    need_gpu!();
    let mut rng = Rng(52);
    let ctx = Ctx::new(Weights::random(&mut rng));
    let rows = 64;
    let inp = Inp::random(&mut rng, rows);
    let start = HostPools::random(&mut rng, 2);
    let mut pp = DevPools::new(&start);
    let op = run_prefill(&ctx, &inp, &[rows], &[1], &mut pp);
    let mut pd = DevPools::new(&start);
    let run = run_pre_window(&ctx, &inp, &[rows], &[1], &pd);
    let od = run.host_o();
    run.commit(&ctx, &[rows as i32], &mut pd);
    let (cos, rr) = rel(&bf_to_f32(&op), &bf_to_f32(&od));
    assert!(cos >= 0.99999 && rr <= 2e-3, "o: cos {cos} relRMS {rr}");
    let (hp, hd) = (pp.host(), pd.host());
    let (cos, rr) = rel(hp.slot_s(1), hd.slot_s(1));
    assert!(cos >= 0.99999 && rr <= 2e-3, "S: cos {cos} relRMS {rr}");
    assert_bits_eq("conv", hp.slot_conv(1), hd.slot_conv(1));
}

// ------------------------------------------------------------------------------------------------ (d) ExLlamaV3 dumps

fn pack_dir() -> std::path::PathBuf {
    std::env::var("REXL3_QWEN_DIR").map(Into::into).unwrap_or_default()
}

fn layer0_weights() -> Option<Weights> {
    let dir = pack_dir();
    if !dir.join("model.safetensors.index.json").exists() {
        return None;
    }
    let ck = gb_exl3::checkpoint::scan(&dir, false).ok()?;
    let get = |n: &str| -> Vec<u8> {
        let e = ck.plain.get(&format!("model.language_model.layers.0.linear_attn.{n}")).unwrap_or_else(|| panic!("{n} missing"));
        ck.read_tensor(e).unwrap()
    };
    let u16s = |b: Vec<u8>| -> Vec<u16> { b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() };
    let conv_w = u16s(get("conv1d.weight"));
    assert_eq!(conv_w.len(), C * 4);
    let f = |b: Vec<u8>| -> Vec<f32> { u16s(b).iter().map(|&x| bf16_to_f32(x)).collect() };
    Some(Weights { conv_w, a_log: f(get("A_log")), dt_bias: f(get("dt_bias")) })
}

#[test]
fn layer0_core_matches_exllamav3_dumps() {
    need_gpu!();
    if SHAPE != GdnShape::Hv48 {
        eprintln!("not run: the layer-0 dumps are Qwen3.8-27B's (48 value heads)");
        return;
    }
    if !common::have_refs() {
        eprintln!("no reference dumps: skipping");
        return;
    }
    let Some(w) = layer0_weights() else {
        eprintln!("no model pack: skipping");
        return;
    };
    let ctx = Ctx::new(w);
    for (name, via_prefill) in [("short", false), ("short", true), ("long", false), ("long", true)] {
        let f = |n: &str| ref_npy(&format!("{name}/sub/L00.{n}.npy")).to_f32();
        let (qkv, a, b, want) = (f("in_proj_qkv.out"), f("in_proj_a.out"), f("in_proj_b.out"), f("gdn_norm.in"));
        let t = a.len() / VH;
        let bf = |v: &[f32]| -> Vec<u16> { v.iter().map(|&x| f32_to_bf16(x)).collect() };
        let inp = Inp { qkv: bf(&qkv), a: bf(&a), b: bf(&b) };
        let start = HostPools::zeros(1);
        let mut pools = DevPools::new(&start);
        let o = if via_prefill {
            run_prefill(&ctx, &inp, &[t], &[0], &mut pools)
        } else {
            let run = run_pre_window(&ctx, &inp, &[t], &[0], &pools);
            run.host_o()
        };
        let (cos, rr) = rel(&bf_to_f32(&o), &want);
        eprintln!("L00 {name} T={t} prefill={via_prefill}: cos {cos:.7} relRMS {rr:.3e}");
        assert!(cos >= 0.9999 && rr <= 1e-2, "{name} prefill={via_prefill}: cos {cos} relRMS {rr}");
    }
}

// ------------------------------------------------------------------------------------------------ golden hashes

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

fn h_u16(v: &[u16]) -> u64 {
    fnv64(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
}

fn h_f32(v: &[f32]) -> u64 {
    fnv64(&v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect::<Vec<u8>>())
}

/// Golden hashes of the GDN kernels on seeded synthetic inputs (no pack needed): the window's record and output and the committed
/// pools for several window widths and batch shapes, the owed-commit fold, a prefill (whole and in parts), and a tree window with an
/// accepted-path commit. A change that must keep every bit diffs this file against one written by the old build.
///
///     REXL3_GOLDEN_OUT=testdata/gpu/golden_gdn.txt cargo test --release -p gb-cuda --test gdn golden_gdn_hashes -- --ignored
#[test]
#[ignore]
fn golden_gdn_hashes() {
    need_gpu!();
    let path = std::env::var("REXL3_GOLDEN_OUT").expect("REXL3_GOLDEN_OUT");
    let mut rng = Rng(0x0067_646e);
    let ctx = Ctx::new(Weights::random(&mut rng));
    let mut out = String::new();
    // window + commit: widths 1, 4, 8, 16 and mixed batches, from random pools
    let cases: [(&[usize], &[usize], &[i32]); 6] = [
        (&[1], &[2], &[1]),
        (&[4], &[0], &[3]),
        (&[8], &[1], &[8]),
        (&[16], &[3], &[5]),
        (&[8, 3, 1], &[3, 1, 0], &[2, 3, 0]),
        (&[5, 16, 2, 7], &[0, 2, 1, 3], &[5, 1, 2, 7]),
    ];
    for (ci, (lens, slots, keep)) in cases.iter().enumerate() {
        let inp = Inp::random(&mut rng, lens.iter().sum());
        let start = HostPools::random(&mut rng, 4);
        let mut pools = DevPools::new(&start);
        let run = run_pre_window(&ctx, &inp, lens, slots, &pools);
        let rec = run.host_rec();
        out += &format!(
            "window case {ci} lens {lens:?} q {:016x} k {:016x} v {:016x} g {:016x} beta {:016x} o {:016x}\n",
            h_u16(&rec.q),
            h_u16(&rec.k),
            h_u16(&rec.v),
            h_f32(&rec.g),
            h_f32(&rec.beta),
            h_u16(&run.host_o())
        );
        run.commit(&ctx, keep, &mut pools);
        let h = pools.host();
        out += &format!("commit case {ci} keep {keep:?} s {:016x} conv {:016x}\n", h_f32(&h.s), h_u16(&h.conv));
    }
    // the owed-commit fold: a window, then a second window that replays the first one's accepted rows first
    {
        let start = HostPools::random(&mut rng, 3);
        let mut pools = DevPools::new(&start);
        let (lens1, slots) = ([6usize, 9], [2usize, 0]);
        let inp1 = Inp::random(&mut rng, 15);
        let batch1 = BatchBuf::new(&lens1, &slots).unwrap();
        let x1 = DevInp::new(&inp1);
        let mut rec1 = Record::alloc_shaped(15, SHAPE).unwrap();
        let mut o1 = DevBuf::zeroed(15 * VH * D * 2).unwrap();
        pre(&ctx.st, &x1.view(), &ctx.weights(), &pools.conv, &batch1.view(), &mut rec1).unwrap();
        window(&ctx.st, &rec1, &pools.s, &batch1.view(), &mut o1).unwrap();
        let keep1 = [4i32, 9];
        commit_conv_shaped(SHAPE, &ctx.st, &x1.qkv, &mut pools.conv, &batch1.view(), &dev_i32(&keep1)).unwrap();
        let (lens2, inp2) = ([3usize, 5], Inp::random(&mut rng, 8));
        let batch2 = BatchBuf::new(&lens2, &slots).unwrap();
        let x2 = DevInp::new(&inp2);
        let mut rec2 = Record::alloc_shaped(8, SHAPE).unwrap();
        let mut o2 = DevBuf::zeroed(8 * VH * D * 2).unwrap();
        pre(&ctx.st, &x2.view(), &ctx.weights(), &pools.conv, &batch2.view(), &mut rec2).unwrap();
        let (row0, ok) = (dev_i32(&[0, 6]), dev_i32(&keep1));
        window_owed(&ctx.st, &rec2, &Owed { rec: &rec1, row0: &row0, keep: &ok }, &mut pools.s, &batch2.view(), &mut o2).unwrap();
        ctx.st.sync().unwrap();
        let h = pools.host();
        out += &format!(
            "owed fold o {:016x} s {:016x} conv {:016x}\n",
            h_u16(&o2.to_vec::<u16>().unwrap()[..8 * VH * D]),
            h_f32(&h.s),
            h_u16(&h.conv)
        );
    }
    // prefill: whole, in parts, and two sequences
    {
        let total = 300;
        let inp = Inp::random(&mut rng, total);
        let start = HostPools::zeros(3);
        let mut p1 = DevPools::new(&start);
        let o1 = run_prefill(&ctx, &inp, &[total], &[1], &mut p1);
        let h1 = p1.host();
        out += &format!("prefill whole o {:016x} s {:016x} conv {:016x}\n", h_u16(&o1), h_f32(&h1.s), h_u16(&h1.conv));
        let mut p2 = DevPools::new(&start);
        let mut o2 = Vec::new();
        for w in [0usize, 77, 78, 300].windows(2) {
            o2.extend(run_prefill(&ctx, &inp.slice(w[0], w[1]), &[w[1] - w[0]], &[1], &mut p2));
        }
        out += &format!("prefill parts o {:016x}\n", h_u16(&o2));
        let two = Inp::concat(&[Inp::random(&mut rng, 41), Inp::random(&mut rng, 70)]);
        let mut p3 = DevPools::new(&start);
        let o3 = run_prefill(&ctx, &two, &[41, 70], &[0, 2], &mut p3);
        let h3 = p3.host();
        out += &format!("prefill two o {:016x} s {:016x} conv {:016x}\n", h_u16(&o3), h_f32(&h3.s), h_u16(&h3.conv));
    }
    // tree windows with an accepted-path commit
    {
        let trees = vec![Tree::new(&random_parents(9, 9, 3)).unwrap(), Tree::new(&random_parents(17, 17, 4)).unwrap()];
        let slots = [1usize, 3];
        let rows: usize = trees.iter().map(|t| t.len()).sum();
        let inp = Inp::random(&mut rng, rows);
        let mut pools = DevPools::new(&HostPools::random(&mut rng, 4));
        let tb = tree_batch(&trees, &slots);
        let (rec, o) = run_tree(&ctx, &inp, &tb, &mut pools);
        out += &format!(
            "tree 9+17 q {:016x} k {:016x} v {:016x} g {:016x} beta {:016x} o {:016x}\n",
            h_u16(&rec.q),
            h_u16(&rec.k),
            h_u16(&rec.v),
            h_f32(&rec.g),
            h_f32(&rec.beta),
            h_u16(&o)
        );
        // commit the path to each tree's last node (state and conv)
        let x = DevInp::new(&inp);
        let mut rec_d = Record::alloc_shaped(rows, SHAPE).unwrap();
        pre_tree(&ctx.st, &x.view(), &ctx.weights(), &pools.conv, &tb.batch.view(), &mut rec_d, &tb.parent).unwrap();
        let paths: Vec<Vec<i32>> = trees
            .iter()
            .enumerate()
            .map(|(s, t)| t.path(t.len() - 1).iter().map(|&i| (tb.starts[s] + i) as i32).collect())
            .collect();
        let keep: Vec<i32> = paths.iter().map(|p| p.len() as i32).collect();
        let mut flat = vec![0i32; paths.len() * PATH_STRIDE];
        for (i, p) in paths.iter().enumerate() {
            flat[i * PATH_STRIDE..i * PATH_STRIDE + p.len()].copy_from_slice(p);
        }
        let (kd, pd) = (dev_i32(&keep), dev_i32(&flat));
        let path = PathRows { rows: &pd, stride: PATH_STRIDE };
        commit_state_path(&ctx.st, &rec_d, &mut pools.s, &tb.batch.view(), &kd, path).unwrap();
        commit_conv_path_shaped(SHAPE, &ctx.st, &x.qkv, &mut pools.conv, &tb.batch.view(), &kd, path).unwrap();
        ctx.st.sync().unwrap();
        let h = pools.host();
        out += &format!("tree commit path s {:016x} conv {:016x}\n", h_f32(&h.s), h_u16(&h.conv));
    }
    std::fs::write(path, out).unwrap();
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

#[test]
#[ignore]
fn bench_window_and_commit() {
    need_gpu!();
    let mut rng = Rng(61);
    let ctx = Ctx::new(Weights::random(&mut rng));
    let n_slots = 64; // 192 MiB of S: far beyond L2, so every launch reads from DRAM
    let mut pools =
        DevPools { s: DevBuf::zeroed(n_slots * S_SLOT * 4).unwrap(), conv: DevBuf::zeroed(n_slots * CONV_SLOT * 2).unwrap() };
    let mib = (S_SLOT * 4) as f64 / (1024.0 * 1024.0);
    for &n_seq in &[1usize, 4, 6] {
        for &rows in &[1usize, 8] {
            let inp = Inp::random(&mut rng, rows * n_seq);
            let x = DevInp::new(&inp);
            let n_batches = n_slots / n_seq;
            let batches: Vec<BatchBuf> = (0..n_batches)
                .map(|b| BatchBuf::new(&vec![rows; n_seq], &(0..n_seq).map(|j| b * n_seq + j).collect::<Vec<_>>()).unwrap())
                .collect();
            let mut rec = Record::alloc_shaped(rows * n_seq, SHAPE).unwrap();
            let keep = DevBuf::from_slice(&vec![rows as i32; n_seq]).unwrap();
            let mut o = DevBuf::zeroed(rows * n_seq * VH * D * 2).unwrap();
            pre(&ctx.st, &x.view(), &ctx.weights(), &pools.conv, &batches[0].view(), &mut rec).unwrap();
            let tw = timed(&ctx.st, 400, |i| window(&ctx.st, &rec, &pools.s, &batches[i % n_batches].view(), &mut o).unwrap());
            let tc = timed(&ctx.st, 400, |i| {
                commit(&ctx.st, &rec, &x.qkv, &mut pools.s, &mut pools.conv, &batches[i % n_batches].view(), &keep).unwrap()
            });
            let tp = timed(&ctx.st, 400, |i| {
                pre(&ctx.st, &x.view(), &ctx.weights(), &pools.conv, &batches[i % n_batches].view(), &mut rec).unwrap()
            });
            // the deferred form: this window folds in the commit the previous one owes (keep 3 rows, or 1 for a 1-row window)
            let keep3 = DevBuf::from_slice(&vec![rows.min(3) as i32; n_seq]).unwrap();
            let row0: Vec<i32> = (0..n_seq).map(|i| (i * rows) as i32).collect();
            let row0 = DevBuf::from_slice(&row0).unwrap();
            let owed = Owed { rec: &rec, row0: &row0, keep: &keep3 };
            let tf = timed(&ctx.st, 400, |i| {
                window_owed(&ctx.st, &rec, &owed, &mut pools.s, &batches[i % n_batches].view(), &mut o).unwrap()
            });
            let sm = mib * n_seq as f64 / 1024.0;
            eprintln!(
                "rows {rows} x seqs {n_seq}: window {:.1} us ({:.0} GB/s of S read, {:.0} GB/s per-row-sequence logical) | commit {:.1} us ({:.0} GB/s read+write) | window+commit {:.1} us | window with the commit folded in (keep {}) {:.1} us ({:.0} GB/s read+write) | pre {:.1} us",
                tw * 1e6,
                sm * 1.0737 / tw,
                sm * 1.0737 * rows as f64 / tw,
                tc * 1e6,
                2.0 * sm * 1.0737 / tc,
                (tw + tc) * 1e6,
                rows.min(3),
                tf * 1e6,
                2.0 * sm * 1.0737 / tf,
                tp * 1e6
            );
        }
    }
}

#[test]
#[ignore]
fn bench_prefill_4096() {
    need_gpu!();
    let mut rng = Rng(62);
    let ctx = Ctx::new(Weights::random(&mut rng));
    let rows = 4096;
    let inp = Inp::random(&mut rng, rows);
    let x = DevInp::new(&inp);
    let mut pools = DevPools { s: DevBuf::zeroed(S_SLOT * 4).unwrap(), conv: DevBuf::zeroed(CONV_SLOT * 2).unwrap() };
    let batch = BatchBuf::new(&[rows], &[0]).unwrap();
    let mut ws = DevBuf::zeroed(prefill_workspace_bytes_shaped(SHAPE, rows)).unwrap();
    let mut o = DevBuf::zeroed(rows * VH * D * 2).unwrap();
    let t = timed(&ctx.st, 10, |_| {
        prefill_shaped(SHAPE, &ctx.st, &x.view(), &ctx.weights(), &mut pools.s, &mut pools.conv, &batch.view(), &mut ws, &mut o)
            .unwrap()
    });
    eprintln!(
        "prefill {rows} rows, 1 layer: {:.2} ms; 48 layers: {:.1} ms = {:.0} tokens/s",
        t * 1e3,
        t * 48e3,
        rows as f64 / (t * 48.0)
    );
    // 8 sequences of 512 rows
    let batch8 = BatchBuf::new(&[512; 8], &[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
    let mut pools8 = DevPools { s: DevBuf::zeroed(8 * S_SLOT * 4).unwrap(), conv: DevBuf::zeroed(8 * CONV_SLOT * 2).unwrap() };
    let t = timed(&ctx.st, 10, |_| {
        prefill_shaped(
            SHAPE,
            &ctx.st,
            &x.view(),
            &ctx.weights(),
            &mut pools8.s,
            &mut pools8.conv,
            &batch8.view(),
            &mut ws,
            &mut o,
        )
        .unwrap()
    });
    eprintln!(
        "prefill 8 x 512 rows, 1 layer: {:.2} ms; 48 layers: {:.1} ms = {:.0} tokens/s",
        t * 1e3,
        t * 48e3,
        rows as f64 / (t * 48.0)
    );
}
