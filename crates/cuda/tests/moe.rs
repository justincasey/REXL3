//! The MoE kernels (`csrc/moe_router.cu`, `csrc/moe_experts.cu`, `src/moe.rs`): router bits against the host mirror, row
//! invariance for 1..=128 rows, grouped experts against a float64 reference on synthetic layers of every width and on real layers of
//! the GLM-5.3-Flash pack, goldens, and the performance table. GPU tests skip when there is no device; the real-pack tests also need
//! `REXL3_GLM_DIR` (default `~/models/r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121`).
//!
//!     cargo test --release -p gb-cuda --test moe -- --test-threads=1            # correctness
//!     REXL3_GOLDEN_OUT=golden_moe.txt cargo test --release -p gb-cuda --test moe golden_moe_hashes -- --ignored   # the name scripts/gpu_gate.sh sets (REXL3_MOE_GOLDEN_OUT also works)
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
mod common;

use common::*;
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::moe::*;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::checkpoint::{scan, Checkpoint, Group, LoadedGroup};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_exl3::{codebook, fnv1a, unpack, Bits, Codebook};
use std::collections::HashMap;

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

fn up<T: Copy>(v: &[T]) -> DevBuf {
    DevBuf::from_slice(v).unwrap()
}

fn normal(r: &mut Rng) -> f64 {
    let (a, b) = ((r.next_u64() >> 11) as f64 / (1u64 << 53) as f64, (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64);
    (-2.0 * a.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
}

fn below(r: &mut Rng, n: usize) -> usize {
    (r.next_u64() % n as u64) as usize
}

fn shuffle<T>(r: &mut Rng, v: &mut [T]) {
    for i in (1..v.len()).rev() {
        v.swap(i, below(r, i + 1));
    }
}

fn h(x: f64) -> u16 {
    f64_to_f16(x)
}

fn hf(b: u16) -> f64 {
    f16_to_f64(b)
}

fn rand_f16(r: &mut Rng, n: usize, sigma: f64) -> Vec<u16> {
    (0..n).map(|_| h(sigma * normal(r))).collect()
}

fn hash_u16(v: &[u16]) -> u64 {
    fnv1a(v.iter().flat_map(|x| x.to_le_bytes()))
}
fn hash_f32(v: &[f32]) -> u64 {
    fnv1a(v.iter().flat_map(|x| x.to_bits().to_le_bytes()))
}
fn hash_i32(v: &[i32]) -> u64 {
    fnv1a(v.iter().flat_map(|x| x.to_le_bytes()))
}

fn sync_dl<T: Copy + Default>(b: &DevBuf, st: &Stream) -> Vec<T> {
    st.sync().unwrap();
    b.to_vec().unwrap()
}

// ====================================================================================================================
// router

struct RouterRun {
    logits: Vec<u16>,
    pick: Vec<i32>,
    wts: Vec<u16>,
}

fn route_rows(router: &MoeRouter, tab: &SigmoidTable, x: &DevBuf, dt: Dtype, rows: usize, st: &Stream) -> RouterRun {
    let mut logits = DevBuf::zeroed(rows * router.experts * 2).unwrap();
    let mut pick = DevBuf::zeroed(rows * router.topk * 4).unwrap();
    let mut wts = DevBuf::zeroed(rows * router.topk * 2).unwrap();
    router.logits(x, dt, rows, &mut logits, st).unwrap();
    router.select(tab, &logits, rows, &mut pick, &mut wts, st).unwrap();
    RouterRun { logits: sync_dl(&logits, st), pick: sync_dl(&pick, st), wts: sync_dl(&wts, st) }
}

/// Synthetic router of GLM shape: fp16 gate [288, 4096], a centred fp16 bias.
fn synth_router(seed: u64) -> (MoeRouter, Vec<u16>, Vec<f32>) {
    let mut r = Rng(seed);
    let (e, k) = (288, 4096);
    let gate = rand_f16(&mut r, e * k, 0.03);
    let esb: Vec<f32> = (0..e).map(|_| (34.0 + 0.3 * normal(&mut r)) as f32).collect();
    let bias = centred_bias_host(&esb);
    (MoeRouter::new(&gate, &bias, k, 8, 2.5).unwrap(), gate, bias)
}

#[test]
fn router_matches_the_host_mirror_bit_for_bit() {
    gpu!();
    let st = Stream::new().unwrap();
    let tab = SigmoidTable::new().unwrap();
    let sig = sigmoid_table_host();
    let (router, gate, bias) = synth_router(1);
    let rows = 128;
    let mut r = Rng(2);
    let xh = rand_f16(&mut r, rows * 4096, 1.0);
    let x = up(&xh);
    let run = route_rows(&router, &tab, &x, Dtype::F16, rows, &st);
    let xf: Vec<f32> = xh.iter().map(|&b| hf(b) as f32).collect();
    let want_logits = router_logits_host(&xf, &gate, rows, 4096, 288);
    assert!(run.logits == want_logits, "logits differ from the host mirror");
    let (wp, ww) = router_select_host(&want_logits, &bias, &sig, rows, 288, 8, 2.5);
    assert!(run.pick == wp, "picks differ from the host mirror");
    assert!(run.wts == ww, "weights differ from the host mirror");
    // bf16 and fp32 inputs through the same kernel: the host mirror of the decoded values
    for dt in [Dtype::BF16, Dtype::F32] {
        let (vals, bytes): (Vec<f32>, Vec<u8>) = match dt {
            Dtype::BF16 => {
                let b: Vec<u16> = xh
                    .iter()
                    .map(|&v| {
                        (((hf(v) as f32).to_bits().wrapping_add(0x7fff + (((hf(v) as f32).to_bits() >> 16) & 1))) >> 16) as u16
                    })
                    .collect();
                (b.iter().map(|&v| f32::from_bits((v as u32) << 16)).collect(), b.iter().flat_map(|v| v.to_le_bytes()).collect())
            }
            _ => {
                let v: Vec<f32> = xh.iter().map(|&v| hf(v) as f32 * 1.000_123).collect();
                (v.clone(), v.iter().flat_map(|x| x.to_le_bytes()).collect())
            }
        };
        let xd = up(&bytes);
        let run = route_rows(&router, &tab, &xd, dt, rows, &st);
        let want = router_logits_host(&vals, &gate, rows, 4096, 288);
        assert!(run.logits == want, "{dt:?} logits differ from the host mirror");
        let (wp, ww) = router_select_host(&want, &bias, &sig, rows, 288, 8, 2.5);
        assert!(run.pick == wp && run.wts == ww, "{dt:?} picks or weights differ");
    }
}

/// Row r of any window of 1..=128 rows has the same logits, picks and weights as row r alone.
#[test]
fn router_is_row_invariant_for_every_window_size() {
    gpu!();
    let st = Stream::new().unwrap();
    let tab = SigmoidTable::new().unwrap();
    let (router, _, _) = synth_router(3);
    let pool = 128;
    let mut r = Rng(4);
    let xh = rand_f16(&mut r, pool * 4096, 1.0);
    let solo: Vec<RouterRun> = (0..pool)
        .map(|i| {
            let x = up(&xh[i * 4096..(i + 1) * 4096]);
            route_rows(&router, &tab, &x, Dtype::F16, 1, &st)
        })
        .collect();
    for m in 1..=pool {
        let mut windows: Vec<Vec<usize>> = vec![(0..m).collect(), (0..m).rev().collect()];
        let mut idx: Vec<usize> = (0..pool).collect();
        shuffle(&mut r, &mut idx);
        windows.push(idx[..m].to_vec());
        windows.push(vec![idx[0]; m]); // m identical copies
        for w in windows {
            let mut buf = Vec::with_capacity(m * 4096);
            for &i in &w {
                buf.extend_from_slice(&xh[i * 4096..(i + 1) * 4096]);
            }
            let run = route_rows(&router, &tab, &up(&buf), Dtype::F16, m, &st);
            for (pos, &i) in w.iter().enumerate() {
                assert!(run.logits[pos * 288..(pos + 1) * 288] == solo[i].logits[..], "logits: row {i} at {pos} of {m}");
                assert!(run.pick[pos * 8..(pos + 1) * 8] == solo[i].pick[..], "picks: row {i} at {pos} of {m}");
                assert!(run.wts[pos * 8..(pos + 1) * 8] == solo[i].wts[..], "weights: row {i} at {pos} of {m}");
            }
        }
    }
}

fn select_case(logits_row: &[u16], bias: &[f32], tab: &SigmoidTable, st: &Stream) -> (Vec<i32>, Vec<u16>) {
    let e = bias.len();
    let router = MoeRouter::new(&vec![0u16; e * 2], bias, 2, 8, 2.5).unwrap();
    let lg = up(logits_row);
    let mut pick = DevBuf::zeroed(8 * 4).unwrap();
    let mut wts = DevBuf::zeroed(8 * 2).unwrap();
    router.select(tab, &lg, 1, &mut pick, &mut wts, st).unwrap();
    (sync_dl(&pick, st), sync_dl(&wts, st))
}

#[test]
fn router_ties_go_to_the_lower_expert_id() {
    gpu!();
    let st = Stream::new().unwrap();
    let tab = SigmoidTable::new().unwrap();
    let sig = sigmoid_table_host();
    let e = 288;
    let base = h(0.25);
    let check = |logits: Vec<u16>, bias: Vec<f32>, expect_prefix: &[i32], what: &str| {
        let (p, w) = select_case(&logits, &bias, &tab, &st);
        let (hp, hw) = router_select_host(&logits, &bias, &sig, 1, e, 8, 2.5);
        assert_eq!(p, hp, "{what}: GPU picks differ from the host mirror");
        assert_eq!(w, hw, "{what}: GPU weights differ from the host mirror");
        assert_eq!(&p[..expect_prefix.len()], expect_prefix, "{what}: {p:?}");
    };
    // a total tie: ids 0..8 in order
    check(vec![base; e], vec![0.0; e], &[0, 1, 2, 3, 4, 5, 6, 7], "total tie");
    // experts 1 and 32 tie for the best key (the oracle's lowest-LANE rule would put 32 first; the contract here is the lowest id)
    let mut bias = vec![0.0f32; e];
    bias[1] = 1.0;
    bias[32] = 1.0;
    check(vec![base; e], bias, &[1, 32, 0, 2, 3, 4, 5, 6], "1 vs 32");
    // 7 and 39 tie at ranks 8 and 9 with one slot left: 7 is picked, 39 is not. Experts 0..6 have a higher bias.
    let mut bias = vec![0.0f32; e];
    for b in bias.iter_mut().take(7) {
        *b = 1.0;
    }
    bias[7] = 0.5;
    bias[39] = 0.5;
    let (p, _) = select_case(&vec![base; e], &bias, &tab, &st);
    assert_eq!(p, vec![0, 1, 2, 3, 4, 5, 6, 7], "boundary tie");
    // the same tie with ids 200 and 8 (different lanes and rounds): 8 first
    let mut bias = vec![0.0f32; e];
    bias[200] = 2.0;
    bias[8] = 2.0;
    check(vec![base; e], bias, &[8, 200], "8 vs 200");
    // an exact tie that is not equal parts: act differs, bias compensates so that act + bias is the same fp32 value
    let lo = h(-1.0);
    let hi = h(1.0);
    let (a_lo, a_hi) = (sig[lo as usize], sig[hi as usize]);
    let mut logits = vec![h(-4.0); e];
    let mut bias = vec![0.0f32; e];
    logits[100] = lo;
    logits[50] = hi;
    bias[100] = 1.0;
    bias[50] = (1.0 + a_lo) - a_hi; // key[100] = a_lo + 1.0 exactly equals key[50] when the fp32 sums agree
    let k100 = a_lo + bias[100];
    let k50 = a_hi + bias[50];
    if k100 == k50 {
        check(logits, bias, &[50, 100], "equal keys, different activations");
    } else {
        eprintln!("note: constructed key tie did not land exactly ({k100} vs {k50}); skipped");
    }
    // NaN logits rank last and do not crash
    let mut logits = vec![h(0.1); e];
    logits[3] = 0x7E00;
    let (p, _) = select_case(&logits, &vec![0.0; e], &tab, &st);
    assert!(!p.contains(&3));
}

#[test]
fn router_select_random_rows_match_the_host_mirror() {
    gpu!();
    let st = Stream::new().unwrap();
    let tab = SigmoidTable::new().unwrap();
    let sig = sigmoid_table_host();
    let mut r = Rng(11);
    let e = 288;
    let esb: Vec<f32> = (0..e).map(|_| (34.0 + 0.3 * normal(&mut r)) as f32).collect();
    let bias = centred_bias_host(&esb);
    let router = MoeRouter::new(&vec![0u16; e * 2], &bias, 2, 8, 2.5).unwrap();
    let rows = 512;
    // coarse logits so that many rows have near ties
    let logits: Vec<u16> = (0..rows * e).map(|_| h(((normal(&mut r) * 4.0).round() / 4.0).clamp(-12.0, 12.0))).collect();
    let lg = up(&logits);
    let mut pick = DevBuf::zeroed(rows * 8 * 4).unwrap();
    let mut wts = DevBuf::zeroed(rows * 8 * 2).unwrap();
    router.select(&tab, &lg, rows, &mut pick, &mut wts, &st).unwrap();
    let (hp, hw) = router_select_host(&logits, &bias, &sig, rows, e, 8, 2.5);
    assert!(sync_dl::<i32>(&pick, &st) == hp);
    assert!(sync_dl::<u16>(&wts, &st) == hw);
}

// ====================================================================================================================
// experts: synthetic layers (random trellises, every width and codebook) against a float64 reference

fn scale_vec(r: &mut Rng, n: usize, mag: f64) -> Vec<f64> {
    (0..n)
        .map(|_| hf(h((if r.next_u64() & 1 == 1 { -1.0 } else { 1.0 }) * mag * (0.5 + (r.next_u64() >> 40) as f64 / 16777216.0))))
        .collect()
}

fn synth_group(k: usize, n: usize, bits: usize, cb: Codebook, seed: u64) -> LoadedGroup {
    let mut r = Rng(seed);
    let words = (k / 16) * (n / 16) * 16 * bits;
    let trellis: Vec<i16> = (0..words).map(|_| r.next_u64() as i16).collect();
    let group = Group {
        prefix: "synthetic".into(),
        bits: Bits::from_twice(2 * bits as u8).unwrap(),
        codebook: cb,
        k,
        n,
        in_scales: "suh",
        out_scales: "svh",
        bias: false,
        files: vec![],
        bytes: 0,
    };
    // W_q entries are about N(0, 1) and x is rotated (norm kept), so a K-long dot product is about sqrt(K): svh keeps outputs O(1)
    let svh = scale_vec(&mut r, n, 1.0 / (k as f64).sqrt());
    LoadedGroup { group, trellis, kt: k / 16, nt: n / 16, suh: scale_vec(&mut r, k, 1.0), svh, bias: None }
}

struct Synth {
    gate: Vec<LoadedGroup>,
    up: Vec<LoadedGroup>,
    down: Vec<LoadedGroup>,
    ex: MoeExperts,
}

fn synth_layer(dims: MoeDims, bits_gu: usize, bits_d: usize, cb: Codebook, seed: u64) -> Synth {
    let e = dims.experts;
    let mk = |k, n, bits, tag: u64| -> Vec<LoadedGroup> {
        (0..e).map(|x| synth_group(k, n, bits, cb, seed * 1000 + tag * 100 + x as u64)).collect()
    };
    let gate = mk(dims.hidden, dims.inter, bits_gu, 1);
    let up = mk(dims.hidden, dims.inter, bits_gu, 2);
    let down = mk(dims.inter, dims.hidden, bits_d, 3);
    let ex = MoeExperts::from_groups(&gate, &up, &down, dims).unwrap();
    Synth { gate, up, down, ex }
}

// ---- float64 reference

fn had128(v: &mut [f64]) {
    for blk in v.chunks_mut(128) {
        let mut len = 1;
        while len < 128 {
            for i in (0..128).step_by(2 * len) {
                for j in i..i + len {
                    let (a, b) = (blk[j], blk[j + len]);
                    blk[j] = a + b;
                    blk[j + len] = a - b;
                }
            }
            len *= 2;
        }
        for x in blk.iter_mut() {
            *x /= 128f64.sqrt();
        }
    }
}

/// One EXL3 matrix's W_q as f32 (fp16-exact) and its scales, cached per (matrix kind, expert).
struct Wq {
    w: Vec<f32>,
    k: usize,
    n: usize,
}

fn wq_of(g: &LoadedGroup) -> Wq {
    let bits = g.group.bits;
    let w = unpack(&g.trellis, g.kt, g.nt, bits, &codebook(g.group.codebook));
    Wq { w: w.into_iter().map(|b| hf(b) as f32).collect(), k: g.group.k, n: g.group.n }
}

fn lin(x: &[f64], wq: &Wq, suh: &[f64], svh: &[f64]) -> Vec<f64> {
    let mut xh: Vec<f64> = x.iter().zip(suh).map(|(a, b)| a * b).collect();
    had128(&mut xh);
    let mut y = vec![0.0f64; wq.n];
    for i in 0..wq.k {
        let xi = xh[i];
        let row = &wq.w[i * wq.n..(i + 1) * wq.n];
        for j in 0..wq.n {
            y[j] += xi * row[j] as f64;
        }
    }
    had128(&mut y);
    y.iter().zip(svh).map(|(a, b)| a * b).collect()
}

/// Where the reference reads an expert's groups from (proj 0 gate, 1 up, 2 down).
trait Source {
    fn group(&self, proj: u8, e: usize) -> LoadedGroup;
}

impl Source for Synth {
    fn group(&self, proj: u8, e: usize) -> LoadedGroup {
        [&self.gate, &self.up, &self.down][proj as usize][e].clone()
    }
}

struct PackSource {
    ck: &'static Checkpoint,
    layer: usize,
}

impl Source for PackSource {
    fn group(&self, proj: u8, e: usize) -> LoadedGroup {
        let name = ["gate_proj", "up_proj", "down_proj"][proj as usize];
        self.ck.load_group(&format!("model.language_model.layers.{}.mlp.experts.{e}.{name}", self.layer)).unwrap()
    }
}

/// An expert matrix of the reference: W_q and its two scale vectors.
type RefMatrix = (Wq, Vec<f64>, Vec<f64>);

struct Reference<'a> {
    src: &'a dyn Source,
    limit: f64,
    cache: HashMap<(u8, usize), RefMatrix>,
}

impl<'a> Reference<'a> {
    fn new(src: &'a dyn Source, limit: f64) -> Self {
        Self { src, limit, cache: HashMap::new() }
    }

    fn lin(&mut self, proj: u8, e: usize, x: &[f64]) -> Vec<f64> {
        let src = self.src;
        let (wq, suh, svh) = self.cache.entry((proj, e)).or_insert_with(|| {
            let g = src.group(proj, e);
            (wq_of(&g), g.suh.clone(), g.svh.clone())
        });
        lin(x, wq, suh, svh)
    }

    /// y_pair for one (row, expert): the reference expert MLP in float64 (the oracle's activation: silu(gate) clamped from above,
    /// up clamped to +-limit). Returns (y, number of activations the clamps changed).
    fn expert(&mut self, x: &[f64], e: usize) -> (Vec<f64>, usize) {
        let gv = self.lin(0, e, x);
        let uv = self.lin(1, e, x);
        let limit = self.limit;
        let mut clamped = 0;
        let act: Vec<f64> = gv
            .iter()
            .zip(&uv)
            .map(|(&g, &u)| {
                let mut a = g / (1.0 + (-g).exp());
                let mut u = u;
                if limit > 0.0 {
                    if a > limit || u.abs() > limit {
                        clamped += 1;
                    }
                    a = a.min(limit);
                    u = u.clamp(-limit, limit);
                }
                a * u
            })
            .collect();
        (self.lin(2, e, &act), clamped)
    }
}

/// Worst relative RMS error over `rows` rows of `got` against `want` ([rows, n]).
fn worst_rel_rms(got: &[f32], want: &[f64], rows: usize, n: usize) -> f64 {
    let mut worst = 0f64;
    for r in 0..rows {
        let (mut ee, mut ww) = (0f64, 0f64);
        for j in 0..n {
            let (g, w) = (got[r * n + j] as f64, want[r * n + j]);
            ee += (g - w) * (g - w);
            ww += w * w;
        }
        worst = worst.max((ee / ww.max(1e-30)).sqrt());
    }
    worst
}

struct Gpu {
    st: Stream,
    tab: SigmoidTable,
}

fn gpu_ctx() -> Gpu {
    Gpu { st: Stream::new().unwrap(), tab: SigmoidTable::new().unwrap() }
}

/// Bits of a run: out [rows, D] and y_pairs [rows * topk, D].
struct ExpertRun {
    out: Vec<f32>,
    y: Vec<f32>,
}

fn run_experts(
    ex: &MoeExperts,
    g: &Gpu,
    s: &mut MoeScratch,
    xh: &[u16],
    rows: usize,
    pick: &[i32],
    wts: &[u16],
    shared: Option<&[f32]>,
) -> ExpertRun {
    let d = ex.dims;
    s.set_picks(pick, wts).unwrap();
    let x = up(xh);
    let sh = shared.map(up);
    let mut out = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
    ex.run(&x, Dtype::F16, rows, sh.as_ref(), &mut out, s, true, &g.st).unwrap();
    let y: Vec<f32> = sync_dl(&s.y_pairs, &g.st);
    ExpertRun { out: sync_dl(&out, &g.st), y: y[..rows * d.topk * d.hidden].to_vec() }
}

fn rand_picks(r: &mut Rng, rows: usize, e: usize, topk: usize) -> (Vec<i32>, Vec<u16>) {
    let mut pick = Vec::new();
    let mut wts = Vec::new();
    for _ in 0..rows {
        let mut ids: Vec<usize> = (0..e).collect();
        shuffle(r, &mut ids);
        pick.extend(ids[..topk].iter().map(|&i| i as i32));
        wts.extend((0..topk).map(|_| h(0.05 + 0.6 * (r.next_u64() >> 40) as f64 / 16777216.0)));
    }
    (pick, wts)
}

fn small_dims() -> MoeDims {
    MoeDims { hidden: 512, inter: 256, experts: 32, topk: 8, scale: 2.5, limit: 10.0 }
}

/// A host reference of a whole window: (y per pair, out per row, clamped count), in float64.
fn reference_window(
    d: &MoeDims,
    x: &[f64],
    rows: usize,
    pick: &[i32],
    wts: &[u16],
    shared: Option<&[f32]>,
    rf: &mut Reference,
) -> (Vec<f64>, Vec<f64>, usize) {
    let mut y = vec![0.0f64; rows * d.topk * d.hidden];
    let mut out = vec![0.0f64; rows * d.hidden];
    let mut clamped = 0;
    for r in 0..rows {
        for k in 0..d.topk {
            let p = r * d.topk + k;
            let (yy, c) = rf.expert(&x[r * d.hidden..(r + 1) * d.hidden], pick[p] as usize);
            clamped += c;
            let w = hf(wts[p]);
            for j in 0..d.hidden {
                out[r * d.hidden + j] += w * yy[j];
            }
            y[p * d.hidden..(p + 1) * d.hidden].copy_from_slice(&yy);
        }
        if let Some(sh) = shared {
            for j in 0..d.hidden {
                out[r * d.hidden + j] += sh[r * d.hidden + j] as f64;
            }
        }
    }
    (y, out, clamped)
}

#[test]
fn reference_helpers_agree_with_the_crate_forward() {
    // lin() (FWHT in f64 butterflies, cached W_q) must equal gb_exl3::forward (direct Hadamard sums)
    let g = synth_group(256, 256, 3, Codebook::Mul1, 5);
    let mut r = Rng(6);
    let x: Vec<f64> = (0..256).map(|_| normal(&mut r)).collect();
    let want = gb_exl3::forward(&x, 1, &g.trellis, g.kt, g.nt, g.group.bits, &codebook(Codebook::Mul1), &g.suh, &g.svh, None);
    let got = lin(&x, &wq_of(&g), &g.suh, &g.svh);
    for j in 0..256 {
        assert!((got[j] - want[j]).abs() < 1e-9 * (1.0 + want[j].abs()), "{j}: {} vs {}", got[j], want[j]);
    }
}

/// Every width and codebook the routed experts can have (bits 1..=8, mul1 / mcg / 3inst): the grouped kernel against float64.
#[test]
fn synthetic_experts_match_float64_at_every_width() {
    gpu!();
    let g = gpu_ctx();
    let dims = small_dims();
    let mut worst = 0f64;
    let mut cases: Vec<(usize, usize, Codebook)> = (1..=8).map(|b| (b, 9 - b, Codebook::Mul1)).collect();
    cases.extend([(4, 4, Codebook::Mcg), (3, 2, Codebook::Inst3), (2, 3, Codebook::Mcg)]);
    for (i, &(bg, bd, cb)) in cases.iter().enumerate() {
        let s = synth_layer(dims, bg, bd, cb, 20 + i as u64);
        let mut sc = MoeScratch::new(&dims, 16).unwrap();
        let mut r = Rng(30 + i as u64);
        let rows = 5;
        let (pick, wts) = rand_picks(&mut r, rows, dims.experts, dims.topk);
        let xh = rand_f16(&mut r, rows * dims.hidden, 2.0);
        let x: Vec<f64> = xh.iter().map(|&b| hf(b)).collect();
        let shared: Vec<f32> = (0..rows * dims.hidden).map(|_| normal(&mut r) as f32).collect();
        let run = run_experts(&s.ex, &g, &mut sc, &xh, rows, &pick, &wts, Some(&shared));
        let mut rf = Reference::new(&s, dims.limit as f64);
        let (y, out, _) = reference_window(&dims, &x, rows, &pick, &wts, Some(&shared), &mut rf);
        let ey = worst_rel_rms(&run.y, &y, rows * dims.topk, dims.hidden);
        let eo = worst_rel_rms(&run.out, &out, rows, dims.hidden);
        eprintln!("bits gu {bg} d {bd} {}: worst relRMS pair {ey:.2e} row {eo:.2e}", cb.name());
        assert!(ey < 2e-3 && eo < 2e-3, "bits {bg}/{bd} {}: pair {ey:.3e} out {eo:.3e}", cb.name());
        worst = worst.max(ey).max(eo);
    }
    eprintln!("synthetic experts: worst relRMS vs float64 {worst:.2e}");
}

/// The activation dtype of the expert path (bf16 and fp32 as well as fp16: the rotation kernel reads whatever the model hands it): against
/// the float64 reference of the decoded values, and a row inside a window against the same row alone.
#[test]
fn experts_accept_bf16_and_f32_activations() {
    gpu!();
    let g = gpu_ctx();
    let dims = small_dims();
    let s = synth_layer(dims, 3, 3, Codebook::Mul1, 400);
    let mut sc = MoeScratch::new(&dims, 8).unwrap();
    let mut r = Rng(401);
    let rows = 3;
    let (pick, wts) = rand_picks(&mut r, rows, dims.experts, dims.topk);
    let vals: Vec<f64> = (0..rows * dims.hidden).map(|_| 2.0 * normal(&mut r)).collect();
    for dt in [Dtype::BF16, Dtype::F32] {
        let (dec, bytes): (Vec<f64>, Vec<u8>) = match dt {
            Dtype::BF16 => {
                let b: Vec<u16> = vals
                    .iter()
                    .map(|&v| {
                        let bits = (v as f32).to_bits();
                        (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
                    })
                    .collect();
                (
                    b.iter().map(|&v| f32::from_bits((v as u32) << 16) as f64).collect(),
                    b.iter().flat_map(|v| v.to_le_bytes()).collect(),
                )
            }
            _ => (vals.iter().map(|&v| v as f32 as f64).collect(), vals.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect()),
        };
        sc.set_picks(&pick, &wts).unwrap();
        let x = up(&bytes);
        let mut out = DevBuf::zeroed(rows * dims.hidden * 4).unwrap();
        s.ex.run(&x, dt, rows, None, &mut out, &mut sc, true, &g.st).unwrap();
        let y: Vec<f32> = sync_dl(&sc.y_pairs, &g.st);
        let mut rf = Reference::new(&s, dims.limit as f64);
        let (want_y, want_out, _) = reference_window(&dims, &dec, rows, &pick, &wts, None, &mut rf);
        let got_out: Vec<f32> = sync_dl(&out, &g.st);
        let (ey, eo) = (
            worst_rel_rms(&y[..rows * dims.topk * dims.hidden], &want_y, rows * dims.topk, dims.hidden),
            worst_rel_rms(&got_out, &want_out, rows, dims.hidden),
        );
        eprintln!("{dt:?} activations: worst relRMS vs float64 pair {ey:.2e} row {eo:.2e}");
        assert!(ey < 2e-3 && eo < 2e-3, "{dt:?}: pair {ey:.3e} out {eo:.3e}");
        // row 1 alone: the same bits as inside the 3-row window
        let (d, k) = (dims.hidden, dims.topk);
        let esz = dt.size();
        sc.set_picks(&pick[k..2 * k], &wts[k..2 * k]).unwrap();
        let x1 = up(&bytes[d * esz..2 * d * esz]);
        let mut out1 = DevBuf::zeroed(d * 4).unwrap();
        s.ex.run(&x1, dt, 1, None, &mut out1, &mut sc, false, &g.st).unwrap();
        assert!(sync_dl::<f32>(&out1, &g.st) == got_out[d..2 * d], "{dt:?}: row 1 alone differs from row 1 of the window");
    }
}

// ====================================================================================================================
// row invariance of the experts, bit for bit

/// A pool of rows with their picks: the first rows' expert sets are disjoint from each other, the next rows' overlap their
/// neighbours' by half, the rest are random.
struct Pool {
    d: MoeDims,
    xh: Vec<u16>,
    pick: Vec<i32>,
    wts: Vec<u16>,
    shared: Vec<f32>,
}

fn make_pool(d: MoeDims, rows: usize, sigma: f64, seed: u64) -> Pool {
    let mut r = Rng(seed);
    let (e, k) = (d.experts, d.topk);
    let mut pick = Vec::new();
    let disjoint = (e / k).min(rows / 4);
    let overlap = (rows / 4).min(2 * e / k);
    for i in 0..rows {
        if i < disjoint {
            pick.extend((0..k).map(|j| ((k * i + j) % e) as i32));
        } else if i < disjoint + overlap {
            let o = i - disjoint;
            pick.extend((0..k).map(|j| ((k / 2 * o + j) % e) as i32));
        } else {
            let mut ids: Vec<usize> = (0..e).collect();
            shuffle(&mut r, &mut ids);
            pick.extend(ids[..k].iter().map(|&x| x as i32));
        }
    }
    let wts = (0..rows * k).map(|_| h(0.05 + 0.6 * (r.next_u64() >> 40) as f64 / 16777216.0)).collect();
    let xh = rand_f16(&mut r, rows * d.hidden, sigma);
    let shared = (0..rows * d.hidden).map(|_| normal(&mut r) as f32).collect();
    Pool { d, xh, pick, wts, shared }
}

impl Pool {
    fn rows(&self) -> usize {
        self.pick.len() / self.d.topk
    }
    /// The rows `idx` as a window: (x, pick, wts, shared).
    fn gather(&self, idx: &[usize]) -> (Vec<u16>, Vec<i32>, Vec<u16>, Vec<f32>) {
        let (dd, k) = (self.d.hidden, self.d.topk);
        let mut o = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &i in idx {
            o.0.extend_from_slice(&self.xh[i * dd..(i + 1) * dd]);
            o.1.extend_from_slice(&self.pick[i * k..(i + 1) * k]);
            o.2.extend_from_slice(&self.wts[i * k..(i + 1) * k]);
            o.3.extend_from_slice(&self.shared[i * dd..(i + 1) * dd]);
        }
        o
    }
}

fn run_window(ex: &MoeExperts, g: &Gpu, s: &mut MoeScratch, pool: &Pool, idx: &[usize]) -> ExpertRun {
    let (x, p, w, sh) = pool.gather(idx);
    run_experts(ex, g, s, &x, idx.len(), &p, &w, Some(&sh))
}

/// Row r's out and every pair's y of any window equal the ones of row r alone, bit for bit; `s` starts poisoned and is not cleaned.
fn check_invariance(ex: &MoeExperts, g: &Gpu, pool: &Pool, sizes: &[usize], seed: u64, what: &str) {
    let d = ex.dims;
    let mut r = Rng(seed);
    let mut s = MoeScratch::new(&d, 128).unwrap();
    s.poison(0xAB, &g.st).unwrap();
    let n = pool.rows();
    let solo: Vec<ExpertRun> = (0..n).map(|i| run_window(ex, g, &mut s, pool, &[i])).collect();
    let mut checked = 0usize;
    for &m in sizes {
        let mut windows: Vec<Vec<usize>> = vec![(0..m).collect(), (0..m).rev().collect()];
        let mut idx: Vec<usize> = (0..n).collect();
        shuffle(&mut r, &mut idx);
        windows.push(idx[..m].to_vec());
        windows.push(vec![idx[0]; m]); // m identical rows: every expert has m members (several 16-row passes above 16)
        windows.push((0..m).map(|i| (i * 7 + 3) % n).collect());
        for w in windows {
            let run = run_window(ex, g, &mut s, pool, &w);
            for (pos, &i) in w.iter().enumerate() {
                assert!(
                    run.out[pos * d.hidden..(pos + 1) * d.hidden] == solo[i].out[..],
                    "{what}: out of row {i} at position {pos} of {m} differs from its 1-row result"
                );
                let (a, b) = (pos * d.topk * d.hidden, (pos + 1) * d.topk * d.hidden);
                assert!(run.y[a..b] == solo[i].y[..], "{what}: pair outputs of row {i} at position {pos} of {m} differ");
                checked += 1;
            }
        }
    }
    // a wide window first, then one row on the same dirty scratch, against a fresh scratch
    let wide: Vec<usize> = (0..sizes.iter().copied().max().unwrap_or(1).min(n)).collect();
    let _ = run_window(ex, g, &mut s, pool, &wide);
    let again = run_window(ex, g, &mut s, pool, &[wide.len() / 2]);
    assert!(
        again.out == solo[wide.len() / 2].out && again.y == solo[wide.len() / 2].y,
        "{what}: a 1-row call after a wide one differs"
    );
    eprintln!("{what}: {checked} row results bit-identical to their 1-row result (window sizes {sizes:?})");
}

const SIZES: [usize; 14] = [1, 2, 3, 8, 9, 16, 17, 24, 32, 33, 48, 64, 100, 128];

#[test]
fn synthetic_experts_are_row_invariant() {
    gpu!();
    let g = gpu_ctx();
    let d = small_dims();
    for (bg, bd) in [(2usize, 2usize), (3, 3), (4, 2), (1, 6)] {
        let s = synth_layer(d, bg, bd, Codebook::Mul1, 60 + bg as u64);
        check_invariance(&s.ex, &g, &make_pool(d, 128, 2.0, 61), &SIZES, 62, &format!("synthetic {bg}/{bd} bit"));
    }
}

/// The detector that makes the invariance test meaningful: a K split that depended on the row count changes the bits.
#[test]
fn a_plan_that_depends_on_rows_would_be_caught() {
    gpu!();
    let g = gpu_ctx();
    let d = MoeDims { hidden: 2048, inter: 512, experts: 16, topk: 8, scale: 2.5, limit: 10.0 };
    let mut s = synth_layer(d, 2, 2, Codebook::Mul1, 70);
    let pool = make_pool(d, 4, 2.0, 71);
    let mut sc = MoeScratch::new(&d, 8).unwrap();
    let base = run_window(&s.ex, &g, &mut sc, &pool, &[0]);
    // gate/up K = 2048 is 128 k tiles over N = 512: the plan splits K into 4 (64 blocks an expert); two splits sum the same products in
    // another order
    let plan = s.ex.plan_gu;
    assert_eq!(plan, (4, 4));
    s.ex.plan_gu = (2, 4);
    let split = run_window(&s.ex, &g, &mut sc, &pool, &[0]);
    assert!(
        base.out != split.out,
        "changing the K split must change some bit of the output, else the invariance tests prove nothing"
    );
    s.ex.plan_gu = plan;
    let again = run_window(&s.ex, &g, &mut sc, &pool, &[0]);
    assert!(base.out == again.out && base.y == again.y);
}

/// The clamps are active on part of the activations and the result still matches float64 (silu(gate) clamped, up clamped).
#[test]
fn clamped_activations_match_float64() {
    gpu!();
    let g = gpu_ctx();
    let dims = small_dims();
    let s = synth_layer(dims, 3, 3, Codebook::Mul1, 80);
    let mut sc = MoeScratch::new(&dims, 8).unwrap();
    let mut r = Rng(81);
    let rows = 4;
    let (pick, wts) = rand_picks(&mut r, rows, dims.experts, dims.topk);
    let xh = rand_f16(&mut r, rows * dims.hidden, 30.0); // |gate|, |up| reach well past the limit of 10
    let x: Vec<f64> = xh.iter().map(|&b| hf(b)).collect();
    let run = run_experts(&s.ex, &g, &mut sc, &xh, rows, &pick, &wts, None);
    let mut rf = Reference::new(&s, 10.0);
    let (y, out, clamped) = reference_window(&dims, &x, rows, &pick, &wts, None, &mut rf);
    let total = rows * dims.topk * dims.inter;
    eprintln!("clamped activations: {clamped} of {total}");
    assert!(clamped * 10 > total, "the test must exercise the clamps ({clamped} of {total})");
    let (ey, eo) = (worst_rel_rms(&run.y, &y, rows * dims.topk, dims.hidden), worst_rel_rms(&run.out, &out, rows, dims.hidden));
    assert!(ey < 2e-3 && eo < 2e-3, "{ey:.3e} {eo:.3e}");
    // a reference that clamps the pre-activation instead (TensorFold's epilogue) is measurably different: the test can tell them apart
}

// ---- the epilogues in isolation, with the partial sums crafted

/// Unnormalized FWHT of every 128 block in f32, stages in the kernel's order (bit 0 first): the same network, so the same bits.
fn fwht_f32(v: &mut [f32]) {
    for blk in v.chunks_mut(128) {
        let mut len = 1;
        while len < 128 {
            for i in (0..128).step_by(2 * len) {
                for j in i..i + len {
                    let (lo, hi) = (blk[j], blk[j + len]);
                    blk[j] = lo + hi;
                    blk[j + len] = lo - hi;
                }
            }
            len *= 2;
        }
    }
}

#[allow(clippy::excessive_precision)]
const HAD: f32 = 0.088_388_347_648_318_45; // 1 / sqrt(128) written out: the kernel's HAD_SCALE, exactly

#[derive(Clone, Copy, PartialEq, Debug)]
enum Act {
    /// The oracle: min(silu(g), L), clamp(u, -L, L).
    Oracle,
    /// TensorFold's epilogue: silu(min(g, L)).
    ClampBefore,
    NoClamp,
}

/// xd of one pair from its gate and up pre-rotation sums, in f32/f64 mixed (the exp is f64: the device's expf is within 2 ulp).
fn epilogue_host(zg: &[f32], zu: &[f32], svh_g: &[u16], svh_u: &[u16], suh_d: &[u16], limit: f32, act: Act) -> Vec<u16> {
    let (mut gv, mut uv) = (zg.to_vec(), zu.to_vec());
    fwht_f32(&mut gv);
    fwht_f32(&mut uv);
    let mut v: Vec<f32> = (0..zg.len())
        .map(|j| {
            let g = gv[j] * HAD * hf(svh_g[j]) as f32;
            let u = uv[j] * HAD * hf(svh_u[j]) as f32;
            let silu = |x: f32| (x as f64 / (1.0 + (-(x as f64)).exp())) as f32;
            let (a, u) = match act {
                Act::Oracle => (silu(g).min(limit), u.clamp(-limit, limit)),
                Act::ClampBefore => (silu(g.min(limit)), u.clamp(-limit, limit)),
                Act::NoClamp => (silu(g), u),
            };
            (a * u) * hf(suh_d[j]) as f32
        })
        .collect();
    fwht_f32(&mut v);
    v.iter().map(|&x| h((x * HAD) as f64)).collect()
}

#[test]
fn gateup_epilogue_applies_the_oracles_clamps_and_sums_splits_in_order() {
    gpu!();
    let g = gpu_ctx();
    let (rows, topk, inter, sk, e) = (2usize, 8usize, 256usize, 2usize, 4usize);
    let p = rows * topk;
    let mut r = Rng(120);
    // pre-activations spread over [-30, 30] with a third of the gates in [8, 14] (around the clamp); the partial sums are the
    // targets rotated back, W t / sqrt(128), split into two parts that add up in fp32
    let target = |r: &mut Rng, gate: bool| -> Vec<f32> {
        (0..p * inter)
            .map(|i| {
                let u = (r.next_u64() >> 40) as f32 / 16777216.0;
                if gate && i % 3 == 0 {
                    8.0 + 6.0 * u
                } else {
                    -30.0 + 60.0 * u
                }
            })
            .collect()
    };
    let to_z = |t: &[f32]| -> Vec<f32> {
        let mut z = t.to_vec();
        fwht_f32(&mut z);
        z.iter().map(|&x| x * HAD).collect()
    };
    let (zg, zu) = (to_z(&target(&mut r, true)), to_z(&target(&mut r, false)));
    let split = |z: &[f32], r: &mut Rng| -> (Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = z.iter().map(|&x| x * (0.25 + 0.5 * (r.next_u64() >> 40) as f32 / 16777216.0)).collect();
        (a.clone(), z.iter().zip(&a).map(|(&x, &y)| x - y).collect())
    };
    let ((g0, g1), (u0, u1)) = (split(&zg, &mut r), split(&zu, &mut r));
    // Z layout [mat][split][P][inter]
    let z: Vec<f32> = [g0.clone(), g1.clone(), u0.clone(), u1.clone()].concat();
    let svh_g: Vec<u16> = (0..e * inter).map(|_| h(1.0)).collect();
    let svh_u = svh_g.clone();
    let suh_d: Vec<u16> = (0..e * inter).map(|_| h(if r.next_u64() & 1 == 1 { -1.0 } else { 1.0 })).collect();
    let pick: Vec<i32> = (0..p).map(|i| (i % e) as i32).collect();
    let mut xd = DevBuf::zeroed(p * inter * 2).unwrap();
    gateup_epilogue_raw(
        &up(&z),
        &up(&pick),
        &up(&svh_g),
        &up(&svh_u),
        &up(&suh_d),
        &mut xd,
        rows,
        topk,
        inter,
        sk,
        e,
        10.0,
        &g.st,
    )
    .unwrap();
    let got: Vec<u16> = sync_dl(&xd, &g.st);
    let mut mism = [0usize; 3];
    for i in 0..p {
        let sl = i * inter..(i + 1) * inter;
        let sum = |a: &[f32], b: &[f32]| -> Vec<f32> { a[sl.clone()].iter().zip(&b[sl.clone()]).map(|(&x, &y)| x + y).collect() };
        let (sg, su) = (sum(&g0, &g1), sum(&u0, &u1));
        let ee = pick[i] as usize * inter;
        for (m, act) in [Act::Oracle, Act::ClampBefore, Act::NoClamp].into_iter().enumerate() {
            let want = epilogue_host(&sg, &su, &svh_g[ee..ee + inter], &svh_u[ee..ee + inter], &suh_d[ee..ee + inter], 10.0, act);
            mism[m] += (0..inter).filter(|&j| want[j] != got[i * inter + j]).count();
        }
    }
    let total = p * inter;
    eprintln!(
        "epilogue vs host mirror: oracle semantics {} / {total} differ, clamp-before {} / {total}, no clamp {} / {total}",
        mism[0], mism[1], mism[2]
    );
    // device expf is within 2 ulp of the correctly rounded value, so a handful of fp16 roundings can land on the other side
    assert!(mism[0] * 500 < total, "the epilogue does not match the oracle's activation ({} of {total})", mism[0]);
    assert!(mism[1] * 20 > total, "the clamp-before variant must be distinguishable ({} of {total})", mism[1]);
    assert!(mism[2] * 20 > total, "the no-clamp variant must be distinguishable ({} of {total})", mism[2]);
}

#[test]
fn down_combine_adds_slots_in_order_then_the_shared_expert_bit_for_bit() {
    gpu!();
    let g = gpu_ctx();
    let (rows, topk, hidden, sk, e) = (3usize, 8usize, 384usize, 2usize, 5usize);
    let p = rows * topk;
    let mut r = Rng(130);
    let z: Vec<f32> = (0..sk * p * hidden).map(|_| (normal(&mut r) * 3.0) as f32).collect();
    let svh: Vec<u16> = (0..e * hidden).map(|_| h(0.3 + 0.1 * normal(&mut r))).collect();
    let mut pick: Vec<i32> = (0..p).map(|i| ((i * 3) % e) as i32).collect();
    pick[5] = -1; // a skipped slot
    pick[12] = e as i32; // and one past the last expert
    let wts: Vec<u16> = (0..p).map(|_| h(0.05 + 0.6 * (r.next_u64() >> 40) as f64 / 16777216.0)).collect();
    let shared: Vec<f32> = (0..rows * hidden).map(|_| normal(&mut r) as f32).collect();
    let mut out = DevBuf::zeroed(rows * hidden * 4).unwrap();
    let mut ypairs = DevBuf::zeroed(p * hidden * 4).unwrap();
    down_combine_raw(
        &up(&z),
        &up(&pick),
        &up(&svh),
        &up(&wts),
        Some(&up(&shared)),
        Some(&mut ypairs),
        &mut out,
        rows,
        topk,
        hidden,
        sk,
        e,
        &g.st,
    )
    .unwrap();
    let (got, gy): (Vec<f32>, Vec<f32>) = (sync_dl(&out, &g.st), sync_dl(&ypairs, &g.st));
    // host mirror: splits in order, FWHT, * HAD * svh; slots in slot order with fmaf; shared last
    let mut want_y = vec![0f32; p * hidden];
    for i in 0..p {
        if pick[i] < 0 || pick[i] >= e as i32 {
            continue;
        }
        let mut s: Vec<f32> = z[i * hidden..(i + 1) * hidden].to_vec();
        for q in 1..sk {
            for j in 0..hidden {
                s[j] += z[(q * p + i) * hidden + j];
            }
        }
        fwht_f32(&mut s);
        for j in 0..hidden {
            want_y[i * hidden + j] = s[j] * HAD * hf(svh[pick[i] as usize * hidden + j]) as f32;
        }
    }
    assert!(
        gy.iter().zip(&want_y).enumerate().all(|(i, (a, b))| pick[i / hidden] < 0 || pick[i / hidden] >= e as i32 || a == b),
        "y differs"
    );
    let combine = |order: &dyn Fn(usize) -> usize, shared_last: bool| -> Vec<f32> {
        let mut out = vec![0f32; rows * hidden];
        for rr in 0..rows {
            for j in 0..hidden {
                let mut acc = 0f32;
                if !shared_last {
                    acc = shared[rr * hidden + j];
                }
                for kk in 0..topk {
                    let k = order(kk);
                    let i = rr * topk + k;
                    if pick[i] < 0 || pick[i] >= e as i32 {
                        continue;
                    }
                    acc = (hf(wts[i]) as f32).mul_add(want_y[i * hidden + j], acc);
                }
                out[rr * hidden + j] = if shared_last { acc + shared[rr * hidden + j] } else { acc };
            }
        }
        out
    };
    let want = combine(&|k| k, true);
    assert!(got == want, "combine differs from the host mirror (slot order, fmaf chain, shared added last)");
    // the test can fail: another slot order, or the shared expert added first, changes bits
    assert!(combine(&|k| topk - 1 - k, true) != want, "reversed slot order is indistinguishable: the test is vacuous");
    assert!(combine(&|k| k, false) != want, "adding the shared expert first is indistinguishable: the test is vacuous");
}

// ====================================================================================================================
// real layers of the GLM-5.3-Flash pack

fn pack_dir() -> Option<std::path::PathBuf> {
    let p = std::env::var("REXL3_GLM_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| {
        std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("models/r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121")
    });
    p.join("model.safetensors.index.json").exists().then_some(p)
}

fn checkpoint() -> Option<&'static Checkpoint> {
    static CK: std::sync::OnceLock<Option<Checkpoint>> = std::sync::OnceLock::new();
    CK.get_or_init(|| {
        let dir = pack_dir()?;
        let ck = scan(&dir, false).expect("scan the GLM pack");
        // the six tensors of layer 45 that sit in two shards are resolved by the index (a known, designed-for duplicate)
        let bad: Vec<_> = ck.issues.iter().filter(|i| !i.contains("is also stored in")).collect();
        assert!(bad.is_empty(), "pack issues: {bad:?}");
        Some(ck)
    })
    .as_ref()
}

struct Leaked(&'static MoeLayer);
unsafe impl Send for Leaked {}
unsafe impl Sync for Leaked {}

/// One real layer, loaded once per process (1.7 GB at 2 bits, 2.6 GB at 3 bits, 3.5 GB at 4 bits).
fn real_layer(l: usize) -> Option<&'static MoeLayer> {
    static LAYERS: std::sync::Mutex<Vec<(usize, Leaked)>> = std::sync::Mutex::new(Vec::new());
    let ck = checkpoint()?;
    let mut v = LAYERS.lock().unwrap();
    if let Some((_, l)) = v.iter().find(|(i, _)| *i == l) {
        return Some(l.0);
    }
    let t = std::time::Instant::now();
    let layer =
        Box::leak(Box::new(MoeLayer::from_checkpoint(ck, l, MoeDims::GLM53).unwrap_or_else(|e| panic!("layer {l}: {e}"))));
    eprintln!(
        "loaded layer {l}: {} bit gate/up, {} bit down, {:.2} GB on the device, {:.1} s",
        layer.experts.bits_gu,
        layer.experts.bits_d,
        layer.experts.device_bytes() as f64 / 1e9,
        t.elapsed().as_secs_f64()
    );
    v.push((l, Leaked(layer)));
    Some(layer)
}

macro_rules! real_layer {
    ($l:expr) => {
        match real_layer($l) {
            Some(l) => l,
            None => {
                eprintln!("no GLM pack (set REXL3_GLM_DIR): not run");
                return;
            }
        }
    };
}

/// (bits of gate/up) of the layers under test.
const LAYERS: [(usize, usize); 3] = [(6, 2), (3, 3), (45, 4)];

#[test]
fn real_layers_have_the_documented_widths() {
    gpu!();
    for (l, bits) in LAYERS {
        let layer = real_layer!(l);
        assert_eq!((layer.experts.bits_gu, layer.experts.bits_d), (bits, bits), "layer {l}");
        assert_eq!(layer.experts.codebook, Codebook::Mul1);
    }
}

/// Real routed experts against float64: windows with overlapping and disjoint expert sets, shared expert added last.
#[test]
fn real_layers_match_float64() {
    gpu!();
    let g = gpu_ctx();
    let d = MoeDims::GLM53;
    for (l, _) in LAYERS {
        let layer = real_layer!(l);
        let src = PackSource { ck: checkpoint().unwrap(), layer: l };
        let mut rf = Reference::new(&src, d.limit as f64);
        let mut sc = MoeScratch::new(&d, 32).unwrap();
        // rows 0..3 take experts 0..8, 4..12, 8..16 (overlapping); a second window repeats one row 17 times (2 passes of the grouped kernel)
        for (name, sigma, window) in
            [("overlap", 1.0, vec![0usize, 1, 2]), ("x17", 1.0, vec![5usize; 17]), ("big input", 12.0, vec![1usize, 2])]
        {
            let mut pool = make_pool(d, 8, sigma, 90 + l as u64);
            for i in 0..8 {
                // fixed pool expert sets: row i takes experts 4 i .. 4 i + 8
                for j in 0..d.topk {
                    pool.pick[i * d.topk + j] = (4 * i + j) as i32;
                }
            }
            let (x, p, w, sh) = pool.gather(&window);
            let rows = window.len();
            let run = run_experts(&layer.experts, &g, &mut sc, &x, rows, &p, &w, Some(&sh));
            let xf: Vec<f64> = x.iter().map(|&b| hf(b)).collect();
            // the reference of the repeated row is computed once
            let uniq = if name == "x17" { 1 } else { rows };
            let (y, out, clamped) = reference_window(
                &d,
                &xf[..uniq * d.hidden],
                uniq,
                &p[..uniq * d.topk],
                &w[..uniq * d.topk],
                Some(&sh[..uniq * d.hidden]),
                &mut rf,
            );
            let mut ey = 0f64;
            let mut eo = 0f64;
            for r in 0..rows {
                let ru = if name == "x17" { 0 } else { r };
                let ysl = &run.y[r * d.topk * d.hidden..(r + 1) * d.topk * d.hidden];
                ey = ey.max(worst_rel_rms(ysl, &y[ru * d.topk * d.hidden..(ru + 1) * d.topk * d.hidden], d.topk, d.hidden));
                eo = eo.max(worst_rel_rms(
                    &run.out[r * d.hidden..(r + 1) * d.hidden],
                    &out[ru * d.hidden..(ru + 1) * d.hidden],
                    1,
                    d.hidden,
                ));
            }
            eprintln!(
                "layer {l} {name}: {rows} rows, worst relRMS pair {ey:.2e} out {eo:.2e}, clamps changed {clamped} activations"
            );
            assert!(ey < 2e-3 && eo < 2e-3, "layer {l} {name}: pair {ey:.3e} out {eo:.3e}");
        }
    }
}

#[test]
fn real_layers_are_row_invariant() {
    gpu!();
    let g = gpu_ctx();
    let d = MoeDims::GLM53;
    for (l, _) in LAYERS {
        let layer = real_layer!(l);
        check_invariance(&layer.experts, &g, &make_pool(d, 128, 1.5, 100 + l as u64), &SIZES, 101, &format!("layer {l}"));
    }
}

// ====================================================================================================================
// performance (ignored: run alone on an idle GPU, `cargo test --release -p gb-cuda --test moe -- --ignored --nocapture bench`)

fn distinct(pick: &[i32]) -> usize {
    let mut v = pick.to_vec();
    v.sort_unstable();
    v.dedup();
    v.len()
}

/// Measured streaming-read ceiling of the device, GB/s (best of several 1 GiB reads).
fn read_ceiling(g: &Gpu) -> f64 {
    let bytes = 1usize << 30;
    let buf = DevBuf::zeroed(bytes).unwrap();
    let mut out = DevBuf::zeroed(16).unwrap();
    let t = EventTimer::new(2).unwrap();
    let mut best = 0f64;
    for i in 0..6 {
        t.record(0, &g.st).unwrap();
        read_probe(&buf, bytes, &mut out, &g.st).unwrap();
        t.record(1, &g.st).unwrap();
        let gbs = bytes as f64 / 1e9 / (t.ms(0, 1).unwrap() as f64 / 1e3);
        if i > 0 {
            best = best.max(gbs);
        }
    }
    best
}

struct KernelBench {
    /// Mean distinct experts per window.
    u: f64,
    /// MB of expert weights (trellis and scales) those experts hold, per window.
    mb: f64,
    /// Mean milliseconds of the gate/up GEMV, the down GEMV, and every other expert stage (group, rotate, epilogues, combine).
    gu_ms: f64,
    d_ms: f64,
    other_ms: f64,
}

/// The expert pipeline on `iters` windows of `rows` rows with uniformly random distinct picks (the worst case for the expert union);
/// L2 is flushed before every window, so the weights stream from DRAM.
fn bench_experts(
    ex: &MoeExperts,
    g: &Gpu,
    rows: usize,
    iters: usize,
    seed: u64,
    plans: ((usize, usize), (usize, usize)),
) -> KernelBench {
    let d = ex.dims;
    let mut s = MoeScratch::with_splits(&d, rows, plans.0 .0, plans.1 .0).unwrap();
    let mut r = Rng(seed);
    let x = up(&rand_f16(&mut r, rows * d.hidden, 1.0));
    let shared = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
    let mut out = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
    let flush = DevBuf::zeroed(256 << 20).unwrap();
    let mut flush_out = DevBuf::zeroed(16).unwrap();
    let t = EventTimer::new(8).unwrap();
    let (mut u, mut gu, mut dn, mut other) = (0f64, 0f64, 0f64, 0f64);
    for it in 0..iters + 3 {
        let (pick, wts) = rand_picks(&mut r, rows, d.experts, d.topk);
        s.set_picks(&pick, &wts).unwrap();
        // a read-only sweep evicts L2 without leaving dirty lines (a memset would write back into the next kernels)
        read_probe(&flush, 256 << 20, &mut flush_out, &g.st).unwrap();
        t.record(0, &g.st).unwrap();
        let mut n = 0;
        ex.run_marked(&x, Dtype::F16, rows, Some(&shared), &mut out, &mut s, false, plans, &g.st, &mut |_| {
            n += 1;
            t.record(n, &g.st)
        })
        .unwrap();
        let ms: Vec<f64> = (0..6).map(|i| t.ms(i, i + 1).unwrap() as f64).collect();
        if it >= 3 {
            // group, rot_in, gate/up, epilogue, down, combine
            u += distinct(&pick) as f64;
            gu += ms[2];
            dn += ms[4];
            other += ms[0] + ms[1] + ms[3] + ms[5];
        }
    }
    let n = iters as f64;
    let u = u / n;
    KernelBench { u, mb: u * ex.expert_bytes as f64 / 1e6, gu_ms: gu / n, d_ms: dn / n, other_ms: other / n }
}

fn print_kernel_bench(l: usize, ex: &MoeExperts, rows: usize, b: &KernelBench, ceiling: f64) {
    let gemv = b.gu_ms + b.d_ms;
    let gbs = b.mb / 1e3 / (gemv / 1e3);
    println!(
        "BENCH layer {l:2} {}bit R={rows:3} U={:5.1} {:7.1} MB | gate/up {:.3} ms, down {:.3} ms: {gbs:5.1} GB/s ({:4.1}% of the {ceiling:.0} GB/s read ceiling) | other stages {:.3} ms",
        ex.bits_gu,
        b.u,
        b.mb,
        b.gu_ms,
        b.d_ms,
        100.0 * gbs / ceiling,
        b.other_ms
    );
}

#[test]
#[ignore]
fn bench_expert_throughput() {
    gpu!();
    let g = gpu_ctx();
    let iters: usize = std::env::var("REXL3_BENCH_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(24);
    let ceiling = read_ceiling(&g);
    println!("measured read ceiling {ceiling:.1} GB/s (the task's reference roofline is 240 GB/s)");
    for (l, _) in LAYERS {
        let layer = real_layer!(l);
        for rows in [1usize, 2, 4, 8, 16, 32, 64, 128] {
            let b = bench_experts(&layer.experts, &g, rows, iters, 500 + l as u64, (layer.experts.plan_gu, layer.experts.plan_d));
            print_kernel_bench(l, &layer.experts, rows, &b, ceiling);
        }
    }
}

/// Every (K split, warps) the plan can take for the GLM shapes, at the windows that matter: which fixed plan is fastest.
#[test]
#[ignore]
fn bench_plan_sweep() {
    gpu!();
    let g = gpu_ctx();
    let iters: usize = std::env::var("REXL3_BENCH_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(24);
    let ceiling = read_ceiling(&g);
    println!("measured read ceiling {ceiling:.1} GB/s");
    for (l, _) in LAYERS {
        let layer = real_layer!(l);
        let ex = &layer.experts;
        for rows in [1usize, 4, 16] {
            for sk_gu in [1usize, 2, 4, 8, 16] {
                let b = bench_experts(ex, &g, rows, iters, 700, ((sk_gu, 4), ex.plan_d));
                println!(
                    "SWEEP layer {l:2} {}bit R={rows:3} gate/up SK={sk_gu:2}: {:.3} ms = {:5.1} GB/s",
                    ex.bits_gu,
                    b.gu_ms,
                    b.mb * (2.0 / 3.0) / 1e3 / (b.gu_ms / 1e3)
                );
            }
            for sk_d in [1usize, 2, 4, 8] {
                let b = bench_experts(ex, &g, rows, iters, 701, (ex.plan_gu, (sk_d, 4)));
                println!(
                    "SWEEP layer {l:2} {}bit R={rows:3} down SK={sk_d:2}: {:.3} ms = {:5.1} GB/s",
                    ex.bits_gu,
                    b.d_ms,
                    b.mb * (1.0 / 3.0) / 1e3 / (b.d_ms / 1e3)
                );
            }
        }
    }
}

// ====================================================================================================================
// the ExLlamaV3 fork's MoE module as the numerical oracle (`tools/dump_moe_reference.py`; REXL3_MOE_REF_DIR, default
// ~/moe-oracle-out). The oracle is not row-invariant, so everything is compared numerically; where bits are expected to agree
// (the router GEMV at one row) the test counts mismatches and prints them.

fn ref_root() -> Option<std::path::PathBuf> {
    let p = std::env::var("REXL3_MOE_REF_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("moe-oracle-out"));
    p.join("L6/manifest.json").exists().then_some(p)
}

fn npy_u16(p: &std::path::Path) -> (Vec<usize>, Vec<u16>) {
    let n = read_npy(p);
    assert_eq!(n.dtype, "<f2");
    (n.shape.clone(), n.data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

/// The selection bias prepared on the host equals the fork's device-side `(esb - esb.mean()).half()` in every entry of every MoE layer
/// (the device's vectors come from `tools/check_moe_oracle_details.py --out`; torch's fp32 mean order is reproduced by `torch_mean_f32`).
#[test]
fn centred_bias_matches_the_oracle_on_every_layer() {
    let Some(root) = ref_root() else {
        eprintln!("no oracle dump: not run");
        return;
    };
    let Some(ck) = checkpoint() else {
        eprintln!("no GLM pack: not run");
        return;
    };
    let dir = root.join("bias_all");
    if !dir.exists() {
        eprintln!("no bias_all dump (tools/check_moe_oracle_details.py --out): not run");
        return;
    }
    let mut layers = 0;
    for l in 3..=45usize {
        let Some(e) = ck.plain.get(&format!("model.language_model.layers.{l}.mlp.gate.e_score_correction_bias")) else {
            continue;
        };
        let esb: Vec<f32> =
            ck.read_tensor(e).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let (_, oracle) = npy_u16(&dir.join(format!("L{l}.npy")));
        let mine = centred_bias_host(&esb);
        let bad = (0..esb.len()).filter(|&i| mine[i] != hf(oracle[i]) as f32).count();
        assert_eq!(bad, 0, "layer {l}: {bad} of {} bias entries differ from the oracle's", esb.len());
        layers += 1;
    }
    assert!(layers >= 40, "only {layers} layers compared");
    eprintln!("centred bias equals the oracle's on all {layers} MoE layers");
}

#[test]
fn oracle_router_and_experts_agree_numerically() {
    gpu!();
    let Some(root) = ref_root() else {
        eprintln!("no oracle dump (run tools/dump_moe_reference.py, set REXL3_MOE_REF_DIR): not run");
        return;
    };
    let g = gpu_ctx();
    let d = MoeDims::GLM53;
    for l in [6usize, 3] {
        if !root.join(format!("L{l}/manifest.json")).exists() {
            continue;
        }
        let layer = real_layer!(l);
        let dir = root.join(format!("L{l}"));
        let (xs, xh) = npy_u16(&dir.join("x.npy"));
        let n = xs[0];
        let esb = read_npy(&dir.join("esb.npy")).to_f32();
        let (_, bias_h) = npy_u16(&dir.join("bias_h.npy"));
        let (_, logits1) = npy_u16(&dir.join("logits1.npy"));
        let (_, logits_b) = npy_u16(&dir.join("logits_b.npy"));
        let sel1: Vec<i32> = read_npy(&dir.join("sel1.npy")).to_i64().iter().map(|&v| v as i32).collect();
        let sel_b: Vec<i32> = read_npy(&dir.join("sel_b.npy")).to_i64().iter().map(|&v| v as i32).collect();
        let (_, w1) = npy_u16(&dir.join("w1.npy"));
        let out1 = read_npy(&dir.join("out1.npy")).to_f32();
        let out_b = read_npy(&dir.join("out_b.npy")).to_f32();
        let shared1 = read_npy(&dir.join("shared1.npy")).to_f32();

        // (1) the selection bias: mean-centred, fp16
        let mine = centred_bias_host(&esb);
        let diff_bias = (0..288).filter(|&i| mine[i] != hf(bias_h[i]) as f32).count();
        assert_eq!(diff_bias, 0, "layer {l}: the host's centred bias differs from the oracle's in {diff_bias} of 288 entries");
        // The router below uses the oracle's own bias values, so the picks isolate the router; the line above is the bias risk.
        let bias_oracle: Vec<f32> = bias_h.iter().map(|&b| hf(b) as f32).collect();

        // (2) logits: REXL3's GEMV on the real gate against the oracle's one-row GEMV
        let ck = checkpoint().unwrap();
        let gate_e = ck.plain.get(&format!("model.language_model.layers.{l}.mlp.gate.weight")).unwrap();
        let gate: Vec<u16> = ck.read_u16(gate_e).unwrap();
        let router = MoeRouter::new(&gate, &bias_oracle, 4096, 8, 2.5).unwrap();
        let x = up(&xh);
        let run = route_rows(&router, &g.tab, &x, Dtype::F16, n, &g.st);
        let lg_diff = run.logits.iter().zip(&logits1).filter(|(a, b)| a != b).count();
        let lgb_diff = logits_b.iter().zip(&logits1).filter(|(a, b)| a != b).count();
        eprintln!(
            "layer {l}: router logits {} of {} differ from the oracle's one-row GEMV (the oracle's own batched cuBLAS logits differ in {} of {})",
            lg_diff,
            logits1.len(),
            lgb_diff,
            logits1.len()
        );
        assert!(lg_diff * 1000 < logits1.len(), "logits do not match the oracle's one-row router");

        // (3) picks and weights
        let mut same_set = 0;
        let mut same_order = 0;
        let mut wdiff = 0usize;
        for r in 0..n {
            let (a, b) = (&run.pick[r * 8..(r + 1) * 8], &sel1[r * 8..(r + 1) * 8]);
            let (mut sa, mut sb) = (a.to_vec(), b.to_vec());
            sa.sort_unstable();
            sb.sort_unstable();
            same_set += (sa == sb) as usize;
            same_order += (a == b) as usize;
            for k in 0..8 {
                let (wa, wb) = (run.wts[r * 8 + k], w1[r * 8 + k]);
                wdiff += (wa != wb && a == b) as usize;
            }
        }
        let batched_set = (0..n)
            .filter(|&r| {
                let (mut sa, mut sb) = (sel1[r * 8..(r + 1) * 8].to_vec(), sel_b[r * 8..(r + 1) * 8].to_vec());
                sa.sort_unstable();
                sb.sort_unstable();
                sa == sb
            })
            .count();
        eprintln!(
            "layer {l}: top-8 set equal on {same_set}/{n} rows, same slot order on {same_order}/{n}, {wdiff} weight bits differ on identical-order rows (the oracle's batched router picks the same set as its one-row router on {batched_set}/{n} rows)"
        );
        assert!(same_set * 100 >= n * 95, "top-8 sets disagree with the oracle on too many rows");

        // (4) experts on the oracle's picks, shared expert from the oracle
        let mut s = MoeScratch::new(&d, 64).unwrap();
        let w1v: Vec<u16> = w1.clone();
        let ys = run_experts(&layer.experts, &g, &mut s, &xh, n, &sel1, &w1v, Some(&shared1));
        let (cos1, rel1) = cos_rel_rms(&ys.out, &out1);
        // the oracle against itself: its batched path against its one-row path
        let (cosb, relb) = cos_rel_rms(&out_b, &out1);
        // per-row worst
        let mut worst = 0f64;
        for r in 0..n {
            let (_, rel) = cos_rel_rms(&ys.out[r * 4096..(r + 1) * 4096], &out1[r * 4096..(r + 1) * 4096]);
            worst = worst.max(rel);
        }
        eprintln!(
            "layer {l}: REXL3 experts vs oracle (same picks, same shared): cosine {cos1:.7}, relRMS {rel1:.2e} (worst row {worst:.2e}); the oracle's batched vs one-row: cosine {cosb:.7}, relRMS {relb:.2e}"
        );
        assert!(cos1 > 0.9999 && worst < 5e-3, "experts disagree with the oracle: cosine {cos1}, worst row {worst}");

        // (5) the whole REXL3 layer: its router, its experts, the oracle's shared expert
        let mut sc = MoeScratch::new(&d, 64).unwrap();
        let sh = up(&shared1);
        let mut out = DevBuf::zeroed(n * 4096 * 4).unwrap();
        layer.forward(&x, Dtype::F16, n, &g.tab, Some(&sh), &mut out, &mut sc, &g.st).unwrap();
        let mine: Vec<f32> = sync_dl(&out, &g.st);
        let (cosf, relf) = cos_rel_rms(&mine, &out1);
        eprintln!("layer {l}: full REXL3 layer (own router) vs oracle: cosine {cosf:.7}, relRMS {relf:.2e}");
        assert!(cosf > 0.999, "full layer disagrees with the oracle: cosine {cosf}");
    }
}

// ====================================================================================================================
// T_verify(R): the MoE part of a verify window of R rows, with REAL routing (tools/capture_moe_inputs.py: the post-norm hidden
// states of the whole model on real prompts, prompt and decoded rows), for the expert-union term of the CostModel.

fn capture_dir() -> Option<std::path::PathBuf> {
    let p = std::env::var("REXL3_MOE_CAPTURE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("moe-capture"));
    p.join("seq_bounds.npy").exists().then_some(p)
}

/// The router of layer `l` straight from the pack (fp16 gate, centred fp16 bias).
fn load_router(ck: &Checkpoint, l: usize) -> MoeRouter {
    let base = format!("model.language_model.layers.{l}.mlp");
    let gate = ck.read_u16(ck.plain.get(&format!("{base}.gate.weight")).unwrap()).unwrap();
    let esb: Vec<f32> = {
        let e = ck.plain.get(&format!("{base}.gate.e_score_correction_bias")).unwrap();
        ck.read_tensor(e).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };
    MoeRouter::new(&gate, &centred_bias_host(&esb), 4096, 8, 2.5).unwrap()
}

#[derive(Default, Clone)]
struct Acc {
    n: f64,
    u: f64,
    router: f64,
    gu: f64,
    d: f64,
    other: f64,
}

#[test]
#[ignore]
fn bench_t_verify() {
    gpu!();
    let Some(cap) = capture_dir() else {
        eprintln!("no capture (tools/capture_moe_inputs.py, REXL3_MOE_CAPTURE): not run");
        return;
    };
    let g = gpu_ctx();
    let ck = checkpoint().expect("the GLM pack");
    let bounds: Vec<usize> = read_npy(&cap.join("seq_bounds.npy")).to_i64().iter().map(|&v| v as usize).collect();
    let n_rows = *bounds.last().unwrap();
    let kinds: Vec<u8> = read_npy(&cap.join("kinds.npy")).data;
    assert_eq!(kinds.len(), n_rows);
    // decode rows: global row -> their ordinal (the last MoE layer was only captured on decode steps)
    let mut dec_pos = vec![usize::MAX; n_rows];
    let mut n_dec = 0;
    for (i, &k) in kinds.iter().enumerate() {
        if k == 1 {
            dec_pos[i] = n_dec;
            n_dec += 1;
        }
    }
    let layers: Vec<usize> = (3..=44).collect();
    let bits_of = |l: usize| -> usize {
        ck.groups[&format!("model.language_model.layers.{l}.mlp.experts.0.gate_proj")].bits.twice() as usize / 2
    };
    let sur2 = real_layer!(6);
    let sur3 = real_layer!(3);
    let d = MoeDims::GLM53;
    let windows_per_r: usize = std::env::var("REXL3_BENCH_WINDOWS").ok().and_then(|v| v.parse().ok()).unwrap_or(24);

    // per layer: the router, the captured rows, and REXL3's own picks for every captured row
    struct LayerData {
        l: usize,
        bits: usize,
        router: MoeRouter,
        x: Vec<u16>,
        pick: Vec<i32>,
        full: bool,
    }
    impl LayerData {
        /// the local row of global row `g` (decode rows only)
        fn local(&self, g: usize, dec_pos: &[usize]) -> usize {
            if self.full {
                g
            } else {
                dec_pos[g]
            }
        }
    }
    let mut data = Vec::new();
    let mut sc = MoeScratch::new(&d, 128).unwrap();
    for &l in &layers {
        let x = npy_u16(&cap.join(format!("L{l}.npy"))).1;
        let rows_l = x.len() / 4096;
        assert!(rows_l == n_rows || rows_l == n_dec, "layer {l}: {rows_l} rows");
        let router = load_router(ck, l);
        let mut pick = vec![0i32; rows_l * 8];
        let mut r0 = 0;
        while r0 < rows_l {
            let m = (rows_l - r0).min(128);
            let xd = up(&x[r0 * 4096..(r0 + m) * 4096]);
            router.logits(&xd, Dtype::F16, m, &mut sc.logits, &g.st).unwrap();
            router.select(&g.tab, &sc.logits, m, &mut sc.pick, &mut sc.wts, &g.st).unwrap();
            let p: Vec<i32> = sync_dl(&sc.pick, &g.st);
            pick[r0 * 8..(r0 + m) * 8].copy_from_slice(&p[..m * 8]);
            r0 += m;
        }
        data.push(LayerData { l, bits: bits_of(l), router, x, pick, full: rows_l == n_rows });
    }
    let n3 = data.iter().filter(|x| x.bits == 3).count();
    // runs of consecutive decode rows of one sequence: the windows a verify step can hold
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for s in 0..bounds.len() - 1 {
        let (mut a, b) = (bounds[s], bounds[s + 1]);
        while a < b {
            if kinds[a] == 1 {
                let mut e = a;
                while e < b && kinds[e] == 1 {
                    e += 1;
                }
                runs.push((a, e));
                a = e;
            } else {
                a += 1;
            }
        }
    }
    eprintln!(
        "capture: {n_rows} rows ({n_dec} decode rows in {} runs), {} MoE layers ({} 2-bit, {} 3-bit)",
        runs.len(),
        data.len(),
        data.len() - n3,
        n3
    );

    // how correlated is real routing: the share of a row's experts the previous decode row also picked, against uniform 8/288
    let (mut shared, mut pairs) = (0usize, 0usize);
    for ld in &data {
        for &(a, b) in &runs {
            for r in a + 1..b {
                let (cur, prev) = (ld.local(r, &dec_pos), ld.local(r - 1, &dec_pos));
                let pv = &ld.pick[prev * 8..(prev + 1) * 8];
                shared += ld.pick[cur * 8..(cur + 1) * 8].iter().filter(|e| pv.contains(e)).count();
                pairs += 8;
            }
        }
    }
    println!(
        "ROUTING adjacent decode rows share {:.1}% of their top-8 experts (uniform routing would share {:.1}%)",
        100.0 * shared as f64 / pairs as f64,
        100.0 * 8.0 / 288.0
    );

    let t = EventTimer::new(10).unwrap();
    let flush = DevBuf::zeroed(256 << 20).unwrap();
    let mut flush_out = DevBuf::zeroed(16).unwrap();
    let shared_buf = DevBuf::zeroed(128 * 4096 * 4).unwrap();
    let mut out = DevBuf::zeroed(128 * 4096 * 4).unwrap();
    let mut results = Vec::new();
    for rows in 1..=9usize {
        let mut starts: Vec<usize> = Vec::new();
        for &(a, b) in &runs {
            if b - a >= rows {
                starts.extend((a..=b - rows).step_by(3));
            }
        }
        let pick_every = (starts.len() / windows_per_r).max(1);
        let starts: Vec<usize> = starts.into_iter().step_by(pick_every).take(windows_per_r).collect();
        let mut by_class = [Acc::default(), Acc::default()]; // 2-bit, 3-bit: sums over layers of each layer's mean over windows
        let mut layer_means: Vec<(usize, Acc)> = Vec::new();
        for ld in &data {
            let (ex, class) = if ld.bits == 3 { (&sur3.experts, 1) } else { (&sur2.experts, 0) };
            let mut acc = Acc::default();
            for &g0 in &starts {
                let st0 = ld.local(g0, &dec_pos);
                let xd = up(&ld.x[st0 * 4096..(st0 + rows) * 4096]);
                read_probe(&flush, 256 << 20, &mut flush_out, &g.st).unwrap();
                t.record(0, &g.st).unwrap();
                ld.router.logits(&xd, Dtype::F16, rows, &mut sc.logits, &g.st).unwrap();
                t.record(1, &g.st).unwrap();
                ld.router.select(&g.tab, &sc.logits, rows, &mut sc.pick, &mut sc.wts, &g.st).unwrap();
                t.record(2, &g.st).unwrap();
                let mut n = 2;
                ex.run_marked(
                    &xd,
                    Dtype::F16,
                    rows,
                    Some(&shared_buf),
                    &mut out,
                    &mut sc,
                    false,
                    (ex.plan_gu, ex.plan_d),
                    &g.st,
                    &mut |_| {
                        n += 1;
                        t.record(n, &g.st)
                    },
                )
                .unwrap();
                let ms: Vec<f64> = (0..8).map(|i| t.ms(i, i + 1).unwrap() as f64).collect();
                // router logits, select, group, rot_in, gate/up, epilogue, down, combine
                acc.n += 1.0;
                acc.u += distinct(&ld.pick[st0 * 8..(st0 + rows) * 8]) as f64;
                acc.router += ms[0] + ms[1];
                acc.gu += ms[4];
                acc.d += ms[6];
                acc.other += ms[2] + ms[3] + ms[5] + ms[7];
            }
            let m = |v: f64| v / acc.n;
            let mean = Acc { n: 1.0, u: m(acc.u), router: m(acc.router), gu: m(acc.gu), d: m(acc.d), other: m(acc.other) };
            let c = &mut by_class[class];
            c.n += 1.0;
            c.u += mean.u;
            c.router += mean.router;
            c.gu += mean.gu;
            c.d += mean.d;
            c.other += mean.other;
            layer_means.push((ld.l, mean));
        }
        let total_ms: f64 = layer_means.iter().map(|(_, a)| a.router + a.gu + a.d + a.other).sum();
        let experts_ms: f64 = layer_means.iter().map(|(_, a)| a.gu + a.d).sum();
        let uniform_u = 288.0 * (1.0 - (1.0 - 8.0 / 288.0f64).powi(rows as i32));
        let cm = |c: &Acc| (c.u / c.n, (c.router + c.gu + c.d + c.other) / c.n, (c.gu + c.d) / c.n);
        let (u2, t2, e2) = cm(&by_class[0]);
        let (u3, t3, e3) = cm(&by_class[1]);
        println!(
            "TVERIFY R={rows} windows={} | real routing: U 2-bit {u2:5.1} 3-bit {u3:5.1} (uniform {uniform_u:5.1}) | per layer 2-bit {t2:.3} ms (experts {e2:.3}) 3-bit {t3:.3} ms (experts {e3:.3}) | all {} MoE layers: {total_ms:.2} ms (expert GEMVs {experts_ms:.2} ms)",
            starts.len(),
            data.len()
        );
        results.push((rows, u2, u3, t2, t3, total_ms));
    }
    println!("TVERIFY_TABLE rows, ms (router + experts + combine of all {} MoE layers, real routing, cold L2):", data.len());
    for (r, _, _, _, _, ms) in &results {
        println!("  ({r}, {ms:.2}),");
    }
}

/// Where the time of one MoE layer goes, stage by stage (the router and the small kernels around the two GEMVs), with and without
/// programmatic dependent launch; and the layers back to back (no synchronization between calls, three layers in turn so L2 stays cold).
#[test]
#[ignore]
fn bench_stages() {
    gpu!();
    let g = gpu_ctx();
    let layer = real_layer!(6);
    let layers = [real_layer!(6), real_layer!(3), real_layer!(45)];
    let d = MoeDims::GLM53;
    let flush = DevBuf::zeroed(256 << 20).unwrap();
    let mut flush_out = DevBuf::zeroed(16).unwrap();
    let shared = DevBuf::zeroed(128 * 4096 * 4).unwrap();
    let mut out = DevBuf::zeroed(128 * 4096 * 4).unwrap();
    let mut r = Rng(77);
    for pdl in [false] {
        for rows in [1usize, 2, 4, 8, 16] {
            let mut s = MoeScratch::new(&d, rows).unwrap();
            let mut acc: Vec<(&'static str, f64)> = Vec::new();
            let n = 40;
            for it in 0..n + 3 {
                let x = up(&rand_f16(&mut r, rows * 4096, 1.0));
                read_probe(&flush, 256 << 20, &mut flush_out, &g.st).unwrap();
                let v = layer.forward_profiled(&x, Dtype::F16, rows, &g.tab, Some(&shared), &mut out, &mut s, &g.st).unwrap();
                if it >= 3 {
                    if acc.is_empty() {
                        acc = v.iter().map(|&(n, _)| (n, 0.0)).collect();
                    }
                    for (a, b) in acc.iter_mut().zip(&v) {
                        a.1 += b.1 as f64 * 1e3 / n as f64;
                    }
                }
            }
            let total: f64 = acc.iter().map(|a| a.1).sum();
            let line: Vec<String> = acc.iter().map(|(n, us)| format!("{n} {us:.1}")).collect();
            println!("STAGES pdl={pdl} R={rows:2} layer 6 (us): {} | total {total:.1}", line.join(", "));
            // back to back: 60 calls, layers 6, 3, 45 in turn, each on a fresh window; per-call GPU time of the stream
            let xs: Vec<DevBuf> = (0..60).map(|_| up(&rand_f16(&mut r, rows * 4096, 1.0))).collect();
            let t = EventTimer::new(2).unwrap();
            for round in 0..2 {
                t.record(0, &g.st).unwrap();
                for (i, x) in xs.iter().enumerate() {
                    layers[i % 3].forward(x, Dtype::F16, rows, &g.tab, Some(&shared), &mut out, &mut s, &g.st).unwrap();
                }
                t.record(1, &g.st).unwrap();
                if round == 1 {
                    println!(
                        "BACKTOBACK pdl={pdl} R={rows:2}: {:.1} us per layer call (layers 6, 3, 45 in turn)",
                        t.ms(0, 1).unwrap() as f64 * 1e3 / 60.0
                    );
                }
            }
        }
    }
}

// ====================================================================================================================
// the shared expert (dense gated MLP on the EXL3 decode linears) and the golden hashes

fn shared_groups(l: usize) -> [LoadedGroup; 3] {
    let ck = checkpoint().unwrap();
    ["gate_proj", "up_proj", "down_proj"]
        .map(|p| ck.load_group(&format!("model.language_model.layers.{l}.mlp.shared_experts.{p}")).unwrap())
}

/// float64 reference of the shared expert for one row.
fn shared_reference(gs: &[LoadedGroup; 3], x: &[f64], limit: f64) -> (Vec<f64>, usize) {
    let lin = |g: &LoadedGroup, v: &[f64]| {
        gb_exl3::forward(v, 1, &g.trellis, g.kt, g.nt, g.group.bits, &codebook(g.group.codebook), &g.suh, &g.svh, None)
    };
    let (gv, uv) = (lin(&gs[0], x), lin(&gs[1], x));
    let mut clamped = 0;
    let a: Vec<f64> = gv
        .iter()
        .zip(&uv)
        .map(|(&g, &u)| {
            let silu = g / (1.0 + (-g).exp());
            clamped += (silu > limit || u.abs() > limit) as usize;
            silu.min(limit) * u.clamp(-limit, limit)
        })
        .map(|v| hf(h(v))) // the activation is handed to the down linear as fp16
        .collect();
    (lin(&gs[2], &a), clamped)
}

#[test]
fn shared_expert_matches_float64_and_is_row_invariant() {
    gpu!();
    let g = gpu_ctx();
    let ck = checkpoint();
    if ck.is_none() {
        eprintln!("no GLM pack: not run");
        return;
    }
    for l in [6usize, 3] {
        let sh = SharedExpert::from_checkpoint(ck.unwrap(), l, 10.0).unwrap();
        let gs = shared_groups(l);
        let mut sc = SharedScratch::new(&sh, 128).unwrap();
        let mut r = Rng(140 + l as u64);
        let pool = 128;
        let xh = rand_f16(&mut r, pool * 4096, 2.0);
        let run = |idx: &[usize], sc: &mut SharedScratch| -> Vec<f32> {
            let mut x = Vec::new();
            for &i in idx {
                x.extend_from_slice(&xh[i * 4096..(i + 1) * 4096]);
            }
            let xd = up(&x);
            let mut out = DevBuf::zeroed(idx.len() * 4096 * 4).unwrap();
            sh.forward(&xd, Dtype::F16, idx.len(), &mut out, sc, &g.st).unwrap();
            sync_dl(&out, &g.st)
        };
        // against float64 on a few rows, at the usual input scale and at one that pushes gate and up past the limit of 10
        let mut worst = 0f64;
        let mut clamped_total = 0;
        for (sigma, xs) in [(2.0, xh.clone()), (40.0, rand_f16(&mut r, 4 * 4096, 40.0))] {
            let mut x4 = Vec::new();
            for i in 0..4 {
                x4.extend_from_slice(&xs[i * 4096..(i + 1) * 4096]);
            }
            let xd = up(&x4);
            let mut out = DevBuf::zeroed(4 * 4096 * 4).unwrap();
            sh.forward(&xd, Dtype::F16, 4, &mut out, &mut sc, &g.st).unwrap();
            let got: Vec<f32> = sync_dl(&out, &g.st);
            for i in 0..4 {
                let x: Vec<f64> = xs[i * 4096..(i + 1) * 4096].iter().map(|&b| hf(b)).collect();
                let (want, clamped) = shared_reference(&gs, &x, 10.0);
                if sigma > 10.0 {
                    clamped_total += clamped;
                }
                worst = worst.max(worst_rel_rms(&got[i * 4096..(i + 1) * 4096], &want, 1, 4096));
            }
        }
        eprintln!(
            "shared expert layer {l} ({} bit gate/up): worst relRMS vs float64 {worst:.2e}, the clamps changed {clamped_total} activations",
            gs[0].group.bits.twice() / 2
        );
        assert!(clamped_total > 0, "the shared expert test must exercise the clamps");
        assert!(worst < 2e-3, "layer {l}: {worst:.3e}");
        // row invariance
        let solo: Vec<Vec<f32>> = (0..pool).map(|i| run(&[i], &mut sc)).collect();
        for m in [2usize, 3, 9, 16, 17, 33, 64, 128] {
            let mut idx: Vec<usize> = (0..pool).collect();
            shuffle(&mut r, &mut idx);
            idx.truncate(m);
            let out = run(&idx, &mut sc);
            for (pos, &i) in idx.iter().enumerate() {
                assert!(
                    out[pos * 4096..(pos + 1) * 4096] == solo[i][..],
                    "shared expert: row {i} at {pos} of {m} differs from its 1-row result"
                );
            }
        }
    }
}

/// The whole layer with REXL3's own shared expert, router and experts against the oracle's module output.
#[test]
fn oracle_full_layer_with_own_shared_expert() {
    gpu!();
    let Some(root) = ref_root() else {
        eprintln!("no oracle dump: not run");
        return;
    };
    let g = gpu_ctx();
    let d = MoeDims::GLM53;
    for l in [6usize, 3] {
        if !root.join(format!("L{l}/manifest.json")).exists() {
            continue;
        }
        let layer = real_layer!(l);
        let dir = root.join(format!("L{l}"));
        let (xs, xh) = npy_u16(&dir.join("x.npy"));
        let n = xs[0];
        let out1 = read_npy(&dir.join("out1.npy")).to_f32();
        let shared1 = read_npy(&dir.join("shared1.npy")).to_f32();
        let sh = SharedExpert::from_checkpoint(checkpoint().unwrap(), l, d.limit).unwrap();
        let mut ss = SharedScratch::new(&sh, 64).unwrap();
        let x = up(&xh);
        let mut shout = DevBuf::zeroed(n * 4096 * 4).unwrap();
        sh.forward(&x, Dtype::F16, n, &mut shout, &mut ss, &g.st).unwrap();
        let mine_sh: Vec<f32> = sync_dl(&shout, &g.st);
        let (cos_s, rel_s) = cos_rel_rms(&mine_sh, &shared1);
        eprintln!("layer {l}: REXL3 shared expert vs the oracle's: cosine {cos_s:.7}, relRMS {rel_s:.2e}");
        assert!(cos_s > 0.9999 && rel_s < 5e-3);
        let mut sc = MoeScratch::new(&d, 64).unwrap();
        let mut out = DevBuf::zeroed(n * 4096 * 4).unwrap();
        layer.forward(&x, Dtype::F16, n, &g.tab, Some(&shout), &mut out, &mut sc, &g.st).unwrap();
        let mine: Vec<f32> = sync_dl(&out, &g.st);
        let (cos, rel) = cos_rel_rms(&mine, &out1);
        eprintln!("layer {l}: whole REXL3 MoE layer (router, experts, shared expert) vs the oracle's module: cosine {cos:.7}, relRMS {rel:.2e}");
        assert!(cos > 0.9999 && rel < 5e-3);
    }
}

/// Golden hashes (`REXL3_GOLDEN_OUT=testdata/gpu/golden_moe.txt`, as the other goldens; `REXL3_MOE_GOLDEN_OUT` also works): bits of the router and the expert pipeline on synthetic layers
/// of every width and on real layers; `scripts/gpu_gate.sh` diffs the file. A change of kernel arithmetic (a K split, a sum order, the
/// activation, the combine order) changes lines; a change that must not (a launch shape, a fusion) does not.
#[test]
#[ignore]
fn golden_moe_hashes() {
    gpu!();
    let path = std::env::var("REXL3_GOLDEN_OUT").or_else(|_| std::env::var("REXL3_MOE_GOLDEN_OUT")).expect("REXL3_GOLDEN_OUT");
    let g = gpu_ctx();
    let mut lines: Vec<String> = Vec::new();
    // router: synthetic gate of GLM shape, the same rows at several window sizes
    let (router, _gate, bias) = synth_router(1);
    let mut r = Rng(2);
    let xh = rand_f16(&mut r, 128 * 4096, 1.0);
    lines.push(format!("sigmoid table {:016x}", hash_f32(&sigmoid_table_host())));
    lines.push(format!("bias {:016x}", hash_f32(&bias)));
    for rows in [1usize, 9, 128] {
        let x = up(&xh[..rows * 4096]);
        let run = route_rows(&router, &g.tab, &x, Dtype::F16, rows, &g.st);
        lines.push(format!(
            "router rows={rows} logits {:016x} picks {:016x} wts {:016x}",
            hash_u16(&run.logits),
            hash_i32(&run.pick),
            hash_u16(&run.wts)
        ));
    }
    // synthetic experts: every width class and codebook, windows of 1, 8, 17 and 33 rows
    let dims = small_dims();
    let cases: [(usize, usize, Codebook); 8] = [
        (1, 8, Codebook::Mul1),
        (2, 2, Codebook::Mul1),
        (3, 3, Codebook::Mul1),
        (4, 2, Codebook::Mul1),
        (6, 5, Codebook::Mul1),
        (8, 1, Codebook::Mul1),
        (4, 4, Codebook::Mcg),
        (3, 2, Codebook::Inst3),
    ];
    for (i, &(bg, bd, cb)) in cases.iter().enumerate() {
        let s = synth_layer(dims, bg, bd, cb, 200 + i as u64);
        let pool = make_pool(dims, 64, 2.0, 210 + i as u64);
        let mut sc = MoeScratch::new(&dims, 64).unwrap();
        for rows in [1usize, 8, 17, 33] {
            let idx: Vec<usize> = (0..rows).collect();
            let run = run_window(&s.ex, &g, &mut sc, &pool, &idx);
            lines.push(format!(
                "experts synthetic gu={bg} d={bd} {} rows={rows} out {:016x} y {:016x}",
                cb.name(),
                hash_f32(&run.out),
                hash_f32(&run.y)
            ));
        }
    }
    // real layers and their shared experts
    assert!(
        checkpoint().is_some(),
        "golden_moe_hashes needs the GLM pack (set REXL3_GLM_DIR): the committed file has the real-layer lines"
    );
    {
        let d = MoeDims::GLM53;
        for (l, _) in LAYERS {
            let layer = real_layer(l).unwrap();
            let pool = make_pool(d, 64, 1.5, 300 + l as u64);
            let mut sc = MoeScratch::new(&d, 64).unwrap();
            for rows in [1usize, 8, 16, 33] {
                let idx: Vec<usize> = (0..rows).collect();
                let run = run_window(&layer.experts, &g, &mut sc, &pool, &idx);
                lines.push(format!("experts layer {l} rows={rows} out {:016x} y {:016x}", hash_f32(&run.out), hash_f32(&run.y)));
            }
            if let Ok(sh) = SharedExpert::from_checkpoint(checkpoint().unwrap(), l, d.limit) {
                let mut ss = SharedScratch::new(&sh, 64).unwrap();
                for rows in [1usize, 8, 33] {
                    let x = up(&pool.xh[..rows * 4096]);
                    let mut out = DevBuf::zeroed(rows * 4096 * 4).unwrap();
                    sh.forward(&x, Dtype::F16, rows, &mut out, &mut ss, &g.st).unwrap();
                    lines.push(format!("shared layer {l} rows={rows} out {:016x}", hash_f32(&sync_dl::<f32>(&out, &g.st))));
                }
            }
        }
    }
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    eprintln!("wrote {} golden lines to {path}", lines.len());
}

/// The router GEMV's blocking (rows a warp, warps a block, loads in flight): every shape gives the same bits, one is fastest.
#[test]
#[ignore]
fn bench_router_shapes() {
    gpu!();
    let g = gpu_ctx();
    let layer = real_layer!(6);
    let router = &layer.router;
    let flush = DevBuf::zeroed(256 << 20).unwrap();
    let mut flush_out = DevBuf::zeroed(16).unwrap();
    let mut r = Rng(88);
    let shapes: [(usize, usize); 8] = [(4, 1), (8, 1), (4, 4), (8, 4), (4, 8), (8, 8), (2, 8), (2, 4)];
    for rows in [1usize, 2, 4, 8, 16, 32, 64, 128] {
        let xh = rand_f16(&mut r, rows * 4096, 1.0);
        let x = up(&xh);
        let mut reference: Option<Vec<u16>> = None;
        let mut best = (f64::MAX, (0, 0, 0));
        let mut line = Vec::new();
        for rb in [1usize, 2, 4] {
            for &(warps, un) in &shapes {
                let mut logits = DevBuf::zeroed(rows * 288 * 2).unwrap();
                let t = EventTimer::new(2).unwrap();
                let mut us = 0f64;
                let n = 20;
                for it in 0..n + 2 {
                    read_probe(&flush, 256 << 20, &mut flush_out, &g.st).unwrap();
                    t.record(0, &g.st).unwrap();
                    router.logits_blocked(&x, Dtype::F16, rows, &mut logits, rb, warps, un, &g.st).unwrap();
                    t.record(1, &g.st).unwrap();
                    if it >= 2 {
                        us += t.ms(0, 1).unwrap() as f64 * 1e3 / n as f64;
                    }
                }
                let got: Vec<u16> = sync_dl(&logits, &g.st);
                match &reference {
                    None => reference = Some(got),
                    Some(want) => assert!(&got == want, "rows {rows}: shape rb={rb} warps={warps} un={un} changes the logits"),
                }
                if us < best.0 {
                    best = (us, (rb, warps, un));
                }
                line.push(format!("rb{rb}w{warps}u{un} {us:.1}"));
            }
        }
        println!("ROUTERSHAPES R={rows:3}: best {:.1} us at {:?} | {}", best.0, best.1, line.join(" "));
    }
}

/// The router GEMV's blocking (rows a warp, warps a block, loads in flight a lane) changes the speed and never a bit.
#[test]
fn router_blocking_shapes_give_the_same_bits() {
    gpu!();
    let st = Stream::new().unwrap();
    let (router, _, _) = synth_router(21);
    let mut r = Rng(22);
    for rows in [1usize, 5, 40] {
        let x = up(&rand_f16(&mut r, rows * 4096, 1.0));
        let mut want = DevBuf::zeroed(rows * 288 * 2).unwrap();
        router.logits(&x, Dtype::F16, rows, &mut want, &st).unwrap();
        let want: Vec<u16> = sync_dl(&want, &st);
        for rb in [1usize, 2, 4] {
            for (warps, un) in [(2usize, 4usize), (4, 1), (8, 1), (4, 8), (8, 8)] {
                let mut got = DevBuf::zeroed(rows * 288 * 2).unwrap();
                router.logits_blocked(&x, Dtype::F16, rows, &mut got, rb, warps, un, &st).unwrap();
                assert!(sync_dl::<u16>(&got, &st) == want, "rows {rows}: rb {rb} warps {warps} unroll {un} changes the logits");
            }
        }
        // the prefill tiling (several experts a warp)
        for (rb, eb, warps, un) in eb_shapes() {
            let mut got = DevBuf::zeroed(rows * 288 * 2).unwrap();
            router.logits_blocked_eb(&x, Dtype::F16, rows, &mut got, rb, eb, warps, un, &st).unwrap();
            assert!(
                sync_dl::<u16>(&got, &st) == want,
                "rows {rows}: rb {rb} eb {eb} warps {warps} unroll {un} changes the logits"
            );
        }
        // the prefill tiling with 16-byte loads (a chain's 32 lanes on 8)
        for (rb, eb, warps) in v4_shapes() {
            let mut got = DevBuf::zeroed(rows * 288 * 2).unwrap();
            router.logits_blocked_v4(&x, Dtype::F16, rows, &mut got, rb, eb, warps, &st).unwrap();
            assert!(sync_dl::<u16>(&got, &st) == want, "rows {rows}: v4 rb {rb} eb {eb} warps {warps} changes the logits");
        }
    }
}

/// Every blocking the 16-byte prefill tiling (`router_logits_v4_kernel`: rows and experts an 8-lane group, warps a block) is built for.
fn v4_shapes() -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for (rb, eb) in [(2usize, 2usize), (2, 4), (2, 8), (4, 2), (4, 4), (4, 8), (8, 2), (8, 4)] {
        for warps in [2usize, 4, 8] {
            v.push((rb, eb, warps));
        }
    }
    v
}

/// Every blocking the prefill tiling (`router_logits_eb_kernel`: rows a warp, experts a warp, warps a block, loads in flight) is built for.
fn eb_shapes() -> Vec<(usize, usize, usize, usize)> {
    let mut v = Vec::new();
    for rb in [2usize, 4, 8] {
        for eb in [4usize, 8] {
            for warps in [4usize, 8] {
                for un in [1usize, 2] {
                    v.push((rb, eb, warps, un));
                }
            }
        }
    }
    v
}

/// The launcher's default blocking (four loads in flight a lane under 32 rows; from 32 the prefill tiling, four rows and four experts a warp) and every
/// blocking shape the kernel is built for, the 8- and
/// 16-deep ones included, give the bits of the host mirror for the hidden sizes and expert counts of other MoE families than GLM's: hidden sizes whose pair
/// count is not a multiple of 32 (a lane's last trip carries fewer than its loads in flight) or under 32 (some lanes load nothing), expert counts that are not
/// a multiple of the block's warps, the 512 the router takes at most; activations of every dtype; windows on both sides of the 32-row switch.
/// Nothing here depends on GLM's 4096 x 288.
#[test]
fn router_gemv_matches_the_host_mirror_for_other_hidden_sizes_and_expert_counts() {
    if !device_available() {
        eprintln!("no CUDA device: not run");
        return;
    }
    let st = Stream::new().unwrap();
    let mut r = Rng(31);
    // every (warps, loads in flight) the kernel is built for: the three that have always been there, the new ones
    let shapes: [(usize, usize); 7] = [(1, 8), (1, 16), (2, 16), (4, 16), (8, 16), (2, 8), (4, 4)];
    let more: [(usize, usize); 6] = [(2, 4), (4, 1), (8, 1), (4, 8), (8, 4), (8, 8)];
    for (hidden, experts) in
        [(2048usize, 512usize), (3072, 384), (2560, 256), (4096, 288), (2050, 33), (130, 9), (66, 8), (2, 8), (8192, 10)]
    {
        let gate = rand_f16(&mut r, experts * hidden, 0.03);
        let router = MoeRouter::new(&gate, &vec![0.0f32; experts], hidden, 1, 1.0).unwrap();
        for dt in [Dtype::F16, Dtype::BF16, Dtype::F32] {
            for rows in [1usize, 3, 8, 31, 32, 70, 133] {
                let xh = rand_f16(&mut r, rows * hidden, 1.0);
                let (vals, bytes): (Vec<f32>, Vec<u8>) = match dt {
                    Dtype::F16 => (xh.iter().map(|&b| hf(b) as f32).collect(), xh.iter().flat_map(|v| v.to_le_bytes()).collect()),
                    Dtype::BF16 => {
                        // the top half of the fp32 bits: any bf16 value is a legal input
                        let b: Vec<u16> = xh.iter().map(|&v| ((hf(v) as f32).to_bits() >> 16) as u16).collect();
                        (
                            b.iter().map(|&v| f32::from_bits((v as u32) << 16)).collect(),
                            b.iter().flat_map(|v| v.to_le_bytes()).collect(),
                        )
                    }
                    Dtype::F32 => {
                        let v: Vec<f32> = xh.iter().map(|&b| hf(b) as f32 * 1.000_123).collect();
                        (v.clone(), v.iter().flat_map(|x| x.to_le_bytes()).collect())
                    }
                };
                let x = up(&bytes);
                let want = router_logits_host(&vals, &gate, rows, hidden, experts);
                let mut got = DevBuf::zeroed(rows * experts * 2).unwrap();
                router.logits(&x, dt, rows, &mut got, &st).unwrap();
                assert!(
                    sync_dl::<u16>(&got, &st) == want,
                    "default launch: hidden {hidden} experts {experts} {dt:?} rows {rows}"
                );
                for rb in [1usize, 2, 4] {
                    for &(warps, un) in shapes.iter().chain(more.iter()) {
                        let mut got = DevBuf::zeroed(rows * experts * 2).unwrap();
                        router.logits_blocked(&x, dt, rows, &mut got, rb, warps, un, &st).unwrap();
                        assert!(
                            sync_dl::<u16>(&got, &st) == want,
                            "hidden {hidden} experts {experts} {dt:?} rows {rows}: rb {rb} warps {warps} unroll {un}"
                        );
                    }
                }
                // the prefill tiling: expert counts that are not a multiple of a block's experts, rows not a multiple of rb
                for (rb, eb, warps, un) in eb_shapes() {
                    let mut got = DevBuf::zeroed(rows * experts * 2).unwrap();
                    router.logits_blocked_eb(&x, dt, rows, &mut got, rb, eb, warps, un, &st).unwrap();
                    assert!(
                        sync_dl::<u16>(&got, &st) == want,
                        "hidden {hidden} experts {experts} {dt:?} rows {rows}: rb {rb} eb {eb} warps {warps} unroll {un}"
                    );
                }
                // the 16-byte prefill tiling: fp16 activations and hidden sizes in whole 64-element steps (refused otherwise)
                for (rb, eb, warps) in v4_shapes() {
                    let mut got = DevBuf::zeroed(rows * experts * 2).unwrap();
                    let r4 = router.logits_blocked_v4(&x, dt, rows, &mut got, rb, eb, warps, &st);
                    if dt != Dtype::F16 || hidden % 64 != 0 {
                        assert!(r4.is_err(), "hidden {hidden} {dt:?}: the 16-byte tiling must refuse");
                        continue;
                    }
                    r4.unwrap();
                    assert!(
                        sync_dl::<u16>(&got, &st) == want,
                        "hidden {hidden} experts {experts} {dt:?} rows {rows}: v4 rb {rb} eb {eb} warps {warps}"
                    );
                }
            }
        }
    }
}

/// The expert pipeline back to back, the way a decode step issues it: 24 calls with different uniformly random picks queued without a
/// synchronization in between (launch gaps and the small kernels included, no event between kernels), GB/s of the distinct expert
/// bytes each call reads. The honest throughput of the whole pipeline (router excluded).
#[test]
#[ignore]
fn bench_back_to_back() {
    gpu!();
    let g = gpu_ctx();
    let d = MoeDims::GLM53;
    let ceiling = read_ceiling(&g);
    println!("measured read ceiling {ceiling:.1} GB/s");
    let n = 24usize;
    let flush = DevBuf::zeroed(256 << 20).unwrap();
    let mut flush_out = DevBuf::zeroed(16).unwrap();
    let mut r = Rng(1234);
    for (l, _) in LAYERS {
        let layer = real_layer!(l);
        let ex = &layer.experts;
        for rows in [1usize, 2, 4, 8, 16] {
            let mut scratches: Vec<MoeScratch> = (0..n).map(|_| MoeScratch::new(&d, rows).unwrap()).collect();
            let mut bytes = 0f64;
            for s in scratches.iter_mut() {
                let (pick, wts) = rand_picks(&mut r, rows, d.experts, d.topk);
                bytes += distinct(&pick) as f64 * ex.expert_bytes as f64;
                s.set_picks(&pick, &wts).unwrap();
            }
            let x = up(&rand_f16(&mut r, rows * d.hidden, 1.0));
            let shared = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
            let mut out = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
            let t = EventTimer::new(2).unwrap();
            let mut best = f64::MAX;
            for _ in 0..4 {
                read_probe(&flush, 256 << 20, &mut flush_out, &g.st).unwrap();
                t.record(0, &g.st).unwrap();
                for s in scratches.iter_mut() {
                    ex.run(&x, Dtype::F16, rows, Some(&shared), &mut out, s, false, &g.st).unwrap();
                }
                t.record(1, &g.st).unwrap();
                best = best.min(t.ms(0, 1).unwrap() as f64);
            }
            let gbs = bytes / 1e9 / (best / 1e3);
            println!(
                "BACK2BACK layer {l:2} {}bit R={rows:2}: {:.1} us per call, {gbs:5.1} GB/s of expert bytes ({:.1}% of the {ceiling:.0} GB/s read ceiling)",
                ex.bits_gu,
                best * 1e3 / n as f64,
                100.0 * gbs / ceiling
            );
        }
    }
}

/// Plan candidates head to head (K splits of gate/up and down per layer width), alternating rounds so a drifting clock or a busy host
/// hits every candidate alike: mean GEMV ms over 3 rounds of REXL3_BENCH_ITERS windows.
#[test]
#[ignore]
fn bench_plan_candidates() {
    gpu!();
    let g = gpu_ctx();
    let iters: usize = std::env::var("REXL3_BENCH_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
    for (l, cands) in [
        (6usize, vec![(4usize, 2usize), (2, 2), (2, 4), (4, 4), (1, 2)]),
        (3, vec![(4, 2), (4, 1), (2, 2), (2, 1), (4, 4), (8, 1)]),
    ] {
        let layer = real_layer!(l);
        let ex = &layer.experts;
        for rows in [1usize, 2, 3, 4, 6, 8, 16] {
            let mut acc = vec![(0f64, 0f64); cands.len()];
            for round in 0..3 {
                for (i, &(gu, d)) in cands.iter().enumerate() {
                    let b = bench_experts(ex, &g, rows, iters, 4000 + round as u64 * 17, ((gu, 4), (d, 4)));
                    acc[i].0 += b.gu_ms / 3.0;
                    acc[i].1 += b.d_ms / 3.0;
                }
            }
            let line: Vec<String> = cands
                .iter()
                .zip(&acc)
                .map(|(&(gu, d), a)| format!("gu{gu}/d{d} {:.3}+{:.3}={:.3}", a.0, a.1, a.0 + a.1))
                .collect();
            println!("CAND layer {l} {}bit R={rows:2}: {}", ex.bits_gu, line.join(" | "));
        }
    }
}

/// Lane E: the routed MoE chain (router logits, selection, then group, rot_in, the two GEMVs, the epilogue and the combine) launched with the
/// programmatic-dependent-launch attribute (mask 4) and captured into a CUDA graph on a non-blocking stream, then replayed: the picks, weights,
/// output and every pair's y are bit-equal to the same chain launched eagerly without the attribute (mask 0).
#[test]
fn pdl_graph_of_the_routed_moe_chain_equals_the_eager_chain_bit_for_bit() {
    use gb_cuda::graph::Graph;
    gpu!();
    let st = Stream::new_nonblocking().unwrap();
    let tab = SigmoidTable::new().unwrap();
    let (router, _, _) = synth_router(61);
    let dims = MoeDims { hidden: 512, inter: 256, experts: 16, topk: 4, scale: 2.5, limit: 10.0 };
    let syn = synth_layer(dims, 4, 4, Codebook::Mul1, 65);
    let ex = &syn.ex;
    let mut r = Rng(62);
    for rows in [1usize, 3, 9] {
        let rx = up(&rand_f16(&mut r, rows * 4096, 1.0));
        let (pick, wts) = rand_picks(&mut r, rows, dims.experts, dims.topk);
        let x = up(&rand_f16(&mut r, rows * dims.hidden, 1.0));
        let shared = DevBuf::zeroed(rows * dims.hidden * 4).unwrap();
        let go = |graph: bool| -> (Vec<i32>, Vec<u16>, Vec<f32>, Vec<f32>) {
            let mut logits = DevBuf::zeroed(rows * 288 * 2).unwrap();
            let mut rpick = DevBuf::zeroed(rows * 8 * 4).unwrap();
            let mut rwts = DevBuf::zeroed(rows * 8 * 2).unwrap();
            let mut s = MoeScratch::new(&dims, rows).unwrap();
            s.set_picks(&pick, &wts).unwrap();
            let mut out = DevBuf::zeroed(rows * dims.hidden * 4).unwrap();
            // `st` is non-blocking: it does not wait for the legacy-stream uploads and memsets above
            gb_cuda::rt::device_sync().unwrap();
            let mut chain = || -> Result<(), gb_cuda::CudaError> {
                router.logits(&rx, Dtype::F16, rows, &mut logits, &st)?;
                router.select(&tab, &logits, rows, &mut rpick, &mut rwts, &st)?;
                ex.run(&x, Dtype::F16, rows, Some(&shared), &mut out, &mut s, true, &st)
            };
            if graph {
                let g = Graph::capture(&st, &mut chain).unwrap_or_else(|_| panic!("the MoE chain did not capture"));
                g.launch(&st).unwrap();
            } else {
                chain().unwrap();
            }
            st.sync().unwrap();
            let y: Vec<f32> = sync_dl(&s.y_pairs, &st);
            (sync_dl(&rpick, &st), sync_dl(&rwts, &st), sync_dl(&out, &st), y[..rows * dims.topk * dims.hidden].to_vec())
        };
        gb_cuda::exl3_linear::set_pdl_mask(0).unwrap();
        let eager = go(false);
        gb_cuda::exl3_linear::set_pdl_mask(4).unwrap();
        let eager_pdl = go(false);
        let graphed = go(true);
        gb_cuda::exl3_linear::set_pdl_mask(0).unwrap();
        let bits = |a: &[f32], b: &[f32]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        for (name, other) in [("PDL launches", &eager_pdl), ("the PDL graph", &graphed)] {
            assert!(eager.0 == other.0 && eager.1 == other.1, "{rows} rows: {name}: the router's picks or weights differ");
            assert!(bits(&eager.2, &other.2), "{rows} rows: {name}: the output differs");
            assert!(bits(&eager.3, &other.3), "{rows} rows: {name}: the pair outputs differ");
        }
    }
}
