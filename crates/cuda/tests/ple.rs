//! PLE kernels (`gb_cuda::ple`) on synthetic data: the n-gram table row decode bit for bit against a CPU decoder, the
//! projection / gate / conv stage against a CPU reference, row invariance (a row's result does not depend on the window
//! or chunk it runs in), and the golden hashes `scripts/gpu_gate.sh` compares:
//!
//!     REXL3_GOLDEN_OUT=golden_ple.txt cargo test --release -p gb-cuda --test ple golden_ple_hashes -- --ignored
use gb_cuda::ple::{self, ROW};
use gb_cuda::{DevBuf, Stream};

const D: usize = 256; // stream width (the model's 2560, scaled down; the kernels take it as a parameter)
const HC: usize = 4;
const HEADS: usize = 16;
const PD: usize = HEADS * ROW; // n-gram embedding width
const K: usize = 5;
const RING: usize = 128;
const EPS: f32 = 1e-6;

fn gpu() -> Option<Stream> {
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    gb_cuda::rt::init_context().ok()?;
    Stream::new().ok()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next() >> 29) as f32 / (1u64 << 24) as f32)
    }
}

/// f32 -> f16 bits, round to nearest even (normal and subnormal range; inputs here are small).
fn f16(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let a = f32::from_bits(b & 0x7fff_ffff);
    if a >= 65520.0 {
        return s | 0x7c00;
    }
    if a < 6.103_515_6e-5 {
        // subnormal: multiples of 2^-24
        let m = (a / 5.960_464_5e-8).round_ties_even() as u16;
        return s | m;
    }
    let e = ((a.to_bits() >> 23) as i32) - 127;
    let m = a.to_bits() & 0x7f_ffff;
    let mut h = (((e + 15) as u32) << 10) | (m >> 13);
    let rest = m & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    s | h as u16
}
fn h2f(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 31) as u32, (h & 1023) as u32);
    let v = if e == 0 {
        m as f32 * 2f32.powi(-24)
    } else if e == 31 {
        f32::INFINITY
    } else {
        f32::from_bits(((e + 112) << 23) | (m << 13))
    };
    if s == 1 {
        -v
    } else {
        v
    }
}
fn rh(x: f32) -> f32 {
    h2f(f16(x))
}
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}
fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain data viewed as bytes
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// CPU decode of one packed row: the 16-bit state ending at ring bit (i + 1) K - 1, mul1 codebook, * scale + bias.
fn decode_row(words: &[u16], bias: &[u16]) -> Vec<u16> {
    let bit = |b: usize| (words[1 + b / 16] >> (b % 16)) & 1;
    let scale = h2f(words[0]);
    let (kinv, kbias) = (h2f(0x1eee), h2f(0xc931));
    (0..ROW)
        .map(|i| {
            let mut state = 0u32;
            for m in 0..16 {
                let pos = (i + ROW - m / K) % ROW;
                state |= (bit(pos * K + m % K) as u32) << m;
            }
            let p = state.wrapping_mul(0x83DC_D12D);
            let hs = 1024.0 + ((p & 255) + ((p >> 8) & 255) + ((p >> 16) & 255) + (p >> 24)) as f32;
            let cb = rh(hs * kinv + kbias);
            f16(cb * scale + h2f(bias[i]))
        })
        .collect()
}

struct Stage {
    bias: Vec<u16>,
    key_w: Vec<u16>,
    val_w: Vec<u16>,
    nk: Vec<f32>,
    nq: Vec<f32>,
    nc: Vec<f32>,
    conv_w: Vec<u16>,
    dev: [DevBuf; 7],
}

impl Stage {
    fn synthetic(seed: u64) -> Self {
        let mut r = Rng(seed);
        let mut h = |n: usize, a: f32| -> Vec<u16> { (0..n).map(|_| f16(r.uniform(-a, a))).collect() };
        let bias = h(HEADS * ROW, 0.05);
        let key_w = h(HC * D * PD, 0.08);
        let val_w = h(D * PD, 0.08);
        let conv_w = h(HC * D * 4, 0.5);
        let mut r = Rng(seed ^ 0x55);
        let mut w = |n: usize| -> Vec<f32> { (0..n).map(|_| 1.0 + r.uniform(-0.3, 0.3)).collect() };
        let (nk, nq, nc) = (w(HC * D), w(HC * D), w(HC * D));
        let dev = [
            DevBuf::from_slice(&bias).unwrap(),
            DevBuf::from_slice(&key_w).unwrap(),
            DevBuf::from_slice(&val_w).unwrap(),
            DevBuf::from_slice(&nk).unwrap(),
            DevBuf::from_slice(&nq).unwrap(),
            DevBuf::from_slice(&nc).unwrap(),
            DevBuf::from_slice(&conv_w).unwrap(),
        ];
        Stage { bias, key_w, val_w, nk, nq, nc, conv_w, dev }
    }
}

/// Packed rows (16 heads a token) and input streams for `n` tokens.
fn inputs(n: usize, seed: u64) -> (Vec<u16>, Vec<f32>) {
    let mut r = Rng(seed);
    let words = ple::words_per_row(K);
    let packed: Vec<u16> = (0..n * HEADS * words)
        .map(|i| if i % words == 0 { f16(r.uniform(0.01, 0.05)) } else { (r.next() & 0xffff) as u16 })
        .collect();
    let streams: Vec<f32> = (0..n * HC * D).map(|_| r.uniform(-2.0, 2.0)).collect();
    (packed, streams)
}

/// Run the whole stage over `n` tokens of one sequence (slot 0, positions from 0) in prefill chunks or decode windows of `chunk`
/// rows; returns
/// the updated streams and the embeddings (f16 bits).
fn run(s: &Stage, packed: &[u16], streams: &[f32], chunk: usize, prefill: bool, st: &Stream) -> (Vec<f32>, Vec<u16>) {
    let n = streams.len() / (HC * D);
    let (ch, words) = (HC * D, ple::words_per_row(K));
    let mut ring = DevBuf::zeroed(RING * ch * 2).unwrap();
    let (mut out, mut emb_all) = (streams.to_vec(), vec![0u16; n * PD]);
    let [bias, key_w, val_w, nk, nq, nc, conv_w] = &s.dev;
    let mut at = 0;
    while at < n {
        let r = chunk.min(n - at);
        let pk = DevBuf::from_slice(&packed[at * HEADS * words..(at + r) * HEADS * words]).unwrap();
        let (mut emb, mut key, mut val) =
            (DevBuf::zeroed(r * PD * 2).unwrap(), DevBuf::zeroed(r * ch * 2).unwrap(), DevBuf::zeroed(r * D * 2).unwrap());
        let (mut sg, mut nb) = (DevBuf::zeroed(r * HC * 4).unwrap(), DevBuf::zeroed(r * ch * 2).unwrap());
        let mut xs = DevBuf::from_slice(&out[at * ch..(at + r) * ch]).unwrap();
        let pos: Vec<i32> = (at..at + r).map(|p| p as i32).collect();
        let (pos, ss, sl) = (
            DevBuf::from_slice(&pos).unwrap(),
            DevBuf::from_slice(&[0i32, r as i32]).unwrap(),
            DevBuf::from_slice(&[0i32]).unwrap(),
        );
        ple::rows(&pk, bias, &mut emb, r, HEADS, K, st).unwrap();
        let proj = if prefill { ple::proj_prefill } else { ple::proj_window };
        proj(&emb, key_w, &mut key, r, PD, ch, st).unwrap();
        proj(&emb, val_w, &mut val, r, PD, D, st).unwrap();
        let norms = ple::GateNorms { key: nk, query: nq, conv: nc };
        ple::gate(&xs, &key, &val, &norms, &mut sg, &mut nb, r, D, HC, EPS, 1.0 / (D as f32).sqrt(), st).unwrap();
        let br = ple::BatchRows { positions: &pos, seq_start: &ss, seq_slot: &sl, n_seq: 1, rows: r };
        ple::conv(&mut xs, &sg, &val, &nb, &ring, conv_w, &br, D, HC, 1, RING, st).unwrap();
        ple::ring_store(&nb, &mut ring, &br, ch, 1, RING, st).unwrap();
        st.sync().unwrap();
        xs.download(&mut out[at * ch..(at + r) * ch]).unwrap();
        emb.download(&mut emb_all[at * PD..(at + r) * PD]).unwrap();
        at += r;
    }
    (out, emb_all)
}

/// CPU reference of the stage from given embeddings (so the decode and the rest are checked separately).
fn reference(s: &Stage, emb: &[u16], streams: &[f32]) -> Vec<f32> {
    let n = streams.len() / (HC * D);
    let ch = HC * D;
    let mm = |x: &[f32], w: &[u16], rows_out: usize| -> Vec<f32> {
        (0..rows_out).map(|o| rh((0..PD).map(|i| x[i] * h2f(w[o * PD + i])).sum())).collect()
    };
    let rms = |x: &[f32], w: &[f32]| -> Vec<f32> {
        let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + EPS).sqrt();
        x.iter().zip(w).map(|(v, w)| v * inv * w).collect()
    };
    let mut nb_all = vec![0f32; n * ch];
    let mut gated = vec![0f32; n * ch];
    for t in 0..n {
        let e: Vec<f32> = emb[t * PD..(t + 1) * PD].iter().map(|&h| h2f(h)).collect();
        let (key, val) = (mm(&e, &s.key_w, ch), mm(&e, &s.val_w, D));
        for st in 0..HC {
            let k = rms(&key[st * D..(st + 1) * D], &s.nk[st * D..(st + 1) * D]);
            let q = rms(&streams[(t * HC + st) * D..(t * HC + st + 1) * D], &s.nq[st * D..(st + 1) * D]);
            let g = q.iter().zip(&k).map(|(a, b)| a * b).sum::<f32>() / (D as f32).sqrt();
            let ss = g.signum() * g.abs().max(1e-6).sqrt() * if g == 0.0 { 0.0 } else { 1.0 };
            let gate = 1.0 / (1.0 + (-ss).exp());
            let gv: Vec<f32> = val.iter().map(|v| gate * v).collect();
            let c = rms(&gv, &s.nc[st * D..(st + 1) * D]);
            for i in 0..D {
                gated[t * ch + st * D + i] = gv[i];
                nb_all[t * ch + st * D + i] = rh(c[i]);
            }
        }
    }
    let mut out = streams.to_vec();
    for t in 0..n {
        for c in 0..ch {
            let mut acc = 0f32;
            for j in 0..4 {
                let back = 3 * (3 - j);
                if t >= back {
                    acc += h2f(s.conv_w[c * 4 + j]) * nb_all[(t - back) * ch + c];
                }
            }
            let y = rh(acc);
            out[t * ch + c] += gated[t * ch + c] + rh(y / (1.0 + (-y).exp()));
        }
    }
    out
}

const N: usize = 300;

#[test]
#[ignore]
fn ple_matches_cpu_reference() {
    let Some(st) = gpu() else { return };
    let s = Stage::synthetic(11);
    let (packed, streams) = inputs(N, 12);
    let words = ple::words_per_row(K);
    // table rows: bit for bit
    let (out, emb) = run(&s, &packed, &streams, N, true, &st);
    for t in 0..N {
        for h in 0..HEADS {
            let want = decode_row(&packed[(t * HEADS + h) * words..(t * HEADS + h + 1) * words], &s.bias[h * ROW..(h + 1) * ROW]);
            assert_eq!(emb[(t * HEADS + h) * ROW..(t * HEADS + h + 1) * ROW], want[..], "row decode token {t} head {h}");
        }
    }
    // the rest, against the CPU reference from the same embeddings (prefill path and window path)
    let want = reference(&s, &emb, &streams);
    for (name, o) in
        [("one batch (tiled projections)", out), ("windows of 8 (warp projections)", run(&s, &packed, &streams, 8, false, &st).0)]
    {
        let (mut num, mut den) = (0f64, 0f64);
        for ((a, b), x) in o.iter().zip(&want).zip(&streams) {
            num += ((a - b) as f64).powi(2);
            den += ((b - x) as f64).powi(2);
        }
        let rel = (num / den).sqrt();
        eprintln!("{name}: delta relative L2 vs CPU {rel:e}");
        assert!(rel < 5e-3, "{name}: {rel}");
    }
}

#[test]
#[ignore]
fn ple_rows_are_invariant() {
    let Some(st) = gpu() else { return };
    let s = Stage::synthetic(11);
    let (packed, streams) = inputs(N, 12);
    // decode windows (<= 64 rows: warp projections) of any size give the same bits
    let w1 = run(&s, &packed, &streams, 1, false, &st).0;
    for c in [7usize, 37, 64] {
        assert!(run(&s, &packed, &streams, c, false, &st).0 == w1, "windows of {c} differ from windows of 1");
    }
    // prefill chunks (tiled projections) of different sizes give the same bits
    let p300 = run(&s, &packed, &streams, 300, true, &st).0;
    for c in [100usize, 30, 7] {
        // 300 = 3 x 100, 10 x 30, 42 x 7 + 6: chunks below the window size and a short last chunk included
        assert!(run(&s, &packed, &streams, c, true, &st).0 == p300, "prefill chunks of {c} differ from one batch");
    }
}

/// The golden text (hashes of the embeddings and the updated streams) of the PLE kernels on synthetic data.
fn golden_text(st: &Stream) -> String {
    let s = Stage::synthetic(11);
    let (packed, streams) = inputs(N, 12);
    let mut out = String::new();
    for (name, chunk, prefill) in [
        ("windows of 1", 1usize, false),
        ("windows of 37", 37, false),
        ("prefill 300", 300, true),
        ("prefill chunks of 30", 30, true),
    ] {
        let (o, emb) = run(&s, &packed, &streams, chunk, prefill, st);
        out += &format!("{name} embedding {:016x}\n", fnv(bytes_of(&emb)));
        out += &format!("{name} streams {:016x}\n", fnv(bytes_of(&o)));
    }
    out
}

/// Golden hashes of the PLE kernels on synthetic data.
///     REXL3_GOLDEN_OUT=golden_ple.txt cargo test --release -p gb-cuda --test ple golden_ple_hashes -- --ignored
#[test]
#[ignore]
fn golden_ple_hashes() {
    use std::io::Write;
    let Some(st) = gpu() else { return };
    let path = std::env::var("REXL3_GOLDEN_OUT").expect("REXL3_GOLDEN_OUT");
    std::fs::File::create(path).unwrap().write_all(golden_text(&st).as_bytes()).unwrap();
}

/// FNV-1a of `golden_text` as the kernels were before they carried the programmatic-launch attribute.
const PLE_GOLDEN_HASH: u64 = 0x3f56aca1be85046f;

/// The PLE stage (rows, the two projections, gate, conv, ring) launched with the programmatic-dependent-launch attribute (the Flash family
/// bit of the mask) gives the bits of plain launches, and the plain bits are the recorded ones.
#[test]
fn ple_kernels_are_exact_under_programmatic_launches() {
    let Some(st) = gpu() else { return };
    let mut texts = Vec::new();
    for mask in [0, 8] {
        gb_cuda::exl3_linear::set_pdl_mask(mask).unwrap();
        texts.push(golden_text(&st));
    }
    gb_cuda::exl3_linear::set_pdl_mask(0).unwrap();
    let hash = fnv(texts[0].as_bytes());
    println!("PLE golden hash of the plain launches: 0x{hash:016x}");
    assert_eq!(texts[0], texts[1], "programmatic launches changed a result");
    assert_eq!(hash, PLE_GOLDEN_HASH, "the plain launches differ from the recorded results:\n{}", texts[0]);
}

/// f16 bits of values of mixed magnitude (1e-4 .. 64, both signs, some exact zeros, a few subnormals and 65504s): products of any two
/// of them are exact in f32, which the tiled projection relies on.
fn mixed_f16(n: usize, seed: u64) -> Vec<u16> {
    let mut r = Rng(seed);
    (0..n)
        .map(|_| {
            let kind = r.next() % 64;
            let mag = 2f32.powf(r.uniform(-13.0, 6.0));
            let v = match kind {
                0 => 0.0,
                1 => -0.0,
                2 => 65504.0,
                3 => -65504.0,
                4 => 5.960_464_5e-8 * (1 + r.next() % 1023) as f32,
                _ => mag * r.uniform(-1.0, 1.0),
            };
            f16(v)
        })
        .collect()
}

type Proj = fn(&DevBuf, &DevBuf, &mut DevBuf, usize, usize, usize, &Stream) -> Result<(), gb_cuda::CudaError>;

fn project(
    f: Proj,
    x: &DevBuf,
    w: &DevBuf,
    rows: usize,
    k: usize,
    n: usize,
    st: &Stream,
) -> Vec<u16> {
    let mut y = DevBuf::zeroed(rows * n * 2).unwrap();
    f(x, w, &mut y, rows, k, n, st).unwrap();
    st.sync().unwrap();
    y.to_vec().unwrap()
}

/// The 128 x 128 tiled prefill projection gives the previous 64 x 64 kernel's bits on odd row counts, tail tiles (rows, columns and k not
/// multiples of the tiles or of 8), the model's widths and magnitudes spanning the f16 range.
#[test]
#[ignore]
fn ple_tiled_projection_equals_the_previous_kernel_bit_for_bit() {
    let Some(st) = gpu() else { return };
    let shapes: [(usize, usize, usize); 14] = [
        (1, 2560, 2560),
        (7, 2560, 129),
        (63, 2560, 2560),
        (64, 2560, 2560),
        (65, 2560, 100),
        (127, 640, 257),
        (129, 2560, 2560),
        (130, 17, 130),
        (200, 2563, 131),
        (255, 16, 128),
        (256, 33, 256),
        (1000, 2560, 2560),
        (4012, 2560, 10240),
        (4012, 2560, 2560),
    ];
    for (i, &(rows, k, n)) in shapes.iter().enumerate() {
        let x = DevBuf::from_slice(&mixed_f16(rows * k, 100 + i as u64)).unwrap();
        let w = DevBuf::from_slice(&mixed_f16(n * k, 200 + i as u64)).unwrap();
        let new = project(ple::proj_prefill, &x, &w, rows, k, n, &st);
        let old = project(ple::proj_prefill_ref, &x, &w, rows, k, n, &st);
        let bad = new.iter().zip(&old).filter(|(a, b)| a != b).count();
        assert_eq!(bad, 0, "{rows} rows, k {k}, n {n}: {bad} of {} outputs differ", new.len());
    }
}

/// The two prefill projections at the real shape (a 4,012-row chunk: 2,560 -> 10,240 key and 2,560 -> 2,560 value columns); prints the time
/// of each kernel.
///     cargo test --release -p gb-cuda --test ple -- --ignored --nocapture bench_ple_projection_chunk
#[test]
#[ignore]
fn bench_ple_projection_chunk() {
    let Some(st) = gpu() else { return };
    let (rows, k) = (4012usize, 2560usize);
    let x = DevBuf::from_slice(&mixed_f16(rows * k, 1)).unwrap();
    for n in [10240usize, 2560] {
        let w = DevBuf::from_slice(&mixed_f16(n * k, 2)).unwrap();
        let mut y = DevBuf::zeroed(rows * n * 2).unwrap();
        for (name, f) in [("previous", ple::proj_prefill_ref as Proj), ("tiled", ple::proj_prefill as Proj)] {
            let mut best = f64::MAX;
            for it in 0..6 {
                st.sync().unwrap();
                let t0 = std::time::Instant::now();
                f(&x, &w, &mut y, rows, k, n, &st).unwrap();
                st.sync().unwrap();
                if it > 0 {
                    best = best.min(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            eprintln!("BENCH n {n} {name}: {best:.3} ms");
        }
    }
}
