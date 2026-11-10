//! FrostMoE decode at the Flash-Next pack's real layer shape: hidden 2560, expert intermediate 640, 512 routed experts + the shared
//! expert (id 512), top-10, gate/up 3 bits, down 3/4/5 bits, shared-expert widths 4..8. Two jobs:
//!
//! * `fmoe_decode_bits_match_the_pinned_digests`: `fmoe::forward` over many seeded routings (random, correlated rows that share most of
//!   their experts, identical rows, tie-heavy logits) at 1..=8 and 16 rows for every width mix of the pack, one FNV digest of the
//!   outputs per (mix, rows). The digests were recorded on the engine before the decode-MoE rework and pin
//!   every bit of `out`: a faster kernel that moves one is rejected.
//! * `bench_decode_moe`: microseconds a layer at 1..=7 rows against the unique experts' bytes (the 237 GB/s floor of the GB10).
//!
//!     OWNER=<lane> LOCK_ID=<id> <recipe>/rexl3/node.sh -p A hold
//!     cargo test --release -p gb-cuda --test flash_next_moe -- --ignored --nocapture <name>
mod common;
const H: usize = 2560;
const I: usize = 640;
const NE: usize = 512;
const TK: usize = 10;
const KC: usize = fmoe::DEFAULT_KC;
const SEED_STRIDE: usize = 1000;
fn bits_of(matrix: usize, _expert: usize) -> usize {
    if matrix < 2 { 3 } else { 4 }
}
include!("fn_moe/cases.rs");

/// (down bits of the routed experts, shared expert's gate / up / down bits): the pack's 3/4/5-bit down layers against shared widths 4..8.
const MIXES: [(usize, [usize; 3]); 5] = [(3, [4, 4, 4]), (4, [5, 6, 7]), (5, [8, 8, 8]), (4, [6, 5, 8]), (3, [7, 8, 6])];

fn layer(down: usize, shared: [usize; 3]) -> Owned {
    Owned::with_bits(move |m, e| {
        if e == NE {
            shared[m]
        } else if m < 2 {
            3
        } else {
            down
        }
    })
}

/// Row inputs of one routing pattern: 0 uniform logits, 1 rows that share a base logit vector (few unique experts), 2 identical rows,
/// 3 every logit on a coarse grid (ties everywhere), 4 one hot row repeated with tiny noise.
fn pattern(r: &mut Rng, rows: usize, mode: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let x: Vec<u16> = (0..rows * H).map(|_| bf(r.uniform(-0.2, 0.2))).collect();
    let base: Vec<f32> = (0..NE).map(|_| r.uniform(-2., 2.)).collect();
    let mut l = Vec::with_capacity(rows * NE);
    for t in 0..rows {
        for e in 0..NE {
            let v = match mode {
                0 => r.uniform(-2., 2.),
                1 => base[e] + r.uniform(-0.6, 0.6),
                2 => base[e],
                3 => (r.uniform(-2., 2.) * 4.0).round() / 4.0,
                _ => base[e] + if t == 0 { 0.0 } else { r.uniform(-0.01, 0.01) },
            };
            l.push(bf(v));
        }
    }
    let g: Vec<u16> = (0..rows).map(|_| bf(r.uniform(-1., 1.))).collect();
    (x, l, g)
}

fn digest(o: &Owned, rows: usize, seeds: usize, st: &Stream) -> u64 {
    let mut ws = fmoe::workspace(64, TK, H, I, KC).unwrap();
    let mut h = 0xcbf29ce484222325u64;
    let mut r = Rng(0xf1a5 + rows as u64);
    for s in 0..seeds {
        let (x, l, g) = pattern(&mut r, rows, s % 5);
        let (x, l, g) = (DevBuf::from_slice(&x).unwrap(), DevBuf::from_slice(&l).unwrap(), DevBuf::from_slice(&g).unwrap());
        let mut out = DevBuf::zeroed(rows * H * 2).unwrap();
        let shape = fmoe::Shape { t: rows, ne: NE, top_k: TK, h: H, i: I, kc: KC, max_tokens: 64 };
        fmoe::forward(&x, &l, &g, &o.tables(), &mut out, shape, None, fmoe::SwigluLimit::default(), &mut ws, st).unwrap();
        let v: Vec<u16> = dl(&out, st);
        h = (h ^ fnv(&v)).wrapping_mul(0x100000001b3);
    }
    h
}

/// Digests of `fmoe::forward` per (mix, rows), recorded before the decode-MoE rework.
const GOLDEN: &[u64] = &[
    0xbc6ff13dafbacbd8, 0xd321b08230f276d7, 0xaeac419e5d35237c,
    0x59e8bddc1a0cea7d, 0x98a2bf62fd84e99a, 0x0ecff837f97e15f1,
    0x424bba7eae591d11, 0x49f82bc529c3b8a0, 0x5e650d25cec59692,
    0x778b48ac153e4982, 0xdb6667ba7b13f9a7, 0xb491df93c1fe5c0d,
    0xb98a09eb0da20128, 0x8e31ea79660efb71, 0x240e1747c2312bed,
    0xff7228451bbabffc, 0x7b1593921aec28a6, 0x764ed6efddb6b5a3,
    0xfb7eaa4865454c78, 0x598bea2a3973c86f, 0x2ca74f0aaf945560,
    0x3f1109e9b8a9a4f0, 0x8cf6cbf9f25904eb, 0x535a30012883ded8,
    0x66e99c3836ed78e8, 0xcd4275e592207787, 0x7465defc21648d24,
    0xeb101733b9c2fde6, 0xcd8955b0323b13a8, 0x8bbfdc7e46cb6a82,
    0x54c1c014fd52cc1c, 0xa9b9ac4fa980aad3, 0x4d234be68c6120e5,
    0xd3c3c9bd3ba6c743, 0x7052ca45ec385fc8, 0x0400752c6a5cc783,
    0xb073e3af0f4b8ffc, 0xd42c150d6090cd71, 0xa2604b193fd748ad,
    0xe561c0894177004a, 0x22fb79dcb921b226, 0x6b5c0b1ed2603e20,
    0x3d37572b77a4a455, 0x0f55a1324e873154, 0x6314c34d5d198b34,
];

#[test]
#[ignore]
fn fmoe_decode_bits_match_the_pinned_digests() {
    let Some(st) = gpu() else { return };
    let mut got = Vec::new();
    for (mi, (down, shared)) in MIXES.iter().enumerate() {
        let o = layer(*down, *shared);
        for rows in [1usize, 2, 3, 4, 5, 6, 7, 8, 16] {
            let h = digest(&o, rows, 20, &st);
            eprintln!("mix {mi} down {down} shared {shared:?} rows {rows:>2}: {h:016x}");
            got.push(h);
        }
    }
    if GOLDEN.is_empty() {
        eprintln!("GOLDEN = &[{}]", got.iter().map(|h| format!("0x{h:016x}")).collect::<Vec<_>>().join(", "));
        return;
    }
    assert_eq!(got, GOLDEN, "decode MoE output differs from the pinned digests");
}

fn ms_per_call(st: &Stream, iters: usize, mut f: impl FnMut(usize)) -> f64 {
    for i in 0..8 {
        f(i);
    }
    st.sync().unwrap();
    let t0 = std::time::Instant::now();
    for i in 0..iters {
        f(i);
    }
    st.sync().unwrap();
    t0.elapsed().as_secs_f64() * 1e3 / iters as f64
}

/// Host router: the top-`TK` of every row by (value DESC, index ASC); the unique experts of the window and the bytes they stream
/// (gate + up at 3 bits, down at `down` bits, plus the shared expert at `shared` bits, which every window reads).
fn window_bytes(l: &[u16], rows: usize, down: usize, shared: [usize; 3]) -> (usize, usize) {
    let mut seen = vec![false; NE];
    for t in 0..rows {
        let mut idx: Vec<usize> = (0..NE).collect();
        idx.sort_by(|&a, &b| bff(l[t * NE + b]).partial_cmp(&bff(l[t * NE + a])).unwrap().then(a.cmp(&b)));
        for &e in &idx[..TK] {
            seen[e] = true;
        }
    }
    let u = seen.iter().filter(|s| **s).count();
    let mat = |bits: usize| H * I * bits / 8;
    (u, u * (2 * mat(3) + mat(down)) + shared.iter().map(|&b| mat(b)).sum::<usize>())
}

/// Layer time (best of three passes of 256 calls) at 1..=7 rows against the bytes the unique experts stream, weights from DRAM (a pool of 64 distinct row sets cycles
/// through the layer's experts, far beyond the L2). Rows share a base logit vector (pattern 1) like an MTP verify window's.
#[test]
#[ignore]
fn bench_decode_moe() {
    let Some(st) = gpu() else { return };
    for (down, shared) in [(3usize, [6usize, 6, 6]), (4, [6, 6, 6]), (5, [6, 6, 6])] {
        let o = layer(down, shared);
        eprintln!("down {down} bits, shared {shared:?}");
        for rows in 1..=7usize {
            let mut r = Rng(0xbe9c + rows as u64);
            let (mut sets, mut us, mut bytes) = (Vec::new(), 0usize, 0usize);
            for k in 0..64 {
                let (x, l, g) = pattern(&mut r, rows, if k % 2 == 0 { 1 } else { 0 });
                let (u, b) = window_bytes(&l, rows, down, shared);
                us += u;
                bytes += b;
                sets.push((DevBuf::from_slice(&x).unwrap(), DevBuf::from_slice(&l).unwrap(), DevBuf::from_slice(&g).unwrap()));
            }
            let mut ws = fmoe::workspace(64, TK, H, I, KC).unwrap();
            let mut out = DevBuf::zeroed(rows * H * 2).unwrap();
            let shape = fmoe::Shape { t: rows, ne: NE, top_k: TK, h: H, i: I, kc: KC, max_tokens: 64 };
            let ms = (0..3)
                .map(|_| {
                    ms_per_call(&st, 256, |i| {
                        let (x, l, g) = &sets[i % 64];
                        fmoe::forward(x, l, g, &o.tables(), &mut out, shape, None, fmoe::SwigluLimit::default(), &mut ws, &st).unwrap();
                    })
                })
                .fold(f64::MAX, f64::min);
            let (u, b) = (us as f64 / 64.0, bytes as f64 / 64.0);
            eprintln!(
                "rows {rows}: {:6.1} us a layer, unique experts {u:5.1}, {:5.1} MB, {:5.1} GB/s ({:4.0}% of 237)",
                ms * 1e3,
                b / 1e6,
                b / (ms * 1e-3) / 1e9,
                100.0 * b / (ms * 1e-3) / 237e9
            );
        }
    }
}
