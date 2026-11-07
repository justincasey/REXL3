//! GLM-5.3-Flash MLA kernels (dense regime): latent append, absorb, attention over the latent, expand. f64 host
//! references with the kernels' rounding points, bit-exact row invariance (1..128 rows, window offsets, mixed slots,
//! padding rows, grid padding, block placement, stale data), the cache layout contract, golden hashes, and (ignored)
//! benchmarks. Every check runs for fp16 and bf16. GPU tests skip when there is no device. The comparison with the
//! ExLlamaV3 fork on real layer weights is in `mla_oracle.rs`. The fp8 latent page (`--kv-dtype fp8`, `csrc/mla_kv8.cuh`): the host
//! quantiser, the device write against it, every attention entry point against the T path on the dequantised cache, determinism.
#![cfg(not(gb_no_cuda))]

mod common;

use common::{cos_rel_rms, Rng};
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::mla::*;
use gb_cuda::num::{bf16_to_f32, f32_to_bf16};
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use std::sync::OnceLock;

pub const POISON: u32 = 0x3fff_fff0; // an unmapped table entry: reading it would fault, so a kernel that does fails loudly
const DTYPES: [Dtype; 2] = [Dtype::F16, Dtype::BF16];

fn gpu() -> Option<Stream> {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return None;
    }
    Some(Stream::new().unwrap())
}

// ---------------------------------------------------------------------------------------------------- numbers

#[derive(Clone, Copy)]
struct Fmt(Dtype);

impl Fmt {
    fn enc(self, x: f32) -> u16 {
        match self.0 {
            Dtype::BF16 => f32_to_bf16(x),
            _ => f64_to_f16(x as f64),
        }
    }
    fn dec(self, b: u16) -> f32 {
        match self.0 {
            Dtype::BF16 => bf16_to_f32(b),
            _ => f16_to_f64(b) as f32,
        }
    }
    fn decs(self, v: &[u16]) -> Vec<f32> {
        v.iter().map(|&b| self.dec(b)).collect()
    }
}

// ---------------------------------------------------------------------------------------------------- data

fn hash_seed(a: u64, b: u64) -> u64 {
    a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0x1000_0000_01B3) ^ 0x1234_5678
}

/// Deterministic latents of a sequence: `[len][512]` (a per-token amplitude keeps softmax peaked).
fn gen_latent(f: Fmt, sid: u64, len: usize) -> Vec<u16> {
    let mut r = Rng(hash_seed(sid, 1));
    let mut v = Vec::with_capacity(len * LATENT_DIM);
    for _ in 0..len {
        let amp = r.uniform(0.5, 2.0);
        v.extend((0..LATENT_DIM).map(|_| f.enc(r.uniform(-1.0, 1.0) * amp)));
    }
    v
}

/// The q row (64 heads x 256, nope) of sequence `sid` at `pos`.
fn qvec(f: Fmt, sid: u64, pos: usize) -> Vec<u16> {
    let mut r = Rng(hash_seed(0xABCD ^ sid, pos as u64));
    (0..N_HEADS * NOPE_DIM).map(|_| f.enc(r.uniform(-1.5, 1.5))).collect()
}

/// A synthetic `kv_b_proj.weight` in checkpoint layout `[64*512][512]` as bf16 bits (what the pack stores).
fn kv_b_host() -> &'static Vec<u16> {
    static W: OnceLock<Vec<u16>> = OnceLock::new();
    W.get_or_init(|| {
        let mut r = Rng(0x6B76_5F62);
        (0..N_HEADS * KV_B_ROWS_PER_HEAD * LATENT_DIM).map(|_| f32_to_bf16(r.uniform(-0.2, 0.2))).collect()
    })
}

/// The packed `(uk, uvt)` of the synthetic weights in `dst` precision.
fn packed(dst: Dtype) -> &'static (Vec<u16>, Vec<u16>) {
    static P: OnceLock<[(Vec<u16>, Vec<u16>); 2]> = OnceLock::new();
    let both =
        P.get_or_init(|| [pack_kv_b(kv_b_host(), Dtype::BF16, Dtype::F16), pack_kv_b(kv_b_host(), Dtype::BF16, Dtype::BF16)]);
    &both[(dst == Dtype::BF16) as usize]
}

fn weights(dt: Dtype) -> MlaWeights {
    MlaWeights::from_kv_b(kv_b_host(), Dtype::BF16, dt).unwrap()
}

struct Seq {
    sid: u64,
    len: usize,
    lat: Vec<u16>,
}

fn seqs_of(f: Fmt, lens: &[usize]) -> Vec<Seq> {
    lens.iter().enumerate().map(|(i, &len)| Seq { sid: i as u64 + 1, len, lat: gen_latent(f, i as u64 + 1, len) }).collect()
}

struct World {
    f: Fmt,
    cache: LatentCache,
    table: BlockTable,
    seqs: Vec<Seq>,
    w: MlaWeights,
    stream: Stream,
}

use gb_cuda::attention::{BlockTable, RowIndex};

fn shuffle<T>(v: &mut [T], r: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = (r.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

/// Fills the cache with NaN bits, places each sequence's pages at shuffled blocks (`placement` seeds the shuffle),
/// poisons the unused table entries and writes all latents through `latent_write` in one launch.
fn make_world(dt: Dtype, seqs: Vec<Seq>, placement: u64, spare_blocks: usize) -> World {
    make_world_kv(dt, seqs, placement, spare_blocks, LatentKv::Plain)
}

/// [`make_world`] over a cache of format `kv` (the fp8 page: `latent_write` quantises).
fn make_world_kv(dt: Dtype, seqs: Vec<Seq>, placement: u64, spare_blocks: usize, kv: LatentKv) -> World {
    let f = Fmt(dt);
    let stream = Stream::new().unwrap();
    let need: usize = seqs.iter().map(|s| s.len.div_ceil(BLOCK_TOKENS)).sum();
    let n_blocks = need + spare_blocks;
    let mut cache = LatentCache::new(n_blocks, dt, kv).unwrap();
    cache.lat.memset_async(0xFF, &stream).unwrap();
    let mut pool: Vec<u32> = (0..n_blocks as u32).collect();
    shuffle(&mut pool, &mut Rng(placement));
    let max_blocks = seqs.iter().map(|s| s.len.div_ceil(BLOCK_TOKENS)).max().unwrap() + 4;
    let mut tables = Vec::new();
    let mut next = 0;
    for s in &seqs {
        let nb = s.len.div_ceil(BLOCK_TOKENS);
        let mut t: Vec<u32> = pool[next..next + nb].to_vec();
        next += nb;
        t.resize(max_blocks, POISON);
        tables.push(t);
    }
    let table = BlockTable::from_host(&tables, max_blocks).unwrap();
    let (mut lats, mut slot, mut pos) = (Vec::new(), Vec::new(), Vec::new());
    for (si, s) in seqs.iter().enumerate() {
        lats.extend_from_slice(&s.lat);
        slot.extend((0..s.len).map(|_| si as i32));
        pos.extend(0..s.len as i32);
    }
    let ld = DevBuf::from_slice(&lats).unwrap();
    let ri = RowIndex::from_host(&slot, &pos).unwrap();
    latent_write(&ld, &ri, &table, &mut cache, slot.len(), &stream).unwrap();
    stream.sync().unwrap();
    World { f, cache, table, seqs, w: weights(dt), stream }
}

/// Stages of one pipeline run over `rows = [(seq index, position)]` (a negative position is a padding row): the absorbed
/// queries, the attended latents and the expanded output, all as 16-bit bits.
struct Stages {
    q_lat: Vec<u16>,
    o_lat: Vec<u16>,
    out: Vec<u16>,
}

fn run_pipeline(w: &World, rows: &[(usize, i32)]) -> Stages {
    run_pipeline_ctx(w, rows, None)
}

/// As `run_pipeline` with an explicit `max_context` (default: the longest row's context, padded to exercise a wider grid).
fn run_pipeline_ctx(w: &World, rows: &[(usize, i32)], max_ctx: Option<usize>) -> Stages {
    let n = rows.len();
    let mut q = Vec::new();
    for &(si, p) in rows {
        q.extend(qvec(w.f, w.seqs[si].sid, p.max(0) as usize));
    }
    let max_ctx = max_ctx.unwrap_or_else(|| {
        let m = rows.iter().map(|r| (r.1 + 1).max(1) as usize).max().unwrap();
        m + (m % 3) * 700
    });
    let max_ctx = max_ctx.min(w.table.max_blocks * BLOCK_TOKENS);
    let slot: Vec<i32> = rows.iter().map(|r| r.0 as i32).collect();
    let pos: Vec<i32> = rows.iter().map(|r| r.1).collect();
    let ri = RowIndex::from_host(&slot, &pos).unwrap();
    let qd = DevBuf::from_slice(&q).unwrap();
    let row = N_HEADS * LATENT_DIM * 2;
    let mut q_lat = DevBuf::from_slice(&vec![0xABCDu16; n * N_HEADS * LATENT_DIM]).unwrap();
    let mut o_lat = DevBuf::from_slice(&vec![0xABCDu16; n * N_HEADS * LATENT_DIM]).unwrap();
    let mut out = DevBuf::from_slice(&vec![0xABCDu16; n * N_HEADS * NOPE_DIM]).unwrap();
    // the workspace holds finite garbage (0x3F bytes = 0.75f): a kernel that reads a partial it did not write changes its row (NaN would
    // not do: the merge skips empty chunks by `l > 0`, which a NaN fails)
    let mut ws = DevBuf::alloc(attend_workspace_bytes(n, max_ctx)).unwrap();
    ws.memset_async(0x3F, &w.stream).unwrap();
    assert!(q_lat.bytes() >= n * row);
    absorb(&qd, N_HEADS * NOPE_DIM, &w.w, &mut q_lat, n, &w.stream).unwrap();
    attend(&q_lat, &w.cache, &w.table, &ri, n, &mut o_lat, max_ctx, &mut ws, &w.stream).unwrap();
    expand(&o_lat, &w.w, &mut out, n, &w.stream).unwrap();
    w.stream.sync().unwrap();
    Stages { q_lat: q_lat.to_vec().unwrap(), o_lat: o_lat.to_vec().unwrap(), out: out.to_vec().unwrap() }
}

const ROW_Q: usize = N_HEADS * LATENT_DIM;
const ROW_OUT: usize = N_HEADS * NOPE_DIM;

// ---------------------------------------------------------------------------------------------------- f64 references

/// `sum_d q[h][d] W_UK[h][d][c]` in f64 (unrounded) and the sum of the absolute terms (the scale of the rounding error
/// of any fp32 chain over the same terms), `[64 * 512]` each.
fn ref_absorb(q: &[u16], uk: &[u16], f: Fmt) -> Vec<(f64, f64)> {
    let mut out = vec![(0f64, 0f64); N_HEADS * LATENT_DIM];
    for h in 0..N_HEADS {
        let o = &mut out[h * LATENT_DIM..(h + 1) * LATENT_DIM];
        for d in 0..NOPE_DIM {
            let qd = f.dec(q[h * NOPE_DIM + d]) as f64;
            let w = &uk[(h * NOPE_DIM + d) * LATENT_DIM..][..LATENT_DIM];
            for (c, &wc) in w.iter().enumerate() {
                let t = qd * f.dec(wc) as f64;
                o[c].0 += t;
                o[c].1 += t.abs();
            }
        }
    }
    out
}

/// Dense attention of one row over `lat` (positions 0..=pos), heads `heads` only, in f64; `[heads.len() * 512]`.
fn ref_attend(q_lat: &[u16], lat: &[u16], pos: usize, heads: &[usize], f: Fmt) -> Vec<f64> {
    let all: Vec<usize> = (0..=pos).collect();
    ref_attend_list(q_lat, lat, &all, heads, f)
}

/// Attention of one row over the keys at `positions` of `lat` (any order, as the softmax is symmetric), heads `heads` only, in f64.
fn ref_attend_list(q_lat: &[u16], lat: &[u16], positions: &[usize], heads: &[usize], f: Fmt) -> Vec<f64> {
    let n = positions.len();
    let mut keys: Vec<f32> = Vec::with_capacity(n * LATENT_DIM);
    for &p in positions {
        keys.extend(f.decs(&lat[p * LATENT_DIM..(p + 1) * LATENT_DIM]));
    }
    let mut out = vec![0f64; heads.len() * LATENT_DIM];
    for (hi, &h) in heads.iter().enumerate() {
        let qh: Vec<f64> = q_lat[h * LATENT_DIM..(h + 1) * LATENT_DIM].iter().map(|&b| f.dec(b) as f64).collect();
        let mut sc = vec![0f64; n];
        for (j, scj) in sc.iter_mut().enumerate() {
            let kr = &keys[j * LATENT_DIM..(j + 1) * LATENT_DIM];
            *scj = qh.iter().zip(kr).map(|(a, &b)| a * b as f64).sum::<f64>() / 16.0;
        }
        let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut l = 0f64;
        let acc = &mut out[hi * LATENT_DIM..(hi + 1) * LATENT_DIM];
        for (j, &scj) in sc.iter().enumerate() {
            let p = (scj - mx).exp();
            l += p;
            let kr = &keys[j * LATENT_DIM..(j + 1) * LATENT_DIM];
            for (a, &x) in acc.iter_mut().zip(kr) {
                *a += p * x as f64;
            }
        }
        for a in acc.iter_mut() {
            *a /= l;
        }
    }
    out
}

/// `sum_c o[h][c] W_UV[h][c][n]` in f64 (unrounded) and the sum of the absolute terms, `[64 * 256]` each.
fn ref_expand(o_lat: &[u16], uvt: &[u16], f: Fmt) -> Vec<(f64, f64)> {
    let mut out = vec![(0f64, 0f64); N_HEADS * NOPE_DIM];
    for h in 0..N_HEADS {
        let o = &mut out[h * NOPE_DIM..(h + 1) * NOPE_DIM];
        for c in 0..LATENT_DIM {
            let oc = f.dec(o_lat[h * LATENT_DIM + c]) as f64;
            let w = &uvt[(h * LATENT_DIM + c) * NOPE_DIM..][..NOPE_DIM];
            for (n, &wn) in w.iter().enumerate() {
                let t = oc * f.dec(wn) as f64;
                o[n].0 += t;
                o[n].1 += t.abs();
            }
        }
    }
    out
}

fn assert_bits_eq(what: &str, got: &[u16], want: &[u16], width: usize) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!("{what}: bits differ at element {i} (head {}, col {}): {:#06x} vs {:#06x}", i / width, i % width, got[i], want[i]);
    }
}

/// Every element is the f64 sum rounded to T, up to one T rounding step plus the worst-case error of a `k`-term fp32 chain
/// (`k * 2^-24` of the sum of absolute terms): a cancellation that leaves a tiny result may differ in many ulps of that
/// result, but never by more than the chain's own rounding; a wrong index, stride or transposition is O(1) off.
fn assert_close_chain(what: &str, f: Fmt, got: &[u16], want: &[(f64, f64)], k: usize) {
    assert_eq!(got.len(), want.len());
    let eps_t = if f.0 == Dtype::BF16 { 2f64.powi(-8) } else { 2f64.powi(-11) };
    let mut worst = 0f64;
    for (i, (&g, &(w, abs))) in got.iter().zip(want).enumerate() {
        let gv = f.dec(g) as f64;
        let tol = eps_t * 1.01 * w.abs().max(gv.abs()) + k as f64 * 2f64.powi(-24) * abs;
        let err = (gv - w).abs();
        worst = worst.max(err / tol);
        assert!(err <= tol, "{what}: element {i}: gpu {g:#06x} ({gv}) vs f64 {w} (|terms| {abs}): error {err} > {tol}");
    }
    let _ = worst;
}

static WORST: std::sync::Mutex<[(f64, f64); 2]> = std::sync::Mutex::new([(1.0, 0.0); 2]);

/// Cosine and relative RMS of the attended latent against the f64 reference on `heads`.
fn check_attend(what: &str, w: &World, st: &Stages, r: usize, si: usize, pos: usize, heads: &[usize]) {
    let q_lat = &st.q_lat[r * ROW_Q..(r + 1) * ROW_Q];
    let want = ref_attend(q_lat, &w.seqs[si].lat, pos, heads, w.f);
    let mut got = Vec::new();
    for &h in heads {
        got.extend(w.f.decs(&st.o_lat[r * ROW_Q + h * LATENT_DIM..][..LATENT_DIM]));
    }
    let want32: Vec<f32> = want.iter().map(|&x| x as f32).collect();
    let (cos, rel) = cos_rel_rms(&got, &want32);
    {
        let mut ws = WORST.lock().unwrap();
        let i = (w.f.0 == Dtype::BF16) as usize;
        ws[i] = (ws[i].0.min(cos), ws[i].1.max(rel));
    }
    // bf16 rounds the probabilities fed to the PV product to 8 bits and fp16 to 11
    let (cmin, rmax) = if w.f.0 == Dtype::BF16 { (0.99995, 5e-3) } else { (0.999995, 8e-4) };
    assert!(cos >= cmin && rel <= rmax, "{what}: {:?} len {} pos {pos}: cos {cos} relRMS {rel}", w.f.0, w.seqs[si].len);
}

/// Tuning variants of the attention kernel (1..=VARIANTS besides the default 0): (tile keys, chunk keys, head tiles per CTA)
/// 1: 32/128/1, 2: 32/512/1, 3: 16/256/1, 4: 16/128/1, 5: 32/256/2, 6: 32/512/2, 7: 32/128/2, 8: 16/256/2.
const VARIANTS: i32 = 8;

const CONTEXTS: [usize; 11] = [1, 2, 255, 256, 257, 511, 512, 513, 1000, 2048, 2051];

// ---------------------------------------------------------------------------------------------------- layout

#[test]
fn latent_write_places_rows_by_block_and_token() {
    let Some(stream) = gpu() else { return };
    for dt in DTYPES {
        let f = Fmt(dt);
        let lens = [700usize, 256, 3];
        let w = make_world(dt, seqs_of(f, &lens), 5, 2);
        // every latent landed at its (block, token): read the whole cache back and compare through plane_offset
        let all: Vec<u16> = w.cache.lat.to_vec().unwrap();
        let tables: Vec<Vec<u32>> = {
            let flat: Vec<i32> = w.table.buf.to_vec().unwrap();
            (0..lens.len())
                .map(|s| flat[s * w.table.max_blocks..(s + 1) * w.table.max_blocks].iter().map(|&b| b as u32).collect())
                .collect()
        };
        let mut written = std::collections::HashSet::new();
        for (si, s) in w.seqs.iter().enumerate() {
            for p in 0..s.len {
                let blk = tables[si][p / BLOCK_TOKENS] as usize;
                let off = plane_offset(blk, p % BLOCK_TOKENS, BLOCK_TOKENS, LATENT_DIM);
                assert_eq!(&all[off..off + LATENT_DIM], &s.lat[p * LATENT_DIM..(p + 1) * LATENT_DIM], "seq {si} pos {p}");
                written.insert((blk, p % BLOCK_TOKENS));
            }
        }
        // nothing else was touched (still the NaN fill)
        for blk in 0..w.cache.n_blocks {
            for t in 0..BLOCK_TOKENS {
                if !written.contains(&(blk, t)) {
                    let off = plane_offset(blk, t, BLOCK_TOKENS, LATENT_DIM);
                    assert!(all[off..off + LATENT_DIM].iter().all(|&b| b == 0xFFFF), "block {blk} token {t} was written");
                }
            }
        }
        // a padding row (-1) writes nothing, and a row with a position equal to an existing one overwrites exactly that token
        let ri = RowIndex::from_host(&[0, 2], &[-1, 1]).unwrap();
        let junk = vec![0x1234u16; 2 * LATENT_DIM];
        let jd = DevBuf::from_slice(&junk).unwrap();
        let mut w = w;
        latent_write(&jd, &ri, &w.table, &mut w.cache, 2, &stream).unwrap();
        stream.sync().unwrap();
        let after: Vec<u16> = w.cache.lat.to_vec().unwrap();
        let blk2 = tables[2][0] as usize;
        let off = plane_offset(blk2, 1, BLOCK_TOKENS, LATENT_DIM);
        assert!(after[off..off + LATENT_DIM].iter().all(|&b| b == 0x1234));
        let diff = all.iter().zip(&after).filter(|(a, b)| a != b).count();
        assert_eq!(diff, LATENT_DIM, "only the one real row may change the cache");
    }
}

// ---------------------------------------------------------------------------------------------------- (a) accuracy

#[test]
fn absorb_matches_f64_reference() {
    let Some(stream) = gpu() else { return };
    for dt in DTYPES {
        let f = Fmt(dt);
        let wt = weights(dt);
        for (rows, stride) in [(1usize, N_HEADS * NOPE_DIM), (5, N_HEADS * NOPE_DIM), (17, N_HEADS * NOPE_DIM + 64)] {
            // q rows with a padded stride: the padding must never be read as data
            let mut q = vec![0xFFFFu16; (rows - 1) * stride + N_HEADS * NOPE_DIM];
            for r in 0..rows {
                q[r * stride..r * stride + N_HEADS * NOPE_DIM].copy_from_slice(&qvec(f, 9, r));
            }
            let qd = DevBuf::from_slice(&q).unwrap();
            let mut out = DevBuf::zeroed(rows * ROW_Q * 2).unwrap();
            absorb(&qd, stride, &wt, &mut out, rows, &stream).unwrap();
            stream.sync().unwrap();
            let got: Vec<u16> = out.to_vec().unwrap();
            for r in 0..rows {
                let want = ref_absorb(&qvec(f, 9, r), &packed(dt).0, f);
                assert_close_chain(
                    &format!("absorb {dt:?} rows {rows} row {r}"),
                    f,
                    &got[r * ROW_Q..(r + 1) * ROW_Q],
                    &want,
                    NOPE_DIM,
                );
            }
        }
    }
}

#[test]
fn expand_matches_f64_reference() {
    let Some(stream) = gpu() else { return };
    for dt in DTYPES {
        let f = Fmt(dt);
        let wt = weights(dt);
        for rows in [1usize, 3, 16, 20] {
            let mut r = Rng(77 + rows as u64);
            let o: Vec<u16> = (0..rows * ROW_Q).map(|_| f.enc(r.uniform(-1.0, 1.0))).collect();
            let od = DevBuf::from_slice(&o).unwrap();
            let mut out = DevBuf::zeroed(rows * ROW_OUT * 2).unwrap();
            expand(&od, &wt, &mut out, rows, &stream).unwrap();
            stream.sync().unwrap();
            let got: Vec<u16> = out.to_vec().unwrap();
            for r in 0..rows {
                let want = ref_expand(&o[r * ROW_Q..(r + 1) * ROW_Q], &packed(dt).1, f);
                assert_close_chain(
                    &format!("expand {dt:?} rows {rows} row {r}"),
                    f,
                    &got[r * ROW_OUT..(r + 1) * ROW_OUT],
                    &want,
                    LATENT_DIM,
                );
            }
        }
    }
}

#[test]
fn attend_matches_f64_reference() {
    if gpu().is_none() {
        return;
    }
    let heads_all: Vec<usize> = (0..N_HEADS).collect();
    for dt in DTYPES {
        let w = make_world(dt, seqs_of(Fmt(dt), &CONTEXTS), 7, 5);
        // the last position of every sequence, all in one launch (eleven slots)
        let rows: Vec<(usize, i32)> = CONTEXTS.iter().enumerate().map(|(i, &l)| (i, l as i32 - 1)).collect();
        let st = run_pipeline(&w, &rows);
        for (r, &(si, p)) in rows.iter().enumerate() {
            check_attend("last pos", &w, &st, r, si, p as usize, &heads_all);
        }
        // interior positions around the chunk / tile / page edges (a subset of heads: the reference is O(keys x 512) per head)
        let mut rows = Vec::new();
        for (si, &len) in CONTEXTS.iter().enumerate() {
            for p in [0usize, 15, 16, 31, 32, 33, 127, 128, 255, 256, 257, 511, 512, 1023, 1500, 2047] {
                if p < len {
                    rows.push((si, p as i32));
                }
            }
        }
        for part in rows.chunks(61) {
            let st = run_pipeline(&w, part);
            for (r, &(si, p)) in part.iter().enumerate() {
                check_attend("interior", &w, &st, r, si, p as usize, &[0, 13, 31, 32, 50, 63]);
            }
        }
        // expand of the attended latents against f64 on the same rows
        let st = run_pipeline(&w, &[(9, 2047), (4, 255)]);
        for r in 0..2 {
            let want = ref_expand(&st.o_lat[r * ROW_Q..(r + 1) * ROW_Q], &packed(dt).1, w.f);
            assert_close_chain("pipeline expand", w.f, &st.out[r * ROW_OUT..(r + 1) * ROW_OUT], &want, LATENT_DIM);
        }
    }
    let ws = WORST.lock().unwrap();
    eprintln!("attend vs f64: worst (cos, relRMS)  fp16 {:?}  bf16 {:?}", ws[0], ws[1]);
}

// ---------------------------------------------------------------------------------------------------- (b) row invariance

/// The whole pipeline: a row alone equals the same row in any window, batch, placement and grid.
#[test]
fn pipeline_row_invariance_bit_exact() {
    if gpu().is_none() {
        return;
    }
    for dt in DTYPES {
        let f = Fmt(dt);
        let lens = [2051usize, 1500, 700, 513];
        let w1 = make_world(dt, seqs_of(f, &lens), 1, 4);
        let w2 = make_world(dt, seqs_of(f, &lens), 99, 11); // different block placement, same content
        let alone = |w: &World, si: usize, p: usize| run_pipeline(w, &[(si, p as i32)]);
        let probes = [
            (0usize, 2050usize),
            (0, 2047),
            (0, 1500),
            (0, 511),
            (1, 1499),
            (1, 1000),
            (1, 600),
            (2, 699),
            (2, 0),
            (3, 512),
            (3, 3),
        ];
        let base: Vec<Stages> = probes.iter().map(|&(s, p)| alone(&w1, s, p)).collect();
        for (i, &(s, p)) in probes.iter().enumerate() {
            let a = alone(&w2, s, p);
            assert_bits_eq("placement q_lat", &a.q_lat, &base[i].q_lat, LATENT_DIM);
            assert_bits_eq("placement o_lat", &a.o_lat, &base[i].o_lat, LATENT_DIM);
            assert_bits_eq("placement out", &a.out, &base[i].out, NOPE_DIM);
        }
        // windows of 2..128 consecutive rows of one sequence, the probe at different offsets inside
        for &wn in &[2usize, 3, 4, 5, 8, 9, 16, 17, 31, 64, 100, 128] {
            for &(si, p) in &[(0usize, 2050usize), (0, 1999), (1, 1499), (0, 511), (2, 699)] {
                for off in [0usize, wn / 2, wn - 1] {
                    if p < off || p - off + wn > lens[si] {
                        continue;
                    }
                    let p0 = p - off;
                    let rows: Vec<(usize, i32)> = (0..wn).map(|r| (si, (p0 + r) as i32)).collect();
                    let want = alone(&w1, si, p);
                    for w in [&w1, &w2] {
                        let st = run_pipeline(w, &rows);
                        let tag = format!("{dt:?} window {wn} off {off} probe ({si}, {p})");
                        assert_bits_eq(
                            &format!("{tag} q_lat"),
                            &st.q_lat[off * ROW_Q..(off + 1) * ROW_Q],
                            &want.q_lat,
                            LATENT_DIM,
                        );
                        assert_bits_eq(
                            &format!("{tag} o_lat"),
                            &st.o_lat[off * ROW_Q..(off + 1) * ROW_Q],
                            &want.o_lat,
                            LATENT_DIM,
                        );
                        assert_bits_eq(&format!("{tag} out"), &st.out[off * ROW_OUT..(off + 1) * ROW_OUT], &want.out, NOPE_DIM);
                    }
                }
            }
        }
        // mixed batches: random rows of all sequences in random order
        let mut r = Rng(5);
        for trial in 0..14 {
            let n = [1usize, 2, 7, 8, 9, 24, 50, 128][trial % 8];
            let mut rows: Vec<(usize, i32)> = (0..n)
                .map(|_| {
                    let si = (r.next_u64() % lens.len() as u64) as usize;
                    (si, (r.next_u64() % lens[si] as u64) as i32)
                })
                .collect();
            if trial % 3 == 0 {
                rows.sort(); // contiguous runs per slot
            }
            let st = run_pipeline(&w1, &rows);
            for (i, &(si, p)) in rows.iter().enumerate() {
                let want = alone(&w1, si, p as usize);
                assert_bits_eq(
                    &format!("batch {trial} row {i} out"),
                    &st.out[i * ROW_OUT..(i + 1) * ROW_OUT],
                    &want.out,
                    NOPE_DIM,
                );
                assert_bits_eq(
                    &format!("batch {trial} row {i} o_lat"),
                    &st.o_lat[i * ROW_Q..(i + 1) * ROW_Q],
                    &want.o_lat,
                    LATENT_DIM,
                );
            }
        }
    }
}

/// Absorb and expand alone: a row's bits do not depend on the row count (all kernel variants: 1, <= 4 and > 4 rows, and
/// several passes of 16).
#[test]
fn absorb_and_expand_are_row_count_invariant() {
    let Some(stream) = gpu() else { return };
    for dt in DTYPES {
        let f = Fmt(dt);
        let wt = weights(dt);
        let n_max = MAX_ROWS;
        let mut q = Vec::new();
        for r in 0..n_max {
            q.extend(qvec(f, 3, r));
        }
        let mut rg = Rng(31);
        let o: Vec<u16> = (0..n_max * ROW_Q).map(|_| f.enc(rg.uniform(-1.0, 1.0))).collect();
        let (qd, od) = (DevBuf::from_slice(&q).unwrap(), DevBuf::from_slice(&o).unwrap());
        let run = |n: usize| -> (Vec<u16>, Vec<u16>) {
            let mut a = DevBuf::from_slice(&vec![0xABCDu16; n_max * ROW_Q]).unwrap();
            let mut e = DevBuf::from_slice(&vec![0xABCDu16; n_max * ROW_OUT]).unwrap();
            absorb(&qd, N_HEADS * NOPE_DIM, &wt, &mut a, n, &stream).unwrap();
            expand(&od, &wt, &mut e, n, &stream).unwrap();
            stream.sync().unwrap();
            (a.to_vec().unwrap(), e.to_vec().unwrap())
        };
        let (a_all, e_all) = run(n_max);
        for n in [1usize, 2, 3, 4, 5, 8, 15, 16, 17, 31, 32, 33, 64, 100, 127] {
            let (a, e) = run(n);
            assert_bits_eq(&format!("{dt:?} absorb n={n}"), &a[..n * ROW_Q], &a_all[..n * ROW_Q], LATENT_DIM);
            assert_bits_eq(&format!("{dt:?} expand n={n}"), &e[..n * ROW_OUT], &e_all[..n * ROW_OUT], NOPE_DIM);
        }
        // and single rows from the middle of the batch
        for r in [0usize, 1, 7, 16, 17, 99, 127] {
            let mut a = DevBuf::zeroed(ROW_Q * 2).unwrap();
            let mut e = DevBuf::zeroed(ROW_OUT * 2).unwrap();
            let q1 = DevBuf::from_slice(&q[r * ROW_OUT..(r + 1) * ROW_OUT]).unwrap();
            let o1 = DevBuf::from_slice(&o[r * ROW_Q..(r + 1) * ROW_Q]).unwrap();
            absorb(&q1, N_HEADS * NOPE_DIM, &wt, &mut a, 1, &stream).unwrap();
            expand(&o1, &wt, &mut e, 1, &stream).unwrap();
            stream.sync().unwrap();
            assert_bits_eq(
                &format!("{dt:?} absorb single row {r}"),
                &a.to_vec::<u16>().unwrap(),
                &a_all[r * ROW_Q..(r + 1) * ROW_Q],
                LATENT_DIM,
            );
            assert_bits_eq(
                &format!("{dt:?} expand single row {r}"),
                &e.to_vec::<u16>().unwrap(),
                &e_all[r * ROW_OUT..(r + 1) * ROW_OUT],
                NOPE_DIM,
            );
        }
    }
}

/// A verify window written over stale data attends exactly like serial rows, and only rows up to its own position are
/// visible to each window row (the cache beyond the committed length is NaN-poisoned or stale).
#[test]
fn verify_window_equals_serial() {
    if gpu().is_none() {
        return;
    }
    for dt in DTYPES {
        let f = Fmt(dt);
        let len = 2100usize;
        let serial_world = make_world(dt, seqs_of(f, &[len]), 3, 2);
        for &(p, wn) in &[(1000usize, 8usize), (505, 8), (2043, 8), (255, 16), (2090, 10), (0, 8), (511, 128), (1919, 9)] {
            // sequence 1's true latents below P, stale (different) data from P on, window then written
            let truth = gen_latent(f, 1, len);
            let stale = gen_latent(f, 777, len);
            let mut mixed = Seq { sid: 1, len, lat: truth.clone() };
            mixed.lat[p * LATENT_DIM..].copy_from_slice(&stale[p * LATENT_DIM..]);
            let mut ww = make_world(dt, vec![mixed], 21, 6);
            // write the window rows' true latents over positions P..P+W-1
            let hi = p + wn.min(len - p);
            let nw = hi - p;
            let ld = DevBuf::from_slice(&truth[p * LATENT_DIM..hi * LATENT_DIM]).unwrap();
            let ri = RowIndex::from_host(&vec![0; nw], &(p as i32..hi as i32).collect::<Vec<_>>()).unwrap();
            latent_write(&ld, &ri, &ww.table, &mut ww.cache, nw, &ww.stream).unwrap();
            ww.stream.sync().unwrap();
            let rows: Vec<(usize, i32)> = (p..hi).map(|x| (0, x as i32)).collect();
            let out = run_pipeline(&ww, &rows);
            for (r, &(_, pos)) in rows.iter().enumerate() {
                let want = run_pipeline(&serial_world, &[(0, pos)]);
                assert_bits_eq(
                    &format!("{dt:?} window P={p} row {r} out"),
                    &out.out[r * ROW_OUT..(r + 1) * ROW_OUT],
                    &want.out,
                    NOPE_DIM,
                );
                assert_bits_eq(
                    &format!("{dt:?} window P={p} row {r} o_lat"),
                    &out.o_lat[r * ROW_Q..(r + 1) * ROW_Q],
                    &want.o_lat,
                    LATENT_DIM,
                );
            }
        }
    }
}

/// A padding row (position -1, what a CUDA-graph bucket uses) is inert: real rows keep their bits, the padding row's
/// attended latent is zeros, and `latent_write` of a padding row leaves the cache alone.
#[test]
fn padding_rows_are_inert() {
    let Some(stream) = gpu() else { return };
    for dt in DTYPES {
        let f = Fmt(dt);
        let lens = [2000usize, 700];
        let mut w = make_world(dt, seqs_of(f, &lens), 17, 3);
        let real = [(0usize, 1999i32), (0, 1500), (1, 699), (0, 3)];
        let alone: Vec<Stages> = real.iter().map(|&r| run_pipeline(&w, &[r])).collect();
        for pad_at in 0..=real.len() {
            for pad_slot in [0usize, 1] {
                let mut rows = real.to_vec();
                rows.insert(pad_at, (pad_slot, -1));
                let st = run_pipeline(&w, &rows);
                let mut ri = 0;
                for (i, &(_, p)) in rows.iter().enumerate() {
                    if p < 0 {
                        assert!(
                            st.o_lat[i * ROW_Q..(i + 1) * ROW_Q].iter().all(|&b| b == 0),
                            "padding row {i}: o_lat must be zeros"
                        );
                        assert!(
                            st.out[i * ROW_OUT..(i + 1) * ROW_OUT].iter().all(|&b| b == 0),
                            "padding row {i}: out must be zeros"
                        );
                    } else {
                        assert_bits_eq(
                            &format!("{dt:?} pad at {pad_at} out"),
                            &st.out[i * ROW_OUT..(i + 1) * ROW_OUT],
                            &alone[ri].out,
                            NOPE_DIM,
                        );
                        ri += 1;
                    }
                }
            }
        }
        let junk = DevBuf::from_slice(&vec![0x1234u16; 2 * LATENT_DIM]).unwrap();
        let ri = RowIndex::from_host(&[0, 0], &[-1, -1]).unwrap();
        latent_write(&junk, &ri, &w.table, &mut w.cache, 2, &stream).unwrap();
        stream.sync().unwrap();
        for (i, &r) in real.iter().enumerate() {
            assert_bits_eq("after padding latent_write", &run_pipeline(&w, &[r]).out, &alone[i].out, NOPE_DIM);
        }
    }
}

/// The grid is padded by `max_context`: any value at or above the rows' contexts must give the same bits (the extra chunks
/// are exact no-ops), which is also what keeps the result independent of how the caller sizes its launch.
#[test]
fn max_context_padding_is_a_noop() {
    if gpu().is_none() {
        return;
    }
    for dt in DTYPES {
        let f = Fmt(dt);
        let lens = [2100usize, 600];
        let w = make_world(dt, seqs_of(f, &lens), 4, 3);
        let rows = [(0usize, 1500i32), (0, 1501), (1, 599), (0, 300)];
        let tight = run_pipeline_ctx(&w, &rows, Some(1502));
        let cap = w.table.max_blocks * BLOCK_TOKENS;
        for ctx in [1503usize, 1536, 1700, 2048, 2100, cap] {
            let st = run_pipeline_ctx(&w, &rows, Some(ctx.min(cap)));
            assert_bits_eq(&format!("{dt:?} max_context {ctx} o_lat"), &st.o_lat, &tight.o_lat, LATENT_DIM);
            assert_bits_eq(&format!("{dt:?} max_context {ctx} out"), &st.out, &tight.out, NOPE_DIM);
        }
    }
}

/// ~100k tokens: 391 chunks merged in order, block ids past 1000, against the f64 reference on a few heads.
#[test]
fn long_context_matches_f64_reference() {
    if gpu().is_none() {
        return;
    }
    for dt in DTYPES {
        let len = 100_000usize;
        let w = make_world(dt, seqs_of(Fmt(dt), &[len]), 23, 700);
        assert!(w.cache.n_blocks > 1000);
        let rows: Vec<(usize, i32)> = vec![(0, len as i32 - 1), (0, 65_535), (0, 511)];
        let st = run_pipeline(&w, &rows);
        for (r, &(si, p)) in rows.iter().enumerate() {
            check_attend("long", &w, &st, r, si, p as usize, &[3, 40]);
        }
    }
}

/// The tuning variants are different tilings of the same math: close to the default, never the contract.
#[test]
fn tuning_variants_agree_with_the_default() {
    let Some(stream) = gpu() else { return };
    for dt in DTYPES {
        let f = Fmt(dt);
        let w = make_world(dt, seqs_of(f, &[2051, 777]), 8, 3);
        let rows = [(0usize, 2050i32), (0, 1000), (1, 776), (1, 5)];
        let st = run_pipeline(&w, &rows);
        let n = rows.len();
        let slot: Vec<i32> = rows.iter().map(|r| r.0 as i32).collect();
        let pos: Vec<i32> = rows.iter().map(|r| r.1).collect();
        let ri = RowIndex::from_host(&slot, &pos).unwrap();
        let q_lat = DevBuf::from_slice(&st.q_lat).unwrap();
        let mut outs: Vec<Vec<u16>> = vec![st.o_lat.clone()];
        for variant in 1..=VARIANTS {
            let max_ctx = 2051;
            let mut ws = DevBuf::alloc(attend_workspace_bytes(n, max_ctx) * 2).unwrap();
            let mut o = DevBuf::zeroed(n * ROW_Q * 2).unwrap();
            attend_variant(&q_lat, &w.cache, &w.table, &ri, n, &mut o, max_ctx, &mut ws, variant, &stream).unwrap();
            stream.sync().unwrap();
            let got: Vec<u16> = o.to_vec().unwrap();
            let (cos, rel) = cos_rel_rms(&f.decs(&got), &f.decs(&st.o_lat));
            assert!(cos >= 0.99997 && rel <= 5e-3, "{dt:?} variant {variant}: cos {cos} rel {rel}");
            outs.push(got);
        }
        // the number of head tiles a CTA serves (and so how a K/V tile is shared) is invisible in the bits: the same
        // (tile, chunk) shape with one or two head tiles per CTA gives identical outputs
        for (a, b) in [(0usize, 5usize), (2, 6), (1, 7), (3, 8)] {
            assert_bits_eq(&format!("{dt:?} variant {a} vs {b}"), &outs[a], &outs[b], LATENT_DIM);
        }
    }
}

// ---------------------------------------------------------------------------------------------------- sparse (DSA) regime

/// Elements between rows of a selection's `indices`, as `dsa::Selection` lays them out.
const SEL_STRIDE: usize = 2056;

/// A device selection for `lists.len()` rows: `Some(list)` is a sparse row with those positions, `None` a dense row (flag 0, count 7, and
/// an unmapped list). Entries past a row's count hold an unmapped position, so a kernel that reads them faults or poisons the row.
struct Sel {
    indices: DevBuf,
    meta: DevBuf,
}

fn make_sel(lists: &[Option<Vec<i32>>]) -> Sel {
    let n = lists.len();
    let mut idx = vec![0x3fff_fff0i32; n * SEL_STRIDE];
    let mut meta = vec![0i32; n * 2];
    for (r, l) in lists.iter().enumerate() {
        if let Some(l) = l {
            idx[r * SEL_STRIDE..r * SEL_STRIDE + l.len()].copy_from_slice(l);
            meta[2 * r] = 1;
            meta[2 * r + 1] = l.len() as i32;
        } else {
            meta[2 * r + 1] = 7; // a dense row's count is not guaranteed to be 0: only the flag says "not sparse"
        }
    }
    Sel { indices: DevBuf::from_slice(&idx).unwrap(), meta: DevBuf::from_slice(&meta).unwrap() }
}

impl Sel {
    fn view(&self) -> SparseSel<'_> {
        SparseSel { indices: &self.indices, meta: &self.meta, stride: SEL_STRIDE }
    }
}

/// What the indexer selects for a row at `pos`: 512 random pools of 4 tokens (all of them when fewer exist) in ascending order,
/// then the raw tail tokens of the incomplete last pool.
fn dsa_list(pos: usize, seed: u64) -> Vec<i32> {
    let vis = pos + 1;
    let (pools, tail) = (vis / 4, vis % 4);
    let mut ids: Vec<usize> = (0..pools).collect();
    shuffle(&mut ids, &mut Rng(hash_seed(seed, pos as u64)));
    ids.truncate(pools.min(512));
    ids.sort();
    let mut v: Vec<i32> = ids.iter().flat_map(|&p| (0..4).map(move |t| (p * 4 + t) as i32)).collect();
    v.extend((0..tail).map(|t| (pools * 4 + t) as i32));
    v
}

/// `count` distinct positions from `0..=pos`, ascending.
fn rand_list(pos: usize, count: usize, seed: u64) -> Vec<i32> {
    let mut ids: Vec<i32> = (0..=pos as i32).collect();
    shuffle(&mut ids, &mut Rng(hash_seed(seed, count as u64 ^ ((pos as u64) << 20))));
    ids.truncate(count);
    ids.sort();
    ids
}

/// The pipeline with the per-row regime: absorb, `attend_auto` (dense rows dense, `Some(list)` rows sparse), expand. The attended
/// latents start as a sentinel, so a row neither kernel served stays visibly untouched.
fn run_auto(w: &World, rows: &[(usize, i32)], lists: &[Option<Vec<i32>>], max_ctx: Option<usize>) -> Stages {
    assert_eq!(rows.len(), lists.len());
    let n = rows.len();
    let mut q = Vec::new();
    for &(si, p) in rows {
        q.extend(qvec(w.f, w.seqs[si].sid, p.max(0) as usize));
    }
    let max_ctx = max_ctx.unwrap_or_else(|| rows.iter().map(|r| (r.1 + 1).max(1) as usize).max().unwrap());
    let max_ctx = max_ctx.min(w.table.max_blocks * BLOCK_TOKENS);
    let ri =
        RowIndex::from_host(&rows.iter().map(|r| r.0 as i32).collect::<Vec<_>>(), &rows.iter().map(|r| r.1).collect::<Vec<_>>())
            .unwrap();
    let sel = make_sel(lists);
    let qd = DevBuf::from_slice(&q).unwrap();
    let mut q_lat = DevBuf::from_slice(&vec![0xABCDu16; n * ROW_Q]).unwrap();
    let mut o_lat = DevBuf::from_slice(&vec![0xABCDu16; n * ROW_Q]).unwrap();
    let mut out = DevBuf::from_slice(&vec![0xABCDu16; n * ROW_OUT]).unwrap();
    let mut ws = DevBuf::alloc(attend_auto_workspace_bytes(n, max_ctx)).unwrap();
    ws.memset_async(0x3F, &w.stream).unwrap();
    absorb(&qd, N_HEADS * NOPE_DIM, &w.w, &mut q_lat, n, &w.stream).unwrap();
    attend_auto(&q_lat, &w.cache, &w.table, &ri, n, &sel.view(), &mut o_lat, max_ctx, &mut ws, &w.stream).unwrap();
    expand(&o_lat, &w.w, &mut out, n, &w.stream).unwrap();
    w.stream.sync().unwrap();
    Stages { q_lat: q_lat.to_vec().unwrap(), o_lat: o_lat.to_vec().unwrap(), out: out.to_vec().unwrap() }
}

/// `attend_sparse` alone over rows with the given lists, on the queries the pipeline's absorb makes; `o_lat` starts as a sentinel.
fn run_sparse_only(w: &World, rows: &[(usize, i32)], lists: &[Option<Vec<i32>>]) -> Stages {
    let n = rows.len();
    let mut q = Vec::new();
    for &(si, p) in rows {
        q.extend(qvec(w.f, w.seqs[si].sid, p.max(0) as usize));
    }
    let ri =
        RowIndex::from_host(&rows.iter().map(|r| r.0 as i32).collect::<Vec<_>>(), &rows.iter().map(|r| r.1).collect::<Vec<_>>())
            .unwrap();
    let sel = make_sel(lists);
    let qd = DevBuf::from_slice(&q).unwrap();
    let mut q_lat = DevBuf::zeroed(n * ROW_Q * 2).unwrap();
    let mut o_lat = DevBuf::from_slice(&vec![0xABCDu16; n * ROW_Q]).unwrap();
    let mut ws = DevBuf::alloc(attend_sparse_workspace_bytes(n)).unwrap();
    ws.memset_async(0x3F, &w.stream).unwrap();
    absorb(&qd, N_HEADS * NOPE_DIM, &w.w, &mut q_lat, n, &w.stream).unwrap();
    attend_sparse(&q_lat, &w.cache, &w.table, &ri, n, &sel.view(), &mut o_lat, &mut ws, &w.stream).unwrap();
    w.stream.sync().unwrap();
    Stages { q_lat: q_lat.to_vec().unwrap(), o_lat: o_lat.to_vec().unwrap(), out: Vec::new() }
}

fn check_sparse(what: &str, w: &World, st: &Stages, r: usize, si: usize, list: &[i32], heads: &[usize]) {
    let q_lat = &st.q_lat[r * ROW_Q..(r + 1) * ROW_Q];
    let positions: Vec<usize> = list.iter().map(|&p| p as usize).collect();
    let want = ref_attend_list(q_lat, &w.seqs[si].lat, &positions, heads, w.f);
    let mut got = Vec::new();
    for &h in heads {
        got.extend(w.f.decs(&st.o_lat[r * ROW_Q + h * LATENT_DIM..][..LATENT_DIM]));
    }
    let want32: Vec<f32> = want.iter().map(|&x| x as f32).collect();
    let (cos, rel) = cos_rel_rms(&got, &want32);
    let (cmin, rmax) = if w.f.0 == Dtype::BF16 { (0.99995, 5e-3) } else { (0.999995, 8e-4) };
    assert!(cos >= cmin && rel <= rmax, "{what}: {:?} {} entries: cos {cos} relRMS {rel}", w.f.0, list.len());
}

#[test]
fn sparse_matches_f64_reference() {
    if gpu().is_none() {
        return;
    }
    for dt in DTYPES {
        let w = make_world(dt, seqs_of(Fmt(dt), &[6000, 2100, 3000]), 31, 4);
        // DSA-shaped selections of rows deep in the sparse regime, at the boundary, and at several tails
        let mut rows: Vec<(usize, i32)> =
            vec![(0, 5999), (0, 2048), (0, 4097), (0, 4098), (0, 4099), (1, 2099), (2, 2999), (2, 2050)];
        let mut lists: Vec<Option<Vec<i32>>> = rows.iter().map(|&(_, p)| Some(dsa_list(p as usize, 5))).collect();
        // list lengths around the tile (32), chunk (256) and page edges, and the longest list
        for (k, &count) in [1usize, 2, 31, 32, 33, 255, 256, 257, 511, 512, 513, 1000, 2047, 2048, 2051].iter().enumerate() {
            rows.push((0, 5000 + k as i32));
            lists.push(Some(rand_list(5000 + k, count, 9)));
        }
        let st = run_auto(&w, &rows, &lists, None);
        for (r, &(si, _)) in rows.iter().enumerate() {
            check_sparse("sparse", &w, &st, r, si, lists[r].as_ref().unwrap(), &[0, 17, 40, 63]);
        }
        // the expanded output of a sparse row against f64 on its own attended latents
        let want = ref_expand(&st.o_lat[..ROW_Q], &packed(dt).1, w.f);
        assert_close_chain("sparse expand", w.f, &st.out[..ROW_OUT], &want, LATENT_DIM);
    }
}

/// A sparse row whose list is everything it sees (0..=pos) takes exactly the dense kernel's tiles in the dense kernel's order:
/// bit-identical to it. So a row at the regime boundary changes nothing about its bits by which kernel serves it, and the
/// gather itself (page lookup, shuffles, list-entry masks) is proven against the contiguous path.
#[test]
fn sparse_on_the_identity_selection_equals_dense_bit_for_bit() {
    if gpu().is_none() {
        return;
    }
    for dt in DTYPES {
        let w = make_world(dt, seqs_of(Fmt(dt), &[2100, 700]), 13, 3);
        let probes: Vec<(usize, i32)> = [0usize, 1, 31, 32, 255, 256, 257, 511, 512, 700 - 1, 1000, 1023, 1024, 2046, 2047]
            .iter()
            .map(|&p| (if p < 700 { 1 } else { 0 }, p as i32))
            .collect();
        let lists: Vec<Option<Vec<i32>>> = probes.iter().map(|&(_, p)| Some((0..=p).collect())).collect();
        let sparse = run_sparse_only(&w, &probes, &lists);
        let dense = run_pipeline(&w, &probes);
        assert_bits_eq(&format!("{dt:?} identity selection"), &sparse.o_lat, &dense.o_lat, LATENT_DIM);
    }
}

/// Row invariance of the sparse kernel: a row alone equals the same row in any window, batch and block placement, whatever the
/// other rows' lists are, and entries past a row's count are never read.
#[test]
fn sparse_row_invariance_bit_exact() {
    if gpu().is_none() {
        return;
    }
    for dt in DTYPES {
        let f = Fmt(dt);
        let lens = [5000usize, 3000, 2600];
        let w1 = make_world(dt, seqs_of(f, &lens), 2, 4);
        let w2 = make_world(dt, seqs_of(f, &lens), 77, 9);
        let alone = |w: &World, si: usize, p: usize, l: &Vec<i32>| run_sparse_only(w, &[(si, p as i32)], &[Some(l.clone())]);
        // (slot, pos, list) probes: DSA lists and odd list lengths
        let mut probes: Vec<(usize, usize, Vec<i32>)> = Vec::new();
        for (si, p) in [(0usize, 4999usize), (0, 2048), (0, 3333), (1, 2999), (1, 2300), (2, 2599), (2, 2049)] {
            probes.push((si, p, dsa_list(p, 3)));
        }
        for (si, p, c) in [(0usize, 4000usize, 33usize), (1, 2500, 257), (2, 2400, 1), (0, 3000, 2051)] {
            probes.push((si, p, rand_list(p, c, 4)));
        }
        let base: Vec<Stages> = probes.iter().map(|(s, p, l)| alone(&w1, *s, *p, l)).collect();
        for (i, (s, p, l)) in probes.iter().enumerate() {
            let a = alone(&w2, *s, *p, l);
            assert_bits_eq(&format!("{dt:?} placement probe {i}"), &a.o_lat, &base[i].o_lat, LATENT_DIM);
        }
        // batches of 2..128 rows (every probe, in random order, with filler rows of other lists), the probe at every position
        let mut r = Rng(11);
        for &n in &[2usize, 3, 8, 9, 17, 64, 128] {
            for (i, (s, p, l)) in probes.iter().enumerate() {
                let off = (r.next_u64() % n as u64) as usize;
                let mut rows: Vec<(usize, i32)> = Vec::new();
                let mut lists: Vec<Option<Vec<i32>>> = Vec::new();
                for k in 0..n {
                    if k == off {
                        rows.push((*s, *p as i32));
                        lists.push(Some(l.clone()));
                    } else {
                        let (fs, fp, fl) = &probes[(r.next_u64() % probes.len() as u64) as usize];
                        rows.push((*fs, *fp as i32));
                        lists.push(Some(fl.clone()));
                    }
                }
                let st = run_sparse_only(if n % 2 == 0 { &w1 } else { &w2 }, &rows, &lists);
                assert_bits_eq(
                    &format!("{dt:?} batch {n} probe {i} at {off}"),
                    &st.o_lat[off * ROW_Q..(off + 1) * ROW_Q],
                    &base[i].o_lat,
                    LATENT_DIM,
                );
            }
        }
    }
}

/// Per-row regime inside one call: a verify window straddling 2,048 tokens takes the dense kernel for its first rows and the
/// sparse one for the rest, each bit-identical to the row decoded alone, and each correct against f64. Rows neither kernel
/// serves are left untouched.
#[test]
fn auto_decides_the_regime_per_row() {
    if gpu().is_none() {
        return;
    }
    for dt in DTYPES {
        let f = Fmt(dt);
        let w = make_world(dt, seqs_of(f, &[3000, 600, 5000]), 6, 3);
        let list_of = |p: usize| if p < DENSE_MAX_KEYS { None } else { Some(dsa_list(p, 8)) };
        for &(first, n) in &[(2040usize, 17usize), (2047, 2), (2046, 8), (2048, 8), (2000, 120), (2040, 128)] {
            let rows: Vec<(usize, i32)> = (0..n).map(|k| (0usize, (first + k) as i32)).collect();
            let lists: Vec<Option<Vec<i32>>> = rows.iter().map(|&(_, p)| list_of(p as usize)).collect();
            let st = run_auto(&w, &rows, &lists, None);
            for (k, &(_, p)) in rows.iter().enumerate() {
                let one = run_auto(&w, &[(0, p)], &[list_of(p as usize)], None);
                let tag = format!("{dt:?} window {first}+{n} row {k} (pos {p})");
                assert_bits_eq(&format!("{tag} o_lat"), &st.o_lat[k * ROW_Q..(k + 1) * ROW_Q], &one.o_lat, LATENT_DIM);
                assert_bits_eq(&format!("{tag} out"), &st.out[k * ROW_OUT..(k + 1) * ROW_OUT], &one.out, NOPE_DIM);
                if k % 5 == 0 || k + 1 == n {
                    match &lists[k] {
                        Some(l) => check_sparse(&tag, &w, &st, k, 0, l, &[1, 33]),
                        None => check_attend(&tag, &w, &st, k, 0, p as usize, &[1, 33]),
                    }
                }
            }
        }
        // a mixed batch: deep sparse rows of two sequences, dense rows of two others, padding rows
        let rows = [(2usize, 4999i32), (1, 599), (0, 2999), (2, -1), (0, 100), (2, 2048), (1, 0), (0, 2047)];
        let lists: Vec<Option<Vec<i32>>> = rows.iter().map(|&(_, p)| if p < 0 { None } else { list_of(p as usize) }).collect();
        let st = run_auto(&w, &rows, &lists, None);
        for (k, &(si, p)) in rows.iter().enumerate() {
            if p < 0 {
                continue;
            }
            let one = run_auto(&w, &[(si, p)], &[list_of(p as usize)], None);
            assert_bits_eq(&format!("{dt:?} mixed row {k}"), &st.o_lat[k * ROW_Q..(k + 1) * ROW_Q], &one.o_lat, LATENT_DIM);
        }
        // the padding row is zeros in the dense launch's output (it is inert, not skipped)
        assert!(st.o_lat[3 * ROW_Q..4 * ROW_Q].iter().all(|&b| b == 0), "padding row must be zeros");
    }
}

/// The two kernels leave each other's rows alone: `attend_sparse` writes nothing for dense or padding rows, `attend_below` nothing
/// for rows at or past the limit, and `attend_sparse` ignores a dense row's (poisoned) list.
#[test]
fn each_regime_leaves_the_other_rows_untouched() {
    let Some(stream) = gpu() else { return };
    for dt in DTYPES {
        let f = Fmt(dt);
        let w = make_world(dt, seqs_of(f, &[3000]), 12, 3);
        let rows = [(0usize, 2999i32), (0, 100), (0, -1), (0, 2048), (0, 2047)];
        let lists = [Some(dsa_list(2999, 1)), None, None, Some(dsa_list(2048, 1)), None];
        // sparse only: dense and padding rows keep the sentinel
        let st = run_sparse_only(&w, &rows, &lists);
        for (k, l) in lists.iter().enumerate() {
            let untouched = st.o_lat[k * ROW_Q..(k + 1) * ROW_Q].iter().all(|&b| b == 0xABCD);
            assert_eq!(untouched, l.is_none(), "{dt:?} sparse launch, row {k}");
        }
        // dense below the limit only: sparse rows keep the sentinel, dense rows are written, the padding row is zeros
        let n = rows.len();
        let ri = RowIndex::from_host(
            &rows.iter().map(|r| r.0 as i32).collect::<Vec<_>>(),
            &rows.iter().map(|r| r.1).collect::<Vec<_>>(),
        )
        .unwrap();
        let qd = DevBuf::from_slice(
            &rows.iter().flat_map(|&(si, p)| qvec(f, w.seqs[si].sid, p.max(0) as usize)).collect::<Vec<u16>>(),
        )
        .unwrap();
        let mut q_lat = DevBuf::zeroed(n * ROW_Q * 2).unwrap();
        absorb(&qd, N_HEADS * NOPE_DIM, &w.w, &mut q_lat, n, &stream).unwrap();
        let mut o_lat = DevBuf::from_slice(&vec![0xABCDu16; n * ROW_Q]).unwrap();
        let mut ws = DevBuf::zeroed(attend_workspace_bytes(n, DENSE_MAX_KEYS)).unwrap();
        attend_below(&q_lat, &w.cache, &w.table, &ri, n, &mut o_lat, DENSE_MAX_KEYS, 3000, &mut ws, &stream).unwrap();
        stream.sync().unwrap();
        let o: Vec<u16> = o_lat.to_vec().unwrap();
        for (k, &(_, p)) in rows.iter().enumerate() {
            let row = &o[k * ROW_Q..(k + 1) * ROW_Q];
            if p >= DENSE_MAX_KEYS as i32 {
                assert!(row.iter().all(|&b| b == 0xABCD), "{dt:?} dense launch wrote sparse row {k}");
            } else if p < 0 {
                assert!(row.iter().all(|&b| b == 0), "{dt:?} padding row {k} must be zeros");
            } else {
                // a written row can hold the sentinel's bit pattern in a few elements, but never in all of them
                assert!(row.iter().any(|&b| b != 0xABCD), "{dt:?} dense launch skipped dense row {k}");
            }
        }
    }
}

/// A hole (-1) inside a list is no key: it is masked, not scored as a zero key. Lists with scattered holes, a hole run longer than a
/// whole chunk, holes at the start, and an unordered list all match the f64 attention over exactly their real entries; a list of
/// nothing but holes writes zeros (not NaN).
#[test]
fn sparse_holes_are_masked_and_order_is_free() {
    if gpu().is_none() {
        return;
    }
    for dt in DTYPES {
        let w = make_world(dt, seqs_of(Fmt(dt), &[6000]), 41, 3);
        let base = rand_list(5500, 900, 3);
        let mut scattered = Vec::new();
        for (i, &p) in base.iter().enumerate() {
            if i % 7 == 3 {
                scattered.push(-1);
            }
            scattered.push(p);
        }
        let mut long_run = base[..300].to_vec();
        long_run.extend(std::iter::repeat_n(-1, 300)); // more than a 256-entry chunk of nothing
        long_run.extend_from_slice(&base[300..]);
        let mut leading = vec![-1; 5];
        leading.extend_from_slice(&base);
        let mut shuffled = base.clone();
        shuffle(&mut shuffled, &mut Rng(77));
        let cases: Vec<Vec<i32>> = vec![scattered, long_run, leading, shuffled];
        let rows: Vec<(usize, i32)> = cases.iter().map(|_| (0usize, 5500)).chain([(0, 5500)]).collect();
        let mut lists: Vec<Option<Vec<i32>>> = cases.iter().cloned().map(Some).collect();
        lists.push(Some(vec![-1; 300])); // no real key at all
        let st = run_sparse_only(&w, &rows, &lists);
        for (k, l) in cases.iter().enumerate() {
            let real: Vec<i32> = l.iter().copied().filter(|&p| p >= 0).collect();
            check_sparse(&format!("holes case {k}"), &w, &st, k, 0, &real, &[0, 17, 40, 63]);
        }
        let empty = &st.o_lat[cases.len() * ROW_Q..(cases.len() + 1) * ROW_Q];
        assert!(empty.iter().all(|&b| b == 0), "{dt:?}: a list with no real key must write zeros");
    }
}

/// The entry points the model code calls (`mla_rows`, `mla_rows_auto` with an `MlaScratch`) are exactly the composed pipeline:
/// same kernels in the same order, so the same bits as the stage-by-stage runs above.
#[test]
fn model_entry_points_are_the_composed_pipeline() {
    let Some(stream) = gpu() else { return };
    for dt in DTYPES {
        let f = Fmt(dt);
        let w = make_world(dt, seqs_of(f, &[3000, 900]), 14, 3);
        // a window straddling 2,048 plus a short sequence's rows: both regimes in one call
        let rows: Vec<(usize, i32)> = (2044..2052).map(|p| (0usize, p)).chain([(1usize, 899), (1, 0)]).collect();
        let lists: Vec<Option<Vec<i32>>> =
            rows.iter().map(|&(_, p)| if p < DENSE_MAX_KEYS as i32 { None } else { Some(dsa_list(p as usize, 2)) }).collect();
        let want = run_auto(&w, &rows, &lists, Some(2052));
        let n = rows.len();
        let ri = RowIndex::from_host(
            &rows.iter().map(|r| r.0 as i32).collect::<Vec<_>>(),
            &rows.iter().map(|r| r.1).collect::<Vec<_>>(),
        )
        .unwrap();
        let q: Vec<u16> = rows.iter().flat_map(|&(si, p)| qvec(f, w.seqs[si].sid, p as usize)).collect();
        let qd = DevBuf::from_slice(&q).unwrap();
        let sel = make_sel(&lists);
        let mut scratch = MlaScratch::new(16, 4096, dt).unwrap();
        assert!(MlaScratch::bytes(16, 4096, dt) >= scratch.q_lat.bytes() + scratch.o_lat.bytes() + scratch.ws.bytes());
        let mut out = DevBuf::zeroed(n * ROW_OUT * 2).unwrap();
        mla_rows_auto(
            &qd,
            N_HEADS * NOPE_DIM,
            &w.w,
            &w.cache,
            &w.table,
            &ri,
            n,
            &sel.view(),
            &mut out,
            2052,
            &mut scratch,
            &stream,
        )
        .unwrap();
        stream.sync().unwrap();
        assert_bits_eq(&format!("{dt:?} mla_rows_auto"), &out.to_vec::<u16>().unwrap(), &want.out, NOPE_DIM);
        // dense only: the same rows below the boundary through `mla_rows`
        let dense_rows = [(0usize, 2047i32), (0, 1000), (1, 899)];
        let ri = RowIndex::from_host(
            &dense_rows.iter().map(|r| r.0 as i32).collect::<Vec<_>>(),
            &dense_rows.iter().map(|r| r.1).collect::<Vec<_>>(),
        )
        .unwrap();
        let q: Vec<u16> = dense_rows.iter().flat_map(|&(si, p)| qvec(f, w.seqs[si].sid, p as usize)).collect();
        let qd = DevBuf::from_slice(&q).unwrap();
        let mut out = DevBuf::zeroed(3 * ROW_OUT * 2).unwrap();
        mla_rows(&qd, N_HEADS * NOPE_DIM, &w.w, &w.cache, &w.table, &ri, 3, &mut out, 2048, &mut scratch, &stream).unwrap();
        stream.sync().unwrap();
        let want = run_pipeline(&w, &dense_rows);
        assert_bits_eq(&format!("{dt:?} mla_rows"), &out.to_vec::<u16>().unwrap(), &want.out, NOPE_DIM);
    }
}

// ---------------------------------------------------------------------------------------------------- golden hashes

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3))
}

fn hash16(v: &[u16]) -> u64 {
    fnv(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
}

/// (name, `(slot, position)` rows, per-row selection lists) of a golden sparse case.
type GoldenCase = (String, Vec<(usize, i32)>, Vec<Option<Vec<i32>>>);

/// Hashes of the three stages over a spread of contexts, window shapes and dtypes on synthetic data (no pack needed).
fn golden_lines() -> String {
    let mut out = String::new();
    for dt in DTYPES {
        let f = Fmt(dt);
        let lens = [1usize, 255, 257, 1000, 2048, 2051, 3000];
        let w = make_world(dt, seqs_of(f, &lens), 11, 3);
        for (si, &len) in lens.iter().enumerate() {
            for (name, rows) in [
                ("last", vec![len as i32 - 1]),
                ("first", vec![0]),
                ("win2", (len.saturating_sub(2)..len).map(|p| p as i32).collect::<Vec<_>>()),
                ("win8", (len.saturating_sub(8)..len).map(|p| p as i32).collect::<Vec<_>>()),
                ("win9", (len.saturating_sub(9)..len).map(|p| p as i32).collect::<Vec<_>>()),
            ] {
                let r: Vec<(usize, i32)> = rows.iter().map(|&p| (si, p)).collect();
                let st = run_pipeline(&w, &r);
                out += &format!(
                    "{dt:?} len={len} {name} rows={} q_lat {:016x} o_lat {:016x} out {:016x}\n",
                    r.len(),
                    hash16(&st.q_lat),
                    hash16(&st.o_lat),
                    hash16(&st.out)
                );
            }
        }
        // a 128-row window and a mixed 12-sequence-row batch
        let r: Vec<(usize, i32)> = (0..128).map(|p| (6, 2800 + p)).collect();
        let st = run_pipeline(&w, &r);
        out += &format!(
            "{dt:?} win128 q_lat {:016x} o_lat {:016x} out {:016x}\n",
            hash16(&st.q_lat),
            hash16(&st.o_lat),
            hash16(&st.out)
        );
        let mut rg = Rng(2);
        let r: Vec<(usize, i32)> = (0..12)
            .map(|_| {
                let si = (rg.next_u64() % lens.len() as u64) as usize;
                (si, (rg.next_u64() % lens[si] as u64) as i32)
            })
            .collect();
        let st = run_pipeline(&w, &r);
        out += &format!(
            "{dt:?} batch12 q_lat {:016x} o_lat {:016x} out {:016x}\n",
            hash16(&st.q_lat),
            hash16(&st.o_lat),
            hash16(&st.out)
        );
        // the sparse regime: DSA-shaped lists at the boundary and deep in the context, a window straddling 2,048, odd list lengths
        let sw = make_world(dt, seqs_of(f, &[3000, 5000]), 12, 3);
        let cases: Vec<GoldenCase> = vec![
            ("sparse2048".to_string(), vec![(0, 2048)], vec![Some(dsa_list(2048, 21))]),
            ("sparse4999".to_string(), vec![(1, 4999)], vec![Some(dsa_list(4999, 21))]),
            (
                "straddle8".to_string(),
                (2044..2052).map(|p| (0usize, p)).collect(),
                (2044..2052usize).map(|p| if p < 2048 { None } else { Some(dsa_list(p, 22)) }).collect(),
            ),
            ("list257".to_string(), vec![(1, 4000)], vec![Some(rand_list(4000, 257, 23))]),
            ("list2051".to_string(), vec![(1, 4500)], vec![Some(rand_list(4500, 2051, 23))]),
            (
                "mixed6".to_string(),
                vec![(1, 4999), (0, 100), (0, 2999), (1, 2048), (0, 2047), (1, 3000)],
                vec![
                    Some(dsa_list(4999, 24)),
                    None,
                    Some(dsa_list(2999, 24)),
                    Some(dsa_list(2048, 24)),
                    None,
                    Some(dsa_list(3000, 24)),
                ],
            ),
        ];
        for (name, rows, lists) in cases {
            let st = run_auto(&sw, &rows, &lists, None);
            out += &format!("{dt:?} {name} rows={} o_lat {:016x} out {:016x}\n", rows.len(), hash16(&st.o_lat), hash16(&st.out));
        }
    }
    out
}

/// Writes the golden file (`REXL3_GOLDEN_OUT`); the gate regenerates it and diffs against `testdata/gpu/golden_mla.txt`:
///
///     REXL3_GOLDEN_OUT=golden_mla.txt cargo test --release -p gb-cuda --test mla golden_mla_hashes -- --ignored
#[test]
#[ignore]
fn golden_mla_hashes() {
    if gpu().is_none() {
        return;
    }
    let path = std::env::var("REXL3_GOLDEN_OUT").expect("REXL3_GOLDEN_OUT");
    std::fs::write(path, golden_lines()).unwrap();
}

/// The committed golden file matches a fresh run (a kernel change that moves bits must regenerate it in its own commit).
#[test]
fn mla_hashes_match_committed_file() {
    if gpu().is_none() {
        return;
    }
    let path = common::repo_root().join("testdata/gpu/golden_mla.txt");
    let want = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let got = golden_lines();
    if got != want {
        let diff: Vec<String> = got
            .lines()
            .zip(want.lines())
            .filter(|(a, b)| a != b)
            .take(4)
            .map(|(a, b)| format!("  now  {a}\n  file {b}"))
            .collect();
        panic!(
            "MLA golden hashes differ from {} ({} of {} lines):\n{}",
            path.display(),
            got.lines().zip(want.lines()).filter(|(a, b)| a != b).count(),
            want.lines().count(),
            diff.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------------------------------- benchmarks

fn time_ms(stream: &Stream, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..5 {
        f();
    }
    stream.sync().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    stream.sync().unwrap();
    t0.elapsed().as_secs_f64() * 1e3 / iters as f64
}

/// A cache of `ctx` tokens of repeated random latents in one sequence (contiguous blocks), for timing. `REXL3_BENCH_KV=fp8` times the fp8
/// page (default: the T cache).
fn bench_world(dt: Dtype, ctx: usize) -> (LatentCache, BlockTable, Stream) {
    let f = Fmt(dt);
    let stream = Stream::new().unwrap();
    let nb = ctx.div_ceil(BLOCK_TOKENS);
    let kv = std::env::var("REXL3_BENCH_KV").map(|v| LatentKv::parse(&v).unwrap()).unwrap_or_default();
    let mut cache = LatentCache::new(nb, dt, kv).unwrap();
    let table = BlockTable::from_host(&[(0..nb as u32).collect()], nb).unwrap();
    let lat = gen_latent(f, 42, 4096.min(ctx));
    let ld = DevBuf::from_slice(&lat).unwrap();
    let mut done = 0;
    while done < ctx {
        let n = 4096.min(ctx - done);
        let pos: Vec<i32> = (done as i32..(done + n) as i32).collect();
        let ri = RowIndex::from_host(&vec![0; n], &pos).unwrap();
        latent_write(&ld, &ri, &table, &mut cache, n, &stream).unwrap();
        stream.sync().unwrap();
        done += n;
    }
    (cache, table, stream)
}

/// Kernel times: absorb, attention and expand at 1 and 8 rows for the dense contexts, and the same attention at long
/// contexts (what the sparse regime must beat). `REXL3_BENCH_DT=bf16` switches the dtype (default fp16).
#[test]
#[ignore]
fn bench_mla() {
    if gpu().is_none() {
        return;
    }
    let dt = if std::env::var("REXL3_BENCH_DT").map(|v| v == "bf16").unwrap_or(false) { Dtype::BF16 } else { Dtype::F16 };
    let f = Fmt(dt);
    let wt = weights(dt);
    let w_bytes = (wt.uk.bytes()) as f64;
    for ctx in [256usize, 1024, 2048, 8192, 32768, 131072, 262144] {
        let (cache, table, stream) = bench_world(dt, ctx);
        for rows in [1usize, 8] {
            let pos: Vec<i32> = (0..rows).map(|r| (ctx - rows + r) as i32).collect();
            let ri = RowIndex::from_host(&vec![0; rows], &pos).unwrap();
            let q: Vec<u16> = (0..rows).flat_map(|r| qvec(f, 1, ctx - rows + r)).collect();
            let qd = DevBuf::from_slice(&q).unwrap();
            let mut q_lat = DevBuf::zeroed(rows * ROW_Q * 2).unwrap();
            let mut o_lat = DevBuf::zeroed(rows * ROW_Q * 2).unwrap();
            let mut out = DevBuf::zeroed(rows * ROW_OUT * 2).unwrap();
            let mut ws = DevBuf::zeroed(attend_workspace_bytes(rows, ctx)).unwrap();
            absorb(&qd, N_HEADS * NOPE_DIM, &wt, &mut q_lat, rows, &stream).unwrap();
            let iters = if ctx > 100_000 { 20 } else { 200 };
            let t_abs = time_ms(&stream, iters, || absorb(&qd, N_HEADS * NOPE_DIM, &wt, &mut q_lat, rows, &stream).unwrap());
            let t_att =
                time_ms(&stream, iters, || attend(&q_lat, &cache, &table, &ri, rows, &mut o_lat, ctx, &mut ws, &stream).unwrap());
            let t_exp = time_ms(&stream, iters, || expand(&o_lat, &wt, &mut out, rows, &stream).unwrap());
            let lat_bytes = (ctx * LATENT_DIM * 2) as f64;
            eprintln!(
                "{dt:?} ctx {ctx:>6} rows {rows}: absorb {:7.1} us ({:5.1} GB/s)  attend {:8.1} us ({:6.1} GB/s of {:.2} MiB latent)  expand {:7.1} us ({:5.1} GB/s)",
                t_abs * 1e3,
                w_bytes / t_abs / 1e6,
                t_att * 1e3,
                lat_bytes / t_att / 1e6,
                lat_bytes / 1048576.0,
                t_exp * 1e3,
                w_bytes / t_exp / 1e6
            );
        }
    }
}

/// The sparse regime at long contexts: `attend_sparse` over DSA-shaped selections (2048 + tail latents per row, gathered from
/// anywhere in the context) at 1 and 8 rows, next to the dense kernel over the same cache, which is what the selection saves.
#[test]
#[ignore]
fn bench_mla_sparse() {
    if gpu().is_none() {
        return;
    }
    let dt = Dtype::F16;
    let f = Fmt(dt);
    let wt = weights(dt);
    for ctx in [4096usize, 8192, 32768, 131072, 262144] {
        let (cache, table, stream) = bench_world(dt, ctx);
        for rows in [1usize, 8] {
            let pos: Vec<i32> = (0..rows).map(|r| (ctx - rows + r) as i32).collect();
            let ri = RowIndex::from_host(&vec![0; rows], &pos).unwrap();
            let lists: Vec<Option<Vec<i32>>> = pos.iter().map(|&p| Some(dsa_list(p as usize, 3))).collect();
            let sel = make_sel(&lists);
            let q: Vec<u16> = (0..rows).flat_map(|r| qvec(f, 1, ctx - rows + r)).collect();
            let qd = DevBuf::from_slice(&q).unwrap();
            let mut q_lat = DevBuf::zeroed(rows * ROW_Q * 2).unwrap();
            let mut o_lat = DevBuf::zeroed(rows * ROW_Q * 2).unwrap();
            let mut ws = DevBuf::zeroed(attend_sparse_workspace_bytes(rows)).unwrap();
            absorb(&qd, N_HEADS * NOPE_DIM, &wt, &mut q_lat, rows, &stream).unwrap();
            let t_sp = time_ms(&stream, 200, || {
                attend_sparse(&q_lat, &cache, &table, &ri, rows, &sel.view(), &mut o_lat, &mut ws, &stream).unwrap()
            });
            let mut wsd = DevBuf::zeroed(attend_workspace_bytes(rows, ctx)).unwrap();
            let t_de = time_ms(&stream, if ctx > 100_000 { 10 } else { 50 }, || {
                attend(&q_lat, &cache, &table, &ri, rows, &mut o_lat, ctx, &mut wsd, &stream).unwrap()
            });
            // the mixed entry point on a call whose rows are all sparse: the dense launch covers 8 chunks of idle CTAs
            let mut ws_auto = DevBuf::zeroed(attend_auto_workspace_bytes(rows, ctx)).unwrap();
            let t_auto = time_ms(&stream, 200, || {
                attend_auto(&q_lat, &cache, &table, &ri, rows, &sel.view(), &mut o_lat, ctx, &mut ws_auto, &stream).unwrap()
            });
            eprintln!(
                "ctx {ctx:>6} rows {rows}: sparse {:7.1} us ({:5.1} GB/s of {:.2} MiB gathered)  auto {:7.1} us  dense {:9.1} us  (sparse is {:.0}x cheaper)",
                t_sp * 1e3,
                (rows * 2051 * LATENT_DIM * 2) as f64 / t_sp / 1e6,
                (2051 * LATENT_DIM * 2) as f64 / 1048576.0,
                t_auto * 1e3,
                t_de * 1e3,
                t_de / t_sp
            );
        }
    }
}

/// Absorb and expand with the weights L2-resident (the same weights every launch) and from DRAM (a rotation over 12 layers'
/// weights, 384 MiB, as a real decode step sees them), for 1..128 rows. Each product streams 16 MiB of `kv_b_proj`.
#[test]
#[ignore]
fn bench_mla_weights() {
    if gpu().is_none() {
        return;
    }
    let dt = Dtype::F16;
    let f = Fmt(dt);
    let layers: Vec<MlaWeights> = (0..12).map(|_| weights(dt)).collect();
    let stream = Stream::new().unwrap();
    let mb = layers[0].uk.bytes() as f64;
    for rows in [1usize, 2, 4, 8, 16, 32, 64, 128] {
        let q: Vec<u16> = (0..rows).flat_map(|r| qvec(f, 1, r)).collect();
        let qd = DevBuf::from_slice(&q).unwrap();
        let mut q_lat = DevBuf::zeroed(rows * ROW_Q * 2).unwrap();
        let mut out = DevBuf::zeroed(rows * ROW_OUT * 2).unwrap();
        let mut i = 0usize;
        let warm_a = time_ms(&stream, 100, || absorb(&qd, N_HEADS * NOPE_DIM, &layers[0], &mut q_lat, rows, &stream).unwrap());
        let cold_a = time_ms(&stream, 120, || {
            i += 1;
            absorb(&qd, N_HEADS * NOPE_DIM, &layers[i % 12], &mut q_lat, rows, &stream).unwrap()
        });
        let warm_e = time_ms(&stream, 100, || expand(&q_lat, &layers[0], &mut out, rows, &stream).unwrap());
        let cold_e = time_ms(&stream, 120, || {
            i += 1;
            expand(&q_lat, &layers[i % 12], &mut out, rows, &stream).unwrap()
        });
        eprintln!(
            "rows {rows:>3}: absorb warm {:6.1} us ({:5.0} GB/s) cold {:6.1} us ({:5.1} GB/s)   expand warm {:6.1} us ({:5.0} GB/s) cold {:6.1} us ({:5.1} GB/s)",
            warm_a * 1e3,
            mb / warm_a / 1e6,
            cold_a * 1e3,
            mb / cold_a / 1e6,
            warm_e * 1e3,
            mb / warm_e / 1e6,
            cold_e * 1e3,
            mb / cold_e / 1e6
        );
    }
}

/// Three attention launches at `REXL3_PROFILE_CTX` (default 32768) tokens and `REXL3_PROFILE_ROWS` (default 1) rows, for a
/// profiler to capture: `ncu -k regex:mla_partial -c 3 <test binary> bench_profile_attend --ignored`.
#[test]
#[ignore]
fn bench_profile_attend() {
    if gpu().is_none() {
        return;
    }
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (ctx, rows, variant) = (env("REXL3_PROFILE_CTX", 32768), env("REXL3_PROFILE_ROWS", 1), env("REXL3_PROFILE_VARIANT", 0));
    let dt = Dtype::F16;
    let f = Fmt(dt);
    let wt = weights(dt);
    let (cache, table, stream) = bench_world(dt, ctx);
    let pos: Vec<i32> = (0..rows).map(|r| (ctx - rows + r) as i32).collect();
    let ri = RowIndex::from_host(&vec![0; rows], &pos).unwrap();
    let q: Vec<u16> = (0..rows).flat_map(|r| qvec(f, 1, ctx - rows + r)).collect();
    let qd = DevBuf::from_slice(&q).unwrap();
    let mut q_lat = DevBuf::zeroed(rows * ROW_Q * 2).unwrap();
    let mut o_lat = DevBuf::zeroed(rows * ROW_Q * 2).unwrap();
    let mut ws = DevBuf::zeroed(attend_workspace_bytes(rows, ctx) * 2).unwrap();
    absorb(&qd, N_HEADS * NOPE_DIM, &wt, &mut q_lat, rows, &stream).unwrap();
    for _ in 0..3 {
        attend_variant(&q_lat, &cache, &table, &ri, rows, &mut o_lat, ctx, &mut ws, variant as i32, &stream).unwrap();
    }
    stream.sync().unwrap();
}

/// Attention variants (tile, chunk, head tiles per CTA) over rows and contexts, to pick the default. Environment:
/// `REXL3_BENCH_CTX` (default 256,1024,2048,32768), `REXL3_BENCH_ROWS` (1,8), `REXL3_BENCH_VARIANTS` (0..=8).
#[test]
#[ignore]
fn bench_mla_variants() {
    if gpu().is_none() {
        return;
    }
    let list = |k: &str, d: &[usize]| -> Vec<usize> {
        std::env::var(k).map(|v| v.split(',').map(|x| x.trim().parse().unwrap()).collect()).unwrap_or_else(|_| d.to_vec())
    };
    let ctxs = list("REXL3_BENCH_CTX", &[256, 1024, 2048, 32768]);
    let row_list = list("REXL3_BENCH_ROWS", &[1, 8]);
    let variants = list("REXL3_BENCH_VARIANTS", &(0..=VARIANTS as usize).collect::<Vec<_>>());
    let dt = Dtype::F16;
    let f = Fmt(dt);
    let wt = weights(dt);
    for &ctx in &ctxs {
        let (cache, table, stream) = bench_world(dt, ctx);
        for &rows in &row_list {
            let pos: Vec<i32> = (0..rows).map(|r| (ctx - rows + r) as i32).collect();
            let ri = RowIndex::from_host(&vec![0; rows], &pos).unwrap();
            let q: Vec<u16> = (0..rows).flat_map(|r| qvec(f, 1, ctx - rows + r)).collect();
            let qd = DevBuf::from_slice(&q).unwrap();
            let mut q_lat = DevBuf::zeroed(rows * ROW_Q * 2).unwrap();
            let mut o_lat = DevBuf::zeroed(rows * ROW_Q * 2).unwrap();
            let mut ws = DevBuf::zeroed(attend_workspace_bytes(rows, ctx) * 2).unwrap();
            absorb(&qd, N_HEADS * NOPE_DIM, &wt, &mut q_lat, rows, &stream).unwrap();
            let mut line = format!("ctx {ctx:>6} rows {rows}:");
            for &variant in &variants {
                let t = time_ms(&stream, 200, || {
                    attend_variant(&q_lat, &cache, &table, &ri, rows, &mut o_lat, ctx, &mut ws, variant as i32, &stream).unwrap()
                });
                line += &format!("  v{variant} {:7.1} us", t * 1e3);
            }
            eprintln!("{line}");
        }
    }
}

// ---------------------------------------------------------------------------------------------------- the fused large-row attention

use gb_cuda::mla_prefill::attend_pf;

fn gpu_pf() -> Option<Stream> {
    if !device_available() {
        eprintln!("no CUDA device: not run");
        return None;
    }
    Some(Stream::new().unwrap())
}

/// [`run_auto`] with the fused kernel (`attend_pf`) in place of `attend_auto`: the same queries, the same selections, the same sentinel.
fn run_pf(w: &World, rows: &[(usize, i32)], lists: &[Option<Vec<i32>>]) -> Stages {
    let n = rows.len();
    let q: Vec<u16> = rows.iter().flat_map(|&(si, p)| qvec(w.f, w.seqs[si].sid, p.max(0) as usize)).collect();
    let ri =
        RowIndex::from_host(&rows.iter().map(|r| r.0 as i32).collect::<Vec<_>>(), &rows.iter().map(|r| r.1).collect::<Vec<_>>())
            .unwrap();
    let sel = make_sel(lists);
    let qd = DevBuf::from_slice(&q).unwrap();
    let mut q_lat = DevBuf::from_slice(&vec![0xABCDu16; n * ROW_Q]).unwrap();
    let mut o_lat = DevBuf::from_slice(&vec![0xABCDu16; n * ROW_Q]).unwrap();
    absorb(&qd, N_HEADS * NOPE_DIM, &w.w, &mut q_lat, n, &w.stream).unwrap();
    attend_pf(&q_lat, &w.cache, &w.table, &ri, n, &sel.view(), &mut o_lat, &w.stream).unwrap();
    w.stream.sync().unwrap();
    Stages { q_lat: q_lat.to_vec().unwrap(), o_lat: o_lat.to_vec().unwrap(), out: Vec::new() }
}

/// The fused kernel and the two-kernel path (`attend_auto`) give every row the same bits: prefill-shaped slabs (dense, across the 2,048
/// boundary, sparse, 1..128 rows), lists of every count around the 32-entry tile and the 256-entry chunk, holes (scattered, a whole
/// chunk of them, leading, an unordered list, nothing but holes), padding rows, and a sparse-position row whose meta says dense
/// (left untouched by both). Both dtypes; the whole `o_lat` is compared, sentinel rows included.
#[test]
fn fused_attention_equals_the_two_kernel_path_bit_for_bit() {
    if gpu_pf().is_none() {
        return;
    }
    let mut compared = 0usize;
    for dt in DTYPES {
        let f = Fmt(dt);
        let w = make_world(dt, seqs_of(f, &[3000, 600, 9000, 2100]), 5, 3);
        let list_of = |p: usize, seed: u64| if p < DENSE_MAX_KEYS { None } else { Some(dsa_list(p, seed)) };
        let mut check = |what: String, rows: Vec<(usize, i32)>, lists: Vec<Option<Vec<i32>>>| {
            let want = run_auto(&w, &rows, &lists, None);
            let got = run_pf(&w, &rows, &lists);
            assert_bits_eq(&format!("{dt:?} {what} o_lat"), &got.o_lat, &want.o_lat, LATENT_DIM);
            compared += rows.len();
        };
        // slabs of a prefill chunk: the first rows of a prompt, the 2,048 boundary, deep sparse rows of two sequences
        for &(si, first, n) in &[
            (0usize, 0usize, 128usize),
            (0, 1900, 128),
            (0, 2040, 17),
            (0, 2048, 128),
            (0, 2872, 128),
            (2, 8872, 128),
            (2, 4000, 1),
            (2, 4001, 2),
            (2, 5000, 64),
        ] {
            let rows: Vec<(usize, i32)> = (0..n).map(|k| (si, (first + k) as i32)).collect();
            let lists = rows.iter().map(|&(_, p)| list_of(p as usize, 3)).collect();
            check(format!("slab seq {si} rows {first}+{n}"), rows, lists);
        }
        // counts around the tile (32) and the chunk (256)
        let counts = [1usize, 2, 31, 32, 33, 255, 256, 257, 288, 289, 511, 512, 513, 1000, 2048, 2051];
        let rows: Vec<(usize, i32)> = counts.iter().map(|_| (2usize, 6500)).collect();
        let lists = counts.iter().enumerate().map(|(i, &c)| Some(rand_list(6500, c, 10 + i as u64))).collect();
        check("counts around tile and chunk".into(), rows, lists);
        // holes
        let base = rand_list(5500, 900, 3);
        let mut scattered = Vec::new();
        for (i, &p) in base.iter().enumerate() {
            if i % 7 == 3 {
                scattered.push(-1);
            }
            scattered.push(p);
        }
        let mut long_run = base[..300].to_vec();
        long_run.extend(std::iter::repeat_n(-1, 300));
        long_run.extend_from_slice(&base[300..]);
        let mut leading = vec![-1; 5];
        leading.extend_from_slice(&base);
        let mut shuffled = base.clone();
        shuffle(&mut shuffled, &mut Rng(77));
        let mut hole_chunk = base[..256].to_vec(); // a whole chunk of nothing but holes in the middle
        hole_chunk.extend(std::iter::repeat_n(-1, 256));
        hole_chunk.extend_from_slice(&base[256..]);
        // the FIRST chunk is nothing but holes (the merged state is still -inf when a chunk with no real key arrives), and the first two
        let mut lead_chunk = vec![-1; 256];
        lead_chunk.extend_from_slice(&base);
        let mut lead_two = vec![-1; 512];
        lead_two.extend_from_slice(&base[..300]);
        let cases: Vec<Vec<i32>> = vec![
            scattered,
            long_run,
            leading,
            shuffled,
            hole_chunk,
            lead_chunk,
            lead_two,
            vec![-1; 300],
            vec![-1; 1],
            vec![-1; 600],
        ];
        let rows: Vec<(usize, i32)> = cases.iter().map(|_| (2usize, 5500)).collect();
        let lists = cases.iter().cloned().map(Some).collect();
        check("holes".into(), rows, lists);
        // a mixed batch: deep sparse rows, dense rows, a padding row, a row past the limit whose meta says dense
        let rows = vec![
            (2usize, 8999i32),
            (1, 599),
            (0, 2999),
            (2, -1),
            (0, 100),
            (2, 2048),
            (1, 0),
            (0, 2047),
            (2, 3000),
            (3, 2099),
            (0, 2500),
        ];
        let lists =
            rows.iter().enumerate().map(|(k, &(_, p))| if p < 0 || k == 10 { None } else { list_of(p as usize, 8) }).collect();
        check("mixed batch".into(), rows, lists);
    }
    eprintln!("{compared} row results of the fused attention bit-identical to the two-kernel path (2 dtypes)");
}

/// The fused kernel against the COMMITTED golden of the decode kernels (`testdata/gpu/golden_mla.txt`): the `o_lat` hash of every dense
/// case below 2,048 tokens and of every sparse case in that file, recomputed by the fused kernel, equals the committed hash. Ties the new
/// kernel to the file that pins the decode path, independent of the in-process two-kernel comparison above.
#[test]
fn fused_attention_reproduces_the_committed_decode_golden() {
    if gpu_pf().is_none() {
        return;
    }
    let path = common::repo_root().join("testdata/gpu/golden_mla.txt");
    let golden = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let o_lat_of = |prefix: &str| -> String {
        let l = golden.lines().find(|l| l.starts_with(prefix)).unwrap_or_else(|| panic!("no golden line starts with {prefix:?}"));
        let at = l.find("o_lat ").unwrap() + 6;
        l[at..at + 16].to_string()
    };
    let mut n = 0;
    for dt in DTYPES {
        let f = Fmt(dt);
        let lens = [1usize, 255, 257, 1000, 2048, 2051, 3000];
        let w = make_world(dt, seqs_of(f, &lens), 11, 3);
        for (si, &len) in lens.iter().enumerate() {
            for (name, rows) in [
                ("last", vec![len as i32 - 1]),
                ("first", vec![0]),
                ("win2", (len.saturating_sub(2)..len).map(|p| p as i32).collect::<Vec<_>>()),
                ("win8", (len.saturating_sub(8)..len).map(|p| p as i32).collect::<Vec<_>>()),
                ("win9", (len.saturating_sub(9)..len).map(|p| p as i32).collect::<Vec<_>>()),
            ] {
                if rows.iter().any(|&p| p as usize >= DENSE_MAX_KEYS) {
                    continue; // the golden's dense pipeline runs these rows dense; the model runs them sparse
                }
                let r: Vec<(usize, i32)> = rows.iter().map(|&p| (si, p)).collect();
                let st = run_pf(&w, &r, &vec![None; r.len()]);
                let want = o_lat_of(&format!("{dt:?} len={len} {name} rows={} ", r.len()));
                assert_eq!(format!("{:016x}", hash16(&st.o_lat)), want, "{dt:?} len={len} {name}");
                n += 1;
            }
        }
        let sw = make_world(dt, seqs_of(f, &[3000, 5000]), 12, 3);
        let cases: Vec<GoldenCase> = vec![
            ("sparse2048".to_string(), vec![(0, 2048)], vec![Some(dsa_list(2048, 21))]),
            ("sparse4999".to_string(), vec![(1, 4999)], vec![Some(dsa_list(4999, 21))]),
            (
                "straddle8".to_string(),
                (2044..2052).map(|p| (0usize, p)).collect(),
                (2044..2052usize).map(|p| if p < 2048 { None } else { Some(dsa_list(p, 22)) }).collect(),
            ),
            ("list257".to_string(), vec![(1, 4000)], vec![Some(rand_list(4000, 257, 23))]),
            ("list2051".to_string(), vec![(1, 4500)], vec![Some(rand_list(4500, 2051, 23))]),
            (
                "mixed6".to_string(),
                vec![(1, 4999), (0, 100), (0, 2999), (1, 2048), (0, 2047), (1, 3000)],
                vec![
                    Some(dsa_list(4999, 24)),
                    None,
                    Some(dsa_list(2999, 24)),
                    Some(dsa_list(2048, 24)),
                    None,
                    Some(dsa_list(3000, 24)),
                ],
            ),
        ];
        for (name, rows, lists) in cases {
            let st = run_pf(&sw, &rows, &lists);
            let want = o_lat_of(&format!("{dt:?} {name} rows={} ", rows.len()));
            assert_eq!(format!("{:016x}", hash16(&st.o_lat)), want, "{dt:?} {name}");
            n += 1;
        }
    }
    eprintln!("{n} cases of the fused attention reproduce the committed decode golden's o_lat hashes");
}

/// The fused kernel's per-row bits do not depend on the other rows of the slab, on how many there are or on their order (it is the path
/// that serves 128-row slabs and the last, short one).
#[test]
fn fused_attention_is_row_count_and_neighbour_invariant() {
    if gpu_pf().is_none() {
        return;
    }
    for dt in DTYPES {
        let f = Fmt(dt);
        let w = make_world(dt, seqs_of(f, &[2400, 7000]), 9, 3);
        let list_of = |p: usize| if p < DENSE_MAX_KEYS { None } else { Some(dsa_list(p, 5)) };
        let rows: Vec<(usize, i32)> = (0..128).map(|k| (1usize, 1990 + k * 3)).collect();
        let lists: Vec<Option<Vec<i32>>> = rows.iter().map(|&(_, p)| list_of(p as usize)).collect();
        let all = run_pf(&w, &rows, &lists);
        for k in (0..128).step_by(9).chain([127]) {
            let one = run_pf(&w, &[rows[k]], &[lists[k].clone()]);
            assert_bits_eq(&format!("{dt:?} row {k} alone"), &all.o_lat[k * ROW_Q..(k + 1) * ROW_Q], &one.o_lat, LATENT_DIM);
        }
        // a suffix of the slab (rows 40..128) and a reversed order give the same bits per row
        let sub = run_pf(&w, &rows[40..], &lists[40..]);
        assert_bits_eq(&format!("{dt:?} suffix"), &all.o_lat[40 * ROW_Q..], &sub.o_lat, LATENT_DIM);
        let rev_rows: Vec<_> = rows.iter().rev().cloned().collect();
        let rev_lists: Vec<_> = lists.iter().rev().cloned().collect();
        let rev = run_pf(&w, &rev_rows, &rev_lists);
        for k in 0..128 {
            assert_bits_eq(
                &format!("{dt:?} reversed row {k}"),
                &all.o_lat[k * ROW_Q..(k + 1) * ROW_Q],
                &rev.o_lat[(127 - k) * ROW_Q..(128 - k) * ROW_Q],
                LATENT_DIM,
            );
        }
    }
}

/// Time of a 128-row slab, the two-kernel path against the fused one (`REXL3_BENCH_CTX`: positions of the slab's first row, default
/// 0,1920,2048,6016), with real-shaped selections.
#[test]
#[ignore]
fn bench_attend_pf() {
    if gpu_pf().is_none() {
        return;
    }
    let ctxs: Vec<usize> = std::env::var("REXL3_BENCH_CTX")
        .map(|v| v.split(',').map(|x| x.trim().parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![0, 1920, 2048, 6016]);
    let dt = Dtype::F16;
    let f = Fmt(dt);
    let wt = weights(dt);
    let rows = 128usize;
    for first in ctxs {
        let ctx = first + rows;
        let (cache, table, stream) = bench_world(dt, ctx);
        let pos: Vec<i32> = (0..rows).map(|r| (first + r) as i32).collect();
        let ri = RowIndex::from_host(&vec![0; rows], &pos).unwrap();
        let lists: Vec<Option<Vec<i32>>> =
            pos.iter().map(|&p| if (p as usize) < DENSE_MAX_KEYS { None } else { Some(dsa_list(p as usize, 3)) }).collect();
        let sel = make_sel(&lists);
        let q: Vec<u16> = (0..rows).flat_map(|r| qvec(f, 1, first + r)).collect();
        let qd = DevBuf::from_slice(&q).unwrap();
        let mut q_lat = DevBuf::zeroed(rows * ROW_Q * 2).unwrap();
        let mut o_lat = DevBuf::zeroed(rows * ROW_Q * 2).unwrap();
        let mut ws = DevBuf::zeroed(attend_auto_workspace_bytes(rows, ctx)).unwrap();
        absorb(&qd, N_HEADS * NOPE_DIM, &wt, &mut q_lat, rows, &stream).unwrap();
        let t_two = time_ms(&stream, 30, || {
            attend_auto(&q_lat, &cache, &table, &ri, rows, &sel.view(), &mut o_lat, ctx, &mut ws, &stream).unwrap()
        });
        let t_pf =
            time_ms(&stream, 30, || attend_pf(&q_lat, &cache, &table, &ri, rows, &sel.view(), &mut o_lat, &stream).unwrap());
        eprintln!(
            "slab of {rows} rows from position {first:>5}: two kernels {:7.0} us, fused {:7.0} us ({:.2}x)",
            t_two * 1e3,
            t_pf * 1e3,
            t_two / t_pf
        );
    }
}

// ---------------------------------------------------------------------------------------------------- the fp8 latent page (--kv-dtype fp8)

/// The CPU mirror of `csrc/mla_kv8.cuh` (`kv8`): the page's size, the flag, the E4M3 encoder (every code to itself, every midpoint to the
/// even neighbour, saturation at +-448), a zero row (scale 0, zero bytes), and a row's round trip within half an E4M3 step of its value.
#[test]
fn fp8_host_quantiser_and_flag() {
    use gb_cuda::fp8_linear::e4m3_to_f32;
    assert_eq!(kv8::PAGE_BYTES, 132_096);
    assert_eq!(kv8::PAGE_BYTES % 16, 0);
    assert_eq!(LatentKv::Fp8.token_bytes(Dtype::F16), 516);
    assert_eq!(LatentKv::Plain.token_bytes(Dtype::F16), 1024);
    assert_eq!(LatentKv::parse("FP8"), Ok(LatentKv::Fp8));
    assert_eq!(LatentKv::parse("e4m3"), Ok(LatentKv::Fp8));
    assert_eq!(LatentKv::parse(" fp16 "), Ok(LatentKv::Plain));
    assert!(LatentKv::parse("int8").is_err() && LatentKv::parse("bf16").is_err());
    assert_eq!((LatentKv::Plain.name(), LatentKv::Fp8.name()), ("fp16", "fp8"));
    for b in 0u8..=0x7E {
        let v = e4m3_to_f32(b);
        assert_eq!(kv8::e4m3_rne(v), b, "code {b:#04x}");
        assert_eq!(kv8::e4m3_rne(-v), b | 0x80, "code -{b:#04x}");
        if b < 0x7E {
            let mid = (v + e4m3_to_f32(b + 1)) / 2.0; // exact: one more bit than E4M3 has
            let even = if b & 1 == 0 { b } else { b + 1 };
            assert_eq!(kv8::e4m3_rne(mid), even, "midpoint above {b:#04x}");
            assert_eq!(kv8::e4m3_rne(f32::from_bits(mid.to_bits() + 1)), b + 1, "just above the midpoint above {b:#04x}");
            assert_eq!(kv8::e4m3_rne(f32::from_bits(mid.to_bits() - 1)), b, "just below the midpoint above {b:#04x}");
        }
    }
    for x in [448.0f32, 463.9, 464.0, 1e9, f32::INFINITY] {
        assert_eq!(kv8::e4m3_rne(x), 0x7E, "{x}");
        assert_eq!(kv8::e4m3_rne(-x), 0xFE, "-{x}");
    }
    assert_eq!(kv8::quantise_row(&[0.0; LATENT_DIM]), (vec![0u8; LATENT_DIM], 0.0));
    let mut r = Rng(0xF8);
    for dt in DTYPES {
        let f = Fmt(dt);
        for amp in [1e-3f32, 0.7, 3.0, 40.0] {
            let row: Vec<u16> = (0..LATENT_DIM).map(|_| f.enc(r.uniform(-1.0, 1.0) * amp)).collect();
            let x = f.decs(&row);
            let (q, s) = kv8::quantise_row(&x);
            let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert_eq!(s, amax / 448.0);
            assert!(q.iter().any(|&b| b & 0x7F == 0x7E), "the largest value takes the largest code");
            let back = f.decs(&kv8::round_trip(&row, dt));
            for (i, (&a, &b)) in x.iter().zip(&back).enumerate() {
                // half an E4M3 step (2^-4 relative, 2^-10 s in the subnormals), then one T rounding
                let tol = a.abs() / 16.0 + s / 1024.0 + a.abs() / 128.0;
                assert!((a - b).abs() <= tol, "{dt:?} amp {amp} [{i}]: {a} -> {b}");
            }
        }
    }
}

/// The two-cache comparison: the same sequences written to an fp8 cache, and as their host round trip to a T cache.
fn fp8_and_dequantised(dt: Dtype, lens: &[usize], placement: u64) -> (World, World) {
    let f = Fmt(dt);
    let w8 = make_world_kv(dt, seqs_of(f, lens), placement, 3, LatentKv::Fp8);
    let deq: Vec<Seq> = w8
        .seqs
        .iter()
        .map(|s| Seq { sid: s.sid, len: s.len, lat: s.lat.chunks(LATENT_DIM).flat_map(|r| kv8::round_trip(r, dt)).collect() })
        .collect();
    let w16 = make_world_kv(dt, deq, placement, 3, LatentKv::Plain);
    (w8, w16)
}

/// The device quantiser is the host mirror byte for byte (every row's E4M3 bytes and scale bits, a zero row and a row with an outlier
/// included, over shuffled pages of a NaN-filled pool), `read_rows` gives what the kernels stage (the host round trip), and a cache
/// written in windows of 7 rows (decode, verify) holds exactly the bytes one launch of every row (a prefill chunk) wrote.
#[test]
fn fp8_latent_write_is_the_host_quantiser() {
    let Some(stream) = gpu() else { return };
    for dt in DTYPES {
        let f = Fmt(dt);
        let mut seqs = seqs_of(f, &[700, 300]);
        seqs[0].lat[5 * LATENT_DIM..6 * LATENT_DIM].fill(0);
        seqs[1].lat[7 * LATENT_DIM + 3] = f.enc(300.0);
        let lats: Vec<Vec<u16>> = seqs.iter().map(|s| s.lat.clone()).collect();
        let w = make_world_kv(dt, seqs, 4, 2, LatentKv::Fp8);
        let raw: Vec<u8> = w.cache.lat.to_vec().unwrap();
        let flat: Vec<i32> = w.table.buf.to_vec().unwrap();
        for (si, s) in w.seqs.iter().enumerate() {
            let blocks: Vec<u32> = flat[si * w.table.max_blocks..][..s.len.div_ceil(BLOCK_TOKENS)].iter().map(|&b| b as u32).collect();
            for p in 0..s.len {
                let base = blocks[p / BLOCK_TOKENS] as usize * kv8::PAGE_BYTES;
                let (q, sc) = kv8::quantise_row(&f.decs(&lats[si][p * LATENT_DIM..(p + 1) * LATENT_DIM]));
                assert_eq!(&raw[base + (p % BLOCK_TOKENS) * LATENT_DIM..][..LATENT_DIM], &q[..], "{dt:?} seq {si} pos {p}: bytes");
                let got = f32::from_le_bytes(raw[base + kv8::SCALE_OFF + (p % BLOCK_TOKENS) * 4..][..4].try_into().unwrap());
                assert_eq!(got.to_bits(), sc.to_bits(), "{dt:?} seq {si} pos {p}: scale");
            }
            let want: Vec<u16> = lats[si].chunks(LATENT_DIM).flat_map(|r| kv8::round_trip(r, dt)).collect();
            assert_eq!(w.cache.read_rows(&blocks, s.len).unwrap(), want, "{dt:?} seq {si}: read_rows");
        }
        let mut c = LatentCache::new(w.cache.n_blocks, dt, LatentKv::Fp8).unwrap();
        c.lat.memset_async(0xFF, &stream).unwrap();
        for (si, s) in w.seqs.iter().enumerate() {
            for p0 in (0..s.len).step_by(7) {
                let n = 7.min(s.len - p0);
                let ld = DevBuf::from_slice(&lats[si][p0 * LATENT_DIM..(p0 + n) * LATENT_DIM]).unwrap();
                let ri = RowIndex::from_host(&vec![si as i32; n], &(p0 as i32..(p0 + n) as i32).collect::<Vec<_>>()).unwrap();
                latent_write(&ld, &ri, &w.table, &mut c, n, &stream).unwrap();
            }
        }
        stream.sync().unwrap();
        assert!(c.lat.to_vec::<u8>().unwrap() == raw, "{dt:?}: windows of 7 rows wrote other bytes than one launch");
    }
}

/// The fp8 path's output is the T path's on the dequantised cache, bit for bit, at every attention entry point: the dense pipeline,
/// every tuning variant (one and two head tiles a CTA), the per-row regime across 2,048 with DSA selections, the sparse kernel alone
/// with holes and an unordered list, the fused prefill slab, and the head-split launches (32 heads, both regimes).
#[test]
fn fp8_attention_is_the_t_attention_on_the_dequantised_cache() {
    let Some(stream) = gpu() else { return };
    for dt in DTYPES {
        let (w8, w16) = fp8_and_dequantised(dt, &[3000, 600, 2100], 5);
        let eq = |what: &str, a: &[u16], b: &[u16]| assert_bits_eq(&format!("{dt:?} fp8 {what}"), a, b, LATENT_DIM);
        // dense pipeline
        let rows = [(0usize, 0i32), (0, 255), (0, 256), (0, 2047), (1, 599), (1, 300), (2, 1000), (0, -1)];
        let (a, b) = (run_pipeline(&w8, &rows), run_pipeline(&w16, &rows));
        eq("dense o_lat", &a.o_lat, &b.o_lat);
        eq("dense out", &a.out, &b.out);
        // tuning variants
        let n = rows.len();
        let ri = RowIndex::from_host(&rows.iter().map(|r| r.0 as i32).collect::<Vec<_>>(), &rows.iter().map(|r| r.1).collect::<Vec<_>>())
            .unwrap();
        let q_lat = DevBuf::from_slice(&a.q_lat).unwrap();
        let mut ws = DevBuf::alloc(attend_workspace_bytes(n, 2048) * 2).unwrap();
        for variant in 1..=VARIANTS {
            let mut outs = Vec::new();
            for w in [&w8, &w16] {
                let mut o = DevBuf::zeroed(n * ROW_Q * 2).unwrap();
                attend_variant(&q_lat, &w.cache, &w.table, &ri, n, &mut o, 2048, &mut ws, variant, &stream).unwrap();
                stream.sync().unwrap();
                outs.push(o.to_vec::<u16>().unwrap());
            }
            eq(&format!("variant {variant}"), &outs[0], &outs[1]);
        }
        // the per-row regime and the fused slab
        let mut rows: Vec<(usize, i32)> = (0..40).map(|k| (0usize, 2030 + 20 * k)).collect();
        rows.extend([(2, 2099), (1, 599), (0, -1), (2, 2048)]);
        let lists: Vec<Option<Vec<i32>>> =
            rows.iter().map(|&(_, p)| if p < DENSE_MAX_KEYS as i32 { None } else { Some(dsa_list(p as usize, 7)) }).collect();
        let (a, b) = (run_auto(&w8, &rows, &lists, None), run_auto(&w16, &rows, &lists, None));
        eq("auto o_lat", &a.o_lat, &b.o_lat);
        eq("auto out", &a.out, &b.out);
        eq("fused slab", &run_pf(&w8, &rows, &lists).o_lat, &run_pf(&w16, &rows, &lists).o_lat);
        // the sparse kernel alone: holes, an unordered list, a list of holes only
        let base = rand_list(2999, 900, 3);
        let mut holes = base.clone();
        for i in (5..holes.len()).step_by(11) {
            holes[i] = -1;
        }
        let mut unordered = base.clone();
        shuffle(&mut unordered, &mut Rng(9));
        let sl = vec![Some(holes), Some(unordered), Some(vec![-1; 40]), Some(rand_list(2099, 2051, 4))];
        let srows = [(0usize, 2999), (0, 2999), (0, 2999), (2, 2099)];
        eq("sparse only", &run_sparse_only(&w8, &srows, &sl).o_lat, &run_sparse_only(&w16, &srows, &sl).o_lat);
        // the head split (lane E): heads 32..64 of the auto rows, both regimes
        let nh = 32;
        let n = rows.len();
        let q32: Vec<u16> = a.q_lat.chunks(ROW_Q).flat_map(|r| r[32 * LATENT_DIM..].to_vec()).collect();
        let q32 = DevBuf::from_slice(&q32).unwrap();
        let ri = RowIndex::from_host(&rows.iter().map(|r| r.0 as i32).collect::<Vec<_>>(), &rows.iter().map(|r| r.1).collect::<Vec<_>>())
            .unwrap();
        let sel = make_sel(&lists);
        let mut ws = DevBuf::alloc(attend_auto_workspace_bytes(n, 2900)).unwrap();
        let mut outs = Vec::new();
        for w in [&w8, &w16] {
            let mut o = DevBuf::from_slice(&vec![0xABCDu16; n * nh * LATENT_DIM]).unwrap();
            attend_below_heads(&q32, &w.cache, &w.table, &ri, n, &mut o, DENSE_MAX_KEYS, 2900, &mut ws, nh, &stream).unwrap();
            attend_sparse_heads(&q32, &w.cache, &w.table, &ri, n, &sel.view(), &mut o, &mut ws, nh, &stream).unwrap();
            stream.sync().unwrap();
            outs.push(o.to_vec::<u16>().unwrap());
        }
        eq("head split", &outs[0], &outs[1]);
        // and fp8 is not a no-op: the dequantised cache differs from the T values the sequences hold
        assert_ne!(w8.seqs[0].lat[..LATENT_DIM], w16.seqs[0].lat[..LATENT_DIM]);
    }
}

/// fp8 keeps the exactness contract: a run repeats bit for bit, a verify window's rows equal the same rows decoded alone, and the fused
/// prefill slab equals the decode path, across the 2,048 boundary.
#[test]
fn fp8_is_deterministic_and_every_path_gives_the_same_bits() {
    if gpu().is_none() {
        return;
    }
    for dt in DTYPES {
        let f = Fmt(dt);
        let w = make_world_kv(dt, seqs_of(f, &[2600, 900]), 6, 2, LatentKv::Fp8);
        let rows: Vec<(usize, i32)> = (0..9).map(|k| (0usize, 2040 + k)).chain([(1, 899), (0, 2599), (1, 17)]).collect();
        let lists: Vec<Option<Vec<i32>>> =
            rows.iter().map(|&(_, p)| if p < DENSE_MAX_KEYS as i32 { None } else { Some(dsa_list(p as usize, 11)) }).collect();
        let a = run_auto(&w, &rows, &lists, None);
        let again = run_auto(&w, &rows, &lists, None);
        assert_bits_eq(&format!("{dt:?} repeat o_lat"), &a.o_lat, &again.o_lat, LATENT_DIM);
        assert_bits_eq(&format!("{dt:?} repeat out"), &a.out, &again.out, NOPE_DIM);
        for k in 0..rows.len() {
            let one = run_auto(&w, &[rows[k]], &[lists[k].clone()], None);
            assert_bits_eq(&format!("{dt:?} row {k} alone"), &a.o_lat[k * ROW_Q..(k + 1) * ROW_Q], &one.o_lat, LATENT_DIM);
            assert_bits_eq(&format!("{dt:?} row {k} alone, out"), &a.out[k * ROW_OUT..(k + 1) * ROW_OUT], &one.out, NOPE_DIM);
        }
        assert_bits_eq(&format!("{dt:?} fused slab"), &run_pf(&w, &rows, &lists).o_lat, &a.o_lat, LATENT_DIM);
    }
}
