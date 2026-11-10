//! QSA kernels (`gb_cuda::qsa`) on synthetic data: a CPU reference for pooling, scores, selection and attention; row
//! invariance (a row's results do not depend on the batch it runs in, or on how a sequence was chunked); and the golden
//! hashes `scripts/gpu_gate.sh` compares:
//!
//!     REXL3_GOLDEN_OUT=golden_qsa.txt cargo test --release -p gb-cuda --test qsa golden_qsa_hashes -- --ignored
//!     REXL3_REQUIRE_GPU=1 cargo test --release -p gb-cuda --test qsa qsa_attend_parts_one_matches_window_parts -- --ignored --nocapture --test-threads=1
//!
//! GPU setup is optional in ordinary workspace runs (`not run` when CUDA/device/stream setup is unavailable); with `REXL3_REQUIRE_GPU=1`, any
//! unavailable device, context initialization failure or stream initialization failure fails the ignored GPU test instead of passing vacuously.
use gb_cuda::attention::{BlockTable, KvHeads, PagedKv, BLOCK_TOKENS, HEAD_DIM, N_Q_HEADS};

/// Flash-Next's KV heads: the cache the sparse attention reads.
const N_KV_HEADS: usize = KvHeads::Two.count();
use gb_cuda::qsa::{self, CR, DK, N_HEADS, QK, TOPK};
use gb_cuda::{DevBuf, Stream};

const RING: usize = 128;
const EPS: f32 = 1e-6;
const THETA: f32 = 1e7;

#[cfg(test)]
mod gpu_requirement_tests {
    use super::gpu_unavailable;

    #[test]
    fn required_gpu_mode_fails_when_gpu_is_unavailable() {
        let failure = std::panic::catch_unwind(|| {
            gpu_unavailable::<()>(true, "CUDA context initialization failed");
        })
        .expect_err("required GPU mode must fail closed");
        let message = failure
            .downcast_ref::<String>()
            .map(String::as_str)
            .unwrap_or_else(|| failure.downcast_ref::<&str>().copied().unwrap_or(""));
        assert!(message.contains("REXL3_REQUIRE_GPU=1"), "unexpected panic: {message}");
        assert!(message.contains("CUDA context initialization failed"), "unexpected panic: {message}");
    }

    #[test]
    fn optional_gpu_mode_returns_not_run() {
        assert!(gpu_unavailable::<()>(false, "no CUDA device").is_none());
    }
}

fn required_gpu() -> bool {
    std::env::var("REXL3_REQUIRE_GPU").is_ok_and(|value| value == "1")
}

fn gpu_unavailable<T>(required: bool, reason: &str) -> Option<T> {
    if required {
        panic!("REXL3_REQUIRE_GPU=1 but the QSA parity test cannot run: {reason}");
    }
    eprintln!("not run: QSA parity test needs a usable CUDA device and stream ({reason})");
    None
}

fn gpu() -> Option<Stream> {
    if !gb_cuda::device_available() {
        return gpu_unavailable(required_gpu(), "no CUDA device");
    }
    if let Err(error) = gb_cuda::rt::init_context() {
        return gpu_unavailable(required_gpu(), &format!("CUDA context initialization failed: {error}"));
    }
    match Stream::new() {
        Ok(stream) => Some(stream),
        Err(error) => gpu_unavailable(required_gpu(), &format!("CUDA stream initialization failed: {error}")),
    }
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

fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
fn bf2f(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}
fn f16_to_f32(h: u16) -> f32 {
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
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}
fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain data viewed as bytes
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Sequences with paged KV and the indexer planes filled the way the model fills them (stage, pool, ring per chunk).
struct Rig {
    slots: usize,
    table: BlockTable,
    tables: Vec<Vec<u32>>,
    kv: PagedKv,
    pool: DevBuf,
    ring: DevBuf,
    qw: DevBuf,
    kw: DevBuf,
    /// Per slot: the indexer projection rows (bf16 [len][640]) the planes were built from.
    qk: Vec<Vec<u16>>,
    k_host: Vec<u16>,
    v_host: Vec<u16>,
    pages: usize,
}

impl Rig {
    fn new(lens: &[usize], seed: u64, chunk: usize, st: &Stream) -> Self {
        let slots = lens.len();
        let max_ctx = lens.iter().max().unwrap() + 64;
        let max_blocks = max_ctx.div_ceil(BLOCK_TOKENS);
        let pages = slots * max_blocks;
        // pages handed out interleaved across slots, as a shared pool would
        let tables: Vec<Vec<u32>> =
            (0..slots).map(|s| (0..max_blocks).map(|i| (i * slots + (slots - 1 - s)) as u32).collect()).collect();
        let table = BlockTable::from_host(&tables, max_blocks).unwrap();
        let mut rng = Rng(seed);
        let n_kv = pages * BLOCK_TOKENS * N_KV_HEADS * HEAD_DIM;
        let k_host: Vec<u16> = (0..n_kv).map(|_| bf16(rng.uniform(-1.0, 1.0))).collect();
        let v_host: Vec<u16> = (0..n_kv).map(|_| bf16(rng.uniform(-1.0, 1.0))).collect();
        let mut kv = PagedKv::with_kv_heads(pages, KvHeads::Two).unwrap();
        kv.k.upload(&k_host).unwrap();
        kv.v.upload(&v_host).unwrap();
        let qw = DevBuf::from_slice(&(0..DK).map(|_| 1.0 + rng.uniform(-0.3, 0.3)).collect::<Vec<f32>>()).unwrap();
        let kw = DevBuf::from_slice(&(0..DK).map(|_| 1.0 + rng.uniform(-0.3, 0.3)).collect::<Vec<f32>>()).unwrap();
        let qk: Vec<Vec<u16>> = lens
            .iter()
            .map(|&n| {
                // the raw-key part carries a per-sequence bias so pooled keys are not all alike
                let bias: Vec<f32> = (0..DK).map(|_| rng.uniform(-0.5, 0.5)).collect();
                (0..n * QK)
                    .map(|i| {
                        bf16(rng.uniform(-2.0, 2.0) + if i % QK >= N_HEADS * DK { bias[i % QK - N_HEADS * DK] } else { 0.0 })
                    })
                    .collect()
            })
            .collect();
        let mut rig = Rig {
            slots,
            table,
            tables,
            kv,
            pool: DevBuf::zeroed(pages * qsa::PAGE_BLOCKS * DK * 2).unwrap(),
            ring: DevBuf::zeroed(slots * RING * DK * 2).unwrap(),
            qw,
            kw,
            qk,
            k_host,
            v_host,
            pages,
        };
        for (s, &len) in lens.iter().enumerate().take(slots) {
            let mut at = 0;
            while at < len {
                let n = chunk.min(len - at);
                rig.append(s, at, n, st);
                at += n;
            }
        }
        rig
    }

    /// Run stage + pool + ring for `n` rows of slot `s` from position `at` (one sequence a batch).
    fn append(&mut self, s: usize, at: usize, n: usize, st: &Stream) {
        let qk = DevBuf::from_slice(&self.qk[s][at * QK..(at + n) * QK]).unwrap();
        let pos: Vec<i32> = (at..at + n).map(|p| p as i32).collect();
        let pos = DevBuf::from_slice(&pos).unwrap();
        let (ss, sl) = (DevBuf::from_slice(&[0i32, n as i32]).unwrap(), DevBuf::from_slice(&[s as i32]).unwrap());
        let mut q = DevBuf::zeroed(n * N_HEADS * DK * 2).unwrap();
        let mut kraw = DevBuf::zeroed(n * DK * 2).unwrap();
        qsa::stage(&qk, &self.qw, &pos, &mut q, &mut kraw, n, EPS, THETA, st).unwrap();
        let seqs = qsa::Seqs { positions: &pos, seq_start: &ss, seq_slot: &sl, n_seq: 1, rows: n };
        qsa::pool(&kraw, &self.ring, &self.kw, &seqs, &self.table, &mut self.pool, self.pages, self.slots, RING, EPS, THETA, st)
            .unwrap();
        qsa::ring_store(&kraw, &mut self.ring, &seqs, self.slots, RING, st).unwrap();
        st.sync().unwrap();
    }

    /// Indexer queries (f16 bits) of the given rows, from `stage` run on exactly those rows.
    fn queries(&self, rows: &[(usize, usize)], st: &Stream) -> Vec<u16> {
        let qk: Vec<u16> = rows.iter().flat_map(|&(s, p)| self.qk[s][p * QK..(p + 1) * QK].iter().copied()).collect();
        let pos: Vec<i32> = rows.iter().map(|&(_, p)| p as i32).collect();
        let (qk, pos) = (DevBuf::from_slice(&qk).unwrap(), DevBuf::from_slice(&pos).unwrap());
        let n = rows.len();
        let mut q = DevBuf::zeroed(n * N_HEADS * DK * 2).unwrap();
        let mut kraw = DevBuf::zeroed(n * DK * 2).unwrap();
        qsa::stage(&qk, &self.qw, &pos, &mut q, &mut kraw, n, EPS, THETA, st).unwrap();
        st.sync().unwrap();
        q.to_vec::<u16>().unwrap()
    }

    /// scores [rows][nb_cap], selections [rows][512] and counts for `rows` with indexer queries `q`.
    fn select(&self, rows: &[(usize, usize)], q: &[u16], st: &Stream) -> (Vec<f32>, Vec<i32>, Vec<i32>, usize) {
        let n = rows.len();
        let nb_cap = rows.iter().map(|&(_, p)| (p + 1) / CR).max().unwrap().max(1);
        let slot: Vec<i32> = rows.iter().map(|&(s, _)| s as i32).collect();
        let pos: Vec<i32> = rows.iter().map(|&(_, p)| p as i32).collect();
        let (slot, pos, qd) =
            (DevBuf::from_slice(&slot).unwrap(), DevBuf::from_slice(&pos).unwrap(), DevBuf::from_slice(q).unwrap());
        let rr = qsa::Rows { slot: &slot, pos: &pos, rows: n };
        let mut sc = DevBuf::zeroed(n * nb_cap * 4).unwrap();
        let (mut sel, mut sel_n) = (DevBuf::zeroed(n * TOPK * 4).unwrap(), DevBuf::zeroed(n * 4).unwrap());
        qsa::scores(&qd, &self.pool, self.pages, &self.table, &rr, &mut sc, nb_cap, st).unwrap();
        qsa::topk(&sc, &rr, nb_cap, &mut sel, &mut sel_n, st).unwrap();
        st.sync().unwrap();
        (sc.to_vec().unwrap(), sel.to_vec().unwrap(), sel_n.to_vec().unwrap(), nb_cap)
    }

    /// Sparse attention outputs (bf16 bits [rows][24][256]) for main-attention queries `qm`.
    fn attend(&self, rows: &[(usize, usize)], qm: &[u16], sel: &[i32], sel_n: &[i32], parts: usize, st: &Stream) -> Vec<u16> {
        let n = rows.len();
        let slot: Vec<i32> = rows.iter().map(|&(s, _)| s as i32).collect();
        let pos: Vec<i32> = rows.iter().map(|&(_, p)| p as i32).collect();
        let (slot, pos) = (DevBuf::from_slice(&slot).unwrap(), DevBuf::from_slice(&pos).unwrap());
        let rr = qsa::Rows { slot: &slot, pos: &pos, rows: n };
        let (q, sd, nd) = (DevBuf::from_slice(qm).unwrap(), DevBuf::from_slice(sel).unwrap(), DevBuf::from_slice(sel_n).unwrap());
        let mut ws = DevBuf::zeroed(qsa::attend_workspace_bytes(n, parts)).unwrap();
        let mut out = DevBuf::zeroed(n * N_Q_HEADS * HEAD_DIM * 2).unwrap();
        qsa::attend(&q, &self.kv, &self.table, &rr, &sd, &nd, parts, &mut ws, &mut out, st).unwrap();
        st.sync().unwrap();
        out.to_vec().unwrap()
    }

    fn pool_key(&self, plane: &[u16], s: usize, j: usize) -> Vec<f32> {
        let page = self.tables[s][j / qsa::PAGE_BLOCKS] as usize;
        let at = (page * qsa::PAGE_BLOCKS + j % qsa::PAGE_BLOCKS) * DK;
        plane[at..at + DK].iter().map(|&h| f16_to_f32(h)).collect()
    }

    fn kv_row(&self, v: bool, s: usize, tok: usize, kvh: usize) -> &[u16] {
        let page = self.tables[s][tok / BLOCK_TOKENS] as usize;
        let at = ((page * BLOCK_TOKENS + tok % BLOCK_TOKENS) * N_KV_HEADS + kvh) * HEAD_DIM;
        let src = if v { &self.v_host } else { &self.k_host };
        &src[at..at + HEAD_DIM]
    }
}

// ---- CPU reference ----------------------------------------------------------------------------------------------------

fn rope(x: &mut [f32], pos: usize) {
    let half = 32;
    for i in 0..half {
        let inv = (THETA as f64).powf(-((2 * i) as f64) / 64.0) as f32;
        let (sn, cs) = (pos as f32 * inv).sin_cos();
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * cs - b * sn;
        x[i + half] = b * cs + a * sn;
    }
}

fn rms(x: &[f32], w: &[f32]) -> Vec<f32> {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ss + EPS).sqrt();
    x.iter().zip(w).map(|(v, w)| v * inv * w).collect()
}

fn ref_pool_key(rig: &Rig, kw: &[f32], s: usize, j: usize) -> Vec<f32> {
    let mut m = vec![0f32; DK];
    for t in 0..CR {
        let p = j * CR + t;
        for (d, md) in m.iter_mut().enumerate() {
            *md += bf2f(rig.qk[s][p * QK + N_HEADS * DK + d]);
        }
    }
    let m: Vec<f32> = m.iter().map(|v| v * 0.25).collect();
    let mut y = rms(&m, kw);
    rope(&mut y, j * CR);
    y
}

fn ref_scores(rig: &Rig, plane: &[u16], q: &[u16], r: usize, s: usize, p: usize) -> Vec<f32> {
    let nbq = (p + 1) / CR;
    let qf: Vec<f32> = q[r * N_HEADS * DK..(r + 1) * N_HEADS * DK].iter().map(|&h| f16_to_f32(h)).collect();
    (0..nbq)
        .map(|j| {
            let k = rig.pool_key(plane, s, j);
            let mut sc = 0f32;
            for h in 0..N_HEADS {
                let d: f32 = (0..DK).map(|i| qf[h * DK + i] * k[i]).sum();
                sc += d.max(0.0);
            }
            sc / (DK as f32).sqrt()
        })
        .collect()
}

/// Top-512 of `sc` (ties: lower index first), ascending; every block when there are at most 512.
fn ref_topk(sc: &[f32]) -> Vec<i32> {
    let mut idx: Vec<usize> = (0..sc.len()).collect();
    if sc.len() > TOPK {
        idx.sort_by(|&a, &b| sc[b].partial_cmp(&sc[a]).unwrap().then(a.cmp(&b)));
        idx.truncate(TOPK);
        idx.sort();
    }
    idx.into_iter().map(|i| i as i32).collect()
}

fn ref_attend(rig: &Rig, qm: &[u16], r: usize, s: usize, p: usize, sel: &[i32]) -> Vec<f32> {
    let nbq = (p + 1) / CR;
    let mut toks: Vec<usize> = sel.iter().flat_map(|&b| (0..CR).map(move |t| b as usize * CR + t)).collect();
    toks.extend(nbq * CR..=p);
    let mut out = vec![0f32; N_Q_HEADS * HEAD_DIM];
    for h in 0..N_Q_HEADS {
        let kvh = h / (N_Q_HEADS / N_KV_HEADS);
        let q: Vec<f32> = qm[(r * N_Q_HEADS + h) * HEAD_DIM..][..HEAD_DIM].iter().map(|&x| bf2f(x)).collect();
        let sc: Vec<f32> = toks
            .iter()
            .map(|&t| rig.kv_row(false, s, t, kvh).iter().zip(&q).map(|(&k, q)| bf2f(k) * q).sum::<f32>() / 16.0)
            .collect();
        let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = sc.iter().map(|v| (v - mx).exp()).collect();
        let l: f32 = e.iter().sum();
        for (t, &w) in toks.iter().zip(&e) {
            for (o, &v) in out[h * HEAD_DIM..(h + 1) * HEAD_DIM].iter_mut().zip(rig.kv_row(true, s, *t, kvh)) {
                *o += w / l * bf2f(v);
            }
        }
    }
    out
}

fn main_queries(n: usize, seed: u64) -> Vec<u16> {
    let mut rng = Rng(seed);
    (0..n * N_Q_HEADS * HEAD_DIM).map(|_| bf16(rng.uniform(-3.0, 3.0))).collect()
}

const LENS: [usize; 3] = [2600, 4111, 9000];

fn row_sets() -> Vec<(&'static str, Vec<(usize, usize)>)> {
    vec![
        ("R1 slot0 p2599", vec![(0, 2599)]),
        ("R1 slot2 p8999", vec![(2, 8999)]),
        ("R3 slot1 around the threshold", vec![(1, 2049), (1, 2050), (1, 2051)]),
        ("R9 slot1 p4100", (0..9).map(|i| (1, 4100 + i)).collect()),
        ("mixed slots", vec![(0, 2047), (0, 2048), (0, 2599), (2, 8990), (2, 8991), (1, 2050)]),
        ("R40 slot2", (0..40).map(|i| (2, 8950 + i)).collect()),
    ]
}

#[test]
#[ignore]
fn qsa_matches_cpu_reference() {
    let Some(st) = gpu() else { return };
    let rig = Rig::new(&LENS, 7, 97, &st);
    let kw: Vec<f32> = rig.kw.to_vec().unwrap();
    let plane: Vec<u16> = rig.pool.to_vec().unwrap();
    // pooled keys
    let mut worst = 0f32;
    for (s, &len) in LENS.iter().enumerate() {
        for j in (0..len / CR).step_by(37) {
            let (g, w) = (rig.pool_key(&plane, s, j), ref_pool_key(&rig, &kw, s, j));
            for (a, b) in g.iter().zip(&w) {
                worst = worst.max((a - b).abs() / (1.0 + b.abs()));
            }
        }
    }
    eprintln!("pool keys: worst relative error {worst:e}");
    assert!(worst < 4e-3, "pooled keys off by {worst}");
    for (name, rows) in row_sets() {
        let q = rig.queries(&rows, &st);
        let (sc, sel, sel_n, nb_cap) = rig.select(&rows, &q, &st);
        let qm = main_queries(rows.len(), 99);
        let out = rig.attend(&rows, &qm, &sel, &sel_n, qsa::WINDOW_PARTS, &st);
        let (mut serr, mut aerr) = (0f32, 0f64);
        for (r, &(s, p)) in rows.iter().enumerate() {
            let nbq = (p + 1) / CR;
            let want = ref_scores(&rig, &plane, &q, r, s, p);
            for (j, w) in want.iter().enumerate() {
                serr = serr.max((sc[r * nb_cap + j] - w).abs() / (1e-3 + w.abs()));
            }
            // selection: exactly the top-512 of the kernel's own scores (ties to the lower index)
            let got: Vec<i32> = sel[r * TOPK..r * TOPK + sel_n[r] as usize].to_vec();
            assert_eq!(got, ref_topk(&sc[r * nb_cap..r * nb_cap + nbq]), "{name}: row {r} selection");
            let o = ref_attend(&rig, &qm, r, s, p, &got);
            let g: Vec<f32> = out[r * N_Q_HEADS * HEAD_DIM..(r + 1) * N_Q_HEADS * HEAD_DIM].iter().map(|&x| bf2f(x)).collect();
            let (num, den): (f64, f64) =
                g.iter().zip(&o).fold((0.0, 0.0), |(n, d), (a, b)| (n + ((a - b) as f64).powi(2), d + (*b as f64).powi(2)));
            aerr = aerr.max((num / den).sqrt());
        }
        eprintln!("{name}: scores worst rel {serr:e}, attention worst rel L2 {aerr:e}");
        assert!(serr < 1e-4, "{name}: scores off by {serr}");
        assert!(aerr < 1e-2, "{name}: attention rel L2 {aerr}");
    }
}

#[test]
#[ignore]
fn qsa_rows_are_invariant() {
    let Some(st) = gpu() else { return };
    // the planes do not depend on how a sequence was chunked
    let a = Rig::new(&LENS, 7, 97, &st);
    let b = Rig::new(&LENS, 7, 1000, &st);
    let c = Rig::new(&LENS, 7, 5, &st);
    let (pa, pb, pc) = (a.pool.to_vec::<u16>().unwrap(), b.pool.to_vec::<u16>().unwrap(), c.pool.to_vec::<u16>().unwrap());
    assert!(pa == pb && pa == pc, "pool plane depends on the chunking");
    // a row's scores, selection and attention do not depend on the rows it runs with
    for (name, rows) in row_sets() {
        let q = a.queries(&rows, &st);
        let qm = main_queries(rows.len(), 99);
        let (sc, sel, sel_n, nb_cap) = a.select(&rows, &q, &st);
        let out = a.attend(&rows, &qm, &sel, &sel_n, qsa::WINDOW_PARTS, &st);
        for (r, &row) in rows.iter().enumerate() {
            let q1 = a.queries(&[row], &st);
            assert_eq!(q1[..], q[r * N_HEADS * DK..(r + 1) * N_HEADS * DK], "{name}: row {r} queries");
            let (sc1, sel1, n1, cap1) = a.select(&[row], &q1, &st);
            let nbq = (row.1 + 1) / CR;
            assert_eq!(sc1[..nbq], sc[r * nb_cap..r * nb_cap + nbq], "{name}: row {r} scores");
            let _ = cap1;
            assert_eq!(n1[0], sel_n[r], "{name}: row {r} selection size");
            assert_eq!(sel1[..n1[0] as usize], sel[r * TOPK..r * TOPK + n1[0] as usize], "{name}: row {r} selection");
            let qm1 = &qm[r * N_Q_HEADS * HEAD_DIM..(r + 1) * N_Q_HEADS * HEAD_DIM];
            let o1 = a.attend(&[row], qm1, &sel1, &n1, qsa::WINDOW_PARTS, &st);
            assert_eq!(o1[..], out[r * N_Q_HEADS * HEAD_DIM..(r + 1) * N_Q_HEADS * HEAD_DIM], "{name}: row {r} attention");
        }
    }
}

/// Golden hashes of the QSA kernels on synthetic data (pool planes, selections, attention).
///     REXL3_GOLDEN_OUT=golden_qsa.txt cargo test --release -p gb-cuda --test qsa golden_qsa_hashes -- --ignored
/// A prefill chunk's rows (64 and more) take the four-heads-a-warp kernel, a window's few rows the twelve-warp one: every row's bf16 output
/// is the same either way, bit for bit, at one part and at the window's split.
#[test]
#[ignore]
fn qsa_attend_many_rows_equal_one_row_calls_bit_for_bit() {
    let Some(st) = gpu() else { return };
    let rig = Rig::new(&LENS, 7, 97, &st);
    for rows in [
        (0..80).map(|i| (2, 8900 + i)).collect::<Vec<_>>(),
        (0..70).map(|i| (1, 2040 + 29 * i)).collect::<Vec<_>>(),
        (0..65).map(|i| (1 + i % 2, 2100 + 20 * i)).collect::<Vec<_>>(),
    ] {
        let rows: Vec<(usize, usize)> = rows.into_iter().filter(|&(s, p)| p < LENS[s]).collect();
        assert!(rows.len() >= 64, "the chunk-sized kernel needs 64 rows or more");
        let q = rig.queries(&rows, &st);
        let (_, sel, sel_n, _) = rig.select(&rows, &q, &st);
        let qm = main_queries(rows.len(), 123);
        for parts in [1, qsa::WINDOW_PARTS] {
            let many = rig.attend(&rows, &qm, &sel, &sel_n, parts, &st);
            let width = N_Q_HEADS * HEAD_DIM;
            for r in 0..rows.len() {
                let one = rig.attend(&rows[r..r + 1], &qm[r * width..(r + 1) * width], &sel[r * TOPK..(r + 1) * TOPK], &sel_n[r..r + 1], parts, &st);
                assert_eq!(many[r * width..(r + 1) * width], one[..], "row {r} {:?} parts {parts}", rows[r]);
            }
        }
    }
}

#[test]
#[ignore]
fn qsa_attend_parts_one_matches_window_parts() {
    let Some(st) = gpu() else { return };
    let rig = Rig::new(&LENS, 7, 97, &st);
    // Include an off-grid batch and one-row tail, with identical queries, cache, and selection in both arms.
    for rows in [vec![(1, 2050), (1, 4100), (2, 8999)], vec![(2, 8999)]] {
        let q = rig.queries(&rows, &st);
        let (_, sel, sel_n, _) = rig.select(&rows, &q, &st);
        let qm = main_queries(rows.len(), 99);
        let one = rig.attend(&rows, &qm, &sel, &sel_n, 1, &st);
        let window = rig.attend(&rows, &qm, &sel, &sel_n, qsa::WINDOW_PARTS, &st);
        let mismatches = one.iter().zip(&window).filter(|(a, b)| a != b).count();
        eprintln!("parts=1 vs {}: {mismatches}/{} bf16 elements differ", qsa::WINDOW_PARTS, one.len());
        let mut pair_l2 = 0.0f64;
        let (mut one_ref_l2, mut window_ref_l2) = (0.0f64, 0.0f64);
        for (r, &(slot, pos)) in rows.iter().enumerate() {
            let selected = &sel[r * TOPK..r * TOPK + sel_n[r] as usize];
            let reference = ref_attend(&rig, &qm, r, slot, pos, selected);
            let start = r * N_Q_HEADS * HEAD_DIM;
            let end = start + N_Q_HEADS * HEAD_DIM;
            let one_row: Vec<f32> = one[start..end].iter().map(|&x| bf2f(x)).collect();
            let window_row: Vec<f32> = window[start..end].iter().map(|&x| bf2f(x)).collect();
            let rel_l2 = |got: &[f32], want: &[f32]| {
                let (num, den) = got
                    .iter()
                    .zip(want)
                    .fold((0.0f64, 0.0f64), |(n, d), (a, b)| (n + ((*a as f64 - *b as f64).powi(2)), d + (*b as f64).powi(2)));
                (num / den).sqrt()
            };
            pair_l2 = pair_l2.max(rel_l2(&one_row, &window_row));
            one_ref_l2 = one_ref_l2.max(rel_l2(&one_row, &reference));
            window_ref_l2 = window_ref_l2.max(rel_l2(&window_row, &reference));
        }
        eprintln!(
            "parts=1 vs {}: relative L2 pair={pair_l2:e}, parts=1 oracle={one_ref_l2:e}, parts={} oracle={window_ref_l2:e}",
            qsa::WINDOW_PARTS,
            qsa::WINDOW_PARTS
        );
        assert!(pair_l2 < 1e-3, "parts changed QSA attention beyond the split-reduction error budget: {pair_l2}");
        assert!(one_ref_l2 < 1e-2, "parts=1 attention rel L2 to CPU reference {one_ref_l2}");
        assert!(window_ref_l2 < 1e-2, "parts={} attention rel L2 to CPU reference {window_ref_l2}", qsa::WINDOW_PARTS);
    }
}

/// The golden text (hashes of the pool planes, the selections and the attention outputs) of the QSA kernels on synthetic data.
fn golden_text(st: &Stream) -> String {
    let rig = Rig::new(&LENS, 7, 97, st);
    let plane: Vec<u16> = rig.pool.to_vec().unwrap();
    let mut out = String::new();
    for (s, &len) in LENS.iter().enumerate() {
        let keys: Vec<u16> = (0..len / CR)
            .flat_map(|j| {
                let page = rig.tables[s][j / qsa::PAGE_BLOCKS] as usize;
                let at = (page * qsa::PAGE_BLOCKS + j % qsa::PAGE_BLOCKS) * DK;
                plane[at..at + DK].to_vec()
            })
            .collect();
        out += &format!("slot {s} pool plane ({} blocks) {:016x}\n", len / CR, fnv(bytes_of(&keys)));
    }
    for (name, rows) in row_sets() {
        let q = rig.queries(&rows, st);
        let (_, sel, sel_n, _) = rig.select(&rows, &q, st);
        let qm = main_queries(rows.len(), 99);
        let o = rig.attend(&rows, &qm, &sel, &sel_n, qsa::WINDOW_PARTS, st);
        let o1 = rig.attend(&rows, &qm, &sel, &sel_n, 1, st);
        out += &format!("{name} queries {:016x}\n", fnv(bytes_of(&q)));
        out += &format!("{name} selection {:016x} sizes {:?}\n", fnv(bytes_of(&sel)), sel_n);
        out += &format!("{name} attention (window parts) {:016x}\n", fnv(bytes_of(&o)));
        out += &format!("{name} attention (one part) {:016x}\n", fnv(bytes_of(&o1)));
    }
    out
}

#[test]
#[ignore]
fn golden_qsa_hashes() {
    use std::io::Write;
    let Some(st) = gpu() else { return };
    let path = std::env::var("REXL3_GOLDEN_OUT").expect("REXL3_GOLDEN_OUT");
    std::fs::File::create(path).unwrap().write_all(golden_text(&st).as_bytes()).unwrap();
}

/// FNV-1a of `golden_text` as the kernels were before they carried the programmatic-launch attribute.
const QSA_GOLDEN_HASH: u64 = 0xb964ee15cb1a5d9f;

/// The indexer chain (stage, pool, ring while a sequence is built; scores and top-k; attention and its merge) launched with the
/// programmatic-dependent-launch attribute (the Flash family bit of the mask) gives the bits of plain launches, and the plain bits are the
/// recorded ones.
#[test]
fn qsa_kernels_are_exact_under_programmatic_launches() {
    let Some(st) = gpu() else { return };
    let mut texts = Vec::new();
    for mask in [0, 8] {
        gb_cuda::exl3_linear::set_pdl_mask(mask).unwrap();
        texts.push(golden_text(&st));
    }
    gb_cuda::exl3_linear::set_pdl_mask(0).unwrap();
    let hash = fnv(texts[0].as_bytes());
    println!("QSA golden hash of the plain launches: 0x{hash:016x}");
    assert_eq!(texts[0], texts[1], "programmatic launches changed a result");
    assert_eq!(hash, QSA_GOLDEN_HASH, "the plain launches differ from the recorded results:\n{}", texts[0]);
}
