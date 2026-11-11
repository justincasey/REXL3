//! Glue kernels: CPU mirrors (bit-exact where the op is exactly reproducible), row invariance, top-k against a full
//! sort, ExLlamaV3 dumps and benchmarks. GPU tests skip when there is no device; the dump tests also need the reference
//! dumps (`testdata/ref`) and `REXL3_QWEN_DIR`.
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
mod common;

use common::*;
use core::ffi::c_void;
use gb_cuda::glue::{self, AttnPrepCfg};
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

fn up<T: Copy>(v: &[T]) -> DevBuf {
    DevBuf::from_slice(v).unwrap()
}
fn zeros(bytes: usize) -> DevBuf {
    DevBuf::zeroed(bytes).unwrap()
}
fn cp(b: &DevBuf) -> *const c_void {
    b.ptr() as *const c_void
}
fn mp(b: &DevBuf) -> *mut c_void {
    b.ptr()
}
fn dl<T: Copy + Default>(b: &DevBuf) -> Vec<T> {
    b.to_vec().unwrap()
}

/// Rows with a heavy-tailed scale so norms see a wide dynamic range; values are exactly bf16.
fn rand_bf16(rng: &mut Rng, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| {
            let s = rng.uniform(0.0, 1.0);
            f32_to_bf16(rng.uniform(-1.0, 1.0) * (0.05 + 30.0 * s * s * s))
        })
        .collect()
}

fn ulp_diff(a: u16, b: u16) -> u32 {
    let ord = |u: u16| -> i32 {
        if u & 0x8000 != 0 {
            -((u & 0x7FFF) as i32)
        } else {
            u as i32
        }
    };
    (ord(a) - ord(b)).unsigned_abs()
}

/// Max bf16 ulp difference and the number of differing elements.
fn compare_ulp(got: &[u16], want: &[u16]) -> (u32, usize) {
    assert_eq!(got.len(), want.len());
    let mut mx = 0;
    let mut cnt = 0;
    for (&g, &w) in got.iter().zip(want) {
        let d = ulp_diff(g, w);
        mx = mx.max(d);
        cnt += (d != 0) as usize;
    }
    (mx, cnt)
}

// ------------------------------------------------------------------------------------------------ CPU mirrors

fn butterfly32(mut a: [f32; 32]) -> f32 {
    for m in [16usize, 8, 4, 2, 1] {
        let b = a;
        for l in 0..32 {
            a[l] = b[l] + b[l ^ m];
        }
    }
    a[0]
}

/// Sum of squares in the order of `rmsnorm_k`: 256 threads own 8-vectors t, t+256, ...; butterfly; warps 0..7 in order.
fn ss_block(f: &[f32]) -> f32 {
    let nvec = f.len() / 8;
    let mut thr = [0f32; 256];
    for (t, th) in thr.iter_mut().enumerate() {
        let mut s = 0f32;
        let mut v = t;
        while v < nvec {
            for i in 0..8 {
                let x = f[v * 8 + i];
                s += x * x;
            }
            v += 256;
        }
        *th = s;
    }
    let mut s = butterfly32(thr[0..32].try_into().unwrap());
    for w in 1..8 {
        s += butterfly32(thr[w * 32..w * 32 + 32].try_into().unwrap());
    }
    s
}

fn inv_rms(ss: f32, dim: usize, eps: f32) -> f32 {
    1.0f32 / (ss / dim as f32 + eps).sqrt()
}

fn rmsnorm_ref(x: &[u16], w: &[f32], eps: f32) -> Vec<u16> {
    let f = bf16_to_f32_vec(x);
    let inv = inv_rms(ss_block(&f), x.len(), eps);
    f.iter().zip(w).map(|(&v, &w)| f32_to_bf16((v * inv) * w)).collect()
}

fn add_ref(a: &[u16], b: &[u16]) -> Vec<u16> {
    a.iter().zip(b).map(|(&a, &b)| f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b))).collect()
}

fn silu64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}
fn sigmoid64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn f16_bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| f64_to_f16(x as f64)).collect()
}

/// `small_linear` mirror: thread t owns 8-vectors t, t+256, ...; fmaf in element order; butterfly; warps in order.
fn small_linear_ref(x: &[u16], w: &[u16]) -> f32 {
    let k = x.len();
    let nvec = k / 8;
    let mut thr = [0f32; 256];
    for (t, th) in thr.iter_mut().enumerate() {
        let mut a = 0f32;
        let mut v = t;
        while v < nvec {
            for i in 0..8 {
                a = bf16_to_f32(x[v * 8 + i]).mul_add(f16_to_f64(w[v * 8 + i]) as f32, a);
            }
            v += 256;
        }
        *th = a;
    }
    let mut s = butterfly32(thr[0..32].try_into().unwrap());
    for wv in 1..8 {
        s += butterfly32(thr[wv * 32..wv * 32 + 32].try_into().unwrap());
    }
    s
}

// ------------------------------------------------------------------------------------------------ elementwise

#[test]
fn add_swiglu_gate_match_cpu() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(1);
    for &n in &[1usize, 7, 8, 9, 1000, 4096, 17408 * 3 + 5] {
        let a = rand_bf16(&mut rng, n);
        let b = rand_bf16(&mut rng, n);
        let (da, db) = (up(&a), up(&b));
        let out = zeros(n * 2);
        unsafe { glue::add(cp(&da), cp(&db), mp(&out), n, &st).unwrap() };
        assert_eq!(dl::<u16>(&out), add_ref(&a, &b), "add n={n}");
        // swiglu / gate: f64 reference, at most 1 bf16 ulp off (expf is not correctly rounded)
        unsafe { glue::swiglu(cp(&da), cp(&db), mp(&out), n, &st).unwrap() };
        let want: Vec<u16> =
            (0..n).map(|i| f32_to_bf16((silu64(bf16_to_f32(a[i]) as f64) * bf16_to_f32(b[i]) as f64) as f32)).collect();
        let (mx, cnt) = compare_ulp(&dl::<u16>(&out), &want);
        assert!(mx <= 1 && cnt * 100 <= n.max(100), "swiglu n={n}: max ulp {mx}, {cnt} differ");
        unsafe { glue::gate_mul(cp(&da), cp(&db), mp(&out), n, &st).unwrap() };
        let want: Vec<u16> =
            (0..n).map(|i| f32_to_bf16((bf16_to_f32(a[i]) as f64 * sigmoid64(bf16_to_f32(b[i]) as f64)) as f32)).collect();
        let (mx, cnt) = compare_ulp(&dl::<u16>(&out), &want);
        assert!(mx <= 1 && cnt * 100 <= n.max(100), "gate_mul n={n}: max ulp {mx}, {cnt} differ");
    }
    st.sync().unwrap();
}

#[test]
fn elementwise_is_alignment_independent_and_in_place() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(2);
    let n = 4096 + 8;
    let a = rand_bf16(&mut rng, n);
    let b = rand_bf16(&mut rng, n);
    let (da, db) = (up(&a), up(&b));
    let out_aligned = zeros(n * 2);
    unsafe { glue::swiglu(cp(&da), cp(&db), mp(&out_aligned), n, &st).unwrap() };
    // same data shifted by one element (2 bytes): takes the scalar kernel, must give the same bits
    let mut sa = vec![0u16];
    sa.extend(&a);
    let mut sb = vec![0u16];
    sb.extend(&b);
    let (dsa, dsb) = (up(&sa), up(&sb));
    let out_shift = zeros((n + 1) * 2);
    unsafe { glue::swiglu(dsa.ptr_at(2), dsb.ptr_at(2), out_shift.ptr_at(2), n, &st).unwrap() };
    assert_eq!(dl::<u16>(&out_aligned), dl::<u16>(&out_shift)[1..].to_vec());
    // in place: out == a
    unsafe { glue::add(cp(&da), cp(&db), mp(&da), n, &st).unwrap() };
    assert_eq!(dl::<u16>(&da), add_ref(&a, &b));
    st.sync().unwrap();
}

// ------------------------------------------------------------------------------------------------ embedding

#[test]
fn embed_gathers_and_zeroes_bad_ids() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(3);
    for &dim in &[5120usize, 96, 100] {
        let vocab = 300usize;
        let table = rand_bf16(&mut rng, vocab * dim);
        let ids: Vec<i32> = vec![0, 299, 7, -1, 300, 301, i32::MIN, i32::MAX, 123, 123, 5];
        let (dt, di) = (up(&table), up(&ids));
        let out = zeros(ids.len() * dim * 2);
        unsafe { glue::embed(cp(&dt), cp(&di), mp(&out), ids.len(), dim, vocab, &st).unwrap() };
        let got = dl::<u16>(&out);
        for (r, &id) in ids.iter().enumerate() {
            let row = &got[r * dim..(r + 1) * dim];
            if (0..vocab as i32).contains(&id) {
                assert_eq!(row, &table[id as usize * dim..(id as usize + 1) * dim], "dim={dim} id={id}");
            } else {
                assert!(row.iter().all(|&v| v == 0), "dim={dim} id={id} must be zeros");
            }
        }
    }
    st.sync().unwrap();
}

/// The Ling-3.1 embedding read in place (`Block::map_file_readonly`): `glue::embed` from a read-only mapping of the table's file gathers
/// exactly the rows it gathers from a device copy of the same bytes, at a decode row, a window and a prefill chunk. The table starts
/// mid-page and 12 bytes past a 16-byte boundary, as the real pack's does (shard 1, offset 3,106,828), so the mapped table takes the
/// kernel's scalar path while the copy takes the vector path: different loads, the same bytes out.
#[test]
fn embed_from_a_table_read_in_place_matches_the_device_copy() {
    gpu!();
    if !gb_cuda::rt::pageable_memory_access() {
        eprintln!("the device does not read pageable memory: skipping");
        return;
    }
    let st = Stream::new().unwrap();
    let mut rng = Rng(11);
    let (vocab, dim) = (2048usize, 4096usize);
    let table = rand_bf16(&mut rng, vocab * dim);
    let offset = 3 * 4096 + 2060u64;
    let path = std::env::temp_dir().join(format!("rexl3-embed-in-place-{}.bin", std::process::id()));
    let mut bytes: Vec<u8> = (0..offset).map(|i| (i % 251) as u8).collect();
    bytes.extend(table.iter().flat_map(|v| v.to_le_bytes()));
    bytes.extend((0..1000).map(|i| (i % 7) as u8));
    std::fs::write(&path, &bytes).unwrap();
    let block = std::sync::Arc::new(gb_cuda::rt::Block::map_file_readonly(&path, offset, vocab * dim * 2).unwrap());
    let mapped = DevBuf::view(&block, 0, vocab * dim * 2);
    assert_eq!(mapped.ptr() as usize % 16, 12);
    let copy = up(&table);
    for rows in [1usize, 64, 512] {
        let mut ids: Vec<i32> = (0..rows).map(|_| (rng.next_u64() % vocab as u64) as i32).collect();
        ids[0] = (vocab - 1) as i32;
        if rows > 4 {
            ids[1] = 0;
            ids[2] = -1;
            ids[3] = vocab as i32;
        }
        let di = up(&ids);
        let (a, b) = (zeros(rows * dim * 2), zeros(rows * dim * 2));
        unsafe {
            glue::embed(cp(&mapped), cp(&di), mp(&a), rows, dim, vocab, &st).unwrap();
            glue::embed(cp(&copy), cp(&di), mp(&b), rows, dim, vocab, &st).unwrap();
        }
        st.sync().unwrap();
        let (got, want) = (dl::<u16>(&a), dl::<u16>(&b));
        assert_eq!(got, want, "rows={rows}: the mapped table and the device copy gather different bytes");
        for (r, &id) in ids.iter().enumerate() {
            let row = &got[r * dim..(r + 1) * dim];
            if (0..vocab as i32).contains(&id) {
                assert_eq!(row, &table[id as usize * dim..(id as usize + 1) * dim], "rows={rows} id={id}");
            } else {
                assert!(row.iter().all(|&v| v == 0), "rows={rows} id={id} must be zeros");
            }
        }
    }
    drop(mapped);
    drop(block);
    std::fs::remove_file(&path).unwrap();
}

// ------------------------------------------------------------------------------------------------ rmsnorm

fn weights(rng: &mut Rng, n: usize) -> Vec<f32> {
    rng.vec(n, 0.2, 2.5)
}

#[test]
fn rmsnorm_bit_exact_vs_cpu_order() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(4);
    for &dim in &[8usize, 40, 136, 256, 2048, 5120, 5128] {
        for &rows in &[1usize, 3, 8] {
            let x = rand_bf16(&mut rng, rows * dim);
            let w = weights(&mut rng, dim);
            let (dx, dw) = (up(&x), up(&w));
            let dy = zeros(rows * dim * 2);
            unsafe { glue::rmsnorm(cp(&dx), cp(&dw), mp(&dy), rows, dim, 1e-6, &st).unwrap() };
            let got = dl::<u16>(&dy);
            for r in 0..rows {
                let want = rmsnorm_ref(&x[r * dim..(r + 1) * dim], &w, 1e-6);
                assert_eq!(&got[r * dim..(r + 1) * dim], &want[..], "dim={dim} rows={rows} row={r}");
            }
            // and against an f64 reference: within one bf16 ulp plus the f32 summation noise
            let f: Vec<f64> = x[..dim].iter().map(|&u| bf16_to_f32(u) as f64).collect();
            let ms = f.iter().map(|v| v * v).sum::<f64>() / dim as f64;
            let inv = 1.0 / (ms + 1e-6).sqrt();
            let want64: Vec<u16> = (0..dim).map(|i| f32_to_bf16((f[i] * inv * w[i] as f64) as f32)).collect();
            let (mx, _) = compare_ulp(&got[..dim], &want64);
            assert!(mx <= 1, "dim={dim}: f64 reference off by {mx} ulp");
        }
    }
    // in place
    let (dim, rows) = (5120, 4);
    let x = rand_bf16(&mut rng, rows * dim);
    let w = weights(&mut rng, dim);
    let (dx, dw) = (up(&x), up(&w));
    unsafe { glue::rmsnorm(cp(&dx), cp(&dw), mp(&dx), rows, dim, 1e-6, &st).unwrap() };
    let got = dl::<u16>(&dx);
    for r in 0..rows {
        assert_eq!(&got[r * dim..(r + 1) * dim], &rmsnorm_ref(&x[r * dim..(r + 1) * dim], &w, 1e-6)[..]);
    }
    st.sync().unwrap();
}

#[test]
fn rmsnorm_rejects_bad_shapes() {
    gpu!();
    let st = Stream::new().unwrap();
    let dx = zeros(4096);
    let dw = zeros(4096);
    let dy = zeros(4096);
    assert!(unsafe { glue::rmsnorm(cp(&dx), cp(&dw), mp(&dy), 1, 100, 1e-6, &st) }.is_err()); // dim % 8
    assert!(unsafe { glue::rmsnorm(dx.ptr_at(2), cp(&dw), mp(&dy), 1, 64, 1e-6, &st) }.is_err()); // misaligned
    assert!(unsafe { glue::rmsnorm(cp(&dx), cp(&dw), mp(&dy), 0, 64, 1e-6, &st) }.is_ok());
    // zero rows is a no-op
}

/// Runs `f` for the rows `idx` of `x` (batch in the order given) and returns the outputs per original row index.
fn rows_of(x: &[u16], dim: usize, idx: &[usize]) -> Vec<u16> {
    idx.iter().flat_map(|&r| x[r * dim..(r + 1) * dim].iter().copied()).collect()
}

#[test]
fn rmsnorm_and_add_rmsnorm_are_row_invariant() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(5);
    let (dim, total) = (5120usize, 13usize);
    let x = rand_bf16(&mut rng, total * dim);
    let res = rand_bf16(&mut rng, total * dim);
    let w = weights(&mut rng, dim);
    let dw = up(&w);
    let run = |idx: &[usize]| -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let (bx, br) = (rows_of(&x, dim, idx), rows_of(&res, dim, idx));
        let (dx, dr) = (up(&bx), up(&br));
        let (dy1, dh, dy2) = (zeros(bx.len() * 2), zeros(bx.len() * 2), zeros(bx.len() * 2));
        unsafe {
            glue::rmsnorm(cp(&dx), cp(&dw), mp(&dy1), idx.len(), dim, 1e-6, &st).unwrap();
            glue::add_rmsnorm(cp(&dx), cp(&dr), cp(&dw), mp(&dh), mp(&dy2), idx.len(), dim, 1e-6, &st).unwrap();
        }
        (dl(&dy1), dl(&dh), dl(&dy2))
    };
    let all: Vec<usize> = (0..total).collect();
    let (y_all, h_all, ya_all) = run(&all);
    let rev: Vec<usize> = (0..total).rev().collect();
    let (y_rev, h_rev, ya_rev) = run(&rev);
    for r in 0..total {
        let seg = r * dim..(r + 1) * dim;
        let (y1, h1, ya1) = run(&[r]);
        assert_eq!(&y_all[seg.clone()], &y1[..], "rmsnorm alone vs batch, row {r}");
        assert_eq!(&h_all[seg.clone()], &h1[..], "h alone vs batch, row {r}");
        assert_eq!(&ya_all[seg.clone()], &ya1[..], "add_rmsnorm alone vs batch, row {r}");
        let rr = total - 1 - r; // position of row r in the reversed batch
        let seg2 = rr * dim..(rr + 1) * dim;
        assert_eq!(&y_rev[seg2.clone()], &y1[..]);
        assert_eq!(&h_rev[seg2.clone()], &h1[..]);
        assert_eq!(&ya_rev[seg2], &ya1[..]);
    }
    let (y3, _, _) = run(&[4, 4, 9]); // duplicated rows in a small batch
    assert_eq!(&y3[..dim], &y_all[4 * dim..5 * dim]);
    assert_eq!(&y3[dim..2 * dim], &y_all[4 * dim..5 * dim]);
    st.sync().unwrap();
}

#[test]
fn add_rmsnorm_bit_exact_and_aliasing() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(6);
    for &(rows, dim) in &[(1usize, 5120usize), (8, 5120), (5, 136), (2, 256)] {
        let x = rand_bf16(&mut rng, rows * dim);
        let res = rand_bf16(&mut rng, rows * dim);
        let w = weights(&mut rng, dim);
        let (dw, dr) = (up(&w), up(&res));
        let h_want = add_ref(&x, &res);
        let y_want: Vec<u16> = (0..rows).flat_map(|r| rmsnorm_ref(&h_want[r * dim..(r + 1) * dim], &w, 1e-6)).collect();
        // separate buffers
        let dx = up(&x);
        let (dh, dy) = (zeros(x.len() * 2), zeros(x.len() * 2));
        unsafe { glue::add_rmsnorm(cp(&dx), cp(&dr), cp(&dw), mp(&dh), mp(&dy), rows, dim, 1e-6, &st).unwrap() };
        assert_eq!(dl::<u16>(&dh), h_want, "h rows={rows} dim={dim}");
        assert_eq!(dl::<u16>(&dy), y_want, "y rows={rows} dim={dim}");
        // in place: h_out == x
        let dx2 = up(&x);
        let dy2 = zeros(x.len() * 2);
        unsafe { glue::add_rmsnorm(cp(&dx2), cp(&dr), cp(&dw), mp(&dx2), mp(&dy2), rows, dim, 1e-6, &st).unwrap() };
        assert_eq!(dl::<u16>(&dx2), h_want);
        assert_eq!(dl::<u16>(&dy2), y_want);
        // h_out == res and y == x
        let dr2 = up(&res);
        let dx3 = up(&x);
        unsafe { glue::add_rmsnorm(cp(&dx3), cp(&dr2), cp(&dw), mp(&dr2), mp(&dx3), rows, dim, 1e-6, &st).unwrap() };
        assert_eq!(dl::<u16>(&dr2), h_want);
        assert_eq!(dl::<u16>(&dx3), y_want);
        // h_out == y is refused
        assert!(unsafe { glue::add_rmsnorm(cp(&dx), cp(&dr), cp(&dw), mp(&dy), mp(&dy), rows, dim, 1e-6, &st) }.is_err());
    }
    st.sync().unwrap();
}

// ------------------------------------------------------------------------------------------------ attention prep

struct PrepOut {
    q: Vec<u16>,
    gate: Vec<u16>,
    k: Vec<u16>,
}

fn run_prep(cfg: &AttnPrepCfg, st: &Stream, q_in: &[u16], k_in: &[u16], pos: &[i32], qw: &[f32], kw: &[f32]) -> PrepOut {
    let rows = pos.len();
    let (dq, dk, dp, dqw, dkw) = (up(q_in), up(k_in), up(pos), up(qw), up(kw));
    let dqo = zeros(rows * cfg.heads * cfg.head_dim * 2);
    let dgo = zeros(rows * cfg.heads * cfg.head_dim * 2);
    let dko = zeros(rows * cfg.kv_heads * cfg.head_dim * 2);
    let gate = if cfg.has_gate { mp(&dgo) } else { core::ptr::null_mut() };
    unsafe { glue::attn_prep(cfg, cp(&dq), cp(&dk), cp(&dp), cp(&dqw), cp(&dkw), mp(&dqo), gate, mp(&dko), rows, st).unwrap() };
    PrepOut { q: dl(&dqo), gate: if cfg.has_gate { dl(&dgo) } else { vec![] }, k: dl(&dko) }
}

fn run_prep_mrope(cfg: &AttnPrepCfg, st: &Stream, q_in: &[u16], k_in: &[u16], pos3: &[i32], qw: &[f32], kw: &[f32]) -> PrepOut {
    let rows = pos3.len() / 3;
    let (dq, dk, dp, dqw, dkw) = (up(q_in), up(k_in), up(pos3), up(qw), up(kw));
    let dqo = zeros(rows * cfg.heads * cfg.head_dim * 2);
    let dgo = zeros(rows * cfg.heads * cfg.head_dim * 2);
    let dko = zeros(rows * cfg.kv_heads * cfg.head_dim * 2);
    let gate = if cfg.has_gate { mp(&dgo) } else { core::ptr::null_mut() };
    unsafe {
        glue::attn_prep_mrope(cfg, cp(&dq), cp(&dk), cp(&dp), cp(&dqw), cp(&dkw), mp(&dqo), gate, mp(&dko), rows, st).unwrap()
    };
    PrepOut { q: dl(&dqo), gate: if cfg.has_gate { dl(&dgo) } else { vec![] }, k: dl(&dko) }
}

/// Interleaved M-RoPE: with the three streams equal the result is bit-identical to the single-position kernel (so text rows keep
/// their bits), and with distinct streams every frequency pair equals the single-position kernel's pair at that pair's stream
/// (pair i uses T, H or W for i % 3 = 0, 1, 2, H only below 33 and W only below 30).
#[test]
fn mrope_streams_pick_their_frequency_pairs() {
    check_mrope(AttnPrepCfg::QWEN_TARGET);
}

/// The same at MiniCPM-V-4.7's head counts (16 query / 2 KV heads: its canvas M-RoPE, sections 11/11/10, rotary 64 of 256).
#[test]
fn mrope_streams_pick_their_frequency_pairs_at_minicpm_heads() {
    check_mrope(AttnPrepCfg::MINICPM_TARGET);
}

fn check_mrope(cfg: AttnPrepCfg) {
    gpu!();
    let st = Stream::new().unwrap();
    let (h, kv, hd, rows) = (cfg.heads, cfg.kv_heads, cfg.head_dim, 9usize);
    let mut rng = Rng(77);
    let qs = 2 * hd;
    let q_in = rand_bf16(&mut rng, rows * h * qs);
    let k_in = rand_bf16(&mut rng, rows * kv * hd);
    let (qw, kw) = (weights(&mut rng, hd), weights(&mut rng, hd));
    let pos: Vec<i32> = (0..rows as i32).map(|r| 3 + r * 211).collect();
    // equal streams == the existing kernel
    let eq3: Vec<i32> = pos.iter().flat_map(|&p| [p, p, p]).collect();
    let base = run_prep(&cfg, &st, &q_in, &k_in, &pos, &qw, &kw);
    let same = run_prep_mrope(&cfg, &st, &q_in, &k_in, &eq3, &qw, &kw);
    assert_eq!(same.q, base.q, "equal streams must reproduce the text kernel's q");
    assert_eq!(same.k, base.k);
    assert_eq!(same.gate, base.gate);
    // distinct streams: T, H and W differ per row
    let pos3: Vec<i32> = (0..rows as i32).flat_map(|r| [100 + r, 7 + 3 * r, 1000 - 17 * r]).collect();
    let got = run_prep_mrope(&cfg, &st, &q_in, &k_in, &pos3, &qw, &kw);
    let per_axis: Vec<PrepOut> = (0..3)
        .map(|a| {
            let p: Vec<i32> = (0..rows).map(|r| pos3[r * 3 + a]).collect();
            run_prep(&cfg, &st, &q_in, &k_in, &p, &qw, &kw)
        })
        .collect();
    let axis_of = |i: usize| {
        if i % 3 == 1 && i < 33 {
            1
        } else if i % 3 == 2 && i < 30 {
            2
        } else {
            0
        }
    };
    let half = cfg.rotary_dim / 2;
    let check = |got: &[u16], per: &[&[u16]], heads: usize, what: &str| {
        for r in 0..rows {
            for hh in 0..heads {
                let base = (r * heads + hh) * hd;
                for d in 0..hd {
                    let pair = if d < cfg.rotary_dim { d % half } else { usize::MAX };
                    let want = if pair == usize::MAX { per[0][base + d] } else { per[axis_of(pair)][base + d] };
                    assert_eq!(got[base + d], want, "{what} r{r} h{hh} d{d}");
                }
            }
        }
    };
    check(&got.q, &[&per_axis[0].q, &per_axis[1].q, &per_axis[2].q], h, "q");
    check(&got.k, &[&per_axis[0].k, &per_axis[1].k, &per_axis[2].k], kv, "k");
    assert_eq!(got.gate, base.gate, "the gate half is position-independent");
    // the streams must actually matter
    assert_ne!(got.q, same.q);
}

/// One head: rmsnorm (kernel order, bit exact) then rope in f64 on the rounded values. Returns (reference, scale per element
/// = max(|a|, |b|) of the rotated pair, for the tolerance).
fn prep_head_ref(x: &[u16], w: &[f32], eps: f32, pos: i32, rotary: usize, theta: f32) -> (Vec<f64>, Vec<f64>) {
    let hd = x.len();
    let f = bf16_to_f32_vec(x);
    let mut lanes = [0f32; 32];
    for (l, la) in lanes.iter_mut().enumerate() {
        let mut s = 0f32;
        for j in 0..hd / 32 {
            let v = f[l + 32 * j];
            s += v * v;
        }
        *la = s;
    }
    let inv = inv_rms(butterfly32(lanes), hd, eps);
    let n: Vec<f32> = (0..hd).map(|d| round_bf16((f[d] * inv) * w[d])).collect();
    let mut out: Vec<f64> = n.iter().map(|&v| v as f64).collect();
    let mut scale: Vec<f64> = n.iter().map(|&v| (v as f64).abs()).collect();
    let half = rotary / 2;
    for i in 0..half {
        let invf = (theta as f64).powf(-((2 * i) as f64) / rotary as f64) as f32;
        let ang = (pos as f32 * invf) as f64;
        let (c, s) = (ang.cos(), ang.sin());
        let (a, b) = (n[i] as f64, n[i + half] as f64);
        out[i] = a * c - b * s;
        out[i + half] = b * c + a * s;
        scale[i] = a.abs().max(b.abs());
        scale[i + half] = scale[i];
    }
    (out, scale)
}

fn check_prep(cfg: &AttnPrepCfg, rows: usize, positions: &[i32], rng: &mut Rng, st: &Stream) {
    let (h, kv, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
    let qs = if cfg.has_gate { 2 * hd } else { hd };
    let q_in = rand_bf16(rng, rows * h * qs);
    let k_in = rand_bf16(rng, rows * kv * hd);
    let qw = weights(rng, hd);
    let kw = weights(rng, hd);
    let o = run_prep(cfg, st, &q_in, &k_in, positions, &qw, &kw);
    let (mut bad, mut total) = (0usize, 0usize);
    let check = |got: &[u16], want: &(Vec<f64>, Vec<f64>), what: &str, bad: &mut usize, total: &mut usize| {
        for d in 0..got.len() {
            let g = bf16_to_f32(got[d]) as f64;
            let tol = want.0[d].abs() / 256.0 + 1e-5 * want.1[d] + 1e-30;
            *total += 1;
            assert!((g - want.0[d]).abs() <= tol, "{what} d={d}: got {g} want {} (tol {tol})", want.0[d]);
            *bad += (f32_to_bf16(want.0[d] as f32) != got[d]) as usize;
        }
    };
    for r in 0..rows {
        for hh in 0..h {
            let src = &q_in[(r * h + hh) * qs..(r * h + hh) * qs + hd];
            let want = prep_head_ref(src, &qw, cfg.eps, positions[r], cfg.rotary_dim, cfg.theta);
            let base = (r * h + hh) * hd;
            check(&o.q[base..base + hd], &want, &format!("q r{r} h{hh}"), &mut bad, &mut total);
            if cfg.has_gate {
                assert_eq!(
                    &o.gate[base..base + hd],
                    &q_in[(r * h + hh) * qs + hd..(r * h + hh + 1) * qs],
                    "gate copy r{r} h{hh}"
                );
            }
            // dims past rotary_dim are the plain norm output: exact
            let (plain, _) = prep_head_ref(src, &qw, cfg.eps, positions[r], 0, cfg.theta);
            for d in cfg.rotary_dim..hd {
                assert_eq!(bf16_to_f32(o.q[base + d]) as f64, plain[d], "pass-through dims r{r} h{hh} d{d}");
            }
        }
        for hh in 0..kv {
            let src = &k_in[(r * kv + hh) * hd..(r * kv + hh + 1) * hd];
            let want = prep_head_ref(src, &kw, cfg.eps, positions[r], cfg.rotary_dim, cfg.theta);
            let base = (r * kv + hh) * hd;
            check(&o.k[base..base + hd], &want, &format!("k r{r} h{hh}"), &mut bad, &mut total);
        }
    }
    assert!(bad * 50 <= total, "{bad} of {total} elements are not the nearest bf16 of the f64 reference");
}

#[test]
fn attn_prep_matches_cpu_target_and_draft_layouts() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(7);
    let pos: Vec<i32> = vec![0, 1, 2, 39, 1000, 40_000, 131_071, 1_000_000];
    check_prep(&AttnPrepCfg::QWEN_TARGET, 8, &pos, &mut rng, &st);
    check_prep(&AttnPrepCfg::DFLASH2_DRAFT, 8, &pos, &mut rng, &st);
    // other shapes the kernel advertises
    let c = AttnPrepCfg { heads: 4, kv_heads: 2, head_dim: 64, rotary_dim: 64, has_gate: true, eps: 1e-6, theta: 10000.0 };
    check_prep(&c, 3, &pos[..3], &mut rng, &st);
    let c = AttnPrepCfg { heads: 3, kv_heads: 1, head_dim: 128, rotary_dim: 64, has_gate: false, eps: 1e-5, theta: 1e6 };
    check_prep(&c, 3, &pos[3..6], &mut rng, &st);
    let c = AttnPrepCfg { heads: 2, kv_heads: 2, head_dim: 256, rotary_dim: 0, has_gate: true, eps: 1e-6, theta: 1e7 };
    check_prep(&c, 2, &pos[..2], &mut rng, &st);
    let c = AttnPrepCfg { heads: 2, kv_heads: 2, head_dim: 512, rotary_dim: 256, has_gate: true, eps: 1e-6, theta: 1e7 };
    check_prep(&c, 2, &pos[..2], &mut rng, &st);
    // MiniCPM-V-4.7: the target layers (16 / 2 heads of 256, gate, rotary 64) and the DSpark draft's (8 / 2, no gate; its rotary dims
    // are read from the draft config later, so 64, 128 and 256 are all checked)
    check_prep(&AttnPrepCfg::MINICPM_TARGET, 8, &pos, &mut rng, &st);
    for rotary_dim in [64usize, 128, 256] {
        let c = AttnPrepCfg { heads: 8, kv_heads: 2, head_dim: 256, rotary_dim, has_gate: false, eps: 1e-6, theta: 1e7 };
        check_prep(&c, 8, &pos, &mut rng, &st);
    }
    st.sync().unwrap();
}

#[test]
fn attn_prep_is_row_invariant() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(8);
    for cfg in [AttnPrepCfg::QWEN_TARGET, AttnPrepCfg::DFLASH2_DRAFT, AttnPrepCfg::MINICPM_TARGET] {
        let (h, kv, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
        let qs = if cfg.has_gate { 2 * hd } else { hd };
        let total = 9usize;
        let q_in = rand_bf16(&mut rng, total * h * qs);
        let k_in = rand_bf16(&mut rng, total * kv * hd);
        let qw = weights(&mut rng, hd);
        let kw = weights(&mut rng, hd);
        let pos: Vec<i32> = (0..total as i32).map(|i| 3 + 17 * i * i).collect();
        let all = run_prep(&cfg, &st, &q_in, &k_in, &pos, &qw, &kw);
        // reversed batch
        let rev: Vec<usize> = (0..total).rev().collect();
        let rq = rows_of(&q_in, h * qs, &rev);
        let rk = rows_of(&k_in, kv * hd, &rev);
        let rp: Vec<i32> = rev.iter().map(|&r| pos[r]).collect();
        let rv = run_prep(&cfg, &st, &rq, &rk, &rp, &qw, &kw);
        for r in 0..total {
            let one = run_prep(
                &cfg,
                &st,
                &q_in[r * h * qs..(r + 1) * h * qs],
                &k_in[r * kv * hd..(r + 1) * kv * hd],
                &pos[r..r + 1],
                &qw,
                &kw,
            );
            assert_eq!(&all.q[r * h * hd..(r + 1) * h * hd], &one.q[..], "q row {r}");
            assert_eq!(&all.k[r * kv * hd..(r + 1) * kv * hd], &one.k[..], "k row {r}");
            let rr = total - 1 - r;
            assert_eq!(&rv.q[rr * h * hd..(rr + 1) * h * hd], &one.q[..], "q rev row {r}");
            assert_eq!(&rv.k[rr * kv * hd..(rr + 1) * kv * hd], &one.k[..], "k rev row {r}");
            if cfg.has_gate {
                assert_eq!(&all.gate[r * h * hd..(r + 1) * h * hd], &one.gate[..]);
            }
        }
    }
    st.sync().unwrap();
}

// ------------------------------------------------------------------------------------------------ gated norm

fn gated_ref(o: &[u16], z: &[u16], w: &[f32], eps: f32) -> Vec<f64> {
    let dim = o.len();
    let f = bf16_to_f32_vec(o);
    let epl = dim / 32;
    let mut lanes = [0f32; 32];
    for (l, la) in lanes.iter_mut().enumerate() {
        let mut s = 0f32;
        for i in 0..epl {
            s += f[l * epl + i] * f[l * epl + i];
        }
        *la = s;
    }
    let inv = inv_rms(butterfly32(lanes), dim, eps);
    (0..dim).map(|d| ((f[d] * inv) * w[d]) as f64 * silu64(bf16_to_f32(z[d]) as f64)).collect()
}

#[test]
fn gated_rmsnorm_matches_cpu_and_is_row_invariant() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(9);
    for &(heads, dim) in &[(48usize, 128usize), (3, 256), (40, 128)] {
        let total = 7;
        let o = rand_bf16(&mut rng, total * heads * dim);
        let z = rand_bf16(&mut rng, total * heads * dim);
        let w = rng.vec(dim, 0.2, 2.0);
        let (dw, dz) = (up(&w), up(&z));
        let run = |idx: &[usize]| -> Vec<u16> {
            let bo = rows_of(&o, heads * dim, idx);
            let bz = rows_of(&z, heads * dim, idx);
            let (do_, dz) = (up(&bo), up(&bz));
            let out = zeros(bo.len() * 2);
            unsafe { glue::gated_rmsnorm(cp(&do_), cp(&dz), cp(&dw), mp(&out), idx.len(), heads, dim, 1e-6, &st).unwrap() };
            dl(&out)
        };
        let all: Vec<usize> = (0..total).collect();
        let got = run(&all);
        let _ = &dz;
        let mut bad = 0;
        for r in 0..total {
            for h in 0..heads {
                let b = (r * heads + h) * dim;
                let want = gated_ref(&o[b..b + dim], &z[b..b + dim], &w, 1e-6);
                for d in 0..dim {
                    let g = bf16_to_f32(got[b + d]) as f64;
                    assert!((g - want[d]).abs() <= want[d].abs() / 128.0 + 1e-30, "r{r} h{h} d{d}: {g} vs {}", want[d]);
                    bad += (f32_to_bf16(want[d] as f32) != got[b + d]) as usize;
                }
            }
        }
        assert!(bad * 100 <= got.len(), "{bad} of {} not nearest", got.len());
        for r in 0..total {
            let one = run(&[r]);
            assert_eq!(&got[r * heads * dim..(r + 1) * heads * dim], &one[..], "row invariance, row {r}");
        }
        let rev: Vec<usize> = (0..total).rev().collect();
        let rv = run(&rev);
        for r in 0..total {
            let rr = total - 1 - r;
            assert_eq!(&got[r * heads * dim..(r + 1) * heads * dim], &rv[rr * heads * dim..(rr + 1) * heads * dim]);
        }
    }
    st.sync().unwrap();
}

// ------------------------------------------------------------------------------------------------ small linear

/// The many-row kernel (`small_linear_f16w_rows`, what a prefill chunk's router and gates run) leaves the very bits of the per-row
/// kernel: every row count around the 8-row tile, widths around the 8-column group (a 1-wide gate included), `k` of one vector a
/// thread (2,048), several (4,096, 5,120) and ragged (2,056), bf16 and f32 outputs, and the CPU mirror on a few outputs.
#[test]
fn small_linear_rows_kernel_is_bit_identical_to_the_per_row_kernel() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(11);
    for &(k, n) in &[
        (2048usize, 256usize),
        (2048, 1),
        (2048, 32),
        (2048, 9),
        (4096, 48),
        (5120, 256),
        (2056, 5),
        (1032, 11),
        (64, 24),
        (64, 3),
        (8, 1),
    ] {
        let w16 = f16_bits(&rng.vec(n * k, -0.2, 0.2));
        let dw = up(&w16);
        for &rows in &[1usize, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 200] {
            let x = rand_bf16(&mut rng, rows * k);
            let dx = up(&x);
            let (a, af) = (zeros(rows * n * 2), zeros(rows * n * 4));
            let (b, bf) = (zeros(rows * n * 2), zeros(rows * n * 4));
            unsafe {
                glue::small_linear_f16w(cp(&dx), cp(&dw), mp(&a), rows, k, n, &st).unwrap();
                glue::small_linear_f16w_f32out(cp(&dx), cp(&dw), mp(&af), rows, k, n, &st).unwrap();
                glue::small_linear_f16w_rows(cp(&dx), cp(&dw), mp(&b), rows, k, n, &st).unwrap();
                glue::small_linear_f16w_f32out_rows(cp(&dx), cp(&dw), mp(&bf), rows, k, n, &st).unwrap();
            }
            assert_eq!(dl::<u16>(&a), dl::<u16>(&b), "bf16 out, rows {rows} k {k} n {n}");
            let (want, got) = (dl::<f32>(&af), dl::<f32>(&bf));
            for (i, (x, y)) in want.iter().zip(&got).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "f32 out, rows {rows} k {k} n {n} at {i}");
            }
            if rows == 9 || rows == 65 {
                for &(r, j) in &[(0usize, 0usize), (rows - 1, n - 1), (rows / 2, n / 2)] {
                    let cpu = small_linear_ref(&x[r * k..(r + 1) * k], &w16[j * k..(j + 1) * k]);
                    assert_eq!(got[r * n + j].to_bits(), cpu.to_bits(), "CPU mirror rows {rows} k {k} n {n} ({r}, {j})");
                }
            }
        }
    }
    st.sync().unwrap();
}

/// `small_linear_f16w_pair` (the router and the shared gate, the GDN gate projections: one launch of two weights on one input) leaves the
/// very bits of two `small_linear_f16w` calls, for the model's widths (512 + 1, 48 + 48) and ragged ones, at window and chunk row counts.
#[test]
fn small_linear_pair_is_bit_identical_to_two_calls() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(12);
    for &(k, n1, n2) in &[(2560usize, 512usize, 1usize), (2560, 48, 48), (2048, 256, 1), (2056, 5, 11), (64, 24, 3)] {
        let (w1, w2) = (f16_bits(&rng.vec(n1 * k, -0.2, 0.2)), f16_bits(&rng.vec(n2 * k, -0.2, 0.2)));
        let (d1, d2) = (up(&w1), up(&w2));
        for &rows in &[1usize, 2, 3, 7, 8, 9, 17, 64, 200] {
            let dx = up(&rand_bf16(&mut rng, rows * k));
            let (a1, a2) = (zeros(rows * n1 * 2), zeros(rows * n2 * 2));
            let (b1, b2) = (zeros(rows * n1 * 2), zeros(rows * n2 * 2));
            unsafe {
                glue::small_linear_f16w(cp(&dx), cp(&d1), mp(&a1), rows, k, n1, &st).unwrap();
                glue::small_linear_f16w(cp(&dx), cp(&d2), mp(&a2), rows, k, n2, &st).unwrap();
                glue::small_linear_f16w_pair(cp(&dx), cp(&d1), mp(&b1), n1, cp(&d2), mp(&b2), n2, rows, k, &st).unwrap();
            }
            assert_eq!(dl::<u16>(&a1), dl::<u16>(&b1), "first output, rows {rows} k {k} n {n1}+{n2}");
            assert_eq!(dl::<u16>(&a2), dl::<u16>(&b2), "second output, rows {rows} k {k} n {n1}+{n2}");
        }
    }
    st.sync().unwrap();
}

/// A timing lane (run alone under `flock -x`): the per-row and the many-row kernel at a prefill chunk's widths (MiniCPM-V-4.7's router,
/// delta-net a/b and shared-expert gate at 4,096 rows).
///
///     flock -x /tmp/rexl3-gpu.lock cargo test --release -p gb-cuda --test glue bench_small_linear -- --ignored --nocapture
#[test]
#[ignore]
fn bench_small_linear_rows() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(12);
    let k = 2048usize;
    for &rows in &[512usize, 4096] {
        for &n in &[256usize, 32, 1] {
            let x = rand_bf16(&mut rng, rows * k);
            let w16 = f16_bits(&rng.vec(n * k, -0.2, 0.2));
            let (dx, dw) = (up(&x), up(&w16));
            let dy = zeros(rows * n * 2);
            let time = |many: bool| -> f64 {
                let run = || unsafe {
                    if many {
                        glue::small_linear_f16w_rows(cp(&dx), cp(&dw), mp(&dy), rows, k, n, &st).unwrap()
                    } else {
                        glue::small_linear_f16w(cp(&dx), cp(&dw), mp(&dy), rows, k, n, &st).unwrap()
                    }
                };
                for _ in 0..3 {
                    run();
                }
                st.sync().unwrap();
                let t0 = std::time::Instant::now();
                for _ in 0..20 {
                    run();
                }
                st.sync().unwrap();
                t0.elapsed().as_secs_f64() * 1e6 / 20.0
            };
            let (a, b) = (time(false), time(true));
            eprintln!(
                "BENCH small_linear rows {rows:>4} k {k} n {n:>3}: per-row {a:8.1} us, many-row {b:8.1} us ({:.1}x)",
                a / b
            );
        }
    }
}

/// The weight-stationary many-row kernel (registers permuted per lane, a select-free reduce-scatter) leaves the per-row kernel's bits at the
/// model's prefill widths and the shapes around them: the router (2,560 -> 512), the delta-net gates (2,560 -> 48), `k` of one vector
/// a thread (2,048), of two (2,560, 3,072) and ragged in columns (n of 9, 17, 100, 513), row counts around the 4-row step and the row
/// slices (1 .. 4,012), bf16 and f32 outputs; and it equals the previous weight-stationary kernel on the same inputs.
#[test]
fn small_linear_cols_kernel_equals_the_per_row_kernel_and_its_predecessor() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(31);
    for &(k, n) in &[
        (2560usize, 512usize),
        (2560, 48),
        (2560, 17),
        (2560, 9),
        (2048, 100),
        (3072, 513),
        (1024, 24),
        (2560, 8),
    ] {
        let w16 = f16_bits(&rng.vec(n * k, -0.2, 0.2));
        let dw = up(&w16);
        for &rows in &[1usize, 2, 3, 4, 5, 7, 8, 9, 31, 63, 64, 65, 200, 1001, 4012] {
            if rows * n > 1_200_000 && rows != 4012 {
                continue;
            }
            let dx = up(&rand_bf16(&mut rng, rows * k));
            let (a, af) = (zeros(rows * n * 2), zeros(rows * n * 4));
            let (b, bf) = (zeros(rows * n * 2), zeros(rows * n * 4));
            let c = zeros(rows * n * 2);
            unsafe {
                glue::small_linear_f16w(cp(&dx), cp(&dw), mp(&a), rows, k, n, &st).unwrap();
                glue::small_linear_f16w_f32out(cp(&dx), cp(&dw), mp(&af), rows, k, n, &st).unwrap();
                glue::small_linear_f16w_rows(cp(&dx), cp(&dw), mp(&b), rows, k, n, &st).unwrap();
                glue::small_linear_f16w_f32out_rows(cp(&dx), cp(&dw), mp(&bf), rows, k, n, &st).unwrap();
                glue::small_linear_f16w_rows_ref(cp(&dx), cp(&dw), mp(&c), rows, k, n, &st).unwrap();
            }
            assert_eq!(dl::<u16>(&a), dl::<u16>(&b), "bf16 out vs per-row, rows {rows} k {k} n {n}");
            assert_eq!(dl::<u16>(&c), dl::<u16>(&b), "bf16 out vs predecessor, rows {rows} k {k} n {n}");
            let (want, got) = (dl::<f32>(&af), dl::<f32>(&bf));
            for (i, (x, y)) in want.iter().zip(&got).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "f32 out, rows {rows} k {k} n {n} at {i}");
            }
        }
    }
    st.sync().unwrap();
}

/// The weight-stationary kernel and its predecessor at the model's prefill widths, 4,012 rows (router, GDN gate projection).
///     cargo test --release -p gb-cuda --test glue bench_small_linear_cols -- --ignored --nocapture
#[test]
#[ignore]
fn bench_small_linear_cols() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(32);
    let (rows, k) = (4012usize, 2560usize);
    let dx = up(&rand_bf16(&mut rng, rows * k));
    for &n in &[512usize, 48] {
        let dw = up(&f16_bits(&rng.vec(n * k, -0.2, 0.2)));
        let dy = zeros(rows * n * 2);
        for (name, reference) in [("previous", true), ("permuted", false)] {
            let run = || unsafe {
                if reference {
                    glue::small_linear_f16w_rows_ref(cp(&dx), cp(&dw), mp(&dy), rows, k, n, &st).unwrap()
                } else {
                    glue::small_linear_f16w_rows(cp(&dx), cp(&dw), mp(&dy), rows, k, n, &st).unwrap()
                }
            };
            for _ in 0..5 {
                run();
            }
            st.sync().unwrap();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t0 = std::time::Instant::now();
                for _ in 0..10 {
                    run();
                }
                st.sync().unwrap();
                best = best.min(t0.elapsed().as_secs_f64() * 1e6 / 10.0);
            }
            eprintln!("BENCH small_linear_cols n {n:>3} {name}: {best:.1} us");
        }
    }
}

#[test]
fn small_linear_bit_exact_and_row_invariant() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(10);
    for &(k, n) in &[(5120usize, 48usize), (5120, 256), (2056, 5), (64, 3), (8, 1)] {
        let total = 19usize; // spans two 8-row tiles plus a partial one
        let x = rand_bf16(&mut rng, total * k);
        let w16 = f16_bits(&rng.vec(n * k, -0.2, 0.2));
        let (dw, dx) = (up(&w16), up(&x));
        let (dy, dyf) = (zeros(total * n * 2), zeros(total * n * 4));
        unsafe {
            glue::small_linear_f16w(cp(&dx), cp(&dw), mp(&dy), total, k, n, &st).unwrap();
            glue::small_linear_f16w_f32out(cp(&dx), cp(&dw), mp(&dyf), total, k, n, &st).unwrap();
        }
        let (y, yf) = (dl::<u16>(&dy), dl::<f32>(&dyf));
        for r in 0..total {
            for j in 0..n {
                let want = small_linear_ref(&x[r * k..(r + 1) * k], &w16[j * k..(j + 1) * k]);
                assert_eq!(yf[r * n + j].to_bits(), want.to_bits(), "f32 out k={k} r={r} j={j}");
                assert_eq!(y[r * n + j], f32_to_bf16(want), "bf16 out k={k} r={r} j={j}");
            }
        }
        // one row alone / in a different batch
        for &r in &[0usize, 8, 18] {
            let d1 = up(&x[r * k..(r + 1) * k]);
            let o1 = zeros(n * 2);
            unsafe { glue::small_linear_f16w(cp(&d1), cp(&dw), mp(&o1), 1, k, n, &st).unwrap() };
            assert_eq!(dl::<u16>(&o1), y[r * n..(r + 1) * n].to_vec(), "alone vs batch k={k} r={r}");
        }
        let sub = rows_of(&x, k, &[17, 3, 11]);
        let (ds, os) = (up(&sub), zeros(3 * n * 2));
        unsafe { glue::small_linear_f16w(cp(&ds), cp(&dw), mp(&os), 3, k, n, &st).unwrap() };
        let o = dl::<u16>(&os);
        for (i, &r) in [17usize, 3, 11].iter().enumerate() {
            assert_eq!(&o[i * n..(i + 1) * n], &y[r * n..(r + 1) * n]);
        }
    }
    st.sync().unwrap();
}

// ------------------------------------------------------------------------------------------------ top-k

fn topk_ref(row: &[u16], count: usize) -> Vec<(i32, f32)> {
    let mut v: Vec<(f32, u32)> = row
        .iter()
        .enumerate()
        .map(|(i, &u)| {
            let f = bf16_to_f32(u);
            (if f.is_nan() { f32::NEG_INFINITY } else { f }, i as u32)
        })
        .collect();
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    v.iter().take(count).map(|&(f, i)| (i as i32, f)).collect()
}

fn run_topk(st: &Stream, logits: &[u16], rows: usize, vocab: usize, count: usize) -> (Vec<i32>, Vec<f32>) {
    let dl_ = up(logits);
    let ids = zeros(rows * count * 4);
    let vals = zeros(rows * count * 4);
    let wsb = glue::topk_workspace_bytes(rows, vocab, count);
    assert_eq!(wsb, glue::topk_workspace_bytes_c(rows, vocab, count));
    let ws = zeros(wsb);
    unsafe { glue::topk(cp(&dl_), rows, vocab, count, mp(&ids), mp(&vals), mp(&ws), wsb, st).unwrap() };
    (dl(&ids), dl(&vals))
}

fn check_topk(st: &Stream, logits: &[u16], rows: usize, vocab: usize, count: usize, what: &str) {
    let (ids, vals) = run_topk(st, logits, rows, vocab, count);
    for r in 0..rows {
        let want = topk_ref(&logits[r * vocab..(r + 1) * vocab], count);
        for (i, &(wid, wv)) in want.iter().enumerate() {
            let (gid, gv) = (ids[r * count + i], vals[r * count + i]);
            assert!(gid == wid && gv == wv, "{what}: row {r} rank {i}: got ({gid}, {gv}) want ({wid}, {wv})");
        }
    }
}

#[test]
fn topk_ties_across_blocks_vs_full_sort() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(11);
    let vocab = 248320usize;
    let rows = 4usize;
    // Few distinct values: massive exact ties everywhere, +0/-0, NaN and +-inf sprinkled in.
    let pool = [-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 2.0];
    let mut logits: Vec<u16> = (0..rows * vocab).map(|_| f32_to_bf16(pool[(rng.next_u64() % 8) as usize])).collect();
    for r in 0..rows {
        let b = r * vocab;
        logits[b + 5] = 0x8000; // -0.0
        logits[b + 12] = 0x7FC1; // NaN
        logits[b + 99] = 0xFF80; // -inf
        logits[b + 100] = 0x7F80; // +inf
        for c in 1..4 {
            // block boundaries (the stage-1 chunk is glue::TOPK_CHUNK): the same max value on both sides
            logits[b + c * glue::TOPK_CHUNK - 1] = 0x7F80;
            logits[b + c * glue::TOPK_CHUNK] = 0x7F80;
        }
        logits[b + vocab - 1] = 0x7F80;
    }
    for &count in &[1usize, 16, 28, 64] {
        check_topk(&st, &logits, rows, vocab, count, &format!("ties count={count}"));
    }
    // all equal row: the answer is ids 0..count
    let eq = vec![f32_to_bf16(1.5); vocab];
    for &count in &[1usize, 16, 28, 64] {
        let (ids, _) = run_topk(&st, &eq, 1, vocab, count);
        assert_eq!(ids, (0..count as i32).collect::<Vec<_>>(), "all-equal count={count}");
    }
    // zeros of both signs only: -0 and +0 tie, lowest ids win whatever the sign
    let mut zz: Vec<u16> = (0..vocab).map(|i| if i % 3 == 0 { 0x8000 } else { 0 }).collect();
    zz[vocab - 7] = f32_to_bf16(1e-30); // a tiny positive above both zeros
    check_topk(&st, &zz, 1, vocab, 16, "signed zeros");
    // maxima exactly on chunk edges only
    let mut edge = vec![f32_to_bf16(-1.0); vocab];
    for c in 1..vocab / glue::TOPK_CHUNK {
        edge[c * glue::TOPK_CHUNK] = f32_to_bf16(3.0);
        edge[c * glue::TOPK_CHUNK - 1] = f32_to_bf16(3.0);
    }
    for &count in &[16usize, 28, 64] {
        check_topk(&st, &edge, 1, vocab, count, &format!("chunk edges count={count}"));
    }
    st.sync().unwrap();
}

#[test]
fn topk_random_patterns_and_sizes() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(12);
    let vocab = 248320usize;
    // every bf16 bit pattern (NaNs, infs, denormals, both zeros included)
    let any: Vec<u16> = (0..3 * vocab).map(|_| (rng.next_u64() & 0xFFFF) as u16).collect();
    for &count in &[1usize, 16, 28, 64] {
        check_topk(&st, &any, 3, vocab, count, &format!("any bits count={count}"));
    }
    // continuous values
    let cont: Vec<u16> = (0..8 * vocab).map(|_| f32_to_bf16(rng.uniform(-19.0, 29.0))).collect();
    check_topk(&st, &cont, 8, vocab, 20, "continuous");
    // sorted worst cases for the candidate gate: ascending and descending
    let asc: Vec<u16> = (0..vocab).map(|i| f32_to_bf16(i as f32 * 1e-3)).collect();
    check_topk(&st, &asc, 1, vocab, 64, "ascending");
    let desc: Vec<u16> = asc.iter().rev().copied().collect();
    check_topk(&st, &desc, 1, vocab, 64, "descending");
    // small and odd vocabularies, count == vocab
    for &(v, count) in &[
        (64usize, 64usize),
        (100, 64),
        (1001, 28),
        (glue::TOPK_CHUNK, 16),
        (glue::TOPK_CHUNK + 1, 16),
        (2 * glue::TOPK_CHUNK + 3, 64),
        (24576 + 5, 64),
        (17, 1),
        (1, 1),
        (16, 16),
    ] {
        let l: Vec<u16> = (0..2 * v).map(|_| f32_to_bf16(((rng.next_u64() % 9) as f32) * 0.25)).collect();
        check_topk(&st, &l, 2, v, count, &format!("vocab={v} count={count}"));
    }
    // errors
    let l = up(&any[..vocab]);
    let (ids, vals, ws) = (zeros(256), zeros(256), zeros(1 << 20));
    let bad =
        |count: usize, vocab: usize| unsafe { glue::topk(cp(&l), 1, vocab, count, mp(&ids), mp(&vals), mp(&ws), 1 << 20, &st) };
    assert!(bad(0, 100).is_err() && bad(65, 1000).is_err() && bad(10, 5).is_err());
    st.sync().unwrap();
}

#[test]
fn topk_is_row_invariant() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(13);
    let vocab = 248320usize;
    let total = 6usize;
    let l: Vec<u16> = (0..total * vocab).map(|_| f32_to_bf16(((rng.next_u64() % 40) as f32) * 0.5)).collect();
    let (ids_all, vals_all) = run_topk(&st, &l, total, vocab, 28);
    for r in [0usize, 3, 5] {
        let (ids, vals) = run_topk(&st, &l[r * vocab..(r + 1) * vocab], 1, vocab, 28);
        assert_eq!(&ids_all[r * 28..(r + 1) * 28], &ids[..]);
        assert_eq!(&vals_all[r * 28..(r + 1) * 28], &vals[..]);
    }
    let rev = rows_of(&l, vocab, &[5, 4, 3, 2, 1, 0]);
    let (ids_r, _) = run_topk(&st, &rev, total, vocab, 28);
    for r in 0..total {
        let rr = total - 1 - r;
        assert_eq!(&ids_all[r * 28..(r + 1) * 28], &ids_r[rr * 28..(rr + 1) * 28]);
    }
    st.sync().unwrap();
}

// ------------------------------------------------------------------------------------------------ top-k over the valid columns

/// MiniCPM-V-4.7's head: 248,144 tokens padded to 248,192 columns (a multiple of 128) by the quantizer; the pad rows of the lm_head
/// produce garbage logits that must never be a candidate, a sampled token or a scored one.
const PADDED_VOCAB: usize = 248_192;
const TOKEN_VOCAB: usize = 248_144;

/// The pad's poison: the largest finite bf16, +inf, and a NaN (reported as -inf, but it must not even be read), in turn.
fn poison_pad(logits: &mut [u16], rows: usize, stride: usize, valid: usize) {
    for r in 0..rows {
        for (i, v) in logits[r * stride + valid..(r + 1) * stride].iter_mut().enumerate() {
            *v = [0x7F7F, 0x7F80, 0x7FC1][(i + r) % 3];
        }
    }
}

fn run_topk_valid(st: &Stream, logits: &[u16], rows: usize, stride: usize, valid: usize, count: usize) -> (Vec<i32>, Vec<f32>) {
    let dl_ = up(logits);
    let ids = zeros(rows * count * 4);
    let vals = zeros(rows * count * 4);
    let wsb = glue::topk_valid_workspace_bytes(rows, valid, count);
    assert_eq!(wsb, glue::topk_valid_workspace_bytes_c(rows, valid, count));
    let ws = zeros(wsb);
    unsafe { glue::topk_valid(cp(&dl_), rows, stride, valid, count, mp(&ids), mp(&vals), mp(&ws), wsb, st).unwrap() };
    (dl(&ids), dl(&vals))
}

/// The pad is never read: with every pad column poisoned the answer is the full sort of the valid columns alone, at the real vocabulary
/// and at sizes that put `valid` on and around a chunk edge (16,384) or off a multiple of 4 (the scalar load path).
#[test]
fn topk_valid_matches_a_full_sort_of_the_valid_columns_and_never_returns_the_pad() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(21);
    let chunk = glue::TOPK_CHUNK;
    let shapes = [
        (3usize, PADDED_VOCAB, TOKEN_VOCAB),
        (2, 1000, 987),
        (2, 20_000, 19_999),
        (2, 2 * chunk + 64, 2 * chunk),
        (2, 2 * chunk + 64, 2 * chunk + 1),
        (2, 2 * chunk + 64, 2 * chunk - 1),
        (2, 4096, 17),
    ];
    for (rows, stride, valid) in shapes {
        for count in [1usize, 16, 28, 64] {
            if count > valid {
                continue;
            }
            let mut logits = rand_bf16(&mut rng, rows * stride);
            for r in 0..rows {
                let row = &mut logits[r * stride..(r + 1) * stride];
                // ties for the maximum across chunk edges and at the last valid column, so a read past it would win
                for c in [0usize, valid - 1, valid / 2, chunk.min(valid - 1), (chunk - 1).min(valid - 1)] {
                    row[c] = f32_to_bf16(40.0);
                }
            }
            poison_pad(&mut logits, rows, stride, valid);
            let (ids, vals) = run_topk_valid(&st, &logits, rows, stride, valid, count);
            for r in 0..rows {
                let want = topk_ref(&logits[r * stride..r * stride + valid], count);
                for (i, &(wid, wv)) in want.iter().enumerate() {
                    let (gid, gv) = (ids[r * count + i], vals[r * count + i]);
                    assert!(
                        gid == wid && gv == wv,
                        "stride {stride} valid {valid} count {count} row {r} rank {i}: got ({gid}, {gv}) want ({wid}, {wv})"
                    );
                    assert!((gid as usize) < valid, "the pad (id {gid} >= valid {valid}) was returned");
                }
            }
        }
    }
}

/// With the whole row valid the new entry point is `topk`, bit for bit (the old launcher is this one with stride == valid).
#[test]
fn topk_valid_with_the_whole_row_valid_is_topk_bit_for_bit() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(22);
    let (rows, vocab) = (4usize, 248_320usize);
    let l: Vec<u16> = (0..rows * vocab).map(|_| f32_to_bf16(((rng.next_u64() % 61) as f32) * 0.5 - 15.0)).collect();
    for count in [1usize, 16, 28, 64] {
        let (i0, v0) = run_topk(&st, &l, rows, vocab, count);
        let (i1, v1) = run_topk_valid(&st, &l, rows, vocab, vocab, count);
        assert_eq!(i0, i1, "ids, count {count}");
        assert_eq!(v0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), v1.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
}

/// A row's candidates do not depend on the other rows, how many there are or where it sits, with the pad poisoned.
#[test]
fn topk_valid_is_row_invariant() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(23);
    let rows = 7usize;
    let mut l: Vec<u16> = (0..rows * PADDED_VOCAB).map(|_| f32_to_bf16(((rng.next_u64() % 40) as f32) * 0.5)).collect();
    poison_pad(&mut l, rows, PADDED_VOCAB, TOKEN_VOCAB);
    let (ids_all, vals_all) = run_topk_valid(&st, &l, rows, PADDED_VOCAB, TOKEN_VOCAB, 28);
    for r in [0usize, 3, 6] {
        let (ids, vals) = run_topk_valid(&st, &l[r * PADDED_VOCAB..(r + 1) * PADDED_VOCAB], 1, PADDED_VOCAB, TOKEN_VOCAB, 28);
        assert_eq!(&ids_all[r * 28..(r + 1) * 28], &ids[..], "row {r} alone");
        assert_eq!(&vals_all[r * 28..(r + 1) * 28], &vals[..]);
    }
    assert!(ids_all.iter().all(|&i| (i as usize) < TOKEN_VOCAB));
}

#[test]
fn topk_valid_refuses_what_it_cannot_do() {
    gpu!();
    let st = Stream::new().unwrap();
    let l = up(&vec![0u16; 2 * 100]);
    let (ids, vals, ws) = (zeros(1024), zeros(1024), zeros(1 << 20));
    let go = |rows: usize, stride: usize, valid: usize, count: usize, wsb: usize| unsafe {
        glue::topk_valid(cp(&l), rows, stride, valid, count, mp(&ids), mp(&vals), mp(&ws), wsb, &st)
    };
    assert!(go(2, 100, 100, 10, 1 << 20).is_ok());
    assert!(go(2, 100, 101, 10, 1 << 20).is_err(), "valid above stride");
    assert!(go(2, 100, 10, 11, 1 << 20).is_err(), "count above valid");
    assert!(go(2, 100, 100, 65, 1 << 20).is_err(), "count above 64");
    assert!(go(2, 100, 0, 1, 1 << 20).is_err(), "no valid column");
    assert!(go(2, 100, 100, 10, 8).is_err(), "short workspace");
    assert_eq!(glue::topk_valid_workspace_bytes(1, 100, 65), 0);
    st.sync().unwrap();
}

/// The consumers downstream of the top-k see no pad: the target sampler (keyed Gumbel over the candidates, every seed and position)
/// picks the tokens it picks from the valid columns alone, and the scoring path (the logits rows a `score` request reads whole, copied
/// out of the padded rows with `copy_rows`) carries `valid` columns, equal to the valid columns of the head's output.
#[test]
fn the_sampler_and_the_score_export_never_see_the_pad() {
    gpu!();
    use gb_sampling::{try_choose, try_top_candidates, Sampling, MARGIN};
    let st = Stream::new().unwrap();
    let mut rng = Rng(24);
    let rows = 4usize;
    let mut l: Vec<u16> = (0..rows * PADDED_VOCAB).map(|_| f32_to_bf16(rng.uniform(-6.0, 6.0))).collect();
    poison_pad(&mut l, rows, PADDED_VOCAB, TOKEN_VOCAB);
    let sampling = Sampling { seed: 0, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 };
    let count = sampling.candidate_count(TOKEN_VOCAB);
    assert_eq!(count, 20 + MARGIN);
    let (ids, vals) = run_topk_valid(&st, &l, rows, PADDED_VOCAB, TOKEN_VOCAB, count);
    let mut picked = std::collections::BTreeSet::new();
    for r in 0..rows {
        let cid: Vec<u32> = ids[r * count..(r + 1) * count].iter().map(|&i| i as u32).collect();
        let cv = &vals[r * count..(r + 1) * count];
        // the CPU reference candidates of the valid columns alone
        let clean: Vec<f32> = l[r * PADDED_VOCAB..r * PADDED_VOCAB + TOKEN_VOCAB].iter().map(|&b| bf16_to_f32(b)).collect();
        let (wid, wv) = try_top_candidates(&clean, count).unwrap();
        assert_eq!(cid, wid, "row {r} candidates");
        assert_eq!(cv, &wv[..], "row {r} candidate values");
        for seed in 0..64u64 {
            for temperature in [0.0, 0.7, 1.0] {
                let s = Sampling { seed, temperature, ..sampling };
                let t = try_choose(cv, &cid, 100 + seed, &s).unwrap();
                assert!((t as usize) < TOKEN_VOCAB, "sampled the pad: id {t}");
                assert_eq!(
                    t,
                    try_choose(&wv, &wid, 100 + seed, &s).unwrap(),
                    "the sampler differs from the reference's candidates"
                );
                picked.insert(t);
            }
        }
    }
    assert!(picked.len() > 8, "the sampler must have had a choice ({} distinct tokens)", picked.len());
    // the score path: [rows, stride] -> [rows, valid] through copy_rows (what a host-side scorer reads), pad poisoned
    let src = up(&l);
    let dst = zeros(rows * TOKEN_VOCAB * 2);
    unsafe { glue::copy_rows(cp(&src), PADDED_VOCAB, mp(&dst), TOKEN_VOCAB, rows, TOKEN_VOCAB, &st).unwrap() };
    let got: Vec<u16> = dl(&dst);
    for r in 0..rows {
        assert_eq!(&got[r * TOKEN_VOCAB..(r + 1) * TOKEN_VOCAB], &l[r * PADDED_VOCAB..r * PADDED_VOCAB + TOKEN_VOCAB], "row {r}");
        // log-sum-exp over the exported columns equals the one over the clean valid columns, and is finite
        let lse = |row: &[u16]| {
            let m = row.iter().map(|&b| bf16_to_f32(b) as f64).fold(f64::NEG_INFINITY, f64::max);
            m + row.iter().map(|&b| (bf16_to_f32(b) as f64 - m).exp()).sum::<f64>().ln()
        };
        let a = lse(&got[r * TOKEN_VOCAB..(r + 1) * TOKEN_VOCAB]);
        assert!(a.is_finite() && a < 20.0, "log-sum-exp {a}");
    }
}

// ------------------------------------------------------------------------------------------------ copy / gather

#[test]
fn copy_and_gather_rows() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(14);
    // tap export: 5 taps of 5120 into [rows, 5, 5120]
    let (rows, w) = (6usize, 5120usize);
    let src = rand_bf16(&mut rng, rows * w);
    let dsrc = up(&src);
    let dst = zeros(rows * 5 * w * 2);
    for tap in [0usize, 3] {
        unsafe { glue::copy_rows(cp(&dsrc), w, dst.ptr_at(tap * w * 2), 5 * w, rows, w, &st).unwrap() };
    }
    let out = dl::<u16>(&dst);
    for r in 0..rows {
        for tap in 0..5 {
            let seg = &out[(r * 5 + tap) * w..(r * 5 + tap + 1) * w];
            if tap == 0 || tap == 3 {
                assert_eq!(seg, &src[r * w..(r + 1) * w]);
            } else {
                assert!(seg.iter().all(|&v| v == 0));
            }
        }
    }
    // odd widths and strides (scalar path), source with a stride larger than width
    let (w2, ss, ds) = (37usize, 50usize, 41usize);
    let s2 = rand_bf16(&mut rng, 4 * ss);
    let (d_s2, d_d2) = (up(&s2), zeros(4 * ds * 2));
    unsafe { glue::copy_rows(cp(&d_s2), ss, mp(&d_d2), ds, 4, w2, &st).unwrap() };
    let o2 = dl::<u16>(&d_d2);
    for r in 0..4 {
        assert_eq!(&o2[r * ds..r * ds + w2], &s2[r * ss..r * ss + w2]);
    }
    for &w in &[5120usize, 128, 37] {
        let table = rand_bf16(&mut rng, 20 * w);
        let idx: Vec<i32> = vec![19, 0, 7, 7, -1, 3];
        let (dt, di) = (up(&table), up(&idx));
        let o = zeros(idx.len() * w * 2);
        unsafe { glue::gather_rows(cp(&dt), cp(&di), mp(&o), idx.len(), w, &st).unwrap() };
        let got = dl::<u16>(&o);
        for (i, &r) in idx.iter().enumerate() {
            let seg = &got[i * w..(i + 1) * w];
            if r < 0 {
                assert!(seg.iter().all(|&v| v == 0));
            } else {
                assert_eq!(seg, &table[r as usize * w..(r as usize + 1) * w], "w={w} i={i}");
            }
        }
    }
    st.sync().unwrap();
}

// ------------------------------------------------------------------------------------------------ ExLlamaV3 dumps

fn pack() -> Option<gb_exl3::checkpoint::Checkpoint> {
    let dir = std::path::PathBuf::from(std::env::var("REXL3_QWEN_DIR").ok()?);
    if !dir.join("config.json").exists() || !have_refs() {
        return None;
    }
    gb_exl3::checkpoint::scan(&dir, false).ok()
}

const P: &str = "model.language_model.";

fn plain_f32(ck: &gb_exl3::checkpoint::Checkpoint, name: &str) -> Vec<f32> {
    let e = ck.plain.get(name).unwrap_or_else(|| panic!("no tensor {name}"));
    let b = ck.read_tensor(e).unwrap();
    match e.dtype.as_str() {
        "BF16" => b.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        "F16" => b.chunks_exact(2).map(|c| f16_to_f64(u16::from_le_bytes([c[0], c[1]])) as f32).collect(),
        "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        d => panic!("dtype {d}"),
    }
}

fn plain_u16(ck: &gb_exl3::checkpoint::Checkpoint, name: &str) -> Vec<u16> {
    let e = ck.plain.get(name).unwrap_or_else(|| panic!("no tensor {name}"));
    ck.read_tensor(e).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn plus_one(v: Vec<f32>) -> Vec<f32> {
    v.into_iter().map(|x| x + 1.0).collect()
}

fn report(what: &str, got: &[f32], want: &[f32], min_cos: f64, max_rel: f64) {
    let (cos, rel) = cos_rel_rms(got, want);
    eprintln!("{what}: cos {cos:.7} relRMS {rel:.3e}");
    assert!(cos >= min_cos && rel <= max_rel, "{what}: cos {cos} relRMS {rel}");
}

#[test]
fn dumps_input_layernorm_embedding_and_layer0_pieces() {
    gpu!();
    let Some(ck) = pack() else { return eprintln!("REXL3_QWEN_DIR / reference dumps absent: skipping") };
    let st = Stream::new().unwrap();
    let t = 40usize;
    let bf = |name: &str| -> Vec<u16> { bf16_vec(&ref_npy(name).to_f32()) };

    // layer 0 input_layernorm (weight stored as gamma - 1)
    let w = plus_one(plain_f32(&ck, &format!("{P}layers.0.input_layernorm.weight")));
    let x = bf("short/sub/L00.attn_norm.in.npy");
    let (dx, dw, dy) = (up(&x), up(&w), zeros(x.len() * 2));
    unsafe { glue::rmsnorm(cp(&dx), cp(&dw), mp(&dy), t, 5120, 1e-6, &st).unwrap() };
    let got = bf16_to_f32_vec(&dl::<u16>(&dy));
    report("L0 input_layernorm", &got, &ref_npy("short/sub/L00.attn_norm.out.npy").to_f32(), 0.9999, 1e-2);

    // the fused add + norm gives the same y for (x - r) + r split (r = 0 gives h = x)
    let zero = zeros(x.len() * 2);
    let (dh, dy2) = (zeros(x.len() * 2), zeros(x.len() * 2));
    unsafe { glue::add_rmsnorm(cp(&dx), cp(&zero), cp(&dw), mp(&dh), mp(&dy2), t, 5120, 1e-6, &st).unwrap() };
    assert_eq!(dl::<u16>(&dh), x);
    assert_eq!(dl::<u16>(&dy2), dl::<u16>(&dy));

    // GDN gated norm of layer 0: o = gdn_norm.in, z = gdn_norm.gate, plain gamma
    let w = plain_f32(&ck, &format!("{P}layers.0.linear_attn.norm.weight"));
    let (o, z) = (bf("short/sub/L00.gdn_norm.in.npy"), bf("short/sub/L00.gdn_norm.gate.npy"));
    let (do_, dz, dw, out) = (up(&o), up(&z), up(&w), zeros(o.len() * 2));
    unsafe { glue::gated_rmsnorm(cp(&do_), cp(&dz), cp(&dw), mp(&out), t, 48, 128, 1e-6, &st).unwrap() };
    let got = bf16_to_f32_vec(&dl::<u16>(&out));
    report("L0 gated norm", &got, &ref_npy("short/sub/L00.gdn_norm.out.npy").to_f32(), 0.9999, 1e-2);

    // in_proj_a / in_proj_b (fp16 weights [48, 5120])
    for nm in ["a", "b"] {
        let w16 = plain_u16(&ck, &format!("{P}layers.0.linear_attn.in_proj_{nm}.weight"));
        assert_eq!(w16.len(), 48 * 5120);
        let x = bf(&format!("short/sub/L00.in_proj_{nm}.in.npy"));
        let (dx, dw) = (up(&x), up(&w16));
        let (dy, dyf) = (zeros(t * 48 * 2), zeros(t * 48 * 4));
        unsafe {
            glue::small_linear_f16w(cp(&dx), cp(&dw), mp(&dy), t, 5120, 48, &st).unwrap();
            glue::small_linear_f16w_f32out(cp(&dx), cp(&dw), mp(&dyf), t, 5120, 48, &st).unwrap();
        }
        let want = ref_npy(&format!("short/sub/L00.in_proj_{nm}.out.npy")).to_f32();
        report(&format!("L0 in_proj_{nm} (f32 out)"), &dl::<f32>(&dyf), &want, 0.9999, 1e-2);
        report(&format!("L0 in_proj_{nm} (bf16 out)"), &bf16_to_f32_vec(&dl::<u16>(&dy)), &want, 0.9999, 1e-2);
    }

    // embedding of the short prompt from the real 248320 x 5120 table (byte offsets past 2^31)
    let e = ck.plain.get(&format!("{P}embed_tokens.weight")).expect("embed table");
    assert_eq!(e.shape, vec![248320, 5120]);
    let table = ck.read_tensor(e).unwrap();
    let ids: Vec<i32> = ref_npy("short/ids.npy").to_i64().iter().map(|&v| v as i32).collect();
    let (dt, di, out) = (DevBuf::from_slice(&table).unwrap(), up(&ids), zeros(t * 5120 * 2));
    unsafe { glue::embed(cp(&dt), cp(&di), mp(&out), t, 5120, 248320, &st).unwrap() };
    let got = dl::<u16>(&out);
    for (r, &id) in ids.iter().enumerate() {
        let o = id as usize * 5120 * 2;
        let want: Vec<u16> = table[o..o + 5120 * 2].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        assert_eq!(&got[r * 5120..(r + 1) * 5120], &want[..], "embed row {r}");
    }
    report("embedding vs EX embed.npy", &bf16_to_f32_vec(&got), &ref_npy("short/embed.npy").to_f32(), 0.99999, 1e-3);
    st.sync().unwrap();
}

#[test]
fn dumps_layer3_qk_prep_with_rope() {
    gpu!();
    let Some(ck) = pack() else { return eprintln!("REXL3_QWEN_DIR / reference dumps absent: skipping") };
    let st = Stream::new().unwrap();
    let t = 40usize;
    let bf = |name: &str| -> Vec<u16> { bf16_vec(&ref_npy(name).to_f32()) };
    let qg = bf("short/sub/L03.q_proj.out.npy");
    let k = bf("short/sub/L03.k_proj.out.npy");
    assert_eq!(qg.len(), t * 24 * 512);
    let qw = plus_one(plain_f32(&ck, &format!("{P}layers.3.self_attn.q_norm.weight")));
    let kw = plus_one(plain_f32(&ck, &format!("{P}layers.3.self_attn.k_norm.weight")));
    let pos: Vec<i32> = (0..t as i32).collect();
    let o = run_prep(&AttnPrepCfg::QWEN_TARGET, &st, &qg, &k, &pos, &qw, &kw);
    report("L3 q norm+rope", &bf16_to_f32_vec(&o.q), &ref_npy("short/sub/L03.rope.q_out.npy").to_f32(), 0.9999, 1e-2);
    report("L3 k norm+rope", &bf16_to_f32_vec(&o.k), &ref_npy("short/sub/L03.rope.k_out.npy").to_f32(), 0.9999, 1e-2);
    // layout guards: the gate output is the second half of every head, bit for bit; rope.q_in is the q half
    for r in 0..t {
        for h in 0..24 {
            let b = (r * 24 + h) * 512;
            assert_eq!(&o.gate[(r * 24 + h) * 256..(r * 24 + h + 1) * 256], &qg[b + 256..b + 512]);
        }
    }
    let q_in = ref_npy("short/sub/L03.rope.q_in.npy").to_f32();
    let q_half: Vec<f32> = (0..t * 24).flat_map(|i| bf16_to_f32_vec(&qg[i * 512..i * 512 + 256])).collect();
    report("L3 rope.q_in is the q half of q_proj.out", &q_half, &q_in, 0.9999, 1e-2);
    st.sync().unwrap();
}

// ------------------------------------------------------------------------------------------------ benchmarks

fn bench<F: FnMut()>(st: &Stream, iters: usize, mut f: F) -> f64 {
    for _ in 0..20 {
        f();
    }
    st.sync().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    st.sync().unwrap();
    t0.elapsed().as_secs_f64() / iters as f64
}

#[test]
#[ignore = "benchmark: cargo test --release -p gb-cuda --test glue -- --ignored --nocapture"]
fn bench_glue() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(99);
    let dim = 5120usize;
    let w = weights(&mut rng, dim);
    let dw = up(&w);
    println!("(timings are wall-clock over 1000 back-to-back launches on one stream; rows=1 is launch-bound)");
    for &rows in &[1usize, 8, 64] {
        let x = rand_bf16(&mut rng, rows * dim);
        let (dx, dr) = (up(&x), up(&x));
        let (dy, dh) = (zeros(rows * dim * 2), zeros(rows * dim * 2));
        let s = bench(&st, 1000, || unsafe { glue::rmsnorm(cp(&dx), cp(&dw), mp(&dy), rows, dim, 1e-6, &st).unwrap() });
        let bytes = (rows * dim * 2 * 2 + dim * 4) as f64;
        println!("rmsnorm      rows={rows:3} dim=5120: {:7.2} us/call  {:7.1} GB/s", s * 1e6, bytes / s / 1e9);
        let s = bench(&st, 1000, || unsafe {
            glue::add_rmsnorm(cp(&dx), cp(&dr), cp(&dw), mp(&dh), mp(&dy), rows, dim, 1e-6, &st).unwrap()
        });
        let bytes = (rows * dim * 2 * 4 + dim * 4) as f64;
        println!("add_rmsnorm  rows={rows:3} dim=5120: {:7.2} us/call  {:7.1} GB/s", s * 1e6, bytes / s / 1e9);
    }
    // topk
    let vocab = 248320usize;
    for &(rows, count) in &[(1usize, 20usize), (8, 20), (8, 16), (8, 28), (8, 64)] {
        let l: Vec<u16> = (0..rows * vocab).map(|_| f32_to_bf16(rng.uniform(-19.0, 29.0))).collect();
        let dlg = up(&l);
        let (ids, vals) = (zeros(rows * count * 4), zeros(rows * count * 4));
        let wsb = glue::topk_workspace_bytes(rows, vocab, count);
        let ws = zeros(wsb);
        let s = bench(&st, 300, || unsafe {
            glue::topk(cp(&dlg), rows, vocab, count, mp(&ids), mp(&vals), mp(&ws), wsb, &st).unwrap()
        });
        let bytes = (rows * vocab * 2) as f64;
        println!(
            "topk         rows={rows:3} vocab=248320 count={count:2}: {:7.2} us/call  {:7.1} GB/s",
            s * 1e6,
            bytes / s / 1e9
        );
    }
    // small linear n=48 k=5120 (+ the draft selector projection n=256)
    for &(n, rows) in &[(48usize, 1usize), (48, 8), (48, 64), (256, 1), (256, 8)] {
        let k = 5120usize;
        let x = rand_bf16(&mut rng, rows * k);
        let w16 = f16_bits(&rng.vec(n * k, -0.2, 0.2));
        let (dx, dw16) = (up(&x), up(&w16));
        let dy = zeros(rows * n * 2);
        let s = bench(&st, 1000, || unsafe { glue::small_linear_f16w(cp(&dx), cp(&dw16), mp(&dy), rows, k, n, &st).unwrap() });
        let bytes = (n * k * 2 + rows * k * 2 + rows * n * 2) as f64;
        println!("small_linear rows={rows:3} k=5120 n={n:3}: {:7.2} us/call  {:7.1} GB/s", s * 1e6, bytes / s / 1e9);
    }
    // attn prep and the rest at decode / verify sizes
    for &rows in &[1usize, 8, 512] {
        let cfg = AttnPrepCfg::QWEN_TARGET;
        let q = rand_bf16(&mut rng, rows * 24 * 512);
        let k = rand_bf16(&mut rng, rows * 4 * 256);
        let pos: Vec<i32> = (0..rows as i32).collect();
        let (dq, dk, dp) = (up(&q), up(&k), up(&pos));
        let (qw, kw) = (up(&weights(&mut rng, 256)), up(&weights(&mut rng, 256)));
        let (dqo, dgo, dko) = (zeros(rows * 24 * 256 * 2), zeros(rows * 24 * 256 * 2), zeros(rows * 4 * 256 * 2));
        let s = bench(&st, 1000, || unsafe {
            glue::attn_prep(&cfg, cp(&dq), cp(&dk), cp(&dp), cp(&qw), cp(&kw), mp(&dqo), mp(&dgo), mp(&dko), rows, &st).unwrap()
        });
        let bytes = (q.len() * 2 + k.len() * 2 + rows * 24 * 256 * 2 * 2 + rows * 4 * 256 * 2) as f64;
        println!("attn_prep    rows={rows:3} (24q/4kv x256):  {:7.2} us/call  {:7.1} GB/s", s * 1e6, bytes / s / 1e9);
    }
    for &rows in &[1usize, 8, 512] {
        let n = rows * 17408;
        let g = rand_bf16(&mut rng, n);
        let (dg, du, dout) = (up(&g), up(&g), zeros(n * 2));
        let s = bench(&st, 1000, || unsafe { glue::swiglu(cp(&dg), cp(&du), mp(&dout), n, &st).unwrap() });
        println!("swiglu       rows={rows:3} n=17408:         {:7.2} us/call  {:7.1} GB/s", s * 1e6, (n * 6) as f64 / s / 1e9);
        let o = rand_bf16(&mut rng, rows * 48 * 128);
        let (do_, dz, dout, dw) = (up(&o), up(&o), zeros(o.len() * 2), up(&rng.vec(128, 0.5, 1.5)));
        let s = bench(&st, 1000, || unsafe {
            glue::gated_rmsnorm(cp(&do_), cp(&dz), cp(&dw), mp(&dout), rows, 48, 128, 1e-6, &st).unwrap()
        });
        println!(
            "gated_norm   rows={rows:3} 48x128:           {:7.2} us/call  {:7.1} GB/s",
            s * 1e6,
            (o.len() * 6) as f64 / s / 1e9
        );
    }
}
