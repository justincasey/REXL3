//! The MLA kernels against the ExLlamaV3 fork's `MLAttention` on REAL layer weights (GLM-5.3-Flash, layer 3): the plain
//! bf16/fp16 `kv_b_proj`, real latents of a real token sequence, the fork's own q rows. The dumps come from
//! `tools/dump_mla_reference.py` (run it on the GB10 that holds the pack); point `REXL3_MLA_REF_DIR` at its output
//! directory. Without it these tests print that they did not run: the GLM pack is not part of the Qwen gate, so a missing
//! dump must not turn the Qwen reference gate red (the GLM family gate sets the variable, and `REXL3_MLA_REQUIRE_REF=1` makes a
//! missing dump a failure there).
//!
//! The oracle is not row-invariant, so every comparison is numeric (cosine, relative RMS), never a hash. Each stage is
//! checked on the oracle's OWN inputs, so the EXL3 projection differences (about 2e-3) cannot mask attention error:
//! q -> absorb vs the fork's absorbed queries; the fork's absorbed queries -> attention vs its attended latents (also at
//! 4x and 16x the queries, because real q gives near-uniform attention and a mask or softmax bug would hide in it);
//! the fork's attended latents -> expand vs its unfolded output; and the whole chain, plus the last prefill rows against the
//! fork's MHA-form prefill (a different arithmetic: per-head up-projected K and V).
#![cfg(not(gb_no_cuda))]

mod common;

use common::{cos_rel_rms, read_npy};
use gb_cuda::attention::{BlockTable, RowIndex};
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::mla::*;
use gb_cuda::num::bf16_to_f32;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::f16_to_f64;
use std::path::PathBuf;

fn ref_dir() -> Option<PathBuf> {
    let Ok(d) = std::env::var("REXL3_MLA_REF_DIR") else {
        // REXL3_MLA_REQUIRE_REF=1 (the GLM family gate) turns a missing dump into a failure, so that gate cannot pass vacuously
        assert!(std::env::var("REXL3_MLA_REQUIRE_REF").is_err(), "REXL3_MLA_REQUIRE_REF is set but REXL3_MLA_REF_DIR is not");
        eprintln!("not run: REXL3_MLA_REF_DIR is not set (GLM MLA oracle dumps from tools/dump_mla_reference.py)");
        return None;
    };
    let d = PathBuf::from(d);
    if !d.join("manifest.json").exists() {
        panic!("REXL3_MLA_REF_DIR={} has no manifest.json", d.display());
    }
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return None;
    }
    Some(d)
}

fn u16s(p: &std::path::Path) -> Vec<u16> {
    let n = read_npy(p);
    assert!(n.dtype == "<f2" || n.dtype == "<u2", "{}: dtype {}", p.display(), n.dtype);
    n.data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn f32s(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| f16_to_f64(b) as f32).collect()
}

struct Ref {
    dir: PathBuf,
    ctx: usize,
    windows: Vec<usize>,
    kv_b_fp16: bool,
    w: MlaWeights,
    prefix: Vec<u16>, // [ctx][512]
    stream: Stream,
}

fn load() -> Option<Ref> {
    let dir = ref_dir()?;
    let man = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    // the manifest is flat JSON written by json.dump: pull `"key": value` out by hand (no JSON dependency in this crate)
    let raw = |key: &str| -> String {
        let i = man.find(&format!("\"{key}\":")).unwrap_or_else(|| panic!("manifest has no {key}")) + key.len() + 3;
        let rest = &man[i..];
        if let Some(inner) = rest.trim_start().strip_prefix('[') {
            inner[..inner.find(']').unwrap()].to_string()
        } else {
            rest[..rest.find([',', '\n']).unwrap()].trim().trim_matches('"').to_string()
        }
    };
    let num = |key: &str| -> f64 { raw(key).parse().unwrap_or_else(|_| panic!("manifest {key}: {}", raw(key))) };
    assert_eq!(num("latent"), 512.0);
    assert_eq!(num("heads"), 64.0);
    assert_eq!(num("nope"), 256.0);
    assert_eq!(num("v"), 256.0);
    assert!((num("sm_scale") - 0.0625).abs() < 1e-12, "the kernels hardcode sm_scale 1/16");
    let ctx = num("ctx") as usize;
    let windows: Vec<usize> = raw("windows").split(',').map(|x| x.trim().parse().unwrap()).collect();
    let kv_b_fp16 = raw("kv_b_dtype").contains("float16");
    let kv_b = u16s(&dir.join("kv_b.u16.npy"));
    // the checkpoint tensor in its stored dtype, packed for the kernels in fp16 (the fork runs fp16 end to end)
    let src = if kv_b_fp16 { Dtype::F16 } else { Dtype::BF16 };
    let w = MlaWeights::from_kv_b(&kv_b, src, Dtype::F16).unwrap();
    let prefix = u16s(&dir.join("lat.npy"));
    assert_eq!(prefix.len(), ctx * LATENT_DIM);
    Some(Ref { dir, ctx, windows, kv_b_fp16, w, prefix, stream: Stream::new().unwrap() })
}

/// Cache holding the prefix `[ctx]` plus `extra` latents at positions ctx.. (one slot, identity block table).
fn cache_with(r: &Ref, extra: &[u16]) -> (LatentCache, BlockTable) {
    cache_from(r, &r.prefix, extra)
}

/// As `cache_with` with an explicit prefix (the sparse dump has its own, longer one).
fn cache_from(r: &Ref, prefix: &[u16], extra: &[u16]) -> (LatentCache, BlockTable) {
    let n = prefix.len() / LATENT_DIM + extra.len() / LATENT_DIM;
    let nb = n.div_ceil(BLOCK_TOKENS) + 1;
    let mut cache = LatentCache::new(nb, Dtype::F16, gb_cuda::mla::LatentKv::Plain).unwrap();
    cache.lat.memset_async(0xFF, &r.stream).unwrap(); // NaN fill: nothing past the context may be read
    let table = BlockTable::from_host(&[(0..nb as u32).collect()], nb).unwrap();
    let mut all = prefix.to_vec();
    all.extend_from_slice(extra);
    let ld = DevBuf::from_slice(&all).unwrap();
    let ri = RowIndex::from_host(&vec![0; n], &(0..n as i32).collect::<Vec<_>>()).unwrap();
    latent_write(&ld, &ri, &table, &mut cache, n, &r.stream).unwrap();
    r.stream.sync().unwrap();
    (cache, table)
}

fn attend_rows(r: &Ref, cache: &LatentCache, table: &BlockTable, q_lat: &[u16], pos0: usize, rows: usize) -> Vec<u16> {
    let ri = RowIndex::from_host(&vec![0; rows], &(pos0 as i32..(pos0 + rows) as i32).collect::<Vec<_>>()).unwrap();
    let qd = DevBuf::from_slice(q_lat).unwrap();
    let mut o = DevBuf::zeroed(rows * N_HEADS * LATENT_DIM * 2).unwrap();
    let max_ctx = pos0 + rows;
    let mut ws = DevBuf::alloc(attend_workspace_bytes(rows, max_ctx)).unwrap();
    ws.memset_async(0x3F, &r.stream).unwrap();
    attend(&qd, cache, table, &ri, rows, &mut o, max_ctx, &mut ws, &r.stream).unwrap();
    r.stream.sync().unwrap();
    o.to_vec().unwrap()
}

fn absorb_rows(r: &Ref, q: &[u16], rows: usize) -> Vec<u16> {
    let qd = DevBuf::from_slice(q).unwrap();
    let mut out = DevBuf::zeroed(rows * N_HEADS * LATENT_DIM * 2).unwrap();
    absorb(&qd, N_HEADS * NOPE_DIM, &r.w, &mut out, rows, &r.stream).unwrap();
    r.stream.sync().unwrap();
    out.to_vec().unwrap()
}

fn expand_rows(r: &Ref, o_lat: &[u16], rows: usize) -> Vec<u16> {
    let od = DevBuf::from_slice(o_lat).unwrap();
    let mut out = DevBuf::zeroed(rows * N_HEADS * NOPE_DIM * 2).unwrap();
    expand(&od, &r.w, &mut out, rows, &r.stream).unwrap();
    r.stream.sync().unwrap();
    out.to_vec().unwrap()
}

fn scale_f16(v: &[u16], s: f32) -> Vec<u16> {
    v.iter().map(|&b| gb_exl3::f16::f64_to_f16(f16_to_f64(b) * s as f64)).collect()
}

fn check(what: &str, got: &[u16], want: &[u16], min_cos: f64, max_rel: f64) {
    let (cos, rel) = cos_rel_rms(&f32s(got), &f32s(want));
    eprintln!("{what}: cos {cos:.9} relRMS {rel:.3e}");
    assert!(cos >= min_cos && rel <= max_rel, "{what}: cos {cos} relRMS {rel} (bounds {min_cos}, {max_rel})");
}

/// The real `kv_b_proj.weight` is finite and of weight magnitude, and the packed halves are exactly its rows: head `h`'s `W_UK` is
/// rows `h*512 + d` and its `W_UV` rows `h*512 + 256 + n` (the fork's `w.view(H, nope + v, D_c)` split), `W_UV` transposed.
#[test]
fn oracle_kv_b_is_the_plain_dense_weight_the_pack_stores() {
    let Some(r) = load() else { return };
    eprintln!("kv_b_proj stored as {}", if r.kv_b_fp16 { "fp16" } else { "bf16" });
    let kv_b = u16s(&r.dir.join("kv_b.u16.npy"));
    let rms = (kv_b.iter().map(|&b| if r.kv_b_fp16 { f16_to_f64(b) } else { bf16_to_f32(b) as f64 }).map(|x| x * x).sum::<f64>()
        / kv_b.len() as f64)
        .sqrt();
    assert!(rms.is_finite() && rms > 1e-4 && rms < 1.0, "kv_b rms {rms}");
    let src = if r.kv_b_fp16 { Dtype::F16 } else { Dtype::BF16 };
    let (uk, uvt) = pack_kv_b(&kv_b, src, src);
    let mut rg = 0x9E37_79B9u64;
    for _ in 0..4000 {
        rg = rg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let (h, d, c) = ((rg >> 33) as usize % N_HEADS, (rg >> 40) as usize % NOPE_DIM, (rg >> 20) as usize % LATENT_DIM);
        assert_eq!(
            uk[(h * NOPE_DIM + d) * LATENT_DIM + c],
            kv_b[(h * KV_B_ROWS_PER_HEAD + d) * LATENT_DIM + c],
            "uk h{h} d{d} c{c}"
        );
        assert_eq!(
            uvt[(h * LATENT_DIM + c) * NOPE_DIM + d],
            kv_b[(h * KV_B_ROWS_PER_HEAD + NOPE_DIM + d) * LATENT_DIM + c],
            "uvt h{h} n{d} c{c}"
        );
    }
}

#[test]
fn oracle_absorb_matches_the_fork() {
    let Some(r) = load() else { return };
    for &w in &r.windows {
        let q = u16s(&r.dir.join(format!("win{w}/q.npy")));
        let want = u16s(&r.dir.join(format!("win{w}/qlat.npy")));
        let got = absorb_rows(&r, &q, w);
        // measured on anonymized host-d (layer 3, 1500-token context): relRMS 1.3e-5 .. 1.9e-5; bounds are about 5x the worst
        check(&format!("absorb W={w}"), &got, &want, 0.99999999, 1e-4);
    }
}

#[test]
fn oracle_attention_matches_the_fork_dense() {
    let Some(r) = load() else { return };
    for &w in &r.windows {
        let lat = u16s(&r.dir.join(format!("win{w}/lat.npy")));
        let (cache, table) = cache_with(&r, &lat);
        let qlat = u16s(&r.dir.join(format!("win{w}/qlat.npy")));
        // real queries: near-uniform attention
        let got = attend_rows(&r, &cache, &table, &qlat, r.ctx, w);
        // measured: relRMS 1.0e-4 .. 1.4e-4 (the probabilities fed to the PV product are rounded to fp16 in both); bounds 3x
        check(&format!("attend W={w} (real q)"), &got, &u16s(&r.dir.join(format!("win{w}/olat.npy"))), 0.9999999, 4e-4);
        // peaked: 4x and 16x the queries against the fork's decode kernel on the same inputs
        for (s, min_cos, max_rel) in [(4.0f32, 0.9999999, 4e-4), (16.0, 0.9999999, 4e-4)] {
            let got = attend_rows(&r, &cache, &table, &scale_f16(&qlat, s), r.ctx, w);
            let want = u16s(&r.dir.join(format!("win{w}/olat_x{}.npy", s as usize)));
            check(&format!("attend W={w} (q x{s})"), &got, &want, min_cos, max_rel);
        }
    }
}

#[test]
fn oracle_expand_matches_the_fork() {
    let Some(r) = load() else { return };
    for &w in &r.windows {
        let olat = u16s(&r.dir.join(format!("win{w}/olat.npy")));
        let want = u16s(&r.dir.join(format!("win{w}/o.npy")));
        let got = expand_rows(&r, &olat, w);
        // measured: relRMS 1.3e-5 .. 2.0e-5
        check(&format!("expand W={w}"), &got, &want, 0.99999999, 1e-4);
    }
}

#[test]
fn oracle_whole_chain_matches_the_fork_decode_windows() {
    let Some(r) = load() else { return };
    for &w in &r.windows {
        let lat = u16s(&r.dir.join(format!("win{w}/lat.npy")));
        let (cache, table) = cache_with(&r, &lat);
        let q = u16s(&r.dir.join(format!("win{w}/q.npy")));
        let o_lat = attend_rows(&r, &cache, &table, &absorb_rows(&r, &q, w), r.ctx, w);
        let got = expand_rows(&r, &o_lat, w);
        // measured: relRMS 2.2e-4 .. 2.5e-4; bound 3x
        check(&format!("chain W={w}"), &got, &u16s(&r.dir.join(format!("win{w}/o.npy"))), 0.9999998, 8e-4);
    }
}

/// The last prefill rows through the absorbed form against the fork's MHA-form prefill (per-head up-projected K and V).
#[test]
fn oracle_prefill_rows_match_the_fork_mha_form() {
    let Some(r) = load() else { return };
    let q = u16s(&r.dir.join("prefill/q.npy"));
    let o = u16s(&r.dir.join("prefill/o.npy"));
    let (cache, table) = cache_with(&r, &[]);
    let row = N_HEADS * NOPE_DIM;
    for (start, n) in [(r.ctx - 128, 128usize), (r.ctx - 16, 16), (700, 64), (0, 32), (255, 3)] {
        let qs = &q[start * row..(start + n) * row];
        let o_lat = attend_rows(&r, &cache, &table, &absorb_rows(&r, qs, n), start, n);
        let got = expand_rows(&r, &o_lat, n);
        // measured: relRMS 3.9e-4 (the fork rounds the expanded per-head K and V to fp16); bound 3x
        check(&format!("prefill rows {start}..{}", start + n), &got, &o[start * row..(start + n) * row], 0.9999998, 1.2e-3);
    }
}

/// The sparse regime against the fork's `dsa_attn` (layer 3, context 3,000 so the windows select): the fork's own selection (its
/// k-pool top-512 pools expanded to tokens, in score order, plus the raw tail; -1 padded) and its absorbed queries go through
/// `attend_sparse`, and the attended latents are compared with the fork's, for the real queries and for 4x and 16x of them.
#[test]
fn oracle_sparse_attention_matches_the_fork_on_its_own_selection() {
    let Some(r) = load() else { return };
    let man = std::fs::read_to_string(r.dir.join("manifest.json")).unwrap();
    let Some(at) = man.find("\"sparse\":") else {
        assert!(
            std::env::var("REXL3_MLA_REQUIRE_REF").is_err(),
            "the dump has no sparse section (rerun tools/dump_mla_reference.py)"
        );
        eprintln!("not run: the dump has no sparse section (rerun tools/dump_mla_reference.py)");
        return;
    };
    let sub = &man[at..];
    let field = |key: &str| -> String {
        let i = sub.find(&format!("\"{key}\":")).unwrap() + key.len() + 3;
        let rest = &sub[i..];
        match rest.trim_start().strip_prefix('[') {
            Some(inner) => inner[..inner.find(']').unwrap()].to_string(),
            None => rest[..rest.find([',', '\n']).unwrap()].trim().to_string(),
        }
    };
    let sctx: usize = field("ctx").parse().unwrap();
    let swin: Vec<usize> = field("windows").split(',').map(|x| x.trim().parse().unwrap()).collect();
    let sd = r.dir.join("sparse");
    let prefix = u16s(&sd.join("lat.npy"));
    assert_eq!(prefix.len(), sctx * LATENT_DIM);
    for &w in &swin {
        let wd = sd.join(format!("win{w}"));
        let (cache, table) = cache_from(&r, &prefix, &u16s(&wd.join("lat.npy")));
        let qlat = u16s(&wd.join("qlat.npy"));
        let sel = read_npy(&wd.join("sel.npy"));
        assert_eq!(sel.dtype, "<i4");
        let kpad = sel.shape[1];
        let all = sel.to_i64();
        // the fork's rows with their -1 padding dropped, in the fork's order
        let mut idx = vec![0x3fff_fff0i32; w * 2056];
        let mut meta = vec![0i32; w * 2];
        let mut counts = Vec::new();
        for row in 0..w {
            let l: Vec<i32> = all[row * kpad..(row + 1) * kpad].iter().filter(|&&p| p >= 0).map(|&p| p as i32).collect();
            assert!(
                l.len() <= MAX_SELECTION && l.iter().all(|&p| (p as usize) < sctx + row + 1),
                "row {row}: list of {}",
                l.len()
            );
            idx[row * 2056..row * 2056 + l.len()].copy_from_slice(&l);
            meta[2 * row] = 1;
            meta[2 * row + 1] = l.len() as i32;
            counts.push(l.len());
        }
        let (idx_d, meta_d) = (DevBuf::from_slice(&idx).unwrap(), DevBuf::from_slice(&meta).unwrap());
        let sel_view = SparseSel { indices: &idx_d, meta: &meta_d, stride: 2056 };
        let ri = RowIndex::from_host(&vec![0; w], &(sctx as i32..(sctx + w) as i32).collect::<Vec<_>>()).unwrap();
        for (scale, name) in [(1.0f32, "olat"), (4.0, "olat_x4"), (16.0, "olat_x16")] {
            let qd = DevBuf::from_slice(&if scale == 1.0 { qlat.clone() } else { scale_f16(&qlat, scale) }).unwrap();
            let mut o = DevBuf::zeroed(w * N_HEADS * LATENT_DIM * 2).unwrap();
            let mut ws = DevBuf::alloc(attend_sparse_workspace_bytes(w)).unwrap();
            ws.memset_async(0x3F, &r.stream).unwrap();
            attend_sparse(&qd, &cache, &table, &ri, w, &sel_view, &mut o, &mut ws, &r.stream).unwrap();
            r.stream.sync().unwrap();
            let got: Vec<u16> = o.to_vec().unwrap();
            let want = u16s(&wd.join(format!("{name}.npy")));
            check(&format!("sparse W={w} {name} (lists of {counts:?})"), &got, &want, 0.9999999, 4e-4);
        }
    }
}
