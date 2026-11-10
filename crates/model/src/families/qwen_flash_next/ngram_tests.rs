//! Host tests for the completion-tracked n-gram warm worker. Every test builds a real, small
//! n-gram safetensors sidecar in a temp directory and mmaps it through [`NgramTable::open`] —
//! no mocks of the behavior under test.

use super::*;
use std::path::PathBuf;
use std::sync::Arc;

/// Rows for the "large" warm file: 64 MiB of real trellis payload at K=8.
const LARGE_ROWS: usize = 414_000;
/// Small table for success/idempotence/failure tests.
const SMALL_ROWS: usize = 64;
const K: usize = 8;
const HEADS_PER_NGRAM: usize = 8;

fn cfg_for_test() -> FlashNextConfig {
    let root: serde_json::Value =
        serde_json::from_str(include_str!("../../../../../testdata/qwen_flash_next/config.json")).unwrap();
    let mut cfg = FlashNextConfig::from_json(&root).unwrap();
    cfg.heads_per_ngram = HEADS_PER_NGRAM;
    cfg
}

fn unique_dir(tag: &str) -> PathBuf {
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("rexl3-fn-ngram-warm-{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Write a real safetensors sidecar with `rows` trellis rows. Returns the dir, the config the
/// table opens with, and the total file length the warm worker must fault.
fn write_ngram_file(rows: usize, tag: &str) -> (PathBuf, FlashNextConfig, u64) {
    let cfg = cfg_for_test();
    let dir = unique_dir(tag);
    let layer = cfg.ple_layer_ids.first().copied().unwrap() - 1;
    let p = format!("model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding");
    let words = inventory::ngram_row_words(K);
    let heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;

    let trellis_len = rows * words * 2;
    let offsets_len = heads * 8;
    let sizes_len = heads * 8;
    let mult_len = 3 * 8;
    let bias_len = heads * ROW * 2;
    let mut at = 0usize;
    let mut span = |len: usize| -> (usize, usize) {
        let s = at;
        at += len;
        (s, s + len)
    };
    let (t0, t1) = span(trellis_len);
    let (o0, o1) = span(offsets_len);
    let (s0, s1) = span(sizes_len);
    let (m0, m1) = span(mult_len);
    let (b0, b1) = span(bias_len);

    let mut tensors = serde_json::Map::new();
    tensors.insert(
        format!("{p}.shard_0.trellis"),
        serde_json::json!({"dtype": "I16", "shape": [rows, words], "data_offsets": [t0, t1]}),
    );
    tensors.insert(format!("{p}.head_offsets"), serde_json::json!({"dtype": "I64", "shape": [heads], "data_offsets": [o0, o1]}));
    tensors
        .insert(format!("{p}.head_vocab_sizes"), serde_json::json!({"dtype": "I64", "shape": [heads], "data_offsets": [s0, s1]}));
    tensors.insert(format!("{p}.layer_multipliers"), serde_json::json!({"dtype": "I64", "shape": [3], "data_offsets": [m0, m1]}));
    tensors
        .insert(format!("{p}.head_bias"), serde_json::json!({"dtype": "F16", "shape": [heads, ROW], "data_offsets": [b0, b1]}));
    let mut meta = serde_json::Map::new();
    meta.insert("format".into(), serde_json::Value::String("exl3_ngram_trellis".into()));
    meta.insert("codebook".into(), serde_json::Value::String("mul1".into()));
    meta.insert("K".into(), serde_json::Value::String(K.to_string()));
    meta.insert("rows".into(), serde_json::Value::String(rows.to_string()));
    meta.insert("shard_rows".into(), serde_json::Value::String(rows.to_string()));
    meta.insert("row_dim".into(), serde_json::Value::String("160".into()));
    let mut root = serde_json::Map::new();
    root.insert("__metadata__".into(), serde_json::Value::Object(meta));
    for (name, t) in tensors {
        root.insert(name, t);
    }
    let header = serde_json::Value::Object(root).to_string();
    let header_len = header.len();

    let path = dir.join(inventory::NGRAM_FILE);
    let mut file = std::fs::File::create(&path).unwrap();
    use std::io::Write;
    file.write_all(&(header_len as u64).to_le_bytes()).unwrap();
    file.write_all(header.as_bytes()).unwrap();

    // trellis: a position-dependent pseudo-random stream (splitmix64 of the 8-byte word index), so
    // every row differs and a misrouted row can never compare equal; not zeros either (a
    // sparse-cache fast path must not hide the read)
    let mut chunk = vec![0u8; 1 << 20];
    let mut at = 0usize;
    while at < trellis_len {
        let n = (trellis_len - at).min(chunk.len());
        for (j, w) in chunk[..n].chunks_mut(8).enumerate() {
            let mut z = ((at / 8 + j) as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let l = w.len();
            w.copy_from_slice(&z.to_le_bytes()[..l]);
        }
        file.write_all(&chunk[..n]).unwrap();
        at += n;
    }
    for _ in 0..heads {
        file.write_all(&0i64.to_le_bytes()).unwrap();
    }
    for _ in 0..heads {
        file.write_all(&(rows as i64).to_le_bytes()).unwrap();
    }
    for m in [101i64, 1009, 10007] {
        file.write_all(&m.to_le_bytes()).unwrap();
    }
    let bias = vec![0u16; heads * ROW];
    for w in bias {
        file.write_all(&w.to_le_bytes()).unwrap();
    }
    file.flush().unwrap();
    drop(file);
    (dir, cfg, path.metadata().unwrap().len())
}

fn open_large() -> (PathBuf, Arc<NgramTable>, u64) {
    let (dir, cfg, len) = write_ngram_file(LARGE_ROWS, "large");
    let table = Arc::new(NgramTable::open(&dir, &cfg).unwrap());
    (dir, table, len)
}

fn open_small(tag: &str) -> (PathBuf, Arc<NgramTable>, u64) {
    let (dir, cfg, len) = write_ngram_file(SMALL_ROWS, tag);
    let table = Arc::new(NgramTable::open(&dir, &cfg).unwrap());
    (dir, table, len)
}

#[test]
fn the_test_fixture_opens_a_real_small_ngram_table() {
    let (dir, table, len) = open_small("open");
    assert_eq!(table.rows, SMALL_ROWS);
    assert_eq!(table.words, inventory::ngram_row_words(K));
    assert!(len > (SMALL_ROWS * inventory::ngram_row_words(K) * 2) as u64);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn warm_reports_completion_only_after_every_byte_is_read() {
    let (dir, table, len) = open_large();
    let h = table.warm();
    // 64 MiB of real data through 1 MiB chunks: the worker cannot have finished this fast.
    assert!(h.done().is_none(), "warm worker finished before the test could observe it running");
    h.wait().unwrap();
    assert_eq!(h.done(), Some(Ok(())));
    assert_eq!(h.bytes_read(), len, "every byte of the mmap file must be faulted");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn finish_warm_blocks_until_the_worker_completes() {
    let (dir, table, len) = open_large();
    let h = table.warm();
    // the readiness gate: finish must not return before the worker's final byte is read
    finish_warm(Some(h.clone())).unwrap();
    assert_eq!(h.bytes_read(), len);
    assert_eq!(h.done(), Some(Ok(())));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn warm_failure_propagates_an_explicit_error_naming_the_file() {
    let (dir, table, _len) = open_small("fail");
    // the map stays valid (it owns its fd), but the worker's own open now fails: a real I/O failure
    std::fs::remove_file(dir.join(inventory::NGRAM_FILE)).unwrap();
    let h = table.warm();
    let err = finish_warm(Some(h.clone())).unwrap_err();
    assert!(err.contains(inventory::NGRAM_FILE), "{err}");
    assert!(err.contains("warm"), "{err}");
    assert_eq!(h.done(), Some(Err(err.clone())));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn disabled_warm_starts_no_worker_and_finishes_immediately() {
    let (dir, table, _len) = open_small("off");
    assert!(warm_if(false, &table).is_none());
    assert!(finish_warm(None).is_ok());
    assert!(warm_preflight(false, Some(&table)).is_ok(), "off bypass must not touch the table");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn repeated_warm_shares_one_worker_and_stays_idempotent() {
    let (dir, table, _len) = open_small("idem");
    let h1 = table.warm();
    let h2 = table.warm();
    assert!(Arc::ptr_eq(&h1, &h2), "a second warm must reuse the first worker's handle");
    h1.wait().unwrap();
    let h3 = table.warm();
    assert!(Arc::ptr_eq(&h1, &h3));
    assert_eq!(h3.done(), Some(Ok(())));
    assert!(warm_preflight(true, Some(&table)).is_ok());
    let _ = std::fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------------------------------------
// gather: new paths against the original implementation
// ---------------------------------------------------------------------------------------------

/// Rows of the equivalence fixture (K=8, 162-byte rows, ~19 MiB): every head spans the whole table.
const EQ_ROWS: usize = 120_000;

/// The gather exactly as it was before the latency work (per-position `vec!`, `row_ids`, word by
/// word `u16::from_le_bytes`): the oracle every new path must match bit for bit.
fn oracle_gather(t: &NgramTable, history: &[u32], ctx: usize, out: &mut [u16]) -> Result<(), String> {
    if ctx < 2 || ctx > history.len() {
        return Err("history needs two context tokens".into());
    }
    let n = history.len() - ctx;
    if out.len() < n * t.heads * t.words {
        return Err("packed row buffer is too short".into());
    }
    let all = t.inner.map.bytes();
    for (p, dst) in out.chunks_exact_mut(t.heads * t.words).take(n).enumerate() {
        let q = p + ctx;
        let mut ids = vec![0; t.heads];
        t.row_ids(history[q], history[q - 1], history[q - 2], &mut ids)?;
        for (j, &id) in ids.iter().enumerate() {
            let at = t.inner.base + id as usize * t.words * 2;
            for (w, b) in dst[j * t.words..(j + 1) * t.words].iter_mut().zip(all[at..at + t.words * 2].chunks_exact(2)) {
                *w = u16::from_le_bytes(b.try_into().unwrap());
            }
        }
    }
    Ok(())
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn variants() -> Vec<(&'static str, Tuning)> {
    let base = Tuning::default();
    vec![
        ("serial", base.clone()),
        ("pool4", Tuning { threads: 4, par_min_positions: 1, ..base.clone() }),
        ("pool4+willneed", Tuning { threads: 4, par_min_positions: 1, willneed: true, ..base.clone() }),
        ("willneed", Tuning { willneed: true, ..base.clone() }),
        // 1 MiB of 162-byte rows is ~6k slots for 120k rows x 16 heads: constant conflict evictions
        ("cache1", Tuning { cache_mib: 1, ..base.clone() }),
        ("pool4+willneed+cache1", Tuning { threads: 4, par_min_positions: 1, willneed: true, cache_mib: 1, ahead: false }),
        ("pool3-default-min", Tuning { threads: 3, ..base }),
    ]
}

fn open_eq(tag: &str, tuning: Tuning) -> (PathBuf, NgramTable) {
    let (dir, cfg, _) = write_ngram_file(EQ_ROWS, tag);
    let t = NgramTable::open_with(&dir, &cfg, tuning).unwrap();
    (dir, t)
}

/// Token streams that hit the hash's edge cases. `eos` runs, 0, u32::MAX, repeats, long runs.
fn adversarial_streams(eos: u32) -> Vec<(String, Vec<u32>)> {
    let mut s = 0x1234_5678_9ABC_DEF0u64;
    let mut out: Vec<(String, Vec<u32>)> = Vec::new();
    let random: Vec<u32> = (0..2200).map(|_| xorshift(&mut s) as u32).collect();
    out.push(("random-u32".into(), random));
    let vocab: Vec<u32> = (0..2200).map(|_| (xorshift(&mut s) % 248_320) as u32).collect();
    out.push(("random-vocab".into(), vocab.clone()));
    // eos in every pattern that matters: at the context slots, mid-stream, back to back, last
    let mut e = vocab.clone();
    for i in [0usize, 1, 2, 3, 5, 6, 7, 64, 65, 66, 127, 128, 129, 1000, 1001, 2198, 2199] {
        e[i] = eos;
    }
    out.push(("eos-boundaries".into(), e));
    out.push(("all-eos".into(), vec![eos; 300]));
    out.push(("all-zero".into(), vec![0; 300]));
    out.push(("all-max".into(), vec![u32::MAX; 300]));
    out.push(("eos+1/-1".into(), (0..300).map(|i| if i % 2 == 0 { eos.wrapping_add(1) } else { eos.wrapping_sub(1) }).collect()));
    out.push(("alternating-eos".into(), (0..400).map(|i| if i % 2 == 0 { eos } else { (i * 7919) as u32 }).collect()));
    // heavy repetition: every row is a cache hit after the first period
    out.push(("repeating-period-5".into(), (0..1500).map(|i| 100 + (i % 5) as u32).collect()));
    out
}

fn check_equal(t: &NgramTable, hist: &[u32], ctx: usize, what: &str) {
    let per = t.heads * t.words;
    let n = hist.len() - ctx;
    let mut want = vec![0u16; n * per];
    oracle_gather(t, hist, ctx, &mut want).unwrap();
    // an output buffer longer than needed, pre-filled with a sentinel: the tail must stay untouched
    let mut got = vec![0xBEEFu16; n * per + 17];
    t.gather(hist, ctx, &mut got).unwrap();
    assert_eq!(&got[..n * per], &want[..], "{what}: gather (ctx {ctx}, n {n})");
    assert!(got[n * per..].iter().all(|&w| w == 0xBEEF), "{what}: wrote past the positions");
    let mut got = vec![0u16; n * per];
    t.gather_serial(hist, ctx, &mut got).unwrap();
    assert_eq!(got, want, "{what}: gather_serial (ctx {ctx}, n {n})");
}

#[test]
fn every_gather_path_is_bit_identical_to_the_original_on_random_and_adversarial_ids() {
    for (name, tuning) in variants() {
        let (dir, t) = open_eq(&format!("eq-{}", name.replace('+', "_")), tuning);
        for (stream, tokens) in adversarial_streams(t.eos) {
            for ctx in [2usize, 3, 9] {
                // every length around a worker block boundary, the whole stream, and the 1-position case
                for n in [1usize, 2, 7, 63, 64, 65, 127, 128, 129, tokens.len() - ctx] {
                    let hist = &tokens[..(ctx + n).min(tokens.len())];
                    // twice: the second pass runs against a warm row cache
                    check_equal(&t, hist, ctx, &format!("{name}/{stream}"));
                    check_equal(&t, hist, ctx, &format!("{name}/{stream}/warm"));
                }
            }
        }
        // zero positions is a valid empty gather
        let mut none = [0u16; 0];
        t.gather(&[1, 2], 2, &mut none).unwrap();
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[test]
fn gather_argument_errors_are_unchanged() {
    let (dir, t) = open_eq("errs", Tuning { threads: 4, par_min_positions: 1, ..Tuning::default() });
    let mut out = vec![0u16; t.heads * t.words * 4];
    for call in [
        t.gather(&[1, 2, 3], 1, &mut out),
        t.gather(&[1], 2, &mut out),
        t.gather(&[1, 2, 3, 4, 5, 6], 2, &mut out[..t.heads * t.words * 3]),
        t.prefetch(&[1, 2, 3], 1),
    ] {
        assert!(call.is_err());
    }
    assert!(Arc::new(t).gather_ahead(vec![1, 2, 3], 1).is_err());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn prefetch_and_ahead_never_change_the_gathered_bytes() {
    let tuning = Tuning { threads: 4, par_min_positions: 1, willneed: true, cache_mib: 1, ..Tuning::default() };
    let (dir, t) = open_eq("ahead", tuning);
    let t = Arc::new(t);
    let (_, tokens) = adversarial_streams(t.eos).into_iter().nth(2).unwrap();
    let ctx = 2;
    // prefetch first, then the gather: same bytes as the oracle
    t.prefetch(&tokens, ctx).unwrap();
    check_equal(&t, &tokens, ctx, "after prefetch");
    // ahead with the matching request returns the oracle's bytes ...
    let per = t.heads * t.words;
    let mut want = vec![0u16; (tokens.len() - ctx) * per];
    oracle_gather(&t, &tokens, ctx, &mut want).unwrap();
    let ahead = t.gather_ahead(tokens.clone(), ctx).unwrap();
    assert_eq!(ahead.take(&tokens, ctx).expect("same request"), want);
    // ... a different token anywhere, a different length, or a different ctx discards it
    let mut other = tokens.clone();
    other[1500] ^= 1;
    assert!(t.gather_ahead(tokens.clone(), ctx).unwrap().take(&other, ctx).is_none());
    assert!(t.gather_ahead(tokens.clone(), ctx).unwrap().take(&tokens[..tokens.len() - 1], ctx).is_none());
    assert!(t.gather_ahead(tokens.clone(), ctx).unwrap().take(&tokens, 3).is_none());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn concurrent_gathers_on_one_table_with_a_tiny_row_cache_stay_exact() {
    let tuning = Tuning { threads: 4, par_min_positions: 1, willneed: true, cache_mib: 1, ..Tuning::default() };
    let (dir, t) = open_eq("conc", tuning);
    let t = Arc::new(t);
    let mut s = 99u64;
    // overlapping vocab so threads fight for the same cache slots
    let streams: Vec<Vec<u32>> =
        (0..6).map(|_| (0..1500).map(|_| (xorshift(&mut s) % 400) as u32).collect()).collect();
    let per = t.heads * t.words;
    let wants: Vec<Vec<u16>> = streams
        .iter()
        .map(|h| {
            let mut w = vec![0u16; (h.len() - 2) * per];
            oracle_gather(&t, h, 2, &mut w).unwrap();
            w
        })
        .collect();
    std::thread::scope(|sc| {
        for (h, want) in streams.iter().zip(&wants) {
            let t = &t;
            sc.spawn(move || {
                for _ in 0..40 {
                    let mut got = vec![0u16; want.len()];
                    t.gather(h, 2, &mut got).unwrap();
                    assert_eq!(&got, want);
                }
            });
        }
    });
    let (hits, misses) = t.cache_stats();
    assert!(hits > 0 && misses > 0, "the cache must have been exercised: {hits} hits {misses} misses");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn row_cache_hit_returns_the_mmap_bytes_for_every_row_width() {
    // the wide K=8 rows (162 B) and the K=6 rows (122 B), including a width that is not
    // a multiple of 8
    for row_bytes in [122usize, 162, 8, 1] {
        let c = RowCache::new(1, row_bytes);
        let mut s = 7u64;
        for id in [0u32, 1, 2, 17, 100_000, u32::MAX - 1] {
            let src: Vec<u8> = (0..row_bytes).map(|_| xorshift(&mut s) as u8).collect();
            let mut dst = vec![0u8; row_bytes];
            assert!(!c.get(id, &mut dst));
            c.put(id, &src);
            assert!(c.get(id, &mut dst));
            assert_eq!(dst, src);
            assert!(c.contains(id));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// microbenchmark (ignored): the real pack's file; see the NGRAM_BENCH_* variables below
// ---------------------------------------------------------------------------------------------

fn evict_file(path: &Path) {
    // the harness owns the node while its lock directory exists: stop before disturbing its run
    if let Ok(lock) = std::env::var("NGRAM_BENCH_LOCK") {
        assert!(!Path::new(&lock).exists(), "node lock {lock} appeared: aborting the benchmark");
    }
    use std::os::fd::AsRawFd;
    let f = std::fs::File::open(path).unwrap();
    // SAFETY: plain fadvise on an open fd. Drops this file's clean cached pages (not mapped ones).
    unsafe {
        libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

fn resident_pages(path: &Path) -> (usize, usize) {
    use std::os::fd::AsRawFd;
    let f = std::fs::File::open(path).unwrap();
    let len = f.metadata().unwrap().len() as usize;
    // SAFETY: a read-only map used only for mincore, unmapped below.
    unsafe {
        let p = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), 0);
        assert!(p != libc::MAP_FAILED);
        let pages = len.div_ceil(4096);
        let mut v = vec![0u8; pages];
        libc::mincore(p, len, v.as_mut_ptr().cast());
        libc::munmap(p, len);
        (v.iter().filter(|&&b| b & 1 != 0).count(), pages)
    }
}

fn stats_ms(mut v: Vec<f64>) -> String {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    format!("n={} med={:.3} mean={:.3} p90={:.3} p99={:.3} max={:.3} ms", v.len(), q(0.5), mean, q(0.9), q(0.99), v[v.len() - 1])
}

fn read_tokens(path: &str) -> Vec<u32> {
    std::fs::read(path).unwrap().chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()
}

/// Run with: NGRAM_BENCH_DIR=<pack dir> NGRAM_BENCH_TOKENS=<u32-LE token file(s), comma separated>
/// [NGRAM_BENCH_SECTIONS=chunk,window,ahead,cache] cargo test --release -p gb-model ngram_bench -- --ignored --nocapture
#[test]
#[ignore]
fn ngram_bench() {
    let dir = PathBuf::from(std::env::var("NGRAM_BENCH_DIR").expect("NGRAM_BENCH_DIR"));
    let token_files = std::env::var("NGRAM_BENCH_TOKENS").expect("NGRAM_BENCH_TOKENS");
    let streams: Vec<Vec<u32>> = token_files.split(',').map(read_tokens).collect();
    let moby = &streams[0];
    let sections = std::env::var("NGRAM_BENCH_SECTIONS").unwrap_or_else(|_| "chunk,window,ahead,cache".into());
    let on = |s: &str| sections.split(',').any(|x| x == s);
    let cfg = FlashNextConfig::from_dir(&dir).unwrap();
    let file = dir.join(inventory::NGRAM_FILE);
    let vocab = cfg.vocab as u64;
    eprintln!("BENCH moby tokens {} streams {}", moby.len(), streams.len());
    let (r0, tot) = resident_pages(&file);
    evict_file(&file);
    let (r1, _) = resident_pages(&file);
    eprintln!("BENCH file pages resident before/after posix_fadvise(DONTNEED): {r0} / {r1} of {tot}");

    // (label, None = the original gather on the open table, Some(tuning))
    let base = Tuning::default();
    let mut vs: Vec<(String, Option<Tuning>)> = vec![("old".into(), None), ("new-serial".into(), Some(base.clone()))];
    for th in [4usize, 8, 16, 32, 64] {
        vs.push((format!("pool{th}"), Some(Tuning { threads: th, par_min_positions: 1, ..base.clone() })));
    }
    for th in [1usize, 4, 8, 16] {
        vs.push((format!("pool{th}+willneed"), Some(Tuning { threads: th, par_min_positions: 1, willneed: true, ..base.clone() })));
    }
    vs.retain(|(l, _)| std::env::var("NGRAM_BENCH_ONLY").map_or(true, |o| o.split(',').any(|x| x == l)));

    let run = |t: &NgramTable, old: bool, hist: &[u32], out: &mut [u16]| {
        let t0 = std::time::Instant::now();
        if old {
            oracle_gather(t, hist, 2, out).unwrap();
        } else {
            t.gather(hist, 2, out).unwrap();
        }
        t0.elapsed().as_secs_f64() * 1e3
    };

    if on("chunk") {
        // 4096-row prefill chunks; cold = a fresh evicted file, each chunk a different region of the stream
        let reps = 4usize;
        for (label, tuning) in &vs {
            evict_file(&file);
            let t = NgramTable::open_with(&dir, &cfg, tuning.clone().unwrap_or_default()).unwrap();
            let per = t.heads * t.words;
            let mut out = vec![0u16; 4096 * per];
            let mut cold = Vec::new();
            for r in 0..reps {
                let off = 1000 + r * 50_000;
                cold.push(run(&t, tuning.is_none(), &moby[off..off + 4098], &mut out));
            }
            let mut hot = Vec::new();
            for r in 0..reps {
                let off = 1000 + r * 50_000;
                hot.push(run(&t, tuning.is_none(), &moby[off..off + 4098], &mut out));
            }
            let mut rnd = Vec::new();
            let mut s = 42u64;
            let ev: Vec<u32> = (0..4098).map(|_| (xorshift(&mut s) % vocab) as u32).collect();
            evict_file(&file);
            drop(t);
            evict_file(&file);
            let t = NgramTable::open_with(&dir, &cfg, tuning.clone().unwrap_or_default()).unwrap();
            rnd.push(run(&t, tuning.is_none(), &ev, &mut out));
            eprintln!(
                "BENCH chunk4096 {label:>18}: cold-moby {} | hot {} | cold-random-tokens {:.1} ms",
                stats_ms(cold),
                stats_ms(hot),
                rnd[0]
            );
        }
    }

    if on("window") {
        // 7-row verify windows: 100 different regions per variant, cold file first, then the same ones hot
        let wn = 100usize;
        for (label, tuning) in &vs {
            for (kind, rand) in [("moby", false), ("random", true)] {
                evict_file(&file);
                let t = NgramTable::open_with(&dir, &cfg, tuning.clone().unwrap_or_default()).unwrap();
                let per = t.heads * t.words;
                let mut out = vec![0u16; 7 * per];
                let mut s = 4242u64;
                let hists: Vec<Vec<u32>> = (0..wn)
                    .map(|i| {
                        if rand {
                            (0..9).map(|_| (xorshift(&mut s) % vocab) as u32).collect()
                        } else {
                            let off = 2000 + i * 2500;
                            moby[off..off + 9].to_vec()
                        }
                    })
                    .collect();
                let cold: Vec<f64> = hists.iter().map(|h| run(&t, tuning.is_none(), h, &mut out)).collect();
                let hot: Vec<f64> = hists.iter().map(|h| run(&t, tuning.is_none(), h, &mut out)).collect();
                eprintln!("BENCH window7 {label:>18} {kind:>6}: cold {} | hot {}", stats_ms(cold), stats_ms(hot));
            }
        }
    }

    if on("ahead") {
        // prefetch, wait `gap` (stand-in for the GPU running the previous chunk / step), then gather
        let t = NgramTable::open_with(&dir, &cfg, Tuning { threads: 8, par_min_positions: 1, ..base.clone() }).unwrap();
        drop(t);
        for (kind, rows, gaps_ms) in [("chunk4096", 4096usize, vec![0u64, 100, 400, 1500]), ("window7", 7, vec![0, 1, 3, 10])] {
            for gap in gaps_ms {
                evict_file(&file);
                let t = NgramTable::open_with(&dir, &cfg, Tuning { threads: 16, par_min_positions: 1, ..base.clone() }).unwrap();
                let per = t.heads * t.words;
                let mut out = vec![0u16; rows * per];
                let (mut pre, mut after, mut direct) = (Vec::new(), Vec::new(), Vec::new());
                let reps = if rows > 100 { 3 } else { 40 };
                let mut s = 777 + gap;
                for r in 0..reps {
                    let h: Vec<u32> = (0..rows + 2).map(|_| (xorshift(&mut s) % vocab) as u32).collect();
                    let h2: Vec<u32> = (0..rows + 2).map(|_| (xorshift(&mut s) % vocab) as u32).collect();
                    let _ = r;
                    let t0 = std::time::Instant::now();
                    t.prefetch(&h, 2).unwrap();
                    pre.push(t0.elapsed().as_secs_f64() * 1e3);
                    std::thread::sleep(std::time::Duration::from_millis(gap));
                    after.push(run(&t, false, &h, &mut out));
                    direct.push(run(&t, false, &h2, &mut out)); // never prefetched: fully cold
                }
                eprintln!(
                    "BENCH ahead {kind:>9} gap {gap:>5} ms: prefetch call {} | gather after prefetch {} | gather without prefetch (pool16) {}",
                    stats_ms(pre),
                    stats_ms(after),
                    stats_ms(direct)
                );
            }
        }
    }

    if on("sizes") {
        // gather size sweep (positions; 16 rows each): old vs the production tuning vs its serial variant, cold then hot
        let prod = Tuning::production();
        let sv: Vec<(&str, Option<Tuning>)> = vec![
            ("old", None),
            ("serial+willneed", Some(Tuning { threads: 1, ..prod.clone() })),
            ("production(par>=64)", Some(prod.clone())),
            ("production(par>=1)", Some(Tuning { par_min_positions: 1, ..prod.clone() })),
        ];
        for n in [1usize, 7, 32, 63, 64, 128, 512, 4096] {
            for (label, tuning) in &sv {
                evict_file(&file);
                let t = NgramTable::open_with(&dir, &cfg, tuning.clone().unwrap_or_default()).unwrap();
                let per = t.heads * t.words;
                let mut out = vec![0u16; n * per];
                let reps = if n >= 512 { 4 } else { 30 };
                let mut s = 31337 + n as u64;
                let hs: Vec<Vec<u32>> = (0..reps).map(|_| (0..n + 2).map(|_| (xorshift(&mut s) % vocab) as u32).collect()).collect();
                let cold: Vec<f64> = hs.iter().map(|h| run(&t, tuning.is_none(), h, &mut out)).collect();
                let hot: Vec<f64> = hs.iter().map(|h| run(&t, tuning.is_none(), h, &mut out)).collect();
                eprintln!("BENCH size {n:>5} positions {label:>20}: cold {} | hot {}", stats_ms(cold), stats_ms(hot));
            }
        }
    }

    if on("cache") {
        // hit rate: real RowCache (direct-mapped) vs an LRU page cache of the same memory, on the token streams,
        // ids from the real hash params; no table bytes are read.
        let t = NgramTable::open_with(&dir, &cfg, base.clone()).unwrap();
        let heads = t.heads;
        let rb = t.words * 2;
        let ids_of = |s: &[u32]| -> Vec<u32> {
            let mut v = vec![0u32; (s.len() - 2) * heads];
            t.inner.fill_block(s, 2, 0, &mut v);
            v
        };
        let all_ids: Vec<Vec<u32>> = streams.iter().map(|s| ids_of(s)).collect();
        for (i, ids) in all_ids.iter().enumerate() {
            let uniq: std::collections::HashSet<u32> = ids.iter().copied().collect();
            let pages: std::collections::HashSet<u64> = uniq.iter().map(|&r| (r as u64 * rb as u64 + t.inner.base as u64) / 4096).collect();
            eprintln!(
                "BENCH cache stream{i}: {} tokens, {} row lookups, {} unique rows ({:.1}% compulsory-miss floor on a cold cache), {} unique pages, unique rows MiB {:.1}, unique pages MiB {:.1}",
                ids.len() / heads,
                ids.len(),
                uniq.len(),
                100.0 * uniq.len() as f64 / ids.len() as f64,
                pages.len(),
                uniq.len() as f64 * rb as f64 / 1048576.0,
                pages.len() as f64 * 4096.0 / 1048576.0
            );
        }
        for mib in [64usize, 256, 1024, 4096] {
            // row cache: stream 0 cold, then (same cache) every later stream, then stream 0 again
            let c = RowCache::new(mib, rb);
            let mut dummy = vec![0u8; rb];
            let mut report = String::new();
            let order: Vec<usize> = (0..all_ids.len()).chain(std::iter::once(0)).collect();
            for &si in &order {
                let (mut h, mut m) = (0u64, 0u64);
                for &id in &all_ids[si] {
                    if c.get(id, &mut dummy) {
                        h += 1;
                    } else {
                        m += 1;
                        c.put(id, &dummy);
                    }
                }
                report += &format!(" s{si}:{:.1}%", 100.0 * h as f64 / (h + m) as f64);
            }
            // LRU of 4 KiB pages with the same memory
            let cap = mib << 20 >> 12;
            let mut stamp = 0u64;
            let mut at: std::collections::HashMap<u64, u64> = Default::default();
            let mut by: std::collections::BTreeMap<u64, u64> = Default::default();
            let mut preport = String::new();
            for &si in &order {
                let (mut h, mut m) = (0u64, 0u64);
                for &id in &all_ids[si] {
                    let page = (id as u64 * rb as u64 + t.inner.base as u64) / 4096;
                    stamp += 1;
                    if let Some(old) = at.insert(page, stamp) {
                        by.remove(&old);
                        h += 1;
                    } else {
                        m += 1;
                        if at.len() > cap {
                            let (&os, &op) = by.iter().next().unwrap();
                            by.remove(&os);
                            at.remove(&op);
                        }
                    }
                    by.insert(stamp, page);
                }
                preport += &format!(" s{si}:{:.1}%", 100.0 * h as f64 / (h + m) as f64);
            }
            eprintln!("BENCH cache {mib:>5} MiB ({} slots): row-cache hit rate{report} | LRU page cache{preport}", c.slots);
        }
    }
    let (res, tot) = resident_pages(&file);
    eprintln!("BENCH file pages resident at end: {res}/{tot}");
}
