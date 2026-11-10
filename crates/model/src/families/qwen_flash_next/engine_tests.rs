use super::*;
use crate::families::qwen_flash_next::backend::FlashNextRuntime;
use crate::families::qwen_flash_next::model::MtpAsk;
use crate::families::qwen_flash_next::ngram::warm_preflight;
use crate::forward::{Candidates, PrefillPath};
use crate::hostmem::MemInfo;
use crate::rows::RowBatch;
use gb_engine::ProposeRequest;
use gb_sampling::Sampling;

#[test]
fn planning_holds_back_weights_without_reserving_the_reclaimable_warm_table() {
    const G: u64 = 1 << 30;
    let host = MemInfo { total: 121 * G, free: G, available: 90 * G, cached: 89 * G };
    // The warmed n-gram table lives in reclaimable page cache (MemAvailable already counts it as
    // reclaimable): it must never be deducted twice as an anonymous/device reservation.
    assert_eq!(free_after_load(G, Some(&host), 40 * G), 50 * G);
    assert_eq!(free_after_load(80 * G, Some(&host), 40 * G), 50 * G);
    assert_eq!(free_after_load(5 * G, None, 40 * G), 0);
}

#[test]
fn warm_mode_from_env_defaults_to_auto_and_only_zero_is_off() {
    assert_eq!(warm_mode_from_env(None), WarmMode::Auto);
    assert_eq!(warm_mode_from_env(Some("auto")), WarmMode::Auto);
    assert_eq!(warm_mode_from_env(Some("0")), WarmMode::Off);
    assert_eq!(warm_mode_from_env(Some("1")), WarmMode::On);
    assert_eq!(warm_mode_from_env(Some("true")), WarmMode::On);
}

#[test]
fn auto_warm_decides_from_headroom_and_on_off_override_it() {
    const G: u64 = 1 << 30;
    let host = MemInfo { total: 128 * G, free: 2 * G, available: 99 * G, cached: 50 * G };
    let (weights, table, floor) = (61 * G, 36 * G, 2 * G);
    assert!(wants_warm(WarmMode::Auto, Some(&host), weights, table, floor));
    let tight = MemInfo { total: 128 * G, free: 2 * G, available: 60 * G, cached: 40 * G };
    assert!(!wants_warm(WarmMode::Auto, Some(&tight), weights, table, floor));
    assert!(wants_warm(WarmMode::On, Some(&tight), weights, table, floor));
    assert!(!wants_warm(WarmMode::Off, Some(&host), weights, table, floor));
    // no host figure (e.g. an old kernel): plan from the device, and warm by default
    assert!(wants_warm(WarmMode::Auto, None, weights, table, floor));
}

#[test]
fn cf_131k_plan_recovers_full_kv_without_reserving_the_warm_table() {
    const G: u64 = 1 << 30;
    // cruzspark CF numbers from ple-host-readback-analysis.md §6: pack 60.65 GiB, table 36.36 GiB,
    // MemAvailable 122 GiB at boot. The warm table must not shrink the KV plan below CF 131072.
    let host = MemInfo { total: 128 * G, free: 4 * G, available: 122 * G, cached: 30 * G };
    let weights = (60.65 * G as f64) as u64;
    let table = (36.36 * G as f64) as u64;
    let free = free_after_load(110 * G, Some(&host), weights);
    assert!(free >= 58 * G, "free after weights must not lose the table: {free}");
    // auto still decides to warm: plenty of reclaimable headroom after the weights
    assert!(wants_warm(WarmMode::Auto, Some(&host), weights, table, 2 * G));
}

#[test]
fn warm_preflight_returns_an_error_not_serving_before_completion() {
    // The preflight gate is exercised with real small mmaps in ngram_tests.rs; here the
    // engine-level failure text and the off/no-table bypass are pinned.
    assert!(warm_preflight(false, None).is_ok());
    assert!(warm_preflight(true, None).is_ok(), "no PLE table means no warm preflight");
    let err = super::super::ngram::finish_warm(Some(std::sync::Arc::new(super::super::ngram::WarmHandle::new_failed_for_test(
        "ngram_embedding.safetensors: test warm read failure".into(),
    ))));
    assert!(err.unwrap_err().contains("warm read failure"));
}

#[test]
fn unbuilt_vision_or_window_width_is_refused_before_loading() {
    let mut o = crate::serve::ServeOptions::new(std::path::PathBuf::from("/nonexistent"));
    o.window_rows = 128;
    assert!(check_flash_options(&o).unwrap_err().to_string().contains("--window-rows"));
    o.window_rows = 64;
    assert!(check_flash_options(&o).is_ok());
}

/// Flash-Next cuts with the interactive policy (the default one serves neither a second question about a document nor a short tool
/// loop); the environment still overrides each value, and the default config path is unchanged.
#[test]
fn flash_next_cuts_for_interactive_use_and_the_environment_still_decides() {
    use gb_sched::prefix::CutPolicy;
    let from = |vars: &[(&str, &str)]| {
        let vars: Vec<(String, String)> = vars.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        flash_next_prefix_config(&move |n| vars.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone()))
    };
    assert_eq!(from(&[]).unwrap().policy, CutPolicy::interactive());
    assert_ne!(CutPolicy::interactive(), CutPolicy::default());
    let c = from(&[("REXL3_PREFIX_TAIL", "16"), ("REXL3_PREFIX_MIN_CUT", "512")]).unwrap();
    assert_eq!((c.policy.tail, c.policy.min_cut), (16, 512));
    assert_eq!((c.policy.min_gap, c.policy.ghost_min_gain), (128, 256), "what is not overridden stays interactive");
    assert!(from(&[("REXL3_PREFIX_TAIL", "long")]).is_err());
    assert_eq!(crate::serve::prefix_config_from(&|_| None).unwrap(), gb_engine::PrefixConfig::default());
}

fn write_test_sidecar(dir: &std::path::Path, file: &str, name: &str, payload: &[u8]) {
    let header =
        serde_json::json!({name: {"dtype": "U8", "shape": [payload.len()], "data_offsets": [0, payload.len()]}}).to_string();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    std::fs::write(dir.join(file), bytes).unwrap();
}

fn unknown_pack_dir() -> std::path::PathBuf {
    // one directory per call: the tests of this module run in parallel and each removes its own at the end (a name keyed by the process
    // alone made one test delete the directory another was validating)
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("rexl3-fn-unknown-pack-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.safetensors.index.json"), r#"{"weight_map":{}}"#).unwrap();
    dir
}

#[test]
fn qwen_build_pack_validation_refuses_unknown_pack_before_load_callback() {
    let dir = unknown_pack_dir();
    let mut loaded = false;
    let result = validate_pack_before_load(&dir, false, || {
        loaded = true;
        Ok(())
    });
    assert!(result.unwrap_err().to_string().contains("not one this release was validated on"));
    assert!(!loaded, "unknown packs must be refused before CUDA or weight loading");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn qwen_build_pack_validation_records_override_before_load_callback() {
    let dir = unknown_pack_dir();
    let mut loaded = false;
    validate_pack_before_load(&dir, true, || {
        loaded = true;
        Ok(())
    })
    .unwrap();
    assert!(loaded);
    assert!(gb_engine::boot::facts()
        .into_iter()
        .find(|(name, _)| *name == "pack_validated")
        .is_some_and(|(_, value)| !value.is_empty()));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn namespace_fingerprint_changes_with_config_and_both_sidecar_headers_and_sizes() {
    let dir = std::env::temp_dir().join(format!("rexl3-fn-fingerprint-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), "{}").unwrap();
    // The indexed model shards are scanned by upstream; sidecars are deliberately not in this index.
    std::fs::write(dir.join("model.safetensors.index.json"), r#"{"weight_map":{}}"#).unwrap();
    let ng = super::super::inventory::NGRAM_FILE;
    let mtp = super::super::inventory::MTP_MIXER_FILE;
    write_test_sidecar(&dir, ng, "ngram", b"abcd");
    write_test_sidecar(&dir, mtp, "mixer", b"efgh");
    let first = pack_header_fingerprint(&dir).unwrap();
    std::fs::write(dir.join("config.json"), r#"{"changed":1}"#).unwrap();
    assert_ne!(first, pack_header_fingerprint(&dir).unwrap());
    std::fs::write(dir.join("config.json"), "{}").unwrap();
    write_test_sidecar(&dir, ng, "ngram_changed", b"abcd");
    assert_ne!(first, pack_header_fingerprint(&dir).unwrap());
    write_test_sidecar(&dir, ng, "ngram", b"abcde");
    assert_ne!(first, pack_header_fingerprint(&dir).unwrap());
    write_test_sidecar(&dir, ng, "ngram", b"abcd");
    write_test_sidecar(&dir, mtp, "mixer_changed", b"efgh");
    assert_ne!(first, pack_header_fingerprint(&dir).unwrap());
    write_test_sidecar(&dir, mtp, "mixer", b"efgh");
    assert_eq!(first, pack_header_fingerprint(&dir).unwrap());
    // The cheap fingerprint intentionally does not read tensor values.
    write_test_sidecar(&dir, mtp, "mixer", b"wxyz");
    assert_eq!(first, pack_header_fingerprint(&dir).unwrap());
    let mut grown = std::fs::read(dir.join(mtp)).unwrap();
    grown.push(0);
    std::fs::write(dir.join(mtp), grown).unwrap();
    assert_ne!(first, pack_header_fingerprint(&dir).unwrap());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn probed_flash_next_plan_counts_prefix_arena_outside_admission() {
    use crate::family::ModelFamily;
    use crate::plan::{plan_memory_cached, CacheRequest};
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/qwen_flash_next");
    let plan = super::super::FlashNextFamily.probe(&dir).unwrap();
    assert!(plan.missing_components().is_empty());
    let cfg = FlashNextConfig::from_dir(&dir).unwrap();
    let profile = plan.memory_profile(&MemoryOptions { mtp: true, vision: None, ..MemoryOptions::serving_defaults() }).unwrap();
    let snapshot = super::super::prefix::snapshot_bytes(&cfg, true);
    let p = plan_memory_cached(&profile, 40 << 30, 2 << 30, 2, 8192, 0, CacheRequest::Fixed(3 << 30), snapshot, 0).unwrap();
    assert!(p.cache.enabled());
    assert_eq!(p.cache.snapshot_bytes, snapshot);
    assert_eq!(p.cache.page_bytes, profile.kv_page_bytes());
    assert_eq!(
        p.planned_bytes(2),
        p.engine_reserve_bytes + p.state_bytes_per_slot * 2 + p.kv_blocks as u64 * p.block_bytes + p.cache.bytes()
    );
    assert_eq!(p.kv_tokens, (p.kv_blocks - 2) as u64 * 256);
    let mm = gb_sched::ModelMemory::from_profile(&profile, 0, 0);
    let admission = p.admission(mm, 2, 8192).unwrap();
    assert_eq!(admission.request_limit(), Some(8192));
}

// ---------------------------------------------------------------------------
// Batched speculative decoding: a multi-request decode step must still speculate.
//
// The Flash-Next MTP head drafts for up to `MTP_MAX_SEQS` sequences in one chain (`propose_batch_scored`); a stream it does not
// carry (no MTP head, a fifth stream, no trunk rows on the stack) is answered `None`, and the engine's per-sequence Drafter fills
// that gap: when drafting is on, `make_drafter(true)` installs a CopyDrafter per stream, so a full round still speculates. The
// tests below that run with no native head (`mtp_k = 0`) drive that fallback through the real engine over the CPU Fake runtime
// with several concurrent requests and assert that drafts are proposed/verified/accepted per request and that greedy output is
// identical to the no-speculation path.

#[derive(Default)]
struct CpuRuntime {
    /// how many times the MTP head was asked to propose (one ask per native draft round)
    mtp_asks: usize,
}
impl FlashNextRuntime for CpuRuntime {
    fn vocab_size(&self) -> usize {
        42
    }
    fn has_mtp(&self) -> bool {
        true
    }
    fn begin_slot(&mut self, _: usize) -> Result<(), ModelError> {
        Ok(())
    }
    fn end_slot(&mut self, _: usize) {}
    fn prefill_chunk(
        &mut self,
        _b: &RowBatch,
        _path: PrefillPath,
        want: Option<usize>,
    ) -> Result<Option<Candidates>, ModelError> {
        Ok(want.map(|count| Candidates { rows: 1, count, ids: vec![7; count], vals: vec![1.0; count] }))
    }
    fn mtp_after_prefill(
        &mut self,
        _slot: usize,
        _first: usize,
        _chunk: &[u32],
        _next: Option<u32>,
        _path: PrefillPath,
    ) -> Result<(), ModelError> {
        Ok(())
    }
    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        Ok(Candidates { rows: b.rows, count, ids: vec![7; b.rows * count], vals: vec![1.0; b.rows * count] })
    }
    fn commit(&mut self, _: usize, _keep: usize) {}
    fn flush(&mut self) -> Result<(), ModelError> {
        Ok(())
    }
    fn mtp_propose_batch(&mut self, asks: &[MtpAsk<'_>]) -> Result<Vec<Option<(Vec<u32>, Vec<f32>)>>, ModelError> {
        self.mtp_asks += 1;
        Ok(asks.iter().map(|a| Some((vec![7; a.k], vec![0.5; a.k]))).collect())
    }
}

/// The real engine over the CPU runtime, with the production drafter factory
/// (`make_drafter`) so the fallback under test is exactly what ships.
fn batched_engine(drafting: bool) -> Engine<FlashNextBackend<CpuRuntime>> {
    use gb_sched::{Admission, CostModel, ModelMemory};
    let mm = ModelMemory {
        weights_bytes: 1 << 30,
        kv_bytes_per_token: 65536,
        state_bytes_per_slot: 150 << 20,
        activation_reserve: 1 << 30,
    };
    let admission = Admission::new(64 << 30, 6 << 30, mm, 8).unwrap();
    // no native head: the CopyDrafter fallback is what ships for a stream the MTP chain does not carry
    let backend = FlashNextBackend::with_runtime(CpuRuntime::default(), 64, 0, true);
    let make = make_drafter(drafting);
    Engine::new(
        backend,
        admission,
        EngineConfig {
            max_rows: 8,
            max_drafts: 7,
            cost: CostModel::from_table(vec![(1, 70.0), (8, 90.0), (16, 120.0)]),
            drafting,
        },
        make,
    )
}

fn repetitive_requests(n: usize) -> Vec<gb_engine::Request> {
    (0..n)
        .map(|i| gb_engine::Request {
            sampling: Sampling { seed: 3 + i as u64, temperature: 0.0, ..Default::default() },
            prompt: vec![7; 24],
            max_new: 40,
            stop: vec![],
            draft: true,
            images: vec![],
            cache: false,
        })
        .collect()
}

/// The native MTP head drafts for every request of a batch in one chain: a 2-request batch gets one call and a proposal each,
/// every one with the head's probabilities as its accept chances.
#[test]
fn native_mtp_drafts_every_request_of_a_batch_in_one_call() {
    use gb_engine::Backend;
    let mut b = FlashNextBackend::with_runtime(CpuRuntime::default(), 64, 6, true);
    let sampling = Sampling::default();
    let multi = [
        ProposeRequest { slot: 1, tokens: &[3], sampling: &sampling, max: 6 },
        ProposeRequest { slot: 2, tokens: &[3], sampling: &sampling, max: 3 },
    ];
    let got = b.propose_batch_scored(&multi);
    assert_eq!(b.m.mtp_asks, 1, "one chain for the whole batch");
    let lens: Vec<usize> = got.iter().map(|p| p.as_ref().unwrap().tokens.len()).collect();
    assert_eq!(lens, vec![4, 3], "each stream keeps its own ask (the adaptive depth, bounded by its own max)");
    for p in got.iter().flatten() {
        assert_eq!(p.accept.len(), p.tokens.len());
        assert_eq!(p.conf.len(), p.tokens.len());
    }
}

/// With 2 and 4 concurrent requests, drafts are proposed, verified and accepted per
/// request (counters rise) and greedy output matches the no-speculation serial path.
#[test]
fn batched_concurrent_requests_still_draft_and_match_greedy() {
    use gb_engine::harness::drafted_equals_serial;
    for n in [2usize, 4] {
        let r = drafted_equals_serial(&batched_engine, &repetitive_requests(n)).unwrap();
        assert_eq!(r.sequences, n);
        assert!(r.drafted > 0, "{n} concurrent requests: no drafts proposed");
        assert!(r.accepted > 0, "{n} concurrent requests: no drafts accepted");
        eprintln!("{n} concurrent requests: drafted {} accepted {} over {} rounds", r.drafted, r.accepted, r.rounds);
    }
}

// ---------------------------------------------------------------------------
// Rejection-path tests (independent review; scratch only, never committed).
//
// The tests above use a repetitive prompt and a runtime whose candidates are
// always [7], so every copy draft is accepted (38/38 and 76/76) and the
// rejection path never runs. These tests use a position-dependent runtime whose
// greedy output is non-repetitive: copy drafts (an 8-token suffix match) are
// proposed repeatedly but mostly rejected, so `commit(slot, keep < rows)` and
// the recurrent-state rollback are exercised for real. Greedy output must still
// be byte-identical to the no-speculation serial path, and the counters must
// show drafted > accepted.
//
// Token id at absolute position p: blocks of 20 tokens. Block k holds
//   A_k = [0,1,..,7]          (positions 20k..20k+7)
//   X_k = [8,9,10, 31+k]      (positions 20k+8..20k+11; the divergence point)
//   F_k = [14+2k .. 21+2k]    (positions 20k+12..20k+19)
// The 8-gram [0..7] therefore recurs every 20 positions. The drafter proposes the
// continuation of the *previous* A block, which shares the first 3 tokens
// (8,9,10) and then diverges (31+k+1 vs 31+k), so every proposal round accepts
// exactly 3 of 7 drafts and rejects the rest.

#[derive(Default)]
struct DivergentRuntime {
    mtp_asks: usize,
}

fn f(p: usize) -> u32 {
    let k = (p / 20) as u32;
    match p % 20 {
        0..=7 => (p % 20) as u32,
        8..=10 => (p % 20) as u32,
        11 => 31 + k,
        r => 14 + (r - 12) as u32 + 2 * k,
    }
}

impl FlashNextRuntime for DivergentRuntime {
    fn vocab_size(&self) -> usize {
        42
    }
    fn has_mtp(&self) -> bool {
        true
    }
    fn begin_slot(&mut self, _: usize) -> Result<(), ModelError> {
        Ok(())
    }
    fn end_slot(&mut self, _: usize) {}
    fn prefill_chunk(&mut self, b: &RowBatch, _path: PrefillPath, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        Ok(want.map(|count| {
            let id = f(b.positions[b.rows - 1] as usize) as i32;
            let mut ids = vec![0i32; count];
            let mut vals = vec![-1.0f32; count];
            ids[0] = id;
            vals[0] = 1.0;
            Candidates { rows: 1, count, ids, vals }
        }))
    }
    fn mtp_after_prefill(
        &mut self,
        _slot: usize,
        _first: usize,
        _chunk: &[u32],
        _next: Option<u32>,
        _path: PrefillPath,
    ) -> Result<(), ModelError> {
        Ok(())
    }
    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        let mut ids = vec![0i32; b.rows * count];
        let mut vals = vec![-1.0f32; b.rows * count];
        for r in 0..b.rows {
            let id = f(b.positions[r] as usize) as i32;
            ids[r * count] = id;
            vals[r * count] = 1.0;
        }
        Ok(Candidates { rows: b.rows, count, ids, vals })
    }
    fn commit(&mut self, _: usize, _keep: usize) {}
    fn flush(&mut self) -> Result<(), ModelError> {
        Ok(())
    }
    fn mtp_propose_batch(&mut self, asks: &[MtpAsk<'_>]) -> Result<Vec<Option<(Vec<u32>, Vec<f32>)>>, ModelError> {
        self.mtp_asks += 1;
        Ok(asks.iter().map(|a| Some((vec![7; a.k], vec![0.5; a.k]))).collect())
    }
}

/// Prompt = positions 0..40 (two full 20-token blocks): the second block's
/// [0..7] suffix has an earlier occurrence (the first block) whose continuation
/// diverges after 3 tokens.
fn divergent_prompt() -> Vec<u32> {
    (0..40).map(f).collect()
}

fn divergent_requests(n: usize, max_new: usize) -> Vec<gb_engine::Request> {
    (0..n)
        .map(|i| gb_engine::Request {
            sampling: Sampling { seed: 10 + i as u64, temperature: 0.0, ..Default::default() },
            prompt: divergent_prompt(),
            max_new,
            stop: vec![],
            draft: true,
            images: vec![],
            cache: false,
        })
        .collect()
}

/// The production factory, but with no native MTP head (`mtp_k = 0`), so the
/// copy drafter is the only proposer and the rejection path is fully isolated.
fn divergent_engine(drafting: bool) -> Engine<FlashNextBackend<DivergentRuntime>> {
    use gb_sched::{Admission, CostModel, ModelMemory};
    let mm = ModelMemory {
        weights_bytes: 1 << 30,
        kv_bytes_per_token: 65536,
        state_bytes_per_slot: 150 << 20,
        activation_reserve: 1 << 30,
    };
    let admission = Admission::new(64 << 30, 6 << 30, mm, 8).unwrap();
    let backend = FlashNextBackend::with_runtime(DivergentRuntime::default(), 64, 0, true);
    let make = make_drafter(drafting);
    Engine::new(
        backend,
        admission,
        EngineConfig {
            max_rows: 8,
            max_drafts: 7,
            cost: CostModel::from_table(vec![(1, 70.0), (8, 90.0), (16, 120.0)]),
            drafting,
        },
        make,
    )
}

/// The runtime's greedy future diverges from the copy draft after 3 tokens: the
/// drafter proposes 7 drafts and exactly 3 are accepted. This test pins the
/// design before the engine-level runs rely on it.
#[test]
fn divergent_runtime_produces_partially_accepted_copy_drafts() {
    let tokens: Vec<u32> = (0..48).map(f).collect();
    let mut d = gb_spec::CopyDrafter::new(8, 7);
    assert_eq!(d.propose(&tokens, 7), vec![8, 9, 10, 32, 16, 17, 18]);
    // the true continuation after position 47 (the suffix's own position) differs at the 4th token
    assert_eq!((48..55).map(f).collect::<Vec<_>>(), vec![8, 9, 10, 33, 18, 19, 20]);
}

/// Concurrent requests whose copy drafts are mostly rejected: greedy output must
/// still equal the no-speculation serial path, and the rejection path must be
/// exercised (accepted < drafted, with some acceptances too).
#[test]
fn batched_copy_drafts_mostly_rejected_but_greedy_identical() {
    use gb_engine::harness::drafted_equals_serial;
    let r = drafted_equals_serial(&divergent_engine, &divergent_requests(3, 64)).unwrap();
    assert_eq!(r.sequences, 3);
    assert!(r.drafted > 0, "no copy drafts proposed");
    assert!(r.accepted > 0, "no copy draft accepted at all");
    assert!(r.accepted < r.drafted, "rejection path never exercised: accepted == drafted ({}/{})", r.accepted, r.drafted);
    eprintln!("3 concurrent, mostly-rejected: drafted {} accepted {} over {} rounds", r.drafted, r.accepted, r.rounds);
}

/// One request finishes while the others continue decoding: the finished one's
/// slot is dropped, the others keep drafting and stay exact.
#[test]
fn a_request_finishing_mid_batch_leaves_the_others_exact() {
    let max_new = [12usize, 40, 64];
    let want_by_len: std::collections::HashMap<usize, Vec<u32>> = max_new
        .iter()
        .map(|&n| {
            (n, {
                let mut e = divergent_engine(false);
                e.submit(divergent_requests(1, n).remove(0)).unwrap();
                while e.step().unwrap() {}
                e.take_outputs().remove(0).tokens
            })
        })
        .collect();
    let mut e = divergent_engine(true);
    for (i, &n) in max_new.iter().enumerate() {
        let mut r = divergent_requests(1, n).remove(0);
        r.sampling.seed = 10 + i as u64;
        e.submit(r).unwrap();
    }
    while e.step().unwrap() {}
    let outs = e.take_outputs();
    assert_eq!(outs.len(), 3);
    let mut lens: Vec<usize> = outs.iter().map(|o| o.tokens.len()).collect();
    lens.sort_unstable();
    assert_eq!(lens, max_new.to_vec(), "the short request must finish early, the others go the full length");
    for o in &outs {
        assert_eq!(want_by_len.get(&o.tokens.len()), Some(&o.tokens), "a finished stream diverged from serial");
        assert_eq!(o.finish, Some(gb_engine::FinishReason::Length));
    }
}

/// A request submitted while another is mid-decode drafts from its own first
/// round and produces the serial tokens.
#[test]
fn a_request_joining_mid_flight_is_sampled_like_serial() {
    let solo = |n: usize| -> Vec<u32> {
        let mut e = divergent_engine(false);
        e.submit(divergent_requests(1, n).remove(0)).unwrap();
        while e.step().unwrap() {}
        e.take_outputs().remove(0).tokens
    };
    let mut e = divergent_engine(true);
    e.submit(divergent_requests(1, 72).remove(0)).unwrap();
    let mut stepped = 0;
    while stepped < 6 && e.step().unwrap() {
        stepped += 1;
    }
    assert!(e.busy(), "the long request should still be decoding mid-flight");
    e.submit(divergent_requests(1, 32).remove(0)).unwrap();
    while e.step().unwrap() {}
    let outs = e.take_outputs();
    assert_eq!(outs.len(), 2);
    let want72 = solo(72);
    let want32 = solo(32);
    assert_eq!(outs.iter().find(|o| o.tokens.len() == 72).map(|o| &o.tokens), Some(&want72));
    assert_eq!(outs.iter().find(|o| o.tokens.len() == 32).map(|o| &o.tokens), Some(&want32));
}

/// Single-stream native MTP still proposes (precedence over the copy drafter):
/// the backend returns a proposal for exactly one request and never for a batch.
#[test]
fn native_mtp_still_wins_for_single_stream() {
    use gb_engine::Backend;
    let mut b = FlashNextBackend::with_runtime(CpuRuntime::default(), 64, 6, true);
    let sampling = Sampling { seed: 1, temperature: 0.0, ..Default::default() };
    let single = [ProposeRequest { slot: 1, tokens: &[3, 3, 3, 3, 3, 3, 3, 3, 3], sampling: &sampling, max: 6 }];
    let p = b.propose_batch_scored(&single);
    let t = p[0].as_ref().expect("single-stream native MTP must propose").tokens.clone();
    assert!(!t.is_empty() && t.len() <= 6);
    assert_eq!(b.m.mtp_asks, 1);
}

/// The exact MTP draft-head prefix: the plan accounts for the real pack's head copy
/// (k=2560, 5-bit EXL3, nv=65536) as the experiment measured it, +104,993,792 B (100.13 MiB).
#[test]
fn draft_head_plan_bytes_returns_the_real_100_13_mib_for_the_pack_dims() {
    use gb_exl3::{Bits, Codebook};
    use std::collections::BTreeMap;
    let mut groups = BTreeMap::new();
    groups.insert(
        "lm_head".into(),
        crate::families::qwen_flash_next::weights::Exl3Group {
            prefix: "lm_head".into(),
            k: 2560,
            n: 248_320,
            bits: Bits::from_twice(10).unwrap(),
            codebook: Codebook::Inst3,
        },
    );
    assert_eq!(draft_head_plan_bytes(&groups, 248_320).unwrap(), 104_993_792);
}

/// Fail-closed: a pack without the LM head group, or whose head is narrower than the
/// candidate domain (the plane-eta h.n vs cfg.vocab mismatch class), is refused before any
/// plan is sized instead of undercounting the plan.
#[test]
fn draft_head_plan_bytes_refuses_a_missing_or_narrow_lm_head() {
    use gb_exl3::{Bits, Codebook};
    use std::collections::BTreeMap;
    assert!(draft_head_plan_bytes(&BTreeMap::new(), 248_320).is_err());
    let mut groups = BTreeMap::new();
    groups.insert(
        "lm_head".into(),
        crate::families::qwen_flash_next::weights::Exl3Group {
            prefix: "lm_head".into(),
            k: 2560,
            n: 65535,
            bits: Bits::from_twice(10).unwrap(),
            codebook: Codebook::Inst3,
        },
    );
    assert!(draft_head_plan_bytes(&groups, 248_320).is_err(), "head narrower than the candidate domain must fail closed");
}

/// The KV plan is sized after the draft-head bytes: `free_after_load` sees the full weight
/// figure, so the +100.13 MiB stays inside the plan and the 262K contract stays truthful.
#[test]
fn the_kv_plan_is_sized_after_the_draft_head_bytes() {
    const G: u64 = 1 << 30;
    let host = MemInfo { total: 121 * G, free: G, available: 90 * G, cached: 89 * G };
    let weights = 40 * G;
    let extra = 104_993_792;
    assert_eq!(free_after_load(G, Some(&host), weights + extra), free_after_load(G, Some(&host), weights) - extra);
}

/// The Flash-Next boot prints its own KV plan line: the family has its own
/// `build_engine` and never emits the generic serve.rs `kv_plan` marker, so the
/// gate recipe pins `flash_next_kv_plan` with the contract's kv_tokens/slots.
/// The 262K, 2-slot plan must report kv_tokens = 262144 and slots = 2, and the
/// printed line must carry those values in the recipe's field order.
#[test]
fn the_flash_next_kv_plan_line_pins_the_serving_contract() {
    use crate::family::ModelFamily;
    use crate::plan::{plan_memory_cached, CacheRequest};
    const MAX_MODEL_LEN: usize = 262_144;
    const SLOTS: usize = 2;
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/qwen_flash_next");
    let plan = super::super::FlashNextFamily.probe(&dir).unwrap();
    assert!(plan.missing_components().is_empty());
    let cfg = FlashNextConfig::from_dir(&dir).unwrap();
    let profile = plan.memory_profile(&MemoryOptions { mtp: true, vision: None, ..MemoryOptions::serving_defaults() }).unwrap();
    let snapshot = super::super::prefix::snapshot_bytes(&cfg, true);
    let p = plan_memory_cached(
        &profile,
        40 << 30,
        2 << 30,
        SLOTS,
        MAX_MODEL_LEN as u64,
        0,
        CacheRequest::Fixed(3 << 30),
        snapshot,
        0,
    )
    .unwrap();
    let line = format!(
        "rexl3 event=flash_next_kv_plan kv_tokens={} slots={} kv_pages={} free_after_gib={:.2}",
        p.kv_tokens, SLOTS, p.kv_blocks, 40.0
    );
    assert_eq!(p.kv_tokens, MAX_MODEL_LEN as u64, "a big-enough pool serves the whole window");
    assert!(p.kv_blocks > 0);
    // The gate recipe's expect_log pattern must match this line verbatim.
    let recipe = "rexl3 event=flash_next_kv_plan kv_tokens=262144 slots=2 kv_pages=";
    assert!(line.starts_with(recipe), "recipe pattern must match the boot line: {line}");
}

/// Prefix-cache snapshot layout with the MTP head: the MTP one-ahead row (K, V) and its
/// QSA pool entry live at the very tail of the slot, exactly `mtp_at()` = bytes() −
/// (2*mtp_row + mtp_pool). `capture` writes them there and `restore` reads them back from
/// the same offsets, so the copied region round-trips identically; the model's draft-head
/// copy is weights, not slot state, and does not change any snapshot offset or size.
#[test]
fn mtp_snapshot_layout_offsets_cover_exactly_the_mtp_row_and_pool() {
    use super::prefix::Layout;
    use crate::families::qwen_flash_next::config::FlashNextConfig;
    let cfg = FlashNextConfig {
        hidden: 2560,
        layers: 48,
        layer_types: vec!["full_attn".into(); 48],
        vocab: 248_320,
        rms_norm_eps: 1e-6,
        heads: 32,
        kv_heads: 2,
        head_dim: 128,
        rotary_dim: 128,
        rope_theta: 1_000_000.0,
        max_position_embeddings: 262_144,
        lin_key_heads: 48,
        lin_value_heads: 48,
        lin_key_dim: 128,
        lin_value_dim: 128,
        lin_conv_kernel: 9,
        experts: 512,
        experts_per_tok: 10,
        moe_intermediate: 640,
        shared_intermediate: 640,
        hc_streams: 4,
        hc_lowrank: 64,
        indexer_heads: 8,
        indexer_head_dim: 128,
        indexer_kv_heads: 1,
        indexer_budget: 64,
        indexer_compress_ratio: 8,
        ple_layer_ids: vec![0],
        ngram_size: 16,
        heads_per_ngram: 2,
        ple_embed_dim: 64,
        ple_conv_kernel: 3,
        eos_token_id: 151645,
        mtp_layers: 1,
        quant_method: "exl3".into(),
    };
    let no_mtp = Layout::from_config(&cfg, false);
    let mtp = Layout::from_config(&cfg, true);
    assert!(mtp.mtp_row > 0 && mtp.mtp_pool > 0);
    assert_eq!(mtp.mtp_row, cfg.kv_heads * cfg.head_dim * 2);
    assert_eq!(mtp.mtp_pool, cfg.indexer_head_dim * 2);
    // only the MTP tail changes when drafting is on
    assert_eq!(mtp.gdn_layers, no_mtp.gdn_layers);
    assert_eq!(mtp.s, no_mtp.s);
    assert_eq!(mtp.qsa_layers, no_mtp.qsa_layers + 1);
    let mtp_bytes = mtp.bytes();
    assert_eq!(
        mtp_bytes,
        no_mtp.bytes() + mtp.qsa_ring + 2 * mtp.mtp_row + mtp.mtp_pool,
        "MTP adds one QSA ring layer plus the one-ahead row/pool tail"
    );
    // the MTP tail is exactly the last `2*mtp_row + mtp_pool` bytes of the slot
    assert_eq!(mtp.mtp_at(), mtp.bytes() - (2 * mtp.mtp_row + mtp.mtp_pool));
    assert_eq!(no_mtp.bytes() + mtp.qsa_ring, mtp.mtp_at(), "the MTP tail sits right after the extra QSA ring layer");
}

#[test]
fn flash_next_sizes_its_prefix_cache_from_the_window() {
    use crate::family::ModelFamily;
    use crate::plan::{plan_memory_cached, plan_memory_cached_windowed, CacheRequest};
    use gb_sched::prefix::CacheBudget;
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/qwen_flash_next");
    let plan = super::super::FlashNextFamily.probe(&dir).unwrap();
    let cfg = FlashNextConfig::from_dir(&dir).unwrap();
    let profile = plan.memory_profile(&MemoryOptions { mtp: true, vision: None, ..MemoryOptions::serving_defaults() }).unwrap();
    let snapshot = super::super::prefix::snapshot_bytes(&cfg, true);
    let page = profile.kv_page_bytes();
    // the exact bytes behind the figures below (the testdata config: 12 full-attention layers, window 262,144, the MTP carry on)
    assert_eq!((page, super::super::prefix::page_bytes(&cfg, true), snapshot), (7_028_736, 7_028_736, 118_507_776));
    assert_eq!(cfg.max_position_embeddings, 262_144);
    // before: `Auto` at its 8 GiB cap is 49 slots and 392 pages, a reach of 100,352 tokens in a 262,144-token window
    let old = CacheBudget::fit(8 << 30, snapshot, page);
    assert_eq!((old.snapshot_slots, old.page_blocks, old.reach_tokens()), (49, 392, 100_352));
    // after, at two live slots: the target is 8 slots and 1,280 pages (a full window and two eighth-window entries), 327,680 tokens, 9.26 GiB
    let w = 262_144u64;
    let free = 60u64 << 30;
    let auto = CacheRequest::Auto { cap_bytes: crate::plan::AUTO_CACHE_CAP_BYTES };
    let p = plan_memory_cached_windowed(&profile, free, 2 << 30, 2, w, 0, auto, snapshot, 0, w).unwrap();
    assert_eq!((p.cache.snapshot_slots, p.cache.page_blocks, p.cache.reach_tokens()), (8, 1280, 327_680));
    assert_eq!(p.cache.bytes(), 8 * snapshot + 1280 * page);
    // the KV pool and admission are exactly the plan without a cache
    let off = plan_memory_cached(&profile, free, 2 << 30, 2, w, 0, CacheRequest::Off, snapshot, 0).unwrap();
    assert_eq!(
        (p.kv_blocks, p.kv_tokens, p.admit_bytes, p.spare_bytes),
        (off.kv_blocks, off.kv_tokens, off.admit_bytes, off.spare_bytes)
    );
    // a pin is cut by the same rule: 3 GiB holds the target for a window of 8,192 tokens and spends the rest
    let q = plan_memory_cached_windowed(&profile, 40 << 30, 2 << 30, 2, 8192, 0, CacheRequest::Fixed(3 << 30), snapshot, 0, 8192)
        .unwrap();
    assert!(q.cache.enabled() && q.cache.bytes() <= 3 << 30 && q.cache.reach_tokens() >= 8192, "{:?}", q.cache);
}
