//! GLM-5.3-Flash as a TWO-RANK tensor-parallel group ON ONE GPU (lane L2b, `docs/GLM_TP.md` section 6): two models of the same layer
//! prefix in one process, each with its own stream and thread, made ranks 0 and 1 of a group (`GlmModel::enable_tp`: whole-expert
//! ownership 0..144 | 144..288 over a full bank, the pair buffer summed by the in-process loopback collective, the combine on every
//! rank), run the same call script as one model of the same prefix with no group. Every candidate and every fp16 logit of every call must
//! be byte-identical across the two ranks AND equal to the single node's, with graphs on and with graphs off.
//!
//! The script: a 2,300-token prompt in 512-row chunks (the batched expert path, past the DSA sparse threshold of 2,048), a 200-token
//! prompt in 128-row chunks (the decode-window kernels), a 37-token prompt, then multi-sequence windows with partial commits and greedy
//! single-row windows. Env: `REXL3_GLM_DIR` (the pack), `REXL3_GLM_TP_LAYERS` (default 8: layers 0..8 hold dense, 3-bit and 2-bit MoE
//! layers, KDA and MLA/DSA mixers). Run under the GPU lock, one at a time:
//!
//!     flock -x /tmp/rexl3-gpu.lock env REXL3_GLM_DIR=... cargo test --release -p gb-model --test glm_tp -- --ignored --nocapture --test-threads=1

mod glm_support;

use gb_cuda::moe::ExpertOwnership;
use gb_model::families::glm::contract::PrefillContract;
use gb_model::families::glm::load::{boot, boot_mapped, LoadOptions};
use gb_model::families::glm::model::{GlmModel, ModelOptions};
use gb_model::families::nemotron_h_omni::tp::{SingleRank, TpComm};
use gb_model::families::nemotron_h_omni::tp_comm::LoopbackGroup;
use gb_model::rows::RowBatch;
use glm_support::*;
use std::sync::Arc;
use std::time::Duration;

const CHUNK: usize = 512;
const SLOTS: usize = 3;
const COUNT: usize = 8;

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h = (*h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
}

/// The digest of one call's answer: the candidates (ids and value bits) and every fp16 logit of its `rows` head rows.
fn digest_call(m: &GlmModel, c: &gb_model::forward::Candidates, rows: usize) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &i in &c.ids {
        fnv(&mut h, &i.to_le_bytes());
    }
    for &v in &c.vals {
        fnv(&mut h, &v.to_bits().to_le_bytes());
    }
    for x in m.logits_host(0, rows).unwrap() {
        fnv(&mut h, &x.to_le_bytes());
    }
    h
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % 150_000) as u32
    }
    fn tokens(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.next()).collect()
    }
}

/// The script, on one model: one digest per answering call, in call order.
fn script(m: &mut GlmModel) -> Vec<u64> {
    script_with(m, &mut Vec::new())
}

/// What one call of the script launched ON THIS THREAD (the P2 launch audit of a rank: `glm_p2::launches_here`,
/// `glm_p2_moe::runs_here`, the rank's part swaps).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CallAudit {
    /// `true` for a prompt chunk, `false` for a window.
    prompt: bool,
    rows: usize,
    launches: u64,
    runs: u64,
    swaps: u64,
    kda: u64,
    /// The heads of the chunked KDA calls (`kda::chunked_heads_here`).
    heads: u64,
    /// The P2 MLA absorb / expand launches (`glm_p2_mla::runs_here`, step (d)).
    mla: u64,
}

type Counters = (u64, u64, u64, u64, u64, u64);

fn counters(m: &GlmModel) -> Counters {
    (
        gb_cuda::glm_p2::launches_here(),
        gb_cuda::glm_p2_moe::runs_here(),
        m.tp().map_or(0, |t| t.p2_swaps()),
        gb_cuda::kda::chunked_runs_here(),
        gb_cuda::kda::chunked_heads_here(),
        gb_cuda::glm_p2_mla::runs_here(),
    )
}

fn audit_call(m: &GlmModel, at: Counters, prompt: bool, rows: usize) -> CallAudit {
    let now = counters(m);
    CallAudit {
        prompt,
        rows,
        launches: now.0 - at.0,
        runs: now.1 - at.1,
        swaps: now.2 - at.2,
        kda: now.3 - at.3,
        heads: now.4 - at.4,
        mla: now.5 - at.5,
    }
}

/// [`script`], recording each call's [`CallAudit`] in `calls`.
fn script_with(m: &mut GlmModel, calls: &mut Vec<CallAudit>) -> Vec<u64> {
    script_cut(m, calls, &[&[], &[], &[]])
}

/// [`script_with`] with snapshot-style slice ends: a prompt slice of slot `s` ends at the first of `cuts[s]` past its start when that comes
/// before the chunk's natural end (`prefix::slice_end`), so the prompts are prefilled on another grid while every call that answers stays the
/// same call.
fn script_cut(m: &mut GlmModel, calls: &mut Vec<CallAudit>, cuts: &[&[usize]; SLOTS]) -> Vec<u64> {
    let mut out = Vec::new();
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let prompts = [(0usize, r.tokens(2300), CHUNK), (1, r.tokens(200), 128), (2, r.tokens(37), 128)];
    let mut pos = [0usize; SLOTS];
    let mut last = [0u32; SLOTS];
    for (slot, p, chunk) in &prompts {
        m.begin_slot(*slot).unwrap();
        let mut at = 0;
        while at < p.len() {
            let end = gb_model::families::glm::prefix::slice_end(at, p.len(), *chunk, cuts[*slot]);
            let want = (end == p.len()).then_some(COUNT);
            let c0 = counters(m);
            let got = m.prefill_chunk(*slot, at, &p[at..end], want).unwrap();
            calls.push(audit_call(m, c0, true, end - at));
            if let Some(c) = got {
                out.push(digest_call(m, &c, 1));
                last[*slot] = c.ids[0] as u32;
            }
            at = end;
        }
        pos[*slot] = p.len();
    }
    // a verify-like window: 3 rows of slot 0, 1 of slot 1, 5 of slot 2; partial commits
    let reqs = vec![
        gb_engine::RowRequest { slot: 0, tokens: vec![last[0], 11, 12], first_position: pos[0] as u64 },
        gb_engine::RowRequest { slot: 1, tokens: vec![last[1]], first_position: pos[1] as u64 },
        gb_engine::RowRequest { slot: 2, tokens: vec![last[2], 21, 22, 23, 24], first_position: pos[2] as u64 },
    ];
    let b = RowBatch::from_requests(&reqs);
    let c0 = counters(m);
    let c = m.window_candidates(&b, COUNT).unwrap();
    calls.push(audit_call(m, c0, false, b.rows));
    out.push(digest_call(m, &c, b.rows));
    for (slot, keep) in [(0usize, 2usize), (1, 1), (2, 4)] {
        m.commit(slot, keep);
        pos[slot] += keep;
        // the next token is the candidate of the last kept row
        let row = b.seq_rows(slot).start + keep - 1;
        last[slot] = c.ids[row * COUNT] as u32;
    }
    // greedy single-row windows of all three sequences
    for _ in 0..6 {
        let reqs: Vec<gb_engine::RowRequest> =
            (0..SLOTS).map(|s| gb_engine::RowRequest { slot: s, tokens: vec![last[s]], first_position: pos[s] as u64 }).collect();
        let b = RowBatch::from_requests(&reqs);
        let c0 = counters(m);
        let c = m.window_candidates(&b, COUNT).unwrap();
        calls.push(audit_call(m, c0, false, b.rows));
        out.push(digest_call(m, &c, b.rows));
        for s in 0..SLOTS {
            m.commit(s, 1);
            pos[s] += 1;
            last[s] = c.ids[s * COUNT] as u32;
        }
    }
    m.flush().unwrap();
    for s in 0..SLOTS {
        m.end_slot(s);
    }
    out
}

fn load(dir: &std::path::Path, layers: usize) -> GlmModel {
    let b = boot(dir, Some(layers), &LoadOptions::from_env(8 * GIB)).unwrap();
    GlmModel::new_chunked(
        b.pack.config.clone(),
        b.device,
        &ModelOptions { slots: SLOTS, kv_blocks: 64, max_context: 8192, layers },
        CHUNK,
    )
    .unwrap()
}

/// Rank `rank` of two under the KDA head split as the rank planner loads it (lane E, the rank strips): the pack mapped with the rank's
/// heads (`GlmConfig::with_kda_rank`), so only its qkv strips are placed and read, and the model holds its heads' state from the start.
fn load_rank_strips(dir: &std::path::Path, layers: usize, rank: usize) -> GlmModel {
    use gb_model::families::glm::{config::GlmConfig, GlmPack};
    gb_cuda::rt::init_context().unwrap();
    let cfg = GlmConfig::from_dir(dir).unwrap().with_kda_rank(rank, 2).unwrap();
    let ck = gb_exl3::checkpoint::scan(dir, false).unwrap();
    let pack = GlmPack::from_checkpoint(&ck, cfg).unwrap();
    let b = boot_mapped(dir, pack, Some(layers), &LoadOptions::from_env(8 * GIB), 0.0).unwrap();
    eprintln!("rank {rank} with its qkv strips: {:.3} GiB of regions", gib(b.device.regions.bytes()));
    GlmModel::new_chunked(
        b.pack.config.clone(),
        b.device,
        &ModelOptions { slots: SLOTS, kv_blocks: 64, max_context: 8192, layers },
        CHUNK,
    )
    .unwrap()
}

#[test]
#[ignore]
fn two_ranks_on_one_gpu_equal_the_single_node_model_byte_for_byte() {
    let Some(dir) = pack_dir() else { return };
    let layers: usize = std::env::var("REXL3_GLM_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(60 * GIB);
    let guard = Guard::start(20 * GIB);
    let t0 = std::time::Instant::now();
    let mut a = load(&dir, layers);
    let mut b = load(&dir, layers);
    a.set_prefill_contract(PrefillContract::P1).unwrap();
    b.set_prefill_contract(PrefillContract::P1).unwrap();
    eprintln!("two models of {layers} layers loaded in {:.1?}; MemAvailable {:.1} GiB", t0.elapsed(), gib(mem_available()));

    // the single node, graphs on (the serving default) and off
    a.set_graphs(true);
    let ref_graphs = script(&mut a);
    a.set_graphs(false);
    let ref_eager = script(&mut a);
    assert_eq!(ref_graphs, ref_eager, "the single node: graphs and eager differ on this prefix");
    eprintln!("single node: {} answering calls, graphs == eager", ref_graphs.len());

    // the group: ranks 0 and 1, each on its own thread with its own loopback comm
    let comms = LoopbackGroup::with_timeout(2, Duration::from_secs(120));
    let mut comms = comms.into_iter();
    let (c0, c1) = (comms.next().unwrap(), comms.next().unwrap());
    let e = 288;
    a.enable_tp(Arc::new(c0), ExpertOwnership::range(e, 0, 2).unwrap()).unwrap();
    b.enable_tp(Arc::new(c1), ExpertOwnership::range(e, 1, 2).unwrap()).unwrap();
    assert!(!a.graphs_on() && !b.graphs_on(), "a rank runs eagerly");
    let t = std::time::Instant::now();
    let (d0, d1, a, b) = std::thread::scope(|sc| {
        let h0 = sc.spawn(move || {
            let d = script(&mut a);
            (d, a)
        });
        let h1 = sc.spawn(move || {
            let d = script(&mut b);
            (d, b)
        });
        let (d0, a) = h0.join().expect("rank 0");
        let (d1, b) = h1.join().expect("rank 1");
        (d0, d1, a, b)
    });
    eprintln!("two ranks ran the script in {:.1?}", t.elapsed());
    assert_eq!(d0, d1, "the two ranks' answers differ");
    assert_eq!(d0, ref_graphs, "the group's answers differ from the single node's");
    let (t0r, t1r) = (a.tp().unwrap(), b.tp().unwrap());
    assert_eq!((t0r.exchanges, t0r.exchanged_bytes), (t1r.exchanges, t1r.exchanged_bytes), "the ranks exchanged alike");
    eprintln!(
        "BYTE-IDENTICAL: {} answering calls (candidates + every fp16 logit), {} exchanges of {:.1} MiB in all per rank",
        d0.len(),
        t0r.exchanges,
        t0r.exchanged_bytes as f64 / 1048576.0
    );
    drop((a, b));

    // the negative control: both ranks own 0..144 (experts 144..288 nobody computes, 0..144 counted twice): the answers must differ, so
    // the comparison above can see an ownership that does not partition the experts
    let mut a = load(&dir, layers);
    let mut b = load(&dir, layers);
    a.set_prefill_contract(PrefillContract::P1).unwrap();
    b.set_prefill_contract(PrefillContract::P1).unwrap();
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(120)).into_iter();
    a.enable_tp(Arc::new(comms.next().unwrap()), ExpertOwnership::range(e, 0, 2).unwrap()).unwrap();
    b.enable_tp(Arc::new(comms.next().unwrap()), ExpertOwnership::range(e, 0, 2).unwrap()).unwrap();
    let (n0, n1) = std::thread::scope(|sc| {
        let h0 = sc.spawn(|| script(&mut a));
        let h1 = sc.spawn(|| script(&mut b));
        (h0.join().expect("rank 0"), h1.join().expect("rank 1"))
    });
    assert_eq!(n0, n1, "the overlapping ranks still agree with each other (the sum is replicated)");
    let differ = n0.iter().zip(&ref_graphs).filter(|(x, y)| x != y).count();
    assert!(differ > 0, "an ownership that does not partition the experts went unnoticed");
    eprintln!("negative control (both ranks own 0..144): {differ} of {} answering calls differ from the single node", n0.len());
    drop((a, b));
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// The head split (lane E Phase 2, `REXL3_GLM_TP_SPLIT`): the two ranks of the loopback group also split the KDA heads (32 a rank, the
/// rank's qkv strips, 32-head pools, the o_proj by K-split partials in windows and over the gathered heads in large chunks) and / or the
/// lm_head's vocabulary (the fp16 logits swapped so both ranks hold them). Every candidate and fp16 logit of every call of the script must
/// equal the single node's and the other rank's, for `kda`, `lm`, `mla`, `kda,lm,mla`, `sh` (the shared experts' halves in the decode
/// windows) and `kda,lm,mla,sh`. Env as above; `REXL3_GLM_TP_SPLITS=kda,lm` picks the sets
/// (default all three, `+`-separated within one: `kda+lm`). A set with `kda` runs twice: the ranks loaded whole (the strips copied at the
/// split) and as the rank planner loads them, with only their qkv strips (`load_rank_strips`); `REXL3_GLM_TP_RANK_STRIPS=0|1` keeps one.
#[test]
#[ignore]
fn two_ranks_with_the_head_split_equal_the_single_node_model_byte_for_byte() {
    use gb_model::families::glm::model::tp::{split_from_vars, split_names};
    let Some(dir) = pack_dir() else { return };
    let layers: usize = std::env::var("REXL3_GLM_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(60 * GIB);
    let guard = Guard::start(20 * GIB);
    let mut a = load(&dir, layers);
    a.set_prefill_contract(PrefillContract::P1).unwrap();
    a.set_graphs(false);
    let reference = script(&mut a);
    drop(a);
    eprintln!("single node: {} answering calls", reference.len());
    let sets: Vec<String> = std::env::var("REXL3_GLM_TP_SPLITS")
        .ok()
        .map(|v| v.split(',').map(|x| x.trim().replace('+', ",")).collect())
        .unwrap_or_else(|| {
            vec!["none".into(), "kda".into(), "lm".into(), "mla".into(), "kda,lm,mla".into(), "sh".into(), "kda,lm,mla,sh".into()]
        });
    let strips_modes: Vec<bool> = match std::env::var("REXL3_GLM_TP_RANK_STRIPS").ok().as_deref() {
        Some("0") => vec![false],
        Some("1") => vec![true],
        _ => vec![false, true],
    };
    let runs: Vec<(String, bool)> = sets
        .iter()
        .flat_map(|set| {
            let kda = set.split(',').any(|x| x.trim() == "kda");
            let modes = if kda { strips_modes.clone() } else { vec![false] };
            modes.into_iter().map(move |m| (set.clone(), m))
        })
        .collect();
    // the exchanges of each split set without `sh` (the strips copied): the base an `sh` set must equal but for the partials' bytes
    let mut base: std::collections::BTreeMap<u8, (u64, u64)> = Default::default();
    for (set, rank_strips) in &runs {
        let (set, rank_strips) = (set, *rank_strips);
        let bits = split_from_vars(|_| Some(set.clone())).unwrap();
        let (mut a, mut b) = if rank_strips {
            (load_rank_strips(&dir, layers, 0), load_rank_strips(&dir, layers, 1))
        } else {
            (load(&dir, layers), load(&dir, layers))
        };
        a.set_prefill_contract(PrefillContract::P1).unwrap();
        b.set_prefill_contract(PrefillContract::P1).unwrap();
        let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(120)).into_iter();
        let e = 288;
        a.enable_tp(Arc::new(comms.next().unwrap()), ExpertOwnership::range(e, 0, 2).unwrap()).unwrap();
        b.enable_tp(Arc::new(comms.next().unwrap()), ExpertOwnership::range(e, 1, 2).unwrap()).unwrap();
        a.enable_head_split(bits).unwrap();
        b.enable_head_split(bits).unwrap();
        let t = std::time::Instant::now();
        let (d0, d1, a, b) = std::thread::scope(|sc| {
            let h0 = sc.spawn(move || {
                let d = script(&mut a);
                (d, a)
            });
            let h1 = sc.spawn(move || {
                let d = script(&mut b);
                (d, b)
            });
            let (d0, a) = h0.join().expect("rank 0");
            let (d1, b) = h1.join().expect("rank 1");
            (d0, d1, a, b)
        });
        assert_eq!(d0, d1, "split {}: the two ranks' answers differ", split_names(bits));
        let differ = d0.iter().zip(&reference).filter(|(x, y)| x != y).count();
        assert_eq!(
            differ,
            0,
            "split {}: {differ} of {} answering calls differ from the single node",
            split_names(bits),
            d0.len()
        );
        let (t0r, t1r) = (a.tp().unwrap(), b.tp().unwrap());
        assert_eq!((t0r.exchanges, t0r.exchanged_bytes), (t1r.exchanges, t1r.exchanged_bytes), "the ranks exchanged alike");
        let sh = gb_model::families::glm::model::tp::SPLIT_SHARED;
        if bits & sh == 0 {
            base.entry(bits).or_insert((t0r.exchanges, t0r.exchanged_bytes));
            assert_eq!(t0r.sh_exchanged_bytes, 0);
        } else {
            // lane E, E2.6: the shared expert's partials ride in the routed experts' exchanges: not one exchange more, and not one byte more
            // than the partials of the decode windows (a prefill chunk's exchange keeps its bytes)
            assert!(t0r.sh_exchanged_bytes > 0, "split {}: no partials exchanged", split_names(bits));
            match base.get(&(bits & !sh)) {
                Some(&(n, bytes)) => {
                    assert_eq!(t0r.exchanges, n, "split {}: the shared split adds no exchange", split_names(bits));
                    assert_eq!(
                        t0r.exchanged_bytes,
                        bytes + t0r.sh_exchanged_bytes,
                        "split {}: the bytes beyond the base are the windows' partials only",
                        split_names(bits)
                    );
                    eprintln!(
                        "split {}: {} exchanges as {} has, {:.2} MiB more a rank: the windows' shared partials, nothing at prefill",
                        split_names(bits),
                        n,
                        split_names(bits & !sh),
                        t0r.sh_exchanged_bytes as f64 / 1048576.0
                    );
                }
                None => eprintln!(
                    "split {}: its base set {} did not run: bytes not compared",
                    split_names(bits),
                    split_names(bits & !sh)
                ),
            }
        }
        eprintln!(
            "split {}{}: BYTE-IDENTICAL to the single node, {} answering calls (candidates + every fp16 logit), {} exchanges of {:.1} MiB a rank, {:.1?}; MemAvailable {:.1} GiB",
            split_names(bits),
            if rank_strips { " (rank strips loaded)" } else { "" },
            d0.len(),
            t0r.exchanges,
            t0r.exchanged_bytes as f64 / 1048576.0,
            t.elapsed(),
            gib(mem_available())
        );
        drop((a, b));
    }
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// Lane P's P5b: P2 on lane E's head splits. Two ranks under P2 with `kda`, `mla`, `kda,lm,mla`, `sh` and `kda,lm,mla,sh` split (the KDA
/// qkv strips copied, and loaded as the rank planner loads them) answer the script exactly as ONE P2 model of the same prefix does: every
/// candidate and fp16 logit. With `sh` (lane E's E2.6) the decode windows run the shared experts' halves; a P2 prompt chunk keeps the
/// shared expert replicated (deferred to the P2 part swap, the full weights). A P2 prompt chunk on a KDA-split rank runs its strips
/// through the P2 GEMM, the chunked recurrence on its 32 heads and the full P2 o_proj over the gathered heads; it never takes the MLA
/// split's decode-path arm. The launch audit per rank: every prompt chunk all-P2
/// (the dense linears, the expert runs, a part swap a MoE layer) with the chunked recurrence over HALF the single node's heads under a KDA
/// split; windows none. `REXL3_GLM_TP_SPLITS` / `REXL3_GLM_TP_RANK_STRIPS` choose the runs as for the P1 test.
#[test]
#[ignore]
fn p2_two_ranks_with_the_head_split_equal_the_single_node_p2_model_byte_for_byte() {
    use gb_model::families::glm::model::tp::{split_from_vars, split_names, SPLIT_KDA};
    let Some(dir) = pack_dir() else { return };
    let layers: usize = std::env::var("REXL3_GLM_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(60 * GIB);
    let guard = Guard::start(20 * GIB);
    let mut a = load(&dir, layers);
    a.set_prefill_contract(PrefillContract::P2).unwrap();
    a.set_graphs(false);
    let mut ref_audit = Vec::new();
    let reference = script_with(&mut a, &mut ref_audit);
    let heads_single = ref_audit.iter().find(|c| c.prompt).map(|c| c.heads).unwrap();
    drop(a);
    eprintln!("P2 single node: {} answering calls, {} chunked-KDA heads a prompt chunk", reference.len(), heads_single);
    let sets: Vec<String> = std::env::var("REXL3_GLM_TP_SPLITS")
        .ok()
        .map(|v| v.split(',').map(|x| x.trim().replace('+', ",")).collect())
        .unwrap_or_else(|| vec!["kda".into(), "mla".into(), "kda,lm,mla".into(), "sh".into(), "kda,lm,mla,sh".into()]);
    let strips_modes: Vec<bool> = match std::env::var("REXL3_GLM_TP_RANK_STRIPS").ok().as_deref() {
        Some("0") => vec![false],
        Some("1") => vec![true],
        _ => vec![false, true],
    };
    let runs: Vec<(String, bool)> = sets
        .iter()
        .flat_map(|set| {
            let kda = set.split(',').any(|x| x.trim() == "kda");
            let modes = if kda { strips_modes.clone() } else { vec![false] };
            modes.into_iter().map(move |m| (set.clone(), m))
        })
        .collect();
    for (set, rank_strips) in &runs {
        let (set, rank_strips) = (set, *rank_strips);
        let bits = split_from_vars(|_| Some(set.clone())).unwrap();
        let (mut a, mut b) = if rank_strips {
            (load_rank_strips(&dir, layers, 0), load_rank_strips(&dir, layers, 1))
        } else {
            (load(&dir, layers), load(&dir, layers))
        };
        a.set_prefill_contract(PrefillContract::P2).unwrap();
        b.set_prefill_contract(PrefillContract::P2).unwrap();
        let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(120)).into_iter();
        let e = 288;
        a.enable_tp(Arc::new(comms.next().unwrap()), ExpertOwnership::range(e, 0, 2).unwrap()).unwrap();
        b.enable_tp(Arc::new(comms.next().unwrap()), ExpertOwnership::range(e, 1, 2).unwrap()).unwrap();
        a.enable_head_split(bits).unwrap();
        b.enable_head_split(bits).unwrap();
        let t = std::time::Instant::now();
        let (d0, d1, a, b, au0, au1) = std::thread::scope(|sc| {
            let h0 = sc.spawn(move || {
                let mut au = Vec::new();
                let d = script_with(&mut a, &mut au);
                (d, a, au)
            });
            let h1 = sc.spawn(move || {
                let mut au = Vec::new();
                let d = script_with(&mut b, &mut au);
                (d, b, au)
            });
            let (d0, a, au0) = h0.join().expect("rank 0");
            let (d1, b, au1) = h1.join().expect("rank 1");
            (d0, d1, a, b, au0, au1)
        });
        let name = format!("{}{}", split_names(bits), if rank_strips { " (rank strips loaded)" } else { "" });
        assert_eq!(d0, d1, "P2 split {name}: the two ranks' answers differ");
        let differ = d0.iter().zip(&reference).filter(|(x, y)| x != y).count();
        assert_eq!(differ, 0, "P2 split {name}: {differ} of {} answering calls differ from the P2 single node", d0.len());
        let want_heads = if bits & SPLIT_KDA != 0 { heads_single / 2 } else { heads_single };
        for (r, (au, m)) in [(&au0, &a), (&au1, &b)].into_iter().enumerate() {
            for (i, c) in au.iter().enumerate() {
                if c.prompt {
                    let want = (
                        p2_expected_launches(m, c.rows),
                        p2_expected_moe_runs(m),
                        p2_expected_moe_runs(m),
                        p2_expected_kda_runs(m),
                        want_heads,
                        p2_expected_mla_runs(m, c.rows),
                    );
                    assert_eq!(
                        (c.launches, c.runs, c.swaps, c.kda, c.heads, c.mla),
                        want,
                        "P2 split {name}, rank {r} call {i} ({} rows): launches, runs, swaps, chunked KDA, heads, MLA absorb / expand",
                        c.rows
                    );
                } else {
                    assert_eq!(
                        (c.launches, c.runs, c.swaps, c.kda, c.mla),
                        (0, 0, 0, 0, 0),
                        "P2 split {name}, rank {r} window {i}"
                    );
                }
            }
        }
        assert_eq!(au0, au1, "P2 split {name}: the ranks' audits differ");
        let (t0r, t1r) = (a.tp().unwrap(), b.tp().unwrap());
        assert_eq!((t0r.exchanges, t0r.exchanged_bytes), (t1r.exchanges, t1r.exchanged_bytes), "the ranks exchanged alike");
        eprintln!(
            "P2 split {name}: BYTE-IDENTICAL to the P2 single node, {} answering calls; audit: every prompt chunk all-P2, {want_heads} chunked-KDA heads a chunk a rank (single node {heads_single}); {} exchanges of {:.1} MiB a rank, {:.1?}; MemAvailable {:.1} GiB",
            d0.len(),
            t0r.exchanges,
            t0r.exchanged_bytes as f64 / 1048576.0,
            t.elapsed(),
            gib(mem_available())
        );
        drop((a, b));
    }
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// The two-node prefix cache's premise (`docs/GLM_TP.md` 4.2): a group that ends prompt slices at snapshot cuts answers exactly as one that
/// does not. With the cache on, a cold prompt's slices end at the cuts `prefix::cuts_for` keeps under P2 (on the 16-row KDA grid from the
/// prompt's first row), while a request with the cache off runs the natural chunks, and the gate compares the two. Under P2 and the serving
/// split `kda,lm,mla,sh` (`REXL3_GLM_TP_SPLITS` picks others), the sliced script must give every candidate and fp16 logit the unsliced P2 single
/// node gives, on both ranks; the single node sliced at the same cuts must too. A cut off the grid is reported, not asserted.
#[test]
#[ignore]
fn p2_two_ranks_sliced_at_snapshot_cuts_equal_the_unsliced_prefill_byte_for_byte() {
    use gb_model::families::glm::model::tp::{split_from_vars, split_names};
    let Some(dir) = pack_dir() else { return };
    let layers: usize = std::env::var("REXL3_GLM_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(60 * GIB);
    let guard = Guard::start(20 * GIB);
    // cuts on the grid: before the DSA sparse threshold, mid-chunk, and the end cut P - 16 moved down to the grid; one 16-row slice
    let on_grid: [&[usize]; SLOTS] = [&[256, 1040, 2272], &[96, 176], &[16]];
    let off_grid: [&[usize]; SLOTS] = [&[250, 1037], &[90], &[]];
    let mut a = load(&dir, layers);
    a.set_prefill_contract(PrefillContract::P2).unwrap();
    a.set_graphs(false);
    let reference = script(&mut a);
    let sliced = script_cut(&mut a, &mut Vec::new(), &on_grid);
    let differ = sliced.iter().zip(&reference).filter(|(x, y)| x != y).count();
    assert_eq!(differ, 0, "P2 single node: slicing at grid cuts changed {differ} of {} answering calls", reference.len());
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| script_cut(&mut a, &mut Vec::new(), &off_grid))) {
        Ok(d) => {
            let n = d.iter().zip(&reference).filter(|(x, y)| x != y).count();
            eprintln!("P2 single node, cuts OFF the grid: {n} of {} answering calls differ (reported only)", d.len());
        }
        Err(_) => eprintln!("P2 single node, cuts OFF the grid: refused (reported only)"),
    }
    drop(a);
    eprintln!("P2 single node: {} answering calls, sliced at grid cuts == unsliced", reference.len());
    let sets: Vec<String> = std::env::var("REXL3_GLM_TP_SPLITS")
        .ok()
        .map(|v| v.split(',').map(|x| x.trim().replace('+', ",")).collect())
        .unwrap_or_else(|| vec!["kda,lm,mla,sh".into()]);
    for set in &sets {
        let bits = split_from_vars(|_| Some(set.clone())).unwrap();
        let (mut a, mut b) = (load_rank_strips(&dir, layers, 0), load_rank_strips(&dir, layers, 1));
        a.set_prefill_contract(PrefillContract::P2).unwrap();
        b.set_prefill_contract(PrefillContract::P2).unwrap();
        let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(120)).into_iter();
        let e = 288;
        a.enable_tp(Arc::new(comms.next().unwrap()), ExpertOwnership::range(e, 0, 2).unwrap()).unwrap();
        b.enable_tp(Arc::new(comms.next().unwrap()), ExpertOwnership::range(e, 1, 2).unwrap()).unwrap();
        a.enable_head_split(bits).unwrap();
        b.enable_head_split(bits).unwrap();
        let t = std::time::Instant::now();
        let (d0, d1) = std::thread::scope(|sc| {
            let h0 = sc.spawn(|| script_cut(&mut a, &mut Vec::new(), &on_grid));
            let h1 = sc.spawn(|| script_cut(&mut b, &mut Vec::new(), &on_grid));
            (h0.join().expect("rank 0"), h1.join().expect("rank 1"))
        });
        let name = split_names(bits);
        assert_eq!(d0, d1, "P2 split {name}, sliced: the two ranks' answers differ");
        let differ = d0.iter().zip(&reference).filter(|(x, y)| x != y).count();
        assert_eq!(differ, 0, "P2 split {name}: slicing at grid cuts changed {differ} of {} answering calls", d0.len());
        eprintln!(
            "P2 split {name}, sliced at grid cuts: BYTE-IDENTICAL to the unsliced P2 single node, {} answering calls, {:.1?}",
            d0.len(),
            t.elapsed()
        );
        drop((a, b));
    }
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// Prefill `toks[from..]` into `slot` (which holds `toks[..from]`) in slices of [`CHUNK`] rows, answering on the last, then decode `steps`
/// greedy single-row windows: one digest per answering call.
fn prompt_then_greedy(m: &mut GlmModel, slot: usize, toks: &[u32], from: usize, steps: usize) -> Vec<u64> {
    let mut out = Vec::new();
    let mut at = from;
    let mut last = 0u32;
    while at < toks.len() {
        let end = (at + CHUNK).min(toks.len());
        let want = (end == toks.len()).then_some(COUNT);
        if let Some(c) = m.prefill_chunk(slot, at, &toks[at..end], want).unwrap() {
            out.push(digest_call(m, &c, 1));
            last = c.ids[0] as u32;
        }
        at = end;
    }
    let mut pos = toks.len();
    for _ in 0..steps {
        let b = RowBatch::from_requests(&[gb_engine::RowRequest { slot, tokens: vec![last], first_position: pos as u64 }]);
        let c = m.window_candidates(&b, COUNT).unwrap();
        out.push(digest_call(m, &c, 1));
        m.commit(slot, 1);
        pos += 1;
        last = c.ids[0] as u32;
    }
    m.flush().unwrap();
    out
}

/// One rank's (or the single node's) prefix-cache script under P2: the fresh prompt, then for each cut a donor slot prefilled up to the cut, its
/// snapshot taken BETWEEN chunks and its pages retained, a stranger's prompt in the target slot, the restore over it and the rest of the prompt:
/// one digest list per run, the fresh run first.
fn restore_script(m: &mut GlmModel, toks: &[u32], cuts: &[usize]) -> Vec<Vec<u64>> {
    let mut runs = Vec::new();
    m.begin_slot(0).unwrap();
    runs.push(prompt_then_greedy(m, 0, toks, 0, 4));
    m.end_slot(0);
    for &c in cuts {
        m.begin_slot(1).unwrap();
        let mut at = 0;
        while at < c {
            let end = gb_model::families::glm::prefix::slice_end(at, toks.len(), CHUNK, &[c]);
            m.prefill_chunk(1, at, &toks[at..end], None).unwrap();
            at = end;
        }
        let snap = m.alloc_snapshot().expect("a free snapshot slot");
        m.prefix_snapshot(1, snap, c).unwrap();
        let pages = m.retain_prefix_pages(1, c);
        m.end_slot(1);
        m.begin_slot(2).unwrap();
        m.prefill_chunk(2, 0, &toks[toks.len() - 300..], None).unwrap();
        m.restore_prefix(2, snap, &pages, c).unwrap();
        runs.push(prompt_then_greedy(m, 2, toks, c, 4));
        m.end_slot(2);
        m.release_prefix(snap, &pages);
    }
    runs
}

/// The two-node prefix cache at the model level (`docs/GLM_TP.md` 4.2, before the call protocol): each rank of a two-rank P2 group under the serving
/// split `kda,lm,mla,sh` (rank strips loaded) snapshots and restores ITS OWN state between chunks. For cuts at 16, 496, 1,040 and 2,272 tokens of a
/// 2,300-token prompt, the restored prompt answers, and decodes four greedy steps, byte-identically to the fresh prompt, on both ranks and as the
/// P2 single node does with the same script. A snapshot inside a chunk is refused on a split rank (its split arm runs each KDA chunk whole).
#[test]
#[ignore]
fn p2_two_ranks_restore_their_own_snapshots_byte_for_byte() {
    use gb_model::families::glm::model::tp::split_from_vars;
    let Some(dir) = pack_dir() else { return };
    let layers: usize = std::env::var("REXL3_GLM_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(60 * GIB);
    let guard = Guard::start(20 * GIB);
    let toks = Rng(0x51ce_7a11_d00d_f00d).tokens(2300);
    let cuts = [16usize, 496, 1040, 2272];
    let mut a = load(&dir, layers);
    a.set_prefill_contract(PrefillContract::P2).unwrap();
    a.set_graphs(false);
    a.enable_prefix_cache(4).unwrap();
    let reference = restore_script(&mut a, &toks, &cuts);
    for (i, r) in reference.iter().enumerate().skip(1) {
        assert_eq!(r, &reference[0], "P2 single node: the restore at {} differs from the fresh prompt", cuts[i - 1]);
    }
    drop(a);
    eprintln!("P2 single node: the restores at {cuts:?} equal the fresh prompt ({} answering calls each)", reference[0].len());
    let bits = split_from_vars(|_| Some("kda,lm,mla,sh".into())).unwrap();
    let (mut a, mut b) = (load_rank_strips(&dir, layers, 0), load_rank_strips(&dir, layers, 1));
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(120)).into_iter();
    let e = 288;
    for (m, r) in [(&mut a, 0), (&mut b, 1)] {
        m.set_prefill_contract(PrefillContract::P2).unwrap();
        m.enable_tp(Arc::new(comms.next().unwrap()), ExpertOwnership::range(e, r, 2).unwrap()).unwrap();
        m.enable_head_split(bits).unwrap();
        m.enable_prefix_cache(4).unwrap();
    }
    // a split rank takes no snapshot inside a chunk (refused before the chunk runs, so the ranks stay in step)
    for m in [&mut a, &mut b] {
        m.begin_slot(1).unwrap();
        let snap = m.alloc_snapshot().unwrap();
        let err = m.prefill_chunk_cuts(1, 0, &toks[..CHUNK], None, Vec::new(), &[(256, snap)]).unwrap_err().to_string();
        assert!(err.contains("KDA head split"), "{err}");
        m.release_prefix(snap, &[]);
        m.end_slot(1);
    }
    let t = std::time::Instant::now();
    let (d0, d1) = std::thread::scope(|sc| {
        let h0 = sc.spawn(|| restore_script(&mut a, &toks, &cuts));
        let h1 = sc.spawn(|| restore_script(&mut b, &toks, &cuts));
        (h0.join().expect("rank 0"), h1.join().expect("rank 1"))
    });
    assert_eq!(d0, d1, "the two ranks' answers differ");
    assert_eq!(d0, reference, "the group's fresh or restored answers differ from the P2 single node's");
    eprintln!(
        "P2 split kda,lm,mla,sh: every rank restores its own snapshots at {cuts:?}: BYTE-IDENTICAL to the fresh prompt and to the single node, {:.1?}",
        t.elapsed()
    );
    drop((a, b));
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// The rank path on the WHOLE pack at one rank (`SingleRank`, every expert): the pairs stage and the combine over the (unsummed) pair
/// buffer instead of the fused decode epilogue, on all 45 layers, against the single node. Needs about 110 GiB of MemAvailable (one 87
/// GiB model); prints "not run" otherwise.
#[test]
#[ignore]
fn the_rank_path_on_the_whole_pack_at_one_rank_equals_the_single_node_model() {
    let Some(dir) = pack_dir() else { return };
    if mem_available() < 110 * GIB {
        eprintln!(
            "not run: MemAvailable is {:.1} GiB, this test needs 110 GiB (the whole pack and 20 GiB to spare)",
            gib(mem_available())
        );
        return;
    }
    let guard = Guard::start(20 * GIB);
    evict_pack(&dir);
    let mut m = load(&dir, 45);
    m.set_graphs(true);
    let reference = script(&mut m);
    let comm: Arc<dyn TpComm> = Arc::new(SingleRank);
    m.enable_tp(comm, ExpertOwnership::all(288)).unwrap();
    let ranked = script(&mut m);
    assert_eq!(ranked, reference, "the rank path differs from the single node on the whole pack");
    eprintln!(
        "BYTE-IDENTICAL on 45 layers: {} answering calls; {} exchanges (no-ops at one rank)",
        ranked.len(),
        m.tp().unwrap().exchanges
    );
    drop(m);
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// Lane P step (c): the same group under the P2 prefill contract (`REXL3_GLM_PREFILL_CONTRACT=p2` semantics, switched per model before the
/// group forms): the prompt chunks' routed experts take the P2 rank path (each rank its own pairs and part, the parts' swap of one fp32 row
/// a prompt row, the sum in rank order: `model/tp/p2.rs`), the windows keep P1's pair all-reduce. Every answer of both ranks must equal
/// the single node's under P2, byte for byte (P2 at TP=2 == P2 on one node). Then a rank whose ownership is not its combine group's is
/// refused by name.
#[test]
#[ignore]
fn p2_two_ranks_on_one_gpu_equal_the_single_node_p2_model_byte_for_byte() {
    let Some(dir) = pack_dir() else { return };
    let layers: usize = std::env::var("REXL3_GLM_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(60 * GIB);
    let guard = Guard::start(20 * GIB);
    let mut a = load(&dir, layers);
    let mut b = load(&dir, layers);
    a.set_prefill_contract(PrefillContract::P2).unwrap();
    b.set_prefill_contract(PrefillContract::P2).unwrap();
    a.set_graphs(true);
    let ref_graphs = script(&mut a);
    a.set_graphs(false);
    let ref_eager = script(&mut a);
    assert_eq!(ref_graphs, ref_eager, "the single node under P2: graphs and eager differ on this prefix");
    let p1_ref = {
        let mut c = load(&dir, layers);
        c.set_prefill_contract(PrefillContract::P1).unwrap();
        c.set_graphs(false);
        let d = script(&mut c);
        drop(c);
        d
    };
    let moved = ref_eager.iter().zip(&p1_ref).filter(|(x, y)| x != y).count();
    assert!(moved > 0, "P2 left every answer at P1's bits: the P2 path did not run");
    eprintln!("single node under P2: {} answering calls, graphs == eager, {moved} differ from P1's (P2 ran)", ref_eager.len());

    let e = 288;
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(120)).into_iter();
    a.enable_tp(Arc::new(comms.next().unwrap()), ExpertOwnership::range(e, 0, 2).unwrap()).unwrap();
    b.enable_tp(Arc::new(comms.next().unwrap()), ExpertOwnership::range(e, 1, 2).unwrap()).unwrap();
    let t = std::time::Instant::now();
    let (d0, d1, a, b, au0, au1) = std::thread::scope(|sc| {
        let h0 = sc.spawn(move || {
            let mut au = Vec::new();
            let d = script_with(&mut a, &mut au);
            (d, a, au)
        });
        let h1 = sc.spawn(move || {
            let mut au = Vec::new();
            let d = script_with(&mut b, &mut au);
            (d, b, au)
        });
        let (d0, a, au0) = h0.join().expect("rank 0");
        let (d1, b, au1) = h1.join().expect("rank 1");
        (d0, d1, a, b, au0, au1)
    });
    eprintln!("two P2 ranks ran the script in {:.1?}", t.elapsed());
    assert_eq!(d0, d1, "the two P2 ranks' answers differ");
    assert_eq!(d0, ref_eager, "the P2 group's answers differ from the P2 single node's");
    // the TP-path launch audit, per rank (each rank's thread counts its own launches): every prompt chunk makes the P2 GEMM launches of
    // every dense linear, one P2 expert run and one part swap a MoE layer; a window makes none of either
    for (r, (au, m)) in [(&au0, &a), (&au1, &b)].into_iter().enumerate() {
        let (mut chunks, mut windows) = (0, 0);
        for (i, c) in au.iter().enumerate() {
            if c.prompt {
                let want = (
                    p2_expected_launches(m, c.rows),
                    p2_expected_moe_runs(m),
                    p2_expected_moe_runs(m),
                    p2_expected_kda_runs(m),
                    p2_expected_mla_runs(m, c.rows),
                );
                assert_eq!(
                    (c.launches, c.runs, c.swaps, c.kda, c.mla),
                    want,
                    "rank {r} call {i} (a {}-row prompt chunk): launches, runs, swaps, chunked KDA, MLA absorb / expand",
                    c.rows
                );
                chunks += 1;
            } else {
                assert_eq!(
                    (c.launches, c.runs, c.swaps, c.kda, c.mla),
                    (0, 0, 0, 0, 0),
                    "rank {r} call {i} (a {}-row window) took a P2 path",
                    c.rows
                );
                windows += 1;
            }
        }
        let first = au.iter().find(|c| c.prompt).unwrap();
        eprintln!(
            "P2 TP LAUNCH AUDIT rank {r}: {chunks} prompt chunks, each its {} P2 GEMM launches ({}-row chunk), {} expert runs, {} part \
             swaps, {} chunked KDA recurrences and {} MLA absorb / expand launches; {windows} windows with none",
            first.launches, first.rows, first.runs, first.swaps, first.kda, first.mla
        );
    }
    assert_eq!(au0, au1, "the two ranks' launch audits differ");
    let (t0r, t1r) = (a.tp().unwrap(), b.tp().unwrap());
    assert_eq!((t0r.exchanges, t0r.exchanged_bytes), (t1r.exchanges, t1r.exchanged_bytes), "the ranks exchanged alike");
    eprintln!(
        "P2 BYTE-IDENTICAL: {} answering calls (candidates + every fp16 logit) == the P2 single node; {} exchanges of {:.1} MiB in all per rank \
         (the prompt chunks' part swaps and the windows' pair all-reduces); the P2 exchange holds {:.1} MiB",
        d0.len(),
        t0r.exchanges,
        t0r.exchanged_bytes as f64 / 1048576.0,
        t0r.p2_exchange_bytes() as f64 / 1048576.0
    );
    drop((a, b));

    // a rank whose ownership is not its combine group's is refused by name (the sum's order is the groups')
    let mut c = load(&dir, layers);
    c.set_prefill_contract(PrefillContract::P2).unwrap();
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(5)).into_iter();
    let _c0 = comms.next().unwrap();
    let err = c.enable_tp(Arc::new(comms.next().unwrap()), ExpertOwnership::range(e, 0, 2).unwrap()).unwrap_err().to_string();
    assert!(err.contains("must own the experts of combine group 1"), "{err}");
    eprintln!("refused by name: {err}");
    drop(c);
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

// ------------------------------------------------------------------ the restore verified per rank (content digests), cold otherwise

/// `prompt` into `slot` from 0 in slices of [`CHUNK`] rows, no answer.
fn prefill_into(m: &mut GlmModel, slot: usize, prompt: &[u32]) {
    let mut at = 0;
    while at < prompt.len() {
        let end = (at + CHUNK).min(prompt.len());
        m.prefill_chunk(slot, at, &prompt[at..end], None).unwrap();
        at = end;
    }
}

/// A donor slot prefilled with `prompt` up to `cut` (slices that end there), its snapshot taken between chunks, its pages retained and the
/// snapshot sealed (its content digest recorded): `(snapshot slot, pages, digest)`.
fn take_sealed(m: &mut GlmModel, slot: usize, prompt: &[u32], cut: usize) -> (u32, Vec<u32>, u64) {
    m.begin_slot(slot).unwrap();
    let mut at = 0;
    while at < cut {
        let end = gb_model::families::glm::prefix::slice_end(at, prompt.len(), CHUNK, &[cut]);
        m.prefill_chunk(slot, at, &prompt[at..end], None).unwrap();
        at = end;
    }
    let snap = m.alloc_snapshot().expect("a free snapshot slot");
    m.prefix_snapshot(slot, snap, cut).unwrap();
    let pages = m.retain_prefix_pages(slot, cut);
    let d = m.seal_snapshot(snap, &pages).unwrap();
    m.end_slot(slot);
    (snap, pages, d)
}

/// One rank's (or the single node's) verified-restore script under P2: the fresh prompt; two sealed snapshots at `cut` (the prompt's and a
/// stranger's); a target slot holding the stranger's state digests unlike the snapshot (a stale restore); the restore of the prompt's
/// snapshot leaves exactly its digest and answers as the fresh prompt; a restore of it over the stranger's pages is REFUSED by the model's
/// own check; and the cold fallback over the refused restore (`begin_slot`, the prompt from 0) answers as the fresh prompt.
fn verify_script(m: &mut GlmModel, toks: &[u32], other: &[u32], cut: usize) -> Vec<Vec<u64>> {
    let mut runs = Vec::new();
    m.begin_slot(0).unwrap();
    runs.push(prompt_then_greedy(m, 0, toks, 0, 4));
    m.end_slot(0);
    let (sa, pa, da) = take_sealed(m, 1, toks, cut);
    let (sb, pb, _) = take_sealed(m, 1, other, cut);
    m.begin_slot(2).unwrap();
    prefill_into(m, 2, &other[..cut + 160]);
    assert_ne!(
        m.prefix_state_digest(2, cut).unwrap(),
        da,
        "a slot holding another prompt (a stale restore) digests unlike the snapshot"
    );
    m.restore_prefix(2, sa, &pa, cut).unwrap();
    assert_eq!(m.prefix_state_digest(2, cut).unwrap(), da, "the restore leaves exactly the snapshot's digest");
    runs.push(prompt_then_greedy(m, 2, toks, cut, 4));
    m.end_slot(2);
    m.begin_slot(2).unwrap();
    prefill_into(m, 2, &other[..cut + 160]);
    let e = m.restore_prefix(2, sa, &pb, cut).expect_err("a restore over another prompt's pages is refused").to_string();
    assert!(e.contains("digest"), "{e}");
    m.begin_slot(2).unwrap();
    runs.push(prompt_then_greedy(m, 2, toks, 0, 4));
    m.end_slot(2);
    m.release_prefix(sa, &pa);
    m.release_prefix(sb, &pb);
    runs
}

/// The restore verified by content digest at the model level, under P2 and the serving split `kda,lm,mla,sh`: the single node and each rank
/// of the two-rank group (rank strips) seal their own snapshots and verify their own restores; the restored and the cold-fallback answers
/// equal the fresh prompt's byte for byte on both ranks and on the single node.
#[test]
#[ignore]
fn p2_two_ranks_seal_and_verify_their_own_restores() {
    use gb_model::families::glm::model::tp::split_from_vars;
    let Some(dir) = pack_dir() else { return };
    let layers: usize = std::env::var("REXL3_GLM_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(60 * GIB);
    let guard = Guard::start(20 * GIB);
    let toks = Rng(0x51ce_7a11_d00d_f00d).tokens(2300);
    let other = Rng(0x0dd5_eed0_c0ff_ee11).tokens(2300);
    let cut = 1040;
    let mut a = load(&dir, layers);
    a.set_prefill_contract(PrefillContract::P2).unwrap();
    a.set_graphs(false);
    a.enable_prefix_cache(4).unwrap();
    let reference = verify_script(&mut a, &toks, &other, cut);
    assert!(
        reference.iter().all(|r| r == &reference[0]),
        "P2 single node: a restored or fallback answer differs from the fresh prompt"
    );
    drop(a);
    let bits = split_from_vars(|_| Some("kda,lm,mla,sh".into())).unwrap();
    let (mut a, mut b) = (load_rank_strips(&dir, layers, 0), load_rank_strips(&dir, layers, 1));
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(120)).into_iter();
    for (m, r) in [(&mut a, 0), (&mut b, 1)] {
        m.set_prefill_contract(PrefillContract::P2).unwrap();
        m.enable_tp(Arc::new(comms.next().unwrap()), ExpertOwnership::range(288, r, 2).unwrap()).unwrap();
        m.enable_head_split(bits).unwrap();
        m.enable_prefix_cache(4).unwrap();
    }
    let (d0, d1) = std::thread::scope(|sc| {
        let h0 = sc.spawn(|| verify_script(&mut a, &toks, &other, cut));
        let h1 = sc.spawn(|| verify_script(&mut b, &toks, &other, cut));
        (h0.join().expect("rank 0"), h1.join().expect("rank 1"))
    });
    assert_eq!(d0, d1, "the two ranks' answers differ");
    assert_eq!(d0, reference, "the group's answers differ from the single node's");
    eprintln!("P2 split kda,lm,mla,sh: every rank seals and verifies its own restore at {cut}; restored and fallback answers BYTE-IDENTICAL");
    drop((a, b));
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// A follower whose restore can be told to leave the slot as it is and answer Ok (a stale restore), everything else its `GlmRank`'s.
struct StaleRank {
    r: gb_model::families::glm::tp::GlmRank,
    stale: Arc<std::sync::atomic::AtomicBool>,
}

impl gb_model::lockstep::LockstepRuntime for StaleRank {
    type Call = gb_model::families::glm::tp::GlmCall;
    fn vocab_size(&self) -> usize {
        self.r.vocab_size()
    }
    fn has_mtp(&self) -> bool {
        self.r.has_mtp()
    }
    fn snapshots_supported() -> bool {
        true
    }
    fn begin_slot(&mut self, slot: usize) -> Result<(), gb_model::error::ModelError> {
        self.r.begin_slot(slot)
    }
    fn end_slot(&mut self, slot: usize) {
        self.r.end_slot(slot)
    }
    fn prefill_chunk(
        &mut self,
        b: &RowBatch,
        want: Option<usize>,
    ) -> Result<Option<gb_model::forward::Candidates>, gb_model::error::ModelError> {
        self.r.prefill_chunk(b, want)
    }
    fn window_candidates(
        &mut self,
        b: &RowBatch,
        count: usize,
    ) -> Result<gb_model::forward::Candidates, gb_model::error::ModelError> {
        self.r.window_candidates(b, count)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.r.commit(slot, keep)
    }
    fn flush(&mut self) -> Result<(), gb_model::error::ModelError> {
        self.r.flush()
    }
    fn apply_call(
        &mut self,
        seq: u64,
        call: gb_model::families::glm::tp::GlmCall,
    ) -> Result<gb_model::lockstep::CallReply, gb_model::error::ModelError> {
        if call.is_prefix() {
            return gb_model::families::glm::tp::apply_prefix(self, seq, call);
        }
        gb_model::families::glm::tp::apply_family(self, call)
    }
}

impl gb_model::families::glm::tp_part::PartRuntime for StaleRank {
    fn prefill_batches(&self) -> bool {
        self.r.prefill_batches()
    }
    fn prefill_part_rows(&self) -> usize {
        self.r.prefill_part_rows()
    }
    fn prefill_layers(&self) -> usize {
        self.r.prefill_layers()
    }
    fn prefill_part(
        &mut self,
        call: &gb_model::families::glm::tp_part::PrefillPart,
    ) -> Result<Option<gb_model::forward::Candidates>, gb_model::error::ModelError> {
        self.r.prefill_part(call)
    }
}

impl gb_model::families::glm::tp::PrefixRuntime for StaleRank {
    fn enable_snapshots(&mut self, slots: usize) -> Result<u64, gb_model::error::ModelError> {
        self.r.enable_snapshots(slots)
    }
    fn free_snapshots(&self) -> usize {
        self.r.free_snapshots()
    }
    fn p2_anchor(&self, slot: usize) -> Option<usize> {
        self.r.p2_anchor(slot)
    }
    fn alloc_snapshot(&mut self) -> Option<u32> {
        self.r.alloc_snapshot()
    }
    fn prefix_snapshot(&mut self, slot: usize, snap: u32, len: usize) -> Result<bool, gb_model::error::ModelError> {
        self.r.prefix_snapshot(slot, snap, len)
    }
    fn retain_prefix_pages(&mut self, slot: usize, len: usize) -> Vec<u32> {
        self.r.retain_prefix_pages(slot, len)
    }
    fn restore_prefix(&mut self, slot: usize, snap: u32, pages: &[u32], len: usize) -> Result<(), gb_model::error::ModelError> {
        if self.stale.load(std::sync::atomic::Ordering::SeqCst) {
            return Ok(());
        }
        self.r.restore_prefix(slot, snap, pages, len)
    }
    fn release_prefix(&mut self, snap: u32, pages: &[u32]) {
        self.r.release_prefix(snap, pages)
    }
    fn seal_snapshot(&mut self, snap: u32, pages: &[u32]) -> Result<u64, gb_model::error::ModelError> {
        self.r.seal_snapshot(snap, pages)
    }
    fn prefix_state_digest(&mut self, slot: usize, len: usize) -> Result<u64, gb_model::error::ModelError> {
        self.r.prefix_state_digest(slot, len)
    }
    fn kv_pages(&self) -> (usize, Vec<usize>) {
        self.r.kv_pages()
    }
    fn snapshot_handles(&mut self) -> &mut gb_model::families::glm::tp::SnapHandles {
        self.r.snapshot_handles()
    }
}

impl gb_model::families::glm::tp_mtp::GlmMtpRuntime for StaleRank {
    fn mtp_book(&self, slot: usize) -> Option<gb_model::families::glm::model::TapBook> {
        self.r.mtp_book(slot)
    }
    fn mtp_draft_keyed(
        &mut self,
        asks: &[gb_model::families::glm::model::DraftAsk<'_>],
        keyed: &[Option<gb_sampling::Sampling>],
    ) -> Result<Vec<Option<gb_model::families::glm::model::Drafts>>, gb_model::error::ModelError> {
        self.r.mtp_draft_keyed(asks, keyed)
    }
    fn mtp_disable_slot(&mut self, slot: usize) {
        self.r.mtp_disable_slot(slot)
    }
}

impl gb_model::families::glm::tp_vision::ImageRuntime for StaleRank {
    fn prefill_image(
        &mut self,
        call: &gb_model::families::glm::tp_vision::PrefillImage,
    ) -> Result<Option<gb_model::forward::Candidates>, gb_model::error::ModelError> {
        self.r.prefill_image(call)
    }
}

/// `prompt` into `slot` from `done` through the backend as the engine drives it (`prefill_slice_cuts`): the answer's bits and the snapshots.
fn backend_prefill<B: gb_engine::Backend>(
    be: &mut B,
    slot: usize,
    prompt: &[u32],
    mut done: usize,
    cuts: &[usize],
) -> (Vec<(u32, u32)>, Vec<gb_engine::Captured>) {
    let (mut caps, mut row) = (Vec::new(), None);
    while done < prompt.len() {
        let s = be.prefill_slice_cuts(slot, prompt, &[], done, COUNT, cuts);
        caps.extend(s.captured);
        done = s.step.done;
        row = s.step.row;
    }
    let r = row.expect("the last slice answers");
    (r.ids.iter().zip(&r.values).map(|(i, v)| (*i, v.to_bits())).collect(), caps)
}

/// The two-node protocol on the real model (two ranks on one GPU, P2, split `kda,lm,mla,sh`, rank 0 a `GlmTpBackend` over a `TpLeader`,
/// rank 1 `run_follower`): a follower that leaves stale state and answers Ok is caught PER REQUEST by its content digest, the restore is
/// refused, the cold fallback answers byte-identically to the cold prompt, the group stays up; then a clean restore is verified and its
/// continuation equals the cold prompt too.
#[test]
#[ignore]
fn p2_two_ranks_refuse_a_stale_follower_restore_and_fall_back_cold() {
    use gb_engine::Backend;
    use gb_model::families::glm::model::tp::split_from_vars;
    use gb_model::families::glm::tp::GlmRank;
    use gb_model::families::glm::tp_backend::GlmTpBackend;
    use gb_model::lockstep::{run_follower, FollowerConfig, FollowerExit, Link, TpLeader, TpTimeouts};
    let Some(dir) = pack_dir() else { return };
    let layers: usize = std::env::var("REXL3_GLM_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(60 * GIB);
    let guard = Guard::start(20 * GIB);
    let toks = Rng(0x51ce_7a11_d00d_f00d).tokens(2300);
    let other = Rng(0x0dd5_eed0_c0ff_ee11).tokens(1800);
    let cut = 1040;
    let bits = split_from_vars(|_| Some("kda,lm,mla,sh".into())).unwrap();
    let (mut a, mut b) = (load_rank_strips(&dir, layers, 0), load_rank_strips(&dir, layers, 1));
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(120)).into_iter();
    for (m, r) in [(&mut a, 0), (&mut b, 1)] {
        m.set_prefill_contract(PrefillContract::P2).unwrap();
        m.enable_tp(Arc::new(comms.next().unwrap()), ExpertOwnership::range(288, r, 2).unwrap()).unwrap();
        m.enable_head_split(bits).unwrap();
    }
    let t =
        TpTimeouts { ready: Duration::from_secs(60), peer: Duration::from_secs(120), hang_exit: None, ..TpTimeouts::default() };
    let stale = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (la, lb) = Link::pair(Duration::from_secs(60)).unwrap();
    let f = StaleRank { r: GlmRank::new(b), stale: stale.clone() };
    let ft = t.clone();
    let h = std::thread::spawn(move || run_follower(f, lb, FollowerConfig::new(ft)));
    let leader = TpLeader::lead(GlmRank::new(a), la, t);
    let budget = gb_engine::CacheBudget { snapshot_slots: 4, page_blocks: 24, snapshot_bytes: 4096, page_bytes: 1 << 20 };
    let mut be = GlmTpBackend::new(leader, CHUNK, SLOTS).with_contract(PrefillContract::P2).with_prefix_cache(7, budget).unwrap();
    let (cold, _) = backend_prefill(&mut be, 0, &toks, 0, &[]);
    be.release(0);
    let (_, caps) = backend_prefill(&mut be, 1, &toks, 0, &[cut]);
    be.release(1);
    let handle = caps.into_iter().find(|c| c.len == cut).expect("the cut was captured").handle;
    let _ = backend_prefill(&mut be, 2, &other, 0, &[]);
    stale.store(true, std::sync::atomic::Ordering::SeqCst);
    let e = be.prefix_restore(2, &handle, cut).expect_err("the stale follower's restore is refused");
    assert!(e.contains("rank 1 restored digest"), "{e}");
    let _ = be.take_error();
    assert_eq!(be.leader.failure(), None, "the group stays up");
    let (fallback, _) = backend_prefill(&mut be, 2, &toks, 0, &[]);
    assert_eq!(fallback, cold, "the cold fallback answers as the cold prompt");
    stale.store(false, std::sync::atomic::Ordering::SeqCst);
    be.prefix_restore(2, &handle, cut).expect("a clean restore is verified on both ranks");
    let (warm, _) = backend_prefill(&mut be, 2, &toks, cut, &[]);
    assert_eq!(warm, cold, "the verified warm continuation answers as the cold prompt");
    be.release(2);
    be.prefix_release(handle);
    assert_eq!(be.leader.failure(), None);
    drop(be);
    assert!(matches!(h.join().unwrap(), Ok(FollowerExit::Shutdown(_))));
    eprintln!(
        "P2 two ranks over the protocol: a stale follower restore refused per request, cold and warm answers BYTE-IDENTICAL"
    );
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// One node (`GlmBackend`, P2, an 8-layer prefix): the backend seals every capture it takes (here the cut at 1,040 falls inside a
/// 512-row chunk, so it is an in-chunk capture), and a restore of a sealed snapshot is verified. A handle that pairs one prompt's snapshot
/// slot with another prompt's pages is refused (the restored state does not digest as the seal), the cold fallback answers as the cold
/// prompt, and the clean restore's continuation answers as the cold prompt too. (Without the backend's seal, nothing is verified and the
/// mismatched restore would be used.)
#[test]
#[ignore]
fn p2_the_single_node_backend_seals_its_captures_and_refuses_a_mismatched_restore() {
    use gb_engine::Backend;
    use gb_model::families::glm::backend::GlmBackend;
    let Some(dir) = pack_dir() else { return };
    let layers: usize = std::env::var("REXL3_GLM_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(30 * GIB);
    let guard = Guard::start(20 * GIB);
    let toks = Rng(0x51ce_7a11_d00d_f00d).tokens(2300);
    let other = Rng(0x0dd5_eed0_c0ff_ee11).tokens(2300);
    let cut = 1040;
    let mut m = load(&dir, layers);
    m.set_prefill_contract(PrefillContract::P2).unwrap();
    let budget = gb_engine::CacheBudget { snapshot_slots: 4, page_blocks: 24, snapshot_bytes: 4096, page_bytes: 1 << 20 };
    let mut be = GlmBackend::new(m).with_prefix_cache(7, budget).unwrap();
    let (cold, _) = backend_prefill(&mut be, 0, &toks, 0, &[]);
    be.release(0);
    let (_, ca) = backend_prefill(&mut be, 1, &toks, 0, &[cut]);
    be.release(1);
    let (_, cb) = backend_prefill(&mut be, 1, &other, 0, &[cut]);
    be.release(1);
    let ha = ca.into_iter().find(|c| c.len == cut).expect("the prompt's cut was captured").handle;
    let hb = cb.into_iter().find(|c| c.len == cut).expect("the stranger's cut was captured").handle;
    let mismatched = gb_engine::PrefixHandle { snap: ha.snap, pages: hb.pages.clone() };
    let e = be.prefix_restore(2, &mismatched, cut).expect_err("a restore over another prompt's pages is refused");
    assert!(e.contains("digest"), "{e}");
    let _ = be.take_error();
    let (fallback, _) = backend_prefill(&mut be, 2, &toks, 0, &[]);
    assert_eq!(fallback, cold, "the cold fallback answers as the cold prompt");
    be.prefix_restore(2, &ha, cut).expect("the clean restore is verified");
    let (warm, _) = backend_prefill(&mut be, 2, &toks, cut, &[]);
    assert_eq!(warm, cold, "the verified warm continuation answers as the cold prompt");
    be.release(2);
    be.prefix_release(ha);
    be.prefix_release(hb);
    eprintln!(
        "one node: the backend's in-chunk capture is sealed, a mismatched restore refused, cold and warm answers BYTE-IDENTICAL"
    );
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}
