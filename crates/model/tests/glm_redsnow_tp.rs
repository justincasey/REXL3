//! The RED-SNOW 4.91bpw pack (4, 5 and 6-bit experts mixed inside a layer, a plain-fp16 indexer `wq_b`) as a TWO-RANK group ON ONE GPU:
//! a layer prefix loaded as ONE model holding every expert (mixed layers bound as subbanks, run as pairs then combine) against two models
//! of the same prefix that each map and load ONLY their share of the routed experts (`GlmFamily::map_pack_owned` with
//! `ExpertOwnership::range(288, r, 2)`: every MoE layer is subbanks of the owned experts) made ranks 0 and 1 of an in-process group
//! (`LoopbackGroup`, a thread and a stream each; the pair buffer summed by the loopback collective). Every candidate and every fp16 logit
//! of every call must be byte-identical across the ranks and equal to the one-model run, which runs with graphs on and with graphs off.
//! The negative control (both ranks own 0..144) must differ.
//!
//! The script (as `glm_tp.rs`): a 2,300-token prompt in 512-row chunks (the batched expert path, past the DSA sparse threshold, so the
//! indexer and its `wq_b` run), a 200-token prompt in 128-row chunks, a 37-token prompt, a 9-row window of three sequences with partial
//! commits, six greedy 3-row windows. Before that, the plain-fp16 `wq_b` of the first MLA layer is checked against a host matmul of the
//! pack's own fp16 matrix (R = 1 against R = 2 cannot see a transposed or misplaced matrix).
//!
//! The same test on the uniform 2.25bpw pack (`REXL3_REDSNOW_DIR` pointed at it) proves a share-only rank load there too (its MoE layers
//! become one-subbank layers on a rank).
//!
//! Env: `REXL3_REDSNOW_DIR` (the pack; "not run" without it), `REXL3_REDSNOW_TP_LAYERS` (default 8: dense 0..2, MoE 3..7 with the
//! uniform 6/6/6 layer 3 and the mixed layers 4..7; 14 adds layer 13 and its gate != up expert). Under the GPU lock, alone:
//!
//!     flock -x /tmp/rexl3-gpu.lock env REXL3_REDSNOW_DIR=... cargo test --release -p gb-model --test glm_redsnow_tp -- --ignored --nocapture --test-threads=1
//!
//! `p2_redsnow_prefix_two_ranks_equal_one_p2_model_byte_for_byte`: the same script under the P2 prefill contract (switched per model
//! before the group forms): the 6-bit experts on the P2 expert GEMMs, every rank's several subbanks a MoE layer through the P2 rank path
//! (`glm_p2_moe::pairs_subbanks_rank`, its part, the part swap), layer 13's split gate/up subbank through the decode pairs stage. Both
//! ranks' answers must equal the one P2 model's, byte for byte, and differ from P1's somewhere (P2 ran).
//!
//! `p1_` / `p2_several_prompts_in_one_forward_equal_each_prompts_solo_forward` and
//! `p1_` / `p2_a_forward_paused_with_windows_and_a_prompt_between_its_parts_equals_the_whole_forward` (`families/glm/model/part.rs`),
//! under each prefill contract: the slices of three prompts (one a continuation) in ONE prompt forward give each prompt the answer, the
//! state (the windows after) and the drafter context (the drafts after) of its own forward; a prompt forward paused at layer bounds, with
//! a verify window, another slot's whole prompt and a window of two slots run between its parts, gives every call the bits of the run
//! with the forward whole. Both on one model and on the two ranks of the serving split (`kda,lm,mla,sh`; under P2 the
//! sequence-parallel residual), whose answers must equal the one model's.
//!
//! `p1_the_last_layers_tail_on_the_head_rows_equals_every_rows` (`ops::plan_forward_head_rows`): on a prefix whose last layer is KDA,
//! the P1 prompt forwards run with that layer's tail on the head rows alone and with it on every row (`GlmModel::set_head_rows`) give
//! every answer, every layer's state after every prompt chunk and the windows after byte for byte (chunked, several prompts in one
//! forward, a forward paused at layer bounds), on one model and on the two ranks of the serving split.
//!
//! `p2_the_last_layers_tail_on_the_head_rows_equals_every_rows`: the same under P2, on one P2 model and on the two serving ranks (the
//! width split, the sequence-parallel residual, the drafter's taps relayed when `REXL3_DFLASH_DIR` is set), whose last chunk and
//! batched forward are row-split so that head rows live on either rank.

mod glm_support;

use gb_cuda::moe::ExpertOwnership;
use gb_cuda::DevBuf;
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_model::families::glm::contract::PrefillContract;
use gb_model::families::glm::device::{FfnDevice, GlmDevice, MixerDevice, WqBDevice};
use gb_model::families::glm::load::{boot_mapped, LoadOptions};
use gb_model::families::glm::model::{GlmModel, ModelOptions};
use gb_model::families::glm::{GlmFamily, RegionKind};
use gb_model::families::nemotron_h_omni::tp_comm::LoopbackGroup;
use gb_model::rows::RowBatch;
use glm_support::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

const CHUNK: usize = 512;
const SLOTS: usize = 3;
const COUNT: usize = 8;
const E: usize = 288;

fn redsnow_dir() -> Option<PathBuf> {
    let d = std::env::var("REXL3_REDSNOW_DIR").ok().map(PathBuf::from).filter(|p| p.join("config.json").exists());
    if d.is_none() {
        eprintln!("REXL3_REDSNOW_DIR not set: not run");
    }
    d
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h = (*h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
}

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

/// The call script (the one of `glm_tp.rs`): one digest per answering call, in call order.
fn script(m: &mut GlmModel) -> Vec<u64> {
    script_chunk(m, CHUNK)
}

/// [`script`] with the first prompt in chunks of `chunk` rows.
fn script_chunk(m: &mut GlmModel, chunk: usize) -> Vec<u64> {
    let mut out = Vec::new();
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let prompts = [(0usize, r.tokens(2300), chunk), (1, r.tokens(200), 128), (2, r.tokens(37), 128)];
    let mut pos = [0usize; SLOTS];
    let mut last = [0u32; SLOTS];
    for (slot, p, chunk) in &prompts {
        m.begin_slot(*slot).unwrap();
        let mut at = 0;
        while at < p.len() {
            let end = (at + chunk).min(p.len());
            let want = (end == p.len()).then_some(COUNT);
            if let Some(c) = m.prefill_chunk(*slot, at, &p[at..end], want).unwrap() {
                out.push(digest_call(m, &c, 1));
                last[*slot] = c.ids[0] as u32;
            }
            at = end;
        }
        pos[*slot] = p.len();
    }
    let reqs = vec![
        gb_engine::RowRequest { slot: 0, tokens: vec![last[0], 11, 12], first_position: pos[0] as u64 },
        gb_engine::RowRequest { slot: 1, tokens: vec![last[1]], first_position: pos[1] as u64 },
        gb_engine::RowRequest { slot: 2, tokens: vec![last[2], 21, 22, 23, 24], first_position: pos[2] as u64 },
    ];
    let b = RowBatch::from_requests(&reqs);
    let c = m.window_candidates(&b, COUNT).unwrap();
    out.push(digest_call(m, &c, b.rows));
    for (slot, keep) in [(0usize, 2usize), (1, 1), (2, 4)] {
        m.commit(slot, keep);
        pos[slot] += keep;
        let row = b.seq_rows(slot).start + keep - 1;
        last[slot] = c.ids[row * COUNT] as u32;
    }
    for _ in 0..6 {
        let reqs: Vec<gb_engine::RowRequest> =
            (0..SLOTS).map(|s| gb_engine::RowRequest { slot: s, tokens: vec![last[s]], first_position: pos[s] as u64 }).collect();
        let b = RowBatch::from_requests(&reqs);
        let c = m.window_candidates(&b, COUNT).unwrap();
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

/// Bytes of the loaded regions by kind: (global, trunk layers, expert regions).
fn region_bytes(plan: &gb_model::families::glm::LoadPlan) -> (u64, u64, u64) {
    let (mut g, mut t, mut e) = (0, 0, 0);
    for r in &plan.regions {
        match r.kind {
            RegionKind::Global => g += r.bytes,
            RegionKind::Trunk(_) => t += r.bytes,
            RegionKind::Experts(_) => e += r.bytes,
            RegionKind::Vision => {}
        }
    }
    (g, t, e)
}

/// A model of the first `layers` layers holding `own`'s routed experts (`None`: every expert, the single-node mapping). Returns the
/// model, the regions' bytes by kind and the device (bound) before the model takes it, through `inspect`.
fn load(
    dir: &Path,
    layers: usize,
    own: Option<&ExpertOwnership>,
    inspect: impl FnOnce(&GlmDevice),
) -> (GlmModel, (u64, u64, u64)) {
    gb_cuda::rt::init_context().unwrap();
    let pack = match own {
        Some(o) => GlmFamily::map_pack_owned(dir, o).unwrap(),
        None => GlmFamily::map_pack(dir).unwrap(),
    };
    let before = mem_available();
    let b = boot_mapped(dir, pack, Some(layers), &LoadOptions::from_env(20 * GIB), 0.0).unwrap();
    let rb = region_bytes(&b.plan);
    eprintln!(
        "loaded {layers} layers ({}): regions {:.3} GiB (global {:.3}, trunk {:.3}, experts {:.3}); MemAvailable fell {:.2} GiB",
        own.map_or("every expert".to_string(), |o| format!("experts {}..={}", o.owned()[0], o.owned()[o.local() - 1])),
        gib(rb.0 + rb.1 + rb.2),
        gib(rb.0),
        gib(rb.1),
        gib(rb.2),
        gib(before.saturating_sub(mem_available()))
    );
    inspect(&b.device);
    let m = GlmModel::new_chunked(
        b.pack.config.clone(),
        b.device,
        &ModelOptions { slots: SLOTS, kv_blocks: 64, max_context: 8192, layers },
        CHUNK,
    )
    .unwrap();
    (m, rb)
}

/// The MoE layers and how each bound: (layer, subbanks or 0 for a single bank, experts held).
fn moe_binding(dev: &GlmDevice) -> Vec<(usize, usize, usize)> {
    dev.layers
        .iter()
        .filter_map(|l| match &l.ffn {
            FfnDevice::Moe { .. } => Some((l.index, 0, E)),
            FfnDevice::MoeSubbanks { layer, .. } => {
                Some((l.index, layer.experts.banks().len(), layer.experts.ownership().local()))
            }
            FfnDevice::Dense(_) => None,
        })
        .collect()
}

/// The first MLA layer's plain-fp16 `wq_b` on the device against a host matmul of the pack's own matrix `[4096, 1536]` (fp64 accumulate):
/// relative L2 per row under 1e-3, and far from the transposed reading.
fn check_wq_b(dir: &Path, dev: &GlmDevice) {
    let (l, w) = dev
        .layers
        .iter()
        .find_map(|l| match &l.mixer {
            MixerDevice::Mla(m) => Some((l.index, &m.indexer.wq_b)),
            MixerDevice::Kda(_) => None,
        })
        .expect("an MLA layer in the prefix");
    let WqBDevice::F16 { k, n, .. } = w else {
        // the same test runs on a uniform pack (the 2.25bpw one) to prove its ranks' share-only load too
        eprintln!("layer {l}: wq_b is an EXL3 group in this pack: no plain matrix to check");
        return;
    };
    let (k, n) = (*k, *n);
    assert_eq!((k, n), (1536, 4096));
    let ck = gb_exl3::checkpoint::scan(dir, false).unwrap();
    let name = format!("model.language_model.layers.{l}.self_attn.indexer.wq_b.weight");
    let host_w: Vec<f64> = ck.read_u16(&ck.plain[&name]).unwrap().iter().map(|&h| f16_to_f64(h)).collect();
    assert_eq!(host_w.len(), n * k);
    let rows = 5;
    let mut r = Rng(77);
    let xh: Vec<u16> = (0..rows * k).map(|_| f64_to_f16((r.next() as f64 / 150_000.0 - 0.5) * 4.0)).collect();
    let x = DevBuf::from_slice(&xh).unwrap();
    let mut y = DevBuf::zeroed(rows * n * 2).unwrap();
    let mut ws = DevBuf::zeroed(256).unwrap();
    let st = gb_cuda::Stream::new().unwrap();
    w.forward(&x, &mut y, rows, &mut ws, &st, false).unwrap();
    st.sync().unwrap();
    let yd: Vec<f64> = y.to_vec::<u16>().unwrap().iter().map(|&h| f16_to_f64(h)).collect();
    for row in 0..rows {
        let xr: Vec<f64> = xh[row * k..(row + 1) * k].iter().map(|&h| f16_to_f64(h)).collect();
        let (mut num, mut den, mut num_t) = (0f64, 0f64, 0f64);
        for j in 0..n {
            let want: f64 = (0..k).map(|i| xr[i] * host_w[j * k + i]).sum();
            // the transposed reading of the same bytes ([1536, 4096]), over the first 1536 outputs only
            let trans: f64 = if j < k { (0..k).map(|i| xr[i] * host_w[i * n + j]).sum() } else { want };
            num += (yd[row * n + j] - want).powi(2);
            num_t += (yd[row * n + j] - trans).powi(2);
            den += want * want;
        }
        let (rel, rel_t) = ((num / den).sqrt(), (num_t / den).sqrt());
        eprintln!("layer {l} wq_b row {row}: rel L2 vs the host matmul {rel:.2e} (vs a transposed reading {rel_t:.2e})");
        assert!(rel < 1e-3 && rel_t > 1e-1, "layer {l} wq_b row {row}: rel {rel:.3e}, transposed {rel_t:.3e}");
    }
}

#[test]
#[ignore]
fn redsnow_prefix_two_ranks_equal_one_model_byte_for_byte() {
    let Some(dir) = redsnow_dir() else { return };
    let layers: usize = std::env::var("REXL3_REDSNOW_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(if layers <= 8 { 60 } else { 80 } * GIB);
    let guard = Guard::start(20 * GIB);

    // one model, every expert: mixed layers are subbanks run as pairs then combine (the single-node mapping of this pack)
    let t0 = std::time::Instant::now();
    let mut binding = Vec::new();
    let (mut a, rb_all) = load(&dir, layers, None, |dev| {
        check_wq_b(&dir, dev);
        binding = moe_binding(dev);
    });
    eprintln!("one model in {:.1?}: MoE layers (layer, subbanks or 0 = one bank, experts) {binding:?}", t0.elapsed());
    a.set_graphs(true);
    let ref_graphs = script(&mut a);
    a.set_graphs(false);
    let ref_eager = script(&mut a);
    assert_eq!(ref_graphs, ref_eager, "one model: graphs and eager differ on this prefix");
    eprintln!("one model: {} answering calls, graphs == eager", ref_graphs.len());
    drop(a);

    // two ranks, each loading only its share
    let owns = [ExpertOwnership::range(E, 0, 2).unwrap(), ExpertOwnership::range(E, 1, 2).unwrap()];
    let mut rb = Vec::new();
    let mut models = Vec::new();
    for o in &owns {
        let mut bind = Vec::new();
        let (m, b) = load(&dir, layers, Some(o), |dev| bind = moe_binding(dev));
        eprintln!("rank of {} experts: MoE layers {bind:?}", o.local());
        assert!(bind.iter().all(|&(_, s, n)| s > 0 && n == o.local()), "every MoE layer of a rank is subbanks of its share");
        rb.push(b);
        models.push(m);
    }
    assert_eq!(rb[0].0, rb_all.0, "the global region is replicated");
    assert_eq!(rb[0].1, rb_all.1, "the trunk regions are replicated");
    eprintln!(
        "expert regions: one model {:.3} GiB = rank 0 {:.3} + rank 1 {:.3} GiB (+ {:.0} KiB alignment)",
        gib(rb_all.2),
        gib(rb[0].2),
        gib(rb[1].2),
        (rb[0].2 + rb[1].2) as f64 / 1024.0 - rb_all.2 as f64 / 1024.0
    );
    let mut b = models.pop().unwrap();
    let mut a = models.pop().unwrap();
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(120)).into_iter();
    a.enable_tp(Arc::new(comms.next().unwrap()), owns[0].clone()).unwrap();
    b.enable_tp(Arc::new(comms.next().unwrap()), owns[1].clone()).unwrap();
    assert!(!a.graphs_on() && !b.graphs_on(), "a rank runs eagerly");
    let t = std::time::Instant::now();
    let (d0, d1, a, b) = std::thread::scope(|sc| {
        let h0 = sc.spawn(move || (script(&mut a), a));
        let h1 = sc.spawn(move || (script(&mut b), b));
        let (d0, a) = h0.join().expect("rank 0");
        let (d1, b) = h1.join().expect("rank 1");
        (d0, d1, a, b)
    });
    eprintln!("two ranks ran the script in {:.1?}", t.elapsed());
    assert_eq!(d0, d1, "the two ranks' answers differ");
    assert_eq!(d0, ref_graphs, "the group's answers differ from the one model's");
    let (t0r, t1r) = (a.tp().unwrap(), b.tp().unwrap());
    assert_eq!((t0r.exchanges, t0r.exchanged_bytes), (t1r.exchanges, t1r.exchanged_bytes), "the ranks exchanged alike");
    eprintln!(
        "BYTE-IDENTICAL: {} answering calls (candidates + every fp16 logit), {} exchanges of {:.1} MiB in all per rank",
        d0.len(),
        t0r.exchanges,
        t0r.exchanged_bytes as f64 / 1048576.0
    );
    drop((a, b));

    // the negative control: both ranks load and own 0..144 (144..288 computed by nobody, 0..144 counted twice): the answers must differ
    let (mut a, _) = load(&dir, layers, Some(&owns[0]), |_| {});
    let (mut b, _) = load(&dir, layers, Some(&owns[0]), |_| {});
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(120)).into_iter();
    a.enable_tp(Arc::new(comms.next().unwrap()), owns[0].clone()).unwrap();
    b.enable_tp(Arc::new(comms.next().unwrap()), owns[0].clone()).unwrap();
    let (n0, n1) = std::thread::scope(|sc| {
        let h0 = sc.spawn(|| script(&mut a));
        let h1 = sc.spawn(|| script(&mut b));
        (h0.join().expect("rank 0"), h1.join().expect("rank 1"))
    });
    assert_eq!(n0, n1, "the overlapping ranks still agree with each other");
    let differ = n0.iter().zip(&ref_graphs).filter(|(x, y)| x != y).count();
    assert!(differ > 0, "an ownership that does not partition the experts went unnoticed");
    eprintln!("negative control (both ranks own 0..144): {differ} of {} answering calls differ from the one model", n0.len());
    drop((a, b));
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

#[test]
#[ignore]
fn p2_redsnow_prefix_two_ranks_equal_one_p2_model_byte_for_byte() {
    let Some(dir) = redsnow_dir() else { return };
    let layers: usize = std::env::var("REXL3_REDSNOW_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(if layers <= 8 { 60 } else { 80 } * GIB);
    let guard = Guard::start(20 * GIB);
    let (mut a, _) = load(&dir, layers, None, |_| {});
    a.set_prefill_contract(PrefillContract::P1).unwrap();
    a.set_graphs(false);
    let p1_ref = script(&mut a);
    a.set_prefill_contract(PrefillContract::P2).unwrap();
    a.set_graphs(true);
    let ref_graphs = script(&mut a);
    a.set_graphs(false);
    let ref_eager = script(&mut a);
    assert_eq!(ref_graphs, ref_eager, "one P2 model: graphs and eager differ on this prefix");
    let moved = ref_eager.iter().zip(&p1_ref).filter(|(x, y)| x != y).count();
    assert!(moved > 0, "P2 left every answer at P1's bits: the P2 path did not run");
    // the first prompt in 336-row chunks as well: a row split of 3 slabs is uneven (256 / 80 rows), the last chunk's 284 rows 256 / 28
    let ref_336 = script_chunk(&mut a, 336);
    eprintln!("one P2 model: {} answering calls, graphs == eager, {moved} differ from P1's (P2 ran)", ref_eager.len());
    drop(a);

    let owns = [ExpertOwnership::range(E, 0, 2).unwrap(), ExpertOwnership::range(E, 1, 2).unwrap()];
    let mut models = Vec::new();
    for o in &owns {
        let (mut m, _) = load(&dir, layers, Some(o), |_| {});
        m.set_prefill_contract(PrefillContract::P2).unwrap();
        models.push(m);
    }
    let mut b = models.pop().unwrap();
    let mut a = models.pop().unwrap();
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(120)).into_iter();
    a.enable_tp(Arc::new(comms.next().unwrap()), owns[0].clone()).unwrap();
    b.enable_tp(Arc::new(comms.next().unwrap()), owns[1].clone()).unwrap();
    let t = std::time::Instant::now();
    let (d0, d1, e0, e1, a, b) = std::thread::scope(|sc| {
        let h0 = sc.spawn(move || (script(&mut a), script_chunk(&mut a, 336), a));
        let h1 = sc.spawn(move || (script(&mut b), script_chunk(&mut b, 336), b));
        let (d0, e0, a) = h0.join().expect("rank 0");
        let (d1, e1, b) = h1.join().expect("rank 1");
        (d0, d1, e0, e1, a, b)
    });
    eprintln!("two P2 ranks ran the script twice in {:.1?}", t.elapsed());
    assert_eq!(d0, d1, "the two P2 ranks' answers differ");
    assert_eq!(d0, ref_eager, "the P2 group's answers differ from the one P2 model's");
    assert_eq!((&e0, &e1), (&ref_336, &ref_336), "336-row chunks: the P2 group's answers differ from the one P2 model's");
    let (t0r, t1r) = (a.tp().unwrap(), b.tp().unwrap());
    assert!(t0r.p2_swaps() > 0 && t0r.p2_swaps() == t1r.p2_swaps(), "part swaps {} / {}", t0r.p2_swaps(), t1r.p2_swaps());
    assert_eq!((t0r.exchanges, t0r.exchanged_bytes), (t1r.exchanges, t1r.exchanged_bytes), "the ranks exchanged alike");
    eprintln!(
        "P2 BYTE-IDENTICAL: {} answering calls == the one P2 model; {} part swaps, {} exchanges of {:.1} MiB in all per rank",
        d0.len(),
        t0r.p2_swaps(),
        t0r.exchanges,
        t0r.exchanged_bytes as f64 / 1048576.0
    );
    drop((a, b));
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// The SERVING rank load (`tp_boot::plan_from` -> `tp_boot::load_rank`, what `build_leader` and `follow` run): two ranks of a layer prefix,
/// each on its own thread with a loopback communicator, each mapping and allocating only its share, the allocation check against the plan,
/// the ownership-vs-place check and `enable_tp` inside `load_rank`; then the script on both against one model of every expert. A rank
/// mapped for a share with no communicator is refused by name before any load.
#[test]
#[ignore]
fn the_serving_rank_load_takes_only_its_share_and_equals_one_model() {
    use gb_model::families::glm::tp_boot::{load_rank, plan_from};
    use gb_model::families::glm::GlmConfig;
    use gb_model::serve::{PrefixCacheOption, ServeOptions};
    let Some(dir) = redsnow_dir() else { return };
    let layers: usize = std::env::var("REXL3_REDSNOW_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(60 * GIB);
    let guard = Guard::start(20 * GIB);
    let (mut one, _) = load(&dir, layers, None, |_| {});
    one.set_graphs(true);
    let reference = script(&mut one);
    drop(one);

    let cfg = GlmConfig::from_dir(&dir).unwrap();
    let mut o = ServeOptions::new(dir.clone());
    (o.slots, o.slots_auto, o.max_model_len, o.kv_tokens, o.prefill_chunk) = (SLOTS, false, 8192, Some(16384), CHUNK);
    o.prefix_cache = PrefixCacheOption::Off;
    o.vision = None;
    let plans: Vec<_> = (0..2)
        .map(|r| {
            let ck = gb_exl3::checkpoint::scan(&dir, false).unwrap();
            plan_from(&o, cfg.clone(), ck, 117 * GIB, 0.0, Some(ExpertOwnership::range(E, r, 2).unwrap())).unwrap()
        })
        .collect();
    // a share with no communicator: refused by name, nothing loaded
    let mo = ModelOptions { slots: SLOTS, kv_blocks: 64, max_context: 8192, layers };
    // load_rank checks the model's allocations against the plan of `cfg`'s layers: a prefix model is checked against a prefix config
    let mut pcfg = cfg.clone();
    pcfg.layers.truncate(layers);
    let e = load_rank(&dir, &pcfg, plans[0].pack.clone(), plans[0].own.as_ref(), &mo, plans[0].chunk, 20 * GIB, 0.0, None)
        .err()
        .expect("a share with no communicator must be refused")
        .to_string();
    assert!(e.contains("share") && e.contains("communicator"), "{e}");
    let comms: Vec<_> = LoopbackGroup::with_timeout(2, Duration::from_secs(120)).into_iter().collect();
    let t = std::time::Instant::now();
    let digests: Vec<Vec<u64>> = std::thread::scope(|sc| {
        let hs: Vec<_> = plans
            .iter()
            .zip(comms)
            .map(|(p, c)| {
                let (dir, cfg, mo) = (&dir, &pcfg, &mo);
                sc.spawn(move || {
                    let mut rank =
                        load_rank(dir, cfg, p.pack.clone(), p.own.as_ref(), mo, p.chunk, 20 * GIB, 0.0, Some(Arc::new(c)))
                            .unwrap_or_else(|e| panic!("load_rank: {e}"));
                    assert_eq!(rank.m.tp().unwrap().ownership(), p.own.as_ref().unwrap());
                    script(&mut rank.m)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("rank")).collect()
    });
    eprintln!("two serving rank loads and the script in {:.1?}", t.elapsed());
    assert_eq!(digests[0], digests[1], "the ranks differ");
    assert_eq!(digests[0], reference, "the serving rank load differs from one model");
    eprintln!("BYTE-IDENTICAL through tp_boot::load_rank: {} answering calls", reference.len());
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// A model of the first `layers` layers of rank `rank` of two under the expert WIDTH split (every routed expert at the rank's columns:
/// `GlmFamily::map_pack_width`), with the regions' bytes by kind.
fn load_width(dir: &Path, layers: usize, rank: usize) -> (GlmModel, (u64, u64, u64)) {
    gb_cuda::rt::init_context().unwrap();
    let pack = GlmFamily::map_pack_width(dir, rank, 2).unwrap();
    let b = boot_mapped(dir, pack, Some(layers), &LoadOptions::from_env(20 * GIB), 0.0).unwrap();
    let rb = region_bytes(&b.plan);
    let bind = moe_binding(&b.device);
    assert!(bind.iter().all(|&(_, s, n)| s > 0 && n == E), "every MoE layer of a width rank is subbanks of every expert: {bind:?}");
    eprintln!(
        "width rank {rank}: {layers} layers, regions {:.3} GiB (global {:.3}, trunk {:.3}, experts {:.3}); MoE layers {bind:?}",
        gib(rb.0 + rb.1 + rb.2),
        gib(rb.0),
        gib(rb.1),
        gib(rb.2)
    );
    let m = GlmModel::new_chunked(
        b.pack.config.clone(),
        b.device,
        &ModelOptions { slots: SLOTS, kv_blocks: 64, max_context: 8192, layers },
        CHUNK,
    )
    .unwrap();
    (m, rb)
}

/// The expert WIDTH split through the SERVING rank load (`tp_boot::plan_from_split` with the rank's width config and every expert ->
/// `tp_boot::load_rank`, the ownership and width checks and `enable_tp` inside it): two ranks on a loopback group, each holding its half
/// of every expert's width, must answer exactly as one model of every expert (P1: the decode windows' and the batched chunks' bits are
/// the single node's). The two ranks' expert regions are about half the whole map's each (the replicated model-width scales aside), and
/// a rank given its peer's width share is refused by name. A rank's contract is the process's: run with `REXL3_GLM_PREFILL_CONTRACT=p1`
/// (under P2 the width ranks' prompt bits are not one node's: `p2_width_split_ranks_agree_repeat_and_are_chunk_invariant`).
#[test]
#[ignore]
fn width_split_ranks_equal_one_model_byte_for_byte() {
    use gb_model::families::glm::tp_boot::{load_rank, plan_from_split};
    use gb_model::families::glm::GlmConfig;
    use gb_model::serve::{PrefixCacheOption, ServeOptions};
    let Some(dir) = redsnow_dir() else { return };
    let contract = gb_model::families::glm::contract::current();
    assert_eq!(contract, PrefillContract::P1, "run with REXL3_GLM_PREFILL_CONTRACT=p1: a rank's contract is the process's");
    let layers: usize = std::env::var("REXL3_REDSNOW_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(if layers <= 8 { 60 } else { 80 } * GIB);
    let guard = Guard::start(20 * GIB);
    let (mut one, rb_all) = load(&dir, layers, None, |_| {});
    one.set_graphs(true);
    let reference = script(&mut one);
    drop(one);

    let cfg = GlmConfig::from_dir(&dir).unwrap();
    let mut o = ServeOptions::new(dir.clone());
    (o.slots, o.slots_auto, o.max_model_len, o.kv_tokens, o.prefill_chunk) = (SLOTS, false, 8192, Some(16384), CHUNK);
    o.prefix_cache = PrefixCacheOption::Off;
    o.vision = None;
    let all = ExpertOwnership::all(E);
    let cfgs: Vec<GlmConfig> = (0..2).map(|r| cfg.with_expert_width(r, 2).unwrap()).collect();
    let plans: Vec<_> = cfgs
        .iter()
        .map(|c| {
            let ck = gb_exl3::checkpoint::scan(&dir, false).unwrap();
            plan_from_split(&o, c.clone(), ck, 117 * GIB, 0.0, Some(all.clone()), None).unwrap()
        })
        .collect();
    // the planner refuses a width config with a whole-expert share
    let ck = gb_exl3::checkpoint::scan(&dir, false).unwrap();
    let e = plan_from_split(&o, cfgs[0].clone(), ck, 117 * GIB, 0.0, Some(ExpertOwnership::range(E, 0, 2).unwrap()), None)
        .err()
        .expect("a width rank with a range share must be refused")
        .to_string();
    assert!(e.contains("width"), "{e}");
    eprintln!(
        "planned weights: rank 0 {:.3} GiB, rank 1 {:.3} GiB (whole pack)",
        gib(plans[0].weight_bytes),
        gib(plans[1].weight_bytes)
    );
    let mo = ModelOptions { slots: SLOTS, kv_blocks: 64, max_context: 8192, layers };
    let pcfgs: Vec<GlmConfig> = cfgs
        .iter()
        .map(|c| {
            let mut p = c.clone();
            p.layers.truncate(layers);
            p
        })
        .collect();
    // ranks 0 and 1 of a loopback group, each loading its width share through load_rank
    let comms: Vec<_> = LoopbackGroup::with_timeout(2, Duration::from_secs(120)).into_iter().collect();
    let t = std::time::Instant::now();
    let digests: Vec<(Vec<u64>, (u64, u64))> = std::thread::scope(|sc| {
        let hs: Vec<_> = plans
            .iter()
            .zip(&pcfgs)
            .zip(comms)
            .map(|((p, pc), c)| {
                let (dir, mo) = (&dir, &mo);
                sc.spawn(move || {
                    let mut rank =
                        load_rank(dir, pc, p.pack.clone(), p.own.as_ref(), mo, p.chunk, 20 * GIB, 0.0, Some(Arc::new(c)))
                            .unwrap_or_else(|e| panic!("load_rank: {e}"));
                    let tp = rank.m.tp().unwrap();
                    assert!(tp.ownership().is_all() && tp.width() == pc.ffn.width);
                    let d = script(&mut rank.m);
                    let tp = rank.m.tp().unwrap();
                    (d, (tp.exchanges, tp.exchanged_bytes))
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("rank")).collect()
    });
    eprintln!("two width ranks loaded and ran the script in {:.1?}", t.elapsed());
    assert_eq!(digests[0].1, digests[1].1, "the ranks exchanged alike");
    assert_eq!(digests[0].0, digests[1].0, "the width ranks' answers differ");
    assert_eq!(digests[0].0, reference, "the width ranks' answers differ from the one model's");
    eprintln!(
        "WIDTH SPLIT BYTE-IDENTICAL: {} answering calls == one model; {} exchanges of {:.1} MiB per rank",
        reference.len(),
        digests[0].1 .0,
        digests[0].1 .1 as f64 / 1048576.0
    );

    // the regions of a width rank, and a rank given its peer's share
    let (m0, rb0) = load_width(&dir, layers, 0);
    let (mut m1, rb1) = load_width(&dir, layers, 0);
    assert_eq!((rb0.0, rb0.1), (rb_all.0, rb_all.1), "the global and trunk regions are replicated");
    eprintln!(
        "expert regions: one model {:.3} GiB, a width rank {:.3} GiB ({:.1}% of the whole: half the trellises, the model-width scales whole)",
        gib(rb_all.2),
        gib(rb0.2),
        100.0 * rb0.2 as f64 / rb_all.2 as f64
    );
    assert!(rb0.2 * 2 >= rb_all.2 && rb0.2 * 2 < rb_all.2 + rb_all.2 / 20, "a width rank holds about half the experts' bytes");
    assert_eq!(rb1.2, rb0.2);
    drop(m0);
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(120)).into_iter();
    let _rank0 = comms.next().unwrap();
    let e = m1.enable_tp(Arc::new(comms.next().unwrap()), all.clone()).unwrap_err().to_string();
    assert!(e.contains("width share"), "{e}");
    eprintln!("rank 1 holding rank 0's width share refuses: {e}");
    drop(m1);
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// The expert WIDTH split under the P2 prefill contract: each rank's P2 expert GEMMs on its half of every expert, its part over every
/// slot (`CombineMode::All`), the part swap, the `Parts` sum. The parts are the width's halves (not the whole-expert split's groups), so
/// the bits are not one P2 model's; the ranks must agree, run P2 (differ from P1), repeat bit for bit, and answer the first prompt
/// alike in 512- and 336-row chunks whenever one P2 model does.
#[test]
#[ignore]
fn p2_width_split_ranks_agree_repeat_and_are_chunk_invariant() {
    let Some(dir) = redsnow_dir() else { return };
    let layers: usize = std::env::var("REXL3_REDSNOW_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(if layers <= 8 { 60 } else { 80 } * GIB);
    let guard = Guard::start(20 * GIB);
    let (mut a, _) = load(&dir, layers, None, |_| {});
    a.set_prefill_contract(PrefillContract::P1).unwrap();
    a.set_graphs(false);
    let p1_ref = script(&mut a);
    a.set_prefill_contract(PrefillContract::P2).unwrap();
    let p2_one = script(&mut a);
    let p2_one_336 = script_chunk(&mut a, 336);
    drop(a);
    let one_invariant = p2_one == p2_one_336;
    eprintln!("one P2 model: 512- and 336-row chunks {}", if one_invariant { "agree" } else { "differ" });

    let mut models = Vec::new();
    for r in 0..2 {
        let (mut m, _) = load_width(&dir, layers, r);
        m.set_prefill_contract(PrefillContract::P2).unwrap();
        models.push(m);
    }
    let mut b = models.pop().unwrap();
    let mut a = models.pop().unwrap();
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(120)).into_iter();
    a.enable_tp(Arc::new(comms.next().unwrap()), ExpertOwnership::all(E)).unwrap();
    b.enable_tp(Arc::new(comms.next().unwrap()), ExpertOwnership::all(E)).unwrap();
    let t = std::time::Instant::now();
    let run = |m: &mut GlmModel| (script(m), script(m), script_chunk(m, 336));
    let ((d0, r0, c0), (d1, r1, c1), a, b) = std::thread::scope(|sc| {
        let h0 = sc.spawn(move || (run(&mut a), a));
        let h1 = sc.spawn(move || (run(&mut b), b));
        let (x, a) = h0.join().expect("rank 0");
        let (y, b) = h1.join().expect("rank 1");
        (x, y, a, b)
    });
    eprintln!("two P2 width ranks ran the script three times in {:.1?}", t.elapsed());
    assert_eq!((&d0, &c0), (&d1, &c1), "the two P2 width ranks' answers differ");
    assert_eq!((&r0, &r1), (&d0, &d0), "a P2 width rank does not repeat its answers bit for bit");
    let moved = d0.iter().zip(&p1_ref).filter(|(x, y)| x != y).count();
    assert!(moved > 0, "the P2 width ranks left every answer at P1's bits: P2 did not run");
    if one_invariant {
        assert_eq!(c0, d0, "the P2 width ranks answer differently in 336-row chunks");
    }
    let vs_one = d0.iter().zip(&p2_one).filter(|(x, y)| x != y).count();
    let (t0r, t1r) = (a.tp().unwrap(), b.tp().unwrap());
    assert!(t0r.p2_swaps() > 0 && t0r.p2_swaps() == t1r.p2_swaps(), "part swaps {} / {}", t0r.p2_swaps(), t1r.p2_swaps());
    eprintln!(
        "P2 WIDTH: {} answering calls, ranks agree, repeat, {moved} differ from P1 (P2 ran), {vs_one} differ from one P2 model (the parts \
         are the width's halves); {} part swaps a rank",
        d0.len(),
        t0r.p2_swaps()
    );
    drop((a, b));
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// A rank's map of the pack WITH the NextN layer: the layer binds as subbanks of the rank's share; outside a group a prefill (which runs
/// the NextN layer) refuses the share by name (`MTP_SHARE`) instead of computing half of it; `enable_tp` accepts it (lane L12: the layer's
/// pairs are exchanged, `docs/GLM_TP.md` 9; the two-rank proof is `tests/glm_tp_mtp.rs`).
#[test]
#[ignore]
#[allow(clippy::err_expect)] // as the GLM TP lane wrote it (a84a4a05); `expect_err` would be the same check
fn a_ranks_nextn_layer_binds_as_subbanks_and_is_refused_by_name_where_it_would_run() {
    use gb_model::families::glm::device::MtpMoe;
    use gb_model::families::glm::load::boot_mapped_with;
    use gb_model::families::glm::model::MTP_SHARE;
    use gb_model::families::nemotron_h_omni::tp::SingleRank;
    let Some(dir) = redsnow_dir() else { return };
    require_available(40 * GIB);
    let guard = Guard::start(20 * GIB);
    gb_cuda::rt::init_context().unwrap();
    let own = ExpertOwnership::range(E, 0, 2).unwrap();
    let pack = GlmFamily::map_pack_owned(&dir, &own).unwrap();
    let layers = 4;
    let b = boot_mapped_with(&dir, pack, Some(layers), true, &LoadOptions::from_env(20 * GIB), 0.0).unwrap();
    match &b.device.mtp.as_ref().expect("the NextN layer was loaded").moe {
        MtpMoe::Subbanks(l) => {
            assert_eq!(l.experts.ownership(), &own);
            eprintln!("the NextN layer of rank 0: {} subbanks of {} experts", l.experts.banks().len(), own.local());
        }
        MtpMoe::Single(_) => panic!("a rank's NextN layer must bind as subbanks of its share"),
    }
    let mut m = GlmModel::new_chunked(
        b.pack.config.clone(),
        b.device,
        &ModelOptions { slots: SLOTS, kv_blocks: 64, max_context: 8192, layers },
        CHUNK,
    )
    .unwrap();
    m.begin_slot(0).unwrap();
    let toks: Vec<u32> = (0..37).map(|i| 1000 + i).collect();
    let e = m.prefill_chunk(0, 0, &toks, Some(COUNT)).err().expect("the NextN layer's share must be refused").to_string();
    assert!(e.contains(MTP_SHARE), "{e}");
    eprintln!("a prefill outside a group refuses: {e}");
    m.end_slot(0);
    let other = ExpertOwnership::range(E, 1, 2).unwrap();
    let e = m.enable_tp(Arc::new(SingleRank), other).unwrap_err().to_string();
    eprintln!("enable_tp with another rank's share refuses: {e}");
    m.enable_tp(Arc::new(SingleRank), own.clone()).expect("a rank's NextN share is accepted (lane L12)");
    drop(m);
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// The DFlash2 drafter of `REXL3_DFLASH_DIR` attached to `m` tapping `taps` (layers of the prefix: the pack's own taps reach past it), the
/// whole drafter or `share` of a split one (attached after `enable_tp`: its rounds swap with the peer).
fn attach_drafter(m: &mut GlmModel, dir: &Path, taps: &[usize], share: Option<gb_model::draft::DraftShare>) {
    use gb_model::draft::{Draft, BLOCK};
    let dims = gb_model::shapes::read_draft_config(dir).unwrap().dims(gb_model::families::glm::alloc::TOKENIZER_VOCAB);
    let pack = gb_model::pack::Pack::open(dir).unwrap();
    let mut ctx = gb_model::loadctx::LoadCtx::from_env();
    let rows = gb_model::families::glm::dflash::ctx_rows(m.chunk_rows());
    let (draft, _) = Draft::declare_bf16(&mut ctx, &pack, dims, m.cfg.vocab, SLOTS, rows, BLOCK, share).unwrap();
    ctx.finish().unwrap();
    m.attach_dflash(draft, taps).unwrap();
}

/// FNV-1a 64 of a slice of u16.
fn fnv16(v: &[u16]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in v {
        fnv(&mut h, &x.to_le_bytes());
    }
    h
}

/// The first prompt of [`script`] in `chunk`-row chunks, the digest of its answer and, with a drafter, the digest of the drafter's tap
/// rows after every chunk.
fn prompt_taps(m: &mut GlmModel, chunk: usize) -> (u64, Vec<u64>) {
    let p = Rng(0x9e37_79b9_7f4a_7c15).tokens(2300);
    m.begin_slot(0).unwrap();
    let (mut at, mut taps, mut answer) = (0, Vec::new(), 0);
    while at < p.len() {
        let end = (at + chunk).min(p.len());
        let want = (end == p.len()).then_some(COUNT);
        let c = m.prefill_chunk(0, at, &p[at..end], want).unwrap();
        if m.has_dflash() {
            taps.push(fnv16(&m.dflash_taps_host(end - at).unwrap()));
        }
        if let Some(c) = c {
            answer = digest_call(m, &c, 1);
        }
        at = end;
    }
    m.end_slot(0);
    (answer, taps)
}

/// The sequence-parallel residual of a P2 prompt chunk (`GlmModel::run`'s `sp`, `REXL3_GLM_P2_SP` on): two P2 ranks under the serving split
/// `kda,lm,mla,sh` keep only their rows of the mHC streams in every chunk of two 128-row slabs or more (512-row chunks; 336-row chunks
/// cut 256 / 80, the 2,300-token prompt's last 252 rows run unsplit). Both ranks' answers equal ONE P2 model's byte for byte. With a
/// DFlash2 drafter (`REXL3_DFLASH_DIR`: on rank 0 and on the one model, tapping five layers of the prefix; both ranks relay), rank 0's
/// tap rows after every chunk equal the one model's, and the one model's drafts after a long prompt are the same with and without the
/// backend naming the prompt's end (the rows more than a ring before it neither tapped nor absorbed).
#[test]
#[ignore]
fn p2_redsnow_sequence_parallel_two_ranks_equal_one_p2_model_and_relay_the_taps() {
    use gb_model::families::glm::model::tp::split_from_vars;
    let Some(dir) = redsnow_dir() else { return };
    assert!(gb_model::families::glm::contract::p2_sp_on(), "REXL3_GLM_P2_SP is off: nothing to test");
    let layers: usize = std::env::var("REXL3_REDSNOW_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let draft_dir = std::env::var("REXL3_DFLASH_DIR").ok().map(PathBuf::from).filter(|p| p.join("config.json").exists());
    let taps: Vec<usize> = [1, 3, 4, 5, 7].iter().map(|&l| l.min(layers - 1)).collect();
    require_available(if layers <= 8 { 60 } else { 80 } * GIB);
    let guard = Guard::start(20 * GIB);
    let (mut a, _) = load(&dir, layers, None, |_| {});
    a.set_prefill_contract(PrefillContract::P2).unwrap();
    a.set_graphs(false);
    if let Some(d) = &draft_dir {
        attach_drafter(&mut a, d, &taps, None);
    }
    let reference = script(&mut a);
    let ref_336 = script_chunk(&mut a, 336);
    let ref_taps = prompt_taps(&mut a, CHUNK);
    let ref_taps_336 = prompt_taps(&mut a, 336);
    if draft_dir.is_some() {
        // the drafts after a 4,000-token prompt: every row absorbed (no prompt end named) == only the rows that reach the ring
        let p = Rng(0x51_7cc1_b727_220a).tokens(4000);
        let mut picks = Vec::new();
        for (slot, end) in [(0usize, None), (1, Some(p.len()))] {
            a.begin_slot(slot).unwrap();
            a.dflash_prompt_end(slot, end, None);
            let mut at = 0;
            let mut last = 0;
            while at < p.len() {
                let e = (at + CHUNK).min(p.len());
                let want = (e == p.len()).then_some(COUNT);
                if let Some(c) = a.prefill_chunk(slot, at, &p[at..e], want).unwrap() {
                    last = c.ids[0] as u32;
                }
                at = e;
            }
            let mut toks = p.clone();
            toks.push(last);
            let s = gb_sampling::Sampling { temperature: 0.0, ..Default::default() };
            let ask = gb_model::families::glm::model::DflashAsk { slot, tokens: &toks, sampling: &s, want: 7 };
            picks.push(a.dflash_draft(&[ask]).unwrap().remove(0).tokens);
        }
        let skipped = a.dflash_mut().unwrap().skipped;
        assert!(skipped >= 4000 - 2560, "the rows more than a ring before the prompt's end were absorbed ({skipped} skipped)");
        assert_eq!(picks[0], picks[1], "the drafts differ when the prompt's end is named");
        eprintln!("one P2 model: the drafts after 4,000 rows are {:?} with and without the prompt's end ({skipped} rows skipped)", picks[0]);
        for s in 0..2 {
            a.end_slot(s);
        }
    }
    eprintln!("one P2 model: {} answering calls, {} tap digests", reference.len(), ref_taps.1.len());
    drop(a);

    let bits = split_from_vars(|_| Some("kda,lm,mla,sh".into())).unwrap();
    let owns = [ExpertOwnership::range(E, 0, 2).unwrap(), ExpertOwnership::range(E, 1, 2).unwrap()];
    let mut models = Vec::new();
    for o in &owns {
        let (mut m, _) = load(&dir, layers, Some(o), |_| {});
        m.set_prefill_contract(PrefillContract::P2).unwrap();
        models.push(m);
    }
    let mut b = models.pop().unwrap();
    let mut a = models.pop().unwrap();
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(120)).into_iter();
    a.enable_tp(Arc::new(comms.next().unwrap()), owns[0].clone()).unwrap();
    b.enable_tp(Arc::new(comms.next().unwrap()), owns[1].clone()).unwrap();
    a.enable_head_split(bits).unwrap();
    b.enable_head_split(bits).unwrap();
    if let Some(d) = &draft_dir {
        attach_drafter(&mut a, d, &taps, None);
        a.enable_tap_relay(&taps).unwrap();
        b.enable_tap_relay(&taps).unwrap();
    }
    let t = std::time::Instant::now();
    let run = |m: &mut GlmModel| (script(m), script_chunk(m, 336), prompt_taps(m, CHUNK), prompt_taps(m, 336));
    let (r0, r1, a, b) = std::thread::scope(|sc| {
        let h0 = sc.spawn(move || (run(&mut a), a));
        let h1 = sc.spawn(move || (run(&mut b), b));
        let (r0, a) = h0.join().expect("rank 0");
        let (r1, b) = h1.join().expect("rank 1");
        (r0, r1, a, b)
    });
    eprintln!("two P2 ranks (split kda,lm,mla,sh) ran the script in {:.1?}", t.elapsed());
    assert!(a.sp_chunks() > 0 && a.sp_chunks() == b.sp_chunks(), "sp chunks {} / {}", a.sp_chunks(), b.sp_chunks());
    assert_eq!((&r0.0, &r1.0), (&reference, &reference), "512-row chunks: the ranks' answers differ from the one P2 model's");
    assert_eq!((&r0.1, &r1.1), (&ref_336, &ref_336), "336-row chunks: the ranks' answers differ from the one P2 model's");
    assert_eq!((r0.2 .0, r1.2 .0, r0.3 .0, r1.3 .0), (ref_taps.0, ref_taps.0, ref_taps_336.0, ref_taps_336.0), "the prompt's answers");
    assert_eq!((&r0.2 .1, &r0.3 .1), (&ref_taps.1, &ref_taps_336.1), "rank 0's tap rows differ from the one model's");
    let (t0r, t1r) = (a.tp().unwrap(), b.tp().unwrap());
    assert_eq!((t0r.exchanges, t0r.exchanged_bytes), (t1r.exchanges, t1r.exchanged_bytes), "the ranks exchanged alike");
    eprintln!(
        "P2 SEQUENCE-PARALLEL BYTE-IDENTICAL: {} + {} answering calls == the one P2 model, {} tap digests; {} sp chunks, {} exchanges of \
         {:.1} MiB a rank",
        reference.len(),
        ref_336.len(),
        ref_taps.1.len() + ref_taps_336.1.len(),
        a.sp_chunks(),
        t0r.exchanges,
        t0r.exchanged_bytes as f64 / 1048576.0
    );
    drop((a, b));
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// The digest of row `i` of a prompt forward's candidates and of its fp16 logits (a forward of several prompts answers one row each).
fn digest_row(m: &GlmModel, c: &gb_model::forward::Candidates, i: usize) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let k = c.count;
    for &x in &c.ids[i * k..(i + 1) * k] {
        fnv(&mut h, &x.to_le_bytes());
    }
    for &v in &c.vals[i * k..(i + 1) * k] {
        fnv(&mut h, &v.to_bits().to_le_bytes());
    }
    for x in m.logits_host(i, 1).unwrap() {
        fnv(&mut h, &x.to_le_bytes());
    }
    h
}

/// The slots' sequences while a script runs: the tokens fed so far and the next one (the greedy pick of the last answer).
struct Seqs {
    fed: Vec<Vec<u32>>,
    next: Vec<u32>,
}

impl Seqs {
    fn new() -> Self {
        Seqs { fed: vec![Vec::new(); SLOTS], next: vec![0; SLOTS] }
    }

    /// A prompt slice of `slot` ran; with `c`, it ended the prompt and row `i` of `c` is its answer.
    fn prompt(&mut self, slot: usize, tokens: &[u32], c: Option<(&gb_model::forward::Candidates, usize)>) {
        self.fed[slot].extend_from_slice(tokens);
        if let Some((c, i)) = c {
            self.next[slot] = c.ids[i * c.count] as u32;
        }
    }
}

/// One window of `rows` (each `(slot, drafts, keep)`: the slot fed its next token then `drafts`, its first `keep` rows committed, the
/// next token the pick after the last kept row): the window's digest.
fn window(m: &mut GlmModel, sq: &mut Seqs, rows: &[(usize, Vec<u32>, usize)]) -> u64 {
    let reqs: Vec<gb_engine::RowRequest> = rows
        .iter()
        .map(|(s, drafts, _)| {
            let tokens = std::iter::once(sq.next[*s]).chain(drafts.iter().copied()).collect();
            gb_engine::RowRequest { slot: *s, tokens, first_position: sq.fed[*s].len() as u64 }
        })
        .collect();
    let b = RowBatch::from_requests(&reqs);
    let c = m.window_candidates(&b, COUNT).unwrap();
    let d = digest_call(m, &c, b.rows);
    for (i, (s, _, keep)) in rows.iter().enumerate() {
        m.commit(*s, *keep);
        sq.fed[*s].extend_from_slice(&reqs[i].tokens[..*keep]);
        sq.next[*s] = c.ids[(b.seq_rows(i).start + keep - 1) * COUNT] as u32;
    }
    d
}

/// `n` greedy one-row windows of every slot.
fn greedy_windows(m: &mut GlmModel, sq: &mut Seqs, n: usize) -> Vec<u64> {
    (0..n).map(|_| window(m, sq, &(0..SLOTS).map(|s| (s, Vec::new(), 1)).collect::<Vec<_>>())).collect()
}

/// Each slot's seven greedy DFlash2 drafts from where it stands, its last window's commits applied first (`dflash_draft` reads the
/// committed context; none without a drafter, or when `with` is off: a rank of a group drafts through the group's calls, not alone).
fn slot_drafts(m: &mut GlmModel, sq: &Seqs, with: bool) -> Vec<Vec<u32>> {
    if !with || !m.has_dflash() {
        return Vec::new();
    }
    m.flush().unwrap();
    let s = gb_sampling::Sampling { temperature: 0.0, ..Default::default() };
    (0..SLOTS)
        .map(|slot| {
            let toks: Vec<u32> = sq.fed[slot].iter().copied().chain(std::iter::once(sq.next[slot])).collect();
            let ask = gb_model::families::glm::model::DflashAsk { slot, tokens: &toks, sampling: &s, want: 7 };
            m.dflash_draft(&[ask]).unwrap().remove(0).tokens
        })
        .collect()
}

/// One prompt slice of `slot`: `tokens` at positions from `at`.
fn slice_of(slot: usize, at: usize, tokens: &[u32]) -> RowBatch {
    RowBatch::from_requests(&[gb_engine::RowRequest { slot, tokens: tokens.to_vec(), first_position: at as u64 }])
}

/// [`batch_script`]'s prompts: slot 0's 230 tokens and slot 1's 37, slot 2's 600 (a 400-row slice alone, then its last 200 rows).
fn batch_prompts() -> Vec<Vec<u32>> {
    let mut r = Rng(0x2545_f491_4f6c_dd1d);
    vec![r.tokens(230), r.tokens(37), r.tokens(600)]
}

/// Slot 2's first 400 rows alone, then the last slices of the three prompts (467 rows: slot 2's a continuation of the segment its first
/// slice opened) `together` in ONE prompt forward (`GlmModel::prefill_rows` of the three sequences), or each in its own forward; then four
/// greedy windows of the three. One digest per prompt's answer (its row of the candidates and of the logits) and per window, in call
/// order, and (with `drafts`) each slot's drafts after.
fn batch_script(m: &mut GlmModel, together: bool, drafts: bool) -> (Vec<u64>, Vec<Vec<u32>>) {
    let p = batch_prompts();
    let mut sq = Seqs::new();
    let mut out = Vec::new();
    for s in 0..SLOTS {
        m.begin_slot(s).unwrap();
    }
    assert!(m.prefill_chunk(2, 0, &p[2][..400], None).unwrap().is_none());
    sq.prompt(2, &p[2][..400], None);
    let tails = [(0usize, 0usize), (1, 0), (2, 400)];
    if together {
        let reqs: Vec<gb_engine::RowRequest> = tails
            .iter()
            .map(|&(s, at)| gb_engine::RowRequest { slot: s, tokens: p[s][at..].to_vec(), first_position: at as u64 })
            .collect();
        let b = RowBatch::from_requests(&reqs);
        let c = m.prefill_rows(&b, Some(COUNT), usize::MAX).unwrap().expect("the forward's candidates");
        assert_eq!(c.rows, tails.len(), "one answer per prompt");
        for (i, &(s, at)) in tails.iter().enumerate() {
            out.push(digest_row(m, &c, i));
            sq.prompt(s, &p[s][at..], Some((&c, i)));
        }
    } else {
        for &(s, at) in &tails {
            let c = m.prefill_chunk(s, at, &p[s][at..], Some(COUNT)).unwrap().expect("the slice's candidates");
            out.push(digest_row(m, &c, 0));
            sq.prompt(s, &p[s][at..], Some((&c, 0)));
        }
    }
    out.extend(greedy_windows(m, &mut sq, 4));
    let d = slot_drafts(m, &sq, drafts);
    m.flush().unwrap();
    for s in 0..SLOTS {
        m.end_slot(s);
    }
    (out, d)
}

/// The rows of the stash of [`pause_script`]'s models: less than the 512-row slice paused, so the slice has rows past the stash, and
/// more than slot 2's prompt, which runs (whole or in parts of its own) while the slice is paused.
const PART_ROOM: usize = 192;

/// [`pause_script`]'s prompts: slot 0's 200 tokens, slot 1's 900 (its first 512-row slice the forward that pauses), slot 2's 160 (more
/// than a window's 128 rows: under P2 a row-split forward on two ranks, as a short prompt's in a server is).
fn pause_prompts() -> Vec<Vec<u32>> {
    let mut r = Rng(0x6a09_e667_f3bc_c908);
    vec![r.tokens(200), r.tokens(900), r.tokens(160)]
}

/// Slot 0's prompt; slot 1's first 512-row slice, paused at the first of `parts` (layer bounds, ascending; none: the slice whole); a
/// verify window of slot 0 (4 rows, 2 kept); resumed to the next bound; slot 2's whole prompt, itself paused at `inner`'s bounds (none:
/// whole) with a window of slot 0 between its parts (both forwards paused while it runs: a short prompt in parts on top of a long one;
/// the window after slot 2's prompt when it runs whole); resumed to the next; a window of slots 0 and 2; the slice's remaining parts; then
/// the rest of slot 1's prompt and three greedy windows of the three. Every answering call's digest in call order (the slice does not
/// end its prompt: it answers nothing wherever it pauses), and (with `drafts`) each slot's drafts after. While paused, a forward of more
/// rows than the stash and one on the paused slot are refused, and a third forward in parts while two are paused.
fn pause_script(m: &mut GlmModel, parts: &[usize], inner: &[usize], drafts: bool) -> (Vec<u64>, Vec<Vec<u32>>) {
    let p = pause_prompts();
    let mut sq = Seqs::new();
    let mut out = Vec::new();
    m.begin_slot(0).unwrap();
    let c = m.prefill_chunk(0, 0, &p[0], Some(COUNT)).unwrap().expect("slot 0's candidates");
    out.push(digest_row(m, &c, 0));
    sq.prompt(0, &p[0], Some((&c, 0)));
    m.begin_slot(1).unwrap();
    let mut bounds = parts.iter().copied();
    let until = bounds.next().unwrap_or(usize::MAX);
    assert!(m.prefill_rows(&slice_of(1, 0, &p[1][..CHUNK]), None, until).unwrap().is_none());
    sq.prompt(1, &p[1][..CHUNK], None);
    if !parts.is_empty() {
        assert_eq!(m.prefill_paused().map(|b| b.seq_slot.clone()), Some(vec![1]), "the slice is paused");
        let room = m.prefill_part_rows();
        assert!(m.prefill_rows(&slice_of(2, 0, &p[1][..room + 1]), None, usize::MAX).is_err(), "a forward past the stash");
        assert!(m.prefill_rows(&slice_of(2, 0, &p[1][..room + 1]), None, 1).is_err(), "a forward in parts past the stash");
        assert!(m.prefill_rows(&slice_of(1, CHUNK, &p[1][CHUNK..CHUNK + 16]), None, usize::MAX).is_err(), "a forward on the paused slot");
    }
    out.push(window(m, &mut sq, &[(0, vec![11, 12, 13], 2)]));
    if let Some(k) = bounds.next() {
        assert!(m.prefill_resume(k).unwrap().is_none());
    }
    m.begin_slot(2).unwrap();
    // slot 2's answer is digested as soon as it comes back (the digest reads the logits, which the next window overwrites), and pushed
    // after the window either way
    let (c, d) = match inner.split_first() {
        None => {
            let c = m.prefill_chunk(2, 0, &p[2], Some(COUNT)).unwrap().expect("slot 2's candidates");
            let d = digest_row(m, &c, 0);
            out.push(window(m, &mut sq, &[(0, Vec::new(), 1)]));
            (c, d)
        }
        Some((&first, rest)) => {
            assert!(m.prefill_rows(&slice_of(2, 0, &p[2]), Some(COUNT), first).unwrap().is_none());
            assert_eq!(m.prefill_paused().map(|b| b.seq_slot.clone()), Some(vec![2]), "slot 2's forward goes on first");
            if !parts.is_empty() {
                assert!(m.prefill_rows(&slice_of(0, 200, &p[0][..16]), None, 1).is_err(), "a third forward in parts");
            }
            out.push(window(m, &mut sq, &[(0, Vec::new(), 1)]));
            for &k in rest {
                assert!(m.prefill_resume(k).unwrap().is_none());
            }
            let c = m.prefill_resume(usize::MAX).unwrap().expect("slot 2's candidates");
            let d = digest_row(m, &c, 0);
            (c, d)
        }
    };
    out.push(d);
    sq.prompt(2, &p[2], Some((&c, 0)));
    if let Some(k) = bounds.next() {
        assert!(m.prefill_resume(k).unwrap().is_none());
    }
    out.push(window(m, &mut sq, &[(0, vec![21], 1), (2, vec![31, 32], 2)]));
    for k in bounds {
        assert!(m.prefill_resume(k).unwrap().is_none());
    }
    if !parts.is_empty() {
        assert!(m.prefill_resume(usize::MAX).unwrap().is_none(), "the slice's last part answers nothing (it does not end the prompt)");
        assert!(m.prefill_paused().is_none());
    }
    let c = m.prefill_chunk(1, CHUNK, &p[1][CHUNK..], Some(COUNT)).unwrap().expect("slot 1's candidates");
    out.push(digest_row(m, &c, 0));
    sq.prompt(1, &p[1][CHUNK..], Some((&c, 0)));
    out.extend(greedy_windows(m, &mut sq, 3));
    let d = slot_drafts(m, &sq, drafts);
    m.flush().unwrap();
    for s in 0..SLOTS {
        m.end_slot(s);
    }
    (out, d)
}

/// The layers of the prefix (`REXL3_REDSNOW_TP_LAYERS`, default 8), the DFlash2 drafter's directory (`REXL3_DFLASH_DIR`, optional) and
/// the layers it taps (five of the prefix, as the sequence-parallel test).
fn prefix_setup() -> (usize, Option<PathBuf>, Vec<usize>) {
    let layers: usize = std::env::var("REXL3_REDSNOW_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let draft = std::env::var("REXL3_DFLASH_DIR").ok().map(PathBuf::from).filter(|p| p.join("config.json").exists());
    let taps = [1, 3, 4, 5, 7].iter().map(|&l: &usize| l.min(layers - 1)).collect();
    (layers, draft, taps)
}

/// One model of every expert of the prefix under `contract`, drafting with `draft` when given, with the stash of `room` rows (0: none).
fn one_model(contract: PrefillContract, dir: &Path, layers: usize, draft: Option<&Path>, taps: &[usize], room: usize) -> GlmModel {
    let (mut m, _) = load(dir, layers, None, |_| {});
    m.set_prefill_contract(contract).unwrap();
    if let Some(d) = draft {
        attach_drafter(&mut m, d, taps, None);
    }
    if room > 0 {
        assert_eq!(m.enable_prefill_parts(room).map(|_| m.prefill_part_rows()).unwrap(), room);
    }
    m
}

/// The two ranks of the prefix under `contract` and the serving split (`kda,lm,mla,sh`; under P2 the sequence-parallel residual of a
/// long prompt forward) on a loopback group, rank 0 drafting with `draft` when given (both ranks relaying its taps), each with the stash
/// of `room` rows (0: none).
fn two_ranks(
    contract: PrefillContract,
    dir: &Path,
    layers: usize,
    draft: Option<&Path>,
    taps: &[usize],
    room: usize,
) -> (GlmModel, GlmModel) {
    use gb_model::families::glm::model::tp::split_from_vars;
    let p2 = contract == PrefillContract::P2;
    assert!(!p2 || gb_model::families::glm::contract::p2_sp_on(), "REXL3_GLM_P2_SP is off: the serving split's residual would not be tested");
    let bits = split_from_vars(|_| Some("kda,lm,mla,sh".into())).unwrap();
    let owns = [ExpertOwnership::range(E, 0, 2).unwrap(), ExpertOwnership::range(E, 1, 2).unwrap()];
    let mut ms: Vec<GlmModel> = owns
        .iter()
        .map(|o| {
            let (mut m, _) = load(dir, layers, Some(o), |_| {});
            m.set_prefill_contract(contract).unwrap();
            m
        })
        .collect();
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(120)).into_iter();
    for (m, o) in ms.iter_mut().zip(&owns) {
        m.enable_tp(Arc::new(comms.next().unwrap()), o.clone()).unwrap();
        m.enable_head_split(bits).unwrap();
    }
    if let Some(d) = draft {
        attach_drafter(&mut ms[0], d, taps, None);
        for m in ms.iter_mut() {
            m.enable_tap_relay(taps).unwrap();
        }
    }
    if room > 0 {
        for m in ms.iter_mut() {
            m.enable_prefill_parts(room).unwrap();
        }
    }
    let b = ms.pop().unwrap();
    (ms.pop().unwrap(), b)
}

/// `f` on both ranks at once, each on its own thread: the two results and the ranks back.
fn on_ranks<T: Send>(a: GlmModel, b: GlmModel, f: impl Fn(&mut GlmModel) -> T + Sync) -> (T, T, GlmModel, GlmModel) {
    let f = &f;
    std::thread::scope(|sc| {
        let h0 = sc.spawn(move || {
            let mut a = a;
            (f(&mut a), a)
        });
        let h1 = sc.spawn(move || {
            let mut b = b;
            (f(&mut b), b)
        });
        let (r0, a) = h0.join().expect("rank 0");
        let (r1, b) = h1.join().expect("rank 1");
        (r0, r1, a, b)
    })
}

/// Several prompts in ONE prompt forward under `contract` (`model/part.rs`): three prompts' last slices (one continuing its prompt) give
/// each prompt the answer, the state (the windows after) and, with a drafter (`REXL3_DFLASH_DIR`), the drafter context (the drafts after)
/// of its own forward, on one model (twice alike) and on the two ranks of the serving split (under P2 the 467-row forward is row-split
/// across the ranks: slot 1's 37 rows straddle the split).
fn several_prompts_in_one_forward(contract: PrefillContract) {
    let Some(dir) = redsnow_dir() else { return };
    let (layers, draft, taps) = prefix_setup();
    require_available(if layers <= 8 { 60 } else { 80 } * GIB);
    let guard = Guard::start(20 * GIB);
    let mut a = one_model(contract, &dir, layers, draft.as_deref(), &taps, 0);
    let solo = batch_script(&mut a, false, true);
    let together = batch_script(&mut a, true, true);
    assert_eq!(batch_script(&mut a, true, true), together, "one {contract:?} model: the forward of three prompts twice");
    assert_eq!(together.0, solo.0, "one {contract:?} model: an answer (or a window after) differs from the prompt's own forward");
    assert_eq!(together.1, solo.1, "one {contract:?} model: the drafts after differ from the prompts' own forwards'");
    eprintln!("one {contract:?} model: {} answering calls and {} slots' drafts equal to each prompt's own forward", solo.0.len(), solo.1.len());
    drop(a);

    let (a, b) = two_ranks(contract, &dir, layers, draft.as_deref(), &taps, 0);
    let (r0, r1, a, b) = on_ranks(a, b, |m| batch_script(m, true, false).0);
    assert_eq!(a.sp_chunks(), b.sp_chunks());
    assert_eq!(a.sp_chunks() > 0, contract == PrefillContract::P2, "sp chunks {} (P2's alone)", a.sp_chunks());
    assert_eq!((&r0, &r1), (&solo.0, &solo.0), "two {contract:?} ranks: the forward of three prompts differs from each prompt's own forward");
    eprintln!(
        "SEVERAL PROMPTS IN ONE FORWARD BYTE-IDENTICAL ({contract:?}): {} answering calls on one model and on two ranks ({} sp forwards)",
        solo.0.len(),
        a.sp_chunks()
    );
    drop((a, b));
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

#[test]
#[ignore]
fn p2_several_prompts_in_one_forward_equal_each_prompts_solo_forward() {
    several_prompts_in_one_forward(PrefillContract::P2);
}

#[test]
#[ignore]
fn p1_several_prompts_in_one_forward_equal_each_prompts_solo_forward() {
    several_prompts_in_one_forward(PrefillContract::P1);
}

/// A prompt forward in parts under `contract` (`model/part.rs`): slot 1's 512-row slice paused at one, two and three layer bounds, with
/// a verify window of slot 0, slot 2's whole 160-row prompt and windows run between its parts (the stash holds 192 rows: the slice's
/// later rows are kept where they are), and slot 2's prompt itself in parts of its own on top of it (two forwards paused, a window
/// between) or alone, gives every call the bits of the run with everything whole, and the same drafts after, on one model and on the two
/// ranks of the serving split (under P2 the slice and slot 2's prompt row-split across them).
fn a_forward_paused_equals_the_whole_forward(contract: PrefillContract) {
    let Some(dir) = redsnow_dir() else { return };
    let (layers, draft, taps) = prefix_setup();
    assert!(layers >= 4, "parts need a few layers");
    require_available(if layers <= 8 { 60 } else { 80 } * GIB);
    let guard = Guard::start(20 * GIB);
    // (the slice's bounds, slot 2's bounds)
    let cases: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![layers / 2], vec![]),
        (vec![2, layers - 3], vec![]),
        (vec![1, layers / 2, layers - 1], vec![]),
        (vec![], vec![layers / 2]),
        (vec![2, layers - 3], vec![layers / 2]),
        (vec![1, layers / 2, layers - 1], vec![1, layers - 1]),
    ];
    let mut a = one_model(contract, &dir, layers, draft.as_deref(), &taps, PART_ROOM);
    let whole = pause_script(&mut a, &[], &[], true);
    let differs: Vec<String> = cases
        .iter()
        .filter_map(|(parts, inner)| {
            let got = pause_script(&mut a, parts, inner, true);
            let call = got.0.iter().zip(&whole.0).position(|(x, y)| x != y);
            (got != whole).then(|| format!("{parts:?}/{inner:?}: first differing call {call:?}, drafts equal {}", got.1 == whole.1))
        })
        .collect();
    assert!(differs.is_empty(), "one {contract:?} model: the slice paused / slot 2's prompt paused differ from everything whole: {differs:?}");
    eprintln!("one {contract:?} model: paused at {cases:?} == whole ({} answering calls, {} slots' drafts)", whole.0.len(), whole.1.len());
    drop(a);

    let (a, b) = two_ranks(contract, &dir, layers, draft.as_deref(), &taps, PART_ROOM);
    let all: Vec<(Vec<usize>, Vec<usize>)> = std::iter::once((Vec::new(), Vec::new())).chain(cases.iter().cloned()).collect();
    let (r0, r1, a, b) = on_ranks(a, b, |m| all.iter().map(|(p, q)| pause_script(m, p, q, false).0).collect::<Vec<_>>());
    assert_eq!(a.sp_chunks(), b.sp_chunks());
    assert_eq!(a.sp_chunks() > 0, contract == PrefillContract::P2, "sp chunks {} (P2's alone)", a.sp_chunks());
    for (i, (parts, inner)) in all.iter().enumerate() {
        assert_eq!(
            (&r0[i], &r1[i]),
            (&whole.0, &whole.0),
            "two {contract:?} ranks: the slice paused at {parts:?}, slot 2's prompt at {inner:?}, differs from the one model's whole"
        );
    }
    eprintln!(
        "A FORWARD IN PARTS BYTE-IDENTICAL ({contract:?}): paused at {cases:?} (the slice's bounds, slot 2's) with windows and a prompt \
         between == whole, one model and two ranks ({} sp forwards)",
        a.sp_chunks()
    );
    drop((a, b));
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

#[test]
#[ignore]
fn p2_a_forward_paused_with_windows_and_a_prompt_between_its_parts_equals_the_whole_forward() {
    a_forward_paused_equals_the_whole_forward(PrefillContract::P2);
}

#[test]
#[ignore]
fn p1_a_forward_paused_with_windows_and_a_prompt_between_its_parts_equals_the_whole_forward() {
    a_forward_paused_equals_the_whole_forward(PrefillContract::P1);
}

/// [`cache_script`]'s prompts: A of 700 tokens, B of 224 (its end on the 16-row grid) and the 32 tokens of B's follow-up turn.
fn cache_prompts() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut r = Rng(0x5be0_cd19_137e_2179);
    (r.tokens(700), r.tokens(224), r.tokens(32))
}

/// [`cache_script`]'s cut of A: on the 16-row grid, past two whole pages (the entry's last page is partial: copied at the restore).
const CUT_A: usize = 528;

/// A snapshot of `slot` after `len` tokens as every rank takes one (`tp::take_snapshot`): the snapshot slot and the pages it holds.
fn take_snapshot(m: &mut GlmModel, slot: usize, len: usize) -> (u32, Vec<u32>) {
    let s = m.alloc_snapshot().expect("a free snapshot slot");
    m.prefix_snapshot(slot, s, len).unwrap();
    let pages = m.retain_prefix_pages(slot, len);
    m.seal_snapshot(s, &pages).unwrap();
    (s, pages)
}

/// Every slot's drafts from where it stands (none on a model without the drafter), the windows' commits applied on every rank first.
fn drafts_now(m: &mut GlmModel, sq: &Seqs) -> Vec<Vec<u32>> {
    m.flush().unwrap();
    slot_drafts(m, sq, true)
}

/// The prefix cache's moves as the serving group makes them (`GlmTpBackend`: snapshots between forwards, the batched call with a restored
/// slot, the end cut, a follow-up turn's slice in parts), `cached`, or the same requests uncached in their own chunks:
/// * slot 0, the donor: A in 512-row chunks (cached: one more chunk boundary at the cut, 528, and a snapshot there), answered, then a
///   window of its own (the donor writes on in the page the entry shares);
/// * slot 1, A again: cached, restored at 528 after the donor went on; uncached, its first 512 rows alone. Its rest goes through ONE
///   prompt forward with slot 2's whole B, each answered; cached, B's end cut (224) is a snapshot after that forward;
/// * four greedy windows of the three slots, and the drafts after;
/// * slot 2's follow-up B + C: cached, restored from B's end cut, its last 32 rows; uncached, all 256 from 0. A prompt forward paused at
///   layer `bound` with a window of slots 0 and 1 between its parts, answered; three greedy windows, the drafts after.
///
/// One digest per answering call in call order (the prompt's row of the candidates and its fp16 logits), and every slot's drafts after
/// the windows (rank 0's, or one model's, drafter). Every entry is released at the end and every slot ended.
fn cache_script(m: &mut GlmModel, cached: bool, bound: usize) -> (Vec<u64>, Vec<Vec<u32>>) {
    let (a, b, c) = cache_prompts();
    let mut sq = Seqs::new();
    let mut out = Vec::new();
    let mut entries = Vec::new();
    // slot 0, the donor; the drafter keeps the window before the cut (`GlmModel::dflash_prompt_end`, as the backend names it)
    m.begin_slot(0).unwrap();
    m.dflash_prompt_end(0, Some(a.len()), cached.then_some(CUT_A));
    let ends: &[usize] = if cached { &[CHUNK, CUT_A, 700] } else { &[CHUNK, 700] };
    let mut at = 0;
    for &end in ends {
        let want = (end == a.len()).then_some(COUNT);
        let cands = m.prefill_chunk(0, at, &a[at..end], want).unwrap();
        if let Some(cands) = &cands {
            out.push(digest_row(m, cands, 0));
        }
        sq.prompt(0, &a[at..end], cands.as_ref().map(|x| (x, 0)));
        if cached && end == CUT_A {
            entries.push(take_snapshot(m, 0, CUT_A));
        }
        at = end;
    }
    out.push(window(m, &mut sq, &[(0, Vec::new(), 1)]));
    // slot 1, A again, its rest in one forward with slot 2's B
    let from = if cached {
        let (s, pages) = &entries[0];
        m.restore_prefix(1, *s, pages, CUT_A).unwrap();
        CUT_A
    } else {
        m.begin_slot(1).unwrap();
        m.dflash_prompt_end(1, Some(a.len()), None);
        assert!(m.prefill_chunk(1, 0, &a[..CHUNK], None).unwrap().is_none());
        CHUNK
    };
    sq.prompt(1, &a[..from], None);
    m.dflash_prompt_end(1, Some(a.len()), None);
    m.begin_slot(2).unwrap();
    m.dflash_prompt_end(2, Some(b.len()), cached.then_some(b.len()));
    let reqs = [
        gb_engine::RowRequest { slot: 1, tokens: a[from..].to_vec(), first_position: from as u64 },
        gb_engine::RowRequest { slot: 2, tokens: b.clone(), first_position: 0 },
    ];
    let cands = m.prefill_rows(&RowBatch::from_requests(&reqs), Some(COUNT), usize::MAX).unwrap().expect("the forward's candidates");
    assert_eq!(cands.rows, 2, "one answer per prompt");
    for (i, (s, toks)) in [(1usize, &a[from..]), (2, &b[..])].into_iter().enumerate() {
        out.push(digest_row(m, &cands, i));
        sq.prompt(s, toks, Some((&cands, i)));
    }
    if cached {
        entries.push(take_snapshot(m, 2, b.len()));
    }
    out.extend(greedy_windows(m, &mut sq, 4));
    let mut drafts = drafts_now(m, &sq);
    // slot 2's follow-up turn, a forward in parts with a window of the other two between them
    let bc: Vec<u32> = b.iter().chain(&c).copied().collect();
    m.end_slot(2);
    let from = if cached {
        let (s, pages) = &entries[1];
        m.restore_prefix(2, *s, pages, b.len()).unwrap();
        b.len()
    } else {
        m.begin_slot(2).unwrap();
        0
    };
    sq.fed[2] = bc[..from].to_vec();
    m.dflash_prompt_end(2, Some(bc.len()), None);
    assert!(m.prefill_rows(&slice_of(2, from, &bc[from..]), Some(COUNT), bound).unwrap().is_none(), "the follow-up pauses");
    out.push(window(m, &mut sq, &[(0, Vec::new(), 1), (1, Vec::new(), 1)]));
    let cands = m.prefill_resume(usize::MAX).unwrap().expect("the follow-up's candidates");
    out.push(digest_row(m, &cands, 0));
    sq.prompt(2, &bc[from..], Some((&cands, 0)));
    out.extend(greedy_windows(m, &mut sq, 3));
    drafts.extend(drafts_now(m, &sq));
    for s in 0..SLOTS {
        m.end_slot(s);
    }
    for (s, pages) in entries {
        m.release_prefix(s, &pages);
    }
    (out, drafts)
}

/// The two ranks of the serving group (`tp_boot::plan_from_split` and `load_rank_with`'s moves on the test's load): the expert WIDTH
/// split (every expert at the rank's columns), the head split of `REXL3_GLM_TP_SPLIT` (default all five; under the KDA split the rank is
/// built from its heads' strips, so its state and its snapshot slots are its heads'), P2 with the sequence-parallel residual, drafting
/// with `draft` (rank 0's whole drafter, or with `split` each rank's half of it; both ranks relaying its taps), the stash of `room` rows
/// and four snapshot slots on each rank.
fn serving_ranks(dir: &Path, layers: usize, draft: Option<&Path>, taps: &[usize], room: usize, split: bool) -> (GlmModel, GlmModel) {
    let mo = ModelOptions { slots: SLOTS, kv_blocks: 64, max_context: 8192, layers };
    serving_ranks_at(dir, layers, draft, taps, room, split, CHUNK, mo)
}

/// [`serving_ranks`] with scratch for `chunk`-row prompt chunks, at `mo` (slots, pages, context).
#[allow(clippy::too_many_arguments)]
fn serving_ranks_at(
    dir: &Path,
    layers: usize,
    draft: Option<&Path>,
    taps: &[usize],
    room: usize,
    split: bool,
    chunk: usize,
    mo: ModelOptions,
) -> (GlmModel, GlmModel) {
    use gb_model::families::glm::{GlmConfig, GlmPack};
    assert!(gb_model::families::glm::contract::p2_sp_on(), "REXL3_GLM_P2_SP is off: the serving residual would not be tested");
    let bits = gb_model::families::glm::model::tp::split_from_env().unwrap();
    let mut ms: Vec<GlmModel> = (0..2)
        .map(|r| {
            gb_cuda::rt::init_context().unwrap();
            let mut cfg = GlmConfig::from_dir(dir).unwrap().with_expert_width(r, 2).unwrap();
            if bits & gb_model::families::glm::model::tp::SPLIT_KDA != 0 {
                cfg = cfg.with_kda_rank(r, 2).unwrap();
            }
            let ck = gb_exl3::checkpoint::scan(dir, false).unwrap();
            let pack = GlmPack::from_checkpoint_with(&ck, cfg, true).unwrap();
            let b = boot_mapped(dir, pack, Some(layers), &LoadOptions::from_env(20 * GIB), 0.0).unwrap();
            let mut m = GlmModel::new_chunked(b.pack.config.clone(), b.device, &mo, chunk).unwrap();
            m.set_prefill_contract(PrefillContract::P2).unwrap();
            m
        })
        .collect();
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(120)).into_iter();
    for m in ms.iter_mut() {
        m.enable_tp(Arc::new(comms.next().unwrap()), ExpertOwnership::all(E)).unwrap();
        m.enable_head_split(bits).unwrap();
    }
    if let Some(d) = draft {
        for (r, m) in ms.iter_mut().enumerate() {
            if split {
                attach_drafter(m, d, taps, Some(gb_model::draft::DraftShare::of(r, 2).unwrap()));
            } else if r == 0 {
                attach_drafter(m, d, taps, None);
            }
            m.enable_tap_relay(taps).unwrap();
        }
    }
    for m in ms.iter_mut() {
        m.keep_tapped_rows(taps);
        m.enable_prefill_parts(room).unwrap();
        m.enable_prefix_cache(4).unwrap();
    }
    let b = ms.pop().unwrap();
    (ms.pop().unwrap(), b)
}

/// The widest P2 chunk the serving group takes (`alloc::P2_MAX_PREFILL_CHUNK`).
const WIDE: usize = 8192;

/// The script of [`p2_serving_ranks_answer_alike_in_chunks_of_8192_6144_4096_and_512_rows`]: slot 0's 10,000-token prompt (past one
/// 8,192-row chunk), slot 1's 37 and slot 2's 600, each in `chunk`-row chunks and answered (the drafter told each prompt's end, as the
/// backend does), four greedy windows of the three slots, and every slot's drafts after. One digest per answering call (the prompt's row
/// of the candidates and its fp16 logits) and per window, in call order, and the drafts (none without a drafter).
fn wide_script(m: &mut GlmModel, chunk: usize) -> (Vec<u64>, Vec<Vec<u32>>) {
    let mut r = Rng(0x6a09_e667_f3bc_c908);
    let prompts = [r.tokens(10_000), r.tokens(37), r.tokens(600)];
    let mut sq = Seqs::new();
    let mut out = Vec::new();
    for (slot, p) in prompts.iter().enumerate() {
        m.begin_slot(slot).unwrap();
        m.dflash_prompt_end(slot, Some(p.len()), None);
        let mut at = 0;
        while at < p.len() {
            let end = (at + chunk).min(p.len());
            let want = (end == p.len()).then_some(COUNT);
            let c = m.prefill_chunk(slot, at, &p[at..end], want).unwrap();
            if let Some(c) = &c {
                out.push(digest_row(m, c, 0));
            }
            sq.prompt(slot, &p[at..end], c.as_ref().map(|x| (x, 0)));
            at = end;
        }
    }
    out.extend(greedy_windows(m, &mut sq, 4));
    let drafts = drafts_now(m, &sq);
    for s in 0..SLOTS {
        m.end_slot(s);
    }
    (out, drafts)
}

/// The P2 chunk cap at 8,192 rows (`alloc::P2_MAX_PREFILL_CHUNK`, `moe_prefill::MAX_ROWS`): the two ranks of the serving group
/// ([`serving_ranks`]: the expert width split, the head split, the sequence-parallel residual, the DFlash2 drafter of `REXL3_DFLASH_DIR`
/// with its taps relayed, split unless `REXL3_GLM_DRAFT_SPLIT=0`), built for 8,192-row chunks, give every answer, every window after and every
/// draft of [`wide_script`] alike in chunks of 8,192, 6,144, 4,096 and 512 rows: a P2 row's bits are its own, whatever chunk it is in,
/// so the wider chunks move no bit (the 8,192-row chunk runs the routed experts, the P2 GEMMs, the exchanges and the drafter's absorb
/// at 8,192 rows, 4,096 a rank's own).
///
///     flock -x /tmp/rexl3-gpu.lock env REXL3_REDSNOW_DIR=... REXL3_DFLASH_DIR=... REXL3_GLM_KV=fp8 cargo test --release -p gb-model \
///         --test glm_redsnow_tp p2_serving_ranks_answer_alike -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn p2_serving_ranks_answer_alike_in_chunks_of_8192_6144_4096_and_512_rows() {
    let Some(dir) = redsnow_dir() else { return };
    let (layers, draft, taps) = prefix_setup();
    require_available(if layers <= 8 { 60 } else { 80 } * GIB);
    let guard = Guard::start(20 * GIB);
    let split = gb_model::families::glm::dflash::split_on() && draft.is_some();
    let mo = ModelOptions { slots: SLOTS, kv_blocks: 96, max_context: 16384, layers };
    let (a, b) = serving_ranks_at(&dir, layers, draft.as_deref(), &taps, PART_ROOM, split, WIDE, mo);
    assert_eq!((a.chunk_rows(), b.chunk_rows()), (WIDE, WIDE));
    let chunks = [WIDE, 6144, 4096, 512];
    let t = std::time::Instant::now();
    let (r0, r1, a, b) = on_ranks(a, b, |m| chunks.map(|c| wide_script(m, c)));
    eprintln!("two serving ranks ran the script in chunks of {chunks:?} in {:.1?}", t.elapsed());
    for (i, &c) in chunks.iter().enumerate() {
        assert_eq!(r1[i].0, r0[i].0, "chunks of {c}: the two ranks' answers differ");
        if split {
            assert_eq!(r1[i].1, r0[i].1, "chunks of {c}: the split drafter's halves drafted differently on the two ranks");
        }
        assert_eq!(r0[i].0, r0[0].0, "chunks of {c}: an answer (or a window after) differs from 8,192-row chunks'");
        assert_eq!(r0[i].1, r0[0].1, "chunks of {c}: the drafts differ from 8,192-row chunks'");
    }
    assert!(a.sp_chunks() > 0 && a.sp_chunks() == b.sp_chunks(), "sp chunks {} / {}", a.sp_chunks(), b.sp_chunks());
    eprintln!(
        "P2 CHUNKS OF 8,192 / 6,144 / 4,096 / 512 ROWS BYTE-IDENTICAL on the serving ranks: {} answering calls and windows, {} slots' drafts \
         ({}), {} sp chunks",
        r0[0].0.len(),
        r0[0].1.len(),
        if draft.is_none() { "no drafter" } else if split { "split drafter" } else { "rank 0's drafter" },
        a.sp_chunks()
    );
    drop((a, b));
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// The prefix cache under the serving contract (`--prefix-cache on`, P2 with the sequence-parallel residual, the latent cache of
/// `REXL3_GLM_KV`, the DFlash2 drafter of `REXL3_DFLASH_DIR`): a prompt restored from a snapshot and prefilled on, inside a forward of
/// several prompts and inside a forward in parts, then decoded, answers every call and drafts exactly as the same requests uncached
/// ([`cache_script`]), on one P2 model of every expert and on the two ranks of the serving group (whose bits are not one model's under
/// the width split: each run is checked against its own uncached run, and the ranks against each other).
///
///     flock -x /tmp/rexl3-gpu.lock env REXL3_REDSNOW_DIR=... REXL3_DFLASH_DIR=... REXL3_GLM_KV=fp8 cargo test --release -p gb-model \
///         --test glm_redsnow_tp p2_a_restored_prefix -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn p2_a_restored_prefix_in_a_batched_and_a_paused_forward_equals_uncached_on_one_model_and_the_serving_ranks() {
    let Some(dir) = redsnow_dir() else { return };
    let (layers, draft, taps) = prefix_setup();
    assert!(layers >= 4, "parts need a few layers");
    require_available(if layers <= 8 { 60 } else { 80 } * GIB);
    let guard = Guard::start(20 * GIB);
    let bound = layers / 2;
    let mut one = one_model(PrefillContract::P2, &dir, layers, draft.as_deref(), &taps, PART_ROOM);
    one.enable_prefix_cache(4).unwrap();
    let kv = one.cfg.latent_kv.name().to_string();
    let cold = cache_script(&mut one, false, bound);
    let warm = cache_script(&mut one, true, bound);
    assert_eq!(warm.0, cold.0, "one P2 model: a cached answer (or a window after) differs from the uncached run");
    assert_eq!(warm.1, cold.1, "one P2 model: the drafts after a restore differ from the uncached run's");
    assert_eq!(cache_script(&mut one, true, bound), warm, "one P2 model: the cached run twice");
    eprintln!("one P2 model ({kv} latent cache): {} answering calls and {} drafts, cached == uncached", cold.0.len(), cold.1.len());
    drop(one);

    let split = gb_model::families::glm::dflash::split_on() && draft.is_some();
    let (a, b) = serving_ranks(&dir, layers, draft.as_deref(), &taps, PART_ROOM, split);
    let (r0, r1, a, b) = on_ranks(a, b, |m| (cache_script(m, false, bound), cache_script(m, true, bound)));
    assert_eq!((&r0.0 .0, &r0.1 .0), (&r1.0 .0, &r1.1 .0), "the two ranks' answers differ");
    assert_eq!(r0.1 .0, r0.0 .0, "two ranks: a cached answer (or a window after) differs from the uncached run");
    assert_eq!(r0.1 .1, r0.0 .1, "rank 0: the drafts after a restore differ from the uncached run's");
    if split {
        assert_eq!((&r1.0 .1, &r1.1 .1), (&r0.0 .1, &r0.1 .1), "the split drafter's halves drafted differently on the two ranks");
    }
    assert!(a.sp_chunks() > 0 && a.sp_chunks() == b.sp_chunks(), "sp chunks {} / {}", a.sp_chunks(), b.sp_chunks());
    eprintln!(
        "A RESTORED PREFIX BYTE-IDENTICAL TO UNCACHED (P2, {kv}, the drafter {}): {} answering calls and {} drafts on one model and on \
         two width ranks ({} sp forwards); in a forward of two prompts, in a forward in parts, at an end cut",
        if draft.is_some() { "on" } else { "off" },
        r0.0 .0.len(),
        r0.0 .1.len(),
        a.sp_chunks()
    );
    drop((a, b));
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// [`inner_cut_script`]'s prompts: A of 700 tokens (its end cut 688 on the 16-row grid, inside its last 512-row chunk), B of 230 (end
/// cut 224), and the 32 and 20 tokens of B's two follow-up turns.
fn inner_cut_prompts() -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut r = Rng(0x7a3c_51e9_04b2_d68f);
    (r.tokens(700), r.tokens(230), r.tokens(32), r.tokens(20))
}

/// The snapshot armed at `len` in `slot` completed after the forward that passed it, as every rank completes one (`tp::take_snapshot`
/// with `GlmModel::finish_cut`): the snapshot slot and the pages it holds.
fn take_inner(m: &mut GlmModel, slot: usize, len: usize) -> (u32, Vec<u32>) {
    use gb_model::families::glm::model::snapshot::FinishedCut;
    let snap = match m.finish_cut(slot, len).unwrap() {
        FinishedCut::Taken(s) => s,
        other => panic!("the snapshot armed at {len} of slot {slot}: {other:?}"),
    };
    let pages = m.retain_prefix_pages(slot, len);
    m.seal_snapshot(snap, &pages).unwrap();
    (snap, pages)
}

/// The tensor-parallel end cut as the serving group takes it (`GlmTpBackend::inner_cut`: the cut ARMED before the prompt's last slice,
/// `GlmModel::arm_cut`, the slice running to the prompt's end, the snapshot completed after it), `cached`, or the same requests uncached:
/// * slot 0, the donor: A in two chunks (0..512, 512..700), the second passing its end cut 688 (cached: armed in it), answered, then a
///   window of its own;
/// * slot 1, A again: cached, first prefilled to exactly 688 by chunks (0..512, 512..688), the state a fresh prefill leaves there, then
///   restored from the in-chunk snapshot over it, which must leave the same state (`prefix_digest`, asserted); uncached, 0..512. Its
///   rest goes through ONE prompt forward with slot 2's whole B, whose end cut 224 is armed in that forward (cached);
/// * four greedy windows of the three slots, the drafts after;
/// * slot 2's follow-up B + C: cached, restored from B's in-chunk snapshot (224), its last 38 rows in a forward paused at layer `bound`
///   with a window of slots 0 and 1 between its parts, its own end cut 256 armed in it; uncached, all 262 from 0. Three greedy windows,
///   the drafts after;
/// * slot 0's turn B + C + D: cached, restored from the follow-up's in-chunk snapshot (256), its last 26 rows; uncached, all 282.
///   Three greedy windows, the drafts after.
///
/// One digest per answering call in call order, and every slot's drafts after the windows (rank 0's, or one model's, drafter). Every
/// entry is released at the end and every slot ended.
fn inner_cut_script(m: &mut GlmModel, cached: bool, bound: usize) -> (Vec<u64>, Vec<Vec<u32>>) {
    const P2: usize = gb_cuda::kda::P2_CHUNK_ROWS;
    let (a, b, c, d) = inner_cut_prompts();
    let (cut_a, cut_b, cut_bc) = (a.len() / P2 * P2, b.len() / P2 * P2, 224 + 32);
    assert!(cut_a > CHUNK && cut_a < a.len() && cut_b < b.len(), "each end cut inside its prompt's last chunk");
    let mut sq = Seqs::new();
    let mut out = Vec::new();
    let mut entries = Vec::new();
    // slot 0, the donor: its second chunk passes the end cut armed in it
    m.begin_slot(0).unwrap();
    m.dflash_prompt_end(0, Some(a.len()), cached.then_some(cut_a));
    if cached {
        assert!(m.arm_cut(0, cut_a).unwrap(), "a free snapshot slot");
    }
    assert!(m.prefill_chunk(0, 0, &a[..CHUNK], None).unwrap().is_none());
    sq.prompt(0, &a[..CHUNK], None);
    let cands = m.prefill_chunk(0, CHUNK, &a[CHUNK..], Some(COUNT)).unwrap().expect("A's candidates");
    out.push(digest_row(m, &cands, 0));
    sq.prompt(0, &a[CHUNK..], Some((&cands, 0)));
    if cached {
        entries.push(take_inner(m, 0, cut_a));
    }
    out.push(window(m, &mut sq, &[(0, Vec::new(), 1)]));
    // slot 1, A again
    let from = if cached {
        m.begin_slot(1).unwrap();
        assert!(m.prefill_chunk(1, 0, &a[..CHUNK], None).unwrap().is_none());
        assert!(m.prefill_chunk(1, CHUNK, &a[CHUNK..cut_a], None).unwrap().is_none());
        let fresh = m.prefix_digest(1, cut_a).unwrap();
        let (s, pages) = &entries[0];
        m.restore_prefix(1, *s, pages, cut_a).unwrap();
        let restored = m.prefix_digest(1, cut_a).unwrap();
        assert!(
            restored == fresh,
            "the state restored from the snapshot taken inside the chunk differs from a prefill that ended at the cut: {}",
            restored.describe_diff(&fresh)
        );
        cut_a
    } else {
        m.begin_slot(1).unwrap();
        m.dflash_prompt_end(1, Some(a.len()), None);
        assert!(m.prefill_chunk(1, 0, &a[..CHUNK], None).unwrap().is_none());
        CHUNK
    };
    sq.prompt(1, &a[..from], None);
    m.dflash_prompt_end(1, Some(a.len()), None);
    m.begin_slot(2).unwrap();
    m.dflash_prompt_end(2, Some(b.len()), cached.then_some(cut_b));
    if cached {
        assert!(m.arm_cut(2, cut_b).unwrap(), "a free snapshot slot");
    }
    let reqs = [
        gb_engine::RowRequest { slot: 1, tokens: a[from..].to_vec(), first_position: from as u64 },
        gb_engine::RowRequest { slot: 2, tokens: b.clone(), first_position: 0 },
    ];
    let cands = m.prefill_rows(&RowBatch::from_requests(&reqs), Some(COUNT), usize::MAX).unwrap().expect("the forward's candidates");
    for (i, (s, toks)) in [(1usize, &a[from..]), (2, &b[..])].into_iter().enumerate() {
        out.push(digest_row(m, &cands, i));
        sq.prompt(s, toks, Some((&cands, i)));
    }
    if cached {
        entries.push(take_inner(m, 2, cut_b));
    }
    out.extend(greedy_windows(m, &mut sq, 4));
    let mut drafts = drafts_now(m, &sq);
    // slot 2's follow-up turn, a forward in parts with a window of the other two between them, its end cut armed in it
    let bc: Vec<u32> = b.iter().chain(&c).copied().collect();
    m.end_slot(2);
    let from = if cached {
        let (s, pages) = &entries[1];
        m.restore_prefix(2, *s, pages, cut_b).unwrap();
        cut_b
    } else {
        m.begin_slot(2).unwrap();
        0
    };
    sq.fed[2] = bc[..from].to_vec();
    m.dflash_prompt_end(2, Some(bc.len()), cached.then_some(cut_bc));
    if cached {
        assert!(cut_bc < bc.len() && bc.len() - cut_bc < P2);
        assert!(m.arm_cut(2, cut_bc).unwrap(), "a free snapshot slot");
    }
    assert!(m.prefill_rows(&slice_of(2, from, &bc[from..]), Some(COUNT), bound).unwrap().is_none(), "the follow-up pauses");
    out.push(window(m, &mut sq, &[(0, Vec::new(), 1), (1, Vec::new(), 1)]));
    let cands = m.prefill_resume(usize::MAX).unwrap().expect("the follow-up's candidates");
    out.push(digest_row(m, &cands, 0));
    sq.prompt(2, &bc[from..], Some((&cands, 0)));
    if cached {
        entries.push(take_inner(m, 2, cut_bc));
    }
    out.extend(greedy_windows(m, &mut sq, 3));
    drafts.extend(drafts_now(m, &sq));
    // slot 0's turn B + C + D, from the follow-up's snapshot
    let bcd: Vec<u32> = bc.iter().chain(&d).copied().collect();
    m.end_slot(0);
    let from = if cached {
        let (s, pages) = &entries[2];
        m.restore_prefix(0, *s, pages, cut_bc).unwrap();
        cut_bc
    } else {
        m.begin_slot(0).unwrap();
        0
    };
    sq.fed[0] = bcd[..from].to_vec();
    m.dflash_prompt_end(0, Some(bcd.len()), None);
    let cands = m.prefill_chunk(0, from, &bcd[from..], Some(COUNT)).unwrap().expect("the turn's candidates");
    out.push(digest_row(m, &cands, 0));
    sq.prompt(0, &bcd[from..], Some((&cands, 0)));
    out.extend(greedy_windows(m, &mut sq, 3));
    drafts.extend(drafts_now(m, &sq));
    for s in 0..SLOTS {
        m.end_slot(s);
    }
    for (s, pages) in entries {
        m.release_prefix(s, &pages);
    }
    assert_eq!(m.prefix_free_snapshots(), 4, "every snapshot slot back: nothing armed is left");
    (out, drafts)
}

/// The tensor-parallel end cut (`GlmTpBackend::inner_cut`, `REXL3_GLM_TP_INNER_CUT`): a snapshot ARMED before a prompt's last slice
/// (`GlmModel::arm_cut`) and taken at its 16-row boundary INSIDE that forward (every KDA layer's recurrence split there, its state copied
/// out between the two segments; the ring tails, the pages and the drafter's window after the forward: `finish_cut`) restores the state a
/// prefill ending at the cut leaves (asserted per model and per rank), and a prompt restored from one and prefilled on (whole, inside a
/// forward of two prompts, inside a forward in parts), then decoded, answers every call and drafts exactly as the same requests uncached
/// ([`inner_cut_script`]), under P2 with the sequence-parallel residual, the latent cache of `REXL3_GLM_KV`, the DFlash2 drafter of
/// `REXL3_DFLASH_DIR` (split with `REXL3_GLM_DRAFT_SPLIT`), on one P2 model and on the two serving ranks.
///
///     flock -x /tmp/rexl3-gpu.lock env REXL3_REDSNOW_DIR=... REXL3_DFLASH_DIR=... REXL3_GLM_KV=fp8 cargo test --release -p gb-model \
///         --test glm_redsnow_tp p2_an_end_cut_inside -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn p2_an_end_cut_inside_the_last_chunk_restores_as_uncached_on_one_model_and_the_serving_ranks() {
    let Some(dir) = redsnow_dir() else { return };
    let (layers, draft, taps) = prefix_setup();
    assert!(layers >= 4, "parts need a few layers");
    require_available(if layers <= 8 { 60 } else { 80 } * GIB);
    let guard = Guard::start(20 * GIB);
    let bound = layers / 2;
    let mut one = one_model(PrefillContract::P2, &dir, layers, draft.as_deref(), &taps, PART_ROOM);
    one.enable_prefix_cache(4).unwrap();
    assert!(one.cuts_inside_forwards());
    let kv = one.cfg.latent_kv.name().to_string();
    let cold = inner_cut_script(&mut one, false, bound);
    let warm = inner_cut_script(&mut one, true, bound);
    assert_eq!(warm.0, cold.0, "one P2 model: a cached answer (or a window after) differs from the uncached run");
    assert_eq!(warm.1, cold.1, "one P2 model: the drafts after a restore differ from the uncached run's");
    assert_eq!(inner_cut_script(&mut one, true, bound), warm, "one P2 model: the cached run twice");
    eprintln!("one P2 model ({kv} latent cache): {} answering calls and {} drafts, cached == uncached", cold.0.len(), cold.1.len());
    drop(one);

    let split = gb_model::families::glm::dflash::split_on() && draft.is_some();
    let (a, b) = serving_ranks(&dir, layers, draft.as_deref(), &taps, PART_ROOM, split);
    assert!(a.cuts_inside_forwards() && b.cuts_inside_forwards());
    let (r0, r1, a, b) = on_ranks(a, b, |m| (inner_cut_script(m, false, bound), inner_cut_script(m, true, bound)));
    assert_eq!((&r0.0 .0, &r0.1 .0), (&r1.0 .0, &r1.1 .0), "the two ranks' answers differ");
    assert_eq!(r0.1 .0, r0.0 .0, "two ranks: a cached answer (or a window after) differs from the uncached run");
    assert_eq!(r0.1 .1, r0.0 .1, "rank 0: the drafts after a restore differ from the uncached run's");
    if split {
        assert_eq!((&r1.0 .1, &r1.1 .1), (&r0.0 .1, &r0.1 .1), "the split drafter's halves drafted differently on the two ranks");
    }
    assert!(a.sp_chunks() > 0 && a.sp_chunks() == b.sp_chunks(), "sp chunks {} / {}", a.sp_chunks(), b.sp_chunks());
    eprintln!(
        "AN END CUT TAKEN INSIDE THE LAST CHUNK RESTORES AS UNCACHED (P2, {kv}, the drafter {}{}): {} answering calls and {} drafts on one \
         model and on two width ranks ({} sp forwards); armed in a whole chunk, in a forward of two prompts and in a forward in parts",
        if draft.is_some() { "on" } else { "off" },
        if split { ", split" } else { "" },
        r0.0 .0.len(),
        r0.0 .1.len(),
        a.sp_chunks()
    );
    drop((a, b));
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// The DFlash2 drafter split over the two serving ranks (`REXL3_GLM_DRAFT_SPLIT`, `draft::DraftShare`: each rank its half of the heads,
/// the intermediate columns and the head's columns, the o / down partials and the head's top-k swapped every round) against rank 0's
/// whole drafter on the same serving ranks ([`cache_script`] uncached, cached, and uncached again):
/// * the trunk does not see it: every answer equals the whole drafter's run (the follower's taps, absorbs and snapshot windows are its own);
/// * both ranks hold the same drafts, every run alike, and the drafts after a restore are the uncached run's (each rank's half of the
///   window snapshotted and put back);
/// * the drafts track the whole drafter's (each partial is rounded to bf16 before the add, so a near tie may fall the other way): the
///   share of draft tokens equal at their place is printed, and at least half of the first drafts must agree.
///
///     flock -x /tmp/rexl3-gpu.lock env REXL3_REDSNOW_DIR=... REXL3_DFLASH_DIR=... REXL3_GLM_KV=fp8 cargo test --release -p gb-model \
///         --test glm_redsnow_tp p2_a_split_drafter -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn p2_a_split_drafter_agrees_on_both_ranks_repeats_and_tracks_the_whole_drafter() {
    let Some(dir) = redsnow_dir() else { return };
    let (layers, draft, taps) = prefix_setup();
    let Some(draft) = draft else {
        eprintln!("REXL3_DFLASH_DIR not set: not run");
        return;
    };
    assert!(layers >= 4, "parts need a few layers");
    require_available(if layers <= 8 { 60 } else { 80 } * GIB);
    let guard = Guard::start(20 * GIB);
    let bound = layers / 2;
    let (a, b) = serving_ranks(&dir, layers, Some(&draft), &taps, PART_ROOM, false);
    let (whole, _, a, b) = on_ranks(a, b, |m| cache_script(m, false, bound));
    assert!(!b.has_dflash() && !a.dflash_split());
    drop((a, b));

    let (a, b) = serving_ranks(&dir, layers, Some(&draft), &taps, PART_ROOM, true);
    assert!(a.dflash_split() && b.dflash_split());
    let t = std::time::Instant::now();
    let (r0, r1, a, b) =
        on_ranks(a, b, |m| (cache_script(m, false, bound), cache_script(m, true, bound), cache_script(m, false, bound)));
    eprintln!("two ranks with a split drafter ran the script three times in {:.1?}", t.elapsed());
    assert_eq!(r0, r1, "the two ranks' answers or drafts differ");
    let (cold, warm, again) = r0;
    assert_eq!(cold.0, whole.0, "the split drafter moved an answer of the trunk");
    assert_eq!((&warm.0, &again.0), (&cold.0, &cold.0), "a cached (or repeated) answer differs from the uncached run");
    assert_eq!(again.1, cold.1, "the split drafter drafted differently on the same requests");
    assert_eq!(warm.1, cold.1, "the split drafter's drafts after a restore differ from the uncached run's");
    let (t0, t1) = (a.tp().unwrap(), b.tp().unwrap());
    assert_eq!((t0.exchanges, t0.exchanged_bytes), (t1.exchanges, t1.exchanged_bytes), "the ranks exchanged alike");
    assert_eq!(cold.1.len(), whole.1.len());
    let (mut same, mut all, mut first) = (0usize, 0usize, 0usize);
    for (s, w) in cold.1.iter().zip(&whole.1) {
        assert_eq!(s.len(), w.len(), "every slot drafts its seven");
        same += s.iter().zip(w).filter(|(x, y)| x == y).count();
        all += s.len();
        first += usize::from(s.first() == w.first());
    }
    eprintln!(
        "A SPLIT DRAFTER (two ranks, {} exchanges a rank): both ranks alike, repeat and restore alike, the trunk's {} answers unchanged; \
         {same} of {all} draft tokens and {first} of {} first drafts equal to the whole drafter's\n  split {:?}\n  whole {:?}",
        t0.exchanges,
        cold.0.len(),
        cold.1.len(),
        cold.1,
        whole.1
    );
    assert!(2 * first >= cold.1.len(), "the split drafter's first drafts left the whole drafter's: {first} of {}", cold.1.len());
    drop((a, b));
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// Draft rounds as the serving loop runs them, on one model with the drafter (graphs on): two prompts, then eight rounds of the flush,
/// each slot's greedy drafts (slot 0 three, slot 1 one to three) and a verify window of both, each slot keeping its drafts up to the first
/// that differs from the greedy pick (partial accepts: the KDA commits replay). `lazy`: the round's flush is `GlmModel::flush_lazy` (the
/// KDA commits beside the draft round, joined by the window's flush); otherwise `GlmModel::flush`. `(digests, drafts, partial rounds)`:
/// the prompts' and the windows' answers then every layer's state of both slots; every round's drafts; the rounds that kept fewer rows
/// than a window held.
fn draft_rounds(m: &mut GlmModel, layers: usize, lazy: bool) -> (Vec<u64>, Vec<Vec<u32>>, usize) {
    let mut r = Rng(0x1f83_d9ab_fb41_bd6b);
    let prompts = [r.tokens(300), r.tokens(90)];
    let mut sq = Seqs::new();
    let mut out = Vec::new();
    let mut drafts = Vec::new();
    let mut partial = 0;
    for (s, p) in prompts.iter().enumerate() {
        m.begin_slot(s).unwrap();
        let c = m.prefill_chunk(s, 0, p, Some(COUNT)).unwrap().expect("the prompt's candidates");
        out.push(digest_call(m, &c, 1));
        sq.prompt(s, p, Some((&c, 0)));
    }
    let greedy = gb_sampling::Sampling { temperature: 0.0, ..Default::default() };
    for round in 0..8 {
        if lazy {
            m.flush_lazy().unwrap();
        } else {
            m.flush().unwrap();
        }
        let mut reqs = Vec::new();
        for s in 0..2 {
            let toks: Vec<u32> = sq.fed[s].iter().copied().chain(std::iter::once(sq.next[s])).collect();
            let want = if s == 0 { 3 } else { 1 + round % 3 };
            let ask = gb_model::families::glm::model::DflashAsk { slot: s, tokens: &toks, sampling: &greedy, want };
            let d = m.dflash_draft(&[ask]).unwrap().remove(0).tokens;
            let tokens = std::iter::once(sq.next[s]).chain(d.iter().copied()).collect();
            reqs.push(gb_engine::RowRequest { slot: s, tokens, first_position: sq.fed[s].len() as u64 });
            drafts.push(d);
        }
        let b = RowBatch::from_requests(&reqs);
        let c = m.window_candidates(&b, COUNT).unwrap();
        out.push(digest_call(m, &c, b.rows));
        let pick = |row: usize| c.ids[row * COUNT] as u32;
        for (i, q) in reqs.iter().enumerate() {
            let st = b.seq_rows(i).start;
            let mut keep = 1;
            while keep < q.tokens.len() && pick(st + keep - 1) == q.tokens[keep] {
                keep += 1;
            }
            partial += usize::from(keep < q.tokens.len());
            m.commit(q.slot, keep);
            sq.fed[q.slot].extend_from_slice(&q.tokens[..keep]);
            sq.next[q.slot] = pick(st + keep - 1);
        }
    }
    m.flush().unwrap();
    for s in 0..2 {
        out.extend(slot_state(m, layers, s, sq.fed[s].len()));
    }
    for s in 0..SLOTS {
        m.end_slot(s);
    }
    (out, drafts, partial)
}

/// A draft round's lazy flush (`GlmModel::flush_lazy`: the window's KDA commits on the side's commit stream, beside the draft round,
/// joined by the next flush) against the joined flush: [`draft_rounds`] gives every answer, every draft and every layer's state alike,
/// with partial accepts (replays) among its rounds. The joined run twice is the control.
///
///     flock -x /tmp/rexl3-gpu.lock env REXL3_REDSNOW_DIR=... REXL3_DFLASH_DIR=... REXL3_GLM_KV=fp8 cargo test --release -p gb-model \
///         --test glm_redsnow_tp a_draft_rounds_lazy_flush -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn a_draft_rounds_lazy_flush_equals_the_joined_flush() {
    let Some(dir) = redsnow_dir() else { return };
    let (layers, draft, taps) = prefix_setup();
    let Some(draft) = draft else {
        eprintln!("REXL3_DFLASH_DIR not set: not run");
        return;
    };
    require_available(if layers <= 8 { 60 } else { 80 } * GIB);
    let guard = Guard::start(20 * GIB);
    let mut m = one_model(PrefillContract::P2, &dir, layers, Some(&draft), &taps, 0);
    m.set_graphs(true);
    let joined = draft_rounds(&mut m, layers, false);
    let lazy = draft_rounds(&mut m, layers, true);
    let again = draft_rounds(&mut m, layers, false);
    assert_eq!(again, joined, "the control: the joined run twice");
    assert!(joined.2 > 0, "no round kept fewer rows than its window held (no replay ran)");
    assert_eq!(lazy.1, joined.1, "a draft differs after a lazy flush");
    assert_eq!(lazy.0, joined.0, "an answer or a layer's state differs after a lazy flush");
    eprintln!(
        "LAZY FLUSH: {} answers and layer states, {} drafts, {} partial rounds equal to the joined flush's",
        joined.0.len(),
        joined.1.len(),
        joined.2
    );
    drop(m);
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// Every layer's state of `slot` after its first `n` prompt rows, one digest a layer: a KDA layer's current state and conv history (its
/// other buffer and the parity carry what earlier runs on the model left); an MLA layer's latent rows (through the slot's pages) and DSA
/// pool rows.
fn slot_state(m: &mut GlmModel, layers: usize, slot: usize, n: usize) -> Vec<u64> {
    (0..layers)
        .map(|l| {
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            if l % 4 == 3 {
                for x in m.latent_rows(l, slot, n).unwrap() {
                    fnv(&mut h, &x.to_le_bytes());
                }
                for x in m.pool_rows(l, slot, n / gb_cuda::dsa::P).unwrap() {
                    fnv(&mut h, &x.to_le_bytes());
                }
            } else {
                let (state, conv) = m.kda_state(l, slot).unwrap();
                for x in state {
                    fnv(&mut h, &x.to_bits().to_le_bytes());
                }
                for x in conv {
                    fnv(&mut h, &x.to_le_bytes());
                }
            }
            h
        })
        .collect()
}

/// The head-rows test's calls: three prompts chunked (2,300 rows in 512-row chunks: the large path; 200 in 128-row chunks and 37 whole:
/// the window-sized path) with every layer's state after every chunk, then [`script`] (its windows after), [`batch_script`]'s three
/// prompts in one forward and [`pause_script`]'s slice whole and paused at three bounds. `(answers, states)`: the answering calls'
/// digests in call order, the state digests.
fn head_rows_script(m: &mut GlmModel, layers: usize) -> (Vec<u64>, Vec<u64>) {
    head_rows_script_with(m, layers, 2300)
}

/// [`head_rows_script`] with a first prompt of `first` rows (in 512-row chunks).
fn head_rows_script_with(m: &mut GlmModel, layers: usize, first: usize) -> (Vec<u64>, Vec<u64>) {
    let (mut answers, mut states) = (Vec::new(), Vec::new());
    let mut r = Rng(0x3c6e_f372_fe94_f82b);
    for (slot, n, chunk) in [(0usize, first, CHUNK), (1, 200, 128), (2, 37, 128)] {
        let p = r.tokens(n);
        m.begin_slot(slot).unwrap();
        let mut at = 0;
        while at < n {
            let end = (at + chunk).min(n);
            if let Some(c) = m.prefill_chunk(slot, at, &p[at..end], (end == n).then_some(COUNT)).unwrap() {
                answers.push(digest_call(m, &c, 1));
            }
            states.extend(slot_state(m, layers, slot, end));
            at = end;
        }
    }
    m.flush().unwrap();
    for s in 0..SLOTS {
        m.end_slot(s);
    }
    answers.extend(script(m));
    answers.extend(batch_script(m, true, false).0);
    answers.extend(pause_script(m, &[], &[], false).0);
    answers.extend(pause_script(m, &[1, layers / 2, layers - 1], &[layers / 2], false).0);
    (answers, states)
}

/// The last layer's tail on the head rows alone (`ops::plan_forward_head_rows`, on by default) against every row: a prefix whose last
/// layer is KDA (`REXL3_REDSNOW_TP_LAYERS`, one fewer when it ends on an MLA layer), [`head_rows_script`] with it and without it
/// (`GlmModel::set_head_rows`) on one model and on the two ranks of the serving split (each rank against itself for the states, which
/// hold its heads; every rank's answers against the one model's).
#[test]
#[ignore]
fn p1_the_last_layers_tail_on_the_head_rows_equals_every_rows() {
    let Some(dir) = redsnow_dir() else { return };
    let (layers, _, _) = prefix_setup();
    let layers = if layers % 4 == 0 { layers - 1 } else { layers };
    assert!(layers >= 4, "parts need a few layers");
    require_available(if layers <= 8 { 60 } else { 80 } * GIB);
    let guard = Guard::start(20 * GIB);
    let mut a = one_model(PrefillContract::P1, &dir, layers, None, &[], PART_ROOM);
    a.set_head_rows(false);
    let every = head_rows_script(&mut a, layers);
    assert_eq!(a.head_rows_runs(), 0, "the head rows' tail is off");
    a.set_head_rows(true);
    let head = head_rows_script(&mut a, layers);
    assert!(a.head_rows_runs() > 0, "the head rows' tail ran");
    // (the control: the script again without it, after both, equals its first run)
    a.set_head_rows(false);
    assert_eq!(head_rows_script(&mut a, layers), every, "one model: the script without the head rows' tail twice");
    let first_diff = |x: &[u64], y: &[u64]| x.iter().zip(y).position(|(p, q)| p != q).map(|i| (i / layers, i % layers));
    assert_eq!(head.0, every.0, "one model: an answer differs with the last layer's tail on the head rows");
    assert_eq!(
        first_diff(&head.1, &every.1),
        None,
        "one model: a layer's state differs with the last layer's tail on the head rows (chunk, layer)"
    );
    eprintln!(
        "one model ({layers} layers): {} answering calls and {} layer states equal with the tail on the head rows ({} forwards) and on every row",
        every.0.len(),
        every.1.len(),
        a.head_rows_runs()
    );
    drop(a);

    let (a, b) = two_ranks(PrefillContract::P1, &dir, layers, None, &[], PART_ROOM);
    let (h0, h1, a, b) = on_ranks(a, b, |m| head_rows_script(m, layers));
    let (e0, e1, a, b) = on_ranks(a, b, |m| {
        m.set_head_rows(false);
        head_rows_script(m, layers)
    });
    assert!(a.head_rows_runs() > 0 && a.head_rows_runs() == b.head_rows_runs(), "both ranks ran the head rows' tail alike");
    assert_eq!((&h0.0, &h1.0), (&every.0, &every.0), "two ranks with the tail on the head rows: an answer differs from the one model's");
    assert_eq!((&e0.0, &e1.0), (&every.0, &every.0), "two ranks with the tail on every row: an answer differs from the one model's");
    assert_eq!((&h0.1, &h1.1), (&e0.1, &e1.1), "two ranks: a layer's state differs with the last layer's tail on the head rows");
    eprintln!(
        "HEAD ROWS' TAIL BYTE-IDENTICAL (P1, {layers} layers): {} answering calls and {} layer states, one model and two ranks ({} forwards a rank)",
        every.0.len(),
        every.1.len(),
        a.head_rows_runs()
    );
    drop((a, b));
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// [`p1_the_last_layers_tail_on_the_head_rows_equals_every_rows`] under P2: [`head_rows_script_with`] with a first prompt of 2,400 rows
/// (its last chunk of 352 rows row-split, the head row on rank 1; [`batch_script`]'s 467-row forward puts a head row on each rank) with
/// the tail on the head rows and on every row, on one P2 model and on the two serving ranks ([`serving_ranks`]: the width split, the
/// sequence-parallel residual, the drafter of `REXL3_DFLASH_DIR` with its taps relayed when set, none on the last layer). Each run is
/// checked against its own run with the tail on every row (the width ranks' P2 bits are not one model's), the ranks against each other.
#[test]
#[ignore]
fn p2_the_last_layers_tail_on_the_head_rows_equals_every_rows() {
    let Some(dir) = redsnow_dir() else { return };
    let (layers, draft, _) = prefix_setup();
    let layers = if layers % 4 == 0 { layers - 1 } else { layers };
    assert!(layers >= 4, "parts need a few layers");
    // the drafter taps five layers of the prefix, never the last (a tap of it keeps every row's tail)
    let taps: Vec<usize> = [1, 3, 4, 5, 7].iter().map(|&l: &usize| l.min(layers - 2)).collect();
    require_available(if layers <= 8 { 60 } else { 80 } * GIB);
    let guard = Guard::start(20 * GIB);
    let first = 2400;
    let mut a = one_model(PrefillContract::P2, &dir, layers, draft.as_deref(), &taps, PART_ROOM);
    a.set_head_rows(false);
    let every = head_rows_script_with(&mut a, layers, first);
    assert_eq!(a.head_rows_runs(), 0, "the head rows' tail is off");
    a.set_head_rows(true);
    let head = head_rows_script_with(&mut a, layers, first);
    assert!(a.head_rows_runs() > 0, "the head rows' tail ran");
    a.set_head_rows(false);
    assert_eq!(head_rows_script_with(&mut a, layers, first), every, "one P2 model: the script without the head rows' tail twice");
    let first_diff = |x: &[u64], y: &[u64]| x.iter().zip(y).position(|(p, q)| p != q).map(|i| (i / layers, i % layers));
    assert_eq!(head.0, every.0, "one P2 model: an answer differs with the last layer's tail on the head rows");
    assert_eq!(first_diff(&head.1, &every.1), None, "one P2 model: a layer's state differs with the tail on the head rows (chunk, layer)");
    eprintln!(
        "one P2 model ({layers} layers): {} answering calls and {} layer states equal with the tail on the head rows ({} forwards)",
        every.0.len(),
        every.1.len(),
        a.head_rows_runs()
    );
    drop(a);

    let split = gb_model::families::glm::dflash::split_on() && draft.is_some();
    let (a, b) = serving_ranks(&dir, layers, draft.as_deref(), &taps, PART_ROOM, split);
    let (h0, h1, a, b) = on_ranks(a, b, |m| head_rows_script_with(m, layers, first));
    let runs = (a.head_rows_runs(), b.head_rows_runs(), a.sp_chunks());
    let (e0, e1, a, b) = on_ranks(a, b, |m| {
        m.set_head_rows(false);
        head_rows_script_with(m, layers, first)
    });
    assert!(runs.0 > 0 && runs.0 == runs.1, "both ranks ran the head rows' tail alike: {runs:?}");
    assert!(runs.2 > 0, "the serving ranks ran no forward under the sequence-parallel residual");
    assert_eq!(h0.0, h1.0, "two P2 ranks with the tail on the head rows: their answers differ");
    assert_eq!((&h0.0, &h1.0), (&e0.0, &e1.0), "two P2 ranks: an answer differs with the tail on the head rows");
    assert_eq!(first_diff(&h0.1, &e0.1), None, "rank 0: a layer's state differs with the tail on the head rows (chunk, layer)");
    assert_eq!(first_diff(&h1.1, &e1.1), None, "rank 1: a layer's state differs with the tail on the head rows (chunk, layer)");
    eprintln!(
        "HEAD ROWS' TAIL BYTE-IDENTICAL (P2, {layers} layers, the drafter {}): {} answering calls and {} layer states, one model and two \
         serving ranks ({} forwards a rank, {} sp forwards)",
        if draft.is_some() { "on" } else { "off" },
        every.0.len(),
        every.1.len(),
        runs.0,
        runs.2
    );
    drop((a, b));
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}
