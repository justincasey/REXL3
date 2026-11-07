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
