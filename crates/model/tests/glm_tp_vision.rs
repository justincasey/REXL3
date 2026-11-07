//! GLM-5.3-Flash IMAGE prompts on a two-rank group ON ONE GPU (`docs/GLM_TP.md` section 10), against one model with no group.
//!
//! * `an_image_prompt_on_two_ranks_equals_one_model_byte_for_byte`: synthetic images through the REAL preprocessing (the pack's
//!   `processor_config.json`, the server's `image_preprocessor`) and the REAL encoder (`GlmEncoder` over the tower of rank 0's map of the
//!   pack); a layer prefix as one model behind the single-node `GlmBackend`, against two share-only ranks (`ExpertOwnership::range(288, r,
//!   2)`, `LoopbackGroup`) behind rank 0's `GlmTpBackend` over a `TpLeader` and a follower thread (`run_follower` over a socket pair). The
//!   follower holds NO tower: it is handed the rows. Each rank's runtime digests the candidates and every fp16 logit of every answering
//!   call; the single node digests the same from its backend's answers and its model's logits. Script, at prefill chunks of 512 rows and of
//!   128 (the deadline row limit): a prompt with two images (one straddling chunk boundaries at 128 rows, both inside one chunk at 512) and
//!   text between and after them, a text prompt in another slot, a mixed window with partial commits and greedy windows. One model with
//!   graphs on == graphs off. Negative controls: (a) one spliced row perturbed by one bf16 ulp in every value (f32 bit 16) on BOTH ranks: the answers differ from
//!   the single node; (b) perturbed on the follower ONLY: rank 0 loses the group at the call whose digest differs.
//! * `the_redsnow_tower_equals_the_225bpw_tower_byte_for_byte` (V0): the same images through the tower of each pack: every feature byte
//!   equal (the 347 tower tensors of the two packs are byte-identical), and the two packs' preprocessing equal.
//!
//! Env: `REXL3_GLM_DIR` (the 2.25bpw pack), `REXL3_REDSNOW_DIR` (V0), `REXL3_GLM_TP_LAYERS` (default 8). Under the GPU lock, alone:
//!
//!     flock -x /tmp/rexl3-gpu.lock env REXL3_GLM_DIR=... cargo test --release -p gb-model --test glm_tp_vision -- --ignored --nocapture --test-threads=1

mod glm_support;

use gb_cuda::moe::ExpertOwnership;
use gb_engine::{Backend, ImageSpan, RowOutput, RowRequest};
use gb_model::error::ModelError;
use gb_model::families::glm::backend::GlmBackend;
use gb_model::families::glm::load::{boot_mapped, LoadOptions};
use gb_model::families::glm::model::{DraftAsk, Drafts, GlmModel, ModelOptions, TapBook};
use gb_model::families::glm::tp::{self, GlmCall, GlmRank, PrefixRuntime, SnapHandles};
use gb_model::families::glm::tp_backend::GlmTpBackend;
use gb_model::families::glm::tp_mtp::GlmMtpRuntime;
use gb_model::families::glm::tp_vision::{self, ImageRuntime, PrefillImage, TpVision};
use gb_model::families::glm::vision::{GlmVisionTower, TowerOptions};
use gb_model::families::glm::vision_serve::{FeatureBudget, GlmEncoder, GlmImageFeatures};
use gb_model::families::glm::GlmFamily;
use gb_model::families::nemotron_h_omni::tp_comm::LoopbackGroup;
use gb_model::forward::Candidates;
use gb_model::lockstep::{run_follower, CallReply, FollowerConfig, FollowerExit, Link, LockstepRuntime, TpLeader, TpTimeouts};
use gb_model::rows::RowBatch;
use gb_model::serve::{ImageEncoder, ServeOptions};
use glm_support::*;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const CHUNK: usize = 512;
const SLOTS: usize = 3;
const COUNT: usize = 8;
const E: usize = 288;
const OUT: usize = 4096;

type Log = Arc<Mutex<Vec<u64>>>;

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h = (*h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
}

/// The digest of an answering call: every candidate (ids, value bits) of its rows in order, then every fp16 logit of those rows.
fn digest(ids: &[i32], vals: &[f32], logits: &[u16]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &i in ids {
        fnv(&mut h, &i.to_le_bytes());
    }
    for &v in vals {
        fnv(&mut h, &v.to_bits().to_le_bytes());
    }
    for &x in logits {
        fnv(&mut h, &x.to_le_bytes());
    }
    h
}

/// A rank's runtime that digests every answering call (and, for the controls, perturbs the rows it splices).
struct Probe {
    r: GlmRank,
    log: Log,
    /// Flip f32 bit 16 (one bf16 ulp) of every value of the first row of the first splice of every image call.
    perturb: bool,
}

impl Probe {
    fn note(&self, c: &Candidates) {
        let logits = self.r.m.logits_host(0, c.rows).unwrap();
        self.log.lock().unwrap().push(digest(&c.ids, &c.vals, &logits));
    }
}

impl LockstepRuntime for Probe {
    type Call = GlmCall;
    fn vocab_size(&self) -> usize {
        self.r.vocab_size()
    }
    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        self.r.begin_slot(slot)
    }
    fn end_slot(&mut self, slot: usize) {
        self.r.end_slot(slot)
    }
    fn prefill_chunk(&mut self, b: &RowBatch, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        let c = self.r.prefill_chunk(b, want)?;
        if let Some(c) = &c {
            self.note(c);
        }
        Ok(c)
    }
    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        let c = self.r.window_candidates(b, count)?;
        self.note(&c);
        Ok(c)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.r.commit(slot, keep)
    }
    fn flush(&mut self) -> Result<(), ModelError> {
        self.r.flush()
    }
    fn apply_call(&mut self, _seq: u64, call: GlmCall) -> Result<CallReply, ModelError> {
        tp::apply_family(self, call)
    }
    fn has_mtp(&self) -> bool {
        self.r.has_mtp()
    }
}

// every runtime of a GLM backend can draft (`GlmTpBackend` is bound on both): this file's model has no NextN layer, so it never does
impl GlmMtpRuntime for Probe {
    fn mtp_book(&self, slot: usize) -> Option<TapBook> {
        self.r.mtp_book(slot)
    }
    fn mtp_draft_keyed(
        &mut self,
        asks: &[DraftAsk<'_>],
        keyed: &[Option<gb_sampling::Sampling>],
    ) -> Result<Vec<Option<Drafts>>, ModelError> {
        self.r.mtp_draft_keyed(asks, keyed)
    }
    fn mtp_disable_slot(&mut self, slot: usize) {
        self.r.mtp_disable_slot(slot)
    }
}

impl ImageRuntime for Probe {
    fn prefill_image(&mut self, call: &PrefillImage) -> Result<Option<Candidates>, ModelError> {
        let c = if self.perturb {
            let mut p = call.clone();
            // one bf16 ulp (f32 bit 16) of every value of the first row: the model reads the f32 stream at half precision, so a single
            // f32 ulp of one value moved none of the 14 answers (measured: the first run of this test)
            for v in 0..OUT {
                p.splice[0].data[4 * v + 2] ^= 1;
            }
            self.r.prefill_image(&p)?
        } else {
            self.r.prefill_image(call)?
        };
        if let Some(c) = &c {
            self.note(c);
        }
        Ok(c)
    }
}

// every runtime of a GLM backend can keep snapshots (`GlmTpBackend` is bound on it): this file's backends run without a prefix cache
impl PrefixRuntime for Probe {
    fn enable_snapshots(&mut self, slots: usize) -> Result<u64, ModelError> {
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
    fn prefix_snapshot(&mut self, slot: usize, snap: u32, len: usize) -> Result<bool, ModelError> {
        self.r.prefix_snapshot(slot, snap, len)
    }
    fn retain_prefix_pages(&mut self, slot: usize, len: usize) -> Vec<u32> {
        self.r.retain_prefix_pages(slot, len)
    }
    fn restore_prefix(&mut self, slot: usize, snap: u32, pages: &[u32], len: usize) -> Result<(), ModelError> {
        self.r.restore_prefix(slot, snap, pages, len)
    }
    fn release_prefix(&mut self, snap: u32, pages: &[u32]) {
        self.r.release_prefix(snap, pages)
    }
    fn seal_snapshot(&mut self, snap: u32, pages: &[u32]) -> Result<u64, ModelError> {
        self.r.seal_snapshot(snap, pages)
    }
    fn prefix_state_digest(&mut self, slot: usize, len: usize) -> Result<u64, ModelError> {
        self.r.prefix_state_digest(slot, len)
    }
    fn kv_pages(&self) -> (usize, Vec<usize>) {
        self.r.kv_pages()
    }
    fn snapshot_handles(&mut self) -> &mut SnapHandles {
        self.r.snapshot_handles()
    }
}

// every runtime of a GLM backend says whether it runs prompt forwards of several prompts or in parts (`GlmTpBackend` is bound on it):
// this file's backends prefill one prompt a call, whole, as the single node it is compared with
impl gb_model::families::glm::tp_part::PartRuntime for Probe {}

/// A deterministic synthetic RGB image: gradients, a checkerboard and a disc, so the tower sees structure, not noise.
fn synth(width: usize, height: usize, seed: u32) -> gb_vision::Rgb {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let (fx, fy) = (x as f64 / width as f64, y as f64 / height as f64);
            let disc = ((fx - 0.5).powi(2) + (fy - 0.45).powi(2)).sqrt() < 0.25;
            let check = ((x / 32) + (y / 32) + seed as usize) % 2 == 0;
            let r = if disc { 230 } else { (fx * 200.0) as u8 };
            let g = if check { 180 } else { (fy * 160.0) as u8 };
            let b = ((fx + fy) * 100.0) as u8 ^ (seed as u8).wrapping_mul(37);
            data.extend_from_slice(&[r, g, b]);
        }
    }
    gb_vision::Rgb { width, height, data }
}

fn options(dir: &Path) -> ServeOptions {
    let mut o = ServeOptions::new(dir.to_path_buf());
    o.max_image_tokens = 1024;
    o.vision = Some(Default::default());
    o
}

/// The encoder of rank 0's tower (`GlmEncoder` over the tower of `pack`) and the server's preprocessing of `dir`.
fn encoder(
    dir: &Path,
    pack: &gb_model::families::glm::GlmPack,
) -> (GlmEncoder, Arc<dyn gb_vision::preprocessor::ImagePreprocessor>) {
    let o = options(dir);
    let pp = gb_model::families::glm::image_preprocessor(dir, &o).unwrap().expect("the pack has a vision tower");
    let tower = GlmVisionTower::load(pack, dir, &TowerOptions::new(1024 * 4, 8 * GIB)).unwrap();
    (GlmEncoder::new(tower, FeatureBudget::new(4 * GIB), &pp.id()).unwrap(), pp)
}

fn features_bytes(f: &Arc<dyn std::any::Any + Send + Sync>) -> Vec<u8> {
    let g = f.downcast_ref::<GlmImageFeatures>().unwrap();
    let mut out = vec![0u8; g.tokens * tp_vision::ROW_BYTES];
    g.buf.download_range(0, &mut out).unwrap();
    out
}

/// A layer prefix holding `own`'s experts (`None`: every expert).
fn load(dir: &Path, layers: usize, own: Option<&ExpertOwnership>) -> GlmModel {
    gb_cuda::rt::init_context().unwrap();
    let pack = match own {
        Some(o) => GlmFamily::map_pack_owned(dir, o).unwrap(),
        None => GlmFamily::map_pack(dir).unwrap(),
    };
    let b = boot_mapped(dir, pack, Some(layers), &LoadOptions::from_env(20 * GIB), 0.0).unwrap();
    GlmModel::new_chunked(
        b.pack.config.clone(),
        b.device,
        &ModelOptions { slots: SLOTS, kv_blocks: 64, max_context: 8192, layers },
        CHUNK,
    )
    .unwrap()
}

struct Prompt {
    tokens: Vec<u32>,
    images: Vec<ImageSpan>,
}

/// text (5) + image A + text (40) + image B + text (30).
fn image_prompt(image_token: u32, a: &gb_model::serve::EncodedImage, b: &gb_model::serve::EncodedImage) -> Prompt {
    let mut t: Vec<u32> = (0..5).map(|i| 1000 + i).collect();
    let mut images = Vec::new();
    for (k, im) in [a, b].into_iter().enumerate() {
        images.push(ImageSpan {
            start: t.len(),
            tokens: im.tokens,
            merged: im.merged,
            features: im.features.clone(),
            key: k as u64,
        });
        t.extend(std::iter::repeat_n(image_token, im.tokens));
        t.extend((0..if k == 0 { 40 } else { 30 }).map(|i| 2000 + 37 * i + k as u32));
    }
    Prompt { tokens: t, images }
}

/// The engine calls of the script on a backend; `answer` sees every answering call (a prefill's last row, a window's rows).
fn script<B: Backend>(be: &mut B, p: &Prompt, rows_limit: Option<usize>, answer: &mut dyn FnMut(&mut B, &[RowOutput])) {
    be.set_prefill_row_limit(rows_limit);
    let mut done = 0;
    let first = loop {
        let s = be.prefill_slice(0, &p.tokens, &p.images, done, COUNT);
        done = s.done;
        if let Some(r) = s.row {
            break r;
        }
    };
    answer(be, std::slice::from_ref(&first));
    let text: Vec<u32> = (0..37).map(|i| 3000 + 11 * i).collect();
    let mut done = 0;
    let second = loop {
        let s = be.prefill_slice(1, &text, &[], done, COUNT);
        done = s.done;
        if let Some(r) = s.row {
            break r;
        }
    };
    answer(be, std::slice::from_ref(&second));
    let (mut pos, mut last) = ([p.tokens.len(), text.len()], [first.ids[0], second.ids[0]]);
    let w = vec![
        RowRequest { slot: 0, tokens: vec![last[0], 11, 12], first_position: pos[0] as u64 },
        RowRequest { slot: 1, tokens: vec![last[1]], first_position: pos[1] as u64 },
    ];
    let out = be.step(&w, COUNT);
    let flat: Vec<RowOutput> = out.iter().flatten().cloned().collect();
    answer(be, &flat);
    for (s, keep) in [(0usize, 2usize), (1, 1)] {
        be.commit(s, keep);
        pos[s] += keep;
        last[s] = out[s][keep - 1].ids[0];
    }
    for _ in 0..4 {
        let w: Vec<RowRequest> =
            (0..2).map(|s| RowRequest { slot: s, tokens: vec![last[s]], first_position: pos[s] as u64 }).collect();
        let out = be.step(&w, COUNT);
        let flat: Vec<RowOutput> = out.iter().flatten().cloned().collect();
        answer(be, &flat);
        for s in 0..2 {
            be.commit(s, 1);
            pos[s] += 1;
            last[s] = out[s][0].ids[0];
        }
    }
    be.release(0);
    be.release(1);
}

fn timeouts() -> TpTimeouts {
    let d = Duration::from_secs(600);
    TpTimeouts { connect: d, ready: d, peer: d, write: d, barrier: d, hang_exit: None, ..TpTimeouts::default() }
}

/// The single node's digests of the script at both chunk grids (the backend's answers and its model's logits).
fn single(be: &mut GlmBackend, p: &Prompt) -> Vec<u64> {
    let mut d = Vec::new();
    for limit in [None, Some(128)] {
        script(be, p, limit, &mut |b: &mut GlmBackend, rows: &[RowOutput]| {
            let ids: Vec<i32> = rows.iter().flat_map(|r| r.ids.iter().map(|&x| x as i32)).collect();
            let vals: Vec<f32> = rows.iter().flat_map(|r| r.values.iter().copied()).collect();
            let logits = b.m.logits_host(0, rows.len()).unwrap();
            d.push(digest(&ids, &vals, &logits));
        });
        assert!(b_ok(be), "the single node failed: {:?}", be.take_error());
    }
    d
}

fn b_ok(be: &mut dyn Backend) -> bool {
    be.take_error().is_none()
}

/// The script at both chunk grids on a group of the two ranks (`perturb`: per rank), with the follower on its own thread. Returns each
/// rank's digests and the leader's failure (`None`: the group served everything).
fn group_run(ranks: [GlmModel; 2], perturb: [bool; 2], image_token: u32, p: &Prompt) -> (Vec<u64>, Vec<u64>, Option<String>) {
    let [m0, m1] = ranks;
    let (l0, l1): (Log, Log) = (Arc::default(), Arc::default());
    let (a, b) = Link::<GlmCall>::pair(Duration::from_secs(600)).unwrap();
    let f = Probe { r: GlmRank::new(m1), log: l1.clone(), perturb: perturb[1] };
    let h = std::thread::spawn(move || run_follower(f, b, FollowerConfig::new(timeouts())));
    let leader = TpLeader::lead(Probe { r: GlmRank::new(m0), log: l0.clone(), perturb: perturb[0] }, a, timeouts());
    let mut be = GlmTpBackend::new(leader, CHUNK, SLOTS).with_vision(TpVision::device(Some(image_token)));
    let mut failure = None;
    for limit in [None, Some(128)] {
        script(&mut be, p, limit, &mut |_, _| {});
        let e = be.take_error();
        if e.is_some() || be.leader.failure().is_some() {
            failure = Some(be.leader.failure().unwrap_or_else(|| format!("{e:?}")));
            break;
        }
    }
    if failure.is_none() {
        be.quiesce().unwrap();
    }
    drop(be);
    match h.join().expect("the follower thread") {
        Ok(FollowerExit::Shutdown(_)) if failure.is_none() => {}
        x => eprintln!("follower exit: {x:?}"),
    }
    let (d0, d1) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
    (d0, d1, failure)
}

fn ranks(dir: &Path, layers: usize, owns: &[ExpertOwnership; 2]) -> [GlmModel; 2] {
    let r = [load(dir, layers, Some(&owns[0])), load(dir, layers, Some(&owns[1]))];
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(600)).into_iter();
    let [mut a, mut b] = r;
    a.enable_tp(Arc::new(comms.next().unwrap()), owns[0].clone()).unwrap();
    b.enable_tp(Arc::new(comms.next().unwrap()), owns[1].clone()).unwrap();
    assert!(!a.graphs_on() && !b.graphs_on(), "a rank runs eagerly");
    [a, b]
}

#[test]
#[ignore]
fn an_image_prompt_on_two_ranks_equals_one_model_byte_for_byte() {
    let Some(dir) = pack_dir() else { return };
    let layers: usize = std::env::var("REXL3_GLM_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(60 * GIB);
    let guard = Guard::start(20 * GIB);
    gb_cuda::rt::init_context().unwrap();
    let owns = [ExpertOwnership::range(E, 0, 2).unwrap(), ExpertOwnership::range(E, 1, 2).unwrap()];

    // rank 0's tower, from rank 0's map of the pack (the serving path's), and the server's preprocessing
    let pack0 = GlmFamily::map_pack_owned(&dir, &owns[0]).unwrap();
    let image_token = pack0.config.vision.as_ref().unwrap().image_token_id;
    let (mut enc, pp) = encoder(&dir, &pack0);
    let (ia, ib) = (synth(448, 448, 1), synth(336, 224, 2));
    let ea = enc.encode(&pp.preprocess(&ia).unwrap()).unwrap();
    let eb = enc.encode(&pp.preprocess(&ib).unwrap()).unwrap();
    // an image is a function of itself (the tower is deterministic run to run)
    let ea2 = enc.encode(&pp.preprocess(&ia).unwrap()).unwrap();
    assert_eq!(features_bytes(&ea.features), features_bytes(&ea2.features), "the tower is not deterministic");
    drop(ea2);
    drop(enc); // the tower's job is done: the group splices the rows it was given
    let p = image_prompt(image_token, &ea, &eb);
    eprintln!(
        "image prompt: {} tokens; image A {} rows ({}x{}) at {}, image B {} rows ({}x{}) at {}",
        p.tokens.len(),
        ea.tokens,
        ea.merged.0,
        ea.merged.1,
        p.images[0].start,
        eb.tokens,
        eb.merged.0,
        eb.merged.1,
        p.images[1].start
    );
    assert!(
        p.images[0].start < 128 && p.images[0].start + p.images[0].tokens > 256,
        "image A straddles two 128-row chunk boundaries"
    );
    assert!(p.tokens.len() <= CHUNK, "both images inside one 512-row chunk");

    // one model, every expert, behind the single-node backend: graphs on, then off
    let one = load(&dir, layers, None);
    let mut be = GlmBackend::new(one).with_chunk(CHUNK);
    be.m.set_graphs(true);
    let reference = single(&mut be, &p);
    be.m.set_graphs(false);
    let eager = single(&mut be, &p);
    assert_eq!(reference, eager, "one model: graphs and eager differ");
    // the control of the control: text in place of the images (the placeholders' own embeddings) differs
    let text_only = Prompt { tokens: p.tokens.clone(), images: Vec::new() };
    be.m.set_graphs(true);
    let plain = single(&mut be, &text_only);
    assert_ne!(plain[0], reference[0], "the image rows changed nothing: the splice is not seen");
    drop(be);
    eprintln!("one model: {} answering calls, graphs == eager; without the splice the first answer differs", reference.len());

    // two ranks, the follower with no tower, behind rank 0's backend
    let t = std::time::Instant::now();
    let (d0, d1, failure) = group_run(ranks(&dir, layers, &owns), [false, false], image_token, &p);
    assert!(failure.is_none(), "the group failed: {failure:?}");
    eprintln!("two ranks ran the script in {:.1?}", t.elapsed());
    assert_eq!(d0, d1, "the two ranks' answers differ");
    assert_eq!(d0, reference, "the group's answers differ from the one model's");
    eprintln!("BYTE-IDENTICAL: {} answering calls (candidates + every fp16 logit), chunks of 512 and 128 rows", d0.len());

    // control (a): the same one-bf16-ulp change of one spliced row on both ranks: the group agrees with itself and differs from one model
    let (n0, n1, failure) = group_run(ranks(&dir, layers, &owns), [true, true], image_token, &p);
    assert!(failure.is_none(), "{failure:?}");
    assert_eq!(n0, n1);
    let differ = n0.iter().zip(&reference).filter(|(x, y)| x != y).count();
    assert!(differ > 0, "a one-bf16-ulp change of a spliced row went unnoticed");
    eprintln!(
        "control (a), one bf16 ulp of every value of one spliced row, on both ranks: {differ} of {} answering calls differ",
        n0.len()
    );

    // control (b): the change on the follower only: rank 0 loses the group, naming the call
    let (_, _, failure) = group_run(ranks(&dir, layers, &owns), [false, true], image_token, &p);
    let why = failure.expect("a follower that spliced other bytes must lose the group");
    assert!(why.contains("prefill_image") || why.contains("rank 1"), "{why}");
    eprintln!("control (b), the follower alone splices other bytes: the group is lost: {why}");
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

#[test]
#[ignore]
fn the_redsnow_tower_equals_the_225bpw_tower_byte_for_byte() {
    let Some(a_dir) = pack_dir() else { return };
    let Some(b_dir) = std::env::var("REXL3_REDSNOW_DIR").ok().map(PathBuf::from).filter(|p| p.join("config.json").exists())
    else {
        eprintln!("REXL3_REDSNOW_DIR not set: not run");
        return;
    };
    require_available(20 * GIB);
    let guard = Guard::start(20 * GIB);
    gb_cuda::rt::init_context().unwrap();
    let pc = |d: &Path| std::fs::read(d.join("processor_config.json")).unwrap();
    eprintln!("processor_config.json byte-identical: {}", pc(&a_dir) == pc(&b_dir));
    let demo = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vision/demo_scene.png")).unwrap();
    let run = |dir: &Path| -> (Vec<Vec<u16>>, Vec<Vec<u8>>) {
        let pack = GlmFamily::map_pack(dir).unwrap();
        let (mut enc, pp) = encoder(dir, &pack);
        let imgs = [synth(448, 448, 1), synth(336, 224, 2), pp.decode(&demo).unwrap()];
        let mut patches = Vec::new();
        let mut feats = Vec::new();
        for im in &imgs {
            let pre = pp.preprocess(im).unwrap();
            patches.push(pre.patches.clone());
            feats.push(features_bytes(&enc.encode(&pre).unwrap().features));
        }
        (patches, feats)
    };
    let (pa, fa) = run(&a_dir);
    let (pb, fb) = run(&b_dir);
    assert_eq!(pa, pb, "the two packs' preprocessing gives other patches");
    for (i, (x, y)) in fa.iter().zip(&fb).enumerate() {
        let same = x == y;
        eprintln!("image {i}: {} feature bytes, byte-identical: {same}", x.len());
        assert!(same, "image {i}: the RED-SNOW tower's features differ from the 2.25bpw tower's");
    }
    eprintln!(
        "V0: the RED-SNOW tower's features equal the 2.25bpw tower's for {} images; MemAvailable min {:.1} GiB",
        fa.len(),
        gib(guard.finish())
    );
}

/// Rank 0's boot path for the tower (`tp_vision::load_leader_tower`, then `publish_encoder`, what `tp_boot::build_leader` runs): rank 0
/// serving images loads the tower from its own map of the pack and publishes it as the server's encoder (whose features equal the
/// tower's, encoded directly); a follower, and rank 0 with `--no-vision`, load none.
#[test]
#[ignore]
fn rank_zeros_boot_loads_and_publishes_the_tower_and_a_follower_loads_none() {
    use gb_model::families::glm::tp_vision::{holds_tower, load_leader_tower, publish_encoder};
    use gb_model::serve::TpOptions;
    let Some(dir) = pack_dir() else { return };
    require_available(20 * GIB);
    let guard = Guard::start(20 * GIB);
    gb_cuda::rt::init_context().unwrap();
    let d = Duration::from_secs(1);
    let mut o = options(&dir);
    o.tp =
        Some(TpOptions { ranks: 2, rank: 0, peer: "127.0.0.1:1".into(), connect_timeout: d, ready_timeout: d, peer_timeout: d });
    let packs = [0, 1].map(|r| GlmFamily::map_pack_owned(&dir, &ExpertOwnership::range(E, r, 2).unwrap()).unwrap());
    // a follower (with a vision slot it would never get from the server) and rank 0 with --no-vision: no tower
    let mut f = o.clone();
    f.tp.as_mut().unwrap().rank = 1;
    assert!(!holds_tower(&f, &packs[1]) && load_leader_tower(&f, &packs[1]).unwrap().is_none());
    let mut nv = o.clone();
    nv.vision = None;
    assert!(!holds_tower(&nv, &packs[0]) && load_leader_tower(&nv, &packs[0]).unwrap().is_none());
    // rank 0: the tower, published
    assert!(holds_tower(&o, &packs[0]));
    let before = mem_available();
    let t = load_leader_tower(&o, &packs[0]).unwrap().expect("rank 0 loads the tower");
    eprintln!(
        "rank 0's tower: weights {:.3} GiB, scratch {:.3} GiB; MemAvailable fell {:.2} GiB",
        gib(t.weight_bytes()),
        gib(t.scratch_bytes()),
        gib(before.saturating_sub(mem_available()))
    );
    publish_encoder(&o, t, GIB).unwrap();
    let svc = o.vision.as_ref().unwrap().get().expect("the encoder is published").clone();
    let pp = gb_model::families::glm::image_preprocessor(&dir, &o).unwrap().unwrap();
    let pre = pp.preprocess(&synth(336, 224, 2)).unwrap();
    let via_service = features_bytes(&svc.encode(&pre).unwrap().features);
    let (mut enc, _) = encoder(&dir, &packs[0]);
    let direct = features_bytes(&enc.encode(&pre).unwrap().features);
    assert_eq!(via_service, direct, "the published encoder's features are the tower's");
    eprintln!(
        "published encoder == the tower encoded directly ({} bytes); MemAvailable min {:.1} GiB",
        direct.len(),
        gib(guard.finish())
    );
}

/// PASS_RULES_GLM R-G-vision (a) on one GPU: the features of the tower that rank 0's boot path loads from ITS map of the pack (its share
/// of the experts; `load_leader_tower` then `publish_encoder`, what `tp_boot::build_leader` runs) at the image kit's
/// `--max-image-tokens 4096`, against the same binary's tower alone on one node (the whole pack mapped, `GlmVisionTower::load` with the
/// same token limit), byte for byte, on the rule's images: the two synthetic ones, testdata/vision/demo_scene.png and the kit's large
/// image (grad_6000x4000_cap.png, scaled to the limit by the server's preprocessing). `REXL3_GLM_DIR` (the pack).
#[test]
#[ignore]
fn rank_zeros_published_tower_equals_the_single_node_tower_on_the_kit_images() {
    use gb_model::families::glm::tp_vision::{load_leader_tower, publish_encoder};
    use gb_model::serve::TpOptions;
    let Some(dir) = pack_dir() else { return };
    require_available(20 * GIB);
    let guard = Guard::start(20 * GIB);
    gb_cuda::rt::init_context().unwrap();
    let d = Duration::from_secs(1);
    let mut o = options(&dir);
    o.max_image_tokens = 4096;
    o.tp =
        Some(TpOptions { ranks: 2, rank: 0, peer: "127.0.0.1:1".into(), connect_timeout: d, ready_timeout: d, peer_timeout: d });
    let pack0 = GlmFamily::map_pack_owned(&dir, &ExpertOwnership::range(E, 0, 2).unwrap()).unwrap();
    let t = load_leader_tower(&o, &pack0).unwrap().expect("rank 0 loads the tower");
    let leader_patches = t.max_patches;
    publish_encoder(&o, t, GIB).unwrap();
    let svc = o.vision.as_ref().unwrap().get().expect("the encoder is published").clone();
    let pp = gb_model::families::glm::image_preprocessor(&dir, &o).unwrap().unwrap();
    // one node: the whole pack mapped, a tower for the same token limit
    let whole = GlmFamily::map_pack(&dir).unwrap();
    let tower = GlmVisionTower::load(&whole, &dir, &TowerOptions::new(4096 * 4, 8 * GIB)).unwrap();
    assert_eq!(tower.max_patches, leader_patches, "the two towers are sized for different images");
    let mut single = GlmEncoder::new(tower, FeatureBudget::new(4 * GIB), &pp.id()).unwrap();
    let read =
        |name: &str| std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vision").join(name)).unwrap();
    let images = [
        ("synthetic 448x448", synth(448, 448, 1)),
        ("synthetic 336x224", synth(336, 224, 2)),
        ("demo_scene.png", pp.decode(&read("demo_scene.png")).unwrap()),
        ("grad_6000x4000_cap.png", pp.decode(&read("grad_6000x4000_cap.png")).unwrap()),
    ];
    for (name, im) in &images {
        let pre = pp.preprocess(im).unwrap();
        let a = features_bytes(&svc.encode(&pre).unwrap().features);
        let b = features_bytes(&single.encode(&pre).unwrap().features);
        let tokens = a.len() / tp_vision::ROW_BYTES;
        eprintln!(
            "{name}: {}x{} -> grid {:?}, {tokens} tokens, {} feature bytes: rank 0's published tower == the single node's: {}",
            im.width,
            im.height,
            pre.grid,
            a.len(),
            a == b
        );
        assert_eq!(a, b, "{name}: rank 0's tower differs from the single node's");
    }
    eprintln!("R-G-vision (a): {} images byte-identical; MemAvailable min {:.1} GiB", images.len(), gib(guard.finish()));
}
