//! Lane P (GLM-5.3-Flash, prefill contract P2: `families/glm/contract.rs`) on the whole model with the NextN layer (node 4, 90.5 GiB
//! resident). All `#[ignore]` and env-gated: REXL3_GLM_DIR (the pack), REXL3_GLM_ORACLE_DIR (real token ids), REXL3_GLM_CHUNK_ROWS (the
//! chunk the model is built for: at least the largest chunk of the test, 2,048 for the grids below). One model per process:
//! `--test-threads=1`, under the GPU lock. The model is built under the process's contract (P1 by default) and switched with
//! `GlmModel::set_prefill_contract` (one load serves both contracts: the same weights, the same scratch, the same thermal state).
//!
//! * `p2_chunk_grids_are_identical_and_every_dense_linear_takes_p2`: under P2 one prompt through many chunk grids (cuts at multiples of
//!   16 rows, P2's KDA chunk: 16-row chunks and tails, chunks at and around the decode kernels' 128 rows, 2,048- and 4,096-row chunks)
//!   leaves identical state (the 45 layers and the NextN layer's cache), identical next-token logits and identical decoding after it; the
//!   same grid twice is identical (two runs); every chunk's P2 GEMM launches equal the dense linears its rows pass through, its expert
//!   blocks the MoE layers, its chunked KDA recurrences the KDA layers (the every-row-through-P2 audit); a chunk that continues a segment
//!   off its 16-row grid is refused by name. P2's bits differ from P1's (P2 is active), and the P2 digest of the reference grid is printed
//!   (`P2DIGEST`) for the golden file.
//! * `p2_a_continuation_after_decode_rows_is_grid_independent`: a second turn prefilled into the same slot after decode rows starts a new
//!   segment at an unaligned position; its grids (and the first turn's) leave identical state, logits and decoding.
//! * `p2_against_p1_on_the_oracle_prompts` (natural routing) and `p2_against_p1_with_the_oracles_routing_replayed` (forced: only the
//!   kernel numerics differ): the logits of every prompt row of the oracle's text prompts under P1 and under P2: KL(P1 || P2) per row
//!   (natural log, over the tokenizer's 154,856 ids), top-1 agreement, the teacher-forced NLL of each, the greedy next token.
//! * `p2_image_chunks_are_grid_independent`: `img_small` with the oracle's features, under P2, through grids that cut the image.
//! * `bench_prefill_p1_against_p2`: prompt tokens per second at 2,048 and 8,192 tokens, the model's chunk, P1 and P2 interleaved
//!   A B B A (two rounds), one warm-up chunk each. MEASURED on the box it runs on; run it alone (the lane's quiet lock).

mod glm_support;

use gb_model::families::glm::alloc;
use gb_model::families::glm::contract::PrefillContract;
use gb_model::families::glm::model::{ForcedRouting, GlmModel, Head, Mode};
use gb_model::families::glm::npy;
use gb_model::families::glm::prefix;
use gb_model::families::glm::vision_serve::{GlmImageFeatures, SpliceOp};
use gb_model::rows::RowBatch;
use glm_support::*;
use std::path::PathBuf;

const KV_BLOCKS: usize = 80; // 20,480 tokens
const MAX_CONTEXT: usize = 12288;
const SLOTS: usize = 3;

/// The model of this process: REXL3_P2_KV_BLOCKS / REXL3_P2_MAX_CONTEXT override the KV pages and the context (the 32k bench needs
/// both larger).
fn model() -> Option<std::sync::MutexGuard<'static, GlmModel>> {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    shared_model_mtp(SLOTS, env("REXL3_P2_KV_BLOCKS", KV_BLOCKS), env("REXL3_P2_MAX_CONTEXT", MAX_CONTEXT))
}

fn oracle() -> Option<PathBuf> {
    match std::env::var("REXL3_GLM_ORACLE_DIR") {
        Ok(d) => Some(PathBuf::from(d)),
        Err(_) => {
            eprintln!("REXL3_GLM_ORACLE_DIR is not set: not run");
            None
        }
    }
}

fn ids_of(prompt: &str) -> Option<Vec<u32>> {
    let ids = npy::read(&oracle()?.join(prompt).join("ids.npy")).unwrap().to_i32().unwrap();
    Some(ids.iter().map(|&x| x as u32).collect())
}

/// `n` real token ids (the oracle's 2,278-token prompt, tiled).
fn tiled(n: usize) -> Option<Vec<u32>> {
    let ids = ids_of("long_2300")?;
    Some((0..n).map(|i| ids[i % ids.len()]).collect())
}

type KdaState = Vec<(Vec<u32>, Vec<u16>)>;

/// Everything a prompt leaves: the last row's logits bits, the committed state of every layer and of the NextN layer's cache, and what
/// decoding on from it gives.
#[derive(PartialEq)]
struct Snap {
    last_logits: Vec<u16>,
    top: (Vec<i32>, Vec<u32>),
    kda: KdaState,
    latent: Vec<Vec<u16>>,
    pools: Vec<Vec<u16>>,
    mtp_latent: Vec<u16>,
    mtp_pools: Vec<u16>,
    decoded: Vec<Vec<u16>>,
}

impl Snap {
    /// FNV-1a 64 over every part, in order (the P2 golden's line).
    fn digest(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |b: &[u8]| {
            for &x in b {
                h = (h ^ x as u64).wrapping_mul(0x100_0000_01b3);
            }
        };
        let u16s = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        eat(&u16s(&self.last_logits));
        eat(&self.top.0.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        eat(&self.top.1.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        for (s, c) in &self.kda {
            eat(&s.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
            eat(&u16s(c));
        }
        for v in self.latent.iter().chain(&self.pools) {
            eat(&u16s(v));
        }
        eat(&u16s(&self.mtp_latent));
        eat(&u16s(&self.mtp_pools));
        for v in &self.decoded {
            eat(&u16s(v));
        }
        h
    }
}

fn state(m: &mut GlmModel, slot: usize, committed: usize) -> (KdaState, Vec<Vec<u16>>, Vec<Vec<u16>>) {
    let (mut kda, mut latent, mut pools) = (Vec::new(), Vec::new(), Vec::new());
    for l in 0..45 {
        if l % 4 == 3 {
            latent.push(m.latent_rows(l, slot, committed).unwrap());
            pools.push(m.pool_rows(l, slot, committed / 4).unwrap());
        } else {
            let (s, c) = m.kda_state(l, slot).unwrap();
            kda.push((s.iter().map(|x| x.to_bits()).collect(), c));
        }
    }
    (kda, latent, pools)
}

/// An image of a prompt: its features (on the device) and the position of its first row.
type Image = (std::sync::Arc<GlmImageFeatures>, usize);

/// Prefill `ids` into `slot` in chunks of the sizes in `grid` (the last size repeats), then decode `decode` tokens greedily; with
/// `audit`, check each chunk's P2 launches against [`p2_expected_launches`].
fn run_grid(m: &mut GlmModel, slot: usize, ids: &[u32], grid: &[usize], decode: usize, audit: bool) -> Snap {
    run_grid_mm(m, slot, ids, grid, decode, audit, None)
}

/// [`run_grid`] for a prompt with an image: each chunk splices the image rows it holds (`prefill_chunk_mm`).
fn run_grid_mm(
    m: &mut GlmModel,
    slot: usize,
    ids: &[u32],
    grid: &[usize],
    decode: usize,
    audit: bool,
    img: Option<&Image>,
) -> Snap {
    m.begin_slot(slot).unwrap();
    let c = prefill_grid(m, slot, ids, 0, grid, audit.then_some(true), img).unwrap();
    snap_after(m, slot, ids.len(), ids.len(), &c, decode)
}

/// Prefill `ids` at positions `start..` of `slot` in chunks of the sizes in `grid` (the last size repeats); with `audit = Some(mtp)`, check
/// each chunk's P2 launches (every dense linear, every routed-expert block, every KDA layer's chunked recurrence; the NextN layer's prefill
/// counted when `mtp`: the drafter turns itself off for a sequence whose taps have a gap, e.g. a prompt appended after decode windows that
/// did not tap). The first refusal ends the prompt.
fn prefill_grid(
    m: &mut GlmModel,
    slot: usize,
    ids: &[u32],
    start: usize,
    grid: &[usize],
    audit: Option<bool>,
    img: Option<&Image>,
) -> Result<gb_model::forward::Candidates, String> {
    let (mut at, mut i) = (0, 0);
    let mut last = None;
    while at < ids.len() {
        let size = grid[i.min(grid.len() - 1)];
        i += 1;
        let end = (at + size).min(ids.len());
        let want = (end == ids.len()).then_some(8usize);
        let l0 = gb_cuda::glm_p2::launches();
        let e0 = gb_cuda::glm_p2_moe::runs();
        let k0 = gb_cuda::kda::chunked_runs();
        let a0 = gb_cuda::glm_p2_mla::runs();
        let splice: Vec<SpliceOp> = img
            .iter()
            .filter_map(|(f, s0)| {
                let (lo, hi) = ((*s0).max(start + at), (s0 + f.tokens).min(start + end));
                (lo < hi).then(|| SpliceOp { feats: f.clone(), dst_row: lo - start - at, src_row: lo - s0, n: hi - lo })
            })
            .collect();
        last = m.prefill_chunk_mm(slot, start + at, &ids[at..end], want, splice).map_err(|e| e.to_string())?;
        let got = gb_cuda::glm_p2::launches() - l0;
        let got_e = gb_cuda::glm_p2_moe::runs() - e0;
        let got_k = gb_cuda::kda::chunked_runs() - k0;
        let got_a = gb_cuda::glm_p2_mla::runs() - a0;
        if let Some(mtp) = audit {
            let off = m.has_mtp() && !mtp;
            let want = p2_expected_launches(m, end - at) - if off { 1 + 4 + 3 } else { 0 };
            assert_eq!(got, want, "grid {grid:?}: the chunk of rows {at}..{end} made {got} P2 GEMM launches, not {want}");
            let want_e = p2_expected_moe_runs(m) - off as u64;
            assert_eq!(got_e, want_e, "grid {grid:?}: the chunk of rows {at}..{end} ran {got_e} P2 expert blocks, not {want_e}");
            let want_k = p2_expected_kda_runs(m);
            assert_eq!(
                got_k, want_k,
                "grid {grid:?}: the chunk of rows {at}..{end} ran {got_k} chunked KDA recurrences, not {want_k}"
            );
            let want_a = p2_expected_mla_runs(m, end - at) - if off { p2_mla_runs_of_nextn(end - at) } else { 0 };
            assert_eq!(
                got_a, want_a,
                "grid {grid:?}: the chunk of rows {at}..{end} made {got_a} P2 MLA absorb / expand launches, not {want_a}"
            );
        }
        at = end;
    }
    last.ok_or_else(|| "no chunk answered".into())
}

/// Greedy decode windows of one row from `tok` at `pos` (every window P1: no P2 launch, no chunked recurrence): the logits of each and the
/// next token.
fn decode_greedy(m: &mut GlmModel, slot: usize, mut tok: u32, pos: usize, n: usize) -> (Vec<Vec<u16>>, u32) {
    let mut decoded = Vec::new();
    let (l0, e0, k0) = (gb_cuda::glm_p2::launches(), gb_cuda::glm_p2_moe::runs(), gb_cuda::kda::chunked_runs());
    let a0 = gb_cuda::glm_p2_mla::runs();
    for j in 0..n {
        let b = RowBatch::from_requests(&[gb_engine::RowRequest { slot, tokens: vec![tok], first_position: (pos + j) as u64 }]);
        m.forward(&b, Mode::Window, Head::AllRows).unwrap();
        decoded.push(m.logits_host(0, 1).unwrap());
        let cand = m.candidates_of(1, 1).unwrap();
        m.commit(slot, 1);
        tok = cand.ids[0] as u32;
    }
    assert_eq!(gb_cuda::glm_p2::launches(), l0, "a decode window took the P2 GEMM");
    assert_eq!(gb_cuda::glm_p2_moe::runs(), e0, "a decode window took the P2 expert GEMMs");
    assert_eq!(gb_cuda::kda::chunked_runs(), k0, "a decode window took the chunked KDA recurrence");
    assert_eq!(gb_cuda::glm_p2_mla::runs(), a0, "a decode window took the P2 MLA absorb / expand");
    (decoded, tok)
}

/// What `slot` holds after `total` committed positions whose last prompt chunk answered `c` (the NextN layer's cache over its first
/// `mtp_rows` positions: the rows its drafter wrote), and `decode` greedy steps on from it.
fn snap_after(
    m: &mut GlmModel,
    slot: usize,
    total: usize,
    mtp_rows: usize,
    c: &gb_model::forward::Candidates,
    decode: usize,
) -> Snap {
    let last_logits = m.logits_host(0, 1).unwrap();
    let (kda, latent, pools) = state(m, slot, total);
    let mtp_latent = m.mtp_latent_rows(slot, mtp_rows).unwrap();
    let mtp_pools = m.mtp_pool_rows(slot, mtp_rows / 4).unwrap();
    let (decoded, _) = decode_greedy(m, slot, c.ids[0] as u32, total, decode);
    Snap {
        last_logits,
        top: (c.ids.clone(), c.vals.iter().map(|v| v.to_bits()).collect()),
        kda,
        latent,
        pools,
        mtp_latent,
        mtp_pools,
        decoded,
    }
}

fn describe(a: &Snap, b: &Snap) -> String {
    let mut d = Vec::new();
    if a.last_logits != b.last_logits {
        d.push(format!(
            "last-row logits differ in {} of {} elements",
            a.last_logits.iter().zip(&b.last_logits).filter(|(x, y)| x != y).count(),
            a.last_logits.len()
        ));
    }
    if a.top != b.top {
        d.push("top candidates differ".into());
    }
    if let Some(l) = a.kda.iter().zip(&b.kda).position(|(x, y)| x != y) {
        d.push(format!("KDA state or conv of KDA layer #{l} differs"));
    }
    if let Some(l) = a.latent.iter().zip(&b.latent).position(|(x, y)| x != y) {
        d.push(format!("MLA latent rows of MLA layer #{l} differ"));
    }
    if let Some(l) = a.pools.iter().zip(&b.pools).position(|(x, y)| x != y) {
        d.push(format!("DSA pool rows of MLA layer #{l} differ"));
    }
    if a.mtp_latent != b.mtp_latent || a.mtp_pools != b.mtp_pools {
        d.push("the NextN layer's cache differs".into());
    }
    if let Some(j) = a.decoded.iter().zip(&b.decoded).position(|(x, y)| x != y) {
        d.push(format!("the logits of decode step {j} after the prompt differ"));
    }
    d.join("; ")
}

#[test]
#[ignore]
fn p2_chunk_grids_are_identical_and_every_dense_linear_takes_p2() {
    let Some(all) = ids_of("long_2300") else { return };
    let Some(mut m) = model() else { return };
    let max = m.chunk_rows();
    assert!(max >= 2048, "build the model for 2,048-row chunks (REXL3_GLM_CHUNK_ROWS=2048), not {max}");
    let p1_ref = {
        m.set_prefill_contract(PrefillContract::P1).unwrap();
        run_grid(&mut m, 0, &all[..1500], &[max.min(2048)], 6, false)
    };
    m.set_prefill_contract(PrefillContract::P2).unwrap();
    // 1,500: dense attention throughout; 2,278: the later rows are past the 2,048 boundary where the attention turns sparse; 5,000 (the
    // oracle's prompt tiled, a model built for 4,096-row chunks only): real 4,096-row chunks and their tails
    let long = tiled(5000).unwrap();
    let prompts: Vec<&[u32]> =
        if max >= 4096 { vec![&all[..1500], &all[..2278], &long[..]] } else { vec![&all[..1500], &all[..2278]] };
    for ids in prompts {
        let n = ids.len();
        let t = std::time::Instant::now();
        let reference = run_grid(&mut m, 0, ids, &[128], 6, true);
        eprintln!("P2DIGEST prompt={n} grid=[128] {:016x} ({:.1} s)", reference.digest(), t.elapsed().as_secs_f64());
        if n == 1500 {
            let why = describe(&reference, &p1_ref);
            assert!(!why.is_empty(), "P2 left P1's bits: the P2 GEMM did not run");
            eprintln!("  P2 against P1 (2,048-row chunks), as it must: {why}");
        }
        let again = run_grid(&mut m, 1, ids, &[128], 6, true);
        assert!(describe(&again, &reference).is_empty(), "prompt of {n}: two runs of the same grid differ (another slot)");
        // under P2 every call of a prefill segment but its last starts on the segment's 16-row KDA grid (docs/GLM_PREFILL_P2.md 8.1): the
        // grids below cut at multiples of 16 (the last chunk ends anywhere); an off-grid cut is refused by name (after the loop)
        let mut grids = vec![
            vec![max.min(2048)],
            vec![1024],
            vec![384],
            vec![16],
            vec![256],
            vec![112],
            vec![704, 144, 304],
            vec![16, 1008, 144],
            vec![96, 912, 16, 128],
            vec![2032, 16],
        ];
        if max >= 4096 {
            grids.extend([vec![4096], vec![4080, 16], vec![1008, 4096], vec![3008, 2048]]);
        }
        if n > 2278 {
            // the long prompt: the 128-row reference against the big chunks only (the small grids are the shorter prompts')
            grids.retain(|g| g[0] >= 1000);
        }
        for grid in grids {
            let t = std::time::Instant::now();
            let got = run_grid(&mut m, 0, ids, &grid, 6, true);
            let why = describe(&got, &reference);
            assert!(why.is_empty(), "P2, prompt of {n}, chunk grid {grid:?}: {why}");
            eprintln!(
                "  P2 grid {grid:?}: bit-identical to the reference (45 layers, the NextN cache, last-row logits, 6 decode steps), every chunk's dense linears and routed experts on P2, {:.1} s",
                t.elapsed().as_secs_f64()
            );
        }
        // a chunk that continues the segment off its grid is refused by name, before any work on the slot
        m.begin_slot(2).unwrap();
        prefill_grid(&mut m, 2, &ids[..129], 0, &[129], None, None).unwrap();
        let err = prefill_grid(&mut m, 2, &ids[129..300], 129, &[171], None, None).unwrap_err();
        assert!(err.contains("whole number of 16-row KDA chunks"), "{err}");
        eprintln!("  refused by name: {err}");
    }
    m.set_prefill_contract(PrefillContract::P1).unwrap();
}

/// Under P2, a prompt that CONTINUES a slot after decode rows (a second turn appended to the same slot) is a new prefill segment: it may start
/// at any position (here 1,500 + 5 decoded rows), and its own chunk grids (multiples of 16 from its first row) leave identical state,
/// logits and decoding. The first turn's grid does not matter either (its own segment).
#[test]
#[ignore]
fn p2_a_continuation_after_decode_rows_is_grid_independent() {
    let Some(all) = ids_of("long_2300") else { return };
    let Some(mut m) = model() else { return };
    m.set_prefill_contract(PrefillContract::P2).unwrap();
    let (a, b) = (&all[..1500], &all[1500..2200]);
    let run = |m: &mut GlmModel, slot: usize, ga: &[usize], gb: &[usize]| -> Snap {
        m.begin_slot(slot).unwrap();
        let c = prefill_grid(m, slot, a, 0, ga, Some(true), None).unwrap();
        let (_, tok) = decode_greedy(m, slot, c.ids[0] as u32, a.len(), 5);
        // the continuation: the first turn's reply's last token, then the second turn
        let mut cont = vec![tok];
        cont.extend_from_slice(b);
        // (the decode windows above did not tap the drafter, so it is off for the rest of this sequence: its prefill is not counted)
        let c = prefill_grid(m, slot, &cont, a.len() + 5, gb, Some(false), None).unwrap();
        // the NextN cache compared over the first turn only: the drafter, off for this sequence after the gap, writes no later row
        snap_after(m, slot, a.len() + 5 + cont.len(), a.len(), &c, 6)
    };
    let reference = run(&mut m, 0, &[512], &[128]);
    eprintln!("P2DIGEST continuation 1500+5+701 grid=[512]/[128] {:016x}", reference.digest());
    for (ga, gb) in
        [(vec![512usize], vec![128usize]), (vec![1024], vec![701]), (vec![128], vec![16, 336, 64]), (vec![1500], vec![240, 461])]
    {
        let got = run(&mut m, 1, &ga, &gb);
        let why = describe(&got, &reference);
        assert!(why.is_empty(), "P2 continuation, grids {ga:?} / {gb:?}: {why}");
        eprintln!("  P2 continuation grids {ga:?} / {gb:?}: bit-identical to the reference (state, logits, 6 decode steps)");
    }
    m.set_prefill_contract(PrefillContract::P1).unwrap();
}

/// Under P2 (a process whose contract is P2: the prefix cache binds the contract), a snapshot taken on its segment's 16-row grid and restored
/// into another (dirty) slot resumes with the bits of the fresh prefill: state of every layer, the NextN cache, the last row's logits and 6
/// decode steps, for cuts at the start, mid-prompt, past the 2,048 sparse boundary and at the backend's moved end cut (`prefix::cuts_for`:
/// a planned P - 16 lands on the grid); the donor and the resumed slot in different 16-row grids. A snapshot off the grid is refused by name.
#[test]
#[ignore]
fn p2_a_prefix_restore_at_a_grid_cut_equals_the_fresh_prefill() {
    if gb_model::families::glm::contract::current() != PrefillContract::P2 {
        return eprintln!("needs REXL3_GLM_PREFILL_CONTRACT=p2 (the prefix cache binds the process's contract): not run");
    }
    let Some(all) = ids_of("long_2300") else { return };
    let Some(mut m) = model() else { return };
    let _ = m.enable_prefix_cache(4);
    let n = 2278;
    let toks = &all[..n];
    let stranger = &all[..300];
    // the fresh prefill
    m.begin_slot(0).unwrap();
    let c = prefill_grid(&mut m, 0, toks, 0, &[512], Some(true), None).unwrap();
    let want = snap_after(&mut m, 0, n, n, &c, 6);
    m.end_slot(0);
    let none = |_: usize| false;
    let end_cut = prefix::cuts_for(PrefillContract::P2, 0, 0, &[n - 16], &none)[0];
    assert_eq!(end_cut, (n - 16) / 16 * 16);
    let grids: [(&[usize], &[usize]); 3] = [(&[128], &[512]), (&[1024], &[16, 1008, 240]), (&[16, 496], &[2048])];
    for (k, &c) in [16usize, 496, 1024, 2064, end_cut].iter().enumerate() {
        let (gd, gr) = grids[k % grids.len()];
        // the donor: the prompt up to the cut (its chunks on the grid), a snapshot when it is quiet, its pages by reference
        m.begin_slot(1).unwrap();
        prefill_grid(&mut m, 1, &toks[..c], 0, gd, None, None).unwrap();
        let snap = m.alloc_snapshot().expect("a free snapshot slot");
        m.prefix_snapshot(1, snap, c).unwrap();
        let pages = m.retain_prefix_pages(1, c);
        m.end_slot(1);
        // a stranger's prompt in the slot that will be restored into
        m.begin_slot(2).unwrap();
        prefill_grid(&mut m, 2, stranger, 0, &[256], None, None).unwrap();
        m.restore_prefix(2, snap, &pages, c).unwrap();
        let cand = prefill_grid(&mut m, 2, &toks[c..], c, gr, Some(true), None).unwrap();
        let got = snap_after(&mut m, 2, n, n, &cand, 6);
        let why = describe(&got, &want);
        assert!(why.is_empty(), "P2 restore at {c} (donor {gd:?}, resumed {gr:?}): {why}");
        m.end_slot(2);
        m.release_prefix(snap, &pages);
        eprintln!("  P2 restore at cut {c:5} (donor grid {gd:?}, resumed grid {gr:?}): bit-identical to the fresh prefill");
    }
    // a snapshot off the segment's grid is refused by name
    m.begin_slot(1).unwrap();
    prefill_grid(&mut m, 1, &toks[..1001], 0, &[512], None, None).unwrap();
    let snap = m.alloc_snapshot().expect("a free snapshot slot");
    let err = m.prefix_snapshot(1, snap, 1001).unwrap_err().to_string();
    assert!(err.contains("16-row KDA"), "{err}");
    eprintln!("  refused by name: {err}");
    m.end_slot(1);
}

/// The oracle's image prompt `img_small` (368 image rows from position 11) with the oracle's own features (no tower: the language model's
/// splice alone): under P2 every chunk grid, chunks that cut the image included, leaves identical state, logits and decoding, every chunk's
/// dense linears on P2 (the image rows go through the same linears as text rows).
#[test]
#[ignore]
fn p2_image_chunks_are_grid_independent() {
    let Some(dir) = oracle() else { return };
    let Some(ids) = ids_of("img_small") else { return };
    let Some(mut m) = model() else { return };
    let vd = dir.join("img_small").join("vision");
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(vd.join("meta.json")).unwrap()).unwrap();
    let start = meta["image_start_position"].as_u64().unwrap() as usize;
    let rows = meta["image_rows"].as_u64().unwrap() as usize;
    let g = meta["grid_thw"].as_array().unwrap();
    let (gh, gw) = (g[1].as_u64().unwrap() as usize, g[2].as_u64().unwrap() as usize);
    let f = npy::read(&vd.join("features.npy")).unwrap().to_f32().unwrap();
    let mut buf = gb_cuda::DevBuf::zeroed(rows * 4096 * 4).unwrap();
    buf.upload(&f).unwrap();
    let img: Image =
        (std::sync::Arc::new(GlmImageFeatures { buf, tokens: rows, merged: (gh / 2, gw / 2), key: 0, permit: None }), start);
    m.set_prefill_contract(PrefillContract::P2).unwrap();
    let reference = run_grid_mm(&mut m, 0, &ids, &[128], 6, true, Some(&img));
    eprintln!(
        "P2DIGEST prompt=img_small grid=[128] {:016x} ({} tokens, image rows {start}..{})",
        reference.digest(),
        ids.len(),
        start + rows
    );
    for grid in [vec![2048usize], vec![16, 32, 128], vec![208], vec![64, 64, 304], vec![384]] {
        let got = run_grid_mm(&mut m, 0, &ids, &grid, 6, true, Some(&img));
        let why = describe(&got, &reference);
        assert!(why.is_empty(), "P2 img_small, chunk grid {grid:?}: {why}");
        eprintln!(
            "  P2 img_small grid {grid:?}: bit-identical to the reference, every chunk's dense linears and routed experts on P2"
        );
    }
    m.set_prefill_contract(PrefillContract::P1).unwrap();
}

/// Per-row KL(P || Q) in nats over the first `vocab` columns of two fp16 logit rows, and whether their argmaxes agree.
fn kl_row(p: &[u16], q: &[u16], vocab: usize) -> (f64, bool, usize, usize) {
    let f = |b: u16| gb_exl3::f16::f16_to_f64(b);
    let (mut mp, mut mq, mut ap, mut aq) = (f64::NEG_INFINITY, f64::NEG_INFINITY, 0usize, 0usize);
    for i in 0..vocab {
        let (x, y) = (f(p[i]), f(q[i]));
        if x > mp {
            mp = x;
            ap = i;
        }
        if y > mq {
            mq = y;
            aq = i;
        }
    }
    let (mut sp, mut sq) = (0f64, 0f64);
    for i in 0..vocab {
        sp += (f(p[i]) - mp).exp();
        sq += (f(q[i]) - mq).exp();
    }
    let (lp, lq) = (mp + sp.ln(), mq + sq.ln());
    let mut kl = 0f64;
    for i in 0..vocab {
        let a = f(p[i]) - lp;
        let b = f(q[i]) - lq;
        kl += a.exp() * (a - b);
    }
    (kl.max(0.0), ap == aq, ap, aq)
}

/// The fp16 logits of every row of `ids` prefilled from an empty slot in the model's chunks (the all-row head).
fn all_row_logits(m: &mut GlmModel, ids: &[u32]) -> Vec<u16> {
    let v = m.cfg.vocab;
    m.enable_dump_logits().unwrap();
    m.begin_slot(0).unwrap();
    let chunk = m.chunk_rows();
    let mut out = Vec::with_capacity(ids.len() * v);
    let mut at = 0;
    while at < ids.len() {
        let end = (at + chunk).min(ids.len());
        let b = RowBatch::from_requests(&[gb_engine::RowRequest {
            slot: 0,
            tokens: ids[at..end].to_vec(),
            first_position: at as u64,
        }]);
        m.forward(&b, Mode::Prefill, Head::AllRows).unwrap();
        out.extend(m.logits_host(0, end - at).unwrap());
        at = end;
    }
    out
}

/// The oracle's routing of `prompt` (expert ids and gate weights at every MoE layer and position), as glm_full replays it.
fn forced_of(prompt: &str) -> Option<ForcedRouting> {
    let d = oracle()?.join(prompt);
    let ids = npy::read(&d.join("route_ids.npy")).unwrap();
    let w = npy::read(&d.join("route_w.npy")).unwrap();
    let lay = npy::read(&d.join("route_layers.npy")).unwrap().to_i32().unwrap();
    Some(ForcedRouting {
        layers: lay.iter().map(|&l| l as usize).collect(),
        t: ids.shape[1],
        ids: ids.to_i32().unwrap(),
        w: w.to_f32().unwrap().iter().map(|&x| gb_exl3::f16::f64_to_f16(x as f64)).collect(),
    })
}

/// Teacher-forced NLL (nats) of `next` under one fp16 logit row (log-softmax over the first `vocab` columns).
fn nll_row(p: &[u16], next: usize, vocab: usize) -> f64 {
    let f = |b: u16| gb_exl3::f16::f16_to_f64(b);
    let m = (0..vocab).map(|i| f(p[i])).fold(f64::NEG_INFINITY, f64::max);
    let s: f64 = (0..vocab).map(|i| (f(p[i]) - m).exp()).sum();
    m + s.ln() - f(p[next])
}

/// KL(P1 || P2), top-1 agreement and the teacher-forced NLL of P1 and P2 over every row of the oracle's text prompts (REXL3_GLM_PROMPTS),
/// with the routing natural (each contract's router decides) or `forced` (the oracle's routing replayed into both: only the kernel
/// numerics differ, as in docs/GLM53.md's forced tables).
fn kl_table(m: &mut GlmModel, forced: bool) {
    let mode = if forced { "forced" } else { "natural" };
    let v = m.cfg.vocab;
    let vocab = alloc::TOKENIZER_VOCAB;
    let prompts = std::env::var("REXL3_GLM_PROMPTS")
        .unwrap_or_else(|_| "one_token,short_20,chat_300,code_700,long_2300,think_prefix".into());
    let (mut all_kl, mut all_rows, mut all_agree, mut all_max) = (0f64, 0usize, 0usize, 0f64);
    for p in prompts.split(',') {
        let Some(ids) = ids_of(p) else { return };
        m.forced = if forced { forced_of(p) } else { None };
        m.set_prefill_contract(PrefillContract::P1).unwrap();
        let a = all_row_logits(m, &ids);
        m.set_prefill_contract(PrefillContract::P2).unwrap();
        let b = all_row_logits(m, &ids);
        m.set_prefill_contract(PrefillContract::P1).unwrap();
        m.forced = None;
        let (mut sum, mut mx, mut agree, mut differ_bits) = (0f64, 0f64, 0usize, 0usize);
        let (mut n1, mut n2, mut nd) = (0f64, 0f64, 0f64);
        let mut kls = Vec::with_capacity(ids.len());
        for r in 0..ids.len() {
            let (pa, pb) = (&a[r * v..(r + 1) * v], &b[r * v..(r + 1) * v]);
            if pa != pb {
                differ_bits += 1;
            }
            let (kl, same, _, _) = kl_row(pa, pb, vocab);
            kls.push(kl);
            sum += kl;
            mx = mx.max(kl);
            agree += same as usize;
            if r + 1 < ids.len() {
                let next = ids[r + 1] as usize;
                let (x, y) = (nll_row(pa, next, vocab), nll_row(pb, next, vocab));
                n1 += x;
                n2 += y;
                nd += (x - y).abs();
            }
        }
        kls.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let last = ids.len() - 1;
        let (_, same_last, ga, gb) = kl_row(&a[last * v..(last + 1) * v], &b[last * v..(last + 1) * v], vocab);
        let t = (ids.len() - 1).max(1) as f64;
        eprintln!(
            "KL {mode} {p:13} rows {:5}: mean KL(P1||P2) {:.3e} nats, median {:.3e}, p99 {:.3e}, max {:.3e}; top-1 agree {}/{} ({:.2} %); NLL P1 {:.4} P2 {:.4} mean|d| {:.4}; rows with other bits {}; greedy next P1 {ga} P2 {gb} {}",
            ids.len(),
            sum / ids.len() as f64,
            kls[kls.len() / 2],
            kls[(kls.len() * 99 / 100).min(kls.len() - 1)],
            mx,
            agree,
            ids.len(),
            100.0 * agree as f64 / ids.len() as f64,
            n1 / t,
            n2 / t,
            nd / t,
            differ_bits,
            if same_last { "same" } else { "DIFFERENT" }
        );
        all_kl += sum;
        all_rows += ids.len();
        all_agree += agree;
        all_max = all_max.max(mx);
    }
    eprintln!(
        "KL {mode} all rows {all_rows}: mean KL(P1||P2) {:.3e} nats, max {:.3e}; top-1 agree {all_agree}/{all_rows} ({:.2} %)",
        all_kl / all_rows as f64,
        all_max,
        100.0 * all_agree as f64 / all_rows as f64
    );
}

#[test]
#[ignore]
fn p2_against_p1_on_the_oracle_prompts() {
    let Some(_) = oracle() else { return };
    let Some(mut m) = model() else { return };
    kl_table(&mut m, false);
}

#[test]
#[ignore]
fn p2_against_p1_with_the_oracles_routing_replayed() {
    let Some(_) = oracle() else { return };
    let Some(mut m) = model() else { return };
    kl_table(&mut m, true);
}

#[test]
#[ignore]
fn bench_prefill_p1_against_p2() {
    let Some(ids) = tiled(8192) else { return };
    let Some(mut m) = model() else { return };
    let chunk = m.chunk_rows();
    let mut results: Vec<(usize, PrefillContract, f64)> = Vec::new();
    let order = [PrefillContract::P1, PrefillContract::P2, PrefillContract::P2, PrefillContract::P1];
    for round in 0..2 {
        for &n in &[2048usize, 8192] {
            for &c in &order {
                m.set_prefill_contract(c).unwrap();
                // one warm-up chunk, then the whole prompt timed
                m.begin_slot(0).unwrap();
                m.prefill_chunk(0, 0, &ids[..chunk.min(n)], None).unwrap();
                m.begin_slot(0).unwrap();
                let t = std::time::Instant::now();
                let mut at = 0;
                while at < n {
                    let end = (at + chunk).min(n);
                    m.prefill_chunk(0, at, &ids[at..end], (end == n).then_some(1)).unwrap();
                    at = end;
                }
                m.sync().unwrap();
                let s = t.elapsed().as_secs_f64();
                eprintln!("BENCH round {round} {n:5} tokens {:2}: {:.3} s, {:.1} tok/s", c.id(), s, n as f64 / s);
                results.push((n, c, n as f64 / s));
            }
        }
    }
    m.set_prefill_contract(PrefillContract::P1).unwrap();
    for &n in &[2048usize, 8192] {
        let med = |c: PrefillContract| {
            let mut v: Vec<f64> = results.iter().filter(|r| r.0 == n && r.1 == c).map(|r| r.2).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (v[(v.len() - 1) / 2] + v[v.len() / 2]) / 2.0
        };
        let (a, b) = (med(PrefillContract::P1), med(PrefillContract::P2));
        eprintln!(
            "BENCH {n:5} tokens, chunk {chunk}: P1 {a:.1} tok/s, P2 {b:.1} tok/s (median of 4 each, A B B A x 2): P2 / P1 = {:.3}; MemAvailable {:.1} GiB",
            b / a,
            gib(mem_available())
        );
    }
}

/// Lane P step 3's measurement: prompt tokens per second of one prefill (the trunk and the NextN layer, every chunk, the last row's logits),
/// P1 and P2 at the chunk sizes given, interleaved A B B A (two rounds; a warm-up chunk before each run): `REXL3_BENCH_TOKENS` (default
/// 2048,8192,32768: the 32k prompt needs REXL3_P2_KV_BLOCKS and REXL3_P2_MAX_CONTEXT large enough) and `REXL3_BENCH_ARMS` (default
/// `p1:2048,p2:2048,p1:4096,p2:4096`; each arm is contract:chunk, the chunk at most the model's). MEASURED on the box it runs on; run it alone
/// (the lane's quiet lock).
#[test]
#[ignore]
fn bench_prefill_contracts() {
    let lens: Vec<usize> = std::env::var("REXL3_BENCH_TOKENS")
        .unwrap_or_else(|_| "2048,8192,32768".into())
        .split(',')
        .map(|v| v.trim().parse().unwrap())
        .collect();
    let Some(ids) = tiled(*lens.iter().max().unwrap()) else { return };
    let Some(mut m) = model() else { return };
    let max = m.chunk_rows();
    let arms: Vec<(PrefillContract, usize, usize, usize)> = std::env::var("REXL3_BENCH_ARMS")
        .unwrap_or_else(|_| "p1:2048,p2:2048,p1:4096,p2:4096".into())
        .split(',')
        .map(|a| {
            let mut it = a.split(':');
            let c = gb_model::families::glm::contract::parse(it.next()).unwrap();
            let k = it.next().unwrap().trim().parse::<usize>().unwrap();
            // an optional third field: the P2 shared expert on the side stream (1) or the main stream (0, the default)
            let side = it.next().map_or(0, |v| v.trim().parse::<usize>().unwrap());
            // an optional fourth field: under P2 the chunked KDA recurrence (1, the default) or P1's row-by-row path (0: step (e)'s A/B)
            let kda = it.next().map_or(1, |v| v.trim().parse::<usize>().unwrap());
            (c, k, side, kda)
        })
        .filter(|&(_, k, _, _)| k <= max)
        .collect();
    assert!(!arms.is_empty(), "no arm fits the model's chunk of {max} rows");
    // A B B A over consecutive pairs of arms (an odd last arm runs twice in a row)
    let mut order = Vec::new();
    for pair in arms.chunks(2) {
        let (a, b) = (pair[0], *pair.get(1).unwrap_or(&pair[0]));
        order.extend([a, b, b, a]);
    }
    type BenchRow = (usize, (PrefillContract, usize, usize, usize), f64);
    let mut results: Vec<BenchRow> = Vec::new();
    let mut min_avail = f64::MAX;
    for round in 0..2 {
        for &n in &lens {
            for &arm in &order {
                let (c, chunk, side, kda) = arm;
                m.set_prefill_contract(c).unwrap();
                m.set_p2_shared_side(side != 0);
                m.set_p2_kda_chunked(kda != 0).unwrap();
                m.begin_slot(0).unwrap();
                m.prefill_chunk(0, 0, &ids[..chunk.min(n)], None).unwrap();
                m.begin_slot(0).unwrap();
                m.sync().unwrap();
                let t = std::time::Instant::now();
                let mut at = 0;
                while at < n {
                    let end = (at + chunk).min(n);
                    m.prefill_chunk(0, at, &ids[at..end], (end == n).then_some(1)).unwrap();
                    at = end;
                }
                m.sync().unwrap();
                let s = t.elapsed().as_secs_f64();
                min_avail = min_avail.min(gib(mem_available()));
                eprintln!(
                    "BENCH round {round} {n:6} tokens {} chunk {chunk:4} side {side} kda {kda}: {s:.3} s, {:.1} tok/s",
                    c.id(),
                    n as f64 / s
                );
                results.push((n, arm, n as f64 / s));
            }
        }
    }
    m.set_prefill_contract(PrefillContract::P1).unwrap();
    m.set_p2_shared_side(false);
    m.set_p2_kda_chunked(true).unwrap();
    for &n in &lens {
        let mut line = format!("BENCH {n:6} tokens:");
        let mut base = None;
        for &arm in &arms {
            let (c, chunk, side, kda) = arm;
            let mut v: Vec<f64> = results.iter().filter(|r| r.0 == n && r.1 == arm).map(|r| r.2).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = (v[(v.len() - 1) / 2] + v[v.len() / 2]) / 2.0;
            let b = *base.get_or_insert(med);
            line += &format!(
                " {}@{chunk}{} {med:.1} tok/s ({} runs, {:.1}-{:.1}, x{:.3});",
                c.id(),
                match (c == PrefillContract::P2, side, kda) {
                    (true, 1, 0) => "/side/kda-p1",
                    (true, 1, _) => "/side",
                    (true, _, 0) => "/kda-p1",
                    _ => "",
                },
                v.len(),
                v[0],
                v[v.len() - 1],
                med / b
            );
        }
        eprintln!("{line} (x against the first arm; MemAvailable min {min_avail:.1} GiB)");
    }
}

/// Where a prompt chunk's time goes under P1 and under P2 (the model's per-op-class event timer, `GlmModel::profile`): one chunk of
/// `REXL3_BENCH_CHUNK` rows (default the model's) at positions chunk..2 chunk of the tiled oracle prompt (the attention sparse past
/// 2,048), after the first chunk unprofiled, each contract twice. MEASURED on the box it runs on (the quiet lock).
#[test]
#[ignore]
fn profile_p1_against_p2() {
    let Some(mut m) = model() else { return };
    let chunk: usize = std::env::var("REXL3_BENCH_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(m.chunk_rows());
    let Some(ids) = tiled(2 * chunk.max(2048)) else { return };
    let warm = chunk.max(2048);
    for (c, side) in
        [(PrefillContract::P1, true), (PrefillContract::P2, false), (PrefillContract::P1, true), (PrefillContract::P2, true)]
    {
        m.set_prefill_contract(c).unwrap();
        m.set_p2_shared_side(side);
        m.begin_slot(0).unwrap();
        let mut at = 0;
        while at < warm {
            let end = (at + chunk).min(warm);
            m.prefill_chunk(0, at, &ids[at..end], None).unwrap();
            at = end;
        }
        m.sync().unwrap();
        m.profile = Some(Default::default());
        m.prefill_chunk(0, at, &ids[at..at + chunk], None).unwrap();
        m.sync().unwrap();
        let p = m.profile.take().unwrap();
        let total: f64 = p.ms.values().sum();
        eprintln!(
            "PROFILE {}{}: a chunk of {chunk} rows at {at}..{}: {total:.1} ms ({:.1} tok/s)",
            c.id(),
            if c == PrefillContract::P2 && side {
                " (shared expert on the side stream: overlapped, its class near 0)"
            } else {
                ""
            },
            at + chunk,
            chunk as f64 / total * 1e3
        );
        let mut v: Vec<_> = p.ms.iter().collect();
        v.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        for (k, ms) in v {
            eprintln!("PROFILE {}   {k:40} {ms:9.2} ms {:5.1} %", c.id(), 100.0 * ms / total);
        }
        for (k, ms) in &p.sub {
            eprintln!("PROFILE {}     . {k:60} {ms:9.2} ms {:5.1} %", c.id(), 100.0 * ms / total);
        }
    }
    m.set_prefill_contract(PrefillContract::P1).unwrap();
    m.set_p2_shared_side(false);
}
