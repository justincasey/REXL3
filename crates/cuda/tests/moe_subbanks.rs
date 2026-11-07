//! Mixed-width routed experts as homogeneous SUBBANKS (`src/moe_subbanks.rs`): on real layers of the uniform GLM-5.3-Flash 2.25bpw
//! pack, a layer's experts are split into ARTIFICIAL subbanks (same width, arbitrary partitions: scattered thirds, a skewed 1 + 287,
//! eight interleaved, a split gate/up launch, two ranks of three subbanks each, and the region-built constructor), and the subbank path
//! (pairs per subbank into one buffer, the `-0.0` filler from the first, one combine with global picks) must equal the single bank
//! BYTE FOR BYTE: `out` of every row and `y` of every valid pair, decode and batched, at 1, 2, 5, 22 and 64 rows.
//!
//! `real_mixed_layers_run_p2_per_subbank` (the 2.49bpw or the 4.91bpw pack): REAL mixed layers under the P2 prefill contract
//! (`glm_p2_moe::run_subbanks`: the P2 pairs stage once per subbank, a split gate/up subbank through the decode pairs stage in windows,
//! one combine): row independence, chunked == whole, determinism, two ranks of the range ownership (`pairs_subbanks_rank`, `Own(r)`,
//! `Parts`) == one node, and the error against an f64 reference of the exact dequantized experts beside P1's, within the tolerances
//! `docs/GLM_PREFILL_P2.md` section 6 states.
//!
//! With `REXL3_REDSNOW_DIR` (a mixed-width pack, default `~/models/vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw`), the second test
//! loads REAL mixed layers (4 and 7) and checks the subbank path against every expert run alone in a one-expert bank.
//!
//!     cargo test --release -p gb-cuda --test moe_subbanks -- --test-threads=1 --nocapture
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments)]
mod common;
mod exl3_ref;

use common::*;
use exl3_ref::{err, expert_ref, weights};
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::moe::*;
use gb_cuda::moe_prefill::PrefillScratch;
use gb_cuda::{device_available, nccl, DevBuf, Stream};
use gb_exl3::checkpoint::{scan, Checkpoint, LoadedGroup};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};

const ROWS: [usize; 5] = [1, 2, 5, 22, 64];
const NEG_ZERO: u32 = 0x8000_0000;

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

fn dir_of(var: &str, default: &str) -> Option<std::path::PathBuf> {
    let p = std::env::var(var)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(default));
    p.join("model.safetensors.index.json").exists().then_some(p)
}

struct LayerGroups {
    gate: Vec<LoadedGroup>,
    up: Vec<LoadedGroup>,
    down: Vec<LoadedGroup>,
}

fn load_layer(ck: &Checkpoint, l: usize, n: usize) -> LayerGroups {
    let t = std::time::Instant::now();
    let load = |proj: &str| -> Vec<LoadedGroup> {
        let threads = 8usize;
        let per = n.div_ceil(threads);
        let mut out: Vec<Option<LoadedGroup>> = (0..n).map(|_| None).collect();
        std::thread::scope(|sc| {
            for (t, chunk) in out.chunks_mut(per).enumerate() {
                sc.spawn(move || {
                    for (j, slot) in chunk.iter_mut().enumerate() {
                        let prefix = format!("model.language_model.layers.{l}.mlp.experts.{}.{proj}", t * per + j);
                        *slot = Some(ck.load_group(&prefix).unwrap_or_else(|e| panic!("{prefix}: {e}")));
                    }
                });
            }
        });
        out.into_iter().map(|o| o.unwrap()).collect()
    };
    let g = LayerGroups { gate: load("gate_proj"), up: load("up_proj"), down: load("down_proj") };
    eprintln!("layer {l}: groups read in {:.1} s", t.elapsed().as_secs_f64());
    g
}

fn pick(v: &[LoadedGroup], ids: &[u32]) -> Vec<LoadedGroup> {
    ids.iter().map(|&g| v[g as usize].clone()).collect()
}

/// A rank's subbanks from explicit id sets (each a bank of L1's kind over the same width: an ARTIFICIAL partition).
fn subbanks(lg: &LayerGroups, d: MoeDims, own: &ExpertOwnership, parts: &[Vec<u32>], split: &[usize]) -> MoeSubbanks {
    let mut banks = Vec::new();
    for (i, ids) in parts.iter().enumerate() {
        let o = ExpertOwnership::from_ids(d.experts, ids.clone()).unwrap();
        let mut b = MoeExperts::from_groups_owned(&pick(&lg.gate, ids), &pick(&lg.up, ids), &pick(&lg.down, ids), d, o).unwrap();
        if split.contains(&i) {
            b.force_split_gate_up(true);
        }
        banks.push(b);
    }
    MoeSubbanks::new(d, own.clone(), banks).unwrap()
}

/// Split `own`'s experts by `f(global id)` into `n` id sets (empty ones dropped).
fn split_by(own: &ExpertOwnership, n: usize, f: impl Fn(u32) -> usize) -> Vec<Vec<u32>> {
    let mut v = vec![Vec::new(); n];
    for &g in own.owned() {
        v[f(g)].push(g);
    }
    v.into_iter().filter(|x| !x.is_empty()).collect()
}

fn up<T: Copy>(v: &[T]) -> DevBuf {
    DevBuf::from_slice(v).unwrap()
}

fn dl<T: Copy + Default>(b: &DevBuf, st: &Stream) -> Vec<T> {
    st.sync().unwrap();
    b.to_vec().unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Random distinct picks, then special rows: eight consecutive experts (eight different subbanks of the `e % 8` split), all eight in
/// one third, invalid picks (-1, 999), the 143/144 boundary.
fn picks(r: &mut Rng, rows: usize, d: &MoeDims) -> (Vec<i32>, Vec<u16>) {
    let (e, k) = (d.experts, d.topk);
    let mut pick = Vec::with_capacity(rows * k);
    for _ in 0..rows {
        let mut ids: Vec<i32> = (0..e as i32).collect();
        for i in (1..ids.len()).rev() {
            ids.swap(i, (r.next_u64() % (i as u64 + 1)) as usize);
        }
        pick.extend_from_slice(&ids[..k]);
    }
    let special: [Vec<i32>; 4] = [
        (0..k as i32).map(|j| 200 + j).collect(),
        (0..k as i32).map(|j| 3 * j).collect(),
        (0..k as i32)
            .map(|j| {
                if j == 2 {
                    -1
                } else if j == 5 {
                    999
                } else {
                    74 + j
                }
            })
            .collect(),
        (0..k as i32).map(|j| 140 + j).collect(),
    ];
    for (i, s) in special.iter().enumerate() {
        let row = [rows / 2, rows - 1, 0, rows / 3][i];
        if rows > i {
            pick[row * k..(row + 1) * k].copy_from_slice(s);
        }
    }
    let wts = (0..rows * k).map(|_| f64_to_f16(0.05 + 0.6 * (r.next_u64() >> 40) as f64 / 16777216.0)).collect();
    (pick, wts)
}

struct Case {
    x: DevBuf,
    shared: DevBuf,
    pick: Vec<i32>,
    wts: Vec<u16>,
}

fn case(d: &MoeDims, rows: usize, seed: u64) -> Case {
    let mut r = Rng(seed);
    let xh: Vec<u16> = r.vec(rows * d.hidden, -2.0, 2.0).iter().map(|&v| f64_to_f16(v as f64)).collect();
    let sh = r.vec(rows * d.hidden, -0.5, 0.5);
    let (pick, wts) = picks(&mut r, rows, d);
    Case { x: up(&xh), shared: up(&sh), pick, wts }
}

/// The reference (`out`, `y`) of a case on one path: the single bank (`whole`), or, without one, every expert alone in a one-expert
/// bank (its pairs stage writes the `y` of its own pairs; the reference `y` is assembled pair by pair and combined once).
enum Reference<'a> {
    Whole(&'a MoeExperts),
    PerExpert(&'a [MoeSubbanks]),
}

fn reference(rf: &Reference, d: &MoeDims, c: &Case, rows: usize, batched: bool, st: &Stream) -> (Vec<f32>, Vec<f32>) {
    let p = rows * d.topk;
    let mut out = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
    match rf {
        Reference::Whole(w) => {
            if batched {
                let mut s = PrefillScratch::new(d, rows.max(64)).unwrap();
                s.set_picks(&c.pick, &c.wts).unwrap();
                w.run_prefill(&c.x, Dtype::F16, rows, Some(&c.shared), &mut out, &mut s, st).unwrap();
                (dl(&out, st), dl::<f32>(&s.y_pairs, st)[..p * d.hidden].to_vec())
            } else {
                let mut s = MoeScratch::new(d, 64).unwrap();
                s.set_picks(&c.pick, &c.wts).unwrap();
                w.run(&c.x, Dtype::F16, rows, Some(&c.shared), &mut out, &mut s, true, st).unwrap();
                (dl(&out, st), dl::<f32>(&s.y_pairs, st)[..p * d.hidden].to_vec())
            }
        }
        Reference::PerExpert(banks) => {
            // y of pair q from the one-expert bank of its pick (each bank's pairs stage alone, filler on, decode or batched)
            let mut y = vec![f32::from_bits(NEG_ZERO); p * d.hidden];
            let mut os = OwnScratch::new(d, rows.max(64)).unwrap();
            let mut s = MoeScratch::new(d, 64).unwrap();
            let mut ps = PrefillScratch::new(d, rows.max(64)).unwrap();
            s.set_picks(&c.pick, &c.wts).unwrap();
            ps.set_picks(&c.pick, &c.wts).unwrap();
            let mut done = vec![false; d.experts];
            for &e in &c.pick {
                if e < 0 || e as usize >= d.experts || done[e as usize] {
                    continue;
                }
                done[e as usize] = true;
                let b = &banks[e as usize];
                assert_eq!(b.banks().len(), 1);
                let yb: Vec<f32> = if batched {
                    b.run_prefill_pairs(&c.x, Dtype::F16, rows, &mut ps, &mut os, st).unwrap();
                    dl(&ps.y_pairs, st)
                } else {
                    b.run_pairs(&c.x, Dtype::F16, rows, &mut s, &mut os, st).unwrap();
                    dl(&s.y_pairs, st)
                };
                for q in 0..p {
                    if c.pick[q] == e {
                        y[q * d.hidden..(q + 1) * d.hidden].copy_from_slice(&yb[q * d.hidden..(q + 1) * d.hidden]);
                    }
                }
            }
            let yd = up(&y);
            let pk = up(&c.pick);
            let w = up(&c.wts);
            combine_pairs(&yd, &pk, &w, Some(&c.shared), &mut out, rows, d, st).unwrap();
            (dl(&out, st), y)
        }
    }
}

/// Every rank's subbank pairs stage (poisoned scratch), per-rank `y` checked (held pairs = reference, the rest `-0.0`), the ranks'
/// buffers summed, one combine; `out` must equal the reference. With one all-owned rank, `MoeSubbanks::run`/`run_prefill` too.
fn check_case(rf: (&[f32], &[f32]), ranks: &[&MoeSubbanks], c: &Case, rows: usize, batched: bool, st: &Stream, tag: &str) {
    let (out_ref, y_ref) = rf;
    let d = ranks[0].dims;
    let p = rows * d.topk;
    let cap = rows.max(64);
    let mut bufs: Vec<DevBuf> = Vec::new();
    for (ri, sb) in ranks.iter().enumerate() {
        let mut os = OwnScratch::new(&d, cap).unwrap();
        let raw: Vec<f32> = if batched {
            let mut s = PrefillScratch::new(&d, cap).unwrap();
            s.poison(0x7f, st).unwrap();
            s.set_picks(&c.pick, &c.wts).unwrap();
            if sb.ownership().is_all() {
                let mut o = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
                sb.run_prefill(&c.x, Dtype::F16, rows, Some(&c.shared), &mut o, &mut s, &mut os, st).unwrap();
                assert_eq!(bits(&dl::<f32>(&o, st)), bits(out_ref), "{tag}: run_prefill out");
                s.poison(0x7f, st).unwrap();
                s.set_picks(&c.pick, &c.wts).unwrap();
            }
            sb.run_prefill_pairs(&c.x, Dtype::F16, rows, &mut s, &mut os, st).unwrap();
            dl::<f32>(&s.y_pairs, st)[..p * d.hidden].to_vec()
        } else {
            let mut s = MoeScratch::new(&d, 64).unwrap();
            s.poison(0x7f, st).unwrap();
            s.set_picks(&c.pick, &c.wts).unwrap();
            if sb.ownership().is_all() {
                let mut o = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
                sb.run(&c.x, Dtype::F16, rows, Some(&c.shared), &mut o, &mut s, &mut os, st).unwrap();
                assert_eq!(bits(&dl::<f32>(&o, st)), bits(out_ref), "{tag}: run out");
                s.poison(0x7f, st).unwrap();
                s.set_picks(&c.pick, &c.wts).unwrap();
            }
            sb.run_pairs(&c.x, Dtype::F16, rows, &mut s, &mut os, st).unwrap();
            dl::<f32>(&s.y_pairs, st)[..p * d.hidden].to_vec()
        };
        for q in 0..p {
            let e = c.pick[q];
            let held = e >= 0 && (e as usize) < d.experts && sb.ownership().owns(e as usize);
            let row = &raw[q * d.hidden..(q + 1) * d.hidden];
            if held {
                assert_eq!(bits(row), bits(&y_ref[q * d.hidden..(q + 1) * d.hidden]), "{tag} rank {ri} pair {q}: y");
            } else {
                assert!(row.iter().all(|v| v.to_bits() == NEG_ZERO), "{tag} rank {ri} pair {q}: filler is not -0.0");
            }
        }
        bufs.push(up(&raw));
    }
    // the exchange: rank 0's buffer += every other rank's (one IEEE add, no flush to zero)
    let (first, rest) = bufs.split_at_mut(1);
    for b in rest.iter() {
        nccl::add_f32_into(&mut first[0], 0, b, 0, p * d.hidden, st).unwrap();
    }
    let sum = &first[0];
    let (pk, w) = (up(&c.pick), up(&c.wts));
    let mut out = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
    combine_pairs(sum, &pk, &w, Some(&c.shared), &mut out, rows, &d, st).unwrap();
    let ys: Vec<f32> = dl(sum, st);
    for q in 0..p {
        let e = c.pick[q];
        if e >= 0 && (e as usize) < d.experts {
            assert_eq!(
                bits(&ys[q * d.hidden..(q + 1) * d.hidden]),
                bits(&y_ref[q * d.hidden..(q + 1) * d.hidden]),
                "{tag}: summed y {q}"
            );
        }
    }
    assert_eq!(bits(&dl::<f32>(&out, st)), bits(out_ref), "{tag}: out");
}

#[test]
fn artificial_subbanks_equal_the_single_bank_byte_for_byte() {
    gpu!();
    let Some(dir) = dir_of("REXL3_GLM_DIR", "models/r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121") else {
        eprintln!("no GLM pack (set REXL3_GLM_DIR): not run");
        return;
    };
    let ck = scan(&dir, false).expect("scan the GLM pack");
    let d = MoeDims::GLM53;
    let st = Stream::new().unwrap();
    for l in [6usize, 45] {
        let lg = load_layer(&ck, l, d.experts);
        let whole = MoeExperts::from_groups(&lg.gate, &lg.up, &lg.down, d).unwrap();
        let all = ExpertOwnership::all(d.experts);
        let r: Vec<ExpertOwnership> = (0..2).map(|k| ExpertOwnership::range(d.experts, k, 2).unwrap()).collect();
        // (name, ranks of subbanks)
        let configs: Vec<(&str, Vec<MoeSubbanks>)> = vec![
            ("thirds-scattered", vec![subbanks(&lg, d, &all, &split_by(&all, 3, |e| ((e * 7 + e / 5) % 3) as usize), &[])]),
            ("skewed-1+287", vec![subbanks(&lg, d, &all, &[vec![77], (0..288).filter(|&e| e != 77).collect()], &[])]),
            ("skewed-287+1", vec![subbanks(&lg, d, &all, &[(0..288).filter(|&e| e != 200).collect(), vec![200]], &[])]),
            ("eight-interleaved", vec![subbanks(&lg, d, &all, &split_by(&all, 8, |e| (e % 8) as usize), &[])]),
            ("split-gate-up", vec![subbanks(&lg, d, &all, &split_by(&all, 2, |e| (e % 2) as usize), &[1])]),
            ("R2-x-thirds", r.iter().map(|o| subbanks(&lg, d, o, &split_by(o, 3, |e| (e % 3) as usize), &[])).collect()),
            ("region-built-uniform", vec![MoeSubbanks::from_groups(&lg.gate, &lg.up, &lg.down, d, all.clone()).unwrap()]),
        ];
        drop(lg);
        assert_eq!(configs[6].1[0].banks().len(), 1, "a uniform layer is one subbank");
        for rows in ROWS {
            let c = case(&d, rows, 31 * l as u64 + rows as u64);
            let t = std::time::Instant::now();
            for batched in [false, true] {
                let (out_ref, y_ref) = reference(&Reference::Whole(&whole), &d, &c, rows, batched, &st);
                for (name, ranks) in &configs {
                    let tag = format!("layer {l} {} rows {rows} {name}", if batched { "batched" } else { "decode" });
                    if batched && *name == "split-gate-up" {
                        // the forced split is a decode-only launch; its batched stage is refused, not silently fused
                        let mut s = PrefillScratch::new(&d, 64).unwrap();
                        s.set_picks(&c.pick, &c.wts).unwrap();
                        let mut os = OwnScratch::new(&d, 64).unwrap();
                        assert!(ranks[0].run_prefill_pairs(&c.x, Dtype::F16, rows, &mut s, &mut os, &st).is_err(), "{tag}");
                        continue;
                    }
                    let rr: Vec<&MoeSubbanks> = ranks.iter().collect();
                    check_case((&out_ref, &y_ref), &rr, &c, rows, batched, &st, &tag);
                }
            }
            eprintln!(
                "layer {l} rows {rows}: {} subbank configs, decode + batched: byte-identical ({:.2} s)",
                configs.len(),
                t.elapsed().as_secs_f64()
            );
        }
        let launches: Vec<(&str, usize)> = configs.iter().map(|(n, r)| (*n, r[0].decode_launches())).collect();
        eprintln!("layer {l}: decode launches per rank {launches:?}");
    }
}

/// REAL mixed layers of the RED-SNOW pack: the subbank path (subbanks by key, built through the loader's region constructor) against
/// every expert run alone in a one-expert bank, byte for byte, decode and batched, rows 1, 2, 5, 22, 64; R = 1 and R = 2.
#[test]
fn real_mixed_layers_equal_one_expert_banks_byte_for_byte() {
    gpu!();
    let Some(dir) = dir_of("REXL3_REDSNOW_DIR", "models/vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw") else {
        eprintln!("no RED-SNOW pack (set REXL3_REDSNOW_DIR): not run");
        return;
    };
    let layers: Vec<usize> = std::env::var("REXL3_REDSNOW_LAYERS")
        .ok()
        .map(|s| s.split(',').map(|x| x.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![4, 7]);
    let ck = scan(&dir, false).expect("scan the RED-SNOW pack");
    let d = MoeDims::GLM53;
    let st = Stream::new().unwrap();
    for l in layers {
        let lg = load_layer(&ck, l, d.experts);
        let keys: Vec<SubbankKey> =
            (0..d.experts).map(|e| SubbankKey::of_groups(&lg.gate[e], &lg.up[e], &lg.down[e]).unwrap()).collect();
        let mut hist: std::collections::BTreeMap<SubbankKey, usize> = Default::default();
        for k in &keys {
            *hist.entry(*k).or_default() += 1;
        }
        eprintln!(
            "layer {l}: {} subbanks {:?}",
            hist.len(),
            hist.iter().map(|(k, n)| ((k.bits_gate, k.bits_up, k.bits_down), *n)).collect::<Vec<_>>()
        );
        let has_split = keys.iter().any(|k| k.bits_gate != k.bits_up);
        // the grouped constructor builds one band per (gate/up, down) width pair, so mixed down widths are accepted; it refuses a
        // layer only when an expert's gate and up widths differ (the grouped kernels fuse gate and up), so it refuses exactly the
        // split layers
        assert_eq!(
            MoeExperts::from_groups(&lg.gate, &lg.up, &lg.down, d).is_err(),
            has_split,
            "layer {l}: from_groups must refuse exactly the layers whose experts split gate and up widths"
        );
        let all = ExpertOwnership::all(d.experts);
        let whole = MoeSubbanks::from_groups(&lg.gate, &lg.up, &lg.down, d, all).unwrap();
        let ranks: Vec<MoeSubbanks> = (0..2)
            .map(|k| {
                MoeSubbanks::from_groups(&lg.gate, &lg.up, &lg.down, d, ExpertOwnership::range(d.experts, k, 2).unwrap()).unwrap()
            })
            .collect();
        let per_rank: Vec<usize> = ranks.iter().map(|r| r.expert_bytes()).collect();
        eprintln!(
            "layer {l}: expert bytes all {} = rank0 {} + rank1 {} (skew {:.3}); decode launches {} (single bank 6)",
            whole.expert_bytes(),
            per_rank[0],
            per_rank[1],
            per_rank[0] as f64 / per_rank[1] as f64,
            whole.decode_launches()
        );
        assert_eq!(per_rank[0] + per_rank[1], whole.expert_bytes());
        // one-expert banks: the reference
        let singles: Vec<MoeSubbanks> = (0..d.experts as u32)
            .map(|e| {
                let o = ExpertOwnership::from_ids(d.experts, vec![e]).unwrap();
                MoeSubbanks::from_groups(&lg.gate, &lg.up, &lg.down, d, o).unwrap()
            })
            .collect();
        // an INDEPENDENT reference for the gate != up experts (and a fused control expert of the same gate width): the expert as a
        // dense gated MLP of three EXL3 linears (`SharedExpert`, the row-wise linear kernels, not the grouped ones), compared with
        // a tolerance: the split launch must compute the real expert, not only agree with a bank built by the same constructor
        let split_ids: Vec<usize> = (0..d.experts).filter(|&e| keys[e].bits_gate != keys[e].bits_up).collect();
        let mut probes: Vec<(usize, &str, SharedExpert)> = Vec::new();
        if let Some(&s0) = split_ids.first() {
            let ctl = (0..d.experts)
                .find(|&e| {
                    keys[e].fused_gate_up() && keys[e].bits_gate == keys[s0].bits_gate && keys[e].bits_down == keys[s0].bits_down
                })
                .unwrap();
            for (e, what) in split_ids.iter().map(|&e| (e, "split")).chain([(ctl, "fused control")]) {
                let w = |g: &LoadedGroup| gb_cuda::exl3_linear::Exl3Weights::from_loaded(g).unwrap();
                probes.push((e, what, SharedExpert::new(w(&lg.gate[e]), w(&lg.up[e]), w(&lg.down[e]), d.limit).unwrap()));
            }
            eprintln!("layer {l}: gate != up experts {split_ids:?} (keys {:?}); fused control {ctl}", keys[s0]);
        }
        drop(lg);
        for rows in ROWS {
            let mut c = case(&d, rows, 777 * l as u64 + rows as u64);
            // route every probe expert in the last row (slots 1, 2, ...), so the split launch runs at every row count
            for (j, (e, _, _)) in probes.iter().enumerate() {
                let row = &mut c.pick[(rows - 1) * d.topk..rows * d.topk];
                if !row.contains(&(*e as i32)) {
                    row[1 + j] = *e as i32;
                }
            }
            let t = std::time::Instant::now();
            for batched in [false, true] {
                if batched && has_split {
                    eprintln!("layer {l}: a split gate/up subbank: batched path refused (not built)");
                    continue;
                }
                let (out_ref, y_ref) = reference(&Reference::PerExpert(&singles), &d, &c, rows, batched, &st);
                if !batched {
                    for (e, what, mlp) in &probes {
                        let mut ss = SharedScratch::new(mlp, 64).unwrap();
                        let mut o = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
                        mlp.forward(&c.x, Dtype::F16, rows, &mut o, &mut ss, &st).unwrap();
                        let o: Vec<f32> = dl(&o, &st);
                        let mut n = 0;
                        for q in (0..rows * d.topk).filter(|&q| c.pick[q] == *e as i32) {
                            let r = q / d.topk;
                            let (a, b) = (&y_ref[q * d.hidden..(q + 1) * d.hidden], &o[r * d.hidden..(r + 1) * d.hidden]);
                            let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>().sqrt();
                            let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt();
                            let rel = num / den.max(1e-30);
                            if n == 0 {
                                eprintln!("layer {l} rows {rows}: {what} expert {e} vs a dense EXL3 MLP: rel L2 {rel:.2e}");
                            }
                            assert!(rel < 1e-2, "layer {l} rows {rows} {what} expert {e} pair {q}: rel L2 {rel:.3e}");
                            n += 1;
                        }
                        assert!(n > 0, "layer {l} rows {rows}: expert {e} was routed");
                    }
                }
                let tag = format!("RED-SNOW layer {l} {} rows {rows}", if batched { "batched" } else { "decode" });
                check_case((&out_ref, &y_ref), &[&whole], &c, rows, batched, &st, &format!("{tag} R=1"));
                let rr: Vec<&MoeSubbanks> = ranks.iter().collect();
                check_case((&out_ref, &y_ref), &rr, &c, rows, batched, &st, &format!("{tag} R=2"));
            }
            eprintln!(
                "RED-SNOW layer {l} rows {rows}: R = 1, 2, {}: byte-identical ({:.2} s)",
                if has_split { "decode" } else { "decode + batched" },
                t.elapsed().as_secs_f64()
            );
        }
    }
}

/// The batched pairs stage of REAL RED-SNOW layers at chunk sizes past one decode window (129, 200, 300 rows), R = 1 and R = 2, against
/// the decode pairs stage of the same rows in 64-row windows, PAIR BY PAIR: `run_prefill_pairs_windowed` (split-width subbanks through
/// decode windows of 128 and of 7 rows, fused ones on the batched kernels) and, on a layer with no split subbank, `run_prefill_pairs` (the
/// batched kernels alone: decode == batched per pair at the pack's 4/5/6-bit widths). Every valid pair's `y` equals the decode `y` bit for
/// bit, every pair no subbank of the rank holds is `-0.0`, and the combine equals the decode reference's. Layers: `REXL3_REDSNOW_PF_LAYERS`
/// (default 13, the one with a gate != up expert, and 4).
#[test]
fn real_layers_batched_pairs_equal_the_decode_pairs_past_one_window() {
    gpu!();
    let Some(dir) = dir_of("REXL3_REDSNOW_DIR", "models/vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw") else {
        eprintln!("no RED-SNOW pack (set REXL3_REDSNOW_DIR): not run");
        return;
    };
    let layers: Vec<usize> = std::env::var("REXL3_REDSNOW_PF_LAYERS")
        .ok()
        .map(|s| s.split(',').map(|x| x.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![13, 4]);
    let ck = scan(&dir, false).expect("scan the RED-SNOW pack");
    let d = MoeDims::GLM53;
    let st = Stream::new().unwrap();
    for l in layers {
        let lg = load_layer(&ck, l, d.experts);
        let whole = MoeSubbanks::from_groups(&lg.gate, &lg.up, &lg.down, d, ExpertOwnership::all(d.experts)).unwrap();
        let ranks: Vec<MoeSubbanks> = (0..2)
            .map(|k| {
                MoeSubbanks::from_groups(&lg.gate, &lg.up, &lg.down, d, ExpertOwnership::range(d.experts, k, 2).unwrap()).unwrap()
            })
            .collect();
        let split: Vec<usize> = whole
            .banks()
            .iter()
            .filter(|b| !b.subbank_key().fused_gate_up())
            .flat_map(|b| b.ownership().unwrap().owned().iter().map(|&e| e as usize))
            .collect();
        eprintln!("layer {l}: {} subbanks, split gate/up experts {split:?}", whole.banks().len());
        assert_eq!(whole.has_split_gate_up(), !split.is_empty());
        drop(lg);
        for rows in [129usize, 200, 300] {
            let mut r = Rng(9000 + 31 * l as u64 + rows as u64);
            let xh: Vec<u16> = r.vec(rows * d.hidden, -2.0, 2.0).iter().map(|&v| f64_to_f16(v as f64)).collect();
            let sh = r.vec(rows * d.hidden, -0.5, 0.5);
            let (mut pick, wts) = picks(&mut r, rows, &d);
            // every split expert routed in several rows of every 64-row window, the last row of a 128-row window and the chunk's last row
            for (j, &e) in split.iter().enumerate() {
                for row in (0..rows).step_by(37).chain([127usize, 128, rows - 1]).filter(|&x| x < rows) {
                    let s = &mut pick[row * d.topk..(row + 1) * d.topk];
                    if !s.contains(&(e as i32)) {
                        s[(3 + j) % d.topk] = e as i32;
                    }
                }
            }
            let p = rows * d.topk;
            let (x, shared) = (up(&xh), up(&sh));
            // the decode reference: the whole subbank set's decode pairs stage in 64-row windows, and its combine
            let mut y_ref = vec![0f32; p * d.hidden];
            {
                let mut s = MoeScratch::new(&d, 64).unwrap();
                let mut os = OwnScratch::new(&d, 64).unwrap();
                let mut a = 0;
                while a < rows {
                    let n = (rows - a).min(64);
                    s.set_picks(&pick[a * d.topk..(a + n) * d.topk], &wts[a * d.topk..(a + n) * d.topk]).unwrap();
                    let xw = up(&xh[a * d.hidden..(a + n) * d.hidden]);
                    whole.run_pairs(&xw, Dtype::F16, n, &mut s, &mut os, &st).unwrap();
                    let yw: Vec<f32> = dl(&s.y_pairs, &st);
                    y_ref[a * d.topk * d.hidden..(a + n) * d.topk * d.hidden].copy_from_slice(&yw[..n * d.topk * d.hidden]);
                    a += n;
                }
            }
            let (pk, w) = (up(&pick), up(&wts));
            let mut out_ref = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
            combine_pairs(&up(&y_ref), &pk, &w, Some(&shared), &mut out_ref, rows, &d, &st).unwrap();
            let out_ref: Vec<f32> = dl(&out_ref, &st);
            let valid = |q: usize| pick[q] >= 0 && (pick[q] as usize) < d.experts;
            for window in [128usize, 7] {
                let mut sums: Vec<f32> = vec![-0.0; p * d.hidden];
                for (ri, sb) in std::iter::once(&whole).chain(ranks.iter()).enumerate() {
                    let mut ps = PrefillScratch::new(&d, rows).unwrap();
                    let mut s = MoeScratch::new(&d, window).unwrap();
                    let mut os = OwnScratch::new(&d, rows).unwrap();
                    ps.poison(0x7f, &st).unwrap();
                    s.poison(0x7f, &st).unwrap();
                    ps.set_picks(&pick, &wts).unwrap();
                    sb.run_prefill_pairs_windowed(&x, Dtype::F16, rows, &mut ps, &mut s, &mut os, &st).unwrap();
                    let y: Vec<f32> = dl::<f32>(&ps.y_pairs, &st)[..p * d.hidden].to_vec();
                    let tag = format!(
                        "layer {l} rows {rows} window {window} {}",
                        if ri == 0 { "R=1".into() } else { format!("rank {}", ri - 1) }
                    );
                    for q in 0..p {
                        let row = &y[q * d.hidden..(q + 1) * d.hidden];
                        if valid(q) && sb.ownership().owns(pick[q] as usize) {
                            assert_eq!(
                                bits(row),
                                bits(&y_ref[q * d.hidden..(q + 1) * d.hidden]),
                                "{tag} pair {q} (expert {}): y",
                                pick[q]
                            );
                        } else {
                            assert!(row.iter().all(|v| v.to_bits() == NEG_ZERO), "{tag} pair {q}: filler is not -0.0");
                        }
                    }
                    if ri == 0 {
                        let mut o = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
                        combine_pairs(&ps.y_pairs, &pk, &w, Some(&shared), &mut o, rows, &d, &st).unwrap();
                        assert_eq!(bits(&dl::<f32>(&o, &st)), bits(&out_ref), "{tag}: out");
                    } else {
                        for (a, b) in sums.iter_mut().zip(&y) {
                            *a += *b;
                        }
                    }
                }
                let mut o = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
                combine_pairs(&up(&sums), &pk, &w, Some(&shared), &mut o, rows, &d, &st).unwrap();
                assert_eq!(bits(&dl::<f32>(&o, &st)), bits(&out_ref), "layer {l} rows {rows} window {window} R=2 summed: out");
            }
            if split.is_empty() {
                // the batched kernels alone (no window): decode == batched per pair at these widths
                let mut ps = PrefillScratch::new(&d, rows).unwrap();
                let mut os = OwnScratch::new(&d, rows).unwrap();
                ps.poison(0x7f, &st).unwrap();
                ps.set_picks(&pick, &wts).unwrap();
                whole.run_prefill_pairs(&x, Dtype::F16, rows, &mut ps, &mut os, &st).unwrap();
                let y: Vec<f32> = dl::<f32>(&ps.y_pairs, &st)[..p * d.hidden].to_vec();
                for q in (0..p).filter(|&q| valid(q)) {
                    assert_eq!(
                        bits(&y[q * d.hidden..(q + 1) * d.hidden]),
                        bits(&y_ref[q * d.hidden..(q + 1) * d.hidden]),
                        "layer {l} rows {rows} batched pair {q}"
                    );
                }
            } else {
                // the batched-only entry still refuses a split subbank by name
                let mut ps = PrefillScratch::new(&d, rows).unwrap();
                let mut os = OwnScratch::new(&d, rows).unwrap();
                ps.set_picks(&pick, &wts).unwrap();
                assert_eq!(
                    whole.run_prefill_pairs(&x, Dtype::F16, rows, &mut ps, &mut os, &st).unwrap_err(),
                    gb_cuda::moe_subbanks::SPLIT_PREFILL
                );
            }
            eprintln!(
                "RED-SNOW layer {l} rows {rows}: batched pairs == decode pairs, R = 1 and 2, windows 128 and 7: byte-identical"
            );
        }
    }
}

/// REAL mixed-width layers of the RED-SNOW pack under P2 (`REXL3_REDSNOW_DIR`: the 2.49bpw pack's widths are 2 to 4 bits, the 4.91bpw
/// pack's 4 to 6 with layer 13's split gate/up subbank; layers `REXL3_REDSNOW_P2_LAYERS`, default 4, 13, 16 and 44; singleton and split
/// subbanks' experts routed in several rows). A layer a P2 prefill does not take is checked to be refused by name, a uniform layer is
/// skipped.
#[test]
fn real_mixed_layers_run_p2_per_subbank() {
    use gb_cuda::glm_p2_moe;
    gpu!();
    let Some(dir) = dir_of("REXL3_REDSNOW_DIR", "models/vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw") else {
        eprintln!("no RED-SNOW pack (set REXL3_REDSNOW_DIR): not run");
        return;
    };
    let ck = scan(&dir, false).expect("scan the RED-SNOW pack");
    let d = MoeDims::GLM53;
    let st = Stream::new().unwrap();
    let rows = 300usize;
    let layers: Vec<usize> = std::env::var("REXL3_REDSNOW_P2_LAYERS")
        .ok()
        .map(|s| s.split(',').map(|x| x.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![4, 13, 16, 44]);
    for l in layers {
        let lg = load_layer(&ck, l, d.experts);
        let whole = MoeSubbanks::from_groups(&lg.gate, &lg.up, &lg.down, d, ExpertOwnership::all(d.experts)).unwrap();
        let keys: Vec<String> = whole
            .banks()
            .iter()
            .map(|b| {
                let k = b.subbank_key();
                format!("{}/{}/{}x{}", k.bits_gate, k.bits_up, k.bits_down, b.experts_local())
            })
            .collect();
        eprintln!("layer {l}: {} subbanks (gate/up/down bits x experts): {}", whole.banks().len(), keys.join(" "));
        if let Err(why) = glm_p2_moe::supported_subbanks(&whole) {
            eprintln!("layer {l}: refused by name under P2: {why}");
            assert!(why.starts_with("subbank "), "{why}");
            continue;
        }
        if whole.banks().len() == 1 {
            eprintln!("layer {l} is uniform on this pack: not a mixed-width case");
            continue;
        }
        // singleton subbanks' experts and every split subbank's experts, routed in several rows
        let singletons: Vec<usize> = whole
            .banks()
            .iter()
            .filter(|b| b.experts_local() <= 2 || !b.subbank_key().fused_gate_up())
            .flat_map(|b| b.ownership().unwrap().owned().iter().map(|&e| e as usize))
            .collect();
        let ranks: Vec<MoeSubbanks> = (0..2)
            .map(|k| {
                MoeSubbanks::from_groups(&lg.gate, &lg.up, &lg.down, d, ExpertOwnership::range(d.experts, k, 2).unwrap()).unwrap()
            })
            .collect();
        if singletons.is_empty() {
            eprintln!(
                "layer {l}: no singleton subbank experts to force-route (the 2.49bpw pack's layers 4, 16 and 44 have some)"
            );
        }
        let mut r = Rng(7000 + l as u64);
        let xh: Vec<u16> = r.vec(rows * d.hidden, -2.0, 2.0).iter().map(|&v| f64_to_f16(v as f64)).collect();
        let sh = r.vec(rows * d.hidden, -0.5, 0.5);
        let (mut pick, wts) = picks(&mut r, rows, &d);
        for (j, &e) in singletons.iter().enumerate() {
            for row in (0..rows).step_by(37).chain([rows - 1]) {
                let s = &mut pick[row * d.topk..(row + 1) * d.topk];
                if !s.contains(&(e as i32)) {
                    s[(3 + j) % d.topk] = e as i32;
                }
            }
        }
        let (p, h) = (rows * d.topk, d.hidden);
        let valid = |q: usize| pick[q] >= 0 && (pick[q] as usize) < d.experts;
        let (shared, pk, w) = (up(&sh), up(&pick), up(&wts));
        let mut ps = PrefillScratch::new(&d, rows).unwrap();
        let mut os = OwnScratch::new(&d, rows).unwrap();
        let mut sd = MoeScratch::new(&d, 128).unwrap();
        // P2 on the given rows (calls of `chunk` rows, scratch poisoned first): (out, y of every pair)
        let mut p2 = |sel: &[usize], chunk: usize, ps: &mut PrefillScratch| -> (Vec<f32>, Vec<f32>) {
            let (mut out, mut y) = (Vec::new(), Vec::new());
            for part in sel.chunks(chunk) {
                let n = part.len();
                let gx: Vec<u16> = part.iter().flat_map(|&q| xh[q * h..(q + 1) * h].to_vec()).collect();
                let gp: Vec<i32> = part.iter().flat_map(|&q| pick[q * d.topk..(q + 1) * d.topk].to_vec()).collect();
                let gw: Vec<u16> = part.iter().flat_map(|&q| wts[q * d.topk..(q + 1) * d.topk].to_vec()).collect();
                let gs: Vec<f32> = part.iter().flat_map(|&q| sh[q * h..(q + 1) * h].to_vec()).collect();
                ps.poison(0x7f, &st).unwrap();
                ps.set_picks(&gp, &gw).unwrap();
                let mut o = DevBuf::zeroed(n * h * 4).unwrap();
                glm_p2_moe::run_subbanks(&whole, &up(&gx), Dtype::F16, n, Some(&up(&gs)), &mut o, ps, &mut os, &mut sd, &st)
                    .unwrap();
                out.extend(dl::<f32>(&o, &st));
                y.extend_from_slice(&dl::<f32>(&ps.y_pairs, &st)[..n * d.topk * h]);
            }
            (out, y)
        };
        let all: Vec<usize> = (0..rows).collect();
        let (out0, y0) = p2(&all, rows, &mut ps);
        assert!(out0.iter().all(|v| v.is_finite()) && out0.iter().any(|&v| v != 0.0), "layer {l}: degenerate P2 output");
        // determinism: run to run (poisoned scratch)
        let (out1, y1) = p2(&all, rows, &mut ps);
        assert_eq!(bits(&out1), bits(&out0), "layer {l}: P2 out, run to run");
        for q in (0..p).filter(|&q| valid(q)) {
            assert_eq!(bits(&y1[q * h..(q + 1) * h]), bits(&y0[q * h..(q + 1) * h]), "layer {l}: P2 y of pair {q}, run to run");
        }
        // chunked == whole, row independence from chunk width and padding (calls of 1, 7, 16, 129 rows), and from the other rows
        // (the rows reversed, so every row has other neighbours and another place in every tile)
        let check = |what: &str, sel: &[usize], out: &[f32], y: &[f32]| {
            for (j, &q) in sel.iter().enumerate() {
                assert_eq!(bits(&out[j * h..(j + 1) * h]), bits(&out0[q * h..(q + 1) * h]), "layer {l} {what}: out of row {q}");
                for k in 0..d.topk {
                    let (pj, pq) = (j * d.topk + k, q * d.topk + k);
                    if valid(pq) {
                        assert_eq!(
                            bits(&y[pj * h..(pj + 1) * h]),
                            bits(&y0[pq * h..(pq + 1) * h]),
                            "layer {l} {what}: y of row {q} slot {k} (expert {})",
                            pick[pq]
                        );
                    }
                }
            }
        };
        for chunk in [7usize, 16, 129] {
            let (o, y) = p2(&all, chunk, &mut ps);
            check(&format!("calls of {chunk} rows"), &all, &o, &y);
        }
        let rev: Vec<usize> = (0..rows).rev().collect();
        let (o, y) = p2(&rev, rows, &mut ps);
        check("rows reversed", &rev, &o, &y);
        let lone: Vec<usize> = (0..rows).step_by(37).chain([rows - 1]).collect();
        let (o, y) = p2(&lone, 1, &mut ps);
        check("one row a call", &lone, &o, &y);
        let (o, y) = p2(&lone, lone.len(), &mut ps);
        check("only the sampled rows in one call", &lone, &o, &y);
        // two ranks of the range ownership: each its share's subbanks and its own group's part, then the parts' sum == one node
        {
            let x = up(&xh);
            let mut parts = [DevBuf::zeroed(rows * h * 4).unwrap(), DevBuf::zeroed(rows * h * 4).unwrap()];
            for (k, sb) in ranks.iter().enumerate() {
                ps.poison(0x33 + k as u8, &st).unwrap();
                ps.set_picks(&pick, &wts).unwrap();
                let ord = glm_p2_moe::DEFAULT_ORDER;
                glm_p2_moe::pairs_subbanks_rank(sb, &x, Dtype::F16, rows, &mut ps, &mut os, &mut sd, ord, &st, &mut |_| Ok(()))
                    .unwrap();
                let mode = glm_p2_moe::CombineMode::Own(k);
                glm_p2_moe::combine_dims(mode, &ps.y_pairs, &pk, &w, None, None, &mut parts[k], rows, &d, &st).unwrap();
                let yr: Vec<f32> = dl::<f32>(&ps.y_pairs, &st)[..p * h].to_vec();
                for q in (0..p).filter(|&q| valid(q) && sb.ownership().owns(pick[q] as usize)) {
                    assert_eq!(bits(&yr[q * h..(q + 1) * h]), bits(&y0[q * h..(q + 1) * h]), "layer {l} rank {k}: y of pair {q}");
                }
            }
            let mut o = DevBuf::zeroed(rows * h * 4).unwrap();
            let pp = Some((&parts[0], &parts[1]));
            let mode = glm_p2_moe::CombineMode::Parts;
            glm_p2_moe::combine_dims(mode, &ps.y_pairs, &pk, &w, Some(&shared), pp, &mut o, rows, &d, &st).unwrap();
            assert_eq!(bits(&dl::<f32>(&o, &st)), bits(&out0), "layer {l}: two ranks' P2 out != one node's");
        }
        // closeness: P1's subbank path (the decode path's bits; a split subbank through its decode windows) and P2, each against the f64
        // reference of the exact dequantized experts
        ps.poison(0x7f, &st).unwrap();
        ps.set_picks(&pick, &wts).unwrap();
        whole.run_prefill_pairs_windowed(&up(&xh), Dtype::F16, rows, &mut ps, &mut sd, &mut os, &st).unwrap();
        let y_p1: Vec<f32> = dl::<f32>(&ps.y_pairs, &st)[..p * h].to_vec();
        let mut o_p1 = DevBuf::zeroed(rows * h * 4).unwrap();
        combine_pairs(&ps.y_pairs, &pk, &w, Some(&shared), &mut o_p1, rows, &d, &st).unwrap();
        let o_p1: Vec<f32> = dl(&o_p1, &st);
        let sample: Vec<usize> = (0..rows).step_by(37).take(6).collect();
        let lim = d.limit as f64;
        let mut yref = vec![0f64; sample.len() * d.topk * h];
        let experts: std::collections::BTreeSet<i32> =
            sample.iter().flat_map(|&q| pick[q * d.topk..(q + 1) * d.topk].to_vec()).collect();
        for &e in experts.iter().filter(|&&e| e >= 0 && (e as usize) < d.experts) {
            let e = e as usize;
            let (wg, wu, wd) = (weights(&lg.gate[e], &st), weights(&lg.up[e], &st), weights(&lg.down[e], &st));
            for (j, &q) in sample.iter().enumerate() {
                let x: Vec<f64> = (0..h).map(|i| f16_to_f64(xh[q * h + i])).collect();
                for k in 0..d.topk {
                    if pick[q * d.topk + k] == e as i32 {
                        let y = expert_ref([&lg.gate[e], &lg.up[e], &lg.down[e]], [&wg, &wu, &wd], &x, lim);
                        yref[(j * d.topk + k) * h..(j * d.topk + k + 1) * h].copy_from_slice(&y);
                    }
                }
            }
        }
        let mut oref = vec![0f64; sample.len() * h];
        let (mut ys1, mut ys2, mut os1, mut os2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut yr = Vec::new();
        for (j, &q) in sample.iter().enumerate() {
            for i in 0..h {
                oref[j * h + i] = sh[q * h + i] as f64;
            }
            for k in 0..d.topk {
                let pq = q * d.topk + k;
                if !valid(pq) {
                    continue;
                }
                let wk = f16_to_f64(wts[pq]);
                for i in 0..h {
                    oref[j * h + i] += wk * yref[(j * d.topk + k) * h + i];
                }
                yr.extend_from_slice(&yref[(j * d.topk + k) * h..(j * d.topk + k + 1) * h]);
                ys1.extend_from_slice(&y_p1[pq * h..(pq + 1) * h]);
                ys2.extend_from_slice(&y0[pq * h..(pq + 1) * h]);
            }
            os1.extend_from_slice(&o_p1[q * h..(q + 1) * h]);
            os2.extend_from_slice(&out0[q * h..(q + 1) * h]);
        }
        let (m1, r1, _, k1) = err(&yr, &ys1);
        let (m2, r2, _, k2) = err(&yr, &ys2);
        let (_, or1, _, _) = err(&oref, &os1);
        let (_, or2, _, _) = err(&oref, &os2);
        eprintln!(
            "layer {l}: {} sampled rows, {} pairs vs f64: y rms P1 {r1:.2e} (max {m1:.2e}, shrink {k1:+.1e}) | P2 {r2:.2e} (max {m2:.2e}, shrink {k2:+.1e}); out rms P1 {or1:.2e} | P2 {or2:.2e}",
            sample.len(),
            yr.len() / h
        );
        // the P2 contract's gate (docs/GLM_PREFILL_P2.md section 6, `p2_experts_are_as_close_to_fp64_as_the_decode_path`)
        assert!(r2 <= 1.5 * r1 + 1e-7 && r2 < 1e-3, "layer {l}: P2 y rms {r2:e} against P1's {r1:e}");
        assert!(or2 <= 1.5 * or1 + 1e-7, "layer {l}: P2 out rms {or2:e} against P1's {or1:e}");
        eprintln!(
            "RED-SNOW layer {l}: P2 per subbank ({} subbanks{}): deterministic, chunked (7/16/129 rows) == whole, reversed rows == whole, one row a call == whole, two ranks == one node, bit for bit; within the P2 fp64 gate",
            whole.banks().len(),
            if whole.has_split_gate_up() { ", a split gate/up one on the decode pairs" } else { "" }
        );
    }
}
