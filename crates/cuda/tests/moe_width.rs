//! The WIDTH SPLIT of the routed experts (`src/moe_width.rs`): two ranks, each holding its half of EVERY expert's width (gate/up
//! columns, down K rows), run their pairs stages; the host sums the two ranks' K-split partials (the exchange: `a + b`, IEEE round to
//! nearest, the two-operand sum the collective makes) and each rank's `finish` must give the single node's output BYTE FOR BYTE: decode
//! windows (1 to 64 rows) and the batched path (a prefill chunk, a split gate/up subbank through its decode windows), on mixed widths
//! with a split gate/up subbank.
//!
//! `synthetic_mixed_widths_equal_the_whole_width` needs no pack (32 synthetic experts of four keys at the GLM shape).
//! `real_mixed_layers_equal_the_whole_width` loads REAL mixed layers of the RED-SNOW 4.91bpw pack (`REXL3_REDSNOW_DIR`, default
//! `~/models/vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw`; layers `REXL3_REDSNOW_WIDTH_LAYERS`, default 4 and 13, layer 13 with its
//! split gate/up subbank): the same equality, then the P2 prefill of the two width shares (`pairs_subbanks_rank`, `CombineMode::All`,
//! `Parts`): deterministic, chunked (1 / 7 / 129 rows) == whole, rows reversed == whole, and as close to P1 as the one node's P2.
//!
//!     cargo test --release -p gb-cuda --test moe_width -- --test-threads=1 --nocapture
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments)]
mod common;

use common::exl3_synth::synth_group;
use common::*;
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::moe::*;
use gb_cuda::moe_prefill::PrefillScratch;
use gb_cuda::moe_width::{group_cols, group_rows, width_plans};
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::checkpoint::{scan, Checkpoint, LoadedGroup};
use gb_exl3::f16::f64_to_f16;
use gb_exl3::Codebook;

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

fn dl<T: Copy + Default>(b: &DevBuf, st: &Stream) -> Vec<T> {
    st.sync().unwrap();
    b.to_vec().unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

struct Layer {
    gate: Vec<LoadedGroup>,
    up: Vec<LoadedGroup>,
    down: Vec<LoadedGroup>,
}

/// Random distinct picks per row, then an invalid pick (-1, then E + 1) in two rows.
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
    if rows > 2 {
        pick[(rows / 2) * k + 2] = -1;
        pick[(rows - 1) * k + 5] = e as i32 + 1;
    }
    let wts = (0..rows * k).map(|_| f64_to_f16(0.05 + 0.6 * (r.next_u64() >> 40) as f64 / 16777216.0)).collect();
    (pick, wts)
}

struct Case {
    xh: Vec<u16>,
    sh: Vec<f32>,
    pick: Vec<i32>,
    wts: Vec<u16>,
}

fn case(d: &MoeDims, rows: usize, seed: u64) -> Case {
    let mut r = Rng(seed);
    let xh: Vec<u16> = r.vec(rows * d.hidden, -2.0, 2.0).iter().map(|&v| f64_to_f16(v as f64)).collect();
    let sh = r.vec(rows * d.hidden, -0.5, 0.5);
    let (pick, wts) = picks(&mut r, rows, d);
    Case { xh, sh, pick, wts }
}

/// The single node's output of `c` (decode window, or the batched path).
fn whole_out(whole: &MoeSubbanks, d: &MoeDims, c: &Case, rows: usize, batched: bool, st: &Stream) -> Vec<f32> {
    let (x, shared) = (up(&c.xh), up(&c.sh));
    let mut out = DevBuf::zeroed(rows * d.hidden * 4).unwrap();
    let mut os = OwnScratch::new(d, rows.max(64)).unwrap();
    let mut s = MoeScratch::new(d, 64).unwrap();
    if batched {
        let mut ps = PrefillScratch::new(d, rows).unwrap();
        ps.set_picks(&c.pick, &c.wts).unwrap();
        whole.run_prefill_pairs_windowed(&x, Dtype::F16, rows, &mut ps, &mut s, &mut os, st).unwrap();
        combine_pairs(&ps.y_pairs, &ps.pick, &ps.wts, Some(&shared), &mut out, rows, d, st).unwrap();
    } else {
        s.set_picks(&c.pick, &c.wts).unwrap();
        whole.run(&x, Dtype::F16, rows, Some(&shared), &mut out, &mut s, &mut os, st).unwrap();
    }
    dl(&out, st)
}

/// Two width shares: each rank's pairs stage on a poisoned scratch, the partials of every valid pair summed on the host (rank 0's +
/// rank 1's), uploaded back into each rank's pair buffer, each rank's `finish`: both outputs must be `want`.
fn check_shares(shares: &[MoeSubbanks], d: &MoeDims, c: &Case, rows: usize, batched: bool, want: &[f32], st: &Stream, tag: &str) {
    let (p, h) = (rows * d.topk, d.hidden);
    let (x, shared) = (up(&c.xh), up(&c.sh));
    let valid = |q: usize| c.pick[q] >= 0 && (c.pick[q] as usize) < d.experts;
    let mut ss: Vec<MoeScratch> = (0..2).map(|_| MoeScratch::new(d, 64).unwrap()).collect();
    let mut ps: Vec<PrefillScratch> = (0..2).map(|_| PrefillScratch::new(d, rows).unwrap()).collect();
    let mut parts: Vec<Vec<f32>> = Vec::new();
    for (r, sb) in shares.iter().enumerate() {
        let mut os = OwnScratch::new(d, rows.max(64)).unwrap();
        ss[r].poison(0x5a + r as u8, st).unwrap();
        if batched {
            // the decode scratch is the split gate/up subbank's windows' (they read the prefill scratch's picks)
            ps[r].poison(0x6b + r as u8, st).unwrap();
            ps[r].set_picks(&c.pick, &c.wts).unwrap();
            sb.run_prefill_pairs_windowed(&x, Dtype::F16, rows, &mut ps[r], &mut ss[r], &mut os, st).unwrap();
            parts.push(dl::<f32>(&ps[r].y_pairs, st)[..p * h].to_vec());
        } else {
            ss[r].set_picks(&c.pick, &c.wts).unwrap();
            sb.run_pairs(&x, Dtype::F16, rows, &mut ss[r], &mut os, st).unwrap();
            parts.push(dl::<f32>(&ss[r].y_pairs, st)[..p * h].to_vec());
        }
    }
    let mut sum = parts[0].clone();
    for q in (0..p).filter(|&q| valid(q)) {
        for i in q * h..(q + 1) * h {
            sum[i] += parts[1][i];
            assert_eq!(sum[i].to_bits(), (parts[1][i] + parts[0][i]).to_bits());
        }
    }
    for (r, sb) in shares.iter().enumerate() {
        let mut out = DevBuf::zeroed(rows * h * 4).unwrap();
        let (y, pk, w) = if batched {
            ps[r].y_pairs.upload(&sum).unwrap();
            (&ps[r].y_pairs, &ps[r].pick, &ps[r].wts)
        } else {
            ss[r].y_pairs.upload(&sum).unwrap();
            (&ss[r].y_pairs, &ss[r].pick, &ss[r].wts)
        };
        sb.finish(y, pk, w, Some(&shared), &mut out, rows, st).unwrap();
        let got: Vec<f32> = dl(&out, st);
        let bad = (0..rows * h).filter(|&i| got[i].to_bits() != want[i].to_bits()).count();
        assert_eq!(bad, 0, "{tag}: rank {r}, {rows} rows ({}): {bad} values differ from the single node", if batched { "batched" } else { "decode" });
    }
    // the row-split finish of a prefill chunk at two ranks (`MoeSubbanks::finish_sum`, `GlmTp::subbanks_big`): rank r finishes rows
    // [lo, hi) from its own partials and the peer's of those rows; its rows of the output must be the single node's
    if batched && rows >= 2 {
        let (m, k) = (rows / 2, d.topk);
        for (r, sb) in shares.iter().enumerate() {
            let (lo, hi) = if r == 0 { (0, m) } else { (m, rows) };
            let n = hi - lo;
            ps[r].y_pairs.upload(&parts[r]).unwrap();
            let peer = up(&parts[1 - r][lo * k * h..hi * k * h]);
            let y = ps[r].y_pairs.subview(lo * k * h * 4, n * k * h * 4);
            let pk = ps[r].pick.subview(lo * k * 4, n * k * 4);
            let w = ps[r].wts.subview(lo * k * 2, n * k * 2);
            let sh = shared.subview(lo * h * 4, n * h * 4);
            let mut out = DevBuf::zeroed(n * h * 4).unwrap();
            sb.finish_sum(&y, &peer, &pk, &w, Some(&sh), &mut out, n, st).unwrap();
            let got: Vec<f32> = dl(&out, st);
            let bad = (0..n * h).filter(|&i| got[i].to_bits() != want[lo * h + i].to_bits()).count();
            assert_eq!(bad, 0, "{tag}: rank {r}, rows {lo}..{hi} of {rows}: {bad} values of the row-split finish differ from the single node");
        }
    }
}

fn shares_of(lay: &Layer, d: &MoeDims) -> Vec<MoeSubbanks> {
    (0..2).map(|r| MoeSubbanks::from_groups_width(&lay.gate, &lay.up, &lay.down, *d, WidthShare::new(r, 2).unwrap()).unwrap()).collect()
}

#[test]
fn width_plans_are_one_split_of_the_whole_widths() {
    gpu!();
    let d = MoeDims::GLM53;
    for bits in [2usize, 3, 4, 5, 6] {
        let whole_d = default_plan(d.inter, d.hidden, bits, 1).unwrap();
        let whole_gu = default_plan(d.hidden, d.inter, bits, 2).unwrap();
        for r in 0..2 {
            let (gu, dn) = width_plans(&d, WidthShare::new(r, 2).unwrap(), bits, bits).unwrap();
            // gate/up: the whole width's K split and warps (a column's k tiles a warp are the whole width's)
            assert_eq!(gu, whole_gu, "{bits} bits: gate/up plan");
            // down: one split of the whole width's two, the same k tiles a warp
            assert_eq!(whole_d.0, 2, "{bits} bits: the whole width's down has two K splits");
            assert_eq!(dn, (1, whole_d.1));
            let per_warp = |k: usize, (sk, wk): (usize, usize)| (k / 16) / sk / wk;
            assert_eq!(per_warp(d.inter / 2, dn), per_warp(d.inter, whole_d), "{bits} bits: k tiles a warp");
        }
    }
    // a down plan of four splits (a wider expert) is refused: a rank's partial would be a sum of two splits
    let wide = MoeDims { inter: 8192, hidden: 2048, ..d };
    let sk = default_plan(wide.inter, wide.hidden, 4, 1).unwrap().0;
    assert_eq!(width_plans(&wide, WidthShare::new(0, 2).unwrap(), 4, 4).is_ok(), sk == 2, "down splits {sk}");
}

#[test]
fn group_slices_partition_the_trellis_and_scales() {
    let g = synth_group(512, 384, 3, Codebook::Mul1, 11);
    let (a, b) = (group_cols(&g, 0, 128).unwrap(), group_cols(&g, 128, 384).unwrap());
    let tw = 16 * 3;
    for kk in 0..512 / 16 {
        let row = &g.trellis[kk * 24 * tw..(kk + 1) * 24 * tw];
        assert_eq!(&row[..8 * tw], &a.trellis[kk * 8 * tw..(kk + 1) * 8 * tw]);
        assert_eq!(&row[8 * tw..], &b.trellis[kk * 16 * tw..(kk + 1) * 16 * tw]);
    }
    assert_eq!([a.svh.clone(), b.svh.clone()].concat(), g.svh);
    assert_eq!((a.group.n, a.nt, a.suh.len()), (128, 8, 512));
    let (c, e) = (group_rows(&g, 0, 256).unwrap(), group_rows(&g, 256, 512).unwrap());
    assert_eq!([c.trellis.clone(), e.trellis.clone()].concat(), g.trellis);
    assert_eq!([c.suh.clone(), e.suh.clone()].concat(), g.suh);
    assert_eq!((c.group.k, c.kt, c.svh.len()), (256, 16, 384));
    assert!(group_cols(&g, 0, 100).is_none() && group_rows(&g, 0, 640).is_none());
}

#[test]
fn synthetic_mixed_widths_equal_the_whole_width() {
    gpu!();
    let d = MoeDims { experts: 32, ..MoeDims::GLM53 };
    // four keys: 6/6/4, 5/5/4, 4/4/6 and a split gate/up 6/5/4
    let key = |e: usize| [(6, 6, 4), (5, 5, 4), (4, 4, 6), (6, 5, 4)][e % 4];
    let mk = |k, n, b, seed| synth_group(k, n, b, Codebook::Mul1, seed);
    let lay = Layer {
        gate: (0..d.experts).map(|e| mk(d.hidden, d.inter, key(e).0, 100 + e as u64)).collect(),
        up: (0..d.experts).map(|e| mk(d.hidden, d.inter, key(e).1, 200 + e as u64)).collect(),
        down: (0..d.experts).map(|e| mk(d.inter, d.hidden, key(e).2, 300 + e as u64)).collect(),
    };
    let st = Stream::new().unwrap();
    let whole = MoeSubbanks::from_groups(&lay.gate, &lay.up, &lay.down, d, ExpertOwnership::all(d.experts)).unwrap();
    let shares = shares_of(&lay, &d);
    assert!(whole.has_split_gate_up() && shares.iter().all(|s| s.has_split_gate_up() && s.banks().len() == 4));
    assert!(shares.iter().all(|s| s.decode_launches() == 6), "five launches over every subbank, then the finish");
    for (i, rows) in [1usize, 2, 3, 5, 16, 64].into_iter().enumerate() {
        let c = case(&d, rows, 1000 + i as u64);
        let want = whole_out(&whole, &d, &c, rows, false, &st);
        check_shares(&shares, &d, &c, rows, false, &want, &st, "synthetic");
    }
    for (i, rows) in [70usize, 300].into_iter().enumerate() {
        let c = case(&d, rows, 2000 + i as u64);
        let want = whole_out(&whole, &d, &c, rows, true, &st);
        check_shares(&shares, &d, &c, rows, true, &want, &st, "synthetic");
    }
    eprintln!("synthetic mixed widths (4 keys, a split gate/up one): two width shares == the whole width, decode 1-64 rows and batched 70 / 300 rows, bit for bit");
}

fn dir_of(var: &str, default: &str) -> Option<std::path::PathBuf> {
    let p = std::env::var(var)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(default));
    p.join("model.safetensors.index.json").exists().then_some(p)
}

fn load_layer(ck: &Checkpoint, l: usize, n: usize) -> Layer {
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
    Layer { gate: load("gate_proj"), up: load("up_proj"), down: load("down_proj") }
}

fn rms(a: &[f32], b: &[f32]) -> f64 {
    let n: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum();
    let s: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (n / s.max(1e-30)).sqrt()
}

#[test]
fn real_mixed_layers_equal_the_whole_width() {
    use gb_cuda::glm_p2_moe::{self, CombineMode};
    gpu!();
    let Some(dir) = dir_of("REXL3_REDSNOW_DIR", "models/vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw") else {
        eprintln!("no RED-SNOW pack (set REXL3_REDSNOW_DIR): not run");
        return;
    };
    let ck = scan(&dir, false).expect("scan the RED-SNOW pack");
    let d = MoeDims::GLM53;
    let st = Stream::new().unwrap();
    let layers: Vec<usize> = std::env::var("REXL3_REDSNOW_WIDTH_LAYERS")
        .ok()
        .map(|s| s.split(',').map(|x| x.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![4, 13]);
    for l in layers {
        let lay = load_layer(&ck, l, d.experts);
        let whole = MoeSubbanks::from_groups(&lay.gate, &lay.up, &lay.down, d, ExpertOwnership::all(d.experts)).unwrap();
        let shares = shares_of(&lay, &d);
        eprintln!(
            "layer {l}: {} subbanks{}; a share holds {:.3} GiB of the whole width's {:.3}",
            whole.banks().len(),
            if whole.has_split_gate_up() { " (a split gate/up one)" } else { "" },
            shares[0].expert_bytes() as f64 / (1u64 << 30) as f64,
            whole.expert_bytes() as f64 / (1u64 << 30) as f64
        );
        for (i, rows) in [1usize, 2, 3, 8, 64].into_iter().enumerate() {
            let c = case(&d, rows, 31 * l as u64 + i as u64);
            let want = whole_out(&whole, &d, &c, rows, false, &st);
            check_shares(&shares, &d, &c, rows, false, &want, &st, &format!("layer {l}"));
        }
        let rows = 300usize;
        let c = case(&d, rows, 777 + l as u64);
        let want = whole_out(&whole, &d, &c, rows, true, &st);
        check_shares(&shares, &d, &c, rows, true, &want, &st, &format!("layer {l}"));
        eprintln!("layer {l}: two width shares == the whole width bit for bit (decode 1/2/3/8/64 rows, batched {rows} rows)");

        // P2 on the two width shares: (out of the selected rows in calls of `chunk` rows)
        let h = d.hidden;
        let mut os = OwnScratch::new(&d, rows).unwrap();
        let mut sd = MoeScratch::new(&d, 128).unwrap();
        let mut ps = PrefillScratch::new(&d, rows).unwrap();
        let mut p2 = |sel: &[usize], chunk: usize| -> Vec<f32> {
            let mut out = Vec::new();
            for part in sel.chunks(chunk) {
                let n = part.len();
                let gx: Vec<u16> = part.iter().flat_map(|&q| c.xh[q * h..(q + 1) * h].to_vec()).collect();
                let gp: Vec<i32> = part.iter().flat_map(|&q| c.pick[q * d.topk..(q + 1) * d.topk].to_vec()).collect();
                let gw: Vec<u16> = part.iter().flat_map(|&q| c.wts[q * d.topk..(q + 1) * d.topk].to_vec()).collect();
                let gs: Vec<f32> = part.iter().flat_map(|&q| c.sh[q * h..(q + 1) * h].to_vec()).collect();
                let x = up(&gx);
                let mut parts = [DevBuf::zeroed(n * h * 4).unwrap(), DevBuf::zeroed(n * h * 4).unwrap()];
                for (r, sb) in shares.iter().enumerate() {
                    ps.poison(0x7f - r as u8, &st).unwrap();
                    ps.set_picks(&gp, &gw).unwrap();
                    let ord = glm_p2_moe::DEFAULT_ORDER;
                    glm_p2_moe::pairs_subbanks_rank(sb, &x, Dtype::F16, n, &mut ps, &mut os, &mut sd, ord, &st, &mut |_| Ok(())).unwrap();
                    glm_p2_moe::combine_dims(CombineMode::All, &ps.y_pairs, &ps.pick, &ps.wts, None, None, &mut parts[r], n, &d, &st)
                        .unwrap();
                }
                let mut o = DevBuf::zeroed(n * h * 4).unwrap();
                let pp = Some((&parts[0], &parts[1]));
                glm_p2_moe::combine_dims(CombineMode::Parts, &ps.y_pairs, &ps.pick, &ps.wts, Some(&up(&gs)), pp, &mut o, n, &d, &st)
                    .unwrap();
                out.extend(dl::<f32>(&o, &st));
            }
            out
        };
        let all: Vec<usize> = (0..rows).collect();
        let out0 = p2(&all, rows);
        assert!(out0.iter().all(|v| v.is_finite()), "layer {l}: P2 width output not finite");
        assert_eq!(bits(&p2(&all, rows)), bits(&out0), "layer {l}: P2 width, run to run");
        let same = |what: &str, sel: &[usize], o: &[f32]| {
            for (j, &q) in sel.iter().enumerate() {
                assert_eq!(bits(&o[j * h..(j + 1) * h]), bits(&out0[q * h..(q + 1) * h]), "layer {l} P2 width {what}: row {q}");
            }
        };
        for chunk in [7usize, 129] {
            same(&format!("calls of {chunk} rows"), &all, &p2(&all, chunk));
        }
        let rev: Vec<usize> = (0..rows).rev().collect();
        same("rows reversed", &rev, &p2(&rev, rows));
        let lone: Vec<usize> = (0..rows).step_by(37).collect();
        same("one row a call", &lone, &p2(&lone, 1));
        // as close to P1 (the single node's bits, `want`) as the one node's P2 is
        let mut o_node = DevBuf::zeroed(rows * h * 4).unwrap();
        ps.set_picks(&c.pick, &c.wts).unwrap();
        glm_p2_moe::run_subbanks(&whole, &up(&c.xh), Dtype::F16, rows, Some(&up(&c.sh)), &mut o_node, &mut ps, &mut os, &mut sd, &st)
            .unwrap();
        let (e_w, e_n) = (rms(&out0, &want), rms(&dl::<f32>(&o_node, &st), &want));
        eprintln!("layer {l}: P2 out vs P1, relative rms: width shares {e_w:.2e}, one node {e_n:.2e}");
        assert!(e_w <= 2.0 * e_n + 1e-7, "layer {l}: P2 of the width shares {e_w:e} against the one node's {e_n:e}");
        eprintln!("layer {l}: P2 of the width shares deterministic, chunked (7/129/1 rows) == whole, reversed == whole");
    }
}
