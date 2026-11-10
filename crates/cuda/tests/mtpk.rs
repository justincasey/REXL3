//! Synthetic MTP input, draft-state and Flash-Next glue checks.
#![allow(clippy::needless_range_loop)]
mod common;
use common::Rng;
use gb_cuda::{fn_glue, mtpk, DevBuf, Stream};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};

fn gpu() -> Option<Stream> {
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    gb_cuda::rt::init_context().ok()?;
    Stream::new().ok()
}
fn bf(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
fn bff(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}
fn hf(x: f32) -> u16 {
    f64_to_f16(x as f64)
}
fn hff(x: u16) -> f32 {
    f16_to_f64(x) as f32
}
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ u64::from(*x)).wrapping_mul(0x100000001b3))
}
fn hash<T: Copy>(v: &[T]) -> u64 {
    let p = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) };
    fnv(p)
}
fn compare(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len());
    let max = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(max <= tol, "{label}: max_abs={max}, tol={tol}");
}
fn dl<T: Copy + Default>(b: &DevBuf, st: &Stream) -> Vec<T> {
    st.sync().unwrap();
    b.to_vec().unwrap()
}

struct Batch {
    stack: Vec<f32>,
    nh: Vec<u16>,
    emb: Vec<u16>,
    ne: Vec<f32>,
    gate: Vec<u16>,
    z: Vec<u16>,
    gamma: Vec<f32>,
    sh: Vec<u16>,
    mo: Vec<u16>,
}
const H: usize = 128;
const HC: usize = 2;
fn batch(rows: usize) -> Batch {
    let mut r = Rng(0x451287a);
    let stack = (0..rows * HC * H).map(|_| r.uniform(-1., 1.)).collect();
    let nh = (0..HC * H).map(|_| hf(r.uniform(-0.2, 0.2))).collect();
    let emb = (0..rows * H).map(|_| bf(r.uniform(-1., 1.))).collect();
    let ne = (0..H).map(|_| r.uniform(0.8, 1.2)).collect();
    let gate = (0..rows).map(|_| bf(r.uniform(-1., 1.))).collect();
    let z = (0..rows * H).map(|_| bf(r.uniform(-1., 1.))).collect();
    let gamma = (0..H).map(|_| r.uniform(0.8, 1.2)).collect();
    let sh = (0..rows * H).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
    let mo = (0..rows * H).map(|_| bf(r.uniform(-0.5, 0.5))).collect();
    Batch { stack, nh, emb, ne, gate, z, gamma, sh, mo }
}
struct ResultRows {
    xs: Vec<u16>,
    xe: Vec<u16>,
    combined: Vec<f32>,
    norm: Vec<u16>,
    shared: Vec<u16>,
}
fn run(b: &Batch, rows: usize, st: &Stream) -> ResultRows {
    let (stack, nh, emb, ne) = (
        DevBuf::from_slice(&b.stack).unwrap(),
        DevBuf::from_slice(&b.nh).unwrap(),
        DevBuf::from_slice(&b.emb).unwrap(),
        DevBuf::from_slice(&b.ne).unwrap(),
    );
    let mut xs = DevBuf::zeroed(rows * HC * H * 2).unwrap();
    let mut xe = DevBuf::zeroed(rows * H * 2).unwrap();
    mtpk::input(&stack, &nh, &emb, &ne, &mut xs, &mut xe, rows, HC, H, 1e-6, st).unwrap();
    let mut combined = DevBuf::zeroed(rows * HC * H * 4).unwrap();
    mtpk::combine(&xs, &xe, &mut combined, rows, HC, H, st).unwrap();
    let (z, gamma, sh, gate) = (
        DevBuf::from_slice(&b.z).unwrap(),
        DevBuf::from_slice(&b.gamma).unwrap(),
        DevBuf::from_slice(&b.sh).unwrap(),
        DevBuf::from_slice(&b.gate).unwrap(),
    );
    let mut norm = DevBuf::zeroed(rows * H * 2).unwrap();
    fn_glue::gated_rmsnorm_sigmoid(Some(&emb), &z, &gamma, &mut norm, rows, 1, H, 1e-6, st).unwrap();
    let mut shared = DevBuf::from_slice(&b.mo).unwrap();
    fn_glue::add_gated_shared(&sh, &gate, &mut shared, H, rows, st).unwrap();
    let shared = dl::<u16>(&shared, st);
    ResultRows {
        xs: dl(&xs, st),
        xe: dl(&xe, st),
        combined: dl(&combined, st),
        norm: dl(&norm, st),
        shared: shared[..rows * H].to_vec(),
    }
}
#[test]
#[ignore]
fn synthetic_mtp_glue_cpu_reference_and_row_invariance() {
    let Some(st) = gpu() else { return };
    let big = batch(64);
    let full = run(&big, 64, &st);
    let mut changed = batch(64);
    changed.stack[HC * H..].fill(0.75);
    changed.emb[H..].fill(bf(0.75));
    changed.z[H..].fill(bf(0.75));
    changed.sh[H..].fill(bf(0.75));
    changed.mo[H..].fill(bf(0.75));
    changed.gate[1..].fill(bf(0.25));
    changed.stack[36 * HC * H..37 * HC * H].copy_from_slice(&big.stack[..HC * H]);
    changed.emb[36 * H..37 * H].copy_from_slice(&big.emb[..H]);
    changed.z[36 * H..37 * H].copy_from_slice(&big.z[..H]);
    changed.sh[36 * H..37 * H].copy_from_slice(&big.sh[..H]);
    changed.mo[36 * H..37 * H].copy_from_slice(&big.mo[..H]);
    changed.gate[36] = big.gate[0];
    let mixed = run(&changed, 64, &st);
    for (name, a, b, stride) in [
        ("mtp stack", &full.xs, &mixed.xs, HC * H),
        ("mtp embedding", &full.xe, &mixed.xe, H),
        ("sigmoid norm", &full.norm, &mixed.norm, H),
        ("shared gate", &full.shared, &mixed.shared, H),
    ] {
        assert_eq!(&a[..stride], &b[..stride], "{name} other-row invariance");
        assert_eq!(&a[..stride], &b[36 * stride..37 * stride], "{name} position invariance");
    }
    assert_eq!(&full.combined[..HC * H], &mixed.combined[36 * HC * H..37 * HC * H], "mtp combine position invariance");

    for rows in [1, 7, 37, 64] {
        let b = &big;
        let got = run(b, rows, &st);
        assert_eq!(got.xs, &full.xs[..rows * HC * H], "mtp input streams row invariant");
        assert_eq!(got.xe, &full.xe[..rows * H], "mtp input embedding row invariant");
        assert_eq!(got.combined, &full.combined[..rows * HC * H], "mtp combine row invariant");
        assert_eq!(got.norm, &full.norm[..rows * H], "sigmoid norm row invariant");
        assert_eq!(got.shared, &full.shared[..rows * H], "shared gate row invariant");
        let mut sx = Vec::new();
        let mut ex = Vec::new();
        let mut combined = Vec::new();
        let mut norm = Vec::new();
        let mut shared = Vec::new();
        for row in 0..rows {
            for s in 0..HC {
                let a = &b.stack[(row * HC + s) * H..(row * HC + s + 1) * H];
                let inv = (a.iter().map(|v| v * v).sum::<f32>() / H as f32 + 1e-6).sqrt().recip();
                for d in 0..H {
                    sx.push(hf(a[d] * inv * (1. + hff(b.nh[s * H + d]))));
                }
            }
            let a = &b.emb[row * H..(row + 1) * H];
            let inv = (a.iter().map(|v| bff(*v).powi(2)).sum::<f32>() / H as f32 + 1e-6).sqrt().recip();
            for d in 0..H {
                ex.push(hf(bff(a[d]) * inv * b.ne[d]));
            }
            let ss = a.iter().map(|v| bff(*v).powi(2)).sum::<f32>();
            let scale = (ss / H as f32 + 1e-6).sqrt().recip();
            for d in 0..H {
                let g = 1. / (1. + (-bff(b.z[row * H + d])).exp());
                norm.push(bff(got.norm[row * H + d]));
                let want = bff(a[d]) * scale * b.gamma[d] * g;
                assert!((bff(got.norm[row * H + d]) - want).abs() < 0.016, "sigmoid norm CPU reference row {row} d {d}");
                let sg = 1. / (1. + (-bff(b.gate[row])).exp());
                shared.push(bff(b.mo[row * H + d]) + bff(b.sh[row * H + d]) * sg);
            }
        }
        compare(
            &got.xs.iter().map(|x| hff(*x)).collect::<Vec<_>>(),
            &sx.iter().map(|x| hff(*x)).collect::<Vec<_>>(),
            0.003,
            "mtp stack CPU reference",
        );
        compare(
            &got.xe.iter().map(|x| hff(*x)).collect::<Vec<_>>(),
            &ex.iter().map(|x| hff(*x)).collect::<Vec<_>>(),
            0.003,
            "mtp embedding CPU reference",
        );
        for row in 0..rows {
            for s in 0..HC {
                for d in 0..H {
                    combined.push(hff(got.xs[(row * HC + s) * H + d]) + hff(got.xe[row * H + d]));
                }
            }
        }
        compare(&got.combined, &combined, 1e-6, "mtp combine CPU reference");
        compare(&got.shared.iter().map(|x| bff(*x)).collect::<Vec<_>>(), &shared, 0.008, "shared gate CPU reference");
        assert!(norm.iter().all(|v| v.is_finite()));
    }
}
#[test]
#[ignore]
fn mtp_draft_copy_stage_pick_reference() {
    let Some(st) = gpu() else { return };
    let src = DevBuf::from_slice(&[4i32, 8, 12, 16, 20]).unwrap();
    let mut dst = DevBuf::zeroed(20).unwrap();
    mtpk::copy_i32(&mut dst, &src, 5, &st).unwrap();
    assert_eq!(dl::<i32>(&dst, &st), vec![4, 8, 12, 16, 20]);
    let chain = DevBuf::from_slice(&[100i32, 101, 102, 103, 104, 105]).unwrap();
    let mut ids = DevBuf::zeroed(8).unwrap();
    let mut positions = DevBuf::zeroed(8).unwrap();
    let mut row_pos = DevBuf::zeroed(8).unwrap();
    let mut row_slot = DevBuf::zeroed(8).unwrap();
    let mut seq_start = DevBuf::zeroed(12).unwrap();
    let seq_slot = DevBuf::from_slice(&[5i32, 9]).unwrap();
    // one sequence, arrays of stride 6: token = drafts[2], position = chain[3]
    mtpk::stage(
        mtpk::StepInputs {
            ids: &mut ids,
            positions: &mut positions,
            row_pos: &mut row_pos,
            row_slot: &mut row_slot,
            seq_start: &mut seq_start,
        },
        &seq_slot,
        &dst,
        &chain,
        1,
        6,
        3,
        &st,
    )
    .unwrap();
    assert_eq!(dl::<i32>(&ids, &st)[..1], [12]);
    assert_eq!(dl::<i32>(&positions, &st)[..1], [103]);
    assert_eq!(dl::<i32>(&row_pos, &st)[..1], [103]);
    assert_eq!(dl::<i32>(&row_slot, &st)[..1], [5]);
    assert_eq!(dl::<i32>(&seq_start, &st)[..2], [0, 1]);
    // two sequences, stride 3: sequence s drafts into [3s, 3s + 3), its positions likewise
    let drafts2 = DevBuf::from_slice(&[1i32, 2, 3, 11, 12, 13]).unwrap();
    let chain2 = DevBuf::from_slice(&[100i32, 101, 102, 200, 201, 202]).unwrap();
    mtpk::stage(
        mtpk::StepInputs {
            ids: &mut ids,
            positions: &mut positions,
            row_pos: &mut row_pos,
            row_slot: &mut row_slot,
            seq_start: &mut seq_start,
        },
        &seq_slot,
        &drafts2,
        &chain2,
        2,
        3,
        2,
        &st,
    )
    .unwrap();
    assert_eq!(dl::<i32>(&ids, &st)[..2], [2, 12]);
    assert_eq!(dl::<i32>(&positions, &st)[..2], [102, 202]);
    assert_eq!(dl::<i32>(&row_pos, &st)[..2], [102, 202]);
    assert_eq!(dl::<i32>(&row_slot, &st)[..2], [5, 9]);
    assert_eq!(dl::<i32>(&seq_start, &st)[..3], [0, 1, 2]);
    let logits = DevBuf::from_slice(&[bf(-1.), bf(2.), bf(2.), bf(0.), bf(5.), bf(1.)]).unwrap();
    let mut id = DevBuf::zeroed(16).unwrap();
    let mut p = DevBuf::zeroed(16).unwrap();
    mtpk::pick(&logits, 0, 4, &mut id, &mut p, 3, &st).unwrap();
    assert_eq!(dl::<i32>(&id, &st)[3], 1, "ties select smallest id");
    let want = 1.0 / ((-3.0f32).exp() + 1. + 1. + (-2.0f32).exp());
    assert!((dl::<f32>(&p, &st)[3] - want).abs() < 2e-6);
    // a row further into the buffer: the two logits at offset 4
    mtpk::pick(&logits, 4, 2, &mut id, &mut p, 2, &st).unwrap();
    assert_eq!(dl::<i32>(&id, &st)[2], 0);
    assert!((dl::<f32>(&p, &st)[2] - 1.0 / (1.0 + (-4.0f32).exp())).abs() < 2e-6);
    assert!(mtpk::pick(&logits, 0, 4, &mut id, &mut p, 16, &st).is_err(), "short buffer must fail before launch");
    assert!(mtpk::pick(&logits, 4, 4, &mut id, &mut p, 0, &st).is_err(), "an offset past the logits must fail before launch");
}
#[test]
#[ignore]
fn golden_mtpk_hashes() {
    let Some(st) = gpu() else { return };
    let b = batch(37);
    let out = run(&b, 37, &st);
    let mut lines = Vec::new();
    for (name, h) in [
        ("xs", hash(&out.xs)),
        ("xe", hash(&out.xe)),
        ("combine", hash(&out.combined)),
        ("sigmoid_norm", hash(&out.norm)),
        ("shared_gate", hash(&out.shared)),
    ] {
        lines.push(format!("{name} {h:016x}"));
    }
    let path = std::env::var("REXL3_GOLDEN_OUT").expect("REXL3_GOLDEN_OUT");
    std::fs::write(path, format!("{}\n", lines.join("\n"))).unwrap();
}

// ---- pick: bit-exact against recorded results of the reference kernel ----------------------------------------------------------

/// One logits row of `n` bf16 values (as bits) of the named kind. The generator is part of the test's identity: the recorded results
/// below are those of the reference kernel on exactly these rows.
fn pick_row(kind: &str, seed: u64, n: usize) -> Vec<u16> {
    let mut r = Rng(0x9e37_79b9_0000_0000 ^ seed);
    let uni = |r: &mut Rng, lo: f32, hi: f32| bf(r.uniform(lo, hi));
    match kind {
        // bf16 rounding of a uniform draw makes the top few grid values repeat many times: ties across threads and warps
        "uniform" => (0..n).map(|_| uni(&mut r, -6., 6.)).collect(),
        // a wide range: a few exps dominate the sum, the rest are tiny
        "wide" => (0..n).map(|_| uni(&mut r, -30., 30.)).collect(),
        // eight distinct values, the maximum shared by about an eighth of the row
        "levels" => (0..n).map(|_| bf((r.next_u64() % 8) as f32 * 0.5 - 1.)).collect(),
        // every element equal: the whole row ties
        "plateau" => vec![bf(0.5); n],
        // strictly increasing, then strictly decreasing: the maximum moves with n
        "ramp_up" => (0..n).map(|i| bf(i as f32 / 4096.)).collect(),
        "ramp_down" => (0..n).map(|i| bf((n - i) as f32 / 4096.)).collect(),
        // a spike (and a second equal one after it) on noise: the first must win
        k if k.starts_with("spike") => {
            let at: usize = k[5..].parse().unwrap();
            let mut v: Vec<u16> = (0..n).map(|_| uni(&mut r, -2., 2.)).collect();
            v[at] = bf(9.);
            if at + 1500 < n {
                v[at + 1500] = bf(9.);
            }
            v
        }
        // two equal maxima on adjacent elements: the lower index must win, whichever threads, words and warps own the two
        k if k.starts_with("pair") => {
            let at: usize = k[4..].parse().unwrap();
            let mut v: Vec<u16> = (0..n).map(|_| uni(&mut r, -2., 2.)).collect();
            v[at] = bf(9.);
            v[at + 1] = bf(9.);
            v
        }
        _ => panic!("pick row kind {kind}"),
    }
}

/// `(label, kind, seed, n, at)`: a row of `n` logits starting `at` elements into the buffer.
const PICK_CASES: &[(&str, &str, u64, usize, usize)] = &[
    ("uniform65536/1", "uniform", 1, 65536, 0),
    ("uniform65536/2", "uniform", 2, 65536, 0),
    ("uniform65536/3", "uniform", 3, 65536, 0),
    ("uniform65536/4", "uniform", 4, 65536, 0),
    ("uniform65536/5@7", "uniform", 5, 65536, 7),
    ("wide65536/1", "wide", 1, 65536, 0),
    ("wide65536/2", "wide", 2, 65536, 0),
    ("levels65536/1", "levels", 1, 65536, 0),
    ("levels65536/2", "levels", 2, 65536, 0),
    ("levels65536/3@3", "levels", 3, 65536, 3),
    ("plateau65536", "plateau", 0, 65536, 0),
    ("ramp_up65536", "ramp_up", 0, 65536, 0),
    ("ramp_down65536", "ramp_down", 0, 65536, 0),
    ("spike0/65536", "spike0", 1, 65536, 0),
    ("spike1023/65536", "spike1023", 2, 65536, 0),
    ("spike1024/65536", "spike1024", 3, 65536, 0),
    ("spike40000/65536", "spike40000", 4, 65536, 0),
    ("spike64035/65536", "spike64035", 5, 65536, 0),
    ("uniform65535/1", "uniform", 11, 65535, 0),
    ("uniform65535/2@5", "uniform", 12, 65535, 5),
    ("uniform64513", "uniform", 13, 64513, 0),
    ("uniform64512", "uniform", 14, 64512, 0),
    ("levels64000", "levels", 15, 64000, 0),
    ("wide50001", "wide", 16, 50001, 0),
    ("uniform33000", "uniform", 17, 33000, 0),
    ("uniform4097", "uniform", 18, 4097, 0),
    ("uniform1024", "uniform", 19, 1024, 0),
    ("uniform1000", "uniform", 20, 1000, 0),
    ("levels129", "levels", 21, 129, 0),
    ("uniform33", "uniform", 22, 33, 0),
    ("plateau2", "plateau", 0, 2, 0),
    ("uniform1", "uniform", 23, 1, 0),
];

/// The reference kernel's `(out_id, bits of out_p)` for each case of [`PICK_CASES`], recorded from the kernel before it was
/// restructured (one block of 1024 threads, two sequential passes over the row).
const PICK_GOLDEN: &[(&str, i32, u32)] = &[
    ("uniform65536/1", 76, 0x3940cf6c),
    ("uniform65536/2", 505, 0x393cf004),
    ("uniform65536/3", 1232, 0x393e3fb3),
    ("uniform65536/4", 1528, 0x39400ba9),
    ("uniform65536/5@7", 308, 0x3941c9d6),
    ("wide65536/1", 76, 0x3a70be04),
    ("wide65536/2", 827, 0x3a6cd3a3),
    ("levels65536/1", 24, 0x384d2ddd),
    ("levels65536/2", 2, 0x384b476b),
    ("levels65536/3@3", 11, 0x384c73ea),
    ("plateau65536", 0, 0x37800000),
    ("ramp_up65536", 65408, 0x397ff26f),
    ("ramp_down65536", 0, 0x397fe26f),
    ("spike0/65536", 0, 0x3d764d14),
    ("spike1023/65536", 1023, 0x3d73bd84),
    ("spike1024/65536", 1024, 0x3d74adda),
    ("spike40000/65536", 40000, 0x3d75b953),
    ("spike64035/65536", 64035, 0x3d777abd),
    ("uniform65535/1", 775, 0x3941a0fd),
    ("uniform65535/2@5", 845, 0x393f2a3e),
    ("uniform64513", 677, 0x3941d503),
    ("uniform64512", 625, 0x3942ad5e),
    ("levels64000", 19, 0x3852fd35),
    ("wide50001", 1103, 0x3a9bb177),
    ("uniform33000", 184, 0x39be219d),
    ("uniform4097", 1200, 0x3b3e06ed),
    ("uniform1024", 160, 0x3c2d5dbc),
    ("uniform1000", 412, 0x3c43a2c5),
    ("levels129", 2, 0x3cc95890),
    ("uniform33", 31, 0x3e84a54e),
    ("plateau2", 0, 0x3f000000),
    ("uniform1", 0, 0x3f800000),
];

fn run_pick(st: &Stream, kind: &str, seed: u64, n: usize, at: usize) -> (i32, u32, Vec<u16>) {
    let mut buf = pick_row("uniform", 0xfeed ^ seed, at + n + 9); // the elements around the row are random, never read
    let row = pick_row(kind, seed, n);
    buf[at..at + n].copy_from_slice(&row);
    let logits = DevBuf::from_slice(&buf).unwrap();
    let mut id = DevBuf::from_slice(&[-7i32, -7, -7]).unwrap();
    let mut p = DevBuf::from_slice(&[f32::NAN; 3]).unwrap();
    mtpk::pick(&logits, at, n, &mut id, &mut p, 1, st).unwrap();
    let (id, p) = (dl::<i32>(&id, st), dl::<f32>(&p, st));
    assert_eq!((id[0], id[2]), (-7, -7), "pick wrote outside out_id[j]");
    assert!(p[0].is_nan() && p[2].is_nan(), "pick wrote outside out_p[j]");
    (id[1], p[1].to_bits(), row)
}

#[test]
fn pick_matches_reference_kernel_bits() {
    let Some(st) = gpu() else { return };
    let mut wrong = Vec::new();
    for &(label, kind, seed, n, at) in PICK_CASES {
        let (id, bits, row) = run_pick(&st, kind, seed, n, at);
        // the line to record in PICK_GOLDEN, from a run of the reference kernel
        println!("    (\"{label}\", {id}, 0x{bits:08x}),");
        // semantics, independent of the recorded bits: the lowest index of the maximum; a probability near the f64 softmax
        let vals: Vec<f32> = row.iter().map(|&x| bff(x)).collect();
        let mx = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let want_id = vals.iter().position(|&v| v == mx).unwrap() as i32;
        assert_eq!(id, want_id, "{label}: argmax (lowest index on ties)");
        let sum: f64 = vals.iter().map(|&v| ((v - mx) as f64).exp()).sum();
        let p = f32::from_bits(bits);
        assert!(((p as f64) * sum - 1.).abs() < 2e-4, "{label}: p={p} vs softmax {}", 1. / sum);
        let g = PICK_GOLDEN.iter().find(|g| g.0 == label);
        if g.map(|g| (g.1, g.2)) != Some((id, bits)) {
            wrong.push(format!("{label}: id {id} p {p} (bits 0x{bits:08x}) differ from the reference kernel's {g:?}"));
        }
    }
    assert!(wrong.is_empty(), "out_id/out_p differ from the reference kernel:\n{}", wrong.join("\n"));
    assert_eq!(PICK_GOLDEN.len(), PICK_CASES.len(), "a recorded result without a case");
}

/// A production-sized row that starts on a 4-byte boundary is picked by the 512-thread kernel; the same row one element further into
/// a buffer is not aligned and is picked by the 1024-thread reference kernel. Both must give the same index and the same probability
/// bits, on rows of every kind, with the maximum (and a tie for it) at every kind of position.
#[test]
fn pick_wide_kernel_equals_reference_kernel() {
    let Some(st) = gpu() else { return };
    let kinds = [
        "uniform", "wide", "levels", "plateau", "ramp_up", "ramp_down", "spike0", "spike1", "spike511", "spike1023", "spike1024",
        "spike30001", "spike64035", "pair0", "pair510", "pair1022", "pair1023", "pair2048", "pair40001", "pair64000",
    ];
    let mut rows = 0;
    for (s, kind) in kinds.iter().enumerate() {
        for seed in 0..6u64 {
            let seed = 1000 + 17 * s as u64 + seed;
            let (id_a, bits_a, row) = run_pick(&st, kind, seed, 65536, 0);
            let (id_b, bits_b, _) = run_pick(&st, kind, seed, 65536, 1);
            assert_eq!((id_a, bits_a), (id_b, bits_b), "{kind} seed {seed}: aligned (wide) vs misaligned (reference) row");
            let mx = row.iter().map(|&x| bff(x)).fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(id_a as usize, row.iter().position(|&x| bff(x) == mx).unwrap(), "{kind} seed {seed}: argmax");
            rows += 1;
        }
    }
    println!("{rows} rows: the wide kernel and the reference kernel agree on every index and every probability bit");
}

/// Repeated launches on one row: the same bits every time (no race between the passes), and a mean kernel time per launch.
#[test]
fn pick_is_repeatable_and_timed() {
    let Some(st) = gpu() else { return };
    let row = pick_row("uniform", 77, 65536);
    let logits = DevBuf::from_slice(&row).unwrap();
    let mut id = DevBuf::zeroed(16 * 4).unwrap();
    let mut p = DevBuf::zeroed(16 * 4).unwrap();
    mtpk::pick(&logits, 0, 65536, &mut id, &mut p, 0, &st).unwrap();
    let first = (dl::<i32>(&id, &st)[0], dl::<f32>(&p, &st)[0].to_bits());
    let iters = 400;
    unsafe {
        use gb_cuda::rt::timed_events as te;
        let (mut e0, mut e1) = (std::ptr::null_mut(), std::ptr::null_mut());
        assert_eq!(te::create(&mut e0), 0);
        assert_eq!(te::create(&mut e1), 0);
        assert_eq!(te::record(e0, st.raw()), 0);
        for k in 0..iters {
            mtpk::pick(&logits, 0, 65536, &mut id, &mut p, k % 16, &st).unwrap();
        }
        assert_eq!(te::record(e1, st.raw()), 0);
        assert_eq!(te::synchronize(e1), 0);
        let mut ms = 0f32;
        assert_eq!(te::elapsed_ms(&mut ms, e0, e1), 0);
        println!("pick 65536 bf16: {:.2} us per launch ({iters} back-to-back launches, warm L2)", ms * 1e3 / iters as f32);
    }
    let (ids, ps) = (dl::<i32>(&id, &st), dl::<f32>(&p, &st));
    for j in 0..16 {
        assert_eq!((ids[j], ps[j].to_bits()), first, "launch result {j} differs from the first");
    }
}
