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
