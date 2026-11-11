//! The decode linear that rotates its own input (the 4-warp ring kernel rotates the 128-column blocks of x it multiplies, in shared memory,
//! ahead of its k loop, under `set_fuse_mask`) is the unfused chain (a `rot_in` launch, then the linear) bit for bit.
//!
//! Every case runs the same call on a workspace full of junk with the fuse mask off (the `rot_in_kernel` launch followed by the linear
//! kernel, the code the fused kernel replaces), with mask 1 (the layers where the fusion pays: 4 warps a block and a grid of 128 blocks or
//! more) and with mask 2 (every 4-warp layer that fits, whatever its grid), and compares the output bytes. They cover:
//!
//! * every width 1..=8 and codebook on shapes whose plans put the warps' k ranges across the 128-column blocks in different ways
//!   (ranges that start mid-block, ranges of an odd number of tiles, one split and several, 4 and 8 warps a block);
//! * the Flash-Next layer shapes (qkv, z, out_proj, q, k / v, indexer qk) at the widths the pack uses;
//! * 1 to 8 rows (the fused window), and 9, 12, 16, 17, 40 (past it: the launch it falls back to, still the same bits);
//! * activations in fp16, bf16 and fp32, outputs in fp16, bf16 and fp32, with and without a bias;
//! * the programmatic-dependent-launch attribute on and off;
//! * the K-split partials and the column strips of the tensor-parallel cuts, and the half-bit widths (`gb_exl3_linear_k2`).
//!
//! It also checks that the fused kernel is the one that ran (`fused_in`) for the layers and row counts that take it.
//!
//!     cargo test --release -p gb-cuda --test exl3_fused_rot -- --test-threads=1 --nocapture
#![cfg(not(gb_no_cuda))]
mod common;

use common::*;
use gb_cuda::exl3_linear::{fused_in, set_fuse_mask, set_pdl_mask, Dtype, Exl3Weights};
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::checkpoint::{Group, LoadedGroup};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_exl3::{Bits, Codebook};

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: not run");
            return;
        }
    };
}

fn normal(r: &mut Rng) -> f64 {
    let (a, b) = ((r.next_u64() >> 11) as f64 / (1u64 << 53) as f64, (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64);
    (-2.0 * a.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
}

fn scale_vec(r: &mut Rng, n: usize, mag: f64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            f16_to_f64(f64_to_f16(
                (if r.next_u64() & 1 == 1 { -1.0 } else { 1.0 }) * mag * (0.5 + (r.next_u64() >> 40) as f64 / 16777216.0),
            ))
        })
        .collect()
}

fn synth(k: usize, n: usize, bits: usize, cb: Codebook, bias: bool, seed: u64) -> Exl3Weights {
    let mut r = Rng(seed);
    let words = (k / 16) * (n / 16) * 16 * bits;
    let trellis: Vec<i16> = (0..words).map(|_| r.next_u64() as i16).collect();
    let group = Group {
        prefix: "synthetic".into(),
        bits: Bits::from_twice(2 * bits as u8).unwrap(),
        codebook: cb,
        k,
        n,
        in_scales: "suh",
        out_scales: "svh",
        bias,
        files: vec![],
        bytes: 0,
    };
    let svh = scale_vec(&mut r, n, 1.0 / (k as f64).sqrt());
    let b = bias.then(|| scale_vec(&mut r, n, 0.5));
    let g = LoadedGroup { group, trellis, kt: k / 16, nt: n / 16, suh: scale_vec(&mut r, k, 1.0), svh, bias: b };
    Exl3Weights::from_loaded(&g).unwrap()
}

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
}

/// `rows` rows of `k` activations as `dtype` bytes. Rows differ in scale (a row of 10^-1, 1 and 10 sigma in turn, one all zero, one with a few
/// large outliers) so the rotated sums span many binades and a rounding that depends on the order of the sums would show.
fn activations(r: &mut Rng, rows: usize, k: usize, dtype: Dtype) -> Vec<u8> {
    let mut out = Vec::new();
    for row in 0..rows {
        for j in 0..k {
            let mut v = normal(r) * [0.1, 1.0, 10.0][row % 3];
            if row % 7 == 6 {
                v = 0.0;
            } else if row % 5 == 4 && j % 97 == 3 {
                v *= 300.0;
            }
            match dtype {
                Dtype::F16 => out.extend_from_slice(&f64_to_f16(v).to_le_bytes()),
                Dtype::BF16 => out.extend_from_slice(&bf16_bits(v as f32).to_le_bytes()),
                Dtype::F32 => out.extend_from_slice(&(v as f32).to_le_bytes()),
            }
        }
    }
    out
}

fn junk(bytes: usize, st: &Stream) -> DevBuf {
    let mut b = DevBuf::alloc(bytes.max(16)).unwrap();
    b.memset_async(0xAB, st).unwrap();
    st.sync().unwrap();
    b
}

/// The layer's decode forward at `m` rows with the fuse mask `mask`, as bytes.
fn run(w: &Exl3Weights, x: &DevBuf, m: usize, xd: Dtype, yd: Dtype, mask: i32, st: &Stream) -> Vec<u8> {
    set_fuse_mask(mask).unwrap();
    let mut ws = junk(w.workspace_bytes(m), st);
    let mut y = DevBuf::alloc(m * w.n * yd.size()).unwrap();
    y.memset_async(0xCD, st).unwrap();
    w.forward(x, &mut y, m, xd, yd, &mut ws, st).unwrap();
    st.sync().unwrap();
    y.to_vec().unwrap()
}

/// The fused call equals the unfused one at every row count in `rows`, under the fuse mask that takes the layers where it pays (1) and the one
/// that takes every layer that fits (2); returns how many calls ran the fused kernel under each.
fn check(w: &Exl3Weights, rows: &[usize], combos: &[(Dtype, Dtype)], pdl: &[i32], st: &Stream, seed: u64) -> [usize; 2] {
    let mut r = Rng(seed);
    let mut fused = [0usize; 2];
    for &(xd, yd) in combos {
        let max_m = *rows.iter().max().unwrap();
        let xb = activations(&mut r, max_m, w.k, xd);
        for &m in rows {
            let x = DevBuf::from_slice(&xb[..m * w.k * xd.size()]).unwrap();
            for &pdl_mask in pdl {
                set_pdl_mask(pdl_mask).unwrap();
                let want = run(w, &x, m, xd, yd, 0, st);
                for mask in [1, 2] {
                    set_fuse_mask(mask).unwrap();
                    let took = fused_in(m, w.k, w.n, w.bits);
                    let got = run(w, &x, m, xd, yd, mask, st);
                    if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
                        panic!(
                            "{}x{} {}-bit {:?} {xd:?}->{yd:?} pdl {pdl_mask} fuse mask {mask} at {m} rows (fused: {took}): \
                             byte {i} differs ({:#04x} vs {:#04x})",
                            w.k,
                            w.n,
                            w.bits,
                            w.codebook,
                            got[i],
                            want[i]
                        );
                    }
                    fused[mask as usize - 1] += took as usize;
                }
            }
        }
    }
    set_pdl_mask(0).unwrap();
    set_fuse_mask(0).unwrap();
    fused
}

const WINDOW: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
const PAST: [usize; 5] = [9, 12, 16, 17, 40];

fn all_rows() -> Vec<usize> {
    WINDOW.iter().chain(PAST.iter()).copied().collect()
}

/// Every width and codebook, on shapes whose plans cut the warps' k ranges against the 128-column blocks differently:
/// (1280, 8192): 8 warps, one split, 10 tiles a warp (ranges start mid-block); (2688, 8192): 21 tiles a warp; (1024, 1536): 4 warps with
/// K splits; (640, 640): the smallest tile counts; (384, 256): fewer than 8 tiles a warp; (2560, 4096) and (4096, 4096): 4 warps, 4 K
/// splits and a grid of 128 blocks, the layers the fuse policy takes.
#[test]
fn fused_rot_in_is_bit_exact_for_every_width_and_codebook() {
    gpu!();
    let st = Stream::new().unwrap();
    let shapes = [(1280usize, 8192usize), (2688, 8192), (1024, 1536), (640, 640), (384, 256), (2560, 4096), (4096, 4096)];
    let mut total = 0;
    let mut fused = [0usize; 2];
    for (si, &(k, n)) in shapes.iter().enumerate() {
        let plan = gb_cuda::exl3_linear::decode_plan(k, n).unwrap();
        eprintln!("{k}x{n}: plan (sk, wk) = {plan:?}, {} tiles a warp", k / 16 / plan.0 / plan.1);
        for bits in 1..=8usize {
            for cb in Codebook::ALL {
                let w = synth(k, n, bits, cb, bits % 2 == 0, 7 + 31 * si as u64 + bits as u64);
                let combos: &[(Dtype, Dtype)] = if bits % 3 == 0 {
                    &[(Dtype::BF16, Dtype::BF16), (Dtype::F32, Dtype::F32)]
                } else {
                    &[(Dtype::F16, Dtype::F16), (Dtype::BF16, Dtype::F32)]
                };
                let f = check(&w, &all_rows(), combos, &[0], &st, 1000 + si as u64);
                fused[0] += f[0];
                fused[1] += f[1];
                total += all_rows().len() * combos.len();
            }
        }
    }
    eprintln!("{} (the layers where it pays) and {} (every layer that fits) of {total} calls ran the fused kernel", fused[0], fused[1]);
    assert!(fused[0] > 0 && fused[1] > fused[0], "the fused kernel must run under both masks, more under mask 2");
}

/// The layers of the Flash-Next pack at the widths it uses (6 bits mostly, a few 7 and 8), mul1 codebook, the model's activation type, with the
/// programmatic launch attribute on and off.
#[test]
fn fused_rot_in_is_bit_exact_at_the_pack_shapes() {
    gpu!();
    let st = Stream::new().unwrap();
    // (k, n, name): qkv, q, z, k / v, indexer qk, out_proj / o_proj
    let layers = [
        (2560usize, 10240usize, "qkv"),
        (2560, 12288, "q"),
        (2560, 6144, "z"),
        (2560, 512, "k"),
        (2560, 640, "indexer"),
        (6144, 2560, "out"),
    ];
    for &(k, n, name) in &layers {
        let plan = gb_cuda::exl3_linear::decode_plan(k, n).unwrap();
        for bits in [6usize, 7, 8] {
            let w = synth(k, n, bits, Codebook::Mul1, false, 4242 + (k + n + bits) as u64);
            let fused = check(&w, &all_rows(), &[(Dtype::BF16, Dtype::BF16)], &[0, 7], &st, 77);
            eprintln!("{name} {k}x{n} {bits}-bit plan {plan:?}: {fused:?} of {} calls fused", all_rows().len() * 2);
            if bits == 6 {
                for m in 1..=5usize {
                    // every 4-warp layer that fits (mask 2): z, k, v, the indexer and out_proj / o_proj; the 8-warp layers never fuse
                    set_fuse_mask(2).unwrap();
                    assert_eq!(fused_in(m, k, n, bits), plan.1 == 4, "{name} at {m} rows under mask 2");
                    // the policy (mask 1): the 4-warp layers with a grid of 128 blocks or more (z, out_proj)
                    set_fuse_mask(1).unwrap();
                    let wide = plan.1 == 4 && (n / 128) * plan.0 >= 128;
                    assert_eq!(fused_in(m, k, n, bits), wide, "{name} at {m} rows under mask 1");
                }
                set_fuse_mask(0).unwrap();
            }
        }
        // the other activation / output types, once, at the 6-bit width
        let w = synth(k, n, 6, Codebook::Mul1, true, 99 + n as u64);
        check(
            &w,
            &[1, 3, 8],
            &[(Dtype::F16, Dtype::F16), (Dtype::F32, Dtype::F32), (Dtype::BF16, Dtype::F32), (Dtype::F32, Dtype::BF16)],
            &[0, 7],
            &st,
            55,
        );
    }
    // the largest layer, the logits head, at the rows a window has
    let w = synth(2560, 248320, 6, Codebook::Mul1, false, 5);
    check(&w, &[1, 3, 8], &[(Dtype::BF16, Dtype::BF16)], &[0, 7], &st, 11);
}

/// The tensor-parallel cuts run the same kernel: the partials of a K range (nothing finished) and a column strip read in place.
#[test]
fn fused_rot_in_is_bit_exact_on_the_split_and_strip_paths() {
    gpu!();
    let st = Stream::new().unwrap();
    for (k, n, bits) in [(2560usize, 6144usize, 6usize), (6144, 2560, 6), (1024, 1536, 4), (2560, 10240, 6)] {
        let w = synth(k, n, bits, Codebook::Mul1, true, 321 + n as u64);
        let (sk, wk) = w.plan().unwrap();
        let mut r = Rng(9);
        for m in [1usize, 3, 5, 8, 12, 33] {
            let xb = activations(&mut r, m, k, Dtype::F16);
            // K ranges (sk > 1): the partials of the first half and the second half of the splits, bytes
            if sk > 1 {
                let cut = sk / 2;
                for &(s0, s1) in &[(0usize, cut), (cut, sk)] {
                    let (k0, k1) = w.split_k_range(s0, s1).unwrap();
                    let xr: Vec<u8> = (0..m).flat_map(|row| xb[(row * k + k0) * 2..(row * k + k1) * 2].to_vec()).collect();
                    let xd = DevBuf::from_slice(&xr).unwrap();
                    let mut zs = Vec::new();
                    for mask in [0, 2] {
                        set_fuse_mask(mask).unwrap();
                        let mut z = DevBuf::alloc(sk * m * n * 4).unwrap();
                        z.memset_async(0xCD, &st).unwrap();
                        let mut ws = junk(w.split_partials_workspace_bytes(s0, s1, m), &st);
                        w.forward_split_partials(s0, s1, &xd, Dtype::F16, &mut z, s0 * m * n * 4, m, &mut ws, &st).unwrap();
                        st.sync().unwrap();
                        zs.push(z.to_vec::<u8>().unwrap());
                    }
                    assert!(zs[0] == zs[1], "{k}x{n} K range {s0}..{s1} partials at {m} rows (plan {sk}, {wk})");
                }
            }
            // column strips in place
            let strips = n / 128;
            let x = DevBuf::from_slice(&xb).unwrap();
            for &(lo, hi) in &[(0usize, 1usize), (1, strips.min(5)), (strips / 2, strips)] {
                let mut outs = Vec::new();
                for mask in [0, 2] {
                    set_fuse_mask(mask).unwrap();
                    let mut ws = junk(w.strip_workspace_bytes(hi - lo, m), &st);
                    let mut y = DevBuf::alloc(m * (hi - lo) * 128 * 4).unwrap();
                    y.memset_async(0xCD, &st).unwrap();
                    w.forward_strip(lo, hi, &x, &mut y, m, Dtype::F16, Dtype::F32, &mut ws, &st).unwrap();
                    st.sync().unwrap();
                    outs.push(y.to_vec::<u8>().unwrap());
                }
                assert!(outs[0] == outs[1], "{k}x{n} strips {lo}..{hi} at {m} rows");
            }
        }
    }
    set_fuse_mask(0).unwrap();
}

/// The half-bit widths (1.5, 2.5 and 3.5 bits, mul1) and the even k2 through `gb_exl3_linear_k2`, raw trellis words.
#[test]
fn fused_rot_in_is_bit_exact_for_the_half_bit_widths() {
    gpu!();
    let st = Stream::new().unwrap();
    for (k, n) in [(1280usize, 8192usize), (1024, 1536), (2560, 6144)] {
        for k2 in (2..=16usize).filter(|k| k % 2 == 0 || *k <= 7) {
            let mut r = Rng(1000 + (k2 * n) as u64);
            let words = (k / 16) * (n / 16) * 4 * k2;
            let trellis = DevBuf::from_slice(&(0..words).map(|_| r.next_u64() as u32).collect::<Vec<u32>>()).unwrap();
            let suh = DevBuf::from_slice(&scale_vec(&mut r, k, 1.0).iter().map(|&v| f64_to_f16(v)).collect::<Vec<u16>>()).unwrap();
            let svh = DevBuf::from_slice(&scale_vec(&mut r, n, 0.02).iter().map(|&v| f64_to_f16(v)).collect::<Vec<u16>>()).unwrap();
            let codebooks: &[Codebook] = if k2 % 2 == 1 { &[Codebook::Mul1] } else { &Codebook::ALL };
            for &cb in codebooks {
                for m in [1usize, 2, 3, 8, 9, 17] {
                    let xb = activations(&mut r, m, k, Dtype::BF16);
                    let x = DevBuf::from_slice(&xb).unwrap();
                    let mut outs = Vec::new();
                    for mask in [0, 2] {
                        set_fuse_mask(mask).unwrap();
                        let mut ws = junk(gb_cuda::exl3_linear::linear_workspace_bytes(m, k, n, 1), &st);
                        let mut y = DevBuf::alloc(m * n * 2).unwrap();
                        y.memset_async(0xCD, &st).unwrap();
                        Exl3Weights::k2_forward(
                            &x,
                            &mut y,
                            m,
                            Dtype::BF16,
                            Dtype::BF16,
                            k,
                            n,
                            k2,
                            cb,
                            &trellis,
                            &suh,
                            &svh,
                            None,
                            &mut ws,
                            &st,
                        )
                        .unwrap();
                        st.sync().unwrap();
                        outs.push(y.to_vec::<u8>().unwrap());
                    }
                    assert!(outs[0] == outs[1], "{k}x{n} k2 {k2} {cb:?} at {m} rows");
                }
            }
        }
    }
    set_fuse_mask(0).unwrap();
}
