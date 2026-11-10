//! Qwen3.8-Flash-Next dense prefill linears on the fused tensor-core chain GEMM (`gb_cuda::glm_p2`, `Variant::Chain`; what
//! `families/qwen_flash_next/model.rs::prefill_gemm` runs) against the pinned cuBLAS path (`Exl3Weights::forward_prefill_pinned`, what they ran
//! before), on SYNTHETIC Mul1 EXL3 weights of the pack's shapes at the bit widths of its mix: the output is the pinned path's bit for bit, at
//! every row count the server can hand a chunk. bf16 in and out for the trunk linears, fp16 in and out for the MTP head's `fc` projections.
//!
//!     cargo test -p gb-cuda --release --test flash_next_prefill_fused -- --test-threads=1 --nocapture
#![cfg(not(gb_no_cuda))]
mod common;

use common::exl3_synth::synth_group;
use common::Rng;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::glm_p2::{self, Variant};
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::f64_to_f16;
use gb_exl3::Codebook;

/// (name, K, N): the dense EXL3 linears of a Flash-Next trunk layer and of the MTP head.
const SHAPES: &[(&str, usize, usize)] = &[
    ("gdn in_proj_qkv", 2560, 10240),
    ("gdn in_proj_z", 2560, 6144),
    ("gdn out_proj", 6144, 2560),
    ("attn q_proj", 2560, 12288),
    ("attn o_proj", 6144, 2560),
    ("attn k/v", 2560, 512),
    ("indexer qk", 2560, 640),
    ("mtp fc", 2560, 2560),
];

fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn acts(seed: u64, rows: usize, k: usize, dt: Dtype) -> Vec<u16> {
    let mut r = Rng(seed);
    (0..rows * k)
        .map(|_| {
            let v = r.uniform(-1.5, 1.5);
            if dt == Dtype::F16 {
                f64_to_f16(v as f64)
            } else {
                bf16_bits(v)
            }
        })
        .collect()
}

fn upload(x: &[u16]) -> DevBuf {
    let mut d = DevBuf::zeroed_view(x.len() * 2).unwrap();
    d.upload(x).unwrap();
    d
}

#[test]
fn the_fused_chain_gemm_equals_the_pinned_cublas_path_on_the_flash_next_shapes_and_widths() {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let st = Stream::new().unwrap();
    let mut cases = 0;
    for (i, &(name, k, n)) in SHAPES.iter().enumerate() {
        let dt = if name == "mtp fc" { Dtype::F16 } else { Dtype::BF16 };
        for bits in [3usize, 4, 5, 6, 8] {
            let w = Exl3Weights::from_loaded(&synth_group(k, n, bits, Codebook::Mul1, 700 + (i * 10 + bits) as u64)).unwrap();
            assert!(glm_p2::supported(&w), "{name} {bits}b");
            for rows in [1usize, 64, 129, 1000, 4096] {
                let x = upload(&acts(11 + rows as u64, rows, k, dt));
                let mut yf = DevBuf::from_slice(&vec![0x7fc1u16; rows * n]).unwrap();
                let mut ws = DevBuf::alloc(glm_p2::workspace_bytes(rows, k)).unwrap();
                glm_p2::linear_variant(&w, &x, &mut yf, rows, dt, dt, &mut ws, Variant::Chain, 0, &st).unwrap();
                let mut yp = DevBuf::zeroed(rows * n * 2).unwrap();
                let mut scratch = DevBuf::alloc(w.prefill_scratch_bytes(rows)).unwrap();
                w.forward_prefill_pinned(&x, &mut yp, rows, dt, &mut scratch, &st).unwrap();
                st.sync().unwrap();
                assert_eq!(yf.to_vec::<u16>().unwrap(), yp.to_vec::<u16>().unwrap(), "{name} {bits}b, {rows} rows");
                cases += 1;
            }
        }
    }
    eprintln!("fused chain GEMM == pinned path: {cases} cases (8 shapes x 5 widths x 5 row counts), every output bit equal");
}
