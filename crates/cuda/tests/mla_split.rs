//! The MLA head split (lane E, GLM Phase 2 item 5): `mla_rows_auto_heads` on heads `[h0, h0 + 32)` (a rank's half: its heads' q columns,
//! the layer's absorb / expand weights at head h0, 32-head q_lat / o_lat) gives exactly those heads' outputs of the 64-head
//! `mla_rows_auto`, for dense rows, sparse (DSA) rows and windows that mix both, fp16 and bf16, both halves. Bit for bit.
//!
//!     cargo test --release -p gb-cuda --test mla_split -- --test-threads=1 --nocapture
#![cfg(not(gb_no_cuda))]

mod common;

use common::Rng;
use gb_cuda::attention::{BlockTable, RowIndex};
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::mla::*;
use gb_cuda::num::f32_to_bf16;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::f64_to_f16;

fn enc(dt: Dtype, x: f32) -> u16 {
    match dt {
        Dtype::BF16 => f32_to_bf16(x),
        _ => f64_to_f16(x as f64),
    }
}

fn kv_b_host() -> Vec<u16> {
    let mut r = Rng(0x6B76_5F62);
    (0..N_HEADS * KV_B_ROWS_PER_HEAD * LATENT_DIM).map(|_| f32_to_bf16(r.uniform(-0.2, 0.2))).collect()
}

/// A latent cache holding `lens` sequences (sequence i in slot i), their pages placed in order.
fn world(dt: Dtype, lens: &[usize], st: &Stream) -> (LatentCache, BlockTable) {
    let n_blocks: usize = lens.iter().map(|l| l.div_ceil(BLOCK_TOKENS)).sum::<usize>() + 2;
    let mut cache = LatentCache::new(n_blocks, dt, gb_cuda::mla::LatentKv::Plain).unwrap();
    let max_blocks = lens.iter().map(|l| l.div_ceil(BLOCK_TOKENS)).max().unwrap() + 2;
    let mut tables = Vec::new();
    let mut next = 0u32;
    for &l in lens {
        let nb = l.div_ceil(BLOCK_TOKENS) as u32;
        let mut t: Vec<u32> = (next..next + nb).collect();
        next += nb;
        t.resize(max_blocks, 0x3fff_fff0);
        tables.push(t);
    }
    let table = BlockTable::from_host(&tables, max_blocks).unwrap();
    let mut r = Rng(0x1A7E);
    let (mut lat, mut slot, mut pos) = (Vec::new(), Vec::new(), Vec::new());
    for (si, &l) in lens.iter().enumerate() {
        for p in 0..l {
            let amp = r.uniform(0.5, 2.0);
            lat.extend((0..LATENT_DIM).map(|_| enc(dt, r.uniform(-1.0, 1.0) * amp)));
            slot.push(si as i32);
            pos.push(p as i32);
        }
    }
    let ld = DevBuf::from_slice(&lat).unwrap();
    let ri = RowIndex::from_host(&slot, &pos).unwrap();
    latent_write(&ld, &ri, &table, &mut cache, slot.len(), st).unwrap();
    st.sync().unwrap();
    (cache, table)
}

/// Columns `[h0 * w, (h0 + nh) * w)` of rows of `64 * w` values.
fn heads_of(v: &[u16], w: usize, h0: usize, nh: usize) -> Vec<u16> {
    v.chunks(N_HEADS * w).flat_map(|row| row[h0 * w..(h0 + nh) * w].to_vec()).collect()
}

#[test]
fn a_rank_of_32_mla_heads_gives_the_64_head_layers_outputs_for_its_heads() {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let st = Stream::new().unwrap();
    let kvb = kv_b_host();
    let mut cases = 0;
    for dt in [Dtype::F16, Dtype::BF16] {
        let w = MlaWeights::from_kv_b(&kvb, Dtype::BF16, dt).unwrap();
        // dense sequences and two past the DSA threshold (2,048 keys)
        let lens = [300usize, 1, 2047, 700, 3000, 5000];
        let (cache, table) = world(dt, &lens, &st);
        let max_ctx = 5000;
        // windows: (slot, position) rows; -1 is a padding row
        let windows: Vec<Vec<(usize, i32)>> = vec![
            vec![(0, 299)],
            vec![(1, 0), (0, 150), (2, 2046), (3, 699), (0, -1)],
            vec![(4, 2999)],
            vec![(4, 2500), (5, 4999), (2, 2000), (4, 2048), (5, 2047)],
            (0..17).map(|i| (5usize, 4983 + i)).collect(),
        ];
        for (wi, rows) in windows.iter().enumerate() {
            let n = rows.len();
            let mut r = Rng(0xABCD + wi as u64);
            let q: Vec<u16> = (0..n * N_HEADS * NOPE_DIM).map(|_| enc(dt, r.uniform(-1.5, 1.5))).collect();
            let slot: Vec<i32> = rows.iter().map(|x| x.0 as i32).collect();
            let pos: Vec<i32> = rows.iter().map(|x| x.1).collect();
            let ri = RowIndex::from_host(&slot, &pos).unwrap();
            // the selection of each sparse row: its last 2,051 positions, ascending (DSA's list shape)
            let stride = MAX_SELECTION;
            let mut idx = vec![-1i32; n * stride];
            let mut meta = vec![0i32; n * 2];
            for (i, &(_, p)) in rows.iter().enumerate() {
                if p >= DENSE_MAX_KEYS as i32 {
                    let first = p + 1 - MAX_SELECTION as i32;
                    for j in 0..MAX_SELECTION {
                        idx[i * stride + j] = first + j as i32;
                    }
                    meta[2 * i] = 1;
                    meta[2 * i + 1] = MAX_SELECTION as i32;
                }
            }
            let (idx_d, meta_d) = (DevBuf::from_slice(&idx).unwrap(), DevBuf::from_slice(&meta).unwrap());
            let sel = SparseSel { indices: &idx_d, meta: &meta_d, stride };
            let qd = DevBuf::from_slice(&q).unwrap();
            let mut scratch = MlaScratch::new(n, max_ctx, dt).unwrap();
            let mut out = DevBuf::zeroed(n * N_HEADS * NOPE_DIM * 2).unwrap();
            mla_rows_auto(&qd, N_HEADS * NOPE_DIM, &w, &cache, &table, &ri, n, &sel, &mut out, max_ctx, &mut scratch, &st)
                .unwrap();
            st.sync().unwrap();
            let full: Vec<u16> = out.to_vec().unwrap();
            for h0 in [0usize, 32] {
                let qh = heads_of(&q, NOPE_DIM, h0, 32);
                let qhd = DevBuf::from_slice(&qh).unwrap();
                let mut sc = MlaScratch::new(n, max_ctx, dt).unwrap();
                sc.ws.memset_async(0x3F, &st).unwrap();
                let mut oh = DevBuf::zeroed(n * 32 * NOPE_DIM * 2).unwrap();
                mla_rows_auto_heads(
                    &qhd,
                    32 * NOPE_DIM,
                    &w,
                    h0,
                    32,
                    &cache,
                    &table,
                    &ri,
                    n,
                    &sel,
                    &mut oh,
                    max_ctx,
                    &mut sc,
                    &st,
                )
                .unwrap();
                st.sync().unwrap();
                let got: Vec<u16> = oh.to_vec().unwrap();
                let want = heads_of(&full, NOPE_DIM, h0, 32);
                if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
                    panic!("{dt:?} window {wi} heads {h0}..{}: element {i} (row {}) differs", h0 + 32, i / (32 * NOPE_DIM));
                }
                cases += 1;
            }
        }
    }
    eprintln!("{cases} cases: the 32-head halves equal the 64-head MLA call bit for bit (dense, sparse and mixed windows, fp16 and bf16)");
}
