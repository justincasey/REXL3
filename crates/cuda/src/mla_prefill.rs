//! Latent attention for the slabs of a large prefill chunk: one fused kernel (`csrc/mla_prefill.cu`) in place of [`crate::mla::attend_auto`]'s
//! dense launch, sparse launch and the merges, with the same bits. A CTA serves one row and two head tiles, walks the row's chunks of 256 keys in
//! order and folds each chunk's partial into a merged state in registers with the merge kernel's formula, so the fp32 chunk partials (151 MB for a
//! 128-row slab in the sparse regime) are never written and no workspace exists. The regime of a row is `attend_auto`'s: `pos < DENSE_MAX_KEYS` is
//! dense (a padding row writes zeros), otherwise the row is sparse when its selection meta says so and untouched when it does not.
//!
//! The decode path keeps `attend_auto` (its calls are 1..8 rows, where the partials are small and the two-kernel shape has the parallelism);
//! this module is for slabs of up to 128 rows and is tested against that path bit for bit (`tests/mla.rs`: 1,518 rows, the committed
//! `testdata/gpu/golden_mla.txt` hashes, row-count and neighbour invariance). It is specific to GLM-5.3-Flash's attention (64 heads sharing one
//! 512-wide latent, `DENSE_MAX_KEYS` = 2,048, the DSA selection layout of `mla::SparseSel`), unlike `moe_prefill` and `exl3_big`, which any family
//! can call.

use crate::attention::{BlockTable, RowIndex};
use crate::mla::{absorb, expand, LatentCache, MlaScratch, MlaWeights, SparseSel, DENSE_MAX_KEYS, LATENT_DIM, MAX_ROWS, N_HEADS};
use crate::rt::{CudaError, DevBuf, Stream};

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        #[allow(clippy::too_many_arguments)]
        pub fn gb_mla_attend_pf(
            q_lat: *const c_void,
            cache: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            dense_limit: i32,
            sel_idx: *const i32,
            sel_stride: i32,
            sel_meta: *const i32,
            out: *mut c_void,
            rows: i32,
            dtype: i32,
            kv: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// Attention for `n_rows` (1..=128) rows of any mix of regimes into `o_lat` (`T [n_rows][64][512]`), bit for bit what
/// [`crate::mla::attend_auto`] writes, with no workspace. `sel` is read only for rows at or past [`DENSE_MAX_KEYS`].
#[allow(clippy::too_many_arguments)]
pub fn attend_pf(
    q_lat: &DevBuf,
    cache: &LatentCache,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    sel: &SparseSel,
    o_lat: &mut DevBuf,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert!((1..=MAX_ROWS).contains(&n_rows) && n_rows <= rows.rows);
    let row_bytes = N_HEADS * LATENT_DIM * cache.dtype.size();
    assert!(q_lat.bytes() >= n_rows * row_bytes && o_lat.bytes() >= n_rows * row_bytes);
    assert!(
        sel.stride >= crate::mla::MAX_SELECTION && sel.indices.bytes() >= n_rows * sel.stride * 4,
        "selection indices too small"
    );
    assert!(sel.meta.bytes() >= n_rows * 2 * 4, "selection meta too small");
    assert!(matches!(cache.dtype, crate::exl3_linear::Dtype::F16 | crate::exl3_linear::Dtype::BF16));
    let rc = unsafe {
        ffi::gb_mla_attend_pf(
            q_lat.ptr(),
            cache.lat.ptr(),
            table.buf.ptr() as *const i32,
            table.max_blocks as i32,
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            DENSE_MAX_KEYS as i32,
            sel.indices.ptr() as *const i32,
            sel.stride as i32,
            sel.meta.ptr() as *const i32,
            o_lat.ptr(),
            n_rows as i32,
            cache.dtype as i32,
            cache.kv.id(),
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// [`crate::mla::mla_rows_auto`] with the fused attention: absorb, [`attend_pf`], expand. Same bits, no attention workspace.
#[allow(clippy::too_many_arguments)]
pub fn mla_rows_pf(
    q: &DevBuf,
    q_row_stride: usize,
    w: &MlaWeights,
    cache: &LatentCache,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    sel: &SparseSel,
    out: &mut DevBuf,
    scratch: &mut MlaScratch,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert_eq!(w.dtype, cache.dtype, "weights and cache must share one dtype");
    assert!(n_rows <= scratch.max_rows, "scratch too small");
    absorb(q, q_row_stride, w, &mut scratch.q_lat, n_rows, stream)?;
    attend_pf(&scratch.q_lat, cache, table, rows, n_rows, sel, &mut scratch.o_lat, stream)?;
    expand(&scratch.o_lat, w, out, n_rows, stream)
}
