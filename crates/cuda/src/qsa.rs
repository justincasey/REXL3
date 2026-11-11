//! QSA, the sparse-attention indexer of Flash-Next's full-attention layers (kernels in `csrc/qsa.cu`): the indexer
//! stage (query norm + rope, raw keys), block pooling into a paged pool plane, block scores and top-512 selection, and
//! attention over the selected tokens.
//!
//! The pool plane is f16 `[n_pages][64][128]`, paged with the KV cache's block table (page `table[slot][4 j / 256]`
//! holds block `j` at `j % 64`). The raw-key ring is f16 `[slots][ring][128]`, indexed by position mod `ring`.
#![allow(clippy::too_many_arguments)]

use crate::attention::{BlockTable, KvHeads, PagedKv, BLOCK_TOKENS, HEAD_DIM, N_Q_HEADS};
use crate::rt::{CudaError, DevBuf, Stream};

pub const DK: usize = 128;
pub const N_HEADS: usize = 4;
/// Width of the indexer projection (4 query heads and one raw key).
pub const QK: usize = (N_HEADS + 1) * DK;
pub const CR: usize = 4;
pub const TOPK: usize = 512;
/// Blocks of one KV page.
pub const PAGE_BLOCKS: usize = BLOCK_TOKENS / CR;
/// Highest query position + 1 below which the selection is every block (dense attention is the same function).
pub const SPARSE_FROM: usize = CR * TOPK + 3;
/// Partial sums a decode-window row's attention is split into. Fixed, so a row's result never depends on how many rows
/// share its window (drafted == serial, bit for bit): each part is a fixed share of that row's own token list.
pub const WINDOW_PARTS: usize = 32;
/// Rows from which one part's attention and merge are one kernel (`qsa.cu`'s ATTEND_H_MIN_ROWS).
const ATTEND_MERGED_MIN_ROWS: usize = 64;

/// Refuses a buffer smaller than what a launch touches, before anything is queued (`CudaError(1)`, as an invalid value).
fn need(b: &DevBuf, bytes: usize, what: &str) -> Result<(), CudaError> {
    if b.bytes() < bytes {
        eprintln!("{what}: buffer of {} bytes, the launch touches {bytes}", b.bytes());
        return Err(CudaError(1));
    }
    Ok(())
}

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_qsa_stage(qk: *const c_void, qw: *const c_void, positions: *const c_void, q: *mut c_void,
                            kraw: *mut c_void, rows: i32, eps: f32, theta: f32, stream: *mut c_void) -> i32;
        pub fn gb_qsa_pool(kraw: *const c_void, ring: *const c_void, kw: *const c_void, positions: *const c_void,
                           seq_start: *const c_void, seq_slot: *const c_void, n_seq: i32, table: *const c_void,
                           max_blocks: i32, pool: *mut c_void, rows: i32, ring_len: i32, eps: f32, theta: f32,
                           stream: *mut c_void) -> i32;
        pub fn gb_qsa_ring(kraw: *const c_void, ring: *mut c_void, positions: *const c_void, seq_start: *const c_void,
                           seq_slot: *const c_void, n_seq: i32, rows: i32, ring_len: i32, stream: *mut c_void) -> i32;
        pub fn gb_qsa_scores(q: *const c_void, pool: *const c_void, table: *const c_void, max_blocks: i32,
                             row_slot: *const c_void, positions: *const c_void, scores: *mut c_void, rows: i32,
                             nb_cap: i32, stream: *mut c_void) -> i32;
        pub fn gb_qsa_topk(scores: *const c_void, positions: *const c_void, rows: i32, nb_cap: i32, sel: *mut c_void,
                           sel_n: *mut c_void, stream: *mut c_void) -> i32;
        pub fn gb_qsa_attend(q: *const c_void, k_cache: *const c_void, v_cache: *const c_void, table: *const c_void,
                             max_blocks: i32, row_slot: *const c_void, positions: *const c_void, sel: *const c_void,
                             sel_n: *const c_void, part_o: *mut c_void, part_m: *mut c_void, part_l: *mut c_void,
                             rows: i32, parts: i32, stream: *mut c_void) -> i32;
        pub fn gb_qsa_attend_merged(q: *const c_void, k_cache: *const c_void, v_cache: *const c_void, table: *const c_void,
                                    max_blocks: i32, row_slot: *const c_void, positions: *const c_void, sel: *const c_void,
                                    sel_n: *const c_void, out: *mut c_void, rows: i32, stream: *mut c_void) -> i32;
        pub fn gb_qsa_attend_merged_ref(q: *const c_void, k_cache: *const c_void, v_cache: *const c_void, table: *const c_void,
                                        max_blocks: i32, row_slot: *const c_void, positions: *const c_void, sel: *const c_void,
                                        sel_n: *const c_void, out: *mut c_void, rows: i32, stream: *mut c_void) -> i32;
        pub fn gb_qsa_merge(part_o: *const c_void, part_m: *const c_void, part_l: *const c_void, out: *mut c_void,
                            rows: i32, parts: i32, stream: *mut c_void) -> i32;
    }
}

/// The batch layout the pool and ring kernels read.
pub struct Seqs<'a> {
    pub positions: &'a DevBuf,
    pub seq_start: &'a DevBuf,
    pub seq_slot: &'a DevBuf,
    pub n_seq: usize,
    pub rows: usize,
}

/// Query heads (normed, roped at each row's position) and raw keys from the indexer projection `qk` (bf16 `[rows][640]`).
pub fn stage(
    qk: &DevBuf,
    qw: &DevBuf,
    positions: &DevBuf,
    q: &mut DevBuf,
    kraw: &mut DevBuf,
    rows: usize,
    eps: f32,
    theta: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    need(qk, rows * QK * 2, "qsa stage qk")?;
    need(qw, DK * 4, "qsa stage q norm")?;
    need(positions, rows * 4, "qsa stage positions")?;
    need(q, rows * N_HEADS * DK * 2, "qsa stage q")?;
    let qp = q.ptr();
    need(kraw, rows * DK * 2, "qsa stage kraw")?;
    let kp = kraw.ptr();
    CudaError::check(unsafe { ffi::gb_qsa_stage(qk.ptr(), qw.ptr(), positions.ptr(), qp, kp, rows as i32, eps, theta, st.raw()) })
}

/// Pool every block a row of the batch completes into `pool` (f16 `[pages][64][128]`, `pages` of the KV cache).
pub fn pool(
    kraw: &DevBuf,
    ring: &DevBuf,
    kw: &DevBuf,
    b: &Seqs<'_>,
    table: &BlockTable,
    pool: &mut DevBuf,
    pages: usize,
    slots: usize,
    ring_len: usize,
    eps: f32,
    theta: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    need(kraw, b.rows * DK * 2, "qsa pool kraw")?;
    need(ring, slots * ring_len * DK * 2, "qsa pool ring")?;
    need(kw, DK * 4, "qsa pool k norm")?;
    need(b.positions, b.rows * 4, "qsa pool positions")?;
    need(b.seq_start, (b.n_seq + 1) * 4, "qsa pool seq_start")?;
    need(b.seq_slot, b.n_seq * 4, "qsa pool seq_slot")?;
    need(pool, pages * PAGE_BLOCKS * DK * 2, "qsa pool plane")?;
    let pp = pool.ptr();
    CudaError::check(unsafe {
        ffi::gb_qsa_pool(
            kraw.ptr(),
            ring.ptr(),
            kw.ptr(),
            b.positions.ptr(),
            b.seq_start.ptr(),
            b.seq_slot.ptr(),
            b.n_seq as i32,
            table.buf.ptr(),
            table.max_blocks as i32,
            pp,
            b.rows as i32,
            ring_len as i32,
            eps,
            theta,
            st.raw(),
        )
    })
}

/// The last `ring_len` raw keys of each sequence of the batch into its slot ring.
pub fn ring_store(
    kraw: &DevBuf,
    ring: &mut DevBuf,
    b: &Seqs<'_>,
    slots: usize,
    ring_len: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    need(kraw, b.rows * DK * 2, "qsa ring kraw")?;
    need(ring, slots * ring_len * DK * 2, "qsa ring")?;
    let rp = ring.ptr();
    CudaError::check(unsafe {
        ffi::gb_qsa_ring(
            kraw.ptr(),
            rp,
            b.positions.ptr(),
            b.seq_start.ptr(),
            b.seq_slot.ptr(),
            b.n_seq as i32,
            b.rows as i32,
            ring_len as i32,
            st.raw(),
        )
    })
}

/// Scores `[rows][nb_cap]` f32 of the blocks each row can see (`nb_cap` >= every row's `(position + 1) / 4`).
pub fn scores(
    q: &DevBuf,
    pool: &DevBuf,
    pages: usize,
    table: &BlockTable,
    rr: &Rows<'_>,
    scores: &mut DevBuf,
    nb_cap: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    let rows = rr.rows;
    need(q, rows * N_HEADS * DK * 2, "qsa scores q")?;
    need(pool, pages * PAGE_BLOCKS * DK * 2, "qsa scores pool")?;
    rr.check()?;
    assert!(nb_cap * CR <= table.max_blocks * BLOCK_TOKENS);
    need(scores, rows * nb_cap * 4, "qsa scores")?;
    let sp = scores.ptr();
    CudaError::check(unsafe {
        ffi::gb_qsa_scores(
            q.ptr(),
            pool.ptr(),
            table.buf.ptr(),
            table.max_blocks as i32,
            rr.slot.ptr(),
            rr.pos.ptr(),
            sp,
            rows as i32,
            nb_cap as i32,
            st.raw(),
        )
    })
}

/// Per-row slot and position arrays (i32 `[rows]`) of the rows a selection call covers.
pub struct Rows<'a> {
    pub slot: &'a DevBuf,
    pub pos: &'a DevBuf,
    pub rows: usize,
}

impl Rows<'_> {
    fn check(&self) -> Result<(), CudaError> {
        need(self.slot, self.rows * 4, "qsa row slots")?;
        need(self.pos, self.rows * 4, "qsa row positions")
    }
}

/// Per row the selected blocks (`sel` i32 `[rows][512]`, ascending) and their count (`sel_n` i32 `[rows]`).
pub fn topk(
    scores: &DevBuf,
    rr: &Rows<'_>,
    nb_cap: usize,
    sel: &mut DevBuf,
    sel_n: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    let rows = rr.rows;
    rr.check()?;
    need(scores, rows * nb_cap * 4, "qsa topk scores")?;
    need(sel, rows * TOPK * 4, "qsa topk sel")?;
    let sp = sel.ptr();
    need(sel_n, rows * 4, "qsa topk sel_n")?;
    let np = sel_n.ptr();
    CudaError::check(unsafe { ffi::gb_qsa_topk(scores.ptr(), rr.pos.ptr(), rows as i32, nb_cap as i32, sp, np, st.raw()) })
}

/// Bytes of the partial sums [`attend`] needs for `rows` rows in `parts` parts.
pub fn attend_workspace_bytes(rows: usize, parts: usize) -> usize {
    let pairs = rows * N_Q_HEADS * parts;
    let a = |x: usize| x.div_ceil(256) * 256;
    a(pairs * HEAD_DIM * 4) + 2 * a(pairs * 4)
}

/// Attention of each row (`q`, `out` bf16 `[rows][24][256]`) over its selected blocks plus its tail block.
pub fn attend(
    q: &DevBuf,
    kv: &PagedKv,
    table: &BlockTable,
    rr: &Rows<'_>,
    sel: &DevBuf,
    sel_n: &DevBuf,
    parts: usize,
    ws: &mut DevBuf,
    out: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    if kv.kv_heads != KvHeads::Two {
        return Err(CudaError(1)); // the sparse attention is built for Flash-Next's 2 kv heads
    }
    let rows = rr.rows;
    rr.check()?;
    let qn = rows * N_Q_HEADS * HEAD_DIM * 2;
    need(q, qn, "qsa attend q")?;
    need(sel, rows * TOPK * 4, "qsa attend sel")?;
    need(sel_n, rows * 4, "qsa attend sel_n")?;
    assert!(parts >= 1);
    let ws_bytes = attend_workspace_bytes(rows, parts);
    need(ws, ws_bytes, "qsa attend workspace")?;
    need(out, qn, "qsa attend out")?;
    if parts == 1 && rows >= ATTEND_MERGED_MIN_ROWS {
        // a prefill chunk: the attention kernel writes the merged output itself, no partial sums
        return CudaError::check(unsafe {
            ffi::gb_qsa_attend_merged(
                q.ptr(),
                kv.k.ptr(),
                kv.v.ptr(),
                table.buf.ptr(),
                table.max_blocks as i32,
                rr.slot.ptr(),
                rr.pos.ptr(),
                sel.ptr(),
                sel_n.ptr(),
                out.ptr(),
                rows as i32,
                st.raw(),
            )
        });
    }
    let base = ws.ptr() as *mut u8;
    let pairs = rows * N_Q_HEADS * parts;
    let a = |x: usize| x.div_ceil(256) * 256;
    let (o_b, m_b) = (a(pairs * HEAD_DIM * 4), a(pairs * 4));
    // SAFETY: the three parts lie inside the `ws_bytes` checked above
    let (po, pm, pl) = unsafe { (base, base.add(o_b), base.add(o_b + m_b)) };
    let op = out.ptr();
    CudaError::check(unsafe {
        ffi::gb_qsa_attend(
            q.ptr(),
            kv.k.ptr(),
            kv.v.ptr(),
            table.buf.ptr(),
            table.max_blocks as i32,
            rr.slot.ptr(),
            rr.pos.ptr(),
            sel.ptr(),
            sel_n.ptr(),
            po as _,
            pm as _,
            pl as _,
            rows as i32,
            parts as i32,
            st.raw(),
        )
    })?;
    CudaError::check(unsafe { ffi::gb_qsa_merge(po as _, pm as _, pl as _, op, rows as i32, parts as i32, st.raw()) })
}

/// [`attend`] for one part and at least 64 rows through the kernel that staged shared K/V rows replaced (every warp loads its own rows). Only
/// the tests call it: the two must agree bit for bit.
pub fn attend_merged_ref(
    q: &DevBuf,
    kv: &PagedKv,
    table: &BlockTable,
    rr: &Rows<'_>,
    sel: &DevBuf,
    sel_n: &DevBuf,
    out: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    if kv.kv_heads != KvHeads::Two || rr.rows < ATTEND_MERGED_MIN_ROWS {
        return Err(CudaError(1));
    }
    let rows = rr.rows;
    rr.check()?;
    let qn = rows * N_Q_HEADS * HEAD_DIM * 2;
    need(q, qn, "qsa attend q")?;
    need(sel, rows * TOPK * 4, "qsa attend sel")?;
    need(sel_n, rows * 4, "qsa attend sel_n")?;
    need(out, qn, "qsa attend out")?;
    CudaError::check(unsafe {
        ffi::gb_qsa_attend_merged_ref(
            q.ptr(),
            kv.k.ptr(),
            kv.v.ptr(),
            table.buf.ptr(),
            table.max_blocks as i32,
            rr.slot.ptr(),
            rr.pos.ptr(),
            sel.ptr(),
            sel_n.ptr(),
            out.ptr(),
            rows as i32,
            st.raw(),
        )
    })
}
