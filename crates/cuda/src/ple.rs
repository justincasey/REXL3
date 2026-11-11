//! The PLE stage of Flash-Next: n-gram table row decode, the key/value projections, the stream gate, the dilated conv and
//! its per-slot ring (kernels in `csrc/ple.cu`).
//!
//! Shapes, for `rows` rows of `hc` streams of `d` (the hidden size), `ch = hc * d`:
//! - `packed` u16 `[rows * heads][words]`, `bias` f16 `[heads][160]`, `emb` f16 `[rows][heads * 160]`
//! - `key` f16 `[rows][ch]`, `value` f16 `[rows][d]`, `streams` f32 `[rows][hc][d]`
//! - `sg` f32 `[rows][hc]`, `nb` f16 `[rows][ch]`, `ring` f16 `[slots][ring_len][ch]`
//! - norm weights f32 `[ch]` (already `1 + w`), `conv_w` f16 `[ch][4]`
#![allow(clippy::too_many_arguments)]

use crate::rt::{CudaError, DevBuf, Stream};

/// Elements of one n-gram table row.
pub const ROW: usize = 160;

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
        pub fn gb_ple_rows(packed: *const c_void, bias: *const c_void, emb: *mut c_void, entries: i32, heads: i32, k: i32,
                           words: i32, stream: *mut c_void) -> i32;
        pub fn gb_ple_proj(x: *const c_void, w: *const c_void, y: *mut c_void, rows: i32, k: i32, n: i32,
                           stream: *mut c_void) -> i32;
        pub fn gb_ple_gemm(x: *const c_void, w: *const c_void, y: *mut c_void, rows: i32, k: i32, n: i32,
                           stream: *mut c_void) -> i32;
        pub fn gb_ple_gemm_ref(x: *const c_void, w: *const c_void, y: *mut c_void, rows: i32, k: i32, n: i32,
                               stream: *mut c_void) -> i32;
        pub fn gb_ple_gate(streams: *const c_void, key: *const c_void, value: *const c_void, wk: *const c_void,
                           wq: *const c_void, wc: *const c_void, sg: *mut c_void, nb: *mut c_void, rows: i32, d: i32,
                           hc: i32, eps: f32, scale: f32, stream: *mut c_void) -> i32;
        pub fn gb_ple_conv(streams: *mut c_void, sg: *const c_void, value: *const c_void, nb: *const c_void,
                           ring: *const c_void, conv_w: *const c_void, positions: *const c_void, seq_start: *const c_void,
                           seq_slot: *const c_void, n_seq: i32, rows: i32, d: i32, hc: i32, ring_len: i32,
                           stream: *mut c_void) -> i32;
        pub fn gb_ple_ring(nb: *const c_void, ring: *mut c_void, positions: *const c_void, seq_start: *const c_void,
                           seq_slot: *const c_void, n_seq: i32, rows: i32, ch: i32, ring_len: i32,
                           stream: *mut c_void) -> i32;
    }
}

/// u16 words of one packed `k`-bit table row.
pub fn words_per_row(k: usize) -> usize {
    1 + ROW * k / 16
}

/// Decode `rows * heads` gathered table rows into `emb` (entry `r * heads + h` is head `h` of row `r`).
pub fn rows(
    packed: &DevBuf,
    bias: &DevBuf,
    emb: &mut DevBuf,
    rows: usize,
    heads: usize,
    k: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    let words = words_per_row(k);
    let e = rows * heads;
    need(packed, e * words * 2, "ple rows packed")?;
    need(bias, heads * ROW * 2, "ple rows bias")?;
    need(emb, e * ROW * 2, "ple rows emb")?;
    let ep = emb.ptr();
    CudaError::check(unsafe {
        ffi::gb_ple_rows(packed.ptr(), bias.ptr(), ep, e as i32, heads as i32, k as i32, words as i32, st.raw())
    })
}

/// Rows a window projection takes at most.
pub const WINDOW_ROWS: usize = 64;

/// `y [rows][n] = x [rows][k] . w [n][k]^T`, all f16 with fp32 sums, for a decode or verify window (1..=64 rows): a warp per
/// output column. A row's bits do not depend on the window it is in, so a drafted window equals serial decode.
pub fn proj_window(
    x: &DevBuf,
    w: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    k: usize,
    n: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if !(1..=WINDOW_ROWS).contains(&rows) {
        return Err(CudaError(1));
    }
    need(x, rows * k * 2, "ple proj x")?;
    need(w, n * k * 2, "ple proj w")?;
    need(y, rows * n * 2, "ple proj y")?;
    CudaError::check(unsafe { ffi::gb_ple_proj(x.ptr(), w.ptr(), y.ptr(), rows as i32, k as i32, n as i32, st.raw()) })
}

/// The same product for a prefill chunk of any size: 128 x 128 tiles, each output summed over `k` in ascending order, so a row's
/// bits do not depend on how the prompt is chunked (a short last chunk included). They differ from [`proj_window`]'s, as prefill
/// attention's differ from decode's.
pub fn proj_prefill(
    x: &DevBuf,
    w: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    k: usize,
    n: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    need(x, rows * k * 2, "ple proj x")?;
    need(w, n * k * 2, "ple proj w")?;
    need(y, rows * n * 2, "ple proj y")?;
    CudaError::check(unsafe { ffi::gb_ple_gemm(x.ptr(), w.ptr(), y.ptr(), rows as i32, k as i32, n as i32, st.raw()) })
}

/// [`proj_prefill`]'s previous kernel (64 x 64 tiles, `acc = acc + x * w` as two rounded operations), kept as the oracle of its test.
pub fn proj_prefill_ref(
    x: &DevBuf,
    w: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    k: usize,
    n: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    need(x, rows * k * 2, "ple proj x")?;
    need(w, n * k * 2, "ple proj w")?;
    need(y, rows * n * 2, "ple proj y")?;
    CudaError::check(unsafe { ffi::gb_ple_gemm_ref(x.ptr(), w.ptr(), y.ptr(), rows as i32, k as i32, n as i32, st.raw()) })
}

/// The norms used by [`gate`] (each f32 `[hc * d]`, already `1 + w`).
pub struct GateNorms<'a> {
    pub key: &'a DevBuf,
    pub query: &'a DevBuf,
    pub conv: &'a DevBuf,
}

/// Per (row, stream): the gate `sg` and the conv input `nb` (see the module docs).
pub fn gate(
    streams: &DevBuf,
    key: &DevBuf,
    value: &DevBuf,
    norms: &GateNorms<'_>,
    sg: &mut DevBuf,
    nb: &mut DevBuf,
    rows: usize,
    d: usize,
    hc: usize,
    eps: f32,
    scale: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    let ch = hc * d;
    need(streams, rows * ch * 4, "ple gate streams")?;
    need(key, rows * ch * 2, "ple gate key")?;
    need(value, rows * d * 2, "ple gate value")?;
    need(norms.key, ch * 4, "ple gate norm_key")?;
    need(norms.query, ch * 4, "ple gate norm_query")?;
    need(norms.conv, ch * 4, "ple gate norm_conv")?;
    need(sg, rows * hc * 4, "ple gate sg")?;
    let sp = sg.ptr();
    need(nb, rows * ch * 2, "ple gate nb")?;
    let np = nb.ptr();
    CudaError::check(unsafe {
        ffi::gb_ple_gate(
            streams.ptr(),
            key.ptr(),
            value.ptr(),
            norms.key.ptr(),
            norms.query.ptr(),
            norms.conv.ptr(),
            sp,
            np,
            rows as i32,
            d as i32,
            hc as i32,
            eps,
            scale,
            st.raw(),
        )
    })
}

/// The batch layout the conv and ring kernels read: per row its position, per sequence its first row and slot.
pub struct BatchRows<'a> {
    pub positions: &'a DevBuf,
    pub seq_start: &'a DevBuf,
    pub seq_slot: &'a DevBuf,
    pub n_seq: usize,
    pub rows: usize,
}

/// `streams += sg * value + silu(conv(nb, ring))`, in place.
pub fn conv(
    streams: &mut DevBuf,
    sg: &DevBuf,
    value: &DevBuf,
    nb: &DevBuf,
    ring: &DevBuf,
    conv_w: &DevBuf,
    b: &BatchRows<'_>,
    d: usize,
    hc: usize,
    slots: usize,
    ring_len: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    let (rows, ch) = (b.rows, hc * d);
    need(sg, rows * hc * 4, "ple conv sg")?;
    need(value, rows * d * 2, "ple conv value")?;
    need(nb, rows * ch * 2, "ple conv nb")?;
    need(ring, slots * ring_len * ch * 2, "ple conv ring")?;
    need(conv_w, ch * 4 * 2, "ple conv weight")?;
    need(b.positions, rows * 4, "ple conv positions")?;
    need(b.seq_start, (b.n_seq + 1) * 4, "ple conv seq_start")?;
    need(b.seq_slot, b.n_seq * 4, "ple conv seq_slot")?;
    need(streams, rows * ch * 4, "ple conv streams")?;
    let xp = streams.ptr();
    CudaError::check(unsafe {
        ffi::gb_ple_conv(
            xp,
            sg.ptr(),
            value.ptr(),
            nb.ptr(),
            ring.ptr(),
            conv_w.ptr(),
            b.positions.ptr(),
            b.seq_start.ptr(),
            b.seq_slot.ptr(),
            b.n_seq as i32,
            rows as i32,
            d as i32,
            hc as i32,
            ring_len as i32,
            st.raw(),
        )
    })
}

/// The last `ring_len` positions of every sequence's `nb` rows into its slot ring.
pub fn ring_store(
    nb: &DevBuf,
    ring: &mut DevBuf,
    b: &BatchRows<'_>,
    ch: usize,
    slots: usize,
    ring_len: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    need(nb, b.rows * ch * 2, "ple ring nb")?;
    need(b.positions, b.rows * 4, "ple ring positions")?;
    need(b.seq_start, (b.n_seq + 1) * 4, "ple ring seq_start")?;
    need(b.seq_slot, b.n_seq * 4, "ple ring seq_slot")?;
    need(ring, slots * ring_len * ch * 2, "ple ring")?;
    let rp = ring.ptr();
    CudaError::check(unsafe {
        ffi::gb_ple_ring(
            nb.ptr(),
            rp,
            b.positions.ptr(),
            b.seq_start.ptr(),
            b.seq_slot.ptr(),
            b.n_seq as i32,
            b.rows as i32,
            ch as i32,
            ring_len as i32,
            st.raw(),
        )
    })
}
