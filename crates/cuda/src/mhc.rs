//! mHC (manifold-constrained hyper-connections) for GLM-5.3-Flash: 4 fp32 residual streams of 4096 per row, mixed at every
//! attention and FFN site (kernels in `csrc/mhc.cu`; the design, the arithmetic and the row-invariance argument are in that
//! file's header).
//!
//! Streams are `x[row][stream][d]` fp32, contiguous (`rows * 16384` floats). Everything here is row-invariant by construction:
//! row `r`'s output bits do not depend on how many rows share the call (1..=65535) nor on its position in the batch.
//!
//! # What the model code calls, per layer (4 launches per layer instead of the 8 of the unfused sequence)
//! ```text
//! start of the forward:   expand(embeddings -> x)                 // or embed_expand(table, ids -> x)
//!                         partials(x, layer0.attn_site -> part)
//! for each layer L:
//!     finalize(x, part, attn_site -> hidden = RMSNorm(collapsed), post, comb)      // 1 launch
//!     y = attention(hidden)
//!     apply_partials(x, y, post, comb, ffn_site -> part)                           // 1 launch: apply + the next partials
//!     finalize(x, part, ffn_site -> hidden, post, comb)                            // 1 launch
//!     y = ffn(hidden)
//!     apply_partials(x, y, post, comb, next layer's attn_site -> part)             // 1 launch (apply alone for the last layer)
//! head:                   mean(x, Some(final_norm_w) -> hidden)    // x.mean(dim = 2) then the model's final RMSNorm, one pass
//! ```
//! The unfused kernels (`apply`, `partials`) compute the same bits and are used for the last layer, after a rollback and in
//! the tests. The MTP layer has no hyper-connection and does not use this module. Pass `Tune::AUTO` for the scheduling knobs
//! (chosen from the row count on a GB10); no value of them changes a bit. `MhcWork::new(max_rows)` holds the partials, post and
//! comb buffers; one set serves every site of a forward, and a draft window or a verify window with another row count simply
//! uses the first `rows` entries (row `r` of any call is bit-identical to the same row alone).
//!
//! `hidden` is the layer's RMSNorm output in `out_ty` (the norm weight is passed as given, so fold any `+ 1` first); with `norm_w =
//! None` the plain collapsed row comes back instead. `y` may be f32, f16 or bf16 (widened exactly). The first site's partials come from
//! `partials` (after `expand`); every later site's partials come out of the previous `apply_partials`, so the model owns the order
//! and must call `apply_partials` with the NEXT site's weights.
//!
//! # Safety
//! All launchers are `unsafe`: pointers must be valid device addresses for the sizes implied by the arguments (documented per
//! function), 16-byte aligned (fp16/bf16 outputs 8-byte), and the buffers must stay alive until the stream has run the launch.
//! Bad shapes or alignment return `CudaError(1)` (cudaErrorInvalidValue) and never fault. Zero rows is a no-op.
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use core::ffi::c_void;

use crate::rt::{CudaError, DevBuf, Stream};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};

/// Streams per row (`hc_mult`).
pub const STREAMS: usize = 4;
/// Hidden size of a stream.
pub const HIDDEN: usize = 4096;
/// Values per row: `STREAMS * HIDDEN`.
pub const ROW: usize = STREAMS * HIDDEN;
/// Rows of `fn`: 4 pre, 4 post, 16 comb.
pub const NFN: usize = 24;
/// The pinned K split: atoms per row (128 d x 4 streams each). Part of the numerical contract; never a function of rows.
pub const ATOMS: usize = 32;
/// Floats of partials workspace per row.
pub const PART_FLOATS_PER_ROW: usize = ATOMS * 32;
/// Rows a single launch accepts.
pub const MAX_ROWS: usize = 65535;
/// Rows of a slab of [`apply_partials_finalize`] in a prompt chunk: 256 rows of streams (16 MiB) stay in the GB10's L2 between a slab's
/// apply-partials and its finalize. Measured on the GB10 (`tests/mhc.rs` with `REXL3_BENCH_SLABS=1`), the pair at 2,048 rows: 2.10 ms
/// whole, 1.99 / 1.89 / 1.85 / 1.94 / 2.20 ms in slabs of 128 / 192 / 256 / 320 / 512.
pub const SLAB_ROWS: usize = 256;

/// Element type of an activation / weight buffer handed to a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Ty {
    F32 = 0,
    F16 = 1,
    Bf16 = 2,
}

impl Ty {
    pub fn size(self) -> usize {
        if self == Ty::F32 {
            4
        } else {
            2
        }
    }
}

/// Numeric constants of the mixer (from the model config).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MhcCfg {
    /// `rms_norm_eps`: the epsilon of the unweighted RMS over the 16384 stream values (1e-5 in GLM-5.3-Flash).
    pub rms_eps: f32,
    /// `hc_eps`: added to pre and to the softmax / Sinkhorn denominators (1e-6).
    pub hc_eps: f32,
    /// `hc_sinkhorn_iters` (20): one column normalisation, then `iters - 1` rounds of (row, column).
    pub sinkhorn_iters: i32,
}

impl MhcCfg {
    pub const GLM53: MhcCfg = MhcCfg { rms_eps: 1e-5, hc_eps: 1e-6, sinkhorn_iters: 20 };
}

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_mhc_expand(inp: *const c_void, in_type: i32, x: *mut c_void, rows: i32, s: *mut c_void) -> i32;
        pub fn gb_mhc_embed_expand(table: *const c_void, ids: *const c_void, x: *mut c_void, rows: i32, vocab: i32, s: *mut c_void) -> i32;
        pub fn gb_mhc_partials(
            x: *const c_void,
            fnw: *const c_void,
            fn_type: i32,
            part: *mut c_void,
            rows: i32,
            split: i32,
            rb: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_mhc_apply_partials(
            x: *mut c_void,
            y: *const c_void,
            y_type: i32,
            post: *const c_void,
            comb: *const c_void,
            fn_next: *const c_void,
            fn_type: i32,
            part: *mut c_void,
            rows: i32,
            split: i32,
            rb: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_mhc_finalize(
            x: *const c_void,
            part: *const c_void,
            base: *const c_void,
            scale: *const c_void,
            norm_w: *const c_void,
            post: *mut c_void,
            comb: *mut c_void,
            pre_out: *mut c_void,
            out: *mut c_void,
            out_type: i32,
            rows: i32,
            rms_eps: f32,
            hc_eps: f32,
            norm_eps: f32,
            iters: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_mhc_apply(x: *mut c_void, y: *const c_void, y_type: i32, post: *const c_void, comb: *const c_void, rows: i32, s: *mut c_void)
            -> i32;
        pub fn gb_mhc_mean(x: *const c_void, norm_w: *const c_void, out: *mut c_void, out_type: i32, rows: i32, norm_eps: f32, s: *mut c_void)
            -> i32;
    }
}

fn i(v: usize) -> Result<i32, CudaError> {
    if v > MAX_ROWS {
        return Err(CudaError(1));
    }
    Ok(v as i32)
}

fn vocab_i(v: usize) -> Result<i32, CudaError> {
    i32::try_from(v).map_err(|_| CudaError(1))
}

/// Scheduling knobs of the partials kernels: `split` = warps (m-groups) per atom (1, 2, 4, 8) and `rows_per_block` (1, 2, 4); 0 lets the
/// kernel pick from the row count. Pure latency / bandwidth knobs: every setting gives the same bits (tested).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tune {
    pub split: usize,
    pub rows_per_block: usize,
}

impl Tune {
    pub const AUTO: Tune = Tune { split: 0, rows_per_block: 0 };
}

/// Weights of one mixing site (`layers.N.hc_attn` or `hc_ffn`): `fn` `[24][16384]`, `base` `[24]`, `scale` `[3]`.
///
/// `fn` is kept on the device in the narrowest type that holds every value EXACTLY (fp16, else bf16, else fp32), so the
/// precision is a property of the checkpoint and never of the row count, and halving the dominant decode read costs no bits:
/// on the GLM-5.3-Flash pack all 90 `fn` tensors are exactly fp16 (and bf16) representable. A forced narrow type that would
/// round is refused, never applied silently.
pub struct MhcSite {
    pub fn_buf: DevBuf,
    pub fn_ty: Ty,
    pub base: DevBuf,
    pub scale: DevBuf,
}

/// Narrowest lossless type for `fnw` (fp16 preferred: it is what the ExLlamaV3 decode path uses), and whether each type is exact.
pub fn lossless_types(fnw: &[f32]) -> (bool, bool) {
    let mut f16 = true;
    let mut bf16 = true;
    for &v in fnw {
        if f16 {
            f16 = v.is_finite() && (f16_to_f64(f64_to_f16(v as f64)) as f32).to_bits() == v.to_bits();
        }
        if bf16 {
            bf16 = v.is_finite() && v.to_bits() & 0xFFFF == 0;
        }
        if !f16 && !bf16 {
            break;
        }
    }
    (f16, bf16)
}

impl MhcSite {
    /// Upload a site, choosing the narrowest lossless storage for `fn`.
    pub fn from_f32(fnw: &[f32], base: &[f32], scale: &[f32]) -> Result<MhcSite, String> {
        let (f16, bf16) = lossless_types(fnw);
        let ty = if f16 {
            Ty::F16
        } else if bf16 {
            Ty::Bf16
        } else {
            Ty::F32
        };
        Self::from_f32_as(fnw, base, scale, ty)
    }

    /// Upload a site with `fn` stored as `ty`; fails if `ty` cannot represent every value exactly (use `Ty::F32` to always succeed).
    pub fn from_f32_as(fnw: &[f32], base: &[f32], scale: &[f32], ty: Ty) -> Result<MhcSite, String> {
        if fnw.len() != NFN * ROW || base.len() != NFN || scale.len() != 3 {
            return Err(format!(
                "mhc site: fn {} (want {}), base {} (want {NFN}), scale {} (want 3)",
                fnw.len(),
                NFN * ROW,
                base.len(),
                scale.len()
            ));
        }
        if fnw.iter().chain(base).chain(scale).any(|v| !v.is_finite()) {
            return Err("mhc site: non-finite weight".into());
        }
        let (f16, bf16) = lossless_types(fnw);
        let up = |e: crate::rt::CudaError| format!("mhc site upload: {e}");
        let fn_buf = match ty {
            Ty::F32 => DevBuf::from_slice(fnw).map_err(up)?,
            Ty::F16 => {
                if !f16 {
                    return Err("mhc site: fn is not exactly representable in fp16 (refusing to round it)".into());
                }
                let h: Vec<u16> = fnw.iter().map(|&v| f64_to_f16(v as f64)).collect();
                DevBuf::from_slice(&h).map_err(up)?
            }
            Ty::Bf16 => {
                if !bf16 {
                    return Err("mhc site: fn is not exactly representable in bf16 (refusing to round it)".into());
                }
                let h: Vec<u16> = fnw.iter().map(|&v| (v.to_bits() >> 16) as u16).collect();
                DevBuf::from_slice(&h).map_err(up)?
            }
        };
        Ok(MhcSite {
            fn_buf,
            fn_ty: ty,
            base: DevBuf::from_slice(base).map_err(up)?,
            scale: DevBuf::from_slice(scale).map_err(up)?,
        })
    }

    /// Read `{prefix}_fn`, `{prefix}_base`, `{prefix}_scale` (for example `model.language_model.layers.3.hc_attn`) from a checkpoint
    /// (any of f32, f16, bf16 on disk; widened exactly) and upload them.
    pub fn load(ck: &gb_exl3::checkpoint::Checkpoint, prefix: &str) -> Result<MhcSite, String> {
        let get = |name: &str| -> Result<Vec<f32>, String> {
            let key = format!("{prefix}_{name}");
            let e = ck.plain.get(&key).ok_or_else(|| format!("tensor {key} not in the checkpoint"))?;
            let b = ck.read_tensor(e).map_err(|err| format!("{key}: {err}"))?;
            match e.dtype.as_str() {
                "F32" => Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
                "BF16" => {
                    Ok(b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect())
                }
                "F16" => Ok(b.chunks_exact(2).map(|c| f16_to_f64(u16::from_le_bytes([c[0], c[1]])) as f32).collect()),
                d => Err(format!("{key}: dtype {d}")),
            }
        };
        Self::from_f32(&get("fn")?, &get("base")?, &get("scale")?)
    }

    /// Bytes read per mix at one row: `fn` plus base and scale (what a decode step pays for the weights of a site).
    pub fn weight_bytes(&self) -> usize {
        NFN * ROW * self.fn_ty.size() + (NFN + 3) * 4
    }
}

/// Scratch for the sites of one forward: partials `[rows][32][32]`, post `[rows][4]`, comb `[rows][4][4]` (all f32). One
/// workspace serves every site of a forward (a site's post/comb are consumed by its apply before the next site's finalize).
pub struct MhcWork {
    pub part: DevBuf,
    pub post: DevBuf,
    pub comb: DevBuf,
    pub rows_cap: usize,
}

impl MhcWork {
    pub fn new(rows_cap: usize) -> Result<MhcWork, CudaError> {
        Ok(MhcWork {
            part: DevBuf::zeroed(rows_cap * PART_FLOATS_PER_ROW * 4)?,
            post: DevBuf::zeroed(rows_cap * STREAMS * 4)?,
            comb: DevBuf::zeroed(rows_cap * STREAMS * STREAMS * 4)?,
            rows_cap,
        })
    }
    pub fn bytes(rows_cap: usize) -> usize {
        rows_cap * (PART_FLOATS_PER_ROW + STREAMS + STREAMS * STREAMS) * 4
    }
}

/// `x[r][h][:] = float(inp[r][:])` for the 4 streams (ExpandStreams). `inp` is `[rows][4096]` of `in_ty` (any activation type,
/// widened exactly); `x` is `[rows][4][4096]` f32.
pub unsafe fn expand(inp: *const c_void, in_ty: Ty, x: *mut f32, rows: usize, stream: &Stream) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_mhc_expand(inp, in_ty as i32, x as *mut c_void, i(rows)?, stream.raw()))
}

/// The same from a bf16 embedding table `[vocab][4096]` and i32 token ids `[rows]`; an id outside `0..vocab` gives zeros.
pub unsafe fn embed_expand(
    table: *const c_void,
    ids: *const c_void,
    x: *mut f32,
    rows: usize,
    vocab: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_mhc_embed_expand(table, ids, x as *mut c_void, i(rows)?, vocab_i(vocab)?, stream.raw()))
}

/// Atom partials of `site`'s 24 dots and the sum of squares over `x` (`rows` rows) into `part` (`rows * PART_FLOATS_PER_ROW` f32).
pub unsafe fn partials(
    x: *const f32,
    site: &MhcSite,
    part: *mut f32,
    rows: usize,
    tune: Tune,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_mhc_partials(
        x as *const c_void,
        site.fn_buf.ptr() as *const c_void,
        site.fn_ty as i32,
        part as *mut c_void,
        i(rows)?,
        tune.split as i32,
        tune.rows_per_block as i32,
        stream.raw(),
    ))
}

/// Apply the previous site's update to `x` in place (`x[h] <- post[h] y + sum_g comb[g][h] x[g]`, `y` `[rows][4096]` of `y_ty`) and
/// in the same pass take `next`'s partials of the updated streams. Bit-identical to `apply` followed by `partials`.
pub unsafe fn apply_partials(
    x: *mut f32,
    y: *const c_void,
    y_ty: Ty,
    post: *const f32,
    comb: *const f32,
    next: &MhcSite,
    part: *mut f32,
    rows: usize,
    tune: Tune,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_mhc_apply_partials(
        x as *mut c_void,
        y,
        y_ty as i32,
        post as *const c_void,
        comb as *const c_void,
        next.fn_buf.ptr() as *const c_void,
        next.fn_ty as i32,
        part as *mut c_void,
        i(rows)?,
        tune.split as i32,
        tune.rows_per_block as i32,
        stream.raw(),
    ))
}

/// Reduce the partials, derive `pre`, `post` (`[rows][4]`), `comb` (`[rows][4][4]`, `comb[g][h]` = row g column h) with the
/// Sinkhorn iterations, and collapse the streams. With `norm_w` (f32 `[4096]`, multiplicative) the collapsed row is RMSNorm'ed
/// with `norm_eps` and written to `out` (`[rows][4096]` of `out_ty`, one rounding); without it `out` is the plain collapsed
/// row. `pre_out` (`[rows][4]`) is a test hook.
pub unsafe fn finalize(
    x: *const f32,
    part: *const f32,
    site: &MhcSite,
    norm_w: Option<*const f32>,
    post: *mut f32,
    comb: *mut f32,
    pre_out: Option<*mut f32>,
    out: *mut c_void,
    out_ty: Ty,
    rows: usize,
    cfg: &MhcCfg,
    norm_eps: f32,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_mhc_finalize(
        x as *const c_void,
        part as *const c_void,
        site.base.ptr() as *const c_void,
        site.scale.ptr() as *const c_void,
        norm_w.map_or(core::ptr::null(), |p| p as *const c_void),
        post as *mut c_void,
        comb as *mut c_void,
        pre_out.map_or(core::ptr::null_mut(), |p| p as *mut c_void),
        out,
        out_ty as i32,
        i(rows)?,
        cfg.rms_eps,
        cfg.hc_eps,
        norm_eps,
        cfg.sinkhorn_iters,
        stream.raw(),
    ))
}

/// [`apply_partials`] of `next` followed by `next`'s [`finalize`] (with `norm_w`, into `out` of `out_ty`), in slabs of `slab` rows:
/// each slab's apply-partials, then its finalize, which re-reads the slab's streams while the L2 still holds them (over a whole prompt
/// chunk the finalize reads them back from DRAM). Both kernels are row-independent and `post`, `comb`, `part` and `out` are per row (a
/// slab's finalize overwrites only post and comb rows its own apply has read), so the bits are those of the two calls over every row.
#[allow(clippy::too_many_arguments)]
pub unsafe fn apply_partials_finalize(
    x: *mut f32,
    y: *const c_void,
    y_ty: Ty,
    post: *mut f32,
    comb: *mut f32,
    next: &MhcSite,
    part: *mut f32,
    norm_w: Option<*const f32>,
    out: *mut c_void,
    out_ty: Ty,
    rows: usize,
    slab: usize,
    cfg: &MhcCfg,
    norm_eps: f32,
    stream: &Stream,
) -> Result<(), CudaError> {
    let slab = slab.max(1);
    let mut r0 = 0;
    while r0 < rows {
        let n = (rows - r0).min(slab);
        let xs = x.add(r0 * ROW);
        let (ps, cs, pt) = (post.add(r0 * STREAMS), comb.add(r0 * STREAMS * STREAMS), part.add(r0 * PART_FLOATS_PER_ROW));
        apply_partials(xs, (y as *const u8).add(r0 * HIDDEN * y_ty.size()) as *const c_void, y_ty, ps, cs, next, pt, n, Tune::AUTO, stream)?;
        let o = (out as *mut u8).add(r0 * HIDDEN * out_ty.size()) as *mut c_void;
        finalize(xs, pt, next, norm_w, ps, cs, None, o, out_ty, n, cfg, norm_eps, stream)?;
        r0 += n;
    }
    Ok(())
}

/// `x[h] <- post[h] y + sum_g comb[g][h] x[g]` in place; `y` `[rows][4096]` of `y_ty`.
pub unsafe fn apply(
    x: *mut f32,
    y: *const c_void,
    y_ty: Ty,
    post: *const f32,
    comb: *const f32,
    rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_mhc_apply(
        x as *mut c_void,
        y,
        y_ty as i32,
        post as *const c_void,
        comb as *const c_void,
        i(rows)?,
        stream.raw(),
    ))
}

/// Head collapse: `out = ((x0 + x1) + x2) + x3) * 0.25`, optionally RMSNorm'ed with `norm_w` and `norm_eps` in the same pass.
pub unsafe fn mean(
    x: *const f32,
    norm_w: Option<*const f32>,
    out: *mut c_void,
    out_ty: Ty,
    rows: usize,
    norm_eps: f32,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_mhc_mean(
        x as *const c_void,
        norm_w.map_or(core::ptr::null(), |p| p as *const c_void),
        out,
        out_ty as i32,
        i(rows)?,
        norm_eps,
        stream.raw(),
    ))
}
