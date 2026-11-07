//! The affine 4-bit linear of the GLM DFlash2 drafter (`csrc/q4_linear.cu`): `y [m, n] = x [m, k] . W^T`, `x` and `y` bf16,
//! `W` stored as 4-bit codes with a bf16 scale and bias per 64 inputs of a row (`w ~ s q + b`), fp32 accumulation.
//!
//! The drafter's checkpoint is bf16 (2.3 GB); a draft round reads every linear once, so it is quantized to this format at load
//! ([`quantize_bf16`], on the loader threads) and reads 0.6 GB a round. A draft only proposes tokens (the target's verify
//! decides every one), so the drafter's numerics never change an output.
//!
//! # The layout
//!
//! One device buffer per matrix, three parts laid end to end ([`Q4Weights`]):
//! * codes, `[npad / 64][k / 64][8][32][2]` u32: per 64-row tile and 64-input group, the words lane `l` of the MMA reads for its
//!   n8 tile `j` (row `64 t + 8 j + l / 4`), each word eight codes of inputs `32 v + 2 (l % 4) + {0, 8, 16, 24, 1, 9, 17, 25}`;
//! * scales, `[k / 64][npad]` bf16;
//! * biases, `[k / 64][npad]` bf16.
//!
//! `npad` is `n` rounded up to 64 (the padding rows are zero codes, scales and biases).

use crate::num::{bf16_to_f32, f32_to_bf16};
use crate::rt::{CudaError, DevBuf, Stream};

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_q4_split_k(n: i32, k: i32) -> i32;
        pub fn gb_q4_linear(x: *const c_void, xs: *const f32, w: *const c_void, scales: *const c_void, biases: *const c_void,
                            y: *mut c_void, m: i32, k: i32, n: i32, npad: i32, ldx: i32, stream: *mut c_void) -> i32;
        pub fn gb_q4_gsum(x: *const c_void, ldx: i32, k: i32, xs: *mut f32, rows: i32, stream: *mut c_void) -> i32;
    }
}

/// Inputs a scale group.
pub const GROUP: usize = 64;
/// The code order inside a word: input offsets of nibbles 0..8 (from the MMA's B fragment: k = 2c, 2c + 1 low, 2c + 8, 2c + 9 high).
const OFFS: [usize; 8] = [0, 8, 16, 24, 1, 9, 17, 25];

/// `n` rounded up to the 64-row tile.
pub fn npad_of(n: usize) -> usize {
    n.div_ceil(64) * 64
}

/// Bytes of the codes, the scales and the biases of an `[n, k]` matrix.
pub fn part_bytes(n: usize, k: usize) -> [usize; 3] {
    let p = npad_of(n);
    [p * k / 2, k / GROUP * p * 2, k / GROUP * p * 2]
}

/// Bytes of the whole device buffer of an `[n, k]` matrix.
pub fn bytes(n: usize, k: usize) -> usize {
    part_bytes(n, k).iter().sum()
}

/// Bytes of the group sums an `m`-row call over `k` inputs needs.
pub fn gsum_bytes(m: usize, k: usize) -> usize {
    m * (k / GROUP) * 4
}

/// The scale and bias of one group (bf16 bits) and its codes: `s = (max - min) / 15`, `b = min`, both rounded to bf16 first, then
/// `q = round((w - b) / s)` clamped to 0..=15.
fn quantize_group(v: &[f32], q: &mut [u8]) -> (u16, u16) {
    let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in v {
        mn = mn.min(x);
        mx = mx.max(x);
    }
    let s = f32_to_bf16(if mx > mn { (mx - mn) / 15.0 } else { 1.0 });
    let b = f32_to_bf16(mn);
    let (sf, bf) = (bf16_to_f32(s), bf16_to_f32(b));
    for (o, &x) in q.iter_mut().zip(v) {
        *o = ((x - bf) / sf).round().clamp(0.0, 15.0) as u8;
    }
    (s, b)
}

/// A bf16 `[n, k]` matrix (row-major, as stored in a checkpoint) to the device buffer's bytes ([`bytes`] long). `k % 64 == 0`.
pub fn quantize_bf16(w: &[u16], n: usize, k: usize) -> Vec<u8> {
    assert!(k % GROUP == 0 && w.len() == n * k, "a [{n}, {k}] bf16 matrix with k a multiple of {GROUP}");
    let (npad, kg) = (npad_of(n), k / GROUP);
    let [cb, sb, _] = part_bytes(n, k);
    let mut out = vec![0u8; bytes(n, k)];
    let mut codes = vec![0u8; n * k];
    let mut row = vec![0f32; k];
    for r in 0..n {
        for (o, &x) in row.iter_mut().zip(&w[r * k..(r + 1) * k]) {
            *o = bf16_to_f32(x);
        }
        for g in 0..kg {
            let (s, b) = quantize_group(&row[g * GROUP..(g + 1) * GROUP], &mut codes[r * k + g * GROUP..r * k + (g + 1) * GROUP]);
            let at = (g * npad + r) * 2;
            out[cb + at..cb + at + 2].copy_from_slice(&s.to_le_bytes());
            out[cb + sb + at..cb + sb + at + 2].copy_from_slice(&b.to_le_bytes());
        }
    }
    let words = &mut out[..cb];
    for t in 0..npad / 64 {
        for g in 0..kg {
            for j in 0..8 {
                for lane in 0..32 {
                    let r = t * 64 + j * 8 + lane / 4;
                    if r >= n {
                        continue;
                    }
                    for v in 0..2 {
                        let base = r * k + g * GROUP + 32 * v + 2 * (lane % 4);
                        let mut word = 0u32;
                        for (p, &o) in OFFS.iter().enumerate() {
                            word |= (codes[base + o] as u32) << (4 * p);
                        }
                        let idx = (((t * kg + g) * 8 + j) * 32 + lane) * 2 + v;
                        words[idx * 4..idx * 4 + 4].copy_from_slice(&word.to_le_bytes());
                    }
                }
            }
        }
    }
    out
}

/// The matrix a buffer of [`quantize_bf16`] holds, `[n, k]` f32 (`s q + b` per element): the CPU reference of the kernel.
pub fn dequantize(bytes_: &[u8], n: usize, k: usize) -> Vec<f32> {
    let (npad, kg) = (npad_of(n), k / GROUP);
    let [cb, sb, _] = part_bytes(n, k);
    let u16_at = |off: usize| u16::from_le_bytes([bytes_[off], bytes_[off + 1]]);
    let mut w = vec![0f32; n * k];
    for t in 0..npad / 64 {
        for g in 0..kg {
            for j in 0..8 {
                for lane in 0..32 {
                    let r = t * 64 + j * 8 + lane / 4;
                    if r >= n {
                        continue;
                    }
                    let s = bf16_to_f32(u16_at(cb + (g * npad + r) * 2));
                    let b = bf16_to_f32(u16_at(cb + sb + (g * npad + r) * 2));
                    for v in 0..2 {
                        let idx = (((t * kg + g) * 8 + j) * 32 + lane) * 2 + v;
                        let word = u32::from_le_bytes(bytes_[idx * 4..idx * 4 + 4].try_into().unwrap());
                        let base = g * GROUP + 32 * v + 2 * (lane % 4);
                        for (p, &o) in OFFS.iter().enumerate() {
                            w[r * k + base + o] = s * ((word >> (4 * p)) & 15) as f32 + b;
                        }
                    }
                }
            }
        }
    }
    w
}

/// A quantized `[n, k]` matrix on the device (one buffer: codes, scales, biases).
pub struct Q4Weights {
    pub n: usize,
    pub k: usize,
    pub buf: DevBuf,
}

impl Q4Weights {
    /// Over a device buffer holding [`quantize_bf16`]'s bytes (it may still be filling: a load's pipeline).
    pub fn from_buf(buf: DevBuf, n: usize, k: usize) -> Result<Self, CudaError> {
        if k % GROUP != 0 || buf.bytes() < bytes(n, k) {
            return Err(CudaError(1));
        }
        Ok(Self { n, k, buf })
    }

    /// Device bytes of the matrix.
    pub fn weight_bytes(&self) -> usize {
        bytes(self.n, self.k)
    }

    /// `y [m, n] = x [m, k] . W^T` (`x` rows `ldx` elements apart, `y` packed). `xs` holds [`gsum_bytes`]`(m, k)` bytes of scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_ld(
        &self,
        x: &DevBuf,
        ldx: usize,
        xs: &mut DevBuf,
        y: &mut DevBuf,
        m: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if m == 0 {
            return Ok(());
        }
        let need = |b: &DevBuf, n: usize| if b.bytes() < n { Err(CudaError(1)) } else { Ok(()) };
        need(x, ((m - 1) * ldx + self.k) * 2)?;
        need(xs, gsum_bytes(m, self.k))?;
        need(y, m * self.n * 2)?;
        let [cb, sb, _] = part_bytes(self.n, self.k);
        let i = |v: usize| i32::try_from(v).map_err(|_| CudaError(1));
        unsafe {
            CudaError::check(ffi::gb_q4_gsum(x.ptr(), i(ldx)?, i(self.k)?, xs.ptr() as *mut f32, i(m)?, st.raw()))?;
            CudaError::check(ffi::gb_q4_linear(
                x.ptr(),
                xs.ptr() as *const f32,
                self.buf.ptr(),
                self.buf.ptr_at(cb),
                self.buf.ptr_at(cb + sb),
                y.ptr(),
                i(m)?,
                i(self.k)?,
                i(self.n)?,
                i(npad_of(self.n))?,
                i(ldx)?,
                st.raw(),
            ))
        }
    }

    /// [`Q4Weights::forward_ld`] with packed `x` rows.
    pub fn forward(&self, x: &DevBuf, xs: &mut DevBuf, y: &mut DevBuf, m: usize, st: &Stream) -> Result<(), CudaError> {
        self.forward_ld(x, self.k, xs, y, m, st)
    }
}

/// The K slices the kernel cuts an `[n, k]` matrix into (a function of its shape alone).
pub fn split_k(n: usize, k: usize) -> usize {
    unsafe { ffi::gb_q4_split_k(n as i32, k as i32) as usize }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    #[test]
    fn quantize_then_dequantize_is_within_half_a_step_of_every_weight() {
        let (n, k) = (70, 192);
        let mut seed = 7u64;
        let w: Vec<u16> = (0..n * k).map(|_| f32_to_bf16(lcg(&mut seed) * 0.05)).collect();
        let q = quantize_bf16(&w, n, k);
        assert_eq!(q.len(), bytes(n, k));
        let d = dequantize(&q, n, k);
        for r in 0..n {
            for g in 0..k / GROUP {
                let v: Vec<f32> = (0..GROUP).map(|i| bf16_to_f32(w[r * k + g * GROUP + i])).collect();
                let (mn, mx) = v.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
                let step = (mx - mn) / 15.0;
                for (i, &x) in v.iter().enumerate() {
                    let e = (d[r * k + g * GROUP + i] - x).abs();
                    assert!(e <= 0.5 * step * 1.02 + 1e-3 * mx.abs().max(mn.abs()), "row {r} group {g} input {i}: {x} -> {e}");
                }
            }
        }
    }

    #[test]
    fn a_constant_group_is_exact_and_padding_rows_stay_zero() {
        let (n, k) = (3, 64);
        let w = vec![f32_to_bf16(0.25); n * k];
        let q = quantize_bf16(&w, n, k);
        assert!(dequantize(&q, n, k).iter().all(|&x| x == 0.25));
        let [cb, sb, bb] = part_bytes(n, k);
        assert_eq!((cb, sb, bb), (64 * 64 / 2, 64 * 2, 64 * 2));
        // rows 3..64 of the tile hold nothing
        assert!(q[cb + 3 * 2..cb + sb].iter().all(|&b| b == 0));
    }
}
