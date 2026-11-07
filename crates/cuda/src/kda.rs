//! Kimi delta attention (KDA) kernels for GLM-5.3-Flash: 34 of the 45 layers (`linear_attn_config.kda_layers`).
//!
//! One layer is `heads` (64) x 128 keys x 128 values with a PER-KEY-CHANNEL decay (the GDN of [`crate::gdn`] has one decay per
//! head), a depthwise causal conv of kernel 4 over the `3 * heads * 128` channels `[q | k | v]`, and a sigmoid output gate.
//! This module starts at the layer input `x` and the output of the EXL3 `qkv_proj`, and ends at the input of the EXL3 `o_proj`:
//!
//! ```text
//! x ──proj_in──> b, fa, ga (fp32)     beta = bf16(sigmoid(b))                          (record)
//! fa, ga ─proj_out─> g = exp(lower * sigmoid(exp(A_log[h]) * (fb + dt_bias[c])))      (record)   z = gb (fp32 gate)
//! qkv (f32 or bf16) ──prep──> raw = bf16_rn(qkv) (record), conv(4 taps, fmaf) + silu + bf16_rn, l2norm(q, k), q *= 1/sqrt(128)
//! window: per value row j, per key channel i:  s = s*g_i; mem += s*k_i; delta = (v_j - mem)*beta; s += k_i*delta; y_j = sum s*q
//! y (bf16) ──norm──> ((y * w) * rsqrt(mean(y^2) + eps)) * sigmoid(z)  as fp16  ──> o_proj
//! ```
//!
//! # Exactness by construction
//!
//! * **One update routine.** `update()` in `kda.cu` is compiled `--fmad=false` (every multiply and add rounded alone) and is the
//!   only code that advances the state. A serial decode step, a K+1-row verify window, a prefill row and the replay of an
//!   accepted prefix inline the same instruction sequence on the same operands, so row r of any window == the 1-row step ==
//!   the replayed state, bit for bit. Sums are lane-local in element order followed by an xor butterfly; no atomics.
//! * **Row-invariant.** Value rows are independent of each other (the reduction is over keys only), the reduction orders
//!   do not depend on the number of rows or sequences in the launch, and each row reads only its own inputs and its sequence's
//!   previous 3 raw rows. The GEMVs compute every output with a fixed per-output order. Tests run 1..128 rows per sequence and
//!   1..128 sequences per launch.
//! * **Double-buffered state, committed state read-only during a window.** The pool holds two f32 buffers per slot and a
//!   device-side `cur` index. `window` reads `S[cur]` and writes the state after its LAST row into `S[1-cur]` (from the same
//!   registers, one read and one write pass per layer: 8 MiB at 64 heads); `commit` then, per sequence, flips `cur` when at
//!   least one row is accepted: a FULL accept needs no replay, a partial one replays the accepted prefix from `S[cur]` into
//!   `S[1-cur]` first, `keep = 0` changes nothing. All of it is on the device, so `keep` may come straight from the on-GPU
//!   acceptance walk (no host readback) and the sequence of launches is graph-capturable (no host sync, no allocation, fixed pointers; tested by capturing a whole round once and replaying it with new keeps).
//! * **The record is the window's INPUTS** (not k and v): bf16 raw qkv rows (`2 * C` bytes), the per-channel decay `g` (f32),
//!   `beta` (f32): 82,176 B per layer per row at 64 heads, 2.79 MB per row over 34 layers (the task's `(k, v, g, beta)` record is
//!   82,176 B as well, and would need the raw qkv rows ON TOP, 49,152 B, for the conv shift: 4.47 MB). Replay and the conv shift
//!   both read the raw rows: replay re-runs the same `prep` kernel on the record (bf16 input path, same bits) into scratch,
//!   then the same state kernel. Because nothing else is kept, the commit order matters: replay (reads the OLD conv state),
//!   then conv shift, then the parity flip; [`commit`] does it in that order.
//!
//! # Memory (for the memory model)
//!
//! Per sequence the committed state is `34 x 2 x 64 x 128 x 128 x 4 B = 272 MiB` (two buffers per layer: the price of "a full
//! accept needs no replay"; 136 MiB if single-buffered) plus the conv state `34 x 3 x 24576 x 2 B = 4.8 MiB`. Per window row
//! the record is 2.79 MB over 34 layers. Scratch ([`Workspace`], shared by all layers, transient): q, k f32 and v bf16 per row
//! (`80 KiB`), z f32 (`32 KiB`), y bf16 (`16 KiB`), fa, ga (`1 KiB`), the tiled proj_in's 8 partial sums of each output (`10 KiB`);
//! a pipelined workspace ([`Workspace::add_pipe`]) adds a second q, k, v, z, g, beta set (`Workspace::pipe_bytes_per_row`).
//!
//! # Numerics: what is matched to the oracle, and the decisions
//!
//! The oracle is the ExLlamaV3 fork's decode (BC) graph, built with `--use_fast_math` and a different summation order, so it is
//! matched numerically (cosine / relRMS), never by hash.
//!
//! * **Output rounding: round to nearest even** (the oracle's recurrent kernel rounds the core output TOWARD ZERO to bf16).
//!   Both are implemented (`Round`); RN is the default. RZ is a biased quantiser (mean error -0.5 ulp: a systematic
//!   shrink of every core output by about 0.1-0.4 %) while the semantic references (HF `.to(bfloat16)`, FLA, TensorFold) round to nearest; the
//!   gated RMSNorm that follows cancels most of a uniform scale but not the per-element bias. The measured comparison is in the
//!   tests (`rounding_decision`) and the commit message.
//! * **`dt_bias` is used as stored (f32).** The pack stores f32 `dt_bias[8192]` per layer, and NO value in any KDA layer is
//!   bf16-exact (8192 of 8192 differ in 30 layers; 8191 in the other four), but the oracle's BC graph reads a bf16 copy
//!   (`dt_bias_bc`), perturbing the log-decay argument by up to 2^-9 relative (an absolute shift up to 0.016 in dt_bias, times
//!   `exp(A_log)`). The oracle's torch path uses the f32 value. REXL3 keeps f32. To compare with the BC graph at tight bounds, round `dt_bias`
//!   to bf16 on the host before upload (the tests do).
//! * `f_b`/`g_b` take the fp32 output of the first stage (the BC graph's choice; the torch path rounds it to half).
//! * qkv is rounded to bf16 ONCE from fp32 (RN) before the conv, as the oracle does; the conv taps use `fmaf` in tap order
//!   0..3 (the oracle's order); silu uses IEEE `expf` and division (the oracle: `__expf`, approximate divide): last-ulp differences
//!   before the bf16 rounding.
//!
//! # Prefill
//!
//! [`prefill`] applies the SAME state kernel to the prompt rows, [`PREFILL_PIECE_ROWS`] rows at a time (bounded scratch):
//! slow but exact, and equal bit for bit to decoding the same tokens serially, so it is chunk-invariant (any cut of a prompt
//! into calls gives identical bits) with no extra work. Its measured cost and the plan for a chunked kernel are in `docs/KDA.md`.
//!
//! Preconditions the host cannot cheaply check (violations are undefined behaviour on the device): slots within one launch
//! that writes the state (window, commit) are distinct; `keep[i]` is at most the length of sequence i (it is clamped on the
//! device); `seq_start` is non-decreasing with `seq_start[n_seq] == Batch::rows`; every `slot[i] < n_slots`; `heads % 4 == 0`; activation
//! buffers (`x`, `qkv`, the outputs) are 16-byte aligned with row strides that are multiples of 16 bytes (true for every row of a `DevBuf`).
//! Activations are bf16 stored as `u16`, fp16 as `u16`. Nothing here synchronizes the host or allocates device memory (`prefill` uploads one small
//! descriptor buffer per call).

use crate::exl3_linear::Dtype;
use crate::rt::{CudaError, DevBuf, Event, Stream};

pub const HEAD_DIM: usize = 128;
pub const CONV_TAPS: usize = 4;
/// Raw qkv rows kept per slot as conv history.
pub const CONV_STATE_ROWS: usize = 3;
/// GLM-5.3-Flash: 64 heads, hidden 4096.
pub const GLM_HEADS: usize = 64;
pub const GLM_HIDDEN: usize = 4096;
/// Rank of the low-rank f/g projections (fixed at the head dimension).
pub const RANK: usize = 128;
/// Partial sums per output of the tiled proj_in (`csrc/kda.cu` proj_in_tile_kernel: one per warp of proj_in_kernel's 8).
const PROJ_IN_PARTS: usize = 8;
/// Rows [`prefill`] processes per piece (the scratch and the record are sized for this many).
pub const PREFILL_PIECE_ROWS: usize = 128;

/// Geometry of one KDA layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub heads: usize,
    pub hidden: usize,
}

impl Dims {
    pub fn glm() -> Self {
        Self { heads: GLM_HEADS, hidden: GLM_HIDDEN }
    }
    /// `heads * 128`: width of v, of the core output and of the norm output.
    pub fn v_dim(&self) -> usize {
        self.heads * HEAD_DIM
    }
    /// `3 * heads * 128`: conv channels, `[q | k | v]`.
    pub fn conv_dim(&self) -> usize {
        3 * self.heads * HEAD_DIM
    }
    /// f32 elements of ONE state buffer of a slot (`heads x 128 value rows x 128 keys`).
    pub fn s_buf_elems(&self) -> usize {
        self.heads * HEAD_DIM * HEAD_DIM
    }
    /// bf16 elements of the conv state of a slot.
    pub fn conv_slot_elems(&self) -> usize {
        CONV_STATE_ROWS * self.conv_dim()
    }
    /// Bytes of one window row's record in one layer.
    pub fn record_bytes_per_row(&self) -> usize {
        2 * self.conv_dim() + 4 * self.v_dim() + 4 * self.heads
    }
    /// Committed bytes per slot and layer: both state buffers, the conv state and the parity word.
    pub fn slot_bytes(&self) -> usize {
        2 * self.s_buf_elems() * 4 + self.conv_slot_elems() * 2 + 4
    }
    fn check(&self) {
        assert!(self.heads > 0 && self.heads % 4 == 0 && self.heads <= 1024, "kda: heads must be a multiple of 4");
        assert!(self.hidden > 0 && self.hidden % 8 == 0, "kda: hidden must be a multiple of 8");
    }
}

/// Rounding of the core output (the read-out `q . S`) to bf16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Round {
    /// Round to nearest even: REXL3's choice (see the module docs).
    NearestEven,
    /// Round toward zero: what the ExLlamaV3 recurrent kernel does.
    TowardZero,
}

/// Layer constants and kernel choices.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// `gate_lower_bound` (-5): log decay = lower * sigmoid(..).
    pub lower: f32,
    /// RMSNorm epsilon of `o_norm` (the model's `rms_norm_eps`, 1e-5).
    pub eps: f32,
    pub round: Round,
    /// Value rows per warp in the state kernel (4 or 8); a performance knob, never a bits knob.
    pub r_step: usize,
}

impl Params {
    pub fn glm() -> Self {
        Self { lower: -5.0, eps: 1e-5, round: Round::NearestEven, r_step: 8 }
    }
}

/// The layer's weights, borrowed from device buffers the caller owns.
pub struct Weights<'a> {
    /// f16 `[heads, hidden]`
    pub b: &'a DevBuf,
    /// f16 `[128, hidden]`
    pub f_a: &'a DevBuf,
    /// f16 `[128, hidden]`
    pub g_a: &'a DevBuf,
    /// f16 `[heads * 128, 128]`
    pub f_b: &'a DevBuf,
    /// f16 `[heads * 128, 128]`
    pub g_b: &'a DevBuf,
    /// bf16 `[conv_dim, 4]`: `weight[ch, 0, k]` (tap k multiplies x[t-3+k])
    pub conv_w: &'a DevBuf,
    /// f32 `[heads]`
    pub a_log: &'a DevBuf,
    /// f32 `[heads * 128]`
    pub dt_bias: &'a DevBuf,
    /// bf16 `[128]`
    pub o_norm: &'a DevBuf,
}

/// One KDA layer's weights uploaded to the device exactly as the pack stores them (no transpose, no conversion); [`LayerWeights::view`]
/// is what the kernels take.
pub struct LayerWeights {
    b: DevBuf,
    f_a: DevBuf,
    g_a: DevBuf,
    f_b: DevBuf,
    g_b: DevBuf,
    conv_w: DevBuf,
    a_log: DevBuf,
    dt_bias: DevBuf,
    o_norm: DevBuf,
}

impl LayerWeights {
    /// Reads `model.language_model.layers.{layer}.self_attn.*` from a scanned pack. Fail-closed: a missing tensor, or one whose dtype or shape is not
    /// what `d` implies (f16 `b_proj`, `f_a_proj`, `g_a_proj`, `f_b_proj`, `g_b_proj`; bf16 `conv1d.weight` `[conv_dim, 1, 4]` and `o_norm.weight`; f32
    /// `A_log` and `dt_bias`), is an error naming it.
    pub fn from_checkpoint(ck: &gb_exl3::checkpoint::Checkpoint, layer: usize, d: Dims) -> Result<Self, String> {
        d.check();
        let get = |name: &str, dtype: &str, shape: &[usize]| -> Result<DevBuf, String> {
            let full = format!("model.language_model.layers.{layer}.self_attn.{name}");
            let e = ck.plain.get(&full).ok_or_else(|| format!("{full} is missing"))?;
            if e.dtype != dtype || e.shape != shape {
                return Err(format!("{full} is {} {:?}, expected {dtype} {shape:?}", e.dtype, e.shape));
            }
            let bytes = ck.read_tensor(e).map_err(|x| format!("{full}: {x}"))?;
            DevBuf::from_slice(&bytes).map_err(|x| format!("{full}: upload failed: {x}"))
        };
        Ok(Self {
            b: get("b_proj.weight", "F16", &[d.heads, d.hidden])?,
            f_a: get("f_a_proj.weight", "F16", &[RANK, d.hidden])?,
            g_a: get("g_a_proj.weight", "F16", &[RANK, d.hidden])?,
            f_b: get("f_b_proj.weight", "F16", &[d.v_dim(), RANK])?,
            g_b: get("g_b_proj.weight", "F16", &[d.v_dim(), RANK])?,
            conv_w: get("conv1d.weight", "BF16", &[d.conv_dim(), 1, CONV_TAPS])?,
            a_log: get("A_log", "F32", &[d.heads])?,
            dt_bias: get("dt_bias", "F32", &[d.v_dim()])?,
            o_norm: get("o_norm.weight", "BF16", &[HEAD_DIM])?,
        })
    }

    /// The weights of heads `[h0, h1)` of a layer of geometry `d` (lane E, the TP head split: a rank of 32 heads runs the kernels with
    /// `Dims { heads: h1 - h0, .. }` on these): the per-head rows of `b`, `f_b`, `g_b`, `A_log` and `dt_bias`, the heads' channels of each
    /// of the conv's `[q | k | v]` thirds, and the shared `f_a`, `g_a` (the low-rank projections, rank 128) and `o_norm` whole. Device to
    /// device copies on `st` (synchronised before returning); every kernel computes a head from these alone, so the heads' outputs and
    /// state are the full layer's bit for bit (`crates/cuda/tests/kda_split.rs`).
    pub fn heads_of(w: &Weights<'_>, d: Dims, h0: usize, h1: usize, st: &Stream) -> Result<Self, CudaError> {
        d.check();
        let nh = h1.checked_sub(h0).filter(|&n| n > 0 && n % 4 == 0 && h1 <= d.heads).ok_or(CudaError(1))?;
        let part = |src: &DevBuf, ranges: &[(usize, usize)]| -> Result<DevBuf, CudaError> {
            let mut dst = DevBuf::alloc(ranges.iter().map(|r| r.1).sum::<usize>())?;
            let mut at = 0;
            for &(off, len) in ranges {
                dst.copy_range_from(at, src, off, len, st)?;
                at += len;
            }
            Ok(dst)
        };
        let whole = |src: &DevBuf| part(src, &[(0, src.bytes())]);
        let vd = HEAD_DIM; // channels a head
        let c = d.v_dim(); // channels of one third of the conv
        let conv_row = CONV_TAPS * 2; // bf16 taps a channel
        let out = Self {
            b: part(w.b, &[(h0 * d.hidden * 2, nh * d.hidden * 2)])?,
            f_a: whole(w.f_a)?,
            g_a: whole(w.g_a)?,
            f_b: part(w.f_b, &[(h0 * vd * RANK * 2, nh * vd * RANK * 2)])?,
            g_b: part(w.g_b, &[(h0 * vd * RANK * 2, nh * vd * RANK * 2)])?,
            conv_w: part(
                w.conv_w,
                &[
                    (h0 * vd * conv_row, nh * vd * conv_row),
                    ((c + h0 * vd) * conv_row, nh * vd * conv_row),
                    ((2 * c + h0 * vd) * conv_row, nh * vd * conv_row),
                ],
            )?,
            a_log: part(w.a_log, &[(h0 * 4, nh * 4)])?,
            dt_bias: part(w.dt_bias, &[(h0 * vd * 4, nh * vd * 4)])?,
            o_norm: whole(w.o_norm)?,
        };
        st.sync()?;
        Ok(out)
    }

    pub fn view(&self) -> Weights<'_> {
        Weights {
            b: &self.b,
            f_a: &self.f_a,
            g_a: &self.g_a,
            f_b: &self.f_b,
            g_b: &self.g_b,
            conv_w: &self.conv_w,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            o_norm: &self.o_norm,
        }
    }
}

/// Sequences of one launch (device arrays the caller owns): `seq_start` i32 `[n_seq + 1]`, `slot` i32 `[n_seq]`.
#[derive(Clone, Copy)]
pub struct Batch<'a> {
    pub seq_start: &'a DevBuf,
    pub slot: &'a DevBuf,
    pub n_seq: usize,
    /// Total rows (`seq_start[n_seq]`).
    pub rows: usize,
}

/// An owning [`Batch`] built from host lengths and slots (uploads once; for tests, benches and non-hot paths).
pub struct BatchBuf {
    seq_start: DevBuf,
    slot: DevBuf,
    n_seq: usize,
    rows: usize,
}

impl BatchBuf {
    pub fn new(lens: &[usize], slots: &[usize]) -> Result<Self, CudaError> {
        assert_eq!(lens.len(), slots.len());
        let mut start = vec![0i32; lens.len() + 1];
        for (i, &l) in lens.iter().enumerate() {
            start[i + 1] = start[i] + l as i32;
        }
        let slot: Vec<i32> = slots.iter().map(|&s| s as i32).collect();
        Ok(Self {
            seq_start: DevBuf::from_slice(&start)?,
            slot: DevBuf::from_slice(&slot)?,
            n_seq: lens.len(),
            rows: start[lens.len()] as usize,
        })
    }
    pub fn view(&self) -> Batch<'_> {
        Batch { seq_start: &self.seq_start, slot: &self.slot, n_seq: self.n_seq, rows: self.rows }
    }
}

/// The committed recurrent state of every slot of one layer: two f32 state buffers, a device-side index of the current one, and
/// the conv history.
pub struct Pool {
    pub dims: Dims,
    pub n_slots: usize,
    /// f32 `[slots][2][heads][128 value row][128 key]`
    pub s: DevBuf,
    /// i32 `[slots]`: index (0 or 1) of the current buffer
    pub parity: DevBuf,
    /// bf16 `[slots][3][conv_dim]`: the last 3 raw qkv rows, oldest first
    pub conv: DevBuf,
}

impl Pool {
    pub fn new(dims: Dims, n_slots: usize) -> Result<Self, CudaError> {
        dims.check();
        Ok(Self {
            dims,
            n_slots,
            s: DevBuf::zeroed(n_slots * 2 * dims.s_buf_elems() * 4)?,
            parity: DevBuf::zeroed(n_slots * 4)?,
            conv: DevBuf::zeroed(n_slots * dims.conv_slot_elems() * 2)?,
        })
    }
    /// Start a new sequence in `slot`: zero both state buffers and the conv history, asynchronously on `st`. The parity word is
    /// left alone (either buffer may be current, both are zero), so this is safe to enqueue between a window and its commit
    /// only for OTHER slots.
    pub fn reset_slot(&mut self, slot: usize, st: &Stream) -> Result<(), CudaError> {
        assert!(slot < self.n_slots);
        let sb = 2 * self.dims.s_buf_elems() * 4;
        self.s.memset_range_async(slot * sb, sb, 0, st)?;
        let cb = self.dims.conv_slot_elems() * 2;
        self.conv.memset_range_async(slot * cb, cb, 0, st)
    }
    /// The slot's CURRENT state, f32 `[heads][128][128]`.
    pub fn read_state(&self, slot: usize) -> Result<Vec<f32>, CudaError> {
        let par: Vec<i32> = self.parity.to_vec()?;
        let cur = (par[slot] & 1) as usize;
        let e = self.dims.s_buf_elems();
        let all: Vec<f32> = self.s.to_vec()?;
        Ok(all[(slot * 2 + cur) * e..(slot * 2 + cur + 1) * e].to_vec())
    }
    /// The exact per-slot recurrent state at a committed prefix: both state buffers (f32), the parity word and
    /// the conv history (bf16). `restore_slot_state` puts back exactly this.
    pub fn slot_snapshot(&self, slot: usize) -> Result<(Vec<f32>, i32, Vec<u16>), CudaError> {
        assert!(slot < self.n_slots);
        let e = self.dims.s_buf_elems();
        let sb = 2 * e * 4;
        let mut s = vec![0f32; 2 * e];
        self.s.download_range(slot * sb, &mut s)?;
        let mut par = vec![0i32; 1];
        self.parity.download_range(slot * 4, &mut par)?;
        let ce = self.dims.conv_slot_elems();
        let mut conv = vec![0u16; ce];
        self.conv.download_range(slot * ce * 2, &mut conv)?;
        Ok((s, par[0], conv))
    }
    /// Restore the exact per-slot recurrent state captured by `slot_snapshot` (both buffers, parity, conv).
    pub fn restore_slot_state(
        &mut self,
        slot: usize,
        s: &[f32],
        parity: i32,
        conv: &[u16],
        st: &Stream,
    ) -> Result<(), CudaError> {
        assert!(slot < self.n_slots);
        let e = self.dims.s_buf_elems();
        assert_eq!(s.len(), 2 * e);
        assert_eq!(conv.len(), self.dims.conv_slot_elems());
        let sb = 2 * e * 4;
        let cb = conv.len() * 2;
        unsafe {
            st.memcpy_h2d_async(self.s.ptr_at(slot * sb), s.as_ptr() as *const u8, sb)?;
            st.memcpy_h2d_async(self.parity.ptr_at(slot * 4), (&parity as *const i32) as *const u8, 4)?;
            st.memcpy_h2d_async(self.conv.ptr_at(slot * cb), conv.as_ptr() as *const u8, cb)?;
        }
        Ok(())
    }
    /// Both buffers of a slot (current first is NOT guaranteed): f32 `[2][heads][128][128]`.
    pub fn read_both(&self, slot: usize) -> Result<Vec<f32>, CudaError> {
        let e = self.dims.s_buf_elems();
        let all: Vec<f32> = self.s.to_vec()?;
        Ok(all[slot * 2 * e..(slot + 1) * 2 * e].to_vec())
    }
    pub fn read_parity(&self) -> Result<Vec<i32>, CudaError> {
        self.parity.to_vec()
    }
    pub fn read_conv(&self, slot: usize) -> Result<Vec<u16>, CudaError> {
        let e = self.dims.conv_slot_elems();
        let all: Vec<u16> = self.conv.to_vec()?;
        Ok(all[slot * e..(slot + 1) * e].to_vec())
    }
    /// Overwrite the slot: `state` goes into the current buffer, the other buffer is zeroed, `conv` is the 3 history rows.
    pub fn write_slot(&mut self, slot: usize, state: &[f32], conv: &[u16]) -> Result<(), CudaError> {
        let e = self.dims.s_buf_elems();
        assert_eq!(state.len(), e);
        assert_eq!(conv.len(), self.dims.conv_slot_elems());
        let par: Vec<i32> = self.parity.to_vec()?;
        let cur = (par[slot] & 1) as usize;
        let mut all: Vec<f32> = self.s.to_vec()?;
        all[(slot * 2 + cur) * e..(slot * 2 + cur + 1) * e].copy_from_slice(state);
        all[(slot * 2 + (cur ^ 1)) * e..(slot * 2 + (cur ^ 1) + 1) * e].fill(0.0);
        self.s.upload(&all)?;
        let ce = self.dims.conv_slot_elems();
        let mut call: Vec<u16> = self.conv.to_vec()?;
        call[slot * ce..(slot + 1) * ce].copy_from_slice(conv);
        self.conv.upload(&call)
    }
}

/// What a window leaves behind for the commit: the raw (bf16-rounded) qkv rows, the per-channel decay and beta of each row.
pub struct Record {
    pub dims: Dims,
    pub rows: usize,
    /// bf16 `[rows, conv_dim]`
    pub qkv: DevBuf,
    /// f32 `[rows, heads * 128]`: the decay factor (exp of the log decay) per key channel
    pub g: DevBuf,
    /// f32 `[rows, heads]`: bf16(sigmoid(b)) held in f32
    pub beta: DevBuf,
}

impl Record {
    pub fn alloc(dims: Dims, rows: usize) -> Result<Self, CudaError> {
        dims.check();
        Ok(Self {
            dims,
            rows,
            qkv: DevBuf::alloc(rows * dims.conv_dim() * 2)?,
            g: DevBuf::alloc(rows * dims.v_dim() * 4)?,
            beta: DevBuf::alloc(rows * dims.heads * 4)?,
        })
    }
}

/// Transient scratch, shared by all layers (a layer's window and its commit are not interleaved with another layer's).
pub struct Workspace {
    pub dims: Dims,
    pub rows: usize,
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    fa: DevBuf,
    ga: DevBuf,
    z: DevBuf,
    y: DevBuf,
    /// f32 `[8][rows][heads + 2 * RANK]`: the tiled proj_in's per-warp partial sums
    part: DevBuf,
    pipe: Option<Pipe>,
}

/// The second buffer set and the side stream of the pipelined P2 prefill ([`prefill_chunked`] on bf16 qkv rows after
/// [`Workspace::add_pipe`]): the projections, prep and conv shift of piece `i + 1` run on `side` into one set while the main stream runs
/// piece `i`'s chunked recurrence and norm on the other. Set 0 is the workspace's own q, k, v, z with the record's g and beta.
struct Pipe {
    side: Stream,
    start: Event,
    ready: [Event; 2],
    free: [Event; 2],
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    z: DevBuf,
    g: DevBuf,
    beta: DevBuf,
}

impl Workspace {
    pub fn alloc(dims: Dims, rows: usize) -> Result<Self, CudaError> {
        dims.check();
        Ok(Self {
            dims,
            rows,
            q: DevBuf::alloc(rows * dims.v_dim() * 4)?,
            k: DevBuf::alloc(rows * dims.v_dim() * 4)?,
            v: DevBuf::alloc(rows * dims.v_dim() * 2)?,
            fa: DevBuf::alloc(rows * RANK * 4)?,
            ga: DevBuf::alloc(rows * RANK * 4)?,
            z: DevBuf::alloc(rows * dims.v_dim() * 4)?,
            y: DevBuf::alloc(rows * dims.v_dim() * 2)?,
            part: DevBuf::alloc(rows * PROJ_IN_PARTS * (dims.heads + 2 * RANK) * 4)?,
            pipe: None,
        })
    }
    /// Scratch bytes per row.
    pub fn bytes_per_row(dims: Dims) -> usize {
        dims.v_dim() * (4 + 4 + 2 + 4 + 2) + 2 * RANK * 4 + PROJ_IN_PARTS * (dims.heads + 2 * RANK) * 4
    }
    /// Bytes per row of the pipelined prefill's second buffer set ([`Workspace::add_pipe`]): q, k, z, g f32, v bf16, beta f32.
    pub fn pipe_bytes_per_row(dims: Dims) -> usize {
        dims.v_dim() * (4 + 4 + 2 + 4 + 4) + dims.heads * 4
    }
    /// Give [`prefill_chunked`] a second buffer set of this workspace's rows and a side stream: on bf16 qkv rows it then overlaps each
    /// piece's projections, prep and conv shift with the previous piece's recurrence. Every kernel reads and writes what it did in the
    /// serial order (the events order each set's reuse), so not a bit moves.
    pub fn add_pipe(&mut self) -> Result<(), CudaError> {
        if self.pipe.is_some() {
            return Ok(());
        }
        let (d, rows) = (self.dims, self.rows);
        self.pipe = Some(Pipe {
            side: Stream::new_nonblocking()?,
            start: Event::new()?,
            ready: [Event::new()?, Event::new()?],
            free: [Event::new()?, Event::new()?],
            q: DevBuf::alloc(rows * d.v_dim() * 4)?,
            k: DevBuf::alloc(rows * d.v_dim() * 4)?,
            v: DevBuf::alloc(rows * d.v_dim() * 2)?,
            z: DevBuf::alloc(rows * d.v_dim() * 4)?,
            g: DevBuf::alloc(rows * d.v_dim() * 4)?,
            beta: DevBuf::alloc(rows * d.heads * 4)?,
        });
        Ok(())
    }
    /// Whether [`Workspace::add_pipe`] has run.
    pub fn pipelined(&self) -> bool {
        self.pipe.is_some()
    }
    /// The core output of the last [`window`], bf16 `[rows, heads * 128]`.
    pub fn y(&self) -> &DevBuf {
        &self.y
    }
    /// The output gate of the last [`project`], f32 `[rows, heads * 128]`.
    pub fn z(&self) -> &DevBuf {
        &self.z
    }
    /// Mutable access to the output gate buffer (the full-rank KDA gate writes it directly).
    pub fn z_mut(&mut self) -> &mut DevBuf {
        &mut self.z
    }
    /// q (scaled by 1/sqrt(128), l2-normalised) of the last [`prep`], f32 `[rows, heads, 128]`.
    pub fn q(&self) -> &DevBuf {
        &self.q
    }
    /// k (l2-normalised) of the last [`prep`], f32 `[rows, heads, 128]`.
    pub fn k(&self) -> &DevBuf {
        &self.k
    }
    /// v (conv + silu, bf16) of the last [`prep`], bf16 `[rows, heads, 128]`.
    pub fn v(&self) -> &DevBuf {
        &self.v
    }
}

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_kda_proj_in(
            x: *const c_void,
            x_dtype: i32,
            w_b: *const c_void,
            w_fa: *const c_void,
            w_ga: *const c_void,
            rows: i32,
            hidden: i32,
            heads: i32,
            rank: i32,
            beta: *mut f32,
            fa: *mut f32,
            ga: *mut f32,
            part: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_kda_proj_out(
            fa: *const f32,
            ga: *const f32,
            w_fb: *const c_void,
            w_gb: *const c_void,
            a_log: *const f32,
            dt_bias: *const f32,
            lower: f32,
            rows: i32,
            heads: i32,
            g: *mut f32,
            z: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_kda_gate_from_full_fg(
            f_out: *const c_void,
            g_out: *const c_void,
            x: *const c_void,
            b_weight: *const c_void,
            a_log: *const f32,
            dt_bias: *const f32,
            lower: f32,
            g_decay_out: *mut f32,
            z_gate_out: *mut f32,
            beta_out: *mut c_void,
            rows: i32,
            hidden: i32,
            heads: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_kda_prep(
            qkv: *const c_void,
            in_f32: i32,
            conv_w: *const c_void,
            conv_pool: *const c_void,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            rows: i32,
            heads: i32,
            emit_q: i32,
            skip_full: i32,
            raw_out: *mut c_void,
            q: *mut f32,
            k: *mut f32,
            v: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_kda_window(
            q: *const f32,
            k: *const f32,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            parity: *const i32,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            heads: i32,
            rz: i32,
            write_final: i32,
            r_step: i32,
            y: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_kda_replay(
            k: *const f32,
            v: *const c_void,
            g: *const f32,
            beta: *const f32,
            s: *mut f32,
            parity: *const i32,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            heads: i32,
            skip_full: i32,
            r_step: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_kda_norm(
            y: *const c_void,
            z: *const f32,
            w: *const c_void,
            eps: f32,
            rows: i32,
            heads: i32,
            out: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_kda_reduce_check(input: *const f32, out: *mut f32, r: i32, stream: *mut c_void) -> i32;
        pub fn gb_glm_p2_kda_chunk(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            g: *const c_void,
            beta: *const c_void,
            s: *mut c_void,
            parity: *const c_void,
            slot: i32,
            rows: i32,
            heads: i32,
            vs: i32,
            rz: i32,
            y: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_kda_commit_tail(
            raw: *const c_void,
            conv_pool: *mut c_void,
            parity: *mut i32,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            heads: i32,
            flip: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_kda_flip(parity: *mut i32, slot: *const i32, n_seq: i32, stream: *mut c_void) -> i32;
    }
}

fn need(buf: &DevBuf, bytes: usize, what: &str) {
    assert!(buf.bytes() >= bytes, "kda: {what} holds {} bytes, needs {bytes}", buf.bytes());
}

fn check_weights(d: Dims, w: &Weights) {
    need(w.b, d.heads * d.hidden * 2, "b_proj");
    need(w.f_a, RANK * d.hidden * 2, "f_a_proj");
    need(w.g_a, RANK * d.hidden * 2, "g_a_proj");
    need(w.f_b, d.v_dim() * RANK * 2, "f_b_proj");
    need(w.g_b, d.v_dim() * RANK * 2, "g_b_proj");
    need(w.conv_w, d.conv_dim() * CONV_TAPS * 2, "conv_w");
    need(w.a_log, d.heads * 4, "a_log");
    need(w.dt_bias, d.v_dim() * 4, "dt_bias");
    need(w.o_norm, HEAD_DIM * 2, "o_norm");
}

/// Raw launches over `rows` rows starting at row offsets into the caller's buffers: the typed API below and [`prefill`] share
/// them so the piece loop adds no logic of its own.
struct Sel<'a> {
    seq_start: *const i32,
    slot: *const i32,
    n_seq: usize,
    rows: usize,
    _life: core::marker::PhantomData<&'a ()>,
}

impl<'a> Sel<'a> {
    fn of(b: &Batch<'a>) -> Self {
        Self {
            seq_start: b.seq_start.ptr() as *const i32,
            slot: b.slot.ptr() as *const i32,
            n_seq: b.n_seq,
            rows: b.rows,
            _life: core::marker::PhantomData,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn project_raw(
    st: &Stream,
    d: Dims,
    w: &Weights,
    p: &Params,
    x: *const core::ffi::c_void,
    x_dtype: Dtype,
    rows: usize,
    rec_g: *mut f32,
    rec_beta: *mut f32,
    ws: &Workspace,
    z: *mut f32,
) -> Result<(), CudaError> {
    project_stages_raw(st, d, w, p, x, x_dtype, rows, rec_g, rec_beta, ws, z, 3)
}

/// `stages`: bit 0 = the b / f_a / g_a GEMVs, bit 1 = f_b / g_b and the gate (needs the first stage's output).
#[allow(clippy::too_many_arguments)]
fn project_stages_raw(
    st: &Stream,
    d: Dims,
    w: &Weights,
    p: &Params,
    x: *const core::ffi::c_void,
    x_dtype: Dtype,
    rows: usize,
    rec_g: *mut f32,
    rec_beta: *mut f32,
    ws: &Workspace,
    z: *mut f32,
    stages: u8,
) -> Result<(), CudaError> {
    let fa = ws.fa.ptr() as *mut f32;
    let ga = ws.ga.ptr() as *mut f32;
    if stages & 1 != 0 {
        let rc = unsafe {
            ffi::gb_kda_proj_in(
                x,
                x_dtype as i32,
                w.b.ptr(),
                w.f_a.ptr(),
                w.g_a.ptr(),
                rows as i32,
                d.hidden as i32,
                d.heads as i32,
                RANK as i32,
                rec_beta,
                fa,
                ga,
                ws.part.ptr() as *mut f32,
                st.raw(),
            )
        };
        CudaError::check(rc)?;
    }
    if stages & 2 == 0 {
        return Ok(());
    }
    let rc = unsafe {
        ffi::gb_kda_proj_out(
            fa,
            ga,
            w.f_b.ptr(),
            w.g_b.ptr(),
            w.a_log.ptr() as *const f32,
            w.dt_bias.ptr() as *const f32,
            p.lower,
            rows as i32,
            d.heads as i32,
            rec_g,
            z,
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// Benchmark hook: [`project`] restricted to `stages` (1 = b / f_a / g_a, 2 = f_b / g_b + gate).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn project_stage(
    st: &Stream,
    d: Dims,
    w: &Weights,
    p: &Params,
    x: &DevBuf,
    x_dtype: Dtype,
    rows: usize,
    rec: &mut Record,
    ws: &mut Workspace,
    stages: u8,
) -> Result<(), CudaError> {
    d.check();
    check_weights(d, w);
    assert!(rows <= rec.rows && rows <= ws.rows);
    project_stages_raw(
        st,
        d,
        w,
        p,
        x.ptr(),
        x_dtype,
        rows,
        rec.g.ptr() as *mut f32,
        rec.beta.ptr() as *mut f32,
        ws,
        ws.z.ptr() as *mut f32,
        stages,
    )
}

/// The low-rank projections and the gates for `rows` rows of `x` (`[rows, hidden]`, f16, bf16 or f32): fills `rec.beta`, `rec.g`
/// (rows `0..rows`) and `ws.z`. `x` rows are independent of each other; the bits of a row do not depend on `rows`.
#[allow(clippy::too_many_arguments)]
pub fn project(
    st: &Stream,
    d: Dims,
    w: &Weights,
    p: &Params,
    x: &DevBuf,
    x_dtype: Dtype,
    rows: usize,
    rec: &mut Record,
    ws: &mut Workspace,
) -> Result<(), CudaError> {
    d.check();
    check_weights(d, w);
    assert!(rows <= rec.rows && rows <= ws.rows, "kda::project: {rows} rows exceed the record/workspace");
    need(x, rows * d.hidden * x_dtype.size(), "x");
    project_raw(
        st,
        d,
        w,
        p,
        x.ptr(),
        x_dtype,
        rows,
        rec.g.ptr() as *mut f32,
        rec.beta.ptr() as *mut f32,
        ws,
        ws.z.ptr() as *mut f32,
    )
}

/// The full-rank KDA gate of Ling-3.1-flash-1019 (`no_kda_lora`): `f_out` / `g_out` fp16 `[rows, heads * 128]`
/// (full-rank EXL3 `f_proj` / `g_proj` outputs), `x` bf16 `[rows, hidden]`, `b_weight` bf16 `[heads, hidden]`,
/// `a_log` f32 `[heads]`, `dt_bias` f32 `[heads * 128]`, `lower` = `kda_lower_bound` (-5.0). Fills `g_decay`
/// f32 `[rows, heads * 128]` (the per-channel decay), `z_gate` f32 `[rows, heads * 128]` (the output gate) and
/// `beta` f32 `[rows, heads]` (the bf16-rounded sigmoid widened to f32); the existing `window` / `replay` / `norm` run unmodified on them.
#[allow(clippy::too_many_arguments)]
pub fn gate_from_full_fg(
    st: &Stream,
    f_out: &DevBuf,
    g_out: &DevBuf,
    x: &DevBuf,
    b_weight: &DevBuf,
    a_log: &DevBuf,
    dt_bias: &DevBuf,
    lower: f32,
    rows: usize,
    hidden: usize,
    heads: usize,
    g_decay: &mut DevBuf,
    z_gate: &mut DevBuf,
    beta: &mut DevBuf,
) -> Result<(), CudaError> {
    if rows == 0 || hidden % 8 != 0 || heads == 0 || heads % 4 != 0 {
        return Err(CudaError(1));
    }
    let c = heads * 128;
    need(f_out, rows * c * 2, "f_out");
    need(g_out, rows * c * 2, "g_out");
    need(x, rows * hidden * 2, "x");
    need(b_weight, heads * hidden * 2, "b_weight");
    need(a_log, heads * 4, "a_log");
    need(dt_bias, c * 4, "dt_bias");
    need(g_decay, rows * c * 4, "g_decay");
    need(z_gate, rows * c * 4, "z_gate");
    need(beta, rows * heads * 4, "beta");
    let rc = unsafe {
        ffi::gb_kda_gate_from_full_fg(
            f_out.ptr(),
            g_out.ptr(),
            x.ptr(),
            b_weight.ptr(),
            a_log.ptr() as *const f32,
            dt_bias.ptr() as *const f32,
            lower,
            g_decay.ptr() as *mut f32,
            z_gate.ptr() as *mut f32,
            beta.ptr(),
            rows as i32,
            hidden as i32,
            heads as i32,
            st.raw(),
        )
    };
    CudaError::check(rc)
}

#[allow(clippy::too_many_arguments)]
fn prep_raw(
    st: &Stream,
    d: Dims,
    w: &Weights,
    qkv: *const core::ffi::c_void,
    in_f32: bool,
    pool: &Pool,
    sel: &Sel,
    keep: *const i32,
    emit_q: bool,
    skip_full: bool,
    raw_out: *mut core::ffi::c_void,
    ws: &Workspace,
) -> Result<(), CudaError> {
    let rc = unsafe {
        ffi::gb_kda_prep(
            qkv,
            in_f32 as i32,
            w.conv_w.ptr(),
            pool.conv.ptr(),
            sel.seq_start,
            sel.slot,
            keep,
            sel.n_seq as i32,
            sel.rows as i32,
            d.heads as i32,
            emit_q as i32,
            skip_full as i32,
            raw_out,
            ws.q.ptr() as *mut f32,
            ws.k.ptr() as *mut f32,
            ws.v.ptr(),
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// conv + silu + l2norm of the window's rows (`qkv` is the projection output, f32 or bf16 `[rows, conv_dim]`): writes the raw
/// bf16 rows to `rec.qkv` and q (scaled), k, v to the workspace. Reads, never writes, the conv state.
#[allow(clippy::too_many_arguments)]
pub fn prep(
    st: &Stream,
    d: Dims,
    w: &Weights,
    qkv: &DevBuf,
    qkv_dtype: Dtype,
    pool: &Pool,
    batch: &Batch<'_>,
    rec: &mut Record,
    ws: &mut Workspace,
) -> Result<(), CudaError> {
    d.check();
    assert!(matches!(qkv_dtype, Dtype::F32 | Dtype::BF16), "kda::prep: qkv must be f32 or bf16");
    assert!(batch.rows <= rec.rows && batch.rows <= ws.rows, "kda::prep: {} rows exceed the record/workspace", batch.rows);
    need(qkv, batch.rows * d.conv_dim() * qkv_dtype.size(), "qkv");
    need(w.conv_w, d.conv_dim() * CONV_TAPS * 2, "conv_w");
    need(&pool.conv, pool.n_slots * d.conv_slot_elems() * 2, "conv pool");
    let sel = Sel::of(batch);
    prep_raw(st, d, w, qkv.ptr(), qkv_dtype == Dtype::F32, pool, &sel, core::ptr::null(), true, false, rec.qkv.ptr(), ws)
}

/// The window: steps each sequence's rows from `S[cur]` (read-only) and writes the core output `ws.y()` (bf16
/// `[rows, heads * 128]`). With `write_final` the state after each sequence's last row goes to `S[1-cur]` (so that a full accept
/// is a flip); without it, [`commit`] must be told (`window_wrote_final = false`) and replays even a full accept.
#[allow(clippy::too_many_arguments)]
pub fn window(
    st: &Stream,
    d: Dims,
    p: &Params,
    pool: &mut Pool,
    batch: &Batch<'_>,
    rec: &Record,
    ws: &mut Workspace,
    write_final: bool,
) -> Result<(), CudaError> {
    d.check();
    assert!(p.r_step == 4 || p.r_step == 8);
    assert!(batch.rows <= rec.rows && batch.rows <= ws.rows);
    need(&pool.s, pool.n_slots * 2 * d.s_buf_elems() * 4, "S pool");
    let sel = Sel::of(batch);
    let rc = unsafe {
        ffi::gb_kda_window(
            ws.q.ptr() as *const f32,
            ws.k.ptr() as *const f32,
            ws.v.ptr(),
            rec.g.ptr() as *const f32,
            rec.beta.ptr() as *const f32,
            pool.s.ptr() as *mut f32,
            pool.parity.ptr() as *const i32,
            sel.seq_start,
            sel.slot,
            sel.n_seq as i32,
            d.heads as i32,
            (p.round == Round::TowardZero) as i32,
            write_final as i32,
            p.r_step as i32,
            ws.y.ptr(),
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// The chunked prefill recurrence: [`chunk_raw`] with the layer's round mode and 64 value rows per CTA, over one
/// sequence of `rows` rows in `slot` (a prefill chunk; the batch is a single sequence). Reads `S[slot][cur]`, writes
/// `S[slot][1 - cur]` and `ws.y()` (bf16 `[rows, heads, 128]`); the parity is not flipped here — [`commit`] with
/// `window_wrote_final = true` flips it and shifts the conv history, exactly as after [`window`]. The gated norm
/// [`norm`] reads `ws.y()` and `ws.z()` (the gate, written by `gate_from_full_fg` before this).
#[allow(clippy::too_many_arguments)]
pub fn window_chunked(
    st: &Stream,
    d: Dims,
    p: &Params,
    pool: &mut Pool,
    slot: usize,
    rows: usize,
    rec: &Record,
    ws: &mut Workspace,
) -> Result<(), CudaError> {
    d.check();
    assert!(p.r_step == 4 || p.r_step == 8);
    assert!(rows <= rec.rows && rows <= ws.rows);
    need(&pool.s, pool.n_slots * 2 * d.s_buf_elems() * 4, "S pool");
    assert!(slot < pool.n_slots && d.heads == pool.dims.heads);
    let rc = unsafe {
        ffi::gb_glm_p2_kda_chunk(
            ws.q.ptr(),
            ws.k.ptr(),
            ws.v.ptr(),
            rec.g.ptr(),
            rec.beta.ptr(),
            pool.s.ptr(),
            pool.parity.ptr(),
            slot as i32,
            rows as i32,
            d.heads as i32,
            64,
            (p.round == Round::TowardZero) as i32,
            ws.y.ptr(),
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// Gated RMSNorm of `ws.y()` with the sigmoid gate `ws.z()`: out fp16 `[rows, heads * 128]`, the input of `o_proj`.
pub fn norm(
    st: &Stream,
    d: Dims,
    w: &Weights,
    p: &Params,
    rows: usize,
    ws: &Workspace,
    out: &mut DevBuf,
) -> Result<(), CudaError> {
    d.check();
    assert!(rows <= ws.rows);
    need(out, rows * d.v_dim() * 2, "norm out");
    let rc = unsafe {
        ffi::gb_kda_norm(
            ws.y.ptr(),
            ws.z.ptr() as *const f32,
            w.o_norm.ptr(),
            p.eps,
            rows as i32,
            d.heads as i32,
            out.ptr(),
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// project + prep + window + norm for a decode step or a verify window: `x` the layer input, `qkv` the EXL3 `qkv_proj` output.
/// Leaves the record and the (uncommitted) next state behind; call [`commit`] with the accepted counts afterwards.
#[allow(clippy::too_many_arguments)]
pub fn forward_window(
    st: &Stream,
    d: Dims,
    w: &Weights,
    p: &Params,
    x: &DevBuf,
    x_dtype: Dtype,
    qkv: &DevBuf,
    qkv_dtype: Dtype,
    pool: &mut Pool,
    batch: &Batch<'_>,
    rec: &mut Record,
    ws: &mut Workspace,
    out: &mut DevBuf,
    write_final: bool,
) -> Result<(), CudaError> {
    project(st, d, w, p, x, x_dtype, batch.rows, rec, ws)?;
    prep(st, d, w, qkv, qkv_dtype, pool, batch, rec, ws)?;
    window(st, d, p, pool, batch, rec, ws, write_final)?;
    norm(st, d, w, p, batch.rows, ws, out)
}

/// Commit the first `keep[i]` rows of sequence i (device i32 `[n_seq]`, `0 <= keep[i] <= len_i`; `None` = every row of every
/// sequence). Per sequence: `keep == 0` leaves everything untouched; a full accept needs no replay when the window wrote its
/// final state (`window_wrote_final`), otherwise (and for a partial accept) the accepted prefix is replayed from the OLD state
/// with the window's own update routine; then the conv state shifts and the current-buffer index flips.
#[allow(clippy::too_many_arguments)]
pub fn commit(
    st: &Stream,
    d: Dims,
    w: &Weights,
    p: &Params,
    pool: &mut Pool,
    batch: &Batch<'_>,
    rec: &Record,
    ws: &mut Workspace,
    keep: Option<&DevBuf>,
    window_wrote_final: bool,
) -> Result<(), CudaError> {
    d.check();
    assert!(batch.rows <= rec.rows && batch.rows <= ws.rows);
    let sel = Sel::of(batch);
    let keep_ptr = keep.map_or(core::ptr::null(), |k| {
        need(k, batch.n_seq * 4, "keep");
        k.ptr() as *const i32
    });
    // replay is needed for a partial accept, and for every accept if the window did not write its final state. A null keep means
    // "every row" for both kernels. The raw rows and the conv state are still the PRE-commit ones here.
    if !keep_ptr.is_null() || !window_wrote_final {
        prep_raw(st, d, w, rec.qkv.ptr(), false, pool, &sel, keep_ptr, false, window_wrote_final, core::ptr::null_mut(), ws)?;
        let rc = unsafe {
            ffi::gb_kda_replay(
                ws.k.ptr() as *const f32,
                ws.v.ptr(),
                rec.g.ptr() as *const f32,
                rec.beta.ptr() as *const f32,
                pool.s.ptr() as *mut f32,
                pool.parity.ptr() as *const i32,
                sel.seq_start,
                sel.slot,
                keep_ptr,
                sel.n_seq as i32,
                d.heads as i32,
                window_wrote_final as i32,
                p.r_step as i32,
                st.raw(),
            )
        };
        CudaError::check(rc)?;
    }
    let rc = unsafe {
        ffi::gb_kda_commit_tail(
            rec.qkv.ptr(),
            pool.conv.ptr(),
            pool.parity.ptr() as *mut i32,
            sel.seq_start,
            sel.slot,
            keep_ptr,
            sel.n_seq as i32,
            d.heads as i32,
            1,
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// Exact prefill of one sequence in `slot`: `rows` prompt rows of `x` (`[rows, hidden]`) and `qkv` (`[rows, conv_dim]`), advancing
/// the pool in place and writing the norm output `out` (fp16 `[rows, heads * 128]`). The same kernels as decode on
/// [`PREFILL_PIECE_ROWS`] rows at a time with an all-rows commit after each piece: bit-identical to decoding the same rows
/// serially, so any cut of a prompt into calls gives the same bits. `rec` and `ws` need at least `min(rows, PREFILL_PIECE_ROWS)` rows.
#[allow(clippy::too_many_arguments)]
pub fn prefill(
    st: &Stream,
    d: Dims,
    w: &Weights,
    p: &Params,
    x: &DevBuf,
    x_dtype: Dtype,
    qkv: &DevBuf,
    qkv_dtype: Dtype,
    pool: &mut Pool,
    slot: usize,
    rows: usize,
    rec: &mut Record,
    ws: &mut Workspace,
    out: &mut DevBuf,
) -> Result<(), CudaError> {
    if rows == 0 {
        return Ok(());
    }
    let desc = prefill_descriptor(slot, rows)?;
    prefill_with(st, d, w, p, x, x_dtype, qkv, qkv_dtype, pool, slot, rows, rec, ws, out, &desc)
}

/// The descriptors of [`prefill_with`] for a prompt chunk of `rows` rows of the sequence in `slot`: `[0, len, slot, len]` for each piece of
/// [`PREFILL_PIECE_ROWS`] rows. One per chunk serves every layer (nothing in it depends on the layer), so a model uploads it once per forward
/// instead of allocating, copying and freeing one a layer (a free synchronises the device).
pub fn prefill_descriptor(slot: usize, rows: usize) -> Result<DevBuf, CudaError> {
    let n_pieces = rows.div_ceil(PREFILL_PIECE_ROWS);
    let mut desc = Vec::with_capacity(n_pieces * 4);
    for i in 0..n_pieces {
        let len = (rows - i * PREFILL_PIECE_ROWS).min(PREFILL_PIECE_ROWS) as i32;
        desc.extend_from_slice(&[0, len, slot as i32, len]);
    }
    DevBuf::from_slice(&desc)
}

/// [`prefill`] with the descriptor built by [`prefill_descriptor`] for the same `slot` and `rows`.
#[allow(clippy::too_many_arguments)]
pub fn prefill_with(
    st: &Stream,
    d: Dims,
    w: &Weights,
    p: &Params,
    x: &DevBuf,
    x_dtype: Dtype,
    qkv: &DevBuf,
    qkv_dtype: Dtype,
    pool: &mut Pool,
    slot: usize,
    rows: usize,
    rec: &mut Record,
    ws: &mut Workspace,
    out: &mut DevBuf,
    desc: &DevBuf,
) -> Result<(), CudaError> {
    d.check();
    check_weights(d, w);
    assert!(matches!(qkv_dtype, Dtype::F32 | Dtype::BF16));
    assert!(slot < pool.n_slots);
    need(x, rows * d.hidden * x_dtype.size(), "x");
    need(qkv, rows * d.conv_dim() * qkv_dtype.size(), "qkv");
    need(out, rows * d.v_dim() * 2, "norm out");
    let piece = PREFILL_PIECE_ROWS.min(rows);
    assert!(rec.rows >= piece && ws.rows >= piece, "kda::prefill: record/workspace hold fewer than {piece} rows");
    if rows == 0 {
        return Ok(());
    }
    let n_pieces = rows.div_ceil(PREFILL_PIECE_ROWS);
    assert!(desc.bytes() >= n_pieces * 16, "kda::prefill_with: the descriptor holds fewer than {n_pieces} pieces");
    for i in 0..n_pieces {
        let r0 = i * PREFILL_PIECE_ROWS;
        let len = (rows - r0).min(PREFILL_PIECE_ROWS);
        let base = i * 16;
        let sel = Sel {
            seq_start: desc.ptr_at(base) as *const i32,
            slot: desc.ptr_at(base + 8) as *const i32,
            n_seq: 1,
            rows: len,
            _life: core::marker::PhantomData,
        };
        let xp = x.ptr_at(r0 * d.hidden * x_dtype.size());
        project_raw(
            st,
            d,
            w,
            p,
            xp,
            x_dtype,
            len,
            rec.g.ptr() as *mut f32,
            rec.beta.ptr() as *mut f32,
            ws,
            ws.z.ptr() as *mut f32,
        )?;
        let qp = qkv.ptr_at(r0 * d.conv_dim() * qkv_dtype.size());
        prep_raw(st, d, w, qp, qkv_dtype == Dtype::F32, pool, &sel, core::ptr::null(), true, false, rec.qkv.ptr(), ws)?;
        let rc = unsafe {
            ffi::gb_kda_window(
                ws.q.ptr() as *const f32,
                ws.k.ptr() as *const f32,
                ws.v.ptr(),
                rec.g.ptr() as *const f32,
                rec.beta.ptr() as *const f32,
                pool.s.ptr() as *mut f32,
                pool.parity.ptr() as *const i32,
                sel.seq_start,
                sel.slot,
                1,
                d.heads as i32,
                (p.round == Round::TowardZero) as i32,
                1,
                p.r_step as i32,
                ws.y.ptr(),
                st.raw(),
            )
        };
        CudaError::check(rc)?;
        let op = out.ptr_at(r0 * d.v_dim() * 2);
        let rc = unsafe {
            ffi::gb_kda_norm(
                ws.y.ptr(),
                ws.z.ptr() as *const f32,
                w.o_norm.ptr(),
                p.eps,
                len as i32,
                d.heads as i32,
                op,
                st.raw(),
            )
        };
        CudaError::check(rc)?;
        let rc = unsafe {
            ffi::gb_kda_commit_tail(
                rec.qkv.ptr(),
                pool.conv.ptr(),
                pool.parity.ptr() as *mut i32,
                sel.seq_start,
                sel.slot,
                core::ptr::null(),
                1,
                d.heads as i32,
                1,
                st.raw(),
            )
        };
        CudaError::check(rc)?;
    }
    Ok(())
}

/// The version of P2's KDA arithmetic (`csrc/glm_p2_kda.cu`: 16-row chunks anchored per prefill segment, the chunk-relative decay
/// products, the fixed dot-product tree): part of P2's prefix namespace and of its TP Hello fact. Bump it with any change that moves a bit.
pub const P2_ARITH_ID: &str = "glm-kda-chunk16-v1";

/// Rows of one chunk of the P2 chunked recurrence (`csrc/glm_p2_kda.cu`, `docs/GLM_PREFILL_P2.md` section 8.1): a P2 prefill segment's
/// chunks are rows `anchor + 16 k ..`, so every call of a segment but its last starts and every piece but a call's last ends on a multiple.
pub const P2_CHUNK_ROWS: usize = 16;

/// The rows of one piece of [`prefill_chunked`] when its scratch holds them (a whole number of chunks): one projection, prep, state, norm
/// and commit launch a piece. The piece size moves no bit (pieces end on chunk boundaries and the state between them is stored exactly).
pub const P2_PIECE_ROWS: usize = 512;

static CHUNK_VS: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(64);
static CHUNKED_RUNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
thread_local! {
    static CHUNKED_RUNS_HERE: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
    static CHUNKED_HEADS_HERE: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
}

/// The chunked kernel's block shape: value rows a CTA, 64 (two CTAs a head, the default: two CTAs fit an SM) or 128. Never a bit (a
/// test checks both); for the
/// benches.
pub fn set_chunk_vs(vs: usize) {
    assert!(vs == 128 || vs == 64, "kda::set_chunk_vs: 128 or 64");
    CHUNK_VS.store(vs as i32, std::sync::atomic::Ordering::Relaxed);
}

/// [`prefill_chunked`] calls of this process so far (one a KDA layer of a P2 prompt chunk or segment): the every-row-through-P2 audit.
pub fn chunked_runs() -> u64 {
    CHUNKED_RUNS.load(std::sync::atomic::Ordering::Relaxed)
}

/// [`chunked_runs`] made by this thread (a loopback TP rank is a thread).
pub fn chunked_runs_here() -> u64 {
    CHUNKED_RUNS_HERE.with(|c| c.get())
}

/// The heads of this thread's [`prefill_chunked`] calls, summed (a KDA-split rank runs half the single node's: the audit of P5b).
pub fn chunked_heads_here() -> u64 {
    CHUNKED_HEADS_HERE.with(|c| c.get())
}

/// The descriptors of [`prefill_chunked`] for a call of `rows` rows of the sequence in `slot` in pieces of `piece` rows:
/// `[0, len, slot, len]` a piece (as [`prefill_descriptor`] for 128-row pieces).
pub fn prefill_descriptor_pieces(slot: usize, rows: usize, piece: usize) -> Result<DevBuf, CudaError> {
    assert!(piece > 0 && piece % P2_CHUNK_ROWS == 0, "kda: a P2 piece is a whole number of {P2_CHUNK_ROWS}-row chunks");
    let n_pieces = rows.div_ceil(piece);
    let mut desc = Vec::with_capacity(n_pieces * 4);
    for i in 0..n_pieces {
        let len = (rows - i * piece).min(piece) as i32;
        desc.extend_from_slice(&[0, len, slot as i32, len]);
    }
    DevBuf::from_slice(&desc)
}

/// The rows of a [`prefill_chunked`] piece over this scratch: [`P2_PIECE_ROWS`], or the largest whole number of chunks the record and the
/// workspace hold (a model built under P1 and switched to P2 runs smaller pieces, with the same bits).
pub fn p2_piece_rows(rec: &Record, ws: &Workspace) -> usize {
    let cap = rec.rows.min(ws.rows);
    P2_PIECE_ROWS.min(cap - cap % P2_CHUNK_ROWS)
}

/// The P2 prefill of `rows` prompt rows of one sequence in `slot` (contract P2, `kda=glm-kda-chunk16-v1`): P1's projections and prep on
/// each piece, then the CHUNKED recurrence (`csrc/glm_p2_kda.cu`: chunks of [`P2_CHUNK_ROWS`] rows from the call's first row, which the
/// caller puts on a chunk boundary of its prefill segment), P1's norm and commit tail (the conv history shift and the parity flip). `x`
/// `[rows, hidden]`, `qkv` `[rows, conv_dim]`, `out` fp16 `[rows, heads * 128]` as for [`prefill_with`]; `desc` from
/// [`prefill_descriptor_pieces`] with `piece` (a whole number of chunks the record and workspace hold: [`p2_piece_rows`]).
///
/// bf16 `qkv` rows are the record's raw rows already (prep's rounding of an f32 row is the bf16 the GEMM would have stored): prep writes no
/// record and the conv shift reads `qkv` itself. With bf16 rows and a pipelined workspace ([`Workspace::add_pipe`]) the projections, prep
/// and conv shift of piece `i + 1` run on the workspace's side stream while `st` runs piece `i`'s recurrence, norm and parity flip; every
/// launch has the operands it has in the serial order, so the bits are the serial order's.
#[allow(clippy::too_many_arguments)]
pub fn prefill_chunked(
    st: &Stream,
    d: Dims,
    w: &Weights,
    p: &Params,
    x: &DevBuf,
    x_dtype: Dtype,
    qkv: &DevBuf,
    qkv_dtype: Dtype,
    pool: &mut Pool,
    slot: usize,
    rows: usize,
    rec: &mut Record,
    ws: &mut Workspace,
    out: &mut DevBuf,
    desc: &DevBuf,
    piece: usize,
) -> Result<(), CudaError> {
    d.check();
    check_weights(d, w);
    assert!(matches!(qkv_dtype, Dtype::F32 | Dtype::BF16));
    assert!(slot < pool.n_slots);
    assert!(
        piece > 0 && piece % P2_CHUNK_ROWS == 0 && piece <= rec.rows && piece <= ws.rows,
        "kda::prefill_chunked: piece {piece}"
    );
    if rows == 0 {
        return Ok(());
    }
    need(x, rows * d.hidden * x_dtype.size(), "x");
    need(qkv, rows * d.conv_dim() * qkv_dtype.size(), "qkv");
    need(out, rows * d.v_dim() * 2, "norm out");
    need(&pool.s, pool.n_slots * 2 * d.s_buf_elems() * 4, "S pool");
    let n_pieces = rows.div_ceil(piece);
    assert!(desc.bytes() >= n_pieces * 16, "kda::prefill_chunked: the descriptor holds fewer than {n_pieces} pieces");
    CHUNKED_RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    CHUNKED_RUNS_HERE.with(|c| c.set(c.get() + 1));
    CHUNKED_HEADS_HERE.with(|c| c.set(c.get() + d.heads as u64));
    let vs = CHUNK_VS.load(std::sync::atomic::Ordering::Relaxed);
    let bf16_rows = qkv_dtype == Dtype::BF16;
    let rz = (p.round == Round::TowardZero) as i32;
    let set0 = Set {
        q: ws.q.ptr() as *mut f32,
        k: ws.k.ptr() as *mut f32,
        v: ws.v.ptr(),
        z: ws.z.ptr() as *mut f32,
        g: rec.g.ptr() as *mut f32,
        beta: rec.beta.ptr() as *mut f32,
    };
    let ws: &Workspace = ws;
    // piece i's projections and prep into `set` on `s` (with `shift`, the conv shift of bf16 rows too); its recurrence and norm on `s`
    let front = |s: &Stream, set: &Set, i: usize, shift: bool| -> Result<(), CudaError> {
        let r0 = i * piece;
        let len = (rows - r0).min(piece);
        let (seq_start, sl) = (desc.ptr_at(i * 16) as *const i32, desc.ptr_at(i * 16 + 8) as *const i32);
        project_raw(s, d, w, p, x.ptr_at(r0 * d.hidden * x_dtype.size()), x_dtype, len, set.g, set.beta, ws, set.z)?;
        let qp = qkv.ptr_at(r0 * d.conv_dim() * qkv_dtype.size());
        let raw_out = if bf16_rows { core::ptr::null_mut() } else { rec.qkv.ptr() };
        // SAFETY: the set holds `len <= piece` rows of this layer's dims, the record's raw rows too; the conv pool is the layer's
        let rc = unsafe {
            ffi::gb_kda_prep(
                qp,
                !bf16_rows as i32,
                w.conv_w.ptr(),
                pool.conv.ptr(),
                seq_start,
                sl,
                core::ptr::null(),
                1,
                len as i32,
                d.heads as i32,
                1,
                0,
                raw_out,
                set.q,
                set.k,
                set.v,
                s.raw(),
            )
        };
        CudaError::check(rc)?;
        if shift {
            CudaError::check(unsafe {
                ffi::gb_kda_commit_tail(qp, pool.conv.ptr(), pool.parity.ptr() as *mut i32, seq_start, sl, core::ptr::null(), 1, d.heads as i32, 0, s.raw())
            })?;
        }
        Ok(())
    };
    let back = |s: &Stream, set: &Set, i: usize| -> Result<(), CudaError> {
        let r0 = i * piece;
        let len = (rows - r0).min(piece);
        // SAFETY: as for `front`; S and parity are the pool's
        CudaError::check(unsafe {
            ffi::gb_glm_p2_kda_chunk(
                set.q as *const _,
                set.k as *const _,
                set.v,
                set.g as *const _,
                set.beta as *const _,
                pool.s.ptr(),
                pool.parity.ptr(),
                slot as i32,
                len as i32,
                d.heads as i32,
                vs,
                rz,
                ws.y.ptr(),
                s.raw(),
            )
        })?;
        CudaError::check(unsafe {
            ffi::gb_kda_norm(ws.y.ptr(), set.z, w.o_norm.ptr(), p.eps, len as i32, d.heads as i32, out.ptr_at(r0 * d.v_dim() * 2), s.raw())
        })
    };
    let tail_flip = |s: &Stream, i: usize, shift: bool| -> Result<(), CudaError> {
        let (seq_start, sl) = (desc.ptr_at(i * 16) as *const i32, desc.ptr_at(i * 16 + 8) as *const i32);
        if !shift {
            return CudaError::check(unsafe { ffi::gb_kda_flip(pool.parity.ptr() as *mut i32, sl, 1, s.raw()) });
        }
        let raw = if bf16_rows { qkv.ptr_at(i * piece * d.conv_dim() * 2) } else { rec.qkv.ptr() };
        CudaError::check(unsafe {
            ffi::gb_kda_commit_tail(raw, pool.conv.ptr(), pool.parity.ptr() as *mut i32, seq_start, sl, core::ptr::null(), 1, d.heads as i32, 1, s.raw())
        })
    };
    match ws.pipe.as_ref().filter(|_| bf16_rows && n_pieces > 1) {
        None => {
            for i in 0..n_pieces {
                front(st, &set0, i, false)?;
                back(st, &set0, i)?;
                tail_flip(st, i, true)?;
            }
        }
        Some(pp) => {
            let set1 = Set {
                q: pp.q.ptr() as *mut f32,
                k: pp.k.ptr() as *mut f32,
                v: pp.v.ptr(),
                z: pp.z.ptr() as *mut f32,
                g: pp.g.ptr() as *mut f32,
                beta: pp.beta.ptr() as *mut f32,
            };
            let sets = [&set0, &set1];
            // the side stream starts after everything `st` has queued (this layer's x and qkv, the last users of both sets)
            pp.start.record(st)?;
            crate::graph::wait(&pp.side, &pp.start)?;
            for i in 0..n_pieces {
                let b = i & 1;
                if i >= 2 {
                    crate::graph::wait(&pp.side, &pp.free[b])?;
                }
                front(&pp.side, sets[b], i, true)?;
                pp.ready[b].record(&pp.side)?;
                crate::graph::wait(st, &pp.ready[b])?;
                back(st, sets[b], i)?;
                tail_flip(st, i, false)?;
                pp.free[b].record(st)?;
            }
        }
    }
    Ok(())
}

/// One piece's buffers of [`prefill_chunked`]: q, k, z, g, beta f32 and v bf16, `piece` rows each.
struct Set {
    q: *mut f32,
    k: *mut f32,
    v: *mut core::ffi::c_void,
    z: *mut f32,
    g: *mut f32,
    beta: *mut f32,
}

/// Test hook: the P2 chunked recurrence alone on raw inputs (`q`, `k` f32 `[rows][heads][128]`, `v` bf16, `g` f32 `[rows][heads * 128]`,
/// `beta` f32 `[rows][heads]`): reads `S[slot][cur]`, writes `S[slot][1 - cur]` and `y` bf16 `[rows][heads][128]`; the parity is not
/// flipped. `vs` = value rows a CTA (128 or 64).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn chunk_raw(
    st: &Stream,
    heads: usize,
    q: &DevBuf,
    k: &DevBuf,
    v: &DevBuf,
    g: &DevBuf,
    beta: &DevBuf,
    pool: &mut Pool,
    slot: usize,
    rows: usize,
    vs: usize,
    y: &mut DevBuf,
) -> Result<(), CudaError> {
    let e = rows * heads * HEAD_DIM;
    need(q, e * 4, "q");
    need(k, e * 4, "k");
    need(v, e * 2, "v");
    need(g, e * 4, "g");
    need(beta, rows * heads * 4, "beta");
    need(y, e * 2, "y");
    assert!(slot < pool.n_slots && heads == pool.dims.heads);
    let rc = unsafe {
        ffi::gb_glm_p2_kda_chunk(
            q.ptr(),
            k.ptr(),
            v.ptr(),
            g.ptr(),
            beta.ptr(),
            pool.s.ptr(),
            pool.parity.ptr(),
            slot as i32,
            rows as i32,
            heads as i32,
            vs as i32,
            0,
            y.ptr(),
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// Test hook: P1's state kernel (the row-by-row `update()`) on the same raw inputs as [`chunk_raw`], one sequence of `rows` rows in `slot`
/// (`seq_start` = device `[0, rows]`, `slots` = device `[slot]`): reads `S[cur]`, writes the state after the last row to `S[1 - cur]` and `y`.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn p1_state_raw(
    st: &Stream,
    heads: usize,
    p: &Params,
    q: &DevBuf,
    k: &DevBuf,
    v: &DevBuf,
    g: &DevBuf,
    beta: &DevBuf,
    pool: &mut Pool,
    seq_start: &DevBuf,
    slots: &DevBuf,
    y: &mut DevBuf,
) -> Result<(), CudaError> {
    let rc = unsafe {
        ffi::gb_kda_window(
            q.ptr() as *const f32,
            k.ptr() as *const f32,
            v.ptr(),
            g.ptr() as *const f32,
            beta.ptr() as *const f32,
            pool.s.ptr() as *mut f32,
            pool.parity.ptr() as *const i32,
            seq_start.ptr() as *const i32,
            slots.ptr() as *const i32,
            1,
            heads as i32,
            (p.round == Round::TowardZero) as i32,
            1,
            p.r_step as i32,
            y.ptr(),
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// Test hook: for `r` (4 or 8) values per lane (`input` f32 `[32][r]`), the plain xor-butterfly sum of the row each lane holds and the
/// halves-exchanging reduction the state kernel uses, `out` f32 `[32][2]`; the two columns must be bit-identical.
#[doc(hidden)]
pub fn selftest_reduce(st: &Stream, input: &DevBuf, out: &mut DevBuf, r: usize) -> Result<(), CudaError> {
    need(input, 32 * r * 4, "reduce input");
    need(out, 32 * 2 * 4, "reduce output");
    let rc = unsafe { ffi::gb_kda_reduce_check(input.ptr() as *const f32, out.ptr() as *mut f32, r as i32, st.raw()) };
    CudaError::check(rc)
}
