// gb-nvcc: --fmad=false
// Gated delta net (linear attention) kernels for Qwen3.8-27B: 16 key heads x 128, 48 value heads x 128, conv kernel 4
// over 10240 channels. Spec: docs/QWEN38_SPEC.md section 5.
//
// The value head count is the one shape parameter (GB_GDN_HV; the conv width follows: 2 x 16 x 128 + HV x 128). This file
// builds HV = 48 with the launcher names below; gdn_hv32.cu builds the same source with HV = 32 (the Qwen3.5/3.6-35B-A3B
// backbone of MiniCPM-V-4.7: 16 key x 32 value heads, 8192 channels, value head h reads key head h / 2) and every launcher
// suffixed `_hv32`. One source, so a row's arithmetic is the same code at both shapes: the 48-head kernels are unchanged
// by the parameter (tests/gdn.rs and its golden pin them), and the 32-head build has its own tests and golden.
//
// Ported from TensorFold (MIT, commit 9cd52ab): the replay `step()` and the window/replay kernels from
// src/tensorfold/cuda/kernels/gdn.cu, the prefill chain kernel from gdn_prefill.cu, and the conv/silu/l2norm/gating
// prep rewritten in CUDA from the Triton `_gdn_pre` in src/tensorfold/families/qwen3_5/cuda/glue.py.
// See THIRD_PARTY_NOTICES.md.
//
// The file is compiled with --fmad=false: every multiply and add of the decode path is rounded alone, so one
// `step()` gives the same bits wherever it is inlined (serial 1-row window, verify window, commit replay). The prefill
// chain kernel uses explicit __fmaf_rn and its own summation order; its bits legitimately differ from the decode
// path's but are deterministic and independent of how a prompt is cut into chunks.
//
// State pools (per layer, indexed by slot; HV = 48 here):
//   S    f32  [slots][HV][128 value row j][128 key i]   (a warp owns value rows, each lane 4 consecutive keys)
//   conv bf16 [slots][3][CONV]                          (last 3 pre-conv qkv rows, oldest first)
// Sequences: seq_start i32[n_seq+1] (row offsets), slot i32[n_seq]. Slots within one launch that writes the pools
// (commit, prefill) must be distinct.
#include "common.cuh"
#include "pdl.cuh"

#ifndef GB_GDN_HV
#define GB_GDN_HV 48
#define GB_GDN_EXPORT(name) name
#endif

namespace {

// Flash-Next decode launches (gbp_pdl::kFlash): a kernel launched with the programmatic-dependent-launch attribute may start before its
// predecessor finishes, so it calls pdl_wait() before it reads anything an earlier kernel wrote (a no-op in a launch without the
// attribute). Only timing changes, never a bit.
using gbp_pdl::pdl_wait;
using gbp_pdl::pdl_trigger;
inline bool flash_pdl() { return gbp_pdl::on(gbp_pdl::kFlash); }

constexpr int HK = 16;                       // key heads
constexpr int HV = GB_GDN_HV;                // value heads
constexpr int DK = 128;                      // key dim == value dim
constexpr int CONV = 2 * HK * DK + HV * DK;  // conv channels: [q 16x128 | k 16x128 | v HV x128]: 10240 at HV = 48, 8192 at 32
static_assert(HV == 64 || HV == 48 || HV == 32, "the kernels are built and tested for 64, 48 and 32 value heads");
static_assert(HV % HK == 0 && CONV % 256 == 0, "value heads are a multiple of the key heads; the conv commit takes 256 channels a block");
constexpr int KEEP = 3;                      // conv history rows
constexpr int NUNITS = 2 * HK + HV;          // 128-channel units in a row: 16 q, 16 k, HV v
constexpr size_t S_HEAD = (size_t)DK * DK;
constexpr size_t S_SLOT = (size_t)HV * S_HEAD;
constexpr float Q_SCALE = 0.08838834764831845f;  // 1/sqrt(128)
constexpr float L2_EPS = 1e-6f;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int m = 16; m; m >>= 1) x += __shfl_xor_sync(0xffffffffu, x, m);
    return x;
}

__device__ __forceinline__ void load4(const __nv_bfloat16* p, float (&x)[4]) {
    const uint2 w = *reinterpret_cast<const uint2*>(p);
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&w.x);
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&w.y);
    x[0] = __low2float(a);
    x[1] = __high2float(a);
    x[2] = __low2float(b);
    x[3] = __high2float(b);
}

__device__ __forceinline__ float4 load4f(const __nv_bfloat16* p) {
    float x[4];
    load4(p, x);
    return make_float4(x[0], x[1], x[2], x[3]);
}

__device__ __forceinline__ void store4(__nv_bfloat16* p, const float (&x)[4]) {
    const __nv_bfloat162 a = __floats2bfloat162_rn(x[0], x[1]);
    const __nv_bfloat162 b = __floats2bfloat162_rn(x[2], x[3]);
    uint2 w;
    w.x = *reinterpret_cast<const unsigned*>(&a);
    w.y = *reinterpret_cast<const unsigned*>(&b);
    *reinterpret_cast<uint2*>(p) = w;
}

__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// The sequence a global row belongs to: the largest s with seq_start[s] <= row (empty sequences are skipped).
__device__ __forceinline__ int find_seq(const int* __restrict__ seq_start, int n_seq, int row) {
    int lo = 0, hi = n_seq - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (seq_start[mid] <= row) lo = mid; else hi = mid - 1;
    }
    return lo;
}

// ---------------------------------------------------------------------------------------------------------------
// pre: depthwise causal conv over [3 stored rows; window rows] + silu (one rounding to bf16), l2norm of q and k,
// decay g and beta. One warp per (row, 128-channel unit), 4 channels per lane. Reads only its own row's window.
// ---------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(128) pre_kernel(
        const __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
        const __nv_bfloat16* __restrict__ conv_w, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        const __nv_bfloat16* __restrict__ conv_pool, const int* __restrict__ seq_start, const int* __restrict__ slot,
        int n_seq, __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v,
        float* __restrict__ g, float* __restrict__ beta, const int* __restrict__ parent) {
    pdl_wait();
    pdl_trigger();  // the window kernel may launch and load the recurrent state; it waits for this grid before it reads q, k, v
    const int row = blockIdx.x;
    const int unit = blockIdx.y * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int seq = find_seq(seq_start, n_seq, row);
    const int start = seq_start[seq], lt = row - start;
    const size_t pool = (size_t)slot[seq] * KEEP;
    const int ch0 = unit * DK + lane * 4;

    // A tree window (`parent` != nullptr: the parent's global row, -1 for the root): the conv history is the row's own root path,
    // x[t-back] = the ancestor `back` levels up or, past the root, committed history row 3 + depth - back. A chain
    // (parent[i] = i - 1) reads exactly the rows and pool entries the arithmetic without `parent` gives it, so the two agree bit for bit.
    int anc[4] = {row, -1, -1, -1}, first_missing = 4;
    if (parent) {
#pragma unroll
        for (int b = 1; b < 4; ++b) anc[b] = anc[b - 1] >= 0 ? parent[anc[b - 1]] : -1;
        // a row under 3 levels deep: its first missing ancestor is at first_missing = depth + 1
        first_missing = anc[1] < 0 ? 1 : anc[2] < 0 ? 2 : anc[3] < 0 ? 3 : 4;
    }

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float x[4];
        if (parent) {
            const int back = 3 - j;
            if (anc[back] >= 0) load4(qkv + (size_t)anc[back] * CONV + ch0, x);
            else load4(conv_pool + (pool + 3 + (first_missing - 1) - back) * CONV + ch0, x);
        } else {
            const int idx = lt - 3 + j;  // window row x[t-3+j]
            if (idx >= 0) load4(qkv + (size_t)(start + idx) * CONV + ch0, x);
            else load4(conv_pool + (pool + 3 + idx) * CONV + ch0, x);
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float w = __bfloat162float(conv_w[(size_t)(ch0 + c) * 4 + j]);
            acc[c] = acc[c] + x[c] * w;
        }
    }
    float c[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) c[i] = round_bf16(acc[i] / (1.0f + expf(-acc[i])));

    if (unit < 2 * HK) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) ss = ss + c[i] * c[i];
        ss = warp_sum(ss);
        const float inv = 1.0f / sqrtf(ss + L2_EPS);
        const bool is_q = unit < HK;
        float out[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float t = c[i] * inv;
            out[i] = is_q ? t * Q_SCALE : t;
        }
        __nv_bfloat16* dst = (is_q ? q : k) + ((size_t)row * HK + (is_q ? unit : unit - HK)) * DK + lane * 4;
        store4(dst, out);
    } else {
        const int hv = unit - 2 * HK;
        store4(v + ((size_t)row * HV + hv) * DK + lane * 4, c);
        if (lane == 0) {
            const float av = __bfloat162float(a[(size_t)row * HV + hv]) + dt_bias[hv];
            const float sp = av > 20.0f ? av : log1pf(expf(av));
            g[(size_t)row * HV + hv] = expf(-expf(a_log[hv]) * sp);
            const float bv = __bfloat162float(b[(size_t)row * HV + hv]);
            beta[(size_t)row * HV + hv] = 1.0f / (1.0f + expf(-bv));
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Decode-path step (window and commit replay): the ONE routine that advances state. A warp holds R value rows, lane
// l the keys 4l..4l+3 of each. Every op is rounded alone (--fmad=false); sums are lane-local then an xor butterfly.
// ---------------------------------------------------------------------------------------------------------------
template <int R>
__device__ __forceinline__ void step(float (&s)[R][4], const float (&k)[4], const float (&v)[R], float g, float beta) {
    float mem[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        mem[r] = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[r][i] = s[r][i] * g;
            mem[r] = mem[r] + s[r][i] * k[i];
        }
    }
#pragma unroll
    for (int m = 16; m; m >>= 1) {
#pragma unroll
        for (int r = 0; r < R; ++r) mem[r] += __shfl_xor_sync(0xffffffffu, mem[r], m);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float delta = (v[r] - mem[r]) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[r][i] = s[r][i] + k[i] * delta;
    }
}

// o[r] = sum_i s[r][i] * q[i] on the (already updated) state; every lane ends with the full sum.
template <int R>
__device__ __forceinline__ void readout(const float (&s)[R][4], const float (&q)[4], float (&out)[R]) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
        out[r] = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) out[r] = out[r] + s[r][i] * q[i];
    }
#pragma unroll
    for (int m = 16; m; m >>= 1) {
#pragma unroll
        for (int r = 0; r < R; ++r) out[r] += __shfl_xor_sync(0xffffffffu, out[r], m);
    }
}

// One row's inputs for a warp: its key head's k (and q) at this lane's 4 keys, the R value elements, the gates.
template <int R>
struct RowIn {
    float q[4], k[4], v[R], g, beta;

    __device__ __forceinline__ void load(const __nv_bfloat16* qp, const __nv_bfloat16* kp, const __nv_bfloat16* vp,
                                         const float* gp, const float* bp, int row, int head, int value0, int lane) {
        const size_t key = ((size_t)row * HK + head / (HV / HK)) * DK + lane * 4;
        if (qp) load4(qp + key, q);
        load4(kp + key, k);
        const size_t val = ((size_t)row * HV + head) * DK + value0;
#pragma unroll
        for (int r = 0; r < R; ++r) v[r] = __bfloat162float(vp[val + r]);
        g = gp[(size_t)row * HV + head];
        beta = bp[(size_t)row * HV + head];
    }
};

template <int R>
__device__ __forceinline__ void load_state(float (&s)[R][4], const float* base, int value0, int lane) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float4 t = *reinterpret_cast<const float4*>(base + (size_t)(value0 + r) * DK + lane * 4);
        s[r][0] = t.x; s[r][1] = t.y; s[r][2] = t.z; s[r][3] = t.w;
    }
}

template <int R>
__device__ __forceinline__ void store_state(float* base, const float (&s)[R][4], int value0, int lane) {
#pragma unroll
    for (int r = 0; r < R; ++r)
        *reinterpret_cast<float4*>(base + (size_t)(value0 + r) * DK + lane * 4) = make_float4(s[r][0], s[r][1], s[r][2], s[r][3]);
}

constexpr int R_STEP = 8;      // value rows per warp
constexpr int W_STEP = 4;      // warps per block -> 32 value rows per block

// The previous window's commit, still owed for the sequences of this launch: for sequence i, the first keep[i] rows of its
// previous window (rows row0[i].. of the k/v/g/beta record) have not yet been replayed into S. keep == nullptr: nothing is owed.
struct Owed {
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const int* row0;
    const int* keep;
    // A tree window commits an accepted PATH, not a prefix: when `rows` is set, step j of sequence i replays record row
    // rows[i * rows_stride + j] (a global row of the previous record) instead of row0[i] + j. nullptr: a prefix, as before.
    const int* rows;
    int rows_stride;
};

// The record row step j of sequence `seq`'s owed replay reads: the j-th row of its accepted path, or row0 + j for a prefix.
__device__ __forceinline__ int owed_row(const Owed& ow, int seq, int j) {
    return ow.rows ? ow.rows[(size_t)seq * ow.rows_stride + j] : ow.row0[seq] + j;
}

// window: the committed state is read from a register copy and never written, except for a commit the previous window still
// owes (`ow`): then the block first replays those rows with the same step() (the very loop commit_kernel runs), stores the
// committed state back, and goes on with this window from it. That is one read and one write of S per round instead of the
// window's read plus the commit's read and write. Each thread stores exactly the floats it loaded and no two blocks share a
// (slot, head, value row), so the in-place store cannot race.
__global__ void __launch_bounds__(32 * W_STEP) window_kernel(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const float* __restrict__ g, const float* __restrict__ beta, float* __restrict__ S,
        const int* __restrict__ seq_start, const int* __restrict__ slot, __nv_bfloat16* __restrict__ o, Owed ow) {
    constexpr int R = R_STEP;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head = blockIdx.y, seq = blockIdx.z;
    const int value0 = (blockIdx.x * W_STEP + warp) * R;
    const int begin = seq_start[seq], end = seq_start[seq + 1];
    if (end <= begin) return;
    float* base = S + (size_t)slot[seq] * S_SLOT + (size_t)head * S_HEAD;
    float s[R][4];
    load_state<R>(s, base, value0, lane);
    if (ow.keep) {
        const int n = max(ow.keep[seq], 0);
        if (n > 0) {
            RowIn<R> nx;
            nx.load(nullptr, ow.k, ow.v, ow.g, ow.beta, owed_row(ow, seq, 0), head, value0, lane);
            for (int j = 0; j < n; ++j) {
                const RowIn<R> in = nx;
                if (j + 1 < n) nx.load(nullptr, ow.k, ow.v, ow.g, ow.beta, owed_row(ow, seq, j + 1), head, value0, lane);
                step<R>(s, in.k, in.v, in.g, in.beta);
            }
            store_state<R>(base, s, value0, lane);
        }
    }
    // the state and the owed record are no earlier kernel's output of this window; q, k, v, g and beta are the pre kernel's
    pdl_wait();
    pdl_trigger();
    RowIn<R> next;
    next.load(q, k, v, g, beta, begin, head, value0, lane);
    for (int row = begin; row < end; ++row) {
        const RowIn<R> in = next;
        if (row + 1 < end) next.load(q, k, v, g, beta, row + 1, head, value0, lane);
        step<R>(s, in.k, in.v, in.g, in.beta);
        float out[R];
        readout<R>(s, in.q, out);
        if (lane < R) {
            float x = out[0];
#pragma unroll
            for (int r = 1; r < R; ++r) x = lane == r ? out[r] : x;
            o[((size_t)row * HV + head) * DK + value0 + lane] = __float2bfloat16_rn(x);
        }
    }
}

// commit (state): replay the first keep rows of each sequence with the same step(), writing S back in place. Each
// thread stores exactly the floats it loaded, so blocks never overlap.
__global__ void __launch_bounds__(32 * W_STEP) commit_kernel(
        const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
        const float* __restrict__ beta, float* S, const int* __restrict__ seq_start, const int* __restrict__ slot,
        const int* __restrict__ keep, const int* __restrict__ rows, int rows_stride) {
    constexpr int R = R_STEP;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head = blockIdx.y, seq = blockIdx.z;
    const int value0 = (blockIdx.x * W_STEP + warp) * R;
    const int begin = seq_start[seq], len = seq_start[seq + 1] - begin;
    const int n = min(max(keep[seq], 0), len);
    if (n <= 0) return;
    // `rows` (nullable): the accepted path's window rows (global), step j replays rows[seq * rows_stride + j]; else the prefix begin + j
    auto row_of = [&](int j) { return rows ? rows[(size_t)seq * rows_stride + j] : begin + j; };
    float* base = S + (size_t)slot[seq] * S_SLOT + (size_t)head * S_HEAD;
    float s[R][4];
    load_state<R>(s, base, value0, lane);
    RowIn<R> next;
    next.load(nullptr, k, v, g, beta, row_of(0), head, value0, lane);
    for (int j = 0; j < n; ++j) {
        const RowIn<R> in = next;
        if (j + 1 < n) next.load(nullptr, k, v, g, beta, row_of(j + 1), head, value0, lane);
        step<R>(s, in.k, in.v, in.g, in.beta);
    }
    store_state<R>(base, s, value0, lane);
}

// ---------------------------------------------------------------------------------------------------------------
// tree window: the window kernel over rows that form a tree per sequence. The recurrence visits the rows in the order of a
// host-built plan (docs/TREE_DRAFTING.md; cuda/src/tree.rs): entry e = (node, source, dest) says where the row's input state
// comes from -- the committed state (the root), the state the previous entry left in registers (the row's parent ran just before
// it), or a slot a sibling's parent saved -- and, with dest >= 0, where to save the state the row leaves for a later sibling.
// The row's computation is the window kernel's, op for op (the same step<R> and readout<R>, from a register state), so a row's
// output and the state it leaves are the bits serial decoding gives the token at the end of its root path, and a chain's plan
// (one entry per row, source -2) is the window kernel's loop. R, WARPS and SLOTS only trade shared memory for occupancy: every
// value row's arithmetic is independent of them. The slots live in shared memory, lane private (a lane writes and reads only its
// own float4s), so they need no barrier; the plan is copied to shared memory once.
// ---------------------------------------------------------------------------------------------------------------
template <int SLOTS, int R, int WARPS>
__global__ void __launch_bounds__(32 * WARPS) tree_window_kernel(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const float* __restrict__ g, const float* __restrict__ beta, float* __restrict__ S,
        const int* __restrict__ seq_start, const int* __restrict__ slot, __nv_bfloat16* __restrict__ o, Owed ow,
        const int* __restrict__ plan, int plan_stride) {
    constexpr int SL = SLOTS > 0 ? SLOTS : 1;
    extern __shared__ float4 smem[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float4* slots = smem + (size_t)warp * SL * R * 32;
    int* order = reinterpret_cast<int*>(smem + (size_t)WARPS * SL * R * 32);
    const int head = blockIdx.y, seq = blockIdx.z;
    const int value0 = (blockIdx.x * WARPS + warp) * R;
    const int begin = seq_start[seq], end = seq_start[seq + 1];
    const int count = end - begin;
    if (count <= 0) return;  // uniform across the block
    for (int i = threadIdx.x; i < 3 * count; i += 32 * WARPS) order[i] = plan[(size_t)seq * plan_stride + i];
    __syncthreads();
    float* base = S + (size_t)slot[seq] * S_SLOT + (size_t)head * S_HEAD;
    float s[R][4];
    load_state<R>(s, base, value0, lane);
    if (ow.keep) {  // the previous window's accepted rows (a path), replayed and stored back, exactly as in window_kernel
        const int n = max(ow.keep[seq], 0);
        if (n > 0) {
            RowIn<R> nx;
            nx.load(nullptr, ow.k, ow.v, ow.g, ow.beta, owed_row(ow, seq, 0), head, value0, lane);
            for (int j = 0; j < n; ++j) {
                const RowIn<R> in = nx;
                if (j + 1 < n) nx.load(nullptr, ow.k, ow.v, ow.g, ow.beta, owed_row(ow, seq, j + 1), head, value0, lane);
                step<R>(s, in.k, in.v, in.g, in.beta);
            }
            store_state<R>(base, s, value0, lane);
        }
    }
    RowIn<R> next;
    next.load(q, k, v, g, beta, begin + order[0], head, value0, lane);
    for (int e = 0; e < count; ++e) {
        const RowIn<R> in = next;
        const int node = begin + order[3 * e], source = order[3 * e + 1], dest = order[3 * e + 2];
        if (e + 1 < count) next.load(q, k, v, g, beta, begin + order[3 * e + 3], head, value0, lane);
        if (source >= 0) {  // a saved state (SLOTS == 0 plans never read one); -1 and -2 continue from the registers
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float4 t = slots[(size_t)(source * R + r) * 32 + lane];
                s[r][0] = t.x; s[r][1] = t.y; s[r][2] = t.z; s[r][3] = t.w;
            }
        }
        step<R>(s, in.k, in.v, in.g, in.beta);
        float out[R];
        readout<R>(s, in.q, out);
        if (lane < R) {
            float x = out[0];
#pragma unroll
            for (int r = 1; r < R; ++r) x = lane == r ? out[r] : x;
            o[((size_t)node * HV + head) * DK + value0 + lane] = __float2bfloat16_rn(x);
        }
        if (dest >= 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) slots[(size_t)(dest * R + r) * 32 + lane] = make_float4(s[r][0], s[r][1], s[r][2], s[r][3]);
        }
    }
}

template <int SLOTS, int R, int WARPS>
int launch_tree_window(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const float* g, const float* beta,
                       float* S, const int* seq_start, const int* slot, int n_seq, __nv_bfloat16* o, const Owed& ow,
                       const int* plan, int plan_stride, cudaStream_t st) {
    constexpr int SL = SLOTS > 0 ? SLOTS : 1;
    constexpr size_t smem = (size_t)WARPS * SL * R * 32 * sizeof(float4) + sizeof(int) * 3 * 128;
    static_assert(smem <= 48 * 1024, "the tree window's shared memory must stay within the default limit");
    tree_window_kernel<SLOTS, R, WARPS><<<dim3(DK / (R * WARPS), HV, n_seq), 32 * WARPS, smem, st>>>(
        q, k, v, g, beta, S, seq_start, slot, o, ow, plan, plan_stride);
    return GB_CHECK_LAUNCH();
}

// commit (conv): new state = last 3 rows of concat(old 3 rows, qkv rows 0..keep). One thread owns one (sequence,
// channel): it reads every source value before writing, and no other thread touches that channel's column.
// keep == nullptr means "all rows of the sequence" (prefill).
__global__ void __launch_bounds__(256) conv_commit_kernel(
        const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* conv_pool, const int* __restrict__ seq_start,
        const int* __restrict__ slot, const int* __restrict__ keep, const int* __restrict__ rows, int rows_stride) {
    const int seq = blockIdx.y, ch = blockIdx.x * 256 + threadIdx.x;
    const int begin = seq_start[seq], len = seq_start[seq + 1] - begin;
    const int n = keep ? min(max(keep[seq], 0), len) : len;
    if (n <= 0 || ch >= CONV) return;
    __nv_bfloat16* pool = conv_pool + (size_t)slot[seq] * KEEP * CONV + ch;
    __nv_bfloat16 nw[KEEP];
#pragma unroll
    for (int i = 0; i < KEEP; ++i) {
        const int e = n + i;  // index into concat(old 3 rows, the kept rows' qkv): a prefix of the window, or the accepted path's rows
        if (e < KEEP) {
            nw[i] = pool[(size_t)e * CONV];
        } else {
            const int qrow = rows ? rows[(size_t)seq * rows_stride + (e - KEEP)] : begin + e - KEEP;
            nw[i] = qkv[(size_t)qrow * CONV + ch];
        }
    }
#pragma unroll
    for (int i = 0; i < KEEP; ++i) pool[(size_t)i * CONV] = nw[i];
}

// ---------------------------------------------------------------------------------------------------------------
// prefill chain (TensorFold gdn_prefill.cu): the sequential recurrence with the state in registers, TPR threads per
// value row each holding 128/TPR keys; steps staged through shared memory STEPS at a time. Summation order differs
// from step() (and fmaf is used), but nothing depends on where a chunk starts, so any chunking gives the same bits.
// ---------------------------------------------------------------------------------------------------------------
template <int ROWS, int TPR, int STEPS>
__global__ void __launch_bounds__(TPR * ROWS) chain_kernel(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const float* __restrict__ g, const float* __restrict__ beta, float* S, const int* __restrict__ seq_start,
        const int* __restrict__ slot, __nv_bfloat16* __restrict__ o) {
    constexpr int NT = TPR * ROWS;   // threads per block
    constexpr int J = 32 / TPR;      // float4 chunks of a key row each thread owns (chunk index TPR*j + part)
    __shared__ float4 ks[STEPS][DK / 4], qs[STEPS][DK / 4];
    __shared__ float gs[STEPS], bs[STEPS];
    __shared__ __nv_bfloat16 vs[STEPS][ROWS];
    const int head = blockIdx.x, seq = blockIdx.z;
    const int begin = seq_start[seq], W = seq_start[seq + 1] - begin;
    if (W <= 0) return;
    const int row0 = blockIdx.y * ROWS, lrow = threadIdx.x / TPR, row = row0 + lrow, part = threadIdx.x % TPR;
    const int key_head = head / (HV / HK);
    float s[4 * J];
    float* sp = S + (size_t)slot[seq] * S_SLOT + ((size_t)head * DK + row) * DK;
#pragma unroll
    for (int j = 0; j < J; ++j) {
        const float4 t = *reinterpret_cast<const float4*>(sp + 4 * (TPR * j + part));
        s[4 * j] = t.x; s[4 * j + 1] = t.y; s[4 * j + 2] = t.z; s[4 * j + 3] = t.w;
    }
    for (int t0 = 0; t0 < W; t0 += STEPS) {
        const int n = min(STEPS, W - t0);
        __syncthreads();
        for (int i = threadIdx.x; i < n * (DK / 4); i += NT) {
            const int st = i / (DK / 4), c = i % (DK / 4);
            const size_t at = ((size_t)(begin + t0 + st) * HK + key_head) * DK + 4 * c;
            ks[st][c] = load4f(k + at);
            qs[st][c] = load4f(q + at);
        }
        for (int i = threadIdx.x; i < n; i += NT) {
            gs[i] = g[(size_t)(begin + t0 + i) * HV + head];
            bs[i] = beta[(size_t)(begin + t0 + i) * HV + head];
        }
        for (int i = threadIdx.x; i < n * ROWS; i += NT) {
            const int st = i / ROWS, rr = i % ROWS;
            vs[st][rr] = v[((size_t)(begin + t0 + st) * HV + head) * DK + row0 + rr];
        }
        __syncthreads();
        for (int tt = 0; tt < n; ++tt) {
            const float gt = gs[tt], bt = bs[tt];
            const float vt = __bfloat162float(vs[tt][lrow]);
            float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < J; ++j) {
                const float4 kk = ks[tt][TPR * j + part];
                s[4 * j] = s[4 * j] * gt;
                s[4 * j + 1] = s[4 * j + 1] * gt;
                s[4 * j + 2] = s[4 * j + 2] * gt;
                s[4 * j + 3] = s[4 * j + 3] * gt;
                m[0] = __fmaf_rn(s[4 * j], kk.x, m[0]);
                m[1] = __fmaf_rn(s[4 * j + 1], kk.y, m[1]);
                m[2] = __fmaf_rn(s[4 * j + 2], kk.z, m[2]);
                m[3] = __fmaf_rn(s[4 * j + 3], kk.w, m[3]);
            }
            float mem = (m[0] + m[1]) + (m[2] + m[3]);
#pragma unroll
            for (int d = 1; d < TPR; d <<= 1) mem = mem + __shfl_xor_sync(0xffffffffu, mem, d);
            const float delta = (vt - mem) * bt;
            float oo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < J; ++j) {
                const float4 kk = ks[tt][TPR * j + part], qq = qs[tt][TPR * j + part];
                s[4 * j] = __fmaf_rn(kk.x, delta, s[4 * j]);
                s[4 * j + 1] = __fmaf_rn(kk.y, delta, s[4 * j + 1]);
                s[4 * j + 2] = __fmaf_rn(kk.z, delta, s[4 * j + 2]);
                s[4 * j + 3] = __fmaf_rn(kk.w, delta, s[4 * j + 3]);
                oo[0] = __fmaf_rn(s[4 * j], qq.x, oo[0]);
                oo[1] = __fmaf_rn(s[4 * j + 1], qq.y, oo[1]);
                oo[2] = __fmaf_rn(s[4 * j + 2], qq.z, oo[2]);
                oo[3] = __fmaf_rn(s[4 * j + 3], qq.w, oo[3]);
            }
            float out = (oo[0] + oo[1]) + (oo[2] + oo[3]);
#pragma unroll
            for (int d = 1; d < TPR; d <<= 1) out = out + __shfl_xor_sync(0xffffffffu, out, d);
            if (part == 0) o[((size_t)(begin + t0 + tt) * HV + head) * DK + row] = __float2bfloat16_rn(out);
        }
    }
#pragma unroll
    for (int j = 0; j < J; ++j)
        *reinterpret_cast<float4*>(sp + 4 * (TPR * j + part)) = make_float4(s[4 * j], s[4 * j + 1], s[4 * j + 2], s[4 * j + 3]);
}

constexpr int PF_ROWS = 64;   // value rows per block
constexpr int PF_TPR = 4;     // threads per value row
constexpr int PF_STEPS = 16;  // steps staged in shared memory per round

constexpr size_t align256(size_t x) { return (x + 255) & ~(size_t)255; }

struct PrefillWs {
    size_t q, k, v, g, beta, total;
    explicit PrefillWs(size_t rows) {
        q = 0;
        k = q + align256(rows * HK * DK * 2);
        v = k + align256(rows * HK * DK * 2);
        g = v + align256(rows * HV * DK * 2);
        beta = g + align256(rows * HV * 4);
        total = beta + align256(rows * HV * 4);
    }
};

int launch_pre(const __nv_bfloat16* qkv, const __nv_bfloat16* a, const __nv_bfloat16* b, const __nv_bfloat16* conv_w,
               const float* a_log, const float* dt_bias, const __nv_bfloat16* conv_pool, const int* seq_start,
               const int* slot, int n_seq, int rows, __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v, float* g,
               float* beta, cudaStream_t st, const int* parent = nullptr) {
    if (rows <= 0 || n_seq <= 0) return 0;
    gbp_pdl::launch_pdl(flash_pdl(), pre_kernel, dim3(rows, NUNITS / 4), dim3(128), 0, st, qkv, a, b, conv_w, a_log, dt_bias, conv_pool,
                        seq_start, slot, n_seq, q, k, v, g, beta, parent);
    return GB_CHECK_LAUNCH();
}

}  // namespace

GB_EXPORT int GB_GDN_EXPORT(gb_gdn_pre)(const void* qkv, const void* a, const void* b, const void* conv_w, const float* a_log,
                         const float* dt_bias, const void* conv_pool, const int* seq_start, const int* slot, int n_seq,
                         int rows, void* q, void* k, void* v, float* g, float* beta, void* stream) {
    return launch_pre((const __nv_bfloat16*)qkv, (const __nv_bfloat16*)a, (const __nv_bfloat16*)b,
                      (const __nv_bfloat16*)conv_w, a_log, dt_bias, (const __nv_bfloat16*)conv_pool, seq_start, slot,
                      n_seq, rows, (__nv_bfloat16*)q, (__nv_bfloat16*)k, (__nv_bfloat16*)v, g, beta, gb_stream(stream));
}

GB_EXPORT int GB_GDN_EXPORT(gb_gdn_window)(const void* q, const void* k, const void* v, const float* g, const float* beta,
                            const float* S, const int* seq_start, const int* slot, int n_seq, void* o, void* stream) {
    if (n_seq <= 0) return 0;
    const Owed none{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0};
    gbp_pdl::launch_pdl(flash_pdl(), window_kernel, dim3(DK / (R_STEP * W_STEP), HV, n_seq), dim3(32 * W_STEP), 0, gb_stream(stream),
                        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v, g, beta, const_cast<float*>(S),
                        seq_start, slot, (__nv_bfloat16*)o, none);
    return GB_CHECK_LAUNCH();
}

// gb_gdn_window plus the commit the previous window still owes (see Owed): S is read and, for a sequence with owed rows,
// written. The owed record is the previous window's (k, v, g, beta, as gb_gdn_pre filled them).
GB_EXPORT int GB_GDN_EXPORT(gb_gdn_window_owed)(const void* q, const void* k, const void* v, const float* g, const float* beta, float* S,
                                 const int* seq_start, const int* slot, int n_seq, void* o, const void* owed_k,
                                 const void* owed_v, const float* owed_g, const float* owed_beta, const int* owed_row0,
                                 const int* owed_keep, void* stream) {
    if (n_seq <= 0) return 0;
    const Owed ow{(const __nv_bfloat16*)owed_k, (const __nv_bfloat16*)owed_v, owed_g, owed_beta, owed_row0, owed_keep, nullptr, 0};
    gbp_pdl::launch_pdl(flash_pdl(), window_kernel, dim3(DK / (R_STEP * W_STEP), HV, n_seq), dim3(32 * W_STEP), 0, gb_stream(stream),
                        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v, g, beta, S, seq_start, slot,
                        (__nv_bfloat16*)o, ow);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int GB_GDN_EXPORT(gb_gdn_commit)(const void* k, const void* v, const float* g, const float* beta, const void* qkv, float* S,
                            void* conv_pool, const int* seq_start, const int* slot, const int* keep, int n_seq,
                            void* stream) {
    if (n_seq <= 0) return 0;
    cudaStream_t st = gb_stream(stream);
    commit_kernel<<<dim3(DK / (R_STEP * W_STEP), HV, n_seq), 32 * W_STEP, 0, st>>>(
        (const __nv_bfloat16*)k, (const __nv_bfloat16*)v, g, beta, S, seq_start, slot, keep, nullptr, 0);
    int rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    conv_commit_kernel<<<dim3(CONV / 256, n_seq), 256, 0, st>>>((const __nv_bfloat16*)qkv, (__nv_bfloat16*)conv_pool,
                                                                seq_start, slot, keep, nullptr, 0);
    return GB_CHECK_LAUNCH();
}

// The two halves of gb_gdn_commit on their own: the recurrent state S, and the conv history. The model keeps the conv
// commit eager (the next window's pre reads it) and defers the S replay into the next window kernel (gb_gdn_window_owed).
GB_EXPORT int GB_GDN_EXPORT(gb_gdn_commit_state)(const void* k, const void* v, const float* g, const float* beta, float* S,
                                  const int* seq_start, const int* slot, const int* keep, int n_seq, void* stream) {
    if (n_seq <= 0) return 0;
    commit_kernel<<<dim3(DK / (R_STEP * W_STEP), HV, n_seq), 32 * W_STEP, 0, gb_stream(stream)>>>(
        (const __nv_bfloat16*)k, (const __nv_bfloat16*)v, g, beta, S, seq_start, slot, keep, nullptr, 0);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int GB_GDN_EXPORT(gb_gdn_commit_conv)(const void* qkv, void* conv_pool, const int* seq_start, const int* slot, const int* keep,
                                 int n_seq, void* stream) {
    if (n_seq <= 0) return 0;
    conv_commit_kernel<<<dim3(CONV / 256, n_seq), 256, 0, gb_stream(stream)>>>(
        (const __nv_bfloat16*)qkv, (__nv_bfloat16*)conv_pool, seq_start, slot, keep, nullptr, 0);
    return GB_CHECK_LAUNCH();
}

// ---- tree windows (docs/TREE_DRAFTING.md): the same kernels over a window whose rows form a tree per sequence ----

// pre over a tree: `parent` i32 [rows] (the parent's global row, -1 for a root) gives each row's conv history along its root path.
GB_EXPORT int GB_GDN_EXPORT(gb_gdn_pre_tree)(const void* qkv, const void* a, const void* b, const void* conv_w, const float* a_log,
                              const float* dt_bias, const void* conv_pool, const int* seq_start, const int* slot, int n_seq,
                              int rows, void* q, void* k, void* v, float* g, float* beta, const int* parent, void* stream) {
    if (!parent) return (int)cudaErrorInvalidValue;
    return launch_pre((const __nv_bfloat16*)qkv, (const __nv_bfloat16*)a, (const __nv_bfloat16*)b,
                      (const __nv_bfloat16*)conv_w, a_log, dt_bias, (const __nv_bfloat16*)conv_pool, seq_start, slot,
                      n_seq, rows, (__nv_bfloat16*)q, (__nv_bfloat16*)k, (__nv_bfloat16*)v, g, beta, gb_stream(stream),
                      parent);
}

// The window kernel over a tree per sequence: `plan` i32 [n_seq][plan_stride] (plan_stride >= 3 * the longest window, at most 384) holds
// each sequence's entries (node, source, dest) with LOCAL node indices (see tree_window_kernel); `tier` is the slot tier the plans need
// (0, 1, 2, 4, 8, 16 or 32). The owed replay is the window_owed one, with an optional accepted path (`owed_rows` i32
// [n_seq][owed_rows_stride], global rows of the previous record; nullptr: a prefix from owed_row0). S is read, and written for a
// sequence with owed rows. owed_keep == nullptr: nothing is owed.
GB_EXPORT int GB_GDN_EXPORT(gb_gdn_window_tree)(const void* q, const void* k, const void* v, const float* g, const float* beta, float* S,
                                 const int* seq_start, const int* slot, int n_seq, void* o, const void* owed_k,
                                 const void* owed_v, const float* owed_g, const float* owed_beta, const int* owed_row0,
                                 const int* owed_keep, const int* owed_rows, int owed_rows_stride, const int* plan,
                                 int plan_stride, int tier, void* stream) {
    if (n_seq <= 0) return 0;
    if (!plan || plan_stride < 3 || plan_stride > 3 * 128 || (owed_rows && owed_rows_stride <= 0)) return (int)cudaErrorInvalidValue;
    const Owed ow{(const __nv_bfloat16*)owed_k, (const __nv_bfloat16*)owed_v, owed_g, owed_beta, owed_row0, owed_keep, owed_rows,
                  owed_rows_stride};
    cudaStream_t st = gb_stream(stream);
    const __nv_bfloat16 *qq = (const __nv_bfloat16*)q, *kk = (const __nv_bfloat16*)k, *vv = (const __nv_bfloat16*)v;
    __nv_bfloat16* oo = (__nv_bfloat16*)o;
    switch (tier) {
        case 0: return launch_tree_window<0, 8, 4>(qq, kk, vv, g, beta, S, seq_start, slot, n_seq, oo, ow, plan, plan_stride, st);
        case 1: return launch_tree_window<1, 8, 4>(qq, kk, vv, g, beta, S, seq_start, slot, n_seq, oo, ow, plan, plan_stride, st);
        case 2: return launch_tree_window<2, 8, 4>(qq, kk, vv, g, beta, S, seq_start, slot, n_seq, oo, ow, plan, plan_stride, st);
        case 4: return launch_tree_window<4, 4, 4>(qq, kk, vv, g, beta, S, seq_start, slot, n_seq, oo, ow, plan, plan_stride, st);
        case 8: return launch_tree_window<8, 2, 4>(qq, kk, vv, g, beta, S, seq_start, slot, n_seq, oo, ow, plan, plan_stride, st);
        case 16: return launch_tree_window<16, 2, 2>(qq, kk, vv, g, beta, S, seq_start, slot, n_seq, oo, ow, plan, plan_stride, st);
        case 32: return launch_tree_window<32, 2, 1>(qq, kk, vv, g, beta, S, seq_start, slot, n_seq, oo, ow, plan, plan_stride, st);
        default: return (int)cudaErrorInvalidValue;
    }
}

// The two halves of a commit over an accepted PATH: keep[i] rows, row j of them rows[i * rows_stride + j] (global window rows).
GB_EXPORT int GB_GDN_EXPORT(gb_gdn_commit_state_rows)(const void* k, const void* v, const float* g, const float* beta, float* S,
                                       const int* seq_start, const int* slot, const int* keep, const int* rows,
                                       int rows_stride, int n_seq, void* stream) {
    if (n_seq <= 0) return 0;
    if (!rows || rows_stride <= 0) return (int)cudaErrorInvalidValue;
    commit_kernel<<<dim3(DK / (R_STEP * W_STEP), HV, n_seq), 32 * W_STEP, 0, gb_stream(stream)>>>(
        (const __nv_bfloat16*)k, (const __nv_bfloat16*)v, g, beta, S, seq_start, slot, keep, rows, rows_stride);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int GB_GDN_EXPORT(gb_gdn_commit_conv_rows)(const void* qkv, void* conv_pool, const int* seq_start, const int* slot, const int* keep,
                                      const int* rows, int rows_stride, int n_seq, void* stream) {
    if (n_seq <= 0) return 0;
    if (!rows || rows_stride <= 0) return (int)cudaErrorInvalidValue;
    conv_commit_kernel<<<dim3(CONV / 256, n_seq), 256, 0, gb_stream(stream)>>>(
        (const __nv_bfloat16*)qkv, (__nv_bfloat16*)conv_pool, seq_start, slot, keep, rows, rows_stride);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT size_t GB_GDN_EXPORT(gb_gdn_prefill_workspace_bytes)(int rows) { return PrefillWs((size_t)(rows > 0 ? rows : 0)).total; }

GB_EXPORT int GB_GDN_EXPORT(gb_gdn_prefill)(const void* qkv, const void* a, const void* b, const void* conv_w, const float* a_log,
                             const float* dt_bias, float* S, void* conv_pool, const int* seq_start, const int* slot,
                             int n_seq, int rows, void* workspace, void* o, void* stream) {
    if (rows <= 0 || n_seq <= 0) return 0;
    cudaStream_t st = gb_stream(stream);
    const PrefillWs w((size_t)rows);
    char* ws = (char*)workspace;
    __nv_bfloat16 *q = (__nv_bfloat16*)(ws + w.q), *k = (__nv_bfloat16*)(ws + w.k), *v = (__nv_bfloat16*)(ws + w.v);
    float *g = (float*)(ws + w.g), *beta = (float*)(ws + w.beta);
    int rc = launch_pre((const __nv_bfloat16*)qkv, (const __nv_bfloat16*)a, (const __nv_bfloat16*)b,
                        (const __nv_bfloat16*)conv_w, a_log, dt_bias, (const __nv_bfloat16*)conv_pool, seq_start, slot,
                        n_seq, rows, q, k, v, g, beta, st);
    if (rc) return rc;
    chain_kernel<PF_ROWS, PF_TPR, PF_STEPS><<<dim3(HV, DK / PF_ROWS, n_seq), PF_TPR * PF_ROWS, 0, st>>>(q, k, v, g, beta, S, seq_start, slot,
                                                                            (__nv_bfloat16*)o);
    rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    conv_commit_kernel<<<dim3(CONV / 256, n_seq), 256, 0, st>>>((const __nv_bfloat16*)qkv, (__nv_bfloat16*)conv_pool,
                                                                seq_start, slot, nullptr, nullptr, 0);
    return GB_CHECK_LAUNCH();
}
