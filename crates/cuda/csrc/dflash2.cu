// DFlash2 draft kernels: grouped dynamic convolution, context K/V write (k_norm + rope into a ring), draft attention
// (split-key, deterministic) and the candidate-selector chain walk.
//
// Conventions (docs/QWEN38_SPEC.md section 8): activations bf16, fp32 accumulation, one rounding to bf16 per op
// boundary, no atomics, fixed reduction orders, no cudaMalloc and no host sync in any launcher.
//
// Derived from (MIT): ExLlamaV3 fork r0b0tlab/exllamav3 @44636f0 exllamav3_ext/dflash2.cu (dynconv recurrence,
// selector-walk reduction structure) and exllamav3/architecture/dflash.py (ctx K/V recipe). See THIRD_PARTY_NOTICES.md.
#include "common.cuh"
#include "keyed.cuh"
#include <math.h>

namespace {

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 f2bf(float v) { return __float2bfloat16_rn(v); }
// bf16 pair packed in a 32-bit word: low half is the lower address
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

// ---------------------------------------------------------------------------------------------------------------
// 1. Grouped dynamic conv (two taps), one row block = `rows` consecutive rows of one sequence.
//    out[t,c] = bf16( [resid[t,c] +] sum_{tap<2} (base[which,tap,c] + dyn[t,which,tap,c/G]) * u[t-tap,c] )
//    dyn row layout [which(2)][tap(2)][group(hidden/G)]. u[t<0] = 0 inside a block (never crosses blocks).
// ---------------------------------------------------------------------------------------------------------------
constexpr int DC_THREADS = 128;
constexpr int DC_VEC = 8;  // channels per thread (one 16-byte load of bf16)

__global__ void __launch_bounds__(DC_THREADS)
    dynconv_kernel(const __nv_bfloat16* __restrict__ u, const __nv_bfloat16* __restrict__ dyn, const __half* __restrict__ base,
                   const __nv_bfloat16* resid, __nv_bfloat16* out, int which, int rows, int hidden, int group) {
    const int c0 = (blockIdx.x * DC_THREADS + threadIdx.x) * DC_VEC;
    if (c0 >= hidden) return;
    const int row = blockIdx.y;
    const int t = row % rows;
    const int groups = hidden / group;
    const int g = c0 / group;  // group is a multiple of 8, so the 8 channels share one group
    const __nv_bfloat16* d = dyn + (size_t)row * 4 * groups + (size_t)which * 2 * groups;
    const float dy0 = bf2f(d[g]);
    const float dy1 = bf2f(d[groups + g]);

    const uint4 b0 = *reinterpret_cast<const uint4*>(base + ((size_t)which * 2 + 0) * hidden + c0);
    const uint4 b1 = *reinterpret_cast<const uint4*>(base + ((size_t)which * 2 + 1) * hidden + c0);
    const uint4 x0 = *reinterpret_cast<const uint4*>(u + (size_t)row * hidden + c0);
    uint4 x1 = make_uint4(0, 0, 0, 0);
    if (t >= 1) x1 = *reinterpret_cast<const uint4*>(u + (size_t)(row - 1) * hidden + c0);

    const __half2* b0h = reinterpret_cast<const __half2*>(&b0);
    const __half2* b1h = reinterpret_cast<const __half2*>(&b1);
    const uint32_t* x0w = reinterpret_cast<const uint32_t*>(&x0);
    const uint32_t* x1w = reinterpret_cast<const uint32_t*>(&x1);
    float r[DC_VEC];
#pragma unroll
    for (int i = 0; i < DC_VEC / 2; ++i) {
        const float2 bw0 = __half22float2(b0h[i]);
        const float2 bw1 = __half22float2(b1h[i]);
        const float xa0 = bf_lo(x0w[i]), xb0 = bf_hi(x0w[i]);
        const float xa1 = bf_lo(x1w[i]), xb1 = bf_hi(x1w[i]);
        // tap 0 then tap 1, each (base + dyn) formed in fp32, fused multiply-adds in a fixed order
        float acc_a = __fmaf_rn(bw0.x + dy0, xa0, 0.0f);
        float acc_b = __fmaf_rn(bw0.y + dy0, xb0, 0.0f);
        if (t >= 1) {
            acc_a = __fmaf_rn(bw1.x + dy1, xa1, acc_a);
            acc_b = __fmaf_rn(bw1.y + dy1, xb1, acc_b);
        }
        r[2 * i] = acc_a;
        r[2 * i + 1] = acc_b;
    }
    if (resid) {
        const uint4 rv = *reinterpret_cast<const uint4*>(resid + (size_t)row * hidden + c0);
        const uint32_t* rw = reinterpret_cast<const uint32_t*>(&rv);
#pragma unroll
        for (int i = 0; i < DC_VEC / 2; ++i) {
            r[2 * i] = bf_lo(rw[i]) + r[2 * i];
            r[2 * i + 1] = bf_hi(rw[i]) + r[2 * i + 1];
        }
    }
    uint4 o;
    __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int i = 0; i < DC_VEC / 2; ++i) oh[i] = __floats2bfloat162_rn(r[2 * i], r[2 * i + 1]);
    *reinterpret_cast<uint4*>(out + (size_t)row * hidden + c0) = o;
}

// ---------------------------------------------------------------------------------------------------------------
// 2. Context K/V write. One block per context row, one warp per kv head.
//    k = rope(rmsnorm_plain(kproj_head) * gamma, pos)   (one rounding, at the store),  v = vproj copied.
//    Cache layout: [slots, cap, kv_heads, 128] bf16 for K and for V; position pos lives at ring index pos % cap.
// ---------------------------------------------------------------------------------------------------------------
constexpr int HD = 128;

__global__ void __launch_bounds__(32 * 8)
    ctx_kv_kernel(const __nv_bfloat16* __restrict__ kproj, const __nv_bfloat16* __restrict__ vproj, const float* __restrict__ gamma,
                  const int* __restrict__ positions, const int* __restrict__ slots, __nv_bfloat16* kc, __nv_bfloat16* vc, int cap,
                  int kv_heads, float eps, double theta) {
    __shared__ float s_cos[HD / 2], s_sin[HD / 2];
    const int row = blockIdx.x;
    const int pos = positions[row];
    const int slot = slots[row];
    if (threadIdx.x < HD / 2) {
        // angle in double: pos * theta^(-2i/128) reaches ~1e5 rad, fp32 would lose ~1e-2 rad
        const double inv_freq = pow(theta, -(double)threadIdx.x / (double)(HD / 2));
        double sn, cs;
        sincos((double)pos * inv_freq, &sn, &cs);
        s_cos[threadIdx.x] = (float)cs;
        s_sin[threadIdx.x] = (float)sn;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t ring = (size_t)slot * cap + (size_t)(pos % cap);
    for (int h = warp; h < kv_heads; h += 8) {
        const size_t src = ((size_t)row * kv_heads + h) * HD;
        const size_t dst = (ring * kv_heads + h) * HD;
        float x[4];
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            x[j] = bf2f(kproj[src + lane + 32 * j]);
            ss = __fmaf_rn(x[j], x[j], ss);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float inv = rsqrtf(ss * (1.0f / HD) + eps);
        float y[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) y[j] = x[j] * inv * gamma[lane + 32 * j];
        // rotate-half pairs (d, d+64): lane -> (j0, j2) uses freq `lane`; (j1, j3) uses freq `lane+32`
        const float c0 = s_cos[lane], n0 = s_sin[lane], c1 = s_cos[lane + 32], n1 = s_sin[lane + 32];
        kc[dst + lane] = f2bf(y[0] * c0 - y[2] * n0);
        kc[dst + lane + 64] = f2bf(y[2] * c0 + y[0] * n0);
        kc[dst + lane + 32] = f2bf(y[1] * c1 - y[3] * n1);
        kc[dst + lane + 96] = f2bf(y[3] * c1 + y[1] * n1);
#pragma unroll
        for (int j = 0; j < 4; ++j) vc[dst + lane + 32 * j] = vproj[src + lane + 32 * j];
    }
}

// ---------------------------------------------------------------------------------------------------------------
// 3. Draft attention. 8..16 query rows per sequence (the draft block: REXL3_DRAFT_BLOCK, 8 the trained value), 32 q heads,
//    8 kv heads (group 4), head_dim 128. Row t (query position P+t) sees ctx keys kpos in [max(0,P-window)... with
//    kpos >= P+t-window, kpos < P] and block keys 0..=t. Phase 1: one block per (key chunk, kv head, sequence) writes
//    (m, l, acc[128]) per (row, q head); chunk `nchunks` is the in-block (causal) chunk, ONE 64-key tile whatever the
//    block: keys `block..63` of it are padding and never visible (vis = key <= i). Phase 2 merges chunks in index order.
//    m is kept in the log2 domain (scores pre-multiplied by scale * log2(e)).
//
//    The per-query reduction structure is the same at every block (8 lanes x 8 keys a query in the softmax, the P.V fmaf
//    order over the tile's keys; the queries past 32 run as further passes of the same shape), and a row's visible keys do
//    not change with the block, so rows 0..7 of a 16-block are the 8-block's bits exactly (attn_block16_levels_1_to_7_
//    equal_block8 pins it: the deeper levels must not move the trained ones).
// ---------------------------------------------------------------------------------------------------------------
constexpr int GQA = 4;
constexpr int KT = 64;          // keys per chunk
constexpr int AT = 256;         // threads
constexpr int KSTRIDE_W = 65;   // K tile row stride in 32-bit words (odd: conflict free per-thread row reads)
constexpr int VSTRIDE_E = 136;  // V tile row stride in bf16 elements
constexpr int SSTRIDE = KT + 1;
constexpr int WS_STRIDE = 132;  // floats per (row, q head, chunk): m, l, pad, pad, acc[128]
constexpr float LOG2E = 1.4426950408889634f;

constexpr int TPB = 2;      // tiles (of KT keys) per block; the block keeps a running softmax across them
constexpr int QSUB = 32;    // queries the softmax and P.V passes handle at once (one pass at the trained block 8)

template <int ROWS>
__global__ void __launch_bounds__(AT)
    attn_split_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ bk, const __nv_bfloat16* __restrict__ bv,
                      const __nv_bfloat16* __restrict__ kc, const __nv_bfloat16* __restrict__ vc, const int* __restrict__ slots,
                      const int* __restrict__ Pv, float* __restrict__ ws, int cap, int window, int nchunks, int n_kv_heads, int n_q_heads) {
    constexpr int NQ = GQA * ROWS;  // queries per kv head
    extern __shared__ float smem_dyn[];  // Qs, Ss, s_alpha, s_m, s_l (dynamic: NQ=64 needs more than the static limit)
    float* Qs = smem_dyn;
    float* Ss = Qs + NQ * HD;
    float* s_alpha = Ss + NQ * SSTRIDE;
    float* s_m = s_alpha + NQ;
    float* s_l = s_m + NQ;
    __shared__ __align__(16) uint32_t tile[KT * KSTRIDE_W > KT * (VSTRIDE_E / 2) ? KT * KSTRIDE_W : KT * (VSTRIDE_E / 2)];

    const int c = blockIdx.x, h = blockIdx.y, s = blockIdx.z;
    const int tid = threadIdx.x;
    const int P = Pv[s];
    const bool is_blk = (c == nchunks);
    const int lo_min = max(0, P - window);
    const int kpos_chunk = lo_min + c * (KT * TPB);
    const size_t wrow_base = ((size_t)s * ROWS) * n_q_heads;  // (s*8 + t) * n_q_heads + head

    auto ws_ptr = [&](int qi, int chunk) -> float* {
        const int hq = qi / ROWS, t = qi % ROWS;
        return ws + ((wrow_base + (size_t)t * n_q_heads + h * GQA + hq) * (nchunks + 1) + chunk) * WS_STRIDE;
    };

    if (!is_blk && kpos_chunk >= P) {  // nothing to attend in this chunk
        if (tid < NQ) {
            float* w = ws_ptr(tid, c);
            w[0] = -INFINITY;
            w[1] = 0.f;
        }
        return;
    }

    // ---- Q (pre-scaled to the log2 domain) ----
    const float qscale = rsqrtf((float)HD) * LOG2E;
#pragma unroll 4
    for (int i = tid; i < NQ * HD; i += AT) {
        const int qi = i >> 7, d = i & (HD - 1);
        const int hq = qi / ROWS, t = qi % ROWS;
        Qs[i] = bf2f(q[(((size_t)s * ROWS + t) * n_q_heads + h * GQA + hq) * HD + d]) * qscale;
    }
    if (tid < NQ) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.f;
    }

    // PV accumulators: thread = (4 dims, 4 queries); NQ > QSUB (a 16-block) runs further passes of the same shape
    constexpr int NPASS = (NQ + QSUB - 1) / QSUB;
    const int dq = tid & 31, g8 = tid >> 5;
    float acc[NPASS][4][4];
#pragma unroll
    for (int p = 0; p < NPASS; ++p)
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[p][i][j] = 0.f;

#pragma unroll
    for (int tl = 0; tl < TPB; ++tl) {
        if (is_blk && tl > 0) break;
        const int kpos0 = kpos_chunk + tl * KT;
        if (!is_blk && kpos0 >= P) break;  // uniform across the block

        // ---- K tile ----
        {
            const int seg = tid & 15, r0 = tid >> 4;  // 16 threads per key row, 16 rows per pass
#pragma unroll
            for (int pass = 0; pass < KT / 16; ++pass) {
                const int r = r0 + pass * 16;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (is_blk) {
                    if (r < ROWS) v = *reinterpret_cast<const uint4*>(bk + (((size_t)s * ROWS + r) * n_kv_heads + h) * HD + seg * 8);
                } else {
                    const int kp = kpos0 + r;
                    if (kp < P) {
                        const size_t ring = (size_t)slots[s] * cap + (size_t)(kp % cap);
                        v = *reinterpret_cast<const uint4*>(kc + (ring * n_kv_heads + h) * HD + seg * 8);
                    }
                }
                uint32_t* dst = tile + r * KSTRIDE_W + seg * 4;
                dst[0] = v.x;
                dst[1] = v.y;
                dst[2] = v.z;
                dst[3] = v.w;
            }
        }
        __syncthreads();

        // ---- scores: thread = (key, q head); loops over the block's rows ----
        {
            const int key = tid & (KT - 1), hq = tid >> 6;
            float sc[ROWS];
#pragma unroll
            for (int i = 0; i < ROWS; ++i) sc[i] = 0.f;
            const uint32_t* krow = tile + key * KSTRIDE_W;
#pragma unroll
            for (int dc = 0; dc < HD / 32; ++dc) {
                float kv[32];
#pragma unroll
                for (int w = 0; w < 16; ++w) {
                    const uint32_t word = krow[dc * 16 + w];
                    kv[2 * w] = bf_lo(word);
                    kv[2 * w + 1] = bf_hi(word);
                }
#pragma unroll
                for (int i = 0; i < ROWS; ++i) {
                    const float4* qp = reinterpret_cast<const float4*>(Qs + (hq * ROWS + i) * HD + dc * 32);
                    float a = sc[i];
#pragma unroll
                    for (int w = 0; w < 8; ++w) {
                        const float4 qv = qp[w];
                        a = __fmaf_rn(qv.x, kv[4 * w], a);
                        a = __fmaf_rn(qv.y, kv[4 * w + 1], a);
                        a = __fmaf_rn(qv.z, kv[4 * w + 2], a);
                        a = __fmaf_rn(qv.w, kv[4 * w + 3], a);
                    }
                    sc[i] = a;
                }
            }
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                bool vis;
                if (is_blk) {
                    vis = key <= i;  // causal inside the block (keys `block..63` are padding and never visible)
                } else {
                    const int kp = kpos0 + key;
                    vis = (kp < P) && (kp + window >= P + i);
                }
                Ss[(hq * ROWS + i) * SSTRIDE + key] = vis ? sc[i] : -INFINITY;
            }
        }
        __syncthreads();

        // ---- V tile (overwrites the K tile) ----
        {
            const int seg = tid & 15, r0 = tid >> 4;
            __nv_bfloat16* vt = reinterpret_cast<__nv_bfloat16*>(tile);
#pragma unroll
            for (int pass = 0; pass < KT / 16; ++pass) {
                const int r = r0 + pass * 16;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (is_blk) {
                    if (r < ROWS) v = *reinterpret_cast<const uint4*>(bv + (((size_t)s * ROWS + r) * n_kv_heads + h) * HD + seg * 8);
                } else {
                    const int kp = kpos0 + r;
                    if (kp < P) {
                        const size_t ring = (size_t)slots[s] * cap + (size_t)(kp % cap);
                        v = *reinterpret_cast<const uint4*>(vc + (ring * n_kv_heads + h) * HD + seg * 8);
                    }
                }
                *reinterpret_cast<uint4*>(vt + r * VSTRIDE_E + seg * 8) = v;
            }
        }
        // ---- running softmax: 8 lanes per query, 8 keys each, butterfly reductions; QSUB queries a pass ----
        // (a query's 8 lanes always enter or skip together: NQ is a multiple of 4 and a warp covers 4 consecutive queries)
#pragma unroll
        for (int q0 = 0; q0 < NQ; q0 += QSUB) {
            const int qi = q0 + (tid >> 3), sub = tid & 7;
            if (qi >= NQ) break;
            float* srow = Ss + qi * SSTRIDE + sub * 8;
            float mt = -INFINITY;
#pragma unroll
            for (int j = 0; j < 8; ++j) mt = fmaxf(mt, srow[j]);
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, o));
            const float m_old = s_m[qi];
            const float m_new = fmaxf(m_old, mt);
            const float alpha = (m_new == -INFINITY || m_old == -INFINITY) ? ((m_new == -INFINITY) ? 1.f : 0.f) : exp2f(m_old - m_new);
            float lt = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float p = (m_new == -INFINITY) ? 0.f : exp2f(srow[j] - m_new);
                srow[j] = p;
                lt += p;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) lt += __shfl_xor_sync(0xffffffffu, lt, o);
            __syncwarp();  // every lane of the query has read s_m before lane 0 updates it
            if (sub == 0) {
                s_alpha[qi] = alpha;
                s_l[qi] = s_l[qi] * alpha + lt;
                s_m[qi] = m_new;
            }
        }
        __syncthreads();

        // ---- P.V: QSUB queries a pass; the keys past a causal row's own carry p = 0 and change nothing ----
        {
            const __nv_bfloat16* vt = reinterpret_cast<const __nv_bfloat16*>(tile);
            const int nk = is_blk ? ROWS : KT;
#pragma unroll
            for (int q0 = 0; q0 < NQ; q0 += QSUB) {
                if (q0 + g8 * 4 >= NQ) break;  // whole 32-lane group past the queries of this pass
                const int p = q0 / QSUB;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float a = s_alpha[q0 + g8 * 4 + i];
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[p][i][j] *= a;
                }
                for (int k = 0; k < nk; ++k) {
                    const uint2 raw = *reinterpret_cast<const uint2*>(vt + k * VSTRIDE_E + dq * 4);
                    const float v0 = bf_lo(raw.x), v1 = bf_hi(raw.x), v2 = bf_lo(raw.y), v3 = bf_hi(raw.y);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float p_s = Ss[(q0 + g8 * 4 + i) * SSTRIDE + k];
                        acc[p][i][0] = __fmaf_rn(p_s, v0, acc[p][i][0]);
                        acc[p][i][1] = __fmaf_rn(p_s, v1, acc[p][i][1]);
                        acc[p][i][2] = __fmaf_rn(p_s, v2, acc[p][i][2]);
                        acc[p][i][3] = __fmaf_rn(p_s, v3, acc[p][i][3]);
                    }
                }
            }
        }
        __syncthreads();  // tile and Ss are rewritten by the next tile
    }

    // ---- partial (m, l, acc) of this chunk ----
    if (tid < NQ) {
        float* w = ws_ptr(tid, c);
        w[0] = s_m[tid];
        w[1] = s_l[tid];
    }
#pragma unroll
    for (int q0 = 0; q0 < NQ; q0 += QSUB) {
        if (q0 + g8 * 4 >= NQ) break;
        const int p = q0 / QSUB;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float* w = ws_ptr(q0 + g8 * 4 + i, c);
            *reinterpret_cast<float4*>(w + 4 + dq * 4) = make_float4(acc[p][i][0], acc[p][i][1], acc[p][i][2], acc[p][i][3]);
        }
    }
}

// grid (rows * n_q_heads, n_seq), 128 threads (one per dim); `rows` matches the split kernel's block
__global__ void __launch_bounds__(HD)
    attn_merge_kernel(const float* __restrict__ ws, __nv_bfloat16* out, int nsplit, int n_q_heads, int rows) {
    const int rh = blockIdx.x;  // t * n_q_heads + head
    const int s = blockIdx.y;
    const int d = threadIdx.x;
    const float* base = ws + ((size_t)s * rows * n_q_heads + rh) * nsplit * WS_STRIDE;
    float M = -INFINITY;
    for (int c = 0; c < nsplit; ++c) M = fmaxf(M, base[(size_t)c * WS_STRIDE]);
    float L = 0.f, O = 0.f;
    for (int c = 0; c < nsplit; ++c) {
        const float* w = base + (size_t)c * WS_STRIDE;
        const float mc = w[0];
        if (mc == -INFINITY) continue;  // empty chunk: its accumulator was never written
        const float f = exp2f(mc - M);
        L = __fmaf_rn(w[1], f, L);
        O = __fmaf_rn(w[4 + d], f, O);
    }
    out[((size_t)s * rows * n_q_heads + rh) * HD + d] = f2bf(O / L);
}

// ---------------------------------------------------------------------------------------------------------------
// 4. Selector chain walk, one block per sequence. score[c] = unary[c] + sum_r (A[pred,r]*gate[t,r]) * B[cand[c],r].
// ---------------------------------------------------------------------------------------------------------------
constexpr int WT = 256;

__device__ __forceinline__ float gate_at(const void* g, bool f32, size_t i) {
    return f32 ? reinterpret_cast<const float*>(g)[i] : bf2f(reinterpret_cast<const __nv_bfloat16*>(g)[i]);
}

using gb_keyed::keyed_choose;  // csrc/keyed.cuh: the target sampler's keyed draw, shared with the GLM NextN head

struct WalkSampling {
    unsigned long long seed;
    unsigned long long first_position;  // absolute position of draft row 0 (the token the target samples after the anchor)
    double temperature, top_p, min_p;
    int keyed;  // 0: argmax of the selector score (the temperature-0 path)
};

__global__ void __launch_bounds__(WT)
    walk_kernel(const int* __restrict__ anchor, const int* __restrict__ ids, const float* __restrict__ vals, const void* gate, int gate_f32,
                const __half* __restrict__ pred_cb, const __half* __restrict__ succ_cb, int* __restrict__ out, float* conf, int rows, int k,
                int rank, const WalkSampling* __restrict__ ws, const int* __restrict__ row_off, float* margin) {
    extern __shared__ float smem[];
    float* a_g = smem;          // [rank]
    float* scores = smem + rank;  // [k]
    __shared__ int s_pred;
    const int b = blockIdx.x, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    // uniform blocks of `rows` rows each, or (row_off != nullptr) sequence b owns rows row_off[b] .. row_off[b + 1] of a packed array
    const int base = row_off ? row_off[b] : b * rows;
    const int nrows = row_off ? row_off[b + 1] - base : rows;
    if (t == 0) s_pred = anchor[b];
    __syncthreads();
    for (int i = 0; i < nrows; ++i) {
        const __half* a_row = pred_cb + (size_t)s_pred * rank;
        for (int r = t; r < rank; r += WT) a_g[r] = __half2float(a_row[r]) * gate_at(gate, gate_f32 != 0, ((size_t)(base + i)) * rank + r);
        __syncthreads();
        const int* c_row = ids + ((size_t)(base + i)) * k;
        const float* u_row = vals + ((size_t)(base + i)) * k;
        for (int c = warp; c < k; c += WT / 32) {
            const __half* b_row = succ_cb + (size_t)c_row[c] * rank;
            float dot = 0.f;
            for (int r = lane; r < rank; r += 32) dot = __fmaf_rn(a_g[r], __half2float(b_row[r]), dot);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
            if (lane == 0) scores[c] = u_row[c] + dot;
        }
        __syncthreads();
        if (t == 0) {
            int best = 0;
            float bs = scores[0], mg = 30.f;
            if (ws != nullptr && ws[b].keyed) {
                best = keyed_choose(scores, c_row, k, ws[b].seed, ws[b].temperature, ws[b].top_p, ws[b].min_p, ws[b].first_position + i,
                                    margin ? &mg : nullptr);
                bs = scores[best];
            } else {
                for (int c = 1; c < k; ++c)
                    if (scores[c] > bs) {
                        bs = scores[c];
                        best = c;
                    }
                if (margin) {  // the gap to the runner-up (0 when two candidates tie for the lead)
                    float second = -INFINITY;
                    for (int c = 0; c < k; ++c)
                        if (c != best && scores[c] > second) second = scores[c];
                    mg = k > 1 ? fminf(bs - second, 30.f) : 30.f;
                }
            }
            s_pred = c_row[best];
            out[(size_t)(base + i)] = s_pred;
            if (conf) conf[(size_t)(base + i)] = bs;
            if (margin) margin[(size_t)(base + i)] = mg;
        }
        __syncthreads();
    }
}

}  // namespace

// =================================================================================================================
// Launchers
// =================================================================================================================

// u,dyn,out: bf16 [n_seq*rows, ...]; base: f16 [2][2][hidden]; resid (optional, may alias out): bf16 [n_seq*rows, hidden].
// dyn row stride is 4*(hidden/group) elements. Every block of `rows` consecutive rows is an independent causal block.
GB_EXPORT int gb_dflash_dynconv(const void* u, const void* dyn, const void* base, int which, int n_seq, int rows, int hidden, int group,
                                const void* resid, void* out, void* stream) {
    if (n_seq <= 0 || rows <= 0) return 0;
    if (hidden % DC_VEC != 0 || group % DC_VEC != 0 || hidden % group != 0 || which < 0 || which > 1) return (int)cudaErrorInvalidValue;
    const int nthreads = hidden / DC_VEC;
    dim3 grid((nthreads + DC_THREADS - 1) / DC_THREADS, n_seq * rows);
    dynconv_kernel<<<grid, DC_THREADS, 0, gb_stream(stream)>>>((const __nv_bfloat16*)u, (const __nv_bfloat16*)dyn, (const __half*)base,
                                                               (const __nv_bfloat16*)resid, (__nv_bfloat16*)out, which, rows, hidden, group);
    return GB_CHECK_LAUNCH();
}

// kproj,vproj: bf16 [n, kv_heads*128]; gamma: f32 [128]; positions, slots: i32 [n] (per row); kc,vc: bf16 [n_slots, cap, kv_heads, 128].
// Rows of one slot in one launch must span fewer than `cap` consecutive positions (otherwise two rows share a ring entry).
GB_EXPORT int gb_dflash_ctx_kv(const void* kproj, const void* vproj, const void* gamma, const void* positions, const void* slots, int n,
                               int kv_heads, int cap, float eps, double rope_theta, void* kc, void* vc, void* stream) {
    if (n <= 0) return 0;
    if (cap <= 0 || kv_heads <= 0) return (int)cudaErrorInvalidValue;
    ctx_kv_kernel<<<n, 32 * 8, 0, gb_stream(stream)>>>((const __nv_bfloat16*)kproj, (const __nv_bfloat16*)vproj, (const float*)gamma,
                                                       (const int*)positions, (const int*)slots, (__nv_bfloat16*)kc, (__nv_bfloat16*)vc,
                                                       cap, kv_heads, eps, rope_theta);
    return GB_CHECK_LAUNCH();
}

static int attn_nchunks(int window) { return (window + KT * TPB - 1) / (KT * TPB); }

// The two blocks the attention is built for: the trained 8-row block and the 16-row one the depth study runs
// (`REXL3_DRAFT_BLOCK`). Anything else is refused at the launcher, as before.
static bool attn_rows_ok(int rows) { return rows == 8 || rows == 16; }

template <int ROWS>
static int attn_launch(const void* q, const void* block_k, const void* block_v, const void* kc, const void* vc, const void* slots,
                       const void* P, int n_seq, int n_q_heads, int n_kv_heads, int cap, int window, void* ws, void* out,
                       void* stream) {
    constexpr int NQ = GQA * ROWS;
    // Qs, Ss, s_alpha, s_m, s_l: dynamic shared (NQ=64 of a 16-block is past the 48 KB static limit); the tile stays static
    const size_t smem = (size_t)(NQ * HD + NQ * SSTRIDE + 3 * NQ) * sizeof(float);
    if (smem > 48 * 1024) {  // past the default dynamic limit: opt the kernel in (no allocation, no stream sync)
        cudaError_t e = cudaFuncSetAttribute(attn_split_kernel<ROWS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        if (e != cudaSuccess) return (int)e;
    }
    const int nchunks = attn_nchunks(window);
    dim3 grid(nchunks + 1, n_kv_heads, n_seq);
    attn_split_kernel<ROWS><<<grid, AT, smem, gb_stream(stream)>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)block_k, (const __nv_bfloat16*)block_v, (const __nv_bfloat16*)kc,
        (const __nv_bfloat16*)vc, (const int*)slots, (const int*)P, (float*)ws, cap, window, nchunks, n_kv_heads, n_q_heads);
    int rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    dim3 mg(ROWS * n_q_heads, n_seq);
    attn_merge_kernel<<<mg, HD, 0, gb_stream(stream)>>>((const float*)ws, (__nv_bfloat16*)out, nchunks + 1, n_q_heads, ROWS);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT size_t gb_dflash_attn_workspace_bytes(int n_seq, int window, int n_q_heads, int rows) {
    if (!attn_rows_ok(rows)) return 0;
    return (size_t)n_seq * rows * n_q_heads * (attn_nchunks(window) + 1) * WS_STRIDE * sizeof(float);
}

// q: bf16 [n_seq*rows, n_q_heads, 128] (q_norm + rope applied); block_k, block_v: bf16 [n_seq*rows, n_kv_heads, 128] (k_norm + rope for k);
// kc, vc: bf16 [n_slots, cap, n_kv_heads, 128] (this layer's ring); slots, P: i32 [n_seq] (cache slot, position of block row 0 = number of
// committed context tokens); out: bf16 [n_seq*rows, n_q_heads*128]. n_q_heads == 4 * n_kv_heads. window: 2048. cap >= window.
// rows: the draft block, 8 (the trained value) or 16 (the depth study's; REXL3_DRAFT_BLOCK).
GB_EXPORT int gb_dflash_attn(const void* q, const void* block_k, const void* block_v, const void* kc, const void* vc, const void* slots,
                             const void* P, int n_seq, int n_q_heads, int n_kv_heads, int cap, int window, int rows, void* ws, void* out,
                             void* stream) {
    if (n_seq <= 0) return 0;
    if (n_q_heads != GQA * n_kv_heads || cap < window || window <= 0 || !attn_rows_ok(rows)) return (int)cudaErrorInvalidValue;
    if (rows == 16)
        return attn_launch<16>(q, block_k, block_v, kc, vc, slots, P, n_seq, n_q_heads, n_kv_heads, cap, window, ws, out, stream);
    return attn_launch<8>(q, block_k, block_v, kc, vc, slots, P, n_seq, n_q_heads, n_kv_heads, cap, window, ws, out, stream);
}

// anchor: i32 [n_seq]; topk_ids: i32 [n_seq, rows, k]; topk_vals: f32 [n_seq, rows, k]; gate: bf16 or f32 [n_seq, rows, rank];
// codebooks: f16 [V, rank]; out_tokens: i32 [n_seq, rows] (the d_1..d_rows, anchor not included); conf (nullable): f32 [n_seq, rows].
// As above, with `sampling` (device array of `n_seq` WalkSampling records, nullable = all greedy): a keyed sequence picks each draft with the
// target's own sampler draw at that draft's absolute position, so the draft agrees with the target wherever their distributions do.
GB_EXPORT int gb_dflash_selector_walk_keyed(const void* anchor, const void* topk_ids, const void* topk_vals, const void* gate, int gate_is_f32,
                                            const void* pred_codebook, const void* succ_codebook, int n_seq, int rows, int k, int rank,
                                            const void* sampling, void* out_tokens, void* conf, void* stream) {
    if (n_seq <= 0 || rows <= 0) return 0;
    if (k < 1 || k > 64 || rank < 1) return (int)cudaErrorInvalidValue;
    const size_t smem = (size_t)(rank + k) * sizeof(float);
    walk_kernel<<<n_seq, WT, smem, gb_stream(stream)>>>((const int*)anchor, (const int*)topk_ids, (const float*)topk_vals, gate, gate_is_f32,
                                                        (const __half*)pred_codebook, (const __half*)succ_codebook, (int*)out_tokens,
                                                        (float*)conf, rows, k, rank, (const WalkSampling*)sampling, nullptr, nullptr);
    return GB_CHECK_LAUNCH();
}

// The same walk over sequences that have different numbers of rows. topk_ids, topk_vals, gate, out_tokens and conf hold the
// sequences' rows packed one after another; `row_off` (device i32 [n_seq + 1], row_off[0] = 0) says where each sequence's rows start, so
// sequence b walks row_off[b + 1] - row_off[b] rows. Row i of sequence b is drawn at absolute position first_position + i exactly as in
// the uniform launcher, so a sequence's first r picks are the same whether it was given r rows or more.
GB_EXPORT int gb_dflash_selector_walk_ragged(const void* anchor, const void* topk_ids, const void* topk_vals, const void* gate, int gate_is_f32,
                                             const void* pred_codebook, const void* succ_codebook, int n_seq, int k, int rank,
                                             const void* sampling, const void* row_off, void* out_tokens, void* conf, void* margin, void* stream) {
    if (n_seq <= 0) return 0;
    if (k < 1 || k > 64 || rank < 1 || !row_off) return (int)cudaErrorInvalidValue;
    const size_t smem = (size_t)(rank + k) * sizeof(float);
    walk_kernel<<<n_seq, WT, smem, gb_stream(stream)>>>((const int*)anchor, (const int*)topk_ids, (const float*)topk_vals, gate, gate_is_f32,
                                                        (const __half*)pred_codebook, (const __half*)succ_codebook, (int*)out_tokens,
                                                        (float*)conf, 0, k, rank, (const WalkSampling*)sampling, (const int*)row_off, (float*)margin);
    return GB_CHECK_LAUNCH();
}

// Map top-k ids from a restricted candidate set back to token ids. The set is up to 4 ranges of token ids laid end to end:
// compact id c in [cum[r], cum[r + 1]) is token base[r] + (c - cum[r]). Ranges ascend, so top-k's tie order (lowest id first) is the same
// in both numberings.
namespace {
struct IdRanges {
    int n;
    int cum[5];   // cum[0] = 0, cum[r + 1] = ids in ranges 0..=r
    int base[4];  // token id of the first id of range r
};

__global__ void remap_ids_kernel(int* __restrict__ ids, int count, IdRanges rg) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int c = ids[i];
    int r = 0;
    while (r + 1 < rg.n && c >= rg.cum[r + 1]) ++r;
    ids[i] = rg.base[r] + (c - rg.cum[r]);
}
}  // namespace

// ids: i32 [count], in place. cum has n + 1 entries (n <= 4), base n entries.
GB_EXPORT int gb_dflash_remap_ids(void* ids, int count, int n_ranges, const int* cum, const int* base, void* stream) {
    if (count <= 0) return 0;
    if (n_ranges < 1 || n_ranges > 4 || !cum || !base || !ids) return (int)cudaErrorInvalidValue;
    IdRanges rg;
    rg.n = n_ranges;
    for (int r = 0; r <= n_ranges; ++r) rg.cum[r] = cum[r];
    for (int r = 0; r < n_ranges; ++r) rg.base[r] = base[r];
    remap_ids_kernel<<<(count + 127) / 128, 128, 0, gb_stream(stream)>>>((int*)ids, count, rg);
    return GB_CHECK_LAUNCH();
}

// The top-k of a split draft head: each rank of a two-rank group has the top k (ids, values) of its own columns; per row, the k largest of
// the 2k candidates by (value DESC, id ASC), the full head's top-k order (glue topk). The halves' ids are disjoint, so the result does not
// depend on which list is `a`: both ranks get the same lists from (own, peer). One warp a row, lane j holding candidate j (2k <= 32).
namespace {
__device__ __forceinline__ uint64_t merge_key(float v, int id) {
    const uint32_t u = v == 0.f ? 0u : __float_as_uint(v);                            // -0 ties with +0, as in the head's top-k
    const uint32_t o = (v != v) ? 0u : (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // NaN lowest, as the head's top-k reports it
    return ((uint64_t)o << 32) | (uint64_t)(0xFFFFFFFFu - (uint32_t)id);
}

__global__ void merge_topk_kernel(const int* ia, const float* va, const int* __restrict__ ib, const float* __restrict__ vb, int* io, float* vo,
                                  int rows, int k) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= rows) return;
    int id = -1;
    float v = 0.f;
    if (lane < k) {
        id = ia[(size_t)row * k + lane];
        v = va[(size_t)row * k + lane];
    } else if (lane < 2 * k) {
        id = ib[(size_t)row * k + lane - k];
        v = vb[(size_t)row * k + lane - k];
    }
    uint64_t key = lane < 2 * k ? merge_key(v, id) : 0ull;
    __syncwarp();                                  // every input read before the first write (io / vo may alias ia / va)
    for (int j = 0; j < k; ++j) {
        uint64_t best = key;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            const uint64_t o = ((uint64_t)__shfl_xor_sync(0xffffffffu, (uint32_t)(best >> 32), m) << 32) |
                               (uint64_t)__shfl_xor_sync(0xffffffffu, (uint32_t)best, m);
            best = o > best ? o : best;
        }
        if (key == best && lane < 2 * k) {          // the one holder (keys are distinct: disjoint ids)
            io[(size_t)row * k + j] = id;
            vo[(size_t)row * k + j] = v;
            key = 0ull;
        }
    }
}
}  // namespace

// ids / vals: i32 / f32 [rows, k] each (a: this rank's, b: the peer's); the merged top-k into io / vo (may be a's buffers). k <= 16.
GB_EXPORT int gb_dflash_merge_topk(const void* ia, const void* va, const void* ib, const void* vb, void* io, void* vo, int rows, int k,
                                   void* stream) {
    if (rows <= 0) return 0;
    if (k < 1 || k > 16 || !ia || !va || !ib || !vb || !io || !vo) return (int)cudaErrorInvalidValue;
    merge_topk_kernel<<<(rows + 3) / 4, 128, 0, gb_stream(stream)>>>((const int*)ia, (const float*)va, (const int*)ib, (const float*)vb,
                                                                      (int*)io, (float*)vo, rows, k);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dflash_selector_walk(const void* anchor, const void* topk_ids, const void* topk_vals, const void* gate, int gate_is_f32,
                                      const void* pred_codebook, const void* succ_codebook, int n_seq, int rows, int k, int rank,
                                      void* out_tokens, void* conf, void* stream) {
    return gb_dflash_selector_walk_keyed(anchor, topk_ids, topk_vals, gate, gate_is_f32, pred_codebook, succ_codebook, n_seq, rows, k, rank,
                                         nullptr, out_tokens, conf, stream);
}

// =================================================================================================================
// 5. Selector lattice edges (tree drafting). The chain walk scores a candidate c of row i given the previous pick `pred` as
//    unary[c] + dot(A[pred] * gate_i, B[c]); the dot depends on the previous pick only through A[pred]. A tree search needs it
//    for every parent a node can have: the anchor for the first row, the 16 candidates of row i - 1 for row i. This kernel
//    computes them with the walk kernel's own arithmetic (the same A * gate product, the same fmaf lane loop and butterfly), so
//    the chain through the lattice reproduces the walk's scores bit for bit (unary + edge[parent][c]).
// =================================================================================================================
namespace {
__global__ void __launch_bounds__(WT)
    edge_kernel(const int* __restrict__ anchor, const int* __restrict__ ids, const void* gate, int gate_f32,
                const __half* __restrict__ pred_cb, const __half* __restrict__ succ_cb, int n_seq, int k, int rank,
                const int* __restrict__ row_off, float* __restrict__ edge) {
    extern __shared__ float smem[];
    float* a_g = smem;  // [rank]
    const int g = blockIdx.x, p = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    int b = 0;
    while (b + 1 < n_seq && row_off[b + 1] <= g) ++b;
    const int i = g - row_off[b];
    float* out = edge + ((size_t)g * k + p) * k;
    if (i == 0 && p > 0) {  // the first row has one parent, the anchor
        for (int c = t; c < k; c += WT) out[c] = 0.0f;
        return;
    }
    const int parent = i == 0 ? anchor[b] : ids[(size_t)(g - 1) * k + p];
    const __half* a_row = pred_cb + (size_t)parent * rank;
    for (int r = t; r < rank; r += WT) a_g[r] = __half2float(a_row[r]) * gate_at(gate, gate_f32 != 0, (size_t)g * rank + r);
    __syncthreads();
    const int* c_row = ids + (size_t)g * k;
    for (int c = warp; c < k; c += WT / 32) {
        const __half* b_row = succ_cb + (size_t)c_row[c] * rank;
        float dot = 0.f;
        for (int r = lane; r < rank; r += 32) dot = __fmaf_rn(a_g[r], __half2float(b_row[r]), dot);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if (lane == 0) out[c] = dot;
    }
}
}  // namespace

// anchor: i32 [n_seq]; topk_ids: i32 [total_rows, k] (the rows of every sequence packed, `row_off` i32 [n_seq + 1] as in the ragged walk);
// gate: bf16 or f32 [total_rows, rank]; codebooks f16 [V, rank]; edge: f32 [total_rows, k parents, k candidates]. edge[row][p][c] is the
// dot term of candidate c of the row when the previous pick is the row's parent p (the anchor, p = 0, for a sequence's first row; the
// candidates of the row before it otherwise). Rows past the first of a sequence fill every p; a first row fills p = 0 and zeroes the rest.
GB_EXPORT int gb_dflash_selector_edges(const void* anchor, const void* topk_ids, const void* gate, int gate_is_f32, const void* pred_codebook,
                                       const void* succ_codebook, int n_seq, int total_rows, int k, int rank, const void* row_off, void* edge,
                                       void* stream) {
    if (n_seq <= 0 || total_rows <= 0) return 0;
    if (k < 1 || k > 64 || rank < 1 || !row_off) return (int)cudaErrorInvalidValue;
    const size_t smem = (size_t)rank * sizeof(float);
    edge_kernel<<<dim3(total_rows, k), WT, smem, gb_stream(stream)>>>((const int*)anchor, (const int*)topk_ids, gate, gate_is_f32,
                                                                    (const __half*)pred_codebook, (const __half*)succ_codebook, n_seq, k, rank,
                                                                    (const int*)row_off, (float*)edge);
    return GB_CHECK_LAUNCH();
}
