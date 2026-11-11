// gb-nvcc: --fmad=false
// QSA: the sparse-attention indexer of Qwen3.8-Flash-Next full-attention layers. Original REXL3 code, Victor Cruz.
//
// Every token is projected (by an EXL3 linear, outside) to 4 indexer query heads and one raw key of 128. Each complete
// 4-token block of raw keys is mean-pooled, RMS-normed and roped at the block's start. A query at position p keeps the
// 512 complete blocks (of the (p + 1) / 4 it can see) with the highest score sum_h relu(q_h . k_block) / sqrt(128), plus
// the incomplete block it sits in; it attends to those tokens only. Up to position 2050 that is every block, so dense
// attention is the same function there.
//
//   gb_qsa_stage   qk bf16 [rows][640] -> q f16 [rows][4][128] (RMS norm * w1, rope at the row position) and the raw
//                  keys f16 [rows][128]
//   gb_qsa_pool    each row whose position ends a block pools the block's 4 raw keys (rows of this batch, or the slot's
//                  raw ring for earlier positions) into the paged pool plane: f16 mean, RMS norm * w1, rope at the start
//   gb_qsa_ring    the last `ring` raw keys of each sequence into its slot's ring (indexed by position mod ring)
//   gb_qsa_scores  scores f32 [rows][nb_cap] of every block a row can see (other entries untouched)
//   gb_qsa_topk    per row: the selected blocks in ascending order, and their count
//   gb_qsa_attend  attention of a row over its selected blocks plus its tail block, in `parts` partial sums
//   gb_qsa_merge   combine the partial sums into bf16 [rows][24][256]
//
// The pool plane is paged like the KV cache: page `table[slot][4 j / 256]` holds blocks with j / 64 equal, at j % 64.
// Rows of one sequence are contiguous with contiguous positions (seq_start [n_seq + 1], seq_slot [n_seq]).
#include "common.cuh"
#include "pdl.cuh"
#include <cuda_bf16.h>

// Decode launches (gbp_pdl::kFlash): a kernel launched with the programmatic-dependent-launch attribute may start before its predecessor
// finishes, so it calls pdl_wait() -- first in every thread, early exits included -- before it reads anything an earlier kernel wrote or
// writes anything an earlier kernel may still read; what depends on no earlier kernel (weights, the rope frequencies) is read or computed
// before it. A launch without the attribute makes both calls no-ops (pdl.cuh), so only timing changes, never a bit.
using gbp_pdl::pdl_trigger;
using gbp_pdl::pdl_wait;
inline bool flash_pdl() { return gbp_pdl::on(gbp_pdl::kFlash); }

namespace qsa {

constexpr int DK = 128;        // indexer head dim
constexpr int NH = 4;          // indexer query heads
constexpr int QK = (NH + 1) * DK;
constexpr int CR = 4;          // tokens per block
constexpr int PAGE = 256;      // tokens per KV page
constexpr int PB = PAGE / CR;  // blocks per page
constexpr int TOPK = 512;      // blocks kept per query
constexpr int RD = 64;         // rotary dims (pairs d, d + 32)
constexpr int D = 256, NQ = 24, NKV = 2, GRP = 12;  // main attention
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }
__device__ __forceinline__ float rh(float v) { return __half2float(__float2half_rn(v)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ int seq_of(const int* seq_start, int n_seq, int r) {
    int s = 0;
    while (s + 1 < n_seq && seq_start[s + 1] <= r) ++s;
    return s;
}

// inverse frequency of rope pair i (of RD / 2), as the main attention computes it
__device__ __forceinline__ float inv_freq(float theta, int i) {
    return (float)pow((double)theta, -(double)(2 * i) / (double)RD);
}

// rope one 128-dim vector held one element per thread (d = threadIdx.x) through `sh`; result rounded to f16. `invf` is inv_freq(theta, d % 32)
// for d < RD (the caller computes it: it depends on nothing a predecessor wrote), unused otherwise.
__device__ __forceinline__ float rope_elem(float x, int d, int pos, float invf, float* sh) {
    sh[d] = x;
    __syncthreads();
    float o = x;
    if (d < RD) {
        const float ang = (float)pos * invf;
        const float c = cosf(ang), s = sinf(ang);
        o = d < RD / 2 ? x * c - sh[d + RD / 2] * s : x * c + sh[d - RD / 2] * s;
    }
    __syncthreads();
    return rh(o);
}

__device__ __forceinline__ float block_sum128(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    const float t = red[0] + red[1] + red[2] + red[3];
    __syncthreads();
    return t;
}

// grid rows, block 160: warps 0..3 the query heads, warp 4 the raw key
__global__ void __launch_bounds__(160) stage_k(const uint16_t* __restrict__ qk, const float* __restrict__ qw,
                                               const int* __restrict__ positions, __half* __restrict__ q,
                                               __half* __restrict__ kraw, float eps, float theta) {
    pdl_trigger();  // pool_k may launch: it waits for this grid before it reads kraw
    const int r = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    // before the wait: this lane's rope frequency (a double-precision pow, the slowest part of the kernel) and the norm weights
    float invf = 0.f, wq[4] = {0.f, 0.f, 0.f, 0.f};
    if (w < NH) {
        invf = inv_freq(theta, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) wq[j] = qw[lane + 32 * j];
    }
    pdl_wait();  // qk and positions are earlier kernels' outputs; q and kraw may still be read by earlier kernels
    const uint16_t* src = qk + (size_t)r * QK + w * DK;
    if (w == NH) {
        for (int d = lane; d < DK; d += 32) kraw[(size_t)r * DK + d] = __float2half_rn(bf(src[d]));
        return;
    }
    float x[4];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        x[j] = bf(src[lane + 32 * j]);
        ss = ss + x[j] * x[j];
    }
    ss = warp_sum(ss);
    const float inv = 1.0f / sqrtf(ss / (float)DK + eps);
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = rh((x[j] * inv) * wq[j]);
    // rope pair (lane, lane + 32) sits in this lane as x[0], x[1]
    const float ang = (float)positions[r] * invf;
    const float c = cosf(ang), s = sinf(ang);
    const float a = x[0], b = x[1];
    x[0] = rh(a * c - b * s);
    x[1] = rh(b * c + a * s);
    __half* dst = q + ((size_t)r * NH + w) * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j) dst[lane + 32 * j] = __float2half_rn(x[j]);
}

// grid rows, block 128 (one thread per dim)
__global__ void __launch_bounds__(DK) pool_k(const __half* __restrict__ kraw, const __half* __restrict__ ring,
                                             const float* __restrict__ kw, const int* __restrict__ positions,
                                             const int* __restrict__ seq_start, const int* __restrict__ seq_slot,
                                             int n_seq, const int* __restrict__ table, int max_blocks,
                                             __half* __restrict__ pool, int ring_len, float eps, float theta) {
    __shared__ float red[4], sh[DK];
    pdl_trigger();  // ring_k may launch: it waits for this grid before it reads kraw or writes the ring this kernel reads
    const int r = blockIdx.x, d = threadIdx.x;
    // before the wait: this thread's norm weight and rope frequency, which no predecessor writes
    const float kwd = kw[d];
    const float invf = d < RD ? inv_freq(theta, d % (RD / 2)) : 0.f;
    pdl_wait();  // every early exit below is after it
    const int p = positions[r];
    if (p % CR != CR - 1) return;
    const int sq = seq_of(seq_start, n_seq, r);
    const int p0 = positions[seq_start[sq]], slot = seq_slot[sq];
    float sum = 0.f;
#pragma unroll
    for (int t = 0; t < CR; ++t) {
        const int q = p - (CR - 1) + t;
        const float v = q >= p0 ? __half2float(kraw[(size_t)(r - (p - q)) * DK + d])
                                : __half2float(ring[((size_t)slot * ring_len + q % ring_len) * DK + d]);
        sum = sum + v;
    }
    const float m = rh(sum * 0.25f);
    const float ss = block_sum128(m * m, red);
    const float inv = 1.0f / sqrtf(ss / (float)DK + eps);
    const float y = rh((m * inv) * kwd);
    const int j = p / CR;
    const float o = rope_elem(y, d, j * CR, invf, sh);
    const int page = table[(size_t)slot * max_blocks + j / PB];
    pool[((size_t)page * PB + j % PB) * DK + d] = __float2half_rn(o);
}

__global__ void __launch_bounds__(DK) ring_k(const __half* __restrict__ kraw, __half* __restrict__ ring,
                                             const int* __restrict__ positions, const int* __restrict__ seq_start,
                                             const int* __restrict__ seq_slot, int n_seq, int ring_len) {
    pdl_trigger();  // the successor may launch: it waits for this grid before it reads or writes anything
    pdl_wait();     // kraw is stage_k's output, and the ring is read by pool_k
    const int r = blockIdx.x, d = threadIdx.x;
    const int sq = seq_of(seq_start, n_seq, r);
    const int p_last = positions[seq_start[sq + 1] - 1], p = positions[r];
    if (p <= p_last - ring_len) return;
    ring[((size_t)seq_slot[sq] * ring_len + p % ring_len) * DK + d] = kraw[(size_t)r * DK + d];
}

constexpr int RG = 16;  // rows a scores CTA serves

// grid (ceil(nb_cap / 256), ceil(rows / RG)), block 256: one thread per block j, for every row of the group. Rows of one
// sequence are adjacent, so a thread reads its key once per run of rows with the same slot.
// Both operands are f16, so a product is exact in f32 and `dot + q * k` rounded once (an fma) is the same number as the product rounded
// and then added: one instruction a term instead of two, the chain over the 128 dims in the same order. The queries are read four dims
// (16 bytes) at a time, one load for four terms instead of one for each.
__global__ void __launch_bounds__(256) scores_k(const __half* __restrict__ q, const __half* __restrict__ pool,
                                                const int* __restrict__ table, int max_blocks,
                                                const int* __restrict__ row_slot, const int* __restrict__ positions,
                                                float* __restrict__ scores, int rows, int nb_cap, float scale) {
    __shared__ __align__(16) float qs[RG][NH * DK];
    __shared__ int nbq_s[RG], slot_s[RG];
    __shared__ int nbq_max;
    pdl_trigger();  // the successor may launch: it waits for this grid before it reads or writes anything
    pdl_wait();     // q, the pool plane, the table and the positions are earlier kernels' outputs; every early exit is after it
    const int r0 = blockIdx.y * RG, rn = min(RG, rows - r0);
    if (threadIdx.x < RG) {
        const int r = r0 + min((int)threadIdx.x, rn - 1);
        nbq_s[threadIdx.x] = (positions[r] + 1) / CR;
        slot_s[threadIdx.x] = row_slot[r];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int mx = 0;
        for (int k = 0; k < rn; ++k) mx = max(mx, nbq_s[k]);
        nbq_max = mx;
    }
    __syncthreads();
    const int j0 = blockIdx.x * 256;
    if (j0 >= nbq_max) return;
    for (int i = threadIdx.x; i < rn * NH * DK; i += 256) qs[i / (NH * DK)][i % (NH * DK)] = __half2float(q[(size_t)r0 * NH * DK + i]);
    __syncthreads();
    const int j = j0 + threadIdx.x;
    if (j >= nbq_max) return;
    __half2 key[DK / 2];
    int key_slot = -1;
    for (int k = 0; k < rn; ++k) {
        if (j >= nbq_s[k]) continue;
        if (slot_s[k] != key_slot) {
            key_slot = slot_s[k];
            const int page = table[(size_t)key_slot * max_blocks + j / PB];
            const uint4* kp = reinterpret_cast<const uint4*>(pool + ((size_t)page * PB + j % PB) * DK);
#pragma unroll
            for (int c = 0; c < DK / 8; ++c) {
                const uint4 u = kp[c];
                const __half2* h2 = reinterpret_cast<const __half2*>(&u);
#pragma unroll
                for (int e = 0; e < 4; ++e) key[c * 4 + e] = h2[e];
            }
        }
        float dot[NH] = {0.f, 0.f, 0.f, 0.f};
        const float4* qk = reinterpret_cast<const float4*>(qs[k]);
#pragma unroll
        for (int c = 0; c < DK / 4; ++c) {
            const float2 k0 = __half22float2(key[2 * c]), k1 = __half22float2(key[2 * c + 1]);
#pragma unroll
            for (int h = 0; h < NH; ++h) {
                const float4 q4 = qk[h * (DK / 4) + c];
                dot[h] = __fmaf_rn(q4.x, k0.x, dot[h]);
                dot[h] = __fmaf_rn(q4.y, k0.y, dot[h]);
                dot[h] = __fmaf_rn(q4.z, k1.x, dot[h]);
                dot[h] = __fmaf_rn(q4.w, k1.y, dot[h]);
            }
        }
        float sc = 0.f;
#pragma unroll
        for (int h = 0; h < NH; ++h) sc = sc + fmaxf(dot[h], 0.f);
        scores[(size_t)(r0 + k) * nb_cap + j] = sc * scale;
    }
}

// The previous scores kernel (`dot + q * k + q * k` as separately rounded operations, the queries read a float at a time): kept as the
// oracle of scores_k's test.
__global__ void __launch_bounds__(256) scores_ref_k(const __half* __restrict__ q, const __half* __restrict__ pool,
                                                const int* __restrict__ table, int max_blocks,
                                                const int* __restrict__ row_slot, const int* __restrict__ positions,
                                                float* __restrict__ scores, int rows, int nb_cap, float scale) {
    __shared__ float qs[RG][NH * DK];
    __shared__ int nbq_s[RG], slot_s[RG];
    __shared__ int nbq_max;
    pdl_trigger();  // the successor may launch: it waits for this grid before it reads or writes anything
    pdl_wait();     // q, the pool plane, the table and the positions are earlier kernels' outputs; every early exit is after it
    const int r0 = blockIdx.y * RG, rn = min(RG, rows - r0);
    if (threadIdx.x < RG) {
        const int r = r0 + min((int)threadIdx.x, rn - 1);
        nbq_s[threadIdx.x] = (positions[r] + 1) / CR;
        slot_s[threadIdx.x] = row_slot[r];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int mx = 0;
        for (int k = 0; k < rn; ++k) mx = max(mx, nbq_s[k]);
        nbq_max = mx;
    }
    __syncthreads();
    const int j0 = blockIdx.x * 256;
    if (j0 >= nbq_max) return;
    for (int i = threadIdx.x; i < rn * NH * DK; i += 256) qs[i / (NH * DK)][i % (NH * DK)] = __half2float(q[(size_t)r0 * NH * DK + i]);
    __syncthreads();
    const int j = j0 + threadIdx.x;
    if (j >= nbq_max) return;
    __half2 key[DK / 2];
    int key_slot = -1;
    for (int k = 0; k < rn; ++k) {
        if (j >= nbq_s[k]) continue;
        if (slot_s[k] != key_slot) {
            key_slot = slot_s[k];
            const int page = table[(size_t)key_slot * max_blocks + j / PB];
            const uint4* kp = reinterpret_cast<const uint4*>(pool + ((size_t)page * PB + j % PB) * DK);
#pragma unroll
            for (int c = 0; c < DK / 8; ++c) {
                const uint4 u = kp[c];
                const __half2* h2 = reinterpret_cast<const __half2*>(&u);
#pragma unroll
                for (int e = 0; e < 4; ++e) key[c * 4 + e] = h2[e];
            }
        }
        float dot[NH] = {0.f, 0.f, 0.f, 0.f};
        const float* qk = qs[k];
#pragma unroll
        for (int c = 0; c < DK / 2; ++c) {
            const float2 kf = __half22float2(key[c]);
#pragma unroll
            for (int h = 0; h < NH; ++h) dot[h] = dot[h] + qk[h * DK + 2 * c] * kf.x + qk[h * DK + 2 * c + 1] * kf.y;
        }
        float sc = 0.f;
#pragma unroll
        for (int h = 0; h < NH; ++h) sc = sc + fmaxf(dot[h], 0.f);
        scores[(size_t)(r0 + k) * nb_cap + j] = sc * scale;
    }
}

// grid rows, block 1024. Scores are >= 0, so their bit patterns order like the values. The selection is the radix select of the TOPK-th
// largest bit pattern (8 bits a pass from the top, one block-wide histogram a pass, its bins scanned by one warp) followed by one
// ordered pass that takes every greater pattern and the first `need` equal ones, in index order; the warps' counts of that pass are
// combined by every warp itself (a shuffle scan over the 32 warps' counts), so a pass of 1,024 elements costs one barrier.
__global__ void __launch_bounds__(1024) topk_k(const float* __restrict__ scores, const int* __restrict__ positions,
                                               int nb_cap, int* __restrict__ sel, int* __restrict__ sel_n) {
    __shared__ __align__(16) unsigned hist[4][256];
    __shared__ unsigned sh_prefix, sh_need;
    __shared__ int wgt[2][32], weq[2][32];
    pdl_trigger();
    pdl_wait();  // scores and positions are earlier kernels' outputs; the early exit below is after it
    const int r = blockIdx.x, t = threadIdx.x;
    const int nbq = (positions[r] + 1) / CR;
    int* out = sel + (size_t)r * TOPK;
    if (nbq <= TOPK) {
        for (int i = t; i < nbq; i += blockDim.x) out[i] = i;
        if (t == 0) sel_n[r] = nbq;
        return;
    }
    const unsigned* s = reinterpret_cast<const unsigned*>(scores + (size_t)r * nb_cap);
    const int wid = t >> 5, ln = t & 31;
    (&hist[0][0])[t] = 0;
    __syncthreads();
    // radix select of the TOPK-th largest bit pattern: `prefix` is its decided high bits, `need` the rank wanted among the patterns that
    // share them
    unsigned prefix = 0, need = TOPK;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        const unsigned hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        // scores cluster in a few bins: lanes with the same digit add once
        const int span = (nbq + 31) / 32 * 32;
        for (int i = t; i < span; i += 1024) {
            const unsigned v = i < nbq ? s[i] : 0u;
            const bool in = i < nbq && (v & hi_mask) == prefix;
            const unsigned dgt = in ? ((v >> shift) & 255u) : 256u;
            const unsigned peers = __match_any_sync(0xffffffffu, dgt);
            if (in && ln == __ffs(peers) - 1) atomicAdd(&hist[pass][dgt], (unsigned)__popc(peers));
        }
        __syncthreads();
        if (wid == 0) {
            // lane l owns bins 8 l .. 8 l + 7; S is the count of the bins from 8 l up
            const uint4 h0 = reinterpret_cast<const uint4*>(&hist[pass][8 * ln])[0];
            const uint4 h1 = reinterpret_cast<const uint4*>(&hist[pass][8 * ln])[1];
            const unsigned h[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
            unsigned ls = 0;
#pragma unroll
            for (int b = 0; b < 8; ++b) ls += h[b];
            unsigned S = ls;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned up = __shfl_down_sync(0xffffffffu, S, o);
                if (ln + o < 32) S += up;
            }
            // the answer is the highest bin whose count from there up reaches `need`: it lies in the highest lane whose S does
            const unsigned ge = __ballot_sync(0xffffffffu, S >= need);
            const int lane_star = ge ? 31 - __clz(ge) : 0;
            if (ln == lane_star) {
                unsigned acc = S - ls;  // the bins above this lane
                int dgt = 8 * ln;
                bool found = false;
#pragma unroll
                for (int b = 7; b > 0; --b) {
                    if (found) continue;
                    if (acc + h[b] >= need) {
                        dgt = 8 * ln + b;
                        found = true;
                    } else {
                        acc += h[b];
                    }
                }
                sh_prefix = prefix | ((unsigned)dgt << shift);
                sh_need = need - acc;
            }
        }
        __syncthreads();
        prefix = sh_prefix;
        need = sh_need;
    }
    // prefix is the threshold pattern; take every greater one and the first `need` equal ones, in index order
    const unsigned lt = (1u << ln) - 1u;
    int base_gt = 0, base_eq = 0;  // taken so far, equal so far (every thread tracks them: the totals are the same in every warp)
    int buf = 0;
    for (int i0 = 0; i0 < nbq; i0 += 1024, buf ^= 1) {
        const int i = i0 + t;
        const unsigned v = i < nbq ? s[i] : 0u;
        const bool gt = i < nbq && v > prefix, eq = i < nbq && v == prefix;
        const unsigned bg = __ballot_sync(0xffffffffu, gt), be = __ballot_sync(0xffffffffu, eq);
        if (ln == 0) {
            wgt[buf][wid] = __popc(bg);
            weq[buf][wid] = __popc(be);
        }
        __syncthreads();
        // lane k now stands for warp k: its greater and equal counts, the equal ones before it, and what it takes
        const int cg = wgt[buf][ln], ce = weq[buf][ln];
        int ie = ce;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int up = __shfl_up_sync(0xffffffffu, ie, o);
            if (ln >= o) ie += up;
        }
        const int ex_eq = ie - ce;
        const int room = (int)need - (base_eq + ex_eq);
        const int tk = cg + min(max(room, 0), ce);
        int it = tk;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int up = __shfl_up_sync(0xffffffffu, it, o);
            if (ln >= o) it += up;
        }
        const int eq_before = base_eq + __shfl_sync(0xffffffffu, ex_eq, wid) + __popc(be & lt);
        const bool take = gt || (eq && eq_before < (int)need);
        const unsigned bt = __ballot_sync(0xffffffffu, take);
        const int at = base_gt + __shfl_sync(0xffffffffu, it - tk, wid) + __popc(bt & lt);
        if (take) out[at] = i;
        base_gt += __shfl_sync(0xffffffffu, it, 31);
        base_eq += __shfl_sync(0xffffffffu, ie, 31);
    }
    if (t == 0) sel_n[r] = TOPK;
}

constexpr int TB = 8;  // tokens a warp works on at once. Part of the arithmetic: the running max moves once per TB tokens
constexpr int ATTEND_HPW = 4;           // query heads a warp of attend_h serves (MEASURED at 4,096 rows: 90.9 ms -> 50.2 ms a layer)
constexpr int ATTEND_H_MIN_ROWS = 64;  // rows from which gb_qsa_attend runs attend_h

// grid (rows, NKV, parts), block 32 * GRP: warp w is q head kvh * GRP + w
__global__ void __launch_bounds__(32 * GRP) attend_k(const __nv_bfloat16* __restrict__ qm, const __nv_bfloat16* __restrict__ kc,
                                                     const __nv_bfloat16* __restrict__ vc, const int* __restrict__ table,
                                                     int max_blocks, const int* __restrict__ row_slot,
                                                     const int* __restrict__ positions, const int* __restrict__ sel,
                                                     const int* __restrict__ sel_n, float* __restrict__ part_o,
                                                     float* __restrict__ part_m, float* __restrict__ part_l, int parts) {
    pdl_trigger();
    pdl_wait();  // q, the selection and the positions are earlier kernels' outputs
    const int r = blockIdx.x, kvh = blockIdx.y, part = blockIdx.z;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head = kvh * GRP + w;
    const int p = positions[r], nbq = (p + 1) / CR, n = sel_n[r];
    const int total = CR * n + (p - CR * nbq + 1);
    const int a = (int)((long long)total * part / parts), b = (int)((long long)total * (part + 1) / parts);
    const int slot = row_slot[r];
    const int* sl = sel + (size_t)r * TOPK;
    const int* tb = table + (size_t)slot * max_blocks;
    // q scaled by 1/16 * log2(e), 8 dims a lane
    float qv[8];
    {
        const uint4 u = reinterpret_cast<const uint4*>(qm + ((size_t)r * NQ + head) * D)[lane];
        const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 f = __bfloat1622float2(h2[e]);
            qv[2 * e] = f.x * (0.0625f * LOG2E);
            qv[2 * e + 1] = f.y * (0.0625f * LOG2E);
        }
    }
    float m = -INFINITY, l = 0.f, o[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) o[e] = 0.f;
    for (int i0 = a; i0 < b; i0 += TB) {
        int tok[TB];
        float sc[TB];
        uint4 ku[TB];
        uint4 vu[TB];
#pragma unroll
        for (int k = 0; k < TB; ++k) {
            const int i = i0 + k;
            const int ii = i < b ? i : a;
            tok[k] = ii < CR * n ? CR * sl[ii / CR] + ii % CR : CR * nbq + (ii - CR * n);
            const size_t off = (((size_t)tb[tok[k] / PAGE] * PAGE + tok[k] % PAGE) * NKV + kvh) * D;
            ku[k] = reinterpret_cast<const uint4*>(kc + off)[lane];
            vu[k] = reinterpret_cast<const uint4*>(vc + off)[lane];
        }
#pragma unroll
        for (int k = 0; k < TB; ++k) {
            const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&ku[k]);
            float d = 0.f;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 f = __bfloat1622float2(h2[e]);
                d = d + qv[2 * e] * f.x + qv[2 * e + 1] * f.y;
            }
            sc[k] = warp_sum(d);
        }
        float mx = m;
#pragma unroll
        for (int k = 0; k < TB; ++k)
            if (i0 + k < b) mx = fmaxf(mx, sc[k]);
        const float corr = exp2f(m - mx);
        l = l * corr;
#pragma unroll
        for (int e = 0; e < 8; ++e) o[e] = o[e] * corr;
#pragma unroll
        for (int k = 0; k < TB; ++k) {
            if (i0 + k >= b) continue;
            const float pk = exp2f(sc[k] - mx);
            l = l + pk;
            const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&vu[k]);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 f = __bfloat1622float2(h2[e]);
                o[2 * e] = o[2 * e] + pk * f.x;
                o[2 * e + 1] = o[2 * e + 1] + pk * f.y;
            }
        }
        m = mx;
    }
    const size_t pi = ((size_t)r * NQ + head) * parts + part;
    float* po = part_o + pi * D + lane * 8;
#pragma unroll
    for (int e = 0; e < 8; ++e) po[e] = o[e];
    if (lane == 0) {
        part_m[pi] = m;
        part_l[pi] = l;
    }
}

// attend_k with HPW query heads a warp (block 32 * GRP / HPW). Every head's own arithmetic is attend_k's, operation for operation with
// the same operands (this file is --fmad=false), so a head's part_o / part_m / part_l are attend_k's bit for bit; what changes is how
// many instructions a (token, head) costs, by regrouping work that is the same value in every lane:
//  * The 8-token group's K and V rows are gathered through offsets computed lane-parallel, 32 tokens at a time (one lane a token), and
//    widened once for the warp's four heads.
//  * A token's score is warp_sum of the lanes' partial dots: butterfly over lane bits 4, 3, 2, 1, 0 in that order. The eight tokens of
//    a group (and the four heads) are reduced together instead of one warp_sum each: at each step a lane forwards the half of its
//    values its partner keeps and adds the half it keeps, so the additions and their grouping are warp_sum's exactly (a sum's operands
//    commute) and a step costs half the shuffles and adds of the step before. To make "the half the partner keeps" the same registers
//    in both lanes without selects, lane L loads token (k ^ (L >> 2)) as its k-th K row and takes the query of head (s ^ (L & 3)) as
//    its s-th query. After the five steps lane L holds the score of token L >> 2, head L & 3, as warp_sum would give every lane.
//  * That lane alone takes the group's max of its head (max is exact: a butterfly of fmaxf over lane bits 4, 3, 2) and computes
//    pk = exp2f(score - max), which attend_k computes in every lane from the same two numbers; pk goes to the other lanes through
//    shared memory. Lane L also keeps the running max m and the sum l of its own head L & 3 (l's additions are attend_k's, in token
//    order); only the o accumulators are per head in every lane.
//  * The rescale of l and o by exp2f(m - mx) only matters when the max moved; o * 1.0f is o and l * 1.0f is l, so the multiplies are
//    skipped (warp-uniformly) when no head's max changed.
//  * `d = 0.f + q * k` is one fma (see dot8) and a bf16 widens in one instruction (see widen8).
// Prefill chunks run one part: grid (rows, NKV, parts) as attend_k, block 96.
constexpr int HPW = ATTEND_HPW;
static_assert(GRP % HPW == 0 && HPW == 4 && TB == 8, "the lane layout of attend_h is built for 8 tokens and 4 heads a warp");

// bf16 pairs to f32 (a bf16 is the high half of an f32): one instruction an element
__device__ __forceinline__ void widen8(const uint4& u, float (&f)[8]) {
    f[0] = __uint_as_float(u.x << 16);
    f[1] = __uint_as_float(u.x & 0xffff0000u);
    f[2] = __uint_as_float(u.y << 16);
    f[3] = __uint_as_float(u.y & 0xffff0000u);
    f[4] = __uint_as_float(u.z << 16);
    f[5] = __uint_as_float(u.z & 0xffff0000u);
    f[6] = __uint_as_float(u.w << 16);
    f[7] = __uint_as_float(u.w & 0xffff0000u);
}

// attend_k's `d = 0.f; for e: d = d + q[2e] * k[2e] + q[2e + 1] * k[2e + 1]`. Its first term is 0.f + q[0] * k[0]: the product rounded
// once, then a zero added (which only turns a -0 product into +0): exactly fma(q[0], k[0], 0.f), one instruction instead of two.
__device__ __forceinline__ float dot8(const float (&q)[8], const float (&k)[8]) {
    float d = __fmaf_rn(q[0], k[0], 0.f);
    d = d + q[1] * k[1];
#pragma unroll
    for (int e = 1; e < 4; ++e) d = d + q[2 * e] * k[2 * e] + q[2 * e + 1] * k[2 * e + 1];
    return d;
}

// Scores of one group of TB tokens: `ku[k]` is the lane's K row of token (k ^ g) (g = lane >> 2, qs = lane & 3), qv[s] the query of head
// (s ^ qs). Returns the score of token g, head qs: warp_sum's value of that (token, head) pair, from the tree described above.
__device__ __forceinline__ float group_scores(const uint4 (&ku)[TB], const float (&qv)[HPW][8]) {
    constexpr unsigned FULLMASK = 0xffffffffu;
    // partial dots of this lane's 8 dims, then step 16 of the butterfly (tokens k and k + 4 meet in w[s][k])
    float w[HPW][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float k0[8], k1[8];
        widen8(ku[j], k0);
        widen8(ku[j + 4], k1);
#pragma unroll
        for (int s = 0; s < HPW; ++s) {
            const float d0 = dot8(qv[s], k0), d1 = dot8(qv[s], k1);
            w[s][j] = d0 + __shfl_xor_sync(FULLMASK, d1, 16);
        }
    }
    // steps 8 and 4 over the tokens, then steps 2 and 1 over the heads
    float t[HPW];
#pragma unroll
    for (int s = 0; s < HPW; ++s) {
        const float u0 = w[s][0] + __shfl_xor_sync(FULLMASK, w[s][2], 8);
        const float u1 = w[s][1] + __shfl_xor_sync(FULLMASK, w[s][3], 8);
        t[s] = u0 + __shfl_xor_sync(FULLMASK, u1, 4);
    }
    const float a0 = t[0] + __shfl_xor_sync(FULLMASK, t[2], 2);
    const float a1 = t[1] + __shfl_xor_sync(FULLMASK, t[3], 2);
    return a0 + __shfl_xor_sync(FULLMASK, a1, 1);
}

// The softmax update and the PV products of one group: `vu[k]` is the V row of token k; `nv` valid tokens (all TB when FULL). m and l are
// the running max and sum of head qs (the lane's own head: only lane h < 4 writes head h's, and no other lane needs another's), o is
// per head. `spk` is the warp's 32 floats of scratch.
template <bool FULL>
__device__ __forceinline__ void group_pv(float sc, const uint4 (&vu)[TB], float& m, float& l, float (&o)[HPW][8], int nv, int g, int qs,
                                         float* spk) {
    constexpr unsigned FULLMASK = 0xffffffffu;
    // the group's max of head qs
    float x = FULL || g < nv ? sc : -INFINITY;
    x = fmaxf(x, __shfl_xor_sync(FULLMASK, x, 16));
    x = fmaxf(x, __shfl_xor_sync(FULLMASK, x, 8));
    x = fmaxf(x, __shfl_xor_sync(FULLMASK, x, 4));
    const float mx = fmaxf(m, x);
    if (__any_sync(FULLMASK, mx != m)) {
        const float corr = exp2f(m - mx);
        l = l * corr;
#pragma unroll
        for (int h = 0; h < HPW; ++h) {
            const float c = __shfl_sync(FULLMASK, corr, h);
#pragma unroll
            for (int e = 0; e < 8; ++e) o[h][e] = o[h][e] * c;
        }
    }
    m = mx;
    // pk of (token g, head qs) lands at spk[lane] = spk[4 g + qs]: token k's four heads are one 16-byte read
    __syncwarp();
    spk[4 * g + qs] = exp2f(sc - mx);
    __syncwarp();
#pragma unroll
    for (int k = 0; k < TB; ++k) {
        if (!FULL && k >= nv) continue;
        float vf[8];
        widen8(vu[k], vf);
        const float4 pv = *reinterpret_cast<const float4*>(spk + 4 * k);
        const float pp[HPW] = {pv.x, pv.y, pv.z, pv.w};
        l = l + spk[4 * k + qs];
#pragma unroll
        for (int h = 0; h < HPW; ++h) {
#pragma unroll
            for (int e = 0; e < 8; ++e) o[h][e] = o[h][e] + pp[h] * vf[e];
        }
    }
}

constexpr int SOFF_ROW = 36;  // words a row of the offset tables: a multiple of 4 (uint4 reads) that spreads the 8 copies over the banks

// grid (rows, NKV, parts), block 32 * (GRP / HPW): warp w serves q heads kvh * GRP + w * HPW .. + HPW - 1. With MERGED (one part) the
// kernel ends with merge_k's arithmetic for that one part and writes the bf16 output `out` [rows][24][256] instead of the partial sums.
template <bool MERGED>
__global__ void __launch_bounds__(32 * (GRP / HPW))
    attend_h(const __nv_bfloat16* __restrict__ qm, const __nv_bfloat16* __restrict__ kc, const __nv_bfloat16* __restrict__ vc,
             const int* __restrict__ table, int max_blocks, const int* __restrict__ row_slot, const int* __restrict__ positions,
             const int* __restrict__ sel, const int* __restrict__ sel_n, float* __restrict__ part_o, float* __restrict__ part_m,
             float* __restrict__ part_l, __nv_bfloat16* __restrict__ out, int parts) {
    static_assert(GRP % HPW == 0, "heads a warp must divide the group");
    // Per warp, 8 copies of the row offsets (in uint4 units) of 32 consecutive tokens: in copy c, token 8j + k's offset is at position
    // 8j + (k ^ c), so lane g's K rows come from copy g in token order k ^ g and every lane's V rows from copy 0.
    __shared__ __align__(16) unsigned soff[GRP / HPW][8][SOFF_ROW];
    __shared__ __align__(16) float spk[GRP / HPW][32];
    pdl_trigger();
    pdl_wait();  // q, the selection and the positions are earlier kernels' outputs
    const int r = blockIdx.x, kvh = blockIdx.y, part = blockIdx.z;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, qs = lane & 3;
    const int head0 = kvh * GRP + w * HPW;
    const int p = positions[r], nbq = (p + 1) / CR, n = sel_n[r];
    const int total = CR * n + (p - CR * nbq + 1);
    const int a = (int)((long long)total * part / parts), b = (int)((long long)total * (part + 1) / parts);
    const int slot = row_slot[r];
    const int* sl = sel + (size_t)r * TOPK;
    const int* tb = table + (size_t)slot * max_blocks;
    float qv[HPW][8];
#pragma unroll
    for (int s = 0; s < HPW; ++s) {
        const uint4 u = reinterpret_cast<const uint4*>(qm + ((size_t)r * NQ + head0 + (s ^ qs)) * D)[lane];
        const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 f = __bfloat1622float2(h2[e]);
            qv[s][2 * e] = f.x * (0.0625f * LOG2E);
            qv[s][2 * e + 1] = f.y * (0.0625f * LOG2E);
        }
    }
    float m = -INFINITY, l = 0.f, o[HPW][8];
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
#pragma unroll
        for (int e = 0; e < 8; ++e) o[h][e] = 0.f;
    }
    const char* kp = reinterpret_cast<const char*>(kc) + lane * 16;
    const char* vp = reinterpret_cast<const char*>(vc) + lane * 16;
    // the row offset of this lane's token among the 32 starting at `base`
    auto token_offset = [&](int base) {
        const int i = base + lane;
        const int ii = i < b ? i : a;
        const int tok = ii < CR * n ? CR * sl[ii / CR] + ii % CR : CR * nbq + (ii - CR * n);
        return (((unsigned)tb[tok / PAGE] * PAGE + tok % PAGE) * NKV + kvh) * (D / 8);
    };
    const unsigned* krow = soff[w][g];
    const unsigned* vrow = soff[w][0];
    for (int base = a; base < b; base += 4 * TB) {
        {
            const unsigned off = token_offset(base);
            const int lo = lane & 7, hi = lane & 24;
            __syncwarp();
#pragma unroll
            for (int c = 0; c < 8; ++c) soff[w][c][hi + (lo ^ c)] = off;
            __syncwarp();
        }
#pragma unroll 1
        for (int j = 0; j < 4; ++j) {
            const int i0 = base + TB * j;
            if (i0 >= b) break;
            uint4 ku[TB], vu[TB];
#pragma unroll
            for (int k = 0; k < TB; ++k) {
                ku[k] = *reinterpret_cast<const uint4*>(kp + (size_t)krow[TB * j + k] * 16);
                vu[k] = *reinterpret_cast<const uint4*>(vp + (size_t)vrow[TB * j + k] * 16);
            }
            const float sc = group_scores(ku, qv);
            const int nv = b - i0;
            if (nv >= TB)
                group_pv<true>(sc, vu, m, l, o, TB, g, qs, spk[w]);
            else
                group_pv<false>(sc, vu, m, l, o, nv, g, qs, spk[w]);
        }
    }
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
        if (MERGED) {
            // merge_k over one part: mx = m, c = exp2f(m - mx), l = 0 + l * c, o = 0 + o * c, out = bf16(o / l)
            const float mh = __shfl_sync(0xffffffffu, m, h), lh = __shfl_sync(0xffffffffu, l, h);
            const float c = exp2f(mh - mh);
            const float lq = 0.f + lh * c;
            __nv_bfloat162 pk[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float o0 = 0.f + o[h][2 * e] * c, o1 = 0.f + o[h][2 * e + 1] * c;
                pk[e] = __halves2bfloat162(__float2bfloat16_rn(o0 / lq), __float2bfloat16_rn(o1 / lq));
            }
            *reinterpret_cast<uint4*>(out + ((size_t)r * NQ + head0 + h) * D + lane * 8) = *reinterpret_cast<const uint4*>(pk);
        } else {
            const size_t pi = ((size_t)r * NQ + head0 + h) * parts + part;
            float4* po = reinterpret_cast<float4*>(part_o + pi * D + lane * 8);
            po[0] = make_float4(o[h][0], o[h][1], o[h][2], o[h][3]);
            po[1] = make_float4(o[h][4], o[h][5], o[h][6], o[h][7]);
            if (lane == h) {
                part_m[pi] = m;
                part_l[pi] = l;
            }
        }
    }
}

// attend_h for one part (a prefill chunk's attention, merged): the K and V rows of a group of TB tokens are staged in shared memory by the
// block's three warps together (a row is a 512-byte cp.async copy by one warp; the next group's rows are in flight while this group is
// computed) instead of being loaded by every warp for itself. The three warps of a block serve the same kv head, so they read the same
// rows: staging them once cuts the L2 -> SM traffic of the kernel to a third (52 GB -> 17 GB a call at 4,096 rows of 2,052 tokens) and
// takes the load latency off the warps' critical path. The arithmetic is attend_h's, instruction for instruction: group_scores and
// group_pv are the same functions on the same operands (the K row of token k ^ (lane >> 2) and the V row of token k of the group, read
// from shared memory instead of global), so the output is attend_h's bit for bit.
constexpr int ATT_THREADS = 32 * (GRP / HPW);
constexpr int SOFF_BLK = 128;  // tokens one refresh of the offset table covers (16 groups)
static_assert(ATT_THREADS == 96 && SOFF_BLK % TB == 0, "attend_s is built for three warps");

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(unsigned dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}

// grid (rows, NKV), block 96: warp w serves q heads kvh * GRP + w * HPW .. + HPW - 1, as attend_h<true> with one part.
__global__ void __launch_bounds__(ATT_THREADS)
    attend_s(const __nv_bfloat16* __restrict__ qm, const __nv_bfloat16* __restrict__ kc, const __nv_bfloat16* __restrict__ vc,
             const int* __restrict__ table, int max_blocks, const int* __restrict__ row_slot, const int* __restrict__ positions,
             const int* __restrict__ sel, const int* __restrict__ sel_n, __nv_bfloat16* __restrict__ out) {
    __shared__ __align__(128) uint4 tile[2][2][TB][32];  // [stage][K | V][token of the group][16-byte chunk of the 512-byte row]
    __shared__ __align__(16) unsigned soff[2][SOFF_BLK];  // row offsets (uint4 units) of the tokens, a block of 128 at a time, by parity
    __shared__ __align__(16) float spk[GRP / HPW][32];
    pdl_trigger();
    pdl_wait();  // q, the selection and the positions are earlier kernels' outputs
    const int r = blockIdx.x, kvh = blockIdx.y;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, qs = lane & 3;
    const int head0 = kvh * GRP + w * HPW;
    const int p = positions[r], nbq = (p + 1) / CR, n = sel_n[r];
    const int b = CR * n + (p - CR * nbq + 1);  // one part: the tokens are [0, b)
    const int slot = row_slot[r];
    const int* sl = sel + (size_t)r * TOPK;
    const int* tb = table + (size_t)slot * max_blocks;
    float qv[HPW][8];
#pragma unroll
    for (int s = 0; s < HPW; ++s) {
        const uint4 u = reinterpret_cast<const uint4*>(qm + ((size_t)r * NQ + head0 + (s ^ qs)) * D)[lane];
        const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 f = __bfloat1622float2(h2[e]);
            qv[s][2 * e] = f.x * (0.0625f * LOG2E);
            qv[s][2 * e + 1] = f.y * (0.0625f * LOG2E);
        }
    }
    float m = -INFINITY, l = 0.f, o[HPW][8];
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
#pragma unroll
        for (int e = 0; e < 8; ++e) o[h][e] = 0.f;
    }
    // the offsets of tokens blk * SOFF_BLK .. + SOFF_BLK - 1 (a token past the end reads token 0: its row is never used)
    auto fill = [&](int blk) {
        for (int s = threadIdx.x; s < SOFF_BLK; s += ATT_THREADS) {
            const int i = blk * SOFF_BLK + s;
            const int ii = i < b ? i : 0;
            const int tok = ii < CR * n ? CR * sl[ii / CR] + ii % CR : CR * nbq + (ii - CR * n);
            soff[blk & 1][s] = (((unsigned)tb[tok / PAGE] * PAGE + tok % PAGE) * NKV + kvh) * (D / 8);
        }
    };
    // the K and V rows of group `grp` into stage grp & 1: warp w copies tokens w, w + 3, w + 6, a row a 32-lane cp.async
    const uint4* kg = reinterpret_cast<const uint4*>(kc) + lane;
    const uint4* vg = reinterpret_cast<const uint4*>(vc) + lane;
    auto issue = [&](int grp) {
        const unsigned* so = soff[(grp >> 4) & 1] + (grp & 15) * TB;
#pragma unroll
        for (int u = 0; u < 3; ++u) {
            const int t = w + 3 * u;
            if (t < TB) {
                const unsigned off = so[t];
                cp_async16(smem_u32(&tile[grp & 1][0][t][lane]), kg + off);
                cp_async16(smem_u32(&tile[grp & 1][1][t][lane]), vg + off);
            }
        }
    };
    const int groups = (b + TB - 1) / TB;
    fill(0);
    __syncthreads();
    issue(0);
    asm volatile("cp.async.commit_group;\n" ::);
    for (int j = 0; j < groups; ++j) {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        if ((j & 15) == 0) fill(j / 16 + 1);  // first read by the issue of group 16 (j / 16 + 1), 15 barriers later
        __syncthreads();  // group j's rows have landed (every thread's copies); every warp is done with group j - 1's stage
        if (j + 1 < groups) {
            issue(j + 1);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        const uint4(&kt)[TB][32] = tile[j & 1][0];
        const uint4(&vt)[TB][32] = tile[j & 1][1];
        uint4 ku[TB], vu[TB];
#pragma unroll
        for (int k = 0; k < TB; ++k) {
            ku[k] = kt[k ^ g][lane];
            vu[k] = vt[k][lane];
        }
        const float sc = group_scores(ku, qv);
        const int nv = b - TB * j;
        if (nv >= TB)
            group_pv<true>(sc, vu, m, l, o, TB, g, qs, spk[w]);
        else
            group_pv<false>(sc, vu, m, l, o, nv, g, qs, spk[w]);
    }
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
        // merge_k over one part: mx = m, c = exp2f(m - mx), l = 0 + l * c, o = 0 + o * c, out = bf16(o / l)
        const float mh = __shfl_sync(0xffffffffu, m, h), lh = __shfl_sync(0xffffffffu, l, h);
        const float c = exp2f(mh - mh);
        const float lq = 0.f + lh * c;
        __nv_bfloat162 pk[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float o0 = 0.f + o[h][2 * e] * c, o1 = 0.f + o[h][2 * e + 1] * c;
            pk[e] = __halves2bfloat162(__float2bfloat16_rn(o0 / lq), __float2bfloat16_rn(o1 / lq));
        }
        *reinterpret_cast<uint4*>(out + ((size_t)r * NQ + head0 + h) * D + lane * 8) = *reinterpret_cast<const uint4*>(pk);
    }
}

// grid (rows, NQ), block D
__global__ void __launch_bounds__(D) merge_k(const float* __restrict__ part_o, const float* __restrict__ part_m,
                                             const float* __restrict__ part_l, __nv_bfloat16* __restrict__ out, int parts) {
    pdl_trigger();
    pdl_wait();  // the partial sums are attend's output, and out may still be read by earlier kernels
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const size_t base = ((size_t)r * NQ + h) * parts;
    float mx = -INFINITY;
    for (int k = 0; k < parts; ++k) mx = fmaxf(mx, part_m[base + k]);
    float l = 0.f, o = 0.f;
    for (int k = 0; k < parts; ++k) {
        const float mk = part_m[base + k];
        if (mk == -INFINITY) continue;
        const float c = exp2f(mk - mx);
        l = l + part_l[base + k] * c;
        o = o + part_o[(base + k) * D + d] * c;
    }
    out[((size_t)r * NQ + h) * D + d] = __float2bfloat16_rn(o / l);
}

}  // namespace qsa

GB_EXPORT int gb_qsa_stage(const void* qk, const void* qw, const int* positions, void* q, void* kraw, int rows, float eps,
                           float theta, void* stream) {
    if (rows < 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    gbp_pdl::launch_pdl(flash_pdl(), qsa::stage_k, dim3((unsigned)rows), dim3(160), 0, gb_stream(stream), (const uint16_t*)qk,
                        (const float*)qw, positions, (__half*)q, (__half*)kraw, eps, theta);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_qsa_pool(const void* kraw, const void* ring, const void* kw, const int* positions, const int* seq_start,
                          const int* seq_slot, int n_seq, const int* table, int max_blocks, void* pool, int rows,
                          int ring_len, float eps, float theta, void* stream) {
    if (rows < 0 || n_seq <= 0 || ring_len < qsa::CR) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    gbp_pdl::launch_pdl(flash_pdl(), qsa::pool_k, dim3((unsigned)rows), dim3(qsa::DK), 0, gb_stream(stream), (const __half*)kraw,
                        (const __half*)ring, (const float*)kw, positions, seq_start, seq_slot, n_seq, table, max_blocks,
                        (__half*)pool, ring_len, eps, theta);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_qsa_ring(const void* kraw, void* ring, const int* positions, const int* seq_start, const int* seq_slot,
                          int n_seq, int rows, int ring_len, void* stream) {
    if (rows < 0 || n_seq <= 0 || ring_len <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    gbp_pdl::launch_pdl(flash_pdl(), qsa::ring_k, dim3((unsigned)rows), dim3(qsa::DK), 0, gb_stream(stream), (const __half*)kraw,
                        (__half*)ring, positions, seq_start, seq_slot, n_seq, ring_len);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_qsa_scores(const void* q, const void* pool, const int* table, int max_blocks, const int* row_slot,
                            const int* positions, void* scores, int rows, int nb_cap, void* stream) {
    if (rows < 0 || nb_cap <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    const dim3 grid((nb_cap + 255) / 256, (rows + qsa::RG - 1) / qsa::RG);
    gbp_pdl::launch_pdl(flash_pdl(), qsa::scores_k, grid, dim3(256), 0, gb_stream(stream), (const __half*)q, (const __half*)pool,
                        table, max_blocks, row_slot, positions, (float*)scores, rows, nb_cap, 1.0f / sqrtf((float)qsa::DK));
    return GB_CHECK_LAUNCH();
}

// gb_qsa_scores through the kernel it replaced: the reference the tests compare against.
GB_EXPORT int gb_qsa_scores_ref(const void* q, const void* pool, const int* table, int max_blocks, const int* row_slot,
                                const int* positions, void* scores, int rows, int nb_cap, void* stream) {
    if (rows < 0 || nb_cap <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    const dim3 grid((nb_cap + 255) / 256, (rows + qsa::RG - 1) / qsa::RG);
    qsa::scores_ref_k<<<grid, 256, 0, gb_stream(stream)>>>((const __half*)q, (const __half*)pool, table, max_blocks, row_slot, positions,
                                                           (float*)scores, rows, nb_cap, 1.0f / sqrtf((float)qsa::DK));
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_qsa_topk(const void* scores, const int* positions, int rows, int nb_cap, int* sel, int* sel_n,
                          void* stream) {
    if (rows < 0 || nb_cap <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    gbp_pdl::launch_pdl(flash_pdl(), qsa::topk_k, dim3((unsigned)rows), dim3(1024), 0, gb_stream(stream), (const float*)scores,
                        positions, nb_cap, sel, sel_n);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_qsa_attend(const void* q, const void* k_cache, const void* v_cache, const int* table, int max_blocks,
                            const int* row_slot, const int* positions, const int* sel, const int* sel_n, void* part_o,
                            void* part_m, void* part_l, int rows, int parts, void* stream) {
    if (rows < 0 || parts <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    const dim3 grid(rows, qsa::NKV, parts);
    // many rows (a prefill chunk): four heads a warp, the same bits as one head a warp (see attend_h); the decode windows' few rows
    // keep the twelve-warp block, whose grid fills the SMs through `parts`
    if (rows >= qsa::ATTEND_H_MIN_ROWS) {
        gbp_pdl::launch_pdl(flash_pdl(), qsa::attend_h<false>, grid, dim3(32 * (qsa::GRP / qsa::ATTEND_HPW)), 0, gb_stream(stream),
                            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k_cache, (const __nv_bfloat16*)v_cache, table, max_blocks,
                            row_slot, positions, sel, sel_n, (float*)part_o, (float*)part_m, (float*)part_l,
                            (__nv_bfloat16*)nullptr, parts);
        return GB_CHECK_LAUNCH();
    }
    gbp_pdl::launch_pdl(flash_pdl(), qsa::attend_k, grid, dim3(32 * qsa::GRP), 0, gb_stream(stream), (const __nv_bfloat16*)q,
                        (const __nv_bfloat16*)k_cache, (const __nv_bfloat16*)v_cache, table, max_blocks, row_slot, positions, sel,
                        sel_n, (float*)part_o, (float*)part_m, (float*)part_l, parts);
    return GB_CHECK_LAUNCH();
}

// gb_qsa_attend and gb_qsa_merge for a prefill chunk's one part: the attention kernel ends with merge_k's arithmetic and writes the
// bf16 output [rows][24][256] itself (the output is merge_k's bit for bit; no partial sums are stored). rows >= ATTEND_H_MIN_ROWS.
GB_EXPORT int gb_qsa_attend_merged(const void* q, const void* k_cache, const void* v_cache, const int* table, int max_blocks,
                                   const int* row_slot, const int* positions, const int* sel, const int* sel_n, void* out, int rows,
                                   void* stream) {
    if (rows < qsa::ATTEND_H_MIN_ROWS) return (int)cudaErrorInvalidValue;
    const dim3 grid(rows, qsa::NKV, 1);
    qsa::attend_s<<<grid, qsa::ATT_THREADS, 0, gb_stream(stream)>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k_cache, (const __nv_bfloat16*)v_cache, table, max_blocks, row_slot, positions,
        sel, sel_n, (__nv_bfloat16*)out);
    return GB_CHECK_LAUNCH();
}

// gb_qsa_attend_merged through the kernel it replaced (every warp loads its own K and V rows): the reference the tests compare against.
GB_EXPORT int gb_qsa_attend_merged_ref(const void* q, const void* k_cache, const void* v_cache, const int* table, int max_blocks,
                                       const int* row_slot, const int* positions, const int* sel, const int* sel_n, void* out,
                                       int rows, void* stream) {
    if (rows < qsa::ATTEND_H_MIN_ROWS) return (int)cudaErrorInvalidValue;
    const dim3 grid(rows, qsa::NKV, 1);
    qsa::attend_h<true><<<grid, 32 * (qsa::GRP / qsa::ATTEND_HPW), 0, gb_stream(stream)>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k_cache, (const __nv_bfloat16*)v_cache, table, max_blocks, row_slot, positions,
        sel, sel_n, nullptr, nullptr, nullptr, (__nv_bfloat16*)out, 1);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_qsa_merge(const void* part_o, const void* part_m, const void* part_l, void* out, int rows, int parts,
                           void* stream) {
    if (rows < 0 || parts <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    const dim3 grid(rows, qsa::NQ);
    gbp_pdl::launch_pdl(flash_pdl(), qsa::merge_k, grid, dim3(qsa::D), 0, gb_stream(stream), (const float*)part_o,
                        (const float*)part_m, (const float*)part_l, (__nv_bfloat16*)out, parts);
    return GB_CHECK_LAUNCH();
}
