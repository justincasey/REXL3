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
#include <cuda_bf16.h>

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

// rope one 128-dim vector held one element per thread (d = threadIdx.x) through `sh`; result rounded to f16
__device__ __forceinline__ float rope_elem(float x, int d, int pos, float theta, float* sh) {
    sh[d] = x;
    __syncthreads();
    float o = x;
    if (d < RD) {
        const int i = d % (RD / 2);
        const float ang = (float)pos * inv_freq(theta, i);
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
    const int r = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
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
    for (int j = 0; j < 4; ++j) x[j] = rh((x[j] * inv) * qw[lane + 32 * j]);
    // rope pair (lane, lane + 32) sits in this lane as x[0], x[1]
    const float ang = (float)positions[r] * inv_freq(theta, lane);
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
    const int r = blockIdx.x, d = threadIdx.x;
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
    const float y = rh((m * inv) * kw[d]);
    const int j = p / CR;
    const float o = rope_elem(y, d, j * CR, theta, sh);
    const int page = table[(size_t)slot * max_blocks + j / PB];
    pool[((size_t)page * PB + j % PB) * DK + d] = __float2half_rn(o);
}

__global__ void __launch_bounds__(DK) ring_k(const __half* __restrict__ kraw, __half* __restrict__ ring,
                                             const int* __restrict__ positions, const int* __restrict__ seq_start,
                                             const int* __restrict__ seq_slot, int n_seq, int ring_len) {
    const int r = blockIdx.x, d = threadIdx.x;
    const int sq = seq_of(seq_start, n_seq, r);
    const int p_last = positions[seq_start[sq + 1] - 1], p = positions[r];
    if (p <= p_last - ring_len) return;
    ring[((size_t)seq_slot[sq] * ring_len + p % ring_len) * DK + d] = kraw[(size_t)r * DK + d];
}

constexpr int RG = 16;  // rows a scores CTA serves

// grid (ceil(nb_cap / 256), ceil(rows / RG)), block 256: one thread per block j, for every row of the group. Rows of one
// sequence are adjacent, so a thread reads its key once per run of rows with the same slot.
__global__ void __launch_bounds__(256) scores_k(const __half* __restrict__ q, const __half* __restrict__ pool,
                                                const int* __restrict__ table, int max_blocks,
                                                const int* __restrict__ row_slot, const int* __restrict__ positions,
                                                float* __restrict__ scores, int rows, int nb_cap, float scale) {
    __shared__ float qs[RG][NH * DK];
    __shared__ int nbq_s[RG], slot_s[RG];
    __shared__ int nbq_max;
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

// grid rows, block 1024. Scores are >= 0, so their bit patterns order like the values.
__global__ void __launch_bounds__(1024) topk_k(const float* __restrict__ scores, const int* __restrict__ positions,
                                               int nb_cap, int* __restrict__ sel, int* __restrict__ sel_n) {
    __shared__ unsigned hist[256];
    __shared__ unsigned whist[32][256];
    __shared__ unsigned sh_prefix, sh_need;
    __shared__ int wsum[32], weq[32];
    __shared__ int sh_base_gt, sh_base_eq;
    const int r = blockIdx.x, t = threadIdx.x;
    const int nbq = (positions[r] + 1) / CR;
    int* out = sel + (size_t)r * TOPK;
    if (nbq <= TOPK) {
        for (int i = t; i < nbq; i += blockDim.x) out[i] = i;
        if (t == 0) sel_n[r] = nbq;
        return;
    }
    const unsigned* s = reinterpret_cast<const unsigned*>(scores + (size_t)r * nb_cap);
    // radix select of the TOPK-th largest bit pattern, 8 bits a pass from the top
    unsigned prefix = 0, need = TOPK;
    const int nwarps = blockDim.x >> 5, wid = t >> 5, ln = t & 31;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 32 * 256; i += blockDim.x) whist[i / 256][i % 256] = 0;
        __syncthreads();
        const unsigned hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        // scores cluster in a few bins: lanes with the same digit add once (warp-private bins, no cross-warp contention)
        const int span = (nbq + 31) / 32 * 32;
        for (int i = t; i < span; i += blockDim.x) {
            const unsigned v = i < nbq ? s[i] : 0u;
            const bool in = i < nbq && (v & hi_mask) == prefix;
            const unsigned dgt = in ? ((v >> shift) & 255u) : 256u;
            const unsigned peers = __match_any_sync(0xffffffffu, dgt);
            if (in && ln == __ffs(peers) - 1) whist[wid][dgt] += __popc(peers);
        }
        __syncthreads();
        for (int b = t; b < 256; b += blockDim.x) {
            unsigned c = 0;
            for (int w = 0; w < nwarps; ++w) c += whist[w][b];
            hist[b] = c;
        }
        __syncthreads();
        if (t == 0) {
            unsigned acc = 0;
            int dgt = 255;
            for (; dgt > 0; --dgt) {
                if (acc + hist[dgt] >= need) break;
                acc += hist[dgt];
            }
            sh_prefix = prefix | ((unsigned)dgt << shift);
            sh_need = need - acc;
        }
        __syncthreads();
        prefix = sh_prefix;
        need = sh_need;
    }
    // prefix is the threshold pattern; take every greater one and the first `need` equal ones, in index order
    if (t == 0) { sh_base_gt = 0; sh_base_eq = 0; }
    __syncthreads();
    const int lane = ln, w = wid, nw = nwarps;
    const unsigned lt = (1u << lane) - 1u;
    for (int i0 = 0; i0 < nbq; i0 += blockDim.x) {
        const int i = i0 + t;
        const unsigned v = i < nbq ? s[i] : 0u;
        const bool gt = i < nbq && v > prefix, eq = i < nbq && v == prefix;
        // rank of an equal pattern among the equal ones before it, to keep exactly `need` of them
        const unsigned beq = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) weq[w] = __popc(beq);
        __syncthreads();
        int eq_before = sh_base_eq + __popc(beq & lt);
        for (int k = 0; k < w; ++k) eq_before += weq[k];
        const bool take = gt || (eq && eq_before < (int)need);
        const unsigned bt = __ballot_sync(0xffffffffu, take);
        if (lane == 0) wsum[w] = __popc(bt);
        __syncthreads();
        int at = sh_base_gt + __popc(bt & lt);
        for (int k = 0; k < w; ++k) at += wsum[k];
        if (take) out[at] = i;
        __syncthreads();
        if (t == 0) {
            int tk = 0, te = 0;
            for (int k = 0; k < nw; ++k) {
                tk += wsum[k];
                te += weq[k];
            }
            sh_base_gt += tk;
            sh_base_eq += te;
        }
        __syncthreads();
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

// attend_k with HPW query heads a warp (block 32 * GRP / HPW): the 8-token K/V loads, their address arithmetic and the bf16 widening are
// done once for the warp's heads instead of once per head. Every head's own arithmetic is attend_k's, expression for expression and in
// the same order (this file is --fmad=false), so a head's part_o / part_m / part_l are attend_k's bit for bit; only the number of
// instructions and L1 loads a (token, head) costs changes. Grid as attend_k: (rows, NKV, parts).
template <int HPW, int TBH>
__global__ void __launch_bounds__(32 * (GRP / HPW))
    attend_h(const __nv_bfloat16* __restrict__ qm, const __nv_bfloat16* __restrict__ kc, const __nv_bfloat16* __restrict__ vc,
             const int* __restrict__ table, int max_blocks, const int* __restrict__ row_slot, const int* __restrict__ positions,
             const int* __restrict__ sel, const int* __restrict__ sel_n, float* __restrict__ part_o, float* __restrict__ part_m,
             float* __restrict__ part_l, int parts) {
    static_assert(GRP % HPW == 0, "heads a warp must divide the group");
    const int r = blockIdx.x, kvh = blockIdx.y, part = blockIdx.z;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head0 = kvh * GRP + w * HPW;
    const int p = positions[r], nbq = (p + 1) / CR, n = sel_n[r];
    const int total = CR * n + (p - CR * nbq + 1);
    const int a = (int)((long long)total * part / parts), b = (int)((long long)total * (part + 1) / parts);
    const int slot = row_slot[r];
    const int* sl = sel + (size_t)r * TOPK;
    const int* tb = table + (size_t)slot * max_blocks;
    float qv[HPW][8];
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
        const uint4 u = reinterpret_cast<const uint4*>(qm + ((size_t)r * NQ + head0 + h) * D)[lane];
        const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 f = __bfloat1622float2(h2[e]);
            qv[h][2 * e] = f.x * (0.0625f * LOG2E);
            qv[h][2 * e + 1] = f.y * (0.0625f * LOG2E);
        }
    }
    float m[HPW], l[HPW], o[HPW][8];
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) o[h][e] = 0.f;
    }
    for (int i0 = a; i0 < b; i0 += TBH) {
        uint4 ku[TBH], vu[TBH];
#pragma unroll
        for (int k = 0; k < TBH; ++k) {
            const int i = i0 + k;
            const int ii = i < b ? i : a;
            const int tok = ii < CR * n ? CR * sl[ii / CR] + ii % CR : CR * nbq + (ii - CR * n);
            const size_t off = (((size_t)tb[tok / PAGE] * PAGE + tok % PAGE) * NKV + kvh) * D;
            ku[k] = reinterpret_cast<const uint4*>(kc + off)[lane];
            vu[k] = reinterpret_cast<const uint4*>(vc + off)[lane];
        }
        float sc[HPW][TBH];
#pragma unroll
        for (int k = 0; k < TBH; ++k) {
            const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&ku[k]);
            float kf[8];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 f = __bfloat1622float2(h2[e]);
                kf[2 * e] = f.x;
                kf[2 * e + 1] = f.y;
            }
#pragma unroll
            for (int h = 0; h < HPW; ++h) {
                float d = 0.f;
#pragma unroll
                for (int e = 0; e < 4; ++e) d = d + qv[h][2 * e] * kf[2 * e] + qv[h][2 * e + 1] * kf[2 * e + 1];
                sc[h][k] = warp_sum(d);
            }
        }
        float mx[HPW];
#pragma unroll
        for (int h = 0; h < HPW; ++h) {
            mx[h] = m[h];
#pragma unroll
            for (int k = 0; k < TBH; ++k)
                if (i0 + k < b) mx[h] = fmaxf(mx[h], sc[h][k]);
            const float corr = exp2f(m[h] - mx[h]);
            l[h] = l[h] * corr;
#pragma unroll
            for (int e = 0; e < 8; ++e) o[h][e] = o[h][e] * corr;
        }
#pragma unroll
        for (int k = 0; k < TBH; ++k) {
            if (i0 + k >= b) continue;
            const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&vu[k]);
            float vf[8];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 f = __bfloat1622float2(h2[e]);
                vf[2 * e] = f.x;
                vf[2 * e + 1] = f.y;
            }
#pragma unroll
            for (int h = 0; h < HPW; ++h) {
                const float pk = exp2f(sc[h][k] - mx[h]);
                l[h] = l[h] + pk;
#pragma unroll
                for (int e = 0; e < 8; ++e) o[h][e] = o[h][e] + pk * vf[e];
            }
        }
#pragma unroll
        for (int h = 0; h < HPW; ++h) m[h] = mx[h];
    }
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
        const size_t pi = ((size_t)r * NQ + head0 + h) * parts + part;
        float* po = part_o + pi * D + lane * 8;
#pragma unroll
        for (int e = 0; e < 8; ++e) po[e] = o[h][e];
        if (lane == 0) {
            part_m[pi] = m[h];
            part_l[pi] = l[h];
        }
    }
}

// grid (rows, NQ), block D
__global__ void __launch_bounds__(D) merge_k(const float* __restrict__ part_o, const float* __restrict__ part_m,
                                             const float* __restrict__ part_l, __nv_bfloat16* __restrict__ out, int parts) {
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
    qsa::stage_k<<<rows, 160, 0, gb_stream(stream)>>>((const uint16_t*)qk, (const float*)qw, positions, (__half*)q,
                                                      (__half*)kraw, eps, theta);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_qsa_pool(const void* kraw, const void* ring, const void* kw, const int* positions, const int* seq_start,
                          const int* seq_slot, int n_seq, const int* table, int max_blocks, void* pool, int rows,
                          int ring_len, float eps, float theta, void* stream) {
    if (rows < 0 || n_seq <= 0 || ring_len < qsa::CR) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    qsa::pool_k<<<rows, qsa::DK, 0, gb_stream(stream)>>>((const __half*)kraw, (const __half*)ring, (const float*)kw,
                                                         positions, seq_start, seq_slot, n_seq, table, max_blocks,
                                                         (__half*)pool, ring_len, eps, theta);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_qsa_ring(const void* kraw, void* ring, const int* positions, const int* seq_start, const int* seq_slot,
                          int n_seq, int rows, int ring_len, void* stream) {
    if (rows < 0 || n_seq <= 0 || ring_len <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    qsa::ring_k<<<rows, qsa::DK, 0, gb_stream(stream)>>>((const __half*)kraw, (__half*)ring, positions, seq_start, seq_slot,
                                                         n_seq, ring_len);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_qsa_scores(const void* q, const void* pool, const int* table, int max_blocks, const int* row_slot,
                            const int* positions, void* scores, int rows, int nb_cap, void* stream) {
    if (rows < 0 || nb_cap <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    const dim3 grid((nb_cap + 255) / 256, (rows + qsa::RG - 1) / qsa::RG);
    qsa::scores_k<<<grid, 256, 0, gb_stream(stream)>>>((const __half*)q, (const __half*)pool, table, max_blocks, row_slot,
                                                       positions, (float*)scores, rows, nb_cap, 1.0f / sqrtf((float)qsa::DK));
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_qsa_topk(const void* scores, const int* positions, int rows, int nb_cap, int* sel, int* sel_n,
                          void* stream) {
    if (rows < 0 || nb_cap <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    qsa::topk_k<<<rows, 1024, 0, gb_stream(stream)>>>((const float*)scores, positions, nb_cap, sel, sel_n);
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
        qsa::attend_h<qsa::ATTEND_HPW, qsa::TB><<<grid, 32 * (qsa::GRP / qsa::ATTEND_HPW), 0, gb_stream(stream)>>>(
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k_cache, (const __nv_bfloat16*)v_cache, table, max_blocks, row_slot,
            positions, sel, sel_n, (float*)part_o, (float*)part_m, (float*)part_l, parts);
        return GB_CHECK_LAUNCH();
    }
    qsa::attend_k<<<grid, 32 * qsa::GRP, 0, gb_stream(stream)>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k_cache, (const __nv_bfloat16*)v_cache, table, max_blocks, row_slot,
        positions, sel, sel_n, (float*)part_o, (float*)part_m, (float*)part_l, parts);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_qsa_merge(const void* part_o, const void* part_m, const void* part_l, void* out, int rows, int parts,
                           void* stream) {
    if (rows < 0 || parts <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    const dim3 grid(rows, qsa::NQ);
    qsa::merge_k<<<grid, qsa::D, 0, gb_stream(stream)>>>((const float*)part_o, (const float*)part_m, (const float*)part_l,
                                                         (__nv_bfloat16*)out, parts);
    return GB_CHECK_LAUNCH();
}
