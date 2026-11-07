// gb-nvcc: --expt-relaxed-constexpr --fmad=false
// Grouped EXL3 routed experts for a MoE layer (GLM-5.3-Flash: 288 experts, top-8, gate/up 4096 -> 2048, down 2048 -> 4096,
// one width per layer and projection, mul1 codebook), as one pipeline of launches:
//
//   group           (row, slot) pairs -> distinct experts in id order, each with its member pairs in pair order
//   rot_in          per pair p = row * topk + slot: xg = fp16(H(x * suh_gate[e])), xu = fp16(H(x * suh_up[e]))      (H = FWHT/sqrt(128))
//   gemv (gate+up)  per (expert, 128 columns, K split, matrix): the expert's trellis times the member rows' xg / xu, fp32 partials
//   gateup epilogue per pair: sum the K splits in order, H, * svh, silu(gate) clamped to <= limit, up clamped to +-limit,
//                   product, * suh_down, H, fp16 -> xd
//   gemv (down)     the same kernel on xd
//   down + combine  per row: each slot's partials summed in split order, H, * svh_down -> y_k (fp32); then
//                   out = fmaf chain over the slots IN SLOT ORDER of w_k * y_k, then the shared expert's output added LAST
//
// The kernel design follows TensorFold's grouped expert GEMV (cuda/exl3/experts_grouped.cuh, experts.cu; MIT, pinned 9cd52ab; see
// THIRD_PARTY_NOTICES.md) over ExLlamaV3's decode (MIT, Copyright (c) 2025 Turboderp); the tile decode is REXL3's
// exl3_linear_decode.cuh, the step prefetch and warp reduction are exl3_linear.cu's decode kernel, here with the weight tile
// pointer taken per expert.
//
// ROW INVARIANCE BY CONSTRUCTION. Every number a pair p produces is a function of (x[row of p], expert of p, the layer's
// weights) alone:
//   * a pair is one row of a 16-row mma.m16n8k16 tile; mma keeps rows apart (row i of D depends on row i of A only), whatever the
//     other 15 rows hold and whichever of the 16 positions the pair takes, so the tile position and the other members do not matter;
//   * the K split SK, the warps per block WK, the columns per block and the tile order are functions of (K, N, bits) alone
//     (gb_moe_default_plan), never of the row count or of how many experts the window touches; every sum (over k tiles in a warp,
//     over warps in warp order, over splits in split order, over slots in slot order) has a fixed order; there are no atomics;
//   * the grouping only decides WHICH block computes a pair, never how.
// This file is compiled --fmad=false and every fused multiply-add in it is an explicit fmaf.

#include <stdint.h>

#include "common.cuh"
#include "exl3_linear_decode.cuh"
#include "pdl.cuh"

using gbp_pdl::pdl_trigger;
using gbp_pdl::pdl_wait;

using namespace gbx;

namespace {

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

constexpr int kGroupThreads = 1024;          // one expert a thread: E <= 1024
constexpr int kMaxTopK = 16;

__device__ __forceinline__ void load4(const void* p, int dtype, size_t i, float (&v)[4]) {
    if (dtype == F32) {
        const float4 u = *reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
        v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
    } else if (dtype == BF16) {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p) + i);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const half*>(p) + i);
        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    }
}

__device__ __forceinline__ void load4h(const half* p, float (&v)[4]) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
    v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
}

__device__ __forceinline__ void store4h(half* p, const float (&v)[4]) {
    half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
    uint2 u;
    u.x = *reinterpret_cast<uint32_t*>(&a);
    u.y = *reinterpret_cast<uint32_t*>(&b);
    *reinterpret_cast<uint2*>(p) = u;
}

// ---------------------------------------------------------------------------------------------------------------
// grouping

// One block: for every expert e < E (thread e) the pairs that picked it; distinct experts in id order, members in pair order.
// uids[u] / mcount[u] / members[u * maxm + j] (maxm = rows * topk, so no expert can overflow its list); *ucount = distinct count.
// Picks outside [0, E) belong to no expert.
__global__ void __launch_bounds__(kGroupThreads) group_kernel(const int* __restrict__ pick, int* __restrict__ uids,
                                                              int* __restrict__ ucount, int* __restrict__ mcount,
                                                              int* __restrict__ members, int P, int E) {
    extern __shared__ int sh_pick[];
    __shared__ int warp_tot[kGroupThreads / 32];
    pdl_wait();                                  // pick is the predecessor's output
    for (int i = threadIdx.x; i < P; i += kGroupThreads) sh_pick[i] = pick[i];
    __syncthreads();
    pdl_trigger();
    const int e = threadIdx.x;
    int cnt = 0;
    if (e < E)
        for (int i = 0; i < P; ++i) cnt += sh_pick[i] == e;
    const int used = cnt > 0;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = used;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) warp_tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        const int v = warp_tot[lane];
        int s = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int x = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += x;
        }
        warp_tot[lane] = s - v;
        if (lane == 31) ucount[0] = s;
    }
    __syncthreads();
    if (!used) return;
    const int place = warp_tot[warp] + inc - used;
    uids[place] = e;
    mcount[place] = cnt;
    int j = 0;
    for (int i = 0; i < P; ++i)
        if (sh_pick[i] == e) members[(size_t)place * P + j++] = i;
}

// ---------------------------------------------------------------------------------------------------------------
// input rotation: out_mat[p] = fp16(H(x[row] * suh_mat[e])) for both matrices of a pair. One warp a 128-wide block.

// One lane's 4 values of one pair's 128-wide block: x at element xi, the expert's suh at suh_k, the output at out_k.
__device__ __forceinline__ void rot_in_lane(const void* __restrict__ x, int x_dtype, size_t xi, const half* __restrict__ suh_k,
                                            half* __restrict__ out_k, int lane) {
    float v[4], s[4];
    load4(x, x_dtype, xi, v);
    load4h(suh_k, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4h(out_k, v);
}

__global__ void __launch_bounds__(128) rot_in_kernel(const void* __restrict__ x, int x_dtype, const int* __restrict__ pick,
                                                     const half* __restrict__ suh0, const half* __restrict__ suh1,
                                                     half* __restrict__ out0, half* __restrict__ out1, int K, int topk, int E) {
    pdl_wait();                                  // x and pick are the predecessors' outputs; first, early exits included
    pdl_trigger();
    const int p = blockIdx.x, mat = blockIdx.z;
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int row = p / topk;
    const int k = blk * 128 + 4 * lane;
    rot_in_lane(x, x_dtype, (size_t)row * K + k, (mat ? suh1 : suh0) + (size_t)e * K + k, (mat ? out1 : out0) + (size_t)p * K + k,
                lane);
}

// rot_in_kernel over the experts of several banks at once: pick holds global ids (or -1), and each expert's suh row is found through a
// table of device addresses indexed by global id (suh0 / suh1 [E], each the address of the expert's K halves).
__global__ void __launch_bounds__(128) rot_in_multi_kernel(const void* __restrict__ x, int x_dtype, const int* __restrict__ pick,
                                                           const uint64_t* __restrict__ suh0, const uint64_t* __restrict__ suh1,
                                                           half* __restrict__ out0, half* __restrict__ out1, int K, int topk, int E) {
    pdl_wait();
    pdl_trigger();
    const int p = blockIdx.x, mat = blockIdx.z;
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int row = p / topk;
    const int k = blk * 128 + 4 * lane;
    rot_in_lane(x, x_dtype, (size_t)row * K + k, reinterpret_cast<const half*>((mat ? suh1 : suh0)[e]) + k,
                (mat ? out1 : out0) + (size_t)p * K + k, lane);
}

// ---------------------------------------------------------------------------------------------------------------
// the grouped GEMV: Z[mat][split][pair][n] (fp32 partials) = rows of X[mat] (the member pairs) times the expert's W_q over the split's K range

// A lane's words of one k step: at 1 and 2 bits the warp loads the step together and each lane takes its words by shuffle
// (the same scheme as exl3_linear.cu's decode kernel).
template <int K2>
__host__ __device__ constexpr bool step_shuffled() {
    return K2 == 2 || K2 == 4;
}

template <int K2>
__host__ __device__ constexpr int step_regs() {
    return step_shuffled<K2>() ? tile_words<K2>() / 4 : 8 * lane_words<K2>();
}

template <int K2>
__device__ __forceinline__ void load_step(const uint32_t* step, int lane, uint32_t (&raw)[step_regs<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
#pragma unroll
        for (int c = 0; c < step_regs<K2>(); ++c) raw[c] = __ldg(step + c * 32 + lane);
    } else {
        int word, offset;
        lane_start<K2>(lane, word, offset);
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < LW; ++q) raw[j * LW + q] = __ldg(step + j * TW + (word + q) % TW);
    }
}

template <int K2>
__device__ __forceinline__ void step_lane_words(const uint32_t (&raw)[step_regs<K2>()], int j, int lane, bool prev,
                                                uint32_t (&w)[lane_words<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
        static_assert(LW == 2, "a lane's windows span two words at 1 and 2 bits");
        constexpr int LPW = 8 / K2;
        const uint32_t r = raw[j * TW / 32];
        const int base = (j * TW) % 32;
        const uint32_t own = __shfl_sync(0xffffffffu, r, base + lane / LPW);
        const uint32_t before = __shfl_sync(0xffffffffu, r, base + (lane / LPW + TW - 1) % TW);
        w[0] = prev ? before : own;
        w[1] = prev ? own : 0u;
    } else {
#pragma unroll
        for (int q = 0; q < LW; ++q) w[q] = raw[j * LW + q];
    }
}

// One block's work: n block nb of 128 columns, K split `split` of matrix `mat`, one distinct expert (its trellis T, its cnt member pairs
// listed at mem, their rotated inputs rows of X). WK warps split the K range, the members go through in 16-row passes (an expert hit by
// more than 16 pairs repeats the pass; each pass is self-contained, so a pair's bits are the same in any pass at any tile position).
// red [WK * 8 * 128] floats (16-byte aligned) and rows_sh [16] are the block's shared memory.
template <int K2, int CB, int WK>
__device__ __forceinline__ void gemv_block(const half* __restrict__ X, const uint32_t* __restrict__ T, const int* __restrict__ mem,
                                           int cnt, float* __restrict__ Z, int K, int N, int P, int SK, int nb, int split, int mat,
                                           float* red, int* rows_sh) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const long long stride_k = (long long)(N >> 4) * TW;
    const uint32_t* tiles = T + (size_t)nb * 8 * TW;
    bool prev = false;
    if constexpr (step_shuffled<K2>()) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }

    for (int m0 = 0; m0 < cnt; m0 += 16) {
        if (threadIdx.x < 16) {
            const int m = m0 + threadIdx.x;
            rows_sh[threadIdx.x] = m < cnt ? mem[m] : -1;
        }
        __syncthreads();
        const int first = rows_sh[0];                      // a pass has at least one member
        const int ra = rows_sh[g] >= 0 ? rows_sh[g] : first;
        const int rb = rows_sh[g + 8] >= 0 ? rows_sh[g + 8] : first;   // padded rows repeat a member and are never stored
        const half* x0 = X + (size_t)ra * K;
        const half* x1 = X + (size_t)rb * K;

        float acc[8][2][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

        const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
        constexpr bool PF = K2 <= 12;
        constexpr int SR = step_regs<K2>();
        uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
        if constexpr (PF) load_step<K2>(tile, lane, cur);
#pragma unroll 1
        for (int i = 0; i < per_warp; ++i) {
            const int kt = kt0 + i;
            uint32_t a[4];
            a[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
            a[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
            a[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
            a[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
            if constexpr (PF) {
                if (i + 1 < per_warp) load_step<K2>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[LW];
                if constexpr (PF) step_lane_words<K2>(cur, j, lane, prev, w);
                else ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
                mma16816(acc[j][0], a, b0);
                mma16816(acc[j][1], a, b1);
            }
            if constexpr (PF) {
#pragma unroll
                for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
            }
        }

        // the warps' sums in warp order, rows 0-7 and then rows 8-15 of the pass
        for (int rlo = 0; rlo < 16 && m0 + rlo < cnt; rlo += 8) {
            const int rn = min(cnt - m0 - rlo, 8);
            __syncthreads();                               // red is reused by every half; rows_sh is final
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = i * 16 + h * 8 + 2 * t;
                    *reinterpret_cast<float2*>(red + (warp * 8 + g) * 128 + col) =
                        rlo ? make_float2(acc[i][h][2], acc[i][h][3]) : make_float2(acc[i][h][0], acc[i][h][1]);
                }
            __syncthreads();
            for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                const int r = idx >> 5, c = 4 * (idx & 31);
                float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                for (int w = 1; w < WK; ++w) {
                    const float4 q = *reinterpret_cast<const float4*>(red + (w * 8 + r) * 128 + c);
                    s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                }
                const int p = rows_sh[rlo + r];
                *reinterpret_cast<float4*>(Z + (((size_t)mat * SK + split) * P + p) * N + (size_t)nb * 128 + c) = s;
            }
        }
        __syncthreads();                                   // rows_sh and red are reused by the next pass
    }
}

// Block (n block of 128 columns, matrix * SK + split, distinct expert u): gemv_block of the expert's trellis.
template <int K2, int CB, int WK>
__global__ void __launch_bounds__(WK * 32) moe_gemv_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint64_t* __restrict__ TP0,
    const uint64_t* __restrict__ TP1, const int* __restrict__ uids, const int* __restrict__ ucount,
    const int* __restrict__ mcount, const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK) {
    pdl_wait();                                  // uids, ucount, members and the rotated inputs are the predecessors' outputs
    pdl_trigger();
    const int u = blockIdx.z;
    if (u >= ucount[0]) return;
    __shared__ __align__(16) float red[WK * 8 * 128];
    __shared__ int rows_sh[16];
    const int nb = blockIdx.x, split = blockIdx.y % SK, mat = blockIdx.y / SK;
    const int e = uids[u], cnt = mcount[u];
    gemv_block<K2, CB, WK>(mat ? X1 : X0, reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]), members + (size_t)u * P, cnt, Z, K,
                           N, P, SK, nb, split, mat, red, rows_sh);
}

// moe_gemv_kernel over the experts of several banks of different widths at once (one launch a stage instead of one a bank): the
// distinct experts u are global ids, each with its trellis address (TP0 / TP1 [E]) and its width in bits (B0 / B1 [E], gate and up
// may differ). A block runs gemv_block at its expert's width, so every block is the block a one-width launch would run. WMASK: the
// widths compiled in (bit b-1 for b bits); a width outside it computes nothing (the host picks an instantiation that holds every width).
template <int CB, int WK, int WMASK>
__global__ void __launch_bounds__(WK * 32) moe_gemv_multi_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint64_t* __restrict__ TP0,
    const uint64_t* __restrict__ TP1, const int* __restrict__ B0, const int* __restrict__ B1, const int* __restrict__ uids,
    const int* __restrict__ ucount, const int* __restrict__ mcount, const int* __restrict__ members, float* __restrict__ Z, int K,
    int N, int P, int SK) {
    pdl_wait();
    pdl_trigger();
    const int u = blockIdx.z;
    if (u >= ucount[0]) return;
    __shared__ __align__(16) float red[WK * 8 * 128];
    __shared__ int rows_sh[16];
    const int nb = blockIdx.x, split = blockIdx.y % SK, mat = blockIdx.y / SK;
    const int e = uids[u], cnt = mcount[u];
    const half* X = mat ? X1 : X0;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int* mem = members + (size_t)u * P;
    switch (mat ? B1[e] : B0[e]) {
#define GB_MB(B)                                                                                           \
    case B:                                                                                                \
        if constexpr ((WMASK >> ((B) - 1)) & 1)                                                            \
            gemv_block<2 * (B), CB, WK>(X, T, mem, cnt, Z, K, N, P, SK, nb, split, mat, red, rows_sh);      \
        break;
        GB_MB(1) GB_MB(2) GB_MB(3) GB_MB(4) GB_MB(5) GB_MB(6) GB_MB(7) GB_MB(8)
#undef GB_MB
        default:
            break;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// epilogues. One warp a 128-wide block of one pair (or one row for the combine).

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

// One lane's 4 columns n.. of pair p: gate and up partials -> activation -> xd = fp16(H((act * suh_down[e]))). The expert's svh_g,
// svh_u and suh_d at those columns are sg_n, su_n, sd_n. limit > 0 clamps silu(gate) from above and up to +-limit (the oracle's
// act_gate: the activated gate is clamped, not the pre-activation).
__device__ __forceinline__ void gateup_lane(const float* __restrict__ Z, int p, int P, int N, int SK, int n, int lane,
                                            const half* __restrict__ sg_n, const half* __restrict__ su_n, const half* __restrict__ sd_n,
                                            float limit, half* __restrict__ xd_n) {
    float gv[4], uv[4];
    {
        const float* zg = Z + (size_t)p * N + n;
        const float* zu = Z + ((size_t)SK * P + p) * N + n;
        float4 sg = *reinterpret_cast<const float4*>(zg), su = *reinterpret_cast<const float4*>(zu);
        for (int s = 1; s < SK; ++s) {
            const float4 a = *reinterpret_cast<const float4*>(zg + (size_t)s * P * N);
            const float4 b = *reinterpret_cast<const float4*>(zu + (size_t)s * P * N);
            sg.x += a.x; sg.y += a.y; sg.z += a.z; sg.w += a.w;
            su.x += b.x; su.y += b.y; su.z += b.z; su.w += b.w;
        }
        gv[0] = sg.x; gv[1] = sg.y; gv[2] = sg.z; gv[3] = sg.w;
        uv[0] = su.x; uv[1] = su.y; uv[2] = su.z; uv[3] = su.w;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    float sg4[4], su4[4], sd4[4], v[4];
    load4h(sg_n, sg4);
    load4h(su_n, su4);
    load4h(sd_n, sd4);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float g = gv[j] * HAD_SCALE * sg4[j];
        float u = uv[j] * HAD_SCALE * su4[j];
        float x = silu_f(g);
        if (limit > 0.0f) {
            x = fminf(x, limit);
            u = fminf(fmaxf(u, -limit), limit);
        }
        v[j] = (x * u) * sd4[j];
    }
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4h(xd_n, v);
}

// Per pair: gateup_lane over the pair's expert's scales.
__global__ void __launch_bounds__(128) gateup_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                                              const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                                              const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N,
                                                              int SK, int E, float limit) {
    pdl_wait();                                  // Z (the gate / up partials) and pick are the predecessors' outputs
    pdl_trigger();
    const int p = blockIdx.x;
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (blk * 128 >= N) return;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int n = blk * 128 + 4 * lane;
    const size_t en = (size_t)e * N + n;
    gateup_lane(Z, p, P, N, SK, n, lane, svh_g + en, svh_u + en, suh_d + en, limit, xd + (size_t)p * N + n);
}

// gateup_epilogue_kernel over several banks: pick holds global ids (or -1); tab [3][E] the addresses of each expert's svh_g, svh_u
// and suh_d rows (N halves each).
__global__ void __launch_bounds__(128) gateup_epilogue_multi_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                                                    const uint64_t* __restrict__ tab, half* __restrict__ xd, int P,
                                                                    int N, int SK, int E, float limit) {
    pdl_wait();
    pdl_trigger();
    const int p = blockIdx.x;
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (blk * 128 >= N) return;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int n = blk * 128 + 4 * lane;
    gateup_lane(Z, p, P, N, SK, n, lane, reinterpret_cast<const half*>(tab[e]) + n, reinterpret_cast<const half*>(tab[E + e]) + n,
                reinterpret_cast<const half*>(tab[2 * E + e]) + n, limit, xd + (size_t)p * N + n);
}

// One lane's 4 columns n.. of pair p's y: H(sum of the K splits) * svh_down (the expert's svh_down at those columns: sv_n), fp32.
__device__ __forceinline__ void down_lane(const float* __restrict__ Z, int p, int P, int D, int SK, int n, int lane,
                                          const half* __restrict__ sv_n, float (&o)[4]) {
    float4 s = *reinterpret_cast<const float4*>(Z + (size_t)p * D + n);
    for (int q = 1; q < SK; ++q) {
        const float4 a = *reinterpret_cast<const float4*>(Z + ((size_t)q * P + p) * D + n);
        s.x += a.x; s.y += a.y; s.z += a.z; s.w += a.w;
    }
    float v[4] = {s.x, s.y, s.z, s.w}, sv[4];
    fwht128(v, lane);
    load4h(sv_n, sv);
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * sv[j];
}

// The down epilogue of several banks, pairs only (no combine: a rank's pairs go to the exchange first). Block (pair, 4 blocks of 128
// columns), warp: one block. pick holds global ids (or -1: the pair's row is left alone); svh [E] the address of each expert's svh_down.
__global__ void __launch_bounds__(128) down_pairs_multi_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                                               const uint64_t* __restrict__ svh, float* __restrict__ y_pairs, int P,
                                                               int D, int SK, int E) {
    pdl_wait();
    pdl_trigger();
    const int p = blockIdx.x;
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (blk * 128 >= D) return;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int n = blk * 128 + 4 * lane;
    float o[4];
    down_lane(Z, p, P, D, SK, n, lane, reinterpret_cast<const half*>(svh[e]) + n, o);
    *reinterpret_cast<float4*>(y_pairs + (size_t)p * D + n) = make_float4(o[0], o[1], o[2], o[3]);
}

// Block (row, 128 columns of the model width), warp k = slot k: y_k = H(sum of the K splits) * svh_down[e_k] (fp32); warp 0 then
// adds the slots in slot order, acc = fmaf(w_k, y_k, acc) from 0, and the shared expert's output last.
__global__ void __launch_bounds__(32 * kMaxTopK) down_combine_kernel(
    const float* __restrict__ Z, const int* __restrict__ pick, const half* __restrict__ svh_d, const half* __restrict__ wts,
    const float* __restrict__ shared, float* __restrict__ y_pairs, float* __restrict__ out, int P, int D, int SK, int E,
    int topk) {
    __shared__ float4 part[kMaxTopK][32];
    __shared__ int ok_sh[kMaxTopK];
    pdl_wait();                                  // Z, pick, the weights and the shared output are the predecessors' outputs
    pdl_trigger();
    const int r = blockIdx.x, blk = blockIdx.y;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    if (k < topk) {
        const int p = r * topk + k;
        const int e = pick[p];
        const bool ok = e >= 0 && e < E;
        float o[4] = {0.f, 0.f, 0.f, 0.f};
        if (ok) {
            down_lane(Z, p, P, D, SK, n, lane, svh_d + (size_t)e * D + n, o);
            if (y_pairs) *reinterpret_cast<float4*>(y_pairs + (size_t)p * D + n) = make_float4(o[0], o[1], o[2], o[3]);
        }
        part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
        if (lane == 0) ok_sh[k] = ok;
    }
    __syncthreads();
    if (k != 0) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < topk; ++q) {
        if (!ok_sh[q]) continue;
        const float w = __half2float(wts[(size_t)r * topk + q]);
        const float4 u = part[q][lane];
        acc[0] = fmaf(w, u.x, acc[0]);
        acc[1] = fmaf(w, u.y, acc[1]);
        acc[2] = fmaf(w, u.z, acc[2]);
        acc[3] = fmaf(w, u.w, acc[3]);
    }
    if (shared) {
        const float4 sh = *reinterpret_cast<const float4*>(shared + (size_t)r * D + n);
        acc[0] += sh.x; acc[1] += sh.y; acc[2] += sh.z; acc[3] += sh.w;
    }
    *reinterpret_cast<float4*>(out + (size_t)r * D + n) = make_float4(acc[0], acc[1], acc[2], acc[3]);
}

// The shared expert's activation between its dense EXL3 linears (the oracle's act_mul with act_limit): a = fp16(min(silu(g), L) * clamp(u, -L, L)).
__global__ void __launch_bounds__(256) swiglu_limit_kernel(const float* __restrict__ g, const float* __restrict__ u, half* __restrict__ a,
                                                           long long n, float limit) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = silu_f(g[i]), y = u[i];
    if (limit > 0.0f) {
        x = fminf(x, limit);
        y = fminf(fmaxf(y, -limit), limit);
    }
    a[i] = __float2half_rn(x * y);
}

// The BailingMoeV3 (Ling-3.1-flash-1019) clamp: the GATE is clamped BEFORE silu and up to +-limit, then silu(gate) * up:
// a = fp16(silu(min(g, L)) * clamp(u, -L, L)). limit <= 0: no clamp (plain silu(g) * u).
__global__ void __launch_bounds__(256) swiglu_clamped_kernel(const float* __restrict__ g, const float* __restrict__ u, half* __restrict__ a,
                                                             long long n, float limit) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = g[i], y = u[i];
    if (limit > 0.0f) {
        x = fminf(x, limit);
        y = fminf(fmaxf(y, -limit), limit);
    }
    a[i] = __float2half_rn(silu_f(x) * y);
}

// ---------------------------------------------------------------------------------------------------------------
// launch plumbing

bool dtype_ok(int d) { return d >= 0 && d <= 2; }

template <int K2, int CB, int WK>
int launch_gemv(const half* X0, const half* X1, const uint64_t* TP0, const uint64_t* TP1, const int* uids, const int* ucount,
                const int* mcount, const int* members, float* Z, int mats, int K, int N, int P, int SK, int umax,
                cudaStream_t st) {
    const dim3 grid((unsigned)(N / 128), (unsigned)(mats * SK), (unsigned)umax);
    return gbp_pdl::launch_pdl(gbp_pdl::on(gbp_pdl::kMoe), moe_gemv_kernel<K2, CB, WK>, grid, dim3(WK * 32), 0, st, X0, X1, TP0, TP1, uids,
                               ucount, mcount, members, Z, K, N, P, SK);
}

#define GB_MOE_FOR_CB(X, K2, WK) X(K2, 0, WK) X(K2, 1, WK) X(K2, 2, WK)
#define GB_MOE_FOR_ALL(X, WK)                                                                                          \
    GB_MOE_FOR_CB(X, 2, WK) GB_MOE_FOR_CB(X, 4, WK) GB_MOE_FOR_CB(X, 6, WK) GB_MOE_FOR_CB(X, 8, WK)                    \
    GB_MOE_FOR_CB(X, 10, WK) GB_MOE_FOR_CB(X, 12, WK) GB_MOE_FOR_CB(X, 14, WK) GB_MOE_FOR_CB(X, 16, WK)

// The widths the RED-SNOW packs mix (4, 5 and 6 bits): the multi GEMV compiled for those alone, and for every width.
constexpr int kWidths456 = 0x38;
constexpr int kWidthsAll = 0xFF;

template <int CB, int WK, int WMASK>
int launch_gemv_multi(const half* X0, const half* X1, const uint64_t* TP0, const uint64_t* TP1, const int* B0, const int* B1,
                      const int* uids, const int* ucount, const int* mcount, const int* members, float* Z, int mats, int K, int N,
                      int P, int SK, int umax, cudaStream_t st) {
    const dim3 grid((unsigned)(N / 128), (unsigned)(mats * SK), (unsigned)umax);
    return gbp_pdl::launch_pdl(gbp_pdl::on(gbp_pdl::kMoe), moe_gemv_multi_kernel<CB, WK, WMASK>, grid, dim3(WK * 32), 0, st, X0, X1, TP0,
                               TP1, B0, B1, uids, ucount, mcount, members, Z, K, N, P, SK);
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// C ABI

// The plan of a K x N layer: K splits and warps a block, from the SHAPE alone (never from the row count, which would change the reduction
// order). `mats` is the number of matrices one launch runs (2 for gate and up, 1 for down). The rule aims at 64 blocks of 128 columns a
// distinct expert (`nb * mats * sk`), the best of the measured K splits for the GLM shapes (gate/up 4096 -> 2048: 2 splits, down 2048 ->
// 4096: 2 splits; head-to-head at 1..16 rows within 3.6% of each other, see testdata/gpu/moe_tverify.txt), keeps at least 8 k tiles a warp
// and a split that divides. out[0] = SK, out[1] = WK. Returns 0, or cudaErrorInvalidValue for a shape the kernels do not take.
GB_EXPORT int gb_moe_default_plan(int k, int n, int bits, int mats, int* out) {
    if (k <= 0 || n <= 0 || k % 128 || n % 128 || bits < 1 || bits > 8 || mats < 1 || mats > 2 || !out) return (int)cudaErrorInvalidValue;
    const int wk = 4, kt = k / 16, nb = n / 128;
    int sk = 64 / (nb * mats);
    if (sk < 1) sk = 1;
    while (sk > 1 && (kt % (sk * wk) || kt / (sk * wk) < 8)) sk /= 2;
    out[0] = sk;
    out[1] = wk;
    return kt % (sk * wk) ? (int)cudaErrorInvalidValue : 0;
}

// Groups the P = rows * topk picks. uids [min(P, E)], mcount [min(P, E)], members [min(P, E) * P], ucount [1]. E <= 1024.
GB_EXPORT int gb_moe_group(const void* pick, void* uids, void* ucount, void* mcount, void* members, int rows, int topk, int E,
                           void* stream) {
    const int P = rows * topk;
    if (rows < 1 || topk < 1 || topk > kMaxTopK || E < 1 || E > kGroupThreads || !pick || !uids || !ucount || !mcount || !members)
        return (int)cudaErrorInvalidValue;
    return gbp_pdl::launch_pdl(gbp_pdl::on(gbp_pdl::kMoe), group_kernel, dim3(1), dim3(kGroupThreads), (size_t)P * sizeof(int),
                               gb_stream(stream), static_cast<const int*>(pick), static_cast<int*>(uids), static_cast<int*>(ucount),
                               static_cast<int*>(mcount), static_cast<int*>(members), P, E);
}

// out0/out1 [rows * topk, K] fp16: the rotated inputs of gate and up per pair (pairs with a pick outside [0, E) are skipped).
GB_EXPORT int gb_moe_rot_in(const void* x, int x_dtype, const void* pick, const void* suh0, const void* suh1, void* out0,
                            void* out1, int rows, int topk, int K, int E, void* stream) {
    if (rows < 1 || topk < 1 || topk > kMaxTopK || K <= 0 || K % 128 || E < 1 || !dtype_ok(x_dtype) || !x || !pick || !suh0 ||
        !suh1 || !out0 || !out1)
        return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)(rows * topk), (unsigned)((K / 128 + 3) / 4), 2);
    return gbp_pdl::launch_pdl(gbp_pdl::on(gbp_pdl::kMoe), rot_in_kernel, grid, dim3(128), 0, gb_stream(stream), x, x_dtype,
                               static_cast<const int*>(pick), static_cast<const half*>(suh0), static_cast<const half*>(suh1),
                               static_cast<half*>(out0), static_cast<half*>(out1), K, topk, E);
}

// The grouped GEMV of `mats` (1 or 2) matrices: Z [mats][sk][P][n] fp32 partials. tp0/tp1: per-expert trellis device addresses
// (uint64 [E]), each a [K/16, N/16, 16 * bits] int16 trellis. x0/x1 [P, K] fp16. umax = min(P, E).
GB_EXPORT int gb_moe_gemv(const void* x0, const void* x1, const void* tp0, const void* tp1, const void* uids, const void* ucount,
                          const void* mcount, const void* members, void* z, int mats, int K, int N, int P, int bits,
                          int codebook, int sk, int wk, int umax, void* stream) {
    if (mats < 1 || mats > 2 || K <= 0 || N <= 0 || K % 128 || N % 128 || P < 1 || bits < 1 || bits > 8 || codebook < 0 ||
        codebook > 2 || sk < 1 || (wk != 4) || (K / 16) % (sk * wk) || umax < 1 || !x0 || !tp0 || !uids || !ucount || !mcount ||
        !members || !z || (mats == 2 && (!x1 || !tp1)))
        return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    const half* X0 = static_cast<const half*>(x0);
    const half* X1 = static_cast<const half*>(mats == 2 ? x1 : x0);
    const uint64_t* T0 = static_cast<const uint64_t*>(tp0);
    const uint64_t* T1 = static_cast<const uint64_t*>(mats == 2 ? tp1 : tp0);
#define GB_G(K2_, CB_, WK_)                                                                                            \
    if (2 * bits == K2_ && codebook == CB_ && wk == WK_)                                                               \
        return launch_gemv<K2_, CB_, WK_>(X0, X1, T0, T1, static_cast<const int*>(uids), static_cast<const int*>(ucount), \
                                          static_cast<const int*>(mcount), static_cast<const int*>(members),           \
                                          static_cast<float*>(z), mats, K, N, P, sk, umax, st);
    GB_MOE_FOR_ALL(GB_G, 4)
#undef GB_G
    return (int)cudaErrorInvalidValue;
}

// gb_moe_rot_in over several banks: pick [rows * topk] holds global ids (or -1), suh0 / suh1 [E] the device address of each expert's
// suh row (K halves) of gate and of up.
GB_EXPORT int gb_moe_rot_in_multi(const void* x, int x_dtype, const void* pick, const void* suh0, const void* suh1, void* out0,
                                  void* out1, int rows, int topk, int K, int E, void* stream) {
    if (rows < 1 || topk < 1 || topk > kMaxTopK || K <= 0 || K % 128 || E < 1 || !dtype_ok(x_dtype) || !x || !pick || !suh0 ||
        !suh1 || !out0 || !out1)
        return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)(rows * topk), (unsigned)((K / 128 + 3) / 4), 2);
    return gbp_pdl::launch_pdl(gbp_pdl::on(gbp_pdl::kMoe), rot_in_multi_kernel, grid, dim3(128), 0, gb_stream(stream), x, x_dtype,
                               static_cast<const int*>(pick), static_cast<const uint64_t*>(suh0), static_cast<const uint64_t*>(suh1),
                               static_cast<half*>(out0), static_cast<half*>(out1), K, topk, E);
}

// gb_moe_gemv over several banks of different widths in one launch: uids are global ids, tp0 / tp1 [E] each expert's trellis address and
// bits0 / bits1 [E] (i32) its width, of the first and (mats == 2) the second matrix. widths: the mask of the widths present (bit b - 1 for
// b bits), which picks the instantiation. Every block is the block gb_moe_gemv runs at that expert's width with the same sk and wk.
GB_EXPORT int gb_moe_gemv_multi(const void* x0, const void* x1, const void* tp0, const void* tp1, const void* bits0, const void* bits1,
                                const void* uids, const void* ucount, const void* mcount, const void* members, void* z, int mats, int K,
                                int N, int P, int widths, int codebook, int sk, int wk, int umax, void* stream) {
    if (mats < 1 || mats > 2 || K <= 0 || N <= 0 || K % 128 || N % 128 || P < 1 || widths <= 0 || widths > kWidthsAll ||
        codebook < 0 || codebook > 2 || sk < 1 || wk != 4 || (K / 16) % (sk * wk) || umax < 1 || !x0 || !tp0 || !bits0 || !uids ||
        !ucount || !mcount || !members || !z || (mats == 2 && (!x1 || !tp1 || !bits1)))
        return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    const half* X0 = static_cast<const half*>(x0);
    const half* X1 = static_cast<const half*>(mats == 2 ? x1 : x0);
    const uint64_t* T0 = static_cast<const uint64_t*>(tp0);
    const uint64_t* T1 = static_cast<const uint64_t*>(mats == 2 ? tp1 : tp0);
    const int* B0 = static_cast<const int*>(bits0);
    const int* B1 = static_cast<const int*>(mats == 2 ? bits1 : bits0);
    const bool narrow = (widths & ~kWidths456) == 0;
#define GB_GM(CB_)                                                                                                     \
    if (codebook == CB_)                                                                                               \
        return narrow ? launch_gemv_multi<CB_, 4, kWidths456>(X0, X1, T0, T1, B0, B1, static_cast<const int*>(uids),   \
                                                              static_cast<const int*>(ucount),                          \
                                                              static_cast<const int*>(mcount),                          \
                                                              static_cast<const int*>(members), static_cast<float*>(z), \
                                                              mats, K, N, P, sk, umax, st)                              \
                      : launch_gemv_multi<CB_, 4, kWidthsAll>(X0, X1, T0, T1, B0, B1, static_cast<const int*>(uids),   \
                                                              static_cast<const int*>(ucount),                          \
                                                              static_cast<const int*>(mcount),                          \
                                                              static_cast<const int*>(members), static_cast<float*>(z), \
                                                              mats, K, N, P, sk, umax, st);
    GB_GM(0) GB_GM(1) GB_GM(2)
#undef GB_GM
    return (int)cudaErrorInvalidValue;
}

// gb_moe_gateup_epilogue over several banks: tab [3][E] the device addresses of each expert's svh_g, svh_u and suh_d rows (N halves
// each), pick global ids (or -1).
GB_EXPORT int gb_moe_gateup_epilogue_multi(const void* z, const void* pick, const void* tab, void* xd, int rows, int topk, int N, int sk,
                                           int E, float limit, void* stream) {
    const int P = rows * topk;
    if (rows < 1 || topk < 1 || topk > kMaxTopK || N <= 0 || N % 128 || sk < 1 || E < 1 || !z || !pick || !tab || !xd)
        return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)P, (unsigned)((N / 128 + 3) / 4));
    return gbp_pdl::launch_pdl(gbp_pdl::on(gbp_pdl::kMoe), gateup_epilogue_multi_kernel, grid, dim3(128), 0, gb_stream(stream),
                               static_cast<const float*>(z), static_cast<const int*>(pick), static_cast<const uint64_t*>(tab),
                               static_cast<half*>(xd), P, N, sk, E, limit);
}

// y_pairs [rows * topk, D] fp32: each pair's y (the down epilogue of gb_moe_down_combine, no combine) for every pair whose pick (a
// global id) is in [0, E); svh [E] the device address of each expert's svh_down row (D halves). Other rows are left alone.
GB_EXPORT int gb_moe_down_pairs_multi(const void* z, const void* pick, const void* svh, void* y_pairs, int rows, int topk, int D, int sk,
                                      int E, void* stream) {
    const int P = rows * topk;
    if (rows < 1 || topk < 1 || topk > kMaxTopK || D <= 0 || D % 128 || sk < 1 || E < 1 || !z || !pick || !svh || !y_pairs)
        return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)P, (unsigned)((D / 128 + 3) / 4));
    return gbp_pdl::launch_pdl(gbp_pdl::on(gbp_pdl::kMoe), down_pairs_multi_kernel, grid, dim3(128), 0, gb_stream(stream),
                               static_cast<const float*>(z), static_cast<const int*>(pick), static_cast<const uint64_t*>(svh),
                               static_cast<float*>(y_pairs), P, D, sk, E);
}

// xd [P, N] fp16 from the gate/up partials Z [2][sk][P][N] (N = the expert width I). limit <= 0: no clamp.
GB_EXPORT int gb_moe_gateup_epilogue(const void* z, const void* pick, const void* svh_g, const void* svh_u, const void* suh_d,
                                     void* xd, int rows, int topk, int N, int sk, int E, float limit, void* stream) {
    const int P = rows * topk;
    if (rows < 1 || topk < 1 || topk > kMaxTopK || N <= 0 || N % 128 || sk < 1 || E < 1 || !z || !pick || !svh_g || !svh_u ||
        !suh_d || !xd)
        return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)P, (unsigned)((N / 128 + 3) / 4));
    return gbp_pdl::launch_pdl(gbp_pdl::on(gbp_pdl::kMoe), gateup_epilogue_kernel, grid, dim3(128), 0, gb_stream(stream),
                               static_cast<const float*>(z), static_cast<const int*>(pick), static_cast<const half*>(svh_g),
                               static_cast<const half*>(svh_u), static_cast<const half*>(suh_d), static_cast<half*>(xd), P, N, sk, E,
                               limit);
}

// out [rows, D] fp32 = sum over slots in order of w * y, then + shared [rows, D] fp32 (null: nothing added). y_pairs [P, D] fp32
// receives each slot's y when not null.
GB_EXPORT int gb_moe_down_combine(const void* z, const void* pick, const void* svh_d, const void* wts, const void* shared,
                                  void* y_pairs, void* out, int rows, int topk, int D, int sk, int E, void* stream) {
    const int P = rows * topk;
    if (rows < 1 || topk < 1 || topk > kMaxTopK || D <= 0 || D % 128 || sk < 1 || E < 1 || !z || !pick || !svh_d || !wts || !out)
        return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)rows, (unsigned)(D / 128));
    return gbp_pdl::launch_pdl(gbp_pdl::on(gbp_pdl::kMoe), down_combine_kernel, grid, dim3(32 * topk), 0, gb_stream(stream),
                               static_cast<const float*>(z), static_cast<const int*>(pick), static_cast<const half*>(svh_d),
                               static_cast<const half*>(wts), static_cast<const float*>(shared), static_cast<float*>(y_pairs),
                               static_cast<float*>(out), P, D, sk, E, topk);
}

// a [n] fp16 = swiglu with the limit of g [n] fp32 and u [n] fp32 (limit <= 0: no clamp).
GB_EXPORT int gb_moe_swiglu_limit(const void* g, const void* u, void* a, long long n, float limit, void* stream) {
    if (n < 0 || (n > 0 && (!g || !u || !a))) return (int)cudaErrorInvalidValue;
    if (n == 0) return 0;
    swiglu_limit_kernel<<<(unsigned)((n + 255) / 256), 256, 0, gb_stream(stream)>>>(static_cast<const float*>(g), static_cast<const float*>(u),
                                                                                    static_cast<half*>(a), n, limit);
    return GB_CHECK_LAUNCH();
}

// a [n] fp16 = silu(min(g, L)) * clamp(u, -L, L) of g [n] fp32 and u [n] fp32 (the Ling-3.1-flash-1019 clamp;
// limit <= 0: no clamp).
GB_EXPORT int gb_moe_swiglu_clamped(const void* g, const void* u, void* a, long long n, float limit, void* stream) {
    if (n < 0 || (n > 0 && (!g || !u || !a))) return (int)cudaErrorInvalidValue;
    if (n == 0) return 0;
    swiglu_clamped_kernel<<<(unsigned)((n + 255) / 256), 256, 0, gb_stream(stream)>>>(static_cast<const float*>(g), static_cast<const float*>(u),
                                                                                      static_cast<half*>(a), n, limit);
    return GB_CHECK_LAUNCH();
}

