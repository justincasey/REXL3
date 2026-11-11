// gb-nvcc: --expt-relaxed-constexpr
// The EXL3 linear layer for LARGE row counts (a prefill chunk: 1 .. 65,535 rows), with the arithmetic of the decode path (exl3_linear.cu's
// gb_exl3_linear, 1..128 rows) reproduced BIT FOR BIT: a row's output equals what the decode kernel gives that row alone or in any window.
//
// The decode kernel handles at most 128 rows a call and, for a layer whose N is too narrow to fill the GPU, splits K and writes fp32 partials
// that a second kernel adds in split order; at thousands of rows those partials are most of its memory traffic, and its blocks loop over
// every 64-row group of the call one after another. This kernel keeps the decode kernel's reduction TREE and drops the traffic:
//   * (K split SK, warps a block WK) come from the shape alone (the same plan_for), each warp's k range, its accumulators' chains over it and
//     the warp-order sums are exactly the decode kernel's (exl3_grouped.cuh);
//   * a block owns one (64-row group, 128-column strip) and runs ALL K splits itself, adding each split's warp sums to a running sum in shared
//     memory in split order (the decode path's `c = Z[0]; c += Z[1]; ...` in rot_out_kernel);
//   * the epilogue is the decode kernel's `finish` (H / sqrt(128), * svh, + bias) on that sum;
//   * the A operand is the rotated input in fragment order (the decode path's rot_in_frag_kernel), a copy of which is in this file.
// mma keeps rows apart, so the group a row falls in, how many rows the call has and what the other rows hold change no bit. This file is
// compiled with the flags of exl3_linear.cu (no --fmad=false) so that `finish` compiles as it does there.
//
// Derived from csrc/exl3_linear.cu (its header names the TensorFold kernel it was ported from: MIT, pinned 9cd52ab, over ExLlamaV3's decode, MIT,
// Copyright (c) 2025 Turboderp; see THIRD_PARTY_NOTICES.md). The tile decode is csrc/exl3_linear_decode.cuh.

#include <stdint.h>
#include <stdlib.h>

#include "common.cuh"
#include "exl3_grouped.cuh"

using namespace gbx;
using namespace gbp;

namespace {

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

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

__device__ __forceinline__ void store4(void* p, int dtype, size_t i, const float (&v)[4]) {
    if (dtype == F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (dtype == BF16) {
        __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p) + i) = u;
    } else {
        half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<half*>(p) + i) = u;
    }
}

// The finished outputs of one row's 128 columns from their fp32 sums (4 a lane): H / sqrt(128), * svh, + bias (exl3_linear.cu's `finish`).
__device__ __forceinline__ void finish(float (&v)[4], int lane, const half* svh, const half* bias, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = v[j] * HAD_SCALE * __half2float(__ldg(svh + col + j));
        if (bias) v[j] += __half2float(__ldg(bias + col + j));
    }
}

// exl3_linear.cu's rot_in_frag_kernel: xf [ceil(M / 16)][K / 16][32 lanes][4 words], the rotated input in the mma's A-fragment order; rows past
// M repeat row M - 1 (their outputs are dropped).
__global__ void __launch_bounds__(128) rot_in_frag_kernel(const void* __restrict__ x, int x_dtype, const half* __restrict__ suh,
                                                          uint32_t* __restrict__ xf, int K, int M) {
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5), row = blockIdx.x, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(x, x_dtype, (size_t)min(row, M - 1) * K + k, v);
    load4(suh, F16, k, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    const half2 pair[2] = {__floats2half2_rn(v[0], v[1]), __floats2half2_rn(v[2], v[3])};
    const int mt = row >> 4, g = row & 7, rh = (row >> 3) & 1, KT = K >> 4;
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int col = k + 2 * q, kt = col >> 4, kk = col & 15;
        xf[(((size_t)mt * KT + kt) * 32 + g * 4 + ((kk & 7) >> 1)) * 4 + 2 * (kk >> 3) + rh] =
            *reinterpret_cast<const uint32_t*>(&pair[q]);
    }
}

constexpr int kTiles = 4;                                    // 16-row tiles that share a decode: a 64-row group

// Block (group of up to 64 rows, 128-column strip). The group index runs fastest so that the blocks that read one strip's weights run together.
template <int WK, int K2, int CB, int H, int PD, int AP>
// stride_k: words between tile rows of T ((N / 16) * TW for a layer stored whole; a column strip of a wider parent read in place passes its
// parent's, with T, svh and bias at the strip's first block: gb_exl3_linear_big_ex).
__global__ void __launch_bounds__(WK * 32, 1) big_kernel(const uint32_t* __restrict__ xf, const uint32_t* __restrict__ T,
                                                         const half* __restrict__ svh, const half* __restrict__ bias,
                                                         void* __restrict__ y, int y_dtype, int M, int K, int N, int SK,
                                                         int stride_n, int strips_fast) {
    constexpr int NT = kTiles;
    constexpr int TW = tile_words<K2>();
    constexpr int RC = (8 / H) * 16;
    extern __shared__ __align__(16) float smem[];
    float* red = smem;                                       // WK * 8 * RC floats
    float* stage = smem + WK * 8 * RC;                       // 16 NT * 128 floats

    const int m0 = (strips_fast ? blockIdx.y : blockIdx.x) * 16 * NT, nb = strips_fast ? blockIdx.x : blockIdx.y;
    const int nt = min(NT, (M - m0 + 15) >> 4);
    const int rows = min(16 * nt, M - m0);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int KT = K >> 4;
    const uint32_t* xp[NT];
#pragma unroll
    for (int p = 0; p < NT; ++p) xp[p] = xf + (size_t)((m0 >> 4) + (p < nt ? p : 0)) * KT * 128 + lane * 4;

    group_sums<WK, K2, CB, NT, H, PD, AP>(T + (size_t)nb * 8 * TW, (long long)(stride_n >> 4) * TW, xp, KT / SK / WK, SK, nt, red, stage, threadIdx.x);

    const int col0 = nb * 128;
    for (int r = warp; r < rows; r += WK) {
        const float4 u = *reinterpret_cast<const float4*>(stage + r * 128 + 4 * lane);
        float v[4] = {u.x, u.y, u.z, u.w};
        finish(v, lane, svh, bias, col0 + 4 * lane);
        store4(y, y_dtype, (size_t)(m0 + r) * N + col0 + 4 * lane, v);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The second form (big2_kernel), for the Flash-Next shared expert's two shapes at 1,024 rows and more: a block (a 64-row group, a 128-column strip) of 8 warps
// that own 16 columns each. The first form's warps own K quarters of all 128 columns and meet through shared memory (two barriers per 8 rows of every
// column sweep of every split, 128 barriers a block for the gate/up shape); here a warp runs every chain itself and no warp waits for another but at
// the stage barrier of a cp.async ring. Per split s and quarter q the warp's accumulators are one fresh fp32 mma chain from +0.0 over the quarter's k tiles
// (`per` of them, in order); the quarters of a split are added in order (sp = c0; sp += c1; sp += c2; sp += c3), the splits' sums to a running sum in
// split order (tot = sp of split 0; tot += sp of the others): the values reduce_sweep and group_sums build, element by element, so every output is the
// first form's bit for bit. A and the trellis stream through the ring (the A fragments of the group's tiles once for all 8 warps), the tile of a
// (k tile, column tile) is decoded by the one warp that owns the column tile.

__device__ __forceinline__ void cpa16(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cpa_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

constexpr int kLds2 = 128 + 4;     // row stride (floats) of the staging of the sums: 16-byte rows, the 8 rows of a fragment hit different banks
constexpr int kGuard2 = 4;         // tiles of a group

// One ring stage of the block's 256 threads: the A fragments of the group's first `nt` tiles (KPS k tiles each) and the KPS x 8 trellis tiles of the strip
// (`T`: the strip's first tile, `stride_k`: words between k tile rows).
template <int K2, int MT, int KPS>
__device__ __forceinline__ void load_stage2(uint32_t* sa, const uint32_t* xf, const uint32_t* T, long long stride_k, int tile0, int nt, int KT, int kc,
                                            int tt) {
    constexpr int TW = tile_words<K2>();
    constexpr int AU = MT * KPS * 32;
#pragma unroll
    for (int i = 0; i < (AU + 255) / 256; ++i) {
        const int u = i * 256 + tt;
        if (u < AU) {
            const int p = u / (KPS * 32), kk = (u / 32) % KPS, ln = u & 31;
            if (p < nt) cpa16(sa + (size_t)((p * KPS + kk) * 32 + ln) * 4, xf + ((size_t)(tile0 + p) * KT + kc * KPS + kk) * 128 + ln * 4);
        }
    }
    uint32_t* sb = sa + MT * KPS * 128;
    constexpr int BU = KPS * 8 * (TW / 4);
#pragma unroll
    for (int i = 0; i < (BU + 255) / 256; ++i) {
        const int u = i * 256 + tt;
        if (u < BU) {
            const int c = u % (TW / 4), tl = u / (TW / 4), kk = tl >> 3, tc = tl & 7;
            cpa16(sb + (size_t)tl * TW + c * 4, T + (size_t)(kc * KPS + kk) * stride_k + (size_t)tc * TW + c * 4);
        }
    }
}

// mma with the accumulator +0.0 as C: the first step of a chain, which starts from +0.0 (no register is zeroed, nothing is copied).
__device__ __forceinline__ void mma16816_z(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "{\n\t.reg .f32 z;\n\tmov.f32 z, 0f00000000;\n\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {z,z,z,z};\n}"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

constexpr int kPer2 = 10;                               // k tiles of a chain (a warp quarter of one split): both shared-expert shapes

// One warp's chains over all k tiles of the block (see above). SPLIT: the layer has K splits (a separate per-split sum), else the quarters' sum is the whole.
// `chain(d)` runs one quarter's kPer2 k tiles (a whole number of ring stages) into d from +0.0: the first quarter of a split goes straight into the sum it
// starts, the others into `acc` and are added to it. Leaves the warp's fp32 sums (16 columns x MT * 16 rows, in the mma accumulator layout) in `tot`.
template <int K2, int CB, int MT, int KPS, int ST, bool SPLIT>
__device__ __forceinline__ void chains_b(float (&tot)[MT][2][4], uint32_t* sm, const uint32_t* xf, const uint32_t* T, long long stride_k, int tile0, int nt, int KT,
                                         int SK, int tt, int warp, int lane) {
    constexpr int TW = tile_words<K2>();
    constexpr int STAGE_W = MT * KPS * 128 + KPS * 8 * TW;
    constexpr int CLS = kPer2 / KPS;                    // stages of one chain
    const int NS = KT / KPS;
    float acc[MT][2][4];
    float sp[SPLIT ? MT : 1][SPLIT ? 2 : 1][SPLIT ? 4 : 1];
#pragma unroll
    for (int p = 0; p < MT; ++p)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                acc[p][h][c] = 0.f;
                tot[p][h][c] = 0.f;
                if constexpr (SPLIT) sp[p][h][c] = 0.f;
            }
#pragma unroll
    for (int s = 0; s < ST - 1; ++s) {
        if (s < NS) load_stage2<K2, MT, KPS>(sm + s * STAGE_W, xf, T, stride_k, tile0, nt, KT, s, tt);
        cpa_commit();
    }
    int kc = 0;
    auto chain = [&](float (&d)[MT][2][4]) {
#pragma unroll
        for (int i = 0; i < CLS; ++i) {
            cpa_wait<ST - 2>();
            __syncthreads();
            {
                const int nk = kc + ST - 1;
                if (nk < NS) load_stage2<K2, MT, KPS>(sm + (nk % ST) * STAGE_W, xf, T, stride_k, tile0, nt, KT, nk, tt);
                cpa_commit();
            }
            const uint32_t* sa = sm + (kc % ST) * STAGE_W;
            const uint32_t* sb = sa + MT * KPS * 128;
#pragma unroll
            for (int kk = 0; kk < KPS; ++kk) {
                uint32_t w[lane_words<K2>()], b0[2], b1[2];
                load_lane_words<K2>(sb + (kk * 8 + warp) * TW, lane, w);
                decode_lane<K2, CB>(w, lane, b0, b1);
#pragma unroll
                for (int p = 0; p < MT; ++p) {
                    if (p < nt) {
                        const uint4 av = *reinterpret_cast<const uint4*>(sa + (size_t)((p * KPS + kk) * 32 + lane) * 4);
                        const uint32_t a[4] = {av.x, av.y, av.z, av.w};
                        if (i == 0 && kk == 0) {
                            mma16816_z(d[p][0], a, b0);
                            mma16816_z(d[p][1], a, b1);
                        } else {
                            mma16816(d[p][0], a, b0);
                            mma16816(d[p][1], a, b1);
                        }
                    }
                }
            }
            ++kc;
        }
    };
    auto fold = [&](float (&dst)[MT][2][4], const float (&src)[MT][2][4]) {
#pragma unroll
        for (int p = 0; p < MT; ++p)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) dst[p][h][c] += src[p][h][c];
    };
    if constexpr (!SPLIT) {
        chain(tot);
#pragma unroll 1
        for (int q = 1; q < 4; ++q) {
            chain(acc);
            fold(tot, acc);
        }
    } else {
#pragma unroll 1
        for (int s = 0; s < SK; ++s) {
            if (s == 0) chain(tot);
            else chain(sp);
#pragma unroll 1
            for (int q = 1; q < 4; ++q) {
                chain(acc);
                if (s == 0) fold(tot, acc);
                else fold(sp, acc);
            }
            if (s > 0) fold(tot, sp);
        }
    }
    cpa_wait<0>();
}

template <int K2, int KPS, int ST>
constexpr size_t smem_big2() {
    constexpr size_t stage = (size_t)KPS * kTiles * 128 + (size_t)KPS * 8 * tile_words<K2>();
    constexpr size_t pipe = (size_t)ST * stage, epi = (size_t)kTiles * 16 * kLds2;
    return (pipe > epi ? pipe : epi) * 4;
}

// Block (group of up to 64 rows, 128-column strip); `strips_fast` as big_kernel's. KPS k tiles a ring stage, ST stages, MINB blocks an SM.
template <int K2, int CB, bool SPLIT, int KPS, int ST, int MINB>
__global__ void __launch_bounds__(256, MINB) big2_kernel(const uint32_t* __restrict__ xf, const uint32_t* __restrict__ T, const half* __restrict__ svh,
                                                          const half* __restrict__ bias, void* __restrict__ y, int y_dtype, int M, int K, int N, int SK,
                                                          int strips_fast) {
    constexpr int MT = kTiles, TW = tile_words<K2>();
    extern __shared__ __align__(16) uint32_t smem2[];
    const int m0 = (strips_fast ? blockIdx.y : blockIdx.x) * 16 * MT, nb = strips_fast ? blockIdx.x : blockIdx.y;
    const int nt = min(MT, (M - m0 + 15) >> 4);
    const int rows = min(16 * nt, M - m0);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int KT = K >> 4;
    const long long stride_k = (long long)(N >> 4) * TW;
    float tot[MT][2][4];
    chains_b<K2, CB, MT, KPS, ST, SPLIT>(tot, smem2, xf, T + (size_t)nb * 8 * TW, stride_k, m0 >> 4, nt, KT, SK, tid, warp, lane);
    __syncthreads();                                     // the ring is the staging now
    float* stg = reinterpret_cast<float*>(smem2);
    {
        const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
        for (int p = 0; p < MT; ++p) {
            if (p < nt) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = warp * 16 + h * 8 + 2 * t4, r = p * 16 + g;
                    *reinterpret_cast<float2*>(stg + r * kLds2 + col) = make_float2(tot[p][h][0], tot[p][h][1]);
                    *reinterpret_cast<float2*>(stg + (r + 8) * kLds2 + col) = make_float2(tot[p][h][2], tot[p][h][3]);
                }
            }
        }
    }
    __syncthreads();
    const int col0 = nb * 128;
    for (int r = warp; r < rows; r += 8) {
        const float4 u = *reinterpret_cast<const float4*>(stg + r * kLds2 + 4 * lane);
        float v[4] = {u.x, u.y, u.z, u.w};
        finish(v, lane, svh, bias, col0 + 4 * lane);
        store4(y, y_dtype, (size_t)(m0 + r) * N + col0 + 4 * lane, v);
    }
}

// (K splits, warps a block) for a K x N layer: exl3_linear.cu's plan_for, the shape's alone. It must stay identical to that function: a
// different plan sums K in another order, and the bit-identity tests (every shape of the GLM text model) would fail.
struct Plan { int sk, wk; };

Plan plan_for(int k, int n) {
    const int blocks = 192, min_tiles = 8;
    const int kt = k / 16, nb = n / 128;
    int sk = 1, wk = nb >= 64 ? 8 : 4;
    if (nb < 64) {                                // a narrow layer needs K splits to fill the SMs at all
        while (nb * sk < blocks && sk < 64) sk *= 2;
    }
    while ((wk > 4 || sk > 1) && (kt % (sk * wk) || kt / (sk * wk) < min_tiles)) {
        if (wk > 4) wk = 4;
        else if (sk > 1) sk /= 2;
        else break;
    }
    return {sk, wk};
}

size_t align256(size_t v) { return (v + 255) & ~size_t(255); }

// bytes of the rotated input in fragment order: whole 16-row tiles
size_t xf_bytes(int m, int k) { return align256((size_t)((m + 15) / 16) * 16 * k * sizeof(half)); }

bool shape_ok(int k, int n, int bits, int codebook) {
    return k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0 && bits >= 1 && bits <= 8 && codebook >= 0 && codebook <= 2;
}

bool dtype_ok(int d) { return d >= 0 && d <= 2; }

template <int WK, int K2, int CB, int PD, int AP>
int launch_big_pa(const uint32_t* xf, const uint32_t* T, const half* svh, const half* bias, void* y, int y_dtype, int M, int K, int N, int SK,
                  long long stride_n, cudaStream_t st) {
    constexpr int H = sweeps_of<K2, CB>(kTiles);
    constexpr int RC = (8 / H) * 16;
    const size_t smem = ((size_t)WK * 8 * RC + (size_t)16 * kTiles * 128) * sizeof(float);
    // A narrow layer (at most 24 strips of 128 columns: its whole weight set is a few MB) runs the strips of a row group together, so that the
    // group's rotated input is read from DRAM once and the weights stay in L2; a wide one runs the row groups of a strip together (the weights
    // are what is streamed). Which block runs when never changes a bit.
    const bool strips_fast = N / 128 <= 24;
    const unsigned groups = (unsigned)((M + 16 * kTiles - 1) / (16 * kTiles)), strips = (unsigned)(N / 128);
    const dim3 grid(strips_fast ? strips : groups, strips_fast ? groups : strips);
    big_kernel<WK, K2, CB, H, PD, AP><<<grid, WK * 32, smem, st>>>(xf, T, svh, bias, y, y_dtype, M, K, N, SK,
                                                                  (int)stride_n, strips_fast ? 1 : 0);
    return GB_CHECK_LAUNCH();
}

// The loads of the widths the pack uses (2, 3 and 4 bits, the MUL1 codebook) run the weights two steps ahead and the A fragments one step ahead (measured
// on the GB10 at 2,048 rows: KDA qkv 7.0 -> 6.3 ms, MLA o 6.0 -> 4.8, the shared expert's gate/up 0.95 -> 0.78); the other widths keep the plain loop.
template <int WK, int K2, int CB>
int launch_big(const uint32_t* xf, const uint32_t* T, const half* svh, const half* bias, void* y, int y_dtype, int M, int K, int N, int SK,
               long long stride_n, cudaStream_t st) {
    constexpr bool tuned = (K2 == 4 || K2 == 6 || K2 == 8) && CB == 2;
    return launch_big_pa<WK, K2, CB, tuned ? 2 : 1, tuned ? 1 : 0>(xf, T, svh, bias, y, y_dtype, M, K, N, SK, stride_n, st);
}

// `REXL3_BIG_SECOND_FORM=0`: the Flash-Next shared expert's two shapes run the first form (big_kernel) too. gb_exl3_big_set_second_form overrides it (tests
// that run both forms in one process); -1 gives the environment back.
static int g_second_override = -1;
inline bool second_form_on() {
    static const bool env = [] { const char* e = getenv("REXL3_BIG_SECOND_FORM"); return !(e && e[0] == '0'); }();
    return g_second_override >= 0 ? g_second_override != 0 : env;
}

// Which calls the second form takes: the MUL1 codebook at 4 .. 8 bits, 1,024 rows or more (the decode windows and every other family keep big_kernel), the
// Flash-Next shared expert's two shapes (gate/up 2,560 x 640: 4 K splits; down 640 x 2,560: one split), 4 warps in the plan and chains of whole stages.
bool second_form_takes(int m, int k, int n, int bits, int codebook, const Plan& p) {
    return second_form_on() && m >= 1024 && codebook == 2 && bits >= 4 && bits <= 8 && p.wk == 4 && ((k == 2560 && n == 640) || (k == 640 && n == 2560)) &&
           (k / 16) / p.sk / 4 == kPer2;
}

template <int K2, int CB, bool SPLIT, int KPS, int ST, int MINB>
int launch_big2_cfg(const uint32_t* xf, const uint32_t* T, const half* svh, const half* bias, void* y, int y_dtype, int M, int K, int N, int SK,
                    cudaStream_t st) {
    constexpr size_t smem = smem_big2<K2, KPS, ST>();
    auto kernel = big2_kernel<K2, CB, SPLIT, KPS, ST, MINB>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    const bool strips_fast = N / 128 <= 24;
    const unsigned groups = (unsigned)((M + 16 * kTiles - 1) / (16 * kTiles)), strips = (unsigned)(N / 128);
    const dim3 grid(strips_fast ? strips : groups, strips_fast ? groups : strips);
    kernel<<<grid, 256, smem, st>>>(xf, T, svh, bias, y, y_dtype, M, K, N, SK, strips_fast ? 1 : 0);
    return GB_CHECK_LAUNCH();
}

// The ring per shape (MEASURED on the GB10, the shared expert's shapes at 8,192 rows): a layer with K splits keeps 233 registers a thread (the quarter sum, the
// split sum and the running sum are all live) and so one 8-warp block an SM, with a whole chain (10 k tiles) in a stage and two stages; the one-split layer fits
// 128 registers, two blocks an SM, and stages of half a chain.
template <int K2, int CB, bool SPLIT>
int launch_big2(const uint32_t* xf, const uint32_t* T, const half* svh, const half* bias, void* y, int y_dtype, int M, int K, int N, int SK, cudaStream_t st) {
    if constexpr (SPLIT) return launch_big2_cfg<K2, CB, SPLIT, 10, 2, 1>(xf, T, svh, bias, y, y_dtype, M, K, N, SK, st);
    return launch_big2_cfg<K2, CB, SPLIT, 5, 2, 2>(xf, T, svh, bias, y, y_dtype, M, K, N, SK, st);
}

#define GB_BIG_FOR_CB(X, K2) X(K2, 0) X(K2, 1) X(K2, 2)
#define GB_BIG_FOR_ALL(X)                                                                                               \
    GB_BIG_FOR_CB(X, 2) GB_BIG_FOR_CB(X, 4) GB_BIG_FOR_CB(X, 6) GB_BIG_FOR_CB(X, 8) GB_BIG_FOR_CB(X, 10) GB_BIG_FOR_CB(X, 12) \
    GB_BIG_FOR_CB(X, 14) GB_BIG_FOR_CB(X, 16)

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// C ABI

// Bytes of workspace gb_exl3_linear_big needs (the rotated input in fragment order); contents on entry do not matter. 0: an unsupported shape.
GB_EXPORT size_t gb_exl3_big_workspace_bytes(int m, int k, int n, int bits) {
    if (m < 1 || !shape_ok(k, n, bits, 0)) return 0;
    return xf_bytes(m, k);
}

// The K split and warps a block of the layer (the decode path's, exposed for the tests): out[0] = SK, out[1] = WK.
GB_EXPORT int gb_exl3_big_plan(int k, int n, int* out) {
    if (!shape_ok(k, n, 1, 0) || !out) return (int)cudaErrorInvalidValue;
    const Plan p = plan_for(k, n);
    out[0] = p.sk;
    out[1] = p.wk;
    return 0;
}

// Test hook: 1 / 0 force the second form / the first form for the shared expert's shapes for the calls that follow, -1 returns to REXL3_BIG_SECOND_FORM.
GB_EXPORT int gb_exl3_big_set_second_form(int v) {
    g_second_override = v < 0 ? -1 : (v ? 1 : 0);
    return 0;
}

// y [m, n] = EXL3 layer applied to x [m, k] for any m >= 1, the decode kernel's bits (see the header). bias may be null. All device pointers.
GB_EXPORT int gb_exl3_linear_big(const void* x, int x_dtype, const void* trellis, const void* suh, const void* svh, const void* bias,
                                 void* y, int y_dtype, int m, int k, int n, int bits, int codebook, void* workspace,
                                 size_t workspace_bytes, void* stream) {
    if (m < 1 || !shape_ok(k, n, bits, codebook) || !dtype_ok(x_dtype) || !dtype_ok(y_dtype) || !x || !trellis || !suh || !svh || !y ||
        !workspace || workspace_bytes < xf_bytes(m, k))
        return (int)cudaErrorInvalidValue;
    const Plan p = plan_for(k, n);
    cudaStream_t st = gb_stream(stream);
    uint32_t* xf = static_cast<uint32_t*>(workspace);
    {
        const dim3 grid((unsigned)((m + 15) / 16 * 16), (unsigned)((k / 128 + 3) / 4));
        rot_in_frag_kernel<<<grid, 128, 0, st>>>(x, x_dtype, static_cast<const half*>(suh), xf, k, m);
        const int rc = GB_CHECK_LAUNCH();
        if (rc) return rc;
    }
    if (second_form_takes(m, k, n, bits, codebook, p)) {
        const uint32_t* T2 = static_cast<const uint32_t*>(trellis);
        const half* S2 = static_cast<const half*>(svh);
        const half* B2 = static_cast<const half*>(bias);
#define GB_L2(K2_)                                                                                                              \
    if (2 * bits == K2_)                                                                                                        \
        return p.sk > 1 ? launch_big2<K2_, 2, true>(xf, T2, S2, B2, y, y_dtype, m, k, n, p.sk, st)                              \
                        : launch_big2<K2_, 2, false>(xf, T2, S2, B2, y, y_dtype, m, k, n, p.sk, st);
        GB_L2(8) GB_L2(10) GB_L2(12) GB_L2(14) GB_L2(16)
#undef GB_L2
    }
    const uint32_t* T = static_cast<const uint32_t*>(trellis);
    const half* S = static_cast<const half*>(svh);
    const half* B = static_cast<const half*>(bias);
#define GB_L(K2_, CB_)                                                                                                   \
    if (2 * bits == K2_ && codebook == CB_) {                                                                            \
        if (p.wk == 8) return launch_big<8, K2_, CB_>(xf, T, S, B, y, y_dtype, m, k, n, p.sk, n, st);                    \
        return launch_big<4, K2_, CB_>(xf, T, S, B, y, y_dtype, m, k, n, p.sk, n, st);                                   \
    }
    GB_BIG_FOR_ALL(GB_L)
#undef GB_L
    return (int)cudaErrorInvalidValue;
}

// gb_exl3_linear_big with the plan (sk, wk) (0, 0: plan_for(k, n)) and the trellis rows stride_n columns wide: a column strip of a wider
// parent read in place (stride_n = the parent's N, trellis / svh / bias at the strip's first 128-column block) with the parent's plan gives
// the parent's columns bit for bit, as the decode path's gb_exl3_linear_ex does (lane E, the TP splits; exl3_linear.cu).
GB_EXPORT int gb_exl3_linear_big_ex(const void* x, int x_dtype, const void* trellis, int stride_n, const void* suh, const void* svh,
                                    const void* bias, void* y, int y_dtype, int m, int k, int n, int bits, int codebook, int sk, int wk,
                                    void* workspace, size_t workspace_bytes, void* stream) {
    if (m < 1 || !shape_ok(k, n, bits, codebook) || !dtype_ok(x_dtype) || !dtype_ok(y_dtype) || !x || !trellis || !suh || !svh || !y ||
        !workspace || workspace_bytes < xf_bytes(m, k) || stride_n < n || stride_n % 128)
        return (int)cudaErrorInvalidValue;
    const Plan p = (sk == 0 && wk == 0) ? plan_for(k, n) : Plan{sk, wk};
    if (p.sk < 1 || p.sk > 64 || (p.wk != 4 && p.wk != 8) || (k / 16) % (p.sk * p.wk)) return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    uint32_t* xf = static_cast<uint32_t*>(workspace);
    {
        const dim3 grid((unsigned)((m + 15) / 16 * 16), (unsigned)((k / 128 + 3) / 4));
        rot_in_frag_kernel<<<grid, 128, 0, st>>>(x, x_dtype, static_cast<const half*>(suh), xf, k, m);
        const int rc = GB_CHECK_LAUNCH();
        if (rc) return rc;
    }
    const uint32_t* T = static_cast<const uint32_t*>(trellis);
    const half* S = static_cast<const half*>(svh);
    const half* B = static_cast<const half*>(bias);
#define GB_L(K2_, CB_)                                                                                                   \
    if (2 * bits == K2_ && codebook == CB_) {                                                                            \
        if (p.wk == 8) return launch_big<8, K2_, CB_>(xf, T, S, B, y, y_dtype, m, k, n, p.sk, stride_n, st);             \
        return launch_big<4, K2_, CB_>(xf, T, S, B, y, y_dtype, m, k, n, p.sk, stride_n, st);                            \
    }
    GB_BIG_FOR_ALL(GB_L)
#undef GB_L
    return (int)cudaErrorInvalidValue;
}
