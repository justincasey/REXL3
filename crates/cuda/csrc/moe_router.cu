// gb-nvcc: --fmad=false
// MoE router for the sigmoid-with-bias family (GLM-5.3-Flash: 288 routed experts, top-8, scale 2.5), in two kernels:
//
//   1. router_logits_kernel: logits [rows, E] fp16 = x [rows, K] . gate [E, K]^T. ONE arithmetic for every row count: the
//      per-(row, expert) dot product is ExLlamaV3's single-row router GEMV (MIT, Copyright (c) 2025 Turboderp, `routing.cu`
//      `routing_gemv_kernel`): lane j of a warp runs two explicit fmaf per half2 pair over pairs j, j + 32, ..., then a
//      shfl_down tree 16 -> 1, then one rounding to fp16. The oracle uses that kernel for one row and cuBLAS hgemm for more
//      (so its logits depend on the row count); here a warp may carry several rows (RB) but every row keeps its own identical
//      chain, so a row's logits are the same bits at 1 row and at 128, and equal the oracle's one-row logits.
//   2. router_select_kernel: one warp per row. key[e] = sigmoid(logit[e]) + bias[e] in fp32 (sigmoid from a 65,536-entry
//      table indexed by the fp16 logit bits, built on the host in float64 with one rounding, so the host mirror is exact);
//      top-K by key, ties to the LOWER expert id, in rank order; weights = half(act * scale / (tree_sum(act of the K) + 1e-20)).
//
// Everything is IEEE (no fast-math, no FMA contraction: this file is built with --fmad=false and every fused multiply-add
// is an explicit fmaf), no atomics, and a row's output depends on that row alone. docs: crates/cuda/src/moe.rs has the
// host mirror these kernels are tested against bit for bit.
//
// Where this differs from the oracle (routing.cu warp_topk_shared): the oracle breaks an exact key tie towards the lowest
// LANE (id % 32) and then the lowest id within the lane; this file breaks it towards the lowest id (expert 1 beats expert 32,
// where the oracle picks 32). Exact fp32 key ties are rare and both choices are equally good experts.

#include <stdint.h>

#include "common.cuh"
#include "pdl.cuh"

using gbp_pdl::pdl_trigger;
using gbp_pdl::pdl_wait;
using gbp_pdl::prefetch_l2;

namespace {

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

// Two consecutive elements of x as fp32, the element type a template parameter (no per-load branch).
template <int DT>
__device__ __forceinline__ float2 load_pair(const void* p, size_t i) {
    if constexpr (DT == F32) return *reinterpret_cast<const float2*>(static_cast<const float*>(p) + i);
    else if constexpr (DT == BF16) return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(static_cast<const __nv_bfloat16*>(p) + i));
    else return __half22float2(*reinterpret_cast<const half2*>(static_cast<const half*>(p) + i));
}

// Block (e, r): warp w computes expert e * WARPS + w for the RB rows r * RB .. r * RB + RB - 1. Each lane's chain visits the pairs
// j = lane, lane + 32, ... in this order whatever UN (the pairs a lane loads before it consumes them) and WARPS: UN loads are issued
// together, the fmaf chain is consumed in pair order.
template <int RB, int WARPS, int UN, int DT>
__device__ __forceinline__ void router_logits_warp(const void* __restrict__ x, const half* __restrict__ gate, half* __restrict__ logits,
                                                   int rows, int K, int E) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * WARPS + warp;
    // the expert's gate row is a weight: bring it into L2 while the predecessor runs; then wait (first, early exits included)
    if (e < E)
        for (int off = lane * 128; off < K * 2; off += 32 * 128) prefetch_l2(reinterpret_cast<const char*>(gate + (size_t)e * K) + off);
    pdl_wait();
    if (e >= E) return;
    const int row0 = blockIdx.y * RB;
    const half2* w2 = reinterpret_cast<const half2*>(gate + (size_t)e * K);
    const int pairs = K / 2;
    float sum[RB];
#pragma unroll
    for (int r = 0; r < RB; ++r) sum[r] = 0.0f;
    for (int t0 = 0; t0 * 32 < pairs; t0 += UN) {
        half2 wv[UN];
        float2 xv[RB][UN];
#pragma unroll
        for (int u = 0; u < UN; ++u) {
            const int j = lane + 32 * (t0 + u);
            if (j < pairs) {
                wv[u] = w2[j];
#pragma unroll
                for (int r = 0; r < RB; ++r)
                    if (row0 + r < rows) xv[r][u] = load_pair<DT>(x, (size_t)(row0 + r) * K + 2 * (size_t)j);
            }
        }
#pragma unroll
        for (int u = 0; u < UN; ++u) {
            const int j = lane + 32 * (t0 + u);
            if (j < pairs) {
                const float2 wf = __half22float2(wv[u]);
#pragma unroll
                for (int r = 0; r < RB; ++r) {
                    if (row0 + r < rows) {
                        sum[r] = fmaf(xv[r][u].x, wf.x, sum[r]);
                        sum[r] = fmaf(xv[r][u].y, wf.y, sum[r]);
                    }
                }
            }
        }
    }
    pdl_trigger();                               // x is read: the successor may launch
#pragma unroll
    for (int r = 0; r < RB; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) sum[r] += __shfl_down_sync(0xffffffffu, sum[r], off);
        if (lane == 0 && row0 + r < rows) logits[(size_t)(row0 + r) * E + e] = __float2half_rn(sum[r]);
    }
}

// router_logits_warp with the warp's whole gate row held in registers, loaded BEFORE the PDL wait (a weight: nothing the predecessor
// writes), so the predecessor's run hides the trip to DRAM and only the x loads and the chain follow the wait. NP = pairs a lane
// visits (K = 64 NP). THE SAME ARITHMETIC: lane j's chain visits the pairs j, j + 32, ... in this order, two explicit fmaf per pair
// from 0, then the shfl_down tree 16 -> 1 and one rounding to fp16; UN only sets how many x pairs are loaded before they are consumed.
template <int RB, int WARPS, int NP, int UN, int DT>
__device__ __forceinline__ void router_logits_warp_held(const void* __restrict__ x, const half* __restrict__ gate, half* __restrict__ logits,
                                                        int rows, int K, int E) {
    static_assert(NP % UN == 0, "x pairs come in whole batches");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * WARPS + warp;
    half2 wv[NP];
    if (e < E) {
        const half2* w2 = reinterpret_cast<const half2*>(gate + (size_t)e * K);
#pragma unroll
        for (int t = 0; t < NP; ++t) wv[t] = __ldg(w2 + lane + 32 * t);
    }
    pdl_wait();
    if (e >= E) return;
    const int row0 = blockIdx.y * RB;
    float sum[RB];
#pragma unroll
    for (int r = 0; r < RB; ++r) sum[r] = 0.0f;
#pragma unroll
    for (int t0 = 0; t0 < NP; t0 += UN) {
        float2 xv[RB][UN];
#pragma unroll
        for (int u = 0; u < UN; ++u)
#pragma unroll
            for (int r = 0; r < RB; ++r)
                if (row0 + r < rows) xv[r][u] = load_pair<DT>(x, (size_t)(row0 + r) * K + 2 * (size_t)(lane + 32 * (t0 + u)));
#pragma unroll
        for (int u = 0; u < UN; ++u) {
            const float2 wf = __half22float2(wv[t0 + u]);
#pragma unroll
            for (int r = 0; r < RB; ++r) {
                if (row0 + r < rows) {
                    sum[r] = fmaf(xv[r][u].x, wf.x, sum[r]);
                    sum[r] = fmaf(xv[r][u].y, wf.y, sum[r]);
                }
            }
        }
    }
    pdl_trigger();                               // x is read: the successor may launch
#pragma unroll
    for (int r = 0; r < RB; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) sum[r] += __shfl_down_sync(0xffffffffu, sum[r], off);
        if (lane == 0 && row0 + r < rows) logits[(size_t)(row0 + r) * E + e] = __float2half_rn(sum[r]);
    }
}

template <int RB, int WARPS, int UN, int DT>
__global__ void __launch_bounds__(WARPS * 32)
    router_logits_kernel(const void* __restrict__ x, const half* __restrict__ gate, half* __restrict__ logits, int rows, int K, int E) {
    router_logits_warp<RB, WARPS, UN, DT>(x, gate, logits, rows, K, E);
}

// The prefill tiling of the same GEMV (a chunk's rows, the router and the DSA indexer's fp16 GEMVs): warp w of block (b, r) computes
// the EB experts (b * WARPS + w) * EB .. + EB - 1 for the RB rows r * RB .. r * RB + RB - 1, so a lane's x pair feeds EB experts and its
// gate pair RB rows (router_logits_kernel above feeds one expert per x load: at 2,048-4,096 rows it is bound on the loads, not the fmaf).
// THE SAME ARITHMETIC PER (row, expert) as router_logits_kernel: lane j's chain visits the pairs j, j + 32, ... in this order, two
// explicit fmaf per pair (x.x * w.x, then x.y * w.y) from 0, then the shfl_down tree 16 -> 1, one rounding to fp16. EB, RB, WARPS and UN
// only decide which warp holds which (row, expert) accumulator and how many loads are in flight; every accumulator is its own chain.
template <int RB, int EB, int WARPS, int UN, int DT>
__global__ void __launch_bounds__(WARPS * 32)
    router_logits_eb_kernel(const void* __restrict__ x, const half* __restrict__ gate, half* __restrict__ logits, int rows, int K, int E) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e0 = (blockIdx.x * WARPS + warp) * EB;
    if (e0 >= E) return;
    const int row0 = blockIdx.y * RB;
    const int pairs = K / 2;
    const half2* w2 = reinterpret_cast<const half2*>(gate + (size_t)e0 * K);
    float sum[RB][EB];
#pragma unroll
    for (int r = 0; r < RB; ++r)
#pragma unroll
        for (int i = 0; i < EB; ++i) sum[r][i] = 0.0f;
    for (int t0 = 0; t0 * 32 < pairs; t0 += UN) {
        half2 wv[UN][EB];
        float2 xv[RB][UN];
#pragma unroll
        for (int u = 0; u < UN; ++u) {
            const int j = lane + 32 * (t0 + u);
            if (j < pairs) {
#pragma unroll
                for (int i = 0; i < EB; ++i)
                    if (e0 + i < E) wv[u][i] = w2[(size_t)i * pairs + j];
#pragma unroll
                for (int r = 0; r < RB; ++r)
                    if (row0 + r < rows) xv[r][u] = load_pair<DT>(x, (size_t)(row0 + r) * K + 2 * (size_t)j);
            }
        }
#pragma unroll
        for (int u = 0; u < UN; ++u) {
            const int j = lane + 32 * (t0 + u);
            if (j < pairs) {
#pragma unroll
                for (int i = 0; i < EB; ++i) {
                    if (e0 + i < E) {
                        const float2 wf = __half22float2(wv[u][i]);
#pragma unroll
                        for (int r = 0; r < RB; ++r) {
                            if (row0 + r < rows) {
                                sum[r][i] = fmaf(xv[r][u].x, wf.x, sum[r][i]);
                                sum[r][i] = fmaf(xv[r][u].y, wf.y, sum[r][i]);
                            }
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RB; ++r) {
#pragma unroll
        for (int i = 0; i < EB; ++i) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) sum[r][i] += __shfl_down_sync(0xffffffffu, sum[r][i], off);
            if (lane == 0 && row0 + r < rows && e0 + i < E) logits[(size_t)(row0 + r) * E + e0 + i] = __float2half_rn(sum[r][i]);
        }
    }
}

// The prefill tiling with 16-byte loads (fp16 x, K a multiple of 64, x and gate 16-byte aligned): THE SAME ARITHMETIC PER (row, expert)
// as router_logits_kernel, with the 32 chains of an output held by 8 lanes instead of 32. Chain c (c = 0..31) of an output is the fmaf
// chain over the pairs c, c + 32, c + 64, ... (two explicit fmaf per pair, x.x * w.x then x.y * w.y, from 0): lane l (l = 0..7) of an
// 8-lane group holds chains 4 l .. 4 l + 3, so one 16-byte load brings the next pair of each of its four chains, and a group's 8 lanes
// load 128 contiguous bytes. The 32 chain sums then meet in the shfl_down tree's order: a_c = s_c + s_{c+16} (chain c + 16 is lane
// l + 4, the same register), b_c = a_c + a_{c+8} (lane l + 2), c_c = b_c + b_{c+4} (lane l + 1), d_c = c_c + c_{c+2} and d_0 + d_1 in
// the registers of lane 0: the values router_logits_kernel's lane 0 adds, in its order, then one rounding to fp16. A warp is 4 groups,
// each RB rows x EB experts (the groups of a warp share their rows: one transaction serves the four groups' x loads); block (bx, by)
// is WARPS warps of RB rows each over 4 EB experts.
template <int RB, int EB, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    router_logits_v4_kernel(const half* __restrict__ x, const half* __restrict__ gate, half* __restrict__ logits, int rows, int K, int E) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 3, l8 = lane & 7;
    const int row0 = (blockIdx.y * WARPS + warp) * RB;
    const int e0 = (blockIdx.x * 4 + g) * EB;
    // (a warp's rows are its own: a warp past the rows leaves whole; a group past the experts computes the last expert's chains, which
    // it does not store, so that every lane of the warp reaches the shuffles)
    if (row0 >= rows) return;
    const uint4* xr[RB];
    const uint4* wr[EB];
#pragma unroll
    for (int r = 0; r < RB; ++r) xr[r] = reinterpret_cast<const uint4*>(x + (size_t)min(row0 + r, rows - 1) * K) + l8;
#pragma unroll
    for (int i = 0; i < EB; ++i) wr[i] = reinterpret_cast<const uint4*>(gate + (size_t)min(e0 + i, E - 1) * K) + l8;
    float acc[RB][EB][4];
#pragma unroll
    for (int r = 0; r < RB; ++r)
#pragma unroll
        for (int i = 0; i < EB; ++i)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[r][i][q] = 0.0f;
    // step t: pairs 32 t + 4 l8 .. + 3, one 16-byte load (8 halves) a row and an expert
    const int steps = K / 64;
#pragma unroll 2
    for (int t = 0; t < steps; ++t) {
        uint4 xv[RB], wv[EB];
#pragma unroll
        for (int r = 0; r < RB; ++r) xv[r] = __ldg(xr[r] + 8 * t);
#pragma unroll
        for (int i = 0; i < EB; ++i) wv[i] = __ldg(wr[i] + 8 * t);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            float2 xf[RB];
#pragma unroll
            for (int r = 0; r < RB; ++r) xf[r] = __half22float2(reinterpret_cast<const half2*>(&xv[r])[q]);
#pragma unroll
            for (int i = 0; i < EB; ++i) {
                const float2 wf = __half22float2(reinterpret_cast<const half2*>(&wv[i])[q]);
#pragma unroll
                for (int r = 0; r < RB; ++r) {
                    acc[r][i][q] = fmaf(xf[r].x, wf.x, acc[r][i][q]);
                    acc[r][i][q] = fmaf(xf[r].y, wf.y, acc[r][i][q]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RB; ++r) {
#pragma unroll
        for (int i = 0; i < EB; ++i) {
            float s[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) s[q] = acc[r][i][q];
#pragma unroll
            for (int q = 0; q < 4; ++q) s[q] += __shfl_down_sync(0xffffffffu, s[q], 4, 8);
#pragma unroll
            for (int q = 0; q < 4; ++q) s[q] += __shfl_down_sync(0xffffffffu, s[q], 2, 8);
#pragma unroll
            for (int q = 0; q < 4; ++q) s[q] += __shfl_down_sync(0xffffffffu, s[q], 1, 8);
            const float d0 = s[0] + s[2], d1 = s[1] + s[3];
            if (l8 == 0 && row0 + r < rows && e0 + i < E) logits[(size_t)(row0 + r) * E + e0 + i] = __float2half_rn(d0 + d1);
        }
    }
}

constexpr int kMaxExperts = 512;
constexpr int kMaxPerLane = kMaxExperts / 32;
constexpr int kMaxTopK = 16;

// Order-preserving map of a float key to uint32 (larger float, larger code); NaN gets the smallest code.
__device__ __forceinline__ uint32_t key_code(float k) {
    if (k != k) return 0u;
    const uint32_t u = __float_as_uint(k);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// A logit's fp16 bits: a plain load, or (CG) one that bypasses L1 (logits written by other blocks of the same grid).
template <bool CG>
__device__ __forceinline__ unsigned short logit_bits(const half* lg, int id) {
    if constexpr (CG) return __ldcg(reinterpret_cast<const unsigned short*>(lg) + id);
    else return __half_as_ushort(lg[id]);
}

// One warp selects one row.
template <bool CG>
__device__ __forceinline__ void router_select_row(const half* __restrict__ logits, const float* __restrict__ sig,
                                                  const float* __restrict__ bias, int* __restrict__ pick, half* __restrict__ wts,
                                                  float scale, int E, int K, int row, int lane) {
    const half* lg = logits + (size_t)row * E;
    // candidate j of this lane is expert lane + 32 j; comp = key code in the high word, (~id) in the low word, so the largest
    // comp is the largest key and, among equal keys, the lowest id. 0 marks a taken or absent candidate.
    uint64_t comp[kMaxPerLane];
#pragma unroll
    for (int j = 0; j < kMaxPerLane; ++j) {
        const int id = lane + 32 * j;
        comp[j] = 0;
        if (id < E) {
            const float act = __ldg(sig + logit_bits<CG>(lg, id));
            const float key = act + __ldg(bias + id);
            comp[j] = ((uint64_t)key_code(key) << 32) | (uint64_t)(0xFFFFFFFFu - (uint32_t)id);
        }
    }
    int my_id = -1;
    float my_act = 0.0f;
    for (int k = 0; k < K; ++k) {
        uint64_t best = 0;
#pragma unroll
        for (int j = 0; j < kMaxPerLane; ++j) best = comp[j] > best ? comp[j] : best;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            const uint64_t o = ((uint64_t)__shfl_xor_sync(0xffffffffu, (uint32_t)(best >> 32), m) << 32) |
                               (uint64_t)__shfl_xor_sync(0xffffffffu, (uint32_t)best, m);
            best = o > best ? o : best;
        }
        const int id = (int)(0xFFFFFFFFu - (uint32_t)best);          // best is never 0 while K <= E
#pragma unroll
        for (int j = 0; j < kMaxPerLane; ++j)
            if (comp[j] == best) comp[j] = 0;
        if (lane == k) {
            my_id = id;
            my_act = __ldg(sig + logit_bits<CG>(lg, id));
        }
    }
    // The oracle's reduction: lanes >= K contribute 0, a shfl_down tree 16 -> 1 (an out-of-range source returns the lane's own
    // value, which only ever reaches lanes that lane 0 does not read), broadcast from lane 0.
    float s = lane < K ? my_act : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    s = __shfl_sync(0xffffffffu, s, 0);
    s = s + 1e-20f;
    if (lane < K) {
        pick[(size_t)row * K + lane] = my_id;
        wts[(size_t)row * K + lane] = __float2half_rn(__fdiv_rn(__fmul_rn(my_act, scale), s));
    }
}

// One warp per row.
__global__ void __launch_bounds__(32)
    router_select_kernel(const half* __restrict__ logits, const float* __restrict__ sig, const float* __restrict__ bias,
                         int* __restrict__ pick, half* __restrict__ wts, float scale, int E, int K) {
    pdl_wait();                                  // the logits are the predecessor's
    pdl_trigger();
    router_select_row<false>(logits, sig, bias, pick, wts, scale, E, K, blockIdx.x, threadIdx.x);
}

// The decode router in one launch: router_logits_kernel's blocks, then the LAST block to finish (a counter in `done`, back to 0 when it
// is done) selects every row, one warp a row, exactly as router_select_kernel does. The arithmetic of a logit and of a selection is
// the two kernels'; only the launch boundary between them is gone. NP > 0: the logits of router_logits_warp_held (K = 64 NP; the same
// bits), whose gate row is in registers before the PDL wait.
template <int RB, int WARPS, int UN, int DT, int NP = 0>
__global__ void __launch_bounds__(WARPS * 32)
    router_fused_kernel(const void* __restrict__ x, const half* __restrict__ gate, half* __restrict__ logits, const float* __restrict__ sig,
                        const float* __restrict__ bias, int* __restrict__ pick, half* __restrict__ wts, float scale, int rows, int K,
                        int E, int topk, unsigned* __restrict__ done) {
    __shared__ int last;
    if constexpr (NP > 0) router_logits_warp_held<RB, WARPS, NP, UN, DT>(x, gate, logits, rows, K, E);
    else router_logits_warp<RB, WARPS, UN, DT>(x, gate, logits, rows, K, E);
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();                         // this block's logits are visible before it counts itself done
        last = atomicAdd(done, 1u) == gridDim.x * gridDim.y - 1;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int row = warp; row < rows; row += WARPS) router_select_row<true>(logits, sig, bias, pick, wts, scale, E, topk, row, lane);
    if (threadIdx.x == 0) *done = 0;
}

bool dtype_ok(int d) { return d >= 0 && d <= 2; }

template <int RB, int WARPS, int UN, int DT>
int launch_logits(const void* x, const half* gate, half* logits, int rows, int K, int E, cudaStream_t st) {
    const dim3 grid((unsigned)((E + WARPS - 1) / WARPS), (unsigned)((rows + RB - 1) / RB));
    return gbp_pdl::launch_pdl(gbp_pdl::on(gbp_pdl::kMoe), router_logits_kernel<RB, WARPS, UN, DT>, grid, dim3(WARPS * 32), 0, st, x, gate,
                               logits, rows, K, E);
}

template <int RB, int WARPS, int UN>
int launch_logits_dt(const void* x, int dt, const half* gate, half* logits, int rows, int K, int E, cudaStream_t st) {
    if (dt == F32) return launch_logits<RB, WARPS, UN, F32>(x, gate, logits, rows, K, E, st);
    if (dt == BF16) return launch_logits<RB, WARPS, UN, BF16>(x, gate, logits, rows, K, E, st);
    return launch_logits<RB, WARPS, UN, F16>(x, gate, logits, rows, K, E, st);
}

template <int RB>
int launch_logits_shape(const void* x, int dt, const half* gate, half* logits, int rows, int K, int E, int warps, int un, cudaStream_t st) {
    if (warps == 4 && un == 1) return launch_logits_dt<RB, 4, 1>(x, dt, gate, logits, rows, K, E, st);
    if (warps == 8 && un == 1) return launch_logits_dt<RB, 8, 1>(x, dt, gate, logits, rows, K, E, st);
    if (warps == 4 && un == 4) return launch_logits_dt<RB, 4, 4>(x, dt, gate, logits, rows, K, E, st);
    if (warps == 8 && un == 4) return launch_logits_dt<RB, 8, 4>(x, dt, gate, logits, rows, K, E, st);
    if (warps == 4 && un == 8) return launch_logits_dt<RB, 4, 8>(x, dt, gate, logits, rows, K, E, st);
    if (warps == 8 && un == 8) return launch_logits_dt<RB, 8, 8>(x, dt, gate, logits, rows, K, E, st);
    if (warps == 2 && un == 8) return launch_logits_dt<RB, 2, 8>(x, dt, gate, logits, rows, K, E, st);
    if (warps == 2 && un == 4) return launch_logits_dt<RB, 2, 4>(x, dt, gate, logits, rows, K, E, st);
    if (warps == 1 && un == 8) return launch_logits_dt<RB, 1, 8>(x, dt, gate, logits, rows, K, E, st);
    if (warps == 1 && un == 16) return launch_logits_dt<RB, 1, 16>(x, dt, gate, logits, rows, K, E, st);
    if (warps == 2 && un == 16) return launch_logits_dt<RB, 2, 16>(x, dt, gate, logits, rows, K, E, st);
    if (warps == 4 && un == 16) return launch_logits_dt<RB, 4, 16>(x, dt, gate, logits, rows, K, E, st);
    if (warps == 8 && un == 16) return launch_logits_dt<RB, 8, 16>(x, dt, gate, logits, rows, K, E, st);
    return (int)cudaErrorInvalidValue;
}

template <int RB, int EB, int WARPS, int UN>
int launch_eb_dt(const void* x, int dt, const half* gate, half* logits, int rows, int K, int E, cudaStream_t st) {
    const dim3 grid((unsigned)((E + WARPS * EB - 1) / (WARPS * EB)), (unsigned)((rows + RB - 1) / RB));
    if (dt == F32) router_logits_eb_kernel<RB, EB, WARPS, UN, F32><<<grid, WARPS * 32, 0, st>>>(x, gate, logits, rows, K, E);
    else if (dt == BF16) router_logits_eb_kernel<RB, EB, WARPS, UN, BF16><<<grid, WARPS * 32, 0, st>>>(x, gate, logits, rows, K, E);
    else router_logits_eb_kernel<RB, EB, WARPS, UN, F16><<<grid, WARPS * 32, 0, st>>>(x, gate, logits, rows, K, E);
    return GB_CHECK_LAUNCH();
}

template <int RB, int EB>
int launch_eb_shape(const void* x, int dt, const half* gate, half* logits, int rows, int K, int E, int warps, int un, cudaStream_t st) {
    if (warps == 4 && un == 1) return launch_eb_dt<RB, EB, 4, 1>(x, dt, gate, logits, rows, K, E, st);
    if (warps == 8 && un == 1) return launch_eb_dt<RB, EB, 8, 1>(x, dt, gate, logits, rows, K, E, st);
    if (warps == 4 && un == 2) return launch_eb_dt<RB, EB, 4, 2>(x, dt, gate, logits, rows, K, E, st);
    if (warps == 8 && un == 2) return launch_eb_dt<RB, EB, 8, 2>(x, dt, gate, logits, rows, K, E, st);
    return (int)cudaErrorInvalidValue;
}

template <int RB, int EB, int WARPS>
int launch_v4(const void* x, const half* gate, half* logits, int rows, int K, int E, cudaStream_t st) {
    const dim3 grid((unsigned)((E + 4 * EB - 1) / (4 * EB)), (unsigned)((rows + WARPS * RB - 1) / (WARPS * RB)));
    router_logits_v4_kernel<RB, EB, WARPS><<<grid, WARPS * 32, 0, st>>>(static_cast<const half*>(x), gate, logits, rows, K, E);
    return GB_CHECK_LAUNCH();
}

template <int RB, int EB>
int launch_v4_warps(const void* x, const half* gate, half* logits, int rows, int K, int E, int warps, cudaStream_t st) {
    if (warps == 2) return launch_v4<RB, EB, 2>(x, gate, logits, rows, K, E, st);
    if (warps == 4) return launch_v4<RB, EB, 4>(x, gate, logits, rows, K, E, st);
    if (warps == 8) return launch_v4<RB, EB, 8>(x, gate, logits, rows, K, E, st);
    return (int)cudaErrorInvalidValue;
}

// What router_logits_v4_kernel takes: fp16 x, K a multiple of 64 (whole 16-byte steps of every chain), x and gate 16-byte aligned.
bool v4_takes(const void* x, int x_dtype, const void* gate, int K) {
    return x_dtype == F16 && K >= 64 && K % 64 == 0 && ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(gate)) & 15) == 0;
}

}  // namespace

GB_EXPORT int gb_moe_router_logits_ex(const void* x, int x_dtype, const void* gate, void* logits, int rows, int K, int E, int rb,
                                      int warps, int un, void* stream);

// The prefill tiling (router_logits_eb_kernel) with the blocking spelled out: rb rows a warp (2, 4, 8), eb experts a warp (4, 8), warps a
// block (4, 8), loads in flight a lane (1, 2). eb == 1 is router_logits_kernel's blocking (gb_moe_router_logits_ex). Every shape gives the
// same bits (the tests compare them with the host mirror).
GB_EXPORT int gb_moe_router_logits_ex2(const void* x, int x_dtype, const void* gate, void* logits, int rows, int K, int E, int rb, int eb,
                                       int warps, int un, void* stream) {
    if (rows < 1 || K < 2 || (K & 1) || E < 1 || !dtype_ok(x_dtype) || !x || !gate || !logits) return (int)cudaErrorInvalidValue;
    if (eb == 1) return gb_moe_router_logits_ex(x, x_dtype, gate, logits, rows, K, E, rb, warps, un, stream);
    cudaStream_t st = gb_stream(stream);
    const half* g = static_cast<const half*>(gate);
    half* l = static_cast<half*>(logits);
    if (rb == 2 && eb == 4) return launch_eb_shape<2, 4>(x, x_dtype, g, l, rows, K, E, warps, un, st);
    if (rb == 2 && eb == 8) return launch_eb_shape<2, 8>(x, x_dtype, g, l, rows, K, E, warps, un, st);
    if (rb == 4 && eb == 4) return launch_eb_shape<4, 4>(x, x_dtype, g, l, rows, K, E, warps, un, st);
    if (rb == 4 && eb == 8) return launch_eb_shape<4, 8>(x, x_dtype, g, l, rows, K, E, warps, un, st);
    if (rb == 8 && eb == 4) return launch_eb_shape<8, 4>(x, x_dtype, g, l, rows, K, E, warps, un, st);
    if (rb == 8 && eb == 8) return launch_eb_shape<8, 8>(x, x_dtype, g, l, rows, K, E, warps, un, st);
    return (int)cudaErrorInvalidValue;
}

// The prefill tiling with 16-byte loads (router_logits_v4_kernel) with the blocking spelled out: rb rows (2, 4, 8) and eb experts (2, 4, 8)
// an 8-lane group (rb * eb at most 32), warps a block (2, 4, 8). Only for what it takes (fp16 x, K a multiple of 64, x and gate 16-byte
// aligned): cudaErrorInvalidValue otherwise. Every shape gives router_logits_kernel's bits (the tests compare them with the host mirror).
GB_EXPORT int gb_moe_router_logits_v4(const void* x, int x_dtype, const void* gate, void* logits, int rows, int K, int E, int rb, int eb,
                                      int warps, void* stream) {
    if (rows < 1 || E < 1 || !x || !gate || !logits || !v4_takes(x, x_dtype, gate, K)) return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    const half* g = static_cast<const half*>(gate);
    half* l = static_cast<half*>(logits);
    if (rb == 2 && eb == 2) return launch_v4_warps<2, 2>(x, g, l, rows, K, E, warps, st);
    if (rb == 2 && eb == 4) return launch_v4_warps<2, 4>(x, g, l, rows, K, E, warps, st);
    if (rb == 2 && eb == 8) return launch_v4_warps<2, 8>(x, g, l, rows, K, E, warps, st);
    if (rb == 4 && eb == 2) return launch_v4_warps<4, 2>(x, g, l, rows, K, E, warps, st);
    if (rb == 4 && eb == 4) return launch_v4_warps<4, 4>(x, g, l, rows, K, E, warps, st);
    if (rb == 4 && eb == 8) return launch_v4_warps<4, 8>(x, g, l, rows, K, E, warps, st);
    if (rb == 8 && eb == 2) return launch_v4_warps<8, 2>(x, g, l, rows, K, E, warps, st);
    if (rb == 8 && eb == 4) return launch_v4_warps<8, 4>(x, g, l, rows, K, E, warps, st);
    return (int)cudaErrorInvalidValue;
}

// The same with the blocking spelled out: rb rows a warp (1, 2, 4), warps a block (2, 4, 8), loads in flight a lane (1, 4, 8). Every
// shape gives the same bits (a test compares them); the default below is the one measured fastest.
GB_EXPORT int gb_moe_router_logits_ex(const void* x, int x_dtype, const void* gate, void* logits, int rows, int K, int E, int rb,
                                      int warps, int un, void* stream) {
    if (rows < 1 || K < 2 || (K & 1) || E < 1 || !dtype_ok(x_dtype) || !x || !gate || !logits) return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    const half* g = static_cast<const half*>(gate);
    half* l = static_cast<half*>(logits);
    if (rb == 1) return launch_logits_shape<1>(x, x_dtype, g, l, rows, K, E, warps, un, st);
    if (rb == 2) return launch_logits_shape<2>(x, x_dtype, g, l, rows, K, E, warps, un, st);
    if (rb == 4) return launch_logits_shape<4>(x, x_dtype, g, l, rows, K, E, warps, un, st);
    return (int)cudaErrorInvalidValue;
}

// logits [rows, E] fp16 from x [rows, K] (dtype x_dtype) and the fp16 gate [E, K]. K even. The blocking changes the speed, never the bits.
GB_EXPORT int gb_moe_router_logits(const void* x, int x_dtype, const void* gate, void* logits, int rows, int K, int E,
                                   void* stream) {
    // Prefill chunks (32 rows and more: the router and the DSA indexer's fp16 GEMVs of a prompt chunk) take the 16-byte prefill tiling when
    // it applies (router_logits_v4_kernel: fp16 x, K a multiple of 64; four rows and four experts an 8-lane group, two warps a block, four
    // for 128 outputs or fewer): bench_router_v4, GB10, L2 warm, at 2,048 / 4,096 rows: 4096 x 288 838 / 1,707 -> 512 / 1,015 us,
    // 1536 x 512 582 / 1,163 -> 347 / 691 us, 4096 x 128 382 / 765 -> 244 / 479 us. Otherwise the 4-byte tiling, four rows and four experts
    // a warp, four warps a block, one load in flight (router_logits_eb_kernel). Both are the same chain per (row, expert); the tests
    // compare the bits with the host mirror.
    if (rows >= 32 && v4_takes(x, x_dtype, gate, K)) return gb_moe_router_logits_v4(x, x_dtype, gate, logits, rows, K, E, 4, 4, E <= 128 ? 4 : 2, stream);
    if (rows >= 32) return gb_moe_router_logits_ex2(x, x_dtype, gate, logits, rows, K, E, 4, 4, 4, 1, stream);
    const int rb = rows >= 8 ? 2 : 1;
    // Decode and verify windows (under 32 rows) keep sixteen loads in flight a lane: the gate of a layer is cold (a layer's 2.4 MB is read once a
    // step, in the middle of 3.7 GB of other weights), and each round of loads is a trip to DRAM: at 4,096 inputs a lane visits 64 pairs, so 4
    // loads in flight made every warp a chain of 16 round trips, 16 make it 4; the order of the fmaf chain does not change with the number of
    // loads in flight (the shape sweep test compares bits)
    return gb_moe_router_logits_ex(x, x_dtype, gate, logits, rows, K, E, rb, 8, 16, stream);
}

// Top-K per row from fp16 logits: pick [rows, K] int32 expert ids in rank order, wts [rows, K] fp16. sig: the 65,536-entry fp32
// sigmoid table, bias: fp32 [E] selection bias (added to the activation for selection only). K <= 16, E <= 512, K <= E.
GB_EXPORT int gb_moe_router_select(const void* logits, const void* sig, const void* bias, void* pick, void* wts, float scale,
                                   int rows, int E, int K, void* stream) {
    if (rows < 1 || E < 1 || E > kMaxExperts || K < 1 || K > kMaxTopK || K > E || !logits || !sig || !bias || !pick || !wts)
        return (int)cudaErrorInvalidValue;
    return gbp_pdl::launch_pdl(gbp_pdl::on(gbp_pdl::kMoe), router_select_kernel, dim3((unsigned)rows), dim3(32), 0, gb_stream(stream),
                               static_cast<const half*>(logits), static_cast<const float*>(sig), static_cast<const float*>(bias),
                               static_cast<int*>(pick), static_cast<half*>(wts), scale, E, K);
}

// gb_moe_router_logits then gb_moe_router_select, for a decode or verify window (rows < 32), as ONE launch (router_fused_kernel): the
// same logits (the decode blocking) and the same picks and weights. done: a u32 that is 0 between launches (the launch leaves it 0);
// one per stream of router launches. An fp16 x at K = 4096 (GLM-5.3-Flash) takes the held-gate logits (router_logits_warp_held: the
// gate row in registers before the PDL wait) with up to 8 rows a warp, so every row of a block reuses the held row (a standalone
// router with a 9 us predecessor and L2 flushed, after the predecessor: 1 row 21 -> 9 us, 8 rows 36 -> 15 us, 16 rows 65 -> 30 us).
GB_EXPORT int gb_moe_router_route(const void* x, int x_dtype, const void* gate, void* logits, const void* sig, const void* bias, void* pick,
                                  void* wts, float scale, int rows, int K, int E, int topk, void* done, void* stream) {
    if (rows < 1 || rows >= 32 || K < 2 || (K & 1) || E < 1 || E > kMaxExperts || topk < 1 || topk > kMaxTopK || topk > E ||
        !dtype_ok(x_dtype) || !x || !gate || !logits || !sig || !bias || !pick || !wts || !done)
        return (int)cudaErrorInvalidValue;
    constexpr int WARPS = 8, UN = 16;
    const bool held = K == 64 * 64 && x_dtype == F16;
    const int rb = held ? (rows <= 2 ? rows : rows <= 4 ? 4 : 8) : rows >= 8 ? 2 : 1;
    const dim3 grid((unsigned)((E + WARPS - 1) / WARPS), (unsigned)((rows + rb - 1) / rb));
#define GB_RF(RB_, DT_, NP_, UN_)                                                                                                       \
    return gbp_pdl::launch_pdl(gbp_pdl::on(gbp_pdl::kMoe), router_fused_kernel<RB_, WARPS, UN_, DT_, NP_>, grid, dim3(WARPS * 32), 0, \
                               gb_stream(stream), x, static_cast<const half*>(gate), static_cast<half*>(logits),                     \
                               static_cast<const float*>(sig), static_cast<const float*>(bias), static_cast<int*>(pick),             \
                               static_cast<half*>(wts), scale, rows, K, E, topk, static_cast<unsigned*>(done))
    if (held) {
        if (rb == 8) GB_RF(8, F16, 64, 4);
        if (rb == 4) GB_RF(4, F16, 64, 8);
        if (rb == 2) GB_RF(2, F16, 64, 16);
        GB_RF(1, F16, 64, 16);
    }
    if (rb == 2) {
        if (x_dtype == F32) GB_RF(2, F32, 0, UN);
        if (x_dtype == BF16) GB_RF(2, BF16, 0, UN);
        GB_RF(2, F16, 0, UN);
    }
    if (x_dtype == F32) GB_RF(1, F32, 0, UN);
    if (x_dtype == BF16) GB_RF(1, BF16, 0, UN);
    GB_RF(1, F16, 0, UN);
#undef GB_RF
}
