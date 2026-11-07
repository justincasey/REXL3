// gb-nvcc: --expt-relaxed-constexpr --fmad=false
// The routed experts of a MoE layer for LARGE row counts (a prefill chunk of 129..2048 rows of GLM-5.3-Flash), grouped by expert so
// that every expert's weights are streamed once and decoded once per group of up to 64 rows, with the arithmetic of the decode path
// (csrc/moe_experts.cu) reproduced bit for bit. The launches:
//
//   plan       the (row, slot) pairs -> a position in an expert-major SORTED order (each expert's pairs padded to whole 16-row tiles) and
//              three lists of work items (class A: 1 tile, B: 2 tiles, C: 3 or 4 tiles of one expert)
//   rot_in     per pair p: xs_mat[pos(p)] = fp16(H(x[row of p] * suh_mat[e])) for the gate and the up matrix, stored in the mma's
//              A-FRAGMENT order (one 16-byte read a lane and k step, see rot_in_frag_kernel in exl3_linear.cu)
//   gate/up    per (128 columns, work item): two teams of four warps, one a matrix, run the item's tiles of rows times the expert's W_q
//              (ALL K splits inside the block, sums over warps and splits in the decode path's order), then the block itself does
//              the epilogue on the two sums in shared memory: H, * svh, clamped SwiGLU, * suh_down, H, fp16 -> the down GEMM's A fragments
//   down       per (128 columns, work item): the same GEMM on those fragments, then H and * svh_down: each pair's fp32 y
//   combine    per row: out = fmaf chain over the slots IN SLOT ORDER of w_k * y_k, and the shared expert's output added LAST
//
// BIT-IDENTITY WITH THE DECODE PATH. Every number of a pair is a function of (its input row, its expert, the layer's weights) alone,
// and it is computed by the decode kernel's own recipe:
//   * the K range of a (split, warp) is `split * per_warp * WK + warp * per_warp` for `per_warp` k tiles, SK and WK = 4 taken from
//     gb_moe_default_plan (the SHAPE's alone), exactly as moe_gemv_kernel; every accumulator is one fresh fp32 m16n8k16 chain from
//     zero over those k tiles in order;
//   * the warps' chains are added in warp order (red[0] + red[1] + red[2] + red[3], each element from warp 0 on), then the splits in
//     split order: the in-block running sum `stage = split 0; stage += split 1` is the decode epilogue's `sg = Z[0]; sg += Z[1]`;
//   * gate/up epilogue, down epilogue and the slot combine are the decode kernels' own expressions applied to the same fp32 sums (this
//     file, like that one, is built --fmad=false; every fused multiply-add is an explicit fmaf); fusing them into the GEMM block moves
//     where the sums are held (shared memory, not a global buffer), never what they are;
//   * mma keeps rows apart: row i of D depends on row i of A only, so which tile or tile position a pair has, how many tiles a group
//     holds (NT) and what the other rows are cannot change a bit. The grouping decides only WHICH block computes a pair.
// The sorted order is a function of the picks alone (experts in id order, pairs in pair order) and no value depends on it.
//
// Derived from csrc/moe_experts.cu and csrc/exl3_linear.cu (the decode recipe reproduced here; their headers name the TensorFold kernels they follow over
// ExLlamaV3's decode: MIT, pinned 9cd52ab, Copyright (c) 2025 Turboderp; see THIRD_PARTY_NOTICES.md). The tile decode is csrc/exl3_linear_decode.cuh.

#include <stdint.h>

#include "common.cuh"
#include "exl3_grouped.cuh"

using namespace gbx;
using namespace gbp;

namespace {

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

constexpr int kPlanThreads = 1024;          // one expert a thread: E <= 1024
constexpr int kMaxTopK = 16;
constexpr int kWK = 4;                      // warps a block: the decode plan's (gb_moe_default_plan), never the row count's
constexpr int kMaxTiles = 4;                // 16-row tiles that share a decode

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

// Stores the 4 values v (columns n .. n + 3 of sorted row `s`) as fp16 into an A-fragment buffer xf [tile][K / 16][32 lanes][4 words]:
// lane (g, t) of a 16 x 16 tile holds a0 = (row g, columns 2t, 2t + 1), a1 = (row g + 8, same), a2 = (row g, columns 2t + 8, 2t + 9),
// a3 = (row g + 8, same), K the width of the matrix the rows are the input of.
__device__ __forceinline__ void store_frag4(uint32_t* xf, int s, int n, int K, const float (&v)[4]) {
    const half2 pair[2] = {__floats2half2_rn(v[0], v[1]), __floats2half2_rn(v[2], v[3])};
    const int mt = s >> 4, g = s & 7, rh = (s >> 3) & 1, KT = K >> 4;
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int col = n + 2 * q, kt = col >> 4, kk = col & 15;
        xf[(((size_t)mt * KT + kt) * 32 + g * 4 + ((kk & 7) >> 1)) * 4 + 2 * (kk >> 3) + rh] =
            *reinterpret_cast<const uint32_t*>(&pair[q]);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// plan

// Exclusive prefix sum of one int a thread over the whole block (1024 threads); *total receives the sum. ws: 33 ints of shared memory.
__device__ __forceinline__ int block_excl_scan(int v, int* total, int* ws) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int x = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += x;
    }
    if (lane == 31) ws[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        const int w = ws[lane];
        int s = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int x = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += x;
        }
        ws[lane] = s - w;
        if (lane == 31) ws[32] = s;
    }
    __syncthreads();
    const int r = ws[warp] + inc - v;
    *total = ws[32];
    __syncthreads();                                         // ws is reused by the next scan
    return r;
}

// One block. spair[s] = the pair of sorted row s: expert e's pairs take rows toff[e] * 16 + rank, rank = the pair's order among the expert's
// pairs (-1 for the padding rows of an expert's last tile; a pick outside [0, E) belongs to no expert and takes no row).
// nitems[3] = the number of sorted rows. Items: cap int4 slots per class {expert, first tile, tiles, valid rows}; nitems[3].
__global__ void __launch_bounds__(kPlanThreads) plan_kernel(const int* __restrict__ pick, int* __restrict__ spair,
                                                            int4* __restrict__ items, int* __restrict__ nitems, int P, int E,
                                                            int cap) {
    __shared__ int ws[33];
    __shared__ int totals[4];
    const int tid = threadIdx.x;
    int cnt = 0;
    if (tid < E) {
        int i = 0;
        for (; i + 4 <= P; i += 4) {
            const int4 q = __ldg(reinterpret_cast<const int4*>(pick + i));      // P is a multiple of 4 in every use; the tail is scalar
            cnt += (q.x == tid) + (q.y == tid) + (q.z == tid) + (q.w == tid);
        }
        for (; i < P; ++i) cnt += pick[i] == tid;
    }
    const int tiles = (cnt + 15) >> 4;
    int total_tiles;
    const int toff = block_excl_scan(tiles, &total_tiles, ws);
    if (tid < E && cnt > 0) {
        int j = 0;
        for (int i = 0; i < P; ++i)
            if (pick[i] == tid) spair[toff * 16 + j++] = i;
        for (; j < tiles * 16; ++j) spair[toff * 16 + j] = -1;       // the padding rows of the last tile belong to no pair
    }
    const int full = tiles >> 2, rem = tiles & 3;
    const int cC = full + (rem == 3), cB = rem == 2, cA = rem == 1;
    int nA, nB, nC;
    const int oA = block_excl_scan(cA, &nA, ws);
    const int oB = block_excl_scan(cB, &nB, ws);
    const int oC = block_excl_scan(cC, &nC, ws);
    if (tid < E && tiles > 0) {
        for (int gi = 0; gi < full; ++gi) items[2 * cap + oC + gi] = make_int4(tid, toff + 4 * gi, 4, min(64, cnt - 64 * gi));
        const int t0 = toff + 4 * full, vr = cnt - 64 * full;
        if (rem == 1) items[oA] = make_int4(tid, t0, 1, vr);
        if (rem == 2) items[cap + oB] = make_int4(tid, t0, 2, vr);
        if (rem == 3) items[2 * cap + oC + full] = make_int4(tid, t0, 3, vr);
    }
    if (tid == 0) {
        nitems[0] = nA;
        nitems[1] = nB;
        nitems[2] = nC;
        nitems[3] = total_tiles * 16;                                // the sorted rows, padding included
    }
}

// ---------------------------------------------------------------------------------------------------------------
// plan, parallel (multi-block). Produces EXACTLY the spair / items / nitems of plan_kernel above for every input,
// with the counting and the scatter spread over many blocks so the O(E * P) per-thread scans are gone. The sorted
// position of pick i (expert e, chunk b = i / kPlanChunk) is
//   rank(i) = block_off[b][e] + warp_pref(w, e) + within_warp_rank(lane)
// where block_off is the exclusive prefix of the per-block expert counts (a histogram is a sum, so its value is the
// same whatever order the block atomics land), warp_pref is the exclusive prefix of the per-warp counts inside the
// chunk, and within_warp_rank is popc(__match_any_sync(...) & before-mask). All three terms are pure functions of
// the picks alone, so the plan is deterministic and bit-identical to the single-block kernel. No atomics order, no
// block scheduling, no host-side decision enters the result.
//
// Pipeline on one stream (no device-wide sync beyond kernel boundaries):
//   plan_hist_kernel   [B blocks] local_hist[b * E + e] = count of expert e in chunk b
//   plan_prefix_kernel  [1 block]  block_off (in place over local_hist), toff, cnt, class offsets, nitems
//   plan_scatter_kernel [B blocks] spair[toff[e] * 16 + rank] = i
//   plan_items_kernel   [1 block]  items (classes A/B/C at their offsets), spair padding -1 rows

constexpr int kPlanChunk = 256;          // picks per block in the parallel plan
constexpr int kWarpCount = kPlanChunk / 32;

// Per-block histogram: hist[b * E + e] = number of picks in chunk b equal to expert e.
__global__ void __launch_bounds__(kPlanThreads) plan_hist_kernel(const int* __restrict__ pick,
                                                                 int* __restrict__ hist, int B, int P, int E) {
    __shared__ int sh[kPlanThreads];
    const int b = blockIdx.x, tid = threadIdx.x;
    for (int e = tid; e < E; e += kPlanThreads) sh[e] = 0;
    __syncthreads();
    const int base = b * kPlanChunk;
    const int len = P - base < kPlanChunk ? P - base : kPlanChunk;
    if (tid < len) {
        const int e = pick[base + tid];
        if (e >= 0 && e < E) atomicAdd(&sh[e], 1);
    }
    __syncthreads();
    for (int e = tid; e < E; e += kPlanThreads) hist[(size_t)b * E + e] = sh[e];
}

// Exclusive prefix sums across the E experts (E <= 1024): per-expert counts -> cnt; per-expert tile counts -> toff
// (total in nitems[3] * 16); per-expert class counts -> cls[0..E) = oA, cls[E..2E) = oB, cls[2E..3E) = oC with
// totals nitems[0..2]. Cross-block exclusive prefix per expert (block_off) is written back into `off` (the hist
// buffer, reused in place).
__global__ void __launch_bounds__(kPlanThreads) plan_prefix_kernel(const int* __restrict__ hist,
                                                                  int* __restrict__ off, int* __restrict__ toff,
                                                                  int* __restrict__ cnt, int* __restrict__ cls,
                                                                  int* __restrict__ nitems, int B, int E) {
    __shared__ int ws[33];
    const int tid = threadIdx.x;
    if (tid < E) {
        int c = 0;
        for (int b = 0; b < B; ++b) c += hist[(size_t)b * E + tid];
        cnt[tid] = c;
        int run = 0;
        for (int b = 0; b < B; ++b) {
            const int h = hist[(size_t)b * E + tid];
            off[(size_t)b * E + tid] = run;
            run += h;
        }
    }
    const int tiles = tid < E ? (cnt[tid] + 15) >> 4 : 0;
    int total_tiles;
    const int t0 = block_excl_scan(tiles, &total_tiles, ws);
    if (tid < E) toff[tid] = t0;
    const int full = tiles >> 2, rem = tiles & 3;
    const int cC = full + (rem == 3), cB = rem == 2, cA = rem == 1;
    int nA, nB, nC;
    const int oA = block_excl_scan(cA, &nA, ws);
    const int oB = block_excl_scan(cB, &nB, ws);
    const int oC = block_excl_scan(cC, &nC, ws);
    if (tid < E) {
        cls[tid] = oA;
        cls[E + tid] = oB;
        cls[2 * E + tid] = oC;
    }
    if (tid == 0) {
        nitems[0] = nA;
        nitems[1] = nB;
        nitems[2] = nC;
        nitems[3] = total_tiles * 16;
    }
}

// Scatter: spair[toff[e] * 16 + rank(i)] = i for every pick i. rank(i) = block_off[b][e] + warp_pref + wrank as
// described above; within a chunk, wrank is the within-warp rank (popc of the matching lanes before this lane) and
// warp_pref the exclusive prefix of the per-warp counts of expert e. cn [W][E] is the per-warp histogram of the
// chunk (deterministic by the same sum argument).
__global__ void __launch_bounds__(kPlanChunk) plan_scatter_kernel(const int* __restrict__ pick,
                                                                 const int* __restrict__ off,
                                                                 const int* __restrict__ toff,
                                                                 int* __restrict__ spair, int P, int E) {
    __shared__ int cnt[kWarpCount][kPlanThreads];
    const int b = blockIdx.x, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int base = b * kPlanChunk;
    const int len = P - base < kPlanChunk ? P - base : kPlanChunk;
    const bool valid = tid < len;
    const int e = valid ? pick[base + tid] : -1;
    for (int ev = lane; ev < E; ev += 32) cnt[warp][ev] = 0;
    __syncthreads();
    if (valid && e >= 0 && e < E) atomicAdd(&cnt[warp][e], 1);
    __syncthreads();
    const int me = (valid && e >= 0 && e < E) ? e : -1;
    const unsigned grp = __match_any_sync(0xffffffffu, me);
    const unsigned before = grp & ((1u << lane) - 1u);
    const int wrank = __popc(before);
    if (valid && e >= 0 && e < E) {
        int wp = 0;
        for (int w = 0; w < warp; ++w) wp += cnt[w][e];
        const int r = off[(size_t)b * E + e] + wp + wrank;
        spair[(size_t)toff[e] * 16 + r] = base + tid;
    }
}

// Items and padding: one thread per expert writes the expert's work items at their precomputed class offsets (the
// same item array, in the same order, as plan_kernel) and the -1 padding rows of its last tile.
__global__ void __launch_bounds__(kPlanThreads) plan_items_kernel(const int* __restrict__ toff,
                                                                 const int* __restrict__ cnt,
                                                                 const int* __restrict__ cls,
                                                                 int* __restrict__ spair,
                                                                 int4* __restrict__ items, int E, int cap) {
    const int tid = threadIdx.x;
    if (tid >= E) return;
    const int c = cnt[tid];
    const int tiles = (c + 15) >> 4;
    const int t0 = toff[tid];
    for (int j = c; j < tiles * 16; ++j) spair[(size_t)t0 * 16 + j] = -1;
    const int full = tiles >> 2, rem = tiles & 3;
    const int oA = cls[tid], oB = cls[E + tid], oC = cls[2 * E + tid];
    for (int gi = 0; gi < full; ++gi)
        items[2 * cap + oC + gi] = make_int4(tid, t0 + 4 * gi, 4, min(64, c - 64 * gi));
    const int vr = c - 64 * full;
    if (rem == 1) items[oA] = make_int4(tid, t0 + 4 * full, 1, vr);
    if (rem == 2) items[cap + oB] = make_int4(tid, t0 + 4 * full, 2, vr);
    if (rem == 3) items[2 * cap + oC + full] = make_int4(tid, t0 + 4 * full, 3, vr);
}

// ---------------------------------------------------------------------------------------------------------------
// input rotation: xs_mat[s] = fp16(H(x[row of pair spair[s]] * suh_mat[e])), in A-fragment order. One warp a 128-wide block of one SORTED row (so the 16
// rows of a tile, which share its fragment cache lines, are written by neighbouring blocks).

__global__ void __launch_bounds__(128) rot_in_kernel(const void* __restrict__ x, int x_dtype, const int* __restrict__ pick,
                                                     const int* __restrict__ spair, const int* __restrict__ nitems,
                                                     const half* __restrict__ suh0, const half* __restrict__ suh1,
                                                     uint32_t* __restrict__ xs0, uint32_t* __restrict__ xs1, int K, int topk, int E) {
    const int srow = blockIdx.x, mat = blockIdx.z;
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (blk * 128 >= K || srow >= nitems[3]) return;
    const int p = spair[srow];
    if (p < 0) return;
    const int e = pick[p];
    const int row = p / topk;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(x, x_dtype, (size_t)row * K + k, v);
    load4h((mat ? suh1 : suh0) + (size_t)e * K + k, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store_frag4(mat ? xs1 : xs0, srow, k, K, v);
}

// ---------------------------------------------------------------------------------------------------------------
// the grouped GEMMs

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

// Block (128 columns n block, item): two teams of WK = 4 warps, team 0 the gate matrix and team 1 the up matrix of the item's expert, each team a
// group of NT 16-row tiles sharing every tile decode (exl3_grouped.cuh). Both teams run the same loops, barriers included. The sums of the
// two end in shared memory; then the whole block turns them into the down GEMM's input rows (one warp a row): H over the 128 columns, * svh,
// silu(gate) clamped from above and up clamped to +-limit (the oracle's act_gate: the activated gate is clamped, not the pre-activation), the
// product, * suh_down, H again, fp16 into A-fragment order at the row's sorted position.
template <int K2, int CB, int NT, int H, int PD, int AP>
__global__ void __launch_bounds__(2 * kWK * 32, 1) gateup_kernel(const uint32_t* __restrict__ XS0, const uint32_t* __restrict__ XS1,
                                                                 const uint64_t* __restrict__ TP0, const uint64_t* __restrict__ TP1,
                                                                 const int4* __restrict__ items, const int* __restrict__ nitems,
                                                                 const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                                                 const half* __restrict__ suh_d, uint32_t* __restrict__ xd, int K, int N,
                                                                 int SK, float limit) {
    constexpr int TW = tile_words<K2>();
    constexpr int RC = (8 / H) * 16;                           // columns of a sweep
    constexpr int RED = kWK * 8 * RC, STAGE = NT * 16 * 128;   // floats a team
    static_assert(H == 1 || H == 2, "a block row is swept whole or in halves");
    const int it = blockIdx.y;
    if (it >= nitems[0]) return;
    extern __shared__ __align__(16) float smem[];              // [red 0][red 1][stage 0][stage 1]

    const int team = threadIdx.x >> 7, tt = threadIdx.x & 127;
    const int4 item = items[it];
    const int e = item.x, tile0 = item.y, nt = item.z, vrows = item.w;
    const int nb = blockIdx.x;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(team ? TP1[e] : TP0[e]);
    const uint32_t* XS = team ? XS1 : XS0;
    const int lane = threadIdx.x & 31;
    const int KT = K >> 4;
    const uint32_t* xp[NT];
#pragma unroll
    for (int p = 0; p < NT; ++p) xp[p] = XS + (size_t)(tile0 + (p < nt ? p : 0)) * KT * 128 + lane * 4;

    group_sums<kWK, K2, CB, NT, H, PD, AP>(T + (size_t)nb * 8 * TW, (long long)(N >> 4) * TW, xp, KT / SK / kWK, SK, nt, smem + team * RED,
                                           smem + 2 * RED + team * STAGE, tt);

    const float* stage_g = smem + 2 * RED;
    const float* stage_u = stage_g + STAGE;
    const int n = nb * 128 + 4 * lane;
    float sg4[4], su4[4], sd4[4];
    load4h(svh_g + (size_t)e * N + n, sg4);
    load4h(svh_u + (size_t)e * N + n, su4);
    load4h(suh_d + (size_t)e * N + n, sd4);
    for (int r = threadIdx.x >> 5; r < vrows; r += 2 * kWK) {
        const float4 g4 = *reinterpret_cast<const float4*>(stage_g + r * 128 + 4 * lane);
        const float4 u4 = *reinterpret_cast<const float4*>(stage_u + r * 128 + 4 * lane);
        float gv[4] = {g4.x, g4.y, g4.z, g4.w}, uv[4] = {u4.x, u4.y, u4.z, u4.w}, v[4];
        fwht128(gv, lane);
        fwht128(uv, lane);
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
        store_frag4(xd, tile0 * 16 + r, n, N, v);
    }
}

// Block (128 columns n block, item): the item's tiles of xd rows times the expert's down matrix, the sums in shared memory, then per row (one
// warp a row) H and * svh_down: y of the pair that sorted row belongs to, fp32. svh_d null (RAW): the sums alone (the K splits added in
// split order, no H, no scale), the K-split partial a TP rank of the width split exchanges before its epilogue.
template <int K2, int CB, int NT, int H, int PD, int AP>
__global__ void __launch_bounds__(kWK * 32, 1) down_kernel(const uint32_t* __restrict__ XS, const uint64_t* __restrict__ TP,
                                                           const int4* __restrict__ items, const int* __restrict__ nitems,
                                                           const half* __restrict__ svh_d, const int* __restrict__ spair,
                                                           float* __restrict__ y, int K, int N, int SK) {
    constexpr int TW = tile_words<K2>();
    constexpr int RC = (8 / H) * 16;
    constexpr int RED = kWK * 8 * RC;
    static_assert(H == 1 || H == 2, "a block row is swept whole or in halves");
    const int it = blockIdx.y;
    if (it >= nitems[0]) return;
    extern __shared__ __align__(16) float smem[];              // [red][stage]

    const int4 item = items[it];
    const int e = item.x, tile0 = item.y, nt = item.z, vrows = item.w;
    const int nb = blockIdx.x;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(TP[e]);
    const int lane = threadIdx.x & 31;
    const int KT = K >> 4;
    const uint32_t* xp[NT];
#pragma unroll
    for (int p = 0; p < NT; ++p) xp[p] = XS + (size_t)(tile0 + (p < nt ? p : 0)) * KT * 128 + lane * 4;

    const float* stage = smem + RED;
    group_sums<kWK, K2, CB, NT, H, PD, AP>(T + (size_t)nb * 8 * TW, (long long)(N >> 4) * TW, xp, KT / SK / kWK, SK, nt, smem, smem + RED,
                                           threadIdx.x);

    const int n = nb * 128 + 4 * lane;
    float sv[4] = {1.f, 1.f, 1.f, 1.f};
    if (svh_d) load4h(svh_d + (size_t)e * N + n, sv);
    for (int r = threadIdx.x >> 5; r < vrows; r += kWK) {
        const float4 u = *reinterpret_cast<const float4*>(stage + r * 128 + 4 * lane);
        float v[4] = {u.x, u.y, u.z, u.w};
        if (svh_d) {
            fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = v[j] * HAD_SCALE * sv[j];
        }
        *reinterpret_cast<float4*>(y + (size_t)spair[tile0 * 16 + r] * N + n) = make_float4(v[0], v[1], v[2], v[3]);
    }
}

// One warp a (row, 128 columns): out = the slots' y in slot order, acc = fmaf(w_k, y_k, acc) from 0, and the shared expert's output last.
__global__ void __launch_bounds__(32) combine_kernel(const float* __restrict__ y, const int* __restrict__ pick, const half* __restrict__ wts,
                                                     const float* __restrict__ shared, float* __restrict__ out, int D, int E, int topk) {
    const int r = blockIdx.x, lane = threadIdx.x;
    const int n = blockIdx.y * 128 + 4 * lane;
    float4 u[kMaxTopK];
    bool ok[kMaxTopK];
#pragma unroll
    for (int k = 0; k < kMaxTopK; ++k) {
        ok[k] = false;
        if (k < topk) {
            const int e = pick[(size_t)r * topk + k];
            ok[k] = e >= 0 && e < E;
            if (ok[k]) u[k] = *reinterpret_cast<const float4*>(y + ((size_t)r * topk + k) * D + n);
        }
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int q = 0; q < kMaxTopK; ++q) {
        if (q < topk && ok[q]) {
            const float w = __half2float(wts[(size_t)r * topk + q]);
            acc[0] = fmaf(w, u[q].x, acc[0]);
            acc[1] = fmaf(w, u[q].y, acc[1]);
            acc[2] = fmaf(w, u[q].z, acc[2]);
            acc[3] = fmaf(w, u[q].w, acc[3]);
        }
    }
    if (shared) {
        const float4 sh = *reinterpret_cast<const float4*>(shared + (size_t)r * D + n);
        acc[0] += sh.x; acc[1] += sh.y; acc[2] += sh.z; acc[3] += sh.w;
    }
    *reinterpret_cast<float4*>(out + (size_t)r * D + n) = make_float4(acc[0], acc[1], acc[2], acc[3]);
}

// The f32-weight combine (the Qwen grouped path's variant): out = the slots' y in slot order, acc = fmaf(w_k, y_k, acc)
// from 0, with `wts` f32 [rows * topk] as the router wrote them (no f32 -> fp16 rounding), and the shared expert's output
// last. Everything else is the fp16 combine's: the same slot order, the same fmaf chain, the same plain adds of the shared
// slot, the same one fp32 store. Only the weight load differs (float instead of half).
__global__ void __launch_bounds__(32) combine_f32_kernel(const float* __restrict__ y, const int* __restrict__ pick,
                                                         const float* __restrict__ wts, const float* __restrict__ shared,
                                                         float* __restrict__ out, int D, int E, int topk) {
    const int r = blockIdx.x, lane = threadIdx.x;
    const int n = blockIdx.y * 128 + 4 * lane;
    float4 u[kMaxTopK];
    bool ok[kMaxTopK];
#pragma unroll
    for (int k = 0; k < kMaxTopK; ++k) {
        ok[k] = false;
        if (k < topk) {
            const int e = pick[(size_t)r * topk + k];
            ok[k] = e >= 0 && e < E;
            if (ok[k]) u[k] = *reinterpret_cast<const float4*>(y + ((size_t)r * topk + k) * D + n);
        }
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int q = 0; q < kMaxTopK; ++q) {
        if (q < topk && ok[q]) {
            const float w = wts[(size_t)r * topk + q];
            acc[0] = fmaf(w, u[q].x, acc[0]);
            acc[1] = fmaf(w, u[q].y, acc[1]);
            acc[2] = fmaf(w, u[q].z, acc[2]);
            acc[3] = fmaf(w, u[q].w, acc[3]);
        }
    }
    if (shared) {
        const float4 sh = *reinterpret_cast<const float4*>(shared + (size_t)r * D + n);
        acc[0] += sh.x; acc[1] += sh.y; acc[2] += sh.z; acc[3] += sh.w;
    }
    *reinterpret_cast<float4*>(out + (size_t)r * D + n) = make_float4(acc[0], acc[1], acc[2], acc[3]);
}

// The router's fp32 weights (`moe_topk_softmax` writes f32) into the scratch's fp16 weights: out [n] = fp16(in [n]),
// round to nearest even (the format the combine reads; the Qwen path calls this right after routing).
__global__ void wts_f32_f16_kernel(const float* __restrict__ in, half* __restrict__ out, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        out[i] = __float2half_rn(in[i]);
    }
}

// The Qwen shared expert's finish: out [rows, D] bf16 = bf16(routed[r, d] + sigmoid(gate[r]) * shared[r, d]) with gate
// bf16 [rows]. One fp32 accumulate then one bf16 round: the CoopMK slot combine's arithmetic (the routed slots summed
// first in slot order, the shared slot weighted by its sigmoid added last, stored bf16 once).
__global__ void __launch_bounds__(32) finish_shared_kernel(const float* __restrict__ routed, const float* __restrict__ shared,
                                                           const __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ out,
                                                           int D) {
    const int r = blockIdx.x, lane = threadIdx.x;
    const int n = blockIdx.y * 128 + 4 * lane;
    const float g = 1.0f / (1.0f + expf(-__bfloat162float(gate[r])));
    const float4 a = *reinterpret_cast<const float4*>(routed + (size_t)r * D + n);
    const float4 b = *reinterpret_cast<const float4*>(shared + (size_t)r * D + n);
    const __nv_bfloat162 o0 = __floats2bfloat162_rn(a.x + b.x * g, a.y + b.y * g);
    const __nv_bfloat162 o1 = __floats2bfloat162_rn(a.z + b.z * g, a.w + b.w * g);
    uint2 u;
    u.x = *reinterpret_cast<const uint32_t*>(&o0);
    u.y = *reinterpret_cast<const uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(out + (size_t)r * D + n) = u;
}

// ---------------------------------------------------------------------------------------------------------------
// launch plumbing

bool dtype_ok(int d) { return d >= 0 && d <= 2; }

// The loads of the widths the pack uses (2, 3 and 4 bits, the MUL1 codebook) run the A fragments one step ahead (AP = 1; measured on the GB10: a layer
// of 2,048 rows takes 23.3 ms instead of 26.2) and the weights one step ahead; deeper weight prefetch costs the registers it needs and was slower
// (27.2 ms at PD = 2). The other widths and codebooks keep the plain loop (their decode needs the registers).
template <int K2, int CB>
constexpr int kAp = ((K2 == 4 || K2 == 6 || K2 == 8) && CB == 2) ? 1 : 0;

template <typename F>
int set_smem(F kernel, size_t bytes) {
    if (bytes > 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
    return 0;
}

struct GuArgs {
    const uint32_t *xs0, *xs1;
    const uint64_t *tp0, *tp1;
    const int4* items;
    const int* nitems;
    const half *svh_g, *svh_u, *suh_d;
    uint32_t* xd;
    int K, N, SK, cap;
    float limit;
};

template <int K2, int CB, int NT, int PD, int AP>
int launch_gateup_pa(const GuArgs& a, cudaStream_t st) {
    constexpr int H = sweeps_of<K2, CB>(NT);
    constexpr int RC = (8 / H) * 16;
    const size_t smem = 2 * ((size_t)kWK * 8 * RC + (size_t)NT * 16 * 128) * sizeof(float);
    auto kernel = gateup_kernel<K2, CB, NT, H, PD, AP>;
    set_smem(kernel, smem);
    const dim3 grid((unsigned)(a.N / 128), (unsigned)a.cap);
    kernel<<<grid, 2 * kWK * 32, smem, st>>>(a.xs0, a.xs1, a.tp0, a.tp1, a.items, a.nitems, a.svh_g, a.svh_u, a.suh_d, a.xd, a.K, a.N, a.SK,
                                             a.limit);
    return GB_CHECK_LAUNCH();
}

template <int K2, int CB, int NT>
int launch_gateup(const GuArgs& a, cudaStream_t st) {
    return launch_gateup_pa<K2, CB, NT, 1, kAp<K2, CB>>(a, st);
}

struct DnArgs {
    const uint32_t* xs;
    const uint64_t* tp;
    const int4* items;
    const int* nitems;
    const half* svh_d;
    const int* spair;
    float* y;
    int K, N, SK, cap;
};

template <int K2, int CB, int NT, int PD, int AP>
int launch_down_pa(const DnArgs& a, cudaStream_t st) {
    constexpr int H = sweeps_of<K2, CB>(NT);
    constexpr int RC = (8 / H) * 16;
    const size_t smem = ((size_t)kWK * 8 * RC + (size_t)NT * 16 * 128) * sizeof(float);
    auto kernel = down_kernel<K2, CB, NT, H, PD, AP>;
    set_smem(kernel, smem);
    const dim3 grid((unsigned)(a.N / 128), (unsigned)a.cap);
    kernel<<<grid, kWK * 32, smem, st>>>(a.xs, a.tp, a.items, a.nitems, a.svh_d, a.spair, a.y, a.K, a.N, a.SK);
    return GB_CHECK_LAUNCH();
}

template <int K2, int CB, int NT>
int launch_down(const DnArgs& a, cudaStream_t st) {
    return launch_down_pa<K2, CB, NT, 1, kAp<K2, CB>>(a, st);
}

#define GB_PF_FOR_CB(X, K2) X(K2, 0) X(K2, 1) X(K2, 2)
#define GB_PF_FOR_ALL(X)                                                                                                \
    GB_PF_FOR_CB(X, 2) GB_PF_FOR_CB(X, 4) GB_PF_FOR_CB(X, 6) GB_PF_FOR_CB(X, 8) GB_PF_FOR_CB(X, 10) GB_PF_FOR_CB(X, 12)  \
    GB_PF_FOR_CB(X, 14) GB_PF_FOR_CB(X, 16)

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// C ABI

// Item slots a class holds for P pairs and E experts: enough for every grouping (groups of 4 tiles of the padded rows, plus one
// remainder group an expert).
GB_EXPORT int gb_moe_pf_item_cap(int pairs, int experts) {
    if (pairs < 1 || experts < 1) return 0;
    const int e = experts < pairs ? experts : pairs;
    const long long rows = (long long)pairs + 15LL * e;      // the most padded rows
    return (int)(rows / 16 / kMaxTiles + e + 1);
}

// Plans the P = rows * topk picks (E <= 1024). spair [padded rows] i32; items [3 * cap] int4; nitems [4] i32 (the three classes' counts, then the
// number of sorted rows).
//
// Item-tail contract: BOTH planners (this one and gb_moe_pf_plan_parallel) zero the ENTIRE items buffer before
// writing the plan, so every capacity slot beyond the used region carries a deterministic, scratch-independent byte
// pattern. The legacy single-block kernel never wrote past its own nitems region, so the tail bytes used to be the
// caller's leftover buffer contents (stale reads / dirty scratch) rather than a well-defined part of the plan; the
// parallel planner must reproduce the legacy plan bit-for-bit for every capacity element, and with the tail defined
// as "zero", both planners converge on the same full-buffer identity no matter what the scratch held before. Nothing
// downstream reads past nitems, so this changes no valid-plan arithmetic and keeps the output identical.
GB_EXPORT int gb_moe_pf_plan(const void* pick, void* spair, void* items, void* nitems, int rows, int topk, int E, int cap,
                             void* stream) {
    const int P = rows * topk;
    if (rows < 1 || topk < 1 || topk > kMaxTopK || E < 1 || E > kPlanThreads || cap < 1 || cap < gb_moe_pf_item_cap(P, E) || !pick ||
        !spair || !items || !nitems)
        return (int)cudaErrorInvalidValue;
    const cudaStream_t st = gb_stream(stream);
    if (cudaMemsetAsync(items, 0, (size_t)3 * cap * sizeof(int4), st) != cudaSuccess)
        return (int)cudaGetLastError();
    plan_kernel<<<1, kPlanThreads, 0, st>>>(static_cast<const int*>(pick), static_cast<int*>(spair),
                                            static_cast<int4*>(items), static_cast<int*>(nitems), P, E, cap);
    return GB_CHECK_LAUNCH();
}

// Same plan as gb_moe_pf_plan (bit-identical spair / items / nitems) built with the multi-block parallel kernels. The
// extra workspaces are the parallel plan's own: hist [B * E] i32 (local histograms, then block_off in place), toff
// [E], cnt [E], cls [3 * E] (the three class offsets). The old single-block plan stays the fallback (it is the
// default; the switch in the Rust wrapper picks this one).
GB_EXPORT int gb_moe_pf_plan_parallel(const void* pick, void* spair, void* items, void* nitems, void* hist, void* toff,
                                      void* cnt, void* cls, int rows, int topk, int E, int cap, void* stream) {
    const int P = rows * topk;
    if (rows < 1 || topk < 1 || topk > kMaxTopK || E < 1 || E > kPlanThreads || cap < 1 || cap < gb_moe_pf_item_cap(P, E) || !pick ||
        !spair || !items || !nitems || !hist || !toff || !cnt || !cls)
        return (int)cudaErrorInvalidValue;
    const int B = (P + kPlanChunk - 1) / kPlanChunk;
    const cudaStream_t st = gb_stream(stream);
    // Item-tail contract (see gb_moe_pf_plan): zero the whole items buffer so the full-buffer plan identity is
    // scratch-independent and identical to the legacy builder's.
    if (cudaMemsetAsync(items, 0, (size_t)3 * cap * sizeof(int4), st) != cudaSuccess)
        return (int)cudaGetLastError();
    plan_hist_kernel<<<B, kPlanThreads, 0, st>>>(static_cast<const int*>(pick), static_cast<int*>(hist), B, P, E);
    if (cudaGetLastError() != cudaSuccess) return (int)cudaGetLastError();
    plan_prefix_kernel<<<1, kPlanThreads, 0, st>>>(static_cast<const int*>(hist), static_cast<int*>(hist),
                                                   static_cast<int*>(toff), static_cast<int*>(cnt), static_cast<int*>(cls),
                                                   static_cast<int*>(nitems), B, E);
    if (cudaGetLastError() != cudaSuccess) return (int)cudaGetLastError();
    plan_scatter_kernel<<<B, kPlanChunk, 0, st>>>(static_cast<const int*>(pick), static_cast<const int*>(hist),
                                                   static_cast<const int*>(toff), static_cast<int*>(spair), P, E);
    if (cudaGetLastError() != cudaSuccess) return (int)cudaGetLastError();
    plan_items_kernel<<<1, kPlanThreads, 0, st>>>(static_cast<const int*>(toff), static_cast<const int*>(cnt),
                                                  static_cast<const int*>(cls), static_cast<int*>(spair),
                                                  static_cast<int4*>(items), E, cap);
    return GB_CHECK_LAUNCH();
}

// xs0 / xs1: A fragments of the gate / up inputs, [S / 16][K / 16][32][4] u32, S = the padded row capacity.
GB_EXPORT int gb_moe_pf_rot_in(const void* x, int x_dtype, const void* pick, const void* spair, const void* nitems, const void* suh0,
                               const void* suh1, void* xs0, void* xs1, int rows, int topk, int K, int E, void* stream) {
    if (rows < 1 || topk < 1 || topk > kMaxTopK || K <= 0 || K % 128 || E < 1 || !dtype_ok(x_dtype) || !x || !pick || !spair || !nitems ||
        !suh0 || !suh1 || !xs0 || !xs1)
        return (int)cudaErrorInvalidValue;
    // the grid covers the most sorted rows there can be (every expert's pairs padded to whole 16-row tiles); the blocks past nitems[3] return
    const long long P = (long long)rows * topk, padded = (P + 15LL * (E < P ? E : P) + 15) / 16 * 16;
    const dim3 grid((unsigned)padded, (unsigned)((K / 128 + 3) / 4), 2);
    rot_in_kernel<<<grid, 128, 0, gb_stream(stream)>>>(x, x_dtype, static_cast<const int*>(pick), static_cast<const int*>(spair),
                                                       static_cast<const int*>(nitems), static_cast<const half*>(suh0),
                                                       static_cast<const half*>(suh1), static_cast<uint32_t*>(xs0),
                                                       static_cast<uint32_t*>(xs1), K, topk, E);
    return GB_CHECK_LAUNCH();
}

static int gemm_args_ok(int K, int N, int bits, int codebook, int sk, int wk, int cap) {
    return K > 0 && N > 0 && K % 128 == 0 && N % 128 == 0 && bits >= 1 && bits <= 8 && codebook >= 0 && codebook <= 2 && sk >= 1 &&
           wk == kWK && (K / 16) % (sk * wk) == 0 && cap >= 1;
}

// The gate and up GEMMs with the epilogue (K = the model width, N = the expert width I): xd = the down GEMM's A fragments [S / 16][I / 16][32][4]
// u32. tp0/tp1: per-expert trellis device addresses (uint64 [E]). The K split `sk` and wk = 4 are gb_moe_default_plan's (mats = 2).
// limit <= 0: no clamp.
GB_EXPORT int gb_moe_pf_gateup(const void* xs0, const void* xs1, const void* tp0, const void* tp1, const void* items, const void* nitems,
                               const void* svh_g, const void* svh_u, const void* suh_d, void* xd, int K, int N, int bits, int codebook,
                               int sk, int wk, int cap, float limit, void* stream) {
    if (!gemm_args_ok(K, N, bits, codebook, sk, wk, cap) || !xs0 || !xs1 || !tp0 || !tp1 || !items || !nitems || !svh_g || !svh_u || !suh_d || !xd)
        return (int)cudaErrorInvalidValue;
    const GuArgs a{static_cast<const uint32_t*>(xs0), static_cast<const uint32_t*>(xs1), static_cast<const uint64_t*>(tp0),
                   static_cast<const uint64_t*>(tp1), static_cast<const int4*>(items), static_cast<const int*>(nitems),
                   static_cast<const half*>(svh_g), static_cast<const half*>(svh_u), static_cast<const half*>(suh_d),
                   static_cast<uint32_t*>(xd), K, N, sk, cap, limit};
    cudaStream_t st = gb_stream(stream);
    int rc = 0;
    // the heaviest class first: its blocks fill the machine and the single-tile ones fill in behind them
#define GB_G(K2_, CB_)                                                                                                  \
    if (2 * bits == K2_ && codebook == CB_) {                                                                           \
        GuArgs c = a;                                                                                                   \
        c.items = a.items + 2 * cap;                                                                                    \
        c.nitems = a.nitems + 2;                                                                                        \
        rc = launch_gateup<K2_, CB_, 4>(c, st);                                                                         \
        c.items = a.items + cap;                                                                                        \
        c.nitems = a.nitems + 1;                                                                                        \
        if (!rc) rc = launch_gateup<K2_, CB_, 2>(c, st);                                                                \
        c.items = a.items;                                                                                              \
        c.nitems = a.nitems;                                                                                            \
        if (!rc) rc = launch_gateup<K2_, CB_, 1>(c, st);                                                                \
        return rc;                                                                                                      \
    }
    GB_PF_FOR_ALL(GB_G)
#undef GB_G
    return (int)cudaErrorInvalidValue;
}

// The down GEMM (svh_d null: RAW, no epilogue) behind gb_moe_pf_down and gb_moe_pf_down_raw.
static int pf_down(const void* xd, const void* tp, const void* items, const void* nitems, const void* svh_d, const void* spair, void* y, int K,
                   int N, int bits, int codebook, int sk, int wk, int cap, void* stream) {
    if (!gemm_args_ok(K, N, bits, codebook, sk, wk, cap) || !xd || !tp || !items || !nitems || !spair || !y)
        return (int)cudaErrorInvalidValue;
    const DnArgs a{static_cast<const uint32_t*>(xd), static_cast<const uint64_t*>(tp), static_cast<const int4*>(items),
                   static_cast<const int*>(nitems), static_cast<const half*>(svh_d), static_cast<const int*>(spair),
                   static_cast<float*>(y), K, N, sk, cap};
    cudaStream_t st = gb_stream(stream);
    int rc = 0;
#define GB_G(K2_, CB_)                                                                                                  \
    if (2 * bits == K2_ && codebook == CB_) {                                                                           \
        DnArgs c = a;                                                                                                   \
        c.items = a.items + 2 * cap;                                                                                    \
        c.nitems = a.nitems + 2;                                                                                        \
        rc = launch_down<K2_, CB_, 4>(c, st);                                                                           \
        c.items = a.items + cap;                                                                                        \
        c.nitems = a.nitems + 1;                                                                                        \
        if (!rc) rc = launch_down<K2_, CB_, 2>(c, st);                                                                  \
        c.items = a.items;                                                                                              \
        c.nitems = a.nitems;                                                                                            \
        if (!rc) rc = launch_down<K2_, CB_, 1>(c, st);                                                                  \
        return rc;                                                                                                      \
    }
    GB_PF_FOR_ALL(GB_G)
#undef GB_G
    return (int)cudaErrorInvalidValue;
}

// The down GEMM with the epilogue (K = the expert width I, N = the model width): y [P, N] fp32 = H(sum) * svh_down per pair, at the pair index each
// sorted row belongs to (`spair`). The split `sk` is gb_moe_default_plan's for mats = 1.
GB_EXPORT int gb_moe_pf_down(const void* xd, const void* tp, const void* items, const void* nitems, const void* svh_d, const void* spair,
                             void* y, int K, int N, int bits, int codebook, int sk, int wk, int cap, void* stream) {
    if (!svh_d) return (int)cudaErrorInvalidValue;
    return pf_down(xd, tp, items, nitems, svh_d, spair, y, K, N, bits, codebook, sk, wk, cap, stream);
}

// gb_moe_pf_down without the epilogue: y [P, N] fp32 = the K splits' sum in split order per pair (no H, no scale). A TP rank of the width
// split (K = its half of the expert width, sk = 1: one split of the whole width's plan) writes the partial its exchange sums.
GB_EXPORT int gb_moe_pf_down_raw(const void* xd, const void* tp, const void* items, const void* nitems, const void* spair, void* y, int K,
                                 int N, int bits, int codebook, int sk, int wk, int cap, void* stream) {
    return pf_down(xd, tp, items, nitems, nullptr, spair, y, K, N, bits, codebook, sk, wk, cap, stream);
}

// out [rows, D] fp32 = sum over slots in order of w * y, then + shared [rows, D] fp32 (null: nothing added). y [P, D] from gb_moe_pf_down.
GB_EXPORT int gb_moe_pf_combine(const void* y, const void* pick, const void* wts, const void* shared, void* out, int rows, int topk, int D,
                                int E, void* stream) {
    if (rows < 1 || topk < 1 || topk > kMaxTopK || D <= 0 || D % 128 || E < 1 || !y || !pick || !wts || !out) return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)rows, (unsigned)(D / 128));
    combine_kernel<<<grid, 32, 0, gb_stream(stream)>>>(static_cast<const float*>(y), static_cast<const int*>(pick), static_cast<const half*>(wts),
                                                       static_cast<const float*>(shared), static_cast<float*>(out), D, E, topk);
    return GB_CHECK_LAUNCH();
}

// The f32-weight combine (the Qwen grouped path's variant): the same combine as gb_moe_pf_combine with `wts` f32
// [rows * topk] instead of fp16 (the router's weights, no f32 -> fp16 rounding).
GB_EXPORT int gb_moe_pf_combine_f32(const void* y, const void* pick, const void* wts, const void* shared, void* out, int rows, int topk, int D,
                                    int E, void* stream) {
    if (rows < 1 || topk < 1 || topk > kMaxTopK || D <= 0 || D % 128 || E < 1 || !y || !pick || !wts || !out) return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)rows, (unsigned)(D / 128));
    combine_f32_kernel<<<grid, 32, 0, gb_stream(stream)>>>(static_cast<const float*>(y), static_cast<const int*>(pick), static_cast<const float*>(wts),
                                                            static_cast<const float*>(shared), static_cast<float*>(out), D, E, topk);
    return GB_CHECK_LAUNCH();
}

// The router's fp32 weights (gb_moe_topk_softmax writes f32) into the scratch's fp16 weights: out [n] fp16 = fp16(in [n] f32).
GB_EXPORT int gb_moe_pf_wts_f32_f16(const void* in, void* out, long long n, void* stream) {
    if (!in || !out || n < 1) return (int)cudaErrorInvalidValue;
    const size_t blocks = ((size_t)n + 255) / 256;
    wts_f32_f16_kernel<<<(unsigned)blocks, 256, 0, gb_stream(stream)>>>((const float*)in, (half*)out, (size_t)n);
    return GB_CHECK_LAUNCH();
}

// The Qwen shared expert's finish: out [rows, D] bf16 = bf16(routed + sigmoid(gate) * shared), gate bf16 [rows].
GB_EXPORT int gb_moe_pf_finish_shared(const void* routed, const void* shared, const void* gate, void* out, int rows, int D, void* stream) {
    if (rows < 1 || D <= 0 || D % 128 || !routed || !shared || !gate || !out) return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)rows, (unsigned)(D / 128));
    finish_shared_kernel<<<grid, 32, 0, gb_stream(stream)>>>((const float*)routed, (const float*)shared,
                                                               (const __nv_bfloat16*)gate, (__nv_bfloat16*)out, D);
    return GB_CHECK_LAUNCH();
}
