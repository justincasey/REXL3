// gb-nvcc: --expt-relaxed-constexpr
// EXL3 linear layer: y = ((x * suh) H_K . W_q . H_N) * svh (+ bias), H = 128-block Sylvester Hadamard / sqrt(128).
//
// Ported from TensorFold's cuda/exl3/linear.cu (MIT, pinned 9cd52ab; see THIRD_PARTY_NOTICES.md) with the ATen
// wrappers replaced by plain C-ABI launchers. The device code (decode straight into mma B fragments, fixed K plan
// from (K, N) alone, fixed-order reductions) is TensorFold's; design cross-checked against ExLlamaV3's kernels.
//
// ROW INVARIANCE (decode path, gb_exl3_linear, 1..128 rows): a row's output bits do not depend on the other rows
// or on M. mma.sync keeps rows apart; the K split (SK) and warps per block (WK) come from (K, N) only; every sum
// (over k tiles in a warp, over warps, over splits) runs in a fixed order, and there are no atomics at all: a layer
// whose N is too narrow to fill the GPU splits K, writes the fp32 partials to the workspace and a second kernel
// adds them in split order.
//
// INPUT ROTATION (decode path of up to 8 rows, env REXL3_EXL3_FUSE): x * suh, H / sqrt(128) rounded to fp16 is either a launch of its own
// ahead of the linear (rot_in_kernel) or, on the layers where that pays (4-warp blocks, a wide grid: ring_plan), done by the linear kernel's own
// warps: each warp rotates the 128-column blocks of x its k range touches into shared memory (rot_chunk, rot_in_kernel's statements, so its
// bits) one block ahead of its loop and reads its A fragments from there. Nothing about the sums changes: the same halves reach the same mmas.
//
// ABI conventions: x/y/xh pointers 16-byte aligned, K and N multiples of 128, dtype codes F16=0, BF16=1, F32=2,
// codebook codes 3INST=0, MCG=1, MUL1=2, `bits` an integer 1..8 (trellis [K/16, N/16, 16*bits] int16, row-major;
// the kernels read it as int32 words in place, no repack; the decode path of 1 to 16 rows copies it in 16-byte pieces, so the
// trellis pointer is 16-byte aligned too, and so is every strip or K range of it that is passed on). The `k2` variants
// (gb_exl3_linear_k2 / gb_exl3_reconstruct_k2) take half-bit units: even k2 = 2 * bits in 2..16 with any codebook, odd k2 = 3 / 5
// / 7 (1.5 / 2.5 / 3.5 bits) with the mul1 codebook. Every launcher returns 0 or a cudaError_t and never allocates or
// synchronizes.

#include <cublas_v2.h>
#include <cublasLt.h>
#include <stdint.h>

#include "common.cuh"
#include "pdl.cuh"
#include "exl3_linear_decode.cuh"
#include "exl3_linear_ring.cuh"

using namespace gbx;

namespace {

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

constexpr int kMaxRows = 128;

// Occupancy asked of the ring kernel (blocks per SM: 2 of 8 warps, 4 of 4); tuning knobs of the microbenchmark builds.
#ifndef RING_MINB8
#define RING_MINB8 2
#endif
#ifndef RING_MINB4
#define RING_MINB4 4
#endif

// The fused input rotation (decode path, M <= kFuseRows rows): each warp rotates one 128-column block of the rows it is about to
// multiply (see rot_chunk) into a private buffer of kAPitch halves a row (a 16-byte pad keeps the fragment reads conflict free).
constexpr int kFuseRows = 8;
constexpr int kAPitch = 128 + 8;

// Bytes of the ring kernel's shared memory before its mbarriers: RS step slots a warp (128 K2 bytes each) followed, with the fused input
// rotation (arows > 0 rows), by the warp's rotated block, and the two share their space with the cross-warp reduction's red (WK warps x
// rh rows x 128 floats, used once the k loop is over): the largest of the three, 8-byte aligned.
template <int K2, int WK>
__host__ __device__ constexpr size_t ring_region(int rs, int rh, int arows = 0) {
    const size_t slots = (size_t)WK * rs * (128 * K2) + (size_t)WK * arows * kAPitch * sizeof(half);
    const size_t red = (size_t)WK * rh * 128 * sizeof(float);
    return ((slots > red ? slots : red) + 7) & ~size_t(7);
}

// Programmatic dependent launch (pdl.cuh): the decode chain rot_in -> linear -> rot_out; only timing changes.
using gbp_pdl::pdl_wait;
using gbp_pdl::pdl_trigger;
using gbp_pdl::prefetch_l2;

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

// The finished outputs of one row's 128 columns from their fp32 sums (4 a lane): H / sqrt(128), * svh, + bias.
__device__ __forceinline__ void finish(float (&v)[4], int lane, const half* svh, const half* bias, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = v[j] * HAD_SCALE * __half2float(__ldg(svh + col + j));
        if (bias) v[j] += __half2float(__ldg(bias + col + j));
    }
}

// A lane's words of one k step: at 1 and 2 bits the warp loads the step together and each lane takes its words by shuffle.
template <int K2>
__host__ __device__ constexpr bool step_shuffled() {
    return K2 == 2 || K2 == 4;
}

// Registers a lane holds for one k step of JN tiles (the whole 8-tile block row, or a column half of it).
template <int K2, int JN = 8>
__host__ __device__ constexpr int step_regs() {
    return step_shuffled<K2>() ? JN * tile_words<K2>() / 32 : JN * lane_words<K2>();
}

template <int K2, int JN = 8>
__device__ __forceinline__ void load_step(const uint32_t* step, int lane, uint32_t (&raw)[step_regs<K2, JN>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
#pragma unroll
        for (int c = 0; c < step_regs<K2, JN>(); ++c) raw[c] = __ldg(step + c * 32 + lane);
    } else {
        int word, offset;
        lane_start<K2>(lane, word, offset);
#pragma unroll
        for (int j = 0; j < JN; ++j)
#pragma unroll
            for (int q = 0; q < LW; ++q) raw[j * LW + q] = __ldg(step + j * TW + (word + q) % TW);
    }
}

// Tile j's lane words from a loaded step; prev: the lane's first window starts in the word before its own.
template <int K2, int JN = 8>
__device__ __forceinline__ void step_lane_words(const uint32_t (&raw)[step_regs<K2, JN>()], int j, int lane, bool prev,
                                                uint32_t (&w)[lane_words<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
        static_assert(LW == 2, "a lane's windows span two words at 1 and 2 bits");
        constexpr int LPW = 8 / K2;                  // lanes whose windows end in the same word
        const uint32_t r = raw[j * TW / 32];
        const int base = (j * TW) % 32;
        const uint32_t own = __shfl_sync(0xffffffffu, r, base + lane / LPW);
        const uint32_t before = __shfl_sync(0xffffffffu, r, base + (lane / LPW + TW - 1) % TW);
        w[0] = prev ? before : own;
        w[1] = prev ? own : 0u;                      // a window inside one word never reads w[1]
    } else {
#pragma unroll
        for (int q = 0; q < LW; ++q) w[q] = raw[j * LW + q];
    }
}

// xh [M, K] fp16 = (x * suh) H / sqrt(128), fp32 inside, fp16 out. One warp a 128-wide block of one row.
__global__ void __launch_bounds__(128) rot_in_kernel(const void* __restrict__ x, int x_dtype,
                                                     const half* __restrict__ suh, half* __restrict__ xh, int K) {
    pdl_trigger();                               // the successor may launch at once: it waits for this grid before it reads xh
    pdl_wait();                                  // first, in every block, early exits included (see pdl.cuh)
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5), row = blockIdx.x, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(suh, F16, k, s);
    load4(x, x_dtype, (size_t)row * K + k, v);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4(xh, F16, (size_t)row * K + k, v);
}

// rot_in_kernel's work for the decode kernel's own warps (the fused input rotation): the 128-column block c of the first M rows, one
// row at a time (4 columns a lane), into dst [M][kAPitch] halves, RB rows' loads in flight together. The arithmetic is rot_in_kernel's,
// statement for statement (x * suh, fwht128, * HAD_SCALE, round to fp16), so the halves are the ones that kernel stores to xh.
template <int RB, int DT>
__device__ __forceinline__ void rot_chunk_t(const void* x, const half* suh, half* dst, int M, int K, int c, int lane) {
    const int k = c * 128 + 4 * lane;
    float s[4];
    load4(suh, F16, k, s);
#pragma unroll 1
    for (int r0 = 0; r0 < M; r0 += RB) {
        float v[RB][4];
#pragma unroll
        for (int d = 0; d < RB; ++d) load4(x, DT, (size_t)min(r0 + d, M - 1) * K + k, v[d]);   // rows past M repeat the last: not stored
#pragma unroll
        for (int d = 0; d < RB; ++d) {
#pragma unroll
            for (int j = 0; j < 4; ++j) v[d][j] *= s[j];
            fwht128(v[d], lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) v[d][j] *= HAD_SCALE;
            if (r0 + d < M) store4(dst + (size_t)(r0 + d) * kAPitch, F16, 4 * lane, v[d]);
        }
    }
}

template <int RB>
__device__ __forceinline__ void rot_chunk(const void* x, int x_dtype, const half* suh, half* dst, int M, int K, int c, int lane) {
    if (x_dtype == F32) rot_chunk_t<RB, F32>(x, suh, dst, M, K, c, lane);
    else if (x_dtype == BF16) rot_chunk_t<RB, BF16>(x, suh, dst, M, K, c, lane);
    else rot_chunk_t<RB, F16>(x, suh, dst, M, K, c, lane);
}

// The same rotated input in the mma's A-fragment order, for the decode kernel: xf is [ceil(M / 16)][K / 16][32 lanes][4 words],
// lane (g, t) of a 16 x 16 tile holding the words a0 = (row g, columns 2t, 2t + 1), a1 = (row g + 8, same), a2 = (row g,
// columns 2t + 8, 2t + 9), a3 = (row g + 8, same), so a warp loads a tile's fragment in one 512-byte read instead of four
// scattered ones. Rows past M (the last tile's) repeat row M - 1; their outputs are dropped. The values are exactly rot_in's.
__global__ void __launch_bounds__(128) rot_in_frag_kernel(const void* __restrict__ x, int x_dtype,
                                                          const half* __restrict__ suh, uint32_t* __restrict__ xf, int K,
                                                          int M) {
    pdl_wait();
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5), row = blockIdx.x, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(suh, F16, k, s);
    load4(x, x_dtype, (size_t)min(row, M - 1) * K + k, v);
    pdl_trigger();
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

// y [M, N] = (c H / sqrt(128)) * svh + bias, c the fp32 sum of SK partials c + q * M * N added in order q = 0, 1, ...
// (SK = 1 for the prefill epilogue, the K splits of the decode path otherwise). One warp a 128-wide block of one row.
__global__ void __launch_bounds__(128) rot_out_kernel(const float* __restrict__ c, const half* __restrict__ svh,
                                                      const half* __restrict__ bias, void* __restrict__ y,
                                                      int y_dtype, int M, int N, int SK) {
    pdl_trigger();                               // the successor may launch at once: it waits for this grid before it reads y
    pdl_wait();                                  // c is the predecessor's output; first, early exits included
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5), row = blockIdx.x, lane = threadIdx.x & 31;
    if (blk * 128 >= N) return;
    const int col = blk * 128 + 4 * lane;
    const size_t at = (size_t)row * N + col;
    float4 s = __ldcg(reinterpret_cast<const float4*>(c + at));
    for (int q0 = 1; q0 < SK; q0 += 8) {          // a chunk's loads are issued together, the adds stay in order
        float4 u[8];
#pragma unroll
        for (int d = 0; d < 8; ++d)
            if (q0 + d < SK) u[d] = __ldcg(reinterpret_cast<const float4*>(c + (size_t)(q0 + d) * M * N + at));
#pragma unroll
        for (int d = 0; d < 8; ++d)
            if (q0 + d < SK) { s.x += u[d].x; s.y += u[d].y; s.z += u[d].z; s.w += u[d].w; }
    }
    float v[4] = {s.x, s.y, s.z, s.w};
    finish(v, lane, svh, bias, col);
    store4(y, y_dtype, at, v);
}

// NT 16-row mma passes share every tile decode: a pass's rows only ever meet their own accumulators, so a row's sums
// (over k tiles, over warps, over splits) run in the same order whatever NT (and H) is and its bits do not depend on it.
// NT = 1 is the bandwidth-bound 1..16 row kernel; NT > 1 serves larger windows, where decoding a tile once per pass was
// the cost (the weights are re-read from L1/L2, but decoded into B fragments again every time).
//
// The accumulators of a warp are NT x 8 tiles x 8 floats: 64 NT registers, which is all the register file at NT = 3. With
// H = 2 the warp sweeps its k range twice, once over the left and once over the right 64 columns of the block (each tile
// still decoded exactly once and each accumulator still summed over the same k tiles in the same order), so the
// accumulators are half the size and NT = 3 and 4 fit without spilling. A sweep's per-warp sums are reduced across the
// warps in warp order into shared memory ("stage", 16 NT rows x 128 columns) and the block rotates and stores the rows once
// both halves are in; with split K (SK > 1) a sweep writes its columns of the partials straight to Z.
// (minBlocks = 1: left to itself ptxas caps a 128-thread block of the 2-pass kernel at 128 registers and the wide 3INST and MCG
// trellises then spill a few bytes; asked for one block it uses what the decode needs.)
//
// NT = 1 (and H = 1) is the memory-bound kernel and it takes its trellis words through a shared-memory ring: each warp keeps up to
// RING_MAXS of its k steps in flight, a step being 8 tiles stored back to back, brought in by one bulk copy (cp.async.bulk, completed
// on an mbarrier of the warp's own) and read from shared memory by the lanes when it lands. That costs no registers (the register
// prefetch of the NT > 1 kernels is what pinned the old kernel at 200+ registers and one 8-warp block, or two 4-warp blocks, an
// SM): the kernel runs in 128 registers, so 2 blocks of 8 warps or 4 of 4 share an SM, a wave of 80 or 96 blocks is resident at
// once and the blocks' prologues and epilogues overlap each other's streaming. Which words a lane decodes, the order of the mmas
// and every sum after them are those of the register loop; only the timing differs.
// FI (the ring kernel only, M <= kFuseRows rows): the kernel rotates its own input. xh is not read: each warp rotates, after the programmatic
// wait, the 128-column blocks of x it multiplies, one block ahead of its k loop, into shared memory (rot_chunk), instead of a separate
// rot_in_kernel launch and a round trip of xh through global memory. Every A fragment is the word the unfused path reads from xh.
template <int K2, int CB, int WK, int NT, int H, bool FI = false>
__global__ void __launch_bounds__(WK * 32, (NT == 1 && H == 1) ? (WK == 8 ? RING_MINB8 : RING_MINB4) : 1) linear_kernel(
    const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,
    const half* __restrict__ svh, const half* __restrict__ bias, void* __restrict__ y, int y_dtype,
    float* __restrict__ Z, int M, int K, int N, int SK, int ZM, int RS, const void* __restrict__ xin, int x_dtype,
    const half* __restrict__ suh) {
    // ZM (lane E, the TP splits): write every split's fp32 partials to Z and finish nothing, also when SK == 1 (a rank's share of a
    // parent's K splits can be one split). The partials are the same sums the finishing paths use: the warp-order sum per split.
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    constexpr bool FRAG = NT >= 2;                            // the A fragments come in fragment order (see rot_in_frag_kernel)
    constexpr int JN = 8 / H;                                 // 16-column tiles of a sweep
    constexpr int RC = JN * 16;                               // columns of a sweep
    static_assert(H == 1 || H == 2, "a block row is swept whole or in halves");
    extern __shared__ __align__(16) float red[];              // WK * RH * RC floats, then (H > 1, SK == 1) 16 NT * 128 floats
    const int RH = min(M, 8);                                 // rows of red a warp
    float* stage = red + WK * RH * RC;
    constexpr bool RING = NT == 1 && H == 1;                  // M <= 16: the trellis words come through a shared-memory ring
    constexpr uint32_t STEP_BYTES = 128 * K2;                 // a k step: 8 tiles of 16 K2 bytes, contiguous
    char* const ring = reinterpret_cast<char*>(red);          // RS slots a warp, then (past red's space) one mbarrier a slot
    uint64_t* const bars = reinterpret_cast<uint64_t*>(ring + ring_region<K2, WK>(RS, RH, FI ? M : 0));

    const int nb = blockIdx.x, split = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const uint32_t* tiles = T + nb * stride_nb;
    const int col0 = nb * 128;
    bool prev = false;
    if constexpr (step_shuffled<K2>()) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }

    for (int m0 = 0; m0 < M; m0 += 16 * NT) {
        // each pass's A fragments. One pass (M <= 16) reads its rows from the row-major rotated input, clamped so rows past the end
        // read inside the buffer; more passes read 16-row tiles of the fragment-order input (rot_in_frag_kernel), clamped so passes
        // past the last tile do (their outputs are dropped)
        const half* x0[FRAG ? 1 : NT];
        const half* x1[FRAG ? 1 : NT];
        const uint32_t* xp[FRAG ? NT : 1];
        if constexpr (FRAG) {
#pragma unroll
            for (int p = 0; p < NT; ++p)
                xp[p] = reinterpret_cast<const uint32_t*>(xh) + (size_t)min((m0 >> 4) + p, (M - 1) >> 4) * (K >> 4) * 128 + lane * 4;
        } else {
#pragma unroll
            for (int p = 0; p < NT; ++p) {
                x0[p] = xh + (size_t)min(m0 + 16 * p + g, M - 1) * K;
                x1[p] = xh + (size_t)min(m0 + 16 * p + g + 8, M - 1) * K;
            }
        }
#pragma unroll 1
        for (int hh = 0; hh < H; ++hh) {
            float acc[NT][JN][2][4];
#pragma unroll
            for (int p = 0; p < NT; ++p)
#pragma unroll
                for (int i = 0; i < JN; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int c = 0; c < 4; ++c) acc[p][i][h][c] = 0.f;

            const uint32_t* tile = tiles + (size_t)kt0 * stride_k + (size_t)(hh * JN) * TW;
            if constexpr (RING) {
                // Each warp keeps RS of its k steps in flight: lane 0 arms a slot's barrier and bulk-copies the step's STEP_BYTES into it,
                // every lane waits on the barrier, reads its words from the slot, and lane 0 refills the slot once all lanes are done.
                // The words a lane decodes and the order of the mmas are those of the register-prefetch loop below.
                const char* const ring_w = ring + (size_t)warp * RS * STEP_BYTES;
                const uint32_t slot_s = smem_addr(ring_w), bar_s = smem_addr(bars + warp * RS);
                if (lane == 0) {
                    for (int s = 0; s < RS; ++s) mbar_init(bar_s + 8 * s, 1);
                    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                __syncwarp();
                if (lane == 0) {
                    for (int s = 0; s < RS; ++s) {
                        mbar_expect_tx(bar_s + 8 * s, STEP_BYTES);
                        bulk_g2s(slot_s + s * STEP_BYTES, tile + (size_t)s * stride_k, STEP_BYTES, bar_s + 8 * s);
                    }
                }
                int off[LW];                                         // this lane's word offsets inside a tile
                {
                    int word, offset;
                    lane_start<K2>(lane, word, offset);
#pragma unroll
                    for (int q = 0; q < LW; ++q) off[q] = (word + q) % TW;
                }
                pdl_wait();                                          // the rotated input (xh) or x itself (FI) is the predecessor's output
                // FI: this warp's rotated 128-column block [M][kAPitch] and the lane's two A rows of it (g and g + 8, clamped as x0 / x1 are)
                half* const ab = reinterpret_cast<half*>(ring + (size_t)WK * RS * STEP_BYTES) + (size_t)warp * M * kAPitch;
                const half* const ar0 = ab + (size_t)min(g, M - 1) * kAPitch + 2 * t;
                const half* const ar1 = ab + (size_t)min(g + 8, M - 1) * kAPitch + 2 * t;
                uint32_t a[4];
                if constexpr (FI) {
                    rot_chunk<4>(xin, x_dtype, suh, ab, M, K, kt0 >> 3, lane);
                    __syncwarp();                                    // the block is written by every lane, read by every lane
                    const int o = (kt0 & 7) * 16;
                    a[0] = *reinterpret_cast<const uint32_t*>(ar0 + o);
                    a[1] = *reinterpret_cast<const uint32_t*>(ar1 + o);
                    a[2] = *reinterpret_cast<const uint32_t*>(ar0 + o + 8);
                    a[3] = *reinterpret_cast<const uint32_t*>(ar1 + o + 8);
                } else {
                    a[0] = __ldg(reinterpret_cast<const uint32_t*>(x0[0] + kt0 * 16 + 2 * t));
                    a[1] = __ldg(reinterpret_cast<const uint32_t*>(x1[0] + kt0 * 16 + 2 * t));
                    a[2] = __ldg(reinterpret_cast<const uint32_t*>(x0[0] + kt0 * 16 + 2 * t + 8));
                    a[3] = __ldg(reinterpret_cast<const uint32_t*>(x1[0] + kt0 * 16 + 2 * t + 8));
                }
                int slot = 0;
                uint32_t phase = 0;
#pragma unroll 1
                for (int i = 0; i < per_warp; ++i) {
                    const int kt = kt0 + i;
                    uint32_t an[4] = {0u, 0u, 0u, 0u};
                    if (i + 1 < per_warp) {                          // the next step's A fragment, one step ahead
                        if constexpr (FI) {
                            if (((kt + 1) & 7) == 0) {               // the next step opens the next 128-column block: rotate it first,
                                __syncwarp();                        // once every lane has its words of the finished one (it waits for
                                rot_chunk<4>(xin, x_dtype, suh, ab, M, K, (kt + 1) >> 3, lane);   // the weights anyway)
                                __syncwarp();
                            }
                            const int o = ((kt + 1) & 7) * 16;
                            an[0] = *reinterpret_cast<const uint32_t*>(ar0 + o);
                            an[1] = *reinterpret_cast<const uint32_t*>(ar1 + o);
                            an[2] = *reinterpret_cast<const uint32_t*>(ar0 + o + 8);
                            an[3] = *reinterpret_cast<const uint32_t*>(ar1 + o + 8);
                        } else {
                            an[0] = __ldg(reinterpret_cast<const uint32_t*>(x0[0] + (kt + 1) * 16 + 2 * t));
                            an[1] = __ldg(reinterpret_cast<const uint32_t*>(x1[0] + (kt + 1) * 16 + 2 * t));
                            an[2] = __ldg(reinterpret_cast<const uint32_t*>(x0[0] + (kt + 1) * 16 + 2 * t + 8));
                            an[3] = __ldg(reinterpret_cast<const uint32_t*>(x1[0] + (kt + 1) * 16 + 2 * t + 8));
                        }
                    }
                    mbar_wait(bar_s + 8 * slot, phase);
                    const uint32_t* const sm = reinterpret_cast<const uint32_t*>(ring_w + (size_t)slot * STEP_BYTES);
                    constexpr int SR = step_regs<K2, JN>();
                    uint32_t raw[step_shuffled<K2>() ? SR : 1];
                    if constexpr (step_shuffled<K2>()) {
#pragma unroll
                        for (int c = 0; c < SR; ++c) raw[c] = sm[c * 32 + lane];
                    }
#pragma unroll
                    for (int j = 0; j < JN; ++j) {
                        uint32_t w[LW];
                        if constexpr (step_shuffled<K2>()) {
                            step_lane_words<K2, JN>(raw, j, lane, prev, w);
                        } else {
#pragma unroll
                            for (int q = 0; q < LW; ++q) w[q] = sm[j * TW + off[q]];
                        }
                        uint32_t b0[2], b1[2];
                        decode_lane<K2, CB>(w, lane, b0, b1);
                        mma16816(acc[0][j][0], a, b0);
                        mma16816(acc[0][j][1], a, b1);
                    }
                    __syncwarp();                                    // every lane is done with the slot
                    if (lane == 0 && i + RS < per_warp) {
                        mbar_expect_tx(bar_s + 8 * slot, STEP_BYTES);
                        bulk_g2s(slot_s + slot * STEP_BYTES, tile + (size_t)(i + RS) * stride_k, STEP_BYTES, bar_s + 8 * slot);
                    }
#pragma unroll
                    for (int q = 0; q < 4; ++q) a[q] = an[q];
                    if (++slot == RS) {
                        slot = 0;
                        phase ^= 1u;
                    }
                }
            } else {
                // up to 6 bits the next k step's words are loaded while this one is decoded; 7 and 8 bits load per tile
                constexpr bool PF = K2 <= 12;
                constexpr int SR = step_regs<K2, JN>();
                uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
                if constexpr (PF) {
                    load_step<K2, JN>(tile, lane, cur);
                } else {
                    // 7 and 8 bits load per tile inside the loop: bring the first steps' tiles (a step's JN tiles are contiguous) into L2,
                    // 128 bytes a lane, while the predecessor is still running
                    constexpr int kLines = JN * TW * 4 / 128;
                    static_assert(kLines <= 32, "a step's tiles fit one line a lane");
#pragma unroll
                    for (int s = 0; s < 4; ++s)
                        if (s < per_warp && lane < kLines) prefetch_l2(tile + (size_t)s * stride_k + lane * 32);
                }
                pdl_wait();                                          // the rotated input (xh) is the predecessor's output
#pragma unroll 1
                for (int i = 0; i < per_warp; ++i) {
                    const int kt = kt0 + i;
                    uint32_t a[NT][4];
#pragma unroll
                    for (int p = 0; p < NT; ++p) {
                        if constexpr (FRAG) {
                            const uint4 v = __ldg(reinterpret_cast<const uint4*>(xp[p] + (size_t)kt * 128));
                            a[p][0] = v.x; a[p][1] = v.y; a[p][2] = v.z; a[p][3] = v.w;
                        } else {
                            a[p][0] = __ldg(reinterpret_cast<const uint32_t*>(x0[p] + kt * 16 + 2 * t));
                            a[p][1] = __ldg(reinterpret_cast<const uint32_t*>(x1[p] + kt * 16 + 2 * t));
                            a[p][2] = __ldg(reinterpret_cast<const uint32_t*>(x0[p] + kt * 16 + 2 * t + 8));
                            a[p][3] = __ldg(reinterpret_cast<const uint32_t*>(x1[p] + kt * 16 + 2 * t + 8));
                        }
                    }
                    if constexpr (PF) {
                        if (i + 1 < per_warp) load_step<K2, JN>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
                    }
#pragma unroll
                    for (int j = 0; j < JN; ++j) {
                        uint32_t w[LW];
                        if constexpr (PF) step_lane_words<K2, JN>(cur, j, lane, prev, w);
                        else ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
                        uint32_t b0[2], b1[2];
                        decode_lane<K2, CB>(w, lane, b0, b1);
#pragma unroll
                        for (int p = 0; p < NT; ++p) {
                            mma16816(acc[p][j][0], a[p], b0);
                            mma16816(acc[p][j][1], a[p], b1);
                        }
                    }
                    if constexpr (PF) {
#pragma unroll
                        for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
                    }
                }
            }

            pdl_trigger();                                           // xh is read: the successor may launch

            // the warps' sums, added in warp order, one 16-row pass at a time: rows 0-7 and then rows 8-15 of it
#pragma unroll
            for (int p = 0; p < NT; ++p) {
                const int mp = m0 + 16 * p;
                if (mp >= M) break;
                const int R = min(16, M - mp);
                for (int rlo = 0; rlo < R; rlo += 8) {
                    const int rn = min(R - rlo, 8);
                    __syncthreads();                     // red is reused by every half and pass
                    if (g < RH) {
#pragma unroll
                        for (int i = 0; i < JN; ++i)
#pragma unroll
                            for (int h = 0; h < 2; ++h) {
                                const int col = i * 16 + h * 8 + 2 * t;
                                *reinterpret_cast<float2*>(red + (warp * RH + g) * RC + col) =
                                    rlo ? make_float2(acc[p][i][h][2], acc[p][i][h][3]) : make_float2(acc[p][i][h][0], acc[p][i][h][1]);
                            }
                    }
                    __syncthreads();

                    if constexpr (H == 1) {
                        if (SK == 1 && !ZM) {
                            for (int r = warp; r < rn; r += WK) {
                                float v[4];
                                const float4 u = *reinterpret_cast<const float4*>(red + r * RC + 4 * lane);
                                v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                                for (int w = 1; w < WK; ++w) {
                                    const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * RC + 4 * lane);
                                    v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                                }
                                finish(v, lane, svh, bias, col0 + 4 * lane);
                                store4(y, y_dtype, (size_t)(mp + rlo + r) * N + col0 + 4 * lane, v);
                            }
                            continue;
                        }
                    }
                    for (int idx = threadIdx.x; idx < rn * (RC / 4); idx += WK * 32) {
                        const int r = idx / (RC / 4), c = 4 * (idx % (RC / 4));
                        float4 s = *reinterpret_cast<const float4*>(red + r * RC + c);
#pragma unroll
                        for (int w = 1; w < WK; ++w) {
                            const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * RC + c);
                            s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                        }
                        if (SK > 1 || ZM)
                            *reinterpret_cast<float4*>(Z + ((size_t)split * M + mp + rlo + r) * N + col0 + hh * RC + c) = s;
                        else
                            *reinterpret_cast<float4*>(stage + (16 * p + rlo + r) * 128 + hh * RC + c) = s;
                    }
                }
            }
        }
        if constexpr (H > 1) {
            if (SK == 1 && !ZM) {
                __syncthreads();                         // every sweep's sums are in stage
                const int rows = min(16 * NT, M - m0);
                for (int r = warp; r < rows; r += WK) {
                    const float4 u = *reinterpret_cast<const float4*>(stage + r * 128 + 4 * lane);
                    float v[4] = {u.x, u.y, u.z, u.w};
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + r) * N + col0 + 4 * lane, v);
                }
            }
        }
        __syncthreads();                             // red and stage are reused in the next group
    }
}

// W_q [K, N] fp16 from the trellis words (tile (kt, nt) at kt * stride_k + (nt / 8) * stride_nb): one warp per tile.
template <int K2, int CB>
__global__ void __launch_bounds__(32) unpack_kernel(const uint32_t* __restrict__ T, half* __restrict__ W, int N,
                                                    int64_t stride_k, int64_t stride_nb) {
    const int kt = blockIdx.y, nt = blockIdx.x, lane = threadIdx.x;
    uint32_t w[lane_words<K2>()];
    ldg_lane_words<K2>(T + kt * stride_k + (nt >> 3) * stride_nb + (nt & 7) * tile_words<K2>(), lane, w);
    uint32_t b[2][2];
    decode_lane<K2, CB>(w, lane, b[0], b[1]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t v = b[j >> 2][(j >> 1) & 1];
        const unsigned short h = (j & 1) ? (unsigned short)(v >> 16) : (unsigned short)(v & 0xffffu);
        W[(size_t)(kt * 16 + value_row(lane, j)) * N + nt * 16 + value_col(lane, j)] = __ushort_as_half(h);
    }
}

// (K splits, warps a block) for a K x N layer: the shape's alone, so a layer keeps one reduction for every row count.
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

// workspace: [xf: ceil(M / 16) * 16 * K fp16][Z: SK*M*N fp32, only when SK > 1]; no initialization needed
size_t linear_ws_bytes(int m, int k, int n) {
    const Plan p = plan_for(k, n);
    size_t b = xf_bytes(m, k);
    if (p.sk > 1) b += align256((size_t)p.sk * m * n * sizeof(float));
    return b;
}

bool shape_ok(int k, int n, int bits, int codebook) {
    return k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0 && bits >= 1 && bits <= 8 && codebook >= 0 && codebook <= 2;
}

struct LinArgs {
    const half* xh; const uint32_t* T; long long stride_k, stride_nb; const half* svh; const half* bias; void* y;
    int y_dtype; float* Z; int M, K, N; Plan plan;
    int zmode = 0;                                           // 1: the split partials to Z, nothing finished (gb_exl3_linear_ex)
    bool pdl = false;                                        // the launches carry the programmatic-dependent-launch attribute (decode path)
    const void* x = nullptr; int x_dtype = 0; const half* suh = nullptr;   // the unrotated input and its scales: launch_linear rotates it
    int fuse = 0;                                            // fuse_mask(): the ring kernel may rotate its own input (ring_plan decides)
};

// The decode path's launches carry the PDL attribute when the mask has gbp_pdl::kExl3 (env REXL3_GLM_PDL, hook gb_pdl_set_mask).
inline bool pdl_exl3() { return gbp_pdl::on(gbp_pdl::kExl3); }

// Which launches the decode path folds into the linear kernel (env REXL3_EXL3_FUSE, read once; the test hook gb_exl3_set_fuse; 0, every launch
// as before, until a model sets it): bit 0 is the input rotation of 1 to kFuseRows rows (linear_kernel's FI) on the layers where it pays
// (ring_plan); bit 1 takes every 4-warp layer that fits whatever its grid (how the tests reach the kernel on small shapes). Only the launch
// count and the timing change, never a bit.
enum : int { kFuseIn = 1, kFuseInAll = 2, kFuseAll = 3 };
inline int g_fuse = -1;
inline int fuse_mask() {
    if (g_fuse < 0) {
        const char* e = getenv("REXL3_EXL3_FUSE");
        g_fuse = e ? (atoi(e) & kFuseAll) : 0;
    }
    return g_fuse;
}
template <typename... KArgs, typename... Args>
int launch_k(bool pdl, void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args) {
    return gbp_pdl::launch_pdl(pdl, kernel, grid, block, smem, st, args...);
}

// y from SK fp32 partial sums (see rot_out_kernel).
int rotate_out(const float* c, const half* svh, const half* bias, void* y, int y_dtype, int m, int n, int sk,
               cudaStream_t st, bool pdl = false) {
    const dim3 grid((unsigned)m, (unsigned)((n / 128 + 3) / 4));
    return launch_k(pdl, rot_out_kernel, grid, dim3(128), 0, st, c, svh, bias, y, y_dtype, m, n, sk);
}

// 16-row passes that share a tile decode: one per 16 rows up to 4 (a 64 row window decodes each tile once). Past that, groups
// as even as they can be (80 rows are two groups of 40, 3 passes each, not 64 and a straggler of 16).
constexpr int kMaxGroup = 4;

int group_tiles(int m) {
    const int groups = (m + 16 * kMaxGroup - 1) / (16 * kMaxGroup);
    return (((m + groups - 1) / groups) + 15) / 16;
}

// Column sweeps a group of NT passes takes: 3 and 4 passes keep their accumulators half a block row at a time (3 passes in one
// sweep spill 100 to 240 bytes of registers, 4 cannot be built), and so do 2 passes where the wider 3INST/MCG decode of
// 5 bits and up leaves too few registers for them (a small spill). No instantiation spills (the build test reads ptxas).
template <int K2, int CB>
constexpr int sweeps_of(int nt) {
    return nt >= 3 || (nt == 2 && K2 >= 10 && CB != 2) ? 2 : 1;
}

// Steps of a warp's k range the ring kernel keeps in flight: as many as a block's share of the shared memory holds (the budget below,
// per block, sized for the blocks the registers let an SM host), at least 1 and no more than the warp's range has.
#ifndef RING_BUDGET8
#define RING_BUDGET8 40960
#endif
#ifndef RING_BUDGET4
#define RING_BUDGET4 22528
#endif
#ifndef RING_MAXS
#define RING_MAXS 3
#endif
inline int ring_stages(int k2, int wk, int per_warp) {
    const int s = (int)((size_t)(wk == 8 ? RING_BUDGET8 : RING_BUDGET4) / ((size_t)wk * 128 * k2));
    const int cap = per_warp < RING_MAXS ? per_warp : RING_MAXS;
    return s < 1 ? 1 : (s > cap ? cap : s);
}

int rotate_in(const void* x, int x_dtype, const void* suh, void* xh, int m, int k, cudaStream_t st, bool pdl = false) {
    const dim3 grid((unsigned)m, (unsigned)((k / 128 + 3) / 4));
    return launch_k(pdl, rot_in_kernel, grid, dim3(128), 0, st, x, x_dtype, (const half*)suh, (half*)xh, k);
}

// the same, into the decode kernel's fragment order (see rot_in_frag_kernel)
int rotate_in_frag(const void* x, int x_dtype, const void* suh, void* xf, int m, int k, cudaStream_t st, bool pdl = false) {
    const dim3 grid((unsigned)((m + 15) / 16 * 16), (unsigned)((k / 128 + 3) / 4));
    return launch_k(pdl, rot_in_frag_kernel, grid, dim3(128), 0, st, x, x_dtype, (const half*)suh, (uint32_t*)xf, k, m);
}

// What a ring launch (M <= 16) runs: its stages (RS), its dynamic shared memory, and whether the kernel rotates its own input (FI, see
// linear_kernel). That needs the window's rows in shared memory beside the ring, so FI takes the unfused stage count where that fits what an
// SM hosts (RING_MINB4 blocks of the size below), fewer stages (down to 2) where it does not, and is not taken at all (the rotation stays a
// launch of its own) for more than kFuseRows rows or where two stages do not fit either.
struct RingPlan {
    int rs;
    size_t smem;
    bool fused;
};

#ifndef RING_FUSED_BUDGET4
#define RING_FUSED_BUDGET4 24576
#endif

// Layers of 4-warp blocks whose grid is at least this many blocks (the ones whose N is narrow enough to split K and wide enough to fill the SMs:
// z, out_proj and o_proj of the Flash-Next pack) take the fused rotation under kFuseIn: measured on the decode window, it removes their rot_in
// launch and its hop and the loop pays nothing for it, where the 8-warp layers (qkv, q) and the layers of a few blocks (k, v, the indexer) pay
// the rotation's latency in the kernel and lose what the launch saved. The 8-warp blocks never fuse (their instances are not built).
constexpr int kFuseMinBlocks = 128;

template <int K2, int WK>
RingPlan ring_plan(const LinArgs& a, int rh) {
    const int rs = ring_stages(K2, WK, (a.K >> 4) / a.plan.sk / WK);
    if constexpr (WK == 4) {
        const bool pays = (a.N / 128) * a.plan.sk >= kFuseMinBlocks;
        if (a.M <= kFuseRows && (((a.fuse & kFuseIn) && pays) || (a.fuse & kFuseInAll))) {
            for (int s = rs; s >= (rs < 2 ? rs : 2); --s) {
                const size_t smem = ring_region<K2, WK>(s, rh, a.M) + (size_t)WK * s * 8;
                if (smem <= RING_FUSED_BUDGET4) return {s, smem, true};
            }
        }
    }
    return {rs, ring_region<K2, WK>(rs, rh) + (size_t)WK * rs * 8, false};
}

template <int K2>
RingPlan ring_plan_of(const LinArgs& a, int rh) {
    return a.plan.wk == 4 ? ring_plan<K2, 4>(a, rh) : ring_plan<K2, 8>(a, rh);
}

template <int K2, int CB, int NT, int H>
void launch_linear_nt(const LinArgs& a, const dim3& grid, int rh, const RingPlan& rp, cudaStream_t st) {
    constexpr int RC = (8 / H) * 16;
    constexpr bool RING = NT == 1 && H == 1;
    // the stage of finished sums (16 NT rows x 128 columns) exists when a block rotates and stores the rows itself
    const size_t stage = (H > 1 && a.plan.sk == 1 && !a.zmode) ? (size_t)16 * NT * 128 * sizeof(float) : 0;
    const int rs = RING ? rp.rs : 1;
    if (a.plan.wk == 4) {
        const size_t smem = RING ? rp.smem : 4 * rh * RC * sizeof(float) + stage;
        if constexpr (RING) {
            if (rp.fused) {
                launch_k(a.pdl, linear_kernel<K2, CB, 4, NT, H, true>, grid, dim3(128), smem, st, a.xh, a.T, a.stride_k, a.stride_nb, a.svh,
                         a.bias, a.y, a.y_dtype, a.Z, a.M, a.K, a.N, a.plan.sk, a.zmode, rs, a.x, a.x_dtype, a.suh);
                return;
            }
        }
        launch_k(a.pdl, linear_kernel<K2, CB, 4, NT, H>, grid, dim3(128), smem, st, a.xh, a.T, a.stride_k, a.stride_nb, a.svh, a.bias,
                 a.y, a.y_dtype, a.Z, a.M, a.K, a.N, a.plan.sk, a.zmode, rs, a.x, a.x_dtype, a.suh);
    } else {
        const size_t smem = RING ? rp.smem : 8 * rh * RC * sizeof(float) + stage;
        launch_k(a.pdl, linear_kernel<K2, CB, 8, NT, H>, grid, dim3(256), smem, st, a.xh, a.T, a.stride_k, a.stride_nb, a.svh, a.bias,
                 a.y, a.y_dtype, a.Z, a.M, a.K, a.N, a.plan.sk, a.zmode, rs, a.x, a.x_dtype, a.suh);
    }
}

template <int K2, int CB>
int launch_linear(const LinArgs& a, cudaStream_t st) {
    const int nt = group_tiles(a.M);
    if (nt == 1 && (reinterpret_cast<uintptr_t>(a.T) & 15)) return (int)cudaErrorInvalidValue;   // the ring copies 16-byte pieces
    const dim3 grid((unsigned)(a.N / 128), (unsigned)a.plan.sk);
    const int rh = a.M < 8 ? a.M : 8;
    const RingPlan rp = nt == 1 ? ring_plan_of<K2>(a, rh) : RingPlan{1, 0, false};
    if (!rp.fused) {
        // the rotated input: more than one 16-row pass reads its A fragments in fragment order; one pass (up to 16 rows, the
        // bandwidth-bound case) reads the rows themselves, which for a handful of rows is less to fetch than whole padded tiles
        const int rc = nt >= 2 ? rotate_in_frag(a.x, a.x_dtype, a.suh, const_cast<half*>(a.xh), a.M, a.K, st, a.pdl)
                               : rotate_in(a.x, a.x_dtype, a.suh, const_cast<half*>(a.xh), a.M, a.K, st, a.pdl);
        if (rc) return rc;
    }
    switch (nt) {
        case 4: launch_linear_nt<K2, CB, 4, sweeps_of<K2, CB>(4)>(a, grid, rh, rp, st); break;
        case 3: launch_linear_nt<K2, CB, 3, sweeps_of<K2, CB>(3)>(a, grid, rh, rp, st); break;
        case 2: launch_linear_nt<K2, CB, 2, sweeps_of<K2, CB>(2)>(a, grid, rh, rp, st); break;
        default: launch_linear_nt<K2, CB, 1, sweeps_of<K2, CB>(1)>(a, grid, rh, rp, st); break;
    }
    const int rc = GB_CHECK_LAUNCH();
    if (rc || a.plan.sk == 1 || a.zmode) return rc;
    return rotate_out(a.Z, a.svh, a.bias, a.y, a.y_dtype, a.M, a.N, a.plan.sk, st, a.pdl);   // the K splits, added in order
}

template <int K2, int CB>
int launch_unpack(const uint32_t* T, half* W, int k, int n, cudaStream_t st) {
    const dim3 grid((unsigned)(n / 16), (unsigned)(k / 16));
    constexpr int tw = tile_words<K2>();
    unpack_kernel<K2, CB><<<grid, 32, 0, st>>>(T, W, n, (int64_t)(n / 16) * tw, 8 * tw);
    return GB_CHECK_LAUNCH();
}

#define GB_EXL3_FOR_CB(X, K2) X(K2, 0) X(K2, 1) X(K2, 2)
#ifdef GB_EXL3_ONLY
// Microbenchmark builds (tools/exl3_bench.cu) name the (K2, CB) pairs they instantiate: `-D'GB_EXL3_ONLY(X)=X(8, 2)'`.
#define GB_EXL3_FOR_ALL(X) GB_EXL3_ONLY(X)
#else
#define GB_EXL3_FOR_ALL(X)                                                                                          \
    GB_EXL3_FOR_CB(X, 2) GB_EXL3_FOR_CB(X, 4) GB_EXL3_FOR_CB(X, 6) GB_EXL3_FOR_CB(X, 8) GB_EXL3_FOR_CB(X, 10)       \
    GB_EXL3_FOR_CB(X, 12) GB_EXL3_FOR_CB(X, 14) GB_EXL3_FOR_CB(X, 16) X(3, 2) X(5, 2) X(7, 2)
#endif

int dispatch_linear(int bits, int cb, const LinArgs& a, cudaStream_t st) {
#define GB_L(K2_, CB_) if (2 * bits == K2_ && cb == CB_) return launch_linear<K2_, CB_>(a, st);
    GB_EXL3_FOR_ALL(GB_L)
#undef GB_L
    return (int)cudaErrorInvalidValue;
}

int dispatch_unpack(int bits, int cb, const uint32_t* T, half* W, int k, int n, cudaStream_t st) {
#define GB_U(K2_, CB_) if (2 * bits == K2_ && cb == CB_) return launch_unpack<K2_, CB_>(T, W, k, n, st);
    GB_EXL3_FOR_ALL(GB_U)
#undef GB_U
    return (int)cudaErrorInvalidValue;
}

// Half-integer rates: k2 = 2 * bits for integer 1..8 bits; odd k2 = 3, 5, 7 are the 1.5 / 2.5 / 3.5 bit
// rates (mul1 codebook only). Even k2 accepts every codebook.
bool shape_ok_k2(int k, int n, int k2, int codebook) {
    if (!(k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0 && codebook >= 0 && codebook <= 2)) return false;
    if (k2 & 1) return (k2 == 3 || k2 == 5 || k2 == 7) && codebook == 2;
    return k2 >= 2 && k2 <= 16;
}

// dispatch_linear / dispatch_unpack keyed by k2 (half-bit units) instead of bits: k2 is the template K2
// directly, so the odd mul1 instantiations are reachable.
int dispatch_linear_k2(int k2, int cb, const LinArgs& a, cudaStream_t st) {
#define GB_L(K2_, CB_) if (k2 == K2_ && cb == CB_) return launch_linear<K2_, CB_>(a, st);
    GB_EXL3_FOR_ALL(GB_L)
#undef GB_L
    return (int)cudaErrorInvalidValue;
}

int dispatch_unpack_k2(int k2, int cb, const uint32_t* T, half* W, int k, int n, cudaStream_t st) {
#define GB_U(K2_, CB_) if (k2 == K2_ && cb == CB_) return launch_unpack<K2_, CB_>(T, W, k, n, st);
    GB_EXL3_FOR_ALL(GB_U)
#undef GB_U
    return (int)cudaErrorInvalidValue;
}

bool dtype_ok(int d) { return d >= 0 && d <= 2; }

// One cuBLAS handle per host thread, created on first use (the only allocation the library ever makes) and never
// destroyed: destroying it at thread exit can run after the CUDA context is gone.
cublasHandle_t cublas_handle() {
    static thread_local cublasHandle_t h = nullptr;
    if (!h && cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) h = nullptr;
    return h;
}

cublasLtHandle_t lt_handle() {
    static thread_local cublasLtHandle_t h = nullptr;
    if (!h && cublasLtCreate(&h) != CUBLAS_STATUS_SUCCESS) h = nullptr;
    return h;
}

// The pinned prefill GEMM. cublasGemmEx picks its kernel per call from (M, N, K): above 16 rows the picks agree to the
// bit on this device and cuBLAS version, but at 1-16 rows (and 33-64 for K = 17408) they switch to split-K and CUDA-core
// kernels, which sum K in another order, and nothing in the API says the picks keep agreeing. Here the algorithm is ours
// instead: cublasLt algo 67 (the tensor-core GEMM family), SPLIT-K 1, no reduction scheme, no workspace, and a tile, stage
// count and custom option fixed per (K, N). With one K split and no stream-K every output element is one sequential chain
// of m16n8k16 MMAs over K, whatever M is, whichever row of the call it is, and however many CTAs the grid has: M only changes
// how many tiles are launched. Algorithms are not interchangeable in general (the 64x64-tile ones of this family differ in
// every row), which is why the algorithm is pinned here and the sweep test (M = 1..4096, every prefill shape, the bits of
// each row against a 4096-row call) is what proves the claim on a given cuBLAS. The tile choices below are the fastest of the
// algorithms that were bit-identical to cublasGemmEx's output at M = 4096 (measured per shape at M = 129 .. 4096 on GB10).
struct Pin {
    uint32_t tile, stages, custom;
};
constexpr int kPinAlgoId = 67;
constexpr Pin kPinDefault = {CUBLASLT_MATMUL_TILE_128x128, CUBLASLT_MATMUL_STAGES_64xAUTO, 30};
// down_proj (K = 17408, N = 5120): 128x224 tiles are 10% faster than 128x128 at M = 4096.
constexpr Pin kPinDown = {CUBLASLT_MATMUL_TILE_128x224, CUBLASLT_MATMUL_STAGES_64xAUTO, 22};
Pin pin_for(int k, int n) { return (k == 17408 && n == 5120) ? kPinDown : kPinDefault; }
constexpr const char* kPinName =
    "cublasLt algo 67, split-K 1, no reduction, no workspace; tiles 128x128 stages 64xAUTO custom 30, "
    "K=17408 N=5120: 128x224 stages 64xAUTO custom 22";

// One fp16 GEMM problem as cublasLt sees it: c [n, m] (column-major, i.e. [m, n] row-major) = w [n, k] @ xh [k, m].
struct LtProblem {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo;
    LtProblem() = default;
    LtProblem(const LtProblem&) = delete;
    LtProblem& operator=(const LtProblem&) = delete;
    ~LtProblem() {
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (desc) cublasLtMatmulDescDestroy(desc);
    }
    int init(cublasLtHandle_t lt, int m, int k, int n) {
        const cublasOperation_t no = CUBLAS_OP_N;
        if (cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorUnknown;
        if (cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &no, sizeof(no)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &no, sizeof(no)) != CUBLAS_STATUS_SUCCESS)
            return (int)cudaErrorUnknown;
        if (cublasLtMatrixLayoutCreate(&a, CUDA_R_16F, n, k, n) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&b, CUDA_R_16F, k, m, k) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, n, m, n) != CUBLAS_STATUS_SUCCESS)
            return (int)cudaErrorUnknown;
        if (cublasLtMatmulAlgoInit(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F,
                                   kPinAlgoId, &algo) != CUBLAS_STATUS_SUCCESS)
            return (int)cudaErrorNotSupported;
        const uint32_t splitk = 1, red = CUBLASLT_REDUCTION_SCHEME_NONE, swizzle = 0;
        const Pin pin = pin_for(k, n);
        auto set = [&](cublasLtMatmulAlgoConfigAttributes_t attr, const uint32_t& v) {
            return cublasLtMatmulAlgoConfigSetAttribute(&algo, attr, &v, sizeof(v)) == CUBLAS_STATUS_SUCCESS;
        };
        if (!set(CUBLASLT_ALGO_CONFIG_TILE_ID, pin.tile) || !set(CUBLASLT_ALGO_CONFIG_STAGES_ID, pin.stages) ||
            !set(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, pin.custom) || !set(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, splitk) ||
            !set(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, red) || !set(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, swizzle))
            return (int)cudaErrorNotSupported;
        return 0;
    }
};

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Decode path: 1..128 rows, row-invariant.
// ---------------------------------------------------------------------------------------------------------------

// Bytes of workspace gb_exl3_linear needs (rotated input, split-K partials); contents on entry do not matter, so
// one workspace serves every layer on a stream. Returns 0 for an unsupported shape.
GB_EXPORT size_t gb_exl3_linear_workspace_bytes(int m, int k, int n, int bits) {
    if (m < 1 || m > kMaxRows || !shape_ok(k, n, bits, 0)) return 0;
    return linear_ws_bytes(m, k, n);
}

// y [m, n] = EXL3 layer applied to x [m, k], for 1 <= m <= 128. bias may be null. All device pointers.
GB_EXPORT int gb_exl3_linear(const void* x, int x_dtype, const void* trellis, const void* suh, const void* svh,
                             const void* bias, void* y, int y_dtype, int m, int k, int n, int bits, int codebook,
                             void* workspace, size_t workspace_bytes, void* stream) {
    if (m < 1 || m > kMaxRows || !shape_ok(k, n, bits, codebook) || !dtype_ok(x_dtype) || !dtype_ok(y_dtype) || !x ||
        !trellis || !suh || !svh || !y || !workspace)
        return (int)cudaErrorInvalidValue;
    const Plan p = plan_for(k, n);
    if (workspace_bytes < linear_ws_bytes(m, k, n)) return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    char* ws = static_cast<char*>(workspace);
    half* xh = reinterpret_cast<half*>(ws);
    float* Z = p.sk > 1 ? reinterpret_cast<float*>(ws + xf_bytes(m, k)) : nullptr;
    const int tw = 8 * bits;                                 // words a tile: 4 * K2
    LinArgs a{xh, static_cast<const uint32_t*>(trellis), (long long)(n / 16) * tw, 8LL * tw, (const half*)svh,
              (const half*)bias, y, y_dtype, Z, m, k, n, p};
    a.pdl = pdl_exl3();
    a.x = x;
    a.x_dtype = x_dtype;
    a.suh = static_cast<const half*>(suh);
    a.fuse = fuse_mask();
    return dispatch_linear(bits, codebook, a, st);
}

// gb_exl3_linear keyed by k2 (half-bit units) instead of bits: even k2 = 2..16 for every codebook, odd k2 =
// 3 / 5 / 7 (1.5 / 2.5 / 3.5 bits) for the mul1 codebook. Same workspace layout as gb_exl3_linear.
GB_EXPORT int gb_exl3_linear_k2(const void* x, int x_dtype, const void* trellis, const void* suh, const void* svh,
                                const void* bias, void* y, int y_dtype, int m, int k, int n, int k2, int codebook,
                                void* workspace, size_t workspace_bytes, void* stream) {
    if (m < 1 || m > kMaxRows || !shape_ok_k2(k, n, k2, codebook) || !dtype_ok(x_dtype) || !dtype_ok(y_dtype) || !x ||
        !trellis || !suh || !svh || !y || !workspace)
        return (int)cudaErrorInvalidValue;
    const Plan p = plan_for(k, n);
    if (workspace_bytes < linear_ws_bytes(m, k, n)) return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    char* ws = static_cast<char*>(workspace);
    half* xh = reinterpret_cast<half*>(ws);
    float* Z = p.sk > 1 ? reinterpret_cast<float*>(ws + xf_bytes(m, k)) : nullptr;
    const int tw = 4 * k2;                                 // words a tile: 4 * K2
    LinArgs a{xh, static_cast<const uint32_t*>(trellis), (long long)(n / 16) * tw, 8LL * tw, (const half*)svh,
              (const half*)bias, y, y_dtype, Z, m, k, n, p};
    a.pdl = pdl_exl3();
    a.x = x;
    a.x_dtype = x_dtype;
    a.suh = static_cast<const half*>(suh);
    a.fuse = fuse_mask();
    return dispatch_linear_k2(k2, codebook, a, st);
}

// ---------------------------------------------------------------------------------------------------------------
// The tensor-parallel splits of a layer (lane E, GLM Phase 2: docs in exl3_linear.rs, "Strips and K splits"). Two exact cuts of a K x N
// layer, both run by the decode kernel above with the PARENT's plan (sk, wk) instead of their own shape's:
//   * a COLUMN STRIP (whole 128-column blocks): the parent's trellis read in place (stride_n = the parent's N, trellis / svh / bias at the
//     strip's first block) or a copy of the strip; every column's reduction is the parent's, so its bits are the parent's at any width;
//   * a K RANGE of whole splits [s0, s1) of the parent's plan: the trellis rows of those splits (contiguous: tile rows are stride_k apart),
//     the matching input columns and suh, and the plan (s1 - s0, wk): its splits' partials are the parent's partials s0 .. s1 - 1, bit for
//     bit, so partials gathered from every range and added in split order (gb_exl3_rotate_out_sk) are the parent's output.
// ---------------------------------------------------------------------------------------------------------------

namespace {

bool plan_ok(int k, int sk, int wk) {
    const int kt = k / 16;
    return sk >= 1 && sk <= 64 && (wk == 4 || wk == 8) && kt % (sk * wk) == 0;
}

}  // namespace

// Workspace of gb_exl3_linear_ex: the rotated input, plus the split partials when sk > 1 and they are not written to a caller's z.
GB_EXPORT size_t gb_exl3_linear_ex_workspace_bytes(int m, int k, int n, int sk, int partials) {
    if (m < 1 || m > kMaxRows || !shape_ok(k, n, 1, 0) || sk < 1) return 0;
    size_t b = xf_bytes(m, k);
    if (!partials && sk > 1) b += align256((size_t)sk * m * n * sizeof(float));
    return b;
}

// y [m, n] (or the partials z) of an EXL3 layer on x [m, k], 1 <= m <= 128, with the plan (sk, wk) (0, 0: plan_for(k, n), the plain
// layer). stride_n: N of the trellis rows (n for a layer stored whole; a column strip of a wider parent read in place passes the parent's
// N, with trellis, svh and bias pointing at the strip's first 128-column block). z: when not null, the sk fp32 partials go to z[s][m][n]
// (s = 0 .. sk-1, the warp-order sum of split s) and nothing is finished, also when sk == 1; when null the layer finishes into y
// (gb_exl3_linear's path). All device pointers; bias may be null.
GB_EXPORT int gb_exl3_linear_ex(const void* x, int x_dtype, const void* trellis, int stride_n, const void* suh, const void* svh,
                                const void* bias, void* y, int y_dtype, int m, int k, int n, int bits, int codebook, int sk, int wk,
                                void* z, void* workspace, size_t workspace_bytes, void* stream) {
    if (m < 1 || m > kMaxRows || !shape_ok(k, n, bits, codebook) || !dtype_ok(x_dtype) || !dtype_ok(y_dtype) || !x || !trellis ||
        !suh || !svh || (!y && !z) || !workspace || stride_n < n || stride_n % 128)
        return (int)cudaErrorInvalidValue;
    const Plan p = (sk == 0 && wk == 0) ? plan_for(k, n) : Plan{sk, wk};
    if (!plan_ok(k, p.sk, p.wk)) return (int)cudaErrorInvalidValue;
    if (workspace_bytes < gb_exl3_linear_ex_workspace_bytes(m, k, n, p.sk, z != nullptr)) return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    char* ws = static_cast<char*>(workspace);
    half* xh = reinterpret_cast<half*>(ws);
    float* Z = z ? static_cast<float*>(z) : (p.sk > 1 ? reinterpret_cast<float*>(ws + xf_bytes(m, k)) : nullptr);
    const int tw = 8 * bits;
    LinArgs a{xh, static_cast<const uint32_t*>(trellis), (long long)(stride_n / 16) * tw, 8LL * tw, (const half*)svh,
              (const half*)bias, y, y_dtype, Z, m, k, n, p};
    a.zmode = z != nullptr;
    a.pdl = pdl_exl3();
    a.x = x;
    a.x_dtype = x_dtype;
    a.suh = static_cast<const half*>(suh);
    a.fuse = fuse_mask();
    return dispatch_linear(bits, codebook, a, st);
}

// y [m, n] = (c H / sqrt(128)) * svh + bias from sk fp32 partials c [sk][m][n] added in split order (the decode path's second kernel):
// the finish of partials gathered from the K ranges of a layer. bias may be null.
GB_EXPORT int gb_exl3_rotate_out_sk(const void* c, const void* svh, const void* bias, void* y, int y_dtype, int m, int n, int sk,
                                    void* stream) {
    if (m < 1 || n <= 0 || n % 128 || sk < 1 || sk > 64 || !dtype_ok(y_dtype) || !c || !svh || !y) return (int)cudaErrorInvalidValue;
    return rotate_out((const float*)c, (const half*)svh, (const half*)bias, y, y_dtype, m, n, sk, gb_stream(stream));
}

// ---------------------------------------------------------------------------------------------------------------
// Prefill path (heuristic GEMM, used by the draft and the vision encoder): reconstruct W_q, rotate, cublasGemmEx, rotate. Not
// row-invariant (cuBLAS picks its kernel by shape). The target model's prefill uses the pinned path further down.
// ---------------------------------------------------------------------------------------------------------------

// W_q [k, n] fp16 row-major from the trellis.
GB_EXPORT int gb_exl3_reconstruct(const void* trellis, void* w, int k, int n, int bits, int codebook, void* stream) {
    if (!shape_ok(k, n, bits, codebook) || !trellis || !w) return (int)cudaErrorInvalidValue;
    return dispatch_unpack(bits, codebook, static_cast<const uint32_t*>(trellis), static_cast<half*>(w), k, n,
                           gb_stream(stream));
}

// gb_exl3_reconstruct keyed by k2 (half-bit units) instead of bits: even k2 = 2..16 for every codebook, odd
// k2 = 3 / 5 / 7 (1.5 / 2.5 / 3.5 bits) for the mul1 codebook.
GB_EXPORT int gb_exl3_reconstruct_k2(const void* trellis, void* w, int k, int n, int k2, int codebook, void* stream) {
    if (!shape_ok_k2(k, n, k2, codebook) || !trellis || !w) return (int)cudaErrorInvalidValue;
    return dispatch_unpack_k2(k2, codebook, static_cast<const uint32_t*>(trellis), static_cast<half*>(w), k, n,
                              gb_stream(stream));
}

// xh [m, k] fp16 = (x * suh) H / sqrt(128).
GB_EXPORT int gb_exl3_rotate_in(const void* x, int x_dtype, const void* suh, void* xh, int m, int k, void* stream) {
    if (m < 1 || k <= 0 || k % 128 || !dtype_ok(x_dtype) || !x || !suh || !xh) return (int)cudaErrorInvalidValue;
    return rotate_in(x, x_dtype, suh, xh, m, k, gb_stream(stream));
}

// y [m, n] = (c H / sqrt(128)) * svh + bias from fp32 sums c [m, n]; bias may be null.
GB_EXPORT int gb_exl3_rotate_out(const void* c, const void* svh, const void* bias, void* y, int y_dtype, int m, int n,
                                 void* stream) {
    if (m < 1 || n <= 0 || n % 128 || !dtype_ok(y_dtype) || !c || !svh || !y) return (int)cudaErrorInvalidValue;
    return rotate_out((const float*)c, (const half*)svh, (const half*)bias, y, y_dtype, m, n, 1, gb_stream(stream));
}

// Create this thread's cuBLAS handle (the first prefill call does it otherwise; it allocates, so not under graph capture).
GB_EXPORT int gb_exl3_prefill_init() {
    return cublas_handle() && lt_handle() ? 0 : (int)cudaErrorUnknown;
}

// Scratch of gb_exl3_prefill: [xh: m*k fp16][W_q: k*n fp16][c: m*n fp32], each part 256-byte aligned.
GB_EXPORT size_t gb_exl3_prefill_scratch_bytes(int m, int k, int n) {
    if (m < 1 || k <= 0 || n <= 0) return 0;
    return align256((size_t)m * k * 2) + align256((size_t)k * n * 2) + align256((size_t)m * n * 4);
}

// The prefill GEMM on an already reconstructed W_q [k, n] fp16: c [m, n] fp32 = xh [m, k] fp16 @ W_q, fp32 accumulate.
GB_EXPORT int gb_exl3_prefill_gemm(const void* xh, const void* w, void* c, int m, int k, int n, void* stream) {
    if (m < 1 || k <= 0 || n <= 0 || !xh || !w || !c) return (int)cudaErrorInvalidValue;
    cublasHandle_t h = cublas_handle();
    if (!h) return (int)cudaErrorUnknown;
    if (cublasSetStream(h, gb_stream(stream)) != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorUnknown;
    // row-major c = xh @ w is, column-major, c^T [n, m] = w^T [n, k] @ xh^T [k, m]
    const float one = 1.f, zero = 0.f;
    const cublasStatus_t s = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one, w, CUDA_R_16F, n, xh, CUDA_R_16F, k,
                                          &zero, c, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return s == CUBLAS_STATUS_SUCCESS ? 0 : (int)cudaErrorUnknown;
}

// y [m, n] = EXL3 layer applied to x [m, k] for any m: reconstruct W_q, rotate, GEMM, rotate. Row invariance is not
// guaranteed here (cublasGemmEx's pick depends on m); gb_exl3_prefill_pinned has it. bias may be null; scratch needs gb_exl3_prefill_scratch_bytes(m, k, n).
GB_EXPORT int gb_exl3_prefill(const void* x, int x_dtype, const void* trellis, const void* suh, const void* svh,
                              const void* bias, void* y, int y_dtype, int m, int k, int n, int bits, int codebook,
                              void* scratch, size_t scratch_bytes, void* stream) {
    if (m < 1 || !shape_ok(k, n, bits, codebook) || !dtype_ok(x_dtype) || !dtype_ok(y_dtype) || !x || !trellis ||
        !suh || !svh || !y || !scratch || scratch_bytes < gb_exl3_prefill_scratch_bytes(m, k, n))
        return (int)cudaErrorInvalidValue;
    char* s = static_cast<char*>(scratch);
    void* xh = s;
    void* w = s + align256((size_t)m * k * 2);
    void* c = s + align256((size_t)m * k * 2) + align256((size_t)k * n * 2);
    int rc = gb_exl3_rotate_in(x, x_dtype, suh, xh, m, k, stream);
    if (!rc) rc = gb_exl3_reconstruct(trellis, w, k, n, bits, codebook, stream);
    if (!rc) rc = gb_exl3_prefill_gemm(xh, w, c, m, k, n, stream);
    if (!rc) rc = gb_exl3_rotate_out(c, svh, bias, y, y_dtype, m, n, stream);
    return rc;
}

// ---------------------------------------------------------------------------------------------------------------
// Pinned prefill path: the same four steps with the GEMM on a fixed cublasLt algorithm (see kPinName). ROW INVARIANT for
// every m >= 1: a row's output bits do not depend on m, on the row's index in the call, or on the other rows.
// ---------------------------------------------------------------------------------------------------------------

// Human-readable identity of the pinned algorithm (a static string).
GB_EXPORT const char* gb_exl3_prefill_pin_name() { return kPinName; }

// The pinned algorithm's tile, stage count and custom option for a K x N layer (the cublasLt ids of pin_for): what the sweeps
// (tests, tools) report and assert per shape. 0, or an error for a shape the pinned path does not take.
GB_EXPORT int gb_exl3_prefill_pin_for(int k, int n, unsigned* tile, unsigned* stages, unsigned* custom) {
    if (k <= 0 || n <= 0 || !tile || !stages || !custom) return (int)cudaErrorInvalidValue;
    const Pin p = pin_for(k, n);
    *tile = p.tile;
    *stages = p.stages;
    *custom = p.custom;
    return 0;
}

// The decode path's plan of a K x N layer: K splits and warps a block. The shape's alone (never the number of rows), so a layer
// keeps one reduction order for every window width; the sweeps assert it per shape.
// Test hook: which kernel families the GLM decode path launches with the programmatic-dependent-launch attribute (pdl.cuh): bit 1 the EXL3
// chain (rot_in, the linear kernel, rot_out), bit 2 mHC, bit 4 the routed MoE chain; 7 all, 0 none (the default; REXL3_GLM_PDL sets it at
// start). The bits do not depend on it; only the speed.
GB_EXPORT int gb_pdl_set_mask(int mask) {
    gbp_pdl::g_mask = mask & gbp_pdl::kAll;
    return 0;
}

// Test hook: which launches the decode path folds into the linear kernel (env REXL3_EXL3_FUSE sets it at start): bit 1 the input rotation of
// 1 to 8 rows. The bits do not depend on it; only the launch count and the speed.
GB_EXPORT int gb_exl3_set_fuse(int mask) {
    g_fuse = mask & kFuseAll;
    return 0;
}

// Whether gb_exl3_linear on `m` rows of a K x N layer of `bits` bits rotates its own input now (the fuse mask is on, 1 to kFuseRows rows, and
// the rotated rows fit beside the ring): how a test knows it exercises the fused kernel and not the launch it falls back to.
GB_EXPORT int gb_exl3_fused_in(int m, int k, int n, int bits) {
    if (m < 1 || m > kMaxRows || !shape_ok(k, n, bits, 0) || group_tiles(m) != 1) return 0;
    LinArgs a{};
    a.M = m;
    a.K = k;
    a.N = n;
    a.plan = plan_for(k, n);
    a.fuse = fuse_mask();
    const int rh = m < 8 ? m : 8;
    switch (bits) {
        case 1: return ring_plan_of<2>(a, rh).fused;
        case 2: return ring_plan_of<4>(a, rh).fused;
        case 3: return ring_plan_of<6>(a, rh).fused;
        case 4: return ring_plan_of<8>(a, rh).fused;
        case 5: return ring_plan_of<10>(a, rh).fused;
        case 6: return ring_plan_of<12>(a, rh).fused;
        case 7: return ring_plan_of<14>(a, rh).fused;
        default: return ring_plan_of<16>(a, rh).fused;
    }
}

GB_EXPORT int gb_exl3_plan(int k, int n, int* sk, int* wk) {
    if (!shape_ok(k, n, 1, 0) || !sk || !wk) return (int)cudaErrorInvalidValue;
    const Plan p = plan_for(k, n);
    *sk = p.sk;
    *wk = p.wk;
    return 0;
}

// 0 when the pinned algorithm runs a [m, k] x [k, n] problem on this device and cuBLAS with no workspace; an error otherwise.
// Start-up calls it for every prefill shape so a cuBLAS that lacks the algorithm is a boot error, not a bit change.
GB_EXPORT int gb_exl3_prefill_pin_check(int m, int k, int n) {
    if (m < 1 || k <= 0 || n <= 0) return (int)cudaErrorInvalidValue;
    cublasLtHandle_t lt = lt_handle();
    if (!lt) return (int)cudaErrorUnknown;
    LtProblem p;
    int rc = p.init(lt, m, k, n);
    if (rc) return rc;
    cublasLtMatmulHeuristicResult_t r;
    if (cublasLtMatmulAlgoCheck(lt, p.desc, p.a, p.b, p.c, p.c, &p.algo, &r) != CUBLAS_STATUS_SUCCESS)
        return (int)cudaErrorNotSupported;
    return r.workspaceSize == 0 ? 0 : (int)cudaErrorNotSupported;
}

// c [m, n] fp32 = xh [m, k] fp16 @ W_q [k, n] fp16 on the pinned algorithm.
GB_EXPORT int gb_exl3_prefill_gemm_pinned(const void* xh, const void* w, void* c, int m, int k, int n, void* stream) {
    if (m < 1 || k <= 0 || n <= 0 || !xh || !w || !c) return (int)cudaErrorInvalidValue;
    cublasLtHandle_t lt = lt_handle();
    if (!lt) return (int)cudaErrorUnknown;
    LtProblem p;
    int rc = p.init(lt, m, k, n);
    if (rc) return rc;
    const float one = 1.f, zero = 0.f;
    // no workspace: an algorithm that needed one would be refused here, never swapped for another
    const cublasStatus_t s = cublasLtMatmul(lt, p.desc, &one, w, p.a, xh, p.b, &zero, c, p.c, c, p.c, &p.algo, nullptr, 0,
                                            gb_stream(stream));
    return s == CUBLAS_STATUS_SUCCESS ? 0 : (int)cudaErrorNotSupported;
}

// gb_exl3_prefill with the pinned GEMM. Same scratch.
GB_EXPORT int gb_exl3_prefill_pinned(const void* x, int x_dtype, const void* trellis, const void* suh, const void* svh,
                                     const void* bias, void* y, int y_dtype, int m, int k, int n, int bits, int codebook,
                                     void* scratch, size_t scratch_bytes, void* stream) {
    if (m < 1 || !shape_ok(k, n, bits, codebook) || !dtype_ok(x_dtype) || !dtype_ok(y_dtype) || !x || !trellis ||
        !suh || !svh || !y || !scratch || scratch_bytes < gb_exl3_prefill_scratch_bytes(m, k, n))
        return (int)cudaErrorInvalidValue;
    char* s = static_cast<char*>(scratch);
    void* xh = s;
    void* w = s + align256((size_t)m * k * 2);
    void* c = s + align256((size_t)m * k * 2) + align256((size_t)k * n * 2);
    int rc = gb_exl3_rotate_in(x, x_dtype, suh, xh, m, k, stream);
    if (!rc) rc = gb_exl3_reconstruct(trellis, w, k, n, bits, codebook, stream);
    if (!rc) rc = gb_exl3_prefill_gemm_pinned(xh, w, c, m, k, n, stream);
    if (!rc) rc = gb_exl3_rotate_out(c, svh, bias, y, y_dtype, m, n, stream);
    return rc;
}

// gb_exl3_prefill_pinned with W_q from the CALLER: the same rotate_in, the same pinned GEMM, the same rotate_out, on a
// W_q [k, n] fp16 the caller keeps reconstructed (gb_exl3_reconstruct's own output, wherever it lives) instead of one
// reconstructed into the scratch this call. Row invariant for the same reason the pinned path is (one algorithm, one K
// split: see the block comment above), and bit-identical to gb_exl3_prefill_pinned on the same weights: the reconstruct
// is a pure function of the trellis, so a kept W_q holds the bits the per-call one would have held. NOT the same as
// gb_exl3_prefill + gb_exl3_prefill_gemm: that pair runs the heuristic cublasGemmEx, which is not row invariant.
// Same scratch layout as gb_exl3_prefill_pinned (its W_q section is unused; xh and c sit at the same offsets, so one
// scratch serves either path for the same m, k, n).
GB_EXPORT int gb_exl3_prefill_pinned_with_w(const void* x, int x_dtype, const void* w, const void* suh, const void* svh,
                                            const void* bias, void* y, int y_dtype, int m, int k, int n, void* scratch,
                                            size_t scratch_bytes, void* stream) {
    if (m < 1 || k <= 0 || n <= 0 || k % 128 || n % 128 || !dtype_ok(x_dtype) || !dtype_ok(y_dtype) || !x || !w ||
        !suh || !svh || !y || !scratch || scratch_bytes < gb_exl3_prefill_scratch_bytes(m, k, n))
        return (int)cudaErrorInvalidValue;
    char* s = static_cast<char*>(scratch);
    void* xh = s;
    void* c = s + align256((size_t)m * k * 2) + align256((size_t)k * n * 2);
    int rc = gb_exl3_rotate_in(x, x_dtype, suh, xh, m, k, stream);
    if (!rc) rc = gb_exl3_prefill_gemm_pinned(xh, w, c, m, k, n, stream);
    if (!rc) rc = gb_exl3_rotate_out(c, svh, bias, y, y_dtype, m, n, stream);
    return rc;
}
