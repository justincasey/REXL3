// gb-nvcc: --expt-relaxed-constexpr --fmad=false
// GLM-5.3-Flash prefill under contract P2 (Lane P step 3; crates/model/src/families/glm/contract.rs, docs/GLM_PREFILL_P2.md section 6):
// the ROUTED EXPERTS of a prompt chunk on tensor cores, any chunk width. The Rust side is crates/cuda/src/glm_p2_moe.rs.
//
// The arithmetic of a (row, expert) pair; the rounding points are the decode path's (csrc/moe_prefill.cu), only the K order differs:
//   input      A_g, A_u [sorted pair][K] fp16 = fp16((x[row] * suh[e]) H_128 / sqrt 128), the fp32 values of P1's rot_in, row-major;
//   gate / up  per output: the K sum in a fixed function of K (FOLD: per 128-k block b = 0, 1, .. a chain of its 8 m16n8k16 MMAs from
//              +0.0, then tot = tot + part_b, IEEE round to nearest; !FOLD: one chain over every k tile from +0.0), fp16 x fp16 -> fp32;
//              then H_128 / sqrt 128 * svh, silu(gate) clamped from above and up clamped to +-limit (limit > 0), the product, * suh_down,
//              H_128 / sqrt 128, fp16: the down projection's A row of the pair (P1's gateup epilogue, expression for expression);
//   down       the K sum in the same order, H_128 / sqrt 128 * svh_down: the pair's fp32 y (P1's down epilogue);
//   combine    per row, the slots split into two GROUPS by expert id (e < gsplit: group 0, else group 1; gsplit = the TP=2 ownership's
//              cut, 144 of 288), each group an fmaf chain in slot order from +0.0, then (part_0 + part_1) + shared (fp32 adds).
// A row meets other rows only inside an mma.sync (rows independent), padding rows are zero-filled and never stored, the K order depends
// on K alone, and nothing is atomic, so a pair's bits depend on its row and its expert's weights only: not on the chunk, its width, the
// grouping of pairs into items, their order or which block computes them (P2 (a), (b), (d); gated in crates/cuda/tests/glm_p2_moe.rs).
// The weights are the trellis decoded by the decode path's own decoder (exl3_linear_decode.cuh), fp32 accumulation (P2 (c)).
//
// Schedule. The pairs are sorted by expert with P1's plan (csrc/moe_prefill.cu gb_moe_pf_plan: each expert's pairs padded to 16-row tiles,
// the same sorted rows), cut into work items of 1 to 4 tiles of one expert and listed IN EXPERT ORDER (an expert's full 4-tile items, then
// its remainder of 1-3 tiles), all of them in ONE launch per GEMM: an expert's items are neighbours in the grid, so the blocks that read
// its weights run together and all but the first find them in L2 (P1's three item classes, each its own launch, read the weights of an
// expert with a remainder twice: MEASURED, a quarter of the routed GEMMs' time at 4,096 rows). Gate and up run in ONE block per (128
// output columns, item): 8 warps, warps 0-3 the gate matrix and 4-7 the up matrix, warp w owning 32 columns (two 16-column tiles) of its
// matrix and every row of the item; a 3-stage cp.async ring of 64 k streams the two matrices' A rows (16-byte chunks XOR-swizzled by row
// for conflict-free ldmatrix) and raw trellis tiles; each warp decodes its own tiles once per k tile into B fragments and multiplies all
// of the item's m16 tiles by them. The sums meet in shared memory for the SwiGLU epilogue. The down GEMM is the same loop on one matrix
// (warp w: 16 columns). The structure follows
// the dense P2 kernel (glm_p2_dense.cu) and REXL3's Nemotron routed GEMM (nemotron_moe_mma.cu, lane S2); the trellis tile decode is
// exl3_linear_decode.cuh (TensorFold, MIT, in the tree); the ideas of streaming raw trellis tiles through cp.async and decoding them per
// warp are upstream ExLlamaV3's (MIT, Copyright (c) 2025 Turboderp; see THIRD_PARTY_NOTICES.md). No code was copied from either.

#include <math.h>
#include <stdint.h>

#include <type_traits>

#include "common.cuh"
#include "exl3_linear_decode.cuh"

using namespace gbx;

namespace {

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

constexpr int kThreads = 256;  // 8 warps (the plan's and the epilogue's unit; a GEMM block has NW warps)
constexpr int kBK = 64;        // k of one pipeline stage (4 k tiles); a 128-k fold block is two stages
constexpr int kStages = 3;
constexpr int kMaxTopK = 16;
constexpr int kLds = 128 + 4;  // floats a row of the epilogue staging

// ---- primitives ---------------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&a)[4], const void* smem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}

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
    const half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
    uint2 u;
    u.x = *reinterpret_cast<const uint32_t*>(&a);
    u.y = *reinterpret_cast<const uint32_t*>(&b);
    *reinterpret_cast<uint2*>(p) = u;
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

// The half offset of 16-byte chunk c (0..7) of row r in an A stage (rows of 64 values): chunk c of row r lands at c ^ (r & 7), so the 8
// rows an ldmatrix phase reads hit 8 distinct chunks.
__device__ __forceinline__ int a_off(int r, int c) { return r * kBK + ((c ^ (r & 7)) << 3); }

// ---- the plan -----------------------------------------------------------------------------------------------------------------------
//
// The pairs sorted by expert, each expert's pairs padded to whole 16-row tiles (EXACTLY P1's sorted rows: csrc/moe_prefill.cu plan_kernel's
// spair, pairs in pair order inside an expert), and ONE list of work items in expert order: per expert its full items of 4 tiles (64
// rows) in tile order, then the remainder item of 1-3 tiles; the items are P1's (the same expert, first tile, tiles and valid rows),
// listed per expert instead of by class. Computed by many blocks instead of one (P1's single block scans every pair once per expert: 2.8
// ms a layer at 4,096 rows, MEASURED). Three launches: per chunk of kPlanChunk pairs, each expert's count (one thread an expert); one block
// turns the counts into each chunk's first position per expert, the tile offsets and the items; per chunk, each expert's pairs placed in
// order. Deterministic, no atomics; the order is a function of the picks alone.

constexpr int kPlanThreads = 1024;  // one expert a thread: E <= 1024
constexpr int kPlanChunk = 2048;    // pairs a block

// Exclusive prefix sum of one int a thread over the whole block (1024 threads); *total receives the sum. ws: 33 ints of shared memory.
// (P1's block_excl_scan, csrc/moe_prefill.cu.)
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
    __syncthreads();
    return r;
}

// Block b: cnt[b * E + e] = the pairs of chunk b (pairs b * kPlanChunk ..) that pick expert e. A pick outside [0, E) is no expert's.
__global__ void __launch_bounds__(kPlanThreads) plan_count(const int* __restrict__ pick, int* __restrict__ cnt, int P, int E) {
    __shared__ int sp[kPlanChunk];
    const int b = blockIdx.x, p0 = b * kPlanChunk, n = min(kPlanChunk, P - p0);
    for (int i = threadIdx.x; i < kPlanChunk; i += kPlanThreads) sp[i] = i < n ? pick[p0 + i] : -1;
    __syncthreads();
    const int e = threadIdx.x;
    if (e >= E) return;
    int c = 0;
    for (int i = 0; i < n; ++i) c += sp[i] == e;
    cnt[b * E + e] = c;
}

// One block: per expert, the chunks' counts become each chunk's first rank (in place), the expert's tile offset `base` (in rows), the padding
// rows' spair -1, the items in expert order (items [cap]) and nitems: {items, 0, 0, sorted rows}.
__global__ void __launch_bounds__(kPlanThreads) plan_scan(int* __restrict__ cnt, int* __restrict__ base, int* __restrict__ spair,
                                                          int4* __restrict__ items, int* __restrict__ nitems, int nb, int E, int cap) {
    __shared__ int ws[33];
    const int tid = threadIdx.x;
    int c = 0;
    if (tid < E) {
        for (int b = 0; b < nb; ++b) {
            const int v = cnt[b * E + tid];
            cnt[b * E + tid] = c;
            c += v;
        }
    }
    const int tiles = (c + 15) >> 4;
    int total_tiles;
    const int toff = block_excl_scan(tiles, &total_tiles, ws);
    if (tid < E) {
        base[tid] = toff * 16;
        for (int j = c; j < tiles * 16; ++j) spair[toff * 16 + j] = -1;
    }
    const int full = tiles >> 2, rem = tiles & 3;
    int n;
    const int o = block_excl_scan(full + (rem != 0), &n, ws);
    if (tid < E && tiles > 0) {
        for (int gi = 0; gi < full; ++gi) items[o + gi] = make_int4(tid, toff + 4 * gi, 4, min(64, c - 64 * gi));
        if (rem) items[o + full] = make_int4(tid, toff + 4 * full, rem, c - 64 * full);
    }
    if (tid == 0) {
        nitems[0] = n;
        nitems[1] = 0;
        nitems[2] = 0;
        nitems[3] = total_tiles * 16;
    }
}

// Block b: each expert's pairs of chunk b, in pair order, at base[e] + the chunk's first rank.
__global__ void __launch_bounds__(kPlanThreads) plan_place(const int* __restrict__ pick, const int* __restrict__ cnt,
                                                           const int* __restrict__ base, int* __restrict__ spair, int P, int E) {
    __shared__ int sp[kPlanChunk];
    const int b = blockIdx.x, p0 = b * kPlanChunk, n = min(kPlanChunk, P - p0);
    for (int i = threadIdx.x; i < kPlanChunk; i += kPlanThreads) sp[i] = i < n ? pick[p0 + i] : -1;
    __syncthreads();
    const int e = threadIdx.x;
    if (e >= E) return;
    int j = base[e] + cnt[b * E + e];
    for (int i = 0; i < n; ++i)
        if (sp[i] == e) spair[j++] = p0 + i;
}

// ---- input rotation -----------------------------------------------------------------------------------------------------------------

// Sorted row s of matrix z (0 gate, 1 up): A_z[s] = fp16((x[row of pair spair[s]] * suh_z[e]) H_128 / sqrt 128), row-major [S][K]. One block
// a sorted row, both matrices: warp w takes the 128-wide blocks w, w + 8, ..., lane l the values 4 l .. 4 l + 3 of a block (each block's
// arithmetic is a warp's on its lanes, as it has always been), the x values loaded once for the two matrices. Padding rows (spair -1) and
// rows past the plan's count are not written (the GEMMs zero-fill them).
constexpr int kRotThreads = 256;
__global__ void __launch_bounds__(kRotThreads) rot_kernel(const void* __restrict__ x, int x_dtype, const int* __restrict__ pick,
                                                         const int* __restrict__ spair, const int* __restrict__ nitems,
                                                         const half* __restrict__ suh0, const half* __restrict__ suh1,
                                                         half* __restrict__ a0, half* __restrict__ a1, int K, int topk) {
    const int srow = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (srow >= nitems[3]) return;
    const int p = spair[srow];
    if (p < 0) return;
    const int e = pick[p];
    const int row = p / topk;
    for (int blk = warp; blk * 128 < K; blk += kRotThreads / 32) {
        const int k = blk * 128 + 4 * lane;
        float v[4];
        load4(x, x_dtype, (size_t)row * K + k, v);
#pragma unroll
        for (int mat = 0; mat < 2; ++mat) {
            float s[4], w[4];
            load4h((mat ? suh1 : suh0) + (size_t)e * K + k, s);
#pragma unroll
            for (int j = 0; j < 4; ++j) w[j] = v[j] * s[j];
            fwht128(w, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) w[j] *= HAD_SCALE;
            store4h((mat ? a1 : a0) + (size_t)srow * K + k, w);
        }
    }
}

// ---- the grouped GEMM -----------------------------------------------------------------------------------------------------------------

// NW warps a block, NW / MATS a matrix, each owning TPW 16-column tiles of the block's 128 columns: MATS = 2 (gate/up) at NW = 8 has warps
// 0-3 on gate and 4-7 on up with 32 columns a warp, at NW = 16 warps 0-7 and 8-15 with 16 columns; MATS = 1 (down) at NW = 8 has 16
// columns a warp. The shape decides only which warp computes an output, never its arithmetic (the same MMAs on the same decoded values).
template <int MATS, int NW>
struct Shape {
    static constexpr int WPM = NW / MATS;              // warps a matrix
    static constexpr int TPW = 8 / WPM;                // 16-column tiles a warp
    static_assert(WPM * TPW == 8, "a matrix's warps cover its 8 column tiles");
};

template <int K2, int MT, int MATS>
__host__ __device__ constexpr int stage_bytes() {
    return MATS * (16 * MT * kBK * 2 + (kBK / 16) * 8 * tile_words<K2>() * 4);
}

template <int K2, int MT, int MATS, int ST>
__host__ __device__ constexpr int smem_bytes() {
    constexpr int ring = ST * stage_bytes<K2, MT, MATS>();
    constexpr int epi = MATS * 16 * MT * kLds * 4;
    return ring > epi ? ring : epi;
}

struct GemmArgs {
    const half* a0;          // A rows of matrix 0 [S][K] fp16 (sorted rows)
    const half* a1;          // matrix 1 (MATS = 2)
    const uint64_t* t0;      // per-expert trellis device addresses of matrix 0 [E]
    const uint64_t* t1;      // matrix 1
    const int4* items;       // {expert, first tile, tiles, valid rows}: the plan's items in expert order
    const int* nitems;       // their count
    const half* svh0;        // [E][N] output scales of matrix 0
    const half* svh1;        // matrix 1
    const half* suh_next;    // [E][N] the down projection's input scales (MATS = 2)
    half* out16;             // MATS = 2: the down projection's A rows [S][N] fp16
    float* y;                // MATS = 1: y [P][N] fp32 at the pair of each sorted row
    const int* spair;
    int K, N;
    float limit;
};

// Block (x = 128-column block, y = item). See the file comment for the arithmetic; MT is an item's most m16 tiles, the item's own count
// `tiles` (<= MT) is a compile-time constant of the loop (MA) so no MMA is predicated.
template <int K2, int MT, int MATS, bool FOLD, int NW, int ST>
__global__ void __launch_bounds__(NW * 32, MATS == 1 ? 2 : 1) gemm_kernel(const GemmArgs g) {
    using S = Shape<MATS, NW>;
    constexpr int TPW = S::TPW, WPM = S::WPM;
    constexpr int kThreads = NW * 32;
    constexpr int kStages = ST;
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    constexpr int A_BYTES = 16 * MT * kBK * 2;          // one matrix's A stage
    constexpr int B_WORDS = (kBK / 16) * 8 * TW;        // one matrix's B stage: 4 k tiles x 8 column tiles
    constexpr int MSTAGE = A_BYTES + B_WORDS * 4;       // one matrix's share of a stage
    constexpr int STAGE = MATS * MSTAGE;
    extern __shared__ __align__(128) unsigned char smem[];

    if ((int)blockIdx.y >= *g.nitems) return;  // uniform per block
    const int4 item = g.items[blockIdx.y];
    const int e = item.x, row0 = item.y * 16, tiles = item.z, vrows = item.w;
    const int nb = blockIdx.x;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int mat = warp / WPM, wm = warp % WPM;        // this warp's matrix and its index among the matrix's warps
    const int K = g.K, N = g.N;
    const int KC = K / kBK;
    const int tiles_n = N / 16;
    const uint32_t* T0 = reinterpret_cast<const uint32_t*>(g.t0[e]) + (size_t)nb * 8 * TW;
    const uint32_t* T1 = MATS == 2 ? reinterpret_cast<const uint32_t*>(g.t1[e]) + (size_t)nb * 8 * TW : T0;

    auto stage_a = [&](int s, int m) { return reinterpret_cast<half*>(smem + s * STAGE + m * MSTAGE); };
    auto stage_b = [&](int s, int m) { return reinterpret_cast<uint32_t*>(smem + s * STAGE + m * MSTAGE + A_BYTES); };

    // the stage's copies: A rows (16 MT rows x 8 chunks a matrix; rows past `vrows` zero-filled) and the trellis tiles (4 k tiles x 8 column
    // tiles x TW words a matrix, the 8 tiles of a k tile contiguous in memory)
    auto load_stage = [&](int s, int kc) {
        constexpr int ACH = 16 * MT * 8;                  // A chunks a matrix
#pragma unroll
        for (int i = 0; i < (MATS * ACH + kThreads - 1) / kThreads; ++i) {
            const int idx = i * kThreads + t;
            if (idx < MATS * ACH) {
                const int m = idx / ACH, rem = idx % ACH, r = rem >> 3, c = rem & 7;
                const bool ok = r < vrows;
                const half* src = (m ? g.a1 : g.a0) + (size_t)(row0 + (ok ? r : 0)) * K + kc * kBK + c * 8;
                cp_async16(stage_a(s, m) + a_off(r, c), src, ok);
            }
        }
        constexpr int BCH = B_WORDS / 4;                  // 16-byte chunks a matrix
        constexpr int BROW = 8 * TW / 4;                  // chunks a k tile
#pragma unroll
        for (int i = 0; i < (MATS * BCH + kThreads - 1) / kThreads; ++i) {
            const int idx = i * kThreads + t;
            if (idx < MATS * BCH) {
                const int m = idx / BCH, rem = idx % BCH, kl = rem / BROW, w = rem % BROW;
                const uint32_t* src = (m ? T1 : T0) + (size_t)(kc * (kBK / 16) + kl) * tiles_n * TW + w * 4;
                cp_async16(stage_b(s, m) + kl * 8 * TW + w * 4, src, true);
            }
        }
    };

    float acc[MT][TPW][2][4];
    float tot[FOLD ? MT : 1][FOLD ? TPW : 1][2][4];
#pragma unroll
    for (int p = 0; p < MT; ++p)
#pragma unroll
        for (int j = 0; j < TPW; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    acc[p][j][h][c] = 0.f;
                    if constexpr (FOLD) tot[p][j][h][c] = 0.f;
                }

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < KC) load_stage(s, s);
        cp_async_commit();
    }

    const int lr = (lane & 7) + 8 * ((lane >> 3) & 1), lc = lane >> 4;

    auto kloop = [&](auto ma_c) {
        constexpr int MA = decltype(ma_c)::value;
#pragma unroll 1
        for (int kc = 0; kc < KC; ++kc) {
            cp_async_wait<kStages - 2>();
            __syncthreads();  // stage kc has landed for every thread; nobody reads the slot the next load overwrites
            {
                const int nk = kc + kStages - 1;
                if (nk < KC) load_stage(nk % kStages, nk);
                cp_async_commit();
            }
            const half* sa = stage_a(kc % kStages, mat);
            const uint32_t* sb = stage_b(kc % kStages, mat);
#pragma unroll
            for (int kk = 0; kk < kBK / 16; ++kk) {
                uint32_t b[TPW][2][2];
#pragma unroll
                for (int j = 0; j < TPW; ++j) {
                    uint32_t w[LW];
                    load_lane_words<K2>(sb + (kk * 8 + wm * TPW + j) * TW, lane, w);
                    decode_lane<K2, CB_MUL1>(w, lane, b[j][0], b[j][1]);
                }
#pragma unroll
                for (int p = 0; p < MA; ++p) {
                    uint32_t a[4];
                    ldsm_x4(a, sa + a_off(p * 16 + lr, kk * 2 + lc));
#pragma unroll
                    for (int j = 0; j < TPW; ++j) {
                        mma16816(acc[p][j][0], a, b[j][0]);
                        mma16816(acc[p][j][1], a, b[j][1]);
                    }
                }
            }
            if constexpr (FOLD) {
                if (kc & 1) {  // the end of 128-k block kc / 2: its partials join the running sums in block order
#pragma unroll
                    for (int p = 0; p < MA; ++p)
#pragma unroll
                        for (int j = 0; j < TPW; ++j)
#pragma unroll
                            for (int h = 0; h < 2; ++h)
#pragma unroll
                                for (int c = 0; c < 4; ++c) {
                                    tot[p][j][h][c] = __fadd_rn(tot[p][j][h][c], acc[p][j][h][c]);
                                    acc[p][j][h][c] = 0.f;
                                }
                }
            }
        }
    };
    switch (tiles) {
        case 1: kloop(std::integral_constant<int, 1>{}); break;
        case 2: if constexpr (MT >= 2) kloop(std::integral_constant<int, MT >= 2 ? 2 : 1>{}); break;
        case 3: if constexpr (MT >= 3) kloop(std::integral_constant<int, MT >= 3 ? 3 : 1>{}); break;
        default: kloop(std::integral_constant<int, MT>{}); break;
    }
    cp_async_wait<0>();
    __syncthreads();  // the ring's bytes become the epilogue staging

    // the sums of the item's rows through shared memory: matrix m's [16 MT][kLds] f32 at E + m * 16 MT * kLds
    float* E = reinterpret_cast<float*>(smem);
    {
        const int g8 = lane >> 2, t4 = lane & 3;
        float* Em = E + mat * 16 * MT * kLds;
#pragma unroll
        for (int p = 0; p < MT; ++p) {
            if (p < tiles) {
#pragma unroll
                for (int j = 0; j < TPW; ++j)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = (wm * TPW + j) * 16 + h * 8 + 2 * t4;
                        const int r = p * 16 + g8;
                        float v[4];
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            if constexpr (FOLD) v[c] = tot[p][j][h][c];
                            else v[c] = acc[p][j][h][c];
                        }
                        *reinterpret_cast<float2*>(Em + r * kLds + col) = make_float2(v[0], v[1]);
                        *reinterpret_cast<float2*>(Em + (r + 8) * kLds + col) = make_float2(v[2], v[3]);
                    }
            }
        }
    }
    __syncthreads();

    const int n = nb * 128 + 4 * lane;
    if constexpr (MATS == 2) {
        // P1's gateup epilogue (csrc/moe_prefill.cu gateup_kernel), expression for expression, on these sums
        float sg4[4], su4[4], sd4[4];
        load4h(g.svh0 + (size_t)e * N + n, sg4);
        load4h(g.svh1 + (size_t)e * N + n, su4);
        load4h(g.suh_next + (size_t)e * N + n, sd4);
        const float* Eg = E;
        const float* Eu = E + 16 * MT * kLds;
        const float limit = g.limit;
        for (int r = warp; r < vrows; r += NW) {
            const float4 g4 = *reinterpret_cast<const float4*>(Eg + r * kLds + 4 * lane);
            const float4 u4 = *reinterpret_cast<const float4*>(Eu + r * kLds + 4 * lane);
            float gv[4] = {g4.x, g4.y, g4.z, g4.w}, uv[4] = {u4.x, u4.y, u4.z, u4.w}, v[4];
            fwht128(gv, lane);
            fwht128(uv, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float gg = gv[j] * HAD_SCALE * sg4[j];
                float u = uv[j] * HAD_SCALE * su4[j];
                float xx = silu_f(gg);
                if (limit > 0.0f) {
                    xx = fminf(xx, limit);
                    u = fminf(fmaxf(u, -limit), limit);
                }
                v[j] = (xx * u) * sd4[j];
            }
            fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
            store4h(g.out16 + (size_t)(row0 + r) * N + n, v);
        }
    } else {
        // P1's down epilogue: H, * svh_down, the pair's fp32 y
        float sv[4];
        load4h(g.svh0 + (size_t)e * N + n, sv);
        for (int r = warp; r < vrows; r += NW) {
            const float4 u = *reinterpret_cast<const float4*>(E + r * kLds + 4 * lane);
            float v[4] = {u.x, u.y, u.z, u.w};
            fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = v[j] * HAD_SCALE * sv[j];
            *reinterpret_cast<float4*>(g.y + (size_t)g.spair[row0 + r] * N + n) = make_float4(v[0], v[1], v[2], v[3]);
        }
    }
}

// ---- combine ------------------------------------------------------------------------------------------------------------------------

// One warp a (row, 128 columns): part_g = fmaf chain over the slots of group g (e < gsplit: 0, else 1) in slot order from +0.0; then
// out = (part_0 + part_1) + shared. MODE 0: that sum into out [rows, D]; MODE 1: part_{only} alone into out (a TP rank's own group);
// MODE 2: out = (part_0 + part_1) + shared with part_0 / part_1 read from the two halves of `parts` (the exchanged rows), y unused.
template <int MODE>
__global__ void __launch_bounds__(32) combine_kernel(const float* __restrict__ y, const int* __restrict__ pick, const half* __restrict__ wts,
                                                     const float* __restrict__ shared, const float* __restrict__ part0,
                                                     const float* __restrict__ part1, float* __restrict__ out, int D, int E, int topk,
                                                     int gsplit, int only) {
    const int r = blockIdx.x, lane = threadIdx.x;
    const int n = blockIdx.y * 128 + 4 * lane;
    float p0[4] = {0.f, 0.f, 0.f, 0.f}, p1[4] = {0.f, 0.f, 0.f, 0.f};
    if constexpr (MODE == 2) {
        const float4 a = *reinterpret_cast<const float4*>(part0 + (size_t)r * D + n);
        const float4 b = *reinterpret_cast<const float4*>(part1 + (size_t)r * D + n);
        p0[0] = a.x; p0[1] = a.y; p0[2] = a.z; p0[3] = a.w;
        p1[0] = b.x; p1[1] = b.y; p1[2] = b.z; p1[3] = b.w;
    } else {
        for (int k = 0; k < topk; ++k) {
            const int e = pick[(size_t)r * topk + k];
            if (e < 0 || e >= E) continue;
            const int grp = e >= gsplit;
            if (MODE == 1 && grp != only) continue;
            const float w = __half2float(wts[(size_t)r * topk + k]);
            const float4 u = *reinterpret_cast<const float4*>(y + ((size_t)r * topk + k) * D + n);
            float* p = grp ? p1 : p0;
            p[0] = fmaf(w, u.x, p[0]);
            p[1] = fmaf(w, u.y, p[1]);
            p[2] = fmaf(w, u.z, p[2]);
            p[3] = fmaf(w, u.w, p[3]);
        }
    }
    float o[4];
    if constexpr (MODE == 1) {
        const float* p = only ? p1 : p0;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = p[j];
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = __fadd_rn(p0[j], p1[j]);
        if (shared) {
            const float4 sh = *reinterpret_cast<const float4*>(shared + (size_t)r * D + n);
            o[0] = __fadd_rn(o[0], sh.x);
            o[1] = __fadd_rn(o[1], sh.y);
            o[2] = __fadd_rn(o[2], sh.z);
            o[3] = __fadd_rn(o[3], sh.w);
        }
    }
    *reinterpret_cast<float4*>(out + (size_t)r * D + n) = make_float4(o[0], o[1], o[2], o[3]);
}

// ---- launch plumbing ----------------------------------------------------------------------------------------------------------------

template <int K2, int MT, int MATS, bool FOLD, int NW, int ST>
int launch(const GemmArgs& a, int cap, cudaStream_t st) {
    constexpr int sm = smem_bytes<K2, MT, MATS, ST>();
    static_assert(sm <= 101376, "a block's shared memory must fit the 99 KB an SM gives one");
    static bool attr = false;  // idempotent; a race between host threads sets the same value twice
    if (!attr) {
        const cudaError_t e =
            cudaFuncSetAttribute(gemm_kernel<K2, MT, MATS, FOLD, NW, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
        if (e != cudaSuccess) return (int)e;
        attr = true;
    }
    const dim3 grid((unsigned)(a.N / 128), (unsigned)cap);
    gemm_kernel<K2, MT, MATS, FOLD, NW, ST><<<grid, NW * 32, sm, st>>>(a);
    return GB_CHECK_LAUNCH();
}

// Every item of the plan (1 to 4 tiles) in one launch at MT = 4, in the plan's expert order.
template <int K2, int MATS, bool FOLD, int NW, int ST>
int launch_items(const GemmArgs& a, int cap, cudaStream_t st) {
    return launch<K2, 4, MATS, FOLD, NW, ST>(a, cap, st);
}

// The block shapes (`shape`, never a bit): 0 the default (= 1); gate/up 1 = 8 warps (32 columns a warp) and 3 stages, 2 = 16 warps (16
// columns a warp) and 3 stages, 3 = 16 warps and 4 stages; down 1 = 8 warps and 3 stages, 2 = 8 warps and 4 stages.
template <int MATS, int K2, bool FOLD>
int launch_shape(int shape, const GemmArgs& a, int cap, cudaStream_t st) {
    if constexpr (MATS == 2) {
        switch (shape) {
            case 0:  // MEASURED on node 4 (tests/glm_p2_moe.rs bench): 8 warps at least as fast as 16 at 512 to 4,096 rows, 2 and 3 bits
            case 1: return launch_items<K2, 2, FOLD, 8, 3>(a, cap, st);
            case 2: return launch_items<K2, 2, FOLD, 16, 3>(a, cap, st);
            case 3:
                if constexpr (smem_bytes<K2, 4, 2, 4>() <= 101376) return launch_items<K2, 2, FOLD, 16, 4>(a, cap, st);
                else return (int)cudaErrorInvalidValue;
            default: return (int)cudaErrorInvalidValue;
        }
    } else {
        switch (shape) {
            case 0:
            case 1: return launch_items<K2, 1, FOLD, 8, 3>(a, cap, st);
            case 2: return launch_items<K2, 1, FOLD, 8, 4>(a, cap, st);
            default: return (int)cudaErrorInvalidValue;
        }
    }
}

template <int MATS>
int dispatch(int bits, int fold, int shape, const GemmArgs& a, int cap, cudaStream_t st) {
#define GP2M(B_, K2_) \
    case B_: return fold ? launch_shape<MATS, K2_, true>(shape, a, cap, st) : launch_shape<MATS, K2_, false>(shape, a, cap, st);
    switch (bits) {
        GP2M(2, 4)
        GP2M(3, 6)
        GP2M(4, 8)
        GP2M(5, 10)
        GP2M(6, 12)
        default: return (int)cudaErrorInvalidValue;
    }
#undef GP2M
}

bool dtype_ok(int d) { return d >= 0 && d <= 2; }

}  // namespace

// Whether the P2 expert GEMMs take a layer: hidden and expert widths multiples of 128, the Mul1 codebook at 2 to 6 bits (GLM's packs: 2
// and 3 bits on the 2.25 bpw pack, 4 on its NextN layer, 4 and 5 on sc_4.10bpw, 4, 5 and 6 on RED-SNOW 4.91bpw). At 6 bits (K2 = 12) the
// gate/up block of the default shape holds 3 stages of 2 x (8,192 + 6,144) bytes = 86,016 (the 4-stage shape 3 does not fit and is
// refused by its guard); the down block 3 x 14,336 or 4 x 14,336 bytes.
GB_EXPORT int gb_glm_p2m_supported(int hidden, int inter, int bits_gu, int bits_d, int codebook) {
    const bool shape = hidden > 0 && inter > 0 && hidden % 128 == 0 && inter % 128 == 0;
    const bool b = bits_gu >= 2 && bits_gu <= 6 && bits_d >= 2 && bits_d <= 6;
    return shape && b && codebook == CB_MUL1;
}

// Bytes of the plan's workspace for `rows` rows of `topk` picks over E experts (the chunks' counts and the experts' offsets).
GB_EXPORT long long gb_glm_p2m_plan_ws_bytes(int rows, int topk, int E) {
    const long long P = (long long)rows * topk, nb = (P + kPlanChunk - 1) / kPlanChunk;
    return (nb + 1) * E * 4;
}

// The plan (the file's plan section): spair [padded rows] (P1's gb_moe_pf_plan's bytes), items [cap] int4 in expert order (the buffer has
// P1's 3 * cap), nitems [4] = {items, 0, 0, sorted rows}, in three launches over many blocks.
// ws: gb_glm_p2m_plan_ws_bytes bytes of device memory whose contents never matter (the caller may lend any buffer free until the plan ends).
GB_EXPORT int gb_glm_p2m_plan(const void* pick, void* spair, void* items, void* nitems, void* ws, long long ws_bytes, int rows, int topk,
                              int E, int cap, void* stream) {
    const long long P = (long long)rows * topk;
    if (rows < 1 || topk < 1 || topk > kMaxTopK || E < 1 || E > kPlanThreads || cap < 1 || !pick || !spair || !items || !nitems || !ws ||
        ws_bytes < gb_glm_p2m_plan_ws_bytes(rows, topk, E) || P > (1LL << 30))
        return (int)cudaErrorInvalidValue;
    const long long e = E < P ? E : P;
    if ((long long)cap < (P + 15 * e) / 16 / 4 + e + 1) return (int)cudaErrorInvalidValue;  // gb_moe_pf_item_cap
    const int nb = (int)((P + kPlanChunk - 1) / kPlanChunk);
    int* cnt = static_cast<int*>(ws);
    int* base = cnt + (size_t)nb * E;
    cudaStream_t st = gb_stream(stream);
    plan_count<<<nb, kPlanThreads, 0, st>>>(static_cast<const int*>(pick), cnt, (int)P, E);
    plan_scan<<<1, kPlanThreads, 0, st>>>(cnt, base, static_cast<int*>(spair), static_cast<int4*>(items), static_cast<int*>(nitems), nb, E,
                                          cap);
    plan_place<<<nb, kPlanThreads, 0, st>>>(static_cast<const int*>(pick), cnt, base, static_cast<int*>(spair), (int)P, E);
    return GB_CHECK_LAUNCH();
}

// A_g / A_u [padded rows][K] fp16, row-major (the plan's sorted rows; P1's rot_in values): see rot_kernel.
GB_EXPORT int gb_glm_p2m_rotate(const void* x, int x_dtype, const void* pick, const void* spair, const void* nitems, const void* suh0,
                                const void* suh1, void* a0, void* a1, int rows, int topk, int K, int E, void* stream) {
    if (rows < 1 || topk < 1 || topk > kMaxTopK || K <= 0 || K % 128 || E < 1 || !dtype_ok(x_dtype) || !x || !pick || !spair || !nitems ||
        !suh0 || !suh1 || !a0 || !a1)
        return (int)cudaErrorInvalidValue;
    const long long P = (long long)rows * topk, padded = (P + 15LL * (E < P ? E : P) + 15) / 16 * 16;
    rot_kernel<<<(unsigned)padded, kRotThreads, 0, gb_stream(stream)>>>(
        x, x_dtype, static_cast<const int*>(pick), static_cast<const int*>(spair), static_cast<const int*>(nitems),
        static_cast<const half*>(suh0), static_cast<const half*>(suh1), static_cast<half*>(a0), static_cast<half*>(a1), K, topk);
    return GB_CHECK_LAUNCH();
}

// Gate and up with the SwiGLU epilogue: ad [padded rows][N] fp16 (the down projection's A rows). K = the model width, N = the expert width.
// items / nitems: the plan's (one launch over `cap` item slots). fold: the K order (1 fold128, 0 chain). limit <= 0: no clamp.
GB_EXPORT int gb_glm_p2m_gateup(const void* a0, const void* a1, const void* tg, const void* tu, const void* items, const void* nitems,
                                const void* svh_g, const void* svh_u, const void* suh_d, void* ad, int K, int N, int bits, int fold,
                                int shape, int cap, float limit, void* stream) {
    if (K <= 0 || N <= 0 || K % 128 || N % 128 || cap < 1 || !a0 || !a1 || !tg || !tu || !items || !nitems || !svh_g || !svh_u || !suh_d ||
        !ad)
        return (int)cudaErrorInvalidValue;
    GemmArgs a{static_cast<const half*>(a0), static_cast<const half*>(a1), static_cast<const uint64_t*>(tg), static_cast<const uint64_t*>(tu),
               static_cast<const int4*>(items), static_cast<const int*>(nitems), static_cast<const half*>(svh_g),
               static_cast<const half*>(svh_u), static_cast<const half*>(suh_d), static_cast<half*>(ad), nullptr, nullptr, K, N, limit};
    return dispatch<2>(bits, fold, shape, a, cap, gb_stream(stream));
}

// Down with its epilogue: y [P][N] fp32 at the pair of each sorted row (spair). K = the expert width, N = the model width.
GB_EXPORT int gb_glm_p2m_down(const void* ad, const void* td, const void* items, const void* nitems, const void* svh_d, const void* spair,
                              void* y, int K, int N, int bits, int fold, int shape, int cap, void* stream) {
    if (K <= 0 || N <= 0 || K % 128 || N % 128 || cap < 1 || !ad || !td || !items || !nitems || !svh_d || !spair || !y)
        return (int)cudaErrorInvalidValue;
    GemmArgs a{static_cast<const half*>(ad), nullptr, static_cast<const uint64_t*>(td), nullptr, static_cast<const int4*>(items),
               static_cast<const int*>(nitems), static_cast<const half*>(svh_d), nullptr, nullptr, nullptr, static_cast<float*>(y),
               static_cast<const int*>(spair), K, N, 0.f};
    return dispatch<1>(bits, fold, shape, a, cap, gb_stream(stream));
}

// The P2 combine (file comment). mode 0: out = (part_0 + part_1) + shared from y; mode 1: out = part_{only} from y (a TP rank's own
// group); mode 2: out = (part0 + part1) + shared from the two given part buffers [rows, D]. shared may be null (nothing added).
GB_EXPORT int gb_glm_p2m_combine(int mode, const void* y, const void* pick, const void* wts, const void* shared, const void* part0,
                                 const void* part1, void* out, int rows, int topk, int D, int E, int gsplit, int only, void* stream) {
    if (rows < 1 || topk < 1 || topk > kMaxTopK || D <= 0 || D % 128 || E < 1 || gsplit < 0 || gsplit > E || !out ||
        (mode != 2 && (!y || !pick || !wts)) || (mode == 2 && (!part0 || !part1)) || mode < 0 || mode > 2 || only < 0 || only > 1)
        return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)rows, (unsigned)(D / 128));
    const float* yy = static_cast<const float*>(y);
    const int* pk = static_cast<const int*>(pick);
    const half* w = static_cast<const half*>(wts);
    const float* sh = static_cast<const float*>(shared);
    const float* q0 = static_cast<const float*>(part0);
    const float* q1 = static_cast<const float*>(part1);
    float* o = static_cast<float*>(out);
    cudaStream_t st = gb_stream(stream);
    if (mode == 0) combine_kernel<0><<<grid, 32, 0, st>>>(yy, pk, w, sh, q0, q1, o, D, E, topk, gsplit, only);
    else if (mode == 1) combine_kernel<1><<<grid, 32, 0, st>>>(yy, pk, w, sh, q0, q1, o, D, E, topk, gsplit, only);
    else combine_kernel<2><<<grid, 32, 0, st>>>(yy, pk, w, sh, q0, q1, o, D, E, topk, gsplit, only);
    return GB_CHECK_LAUNCH();
}
