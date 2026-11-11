// gb-nvcc: --expt-relaxed-constexpr
// GLM-5.3-Flash prefill under contract P2 (Lane P; crates/model/src/families/glm/contract.rs): the dense EXL3 linears of a prompt chunk on
// tensor cores, any chunk width. The Rust side is crates/cuda/src/glm_p2.rs.
//
// y [M, N] = ((x * suh) H_K . W_q . H_N) * svh (+ bias), H = 128-block Sylvester Hadamard / sqrt(128).
//
// GLM's own copy of the tensor-core GEMM REXL3 wrote for Nemotron's dense prefill linears (csrc/nemotron_dense_mma.cu, lane S3: the tile,
// the cp.async ring, the swizzled A, the per-warp trellis decode into B fragments, the grouped raster, the fp32 epilogue), so that GLM's P2
// bits are a function of this file and of GLM's arithmetic id alone (a change to Nemotron's kernel cannot move them). Two K orders, one
// template switch (FOLD), both P2:
//   chain   (FOLD = false): each output's K sum is ONE chain of m16n8k16 MMAs over the k tiles kt = 0, 1, .., K/16 - 1 in order from +0.0
//           (fp32 accumulate): bit for bit what gb_ndm_gemm gives (gated in crates/cuda/tests/glm_p2_dense.rs);
//   fold128 (FOLD = true):  per 128-k block b = 0, 1, .. in order, the chain of its 8 k tiles from +0.0, then tot = tot + part_b (one fp32
//           add, IEEE round to nearest). A long single chain on GB10 accumulates a systematic shrink that grows with K (measured in the
//           test above: about -1e-5 of the output rms at K = 16,384 against the decode path's none); the per-block partials avoid it at
//           the price of a second set of accumulators.
// In both, nothing depends on M, the tile shape, the grid or the rank (no split K, no stream K, no atomics), so a row's bits depend on that
// row and the weights only (P2 (a), (b)); activations enter the MMA as fp16, the trellis decoded by the decode path's decoder
// (exl3_linear_decode.cuh, bit for bit), fp32 accumulation (P2 (c)); deterministic (P2 (d)).

#include <math.h>
#include <stdint.h>

#include "common.cuh"
#include "exl3_linear_decode.cuh"

using namespace gbx;

namespace {

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

constexpr int kThreads = 256;  // 8 warps, warp w owns columns 16 w .. 16 w + 15 of the block's 128 (a strip)
constexpr int kBK = 64;        // k of one pipeline stage (4 k tiles); a 128-k fold block is two stages
constexpr int kStages = 3;

__device__ __forceinline__ float load_scale(const void* p, int f32, int i) {
    return f32 ? __ldg(static_cast<const float*>(p) + i) : __half2float(__ldg(static_cast<const half*>(p) + i));
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

// 16 bytes global -> shared, zero-filled when !valid (src must still be a readable address).
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

// a [M, K] fp16 = (x * suh) H / sqrt(128), fp32 inside, rounded to fp16 once. One warp a 128-wide block of one row.
__global__ void __launch_bounds__(128) rotate_kernel(const void* __restrict__ x, int x_dtype, long long ldx, const void* __restrict__ suh,
                                                     int suh_f32, half* __restrict__ a, int K) {
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5), row = blockIdx.x, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4];
    load4(x, x_dtype, (size_t)row * ldx + k, v);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= load_scale(suh, suh_f32, k + j);
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4(a, F16, (size_t)row * K + k, v);
}

struct GemmArgs {
    const half* a;      // [M, K] fp16, rotated
    const uint32_t* T;  // trellis [K/16][N/16][TW words]
    const void* svh;    // [N]
    const void* bias;   // [N] or null
    int scale_f32;      // svh / bias are f32 (1) or fp16 (0)
    void* y;            // [M, ldy] of y_dtype at column 0
    int y_dtype;
    long long ldy;
    int M, K, N;
};

// The block's tile: blockIdx.x walks kGroupM M tiles fastest inside a group, then the column blocks, then the groups (the threadblock
// swizzle): the CTAs resident at once share a few column blocks' trellis and a group's A rows, so both stay in L2.
constexpr int kGroupM = 8;
template <int GM = kGroupM>
__device__ __forceinline__ void tile_of(int bid, int num_m, int num_n, int& mt, int& nt) {
    const int group = GM * num_n;
    const int gid = bid / group, first = gid * GM;
    const int gsz = min(num_m - first, GM);
    const int in = bid % group;
    mt = first + in % gsz;
    nt = in / gsz;
}

// One BM x (128 SN) output tile through the whole K: SN 128-column strips (the output Hadamard's unit) a block; warp w owns the SN
// 16-column tiles w SN .. w SN + SN - 1 (strip w SN / 8) and all BM rows. A strip past N (an odd strip count) is skipped whole.
template <int K2, int CB, int BM, int SN, bool FOLD>
__global__ void __launch_bounds__(kThreads, (BM == 64 && !(FOLD && SN == 2)) ? 2 : 1) gemm_kernel(const GemmArgs g) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    constexpr int MT = BM / 16;
    constexpr int A_BYTES = BM * kBK * 2;
    constexpr int TN = 8 * SN;
    constexpr int B_WORDS = (kBK / 16) * TN * TW;
    constexpr int STAGE = A_BYTES + B_WORDS * 4;
    extern __shared__ __align__(128) unsigned char smem[];

    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int g8 = lane >> 2, t4 = lane & 3;
    const int M = g.M, K = g.K, N = g.N;
    const int strips = N / 128;
    int mt, nt;
    tile_of(blockIdx.x, (M + BM - 1) / BM, (strips + SN - 1) / SN, mt, nt);
    const int m0 = mt * BM, s0 = nt * SN;
    const int my_strips = min(SN, strips - s0);
    const bool active = (warp * SN) / 8 < my_strips;
    const int KC = K / kBK;
    const int tiles_n = N / 16;

    auto stage_a = [&](int s) { return reinterpret_cast<half*>(smem + s * STAGE); };
    auto stage_b = [&](int s) { return reinterpret_cast<uint32_t*>(smem + s * STAGE + A_BYTES); };

    auto load_stage = [&](int s, int kc) {
        half* sa = stage_a(s);
        constexpr int CPR = kBK / 8;
#pragma unroll
        for (int i = 0; i < BM * CPR / kThreads; ++i) {
            const int idx = i * kThreads + t, r = idx / CPR, c = idx % CPR;
            const int gr = m0 + r;
            const bool ok = gr < M;
            const half* src = g.a + (size_t)(ok ? gr : 0) * K + kc * kBK + c * 8;
            cp_async16(sa + r * kBK + ((c ^ (r & 7)) * 8), src, ok);
        }
        uint32_t* sb = stage_b(s);
        constexpr int CH = B_WORDS / 4;
        constexpr int CH_ROW = TN * TW / 4;
        const int ch_valid = my_strips * 8 * TW / 4;
#pragma unroll
        for (int i = 0; i < (CH + kThreads - 1) / kThreads; ++i) {
            const int c = i * kThreads + t;
            if (c < CH) {
                const int kl = c / CH_ROW, w = c % CH_ROW;
                if (w < ch_valid) {
                    const uint32_t* src = g.T + ((size_t)(kc * (kBK / 16) + kl) * tiles_n + s0 * 8) * TW + w * 4;
                    cp_async16(sb + kl * TN * TW + w * 4, src, true);
                }
            }
        }
    };

    float acc[MT][SN][2][4];
    float tot[FOLD ? MT : 1][FOLD ? SN : 1][2][4];
#pragma unroll
    for (int p = 0; p < MT; ++p)
#pragma unroll
        for (int j = 0; j < SN; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[p][j][h][c] = 0.f;
    if constexpr (FOLD) {
#pragma unroll
        for (int p = 0; p < MT; ++p)
#pragma unroll
            for (int j = 0; j < SN; ++j)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) tot[p][j][h][c] = 0.f;
    }

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < KC) load_stage(s, s);
        cp_async_commit();
    }

    const int lr = (lane & 7) + 8 * ((lane >> 3) & 1), lc = lane >> 4;

#pragma unroll 1
    for (int kc = 0; kc < KC; ++kc) {
        cp_async_wait<kStages - 2>();
        __syncthreads();
        {
            const int nk = kc + kStages - 1;
            if (nk < KC) load_stage(nk % kStages, nk);
            cp_async_commit();
        }
        if (!active) continue;
        const half* sa = stage_a(kc % kStages);
        const uint32_t* sb = stage_b(kc % kStages);
#pragma unroll
        for (int kk = 0; kk < kBK / 16; ++kk) {
            uint32_t b[SN][2][2];
#pragma unroll
            for (int j = 0; j < SN; ++j) {
                uint32_t w[LW];
                load_lane_words<K2>(sb + (kk * TN + warp * SN + j) * TW, lane, w);
                decode_lane<K2, CB>(w, lane, b[j][0], b[j][1]);
            }
#pragma unroll
            for (int p = 0; p < MT; ++p) {
                uint32_t a[4];
                const int r = p * 16 + lr, c = kk * 2 + lc;
                ldsm_x4(a, sa + r * kBK + ((c ^ (r & 7)) * 8));
#pragma unroll
                for (int j = 0; j < SN; ++j) {
                    mma16816(acc[p][j][0], a, b[j][0]);
                    mma16816(acc[p][j][1], a, b[j][1]);
                }
            }
        }
        if constexpr (FOLD) {
            // the end of 128-k block kc / 2: its partials join the running sums in block order, the chains restart from +0.0
            if (kc & 1) {
#pragma unroll
                for (int p = 0; p < MT; ++p)
#pragma unroll
                    for (int j = 0; j < SN; ++j)
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
    cp_async_wait<0>();

    // Epilogue, one strip at a time: the fp32 sums of the BM x 128 strip through shared memory, then per row (one warp, 4 columns a
    // lane): H_128, * 1/sqrt 128, * svh, + bias.
    constexpr int LDS = 128 + 4;
    float* stg = reinterpret_cast<float*>(smem);
    const int rows = min(BM, M - m0);
#pragma unroll 1
    for (int s = 0; s < my_strips; ++s) {
        __syncthreads();
        if ((warp * SN) / 8 == s) {
#pragma unroll
            for (int p = 0; p < MT; ++p)
#pragma unroll
                for (int j = 0; j < SN; ++j)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = ((warp * SN + j) % 8) * 16 + h * 8 + 2 * t4;
                        const int r = p * 16 + g8;
                        float v[4];
                        if constexpr (FOLD) {
#pragma unroll
                            for (int c = 0; c < 4; ++c) v[c] = tot[p][j][h][c];
                        } else {
#pragma unroll
                            for (int c = 0; c < 4; ++c) v[c] = acc[p][j][h][c];
                        }
                        *reinterpret_cast<float2*>(stg + r * LDS + col) = make_float2(v[0], v[1]);
                        *reinterpret_cast<float2*>(stg + (r + 8) * LDS + col) = make_float2(v[2], v[3]);
                    }
        }
        __syncthreads();

        const int n = (s0 + s) * 128 + 4 * lane;
        float sv[4], bv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sv[j] = load_scale(g.svh, g.scale_f32, n + j);
            if (g.bias) bv[j] = load_scale(g.bias, g.scale_f32, n + j);
        }
        for (int r = warp; r < rows; r += kThreads / 32) {
            const float4 u = *reinterpret_cast<const float4*>(stg + r * LDS + 4 * lane);
            float v[4] = {u.x, u.y, u.z, u.w};
            fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float x = (v[j] * HAD_SCALE) * sv[j];
                if (g.bias) x = x + bv[j];
                v[j] = x;
            }
            store4(g.y, g.y_dtype, (size_t)(m0 + r) * g.ldy + n, v);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The chain GEMM for the 3- to 8-bit packs (Flash-Next prefill): the same arithmetic as gemm_kernel<.., FOLD = false> (every output's
// K sum is ONE chain of m16n8k16 MMAs over kt = 0 .. K/16 - 1 in order from +0.0, the same fragment layouts, the same trellis decoder,
// the same epilogue), only the work split and the loop's bookkeeping differ: NW warps, each owning TPW 16-column tiles of the CTA's
// NW * TPW (a multiple of 8: whole 128-column strips) and all BM rows, BK k per stage, ST stages, MINB CTAs an SM, GM M tiles to a
// raster group. The accumulators of a (row, column) never leave the one thread that owns them and see the k tiles in order, so
// nothing here moves a bit. The per-thread addresses (cp.async sources and destinations, ldmatrix and decoder-word reads) are made
// once before the k loop, so the loop itself is the decode, the MMAs and a few adds.

// 16-byte chunk c of row r of a BM x BK fp16 tile in shared memory (in 16-byte units), swizzled so the 8 rows of an ldmatrix phase and
// the cp.async stores hit distinct banks. The swizzle depends on r mod 8 only (so a row offset of a multiple of 8 never changes it).
template <int BK>
__device__ __forceinline__ int a_swz(int r) {
    return BK == 64 ? (r & 7) : ((r >> 1) & 3);
}
template <int BK>
__device__ __forceinline__ int a_chunk(int r, int c) {
    constexpr int CPR = BK / 8;
    return r * CPR + (c ^ a_swz<BK>(r));
}

__device__ __forceinline__ void cp_async16_u32(uint32_t smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(gmem));
}
__device__ __forceinline__ void cp_async16_zfill(uint32_t smem, const void* gmem, bool valid) {
    const int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem), "l"(gmem), "r"(n));
}
template <int IMM>
__device__ __forceinline__ void ldsm_x4_at(uint32_t (&a)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4 + %5];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(addr), "n"(IMM));
}

// The MMAs of m16 tiles P .. MT - 1 of a k step: A by ldmatrix at la plus the tile's row offset (a compile-time immediate).
template <int P, int MT, int TPW, int CPR>
__device__ __forceinline__ void mma_rows(float (&acc)[MT][TPW][2][4], uint32_t la, const uint32_t (&b)[TPW][2][2]) {
    if constexpr (P < MT) {
        uint32_t a[4];
        ldsm_x4_at<P * 16 * CPR * 16>(a, la);
#pragma unroll
        for (int j = 0; j < TPW; ++j) {
            mma16816(acc[P][j][0], a, b[j][0]);
            mma16816(acc[P][j][1], a, b[j][1]);
        }
        mma_rows<P + 1, MT, TPW, CPR>(acc, la, b);
    }
}

constexpr int kEpiLds = 128 + 8;  // floats per staged row: 16-byte aligned, and the float2 stores of a half warp hit 32 distinct banks

template <int K2, int NW, int TPW, int BM, int BK, int ST>
constexpr size_t smem_bytes_c() {
    constexpr size_t stage = (size_t)BM * BK * 2 + (size_t)(BK / 16) * (NW * TPW) * tile_words<K2>() * 4;
    constexpr size_t pipe = ST * stage;
    constexpr size_t epi = (size_t)64 * kEpiLds * 4;
    return pipe > epi ? pipe : epi;
}

template <int K2, int CB, int NW, int TPW, int BM, int BK, int ST, int MINB, int GM>
__global__ void __launch_bounds__(NW * 32, MINB) gemm_c_kernel(const GemmArgs g) {
    constexpr int NT = NW * 32;
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    constexpr int MT = BM / 16;
    constexpr int TN = NW * TPW;       // 16-column tiles of the CTA
    constexpr int SS = TN / 8;         // 128-column strips of the CTA
    constexpr int KK = BK / 16;
    constexpr int CPR = BK / 8;        // 16-byte chunks of an A row of a stage
    constexpr int A_BYTES = BM * BK * 2;
    constexpr int B_WORDS = KK * TN * TW;
    constexpr int STAGE = A_BYTES + B_WORDS * 4;
    constexpr int NH = BM / 64;        // 64-row halves of the epilogue
    constexpr int A_IT = BM * CPR / NT;   // A chunks a thread copies a stage
    constexpr int ARI = NT / CPR;         // rows between a thread's consecutive A chunks
    constexpr int CH = B_WORDS / 4;       // B chunks of a stage
    constexpr int CH_ROW = TN * TW / 4;   // ... of one k tile row
    constexpr int B_IT = (CH + NT - 1) / NT;
    static_assert(TN % 8 == 0 && BM % 64 == 0 && (BK == 32 || BK == 64) && BM * CPR % NT == 0 && ARI % 8 == 0, "tile shape");
    extern __shared__ __align__(128) unsigned char smem[];

    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int g8 = lane >> 2, t4 = lane & 3;
    const int M = g.M, K = g.K, N = g.N;
    const int strips = N / 128;
    int mt, nt;
    tile_of<GM>(blockIdx.x, (M + BM - 1) / BM, (strips + SS - 1) / SS, mt, nt);
    const int m0 = mt * BM, s0 = nt * SS;
    const int my_strips = min(SS, strips - s0);
    const bool active = (warp * TPW) / 8 < my_strips;
    const int KC = K / BK;
    const int tiles_n = N / 16;
    const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

    // ---- the copies of a stage: this thread's chunks of A and B
    const bool full_m = m0 + BM <= M;
    const int a_r = t / CPR, a_c = t % CPR;
    const uint32_t a_dst = sbase + a_chunk<BK>(a_r, a_c) * 16;
    const half* a_src = g.a + (size_t)(m0 + a_r) * K + a_c * 8;
    const size_t a_stride = (size_t)ARI * K;
    const int ch_valid = my_strips * 8 * TW / 4;
    const uint32_t* b_src[B_IT];
    uint32_t b_dst[B_IT];
    uint32_t b_ok = 0;
#pragma unroll
    for (int i = 0; i < B_IT; ++i) {
        const int c = i * NT + t, kl = c / CH_ROW, w = c % CH_ROW;
        b_dst[i] = sbase + A_BYTES + (kl * TN * TW + w * 4) * 4;
        b_src[i] = g.T + ((size_t)kl * tiles_n + s0 * 8) * TW + w * 4;
        b_ok |= (uint32_t)(c < CH && w < ch_valid) << i;
    }
    const size_t b_kc = (size_t)KK * tiles_n * TW;  // words between a k stage's rows of tiles in the trellis

    auto load_stage = [&](int s, int kc) {
        const uint32_t soff = s * STAGE;
        const half* pa = a_src + (size_t)kc * BK;
#pragma unroll
        for (int i = 0; i < A_IT; ++i) {
            const half* q = pa + (size_t)i * a_stride;
            if (full_m) {
                cp_async16_u32(a_dst + soff + i * NT * 16, q);
            } else {
                const bool ok = m0 + a_r + i * ARI < M;
                cp_async16_zfill(a_dst + soff + i * NT * 16, ok ? q : g.a, ok);
            }
        }
        const size_t bo = (size_t)kc * b_kc;
#pragma unroll
        for (int i = 0; i < B_IT; ++i)
            if ((b_ok >> i) & 1u) cp_async16_u32(b_dst[i] + soff, b_src[i] + bo);
    };

    float acc[MT][TPW][2][4];
#pragma unroll
    for (int p = 0; p < MT; ++p)
#pragma unroll
        for (int j = 0; j < TPW; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[p][j][h][c] = 0.f;

#pragma unroll
    for (int s = 0; s < ST - 1; ++s) {
        if (s < KC) load_stage(s, s);
        cp_async_commit();
    }

    // ---- the reads of a stage: ldmatrix rows of A, and the words of this lane's trellis windows
    const int lr = (lane & 7) + 8 * ((lane >> 3) & 1), lc = lane >> 4;
    uint32_t ld_a[KK];
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) ld_a[kk] = sbase + (lr * CPR + ((kk * 2 + lc) ^ a_swz<BK>(lr))) * 16;
    int w0, w_off;
    lane_start<K2>(lane, w0, w_off);
    int lw[LW];
#pragma unroll
    for (int i = 0; i < LW; ++i) lw[i] = A_BYTES + (warp * TPW * TW + (w0 + i) % TW) * 4;

#pragma unroll 1
    for (int kc = 0; kc < KC; ++kc) {
        cp_async_wait<ST - 2>();
        __syncthreads();
        {
            const int nk = kc + ST - 1;
            if (nk < KC) load_stage(nk % ST, nk);
            cp_async_commit();
        }
        if (!active) continue;
        const int soff = (kc % ST) * STAGE;
#pragma unroll
        for (int kk = 0; kk < KK; ++kk) {
            uint32_t b[TPW][2][2];
#pragma unroll
            for (int j = 0; j < TPW; ++j) {
                uint32_t w[LW];
#pragma unroll
                for (int i = 0; i < LW; ++i)
                    w[i] = *reinterpret_cast<const uint32_t*>(smem + soff + lw[i] + (kk * TN + j) * TW * 4);
                decode_lane<K2, CB>(w, lane, b[j][0], b[j][1]);
            }
            mma_rows<0, MT, TPW, CPR>(acc, ld_a[kk] + soff, b);
        }
    }
    cp_async_wait<0>();

    // Epilogue, a 128-column strip and 64 rows at a time: the fp32 sums through shared memory, then per row (one warp, 4 columns a
    // lane): H_128, * 1/sqrt 128, * svh, + bias: gemm_kernel's epilogue, line for line.
    float* stg = reinterpret_cast<float*>(smem);
    const int rows = min(BM, M - m0);
#pragma unroll 1
    for (int s = 0; s < my_strips; ++s) {
#pragma unroll
        for (int hh = 0; hh < NH; ++hh) {
            if (hh * 64 >= rows) continue;
            __syncthreads();
            if (active && (warp * TPW) / 8 == s) {
#pragma unroll
                for (int pl = 0; pl < 4; ++pl)
#pragma unroll
                    for (int j = 0; j < TPW; ++j)
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const int col = ((warp * TPW + j) % 8) * 16 + h * 8 + 2 * t4;
                            const int r = pl * 16 + g8;
                            const float(&c)[4] = acc[hh * 4 + pl][j][h];
                            *reinterpret_cast<float2*>(stg + r * kEpiLds + col) = make_float2(c[0], c[1]);
                            *reinterpret_cast<float2*>(stg + (r + 8) * kEpiLds + col) = make_float2(c[2], c[3]);
                        }
            }
            __syncthreads();

            const int n = (s0 + s) * 128 + 4 * lane;
            float sv[4], bv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sv[j] = load_scale(g.svh, g.scale_f32, n + j);
                if (g.bias) bv[j] = load_scale(g.bias, g.scale_f32, n + j);
            }
            const int rn = min(64, rows - hh * 64);
            for (int r = warp; r < rn; r += NW) {
                const float4 u = *reinterpret_cast<const float4*>(stg + r * kEpiLds + 4 * lane);
                float v[4] = {u.x, u.y, u.z, u.w};
                fwht128(v, lane);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float x = (v[j] * HAD_SCALE) * sv[j];
                    if (g.bias) x = x + bv[j];
                    v[j] = x;
                }
                store4(g.y, g.y_dtype, (size_t)(m0 + hh * 64 + r) * g.ldy + n, v);
            }
        }
    }
}

template <int K2, int CB, int NW, int TPW, int BM, int BK, int ST, int MINB, int GM>
int launch_gemm_c(const GemmArgs& g, cudaStream_t st) {
    constexpr size_t sm = smem_bytes_c<K2, NW, TPW, BM, BK, ST>();
    static_assert(sm * MINB <= 101376 - 1024 * MINB, "the CTAs of an SM must fit the 99 KB of shared memory");
    static bool attr = false;  // idempotent; a race between host threads sets the same value twice
    if (!attr) {
        const cudaError_t e = cudaFuncSetAttribute(gemm_c_kernel<K2, CB, NW, TPW, BM, BK, ST, MINB, GM>,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
        if (e != cudaSuccess) return (int)e;
        attr = true;
    }
    constexpr int ss = NW * TPW / 8;
    const int strips = g.N / 128;
    const unsigned blocks = (unsigned)(((g.M + BM - 1) / BM) * ((strips + ss - 1) / ss));
    gemm_c_kernel<K2, CB, NW, TPW, BM, BK, ST, MINB, GM><<<blocks, NW * 32, sm, st>>>(g);
    return GB_CHECK_LAUNCH();
}

// The chain GEMM's configurations (cfg = tile_force - 10000; 1 is what a 3- to 8-bit chain layer of enough rows runs by default), one row each: NW,
// TPW, BM, BK, ST, MINB, GM (5 is what the chain layers run by default, see below). MEASURED on one GB10 (crates/cuda/tests/fn_dense_chain.rs bench, 4,096 rows of the Flash-Next pack's dense
// layers): the 6-bit layers run 1 CTA an SM through the old 64 x 2 and 128 x 2 tiles (96 KB of stages each), the 128 x 128 tile with
// two stages is 22 KB a CTA, so two CTAs share an SM and one's epilogue and fill hide under the other's MMAs.
template <int K2, int CB>
int dispatch_c(int cfg, const GemmArgs& g, cudaStream_t st) {
    switch (cfg) {
        case 1: return launch_gemm_c<K2, CB, 4, 2, 128, 64, 2, 2, 16>(g, st);
        case 2: return launch_gemm_c<K2, CB, 8, 1, 128, 64, 2, 2, 16>(g, st);
        case 3: return launch_gemm_c<K2, CB, 4, 2, 128, 32, 4, 2, 16>(g, st);
        case 4: return launch_gemm_c<K2, CB, 8, 1, 64, 64, 3, 2, 16>(g, st);
        // configuration 1's tile and ring with other raster groups (GM M tiles run together: the A rows of a group are re-read by every column block, so
        // the group's rotated rows should fit the L2 beside the trellis columns streaming past them): 5 GM = 8, 6 GM = 4, 7 GM = 32
        case 5: return launch_gemm_c<K2, CB, 4, 2, 128, 64, 2, 2, 8>(g, st);
        case 6: return launch_gemm_c<K2, CB, 4, 2, 128, 64, 2, 2, 4>(g, st);
        case 7: return launch_gemm_c<K2, CB, 4, 2, 128, 64, 2, 2, 32>(g, st);
        default: return (int)cudaErrorInvalidValue;
    }
}

template <int K2, int BM, int SN>
constexpr size_t smem_bytes() {
    constexpr size_t stage = (size_t)BM * kBK * 2 + (size_t)(kBK / 16) * 8 * SN * tile_words<K2>() * 4;
    constexpr size_t pipe = kStages * stage;
    constexpr size_t epi = (size_t)BM * (128 + 4) * 4;
    return pipe > epi ? pipe : epi;
}

template <int K2, int CB, int BM, int SN, bool FOLD>
int launch_gemm(const GemmArgs& g, cudaStream_t st) {
    constexpr size_t sm = smem_bytes<K2, BM, SN>();
    static_assert(sm <= 101376, "a block's shared memory must fit the 99 KB an SM gives one");
    static bool attr = false;  // idempotent; a race between host threads sets the same value twice
    if (!attr) {
        const cudaError_t e = cudaFuncSetAttribute(gemm_kernel<K2, CB, BM, SN, FOLD>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
        if (e != cudaSuccess) return (int)e;
        attr = true;
    }
    const int strips = g.N / 128;
    const unsigned blocks = (unsigned)(((g.M + BM - 1) / BM) * ((strips + SN - 1) / SN));
    gemm_kernel<K2, CB, BM, SN, FOLD><<<blocks, kThreads, sm, st>>>(g);
    return GB_CHECK_LAUNCH();
}

// The tile shape (never changes a bit). 0: pick; else BM * 10 + SN (641, 642, 1281, 1282). The chain's pick is nemotron_dense_mma.cu's
// (MEASURED there): 64 rows by 2 strips whenever that gives at least two waves of resident blocks, else 64 by 1. The fold's (MEASURED on
// node 4 over the 18 GLM shapes at 512 and 2,048 rows, crates/cuda/tests/glm_p2_dense.rs bench): 128 rows by 1 strip for a big layer
// (K x N of 32 Mi or more) at 1,024 rows or more (2-5 % over 64 x 1 there: the decoded tile feeds twice the rows), else 64 by 1 (two
// blocks an SM; the narrow and the short-K layers: 128 x 1 lost up to 12 % on q_b). Its 64 x 2 tile holds twice the accumulators.
int pick_tile(int m, int k, int n, int fold, int force, int& bm, int& sn) {
    if (force) {
        bm = force / 10;
        sn = force % 10;
        return (bm == 64 || bm == 128) && (sn == 1 || sn == 2) ? 0 : (int)cudaErrorInvalidValue;
    }
    const int strips = n / 128, sms = 48;
    bm = 64;
    if (fold) {
        bm = (m >= 1024 && (long long)k * n >= (32LL << 20)) ? 128 : 64;
        sn = 1;
        return 0;
    }
    const long long blocks2 = (long long)((m + 63) / 64) * ((strips + 1) / 2);
    sn = blocks2 >= 2 * (2 * sms) ? 2 : 1;
    return 0;
}

template <bool FOLD>
int dispatch(int bits, int cb, int bm, int sn, const GemmArgs& g, cudaStream_t st) {
    // fold128 at 128 x 2 would hold 256 fp32 accumulators a thread (more than the 255 registers): not instantiated
    if (FOLD && bm == 128 && sn == 2) return (int)cudaErrorInvalidValue;
#define GP2_T(K2_, CB_)                                                                                                          \
    if (bm == 64) return sn == 1 ? launch_gemm<K2_, CB_, 64, 1, FOLD>(g, st) : launch_gemm<K2_, CB_, 64, 2, FOLD>(g, st); \
    if constexpr (FOLD) return launch_gemm<K2_, CB_, 128, 1, FOLD>(g, st);                                                   \
    else return sn == 1 ? launch_gemm<K2_, CB_, 128, 1, FOLD>(g, st) : launch_gemm<K2_, CB_, 128, 2, FOLD>(g, st);
    if (cb != CB_MUL1) return (int)cudaErrorInvalidValue;  // GLM's packs are Mul1 throughout
    switch (bits) {
        case 2: { GP2_T(4, CB_MUL1) }
        case 3: { GP2_T(6, CB_MUL1) }
        case 4: { GP2_T(8, CB_MUL1) }
        case 5: { GP2_T(10, CB_MUL1) }
        case 6: { GP2_T(12, CB_MUL1) }
        case 7: { GP2_T(14, CB_MUL1) }
        case 8: { GP2_T(16, CB_MUL1) }
        default: return (int)cudaErrorInvalidValue;
    }
#undef GP2_T
}

// The chain GEMM (gemm_c_kernel) at configuration cfg.
int dispatch_chain_c(int bits, int cb, int cfg, const GemmArgs& g, cudaStream_t st) {
    if (cb != CB_MUL1) return (int)cudaErrorInvalidValue;
    switch (bits) {
        case 2: return dispatch_c<4, CB_MUL1>(cfg, g, st);
        case 3: return dispatch_c<6, CB_MUL1>(cfg, g, st);
        case 4: return dispatch_c<8, CB_MUL1>(cfg, g, st);
        case 5: return dispatch_c<10, CB_MUL1>(cfg, g, st);
        case 6: return dispatch_c<12, CB_MUL1>(cfg, g, st);
        case 7: return dispatch_c<14, CB_MUL1>(cfg, g, st);
        case 8: return dispatch_c<16, CB_MUL1>(cfg, g, st);
        default: return (int)cudaErrorInvalidValue;
    }
}

bool dtype_ok(int d) { return d >= 0 && d <= 2; }

}  // namespace

// Whether GLM's P2 GEMM takes a layer: K and N multiples of 128, the Mul1 codebook at 2 to 8 bits.
GB_EXPORT int gb_glm_p2_supported(int k, int n, int bits, int codebook) {
    const bool shape = k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0;
    return shape && bits >= 2 && bits <= 8 && codebook == CB_MUL1;
}

// a [m, k] fp16 = (x * suh) H / sqrt(128); x [m, ldx] of x_dtype (F16 0, BF16 1, F32 2), suh [k] f32 (suh_f32) or fp16.
GB_EXPORT int gb_glm_p2_rotate(const void* x, int x_dtype, long long ldx, const void* suh, int suh_f32, void* a, int m, int k,
                               void* stream) {
    if (m < 1 || k <= 0 || k % 128 || ldx < k || !dtype_ok(x_dtype) || !x || !suh || !a) return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)m, (unsigned)((k / 128 + 3) / 4));
    rotate_kernel<<<grid, 128, 0, gb_stream(stream)>>>(x, x_dtype, ldx, suh, suh_f32, static_cast<half*>(a), k);
    return GB_CHECK_LAUNCH();
}

// y [m, ldy] (y_dtype, from column 0) = the EXL3 layer applied to the rotated rows a [m, k] fp16 (gb_glm_p2_rotate's output). svh and
// bias f32 (scale_f32) or fp16; bias may be null. fold: the K order (0 chain, 1 fold128: see the file comment). tile_force 0 (pick) or
// BM * 10 + SN (641, 642, 1281, 1282; fold128 not 1282) or, chain only, 10000 + a gemm_c_kernel configuration (10001 to 10007), which
// never changes a bit. The pick of a chain layer of 3 to 8 bits with 1,024 rows or more (512 rows or more and 2,048 columns or more) is gemm_c_kernel's configuration 5 (configuration 1's tile with raster groups of 8 M tiles); every other call's is pick_tile's
// (the 641 .. 1282 kernels stay the reference of the tests). Returns 0 or a cudaError_t; never allocates or synchronizes.
GB_EXPORT int gb_glm_p2_gemm(const void* a, const void* trellis, const void* svh, const void* bias, int scale_f32, void* y, int y_dtype,
                             long long ldy, int m, int k, int n, int bits, int codebook, int fold, int tile_force, void* stream) {
    if (m < 1 || !gb_glm_p2_supported(k, n, bits, codebook) || !dtype_ok(y_dtype) || !a || !trellis || !svh || !y || ldy < n)
        return (int)cudaErrorInvalidValue;
    GemmArgs g{static_cast<const half*>(a), static_cast<const uint32_t*>(trellis), svh, bias, scale_f32, y, y_dtype, ldy, m, k, n};
    if (!fold && tile_force >= 10000) return dispatch_chain_c(bits, codebook, tile_force - 10000, g, gb_stream(stream));
    // MEASURED at 16 to 2,048 rows (crates/cuda/tests/fn_dense_chain.rs bench_dense_chain_row_counts): the 128 x 128 tile wins from 1,024
    // rows on and from 512 on a layer with 2,048 columns or more; below that the 64-row tiles' smaller grids fill the SMs better (a 512-column
    // layer at 512 rows: 47 against 74 us, 2,560 columns at 16 to 128 rows: 83 to 88 against 139 to 148 us).
    if (!fold && tile_force == 0 && bits >= 3 && (m >= 1024 || (m >= 512 && n >= 2048)))
        return dispatch_chain_c(bits, codebook, 5, g, gb_stream(stream));
    int bm = 0, sn = 0;
    const int rc = pick_tile(m, k, n, fold, tile_force, bm, sn);
    if (rc) return rc;
    return fold ? dispatch<true>(bits, codebook, bm, sn, g, gb_stream(stream)) : dispatch<false>(bits, codebook, bm, sn, g, gb_stream(stream));
}

// The tile of the 641 .. 1282 kernels that gb_glm_p2_gemm picks for (m, k, n) and a K order when not forced, as BM * 10 + SN (for the
// tables; a chain layer of 3 to 8 bits and enough rows runs gemm_c_kernel instead, see gb_glm_p2_gemm).
GB_EXPORT int gb_glm_p2_pick_tile(int m, int k, int n, int fold) {
    int bm = 0, sn = 0;
    pick_tile(m, k, n, fold, 0, bm, sn);
    return bm * 10 + sn;
}
