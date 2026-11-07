// gb-nvcc: --fmad=false
// GLM-5.3-Flash multi-head latent attention (NoPE, 64 heads, 512-wide latent), dense and sparse regimes. One latent per token per layer
// is the whole KV state: the cache holds the normalised 512-wide latent c_kv (no rope key: qk_rope_head_dim is 0) and
// that same vector is both K and V of all 64 heads (MQA over a compressed latent). kv_b_proj is a plain dense weight
// used twice, never as a GEMM over the context:
//
//   q_lat[h][c]  = sum_d q_nope[h][d] * W_UK[h][d][c]              (absorb)
//   score[h][j]  = (q_lat[h] . c_kv[j]) / 16                       (sm_scale = 256^-0.5, exact in fp32)
//   o_lat[h][c]  = sum_j softmax_j(score[h][j]) * c_kv[j][c]
//   out[h][n]    = sum_c o_lat[h][c] * W_UV[h][c][n]               (expand, n < 256 = v_head_dim)
//
// Layouts (T = fp16 or bf16 for every activation, the cache and the weights; one T per call chain):
//   latent cache   T [n_blocks][256 tokens][512]   one buffer per layer; a block table i32 [slots][max_blocks] maps
//                  (slot, position / 256) -> block id, exactly as the GQA cache does
//   w_uk           T [64][256 d][512 c]            kv_b_proj rows h*512 + d (d < 256), repacked contiguous
//   w_uvt          T [64][512 c][256 n]            kv_b_proj rows h*512 + 256 + n, TRANSPOSED to [c][n]
//   q (nope)       T [rows][row_stride]            head h at h*256 (row_stride = 64*256 for a plain q_b_proj output)
//   q_lat, o_lat   T [rows][64][512]
//   out            T [rows][64*256]                the o_proj input
//
// ROW-INVARIANCE BY CONSTRUCTION. Row r's output bits depend only on its own q, its slot's latent rows 0..pos[r] and
// its own position: never on the other rows, how many there are, their order, other sequences, the block placement,
// max_context, the SM count or the grid.
//  * absorb / expand: every output element is ONE fp32 chain of __fmaf_rn over the contraction index in ascending
//    order (K = 256 resp. 512), computed by one thread. The kernel variant (rows per pass) changes how many rows a
//    thread carries, never the per-row operation sequence. No atomics, no split-K.
//  * attention: keys are split into ABSOLUTE chunks of CH keys; a CTA serves (one chunk, 16 heads, ONE row), so its
//    tile stream, mask and arithmetic are those of that row alone. The scores of a tile are the sum, in fixed order,
//    of four d-quarter mma chains (the four warps); softmax state is fp32 online softmax recomputed identically in
//    every warp. A chunk's (m, l, o[512]) partial is merged IN CHUNK ORDER by a fixed formula, one thread per
//    column group, and rounded once to T. A row visits exactly the chunks 0..pos/CH. Everything on tensor cores runs
//    through mma.m16n8k16 where each (row, head) pair occupies its own mma row, so a pair's bits do not depend on its
//    tile neighbours.
//  * no float atomics anywhere; all numeric operations use explicit rounding intrinsics and the file is built with
//    --fmad=false, so the compiler cannot contract or reassociate them.
//
// Sparse (DSA) regime: the same kernel with the chunks tiling the row's SELECTION LIST instead of its key positions. Row r
// attends to the keys at sel_idx[r][0 .. count) (ascending absolute positions from the indexer), taken in list order in
// chunks of 256 entries; a tile gathers its 32 keys' 1 KiB rows through the block table (lane j resolves entry j and the copy
// loop fetches position and block by shuffle), an entry past the count or holding -1 (a hole inside the list) is masked, the
// list may be in any order, the merge visits the row's
// ceil(count / 256) chunks. On the identity list (0..=pos) it is bit-identical to the dense kernel (tested), so which kernel
// serves a row at the regime boundary is invisible in its bits. Each kernel skips the other regime's rows entirely (the dense
// one rows with pos >= dense_limit, the sparse one rows whose meta says dense), so one o_lat holds both.
//
// The tiling (CH, KT) is a compile-time constant of the default variant. It is NOT derived from the row count, the
// context length or the SM count (the ExLlamaV3 oracle derives its split from all three, which is why its bits move).
//
// Structure after TensorFold (MIT, commit 9cd52ab) families/glm5_next/cuda/latent.py (absolute 512-key chunk programs,
// per-chunk fp32 partials merged in chunk order, absorb/expand as per-row fixed-order sums) and the REXL3 GQA decode
// kernel in attention.cu (mma/ldmatrix/cp.async helpers, log2-domain fold). See THIRD_PARTY_NOTICES.md.
#include "common.cuh"
#include "mla_kv8.cuh"
#include <type_traits>

namespace {

constexpr int DC = 512;          // latent width (kv_lora_rank)
constexpr int NH = 64;           // heads
constexpr int DN = 256;          // qk_nope_head_dim == v_head_dim
constexpr int PAGE = 256;        // tokens per block
constexpr int HT = 16;           // heads per CTA: one mma row tile
constexpr int NWARP = 4;         // warps per CTA = d-quarters
constexpr int DQ = DC / NWARP;   // 128 latent columns per warp
constexpr int PQ = DQ / 8;       // 16-byte pieces of one key row held by one warp
constexpr int MAX_ROWS = 128;

constexpr float LOG2E = 1.4426950408889634f;
constexpr float SC2 = 0.0625f * LOG2E;  // (1 / 16) * log2(e): scores live in the log2 domain, exp is a bare ex2

typedef __nv_bfloat16 bf16;
typedef __half f16;

// ---------------------------------------------------------------- element type helpers

template <typename T> __device__ __forceinline__ float to_f32(T v);
template <> __device__ __forceinline__ float to_f32<bf16>(bf16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f32<f16>(f16 v) { return __half2float(v); }

template <typename T> __device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ bf16 from_f32<bf16>(float v) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ f16 from_f32<f16>(float v) { return __float2half_rn(v); }

// two packed T in a u32 (low half = first element)
template <typename T> __device__ __forceinline__ uint32_t pack2(float lo, float hi) {
    if constexpr (std::is_same<T, bf16>::value) {
        __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
        return *reinterpret_cast<uint32_t*>(&v);
    } else {
        __half2 v = __floats2half2_rn(lo, hi);
        return *reinterpret_cast<uint32_t*>(&v);
    }
}
template <typename T> __device__ __forceinline__ void unpack2(uint32_t w, float& lo, float& hi) {
    if constexpr (std::is_same<T, bf16>::value) {
        lo = __uint_as_float(w << 16);
        hi = __uint_as_float(w & 0xffff0000u);
    } else {
        lo = __half2float(__ushort_as_half((unsigned short)(w & 0xffffu)));
        hi = __half2float(__ushort_as_half((unsigned short)(w >> 16)));
    }
}

// ---------------------------------------------------------------- fragments (after TensorFold qmm_frag.cuh)

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

// 16 bytes, or zeros without reading when `read` is false
__device__ __forceinline__ void cp16z(void* dst, const void* src, bool read) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(read ? 16 : 0));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
template <typename T>
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    if constexpr (std::is_same<T, bf16>::value) {
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    } else {
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
}
// the same mma from a zero accumulator (a chain's first k16 step)
template <typename T>
__device__ __forceinline__ void mma0(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    const float z = 0.0f;
    if constexpr (std::is_same<T, bf16>::value) {
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%10, %10, %10, %10};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(z));
    } else {
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%10, %10, %10, %10};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(z));
    }
}

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

// l reduced over the quad that shares a row: lanes xor 1, then xor 2 (every lane ends with the same bits)
__device__ __forceinline__ float quad_sum(float x) {
    x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 1));
    return __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 2));
}

// 16-byte piece c (of PQ = 16) of key row r, XOR-swizzled so ldmatrix's eight rows hit eight bank groups
__device__ __forceinline__ int sw(int r, int c) { return r * PQ + (c ^ (r & 7)); }

// ---------------------------------------------------------------- gb_mla_latent_write

// One block per row: the row's 512 latent values (1 KiB) go to (block, token) of its slot; a negative position skips.
__global__ void __launch_bounds__(64) latent_write_kernel(const uint4* __restrict__ lat, const int* __restrict__ row_slot,
                                                          const int* __restrict__ row_pos, const int* __restrict__ table,
                                                          int max_blocks, uint4* __restrict__ cache) {
    const int r = blockIdx.x;
    const int pos = row_pos[r];
    if (pos < 0) return;
    const int blk = table[(int64_t)row_slot[r] * max_blocks + pos / PAGE];
    const int64_t dst = ((int64_t)blk * PAGE + pos % PAGE) * (DC / 8);  // uint4 units
    cache[dst + threadIdx.x] = lat[(int64_t)r * (DC / 8) + threadIdx.x];
}

// The fp8 page (mla_kv8.cuh): 64 threads x 8 values; the row's maximum magnitude over both warps (fmaxf: the order does not matter), the
// scale amax / 448, then every value rounded once to E4M3 at it. A row of zeros (scale 0) writes zero bytes.
template <typename T>
__global__ void __launch_bounds__(64) latent_write8_kernel(const uint4* __restrict__ lat, const int* __restrict__ row_slot,
                                                           const int* __restrict__ row_pos, const int* __restrict__ table,
                                                           int max_blocks, uint8_t* __restrict__ cache) {
    __shared__ float wmax[2];
    const int r = blockIdx.x;
    const int pos = row_pos[r];
    if (pos < 0) return;  // block-uniform
    const int blk = table[(int64_t)row_slot[r] * max_blocks + pos / PAGE], tok = pos % PAGE;
    const uint4 u = lat[(int64_t)r * (DC / 8) + threadIdx.x];
    const uint32_t uw[4] = {u.x, u.y, u.z, u.w};
    float v[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) unpack2<T>(uw[e], v[2 * e], v[2 * e + 1]);
    float a = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) a = fmaxf(a, fabsf(v[e]));
#pragma unroll
    for (int x = 16; x > 0; x >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, x));
    if ((threadIdx.x & 31) == 0) wmax[threadIdx.x >> 5] = a;
    __syncthreads();
    const float s = __fdiv_rn(fmaxf(wmax[0], wmax[1]), mla8::E4M3_MAX);
    uint2 q = make_uint2(0u, 0u);
    if (s > 0.0f) {
        q.x = mla8::q2(v[0], v[1], s) | (mla8::q2(v[2], v[3], s) << 16);
        q.y = mla8::q2(v[4], v[5], s) | (mla8::q2(v[6], v[7], s) << 16);
    }
    *reinterpret_cast<uint2*>(mla8::row_mut(cache, blk, tok) + threadIdx.x * 8) = q;
    if (threadIdx.x == 0) *mla8::scale_mut(cache, blk, tok) = s;
}

// ---------------------------------------------------------------- absorb / expand: per-head skinny GEMM, fixed-order fp32 chains

// y[r][h][j] = round_T( sum_{k = 0..K-1} x[r][h][k] * W[h][k][j] ), k ascending, one __fmaf_rn chain per output element.
// One CTA = one warp = (head h, 32 * CPT consecutive columns j); each lane owns CPT consecutive columns; weights are read
// once per pass of RB rows (a pass re-reads them from L2 for further passes). x of the pass is staged as fp32 in shared
// memory (a broadcast read per FMA group); rows past `rows` are zero and never stored. The per-row operation sequence is
// independent of RB, so every instantiation gives identical bits for a row.
template <typename T, int K, int J, int CPT, int RB, int UNROLL>
__global__ void __launch_bounds__(32) skinny_kernel(const T* __restrict__ x, int64_t x_stride, const T* __restrict__ w,
                                                    T* __restrict__ y, int64_t y_stride, int rows) {
    static_assert(K % UNROLL == 0 && UNROLL % 4 == 0 && K % 8 == 0, "tiling");
    static_assert(CPT == 4 || CPT == 8, "16-byte or 8-byte weight loads");
    __shared__ __align__(16) float sx[RB * K];
    const int lane = threadIdx.x, h = blockIdx.y, j0 = blockIdx.x * (32 * CPT) + lane * CPT;
    const T* wp = w + (int64_t)h * K * J + j0;
    for (int r0 = 0; r0 < rows; r0 += RB) {
        // stage x[r0 .. r0 + RB)[h][0..K) as fp32 (zeros past the last row)
        __syncwarp();
        for (int i = lane; i < RB * (K / 8); i += 32) {
            const int rr = i / (K / 8), c8 = i % (K / 8);
            float v[8];
            if (r0 + rr < rows) {
                const uint4 u = *reinterpret_cast<const uint4*>(x + (int64_t)(r0 + rr) * x_stride + (int64_t)h * K + c8 * 8);
                const uint32_t uw[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int e = 0; e < 4; ++e) unpack2<T>(uw[e], v[2 * e], v[2 * e + 1]);
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) v[e] = 0.0f;
            }
            float4* d = reinterpret_cast<float4*>(sx + rr * K + c8 * 8);
            d[0] = make_float4(v[0], v[1], v[2], v[3]);
            d[1] = make_float4(v[4], v[5], v[6], v[7]);
        }
        __syncwarp();
        float acc[RB][CPT];
#pragma unroll
        for (int r = 0; r < RB; ++r)
#pragma unroll
            for (int c = 0; c < CPT; ++c) acc[r][c] = 0.0f;
        for (int k0 = 0; k0 < K; k0 += UNROLL) {
            // issue the unrolled weight loads first so they are all in flight together
            uint32_t wv[UNROLL][CPT / 2];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const T* p = wp + (int64_t)(k0 + u) * J;
                if constexpr (CPT == 8) {
                    const uint4 t = *reinterpret_cast<const uint4*>(p);
                    wv[u][0] = t.x; wv[u][1] = t.y; wv[u][2] = t.z; wv[u][3] = t.w;
                } else {
                    const uint2 t = *reinterpret_cast<const uint2*>(p);
                    wv[u][0] = t.x; wv[u][1] = t.y;
                }
            }
#pragma unroll
            for (int u4 = 0; u4 < UNROLL; u4 += 4) {
                float wf[4][CPT];
#pragma unroll
                for (int u = 0; u < 4; ++u)
#pragma unroll
                    for (int e = 0; e < CPT / 2; ++e) unpack2<T>(wv[u4 + u][e], wf[u][2 * e], wf[u][2 * e + 1]);
#pragma unroll
                for (int r = 0; r < RB; ++r) {
                    const float4 xv = *reinterpret_cast<const float4*>(sx + r * K + k0 + u4);
                    const float xs[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
                    for (int u = 0; u < 4; ++u)
#pragma unroll
                        for (int c = 0; c < CPT; ++c) acc[r][c] = __fmaf_rn(xs[u], wf[u][c], acc[r][c]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RB; ++r) {
            if (r0 + r >= rows) break;
            T* yp = y + (int64_t)(r0 + r) * y_stride + (int64_t)h * J + j0;
#pragma unroll
            for (int c = 0; c < CPT; c += 2) *reinterpret_cast<uint32_t*>(yp + c) = pack2<T>(acc[r][c], acc[r][c + 1]);
        }
    }
}

// nh: heads of the call (a block a head; x, w and y point at the first of them): NH for the whole layer, a rank's share of a head split.
template <typename T, int K, int J, int CPT, int RB, int UNROLL>
int launch_skinny(const T* x, int64_t xs, const T* w, T* y, int64_t ys, int rows, int nh, cudaStream_t s) {
    skinny_kernel<T, K, J, CPT, RB, UNROLL><<<dim3(J / (32 * CPT), nh), 32, 0, s>>>(x, xs, w, y, ys, rows);
    return GB_CHECK_LAUNCH();
}

template <typename T> int absorb_t(const void* q, int64_t qs, const void* w, void* out, int rows, int nh, cudaStream_t s) {
    const T *x = (const T*)q, *wp = (const T*)w;
    T* y = (T*)out;
    const int64_t ys = (int64_t)nh * DC;
    if (rows <= 1) return launch_skinny<T, DN, DC, 8, 1, 8>(x, qs, wp, y, ys, rows, nh, s);
    if (rows <= 4) return launch_skinny<T, DN, DC, 8, 4, 8>(x, qs, wp, y, ys, rows, nh, s);
    return launch_skinny<T, DN, DC, 8, 16, 8>(x, qs, wp, y, ys, rows, nh, s);
}
template <typename T> int expand_t(const void* o_lat, const void* w, void* out, int rows, int nh, cudaStream_t s) {
    const T *x = (const T*)o_lat, *wp = (const T*)w;
    T* y = (T*)out;
    const int64_t xs = (int64_t)nh * DC, ys = (int64_t)nh * DN;
    if (rows <= 1) return launch_skinny<T, DC, DN, 4, 1, 16>(x, xs, wp, y, ys, rows, nh, s);
    if (rows <= 4) return launch_skinny<T, DC, DN, 4, 4, 16>(x, xs, wp, y, ys, rows, nh, s);
    return launch_skinny<T, DC, DN, 4, 16, 8>(x, xs, wp, y, ys, rows, nh, s);
}

// ---------------------------------------------------------------- attention over the latent

// Online-softmax state of the two heads (g and g + 8) a thread holds. m is in the log2 domain; l is this thread's
// partial over its columns (reduced across the quad at the end).
struct Soft {
    float m0, m1, l0, l1;
};

// One CTA = (chunk of CH absolute keys, MT * 16 heads, one row). Warp (mi, w) owns head tile mi and latent columns
// [128 w, 128 w + 128): it holds the A fragments of its quarter of q_lat in registers (32 regs), reads only its quarter of
// every key tile, adds its quarter's mma chain to the scores through shared memory (summed in the fixed order w = 0..3 by
// every warp of the head tile, so all four hold identical scores and run identical softmax arithmetic), and accumulates
// the output columns of its quarter (64 fp32 registers). K and V are the same tile, shared by the MT head tiles.
// A pair's arithmetic does not depend on MT, so every MT gives identical bits (tested).
// Q8: the cache is the fp8 page (mla_kv8.cuh). A thread loads its pieces of a tile into registers one tile ahead (the loads in
// flight under the current tile's arithmetic), then dequantises them into the same T tile the cp.async path fills: everything
// after the stage is unchanged, so the output is the T kernel's on the dequantised cache, bit for bit.
template <typename T, int KT, int CH, int MT, bool SPARSE, bool Q8>
__global__ void __launch_bounds__(128 * MT)
mla_partial_kernel(const T* __restrict__ qlat, const void* __restrict__ cache, const int* __restrict__ table, int max_blocks,
                   const int* __restrict__ row_slot, const int* __restrict__ row_pos, int dense_limit,
                   const int* __restrict__ sel_idx, int sel_stride, const int* __restrict__ sel_meta, int nch_max,
                   float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l, int nh) {
    const T* lat = static_cast<const T*>(cache);
    const uint8_t* lat8 = static_cast<const uint8_t*>(cache);
    constexpr int NBS = KT / 8;           // n8 blocks of scores per tile
    constexpr int NKS = KT / 16;          // k16 steps of the PV product per tile
    constexpr int TILE_U4 = KT * PQ;      // uint4 per quarter-stage
    constexpr int SX_F4 = NBS * 32;       // float4 of one warp's exchange buffer: [n block][lane], conflict-free
    constexpr int NTHR = 128 * MT;
    static_assert(PAGE % KT == 0 && CH % KT == 0 && (KT == 16 || KT == 32) && (MT == 1 || MT == 2) && (!SPARSE || MT == 1), "tiling");
    extern __shared__ uint4 smem[];
    float4* sx = reinterpret_cast<float4*>(smem + 2 * NWARP * TILE_U4);
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, g = lane >> 2, qd = lane & 3, mt = lane >> 3;
    const int w = wid & (NWARP - 1), mi = wid / NWARP;  // d-quarter, head tile within the CTA
    const int GRP = (nh / HT) / MT;                      // CTAs per row along the heads (nh: the call's heads, NH or a rank's share)
    const int row = blockIdx.x / GRP, hb = (blockIdx.x % GRP) * MT + mi, chunk = blockIdx.y;
    // `last` is the row's last valid index in the space the chunks tile: its key position (dense) or the last entry of its
    // selection list (sparse). A row the other regime serves (or a padding row) does nothing here.
    const int pos_row = row_pos[row];
    int last;
    if constexpr (SPARSE) {
        if (pos_row < 0 || sel_meta[2 * row] == 0) return;
        last = sel_meta[2 * row + 1] - 1;
    } else {
        if (pos_row >= dense_limit) return;
        last = pos_row;
    }
    const int cstart = chunk * CH;
    if (last < cstart) return;  // CTA-uniform: a chunk past the row's last key (or a padding row) does nothing
    const int limit = last + 1;
    const int nt = (min(limit, cstart + CH) - cstart + KT - 1) / KT;
    const int* tbl = table + (int64_t)row_slot[row] * max_blocks;
    unsigned vm0 = 0, vm1 = 0;  // sparse: per stage, bit j = entry j of the staged tile is a real key (not past the count, not a hole)

    // ---- stage tile t (keys [cstart + KT t, + KT)) into stage st. MT = 1: each warp stages its own quarter (no cross-warp
    // dependency); MT > 1: all threads stage the whole tile (the quarters are shared by the head tiles), made visible by a
    // CTA barrier. Past the last tile it commits an empty group, so the group count stays uniform.
    auto stage = [&](int t, int st) {
        const int key0 = cstart + t * KT;
        const bool any = t < nt;
        if constexpr (SPARSE) {
            // entry j of the tile is selection-list entry key0 + j: lane j resolves its position and block once, the copy
            // loop fetches them by shuffle
            const int e = key0 + lane;
            const int pl = (any && e < limit) ? sel_idx[(int64_t)row * sel_stride + e] : -1;
            const int bl = pl >= 0 ? tbl[pl / PAGE] : 0;
            const unsigned vmask = __ballot_sync(0xffffffffu, pl >= 0);
            if (st == 0) {
                vm0 = vmask;
            } else {
                vm1 = vmask;
            }
            const int jb = lane / PQ, c = lane % PQ;
            uint4* dst = smem + (st * NWARP + w) * TILE_U4;
#pragma unroll
            for (int it = 0; it < KT * PQ / 32; ++it) {
                const int j = jb + 2 * it;
                const int pj = __shfl_sync(0xffffffffu, pl, j), bj = __shfl_sync(0xffffffffu, bl, j);
                const bool in = pj >= 0;  // entries past the list's end and holes (< 0): zeros, never read, masked in the fold
                cp16z(dst + j * PQ + (c ^ (j & 7)), in ? lat + ((int64_t)bj * PAGE + pj % PAGE) * DC + w * DQ + c * 8 : lat, in);
            }
            return;
        }
        const int blk = any ? tbl[key0 / PAGE] : 0;
        const T* base = lat + ((int64_t)blk * PAGE + key0 % PAGE) * DC;
        if constexpr (MT == 1) {
            // lane = (key parity jb, piece c): key j = jb + 2 it, piece c is fixed per lane
            const int jb = lane / PQ, c = lane % PQ;
            uint4* dst = smem + (st * NWARP + w) * TILE_U4;
            const T* src = base + w * DQ + c * 8;
#pragma unroll
            for (int it = 0; it < KT * PQ / 32; ++it) {
                const int j = jb + 2 * it;
                const bool in = any && key0 + j < limit;  // keys past the row's last: zeros, never read
                cp16z(dst + j * PQ + (c ^ (j & 7)), in ? src + (int64_t)j * DC : lat, in);
            }
        } else {
            const int p = tid & 63, q = p >> 4, c = p & 15;  // 64 pieces of 16 B per key row, in 4 quarters
#pragma unroll
            for (int it = 0; it < KT * 64 / NTHR; ++it) {
                const int j = (tid >> 6) + (NTHR / 64) * it;
                const bool in = any && key0 + j < limit;
                cp16z(smem + (st * NWARP + q) * TILE_U4 + j * PQ + (c ^ (j & 7)), in ? base + (int64_t)j * DC + p * 8 : lat, in);
            }
        }
    };
    // ---- Q8: load8 takes tile t's pieces (and, lane j, the scale of key j) into registers; store8 dequantises them into stage st.
    // A masked key (past the row's last, past the list's end, a hole) loads nothing and stores zeros, as cp16z does.
    constexpr int NPC = MT == 1 ? KT * PQ / 32 : KT * 64 / NTHR;  // pieces of 8 values a thread stages per tile
    uint2 pv[Q8 ? NPC : 1];
    float psk = 0.0f;  // lane j: the scale of the tile's key j
    unsigned pvm = 0;  // sparse: the tile's ballot
    auto load8 = [&](int t) {
        const int key0 = cstart + t * KT;
        const bool any = t < nt;
        if constexpr (SPARSE) {
            const int e = key0 + lane;
            const int pl = (any && e < limit) ? sel_idx[(int64_t)row * sel_stride + e] : -1;
            const int bl = pl >= 0 ? tbl[pl / PAGE] : 0;
            pvm = __ballot_sync(0xffffffffu, pl >= 0);
            psk = pl >= 0 ? mla8::scale(lat8, bl, pl % PAGE) : 0.0f;
            const int jb = lane / PQ, c = lane % PQ;
#pragma unroll
            for (int it = 0; it < NPC; ++it) {
                const int j = jb + 2 * it;
                const int pj = __shfl_sync(0xffffffffu, pl, j), bj = __shfl_sync(0xffffffffu, bl, j);
                pv[it] = pj >= 0 ? *reinterpret_cast<const uint2*>(mla8::row(lat8, bj, pj % PAGE) + w * DQ + c * 8) : make_uint2(0u, 0u);
            }
            return;
        }
        const int blk = any ? tbl[key0 / PAGE] : 0, tok0 = key0 % PAGE;
        psk = (any && lane < KT && key0 + lane < limit) ? mla8::scale(lat8, blk, tok0 + lane) : 0.0f;
        if constexpr (MT == 1) {
            const int jb = lane / PQ, c = lane % PQ;
#pragma unroll
            for (int it = 0; it < NPC; ++it) {
                const int j = jb + 2 * it;
                const bool in = any && key0 + j < limit;
                pv[it] = in ? *reinterpret_cast<const uint2*>(mla8::row(lat8, blk, tok0 + j) + w * DQ + c * 8) : make_uint2(0u, 0u);
            }
        } else {
            const int p = tid & 63;
#pragma unroll
            for (int it = 0; it < NPC; ++it) {
                const int j = (tid >> 6) + (NTHR / 64) * it;
                const bool in = any && key0 + j < limit;
                pv[it] = in ? *reinterpret_cast<const uint2*>(mla8::row(lat8, blk, tok0 + j) + p * 8) : make_uint2(0u, 0u);
            }
        }
    };
    auto store8 = [&](int st) {
        if constexpr (SPARSE) {
            if (st == 0) {
                vm0 = pvm;
            } else {
                vm1 = pvm;
            }
        }
        if constexpr (MT == 1) {
            const int jb = lane / PQ, c = lane % PQ;
            uint4* dst = smem + (st * NWARP + w) * TILE_U4;
#pragma unroll
            for (int it = 0; it < NPC; ++it) {
                const int j = jb + 2 * it;
                dst[j * PQ + (c ^ (j & 7))] = mla8::dq8<T>(pv[it], __shfl_sync(0xffffffffu, psk, j));
            }
        } else {
            const int p = tid & 63, q = p >> 4, c = p & 15;
#pragma unroll
            for (int it = 0; it < NPC; ++it) {
                const int j = (tid >> 6) + (NTHR / 64) * it;
                smem[(st * NWARP + q) * TILE_U4 + j * PQ + (c ^ (j & 7))] = mla8::dq8<T>(pv[it], __shfl_sync(0xffffffffu, psk, j));
            }
        }
    };
    if constexpr (Q8) {
        load8(0);
        store8(0);
        if constexpr (MT == 1) {
            load8(1);
            store8(1);
        }
    } else {
        stage(0, 0);
        commit();
        if constexpr (MT == 1) {
            stage(1, 1);
            commit();
        }
    }

    // ---- this warp's quarter of the 16 heads' q_lat as A fragments (k16 steps 0..7 of its 128 columns)
    uint32_t qa[DQ / 16][4];
    {
        const T* qrow = qlat + ((int64_t)row * nh + hb * HT) * DC + w * DQ + 2 * qd;
#pragma unroll
        for (int k = 0; k < DQ / 16; ++k) {
            qa[k][0] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)g * DC + 16 * k);
            qa[k][1] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)(g + 8) * DC + 16 * k);
            qa[k][2] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)g * DC + 16 * k + 8);
            qa[k][3] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)(g + 8) * DC + 16 * k + 8);
        }
    }
    float o[DQ / 8][4];
#pragma unroll
    for (int i = 0; i < DQ / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    Soft st = {-INFINITY, -INFINITY, 0.0f, 0.0f};

    for (int t = 0; t < nt; ++t) {
        if constexpr (MT == 1) {
            if constexpr (!Q8) wait_group<1>();  // tile t has landed (tile t + 1 may still be in flight)
            __syncwarp();
            if constexpr (Q8) load8(t + 2);  // in flight under tile t's arithmetic
        } else {
            if constexpr (!Q8) wait_group<0>();  // tile t has landed (it is the only group in flight)
            __syncthreads();   // every thread's pieces are visible, and every warp is past the last read of stage (t + 1) & 1
            if constexpr (Q8) {
                load8(t + 1);  // stored after tile t's arithmetic
            } else {
                stage(t + 1, (t + 1) & 1);
                commit();
            }
        }
        const uint4* tile = smem + ((t & 1) * NWARP + w) * TILE_U4;
        // ---- this quarter's score chain: 8 k16 steps from zero
        float s[NBS][4];
#pragma unroll
        for (int k = 0; k < DQ / 16; ++k) {
#pragma unroll
            for (int n = 0; n < NBS; n += 2) {
                uint32_t b[4];
                ldmatrix4(b, tile + sw(8 * (n + (mt >> 1)) + (lane & 7), 2 * k + (mt & 1)));
                if (k == 0) {
                    mma0<T>(s[n], qa[k], b[0], b[1]);
                    mma0<T>(s[n + 1], qa[k], b[2], b[3]);
                } else {
                    mma<T>(s[n], qa[k], b[0], b[1]);
                    mma<T>(s[n + 1], qa[k], b[2], b[3]);
                }
            }
        }
        // ---- exchange the four quarters' partial scores of this head tile and add them in the fixed order 0, 1, 2, 3
        float4* mine = sx + (((t & 1) * MT + mi) * NWARP + w) * SX_F4;
#pragma unroll
        for (int n = 0; n < NBS; ++n) mine[n * 32 + lane] = make_float4(s[n][0], s[n][1], s[n][2], s[n][3]);
        if constexpr (MT == 1) {
            __syncthreads();
        } else {
            asm volatile("bar.sync %0, 128;\n" ::"r"(1 + mi) : "memory");  // the four warps of this head tile
        }
        const float4* all = sx + ((t & 1) * MT + mi) * NWARP * SX_F4;
        float sc[NBS][4];
#pragma unroll
        for (int n = 0; n < NBS; ++n) {
            const float4 a = all[0 * SX_F4 + n * 32 + lane], b = all[1 * SX_F4 + n * 32 + lane];
            const float4 c = all[2 * SX_F4 + n * 32 + lane], d = all[3 * SX_F4 + n * 32 + lane];
            sc[n][0] = __fadd_rn(__fadd_rn(__fadd_rn(a.x, b.x), c.x), d.x);
            sc[n][1] = __fadd_rn(__fadd_rn(__fadd_rn(a.y, b.y), c.y), d.y);
            sc[n][2] = __fadd_rn(__fadd_rn(__fadd_rn(a.z, b.z), c.z), d.z);
            sc[n][3] = __fadd_rn(__fadd_rn(__fadd_rn(a.w, b.w), c.w), d.w);
        }
        // ---- fold the tile into the online softmax (the mask is the row's own position; all 16 heads share it)
        // visible state of the two key columns (2 qd, 2 qd + 1) of each n8 block: dense, key index <= the row's position; sparse,
        // the entry is a real key (the ballot of its tile). All 16 heads of the tile share the mask.
        const int kb = cstart + t * KT + 2 * qd;
        const unsigned vm = (t & 1) ? vm1 : vm0;
        bool v0[NBS], v1[NBS];
#pragma unroll
        for (int n = 0; n < NBS; ++n) {
            if constexpr (SPARSE) {
                v0[n] = (vm >> (8 * n + 2 * qd)) & 1u;
                v1[n] = (vm >> (8 * n + 2 * qd + 1)) & 1u;
            } else {
                v0[n] = kb + 8 * n <= last;
                v1[n] = kb + 8 * n + 1 <= last;
            }
        }
        float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < NBS; ++n) {
            sc[n][0] = v0[n] ? __fmul_rn(sc[n][0], SC2) : -INFINITY;
            sc[n][1] = v1[n] ? __fmul_rn(sc[n][1], SC2) : -INFINITY;
            sc[n][2] = v0[n] ? __fmul_rn(sc[n][2], SC2) : -INFINITY;
            sc[n][3] = v1[n] ? __fmul_rn(sc[n][3], SC2) : -INFINITY;
            tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
            tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
        }
#pragma unroll
        for (int x = 1; x <= 2; x *= 2) {
            tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
            tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
        }
        // dense: tile t's first key is visible (t < nt), so tm0 and tm1 are finite. Sparse: a tile of holes leaves both at -inf; then
        // n0 = m0, nothing below is computed from them (masked columns skip the subtraction) and the state is unchanged
        const float n0 = fmaxf(st.m0, tm0), n1 = fmaxf(st.m1, tm1);
        const float a0 = st.m0 == -INFINITY ? 0.0f : ex2(__fsub_rn(st.m0, n0));
        const float a1 = st.m1 == -INFINITY ? 0.0f : ex2(__fsub_rn(st.m1, n1));
#pragma unroll
        for (int n = 0; n < NBS; ++n) {
            sc[n][0] = v0[n] ? ex2(__fsub_rn(sc[n][0], n0)) : 0.0f;
            sc[n][1] = v1[n] ? ex2(__fsub_rn(sc[n][1], n0)) : 0.0f;
            sc[n][2] = v0[n] ? ex2(__fsub_rn(sc[n][2], n1)) : 0.0f;
            sc[n][3] = v1[n] ? ex2(__fsub_rn(sc[n][3], n1)) : 0.0f;
        }
        uint32_t pa[NKS][4];
#pragma unroll
        for (int j = 0; j < NKS; ++j) {
            pa[j][0] = pack2<T>(sc[2 * j][0], sc[2 * j][1]);
            pa[j][1] = pack2<T>(sc[2 * j][2], sc[2 * j][3]);
            pa[j][2] = pack2<T>(sc[2 * j + 1][0], sc[2 * j + 1][1]);
            pa[j][3] = pack2<T>(sc[2 * j + 1][2], sc[2 * j + 1][3]);
        }
        // l sums the T-rounded probabilities the PV mma consumes, so the weights o is built from sum to l exactly
        float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
        for (int n = 0; n < NBS; ++n) {
            float x0, x1, y0, y1;
            unpack2<T>(pa[n >> 1][(n & 1) * 2], x0, x1);
            unpack2<T>(pa[n >> 1][(n & 1) * 2 + 1], y0, y1);
            s0 = __fadd_rn(s0, __fadd_rn(x0, x1));
            s1 = __fadd_rn(s1, __fadd_rn(y0, y1));
        }
        st.l0 = __fmaf_rn(st.l0, a0, s0);
        st.l1 = __fmaf_rn(st.l1, a1, s1);
        st.m0 = n0;
        st.m1 = n1;
#pragma unroll
        for (int i = 0; i < DQ / 8; ++i) {
            o[i][0] = __fmul_rn(o[i][0], a0);
            o[i][1] = __fmul_rn(o[i][1], a0);
            o[i][2] = __fmul_rn(o[i][2], a1);
            o[i][3] = __fmul_rn(o[i][3], a1);
        }
        // ---- PV over this quarter's output columns, keys in order
#pragma unroll
        for (int ks = 0; ks < NKS; ++ks) {
#pragma unroll
            for (int i = 0; i < DQ / 8; i += 2) {
                uint32_t b[4];
                ldmatrix4t(b, tile + sw(16 * ks + 8 * (mt & 1) + (lane & 7), i + (mt >> 1)));
                mma<T>(o[i], pa[ks], b[0], b[1]);
                mma<T>(o[i + 1], pa[ks], b[2], b[3]);
            }
        }
        if constexpr (MT == 1) {
            __syncwarp();  // every lane's ldmatrix of this stage is done before it is restaged
            if constexpr (Q8) {
                store8(t & 1);  // tile t + 2
            } else {
                stage(t + 2, t & 1);
                commit();
            }
        } else if constexpr (Q8) {
            store8((t + 1) & 1);  // tile t + 1: every warp was past its last read of that stage at this tile's barrier
        }
    }
    if constexpr (!Q8) wait_group<0>();
    // ---- write the chunk partial: o columns of this quarter, and (m, l) once
    const float l0 = quad_sum(st.l0), l1 = quad_sum(st.l1);
    const int h0 = hb * HT + g, h1 = h0 + 8;
    const int64_t p0 = ((int64_t)row * nh + h0) * nch_max + chunk, p1 = ((int64_t)row * nh + h1) * nch_max + chunk;
#pragma unroll
    for (int i = 0; i < DQ / 8; ++i) {
        const int d = w * DQ + 8 * i + 2 * qd;
        *reinterpret_cast<float2*>(part_o + p0 * DC + d) = make_float2(o[i][0], o[i][1]);
        *reinterpret_cast<float2*>(part_o + p1 * DC + d) = make_float2(o[i][2], o[i][3]);
    }
    if (w == 0 && qd == 0) {
        part_m[p0] = st.m0;
        part_l[p0] = l0;
        part_m[p1] = st.m1;
        part_l[p1] = l1;
    }
}

// (row, head): merges the row's chunk partials in chunk order, 4 columns per thread. Every thread folds the same (m, l)
// sequence, so a column's bits do not depend on how the columns are split. A padding row (position < 0) writes zeros.
template <typename T, int CH, bool SPARSE>
__global__ void __launch_bounds__(DC / 4)
mla_merge_kernel(const float* __restrict__ part_o, const float* __restrict__ part_m, const float* __restrict__ part_l,
                 const int* __restrict__ row_pos, int dense_limit, const int* __restrict__ sel_meta, T* __restrict__ out,
                 int nch_max, int nh) {
    const int head = blockIdx.x, row = blockIdx.y, d = threadIdx.x * 4;
    const int pos = row_pos[row];
    int nch;
    if constexpr (SPARSE) {
        if (pos < 0 || sel_meta[2 * row] == 0 || sel_meta[2 * row + 1] < 1) return;  // not a sparse row
        nch = min((sel_meta[2 * row + 1] - 1) / CH + 1, nch_max);
    } else {
        if (pos >= dense_limit) return;  // the sparse kernel serves this row
        nch = pos < 0 ? 0 : min(pos / CH + 1, nch_max);
    }
    const int64_t base = ((int64_t)row * nh + head) * nch_max;
    float m = -INFINITY, l = 0.0f;
    float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 2
    for (int c = 0; c < nch; ++c) {
        const float cm = part_m[base + c], cl = part_l[base + c];
        if (!(cl > 0.0f)) continue;  // a chunk with no real key (sparse holes); a dense chunk always has one
        const float4 co = *reinterpret_cast<const float4*>(part_o + (base + c) * DC + d);
        const float nm = fmaxf(m, cm);
        const float a = m == -INFINITY ? 0.0f : ex2(__fsub_rn(m, nm));
        const float b = ex2(__fsub_rn(cm, nm));
        o[0] = __fmaf_rn(co.x, b, __fmul_rn(o[0], a));
        o[1] = __fmaf_rn(co.y, b, __fmul_rn(o[1], a));
        o[2] = __fmaf_rn(co.z, b, __fmul_rn(o[2], a));
        o[3] = __fmaf_rn(co.w, b, __fmul_rn(o[3], a));
        l = __fmaf_rn(cl, b, __fmul_rn(l, a));
        m = nm;
    }
    T* dst = out + ((int64_t)row * nh + head) * DC + d;
    if (nch > 0 && l > 0.0f) {  // a row with no real key at all (every entry a hole) writes zeros
        *reinterpret_cast<uint32_t*>(dst) = pack2<T>(__fdiv_rn(o[0], l), __fdiv_rn(o[1], l));
        *reinterpret_cast<uint32_t*>(dst + 2) = pack2<T>(__fdiv_rn(o[2], l), __fdiv_rn(o[3], l));
    } else {
        *reinterpret_cast<uint32_t*>(dst) = 0u;
        *reinterpret_cast<uint32_t*>(dst + 2) = 0u;
    }
}

size_t align256(size_t x) { return (x + 255) & ~(size_t)255; }

template <int CH> size_t workspace_bytes_ch(int rows, int max_context) {
    const size_t nch = (size_t)((max_context + CH - 1) / CH > 0 ? (max_context + CH - 1) / CH : 1);
    const size_t pairs = (size_t)rows * NH * nch;
    return align256(pairs * DC * 4) + 2 * align256(pairs * 4);
}

template <typename T, int KT, int CH, int MT, bool SPARSE, bool Q8>
int attend_t(const void* q_lat, const void* cache, const int* table, int max_blocks, const int* row_slot,
             const int* row_pos, int dense_limit, const int* sel_idx, int sel_stride, const int* sel_meta, void* out,
             int rows, int max_context, void* workspace, size_t workspace_bytes, cudaStream_t s, int nh = NH) {
    // two stages of KT keys x 1 KiB, and per head tile two buffers of the four warps' partial scores
    constexpr int SMEM = 2 * NWARP * KT * PQ * 16 + 2 * MT * NWARP * (KT / 8) * 32 * 16;
    if (workspace_bytes < workspace_bytes_ch<CH>(rows, max_context) || !workspace) return (int)cudaErrorInvalidValue;
    static const cudaError_t configured = cudaFuncSetAttribute(mla_partial_kernel<T, KT, CH, MT, SPARSE, Q8>,
                                                                cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    if (configured != cudaSuccess) return (int)configured;
    const int nch = (max_context + CH - 1) / CH;
    if (nch > 65535) return (int)cudaErrorInvalidValue;  // grid.y limit: chunks past 16.7M tokens
    if (nh < HT || nh > NH || nh % (HT * MT) != 0) return (int)cudaErrorInvalidValue;
    const size_t pairs = (size_t)rows * nh * nch;
    float* po = (float*)workspace;
    float* pm = (float*)((char*)workspace + align256(pairs * DC * 4));
    float* pl = (float*)((char*)workspace + align256(pairs * DC * 4) + align256(pairs * 4));
    mla_partial_kernel<T, KT, CH, MT, SPARSE, Q8><<<dim3(rows * ((nh / HT) / MT), nch), 128 * MT, SMEM, s>>>(
        (const T*)q_lat, cache, table, max_blocks, row_slot, row_pos, dense_limit, sel_idx, sel_stride, sel_meta,
        nch, po, pm, pl, nh);
    int rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    mla_merge_kernel<T, CH, SPARSE><<<dim3(nh, rows), DC / 4, 0, s>>>(po, pm, pl, row_pos, dense_limit, sel_meta, (T*)out, nch, nh);
    return GB_CHECK_LAUNCH();
}

// `f(T{}, std::bool_constant<Q8>{})` for dtype (0 fp16, 1 bf16) and kv (0: the cache holds T, 1: the fp8 page of mla_kv8.cuh).
template <typename F> int by_format(int dtype, int kv, F&& f) {
    if (dtype == 1) return kv ? f(bf16{}, std::true_type{}) : f(bf16{}, std::false_type{});
    return kv ? f(f16{}, std::true_type{}) : f(f16{}, std::false_type{});
}

// default tiling: 32-key tiles, 256-key chunks (= one page), one 16-head tile per CTA
constexpr int DEF_KT = 32, DEF_CH = 256, DEF_MT = 1;

#define MLA_ARGS q_lat, cache, table, max_blocks, row_slot, row_pos, dense_limit, nullptr, 0, nullptr, out, rows, max_context, ws, wsb, s

template <typename T, bool Q8>
int attend_variant_t(int variant, const void* q_lat, const void* cache, const int* table, int max_blocks,
                     const int* row_slot, const int* row_pos, int dense_limit, void* out, int rows, int max_context,
                     void* ws, size_t wsb, cudaStream_t s) {
    switch (variant) {
        case 1: return attend_t<T, 32, 128, 1, false, Q8>(MLA_ARGS);
        case 2: return attend_t<T, 32, 512, 1, false, Q8>(MLA_ARGS);
        case 3: return attend_t<T, 16, 256, 1, false, Q8>(MLA_ARGS);
        case 4: return attend_t<T, 16, 128, 1, false, Q8>(MLA_ARGS);
        case 5: return attend_t<T, 32, 256, 2, false, Q8>(MLA_ARGS);
        case 6: return attend_t<T, 32, 512, 2, false, Q8>(MLA_ARGS);
        case 7: return attend_t<T, 32, 128, 2, false, Q8>(MLA_ARGS);
        case 8: return attend_t<T, 16, 256, 2, false, Q8>(MLA_ARGS);
        default: return attend_t<T, DEF_KT, DEF_CH, DEF_MT, false, Q8>(MLA_ARGS);
    }
}

#undef MLA_ARGS

size_t variant_workspace(int variant, int rows, int max_context) {
    switch (variant) {
        case 1: case 4: case 7: return workspace_bytes_ch<128>(rows, max_context);
        case 2: case 6: return workspace_bytes_ch<512>(rows, max_context);
        case 3: case 5: case 8: return workspace_bytes_ch<256>(rows, max_context);
        default: return workspace_bytes_ch<DEF_CH>(rows, max_context);
    }
}

}  // namespace

// ---------------------------------------------------------------- exported launchers (dtype: 0 = fp16, 1 = bf16; kv: 0 = the cache holds
// dtype values, T [n_blocks][256][512]; 1 = the fp8 page of mla_kv8.cuh, E4M3 [n_blocks][256][512] then fp32 [n_blocks][256] scales)

GB_EXPORT int gb_mla_latent_write(const void* lat, const int* row_slot, const int* row_pos, const int* block_table,
                                  int max_blocks, void* cache, int rows, int dtype, int kv, void* stream) {
    if (rows < 0 || max_blocks <= 0 || dtype < 0 || dtype > 1 || kv < 0 || kv > 1) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    cudaStream_t s = gb_stream(stream);
    if (kv == 0) {
        latent_write_kernel<<<rows, DC / 8, 0, s>>>((const uint4*)lat, row_slot, row_pos, block_table, max_blocks, (uint4*)cache);
    } else if (dtype == 1) {
        latent_write8_kernel<bf16><<<rows, DC / 8, 0, s>>>((const uint4*)lat, row_slot, row_pos, block_table, max_blocks, (uint8_t*)cache);
    } else {
        latent_write8_kernel<f16><<<rows, DC / 8, 0, s>>>((const uint4*)lat, row_slot, row_pos, block_table, max_blocks, (uint8_t*)cache);
    }
    return GB_CHECK_LAUNCH();
}

// q: T [rows][...] with x_stride elements between rows (head h at h * 256); w_uk: T [64][256][512]; out: T [rows][64][512].
GB_EXPORT int gb_mla_absorb(const void* q, int64_t q_row_stride, const void* w_uk, void* q_lat, int rows, int dtype,
                            void* stream) {
    if (rows < 0 || q_row_stride < (int64_t)NH * DN || dtype < 0 || dtype > 1) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    cudaStream_t s = gb_stream(stream);
    return dtype == 1 ? absorb_t<bf16>(q, q_row_stride, w_uk, q_lat, rows, NH, s) : absorb_t<f16>(q, q_row_stride, w_uk, q_lat, rows, NH, s);
}

// o_lat: T [rows][64][512]; w_uvt: T [64][512][256]; out: T [rows][64 * 256].
GB_EXPORT int gb_mla_expand(const void* o_lat, const void* w_uvt, void* out, int rows, int dtype, void* stream) {
    if (rows < 0 || dtype < 0 || dtype > 1) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    cudaStream_t s = gb_stream(stream);
    return dtype == 1 ? expand_t<bf16>(o_lat, w_uvt, out, rows, NH, s) : expand_t<f16>(o_lat, w_uvt, out, rows, NH, s);
}

// fp32 partials o [rows][64][nch][512], m and l [rows][64][nch] with nch = ceil(max_context / 256).
GB_EXPORT size_t gb_mla_attend_workspace_bytes(int rows, int max_context) {
    return workspace_bytes_ch<DEF_CH>(rows, max_context);
}
GB_EXPORT size_t gb_mla_attend_workspace_bytes_variant(int variant, int rows, int max_context) {
    return variant_workspace(variant, rows, max_context);
}

// q_lat, out: T [rows][64][512]; cache: kv's layout; row_slot, row_pos: i32 [rows] (device). Row r attends to
// keys 0..=row_pos[r] of slot row_slot[r] (already written); a negative position is a padding row (output zeros). Every
// position must be < max_context, which sizes the grid and the workspace; rows <= 128 (a verify window or several
// streams' rows). Rows with row_pos >= dense_limit are left untouched (the sparse kernel serves them; INT_MAX = all rows are
// dense). `variant` selects a tuning tile/chunk shape (0 = the default; the others change bits, tuning only).
GB_EXPORT int gb_mla_attend_variant(const void* q_lat, const void* cache, const int* block_table, int max_blocks,
                                    const int* row_slot, const int* row_pos, int dense_limit, void* out, int rows,
                                    int max_context, void* workspace, size_t workspace_bytes, int dtype, int kv, int variant,
                                    void* stream) {
    if (rows < 1 || rows > MAX_ROWS || max_context < 1 || max_blocks <= 0 || dtype < 0 || dtype > 1 || kv < 0 || kv > 1)
        return (int)cudaErrorInvalidValue;
    cudaStream_t s = gb_stream(stream);
    return by_format(dtype, kv, [&](auto t, auto q) {
        return attend_variant_t<decltype(t), decltype(q)::value>(variant, q_lat, cache, block_table, max_blocks, row_slot, row_pos,
                                                                 dense_limit, out, rows, max_context, workspace, workspace_bytes, s);
    });
}

// Sparse (DSA) regime: row r attends to the keys at positions sel_idx[r * sel_stride + 0 .. sel_meta[2 r + 1]) of its slot
// (ascending, absolute; the layout `dsa::Selection` produces), taken in list order in chunks of 256 entries merged in chunk
// order. Only rows with sel_meta[2 r] == 1 are served; every other row (dense, padding) is left untouched. max_sel bounds the
// list length (2051 for GLM-5.3-Flash) and sizes the grid and the workspace.
GB_EXPORT int gb_mla_attend_sparse(const void* q_lat, const void* cache, const int* block_table, int max_blocks,
                                   const int* row_slot, const int* row_pos, const int* sel_idx, int sel_stride,
                                   const int* sel_meta, void* out, int rows, int max_sel, void* workspace,
                                   size_t workspace_bytes, int dtype, int kv, void* stream) {
    if (rows < 1 || rows > MAX_ROWS || max_sel < 1 || sel_stride < max_sel || max_blocks <= 0 || dtype < 0 || dtype > 1 || kv < 0 ||
        kv > 1)
        return (int)cudaErrorInvalidValue;
    cudaStream_t s = gb_stream(stream);
    return by_format(dtype, kv, [&](auto t, auto q) {
        return attend_t<decltype(t), DEF_KT, DEF_CH, 1, true, decltype(q)::value>(q_lat, cache, block_table, max_blocks, row_slot, row_pos,
                                                                                  0, sel_idx, sel_stride, sel_meta, out, rows, max_sel,
                                                                                  workspace, workspace_bytes, s);
    });
}

// ---------------------------------------------------------------- a rank's heads (lane E, the TP head split)
// The same kernels on heads [h0, h0 + nh) of a layer: q, w_uk and w_uvt point at head h0 (q's rows keep q_row_stride; q_lat, o_lat and out
// hold the nh heads only). Every head's arithmetic is the whole layer's (a block or a 16-head tile computes a head from its own operands, the
// partials and the merge are per head), so each head's output is the 64-head call's bit for bit (crates/cuda/tests/mla_split.rs). nh: a
// multiple of 16, at most 64.
GB_EXPORT int gb_mla_absorb_nh(const void* q, int64_t q_row_stride, const void* w_uk, void* q_lat, int rows, int dtype, int nh,
                               void* stream) {
    if (rows < 0 || nh < HT || nh > NH || nh % HT || q_row_stride < (int64_t)nh * DN || dtype < 0 || dtype > 1)
        return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    cudaStream_t s = gb_stream(stream);
    return dtype == 1 ? absorb_t<bf16>(q, q_row_stride, w_uk, q_lat, rows, nh, s) : absorb_t<f16>(q, q_row_stride, w_uk, q_lat, rows, nh, s);
}

GB_EXPORT int gb_mla_expand_nh(const void* o_lat, const void* w_uvt, void* out, int rows, int dtype, int nh, void* stream) {
    if (rows < 0 || nh < HT || nh > NH || nh % HT || dtype < 0 || dtype > 1) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    cudaStream_t s = gb_stream(stream);
    return dtype == 1 ? expand_t<bf16>(o_lat, w_uvt, out, rows, nh, s) : expand_t<f16>(o_lat, w_uvt, out, rows, nh, s);
}

// The default tiling (variant 0) and the sparse regime of gb_mla_attend_variant / gb_mla_attend_sparse on nh heads.
GB_EXPORT int gb_mla_attend_nh(const void* q_lat, const void* cache, const int* block_table, int max_blocks, const int* row_slot,
                               const int* row_pos, int dense_limit, void* out, int rows, int max_context, void* workspace,
                               size_t workspace_bytes, int dtype, int kv, int nh, void* stream) {
    if (rows < 1 || rows > MAX_ROWS || max_context < 1 || max_blocks <= 0 || dtype < 0 || dtype > 1 || kv < 0 || kv > 1)
        return (int)cudaErrorInvalidValue;
    cudaStream_t s = gb_stream(stream);
    return by_format(dtype, kv, [&](auto t, auto q) {
        return attend_t<decltype(t), DEF_KT, DEF_CH, DEF_MT, false, decltype(q)::value>(q_lat, cache, block_table, max_blocks, row_slot,
                                                                                        row_pos, dense_limit, nullptr, 0, nullptr, out,
                                                                                        rows, max_context, workspace, workspace_bytes,
                                                                                        s, nh);
    });
}

GB_EXPORT int gb_mla_attend_sparse_nh(const void* q_lat, const void* cache, const int* block_table, int max_blocks,
                                      const int* row_slot, const int* row_pos, const int* sel_idx, int sel_stride,
                                      const int* sel_meta, void* out, int rows, int max_sel, void* workspace,
                                      size_t workspace_bytes, int dtype, int kv, int nh, void* stream) {
    if (rows < 1 || rows > MAX_ROWS || max_sel < 1 || sel_stride < max_sel || max_blocks <= 0 || dtype < 0 || dtype > 1 || kv < 0 ||
        kv > 1)
        return (int)cudaErrorInvalidValue;
    cudaStream_t s = gb_stream(stream);
    return by_format(dtype, kv, [&](auto t, auto q) {
        return attend_t<decltype(t), DEF_KT, DEF_CH, 1, true, decltype(q)::value>(q_lat, cache, block_table, max_blocks, row_slot, row_pos,
                                                                                  0, sel_idx, sel_stride, sel_meta, out, rows, max_sel,
                                                                                  workspace, workspace_bytes, s, nh);
    });
}
