// HCG: the multi-HyperConnection pre-mix for large row counts (prefill chunks) as two GEMMs. Original REXL3 code,
// Victor Cruz. Same math as HCX (hcx.cu) / gb_hc_pre:
//   normed[t][s][d] = x[t][s][d] * rms[t][s] * (1 + nw[s * H + d]),   rms = rsqrt(mean_d x^2 + eps)
//   low[t]  = silu(normed[t] . down^T / hc)          (down [rank][hc * H])
//   inj[t]  = 2 sigmoid(normed[t] . inject^T / hc)    (inject [hc][hc * H], optional)
//   y[t][d] = (1 / hc) sum_s sigmoid((low[t] . up^T)[s * H + d]) * normed[t][s][d]     (up [hc * H][rank])
// The two products run on cuBLAS tensor cores with fp16 operands (normed and low are O(1) after the norm) and fp32
// accumulation; the final mix recomputes normed in fp32 from the streams.
// Weight layout contract: every operand is read exactly at its declared stride, never past its own buffer. down is
// [rank][hc*H] and up is [hc*H][rank], both at row stride `rank`; the inject weights are their own [hc][hc*H] buffer
// read by the separate inject GEMMs, never trailing columns of down/up. The low-row scratch (L, low hi+lo) is stride
// `rank` too. Nothing reads past the end of a tensor: a caller that wants spare columns must size its buffers for them.
#include "common.cuh"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cstdio>

namespace hcg {

__device__ __forceinline__ float block_sum(float v, float* red) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = 0.f;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) t += red[w];
    __syncthreads();
    return t;
}

// one block a row: per stream the rms, and normed as an fp16 hi+lo split for the GEMMs. The split is the error-free
// Dekker pairing of the fp32 value: hi carries the top 11 mantissa bits, lo the next 11, so hi+lo reconstructs the fp32
// value to ~22 bits. The tensor-core GEMMs (fp16 operands, fp32 accumulate) are then exact in the products; the only
// remaining error is the ~2^-22 residual of the split, not the ~2^-11 of a single fp16 rounding.
__global__ void __launch_bounds__(256) norm_kernel(const float* __restrict__ x, const half* __restrict__ nw,
                                                   half* __restrict__ normed_hi, half* __restrict__ normed_lo,
                                                   float* __restrict__ rms, int H, int hc, float eps) {
    __shared__ float red[8];
    const int t = blockIdx.x;
    for (int s = 0; s < hc; ++s) {
        const float* xr = x + ((size_t)t * hc + s) * H;
        float a = 0.f;
        for (int d = threadIdx.x * 4; d < H; d += blockDim.x * 4) {
            const float4 v = *reinterpret_cast<const float4*>(xr + d);
            a += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        const float r = rsqrtf(block_sum(a, red) / (float)H + eps);
        if (threadIdx.x == 0) rms[t * hc + s] = r;
        half* hi = normed_hi + ((size_t)t * hc + s) * H;
        half* lo = normed_lo + ((size_t)t * hc + s) * H;
        const half* w = nw + (size_t)s * H;
        for (int d = threadIdx.x * 4; d < H; d += blockDim.x * 4) {
            const float4 v = *reinterpret_cast<const float4*>(xr + d);
            const float2 w01 = __half22float2(*reinterpret_cast<const half2*>(w + d));
            const float2 w23 = __half22float2(*reinterpret_cast<const half2*>(w + d + 2));
            const float n0 = v.x * r * (1.f + w01.x), n1 = v.y * r * (1.f + w01.y);
            const float n2 = v.z * r * (1.f + w23.x), n3 = v.w * r * (1.f + w23.y);
            const float h0 = __half2float(__float2half_rn(n0)), h1 = __half2float(__float2half_rn(n1));
            const float h2 = __half2float(__float2half_rn(n2)), h3 = __half2float(__float2half_rn(n3));
            *reinterpret_cast<half2*>(hi + d) = __floats2half2_rn(h0, h1);
            *reinterpret_cast<half2*>(hi + d + 2) = __floats2half2_rn(h2, h3);
            *reinterpret_cast<half2*>(lo + d) = __floats2half2_rn(n0 - h0, n1 - h1);
            *reinterpret_cast<half2*>(lo + d + 2) = __floats2half2_rn(n2 - h2, n3 - h3);
        }
    }
}

// low (fp16 hi+lo split) = silu(L / hc); inj = 2 sigmoid(Ij / hc). `rank` is both the L/low row stride and the width
// of the low columns (the inject columns are the separate Ij block, not trailing low columns).
__global__ void low_kernel(const float* __restrict__ L, const float* __restrict__ Ij, half* __restrict__ low_hi,
                           half* __restrict__ low_lo, float* __restrict__ inj, int T, int rank, int hc) {
    const float inv = 1.f / (float)hc;
    const size_t nl = (size_t)T * rank, ni = Ij ? (size_t)T * hc : 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nl + ni; i += (size_t)gridDim.x * blockDim.x) {
        if (i < nl) {
            const float v = L[i] * inv;
            const float sv = v / (1.f + __expf(-v));
            const float h = __half2float(__float2half_rn(sv));
            low_hi[i] = __float2half_rn(sv);
            low_lo[i] = __float2half_rn(sv - h);
        } else {
            const size_t j = i - nl;
            inj[j] = 2.f / (1.f + __expf(-Ij[j] * inv));
        }
    }
}

// y[t][d] = bf16((1 / hc) sum_s sigmoid(G[t][s * H + d]) * x[t][s][d] * rms[t][s] * (1 + nw[s * H + d]))
__global__ void __launch_bounds__(256) mix_kernel(const float* __restrict__ x, const float* __restrict__ G,
                                                  const float* __restrict__ rms, const half* __restrict__ nw,
                                                  __nv_bfloat16* __restrict__ y, int H, int hc) {
    const int t = blockIdx.x;
    const float inv = 1.f / (float)hc;
    for (int d = threadIdx.x; d < H; d += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < hc; ++s) {
            const size_t i = ((size_t)t * hc + s) * H + d;
            const float g = 1.f / (1.f + __expf(-G[i]));
            acc += g * x[i] * rms[t * hc + s] * (1.f + __half2float(nw[(size_t)s * H + d]));
        }
        y[(size_t)t * H + d] = __float2bfloat16_rn(acc * inv);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// up + mix in one kernel (the Flash-Next shape: hc = 4, rank % 32 == 0, H % 32 == 0).
//
//   G[t][s * H + d] = (chain over k of low_hi[t][k] * up[s * H + d][k]) + (chain over k of low_lo[t][k] * up[s * H + d][k])
//   y[t][d]         = bf16(sum_s sigmoid(G) * x * rms * (1 + nw))        in stream order, exactly mix_kernel's expression
//
// The two products of the slab loop are plain sequential m16n8k16 chains over k from +0.0 (no split-K at K = rank; every tile and
// every GEMM M of every cuBLAS algorithm measured gives these bits), the lo GEMM's beta = 1 epilogue adds the hi sum once
// (fl(lo + hi), commutative), and mix_kernel's per-stream terms follow in the same order: fusing them moves where G is held (registers
// and shared memory instead of 168 MB of fp32 round trips through DRAM), never what it is. A CTA is 64 rows by 32 d of ALL four streams:
// warp w owns stream w's 32 columns (the up rows w * H + d0 .. + 31), both chains, so the epilogue sums the four streams of a (row, d)
// from shared memory in stream order.
constexpr int kUmRows = 64, kUmD = 32, kUmKB = 32, kUmPad = kUmKB + 8, kUmThreads = 128;
constexpr size_t kUmStageHalves = (size_t)(2 * kUmRows + 4 * kUmD) * kUmPad;
constexpr size_t kUmSmem = 2 * kUmStageHalves * sizeof(half);

__device__ __forceinline__ void um_cp16(void* smem, const void* g) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(g));
}
__device__ __forceinline__ void um_ldsm4(uint32_t (&r)[4], const half* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void um_mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// grid (H / 32, padded / 64), block 128, dynamic smem kUmSmem
__global__ void __launch_bounds__(kUmThreads, 2)
    up_mix_kernel(const half* __restrict__ low_hi, const half* __restrict__ low_lo, const half* __restrict__ up_w,
                  const float* __restrict__ x, const float* __restrict__ rms, const half* __restrict__ nw,
                  __nv_bfloat16* __restrict__ y, int T, int H, int rank) {
    extern __shared__ __align__(16) unsigned char um_smem[];
    half* sm = reinterpret_cast<half*>(um_smem);
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t4 = lane & 3;
    const int d0 = blockIdx.x * kUmD, r0 = blockIdx.y * kUmRows;
    const int nk = rank / kUmKB;
    auto stage = [&](int st) { return sm + (size_t)st * kUmStageHalves; };
    auto load_stage = [&](int st, int c) {
        half* base = stage(st);
        const int k0 = c * kUmKB;
        constexpr int kChunks = kUmKB / 8;  // 16-byte pieces a row of the tile
        for (int i = tid; i < (2 * kUmRows + 4 * kUmD) * kChunks; i += kUmThreads) {
            const int row = i / kChunks, ch = i % kChunks;
            const half* src;
            if (row < kUmRows) src = low_hi + (size_t)(r0 + row) * rank + k0 + ch * 8;
            else if (row < 2 * kUmRows) src = low_lo + (size_t)(r0 + row - kUmRows) * rank + k0 + ch * 8;
            else {
                const int n = row - 2 * kUmRows;  // 0..127: stream n / 32, column n % 32
                src = up_w + (size_t)((n / kUmD) * H + d0 + n % kUmD) * rank + k0 + ch * 8;
            }
            um_cp16(base + (size_t)row * kUmPad + ch * 8, src);
        }
    };
    float acc_hi[4][4][4], acc_lo[4][4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc_hi[a][b][e] = acc_lo[a][b][e] = 0.f;

    load_stage(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);
    for (int c = 0; c < nk; ++c) {
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        if (c + 1 < nk) load_stage((c + 1) & 1, c + 1);
        asm volatile("cp.async.commit_group;\n" ::);
        const half* ahi = stage(c & 1);
        const half* alo = ahi + (size_t)kUmRows * kUmPad;
        const half* bt = alo + (size_t)kUmRows * kUmPad + (size_t)w * kUmD * kUmPad;
#pragma unroll
        for (int kk = 0; kk < kUmKB / 16; ++kk) {
            uint32_t bf[4][2];
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                uint32_t r[4];
                um_ldsm4(r, bt + (size_t)(16 * p + (lane & 7) + ((lane >> 4) << 3)) * kUmPad + kk * 16 + ((lane >> 3) & 1) * 8);
                bf[2 * p][0] = r[0]; bf[2 * p][1] = r[1]; bf[2 * p + 1][0] = r[2]; bf[2 * p + 1][1] = r[3];
            }
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                uint32_t ah[4], al[4];
                const size_t off = (size_t)(mt * 16 + (lane & 15)) * kUmPad + kk * 16 + (lane >> 4) * 8;
                um_ldsm4(ah, ahi + off);
                um_ldsm4(al, alo + off);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    um_mma(acc_hi[mt][nt], ah, bf[nt][0], bf[nt][1]);
                    um_mma(acc_lo[mt][nt], al, bf[nt][0], bf[nt][1]);
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    // G = hi chain + lo chain, per stream into shared memory [stream][row][d], row stride 33 floats
    float* gs = reinterpret_cast<float*>(um_smem);
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = mt * 16 + g + (e >> 1) * 8, col = nt * 8 + 2 * t4 + (e & 1);
                gs[((size_t)w * kUmRows + row) * 33 + col] = acc_lo[mt][nt][e] + acc_hi[mt][nt][e];
            }
    __syncthreads();
    // epilogue: thread = (row tid / 2, 16 d from (tid & 1) * 16); the sum over the four streams in stream order (mix_kernel's expression)
    const int row = tid >> 1, dh = (tid & 1) * 16;
    const int t = r0 + row;
    if (t >= T) return;
    const float inv = 1.f / 4.f;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const float rs = rms[(size_t)t * 4 + s];
        const float* xr = x + ((size_t)t * 4 + s) * H + d0 + dh;
        const half* nwr = nw + (size_t)s * H + d0 + dh;
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            const float4 xv = *reinterpret_cast<const float4*>(xr + i);
            const float xs[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float gv = 1.f / (1.f + __expf(-gs[((size_t)s * kUmRows + row) * 33 + dh + i + j]));
                acc[i + j] += gv * xs[j] * rs * (1.f + __half2float(nwr[i + j]));
            }
        }
    }
    __nv_bfloat16* yr = y + (size_t)t * H + d0 + dh;
#pragma unroll
    for (int i = 0; i < 16; ++i) yr[i] = __float2bfloat16_rn(acc[i] * inv);
}

// One cuBLAS handle per host thread, created on first use and never destroyed (destroying it at thread exit can run
// after the CUDA context is gone). The static handle it replaces was a cross-thread cublasSetStream race if the server
// ever overlaps prefill and decode threads (same fix as exl3_linear.cu's cublas_handle).
cublasHandle_t handle() {
    static thread_local cublasHandle_t h = nullptr;
    if (!h && cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) h = nullptr;
    if (h) cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    return h;
}

// row-major C [T][N] fp32 = beta*C + alpha*A [T][K] fp16 . B^T, B row-major [N][K] fp16
int gemm_abt(cublasHandle_t h, const half* A, const half* B, float* C, int T, int N, int K, float beta) {
    const float one = 1.f;
    const cublasStatus_t s = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, B, CUDA_R_16F, K, A, CUDA_R_16F, K,
                                          &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return s == CUBLAS_STATUS_SUCCESS ? 0 : (int)cudaErrorUnknown;
}


// ---------------------------------------------------------------------------------------------------------------------
// The Flash-Next shape (H = 2560, hc = 4, rank = 320) as a few large cublasLt calls instead of 64-row slab GEMMs.
//
// WHY THIS IS THE SAME NUMBER. cublasGemmEx at M = 64 (the slab) is split-K for the two long products: the down product (K = 10240,
// N = 320) is cublasLt algo 67 with SPLIT-K 4 and the compute-type reduction (nvjet_sm121_..._64x32x64_splitK + splitKreduce),
// the inject product (K = 10240, N = 4) is algo 23 with SPLIT-K 20 and the output-type reduction, and the up product (K = 320,
// N = 10240) has no split. A row's output is therefore one chain of m16n8k16 MMAs per K split followed by the library's reduction
// of the splits, whatever the tile and whatever M is: the same algorithm family and split count, with a different tile, bit for bit
// equal at every M (MEASURED on GB10 with CUDA 13 cuBLAS, hcgtest: slab loop vs M = 4096 call, 1,310,720 / 16,384 / 41,943,040
// fp32 outputs of down / inject / up, 0 differing; a 64x64 algo-67 tile or a different split count is NOT equal). Here the algorithm
// is ours, fixed per product (id, tile, stages, custom option, split-K, reduction scheme), checked by cublasLtMatmulAlgoCheck
// before every use; anything that fails the check (or any other shape) takes the slab loop, which stays as the reference
// (gb_hcg_pre_ref) the tests compare against.
struct AlgoSpec {
    int id;
    uint32_t tile, stages, custom, splitk, red;
};
constexpr AlgoSpec kDownAlgo = {67, 15, 35, 90, 4, CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE};
constexpr AlgoSpec kInjectAlgo = {23, 5, 20, 0, 20, CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE};
constexpr int kLtH = 2560, kLtHc = 4, kLtRank = 320;
bool lt_shape(int H, int hc, int rank) { return H == kLtH && hc == kLtHc && rank == kLtRank; }
// workspace the split-K partials need for `padded` rows: down 4 x [rows][rank] f32, inject 20 x [rows][hc] f32 (one stream, one at a time)
size_t lt_ws_bytes(int padded, int hc, int rank) {
    const size_t down = (size_t)kDownAlgo.splitk * padded * rank * 4, inj = (size_t)kInjectAlgo.splitk * padded * hc * 4;
    return down > inj ? down : inj;
}

cublasLtHandle_t lt_handle() {
    static thread_local cublasLtHandle_t h = nullptr;
    if (!h && cublasLtCreate(&h) != CUBLAS_STATUS_SUCCESS) h = nullptr;
    return h;
}

// row-major C [T][N] f32 = beta * C + A [T][K] f16 . B^T (B row-major [N][K] f16) under the fixed algorithm; returns -1 when the
// algorithm does not take this problem (the caller falls back), 0 on success, else a cudaError_t.
int lt_gemm_abt(cublasLtHandle_t lt, const AlgoSpec& sp, const half* A, const half* B, float* C, int T, int N, int K, float beta,
                void* ws, size_t ws_bytes, cudaStream_t st) {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    int rc = -1;
    do {
        if (cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) { rc = (int)cudaErrorUnknown; break; }
        const cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
        if (cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)) != CUBLAS_STATUS_SUCCESS) { rc = (int)cudaErrorUnknown; break; }
        // cuBLAS column-major view of the row-major call: A' = B [K x N] (op T), B' = A [K x T], C' = C [N x T]
        if (cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, K, N, K) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, K, T, K) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, N, T, N) != CUBLAS_STATUS_SUCCESS) { rc = (int)cudaErrorUnknown; break; }
        cublasLtMatmulAlgo_t algo;
        if (cublasLtMatmulAlgoInit(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F, sp.id, &algo) !=
            CUBLAS_STATUS_SUCCESS) break;
        const uint32_t swizzle = 0;
        auto set = [&](cublasLtMatmulAlgoConfigAttributes_t a, const uint32_t& v) {
            return cublasLtMatmulAlgoConfigSetAttribute(&algo, a, &v, sizeof(v)) == CUBLAS_STATUS_SUCCESS;
        };
        if (!set(CUBLASLT_ALGO_CONFIG_TILE_ID, sp.tile) || !set(CUBLASLT_ALGO_CONFIG_STAGES_ID, sp.stages) ||
            !set(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, sp.custom) || !set(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, sp.splitk) ||
            !set(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, sp.red) || !set(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, swizzle)) break;
        cublasLtMatmulHeuristicResult_t chk;
        if (cublasLtMatmulAlgoCheck(lt, desc, la, lb, lc, lc, &algo, &chk) != CUBLAS_STATUS_SUCCESS || chk.workspaceSize > ws_bytes) break;
        const float one = 1.f;
        if (cublasLtMatmul(lt, desc, &one, B, la, A, lb, &beta, C, lc, C, lc, &algo, ws, ws_bytes, st) != CUBLAS_STATUS_SUCCESS) {
            rc = (int)cudaErrorUnknown;
            break;
        }
        rc = 0;
    } while (false);
    if (lc) cublasLtMatrixLayoutDestroy(lc);
    if (lb) cublasLtMatrixLayoutDestroy(lb);
    if (la) cublasLtMatrixLayoutDestroy(la);
    if (desc) cublasLtMatmulDescDestroy(desc);
    return rc;
}

}  // namespace hcg

using namespace hcg;

// Scratch bytes for up to `rows` rows (normed hi+lo, L, Ij, low hi+lo, G, rms; each 256-aligned). `rank` is the low
// row stride; nothing here depends on any other layout choice.
GB_EXPORT size_t gb_hcg_scratch_bytes(int rows, int H, int hc, int rank) {
    auto al = [](size_t v) { return (v + 255) & ~(size_t)255; };
    const size_t K = (size_t)hc * H;
    const size_t padded = (size_t)((rows + 63) / 64) * 64;
    return 2 * al(padded * K * 2) + al(padded * rank * 4) + al(padded * hc * 4) + 2 * al(padded * rank * 2) +
           al(padded * K * 4) + 2 * al(padded * hc * 4) + (lt_shape(H, hc, rank) ? al(lt_ws_bytes((int)padded, hc, rank)) : 0);
}

// y [T][H] bf16 (+ inj [T][hc] f32 when inject_w is non-null) from streams [T][hc][H] f32. scratch holds
// gb_hcg_scratch_bytes(T, H, hc, rank). H % 4 == 0, rank >= 1. The weight operands are read exactly at their declared
// stride `rank` (down [rank][hc*H], up [hc*H][rank], inject [hc][hc*H]); no read extends past a weight buffer.
static int hcg_pre(const void* streams, const void* norm_w, const void* down_w, const void* up_w,
                   const void* inject_w, void* y_out, float* inj_out, void* scratch, int H, int hc, int rank,
                   float eps, int T, void* stream, bool fast) {
    if (T < 1 || H % 4 || hc < 1 || rank < 1 || !scratch || (inject_w != nullptr) != (inj_out != nullptr))
        return (int)cudaErrorInvalidValue;
    cublasHandle_t h = handle();
    if (!h) return (int)cudaErrorUnknown;
    cudaStream_t st = gb_stream(stream);
    if (cublasSetStream(h, st) != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorUnknown;
    auto al = [](size_t v) { return (v + 255) & ~(size_t)255; };
    const int K = hc * H;
    const int padded = ((T + 63) / 64) * 64;
    char* p = static_cast<char*>(scratch);
    half* normed_hi = (half*)p;     p += al((size_t)padded * K * 2);
    half* normed_lo = (half*)p;     p += al((size_t)padded * K * 2);
    float* L = (float*)p;           p += al((size_t)padded * rank * 4);
    float* Ij = (float*)p;          p += al((size_t)padded * hc * 4);
    half* low_hi = (half*)p;        p += al((size_t)padded * rank * 2);
    half* low_lo = (half*)p;        p += al((size_t)padded * rank * 2);
    float* G = (float*)p;           p += al((size_t)padded * K * 4);
    float* rms = (float*)p;         p += al((size_t)padded * hc * 4);
    // inj_pad occupies the trailing hc*4 block already counted by gb_hcg_scratch_bytes
    // (the second `2 * al(padded*hc*4)` term); make the reservation explicit.
    float* inj_pad = (float*)p;     p += al((size_t)padded * hc * 4);
    // split-K workspace of the large-call path (the shape the algorithms are fixed for; the slab loop needs none)
    void* lt_ws = p;
    const size_t lt_ws_n = lt_shape(H, hc, rank) ? lt_ws_bytes(padded, hc, rank) : 0;
    cublasLtHandle_t lt = (fast && lt_ws_n) ? lt_handle() : nullptr;

    // A fixed 64-row GEMM tile gives the same accumulation for a row in every decode window.
    // Initialize only the unused tail of the final tile; valid rows are overwritten by norm_kernel.
    if (padded != T) {
        if (cudaMemsetAsync(normed_hi + (size_t)T * K, 0, (size_t)(padded - T) * K * sizeof(half), st) != cudaSuccess)
            return (int)cudaErrorUnknown;
        if (cudaMemsetAsync(normed_lo + (size_t)T * K, 0, (size_t)(padded - T) * K * sizeof(half), st) != cudaSuccess)
            return (int)cudaErrorUnknown;
    }
    norm_kernel<<<T, 256, 0, st>>>((const float*)streams, (const half*)norm_w, normed_hi, normed_lo, rms, H, hc, eps);
    int rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    // down: one split-K call per half over every row when the large-call algorithm takes it, else the 64-row slab loop
    bool down_done = false;
    static bool warned = false;
    if (lt) {
        const int a = lt_gemm_abt(lt, kDownAlgo, normed_hi, (const half*)down_w, L, padded, rank, K, 0.f, lt_ws, lt_ws_n, st);
        if (a > 0) return a;
        if (a == 0) {
            const int b = lt_gemm_abt(lt, kDownAlgo, normed_lo, (const half*)down_w, L, padded, rank, K, 1.f, lt_ws, lt_ws_n, st);
            if (b) return b > 0 ? b : (int)cudaErrorUnknown;  // the first half ran: the second must too
            down_done = true;
        }
    }
    bool inj_done = !inject_w;
    if (!inj_done && lt) {
        const int a = lt_gemm_abt(lt, kInjectAlgo, normed_hi, (const half*)inject_w, Ij, padded, hc, K, 0.f, lt_ws, lt_ws_n, st);
        if (a > 0) return a;
        if (a == 0) {
            const int b = lt_gemm_abt(lt, kInjectAlgo, normed_lo, (const half*)inject_w, Ij, padded, hc, K, 1.f, lt_ws, lt_ws_n, st);
            if (b) return b > 0 ? b : (int)cudaErrorUnknown;
            inj_done = true;
        }
    }
    if (lt && (!down_done || !inj_done) && !warned) {
        warned = true;
        fprintf(stderr, "rexl3 level=warn hcg: the large-call split-K algorithm refused rows=%d (down %d, inject %d): slab loop\n", padded,
                (int)down_done, (int)inj_done);
    }
    if (!down_done || !inj_done) {
        for (int row = 0; row < padded; row += 64) {
            if (!down_done) {
                if ((rc = gemm_abt(h, normed_hi + (size_t)row * K, (const half*)down_w, L + (size_t)row * rank, 64, rank, K, 0.f))) return rc;
                if ((rc = gemm_abt(h, normed_lo + (size_t)row * K, (const half*)down_w, L + (size_t)row * rank, 64, rank, K, 1.f))) return rc;
            }
            if (!inj_done) {
                if ((rc = gemm_abt(h, normed_hi + (size_t)row * K, (const half*)inject_w,
                                    Ij + (size_t)row * hc, 64, hc, K, 0.f))) return rc;
                if ((rc = gemm_abt(h, normed_lo + (size_t)row * K, (const half*)inject_w,
                                    Ij + (size_t)row * hc, 64, hc, K, 1.f))) return rc;
            }
        }
    }
    const size_t n = (size_t)padded * rank + (inject_w ? (size_t)padded * hc : 0);
    low_kernel<<<(unsigned)((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, st>>>(
        L, inject_w ? Ij : nullptr, low_hi, low_lo, inject_w ? inj_pad : nullptr, padded, rank, hc);
    if ((rc = GB_CHECK_LAUNCH())) return rc;
    if (inject_w && cudaMemcpyAsync(inj_out, inj_pad, (size_t)T * hc * sizeof(float), cudaMemcpyDeviceToDevice, st) != cudaSuccess)
        return (int)cudaErrorUnknown;
    // up has no split (K = rank): a row is one chain of MMAs over K whatever the GEMM's M, so the slab only sets how long the fp32 G slab
    // stays in L2 between the hi pass, the lo pass and the mix that reads it. The large-call path takes 128-row slabs (MEASURED at 4096
    // rows, hi + lo of up: 2.18 ms against 3.36 ms for 64; the mix per slab: 7.36 ms per pre-mix against 7.69 ms with one mix launch).
    if (!lt) {  // the slab loop: the reference, and every shape the large-call algorithms are not fixed for
        for (int row = 0; row < padded; row += 64) {
            if ((rc = gemm_abt(h, low_hi + (size_t)row * rank, (const half*)up_w, G + (size_t)row * K, 64, K, rank, 0.f))) return rc;
            if ((rc = gemm_abt(h, low_lo + (size_t)row * rank, (const half*)up_w, G + (size_t)row * K, 64, K, rank, 1.f))) return rc;
        }
        mix_kernel<<<T, 256, 0, st>>>((const float*)streams, G, rms, (const half*)norm_w, (__nv_bfloat16*)y_out, H, hc);
    } else {
        // up + mix as one kernel: G never reaches DRAM
        static bool attr = false;
        if (!attr) {
            if (cudaFuncSetAttribute(up_mix_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kUmSmem) != cudaSuccess) return (int)cudaErrorUnknown;
            attr = true;
        }
        up_mix_kernel<<<dim3(H / kUmD, padded / kUmRows), kUmThreads, kUmSmem, st>>>(
            low_hi, low_lo, (const half*)up_w, (const float*)streams, rms, (const half*)norm_w, (__nv_bfloat16*)y_out, T, H, rank);
    }
    return GB_CHECK_LAUNCH();
}

// the large-call path where the shape allows it, the 64-row slab loop otherwise
GB_EXPORT int gb_hcg_pre(const void* streams, const void* norm_w, const void* down_w, const void* up_w,
                         const void* inject_w, void* y_out, float* inj_out, void* scratch, int H, int hc, int rank,
                         float eps, int T, void* stream) {
    return hcg_pre(streams, norm_w, down_w, up_w, inject_w, y_out, inj_out, scratch, H, hc, rank, eps, T, stream, true);
}

// the 64-row slab loop on every shape: the reference gb_hcg_pre is compared against bit for bit
GB_EXPORT int gb_hcg_pre_ref(const void* streams, const void* norm_w, const void* down_w, const void* up_w,
                             const void* inject_w, void* y_out, float* inj_out, void* scratch, int H, int hc, int rank,
                             float eps, int T, void* stream) {
    return hcg_pre(streams, norm_w, down_w, up_w, inject_w, y_out, inj_out, scratch, H, hc, rank, eps, T, stream, false);
}

// ------------------------------------------------------------------------------------------------ small linear GEMM
// y [rows][n] bf16 = x [rows][k] bf16 . w^T (w fp16 [n][k]), for prefill chunks: x widened to fp16 (exact for the
// O(1) mixed rows), one tensor-core GEMM with fp32 accumulation, rounded to bf16. scratch: rows * k * 2 + rows * n * 4.
namespace hcg {
__global__ void bf16_to_f16_kernel(const __nv_bfloat16* __restrict__ x, half* __restrict__ y, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        y[i] = __float2half_rn(__bfloat162float(x[i]));
}
__global__ void f32_to_bf16_kernel(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        y[i] = __float2bfloat16_rn(x[i]);
}
static unsigned grid_for(size_t n) { return (unsigned)((n + 255) / 256 < 8192 ? (n + 255) / 256 : 8192); }
}  // namespace hcg

GB_EXPORT size_t gb_lin16_scratch_bytes(int rows, int k, int n) {
    auto al = [](size_t v) { return (v + 255) & ~(size_t)255; };
    return al((size_t)rows * k * 2) + al((size_t)rows * n * 4);
}

GB_EXPORT int gb_lin16_gemm(const void* x, const void* w, void* y, int rows, int k, int n, void* scratch, void* stream) {
    if (rows < 1 || k < 1 || n < 1) return (int)cudaErrorInvalidValue;
    cublasHandle_t h = handle();
    if (!h) return (int)cudaErrorUnknown;
    cudaStream_t st = gb_stream(stream);
    if (cublasSetStream(h, st) != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorUnknown;
    auto al = [](size_t v) { return (v + 255) & ~(size_t)255; };
    half* x16 = static_cast<half*>(scratch);
    float* yf = reinterpret_cast<float*>(static_cast<char*>(scratch) + al((size_t)rows * k * 2));
    const size_t nx = (size_t)rows * k, ny = (size_t)rows * n;
    bf16_to_f16_kernel<<<grid_for(nx), 256, 0, st>>>((const __nv_bfloat16*)x, x16, nx);
    int rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    if ((rc = gemm_abt(h, x16, (const half*)w, yf, rows, n, k, 0.f))) return rc;
    f32_to_bf16_kernel<<<grid_for(ny), 256, 0, st>>>(yf, (__nv_bfloat16*)y, ny);
    return GB_CHECK_LAUNCH();
}
