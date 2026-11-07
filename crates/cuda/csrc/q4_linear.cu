// The affine 4-bit linear of the GLM DFlash2 drafter (`src/q4_linear.rs`): y [m, n] bf16 = x [m, k] bf16 . W^T, W stored as
// 4-bit codes q with a bf16 scale s and bias b per 64 inputs of a row (w ~ s q + b, the MLX affine layout), tiled for a
// bf16 m16n8k16 MMA. Ported from TensorFold's `qmm_group.cu` (tile 2, LaneTile<64, 16, 64, 1, 4, 8>), which served the
// same drafter; one wider instantiation (64 rows a block) takes the context absorbs of a prompt chunk.
//
// Per (row, 64-input group): acc = fma(xs, b, fma(sum_k q[k] x[k], s, acc)) with xs = sum_k x[k] (fp32, `gb_q4_gsum`).
// K slices are a function of the weight's shape only (`q4_split_k`), never of the row count, so a row's bits do not depend
// on how many rows the call holds. A slice's partial sums meet in its cluster's shared memory (slice 0 adds them in order).

#include "common.cuh"

#include <cooperative_groups.h>
#include <type_traits>

namespace gb_q4 {

constexpr int GS = 64;  // inputs a scale group

template <int BM, int BN, int WM, int WN, int STAGES>
struct Tile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;  // m16 tiles a warp
    static constexpr int NT = BN / WN / 8;   // n8 tiles a warp
    static constexpr int ROW = GS * 2;       // bytes of one input row a group
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;       // stage bytes: inputs,
    static constexpr int W = BN * GS / 2;    // weights,
    static constexpr int S = BN * 2;         // scales, biases (bf16),
    static constexpr int XS = BM * 4;        // and input sums (fp32)
    static constexpr int STAGE = X + W + 2 * S + XS;
    static constexpr int PARTIALS = MT * NT * 4 * THREADS * 4;
    static constexpr int SMEM = STAGES * STAGE > PARTIALS ? STAGES * STAGE : PARTIALS;
};

__device__ __forceinline__ uint32_t smem(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem(dst)), "l"(src));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem(dst)), "l"(src));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem(p)));
}
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma0(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    const float z = 0.0f;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(z));
}
template <int CHUNKS>
__device__ __forceinline__ int swz(int r, int c) {
    return c ^ (r % CHUNKS);
}
// Codes at bits s and s + 16 of w as two exact bf16 values (q + 128 - 128).
__device__ __forceinline__ uint32_t pairm(uint32_t w, int s, uint32_t mask) {
    const uint32_t t = ((w >> s) & mask) | 0x43004300u;
    uint32_t r;
#if __CUDA_ARCH__ >= 900
    asm("sub.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(t), "r"(0x43004300u));
#else
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x3F803F80u), "r"(0xC300C300u));
#endif
    return r;
}

template <int N>
using ic = std::integral_constant<int, N>;

// One block: BM rows x BN columns of the output over one K slice of `sk` (a cluster of sk blocks covers a column tile).
template <int BM, int BN, int WM, int WN, int STAGES>
__global__ void __launch_bounds__(WM * WN * 32) q4_kernel(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,
                                                         const uint32_t* __restrict__ wq, const __nv_bfloat16* __restrict__ scales,
                                                         const __nv_bfloat16* __restrict__ biases, __nv_bfloat16* __restrict__ out,
                                                         int M, int K, int N, int npad, int ldx, int rows_t, int sk) {
    using T = Tile<BM, BN, WM, WN, STAGES>;
    constexpr int I = T::MT, J = T::NT;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, cid = blockIdx.x / sk, slice = blockIdx.x % sk;
    const int tiles = (N + BN - 1) / BN;
    const int tile = cid / rows_t;
    const int m0 = cid % rows_t * BM, n0 = tile * BN, per = KG / sk, g0 = slice * per;
    const bool live = tile < tiles;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    const int rows = min(BM, M - m0);
    constexpr int TILE_BYTES = 64 * GS / 2;
    constexpr int XC = (BM * T::CHUNKS + T::THREADS - 1) / T::THREADS, WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    constexpr int SC = (2 * T::S / 16 + T::THREADS - 1) / T::THREADS;
    static_assert(BM <= T::THREADS, "one input sum a thread");
    const __nv_bfloat16* xsrc[XC];
    const unsigned char* wsrc[WC];
    const __nv_bfloat16* ssrc[SC];
    int xdst[XC], wdst[WC], sdst[SC];
    bool xok[XC], wok[WC], sok[SC];
#pragma unroll
    for (int j = 0; j < XC; ++j) {
        const int c = tid + j * T::THREADS, r = c / T::CHUNKS, ch = c % T::CHUNKS;
        xok[j] = c < BM * T::CHUNKS && r < rows;
        xsrc[j] = x + static_cast<size_t>(m0 + min(r, rows - 1)) * ldx + ch * 8;
        xdst[j] = r * T::ROW + swz<T::CHUNKS>(r, ch) * 16;
    }
#pragma unroll
    for (int j = 0; j < WC; ++j) {
        const int c = tid + j * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
        wok[j] = c < T::W / 16;
        wsrc[j] = reinterpret_cast<const unsigned char*>(wq) + static_cast<size_t>(n0 / 64 + t) * KG * TILE_BYTES + off * 16;
        wdst[j] = T::X + c * 16;
    }
#pragma unroll
    for (int j = 0; j < SC; ++j) {
        const int c = tid + j * T::THREADS, which = c / (T::S / 16), off = c % (T::S / 16);
        sok[j] = c < 2 * (T::S / 16);
        ssrc[j] = (which ? biases : scales) + n0 + off * 8;
        sdst[j] = T::X + T::W + which * T::S + off * 16;
    }
    const bool xsok = tid < rows;
    const float* xssrc = xs + static_cast<size_t>(m0 + min(tid, rows - 1)) * KG;
    auto load_x = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < XC; ++j)
            if (xok[j]) cp16(p + xdst[j], xsrc[j] + g * GS);
        if (xsok) cp4(p + T::X + T::W + 2 * T::S + tid * 4, xssrc + g);
    };
    auto load_w = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < WC; ++j)
            if (wok[j]) cp16(p + wdst[j], wsrc[j] + static_cast<size_t>(g) * TILE_BYTES);
#pragma unroll
        for (int j = 0; j < SC; ++j)
            if (sok[j]) cp16(p + sdst[j], ssrc[j] + static_cast<size_t>(g) * npad);
    };
    uint32_t mask;  // 0x000F000F, opaque to the compiler (kept in a register)
    asm volatile("mov.b32 %0, 0x000F000F;\n" : "=r"(mask));

    float acc[I][J][4];
#pragma unroll
    for (int i = 0; i < I; ++i)
#pragma unroll
        for (int j = 0; j < J; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    if (live) {
        // rows past M stay zero in every stage
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS)
            if (c / T::CHUNKS >= rows)
#pragma unroll
                for (int s = 0; s < STAGES; ++s)
                    *reinterpret_cast<uint4*>(stage(s) + c / T::CHUNKS * T::ROW + swz<T::CHUNKS>(c / T::CHUNKS, c % T::CHUNKS) * 16) =
                        uint4{};
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < per) {
                load_w(s, g0 + s);
                load_x(s, g0 + s);
            }
            commit();
        }
        auto steps = [&](auto live_c) {
            constexpr int LIVE = decltype(live_c)::value;
            for (int it = 0; it < per; ++it) {
                wait<STAGES - 2>();
                __syncthreads();
                const int next = it + STAGES - 1;
                if (next < per) {
                    load_x(next % STAGES, g0 + next);
                    load_w(next % STAGES, g0 + next);
                }
                commit();
                const unsigned char* p = stage(it % STAGES);
                const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
                const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W);
                const float* px = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
                uint32_t words[J][GS / 32];
#pragma unroll
                for (int j = 0; j < J; ++j)
#pragma unroll
                    for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * J + j) * 32 + lane) * (GS / 32) + v];
                float d[I][J][4];
#pragma unroll
                for (int kt = 0; kt < GS / 16; ++kt) {
                    uint32_t a[LIVE > 0 ? LIVE : 1][4];
#pragma unroll
                    for (int i = 0; i < LIVE; ++i) {
                        const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                        const int ch = kt * 2 + (lane >> 4);
                        ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
                    }
#pragma unroll
                    for (int j = 0; j < J; ++j) {
                        const uint32_t b0 = pairm(words[j][kt / 2], (kt & 1) * 8, mask);
                        const uint32_t b1 = pairm(words[j][kt / 2], (kt & 1) * 8 + 4, mask);
#pragma unroll
                        for (int i = 0; i < LIVE; ++i) {
                            if (kt == 0) mma0(d[i][j], a[i], b0, b1);
                            else mma(d[i][j], a[i], b0, b1);
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < J; ++j) {
                    const int col = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
                    const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + col);
                    const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + BN + col);
                    const float sv[2] = {__low2float(s2), __high2float(s2)};
                    const float bv[2] = {__low2float(b2), __high2float(b2)};
#pragma unroll
                    for (int i = 0; i < LIVE; ++i) {
                        const int row = wm * (BM / WM) + i * 16 + (lane >> 2);
                        const float xv[2] = {px[row], px[row + 8]};
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            acc[i][j][e] = __fmaf_rn(xv[e >> 1], bv[e & 1], __fmaf_rn(d[i][j][e], sv[e & 1], acc[i][j][e]));
                    }
                }
            }
        };
        // the warp's m16 tiles that hold rows below M (the others' MMAs are skipped: their rows are never stored)
        const int live_t = min(I, max(0, (M - m0 - wm * (BM / WM) + 15) / 16));
        if constexpr (I == 1) {
            steps(ic<1>());
        } else {
            if (live_t <= 1) steps(ic<1>());
            else if (live_t == 2) steps(ic<(I > 2 ? 2 : I)>());
            else steps(ic<I>());
        }
        wait<0>();
        __syncthreads();
    }
    auto emit = [&](int i, int j, int h, float v0, float v1) {
        const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
        const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
        if (row >= M || col >= N) return;
        __nv_bfloat16* dst = out + static_cast<size_t>(row) * N + col;
        if (col + 1 < N && (N & 1) == 0) {
            *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
        } else {
            dst[0] = __float2bfloat16_rn(v0);
            if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
        }
    };
    if (sk > 1) {
        auto cluster = cooperative_groups::this_cluster();
        float* mine = reinterpret_cast<float*>(buf);
        if (slice != 0) {
#pragma unroll
            for (int i = 0; i < I; ++i)
#pragma unroll
                for (int j = 0; j < J; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) mine[((i * J + j) * 4 + e) * T::THREADS + tid] = acc[i][j][e];
        }
        cluster.sync();
        if (slice == 0 && live) {  // slice 0 adds its tile's slices in order
            for (int peer = 1; peer < sk; ++peer) {
                const float* theirs = cluster.map_shared_rank(mine, peer);
#pragma unroll
                for (int i = 0; i < I; ++i)
#pragma unroll
                    for (int j = 0; j < J; ++j)
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc[i][j][e] = acc[i][j][e] + theirs[((i * J + j) * 4 + e) * T::THREADS + tid];
            }
        }
        cluster.sync();  // peers keep their memory until slice 0 has read it
        if (slice != 0) return;
    }
    if (!live) return;
#pragma unroll
    for (int i = 0; i < I; ++i)
#pragma unroll
        for (int j = 0; j < J; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) emit(i, j, h, acc[i][j][2 * h], acc[i][j][2 * h + 1]);
}

// A row's sums over each 64-input group (the affine bias term's input), fp32 [rows][k / 64].
__global__ void gsum_kernel(const __nv_bfloat16* __restrict__ x, int ldx, int k, float* __restrict__ xs) {
    const int r = blockIdx.x, groups = k / GS;
    const __nv_bfloat16* xr = x + static_cast<size_t>(r) * ldx;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        float s = 0.f;
        for (int i = 0; i < GS; ++i) s += __bfloat162float(xr[g * GS + i]);
        xs[static_cast<size_t>(r) * groups + g] = s;
    }
}

template <int BM, int STAGES>
int launch(const void* x, const float* xs, const uint32_t* w, const void* s, const void* b, void* y, int m, int k, int n, int npad,
           int ldx, int sk, cudaStream_t st) {
    using T = Tile<BM, 64, 1, 4, STAGES>;
    auto kern = q4_kernel<BM, 64, 1, 4, STAGES>;
    const int rows_t = (m + BM - 1) / BM, tiles = (n + 63) / 64;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>(rows_t * tiles * sk));
    cfg.blockDim = dim3(T::THREADS);
    cfg.dynamicSmemBytes = T::SMEM;
    cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = sk;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = sk > 1 ? 1 : 0;
    if (T::SMEM > 48 * 1024) {
        const cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        if (e != cudaSuccess) return (int)e;
    }
    return (int)cudaLaunchKernelEx(&cfg, kern, static_cast<const __nv_bfloat16*>(x), xs, w, static_cast<const __nv_bfloat16*>(s),
                                   static_cast<const __nv_bfloat16*>(b), static_cast<__nv_bfloat16*>(y), m, k, n, npad, ldx,
                                   rows_t, sk);
}

}  // namespace gb_q4

// K slices of a weight: up to 8, while the column tiles times the slices stay under 192 blocks and each slice keeps 8+ groups.
GB_EXPORT int gb_q4_split_k(int n, int k) {
    const int tiles = (n + 63) / 64, groups = k / gb_q4::GS;
    int sk = 1;
    while (sk < 8 && tiles * sk < 192 && groups % (sk * 2) == 0 && groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

// y [m, n] bf16 = x [m, k] (row stride ldx) . W^T. xs: gb_q4_gsum's [m][k / 64]. k % 64 == 0, npad = n rounded up to 64s.
GB_EXPORT int gb_q4_linear(const void* x, const float* xs, const void* w, const void* scales, const void* biases, void* y, int m,
                           int k, int n, int npad, int ldx, void* stream) {
    if (!x || !xs || !w || !scales || !biases || !y || m <= 0 || k <= 0 || n <= 0 || k % gb_q4::GS != 0 || npad % 64 != 0 ||
        npad < n || ldx < k || ldx % 8 != 0)
        return (int)cudaErrorInvalidValue;
    const int sk = gb_q4_split_k(n, k);
    const uint32_t* wq = static_cast<const uint32_t*>(w);
    cudaStream_t st = gb_stream(stream);
    if (m <= 32) return gb_q4::launch<16, 8>(x, xs, wq, scales, biases, y, m, k, n, npad, ldx, sk, st);
    return gb_q4::launch<64, 4>(x, xs, wq, scales, biases, y, m, k, n, npad, ldx, sk, st);
}

GB_EXPORT int gb_q4_gsum(const void* x, int ldx, int k, float* xs, int rows, void* stream) {
    if (!x || !xs || rows <= 0 || k % gb_q4::GS != 0 || ldx < k) return (int)cudaErrorInvalidValue;
    gb_q4::gsum_kernel<<<rows, 128, 0, gb_stream(stream)>>>(static_cast<const __nv_bfloat16*>(x), ldx, k, xs);
    return GB_CHECK_LAUNCH();
}
