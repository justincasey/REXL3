// Multi-HyperConnection (mHC) low-rank mixer for Qwen4_Exp / Flash-Next
// Optimized warp-cooperative coalesced implementation for sm_121 / DGX Spark (GB10).
#include "common.cuh"
#include "pdl.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define QHC_BLOCK 256

__device__ __forceinline__ float qhc_silu(float v) {
    return v / (1.0f + __expf(-v));
}

__device__ __forceinline__ float qhc_sigmoid(float v) {
    return 1.0f / (1.0f + __expf(-v));
}

extern "C" __global__ void hc_expand_kernel(
    const __nv_bfloat16* __restrict__ hidden, // [T, H]
    float* __restrict__ streams,              // [T, hc, H] FP32 highway
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const __nv_bfloat16* x = hidden + (size_t)t * H;
    float* s = streams + (size_t)t * hc_mult * H;
    for (unsigned int d = tid; d < H; d += QHC_BLOCK) {
        float v = (float)x[d];
        for (unsigned int i = 0; i < hc_mult; ++i) s[i * H + d] = v;
    }
}

// ── Kernel 1: Per-stream RMS & norm stage ──
// Grid: (hc, T), Block: 256
extern "C" __global__ void hc_norm_stage_kernel(
    const float* __restrict__ streams,
    const half* __restrict__ hc_norm_w,
    float* __restrict__ normed_out,
    const unsigned int H,
    const unsigned int hc,
    const float eps
) {
    const unsigned int t = blockIdx.y;
    const unsigned int s2 = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int warps = blockDim.x >> 5;
    __shared__ float smem_red[32];
    __shared__ float smem_rms;

    const float* xs = streams + (size_t)t * hc * H + (size_t)s2 * H;
    float acc = 0.0f;
    for (unsigned int d = tid; d < H; d += blockDim.x) {
        float v = xs[d];
        acc += v * v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) smem_red[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (unsigned int w = 0; w < warps; ++w) tot += smem_red[w];
        smem_rms = rsqrtf(tot / (float)H + eps);
    }
    __syncthreads();

    float rms = smem_rms;
    const half* nw = hc_norm_w + (size_t)s2 * H;
    float* out = normed_out + (size_t)t * hc * H + (size_t)s2 * H;
    for (unsigned int d = tid; d < H; d += blockDim.x) {
        out[d] = xs[d] * rms * (1.0f + __half2float(nw[d]));
    }
}

// ── Kernel 2: Coalesced down projection ──
// Grid: ((rank + 7) / 8, T), Block: 256 (32 warps across blocks, 1 warp per rank row)
extern "C" __global__ void hc_down_coalesced_kernel(
    const float* __restrict__ normed,
    const half* __restrict__ down_w,
    float* __restrict__ low_out,
    const unsigned int hc_dim,
    const unsigned int rank,
    const float inv_hc
) {
    const unsigned int t = blockIdx.y;
    const unsigned int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= rank) return;
    const unsigned int lane = threadIdx.x & 31u;

    const float* nx = normed + (size_t)t * hc_dim;
    const half* row = down_w + (size_t)r * hc_dim;
    float acc = 0.0f;
    for (unsigned int i = lane; i < hc_dim; i += 32) {
        acc += __half2float(row[i]) * nx[i];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) low_out[(size_t)t * rank + r] = qhc_silu(acc * inv_hc);
}

// ── Kernel 3: Coalesced up projection + stream mean ──
// Grid: ((H + 7) / 8, T), Block: 256 (1 warp per hidden dimension d)
extern "C" __global__ void hc_up_coalesced_kernel(
    const float* __restrict__ normed,
    const float* __restrict__ low,
    const half* __restrict__ up_w,
    __nv_bfloat16* __restrict__ y_out,
    const unsigned int H,
    const unsigned int hc,
    const unsigned int rank,
    const float inv_hc
) {
    const unsigned int t = blockIdx.y;
    const unsigned int d = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (d >= H) return;
    const unsigned int lane = threadIdx.x & 31u;

    const float* nx = normed + (size_t)t * hc * H;
    const float* lo = low + (size_t)t * rank;
    const unsigned int niters = rank >> 5;

    float mixed = 0.0f;
    for (unsigned int s2 = 0; s2 < hc; ++s2) {
        const unsigned int i = s2 * H + d;
        const half* up_row = up_w + (size_t)i * rank;
        float acc = 0.0f;
        for (unsigned int k = 0; k < niters; ++k) {
            unsigned int r = lane + (k << 5);
            acc += __half2float(up_row[r]) * lo[r];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        acc = __shfl_sync(0xFFFFFFFFu, acc, 0);
        mixed += qhc_sigmoid(acc) * nx[i];
    }
    if (lane == 0) {
        y_out[(size_t)t * H + d] = __float2bfloat16(mixed * inv_hc);
    }
}

// ── Kernel 4: Coalesced block injection ──
// Grid: (1, T), Block: 256
extern "C" __global__ void hc_inject_coalesced_kernel(
    const float* __restrict__ normed,
    const half* __restrict__ inject_w,
    float* __restrict__ inj_out,
    const unsigned int hc_dim,
    const unsigned int hc,
    const float inv_hc
) {
    const unsigned int t = blockIdx.y;
    const unsigned int s2 = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (s2 >= hc) return;
    const unsigned int lane = threadIdx.x & 31u;

    const float* nx = normed + (size_t)t * hc_dim;
    const half* row = inject_w + (size_t)s2 * hc_dim;
    float acc = 0.0f;
    for (unsigned int i = lane; i < hc_dim; i += 32) {
        acc += __half2float(row[i]) * nx[i];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) inj_out[(size_t)t * hc + s2] = 2.0f * qhc_sigmoid(acc * inv_hc);
}

// ── Kernel 5: Post injection ──
// `out` may be `residual` (the engine updates the streams in place), so those two are not __restrict__: every element
// is read and then written by the same thread, never touched by another. HC is a template argument so the per-thread
// stream arrays live in registers, and all of a column's residual loads issue before its stores.
//
// Mapping: one block per (t, d-strip); blockIdx.y is the token and the x grid covers the hidden width in 256-wide
// strips. Each thread loads its bf16 hidden value `vx` exactly once and evaluates all HC injection streams for it
// (the per-element `rv + vx * wv[s]` fused multiply-add is the reference arithmetic), so there are no integer
// divisions and `block_out` is not re-read per stream.
template <int HC>
__global__ void hc_post_kernel(
    const __nv_bfloat16* __restrict__ block_out, // [T, H]
    const float* residual,                       // [T, hc, H]
    const float* __restrict__ inj,               // [T, hc]
    float* out,                                  // [T, hc, H]
    const unsigned int hidden_size
) {
    gbp_pdl::pdl_trigger();  // the next pre-mix may start loading its weights now; it waits for this grid before reading x
    gbp_pdl::pdl_wait();     // block_out and inj are the previous kernels' output, and `out` may still be read by them
    const unsigned int t = blockIdx.y;
    const unsigned int H = hidden_size;
    const __nv_bfloat16* x = block_out + (size_t)t * H;
    const float* res = residual + (size_t)t * HC * H;
    const float* w = inj + (size_t)t * HC;
    float* o = out + (size_t)t * HC * H;

    float wv[HC];
#pragma unroll
    for (int s = 0; s < HC; ++s) wv[s] = w[s];

    for (unsigned int d = blockIdx.x * blockDim.x + threadIdx.x; d < H; d += gridDim.x * blockDim.x) {
        const float vx = (float)x[d];
        float rv[HC];
#pragma unroll
        for (int s = 0; s < HC; ++s) rv[s] = res[(size_t)s * H + d];
#pragma unroll
        for (int s = 0; s < HC; ++s) o[(size_t)s * H + d] = rv[s] + vx * wv[s];
    }
}

// ── Host Launchers ──

GB_EXPORT int gb_fn_hc_expand(
    const void* hidden,
    void* streams,
    unsigned int hidden_size,
    unsigned int hc_mult,
    unsigned int num_tokens,
    void* stream
) {
    if (num_tokens == 0) return 0;
    hc_expand_kernel<<<num_tokens, QHC_BLOCK, 0, gb_stream(stream)>>>(
        (const __nv_bfloat16*)hidden,
        (float*)streams,
        hidden_size,
        hc_mult
    );
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_fn_hc_pre(
    const void* streams,
    const void* hc_norm_w,
    const void* down_w,
    const void* up_w,
    const void* inject_w,
    void* normed_scratch,
    float* low_scratch,
    void* y_out,
    float* inj_out,
    unsigned int hidden_size,
    unsigned int hc_mult,
    unsigned int rank,
    float norm_eps,
    unsigned int num_tokens,
    void* stream
) {
    if (num_tokens == 0) return 0;
    cudaStream_t st = gb_stream(stream);
    const unsigned int hc_dim = hc_mult * hidden_size;
    const float inv_hc = 1.0f / (float)hc_mult;

    // 1. Stage normed
    hc_norm_stage_kernel<<<dim3(hc_mult, num_tokens), QHC_BLOCK, 0, st>>>(
        (const float*)streams,
        (const half*)hc_norm_w,
        (float*)normed_scratch,
        hidden_size,
        hc_mult,
        norm_eps
    );

    // 2. Coalesced down projection
    hc_down_coalesced_kernel<<<dim3((rank + 7) / 8, num_tokens), QHC_BLOCK, 0, st>>>(
        (const float*)normed_scratch,
        (const half*)down_w,
        low_scratch,
        hc_dim,
        rank,
        inv_hc
    );

    // 3. Coalesced up projection
    hc_up_coalesced_kernel<<<dim3((hidden_size + 7) / 8, num_tokens), QHC_BLOCK, 0, st>>>(
        (const float*)normed_scratch,
        low_scratch,
        (const half*)up_w,
        (__nv_bfloat16*)y_out,
        hidden_size,
        hc_mult,
        rank,
        inv_hc
    );

    // 4. Coalesced inject (if requested)
    if (inject_w != nullptr && inj_out != nullptr) {
        hc_inject_coalesced_kernel<<<dim3((hc_mult + 7) / 8, num_tokens), QHC_BLOCK, 0, st>>>(
            (const float*)normed_scratch,
            (const half*)inject_w,
            inj_out,
            hc_dim,
            hc_mult,
            inv_hc
        );
    }
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_fn_hc_head(
    const void* streams,
    const void* hc_norm_w,
    const void* down_w,
    const void* up_w,
    void* normed_scratch,
    float* low_scratch,
    void* y_out,
    unsigned int hidden_size,
    unsigned int hc_mult,
    unsigned int rank,
    float norm_eps,
    unsigned int num_tokens,
    void* stream
) {
    return gb_fn_hc_pre(
        streams,
        hc_norm_w,
        down_w,
        up_w,
        nullptr,
        normed_scratch,
        low_scratch,
        y_out,
        nullptr,
        hidden_size,
        hc_mult,
        rank,
        norm_eps,
        num_tokens,
        stream
    );
}

GB_EXPORT int gb_fn_hc_post(
    const void* block_out,
    const void* residual,
    const float* inj,
    void* out,
    unsigned int hidden_size,
    unsigned int hc_mult,
    unsigned int num_tokens,
    void* stream
) {
    if (num_tokens == 0) return 0;
    const __nv_bfloat16* bo = (const __nv_bfloat16*)block_out;
    const float* res = (const float*)residual;
    float* o = (float*)out;
    cudaStream_t st = gb_stream(stream);
    // One block per (token, 256-wide hidden strip): zero divisions in the kernel, and the hidden value per thread is
    // loaded once for all HC streams.
    const unsigned int blocks_x = (hidden_size + QHC_BLOCK - 1) / QHC_BLOCK;
    const dim3 grid(blocks_x, num_tokens);
    // Under gbp_pdl::kHcx the launch carries the programmatic-dependent-launch attribute (the kernel waits before it reads).
    const bool pdl = gbp_pdl::on(gbp_pdl::kHcx);
    int rc = 0;
    switch (hc_mult) {
#define GB_HC_POST(N)                                                                                                  \
    case N:                                                                                                            \
        rc = gbp_pdl::launch_pdl(pdl, hc_post_kernel<N>, grid, dim3(QHC_BLOCK), 0, st, bo, res, inj, o, hidden_size);  \
        break;
        GB_HC_POST(1) GB_HC_POST(2) GB_HC_POST(3) GB_HC_POST(4) GB_HC_POST(5) GB_HC_POST(6) GB_HC_POST(7) GB_HC_POST(8)
#undef GB_HC_POST
        default: return (int)cudaErrorInvalidValue;  // per-thread stream arrays are sized at compile time, up to 8
    }
    return rc;
}
