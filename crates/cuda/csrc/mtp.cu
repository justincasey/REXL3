// MTP head input kernels for Flash-Next. Original REXL3 code, Victor Cruz.
//
//   xs [R * hc][H] f16 = stack[r][s] * rsqrt(mean_d stack[r][s][d]^2 + eps) * (1 + nh[s * H + d])   (per stream)
//   xe [R][H]      f16 = emb[r] * rsqrt(mean_d emb[r][d]^2 + eps) * ne[d]                           (ne = 1 + w)
//   dst [R][hc][H] f32 = h[r * hc + s] + e[r]       (the embedding branch is broadcast into every stream)
#include "common.cuh"

namespace mtpk {

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

// grid: R * (hc + 1) blocks of 256; block (r, s): s < hc a stream of the stack, s == hc the embedding row
__global__ void __launch_bounds__(256) in_kernel(const float* __restrict__ stack, const half* __restrict__ nh,
                                                 const __nv_bfloat16* __restrict__ emb, const float* __restrict__ ne,
                                                 half* __restrict__ xs, half* __restrict__ xe, int H, int hc,
                                                 float eps) {
    __shared__ float red[32];
    const int r = blockIdx.x / (hc + 1), s = blockIdx.x % (hc + 1);
    if (s < hc) {
        const float* x = stack + ((size_t)r * hc + s) * H;
        float a = 0.f;
        for (int d = threadIdx.x; d < H; d += blockDim.x) a += x[d] * x[d];
        const float inv = rsqrtf(block_sum(a, red) / (float)H + eps);
        half* o = xs + ((size_t)r * hc + s) * H;
        for (int d = threadIdx.x; d < H; d += blockDim.x)
            o[d] = __float2half_rn(x[d] * inv * (1.f + __half2float(nh[(size_t)s * H + d])));
    } else {
        const __nv_bfloat16* x = emb + (size_t)r * H;
        float a = 0.f;
        for (int d = threadIdx.x; d < H; d += blockDim.x) {
            const float v = __bfloat162float(x[d]);
            a += v * v;
        }
        const float inv = rsqrtf(block_sum(a, red) / (float)H + eps);
        half* o = xe + (size_t)r * H;
        for (int d = threadIdx.x; d < H; d += blockDim.x) o[d] = __float2half_rn(__bfloat162float(x[d]) * inv * ne[d]);
    }
}

__global__ void combine_kernel(const half* __restrict__ h, const half* __restrict__ e, float* __restrict__ dst,
                               int R, int H, int hc) {
    const size_t n = (size_t)R * hc * H;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const size_t d = i % H, rs = i / H, r = rs / hc;
        dst[i] = __half2float(h[i]) + __half2float(e[r * H + d]);
    }
}

__global__ void copy_i32_kernel(int* __restrict__ dst, const int* __restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// One draft step's inputs from device state for `n` sequences of one row each (sequence s drafts into `drafts[s * stride + ...]`): the
// previous step's draft becomes the token, and the step's position is written everywhere a window of `n` one-row sequences reads it.
__global__ void stage_kernel(int* __restrict__ ids, int* __restrict__ positions, int* __restrict__ row_pos,
                             int* __restrict__ row_slot, int* __restrict__ seq_start, const int* __restrict__ seq_slot,
                             const int* __restrict__ drafts, const int* __restrict__ chain_pos, int n, int stride, int j) {
    const int s = threadIdx.x;
    if (s < n) {
        ids[s] = drafts[s * stride + j - 1];
        const int p = chain_pos[s * stride + j];
        positions[s] = p;
        row_pos[s] = p;
        row_slot[s] = seq_slot[s];
        seq_start[s] = s;
    }
    if (s == 0) seq_start[n] = n;
}

// Draft j from one row of bf16 logits [n]: argmax (lowest id on ties) and its softmax probability.
__global__ void __launch_bounds__(1024) pick_kernel(const __nv_bfloat16* __restrict__ logits, int n,
                                                    int* __restrict__ out_id, float* __restrict__ out_p, int j) {
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ float ss[32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = __bfloat162float(logits[i]);
        if (v > bv) { bv = v; bi = i; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    if (tid < 32) {
        float v = tid < nw ? sv[tid] : -INFINITY;
        int ix = tid < nw ? si[tid] : 0x7fffffff;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, ix, o);
            if (ov > v || (ov == v && oi < ix)) { v = ov; ix = oi; }
        }
        if (tid == 0) { sv[0] = v; si[0] = ix; }
    }
    __syncthreads();
    const float mx = sv[0];
    float a = 0.f;
    for (int i = tid; i < n; i += blockDim.x) a += __expf(__bfloat162float(logits[i]) - mx);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) ss[warp] = a;
    __syncthreads();
    if (tid == 0) {
        float t = 0.f;
        for (int w = 0; w < nw; ++w) t += ss[w];
        out_id[j] = (si[0] >= 0 && si[0] < n) ? si[0] : 0;
        out_p[j] = 1.f / t;
    }
}

}  // namespace mtpk

using namespace mtpk;

GB_EXPORT int gb_mtp_input(const void* stack, const void* norm_hidden, const void* emb, const void* norm_emb, void* xs,
                           void* xe, int rows, int hc, int H, float eps, void* stream) {
    if (rows < 1 || hc < 1 || H < 1) return (int)cudaErrorInvalidValue;
    in_kernel<<<rows * (hc + 1), 256, 0, gb_stream(stream)>>>((const float*)stack, (const half*)norm_hidden,
                                                             (const __nv_bfloat16*)emb, (const float*)norm_emb,
                                                             (half*)xs, (half*)xe, H, hc, eps);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_mtp_combine(const void* h, const void* e, void* dst, int rows, int hc, int H, void* stream) {
    if (rows < 1) return (int)cudaErrorInvalidValue;
    const size_t n = (size_t)rows * hc * H;
    const int blocks = (int)((n + 255) / 256 < 1024 ? (n + 255) / 256 : 1024);
    combine_kernel<<<blocks, 256, 0, gb_stream(stream)>>>((const half*)h, (const half*)e, (float*)dst, rows, H, hc);
    return GB_CHECK_LAUNCH();
}

// dst[0 .. n) = src[0 .. n) (int32) on the stream: device-resident token hand-off between draft steps.
GB_EXPORT int gb_copy_i32(void* dst, const void* src, int n, void* stream) {
    if (n < 1) return 0;
    copy_i32_kernel<<<(n + 127) / 128, 128, 0, gb_stream(stream)>>>((int*)dst, (const int*)src, n);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_mtp_stage(void* ids, void* positions, void* row_pos, void* row_slot, void* seq_start, const void* seq_slot,
                           const void* drafts, const void* chain_pos, int n, int stride, int j, void* stream) {
    if (j < 1 || n < 1 || n > 32 || stride < j + 1) return (int)cudaErrorInvalidValue;
    stage_kernel<<<1, 32, 0, gb_stream(stream)>>>((int*)ids, (int*)positions, (int*)row_pos, (int*)row_slot, (int*)seq_start,
                                                   (const int*)seq_slot, (const int*)drafts, (const int*)chain_pos, n, stride, j);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_mtp_pick(const void* logits, int n, void* out_id, void* out_p, int j, void* stream) {
    if (n < 1) return (int)cudaErrorInvalidValue;
    pick_kernel<<<1, 1024, 0, gb_stream(stream)>>>((const __nv_bfloat16*)logits, n, (int*)out_id, (float*)out_p, j);
    return GB_CHECK_LAUNCH();
}
