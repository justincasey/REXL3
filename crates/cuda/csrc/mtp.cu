// MTP head input kernels for Flash-Next. Original REXL3 code, Victor Cruz.
//
//   xs [R * hc][H] f16 = stack[r][s] * rsqrt(mean_d stack[r][s][d]^2 + eps) * (1 + nh[s * H + d])   (per stream)
//   xe [R][H]      f16 = emb[r] * rsqrt(mean_d emb[r][d]^2 + eps) * ne[d]                           (ne = 1 + w)
//   dst [R][hc][H] f32 = h[r * hc + s] + e[r]       (the embedding branch is broadcast into every stream)
#include "common.cuh"
#include "pdl.cuh"

// Decode launches (gbp_pdl::kFlash): a kernel launched with the programmatic-dependent-launch attribute may start before its predecessor
// finishes, so it calls pdl_wait() -- first in every thread -- before it reads anything an earlier kernel wrote or writes anything an
// earlier kernel may still read; weights are read before it. A launch without the attribute makes both calls no-ops (pdl.cuh), so only
// timing changes, never a bit.
using gbp_pdl::pdl_trigger;
using gbp_pdl::pdl_wait;
inline bool flash_pdl() { return gbp_pdl::on(gbp_pdl::kFlash); }

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

// Elements of a block's row kept in registers: 12 a thread of the block's 256 (hidden sizes up to 3,072; the rest of a longer row is read
// from memory as before). The weight row across the wait, the input row after it, so each is read once.
constexpr int IN_NT = 256, IN_MAXC = 12;

// grid: R * (hc + 1) blocks of 256; block (r, s): s < hc a stream of the stack, s == hc the embedding row
__global__ void __launch_bounds__(IN_NT) in_kernel(const float* __restrict__ stack, const half* __restrict__ nh,
                                                   const __nv_bfloat16* __restrict__ emb, const float* __restrict__ ne,
                                                   half* __restrict__ xs, half* __restrict__ xe, int H, int hc, float eps) {
    __shared__ float red[32];
    pdl_trigger();  // the next kernel may launch: it waits for this grid before it reads xs or xe
    const int r = blockIdx.x / (hc + 1), s = blockIdx.x % (hc + 1);
    // before the wait: this block's weight row (nh[s] as f32, or ne), the elements d = threadIdx.x + 256 c, c < 12
    float cst[IN_MAXC];
#pragma unroll
    for (int c = 0; c < IN_MAXC; ++c) {
        const int d = threadIdx.x + c * IN_NT;
        cst[c] = d >= H ? 0.f : s < hc ? __half2float(nh[(size_t)s * H + d]) : ne[d];
    }
    pdl_wait();  // stack and emb are earlier kernels' outputs; xs and xe may still be read by earlier kernels
    float xr[IN_MAXC];
    float a = 0.f;
    if (s < hc) {
        const float* x = stack + ((size_t)r * hc + s) * H;
#pragma unroll
        for (int c = 0; c < IN_MAXC; ++c) {
            const int d = threadIdx.x + c * IN_NT;
            xr[c] = d < H ? x[d] : 0.f;
        }
        // the sum of squares in increasing d, as a loop over d = threadIdx.x, + 256, ... would add it
#pragma unroll
        for (int c = 0; c < IN_MAXC; ++c) {
            if (threadIdx.x + c * IN_NT < H) a += xr[c] * xr[c];
        }
        for (int d = threadIdx.x + IN_MAXC * IN_NT; d < H; d += IN_NT) a += x[d] * x[d];
        const float inv = rsqrtf(block_sum(a, red) / (float)H + eps);
        half* o = xs + ((size_t)r * hc + s) * H;
#pragma unroll
        for (int c = 0; c < IN_MAXC; ++c) {
            const int d = threadIdx.x + c * IN_NT;
            if (d < H) o[d] = __float2half_rn(xr[c] * inv * (1.f + cst[c]));
        }
        for (int d = threadIdx.x + IN_MAXC * IN_NT; d < H; d += IN_NT)
            o[d] = __float2half_rn(x[d] * inv * (1.f + __half2float(nh[(size_t)s * H + d])));
    } else {
        const __nv_bfloat16* x = emb + (size_t)r * H;
#pragma unroll
        for (int c = 0; c < IN_MAXC; ++c) {
            const int d = threadIdx.x + c * IN_NT;
            xr[c] = d < H ? __bfloat162float(x[d]) : 0.f;
        }
#pragma unroll
        for (int c = 0; c < IN_MAXC; ++c) {
            if (threadIdx.x + c * IN_NT < H) a += xr[c] * xr[c];
        }
        for (int d = threadIdx.x + IN_MAXC * IN_NT; d < H; d += IN_NT) {
            const float v = __bfloat162float(x[d]);
            a += v * v;
        }
        const float inv = rsqrtf(block_sum(a, red) / (float)H + eps);
        half* o = xe + (size_t)r * H;
#pragma unroll
        for (int c = 0; c < IN_MAXC; ++c) {
            const int d = threadIdx.x + c * IN_NT;
            if (d < H) o[d] = __float2half_rn(xr[c] * inv * cst[c]);
        }
        for (int d = threadIdx.x + IN_MAXC * IN_NT; d < H; d += IN_NT) o[d] = __float2half_rn(__bfloat162float(x[d]) * inv * ne[d]);
    }
}

__global__ void combine_kernel(const half* __restrict__ h, const half* __restrict__ e, float* __restrict__ dst,
                               int R, int H, int hc) {
    pdl_trigger();
    pdl_wait();  // h and e are earlier kernels' outputs; dst may still be read by earlier kernels
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
    pdl_trigger();
    pdl_wait();  // drafts are the previous pick's output; the staged arrays may still be read by earlier kernels
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

// Draft j from one row of bf16 logits [n], any n >= 1: argmax (lowest id on ties) and its softmax probability. One block of 1024 threads;
// thread t owns the elements t, t + 1024, ... and walks them twice (the argmax, then the sum of exps) with one dependent global load per
// element. `pick_wide_kernel` below is the same function for the n of the production draft head.
// The result is defined by the arithmetic order of this kernel, and pick_wide_kernel reproduces it exactly: out_p = 1 / t, t = the 32 warp
// sums added in order by thread 0, a warp sum = the xor-shuffle tree over the 32 per-thread sums, a thread's sum = __expf(x - max) added in
// increasing element order starting from 0. The argmax is the lowest index of the maximum whatever the reduction order.
__global__ void __launch_bounds__(1024) pick_kernel(const __nv_bfloat16* __restrict__ logits, int n,
                                                    int* __restrict__ out_id, float* __restrict__ out_p, int j) {
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ float ss[32];
    pdl_trigger();
    pdl_wait();  // logits are the draft head's output; out_id and out_p may still be read by earlier kernels
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

// pick_kernel for n == 64 * 1024 (the draft head's domain) and a 4-byte aligned row, in 512 threads. Thread t owns the 64 words
// (two bf16 each) k * 512 + t: the elements 1024 k + 2 t and 1024 k + 2 t + 1, which are elements k of pick_kernel's threads 2 t and 2 t + 1.
// It keeps those two threads' sequences apart (two argmax scans, two sums, each over k in increasing order from 0: the same additions of
// the same __expf values), so a thread's sum is pick_kernel's bit for bit. pick_kernel's warp w adds the 32 sums of its threads 32 w ..
// 32 w + 31 with an xor tree over distances 16, 8, 4, 2, 1; here those 32 sums are the even and odd chains of a half warp (lanes 16 h ..
// 16 h + 15), and the same tree is xor shuffles over 8, 4, 2, 1 lanes of each chain followed by the even chain plus the odd one: the
// same additions in the same association (the operands of an add commute exactly). The 32 group sums are added in order by thread 0
// as before. Every one of a thread's 64 words is loaded up front (independent loads into 64 registers; the block is 512 threads, so a
// thread may use 128), where pick_kernel waits for one load after another, twice. The scans remember the k of the best element, not
// its index (a constant of each unrolled step, so no index registers), and the index is made after the loop. The cross-warp argmax is
// reduced by every warp (the result is unique: the maximum with the lowest index, whatever the order), which saves a barrier.
__global__ void __launch_bounds__(512) pick_wide_kernel(const uint32_t* __restrict__ logits, int* __restrict__ out_id,
                                                        float* __restrict__ out_p, int j) {
    constexpr int T = 512, K = 64;
    __shared__ float sv[16];
    __shared__ int si[16];
    __shared__ float ss[32];
    pdl_trigger();
    pdl_wait();  // logits are the draft head's output; out_id and out_p may still be read by earlier kernels
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    uint32_t w[K];
#pragma unroll
    for (int k = 0; k < K; ++k) w[k] = logits[k * T + tid];
    float bv0 = -INFINITY, bv1 = -INFINITY;
    int bk0 = -1, bk1 = -1;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const float v0 = __uint_as_float(w[k] << 16), v1 = __uint_as_float(w[k] & 0xffff0000u);
        if (v0 > bv0) { bv0 = v0; bk0 = k; }
        if (v1 > bv1) { bv1 = v1; bk1 = k; }
    }
    float bv = bv0;
    int bi = bk0 < 0 ? 0x7fffffff : 2 * (bk0 * T + tid);
    const int bi1 = bk1 < 0 ? 0x7fffffff : 2 * (bk1 * T + tid) + 1;
    if (bv1 > bv || (bv1 == bv && bi1 < bi)) { bv = bv1; bi = bi1; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    float mx = sv[lane & 15];
    int ix = si[lane & 15];
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, mx, o);
        const int oi = __shfl_xor_sync(0xffffffffu, ix, o);
        if (ov > mx || (ov == mx && oi < ix)) { mx = ov; ix = oi; }
    }
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        a0 += __expf(__uint_as_float(w[k] << 16) - mx);
        a1 += __expf(__uint_as_float(w[k] & 0xffff0000u) - mx);
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) {
        a0 += __shfl_xor_sync(0xffffffffu, a0, o);
        a1 += __shfl_xor_sync(0xffffffffu, a1, o);
    }
    if ((lane & 15) == 0) ss[2 * warp + (lane >> 4)] = a0 + a1;
    __syncthreads();
    if (tid == 0) {
        float t = 0.f;
        for (int q = 0; q < 32; ++q) t += ss[q];
        out_id[j] = ix;
        out_p[j] = 1.f / t;
    }
}

}  // namespace mtpk

using namespace mtpk;

GB_EXPORT int gb_mtp_input(const void* stack, const void* norm_hidden, const void* emb, const void* norm_emb, void* xs,
                           void* xe, int rows, int hc, int H, float eps, void* stream) {
    if (rows < 1 || hc < 1 || H < 1) return (int)cudaErrorInvalidValue;
    gbp_pdl::launch_pdl(flash_pdl(), in_kernel, dim3((unsigned)(rows * (hc + 1))), dim3(IN_NT), 0, gb_stream(stream),
                        (const float*)stack, (const half*)norm_hidden, (const __nv_bfloat16*)emb, (const float*)norm_emb, (half*)xs,
                        (half*)xe, H, hc, eps);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_mtp_combine(const void* h, const void* e, void* dst, int rows, int hc, int H, void* stream) {
    if (rows < 1) return (int)cudaErrorInvalidValue;
    const size_t n = (size_t)rows * hc * H;
    const int blocks = (int)((n + 255) / 256 < 1024 ? (n + 255) / 256 : 1024);
    gbp_pdl::launch_pdl(flash_pdl(), combine_kernel, dim3((unsigned)blocks), dim3(256), 0, gb_stream(stream), (const half*)h,
                        (const half*)e, (float*)dst, rows, H, hc);
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
    gbp_pdl::launch_pdl(flash_pdl(), stage_kernel, dim3(1), dim3(32), 0, gb_stream(stream), (int*)ids, (int*)positions,
                        (int*)row_pos, (int*)row_slot, (int*)seq_start, (const int*)seq_slot, (const int*)drafts,
                        (const int*)chain_pos, n, stride, j);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_mtp_pick(const void* logits, int n, void* out_id, void* out_p, int j, void* stream) {
    if (n < 1) return (int)cudaErrorInvalidValue;
    if (n == 64 * 1024 && ((uintptr_t)logits & 3) == 0) {
        gbp_pdl::launch_pdl(flash_pdl(), pick_wide_kernel, dim3(1), dim3(512), 0, gb_stream(stream), (const uint32_t*)logits,
                            (int*)out_id, (float*)out_p, j);
    } else {
        gbp_pdl::launch_pdl(flash_pdl(), pick_kernel, dim3(1), dim3(1024), 0, gb_stream(stream), (const __nv_bfloat16*)logits, n,
                            (int*)out_id, (float*)out_p, j);
    }
    return GB_CHECK_LAUNCH();
}
