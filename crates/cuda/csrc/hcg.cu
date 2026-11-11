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

// norm_kernel after hc_post, in one pass over the streams: x[t][s][d] = x[t][s][d] + bf16(block_out[t][d]) * inj_prev[t][s] (hc_post's
// expression, a fused multiply-add, written back in place) and then norm_kernel's rms and fp16 hi+lo split of the NEW values, held in
// registers instead of being read back. Same thread-to-element map, same sums, same rounding as the two kernels it replaces, so the
// streams, rms, hi and lo are theirs bit for bit. H <= NCH * 1024 (one row a block of 256 threads, four floats a thread a step).
template <int HC, int NCH>
__global__ void __launch_bounds__(256) post_norm_kernel(float* __restrict__ x, const __nv_bfloat16* __restrict__ block_out,
                                                        const float* __restrict__ inj_prev, const half* __restrict__ nw,
                                                        half* __restrict__ normed_hi, half* __restrict__ normed_lo,
                                                        float* __restrict__ rms, int H, float eps) {
    __shared__ float red[8];
    const int t = blockIdx.x;
    float wv[HC];
#pragma unroll
    for (int s = 0; s < HC; ++s) wv[s] = inj_prev[(size_t)t * HC + s];
    float4 xn[HC][NCH];
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
        const int d = threadIdx.x * 4 + c * 1024;
        if (d >= H) continue;
        const uint2 u = *reinterpret_cast<const uint2*>(block_out + (size_t)t * H + d);
        const float2 b01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
#pragma unroll
        for (int s = 0; s < HC; ++s) {
            float4* xp = reinterpret_cast<float4*>(x + ((size_t)t * HC + s) * H + d);
            const float4 rv = *xp;
            float4 n;
            n.x = fmaf(b01.x, wv[s], rv.x);
            n.y = fmaf(b01.y, wv[s], rv.y);
            n.z = fmaf(b23.x, wv[s], rv.z);
            n.w = fmaf(b23.y, wv[s], rv.w);
            *xp = n;
            xn[s][c] = n;
        }
    }
#pragma unroll
    for (int s = 0; s < HC; ++s) {
        float a = 0.f;
#pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const int d = threadIdx.x * 4 + c * 1024;
            if (d >= H) continue;
            const float4 v = xn[s][c];
            a += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        const float r = rsqrtf(block_sum(a, red) / (float)H + eps);
        if (threadIdx.x == 0) rms[t * HC + s] = r;
        half* hi = normed_hi + ((size_t)t * HC + s) * H;
        half* lo = normed_lo + ((size_t)t * HC + s) * H;
        const half* w = nw + (size_t)s * H;
#pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const int d = threadIdx.x * 4 + c * 1024;
            if (d >= H) continue;
            const float4 v = xn[s][c];
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
// up + mix in one kernel (the Flash-Next shape).
//
//   G[t][s * H + d] = (chain over k of low_hi[t][k] * up[s * H + d][k]) + (chain over k of low_lo[t][k] * up[s * H + d][k])
//   y[t][d]         = bf16(sum_s sigmoid(G) * x * rms * (1 + nw))        in stream order, exactly mix_kernel's expression
//
// The two products of the slab loop are plain sequential m16n8k16 chains over k from +0.0 (no split-K at K = rank; every tile and
// every GEMM M of every cuBLAS algorithm measured gives these bits), the lo GEMM's beta = 1 epilogue adds the hi sum once
// (fl(lo + hi), commutative), and mix_kernel's per-stream terms follow in the same order: fusing them moves where G is held (registers
// and shared memory instead of 168 MB of fp32 round trips through DRAM), never what it is.

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

// up + mix on a 64-row by 64-d tile of ALL four streams (the Flash-Next shape: hc = 4, rank % 32 == 0, H % 64 == 0), 16 warps: warp
// (stream s, d half h, row half r) owns the 32 rows r * 32 .., the 32 d h * 32 .. of stream s and both chains (hi, lo), so the epilogue sums
// the four streams of a (row, d) from shared memory in stream order. The operands of a tile are the hi+lo rows of `low` (128 rows) and the
// up rows of its 4 x 64 outputs (256 rows) for 4,096 outputs, half the L2 traffic of a 32 x 32 tile (the product is L2-bound, not MMA-bound).
// The epilogue's loads and stores are whole sectors: a thread owns d = 32 c + 4 q .. + 3 (c = 0, 1; q = the thread's index in its row's
// eight), so a row's eight threads read 128 contiguous bytes of x and write 64 of y a step.
constexpr int kU2Rows = 64, kU2D = 64, kU2KB = 32, kU2Pad = kU2KB + 8, kU2Threads = 512, kU2Ring = 3, kU2GsPad = kU2D + 8;
constexpr size_t kU2StageHalves = (size_t)(2 * kU2Rows + 4 * kU2D) * kU2Pad;
constexpr size_t kU2RingBytes = kU2Ring * kU2StageHalves * sizeof(half);
constexpr size_t kU2GsBytes = (size_t)4 * kU2Rows * kU2GsPad * sizeof(float);
constexpr size_t kU2Smem = kU2RingBytes > kU2GsBytes ? kU2RingBytes : kU2GsBytes;

// grid (H / 64, padded / 64), block 512, dynamic smem kU2Smem. low rows are valid up to the 64-row padding (zeros past T). LAZY: the streams
// hold the values BEFORE the post-inject of (block_out, inj_prev), applied to every x read here (fmaf(bf16, weight, x), hc_post's expression).
template <bool LAZY>
__global__ void __launch_bounds__(kU2Threads, 1)
    up_mix_kernel(const half* __restrict__ low_hi, const half* __restrict__ low_lo, const half* __restrict__ up_w,
                  const float* __restrict__ x, const __nv_bfloat16* __restrict__ block_out, const float* __restrict__ inj_prev,
                  const float* __restrict__ rms, const half* __restrict__ nw, __nv_bfloat16* __restrict__ y, int T, int H, int rank) {
    extern __shared__ __align__(16) unsigned char u2_smem[];
    half* sm = reinterpret_cast<half*>(u2_smem);
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t4 = lane & 3;
    const int ws = w & 3, wd = (w >> 2) & 1, wr = w >> 3;
    const int d0 = blockIdx.x * kU2D, r0 = blockIdx.y * kU2Rows;
    const int nk = rank / kU2KB;
    auto stage = [&](int st) { return sm + (size_t)st * kU2StageHalves; };
    auto load_stage = [&](int st, int c) {
        half* base = stage(st);
        const int k0 = c * kU2KB;
        constexpr int kChunks = kU2KB / 8;  // 16-byte pieces a row of the tile
#pragma unroll
        for (int it = 0; it < (2 * kU2Rows + 4 * kU2D) * kChunks / kU2Threads; ++it) {
            const int i = tid + it * kU2Threads;
            const int row = i / kChunks, ch = i % kChunks;
            const half* src;
            if (row < kU2Rows) src = low_hi + (size_t)(r0 + row) * rank + k0 + ch * 8;
            else if (row < 2 * kU2Rows) src = low_lo + (size_t)(r0 + row - kU2Rows) * rank + k0 + ch * 8;
            else {
                const int n = row - 2 * kU2Rows;  // 0..255: stream n / 64, column n % 64
                src = up_w + (size_t)((n / kU2D) * H + d0 + n % kU2D) * rank + k0 + ch * 8;
            }
            um_cp16(base + (size_t)row * kU2Pad + ch * 8, src);
        }
    };
    float acc_hi[2][4][4], acc_lo[2][4][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc_hi[a][b][e] = acc_lo[a][b][e] = 0.f;

    load_stage(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);
    load_stage(1, 1);
    asm volatile("cp.async.commit_group;\n" ::);
#pragma unroll 1
    for (int c = 0; c < nk; ++c) {
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        if (c + 2 < nk) load_stage((c + 2) % kU2Ring, c + 2);
        asm volatile("cp.async.commit_group;\n" ::);
        const half* ahi = stage(c % kU2Ring) + (size_t)(wr * 32) * kU2Pad;
        const half* alo = ahi + (size_t)kU2Rows * kU2Pad;
        const half* bt = stage(c % kU2Ring) + (size_t)(2 * kU2Rows + ws * kU2D + wd * 32) * kU2Pad;
#pragma unroll
        for (int kk = 0; kk < kU2KB / 16; ++kk) {
            uint32_t bf[4][2];
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                uint32_t r[4];
                um_ldsm4(r, bt + (size_t)(16 * p + (lane & 7) + ((lane >> 4) << 3)) * kU2Pad + kk * 16 + ((lane >> 3) & 1) * 8);
                bf[2 * p][0] = r[0]; bf[2 * p][1] = r[1]; bf[2 * p + 1][0] = r[2]; bf[2 * p + 1][1] = r[3];
            }
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                uint32_t ah[4], al[4];
                const size_t off = (size_t)(mt * 16 + (lane & 15)) * kU2Pad + kk * 16 + (lane >> 4) * 8;
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
    // the epilogue's x, rms and norm weights: thread = (row tid / 8, d = 32 c + 4 q .. + 3), their latency hides behind the shared-memory write
    const int row = tid >> 3, q = tid & 7, t = r0 + row;
    float4 xp[4][2];
    float rp[4];
    uint2 wp[4][2], bp[2];
    float4 pw;
    if (t < T) {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            rp[s] = rms[(size_t)t * 4 + s];
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                xp[s][c] = *reinterpret_cast<const float4*>(x + ((size_t)t * 4 + s) * H + d0 + 32 * c + 4 * q);
                wp[s][c] = *reinterpret_cast<const uint2*>(nw + (size_t)s * H + d0 + 32 * c + 4 * q);
            }
        }
        if (LAZY) {
            pw = *reinterpret_cast<const float4*>(inj_prev + (size_t)t * 4);
#pragma unroll
            for (int c = 0; c < 2; ++c) bp[c] = *reinterpret_cast<const uint2*>(block_out + (size_t)t * H + d0 + 32 * c + 4 * q);
        }
    }
    __syncthreads();
    // G = hi chain + lo chain, per stream into shared memory [stream][row][d], row stride kU2GsPad floats
    float* gs = reinterpret_cast<float*>(u2_smem);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int rr = wr * 32 + mt * 16 + g + 8 * h, cc = wd * 32 + nt * 8 + 2 * t4;
                float2 v;
                v.x = acc_lo[mt][nt][2 * h] + acc_hi[mt][nt][2 * h];
                v.y = acc_lo[mt][nt][2 * h + 1] + acc_hi[mt][nt][2 * h + 1];
                *reinterpret_cast<float2*>(gs + ((size_t)ws * kU2Rows + rr) * kU2GsPad + cc) = v;
            }
    __syncthreads();
    if (t >= T) return;
    const float inv = 1.f / 4.f;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const float rs = rp[s];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const float4 gv4 = *reinterpret_cast<const float4*>(gs + ((size_t)s * kU2Rows + row) * kU2GsPad + 32 * c + 4 * q);
            float4 xv = xp[s][c];
            if (LAZY) {
                const float pws = s == 0 ? pw.x : s == 1 ? pw.y : s == 2 ? pw.z : pw.w;
                const float2 b01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bp[c].x));
                const float2 b23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bp[c].y));
                xv.x = fmaf(b01.x, pws, xv.x);
                xv.y = fmaf(b01.y, pws, xv.y);
                xv.z = fmaf(b23.x, pws, xv.z);
                xv.w = fmaf(b23.y, pws, xv.w);
            }
            const float gg[4] = {gv4.x, gv4.y, gv4.z, gv4.w};
            const float xs[4] = {xv.x, xv.y, xv.z, xv.w};
            const float2 w01 = __half22float2(*reinterpret_cast<const half2*>(&wp[s][c].x));
            const float2 w23 = __half22float2(*reinterpret_cast<const half2*>(&wp[s][c].y));
            const float wv[4] = {w01.x, w01.y, w23.x, w23.y};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float gv = 1.f / (1.f + __expf(-gg[j]));
                acc[4 * c + j] += gv * xs[j] * rs * (1.f + wv[j]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        const __nv_bfloat162 lo2 = __floats2bfloat162_rn(acc[4 * c] * inv, acc[4 * c + 1] * inv);
        const __nv_bfloat162 hi2 = __floats2bfloat162_rn(acc[4 * c + 2] * inv, acc[4 * c + 3] * inv);
        uint2 o;
        o.x = *reinterpret_cast<const uint32_t*>(&lo2);
        o.y = *reinterpret_cast<const uint32_t*>(&hi2);
        *reinterpret_cast<uint2*>(y + (size_t)t * H + d0 + 32 * c + 4 * q) = o;
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// post_norm_kernel / norm_kernel and the inject product in one kernel (the Flash-Next shape: H = 2,560, hc = 4, inject [hc][hc H]).
//
// The library's inject product (K = hc H = 10,240, N = hc) is cublasLt algo 23 with split-K 20: split j is one chain of m16n8k16 MMAs over
// the 512 k of [512 j, 512 j + 512) from +0.0, the splits are added in order 0 .. 19, and the lo product is the same again over normed_lo;
// the output is the lo sum plus the hi sum (MEASURED against the library: tests/hcg_inject.rs). Stream z of a row holds splits 5 z .. 5 z + 4,
// so one CTA (32 rows, stream z) runs those five chains itself, straight from the normed rows it forms for the down product anyway, and
// the library's inject launches (a second full read of normed_hi and normed_lo, 168 MB a pre-mix) are gone. The kernel's other work is
// the unchanged norm_kernel / post_norm_kernel: pass 1 (the previous block's post-inject written back in place, the rms of the new values)
// has post_norm_kernel's thread-to-element map and sums, pass 2 forms normed = x * rms * (1 + nw) in its expressions and writes the fp16
// hi / lo split of it for the down GEMMs.
//
// The post-inject comes in three forms (NP posts, XW = the new streams are written back): NP = 1 with XW is the previous block's post written
// in place; NP = 2 with XW applies two blocks' posts in order, x1 = fma(b1, w1, x), x2 = fma(b2, w2, x1), and writes x2 (the two in-place
// hc_post launches in one pass); NP = 1 without XW leaves the streams untouched and applies the post to every value it reads (pass 1 and
// the stages of pass 2, whose block_out tile rides the ring beside the x tile), so that the later readers of the streams (down_x_kernel
// and up_mix_kernel with LAZY) apply the same post to the same values: the same fma on the same operands, never written to memory in between.
constexpr int kNiH = 2560, kNiBM = 16, kNiKB = 64, kNiPad = kNiKB + 8, kNiThreads = 256, kNiRing = 5, kNiAhead = kNiRing - 1;
constexpr size_t kNiSmemB = (size_t)kNiRing * 8 * kNiPad * sizeof(half);               // the 4 inject rows (and 4 zero rows) of the n8 tile
constexpr size_t kNiSmemX = (size_t)kNiRing * kNiBM * kNiKB * sizeof(float);
constexpr size_t kNiSmemA = (size_t)2 * 2 * kNiBM * kNiPad * sizeof(half);
constexpr size_t kNiSmemO = (size_t)kNiRing * kNiBM * kNiKB * sizeof(__nv_bfloat16);  // the block_out tiles of the on-the-fly form
constexpr size_t kNiSmemW = (size_t)kNiRing * kNiKB * sizeof(half);                    // the stage's 64 norm weights
constexpr size_t kNiSmem = kNiSmemB + kNiSmemX + kNiSmemA + kNiBM * sizeof(float) + kNiSmemO + kNiSmemW;

__device__ __forceinline__ void ni_ldsm2(uint32_t& r0, uint32_t& r1, const half* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}

// grid (4 ceil(T / 16)): CTA i is row tile i / 4, stream i % 4; block 256, dynamic smem kNiLaunchSmem (what two CTAs an SM leave of the 99 KB: a CTA's 16 rows of one stream are
// 164 KB, so the 96 CTAs in flight keep 15.7 MB of streams between pass 1 and pass 2 in L2). Q [2 (hi, lo)][20][T][4] gets the inject
// splits' sums.
constexpr size_t kNiLaunchSmem = 49152;
static_assert(kNiSmem <= kNiLaunchSmem, "the stages fit the launch's shared memory");
template <int NP, bool XW>
__global__ void __launch_bounds__(kNiThreads, 2)
    norm_inj_kernel(float* __restrict__ x, const __nv_bfloat16* __restrict__ block_out, const float* __restrict__ inj_prev,
                    const __nv_bfloat16* __restrict__ block_out2, const float* __restrict__ inj_prev2, const half* __restrict__ nw,
                    const half* __restrict__ inj_w, float* __restrict__ Q, float* __restrict__ rms_out, int T, float eps) {
    static_assert(NP >= 0 && NP <= 2 && (XW || NP == 1), "the forms: no post, one or two posts written back, one post applied on the fly");
    constexpr bool kLazy = NP == 1 && !XW;
    extern __shared__ __align__(16) unsigned char ni_smem[];
    half* sB = reinterpret_cast<half*>(ni_smem);
    float* sX = reinterpret_cast<float*>(ni_smem + kNiSmemB);
    half* sA = reinterpret_cast<half*>(ni_smem + kNiSmemB + kNiSmemX);
    float* sR = reinterpret_cast<float*>(ni_smem + kNiSmemB + kNiSmemX + kNiSmemA);
    __nv_bfloat16* sO = reinterpret_cast<__nv_bfloat16*>(ni_smem + kNiSmemB + kNiSmemX + kNiSmemA + kNiBM * sizeof(float));
    half* sW = reinterpret_cast<half*>(ni_smem + kNiSmemB + kNiSmemX + kNiSmemA + kNiBM * sizeof(float) + kNiSmemO);
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t4 = lane & 3;
    const int r0 = (int)(blockIdx.x >> 2) * kNiBM, s = blockIdx.x & 3;

    // pass 1: post-inject (in place, when XW) and the rms of (row, stream). Lane l of the warp holds, for j = 0..7, the partial sum of post_norm_kernel's
    // thread 32 j + l (its float4 at m = 32 j + l + 256 c for c = 0, 1, 2, m < 640, summed in c order); the warp butterfly of each j is
    // block_sum's warp sum, and the eight warp sums are added in warp order from 0.f.
    for (int lr = w; lr < kNiBM; lr += kNiThreads / 32) {
        const int t = r0 + lr;
        float rr = 0.f;
        if (t < T) {
            float* xr = x + ((size_t)t * 4 + s) * kNiH;
            float a[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) a[j] = 0.f;
            float wv = 0.f, wv2 = 0.f;
            if (NP >= 1) wv = inj_prev[(size_t)t * 4 + s];
            if (NP == 2) wv2 = inj_prev2[(size_t)t * 4 + s];
#pragma unroll
            for (int c = 0; c < 3; ++c) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int m = 32 * j + lane + 256 * c;
                    if (m >= kNiH / 4) continue;
                    float4 v = *reinterpret_cast<const float4*>(xr + 4 * m);
                    if (NP >= 1) {
                        const uint2 u = *reinterpret_cast<const uint2*>(block_out + (size_t)t * kNiH + 4 * m);
                        const float2 b01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
                        const float2 b23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
                        float4 n;
                        n.x = fmaf(b01.x, wv, v.x);
                        n.y = fmaf(b01.y, wv, v.y);
                        n.z = fmaf(b23.x, wv, v.z);
                        n.w = fmaf(b23.y, wv, v.w);
                        if (NP == 2) {
                            const uint2 u2 = *reinterpret_cast<const uint2*>(block_out2 + (size_t)t * kNiH + 4 * m);
                            const float2 c01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u2.x));
                            const float2 c23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u2.y));
                            n.x = fmaf(c01.x, wv2, n.x);
                            n.y = fmaf(c01.y, wv2, n.y);
                            n.z = fmaf(c23.x, wv2, n.z);
                            n.w = fmaf(c23.y, wv2, n.w);
                        }
                        if (XW) *reinterpret_cast<float4*>(xr + 4 * m) = n;
                        v = n;
                    }
                    a[j] += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
                }
            }
            float tot = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float v = a[j];
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                tot += v;
            }
            rr = rsqrtf(tot / (float)kNiH + eps);
            if (lane == 0) rms_out[(size_t)t * 4 + s] = rr;
        }
        if (lane == 0) sR[lr] = rr;
    }
    // the zero rows 4..7 of the inject B tile (never written by the stages)
    if (tid < kNiRing * 4 * (kNiPad / 8)) {
        const int slot = tid / (4 * (kNiPad / 8)), rem = tid % (4 * (kNiPad / 8));
        *reinterpret_cast<uint4*>(sB + ((size_t)slot * 8 + 4 + rem / (kNiPad / 8)) * kNiPad + (rem % (kNiPad / 8)) * 8) = make_uint4(0u, 0u, 0u, 0u);
    }
    __threadfence_block();
    __syncthreads();

    constexpr int kRow = 4 * kNiH;
    const half* iw = inj_w + (size_t)s * kNiH;
    // the post weight of the row this thread transforms (the on-the-fly form applies the post to the raw stage tiles)
    float wvr = 0.f;
    if (kLazy && r0 + (tid >> 4) < T) wvr = inj_prev[(size_t)(r0 + (tid >> 4)) * 4 + s];
    // stage st (k in [64 st, 64 st + 64) of the stream) into ring slot st % kNiRing: the four inject rows, the raw x tile, the stage's norm weights
    // and, on the fly, the block_out tile of the same columns
    auto issue = [&](int st) {
        const int slot = st % kNiRing, k0 = st * kNiKB;
        if (tid < 32) um_cp16(sB + ((size_t)slot * 8 + (tid >> 3)) * kNiPad + (tid & 7) * 8, iw + (size_t)(tid >> 3) * kRow + k0 + (tid & 7) * 8);
        if (tid >= 32 && tid < 32 + kNiKB / 8) um_cp16(sW + (size_t)slot * kNiKB + (tid - 32) * 8, nw + (size_t)s * kNiH + k0 + (tid - 32) * 8);
        const int r = tid >> 4, q = tid & 15, t = r0 + r;
        if (t < T) um_cp16(sX + (size_t)slot * kNiBM * kNiKB + r * kNiKB + q * 4, x + ((size_t)t * 4 + s) * kNiH + k0 + q * 4);
        if (kLazy && tid < kNiBM * kNiKB / 8) {
            const int ro = tid >> 3, ch = tid & 7;
            if (r0 + ro < T) um_cp16(sO + (size_t)slot * kNiBM * kNiKB + ro * kNiKB + ch * 8, block_out + (size_t)(r0 + ro) * kNiH + k0 + ch * 8);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    // raw stage st -> A tile (st & 1): the hi / lo columns of the normed rows, post_norm_kernel's expressions
    auto transform = [&](int st) {
        const int slot = st % kNiRing, r = tid >> 4, q = tid & 15, t = r0 + r;
        uint2 hi = make_uint2(0u, 0u), lo = make_uint2(0u, 0u);
        if (t < T) {
            float4 v = *reinterpret_cast<const float4*>(sX + (size_t)slot * kNiBM * kNiKB + r * kNiKB + q * 4);
            if (kLazy) {
                const uint2 u = *reinterpret_cast<const uint2*>(sO + (size_t)slot * kNiBM * kNiKB + r * kNiKB + q * 4);
                const float2 b01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
                const float2 b23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
                v.x = fmaf(b01.x, wvr, v.x);
                v.y = fmaf(b01.y, wvr, v.y);
                v.z = fmaf(b23.x, wvr, v.z);
                v.w = fmaf(b23.y, wvr, v.w);
            }
            const half* wp = sW + (size_t)slot * kNiKB + q * 4;
            const float2 w01 = __half22float2(*reinterpret_cast<const half2*>(wp));
            const float2 w23 = __half22float2(*reinterpret_cast<const half2*>(wp + 2));
            const float rr = sR[r];
            const float n0 = v.x * rr * (1.f + w01.x), n1 = v.y * rr * (1.f + w01.y);
            const float n2 = v.z * rr * (1.f + w23.x), n3 = v.w * rr * (1.f + w23.y);
            const float h0 = __half2float(__float2half_rn(n0)), h1 = __half2float(__float2half_rn(n1));
            const float h2 = __half2float(__float2half_rn(n2)), h3 = __half2float(__float2half_rn(n3));
            const half2 a = __floats2half2_rn(h0, h1), b = __floats2half2_rn(h2, h3);
            const half2 c = __floats2half2_rn(n0 - h0, n1 - h1), d = __floats2half2_rn(n2 - h2, n3 - h3);
            hi = make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
            lo = make_uint2(*reinterpret_cast<const uint32_t*>(&c), *reinterpret_cast<const uint32_t*>(&d));
        }
        half* ah = sA + (size_t)(st & 1) * 2 * kNiBM * kNiPad + r * kNiPad + q * 4;
        *reinterpret_cast<uint2*>(ah) = hi;
        *reinterpret_cast<uint2*>(ah + kNiBM * kNiPad) = lo;
    };

    float ih[4] = {0.f, 0.f, 0.f, 0.f}, il[4] = {0.f, 0.f, 0.f, 0.f};     // warp 0 (rows 0..15): the inject chains
    constexpr int kNK = kNiH / kNiKB;
    // kNiAhead stages are in flight: the group of stage st is the st-th commit (an empty group once the stages run out), so at the top of
    // iteration i the groups up to stage i + 1 have landed when at most kNiAhead - 2 are pending
#pragma unroll
    for (int st = 0; st < kNiAhead; ++st) issue(st);
    asm volatile("cp.async.wait_group %0;\n" ::"n"(kNiAhead - 1));
    __syncthreads();
    transform(0);
#pragma unroll 1
    for (int i = 0; i < kNK; ++i) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(kNiAhead - 2));
        __syncthreads();
        if (i + kNiAhead < kNK) issue(i + kNiAhead);
        else asm volatile("cp.async.commit_group;\n" ::);
        if (i + 1 < kNK) transform(i + 1);
        if (w == 0) {
            const half* a_hi = sA + (size_t)(i & 1) * 2 * kNiBM * kNiPad;
            const half* a_lo = a_hi + kNiBM * kNiPad;
            const half* bs = sB + (size_t)(i % kNiRing) * 8 * kNiPad;
#pragma unroll
            for (int kk = 0; kk < kNiKB / 16; ++kk) {
                uint32_t fh[4], fl[4], b0, b1;
                const size_t off = (size_t)(lane & 15) * kNiPad + kk * 16 + (lane >> 4) * 8;
                um_ldsm4(fh, a_hi + off);
                um_ldsm4(fl, a_lo + off);
                ni_ldsm2(b0, b1, bs + (size_t)(lane & 7) * kNiPad + kk * 16 + ((lane >> 3) & 1) * 8);
                um_mma(ih, fh, b0, b1);
                um_mma(il, fl, b0, b1);
            }
            if ((i & 7) == 7) {                                    // a 512-wide inject split ends: its sums leave, the chains restart
                const int jj = s * 5 + (i >> 3);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int row = r0 + g + (e >> 1) * 8, col = 2 * t4 + (e & 1);
                    if (row < T && col < 4) {
                        Q[(((size_t)0 * 20 + jj) * T + row) * 4 + col] = ih[e];
                        Q[(((size_t)1 * 20 + jj) * T + row) * 4 + col] = il[e];
                    }
                    ih[e] = il[e] = 0.f;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The down product from the streams (the Flash-Next shape: H = 2,560, hc = 4, rank = 320, K = hc H = 10,240), bit for bit what the library's
// two split-K calls and low_kernel leave behind, with the normed rows formed on the fly from the post-inject streams instead of read back
// from two 84 MB fp16 planes.
//
// The library's order (cublasLt algo 67, split-K 4, compute-type reduction; reproduced from the outside with three-leaf experiments and
// checked on random data, tests/hcg_down.rs): K is cut into stages of 64 k (four k16 units) and split z takes the stages z, z + 4, z + 8,
// ..; within a split three accumulators run, A = units 0 and 3 of every stage (unit 0 first), B = unit 1, C = unit 2, every unit one
// m16n8k16 MMA from the accumulator's running value, from +0.0; the split's sum adds the three in an order that depends on the 16-column
// group g = (n / 16) % 4 of the output column: (A + B) + C for g = 0 and 3, (A + C) + B for g = 1, A + (B + C) for g = 2 (fp32 adds); the
// splits are added in order, ((S0 + S1) + S2) + S3. The hi and lo planes of the normed rows each run all of that, and the output is the lo
// total plus the hi total (the second call's beta = 1). low_kernel's expression then turns it into the fp16 hi / lo planes of the low rows.
//
// A CTA is 32 rows by 64 columns, 8 warps (warp (wm, wn) = rows 16 wm.., columns 16 wn..), the whole K in the library's order: split by
// split, 40 stages each. The down weights and the stage's norm weights arrive through a cp.async ring; the stream tiles go global ->
// registers (three stages in flight) -> a pass that forms the fp16 hi / lo A tiles with post_norm_kernel's expressions, spread between the
// MMAs of the previous stage -> shared memory, and the MMAs read A and B with ldmatrix. Two CTAs share an SM (99 KB of shared memory and
// 64K registers between them); the five column CTAs of a row tile are adjacent in the grid, so the streams are read from DRAM once.
constexpr int kDxBM = 32, kDxBN = 64, kDxKB = 64, kDxThreads = 256, kDxRing = 3, kDxSplit = 4, kDxRank = 320;
constexpr int kDxK = 4 * kNiH, kDxStages = kDxK / kDxKB, kDxPerSplit = kDxStages / kDxSplit;
constexpr int kDxBPad = kDxKB + 8, kDxAPad = kDxKB + 8;
constexpr size_t kDxSmemB = (size_t)kDxRing * kDxBN * kDxBPad * sizeof(half);
constexpr size_t kDxSmemA = (size_t)2 * 2 * kDxBM * kDxAPad * sizeof(half);
constexpr size_t kDxSmemW = (size_t)kDxRing * kDxKB * sizeof(half);  // the stage's 64 norm weights
constexpr size_t kDxSmem = kDxSmemB + kDxSmemA + kDxSmemW;
static_assert(kNiH % kDxKB == 0 && kDxStages % kDxSplit == 0 && kDxRank % kDxBN == 0, "stages of one stream, a whole number of stages a split");
static_assert(2 * (kDxSmem + 1024) <= 100 * 1024, "two CTAs an SM");

// One thread's raw inputs of a stage: its two float4 of the streams and, in the on-the-fly form, the matching bf16 block_out values.
struct DxRaw {
    float4 v[2];
    uint2 b[2];
};

// A 32-bit global load into any register. The block_out values were loaded as a 64-bit pair, which needs an even register pair, while the
// values are used one word at a time from registers the loop keeps for them: the compiler loaded into a temporary pair and moved the words
// into place at once, so every stage waited for the load it had just issued. Word loads land where they are used.
__device__ __forceinline__ uint32_t dx_ld32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
    return v;
}

// grid (kDxRank / kDxBN, padded / kDxBM), block 256, dynamic smem kDxSmem. low_hi / low_lo [padded][kDxRank]. LAZY: the streams hold the values
// BEFORE the post-inject of (block_out, inj_prev), which is applied to every value read, fmaf(bf16, weight, x) as hc_post writes it.
template <bool LAZY>
__global__ void __launch_bounds__(kDxThreads, 2)
    down_x_kernel(const float* __restrict__ x, const __nv_bfloat16* __restrict__ block_out, const float* __restrict__ inj_prev,
                  const float* __restrict__ rms, const half* __restrict__ nw, const half* __restrict__ down_w, half* __restrict__ low_hi,
                  half* __restrict__ low_lo, int T) {
    extern __shared__ __align__(16) unsigned char dx_smem[];
    half* sB = reinterpret_cast<half*>(dx_smem);
    half* sA = reinterpret_cast<half*>(dx_smem + kDxSmemB);
    half* sW = reinterpret_cast<half*>(dx_smem + kDxSmemB + kDxSmemA);
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t4 = lane & 3;
    const int wm = w & 1, wn = w >> 1;
    const int col0 = blockIdx.x * kDxBN, row0 = blockIdx.y * kDxBM;

    // stage j of the loop: split j / 40, its (j % 40)-th stage, which is stage z + 4 i of the 160
    auto stage_of = [](int j) { return j / kDxPerSplit + kDxSplit * (j % kDxPerSplit); };
    // the down weights of stage j into ring slot j % 3, and the stage's 64 norm weights (norm_w [hc * H]: k0 is the stream's offset too);
    // neither commits: a group is a loop iteration's loads (B of stage i + 2, W of stage i + 3), so B has two iterations to land
    auto issue_b = [&](int j) {
        if (j < kDxStages) {
            const int slot = j % kDxRing, k0 = stage_of(j) * kDxKB;
#pragma unroll
            for (int c = 0; c < 512 / kDxThreads; ++c) {
                const int idx = tid + c * kDxThreads, n = idx >> 3, q = idx & 7;
                um_cp16(sB + ((size_t)slot * kDxBN + n) * kDxBPad + 8 * q, down_w + (size_t)(col0 + n) * kDxK + k0 + 8 * q);
            }
        }
    };
    auto issue_w = [&](int j) {
        if (j < kDxStages && tid < 8) um_cp16(sW + (size_t)(j % kDxRing) * kDxKB + 8 * tid, nw + stage_of(j) * kDxKB + 8 * tid);
    };
    // thread (row r = tid / 8, float4 columns kq and kq + 8 of the stage's 64): the raw streams of stage j and, on the fly, the block_out
    // values of the same columns
    const int tr = tid >> 3, tq = tid & 7, trow = row0 + tr;
    auto load_raw = [&](int j, DxRaw& rw) {
        if (j < kDxStages) {
            const int k0 = stage_of(j) * kDxKB, stream = k0 / kNiH, d0 = k0 - stream * kNiH;
            const float* src = x + ((size_t)(trow < T ? trow : 0) * 4 + stream) * kNiH + d0;
#pragma unroll
            for (int i = 0; i < 2; ++i) rw.v[i] = trow < T ? *reinterpret_cast<const float4*>(src + 4 * (tq + 8 * i)) : make_float4(0.f, 0.f, 0.f, 0.f);
            if (LAZY) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    // from a valid row whatever the row: a select on the registers where they are used keeps the load's latency out of the loop
                    const uint32_t* bp = reinterpret_cast<const uint32_t*>(block_out + (size_t)(trow < T ? trow : 0) * kNiH + d0 + 4 * (tq + 8 * i));
                    rw.b[i].x = dx_ld32(bp);
                    rw.b[i].y = dx_ld32(bp + 1);
                }
            }
        }
    };
    // this thread's row's four rms values: stage j's is the one of its stream (and, on the fly, the four post weights)
    const float4 rms4 = trow < T ? *reinterpret_cast<const float4*>(rms + (size_t)trow * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    float4 pw4 = make_float4(0.f, 0.f, 0.f, 0.f);
    if (LAZY && trow < T) pw4 = *reinterpret_cast<const float4*>(inj_prev + (size_t)trow * 4);
    // raw stage j -> A tile (j & 1), part i of 2 (float4 i of this thread's two): post_norm_kernel's expressions
    auto transform_part = [&](int j, int i, const DxRaw& rw) {
        float4 v = rw.v[i];
        const int stream = stage_of(j) * kDxKB / kNiH;
        if (LAZY) {
            const float pw = stream == 0 ? pw4.x : stream == 1 ? pw4.y : stream == 2 ? pw4.z : pw4.w;
            // the block_out values are loaded from a valid row whatever the row: a select on the registers here, not after the load, keeps the
            // load's latency out of the loop
            const uint2 bb = trow < T ? rw.b[i] : make_uint2(0u, 0u);
            const float2 b01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bb.x));
            const float2 b23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bb.y));
            v.x = fmaf(b01.x, pw, v.x);
            v.y = fmaf(b01.y, pw, v.y);
            v.z = fmaf(b23.x, pw, v.z);
            v.w = fmaf(b23.y, pw, v.w);
        }
        const float rr = stream == 0 ? rms4.x : stream == 1 ? rms4.y : stream == 2 ? rms4.z : rms4.w;
        const half* wp = sW + (size_t)(j % kDxRing) * kDxKB;
        half* ah = sA + (size_t)(j & 1) * 2 * kDxBM * kDxAPad + (size_t)tr * kDxAPad;
        half* al = ah + (size_t)kDxBM * kDxAPad;
        const int c = 4 * (tq + 8 * i);
        const float2 w01 = __half22float2(*reinterpret_cast<const half2*>(wp + c));
        const float2 w23 = __half22float2(*reinterpret_cast<const half2*>(wp + c + 2));
        const float n0 = v.x * rr * (1.f + w01.x), n1 = v.y * rr * (1.f + w01.y);
        const float n2 = v.z * rr * (1.f + w23.x), n3 = v.w * rr * (1.f + w23.y);
        const float h0 = __half2float(__float2half_rn(n0)), h1 = __half2float(__float2half_rn(n1));
        const float h2 = __half2float(__float2half_rn(n2)), h3 = __half2float(__float2half_rn(n3));
        const half2 a = __floats2half2_rn(h0, h1), b = __floats2half2_rn(h2, h3);
        const half2 cc = __floats2half2_rn(n0 - h0, n1 - h1), d = __floats2half2_rn(n2 - h2, n3 - h3);
        *reinterpret_cast<uint2*>(ah + c) = make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
        *reinterpret_cast<uint2*>(al + c) = make_uint2(*reinterpret_cast<const uint32_t*>(&cc), *reinterpret_cast<const uint32_t*>(&d));
    };

    float acc_hi[3][2][4], acc_lo[3][2][4], tot_hi[2][4], tot_lo[2][4];
#pragma unroll
    for (int c = 0; c < 3; ++c)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc_hi[c][j][e] = acc_lo[c][j][e] = 0.f;
    const int grp = wn & 3;  // the 16-column group of this warp's two n8 tiles

    auto mma_stage = [&](int j, const DxRaw& nv) {
        const int slot = j % kDxRing;
        const half* ahi = sA + (size_t)(j & 1) * 2 * kDxBM * kDxAPad;
        const half* alo = ahi + (size_t)kDxBM * kDxAPad;
        const half* bs = sB + (size_t)slot * kDxBN * kDxBPad + (size_t)(wn * 16) * kDxBPad;
        // the units in the order 0, 1, 3, 2: every accumulator's next MMA is eight or more MMAs behind its last (chain A takes units 0 and 3,
        // so with 0, 1, 2, 3 its unit 3 was followed at once by the next stage's unit 0). Each chain still sees its units in the library's order.
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int u = p == 2 ? 3 : p == 3 ? 2 : p;
            uint32_t fh[4], fl[4], r4[4];
            const size_t off = (size_t)(wm * 16 + (lane & 15)) * kDxAPad + u * 16 + (lane >> 4) * 8;
            um_ldsm4(fh, ahi + off);
            um_ldsm4(fl, alo + off);
            um_ldsm4(r4, bs + (size_t)((lane & 7) + ((lane >> 4) << 3)) * kDxBPad + u * 16 + ((lane >> 3) & 1) * 8);
            const int c = (u == 3) ? 0 : u;
            um_mma(acc_hi[c][0], fh, r4[0], r4[1]);
            um_mma(acc_lo[c][0], fl, r4[0], r4[1]);
            um_mma(acc_hi[c][1], fh, r4[2], r4[3]);
            um_mma(acc_lo[c][1], fl, r4[2], r4[3]);
            // the next stage's A tile (half of this thread's elements after the second and the fourth unit) between the MMAs
            if ((p & 1) && j + 1 < kDxStages) transform_part(j + 1, p >> 1, nv);
        }
    };
    // the split's three accumulators added in the order of the output column's 16-column group, folded into the running total
    auto fold_split = [&](int z) {
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float sh, sl;
                {
                    const float a = acc_hi[0][j][e], b = acc_hi[1][j][e], c = acc_hi[2][j][e];
                    sh = grp == 1 ? __fadd_rn(__fadd_rn(a, c), b) : grp == 2 ? __fadd_rn(a, __fadd_rn(b, c)) : __fadd_rn(__fadd_rn(a, b), c);
                }
                {
                    const float a = acc_lo[0][j][e], b = acc_lo[1][j][e], c = acc_lo[2][j][e];
                    sl = grp == 1 ? __fadd_rn(__fadd_rn(a, c), b) : grp == 2 ? __fadd_rn(a, __fadd_rn(b, c)) : __fadd_rn(__fadd_rn(a, b), c);
                }
                tot_hi[j][e] = z == 0 ? sh : __fadd_rn(tot_hi[j][e], sh);
                tot_lo[j][e] = z == 0 ? sl : __fadd_rn(tot_lo[j][e], sl);
            }
#pragma unroll
        for (int c = 0; c < 3; ++c)
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc_hi[c][j][e] = acc_lo[c][j][e] = 0.f;
    };

    // pipeline: the loads of iteration i are one cp.async group (B of stage i + 2, W of stage i + 3), so at the top of iteration j the groups
    // up to j - 2 have landed (B of stage j, W of stage j + 1) and the previous iteration's group may still fly. The raw streams of stages
    // j + 1, j + 2, j + 3 sit in three register sets used round robin (the loop is unrolled by three: no copies between sets): iteration j
    // transforms stage j + 1 from set (j + 1) % 3 between its MMAs, then loads stage j + 4 into that set.
    DxRaw rb[3];
    load_raw(0, rb[0]);
    load_raw(1, rb[1]);
    load_raw(2, rb[2]);
    issue_b(0);
    issue_w(0);
    issue_w(1);
    asm volatile("cp.async.commit_group;\n" ::);
    issue_b(1);
    issue_w(2);
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; ++i) transform_part(0, i, rb[0]);
    load_raw(3, rb[0]);
#pragma unroll 1
    for (int j0 = 0; j0 < kDxStages; j0 += 3) {
#pragma unroll
        for (int u = 0; u < 3; ++u) {
            const int j = j0 + u;
            if (j < kDxStages) {
                asm volatile("cp.async.wait_group 1;\n" ::);
                __syncthreads();  // A tile j (transformed last iteration) is visible, ring slot (j + 2) % 3 is free
                issue_b(j + 2);
                issue_w(j + 3);
                asm volatile("cp.async.commit_group;\n" ::);
                mma_stage(j, rb[(u + 1) % 3]);
                load_raw(j + 4, rb[(u + 1) % 3]);
                if ((j + 1) % kDxPerSplit == 0) fold_split(j / kDxPerSplit);
            }
        }
    }
    // output = lo total + hi total, then low_kernel's expressions
    const float inv = 1.f / 4.f;
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = row0 + wm * 16 + g + 8 * h, col = col0 + wn * 16 + 8 * j + 2 * t4;
            float hv[2], lv[2];
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const float L = __fadd_rn(tot_lo[j][2 * h + q], tot_hi[j][2 * h + q]);
                const float v = L * inv;
                const float sv = v / (1.f + __expf(-v));
                const float hh = __half2float(__float2half_rn(sv));
                hv[q] = sv;
                lv[q] = sv - hh;
            }
            *reinterpret_cast<half2*>(low_hi + (size_t)row * kDxRank + col) = __floats2half2_rn(hv[0], hv[1]);
            *reinterpret_cast<half2*>(low_lo + (size_t)row * kDxRank + col) = __floats2half2_rn(lv[0], lv[1]);
        }
}

// the inject product's output from the 20 splits' sums: the hi sum and the lo sum in split order, the lo sum plus the hi sum, then
// low_kernel's expression of inj. One thread a (row, column).
__global__ void __launch_bounds__(256) inj_reduce_kernel(const float* __restrict__ Q, float* __restrict__ inj, int T, int hc) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= T * 4) return;
    const int t = i >> 2, c = i & 3;
    float hi = Q[((size_t)0 * 20 * T + t) * 4 + c], lo = Q[((size_t)1 * 20 * T + t) * 4 + c];
    for (int j = 1; j < 20; ++j) {
        hi += Q[(((size_t)0 * 20 + j) * T + t) * 4 + c];
        lo += Q[(((size_t)1 * 20 + j) * T + t) * 4 + c];
    }
    const float inv = 1.f / (float)hc;
    inj[i] = 2.f / (1.f + __expf(-(lo + hi) * inv));
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

// The previous blocks' post-injects a pre-mix takes (hc = 4): `block` / `inj` is one (the streams become streams + bf16(block) * inj per
// stream, in place), `block2` / `inj2` a second one applied after it (written back as well), `lazy` applies the first to every value the
// pre-mix reads and leaves the streams as they were. The second and the lazy form exist for the Flash-Next shape only.
struct PostArgs {
    const void* block = nullptr;
    const float* inj = nullptr;
    const void* block2 = nullptr;
    const float* inj2 = nullptr;
    bool lazy = false;
};

// y [T][H] bf16 (+ inj [T][hc] f32 when inject_w is non-null) from streams [T][hc][H] f32. scratch holds
// gb_hcg_scratch_bytes(T, H, hc, rank). H % 4 == 0, rank >= 1. The weight operands are read exactly at their declared
// stride `rank` (down [rank][hc*H], up [hc*H][rank], inject [hc][hc*H]); no read extends past a weight buffer.
static int hcg_pre(const void* streams, const void* norm_w, const void* down_w, const void* up_w,
                   const void* inject_w, void* y_out, float* inj_out, void* scratch, int H, int hc, int rank,
                   float eps, int T, void* stream, bool fast, const PostArgs& post = PostArgs{}) {
    if (T < 1 || H % 4 || hc < 1 || rank < 1 || !scratch || (inject_w != nullptr) != (inj_out != nullptr))
        return (int)cudaErrorInvalidValue;
    // the fused hc_post: in place on `streams` (or on the fly), hc = 4, H up to 3 x 1024
    if ((post.block != nullptr) != (post.inj != nullptr) || (post.block && (hc != 4 || H > 3 * 1024))) return (int)cudaErrorInvalidValue;
    if ((post.block2 != nullptr) != (post.inj2 != nullptr) || (post.block2 && (!post.block || post.lazy))) return (int)cudaErrorInvalidValue;
    if (post.lazy && !post.block) return (int)cudaErrorInvalidValue;
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

    // the Flash-Next shape with inject rows: norm_inj_kernel forms the inject product's splits (no library inject launches) and the rms, and
    // down_x_kernel forms the normed rows again on the fly for the down product (no fp16 hi / lo planes through DRAM, no library down GEMMs)
    const bool ni = lt && inject_w && hc == 4 && H == kNiH;
    // A fixed 64-row GEMM tile gives the same accumulation for a row in every decode window.
    // Initialize only the unused tail of the final tile; valid rows are overwritten by norm_kernel.
    if (padded != T && !ni) {
        if (cudaMemsetAsync(normed_hi + (size_t)T * K, 0, (size_t)(padded - T) * K * sizeof(half), st) != cudaSuccess)
            return (int)cudaErrorUnknown;
        if (cudaMemsetAsync(normed_lo + (size_t)T * K, 0, (size_t)(padded - T) * K * sizeof(half), st) != cudaSuccess)
            return (int)cudaErrorUnknown;
    }
    int rc = 0;
    // one CTA a (16 rows, stream), the four streams of a row tile adjacent in the grid: they all read the tile's rows of block_out, which
    // the first of them brings from DRAM and the others find in L2 (stream-major order read it from DRAM four times, 20 MB each)
    const dim3 ni_grid(((T + kNiBM - 1) / kNiBM) * 4, 1);
    if (ni) {
        if (post.block2)
            norm_inj_kernel<2, true><<<ni_grid, kNiThreads, kNiLaunchSmem, st>>>(
                (float*)streams, (const __nv_bfloat16*)post.block, post.inj, (const __nv_bfloat16*)post.block2, post.inj2, (const half*)norm_w,
                (const half*)inject_w, G, rms, T, eps);
        else if (post.lazy)
            norm_inj_kernel<1, false><<<ni_grid, kNiThreads, kNiLaunchSmem, st>>>(
                (float*)streams, (const __nv_bfloat16*)post.block, post.inj, nullptr, nullptr, (const half*)norm_w, (const half*)inject_w, G, rms, T,
                eps);
        else if (post.block)
            norm_inj_kernel<1, true><<<ni_grid, kNiThreads, kNiLaunchSmem, st>>>(
                (float*)streams, (const __nv_bfloat16*)post.block, post.inj, nullptr, nullptr, (const half*)norm_w, (const half*)inject_w, G, rms, T,
                eps);
        else
            norm_inj_kernel<0, true><<<ni_grid, kNiThreads, kNiLaunchSmem, st>>>(
                (float*)streams, nullptr, nullptr, nullptr, nullptr, (const half*)norm_w, (const half*)inject_w, G, rms, T, eps);
    } else if (post.lazy || post.block2) {
        return (int)cudaErrorInvalidValue;  // the on-the-fly and the two-post forms exist for the Flash-Next shape only
    } else if (post.block) {
        post_norm_kernel<4, 3><<<T, 256, 0, st>>>((float*)streams, (const __nv_bfloat16*)post.block, post.inj, (const half*)norm_w, normed_hi,
                                                  normed_lo, rms, H, eps);
    } else {
        norm_kernel<<<T, 256, 0, st>>>((const float*)streams, (const half*)norm_w, normed_hi, normed_lo, rms, H, hc, eps);
    }
    rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    if (ni) {
        static bool dxattr[2] = {false, false};
        const int li = post.lazy ? 1 : 0;
        if (!dxattr[li]) {
            const auto fn = post.lazy ? down_x_kernel<true> : down_x_kernel<false>;
            if (cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kDxSmem) != cudaSuccess) return (int)cudaErrorUnknown;
            dxattr[li] = true;
        }
        const dim3 dg(kDxRank / kDxBN, padded / kDxBM);
        if (post.lazy)
            down_x_kernel<true><<<dg, kDxThreads, kDxSmem, st>>>((const float*)streams, (const __nv_bfloat16*)post.block, post.inj, rms,
                                                                 (const half*)norm_w, (const half*)down_w, low_hi, low_lo, T);
        else
            down_x_kernel<false><<<dg, kDxThreads, kDxSmem, st>>>((const float*)streams, nullptr, nullptr, rms, (const half*)norm_w,
                                                                  (const half*)down_w, low_hi, low_lo, T);
        if ((rc = GB_CHECK_LAUNCH())) return rc;
        inj_reduce_kernel<<<(T * 4 + 255) / 256, 256, 0, st>>>(G, inj_out, T, hc);
        if ((rc = GB_CHECK_LAUNCH())) return rc;
    } else {
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
        if (inject_w && cudaMemcpyAsync(inj_out, inj_pad, (size_t)T * hc * sizeof(float), cudaMemcpyDeviceToDevice, st) != cudaSuccess) {
            return (int)cudaErrorUnknown;
        }
    }
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
        static bool attr[2] = {false, false};
        const int li = post.lazy ? 1 : 0;
        if (!attr[li]) {
            const auto fn = post.lazy ? up_mix_kernel<true> : up_mix_kernel<false>;
            if (cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kU2Smem) != cudaSuccess) return (int)cudaErrorUnknown;
            attr[li] = true;
        }
        const dim3 ug(H / kU2D, padded / kU2Rows);
        if (post.lazy)
            up_mix_kernel<true><<<ug, kU2Threads, kU2Smem, st>>>(low_hi, low_lo, (const half*)up_w, (const float*)streams,
                                                                 (const __nv_bfloat16*)post.block, post.inj, rms, (const half*)norm_w,
                                                                 (__nv_bfloat16*)y_out, T, H, rank);
        else
            up_mix_kernel<false><<<ug, kU2Threads, kU2Smem, st>>>(low_hi, low_lo, (const half*)up_w, (const float*)streams, nullptr, nullptr, rms,
                                                                  (const half*)norm_w, (__nv_bfloat16*)y_out, T, H, rank);
    }
    return GB_CHECK_LAUNCH();
}

// the large-call path where the shape allows it, the 64-row slab loop otherwise
GB_EXPORT int gb_hcg_pre(const void* streams, const void* norm_w, const void* down_w, const void* up_w,
                         const void* inject_w, void* y_out, float* inj_out, void* scratch, int H, int hc, int rank,
                         float eps, int T, void* stream) {
    return hcg_pre(streams, norm_w, down_w, up_w, inject_w, y_out, inj_out, scratch, H, hc, rank, eps, T, stream, true);
}

// gb_hcg_pre after the hc_post of the previous block: `streams` first becomes streams + bf16(post_block) * post_inj (per stream, the
// in-place hc_post), then the pre-mix of the new streams. hc = 4, H <= 3,072.
GB_EXPORT int gb_hcg_pre_post(void* streams, const void* post_block, const float* post_inj, const void* norm_w, const void* down_w,
                              const void* up_w, const void* inject_w, void* y_out, float* inj_out, void* scratch, int H, int hc,
                              int rank, float eps, int T, void* stream) {
    if (!post_block || !post_inj) return (int)cudaErrorInvalidValue;
    PostArgs post;
    post.block = post_block;
    post.inj = post_inj;
    return hcg_pre(streams, norm_w, down_w, up_w, inject_w, y_out, inj_out, scratch, H, hc, rank, eps, T, stream, true, post);
}

// gb_hcg_pre_post with two blocks' post-injects: `streams` first becomes (streams + bf16(block1) * inj1) + bf16(block2) * inj2 per stream, the
// two in-place hc_post launches in one pass, written back. The Flash-Next shape only (hidden 2,560, four streams, rank 320, inject rows).
GB_EXPORT int gb_hcg_pre_post2(void* streams, const void* block1, const float* inj1, const void* block2, const float* inj2, const void* norm_w,
                               const void* down_w, const void* up_w, const void* inject_w, void* y_out, float* inj_out, void* scratch, int H,
                               int hc, int rank, float eps, int T, void* stream) {
    if (!block1 || !inj1 || !block2 || !inj2) return (int)cudaErrorInvalidValue;
    PostArgs post;
    post.block = block1;
    post.inj = inj1;
    post.block2 = block2;
    post.inj2 = inj2;
    return hcg_pre(streams, norm_w, down_w, up_w, inject_w, y_out, inj_out, scratch, H, hc, rank, eps, T, stream, true, post);
}

// gb_hcg_pre_post without writing the post-injected streams back: the pre-mix reads streams + bf16(post_block) * post_inj (the same fma, on
// every value it reads) and leaves `streams` as it was. The caller owes the write (a later pre-mix or hc_post applies the same post again,
// on the streams as they stand). The Flash-Next shape only (hidden 2,560, four streams, rank 320, inject rows).
GB_EXPORT int gb_hcg_pre_post_lazy(const void* streams, const void* post_block, const float* post_inj, const void* norm_w, const void* down_w,
                                   const void* up_w, const void* inject_w, void* y_out, float* inj_out, void* scratch, int H, int hc,
                                   int rank, float eps, int T, void* stream) {
    if (!post_block || !post_inj) return (int)cudaErrorInvalidValue;
    PostArgs post;
    post.block = post_block;
    post.inj = post_inj;
    post.lazy = true;
    return hcg_pre(const_cast<void*>(streams), norm_w, down_w, up_w, inject_w, y_out, inj_out, scratch, H, hc, rank, eps, T, stream, true, post);
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
