// HCX: fused multi-HyperConnection pre-mix for Flash-Next (Qwen4_Exp). Original REXL3 code, Victor Cruz.
//
// Same math as gb_hc_pre / gb_hc_head (mhc.cu), restructured so the 13 MB of fp16 mixer weights a call are read once
// for every row of the window and the per-stream RMS no longer needs its own pass:
//   normed[s][d] = x[s][d] * rms_s * (1 + nw[s][d]),   rms_s = rsqrt(mean_d x[s][d]^2 + eps)
//   low[r]  = silu(  (1/hc) sum_s rms_s * P[r][s] ),   P[r][s] = sum_d down[r][s*H + d] * (1 + nw[s*H + d]) * x[s][d]
//   inj[q]  = 2 sigmoid((1/hc) sum_s rms_s * P[rank + q][s])     (inject rows ride along as extra rows of down)
//   y[d]    = (1/hc) sum_s sigmoid(up[s*H + d] . low) * normed[s][d]
// K1 computes P (one warp a (row, stream), weights read once for all rows of the window) and the sums of squares;
// K2 turns them into low / inj in every block and applies up. fp32 throughout, weights fp16 as stored.
//
// Decode fast path (k1u/k2t, k2c for other shapes): the same arithmetic, bit for bit, tuned for the 1..8-row window:
// - k1u: a warp owns nine rows of one stream, so the window's x rows are loaded once a chunk for nine weight rows (the
//   reference k1 re-read them per row: at 7 rows it was bound by the x loads, not by DRAM), and the 32-wide warp
//   reductions run as one reduce-scatter (scatter_sum32, same xor tree as warp_sum);
// - k2c gives each block 16 warps (512 threads) covering 16 columns, so the k2 grid halves (320 -> 160 blocks) with
//   the reference's per-column body untouched;
// - k2t (hc = 4, rank <= 512, the Flash-Next shape) replaces k2c: one warp owns four columns, the low chunks are read from
//   shared memory once for the four streams, and the 4 x rows per-lane partial sums are reduce-scattered (same xor tree as
//   warp_sum) so the exp/divide tail runs once a warp instead of once a (stream, row) per lane; see the block comment above it;
// - hc_post is regridded over the whole [t, s, d] volume (elementwise, so any grid gives the same bits).
// - pre_post (decode, hc = 4, rank <= 512): the previous block's hc_post rides in k1u and k2t instead of a launch of its own. Both kernels
//   read x' = fma(bf16(block_out), inj_prev, x), the value hc_post_kernel writes, wherever they read the streams; k1u never writes (its
//   blocks of one stream all read the same x), k2t writes x' back from the one lane that owns each element once k1u has completed.
// - with the gbp_pdl::kHcx bit of the PDL mask (pdl.cuh) k1u, k2t (and hc_post, fn_mhc.cu) launch with the
//   programmatic-dependent-launch attribute: each triggers its successor at its start and waits for its predecessor before it
//   reads anything an earlier kernel wrote, after issuing what depends on nothing (k1u its first weight chunk, k2t its first
//   column's weights). Timing only: with the bit off every launch is a plain launch and the calls are no-ops.
// The fast path is unconditional: the gate needs the bit-exact fast kernels always on. The reference launches are
// still reachable through gb_hcx_pre_ref / gb_hcx_pre_z_ref (used by the bit-exact GPU tests in tests/hcx_perf.rs).
#include "common.cuh"
#include "pdl.cuh"

namespace hcx {

constexpr int TMAX = 8;   // rows a pass (more rows loop in passes)

__device__ __forceinline__ void h8(const half* p, float (&v)[8]) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
    const half2* h = reinterpret_cast<const half2*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(h[i]);
        v[2 * i] = f.x;
        v[2 * i + 1] = f.y;
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float dot8_acc(float acc, const float (&w)[8], const float4 a, const float4 b) {
    acc += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
    return acc;
}

__device__ __forceinline__ void dec_half8(const uint4 u, float (&v)[8]) {
    const half2* h = reinterpret_cast<const half2*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(h[i]);
        v[2 * i] = f.x;
        v[2 * i + 1] = f.y;
    }
}

// 32 values a lane in, lane l out: warp_sum of element l (the sum over the 32 lanes of v[l]), same pairing tree as warp_sum.
__device__ __forceinline__ float scatter_sum32(const float (&v)[32], int lane) {
    constexpr unsigned FULL = 0xffffffffu;
    float b[16], c[8], d[4], e[2];
    {
        const bool hi = lane & 16;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const float mine = hi ? v[k + 16] : v[k];
            const float send = hi ? v[k] : v[k + 16];
            b[k] = mine + __shfl_xor_sync(FULL, send, 16);
        }
    }
    {
        const bool hi = lane & 8;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float mine = hi ? b[k + 8] : b[k];
            const float send = hi ? b[k] : b[k + 8];
            c[k] = mine + __shfl_xor_sync(FULL, send, 8);
        }
    }
    {
        const bool hi = lane & 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float mine = hi ? c[k + 4] : c[k];
            const float send = hi ? c[k] : c[k + 4];
            d[k] = mine + __shfl_xor_sync(FULL, send, 4);
        }
    }
    {
        const bool hi = lane & 2;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const float mine = hi ? d[k + 2] : d[k];
            const float send = hi ? d[k] : d[k + 2];
            e[k] = mine + __shfl_xor_sync(FULL, send, 2);
        }
    }
    const bool hi = lane & 1;
    const float mine = hi ? e[1] : e[0];
    const float send = hi ? e[0] : e[1];
    return mine + __shfl_xor_sync(FULL, send, 1);
}

// K1. Jobs: (s, r) for r < R (rank down rows, then ninj inject rows), r fastest; then (t, s) sum-of-squares jobs.
// P [T][R][hc] fp32, ss [T][hc] fp32. Rows t0 .. t0 + nt - 1 of x [*, hc, H] fp32.
__global__ void __launch_bounds__(128) k1_kernel(const float* __restrict__ x, const half* __restrict__ nw,
                                                 const half* __restrict__ down, const half* __restrict__ inj,
                                                 float* __restrict__ P, float* __restrict__ ss, int t0, int nt, int H,
                                                 int hc, int rank, int ninj) {
    const int lane = threadIdx.x & 31;
    const int job = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int R = rank + ninj;
    const size_t HC = (size_t)hc * H;
    if (job < R * hc) {
        const int s = job / R, r = job % R;
        const half* w = (r < rank ? down + (size_t)r * HC : inj + (size_t)(r - rank) * HC) + (size_t)s * H;
        const half* n = nw + (size_t)s * H;
        float acc[TMAX];
#pragma unroll
        for (int t = 0; t < TMAX; ++t) acc[t] = 0.f;
        for (int d = 8 * lane; d < H; d += 256) {
            float wv[8], nv[8];
            h8(w + d, wv);
            h8(n + d, nv);
#pragma unroll
            for (int i = 0; i < 8; ++i) wv[i] *= 1.f + nv[i];
#pragma unroll
            for (int t = 0; t < TMAX; ++t) {
                if (t < nt) {
                    const float* xr = x + (size_t)(t0 + t) * HC + (size_t)s * H + d;
                    const float4 a = __ldg(reinterpret_cast<const float4*>(xr));
                    const float4 b = __ldg(reinterpret_cast<const float4*>(xr + 4));
                    acc[t] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x + wv[5] * b.y +
                              wv[6] * b.z + wv[7] * b.w;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TMAX; ++t) {
            if (t < nt) {
                const float v = warp_sum(acc[t]);
                if (lane == 0) P[((size_t)t * R + r) * hc + s] = v;
            }
        }
        return;
    }
    const int q = job - R * hc;
    if (q >= nt * hc) return;
    const int t = q / hc, s = q % hc;
    const float* xr = x + (size_t)(t0 + t) * HC + (size_t)s * H;
    float a = 0.f;
    for (int d = 4 * lane; d < H; d += 128) {
        const float4 v = __ldg(reinterpret_cast<const float4*>(xr + d));
        a += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    a = warp_sum(a);
    if (lane == 0) ss[t * hc + s] = a;
}

// K2. Block: 8 warps, one hidden column d a warp. Every block first rebuilds low [nt][rank] in shared memory
// (block 0 also writes inj [nt][ninj]). y [*, H] bf16 rows t0 .. t0 + nt - 1.
__global__ void __launch_bounds__(256) k2_kernel(const float* __restrict__ x, const half* __restrict__ nw,
                                                 const half* __restrict__ up, const float* __restrict__ P,
                                                 const float* __restrict__ ss, __nv_bfloat16* __restrict__ y,
                                                 float* __restrict__ inj_out, int t0, int nt, int H, int hc, int rank,
                                                 int ninj, float eps, float* __restrict__ dbg_z) {
    extern __shared__ float s_low[];  // [nt][rank]
    __shared__ float s_rms[TMAX * 8];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int R = rank + ninj;
    const float inv_hc = 1.f / (float)hc;
    if (tid < nt * hc) s_rms[tid] = rsqrtf(ss[tid] / (float)H + eps);
    __syncthreads();
    for (int i = tid; i < nt * rank; i += blockDim.x) {
        const int t = i / rank, r = i % rank;
        const float* p = P + ((size_t)t * R + r) * hc;
        float a = 0.f;
        for (int s = 0; s < hc; ++s) a += s_rms[t * hc + s] * p[s];
        a *= inv_hc;
        s_low[i] = a / (1.f + __expf(-a));
    }
    if (blockIdx.x == 0 && inj_out != nullptr) {
        for (int i = tid; i < nt * ninj; i += blockDim.x) {
            const int t = i / ninj, qq = i % ninj;
            const float* p = P + ((size_t)t * R + rank + qq) * hc;
            float a = 0.f;
            for (int s = 0; s < hc; ++s) a += s_rms[t * hc + s] * p[s];
            inj_out[(size_t)(t0 + t) * ninj + qq] = 2.f / (1.f + __expf(-a * inv_hc));
        }
    }
    __syncthreads();

    const int d = blockIdx.x * 8 + warp;
    if (d >= H) return;
    const size_t HC = (size_t)hc * H;
    float out[TMAX];
#pragma unroll
    for (int t = 0; t < TMAX; ++t) out[t] = 0.f;
    for (int s = 0; s < hc; ++s) {
        const half* row = up + ((size_t)s * H + d) * rank;
        float acc[TMAX];
#pragma unroll
        for (int t = 0; t < TMAX; ++t) acc[t] = 0.f;
        for (int c = 8 * lane; c < rank; c += 256) {
            float wv[8];
            h8(row + c, wv);
#pragma unroll
            for (int t = 0; t < TMAX; ++t) {
                if (t < nt) {
                    const float* l = s_low + t * rank + c;
                    const float4 a = *reinterpret_cast<const float4*>(l);
                    const float4 b = *reinterpret_cast<const float4*>(l + 4);
                    acc[t] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x + wv[5] * b.y +
                              wv[6] * b.z + wv[7] * b.w;
                }
            }
        }
        const float nv = 1.f + __half2float(nw[(size_t)s * H + d]);
#pragma unroll
        for (int t = 0; t < TMAX; ++t) {
            if (t < nt) {
                const float z = warp_sum(acc[t]);
                if (dbg_z) dbg_z[((size_t)s * nt + t) * H + d] = z;
                const float xv = x[(size_t)(t0 + t) * HC + (size_t)s * H + d];
                out[t] += xv * s_rms[t * hc + s] * nv / (1.f + __expf(-z));
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < TMAX; ++t)
            if (t < nt) y[(size_t)(t0 + t) * H + d] = __float2bfloat16(out[t] * inv_hc);
    }
}

// k1u: the decode fast k1. A warp owns K1U_NJ consecutive rows r of one stream s (r = down rows, then the inject rows) and
// walks the 256-column chunks once: the x rows of the window are loaded into registers once a chunk and feed all the warp's
// rows, and the warp's weight chunks are all in flight together. Every (row r, window row t) pair keeps the reference's
// per-lane add sequence (lane l owns columns 8l + 256i) and its warp-sum tree: the 32-wide reduction runs as
// scatter_sum32 over the NJ x NT partial sums, so P and ss are bit-identical to k1_kernel's. The ss jobs (sum of squares of
// x[t][s]) follow the P jobs as one warp each.
// POST: the previous block's hc_post is folded in: the kernel reads x' = x + bf16(block_out) * inj_prev (one fused multiply-add an
// element, the arithmetic of hc_post_kernel) wherever it reads the streams, and never writes them (k2t writes x' back).
template <int NJ, int NT, bool POST>
__global__ void __launch_bounds__(32) k1u_kernel(const float* __restrict__ x, const half* __restrict__ nw,
                                                 const half* __restrict__ down, const half* __restrict__ inj,
                                                 float* __restrict__ P, float* __restrict__ ss, int t0, int H, int hc,
                                                 int rank, int ninj, const __nv_bfloat16* __restrict__ bo,
                                                 const float* __restrict__ ijp) {
    const int lane = threadIdx.x;
    const int R = rank + ninj;
    const int gps = (R + NJ - 1) / NJ;  // row groups a stream
    const size_t HC = (size_t)hc * H;
    if ((int)blockIdx.x < gps * hc) {
        const int s = blockIdx.x / gps, r0 = (blockIdx.x % gps) * NJ;
        const int nj = min(NJ, R - r0);  // the last group of a stream may be short: its spare rows re-read the last row
        const half* w[NJ];                // (no per-row predicates: they split the FMA chains into separate blocks)
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int r = min(r0 + j, R - 1);
            w[j] = (r < rank ? down + (size_t)r * HC : inj + (size_t)(r - rank) * HC) + (size_t)s * H;
        }
        const half* n = nw + (size_t)s * H;
        const float* xs = x + (size_t)t0 * HC + (size_t)s * H;
        float acc[NJ][NT];
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int t = 0; t < NT; ++t) acc[j][t] = 0.f;
        // The first chunk's weights are constants of the call: issue them before waiting for the predecessor (a successor of
        // hc_post or of an early-triggering kernel starts streaming while the previous kernel still drains).
        uint4 wq[NJ];
        float nv[8];
#pragma unroll
        for (int j = 0; j < NJ; ++j) wq[j] = __ldg(reinterpret_cast<const uint4*>(w[j] + 8 * lane));
        h8(n + 8 * lane, nv);
        gbp_pdl::pdl_trigger();
        gbp_pdl::pdl_wait();  // x (and P/ss, last read by the previous call's k2) belong to earlier kernels of the stream
        float ij[POST ? NT : 1];
        if constexpr (POST) {
#pragma unroll
            for (int t = 0; t < NT; ++t) ij[t] = ijp[(size_t)(t0 + t) * hc + s];
        }
#pragma unroll 1
        for (int d = 8 * lane; d < H; d += 256) {
            float4 xa[NT], xb[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float* xr = xs + (size_t)t * HC + d;
                xa[t] = __ldg(reinterpret_cast<const float4*>(xr));
                xb[t] = __ldg(reinterpret_cast<const float4*>(xr + 4));
                if constexpr (POST) {
                    const uint4 u = __ldg(reinterpret_cast<const uint4*>(bo + (size_t)(t0 + t) * H + d));
                    xa[t].x = fmaf(__uint_as_float(u.x << 16), ij[t], xa[t].x);
                    xa[t].y = fmaf(__uint_as_float(u.x & 0xffff0000u), ij[t], xa[t].y);
                    xa[t].z = fmaf(__uint_as_float(u.y << 16), ij[t], xa[t].z);
                    xa[t].w = fmaf(__uint_as_float(u.y & 0xffff0000u), ij[t], xa[t].w);
                    xb[t].x = fmaf(__uint_as_float(u.z << 16), ij[t], xb[t].x);
                    xb[t].y = fmaf(__uint_as_float(u.z & 0xffff0000u), ij[t], xb[t].y);
                    xb[t].z = fmaf(__uint_as_float(u.w << 16), ij[t], xb[t].z);
                    xb[t].w = fmaf(__uint_as_float(u.w & 0xffff0000u), ij[t], xb[t].w);
                }
            }
            float onep[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) onep[i] = 1.f + nv[i];
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                float wv[8];
                dec_half8(wq[j], wv);
#pragma unroll
                for (int i = 0; i < 8; ++i) wv[i] *= onep[i];
#pragma unroll
                for (int t = 0; t < NT; ++t) acc[j][t] = dot8_acc(acc[j][t], wv, xa[t], xb[t]);
            }
            if (d + 256 < H) {  // next chunk
#pragma unroll
                for (int j = 0; j < NJ; ++j) wq[j] = __ldg(reinterpret_cast<const uint4*>(w[j] + d + 256));
                h8(n + d + 256, nv);
            }
        }
        constexpr int NEL = NJ * NT;
#pragma unroll
        for (int c = 0; c < (NEL + 31) / 32; ++c) {
            float v[32];
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                const int e = c * 32 + k;
                v[k] = e < NEL ? acc[(e / NT) % NJ][e % NT] : 0.f;
            }
            const float z = scatter_sum32(v, lane);
            const int e = c * 32 + lane;
            if (e < NEL && e / NT < nj) P[((size_t)(e % NT) * R + r0 + e / NT) * hc + s] = z;
        }
        return;
    }
    gbp_pdl::pdl_trigger();
    gbp_pdl::pdl_wait();
    const int q = (int)blockIdx.x - gps * hc;
    if (q >= NT * hc) return;
    const int t = q / hc, s = q % hc;
    const float* xr = x + (size_t)(t0 + t) * HC + (size_t)s * H;
    float ijv = 0.f;
    if constexpr (POST) ijv = ijp[(size_t)(t0 + t) * hc + s];
    float a = 0.f;
    for (int d = 4 * lane; d < H; d += 128) {
        float4 v = __ldg(reinterpret_cast<const float4*>(xr + d));
        if constexpr (POST) {
            const uint2 u = __ldg(reinterpret_cast<const uint2*>(bo + (size_t)(t0 + t) * H + d));
            v.x = fmaf(__uint_as_float(u.x << 16), ijv, v.x);
            v.y = fmaf(__uint_as_float(u.x & 0xffff0000u), ijv, v.y);
            v.z = fmaf(__uint_as_float(u.y << 16), ijv, v.z);
            v.w = fmaf(__uint_as_float(u.y & 0xffff0000u), ijv, v.w);
        }
        a += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    a = warp_sum(a);
    if (lane == 0) ss[t * hc + s] = a;
}


// k2c: the reference k2 body with 16 warps a block (512 threads) covering 16 columns, so the encode grid halves
// (320 -> 160 blocks) while every column's arithmetic is exactly the reference's (same per-column add order, same
// warp-shuffle reduction). Bit-identical to k2 by construction; the bit-exact GPU tests assert it.
__global__ void __launch_bounds__(512) k2c_kernel(const float* __restrict__ x, const half* __restrict__ nw,
                                                  const half* __restrict__ up, const float* __restrict__ P,
                                                  const float* __restrict__ ss, __nv_bfloat16* __restrict__ y,
                                                  float* __restrict__ inj_out, int t0, int nt, int H, int hc,
                                                  int rank, int ninj, float eps, float* __restrict__ dbg_z) {
    extern __shared__ float s_low[];  // [nt][rank]
    __shared__ float s_rms[TMAX * 8];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int R = rank + ninj;
    const float inv_hc = 1.f / (float)hc;
    if (tid < nt * hc) s_rms[tid] = rsqrtf(ss[tid] / (float)H + eps);
    __syncthreads();
    for (int i = tid; i < nt * rank; i += blockDim.x) {
        const int t = i / rank, r = i % rank;
        const float* p = P + ((size_t)t * R + r) * hc;
        float a = 0.f;
        for (int s = 0; s < hc; ++s) a += s_rms[t * hc + s] * p[s];
        a *= inv_hc;
        s_low[i] = a / (1.f + __expf(-a));
    }
    if (blockIdx.x == 0 && inj_out != nullptr) {
        for (int i = tid; i < nt * ninj; i += blockDim.x) {
            const int t = i / ninj, qq = i % ninj;
            const float* p = P + ((size_t)t * R + rank + qq) * hc;
            float a = 0.f;
            for (int s = 0; s < hc; ++s) a += s_rms[t * hc + s] * p[s];
            inj_out[(size_t)(t0 + t) * ninj + qq] = 2.f / (1.f + __expf(-a * inv_hc));
        }
    }
    __syncthreads();

    const int d = blockIdx.x * 16 + warp;
    if (d >= H) return;
    const size_t HC = (size_t)hc * H;
    float out[TMAX];
#pragma unroll
    for (int t = 0; t < TMAX; ++t) out[t] = 0.f;
    for (int s = 0; s < hc; ++s) {
        const half* row = up + ((size_t)s * H + d) * rank;
        float acc[TMAX];
#pragma unroll
        for (int t = 0; t < TMAX; ++t) acc[t] = 0.f;
        for (int c = 8 * lane; c < rank; c += 256) {
            float wv[8];
            h8(row + c, wv);
#pragma unroll
            for (int t = 0; t < TMAX; ++t) {
                if (t < nt) {
                    const float* l = s_low + t * rank + c;
                    const float4 a = *reinterpret_cast<const float4*>(l);
                    const float4 b = *reinterpret_cast<const float4*>(l + 4);
                    acc[t] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x + wv[5] * b.y +
                              wv[6] * b.z + wv[7] * b.w;
                }
            }
        }
        const float nv = 1.f + __half2float(nw[(size_t)s * H + d]);
#pragma unroll
        for (int t = 0; t < TMAX; ++t) {
            if (t < nt) {
                const float z = warp_sum(acc[t]);
                if (dbg_z) dbg_z[((size_t)s * nt + t) * H + d] = z;
                const float xv = x[(size_t)(t0 + t) * HC + (size_t)s * H + d];
                out[t] += xv * s_rms[t * hc + s] * nv / (1.f + __expf(-z));
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < TMAX; ++t)
            if (t < nt) y[(size_t)(t0 + t) * H + d] = __float2bfloat16(out[t] * inv_hc);
    }
}

// ---- k2t: the decode fast k2 for hc = 4 (the Flash-Next family) ------------------------------------------------------
// Same arithmetic as k2c bit for bit, restructured for the 1..8-row window (at 7 rows k2c is issue-bound, not DRAM-bound:
// every warp ran 28 warp_sum + exp + IEEE divide chains redundantly across its 32 lanes):
// - a warp owns K2T_CPW consecutive columns; for each column the four streams' weight chunks are loaded once, and each
//   low chunk is read from shared memory once for the four streams;
// - low lives in shared memory swizzled so a lane's two float4 are conflict free;
// - the first column's weights are loaded before waiting for k1; under a PDL launch the other columns are bulk-prefetched
//   into L2 right after the wait (before the low computation), so the low computation overlaps their DRAM latency;
// - the 4 x NT per-lane partial sums are reduced by scatter_sum32 (a reduce-scatter with the same xor 16/8/4/2/1 pairings
//   as warp_sum, so each element is bit-identical to warp_sum of it), after which lane s*8+t owns z[s][t] and evaluates the
//   sigmoid mix once; the four streams' terms are added in the reference order s = 0..3 through three shuffles.
// rank <= 512 and a multiple of 8 (at most two 256-column chunks a lane). H % 256 == 0 is enforced by the launcher.
constexpr int K2T_CPW = 4;  // columns a warp
constexpr int K2T_WPB = 8;  // warps a block

// low[t][r] position in the swizzled shared array: chunk 0 (r < 256) as [t][half][lane*4 + i], chunk 1 after it.
__device__ __forceinline__ int k2t_low_pos(int t, int r, int base1, int stride1) {
    const int it = r >> 8, c = r & 255, l = c >> 3, h = (c >> 2) & 1, i = c & 3;
    return it == 0 ? (t * 2 + h) * 128 + l * 4 + i : base1 + (t * 2 + h) * stride1 + l * 4 + i;
}

__device__ __forceinline__ void k2t_prefetch_l2(const void* p, unsigned bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}

// POST: as in k1u, x' = x + bf16(block_out) * inj_prev is what the kernel takes for x, and each (row, stream, column) is written back
// (to `x`, in place) by the one lane that reads it: k1u has completed by then, so no other reader of the old values is left.
template <int NT, bool POST>
__global__ void __launch_bounds__(K2T_WPB * 32, 2) k2t_kernel(float* x, const half* __restrict__ nw,
                                                              const half* __restrict__ up, const float* __restrict__ P,
                                                              const float* __restrict__ ss, __nv_bfloat16* __restrict__ y,
                                                              float* __restrict__ inj_out, int t0, int H, int rank,
                                                              int ninj, float eps, float* __restrict__ dbg_z, int pf,
                                                              const __nv_bfloat16* __restrict__ bo,
                                                              const float* __restrict__ ijp) {
    constexpr int HCN = 4;
    extern __shared__ float s_low[];  // swizzled low [NT][rank]
    __shared__ float s_rms[NT * HCN];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int R = rank + ninj;
    const float inv_hc = 1.f / (float)HCN;
    const int col0 = (blockIdx.x * K2T_WPB + warp) * K2T_CPW;
    const size_t HC = (size_t)HCN * H;
    const bool has0 = 8 * lane < rank, has1 = 8 * lane + 256 < rank;
    const int base1 = NT * 2 * 128;
    const int stride1 = rank > 320 ? 128 : 32;

    gbp_pdl::pdl_trigger();  // a successor may launch now; it waits for this grid before touching anything
    // the first column's weights are independent of k1: issue them before waiting for it
    uint4 q0[HCN], q1[HCN];
    if (col0 < H) {
#pragma unroll
        for (int s = 0; s < HCN; ++s) {
            const half* row = up + ((size_t)s * H + col0) * rank;
            if (has0) q0[s] = __ldg(reinterpret_cast<const uint4*>(row + 8 * lane));
            if (has1) q1[s] = __ldg(reinterpret_cast<const uint4*>(row + 8 * lane + 256));
        }
    }
    gbp_pdl::pdl_wait();  // P and ss are k1's output (and y / inj_out were last read by earlier kernels)
    // PDL launches only (the weights then stream while k1 drains; a plain launch measured slower with it): the warp's other
    // columns, one bulk L2 prefetch a stream (their rows are contiguous), issued after the wait so they queue behind k1's
    // stream but ahead of the low computation below, whose latency they now cover.
    if (pf && col0 + 1 < H && lane < HCN)
        k2t_prefetch_l2(up + ((size_t)lane * H + col0 + 1) * rank, (unsigned)(2 * rank * (min(K2T_CPW, H - col0) - 1)));
    if (tid < NT * HCN) s_rms[tid] = rsqrtf(ss[tid] / (float)H + eps);
    __syncthreads();
    for (int i = tid; i < NT * rank; i += K2T_WPB * 32) {
        const int t = i / rank, r = i % rank;
        const float4 p = *reinterpret_cast<const float4*>(P + ((size_t)t * R + r) * HCN);
        float a = 0.f;
        a += s_rms[t * HCN + 0] * p.x;
        a += s_rms[t * HCN + 1] * p.y;
        a += s_rms[t * HCN + 2] * p.z;
        a += s_rms[t * HCN + 3] * p.w;
        a *= inv_hc;
        s_low[k2t_low_pos(t, r, base1, stride1)] = a / (1.f + __expf(-a));
    }
    if (blockIdx.x == 0 && inj_out != nullptr) {
        for (int i = tid; i < NT * ninj; i += K2T_WPB * 32) {
            const int t = i / ninj, qq = i % ninj;
            const float* p = P + ((size_t)t * R + rank + qq) * HCN;
            float a = 0.f;
#pragma unroll
            for (int s = 0; s < HCN; ++s) a += s_rms[t * HCN + s] * p[s];
            inj_out[(size_t)(t0 + t) * ninj + qq] = 2.f / (1.f + __expf(-a * inv_hc));
        }
    }
    __syncthreads();

    const int se = lane >> 3, te = lane & 7;  // this lane's (stream, row) once the partial sums are scattered
#pragma unroll 1
    for (int k = 0; k < K2T_CPW; ++k) {
        const int col = col0 + k;
        if (col >= H) break;
        if (k > 0) {
#pragma unroll
            for (int s = 0; s < HCN; ++s) {
                const half* row = up + ((size_t)s * H + col) * rank;
                if (has0) q0[s] = __ldg(reinterpret_cast<const uint4*>(row + 8 * lane));
                if (has1) q1[s] = __ldg(reinterpret_cast<const uint4*>(row + 8 * lane + 256));
            }
        }
        float acc[HCN][NT];
#pragma unroll
        for (int s = 0; s < HCN; ++s)
#pragma unroll
            for (int t = 0; t < NT; ++t) acc[s][t] = 0.f;
        if (has0) {
            float wv[HCN][8];
#pragma unroll
            for (int s = 0; s < HCN; ++s) dec_half8(q0[s], wv[s]);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float4 a = *reinterpret_cast<const float4*>(s_low + (t * 2 + 0) * 128 + lane * 4);
                const float4 b = *reinterpret_cast<const float4*>(s_low + (t * 2 + 1) * 128 + lane * 4);
#pragma unroll
                for (int s = 0; s < HCN; ++s) acc[s][t] = dot8_acc(acc[s][t], wv[s], a, b);
            }
        }
        if (has1) {
            float wv[HCN][8];
#pragma unroll
            for (int s = 0; s < HCN; ++s) dec_half8(q1[s], wv[s]);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float4 a = *reinterpret_cast<const float4*>(s_low + base1 + (t * 2 + 0) * stride1 + lane * 4);
                const float4 b = *reinterpret_cast<const float4*>(s_low + base1 + (t * 2 + 1) * stride1 + lane * 4);
#pragma unroll
                for (int s = 0; s < HCN; ++s) acc[s][t] = dot8_acc(acc[s][t], wv[s], a, b);
            }
        }
        float v[32];
#pragma unroll
        for (int s = 0; s < HCN; ++s)
#pragma unroll
            for (int t = 0; t < 8; ++t) v[s * 8 + t] = (t < NT) ? acc[s][t < NT ? t : 0] : 0.f;
        const float z = scatter_sum32(v, lane);
        float term = 0.f;
        if (te < NT) {
            if (dbg_z) dbg_z[((size_t)se * NT + te) * H + col] = z;
            const float nv = 1.f + __half2float(nw[(size_t)se * H + col]);
            float xv = x[(size_t)(t0 + te) * HC + (size_t)se * H + col];
            if constexpr (POST) {
                xv = fmaf(__bfloat162float(bo[(size_t)(t0 + te) * H + col]), ijp[(size_t)(t0 + te) * HCN + se], xv);
                x[(size_t)(t0 + te) * HC + (size_t)se * H + col] = xv;
            }
            term = xv * s_rms[te * HCN + se] * nv / (1.f + __expf(-z));
        }
        const float t1 = __shfl_sync(0xffffffffu, term, (lane + 8) & 31);
        const float t2 = __shfl_sync(0xffffffffu, term, (lane + 16) & 31);
        const float t3 = __shfl_sync(0xffffffffu, term, (lane + 24) & 31);
        float o = 0.f;
        o += term;
        o += t1;
        o += t2;
        o += t3;
        if (lane < NT) y[(size_t)(t0 + lane) * H + col] = __float2bfloat16(o * inv_hc);
    }
}

#include "hcx_pk.cuh"

}  // namespace hcx

using namespace hcx;

// Scratch floats: P [8][rank + ninj][hc] + ss [8][hc].
GB_EXPORT size_t gb_hcx_scratch_bytes(int hc, int rank, int ninj) {
    return ((size_t)TMAX * (rank + ninj) * hc + (size_t)TMAX * hc) * sizeof(float);
}

// k2t launch for 1..8 rows (rows are a template parameter: the per-lane partial sums live in registers). With the kHcx PDL bit
// the launch carries the programmatic-dependent-launch attribute (k2t waits before it reads k1's output).
static int launch_k2t(int nt, float* x, const half* nw, const half* up, const float* P, const float* ss,
                      __nv_bfloat16* y, float* inj_out, int t0, int H, int rank, int ninj, float eps, float* dbg_z,
                      cudaStream_t st, const __nv_bfloat16* bo, const float* ijp) {
    const int blocks = (H + K2T_WPB * K2T_CPW - 1) / (K2T_WPB * K2T_CPW);
    const int stride1 = rank > 320 ? 128 : 32;
    const size_t smem = (size_t)nt * (256 + (rank > 256 ? 2 * stride1 : 0)) * sizeof(float);
    const bool pdl = gbp_pdl::on(gbp_pdl::kHcx);
    switch (nt) {
#define K2T_CASE(N)                                                                                                    \
    case N:                                                                                                            \
        return bo ? gbp_pdl::launch_pdl(pdl, k2t_kernel<N, true>, dim3(blocks), dim3(K2T_WPB * 32), smem, st, x, nw,   \
                                        up, P, ss, y, inj_out, t0, H, rank, ninj, eps, dbg_z, pdl ? 1 : 0, bo, ijp)    \
                  : gbp_pdl::launch_pdl(pdl, k2t_kernel<N, false>, dim3(blocks), dim3(K2T_WPB * 32), smem, st, x, nw,  \
                                        up, P, ss, y, inj_out, t0, H, rank, ninj, eps, dbg_z, pdl ? 1 : 0, bo, ijp);
        K2T_CASE(1) K2T_CASE(2) K2T_CASE(3) K2T_CASE(4) K2T_CASE(5) K2T_CASE(6) K2T_CASE(7) K2T_CASE(8)
#undef K2T_CASE
        default: return (int)cudaErrorInvalidValue;
    }
}

// k1u launch for 1..8 rows: NJ rows of a stream a warp. 9 keeps the registers inside what a thread has up to 7 rows (the
// 324 = 320 + 4 rows of the real mixers split into 36 groups a stream); 8 rows use 6. Same PDL attribute rule as k2t.
static int launch_k1u(int nt, const float* x, const half* nw, const half* down, const half* inj, float* P, float* ss,
                      int t0, int H, int hc, int rank, int ninj, cudaStream_t st, const __nv_bfloat16* bo,
                      const float* ijp) {
    const int R = rank + ninj;
    const bool pdl = gbp_pdl::on(gbp_pdl::kHcx);
    switch (nt) {
#define K1U_CASE(NJ, N)                                                                                                \
    case N: {                                                                                                          \
        const int warps = ((R + NJ - 1) / NJ) * hc + N * hc;                                                           \
        return bo ? gbp_pdl::launch_pdl(pdl, k1u_kernel<NJ, N, true>, dim3(warps), dim3(32), 0, st, x, nw, down, inj,  \
                                        P, ss, t0, H, hc, rank, ninj, bo, ijp)                                         \
                  : gbp_pdl::launch_pdl(pdl, k1u_kernel<NJ, N, false>, dim3(warps), dim3(32), 0, st, x, nw, down, inj, \
                                        P, ss, t0, H, hc, rank, ninj, bo, ijp);                                        \
    }
        K1U_CASE(9, 1) K1U_CASE(9, 2) K1U_CASE(9, 3) K1U_CASE(9, 4) K1U_CASE(9, 5) K1U_CASE(9, 6) K1U_CASE(9, 7)
        K1U_CASE(6, 8)
#undef K1U_CASE
        default: return (int)cudaErrorInvalidValue;
    }
}

// The packed down / inject rows (hcx_pk.cuh) k1p reads instead of the fp16 ones.
struct PkArgs {
    const uint4* down;
    const uint4* inj;
    const uint4* up;
};

// k1p launch for 1..3 rows (see launch_k1u; at four rows the packed kernels measured slower in the decode graph): PK_NJ rows a warp, jobs = rank / 4 + ninj / 4 groups a stream.
static int launch_k1p(int nt, const float* x, const half* nw, const PkArgs& pk, float* P, float* ss, int t0, int H, int hc, int rank,
                      int ninj, cudaStream_t st, const __nv_bfloat16* bo, const float* ijp) {
    const int R = rank + ninj;
    const bool pdl = gbp_pdl::on(gbp_pdl::kHcx);
    switch (nt) {
#define K1P_CASE(N)                                                                                                    \
    case N: {                                                                                                          \
        const int warps = ((R + PK_NJ - 1) / PK_NJ) * hc + N * hc;                                                     \
        return bo ? gbp_pdl::launch_pdl(pdl, k1p_kernel<N, true>, dim3(warps), dim3(32), 0, st, x, nw, pk.down, pk.inj, \
                                        P, ss, t0, H, hc, rank, ninj, bo, ijp)                       \
                  : gbp_pdl::launch_pdl(pdl, k1p_kernel<N, false>, dim3(warps), dim3(32), 0, st, x, nw, pk.down,       \
                                        pk.inj, P, ss, t0, H, hc, rank, ninj, bo, ijp);              \
    }
        K1P_CASE(1) K1P_CASE(2) K1P_CASE(3)
#undef K1P_CASE
        default: return (int)cudaErrorInvalidValue;
    }
}

// k2tp launch for 1..3 rows (see launch_k2t).
static int launch_k2tp(int nt, float* x, const half* nw, const uint4* pup, const float* P, const float* ss, __nv_bfloat16* y,
                       float* inj_out, int t0, int H, int rank, int ninj, float eps, cudaStream_t st, const __nv_bfloat16* bo,
                       const float* ijp) {
    const int blocks = (H + K2T_WPB * K2T_CPW - 1) / (K2T_WPB * K2T_CPW);
    const size_t smem = (size_t)nt * (256 + 2 * 32) * sizeof(float);
    const bool pdl = gbp_pdl::on(gbp_pdl::kHcx);
    switch (nt) {
#define K2TP_CASE(N)                                                                                                   \
    case N:                                                                                                            \
        return bo ? gbp_pdl::launch_pdl(pdl, k2tp_kernel<N, true>, dim3(blocks), dim3(K2T_WPB * 32), smem, st, x, nw,  \
                                        pup, P, ss, y, inj_out, t0, H, rank, ninj, eps, pdl ? 1 : 0, bo, ijp)          \
                  : gbp_pdl::launch_pdl(pdl, k2tp_kernel<N, false>, dim3(blocks), dim3(K2T_WPB * 32), smem, st, x, nw, \
                                        pup, P, ss, y, inj_out, t0, H, rank, ninj, eps, pdl ? 1 : 0, bo, ijp);
        K2TP_CASE(1) K2TP_CASE(2) K2TP_CASE(3)
#undef K2TP_CASE
        default: return (int)cudaErrorInvalidValue;
    }
}

// y [T][H] bf16 (and inj [T][hc] when inject_w != null) from streams [T][hc][H] fp32. Same contract as gb_hc_pre.
// `ref=false` (gb_hcx_pre / gb_hcx_pre_z) launches the decode fast kernels (k1u, then k2t for hc = 4 with rank <= 512, k2c
// otherwise) unconditionally, bit-identical to k1/k2; `ref=true` (gb_hcx_pre_ref / gb_hcx_pre_z_ref) launches the reference
// k1_kernel/k2_kernel directly for the bit-exact GPU tests.
static int hcx_pre_impl(const void* streams, const void* hc_norm_w, const void* down_w, const void* up_w,
                        const void* inject_w, void* y_out, float* inj_out, void* scratch, unsigned int hidden_size,
                        unsigned int hc_mult, unsigned int rank, float eps, unsigned int num_tokens, void* stream,
                        bool ref, float* dbg_z, const void* block_out = nullptr, const float* inj_prev = nullptr,
                        const PkArgs* pk = nullptr) {
    if (num_tokens == 0) return 0;
    const int H = (int)hidden_size, hc = (int)hc_mult, rk = (int)rank;
    if (H % 256 || rk % 8 || hc > 8) return (int)cudaErrorInvalidValue;
    const int ninj = inject_w ? hc : 0;
    const int R = rk + ninj;
    // the folded hc_post (block_out + inj_prev given) exists in the decode kernels of the Flash-Next shape only
    if (block_out && (ref || !inj_prev || hc != 4 || rk > 512)) return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    float* P = static_cast<float*>(scratch);
    float* ss = P + (size_t)TMAX * R * hc;
    for (int t0 = 0; t0 < (int)num_tokens; t0 += TMAX) {
        const int nt = min(TMAX, (int)num_tokens - t0);
        const bool use_k2b = !ref;
        const bool use_k2t = !ref && hc == 4 && rk <= 512;
        int rc;
        if (!ref && pk && nt <= 3) {
            rc = launch_k1p(nt, (const float*)streams, (const half*)hc_norm_w, *pk, P, ss, t0, H, hc, rk, ninj, st,
                            (const __nv_bfloat16*)block_out, inj_prev);
        } else if (!ref) {
            rc = launch_k1u(nt, (const float*)streams, (const half*)hc_norm_w, (const half*)down_w, (const half*)inject_w,
                            P, ss, t0, H, hc, rk, ninj, st, (const __nv_bfloat16*)block_out, inj_prev);
        } else {
            const int jobs = R * hc + nt * hc;
            k1_kernel<<<(jobs + 3) / 4, 128, 0, st>>>((const float*)streams, (const half*)hc_norm_w,
                                                       (const half*)down_w, (const half*)inject_w, P, ss, t0, nt, H,
                                                       hc, rk, ninj);
            rc = GB_CHECK_LAUNCH();
        }
        if (rc) return rc;
        if (use_k2t && pk && !ref && nt <= 3) {
            rc = launch_k2tp(nt, (float*)streams, (const half*)hc_norm_w, pk->up, P, ss, (__nv_bfloat16*)y_out,
                             inject_w ? inj_out : nullptr, t0, H, rk, ninj, eps, st, (const __nv_bfloat16*)block_out, inj_prev);
        } else if (use_k2t) {
            rc = launch_k2t(nt, (float*)streams, (const half*)hc_norm_w, (const half*)up_w, P, ss,
                            (__nv_bfloat16*)y_out, inject_w ? inj_out : nullptr, t0, H, rk, ninj, eps, dbg_z, st,
                            (const __nv_bfloat16*)block_out, inj_prev);
        } else {
            if (use_k2b) {
                k2c_kernel<<<(H + 15) / 16, 512, (size_t)nt * rk * sizeof(float), st>>>(
                    (const float*)streams, (const half*)hc_norm_w, (const half*)up_w, P, ss, (__nv_bfloat16*)y_out,
                    inject_w ? inj_out : nullptr, t0, nt, H, hc, rk, ninj, eps, dbg_z);
            } else {
                k2_kernel<<<(H + 7) / 8, 256, (size_t)nt * rk * sizeof(float), st>>>(
                    (const float*)streams, (const half*)hc_norm_w, (const half*)up_w, P, ss, (__nv_bfloat16*)y_out,
                    inject_w ? inj_out : nullptr, t0, nt, H, hc, rk, ninj, eps, dbg_z);
            }
            rc = GB_CHECK_LAUNCH();
        }
        if (rc) return rc;
    }
    return 0;
}

GB_EXPORT int gb_hcx_pre(const void* streams, const void* hc_norm_w, const void* down_w, const void* up_w,
                         const void* inject_w, void* y_out, float* inj_out, void* scratch, unsigned int hidden_size,
                         unsigned int hc_mult, unsigned int rank, float eps, unsigned int num_tokens, void* stream) {
    return hcx_pre_impl(streams, hc_norm_w, down_w, up_w, inject_w, y_out, inj_out, scratch, hidden_size, hc_mult,
                        rank, eps, num_tokens, stream, false, nullptr);
}

// gb_hcx_pre after the previous block's hc_post, in the same two launches: `streams` first becomes streams + bf16(block_out) * inj_prev per
// stream (what gb_fn_hc_post writes, bit for bit), then the pre-mix reads the new values. hc = 4, rank <= 512 only.
GB_EXPORT int gb_hcx_pre_post(void* streams, const void* block_out, const float* inj_prev, const void* hc_norm_w,
                              const void* down_w, const void* up_w, const void* inject_w, void* y_out, float* inj_out,
                              void* scratch, unsigned int hidden_size, unsigned int hc_mult, unsigned int rank,
                              float eps, unsigned int num_tokens, void* stream) {
    return hcx_pre_impl(streams, hc_norm_w, down_w, up_w, inject_w, y_out, inj_out, scratch, hidden_size, hc_mult,
                        rank, eps, num_tokens, stream, false, nullptr, block_out, inj_prev);
}

// The same with the down and inject rows read from the packed copy (hcx_pk.cuh) for up to three rows a pass: the same bits. H = 2,560,
// hc = 4, rank = 320 (packed_ok on the host).
GB_EXPORT int gb_hcx_pre_pk(const void* streams, const void* hc_norm_w, const void* down_w, const void* up_w, const void* inject_w,
                            void* y_out, float* inj_out, void* scratch, unsigned int hidden_size, unsigned int hc_mult,
                            unsigned int rank, float eps, unsigned int num_tokens, void* stream, const void* pk_down,
                            const void* pk_inj, const void* pk_up) {
    const PkArgs pk{(const uint4*)pk_down, (const uint4*)pk_inj, (const uint4*)pk_up};
    if (hidden_size != 2560 || hc_mult != 4 || rank != 320 || (inject_w && !pk_inj)) return (int)cudaErrorInvalidValue;
    return hcx_pre_impl(streams, hc_norm_w, down_w, up_w, inject_w, y_out, inj_out, scratch, hidden_size, hc_mult, rank, eps,
                        num_tokens, stream, false, nullptr, nullptr, nullptr, &pk);
}

GB_EXPORT int gb_hcx_pre_post_pk(void* streams, const void* block_out, const float* inj_prev, const void* hc_norm_w,
                                 const void* down_w, const void* up_w, const void* inject_w, void* y_out, float* inj_out,
                                 void* scratch, unsigned int hidden_size, unsigned int hc_mult, unsigned int rank, float eps,
                                 unsigned int num_tokens, void* stream, const void* pk_down, const void* pk_inj,
                                 const void* pk_up) {
    const PkArgs pk{(const uint4*)pk_down, (const uint4*)pk_inj, (const uint4*)pk_up};
    if (hidden_size != 2560 || hc_mult != 4 || rank != 320 || (inject_w && !pk_inj)) return (int)cudaErrorInvalidValue;
    return hcx_pre_impl(streams, hc_norm_w, down_w, up_w, inject_w, y_out, inj_out, scratch, hidden_size, hc_mult, rank, eps,
                        num_tokens, stream, false, nullptr, block_out, inj_prev, &pk);
}

// ---- packed rows: sizes, scan, pack, check (see hcx_pk.cuh) ----
GB_EXPORT size_t gb_hcx_pk_job_bytes() { return PK_JOB_BYTES; }

// st: 3 device ints, {INT_MAX, INT_MIN, 0} on entry (see pk_scan_kernel).
GB_EXPORT int gb_hcx_pk_scan(const void* w, size_t n, int* st, void* stream) {
    if (n == 0) return (int)cudaErrorInvalidValue;
    pk_scan_kernel<<<(unsigned)((n + 255) / 256 < 1024 ? (n + 255) / 256 : 1024), 256, 0, gb_stream(stream)>>>((const half*)w, n, st);
    return GB_CHECK_LAUNCH();
}

// Down / inject rows [rows][4 * 2,560]: the jobs, then the tensor's table (rows * 4 / PK_NJ jobs of PK_JOB_BYTES, 4,096 bytes).
GB_EXPORT int gb_hcx_pk1_pack(const void* w, int rows, int h, int eh0, void* out, void* stream) {
    if (rows < PK_NJ || rows % PK_NJ || h != 2560) return (int)cudaErrorInvalidValue;
    pk1_pack_kernel<<<4 * rows / PK_NJ, 32, 0, gb_stream(stream)>>>((const half*)w, rows, h, eh0, (uint4*)out);
    pk_lut_kernel<<<1, 256, 0, gb_stream(stream)>>>(eh0, (uint32_t*)((uint8_t*)out + (size_t)(4 * rows / PK_NJ) * PK_JOB_BYTES));
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_hcx_pk1_check(const void* w, const void* packed, int rows, int h, unsigned long long* bad, void* stream) {
    if (rows < PK_NJ || rows % PK_NJ || h != 2560) return (int)cudaErrorInvalidValue;
    pk1_check_kernel<<<4 * rows / PK_NJ, 32, 0, gb_stream(stream)>>>((const half*)w, (const uint4*)packed, rows, h, bad);
    return GB_CHECK_LAUNCH();
}

// Up rows [4 * 2,560][320]: 640 jobs of PK2_JOB_BYTES, then the table.
GB_EXPORT int gb_hcx_pk2_pack(const void* w, int h, int eh0, void* out, void* stream) {
    if (h != 2560) return (int)cudaErrorInvalidValue;
    pk2_pack_kernel<<<h / K2T_CPW, 32, 0, gb_stream(stream)>>>((const half*)w, h, eh0, (uint4*)out);
    pk_lut_kernel<<<1, 256, 0, gb_stream(stream)>>>(eh0, (uint32_t*)((uint8_t*)out + (size_t)(h / K2T_CPW) * PK2_JOB_BYTES));
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_hcx_pk2_check(const void* w, const void* packed, int h, unsigned long long* bad, void* stream) {
    if (h != 2560) return (int)cudaErrorInvalidValue;
    pk2_check_kernel<<<h / K2T_CPW, 32, 0, gb_stream(stream)>>>((const half*)w, (const uint4*)packed, h, bad);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT size_t gb_hcx_pk2_job_bytes() { return PK2_JOB_BYTES; }

GB_EXPORT int gb_hcx_pre_ref(const void* streams, const void* hc_norm_w, const void* down_w, const void* up_w,
                             const void* inject_w, void* y_out, float* inj_out, void* scratch,
                             unsigned int hidden_size, unsigned int hc_mult, unsigned int rank, float eps,
                             unsigned int num_tokens, void* stream) {
    return hcx_pre_impl(streams, hc_norm_w, down_w, up_w, inject_w, y_out, inj_out, scratch, hidden_size, hc_mult,
                        rank, eps, num_tokens, stream, true, nullptr);
}

// z-dump entry points (debug aids, not a production path): run the new/ref k2 with dbg_z writes into `z_out`
// ([hc * nt, H] floats) and the same P/ss machinery, so the test can compare the pre-sigmoid sums bit for bit.
GB_EXPORT int gb_hcx_pre_z(const void* streams, const void* hc_norm_w, const void* down_w, const void* up_w,
                           const void* inject_w, void* y_out, float* inj_out, void* scratch, float* z_out,
                           unsigned int hidden_size, unsigned int hc_mult, unsigned int rank, float eps,
                           unsigned int num_tokens, void* stream) {
    return hcx_pre_impl(streams, hc_norm_w, down_w, up_w, inject_w, y_out, inj_out, scratch, hidden_size, hc_mult,
                        rank, eps, num_tokens, stream, false, z_out);
}

GB_EXPORT int gb_hcx_pre_z_ref(const void* streams, const void* hc_norm_w, const void* down_w, const void* up_w,
                               const void* inject_w, void* y_out, float* inj_out, void* scratch, float* z_out,
                               unsigned int hidden_size, unsigned int hc_mult, unsigned int rank, float eps,
                               unsigned int num_tokens, void* stream) {
    return hcx_pre_impl(streams, hc_norm_w, down_w, up_w, inject_w, y_out, inj_out, scratch, hidden_size, hc_mult,
                        rank, eps, num_tokens, stream, true, z_out);
}
