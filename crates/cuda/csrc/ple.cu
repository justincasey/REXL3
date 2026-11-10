// gb-nvcc: --fmad=false
// PLE: the n-gram embedding stage of Qwen3.8-Flash-Next. Original REXL3 code, Victor Cruz.
//
// The PLE stage of Qwen3.8-Flash-Next (hashed n-gram embeddings injected into every residual stream before decoder
// layer 1), written for REXL3 from the model and table-format definitions.
//
//   gb_ple_rows       decode gathered n-gram table rows into the per-token embedding emb [rows][heads * 160] f16. A packed
//                     row is (1 + 10 K) u16 words: word 0 the f16 row scale, then a 160 K-bit tail-biting ring whose element
//                     i is the 16-bit state ending at ring bit (i + 1) K - 1; value = mul1(state) * scale + bias[head][i]
//   gb_ple_proj       y [rows][n] f16 = x [rows][k] f16 . w [n][k]^T f16, fp32 accumulation. A warp per output column
//                     (window-sized batches, graph-safe); gb_ple_gemm is the tiled form for prefill chunks, each output
//                     summed over k in ascending order whatever the row count (chunk-invariant, unlike a vendor GEMM)
//   gb_ple_gate       per (row, stream): grouped RMS of the key and of the stream, their dot, the signed-sqrt sigmoid gate
//                     g (written to sg [rows][hc]), and the conv input rms(g * value) * (1 + w) as f16 into nb [rows][hc*h]
//   gb_ple_conv       per (row, channel): depthwise causal conv (4 taps, dilation 3) over nb and the slot ring, silu, and
//                     streams += (g * value + conv)
//   gb_ple_ring       the last `ring` positions of every sequence of the batch into its slot ring, for the next batch
//
// Rows of one sequence are contiguous with contiguous positions; `seq_start` [n_seq + 1] and `seq_slot` [n_seq] describe
// the batch. The ring is indexed by position (mod ring), so a rejected draft row is simply rewritten by the next window
// before any later row reads it: there is no commit step.
#include "common.cuh"
#include "pdl.cuh"

// Decode launches (gbp_pdl::kFlash): a kernel launched with the programmatic-dependent-launch attribute may start before its predecessor
// finishes, so it calls pdl_wait() -- first in every thread, early exits included -- before it reads anything an earlier kernel wrote or
// writes anything an earlier kernel may still read; weights are read (or pulled towards L2) before it. A launch without the attribute
// makes both calls no-ops (pdl.cuh), so only timing changes, never a bit.
using gbp_pdl::pdl_trigger;
using gbp_pdl::pdl_wait;
using gbp_pdl::prefetch_l2;
inline bool flash_pdl() { return gbp_pdl::on(gbp_pdl::kFlash); }

namespace {

constexpr int ROW = 160;     // elements of one n-gram table row
constexpr int TAPS = 4;      // conv kernel
constexpr int DIL = 3;       // conv dilation (= ngram size)

__device__ __forceinline__ float h2f(__half v) { return __half2float(v); }

__device__ __forceinline__ float block_sum(float v, float* red) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < nw; ++i) t += red[i];
    return t;
}

// The sequence of row r: index, its first row and its slot.
__device__ __forceinline__ int seq_of(const int* seq_start, int n_seq, int r) {
    int s = 0;
    while (s + 1 < n_seq && seq_start[s + 1] <= r) ++s;
    return s;
}

__global__ void __launch_bounds__(ROW) rows_k(const uint16_t* __restrict__ packed, const __half* __restrict__ bias,
                                              __half* __restrict__ emb, int heads, int k, int words) {
    extern __shared__ uint16_t wsm[];
    pdl_trigger();     // proj_k may launch: it waits for this grid before it reads emb
    const int e = blockIdx.x;  // entry = token row * heads + head
    const int head = e % heads;
    const int i = threadIdx.x;
    const float head_bias = h2f(bias[head * ROW + i]);  // a weight: read before the wait
    pdl_wait();        // packed is staged by an earlier operation of the stream; emb may still be read by earlier kernels
    for (int t = i; t < words; t += ROW) wsm[t] = packed[(size_t)e * words + t];
    __syncthreads();
    uint32_t state = 0;
    for (int m = 0; m < 16; ++m) {
        int pos = i - m / k;
        if (pos < 0) pos += ROW;
        const int b = pos * k + m % k;
        state |= ((uint32_t)(wsm[1 + (b >> 4)] >> (b & 15)) & 1u) << m;
    }
    const uint32_t prod = state * 0x83DCD12Du;
    const float hs = 1024.0f + (float)((prod & 255u) + ((prod >> 8) & 255u) + ((prod >> 16) & 255u) + (prod >> 24));
    const float kinv = __half2float(__ushort_as_half((unsigned short)0x1eee));
    const float kbias = __half2float(__ushort_as_half((unsigned short)0xc931));
    const float cb = __half2float(__float2half_rn(__fadd_rn(__fmul_rn(hs, kinv), kbias)));
    const float scale = __half2float(__ushort_as_half(wsm[0]));
    emb[(size_t)e * ROW + i] = __float2half_rn(__fadd_rn(__fmul_rn(cb, scale), head_bias));
}

// One warp per output column; the batch rows are walked in tiles of 8 so each warp streams its weight row once per tile.
__global__ void __launch_bounds__(256) proj_k(const __half* __restrict__ x, const __half* __restrict__ w, __half* __restrict__ y,
                                              int rows, int k, int n) {
    pdl_trigger();  // the successor may launch: it waits for this grid before it reads y
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = blockIdx.x * 8 + warp;
    // before the wait: pull the start of this warp's weight row towards L2 (the kernel streams weights, a predecessor's tail leaves the
    // memory idle, and nothing here depends on what the predecessor writes)
    if (col < n && lane * 64 < k) prefetch_l2(w + (size_t)col * k + lane * 64);
    pdl_wait();     // x is the predecessor's output; y may still be read by earlier kernels; the early exit is after it
    if (col >= n) return;
    const __half* wr = w + (size_t)col * k;
    for (int r0 = 0; r0 < rows; r0 += 8) {
        const int rn = min(8, rows - r0);
        float acc[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = 0.f;
        for (int kk = lane * 8; kk < k; kk += 256) {
            const uint4 wv = *reinterpret_cast<const uint4*>(wr + kk);
            const __half2* wh = reinterpret_cast<const __half2*>(&wv);
            float2 wf[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) wf[q] = __half22float2(wh[q]);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (j < rn) {
                    const uint4 xv = *reinterpret_cast<const uint4*>(x + (size_t)(r0 + j) * k + kk);
                    const __half2* xh = reinterpret_cast<const __half2*>(&xv);
                    float s = 0.f;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const float2 xf = __half22float2(xh[q]);
                        s += xf.x * wf[q].x + xf.y * wf[q].y;
                    }
                    acc[j] += s;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float v = acc[j];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && j < rn) y[(size_t)(r0 + j) * n + col] = __float2half_rn(v);
        }
    }
}

__global__ void __launch_bounds__(256) gate_k(const float* __restrict__ streams, const __half* __restrict__ key,
                                              const __half* __restrict__ value, const float* __restrict__ wk,
                                              const float* __restrict__ wq, const float* __restrict__ wc,
                                              float* __restrict__ sg, __half* __restrict__ nb, int d, int hc, float eps,
                                              float scale) {
    __shared__ float red[32];
    pdl_trigger();  // conv_k may launch: it waits for this grid before it reads sg and nb
    const int r = blockIdx.x / hc, s = blockIdx.x % hc;
    const size_t base = ((size_t)r * hc + s) * d;  // this stream's slice of a [hc * d] row
    const float* xs = streams + base;
    const __half* kr = key + base;
    const __half* vr = value + (size_t)r * d;
    const float* wks = wk + (size_t)s * d;
    const float* wqs = wq + (size_t)s * d;
    const float* wcs = wc + (size_t)s * d;
    // before the wait: this stream's three weight slices towards L2 (a line is 32 floats)
    for (int i = threadIdx.x * 32; i < d; i += blockDim.x * 32) {
        prefetch_l2(wks + i);
        prefetch_l2(wqs + i);
        prefetch_l2(wcs + i);
    }
    pdl_wait();  // streams, key and value are earlier kernels' outputs; sg and nb may still be read by earlier kernels
    float kk = 0.f, qq = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float a = h2f(kr[i]), b = xs[i];
        kk += a * a;
        qq += b * b;
    }
    kk = block_sum(kk, red);
    qq = block_sum(qq, red);
    const float rk = rsqrtf(kk / (float)d + eps), rq = rsqrtf(qq / (float)d + eps);
    float dot = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) dot += (xs[i] * rq * wqs[i]) * (h2f(kr[i]) * rk * wks[i]);
    dot = block_sum(dot, red);
    const float g = dot * scale;
    const float a = sqrtf(fmaxf(fabsf(g), 1e-6f));
    const float ss = g > 0.f ? a : (g < 0.f ? -a : 0.f);
    const float gate = 1.f / (1.f + expf(-ss));
    float gg = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float v = gate * h2f(vr[i]);
        gg += v * v;
    }
    gg = block_sum(gg, red);
    const float rc = rsqrtf(gg / (float)d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) nb[base + i] = __float2half_rn(gate * h2f(vr[i]) * rc * wcs[i]);
    if (threadIdx.x == 0) sg[(size_t)r * hc + s] = gate;
}

__global__ void __launch_bounds__(256) conv_k(float* __restrict__ streams, const float* __restrict__ sg,
                                              const __half* __restrict__ value, const __half* __restrict__ nb,
                                              const __half* __restrict__ ring, const __half* __restrict__ conv_w,
                                              const int* __restrict__ positions, const int* __restrict__ seq_start,
                                              const int* __restrict__ seq_slot, int n_seq, int d, int hc, int ring_len) {
    pdl_trigger();  // ring_k may launch: it waits for this grid before it reads nb
    const int r = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int ch = hc * d;
    // before the wait: this channel's conv taps, a weight
    float cw[TAPS] = {0.f, 0.f, 0.f, 0.f};
    if (c < ch) {
#pragma unroll
        for (int j = 0; j < TAPS; ++j) cw[j] = h2f(conv_w[(size_t)c * TAPS + j]);
    }
    pdl_wait();  // sg, value, nb, the ring and the positions are earlier kernels' outputs; streams may still be read by earlier kernels
    if (c >= ch) return;
    const int sq = seq_of(seq_start, n_seq, r);
    const int r0 = seq_start[sq];
    const int p0 = positions[r0], p = positions[r];
    const int slot = seq_slot[sq];
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < TAPS; ++j) {
        const int q = p - DIL * (TAPS - 1 - j);
        float v = 0.f;
        if (q >= p0) {
            v = h2f(nb[(size_t)(r - (p - q)) * ch + c]);
        } else if (q >= 0) {
            v = h2f(ring[((size_t)slot * ring_len + (q % ring_len)) * ch + c]);
        }
        acc += cw[j] * v;
    }
    const float y = h2f(__float2half_rn(acc));                       // the conv's f16 output
    const float z = h2f(__float2half_rn(y / (1.f + expf(-y))));      // silu, f16 out
    const int s = c / d, i = c - s * d;
    const float gated = sg[(size_t)r * hc + s] * h2f(value[(size_t)r * d + i]);
    float* x = streams + (size_t)r * ch + c;
    *x = *x + (gated + z);
}

__global__ void __launch_bounds__(256) ring_k(const __half* __restrict__ nb, __half* __restrict__ ring,
                                              const int* __restrict__ positions, const int* __restrict__ seq_start,
                                              const int* __restrict__ seq_slot, int n_seq, int ch, int ring_len) {
    pdl_trigger();  // the successor may launch: it waits for this grid before it reads or writes anything
    pdl_wait();     // nb is conv_k's input and gate_k's output; the ring may still be read by conv_k
    const int r = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    const int sq = seq_of(seq_start, n_seq, r);
    const int p_last = positions[seq_start[sq + 1] - 1];
    const int p = positions[r];
    if (p <= p_last - ring_len) return;
    ring[((size_t)seq_slot[sq] * ring_len + (p % ring_len)) * ch + c] = nb[(size_t)r * ch + c];
}

constexpr int GT = 64, GK = 32;  // gemm tile (rows and columns) and k step

// grid (ceil(n / 64), ceil(rows / 64)), block 256 (16 x 16 threads, 4 x 4 outputs each)
__global__ void __launch_bounds__(256) gemm_k(const __half* __restrict__ x, const __half* __restrict__ w, __half* __restrict__ y,
                                              int rows, int k, int n) {
    __shared__ float xs[GK][GT + 1], ws[GK][GT + 1];
    const int r0 = blockIdx.y * GT, c0 = blockIdx.x * GT;
    const int tr = threadIdx.x / 16, tc = threadIdx.x % 16;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < k; k0 += GK) {
        for (int i = threadIdx.x; i < GT * GK; i += 256) {
            const int rr = i / GK, kk = i % GK, r = r0 + rr, c = c0 + rr;
            xs[kk][rr] = (r < rows && k0 + kk < k) ? h2f(x[(size_t)r * k + k0 + kk]) : 0.f;
            ws[kk][rr] = (c < n && k0 + kk < k) ? h2f(w[(size_t)c * k + k0 + kk]) : 0.f;
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < GK; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = xs[kk][tr * 4 + i];
                b[i] = ws[kk][tc * 4 + i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = acc[i][j] + a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = r0 + tr * 4 + i;
        if (r >= rows) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = c0 + tc * 4 + j;
            if (c < n) y[(size_t)r * n + c] = __float2half_rn(acc[i][j]);
        }
    }
}

}  // namespace

GB_EXPORT int gb_ple_rows(const void* packed, const void* bias, void* emb, int entries, int heads, int k, int words,
                          void* stream) {
    if (entries < 0 || heads <= 0 || k < 1 || k > 8 || words != 1 + ROW * k / 16) return (int)cudaErrorInvalidValue;
    if (entries == 0) return 0;
    gbp_pdl::launch_pdl(flash_pdl(), rows_k, dim3((unsigned)entries), dim3(ROW), (size_t)words * sizeof(uint16_t), gb_stream(stream),
                        (const uint16_t*)packed, (const __half*)bias, (__half*)emb, heads, k, words);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_ple_proj(const void* x, const void* w, void* y, int rows, int k, int n, void* stream) {
    if (rows < 0 || k <= 0 || n <= 0 || k % 8 != 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    gbp_pdl::launch_pdl(flash_pdl(), proj_k, dim3((unsigned)((n + 7) / 8)), dim3(256), 0, gb_stream(stream), (const __half*)x,
                        (const __half*)w, (__half*)y, rows, k, n);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_ple_gemm(const void* x, const void* w, void* y, int rows, int k, int n, void* stream) {
    if (rows < 0 || k <= 0 || n <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    const dim3 grid((n + GT - 1) / GT, (rows + GT - 1) / GT);
    gemm_k<<<grid, 256, 0, gb_stream(stream)>>>((const __half*)x, (const __half*)w, (__half*)y, rows, k, n);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_ple_gate(const void* streams, const void* key, const void* value, const void* wk, const void* wq,
                          const void* wc, void* sg, void* nb, int rows, int d, int hc, float eps, float scale, void* stream) {
    if (rows < 0 || d <= 0 || hc <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    gbp_pdl::launch_pdl(flash_pdl(), gate_k, dim3((unsigned)(rows * hc)), dim3(256), 0, gb_stream(stream), (const float*)streams,
                        (const __half*)key, (const __half*)value, (const float*)wk, (const float*)wq, (const float*)wc, (float*)sg,
                        (__half*)nb, d, hc, eps, scale);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_ple_conv(void* streams, const void* sg, const void* value, const void* nb, const void* ring,
                          const void* conv_w, const int* positions, const int* seq_start, const int* seq_slot, int n_seq,
                          int rows, int d, int hc, int ring_len, void* stream) {
    if (rows < 0 || n_seq <= 0 || d <= 0 || hc <= 0 || ring_len < DIL * (TAPS - 1)) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    const dim3 grid((hc * d + 255) / 256, rows);
    gbp_pdl::launch_pdl(flash_pdl(), conv_k, grid, dim3(256), 0, gb_stream(stream), (float*)streams, (const float*)sg,
                        (const __half*)value, (const __half*)nb, (const __half*)ring, (const __half*)conv_w, positions, seq_start,
                        seq_slot, n_seq, d, hc, ring_len);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_ple_ring(const void* nb, void* ring, const int* positions, const int* seq_start, const int* seq_slot,
                          int n_seq, int rows, int ch, int ring_len, void* stream) {
    if (rows < 0 || n_seq <= 0 || ch <= 0 || ring_len <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    const dim3 grid((ch + 255) / 256, rows);
    gbp_pdl::launch_pdl(flash_pdl(), ring_k, grid, dim3(256), 0, gb_stream(stream), (const __half*)nb, (__half*)ring, positions,
                        seq_start, seq_slot, n_seq, ch, ring_len);
    return GB_CHECK_LAUNCH();
}
