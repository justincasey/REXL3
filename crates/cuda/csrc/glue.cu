// gb-nvcc: --fmad=false
// Glue kernels: embedding, RMSNorm (plain and fused with the residual add), SwiGLU, add, attention prep (q/k norm +
// rope), attention output gate, GDN gated norm, small fp16-weight linear, exact top-k, row copy/gather.
//
// Conventions (see the task brief / spec sections 3, 4, 5):
//   * activations are bf16 (stored as uint16_t), math is fp32, ONE rounding to bf16 at each op boundary;
//   * compiled with --fmad=false: every `a*b+c` below is two separately rounded ops. Where a fused multiply-add is
//     wanted it is written as fmaf() (small linear), so the host mirror in the tests is bit-exact;
//   * ROW INVARIANCE: a row's output bits depend on that row's data only. Reductions use a fixed order that is a
//     function of the row length alone: element ownership by lane/thread, sequential accumulation per owner, fixed
//     xor-butterfly across the lanes of a warp, then a fixed warp-0..W-1 sequential sum. No atomics on data, no
//     dependence on pointer alignment (misaligned pointers are rejected with cudaErrorInvalidValue = 1 instead of
//     taking another code path), no dependence on the number of rows;
//   * normalisation is x * (1.0f / sqrtf(ss / dim + eps)) * w with correctly rounded sqrtf and division, so a plain
//     f32 host loop with the same summation order reproduces the bits.
#include "common.cuh"
#include "pdl.cuh"
#include <math.h>

namespace {

// Flash-Next decode launches (gbp_pdl::kFlash): a kernel launched with the programmatic-dependent-launch attribute may start before
// its predecessor finishes, so it calls pdl_wait() before it reads anything an earlier kernel of the stream wrote or writes anything an
// earlier kernel may still read (a no-op in a launch without the attribute). Only timing changes, never a bit.
using gbp_pdl::pdl_wait;
using gbp_pdl::pdl_trigger;
using gbp_pdl::prefetch_l2;
inline bool flash_pdl() { return gbp_pdl::on(gbp_pdl::kFlash); }

typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

constexpr int ERR_INVALID = 1;  // cudaErrorInvalidValue

__device__ __forceinline__ float bf2f(u16 u) { return __uint_as_float((u32)u << 16); }
__device__ __forceinline__ u16 f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }  // round to bf16 precision

// silu(x) = x / (1 + exp(-x)) in fp32 with the accurate expf (never __expf); sigmoid(x) = 1 / (1 + exp(-x)).
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ void unpack8(const uint4& u, float (&f)[8]) {
    f[0] = __uint_as_float(u.x << 16); f[1] = __uint_as_float(u.x & 0xFFFF0000u);
    f[2] = __uint_as_float(u.y << 16); f[3] = __uint_as_float(u.y & 0xFFFF0000u);
    f[4] = __uint_as_float(u.z << 16); f[5] = __uint_as_float(u.z & 0xFFFF0000u);
    f[6] = __uint_as_float(u.w << 16); f[7] = __uint_as_float(u.w & 0xFFFF0000u);
}
__device__ __forceinline__ u32 pack2(float a, float b) { return (u32)f2bf(a) | ((u32)f2bf(b) << 16); }
__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
    uint4 u;
    u.x = pack2(f[0], f[1]); u.y = pack2(f[2], f[3]); u.z = pack2(f[4], f[5]); u.w = pack2(f[6], f[7]);
    return u;
}

static inline __host__ __device__ bool aligned16(const void* p) { return ((uintptr_t)p & 15) == 0; }
static inline __host__ __device__ bool aligned8(const void* p) { return ((uintptr_t)p & 7) == 0; }

// Sum over a warp with the fixed xor-butterfly 16, 8, 4, 2, 1; every lane ends with the same bits.
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// Sum over a block of NT threads: warp butterfly, then warps 0..NT/32-1 added in order. `sm` has NT/32 floats.
template <int NT>
__device__ __forceinline__ float block_sum(float v, float* sm) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sm[wid] = v;
    __syncthreads();
    float s = sm[0];
#pragma unroll
    for (int w = 1; w < NT / 32; ++w) s = s + sm[w];
    __syncthreads();  // sm may be reused by the caller
    return s;
}

// ---------------------------------------------------------------------------------------------------- embedding

__global__ void embed_vec_k(const u16* table, const int* ids, u16* out, int dim, int vocab) {
    const int row = blockIdx.x;
    const int id = ids[row];
    const uint4* src = (const uint4*)(table + (size_t)(unsigned)id * dim);
    uint4* dst = (uint4*)(out + (size_t)row * dim);
    const bool ok = (unsigned)id < (unsigned)vocab;
    for (int v = threadIdx.x; v < dim / 8; v += blockDim.x) dst[v] = ok ? src[v] : make_uint4(0, 0, 0, 0);
}
__global__ void embed_scalar_k(const u16* table, const int* ids, u16* out, int dim, int vocab) {
    const int row = blockIdx.x;
    const int id = ids[row];
    const bool ok = (unsigned)id < (unsigned)vocab;
    const u16* src = table + (size_t)(unsigned)id * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) out[(size_t)row * dim + d] = ok ? src[d] : (u16)0;
}

// ---------------------------------------------------------------------------------------------------- rmsnorm

constexpr int RMS_NT = 256;

// y = bf16((x * inv) * w), inv = 1 / sqrtf(ss / dim + eps), ss = sum x^2 in the fixed order described at the top:
// thread t owns 8-element vectors t, t+256, ...; it adds f[0]^2 .. f[7]^2 of each in turn.
// ADD: x is first replaced by h = bf16(x + res), which is stored to h_out and is what gets normalised.
// h_out may alias x or res, and y may alias x or res (each element is read, then written, by one thread).
template <bool ADD>
__global__ void __launch_bounds__(RMS_NT) rmsnorm_k(const u16* x, const u16* res, const float* w, u16* h_out, u16* y, int dim,
                                                    float eps) {
    __shared__ float sm[RMS_NT / 32];
    const size_t base = (size_t)blockIdx.x * dim;
    const int nvec = dim / 8;
    float ss = 0.f;
    for (int v = threadIdx.x; v < nvec; v += RMS_NT) {
        float f[8];
        unpack8(*(const uint4*)(x + base + (size_t)v * 8), f);
        if constexpr (ADD) {
            float r[8];
            unpack8(*(const uint4*)(res + base + (size_t)v * 8), r);
#pragma unroll
            for (int i = 0; i < 8; ++i) f[i] = rbf(f[i] + r[i]);
            *(uint4*)(h_out + base + (size_t)v * 8) = pack8(f);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) ss = ss + f[i] * f[i];
    }
    ss = block_sum<RMS_NT>(ss, sm);
    const float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    const u16* src = ADD ? h_out : x;
    for (int v = threadIdx.x; v < nvec; v += RMS_NT) {
        float f[8], o[8];
        unpack8(*(const uint4*)(src + base + (size_t)v * 8), f);
        const float4 w0 = *(const float4*)(w + (size_t)v * 8), w1 = *(const float4*)(w + (size_t)v * 8 + 4);
        const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = (f[i] * inv) * wv[i];
        *(uint4*)(y + base + (size_t)v * 8) = pack8(o);
    }
}

// ---------------------------------------------------------------------------------------------------- elementwise

struct OpSwiglu { __device__ static float f(float g, float u) { return silu_f(g) * u; } };
struct OpAdd { __device__ static float f(float a, float b) { return a + b; } };
struct OpGate { __device__ static float f(float o, float g) { return o * sigmoid_f(g); } };

template <class Op>
__global__ void ew_vec_k(const u16* a, const u16* b, u16* out, size_t n8) {
    pdl_wait();
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += (size_t)gridDim.x * blockDim.x) {
        float x[8], y[8], o[8];
        unpack8(((const uint4*)a)[i], x);
        unpack8(((const uint4*)b)[i], y);
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] = Op::f(x[k], y[k]);
        ((uint4*)out)[i] = pack8(o);
    }
}
template <class Op>
__global__ void ew_scalar_k(const u16* a, const u16* b, u16* out, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        out[i] = f2bf(Op::f(bf2f(a[i]), bf2f(b[i])));
}

template <class Op>
int launch_ew(const void* a, const void* b, void* out, long long n, void* stream) {
    if (n < 0) return ERR_INVALID;
    if (n == 0) return 0;
    const bool vec = n % 8 == 0 && aligned16(a) && aligned16(b) && aligned16(out);
    const size_t units = vec ? (size_t)n / 8 : (size_t)n;
    size_t blocks = (units + 255) / 256;
    if (blocks > 1u << 20) blocks = 1u << 20;
    if (vec)
        gbp_pdl::launch_pdl(flash_pdl(), ew_vec_k<Op>, dim3((unsigned)blocks), dim3(256), 0, gb_stream(stream), (const u16*)a,
                            (const u16*)b, (u16*)out, units);
    else
        ew_scalar_k<Op><<<(unsigned)blocks, 256, 0, gb_stream(stream)>>>((const u16*)a, (const u16*)b, (u16*)out, units);
    return GB_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------------------------------- attention prep

struct RopeTab { float inv[128]; };  // inv_freq[i], i < rotary_dim / 2 (<= 128), fp64 pow on the host then rounded to fp32

// One block per row; warp w handles heads w, w+nwarps, ... (heads 0..H-1 are q heads, H..H+HKV-1 k heads).
// Lane l owns elements l + 32*j, j < J = head_dim / 32. norm: ss = sum_j x_j^2 in j order, then the butterfly.
// n = bf16((x * inv) * w); rope pairs (d, d + rotary/2) sit in the same lane (j and j + rotary/64); rotated results are
// rounded once: o1 = a*cos - b*sin, o2 = b*cos + a*sin (every op rounded, no fma).
// pos3 (nullable): int32[rows][3] = the T, H and W position of each row for interleaved M-RoPE; frequency pair i uses stream
// i % 3 (T, H, W) with the H stream only below i = 33 and the W stream only below i = 30 (sections 11, 11, 10). Without it the
// single `pos[row]` drives every pair, and that path is unchanged, so text-only rows keep their bits.
template <int J, int JH>
__global__ void attn_prep_k(const u16* q_in, const u16* k_in, const int* pos, const int* pos3, const float* qw, const float* kw,
                            u16* q_out, u16* gate_out, u16* k_out, int H, int HKV, int rotary, int has_gate, float eps,
                            RopeTab tab) {
    pdl_wait();
    constexpr int HD = J * 32;
    __shared__ float cs[128], sn[128];
    const int row = blockIdx.x;
    const int half = rotary / 2;  // == 32 * JH
    if (JH > 0) {
        for (int i = threadIdx.x; i < half; i += blockDim.x) {
            int pi;
            if (pos3) {
                const int axis = (i % 3 == 1 && i < 33) ? 1 : ((i % 3 == 2 && i < 30) ? 2 : 0);
                pi = pos3[row * 3 + axis];
            } else {
                pi = pos[row];
            }
            const float ang = (float)pi * tab.inv[i];
            cs[i] = cosf(ang);
            sn[i] = sinf(ang);
        }
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int qstride = has_gate ? 2 * HD : HD;
    for (int h = wid; h < H + HKV; h += nw) {
        const bool isq = h < H;
        const u16* src = isq ? q_in + ((size_t)row * H + h) * qstride : k_in + ((size_t)row * HKV + (h - H)) * HD;
        const float* w = isq ? qw : kw;
        u16* dst = isq ? q_out + ((size_t)row * H + h) * HD : k_out + ((size_t)row * HKV + (h - H)) * HD;
        float x[J];
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < J; ++j) {
            x[j] = bf2f(src[lane + 32 * j]);
            ss = ss + x[j] * x[j];
        }
        ss = warp_sum(ss);
        const float inv = 1.0f / sqrtf(ss / (float)HD + eps);
#pragma unroll
        for (int j = 0; j < J; ++j) x[j] = rbf((x[j] * inv) * w[lane + 32 * j]);
        float o[J];
#pragma unroll
        for (int j = 0; j < J; ++j) o[j] = x[j];
#pragma unroll
        for (int j = 0; j < JH; ++j) {  // JH = rotary_dim / 64: the pair distance in units of 32 elements
            const int d = lane + 32 * j;
            const float c = cs[d], s = sn[d];
            const float a = x[j], b = x[j + JH];
            o[j] = a * c - b * s;
            o[j + JH] = b * c + a * s;
        }
#pragma unroll
        for (int j = 0; j < J; ++j) dst[lane + 32 * j] = f2bf(o[j]);
        if (isq && has_gate) {
            u16* gd = gate_out + ((size_t)row * H + h) * HD;
#pragma unroll
            for (int j = 0; j < J; ++j) gd[lane + 32 * j] = src[HD + lane + 32 * j];
        }
    }
}

// ---------------------------------------------------------------------------------------------------- gated rmsnorm

// Block per row, warp per head; lane owns EPL = dim / 32 contiguous elements [lane*EPL, (lane+1)*EPL).
// out = bf16( ((o * inv) * w) * silu(z) ), inv = 1 / sqrtf(ss / dim + eps), all fp32.
template <int EPL>
__global__ void gated_rmsnorm_k(const u16* o, const u16* z, const float* w, u16* out, int heads, float eps) {
    constexpr int DIM = EPL * 32;
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int h = wid; h < heads; h += nw) {
        const size_t off = ((size_t)row * heads + h) * DIM + lane * EPL;
        float ov[EPL], zv[EPL];
        if constexpr (EPL == 8) {
            unpack8(*(const uint4*)(o + off), ov);
            unpack8(*(const uint4*)(z + off), zv);
        } else {  // EPL == 4
#pragma unroll
            for (int i = 0; i < EPL; i += 2) {
                const u32 a = *(const u32*)(o + off + i), b = *(const u32*)(z + off + i);
                ov[i] = __uint_as_float(a << 16); ov[i + 1] = __uint_as_float(a & 0xFFFF0000u);
                zv[i] = __uint_as_float(b << 16); zv[i + 1] = __uint_as_float(b & 0xFFFF0000u);
            }
        }
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < EPL; ++i) ss = ss + ov[i] * ov[i];
        ss = warp_sum(ss);
        const float inv = 1.0f / sqrtf(ss / (float)DIM + eps);
        u16 r[EPL];
#pragma unroll
        for (int i = 0; i < EPL; ++i) {
            const float yn = (ov[i] * inv) * w[lane * EPL + i];
            r[i] = f2bf(yn * silu_f(zv[i]));
        }
        if constexpr (EPL == 8) {
            uint4 u;
            u.x = r[0] | ((u32)r[1] << 16); u.y = r[2] | ((u32)r[3] << 16);
            u.z = r[4] | ((u32)r[5] << 16); u.w = r[6] | ((u32)r[7] << 16);
            *(uint4*)(out + off) = u;
        } else {
#pragma unroll
            for (int i = 0; i < EPL; i += 2) *(u32*)(out + off + i) = r[i] | ((u32)r[i + 1] << 16);
        }
    }
}

// ---------------------------------------------------------------------------------------------------- small linear

constexpr int SL_NT = 256;
constexpr int SL_RT = 8;  // rows per block; a row's accumulation never depends on the tile it sits in

// y[r, j] = sum_k x[r, k] * w[j, k]. Block (j, row tile); thread t owns 8-vectors t, t+256, ...; acc = fmaf(x, w, acc)
// in element order; then the block_sum order (butterfly, warps 0..7). The product of a bf16 and an fp16 is exact in fp32.
template <bool F32OUT>
__global__ void __launch_bounds__(SL_NT) small_linear_k(const u16* x, const __half* w, void* y, int rows, int k, int n) {
    __shared__ float sm[SL_RT][SL_NT / 32];
    const int j = blockIdx.x;
    // the weight row is no kernel's output: bring it toward L2 while the predecessor still runs
    if (threadIdx.x < (unsigned)((k + 63) / 64)) prefetch_l2(w + (size_t)j * k + (size_t)threadIdx.x * 64);
    pdl_wait();
    const int r0 = blockIdx.y * SL_RT;
    const int nvec = k / 8;
    float acc[SL_RT];
#pragma unroll
    for (int r = 0; r < SL_RT; ++r) acc[r] = 0.f;
    const uint4* wr = (const uint4*)(w + (size_t)j * k);
    for (int v = threadIdx.x; v < nvec; v += SL_NT) {
        const uint4 wu = __ldg(wr + v);
        float wf[8];
        {
            const __half2* h2 = (const __half2*)&wu;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f = __half22float2(h2[i]);
                wf[2 * i] = f.x; wf[2 * i + 1] = f.y;
            }
        }
#pragma unroll
        for (int r = 0; r < SL_RT; ++r) {
            if (r0 + r < rows) {
                float xf[8];
                unpack8(__ldg((const uint4*)(x + (size_t)(r0 + r) * k) + v), xf);
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[r] = fmaf(xf[i], wf[i], acc[r]);
            }
        }
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
    for (int r = 0; r < SL_RT; ++r) {
        const float s = warp_sum(acc[r]);
        if (lane == 0) sm[r][wid] = s;
    }
    __syncthreads();
    if (threadIdx.x < SL_RT && r0 + (int)threadIdx.x < rows) {
        float s = sm[threadIdx.x][0];
        for (int wv = 1; wv < SL_NT / 32; ++wv) s = s + sm[threadIdx.x][wv];
        const size_t o = (size_t)(r0 + threadIdx.x) * n + j;
        if (F32OUT) ((float*)y)[o] = s;
        else ((u16*)y)[o] = f2bf(s);
    }
}

// The same arithmetic as small_linear_k, bit for bit, for MANY rows (a prefill chunk): a block is R rows by J columns, SLR_R x SLR_J
// outputs a thread-slice instead of 1. Thread t still owns the 8-vectors t, t+256, ... of k and runs one fmaf chain per output over
// them in element order; the chains are then added in the same tree (xor butterfly 16, 8, 4, 2, 1 inside a warp, then the 8 warps in
// order). The butterfly is done as a reduce-scatter over 32 outputs at a time: at each stage a lane keeps half of its outputs and
// adds the partner's copy of them, which is the add the butterfly makes (a + b is commutative bit for bit): 31 shuffles for 32
// outputs instead of 160. Every x row is read once a J-column group and every w row once an R-row group instead of once a row.
#ifndef SLR_R
#define SLR_R 4
#endif
#ifndef SLR_J
#define SLR_J 8
#endif
#ifndef SLR_MINB
#define SLR_MINB 2
#endif
template <bool F32OUT>
__global__ void __launch_bounds__(SL_NT, SLR_MINB) small_linear_rows_k(const u16* x, const __half* w, void* y, int rows, int k, int n) {
    constexpr int ITEMS = SLR_R * SLR_J;      // outputs a thread holds
    constexpr int GI = ITEMS >= 32 ? 32 : 16;  // outputs reduced together: 32 lanes scatter 32 (or 16, plus one more butterfly stage)
    constexpr int GROUPS = ITEMS / GI;
    static_assert(ITEMS == 64 || ITEMS == 32 || ITEMS == 16, "outputs a block must be 64, 32 or 16");
    __shared__ float sm[GROUPS][SL_NT / 32][GI];
    const int j0 = blockIdx.x * SLR_J;
    const int r0 = blockIdx.y * SLR_R;
    const int nvec = k / 8;
    float acc[SLR_R][SLR_J];
#pragma unroll
    for (int r = 0; r < SLR_R; ++r)
#pragma unroll
        for (int j = 0; j < SLR_J; ++j) acc[r][j] = 0.f;
    for (int v = threadIdx.x; v < nvec; v += SL_NT) {
        float xf[SLR_R][8];
#pragma unroll
        for (int r = 0; r < SLR_R; ++r) {
            const uint4 xu = (r0 + r < rows) ? __ldg((const uint4*)(x + (size_t)(r0 + r) * k) + v) : make_uint4(0u, 0u, 0u, 0u);
            unpack8(xu, xf[r]);
        }
#pragma unroll
        for (int j = 0; j < SLR_J; ++j) {
            const uint4 wu = (j0 + j < n) ? __ldg((const uint4*)(w + (size_t)(j0 + j) * k) + v) : make_uint4(0u, 0u, 0u, 0u);
            float wf[8];
            const __half2* h2 = (const __half2*)&wu;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f = __half22float2(h2[i]);
                wf[2 * i] = f.x;
                wf[2 * i + 1] = f.y;
            }
#pragma unroll
            for (int r = 0; r < SLR_R; ++r)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[r][j] = fmaf(xf[r][i], wf[i], acc[r][j]);
        }
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
    for (int q = 0; q < GROUPS; ++q) {
        float s[GI];  // output q * GI + i = (r * SLR_J + j)
#pragma unroll
        for (int i = 0; i < GI; ++i) s[i] = acc[(q * GI + i) / SLR_J][(q * GI + i) % SLR_J];
#pragma unroll
        for (int m = 16, half = GI / 2; half >= 1; m >>= 1, half >>= 1) {
            const bool hi = (lane & m) != 0;
#pragma unroll
            for (int i = 0; i < half; ++i) {
                const float send = hi ? s[i] : s[i + half];
                const float own = hi ? s[i + half] : s[i];
                s[i] = own + __shfl_xor_sync(0xffffffffu, send, m);
            }
        }
        if (GI == 16) s[0] = s[0] + __shfl_xor_sync(0xffffffffu, s[0], 1);  // the last butterfly stage (m = 1): lanes l, l ^ 1 agree
        // lane l holds output l of the group (32 items) or l >> 1 (16 items)
        if (GI == 32 || (lane & 1) == 0) sm[q][wid][GI == 32 ? lane : lane >> 1] = s[0];
    }
    __syncthreads();
    if (threadIdx.x < ITEMS) {
        const int it = threadIdx.x;
        const int q = it / GI, i = it % GI;
        float t = sm[q][0][i];
        for (int wv = 1; wv < SL_NT / 32; ++wv) t = t + sm[q][wv][i];
        const int r = r0 + it / SLR_J, j = j0 + it % SLR_J;
        if (r < rows && j < n) {
            const size_t o = (size_t)r * n + j;
            if (F32OUT) ((float*)y)[o] = t;
            else ((u16*)y)[o] = f2bf(t);
        }
    }
}

// The same arithmetic again for k <= 2,048 (one 8-vector a thread, so no fmaf chain spans two vectors): a block keeps the weight
// vectors of its SLC_J columns in registers (thread t's vector t of each) and walks a slice of the rows, four at a time. The weights
// cross the L2 once a row slice instead of once a row tile; what remains is the activations, once a column group. The fmaf chains
// and the reduction tree are the other kernels' (a 32-output reduce-scatter per four rows).
constexpr int SLC_J = 8;
constexpr int SLC_RG = 4;
template <bool F32OUT>
__global__ void __launch_bounds__(SL_NT, 2) small_linear_cols_k(const u16* x, const __half* w, void* y, int rows, int k, int n,
                                                                int rows_per_block) {
    __shared__ float sm[2][SL_NT / 32][32];
    const int j0 = blockIdx.x * SLC_J;
    const int rb = blockIdx.y * rows_per_block;
    const int re = min(rows, rb + rows_per_block);
    const int v = threadIdx.x;
    const bool live = v < k / 8;
    uint4 wv[SLC_J];
#pragma unroll
    for (int j = 0; j < SLC_J; ++j)
        wv[j] = (live && j0 + j < n) ? __ldg((const uint4*)(w + (size_t)(j0 + j) * k) + v) : make_uint4(0u, 0u, 0u, 0u);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int buf = 0;
    for (int r0 = rb; r0 < re; r0 += SLC_RG, buf ^= 1) {
        float acc[SLC_RG][SLC_J];
#pragma unroll
        for (int r = 0; r < SLC_RG; ++r)
#pragma unroll
            for (int j = 0; j < SLC_J; ++j) acc[r][j] = 0.f;
#pragma unroll
        for (int r = 0; r < SLC_RG; ++r) {
            const uint4 xu = (live && r0 + r < re) ? __ldg((const uint4*)(x + (size_t)(r0 + r) * k) + v) : make_uint4(0u, 0u, 0u, 0u);
            float xf[8];
            unpack8(xu, xf);
#pragma unroll
            for (int j = 0; j < SLC_J; ++j) {
                float wf[8];
                const __half2* h2 = (const __half2*)&wv[j];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 f = __half22float2(h2[i]);
                    wf[2 * i] = f.x;
                    wf[2 * i + 1] = f.y;
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[r][j] = fmaf(xf[i], wf[i], acc[r][j]);
            }
        }
        float s[32];  // output r * SLC_J + j
#pragma unroll
        for (int it = 0; it < 32; ++it) s[it] = acc[it / SLC_J][it % SLC_J];
#pragma unroll
        for (int m = 16, half = 16; half >= 1; m >>= 1, half >>= 1) {
            const bool hi = (lane & m) != 0;
#pragma unroll
            for (int i = 0; i < half; ++i) {
                const float send = hi ? s[i] : s[i + half];
                const float own = hi ? s[i + half] : s[i];
                s[i] = own + __shfl_xor_sync(0xffffffffu, send, m);
            }
        }
        sm[buf][wid][lane] = s[0];
        __syncthreads();
        if (threadIdx.x < 32) {
            const int it = threadIdx.x;
            float t = sm[buf][0][it];
            for (int wv2 = 1; wv2 < SL_NT / 32; ++wv2) t = t + sm[buf][wv2][it];
            const int r = r0 + it / SLC_J, j = j0 + it % SLC_J;
            if (r < re && j < n) {
                const size_t o = (size_t)r * n + j;
                if (F32OUT) ((float*)y)[o] = t;
                else ((u16*)y)[o] = f2bf(t);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------- top-k
//
// Exact top-`count` per row by (value DESC, id ASC). Every element becomes a unique u64 key
//   key = (ordered16(bf16 bits) << 32) | (0xFFFFFFFF - id)
// so "larger key is better" is a strict total order and ties between blocks resolve to the lower id with no special
// case; the selected set is independent of how the vocabulary is split. NaN is mapped to -inf, -0 to +0 before ordering.
// Key 0 is never produced and marks an empty slot.
//
// Stage 1: block (chunk, row) streams its chunk in tiles of 1024 keys through a sorted pool of P >= count keys held in
// shared memory. A key joins the candidate list only if it beats the current count-th pool key (a gate that is exact:
// anything below it can never be in the top count); the list is sorted (bitonic) and merged into the pool when it holds
// >= 512 keys and at the end. Stage 2: one block per row runs the same machinery over the nchunks*P partial pools.

constexpr int TK_NT = 256;
constexpr int TK_TILE = TK_NT * 4;
constexpr int TK_CAP = 2048;  // candidate list capacity: < 512 left over + one tile of 1024
constexpr int TK_FLUSH = 512;
constexpr int TK_PF = 2;  // tiles of loads kept in flight
constexpr int TK_CHUNK = 16384;

__device__ __forceinline__ u64 make_key(u16 u, u32 id) {
    if ((u & 0x7FFF) > 0x7F80) u = 0xFF80;  // NaN -> -inf
    else if ((u & 0x7FFF) == 0) u = 0;      // -0 -> +0
    const u32 t = (u & 0x8000) ? (~(u32)u & 0xFFFF) : ((u32)u | 0x8000);
    return ((u64)t << 32) | (u64)(~id);
}

// Bitonic sort of arr[0..M) descending; M a power of two. Contains __syncthreads.
__device__ void bitonic_sort_desc(u64* arr, int M) {
    for (int k = 2; k <= M; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int p = threadIdx.x; p < M / 2; p += TK_NT) {
                const int i = ((p & ~(j - 1)) << 1) | (p & (j - 1));
                const int l = i | j;
                const u64 a = arr[i], b = arr[l];
                const bool desc = (i & k) == 0;
                if (desc ? a < b : a > b) { arr[i] = b; arr[l] = a; }
            }
            __syncthreads();
        }
    }
}

template <int P>
struct TkShared {
    u64 pool[P];
    u64 cand[TK_CAP];
    int cnt;
    u64 thr;
};

// Merge the candidate list into the pool and refresh the gate. All threads call this together.
template <int P>
__device__ void tk_merge(TkShared<P>& s, int count) {
    __syncthreads();  // everybody has read s.cnt
    const int cnt = s.cnt;
    int M = P;
    while (M < cnt) M <<= 1;
    for (int i = cnt + threadIdx.x; i < M; i += TK_NT) s.cand[i] = 0;
    __syncthreads();
    bitonic_sort_desc(s.cand, M);
    if (threadIdx.x < P) {
        const u64 a = s.pool[threadIdx.x], b = s.cand[P - 1 - threadIdx.x];
        s.pool[threadIdx.x] = a > b ? a : b;
    }
    __syncthreads();
    for (int j = P / 2; j > 0; j >>= 1) {
        u64 a = 0, b = 0;
        int i = 0, l = 0;
        const bool act = threadIdx.x < P && (threadIdx.x & j) == 0;
        if (act) {
            i = threadIdx.x; l = i | j;
            a = s.pool[i]; b = s.pool[l];
            if (a < b) { s.pool[i] = b; s.pool[l] = a; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { s.cnt = 0; s.thr = s.pool[count - 1]; }
    __syncthreads();
}

// Stream `len` keys (Src::load4(idx) gives keys idx..idx+3, 0 past the end) through the pool; leaves the sorted pool.
template <int P, class Src>
__device__ void tk_select(TkShared<P>& s, const Src& src, int len, int count) {
    if (threadIdx.x < P) s.pool[threadIdx.x] = 0;
    if (threadIdx.x == 0) { s.cnt = 0; s.thr = 0; }
    __syncthreads();
    const int ntile = (len + TK_TILE - 1) / TK_TILE;
    const int lane = threadIdx.x & 31;
    // Memory-level parallelism: TK_PF tiles of loads are in flight while one is being filtered.
    u64 ring[TK_PF][4];
#pragma unroll
    for (int u = 0; u < TK_PF; ++u) src.load4(u * TK_TILE + threadIdx.x * 4, ring[u]);
    for (int t0 = 0; t0 < ntile; t0 += TK_PF) {
#pragma unroll
        for (int u = 0; u < TK_PF; ++u) {
            const int t = t0 + u;
            if (t < ntile) {
                u64 cur[4];
#pragma unroll
                for (int q = 0; q < 4; ++q) cur[q] = ring[u][q];
                src.load4((t + TK_PF) * TK_TILE + threadIdx.x * 4, ring[u]);
                const u64 thr = s.thr;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const bool pass = cur[q] > thr;
                    const u32 mask = __ballot_sync(0xffffffffu, pass);
                    if (mask) {
                        int base = 0;
                        if (lane == 0) base = atomicAdd(&s.cnt, __popc(mask));
                        base = __shfl_sync(0xffffffffu, base, 0);
                        if (pass) s.cand[base + __popc(mask & ((1u << lane) - 1))] = cur[q];
                    }
                }
                __syncthreads();
                const int cnt = s.cnt;
                if (cnt >= TK_FLUSH || (t == ntile - 1 && cnt > 0)) tk_merge<P>(s, count);
                else __syncthreads();  // the next tile appends to the list; everyone must have read cnt
            }
        }
    }
}

struct Stage1Src {
    const u16* row;  // logits row
    int lo, hi;      // chunk [lo, hi)
    bool vec;        // 8-byte loads allowed
    __device__ void load4(int idx, u64 (&k)[4]) const {
        const int e = lo + idx;
        if (vec && e + 4 <= hi) {
            const uint2 u = *(const uint2*)(row + e);
            k[0] = make_key(u.x & 0xFFFF, e); k[1] = make_key(u.x >> 16, e + 1);
            k[2] = make_key(u.y & 0xFFFF, e + 2); k[3] = make_key(u.y >> 16, e + 3);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) k[i] = (e + i < hi) ? make_key(row[e + i], e + i) : 0;
        }
    }
};

struct Stage2Src {
    const u64* keys;
    int len;
    __device__ void load4(int idx, u64 (&k)[4]) const {
        if (idx + 4 <= len) {
            const ulonglong2 a = *(const ulonglong2*)(keys + idx), b = *(const ulonglong2*)(keys + idx + 2);
            k[0] = a.x; k[1] = a.y; k[2] = b.x; k[3] = b.y;
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) k[i] = (idx + i < len) ? keys[idx + i] : 0;
        }
    }
};

// `stride`: elements between rows of the logits; `valid` <= stride: only columns [0, valid) of a row are read (an lm_head padded to a
// multiple of 128 leaves columns [valid, stride) holding whatever the padded rows of the weights produced, and they must never be a
// candidate). valid == stride is the whole row, and every bit of it is what the kernel gave before `valid` existed.
template <int P>
__global__ void __launch_bounds__(TK_NT) topk_stage1_k(const u16* logits, u64* ws, int stride, int valid, int nchunks, int count) {
    pdl_wait();
    __shared__ TkShared<P> s;
    const int chunk = blockIdx.x, row = blockIdx.y;
    Stage1Src src;
    src.row = logits + (size_t)row * stride;
    src.lo = chunk * TK_CHUNK;
    src.hi = min(valid, src.lo + TK_CHUNK);
    src.vec = (stride % 4 == 0) && aligned8(logits);
    tk_select<P>(s, src, src.hi - src.lo, count);
    if (threadIdx.x < P) ws[((size_t)row * nchunks + chunk) * P + threadIdx.x] = s.pool[threadIdx.x];
}

template <int P>
__global__ void __launch_bounds__(TK_NT) topk_stage2_k(const u64* ws, int* ids, float* vals, int nchunks, int count) {
    pdl_wait();
    __shared__ TkShared<P> s;
    const int row = blockIdx.x;
    Stage2Src src;
    src.keys = ws + (size_t)row * nchunks * P;
    src.len = nchunks * P;
    tk_select<P>(s, src, src.len, count);
    if (threadIdx.x < count) {
        const u64 key = s.pool[threadIdx.x];
        int id = -1;
        float v = -INFINITY;
        if (key != 0) {
            id = (int)~(u32)key;
            const u32 t = (u32)(key >> 32);
            const u32 u = (t & 0x8000) ? (t & 0x7FFF) : (~t & 0xFFFF);
            v = __uint_as_float(u << 16);
        }
        ids[(size_t)row * count + threadIdx.x] = id;
        vals[(size_t)row * count + threadIdx.x] = v;
    }
}

static int topk_p(int count) { return count <= 16 ? 16 : count <= 32 ? 32 : 64; }
static size_t topk_nchunks(int vocab) { return ((size_t)vocab + TK_CHUNK - 1) / TK_CHUNK; }

template <int P>
int launch_topk(const void* logits, int rows, int stride, int valid, int count, int* ids, float* vals, u64* ws, void* stream) {
    const int nch = (int)topk_nchunks(valid);
    gbp_pdl::launch_pdl(flash_pdl(), topk_stage1_k<P>, dim3(nch, rows), dim3(TK_NT), 0, gb_stream(stream), (const u16*)logits, ws, stride,
                        valid, nch, count);
    gbp_pdl::launch_pdl(flash_pdl(), topk_stage2_k<P>, dim3(rows), dim3(TK_NT), 0, gb_stream(stream), ws, ids, vals, nch, count);
    return GB_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------------------------------- row copy / gather

__global__ void copy_rows_vec_k(const u16* src, long long ss, u16* dst, long long ds, int w8) {
    const uint4* s = (const uint4*)(src + (size_t)blockIdx.x * ss);
    uint4* d = (uint4*)(dst + (size_t)blockIdx.x * ds);
    for (int v = threadIdx.x; v < w8; v += blockDim.x) d[v] = s[v];
}
__global__ void copy_rows_scalar_k(const u16* src, long long ss, u16* dst, long long ds, int w) {
    const u16* s = src + (size_t)blockIdx.x * ss;
    u16* d = dst + (size_t)blockIdx.x * ds;
    for (int v = threadIdx.x; v < w; v += blockDim.x) d[v] = s[v];
}
__global__ void gather_rows_vec_k(const u16* src, const int* idx, u16* dst, int w) {
    const int i = idx[blockIdx.x];
    const bool ok = i >= 0;
    const uint4* s = (const uint4*)(src + (size_t)(ok ? i : 0) * w);
    uint4* d = (uint4*)(dst + (size_t)blockIdx.x * w);
    for (int v = threadIdx.x; v < w / 8; v += blockDim.x) d[v] = ok ? s[v] : make_uint4(0, 0, 0, 0);
}
__global__ void gather_rows_scalar_k(const u16* src, const int* idx, u16* dst, int w) {
    const int i = idx[blockIdx.x];
    const bool ok = i >= 0;
    const u16* s = src + (size_t)(ok ? i : 0) * w;
    u16* d = dst + (size_t)blockIdx.x * w;
    for (int v = threadIdx.x; v < w; v += blockDim.x) d[v] = ok ? s[v] : (u16)0;
}

}  // namespace

// ==================================================================================================== launchers

GB_EXPORT int gb_embed(const void* table, const int* ids, void* out, int rows, int dim, int vocab, void* stream) {
    if (rows < 0 || dim <= 0 || vocab < 0) return ERR_INVALID;
    if (rows == 0) return 0;
    if (dim % 8 == 0 && aligned16(table) && aligned16(out))
        embed_vec_k<<<rows, 128, 0, gb_stream(stream)>>>((const u16*)table, ids, (u16*)out, dim, vocab);
    else
        embed_scalar_k<<<rows, 128, 0, gb_stream(stream)>>>((const u16*)table, ids, (u16*)out, dim, vocab);
    return GB_CHECK_LAUNCH();
}

// y = bf16((x * rsqrt(mean(x^2) + eps)) * w). dim % 8 == 0; x, y 16-byte aligned; w f32[dim] 16-byte aligned. In place ok.
GB_EXPORT int gb_rmsnorm(const void* x, const float* w, void* y, int rows, int dim, float eps, void* stream) {
    if (rows < 0 || dim <= 0 || dim % 8 != 0) return ERR_INVALID;
    if (!aligned16(x) || !aligned16(y) || !aligned16(w)) return ERR_INVALID;
    if (rows == 0) return 0;
    rmsnorm_k<false><<<rows, RMS_NT, 0, gb_stream(stream)>>>((const u16*)x, nullptr, w, nullptr, (u16*)y, dim, eps);
    return GB_CHECK_LAUNCH();
}

// h = bf16(x + res) -> h_out; y = rmsnorm(h). h_out may alias x or res; y may alias x or res (not h_out).
GB_EXPORT int gb_add_rmsnorm(const void* x, const void* res, const float* w, void* h_out, void* y, int rows, int dim, float eps,
                             void* stream) {
    if (rows < 0 || dim <= 0 || dim % 8 != 0) return ERR_INVALID;
    if (!aligned16(x) || !aligned16(res) || !aligned16(h_out) || !aligned16(y) || !aligned16(w)) return ERR_INVALID;
    if (h_out == y) return ERR_INVALID;
    if (rows == 0) return 0;
    rmsnorm_k<true><<<rows, RMS_NT, 0, gb_stream(stream)>>>((const u16*)x, (const u16*)res, w, (u16*)h_out, (u16*)y, dim, eps);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_swiglu(const void* g, const void* u, void* out, long long n, void* stream) {
    return launch_ew<OpSwiglu>(g, u, out, n, stream);
}
GB_EXPORT int gb_add(const void* a, const void* b, void* out, long long n, void* stream) {
    return launch_ew<OpAdd>(a, b, out, n, stream);
}
GB_EXPORT int gb_gate_mul(const void* o, const void* gate, void* out, long long n, void* stream) {
    return launch_ew<OpGate>(o, gate, out, n, stream);
}

// q_in: has_gate ? [rows, heads, 2*head_dim] (q | gate per head) : [rows, heads, head_dim]; k_in [rows, kv_heads, head_dim].
// head_dim in {64, 128, 256, 512}; rotary_dim in {0 (none), 64, 128, 256}, <= head_dim.
// q_out [rows, heads, head_dim]; gate_out [rows, heads, head_dim] (ignored, may be null, when !has_gate); k_out like k_in.
// positions int32[rows], absolute. q_w/k_w f32[head_dim] with any +1 already folded in.
static int attn_prep_launch(const void* q_in, const void* k_in, const int* positions, const int* pos3, const float* q_w,
                            const float* k_w, void* q_out, void* gate_out, void* k_out, int rows, int heads, int kv_heads,
                            int head_dim, int rotary_dim, int has_gate, float eps, float theta, void* stream) {
    if (rows < 0 || heads <= 0 || kv_heads <= 0) return ERR_INVALID;
    if (head_dim != 64 && head_dim != 128 && head_dim != 256 && head_dim != 512) return ERR_INVALID;
    if (rotary_dim < 0 || rotary_dim > head_dim || (rotary_dim != 0 && rotary_dim != 64 && rotary_dim != 128 && rotary_dim != 256)) return ERR_INVALID;
    if (has_gate && !gate_out) return ERR_INVALID;
    if (rows == 0) return 0;
    RopeTab tab;
    for (int i = 0; i < 128; ++i) tab.inv[i] = 0.f;
    for (int i = 0; i < rotary_dim / 2; ++i)
        tab.inv[i] = (float)pow((double)theta, -(double)(2 * i) / (double)rotary_dim);
    const int nw = (heads + kv_heads) < 32 ? (heads + kv_heads) : 32;
    const dim3 grid(rows), block(32 * nw);
    cudaStream_t st = gb_stream(stream);
#define GB_AP(J, JH)                                                                                                   \
    gbp_pdl::launch_pdl(flash_pdl(), attn_prep_k<J, JH>, grid, block, 0, st, (const u16*)q_in, (const u16*)k_in, positions, pos3, \
                        q_w, k_w, (u16*)q_out, (u16*)gate_out, (u16*)k_out, heads, kv_heads, rotary_dim, has_gate, eps, tab)
    const int jh = rotary_dim / 64;
    if (head_dim == 64) {
        if (jh == 0) GB_AP(2, 0); else GB_AP(2, 1);
    } else if (head_dim == 128) {
        if (jh == 0) GB_AP(4, 0); else if (jh == 1) GB_AP(4, 1); else GB_AP(4, 2);
    } else if (head_dim == 256) {
        if (jh == 0) GB_AP(8, 0); else if (jh == 1) GB_AP(8, 1); else if (jh == 2) GB_AP(8, 2); else GB_AP(8, 4);
    } else {
        if (jh == 0) GB_AP(16, 0); else if (jh == 1) GB_AP(16, 1); else if (jh == 2) GB_AP(16, 2); else GB_AP(16, 4);
    }
#undef GB_AP
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_attn_prep(const void* q_in, const void* k_in, const int* positions, const float* q_w, const float* k_w,
                           void* q_out, void* gate_out, void* k_out, int rows, int heads, int kv_heads, int head_dim,
                           int rotary_dim, int has_gate, float eps, float theta, void* stream) {
    return attn_prep_launch(q_in, k_in, positions, nullptr, q_w, k_w, q_out, gate_out, k_out, rows, heads, kv_heads, head_dim,
                            rotary_dim, has_gate, eps, theta, stream);
}

// As gb_attn_prep with interleaved M-RoPE: pos3 int32[rows][3] (T, H, W per row) replaces `positions`; rotary_dim must be > 0.
GB_EXPORT int gb_attn_prep_mrope(const void* q_in, const void* k_in, const int* pos3, const float* q_w, const float* k_w,
                                 void* q_out, void* gate_out, void* k_out, int rows, int heads, int kv_heads, int head_dim,
                                 int rotary_dim, int has_gate, float eps, float theta, void* stream) {
    if (!pos3 || rotary_dim == 0) return ERR_INVALID;
    return attn_prep_launch(q_in, k_in, nullptr, pos3, q_w, k_w, q_out, gate_out, k_out, rows, heads, kv_heads, head_dim,
                            rotary_dim, has_gate, eps, theta, stream);
}

// o [rows, heads, dim], z same, w f32[dim] plain gamma; dim in {128, 256}. out may alias o.
GB_EXPORT int gb_gated_rmsnorm(const void* o, const void* z, const float* w, void* out, int rows, int heads, int dim, float eps,
                               void* stream) {
    if (rows < 0 || heads <= 0 || (dim != 128 && dim != 256)) return ERR_INVALID;
    if (!aligned16(o) || !aligned16(z) || !aligned16(out) || !aligned16(w)) return ERR_INVALID;
    if (rows == 0) return 0;
    const int nw = heads < 32 ? heads : 32;
    if (dim == 128)
        gated_rmsnorm_k<4><<<rows, 32 * nw, 0, gb_stream(stream)>>>((const u16*)o, (const u16*)z, w, (u16*)out, heads, eps);
    else
        gated_rmsnorm_k<8><<<rows, 32 * nw, 0, gb_stream(stream)>>>((const u16*)o, (const u16*)z, w, (u16*)out, heads, eps);
    return GB_CHECK_LAUNCH();
}

static int small_linear(const void* x, const void* w, void* y, int rows, int k, int n, bool f32out, void* stream) {
    if (rows < 0 || k <= 0 || n < 0 || k % 8 != 0) return ERR_INVALID;
    if (!aligned16(x) || !aligned16(w)) return ERR_INVALID;
    if (rows == 0 || n == 0) return 0;
    const dim3 grid(n, (rows + SL_RT - 1) / SL_RT);
    if (f32out)
        gbp_pdl::launch_pdl(flash_pdl(), small_linear_k<true>, grid, dim3(SL_NT), 0, gb_stream(stream), (const u16*)x, (const __half*)w,
                            y, rows, k, n);
    else
        gbp_pdl::launch_pdl(flash_pdl(), small_linear_k<false>, grid, dim3(SL_NT), 0, gb_stream(stream), (const u16*)x, (const __half*)w,
                            y, rows, k, n);
    return GB_CHECK_LAUNCH();
}
GB_EXPORT int gb_small_linear_f16w(const void* x, const void* w, void* y, int rows, int k, int n, void* stream) {
    return small_linear(x, w, y, rows, k, n, false, stream);
}
GB_EXPORT int gb_small_linear_f16w_f32out(const void* x, const void* w, void* y, int rows, int k, int n, void* stream) {
    return small_linear(x, w, y, rows, k, n, true, stream);
}

// Same results as gb_small_linear_f16w / _f32out bit for bit, faster for many rows (a prefill chunk): weight-stationary blocks for
// k <= 2,048 (4 rows a step), 4 rows x 8 columns a block above, the per-row kernel for narrow outputs (n < 8).
static int small_linear_rows(const void* x, const void* w, void* y, int rows, int k, int n, bool f32out, void* stream) {
    if (rows < 0 || k <= 0 || n < 0 || k % 8 != 0) return ERR_INVALID;
    if (!aligned16(x) || !aligned16(w)) return ERR_INVALID;
    if (rows == 0 || n == 0) return 0;
    if (n < 8) return small_linear(x, w, y, rows, k, n, f32out, stream);
    cudaStream_t st = gb_stream(stream);
    if (k / 8 <= SL_NT) {
        // about 4 blocks an SM (48 SMs): a slice of rows each, a multiple of the 4-row step
        const int col_groups = (n + SLC_J - 1) / SLC_J;
        int slices = (192 + col_groups - 1) / col_groups;
        const int steps = (rows + SLC_RG - 1) / SLC_RG;
        if (slices > steps) slices = steps;
        const int rows_per_block = ((steps + slices - 1) / slices) * SLC_RG;
        const dim3 grid(col_groups, (rows + rows_per_block - 1) / rows_per_block);
        if (f32out)
            small_linear_cols_k<true><<<grid, SL_NT, 0, st>>>((const u16*)x, (const __half*)w, y, rows, k, n, rows_per_block);
        else
            small_linear_cols_k<false><<<grid, SL_NT, 0, st>>>((const u16*)x, (const __half*)w, y, rows, k, n, rows_per_block);
        return GB_CHECK_LAUNCH();
    }
    const dim3 grid((n + SLR_J - 1) / SLR_J, (rows + SLR_R - 1) / SLR_R);
    if (f32out)
        small_linear_rows_k<true><<<grid, SL_NT, 0, st>>>((const u16*)x, (const __half*)w, y, rows, k, n);
    else
        small_linear_rows_k<false><<<grid, SL_NT, 0, st>>>((const u16*)x, (const __half*)w, y, rows, k, n);
    return GB_CHECK_LAUNCH();
}
GB_EXPORT int gb_small_linear_f16w_rows(const void* x, const void* w, void* y, int rows, int k, int n, void* stream) {
    return small_linear_rows(x, w, y, rows, k, n, false, stream);
}
GB_EXPORT int gb_small_linear_f16w_f32out_rows(const void* x, const void* w, void* y, int rows, int k, int n, void* stream) {
    return small_linear_rows(x, w, y, rows, k, n, true, stream);
}

// Workspace of gb_topk_valid: it depends on the valid columns only (the chunks of a row that are read).
GB_EXPORT size_t gb_topk_valid_workspace_bytes(int rows, int valid, int count) {
    if (rows <= 0 || valid <= 0 || count <= 0 || count > 64) return 0;
    return (size_t)rows * topk_nchunks(valid) * topk_p(count) * sizeof(u64);
}

GB_EXPORT size_t gb_topk_workspace_bytes(int rows, int vocab, int count) { return gb_topk_valid_workspace_bytes(rows, vocab, count); }

// Per row the `count` largest of logits[row * stride + 0 .. valid) by (value DESC, id ASC), exact; ids int32 [rows, count], vals f32
// [rows, count] (bf16 values widened; NaN reported as -inf). 1 <= count <= min(64, valid), valid <= stride. Columns [valid, stride)
// are never read: an id >= valid is never returned, whatever those columns hold. `workspace` must hold
// gb_topk_valid_workspace_bytes(rows, valid, count) bytes and be 16-byte aligned.
GB_EXPORT int gb_topk_valid(const void* logits, int rows, int stride, int valid, int count, int* ids, float* vals, void* workspace,
                            size_t ws_bytes, void* stream) {
    if (rows < 0 || stride <= 0 || valid <= 0 || valid > stride || count < 1 || count > 64 || count > valid) return ERR_INVALID;
    if (rows == 0) return 0;
    if (ws_bytes < gb_topk_valid_workspace_bytes(rows, valid, count) || !workspace || !aligned16(workspace)) return ERR_INVALID;
    u64* ws = (u64*)workspace;
    const int p = topk_p(count);
    if (p == 16) return launch_topk<16>(logits, rows, stride, valid, count, ids, vals, ws, stream);
    if (p == 32) return launch_topk<32>(logits, rows, stride, valid, count, ids, vals, ws, stream);
    return launch_topk<64>(logits, rows, stride, valid, count, ids, vals, ws, stream);
}

// Per row the `count` largest of logits[row, 0..vocab): gb_topk_valid with the whole row valid (the bits are what they always were).
GB_EXPORT int gb_topk(const void* logits, int rows, int vocab, int count, int* ids, float* vals, void* workspace, size_t ws_bytes,
                      void* stream) {
    return gb_topk_valid(logits, rows, vocab, vocab, count, ids, vals, workspace, ws_bytes, stream);
}

GB_EXPORT int gb_copy_rows(const void* src, long long src_stride, void* dst, long long dst_stride, int rows, int width,
                           void* stream) {
    if (rows < 0 || width < 0 || src_stride < 0 || dst_stride < 0) return ERR_INVALID;
    if (rows == 0 || width == 0) return 0;
    if (width % 8 == 0 && src_stride % 8 == 0 && dst_stride % 8 == 0 && aligned16(src) && aligned16(dst))
        copy_rows_vec_k<<<rows, 128, 0, gb_stream(stream)>>>((const u16*)src, src_stride, (u16*)dst, dst_stride, width / 8);
    else
        copy_rows_scalar_k<<<rows, 128, 0, gb_stream(stream)>>>((const u16*)src, src_stride, (u16*)dst, dst_stride, width);
    return GB_CHECK_LAUNCH();
}

// dst[i, :] = src[row_idx[i], :] (rows of `width` elements, contiguous); a negative index writes zeros.
GB_EXPORT int gb_gather_rows(const void* src, const int* row_idx, void* dst, int n, int width, void* stream) {
    if (n < 0 || width < 0) return ERR_INVALID;
    if (n == 0 || width == 0) return 0;
    if (width % 8 == 0 && aligned16(src) && aligned16(dst))
        gather_rows_vec_k<<<n, 128, 0, gb_stream(stream)>>>((const u16*)src, row_idx, (u16*)dst, width);
    else
        gather_rows_scalar_k<<<n, 128, 0, gb_stream(stream)>>>((const u16*)src, row_idx, (u16*)dst, width);
    return GB_CHECK_LAUNCH();
}

// Flash-Next gated norm and shared-expert addition. Original REXL3 code, Victor Cruz.
template <int EPL>
__global__ void gated_rmsnorm_sigmoid_k(const u16* o, const u16* z, const float* w, u16* out, int heads, float eps) {
    pdl_wait();
    constexpr int DIM = EPL * 32;
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int h = wid; h < heads; h += nw) {
        const size_t off = ((size_t)row * heads + h) * DIM + lane * EPL;
        float ov[EPL], zv[EPL];
        if constexpr (EPL == 8) {
            unpack8(*(const uint4*)(o + off), ov);
            unpack8(*(const uint4*)(z + off), zv);
        } else {  // EPL == 4
#pragma unroll
            for (int i = 0; i < EPL; i += 2) {
                const u32 a = *(const u32*)(o + off + i), b = *(const u32*)(z + off + i);
                ov[i] = __uint_as_float(a << 16); ov[i + 1] = __uint_as_float(a & 0xFFFF0000u);
                zv[i] = __uint_as_float(b << 16); zv[i + 1] = __uint_as_float(b & 0xFFFF0000u);
            }
        }
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < EPL; ++i) ss = ss + ov[i] * ov[i];
        ss = warp_sum(ss);
        const float inv = 1.0f / sqrtf(ss / (float)DIM + eps);
        u16 r[EPL];
#pragma unroll
        for (int i = 0; i < EPL; ++i) {
            const float yn = (ov[i] * inv) * w[lane * EPL + i];
            float gate = 1.0f / (1.0f + expf(-zv[i]));
            r[i] = f2bf(yn * gate);
        }
        if constexpr (EPL == 8) {
            uint4 u;
            u.x = r[0] | ((u32)r[1] << 16); u.y = r[2] | ((u32)r[3] << 16);
            u.z = r[4] | ((u32)r[5] << 16); u.w = r[6] | ((u32)r[7] << 16);
            *(uint4*)(out + off) = u;
        } else {
#pragma unroll
            for (int i = 0; i < EPL; i += 2) *(u32*)(out + off + i) = r[i] | ((u32)r[i + 1] << 16);
        }
    }
}


GB_EXPORT int gb_gated_rmsnorm_sigmoid(const void* o, const void* z, const float* w, void* out, int rows, int heads, int dim, float eps,
                                       void* stream) {
    if (rows < 0 || heads <= 0 || (dim != 128 && dim != 256)) return ERR_INVALID;
    if (!aligned16(o) || !aligned16(z) || !aligned16(out) || !aligned16(w)) return ERR_INVALID;
    if (rows == 0) return 0;
    const int nw = heads < 32 ? heads : 32;
    if (dim == 128)
        gbp_pdl::launch_pdl(flash_pdl(), gated_rmsnorm_sigmoid_k<4>, dim3(rows), dim3(32 * nw), 0, gb_stream(stream), (const u16*)o,
                            (const u16*)z, w, (u16*)out, heads, eps);
    else
        gbp_pdl::launch_pdl(flash_pdl(), gated_rmsnorm_sigmoid_k<8>, dim3(rows), dim3(32 * nw), 0, gb_stream(stream), (const u16*)o,
                            (const u16*)z, w, (u16*)out, heads, eps);
    return GB_CHECK_LAUNCH();
}

__global__ void add_gated_shared_kernel(
    const __nv_bfloat16* __restrict__ shared_down,
    const __nv_bfloat16* __restrict__ gate_scalar,
    __nv_bfloat16* __restrict__ moe_out,
    unsigned int H,
    unsigned int num_tokens
) {
    const unsigned int t = blockIdx.y;
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tokens) return;
    float g = 1.0f / (1.0f + expf(-(float)gate_scalar[t]));
    const __nv_bfloat16* sh = shared_down + (size_t)t * H;
    __nv_bfloat16* mo = moe_out + (size_t)t * H;
    for (unsigned int d = tid; d < H; d += gridDim.x * blockDim.x) {
        float v_sh = (float)sh[d];
        float v_mo = (float)mo[d];
        mo[d] = __float2bfloat16(v_mo + v_sh * g);
    }
}

GB_EXPORT int gb_add_gated_shared(
    const void* shared_down,
    const void* gate_scalar,
    void* moe_out,
    unsigned int H,
    unsigned int num_tokens,
    void* stream
) {
    if (num_tokens == 0) return 0;
    dim3 grid((H + 255) / 256, num_tokens);
    add_gated_shared_kernel<<<grid, 256, 0, gb_stream(stream)>>>(
        (const __nv_bfloat16*)shared_down,
        (const __nv_bfloat16*)gate_scalar,
        (__nv_bfloat16*)moe_out,
        H,
        num_tokens
    );
    return GB_CHECK_LAUNCH();
}
