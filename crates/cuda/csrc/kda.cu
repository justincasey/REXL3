// gb-nvcc: --fmad=false
// Kimi delta attention (KDA) kernels for GLM-5.3-Flash: 34 of 45 layers, `heads` x 128 keys x 128 values per layer with a
// PER-KEY-CHANNEL decay (the GDN of gdn.cu has one decay per head), a depthwise causal conv of kernel 4 over the
// 3 * heads * 128 channels [q | k | v], a sigmoid output gate. Design, record layout and numerics: crates/cuda/src/kda.rs.
//
// Ported from TensorFold (MIT, commit 9cd52ab): the delta-rule `update()` (decay first, then read with k, correct toward v,
// then write k * delta) and the one-routine-for-chain-and-replay idea from src/tensorfold/families/glm5_next/cuda/kda.cu.
// Everything else is REXL3's own structure (GDN's window/commit split, a double-buffered state, a record of the window's
// INPUTS so the conv shift and the replay need nothing else). The oracle (the ExLlamaV3 fork's BC graph) is matched
// numerically, never by hash: it is built with --use_fast_math and sums in a different order. See THIRD_PARTY_NOTICES.md.
//
// The file is compiled with --fmad=false. Every multiply and add of `update()`/`readout()` is rounded alone, and the one
// routine is inlined into the window kernel (serial 1-row step, K+1-row verify window, prefill rows) and into the replay
// launch, so row r of any window, a replayed prefix and a serial step run the SAME instruction sequence on the same
// operands: rollback == serial replay, bit for bit. Explicit __fmaf_rn is used only where the oracle fuses (the conv taps
// and the small GEMVs), in a fixed order that does not depend on how many rows share a launch.
//
// State pool (per layer): S f32 [slots][2][heads][128 value row j][128 key i], parity i32 [slots] (index of the CURRENT
// buffer), conv bf16 [slots][3][C] (last 3 raw qkv rows, oldest first). A window reads S[cur] (read-only) and writes the
// state after its LAST row into S[1-cur]; commit flips `cur` (device side) when at least one row is accepted, replaying the
// accepted prefix first when it is shorter than the window.
#include "common.cuh"

namespace {

constexpr int D = 128;               // key dim == value dim
constexpr int KEEP = 3;              // conv history rows (kernel 4 reads x[t-3..t])
constexpr float Q_SCALE = 0.08838834764831845f;  // 1/sqrt(128), applied to q (the oracle applies it to the output)
constexpr float L2_EPS = 1e-6f;

enum XType { X_F16 = 0, X_BF16 = 1, X_F32 = 2 };

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int m = 16; m; m >>= 1) x += __shfl_xor_sync(0xffffffffu, x, m);
    return x;
}

// Cross-lane sums of N per-lane values (N = 2, 4, 8, 16 or 32) with the association tree of the plain xor butterfly (offsets 16, 8, 4, 2, 1:
// after each step the two lanes of a pair hold identical bits) but fewer shuffles: the first log2(N) steps exchange HALVES of the live values
// (N/2, N/4, ... shuffles, N - 1 in all), the rest reduce the single value left. Every addition has the same two operands as in the
// butterfly (addition is commutative), so the bits are those of `for m: v[i] += shfl_xor(v[i], m)`; only WHICH lane ends up holding value i
// differs: lane l holds the full sum of value owned_index<N>(l) (the 32 / N lanes that differ in the low bits hold identical copies).
template <int N>
__host__ __device__ constexpr int log2c() {
    return N == 32 ? 5 : N == 16 ? 4 : N == 8 ? 3 : N == 4 ? 2 : 1;
}

template <int N>
__device__ __forceinline__ int owned_index(int lane) {
    int idx = 0;
#pragma unroll
    for (int i = 0; i < log2c<N>(); ++i) idx += ((lane >> (4 - i)) & 1) * (N >> (i + 1));
    return idx;
}

// A lane that holds value i's sum.
template <int N>
__device__ __forceinline__ int lane_of_index(int i) {
    int l = 0;
#pragma unroll
    for (int b = 0; b < log2c<N>(); ++b) l |= ((i >> (log2c<N>() - 1 - b)) & 1) << (4 - b);
    return l;
}

// `v` is clobbered.
template <int N>
__device__ __forceinline__ float halve_reduce(float (&v)[N], int lane) {
#pragma unroll
    for (int step = 0; step < log2c<N>(); ++step) {
        const int half = N >> (step + 1);
        const int off = 16 >> step;
        const bool hi = (lane & off) != 0;
#pragma unroll
        for (int j = 0; j < half; ++j) {
            const float keep = hi ? v[half + j] : v[j];
            const float send = hi ? v[j] : v[half + j];
            v[j] = keep + __shfl_xor_sync(0xffffffffu, send, off);
        }
    }
    float x = v[0];
#pragma unroll
    for (int off = 16 >> log2c<N>(); off; off >>= 1) x += __shfl_xor_sync(0xffffffffu, x, off);
    return x;
}

__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float sigmoid_ieee(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ void load4_bf16(const __nv_bfloat16* p, float (&x)[4]) {
    const uint2 w = *reinterpret_cast<const uint2*>(p);
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&w.x);
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&w.y);
    x[0] = __low2float(a);
    x[1] = __high2float(a);
    x[2] = __low2float(b);
    x[3] = __high2float(b);
}

// The bf16 raw value of 4 consecutive channels of a qkv row: from f32 (one RN rounding, the oracle's cast) or from the
// bf16 record (replay). Both give the same bits for the same row, so the conv that follows is identical.
template <bool FROM_F32>
__device__ __forceinline__ void load_raw4(const void* base, size_t idx, float (&x)[4]) {
    if (FROM_F32) {
        const float4 t = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(base) + idx);
        x[0] = round_bf16(t.x);
        x[1] = round_bf16(t.y);
        x[2] = round_bf16(t.z);
        x[3] = round_bf16(t.w);
    } else {
        load4_bf16(reinterpret_cast<const __nv_bfloat16*>(base) + idx, x);
    }
}

// The sequence a global row belongs to: the largest s with seq_start[s] <= row (empty sequences are skipped).
__device__ __forceinline__ int find_seq(const int* __restrict__ seq_start, int n_seq, int row) {
    int lo = 0, hi = n_seq - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (seq_start[mid] <= row) lo = mid; else hi = mid - 1;
    }
    return lo;
}

// ---------------------------------------------------------------------------------------------------------------
// proj_in: b, f_a, g_a as one launch of small fp16-weight GEMVs off the layer input x (any of f16/bf16/f32), fp32 out.
// y[r, j] = sum_k x[r, k] * w[j, k]. Block (j, row tile); thread t owns 8-vectors t, t+256, ...; acc = fmaf(x, w, acc)
// in element order; then a warp butterfly and the 8 warp sums in order. Row r's value depends only on x[r], w[j]: the same
// bits for any row tile and any number of rows. b_proj outputs go through beta = bf16(sigmoid(b)) (the oracle's gate op).
// ---------------------------------------------------------------------------------------------------------------
constexpr int PI_NT = 256;
constexpr int PI_RT = 8;

template <int XT>
__device__ __forceinline__ void load_x8(const void* x, size_t idx8, float (&f)[8]) {
    if (XT == X_F32) {
        const float4* p = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(x) + idx8 * 8);
        const float4 a = __ldg(p), b = __ldg(p + 1);
        f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
    } else {
        const uint4 u = __ldg(reinterpret_cast<const uint4*>(x) + idx8);
        if (XT == X_F16) {
            const __half2* h2 = reinterpret_cast<const __half2*>(&u);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 t = __half22float2(h2[i]);
                f[2 * i] = t.x; f[2 * i + 1] = t.y;
            }
        } else {
            const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                f[2 * i] = __low2float(b2[i]);
                f[2 * i + 1] = __high2float(b2[i]);
            }
        }
    }
}

// NJ outputs per block share every x load (NJ = 1 for small row counts, where the weight stream is the cost, NJ = 4 for many rows,
// where re-reading x per output would be): the summation order of an output is the same for every NJ and every row tile.
template <int XT, int NJ>
__global__ void __launch_bounds__(PI_NT) proj_in_kernel(
        const void* __restrict__ x, const __half* __restrict__ w_b, const __half* __restrict__ w_fa,
        const __half* __restrict__ w_ga, int rows, int hidden, int heads, int rank, float* __restrict__ beta,
        float* __restrict__ fa, float* __restrict__ ga) {
    __shared__ float sm[NJ][PI_RT][PI_NT / 32];
    const int n_out = heads + 2 * rank;
    const int jb = blockIdx.x * NJ;
    const int r0 = blockIdx.y * PI_RT;
    const __half* w[NJ];
#pragma unroll
    for (int q = 0; q < NJ; ++q) {
        const int j = min(jb + q, n_out - 1);  // a tail output beyond n_out reads a valid row and is discarded
        w[q] = j < heads ? w_b + (size_t)j * hidden
               : j < heads + rank ? w_fa + (size_t)(j - heads) * hidden
                                   : w_ga + (size_t)(j - heads - rank) * hidden;
    }
    const int nvec = hidden / 8;
    float acc[NJ][PI_RT];
#pragma unroll
    for (int q = 0; q < NJ; ++q)
#pragma unroll
        for (int r = 0; r < PI_RT; ++r) acc[q][r] = 0.0f;
    for (int v = threadIdx.x; v < nvec; v += PI_NT) {
        float wf[NJ][8];
#pragma unroll
        for (int q = 0; q < NJ; ++q) {
            const uint4 wu = __ldg(reinterpret_cast<const uint4*>(w[q]) + v);
            const __half2* h2 = reinterpret_cast<const __half2*>(&wu);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f = __half22float2(h2[i]);
                wf[q][2 * i] = f.x; wf[q][2 * i + 1] = f.y;
            }
        }
#pragma unroll
        for (int r = 0; r < PI_RT; ++r) {
            if (r0 + r < rows) {
                float xf[8];
                load_x8<XT>(x, (size_t)(r0 + r) * nvec + v, xf);
#pragma unroll
                for (int q = 0; q < NJ; ++q)
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[q][r] = __fmaf_rn(xf[i], wf[q][i], acc[q][r]);
            }
        }
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    {
        constexpr int N = NJ * PI_RT;
        float flat[N];
#pragma unroll
        for (int q = 0; q < NJ; ++q)
#pragma unroll
            for (int r = 0; r < PI_RT; ++r) flat[q * PI_RT + r] = acc[q][r];
        const float part = halve_reduce<N>(flat, lane);  // this warp's partial sum of output index owned_index<N>(lane)
        const int idx = owned_index<N>(lane);
        if ((lane & ((1 << (5 - log2c<N>())) - 1)) == 0) sm[idx / PI_RT][idx % PI_RT][wid] = part;
    }
    __syncthreads();
    if (threadIdx.x < NJ * PI_RT) {
        const int q = threadIdx.x / PI_RT, rr = threadIdx.x % PI_RT;
        const int j = jb + q;
        if (j < n_out && r0 + rr < rows) {
            float s = sm[q][rr][0];
            for (int wv = 1; wv < PI_NT / 32; ++wv) s = s + sm[q][rr][wv];
            const size_t r = r0 + rr;
            if (j < heads) beta[r * heads + j] = round_bf16(sigmoid_ieee(s));
            else if (j < heads + rank) fa[r * rank + (j - heads)] = s;
            else ga[r * rank + (j - heads - rank)] = s;
        }
    }
}

// The 8 warp sums of proj_in_lane_kernel in proj_in_kernel's order, then its epilogue.
__global__ void __launch_bounds__(256) proj_in_fin_kernel(const float* __restrict__ part, int rows, int heads, int rank,
                                                          float* __restrict__ beta, float* __restrict__ fa, float* __restrict__ ga) {
    const int n_out = heads + 2 * rank;
    const size_t n = (size_t)rows * n_out;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float s = part[idx];
#pragma unroll
    for (int wv = 1; wv < PI_NT / 32; ++wv) s = s + part[(size_t)wv * n + idx];
    const size_t r = idx / n_out;
    const int j = (int)(idx % n_out);
    if (j < heads) beta[r * heads + j] = round_bf16(sigmoid_ieee(s));
    else if (j < heads + rank) fa[r * rank + (j - heads)] = s;
    else ga[r * rank + (j - heads - rank)] = s;
}

// proj_in for many rows at hidden 4096, the same bits as proj_in_kernel with proj_in_kernel's own lane structure. proj_in_kernel's output
// (r, j) is sum_{w = 0..7, in order} B_w, where B_w is the xor butterfly (offsets 16, 8, 4, 2, 1) over the lanes l of warp w of the leaf
// L_t, t = 32 w + l: the fmaf chain from +0 over x[r][k] w[j][k] for k = 8 t .. 8 t + 7, then 2048 + 8 t .. + 7. Here lane l of a warp
// of block z = w is that thread t for PL_R rows x PL_J outputs at once, the warp's halve_reduce (the butterfly's association) gives B_w
// into part[w][r][j], and proj_in_fin_kernel sums w = 0..7 in order and applies proj_in_kernel's epilogue. A block stages the x rows and
// weight rows of warp w's leaves (k in [256 w, 256 w + 256) and [2048 + 256 w, ..)) for PL_RM rows and PL_JN outputs in shared memory
// once (cp.async), then a lane reads its two 8-vectors per row and output as 16-byte vectors, each feeding PL_J (PL_R) chains: one shared
// 16-byte read per 64 fmaf (proj_in_kernel re-reads x for every 4 outputs and w for every 8 rows from L2: L2-bound; the 64 x 32 shared
// tiles this replaced read two 16-byte vectors per 16 fmaf and kept a 5-deep tree stack: as fast alone, but beside the chunked recurrence
// on the pipelined KDA prefill's side stream they shared its shared-memory bandwidth, MEASURED: a 4,096-row chunk 2,425 -> 2,400 ms).
// A block is PL_WR x PL_WJ warps over PL_WR * PL_R rows and PL_WJ * PL_J outputs, two blocks an SM.
constexpr int PL_R = 8, PL_J = 8, PL_WR = 2, PL_WJ = 2;
constexpr int PL_RM = PL_R * PL_WR;  // rows a block
constexpr int PL_JN = PL_J * PL_WJ;  // outputs a block
constexpr int PL_NT = PL_WR * PL_WJ * 32;
constexpr int PL_K = 512;  // k values of one warp's leaves: two runs of 256

template <int XT>
__host__ __device__ constexpr int pl_xbytes() {
    return XT == X_F32 ? 4 : 2;
}
template <int XT>
__host__ __device__ constexpr int pl_smem() {
    return PL_RM * PL_K * pl_xbytes<XT>() + PL_JN * PL_K * 2;
}

__device__ __forceinline__ void pl_cp16(void* smem, const void* gmem, bool valid) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(valid ? 16 : 0));
}

template <int XT>
__device__ __forceinline__ void pl_x8(const unsigned char* p, float (&f)[8]) {
    if (XT == X_F32) {
        const float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 16);
        f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
    } else {
        const uint4 u = *reinterpret_cast<const uint4*>(p);
        if (XT == X_F16) {
            const __half2* h2 = reinterpret_cast<const __half2*>(&u);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 t = __half22float2(h2[i]);
                f[2 * i] = t.x; f[2 * i + 1] = t.y;
            }
        } else {
            const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                f[2 * i] = __low2float(b2[i]);
                f[2 * i + 1] = __high2float(b2[i]);
            }
        }
    }
}

template <int XT>
__global__ void __launch_bounds__(PL_NT, 2) proj_in_lane_kernel(
        const void* __restrict__ x, const __half* __restrict__ w_b, const __half* __restrict__ w_fa,
        const __half* __restrict__ w_ga, int rows, int heads, int rank, float* __restrict__ part) {
    constexpr int HID = 4096, XB = pl_xbytes<XT>();
    constexpr int XC = PL_K * XB / 16;  // 16-byte chunks of a staged x row
    constexpr int WC = PL_K * 2 / 16;   // of a staged weight row
    extern __shared__ __align__(16) unsigned char pl_sm[];
    unsigned char* xs = pl_sm;                       // [PL_RM][PL_K] of x's type
    unsigned char* wsm = pl_sm + PL_RM * PL_K * XB;  // [PL_JN][PL_K] fp16
    const int n_out = heads + 2 * rank;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wv = blockIdx.z;
    const int rb = blockIdx.y * PL_RM, jb = blockIdx.x * PL_JN;
    // stage: chunk c of row r covers k = run * 2048 + 256 wv + (c % (XC / 2)) * (16 / XB) .. (run = c / (XC / 2))
    for (int u = threadIdx.x; u < PL_RM * XC; u += PL_NT) {
        const int r = u / XC, c = u % XC, run = c / (XC / 2), k = run * 2048 + 256 * wv + (c % (XC / 2)) * (16 / XB);
        const bool ok = rb + r < rows;
        const unsigned char* src = static_cast<const unsigned char*>(x) + ((size_t)(ok ? rb + r : 0) * HID + k) * XB;
        pl_cp16(xs + (size_t)r * PL_K * XB + (size_t)c * 16, src, ok);
    }
    for (int u = threadIdx.x; u < PL_JN * WC; u += PL_NT) {
        const int q = u / WC, c = u % WC, run = c / (WC / 2), k = run * 2048 + 256 * wv + (c % (WC / 2)) * 8;
        const int j = min(jb + q, n_out - 1);  // a tail output beyond n_out reads a valid row and is discarded
        const __half* wr = j < heads ? w_b + (size_t)j * HID
                           : j < heads + rank ? w_fa + (size_t)(j - heads) * HID
                                              : w_ga + (size_t)(j - heads - rank) * HID;
        pl_cp16(wsm + (size_t)q * PL_K * 2 + (size_t)c * 16, wr + k, true);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::);
    __syncthreads();
    const int rg = warp % PL_WR, og = warp / PL_WR;
    float acc[PL_R][PL_J];
#pragma unroll
    for (int r = 0; r < PL_R; ++r)
#pragma unroll
        for (int q = 0; q < PL_J; ++q) acc[r][q] = 0.0f;
#pragma unroll
    for (int h = 0; h < 2; ++h) {  // the leaf's two 8-vectors, t then t + 256, in proj_in_kernel's order
        const int e0 = h * 256 + 8 * lane;
        float wf[PL_J][8];
#pragma unroll
        for (int q = 0; q < PL_J; ++q) pl_x8<X_F16>(wsm + ((size_t)(og * PL_J + q) * PL_K + e0) * 2, wf[q]);
#pragma unroll
        for (int r = 0; r < PL_R; ++r) {
            float xf[8];
            pl_x8<XT>(xs + ((size_t)(rg * PL_R + r) * PL_K + e0) * XB, xf);
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int q = 0; q < PL_J; ++q) acc[r][q] = __fmaf_rn(xf[i], wf[q][i], acc[r][q]);
        }
    }
    // the warp's butterfly over its 32 leaves, 32 (row, output) sums at a time: lane l ends with the sum of value owned_index<32>(l)
    const int r0 = rb + rg * PL_R, j0 = jb + og * PL_J;
#pragma unroll
    for (int hh = 0; hh < PL_R * PL_J / 32; ++hh) {
        float flat[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) flat[i] = acc[(hh * 32 + i) / PL_J][(hh * 32 + i) % PL_J];
        const float s = halve_reduce<32>(flat, lane);
        const int idx = hh * 32 + owned_index<32>(lane);
        const int row = r0 + idx / PL_J, j = j0 + idx % PL_J;
        if (row < rows && j < n_out) part[((size_t)wv * rows + row) * n_out + j] = s;
    }
}

template <int XT>
int launch_proj_in_lane(const void* x, const __half* b, const __half* f, const __half* g, int rows, int heads, int rank, float* part,
                        cudaStream_t st) {
    constexpr int sm = pl_smem<XT>();
    static bool attr = false;  // idempotent; a race between host threads sets the same value twice
    if (!attr) {
        const cudaError_t e = cudaFuncSetAttribute(proj_in_lane_kernel<XT>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
        if (e != cudaSuccess) return (int)e;
        attr = true;
    }
    const int n_out = heads + 2 * rank;
    const dim3 grid((n_out + PL_JN - 1) / PL_JN, (rows + PL_RM - 1) / PL_RM, PI_NT / 32);
    proj_in_lane_kernel<XT><<<grid, PL_NT, sm, st>>>(x, b, f, g, rows, heads, rank, part);
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------
// proj_out: f_b and g_b, the low-rank second stages, with an fp32 input (the first stage's fp32 output, not rounded to
// half: the BC graph's choice). One THREAD per output channel, PO_RT rows of the input staged in shared memory and read as
// broadcasts, the 128-term dot product a plain fma chain in index order: no shuffles, and the order of an output does not depend
// on the rows of the launch. Outputs of the forget path go straight through the gate: g = exp(lower * sigmoid(exp(A_log[h]) *
// (f + dt_bias[c]))), written to the record; the others are the output gate z (fp32, read by the norm). (Measured and rejected,
// same bits, slower: four outputs per thread for many rows.)
// ---------------------------------------------------------------------------------------------------------------
constexpr int PO_RT = 16;

// Few rows (decode windows): a block takes PS_C channels of one head of one path, its weight rows (PS_C x 256 bytes) staged in shared
// memory by coalesced 16-byte loads (a thread reading its own 256-byte row touches 32 lines a warp instruction: 134 GB/s cold on GB10,
// 31 us a 64-head layer at 1 row), rows padded by 16 bytes so a quarter warp's row reads hit distinct banks; lane = channel, warp w
// the rows w, w + 4, .. of the PO_RT-row tile.
constexpr int PS_C = 32;
constexpr int PS_NT = 128;
constexpr int PS_LD = D + 8;

__global__ void __launch_bounds__(PS_NT) proj_out_small_kernel(
        const float* __restrict__ fa, const float* __restrict__ ga, const __half* __restrict__ w_fb,
        const __half* __restrict__ w_gb, const float* __restrict__ a_log, const float* __restrict__ dt_bias, float lower,
        int rows, int heads, float* __restrict__ g, float* __restrict__ z) {
    __shared__ __align__(16) __half wsm[PS_C][PS_LD];
    __shared__ __align__(16) float xs[PO_RT][D];
    constexpr int CG = D / PS_C;
    const int hp = blockIdx.x / CG, cg = blockIdx.x % CG;
    const bool is_g = hp < heads;
    const int h = is_g ? hp : hp - heads;
    const int c0 = h * D + cg * PS_C;  // the block's first output channel within the path
    const int r0 = blockIdx.y * PO_RT;
    const int nr = min(PO_RT, rows - r0);
    const uint4* wsrc = reinterpret_cast<const uint4*>((is_g ? w_fb : w_gb) + (size_t)c0 * D);
#pragma unroll
    for (int i = threadIdx.x; i < PS_C * D / 8; i += PS_NT) {
        const int c = i / (D / 8), kq = i % (D / 8);
        *reinterpret_cast<uint4*>(&wsm[c][8 * kq]) = __ldg(wsrc + i);
    }
    const float* x = is_g ? fa : ga;
    for (int i = threadIdx.x; i < nr * (D / 4); i += PS_NT) {
        const int r = i / (D / 4), q = i % (D / 4);
        reinterpret_cast<float4*>(xs[r])[q] = __ldg(reinterpret_cast<const float4*>(x + (size_t)(r0 + r) * D) + q);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    constexpr int RW = PO_RT / (PS_NT / 32);  // rows a warp may hold
    float acc[RW];
#pragma unroll
    for (int j = 0; j < RW; ++j) acc[j] = 0.0f;
#pragma unroll
    for (int kb = 0; kb < D / 8; ++kb) {
        const uint4 wu = *reinterpret_cast<const uint4*>(&wsm[lane][8 * kb]);
        float wf[8];
        const __half2* h2 = reinterpret_cast<const __half2*>(&wu);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(h2[i]);
            wf[2 * i] = f.x; wf[2 * i + 1] = f.y;
        }
#pragma unroll
        for (int j = 0; j < RW; ++j) {
            const int r = wid + 4 * j;
            if (r < nr) {
                const float4 x0 = *reinterpret_cast<const float4*>(&xs[r][kb * 8]);
                const float4 x1 = *reinterpret_cast<const float4*>(&xs[r][kb * 8 + 4]);
                float a = acc[j];
                a = __fmaf_rn(x0.x, wf[0], a);
                a = __fmaf_rn(x0.y, wf[1], a);
                a = __fmaf_rn(x0.z, wf[2], a);
                a = __fmaf_rn(x0.w, wf[3], a);
                a = __fmaf_rn(x1.x, wf[4], a);
                a = __fmaf_rn(x1.y, wf[5], a);
                a = __fmaf_rn(x1.z, wf[6], a);
                a = __fmaf_rn(x1.w, wf[7], a);
                acc[j] = a;
            }
        }
    }
    const int c = c0 + lane;
    const int n_half = heads * D;
    float decay = 0.0f, dtb = 0.0f;
    if (is_g) {
        decay = expf(a_log[h]);
        dtb = dt_bias[c];
    }
#pragma unroll
    for (int j = 0; j < RW; ++j) {
        const int r = wid + 4 * j;
        if (r < nr) {
            const size_t o = (size_t)(r0 + r) * n_half + c;
            if (is_g) g[o] = expf(lower * sigmoid_ieee(decay * (acc[j] + dtb)));
            else z[o] = acc[j];
        }
    }
}

// proj_out for many rows, the same bits as proj_out_small_kernel (each output the 128-term fmaf chain in k order from +0, then the same
// epilogue): a block takes 64 rows of one head of one path, x and w staged transposed in shared memory 64 k at a time, a thread 8 rows x
// 4 channels (x read as warp broadcasts, w as 8-byte vectors).
constexpr int PO2_RM = 64;
constexpr int PO2_KS = 64;

// The tile of rows r0 .. r0 + nr - 1 (nr <= 64) of x [rows, 128] times the head's 128 rows of w [., 128]: thread (tx, ty) = (lane, warp)
// ends with acc[r][c] = the chain of row 8 ty + r, channel 4 tx + c.
__device__ __forceinline__ void po_tile(const float* __restrict__ x, const __half* __restrict__ wh, int r0, int nr, float (&xs)[PO2_KS][PO2_RM],
                                        __half (&wsm)[PO2_KS][D], float (&acc)[8][4]) {
    const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
#pragma unroll
    for (int r = 0; r < 8; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = 0.0f;
#pragma unroll
    for (int k0 = 0; k0 < D; k0 += PO2_KS) {
        __syncthreads();
#pragma unroll
        for (int i = threadIdx.x; i < PO2_RM * PO2_KS / 4; i += 256) {
            const int r = i % PO2_RM, kq = i / PO2_RM;
            const float4 v = r < nr ? __ldg(reinterpret_cast<const float4*>(x + (size_t)(r0 + r) * D + k0) + kq) : make_float4(0.f, 0.f, 0.f, 0.f);
            xs[4 * kq][r] = v.x;
            xs[4 * kq + 1][r] = v.y;
            xs[4 * kq + 2][r] = v.z;
            xs[4 * kq + 3][r] = v.w;
        }
#pragma unroll
        for (int i = threadIdx.x; i < D * PO2_KS / 8; i += 256) {
            const int c = i % D, kq = i / D;
            const uint4 u = __ldg(reinterpret_cast<const uint4*>(wh + (size_t)c * D + k0) + kq);
            const __half* hv = reinterpret_cast<const __half*>(&u);
#pragma unroll
            for (int e = 0; e < 8; ++e) wsm[8 * kq + e][c] = hv[e];
        }
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < PO2_KS; ++k) {
            const float4 xa = *reinterpret_cast<const float4*>(&xs[k][8 * ty]);
            const float4 xb = *reinterpret_cast<const float4*>(&xs[k][8 * ty + 4]);
            const uint2 wu = *reinterpret_cast<const uint2*>(&wsm[k][4 * tx]);
            const float2 w01 = __half22float2(*reinterpret_cast<const __half2*>(&wu.x));
            const float2 w23 = __half22float2(*reinterpret_cast<const __half2*>(&wu.y));
            const float xr[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w}, wc[4] = {w01.x, w01.y, w23.x, w23.y};
#pragma unroll
            for (int r = 0; r < 8; ++r)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[r][c] = __fmaf_rn(xr[r], wc[c], acc[r][c]);
        }
    }
}

// gridDim.x = 2 heads (g then z blocks) or heads (z == nullptr: the g path alone).
__global__ void __launch_bounds__(256) proj_out_tile_kernel(
        const float* __restrict__ fa, const float* __restrict__ ga, const __half* __restrict__ w_fb,
        const __half* __restrict__ w_gb, const float* __restrict__ a_log, const float* __restrict__ dt_bias, float lower,
        int rows, int heads, float* __restrict__ g, float* __restrict__ z) {
    __shared__ __align__(16) float xs[PO2_KS][PO2_RM];
    __shared__ __align__(16) __half wsm[PO2_KS][D];
    const bool is_g = blockIdx.x < heads;
    const int h = is_g ? blockIdx.x : blockIdx.x - heads;
    const int r0 = blockIdx.y * PO2_RM;
    const int nr = min(PO2_RM, rows - r0);
    const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;  // channels 4 tx .. 4 tx + 3, rows 8 ty .. 8 ty + 7
    float acc[8][4];
    po_tile(is_g ? fa : ga, (is_g ? w_fb : w_gb) + (size_t)h * D * D, r0, nr, xs, wsm, acc);
    const int n_half = heads * D;
    const int c0 = h * D + 4 * tx;
    float decay = 0.0f, dtb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (is_g) {
        decay = expf(a_log[h]);
#pragma unroll
        for (int c = 0; c < 4; ++c) dtb[c] = dt_bias[c0 + c];
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int row = 8 * ty + r;
        if (row >= nr) continue;
        float o[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) o[c] = is_g ? expf(lower * sigmoid_ieee(decay * (acc[r][c] + dtb[c]))) : acc[r][c];
        *reinterpret_cast<float4*>((is_g ? g : z) + (size_t)(r0 + row) * n_half + c0) = make_float4(o[0], o[1], o[2], o[3]);
    }
}

// The output gate z of a 64-row tile of one head (proj_out's g_b path: the same chains) and, in its epilogue, norm_kernel's gated RMSNorm
// of the head's core output y with it: lane tx of warp ty holds channels 4 tx .. 4 tx + 3 of its rows, norm_kernel's lane layout, so the
// sum of squares is norm_kernel's (lane-local in element order, the xor butterfly) and every output its expression. z never leaves the
// block (the separate path writes it [rows, heads * 128] f32 and the norm reads it back).
__global__ void __launch_bounds__(256) z_norm_tile_kernel(const float* __restrict__ ga, const __half* __restrict__ w_gb,
                                                          const __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ w_norm,
                                                          float eps, int rows, int heads, __half* __restrict__ out) {
    __shared__ __align__(16) float xs[PO2_KS][PO2_RM];
    __shared__ __align__(16) __half wsm[PO2_KS][D];
    const int h = blockIdx.x;
    const int r0 = blockIdx.y * PO2_RM;
    const int nr = min(PO2_RM, rows - r0);
    const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
    float acc[8][4];
    po_tile(ga, w_gb + (size_t)h * D * D, r0, nr, xs, wsm, acc);
    float wt[4];
    load4_bf16(w_norm + tx * 4, wt);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int row = 8 * ty + r;  // the same for the whole warp
        if (row >= nr) continue;
        const size_t off = ((size_t)(r0 + row) * heads + h) * D + tx * 4;
        float x[4];
        load4_bf16(y + off, x);
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) ss = ss + x[i] * x[i];
        ss = warp_sum(ss);
        const float rmf = 1.0f / sqrtf(ss / (float)D + eps);
        __half2 hh[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            float o[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int c = 2 * i + e;
                o[e] = ((x[c] * wt[c]) * rmf) * sigmoid_ieee(acc[r][c]);
            }
            hh[i] = __floats2half2_rn(o[0], o[1]);
        }
        uint2 u;
        u.x = *reinterpret_cast<const unsigned*>(&hh[0]);
        u.y = *reinterpret_cast<const unsigned*>(&hh[1]);
        *reinterpret_cast<uint2*>(out + off) = u;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The full-rank KDA gate of Ling-3.1-flash-1019 (no_kda_lora): f_proj / g_proj are full EXL3 groups producing fp16
// [rows, heads * 128] (no low-rank f_a/f_b/g_a/g_b), b_proj is plain bf16 [heads, hidden]. One launch computes beta,
// the per-channel decay g and the output gate z exactly as proj_in/proj_out would: beta = f32(bf16(sigmoid(x @ b^T))),
// g = exp(lower * sigmoid(exp(A_log[h]) * (f + dt_bias[c]))), z = g_out (fp32).
// ---------------------------------------------------------------------------------------------------------------
constexpr int BG_NT = 256;
constexpr int BG_RT = 8;

template <int XT>
__global__ void __launch_bounds__(BG_NT) beta_from_b_kernel(const void* x, const __nv_bfloat16* w_b, int rows,
                                                            int hidden, int heads, float* beta) {
    __shared__ float sm[BG_RT][BG_NT / 32];
    const int h = blockIdx.x;
    const int r0 = blockIdx.y * BG_RT;
    const int nvec = hidden / 8;
    const __nv_bfloat16* wrow = w_b + (size_t)h * hidden;
    float acc[BG_RT];
#pragma unroll
    for (int r = 0; r < BG_RT; ++r) acc[r] = 0.0f;
    for (int v = threadIdx.x; v < nvec; v += BG_NT) {
        const uint4 wu = __ldg(reinterpret_cast<const uint4*>(wrow) + v);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&wu);
        float wf[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 t = __bfloat1622float2(b2[i]);
            wf[2 * i] = t.x;
            wf[2 * i + 1] = t.y;
        }
#pragma unroll
        for (int r = 0; r < BG_RT; ++r) {
            if (r0 + r < rows) {
                float xf[8];
                load_x8<XT>(x, (size_t)(r0 + r) * nvec + v, xf);
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[r] = __fmaf_rn(xf[i], wf[i], acc[r]);
            }
        }
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    {
        constexpr int N = BG_RT;
        float flat[N];
#pragma unroll
        for (int r = 0; r < N; ++r) flat[r] = acc[r];
        const float part = halve_reduce<N>(flat, lane);
        const int idx = owned_index<N>(lane);
        if ((lane & ((1 << (5 - log2c<N>())) - 1)) == 0) sm[idx % BG_RT][wid] = part;
    }
    __syncthreads();
    if (threadIdx.x < BG_RT) {
        const int rr = threadIdx.x;
        if (r0 + rr < rows) {
            float s = sm[rr][0];
            for (int wv = 1; wv < BG_NT / 32; ++wv) s = s + sm[rr][wv];
            beta[(size_t)(r0 + rr) * heads + h] = round_bf16(sigmoid_ieee(s));
        }
    }
}

__global__ void __launch_bounds__(256) full_fg_gate_kernel(const half* f, const half* g, const float* a_log,
                                                           const float* dt_bias, float lower, int rows, int heads,
                                                           float* g_decay, float* z) {
    const int C = heads * 128;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (idx >= C) return;
    const int h = idx >> 7;
    const size_t off = (size_t)r * C + idx;
    const float fv = __half2float(f[off]);
    const float zv = __half2float(g[off]);
    const float decay = expf(a_log[h]);
    g_decay[off] = expf(lower * sigmoid_ieee(decay * (fv + dt_bias[idx])));
    z[off] = zv;
}

// ---------------------------------------------------------------------------------------------------------------
// prep: depthwise causal conv over [3 stored rows; window rows] + silu (one rounding to bf16), l2norm of q and k (q also
// scaled by 1/sqrt(128)), q/k as f32 and v as bf16 into scratch. One warp per (tile of `tile_rows` consecutive rows, 128-channel
// unit), 4 channels per lane; within a sequence the 4-row conv window slides in registers, so each raw row is loaded once per tile
// (tile_rows = 1 for decode batches of single rows). Bits do not depend on the tile size: a row computes the same expressions from
// the same 4 raw values however they were fetched.
// FROM_F32: the input is the projection's f32 output (else bf16 rows). `raw_out` (nullable): the record, which receives the
// row's bf16 rounding. Replay passes the record itself as input and no `raw_out`. A row only reads its own sequence's rows
// and the committed conv state.
// `keep` limits the rows processed per sequence (replay); rows past it are skipped, and with skip_full so is a sequence kept whole.
// ---------------------------------------------------------------------------------------------------------------
template <bool FROM_F32>
__global__ void __launch_bounds__(128) prep_kernel(
        const void* __restrict__ qkv, const __nv_bfloat16* __restrict__ conv_w, const __nv_bfloat16* __restrict__ conv_pool,
        const int* __restrict__ seq_start, const int* __restrict__ slot, const int* __restrict__ keep, int n_seq, int rows,
        int tile_rows, int heads, bool emit_q, bool skip_full, __nv_bfloat16* __restrict__ raw_out, float* __restrict__ q,
        float* __restrict__ k, __nv_bfloat16* __restrict__ v) {
    const int unit = blockIdx.y * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (!emit_q && !raw_out && unit < heads) return;  // replay: neither q nor the record is needed
    const int row0 = blockIdx.x * tile_rows;
    const int row_end = min(row0 + tile_rows, rows);
    const size_t conv_dim = (size_t)3 * heads * D;
    const int ch0 = unit * D + lane * 4;

    // 4 channels x 4 taps of the conv weight (tap j multiplies x[t-3+j])
    const uint4 w0 = __ldg(reinterpret_cast<const uint4*>(conv_w + (size_t)ch0 * 4));
    const uint4 w1 = __ldg(reinterpret_cast<const uint4*>(conv_w + (size_t)ch0 * 4) + 1);
    float wt[16];
    {
        const __nv_bfloat162* a = reinterpret_cast<const __nv_bfloat162*>(&w0);
        const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&w1);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            wt[2 * i] = __low2float(a[i]);
            wt[2 * i + 1] = __high2float(a[i]);
            wt[8 + 2 * i] = __low2float(b[i]);
            wt[8 + 2 * i + 1] = __high2float(b[i]);
        }
    }
    int seq = 0, start = 0, send = 0, n_lim = 0;
    size_t pool = 0;
    bool have = false;
    float xw[3][4];  // raw values of rows t-3, t-2, t-1 for this lane's 4 channels
    for (int row = row0; row < row_end; ++row) {
        if (row >= send) {  // first row of the tile, or the tile crossed into the next sequence
            seq = find_seq(seq_start, n_seq, row);
            start = seq_start[seq];
            send = seq_start[seq + 1];
            pool = (size_t)slot[seq] * KEEP;
            n_lim = send - start;
            if (keep) {  // replay: only the accepted prefix, and nothing at all for a sequence the window already finished
                const int n = min(max(keep[seq], 0), n_lim);
                n_lim = (skip_full && n >= n_lim) ? 0 : n;
            }
            have = false;
        }
        const int lt = row - start;
        if (lt >= n_lim) continue;  // (rows are consumed in order, so a skipped row never precedes a processed one of its sequence)
        if (!have) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                const int idx = lt - 3 + j;
                if (idx >= 0) load_raw4<FROM_F32>(qkv, (size_t)(start + idx) * conv_dim + ch0, xw[j]);
                else load4_bf16(conv_pool + (pool + 3 + idx) * conv_dim + ch0, xw[j]);
            }
            have = true;
        }
        float x0[4];
        load_raw4<FROM_F32>(qkv, (size_t)row * conv_dim + ch0, x0);
        if (raw_out) {  // this row's own raw value goes to the record
            const __nv_bfloat162 a = __floats2bfloat162_rn(x0[0], x0[1]), b = __floats2bfloat162_rn(x0[2], x0[3]);
            uint2 u;
            u.x = *reinterpret_cast<const unsigned*>(&a);
            u.y = *reinterpret_cast<const unsigned*>(&b);
            *reinterpret_cast<uint2*>(raw_out + (size_t)row * conv_dim + ch0) = u;
        }
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[c] = __fmaf_rn(wt[4 * c + j], j < 3 ? xw[j][c] : x0[c], acc[c]);
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) {  // slide the window
            xw[0][c] = xw[1][c];
            xw[1][c] = xw[2][c];
            xw[2][c] = x0[c];
        }
        float c[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] = round_bf16(acc[i] / (1.0f + expf(-acc[i])));

        if (unit < 2 * heads) {
            const bool is_q = unit < heads;
            if (!is_q || emit_q) {
                float ss = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) ss = ss + c[i] * c[i];
                ss = warp_sum(ss);
                const float inv = 1.0f / sqrtf(ss + L2_EPS);
                const float t0 = c[0] * inv, t1 = c[1] * inv, t2 = c[2] * inv, t3 = c[3] * inv;
                const float4 out = is_q ? make_float4(t0 * Q_SCALE, t1 * Q_SCALE, t2 * Q_SCALE, t3 * Q_SCALE) : make_float4(t0, t1, t2, t3);
                float* dst = (is_q ? q : k) + ((size_t)row * heads + (is_q ? unit : unit - heads)) * D + lane * 4;
                *reinterpret_cast<float4*>(dst) = out;
            }
        } else {
            const int hv = unit - 2 * heads;
            const __nv_bfloat162 a = __floats2bfloat162_rn(c[0], c[1]), b = __floats2bfloat162_rn(c[2], c[3]);
            uint2 u;
            u.x = *reinterpret_cast<const unsigned*>(&a);
            u.y = *reinterpret_cast<const unsigned*>(&b);
            *reinterpret_cast<uint2*>(v + ((size_t)row * heads + hv) * D + lane * 4) = u;
        }
    }
}

// The R value rows of a warp (R = 4 or 8) reduce over the lanes with halve_reduce: lane l holds the full sum of row row_of_lane(l).
template <int R>
__device__ __forceinline__ int row_of_lane(int lane) {
    static_assert(R == 4 || R == 8, "R is 4 or 8");
    return owned_index<R>(lane);
}

// A lane that holds row r's sum.
template <int R>
__device__ __forceinline__ int lane_of_row(int r) {
    return lane_of_index<R>(r);
}

template <int R>
__device__ __forceinline__ float reduce_rows(const float (&v)[R], int lane) {
    float a[R];
#pragma unroll
    for (int i = 0; i < R; ++i) a[i] = v[i];
    return halve_reduce<R>(a, lane);
}

// ---------------------------------------------------------------------------------------------------------------
// The ONE routine that advances the state. A warp holds R value rows, lane l the keys 4l..4l+3 of each. Every op is rounded
// alone (--fmad=false); sums are lane-local in element order, then the butterfly tree above.
// Order per row, per key channel i: s = s * g_i; mem += s * k_i; ... delta = (v - mem) * beta; s = s + k_i * delta.
// `v_mine` is the value element of row row_of_lane(lane): the lane that holds a row's mem computes its delta and broadcasts it.
// ---------------------------------------------------------------------------------------------------------------
template <int R>
__device__ __forceinline__ void update(float (&s)[R][4], const float (&k)[4], const float (&g)[4], float v_mine, float beta,
                                       int lane) {
    float mem[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        mem[r] = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[r][i] = s[r][i] * g[i];
            mem[r] = mem[r] + s[r][i] * k[i];
        }
    }
    const float delta_mine = (v_mine - reduce_rows<R>(mem, lane)) * beta;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float delta = __shfl_sync(0xffffffffu, delta_mine, lane_of_row<R>(r));
#pragma unroll
        for (int i = 0; i < 4; ++i) s[r][i] = s[r][i] + k[i] * delta;
    }
}

// y of row row_of_lane(lane) = sum_i s[r][i] * q[i] on the updated state.
template <int R>
__device__ __forceinline__ float readout(const float (&s)[R][4], const float (&q)[4], int lane) {
    float out[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        out[r] = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) out[r] = out[r] + s[r][i] * q[i];
    }
    return reduce_rows<R>(out, lane);
}

// One row's inputs for a warp: this lane's 4 keys of k, q and the per-channel decay g, its value element, beta.
template <int R>
struct RowIn {
    float q[4], k[4], g[4], v, beta;

    __device__ __forceinline__ void load(const float* qp, const float* kp, const __nv_bfloat16* vp, const float* gp,
                                         const float* bp, int row, int head, int heads, int value0, int lane) {
        const size_t key = ((size_t)row * heads + head) * D + lane * 4;
        const float4 kk = *reinterpret_cast<const float4*>(kp + key);
        const float4 gg = *reinterpret_cast<const float4*>(gp + key);
        k[0] = kk.x; k[1] = kk.y; k[2] = kk.z; k[3] = kk.w;
        g[0] = gg.x; g[1] = gg.y; g[2] = gg.z; g[3] = gg.w;
        if (qp) {
            const float4 qq = *reinterpret_cast<const float4*>(qp + key);
            q[0] = qq.x; q[1] = qq.y; q[2] = qq.z; q[3] = qq.w;
        }
        v = __bfloat162float(vp[((size_t)row * heads + head) * D + value0 + row_of_lane<R>(lane)]);
        beta = bp[(size_t)row * heads + head];
    }
};

constexpr int W_STEP = 4;  // warps per block

template <int R>
__device__ __forceinline__ void load_state(float (&s)[R][4], const float* base, int value0, int lane) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float4 t = *reinterpret_cast<const float4*>(base + (size_t)(value0 + r) * D + lane * 4);
        s[r][0] = t.x; s[r][1] = t.y; s[r][2] = t.z; s[r][3] = t.w;
    }
}

template <int R>
__device__ __forceinline__ void store_state(float* base, const float (&s)[R][4], int value0, int lane) {
#pragma unroll
    for (int r = 0; r < R; ++r)
        *reinterpret_cast<float4*>(base + (size_t)(value0 + r) * D + lane * 4) = make_float4(s[r][0], s[r][1], s[r][2], s[r][3]);
}

// window / replay. Sequence `seq` of the batch: state S[slot][cur] is READ-ONLY; rows begin .. begin + n are stepped from a
// register copy (n = len, or min(keep, len) when `keep` is given); the state after the last of them goes to S[slot][1-cur].
// EMIT: also read-out q . S per row into y (bf16, RN or RZ). !EMIT is the replay: no q, no y, state only.
// SKIP_FULL (commit replay only): a sequence whose accepted prefix is its whole window already has its final state in
// S[1-cur] from the window, so it returns without touching it.
template <int R, bool EMIT, bool RZ>
__global__ void __launch_bounds__(32 * W_STEP) state_kernel(
        const float* __restrict__ q, const float* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const float* __restrict__ g, const float* __restrict__ beta, float* S, const int* __restrict__ parity,
        const int* __restrict__ seq_start, const int* __restrict__ slot, const int* __restrict__ keep, int heads,
        bool skip_full, bool write_final, __nv_bfloat16* __restrict__ y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head = blockIdx.y, seq = blockIdx.z;
    const int value0 = (blockIdx.x * W_STEP + warp) * R;
    const int begin = seq_start[seq], len = seq_start[seq + 1] - begin;
    const int n = keep ? min(max(keep[seq], 0), len) : len;
    if (n <= 0 || (skip_full && n >= len)) return;
    const size_t s_head = (size_t)D * D;
    const int sl = slot[seq], cur = parity[sl] & 1;
    const float* rd = S + (((size_t)sl * 2 + cur) * heads + head) * s_head;
    float* wr = S + (((size_t)sl * 2 + (cur ^ 1)) * heads + head) * s_head;
    float s[R][4];
    load_state<R>(s, rd, value0, lane);
    RowIn<R> next;
    next.load(EMIT ? q : nullptr, k, v, g, beta, begin, head, heads, value0, lane);
    for (int j = 0; j < n; ++j) {
        const RowIn<R> in = next;
        if (j + 1 < n) next.load(EMIT ? q : nullptr, k, v, g, beta, begin + j + 1, head, heads, value0, lane);
        update<R>(s, in.k, in.g, in.v, in.beta, lane);
        if (EMIT) {
            const float yv = readout<R>(s, in.q, lane);
            if ((lane & (R == 8 ? 3 : 7)) == 0)  // one lane of each group holding the same row's sum writes it
                y[((size_t)(begin + j) * heads + head) * D + value0 + row_of_lane<R>(lane)] =
                    RZ ? __float2bfloat16_rz(yv) : __float2bfloat16_rn(yv);
        }
    }
    if (write_final) store_state<R>(wr, s, value0, lane);
}

// ---------------------------------------------------------------------------------------------------------------
// gated RMSNorm with a sigmoid gate (the oracle's GatedRMSNorm, gate_act 1): y_core bf16 per head of 128, z f32 gate, weight
// bf16 [128]: out = ((x * w) * rsqrt(mean(x^2) + eps)) * sigmoid(z), fp16. One warp per (row, head), 4 elements per lane.
// ---------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(128) norm_kernel(
        const __nv_bfloat16* __restrict__ y, const float* __restrict__ z, const __nv_bfloat16* __restrict__ w, float eps,
        int heads, __half* __restrict__ out) {
    const int row = blockIdx.x;
    const int head = blockIdx.y * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const size_t off = ((size_t)row * heads + head) * D + lane * 4;
    float x[4], wt[4];
    load4_bf16(y + off, x);
    load4_bf16(w + lane * 4, wt);
    const float4 zv = *reinterpret_cast<const float4*>(z + off);
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) ss = ss + x[i] * x[i];
    ss = warp_sum(ss);
    const float rmf = 1.0f / sqrtf(ss / (float)D + eps);
    const float zz[4] = {zv.x, zv.y, zv.z, zv.w};
    __half2 h[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        float o[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int c = 2 * i + e;
            o[e] = ((x[c] * wt[c]) * rmf) * sigmoid_ieee(zz[c]);
        }
        h[i] = __floats2half2_rn(o[0], o[1]);
    }
    uint2 u;
    u.x = *reinterpret_cast<const unsigned*>(&h[0]);
    u.y = *reinterpret_cast<const unsigned*>(&h[1]);
    *reinterpret_cast<uint2*>(out + off) = u;
}

// ---------------------------------------------------------------------------------------------------------------
// commit tail: conv shift (new state = last 3 rows of concat(old 3 rows, raw rows 0..n)) and the parity flip. One thread owns
// one (sequence, channel) and reads every source value before writing, so the in-place shift is safe. The flip happens iff
// n > 0 (thread channel 0): by then the state kernels of this commit have finished (stream order) and S[1-cur] holds the
// state after the n accepted rows. keep == nullptr means every row of the sequence is accepted.
// ---------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) commit_tail_kernel(
        const __nv_bfloat16* __restrict__ raw, __nv_bfloat16* conv_pool, int* parity, const int* __restrict__ seq_start,
        const int* __restrict__ slot, const int* __restrict__ keep, int conv_dim, bool flip) {
    const int seq = blockIdx.y, ch = blockIdx.x * 256 + threadIdx.x;
    const int begin = seq_start[seq], len = seq_start[seq + 1] - begin;
    const int n = keep ? min(max(keep[seq], 0), len) : len;
    if (n <= 0 || ch >= conv_dim) return;
    __nv_bfloat16* pool = conv_pool + (size_t)slot[seq] * KEEP * conv_dim + ch;
    __nv_bfloat16 nw[KEEP];
#pragma unroll
    for (int i = 0; i < KEEP; ++i) {
        const int e = n + i;  // index into concat(old 3 rows, raw rows)
        nw[i] = e < KEEP ? pool[(size_t)e * conv_dim] : raw[(size_t)(begin + e - KEEP) * conv_dim + ch];
    }
#pragma unroll
    for (int i = 0; i < KEEP; ++i) pool[(size_t)i * conv_dim] = nw[i];
    if (flip && blockIdx.x == 0 && threadIdx.x == 0) parity[slot[seq]] ^= 1;
}

// The parity flip alone (the commit tail's last step without its conv shift): the pipelined P2 prefill shifts the conv history on its
// side stream as soon as a piece's prep has run and flips `cur` on the main stream after the piece's state kernel.
__global__ void flip_kernel(int* parity, const int* __restrict__ slot, int n_seq) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_seq) parity[slot[i]] ^= 1;
}

// Test hook: the halves-exchanging reduction against the plain xor butterfly on the same data. in f32 [32 lanes][N]; out f32 [32][2]:
// the butterfly's sum for the value this lane holds, and halve_reduce's (they must be bit-identical).
template <int N>
__global__ void reduce_check_kernel(const float* in, float* out) {
    const int lane = threadIdx.x;
    float v[N], b[N];
#pragma unroll
    for (int r = 0; r < N; ++r) v[r] = b[r] = in[lane * N + r];
#pragma unroll
    for (int m = 16; m; m >>= 1) {
#pragma unroll
        for (int r = 0; r < N; ++r) b[r] += __shfl_xor_sync(0xffffffffu, b[r], m);
    }
    const float t = halve_reduce<N>(v, lane);
    const int idx = owned_index<N>(lane);
    float ref = b[0];
#pragma unroll
    for (int r = 1; r < N; ++r) ref = idx == r ? b[r] : ref;
    out[lane * 2] = ref;
    out[lane * 2 + 1] = t;
}

int bad_heads(int heads) { return heads <= 0 || heads % 4 != 0 || heads > 1024; }

constexpr int ERR_INVALID = 1;  // cudaErrorInvalidValue

template <bool EMIT, bool RZ>
int launch_state(int r_step, const float* q, const float* k, const __nv_bfloat16* v, const float* g, const float* beta,
                 float* S, const int* parity, const int* seq_start, const int* slot, const int* keep, int n_seq, int heads,
                 bool skip_full, bool write_final, __nv_bfloat16* y, cudaStream_t st) {
    if (r_step == 4) {
        state_kernel<4, EMIT, RZ><<<dim3(D / (4 * W_STEP), heads, n_seq), 32 * W_STEP, 0, st>>>(
            q, k, v, g, beta, S, parity, seq_start, slot, keep, heads, skip_full, write_final, y);
    } else {
        state_kernel<8, EMIT, RZ><<<dim3(D / (8 * W_STEP), heads, n_seq), 32 * W_STEP, 0, st>>>(
            q, k, v, g, beta, S, parity, seq_start, slot, keep, heads, skip_full, write_final, y);
    }
    return GB_CHECK_LAUNCH();
}

}  // namespace

// b, f_a, g_a GEMVs (f16 weights [heads or rank, hidden]) off x [rows, hidden] of dtype x_dtype (0 f16, 1 bf16, 2 f32).
// Outputs f32: beta [rows, heads] (already bf16(sigmoid)), fa [rows, rank], ga [rows, rank]. `part` (nullable, f32
// [8][rows][heads + 2 rank]): scratch of the lane schedule (proj_in_lane_kernel), which takes hidden 4096 and 64 rows or more (the same
// bits).
GB_EXPORT int gb_kda_proj_in(const void* x, int x_dtype, const void* w_b, const void* w_fa, const void* w_ga, int rows,
                             int hidden, int heads, int rank, float* beta, float* fa, float* ga, float* part, void* stream) {
    if (rows < 0 || hidden <= 0 || hidden % 8 != 0 || rank <= 0 || bad_heads(heads) || x_dtype < 0 || x_dtype > 2)
        return ERR_INVALID;
    if (rows == 0) return 0;
    const int n_out = heads + 2 * rank;
    cudaStream_t st = gb_stream(stream);
    const __half *b = (const __half*)w_b, *f = (const __half*)w_fa, *gg = (const __half*)w_ga;
    if (part && hidden == 4096 && rows >= 64) {
        const int rc = x_dtype == X_F16    ? launch_proj_in_lane<X_F16>(x, b, f, gg, rows, heads, rank, part, st)
                       : x_dtype == X_BF16 ? launch_proj_in_lane<X_BF16>(x, b, f, gg, rows, heads, rank, part, st)
                                           : launch_proj_in_lane<X_F32>(x, b, f, gg, rows, heads, rank, part, st);
        if (rc) return rc;
        const int lrc = GB_CHECK_LAUNCH();
        if (lrc) return lrc;
        const size_t n = (size_t)rows * n_out;
        proj_in_fin_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(part, rows, heads, rank, beta, fa, ga);
        return GB_CHECK_LAUNCH();
    }
    const int ty = (rows + PI_RT - 1) / PI_RT;
    if (rows <= PI_RT) {  // few rows: the weight stream is the cost, one output per block
        const dim3 grid(n_out, ty);
        if (x_dtype == X_F16) proj_in_kernel<X_F16, 1><<<grid, PI_NT, 0, st>>>(x, b, f, gg, rows, hidden, heads, rank, beta, fa, ga);
        else if (x_dtype == X_BF16) proj_in_kernel<X_BF16, 1><<<grid, PI_NT, 0, st>>>(x, b, f, gg, rows, hidden, heads, rank, beta, fa, ga);
        else proj_in_kernel<X_F32, 1><<<grid, PI_NT, 0, st>>>(x, b, f, gg, rows, hidden, heads, rank, beta, fa, ga);
    } else {  // many rows: 4 outputs share each x load (the same bits as one output per block)
        const dim3 grid((n_out + 3) / 4, ty);
        if (x_dtype == X_F16) proj_in_kernel<X_F16, 4><<<grid, PI_NT, 0, st>>>(x, b, f, gg, rows, hidden, heads, rank, beta, fa, ga);
        else if (x_dtype == X_BF16) proj_in_kernel<X_BF16, 4><<<grid, PI_NT, 0, st>>>(x, b, f, gg, rows, hidden, heads, rank, beta, fa, ga);
        else proj_in_kernel<X_F32, 4><<<grid, PI_NT, 0, st>>>(x, b, f, gg, rows, hidden, heads, rank, beta, fa, ga);
    }
    return GB_CHECK_LAUNCH();
}

// f_b and g_b (f16 weights [heads*128, 128]) off fa/ga f32 [rows, 128] (rank must be 128). g [rows, heads*128] is the final
// per-channel DECAY (exp of the log decay), z [rows, heads*128] the f32 output gate; z == nullptr: the decay alone (gb_kda_z_norm takes
// the gate). 64 rows or more take the tiled schedule (the same bits).
GB_EXPORT int gb_kda_proj_out(const float* fa, const float* ga, const void* w_fb, const void* w_gb, const float* a_log,
                              const float* dt_bias, float lower, int rows, int heads, float* g, float* z, void* stream) {
    if (rows < 0 || bad_heads(heads)) return ERR_INVALID;
    if (rows == 0) return 0;
    const int paths = z ? 2 : 1;
    if (rows >= PO2_RM)
        proj_out_tile_kernel<<<dim3(paths * heads, (rows + PO2_RM - 1) / PO2_RM), 256, 0, gb_stream(stream)>>>(
            fa, ga, (const __half*)w_fb, (const __half*)w_gb, a_log, dt_bias, lower, rows, heads, g, z);
    else
        proj_out_small_kernel<<<dim3(paths * heads * (D / PS_C), (rows + PO_RT - 1) / PO_RT), PS_NT, 0, gb_stream(stream)>>>(
            fa, ga, (const __half*)w_fb, (const __half*)w_gb, a_log, dt_bias, lower, rows, heads, g, z);
    return GB_CHECK_LAUNCH();
}

// The output gate of gb_kda_proj_out (off ga f32 [rows, 128] and g_b) and gb_kda_norm with it, in one pass (z_norm_tile_kernel: the same
// bits as the two): y bf16 [rows, heads, 128] the core output, w the norm's bf16 [128] weight, out fp16 [rows, heads * 128].
GB_EXPORT int gb_kda_z_norm(const float* ga, const void* w_gb, const void* y, const void* w, float eps, int rows, int heads, void* out,
                            void* stream) {
    if (rows < 0 || bad_heads(heads)) return ERR_INVALID;
    if (rows == 0) return 0;
    z_norm_tile_kernel<<<dim3(heads, (rows + PO2_RM - 1) / PO2_RM), 256, 0, gb_stream(stream)>>>(
        ga, (const __half*)w_gb, (const __nv_bfloat16*)y, (const __nv_bfloat16*)w, eps, rows, heads, (__half*)out);
    return GB_CHECK_LAUNCH();
}

// The full-rank KDA gate of Ling-3.1-flash-1019: f_out / g_out fp16 [rows, heads * 128] (the full-rank f_proj /
// g_proj EXL3 outputs), x bf16 [rows, hidden], b_weight bf16 [heads, hidden], a_log f32 [heads], dt_bias f32
// [heads * 128], lower = kda_lower_bound (-5.0). Outputs: g_decay_out f32 [rows, heads * 128], z_gate_out f32
// [rows, heads * 128], beta_out f32 [rows, heads] (the bf16-rounded sigmoid widened to f32, as proj_in stores it). One call replaces proj_in (b only) + proj_out for the no-LoRA
// gate; window/replay/norm then run unmodified. hidden must be a multiple of 8; rows <= 65535 (grid y).
GB_EXPORT int gb_kda_gate_from_full_fg(const void* f_out, const void* g_out, const void* x, const void* b_weight,
                                       const float* a_log, const float* dt_bias, float lower, float* g_decay_out,
                                       float* z_gate_out, float* beta_out, int rows, int hidden, int heads, void* stream) {
    if (rows < 0 || bad_heads(heads) || hidden <= 0 || hidden % 8 != 0) return ERR_INVALID;
    if (rows == 0) return 0;
    cudaStream_t st = gb_stream(stream);
    const int C = heads * 128;
    beta_from_b_kernel<X_BF16><<<dim3(heads, (rows + BG_RT - 1) / BG_RT), BG_NT, 0, st>>>(
        x, (const __nv_bfloat16*)b_weight, rows, hidden, heads, beta_out);
    int rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    full_fg_gate_kernel<<<dim3((C + 255) / 256, rows), 256, 0, st>>>(
        (const half*)f_out, (const half*)g_out, a_log, dt_bias, lower, rows, heads, g_decay_out, z_gate_out);
    return GB_CHECK_LAUNCH();
}

// conv + silu + l2norm. in_f32 != 0: `qkv` is f32 [rows, C], else bf16 [rows, C]; `raw_out` (nullable, bf16 [rows, C])
// receives the bf16 rounding of the rows (the record). Replay: `qkv` is the record itself and `raw_out` is null. keep
// (nullable, i32 [n_seq]) limits rows per sequence. q (may be null with emit_q = 0), k f32 [rows, heads, 128], v bf16 [rows, heads, 128].
GB_EXPORT int gb_kda_prep(const void* qkv, int in_f32, const void* conv_w, const void* conv_pool, const int* seq_start,
                          const int* slot, const int* keep, int n_seq, int rows, int heads, int emit_q, int skip_full,
                          void* raw_out, float* q, float* k, void* v, void* stream) {
    if (rows < 0 || n_seq < 0 || bad_heads(heads)) return ERR_INVALID;
    if (rows == 0 || n_seq == 0) return 0;
    // consecutive rows of a sequence share a register window: tile them when sequences are long (verify windows, prefill),
    // one row per warp for batches of single rows (decode), where tiling would only cut the parallelism
    const int tile_rows = rows >= 4 * n_seq ? 8 : 1;
    const dim3 grid((rows + tile_rows - 1) / tile_rows, 3 * heads / 4);
    cudaStream_t st = gb_stream(stream);
    if (in_f32)
        prep_kernel<true><<<grid, 128, 0, st>>>(qkv, (const __nv_bfloat16*)conv_w, (const __nv_bfloat16*)conv_pool, seq_start,
                                                slot, keep, n_seq, rows, tile_rows, heads, emit_q != 0, skip_full != 0,
                                                (__nv_bfloat16*)raw_out, q, k, (__nv_bfloat16*)v);
    else
        prep_kernel<false><<<grid, 128, 0, st>>>(qkv, (const __nv_bfloat16*)conv_w, (const __nv_bfloat16*)conv_pool, seq_start,
                                                 slot, keep, n_seq, rows, tile_rows, heads, emit_q != 0, skip_full != 0,
                                                 (__nv_bfloat16*)raw_out, q, k, (__nv_bfloat16*)v);
    return GB_CHECK_LAUNCH();
}

// The window: y = core output bf16 [rows, heads, 128] (rz != 0: round toward zero, the oracle's choice; default RN),
// final state to S[1-cur] when write_final != 0. r_step: value rows per warp (4 or 8).
GB_EXPORT int gb_kda_window(const float* q, const float* k, const void* v, const float* g, const float* beta, float* S,
                            const int* parity, const int* seq_start, const int* slot, int n_seq, int heads, int rz,
                            int write_final, int r_step, void* y, void* stream) {
    if (n_seq < 0 || bad_heads(heads) || (r_step != 4 && r_step != 8)) return ERR_INVALID;
    if (n_seq == 0) return 0;
    cudaStream_t st = gb_stream(stream);
    if (rz) return launch_state<true, true>(r_step, q, k, (const __nv_bfloat16*)v, g, beta, S, parity, seq_start, slot, nullptr,
                                            n_seq, heads, false, write_final != 0, (__nv_bfloat16*)y, st);
    return launch_state<true, false>(r_step, q, k, (const __nv_bfloat16*)v, g, beta, S, parity, seq_start, slot, nullptr, n_seq,
                                     heads, false, write_final != 0, (__nv_bfloat16*)y, st);
}

// Replay of the accepted prefix: state only, from S[cur] into S[1-cur], keep rows per sequence (device i32 [n_seq]).
// keep == nullptr replays every row. skip_full != 0: a sequence with keep >= len is left alone (the window already wrote its
// final state).
GB_EXPORT int gb_kda_replay(const float* k, const void* v, const float* g, const float* beta, float* S, const int* parity,
                            const int* seq_start, const int* slot, const int* keep, int n_seq, int heads, int skip_full,
                            int r_step, void* stream) {
    if (n_seq < 0 || bad_heads(heads) || (r_step != 4 && r_step != 8)) return ERR_INVALID;
    if (n_seq == 0) return 0;
    return launch_state<false, false>(r_step, nullptr, k, (const __nv_bfloat16*)v, g, beta, S, parity, seq_start, slot, keep,
                                      n_seq, heads, skip_full != 0, true, nullptr, gb_stream(stream));
}

GB_EXPORT int gb_kda_norm(const void* y, const float* z, const void* w, float eps, int rows, int heads, void* out, void* stream) {
    if (rows < 0 || bad_heads(heads)) return ERR_INVALID;
    if (rows == 0) return 0;
    norm_kernel<<<dim3(rows, heads / 4), 128, 0, gb_stream(stream)>>>((const __nv_bfloat16*)y, z, (const __nv_bfloat16*)w, eps,
                                                                    heads, (__half*)out);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_kda_commit_tail(const void* raw, void* conv_pool, int* parity, const int* seq_start, const int* slot,
                                 const int* keep, int n_seq, int heads, int flip, void* stream) {
    if (n_seq < 0 || bad_heads(heads)) return ERR_INVALID;
    if (n_seq == 0) return 0;
    const int conv_dim = 3 * heads * D;
    commit_tail_kernel<<<dim3((conv_dim + 255) / 256, n_seq), 256, 0, gb_stream(stream)>>>(
        (const __nv_bfloat16*)raw, (__nv_bfloat16*)conv_pool, parity, seq_start, slot, keep, conv_dim, flip != 0);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_kda_flip(int* parity, const int* slot, int n_seq, void* stream) {
    if (n_seq < 0) return ERR_INVALID;
    if (n_seq == 0) return 0;
    flip_kernel<<<(n_seq + 127) / 128, 128, 0, gb_stream(stream)>>>(parity, slot, n_seq);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_kda_reduce_check(const float* in, float* out, int r, void* stream) {
    if (r == 4) reduce_check_kernel<4><<<1, 32, 0, gb_stream(stream)>>>(in, out);
    else if (r == 8) reduce_check_kernel<8><<<1, 32, 0, gb_stream(stream)>>>(in, out);
    else if (r == 16) reduce_check_kernel<16><<<1, 32, 0, gb_stream(stream)>>>(in, out);
    else if (r == 32) reduce_check_kernel<32><<<1, 32, 0, gb_stream(stream)>>>(in, out);
    else return ERR_INVALID;
    return GB_CHECK_LAUNCH();
}
