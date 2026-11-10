// Paged KV cache and attention for the 16 full-attention layers (spec 4, 4.2, 9.1, 10).
//
// Layout: K cache and V cache are bf16 [n_blocks][256 tokens][NKV kv heads][256 dim]; a device block table
// i32 [slots][max_blocks] maps (slot, position / 256) -> block id. Keys are stored after norm + rope, values raw.
// q / out are bf16 [rows][NQ][256]; q head h reads kv head h / GRP (GRP = NQ / NKV); the scale is 1/16.
//
// The query and KV head counts are the shape parameters. This file builds NQ = 24, NKV = 4 (Qwen3.8-27B) with the launcher names
// below; attention_kv2.cu builds the same source with NKV = 2 (Qwen3.8-Flash-Next, 24 query heads: groups of 12) and every
// launcher suffixed `_kv2`; attention_q16kv2.cu builds NQ = 16, NKV = 2 (the target layers of MiniCPM-V-4.7's backbone, groups
// of 8: `_q16kv2`) and attention_q8kv2.cu NQ = 8, NKV = 2 (its DSpark draft layers, groups of 4: `_q8kv2`). The 24-head kernels
// are unchanged by the parameters (the goldens in tests/attention.rs and tests/attention_kv2.rs pin them): a (row, query head)
// pair owns its own row of an mma tile whatever the group size, so a pair's bits depend on its own q, its slot's keys and its
// position, not on how many pairs share the CTA.
//
//   gb_kv_write       scatter rows' K/V into their pages.
//   gb_attn_decode    decode and verify windows (<= 128 rows). Keys are split into ABSOLUTE chunks of 512
//                     positions; per (row, q head, chunk) an fp32 online-softmax partial (m, l, o[256]); the row's
//                     partials are merged IN CHUNK ORDER by a fixed formula and rounded once to bf16. Every
//                     (row, q head) pair sits in its own row of an mma tile, so its bits depend only on the row's
//                     own q, its slot's keys and its position: not on the other rows, their number, their order or
//                     the block placement.
//   gb_attn_prefill   flash-style prefill for a chunk of one sequence: 64-key tiles by absolute position, fp32
//                     online softmax, p rounded to bf16 for the PV mma. Chunk invariant by construction (a tile
//                     wholly past a row's position is an exact no-op); its bits differ from decode's.
//
// Ported in part from TensorFold (MIT, commit 9cd52ab): the tile arithmetic and staging of
// src/tensorfold/cuda/kernels/prefill_attention.cu, the chunk/partial/merge scheme of
// src/tensorfold/cuda/kernels/attention.py, and the mma / ldmatrix / cp.async helpers of qmm_frag.cuh.
// See THIRD_PARTY_NOTICES.md.
#include "common.cuh"
#include "pdl.cuh"

#ifndef GB_ATTN_NKV
#define GB_ATTN_NKV 4
#define GB_ATTN_EXPORT(name) name
#endif
#ifndef GB_ATTN_NQ
#define GB_ATTN_NQ 24
#endif

namespace {

// Flash-Next decode launches (gbp_pdl::kFlash): a kernel launched with the programmatic-dependent-launch attribute may start before its
// predecessor finishes, so it calls pdl_wait() before it reads anything an earlier kernel wrote (a no-op in a launch without the
// attribute). Only timing changes, never a bit.
using gbp_pdl::pdl_wait;
using gbp_pdl::pdl_trigger;
inline bool flash_pdl() { return gbp_pdl::on(gbp_pdl::kFlash); }

constexpr int D = 256, NQ = GB_ATTN_NQ, NKV = GB_ATTN_NKV, GRP = NQ / NKV, PAGE = 256, CHUNK = 512, BN = 64, HALF = 32;
static_assert(NKV == 4 || NKV == 2, "the kernels are built and tested for 4 and 2 kv heads");
static_assert(NQ % NKV == 0 && (GRP == 16 || GRP == 12 || GRP == 6 || GRP == 8 || GRP == 4),
              "the kernels are built and tested for query groups of 12, 6, 8 and 4 (a group divides the 48 pairs of a CTA)");
constexpr int C = D / 8;     // 16-byte pieces per row of D bf16
constexpr int K16 = D / 16;  // k16 steps of a q.k dot
constexpr int NB = D / 8;    // n8 blocks of the output row
constexpr int MAX_ROWS = 128;
constexpr int MAXT = 3;      // decode: mma row tiles (of 16 pairs) per CTA
constexpr int RBLK = MAXT * 16 / GRP;  // decode: rows a CTA serves (8 rows x 6 heads, 4 x 12, 6 x 8 or 12 x 4: 48 pairs = 3 mma row tiles)
static_assert(RBLK * GRP == MAXT * 16, "a row block fills the mma row tiles exactly");
// KV write: one thread per 16 bytes of a token row, exactly, so a row writes its own token and nothing past it
constexpr int KV_TOKEN_U4 = NKV * D / 8;

// (1/16) * log2(e): scores live in the log2 domain, so exp is a bare ex2
constexpr float LOG2E = 1.4426950408889634f;
constexpr float SC2 = 0.0625f * LOG2E;

typedef __nv_bfloat16 bf16;

// ---------------------------------------------------------------- fragments (after TensorFold qmm_frag.cuh)

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

// 16 bytes, or zeros without reading when `read` is false
__device__ __forceinline__ void cp16z(void* dst, const void* src, bool read) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(read ? 16 : 0));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// the same mma from a zero accumulator (a chain's first k16 step)
__device__ __forceinline__ void mma0(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    const float z = 0.0f;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(z));
}

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

// 16-byte piece c of row r, XOR-swizzled so ldmatrix's eight rows hit eight bank groups
__device__ __forceinline__ int sw(int r, int c) { return r * C + (c ^ (r & 7)); }

// ---------------------------------------------------------------- shared tile arithmetic

// scores of a slot's 32 keys into four n8 blocks: a chain over D's k16 steps from zero, as the TF kernel's tl.dot
__device__ __forceinline__ void scores_slot(float (*s)[4], const uint4* qw, const uint4* sl, int lane) {
    const int mt = lane >> 3;
#pragma unroll
    for (int k = 0; k < K16; ++k) {
        uint32_t a[4];
        ldmatrix4(a, qw + sw(lane & 15, 2 * k + (lane >> 4)));
#pragma unroll
        for (int n = 0; n < 4; n += 2) {
            uint32_t b[4];
            ldmatrix4(b, sl + sw(8 * (n + (mt >> 1)) + (lane & 7), 2 * k + (mt & 1)));
            if (k == 0) {
                mma0(s[n], a, b[0], b[1]);
                mma0(s[n + 1], a, b[2], b[3]);
            } else {
                mma(s[n], a, b[0], b[1]);
                mma(s[n + 1], a, b[2], b[3]);
            }
        }
    }
}

// a slot's 32 values: two k16 steps (pa0 = keys 0..15, pa1 = keys 16..31) of the output chain, in key order
__device__ __forceinline__ void values_slot(float (&o)[NB][4], const uint32_t (&pa0)[4], const uint32_t (&pa1)[4],
                                            const uint4* sl, int lane) {
    const int mt = lane >> 3;
#pragma unroll
    for (int i = 0; i < NB; i += 2) {
        uint32_t b[4];
        ldmatrix4t(b, sl + sw(8 * (mt & 1) + (lane & 7), i + (mt >> 1)));
        mma(o[i], pa0, b[0], b[1]);
        mma(o[i + 1], pa0, b[2], b[3]);
    }
#pragma unroll
    for (int i = 0; i < NB; i += 2) {
        uint32_t b[4];
        ldmatrix4t(b, sl + sw(16 + 8 * (mt & 1) + (lane & 7), i + (mt >> 1)));
        mma(o[i], pa1, b[0], b[1]);
        mma(o[i + 1], pa1, b[2], b[3]);
    }
}

// Online-softmax state of the two rows (g and g + 8) a thread holds. m is in the log2 domain; l is this thread's
// partial over its columns (reduced across the quad at the end).
struct Soft {
    float m0, m1, l0, l1;
};

// Folds one tile of 64 scores (8 n8 blocks, fp32 mma accumulators) into the state: masks keys past the row's
// position, rescales o, and returns the bf16 probabilities packed for the PV mma. A tile with no visible key in a
// row leaves that row exactly unchanged (alpha = 1, p = 0), so extra masked tiles never change a row's bits.
// `kb` = the absolute position of this thread's first column in the tile.
__device__ __forceinline__ void fold_tile(float (&sc)[8][4], int kb, int pos0, int pos1, Soft& st, float (&o)[NB][4],
                                          uint32_t (&pa)[4][4]) {
    float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
        const int c = kb + 8 * n;
        sc[n][0] = c <= pos0 ? __fmul_rn(sc[n][0], SC2) : -INFINITY;
        sc[n][1] = c + 1 <= pos0 ? __fmul_rn(sc[n][1], SC2) : -INFINITY;
        sc[n][2] = c <= pos1 ? __fmul_rn(sc[n][2], SC2) : -INFINITY;
        sc[n][3] = c + 1 <= pos1 ? __fmul_rn(sc[n][3], SC2) : -INFINITY;
        tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
        tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
    }
#pragma unroll
    for (int x = 1; x <= 2; x *= 2) {
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
    }
    const bool act0 = tm0 != -INFINITY, act1 = tm1 != -INFINITY;
    const float n0 = act0 ? fmaxf(st.m0, tm0) : st.m0, n1 = act1 ? fmaxf(st.m1, tm1) : st.m1;
    const float a0 = act0 ? (st.m0 == -INFINITY ? 0.0f : ex2(__fsub_rn(st.m0, n0))) : 1.0f;
    const float a1 = act1 ? (st.m1 == -INFINITY ? 0.0f : ex2(__fsub_rn(st.m1, n1))) : 1.0f;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
        const int c = kb + 8 * n;
        sc[n][0] = act0 && c <= pos0 ? ex2(__fsub_rn(sc[n][0], n0)) : 0.0f;
        sc[n][1] = act0 && c + 1 <= pos0 ? ex2(__fsub_rn(sc[n][1], n0)) : 0.0f;
        sc[n][2] = act1 && c <= pos1 ? ex2(__fsub_rn(sc[n][2], n1)) : 0.0f;
        sc[n][3] = act1 && c + 1 <= pos1 ? ex2(__fsub_rn(sc[n][3], n1)) : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        pa[j][0] = pack_bf16(sc[2 * j][0], sc[2 * j][1]);
        pa[j][1] = pack_bf16(sc[2 * j][2], sc[2 * j][3]);
        pa[j][2] = pack_bf16(sc[2 * j + 1][0], sc[2 * j + 1][1]);
        pa[j][3] = pack_bf16(sc[2 * j + 1][2], sc[2 * j + 1][3]);
    }
    // l sums the bf16-rounded probabilities the PV mma consumes, so the weights o is built from sum to l exactly
    float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
        const uint32_t w0 = pa[n >> 1][(n & 1) * 2], w1 = pa[n >> 1][(n & 1) * 2 + 1];
        s0 = __fadd_rn(s0, __fadd_rn(__uint_as_float(w0 << 16), __uint_as_float(w0 & 0xffff0000u)));
        s1 = __fadd_rn(s1, __fadd_rn(__uint_as_float(w1 << 16), __uint_as_float(w1 & 0xffff0000u)));
    }
    st.l0 = __fmaf_rn(st.l0, a0, s0);
    st.l1 = __fmaf_rn(st.l1, a1, s1);
    st.m0 = n0;
    st.m1 = n1;
#pragma unroll
    for (int i = 0; i < NB; ++i) {
        o[i][0] = __fmul_rn(o[i][0], a0);
        o[i][1] = __fmul_rn(o[i][1], a0);
        o[i][2] = __fmul_rn(o[i][2], a1);
        o[i][3] = __fmul_rn(o[i][3], a1);
    }
}

// l reduced over the quad that shares a row: lanes xor 1, then xor 2 (every lane ends with the same bits)
__device__ __forceinline__ float quad_sum(float x) {
    x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 1));
    return __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 2));
}

// element offset of (block, token-in-page, kv head 0) in a cache
__device__ __forceinline__ int64_t page_off(int blk, int tok) { return ((int64_t)blk * PAGE + tok) * (NKV * D); }

// ---------------------------------------------------------------- gb_kv_write

__global__ void __launch_bounds__(KV_TOKEN_U4) kv_write_kernel(const uint4* __restrict__ k, const uint4* __restrict__ v,
                                                       const int* __restrict__ row_slot, const int* __restrict__ row_pos,
                                                       const int* __restrict__ table, int max_blocks,
                                                       uint4* __restrict__ kc, uint4* __restrict__ vc) {
    pdl_trigger();
    pdl_wait();
    const int r = blockIdx.x;
    const int pos = row_pos[r];
    if (pos < 0) return;
    const int blk = table[(int64_t)row_slot[r] * max_blocks + pos / PAGE];
    const int64_t dst = page_off(blk, pos % PAGE) / 8;  // uint4 units; a token's NKV heads are contiguous
    const int64_t src = (int64_t)r * KV_TOKEN_U4;
    kc[dst + threadIdx.x] = k[src + threadIdx.x];
    vc[dst + threadIdx.x] = v[src + threadIdx.x];
}

// ---------------------------------------------------------------- gb_attn_decode

// One CTA = (chunk, kv head, block of RBLK rows). The block's rows are grouped by slot; a group's (row, head) pairs fill
// the rows of up to three mma tiles (one warp each) and share every K/V tile loaded for the chunk. A pair's chunk
// partial depends on its own q, its slot's keys of this chunk and its own position only.
template <int NS>
__global__ void __launch_bounds__(128, 1)
decode_partial_kernel(const bf16* __restrict__ q, const bf16* __restrict__ kc, const bf16* __restrict__ vc,
                      const int* __restrict__ table, int max_blocks, const int* __restrict__ row_slot,
                      const int* __restrict__ row_pos, int rows, int nch_max, float* __restrict__ part_o,
                      float* __restrict__ part_m, float* __restrict__ part_l) {
    pdl_trigger();
    pdl_wait();
    // 128-byte aligned on purpose: the static arrays below put the dynamic region at 16/32/48 mod 64 otherwise, and cp.async
    // fills into a base that is not 64-byte aligned run ~25 % slower (decode at 261k: 3.1 ms -> 2.4 ms a layer).
    extern __shared__ __align__(128) uint4 smem[];
    uint4* slots = smem;
    uint4* qs = smem + NS * HALF * C;
    __shared__ int sh_slot[RBLK], sh_pos[RBLK], sh_grow[RBLK], sh_ppos[MAXT * 16];
    __shared__ int sh_ng, sh_max;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, qd = lane & 3;
    // grid (row block, kv head, chunk): the row blocks of one chunk are adjacent in launch order, so a window of more than RBLK
    // rows reads each chunk's K/V from DRAM once and the other row blocks hit L2 (chunk first, 16 rows at 32k: 943 -> 367 us a layer)
    const int rb0 = blockIdx.x * RBLK, hk = blockIdx.y, chunk = blockIdx.z;
    const int nr = min(RBLK, rows - rb0), cstart = chunk * CHUNK;
    if (tid < nr) {
        sh_slot[tid] = row_slot[rb0 + tid];
        sh_pos[tid] = row_pos[rb0 + tid];
    }
    __syncthreads();
    unsigned handled = 0;
    const unsigned full = (1u << nr) - 1;
    while (handled != full) {
        // ---- the next group: the first unhandled row and every later row of its slot
        const int leader = __ffs(~handled & full) - 1;
        const int slot = sh_slot[leader];
        unsigned gmask = 0;
        for (int i = 0; i < nr; ++i)
            if (sh_slot[i] == slot && !((handled >> i) & 1)) gmask |= 1u << i;
        handled |= gmask;
        __syncthreads();  // everyone has read the previous group's lists
        if (tid == 0) {
            int n = 0, mx = -1;
            for (int i = 0; i < nr; ++i)
                if ((gmask >> i) & 1) {
                    sh_grow[n++] = i;
                    mx = max(mx, sh_pos[i]);
                }
            sh_ng = n;
            sh_max = mx;
        }
        __syncthreads();
        const int ng = sh_ng, maxpos = sh_max, np = ng * GRP;
        if (cstart > maxpos) continue;  // no row of the group sees this chunk (uniform across the CTA)
        const int ntq = (np + 15) / 16;
        if (tid < MAXT * 16) sh_ppos[tid] = tid < np ? sh_pos[sh_grow[tid / GRP]] : -1;
        // ---- stage q for the group's pairs (zeros past the last pair)
        for (int i = tid; i < ntq * 16 * C; i += 128) {
            const int idx = i / C;
            const bool ok = idx < np;
            const int row = rb0 + sh_grow[ok ? idx / GRP : 0];
            const bf16* src = q + ((int64_t)row * NQ + hk * GRP + (ok ? idx % GRP : 0)) * D + (i % C) * 8;
            cp16z(qs + sw(idx, i % C), src, ok);
        }
        commit();
        __syncthreads();  // sh_ppos visible
        const int limit = maxpos + 1;
        const int nt = (min(limit, cstart + CHUNK) - cstart + BN - 1) / BN, items = 4 * nt;
        const int* tbl = table + (int64_t)slot * max_blocks;
        auto stage = [&](int item) {
            const bf16* src = item % 4 < 2 ? kc : vc;
            const int key0 = cstart + (item / 4) * BN + (item % 2) * HALF;
            uint4* dst = slots + (item % NS) * HALF * C;
            const int blk = key0 < limit ? tbl[key0 / PAGE] : 0;
            const bf16* base = src + page_off(blk, key0 % PAGE) + hk * D;
            for (int i = tid; i < HALF * C; i += 128) {
                const bool in = key0 + i / C < limit;  // past the group's last key: zeros, never read
                cp16z(dst + sw(i / C, i % C), base + (int64_t)(in ? i / C : 0) * (NKV * D) + (i % C) * 8, in);
            }
        };
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            if (i < items) stage(i);
            commit();
        }
        float o[NB][4];
#pragma unroll
        for (int i = 0; i < NB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
        float sc[8][4];
        uint32_t pa[4][4];
        Soft st = {-INFINITY, -INFINITY, 0.0f, 0.0f};
        const int wt = warp < MAXT ? warp : 0;
        const int pos0 = sh_ppos[wt * 16 + g], pos1 = sh_ppos[wt * 16 + g + 8];
        int wmax = -1;  // the warp's highest visible position (uniform across the warp)
        for (int i = 0; i < 16; ++i) wmax = max(wmax, sh_ppos[wt * 16 + i]);
        const bool compute = warp < ntq;
        const uint4* qw = qs + warp * 16 * C;
        auto take = [&](int it) -> const uint4* {
            wait_group<NS - 1>();
            __syncthreads();
            return slots + (it % NS) * HALF * C;
        };
        auto done = [&](int it) {
            __syncthreads();
            if (it + NS < items) stage(it + NS);
            commit();
        };
        for (int t = 0; t < nt; ++t) {
            const int it = 4 * t;
            const bool act = compute && cstart + t * BN <= wmax;
            const uint4* s0 = take(it);
            if (act) scores_slot(&sc[0], qw, s0, lane);
            done(it);
            const uint4* s1 = take(it + 1);
            if (act) scores_slot(&sc[4], qw, s1, lane);
            done(it + 1);
            if (act) fold_tile(sc, cstart + t * BN + 2 * qd, pos0, pos1, st, o, pa);
            const uint4* v0 = take(it + 2);
            if (act) values_slot(o, pa[0], pa[1], v0, lane);
            done(it + 2);
            const uint4* v1 = take(it + 3);
            if (act) values_slot(o, pa[2], pa[3], v1, lane);
            done(it + 3);
        }
        wait_group<0>();
        if (compute) {
            const float l0 = quad_sum(st.l0), l1 = quad_sum(st.l1);
            const int i0 = warp * 16 + g, i1 = i0 + 8;
            const bool w0 = i0 < np && cstart <= pos0, w1 = i1 < np && cstart <= pos1;
            const int h0 = hk * GRP + i0 % GRP, h1 = hk * GRP + i1 % GRP;
            const int64_t p0 = ((int64_t)(rb0 + sh_grow[w0 ? i0 / GRP : 0]) * NQ + h0) * nch_max + chunk;
            const int64_t p1 = ((int64_t)(rb0 + sh_grow[w1 ? i1 / GRP : 0]) * NQ + h1) * nch_max + chunk;
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                const int d = 8 * i + 2 * qd;
                if (w0) *reinterpret_cast<float2*>(part_o + p0 * D + d) = make_float2(o[i][0], o[i][1]);
                if (w1) *reinterpret_cast<float2*>(part_o + p1 * D + d) = make_float2(o[i][2], o[i][3]);
            }
            if (qd == 0) {
                if (w0) {
                    part_m[p0] = st.m0;
                    part_l[p0] = l0;
                }
                if (w1) {
                    part_m[p1] = st.m1;
                    part_l[p1] = l1;
                }
            }
        }
        __syncthreads();  // before the next group restages q and the slots
    }
}

// (row, q head): merges the row's chunk partials in chunk order, one thread per output column. Every thread folds
// the same (m, l) sequence, so the column bits do not depend on how the columns are split.
__global__ void __launch_bounds__(D) decode_merge_kernel(const float* __restrict__ part_o, const float* __restrict__ part_m,
                                                         const float* __restrict__ part_l, const int* __restrict__ row_pos,
                                                         bf16* __restrict__ out, int nch_max) {
    pdl_trigger();
    pdl_wait();
    const int row = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const int pos = row_pos[row];
    const int nch = pos < 0 ? 0 : min(pos / CHUNK + 1, nch_max);  // a negative position (padding row) writes zeros
    const int64_t base = ((int64_t)row * NQ + head) * nch_max;
    float m = -INFINITY, l = 0.0f, o = 0.0f;
#pragma unroll 4
    for (int c = 0; c < nch; ++c) {
        const float cm = part_m[base + c], cl = part_l[base + c], co = part_o[(base + c) * D + d];
        const float nm = fmaxf(m, cm);
        const float a = m == -INFINITY ? 0.0f : ex2(__fsub_rn(m, nm));
        const float b = ex2(__fsub_rn(cm, nm));
        o = __fmaf_rn(co, b, __fmul_rn(o, a));
        l = __fmaf_rn(cl, b, __fmul_rn(l, a));
        m = nm;
    }
    out[((int64_t)row * NQ + head) * D + d] = __float2bfloat16_rn(nch > 0 ? __fdiv_rn(o, l) : 0.0f);
}

constexpr int DEC_NS = 4;
constexpr int DEC_SMEM = DEC_NS * HALF * C * 16 + MAXT * 16 * C * 16;

// ---------------------------------------------------------------- gb_attn_decode_tree
//
// Tree windows (docs/TREE_DRAFTING.md). A row's keys are the committed ones (positions < base, the sequence's committed length L,
// read through the block table) followed by its ancestors' window keys at logical positions L, L + 1, ..: key (L + j) of row r is the
// K/V of window row paths[r][j]. The window's K/V live in node-indexed buffers (kn, vn: [rows][4][256], never in the paged cache,
// so siblings that share a logical position cannot collide). A row at depth d has logical position L + d and its bits must be those
// of the chain decode of its own path, so the arithmetic is the decode kernel's, tile for tile: the same absolute chunks of 512
// keys and 64-key tiles, the same scores / online-softmax / PV steps in the same order, the same merge. Only where a tile's keys
// come from changes.
//
// One CTA = (chunk, kv head, block of 8 rows) as in decode_partial_kernel, in two phases per chunk. Phase 1 runs the tiles that lie
// wholly below L exactly like the decode kernel: committed keys, staged once for every row of the sequence the CTA holds. Phase 2
// runs the remaining tiles (the one holding L and the ones after it) one ROW at a time: the row's keys are staged by position
// (committed from the pages below L, its ancestors' from the node buffers at and above L, zeros past the row's own position) and
// only the row's pairs fold the tile (the other pairs' positions are masked to -1, which leaves their state exactly unchanged).
// The mma rows are independent, so a pair's bits do not depend on which rows share the tile or where in it the pair sits.
template <int NS>
__global__ void __launch_bounds__(128, 1)
tree_partial_kernel(const bf16* __restrict__ q, const bf16* __restrict__ kc, const bf16* __restrict__ vc,
                    const bf16* __restrict__ kn, const bf16* __restrict__ vn, const int* __restrict__ table, int max_blocks,
                    const int* __restrict__ row_slot, const int* __restrict__ row_pos, const int* __restrict__ row_base,
                    const int* __restrict__ paths, int path_stride, int rows, int nch_max, float* __restrict__ part_o,
                    float* __restrict__ part_m, float* __restrict__ part_l) {
    // 128-byte aligned on purpose: the static arrays below put the dynamic region at 16/32/48 mod 64 otherwise, and cp.async
    // fills into a base that is not 64-byte aligned run ~25 % slower (decode at 261k: 3.1 ms -> 2.4 ms a layer).
    extern __shared__ __align__(128) uint4 smem[];
    uint4* slots = smem;
    uint4* qs = smem + NS * HALF * C;
    __shared__ int sh_slot[RBLK], sh_pos[RBLK], sh_grow[RBLK], sh_ppos[MAXT * 16], sh_base[RBLK];
    __shared__ int sh_ng, sh_max;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, qd = lane & 3;
    const int chunk = blockIdx.y, hk = blockIdx.z, rb0 = blockIdx.x * RBLK;  // row blocks fastest: the CTAs that read one chunk run together (L2)
    const int nr = min(RBLK, rows - rb0), cstart = chunk * CHUNK;
    if (tid < nr) {
        sh_slot[tid] = row_slot[rb0 + tid];
        sh_pos[tid] = row_pos[rb0 + tid];
        sh_base[tid] = row_base[rb0 + tid];
    }
    __syncthreads();
    unsigned handled = 0;
    const unsigned full = (1u << nr) - 1;
    while (handled != full) {
        // ---- the next group: the first unhandled row and every later row of its slot
        const int leader = __ffs(~handled & full) - 1;
        const int slot = sh_slot[leader];
        unsigned gmask = 0;
        for (int i = 0; i < nr; ++i)
            if (sh_slot[i] == slot && !((handled >> i) & 1)) gmask |= 1u << i;
        handled |= gmask;
        __syncthreads();  // everyone has read the previous group's lists
        if (tid == 0) {
            int n = 0, mx = -1;
            for (int i = 0; i < nr; ++i)
                if ((gmask >> i) & 1) {
                    sh_grow[n++] = i;
                    mx = max(mx, sh_pos[i]);
                }
            sh_ng = n;
            sh_max = mx;
        }
        __syncthreads();
        const int ng = sh_ng, maxpos = sh_max, np = ng * GRP;
        if (cstart > maxpos) continue;  // no row of the group sees this chunk (uniform across the CTA)
        const int ntq = (np + 15) / 16;
        if (tid < MAXT * 16) sh_ppos[tid] = tid < np ? sh_pos[sh_grow[tid / GRP]] : -1;
        // ---- stage q for the group's pairs (zeros past the last pair)
        for (int i = tid; i < ntq * 16 * C; i += 128) {
            const int idx = i / C;
            const bool ok = idx < np;
            const int row = rb0 + sh_grow[ok ? idx / GRP : 0];
            const bf16* src = q + ((int64_t)row * NQ + hk * GRP + (ok ? idx % GRP : 0)) * D + (i % C) * 8;
            cp16z(qs + sw(idx, i % C), src, ok);
        }
        commit();
        __syncthreads();  // sh_ppos visible
        const int base = sh_base[sh_grow[0]];  // the group is one sequence: one committed length
        const int limit = maxpos + 1;
        const int nt = (min(limit, cstart + CHUNK) - cstart + BN - 1) / BN;
        // tiles wholly below the committed length: shared by every row of the group, as in the decode kernel
        const int ns = base > cstart ? min(nt, (base - cstart) / BN) : 0;
        const int* tbl = table + (int64_t)slot * max_blocks;
        float o[NB][4];
#pragma unroll
        for (int i = 0; i < NB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
        float sc[8][4];
        uint32_t pa[4][4];
        Soft st = {-INFINITY, -INFINITY, 0.0f, 0.0f};
        const int wt = warp < MAXT ? warp : 0;
        const int pos0 = sh_ppos[wt * 16 + g], pos1 = sh_ppos[wt * 16 + g + 8];
        int wmax = -1;  // the warp's highest visible position (uniform across the warp)
        for (int i = 0; i < 16; ++i) wmax = max(wmax, sh_ppos[wt * 16 + i]);
        const bool compute = warp < ntq;
        const uint4* qw = qs + warp * 16 * C;

        // Runs `tn` tiles starting at tile `t_first` of the chunk through the ring of NS slots; `stage(item)` stages item `item`
        // of the run (4 per tile: keys 0..31, keys 32..63, values 0..31, values 32..63). `row_gi` >= 0: only the pairs of that group
        // row fold the tiles (their positions are `rp`); -1: every pair, with the group's positions.
        auto run_tiles = [&](int t_first, int tn, auto&& stage, int row_gi, int rp) {
            const int items = 4 * tn;
            const bool mine = row_gi < 0 ? true : (wt * 16 < (row_gi + 1) * GRP && wt * 16 + 16 > row_gi * GRP);
            const int idx0 = wt * 16 + g, idx1 = idx0 + 8;
            const int p0 = row_gi < 0 ? pos0 : (idx0 < np && idx0 / GRP == row_gi ? pos0 : -1);
            const int p1 = row_gi < 0 ? pos1 : (idx1 < np && idx1 / GRP == row_gi ? pos1 : -1);
            const int top = row_gi < 0 ? wmax : rp;  // the highest position any folding pair of this warp can see
#pragma unroll
            for (int i = 0; i < NS; ++i) {
                if (i < items) stage(i);
                commit();
            }
            auto take = [&](int it) -> const uint4* {
                wait_group<NS - 1>();
                __syncthreads();
                return slots + (it % NS) * HALF * C;
            };
            auto done = [&](int it) {
                __syncthreads();
                if (it + NS < items) stage(it + NS);
                commit();
            };
            for (int t = 0; t < tn; ++t) {
                const int it = 4 * t;
                const int kt = cstart + (t_first + t) * BN;
                const bool act = compute && mine && kt <= top;
                const uint4* s0 = take(it);
                if (act) scores_slot(&sc[0], qw, s0, lane);
                done(it);
                const uint4* s1 = take(it + 1);
                if (act) scores_slot(&sc[4], qw, s1, lane);
                done(it + 1);
                if (act) fold_tile(sc, kt + 2 * qd, p0, p1, st, o, pa);
                const uint4* v0 = take(it + 2);
                if (act) values_slot(o, pa[0], pa[1], v0, lane);
                done(it + 2);
                const uint4* v1 = take(it + 3);
                if (act) values_slot(o, pa[2], pa[3], v1, lane);
                done(it + 3);
            }
            wait_group<0>();
            __syncthreads();  // the ring is free for the next run
        };

        // ---- phase 1: the tiles wholly below L, committed keys only, staged once for the whole group
        if (ns > 0) {
            auto stage = [&](int item) {
                const bf16* src = item % 4 < 2 ? kc : vc;
                const int key0 = cstart + (item / 4) * BN + (item % 2) * HALF;
                uint4* dst = slots + (item % NS) * HALF * C;
                const int blk = key0 < limit ? tbl[key0 / PAGE] : 0;
                const bf16* bp = src + page_off(blk, key0 % PAGE) + hk * D;
                for (int i = tid; i < HALF * C; i += 128) {
                    const bool in = key0 + i / C < limit;
                    cp16z(dst + sw(i / C, i % C), bp + (int64_t)(in ? i / C : 0) * (NKV * D) + (i % C) * 8, in);
                }
            };
            run_tiles(0, ns, stage, -1, 0);
        }
        // ---- phase 2: the tiles holding or following L, one row at a time with that row's own keys
        for (int gi = 0; gi < ng; ++gi) {
            const int ri = sh_grow[gi];
            const int rp = sh_pos[ri];  // the row's logical position: L + depth
            if (rp < cstart) continue;  // the row does not see this chunk (uniform)
            const int nt_r = (min(rp + 1, cstart + CHUNK) - cstart + BN - 1) / BN;
            if (nt_r <= ns) continue;
            const int* path = paths + (int64_t)(rb0 + ri) * path_stride;
            auto stage = [&](int item) {
                const bool is_k = item % 4 < 2;
                const int key0 = cstart + (ns + item / 4) * BN + (item % 2) * HALF;
                uint4* dst = slots + (item % NS) * HALF * C;
                for (int i = tid; i < HALF * C; i += 128) {
                    const int kr = i / C, kp = key0 + kr, piece = i % C;
                    const bool in = kp <= rp;
                    const bf16* src = is_k ? kc : vc;  // a valid address even when the 16 bytes are not read
                    if (in) {
                        if (kp < base) {  // committed: through the block table
                            src += page_off(tbl[kp / PAGE], kp % PAGE) + hk * D;
                        } else {  // an ancestor's window row (the row itself at kp == rp)
                            src = (is_k ? kn : vn) + ((int64_t)path[kp - base] * NKV + hk) * D;
                        }
                        src += piece * 8;
                    }
                    cp16z(dst + sw(kr, piece), src, in);
                }
            };
            run_tiles(ns, nt_r - ns, stage, gi, rp);
        }
        if (compute) {
            const float l0 = quad_sum(st.l0), l1 = quad_sum(st.l1);
            const int i0 = warp * 16 + g, i1 = i0 + 8;
            const bool w0 = i0 < np && cstart <= pos0, w1 = i1 < np && cstart <= pos1;
            const int h0 = hk * GRP + i0 % GRP, h1 = hk * GRP + i1 % GRP;
            const int64_t p0 = ((int64_t)(rb0 + sh_grow[w0 ? i0 / GRP : 0]) * NQ + h0) * nch_max + chunk;
            const int64_t p1 = ((int64_t)(rb0 + sh_grow[w1 ? i1 / GRP : 0]) * NQ + h1) * nch_max + chunk;
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                const int d = 8 * i + 2 * qd;
                if (w0) *reinterpret_cast<float2*>(part_o + p0 * D + d) = make_float2(o[i][0], o[i][1]);
                if (w1) *reinterpret_cast<float2*>(part_o + p1 * D + d) = make_float2(o[i][2], o[i][3]);
            }
            if (qd == 0) {
                if (w0) {
                    part_m[p0] = st.m0;
                    part_l[p0] = l0;
                }
                if (w1) {
                    part_m[p1] = st.m1;
                    part_l[p1] = l1;
                }
            }
        }
        __syncthreads();  // before the next group restages q and the slots
    }
}

// ---------------------------------------------------------------- gb_kv_write_rows

// Scatter window rows into their pages by an explicit source row: a tree window's accepted path goes from the node buffers (row
// src_row[i] of k / v) to position row_pos[i] of slot row_slot[i]. A negative position skips the entry. Destinations must be distinct.
__global__ void __launch_bounds__(KV_TOKEN_U4) kv_write_rows_kernel(const uint4* __restrict__ k, const uint4* __restrict__ v,
                                                            const int* __restrict__ src_row, const int* __restrict__ row_slot,
                                                            const int* __restrict__ row_pos, const int* __restrict__ table,
                                                            int max_blocks, uint4* __restrict__ kc, uint4* __restrict__ vc) {
    const int e = blockIdx.x;
    const int pos = row_pos[e];
    if (pos < 0) return;
    const int blk = table[(int64_t)row_slot[e] * max_blocks + pos / PAGE];
    const int64_t dst = page_off(blk, pos % PAGE) / 8;
    const int64_t src = (int64_t)src_row[e] * KV_TOKEN_U4;
    kc[dst + threadIdx.x] = k[src + threadIdx.x];
    vc[dst + threadIdx.x] = v[src + threadIdx.x];
}


// ---------------------------------------------------------------- gb_attn_prefill

// One CTA = HPC q heads of one kv head x RB blocks of 16 consecutive rows (one warp each). Every staged 32-key slot
// of K or V serves all WARPS warps. Rows are the chunk's [p0, p0 + W) positions; keys come from the paged cache.
template <int WARPS, int HPC, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
prefill_kernel(const bf16* __restrict__ q, const bf16* __restrict__ kc, const bf16* __restrict__ vc,
               const int* __restrict__ tbl, bf16* __restrict__ out, int p0, int W) {
    static_assert(WARPS % HPC == 0 && GRP % HPC == 0, "heads a CTA must divide the group and the warps");
    constexpr int RB = WARPS / HPC, THREADS = 32 * WARPS;
    // 128-byte aligned on purpose: the static arrays below put the dynamic region at 16/32/48 mod 64 otherwise, and cp.async
    // fills into a base that is not 64-byte aligned run ~25 % slower (decode at 261k: 3.1 ms -> 2.4 ms a layer).
    extern __shared__ __align__(128) uint4 smem[];
    uint4* slots = smem;
    uint4* qs = smem + NS * HALF * C;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, qd = lane & 3;
    const int groups = GRP / HPC, hk = blockIdx.y / groups, hg = blockIdx.y % groups;
    const int rbi = gridDim.x - 1 - blockIdx.x;  // the heaviest row blocks (most keys) launch first
    const int row0 = rbi * 16 * RB, last = min(row0 + 16 * RB, W) - 1;
    const int head = hk * GRP + hg * HPC + warp % HPC, wrow = row0 + (warp / HPC) * 16;
    const int tiles = (p0 + last) / BN + 1, items = 4 * tiles, limit = p0 + W;
    // item i: tile i / 4's keys [0, 32), keys [32, 64), values [0, 32), values [32, 64)
    auto stage = [&](int item) {
        const bf16* src = item % 4 < 2 ? kc : vc;
        const int key0 = (item / 4) * BN + (item % 2) * HALF;
        uint4* dst = slots + (item % NS) * HALF * C;
        const int blk = key0 < limit ? tbl[key0 / PAGE] : 0;
        const bf16* base = src + page_off(blk, key0 % PAGE) + hk * D;
        for (int i = tid; i < HALF * C; i += THREADS) {
            const bool in = key0 + i / C < limit;  // past the chunk's last key: zeros, never read
            cp16z(dst + sw(i / C, i % C), base + (int64_t)(in ? i / C : 0) * (NKV * D) + (i % C) * 8, in);
        }
    };
    for (int i = tid; i < WARPS * 16 * C; i += THREADS) {
        const int w = i / (16 * C), r = (i / C) % 16, row = row0 + (w / HPC) * 16 + r;
        const int hd = hk * GRP + hg * HPC + w % HPC;
        cp16z(qs + sw(i / C, i % C), q + ((int64_t)min(row, W - 1) * NQ + hd) * D + (i % C) * 8, row < W);
    }
    commit();
#pragma unroll
    for (int i = 0; i < NS; ++i) {
        if (i < items) stage(i);
        commit();
    }
    float o[NB][4];
#pragma unroll
    for (int i = 0; i < NB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float sc[8][4];
    uint32_t pa[4][4];
    Soft st = {-INFINITY, -INFINITY, 0.0f, 0.0f};
    const int pos0 = p0 + wrow + g, pos1 = pos0 + 8;
    const int wmax = p0 + min(wrow + 15, W - 1);  // the warp's highest visible position
    const bool alive = wrow < W;
    const uint4* qw = qs + warp * 16 * C;
    auto take = [&](int it) -> const uint4* {
        wait_group<NS - 1>();
        __syncthreads();
        return slots + (it % NS) * HALF * C;
    };
    auto done = [&](int it) {
        __syncthreads();
        if (it + NS < items) stage(it + NS);
        commit();
    };
    for (int t = 0; t < tiles; ++t) {
        const int it = 4 * t;
        const bool act = alive && t * BN <= wmax;  // a tile wholly past the warp's rows is an exact no-op: skipped
        const uint4* s0 = take(it);
        if (act) scores_slot(&sc[0], qw, s0, lane);
        done(it);
        const uint4* s1 = take(it + 1);
        if (act) scores_slot(&sc[4], qw, s1, lane);
        done(it + 1);
        if (act) fold_tile(sc, t * BN + 2 * qd, pos0, pos1, st, o, pa);
        const uint4* v0 = take(it + 2);
        if (act) values_slot(o, pa[0], pa[1], v0, lane);
        done(it + 2);
        const uint4* v1 = take(it + 3);
        if (act) values_slot(o, pa[2], pa[3], v1, lane);
        done(it + 3);
    }
    wait_group<0>();
    if (!alive) return;
    const float l0 = quad_sum(st.l0), l1 = quad_sum(st.l1);
    const int r0 = wrow + g, r1 = r0 + 8;
#pragma unroll
    for (int i = 0; i < NB; ++i) {
        const int d = 8 * i + 2 * qd;
        if (r0 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r0 * NQ + head) * D + d) =
                pack_bf16(__fdiv_rn(o[i][0], l0), __fdiv_rn(o[i][1], l0));
        if (r1 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r1 * NQ + head) * D + d) =
                pack_bf16(__fdiv_rn(o[i][2], l1), __fdiv_rn(o[i][3], l1));
    }
}

constexpr int SMEM_MAX = 99 * 1024;

template <int WARPS, int HPC, int MAXNS>
int launch_prefill(const bf16* q, const bf16* kc, const bf16* vc, const int* tbl, bf16* out, int p0, int W,
                   cudaStream_t s) {
    constexpr int QBYTES = WARPS * 16 * C * 16, SLOT = HALF * C * 16;
    constexpr int NS0 = (SMEM_MAX - QBYTES) / SLOT, NS = NS0 < MAXNS ? NS0 : MAXNS;
    static_assert(NS >= 2, "shared memory for two slots");
    constexpr int smem = NS * SLOT + QBYTES;
    static const cudaError_t configured =
        cudaFuncSetAttribute(prefill_kernel<WARPS, HPC, NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (configured != cudaSuccess) return (int)configured;
    constexpr int rows = 16 * (WARPS / HPC);
    const dim3 grid((W + rows - 1) / rows, NKV * (GRP / HPC));
    prefill_kernel<WARPS, HPC, NS><<<grid, 32 * WARPS, smem, s>>>(q, kc, vc, tbl, out, p0, W);
    return GB_CHECK_LAUNCH();
}

// A CTA shape exists only where its heads a CTA divide the query group (prefill_kernel's static_assert): 6 heads of a group of 8 or
// 4, or 3 of a group of 8, are no shape at all. Refused at run time with an invalid-value error and never instantiated (the
// `if constexpr` keeps the kernel template out of the build), so a tuning sweep that walks the variants gets an error for those.
template <int WARPS, int HPC, int MAXNS>
int launch_prefill_shape(const bf16* q, const bf16* kc, const bf16* vc, const int* tbl, bf16* out, int p0, int W, cudaStream_t s) {
    if constexpr (WARPS % HPC == 0 && GRP % HPC == 0) {
        return launch_prefill<WARPS, HPC, MAXNS>(q, kc, vc, tbl, out, p0, W, s);
    } else {
        return (int)cudaErrorInvalidValue;
    }
}

size_t align256(size_t x) { return (x + 255) & ~(size_t)255; }

}  // namespace

// ---------------------------------------------------------------- exported launchers

GB_EXPORT int GB_ATTN_EXPORT(gb_kv_write)(const void* k, const void* v, const int* row_slot, const int* row_pos, const int* block_table,
                          int max_blocks, void* k_cache, void* v_cache, int rows, void* stream) {
    if (rows < 0 || max_blocks <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    gbp_pdl::launch_pdl(flash_pdl(), kv_write_kernel, dim3(rows), dim3(KV_TOKEN_U4), 0, gb_stream(stream), (const uint4*)k, (const uint4*)v,
                        row_slot, row_pos, block_table, max_blocks, (uint4*)k_cache, (uint4*)v_cache);
    return GB_CHECK_LAUNCH();
}

// Partials (o fp32 [rows][24][nch][256], m, l fp32 [rows][24][nch]) with nch = ceil(max_context / 512).
GB_EXPORT size_t GB_ATTN_EXPORT(gb_attn_decode_workspace_bytes)(int rows, int max_context) {
    const size_t nch = (size_t)((max_context + CHUNK - 1) / CHUNK > 0 ? (max_context + CHUNK - 1) / CHUNK : 1);
    const size_t pairs = (size_t)rows * NQ * nch;
    return align256(pairs * D * 4) + align256(pairs * 4) + align256(pairs * 4);
}

// q, out: bf16 [rows][24][256]. row_slot, row_pos: i32 [rows] (device); every row_pos must be < max_context and its
// pages mapped in the block table. rows <= 128 (a 1..128-row verify window or several streams' rows).
GB_EXPORT int GB_ATTN_EXPORT(gb_attn_decode)(const void* q, const void* k_cache, const void* v_cache, const int* block_table,
                             int max_blocks, const int* row_slot, const int* row_pos, void* out, int rows,
                             int max_context, void* workspace, size_t workspace_bytes, void* stream) {
    if (rows < 1 || rows > MAX_ROWS || max_context < 1 || max_blocks <= 0) return (int)cudaErrorInvalidValue;
    if (workspace_bytes < GB_ATTN_EXPORT(gb_attn_decode_workspace_bytes)(rows, max_context) || !workspace)
        return (int)cudaErrorInvalidValue;
    static const cudaError_t configured =
        cudaFuncSetAttribute(decode_partial_kernel<DEC_NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, DEC_SMEM);
    if (configured != cudaSuccess) return (int)configured;
    const int nch = (max_context + CHUNK - 1) / CHUNK;
    if (nch > 65535) return (int)cudaErrorInvalidValue;  // the chunk is gridDim.z
    const size_t pairs = (size_t)rows * NQ * nch;
    float* po = (float*)workspace;
    float* pm = (float*)((char*)workspace + align256(pairs * D * 4));
    float* pl = (float*)((char*)workspace + align256(pairs * D * 4) + align256(pairs * 4));
    cudaStream_t s = gb_stream(stream);
    gbp_pdl::launch_pdl(flash_pdl(), decode_partial_kernel<DEC_NS>, dim3((rows + RBLK - 1) / RBLK, NKV, nch), dim3(128), DEC_SMEM, s,
                        (const bf16*)q, (const bf16*)k_cache, (const bf16*)v_cache, block_table, max_blocks, row_slot, row_pos, rows, nch,
                        po, pm, pl);
    int rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    gbp_pdl::launch_pdl(flash_pdl(), decode_merge_kernel, dim3(rows, NQ), dim3(D), 0, s, po, pm, pl, row_pos, (bf16*)out, nch);
    return GB_CHECK_LAUNCH();
}

// Tree windows (see tree_partial_kernel). q, out: bf16 [rows][24][256]; row_slot, row_pos: i32 [rows] (device), row_pos the row's LOGICAL
// position (L + depth); row_base: i32 [rows], the committed length L of the row's sequence (its root's position); paths: i32
// [rows][path_stride], row r's window rows root to r (global row indices into kn / vn), -1 after; kn, vn: bf16 [rows][NKV][256], the
// window rows' K and V. The paged cache must hold the committed keys 0..L-1 of every sequence and is never written. Partials and merge
// are the decode kernel's, so the workspace is gb_attn_decode_workspace_bytes. rows <= 128 (the decode grid), path_stride >= the
// deepest row's depth + 1.
GB_EXPORT int GB_ATTN_EXPORT(gb_attn_decode_tree)(const void* q, const void* k_cache, const void* v_cache, const void* k_nodes, const void* v_nodes,
                                  const int* block_table, int max_blocks, const int* row_slot, const int* row_pos,
                                  const int* row_base, const int* paths, int path_stride, void* out, int rows, int max_context,
                                  void* workspace, size_t workspace_bytes, void* stream) {
    if (rows < 1 || rows > MAX_ROWS || max_context < 1 || max_blocks <= 0 || path_stride < 1) return (int)cudaErrorInvalidValue;
    if (!paths || !row_base || !k_nodes || !v_nodes) return (int)cudaErrorInvalidValue;
    if (workspace_bytes < GB_ATTN_EXPORT(gb_attn_decode_workspace_bytes)(rows, max_context) || !workspace) return (int)cudaErrorInvalidValue;
    static const cudaError_t configured =
        cudaFuncSetAttribute(tree_partial_kernel<DEC_NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, DEC_SMEM);
    if (configured != cudaSuccess) return (int)configured;
    const int nch = (max_context + CHUNK - 1) / CHUNK;
    const size_t pairs = (size_t)rows * NQ * nch;
    float* po = (float*)workspace;
    float* pm = (float*)((char*)workspace + align256(pairs * D * 4));
    float* pl = (float*)((char*)workspace + align256(pairs * D * 4) + align256(pairs * 4));
    cudaStream_t s = gb_stream(stream);
    tree_partial_kernel<DEC_NS><<<dim3((rows + RBLK - 1) / RBLK, nch, NKV), 128, DEC_SMEM, s>>>(
        (const bf16*)q, (const bf16*)k_cache, (const bf16*)v_cache, (const bf16*)k_nodes, (const bf16*)v_nodes, block_table, max_blocks,
        row_slot, row_pos, row_base, paths, path_stride, rows, nch, po, pm, pl);
    int rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    decode_merge_kernel<<<dim3(rows, NQ), D, 0, s>>>(po, pm, pl, row_pos, (bf16*)out, nch);
    return GB_CHECK_LAUNCH();
}

// Entry i copies row src_row[i] of k / v (bf16 [rows][NKV][256], the node buffers) to position row_pos[i] of slot row_slot[i]; a negative
// position skips it. Destinations must be distinct. The commit of an accepted tree path.
GB_EXPORT int GB_ATTN_EXPORT(gb_kv_write_rows)(const void* k, const void* v, const int* src_row, const int* row_slot, const int* row_pos,
                               const int* block_table, int max_blocks, void* k_cache, void* v_cache, int n, void* stream) {
    if (n < 0 || max_blocks <= 0) return (int)cudaErrorInvalidValue;
    if (n == 0) return 0;
    kv_write_rows_kernel<<<n, KV_TOKEN_U4, 0, gb_stream(stream)>>>((const uint4*)k, (const uint4*)v, src_row, row_slot, row_pos, block_table,
                                                            max_blocks, (uint4*)k_cache, (uint4*)v_cache);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT size_t GB_ATTN_EXPORT(gb_attn_prefill_workspace_bytes)(int rows) {
    (void)rows;
    return 0;  // flash-style: nothing is spilled
}

// Prefill chunk of ONE sequence: q, out bf16 [rows][24][256] at positions start .. start + rows - 1; the chunk's own
// K/V must already be in the cache (gb_kv_write); the table row is `slot`. rows <= 4096. `variant` picks the CTA
// shape (tuning); 0 is the default. A variant whose heads a CTA do not divide the query group (2: 6 heads, 3: 3 heads, at groups
// of 8 and 4) is refused with cudaErrorInvalidValue.
GB_EXPORT int GB_ATTN_EXPORT(gb_attn_prefill_variant)(const void* q, const void* k_cache, const void* v_cache, const int* block_table,
                                      int max_blocks, int slot, int start, int rows, void* out, int variant,
                                      void* stream) {
    if (rows < 0 || rows > 4096 || start < 0 || max_blocks <= 0 || slot < 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    const bf16 *qq = (const bf16*)q, *kc = (const bf16*)k_cache, *vc = (const bf16*)v_cache;
    const int* tbl = block_table + (int64_t)slot * max_blocks;
    cudaStream_t s = gb_stream(stream);
    switch (variant) {
        case 1: return launch_prefill_shape<8, 1, 8>(qq, kc, vc, tbl, (bf16*)out, start, rows, s);
        case 2: return launch_prefill_shape<6, 6, 8>(qq, kc, vc, tbl, (bf16*)out, start, rows, s);
        case 3: return launch_prefill_shape<6, 3, 8>(qq, kc, vc, tbl, (bf16*)out, start, rows, s);
        case 4: return launch_prefill_shape<4, 2, 8>(qq, kc, vc, tbl, (bf16*)out, start, rows, s);
        default: return launch_prefill_shape<8, 2, 8>(qq, kc, vc, tbl, (bf16*)out, start, rows, s);
    }
}

GB_EXPORT int GB_ATTN_EXPORT(gb_attn_prefill)(const void* q, const void* k_cache, const void* v_cache, const int* block_table,
                              int max_blocks, int slot, int start, int rows, void* out, void* stream) {
    return GB_ATTN_EXPORT(gb_attn_prefill_variant)(q, k_cache, v_cache, block_table, max_blocks, slot, start, rows, out, 0, stream);
}
