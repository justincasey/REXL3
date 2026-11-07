// gb-nvcc: --fmad=false
// GLM-5.3-Flash latent attention for the rows of a LARGE prefill chunk (slabs of up to 128 rows, dense and sparse rows in one launch): the arithmetic of
// csrc/mla.cu's mla_partial_kernel + mla_merge_kernel reproduced BIT FOR BIT, in one kernel that never writes the fp32 chunk partials.
//
// Why: mla.cu runs a CTA per (row, head tile, chunk of 256 keys) and merges a row's chunk partials in a second kernel. At 128 rows a slab that is 151 MB
// of fp32 partials written and read back (the sparse regime has 9 chunks, 64 heads, 512 columns), and the CTA of the decode path holds 80 KB of shared
// memory, one CTA, four warps, on an SM whose limit is 100 KB. This kernel
//   * serves a (row, pair of head tiles): 8 warps, two head tiles sharing every K/V tile (half the gather traffic, twice the warps an SM runs);
//   * walks the row's chunks IN ORDER inside the CTA, and after the last tile of each chunk folds that chunk's (m, l, o) into a running merged state in
//     registers with mla_merge_kernel's formula (nm = max, a = ex2(m - nm), b = ex2(cm - nm), o = fma(co, b, o * a), l = fma(cl, b, l * a), a chunk with
//     no real key (l = 0) skipped), and writes the final o / l once, rounded to T, as mla_merge_kernel does.
// Every number is the one the two-kernel path computes: a chunk's partial starts from zero state exactly as in mla_partial_kernel (tiles of 32 keys, the
// four d-quarter score chains added in the fixed order 0..3, the log2-domain online softmax, PV in key order), and the merge is the same fp32 sequence per
// element. The regime of a row is the one attend_auto gives it: pos < dense_limit is dense (a padding row, pos < 0, writes zeros), otherwise the row is
// sparse when its meta flag says so (and is left untouched otherwise).
//
// Derived from csrc/mla.cu (its helpers, tiling and arithmetic; that file's header names the TensorFold attention it follows: MIT, pinned 9cd52ab; see
// THIRD_PARTY_NOTICES.md).
#include <stdint.h>

#include <type_traits>

#include "common.cuh"
#include "mla_kv8.cuh"

namespace {

constexpr int DC = 512;
constexpr int NH = 64;
constexpr int PAGE = 256;
constexpr int HT = 16;
constexpr int NWARP = 4;  // d-quarters = warps a head tile
constexpr int DQ = DC / NWARP;
constexpr int PQ = DQ / 8;
constexpr int KT = 32;   // keys a tile
constexpr int CH = 256;  // keys (or list entries) a chunk
constexpr int TPC = CH / KT;
constexpr int MT = 2;  // head tiles a CTA
constexpr int NTHR = 128 * MT;

constexpr float LOG2E = 1.4426950408889634f;
constexpr float SC2 = 0.0625f * LOG2E;

typedef __nv_bfloat16 bf16;
typedef __half f16;

template <typename T> __device__ __forceinline__ uint32_t pack2(float lo, float hi) {
    if constexpr (std::is_same<T, bf16>::value) {
        __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
        return *reinterpret_cast<uint32_t*>(&v);
    } else {
        __half2 v = __floats2half2_rn(lo, hi);
        return *reinterpret_cast<uint32_t*>(&v);
    }
}
template <typename T> __device__ __forceinline__ void unpack2(uint32_t w, float& lo, float& hi) {
    if constexpr (std::is_same<T, bf16>::value) {
        lo = __uint_as_float(w << 16);
        hi = __uint_as_float(w & 0xffff0000u);
    } else {
        lo = __half2float(__ushort_as_half((unsigned short)(w & 0xffffu)));
        hi = __half2float(__ushort_as_half((unsigned short)(w >> 16)));
    }
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16z(void* dst, const void* src, bool read) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(read ? 16 : 0));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
template <typename T>
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    if constexpr (std::is_same<T, bf16>::value) {
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    } else {
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
}
template <typename T>
__device__ __forceinline__ void mma0(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    const float z = 0.0f;
    if constexpr (std::is_same<T, bf16>::value) {
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%10, %10, %10, %10};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(z));
    } else {
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%10, %10, %10, %10};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(z));
    }
}
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float quad_sum(float x) {
    x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 1));
    return __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 2));
}
__device__ __forceinline__ int sw(int r, int c) { return r * PQ + (c ^ (r & 7)); }

struct Soft {
    float m0, m1, l0, l1;
};

// grid: rows x 2 (a pair of head tiles each); 256 threads: warp (mi, w) = head tile mi of the pair, d-quarter w.
// Q8: the cache is the fp8 page (mla_kv8.cuh); a thread loads its pieces of tile t + 1 into registers after tile t's barrier and
// dequantises them into the same T tile after tile t's arithmetic (mla.cu's MT = 2 path), so the output is the T kernel's on the
// dequantised cache, bit for bit.
template <typename T, bool Q8>
__global__ void __launch_bounds__(NTHR, 1)
mla_pf_kernel(const T* __restrict__ qlat, const void* __restrict__ cache, const int* __restrict__ table, int max_blocks,
              const int* __restrict__ row_slot, const int* __restrict__ row_pos, int dense_limit,
              const int* __restrict__ sel_idx, int sel_stride, const int* __restrict__ sel_meta, T* __restrict__ out) {
    const T* lat = static_cast<const T*>(cache);
    const uint8_t* lat8 = static_cast<const uint8_t*>(cache);
    constexpr int NBS = KT / 8;
    constexpr int NKS = KT / 16;
    constexpr int TILE_U4 = KT * PQ;
    constexpr int SX_F4 = NBS * 32;
    extern __shared__ uint4 smem[];
    float4* sx = reinterpret_cast<float4*>(smem + 2 * NWARP * TILE_U4);
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, g = lane >> 2, qd = lane & 3, mt = lane >> 3;
    const int w = wid & (NWARP - 1), mi = wid / NWARP;
    const int row = blockIdx.x / 2, hb = (blockIdx.x % 2) * MT + mi;
    const int pos_row = row_pos[row];
    const bool sparse = pos_row >= dense_limit;
    int last;
    if (sparse) {
        if (sel_meta[2 * row] == 0 || sel_meta[2 * row + 1] < 1) return;  // not a sparse row: left untouched
        last = sel_meta[2 * row + 1] - 1;
    } else {
        last = pos_row;
    }
    const int h0 = hb * HT + g, h1 = h0 + 8;
    T* dst0 = out + ((int64_t)row * NH + h0) * DC + w * DQ + 2 * qd;
    T* dst1 = out + ((int64_t)row * NH + h1) * DC + w * DQ + 2 * qd;
    if (last < 0) {  // a padding row: zeros
#pragma unroll
        for (int i = 0; i < DQ / 8; ++i) {
            *reinterpret_cast<uint32_t*>(dst0 + 8 * i) = 0u;
            *reinterpret_cast<uint32_t*>(dst1 + 8 * i) = 0u;
        }
        return;
    }
    const int limit = last + 1;
    const int nchunks = last / CH + 1;
    const int nt_last = (limit - (nchunks - 1) * CH + KT - 1) / KT;
    const int total = (nchunks - 1) * TPC + nt_last;  // tiles of the row, chunk after chunk
    const int* tbl = table + (int64_t)row_slot[row] * max_blocks;
    unsigned vm0 = 0, vm1 = 0;

    // stage the tile with global index gt into stage st; every thread stages a share of the whole tile (the head tiles share it), a CTA barrier
    // makes it visible. Past the last tile it commits an empty group.
    auto stage = [&](int gt, int st) {
        const bool any = gt < total;
        const int key0 = (gt / TPC) * CH + (gt % TPC) * KT;
        const int p = tid & 63, q = p >> 4, c = p & 15;
        uint4* dst = smem + (st * NWARP + q) * TILE_U4;
        if (sparse) {
            // entry j of the tile is selection-list entry key0 + j: every warp resolves the 32 entries (lane j = entry j), the copy loop fetches
            // the position and block of its keys by shuffle
            const int e = key0 + lane;
            const int pl = (any && e < limit) ? sel_idx[(int64_t)row * sel_stride + e] : -1;
            const int bl = pl >= 0 ? tbl[pl / PAGE] : 0;
            const unsigned vmask = __ballot_sync(0xffffffffu, pl >= 0);
            if (st == 0) {
                vm0 = vmask;
            } else {
                vm1 = vmask;
            }
#pragma unroll
            for (int it = 0; it < KT * 64 / NTHR; ++it) {
                const int j = (tid >> 6) + (NTHR / 64) * it;  // the same for the whole warp
                const int pj = __shfl_sync(0xffffffffu, pl, j), bj = __shfl_sync(0xffffffffu, bl, j);
                const bool in = pj >= 0;  // entries past the list's end and holes (< 0): zeros, never read, masked in the fold
                cp16z(dst + j * PQ + (c ^ (j & 7)), in ? lat + ((int64_t)bj * PAGE + pj % PAGE) * DC + p * 8 : lat, in);
            }
        } else {
            const int blk = any ? tbl[key0 / PAGE] : 0;
            const T* base = lat + ((int64_t)blk * PAGE + key0 % PAGE) * DC;
#pragma unroll
            for (int it = 0; it < KT * 64 / NTHR; ++it) {
                const int j = (tid >> 6) + (NTHR / 64) * it;
                const bool in = any && key0 + j < limit;  // keys past the row's last: zeros, never read
                cp16z(dst + j * PQ + (c ^ (j & 7)), in ? base + (int64_t)j * DC + p * 8 : lat, in);
            }
        }
    };
    // Q8: tile gt's pieces (and, lane j, the scale of key j) into registers, then dequantised into stage st. A masked key loads nothing
    // and stores zeros, as cp16z does.
    constexpr int NPC = KT * 64 / NTHR;
    uint2 pv[Q8 ? NPC : 1];
    float psk = 0.0f;
    unsigned pvm = 0;
    auto load8 = [&](int gt) {
        const bool any = gt < total;
        const int key0 = (gt / TPC) * CH + (gt % TPC) * KT;
        const int p = tid & 63;
        if (sparse) {
            const int e = key0 + lane;
            const int pl = (any && e < limit) ? sel_idx[(int64_t)row * sel_stride + e] : -1;
            const int bl = pl >= 0 ? tbl[pl / PAGE] : 0;
            pvm = __ballot_sync(0xffffffffu, pl >= 0);
            psk = pl >= 0 ? mla8::scale(lat8, bl, pl % PAGE) : 0.0f;
#pragma unroll
            for (int it = 0; it < NPC; ++it) {
                const int j = (tid >> 6) + (NTHR / 64) * it;
                const int pj = __shfl_sync(0xffffffffu, pl, j), bj = __shfl_sync(0xffffffffu, bl, j);
                pv[it] = pj >= 0 ? *reinterpret_cast<const uint2*>(mla8::row(lat8, bj, pj % PAGE) + p * 8) : make_uint2(0u, 0u);
            }
        } else {
            const int blk = any ? tbl[key0 / PAGE] : 0, tok0 = key0 % PAGE;
            psk = (any && key0 + lane < limit) ? mla8::scale(lat8, blk, tok0 + lane) : 0.0f;
#pragma unroll
            for (int it = 0; it < NPC; ++it) {
                const int j = (tid >> 6) + (NTHR / 64) * it;
                const bool in = any && key0 + j < limit;
                pv[it] = in ? *reinterpret_cast<const uint2*>(mla8::row(lat8, blk, tok0 + j) + p * 8) : make_uint2(0u, 0u);
            }
        }
    };
    auto store8 = [&](int st) {
        if (sparse) {
            if (st == 0) {
                vm0 = pvm;
            } else {
                vm1 = pvm;
            }
        }
        const int p = tid & 63, q = p >> 4, c = p & 15;
#pragma unroll
        for (int it = 0; it < NPC; ++it) {
            const int j = (tid >> 6) + (NTHR / 64) * it;
            smem[(st * NWARP + q) * TILE_U4 + j * PQ + (c ^ (j & 7))] = mla8::dq8<T>(pv[it], __shfl_sync(0xffffffffu, psk, j));
        }
    };
    if constexpr (Q8) {
        load8(0);
        store8(0);
    } else {
        stage(0, 0);
        commit();
    }

    // this warp's quarter of the 16 heads' q_lat as A fragments
    uint32_t qa[DQ / 16][4];
    {
        const T* qrow = qlat + ((int64_t)row * NH + hb * HT) * DC + w * DQ + 2 * qd;
#pragma unroll
        for (int k = 0; k < DQ / 16; ++k) {
            qa[k][0] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)g * DC + 16 * k);
            qa[k][1] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)(g + 8) * DC + 16 * k);
            qa[k][2] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)g * DC + 16 * k + 8);
            qa[k][3] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)(g + 8) * DC + 16 * k + 8);
        }
    }
    float o[DQ / 8][4];   // the chunk in flight
    float ro[DQ / 8][4];  // the merged state of the chunks done
#pragma unroll
    for (int i = 0; i < DQ / 8; ++i) {
        o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
        ro[i][0] = ro[i][1] = ro[i][2] = ro[i][3] = 0.0f;
    }
    Soft st = {-INFINITY, -INFINITY, 0.0f, 0.0f};
    float rm0 = -INFINITY, rm1 = -INFINITY, rl0 = 0.0f, rl1 = 0.0f;

    for (int t = 0; t < total; ++t) {
        if constexpr (!Q8) wait_group<0>();  // tile t has landed (it is the only group in flight)
        __syncthreads();  // every thread's pieces are visible, and every warp is past the last read of stage (t + 1) & 1
        if constexpr (Q8) {
            load8(t + 1);  // stored after this tile's arithmetic
        } else {
            stage(t + 1, (t + 1) & 1);
            commit();
        }
        const int c = t / TPC, tc = t % TPC;
        const int cstart = c * CH;
        if (tc == 0) {  // a chunk starts from zero state, as in mla_partial_kernel
            st.m0 = st.m1 = -INFINITY;
            st.l0 = st.l1 = 0.0f;
#pragma unroll
            for (int i = 0; i < DQ / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
        }
        const uint4* tile = smem + ((t & 1) * NWARP + w) * TILE_U4;
        // ---- this quarter's score chain: 8 k16 steps from zero
        float s[NBS][4];
#pragma unroll
        for (int k = 0; k < DQ / 16; ++k) {
#pragma unroll
            for (int n = 0; n < NBS; n += 2) {
                uint32_t b[4];
                ldmatrix4(b, tile + sw(8 * (n + (mt >> 1)) + (lane & 7), 2 * k + (mt & 1)));
                if (k == 0) {
                    mma0<T>(s[n], qa[k], b[0], b[1]);
                    mma0<T>(s[n + 1], qa[k], b[2], b[3]);
                } else {
                    mma<T>(s[n], qa[k], b[0], b[1]);
                    mma<T>(s[n + 1], qa[k], b[2], b[3]);
                }
            }
        }
        // ---- the four quarters' partial scores of this head tile added in the fixed order 0, 1, 2, 3
        float4* mine = sx + (((t & 1) * MT + mi) * NWARP + w) * SX_F4;
#pragma unroll
        for (int n = 0; n < NBS; ++n) mine[n * 32 + lane] = make_float4(s[n][0], s[n][1], s[n][2], s[n][3]);
        asm volatile("bar.sync %0, 128;\n" ::"r"(1 + mi) : "memory");  // the four warps of this head tile
        const float4* all = sx + ((t & 1) * MT + mi) * NWARP * SX_F4;
        float sc[NBS][4];
#pragma unroll
        for (int n = 0; n < NBS; ++n) {
            const float4 a = all[0 * SX_F4 + n * 32 + lane], b = all[1 * SX_F4 + n * 32 + lane];
            const float4 cc = all[2 * SX_F4 + n * 32 + lane], d = all[3 * SX_F4 + n * 32 + lane];
            sc[n][0] = __fadd_rn(__fadd_rn(__fadd_rn(a.x, b.x), cc.x), d.x);
            sc[n][1] = __fadd_rn(__fadd_rn(__fadd_rn(a.y, b.y), cc.y), d.y);
            sc[n][2] = __fadd_rn(__fadd_rn(__fadd_rn(a.z, b.z), cc.z), d.z);
            sc[n][3] = __fadd_rn(__fadd_rn(__fadd_rn(a.w, b.w), cc.w), d.w);
        }
        // ---- fold the tile into the online softmax (the mask is the row's own position or the entry's validity; all 16 heads share it)
        const int kb = cstart + tc * KT + 2 * qd;
        const unsigned vm = (t & 1) ? vm1 : vm0;
        bool v0[NBS], v1[NBS];
#pragma unroll
        for (int n = 0; n < NBS; ++n) {
            if (sparse) {
                v0[n] = (vm >> (8 * n + 2 * qd)) & 1u;
                v1[n] = (vm >> (8 * n + 2 * qd + 1)) & 1u;
            } else {
                v0[n] = kb + 8 * n <= last;
                v1[n] = kb + 8 * n + 1 <= last;
            }
        }
        float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < NBS; ++n) {
            sc[n][0] = v0[n] ? __fmul_rn(sc[n][0], SC2) : -INFINITY;
            sc[n][1] = v1[n] ? __fmul_rn(sc[n][1], SC2) : -INFINITY;
            sc[n][2] = v0[n] ? __fmul_rn(sc[n][2], SC2) : -INFINITY;
            sc[n][3] = v1[n] ? __fmul_rn(sc[n][3], SC2) : -INFINITY;
            tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
            tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
        }
#pragma unroll
        for (int x = 1; x <= 2; x *= 2) {
            tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
            tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
        }
        const float n0 = fmaxf(st.m0, tm0), n1 = fmaxf(st.m1, tm1);
        const float a0 = st.m0 == -INFINITY ? 0.0f : ex2(__fsub_rn(st.m0, n0));
        const float a1 = st.m1 == -INFINITY ? 0.0f : ex2(__fsub_rn(st.m1, n1));
#pragma unroll
        for (int n = 0; n < NBS; ++n) {
            sc[n][0] = v0[n] ? ex2(__fsub_rn(sc[n][0], n0)) : 0.0f;
            sc[n][1] = v1[n] ? ex2(__fsub_rn(sc[n][1], n0)) : 0.0f;
            sc[n][2] = v0[n] ? ex2(__fsub_rn(sc[n][2], n1)) : 0.0f;
            sc[n][3] = v1[n] ? ex2(__fsub_rn(sc[n][3], n1)) : 0.0f;
        }
        uint32_t pa[NKS][4];
#pragma unroll
        for (int j = 0; j < NKS; ++j) {
            pa[j][0] = pack2<T>(sc[2 * j][0], sc[2 * j][1]);
            pa[j][1] = pack2<T>(sc[2 * j][2], sc[2 * j][3]);
            pa[j][2] = pack2<T>(sc[2 * j + 1][0], sc[2 * j + 1][1]);
            pa[j][3] = pack2<T>(sc[2 * j + 1][2], sc[2 * j + 1][3]);
        }
        float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
        for (int n = 0; n < NBS; ++n) {
            float x0, x1, y0, y1;
            unpack2<T>(pa[n >> 1][(n & 1) * 2], x0, x1);
            unpack2<T>(pa[n >> 1][(n & 1) * 2 + 1], y0, y1);
            s0 = __fadd_rn(s0, __fadd_rn(x0, x1));
            s1 = __fadd_rn(s1, __fadd_rn(y0, y1));
        }
        st.l0 = __fmaf_rn(st.l0, a0, s0);
        st.l1 = __fmaf_rn(st.l1, a1, s1);
        st.m0 = n0;
        st.m1 = n1;
#pragma unroll
        for (int i = 0; i < DQ / 8; ++i) {
            o[i][0] = __fmul_rn(o[i][0], a0);
            o[i][1] = __fmul_rn(o[i][1], a0);
            o[i][2] = __fmul_rn(o[i][2], a1);
            o[i][3] = __fmul_rn(o[i][3], a1);
        }
        // ---- PV over this quarter's output columns, keys in order
#pragma unroll
        for (int ks = 0; ks < NKS; ++ks) {
#pragma unroll
            for (int i = 0; i < DQ / 8; i += 2) {
                uint32_t b[4];
                ldmatrix4t(b, tile + sw(16 * ks + 8 * (mt & 1) + (lane & 7), i + (mt >> 1)));
                mma<T>(o[i], pa[ks], b[0], b[1]);
                mma<T>(o[i + 1], pa[ks], b[2], b[3]);
            }
        }
        // ---- the last tile of the chunk: fold the chunk into the merged state (mla_merge_kernel's formula, per head row and element)
        const int nt_c = (c == nchunks - 1) ? nt_last : TPC;
        if (tc == nt_c - 1) {
            const float cl0 = quad_sum(st.l0), cl1 = quad_sum(st.l1);
            if (cl0 > 0.0f) {
                const float nm = fmaxf(rm0, st.m0);
                const float a = rm0 == -INFINITY ? 0.0f : ex2(__fsub_rn(rm0, nm));
                const float b = ex2(__fsub_rn(st.m0, nm));
#pragma unroll
                for (int i = 0; i < DQ / 8; ++i) {
                    ro[i][0] = __fmaf_rn(o[i][0], b, __fmul_rn(ro[i][0], a));
                    ro[i][1] = __fmaf_rn(o[i][1], b, __fmul_rn(ro[i][1], a));
                }
                rl0 = __fmaf_rn(cl0, b, __fmul_rn(rl0, a));
                rm0 = nm;
            }
            if (cl1 > 0.0f) {
                const float nm = fmaxf(rm1, st.m1);
                const float a = rm1 == -INFINITY ? 0.0f : ex2(__fsub_rn(rm1, nm));
                const float b = ex2(__fsub_rn(st.m1, nm));
#pragma unroll
                for (int i = 0; i < DQ / 8; ++i) {
                    ro[i][2] = __fmaf_rn(o[i][2], b, __fmul_rn(ro[i][2], a));
                    ro[i][3] = __fmaf_rn(o[i][3], b, __fmul_rn(ro[i][3], a));
                }
                rl1 = __fmaf_rn(cl1, b, __fmul_rn(rl1, a));
                rm1 = nm;
            }
        }
        if constexpr (Q8) store8((t + 1) & 1);  // tile t + 1: every warp was past its last read of that stage at this tile's barrier
    }
    if constexpr (!Q8) wait_group<0>();
    // ---- o / l, rounded to T (a row with no real key at all writes zeros)
#pragma unroll
    for (int i = 0; i < DQ / 8; ++i) {
        *reinterpret_cast<uint32_t*>(dst0 + 8 * i) =
            rl0 > 0.0f ? pack2<T>(__fdiv_rn(ro[i][0], rl0), __fdiv_rn(ro[i][1], rl0)) : 0u;
        *reinterpret_cast<uint32_t*>(dst1 + 8 * i) =
            rl1 > 0.0f ? pack2<T>(__fdiv_rn(ro[i][2], rl1), __fdiv_rn(ro[i][3], rl1)) : 0u;
    }
}

constexpr int kSmem = 2 * NWARP * KT * PQ * 16 + 2 * MT * NWARP * (KT / 8) * 32 * 16;

template <typename T, bool Q8>
int launch(const void* q_lat, const void* cache, const int* table, int max_blocks, const int* row_slot, const int* row_pos, int dense_limit,
           const int* sel_idx, int sel_stride, const int* sel_meta, void* out, int rows, cudaStream_t s) {
    static const cudaError_t configured =
        cudaFuncSetAttribute(mla_pf_kernel<T, Q8>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    if (configured != cudaSuccess) return (int)configured;
    mla_pf_kernel<T, Q8><<<rows * 2, NTHR, kSmem, s>>>((const T*)q_lat, cache, table, max_blocks, row_slot, row_pos, dense_limit, sel_idx,
                                                       sel_stride, sel_meta, (T*)out);
    return GB_CHECK_LAUNCH();
}

}  // namespace

// q_lat, out: T [rows][64][512]; cache: kv 0, T [n_blocks][256][512]; kv 1, the fp8 page of mla_kv8.cuh; row_slot, row_pos: i32 [rows]. A row
// with pos < dense_limit attends to keys 0..=pos (a negative position writes zeros); a row at or past it, whose sel_meta flag is 1, attends to the
// keys its selection list names (the layout of gb_mla_attend_sparse); any other row is left untouched. No workspace. rows 1..=128 (the slab).
GB_EXPORT int gb_mla_attend_pf(const void* q_lat, const void* cache, const int* block_table, int max_blocks, const int* row_slot,
                               const int* row_pos, int dense_limit, const int* sel_idx, int sel_stride, const int* sel_meta, void* out,
                               int rows, int dtype, int kv, void* stream) {
    if (rows < 1 || rows > 128 || max_blocks <= 0 || dtype < 0 || dtype > 1 || kv < 0 || kv > 1) return (int)cudaErrorInvalidValue;
    cudaStream_t s = gb_stream(stream);
#define PF_ARGS q_lat, cache, block_table, max_blocks, row_slot, row_pos, dense_limit, sel_idx, sel_stride, sel_meta, out, rows, s
    if (dtype == 1) return kv ? launch<bf16, true>(PF_ARGS) : launch<bf16, false>(PF_ARGS);
    return kv ? launch<f16, true>(PF_ARGS) : launch<f16, false>(PF_ARGS);
#undef PF_ARGS
}
