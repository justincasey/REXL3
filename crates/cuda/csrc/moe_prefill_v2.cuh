// The second form of the two grouped GEMMs of moe_prefill.cu (this file is included inside that file's anonymous namespace, after `down_kernel`).
//
// What changes against gateup_kernel / down_kernel: the warps of a team own COLUMNS (warp w the 16 columns 16 w .. 16 w + 15 of the block's 128),
// not the quarters of K, so there is no cross-warp reduction (the 16 barriers a sweep of `reduce_sweep` were a quarter of a block's time). Each warp
// runs the decode path's four K chains (the warp quarters of gb_moe_default_plan: SK = 1, WK = 4) one after the other over ALL k tiles, folding
// each chain into a running sum in warp order (tot = chain 0; tot += chain 1; tot += chain 2; tot += chain 3: the decode epilogue's
// `red[0] + red[1] + red[2] + red[3]`, element by element), every chain a fresh fp32 accumulate from +0.0 over its k tiles in order. So every
// output is the SAME fp32 number as the first form's (same chains, same operands, same adds in the same order); only which thread holds it
// differs, and the epilogues are the first form's own code on those numbers.
//
// A and the trellis stream through a cp.async ring in shared memory: the A fragments of the item's tiles are loaded once for the whole team and read
// by every warp, the tile of a (k tile, column tile) is decoded by the one warp that owns the column tile. A stage is KPS k tiles (a multiple of the
// chain length's divisors; one 128-wide block of K at KPS = 8), the ring ST stages deep; one barrier a stage.
//
// Taken only when its preconditions hold (`v2_gu_shape` / `v2_dn_shape`, and the shared memory fits): SK = 1, WK = 4, MUL1, and the chain length a
// multiple of KPS. Anything else runs the first form. Every device address in the trellis tables must be 16-byte aligned (cp.async); every
// REXL3 tensor is (an arena carve or a cudaMalloc, 256-byte aligned, tiles of 32 * bits bytes).

__device__ __forceinline__ void cpa16(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cpa_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

constexpr int kLds2 = 128 + 4;   // row stride (floats) of the epilogue staging: rows are 16-byte aligned, the 8 rows of a fragment hit different banks

// Loads one pipeline stage of a team of NTH threads: the A fragments of the item's first `nt` tiles (KPS k tiles each) and the KPS x 8 trellis tiles
// of the block (`T`: the expert's trellis, `tn`: tiles across N, `nb`: the block).
template <int K2, int MT, int KPS, int NTH>
__device__ __forceinline__ void load_stage2(uint32_t* sa, const uint32_t* XS, const uint32_t* T, int tile0, int nt, int KT, size_t tn, int nb, int kc,
                                            int tt) {
    constexpr int TW = tile_words<K2>();
    constexpr int AU = MT * KPS * 32;
#pragma unroll
    for (int i = 0; i < (AU + NTH - 1) / NTH; ++i) {
        const int u = i * NTH + tt;
        if (u < AU) {
            const int p = u / (KPS * 32), kk = (u / 32) % KPS, ln = u & 31;
            if (p < nt) cpa16(sa + (size_t)((p * KPS + kk) * 32 + ln) * 4, XS + ((size_t)(tile0 + p) * KT + kc * KPS + kk) * 128 + ln * 4);
        }
    }
    uint32_t* sb = sa + MT * KPS * 128;
    constexpr int BU = KPS * 8 * (TW / 4);
#pragma unroll
    for (int i = 0; i < (BU + NTH - 1) / NTH; ++i) {
        const int u = i * NTH + tt;
        if (u < BU) {
            const int c = u % (TW / 4), tl = u / (TW / 4), kk = tl >> 3, tc = tl & 7;
            cpa16(sb + (size_t)tl * TW + c * 4, T + ((size_t)(kc * KPS + kk) * tn + nb * 8 + tc) * TW + c * 4);
        }
    }
}

// One warp's K chains over the stages of the ring (`sm`: the team's ring). Leaves the warp's running sum in `tot` (its 16 columns x MT * 16 rows, in the
// mma accumulator layout). Every thread of the team calls it with the same trip counts (it holds the team's barriers).
template <int K2, int CB, int MT, int KPS, int ST, int NTH>
__device__ __forceinline__ void chains2(float (&tot)[MT][2][4], uint32_t* sm, const uint32_t* XS, const uint32_t* T, int tile0, int nt, int KT, size_t tn,
                                        int nb, int tt, int warp, int lane) {
    constexpr int TW = tile_words<K2>();
    constexpr int STAGE_W = MT * KPS * 128 + KPS * 8 * TW;
    const int NS = KT / KPS;
    const int cls = (KT / kWK) / KPS;                 // stages of one chain
    float acc[MT][2][4];
#pragma unroll
    for (int p = 0; p < MT; ++p)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[p][h][c] = 0.f;
#pragma unroll
    for (int s = 0; s < ST - 1; ++s) {
        if (s < NS) load_stage2<K2, MT, KPS, NTH>(sm + s * STAGE_W, XS, T, tile0, nt, KT, tn, nb, s, tt);
        cpa_commit();
    }
    int cstage = 0;
    bool first = true;
#pragma unroll 1
    for (int kc = 0; kc < NS; ++kc) {
        cpa_wait<ST - 2>();
        __syncthreads();
        {
            const int nk = kc + ST - 1;
            if (nk < NS) load_stage2<K2, MT, KPS, NTH>(sm + (nk % ST) * STAGE_W, XS, T, tile0, nt, KT, tn, nb, nk, tt);
            cpa_commit();
        }
        const uint32_t* sa = sm + (kc % ST) * STAGE_W;
        const uint32_t* sb = sa + MT * KPS * 128;
#pragma unroll
        for (int kk = 0; kk < KPS; ++kk) {
            uint32_t w[lane_words<K2>()], b0[2], b1[2];
            load_lane_words<K2>(sb + (kk * 8 + warp) * TW, lane, w);
            decode_lane<K2, CB>(w, lane, b0, b1);
#pragma unroll
            for (int p = 0; p < MT; ++p) {
                if (p < nt) {
                    const uint4 av = *reinterpret_cast<const uint4*>(sa + (size_t)((p * KPS + kk) * 32 + lane) * 4);
                    const uint32_t a[4] = {av.x, av.y, av.z, av.w};
                    mma16816(acc[p][0], a, b0);
                    mma16816(acc[p][1], a, b1);
                }
            }
        }
        if (++cstage == cls) {                        // the end of a chain: it joins the running sum (the first one starts it), the next restarts from +0.0
#pragma unroll
            for (int p = 0; p < MT; ++p)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        tot[p][h][c] = first ? acc[p][h][c] : tot[p][h][c] + acc[p][h][c];
                        acc[p][h][c] = 0.f;
                    }
            first = false;
            cstage = 0;
        }
    }
    cpa_wait<0>();
}

// The warp's sums into the team's staging rows [MT * 16][kLds2] (columns 16 warp ..).
template <int MT>
__device__ __forceinline__ void tot_to_stage(const float (&tot)[MT][2][4], float* stg, int nt, int warp, int lane) {
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int p = 0; p < MT; ++p) {
        if (p < nt) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = warp * 16 + h * 8 + 2 * t4, r = p * 16 + g;
                *reinterpret_cast<float2*>(stg + r * kLds2 + col) = make_float2(tot[p][h][0], tot[p][h][1]);
                *reinterpret_cast<float2*>(stg + (r + 8) * kLds2 + col) = make_float2(tot[p][h][2], tot[p][h][3]);
            }
        }
    }
}

// Stage geometry by tile count MT (MEASURED on the GB10, a layer of 4,096 rows with real routing: fewer barriers beat a deeper ring until the shared
// memory runs out): k tiles a stage and stages in the ring. Gate/up: one 128-wide block of K a stage; down: its chain (10 k tiles) in two.
constexpr int kGuKps = 8;
constexpr int kDnKps = 5;
constexpr int gu_st(int mt) { return mt == 4 ? 2 : mt == 2 ? 3 : 4; }
constexpr int dn_st(int mt) { return mt == 4 ? 3 : mt == 2 ? 4 : 6; }
constexpr size_t kSmemMax2 = 101376;                 // what a block may map (99 KB)
constexpr size_t kSmemDn2 = 48000;                   // down runs two blocks an SM: each stays under half of what an SM has

template <int K2, int MT>
constexpr size_t smem_gateup2() {
    constexpr size_t stage = (size_t)MT * kGuKps * 128 + (size_t)kGuKps * 8 * tile_words<K2>();
    constexpr size_t pipe = 2 * gu_st(MT) * stage, epi = (size_t)2 * MT * 16 * kLds2;
    return (pipe > epi ? pipe : epi) * 4;
}

template <int K2, int MT>
constexpr size_t smem_down2() {
    constexpr size_t stage = (size_t)MT * kDnKps * 128 + (size_t)kDnKps * 8 * tile_words<K2>();
    constexpr size_t pipe = dn_st(MT) * stage, epi = (size_t)MT * 16 * kLds2;
    return (pipe > epi ? pipe : epi) * 4;
}

template <int K2, int CB, int MT>
constexpr bool v2_gu_fits() {
    return CB == 2 && smem_gateup2<K2, MT>() <= kSmemMax2;
}
template <int K2, int CB, int MT>
constexpr bool v2_dn_fits() {
    return CB == 2 && smem_down2<K2, MT>() <= kSmemDn2;
}
// The unified launches (gateup2u_kernel / down2u_kernel) run the three tile counts in one kernel and need every one of them to fit; the shared memory a
// block maps is the largest of the three.
template <int K2>
constexpr size_t smem_gateup2u() {
    return smem_gateup2<K2, 4>() > smem_gateup2<K2, 2>() ? smem_gateup2<K2, 4>() : smem_gateup2<K2, 2>();
}
template <int K2>
constexpr size_t smem_down2u() {
    return smem_down2<K2, 4>() > smem_down2<K2, 2>() ? smem_down2<K2, 4>() : smem_down2<K2, 2>();
}
template <int K2, int CB>
constexpr bool v2_gu_fits_u() {
    return CB == 2 && smem_gateup2<K2, 1>() <= smem_gateup2u<K2>() && smem_gateup2u<K2>() <= kSmemMax2;
}
template <int K2, int CB>
constexpr bool v2_dn_fits_u() {
    return CB == 2 && smem_down2<K2, 1>() <= smem_down2u<K2>() && smem_down2u<K2>() <= kSmemDn2;
}
// The shape side of the preconditions: one K split, WK warps' chains whole stages.
inline bool v2_gu_shape(int K, int sk) { return sk == 1 && K % 128 == 0 && ((K / 16) / kWK) % kGuKps == 0; }
inline bool v2_dn_shape(int K, int sk) { return sk == 1 && K % 128 == 0 && ((K / 16) / kWK) % kDnKps == 0; }

// One work item of the second form (block = 128 columns `blockIdx.x` of `item`): the MT-tile instance of the kernel body. Not inlined: the unified
// launches pick the instance at run time, and the three instances inlined into one kernel shared one register allocation and spilled to local memory
// fourteen times as much as the class kernels did (MEASURED: 21 M spill requests a layer against 1.5 M); a call gives each instance its own.
template <int K2, int CB, int MT, int KPS, int ST>
__device__ __noinline__ void gateup2_item(const int4 item, const uint32_t* __restrict__ XS0, const uint32_t* __restrict__ XS1,
                                             const uint64_t* __restrict__ TP0, const uint64_t* __restrict__ TP1,
                                             const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
                                             uint32_t* __restrict__ xd, int K, int N, float limit) {
    constexpr int TW = tile_words<K2>();
    constexpr int STAGE_W = MT * KPS * 128 + KPS * 8 * TW;
    constexpr int STG_W = MT * 16 * kLds2;
    extern __shared__ __align__(16) uint32_t smem2[];          // [team 0 ring][team 1 ring], then the staging [gate rows][up rows] in the same bytes
    const int tid = threadIdx.x, team = tid >> 8, tt = tid & 255, warp = tt >> 5, lane = tid & 31;
    const int e = item.x, tile0 = item.y, nt = item.z, vrows = item.w;
    const int nb = blockIdx.x;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(team ? TP1[e] : TP0[e]);
    const uint32_t* XS = team ? XS1 : XS0;
    const int KT = K >> 4;
    float tot[MT][2][4];
    chains2<K2, CB, MT, KPS, ST, 256>(tot, smem2 + team * (ST * STAGE_W), XS, T, tile0, nt, KT, (size_t)(N >> 4), nb, tt, warp, lane);
    __syncthreads();                                   // the rings are the staging now
    float* stg = reinterpret_cast<float*>(smem2) + team * STG_W;
    tot_to_stage<MT>(tot, stg, nt, warp, lane);
    __syncthreads();
    // the epilogue of gateup_kernel, on the same fp32 sums (the rows spread over 16 warps)
    const float* stage_g = reinterpret_cast<const float*>(smem2);
    const float* stage_u = stage_g + STG_W;
    const int n = nb * 128 + 4 * lane;
    float sg4[4], su4[4], sd4[4];
    load4h(svh_g + (size_t)e * N + n, sg4);
    load4h(svh_u + (size_t)e * N + n, su4);
    load4h(suh_d + (size_t)e * N + n, sd4);
    for (int r = tid >> 5; r < vrows; r += 16) {
        const float4 g4 = *reinterpret_cast<const float4*>(stage_g + r * kLds2 + 4 * lane);
        const float4 u4 = *reinterpret_cast<const float4*>(stage_u + r * kLds2 + 4 * lane);
        float gv[4] = {g4.x, g4.y, g4.z, g4.w}, uv[4] = {u4.x, u4.y, u4.z, u4.w}, v[4];
        fwht128(gv, lane);
        fwht128(uv, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float g = gv[j] * HAD_SCALE * sg4[j];
            float u = uv[j] * HAD_SCALE * su4[j];
            float x = silu_f(g);
            if (limit > 0.0f) {
                x = fminf(x, limit);
                u = fminf(fmaxf(u, -limit), limit);
            }
            v[j] = (x * u) * sd4[j];
        }
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
        store_frag4(xd, tile0 * 16 + r, n, N, v);
    }
}

// One launch a class of items (1 tile, 2 tiles, 3 or 4 tiles): the MT of the class.
template <int K2, int CB, int MT, int KPS, int ST>
__global__ void __launch_bounds__(512, 1) gateup2_kernel(const uint32_t* __restrict__ XS0, const uint32_t* __restrict__ XS1,
                                                         const uint64_t* __restrict__ TP0, const uint64_t* __restrict__ TP1,
                                                         const int4* __restrict__ items, const int* __restrict__ nitems,
                                                         const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                                         const half* __restrict__ suh_d, uint32_t* __restrict__ xd, int K, int N, float limit) {
    const int it = blockIdx.y;
    if (it >= nitems[0]) return;
    gateup2_item<K2, CB, MT, KPS, ST>(items[it], XS0, XS1, TP0, TP1, svh_g, svh_u, suh_d, xd, K, N, limit);
}

// One launch for every item, in the order of the unified list (an expert's items one after the other, so the 5 column blocks of the items of an expert
// that run together stream its trellis from DRAM once): the MT of an item is picked from its tile count (1, 2, 3 or 4 tiles -> MT 1, 2, 4, 4), each with
// its own ring depth.
template <int K2, int CB, int KPS>
__global__ void __launch_bounds__(512, 1) gateup2u_kernel(const uint32_t* __restrict__ XS0, const uint32_t* __restrict__ XS1,
                                                          const uint64_t* __restrict__ TP0, const uint64_t* __restrict__ TP1,
                                                          const int4* __restrict__ items, const int* __restrict__ nitems,
                                                          const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                                          const half* __restrict__ suh_d, uint32_t* __restrict__ xd, int K, int N, float limit) {
    const int it = blockIdx.y;
    if (it >= nitems[0]) return;
    const int4 item = items[it];
    if (item.z == 1)
        gateup2_item<K2, CB, 1, KPS, gu_st(1)>(item, XS0, XS1, TP0, TP1, svh_g, svh_u, suh_d, xd, K, N, limit);
    else if (item.z == 2)
        gateup2_item<K2, CB, 2, KPS, gu_st(2)>(item, XS0, XS1, TP0, TP1, svh_g, svh_u, suh_d, xd, K, N, limit);
    else
        gateup2_item<K2, CB, 4, KPS, gu_st(4)>(item, XS0, XS1, TP0, TP1, svh_g, svh_u, suh_d, xd, K, N, limit);
}

template <int K2, int CB, int MT, int KPS, int ST>
__device__ __noinline__ void down2_item(const int4 item, const uint32_t* __restrict__ XS, const uint64_t* __restrict__ TP,
                                           const half* __restrict__ svh_d, const int* __restrict__ spair, float* __restrict__ y, int K, int N) {
    extern __shared__ __align__(16) uint32_t smem2[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int e = item.x, tile0 = item.y, nt = item.z, vrows = item.w;
    const int nb = blockIdx.x;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(TP[e]);
    const int KT = K >> 4;
    float tot[MT][2][4];
    chains2<K2, CB, MT, KPS, ST, 256>(tot, smem2, XS, T, tile0, nt, KT, (size_t)(N >> 4), nb, tid, warp, lane);
    __syncthreads();
    float* stg = reinterpret_cast<float*>(smem2);
    tot_to_stage<MT>(tot, stg, nt, warp, lane);
    __syncthreads();
    // the epilogue of down_kernel, on the same fp32 sums
    const int n = nb * 128 + 4 * lane;
    float sv[4];
    load4h(svh_d + (size_t)e * N + n, sv);
    for (int r = warp; r < vrows; r += 8) {
        const float4 u = *reinterpret_cast<const float4*>(stg + r * kLds2 + 4 * lane);
        float v[4] = {u.x, u.y, u.z, u.w};
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = v[j] * HAD_SCALE * sv[j];
        *reinterpret_cast<float4*>(y + (size_t)spair[tile0 * 16 + r] * N + n) = make_float4(v[0], v[1], v[2], v[3]);
    }
}

template <int K2, int CB, int MT, int KPS, int ST>
__global__ void __launch_bounds__(256, 2) down2_kernel(const uint32_t* __restrict__ XS, const uint64_t* __restrict__ TP, const int4* __restrict__ items,
                                                       const int* __restrict__ nitems, const half* __restrict__ svh_d, const int* __restrict__ spair,
                                                       float* __restrict__ y, int K, int N) {
    const int it = blockIdx.y;
    if (it >= nitems[0]) return;
    down2_item<K2, CB, MT, KPS, ST>(items[it], XS, TP, svh_d, spair, y, K, N);
}

// The unified list's launch (see gateup2u_kernel).
template <int K2, int CB, int KPS>
__global__ void __launch_bounds__(256, 2) down2u_kernel(const uint32_t* __restrict__ XS, const uint64_t* __restrict__ TP, const int4* __restrict__ items,
                                                        const int* __restrict__ nitems, const half* __restrict__ svh_d, const int* __restrict__ spair,
                                                        float* __restrict__ y, int K, int N) {
    const int it = blockIdx.y;
    if (it >= nitems[0]) return;
    const int4 item = items[it];
    if (item.z == 1)
        down2_item<K2, CB, 1, KPS, dn_st(1)>(item, XS, TP, svh_d, spair, y, K, N);
    else if (item.z == 2)
        down2_item<K2, CB, 2, KPS, dn_st(2)>(item, XS, TP, svh_d, spair, y, K, N);
    else
        down2_item<K2, CB, 4, KPS, dn_st(4)>(item, XS, TP, svh_d, spair, y, K, N);
}
