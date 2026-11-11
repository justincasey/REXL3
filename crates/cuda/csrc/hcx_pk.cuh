// Packed hyper-connection mixer weights (included inside namespace hcx of hcx.cu). Original REXL3 code.
//
// The mixer weights are bf16 numbers the loader rounded to fp16, so every weight is a float whose low 16 bits are zero once widened:
// bf16 = [sign][exponent 8][mantissa 7]. Weight bits split as
//   L  = [exponent bit 0][mantissa 7]  one raw byte,
//   c5 = [sign][c4]                    five bits, c4 = 0 for a zero and eh - eh0 + 1 for the exponent pair eh = exponent >> 1 (a tensor's
//                                      pairs span at most 15 values: fp16 flushes what is smaller than 2^-24),
// 13 bits a weight, 18.75% fewer bytes than fp16 (the exact bytes the fp16 kernels read, decoded back: the decode below is checked
// against every weight of every tensor at load, and a tensor that fails keeps the fp16 path).
//
// Decode of two weights a, b at once: a 1024-entry table in shared memory is indexed by their ten code bits c5a | c5b << 5 and returns
// (Hb_b << 24) | (Hb_a << 8), Hb = sign << 7 | eh the high byte of the bf16; or-ing the two raw bytes in (byte 0 and byte 2) gives the
// word [bf16 b][bf16 a], and the floats are word << 16 and word & 0xffff0000: exactly the widened fp16 weights.
//
// Layout of the down / inject rows (k1p): a job is one stream s and PK_NJ = 4 consecutive rows of one tensor, 33 slots of 16 bytes a lane
// (slot-major: slot k of lane l at (k * 32 + l) * 16 from the job's start, so every slot is one 512-byte coalesced read of the warp):
//   slots 5j + p (j < 4 rows, p < 5 chunk pairs): lane's L bytes of row j, chunks 2p (bytes 0..7) and 2p + 1 (bytes 8..15),
//   slots 20..32: the lane's code stream, bits 320p + 80j + 40c + 5b = code of weight b of chunk c of pair p of row j.
// Jobs of a tensor are ordered s * (rows / 4) + g.
constexpr int PK_NJ = 4;
constexpr int PK_NP = 5;                          // chunk pairs a row (hidden size 2,560 = 10 chunks of 256)
constexpr int PK_LS = PK_NJ * PK_NP;              // L slots of a job
constexpr int PK_CS = (40 * PK_NP + 15) / 16;     // code slots of a job (13)
constexpr int PK_SL = PK_LS + PK_CS;              // 33
constexpr int PK_CW = PK_CS * 4;                  // code words of a lane
constexpr size_t PK_JOB_BYTES = (size_t)PK_SL * 512;

// The ten code bits at stream bit `bit`: a lane's 32-bit words Cw.
template <int N>
__device__ __forceinline__ uint32_t pk_field(const uint32_t (&Cw)[N], int bit) {
    const int w = bit >> 5, sh = bit & 31;
    return (sh <= 22 ? (Cw[w] >> sh) : __funnelshift_r(Cw[w], Cw[w + 1], sh)) & 0x3ffu;
}

// Eight weights of one chunk: l0 / l1 the raw bytes of weights 0..3 / 4..7, `bit` the chunk's first code bit in Cw.
template <int N>
__device__ __forceinline__ void pk_dec8(uint32_t l0, uint32_t l1, const uint32_t (&Cw)[N], int bit, const uint32_t* lut,
                                        float (&v)[8]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const uint32_t src = q < 2 ? l0 : l1;
        const uint32_t spread = __byte_perm(src, 0u, (q & 1) ? 0x4342u : 0x4140u);
        const uint32_t p = lut[pk_field(Cw, bit + 10 * q)] | spread;
        v[2 * q] = __uint_as_float(p << 16);
        v[2 * q + 1] = __uint_as_float(p & 0xffff0000u);
    }
}

// Table entry of the code pair idx for a tensor whose smallest exponent pair is eh0.
__device__ __forceinline__ uint32_t pk_lut_entry(int idx, int eh0) {
    const int ca = idx & 31, cb = idx >> 5;
    const uint32_t ha = ((uint32_t)(ca >> 4) << 7) | ((ca & 15) ? (uint32_t)(eh0 + (ca & 15) - 1) : 0u);
    const uint32_t hb = ((uint32_t)(cb >> 4) << 7) | ((cb & 15) ? (uint32_t)(eh0 + (cb & 15) - 1) : 0u);
    return (hb << 24) | (ha << 8);
}

// The table of a tensor as the packer wrote it after the tensor's jobs (1,024 words), copied into shared memory by `threads` threads.
__device__ __forceinline__ void pk_copy_lut(uint32_t* lut, const uint4* __restrict__ g, int tid, int threads) {
    for (int i = tid; i < 256; i += threads) reinterpret_cast<uint4*>(lut)[i] = __ldg(g + i);
}

// A job's slots into registers (the weights are constants of the call, so this may run before the predecessor has finished).
__device__ __forceinline__ void pk_load_job(const uint4* __restrict__ jb, uint4 (&Ls)[PK_LS], uint32_t (&Cw)[PK_CW]) {
#pragma unroll
    for (int k = 0; k < PK_LS; ++k) Ls[k] = __ldg(jb + k * 32);
#pragma unroll
    for (int k = 0; k < PK_CS; ++k) {
        const uint4 u = __ldg(jb + (PK_LS + k) * 32);
        Cw[4 * k] = u.x;
        Cw[4 * k + 1] = u.y;
        Cw[4 * k + 2] = u.z;
        Cw[4 * k + 3] = u.w;
    }
}

// k1p: k1u with the weights of PK_NJ rows a warp read from the packed copy, all of a warp's 33 slots requested up front. Every sum is the
// one k1u makes (the decoded floats are the fp16 weights widened, the add order per (row, window row) is the chunk order of lane l).
// rows of `pdown` are down rows [0, rank), of `pinj` the inject rows; rank % 4 == 0 and ninj in {0, 4}.
template <int NT, bool POST>
__global__ void __launch_bounds__(32, 16) k1p_kernel(const float* __restrict__ x, const half* __restrict__ nw,
                                                    const uint4* __restrict__ pdown, const uint4* __restrict__ pinj,
                                                    float* __restrict__ P, float* __restrict__ ss, int t0, int H, int hc,
                                                    int rank, int ninj,
                                                    const __nv_bfloat16* __restrict__ bo, const float* __restrict__ ijp) {
    __shared__ __align__(16) uint32_t lut[1024];
    const int lane = threadIdx.x;
    const int R = rank + ninj;
    const int gps = (R + PK_NJ - 1) / PK_NJ;  // row groups a stream
    const size_t HC = (size_t)hc * H;
    if ((int)blockIdx.x < gps * hc) {
        const int s = blockIdx.x / gps, g = blockIdx.x % gps;
        const bool isinj = g * PK_NJ >= rank;
        const uint4* jb = (isinj ? pinj + (size_t)s * (PK_SL * 32) : pdown + ((size_t)s * (rank / PK_NJ) + g) * (PK_SL * 32)) + lane;
        uint4 Ls[PK_LS];
        uint32_t Cw[PK_CW];
        pk_load_job(jb, Ls, Cw);
        pk_copy_lut(lut, isinj ? pinj + (size_t)hc * (PK_SL * 32) : pdown + (size_t)hc * (rank / PK_NJ) * (PK_SL * 32), lane, 32);
        __syncwarp();
        const half* n = nw + (size_t)s * H;
        const float* xs = x + (size_t)t0 * HC + (size_t)s * H;
        float acc[PK_NJ][NT];
#pragma unroll
        for (int j = 0; j < PK_NJ; ++j)
#pragma unroll
            for (int t = 0; t < NT; ++t) acc[j][t] = 0.f;
        gbp_pdl::pdl_trigger();
        gbp_pdl::pdl_wait();  // x (and P/ss, last read by the previous call's k2) belong to earlier kernels of the stream
        float ij[POST ? NT : 1];
        if constexpr (POST) {
#pragma unroll
            for (int t = 0; t < NT; ++t) ij[t] = ijp[(size_t)(t0 + t) * hc + s];
        }
#pragma unroll
        for (int p = 0; p < PK_NP; ++p) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int d = 8 * lane + 256 * (2 * p + c);
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
                float nv[8];
                h8(n + d, nv);
                float onep[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) onep[i] = 1.f + nv[i];
#pragma unroll
                for (int j = 0; j < PK_NJ; ++j) {
                    float wv[8];
                    const uint4 sl = Ls[PK_NP * j + p];
                    pk_dec8(c ? sl.z : sl.x, c ? sl.w : sl.y, Cw, 320 * p + 80 * j + 40 * c, lut, wv);
#pragma unroll
                    for (int i = 0; i < 8; ++i) wv[i] *= onep[i];
#pragma unroll
                    for (int t = 0; t < NT; ++t) acc[j][t] = dot8_acc(acc[j][t], wv, xa[t], xb[t]);
                }
            }
        }
        constexpr int NEL = PK_NJ * NT;
#pragma unroll
        for (int c = 0; c < (NEL + 31) / 32; ++c) {
            float v[32];
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                const int e = c * 32 + k;
                v[k] = e < NEL ? acc[(e / NT) % PK_NJ][e % NT] : 0.f;
            }
            const float z = scatter_sum32(v, lane);
            const int e = c * 32 + lane;
            if (e < NEL) P[((size_t)(e % NT) * R + g * PK_NJ + e / NT) * hc + s] = z;
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

// ---- packing at load: scan, pack, check ------------------------------------------------------------------------------------------
// st[0] = smallest and st[1] = largest exponent pair (exponent >> 1) over the non-zero floats of an fp16 tensor, st[2] = how many values
// the format cannot hold (more than 7 mantissa bits once widened, an infinity or a NaN). The caller initialises st to {INT_MAX, INT_MIN, 0}.
__global__ void __launch_bounds__(256) pk_scan_kernel(const half* __restrict__ w, size_t n, int* st) {
    int mn = 0x7fffffff, mx = -0x7fffffff - 1, bad = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint32_t b = __float_as_uint(__half2float(w[i]));
        if ((b & 0x7fffffffu) == 0) continue;
        const int e = (int)((b >> 23) & 0xffu);
        if (e == 0xff || (b & 0xffffu)) {
            ++bad;
            continue;
        }
        mn = min(mn, e >> 1);
        mx = max(mx, e >> 1);
    }
    atomicMin(&st[0], mn);
    atomicMax(&st[1], mx);
    atomicAdd(&st[2], bad);
}

// One warp a job of a [rows][4 * H] fp16 tensor (H = 2,560): see the layout above. grid = jobs = 4 * rows / PK_NJ.
__global__ void __launch_bounds__(32) pk1_pack_kernel(const half* __restrict__ w, int rows, int H, int eh0, uint4* __restrict__ out) {
    const int gpt = rows / PK_NJ;
    const int job = blockIdx.x, lane = threadIdx.x, s = job / gpt, g = job % gpt;
    const size_t HC = (size_t)4 * H;
    uint32_t Lw[PK_LS][4], Cw[PK_CW];
    for (int k = 0; k < PK_LS; ++k) Lw[k][0] = Lw[k][1] = Lw[k][2] = Lw[k][3] = 0;
    for (int k = 0; k < PK_CW; ++k) Cw[k] = 0;
    for (int j = 0; j < PK_NJ; ++j)
        for (int p = 0; p < PK_NP; ++p)
            for (int c = 0; c < 2; ++c)
                for (int b = 0; b < 8; ++b) {
                    const size_t col = (size_t)s * H + 256 * (2 * p + c) + 8 * lane + b;
                    const uint32_t fb = __float_as_uint(__half2float(w[(size_t)(g * PK_NJ + j) * HC + col]));
                    const bool zero = (fb & 0x7fffffffu) == 0;
                    const uint32_t e = (fb >> 23) & 0xffu;
                    const uint32_t L = zero ? 0u : (((e & 1u) << 7) | ((fb >> 16) & 0x7fu));
                    const uint32_t c4 = zero ? 0u : (uint32_t)((int)(e >> 1) - eh0 + 1);
                    const uint32_t code = ((fb >> 31) << 4) | c4;
                    const int lb = c * 8 + b;
                    Lw[PK_NP * j + p][lb >> 2] |= L << (8 * (lb & 3));
                    const int bit = 320 * p + 80 * j + 40 * c + 5 * b;
                    Cw[bit >> 5] |= code << (bit & 31);
                    if ((bit & 31) > 27) Cw[(bit >> 5) + 1] |= code >> (32 - (bit & 31));
                }
    uint4* o = out + (size_t)job * (PK_SL * 32) + lane;
    for (int k = 0; k < PK_LS; ++k) o[k * 32] = make_uint4(Lw[k][0], Lw[k][1], Lw[k][2], Lw[k][3]);
    for (int k = 0; k < PK_CS; ++k) o[(PK_LS + k) * 32] = make_uint4(Cw[4 * k], Cw[4 * k + 1], Cw[4 * k + 2], Cw[4 * k + 3]);
}

// *bad += the values of a packed tensor that do not decode to the bits of the fp16 weight widened (the decode k1p runs).
__global__ void __launch_bounds__(32) pk1_check_kernel(const half* __restrict__ w, const uint4* __restrict__ packed, int rows, int H,
                                                       unsigned long long* bad) {
    __shared__ __align__(16) uint32_t lut[1024];
    const int gpt = rows / PK_NJ;
    const int job = blockIdx.x, lane = threadIdx.x, s = job / gpt, g = job % gpt;
    const size_t HC = (size_t)4 * H;
    uint4 Ls[PK_LS];
    uint32_t Cw[PK_CW];
    pk_load_job(packed + (size_t)job * (PK_SL * 32) + lane, Ls, Cw);
    pk_copy_lut(lut, packed + (size_t)4 * gpt * (PK_SL * 32), lane, 32);
    __syncwarp();
    int miss = 0;
#pragma unroll
    for (int j = 0; j < PK_NJ; ++j)
#pragma unroll
        for (int p = 0; p < PK_NP; ++p)
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                float v[8];
                const uint4 sl = Ls[PK_NP * j + p];
                pk_dec8(c ? sl.z : sl.x, c ? sl.w : sl.y, Cw, 320 * p + 80 * j + 40 * c, lut, v);
                const half* src = w + (size_t)(g * PK_NJ + j) * HC + (size_t)s * H + 256 * (2 * p + c) + 8 * lane;
#pragma unroll
                for (int b = 0; b < 8; ++b) miss += __float_as_uint(v[b]) != __float_as_uint(__half2float(src[b]));
            }
    if (miss) atomicAdd(bad, (unsigned long long)miss);
}

// The table of a tensor: 1,024 words, written after its jobs. eh0 the tensor's smallest exponent pair.
__global__ void __launch_bounds__(256) pk_lut_kernel(int eh0, uint32_t* __restrict__ out) {
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) out[i] = pk_lut_entry(i, eh0);
}

// ---- the up rows (k2tp) ----------------------------------------------------------------------------------------------------------
// A job is one warp's four columns of k2t: the rows rho = 4k + s (k < 4 columns, s < 4 streams) of up, row (s * H + col0 + k) of 320 values.
// Lane l owns values 8l..8l+7 (chunk 0, every lane) and 256 + 8l..256 + 8l + 7 (chunk 1, lanes < 8). A chunk's data of the 16 rows is 13 slots
// of 16 bytes a lane: slots 0..7 the raw L bytes (row rho at slot rho / 2, bytes 8 (rho & 1) .. +7), slots 8..12 the code stream (bits
// 40 rho + 5 b = code of weight b of row rho; row group k = rho / 4 owns the words 5k..5k+4). Chunk 0 is stored slot-major for 32 lanes
// (slot sigma of lane l at (sigma * 32 + l) * 16 from the job's start), chunk 1 after it for 8 lanes ((sigma * 8 + l) * 16 from 6,656): a job
// is 8,320 bytes, 13 / 16 of the fp16 rows (10,240).
constexpr int PK2_JOB_U4 = 520;     // uint4 a job
constexpr size_t PK2_JOB_BYTES = 8320;

// Column K's weights (four streams, one chunk) from the lane's slots: Lq the two L slots (rows 4K..4K+3), Cq the two code slots that
// hold words 5K..5K+4 of the code stream.
template <int K>
__device__ __forceinline__ void pk2_dec_col(const uint4 (&Lq)[2], const uint4 (&Cq)[2], const uint32_t* lut, float (&wv)[4][8]) {
    uint32_t Cw[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        const int w = 5 * K + i;
        const uint4& u = Cq[(w >> 2) - ((5 * K) >> 2)];
        Cw[i] = (w & 3) == 0 ? u.x : (w & 3) == 1 ? u.y : (w & 3) == 2 ? u.z : u.w;
    }
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const uint4 sl = Lq[s >> 1];
        pk_dec8((s & 1) ? sl.z : sl.x, (s & 1) ? sl.w : sl.y, Cw, 40 * s, lut, wv[s]);
    }
}

// Column K's four slots of a chunk (stride: 32 for chunk 0, 8 for chunk 1).
template <int K>
__device__ __forceinline__ void pk2_load_col(const uint4* __restrict__ jb, int stride, uint4 (&Lq)[2], uint4 (&Cq)[2]) {
    Lq[0] = __ldg(jb + (2 * K) * stride);
    Lq[1] = __ldg(jb + (2 * K + 1) * stride);
    Cq[0] = __ldg(jb + (8 + ((5 * K) >> 2)) * stride);
    Cq[1] = __ldg(jb + (8 + ((5 * K) >> 2) + 1) * stride);
}

// One warp a job of up [4 * H rows][rank = 320] fp16 (H = 2,560): job wg holds the columns 4 wg .. 4 wg + 3. grid = H / 4 warps.
__global__ void __launch_bounds__(32) pk2_pack_kernel(const half* __restrict__ w, int H, int eh0, uint4* __restrict__ out) {
    const int wg = blockIdx.x, lane = threadIdx.x;
    uint4* jb = out + (size_t)wg * PK2_JOB_U4;
    for (int c = 0; c < 2; ++c) {
        if (c == 1 && lane >= 8) break;
        uint32_t Lw[8][4], Cw[20];
        for (int k = 0; k < 8; ++k) Lw[k][0] = Lw[k][1] = Lw[k][2] = Lw[k][3] = 0;
        for (int k = 0; k < 20; ++k) Cw[k] = 0;
        for (int rho = 0; rho < 16; ++rho) {
            const int k = rho >> 2, s = rho & 3;
            const half* row = w + ((size_t)s * H + 4 * wg + k) * 320 + 256 * c + 8 * lane;
            for (int b = 0; b < 8; ++b) {
                const uint32_t fb = __float_as_uint(__half2float(row[b]));
                const bool zero = (fb & 0x7fffffffu) == 0;
                const uint32_t e = (fb >> 23) & 0xffu;
                const uint32_t L = zero ? 0u : (((e & 1u) << 7) | ((fb >> 16) & 0x7fu));
                const uint32_t c4 = zero ? 0u : (uint32_t)((int)(e >> 1) - eh0 + 1);
                const uint32_t code = ((fb >> 31) << 4) | c4;
                const int lb = 8 * (rho & 1) + b;
                Lw[rho >> 1][lb >> 2] |= L << (8 * (lb & 3));
                const int bit = 40 * rho + 5 * b;
                Cw[bit >> 5] |= code << (bit & 31);
                if ((bit & 31) > 27) Cw[(bit >> 5) + 1] |= code >> (32 - (bit & 31));
            }
        }
        uint4* o = jb + (c == 0 ? lane : 416 + lane);
        const int stride = c == 0 ? 32 : 8;
        for (int k = 0; k < 8; ++k) o[k * stride] = make_uint4(Lw[k][0], Lw[k][1], Lw[k][2], Lw[k][3]);
        for (int k = 0; k < 5; ++k) o[(8 + k) * stride] = make_uint4(Cw[4 * k], Cw[4 * k + 1], Cw[4 * k + 2], Cw[4 * k + 3]);
    }
}

// *bad += the values of the packed up rows that do not decode to the bits of the fp16 weight widened (the decode k2tp runs).
__global__ void __launch_bounds__(32) pk2_check_kernel(const half* __restrict__ w, const uint4* __restrict__ packed, int H,
                                                       unsigned long long* bad) {
    __shared__ __align__(16) uint32_t lut[1024];
    const int wg = blockIdx.x, lane = threadIdx.x;
    pk_copy_lut(lut, packed + (size_t)(H / 4) * PK2_JOB_U4, lane, 32);
    __syncwarp();
    int miss = 0;
    for (int c = 0; c < 2; ++c) {
        if (c == 1 && lane >= 8) break;
        const uint4* jb = packed + (size_t)wg * PK2_JOB_U4 + (c == 0 ? lane : 416 + lane);
        const int stride = c == 0 ? 32 : 8;
#define PK2_CHECK_COL(K)                                                                                               \
    {                                                                                                                  \
        uint4 Lq[2], Cq[2];                                                                                            \
        float wv[4][8];                                                                                                \
        pk2_load_col<K>(jb, stride, Lq, Cq);                                                                           \
        pk2_dec_col<K>(Lq, Cq, lut, wv);                                                                               \
        for (int s = 0; s < 4; ++s) {                                                                                  \
            const half* row = w + ((size_t)s * H + 4 * wg + K) * 320 + 256 * c + 8 * lane;                             \
            for (int b = 0; b < 8; ++b) miss += __float_as_uint(wv[s][b]) != __float_as_uint(__half2float(row[b]));    \
        }                                                                                                              \
    }
        PK2_CHECK_COL(0) PK2_CHECK_COL(1) PK2_CHECK_COL(2) PK2_CHECK_COL(3)
#undef PK2_CHECK_COL
    }
    if (miss) atomicAdd(bad, (unsigned long long)miss);
}

// k2tp: k2t (hc = 4, rank = 320, H = 2,560) with the up rows of a warp's four columns read from the packed copy: the warp's first column is
// requested before the wait for k1, the whole job (8,320 bytes) is bulk-prefetched into L2 right after it, and a column's slots are loaded
// at the top of its iteration. Every sum, and the epilogue, are k2t's.
template <int NT, bool POST>
__global__ void __launch_bounds__(K2T_WPB * 32, 2) k2tp_kernel(float* x, const half* __restrict__ nw, const uint4* __restrict__ pup,
                                                               const float* __restrict__ P, const float* __restrict__ ss,
                                                               __nv_bfloat16* __restrict__ y, float* __restrict__ inj_out, int t0,
                                                               int H, int rank, int ninj, float eps, int pf,
                                                               const __nv_bfloat16* __restrict__ bo,
                                                               const float* __restrict__ ijp) {
    constexpr int HCN = 4;
    extern __shared__ float s_low[];  // swizzled low [NT][rank]
    __shared__ float s_rms[NT * HCN];
    __shared__ __align__(16) uint32_t lut[1024];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int R = rank + ninj;
    const float inv_hc = 1.f / (float)HCN;
    const int wg = blockIdx.x * K2T_WPB + warp;
    const int col0 = wg * K2T_CPW;
    const size_t HC = (size_t)HCN * H;
    const bool has1 = lane < 8;  // rank 320: chunk 1 holds 64 values
    const int base1 = NT * 2 * 128;
    const int stride1 = 32;  // rank <= 320
    const uint4* j0 = pup + (size_t)wg * PK2_JOB_U4 + lane;
    const uint4* j1 = pup + (size_t)wg * PK2_JOB_U4 + 416 + lane;

    gbp_pdl::pdl_trigger();  // a successor may launch now; it waits for this grid before touching anything
    uint4 La[2], Ca[2], Lb[2], Cb[2];
    pk2_load_col<0>(j0, 32, La, Ca);
    if (has1) pk2_load_col<0>(j1, 8, Lb, Cb);
    pk_copy_lut(lut, pup + (size_t)(H / K2T_CPW) * PK2_JOB_U4, tid, K2T_WPB * 32);
    gbp_pdl::pdl_wait();  // P and ss are k1's output (and y / inj_out were last read by earlier kernels)
    if (pf && lane == 0) k2t_prefetch_l2(pup + (size_t)wg * PK2_JOB_U4, (unsigned)PK2_JOB_BYTES);
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
#define K2TP_COL(K)                                                                                                    \
    {                                                                                                                  \
        const int col = col0 + K;                                                                                      \
        if (K > 0) {                                                                                                   \
            pk2_load_col<K>(j0, 32, La, Ca);                                                                           \
            if (has1) pk2_load_col<K>(j1, 8, Lb, Cb);                                                                  \
        }                                                                                                              \
        float acc[HCN][NT];                                                                                            \
        _Pragma("unroll") for (int s = 0; s < HCN; ++s) _Pragma("unroll") for (int t = 0; t < NT; ++t) acc[s][t] = 0.f; \
        {                                                                                                              \
            float wv[HCN][8];                                                                                          \
            pk2_dec_col<K>(La, Ca, lut, wv);                                                                           \
            _Pragma("unroll") for (int t = 0; t < NT; ++t) {                                                           \
                const float4 a = *reinterpret_cast<const float4*>(s_low + (t * 2 + 0) * 128 + lane * 4);               \
                const float4 b = *reinterpret_cast<const float4*>(s_low + (t * 2 + 1) * 128 + lane * 4);               \
                _Pragma("unroll") for (int s = 0; s < HCN; ++s) acc[s][t] = dot8_acc(acc[s][t], wv[s], a, b);          \
            }                                                                                                          \
        }                                                                                                              \
        if (has1) {                                                                                                    \
            float wv[HCN][8];                                                                                          \
            pk2_dec_col<K>(Lb, Cb, lut, wv);                                                                           \
            _Pragma("unroll") for (int t = 0; t < NT; ++t) {                                                           \
                const float4 a = *reinterpret_cast<const float4*>(s_low + base1 + (t * 2 + 0) * stride1 + lane * 4);   \
                const float4 b = *reinterpret_cast<const float4*>(s_low + base1 + (t * 2 + 1) * stride1 + lane * 4);   \
                _Pragma("unroll") for (int s = 0; s < HCN; ++s) acc[s][t] = dot8_acc(acc[s][t], wv[s], a, b);          \
            }                                                                                                          \
        }                                                                                                              \
        float v[32];                                                                                                   \
        _Pragma("unroll") for (int s = 0; s < HCN; ++s) _Pragma("unroll") for (int t = 0; t < 8; ++t)                  \
            v[s * 8 + t] = (t < NT) ? acc[s][t < NT ? t : 0] : 0.f;                                                    \
        const float z = scatter_sum32(v, lane);                                                                        \
        float term = 0.f;                                                                                              \
        if (te < NT) {                                                                                                 \
            const float nv = 1.f + __half2float(nw[(size_t)se * H + col]);                                             \
            float xv = x[(size_t)(t0 + te) * HC + (size_t)se * H + col];                                               \
            if constexpr (POST) {                                                                                      \
                xv = fmaf(__bfloat162float(bo[(size_t)(t0 + te) * H + col]), ijp[(size_t)(t0 + te) * HCN + se], xv);   \
                x[(size_t)(t0 + te) * HC + (size_t)se * H + col] = xv;                                                 \
            }                                                                                                          \
            term = xv * s_rms[te * HCN + se] * nv / (1.f + __expf(-z));                                                \
        }                                                                                                              \
        const float t1 = __shfl_sync(0xffffffffu, term, (lane + 8) & 31);                                              \
        const float t2 = __shfl_sync(0xffffffffu, term, (lane + 16) & 31);                                             \
        const float t3 = __shfl_sync(0xffffffffu, term, (lane + 24) & 31);                                             \
        float o = 0.f;                                                                                                 \
        o += term;                                                                                                     \
        o += t1;                                                                                                       \
        o += t2;                                                                                                       \
        o += t3;                                                                                                       \
        if (lane < NT) y[(size_t)(t0 + lane) * H + col] = __float2bfloat16(o * inv_hc);                                \
    }
    K2TP_COL(0) K2TP_COL(1) K2TP_COL(2) K2TP_COL(3)
#undef K2TP_COL
}
