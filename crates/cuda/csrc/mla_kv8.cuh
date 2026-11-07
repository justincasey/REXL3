// The fp8 latent page of GLM's MLA layers (`--kv-dtype fp8`, `REXL3_GLM_KV=fp8`), shared by the latent write and the attention kernels
// (mla.cu: decode, verify, dense, sparse, head-split; mla_prefill.cu: the prefill slab).
//
// Page layout (one layer, 256 tokens): E4M3 values `[256 tokens][512]` (131,072 bytes), then one fp32 scale per token `[256]` (1,024 bytes):
// 132,096 bytes a page against 262,144 for fp16. A value is `e4m3(q) * s`.
//   * per token: a row's bytes depend on its own latent only, so a token written by a decode step, a verify window or any prefill slice is the
//     same bytes, and every reader (decode, verify, prefill, a restored prefix) reads the same quantised cache;
//   * the scale is `amax / 448` (round to nearest even, fp32), the value `rn_e4m3_satfinite(x / s)` (x / s rounded once to fp32, then to E4M3
//     nearest-even, saturating at +-448); a row of zeros gets s = 0 and zero bytes. Deterministic: the maximum does not depend on the order;
//   * the readers dequantise into the very T (fp16 or bf16) tiles the fp16 kernels stage: `round_T(float(e4m3) * s)` (one fp32 product, then
//     one rounding to T), so everything after the load is the fp16 kernel's arithmetic and the fp8 kernel's output is the fp16 kernel's on the
//     dequantised cache, bit for bit (crates/cuda/tests/mla_kv8.rs).
// The CPU mirror is `gb_cuda::mla::kv8` (quantise, dequantise, the page addressing).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include <type_traits>

namespace mla8 {

constexpr int PAGE = 256, DC = 512;
constexpr int64_t SCALE_OFF = (int64_t)PAGE * DC;      // the scales follow the values
constexpr int64_t PAGE_BYTES = SCALE_OFF + PAGE * 4;   // 132,096 (a multiple of 16)
constexpr float E4M3_MAX = 448.0f;
static_assert(PAGE_BYTES % 16 == 0, "pages stay 16-byte aligned");

// The values of (block, token in block), and its scale.
__device__ __forceinline__ const uint8_t* row(const uint8_t* c, int blk, int tok) { return c + (int64_t)blk * PAGE_BYTES + (int64_t)tok * DC; }
__device__ __forceinline__ uint8_t* row_mut(uint8_t* c, int blk, int tok) { return c + (int64_t)blk * PAGE_BYTES + (int64_t)tok * DC; }
__device__ __forceinline__ float scale(const uint8_t* c, int blk, int tok) {
    return *reinterpret_cast<const float*>(c + (int64_t)blk * PAGE_BYTES + SCALE_OFF + (int64_t)tok * 4);
}
__device__ __forceinline__ float* scale_mut(uint8_t* c, int blk, int tok) {
    return reinterpret_cast<float*>(c + (int64_t)blk * PAGE_BYTES + SCALE_OFF + (int64_t)tok * 4);
}

// Two E4M3 bytes (low byte first) times s as a packed T pair (low half first): round_T(float(e4m3) * s).
template <typename T>
__device__ __forceinline__ uint32_t dq2(uint32_t two, float s) {
    const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(two & 0xffffu), __NV_E4M3);
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));  // exact: every E4M3 value is an fp16 value
    const float a = __fmul_rn(f.x, s), b = __fmul_rn(f.y, s);
    if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
        return *reinterpret_cast<uint32_t*>(&v);
    } else {
        __half2 v = __floats2half2_rn(a, b);
        return *reinterpret_cast<uint32_t*>(&v);
    }
}

// Eight E4M3 bytes as eight T values (one 16-byte smem piece of the fp16 kernels' tiles).
template <typename T>
__device__ __forceinline__ uint4 dq8(uint2 raw, float s) {
    return make_uint4(dq2<T>(raw.x, s), dq2<T>(raw.x >> 16, s), dq2<T>(raw.y, s), dq2<T>(raw.y >> 16, s));
}

// Two values at scale s as two E4M3 bytes (low byte first). s > 0.
__device__ __forceinline__ uint32_t q2(float a, float b, float s) {
    return (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(__fdiv_rn(a, s), __fdiv_rn(b, s)), __NV_SATFINITE, __NV_E4M3);
}

}  // namespace mla8
