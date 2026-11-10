// Per-warp shared-memory ring of bulk copies (cp.async.bulk + mbarrier) for the decode linear's trellis words. A warp's k-step is
// 8 tiles stored back to back (128 * K2 contiguous bytes), so one bulk copy brings a whole step; the warp keeps several steps in
// flight with no registers spent on them. Every barrier belongs to one warp (lane 0 arms and issues, all lanes wait), so the ring
// needs no block-wide synchronization. Only timing depends on any of this: the words a lane decodes are the ones it loaded before.

#pragma once

#include <cstdint>

namespace gbx {

__device__ __forceinline__ uint32_t smem_addr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

// the barrier's next phase completes once one arrival (this one) and `bytes` of bulk copies have landed
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra LAB_WAIT;\n"
        "}\n" ::"r"(bar),
        "r"(parity)
        : "memory");
}

__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                 "l"(src), "r"(bytes), "r"(bar)
                 : "memory");
}

}  // namespace gbx
