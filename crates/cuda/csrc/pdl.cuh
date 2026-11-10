// Programmatic dependent launch (PDL) for the GLM decode chain (lane E). Only TIMING changes, never a bit:
//
//   * A kernel launched with cudaLaunchAttributeProgrammaticStreamSerialization (launch_pdl below, pdl = true) may START while its
//     predecessor in the stream is still running.
//   * Before pdl_wait() it may do only what needs nothing the predecessor writes: weight loads, a prefetch of weights into L2.
//     pdl_wait() returns once the predecessor has COMPLETED and its writes are visible: every read of a predecessor's output, and every
//     write (a predecessor or an earlier kernel may still be reading the buffer), comes after it.
//   * pdl_trigger() lets the successor launch early. It still waits (pdl_wait) for this grid to finish before it touches anything. A
//     grid's successor launches when every block has triggered or exited, so every block of this grid is already running by then.
//   * Both are no-ops in a kernel launched without the attribute, so a kernel with these calls is correct under any launch.
//   * A kernel launched WITH the attribute must call pdl_wait before reading its predecessor's output. Add the attribute to a kernel's
//     launch only when the kernel does.
//
// The mask (env REXL3_GLM_PDL, read once; the test hook gb_pdl_set_mask): which kernel families are launched with the attribute.
// Default 0 (every launch as before) until the in-model ABBA decides.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

namespace gbp_pdl {

enum : int { kExl3 = 1, kMhc = 2, kMoe = 4, kFlash = 8, kHcx = 16, kAll = 31 };

inline int g_mask = -1;
inline int mask() {
    if (g_mask < 0) {
        const char* e = getenv("REXL3_GLM_PDL");
        g_mask = e ? (atoi(e) & kAll) : 0;
    }
    return g_mask;
}
inline bool on(int family) { return (mask() & family) != 0; }

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

// A launch with (pdl) or without the programmatic-stream-serialization attribute.
template <typename... KArgs, typename... Args>
int launch_pdl(bool pdl, void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    const cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...);
    return e != cudaSuccess ? (int)e : (int)cudaGetLastError();
}

}  // namespace gbp_pdl
