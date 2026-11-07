//! Minimal CUDA runtime wrappers: errors, device buffers, streams.

use core::ffi::c_void;
use gb_exl3::loadstats::{self, Stage};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaError(pub i32);

impl CudaError {
    pub fn check(rc: i32) -> Result<(), CudaError> {
        if rc == 0 {
            Ok(())
        } else {
            Err(CudaError(rc))
        }
    }

    /// The error leaves the CUDA context unusable or the engine's device state unknown: no later call can be trusted and only a
    /// new process helps. Everything is fatal except the few codes below that spoil only the call that returned them, so a code
    /// this build has never seen fails closed.
    ///
    /// Retryable, by `cudaError_t` value (driver_types.h, CUDA 13.0): 1 `InvalidValue` and 9 `InvalidConfiguration` (a bad
    /// argument to one call), 2 `MemoryAllocation` (this allocation did not fit), 600 `NotReady` (not an error), 701
    /// `LaunchOutOfResources` (this launch shape did not fit). The sticky ones are named in the header: 226 `Contained`, 700
    /// `IllegalAddress`, 702 `LaunchTimeout`, 710 `Assert`, 714-719 (hardware stack, illegal instruction, misaligned address,
    /// invalid address space, invalid PC, launch failure), 721 `TensorMemoryLeak`, 810 and 911; they surface at the next
    /// synchronising call or copy rather than at the faulting launch, which is why this classifies by code, not by call site.
    pub fn is_fatal(&self) -> bool {
        !matches!(self.0, 1 | 2 | 9 | 600 | 701)
    }

    /// `cudaErrorMemoryAllocation`: the device would not give the memory asked for (the process goes on; at boot the caller names what it was
    /// allocating and what the host had, see `gb_model::hostmem::allocation_refusal`).
    pub fn is_out_of_memory(&self) -> bool {
        self.0 == 2
    }

    /// The `cudaError_t` name of the codes this engine can meet in practice, for logs. The codes the NCCL binding makes
    /// (`crate::nccl::NCCL_ERROR_BASE` and up: a collective's result, an aborted communicator, a peer that failed) are named by
    /// [`crate::nccl::error_name`], so a collective's failure never reads "unrecognised".
    pub fn name(&self) -> &'static str {
        match self.0 {
            1 => "cudaErrorInvalidValue",
            2 => "cudaErrorMemoryAllocation",
            3 => "cudaErrorInitializationError",
            9 => "cudaErrorInvalidConfiguration",
            46 => "cudaErrorDevicesUnavailable",
            100 => "cudaErrorNoDevice",
            200 => "cudaErrorInvalidKernelImage",
            209 => "cudaErrorNoKernelImageForDevice",
            214 => "cudaErrorECCUncorrectable",
            226 => "cudaErrorContained",
            600 => "cudaErrorNotReady",
            700 => "cudaErrorIllegalAddress",
            701 => "cudaErrorLaunchOutOfResources",
            702 => "cudaErrorLaunchTimeout",
            709 => "cudaErrorContextIsDestroyed",
            710 => "cudaErrorAssert",
            714 => "cudaErrorHardwareStackError",
            715 => "cudaErrorIllegalInstruction",
            716 => "cudaErrorMisalignedAddress",
            717 => "cudaErrorInvalidAddressSpace",
            718 => "cudaErrorInvalidPc",
            719 => "cudaErrorLaunchFailure",
            721 => "cudaErrorTensorMemoryLeak",
            900 => "cudaErrorStreamCaptureUnsupported",
            901 => "cudaErrorStreamCaptureInvalidated",
            902 => "cudaErrorStreamCaptureMerge",
            903 => "cudaErrorStreamCaptureUnmatched",
            904 => "cudaErrorStreamCaptureUnjoined",
            905 => "cudaErrorStreamCaptureIsolation",
            906 => "cudaErrorStreamCaptureImplicit",
            908 => "cudaErrorStreamCaptureWrongThread",
            910 => "cudaErrorGraphExecUpdateFailure",
            999 => "cudaErrorUnknown",
            -1 => "no CUDA in this build",
            c => crate::nccl::error_name(c).unwrap_or("unrecognised CUDA error"),
        }
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA error {} ({})", self.0, self.name())
    }
}

impl std::error::Error for CudaError {}

#[cfg(not(gb_no_cuda))]
mod ffi {
    use core::ffi::c_void;
    extern "C" {
        pub fn cudaGetDeviceCount(n: *mut i32) -> i32;
        pub fn cudaProfilerStart() -> i32;
        pub fn cudaProfilerStop() -> i32;
        pub fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> i32;
        pub fn cudaFree(p: *mut c_void) -> i32;
        pub fn cudaHostAlloc(p: *mut *mut c_void, bytes: usize, flags: u32) -> i32;
        pub fn cudaFreeHost(p: *mut c_void) -> i32;
        pub fn cudaHostGetDevicePointer(dev: *mut *mut c_void, host: *mut c_void, flags: u32) -> i32;
        pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
        pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32, stream: *mut c_void) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn cudaMemcpy2DAsync(
            dst: *mut c_void,
            dpitch: usize,
            src: *const c_void,
            spitch: usize,
            width: usize,
            height: usize,
            kind: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn cudaMemset(p: *mut c_void, v: i32, bytes: usize) -> i32;
        pub fn cudaMemsetAsync(p: *mut c_void, v: i32, bytes: usize, stream: *mut c_void) -> i32;
        pub fn cudaStreamCreate(s: *mut *mut c_void) -> i32;
        pub fn cudaStreamCreateWithFlags(s: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaEventCreateWithFlags(e: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaEventRecord(e: *mut c_void, stream: *mut c_void) -> i32;
        pub fn cudaEventCreate(e: *mut *mut c_void) -> i32;
        pub fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> i32;
        pub fn cudaEventSynchronize(e: *mut c_void) -> i32;
        pub fn cudaEventDestroy(e: *mut c_void) -> i32;
        pub fn cudaStreamDestroy(s: *mut c_void) -> i32;
        pub fn cudaStreamSynchronize(s: *mut c_void) -> i32;
        pub fn cudaDeviceSynchronize() -> i32;
        pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
        pub fn cudaDeviceGetAttribute(v: *mut i32, attr: i32, dev: i32) -> i32;
        pub fn cudaRuntimeGetVersion(v: *mut i32) -> i32;
        pub fn cudaDriverGetVersion(v: *mut i32) -> i32;
        pub fn cublasGetProperty(kind: i32, v: *mut i32) -> i32;
    }
}

/// What this binary was built with and what it is running against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Versions {
    /// nvcc that compiled the kernels (`13.0.88`), or `none` for a stub build.
    pub nvcc: &'static str,
    /// `sm_121`.
    pub arch: &'static str,
    /// FNV-1a over every kernel source, the arch and the flags (16 hex digits).
    pub kernel_hash: &'static str,
    /// The loaded CUDA runtime (`13.0`), the driver's CUDA version (`13.0`) and the loaded cuBLAS (`13.1.0`); `None` when
    /// unavailable (stub build, no driver).
    pub cudart: Option<String>,
    pub driver: Option<String>,
    pub cublas: Option<String>,
}

/// Build-time and run-time versions. Needs no GPU: it only asks the loaded libraries.
pub fn versions() -> Versions {
    let mm = |v: i32| format!("{}.{}", v / 1000, (v % 1000) / 10);
    #[cfg(not(gb_no_cuda))]
    let (cudart, driver, cublas) = unsafe {
        let (mut r, mut d) = (0, 0);
        let cudart = (ffi::cudaRuntimeGetVersion(&mut r) == 0 && r > 0).then(|| mm(r));
        let driver = (ffi::cudaDriverGetVersion(&mut d) == 0 && d > 0).then(|| mm(d));
        // libraryPropertyType: MAJOR_VERSION 0, MINOR_VERSION 1, PATCH_LEVEL 2
        let (mut a, mut b, mut c) = (0, 0, 0);
        let ok = ffi::cublasGetProperty(0, &mut a) == 0
            && ffi::cublasGetProperty(1, &mut b) == 0
            && ffi::cublasGetProperty(2, &mut c) == 0;
        (cudart, driver, ok.then(|| format!("{a}.{b}.{c}")))
    };
    #[cfg(gb_no_cuda)]
    let (cudart, driver, cublas): (Option<String>, Option<String>, Option<String>) = {
        let _ = mm;
        (None, None, None)
    };
    Versions {
        nvcc: env!("REXL3_NVCC_VERSION"),
        arch: env!("REXL3_CUDA_ARCH"),
        kernel_hash: env!("REXL3_KERNEL_HASH"),
        cudart,
        driver,
        cublas,
    }
}

const H2D: i32 = 1;
const D2H: i32 = 2;
const D2D: i32 = 3;

pub fn device_available() -> bool {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        let mut n = 0;
        ffi::cudaGetDeviceCount(&mut n) == 0 && n > 0
    }
    #[cfg(gb_no_cuda)]
    false
}

/// Create the CUDA context now (the first runtime call otherwise pays for it inside the first allocation).
pub fn init_context() -> Result<(), CudaError> {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        // cudaFree(NULL) does no work beyond initialising the runtime and the primary context
        CudaError::check(ffi::cudaFree(core::ptr::null_mut()))
    }
    #[cfg(gb_no_cuda)]
    Err(CudaError(-1))
}

/// `cudaProfilerStart`: opens the capture range of `nsys profile --capture-range=cudaProfilerApi` (a no-op without nvcc, so tests that mark a range still build in CI).
pub fn profiler_start() {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        ffi::cudaProfilerStart();
    }
}

/// `cudaProfilerStop`: closes the capture range opened by [`profiler_start`].
pub fn profiler_stop() {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        ffi::cudaProfilerStop();
    }
}

/// Whether this build has the CUDA kernels (false in a stub build without nvcc: every launcher fails, and a size query returns 0).
pub const fn kernels_built() -> bool {
    cfg!(not(gb_no_cuda))
}

/// (free, total) device memory in bytes.
pub fn mem_info() -> Result<(usize, usize), CudaError> {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        let (mut f, mut t) = (0usize, 0usize);
        CudaError::check(ffi::cudaMemGetInfo(&mut f, &mut t))?;
        Ok((f, t))
    }
    #[cfg(gb_no_cuda)]
    Err(CudaError(-1))
}

pub fn device_sync() -> Result<(), CudaError> {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        CudaError::check(ffi::cudaDeviceSynchronize())
    }
    #[cfg(gb_no_cuda)]
    Err(CudaError(-1))
}

/// One `cudaMalloc` that [`DevBuf`] views are carved from (see [`crate::arena::Arena`]), or one mapped pinned host
/// allocation (`cudaHostAlloc` with `cudaHostAllocMapped`, used to measure zero-copy weights). It is freed when the arena
/// and the last view carved from it are gone.
#[derive(Debug)]
pub struct Block {
    /// The device address (for a mapped host block, the device-side alias of `host`).
    ptr: *mut c_void,
    bytes: usize,
    /// The host address of a host-backed block (mapped pinned, pageable, or file-backed mmap); null for device memory.
    host: *mut c_void,
    /// `host` came from `posix_memalign` (pageable memory the GPU reads through the CPU's page tables), not `cudaHostAlloc`.
    pageable: bool,
    /// `host` came from a file-backed `MAP_SHARED` `mmap` (read by the GPU via hardware ATS on GB10 while counted as
    /// reclaimable `Cached` file memory by Linux `/proc/meminfo`).
    file_mmap: bool,
    /// The page-aligned start and length of the `mmap` behind a `file_mmap` block, which is what `munmap` takes (`host` is where the
    /// bytes begin, which is not the start of the mapping when the bytes begin mid-page).
    map_base: *mut c_void,
    map_len: usize,
}

// An address and a length; nothing is mutated through the struct.
unsafe impl Send for Block {}
unsafe impl Sync for Block {}

impl Block {
    /// Whether this block is ordinary pageable host memory (read by the GPU through the CPU's page
    /// tables): such bytes still count in `MemAvailable` until migrated, so memory accounting must
    /// subtract them explicitly.
    pub fn is_pageable_host(&self) -> bool {
        self.pageable
    }

    pub fn alloc(bytes: usize) -> Result<Self, CudaError> {
        #[cfg(not(gb_no_cuda))]
        unsafe {
            let mut p = core::ptr::null_mut();
            let _t = loadstats::timer(Stage::Malloc, bytes as u64);
            CudaError::check(ffi::cudaMalloc(&mut p, bytes.max(1)))?;
            Ok(Self {
                ptr: p,
                bytes,
                host: core::ptr::null_mut(),
                pageable: false,
                file_mmap: false,
                map_base: core::ptr::null_mut(),
                map_len: 0,
            })
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = bytes;
            Err(CudaError(-1))
        }
    }
    /// Pinned host memory the GPU can address directly (`cudaHostAllocMapped`). Kernels reading it read host memory;
    /// `tools/load_ceiling.cu zerocopy` and `bench_decode_gbps_mapped` measure what that costs.
    pub fn alloc_mapped_host(bytes: usize) -> Result<Self, CudaError> {
        #[cfg(not(gb_no_cuda))]
        unsafe {
            let (mut h, mut d) = (core::ptr::null_mut(), core::ptr::null_mut());
            let _t = loadstats::timer(Stage::Pin, bytes as u64);
            // cudaHostAllocMapped = 2
            CudaError::check(ffi::cudaHostAlloc(&mut h, bytes.max(1), 2))?;
            if let Err(e) = CudaError::check(ffi::cudaHostGetDevicePointer(&mut d, h, 0)) {
                ffi::cudaFreeHost(h);
                return Err(e);
            }
            Ok(Self { ptr: d, bytes, host: h, pageable: false, file_mmap: false, map_base: core::ptr::null_mut(), map_len: 0 })
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = bytes;
            Err(CudaError(-1))
        }
    }
    /// Ordinary (pageable) host memory the GPU reads in place through the system's address translation
    /// (`cudaDevAttrPageableMemoryAccess`, true on the GB10): no device allocation at all. Measurement only, like
    /// [`Self::alloc_mapped_host`].
    pub fn alloc_pageable_host(bytes: usize) -> Result<Self, CudaError> {
        let mut p: *mut c_void = core::ptr::null_mut();
        let cap = bytes.max(1);
        // SAFETY: posix_memalign writes a pointer or returns an error; 2 MiB is a power of two multiple of the pointer size
        let rc = unsafe { libc::posix_memalign(&mut p, 2 << 20, cap) };
        if rc != 0 {
            return Err(CudaError(2)); // cudaErrorMemoryAllocation
        }
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(p, cap, libc::MADV_HUGEPAGE);
            const MADV_POPULATE_WRITE: libc::c_int = 23;
            libc::madvise(p, cap, MADV_POPULATE_WRITE);
            if libc::mlock(p, cap) != 0 {
                const STEP: usize = 64 << 20;
                let mut off = 0usize;
                while off < cap {
                    let len = (cap - off).min(STEP);
                    if libc::mlock((p as *mut u8).add(off) as *const c_void, len) != 0 {
                        break;
                    }
                    off += len;
                }
            }
        }
        Ok(Self { ptr: p, bytes, host: p, pageable: true, file_mmap: false, map_base: core::ptr::null_mut(), map_len: 0 })
    }

    /// File-backed `MAP_SHARED` host memory in `/var/tmp` (unlinked immediately so the kernel reclaims the inode on unmap/exit).
    /// On GB10 (`cudaDevAttrPageableMemoryAccess = 1`), the GPU reads this memory directly via hardware NVLink-C2C ATS while
    /// the Linux kernel accounts its physical RAM pages under `Cached` (reclaimable file cache), preserving `MemAvailable` above
    /// the floor when the last 1-2 expert layers exceed anonymous `MemAvailable` by ~0.8 GiB.
    pub fn alloc_file_backed(bytes: usize) -> Result<Self, CudaError> {
        #[cfg(target_os = "linux")]
        unsafe {
            let cap = bytes.max(4096).div_ceil(4096) * 4096;
            let mut tmpl = *b"/var/tmp/rexl3-slab-XXXXXX\0";
            let fd = libc::mkstemp(tmpl.as_mut_ptr() as *mut libc::c_char);
            if fd < 0 {
                return Err(CudaError(2));
            }
            libc::unlink(tmpl.as_ptr() as *const libc::c_char);
            if libc::ftruncate(fd, cap as libc::off_t) != 0 {
                libc::close(fd);
                return Err(CudaError(2));
            }
            let p = libc::mmap(core::ptr::null_mut(), cap, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0);
            libc::close(fd);
            if p == libc::MAP_FAILED {
                return Err(CudaError(2));
            }
            libc::madvise(p, cap, libc::MADV_RANDOM);
            Ok(Self { ptr: p, bytes, host: p, pageable: false, file_mmap: true, map_base: p, map_len: cap })
        }
        #[cfg(not(target_os = "linux"))]
        {
            Self::alloc_pageable_host(bytes)
        }
    }
    /// `bytes` bytes at `offset` of the file `path`, mapped read-only and shared, for the GPU to read in place. Nothing is copied and no
    /// device memory is taken: the pages are the file's own page cache, which Linux counts as reclaimable `Cached` memory, faulted in by
    /// the rows that are read and dropped under pressure (the bytes are the file's, so a dropped page costs one re-read and never a
    /// wrong byte). Requires `cudaDevAttrPageableMemoryAccess` (see [`pageable_memory_access`]); the caller decides.
    pub fn map_file_readonly(path: &std::path::Path, offset: u64, bytes: usize) -> Result<Self, CudaError> {
        #[cfg(target_os = "linux")]
        unsafe {
            use std::os::unix::io::AsRawFd;
            let f = std::fs::File::open(path).map_err(|_| CudaError(2))?;
            let page = libc::sysconf(libc::_SC_PAGESIZE).max(4096) as u64;
            let start = offset / page * page;
            let lead = (offset - start) as usize;
            let len = (lead + bytes.max(1)).div_ceil(page as usize) * page as usize;
            let base =
                libc::mmap(core::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), start as libc::off_t);
            if base == libc::MAP_FAILED {
                return Err(CudaError(2));
            }
            // random access: a row is read by itself, and nothing is read ahead that a token never asks for
            libc::madvise(base, len, libc::MADV_RANDOM);
            let p = (base as *mut u8).add(lead) as *mut c_void;
            Ok(Self { ptr: p, bytes, host: p, pageable: false, file_mmap: true, map_base: base, map_len: len })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (path, offset, bytes);
            Err(CudaError(2))
        }
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    /// The CPU address of the block's bytes when it is host memory.
    pub(crate) fn host_base(&self) -> Option<*mut c_void> {
        (!self.host.is_null()).then_some(self.host)
    }
    /// Address `offset` bytes into the block.
    pub fn ptr_at(&self, offset: usize) -> *mut c_void {
        assert!(offset <= self.bytes, "offset past the end of the block");
        unsafe { (self.ptr as *mut u8).add(offset) as *mut c_void }
    }
    /// The CPU address of byte `offset` of a host-backed block (mapped pinned, pageable or file-backed). The
    /// async D2H staging a decode round's read-backs use must be pinned, so the callers keep their staging in
    /// [`Self::alloc_mapped_host`] blocks and read them through this.
    pub fn host_ptr_at(&self, offset: usize) -> *mut u8 {
        assert!(!self.host.is_null(), "host_ptr_at on a device block");
        assert!(offset <= self.bytes, "offset past the end of the block");
        unsafe { (self.host as *mut u8).add(offset) }
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        #[cfg(not(gb_no_cuda))]
        unsafe {
            if self.host.is_null() {
                ffi::cudaFree(self.ptr);
            } else if self.file_mmap {
                #[cfg(target_os = "linux")]
                {
                    libc::munmap(self.map_base, self.map_len);
                }
            } else if self.pageable {
                #[cfg(target_os = "linux")]
                libc::munlock(self.host, self.bytes.max(1));
                libc::free(self.host);
            } else {
                ffi::cudaFreeHost(self.host);
            }
        }
    }
}

/// A synchronous device-to-host copy of `bytes` bytes from the device address `src` (lane S: the routed-pair counts of a TP exchange).
/// The caller synchronises the producing stream first (this copy runs on the legacy default stream).
///
/// # Safety
/// `src` must address `bytes` live device bytes and `dst` `bytes` writable host bytes.
pub unsafe fn download_raw(src: usize, dst: *mut c_void, bytes: usize) -> Result<(), CudaError> {
    #[cfg(not(gb_no_cuda))]
    {
        CudaError::check(ffi::cudaMemcpy(dst, src as *const c_void, bytes, D2H))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (src, dst, bytes);
        Err(CudaError(-1))
    }
}

/// A device allocation, or a view into an arena block (`owner` is set). An owned buffer is freed on drop; a view keeps its
/// block alive and frees nothing.
#[derive(Debug)]
pub struct DevBuf {
    ptr: *mut c_void,
    bytes: usize,
    owner: Option<std::sync::Arc<Block>>,
}

// A device pointer is just an address; the buffer is only used from the thread that drives its stream.
unsafe impl Send for DevBuf {}

impl DevBuf {
    /// A view of `bytes` bytes at `offset` into `block`, which it keeps alive.
    pub fn view(block: &std::sync::Arc<Block>, offset: usize, bytes: usize) -> Self {
        assert!(offset + bytes <= block.bytes, "view past the end of its block");
        Self { ptr: block.ptr_at(offset), bytes, owner: Some(block.clone()) }
    }
    /// A view of `bytes` bytes at `byte_offset` into this buffer, sharing its block (so it keeps the block alive and, like any
    /// view, frees nothing). Only a view or a block's own whole-block view can be sub-viewed; an allocation of its own cannot.
    pub fn subview(&self, byte_offset: usize, bytes: usize) -> Self {
        let owner = self.owner.as_ref().expect("subview of a buffer that is not a view into a block");
        assert!(
            byte_offset + bytes <= self.bytes,
            "subview past the end of its buffer ({byte_offset} + {bytes} > {})",
            self.bytes
        );
        let base = self.ptr as usize - owner.ptr_at(0) as usize;
        Self { ptr: owner.ptr_at(base + byte_offset), bytes, owner: Some(owner.clone()) }
    }
    /// A second handle to this buffer's bytes that frees nothing and holds no reference to a block (lane S: a byte exchange whose send
    /// and receive ranges are disjoint parts of one buffer). It must not outlive `self`.
    pub(crate) fn alias(&self) -> core::mem::ManuallyDrop<DevBuf> {
        core::mem::ManuallyDrop::new(Self { ptr: self.ptr, bytes: self.bytes, owner: None })
    }
    /// The block this buffer is a view of (`None` for an allocation of its own).
    pub fn block(&self) -> Option<&std::sync::Arc<Block>> {
        self.owner.as_ref()
    }
    pub fn alloc(bytes: usize) -> Result<Self, CudaError> {
        #[cfg(not(gb_no_cuda))]
        unsafe {
            let mut p = core::ptr::null_mut();
            // cudaMalloc(0) returns a null pointer; keep one byte so every buffer has a real address
            let _t = loadstats::timer(Stage::Malloc, bytes as u64);
            CudaError::check(ffi::cudaMalloc(&mut p, bytes.max(1)))?;
            Ok(Self { ptr: p, bytes, owner: None })
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = bytes;
            Err(CudaError(-1))
        }
    }
    pub fn zeroed(bytes: usize) -> Result<Self, CudaError> {
        let b = Self::alloc(bytes)?;
        #[cfg(not(gb_no_cuda))]
        unsafe {
            let _t = loadstats::timer(Stage::Memset, bytes as u64);
            CudaError::check(ffi::cudaMemset(b.ptr, 0, bytes.max(1)))?;
        }
        Ok(b)
    }
    /// [`DevBuf::zeroed`] as a view of a block of its own, so that [`DevBuf::subview`] works on it (a chunk's row-indexed activations are cut into
    /// slabs of rows that way). The same one `cudaMalloc`.
    pub fn zeroed_view(bytes: usize) -> Result<Self, CudaError> {
        let block = std::sync::Arc::new(Block::alloc(bytes.max(1))?);
        #[cfg(not(gb_no_cuda))]
        unsafe {
            let _t = loadstats::timer(Stage::Memset, bytes as u64);
            CudaError::check(ffi::cudaMemset(block.ptr_at(0), 0, bytes.max(1)))?;
        }
        Ok(Self::view(&block, 0, bytes))
    }
    pub fn zeroed_pageable_view(bytes: usize) -> Result<Self, CudaError> {
        let block = std::sync::Arc::new(Block::alloc_pageable_host(bytes.max(1))?);
        if let Some(base) = block.host_base() {
            unsafe { core::ptr::write_bytes(base as *mut u8, 0, bytes.max(1)) };
        }
        Ok(Self::view(&block, 0, bytes))
    }
    pub fn from_slice<T: Copy>(v: &[T]) -> Result<Self, CudaError> {
        let bytes = std::mem::size_of_val(v);
        // A view of a block of its own (like `zeroed_view`): this buffer has an owner, so `subview`
        // (used by `MoeBand` to share the fields of a `MoeExperts`) works on it. One `cudaMalloc`, same
        // device memory, freed when the last view of the block drops.
        let block = std::sync::Arc::new(Block::alloc(bytes.max(1))?);
        let mut b = Self::view(&block, 0, bytes);
        b.upload(v)?;
        Ok(b)
    }
    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }
    /// Address `byte_offset` bytes into the buffer (for sub-buffers carved out of one allocation).
    pub fn ptr_at(&self, byte_offset: usize) -> *mut c_void {
        assert!(byte_offset <= self.bytes, "offset past the end of the buffer");
        unsafe { (self.ptr as *mut u8).add(byte_offset) as *mut c_void }
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn upload<T: Copy>(&mut self, v: &[T]) -> Result<(), CudaError> {
        let n = std::mem::size_of_val(v);
        assert!(n <= self.bytes, "upload of {n} bytes into a {}-byte buffer", self.bytes);
        // a buffer in host memory is written by the CPU (a host-to-device copy into a host address is not valid for pageable memory)
        if let Some((base, block_ptr)) = self.owner.as_ref().and_then(|b| b.host_base().map(|h| (h, b.ptr_at(0)))) {
            let off = self.ptr as usize - block_ptr as usize;
            // SAFETY: the view lies inside the block (checked when it was carved) and `v` is `n` readable bytes
            unsafe { core::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, (base as *mut u8).add(off), n) };
            return Ok(());
        }
        #[cfg(not(gb_no_cuda))]
        unsafe {
            let _t = loadstats::timer(Stage::Memcpy, n as u64);
            CudaError::check(ffi::cudaMemcpy(self.ptr, v.as_ptr() as *const c_void, n, H2D))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (n, H2D);
            Err(CudaError(-1))
        }
    }
    pub fn to_vec<T: Copy + Default>(&self) -> Result<Vec<T>, CudaError> {
        let n = self.bytes / std::mem::size_of::<T>();
        let mut v = vec![T::default(); n];
        self.download(&mut v)?;
        Ok(v)
    }
    pub fn download<T: Copy>(&self, out: &mut [T]) -> Result<(), CudaError> {
        let n = std::mem::size_of_val(out);
        assert!(n <= self.bytes, "download of {n} bytes from a {}-byte buffer", self.bytes);
        if let Some((base, block_ptr)) = self.owner.as_ref().and_then(|b| b.host_base().map(|h| (h, b.ptr_at(0)))) {
            let off = self.ptr as usize - block_ptr as usize;
            unsafe { core::ptr::copy_nonoverlapping((base as *const u8).add(off), out.as_mut_ptr() as *mut u8, n) };
            return Ok(());
        }
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::cudaMemcpy(out.as_mut_ptr() as *mut c_void, self.ptr, n, D2H))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (n, D2H, out);
            Err(CudaError(-1))
        }
    }
    pub fn copy_from(&mut self, src: &DevBuf, bytes: usize, stream: &Stream) -> Result<(), CudaError> {
        assert!(bytes <= self.bytes && bytes <= src.bytes);
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::cudaMemcpyAsync(self.ptr, src.ptr, bytes, D2D, stream.raw()))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (bytes, stream, D2D);
            Err(CudaError(-1))
        }
    }
    /// `bytes` from `src` at `src_off` into this buffer at `dst_off` (device to device, async on `stream`) —
    /// the offset form of [`Self::copy_from`] for the save/restore moves of a tree round's path windows,
    /// which copy a slice of a larger buffer in place.
    pub fn copy_from_at(
        &mut self,
        dst_off: usize,
        src: &DevBuf,
        src_off: usize,
        bytes: usize,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        assert!(dst_off + bytes <= self.bytes, "dst slice {dst_off}+{bytes} over {} bytes", self.bytes);
        assert!(src_off + bytes <= src.bytes, "src slice {src_off}+{bytes} over {} bytes", src.bytes);
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::cudaMemcpyAsync(self.ptr_at(dst_off), src.ptr_at(src_off), bytes, D2D, stream.raw()))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (dst_off, src, src_off, bytes, stream, D2D);
            Err(CudaError(-1))
        }
    }
    /// Read `out.len()` elements starting `byte_off` bytes into the buffer (synchronous).
    pub fn download_range<T: Copy>(&self, byte_off: usize, out: &mut [T]) -> Result<(), CudaError> {
        let n = std::mem::size_of_val(out);
        assert!(byte_off + n <= self.bytes, "download of {n} bytes at {byte_off} from a {}-byte buffer", self.bytes);
        if let Some((base, block_ptr)) = self.owner.as_ref().and_then(|b| b.host_base().map(|h| (h, b.ptr_at(0)))) {
            let off = (self.ptr as usize - block_ptr as usize) + byte_off;
            unsafe { core::ptr::copy_nonoverlapping((base as *const u8).add(off), out.as_mut_ptr() as *mut u8, n) };
            return Ok(());
        }
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::cudaMemcpy(
                out.as_mut_ptr() as *mut c_void,
                (self.ptr as *const u8).add(byte_off) as *const c_void,
                n,
                D2H,
            ))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (n, D2H, out);
            Err(CudaError(-1))
        }
    }
    /// Copy `bytes` bytes from `src` at `src_off` into this buffer at `dst_off` (stream-ordered, device to device).
    pub fn copy_range_from(
        &mut self,
        dst_off: usize,
        src: &DevBuf,
        src_off: usize,
        bytes: usize,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        assert!(dst_off + bytes <= self.bytes && src_off + bytes <= src.bytes, "copy range past the end of a buffer");
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::cudaMemcpyAsync(
                (self.ptr as *mut u8).add(dst_off) as *mut c_void,
                (src.ptr as *const u8).add(src_off) as *const c_void,
                bytes,
                D2D,
                stream.raw(),
            ))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (dst_off, src_off, bytes, stream, D2D);
            Err(CudaError(-1))
        }
    }
    /// Copy a rectangle of bytes device to device, stream-ordered: `height` rows of `width` bytes, the rows `src_pitch` bytes apart in
    /// `src` starting at `src_off` and `dst_pitch` bytes apart in this buffer starting at `dst_off` (`cudaMemcpy2DAsync`). With
    /// `height == 1` this is one plain copy and the pitches are not used. This is how a column range of a row-major array is gathered
    /// without the host.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_2d_from(
        &mut self,
        dst_off: usize,
        dst_pitch: usize,
        src: &DevBuf,
        src_off: usize,
        src_pitch: usize,
        width: usize,
        height: usize,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        if height == 1 {
            return self.copy_range_from(dst_off, src, src_off, width, stream);
        }
        let last = |off: usize, pitch: usize| off + (height - 1) * pitch + width;
        assert!(
            height == 1 || (width <= dst_pitch && width <= src_pitch),
            "copy_2d_from: a row of {width} bytes does not fit its pitches ({dst_pitch}, {src_pitch})"
        );
        assert!(
            last(dst_off, dst_pitch) <= self.bytes && last(src_off, src_pitch) <= src.bytes,
            "copy_2d_from past the end of a buffer"
        );
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::cudaMemcpy2DAsync(
                (self.ptr as *mut u8).add(dst_off) as *mut c_void,
                dst_pitch,
                (src.ptr as *const u8).add(src_off) as *const c_void,
                src_pitch,
                width,
                height,
                D2D,
                stream.raw(),
            ))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (dst_off, dst_pitch, src_off, src_pitch, stream, D2D);
            Err(CudaError(-1))
        }
    }
    /// Copy `bytes` bytes inside this buffer from `src_off` to `dst_off` (the two ranges must not overlap).
    pub fn copy_within(&mut self, dst_off: usize, src_off: usize, bytes: usize, stream: &Stream) -> Result<(), CudaError> {
        assert!(dst_off + bytes <= self.bytes && src_off + bytes <= self.bytes, "copy range past the end of the buffer");
        assert!(dst_off + bytes <= src_off || src_off + bytes <= dst_off, "copy_within ranges overlap");
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::cudaMemcpyAsync(
                (self.ptr as *mut u8).add(dst_off) as *mut c_void,
                (self.ptr as *const u8).add(src_off) as *const c_void,
                bytes,
                D2D,
                stream.raw(),
            ))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (dst_off, src_off, bytes, stream, D2D);
            Err(CudaError(-1))
        }
    }
    /// Fill `len` bytes starting `offset` bytes into the buffer.
    pub fn memset_range_async(&mut self, offset: usize, len: usize, v: u8, stream: &Stream) -> Result<(), CudaError> {
        assert!(offset + len <= self.bytes, "memset range past the end of the buffer");
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::cudaMemsetAsync(
                (self.ptr as *mut u8).add(offset) as *mut c_void,
                v as i32,
                len.max(1),
                stream.raw(),
            ))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (offset, len, v, stream);
            Err(CudaError(-1))
        }
    }
    pub fn memset_async(&mut self, v: u8, stream: &Stream) -> Result<(), CudaError> {
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::cudaMemsetAsync(self.ptr, v as i32, self.bytes.max(1), stream.raw()))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (v, stream);
            Err(CudaError(-1))
        }
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        #[cfg(not(gb_no_cuda))]
        if self.owner.is_none() {
            unsafe {
                ffi::cudaFree(self.ptr);
            }
        }
    }
}

/// The host bytes of host-to-device copies queued on one stream, kept unchanged until the stream has run them.
///
/// [`Stream::memcpy_h2d_async`] may read a pageable source after it returns, so the source must outlive the copy and stay
/// untouched until the stream completes. A step's inputs (token ids, positions, sequence starts and slots, a block table)
/// mostly live in temporaries the caller drops or rewrites at once: [`HostStage::upload`] copies them into a buffer the stage
/// owns, queues the copy from that buffer on the stream, and keeps the buffer until [`HostStage::sync`] has waited for the
/// stream. The kept buffers are then reused, so a steady decode loop allocates nothing.
///
/// This is how a model whose stream is non-blocking ([`Stream::new_nonblocking`]) writes its per-step inputs in the stream's
/// order. [`DevBuf::upload`] (a pageable `cudaMemcpy`, which can return before its DMA lands) and [`DevBuf::zeroed`] (an
/// asynchronous `cudaMemset`) run on the legacy default stream, which a non-blocking stream does not wait for: a kernel
/// launched right after them can read the previous step's bytes, and one still running from the previous step can see the
/// new ones. Every copy of one stage goes on one stream.
#[derive(Debug, Default)]
pub struct HostStage {
    /// Buffers whose copies may still be queued on the stream.
    queued: Vec<Vec<u8>>,
    /// Buffers free for the next copies.
    spare: Vec<Vec<u8>>,
}

impl HostStage {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queues a copy of `v` to the start of `dst` on `st`: a kernel launched on `st` after this call reads the new bytes, and
    /// one launched before it has read the old ones. `v` may be dropped or changed as soon as this returns; no device
    /// synchronisation is made. `dst` must be device memory (a host-backed buffer is written by the CPU, outside any stream).
    pub fn upload<T: Copy>(&mut self, dst: &mut DevBuf, v: &[T], st: &Stream) -> Result<(), CudaError> {
        let n = std::mem::size_of_val(v);
        assert!(n <= dst.bytes(), "upload of {n} bytes into a {}-byte buffer", dst.bytes());
        assert!(dst.block().and_then(|b| b.host_base()).is_none(), "HostStage::upload into host memory");
        if n == 0 {
            return Ok(());
        }
        let src = self.keep(v);
        // SAFETY: `dst` is device memory of at least `n` bytes, and `src` is `n` bytes the stage holds unchanged until `sync`
        // has waited for the stream (or, on drop, for the device)
        unsafe { st.memcpy_h2d_async(dst.ptr(), src, n) }
    }

    /// [`HostStage::upload`] to `byte_offset` bytes into `dst` (a part of a buffer, such as one slot's row of a block table).
    pub fn upload_at<T: Copy>(&mut self, dst: &mut DevBuf, byte_offset: usize, v: &[T], st: &Stream) -> Result<(), CudaError> {
        let n = std::mem::size_of_val(v);
        assert!(byte_offset + n <= dst.bytes(), "upload of {n} bytes at {byte_offset} into a {}-byte buffer", dst.bytes());
        assert!(dst.block().and_then(|b| b.host_base()).is_none(), "HostStage::upload_at into host memory");
        if n == 0 {
            return Ok(());
        }
        let src = self.keep(v);
        // SAFETY: `dst` is device memory with `n` bytes at `byte_offset`, and `src` is `n` bytes the stage holds unchanged until `sync`
        // has waited for the stream (or, on drop, for the device)
        unsafe { st.memcpy_h2d_async(dst.ptr_at(byte_offset), src, n) }
    }

    /// Waits for `st`, the stream every copy of this stage was queued on; the kept buffers are reused from here on.
    pub fn sync(&mut self, st: &Stream) -> Result<(), CudaError> {
        st.sync()?;
        self.release();
        Ok(())
    }

    /// Buffers whose copies may still be queued (each is kept until the next [`HostStage::sync`]).
    pub fn queued(&self) -> usize {
        self.queued.len()
    }

    /// Copies `v` into a buffer of the stage (a spare one when there is one) and returns the address of the copy, which stays
    /// valid and unchanged until [`HostStage::release`].
    fn keep<T: Copy>(&mut self, v: &[T]) -> *const u8 {
        let n = std::mem::size_of_val(v);
        let mut buf = self.spare.pop().unwrap_or_default();
        buf.clear();
        // SAFETY: `v` is `n` readable bytes of plain data (`T: Copy`)
        buf.extend_from_slice(unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n) });
        let src = buf.as_ptr();
        // moving the Vec into the list does not move its heap bytes, so `src` stays valid while the buffer is queued
        self.queued.push(buf);
        src
    }

    /// Every queued copy has run: the buffers become spares.
    fn release(&mut self) {
        self.spare.append(&mut self.queued);
    }
}

impl Drop for HostStage {
    fn drop(&mut self) {
        // a copy may still read a kept buffer (an error path skipped the stream's sync, or its stream is already gone): wait
        // for the device, which also covers the work of a destroyed stream, before the buffers are freed
        if !self.queued.is_empty() {
            let _ = device_sync();
        }
    }
}

pub struct Stream(*mut c_void);

unsafe impl Send for Stream {}

impl Stream {
    pub fn new() -> Result<Self, CudaError> {
        #[cfg(not(gb_no_cuda))]
        unsafe {
            let mut s = core::ptr::null_mut();
            CudaError::check(ffi::cudaStreamCreate(&mut s))?;
            Ok(Self(s))
        }
        #[cfg(gb_no_cuda)]
        Err(CudaError(-1))
    }
    /// A stream that does not synchronise with the legacy default stream (`cudaStreamNonBlocking`): the loader's copies
    /// must not wait for, or hold up, whatever the engine thread runs on the default stream.
    pub fn new_nonblocking() -> Result<Self, CudaError> {
        #[cfg(not(gb_no_cuda))]
        unsafe {
            let mut s = core::ptr::null_mut();
            CudaError::check(ffi::cudaStreamCreateWithFlags(&mut s, 1))?;
            Ok(Self(s))
        }
        #[cfg(gb_no_cuda)]
        Err(CudaError(-1))
    }
    /// Queue a host-to-device copy of `bytes` bytes from `src` to the device address `dst`. For pageable `src` the call
    /// returns once the driver has taken the bytes, but `src` must stay untouched until the stream (or an event recorded
    /// after this call) completes, which is the contract the loader's staging buffers follow.
    ///
    /// # Safety
    /// `dst` must be a device address with `bytes` bytes behind it and `src` must be readable for `bytes` bytes.
    pub unsafe fn memcpy_h2d_async(&self, dst: *mut c_void, src: *const u8, bytes: usize) -> Result<(), CudaError> {
        #[cfg(not(gb_no_cuda))]
        {
            CudaError::check(ffi::cudaMemcpyAsync(dst, src as *const c_void, bytes, H2D, self.0))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (dst, src, bytes, H2D);
            Err(CudaError(-1))
        }
    }
    /// Queue a device-to-host copy of `bytes` bytes from the device address `src` to `dst`. `dst` must be
    /// pinned host memory (a [`Block::alloc_mapped_host`] block): pageable host memory is invalid for an async
    /// D2H copy. The caller records an event after the copies and waits on it (see the decode round's read-back
    /// staging), instead of synchronizing the whole stream.
    ///
    /// # Safety
    /// `dst` must be pinned and writable for `bytes` bytes and `src` must be a device address with `bytes` bytes
    /// behind it.
    pub unsafe fn memcpy_d2h_async(&self, dst: *mut u8, src: *const c_void, bytes: usize) -> Result<(), CudaError> {
        #[cfg(not(gb_no_cuda))]
        {
            CudaError::check(ffi::cudaMemcpyAsync(dst as *mut c_void, src, bytes, D2H, self.0))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (dst, src, bytes, D2H);
            Err(CudaError(-1))
        }
    }
    pub fn raw(&self) -> *mut c_void {
        self.0
    }
    pub fn sync(&self) -> Result<(), CudaError> {
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::cudaStreamSynchronize(self.0))
        }
        #[cfg(gb_no_cuda)]
        Err(CudaError(-1))
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        #[cfg(not(gb_no_cuda))]
        unsafe {
            ffi::cudaStreamDestroy(self.0);
        }
    }
}

/// A CUDA event without timing, used to learn that the copies queued before it have finished.
pub struct Event(*mut c_void);

unsafe impl Send for Event {}

impl Event {
    pub fn new() -> Result<Self, CudaError> {
        #[cfg(not(gb_no_cuda))]
        unsafe {
            let mut e = core::ptr::null_mut();
            // cudaEventDisableTiming
            CudaError::check(ffi::cudaEventCreateWithFlags(&mut e, 2))?;
            Ok(Self(e))
        }
        #[cfg(gb_no_cuda)]
        Err(CudaError(-1))
    }
    /// Mark the point in `stream` after everything queued so far.
    pub fn record(&self, stream: &Stream) -> Result<(), CudaError> {
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::cudaEventRecord(self.0, stream.raw()))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = stream;
            Err(CudaError(-1))
        }
    }
    pub fn raw(&self) -> *mut c_void {
        self.0
    }
    /// Block the calling thread until the recorded point has been reached.
    pub fn sync(&self) -> Result<(), CudaError> {
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::cudaEventSynchronize(self.0))
        }
        #[cfg(gb_no_cuda)]
        Err(CudaError(-1))
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        #[cfg(not(gb_no_cuda))]
        unsafe {
            ffi::cudaEventDestroy(self.0);
        }
    }
}

/// Raw CUDA events WITH timing, for profiling tests that keep hundreds of them and ask for the time between any two. Every function
/// is `unsafe` (the pointers are the caller's) and returns the CUDA status; in the stub build each returns `-1`, so a test that uses
/// them links without a CUDA runtime and reports itself unrunnable.
pub mod timed_events {
    use core::ffi::c_void;

    /// `cudaEventCreate`.
    ///
    /// # Safety
    /// `e` must be valid for a pointer write.
    pub unsafe fn create(e: *mut *mut c_void) -> i32 {
        #[cfg(not(gb_no_cuda))]
        {
            super::ffi::cudaEventCreate(e)
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = e;
            -1
        }
    }

    /// `cudaEventRecord` on `stream` (a raw stream handle).
    ///
    /// # Safety
    /// `e` and `stream` must be live handles.
    pub unsafe fn record(e: *mut c_void, stream: *mut c_void) -> i32 {
        #[cfg(not(gb_no_cuda))]
        {
            super::ffi::cudaEventRecord(e, stream)
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (e, stream);
            -1
        }
    }

    /// `cudaEventSynchronize`.
    ///
    /// # Safety
    /// `e` must be a live event.
    pub unsafe fn synchronize(e: *mut c_void) -> i32 {
        #[cfg(not(gb_no_cuda))]
        {
            super::ffi::cudaEventSynchronize(e)
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = e;
            -1
        }
    }

    /// `cudaEventElapsedTime`: milliseconds from `a` to `b`, both recorded and complete.
    ///
    /// # Safety
    /// `ms` must be valid for a write; `a` and `b` must be live, recorded events created by [`create`].
    pub unsafe fn elapsed_ms(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> i32 {
        #[cfg(not(gb_no_cuda))]
        {
            super::ffi::cudaEventElapsedTime(ms, a, b)
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (ms, a, b);
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sticky_device_errors_are_fatal_and_per_call_errors_are_not() {
        // every code driver_types.h (CUDA 13.0) says leaves the process in an inconsistent state, plus the ones that mean the device is
        // gone or its memory unusable
        for code in [46, 100, 101, 201, 214, 220, 226, 700, 702, 709, 710, 714, 715, 716, 717, 718, 719, 721, 810, 911, 999] {
            assert!(CudaError(code).is_fatal(), "{code} ({}) must be fatal", CudaError(code).name());
        }
        // an argument, an allocation or a launch shape that did not fit spoils one call only
        for code in [1, 2, 9, 600, 701] {
            assert!(!CudaError(code).is_fatal(), "{code} ({}) must be retryable", CudaError(code).name());
        }
        // a code this build has never seen fails closed, and so does the stub build's
        assert!(CudaError(123_456).is_fatal() && CudaError(-1).is_fatal());
        assert_eq!(CudaError(700).to_string(), "CUDA error 700 (cudaErrorIllegalAddress)");
    }

    /// The staging contract of [`HostStage`], which needs no device: the copy queued on the stream reads the stage's own copy
    /// of the caller's bytes, that copy survives the caller changing or dropping its slice and later uploads of the same step,
    /// and a buffer is reused only after the stream was waited for.
    #[test]
    fn a_staged_upload_reads_the_stages_own_copy_and_keeps_it_until_the_stream_is_synced() {
        let mut stage = HostStage::new();
        let mut ids = vec![7i32, 8, 9];
        let a = stage.keep(&ids);
        assert_ne!(a, ids.as_ptr() as *const u8, "the copy must not read the caller's slice");
        let read = |p: *const u8, n: usize| unsafe { std::slice::from_raw_parts(p, n).to_vec() };
        let want: Vec<u8> = [7i32, 8, 9].iter().flat_map(|v| v.to_ne_bytes()).collect();
        assert_eq!(read(a, 12), want);
        // the caller rewrites its slice for the next step and uploads it again before the stream has run the first copy
        ids.copy_from_slice(&[1, 2, 3]);
        let b = stage.keep(&ids);
        drop(ids);
        assert_ne!(a, b, "a buffer still queued must not be reused");
        assert_eq!(read(a, 12), want, "a queued copy's bytes changed under it");
        assert_eq!(read(b, 12), [1i32, 2, 3].iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<u8>>());
        assert_eq!(stage.queued(), 2);
        // after the stream was waited for, the buffers are reused: no allocation per step
        stage.release();
        assert_eq!((stage.queued(), stage.spare.len()), (0, 2));
        let c = stage.keep(&[5u16; 4]);
        assert!(c == a || c == b, "a spare buffer is reused once the stream has run its copy");
        assert_eq!((stage.queued(), stage.spare.len()), (1, 1));
        stage.release();
    }

    #[test]
    fn an_nccl_code_is_named_not_unrecognised_and_keeps_its_number() {
        // 20103: the communicator was aborted (the group is lost and the engine thread was inside a collective)
        let t = CudaError(crate::nccl::ERR_ABORTED).to_string();
        assert_eq!(t, "CUDA error 20103 (communicator aborted)");
        assert_eq!(CudaError(20006).to_string(), "CUDA error 20006 (ncclRemoteError)");
        assert_eq!(CudaError(crate::nccl::ERR_PEER).to_string(), "CUDA error 20102 (loopback peer failed or timed out)");
        // a code nobody knows still says so, number included
        assert_eq!(CudaError(123_456).to_string(), "CUDA error 123456 (unrecognised CUDA error)");
        assert_eq!(CudaError(20_050).to_string(), "CUDA error 20050 (unrecognised CUDA error)");
    }
}

/// Whether the GPU reads ordinary pageable host memory, file mappings included, through the CPU's page tables
/// (`cudaDevAttrPageableMemoryAccess`): true on the GB10, where the whole of memory is one coherent pool. Probed once; false when there
/// is no device or the probe fails, so a caller that needs the capability falls back to device memory.
pub fn pageable_memory_access() -> bool {
    static PROBE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PROBE.get_or_init(|| {
        #[cfg(not(gb_no_cuda))]
        unsafe {
            let mut n = 0i32;
            if ffi::cudaGetDeviceCount(&mut n) != 0 || n < 1 {
                return false;
            }
            let mut v = 0i32;
            const PAGEABLE_MEMORY_ACCESS: i32 = 88; // cudaDevAttrPageableMemoryAccess
            ffi::cudaDeviceGetAttribute(&mut v, PAGEABLE_MEMORY_ACCESS, 0) == 0 && v == 1
        }
        #[cfg(gb_no_cuda)]
        {
            false
        }
    })
}
