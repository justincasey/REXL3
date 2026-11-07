//! CUDA access for the engine: device memory, streams, and the kernels in `csrc/` behind plain `extern "C"`
//! launchers. Kernel modules live in their own files; this one holds the runtime wrappers they share.
//!
//! With no nvcc at build time (`cfg(gb_no_cuda)`) the crate still compiles, `device_available()` is false and every
//! allocation fails, so the rest of the workspace builds and tests on machines without a GPU.

// A stub build (no nvcc) compiles the runtime wrappers without the code that reads some fields and uses some imports (graph handles, the pageable flag, load
// statistics); that is not dead code in a GPU build, and CI builds the stub.
#![cfg_attr(gb_no_cuda, allow(dead_code, unused_imports))]

/// Declares kernel launchers. A GPU build gets a plain `extern "C"` block; a stub build (no nvcc, `cfg(gb_no_cuda)`) gets a
/// Rust function per launcher with the same signature that fails (`cudaErrorNoDevice`, or 0 for a size query), so every
/// wrapper compiles everywhere and returns an error at run time instead of the crate needing a cfg on each wrapper.
macro_rules! launchers {
    ($( $(#[$m:meta])* pub fn $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) $(-> $ret:ty)? ; )*) => {
        #[cfg(not(gb_no_cuda))]
        extern "C" { $( $(#[$m])* pub fn $name($($arg: $ty),*) $(-> $ret)?; )* }
        $(
            #[cfg(gb_no_cuda)]
            #[allow(dead_code, unused_variables, clippy::too_many_arguments, clippy::missing_safety_doc)]
            pub unsafe fn $name($($arg: $ty),*) $(-> $ret)? { $crate::Stubbed::stubbed() }
        )*
    };
}
pub(crate) use launchers;

/// What a stub launcher returns.
#[doc(hidden)]
pub trait Stubbed {
    fn stubbed() -> Self;
}
impl Stubbed for i32 {
    fn stubbed() -> Self {
        100 // cudaErrorNoDevice
    }
}
impl Stubbed for usize {
    fn stubbed() -> Self {
        0
    }
}
// a size query that returns i64 (`gb_glm_p2m_plan_ws_bytes`): nothing to size without the kernels, as for `usize`
impl Stubbed for i64 {
    fn stubbed() -> Self {
        0
    }
}

pub mod arena;
pub mod attention;
pub mod bf16_linear;
pub mod dflash2;
pub mod dsa;
pub mod dspark;
pub mod dsv4_attn;
pub mod dsv4_kvq4;
pub mod dsv4_moe;
pub mod dsv4_mtp;
pub mod dsv4_mtp_gpu;
pub mod dsv4_vision;
pub mod exl3_big;
pub mod exl3_encode;
pub mod exl3_linear;
pub mod exl3_split;
pub mod expert_cache;
pub mod fmoe;
pub mod fn_glue;
pub mod fn_mhc;
pub mod fn_moe;
pub mod fp4_linear;
pub mod fp8_linear;
pub mod gdn;
pub mod glm_glue;
pub mod glm_mtp;
pub mod glm_p2;
pub mod glm_p2_mla;
pub mod glm_p2_moe;
pub mod glm_perf;
pub mod glm_vision;
pub mod glue;
pub mod graph;
pub mod hcx;
pub mod kda;
pub mod loader;
pub mod mcv_vision;
pub mod mhc;
pub mod mla;
pub mod mla_gated;
pub mod mla_prefill;
pub mod moe;
pub mod moe_owned;
pub mod moe_prefill;
pub mod moe_subbanks;
pub mod moe_width;
pub mod mtpk;
pub mod nccl;
pub mod nemotron_attn;
pub mod nemotron_attn_prefill;
pub mod nemotron_dense;
pub mod nemotron_dense_mma;
pub mod nemotron_glue;
pub mod nemotron_moe;
pub mod nemotron_moe_prefill;
pub mod nemotron_ssm;
pub mod nemotron_tp;
pub mod nemotron_vit;
pub mod num;
pub mod ple;
pub mod q4_linear;
pub mod qfn_verify;
pub mod qsa;
pub mod roce;
pub mod rt;
pub mod state_digest;
pub mod tree;
pub mod vision;
pub use rt::{device_available, CudaError, DevBuf, Stream};

#[cfg(not(gb_no_cuda))]
mod smoke_ffi {
    extern "C" {
        pub fn gb_smoke_scale_add(a: *const f32, b: *mut f32, n: i32, s: f32, t: f32, stream: *mut core::ffi::c_void) -> i32;
    }
}

/// b[i] = a[i] * s + t on the device; the pipeline smoke test (nvcc -> static lib -> Rust).
#[cfg(not(gb_no_cuda))]
pub fn smoke_scale_add(a: &DevBuf, b: &mut DevBuf, n: usize, s: f32, t: f32, stream: &Stream) -> Result<(), CudaError> {
    let rc = unsafe { smoke_ffi::gb_smoke_scale_add(a.ptr() as *const f32, b.ptr() as *mut f32, n as i32, s, t, stream.raw()) };
    CudaError::check(rc)
}

#[cfg(test)]
mod tests {
    #[cfg(not(gb_no_cuda))]
    #[test]
    fn nvcc_kernel_runs_from_rust() {
        use super::*;
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
        let host: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let a = DevBuf::from_slice(&host).unwrap();
        let mut b = DevBuf::zeroed(host.len() * 4).unwrap();
        let st = Stream::new().unwrap();
        smoke_scale_add(&a, &mut b, host.len(), 2.0, 1.0, &st).unwrap();
        st.sync().unwrap();
        let out: Vec<f32> = b.to_vec().unwrap();
        assert!(out.iter().enumerate().all(|(i, &v)| v == i as f32 * 2.0 + 1.0));
    }
}
