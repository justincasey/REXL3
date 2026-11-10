//! Compiles every `csrc/*.cu` with nvcc into one static library and links it with cudart.
//!
//! No nvcc (CI on x86, a laptop) means a stub build: the crate still compiles, `cfg(gb_no_cuda)` is set, and GPU
//! tests skip. `REXL3_CUDA_ARCH` overrides the target (default `sm_121`, the GB10); `REXL3_NO_CUDA` forces the stub build.
//! The old names `GB_CUDA_ARCH` and `GB_NO_CUDA` still work and print a warning. A source file can request extra nvcc
//! flags on its first line: `// gb-nvcc: --fmad=false`.
//!
//! The build identity is exported to the crate as `REXL3_NVCC_VERSION`, `REXL3_CUDA_ARCH` and `REXL3_KERNEL_HASH` (FNV-1a
//! over every file in `csrc/`, the arch and each file's flags), so a running server can say exactly which kernels it runs.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

/// Flags every kernel file is compiled with (part of the kernel hash).
const NVCC_FLAGS: &[&str] = &["-c", "-O3", "--std=c++17", "-Xcompiler", "-fPIC", "-lineinfo", "-I", "csrc"];

/// `REXL3_<name>`, else the old `GB_<name>` with a warning.
fn setting(name: &str) -> Option<String> {
    if let Ok(v) = env::var(format!("REXL3_{name}")) {
        return Some(v);
    }
    let old = env::var(format!("GB_{name}")).ok();
    if old.is_some() {
        println!("cargo:warning=GB_{name} is deprecated: set REXL3_{name}");
    }
    old
}

fn fnv1a(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

/// `13.0.88` from `nvcc --version` ("Cuda compilation tools, release 13.0, V13.0.88").
fn nvcc_version(nvcc: &Path) -> String {
    let out =
        Command::new(nvcc).arg("--version").output().map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
    out.split_whitespace().rev().find_map(|w| w.strip_prefix('V')).unwrap_or("unknown").to_string()
}

fn find_cuda() -> Option<PathBuf> {
    if setting("NO_CUDA").is_some() {
        return None; // force the stub build (used to test machines without a GPU toolchain)
    }
    if let Ok(h) = env::var("CUDA_HOME").or_else(|_| env::var("CUDA_PATH")) {
        let p = PathBuf::from(h);
        if p.join("bin/nvcc").exists() {
            return Some(p);
        }
    }
    if let Some(paths) = env::var_os("PATH") {
        for dir in env::split_paths(&paths) {
            let candidate = dir.join("nvcc");
            if candidate.exists() {
                if let Some(root) = dir.parent() {
                    return Some(root.to_path_buf());
                }
            }
        }
    }
    for p in ["/usr/local/cuda", "/usr/local/cuda-13.0", "/usr/local/cuda-13"] {
        let p = PathBuf::from(p);
        if p.join("bin/nvcc").exists() {
            return Some(p);
        }
    }
    None
}

fn extra_flags(src: &Path) -> Vec<String> {
    let text = fs::read_to_string(src).unwrap_or_default();
    match text.lines().next().and_then(|l| l.strip_prefix("// gb-nvcc:")) {
        Some(f) => f.split_whitespace().map(str::to_string).collect(),
        None => vec![],
    }
}

fn main() {
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    for v in ["CUDA_ARCH", "NO_CUDA"] {
        println!("cargo:rerun-if-env-changed=REXL3_{v}");
        println!("cargo:rerun-if-env-changed=GB_{v}");
    }
    println!("cargo::rustc-check-cfg=cfg(gb_no_cuda)");
    let Some(cuda) = find_cuda() else {
        println!("cargo:rustc-cfg=gb_no_cuda");
        println!("cargo:rustc-env=REXL3_NVCC_VERSION=none");
        println!("cargo:rustc-env=REXL3_CUDA_ARCH=none");
        println!("cargo:rustc-env=REXL3_KERNEL_HASH=none");
        println!("cargo:warning=nvcc not found: building gb-cuda without GPU support");
        return;
    };
    let nvcc = cuda.join("bin/nvcc");
    let arch = setting("CUDA_ARCH").unwrap_or_else(|| "sm_121".into());
    let mut hash = 0xcbf29ce484222325u64;
    fnv1a(&mut hash, arch.as_bytes());
    fnv1a(&mut hash, NVCC_FLAGS.join(" ").as_bytes());
    let mut all: Vec<PathBuf> = fs::read_dir("csrc").expect("csrc").filter_map(|e| e.ok().map(|e| e.path())).collect();
    all.sort();
    for p in &all {
        fnv1a(&mut hash, p.file_name().unwrap().to_string_lossy().as_bytes());
        fnv1a(&mut hash, &fs::read(p).expect("read a kernel source"));
    }
    println!("cargo:rustc-env=REXL3_NVCC_VERSION={}", nvcc_version(&nvcc));
    println!("cargo:rustc-env=REXL3_CUDA_ARCH={arch}");
    println!("cargo:rustc-env=REXL3_KERNEL_HASH={hash:016x}");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut srcs: Vec<PathBuf> = fs::read_dir("csrc")
        .expect("csrc")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "cu").unwrap_or(false))
        .collect();
    srcs.sort();
    let hdr_mtime = all
        .iter()
        .filter(|p| p.extension().map(|x| x == "cuh").unwrap_or(false))
        .filter_map(|p| fs::metadata(p).and_then(|m| m.modified()).ok())
        .max();
    let objs: Vec<PathBuf> = srcs.iter().map(|s| out.join(s.file_stem().unwrap()).with_extension("o")).collect();
    // a .cu that includes another .cu (gdn_hv32.cu, attention_kv2.cu, ...) is stale when the included file is newer than its object
    let included_newer = |s: &Path, om: std::time::SystemTime| -> bool {
        let text = fs::read_to_string(s).unwrap_or_default();
        text.lines()
            .filter_map(|l| l.trim().strip_prefix("#include \"").and_then(|r| r.strip_suffix('"')))
            .filter(|n| n.ends_with(".cu"))
            .any(|n| fs::metadata(Path::new("csrc").join(n)).and_then(|m| m.modified()).map_or(true, |t| t > om))
    };
    let mut children = Vec::new();
    for (s, obj) in srcs.iter().zip(objs.iter()) {
        let up_to_date =
            match (fs::metadata(obj).and_then(|m| m.modified()).ok(), fs::metadata(s).and_then(|m| m.modified()).ok()) {
                (Some(om), Some(sm)) => om >= sm && hdr_mtime.is_none_or(|hm| om >= hm) && !included_newer(s, om),
                _ => false,
            };
        if up_to_date {
            continue;
        }
        let mut cmd = Command::new(&nvcc);
        if let Some(orig_path) = env::var_os("PATH") {
            let mut paths = vec![cuda.join("bin")];
            paths.extend(env::split_paths(&orig_path));
            if let Ok(joined) = env::join_paths(paths) {
                cmd.env("PATH", joined);
            }
        } else {
            cmd.env("PATH", cuda.join("bin"));
        }
        cmd.args(NVCC_FLAGS).arg(format!("-arch={arch}"));
        cmd.args(extra_flags(s)).arg("-o").arg(obj).arg(s);
        let child = cmd.spawn().expect("spawn nvcc");
        children.push((s.clone(), child));
    }
    for (s, mut child) in children {
        let st = child.wait().expect("wait nvcc");
        assert!(st.success(), "nvcc failed on {}", s.display());
    }
    let lib = out.join("libgbkernels.a");
    let _ = fs::remove_file(&lib);
    let st = Command::new("ar").arg("crs").arg(&lib).args(&objs).status().expect("run ar");
    assert!(st.success(), "ar failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=gbkernels");
    println!("cargo:rustc-link-search=native={}", cuda.join("lib64").display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
