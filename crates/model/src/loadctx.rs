//! One entry point for every weight upload of a boot.
//!
//! The target, draft and vision loaders declare each tensor here instead of reading and uploading it themselves. In
//! **pipeline** mode (the default) a declaration allocates a view in one arena and registers a copy with
//! [`gb_cuda::loader::Loader`]; [`LoadCtx::finish`] then runs every copy of every component in one pipelined pass, and the
//! buffers are valid only after it returns. In **serial** mode (`REXL3_LOAD=serial`) a declaration reads, converts and
//! uploads at once with its own `cudaMalloc`, exactly as the loader did before the pipeline existed; it stays as the
//! reference the pipeline is compared with (device contents must be identical) and as a fallback.
//!
//! Both modes call the same conversion code for everything that is not a plain copy, so they produce the same bytes by
//! construction; `tests` below load a synthetic pack both ways and compare every device buffer.

use crate::error::ModelError;
use crate::pack::Pack;
use gb_cuda::arena::Arena;
use gb_cuda::exl3_linear::Exl3Weights;
use gb_cuda::loader::{LoadError, Loader, Options, Report, Xform};
use gb_cuda::DevBuf;
use gb_exl3::f16::f64_to_f16;
use gb_exl3::loadstats::{self, Stage};
use std::io;
use std::sync::Arc;

// the device wants the file's little-endian bytes as they are
const _: () = assert!(cfg!(target_endian = "little"), "the weight loader copies little-endian file bytes unchanged");

/// Size of one arena block: big enough that the whole model is a couple of dozen `cudaMalloc`s, small enough that the
/// slack left by an oversized tensor stays small.
const ARENA_BLOCK_BYTES: usize = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadMode {
    Pipeline,
    Serial,
}

impl LoadMode {
    /// `REXL3_LOAD=serial` selects the reference loader; anything else the pipeline.
    pub fn from_env() -> Self {
        match std::env::var("REXL3_LOAD").ok().as_deref() {
            Some("serial") => LoadMode::Serial,
            _ => LoadMode::Pipeline,
        }
    }
}

impl From<LoadError> for ModelError {
    fn from(e: LoadError) -> Self {
        match e {
            LoadError::Cuda(c) => ModelError::Cuda(c),
            LoadError::Io(m) => ModelError::Pack(crate::pack::PackError::Io(m)),
            LoadError::Plan(m) => ModelError::Config(m),
        }
    }
}

fn io_err(e: io::Error) -> ModelError {
    ModelError::Pack(crate::pack::PackError::Io(e.to_string()))
}

fn bytes_of_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_of_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Append a linear layer's buffers to a listing, named `{name}.{part}` (for comparing two loads of one pack).
pub fn push_linear<'a>(v: &mut Vec<(String, &'a DevBuf)>, name: String, w: &'a Exl3Weights) {
    for (part, b) in w.buffers() {
        v.push((format!("{name}.{part}"), b));
    }
}

pub struct LoadCtx<'p> {
    mode: LoadMode,
    options: Options,
    arena: Arc<Arena>,
    loader: Loader<'p>,
}

impl<'p> LoadCtx<'p> {
    /// A context with the loader options from the environment (`REXL3_LOAD_WORKERS`, `_SLOTS`, `_SPAN_MIB`, `_DIRECT`).
    pub fn new(mode: LoadMode) -> Self {
        Self::with_options(mode, Options::from_env())
    }

    pub fn with_options(mode: LoadMode, options: Options) -> Self {
        Self { mode, options, arena: Arena::new(ARENA_BLOCK_BYTES), loader: Loader::new() }
    }

    pub fn from_env() -> Self {
        Self::new(LoadMode::from_env())
    }

    /// [`LoadCtx::from_env`] for a load whose carves are known to sum to at most `bytes` (each rounded up to `gb_cuda::arena::ALIGN`): its
    /// arena is one block of that size (`Arena::exact`), so the tail of a fixed 1 GiB block is not stranded for the process's life.
    pub fn exact_from_env(bytes: usize) -> Self {
        Self { arena: Arena::exact(bytes), ..Self::from_env() }
    }

    pub fn mode(&self) -> LoadMode {
        self.mode
    }

    /// A buffer of `bytes` bytes to be filled by a copy declared next: a view in pipeline mode, a plain allocation in serial
    /// mode (where the caller uploads into it at once).
    fn buffer(&self, bytes: usize) -> Result<DevBuf, ModelError> {
        Ok(match self.mode {
            LoadMode::Pipeline => self.arena.alloc(bytes)?,
            LoadMode::Serial => DevBuf::alloc(bytes)?,
        })
    }

    /// A plain 16-bit tensor (`dtype` is `"BF16"` or `"F16"`), its bytes as stored. Serial mode is
    /// `DevBuf::from_slice(&pack.bf16(..)/f16(..))`.
    pub fn raw16(&mut self, pack: &'p Pack, name: &str, dtype: &'static str, shape: &[usize]) -> Result<DevBuf, ModelError> {
        let e = pack.entry(name, dtype, shape)?;
        if self.mode == LoadMode::Serial {
            let v = if dtype == "BF16" { pack.bf16(name, shape)? } else { pack.f16(name, shape)? };
            return Ok(DevBuf::from_slice(&v)?);
        }
        let len = (e.nbytes() / 2 * 2) as usize; // an odd trailing byte is ignored, as the serial reader does
        let dst = self.buffer(len)?;
        let (path, off) = pack.ck.tensor_location(e).map_err(io_err)?;
        let file = self.loader.file(&path);
        self.loader.copy_file(&dst, file, off, len, Xform::None)?;
        Ok(dst)
    }

    /// A bf16 tensor narrowed to fp16 on a loader thread.
    pub fn f16_from_bf16(&mut self, pack: &'p Pack, name: &str, shape: &[usize]) -> Result<DevBuf, ModelError> {
        let e = pack.entry(name, "BF16", shape)?;
        if self.mode == LoadMode::Serial {
            return Ok(DevBuf::from_slice(&pack.f16_from_bf16(name, shape)?)?);
        }
        let len = (e.nbytes() / 2 * 2) as usize;
        let dst = self.buffer(len)?;
        let (name, shape) = (name.to_string(), shape.to_vec());
        self.loader.copy_compute(&dst, len, move || {
            pack.f16_from_bf16(&name, &shape).map(|v| bytes_of_u16(&v)).map_err(|e| io::Error::other(e.to_string()))
        })?;
        Ok(dst)
    }

    /// A bf16 tensor widened to fp32 (see [`Pack::f32_from_bf16`]); done on a loader thread, it is a few KiB.
    pub fn f32_from_bf16(&mut self, pack: &'p Pack, name: &str, shape: &[usize], add_one: bool) -> Result<DevBuf, ModelError> {
        let e = pack.entry(name, "BF16", shape)?;
        if self.mode == LoadMode::Serial {
            return Ok(DevBuf::from_slice(&pack.f32_from_bf16(name, shape, add_one)?)?);
        }
        let len = (e.nbytes() / 2) as usize * 4;
        let dst = self.buffer(len)?;
        let (name, shape) = (name.to_string(), shape.to_vec());
        self.loader.copy_compute(&dst, len, move || {
            pack.f32_from_bf16(&name, &shape, add_one).map(|v| bytes_of_f32(&v)).map_err(|e| io::Error::other(e.to_string()))
        })?;
        Ok(dst)
    }

    /// `len` bytes that `f` computes on a loader thread (for tensors that need a rearrangement); serial mode runs `f` now.
    pub fn computed(
        &mut self,
        len: usize,
        f: impl Fn() -> Result<Vec<u8>, ModelError> + Send + Sync + 'p,
    ) -> Result<DevBuf, ModelError> {
        if self.mode == LoadMode::Serial {
            let v = f()?;
            return Ok(DevBuf::from_slice(&v)?);
        }
        let dst = self.buffer(len)?;
        self.loader.copy_compute(&dst, len, move || f().map_err(|e| io::Error::other(e.to_string())))?;
        Ok(dst)
    }

    /// One EXL3 group of `pack`. Serial mode is `Exl3Weights::from_loaded(&pack.group(prefix)?)`.
    pub fn group(&mut self, pack: &'p Pack, prefix: &str) -> Result<Exl3Weights, ModelError> {
        if self.mode == LoadMode::Serial {
            let g = pack.group(prefix)?;
            let w = Exl3Weights::from_loaded(&g)?;
            let t = loadstats::timer(Stage::Free, (g.trellis.len() * 2) as u64);
            drop(g);
            drop(t);
            return Ok(w);
        }
        let parts = pack.ck.group_parts(prefix).map_err(io_err)?;
        let g = parts.group;
        let bits = Exl3Weights::bits_of(g)?;
        let trellis_len = parts.trellis.nbytes() as usize;
        if trellis_len != (g.k / 16) * (g.n / 16) * 16 * bits * 2 {
            return Err(gb_cuda::CudaError(1).into());
        }
        let trellis = self.buffer(trellis_len)?;
        let (path, off) = pack.ck.tensor_location(parts.trellis).map_err(io_err)?;
        let file = self.loader.file(&path);
        self.loader.copy_file(&trellis, file, off, trellis_len, Xform::None)?;
        let suh = self.scale(pack, parts.in_scales, g.k)?;
        let svh = self.scale(pack, parts.out_scales, g.n)?;
        let bias = match parts.bias {
            Some(e) => {
                let dst = self.buffer(g.n * 2)?;
                self.loader.copy_compute(&dst, g.n * 2, move || {
                    let f = pack.ck.read_floats(e)?;
                    let _t = loadstats::timer(Stage::Convert, f.len() as u64 * 8);
                    Ok(bytes_of_u16(&f.iter().map(|&x| f64_to_f16(x)).collect::<Vec<_>>()))
                })?;
                Some(dst)
            }
            None => None,
        };
        Ok(Exl3Weights::from_parts(g, trellis, suh, svh, bias)?)
    }

    /// A scale vector of `n` fp16 values: fp16 in the file is mapped through the canonicalising table (what the serial
    /// loader's `f16 -> f64 -> f16` does), packed signs go through the serial loader's own code.
    fn scale(&mut self, pack: &'p Pack, e: &'p gb_exl3::checkpoint::Entry, n: usize) -> Result<DevBuf, ModelError> {
        let dst = self.buffer(n * 2)?;
        if e.dtype == "F16" && e.nbytes() as usize == n * 2 {
            let (path, off) = pack.ck.tensor_location(e).map_err(io_err)?;
            let file = self.loader.file(&path);
            self.loader.copy_file(&dst, file, off, n * 2, Xform::CanonF16)?;
        } else {
            self.loader.copy_compute(&dst, n * 2, move || {
                let f = pack.ck.read_scale(e)?;
                let _t = loadstats::timer(Stage::Convert, f.len() as u64 * 8);
                Ok(bytes_of_u16(&f.iter().map(|&x| f64_to_f16(x)).collect::<Vec<_>>()))
            })?;
        }
        Ok(dst)
    }

    /// Bytes the declared copies will move.
    pub fn queued_bytes(&self) -> u64 {
        self.loader.queued_bytes()
    }

    /// What the arena has carved so far: `(bytes carved, alignment padding included; bytes cudaMalloced for its blocks; blocks)`.
    /// All zero in serial mode, which allocates each buffer itself. Read it before [`LoadCtx::finish`] (which consumes the context):
    /// the pipeline's declarations are the carves.
    pub fn arena_stats(&self) -> (u64, u64, usize) {
        (self.arena.used_bytes() as u64, self.arena.reserved_bytes() as u64, self.arena.block_count())
    }

    /// Run every declared copy. In pipeline mode the buffers handed out so far are valid only after this returns.
    pub fn finish(self) -> Result<Option<Report>, ModelError> {
        if self.mode == LoadMode::Serial {
            gb_engine::boot::fact("load_mode", "serial");
            return Ok(None);
        }
        let report = self.loader.run(&self.options)?;
        loadstats::add_for(
            loadstats::Component::All,
            Stage::Wall,
            std::time::Duration::from_secs_f64(report.seconds),
            report.bytes,
        );
        gb_engine::boot::fact("load_mode", "pipeline");
        gb_engine::boot::fact("load_workers", report.workers.to_string());
        gb_engine::boot::fact("load_staging_mib", format!("{:.0}", report.staging_bytes as f64 / (1u64 << 20) as f64));
        gb_engine::boot::fact("load_arena_blocks", self.arena.block_count().to_string());
        eprintln!(
            "rexl3 event=load_pipeline gib={:.2} seconds={:.3} workers={} spans={} direct_spans={} staging_mib={:.0} arena_blocks={} arena_used_gib={:.3} arena_reserved_gib={:.3}",
            report.bytes as f64 / (1u64 << 30) as f64,
            report.seconds,
            report.workers,
            report.spans,
            report.direct_spans,
            report.staging_bytes as f64 / (1u64 << 20) as f64,
            self.arena.block_count(),
            self.arena.used_bytes() as f64 / (1u64 << 30) as f64,
            self.arena.reserved_bytes() as f64 / (1u64 << 30) as f64,
        );
        gb_engine::boot::fact(
            "load_arena_slack_mib",
            format!("{:.0}", (self.arena.reserved_bytes() - self.arena.used_bytes()) as f64 / (1u64 << 20) as f64),
        );
        Ok(Some(report))
    }
}
