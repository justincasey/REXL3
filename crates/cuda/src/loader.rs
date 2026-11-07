//! The pipelined weight loader: file bytes to device memory without a tensor-sized host buffer, without a `cudaMalloc` or a
//! `cudaMemcpy` per tensor, and with the next read overlapping the previous copy.
//!
//! A load is declared first and executed once. The caller registers copies (`copy_file`: file range to device address,
//! optionally through an fp16 canonicalising table; `copy_compute`: bytes a closure produces, for the few small tensors that
//! need CPU conversion) against [`DevBuf`] views of an [`crate::arena::Arena`], then [`Loader::run`] does all of it.
//!
//! `run` sorts the file ranges by (file, offset) and cuts them into spans of at most `span_bytes`. Small tensors that sit
//! close together in a file share a span (one read, several copies), a large tensor becomes consecutive spans of its own.
//! Worker threads claim spans in that order from one counter, so the disk sees a sequential stream split over a few
//! readers. Each worker owns `slots` staging buffers (plain aligned memory: on this platform a copy from pageable memory
//! runs as fast as one from pinned memory, and `cudaHostAlloc` costs about 0.6 ms per MiB, see `tools/load_ceiling.cu`),
//! one non-blocking stream and one event per slot: it reads a span into a slot, queues the copies of its items on its
//! stream, records the slot's event, and moves to the next slot, so the read of span n+1 overlaps the copies of span n. A
//! slot is refilled only after its event has completed. Spans whose pages are mostly not in the page cache are read with
//! `O_DIRECT` (no page-cache pollution, and large requests keep the SSD busy: measured about 10 GB/s against 1-6 GB/s for
//! buffered reads on a cold file); warm spans are read buffered.
//!
//! The staging memory (workers x slots x span) is allocated when `run` starts and freed before it returns, so it never
//! counts against the memory the engine plans from.
//!
//! Byte-exactness: every copy moves file bytes unchanged, except `Xform::CanonF16` (a table lookup that equals the
//! `f16 -> f64 -> f16` round trip the serial loader performs) and `copy_compute` (which runs the serial loader's own code).

use crate::rt::{Block, CudaError, DevBuf, Event, Stream};
use gb_exl3::loadstats::{self, Component, Stage};
use std::alloc::Layout;
use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// Page size assumed for direct reads and residency checks (`O_DIRECT` needs 512 B, 4 KiB is safe on every device).
const PAGE: usize = 4096;

/// The VM page size (4 KiB on the GB10, 16 KiB on Apple Silicon); `PAGE` stays the 4 KiB direct-I/O alignment.
fn vm_page_size() -> usize {
    static SIZE: OnceLock<usize> = OnceLock::new();
    // SAFETY: sysconf has no preconditions
    *SIZE.get_or_init(|| match unsafe { libc::sysconf(libc::_SC_PAGESIZE) } {
        n if n > 0 => n as usize,
        _ => PAGE,
    })
}

#[derive(Debug)]
pub enum LoadError {
    Io(String),
    Cuda(CudaError),
    Plan(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(m) | LoadError::Plan(m) => write!(f, "{m}"),
            LoadError::Cuda(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<CudaError> for LoadError {
    fn from(e: CudaError) -> Self {
        LoadError::Cuda(e)
    }
}

/// What a copy does to the bytes it moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Xform {
    /// Nothing: the file's bytes arrive unchanged.
    None,
    /// Every 16-bit little-endian element goes through [`gb_exl3::f16::canonical_f16_table`].
    CanonF16,
}

/// When to read with `O_DIRECT`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectMode {
    /// Per span: direct when fewer than half of its pages are in the page cache.
    Auto,
    Always,
    Never,
}

#[derive(Clone, Debug)]
pub struct Options {
    /// Reader threads (each with its own stream and staging slots).
    pub workers: usize,
    /// Staging buffers per worker (2 lets a read overlap the previous span's copies).
    pub slots: usize,
    /// Largest read, in bytes.
    pub span_bytes: usize,
    /// Spans merge two ranges in one file when the gap between them is at most this many bytes.
    pub gap_bytes: u64,
    pub direct: DirectMode,
}

impl Options {
    /// Defaults, overridden by `REXL3_LOAD_WORKERS`, `REXL3_LOAD_SLOTS`, `REXL3_LOAD_SPAN_MIB` and
    /// `REXL3_LOAD_DIRECT` (`auto`, `1`/`always`, `0`/`never`).
    pub fn from_env() -> Self {
        let num = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n >= 1);
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        let direct = match std::env::var("REXL3_LOAD_DIRECT").ok().as_deref() {
            Some("1") | Some("always") => DirectMode::Always,
            Some("0") | Some("never") => DirectMode::Never,
            _ => DirectMode::Auto,
        };
        Self {
            workers: num("REXL3_LOAD_WORKERS").unwrap_or_else(|| cores.clamp(1, 8)),
            slots: num("REXL3_LOAD_SLOTS").unwrap_or(2),
            span_bytes: num("REXL3_LOAD_SPAN_MIB").unwrap_or(8) << 20,
            gap_bytes: 256 << 10,
            direct,
        }
    }

    /// Host memory the staging buffers occupy while a load runs (a direct read needs two pages of slack per buffer).
    pub fn staging_bytes(&self) -> usize {
        self.workers * self.slots * (self.span_bytes + 2 * PAGE)
    }
}

/// What every load of this process read, summed (the boot's page-cache breakdown: a rank that read its shard from a cold cache reads
/// from the SSD, a warm one copies from memory; `docs/NEMOTRON_TP_OPERATIONS.md`, cold boot).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Totals {
    /// Bytes read from the pack files.
    pub bytes: u64,
    /// Of those, the bytes whose pages were NOT in the page cache when the span was read (by `mincore`, per span).
    pub cold_bytes: u64,
    /// Bytes read with `O_DIRECT` (cold spans under `DirectMode::Auto`).
    pub direct_bytes: u64,
    /// Seconds the reads took, summed over the workers (they overlap: this is not wall time).
    pub read_s: f64,
}

impl Totals {
    /// What was read between `before` and `self`.
    pub fn since(&self, before: &Totals) -> Totals {
        Totals {
            bytes: self.bytes.saturating_sub(before.bytes),
            cold_bytes: self.cold_bytes.saturating_sub(before.cold_bytes),
            direct_bytes: self.direct_bytes.saturating_sub(before.direct_bytes),
            read_s: (self.read_s - before.read_s).max(0.0),
        }
    }

    /// `warm` (at least 90 % of the bytes were cached), `cold` (at most 10 %), else `mixed`; `none` when nothing was read.
    pub fn cache_state(&self) -> &'static str {
        if self.bytes == 0 {
            return "none";
        }
        let cold = self.cold_bytes as f64 / self.bytes as f64;
        if cold <= 0.10 {
            "warm"
        } else if cold >= 0.90 {
            "cold"
        } else {
            "mixed"
        }
    }
}

struct TotalCounters {
    bytes: std::sync::atomic::AtomicU64,
    cold_bytes: std::sync::atomic::AtomicU64,
    direct_bytes: std::sync::atomic::AtomicU64,
    read_ns: std::sync::atomic::AtomicU64,
}

impl TotalCounters {
    fn note(&self, len: u64, resident: f64, direct: bool, took: std::time::Duration) {
        let cold = ((1.0 - resident.clamp(0.0, 1.0)) * len as f64).round() as u64;
        self.bytes.fetch_add(len, Ordering::Relaxed);
        self.cold_bytes.fetch_add(cold.min(len), Ordering::Relaxed);
        if direct {
            self.direct_bytes.fetch_add(len, Ordering::Relaxed);
        }
        self.read_ns.fetch_add(took.as_nanos() as u64, Ordering::Relaxed);
    }
}

static TOTALS: TotalCounters = TotalCounters {
    bytes: std::sync::atomic::AtomicU64::new(0),
    cold_bytes: std::sync::atomic::AtomicU64::new(0),
    direct_bytes: std::sync::atomic::AtomicU64::new(0),
    read_ns: std::sync::atomic::AtomicU64::new(0),
};

/// The process's read totals so far (take one before a load and one after, and [`Totals::since`]).
pub fn totals() -> Totals {
    Totals {
        bytes: TOTALS.bytes.load(Ordering::Relaxed),
        cold_bytes: TOTALS.cold_bytes.load(Ordering::Relaxed),
        direct_bytes: TOTALS.direct_bytes.load(Ordering::Relaxed),
        read_s: TOTALS.read_ns.load(Ordering::Relaxed) as f64 / 1e9,
    }
}

/// What a load did.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// Bytes copied to the device.
    pub bytes: u64,
    pub jobs: usize,
    pub spans: usize,
    /// Spans read with `O_DIRECT`.
    pub direct_spans: usize,
    pub workers: usize,
    /// Host staging memory held while the load ran.
    pub staging_bytes: usize,
    pub seconds: f64,
}

// ------------------------------------------------------------------------------------------------------------ planning

/// A file range to be copied (the planner's view of a job).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Req {
    pub file: usize,
    pub off: u64,
    pub len: usize,
}

/// One piece of a job inside a span: bytes `rel..rel+len` of the span's read go to `job`'s bytes `job_off..job_off+len`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    pub job: usize,
    pub job_off: usize,
    pub rel: usize,
    pub len: usize,
}

/// One read: `len` bytes of `file` from `start`, and what to copy out of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    pub file: usize,
    pub start: u64,
    pub len: usize,
    pub items: Vec<Item>,
}

/// Cut the requested ranges into reads of at most `span_bytes`, in (file, offset) order. A range of at most half a span is
/// never split (its whole content arrives in one read, which an element-wise transform relies on) and joins the open span
/// when it starts within `gap_bytes` of its end and the span stays within `span_bytes`; a larger range becomes consecutive
/// spans of exactly `span_bytes` (the last one shorter), and a short last piece may take following small ranges. Empty
/// ranges produce nothing. Every byte of every range is covered once, by items that name the job it belongs to.
pub fn plan_spans(reqs: &[Req], span_bytes: usize, gap_bytes: u64) -> Vec<Span> {
    let mut order: Vec<usize> = (0..reqs.len()).filter(|&i| reqs[i].len > 0).collect();
    order.sort_by_key(|&i| (reqs[i].file, reqs[i].off, i));
    let atom_max = span_bytes / 2;
    let mut spans: Vec<Span> = Vec::new();
    for &i in &order {
        let r = reqs[i];
        if r.len <= atom_max {
            let joins = spans.last().is_some_and(|s| {
                let end = s.start + s.len as u64;
                s.file == r.file
                    && r.off >= s.start
                    && r.off <= end.saturating_add(gap_bytes)
                    && (r.off + r.len as u64).max(end) - s.start <= span_bytes as u64
            });
            if !joins {
                spans.push(Span { file: r.file, start: r.off, len: 0, items: Vec::new() });
            }
            let s = spans.last_mut().expect("a span was just pushed");
            let rel = (r.off - s.start) as usize;
            s.items.push(Item { job: i, job_off: 0, rel, len: r.len });
            s.len = s.len.max(rel + r.len);
        } else {
            let mut pos = 0;
            while pos < r.len {
                let n = span_bytes.min(r.len - pos);
                spans.push(Span {
                    file: r.file,
                    start: r.off + pos as u64,
                    len: n,
                    items: vec![Item { job: i, job_off: pos, rel: 0, len: n }],
                });
                pos += n;
            }
        }
    }
    spans
}

/// The row shape of a strided job ([`Loader::copy_file_strided`]): its source span is rows of `width` bytes `src_pitch` apart (the
/// last row ends the span), and they land densely, row `r` at `r * width` of the destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stride {
    pub width: usize,
    pub src_pitch: usize,
}

/// One host-to-device copy of a piece of a strided job: `rows` rows of `len` bytes, row `r` from `src + r * src_pitch` of the piece's
/// bytes to `dst + r * width` of the job's destination (a partial row is one row of fewer than `width` bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowCopy {
    pub src: usize,
    pub dst: usize,
    pub len: usize,
    pub rows: usize,
}

/// The copies that move the row bytes of the piece `[job_off, job_off + len)` of a strided job's source span: at most a partial head
/// row (the piece starts inside a row), ONE copy of every whole row inside the piece, and a partial tail row (it ends inside one). The
/// gap bytes between rows are never copied. Pieces that tile the span move every row byte exactly once, byte `c` of row `r` to
/// destination byte `r * width + c` (`plan_spans` may cut a span anywhere, so a row may straddle two pieces).
pub fn strided_copies(job_off: usize, len: usize, s: Stride) -> [Option<RowCopy>; 3] {
    let (w, p) = (s.width, s.src_pitch);
    let mut out = [None; 3];
    if len == 0 || w == 0 {
        return out;
    }
    let (a, b) = (job_off, job_off + len);
    // the row byte range of row `r` inside the piece, as one copy
    let part = |r: usize| {
        let (lo, hi) = ((r * p).max(a), (r * p + w).min(b));
        (lo < hi).then(|| RowCopy { src: lo - a, dst: r * w + (lo - r * p), len: hi - lo, rows: 1 })
    };
    let mut r = a / p;
    if a % p != 0 {
        // the piece starts inside row r (in its bytes or in the gap after them): what of row r it holds, then the next row
        out[0] = part(r);
        r += 1;
    }
    // rows r..whole end inside the piece
    let whole = if b >= w { ((b - w) / p + 1).max(r) } else { r };
    if whole > r {
        out[1] = Some(RowCopy { src: r * p - a, dst: r * w, len: w, rows: whole - r });
    }
    if whole * p < b {
        out[2] = part(whole);
    }
    out
}

// ----------------------------------------------------------------------------------------------------------- the loader

type ComputeFn<'a> = Box<dyn Fn() -> io::Result<Vec<u8>> + Send + Sync + 'a>;

enum Kind<'a> {
    File {
        file: usize,
        off: u64,
        xform: Xform,
        /// `Some` for [`Loader::copy_file_strided`]: the job's `len` is its source span, only the rows' bytes are copied.
        stride: Option<Stride>,
    },
    Compute(ComputeFn<'a>),
    /// Bytes of a file that must equal `want`: read with the spans around them and compared on the host, never copied.
    Check {
        file: usize,
        off: u64,
        want: Vec<u8>,
        what: String,
    },
}

struct Job<'a> {
    /// Device address of the destination, kept alive by `_keep` (0 for a [`Kind::Check`], which has none).
    dst: usize,
    len: usize,
    comp: Component,
    kind: Kind<'a>,
    _keep: Option<Arc<Block>>,
}

/// A declared load. See the module docs.
pub struct Loader<'a> {
    files: Vec<PathBuf>,
    jobs: Vec<Job<'a>>,
}

impl Default for Loader<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Loader<'a> {
    pub fn new() -> Self {
        Self { files: Vec::new(), jobs: Vec::new() }
    }

    /// Register a source file (the same path twice gives the same id).
    pub fn file(&mut self, path: &Path) -> usize {
        match self.files.iter().position(|p| p == path) {
            Some(i) => i,
            None => {
                self.files.push(path.to_path_buf());
                self.files.len() - 1
            }
        }
    }

    fn target(dst: &DevBuf, len: usize) -> Result<Arc<Block>, LoadError> {
        if len > dst.bytes() {
            return Err(LoadError::Plan(format!("a {len}-byte copy into a {}-byte buffer", dst.bytes())));
        }
        dst.block().cloned().ok_or_else(|| LoadError::Plan("the loader writes into arena views only".into()))
    }

    /// Copy `len` bytes of `file` from byte `off` to the start of `dst`, passed through `xform`.
    pub fn copy_file(&mut self, dst: &DevBuf, file: usize, off: u64, len: usize, xform: Xform) -> Result<(), LoadError> {
        let keep = Self::target(dst, len)?;
        if file >= self.files.len() {
            return Err(LoadError::Plan(format!("unknown file id {file}")));
        }
        if xform == Xform::CanonF16 && len % 2 != 0 {
            return Err(LoadError::Plan("an fp16 transform needs an even byte count".into()));
        }
        self.jobs.push(Job {
            dst: dst.ptr() as usize,
            len,
            comp: loadstats::component(),
            kind: Kind::File { file, off, xform, stride: None },
            _keep: Some(keep),
        });
        Ok(())
    }

    /// Copy the rows of a strided source to the start of `dst`, densely and unchanged: `len` bytes of `file` from `off` are rows of
    /// `width` bytes, `src_pitch` apart (`len = (rows - 1) * src_pitch + width`), and row `r` lands at `r * width` (a column cut of an
    /// EXL3 trellis: the rank's tiles of every k-tile row). The read covers the whole span, gaps included (spans are planned on it as on
    /// any range); each staged piece then goes over in at most three copies (`strided_copies`), never one a row.
    pub fn copy_file_strided(
        &mut self,
        dst: &DevBuf,
        file: usize,
        off: u64,
        len: usize,
        width: usize,
        src_pitch: usize,
    ) -> Result<(), LoadError> {
        if width == 0 || src_pitch < width || len < width || (len - width) % src_pitch != 0 {
            return Err(LoadError::Plan(format!(
                "a strided copy of {len} bytes in rows of {width} bytes {src_pitch} apart is not whole rows"
            )));
        }
        let keep = Self::target(dst, ((len - width) / src_pitch + 1) * width)?;
        if file >= self.files.len() {
            return Err(LoadError::Plan(format!("unknown file id {file}")));
        }
        self.jobs.push(Job {
            dst: dst.ptr() as usize,
            len,
            comp: loadstats::component(),
            kind: Kind::File { file, off, xform: Xform::None, stride: Some(Stride { width, src_pitch }) },
            _keep: Some(keep),
        });
        Ok(())
    }

    /// Require that the bytes of `file` at `off` equal `want`, checked inside the reads the loader makes anyway (the codebook
    /// marker of an EXL3 group sits between its scales and its trellis, so its check costs no read of its own). A mismatch fails
    /// [`Loader::run`] with `what`. The bytes count in no report total: nothing is copied.
    pub fn check_file(&mut self, file: usize, off: u64, want: &[u8], what: impl Into<String>) -> Result<(), LoadError> {
        if file >= self.files.len() {
            return Err(LoadError::Plan(format!("unknown file id {file}")));
        }
        self.jobs.push(Job {
            dst: 0,
            len: want.len(),
            comp: loadstats::component(),
            kind: Kind::Check { file, off, want: want.to_vec(), what: what.into() },
            _keep: None,
        });
        Ok(())
    }

    /// Copy the `len` bytes `f` returns (it runs on a loader thread, once) to the start of `dst`.
    pub fn copy_compute(
        &mut self,
        dst: &DevBuf,
        len: usize,
        f: impl Fn() -> io::Result<Vec<u8>> + Send + Sync + 'a,
    ) -> Result<(), LoadError> {
        let keep = Self::target(dst, len)?;
        self.jobs.push(Job {
            dst: dst.ptr() as usize,
            len,
            comp: loadstats::component(),
            kind: Kind::Compute(Box::new(f)),
            _keep: Some(keep),
        });
        Ok(())
    }

    /// Bytes declared so far (the bytes to be read for copies: a strided job counts its source span, a [`Loader::check_file`]
    /// copies nothing).
    pub fn queued_bytes(&self) -> u64 {
        self.jobs.iter().filter(|j| !matches!(j.kind, Kind::Check { .. })).map(|j| j.len as u64).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Execute every declared copy. On return all of them have completed on the device, or an error is returned (the
    /// destinations are then partly written and must not be used).
    pub fn run(self, o: &Options) -> Result<Report, LoadError> {
        let t0 = Instant::now();
        let atom_max = o.span_bytes / 2;
        for j in &self.jobs {
            if matches!(j.kind, Kind::File { xform: Xform::CanonF16, .. }) && j.len > atom_max {
                return Err(LoadError::Plan(format!("a {}-byte fp16-transformed tensor does not fit one staging read", j.len)));
            }
        }
        let reqs: Vec<Req> = self
            .jobs
            .iter()
            .map(|j| match &j.kind {
                Kind::File { file, off, .. } | Kind::Check { file, off, .. } => Req { file: *file, off: *off, len: j.len },
                Kind::Compute(_) => Req { file: usize::MAX, off: 0, len: 0 },
            })
            .collect();
        let spans = plan_spans(&reqs, o.span_bytes, o.gap_bytes);
        let computes: Vec<usize> =
            (0..self.jobs.len()).filter(|&i| matches!(self.jobs[i].kind, Kind::Compute(_)) && self.jobs[i].len > 0).collect();
        let units = spans.len() + computes.len();
        let mut report = Report {
            bytes: self.queued_bytes(),
            jobs: self.jobs.len(),
            spans: spans.len(),
            workers: 0,
            staging_bytes: 0,
            ..Default::default()
        };
        if units == 0 {
            return Ok(report);
        }
        let src: Vec<SrcFile> = self.files.iter().map(|p| SrcFile::open(p)).collect::<Result<_, _>>()?;
        let workers = o.workers.clamp(1, units);
        report.workers = workers;
        report.staging_bytes = workers * o.slots.max(1) * (o.span_bytes + 2 * PAGE);
        let next = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        let first_error: Mutex<Option<LoadError>> = Mutex::new(None);
        let direct_spans = AtomicUsize::new(0);
        let ctx = Shared {
            o,
            jobs: &self.jobs,
            spans: &spans,
            computes: &computes,
            src: &src,
            next: &next,
            failed: &failed,
            direct_spans: &direct_spans,
        };
        std::thread::scope(|s| {
            // a worker that fails stops the others at their next span, without waiting for the joins below
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    s.spawn(|| {
                        let r = ctx.worker();
                        if r.is_err() {
                            ctx.failed.store(true, Ordering::Relaxed);
                        }
                        r
                    })
                })
                .collect();
            for h in handles {
                let res = h.join().unwrap_or_else(|_| Err(LoadError::Io("a loader thread panicked".into())));
                if let Err(e) = res {
                    failed.store(true, Ordering::Relaxed);
                    let mut slot = first_error.lock().unwrap_or_else(|e| e.into_inner());
                    if slot.is_none() {
                        *slot = Some(e);
                    }
                }
            }
        });
        if let Some(e) = first_error.into_inner().unwrap_or_else(|e| e.into_inner()) {
            return Err(e);
        }
        report.direct_spans = direct_spans.load(Ordering::Relaxed);
        report.seconds = t0.elapsed().as_secs_f64();
        Ok(report)
    }
}

// ------------------------------------------------------------------------------------------------------------ the files

/// A page-aligned host buffer (the memory `O_DIRECT` reads need), zero-filled, freed on drop.
struct AlignedBuf {
    ptr: *mut u8,
    cap: usize,
}

unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    fn new(cap: usize) -> Self {
        let layout = Layout::from_size_align(cap.max(PAGE), PAGE).expect("a valid staging layout");
        // SAFETY: the layout has a non-zero size
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self { ptr, cap: layout.size() }
    }
    fn slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` points at `cap` initialised bytes owned by this buffer
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.cap) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with the same layout
        unsafe { std::alloc::dealloc(self.ptr, Layout::from_size_align(self.cap, PAGE).expect("a valid staging layout")) }
    }
}

/// A read-only mapping used only to ask `mincore` which pages of the file are in the page cache (never dereferenced).
struct Map {
    addr: *mut c_void,
    len: usize,
}

unsafe impl Send for Map {}
unsafe impl Sync for Map {}

impl Drop for Map {
    fn drop(&mut self) {
        // SAFETY: unmapping the range `mmap` returned
        unsafe {
            libc::madvise(self.addr, self.len, libc::MADV_DONTNEED);
            libc::munmap(self.addr, self.len);
        }
    }
}

struct SrcFile {
    path: PathBuf,
    buffered: File,
    len: u64,
    direct: OnceLock<Option<File>>,
    /// Set when a direct read failed (a filesystem that refuses `O_DIRECT`): the rest of the file is read buffered.
    direct_refused: AtomicBool,
    map: OnceLock<Option<Map>>,
}

impl SrcFile {
    fn open(path: &Path) -> Result<Self, LoadError> {
        let ctx = |e: io::Error| LoadError::Io(format!("{}: {e}", path.display()));
        let buffered = File::open(path).map_err(ctx)?;
        let len = buffered.metadata().map_err(ctx)?.len();
        Ok(Self {
            path: path.to_path_buf(),
            buffered,
            len,
            direct: OnceLock::new(),
            direct_refused: AtomicBool::new(false),
            map: OnceLock::new(),
        })
    }

    #[cfg(target_os = "linux")]
    fn direct_file(&self) -> Option<&File> {
        self.direct
            .get_or_init(|| std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&self.path).ok())
            .as_ref()
    }

    /// macOS has no `O_DIRECT`; `F_NOCACHE` on an ordinary descriptor is its page-cache bypass. Unlike `O_DIRECT` it does not
    /// require aligned buffers, so the aligned reads issued for Linux remain valid here.
    #[cfg(target_os = "macos")]
    fn direct_file(&self) -> Option<&File> {
        self.direct
            .get_or_init(|| {
                let f = std::fs::OpenOptions::new().read(true).open(&self.path).ok()?;
                // SAFETY: fcntl on a descriptor this function owns; F_NOCACHE takes an int flag
                let rc = unsafe { libc::fcntl(std::os::unix::io::AsRawFd::as_raw_fd(&f), libc::F_NOCACHE, 1) };
                (rc != -1).then_some(f)
            })
            .as_ref()
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn direct_file(&self) -> Option<&File> {
        None
    }

    /// Fraction of the pages of `start..start+len` that are in the page cache; 1.0 when it cannot be told.
    fn resident_fraction(&self, start: u64, len: usize, scratch: &mut Vec<u8>) -> f64 {
        let map = self.map.get_or_init(|| {
            if self.len == 0 {
                return None;
            }
            // SAFETY: a private read-only mapping of the whole file; the pages are never touched
            let addr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    self.len as usize,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    std::os::unix::io::AsRawFd::as_raw_fd(&self.buffered),
                    0,
                )
            };
            (addr != libc::MAP_FAILED).then_some(Map { addr, len: self.len as usize })
        });
        let Some(map) = map else { return 1.0 };
        // mincore reports one byte per VM page, which is 16 KiB on Apple Silicon: use the system page size here, not the
        // 4 KiB I/O alignment `PAGE`.
        let vm_page = vm_page_size();
        let first = (start as usize / vm_page) * vm_page;
        let end = ((start as usize + len).div_ceil(vm_page) * vm_page).min(map.len.div_ceil(vm_page) * vm_page);
        if end <= first {
            return 1.0;
        }
        let pages = (end - first) / vm_page;
        scratch.clear();
        scratch.resize(pages, 0);
        // SAFETY: the range lies inside the mapping and `scratch` has one byte per page
        let rc =
            unsafe { libc::mincore((map.addr as *mut u8).add(first) as *mut c_void, end - first, scratch.as_mut_ptr().cast()) };
        if rc != 0 {
            return 1.0;
        }
        scratch.iter().filter(|&&b| b & 1 == 1).count() as f64 / pages as f64
    }
}

/// `read_exact_at` that reports a short file as `UnexpectedEof` and a direct read that returns fewer bytes than asked only
/// when the file ends (the last block).
fn read_full_at(f: &File, buf: &mut [u8], off: u64, need: usize) -> io::Result<()> {
    let mut done = 0;
    while done < buf.len() {
        match f.read_at(&mut buf[done..], off + done as u64) {
            Ok(0) => break,
            Ok(n) => done += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    if done < need {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, format!("read {done} of {need} bytes at offset {off}")));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------------------ workers

struct Shared<'s, 'a> {
    o: &'s Options,
    jobs: &'s [Job<'a>],
    spans: &'s [Span],
    computes: &'s [usize],
    src: &'s [SrcFile],
    next: &'s AtomicUsize,
    failed: &'s AtomicBool,
    direct_spans: &'s AtomicUsize,
}

struct Slot {
    buf: AlignedBuf,
    done: Event,
    pending: bool,
}

/// Synchronises the stream when a worker ends for any reason: the staging buffers must not be freed under queued copies.
struct StreamGuard<'s>(&'s Stream);

impl Drop for StreamGuard<'_> {
    fn drop(&mut self) {
        let _ = self.0.sync();
    }
}

fn plan_err(msg: String) -> LoadError {
    LoadError::Plan(msg)
}

impl Shared<'_, '_> {
    fn worker(&self) -> Result<(), LoadError> {
        let stream = Stream::new_nonblocking()?;
        let _guard = StreamGuard(&stream);
        let cap = self.o.span_bytes + 2 * PAGE;
        let mut slots: Vec<Slot> = (0..self.o.slots.max(1))
            .map(|_| Ok(Slot { buf: AlignedBuf::new(cap), done: Event::new()?, pending: false }))
            .collect::<Result<_, CudaError>>()?;
        let table = gb_exl3::f16::canonical_f16_table();
        let mut resident: Vec<u8> = Vec::new();
        let mut turn = 0usize;
        let mut last = Component::All;
        loop {
            if self.failed.load(Ordering::Relaxed) {
                return Ok(());
            }
            let u = self.next.fetch_add(1, Ordering::Relaxed);
            if u >= self.spans.len() + self.computes.len() {
                break;
            }
            let slot = &mut slots[turn % slots_len(self.o)];
            turn += 1;
            // everything this unit records (including the conversion code of a computed tensor, which times itself) belongs
            // to the component whose data it moves, not to whatever the driving thread is loading by now
            let comp = self.unit_component(u);
            last = comp;
            let _scope = loadstats::scoped_component(comp);
            if slot.pending {
                let t = Instant::now();
                slot.done.sync()?;
                slot.pending = false;
                loadstats::add_for(comp, Stage::Wait, t.elapsed(), 0);
            }
            if u < self.spans.len() {
                self.run_span(&self.spans[u], slot, &stream, table, &mut resident)?;
            } else {
                self.run_compute(self.computes[u - self.spans.len()], slot, &stream)?;
            }
        }
        let t = Instant::now();
        stream.sync()?;
        loadstats::add_for(last, Stage::Wait, t.elapsed(), 0);
        Ok(())
    }

    fn unit_component(&self, u: usize) -> Component {
        if u < self.spans.len() {
            self.spans[u].items.first().map_or(Component::Target, |it| self.jobs[it.job].comp)
        } else {
            self.jobs[self.computes[u - self.spans.len()]].comp
        }
    }

    fn run_span(
        &self,
        sp: &Span,
        slot: &mut Slot,
        stream: &Stream,
        table: &[u16; 65536],
        resident: &mut Vec<u8>,
    ) -> Result<(), LoadError> {
        let comp = self.jobs[sp.items[0].job].comp;
        let f = &self.src[sp.file];
        let end = sp.start + sp.len as u64;
        if end > f.len {
            return Err(LoadError::Io(format!("{}: a tensor ends at byte {end} of a {}-byte file", f.path.display(), f.len)));
        }
        let t = Instant::now();
        // how much of the span the page cache held before it was read: the boot's cold / warm split (`totals`)
        let frac = f.resident_fraction(sp.start, sp.len, resident);
        let want_direct = match self.o.direct {
            DirectMode::Never => false,
            DirectMode::Always => true,
            DirectMode::Auto => frac < 0.5,
        } && !f.direct_refused.load(Ordering::Relaxed);
        let mut base = 0usize;
        let mut done = false;
        if want_direct {
            if let Some(df) = f.direct_file() {
                let a0 = sp.start & !(PAGE as u64 - 1);
                let a1 = (end.div_ceil(PAGE as u64) * PAGE as u64).min(f.len.div_ceil(PAGE as u64) * PAGE as u64);
                let need = (end - a0) as usize;
                let n = (a1 - a0) as usize;
                match read_full_at(df, &mut slot.buf.slice()[..n], a0, need) {
                    Ok(()) => {
                        base = (sp.start - a0) as usize;
                        done = true;
                        self.direct_spans.fetch_add(1, Ordering::Relaxed);
                    }
                    // a filesystem that refuses O_DIRECT reads (EINVAL): fall back for this and every later span of the file
                    Err(e) if e.kind() == io::ErrorKind::InvalidInput || e.raw_os_error() == Some(libc::EINVAL) => {
                        f.direct_refused.store(true, Ordering::Relaxed);
                    }
                    Err(e) => return Err(LoadError::Io(format!("{}: {e}", f.path.display()))),
                }
            }
        }
        if !done {
            read_full_at(&f.buffered, &mut slot.buf.slice()[..sp.len], sp.start, sp.len)
                .map_err(|e| LoadError::Io(format!("{}: {e}", f.path.display())))?;
            #[cfg(target_os = "linux")]
            unsafe {
                let a0 = (sp.start & !(PAGE as u64 - 1)) as libc::off_t;
                let a1 = (end & !(PAGE as u64 - 1)) as libc::off_t;
                if a1 > a0 {
                    libc::posix_fadvise(
                        std::os::unix::io::AsRawFd::as_raw_fd(&f.buffered),
                        a0,
                        a1 - a0,
                        libc::POSIX_FADV_DONTNEED,
                    );
                }
            }
        }
        loadstats::add_for(comp, Stage::Read, t.elapsed(), sp.len as u64);
        TOTALS.note(sp.len as u64, frac, done && want_direct, t.elapsed());
        let data = &mut slot.buf.slice()[base..base + sp.len];
        for it in &sp.items {
            let job = &self.jobs[it.job];
            let region = &mut data[it.rel..it.rel + it.len];
            if let Kind::Check { want, what, off, .. } = &job.kind {
                // a check job is whole inside one span (it is smaller than half a span); a mismatch is a pack that would be decoded
                // wrongly, so it ends the load
                if it.job_off != 0 || it.len != want.len() || region != want.as_slice() {
                    return Err(LoadError::Plan(format!(
                        "{what}: the bytes at offset {off} of {} are {:02x?}, expected {:02x?}",
                        self.src[sp.file].path.display(),
                        region,
                        want
                    )));
                }
                continue;
            }
            if let Kind::File { xform: Xform::CanonF16, .. } = job.kind {
                let t = Instant::now();
                for c in region.chunks_exact_mut(2) {
                    let v = table[u16::from_le_bytes([c[0], c[1]]) as usize];
                    c.copy_from_slice(&v.to_le_bytes());
                }
                loadstats::add_for(job.comp, Stage::Convert, t.elapsed(), it.len as u64);
            }
            let t = Instant::now();
            if let Kind::File { stride: Some(s), .. } = job.kind {
                for c in strided_copies(it.job_off, it.len, s).into_iter().flatten() {
                    // SAFETY: a copy's rows are inside the piece (`region`, held until `slot.done` completes) and land inside the dense
                    // destination of `rows x width` bytes that `_keep` keeps alive (checked at declaration)
                    unsafe {
                        stream.memcpy_h2d_2d_async(
                            (job.dst + c.dst) as *mut c_void,
                            s.width,
                            region.as_ptr().add(c.src),
                            s.src_pitch,
                            c.len,
                            c.rows,
                        )?
                    };
                }
                loadstats::add_for(job.comp, Stage::Memcpy, t.elapsed(), it.len as u64);
                continue;
            }
            // SAFETY: `job.dst + job_off` is inside the destination view `_keep` keeps alive (checked at declaration), and
            // `region` is `it.len` readable bytes of this slot, which is not reused before `slot.done` completes
            unsafe { stream.memcpy_h2d_async((job.dst + it.job_off) as *mut c_void, region.as_ptr(), it.len)? };
            loadstats::add_for(job.comp, Stage::Memcpy, t.elapsed(), it.len as u64);
        }
        let t = Instant::now();
        slot.done.record(stream)?;
        slot.pending = true;
        loadstats::add_for(comp, Stage::Wait, t.elapsed(), 0);
        Ok(())
    }

    fn run_compute(&self, j: usize, slot: &mut Slot, stream: &Stream) -> Result<(), LoadError> {
        let job = &self.jobs[j];
        let Kind::Compute(f) = &job.kind else { return Err(plan_err("a copy job was queued as a compute job".into())) };
        // the closure's own timers (read, convert) record under this unit's scoped component
        let bytes = f().map_err(|e| LoadError::Io(e.to_string()))?;
        if bytes.len() != job.len {
            return Err(plan_err(format!("a computed tensor has {} bytes, its destination {}", bytes.len(), job.len)));
        }
        let t = Instant::now();
        if bytes.len() <= slot.buf.cap {
            slot.buf.slice()[..bytes.len()].copy_from_slice(&bytes);
            // SAFETY: as in `run_span`; the slot is held until its event completes
            unsafe { stream.memcpy_h2d_async(job.dst as *mut c_void, slot.buf.ptr, bytes.len())? };
            slot.done.record(stream)?;
            slot.pending = true;
        } else {
            // larger than a staging buffer: copy straight from the vector and wait, so it can be dropped
            // SAFETY: `bytes` is readable for its length and outlives the synchronisation below
            unsafe { stream.memcpy_h2d_async(job.dst as *mut c_void, bytes.as_ptr(), bytes.len())? };
            stream.sync()?;
        }
        loadstats::add_for(job.comp, Stage::Memcpy, t.elapsed(), bytes.len() as u64);
        Ok(())
    }
}

fn slots_len(o: &Options) -> usize {
    o.slots.max(1)
}

#[cfg(all(test, not(gb_no_cuda)))]
mod check_tests {
    use super::*;
    use crate::arena::Arena;

    /// A file of known bytes: a marker (`mul1`, little endian) at offset 50, copied data around it.
    fn file(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("rexl3-loader-check-{tag}-{}", std::process::id()));
        let mut data: Vec<u8> = (0..200u32).map(|i| (i * 7 + 1) as u8).collect();
        data[50..54].copy_from_slice(&0x83DC_D12Du32.to_le_bytes());
        std::fs::write(&p, data).unwrap();
        p
    }

    #[test]
    fn a_marker_is_checked_inside_the_reads_of_the_copies_around_it_and_a_mismatch_fails_the_load() {
        if !crate::device_available() {
            eprintln!("no CUDA device: not run");
            return;
        }
        let path = file("a");
        let want = std::fs::read(&path).unwrap();
        let arena = Arena::new(1 << 20);
        let opts = Options { workers: 2, slots: 2, span_bytes: 4096, gap_bytes: 256, direct: DirectMode::Never };
        // good marker, with a copy before and a copy after it in the same span: both land, nothing else is reported as copied
        let (a, b) = (arena.alloc(50).unwrap(), arena.alloc(40).unwrap());
        let mut l = Loader::new();
        let f = l.file(&path);
        l.copy_file(&a, f, 0, 50, Xform::None).unwrap();
        l.copy_file(&b, f, 54, 40, Xform::None).unwrap();
        l.check_file(f, 50, &0x83DC_D12Du32.to_le_bytes(), "codebook marker of `good`").unwrap();
        assert_eq!(l.queued_bytes(), 90, "a check copies nothing");
        let rep = l.run(&opts).unwrap();
        assert_eq!(rep.bytes, 90);
        assert_eq!(a.to_vec::<u8>().unwrap(), want[..50]);
        assert_eq!(b.to_vec::<u8>().unwrap(), want[54..94]);
        // a marker of the wrong codebook (mcg) at the same place, and one that is not there at all, each fail the load by name
        for (what, bytes, off) in [("mcg", 0xCBAC_1FEDu32.to_le_bytes(), 50u64), ("shifted", 0x83DC_D12Du32.to_le_bytes(), 51)] {
            let mut l = Loader::new();
            let f = l.file(&path);
            let c = arena.alloc(50).unwrap();
            l.copy_file(&c, f, 0, 50, Xform::None).unwrap();
            l.check_file(f, off, &bytes, format!("codebook marker of `{what}`")).unwrap();
            let e = l.run(&opts).unwrap_err().to_string();
            assert!(e.contains(&format!("codebook marker of `{what}`")) && e.contains("expected"), "{e}");
        }
        // a check alone (no copy job at all) is read and compared too
        let mut l = Loader::new();
        let f = l.file(&path);
        l.check_file(f, 50, &0u32.to_le_bytes(), "only a check").unwrap();
        assert!(l.run(&opts).is_err());
        // an unknown file id is a plan error at declaration
        let mut l = Loader::new();
        assert!(l.check_file(3, 0, &[1], "no such file").is_err());
        std::fs::remove_file(path).unwrap();
    }
}

#[cfg(test)]
mod plan_tests {
    use super::*;

    #[test]
    fn the_read_totals_split_cold_from_warm_and_name_the_boots_cache_state() {
        let c = TotalCounters {
            bytes: std::sync::atomic::AtomicU64::new(0),
            cold_bytes: std::sync::atomic::AtomicU64::new(0),
            direct_bytes: std::sync::atomic::AtomicU64::new(0),
            read_ns: std::sync::atomic::AtomicU64::new(0),
        };
        c.note(1000, 1.0, false, std::time::Duration::from_millis(1));
        c.note(1000, 0.0, true, std::time::Duration::from_millis(3));
        c.note(1000, 0.25, false, std::time::Duration::ZERO);
        let t = Totals {
            bytes: c.bytes.load(Ordering::Relaxed),
            cold_bytes: c.cold_bytes.load(Ordering::Relaxed),
            direct_bytes: c.direct_bytes.load(Ordering::Relaxed),
            read_s: c.read_ns.load(Ordering::Relaxed) as f64 / 1e9,
        };
        assert_eq!((t.bytes, t.cold_bytes, t.direct_bytes), (3000, 1750, 1000));
        assert!((t.read_s - 0.004).abs() < 1e-9);
        assert_eq!(t.cache_state(), "mixed");
        let none = Totals::default();
        assert_eq!(none.cache_state(), "none");
        assert_eq!(t.since(&t).bytes, 0);
        assert_eq!(Totals { bytes: 100, cold_bytes: 95, ..none }.cache_state(), "cold");
        assert_eq!(Totals { bytes: 100, cold_bytes: 5, ..none }.cache_state(), "warm");
        assert_eq!(t.since(&Totals { bytes: 1000, cold_bytes: 0, direct_bytes: 0, read_s: 0.001 }).cold_bytes, 1750);
    }

    /// Deterministic pseudo-random sizes.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Every byte of every request is covered exactly once, by items at the right place in the right span, and no span is
    /// longer than allowed.
    fn check(reqs: &[Req], span_bytes: usize, gap: u64) {
        let spans = plan_spans(reqs, span_bytes, gap);
        let mut covered: Vec<Vec<(usize, usize)>> = vec![Vec::new(); reqs.len()];
        for (si, s) in spans.iter().enumerate() {
            assert!(s.len <= span_bytes, "span {si} is {} bytes, limit {span_bytes}", s.len);
            assert!(!s.items.is_empty(), "span {si} is empty");
            for it in &s.items {
                let r = reqs[it.job];
                assert_eq!(s.file, r.file, "item in a span of another file");
                assert!(it.rel + it.len <= s.len, "item runs past its span");
                // the file bytes the item reads are the job's bytes job_off..job_off+len
                assert_eq!(s.start + it.rel as u64, r.off + it.job_off as u64, "item reads the wrong file range");
                covered[it.job].push((it.job_off, it.len));
            }
            if si > 0 {
                let p = &spans[si - 1];
                assert!((p.file, p.start) <= (s.file, s.start), "spans are out of (file, offset) order");
            }
        }
        for (i, r) in reqs.iter().enumerate() {
            let mut pieces = covered[i].clone();
            pieces.sort();
            let mut at = 0;
            for (off, len) in pieces {
                assert_eq!(off, at, "job {i} has a gap or an overlap at {at}");
                at += len;
            }
            assert_eq!(at, r.len, "job {i} is covered for {at} of {} bytes", r.len);
            if r.len > 0 && r.len <= span_bytes / 2 {
                assert_eq!(covered[i].len(), 1, "a small job {i} was split");
            }
        }
    }

    #[test]
    fn spans_cover_every_byte_once_for_every_size_mix() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for round in 0..300 {
            let span = [64usize, 100, 4096, 1 << 20][round % 4];
            let mut reqs = Vec::new();
            let mut off = [0u64; 3];
            for _ in 0..rng.below(60) + 1 {
                let file = rng.below(3) as usize;
                off[file] += rng.below(3 * span as u64 / 2);
                let len = match rng.below(4) {
                    0 => 0,
                    1 => rng.below(span as u64 / 2 + 1) as usize,
                    2 => rng.below(span as u64 * 3) as usize,
                    _ => 1 + rng.below(8) as usize,
                };
                reqs.push(Req { file, off: off[file], len });
                off[file] += len as u64;
            }
            // jobs are not declared in file order
            let n = reqs.len();
            for i in (1..n).rev() {
                reqs.swap(i, rng.below(i as u64 + 1) as usize);
            }
            check(&reqs, span, [0u64, 7, 4096][round % 3]);
        }
    }

    #[test]
    fn neighbours_share_a_span_and_a_big_tensor_gets_its_own() {
        let span = 1000;
        let reqs = [
            Req { file: 0, off: 0, len: 100 },
            Req { file: 0, off: 100, len: 100 },
            Req { file: 0, off: 5000, len: 100 }, // far away: a new span
            Req { file: 1, off: 0, len: 100 },    // another file: a new span
            Req { file: 0, off: 200, len: 2500 }, // big: three pieces
        ];
        let s = plan_spans(&reqs, span, 0);
        // file 0: [0,200) shared by jobs 0 and 1; big job 4 at 200..2700 in pieces 1000,1000,500; then job 2 at 5000; file 1
        assert_eq!(
            s.iter().map(|x| (x.file, x.start, x.len)).collect::<Vec<_>>(),
            vec![(0, 0, 200), (0, 200, 1000), (0, 1200, 1000), (0, 2200, 500), (0, 5000, 100), (1, 0, 100),]
        );
        assert_eq!(s[0].items.len(), 2);
        check(&reqs, span, 0);
    }

    /// Run the copies of every piece of a strided job's span into a dense destination: each row byte arrives once, at
    /// `row * width + col`, no gap byte is copied, and a piece costs at most three copies.
    fn check_strided(width: usize, pitch: usize, rows: usize, cuts: &[usize]) {
        let s = Stride { width, src_pitch: pitch };
        let span = (rows - 1) * pitch + width;
        let src: Vec<u32> = (0..span as u32).collect();
        let mut dst: Vec<Option<u32>> = vec![None; rows * width];
        let mut at = 0;
        for &end in cuts.iter().chain(std::iter::once(&span)) {
            let piece = &src[at..end];
            let copies = strided_copies(at, end - at, s);
            assert!(copies.iter().flatten().all(|c| c.len > 0 && c.rows > 0 && (c.rows == 1 || c.len == width)));
            for c in copies.into_iter().flatten() {
                for r in 0..c.rows {
                    for b in 0..c.len {
                        let v = piece[c.src + r * pitch + b];
                        let (row, col) = (v as usize / pitch, v as usize % pitch);
                        assert!(col < width, "gap byte {v} copied");
                        let d = c.dst + r * width + b;
                        assert_eq!(d, row * width + col, "byte {v} lands at {d}");
                        assert!(dst[d].replace(v).is_none(), "byte {v} copied twice");
                    }
                }
            }
            at = end;
        }
        assert!(dst.iter().all(|v| v.is_some()), "a row byte was not copied ({width} of {pitch}, {rows} rows, cuts {cuts:?})");
    }

    #[test]
    fn strided_pieces_copy_every_row_byte_once_in_at_most_three_copies() {
        check_strided(4, 8, 3, &[]);
        check_strided(8, 8, 3, &[5, 13]);
        check_strided(3, 10, 1, &[1]);
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        for _ in 0..2000 {
            let pitch = 1 + rng.below(40) as usize;
            let width = 1 + rng.below(pitch as u64) as usize;
            let rows = 1 + rng.below(30) as usize;
            let span = (rows - 1) * pitch + width;
            let mut cuts: Vec<usize> = (0..rng.below(6)).map(|_| 1 + rng.below(span as u64) as usize).filter(|&c| c < span).collect();
            cuts.sort();
            cuts.dedup();
            check_strided(width, pitch, rows, &cuts);
        }
        // a whole-span piece of a column half (the GLM gate/up cut) is one 2D copy
        let s = Stride { width: 2048 * 3, src_pitch: 4096 * 3 };
        let span = 255 * s.src_pitch + s.width;
        assert_eq!(strided_copies(0, span, s), [None, Some(RowCopy { src: 0, dst: 0, len: s.width, rows: 256 }), None]);
    }
}
