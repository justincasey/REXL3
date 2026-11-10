//! Host n-gram sidecar reader and per-request token context.
//!
//! `NgramTable::open(dir, cfg)` mmaps only the n-gram safetensors file, validates its metadata,
//! and keeps packed rows on the host; `gather(history, 2, output)` returns interleaved u16 words
//! for the PLE device decoder. `row_ids` is pure and shared by the oracle test. `warm` streams
//! the whole file on a completion-tracked worker: readiness waits for its successful completion
//! (`finish_warm` / `warm_preflight`) before serving, so cold mapped-table faults never hit the
//! first decode rounds. The file's pages are reclaimable page cache, not an anonymous/device
//! reservation, so the memory plan never deducts the table twice.
//!
//! Latency of the gather (a cold row is a 4 KiB NVMe page fault; a 4096-row prefill chunk reads up
//! to 65,536 rows) is handled in four layers that change only WHEN bytes are read, never WHICH:
//! a tight allocation-free loop over precomputed row ids; a persistent worker pool that overlaps the
//! faults of big chunks; `prefetch` / `gather_ahead`, which let a caller that knows the next tokens
//! early page the rows in (or gather them) while the GPU runs; and an optional row-granular cache
//! (`REXL3_NGRAM_ROW_CACHE_MIB`, off by default) in front of the mmap.

use super::{config::FlashNextConfig, inventory};
use serde_json::Value;
use std::{
    path::{Path, PathBuf},
    sync::atomic::{fence, AtomicBool, AtomicU64, AtomicUsize, Ordering},
    sync::{Arc, Condvar, Mutex, OnceLock},
};

// The mmap bytes ARE the u16 words the device decoder reads: the file is little-endian.
const _: () = assert!(cfg!(target_endian = "little"));

const ROW: usize = 160;
const RING: usize = 128;
/// Positions per work item of a parallel gather.
const BLOCK_POSITIONS: usize = 64;
/// Row ids kept on the stack by one gather block (a verify window is at most 112).
const STACK_IDS: usize = 512;
/// Hard cap on the head count (2 n-gram orders x heads_per_ngram) a table may have.
const MAX_HEADS: usize = 64;
/// Bytes faulted per read of the warm worker: small enough that progress and cancellation stay
/// responsive, large enough that a 36 GiB table streams with few syscalls.
const WARM_CHUNK: usize = 1 << 20;

/// Completion state of the n-gram warm worker. `done` flips only after every byte of the file is
/// read (or an error ended the read); [`WarmHandle::wait`] never returns before that.
pub struct WarmHandle {
    done: AtomicBool,
    result: Mutex<Option<Result<(), String>>>,
    read: AtomicU64,
}

impl WarmHandle {
    fn new() -> Self {
        Self { done: AtomicBool::new(false), result: Mutex::new(None), read: AtomicU64::new(0) }
    }
    /// A handle that already failed (test-only): the same completion state a real read error
    /// produces, so the gate's propagation can be checked without a GPU or a large file.
    #[cfg(test)]
    pub fn new_failed_for_test(error: String) -> Self {
        let h = Self::new();
        h.finish(Err(error));
        h
    }
    fn finish(&self, out: Result<(), String>) {
        *self.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(out);
        self.done.store(true, Ordering::Release);
    }
    /// The worker's outcome once it has finished; `None` while it still runs.
    pub fn done(&self) -> Option<Result<(), String>> {
        if self.done.load(Ordering::Acquire) {
            Some(self.result.lock().unwrap_or_else(|e| e.into_inner()).clone().expect("result is set with done"))
        } else {
            None
        }
    }
    /// Bytes read (faulted) so far, on the way to the whole file.
    pub fn bytes_read(&self) -> u64 {
        self.read.load(Ordering::Relaxed)
    }
    /// Block until the worker finishes and report its outcome. Parks briefly between polls: the
    /// caller is the boot/readiness path, which may wait for a multi-GiB read.
    pub fn wait(&self) -> Result<(), String> {
        while !self.done.load(Ordering::Acquire) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        self.result.lock().unwrap_or_else(|e| e.into_inner()).clone().expect("result is set with done")
    }
}

/// Sequentially fault/read the whole file on the warm worker and record the outcome.
fn warm_read(path: PathBuf, total: u64, h: &WarmHandle) -> Result<(), String> {
    use std::io::Read;
    let mut file = std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut buf = vec![0u8; WARM_CHUNK];
    let mut read = 0u64;
    loop {
        match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                read += n as u64;
                h.read.store(read.min(total), Ordering::Relaxed);
            }
            Err(e) => return Err(format!("{}: {e}", path.display())),
        }
    }
    Ok(())
}

struct Mapped {
    ptr: *const u8,
    len: usize,
}
// SAFETY: the map is read-only and is released only when Mapped is dropped.
unsafe impl Send for Mapped {}
// SAFETY: shared read-only bytes are safe to read from multiple threads.
unsafe impl Sync for Mapped {}
impl Mapped {
    fn open(path: &Path) -> Result<Self, String> {
        use std::os::fd::AsRawFd;
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let len = usize::try_from(file.metadata().map_err(|e| e.to_string())?.len()).map_err(|e| e.to_string())?;
        if len < 8 {
            return Err(format!("{}: short safetensors file", path.display()));
        }
        // SAFETY: a read-only shared mapping of an open regular file; MAP_FAILED is checked.
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, file.as_raw_fd(), 0) };
        if ptr == libc::MAP_FAILED {
            return Err(format!("{}: mmap failed", path.display()));
        }
        // SAFETY: ptr is a valid len-byte mapping.
        unsafe {
            libc::madvise(ptr, len, libc::MADV_RANDOM);
        }
        Ok(Self { ptr: ptr.cast(), len })
    }
    fn bytes(&self) -> &[u8] {
        // SAFETY: ptr and len are owned by this live mapping.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}
impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: ptr and len were returned by mmap and not otherwise unmapped.
        unsafe {
            libc::munmap(self.ptr.cast_mut().cast(), self.len);
        }
    }
}

/// Wrapped signed hash arithmetic, Euclidean modulo, and an EOS boundary before t2.
pub fn row_ids(
    tokens: [u32; 3],
    eos: u32,
    multipliers: &[i64],
    offsets: &[i64],
    sizes: &[i64],
    heads_per_ngram: usize,
    out: &mut [i64],
) -> Result<(), String> {
    let ngram = multipliers.len();
    if ngram != 3
        || heads_per_ngram == 0
        || offsets.len() != 2 * heads_per_ngram
        || sizes.len() != offsets.len()
        || out.len() < offsets.len()
        || sizes.contains(&0)
        || sizes.iter().any(|&s| s < 0)
    {
        return Err("invalid 3-gram hash parameters or output".into());
    }
    mix_ids(tokens, eos, multipliers, offsets, sizes, heads_per_ngram, out);
    Ok(())
}

/// The hash of [`row_ids`] without the parameter checks: the caller (the table, which validated
/// them once at open) guarantees 3 multipliers and `2 * heads_per_ngram` positive sizes/offsets.
#[inline(always)]
fn mix_ids(
    tokens: [u32; 3],
    eos: u32,
    multipliers: &[i64],
    offsets: &[i64],
    sizes: &[i64],
    heads_per_ngram: usize,
    out: &mut [i64],
) {
    let [t0, t1, t2] = tokens;
    let tokens = [t0 as i64, t1 as i64, if t1 == eos { eos as i64 } else { t2 as i64 }];
    let mut mix = tokens[0].wrapping_mul(multipliers[0]);
    for n in 1..3 {
        mix ^= tokens[n].wrapping_mul(multipliers[n]);
        for j in (n - 1) * heads_per_ngram..n * heads_per_ngram {
            out[j] = mix.rem_euclid(sizes[j]) + offsets[j];
        }
    }
}

/// Performance knobs of a table. None of them changes a gathered byte.
#[derive(Clone, Debug)]
pub struct Tuning {
    /// Threads a big gather runs on, the caller included; <= 1 means no pool.
    pub threads: usize,
    /// Smallest gather (in positions) that is handed to the pool.
    pub par_min_positions: usize,
    /// Queue `MADV_WILLNEED` for a block's rows before copying them (deeper NVMe queue per thread).
    pub willneed: bool,
    /// Row-granular cache in front of the mmap, MiB; 0 = off.
    pub cache_mib: usize,
    /// Gather the next prefill chunk's rows on a worker while the GPU runs the current one
    /// ([`NgramTable::gather_ahead`], used by `FlashNextModel::ple_prefetch_next`).
    pub ahead: bool,
}

/// All off: the original serial gather. The tests and the benchmark's baseline start here.
impl Default for Tuning {
    fn default() -> Self {
        Self { threads: 1, par_min_positions: 64, willneed: false, cache_mib: 0, ahead: false }
    }
}

impl Tuning {
    /// What the server runs with (measured on the real 39 GB table, see the PERF commit message): 8 threads for
    /// gathers of >= 64 positions, `MADV_WILLNEED` queued per block, the next prefill chunk gathered ahead,
    /// no row cache (its hit rate does not beat the page cache at the sizes the table's neighbours leave).
    pub fn production() -> Self {
        Self { threads: 8, par_min_positions: 64, willneed: true, cache_mib: 0, ahead: true }
    }

    /// `production()` with `REXL3_NGRAM_THREADS`, `REXL3_NGRAM_PAR_MIN`, `REXL3_NGRAM_WILLNEED` (0/1),
    /// `REXL3_NGRAM_AHEAD` (0/1) and `REXL3_NGRAM_ROW_CACHE_MIB` applied; an unparsable value is ignored.
    /// `REXL3_NGRAM_THREADS=1 REXL3_NGRAM_WILLNEED=0 REXL3_NGRAM_AHEAD=0` is the original behaviour.
    pub fn from_env() -> Self {
        let num = |k: &str| std::env::var(k).ok().and_then(|v| v.trim().parse::<usize>().ok());
        let d = Self::production();
        Self {
            threads: num("REXL3_NGRAM_THREADS").unwrap_or(d.threads),
            par_min_positions: num("REXL3_NGRAM_PAR_MIN").unwrap_or(d.par_min_positions),
            willneed: num("REXL3_NGRAM_WILLNEED").map_or(d.willneed, |v| v != 0),
            cache_mib: num("REXL3_NGRAM_ROW_CACHE_MIB").unwrap_or(d.cache_mib),
            ahead: num("REXL3_NGRAM_AHEAD").map_or(d.ahead, |v| v != 0),
        }
    }
}

/// Everything a gather reads: shared (`Arc`) with the pool's threads, immutable after open.
struct Inner {
    map: Mapped,
    base: usize,
    words: usize,
    heads: usize,
    heads_per_ngram: usize,
    offsets: Vec<i64>,
    sizes: Vec<i64>,
    multipliers: Vec<i64>,
    eos: u32,
    willneed: bool,
    page: usize,
    cache: Option<RowCache>,
}

impl Inner {
    /// The row ids of one position: the hash of `row_ids`, narrowed to u32 (open checked `rows`).
    #[inline(always)]
    fn fill_ids(&self, t0: u32, t1: u32, t2: u32, out: &mut [u32]) {
        let mut wide = [0i64; MAX_HEADS];
        mix_ids([t0, t1, t2], self.eos, &self.multipliers, &self.offsets, &self.sizes, self.heads_per_ngram, &mut wide);
        for (o, &w) in out.iter_mut().zip(&wide[..self.heads]) {
            *o = w as u32;
        }
    }

    /// Row ids of `n` positions `p0..p0+n`, position-major, into `ids` (`n * heads`).
    fn fill_block(&self, hist: &[u32], ctx: usize, p0: usize, ids: &mut [u32]) {
        for (i, o) in ids.chunks_exact_mut(self.heads).enumerate() {
            let q = p0 + i + ctx;
            self.fill_ids(hist[q], hist[q - 1], hist[q - 2], o);
        }
    }

    /// Queue the page reads of one row.
    #[inline]
    fn willneed_row(&self, id: u32) {
        let rb = self.words * 2;
        let at = self.base + id as usize * rb;
        let start = at & !(self.page - 1);
        let end = (at + rb + self.page - 1) & !(self.page - 1);
        // SAFETY: the range lies inside the mapping (its last page is mapped whole); WILLNEED never
        // changes the mapped bytes and a failure only means no readahead.
        unsafe {
            libc::madvise(self.map.ptr.add(start).cast_mut().cast(), end - start, libc::MADV_WILLNEED);
        }
    }

    /// Gather the positions `p0 .. p0 + dst.len() / per` into `dst` (one position = `heads` rows).
    fn gather_block(&self, hist: &[u32], ctx: usize, p0: usize, dst: &mut [u16]) {
        let rb = self.words * 2;
        let count = dst.len() / self.words;
        let mut stack = [0u32; STACK_IDS];
        let mut heap;
        let ids: &mut [u32] = if count <= STACK_IDS {
            &mut stack[..count]
        } else {
            heap = vec![0u32; count];
            &mut heap
        };
        self.fill_block(hist, ctx, p0, ids);
        if self.willneed {
            for &id in ids.iter() {
                if !self.cache.as_ref().is_some_and(|c| c.contains(id)) {
                    self.willneed_row(id);
                }
            }
        }
        let all = self.map.bytes();
        // SAFETY: a u16 slice viewed as its bytes (little-endian host, asserted above).
        let dst8 = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), dst.len() * 2) };
        let (mut hit, mut miss) = (0u64, 0u64);
        for (d, &id) in dst8.chunks_exact_mut(rb).zip(ids.iter()) {
            if let Some(c) = &self.cache {
                if c.get(id, d) {
                    hit += 1;
                    continue;
                }
                miss += 1;
            }
            let at = self.base + id as usize * rb;
            d.copy_from_slice(&all[at..at + rb]);
            if let Some(c) = &self.cache {
                c.put(id, d);
            }
        }
        if let Some(c) = &self.cache {
            c.count(hit, miss);
        }
    }

    fn prefetch(&self, hist: &[u32], ctx: usize) {
        let n = hist.len() - ctx;
        let mut ids = [0u32; MAX_HEADS];
        for p in 0..n {
            let q = p + ctx;
            self.fill_ids(hist[q], hist[q - 1], hist[q - 2], &mut ids[..self.heads]);
            for &id in &ids[..self.heads] {
                if !self.cache.as_ref().is_some_and(|c| c.contains(id)) {
                    self.willneed_row(id);
                }
            }
        }
    }
}

/// Direct-mapped cache of packed rows, filled by gathers. An entry is a tag word (`id + 1`, 0 =
/// empty, bit 63 = being written) plus the row's bytes as u64 words; readers validate the tag after
/// the copy (a seqlock), so a torn row is never returned. Row bytes depend only on the id, so a
/// hit is bit-identical to the mmap read. The backing memory is zero pages until first written.
struct RowCache {
    tags: Box<[AtomicU64]>,
    data: Box<[AtomicU64]>,
    slots: usize,
    wpe: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

const LOCK: u64 = 1 << 63;
const MAX_ROW_WORDS: usize = 24;

fn zeroed_atomics(n: usize) -> Box<[AtomicU64]> {
    let mut v = std::mem::ManuallyDrop::new(vec![0u64; n]);
    // SAFETY: AtomicU64 has the size and alignment of u64 and 0 is a valid value of both.
    unsafe { Vec::from_raw_parts(v.as_mut_ptr().cast::<AtomicU64>(), v.len(), v.capacity()) }.into_boxed_slice()
}

impl RowCache {
    fn new(mib: usize, row_bytes: usize) -> Self {
        let wpe = row_bytes.div_ceil(8);
        assert!(wpe <= MAX_ROW_WORDS, "row too wide for the row cache");
        let slots = ((mib << 20) / (8 * (wpe + 1))).clamp(1, u32::MAX as usize);
        Self {
            tags: zeroed_atomics(slots),
            data: zeroed_atomics(slots * wpe),
            slots,
            wpe,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }
    #[inline(always)]
    fn index(&self, id: u32) -> usize {
        ((((id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) * self.slots as u64) >> 32) as usize
    }
    fn contains(&self, id: u32) -> bool {
        self.tags[self.index(id)].load(Ordering::Relaxed) == id as u64 + 1
    }
    fn get(&self, id: u32, dst: &mut [u8]) -> bool {
        let (i, want) = (self.index(id), id as u64 + 1);
        if self.tags[i].load(Ordering::Acquire) != want {
            return false;
        }
        let mut buf = [0u64; MAX_ROW_WORDS];
        for (b, w) in buf.iter_mut().zip(&self.data[i * self.wpe..(i + 1) * self.wpe]) {
            *b = w.load(Ordering::Relaxed);
        }
        fence(Ordering::Acquire);
        if self.tags[i].load(Ordering::Relaxed) != want {
            return false;
        }
        for (c, w) in dst.chunks_mut(8).zip(buf) {
            c.copy_from_slice(&w.to_le_bytes()[..c.len()]);
        }
        true
    }
    fn put(&self, id: u32, src: &[u8]) {
        let (i, want) = (self.index(id), id as u64 + 1);
        let t = self.tags[i].load(Ordering::Relaxed);
        if t == want || t & LOCK != 0 {
            return;
        }
        if self.tags[i].compare_exchange(t, t | LOCK, Ordering::Acquire, Ordering::Relaxed).is_err() {
            return;
        }
        for (w, c) in self.data[i * self.wpe..(i + 1) * self.wpe].iter().zip(src.chunks(8)) {
            let mut b = [0u8; 8];
            b[..c.len()].copy_from_slice(c);
            w.store(u64::from_le_bytes(b), Ordering::Relaxed);
        }
        self.tags[i].store(want, Ordering::Release);
    }
    fn count(&self, hit: u64, miss: u64) {
        self.hits.fetch_add(hit, Ordering::Relaxed);
        self.misses.fetch_add(miss, Ordering::Relaxed);
    }
    fn stats(&self) -> (u64, u64) {
        (self.hits.load(Ordering::Relaxed), self.misses.load(Ordering::Relaxed))
    }
}

/// One parallel gather: the pool's threads and the caller take `BLOCK_POSITIONS` blocks until none
/// is left. It lives on the caller's stack; `Pool::run` returns only after every thread has let go.
struct Job {
    hist: *const u32,
    hist_len: usize,
    ctx: usize,
    n: usize,
    out: *mut u16,
    per: usize,
    next: AtomicUsize,
}

fn work(inner: &Inner, job: &Job) {
    loop {
        let p0 = job.next.fetch_add(1, Ordering::Relaxed) * BLOCK_POSITIONS;
        if p0 >= job.n {
            return;
        }
        let p1 = (p0 + BLOCK_POSITIONS).min(job.n);
        // SAFETY: `hist` is the caller's live slice; blocks are disjoint ranges of the caller's `out`,
        // each handed to exactly one thread by the `next` counter.
        let (hist, dst) = unsafe {
            (
                std::slice::from_raw_parts(job.hist, job.hist_len),
                std::slice::from_raw_parts_mut(job.out.add(p0 * job.per), (p1 - p0) * job.per),
            )
        };
        inner.gather_block(hist, job.ctx, p0, dst);
    }
}

#[derive(Clone, Copy)]
struct JobPtr(*const Job);
// SAFETY: the pointee is only used between publication and the caller's join in `Pool::run`.
unsafe impl Send for JobPtr {}

struct PoolState {
    gen: u64,
    job: Option<JobPtr>,
    active: usize,
    shutdown: bool,
}

struct PoolShared {
    st: Mutex<PoolState>,
    wake: Condvar,
    idle: Condvar,
}

impl PoolShared {
    fn lock(&self) -> std::sync::MutexGuard<'_, PoolState> {
        self.st.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Persistent helper threads, parked on a condvar between gathers.
struct Pool {
    shared: Arc<PoolShared>,
    /// One parallel gather at a time; a second concurrent caller runs serially instead of waiting.
    gate: Mutex<()>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

/// Unpublishes the job and waits for the helpers still inside it (also on unwind).
struct Finish<'a>(&'a PoolShared);
impl Drop for Finish<'_> {
    fn drop(&mut self) {
        let mut st = self.0.lock();
        st.job = None;
        while st.active > 0 {
            st = self.0.idle.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }
}

struct Active<'a>(&'a PoolShared);
impl Drop for Active<'_> {
    fn drop(&mut self) {
        let mut st = self.0.lock();
        st.active -= 1;
        if st.active == 0 {
            self.0.idle.notify_all();
        }
    }
}

impl Pool {
    fn new(inner: &Arc<Inner>, helpers: usize) -> Self {
        let shared = Arc::new(PoolShared {
            st: Mutex::new(PoolState { gen: 0, job: None, active: 0, shutdown: false }),
            wake: Condvar::new(),
            idle: Condvar::new(),
        });
        let workers = (0..helpers)
            .map(|i| {
                let (shared, inner) = (Arc::clone(&shared), Arc::clone(inner));
                std::thread::Builder::new()
                    .name(format!("qwen-ngram-gather-{i}"))
                    .spawn(move || Self::worker(&shared, &inner))
                    .expect("spawn an n-gram gather worker")
            })
            .collect();
        Self { shared, gate: Mutex::new(()), workers }
    }

    fn worker(shared: &PoolShared, inner: &Inner) {
        let mut seen = 0;
        loop {
            let job = {
                let mut st = shared.lock();
                loop {
                    if st.shutdown {
                        return;
                    }
                    if st.gen != seen {
                        seen = st.gen;
                        if let Some(j) = st.job {
                            st.active += 1;
                            break j;
                        }
                    }
                    st = shared.wake.wait(st).unwrap_or_else(|e| e.into_inner());
                }
            };
            let _active = Active(shared);
            // SAFETY: `active` was raised under the lock while the job was published, and `run`
            // does not return before `active` is back to zero.
            work(inner, unsafe { &*job.0 });
        }
    }

    /// Gather on the pool plus the calling thread. False (nothing done) when another gather holds the pool.
    fn run(&self, inner: &Inner, hist: &[u32], ctx: usize, n: usize, out: &mut [u16]) -> bool {
        let Ok(_gate) = self.gate.try_lock() else { return false };
        let job = Job {
            hist: hist.as_ptr(),
            hist_len: hist.len(),
            ctx,
            n,
            out: out.as_mut_ptr(),
            per: out.len() / n.max(1),
            next: AtomicUsize::new(0),
        };
        let _finish = Finish(&self.shared);
        {
            let mut st = self.shared.lock();
            st.job = Some(JobPtr(&job));
            st.gen += 1;
        }
        self.shared.wake.notify_all();
        work(inner, &job);
        true
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.lock().shutdown = true;
        self.shared.wake.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// A gather running on its own thread ([`NgramTable::gather_ahead`]).
pub struct Ahead {
    ctx: usize,
    join: std::thread::JoinHandle<Result<(Vec<u32>, Vec<u16>), String>>,
}

impl Ahead {
    /// Wait for the gather and return its packed rows if it was started for exactly `history`
    /// (and the same `ctx`); `None` (result discarded) for a different request or a failed worker,
    /// in which case the caller gathers inline.
    pub fn take(self, history: &[u32], ctx: usize) -> Option<Vec<u16>> {
        let same_ctx = self.ctx == ctx;
        match self.join.join() {
            Ok(Ok((h, out))) if same_ctx && h == history => Some(out),
            _ => None,
        }
    }
}

/// Read-only mmap of packed table rows plus small hash parameters.
pub struct NgramTable {
    inner: Arc<Inner>,
    /// Helper threads, spawned by the first gather big enough to use them (`tuning.threads > 1`).
    pool: OnceLock<Pool>,
    tuning: Tuning,
    /// u16 words of one position's gathered rows: `heads * words`.
    per: usize,
    path: PathBuf,
    pub rows: usize,
    pub k: usize,
    pub words: usize,
    pub heads: usize,
    pub heads_per_ngram: usize,
    pub eos: u32,
    pub head_bias: Vec<u16>,
    /// The completion-tracked warm worker, started at most once per table.
    warm: OnceLock<Arc<WarmHandle>>,
}

impl NgramTable {
    /// Open with the process's tuning (`Tuning::from_env`).
    pub fn open(dir: &Path, cfg: &FlashNextConfig) -> Result<Self, String> {
        Self::open_with(dir, cfg, Tuning::from_env())
    }

    pub fn open_with(dir: &Path, cfg: &FlashNextConfig, tuning: Tuning) -> Result<Self, String> {
        let path = dir.join(inventory::NGRAM_FILE);
        let map = Mapped::open(&path)?;
        let bytes = map.bytes();
        let hlen = usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap())).map_err(|e| e.to_string())?;
        let data = 8usize
            .checked_add(hlen)
            .filter(|&n| n <= bytes.len() && hlen <= gb_exl3::checkpoint::MAX_HEADER_BYTES as usize)
            .ok_or("ngram header exceeds file")?;
        let head: Value = serde_json::from_slice(&bytes[8..data]).map_err(|e| e.to_string())?;
        let meta = &head["__metadata__"];
        if meta["format"] != "exl3_ngram_trellis" || meta["codebook"] != "mul1" {
            return Err("unsupported ngram format or codebook".into());
        }
        let number = |n: &str| -> Result<usize, String> {
            meta[n].as_str().ok_or_else(|| format!("metadata {n}"))?.parse().map_err(|_| format!("metadata {n}"))
        };
        let (k, rows, shard_rows, dim) = (number("K")?, number("rows")?, number("shard_rows")?, number("row_dim")?);
        if dim != ROW || !(1..=8).contains(&k) || shard_rows == 0 || rows == 0 || cfg.ngram_size != 3 {
            return Err("unsupported ngram row geometry".into());
        }
        let words = inventory::ngram_row_words(k);
        let layer = cfg.ple_layer_ids.first().and_then(|&id| id.checked_sub(1)).ok_or("no valid PLE layer")?;
        let prefix = format!("model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding");
        let span = |name: &str, dtype: &str, shape: &[usize]| -> Result<std::ops::Range<usize>, String> {
            let full = format!("{prefix}.{name}");
            let t = head.get(&full).ok_or_else(|| format!("{full}: missing"))?;
            if t["dtype"] != dtype
                || t["shape"].as_array().is_none_or(|a| {
                    a.iter().map(Value::as_u64).collect::<Option<Vec<_>>>().as_deref()
                        != Some(&shape.iter().map(|&x| x as u64).collect::<Vec<_>>())
                })
            {
                return Err(format!("{full}: dtype or shape mismatch"));
            }
            let a = t["data_offsets"][0].as_u64().ok_or_else(|| format!("{full}: start"))?;
            let b = t["data_offsets"][1].as_u64().ok_or_else(|| format!("{full}: end"))?;
            let unit = gb_exl3::checkpoint::dtype_size(dtype).ok_or("dtype")?;
            let size = shape.iter().try_fold(unit, |v, &d| v.checked_mul(d as u64)).ok_or("shape overflow")?;
            if b.checked_sub(a) != Some(size) || b > (bytes.len() - data) as u64 {
                return Err(format!("{full}: extent"));
            }
            Ok(data + a as usize..data + b as usize)
        };
        let n_shards = rows.div_ceil(shard_rows);
        let base = span("shard_0.trellis", "I16", &[rows.min(shard_rows), words])?.start;
        for s in 0..n_shards {
            let count = shard_rows.min(rows - s * shard_rows);
            let range = span(&format!("shard_{s}.trellis"), "I16", &[count, words])?;
            if range.start != base + s * shard_rows * words * 2 {
                return Err(format!("shard_{s} is not contiguous"));
            }
        }
        let heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
        let parse_i64 = |n: &str, len: usize| -> Result<Vec<i64>, String> {
            let range = span(n, "I64", &[len])?;
            Ok(bytes[range].chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect())
        };
        let offsets = parse_i64("head_offsets", heads)?;
        let sizes = parse_i64("head_vocab_sizes", heads)?;
        let multipliers = parse_i64("layer_multipliers", cfg.ngram_size)?;
        let bias = span("head_bias", "F16", &[heads, ROW])?;
        let head_bias = bytes[bias].chunks_exact(2).map(|b| u16::from_le_bytes(b.try_into().unwrap())).collect();
        for (j, (&offset, &size)) in offsets.iter().zip(&sizes).enumerate() {
            if offset < 0 || size <= 0 || offset.checked_add(size).is_none_or(|end| end as u64 > rows as u64) {
                return Err(format!("head {j} outside the ngram table"));
            }
        }
        if heads > MAX_HEADS || rows > u32::MAX as usize {
            return Err("ngram table has too many heads or rows".into());
        }
        let row_bytes = words * 2;
        let cache = (tuning.cache_mib > 0).then(|| RowCache::new(tuning.cache_mib, row_bytes));
        let inner = Arc::new(Inner {
            map,
            base,
            words,
            heads,
            heads_per_ngram: cfg.heads_per_ngram,
            offsets,
            sizes,
            multipliers,
            eos: cfg.eos_token_id,
            willneed: tuning.willneed,
            page: usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).ok().filter(|&p| p > 0).unwrap_or(4096),
            cache,
        });
        Ok(Self {
            inner,
            pool: OnceLock::new(),
            tuning,
            per: heads * words,
            path,
            rows,
            k,
            words,
            heads,
            heads_per_ngram: cfg.heads_per_ngram,
            eos: cfg.eos_token_id,
            head_bias,
            warm: OnceLock::new(),
        })
    }

    pub fn row_ids(&self, t0: u32, t1: u32, t2: u32, out: &mut [i64]) -> Result<(), String> {
        let i = &self.inner;
        row_ids([t0, t1, t2], self.eos, &i.multipliers, &i.offsets, &i.sizes, self.heads_per_ngram, out)
    }

    /// Gather packed rows (u16 words), with ctx >= 2 context tokens before the positions to decode.
    /// Gathers of at least `par_min_positions` positions run on the worker pool (when one is
    /// configured); the bytes written are identical on every path.
    pub fn gather(&self, history: &[u32], ctx: usize, out: &mut [u16]) -> Result<(), String> {
        let n = self.check_gather(history, ctx, out.len())?;
        let out = &mut out[..n * self.per];
        if self.tuning.threads > 1 && n >= self.tuning.par_min_positions {
            let pool = self.pool.get_or_init(|| Pool::new(&self.inner, self.tuning.threads - 1));
            if pool.run(&self.inner, history, ctx, n, out) {
                return Ok(());
            }
        }
        self.inner.gather_block(history, ctx, 0, out);
        Ok(())
    }

    /// `gather` on the calling thread only (the pool is never used); same bytes.
    pub fn gather_serial(&self, history: &[u32], ctx: usize, out: &mut [u16]) -> Result<(), String> {
        let n = self.check_gather(history, ctx, out.len())?;
        self.inner.gather_block(history, ctx, 0, &mut out[..n * self.per]);
        Ok(())
    }

    /// Ask the kernel to read the pages of every row `gather(history, ctx, ..)` will need
    /// (`MADV_WILLNEED`): returns once the reads are queued, not when they finish, and writes nothing.
    /// Rows already in the row cache are skipped. A later `gather` of the same tokens returns the
    /// same bytes whether or not this ran.
    pub fn prefetch(&self, history: &[u32], ctx: usize) -> Result<(), String> {
        if ctx < 2 || ctx > history.len() {
            return Err("history needs two context tokens".into());
        }
        self.inner.prefetch(history, ctx);
        Ok(())
    }

    /// Start `gather(&history, ctx, ..)` on its own thread and return a handle for its result: the
    /// caller (prefill: the next chunk's tokens are in the prompt) does other work meanwhile.
    pub fn gather_ahead(self: &Arc<Self>, history: Vec<u32>, ctx: usize) -> Result<Ahead, String> {
        let n = self.check_gather(&history, ctx, usize::MAX)?;
        let table = Arc::clone(self);
        let join = std::thread::Builder::new()
            .name("qwen-ngram-ahead".into())
            .spawn(move || {
                let mut out = vec![0u16; n * table.per];
                table.gather(&history, ctx, &mut out)?;
                Ok((history, out))
            })
            .map_err(|e| format!("spawn the n-gram ahead worker: {e}"))?;
        Ok(Ahead { ctx, join })
    }

    /// Row-cache (hits, misses) so far; zero when the cache is off.
    pub fn cache_stats(&self) -> (u64, u64) {
        self.inner.cache.as_ref().map_or((0, 0), RowCache::stats)
    }

    pub fn tuning(&self) -> &Tuning {
        &self.tuning
    }

    /// Positions to decode, after the argument checks every gather entry shares.
    fn check_gather(&self, history: &[u32], ctx: usize, out_len: usize) -> Result<usize, String> {
        if ctx < 2 || ctx > history.len() {
            return Err("history needs two context tokens".into());
        }
        let n = history.len() - ctx;
        if out_len < n * self.per {
            return Err("packed row buffer is too short".into());
        }
        Ok(n)
    }

    /// Sequentially fault/read the whole mmap file on a completion-tracked worker, idempotently:
    /// repeated calls share one worker and its handle. The returned handle's [`WarmHandle::wait`]
    /// reports success or the explicit read failure; readiness waits on it before serving.
    pub fn warm(self: &Arc<Self>) -> Arc<WarmHandle> {
        self.warm
            .get_or_init(|| {
                let path = self.path.clone();
                let total = self.inner.map.len as u64;
                let h = Arc::new(WarmHandle::new());
                let hw = Arc::clone(&h);
                std::thread::Builder::new()
                    .name("qwen-ngram-warm".into())
                    .spawn(move || hw.finish(warm_read(path, total, &hw)))
                    .expect("spawn the n-gram warm worker");
                h
            })
            .clone()
    }
}

/// Start the completion-tracked warm worker when `warm` is on. Off bypasses without touching the
/// table: no worker and no file access.
pub fn warm_if(warm: bool, table: &Arc<NgramTable>) -> Option<Arc<WarmHandle>> {
    warm.then(|| table.warm())
}

/// The readiness gate: block until the warm worker (if any) completes. Failures are reported
/// explicitly (naming the file) and abort load before the server can become ready.
pub fn finish_warm(handle: Option<Arc<WarmHandle>>) -> Result<(), String> {
    handle.map_or(Ok(()), |h| h.wait())
}

/// The warm preflight the loader runs: start warm if requested and wait for its successful
/// completion before load proceeds. `None` table (no PLE stage) or off bypasses outright.
pub fn warm_preflight(warm: bool, table: Option<&Arc<NgramTable>>) -> Result<(), String> {
    match (warm, table) {
        (false, _) | (_, None) => Ok(()),
        (true, Some(t)) => {
            let start = std::time::Instant::now();
            let handle = t.warm();
            finish_warm(Some(Arc::clone(&handle)))?;
            let bytes = handle.bytes_read();
            eprintln!("rexl3 event=qwen_ngram_warm done bytes={} secs={:.3}", bytes, start.elapsed().as_secs_f64());
            gb_engine::boot::mark("ngram_warmed", Some(bytes as f64 / (1u64 << 30) as f64));
            Ok(())
        }
    }
}

/// Per-slot token ring. Rejected draft positions are replaced by the next store at the same indices.
pub struct TokContext {
    ring: Vec<[u32; RING]>,
}
impl TokContext {
    pub fn new(slots: usize) -> Self {
        Self { ring: vec![[0; RING]; slots] }
    }
    pub fn history(&self, slot: usize, first: usize, ids: &[i32], eos: u32) -> Vec<u32> {
        let at = |q: isize| if q < 0 { eos } else { self.ring[slot][q as usize % RING] };
        let mut out = Vec::with_capacity(ids.len() + 2);
        out.push(at(first as isize - 2));
        out.push(at(first as isize - 1));
        out.extend(ids.iter().map(|&v| v as u32));
        out
    }
    pub fn store(&mut self, slot: usize, first: usize, ids: &[i32]) {
        let skip = ids.len().saturating_sub(RING);
        for (j, &token) in ids.iter().enumerate().skip(skip) {
            self.ring[slot][(first + j) % RING] = token as u32;
        }
    }
}

#[cfg(test)]
#[path = "ngram_tests.rs"]
mod tests;
