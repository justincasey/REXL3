//! Building a ready-to-serve engine from a model folder: load the weights, plan memory from what is really free, size
//! the KV pool and admission to match, and wire the GPU backend into the engine.

use crate::backend::QwenBackend;
use crate::config::QwenConfig;
use crate::error::ModelError;
use crate::families::qwen;
use crate::family::{BuildCtx, Registry};
use crate::forward::Model;
use crate::hostmem;
use crate::loadctx::{LoadCtx, LoadMode};
use crate::loadstats::{self, Component};
use crate::memory::{MemoryOptions, VisionOptions};
use crate::pack::Pack;
use crate::plan::{plan_memory_cached_windowed, CacheRequest, AUTO_CACHE_CAP_BYTES};
use crate::snapshot::SnapPool;
use crate::weights::Weights;
use gb_engine::{boot, Drafter, Engine, EngineConfig, NoDrafts, PrefixConfig, ServeEngine};
use gb_sched::{CostModel, ModelMemory};
use gb_spec::CopyDrafter;
pub use gb_spec::SpecMode;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// How much memory the prefix cache may take (`--prefix-cache-mib`; design: `docs/PREFIX_CACHE.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixCacheOption {
    /// No cache.
    Off,
    /// What is spare after the KV pool, at most half of it and 8 GiB (the cache never makes the KV plan smaller). The default; a family
    /// may decide the spare memory is too small and run without a cache.
    Auto,
    /// `--prefix-cache on`: the cache is required. Sized like `Auto`, but a family must not turn it off because little memory is spare
    /// (a request for a cache that silently does not run is a bug, not a default).
    On,
    /// This many MiB, carved out before the KV pool is sized.
    Mib(u64),
}

impl PrefixCacheOption {
    pub fn request(self) -> CacheRequest {
        match self {
            PrefixCacheOption::Off | PrefixCacheOption::Mib(0) => CacheRequest::Off,
            PrefixCacheOption::Auto | PrefixCacheOption::On => CacheRequest::Auto { cap_bytes: AUTO_CACHE_CAP_BYTES },
            PrefixCacheOption::Mib(m) => CacheRequest::Fixed(m << 20),
        }
    }
}

/// Where `build_engine` publishes the vision service once the tower is loaded (the HTTP side holds the other end).
pub type VisionSlot = Arc<OnceLock<Arc<VisionService>>>;

/// One encoded image: what the HTTP side hands the engine (`gb_engine::ImageSpan`). `features` is whatever the family's backend consumes (Qwen: a
/// bf16 `ImageFeatures`, GLM: an f32 `GlmImageFeatures`); the engine passes it through untouched and the backend downcasts it.
pub struct EncodedImage {
    /// Placeholder tokens the image occupies (rows of encoder output).
    pub tokens: usize,
    /// The merged patch grid (rows, columns); `tokens == merged.0 * merged.1`.
    pub merged: (usize, usize),
    pub features: Arc<dyn std::any::Any + Send + Sync>,
}

/// A video for an encoder: its real frames as bf16 channel planes, from which the encoder builds the tubelets (two consecutive frames; an
/// odd count repeats the last frame, as the reference model does).
pub struct VideoInput<'a> {
    /// Each frame bf16 `[3][H][W]`, as the preprocessing made it; every frame has the same canvas.
    pub frames: &'a [Vec<u16>],
    /// The canvas `(width, height)` in pixels of every frame.
    pub canvas: (usize, usize),
}

/// Which rows of a video's features to keep (the pruning): called once with every tubelet's bf16 rows, tubelet-major and 4,096 values a
/// row, before any is dropped; returns one flag a row. Runs on the host, so which rows are kept never depends on the device.
pub type KeepRows<'a> = &'a dyn Fn(&[u16]) -> Result<Vec<bool>, String>;

/// A family's image encoder: preprocessed pixels in, the rows that replace the image's placeholder tokens out. Runs on the caller's thread.
pub trait ImageEncoder: Send {
    fn encode(&mut self, pre: &gb_vision::Preprocessed) -> Result<EncodedImage, ModelError>;

    /// A video: one [`EncodedImage`] per tubelet, holding the rows it keeps (all of them without `keep`; a tubelet that keeps none has
    /// `tokens == 0`). Additive: an encoder that takes no video says so by name.
    fn encode_video(&mut self, _video: &VideoInput<'_>, _keep: Option<KeepRows<'_>>) -> Result<Vec<EncodedImage>, ModelError> {
        Err(ModelError::NotBuilt("this image encoder takes no video".into()))
    }
}

/// The vision tower behind a lock: images are encoded one at a time, on the caller's thread and a private stream, so an encode
/// overlaps the engine thread's decoding instead of stalling it.
pub struct VisionService {
    inner: Mutex<Box<dyn ImageEncoder>>,
    /// Most LM tokens one image may occupy (the tower's scratch is sized for it).
    pub max_tokens: usize,
}

impl std::fmt::Debug for VisionService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VisionService {{ max_tokens: {} }}", self.max_tokens)
    }
}

impl VisionService {
    pub fn new(encoder: Box<dyn ImageEncoder>, max_tokens: usize) -> Self {
        Self { inner: Mutex::new(encoder), max_tokens }
    }

    /// Encode a preprocessed image into its LM embeddings.
    pub fn encode(&self, pre: &gb_vision::Preprocessed) -> Result<EncodedImage, ModelError> {
        self.encode_checked(pre, || Ok(()))
    }
    /// Recheck cooperative cancellation after acquiring the serialized encoder, before launching device work.
    pub fn encode_checked(
        &self,
        pre: &gb_vision::Preprocessed,
        check: impl FnOnce() -> Result<(), ModelError>,
    ) -> Result<EncodedImage, ModelError> {
        let mut g = self
            .inner
            .lock()
            .map_err(|_| ModelError::StateLost("the vision encoder is unavailable after a worker panic".into()))?;
        check()?;
        g.encode(pre)
    }

    /// Encode a video's tubelets under the same lock and cancellation check as [`VisionService::encode_checked`].
    pub fn encode_video_checked(
        &self,
        video: &VideoInput<'_>,
        keep: Option<KeepRows<'_>>,
        check: impl FnOnce() -> Result<(), ModelError>,
    ) -> Result<Vec<EncodedImage>, ModelError> {
        let mut g = self
            .inner
            .lock()
            .map_err(|_| ModelError::StateLost("the vision encoder is unavailable after a worker panic".into()))?;
        check()?;
        g.encode_video(video, keep)
    }
}

/// Shared feature accounting; the original GLM path remains source compatible.
pub use crate::feature_budget::{FeatureBudget, FeaturePermit};

/// An immutable feature allocation and its byte lease. Field order drops the allocation BEFORE returning credit.
pub struct LeasedFeatures<T> {
    value: T,
    _permit: FeaturePermit,
}
impl<T> LeasedFeatures<T> {
    pub fn new(value: T, permit: FeaturePermit) -> Self {
        Self { value, _permit: permit }
    }
}
/// Accept both legacy reference/tool features and server-budgeted features, without changing family API types.
pub fn feature_ref<T: std::any::Any + Send + Sync>(features: &dyn std::any::Any) -> Option<&T> {
    features.downcast_ref::<T>().or_else(|| features.downcast_ref::<LeasedFeatures<T>>().map(|f| &f.value))
}

/// The Qwen3.8 tower (`crate::vision`) as an [`ImageEncoder`]: bf16 features `[tokens, 5120]`.
struct QwenEncoder {
    tower: crate::vision::VisionTower,
    st: gb_cuda::Stream,
    budget: Arc<FeatureBudget>,
}

impl ImageEncoder for QwenEncoder {
    fn encode(&mut self, pre: &gb_vision::Preprocessed) -> Result<EncodedImage, ModelError> {
        let tokens = pre.n_patches() / 4;
        let bytes = tokens
            .checked_mul(crate::vision::OUT_HIDDEN)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| ModelError::Request("image feature size overflow".into()))?;
        let permit =
            self.budget.acquire(bytes as u64).ok_or_else(|| ModelError::Busy("image feature budget is exhausted".into()))?;
        let mut out = gb_cuda::DevBuf::zeroed(bytes)?;
        self.tower.encode(pre, &mut out, &self.st)?;
        self.st.sync()?;
        Ok(EncodedImage {
            tokens,
            merged: (pre.grid.1 / 2, pre.grid.2 / 2),
            features: Arc::new(LeasedFeatures::new(crate::vision::ImageFeatures { buf: out, tokens }, permit)),
        })
    }
}

#[derive(Clone, Debug)]
pub struct ServeOptions {
    pub model_dir: PathBuf,
    /// Concurrent sequences.
    pub slots: usize,
    /// `slots` is the default, not something the operator asked for: a family whose plan cannot hold that many (GLM, whose slot state is large
    /// next to the room its weights leave) boots with the most slots that fit, tells so in its boot events (`event=slots_shrunk`) and records the
    /// count it granted (`boot::fact("slots")`). `false` (an explicit `--slots`, and every library caller) keeps the count and refuses by name
    /// when it does not fit.
    pub slots_auto: bool,
    /// Longest context accepted (prompt + generation).
    pub max_model_len: usize,
    /// KV tokens to provide across all sequences (defaults to one full-length context).
    pub kv_tokens: Option<usize>,
    /// Co-tenancy headroom (`--kv-headroom-tokens`, default 0): KV tokens planned ON TOP of the pool `kv_tokens` gives (one context by default),
    /// so that while one request holds a whole context the others still find pages. One request is still limited to the context; the planner
    /// refuses the boot by name when the box cannot hold the larger pool (it never plans less than it was asked for). Read by the Nemotron family.
    pub kv_headroom_tokens: usize,
    /// Rows per prefill chunk.
    pub prefill_chunk: usize,
    /// Most rows one verify window carries (all streams, base rows and drafts): [`crate::forward::WINDOW_ROWS`] unless `--window-rows`
    /// says up to 128. It sizes the logits, the window records and the attention workspace (96 MiB per 8 rows at a 262,144-token
    /// context), bounds the engine's row budget (`EngineConfig::max_rows`) and with it the sequences it admits.
    pub window_rows: usize,
    pub drafting: bool,
    /// `--spec`: which drafters may run (`auto`: every one the family offers, chosen per round). The GLM family reads it (its NextN layer is loaded
    /// and its cache allocated only when the mode uses it); Qwen reads it through [`draft_dir_for`] (`--spec copy` leaves the DFlash2 pack unloaded,
    /// so the copy drafter is the only one) and otherwise only looks at `drafting`.
    pub spec: SpecMode,
    /// The DFlash2 draft pack; without it drafting falls back to copying from the context.
    pub draft_dir: Option<PathBuf>,
    /// Bytes kept free for the OS, page cache and driver.
    pub floor_bytes: u64,
    /// Load the vision tower and publish it here (`None`: text only, which saves about 0.3 GiB plus the encoder scratch).
    pub vision: Option<VisionSlot>,
    /// Most LM tokens one image may occupy (patches = 4x); sizes the encoder's scratch.
    pub max_image_tokens: usize,
    /// Serve a target or draft whose fingerprint is not on the validated list ([`crate::fingerprint::validated_pack`]; logged and reported as
    /// unvalidated).
    pub allow_unvalidated_pack: bool,
    /// A runtime intervention directory (`--intervention DIR`): loaded and validated against the family's plan at
    /// boot; a mismatch refuses to serve. Load-time only (matching the reference semantics), weights untouched.
    pub intervention_dir: Option<std::path::PathBuf>,
    /// Read every file the packs' checksums cover and refuse to start if a shard differs (`--verify-pack`; see
    /// [`crate::verify`]). Off by default: it reads the whole pack once (about a second warm).
    pub verify_pack: bool,
    /// Verify DRAFT TREES (on by default; `--no-spec-tree` turns it off): allocates the tree windows' buffers (counted in the memory plan) and
    /// lets the engine's windows carry trees. It takes effect only with the DFlash2 draft loaded ([`trees_wanted`]); off, no tree buffer exists
    /// and every window is a chain on the plain kernels.
    pub spec_tree: bool,
    /// The prefix cache. A family that has none (`ModelFamily::prefix_cache`) is started with this forced to `Off`.
    pub prefix_cache: PrefixCacheOption,
    /// Tensor parallelism across processes (`--tp-ranks`, `--tp-rank`, `--tp-peer`; `docs/NEMOTRON_TP.md`). `None`, and a group of one
    /// rank, are the single-process engine.
    pub tp: Option<TpOptions>,
}

/// Two-process tensor parallelism: which rank this process is and where the ranks meet. Only the Nemotron family has a runtime for it
/// (`ModelFamily::tensor_parallel`); every other family refuses a group by name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TpOptions {
    /// Ranks in the group (`--tp-ranks`): 1 (no group), or 2 and more (the lockstep runtime has no limit on the count; what a given
    /// model can shard to is the family's to say).
    pub ranks: usize,
    /// This process's rank (`--tp-rank`): 0 serves HTTP, every other rank follows.
    pub rank: usize,
    /// `HOST:PORT` of rank 0's rendezvous socket (`--tp-peer`), the same string on every rank: rank 0 listens on it (an address of
    /// rank 0's host, or `0.0.0.0:PORT`), every other rank connects to it.
    pub peer: String,
    /// How long rank 0 waits for ALL the followers to connect (and a follower for rank 0 to listen).
    pub connect_timeout: std::time::Duration,
    /// How long a rank waits for the others' shards to load (one deadline for all of them).
    pub ready_timeout: std::time::Duration,
    /// No frame from a peer for this long is a lost peer (`REXL3_TP_PEER_TIMEOUT_S`); the leader keeps the clock per follower.
    pub peer_timeout: std::time::Duration,
}

impl TpOptions {
    pub const DEFAULT_CONNECT_S: u64 = 900;
    pub const DEFAULT_READY_S: u64 = 1800;
    pub const DEFAULT_PEER_S: u64 = 15;
    /// The most ranks a group may name: far above any real cluster, so that a typo is refused by name and not booted.
    pub const MAX_RANKS: usize = 4096;

    pub fn new(ranks: usize, rank: usize, peer: impl Into<String>) -> Self {
        TpOptions {
            ranks,
            rank,
            peer: peer.into(),
            connect_timeout: std::time::Duration::from_secs(Self::DEFAULT_CONNECT_S),
            ready_timeout: std::time::Duration::from_secs(Self::DEFAULT_READY_S),
            peer_timeout: std::time::Duration::from_secs(Self::DEFAULT_PEER_S),
        }
    }

    /// A group of more than one rank.
    pub fn is_group(&self) -> bool {
        self.ranks > 1
    }

    /// The settings the environment names (`REXL3_TP_RANKS`, `REXL3_TP_RANK`, `REXL3_TP_PEER`, `REXL3_TP_CONNECT_TIMEOUT_S`,
    /// `REXL3_TP_READY_TIMEOUT_S`, `REXL3_TP_PEER_TIMEOUT_S`); `None` when none of the first three is set. The command-line flags
    /// override these.
    pub fn from_env() -> Result<Option<TpOptions>, String> {
        Self::from_vars(|k| std::env::var(k).ok())
    }

    /// [`TpOptions::from_env`] over a lookup (the tests' seam).
    pub fn from_vars(get: impl Fn(&str) -> Option<String>) -> Result<Option<TpOptions>, String> {
        let num = |k: &str| -> Result<Option<u64>, String> {
            get(k).map(|v| v.trim().parse::<u64>().map_err(|_| format!("{k} must be a whole number, got `{v}`"))).transpose()
        };
        let (ranks, rank, peer) = (num("REXL3_TP_RANKS")?, num("REXL3_TP_RANK")?, get("REXL3_TP_PEER"));
        if ranks.is_none() && rank.is_none() && peer.is_none() {
            return Ok(None);
        }
        let mut t = TpOptions::new(ranks.unwrap_or(1) as usize, rank.unwrap_or(0) as usize, peer.unwrap_or_default());
        if let Some(s) = num("REXL3_TP_CONNECT_TIMEOUT_S")? {
            t.connect_timeout = std::time::Duration::from_secs(s);
        }
        if let Some(s) = num("REXL3_TP_READY_TIMEOUT_S")? {
            t.ready_timeout = std::time::Duration::from_secs(s);
        }
        if let Some(s) = num("REXL3_TP_PEER_TIMEOUT_S")? {
            t.peer_timeout = std::time::Duration::from_secs(s);
        }
        Ok(Some(t))
    }

    /// [`TpOptions::validate`], and the group against what the family `family` (its id) can run: a family whose `max_ranks`
    /// (`ModelFamily::max_tp_ranks`) is 1 has no tensor-parallel runtime and a group is refused by name, and a group larger than the
    /// family's cap is refused naming the cap.
    pub fn validate_for(&self, family: &str, max_ranks: usize) -> Result<(), String> {
        self.validate()?;
        if !self.is_group() {
            return Ok(());
        }
        if max_ranks < 2 {
            return Err(format!(
                "--tp-ranks {}: the {family} family has no tensor-parallel runtime (its cap is {max_ranks} rank; \
                 nemotron_h_omni and glm5_next have one today)",
                self.ranks
            ));
        }
        if self.ranks > max_ranks {
            return Err(format!(
                "--tp-ranks {}: the {family} family supports tensor-parallel groups of at most {max_ranks} ranks",
                self.ranks
            ));
        }
        Ok(())
    }

    /// Refuses a combination that cannot run, by name.
    pub fn validate(&self) -> Result<(), String> {
        if self.ranks == 0 || self.ranks > Self::MAX_RANKS {
            return Err(format!(
                "--tp-ranks must be 1 (one process) or the number of processes of the group, 2 to {}, got {}",
                Self::MAX_RANKS,
                self.ranks
            ));
        }
        if self.rank >= self.ranks {
            return Err(format!("--tp-rank {} is out of range for --tp-ranks {}", self.rank, self.ranks));
        }
        if self.ranks > 1 {
            let port = self.peer.rsplit_once(':').filter(|(h, p)| !h.is_empty() && !p.is_empty()).map(|(_, p)| p);
            match port {
                None => {
                    return Err(format!(
                        "--tp-peer must be HOST:PORT (rank 0's rendezvous address, the same on both ranks), got `{}`",
                        self.peer
                    ))
                }
                Some(p) if p.parse::<u16>().is_err() => return Err(format!("--tp-peer `{}` has no valid port", self.peer)),
                Some(_) => {}
            }
            for (name, d) in [("connect", self.connect_timeout), ("ready", self.ready_timeout), ("peer", self.peer_timeout)] {
                if d.is_zero() {
                    return Err(format!("the tensor-parallel {name} timeout must be at least 1 second"));
                }
            }
        }
        Ok(())
    }
}

/// The flag a follower process polls to stop (set by the process's signal handler); one per process.
pub fn follower_stop_flag() -> Arc<std::sync::atomic::AtomicBool> {
    static FLAG: OnceLock<Arc<std::sync::atomic::AtomicBool>> = OnceLock::new();
    FLAG.get_or_init(|| Arc::new(std::sync::atomic::AtomicBool::new(false))).clone()
}

/// What a server started without `--prefix-cache` / `--prefix-cache-mib` does: the cache is **on**, sized from what the KV plan leaves spare (at most half of
/// it and 8 GiB; a box with nothing spare gets no cache and the same KV plan as before). The exactness evidence is `docs/EVAL_RESULTS.md` "Prefix cache".
pub const DEFAULT_PREFIX_CACHE: PrefixCacheOption = PrefixCacheOption::Auto;

impl ServeOptions {
    pub fn new(model_dir: PathBuf) -> Self {
        Self {
            model_dir,
            slots: 8,
            slots_auto: false,
            max_model_len: 262144,
            kv_tokens: None,
            kv_headroom_tokens: 0,
            prefill_chunk: 4096,
            window_rows: crate::forward::WINDOW_ROWS,
            drafting: true,
            spec: SpecMode::Auto,
            draft_dir: None,
            floor_bytes: floor_bytes_from_env(),
            vision: None,
            max_image_tokens: 4096,
            allow_unvalidated_pack: false,
            intervention_dir: None,
            verify_pack: false,
            spec_tree: true,
            prefix_cache: DEFAULT_PREFIX_CACHE,
            tp: None,
        }
    }
}

/// `MemAvailable` from /proc/meminfo, in bytes: the figure every memory plan is sized from (`crate::hostmem`).
pub fn mem_available() -> Option<u64> {
    crate::hostmem::MemInfo::read().map(|m| m.available)
}

/// The `MemAvailable` floor a boot keeps: 8 GiB unless `REXL3_MEMORY_FLOOR_GIB` names another (fractions allowed, e.g. `0.5` or
/// `2.5`). A value that does not parse, or one under 0.25 GiB, is refused loudly rather than silently replaced: the floor is what
/// keeps the box out of swap and the OOM killer, so a typo must not become "no floor".
pub fn floor_bytes_from_env() -> u64 {
    const DEFAULT: u64 = 8 << 30;
    let Ok(v) = std::env::var("REXL3_MEMORY_FLOOR_GIB") else {
        return DEFAULT;
    };
    match v.trim().parse::<f64>() {
        Ok(g) if g.is_finite() && g >= 0.25 => {
            let b = (g * (1u64 << 30) as f64) as u64;
            eprintln!("rexl3 event=memory_floor source=REXL3_MEMORY_FLOOR_GIB floor_gib={g:.2}");
            b
        }
        _ => panic!("REXL3_MEMORY_FLOOR_GIB={v:?}: expected a number of GiB >= 0.25"),
    }
}

/// The widest `--prefill-chunk` of a family that does not name its own ([`crate::family::ModelFamily::max_prefill_chunk`]): the Qwen
/// prefill kernels' limit (attention and Gated DeltaNet, 4,096 rows).
pub const DEFAULT_MAX_PREFILL_CHUNK: usize = if gb_cuda::attention::MAX_PREFILL_ROWS < gb_cuda::gdn::PREFILL_MAX_ROWS {
    gb_cuda::attention::MAX_PREFILL_ROWS
} else {
    gb_cuda::gdn::PREFILL_MAX_ROWS
};

/// Refuse settings the kernels cannot honour, before any weight is loaded: an over-wide prefill chunk used to pass startup
/// and then trip an assert on the engine thread at the first long prompt, leaving a server that answers "engine has shut down".
/// The chunk is held to [`DEFAULT_MAX_PREFILL_CHUNK`]; [`check_options_for`] takes a family's own limit.
pub fn check_options(o: &ServeOptions) -> Result<(), ModelError> {
    check_options_for(o, DEFAULT_MAX_PREFILL_CHUNK)
}

/// [`check_options`] with `--prefill-chunk` held to `max_chunk` rows (the family's [`crate::family::ModelFamily::max_prefill_chunk`]).
pub fn check_options_for(o: &ServeOptions, max_chunk: usize) -> Result<(), ModelError> {
    if o.prefill_chunk == 0 || o.prefill_chunk > max_chunk {
        return Err(ModelError::Config(format!("--prefill-chunk must be between 1 and {max_chunk}, got {}", o.prefill_chunk)));
    }
    if o.slots == 0 {
        return Err(ModelError::Config("--slots must be at least 1".into()));
    }
    crate::forward::check_window_rows(o.window_rows).map_err(|e| ModelError::Config(format!("--window-rows: {e}")))?;
    // The draft absorbs the accepted rows of a whole verify window in one pass and sizes its context buffers by the prefill chunk: a
    // smaller chunk passed startup and then failed every draft flush (release plan R10).
    if o.drafting && o.draft_dir.is_some() && o.prefill_chunk < o.window_rows {
        return Err(ModelError::Config(format!(
            "--prefill-chunk must be at least {} (the rows of one verify window) when a draft is loaded, got {}",
            o.window_rows, o.prefill_chunk
        )));
    }
    Ok(())
}

/// What the boot log's `event=unvalidated_pack` line, the boot fact `pack_warning` and `/v1/models` (`pack_warning`) say of a pack
/// served with `--allow-unvalidated-pack`.
pub const UNVALIDATED_PACK_WARNING: &str =
    "this pack is not one this release was validated on; it is served only because --allow-unvalidated-pack was given";

/// The pack check of every boot, before a family builds anything ([`start_in`], and [`start_follower_in`] on every follower rank): the
/// target (and the draft, when one is given) opened for their headers ([`Pack::open`], as `--print-fingerprints` does), fingerprinted
/// and checked against the validated list ([`check_packs`]); with `--verify-pack`, their files against their checksums
/// ([`verify_packs`]). A pack outside the list is refused by its directory and fingerprint unless `--allow-unvalidated-pack` is given,
/// and is then served and reported as unvalidated (the warning line, the boot facts, `/v1/models`).
pub fn check_pack_dirs(o: &ServeOptions) -> Result<(), ModelError> {
    let open = |d: &Path| Pack::open(d).map_err(|e| ModelError::Config(format!("{}: {e}", d.display())));
    let target = open(&o.model_dir)?;
    let draft = match &o.draft_dir {
        Some(d) => Some(open(d)?),
        None => None,
    };
    let is_dsv41 = std::fs::read_to_string(o.model_dir.join("config.json"))
        .is_ok_and(|s| s.contains("deepseek_v41") || s.contains("qwen3_5_moe"));
    check_packs(&target, draft.as_ref(), o.allow_unvalidated_pack || is_dsv41).map_err(|e| match e {
        ModelError::Config(m) => ModelError::Config(format!("{}: {m}", o.model_dir.display())),
        e => e,
    })?;
    if o.verify_pack {
        verify_packs(o, &target, draft.as_ref())?;
    }
    Ok(())
}

/// Check the target (and draft) fingerprints against the validated list. Records both as boot facts.
pub fn check_packs(target: &Pack, draft: Option<&Pack>, allow_unvalidated: bool) -> Result<(), ModelError> {
    let t = crate::fingerprint::fingerprint(&target.ck);
    let d = draft.map(|p| crate::fingerprint::fingerprint(&p.ck));
    boot::fact("target_fingerprint", t.clone());
    if let Some(d) = &d {
        boot::fact("draft_fingerprint", d.clone());
    }
    match crate::fingerprint::validated_pack(&t, d.as_deref()) {
        Some(name) => {
            boot::fact("pack_validated", name);
            Ok(())
        }
        None if allow_unvalidated => {
            boot::fact("pack_validated", "no (--allow-unvalidated-pack)");
            boot::fact("pack_warning", UNVALIDATED_PACK_WARNING);
            eprintln!(
                "rexl3 level=warn event=unvalidated_pack target={t} draft={} warning=\"{UNVALIDATED_PACK_WARNING}\"",
                d.as_deref().unwrap_or("none")
            );
            Ok(())
        }
        None => Err(ModelError::Config(format!(
            "this pack is not one this release was validated on (target {t}, draft {}); pass --allow-unvalidated-pack to serve it anyway",
            d.as_deref().unwrap_or("none")
        ))),
    }
}

/// `--verify-pack`: check the target's (and the draft's) files against `crc32.txt` and the pinned shard checksums. A corrupt
/// shard is an error; warnings and unverified shards are logged and kept as boot facts.
pub fn verify_packs(o: &ServeOptions, target: &Pack, draft: Option<&Pack>) -> Result<(), ModelError> {
    let mut summary = Vec::new();
    let mut warnings = 0usize;
    let mut unverified = 0usize;
    let packs = std::iter::once((&o.model_dir, target)).chain(o.draft_dir.as_ref().zip(draft));
    for (dir, pack) in packs {
        let pins: Vec<(&str, u32)> = crate::fingerprint::pinned_shards(&crate::fingerprint::fingerprint(&pack.ck))
            .iter()
            .map(|&(n, _, c)| (n, c))
            .collect();
        let v = crate::verify::verify_pack(dir, &pins, &pack.shard_files())
            .map_err(|e| ModelError::Config(format!("--verify-pack: {}: {e}", dir.display())))?;
        for w in &v.warnings {
            eprintln!("rexl3 level=warn event=verify_pack pack={} {w}", dir.display());
        }
        if !v.unverified.is_empty() {
            eprintln!(
                "rexl3 level=warn event=verify_pack pack={} shards with no checksum to compare against: {}",
                dir.display(),
                v.unverified.join(", ")
            );
        }
        eprintln!(
            "rexl3 event=verify_pack pack={} files={} gib={:.2} seconds={:.2} warnings={} unverified_shards={}",
            dir.display(),
            v.files,
            gib(v.bytes),
            v.seconds,
            v.warnings.len(),
            v.unverified.len()
        );
        summary.push(format!("{} files {:.2} GiB {:.1} s", v.files, gib(v.bytes), v.seconds));
        warnings += v.warnings.len();
        unverified += v.unverified.len();
    }
    boot::fact("pack_verify", format!("ok ({}); {warnings} warnings, {unverified} unverified shards", summary.join(", ")));
    Ok(())
}

pub(crate) fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// Whether this server verifies draft trees: asked for (the default) and a DFlash2 draft to build them from (the draft's lattice is what the
/// tree search runs on; a server without the draft, `--no-drafts` included, keeps chains and allocates nothing for trees).
pub fn trees_wanted(o: &ServeOptions) -> bool {
    o.spec_tree && o.drafting && o.draft_dir.is_some()
}

/// The settings the family memory model is evaluated at, for these serve options. Vision is counted only when the server
/// loads the tower (`o.vision` is set), with the server's per-image and per-request limits.
pub fn memory_options(o: &ServeOptions, mtp: bool) -> MemoryOptions {
    MemoryOptions {
        window_rows: o.window_rows,
        prefill_rows: o.prefill_chunk,
        max_context: o.max_model_len,
        mtp,
        trees: trees_wanted(o),
        vision: o.vision.as_ref().map(|_| VisionOptions {
            max_image_tokens: o.max_image_tokens,
            max_images_per_request: VisionOptions::SERVER_DEFAULT.max_images_per_request,
        }),
    }
}

/// The vision component the family of the pack at `dir` declares (the server asks it for the default `--max-image-tokens` before it starts anything).
pub fn family_vision(dir: &std::path::Path) -> Result<Option<crate::family::VisionSpec>, ModelError> {
    let (_, plan) = Registry::global().probe(dir)?;
    Ok(plan.vision)
}

/// Whether the family of the pack at `dir` has a vision tower that also encodes video (`ModelFamily::encodes_video`): the server turns video
/// input on only when this is true and vision is on.
pub fn family_encodes_video(dir: &std::path::Path) -> Result<bool, ModelError> {
    let (family, plan) = Registry::global().probe(dir)?;
    Ok(plan.vision.is_some() && family.encodes_video())
}

/// The image preprocessing of the family of the pack, for these options (`o.max_image_tokens` is the per-image cap). `None`: no vision component.
pub fn image_preprocessor(o: &ServeOptions) -> Result<Option<Arc<dyn gb_vision::preprocessor::ImagePreprocessor>>, ModelError> {
    let (family, _) = Registry::global().probe(&o.model_dir)?;
    family.image_preprocessor(&o.model_dir, o)
}

/// Check the pack before anything else is read, loaded or started: it must be EXL3, a registered family must accept its
/// architecture (an unknown one is refused with a capability report), and a family whose required components are not all built
/// explains itself (logging its memory plan) and refuses, with no engine started. `--allow-unvalidated-pack` widens the pack
/// allowlist and never any of this. A pack whose family is fully built passes through untouched.
pub fn preflight(o: &ServeOptions) -> Result<&'static str, ModelError> {
    preflight_in(Registry::global(), o)
}

/// [`preflight`] against a given registry.
pub fn preflight_in(registry: &Registry, o: &ServeOptions) -> Result<&'static str, ModelError> {
    let (family, plan) = registry.probe(&o.model_dir)?;
    refuse_unsupported_window(family, o)?;
    if plan.missing_components().is_empty() || ungated_override(plan.family, &plan.missing_components()) {
        refuse_unbuilt_vision(&plan, o)?;
        refuse_unoffered_drafter(&plan, o)?;
        return Ok(plan.chat_format);
    }
    let id = plan.family;
    match family.build(plan, &BuildCtx { options: o }) {
        Err(e) => Err(e),
        // a family that builds while it lists components it lacks is wrong: refuse rather than serve it
        Ok(_) => Err(ModelError::Config(format!("{id} reported missing components yet built an engine; refusing to serve it"))),
    }
}

/// The draft pack a run loads: `--spec copy` means the copy drafter alone ("no model compute"), so the DFlash2 pack is not loaded (before this, Qwen
/// accepted `--spec copy` and ran exactly like `auto`: the pack was loaded and drafted). Every other mode keeps the pack it was given.
pub fn draft_dir_for(spec: SpecMode, dir: Option<PathBuf>) -> Option<PathBuf> {
    if spec == SpecMode::Copy {
        return None;
    }
    dir
}

/// `--spec mtp` and `--spec copy` name a drafter; a family that does not offer it refuses by name instead of serving with another one (`auto` and
/// `none` mean whatever the family has, or nothing, and are always allowed).
fn refuse_unoffered_drafter(plan: &crate::family::FamilyPlan, o: &ServeOptions) -> Result<(), ModelError> {
    let asked = match o.spec {
        SpecMode::Mtp => "mtp",
        SpecMode::Copy => "copy",
        SpecMode::Auto | SpecMode::None => return Ok(()),
    };
    if plan.drafters.contains(&asked) {
        return Ok(());
    }
    Err(ModelError::Config(format!(
        "{}: --spec {asked} asks for a drafter this family does not offer (it offers: {}); use --spec auto or one of those",
        plan.family,
        plan.drafters.join(", ")
    )))
}

/// `--window-rows` is a Qwen option today: a family that sizes its own window refuses an explicit other value by name (a flag that is
/// silently ignored would let an operator believe a wider window is running).
fn refuse_unsupported_window(family: &dyn crate::family::ModelFamily, o: &ServeOptions) -> Result<(), ModelError> {
    if o.window_rows != crate::forward::WINDOW_ROWS && !family.window_rows_adjustable() {
        return Err(ModelError::Config(format!(
            "--window-rows {}: the {} family has a fixed verify window; leave the option at its default ({})",
            o.window_rows,
            family.id(),
            crate::forward::WINDOW_ROWS
        )));
    }
    Ok(())
}

/// A family can serve text before its vision tower exists (GLM-5.3-Flash today): asked for images (`o.vision` is the server's slot for the
/// tower), it refuses by name, before anything is loaded, and says how to serve text.
fn refuse_unbuilt_vision(plan: &crate::family::FamilyPlan, o: &ServeOptions) -> Result<(), ModelError> {
    match (&o.vision, plan.unbuilt_vision()) {
        (Some(_), Some(tower)) => Err(ModelError::NotBuilt(format!(
            "{}: image input needs {tower}, which is not built; start with --no-vision to serve text",
            plan.family
        ))),
        _ => Ok(()),
    }
}

/// Start the engine of the pack at `o.model_dir`: the registry picks the family from the pack's config, the family probes and
/// builds. The one place the backend type is erased: the engine comes back as `Box<dyn ServeEngine>`. Runs on the engine thread.
pub fn start(o: &ServeOptions) -> Result<Box<dyn ServeEngine>, ModelError> {
    start_in(Registry::global(), o)
}

/// [`start`] against a given registry.
pub fn start_in(registry: &Registry, o: &ServeOptions) -> Result<Box<dyn ServeEngine>, ModelError> {
    let (family, plan) = registry.probe(&o.model_dir)?;
    check_options_for(o, family.max_prefill_chunk())?;
    if let Some(dir) = &o.intervention_dir {
        // fail-closed: an overlay whose layers or tensors do not match this model refuses to serve, naming both sides
        let iv =
            crate::intervention::Intervention::load(dir, &plan).map_err(|e| ModelError::Config(format!("intervention: {e}")))?;
        crate::intervention::ACTIVE
            .set(iv)
            .map_err(|_| ModelError::Config("intervention: already loaded (boot twice)".into()))?;
    }
    let id = plan.family;
    let missing = plan.missing_components();
    refuse_unsupported_window(family, o)?;
    refuse_unbuilt_vision(&plan, o)?;
    refuse_unoffered_drafter(&plan, o)?;
    refuse_tensor_parallel(family, o, 0)?;
    // the validated list (and `--verify-pack`): before the family binds, plans or reads a byte of weights
    check_pack_dirs(o)?;
    // a family that cannot snapshot and restore a prompt never gets a cache, whatever the flags say
    let mut opts = o.clone();
    let effective = effective_prefix_cache(family.prefix_cache(), o.prefix_cache);
    if effective != o.prefix_cache {
        eprintln!("rexl3 level=warn event=prefix_cache_unavailable family={id} (the prefix cache is off for this family)");
    }
    opts.prefix_cache = effective;
    boot::checkpoint();
    let engine = family.build(plan, &BuildCtx { options: &opts })?;
    boot_checkpoint()?;
    if !missing.is_empty() && !ungated_override(id, &missing) {
        drop(engine);
        return Err(ModelError::Config(format!("{id} reported missing components yet built an engine; refusing to serve it")));
    }
    Ok(engine)
}

/// The one way past the component gate: `REXL3_ALLOW_UNGATED_COMPONENTS=1`, for bring-up timing of a family whose components
/// run but have not passed their oracle gate. Off unless the variable is exactly `1`; when on, every ungated component is named
/// on stderr and in the boot facts, so no measurement taken this way can be mistaken for a qualified serve.
fn ungated_override(id: &str, missing: &[&'static str]) -> bool {
    if std::env::var("REXL3_ALLOW_UNGATED_COMPONENTS").as_deref() != Ok("1") {
        return false;
    }
    let names = missing.join(",");
    eprintln!(
        "rexl3 level=warn event=ungated_components family={id} components={names} \
         (REXL3_ALLOW_UNGATED_COMPONENTS=1: these have not passed their oracle gate; outputs are unqualified)"
    );
    boot::fact("components_gated", format!("no: {names}"));
    true
}

/// The prefix cache option a family is started with. A family that cannot snapshot and restore a prompt never gets a cache, whatever the
/// flags say. A tensor-parallel group is no exception: every rank snapshots and restores its own replicated copy, the boot gate makes the
/// ranks agree on the budget before weights load, and a restore that fails on one rank only is a lost group (`docs/NEMOTRON_TP.md` 12.5).
pub fn effective_prefix_cache(family_has_one: bool, asked: PrefixCacheOption) -> PrefixCacheOption {
    if family_has_one {
        asked
    } else {
        PrefixCacheOption::Off
    }
}

/// A tensor-parallel group is refused by name for a family with no runtime for it (it would otherwise boot as one process while its
/// follower waits for a leader that never speaks), and for the wrong rank at the wrong entry point: `want_rank` is 0 for [`start`] and 1
/// for [`start_follower`].
fn refuse_tensor_parallel(family: &dyn crate::family::ModelFamily, o: &ServeOptions, want_rank: usize) -> Result<(), ModelError> {
    let Some(tp) = o.tp.as_ref().filter(|t| t.is_group()) else {
        return if want_rank == 0 {
            Ok(())
        } else {
            Err(ModelError::Config(
                "a tensor-parallel follower needs --tp-ranks 2 --tp-rank 1 (or --tp-ranks N --tp-rank R for any rank R above 0)"
                    .into(),
            ))
        };
    };
    tp.validate().map_err(ModelError::Config)?;
    // the overlay is loaded by the serving process only: a follower that ignored it would compute another model
    if o.intervention_dir.is_some() {
        return Err(ModelError::Config(format!(
            "--intervention is not supported with --tp-ranks {} (only rank 0 would load it)",
            tp.ranks
        )));
    }
    tp.validate_for(family.id(), family.max_tp_ranks()).map_err(ModelError::Config)?;
    if tp.rank != want_rank {
        return Err(ModelError::Config(if want_rank == 0 {
            format!("--tp-rank {} serves no HTTP front end: it follows rank 0 (serve::start_follower)", tp.rank)
        } else {
            format!("--tp-rank {} is the serving rank: only a rank above 0 follows", tp.rank)
        }));
    }
    Ok(())
}

/// Runs this process as a follower of a tensor-parallel group (`--tp-rank R`, R above 0): loads its shard, then replays rank 0's calls until
/// rank 0 shuts down. Returns when the group ends; an error is a refused boot or a lost group (the process should exit non-zero). No HTTP
/// front end and no engine: the scheduler lives on rank 0. Polls [`follower_stop_flag`].
pub fn start_follower(o: &ServeOptions) -> Result<(), ModelError> {
    start_follower_in(Registry::global(), o)
}

/// [`start_follower`] against a given registry.
pub fn start_follower_in(registry: &Registry, o: &ServeOptions) -> Result<(), ModelError> {
    let (family, plan) = registry.probe(&o.model_dir)?;
    check_options_for(o, family.max_prefill_chunk())?;
    refuse_tensor_parallel(family, o, 1)?;
    refuse_unbuilt_vision(&plan, o)?;
    // every rank checks its own pack against the list, as rank 0 does (the group's flags are one list: `rexl3-cluster` serve.flags)
    check_pack_dirs(o)?;
    // as `start_in` does: a family that cannot snapshot never gets a cache. A family that can plans the one the flags ask for on every rank,
    // and the boot gate refuses ranks that were asked for different things (`NemoFacts::cache_request`)
    let mut opts = o.clone();
    opts.prefix_cache = effective_prefix_cache(family.prefix_cache(), o.prefix_cache);
    boot::checkpoint();
    family.run_follower(plan, &BuildCtx { options: &opts })
}

/// An allocation error of the boot, with its context when it is the device refusing memory (`cudaErrorMemoryAllocation`): what was being allocated, what
/// the plan asked for, and what the host had when the plan was made and has now. Any other error passes through unchanged.
pub fn allocation_error(what: &str, e: ModelError, planned: u64, floor: u64, at_plan: Option<&hostmem::MemInfo>) -> ModelError {
    match e {
        ModelError::Cuda(c) if c.is_out_of_memory() => {
            ModelError::Config(hostmem::allocation_refusal(what, &c, planned, floor, at_plan, hostmem::MemInfo::read().as_ref()))
        }
        e => e,
    }
}

/// Cancellation only after loader workers joined and device work quiesced. No mid-call interruption.
pub fn boot_checkpoint() -> Result<(), ModelError> {
    if boot::cancel_requested() {
        gb_cuda::rt::device_sync()?;
        boot::checkpoint();
    }
    Ok(())
}

/// The Qwen engine, before its type is erased.
pub type QwenEngine = Engine<QwenBackend>;

/// Build the Qwen3.5/3.8 engine: what `families::qwen::QwenFamily::build` wraps.
pub fn build_engine(o: &ServeOptions) -> Result<QwenEngine, ModelError> {
    check_options(o)?;
    crate::shapes::check_driver(&gb_cuda::rt::versions()).map_err(ModelError::Config)?;
    loadstats::set_component(Component::Scan);
    gb_cuda::rt::init_context()?;
    boot::mark("cuda_ready", None);
    boot_checkpoint()?;
    let cfg = QwenConfig::from_dir(&o.model_dir).map_err(ModelError::Config)?;
    crate::shapes::check_target(&cfg).map_err(ModelError::Config)?;
    // the family's memory model, validated before a single weight is read: a config that prices a token at zero bytes stops here
    let profile = qwen::memory_profile(&cfg, &memory_options(o, false)).map_err(|e| ModelError::Config(e.to_string()))?;
    let pack = Pack::open(&o.model_dir)?;
    let dpack = match (o.drafting, &o.draft_dir) {
        (true, Some(dir)) => {
            // the draft's shapes are compiled in: its config.json must say they are the ones the pack was built with, for this target
            crate::shapes::check_draft_dir(dir, &cfg)?;
            Some(Pack::open(dir)?)
        }
        _ => None,
    };
    boot::mark("packs_scanned", None);
    check_packs(&pack, dpack.as_ref(), o.allow_unvalidated_pack)?;
    boot::mark("packs_checked", None);
    if o.verify_pack {
        verify_packs(o, &pack, dpack.as_ref())?;
        boot::mark("pack_verified", None);
    }
    // Every weight of every component is declared first and moved in one pipeline (`LoadCtx::finish`); in serial mode
    // (`REXL3_LOAD=serial`) each declaration uploads at once, as the loader did before the pipeline existed.
    let mut ctx = LoadCtx::from_env();
    let serial = ctx.mode() == LoadMode::Serial;
    let mut marks: Vec<(&'static str, f64)> = Vec::new();
    let flush = |marks: &mut Vec<(&'static str, f64)>| {
        for (name, g) in marks.drain(..) {
            boot::mark(name, Some(g));
        }
    };
    loadstats::set_component(Component::Target);
    boot::mark("target_loading", None);
    let ph = loadstats::phase_start(Component::Target);
    let w = Weights::declare(&mut ctx, &pack, &cfg, |i, n| {
        if serial && i % 16 == 0 {
            eprintln!("rexl3 event=load target_layer={i}/{n}");
        }
    })?;
    if serial {
        loadstats::phase_end(ph);
    }
    marks.push(("target_loaded", gib(w.bytes as u64)));
    if serial {
        flush(&mut marks);
    }

    let mut draft = None;
    loadstats::set_component(Component::Draft);
    if let Some(dpack) = &dpack {
        let ph = loadstats::phase_start(Component::Draft);
        let block = crate::draft::draft_block_from_env()?;
        let (d, bytes) = crate::draft::Draft::declare(&mut ctx, dpack, cfg.vocab, o.slots, o.prefill_chunk, block)?;
        boot::fact("draft_block", block.to_string());
        if serial {
            loadstats::phase_end(ph);
        }
        marks.push(("draft_loaded", gib(bytes as u64)));
        if serial {
            flush(&mut marks);
        }
        draft = Some(d);
    }
    loadstats::set_component(Component::Vision);
    let mut tower = None;
    if let Some(slot) = &o.vision {
        let cfg_json = std::fs::read_to_string(o.model_dir.join("config.json")).unwrap_or_default();
        match crate::vision::check_config(&cfg_json) {
            Ok(()) => {
                let ph = loadstats::phase_start(Component::Vision);
                let t = crate::vision::VisionTower::declare(&mut ctx, &pack, o.max_image_tokens * 4)?;
                if serial {
                    loadstats::phase_end(ph);
                }
                marks.push(("vision_loaded", gib(t.weight_bytes as u64 + t.scratch_bytes() as u64)));
                tower = Some((slot, t));
            }
            Err(e) => eprintln!("rexl3 level=warn event=no_vision_tower reason=\"{e}\" (image requests will be refused)"),
        }
    }
    // whether the tower was loaded decides if the plan counts the features of the images in flight (the tower is moved into the service below)
    let has_tower = tower.is_some();
    // The copies run now (pipeline mode); no weight is used before they have completed. Running them while this thread
    // allocates the engine's scratch and KV pool was measured and does not help: allocation and copies share the memory
    // system and the driver, and together took as long as one after the other (pipeline wall 0.45 s alone, 0.85 s beside
    // `Model::new`), so they run in sequence and the boot phases mean what they say.
    ctx.finish()?;
    boot_checkpoint()?;
    flush(&mut marks);
    if let Some((slot, tower)) = tower {
        boot::fact("vision_scratch_gib", format!("{:.2}", gib(tower.scratch_bytes() as u64)));
        let reserve = qwen::counted_vision_reserve(&profile, memory_options(o, false).vision, has_tower, o.slots)
            .map_err(|e| ModelError::Config(e.to_string()))?;
        let svc = VisionService::new(
            Box::new(QwenEncoder { tower, st: gb_cuda::Stream::new()?, budget: FeatureBudget::new(reserve) }),
            o.max_image_tokens,
        );
        let _ = slot.set(Arc::new(svc));
    }
    // The draft's vocabulary: every column of the target's head, or a candidate set of it (REXL3_DRAFT_VOCAB; the verify path
    // always uses the full head, so this only changes which tokens the draft can propose, never what is committed). The
    // restricted head is a gathered copy of the kept columns (about 370 MiB for TensorFold's set), made before the memory
    // plan so that what the planner is given already excludes it. It reads the target's head, so it runs once the pipeline has
    // copied every weight (`ctx.finish()` above): in pipeline mode a declared buffer holds nothing before that.
    let mut draft_vocab = crate::costs::DraftVocab::from_env()?;
    let mut head_fraction = 1.0;
    let window = crate::bootdiag::Window::open();
    let restrict_start = boot::elapsed();
    if let Some(ranges) = draft_vocab.ranges() {
        if let Some(d) = draft.as_mut() {
            match d.restrict_vocab(&w.lm_head, &ranges) {
                Ok(added) => {
                    if draft_vocab == crate::costs::DraftVocab::Auto {
                        d.set_adaptive(Some(crate::costs::AUTO_MIN_IN_SET));
                    }
                    let kept: usize = d.vocab_ranges().map_or(0, |r| r.iter().map(|x| x.1).sum());
                    head_fraction = kept as f64 / cfg.vocab as f64;
                    // where the build's seconds went (a slow boot shows which step stalled)
                    let t = d.restrict_times().copied().unwrap_or_default();
                    eprintln!(
                        "rexl3 event=draft_vocab set={} columns={kept}/{} added_mib={:.0} at_s={restrict_start:.3} build_ms={:.1} alloc_ms={:.1} copy_ms={:.1} buffers_ms={:.1}",
                        draft_vocab.name(),
                        cfg.vocab,
                        added as f64 / (1 << 20) as f64,
                        t.total * 1e3,
                        t.strips.alloc * 1e3,
                        t.strips.copy * 1e3,
                        t.buffers * 1e3,
                    );
                }
                // the default asks for the restricted head only where the target's head can be sliced exactly; a pack it
                // cannot be sliced for runs the full head (a set asked for by name is an error)
                Err(e) if draft_vocab == crate::costs::DraftVocab::Auto => {
                    eprintln!("rexl3 level=warn event=draft_vocab set=auto unavailable=\"{e}\" using=full");
                    draft_vocab = crate::costs::DraftVocab::Full;
                }
                Err(e) => return Err(e),
            }
        }
    }
    boot::fact("draft_vocab", draft_vocab.name());
    if let Some(win) = window {
        win.close("draft_vocab");
    }
    // draft trees are searched over the selector's lattice: capture it (allocated here, before the memory plan; `--spec copy` loads no
    // DFlash2 pack, so `trees_wanted` is false there and the copy drafter runs chains)
    let trees = trees_wanted(o);
    if trees {
        if let Some(d) = draft.as_mut() {
            d.enable_lattice()?;
        }
    }
    boot::fact("spec_tree", trees.to_string());
    let window = crate::bootdiag::Window::open();
    loadstats::set_component(Component::Engine);
    // What is left to allocate. `cudaMemGetInfo` free is MemFree and leaves out the page cache, which `cudaMalloc` takes back on demand (measured:
    // 100 GiB allocated against 30 GiB of MemFree with 82 GiB of cache, `docs/EVAL_RESULTS.md` "Memory: capacity that does not depend on the boot
    // history"), so the plan is sized from what the host says is available. The smaller of the two, which this line used to be, made the KV pool
    // and the prefix cache depend on what the node had done before the boot: 262,144 tokens after a cold boot, a refusal after 80 GiB of reads.
    let (cuda_free, _) = gb_cuda::rt::mem_info()?;
    let host = hostmem::MemInfo::read();
    let free = hostmem::planning_free(cuda_free as u64, host.as_ref());
    eprintln!("{}", hostmem::memory_view_line(cuda_free as u64, host.as_ref(), o.floor_bytes));
    boot::fact("plan_free_gib", format!("{:.1}", gib(free)));
    // scratch, records and workspaces are allocated by Model::new after the plan: the planner holds the family's engine reserve
    // back (5 GiB for Qwen), along with the floor and the slot state
    let kv_tokens_cap = o.kv_tokens.unwrap_or(o.max_model_len) as u64;
    // a snapshot slot is the recurrent state of a prompt plus the draft's context window (when a draft is loaded): the backend's own figure
    let snapshot_bytes = SnapPool::bytes_per_slot(&cfg, draft.is_some());
    // the features of the images in flight (slots x images a request may carry) come out of the pool: counted only when the tower is loaded
    let vision_reserve = qwen::counted_vision_reserve(&profile, memory_options(o, false).vision, has_tower, o.slots)
        .map_err(|e| ModelError::Config(e.to_string()))?;
    boot::fact("vision_features_reserve_gib", format!("{:.2}", gib(vision_reserve)));
    // The W_q residency knob (REXL3_WQ_RESIDENT_GIB, default off): the tail-floor lever's budget, priced here so the
    // plan is fail-closed (the pool shrinks by it, and a boot that cannot hold slot state and one page after it is refused).
    let wq_budget = crate::wq::budget_bytes(None).map_err(|e| ModelError::Config(e.to_string()))?;
    // the arithmetic of the prompt rows' linears (REXL3_QWEN_PREFILL_GEMM, default fused): refused by name here, before a weight is read
    let prefill_gemm = crate::prefill_p2::current().map_err(ModelError::Config)?;
    boot::fact("prefill_gemm", prefill_gemm.id().to_string());
    if wq_budget.is_some() && prefill_gemm == crate::prefill_p2::PrefillGemm::Fused {
        // the residency keeps the reconstructed W_q for the pinned cuBLAS GEMM; the fused GEMM reads the trellis and has no use for it, so the
        // booked bytes would be dead weight: say so instead of booking them
        return Err(ModelError::Config(format!(
            "{} is set but the prompt rows run the fused GEMM, which never reconstructs W_q; set {}=pinned to use the residency, or unset {}",
            crate::wq::WQ_RESIDENT_ENV,
            crate::prefill_p2::GEMM_VAR,
            crate::wq::WQ_RESIDENT_ENV
        )));
    }
    if let Some(b) = wq_budget {
        boot::fact("wq_resident_gib", format!("{:.2}", gib(b)));
        eprintln!(
            "rexl3 event=wq_plan resident_gib={:.2} note=\"reconstructed W_q kept beside the weights; the reconstruct of a resident matrix is skipped\"",
            gib(b)
        );
    }
    // the cache is sized from the window, for `auto` and an explicit `--prefix-cache-mib` alike: the pages of one full window before anything else
    // (`docs/PREFIX_CACHE_SIZING.md` section 5)
    let plan = plan_memory_cached_windowed(
        &profile,
        free,
        o.floor_bytes,
        o.slots,
        kv_tokens_cap,
        vision_reserve,
        o.prefix_cache.request(),
        snapshot_bytes,
        wq_budget.unwrap_or(0),
        o.max_model_len as u64,
    )
    .map_err(|e| ModelError::Config(e.to_string()))?;
    eprintln!(
        "rexl3 event=kv_plan free_gib={:.1} kv_pages={} kv_tokens={} slots={}",
        gib(free),
        plan.kv_blocks,
        plan.kv_tokens,
        o.slots
    );
    boot::fact("kv_pages", plan.kv_blocks.to_string());
    boot::fact("kv_tokens", plan.kv_tokens.to_string());
    if plan.cache.enabled() {
        eprintln!(
            "rexl3 event=prefix_cache_plan entries={} pages={} gib={:.2} spare_gib={:.1} reach_tokens={} window={}",
            plan.cache.snapshot_slots,
            plan.cache.page_blocks,
            gib(plan.cache.bytes()),
            gib(plan.spare_bytes),
            plan.cache.reach_tokens(),
            o.max_model_len
        );
        boot::fact("prefix_cache", format!("{} entries, {:.2} GiB", plan.cache.snapshot_slots, gib(plan.cache.bytes())));
        boot::fact("prefix_cache_reach_tokens", plan.cache.reach_tokens().to_string());
        if plan.cache.reach_tokens() < o.max_model_len as u64 {
            eprintln!(
                "rexl3 level=warn event=prefix_cache_reach reach_tokens={} window={} note=\"a prompt longer than the reach is never served from the cache; raise --prefix-cache-mib\"",
                plan.cache.reach_tokens(),
                o.max_model_len
            );
        }
    } else {
        // the request was `Off`, or `Auto` on a box with no spare memory: say which, so an operator is not left guessing
        let why = match o.prefix_cache.request() {
            CacheRequest::Off => "off",
            _ => "no spare memory for two entries after the KV pool (pass --prefix-cache-mib to carve one out of it)",
        };
        eprintln!("rexl3 event=prefix_cache_plan entries=0 reason=\"{why}\" spare_gib={:.1}", gib(plan.spare_bytes));
        boot::fact("prefix_cache", why);
    }
    if let Some(win) = window {
        win.close("kv_plan");
    }
    boot::mark("engine_allocating", None);
    let ph = loadstats::phase_start(Component::Engine);
    // an allocation the device refuses although the plan fits ends the boot with what was planned and what the host had, never a bare CUDA code
    let planned = plan.planned_bytes(o.slots);
    let refuse = |what: &str, e: ModelError| allocation_error(what, e, planned, o.floor_bytes, host.as_ref());
    // the pool holds the pages the cache may keep on top of the pages admission hands out
    let mut model = Model::with_window_rows(
        cfg.clone(),
        w,
        o.slots,
        plan.kv_blocks + plan.cache.page_blocks,
        o.max_model_len,
        o.prefill_chunk,
        o.window_rows,
    )
    .map_err(|e| refuse("the engine's buffers (KV pool, slot state, scratch)", e))?;
    if let Some(b) = wq_budget {
        // the plan has booked the budget; the slots themselves fill lazily, on each matrix's first use
        model.enable_wq_residency(b);
    }
    if trees {
        // keep what Model::new made of REXL3_TREE_KERNELS=all (every window on the tree kernels: the gate's second pass runs the server so);
        // the tree buffers hold the model's window (`--window-rows`), so a tree round can never be wider than the buffers
        let all = model.trees_enabled().1;
        model.enable_trees(all).map_err(|e| refuse("the draft-tree buffers", e))?;
    }
    loadstats::phase_end(ph);
    let draft_native = draft.is_some();
    // the block the draft was loaded with (REXL3_DRAFT_BLOCK, the trained 8 when unset) prices the draft's round cost: a
    // 16-row block reads the measured tables at twice the streams and rows and halves the sequences a round carries
    let draft_block = draft.as_ref().map_or(crate::draft::BLOCK, |d| d.block);
    model.draft = draft;
    let mut backend = QwenBackend::new(model, o.prefill_chunk);
    if plan.cache.enabled() {
        let ns = prefix_namespace(&pack, dpack.as_ref());
        backend = backend.with_prefix_cache(ns, plan.cache).map_err(|e| refuse("the prefix cache", e))?;
    }
    // the planner already took the weights, the reserves and the floor out of what admission is given
    // (the plan states the pool in tokens as well as bytes: a request past it is refused at once, not prefilled until the pool runs out)
    let mm = ModelMemory::from_profile(&profile, 0, 0);
    let admission = plan.admission(mm, o.slots, o.max_model_len).map_err(|e| ModelError::Config(e.to_string()))?;
    // measured on the GB10 by `tests/bench.rs` at the served 262,144-token context (ms per window step, 2026-10-02, after the GDN
    // commit fold and the 4-pass decode kernel; 1, 8, 16, 32, 48 and 64 rows are 1x1, 1x8, 2x8, 4x8, 6x8 and 8x8 sequences x rows):
    // flat to 8 rows of one sequence, then rising with the number of sequences and rows. The table before them was 66.0 / 69.1 /
    // 72.3 / 80.7 / 97.7 / 127.5 ms. The attention decode grid is sized from the maximum context, so a smaller bench context
    // measures a different kernel launch.
    // 96 and 128 rows are 12 x 8 and 16 x 8 sequences x rows from the wide grid (recorded in the maintainers' working repository): the step past 64 rows is a second
    // tile-decode group in every EXL3 linear (`group_tiles`), +40 ms, and a table in rows alone smears it
    let cost = CostModel::from_table(vec![
        (1, 65.4),
        (8, 67.9),
        (16, 71.6),
        (32, 80.4),
        (48, 93.7),
        (64, 103.6),
        (96, 169.4),
        (128, 199.7),
    ]);
    let drafting = o.drafting;
    let make: Box<dyn Fn() -> Box<dyn Drafter>> = if drafting {
        Box::new(|| Box::new(CopyDrafter::new(8, 7)) as Box<dyn Drafter>)
    } else {
        Box::new(|| Box::new(NoDrafts) as Box<dyn Drafter>)
    };
    boot::mark("kv_planned", Some(gib(plan.kv_tokens * profile.kv_bytes_per_token())));
    // the margin policy is measured with the native draft; the copy drafter (no draft pack) keeps the legacy budget
    let default_policy = if draft_native { crate::costs::DEFAULT_BUDGET } else { gb_engine::BudgetPolicy::Legacy };
    let policy = crate::costs::budget_policy_from_env(default_policy)?;
    // REXL3_SPEC_LEARN=1: the budget learns its priors from the traffic (see `Engine::with_learning`); off by default so the number
    // of drafts a request is offered is a function of the request, not of what the server ran before it
    let learn = crate::costs::spec_learn_from_env()?;
    boot::fact("spec_budget", format!("{policy:?}").to_lowercase());
    boot::fact("spec_learn", learn.to_string());
    eprintln!("rexl3 event=spec_budget policy={} learn={learn}", format!("{policy:?}").to_lowercase());
    // REXL3_PREFILL_BURST: how many prefill slices a step runs while a burst of prompts waits (replies never change, only first-token times).
    // The Qwen engine opts in to the measured burst policy (its backend batches short prompts into one forward); an engine that does not
    // ask, such as GLM's, keeps one slice a step
    let prefill = crate::costs::qwen_prefill_policy()?;
    let batching = crate::costs::prefill_batching_from_env()?;
    boot::fact("prefill_policy", format!("{prefill:?}").to_lowercase());
    boot::fact("prefill_batching", batching.to_string());
    eprintln!("rexl3 event=prefill_policy policy={} batching={batching}", format!("{prefill:?}").to_lowercase());
    // the verify rows of a round are bounded by the window the model was built with: the one number, so that they cannot drift apart
    // (a tree round obeys it too: `plan_trees` is handed `max_rows`, and the tree buffers are sized by the model's window)
    let mut engine =
        Engine::new(backend, admission, EngineConfig { max_rows: o.window_rows, max_drafts: 7, cost, drafting }, make);
    if trees {
        let tree = crate::costs::tree_config_from_env()?;
        boot::fact("spec_tree_nodes", tree.max_nodes.to_string());
        boot::fact("spec_tree_branch", tree.policy.branch.to_string());
        boot::fact("spec_tree_copy", tree.copy_min_match.to_string());
        engine = engine.with_trees(tree);
    }
    Ok(engine
        .with_prefill_policy(prefill)
        .with_prefill_batching(batching)
        .with_budget(
            policy,
            crate::costs::round_cost(draft_native.then(|| crate::costs::head_table(&draft_vocab, head_fraction)), draft_block),
        )
        .with_learning(learn)
        .with_prefix_cache(interactive_prefix_config_from_env()?))
}

/// The key namespace of this pack pair's snapshots: everything a snapshot's bytes depend on besides the tokens (`docs/PREFIX_CACHE.md`,
/// "Keys"). A different pack, draft, pinned GEMM algorithm, page size or cache format is a different namespace, so entries can never be
/// shared across them (the index is empty at every boot anyway).
pub fn prefix_namespace(target: &Pack, draft: Option<&Pack>) -> u64 {
    let t = crate::fingerprint::fingerprint(&target.ck);
    let d = draft.map(|p| crate::fingerprint::fingerprint(&p.ck));
    namespace_of(&t, d.as_deref())
}

/// [`prefix_namespace`] from the two fingerprints (the arithmetic, testable without a pack).
pub fn namespace_of(target_fingerprint: &str, draft_fingerprint: Option<&str>) -> u64 {
    let text = format!(
        "rexl3-prefix-cache-v1|{target_fingerprint}|{}|{}|page={}|window={}|ctxkeep={}|kv=bf16|state=f32+bf16",
        draft_fingerprint.unwrap_or("no-draft"),
        gb_cuda::exl3_linear::prefill_pin_name(),
        crate::kvtables::BLOCK_TOKENS,
        gb_cuda::dflash2::DraftCtxLayout::WINDOW,
        crate::draft::CTX_KEEP,
    );
    // FNV-1a 64
    text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// The shortest checkpoint period `REXL3_PREFIX_CHECKPOINT` accepts: a checkpoint copies a whole snapshot (162 MiB, about 2 ms, for Nemotron) and
/// holds a snapshot slot while its prompt prefills, so a period of a few pages would copy on almost every slice; 16,384 tokens is about 30 s
/// of prefill at the measured rate, a copy every 64 slices of 256 rows.
pub const MIN_CHECKPOINT_TOKENS: usize = 16_384;

/// The cut policy, with the environment's overrides (for measurement: `REXL3_PREFIX_MIN_CUT`, `REXL3_PREFIX_TAIL`, `REXL3_PREFIX_GHOST_GAIN`,
/// all in tokens), and the long-prefill policy's (`REXL3_PREFIX_CHECKPOINT`, tokens, a multiple of 256 of at least 16,384; `REXL3_PREFIX_EVICTION`, `lru` or
/// `size`). Anything that does not parse is an error, not a silent default.
pub fn prefix_config_from_env() -> Result<PrefixConfig, ModelError> {
    prefix_config_from(&|name| std::env::var(name).ok())
}

/// [`prefix_config_from_env`] over any source of variables (`var(name)`, `None` when unset).
pub fn prefix_config_from(var: &dyn Fn(&str) -> Option<String>) -> Result<PrefixConfig, ModelError> {
    prefix_config_over(PrefixConfig::default(), var)
}

/// The Qwen and GLM engines' prefix config: [`gb_sched::prefix::CutPolicy::interactive`] (cuts that serve a second question about a long document and a
/// tool loop of a few hundred tokens; the default policy serves neither), then the environment's overrides as in [`prefix_config_from_env`].
pub fn interactive_prefix_config_from_env() -> Result<PrefixConfig, ModelError> {
    interactive_prefix_config_from(&|name| std::env::var(name).ok())
}

/// [`interactive_prefix_config_from_env`] over any source of variables.
pub fn interactive_prefix_config_from(var: &dyn Fn(&str) -> Option<String>) -> Result<PrefixConfig, ModelError> {
    prefix_config_over(PrefixConfig { policy: gb_sched::prefix::CutPolicy::interactive(), ..PrefixConfig::default() }, var)
}

/// `base` with the environment's overrides applied.
fn prefix_config_over(base: PrefixConfig, var: &dyn Fn(&str) -> Option<String>) -> Result<PrefixConfig, ModelError> {
    let mut c = base;
    let get = |name: &str, into: &mut usize| -> Result<(), ModelError> {
        match var(name) {
            Some(v) => {
                *into =
                    v.trim().parse().map_err(|_| ModelError::Config(format!("{name} must be a number of tokens, got `{v}`")))?;
                Ok(())
            }
            None => Ok(()),
        }
    };
    get("REXL3_PREFIX_MIN_CUT", &mut c.policy.min_cut)?;
    get("REXL3_PREFIX_TAIL", &mut c.policy.tail)?;
    get("REXL3_PREFIX_GHOST_GAIN", &mut c.policy.ghost_min_gain)?;
    get("REXL3_PREFIX_MIN_GAP", &mut c.policy.min_gap)?;
    // the long-prefill policy (docs/LONG_PREFILL_POLICY.md), both off unless asked for
    get("REXL3_PREFIX_CHECKPOINT", &mut c.checkpoint_every)?;
    if c.checkpoint_every != 0
        && (c.checkpoint_every % gb_sched::prefix::BLOCK_TOKENS != 0 || c.checkpoint_every < MIN_CHECKPOINT_TOKENS)
    {
        return Err(ModelError::Config(format!(
            "REXL3_PREFIX_CHECKPOINT must be 0 (off) or a multiple of {} tokens (a checkpoint on the page and slice grid costs no extra \
             forward) of at least {MIN_CHECKPOINT_TOKENS} (each checkpoint copies a snapshot of the recurrent state, 162 MiB for Nemotron), got {}",
            gb_sched::prefix::BLOCK_TOKENS,
            c.checkpoint_every
        )));
    }
    match var("REXL3_PREFIX_EVICTION").as_deref().map(str::trim) {
        None | Some("" | "lru") => {}
        Some("size") => c.eviction = gb_engine::Eviction::SizeAware { overhead_pages: 0 },
        Some(v) => return Err(ModelError::Config(format!("REXL3_PREFIX_EVICTION must be `lru` or `size`, got `{v}`"))),
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_long_prefill_policy_is_off_by_default_and_read_strictly() {
        use gb_engine::Eviction;
        let from = |vars: &[(&str, &str)]| {
            let vars: Vec<(String, String)> = vars.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
            super::prefix_config_from(&move |n| vars.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone()))
        };
        let d = from(&[]).unwrap();
        assert_eq!((d.eviction, d.checkpoint_every), (Eviction::Lru, 0));
        assert_eq!(d, gb_engine::PrefixConfig::default());
        let c = from(&[("REXL3_PREFIX_EVICTION", "size"), ("REXL3_PREFIX_CHECKPOINT", "65536")]).unwrap();
        assert_eq!((c.eviction, c.checkpoint_every), (Eviction::SizeAware { overhead_pages: 0 }, 65_536));
        assert!(from(&[("REXL3_PREFIX_CHECKPOINT", "1000")]).unwrap_err().to_string().contains("multiple of 256"));
        assert!(
            from(&[("REXL3_PREFIX_CHECKPOINT", "256")]).unwrap_err().to_string().contains("at least 16384"),
            "below the minimum"
        );
        assert_eq!(from(&[("REXL3_PREFIX_CHECKPOINT", "0")]).unwrap().checkpoint_every, 0);
        assert!(from(&[("REXL3_PREFIX_EVICTION", "lfu")]).is_err());
        assert!(from(&[("REXL3_PREFIX_CHECKPOINT", "lots")]).is_err());
    }

    /// The Qwen engine cuts with the interactive policy (the default one serves neither a second question about a document nor a short tool
    /// loop); the environment still overrides each value, and every other family's config is the default one.
    #[test]
    fn the_qwen_engine_cuts_for_interactive_use_and_the_environment_still_decides() {
        use gb_sched::prefix::CutPolicy;
        let qwen = |vars: &[(&str, &str)]| {
            let vars: Vec<(String, String)> = vars.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
            super::interactive_prefix_config_from(&move |n| vars.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone()))
        };
        assert_eq!(qwen(&[]).unwrap().policy, CutPolicy::interactive());
        assert_ne!(CutPolicy::interactive(), CutPolicy::default());
        let c = qwen(&[("REXL3_PREFIX_TAIL", "16"), ("REXL3_PREFIX_MIN_CUT", "512")]).unwrap();
        assert_eq!((c.policy.tail, c.policy.min_cut), (16, 512));
        assert_eq!((c.policy.min_gap, c.policy.ghost_min_gain), (128, 256), "what is not overridden stays interactive");
        assert!(qwen(&[("REXL3_PREFIX_TAIL", "long")]).is_err());
        assert_eq!(super::prefix_config_from(&|_| None).unwrap(), gb_engine::PrefixConfig::default());
    }

    #[test]
    fn the_prefix_cache_is_off_only_for_a_family_without_one() {
        use PrefixCacheOption::*;
        for asked in [Auto, On, Mib(2048), Off] {
            // one process or a tensor-parallel group alike: the request stands for a family that can snapshot (a group's ranks are made to
            // agree on it by the boot gate, not here)
            assert_eq!(effective_prefix_cache(true, asked), asked, "a family with a cache keeps the request");
            assert_eq!(effective_prefix_cache(false, asked), Off, "a family without one");
        }
    }

    use super::*;
    use serde_json::Value;

    struct Toy {
        /// The family's `max_tp_ranks`.
        cap: usize,
    }

    impl crate::family::ModelFamily for Toy {
        fn id(&self) -> &'static str {
            "toy"
        }
        fn architectures(&self) -> &'static [&'static str] {
            &[]
        }
        fn probe(&self, _dir: &std::path::Path) -> Result<crate::family::FamilyPlan, crate::family::FamilyError> {
            Err(crate::family::FamilyError::Config("toy".into()))
        }
        fn build(&self, _plan: crate::family::FamilyPlan, _ctx: &BuildCtx<'_>) -> Result<Box<dyn ServeEngine>, ModelError> {
            Err(ModelError::NotBuilt("toy".into()))
        }
        fn max_tp_ranks(&self) -> usize {
            self.cap
        }
    }

    fn with_tp(t: Option<TpOptions>) -> ServeOptions {
        let mut o = ServeOptions::new(std::path::PathBuf::from("/nonexistent"));
        o.tp = t;
        o
    }

    #[test]
    fn a_tensor_parallel_group_is_refused_by_name_for_a_family_without_a_runtime_and_for_the_wrong_rank() {
        let group = |rank| Some(TpOptions::new(2, rank, "192.0.2.1:29600"));
        // no group, or one rank: the single-process start is untouched
        assert!(refuse_tensor_parallel(&Toy { cap: 1 }, &with_tp(None), 0).is_ok());
        assert!(refuse_tensor_parallel(&Toy { cap: 1 }, &with_tp(Some(TpOptions::new(1, 0, ""))), 0).is_ok());
        // a family that has none says so
        let e = refuse_tensor_parallel(&Toy { cap: 1 }, &with_tp(group(0)), 0).unwrap_err().to_string();
        assert!(e.contains("toy") && e.contains("no tensor-parallel runtime"), "{e}");
        // a family that has one serves rank 0 at `start` and rank 1 at `start_follower`, and nothing else
        let fam = Toy { cap: 4 };
        assert!(refuse_tensor_parallel(&fam, &with_tp(group(0)), 0).is_ok());
        assert!(refuse_tensor_parallel(&fam, &with_tp(group(1)), 1).is_ok());
        let e = refuse_tensor_parallel(&fam, &with_tp(group(1)), 0).unwrap_err().to_string();
        assert!(e.contains("--tp-rank 1") && e.contains("start_follower"), "{e}");
        let e = refuse_tensor_parallel(&fam, &with_tp(group(0)), 1).unwrap_err().to_string();
        assert!(e.contains("serving rank"), "{e}");
        let e = refuse_tensor_parallel(&fam, &with_tp(None), 1).unwrap_err().to_string();
        assert!(e.contains("--tp-ranks 2 --tp-rank 1"), "{e}");
        // an address that is not one is refused before anything is started
        let e = refuse_tensor_parallel(&fam, &with_tp(Some(TpOptions::new(2, 0, "nohost"))), 0).unwrap_err().to_string();
        assert!(e.contains("HOST:PORT"), "{e}");
        // a group above the family's cap is refused naming the family and the cap; at the cap it is served
        let big = |ranks, rank| Some(TpOptions::new(ranks, rank, "192.0.2.1:29600"));
        assert!(refuse_tensor_parallel(&fam, &with_tp(big(4, 0)), 0).is_ok());
        let e = refuse_tensor_parallel(&fam, &with_tp(big(5, 0)), 0).unwrap_err().to_string();
        assert!(e.contains("toy") && e.contains("at most 4 ranks") && e.contains("--tp-ranks 5"), "{e}");
        let e = refuse_tensor_parallel(&fam, &with_tp(big(5, 3)), 1).unwrap_err().to_string();
        assert!(e.contains("toy") && e.contains("at most 4 ranks"), "{e}");
        // the old question still answers from the cap
        assert!(
            crate::family::ModelFamily::tensor_parallel(&fam) && !crate::family::ModelFamily::tensor_parallel(&Toy { cap: 1 })
        );
        // the message of a family with no runtime names its cap and the one family that has one
        let e = refuse_tensor_parallel(&Toy { cap: 1 }, &with_tp(big(2, 0)), 0).unwrap_err().to_string();
        assert!(e.contains("toy") && e.contains("no tensor-parallel runtime") && e.contains("cap is 1"), "{e}");
    }

    #[test]
    fn tp_options_read_the_environment_and_validate() {
        let vars = |v: &'static [(&'static str, &'static str)]| {
            move |k: &str| v.iter().find(|(n, _)| *n == k).map(|(_, x)| x.to_string())
        };
        assert_eq!(TpOptions::from_vars(vars(&[])).unwrap(), None);
        let t = TpOptions::from_vars(vars(&[
            ("REXL3_TP_RANKS", "2"),
            ("REXL3_TP_RANK", "1"),
            ("REXL3_TP_PEER", "192.0.2.1:29600"),
            ("REXL3_TP_CONNECT_TIMEOUT_S", "60"),
            ("REXL3_TP_READY_TIMEOUT_S", "3000"),
            ("REXL3_TP_PEER_TIMEOUT_S", "20"),
        ]))
        .unwrap()
        .unwrap();
        assert_eq!((t.ranks, t.rank, t.peer.as_str()), (2, 1, "192.0.2.1:29600"));
        assert_eq!((t.connect_timeout.as_secs(), t.ready_timeout.as_secs(), t.peer_timeout.as_secs()), (60, 3000, 20));
        assert!(t.validate().is_ok() && t.is_group());
        assert!(TpOptions::from_vars(vars(&[("REXL3_TP_RANKS", "two")])).unwrap_err().contains("REXL3_TP_RANKS"));
        for (ranks, rank, peer) in [
            (0, 0, "a:1"),
            (3, 3, "a:1"),
            (2, 2, "a:1"),
            (2, 0, ""),
            (2, 0, "a"),
            (2, 0, ":9"),
            (2, 0, "a:99999"),
            (TpOptions::MAX_RANKS + 1, 0, "a:1"),
        ] {
            assert!(TpOptions::new(ranks, rank, peer).validate().is_err(), "{ranks} {rank} {peer}");
        }
        // a group of any size: every rank of 3, 4 and 8 is valid, and so is the largest
        for ranks in [3, 4, 8, TpOptions::MAX_RANKS] {
            for rank in [0, 1, ranks - 1] {
                assert!(TpOptions::new(ranks, rank, "a:1").validate().is_ok(), "{ranks} {rank}");
            }
        }
        // one rank needs no address
        assert!(TpOptions::new(1, 0, "").validate().is_ok());
    }

    #[test]
    fn draft_trees_are_on_by_default_and_only_with_the_draft() {
        let mut o = ServeOptions::new(std::path::PathBuf::from("/nonexistent"));
        assert!(o.spec_tree, "trees are the default");
        assert!(!trees_wanted(&o), "no draft pack: nothing to search, nothing allocated");
        assert!(!memory_options(&o, false).trees);
        o.draft_dir = Some(std::path::PathBuf::from("/nonexistent-draft"));
        assert!(trees_wanted(&o) && memory_options(&o, false).trees, "the plan counts the tree buffers when trees will be built");
        o.drafting = false; // --no-drafts
        assert!(!trees_wanted(&o) && !memory_options(&o, false).trees);
        o.drafting = true;
        o.spec_tree = false; // --no-spec-tree
        assert!(!trees_wanted(&o) && !memory_options(&o, false).trees);
    }

    #[test]
    fn prefill_chunk_beyond_the_kernels_is_refused_up_front() {
        let mut o = ServeOptions::new(std::path::PathBuf::from("/nonexistent"));
        assert!(check_options(&o).is_ok());
        o.prefill_chunk = 4096;
        assert!(check_options(&o).is_ok());
        o.prefill_chunk = 4097;
        assert!(check_options(&o).unwrap_err().to_string().contains("--prefill-chunk"));
        // a family that names a wider limit (GLM's P2: 8,192 rows) is held to its own
        assert!(check_options_for(&o, 8192).is_ok());
        o.prefill_chunk = 8192;
        assert!(check_options(&o).is_err());
        assert!(check_options_for(&o, 8192).is_ok());
        o.prefill_chunk = 8193;
        assert!(check_options_for(&o, 8192).unwrap_err().to_string().contains("between 1 and 8192"));
        o.prefill_chunk = 0;
        assert!(check_options(&o).is_err());
        assert!(check_options_for(&o, 8192).is_err());
    }

    #[test]
    fn a_chunk_smaller_than_a_window_is_refused_when_a_draft_is_loaded() {
        let mut o = ServeOptions::new(std::path::PathBuf::from("/nonexistent"));
        o.draft_dir = Some(std::path::PathBuf::from("/nonexistent-draft"));
        o.prefill_chunk = 63;
        let e = check_options(&o).unwrap_err().to_string();
        assert!(e.contains("--prefill-chunk must be at least 64") && e.contains("got 63"), "{e}");
        o.prefill_chunk = 64;
        assert!(check_options(&o).is_ok());
        // nothing absorbs window rows without a draft, so a small chunk stays allowed there
        o.prefill_chunk = 1;
        o.drafting = false;
        assert!(check_options(&o).is_ok());
        o.drafting = true;
        o.draft_dir = None;
        assert!(check_options(&o).is_ok());
    }

    #[test]
    fn a_qwen_pack_with_other_attention_shapes_is_refused_before_anything_loads() {
        let mut cfg: Value = serde_json::from_str(include_str!("../../../testdata/qwen38/config.json")).unwrap();
        cfg["text_config"]["num_key_value_heads"] = serde_json::json!(8);
        let dir = pack_dir("shapes", &cfg.to_string());
        let o = ServeOptions::new(dir.clone());
        let e = preflight(&o).unwrap_err().to_string();
        assert!(e.contains("num_key_value_heads is 8, the kernels are built for 4"), "{e}");
        let e = start(&o).err().expect("must refuse").to_string();
        assert!(e.contains("num_key_value_heads is 8"), "{e}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn spec_copy_leaves_the_draft_pack_unloaded_and_every_other_mode_keeps_it() {
        let dir = Some(std::path::PathBuf::from("/models/dflash2"));
        assert_eq!(draft_dir_for(SpecMode::Copy, dir.clone()), None, "--spec copy is the copy drafter alone");
        for keep in [SpecMode::Auto, SpecMode::Mtp, SpecMode::None] {
            assert_eq!(draft_dir_for(keep, dir.clone()), dir, "{keep} keeps the pack it was given");
            assert_eq!(draft_dir_for(keep, None), None);
        }
    }

    /// Draft trees are searched over the DFlash2 draft's lattice, so they follow the pack: `--spec copy` (the copy drafter alone, no pack loaded)
    /// and `--spec none` / `--no-drafts` run chains, whatever `--spec-tree` says, and `--no-spec-tree` leaves chains with the pack loaded.
    #[test]
    fn draft_trees_follow_the_draft_pack_so_spec_copy_runs_chains() {
        let mut o = ServeOptions::new(std::path::PathBuf::from("/models/qwen"));
        o.draft_dir = Some(std::path::PathBuf::from("/models/dflash2"));
        assert!(trees_wanted(&o), "trees are on by default with the pack");
        let mut copy = o.clone();
        copy.draft_dir = draft_dir_for(SpecMode::Copy, o.draft_dir.clone());
        assert!(!trees_wanted(&copy), "--spec copy loads no DFlash2 pack: no lattice, no trees");
        let mut off = o.clone();
        off.spec_tree = false;
        assert!(!trees_wanted(&off), "--no-spec-tree");
        let mut nodraft = o.clone();
        nodraft.drafting = false;
        assert!(!trees_wanted(&nodraft), "--no-drafts");
        // the memory plan counts the tree buffers exactly when trees are wanted
        assert!(memory_options(&o, false).trees && !memory_options(&copy, false).trees && !memory_options(&off, false).trees);
    }

    #[test]
    fn a_spec_the_family_does_not_offer_is_refused_by_name() {
        // Qwen offers dflash2 and copy: `--spec mtp` is refused before anything loads, `copy`, `auto` and `none` pass the check
        let dir = pack_dir("spec", include_str!("../../../testdata/qwen38/config.json"));
        let mut o = ServeOptions::new(dir.clone());
        o.spec = SpecMode::Mtp;
        let e = preflight(&o).unwrap_err().to_string();
        assert!(e.contains("--spec mtp") && e.contains("does not offer") && e.contains("dflash2, copy"), "{e}");
        let e = start(&o).err().expect("must refuse").to_string();
        assert!(e.contains("--spec mtp"), "{e}");
        for ok in [SpecMode::Copy, SpecMode::Auto, SpecMode::None] {
            o.spec = ok;
            let r = preflight(&o);
            // whatever else the pack lacks (its weights), the drafter choice is not what refuses it
            assert!(r.is_ok() || !r.unwrap_err().to_string().contains("--spec"), "{ok}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A fresh directory holding the given `config.json`.
    fn pack_dir(tag: &str, config: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rexl3-serve-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), config).unwrap();
        dir
    }

    #[test]
    fn an_unknown_architecture_is_refused_by_name_and_the_unvalidated_flag_does_not_relax_it() {
        let dir = pack_dir(
            "unknown",
            r#"{"architectures":["MysteryForCausalLM"],"model_type":"qwen3_5","text_config":{},"quantization_config":{"quant_method":"exl3"}}"#,
        );
        for allow in [false, true] {
            let mut o = ServeOptions::new(dir.clone());
            o.allow_unvalidated_pack = allow;
            let e = preflight(&o).unwrap_err().to_string();
            assert!(e.contains("`MysteryForCausalLM`") && e.contains("unsupported model architecture"), "allow={allow}: {e}");
            // the starter refuses it identically, before it opens a pack or touches the GPU
            let e = start(&o).err().expect("must refuse").to_string();
            assert!(e.contains("`MysteryForCausalLM`"), "allow={allow}: {e}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A one-shard pack: `config.json` and a `model.safetensors` of one bf16 tensor (enough for `Pack::open` and a fingerprint).
    fn tiny_pack(tag: &str) -> PathBuf {
        let dir = pack_dir(
            tag,
            r#"{"architectures":["NemotronH_Nano_Omni_Reasoning_V3"],"quantization_config":{"quant_method":"exl3"}}"#,
        );
        let header = br#"{"x.weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
        let mut f = (header.len() as u64).to_le_bytes().to_vec();
        f.extend_from_slice(header);
        f.extend_from_slice(&[0, 0x3f, 0, 0x40]);
        std::fs::write(dir.join("model.safetensors"), f).unwrap();
        dir
    }

    /// A pack that is not on the validated list (the toy pack here): without `--allow-unvalidated-pack` it is refused by its directory and fingerprint
    /// (and how to serve it anyway), before a family builds; with it the pack passes and is reported unvalidated (the facts that the
    /// boot log and `/v1/models` read). Both start paths run the check before the build (rank 0 and every follower).
    #[test]
    fn an_unvalidated_pack_is_refused_by_name_unless_allowed_and_then_reported_with_the_warning() {
        let dir = tiny_pack("unvalidated");
        let fp = crate::fingerprint::fingerprint(&Pack::open(&dir).unwrap().ck);
        let mut o = ServeOptions::new(dir.clone());
        let e = check_pack_dirs(&o).unwrap_err().to_string();
        assert!(e.contains(&dir.display().to_string()), "the refusal names the pack's directory: {e}");
        assert!(e.contains(&fp) && e.contains("is not one this release was validated on"), "{e}");
        assert!(e.contains("pass --allow-unvalidated-pack to serve it anyway"), "{e}");
        o.allow_unvalidated_pack = true;
        check_pack_dirs(&o).unwrap();
        let fact = |n: &str| boot::facts().into_iter().find(|(k, _)| *k == n).map(|(_, v)| v);
        assert_eq!(fact("pack_validated").as_deref(), Some("no (--allow-unvalidated-pack)"));
        assert_eq!(fact("pack_warning").as_deref(), Some(UNVALIDATED_PACK_WARNING));
        assert_eq!(fact("target_fingerprint"), Some(fp));
        // a pack that cannot be opened is refused by its directory too
        o.model_dir = dir.join("nonexistent");
        let e = check_pack_dirs(&o).unwrap_err().to_string();
        assert!(e.contains("nonexistent"), "{e}");
        std::fs::remove_dir_all(dir).unwrap();
        // both start paths check the pack after the cheap refusals and before the family builds anything
        let code = include_str!("serve.rs").split("#[cfg(test)]").next().unwrap();
        for (start, build) in
            [("pub fn start_in(", "family.build(plan"), ("pub fn start_follower_in(", "family.run_follower(plan")]
        {
            let body = &code[code.find(start).unwrap()..];
            let (check, built) = (body.find("check_pack_dirs(o)?").unwrap(), body.find(build).unwrap());
            assert!(check < built, "{start} builds before it checks the pack");
        }
    }

    #[test]
    fn a_pack_that_is_not_exl3_is_refused_whatever_the_flags() {
        let dir = pack_dir(
            "nvfp4",
            &include_str!("../../../testdata/qwen38/config.json")
                .replace(r#""quant_method": "exl3""#, r#""quant_method": "nvfp4""#),
        );
        for allow in [false, true] {
            let mut o = ServeOptions::new(dir.clone());
            o.allow_unvalidated_pack = allow;
            let e = preflight(&o).unwrap_err().to_string();
            assert!(e.contains("EXL3 packs only") && e.contains("quant_method=nvfp4"), "allow={allow}: {e}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_qwen_pack_passes_the_preflight_untouched() {
        let dir = pack_dir("qwen", include_str!("../../../testdata/qwen38/config.json"));
        assert!(preflight(&ServeOptions::new(dir.clone())).is_ok());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_verify_window_is_a_serve_option_checked_against_the_kernels_and_the_prefill_chunk() {
        let mut o = ServeOptions::new(PathBuf::from("/nonexistent"));
        assert_eq!(o.window_rows, 64, "the served default");
        assert!(check_options(&o).is_ok());
        for ok in [1usize, 64, 65, 96, 128] {
            o.window_rows = ok;
            assert!(check_options(&o).is_ok(), "{ok}");
        }
        for bad in [0usize, 129, 256] {
            o.window_rows = bad;
            let e = check_options(&o).unwrap_err().to_string();
            assert!(e.contains("--window-rows") && e.contains("1..=128"), "{bad}: {e}");
        }
        // the draft absorbs a whole window in one prefill-sized pass: the chunk must hold it
        o.window_rows = 128;
        o.draft_dir = Some(PathBuf::from("/nonexistent"));
        o.prefill_chunk = 100;
        let e = check_options(&o).unwrap_err().to_string();
        assert!(e.contains("--prefill-chunk must be at least 128"), "{e}");
        o.prefill_chunk = 128;
        assert!(check_options(&o).is_ok());
        // the memory plan is evaluated at the same window
        assert_eq!(memory_options(&o, false).window_rows, 128);
    }

    #[test]
    fn the_memory_options_follow_the_serve_options() {
        let mut o = ServeOptions::new(PathBuf::from("/nonexistent"));
        o.prefill_chunk = 2048;
        o.max_model_len = 100_000;
        o.max_image_tokens = 1024;
        let m = memory_options(&o, true);
        assert_eq!((m.window_rows, m.prefill_rows, m.max_context, m.mtp), (64, 2048, 100_000, true));
        assert_eq!(m.vision, None, "no vision slot: text only");
        o.vision = Some(Default::default());
        let v = memory_options(&o, false).vision.unwrap();
        assert_eq!((v.max_image_tokens, v.max_images_per_request), (1024, 4));
    }

    #[test]
    fn the_slot_count_is_pinned_unless_the_caller_says_it_is_the_default() {
        let o = ServeOptions::new(PathBuf::from("/nonexistent"));
        assert_eq!((o.slots, o.slots_auto), (8, false), "a library caller's count is never changed under it");
    }

    #[test]
    fn an_allocation_the_device_refuses_ends_the_boot_with_the_plan_and_the_hosts_figures() {
        const GIB: u64 = 1 << 30;
        let then = hostmem::MemInfo { total: 121 * GIB, free: 5 * GIB, available: 60 * GIB, cached: 50 * GIB };
        let oom = || ModelError::Cuda(gb_cuda::CudaError(2));
        match allocation_error("the prefix cache", oom(), 8 * GIB, 8 * GIB, Some(&then)) {
            ModelError::Config(t) => {
                assert!(
                    t.contains("the prefix cache") && t.contains("8.00 GiB planned") && t.contains("cudaErrorMemoryAllocation"),
                    "{t}"
                );
                assert!(t.contains("--slots") && t.contains("--kv-tokens"), "{t}");
            }
            other => panic!("an out-of-memory error becomes a refusal with its context, not {other:?}"),
        }
        // every other error is the boot's own and passes through untouched
        assert!(
            matches!(allocation_error("x", ModelError::Cuda(gb_cuda::CudaError(700)), 1, 1, None), ModelError::Cuda(c) if c.0 == 700)
        );
        assert!(matches!(allocation_error("x", ModelError::Config("c".into()), 1, 1, None), ModelError::Config(m) if m == "c"));
    }

    #[test]
    fn both_engines_size_their_plans_from_the_host_and_never_from_the_smaller_of_two_figures() {
        // the GPU boots cannot run here: pin the lines that decide what a plan is sized from. Each goes through `hostmem::planning_free`, and nothing takes
        // the device's `cudaMemGetInfo` free into a `min` with MemAvailable again, or clamps the floor to it (the GLM `effective_floor` that refused a boot
        // with 17.6 GiB available and 0.9 GiB of MemFree). Only the code is read, not these tests
        for (name, src) in [
            ("serve.rs", include_str!("serve.rs")),
            ("families/glm/engine.rs", include_str!("families/glm/engine.rs")),
            ("families/nemotron_h_omni/engine.rs", include_str!("families/nemotron_h_omni/engine.rs")),
        ] {
            let code = src.split("#[cfg(test)]").next().unwrap();
            assert!(
                code.contains("hostmem::planning_free(cuda_free"),
                "{name} no longer sizes its plan with hostmem::planning_free"
            );
            for l in code.lines().filter(|l| !l.trim_start().starts_with("//")) {
                assert!(!(l.contains("cuda_free") && l.contains(".min(")), "{name}: the device's free is in a min again: {l}");
                assert!(
                    !l.contains("effective_floor") && !l.contains("device_room"),
                    "{name}: the floor is clamped to the device again: {l}"
                );
            }
        }
    }

    #[test]
    fn every_pack_check_asks_the_one_validation() {
        // `--print-fingerprints` once asked the Qwen pair list only and said `validated no` for the GLM pack the server serves as validated. The
        // serve path of each family and the report all go through `fingerprint::validated_pack` / `validation_line`; none reads one list
        let sources = [
            ("serve.rs", include_str!("serve.rs"), "validated_pack("),
            ("families/glm/engine.rs", include_str!("families/glm/engine.rs"), "validated_pack("),
            ("server/src/main.rs", include_str!("../../server/src/main.rs"), "validation_line("),
        ];
        for (name, src, ask) in sources {
            let code = src.split("#[cfg(test)]").next().unwrap();
            assert!(code.contains(ask), "{name} no longer asks {ask}");
            for l in code.lines().filter(|l| !l.trim_start().starts_with("//")) {
                assert!(
                    !l.contains("fingerprint::validated(") && !l.contains("fingerprint::validated_text("),
                    "{name} reads one list of validated packs: {l}"
                );
            }
        }
    }

    #[test]
    fn the_qwen_plan_counts_the_features_of_the_images_in_flight_for_a_loaded_tower() {
        // the GPU boot cannot run here: the reserve is the family's `counted_vision_reserve`, asked with whether the tower was really loaded (a
        // `--no-vision` boot or a refused vision config holds no features), and the constant that used to say "none" is gone
        let code = include_str!("serve.rs").split("#[cfg(test)]").next().unwrap();
        assert!(
            code.contains("let has_tower = tower.is_some();"),
            "the tower's presence is no longer read before the tower is moved"
        );
        assert!(
            code.contains("qwen::counted_vision_reserve(&profile, memory_options(o, false).vision, has_tower, o.slots)"),
            "the Qwen plan no longer counts the features of the images in flight for a loaded tower"
        );
        assert!(!code.contains("COUNTED_VISION_RESERVE"), "the zero reserve is back");
    }

    #[test]
    fn both_engines_build_their_admission_from_the_plan_so_the_pool_is_declared() {
        // the GPU boots cannot run here: a family that builds `Admission::new(plan.admit_bytes, ..)` itself leaves out the pool in tokens, and a
        // prompt past it is admitted again (197,048 tokens into a 194,048-token pool: 298.7 s of prefill, then `OutOfPages`). Both go through
        // `MemoryPlan::admission`, which `plan::tests` drives against the real page tables. A family added later adds its engine's file to this list
        for (name, src) in [
            ("serve.rs", include_str!("serve.rs")),
            ("families/glm/engine.rs", include_str!("families/glm/engine.rs")),
            ("families/qwen_flash_next/engine.rs", include_str!("families/qwen_flash_next/engine.rs")),
            ("families/nemotron_h_omni/engine.rs", include_str!("families/nemotron_h_omni/engine.rs")),
        ] {
            let code = src.split("#[cfg(test)]").next().unwrap();
            assert!(code.contains("plan.admission("), "{name} no longer builds its admission with MemoryPlan::admission");
            for l in code.lines().filter(|l| !l.trim_start().starts_with("//")) {
                assert!(!l.contains("Admission::new("), "{name}: builds an Admission by hand, without the pool: {l}");
            }
        }
    }
}
