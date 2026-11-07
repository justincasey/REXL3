//! `rexl3-serve`: the OpenAI-compatible server (`gb-serve` is kept as an alias for one release).
//!
//!     rexl3-serve --model DIR [--draft DIR] [--port 8889] [--host 127.0.0.1] [--max-model-len 262144]
//!                 [--served-model-name NAME] [--backend mock|cuda] [--slots 8] [--spec auto|mtp|copy|none] [--no-drafts] [--allow-unvalidated-pack]
//!                 [--chat-format ID] [--verify-pack] [--no-spec-tree] [--queue-depth 64] [--queue-timeout 300] [--shutdown-grace 8]
//!                 [--header-read-timeout 30]
//!                 [--prefix-cache on|off] [--prefix-cache-mib N]
//!                 [--tp-ranks N --tp-rank 0..N-1 --tp-peer HOST:PORT]
//!     rexl3-serve --version
//!     rexl3-serve --print-fingerprints --model DIR [--draft DIR]
//!     rexl3-serve --print-crc32 --model DIR      (a fresh crc32.txt for the pack, shards included)
//!
//! The prefix cache is ON by default (`--prefix-cache off` or `--prefix-cache-mib 0` turn it off). It keeps the state of prompts already prefilled and resumes from the longest stored prefix of a new request (multi-turn
//! chat and shared system prompts stop re-prefilling what they share; replies are byte-identical to the uncached path, `docs/PREFIX_CACHE.md`);
//! `off` disables it, and `on` requires it (a family may not switch it off for lack of spare memory, which the default may do).
//! `--prefix-cache-mib N` sizes it: N MiB are carved out before the KV pool is sized, `0` disables it, and without it the
//! cache takes what the KV plan leaves spare (at most half of it and 8 GiB, never any of the KV pool's share). A request opts out with
//! `"prefix_cache": false` in its body. Families without a prefix cache ignore both flags.
//!
//! `--tp-ranks N --tp-rank R --tp-peer HOST:PORT` (or `REXL3_TP_RANKS`, `REXL3_TP_RANK`, `REXL3_TP_PEER`) runs one Nemotron model as N
//! processes in lockstep (`docs/NEMOTRON_TP.md`): rank 0 is this server, unchanged; every other rank loads its shard and replays rank 0's
//! calls, with no HTTP front end (a health line on stderr). `--tp-peer` is rank 0's rendezvous address, the same on every rank: rank 0
//! listens on it, the others connect to it. Every rank takes the same model, `--slots`, `--max-model-len`, `--prefill-chunk`, `--kv-tokens`
//! (and `--kv-headroom-tokens`: a follower that plans fewer KV pages than rank 0 is refused by name, one that plans more loads rank 0's count)
//! and `--spec`; a mismatch refuses the boot by name. Without `--tp-ranks` (or with 1) nothing changes.
//!
//! `--backend mock` runs the CPU mock model (for protocol testing only: its text is noise). `--mock key=value,...` steers it (step and
//! prefill latency, prefill slice, vocabulary, injected faults; see `gb_engine::mock::MockOptions`): test controls, not a feature.
//!
//! SIGTERM (and SIGINT) shut the server down gracefully: it stops admitting, answers queued requests 503, lets running requests finish
//! for up to `--shutdown-grace` seconds (the whole stop, from the signal to the process exiting, fits inside it), ends what is still
//! running as an error (a stream's last event is an error, never a normal finish), and exits 0.
//!
//! An engine fault ends the process non-zero: a fatal CUDA error or backend error, a panic on the engine thread (also while booting),
//! or `--exit-after` consecutive step errors. Readiness (`/health`, `/health/ready`) is 503 while loading, draining, after
//! `--unready-after` consecutive step errors, and once the engine has failed.
#![forbid(unsafe_code)]

use gb_engine::mock::MockOptions;
use gb_engine::{Drafter, Engine, EngineConfig, NoDrafts, ServeEngine};
use gb_sched::{Admission, CostModel, ModelMemory};
use gb_server::api::GenDefaults;
use gb_server::build_info::BuildInfo;
use gb_server::chat_format::FormatRegistry;
use gb_server::http::{router, AppState};
use gb_server::runtime::{spawn_with, EngineHandle, RuntimeConfig};
use gb_server::tokenizer::Tok;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

struct Args {
    model: PathBuf,
    request_limits: gb_server::budget::Limits,
    model_aliases: Vec<String>,
    draft: Option<PathBuf>,
    host: String,
    port: u16,
    max_model_len: usize,
    name: String,
    backend: String,
    slots: usize,
    /// `--slots` was not given: `slots` is the default and a family that cannot hold it boots with the most it can (`ServeOptions::slots_auto`).
    slots_auto: bool,
    drafting: bool,
    /// `--spec auto|mtp|copy|none` (`none` is `--no-drafts`).
    spec: gb_model::serve::SpecMode,
    prefill_chunk: usize,
    window_rows: usize,
    kv_tokens: Option<usize>,
    kv_headroom_tokens: usize,
    default_max_tokens: usize,
    /// `--default-thinking off|template`: what a chat request that names neither `enable_thinking` nor a `reasoning_effort` gets.
    default_thinking: gb_server::reasoning_controls::DefaultThinking,
    /// `--default-reasoning-effort LEVEL`: the effort a chat request that names neither `enable_thinking` nor an effort gets (checked against
    /// the model once its reasoning capabilities are known; replaces `--default-thinking`'s insertion for such a request).
    default_reasoning_effort: Option<String>,
    /// `--reasoning-unverified`: offer the family's declared reasoning controls even where the load-time check against the template fails
    /// (experts only; the boot line and `/v1/models` then say `verdict=overridden`).
    reasoning_unverified: bool,
    vision: bool,
    /// `--max-image-tokens N`; without it the family's own default.
    max_image_tokens: Option<usize>,
    /// Video input, for a family whose tower encodes it: `--max-video-frames`, `--max-video-tokens`, `--video-pruning-rate`,
    /// `--video-evs-strict`; `--no-video` leaves it off.
    video: bool,
    video_options: gb_server::chat_format::VideoOptions,
    image_preparation: gb_server::preparation::PreparationLimits,
    image_ingress: gb_server::vision_ingress::IngressPolicy,
    allow_unvalidated_pack: bool,
    intervention: Option<String>,
    /// `--chat-format ID`: name a registered chat format instead of letting `config.json` choose.
    chat_format: Option<String>,
    print_fingerprints: bool,
    print_crc32: bool,
    verify_pack: bool,
    spec_tree: bool,
    queue_depth: usize,
    queue_timeout_secs: u64,
    request_timeout_secs: u64,
    write_stall_secs: u64,
    heartbeat_stale_secs: u64,
    loading_budget_secs: u64,
    prefill_budget_secs: u64,
    /// `--header-read-timeout SECS`: a request head must arrive within it, on a new connection and on an idle keep-alive one.
    header_read_timeout_secs: u64,
    decision_timeout_secs: u64,
    shutdown_grace_secs: u64,
    unready_after: u32,
    exit_after: u32,
    mock: MockOptions,
    /// `--prefix-cache on|off`
    prefix_cache: Option<bool>,
    /// `--prefix-cache-mib N`
    prefix_cache_mib: Option<u64>,
    /// `--tp-ranks`, `--tp-rank`, `--tp-peer` and the `REXL3_TP_*` variables, resolved (`None`: one process).
    tp: Option<gb_model::serve::TpOptions>,
}

/// The tensor-parallel settings: the environment's, then the flags over them.
#[derive(Default)]
struct TpFlags {
    ranks: Option<usize>,
    rank: Option<usize>,
    peer: Option<String>,
    connect_timeout: Option<u64>,
    ready_timeout: Option<u64>,
    peer_timeout: Option<u64>,
}

fn resolve_tp(flags: &TpFlags, env: impl Fn(&str) -> Option<String>) -> Result<Option<gb_model::serve::TpOptions>, String> {
    use gb_model::serve::TpOptions;
    let mut t = TpOptions::from_vars(env)?;
    let any_flag = flags.ranks.is_some() || flags.rank.is_some() || flags.peer.is_some();
    if t.is_none() && !any_flag {
        return Ok(None);
    }
    let t = t.get_or_insert_with(|| TpOptions::new(1, 0, ""));
    if let Some(n) = flags.ranks {
        t.ranks = n;
    }
    if let Some(r) = flags.rank {
        t.rank = r;
    }
    if let Some(p) = &flags.peer {
        t.peer = p.clone();
    }
    for (v, slot) in [
        (flags.connect_timeout, &mut t.connect_timeout),
        (flags.ready_timeout, &mut t.ready_timeout),
        (flags.peer_timeout, &mut t.peer_timeout),
    ] {
        if let Some(s) = v {
            *slot = Duration::from_secs(s);
        }
    }
    t.validate()?;
    // one rank is no group: nothing about it is looked at again
    Ok(Some(t.clone()))
}

fn parse() -> Result<Args, String> {
    parse_from(std::env::args().skip(1))
}

fn parse_from(mut it: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut a = Args {
        model: PathBuf::new(),
        request_limits: gb_server::budget::Limits::default(),
        model_aliases: Vec::new(),
        draft: None,
        host: "127.0.0.1".into(),
        port: 8889,
        max_model_len: 262144,
        name: String::new(),
        backend: "cuda".into(),
        slots: 8,
        slots_auto: true,
        drafting: true,
        spec: gb_model::serve::SpecMode::Auto,
        prefill_chunk: 4096,
        window_rows: gb_model::forward::WINDOW_ROWS,
        kv_tokens: None,
        kv_headroom_tokens: 0,
        default_max_tokens: 16384,
        default_thinking: gb_server::reasoning_controls::DefaultThinking::Off,
        default_reasoning_effort: None,
        reasoning_unverified: false,
        vision: true,
        max_image_tokens: None,
        video: true,
        video_options: Default::default(),
        image_preparation: Default::default(),
        image_ingress: Default::default(),
        allow_unvalidated_pack: false,
        intervention: None,
        chat_format: None,
        print_fingerprints: false,
        print_crc32: false,
        verify_pack: false,
        spec_tree: true,
        queue_depth: 64,
        queue_timeout_secs: 300,
        request_timeout_secs: 1800,
        write_stall_secs: 30,
        heartbeat_stale_secs: 60,
        loading_budget_secs: 1800,
        prefill_budget_secs: 1800,
        header_read_timeout_secs: 30,
        decision_timeout_secs: 1800,
        shutdown_grace_secs: 8,
        unready_after: 3,
        exit_after: 10,
        mock: MockOptions::default(),
        prefix_cache: None,
        prefix_cache_mib: None,
        tp: None,
    };
    let mut tp_flags = TpFlags::default();
    // the rendered-prompt cap and the processing reserve follow the context window unless a flag names them (in either order)
    let (mut rendered_flag, mut reserve_flag) = (false, false);
    while let Some(k) = it.next() {
        let mut val = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match k.as_str() {
            "--model" => a.model = PathBuf::from(val("--model")?),
            "--model-alias" => a.model_aliases.push(val("--model-alias")?),
            "--max-body-mib" => a.request_limits.body_bytes = mib(&val(&k)?, &k)?,
            "--max-total-request-mib" => a.request_limits.total_bytes = mib(&val(&k)?, &k)?,
            "--max-rendered-prompt-mib" => {
                a.request_limits.rendered_bytes = mib(&val(&k)?, &k)?;
                rendered_flag = true;
            }
            "--processing-reserve-mib" => {
                a.request_limits.processing_reserve_bytes = mib(&val(&k)?, &k)?;
                reserve_flag = true;
            }
            "--max-json-nodes" => {
                a.request_limits.json_nodes = val(&k)?.parse().map_err(|_| "max-json-nodes must be an integer")?
            }
            "--max-json-depth" => {
                a.request_limits.json_depth = val(&k)?.parse().map_err(|_| "max-json-depth must be an integer")?
            }
            "--preparation-queue" => {
                a.request_limits.preparation_queue = val(&k)?.parse().map_err(|_| "preparation-queue must be an integer")?
            }
            "--preparation-concurrency" => {
                a.request_limits.preparation_concurrency =
                    val(&k)?.parse().map_err(|_| "preparation-concurrency must be an integer")?
            }
            "--draft" => a.draft = Some(PathBuf::from(val("--draft")?)),
            "--host" => a.host = val("--host")?,
            "--port" => a.port = val("--port")?.parse().map_err(|_| "--port must be a number")?,
            "--max-model-len" => {
                a.max_model_len = val("--max-model-len")?.parse().map_err(|_| "--max-model-len must be a number")?
            }
            "--served-model-name" => a.name = val("--served-model-name")?,
            "--backend" => a.backend = val("--backend")?,
            "--slots" => {
                a.slots = val("--slots")?.parse().map_err(|_| "--slots must be a number")?;
                a.slots_auto = false;
            }
            "--no-drafts" => a.drafting = false,
            "--spec" => a.spec = val("--spec")?.parse()?,
            "--prefill-chunk" => {
                a.prefill_chunk = val("--prefill-chunk")?.parse().map_err(|_| "--prefill-chunk must be a number")?
            }
            "--window-rows" => a.window_rows = val("--window-rows")?.parse().map_err(|_| "--window-rows must be a number")?,
            "--default-max-tokens" => {
                a.default_max_tokens =
                    val("--default-max-tokens")?.parse().map_err(|_| "--default-max-tokens must be a number")?
            }
            "--default-thinking" => {
                a.default_thinking = gb_server::reasoning_controls::DefaultThinking::parse(&val("--default-thinking")?)
                    .ok_or("--default-thinking must be off or template")?
            }
            "--default-reasoning-effort" => a.default_reasoning_effort = Some(val("--default-reasoning-effort")?),
            "--reasoning-unverified" => a.reasoning_unverified = true,
            "--no-vision" => a.vision = false,
            // the KV cache's element type (Nemotron): fp16 (default) or int8 (a power-of-two scale per token and kv head, opt-in for
            // long-context decode, docs/NEMOTRON_KV_INT8.md). Carried to the model as REXL3_KV_DTYPE, set here before anything is built;
            // every TP rank must pass the same value (compared at the rendezvous).
            // Three families take this flag and each parses only its own spellings, so the value picks the family:
            // `q4` is DeepSeek-V4.1's cache (`REXL3_DSV41_KV`), `int8` is Nemotron's (`REXL3_KV_DTYPE`), `fp8` is GLM's latent
            // cache (`REXL3_GLM_KV`, E4M3 with an fp32 scale per token, gb_cuda::mla::kv8), and `fp16` is all three. An unknown
            // value is an error.
            "--kv-dtype" => {
                let v = val("--kv-dtype")?;
                let dsv41 = gb_cuda::dsv4_kvq4::KvFmt::parse(&v).ok();
                let nemotron = gb_cuda::nemotron_attn::KvDtype::parse(&v);
                let glm = gb_cuda::mla::LatentKv::parse(&v).ok();
                match (dsv41, nemotron, glm) {
                    (None, Err(e), None) => return Err(format!("--kv-dtype: {e} (GLM: fp16 or fp8)")),
                    (dsv41, nemotron, glm) => {
                        if let Some(f) = dsv41 {
                            std::env::set_var(gb_cuda::dsv4_kvq4::KV_FMT_ENV, f.name());
                        }
                        if let Ok(d) = nemotron {
                            std::env::set_var(gb_cuda::nemotron_attn::KV_DTYPE_ENV, d.name());
                        }
                        if let Some(g) = glm {
                            std::env::set_var(gb_cuda::mla::KV_ENV, g.name());
                        }
                    }
                }
            }
            "--image-urls" => {
                a.image_ingress.enabled = match val("--image-urls")?.as_str() {
                    "on" => true,
                    "off" => false,
                    _ => return Err("--image-urls needs on|off".into()),
                }
            }
            "--image-url-max-bytes" => {
                a.image_ingress.max_bytes = val("--image-url-max-bytes")?.parse().map_err(|_| "invalid image URL byte limit")?
            }
            "--image-url-connect-timeout" => {
                a.image_ingress.connect_timeout = Duration::from_secs(
                    val("--image-url-connect-timeout")?.parse().map_err(|_| "invalid image URL connect timeout")?,
                )
            }
            "--image-url-total-timeout" => {
                a.image_ingress.total_timeout =
                    Duration::from_secs(val("--image-url-total-timeout")?.parse().map_err(|_| "invalid image URL total timeout")?)
            }
            "--image-preparation-bytes" => {
                a.image_preparation.bytes =
                    val("--image-preparation-bytes")?.parse().map_err(|_| "invalid image preparation byte limit")?
            }
            "--image-retained-bytes" => {
                a.image_preparation.retained_bytes =
                    val("--image-retained-bytes")?.parse().map_err(|_| "invalid image retained byte limit")?
            }
            "--image-preparation-workers" => {
                a.image_preparation.workers =
                    val("--image-preparation-workers")?.parse().map_err(|_| "invalid image preparation worker limit")?
            }
            "--image-preparation-backlog" => {
                a.image_preparation.backlog =
                    val("--image-preparation-backlog")?.parse().map_err(|_| "invalid image preparation backlog")?
            }
            "--image-preparation-wait" => {
                a.image_preparation.wait =
                    Duration::from_secs(val("--image-preparation-wait")?.parse().map_err(|_| "invalid image preparation wait")?)
            }
            "--image-preparation-deadline" => {
                a.image_preparation.deadline = Duration::from_secs(
                    val("--image-preparation-deadline")?.parse().map_err(|_| "invalid image preparation deadline")?,
                )
            }
            "--max-image-tokens" => {
                a.max_image_tokens = Some(val("--max-image-tokens")?.parse().map_err(|_| "--max-image-tokens must be a number")?)
            }
            "--no-video" => a.video = false,
            "--max-video-frames" => {
                a.video_options.max_frames =
                    val("--max-video-frames")?.parse().map_err(|_| "--max-video-frames must be a number")?
            }
            "--max-video-tokens" => {
                a.video_options.max_tokens =
                    val("--max-video-tokens")?.parse().map_err(|_| "--max-video-tokens must be a number")?
            }
            "--video-pruning-rate" => {
                a.video_options.pruning_rate =
                    val("--video-pruning-rate")?.parse().map_err(|_| "--video-pruning-rate must be a number in [0, 1]")?
            }
            "--video-evs-strict" => a.video_options.require_square_grid = true,
            "--kv-tokens" => a.kv_tokens = Some(val("--kv-tokens")?.parse().map_err(|_| "--kv-tokens must be a number")?),
            "--kv-headroom-tokens" => {
                a.kv_headroom_tokens =
                    val("--kv-headroom-tokens")?.parse().map_err(|_| "--kv-headroom-tokens must be a number of tokens")?
            }
            "--allow-unvalidated-pack" => a.allow_unvalidated_pack = true,
            "--intervention" => a.intervention = Some(val("--intervention")?),
            "--chat-format" => a.chat_format = Some(val("--chat-format")?),
            "--print-fingerprints" => a.print_fingerprints = true,
            "--print-crc32" => a.print_crc32 = true,
            "--verify-pack" => a.verify_pack = true,
            "--spec-tree" => a.spec_tree = true,
            "--no-spec-tree" => a.spec_tree = false,
            "--decision-timeout" => {
                a.decision_timeout_secs =
                    val("--decision-timeout")?.parse().map_err(|_| "--decision-timeout must be a number of seconds")?;
            }
            "--queue-depth" => a.queue_depth = val("--queue-depth")?.parse().map_err(|_| "--queue-depth must be a number")?,
            "--queue-timeout" => {
                a.queue_timeout_secs =
                    val("--queue-timeout")?.parse().map_err(|_| "--queue-timeout must be a number of seconds")?
            }
            "--request-timeout" => {
                a.request_timeout_secs = val("--request-timeout")?.parse().map_err(|_| "invalid request timeout")?
            }
            "--write-stall" => a.write_stall_secs = val("--write-stall")?.parse().map_err(|_| "invalid write stall")?,
            "--heartbeat-stale-secs" => {
                a.heartbeat_stale_secs = val("--heartbeat-stale-secs")?.parse().map_err(|_| "invalid heartbeat budget")?
            }
            "--loading-budget" => {
                a.loading_budget_secs = val("--loading-budget")?.parse().map_err(|_| "invalid loading budget")?
            }
            "--prefill-budget" => {
                a.prefill_budget_secs = val("--prefill-budget")?.parse().map_err(|_| "invalid prefill budget")?
            }
            "--header-read-timeout" => {
                a.header_read_timeout_secs = val("--header-read-timeout")?.parse().map_err(|_| "invalid header read timeout")?
            }
            "--shutdown-grace" => {
                a.shutdown_grace_secs =
                    val("--shutdown-grace")?.parse().map_err(|_| "--shutdown-grace must be a number of seconds")?
            }
            "--unready-after" => {
                a.unready_after = val("--unready-after")?.parse().map_err(|_| "--unready-after must be a number")?
            }
            "--exit-after" => a.exit_after = val("--exit-after")?.parse().map_err(|_| "--exit-after must be a number")?,
            "--mock" => a.mock = MockOptions::parse(&val("--mock")?)?,
            "--prefix-cache" => {
                a.prefix_cache = Some(match val("--prefix-cache")?.as_str() {
                    "on" => true,
                    "off" => false,
                    other => return Err(format!("--prefix-cache is `on` or `off`, got `{other}`")),
                })
            }
            "--prefix-cache-mib" => {
                a.prefix_cache_mib = Some(
                    val("--prefix-cache-mib")?.parse().map_err(|_| "--prefix-cache-mib must be a number of MiB (0 disables)")?,
                )
            }
            "--tp-ranks" => {
                tp_flags.ranks = Some(
                    val("--tp-ranks")?
                        .parse()
                        .map_err(|_| "--tp-ranks must be a number (1: one process; 2 and more: a group)")?,
                )
            }
            "--tp-rank" => {
                tp_flags.rank =
                    Some(val("--tp-rank")?.parse().map_err(|_| "--tp-rank must be a number (0 serves HTTP; 1 and up follow)")?)
            }
            "--tp-peer" => tp_flags.peer = Some(val("--tp-peer")?),
            "--tp-connect-timeout" => {
                tp_flags.connect_timeout =
                    Some(val("--tp-connect-timeout")?.parse().map_err(|_| "--tp-connect-timeout must be a number of seconds")?)
            }
            "--tp-ready-timeout" => {
                tp_flags.ready_timeout =
                    Some(val("--tp-ready-timeout")?.parse().map_err(|_| "--tp-ready-timeout must be a number of seconds")?)
            }
            "--tp-peer-timeout" => {
                tp_flags.peer_timeout =
                    Some(val("--tp-peer-timeout")?.parse().map_err(|_| "--tp-peer-timeout must be a number of seconds")?)
            }
            "-V" | "--version" => {
                for (k, v) in BuildInfo::get().lines() {
                    println!("{k:<20} {v}");
                }
                std::process::exit(0);
            }
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.model.as_os_str().is_empty() {
        return Err("--model is required".into());
    }
    a.tp = resolve_tp(&tp_flags, |k| std::env::var(k).ok())?;
    if a.tp.as_ref().is_some_and(|t| t.is_group()) && a.backend != "cuda" {
        return Err("--tp-ranks needs --backend cuda when it names a group (2 or more)".into());
    }
    if a.name.is_empty() {
        a.name = a.model.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "model".into());
    }
    if a.decision_timeout_secs == 0 || a.decision_timeout_secs > 86400 {
        return Err("--decision-timeout must be 1..86400 seconds".into());
    }
    if a.queue_depth == 0 {
        return Err("--queue-depth must be at least 1".into());
    }
    if a.queue_timeout_secs == 0 {
        return Err("--queue-timeout must be at least 1 second".into());
    }
    for n in [
        a.request_timeout_secs,
        a.write_stall_secs,
        a.heartbeat_stale_secs,
        a.loading_budget_secs,
        a.prefill_budget_secs,
        a.header_read_timeout_secs,
        a.queue_timeout_secs,
        a.shutdown_grace_secs,
    ] {
        if n == 0 || n > 86400 {
            return Err("lifecycle time budgets must be in 1..=86400 seconds".into());
        }
    }
    if a.unready_after == 0 || a.exit_after < a.unready_after {
        return Err("--unready-after must be at least 1 and --exit-after at least --unready-after".into());
    }
    // defaults that follow the window: 16 bytes of rendered prompt per window token (at least 1 MiB), and a reserve that holds one prompt at
    // that cap (`budget::Limits::for_window`); a flag keeps what it names
    if !rendered_flag {
        a.request_limits.rendered_bytes = gb_server::budget::default_rendered_bytes(a.max_model_len);
    }
    if !reserve_flag {
        a.request_limits.processing_reserve_bytes = a.request_limits.derived_reserve();
    }
    a.request_limits.validate()?;
    if a.model_aliases.iter().any(|s| s.is_empty() || s.len() > 256) {
        return Err("model aliases must be 1..256 bytes".into());
    }
    gb_server::preparation::PreparationGate::new(a.image_preparation)?;
    gb_server::vision_ingress::HttpsIngress::new(a.image_ingress)?;
    if a.image_preparation.deadline.is_zero()
        || a.image_preparation.deadline > Duration::from_secs(86400)
        || a.image_preparation.wait.is_zero()
    {
        return Err("image preparation deadlines must be nonzero and at most one day".into());
    }
    Ok(a)
}

fn mib(value: &str, name: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .and_then(|v| v.checked_mul(1 << 20))
        .filter(|v| *v > 0)
        .ok_or_else(|| format!("{name} must be positive MiB without arithmetic overflow"))
}

/// Is this process a member of a tensor-parallel group (rank 0 or a follower)? Decides how every fatal path ends it.
fn is_tp_group(a: &Args) -> bool {
    a.tp.as_ref().is_some_and(|t| t.is_group())
}

fn runtime_config(a: &Args) -> RuntimeConfig {
    RuntimeConfig {
        fatal_exit: gb_model::exit::fatal_exit_mode(is_tp_group(a)),
        queue_depth: a.queue_depth,
        queue_timeout: Duration::from_secs(a.queue_timeout_secs),
        request_timeout: Duration::from_secs(a.request_timeout_secs),
        write_stall: Duration::from_secs(a.write_stall_secs),
        heartbeat_stale: Duration::from_secs(a.heartbeat_stale_secs),
        loading_budget: Duration::from_secs(a.loading_budget_secs),
        prefill_budget: Duration::from_secs(a.prefill_budget_secs),
        decision_timeout: Duration::from_secs(a.decision_timeout_secs),
        unready_after: a.unready_after,
        exit_after: a.exit_after,
        leak_engine_on_fatal: is_tp_group(a),
        ..RuntimeConfig::default()
    }
}

fn serve_options(a: &Args, vision_slot: Option<gb_model::serve::VisionSlot>) -> gb_model::serve::ServeOptions {
    let mut o = gb_model::serve::ServeOptions::new(a.model.clone());
    o.slots = a.slots;
    o.slots_auto = a.slots_auto;
    o.max_model_len = a.max_model_len;
    o.drafting = a.drafting && a.spec.drafts();
    o.spec = a.spec;
    o.draft_dir = gb_model::serve::draft_dir_for(a.spec, a.draft.clone());
    o.prefill_chunk = a.prefill_chunk;
    o.window_rows = a.window_rows;
    o.kv_tokens = a.kv_tokens;
    o.kv_headroom_tokens = a.kv_headroom_tokens;
    o.vision = vision_slot;
    o.max_image_tokens = a.max_image_tokens.unwrap_or(4096);
    o.allow_unvalidated_pack = a.allow_unvalidated_pack;
    o.intervention_dir = a.intervention.clone().map(std::path::PathBuf::from);
    o.verify_pack = a.verify_pack;
    o.spec_tree = a.spec_tree;
    o.tp = a.tp.clone().filter(|t| t.is_group());
    o.prefix_cache = match (a.prefix_cache, a.prefix_cache_mib) {
        (_, Some(0)) | (Some(false), _) => gb_model::serve::PrefixCacheOption::Off,
        (_, Some(mib)) => gb_model::serve::PrefixCacheOption::Mib(mib),
        (Some(true), None) => gb_model::serve::PrefixCacheOption::On,
        (None, None) => gb_model::serve::DEFAULT_PREFIX_CACHE,
    };
    o
}

fn start_engine(
    a: &Args,
    vocab: usize,
    vision_slot: Option<gb_model::serve::VisionSlot>,
) -> Result<(EngineHandle, JoinHandle<()>), String> {
    let cfg = runtime_config(a);
    match a.backend.as_str() {
        "mock" => {
            let (slots, drafting, mock) = (a.slots, a.drafting && a.spec.drafts(), a.mock.clone());
            // like the real engine, the KV pool defaults to one full-length context (`--kv-tokens` shrinks it), plus the co-tenancy headroom
            let kv_tokens = (a.kv_tokens.unwrap_or(a.max_model_len) + a.kv_headroom_tokens) as u64;
            let max_len = a.max_model_len as u64;
            let cache_enabled = a.prefix_cache != Some(false) && a.prefix_cache_mib != Some(0);
            Ok(spawn_with(
                Box::new(move || {
                    let until = std::time::Instant::now() + Duration::from_millis(mock.boot_ms);
                    while std::time::Instant::now() < until {
                        gb_engine::boot::checkpoint();
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    if mock.panic_boot {
                        panic!("injected panic while building the engine");
                    }
                    let mm = ModelMemory {
                        weights_bytes: 0,
                        kv_bytes_per_token: 65536,
                        state_bytes_per_slot: 160 << 20,
                        activation_reserve: 0,
                    };
                    let capacity = kv_tokens * 65536 + slots as u64 * (160 << 20);
                    let prefix = (mock.prefix_cache > 0 && cache_enabled).then_some(gb_engine::PrefixConfig {
                        policy: gb_engine::CutPolicy { min_cut: 64, tail: 8, ghost_min_gain: 256, min_gap: 0 },
                        ghost_blocks: 1 << 12,
                        ..Default::default()
                    });
                    let engine = Engine::new(
                        mock.build(vocab, slots),
                        Admission::new(capacity, 0, mm, slots)
                            .unwrap_or_else(|e| {
                                eprintln!("error: cannot start the engine: {e}");
                                panic!("cannot start the mock engine: {e}")
                            })
                            .with_max_context(max_len)
                            // the pool in tokens, as `MemoryPlan::admission` declares it for the real engines
                            .with_kv_tokens(kv_tokens),
                        EngineConfig {
                            max_rows: 64,
                            max_drafts: 7,
                            cost: CostModel::from_table(vec![(1, 70.0), (16, 74.0), (64, 190.0)]),
                            drafting,
                        },
                        Box::new(|| Box::new(NoDrafts) as Box<dyn Drafter>),
                    );
                    Box::new(match prefix {
                        Some(p) => engine.with_prefix_cache(p),
                        None => engine,
                    }) as Box<dyn ServeEngine>
                }),
                cfg,
            ))
        }
        "cuda" => {
            // the registry picks the family from the pack's config and the family builds its engine; the server names no family
            let o = serve_options(a, vision_slot);
            Ok(spawn_with(
                Box::new(move || {
                    gb_model::serve::start(&o).unwrap_or_else(|e| {
                        eprintln!("error: cannot start the engine: {e}");
                        panic!("cannot start the engine: {e}")
                    })
                }),
                cfg,
            ))
        }
        other => Err(format!("backend `{other}` is not available in this build")),
    }
}

/// `--print-fingerprints`: each pack's fingerprint and whether this release was validated on the pair.
fn print_fingerprints(a: &Args) -> Result<(), String> {
    let open = |d: &std::path::Path| gb_model::pack::Pack::open(d).map_err(|e| format!("{}: {e}", d.display()));
    let t = gb_model::fingerprint::fingerprint(&open(&a.model)?.ck);
    let d = match &a.draft {
        Some(dir) => Some(gb_model::fingerprint::fingerprint(&open(dir)?.ck)),
        None => None,
    };
    println!("target    {t}");
    if let Some(d) = &d {
        println!("draft     {d}");
    }
    // the one validation the serve paths of every family use (a text-only pack is an entry of its own, not a pair)
    println!("{}", gb_model::fingerprint::validation_line(&t, d.as_deref()));
    Ok(())
}

/// The malloc arena cap a process gets when nothing else names one (see [`reexec_arena_cap`]).
const ARENA_CAP: &str = "2";
/// Arguments after which the process prints and exits (`--version`, `--help`, `--print-fingerprints`, `--print-crc32`): never re-executed.
const NON_SERVING_FLAGS: [&str; 6] = ["--version", "-V", "-h", "--help", "--print-fingerprints", "--print-crc32"];

/// The cap `main` should re-execute with, or `None` to go on as it is. `env` reads the process environment.
fn arena_cap_wanted(env: impl Fn(&str) -> Option<String>) -> Option<&'static str> {
    let named = env("MALLOC_ARENA_MAX").is_some()
        || env("GLIBC_TUNABLES").is_some_and(|t| t.split(':').any(|kv| kv.starts_with("glibc.malloc.arena_max=")));
    let opted_out = env("REXL3_MALLOC_REEXEC").is_some_and(|v| matches!(v.as_str(), "0" | "off" | "false" | "no"));
    (!named && !opted_out).then_some(ARENA_CAP)
}

/// glibc gives each thread that allocates its own malloc arena (up to eight a core) and keeps in it what the thread freed. The tokenizer of
/// a large prompt makes millions of one-byte allocations on whichever pool thread picked the job up, so a server that tokenizes on a pool of
/// threads ratchets its resident memory up by a prompt's working set per arena and never gives it back (`docs/REQUEST_CONTRACT.md`,
/// "Request memory"). `MALLOC_ARENA_MAX` bounds that, and glibc reads it once, at process start. A process that finds it unset
/// therefore re-executes itself once with `MALLOC_ARENA_MAX=2`: same pid, arguments, environment and descriptors, nothing added but the
/// variable. It is left alone when the variable (or `glibc.malloc.arena_max` in `GLIBC_TUNABLES`) is set by the caller, and
/// `REXL3_MALLOC_REEXEC=0` turns it off. `exec` is `std::os::unix::process::CommandExt`: no `unsafe`. A re-execution that cannot happen
/// (no `/proc`, a seccomp profile that refuses `execve`) leaves the process as it was.
fn reexec_arena_cap() {
    use std::os::unix::process::CommandExt;
    // the modes that print and exit serve nothing: no extra process image, no extra line on the stderr a script captures from them
    if std::env::args().any(|a| NON_SERVING_FLAGS.contains(&a.as_str())) {
        return;
    }
    let Some(cap) = arena_cap_wanted(|k| std::env::var(k).ok()) else { return };
    let mut args = std::env::args_os();
    // `/proc/self/exe` and not `current_exe()`: it still names the binary when its file was replaced after this process started
    let mut cmd = std::process::Command::new("/proc/self/exe");
    if let Some(argv0) = args.next() {
        cmd.arg0(argv0);
    }
    // `REXL3_MALLOC_REEXEC=0` beside the cap: the new image never re-executes again, even where the loader drops `MALLOC_ARENA_MAX` from the
    // environment of a secure-execution (setuid or file-capability) binary, which would otherwise loop for ever
    cmd.args(args).env("MALLOC_ARENA_MAX", cap).env("REXL3_MALLOC_REEXEC", "0");
    eprintln!(
        "rexl3 event=reexec reason=malloc_arena_max value={cap} (set MALLOC_ARENA_MAX, or REXL3_MALLOC_REEXEC=0, to keep it off)"
    );
    // `exec` returns only when it failed: go on with the arenas glibc picked
    let e = cmd.exec();
    eprintln!("rexl3 level=warn event=reexec_failed error={e}");
}

fn main() {
    // before the runtime and any thread: the arena cap is read by the allocator once, when the process starts
    reexec_arena_cap();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|e| fail(&format!("cannot start the async runtime: {e}")))
        .block_on(serve_main())
}

fn apply_family_defaults(a: &mut Args) {
    if a.backend != "cuda" {
        return;
    }
    let Ok((family, plan)) = gb_model::family::Registry::global().probe(&a.model) else {
        return;
    };
    if family.id() == "deepseek_v41" {
        if std::env::var_os(gb_cuda::dsv4_kvq4::KV_FMT_ENV).is_none() {
            std::env::set_var(gb_cuda::dsv4_kvq4::KV_FMT_ENV, gb_cuda::dsv4_kvq4::KvFmt::Q4.name());
        }
        if a.slots_auto {
            a.slots = 1;
        }
        for alias in [
            "dsv41",
            "deepseek_v41",
            "deepseek-v4.1",
            "deepseek-v4.1-flash",
            "DeepSeek-V4.1-Flash",
            "DeepSeek-V4.1-Flash-TF",
            "DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw",
            "vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw",
            "DSV4.1-Flash-1017-SAGE-EXL3-r0b0t-1.60bpw",
            "r0b0tlab/DSV4.1-Flash-1017-SAGE-EXL3-r0b0t-1.60bpw",
            "DSV4.1-Flash-1017-r0b0t-EXL3",
            "r0b0tlab/DSV4.1-Flash-1017-r0b0t-EXL3",
            "default",
        ] {
            if a.name != alias && !a.model_aliases.iter().any(|s| s == alias) {
                a.model_aliases.push(alias.to_string());
            }
        }
        if let Some(dir_name) = a.model.file_name().map(|n| n.to_string_lossy().to_string()) {
            if !dir_name.is_empty() && a.name != dir_name && !a.model_aliases.iter().any(|s| s == &dir_name) {
                a.model_aliases.push(dir_name);
            }
        }
        if a.max_model_len > 524_288 {
            if a.request_timeout_secs == 1800 {
                a.request_timeout_secs = 14_400;
            }
            if a.prefill_budget_secs == 1800 {
                a.prefill_budget_secs = 14_400;
            }
            if a.decision_timeout_secs == 1800 {
                a.decision_timeout_secs = 14_400;
            }
            if a.queue_timeout_secs == 300 {
                a.queue_timeout_secs = 14_400;
            }
            if let Some(tp) = &mut a.tp {
                if tp.peer_timeout < Duration::from_secs(14_400) {
                    tp.peer_timeout = Duration::from_secs(14_400);
                }
            }
        }
    }
    if family.id() == "qwen3_5" {
        for alias in [
            "ornith",
            "ornith-1.5",
            "Ornith-1.5-122B",
            "Ornith-1.5-122B-EXL3-SAGE-2.58bpw",
            "vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw",
            "qwen3_5_moe",
            "default",
        ] {
            if a.name != alias && !a.model_aliases.iter().any(|s| s == alias) {
                a.model_aliases.push(alias.to_string());
            }
        }
    }
    if a.vision && plan.unbuilt_vision().is_some() {
        eprintln!("rexl3 event=vision_disabled family={} reason=tower_not_built", family.id());
        a.vision = false;
    }
}

async fn serve_main() {
    gb_engine::boot::start();
    // Install before any loading, so an early signal is queued rather than taking the default termination path.
    use tokio::signal::unix::{signal, SignalKind};
    let term = signal(SignalKind::terminate()).unwrap_or_else(|e| fail(&e.to_string()));
    let int = signal(SignalKind::interrupt()).unwrap_or_else(|e| fail(&e.to_string()));
    let mut a = match parse() {
        Ok(a) => a,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("error: {e}");
            }
            eprintln!("usage: rexl3-serve --model DIR [--draft DIR] [--port 8889] [--host 127.0.0.1] [--max-model-len N] [--served-model-name NAME] [--backend mock|cuda] [--slots N (default 8; fewer when the box holds fewer, unless the flag is given)] [--spec auto|mtp|copy|none] [--no-drafts] [--default-max-tokens N (0 = whole window)] [--default-thinking off|template (default off: a request that names neither enable_thinking nor reasoning_effort does not think; template: the chat template's own default)] [--default-reasoning-effort LEVEL (a request that names no effort and does not turn thinking off gets this level; a request that names an effort or sets enable_thinking false keeps its own)] [--reasoning-unverified (offer declared reasoning controls the template check refuses)] [--kv-tokens N] [--kv-headroom-tokens N (0)] [--prefill-chunk N] [--window-rows N (up to 128)] [--no-vision] [--kv-dtype fp16|int8|q4|fp8 (Nemotron: fp16, the default and the qualified one, or int8; DeepSeek-V4.1: fp16 or q4, a 4-bit compressed plane and indexer keys, see crates/cuda/src/dsv4_kvq4.rs; GLM: fp16 or fp8, an E4M3 latent with an fp32 scale per token, see crates/cuda/csrc/mla_kv8.cuh; int8 experimental: not qualified for v0.1.0 (the 1M NIAH, task-level recall and two-node exactness-at-depth checks are pending))] [--max-image-tokens N] [--no-video] [--max-video-frames N (32)] [--max-video-tokens N (4096)] [--video-pruning-rate Q (0 = off)] [--video-evs-strict] [--chat-format ID] [--allow-unvalidated-pack] [--intervention DIR] [--verify-pack] [--no-spec-tree] [--prefix-cache on|off] [--prefix-cache-mib N] [--tp-ranks N (2 or more) --tp-rank 0..N-1 --tp-peer HOST:PORT (rank 0's rendezvous address, the same on every rank; REXL3_TP_RANKS, REXL3_TP_RANK, REXL3_TP_PEER) [--tp-connect-timeout SECS (default 900)] [--tp-ready-timeout SECS (default 1800)] [--tp-peer-timeout SECS (default 15)]] [--queue-depth N] [--queue-timeout SECS] [--request-timeout SECS (default 1800)] [--write-stall SECS (default 30)] [--heartbeat-stale-secs SECS (default 60)] [--loading-budget SECS (default 1800)] [--prefill-budget SECS (default 1800)] [--header-read-timeout SECS (default 30)] [--shutdown-grace SECS] [--unready-after N] [--exit-after N] [--mock SPEC (test only)]");
            eprintln!("       Resource defaults: --max-body-mib 64 --max-total-request-mib 128 --max-json-nodes 300000 --max-json-depth 128 --max-rendered-prompt-mib (16 bytes per --max-model-len token, at least 1; 4 for 262144) --preparation-concurrency 2 --preparation-queue 64 --processing-reserve-mib (2048, or the price of one prompt at the rendered cap, 448 bytes a byte, when that is more); --model-alias NAME (repeatable). TLS/auth belongs at the loopback proxy, not a secret argv flag.");
            eprintln!("image flags: --image-urls off|on (default off; security review required); --image-url-max-bytes N (10485760 max); --image-url-connect-timeout SEC (10); --image-url-total-timeout SEC (30); --image-preparation-bytes N (2147483648); --image-retained-bytes N (134217728); --image-preparation-workers N (2); --image-preparation-backlog N (64); --image-preparation-wait SEC (300); --image-preparation-deadline SEC (1800)");
            eprintln!("       rexl3-serve --version");
            eprintln!("       rexl3-serve --print-fingerprints --model DIR [--draft DIR]");
            eprintln!("       rexl3-serve --print-crc32 --model DIR");
            std::process::exit(2);
        }
    };
    if a.print_fingerprints {
        print_fingerprints(&a).unwrap_or_else(|e| fail(&e));
        return;
    }
    if a.print_crc32 {
        print!("{}", gb_model::verify::manifest_for(&a.model).unwrap_or_else(|e| fail(&e.to_string())));
        return;
    }
    // the boot's prefill rows (`--prefill-chunk`, at most the prefill contract's cap) are a field of `system_fingerprint` when they are
    // not the default's: registered before the start line, so the start line, `/metrics` and every reply name the slice this process cuts
    apply_family_defaults(&mut a);
    let prefill_rows = gb_model::families::nemotron_h_omni::engine::prefill_rows(&serve_options(&a, None));
    gb_model::families::nemotron_h_omni::contract::set_boot_prefill_rows(prefill_rows);
    let b = BuildInfo::get();
    eprintln!("rexl3 event=start {}", b.lines().iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" "));
    // the tensor-parallel boot gate compares the engine commit (and the executable's own bytes) between the ranks
    gb_model::lockstep::set_build_commit(b.commit);
    // a tensor-parallel follower (rank 1) serves no HTTP and no tokenizer: it loads its shard and replays rank 0's calls
    if let Some(tp) = a.tp.clone().filter(|t| t.is_group() && t.rank > 0) {
        run_tp_follower(&a, tp, term, int).await;
    }
    let mut vision_slot: Option<gb_model::serve::VisionSlot> = (a.vision && a.backend == "cuda").then(Default::default);
    // the per-image token limit defaults to the family's own (a pack the registry cannot read is refused by the preflight below, with a reason)
    if vision_slot.is_some() && a.max_image_tokens.is_none() {
        if let Ok(Some(spec)) = gb_model::serve::family_vision(&a.model) {
            a.max_image_tokens = Some(spec.max_image_tokens);
        }
    }
    // Recognise the pack's family before the tokenizer and the chat format are read: an unsupported architecture is refused by name,
    // and a family whose components are not all built gets its memory plan and the not-built error. The family's plan also names its chat format, the one source of truth for it; `--chat-format` overrides.
    let family_format: Option<&'static str> = if a.backend == "cuda" {
        match gb_model::serve::preflight(&serve_options(&a, vision_slot.clone())) {
            Ok(id) => Some(id),
            Err(e) => fail(&format!("cannot start the engine: {e}")),
        }
    } else {
        None
    };
    // Complete fallible CPU setup and binding BEFORE starting the CUDA owner. A tokenizer/template/bind error must never exit concurrent with CUDA initialization.
    let tok = Arc::new(Tok::from_dir(&a.model).unwrap_or_else(|e| fail(&e)));
    if tok.eos.is_empty() {
        // without an end-of-turn id a reply could only end at max_tokens
        fail("generation_config.json names no eos_token_id: a reply could never stop on its own");
    }
    let mut format =
        FormatRegistry::builtin().open(&a.model, a.chat_format.as_deref().or(family_format)).unwrap_or_else(|e| fail(&e));
    // video input: on when vision is on and the family's tower encodes video (the format refuses a video part by name otherwise)
    if a.video && vision_slot.is_some() && a.backend == "cuda" && gb_model::serve::family_encodes_video(&a.model).unwrap_or(false)
    {
        match format.enable_video(&a.video_options) {
            Ok(()) => eprintln!(
                "rexl3 event=video_enabled max_frames={} max_tokens={} pruning_rate={} require_square_grid={}",
                a.video_options.max_frames,
                a.video_options.max_tokens,
                a.video_options.pruning_rate,
                a.video_options.require_square_grid
            ),
            Err(e) => fail(&format!("video input: {e}")),
        }
    }
    let format: Arc<dyn gb_server::chat_format::ChatFormat> = format.into();
    eprintln!("rexl3 event=chat_format id={}", format.id());
    // the model's reasoning capabilities: the family's descriptor checked against its template and tokenizer (or inferred from the template)
    let reasoning_caps =
        Arc::new(gb_server::reasoning_caps::ReasoningSupport::for_format(&*format, &tok, a.reasoning_unverified));
    eprintln!("{}", reasoning_caps.log_line());
    if let Some(level) = &a.default_reasoning_effort {
        a.default_thinking = gb_server::reasoning_controls::default_effort(&reasoning_caps.caps, level)
            .unwrap_or_else(|e| fail(&format!("--default-reasoning-effort {level}: {e}")));
        gb_server::reasoning_controls::check_default_renders(&*format, &reasoning_caps.caps, a.default_thinking)
            .unwrap_or_else(|e| fail(&format!("--default-reasoning-effort {level}: {e}")));
    }
    gb_engine::boot::mark("tokenizer_loaded", None);
    if vision_slot.is_some() && !format.serves_images() {
        eprintln!("rexl3 event=vision_disabled format={} reason=format_text_only", format.id());
        a.vision = false;
        vision_slot = None;
    }

    // the family's image preprocessing, limited like the encoder is
    let image_pp = if vision_slot.is_some() {
        match gb_model::serve::image_preprocessor(&serve_options(&a, None)) {
            Ok(Some(pp)) => Some(pp),
            Ok(None) => {
                eprintln!("rexl3 event=vision_disabled reason=no_image_preprocessor");
                a.vision = false;
                vision_slot = None;
                None
            }
            Err(e) => fail(&format!("cannot set up image preprocessing: {e}")),
        }
    } else {
        None
    };
    let addr = format!("{}:{}", a.host, a.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap_or_else(|e| fail(&format!("cannot bind {addr}: {e}")));
    let bound = listener.local_addr().map(|x| x.to_string()).unwrap_or(addr);
    let (handle, engine_thread) = start_engine(&a, tok.vocab, vision_slot.clone()).unwrap_or_else(|e| fail(&e));
    handle.metrics.observability.slots.store(a.slots as u64, std::sync::atomic::Ordering::Relaxed);
    let state = Arc::new(AppState {
        handle,
        tok,
        format,
        defaults: GenDefaults::from_dir(&a.model),
        model_name: a.name.clone(),
        model_aliases: a.model_aliases.clone(),
        request_budget: gb_server::budget::RequestBudget::new(a.request_limits.clone()).unwrap_or_else(|e| fail(&e)),
        max_model_len: a.max_model_len,
        sse_keepalive: Duration::from_secs(15),
        default_max_tokens: a.default_max_tokens,
        default_thinking: a.default_thinking,
        reasoning_caps,
        vision: vision_slot.zip(image_pp).map(|(slot, pp)| gb_server::http::VisionHandle {
            slot,
            limits: gb_server::vision::Limits { max_tokens_per_image: a.max_image_tokens.unwrap_or(4096), ..Default::default() },
            pp,
            gate: gb_server::preparation::PreparationGate::new(a.image_preparation).unwrap_or_else(|e| fail(&e)),
            ingress: gb_server::vision_ingress::HttpsIngress::new(a.image_ingress).unwrap_or_else(|e| fail(&e)),
        }),
    });

    gb_engine::boot::mark("listening", None);
    eprintln!(
        "rexl3 event=listening model={} addr=http://{bound} max_model_len={} backend={}",
        a.name, a.max_model_len, a.backend
    );
    eprintln!(
        "rexl3 event=reasoning_defaults default_thinking={} default_effort={}",
        a.default_thinking.name(),
        a.default_thinking.effort().unwrap_or("-")
    );
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let monitor = state.handle.metrics.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(250)).await;
            if monitor.wedged() {
                eprintln!(
                    "rexl3 level=fatal event=progress_watchdog heartbeat_age_s={:.3}",
                    monitor.heartbeat_age().as_secs_f64()
                );
                std::process::abort();
            }
        }
    });
    tokio::spawn(shutdown_on_signal(
        state.clone(),
        engine_thread,
        Duration::from_secs(a.shutdown_grace_secs),
        stop_tx,
        term,
        int,
    ));
    // axum::serve's loop with a timer: a request head (and an idle keep-alive connection's next one) must come within the timeout
    gb_server::listen::serve(listener, router(state), Duration::from_secs(a.header_read_timeout_secs), async move {
        let _ = stop_rx.await;
    })
    .await;
    eprintln!("rexl3 level=info event=stopped");
    std::process::exit(0);
}

/// Rank 1 of a tensor-parallel group: the follower runs on a thread of its own (it loads, then replays rank 0's calls until rank 0 shuts
/// down or is lost) and this function waits for it or for a signal. Never returns: the process exits with the follower's verdict.
async fn run_tp_follower(
    a: &Args,
    tp: gb_model::serve::TpOptions,
    mut term: tokio::signal::unix::Signal,
    mut int: tokio::signal::unix::Signal,
) -> ! {
    let o = serve_options(a, None);
    eprintln!("rexl3 event=tp_follower_start rank={} ranks={} peer={}", tp.rank, tp.ranks, tp.peer);
    let stop = gb_model::serve::follower_stop_flag();
    let done = std::thread::Builder::new()
        .name("tp-follower".into())
        .spawn(move || gb_model::serve::start_follower(&o))
        .unwrap_or_else(|e| fail(&format!("cannot start the follower thread: {e}")));
    let grace = Duration::from_secs(a.shutdown_grace_secs);
    let mut joined = tokio::task::spawn_blocking(move || done.join());
    let outcome = tokio::select! {
        r = &mut joined => Some(r),
        _ = term.recv() => None,
        _ = int.recv() => None,
    };
    let r = match outcome {
        Some(r) => r,
        None => {
            eprintln!("rexl3 level=info event=shutdown_signal rank={} grace_ms={}", tp.rank, grace.as_millis());
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            match tokio::time::timeout(grace, &mut joined).await {
                Ok(r) => r,
                Err(_) => {
                    // a boot or a call that does not return: no CUDA teardown racing it, and no claim of a clean stop
                    eprintln!("rexl3 level=fatal event=tp_follower_stop_timeout the follower did not stop within {grace:?}");
                    std::process::abort();
                }
            }
        }
    };
    match r {
        Ok(Ok(Ok(()))) => {
            eprintln!("rexl3 level=info event=stopped rank={}", tp.rank);
            std::process::exit(0)
        }
        // a lost group (rank 0 gone while idle, or a fatal error on this rank) exits 70 like rank 0's `on_fatal` and the exit backstop do;
        // a refused boot or a bad flag stays 1, so a supervisor can tell "restart the group" from "do not restart" (docs/CLUSTER_OPERATIONS.md)
        Ok(Ok(Err(e))) if e.is_fatal() => {
            eprintln!("error: tensor-parallel follower: {e}");
            // `_exit`, like the leader's `on_fatal`: the exit-time CUDA teardown of a context whose communicator was aborted can fault
            // (a status 139 where 70 was logged), and the supervisor must read 70 for a lost group
            gb_model::exit::end_process(70, gb_model::exit::fatal_exit_mode(true))
        }
        Ok(Ok(Err(e))) => fail(&format!("tensor-parallel follower: {e}")),
        _ => {
            eprintln!("rexl3 level=fatal event=tp_follower_panicked");
            std::process::abort()
        }
    }
}

/// Wait for SIGTERM (what `docker stop` sends) or SIGINT, then stop gracefully inside `grace`: stop admitting, answer queued requests
/// 503, let running ones finish until `grace` minus a margin for writing the last responses, end what is left as an error, and have
/// the HTTP server finish its open responses and exit. A watchdog makes sure the process is gone shortly after `grace` whatever hangs.
async fn shutdown_on_signal(
    state: Arc<AppState>,
    engine_thread: JoinHandle<()>,
    grace: Duration,
    stop: tokio::sync::oneshot::Sender<()>,
    mut term: tokio::signal::unix::Signal,
    mut int: tokio::signal::unix::Signal,
) {
    tokio::select! {
        _ = term.recv() => {}
        _ = int.recv() => {}
    }
    let loading = state.handle.metrics.state() == gb_server::runtime::State::Loading;
    if loading {
        eprintln!("rexl3 level=info event=shutdown_signal while=loading");
    }
    let t0 = std::time::Instant::now();
    // the margin: time kept back for the HTTP side to write the last bytes and close
    let margin = (grace / 4).min(Duration::from_millis(1500));
    eprintln!("rexl3 level=info event=shutdown_signal grace_ms={} drain_ms={}", grace.as_millis(), (grace - margin).as_millis());
    let watchdog = grace + Duration::from_secs(3);
    std::thread::spawn(move || {
        std::thread::sleep(watchdog);
        eprintln!("rexl3 level=fatal event=shutdown_watchdog the process did not stop within {watchdog:?}");
        // No CUDA atexit racing an unquiesced owner. A genuine wedge is an explicit failure, never a claimed clean stop.
        std::process::abort();
    });
    // Cancel preparation at worker-owned safe boundaries before CUDA teardown. Dropping an HTTP
    // future is not a join; the vision gate retains credits until the real worker/allocation ends.
    if let Some(v) = &state.vision {
        v.gate.shutdown();
    }
    state.handle.drain(grace - margin);
    if let Some(v) = &state.vision {
        v.gate.wait_idle().await;
    }
    // The watchdog handles actual wedges; never claim a clean stop after a failed join.
    match tokio::task::spawn_blocking(move || engine_thread.join()).await {
        Ok(Ok(())) => {}
        _ => {
            eprintln!("rexl3 level=fatal event=engine_join_failed");
            std::process::abort();
        }
    }
    eprintln!("rexl3 level=info event=drained after_ms={}", t0.elapsed().as_millis());
    let _ = stop.send(()); // the server stops accepting and finishes the responses still open
    tokio::time::sleep(grace.saturating_sub(t0.elapsed())).await;
    eprintln!("rexl3 level=warn event=shutdown_flush_timeout open responses were cut off");
    std::process::exit(0);
}

fn fail(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decisions_timeout_is_explicit_bounded_and_configurable() {
        let base = parse_from(["--model", "/m"].into_iter().map(String::from)).unwrap();
        assert_eq!(runtime_config(&base).decision_timeout, Duration::from_secs(1800));
        for seconds in ["1", "1800", "86400"] {
            let a = parse_from(["--model", "/m", "--decision-timeout", seconds].into_iter().map(String::from)).unwrap();
            assert_eq!(runtime_config(&a).decision_timeout, Duration::from_secs(seconds.parse().unwrap()));
        }
        for bad in ["0", "86401", "-1", "1.5", "abc"] {
            assert!(parse_from(["--model", "/m", "--decision-timeout", bad].into_iter().map(String::from)).is_err());
        }
    }

    #[test]
    fn the_arena_cap_is_set_only_when_nothing_names_one_and_it_is_not_opted_out() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string())
        };
        assert_eq!(arena_cap_wanted(env(&[])), Some("2"));
        assert_eq!(arena_cap_wanted(env(&[("MALLOC_ARENA_MAX", "8")])), None, "the caller's own value is kept");
        assert_eq!(arena_cap_wanted(env(&[("MALLOC_ARENA_MAX", "")])), None, "set, even empty");
        assert_eq!(arena_cap_wanted(env(&[("GLIBC_TUNABLES", "glibc.malloc.arena_max=4")])), None);
        assert_eq!(arena_cap_wanted(env(&[("GLIBC_TUNABLES", "glibc.malloc.tcache_count=0:glibc.malloc.arena_max=4")])), None);
        assert_eq!(arena_cap_wanted(env(&[("GLIBC_TUNABLES", "glibc.malloc.tcache_count=0")])), Some("2"));
        for off in ["0", "off", "false", "no"] {
            let pairs: &'static [(&'static str, &'static str)] = Box::leak(Box::new([("REXL3_MALLOC_REEXEC", off)]));
            assert_eq!(arena_cap_wanted(env(pairs)), None, "REXL3_MALLOC_REEXEC={off}");
        }
        assert_eq!(arena_cap_wanted(env(&[("REXL3_MALLOC_REEXEC", "1")])), Some("2"));
    }

    #[test]
    fn the_rendered_prompt_cap_follows_the_window_and_a_flag_overrides_it_in_either_order() {
        let args = |extra: &[&str]| {
            let mut v = vec!["--model".to_string(), "/m".to_string()];
            v.extend(extra.iter().map(|s| s.to_string()));
            parse_from(v.into_iter())
        };
        let mib = 1usize << 20;
        // the shipped window: 16 bytes a token is 4 MiB, under the 2 GiB reserve
        let a = args(&[]).unwrap();
        assert_eq!((a.max_model_len, a.request_limits.rendered_bytes), (262_144, 4 * mib));
        assert_eq!(a.request_limits.processing_reserve_bytes, 2048 * mib);
        // the window flag moves it, whichever side of the other flags it is on
        assert_eq!(args(&["--max-model-len", "300000"]).unwrap().request_limits.rendered_bytes, 4_800_000);
        assert_eq!(args(&["--max-model-len", "300000", "--port", "9"]).unwrap().request_limits.rendered_bytes, 4_800_000);
        assert_eq!(args(&["--max-model-len", "8192"]).unwrap().request_limits.rendered_bytes, mib, "the floor");
        // a flag keeps what it names, in either order
        for order in [
            ["--max-rendered-prompt-mib", "2", "--max-model-len", "300000"],
            ["--max-model-len", "300000", "--max-rendered-prompt-mib", "2"],
        ] {
            assert_eq!(args(&order).unwrap().request_limits.rendered_bytes, 2 * mib, "{order:?}");
        }
        // a window past what the default reserve prices gets a reserve that holds one prompt, and still boots
        let big = args(&["--max-model-len", "1048576"]).unwrap();
        assert_eq!(big.request_limits.rendered_bytes, 16 * mib);
        assert!(
            big.request_limits.processing_reserve_bytes >= 16 * mib * gb_server::budget::TOKENIZER_BYTES_PER_RENDERED_BYTE,
            "{}",
            big.request_limits.processing_reserve_bytes
        );
        // an explicit 8 MiB cap used to be the default: with the default reserve it now derives the reserve too, so it still boots
        let eight = args(&["--max-rendered-prompt-mib", "8"]).unwrap();
        assert!(eight.request_limits.processing_reserve_bytes >= 8 * mib * gb_server::budget::TOKENIZER_BYTES_PER_RENDERED_BYTE);
        // but a reserve named by hand is held to the price: 8 MiB of rendered prompt is 3.5 GiB at 448 bytes a byte
        let e = args(&["--max-rendered-prompt-mib", "8", "--processing-reserve-mib", "2048"]).err().unwrap();
        assert!(e.contains("processing reserve must be at least"), "{e}");
        assert!(args(&["--max-rendered-prompt-mib", "8", "--processing-reserve-mib", "3600"]).is_ok());
    }

    fn opts(args: &[&str]) -> gb_model::serve::ServeOptions {
        let a = parse_from(args.iter().map(|s| s.to_string())).unwrap();
        serve_options(&a, None)
    }

    #[test]
    fn the_kv_headroom_is_off_by_default_and_reaches_the_family_options() {
        assert_eq!(opts(&["--model", "/m"]).kv_headroom_tokens, 0);
        assert_eq!(opts(&["--model", "/m", "--kv-headroom-tokens", "131072"]).kv_headroom_tokens, 131_072);
        assert!(parse_from(["--model", "/m", "--kv-headroom-tokens", "lots"].into_iter().map(str::to_string)).is_err());
    }

    #[test]
    fn lifecycle_defaults_are_active_and_cli_budgets_cannot_overflow_or_disable_them() {
        let a = parse_from(["--model".into(), "/m".into()].into_iter()).unwrap();
        let cfg = runtime_config(&a);
        assert_eq!(cfg.request_timeout, Duration::from_secs(1800));
        assert_eq!(cfg.write_stall, Duration::from_secs(30));
        assert_eq!(cfg.heartbeat_stale, Duration::from_secs(60));
        assert_eq!(a.header_read_timeout_secs, 30);
        for flag in [
            "--request-timeout",
            "--write-stall",
            "--heartbeat-stale-secs",
            "--loading-budget",
            "--prefill-budget",
            "--header-read-timeout",
        ] {
            for value in ["0", "18446744073709551615", "not-a-number"] {
                assert!(parse_from(["--model", "/m", flag, value].into_iter().map(str::to_string)).is_err());
            }
        }
        for value in ["1", "86400"] {
            let a = parse_from(["--model", "/m", "--header-read-timeout", value].into_iter().map(str::to_string)).unwrap();
            assert_eq!(a.header_read_timeout_secs, value.parse::<u64>().unwrap());
        }
        assert!(parse_from(["--model", "/m", "--header-read-timeout", "86401"].into_iter().map(str::to_string)).is_err());
    }

    fn tp(flags: TpFlags, env: &[(&str, &str)]) -> Result<Option<gb_model::serve::TpOptions>, String> {
        let env: Vec<(String, String)> = env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        resolve_tp(&flags, move |k| env.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone()))
    }

    #[test]
    fn tensor_parallel_flags_and_environment_resolve_and_refuse_by_name() {
        // unset: nothing, and a plain boot never looks at it
        assert_eq!(tp(TpFlags::default(), &[]).unwrap(), None);
        // flags
        let t = tp(TpFlags { ranks: Some(2), rank: Some(1), peer: Some("192.0.2.1:29600".into()), ..Default::default() }, &[])
            .unwrap()
            .unwrap();
        assert_eq!((t.ranks, t.rank, t.peer.as_str(), t.is_group()), (2, 1, "192.0.2.1:29600", true));
        // the environment, with a flag over it
        let t = tp(
            TpFlags { rank: Some(0), ..Default::default() },
            &[
                ("REXL3_TP_RANKS", "2"),
                ("REXL3_TP_RANK", "1"),
                ("REXL3_TP_PEER", "192.0.2.1:7000"),
                ("REXL3_TP_PEER_TIMEOUT_S", "30"),
            ],
        )
        .unwrap()
        .unwrap();
        assert_eq!((t.ranks, t.rank, t.peer.as_str(), t.peer_timeout), (2, 0, "192.0.2.1:7000", Duration::from_secs(30)));
        // --tp-ranks 1 is no group
        let t = tp(TpFlags { ranks: Some(1), ..Default::default() }, &[]).unwrap().unwrap();
        assert!(!t.is_group());
        assert!(serve_options(&parse_from(["--model", "/m", "--tp-ranks", "1"].into_iter().map(String::from)).unwrap(), None)
            .tp
            .is_none());
        // refusals
        for (flags, why) in [
            (TpFlags { ranks: Some(0), rank: Some(0), peer: Some("a:1".into()), ..Default::default() }, "--tp-ranks"),
            (TpFlags { ranks: Some(3), rank: Some(3), peer: Some("a:1".into()), ..Default::default() }, "out of range"),
            (TpFlags { ranks: Some(2), rank: Some(2), peer: Some("a:1".into()), ..Default::default() }, "out of range"),
            (TpFlags { ranks: Some(2), rank: Some(1), ..Default::default() }, "HOST:PORT"),
            (TpFlags { ranks: Some(2), rank: Some(1), peer: Some("nohost".into()), ..Default::default() }, "HOST:PORT"),
            (TpFlags { ranks: Some(2), rank: Some(1), peer: Some("a:notaport".into()), ..Default::default() }, "port"),
        ] {
            let e = tp(flags, &[]).unwrap_err();
            assert!(e.contains(why), "{e}");
        }
        assert!(tp(TpFlags::default(), &[("REXL3_TP_RANK", "x")]).unwrap_err().contains("REXL3_TP_RANK"));
        // a group of three or eight resolves too, any rank of it
        for (ranks, rank) in [(3, 0), (3, 2), (8, 7)] {
            let t = tp(TpFlags { ranks: Some(ranks), rank: Some(rank), peer: Some("a:1".into()), ..Default::default() }, &[])
                .unwrap()
                .unwrap();
            assert_eq!((t.ranks, t.rank, t.is_group()), (ranks, rank, true));
            assert!(t.validate().is_ok());
        }
        // the whole command line
        let a = parse_from(
            ["--model", "/m", "--tp-ranks", "2", "--tp-rank", "0", "--tp-peer", "h:9", "--tp-connect-timeout", "5"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        let o = serve_options(&a, None);
        let t = o.tp.expect("a group");
        assert_eq!((t.ranks, t.rank, t.connect_timeout), (2, 0, Duration::from_secs(5)));
        assert!(parse_from(
            ["--model", "/m", "--tp-ranks", "2", "--tp-rank", "1", "--tp-peer", "h:9", "--backend", "mock"]
                .into_iter()
                .map(String::from)
        )
        .is_err());
    }

    #[test]
    fn only_a_tensor_parallel_group_ends_its_fatal_paths_with_underscore_exit_and_leaks_its_engine() {
        // a single process (no flags, or --tp-ranks 1): `exit` and the engine dropped, as before
        for args in [vec!["--model", "/m"], vec!["--model", "/m", "--tp-ranks", "1"]] {
            let a = parse_from(args.into_iter().map(String::from)).unwrap();
            let cfg = runtime_config(&a);
            assert_eq!((cfg.fatal_exit, cfg.leak_engine_on_fatal), (gb_model::exit::FatalExit::Exit, false));
        }
        // either rank of a group: `_exit` and the engine leaked
        for rank in ["0", "1"] {
            let a = parse_from(
                ["--model", "/m", "--tp-ranks", "2", "--tp-rank", rank, "--tp-peer", "h:9"].into_iter().map(String::from),
            )
            .unwrap();
            let cfg = runtime_config(&a);
            assert_eq!((cfg.fatal_exit, cfg.leak_engine_on_fatal), (gb_model::exit::FatalExit::Immediate, true), "rank {rank}");
        }
    }

    #[test]
    fn the_default_slot_count_may_shrink_and_an_explicit_one_is_pinned() {
        // no `--slots`: 8, and a family that cannot hold them boots with the most that fit
        let o = opts(&["--model", "/m"]);
        assert_eq!((o.slots, o.slots_auto), (8, true));
        // `--slots N` is a request: refused by name when it does not fit, never changed, whatever N is (the default value spelled out included)
        for n in ["2", "8", "32"] {
            let o = opts(&["--model", "/m", "--slots", n]);
            assert_eq!((o.slots, o.slots_auto), (n.parse().unwrap(), false), "--slots {n}");
        }
        // the flag may come anywhere, before or after other settings
        let o = opts(&["--slots", "3", "--model", "/m", "--window-rows", "32"]);
        assert_eq!((o.slots, o.slots_auto, o.window_rows), (3, false, 32));
    }

    #[test]
    fn deepseek_v41_bare_model_flag_applies_q4_kv_and_aliases() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/deepseek_v41");
        std::env::remove_var(gb_cuda::dsv4_kvq4::KV_FMT_ENV);
        let mut a = parse_from(["--model", dir.to_str().unwrap()].into_iter().map(String::from)).unwrap();
        assert!(a.vision);
        apply_family_defaults(&mut a);
        assert!(a.vision, "built vision_deepseek tower remains enabled by default");
        assert_eq!(a.slots, 1, "deepseek_v41 default slots_auto is 1");
        assert_eq!(
            gb_cuda::dsv4_kvq4::configured_fmt(gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap(),
            gb_cuda::dsv4_kvq4::KvFmt::Q4,
            "deepseek_v41 defaults to q4 KV cache unless overridden"
        );
        for alias in [
            "dsv41",
            "DeepSeek-V4.1-Flash-TF",
            "DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw",
            "vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw",
        ] {
            assert!(a.name == alias || a.model_aliases.iter().any(|s| s == alias), "missing alias {alias}");
        }
        let mut a1m =
            parse_from(["--model", dir.to_str().unwrap(), "--max-model-len", "1048576"].into_iter().map(String::from)).unwrap();
        apply_family_defaults(&mut a1m);
        assert_eq!(a1m.request_timeout_secs, 14_400);
        assert_eq!(a1m.prefill_budget_secs, 14_400);
        assert_eq!(a1m.decision_timeout_secs, 14_400);
        assert_eq!(a1m.queue_timeout_secs, 14_400);
        std::env::remove_var(gb_cuda::dsv4_kvq4::KV_FMT_ENV);
    }
}
