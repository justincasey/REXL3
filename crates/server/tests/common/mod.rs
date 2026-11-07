//! Shared helpers of the HTTP tests: a model folder assembled from committed fixtures, an in-process server on the CPU mock backend, and
//! a small raw HTTP client (blocking, one connection per request, so a test can hold a connection open or drop it).
#![allow(dead_code)]

use gb_engine::mock::MockOptions;
use gb_engine::{Drafter, Engine, EngineConfig, NoDrafts, ServeEngine};
use gb_sched::{Admission, CostModel, ModelMemory};
use gb_server::api::GenDefaults;
use gb_server::chat_format::{ChatFormat, FormatRegistry};
use gb_server::http::{router, AppState};
use gb_server::runtime::{spawn_with, EngineHandle, FatalInfo, RuntimeConfig};
use gb_server::tokenizer::Tok;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// A model folder (`tokenizer.json`, `generation_config.json`, `chat_template.jinja`, `config.json`) built from the committed fixtures: a byte-level
/// tokenizer with the `<|im_start|>` chat-template tokens (`tools/gen_mock_tokenizer.py`) and the Nemotron 3.5 Super VL pack's chat template and
/// config (`testdata/nemotron_omni`, which select the `nemotron_omni` chat format). The tests run against it in CI; no model is needed.
pub fn fixture_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| assemble("rexl3-mock-model", None)).clone()
}

/// The fixture with an end-of-turn id the mock can never produce (past its 4,096-token vocabulary): a request runs to its `max_tokens`,
/// so a test that needs a request to stay busy does not depend on what the mock happens to emit.
pub fn unstoppable_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| assemble("rexl3-mock-model-unstoppable", Some(100_000))).clone()
}

/// The same mock tokenizer with the Qwen3.8 chat template and config head (`tests/fixtures`), which select the `qwen` chat format: the
/// folder the format-agnostic tests used before `fixture_dir` moved to the Nemotron template, kept for the tests of what only the Qwen
/// format does (decisions scoring, its template's own refusals).
pub fn qwen_fixture_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let dir = std::env::temp_dir().join(format!("rexl3-mock-model-qwen-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (from, to) in [
            ("mock_model/tokenizer.json", "tokenizer.json"),
            ("mock_model/generation_config.json", "generation_config.json"),
            ("qwen38_chat_template.jinja", "chat_template.jinja"),
            ("qwen38_config_head.json", "config.json"),
        ] {
            std::fs::copy(fx.join(from), dir.join(to)).unwrap();
        }
        dir
    })
    .clone()
}

fn assemble(name: &str, eos: Option<u32>) -> PathBuf {
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let nemo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/nemotron_omni");
    let dir = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (from, to) in [
        (fx.join("mock_model/tokenizer.json"), "tokenizer.json"),
        (fx.join("mock_model/generation_config.json"), "generation_config.json"),
        (nemo.join("chat_template.jinja"), "chat_template.jinja"),
        (nemo.join("config.json"), "config.json"),
    ] {
        std::fs::copy(from, dir.join(to)).unwrap();
    }
    if let Some(id) = eos {
        std::fs::write(dir.join("generation_config.json"), format!("{{\"eos_token_id\": [{id}], \"temperature\": 1.0}}"))
            .unwrap();
    }
    dir
}

/// The folder the e2e tests use: the real pack when `REXL3_NEMO_OMNI_DIR` names one, else the fixture.
pub fn model_dir() -> PathBuf {
    std::env::var("REXL3_NEMO_OMNI_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.join("tokenizer.json").exists() && p.join("chat_template.jinja").exists() && p.join("config.json").exists())
        .unwrap_or_else(fixture_dir)
}

/// What an in-process server is built with.
#[derive(Clone)]
pub struct Opts {
    pub slots: usize,
    pub request_limits: gb_server::budget::Limits,
    pub model_aliases: Vec<String>,
    pub vision_slot: Option<gb_model::serve::VisionSlot>,
    pub max_model_len: usize,
    /// Tokens of KV the engine can hold across all sequences (admission capacity).
    pub kv_tokens: usize,
    /// Declare `kv_tokens` as the pool (`Admission::with_kv_tokens`), as a memory plan does for the real engines. Off by default: most tests
    /// state a byte capacity only, and `lifecycle.rs` relies on that to reach the engine's own `NeverFits` refusal.
    pub pool: bool,
    pub queue_depth: usize,
    pub queue_timeout: Duration,
    pub request_timeout: Duration,
    pub write_stall: Duration,
    pub heartbeat_stale: Duration,
    pub prefill_budget: Duration,
    pub decision_timeout: Duration,
    pub prefix_cache_enabled: bool,
    pub keepalive: Duration,
    pub default_max_tokens: usize,
    pub unready_after: u32,
    pub exit_after: u32,
    pub mock: MockOptions,
    pub dir: Option<PathBuf>,
    pub vision: Option<gb_server::http::VisionHandle>,
    pub format: Option<Arc<dyn ChatFormat>>,
    /// `--default-thinking`.
    pub default_thinking: gb_server::reasoning_controls::DefaultThinking,
    /// `--reasoning-unverified`.
    pub reasoning_unverified: bool,
    /// Build the engine with this instead of the CPU mock (a scripted engine that answers chosen ids, for instance).
    pub engine: Option<Arc<dyn Fn() -> Box<dyn ServeEngine> + Send + Sync>>,
    /// The mock engine's gather window (`Engine::with_gather_window`); zero by default.
    pub gather: Duration,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            slots: 8,
            request_limits: gb_server::budget::Limits::default(),
            model_aliases: Vec::new(),
            vision_slot: None,
            max_model_len: 4096,
            kv_tokens: 1 << 20,
            pool: false,
            queue_depth: 64,
            queue_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(1800),
            write_stall: Duration::from_secs(30),
            heartbeat_stale: Duration::from_secs(60),
            prefill_budget: Duration::from_secs(1800),
            decision_timeout: Duration::from_secs(1800),
            prefix_cache_enabled: true,
            keepalive: Duration::from_secs(15),
            default_max_tokens: 0,
            unready_after: 3,
            exit_after: 10,
            // a vocabulary much larger than the tokenizer's: a mock reply almost never hits an end-of-turn id by accident
            mock: MockOptions { vocab: 4096, ..Default::default() },
            dir: None,
            vision: None,
            format: None,
            default_thinking: Default::default(),
            reasoning_unverified: false,
            engine: None,
            gather: Duration::ZERO,
        }
    }
}

pub struct Server {
    pub port: u16,
    pub state: Arc<AppState>,
    /// Failures `on_fatal` was called with.
    pub fatal: Arc<Mutex<Vec<FatalInfo>>>,
    pub engine_thread: Option<std::thread::JoinHandle<()>>,
    _rt: tokio::runtime::Runtime,
}

impl Server {
    pub fn handle(&self) -> &EngineHandle {
        &self.state.handle
    }
    /// Sequences decoding + prefilling + requests queued: zero when the engine has given everything back.
    pub fn in_flight(&self) -> u64 {
        use std::sync::atomic::Ordering::Relaxed;
        let m = &self.state.handle.metrics;
        m.active.load(Relaxed) + m.prefilling.load(Relaxed) + m.queued.load(Relaxed)
    }
    pub fn metric(&self, f: impl Fn(&gb_server::runtime::Metrics) -> &std::sync::atomic::AtomicU64) -> u64 {
        f(&self.state.handle.metrics).load(std::sync::atomic::Ordering::Relaxed)
    }
    /// Wait until `cond` holds, up to `within`.
    pub fn wait(&self, within: Duration, what: &str, cond: impl Fn(&Server) -> bool) {
        let t0 = Instant::now();
        while !cond(self) {
            assert!(t0.elapsed() < within, "timed out after {within:?} waiting for: {what}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

pub fn start(o: Opts) -> Server {
    let d = o.dir.clone().unwrap_or_else(unstoppable_dir);
    let tok = Arc::new(Tok::from_dir(&d).unwrap());
    // the folder's `config.json` selects the chat format, as it does in the server
    let format: Arc<dyn ChatFormat> =
        o.format.clone().unwrap_or_else(|| FormatRegistry::builtin().open(&d, None).unwrap().into());
    let vocab = tok.vocab;
    let fatal: Arc<Mutex<Vec<FatalInfo>>> = Default::default();
    let hook = fatal.clone();
    let cfg = RuntimeConfig {
        queue_depth: o.queue_depth,
        queue_timeout: o.queue_timeout,
        request_timeout: o.request_timeout,
        write_stall: o.write_stall,
        heartbeat_stale: o.heartbeat_stale,
        prefill_budget: o.prefill_budget,
        decision_timeout: o.decision_timeout,
        unready_after: o.unready_after,
        exit_after: o.exit_after,
        on_fatal: Arc::new(move |info| hook.lock().unwrap().push(info.clone())),
        ..RuntimeConfig::default()
    };
    let cache_enabled = o.prefix_cache_enabled;
    let (slots, mock, kv_tokens, pool, gather) = (o.slots, o.mock.clone(), o.kv_tokens as u64, o.pool, o.gather);
    let custom = o.engine.clone();
    let (handle, th) = spawn_with(
        Box::new(move || {
            if let Some(make) = custom {
                return make();
            }
            let mm =
                ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1024, state_bytes_per_slot: 1 << 20, activation_reserve: 0 };
            let prefix = (mock.prefix_cache > 0 && cache_enabled).then_some(gb_engine::PrefixConfig {
                policy: gb_engine::CutPolicy { min_cut: 64, tail: 8, ghost_min_gain: 256, min_gap: 0 },
                ghost_blocks: 1 << 12,
                ..Default::default()
            });
            let admission = Admission::new(kv_tokens * 1024 + slots as u64 * (1 << 20), 0, mm, slots).unwrap();
            let engine = Engine::new(
                mock.build(vocab, slots),
                if pool { admission.with_kv_tokens(kv_tokens) } else { admission },
                EngineConfig {
                    max_rows: 64,
                    max_drafts: 7,
                    cost: CostModel::from_table(vec![(1, 1.0), (64, 2.0)]),
                    drafting: false,
                },
                Box::new(|| Box::new(NoDrafts) as Box<dyn Drafter>),
            )
            .with_gather_window(gather);
            Box::new(match prefix {
                Some(p) => engine.with_prefix_cache(p),
                None => engine,
            }) as Box<dyn ServeEngine>
        }),
        cfg,
    );
    handle.metrics.observability.slots.store(o.slots as u64, std::sync::atomic::Ordering::Relaxed);
    let reasoning_caps =
        Arc::new(gb_server::reasoning_caps::ReasoningSupport::for_format(&*format, &tok, o.reasoning_unverified));
    let state = Arc::new(AppState {
        handle,
        tok,
        format,
        defaults: GenDefaults::from_dir(&d),
        model_name: "mock".into(),
        model_aliases: o.model_aliases.clone(),
        request_budget: gb_server::budget::RequestBudget::new(o.request_limits.clone()).unwrap(),
        max_model_len: o.max_model_len,
        sse_keepalive: o.keepalive,
        default_max_tokens: o.default_max_tokens,
        default_thinking: o.default_thinking,
        reasoning_caps,
        vision: o.vision.or_else(|| {
            o.vision_slot.clone().map(|slot| gb_server::http::VisionHandle {
                slot,
                limits: Default::default(),
                pp: Arc::new(gb_vision::Preprocessing::qwen38()),
                gate: gb_server::preparation::PreparationGate::new(Default::default()).unwrap(),
                ingress: gb_server::vision_ingress::HttpsIngress::new(Default::default()).unwrap(),
            })
        }),
    });
    let rt = tokio::runtime::Runtime::new().unwrap();
    let listener = rt.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
    let port = listener.local_addr().unwrap().port();
    let app = router(state.clone());
    rt.spawn(async move { axum::serve(listener, app).await.unwrap() });
    let s = Server { port, state, fatal, engine_thread: Some(th), _rt: rt };
    s.wait(Duration::from_secs(10), "the engine to be ready", |s| s.state.handle.metrics.ready());
    s
}

/// One parsed HTTP response.
#[derive(Debug)]
pub struct Reply {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Reply {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
    pub fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|e| panic!("not JSON ({e}): {}", self.body))
    }
}

fn parse_reply(raw: &[u8]) -> Reply {
    let text = String::from_utf8_lossy(raw).to_string();
    let (head, rest) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    let mut lines = head.lines();
    let status: u16 = lines.next().unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
    let headers: Vec<(String, String)> =
        lines.filter_map(|l| l.split_once(':')).map(|(k, v)| (k.trim().to_string(), v.trim().to_string())).collect();
    let chunked =
        headers.iter().any(|(k, v)| k.eq_ignore_ascii_case("transfer-encoding") && v.to_ascii_lowercase().contains("chunked"));
    let body = if chunked { dechunk(rest) } else { rest.to_string() };
    Reply { status, headers, body }
}

pub fn dechunk(mut s: &str) -> String {
    let mut out = String::new();
    while let Some((len, rest)) = s.split_once("\r\n") {
        let n = usize::from_str_radix(len.trim(), 16).unwrap_or(0);
        if n == 0 || rest.len() < n {
            // a partial last chunk (the connection was cut): keep what arrived
            out.push_str(&rest[..rest.len().min(if n == 0 { 0 } else { n })]);
            break;
        }
        out.push_str(&rest[..n]);
        s = rest.get(n + 2..).unwrap_or("");
    }
    out
}

fn request_text(method: &str, path: &str, body: Option<&Value>) -> String {
    let b = body.map(|b| b.to_string()).unwrap_or_default();
    format!(
        "{method} {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}",
        b.len()
    )
}

/// One request, `Connection: close`; returns when the server closes the connection (or after `timeout`).
pub fn http_timeout(port: u16, method: &str, path: &str, body: Option<&Value>, timeout: Duration) -> Reply {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(timeout)).unwrap();
    s.write_all(request_text(method, path, body).as_bytes()).unwrap();
    let mut raw = Vec::new();
    let _ = s.read_to_end(&mut raw);
    parse_reply(&raw)
}

/// One request with exactly the bytes and headers given (no `Content-Type` unless `headers` names one): for what a JSON client cannot send,
/// a body that is not JSON, no content type, a `Content-Length` the body does not match.
pub fn raw(port: u16, method: &str, path: &str, headers: &[(&str, &str)], body: &[u8], timeout: Duration) -> Reply {
    let mut head = format!("{method} {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n");
    for (k, v) in headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    if !headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-length")) {
        head.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    head.push_str("\r\n");
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(timeout)).unwrap();
    s.write_all(head.as_bytes()).unwrap();
    let _ = s.write_all(body); // a server that has already answered may stop reading
    let mut out = Vec::new();
    let _ = s.read_to_end(&mut out);
    parse_reply(&out)
}

pub fn http(port: u16, method: &str, path: &str, body: Option<&Value>) -> (u16, String) {
    let r = http_timeout(port, method, path, body, Duration::from_secs(120));
    (r.status, r.body)
}

/// A connection the test reads from as bytes arrive, and may drop to play a client that goes away.
pub struct Conn {
    pub sock: TcpStream,
    pub raw: Vec<u8>,
}

impl Conn {
    pub fn post(port: u16, path: &str, body: &Value) -> Conn {
        let mut sock = TcpStream::connect(("127.0.0.1", port)).unwrap();
        sock.write_all(request_text("POST", path, Some(body)).as_bytes()).unwrap();
        Conn { sock, raw: Vec::new() }
    }
    /// Read until `pred(received text)` or `timeout`; returns whether it held.
    pub fn read_until(&mut self, timeout: Duration, pred: impl Fn(&str) -> bool) -> bool {
        let t0 = Instant::now();
        let mut buf = [0u8; 4096];
        loop {
            if pred(&String::from_utf8_lossy(&self.raw)) {
                return true;
            }
            let left = timeout.saturating_sub(t0.elapsed());
            if left.is_zero() {
                return false;
            }
            self.sock.set_read_timeout(Some(left.min(Duration::from_millis(100)))).unwrap();
            match self.sock.read(&mut buf) {
                Ok(0) => return pred(&String::from_utf8_lossy(&self.raw)),
                Ok(n) => self.raw.extend_from_slice(&buf[..n]),
                Err(_) => {}
            }
        }
    }
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.raw).to_string()
    }
    /// Read until the response body is complete (the chunked encoding's last chunk, or the connection closed) or `timeout`; returns
    /// whether it was. What a stream that ends with an ERROR needs: it has no `[DONE]` to wait for.
    pub fn read_to_end_of_body(&mut self, timeout: Duration) -> bool {
        self.read_until(timeout, |t| t.contains("\r\n0\r\n\r\n"))
    }
    /// Read to the end of the response.
    pub fn finish(mut self, timeout: Duration) -> Reply {
        let _ = self.read_until(timeout, |_| false);
        parse_reply(&self.raw)
    }
}

/// The JSON objects of an SSE body, in order (`[DONE]` excluded).
pub fn sse(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap_or_else(|e| panic!("bad SSE data ({e}): {d}")))
        .collect()
}

pub fn chat_body(extra: Value) -> Value {
    let mut b =
        json!({"model": "mock", "messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 12, "temperature": 0.0});
    for (k, v) in extra.as_object().unwrap() {
        b[k] = v.clone();
    }
    b
}

/// A plain completion of the token ids `prompt` (the first id is what a mock `poison` option keys on).
pub fn completion_body(prompt: &[u32], max_tokens: usize, stream: bool) -> Value {
    json!({"model": "mock", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "stream": stream})
}
